#!/usr/bin/env python3
"""
Two checks on the confidence of the production model (project doc claude/macro-scores-and-class-balance.md;
experiments README §17).

1. Is one probability threshold enough? (the "0.30 with five alternatives vs 0.30 with a flat tail" question)
   a. Reliability of the CV probability of the top-1 (--predictions, prob column): accuracy per probability band, and
      within each band per tercile of the raw score margin; does the margin add to the probability (study-grouped CV
      AUC of a logistic regression, p vs p + margin)?
   b. Shape of the whole distribution: the prototype scores are recomputed on the same folds and turned into
      probabilities with the production temperature (--calibration). Does the runner-up probability p2, the entropy
      or the mass outside the top 5 add to p1 (same AUC test)? For top-1s with p1 in [0.25, 0.35): accuracy by
      tercile of the tail mass (alternatives concentrated in the top 5 vs spread over many labels).
   -> reliability.tsv, margin_within_band.tsv, shape_auc.tsv, shape_band.tsv
2. Zero-shot fallback when the back-off abstains. How many samples whose gold label never occurs in training are
   abstained on (abstention as a detector), and how good is open-vocabulary zero-shot retrieval on them: the
   retrieval_open predictions of the CV run (term text 'label; synonyms', all terms) and, with --plain_label_vectors
   (term_text/term_variants.h5 of term_text_variants.py), the same retrieval with the plain label as term text
   (ENVO / Uberon terms only). Exact, gold-or-broader, and gold in the top 5.  -> zero_shot_fallback.tsv

Example (~1 min):
python experiments/confidence_checks.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --calibration ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/calibration.json \
  --plain_label_vectors ~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/confidence_checks
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import normalize

from _setup import DEFAULTS, SLOTS, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds
import methods as M  # noqa: E402  (importable once _setup has added scripts/ontology_mapping to the path)
from common import read_tsv, term_ancestors  # noqa: E402

BANDS = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]


def logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def grouped_auc(features, hit, groups, max_iter=100):
    """Study-grouped 5-fold CV of a logistic regression hit ~ features -> (AUC, log loss) of the held-out predictions.
    Folds come from common.study_folds (md5 of the study code), not sklearn's GroupKFold: GroupKFold breaks ties
    between equal-size studies with an unstable sort, so its folds (and these AUCs, in the 3rd decimal) differed
    between the Mac and the cloud."""
    X = np.column_stack(features)
    folds = list(study_folds(groups, 5, 0))
    p = cross_val_predict(LogisticRegression(max_iter=max_iter), X, hit, cv=folds, method="predict_proba")[:, 1]
    return round(roc_auc_score(hit, p), 4), round(log_loss(hit, p), 4)


def terciles(values):
    """'low' / 'mid' / 'high' by rank (ties broken by order), as three equal-size groups."""
    return pd.qcut(pd.Series(values).rank(method="first"), 3, labels=["low", "mid", "high"]).to_numpy()


# ----------------------------------------------------------------------------- 1a. reliability and margin
def reliability(pred):
    rel, within, auc = [], [], []
    for slot in SLOTS:
        d = pred[pred["slot"] == slot].copy()
        d["hit"], d["p"], d["m"] = (d["pred"] == d["gold"]).astype(int), d["prob"].astype(float), d["confidence"].astype(float)
        d["band"] = pd.cut(d["p"], BANDS)
        for band, b in d.groupby("band", observed=True):
            rel.append({"slot": slot, "band": str(band), "n": len(b), "mean_p": round(b["p"].mean(), 3),
                        "accuracy": round(b["hit"].mean(), 3)})
            for t, bt in zip(["low", "mid", "high"], [b[terciles(b["m"].to_numpy()) == k] for k in ["low", "mid", "high"]]):
                within.append({"slot": slot, "band": str(band), "margin_tercile": t, "n": len(bt),
                               "accuracy": round(bt["hit"].mean(), 3)})
        groups, hit = d["study_code"].to_numpy(), d["hit"].to_numpy()
        for name, feats in [("p", [logit(d["p"], 1e-4)]), ("p + margin", [logit(d["p"], 1e-4), d["m"].to_numpy()])]:
            a, ll = grouped_auc(feats, hit, groups)
            auc.append({"slot": slot, "features": name, "auc": a, "log_loss": ll})
        auc.append({"slot": slot, "features": "raw p (no model)", "auc": round(roc_auc_score(hit, d["p"]), 4), "log_loss": None})
    return pd.DataFrame(rel), pd.DataFrame(within), pd.DataFrame(auc)


# ----------------------------------------------------------------------------- 1b. distribution shape
def prototype_distributions(samples, x, term_matrix, term_ids, temperature, fold_seed):
    """One row per labelled test sample and slot: hit, p1, p2, top-5 mass, entropy of softmax(prototype score / T)."""
    rows = []
    for train, test in study_folds(samples["study_code"], 5, fold_seed):
        for slot in SLOTS:
            a, b = samples[slot].to_numpy()[train] != "", samples[slot].to_numpy()[test] != ""
            y, gold = samples[slot].to_numpy()[train][a], samples[slot].to_numpy()[test][b]
            closed = np.isin(term_ids, y)
            model = M.prototype_model(x[train][a], y, term_matrix[closed], term_ids[closed], 0.5, 0.1)
            z = M.prototype_scores(x[test][b], model) / temperature[slot]
            z -= z.max(axis=1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(axis=1, keepdims=True)
            ordered = -np.sort(-P, axis=1)
            top = term_ids[closed][P.argmax(axis=1)]
            entropy = -(P * np.log(P + 1e-12)).sum(axis=1)
            rows.append(pd.DataFrame({"slot": slot, "study": samples["study_code"].to_numpy()[test][b],
                                      "hit": (top == gold).astype(int), "p1": ordered[:, 0], "p2": ordered[:, 1],
                                      "top5": ordered[:, :5].sum(axis=1), "entropy": entropy}))
    return pd.concat(rows, ignore_index=True)


def shape_tests(dist):
    auc, band = [], []
    for slot in SLOTS:
        d = dist[dist["slot"] == slot]
        hit, groups, lp1 = d["hit"].to_numpy(), d["study"].to_numpy(), logit(d["p1"], 1e-6)
        for name, feats in [("p1", [lp1]), ("p1 + p2", [lp1, logit(d["p2"], 1e-6)]),
                            ("p1 + entropy", [lp1, d["entropy"].to_numpy()]),
                            ("p1 + tail mass", [lp1, logit(1 - d["top5"], 1e-6)])]:
            a, ll = grouped_auc(feats, hit, groups, max_iter=1000)
            auc.append({"slot": slot, "features": name, "auc": a, "log_loss": ll})
        b = d[(d["p1"] >= 0.25) & (d["p1"] < 0.35)]
        shape = pd.Series(terciles((1 - b["top5"]).to_numpy())).map(
            {"low": "mass in top 5", "mid": "middle", "high": "flat tail"}).to_numpy()
        for name in ["mass in top 5", "middle", "flat tail"]:
            s = b[shape == name]
            band.append({"slot": slot, "p1_band": "[0.25, 0.35)", "shape": name, "n": len(s),
                         "accuracy": round(s["hit"].mean(), 3), "mean_p1": round(s["p1"].mean(), 3),
                         "mean_top5": round(s["top5"].mean(), 3), "mean_entropy": round(s["entropy"].mean(), 3)})
    return pd.DataFrame(auc), pd.DataFrame(band)


# ----------------------------------------------------------------------------- 2. zero-shot fallback
def plain_label_retrieval(h5_path, terms, x_by_id, sample_ids, block=2000):
    """{sample_id: [top-5 term ids]} of cosine retrieval among all terms whose plain label is in the .h5
    (term_text_variants.py stores every variant text there; the 'label' variant is the bare label)."""
    import h5py
    with h5py.File(path(h5_path), "r") as handle:
        row_of = {(t.decode() if isinstance(t, bytes) else t): i for i, t in enumerate(handle["texts"][:])}
        keep = terms[terms["label"].isin(row_of)].reset_index(drop=True)
        rows = np.array([row_of[l] for l in keep["label"]])
        unique, inverse = np.unique(rows, return_inverse=True)  # h5py needs strictly increasing indices; terms can share a label
        vectors = handle["embeddings"][unique][inverse].astype(np.float32)
    tv = normalize(vectors)
    term_matrix, ids = np.hstack([tv, tv]) / np.sqrt(2), keep["term_id"].to_numpy()  # same space as [kw, sb] / sqrt(2)
    X = np.stack([x_by_id[s] for s in sample_ids])
    top5 = {}
    for start in range(0, len(X), block):
        scores = X[start:start + block] @ term_matrix.T
        best = np.argsort(-scores, axis=1)[:, :5]
        for s, r in zip(sample_ids[start:start + block], best):
            top5[s] = list(ids[r])
    return top5, len(ids)


def zero_shot_fallback(pred, anc, plain=None):
    proto = pred[pred["method"] == "prototype"][["sample_id", "slot", "gold", "backoff", "gold_seen_in_train"]]
    zs = pred[pred["method"] == "retrieval_open"][["sample_id", "slot", "top5"]]
    d = proto.merge(zs, on=["sample_id", "slot"])
    d["unseen"], d["abstain"] = d["gold_seen_in_train"] != "True", d["backoff"] == ""
    sources = {"retrieval_open (label; synonyms, all terms)": [t.split("||") for t in d["top5"]]}
    if plain is not None:
        sources["plain label (ENVO / Uberon terms)"] = [plain[s] for s in d["sample_id"]]
    rows = []
    for slot in SLOTS:
        m = (d["slot"] == slot).to_numpy()
        x = d[m]
        rows_slot = {"slot": slot, "samples": int(m.sum()), "gold_unseen_share": round(x["unseen"].mean(), 3),
                     "abstain_share": round(x["abstain"].mean(), 3),
                     "unseen_that_abstain": round(x.loc[x["unseen"], "abstain"].mean(), 3),
                     "abstained_with_unseen_gold": round(x.loc[x["abstain"], "unseen"].mean(), 3)}
        for name, top5 in sources.items():
            t5 = [t for t, keep in zip(top5, m) if keep]
            exact = np.array([t[0] == g for t, g in zip(t5, x["gold"])])
            broader = np.array([t[0] == g or t[0] in anc.get(g, ()) for t, g in zip(t5, x["gold"])])
            in5 = np.array([g in t for t, g in zip(t5, x["gold"])])
            u, a = x["unseen"].to_numpy(), x["abstain"].to_numpy()
            rows.append({**rows_slot, "zero_shot": name,
                         "unseen_exact": round(exact[u].mean(), 3), "unseen_gold_or_broader": round(broader[u].mean(), 3),
                         "unseen_in_top5": round(in5[u].mean(), 3), "abstained_exact": round(exact[a].mean(), 3),
                         "abstained_seen_exact": round(exact[a & ~u].mean(), 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "keywords", "sub_biomes", "term_vectors"]:
        ap.add_argument(f"--{name}", default=DEFAULTS[name])
    ap.add_argument("--samples", default="~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz")
    ap.add_argument("--predictions", default="~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz")
    ap.add_argument("--calibration", default="~/MicrobeAtlasProject/ontology_mapping/cv_backoff/calibration.json",
                    help="5_evaluate.py calibration.json (prototype temperature per slot)")
    ap.add_argument("--plain_label_vectors", default=None, help="term_text/term_variants.h5 (optional)")
    ap.add_argument("--fold_seed", type=int, default=0, help="The --fold_seed of the 5_evaluate.py run")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out_dir = path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    pred = read_tsv(args.predictions)
    pred = pred[pred["gold"] != ""]
    proto = pred[pred["method"] == "prototype"]

    # 1a
    rel, within, auc = reliability(proto)
    for name, t in [("reliability", rel), ("margin_within_band", within)]:
        t.to_csv(os.path.join(out_dir, f"{name}.tsv"), sep="\t", index=False)

    # 1b
    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)])
    x = np.hstack([kw[[kw_row[s] for s in samples["sample_id"]]],
                   sb[[sb_row[s] for s in samples["sample_id"]]]]).astype(np.float32) / np.sqrt(2)
    terms = load_terms(args.ontology_terms)
    used = sorted(set().union(*[set(samples[s]) for s in SLOTS]) - {""})
    text = dict(zip(terms["term_id"], terms["text"]))
    tv = load_term_vectors(args.term_vectors, [text[t] for t in used]).astype(np.float32)
    calibration = json.load(open(path(args.calibration)))
    temperature = {slot: calibration[slot]["prototype"]["temperature"] for slot in SLOTS}
    dist = prototype_distributions(samples, x, np.hstack([tv, tv]) / np.sqrt(2), np.array(used), temperature, args.fold_seed)
    shape_auc, shape_band = shape_tests(dist)
    auc = pd.concat([auc.assign(test="probability vs margin (CV predictions)"),
                     shape_auc.assign(test="distribution shape (recomputed prototype)")], ignore_index=True)
    auc.to_csv(os.path.join(out_dir, "shape_auc.tsv"), sep="\t", index=False)
    shape_band.to_csv(os.path.join(out_dir, "shape_band.tsv"), sep="\t", index=False)

    # 2
    plain = None
    if args.plain_label_vectors:
        x_by_id = dict(zip(samples["sample_id"], x))
        plain, n_terms = plain_label_retrieval(args.plain_label_vectors, terms, x_by_id, list(proto["sample_id"].unique()))
        print(f"plain-label retrieval over {n_terms} terms", flush=True)
    zs = zero_shot_fallback(pred, term_ancestors(terms), plain)
    zs.to_csv(os.path.join(out_dir, "zero_shot_fallback.tsv"), sep="\t", index=False)

    pd.set_option("display.width", 250)
    print("\nreliability (accuracy per probability band)\n" + rel.pivot(index="band", columns="slot", values="accuracy").to_string())
    print("\ndoes anything add to the probability? (study-grouped CV AUC for 'top-1 is right')\n" + auc.to_string(index=False))
    print("\np1 in [0.25, 0.35): accuracy by shape of the rest of the distribution\n" + shape_band.to_string(index=False))
    print("\nzero-shot fallback\n" + zs.to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
