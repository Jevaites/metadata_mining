#!/usr/bin/env python3
"""
Does the Metalog cleaning (2b_clean_metalog.py) change the ontology mapping?

A. Training data. One fixed gold test set; `linear` (RidgeClassifier(alpha=1), as in 5_evaluate.py)
   trained on progressively cleaner training folds:
     raw            every linked sample (current practice)
     no_controls    minus hard drops (controls, conflicting duplicates)
     no_artificial  also minus Metalog's perturbed + degraded samples
     no_audit       also minus unreviewed audit hits (= the gold definition)
     dedup          also minus repeated texts within a study
   Folds, 50-per-study cap and shuffle are those of 5_evaluate.py (common.select_samples / study_folds),
   so the `raw` arm scored on the raw test set reproduces the pipeline's `linear` numbers.
B. Evaluation. The raw-trained model scored on the raw test set vs the gold test set, and per bucket.
C. Detector. Can the sample embeddings flag artificial samples (for the 6M atlas, and to rank the
   manual review)? Logistic regression, study-grouped CV; unreviewed audit hits are left out of
   training and scored out-of-fold.

The review of audit_review.tsv is not done yet, so unreviewed audit hits are bracketed: the gold test
set is given without them (`test_gold`, as in training_set.gold) and with them (`test_gold_audit_ok`).

cd ~/github/mm-ontology-mapping/scripts/ontology_mapping
python experiments/cleaning_effect.py --clean_dir ~/MicrobeAtlasProject/metalog/clean \
  --output ~/MicrobeAtlasProject/metalog/clean/experiments/cleaning_effect.json
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _setup import parser, save_json  # noqa: E402
from common import SLOTS, load_npz, path, read_tsv, select_samples, study_folds  # noqa: E402

CAP = 50
ARMS = ["raw", "no_controls", "no_artificial", "no_audit", "dedup"]
TESTS = ["test_raw", "test_gold_audit_ok", "test_gold"]


def load_universe(args):
    """Every linked sample with both embeddings and >=1 label, in 5_evaluate.py's shuffle order, plus flags."""
    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    u = select_samples(args.samples, [set(kw_row), set(sb_row)], max_per_study=0)
    X = np.hstack([kw[[kw_row[s] for s in u["sample_id"]]], sb[[sb_row[s] for s in u["sample_id"]]]]) / np.sqrt(2)

    flags = read_tsv(os.path.join(path(args.clean_dir), "metalog_flags.tsv.gz"))
    flags = flags[flags["drop_reason"] != "duplicate_alias"].drop_duplicates("spire_sample_name")
    keep = ["drop_reason", "artificial_bucket", "audit_unreviewed", "audit_hits"]
    u = u.join(flags.set_index("spire_sample_name")[keep], on="spire_sample_name")
    if u["artificial_bucket"].isna().any():
        raise SystemExit(f"{u['artificial_bucket'].isna().sum()} samples not in metalog_flags.tsv.gz: re-run 2b")
    u["audit_unreviewed"] = u["audit_unreviewed"] == "True"
    u["dup_text"] = u.duplicated(["study_code", "text"], keep="first")
    u["capped"] = u.groupby("study_code").cumcount() < CAP  # == select_samples(max_per_study=50)
    return u.reset_index(drop=True), X.astype(np.float32)


def masks(u):
    dropped = u["drop_reason"] != ""
    artificial = u["artificial_bucket"] != "none"
    train = {"raw": np.ones(len(u), bool)}
    train["no_controls"] = train["raw"] & ~dropped
    train["no_artificial"] = train["no_controls"] & ~artificial
    train["no_audit"] = train["no_artificial"] & ~u["audit_unreviewed"]
    train["dedup"] = train["no_audit"] & ~u["dup_text"]
    gold_ok = ~dropped & ~artificial & ~u["dup_text"]
    test = {"test_raw": np.ones(len(u), bool), "test_gold_audit_ok": gold_ok, "test_gold": gold_ok & ~u["audit_unreviewed"]}
    return {k: np.asarray(v) for k, v in train.items()}, {k: np.asarray(v) for k, v in test.items()}


def cap(u, mask):
    """Rows of `mask`, at most CAP per study, keeping the shuffle order (filter first, then cap)."""
    sub = u[mask]
    return sub.index[sub.groupby("study_code").cumcount() < CAP].to_numpy()


def macro(correct, gold):
    return float(pd.Series(correct).groupby(np.asarray(gold)).mean().mean())


def bootstrap_diff(correct_a, correct_b, studies, n=2000, seed=0):
    """Paired bootstrap over test studies of mean(a) - mean(b): (mean, 2.5 %, 97.5 %) in points."""
    df = pd.DataFrame({"a": correct_a, "b": correct_b, "s": studies}).groupby("s").agg(["sum", "count"])
    a, b, cnt = df[("a", "sum")].to_numpy(), df[("b", "sum")].to_numpy(), df[("a", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (a[idx].sum(1) - b[idx].sum(1)) / cnt[idx].sum(1)
    return [round(100 * (a.sum() - b.sum()) / cnt.sum(), 2), *np.round(100 * np.percentile(diffs, [2.5, 97.5]), 2).tolist()]


def part_a_b(u, X, train_m, test_m, fold_seed):
    """Out-of-fold predictions of every arm on every test row of the fold -> tidy frame."""
    raw_capped = u[u["capped"]]
    fold_of = {}
    for f, (_, te) in enumerate(study_folds(raw_capped["study_code"], 5, fold_seed)):
        fold_of.update({s: f for s in raw_capped["study_code"].iloc[te]})
    fold = u["study_code"].map(fold_of).to_numpy()
    test_rows = np.union1d(cap(u, test_m["test_raw"]), np.union1d(cap(u, test_m["test_gold_audit_ok"]), cap(u, test_m["test_gold"])))
    records = []
    for f in range(5):
        te = test_rows[fold[test_rows] == f]
        for arm in ARMS:
            tr = cap(u, train_m[arm] & (fold != f))
            for slot in SLOTS:
                t = tr[u[slot].to_numpy()[tr] != ""]
                e = te[u[slot].to_numpy()[te] != ""]
                clf = RidgeClassifier(alpha=1.0).fit(X[t], u[slot].to_numpy()[t])
                pred = clf.predict(X[e])
                records.append(pd.DataFrame({"row": e, "slot": slot, "arm": arm, "fold": f,
                                             "correct": pred == u[slot].to_numpy()[e]}))
        print(f"  seed {fold_seed} fold {f} done", flush=True)
    pred = pd.concat(records, ignore_index=True)
    # membership of each test set (capped independently, as a pipeline run on that file would)
    for name in TESTS:
        pred[name] = pred["row"].isin(cap(u, test_m[name]))
    return pred


def summarise_a_b(u, preds):
    out = {"A_training_arms": {}, "B_evaluation": {}}
    p0 = preds[0]
    for test in TESTS:
        tab = {}
        for slot in SLOTS:
            row = {}
            for arm in ARMS:
                accs = [p[(p.slot == slot) & (p.arm == arm) & p[test]]["correct"].mean() for p in preds]
                q = p0[(p0.slot == slot) & (p0.arm == arm) & p0[test]].sort_values("row")
                base = p0[(p0.slot == slot) & (p0.arm == "raw") & p0[test]].sort_values("row")
                gold = u[slot].to_numpy()[q["row"]]
                row[arm] = {"top1_mean_over_seeds": round(float(np.mean(accs)), 4),
                            "top1_per_seed": [round(float(a), 4) for a in accs],
                            "macro_top1_seed0": round(macro(q["correct"].to_numpy(), gold), 4),
                            "vs_raw_points_seed0_[mean,lo,hi]": bootstrap_diff(
                                q["correct"].to_numpy(), base["correct"].to_numpy(), u["study_code"].to_numpy()[q["row"]]),
                            "n_test": int(len(q))}
            tab[slot] = row
        out["A_training_arms"][test] = tab
    # B: raw arm, per bucket of the test rows (seed 0, raw test set)
    q = p0[(p0.arm == "raw") & p0["test_raw"]].copy()
    r = u.loc[q["row"]]
    q["group"] = np.select([r["drop_reason"].to_numpy() != "", r["artificial_bucket"].to_numpy() != "none",
                            r["audit_unreviewed"].to_numpy(), r["dup_text"].to_numpy()],
                           ["control/dropped", "perturbed+degraded", "audit_unreviewed", "repeated_text"], "gold")
    out["B_evaluation"]["raw_model_by_group"] = {
        slot: {g: {"top1": round(float(d["correct"].mean()), 4), "n": int(len(d))} for g, d in q[q.slot == slot].groupby("group")}
        for slot in SLOTS}
    return out


def part_c(u, X, fold_seed=0, cap_c=200):
    """Artificial-sample detector: out-of-fold scores for every row (audit hits never trained on)."""
    y = (u["artificial_bucket"] != "none").to_numpy()
    usable = ~u["audit_unreviewed"].to_numpy()
    rows = u.index[u.groupby("study_code").cumcount() < cap_c].to_numpy()
    scores = np.full(len(u), np.nan)
    folds = list(study_folds(u["study_code"].to_numpy()[rows], 5, fold_seed))
    fold_of_study = {}
    for f, (_, te) in enumerate(folds):
        fold_of_study.update({s: f for s in u["study_code"].to_numpy()[rows][te]})
    fold = u["study_code"].map(fold_of_study).to_numpy()
    for f in range(5):
        tr = rows[(fold[rows] != f) & usable[rows]]
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
        te = np.where(fold == f)[0]  # all rows of the test studies, uncapped
        scores[te] = clf.predict_proba(X[te])[:, 1]
    in_rows = np.zeros(len(u), bool)
    in_rows[rows] = True
    ev = usable & in_rows  # scored on the same <=200-per-study rows, so no single study dominates
    res = {"n_pos": int(y[ev].sum()), "n_neg": int((~y[ev]).sum()), "pos_studies": int(u.loc[ev & y, "study_code"].nunique()),
           "roc_auc": round(float(roc_auc_score(y[ev], scores[ev])), 4),
           "average_precision": round(float(average_precision_score(y[ev], scores[ev])), 4),
           "base_rate": round(float(y[ev].mean()), 4)}
    for thr in (0.5, 0.9):
        flag = scores[ev] >= thr
        res[f"at_{thr}"] = {"precision": round(float(y[ev][flag].mean()), 4) if flag.any() else None,
                            "recall": round(float(flag[y[ev]].mean()), 4), "flagged": int(flag.sum())}
    res["recall_at_0.5_by_bucket"] = {b: round(float((scores[ev & (u["artificial_bucket"] == b).to_numpy()] >= 0.5).mean()), 4)
                                      for b in ["perturbed", "degraded", "control"]}
    res["mean_score"] = {"gold_negatives": round(float(np.nanmean(scores[ev & ~y])), 4),
                         "artificial": round(float(np.nanmean(scores[ev & y])), 4),
                         "audit_unreviewed": round(float(np.nanmean(scores[~usable])), 4)}
    return res, scores


def review_with_detector(u, scores, clean_dir, out_path):
    review = read_tsv(os.path.join(clean_dir, "audit_review.tsv"))
    au = u[u["audit_unreviewed"]].assign(score=scores[u["audit_unreviewed"].to_numpy()])
    mean, n = [], []
    for r in review.itertuples():
        sel = au[(au["study_code"] == r.id) & au["audit_hits"].str.split(";").apply(lambda h: r.category in h)]
        mean.append(round(float(sel["score"].mean()), 3) if len(sel) else "")
        n.append(len(sel))
    review.insert(4, "detector_mean_score", mean)
    review.insert(5, "n_linked_embedded", n)
    review.to_csv(out_path, sep="\t", index=False)


def main():
    p = parser(__doc__)
    p.add_argument("--clean_dir", required=True, help="--output_dir of 2b_clean_metalog.py")
    p.add_argument("--fold_seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    clean_dir = path(args.clean_dir)

    u, X = load_universe(args)
    train_m, test_m = masks(u)
    print(f"{len(u)} linked samples with embeddings ({u['capped'].sum()} after the {CAP}-per-study cap), "
          f"{u['study_code'].nunique()} studies", flush=True)
    results = {"n": {"universe": int(len(u)), "raw_capped": int(u["capped"].sum()),
                     **{f"train_{k}_capped": int(len(cap(u, v))) for k, v in train_m.items()},
                     **{f"{k}_capped": int(len(cap(u, v))) for k, v in test_m.items()}}}
    preds = [part_a_b(u, X, train_m, test_m, s) for s in args.fold_seeds]
    results.update(summarise_a_b(u, preds))
    results["C_detector"], scores = part_c(u, X)
    out_dir = os.path.dirname(path(args.output))
    os.makedirs(out_dir, exist_ok=True)
    review_with_detector(u, scores, clean_dir, os.path.join(out_dir, "audit_review_with_detector.tsv"))
    save_json(results, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
