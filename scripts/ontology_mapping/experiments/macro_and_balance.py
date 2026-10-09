#!/usr/bin/env python3
"""
Why is macro accuracy low, and does class balancing help? (project doc claude/macro-scores-and-class-balance.md;
experiments README §16)

1. Macro by label support, from the predictions of a 5_evaluate.py run (--predictions, --methods). A label used by a
   single study is never in the training folds when that study is tested, so every closed-vocabulary method scores 0
   on it. Per slot: share of single-study labels and of their samples, macro over all labels and over labels used by
   >= 2 / 5 / 10 studies, macro after dropping the 10 % rarest labels (by test samples), and per-label accuracy by
   number of studies (1, 2, 3-4, 5-9, 10-19, 20+).  -> support_summary.tsv, support_buckets.tsv
2. Class balancing, re-run with the pipeline's samples and folds for each --fold_seeds:
     linear (RidgeClassifier, as 5_evaluate.py)       plain
       class_weight = n / (K n_c)                     balanced
       weights proportional to n_c ** -gamma          tempered_0.25, tempered_0.5 (rescaled to a sample-weighted mean of 1)
       balanced, dominant class kept at weight 1      balanced_dominant_w1
       plain answer when it is the dominant class,    rule_keep_dominant
       else the balanced one
       plain scores - t * log(class prior)            logit_adj_0.02 / 0.05 / 0.1 (post-hoc logit adjustment)
     prototype (alpha 0.5) with prior weight beta     prototype_b0.1 (production), _b0.05, _b0, _b-0.05
   Metrics: micro (share of samples), macro (mean over gold labels), macro over learnable labels (>= 2 studies).
   For the first seed: which gold labels the balanced model loses and gains vs plain.
   -> balance_seed{S}_fold{F}.tsv.gz (per-sample predictions, cached), balance_scores.tsv (per seed),
      balance_summary.tsv (mean, min, max over seeds), balance_lost_gained.tsv
   Resumable: finished (seed, fold) files are kept; --max_seconds stops between folds (rerun to continue).

Example (~1 min per fold seed in the cloud):
python experiments/macro_and_balance.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --fold_seeds 0 1 2 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/macro_and_balance
"""

import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from _setup import DEFAULTS, SLOTS, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds
import methods as M  # noqa: E402  (importable once _setup has added scripts/ontology_mapping to the path)
from common import read_tsv  # noqa: E402

BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 10**9, "20+")]


def macro(hit, gold, labels=None):
    """Mean over gold labels of their per-label accuracy (optionally only over `labels`)."""
    per = pd.Series(np.asarray(hit, dtype=float)).groupby(np.asarray(gold)).mean()
    return per[per.index.isin(labels)].mean() if labels is not None else per.mean()


def studies_per_label(samples):
    """{slot: Series label -> number of studies using it} among the evaluated samples."""
    return {slot: samples[samples[slot] != ""].groupby(slot)["study_code"].nunique() for slot in SLOTS}


# ----------------------------------------------------------------------------- 1. macro by label support
def support_tables(pred, n_studies, methods):
    summary, buckets = [], []
    for method in methods:
        for slot in SLOTS:
            d = pred[(pred["method"] == method) & (pred["slot"] == slot)]
            hit = (d["pred"] == d["gold"]).to_numpy(float)
            per = pd.DataFrame({"gold": d["gold"].to_numpy(), "hit": hit}).groupby("gold")["hit"].agg(["size", "mean"])
            per["studies"] = n_studies[slot].reindex(per.index).fillna(0).astype(int)
            per = per.sort_values("size", kind="stable")
            single = per["studies"] == 1
            row = {"method": method, "slot": slot, "labels": len(per), "single_study_labels": int(single.sum()),
                   "single_study_label_share": round(single.mean(), 3),
                   "single_study_sample_share": round(per.loc[single, "size"].sum() / len(d), 3),
                   "micro": round(hit.mean(), 4), "macro": round(per["mean"].mean(), 4)}
            for k in (2, 5, 10):
                row[f"macro_ge{k}_studies"] = round(per.loc[per["studies"] >= k, "mean"].mean(), 4)
                row[f"labels_ge{k}_studies"] = int((per["studies"] >= k).sum())
            row["macro_drop_10pct_rarest"] = round(per["mean"].iloc[int(np.ceil(0.1 * len(per))):].mean(), 4)
            summary.append(row)
            for lo, hi, name in BUCKETS:
                b = per[(per["studies"] >= lo) & (per["studies"] <= hi)]
                if len(b):
                    buckets.append({"method": method, "slot": slot, "studies": name, "labels": len(b),
                                    "samples": int(b["size"].sum()), "mean_label_accuracy": round(b["mean"].mean(), 3),
                                    "pooled_accuracy": round(np.average(b["mean"], weights=b["size"]), 3)})
    return pd.DataFrame(summary), pd.DataFrame(buckets)


# ----------------------------------------------------------------------------- 2. class balancing
def ridge_variants(x_train, y, x_test):
    """{variant: top-1 labels} of the linear variants (see the module docstring). Five ridge fits per call."""
    classes, counts = np.unique(y, return_counts=True)
    log_prior = np.log(counts / counts.sum())
    dominant = classes[counts.argmax()]

    def fit(weights):
        clf = RidgeClassifier(alpha=1.0, class_weight=weights).fit(x_train, y)
        return clf.classes_, clf.decision_function(x_test)

    out = {}
    cls, plain = fit(None)
    out["plain"] = cls[plain.argmax(1)]
    for t in (0.02, 0.05, 0.1):  # cls == classes (np.unique order), so log_prior is aligned with the columns
        out[f"logit_adj_{t}"] = cls[(plain - t * log_prior).argmax(1)]
    cls, s = fit("balanced")
    out["balanced"] = cls[s.argmax(1)]
    for gamma in (0.25, 0.5):
        w = counts.astype(float) ** -gamma
        w /= np.average(w, weights=counts)  # same total weight as the plain model
        cls, s = fit(dict(zip(classes, w)))
        out[f"tempered_{gamma}"] = cls[s.argmax(1)]
    w = len(y) / (len(classes) * counts.astype(float))
    w[counts.argmax()] = 1.0
    cls, s = fit(dict(zip(classes, w)))
    out["balanced_dominant_w1"] = cls[s.argmax(1)]
    out["rule_keep_dominant"] = np.where(out["plain"] == dominant, dominant, out["balanced"])
    return out


def run_fold(samples, x, term_matrix, term_ids, train, test, betas):
    """Per-sample predictions of every variant on one fold, all slots."""
    rows = []
    for slot in SLOTS:
        a, b = samples[slot].to_numpy()[train] != "", samples[slot].to_numpy()[test] != ""
        y, x_train, x_test = samples[slot].to_numpy()[train][a], x[train][a], x[test][b]
        preds = ridge_variants(x_train, y, x_test)
        closed = np.isin(term_ids, y)  # prototypes for the labels seen in this training fold, as 5_evaluate.py
        for beta in betas:
            model = M.prototype_model(x_train, y, term_matrix[closed], term_ids[closed], 0.5, beta)
            preds[f"prototype_b{beta:g}"] = term_ids[closed][M.prototype_scores(x_test, model).argmax(1)]
        ids, gold = samples["sample_id"].to_numpy()[test][b], samples[slot].to_numpy()[test][b]
        for variant, p in preds.items():
            rows.append(pd.DataFrame({"slot": slot, "variant": variant, "sample_id": ids, "gold": gold, "pred": p}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "keywords", "sub_biomes", "term_vectors"]:
        ap.add_argument(f"--{name}", default=DEFAULTS[name])
    ap.add_argument("--samples", default="~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz")
    ap.add_argument("--predictions", default="~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz",
                    help="5_evaluate.py predictions.tsv.gz, for part 1")
    ap.add_argument("--methods", nargs="+", default=["prototype", "linear", "knn_study"], help="Part 1 methods")
    ap.add_argument("--fold_seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.1, 0.05, 0.0, -0.05], help="Prototype prior weights")
    ap.add_argument("--max_seconds", type=float, default=0, help="Stop between folds after this long (0 = no limit)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out_dir = path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)])
    x = np.hstack([kw[[kw_row[s] for s in samples["sample_id"]]],
                   sb[[sb_row[s] for s in samples["sample_id"]]]]).astype(np.float32) / np.sqrt(2)
    n_studies = studies_per_label(samples)
    learnable = {slot: set(n[n >= 2].index) for slot, n in n_studies.items()}
    print(f"{len(samples)} samples, {samples['study_code'].nunique()} studies", flush=True)

    # --- 1. macro by label support (from the existing CV predictions)
    pred = read_tsv(args.predictions)
    pred = pred[pred["gold"] != ""]
    summary, buckets = support_tables(pred, n_studies, args.methods)
    summary.to_csv(os.path.join(out_dir, "support_summary.tsv"), sep="\t", index=False)
    buckets.to_csv(os.path.join(out_dir, "support_buckets.tsv"), sep="\t", index=False)
    print("\n" + summary[["method", "slot", "labels", "single_study_label_share", "single_study_sample_share", "micro",
                          "macro", "macro_ge2_studies", "macro_ge5_studies", "macro_ge10_studies",
                          "macro_drop_10pct_rarest"]].to_string(index=False))
    print("\n" + buckets[buckets["method"] == args.methods[0]].to_string(index=False), flush=True)

    # --- 2. class balancing, per fold seed and fold (cached)
    terms = load_terms(args.ontology_terms)
    used = sorted(set().union(*[set(samples[s]) for s in SLOTS]) - {""})
    text = dict(zip(terms["term_id"], terms["text"]))
    tv = load_term_vectors(args.term_vectors, [text[t] for t in used]).astype(np.float32)
    term_matrix, term_ids = np.hstack([tv, tv]) / np.sqrt(2), np.array(used)  # same space as [kw, sb] / sqrt(2)
    for seed in args.fold_seeds:
        for f, (train, test) in enumerate(study_folds(samples["study_code"], 5, seed)):
            out = os.path.join(out_dir, f"balance_seed{seed}_fold{f}.tsv.gz")
            if os.path.exists(out):
                continue
            if args.max_seconds and time.time() - start > args.max_seconds:
                print(f"stopping after {time.time() - start:.0f}s; rerun to continue")
                return
            run_fold(samples, x, term_matrix, term_ids, train, test, args.betas).to_csv(out + ".tmp", sep="\t",
                                                                                         index=False, compression="gzip")
            os.replace(out + ".tmp", out)  # a fold file exists only when complete
            print(f"seed {seed} fold {f} done ({time.time() - start:.0f}s)", flush=True)

    scores = []
    for seed in args.fold_seeds:
        r = pd.concat([read_tsv(os.path.join(out_dir, f"balance_seed{seed}_fold{f}.tsv.gz")) for f in range(5)])
        r["hit"] = (r["pred"] == r["gold"]).astype(float)
        for (slot, variant), d in r.groupby(["slot", "variant"], sort=False):
            scores.append({"fold_seed": seed, "slot": slot, "variant": variant, "micro": round(d["hit"].mean(), 4),
                           "macro": round(macro(d["hit"], d["gold"]), 4),
                           "macro_learnable": round(macro(d["hit"], d["gold"], learnable[slot]), 4)})
        if seed == args.fold_seeds[0]:  # which labels does full balancing lose / gain vs plain?
            lost_gained = []
            for slot in SLOTS:
                d = r[r["slot"] == slot].pivot_table(index=["sample_id", "gold"], columns="variant", values="hit").reset_index()
                lost = d[(d["plain"] == 1) & (d["balanced"] == 0)]
                gained = d[(d["plain"] == 0) & (d["balanced"] == 1)]
                dominant = samples.loc[samples[slot] != "", slot].value_counts().index[0]
                lost_gained.append({"fold_seed": seed, "slot": slot, "lost": len(lost), "gained": len(gained),
                                    "dominant_label": dominant, "lost_on_dominant": int((lost["gold"] == dominant).sum()),
                                    "top_lost_labels": "; ".join(f"{k} ({v})" for k, v in Counter(lost["gold"]).most_common(5))})
            pd.DataFrame(lost_gained).to_csv(os.path.join(out_dir, "balance_lost_gained.tsv"), sep="\t", index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(os.path.join(out_dir, "balance_scores.tsv"), sep="\t", index=False)
    agg = scores.groupby(["slot", "variant"], sort=False)[["micro", "macro", "macro_learnable"]].agg(["mean", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.round(4).reset_index()
    agg.to_csv(os.path.join(out_dir, "balance_summary.tsv"), sep="\t", index=False)
    print(f"\nclass balancing, fold seeds {args.fold_seeds} (mean [min, max])")
    for slot in SLOTS:
        print(f"\n{slot}")
        a = agg[agg["slot"] == slot]
        for r in a.itertuples():
            print(f"  {r.variant:22s} micro {r.micro_mean:.3f} [{r.micro_min:.3f}, {r.micro_max:.3f}]  "
                  f"macro {r.macro_mean:.3f} [{r.macro_min:.3f}, {r.macro_max:.3f}]  "
                  f"learnable {r.macro_learnable_mean:.3f} [{r.macro_learnable_min:.3f}, {r.macro_learnable_max:.3f}]")
    print("\n" + pd.read_csv(os.path.join(out_dir, "balance_lost_gained.tsv"), sep="\t").to_string(index=False))
    print(f"\nwrote {out_dir} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
