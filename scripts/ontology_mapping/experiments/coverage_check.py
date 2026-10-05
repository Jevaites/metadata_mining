#!/usr/bin/env python3
"""
Validate the out-of-coverage flag of 6b_coverage.py (claude/coverage-flag-and-plant-labels.md).

1. Metalog (out-of-fold): per slot, accuracy of the CV predictions (--cv_predictions, --cv_method)
   for held-out samples flagged vs not flagged (metalog_held_out_sim.tsv.gz of 6b_coverage.py =
   similarity to other projects' training samples), and whether coverage adds to the calibrated
   probability (cross-validated AUC of a logistic regression on prob vs prob + coverage).
2. Gold samples outside Metalog: flag rate per gold coarse biome, and coarse consistency
   (term_to_coarse_biome.tsv of gold_check.py) of the top-1 and back-off answers, flagged vs not.

python experiments/coverage_check.py \
  --coverage_dir ~/MicrobeAtlasProject/ontology_mapping/atlas_coverage \
  --gold_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/gold_check \
  --training_set ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --cv_predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SLOTS, ancestor_sets, path, read_tsv  # noqa: E402


def cv_auc(X, y):
    return roc_auc_score(y, cross_val_predict(LogisticRegression(), X, y, cv=5, method="predict_proba")[:, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage_dir", required=True, help="6b_coverage.py --output_dir")
    ap.add_argument("--gold_dir", required=True, help="gold_check.py --dir (after gold_check.py has run)")
    ap.add_argument("--training_set", required=True)
    ap.add_argument("--ontology_terms", required=True)
    ap.add_argument("--cv_predictions", required=True)
    ap.add_argument("--cv_method", default="prototype")
    args = ap.parse_args()
    d = path(args.coverage_dir)
    threshold = json.load(open(os.path.join(d, "coverage_calibration.json")))["threshold"]
    out = {"threshold": threshold}

    # 1. Metalog out-of-fold
    terms = read_tsv(args.ontology_terms)
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p})
    held = read_tsv(os.path.join(d, "metalog_held_out_sim.tsv.gz"))
    cv = read_tsv(args.cv_predictions)
    cv = cv[cv["method"] == args.cv_method].merge(held[["sample_id", "held_out_sim"]], on="sample_id")
    cv["sim"], cv["prob"] = cv["held_out_sim"].astype(float), cv["prob"].astype(float)
    cv["flag"], cv["top1"], cv["answered"] = cv["sim"] < threshold, cv["pred"] == cv["gold"], cv["backoff"] != ""
    cv["strict"] = [b == g or b in anc.get(g, ()) for b, g in zip(cv["backoff"], cv["gold"])]
    rows = []
    for slot in SLOTS:
        g = cv[cv["slot"] == slot]
        a = g[g["answered"]]
        rows.append({"slot": slot, "flagged": g["flag"].mean(),
                     "top1_unflagged": g["top1"][~g["flag"]].mean(), "top1_flagged": g["top1"][g["flag"]].mean(),
                     "answered_unflagged": g["answered"][~g["flag"]].mean(), "answered_flagged": g["answered"][g["flag"]].mean(),
                     "backoff_acc_unflagged": a["strict"][~a["flag"]].mean(), "backoff_acc_flagged": a["strict"][a["flag"]].mean(),
                     "n_answered_flagged": int(a["flag"].sum()),
                     "auc_prob": cv_auc(g[["prob"]].values, g["top1"]), "auc_prob_coverage": cv_auc(g[["prob", "sim"]].values, g["top1"])})
    metalog = pd.DataFrame(rows).set_index("slot")
    print(f"threshold {threshold}\n\nMetalog, out-of-fold ({args.cv_method}):\n{metalog.round(3).T.to_string()}")
    out["metalog"] = metalog.round(4).to_dict(orient="index")

    # 2. gold samples outside Metalog
    gd = path(args.gold_dir)
    cov = read_tsv(os.path.join(d, "atlas_coverage.tsv.gz"))
    linked = set(read_tsv(args.training_set)["sample_id"])
    gold = read_tsv(os.path.join(gd, "gold_labels.tsv")).merge(read_tsv(os.path.join(gd, "atlas_backoff_gold.tsv")), on="sample_id")
    gold = gold.merge(cov, on="sample_id")
    gold = gold[~gold["sample_id"].isin(linked)]
    gold["flag"], gold["sim"] = gold["in_coverage"] != "True", gold["coverage_sim"].astype(float)
    tm = read_tsv(os.path.join(gd, "term_to_coarse_biome.tsv"))
    compatible = dict(zip(tm["term_id"], tm["compatible"].str.split("|")))
    by_class = gold.groupby("gold_biome").agg(n=("flag", "size"), flagged=("flag", "mean"), median_sim=("sim", "median"))
    print(f"\nGold samples outside Metalog: {len(gold)}, flagged {gold['flag'].mean():.3f}\n{by_class.round(3).to_string()}")
    rows = []
    for slot in SLOTS:
        ok_top = np.array([b in compatible.get(t, []) for b, t in zip(gold["gold_biome"], gold[slot])])
        answered = (gold[f"{slot}_backoff"] != "").to_numpy()
        ok_back = np.array([b in compatible.get(t, []) for b, t in zip(gold["gold_biome"], gold[f"{slot}_backoff"])])
        f = gold["flag"].to_numpy()
        rows.append({"slot": slot, "top1_consistent_unflagged": ok_top[~f].mean(), "top1_consistent_flagged": ok_top[f].mean(),
                     "answered_unflagged": answered[~f].mean(), "answered_flagged": answered[f].mean(),
                     "backoff_consistent_unflagged": ok_back[answered & ~f].mean(),
                     "backoff_consistent_flagged": ok_back[answered & f].mean() if (answered & f).any() else np.nan,
                     "n_answered_flagged": int((answered & f).sum())})
    g_tab = pd.DataFrame(rows).set_index("slot")
    print(f"\nGold, coarse consistency (lenient):\n{g_tab.round(3).T.to_string()}")
    out["gold"] = {"n": len(gold), "flagged": round(float(gold["flag"].mean()), 4),
                   "by_class": by_class.round(4).to_dict(orient="index"), "by_slot": g_tab.round(4).to_dict(orient="index")}
    json.dump(out, open(os.path.join(d, "coverage_check.json"), "w"), indent=1)
    print(f"\nWrote {os.path.join(d, 'coverage_check.json')}")


if __name__ == "__main__":
    main()
