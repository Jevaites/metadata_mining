#!/usr/bin/env python3
"""
Summary tables of several 5_evaluate.py runs (trivial-methods-upgrades.md, sections 3 and 4).

  --runs      output dirs of the same configuration with --fold_seed 0..4
  --coarse    (optional) output dir of a run on coarsen_labels.py labels (fold_seed 0)

Prints, per method: mean ± sd of top-1 over the runs, the paired difference to `linear`
(mean [min, max] over runs: every run has the same test rows for all methods), and, for the first
run, top1_near_synonym, top1_gold_or_ancestor and the top-1 on coarser labels.

python experiments/summarise_runs.py --runs ~/MicrobeAtlasProject/ontology_mapping/experiments/cv_seed{0,1,2,3,4} \
  --coarse ~/MicrobeAtlasProject/ontology_mapping/experiments/cv_coarse100 \
  --output ~/MicrobeAtlasProject/ontology_mapping/experiments/summary.json
"""
import argparse
import json
import os

import numpy as np

SLOTS = ["biome", "feature", "material"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--coarse", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    load = lambda d: json.load(open(os.path.join(os.path.expanduser(d), "metrics.json")))
    runs = [load(d) for d in args.runs]
    coarse = load(args.coarse) if args.coarse else None
    out = {}
    for method in runs[0]["biome"]:
        m = {}
        for slot in SLOTS:
            top1 = np.array([r[slot][method]["top1"] for r in runs])
            diff = top1 - np.array([r[slot]["linear"]["top1"] for r in runs])
            first = runs[0][slot][method]
            m[slot] = {"top1_mean": round(top1.mean(), 4), "top1_sd": round(top1.std(), 4),
                       "diff_vs_linear_mean": round(diff.mean(), 4), "diff_min": round(diff.min(), 4), "diff_max": round(diff.max(), 4),
                       "run0_top1": first["top1"], "run0_near_synonym": first.get("top1_near_synonym"),
                       "run0_gold_or_ancestor": first.get("top1_gold_or_ancestor"),
                       "run0_unseen_label": first.get("top1_unseen_label"),
                       "coarse_top1": coarse[slot][method]["top1"] if coarse and method in coarse[slot] else None}
        out[method] = m
    with open(os.path.expanduser(args.output), "w") as handle:
        json.dump(out, handle, indent=1)
    f = lambda x: "  –  " if x is None else f"{x:.3f}"
    print(f"{len(runs)} runs. top-1 mean ± sd | vs linear: mean [min, max] (points)")
    for method, m in sorted(out.items(), key=lambda kv: -kv[1]["feature"]["top1_mean"]):
        print(f"{method:16s} " + "   ".join(
            f"{m[s]['top1_mean']:.3f}±{m[s]['top1_sd']:.3f} {100 * m[s]['diff_vs_linear_mean']:+.1f} "
            f"[{100 * m[s]['diff_min']:+.1f},{100 * m[s]['diff_max']:+.1f}]" for s in SLOTS))
    print("\nrun 0: exact / near-synonym / gold-or-ancestor / coarser labels / unseen-label top-1")
    for method, m in out.items():
        print(f"{method:16s} " + "   ".join("/".join(f(m[s][k]) for k in ["run0_top1", "run0_near_synonym",
              "run0_gold_or_ancestor", "coarse_top1", "run0_unseen_label"]) for s in SLOTS))


if __name__ == "__main__":
    main()
