#!/usr/bin/env python3
"""
Extra analyses behind the findings report. Not needed to run the pipeline.

Reads the outputs of 5_evaluate.py and writes analyses.json with:
  bootstrap        paired difference in top-1 between two runs/methods on the same test rows,
                   95 % CI from resampling *studies* with replacement
  hierarchy        where the `linear` top-1 lands relative to the gold term in the is_a graph:
                   exact / an ancestor of gold (true but coarser) / a descendant (too specific) /
                   another branch (a real error)
  label_ceiling    for every sample, its most similar sample in *another* study; when the two are
                   near-duplicates (cosine >= 0.9), how often Metalog gives them the same label,
                   and the model's top-1 on those samples vs the others
  granularity      top-1 of `linear` after replacing every label by its closest ancestor that has
                   at least N samples (a coarser label policy; N = 100, 300)

python analyses.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --features ~/MicrobeAtlasProject/metalog/keywords__large1024.npz ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --run ~/MicrobeAtlasProject/ontology_mapping/cv_kw_sb \
  --baseline_run ~/MicrobeAtlasProject/ontology_mapping/cv_tfidf \
  --output ~/MicrobeAtlasProject/ontology_mapping/analyses.json
"""

import argparse
import importlib
import json
from collections import deque

import numpy as np
import pandas as pd

from common import SLOTS, load_npz, load_terms, path, read_tsv, select_samples, study_folds

evaluate = importlib.import_module("5_evaluate")  # reuse linear_scores / rank


def ancestor_map(terms):
    """term -> {ancestor: distance} over is_a (distance 1 = direct parent)."""
    parents = {t: [p for p in ps.split("||") if p] for t, ps in zip(terms["term_id"], terms["parents"])}
    result = {}
    for term in parents:
        distance, queue = {}, deque([(term, 0)])
        while queue:
            node, d = queue.popleft()
            for parent in parents.get(node, []):
                if parent not in distance:
                    distance[parent] = d + 1
                    queue.append((parent, d + 1))
        result[term] = distance
    return result


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """a, b: DataFrames with columns study_code, hit (same rows). -> mean diff (b - a) and 95 % CI."""
    diff = pd.DataFrame({"study": a["study_code"].to_numpy(), "d": b["hit"].to_numpy().astype(float) -
                         a["hit"].to_numpy().astype(float)})
    per_study = diff.groupby("study")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_study), size=(n_boot, len(per_study)))
    boot = per_study["sum"].to_numpy()[picks].sum(1) / per_study["count"].to_numpy()[picks].sum(1)
    return {"diff": round(diff["d"].mean(), 4), "ci95": [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]}


def hits(run_dir, slot, method):
    pred = read_tsv(f"{path(run_dir)}/predictions.tsv.gz")
    g = pred[(pred["slot"] == slot) & (pred["method"] == method)].sort_values("sample_id")
    return g.assign(hit=g["pred"] == g["gold"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--features", nargs="+", required=True, help="The .npz blocks of --run")
    parser.add_argument("--run", required=True, help="5_evaluate.py output dir of the main run")
    parser.add_argument("--baseline_run", default=None, help="Another run on the same samples (e.g. tfidf)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_per_study", type=int, default=50)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--fold_seed", type=int, default=0)
    args = parser.parse_args()
    terms = load_terms(args.ontology_terms)
    ancestors = ancestor_map(terms)
    out = {}

    # --- bootstrap: methods of the main run vs its linear, and linear vs the baseline run's linear
    out["bootstrap"] = {}
    for slot in SLOTS:
        linear = hits(args.run, slot, "linear")
        entry = {f"{m} - linear": paired_bootstrap(linear, hits(args.run, slot, m)) for m in ["label_reg", "hybrid"]}
        if args.baseline_run:
            entry["linear - baseline linear"] = paired_bootstrap(hits(args.baseline_run, slot, "linear"), linear)
        out["bootstrap"][slot] = entry

    # --- hierarchy: where do linear's top-1 predictions land?
    out["hierarchy"] = {}
    for slot in SLOTS:
        g = hits(args.run, slot, "linear")
        kind = ["exact" if p == t else "ancestor_of_gold" if p in ancestors.get(t, {}) else
                "descendant_of_gold" if t in ancestors.get(p, {}) else "other_branch" for p, t in zip(g["pred"], g["gold"])]
        out["hierarchy"][slot] = pd.Series(kind).value_counts(normalize=True).round(4).to_dict()

    # --- label ceiling: cross-study near-duplicates
    npz = [load_npz(p) for p in args.features]
    samples = select_samples(args.samples, [set(r) for r, _ in npz], args.max_per_study, args.seed)
    X = np.hstack([v[[r[s] for s in samples["sample_id"]]] for r, v in npz]) / np.sqrt(len(npz))
    study = samples["study_code"].to_numpy()
    nearest, nearest_sim = np.zeros(len(X), int), np.zeros(len(X))
    for start in range(0, len(X), 2000):
        sim = X[start:start + 2000] @ X.T
        sim[study[start:start + 2000, None] == study[None, :]] = -1  # only other studies
        nearest[start:start + 2000], nearest_sim[start:start + 2000] = sim.argmax(1), sim.max(1)
    out["label_ceiling"] = {"share_with_near_duplicate": round(float((nearest_sim >= 0.9).mean()), 4)}
    for slot in SLOTS:
        gold = samples[slot].to_numpy()
        pair = (nearest_sim >= 0.9) & (gold != "") & (gold[nearest] != "")
        correct = hits(args.run, slot, "linear").set_index("sample_id")["hit"]
        on = samples["sample_id"][pair]
        off = samples["sample_id"][~pair & (gold != "")]
        out["label_ceiling"][slot] = {"near_duplicates_share_label": round(float((gold[pair] == gold[nearest][pair]).mean()), 4),
                                      "linear_top1_on_near_duplicates": round(float(correct[on].mean()), 4),
                                      "linear_top1_on_others": round(float(correct[off].mean()), 4)}

    # --- granularity: coarser labels, same linear model and folds
    out["granularity"] = {}
    folds = list(study_folds(study, 5, args.fold_seed))
    for slot in SLOTS:
        gold = samples[slot].to_numpy()
        has = gold != ""
        counts = pd.Series(gold[has]).value_counts()
        out["granularity"][slot] = {}
        for min_support in [0, 100, 300]:
            support = counts.copy()  # samples labelled with the term or any of its descendants
            for term, n in counts.items():
                for ancestor in ancestors.get(term, {}):
                    support[ancestor] = support.get(ancestor, 0) + n
            def coarsen(term):
                if min_support == 0 or support.get(term, 0) >= min_support:
                    return term
                up = sorted((d, a) for a, d in ancestors.get(term, {}).items() if support.get(a, 0) >= min_support)
                return up[0][1] if up else term
            labels = np.array([coarsen(t) if t else "" for t in gold])
            hit = []
            for train_idx, test_idx in folds:
                tr, te = train_idx[has[train_idx]], test_idx[has[test_idx]]
                classes, scores = evaluate.linear_scores(X[te], X[tr], labels[tr])
                hit += list(np.array([r[0] for r in evaluate.rank(scores, classes)[0]]) == labels[te])
            out["granularity"][slot][f"min_support_{min_support}"] = {
                "n_labels": int(pd.Series(labels[has]).nunique()), "top1": round(float(np.mean(hit)), 4)}
        print(slot, out["granularity"][slot])

    with open(path(args.output), "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
