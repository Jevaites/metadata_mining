#!/usr/bin/env python3
"""
Write a copy of the training set with coarser labels (trivial-methods-upgrades.md, section 4):
every label is replaced by its closest is_a ancestor (or itself) that has at least --min_support
samples labelled with it or below it, counted on the evaluated samples (same selection as
5_evaluate.py). Same rule as `granularity` in analyses.py. Then run 5_evaluate.py on the output.

python experiments/coarsen_labels.py --min_support 100 \
  --output ~/MicrobeAtlasProject/metalog/metalog_training_set__coarse100.tsv.gz
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyses import ancestor_map  # noqa: E402
from common import SLOTS, load_npz, load_terms, path, read_tsv, select_samples  # noqa: E402
from _setup import DEFAULTS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--min_support", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    terms = load_terms(args.ontology_terms)
    known = set(terms["term_id"])  # ancestors outside the term index (e.g. BFO roots) cannot be labels
    ancestors = {t: {a: d for a, d in up.items() if a in known} for t, up in ancestor_map(terms).items()}
    required = [set(load_npz(x)[0]) for x in [args.keywords, args.sub_biomes]]
    evaluated = select_samples(args.samples, required)
    table = read_tsv(args.samples)
    for slot in SLOTS:
        counts = evaluated[slot][evaluated[slot] != ""].value_counts()
        support = counts.copy()
        for term, n in counts.items():
            for a in ancestors.get(term, {}):
                support[a] = support.get(a, 0) + n
        mapping = {}
        for term in counts.index:
            up = sorted((d, a) for a, d in ancestors.get(term, {}).items() if support.get(a, 0) >= args.min_support)
            mapping[term] = term if support.get(term, 0) >= args.min_support or not up else up[0][1]
        table[slot] = table[slot].map(lambda t: mapping.get(t, t) if t else t)
        print(f"{slot}: {len(counts)} -> {evaluated[slot].map(lambda t: mapping.get(t, t)).loc[lambda s: s != ''].nunique()} labels")
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    table.to_csv(path(args.output), sep="\t", index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
