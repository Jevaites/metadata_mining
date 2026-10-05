#!/usr/bin/env python3
"""
Write a copy of the training set with coarser labels (trivial-methods-upgrades.md, section 4):
every label is replaced by its closest is_a ancestor (or itself) that has at least --min_support
samples labelled with it or below it, counted on the evaluated samples (same selection as
5_evaluate.py), with common.coarsen_labels (also used by `granularity` in analyses.py). Then run 5_evaluate.py on the output.

python experiments/coarsen_labels.py --min_support 100 \
  --output ~/MicrobeAtlasProject/metalog/metalog_training_set__coarse100.tsv.gz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SLOTS, ancestor_distances, coarsen_labels, load_npz, load_terms, path, read_tsv, select_samples  # noqa: E402
from _setup import DEFAULTS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--min_support", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    terms = load_terms(args.ontology_terms)
    distances = ancestor_distances(terms)
    known = set(terms["term_id"])  # ancestors outside the term index (e.g. BFO roots) cannot be labels
    required = [set(load_npz(x)[0]) for x in [args.keywords, args.sub_biomes]]
    evaluated = select_samples(args.samples, required)  # counted on the evaluated samples, as 5_evaluate.py
    table = read_tsv(args.samples)
    for slot in SLOTS:
        mapping = coarsen_labels(evaluated[slot], distances, args.min_support, known)
        table[slot] = table[slot].map(lambda t: mapping.get(t, t) if t else t)
        print(f"{slot}: {len(mapping)} -> {evaluated[slot].map(lambda t: mapping.get(t, t)).loc[lambda s: s != ''].nunique()} labels")
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    table.to_csv(path(args.output), sep="\t", index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
