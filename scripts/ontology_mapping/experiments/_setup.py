"""Shared set-up for the experiment scripts: the exact samples, vectors and folds of 5_evaluate.py.

Default paths are the ones of the pipeline README; override them with the command-line options.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ontology_mapping
from common import SLOTS, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds  # noqa: E402,F401

P = "~/MicrobeAtlasProject"
DEFAULTS = {
    "ontology_terms": f"{P}/ontology_terms.tsv.gz",
    "samples": f"{P}/metalog/metalog_training_set.tsv.gz",
    "keywords": f"{P}/metalog/keywords__large1024.npz",
    "sub_biomes": f"{P}/metalog/sub_biomes__large1024.npz",
    "term_vectors": f"{P}/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5",
}


def parser(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in DEFAULTS.items():
        p.add_argument(f"--{name}", default=default)
    p.add_argument("--fold_seed", type=int, default=0, help="Study-to-fold assignment, as in 5_evaluate.py")
    p.add_argument("--output", required=True, help="JSON file for the results")
    return p


def load(args):
    """-> samples, kw, sb (row-aligned with samples), terms, term vectors, folds."""
    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)])
    kw = kw[[kw_row[s] for s in samples["sample_id"]]].astype(np.float32)
    sb = sb[[sb_row[s] for s in samples["sample_id"]]].astype(np.float32)
    terms = load_terms(args.ontology_terms)
    term_vectors = load_term_vectors(args.term_vectors, list(terms["text"])).astype(np.float32)
    folds = list(study_folds(samples["study_code"], 5, args.fold_seed))
    print(f"{len(samples)} samples, {samples['study_code'].nunique()} studies, fold_seed {args.fold_seed}", flush=True)
    return samples, kw, sb, terms, term_vectors, folds


def inner_folds(study_codes, idx, n=3):
    """Folds *inside* a training fold (for nested tuning): same procedure, seed 1."""
    return [(idx[a], idx[b]) for a, b in study_folds(np.asarray(study_codes)[idx], n, 1)]


def save_json(results, output):
    import json
    os.makedirs(os.path.dirname(path(output)) or ".", exist_ok=True)
    with open(path(output), "w") as handle:
        json.dump(results, handle, indent=1)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)
