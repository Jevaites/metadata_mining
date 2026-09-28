"""
Helpers shared by the ontology-mapping scripts (imported, never run directly).

Keeping these in one place guarantees that every script sees the same terms,
the same term text, and above all the same evaluation samples.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

SLOTS = ["biome", "feature", "material"]  # Metalog environment_biome / _feature / _material


def path(p):
    return os.path.expanduser(p)


def read_tsv(p):
    """Read a TSV as strings; empty cells stay "" (never NaN)."""
    return pd.read_csv(path(p), sep="\t", dtype=str, keep_default_na=False)


# ----------------------------------------------------------------------------- ontology terms
def term_text(terms):
    """The text that represents a term: 'label; synonym; synonym' (definitions left out on purpose:
    they add generic words such as country names that match metadata noise)."""
    return terms["label"] + "; " + terms["synonyms"].str.replace("||", "; ", regex=False)


def load_terms(p):
    """Non-obsolete terms of the term table (1_build_term_index.py), with a `text` column."""
    terms = read_tsv(p)
    terms = terms[terms["obsolete"] != "True"].reset_index(drop=True)
    terms["text"] = term_text(terms)
    return terms


def load_term_vectors(p, texts):
    """L2-normalised term embeddings (4_embed_terms.py output), one row per text in `texts`."""
    import h5py
    with h5py.File(path(p), "r") as handle:
        row_of = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        missing = [t for t in texts if t not in row_of]
        if missing:
            raise SystemExit(f"{len(missing)} term texts are not in {p} (e.g. {missing[0]!r}): re-run 4_embed_terms.py")
        vectors = handle["embeddings"][:]
    return normalize(vectors[[row_of[t] for t in texts]])


# ----------------------------------------------------------------------------- samples
def load_npz(p):
    """Per-sample vectors written by 3_extract_sample_embeddings.py.
    Returns ({sample_id: row}, L2-normalised matrix with one row per sample)."""
    saved = np.load(path(p))
    return {s: i for i, s in enumerate(saved["sample_ids"])}, normalize(saved["vectors"])[saved["index"]]


def select_samples(samples_path, require_ids=(), max_per_study=50, seed=22):
    """The labelled samples used for training and evaluation.

    1. keep samples with at least one slot label;
    2. keep samples whose id is in every set of `require_ids` (e.g. those with an embedding);
    3. shuffle (seeded) and keep at most `max_per_study` samples per study (0 = no cap), so a few
       huge cohorts do not dominate training or the scores.
    """
    samples = read_tsv(samples_path)
    samples = samples[samples[SLOTS].ne("").any(axis=1)]
    for ids in require_ids:
        samples = samples[samples["sample_id"].isin(ids)]
    samples = samples.sample(frac=1, random_state=seed)
    if max_per_study:
        samples = samples.groupby("study_code").head(max_per_study)
    return samples.reset_index(drop=True)


def study_folds(study_codes, n_folds=5, seed=0):
    """Cross-validation split by study: yields (train_idx, test_idx), no study on both sides.

    Same balancing as sklearn's GroupKFold (largest study first, into the lightest fold), but ties
    between studies of equal size are broken by a seeded shuffle instead of an unstable argsort,
    whose order differs between numpy builds (many studies have exactly --max_per_study samples),
    so the folds are identical on every machine. Change `seed` for a different, equally valid split.
    """
    codes = np.asarray(study_codes)
    studies, sizes = np.unique(codes, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(studies))
    order = order[np.argsort(-sizes[order], kind="stable")]
    load, fold_of = np.zeros(n_folds), {}
    for i in order:
        fold_of[studies[i]] = int(np.argmin(load))
        load[fold_of[studies[i]]] += sizes[i]
    fold = np.array([fold_of[c] for c in codes])
    for f in range(n_folds):
        yield np.where(fold != f)[0], np.where(fold == f)[0]
