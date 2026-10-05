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


def ancestor_sets(parents):
    """{term: set of all its is_a ancestors}, from {term: set of direct parents}."""
    result = {}
    def up(term):
        if term not in result:
            result[term] = set()  # guards against cycles
            result[term] = set().union(*[{p} | up(p) for p in parents.get(term, ())])
        return result[term]
    for term in parents:
        up(term)
    return result


# ----------------------------------------------------------------------------- samples
def load_npz(p):
    """Per-sample vectors written by 3_extract_sample_embeddings.py.
    Returns ({sample_id: row}, L2-normalised matrix with one row per sample)."""
    saved = np.load(path(p))
    return {s: i for i, s in enumerate(saved["sample_ids"])}, normalize(saved["vectors"])[saved["index"]]


def stable_key(seed, values):
    """A seeded pseudo-random key per value that depends on that value only (md5 of 'seed:value'):
    ordering or splitting by it does not change for the other values when values are added."""
    import hashlib
    return np.array([hashlib.md5(f"{seed}:{v}".encode()).hexdigest() for v in values])


def select_samples(samples_path, require_ids=(), max_per_study=50, seed=22):
    """The labelled samples used for training and evaluation.

    1. keep samples with at least one slot label;
    2. keep samples whose id is in every set of `require_ids` (e.g. those with an embedding);
    3. keep at most `max_per_study` samples per study (0 = no cap), so a few huge cohorts do not
       dominate training or the scores: the samples with the smallest stable_key(seed, sample_id).
       The choice of a sample does not depend on the other rows, so adding labels to a training set
       keeps every previously selected sample, unless its study gains samples with smaller keys
       (until 2026-10-05 a seeded shuffle of the whole table was used, which redrew every study).
    """
    samples = read_tsv(samples_path)
    samples = samples[samples[SLOTS].ne("").any(axis=1)]
    for ids in require_ids:
        samples = samples[samples["sample_id"].isin(ids)]
    samples = samples.iloc[np.argsort(stable_key(seed, samples["sample_id"]), kind="stable")]
    if max_per_study:
        samples = samples.groupby("study_code").head(max_per_study)
    return samples.reset_index(drop=True)


def study_folds(study_codes, n_folds=5, seed=0):
    """Cross-validation split by study: yields (train_idx, test_idx), no study on both sides.

    Each study goes to fold stable_key(seed, study) mod n_folds, so a study keeps its fold when other
    studies grow, shrink or are added, and the folds are identical on every machine. Folds are not
    size-balanced (with <= 50 samples per study they differ by up to ~20 %). Until 2026-10-05 studies
    were balanced greedily as in GroupKFold, so one changed study could move many others.
    Change `seed` for a different, equally valid split.
    """
    codes = np.asarray(study_codes)
    studies, inverse = np.unique(codes, return_inverse=True)
    fold = np.array([int(k, 16) % n_folds for k in stable_key(seed, studies)])[inverse]
    for f in range(n_folds):
        yield np.where(fold != f)[0], np.where(fold == f)[0]
