"""
Helpers shared by the ontology-mapping scripts (imported, never run directly).

Keeping these in one place guarantees that every script reads the same terms, cleans metadata the
same way, and above all evaluates and trains on the same samples and folds.

Sections:
  files            path(), read_tsv(), file_md5()
  ontology terms   load_terms(), term_text(), load_term_vectors(), term_ancestors(), ancestor_distances(),
                   coarsen_labels()
  metadata records iter_sample_info(), record_to_text()   (MicrobeAtlas sample.info)
  samples, folds   load_npz(), select_samples(), study_folds()
"""

import gzip
import hashlib
import os
import re
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

SLOTS = ["biome", "feature", "material"]  # Metalog environment_biome / _feature / _material


# ----------------------------------------------------------------------------- files
def path(p):
    return os.path.expanduser(p)


def read_tsv(p):
    """Read a TSV as strings; empty cells stay "" (never NaN)."""
    return pd.read_csv(path(p), sep="\t", dtype=str, keep_default_na=False)


def file_md5(p):
    """md5 of a file's content: identifies an input in resume checks independently of where it is mounted
    (absolute paths change between sessions of the Cowork VM, which made resumed runs refuse to continue)."""
    digest = hashlib.md5()
    with open(path(p), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- ontology terms
def term_text(terms):
    """The text that represents a term: 'label; synonym; synonym'.

    Example: 'fecal material; droppings; frass; pellet'.
    Definitions are left out on purpose: they add generic words (country names, "area") that match
    noise in the metadata, and a sentence-long definition embeds far from a keyword list."""
    return terms["label"] + "; " + terms["synonyms"].str.replace("||", "; ", regex=False)


def load_terms(p):
    """Non-obsolete terms of the term table (1_build_term_index.py), with a `text` column."""
    terms = read_tsv(p)
    terms = terms[terms["obsolete"] != "True"].reset_index(drop=True)
    terms["text"] = term_text(terms)
    return terms


def load_term_vectors(p, texts):
    """L2-normalised term embeddings (4_embed_terms.py output), one row per text in `texts`.
    The .h5 is keyed by text, so the lookup is an exact string match; a missing text means the
    term table changed since the terms were embedded."""
    import h5py
    with h5py.File(path(p), "r") as handle:
        row_of = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        missing = [t for t in texts if t not in row_of]
        if missing:
            raise SystemExit(f"{len(missing)} term texts are not in {p} (e.g. {missing[0]!r}): re-run 4_embed_terms.py")
        vectors = handle["embeddings"][:]
    return normalize(vectors[[row_of[t] for t in texts]])


def ancestor_sets(parents):
    """{term: set of all its is_a ancestors}, from {term: set of direct parents}.

    Example: {"a": {"b"}, "b": {"c"}} -> {"a": {"b", "c"}, "b": {"c"}}.
    Ontologies are DAGs (several parents are common in ENVO), so a term's ancestors are the union
    over all its parents. Memoised recursion; a cycle (never seen in ENVO / Uberon) cannot loop."""
    result = {}

    def up(term):
        if term not in result:
            result[term] = set()  # placeholder: guards against cycles
            result[term] = set().union(*[{p} | up(p) for p in parents.get(term, ())])
        return result[term]
    for term in parents:
        up(term)
    return result


def term_ancestors(terms):
    """{term_id: set of all its is_a ancestors} for a term table (its `parents` column, '||'-joined).
    Ancestors can lie outside the table (e.g. BFO upper classes): they are kept as plain ids."""
    return ancestor_sets({t: set(p.split("||")) - {""} for t, p in zip(terms["term_id"], terms["parents"])})


def ancestor_distances(terms):
    """{term_id: {ancestor: number of is_a steps}} (breadth first, so the shortest path counts).

    Example: soil -> {environmental material: 1, ...}. Used to coarsen labels to the closest
    ancestor (coarsen_labels) and to classify errors as coarser / too specific (experiments)."""
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


def coarsen_labels(labels, distances, min_support, known):
    """A coarser label policy: every label -> its closest ancestor (or itself) with >= min_support
    samples labelled with it *or below it*. Only ancestors in `known` (the term table) are used, so
    a label never becomes an upper class outside ENVO / Uberon (e.g. a BFO root).

    labels: the labels to count (one per sample, "" = none). Returns {old label: new label}.
    Example (min_support 100): 'freshwater lake biome' (40 samples) -> 'freshwater biome' when the
    freshwater biome and its descendants together have >= 100 samples."""
    counts = pd.Series([t for t in labels if t]).value_counts()
    support = counts.copy()  # samples labelled with the term or any of its descendants
    for term, n in counts.items():
        for a in distances.get(term, {}):
            if a in known:
                support[a] = support.get(a, 0) + n
    mapping = {}
    for term in counts.index:
        if support.get(term, 0) >= min_support:
            mapping[term] = term
            continue
        up = sorted((d, a) for a, d in distances.get(term, {}).items() if a in known and support.get(a, 0) >= min_support)
        mapping[term] = up[0][1] if up else term  # closest qualifying ancestor; ties: alphabetical id
    return mapping


# ----------------------------------------------------------------------------- MicrobeAtlas metadata records
CODE_IN_TEXT = re.compile(r"\b(ENVO|UBERON)[:_](\d{7,8})\b")  # ENVO:00001998 or ENVO_00001998
MISSING = {"", "na", "n/a", "nan", "none", "null", "-", "missing", "unknown", "unspecified",
           "not applicable", "not collected", "not provided", "not available", "not determined"}
# keys that are identifiers, dates or coordinates: noise for text matching (and they leak study identity:
# before they were dropped, a whole infant-gut study was mapped through a shared 'first public' date)
DROP_KEYS = re.compile(r"^(experiment|run)|^study$|^sample name$|alias|xref|link|insdc|accession|center|broker"
                       r"|submitter|checklist|library|date|time|update|public|latitude|longitude|lat_lon"
                       r"|taxon_id|_id$| id$|subject|patient|participant|replicate")


def iter_sample_info(p):
    """Yield (sample_id, lines) for each '>SAMPLE' record of sample.info(.gz), streaming (the file is
    ~10 GB uncompressed). Lines are 'key=value' without the trailing newline, e.g.
    ('SRS123', ['sample_env_biome=human gut', 'study_STUDY_TITLE=...'])."""
    sample_id, lines = None, []
    with gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p) as handle:
        for line in handle:
            if line.startswith(">"):
                if sample_id:
                    yield sample_id, lines
                sample_id, lines = line[1:].strip(), []
            else:
                lines.append(line.rstrip("\n"))
    if sample_id:
        yield sample_id, lines


def record_to_text(lines, code_to_label, max_chars):
    """Clean one metadata record into 'key: value; key: value; ...' (the `text` of the training set and
    the reranker's input).

    Per line: drop the 'sample_' / 'study_' key prefix, drop identifier / date / coordinate keys
    (DROP_KEYS) and missing values (MISSING, compared with the whole value), replace ontology codes by
    their label; then drop repeated lines and truncate (sample fields come first, so truncation only
    cuts the end of the study abstract).
    Example: ['sample_env_material=feces [ENVO:00002003]', 'sample_collection_date=2014-05-01',
              'sample_host_age=missing']  ->  'env_material: feces [fecal material]'"""
    kept = []
    for line in lines:
        key, _, value = line.partition("=")
        key = re.sub(r"^(sample|study)_", "", key.strip()).lower()
        value = value.strip()
        if not key or DROP_KEYS.search(key) or value.lower().strip(" .") in MISSING:
            continue
        value = CODE_IN_TEXT.sub(lambda m: code_to_label.get(f"{m[1]}_{m[2]}", m[0]), value)
        kept.append(f"{key}: {value}")
    return "; ".join(dict.fromkeys(kept))[:max_chars]  # dict.fromkeys drops duplicate lines, keeps order


# ----------------------------------------------------------------------------- samples, folds
def load_npz(p):
    """Per-sample vectors written by 3_extract_sample_embeddings.py.
    Returns ({sample_id: row}, L2-normalised matrix with one row per sample)."""
    saved = np.load(path(p))
    return {s: i for i, s in enumerate(saved["sample_ids"])}, normalize(saved["vectors"])[saved["index"]]


def stable_key(seed, values):
    """A seeded pseudo-random key per value that depends on that value only (md5 of 'seed:value'):
    ordering or splitting by it does not change for the other values when values are added.
    Example: stable_key(22, ["SRS1"]) -> array(['<32 hex digits>'])."""
    return np.array([hashlib.md5(f"{seed}:{v}".encode()).hexdigest() for v in values])


def select_samples(samples_path, require_ids=(), max_per_study=50, seed=22):
    """The labelled samples used for training and evaluation (5_evaluate, 6_predict_atlas, 6b, 7 and the
    experiments all call this, so they see exactly the same rows).

    1. keep samples with at least one slot label;
    2. keep samples whose id is in every set of `require_ids` (e.g. those with an embedding);
    3. keep at most `max_per_study` samples per study (0 = no cap), so a few huge cohorts (one infant-gut
       study has 1,679 samples) do not dominate training or the scores: the samples with the smallest
       stable_key(seed, sample_id). A sample's selection does not depend on the other rows, so adding
       labels to a training set keeps every previously selected sample, unless its study gains samples
       with smaller keys.
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
    """Cross-validation split by study: yields (train_idx, test_idx), no study on both sides (samples of
    one study share most of their text, so a random split would test on near-copies of training rows).

    Each study goes to fold int(stable_key(seed, study), 16) mod n_folds, so a study keeps its fold when
    other studies are added or change, and the folds are identical on every machine. Folds are not
    size-balanced (with <= 50 samples per study they differ by up to ~20 %). `seed` gives another,
    equally valid split (fold noise is ~1-3 points on biome, so compare methods on several seeds).
    """
    codes = np.asarray(study_codes)
    studies, inverse = np.unique(codes, return_inverse=True)
    fold = np.array([int(k, 16) % n_folds for k in stable_key(seed, studies)])[inverse]
    for f in range(n_folds):
        yield np.where(fold != f)[0], np.where(fold == f)[0]
