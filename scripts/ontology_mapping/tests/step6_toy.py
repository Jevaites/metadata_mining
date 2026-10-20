"""
Helpers for the tests of 6_predict_atlas.py (test_6_predict_atlas_unit.py, test_6_predict_atlas_integration.py).

Run (from the repository root or from scripts/ontology_mapping):
    python -m pytest scripts/ontology_mapping/tests/test_6_predict_atlas_unit.py \
                     scripts/ontology_mapping/tests/test_6_predict_atlas_integration.py \
                     scripts/ontology_mapping/tests/test_6_predict_atlas_mutations.py -v

1. `predict`, a fixture: the 6_predict_atlas module (a file name starting with a digit cannot be imported
   with a plain `import`). The script imports `iter_samples` from ../embeddings/embed_subbiomes_keywords.py,
   which imports `openai` at module level; nothing here calls the API, so when the package is not
   installed a stub module is registered instead. With the environment variable PREDICT_ATLAS_FILE set,
   that file is loaded instead of 6_predict_atlas.py (used by test_6_predict_atlas_mutations.py).
2. Writers for tiny versions of every input file of 6_predict_atlas.py, in the exact on-disk formats
   of the real pipeline, and run_predict(), which runs its command line in-process:

  input (CLI option)           written by (real pipeline)          format
  --ontology_terms             1_build_term_index.py               TSV(.gz): ontology, term_id, label, synonyms (||), definition,
                                                                       parents (||), obsolete
  --samples                    2_build_training_set.py / 2b        TSV(.gz): sample_id, spire_sample_name, study_code, domain,
                                                                       biome, feature, material, ... (SAMPLE_COLUMNS)
  --train_vectors (kw, sb)     3_extract_sample_embeddings.py      .npz: sample_ids, vectors (unique rows), index (row per sample)
  --keywords_texts             GPT extraction                      text: sample_id<TAB>{kw one, kw two} (some lines
                                                                       'kw one,kw two' without braces; 'N/A' = none)
  --sub_biomes_texts           GPT extraction                      text: sample_id<TAB>sub-biome text ('NA' = none)
  --keywords_h5 / _sub_biomes_h5  embed_subbiomes_keywords.py      .h5: texts (variable-length bytes), embeddings (float32,
                                                                       one row per distinct cleaned text)
  --term_vectors               4_embed_terms.py                    .h5: texts = 'label; synonym; ...', embeddings
  --calibration                5_evaluate.py                       JSON: {slot: {method: {temperature, curve, settings}}}
                                                                       (the real file also has p_top1_quantiles: unused here)
"""
import gzip
import json
import os
import sys

import importlib.util
import types

import h5py
import numpy as np
import pandas as pd
import pytest

STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ontology_mapping
if STEP_DIR not in sys.path:
    sys.path.insert(0, STEP_DIR)  # the script imports `common`, `hierarchy`, `methods` from its folder
try:
    import openai  # noqa: F401  (or the stub another test module already registered)
except ImportError:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=None)


# test_6_predict_atlas_mutations.py points this at a deliberately broken copy of the script
SCRIPT = os.environ.get("PREDICT_ATLAS_FILE", os.path.join(STEP_DIR, "6_predict_atlas.py"))


def _load_predict_atlas():
    if "predict_atlas" not in sys.modules:
        spec = importlib.util.spec_from_file_location("predict_atlas", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["predict_atlas"] = module
        spec.loader.exec_module(module)
    return sys.modules["predict_atlas"]


@pytest.fixture
def predict():
    """The 6_predict_atlas module (train_models, build_index, knn_study_top2, ..., main)."""
    return _load_predict_atlas()


def write_terms(p, terms):
    """terms: [(term_id, label, [synonyms], [parents])]."""
    pd.DataFrame([{"ontology": t.split("_")[0], "term_id": t, "label": label, "synonyms": "||".join(syn),
                   "definition": "", "parents": "||".join(parents), "obsolete": "False"}
                  for t, label, syn, parents in terms]).to_csv(p, sep="\t", index=False, compression="gzip")
    return p


# the columns of metalog/clean/training_set.clean.tsv.gz (2b_clean_metalog.py), in order; the script reads
# sample_id, study_code and the three slots, the others are left empty here
SAMPLE_COLUMNS = ["sample_id", "spire_sample_name", "study_code", "domain", "biome", "feature", "material",
                  "biome_raw", "feature_raw", "material_raw", "artificial", "artificial_bucket", "audit_hits",
                  "audit_unreviewed", "in_gold_eval", "dup_text_in_study", "text"]


def write_samples(p, samples):
    """samples: [(sample_id, study_code, biome, feature, material)], "" = no label in that slot."""
    table = pd.DataFrame(samples, columns=["sample_id", "study_code", "biome", "feature", "material"])
    table.reindex(columns=SAMPLE_COLUMNS, fill_value="").to_csv(p, sep="\t", index=False, compression="gzip")
    return p


def write_sample_vectors(p, vectors):
    """vectors: {sample_id: vector}. As in the real files, each distinct vector is stored once (`vectors`)
    and `index` gives every sample's row (metalog/keywords__large1024.npz: 50,100 samples, 22,618 rows)."""
    ids = list(vectors)
    distinct, index = np.unique(np.array([vectors[s] for s in ids], dtype=np.float32), axis=0, return_inverse=True)
    np.savez(p, sample_ids=np.array(ids), vectors=distinct, index=index.ravel())
    return p


def write_text_embeddings(p, vectors):
    """vectors: {text: vector}, one row per distinct text, in this order (= the .h5 row numbers)."""
    with h5py.File(p, "w") as handle:
        handle.create_dataset("texts", data=[t.encode() for t in vectors], dtype=h5py.string_dtype())  # as the real .h5
        handle["embeddings"] = np.array(list(vectors.values()), dtype=np.float32)
    return p


def write_texts(p, lines):
    """lines: [(sample_id, raw text)] -> 'sample_id<TAB>raw text' per line (raw keywords look like '{a, b}')."""
    with open(p, "w", encoding="utf-8") as handle:
        handle.writelines(f"{sid}\t{raw}\n" for sid, raw in lines)
    return p


def write_calibration(p, method, temperature, curve, settings, slots=("biome", "feature", "material")):
    """The part of 5_evaluate.py's calibration.json that 6_predict_atlas.py reads, the same for every slot.
    curve: [(tau, accuracy)] -> one row per tau as 5_evaluate.py writes it; to keep it simple every
    sample is answered (coverage 1) and an answer is either exact or on another branch."""
    rows = [{"tau": tau, "coverage": 1.0, "accuracy_strict": acc, "accuracy_lenient": acc,
             "exact": acc, "coarser": 0.0, "too_specific": 0.0, "other_branch": round(1 - acc, 4)}
            for tau, acc in curve]
    entry = {"temperature": temperature, "curve": rows, "settings": {"method": method, **settings}}
    with open(p, "w") as handle:
        json.dump({slot: {method: entry} for slot in slots}, handle)
    return p


def run_predict(predict, *argv):
    """Run `python 6_predict_atlas.py <argv>` in-process (sys.argv swapped for the call)."""
    saved = sys.argv
    sys.argv = ["6_predict_atlas.py", *map(str, argv)]
    try:
        predict.main()
    finally:
        sys.argv = saved


def read_predictions(output_dir):
    """atlas_predictions.tsv.gz as a DataFrame of strings indexed by sample_id."""
    with gzip.open(os.path.join(output_dir, "atlas_predictions.tsv.gz"), "rt") as handle:
        return pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False).set_index("sample_id")
