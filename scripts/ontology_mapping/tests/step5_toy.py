"""
Toy inputs for the tests of 5_evaluate.py (imported by test_5_evaluate_integration.py), small enough to
check every result by eye.

Real pipeline input                         toy version here
--------------------------------------      -------------------------------------------------------------
ontology_terms.tsv.gz (step 1, 49k terms)   15 terms (TERMS), with is_a parents
training_set.clean.tsv.gz (step 2b, 58k)    18 samples (SAMPLES): 3 classes x 2 studies x 3 samples, written
                                            gzipped with the 17 columns of the real file, in its order
<features>.npz (step 3, 1024-d vectors)     5-d vectors (SAMPLES); like the real .npz, samples with the same
                                            vector share one row of `vectors` through `index`
term embeddings .h5 (step 4)                5-d vectors (TERMS), same space as the samples

The 5 dimensions are 5 concepts:   axis 0 = sea   1 = freshwater   2 = soil   3 = lake   4 = gut
A term points along its concept's axis, and a sample lies close to the axis of its class, so "which term
is nearest" is obvious from the numbers. Term ids are readable fakes (ENVO_marine_biome), except the two
human terms, which carry their real ids (ENVO_2100002, ENVO_00002003).

Each class uses a different term in each slot (biome, feature, material); human samples have no biome
label, as in the real training set (no human sample has one there):

    class        biome                    feature                  material             domain
    marine       ENVO_marine_biome        ENVO_ocean               ENVO_sea_water       ocean
    freshwater   ENVO_freshwater_biome    ENVO_river               ENVO_fresh_water     environmental
    soil         ENVO_soil_biome          ENVO_field               ENVO_soil            environmental
    lake         ENVO_lake_biome          ENVO_river               ENVO_fresh_water     environmental
    human        ""                       ENVO_2100002 (intestine  ENVO_00002003 (fecal human
                                          environment)             material)
    (lake and human are only used by the tests that ask for LAKE_SAMPLES / HUMAN_SAMPLES)

Folds (2-fold cross-validation, --folds 2, fold seed 0). common.study_folds assigns studies by hashing
their name; with these names the split is (checked by test_toy_world_folds):

    fold 1 tests on   marine_study2,  freshwater_study1,  soil_study2,  lake_study1,  human_study2
    fold 2 tests on   marine_study1,  freshwater_study2,  soil_study1,  human_study1

so each fold tests 3 samples of every class while training on the 3 samples of the other study.
"""
import h5py
import numpy as np
import pandas as pd

from common import term_text

SEA, FRESH, SOIL, LAKE, GUT = np.eye(5)

TERMS = [
    # term_id                 label                 parent                  vector
    ("ENVO_biome",            "biome",              "",                     (SEA + FRESH + SOIL + LAKE) / 2),
    ("ENVO_aquatic_biome",    "aquatic biome",      "ENVO_biome",           (SEA + FRESH) / np.sqrt(2)),
    ("ENVO_marine_biome",     "marine biome",       "ENVO_aquatic_biome",   SEA),
    ("ENVO_freshwater_biome", "freshwater biome",   "ENVO_aquatic_biome",   FRESH),
    ("ENVO_lake_biome",       "lake biome",         "ENVO_freshwater_biome", LAKE),
    ("ENVO_soil_biome",       "soil biome",         "ENVO_biome",           SOIL),
    ("ENVO_ocean",            "ocean",              "",                     SEA),
    ("ENVO_river",            "river",              "",                     FRESH),
    ("ENVO_field",            "field",              "",                     SOIL),
    ("ENVO_sea_water",        "sea water",          "",                     SEA),
    ("ENVO_fresh_water",      "fresh water",        "",                     FRESH),
    ("ENVO_soil",             "soil",               "",                     SOIL),
    ("ENVO_old_soil",         "obsolete soil",      "",                     SOIL),  # obsolete: load_terms drops it
    ("ENVO_2100002",          "intestine environment", "",                  GUT),
    ("ENVO_00002003",         "fecal material",     "",                     GUT),
]
OBSOLETE = {"ENVO_old_soil"}

# class -> (biome, feature, material, domain, a few words of text for the tf-idf features)
CLASSES = {
    "marine":     ("ENVO_marine_biome",     "ENVO_ocean", "ENVO_sea_water",   "ocean",         "sea water ocean"),
    "freshwater": ("ENVO_freshwater_biome", "ENVO_river", "ENVO_fresh_water", "environmental", "river fresh water"),
    "soil":       ("ENVO_soil_biome",       "ENVO_field", "ENVO_soil",        "environmental", "field soil crop"),
    "lake":       ("ENVO_lake_biome",       "ENVO_river", "ENVO_fresh_water", "environmental", "lake fresh water"),
    "human":      ("",                      "ENVO_2100002", "ENVO_00002003",  "human",         "stool gut feces"),
}

SAMPLES = [
    # sample_id               study               class          vector: (sea, fresh, soil, lake, gut)
    ("marine_study1_s1",      "marine_study1",     "marine",      (1.0, 0.1, 0.0, 0.0, 0.0)),
    ("marine_study1_s2",      "marine_study1",     "marine",      (1.0, 0.0, 0.1, 0.0, 0.0)),
    ("marine_study1_s3",      "marine_study1",     "marine",      (1.0, 0.0, 0.0, 0.0, 0.0)),
    ("marine_study2_s1",      "marine_study2",     "marine",      (1.0, 0.1, 0.0, 0.0, 0.0)),
    ("marine_study2_s2",      "marine_study2",     "marine",      (1.0, 0.0, 0.1, 0.0, 0.0)),
    ("marine_study2_s3",      "marine_study2",     "marine",      (1.0, 0.0, 0.0, 0.0, 0.0)),
    ("freshwater_study1_s1",  "freshwater_study1", "freshwater",  (0.1, 1.0, 0.0, 0.0, 0.0)),
    ("freshwater_study1_s2",  "freshwater_study1", "freshwater",  (0.0, 1.0, 0.1, 0.0, 0.0)),
    ("freshwater_study1_s3",  "freshwater_study1", "freshwater",  (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("freshwater_study2_s1",  "freshwater_study2", "freshwater",  (0.1, 1.0, 0.0, 0.0, 0.0)),
    ("freshwater_study2_s2",  "freshwater_study2", "freshwater",  (0.0, 1.0, 0.1, 0.0, 0.0)),
    ("freshwater_study2_s3",  "freshwater_study2", "freshwater",  (0.0, 1.0, 0.0, 0.0, 0.0)),
    ("soil_study1_s1",        "soil_study1",       "soil",        (0.1, 0.0, 1.0, 0.0, 0.0)),
    ("soil_study1_s2",        "soil_study1",       "soil",        (0.0, 0.1, 1.0, 0.0, 0.0)),
    ("soil_study1_s3",        "soil_study1",       "soil",        (0.0, 0.0, 1.0, 0.0, 0.0)),
    ("soil_study2_s1",        "soil_study2",       "soil",        (0.1, 0.0, 1.0, 0.0, 0.0)),
    ("soil_study2_s2",        "soil_study2",       "soil",        (0.0, 0.1, 1.0, 0.0, 0.0)),
    ("soil_study2_s3",        "soil_study2",       "soil",        (0.0, 0.0, 1.0, 0.0, 0.0)),
]

# lake samples: mostly "lake", a little "freshwater"; their study is tested in fold 1 only
LAKE_SAMPLES = [
    ("lake_study1_s1",        "lake_study1",       "lake",        (0.0, 0.3, 0.0, 1.0, 0.0)),
    ("lake_study1_s2",        "lake_study1",       "lake",        (0.0, 0.3, 0.1, 1.0, 0.0)),
    ("lake_study1_s3",        "lake_study1",       "lake",        (0.1, 0.3, 0.0, 1.0, 0.0)),
]

# human samples: one study per fold, so each fold trains on the other's 3 human samples
HUMAN_SAMPLES = [
    ("human_study1_s1",       "human_study1",      "human",       (0.0, 0.0, 0.1, 0.0, 1.0)),
    ("human_study1_s2",       "human_study1",      "human",       (0.0, 0.0, 0.0, 0.0, 1.0)),
    ("human_study1_s3",       "human_study1",      "human",       (0.1, 0.0, 0.0, 0.0, 1.0)),
    ("human_study2_s1",       "human_study2",      "human",       (0.0, 0.0, 0.1, 0.0, 1.0)),
    ("human_study2_s2",       "human_study2",      "human",       (0.0, 0.0, 0.0, 0.0, 1.0)),
    ("human_study2_s3",       "human_study2",      "human",       (0.1, 0.0, 0.0, 0.0, 1.0)),
]

FOLD_TEST_STUDIES = {1: {"marine_study2", "freshwater_study1", "soil_study2", "lake_study1", "human_study2"},
                     2: {"marine_study1", "freshwater_study2", "soil_study1", "human_study1"}}


def write_terms(path):
    terms = pd.DataFrame([{"term_id": t, "label": label, "synonyms": "", "parents": parent,
                           "obsolete": str(t in OBSOLETE)} for t, label, parent, _ in TERMS])
    terms.to_csv(path, sep="\t", index=False)
    return terms


def write_term_vectors(path, terms):
    """The .h5 of 4_embed_terms.py: one vector per term text ('marine biome; '), the key load_term_vectors uses."""
    vectors = np.array([v for *_, v in TERMS], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("texts", data=np.array(list(term_text(terms)), dtype=h5py.string_dtype()))
        handle.create_dataset("embeddings", data=vectors)


def write_samples(path, samples, slots=("biome", "feature", "material")):
    """The training set of 2b_clean_metalog.py (training_set.clean.tsv.gz): its 17 columns in the real order,
    5_evaluate.py reads only sample_id, study_code, domain, the 3 slots and text. Slots not in `slots` are
    left unlabelled (""); `text` is '<class words> <sample_id>', so each text can be traced to its sample."""
    rows = []
    for i, (sample_id, study, cls, _) in enumerate(samples):
        biome, feature, material, domain, text = CLASSES[cls]
        labels = {slot: term if slot in slots else "" for slot, term in
                  zip(("biome", "feature", "material"), (biome, feature, material))}
        rows.append({"sample_id": sample_id, "spire_sample_name": f"SAMN{10351136 + i}", "study_code": study,
                     "domain": domain, **labels, **{f"{slot}_raw": term for slot, term in labels.items()},
                     "artificial": "", "artificial_bucket": "", "audit_hits": "", "audit_unreviewed": "False",
                     "in_gold_eval": "True", "dup_text_in_study": "False", "text": f"{text} {sample_id}"})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def write_npz(path, samples):
    """The .npz of 3_extract_sample_embeddings.py: the vector of sample_ids[i] is vectors[index[i]]; as in the
    real files, identical vectors (identical GPT texts) are stored once."""
    unique, index = np.unique(np.array([v for *_, v in samples], dtype=np.float32), axis=0, return_inverse=True)
    np.savez(path, sample_ids=np.array([s for s, *_ in samples]), index=index.ravel(), vectors=unique)


def write_project(tmp_path, samples=SAMPLES, slots=("biome", "feature", "material"), npz_samples=None):
    """Write all toy inputs into tmp_path; returns their paths. `npz_samples` (default: `samples`) lets a
    test leave samples without a vector. The keys are the plain names; the .tsv files are written as .tsv.gz."""
    paths = {name: str(tmp_path / (name + ".gz" if name.endswith(".tsv") else name)) for name in
             ("ontology_terms.tsv", "term_vectors.h5", "training_set.tsv", "features.npz")}  # tsv: gzipped
    terms = write_terms(paths["ontology_terms.tsv"])
    write_term_vectors(paths["term_vectors.h5"], terms)
    write_samples(paths["training_set.tsv"], samples, slots)
    write_npz(paths["features.npz"], samples if npz_samples is None else npz_samples)
    paths["output_dir"] = str(tmp_path / "out")
    return paths
