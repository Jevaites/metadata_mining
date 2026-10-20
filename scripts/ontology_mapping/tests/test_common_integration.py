"""Integration tests for common.py: the helpers chained the way the pipeline scripts chain them,
on a toy project small enough to check every expected value by eye.

THE TOY PROJECT (fake ids; the real files are much larger but have the same columns)

  Ontology (terms.tsv, as written by 1_build_term_index.py)          arrow = is_a
      BFO_0000040 (upper class, NOT in the table)
        └── ENVO_00000001 environmental material
              ├── ENVO_00000002 soil            synonyms: earth, dirt
              ├── ENVO_00000003 water           (no synonyms)
              │     ├── ENVO_00000004 sea water   synonyms: seawater
              │     └── ENVO_00000005 lake water  (no synonyms)
              ├── ENVO_00000006 fecal material  synonyms: feces
              └── ENVO_00000009 old soil        OBSOLETE

  Term embeddings (terms.h5, as written by 4_embed_terms.py): 2-D, one compass direction per term
                         N  sea water (0, 3)
          NW water (-1, 1)  |
     W  lake water (-1, 0) -+-  E  soil (2, 0)
                            |  SE environmental material (1, -1)
                         S  fecal material (0, -1)

  Labelled samples (training_set.tsv, as written by 2b_clean_metalog.py) and their
  sample embeddings (keywords.npz, as written by 3_extract_sample_embeddings.py)
      sample  study        material label   embedding      note
      S01     STUDY_SOIL   soil             (5, 1)     E   S01 and S02 have the same GPT text,
      S02     STUDY_SOIL   soil             (5, 1)     E   so they share one stored vector
      S03     STUDY_SOIL   soil             (1, -0.2)  E
      S04     STUDY_SEA    sea water        (0.2, 1)   N
      S05     STUDY_SEA    sea water        (-0.2, 1)  N
      S06     STUDY_LAKE   lake water       (-1, 0.2)  W
      S07     STUDY_LAKE   (no label)       (0, -1)    S   -> dropped by select_samples (no label)
      S08     STUDY_GUT    fecal material   (none)         -> dropped by select_samples (no embedding)

  Every sample points (within ~11 degrees) towards its own label's term, so the nearest term is
  the right answer for all of them.

THE WORKFLOW (steps refer to the numbered scripts)
  step 2   iter_sample_info + record_to_text   metadata record -> cleaned 'key: value; ...' text
  step 5/6 load_terms + load_term_vectors      term table -> term texts -> term embeddings
           load_npz + select_samples           which samples are used, and their vectors
           study_folds                         cross-validation split by study
  analyses ancestor_distances + coarsen_labels coarser labels for rare terms
"""

import gzip

import h5py
import numpy as np
import pandas as pd
import pytest

import common

E = {  # readable names for the toy term ids
    "environmental material": "ENVO_00000001", "soil": "ENVO_00000002", "water": "ENVO_00000003",
    "sea water": "ENVO_00000004", "lake water": "ENVO_00000005", "fecal material": "ENVO_00000006",
    "old soil": "ENVO_00000009",
}
NAME = {term_id: name for name, term_id in E.items()}

TERMS = pd.DataFrame([
    # ontology, term_id,                 label,                   synonyms,       definition, parents,                obsolete
    ("ENVO", E["environmental material"], "environmental material", "",             "", "BFO_0000040",           "False"),
    ("ENVO", E["soil"],                   "soil",                   "earth||dirt",  "", E["environmental material"], "False"),
    ("ENVO", E["water"],                  "water",                  "",             "", E["environmental material"], "False"),
    ("ENVO", E["sea water"],              "sea water",              "seawater",     "", E["water"],              "False"),
    ("ENVO", E["lake water"],             "lake water",             "",             "", E["water"],              "False"),
    ("ENVO", E["fecal material"],         "fecal material",         "feces",        "", E["environmental material"], "False"),
    ("ENVO", E["old soil"],               "old soil",               "",             "", E["environmental material"], "True"),
], columns=["ontology", "term_id", "label", "synonyms", "definition", "parents", "obsolete"])

TERM_VECTORS = {  # term text (common.term_text) -> embedding; note the trailing '; ' when there is no synonym
    "environmental material; ": (1, -1),
    "soil; earth; dirt": (2, 0),
    "water; ": (-1, 1),
    "sea water; seawater": (0, 3),
    "lake water; ": (-1, 0),
    "fecal material; feces": (0, -1),
}

SAMPLES = pd.DataFrame([
    # sample_id, study_code,   biome, feature, material
    ("S01", "STUDY_SOIL", "", "", E["soil"]),
    ("S02", "STUDY_SOIL", "", "", E["soil"]),
    ("S03", "STUDY_SOIL", "", "", E["soil"]),
    ("S04", "STUDY_SEA",  "", "", E["sea water"]),
    ("S05", "STUDY_SEA",  "", "", E["sea water"]),
    ("S06", "STUDY_LAKE", "", "", E["lake water"]),
    ("S07", "STUDY_LAKE", "", "", ""),
    ("S08", "STUDY_GUT",  "", "", E["fecal material"]),
], columns=["sample_id", "study_code", "biome", "feature", "material"])

SAMPLE_TEXT_VECTORS = {  # sample -> (GPT text, embedding); the .npz stores each distinct text once
    "S01": ("garden soil", (5, 1)),
    "S02": ("garden soil", (5, 1)),
    "S03": ("forest soil", (1, -0.2)),
    "S04": ("coastal seawater", (0.2, 1)),
    "S05": ("open ocean water", (-0.2, 1)),
    "S06": ("lake water", (-1, 0.2)),
    "S07": ("negative control", (0, -1)),
}

SAMPLE_INFO = """
>S01
sample_center_name=
sample_env_material=garden soil [ENVO:00000002]
sample_collection_date=2019-06-01
sample_lat_lon=47.37 N 8.54 E
experiments=SRX000001
study=SRP000001
study_STUDY_TITLE=Soil microbes of Zurich gardens
runs=SRR000001

>S04
sample_env_material=seawater
sample_depth=missing
sample_geo_loc_name=Mediterranean Sea
study_STUDY_TITLE=Marine plankton survey
study_STUDY_LINK=      

>S09
sample_env_material=old soil sample [ENVO:00000009]
sample_run_accession=SRR000009

"""  # layout of the real sample.info.gz: an empty first line and an empty line after every record


# ----------------------------------------------------------------------------- writers (mimic the pipeline outputs)
def write_terms(p, terms=TERMS):
    terms.to_csv(p, sep="\t", index=False)
    return str(p)


def write_term_h5(p, text_to_vector):
    with h5py.File(p, "w") as handle:
        handle.create_dataset("texts", data=list(text_to_vector), dtype=h5py.string_dtype("utf-8"))
        handle.create_dataset("embeddings", data=np.array(list(text_to_vector.values()), dtype=float))
    return str(p)


def write_npz(p, sample_text_vectors):
    """Layout of 3_extract_sample_embeddings.py: sample_ids, index, vectors (one row per distinct text)."""
    sample_ids = sorted(sample_text_vectors)
    texts = list(dict.fromkeys(sample_text_vectors[s][0] for s in sample_ids))
    vector_of_text = {text: vector for text, vector in sample_text_vectors.values()}
    np.savez(p, sample_ids=np.array(sample_ids), index=np.array([texts.index(sample_text_vectors[s][0]) for s in sample_ids]),
             vectors=np.array([vector_of_text[t] for t in texts], dtype=float))
    return str(p)


@pytest.fixture
def project(tmp_path):
    """Writes the toy project's files; returns their paths."""
    SAMPLES.to_csv(tmp_path / "training_set.tsv", sep="\t", index=False)
    with gzip.open(tmp_path / "sample.info.gz", "wt") as handle:
        handle.write(SAMPLE_INFO)
    return {
        "terms": write_terms(tmp_path / "terms.tsv"),
        "term_h5": write_term_h5(tmp_path / "terms.h5", TERM_VECTORS),
        "samples": str(tmp_path / "training_set.tsv"),
        "keywords": write_npz(tmp_path / "keywords.npz", SAMPLE_TEXT_VECTORS),
        "sample_info": str(tmp_path / "sample.info.gz"),
        "dir": tmp_path,
    }


# ============================================================================= the basic workflow
def test_step2_metadata_records_become_clean_texts(project):
    # code_to_label is built from ALL terms (read_tsv, not load_terms), as in 2_build_training_set.py:
    # obsolete codes still occur in metadata and should still be translated.
    all_terms = common.read_tsv(project["terms"])
    code_to_label = dict(zip(all_terms["term_id"], all_terms["label"]))

    texts = {sample_id: common.record_to_text(lines, code_to_label, max_chars=500)
             for sample_id, lines in common.iter_sample_info(project["sample_info"])}

    assert texts == {
        # code -> label; empty center name, date, coordinates and SRA accessions dropped; 'study_' prefix removed
        "S01": "env_material: garden soil [soil]; study_title: Soil microbes of Zurich gardens",
        # 'missing' depth dropped
        "S04": "env_material: seawater; geo_loc_name: Mediterranean Sea; study_title: Marine plankton survey",
        # obsolete code translated; the run accession is dropped
        "S09": "env_material: old soil sample [old soil]",
    }


def test_term_table_to_term_vectors(project):
    terms = common.load_terms(project["terms"])                         # obsolete 'old soil' removed
    vectors = common.load_term_vectors(project["term_h5"], terms["text"])

    assert terms["text"].tolist() == ["environmental material; ", "soil; earth; dirt", "water; ",
                                      "sea water; seawater", "lake water; ", "fecal material; feces"]
    # row i of `vectors` belongs to terms row i, and every row has length 1
    np.testing.assert_allclose(vectors[1], [1, 0])                      # soil (2, 0) -> (1, 0)
    np.testing.assert_allclose(vectors[3], [0, 1])                      # sea water (0, 3) -> (0, 1)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)


def test_selected_samples_and_nearest_term(project):
    # 1. which samples are used: labelled AND with an embedding
    row_of, sample_vectors = common.load_npz(project["keywords"])
    samples = common.select_samples(project["samples"], require_ids=[set(row_of)])
    assert sorted(samples["sample_id"]) == ["S01", "S02", "S03", "S04", "S05", "S06"]  # not S07, not S08

    # 2. their vectors, in the order of `samples` (select_samples shuffles rows by a stable key)
    X = sample_vectors[[row_of[s] for s in samples["sample_id"]]]
    np.testing.assert_allclose(X[samples["sample_id"].tolist().index("S01")],
                               X[samples["sample_id"].tolist().index("S02")])  # shared text, same vector

    # 3. the simplest use of both vector sets: nearest term by cosine (rows are unit length: dot product)
    terms = common.load_terms(project["terms"])
    T = common.load_term_vectors(project["term_h5"], terms["text"])
    nearest = terms["term_id"].to_numpy()[(X @ T.T).argmax(axis=1)]
    predicted = {s: NAME[t] for s, t in zip(samples["sample_id"], nearest)}

    assert predicted == {"S01": "soil", "S02": "soil", "S03": "soil",
                         "S04": "sea water", "S05": "sea water", "S06": "lake water"}
    assert all(samples["material"].map(NAME) == samples["sample_id"].map(predicted))  # = the gold labels


def test_cross_validation_by_study(project):
    row_of, _ = common.load_npz(project["keywords"])
    samples = common.select_samples(project["samples"], require_ids=[set(row_of)])
    study, label = samples["study_code"].to_numpy(), samples["material"].to_numpy()

    folds = list(common.study_folds(study, n_folds=3))

    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested) == list(range(len(samples)))       # every sample is tested exactly once
    for train, test in folds:
        assert set(study[train]).isdisjoint(study[test])      # no study on both sides
        # consequence in this toy project (one material per study): the held-out labels were never
        # seen in training, so a model is scored on studies, and labels, it has not seen
        assert set(label[train]).isdisjoint(label[test])


def test_coarsen_rare_labels_to_ancestors(project):
    # labels of the selected samples: soil x3, sea water x2, lake water x1
    terms = common.load_terms(project["terms"])
    distances = common.ancestor_distances(terms)
    known = set(terms["term_id"])
    row_of, _ = common.load_npz(project["keywords"])
    labels = common.select_samples(project["samples"], require_ids=[set(row_of)])["material"]

    def coarsen(min_support):
        return {NAME[old]: NAME[new] for old, new in common.coarsen_labels(labels, distances, min_support, known).items()}

    # support (samples at or below):  soil 3, sea water 2, lake water 1, water 3, environmental material 6
    assert coarsen(3) == {"soil": "soil", "sea water": "water", "lake water": "water"}
    assert coarsen(6) == {"soil": "environmental material", "sea water": "environmental material",
                          "lake water": "environmental material"}
    # nothing reaches 7, and BFO_0000040 (not in the table) is never a label: all unchanged
    assert coarsen(7) == {"soil": "soil", "sea water": "sea water", "lake water": "lake water"}


def test_adding_a_study_keeps_previous_selection_and_folds(project):
    """Adding labels (a new study) must not reshuffle the samples or folds already in use."""
    row_of, _ = common.load_npz(project["keywords"])
    before = common.select_samples(project["samples"], require_ids=[set(row_of)])

    grown = pd.concat([SAMPLES, pd.DataFrame([("S20", "STUDY_NEW", "", "", E["soil"]),
                                              ("S21", "STUDY_NEW", "", "", E["soil"])], columns=SAMPLES.columns)])
    grown.to_csv(project["dir"] / "grown.tsv", sep="\t", index=False)
    after = common.select_samples(str(project["dir"] / "grown.tsv"), require_ids=[set(row_of) | {"S20", "S21"}])

    assert set(before["sample_id"]) <= set(after["sample_id"])
    assert set(after["sample_id"]) - set(before["sample_id"]) == {"S20", "S21"}

    def fold_of_study(samples):
        codes = samples["study_code"].to_numpy()
        return {codes[i]: f for f, (_, test) in enumerate(common.study_folds(codes, 3)) for i in test}
    fold_before, fold_after = fold_of_study(before), fold_of_study(after)
    assert {s: fold_after[s] for s in fold_before} == fold_before


# ============================================================================= edge cases
def test_sample_missing_from_one_of_two_feature_files_is_dropped(project):
    # 5_evaluate.py requires both the keyword and the sub-biome vector of a sample
    sub_biomes = {s: v for s, v in SAMPLE_TEXT_VECTORS.items() if s != "S03"}   # S03 has no sub-biome text
    kw_rows, _ = common.load_npz(project["keywords"])
    sb_rows, _ = common.load_npz(write_npz(project["dir"] / "sub_biomes.npz", sub_biomes))
    samples = common.select_samples(project["samples"], require_ids=[set(kw_rows), set(sb_rows)])
    assert sorted(samples["sample_id"]) == ["S01", "S02", "S04", "S05", "S06"]


def test_term_without_synonyms_must_be_embedded_with_the_trailing_separator(project):
    # an .h5 keyed 'water' instead of 'water; ' does not match common.term_text
    wrong = {("water" if t == "water; " else t): v for t, v in TERM_VECTORS.items()}
    terms = common.load_terms(project["terms"])
    with pytest.raises(SystemExit, match="'water; '"):
        common.load_term_vectors(write_term_h5(project["dir"] / "wrong.h5", wrong), terms["text"])


def test_term_table_changed_after_embedding(project):
    # a synonym added to soil after 4_embed_terms.py ran: its new text has no vector -> clear stop
    changed = TERMS.copy()
    changed.loc[changed["label"] == "soil", "synonyms"] = "earth||dirt||ground"
    terms = common.load_terms(write_terms(project["dir"] / "terms_v2.tsv", changed))
    with pytest.raises(SystemExit, match="1 term texts are not in .*soil; earth; dirt; ground.*re-run 4_embed_terms.py"):
        common.load_term_vectors(project["term_h5"], terms["text"])


def test_obsolete_code_kept_verbatim_if_code_to_label_comes_from_load_terms(project):
    # the pitfall that read_tsv avoids in test_step2_...: load_terms drops obsolete terms
    terms = common.load_terms(project["terms"])
    code_to_label = dict(zip(terms["term_id"], terms["label"]))
    lines = dict(common.iter_sample_info(project["sample_info"]))["S09"]
    assert common.record_to_text(lines, code_to_label, 500) == "env_material: old soil sample [ENVO:00000009]"


def test_no_sample_left(project):
    # every sample filtered out (e.g. a wrong --features file): an empty table, not an error
    samples = common.select_samples(project["samples"], require_ids=[{"not", "these"}])
    assert samples.empty
    assert list(common.study_folds(samples["study_code"], n_folds=3)) != []   # 3 empty folds are yielded
    assert all(len(train) == len(test) == 0 for train, test in common.study_folds(samples["study_code"], 3))
