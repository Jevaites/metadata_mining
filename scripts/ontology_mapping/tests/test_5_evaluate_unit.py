"""
Unit tests for the functions of 5_evaluate.py, one section per function, in the order of the script.

Every test uses data small enough to check by eye: 2-4 dimensional vectors that point along one axis per
concept (axis 0 = "marine", axis 1 = "freshwater", ...), so "the nearest term" or "the right label" can be
read off the numbers without computing anything.

The mapping methods themselves (methods.py) and the calibration / back-off maths (hierarchy.py) are only
exercised through 5_evaluate.py here; what is tested is how 5_evaluate.py calls them and assembles their
results.
"""
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import normalize

from step_loader import load_step

ev = load_step("5_evaluate.py", "evaluate")



def unit(rows):
    """L2-normalised rows, as every feature block of 5_evaluate.py is."""
    return normalize(np.asarray(rows, dtype=float))


# =============================================================================================== stack
def test_stack_single_block_is_returned_unchanged():
    block = np.array([[1.0, 2.0]])
    assert ev.stack([block]) is block


def test_stack_dense_blocks_are_concatenated_column_wise():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])
    assert ev.stack([a, b]).tolist() == [[1, 2, 5], [3, 4, 6]]


def test_stack_is_sparse_as_soon_as_one_block_is_sparse():
    dense_block = np.array([[1, 2]])
    sparse_block = sparse.csr_matrix([[0, 3]])
    result = ev.stack([dense_block, sparse_block])
    assert sparse.isspmatrix_csr(result)
    assert result.toarray().tolist() == [[1, 2, 0, 3]]


# =============================================================================================== precision_threshold
def test_precision_threshold_docstring_example():
    # confidences already sorted from high to low; hits [1, 1, 1, 0, 1, 0]
    # running precision: 1/1, 2/2, 3/3, 3/4, 4/5, 4/6 -> only the first 3 reach 90 %
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    threshold, coverage = ev.precision_threshold(confidence, correct, target=0.9)
    assert threshold == 0.7  # keep samples with confidence >= 0.7 ...
    assert coverage == 0.5  # ... which are 3 of the 6 samples


def test_precision_threshold_takes_the_lowest_cutoff_not_the_first_dip():
    # precision dips to 1/2 after the 2nd sample but climbs back to 9/10 at the last one:
    # the function keeps the *largest* prefix that reaches the target, i.e. everything
    confidence = np.arange(10, 0, -1) / 10  # 1.0, 0.9, ..., 0.1
    correct = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    assert ev.precision_threshold(confidence, correct, target=0.9) == (0.1, 1.0)


def test_precision_threshold_order_of_input_does_not_matter():
    confidence = np.array([0.4, 0.9, 0.6, 0.8, 0.5, 0.7])  # the docstring example, shuffled
    correct = np.array([0, 1, 0, 1, 1, 1], dtype=bool)
    assert ev.precision_threshold(confidence, correct, target=0.9) == (0.7, 0.5)


def test_precision_threshold_none_when_the_target_is_never_reached():
    confidence = np.array([0.9, 0.8])
    correct = np.array([0, 1], dtype=bool)  # the most confident answer is wrong: precision 0, then 0.5
    assert ev.precision_threshold(confidence, correct, target=0.9) == (None, 0.0)


def test_precision_threshold_ties_keep_input_order():
    # all-equal confidences (e.g. the majority baseline, confidence 0 everywhere): the stable sort keeps
    # the input order, so the result is reproducible; here the first two are hits -> 2 of 4 samples
    confidence = np.zeros(4)
    correct = np.array([1, 1, 0, 0], dtype=bool)
    assert ev.precision_threshold(confidence, correct, target=0.9) == (0.0, 0.5)


# =============================================================================================== score
# A toy ontology:            biome
#                           /     \
#                   aquatic        soil
#                   /     \
#              marine     freshwater
PARENTS = {"marine": {"aquatic"}, "freshwater": {"aquatic"}, "aquatic": {"biome"}, "soil": {"biome"}}
ANCESTORS = {"marine": {"aquatic", "biome"}, "freshwater": {"aquatic", "biome"}, "aquatic": {"biome"},
             "soil": {"biome"}}
# term vectors: axis 0 = marine, 1 = freshwater, 2 = soil; aquatic lies between marine and freshwater,
# biome between all three
TERM_VECTOR_OF = {"marine": np.array([1.0, 0, 0]), "freshwater": np.array([0, 1.0, 0]),
                  "soil": np.array([0, 0, 1.0]), "aquatic": np.array([1.0, 1.0, 0]) / np.sqrt(2),
                  "biome": np.array([1.0, 1.0, 1.0]) / np.sqrt(3)}


def four_predictions():
    """Four predictions.tsv rows of one (slot, method), each a different kind of answer."""
    return pd.DataFrame([
        # gold          top-5 (top-1 first)        confidence  gold seen in training  domain
        ("marine",      ["marine", "freshwater"],  0.9,        True,                  "ocean"),          # exact
        ("marine",      ["biome", "marine"],       0.8,        True,                  "ocean"),          # coarser (grandparent)
        ("aquatic",     ["marine", "aquatic"],     0.3,        True,                  "environmental"),  # too specific (child)
        ("soil",        ["freshwater", "marine"],  0.1,        False,                 "environmental"),  # wrong, unseen label
    ], columns=["gold", "top5", "confidence", "gold_seen_in_train", "domain"])


def test_score_on_four_hand_made_predictions():
    m = ev.score(four_predictions(), PARENTS, TERM_VECTOR_OF, ANCESTORS)
    assert m["n"] == 4
    assert m["top1"] == 0.25                     # only row 1 is exact
    assert m["top5"] == 0.75                     # rows 1-3 have gold among their 5; row 4 does not
    assert m["top1_or_parent_child"] == 0.5      # rows 1 and 3: exact, child (the grandparent is 2 steps away)
    assert m["top1_gold_or_ancestor"] == 0.5     # rows 1 and 2: exact, any ancestor (the child does not count)
    assert m["macro_top1"] == round((1 / 2 + 0 + 0) / 3, 4)  # per gold label: marine 1/2, aquatic 0, soil 0
    assert m["top1_confident_half"] == 0.5       # the 2 most confident rows (0.9, 0.8): 1 hit of 2
    assert m["n_unseen_label"] == 1 and m["top1_unseen_label"] == 0.0  # row 4
    # sorted by confidence the hits are [1, 0, 0, 0]: 90 % precision only on the first row
    assert m["margin_for_90pct_precision"] == 0.9 and m["coverage_at_90pct_precision"] == 0.25
    # cosine(pred, gold): 1 (same), 1/sqrt(3) (biome vs marine), 1/sqrt(2) (marine vs aquatic), 0 (freshwater vs soil)
    assert m["pred_gold_cosine"] == round((1 + 1 / np.sqrt(3) + 1 / np.sqrt(2) + 0) / 4, 4)
    assert m["top1_near_synonym"] == 0.25        # only cosine 1 is >= 0.8
    assert m["top1_by_domain"] == {"environmental": 0.0, "ocean": 0.5}


def test_score_without_term_vectors_has_no_cosine_metrics():
    m = ev.score(four_predictions(), PARENTS, None, ANCESTORS)
    assert "pred_gold_cosine" not in m and "top1_near_synonym" not in m
    assert m["top1"] == 0.25  # the other metrics are unchanged


def test_score_unseen_metric_is_none_when_every_gold_label_was_seen():
    g = four_predictions().assign(gold_seen_in_train=True)
    m = ev.score(g, PARENTS, None, ANCESTORS)
    assert m["n_unseen_label"] == 0 and m["top1_unseen_label"] is None


def test_score_output_is_json_serialisable():
    # metrics.json is written with json.dump: no numpy types may remain
    json.dumps(ev.score(four_predictions(), PARENTS, TERM_VECTOR_OF, ANCESTORS))


def test_score_single_sample_has_no_confident_half():
    # with one sample the "50 % most confident" set is empty: None (as for an absent unseen label),
    # not NaN, which json.dump would write as the non-standard token NaN
    m = ev.score(four_predictions().head(1), PARENTS, None, ANCESTORS)
    assert m["top1"] == 1.0
    assert m["top1_confident_half"] is None
    json.dumps(m, allow_nan=False)  # strict JSON


# =============================================================================================== text_encoder
def test_tfidf_encoder_is_fitted_on_the_given_texts_only():
    fit_texts = ["sea water sample", "sea water ocean", "soil sample field", "soil field crop"]
    encode = ev.text_encoder("tfidf", fit_texts, args=None, cache_dir=None)
    X = encode(["sea water", "soil field", "desert sand"])
    cos = (X @ X.T).toarray()
    assert np.allclose(np.diag(cos)[:2], 1)  # rows are L2-normalised ...
    assert cos[0, 1] == 0                     # ... and texts without shared words are orthogonal
    # words never seen while fitting (and words in < 2 fitted texts: min_df=2) give an all-zero row:
    # such a sample has cosine 0 with everything
    assert X[2].nnz == 0


class FakeOpenAI:
    """Stands in for openai.OpenAI: the embedding of a text is [number of characters, 1] (before
    normalisation), and every call to embeddings.create is recorded."""
    calls = []

    def __init__(self, api_key, base_url, max_retries):
        self.embeddings = self

    def create(self, model, input, **extra):
        FakeOpenAI.calls.append({"model": model, "input": list(input), **extra})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input])


@pytest.fixture
def fake_openai(monkeypatch, tmp_path):
    """Replace the `openai` package by FakeOpenAI and return the args text_encoder needs."""
    FakeOpenAI.calls = []
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-test\n")
    return SimpleNamespace(api_key_path=str(key_file), base_url=None, dimensions=None)


def test_embedding_encoder_embeds_each_distinct_text_once(fake_openai, tmp_path):
    encode = ev.text_encoder("some/model", [], fake_openai, str(tmp_path))
    X = encode(["sea", "soil", "sea"])
    assert FakeOpenAI.calls == [{"model": "some/model", "input": ["sea", "soil"]}]  # "sea" sent once
    assert np.allclose(X[0], X[2])                       # ... and used for both rows
    assert np.allclose(X[0], unit([[3, 1]])[0])          # len("sea") = 3, normalised
    assert np.allclose(np.linalg.norm(X, axis=1), 1)
    assert (tmp_path / "embedding_cache__some-model__dimNone.npz").exists()  # '/' is replaced in the file name


def test_embedding_cache_is_reused_by_a_later_run(fake_openai, tmp_path):
    ev.text_encoder("m", [], fake_openai, str(tmp_path))(["sea", "soil"])
    FakeOpenAI.calls = []
    encode_again = ev.text_encoder("m", [], fake_openai, str(tmp_path))  # e.g. the next fold or run
    X = encode_again(["soil", "sand"])
    assert [c["input"] for c in FakeOpenAI.calls] == [["sand"]]  # only the new text is embedded
    assert np.allclose(X[0], unit([[4, 1]])[0])                  # "soil" comes from the cache


def test_embedding_encoder_batches_by_1000_and_passes_dimensions(fake_openai, tmp_path):
    fake_openai.dimensions = 256
    texts = [f"text {i}" for i in range(1001)]
    ev.text_encoder("m", [], fake_openai, str(tmp_path))(texts)
    assert [len(c["input"]) for c in FakeOpenAI.calls] == [1000, 1]
    assert all(c["dimensions"] == 256 for c in FakeOpenAI.calls)
    assert (tmp_path / "embedding_cache__m__dim256.npz").exists()


# =============================================================================================== build_features
# two .npz feature blocks for two samples, 2-d each, and two terms in the same 2-d space
#             keywords block   sub-biome block
#   sample s1   (1, 0)           (0, 1)
#   sample s2   (0, 1)           (0, 1)
#   term   T0   (1, 0)
#   term   T1   (0, 1)
NPZ = {"keywords.npz": ({"s1": 0, "s2": 1}, unit([[1, 0], [0, 1]])),
       "sub_biomes.npz": ({"s1": 0, "s2": 1}, unit([[0, 1], [0, 1]]))}
TERMS_2D = unit([[1, 0], [0, 1]])


def test_two_blocks_dot_product_is_the_mean_of_block_cosines():
    args = SimpleNamespace(features=["keywords.npz", "sub_biomes.npz"])
    encode_samples, term_matrix = ev.build_features(args, NPZ, [], ["T0", "T1"], TERMS_2D, cache_dir=None)
    X = encode_samples(pd.DataFrame({"sample_id": ["s1", "s2"]}))
    assert X.shape == (2, 4) and term_matrix.shape == (2, 4)  # blocks side by side
    # sample . term = (cos(keywords, term) + cos(sub_biome, term)) / 2
    #   s1 . T0 = (1 + 0) / 2    s1 . T1 = (0 + 1) / 2
    #   s2 . T0 = (0 + 0) / 2    s2 . T1 = (1 + 1) / 2
    assert np.allclose(X @ term_matrix.T, [[0.5, 0.5],
                                           [0.0, 1.0]])
    assert np.allclose(np.linalg.norm(X, axis=1), 1)  # still unit length


def test_samples_are_looked_up_by_id_not_by_position():
    args = SimpleNamespace(features=["keywords.npz"])
    encode_samples, _ = ev.build_features(args, NPZ, [], ["T0", "T1"], TERMS_2D, cache_dir=None)
    X = encode_samples(pd.DataFrame({"sample_id": ["s2", "s1"]}))
    assert np.allclose(X, [[0, 1], [1, 0]])  # one block: weight 1, vectors unchanged


def test_no_term_matrix_without_term_vectors():
    args = SimpleNamespace(features=["keywords.npz"])
    _, term_matrix = ev.build_features(args, NPZ, [], ["T0", "T1"], None, cache_dir=None)
    assert term_matrix is None  # 5_evaluate then skips every method that needs term vectors


def test_tfidf_block_terms_are_encoded_with_the_fold_vectorizer():
    args = SimpleNamespace(features=["tfidf"])
    train_texts = ["sea water", "sea water ocean", "soil field", "soil field crop"]
    term_texts = ["sea water", "soil field"]
    encode_samples, term_matrix = ev.build_features(args, {}, train_texts, term_texts, None, cache_dir=None)
    X = encode_samples(pd.DataFrame({"text": ["ocean sea water", "crop soil"]}))
    assert sparse.issparse(X) and sparse.issparse(term_matrix)  # tf-idf needs no --term_vectors
    assert (X @ term_matrix.T).toarray().argmax(1).tolist() == [0, 1]  # sea -> "sea water", soil -> "soil field"


def test_npz_and_tfidf_blocks_mix_into_one_sparse_matrix():
    args = SimpleNamespace(features=["keywords.npz", "tfidf"])
    encode_samples, term_matrix = ev.build_features(args, NPZ, ["sea water", "sea water"], ["T0", "T1"],
                                                    TERMS_2D, cache_dir=None)
    X = encode_samples(pd.DataFrame({"sample_id": ["s1"], "text": ["sea water"]}))
    assert sparse.issparse(X) and X.shape[1] == term_matrix.shape[1]
    assert np.isclose((X @ term_matrix.T).toarray()[0, 0], (1 + 0) / 2)  # keyword cos 1, tfidf("sea water", "T0") 0


# =============================================================================================== evaluate_fold
# Toy fold. Term vectors are the axes of a 4-d space:
#   marine = (1,0,0,0)   freshwater = (0,1,0,0)   soil = (0,0,1,0)   lake = (0,0,0,1)
# Training samples sit close to the axis of their label, two per label, one study per label.
# The three slots are separate problems; here they get the same labels so one picture explains all three.
TERM_IDS = np.array(["marine", "freshwater", "soil", "lake"])
TERM_VECTORS = np.eye(4)
TERM_ROW = {t: i for i, t in enumerate(TERM_IDS)}
TRAIN_X = unit([[1, .1, 0, 0], [1, 0, .1, 0],    # marine
                [.1, 1, 0, 0], [0, 1, .1, 0],    # freshwater
                [.1, 0, 1, 0], [0, .1, 1, 0]])   # soil
TRAIN_LABELS = ["marine", "marine", "freshwater", "freshwater", "soil", "soil"]
TEST_X = unit([[1, .05, .05, 0],                 # clearly marine
               [.05, 1, .05, 0],                 # clearly freshwater
               [.05, .05, 1, 0]])                # clearly soil
TEST_LABELS = ["marine", "freshwater", "soil"]
TEST_ROWS = [100, 101, 102]  # the DataFrame index of the test samples: records refer to samples by it

ALL_METHODS = {"majority", "knn", "knn_study", "linear", "retrieval", "hybrid", "retrieval_open", "hybrid_open",
               "retrieval_prior", "prototype", "prototype_open", "label_reg"}


def fold_frame(labels, studies, index=None, **slot_overrides):
    slots = {slot: list(labels) for slot in ev.SLOTS}
    slots.update(slot_overrides)
    return pd.DataFrame({**slots, "study_code": studies}, index=index)


TRAIN = fold_frame(TRAIN_LABELS, ["st_m", "st_m", "st_f", "st_f", "st_s", "st_s"])
TEST = fold_frame(TEST_LABELS, ["st_x", "st_y", "st_z"], index=TEST_ROWS)


def fold_args(**overrides):
    args = dict(k=2, knn_study_k=2, hybrid_weight=2.0, prototype_alpha=0.5, prototype_beta=0.1,
                prototype_unseen_bonus=0.0, closed_only=False, backoff_methods=["prototype", "linear"])
    return SimpleNamespace(**{**args, **overrides})


def run_fold(args=None, train=TRAIN, test=TEST, train_x=TRAIN_X, test_x=TEST_X, term_matrix=TERM_VECTORS,
             term_vectors=TERM_VECTORS):
    records, backoff_inputs = ev.evaluate_fold(args or fold_args(), train, test, train_x, test_x, term_matrix,
                                               TERM_IDS, term_vectors, TERM_ROW)
    return pd.DataFrame(records), backoff_inputs


def test_evaluate_fold_every_method_finds_the_obvious_label():
    records, _ = run_fold()
    assert set(records["method"]) == ALL_METHODS
    assert len(records) == 3 * 3 * len(ALL_METHODS)  # test samples x slots x methods
    records["top1"] = records["top5"].str[0]
    records["gold"] = records["row"].map(dict(zip(TEST_ROWS, TEST_LABELS)))
    learned = records[records["method"] != "majority"]
    assert (learned["top1"] == learned["gold"]).all(), learned[learned["top1"] != learned["gold"]]
    assert records["gold_seen_in_train"].all()


def test_evaluate_fold_majority_ranks_training_label_counts():
    records, _ = run_fold()
    majority = records[(records["method"] == "majority") & (records["slot"] == "biome")]
    # 2 training samples per label: equal counts keep their order of first appearance in training
    assert majority["top5"].tolist() == [["marine", "freshwater", "soil"]] * 3
    assert majority["confidence"].tolist() == [0, 0, 0]  # no confidence: every sample is a tie


def test_evaluate_fold_knn_confidence_is_the_share_of_the_vote():
    records, _ = run_fold()
    biome = records[records["slot"] == "biome"].set_index(["method", "row"])
    # k = 2: both neighbours of each test sample carry its label -> 100 % of the vote
    assert biome.loc[("knn", 100), "top5"] == ["marine"]
    assert biome.loc[("knn", 100), "confidence"] == pytest.approx(1.0)
    # knn_study: the 2 neighbours come from one study, which has one vote in total
    assert biome.loc[("knn_study", 100), "confidence"] == pytest.approx(1.0)


def test_evaluate_fold_closed_methods_only_rank_training_labels():
    records, _ = run_fold()
    lake_ranked = records["top5"].apply(lambda top5: "lake" in top5)
    # "lake" is a term of the table but no training label: only *_open methods may rank it
    assert set(records.loc[lake_ranked, "method"]) <= {"retrieval_open", "hybrid_open", "prototype_open"}


def test_evaluate_fold_backoff_inputs_are_closed_vocabulary_scores():
    _, backoff_inputs = run_fold()
    assert set(backoff_inputs) == {(slot, m) for slot in ev.SLOTS for m in ("linear", "prototype")}
    linear, proto = backoff_inputs[("biome", "linear")], backoff_inputs[("biome", "prototype")]
    assert linear["S"].shape == proto["S"].shape == (3, 3)  # 3 test samples x 3 training labels
    # each score matrix comes with its own column order: sorted classes (linear) vs term-table order (prototype)
    assert list(linear["vocab"]) == ["freshwater", "marine", "soil"]
    assert list(proto["vocab"]) == ["marine", "freshwater", "soil"]
    assert list(linear["gold"]) == TEST_LABELS and list(linear["rows"]) == TEST_ROWS
    # the best score of each row is its gold label
    assert [linear["vocab"][j] for j in linear["S"].argmax(1)] == TEST_LABELS
    assert [proto["vocab"][j] for j in proto["S"].argmax(1)] == TEST_LABELS


def test_evaluate_fold_backoff_inputs_only_for_requested_methods():
    _, backoff_inputs = run_fold(fold_args(backoff_methods=[]))
    assert backoff_inputs == {}


def test_evaluate_fold_closed_only_skips_open_vocabulary_methods():
    records, _ = run_fold(fold_args(closed_only=True))
    assert set(records["method"]) == ALL_METHODS - {"retrieval_open", "hybrid_open", "prototype_open"}


def test_evaluate_fold_without_term_vectors_runs_label_only_methods():
    records, backoff_inputs = run_fold(term_matrix=None, term_vectors=None)
    assert set(records["method"]) == {"majority", "knn", "knn_study", "linear"}
    assert set(backoff_inputs) == {(slot, "linear") for slot in ev.SLOTS}  # no prototype scores to calibrate


def test_evaluate_fold_sparse_features_skip_the_dense_only_methods():
    # TF-IDF features are sparse: prototype / retrieval_prior / label_reg need dense vectors and are skipped
    sparse_terms = sparse.csr_matrix(TERM_VECTORS)
    records, backoff_inputs = run_fold(train_x=sparse.csr_matrix(TRAIN_X), test_x=sparse.csr_matrix(TEST_X),
                                       term_matrix=sparse_terms, term_vectors=None)
    assert set(records["method"]) == {"majority", "knn", "knn_study", "linear",
                                      "retrieval", "hybrid", "retrieval_open", "hybrid_open"}
    assert set(backoff_inputs) == {(slot, "linear") for slot in ev.SLOTS}
    top1 = records[records["method"] == "linear"]["top5"].str[0]
    assert top1.tolist() == TEST_LABELS * 3  # LinearSVC on the sparse input: still right


def test_evaluate_fold_only_samples_labelled_in_a_slot_are_scored():
    # the freshwater test sample (row 101) has no feature label: it gets biome and material records only
    test = fold_frame(TEST_LABELS, ["st_x", "st_y", "st_z"], index=TEST_ROWS,
                      feature=["marine", "", "soil"])
    records, _ = run_fold(test=test)
    rows_per_slot = records[records["method"] == "knn"].groupby("slot")["row"].apply(list).to_dict()
    assert rows_per_slot == {"biome": [100, 101, 102], "feature": [100, 102], "material": [100, 101, 102]}


def test_evaluate_fold_unseen_gold_label():
    # a 4th test sample is a lake (axis 3, slightly freshwater); no training sample is a lake
    test = fold_frame(TEST_LABELS + ["lake"], ["st_x", "st_y", "st_z", "st_l"], index=TEST_ROWS + [103])
    test_x = np.vstack([TEST_X, unit([[0, .3, 0, 1]])])
    records, _ = run_fold(test=test, test_x=test_x)
    lake = records[(records["row"] == 103) & (records["slot"] == "biome")].set_index("method")
    assert not lake["gold_seen_in_train"].any()  # flagged: the gold label is unseen in this fold
    top1 = lake["top5"].str[0]
    assert top1["retrieval_open"] == "lake"  # open vocabulary: the nearest of all terms, seen or not
    assert top1["retrieval"] == "freshwater"  # closed vocabulary: the nearest *training label*
    assert top1["knn"] == "freshwater"  # its nearest training samples are the freshwater ones


def test_evaluate_fold_single_training_label():
    # one label in the training fold (every training sample is "soil" in the material slot):
    # every closed method can only answer "soil"
    train = fold_frame(TRAIN_LABELS, list(TRAIN["study_code"]), material=["soil"] * 6)
    records, _ = run_fold(train=train)
    material = records[records["slot"] == "material"]
    closed = material[~material["method"].str.endswith("_open")]
    assert closed["top5"].tolist() == [["soil"]] * len(closed)
    # material gold is still marine / freshwater / soil: only the soil sample's label was seen in training
    seen = material[material["method"] == "knn"].set_index("row")["gold_seen_in_train"]
    assert seen.to_dict() == {100: False, 101: False, 102: True}


def test_evaluate_fold_labels_must_be_terms_of_the_table():
    # Input contract: with term vectors, every training label must be a term of the term table
    # (2_build_training_set.py / 2b_clean_metalog.py drop obsolete or unknown labels). Otherwise: KeyError.
    train = fold_frame(TRAIN_LABELS[:-1] + ["glacier"], list(TRAIN["study_code"]))
    with pytest.raises(KeyError, match="glacier"):
        run_fold(train=train)
    # without term vectors labels are opaque strings, so this works
    records, _ = run_fold(train=train, term_matrix=None, term_vectors=None)
    assert len(records) > 0


# A slot can be empty on one side of a fold (e.g. a slot only a few studies label). The slot is then skipped in
# that fold: no records and no back-off input for it, the other slots are unaffected. (Before this was handled,
# each of the three cases below crashed: retrieval over an empty vocabulary, prototypes of 0 test rows, and the
# back-off of an empty score matrix.)
@pytest.mark.parametrize("term_vectors", [TERM_VECTORS, None], ids=["with_term_vectors", "without_term_vectors"])
def test_evaluate_fold_slot_without_any_label_is_skipped(term_vectors):
    # nobody (train or test) has a feature label
    train = fold_frame(TRAIN_LABELS, list(TRAIN["study_code"]), feature=[""] * 6)
    test = fold_frame(TEST_LABELS, ["st_x", "st_y", "st_z"], index=TEST_ROWS, feature=[""] * 3)
    records, backoff_inputs = run_fold(train=train, test=test, term_matrix=term_vectors, term_vectors=term_vectors)
    assert set(records["slot"]) == {"biome", "material"}
    assert {slot for slot, _ in backoff_inputs} == {"biome", "material"}


def test_evaluate_fold_slot_unlabelled_in_training_is_skipped_with_a_message(capsys):
    # the test samples have feature labels, but no training sample does: nothing to predict them with
    train = fold_frame(TRAIN_LABELS, list(TRAIN["study_code"]), feature=[""] * 6)
    records, backoff_inputs = run_fold(train=train)
    assert set(records["slot"]) == {"biome", "material"}
    assert ("feature", "linear") not in backoff_inputs
    assert "feature: no labelled training sample in this fold, 3 test samples not scored" in capsys.readouterr().out


def test_evaluate_fold_slot_unlabelled_in_test_is_skipped():
    # the training samples have feature labels, but no test sample does: nothing to score (no message: normal)
    test = fold_frame(TEST_LABELS, ["st_x", "st_y", "st_z"], index=TEST_ROWS, feature=[""] * 3)
    records, backoff_inputs = run_fold(test=test)
    assert set(records["slot"]) == {"biome", "material"}
    assert ("feature", "prototype") not in backoff_inputs


# =============================================================================================== backoff_all
# Hand-made out-of-fold scores in which the hierarchy matters.
# Ontology (only terms in VOCAB are labels, i.e. possible answers):
#                         biome
#                /                      \
#          aquatic biome             terrestrial (not a label)
#           /        \             /     |      |      \
#      marine    freshwater      soil  desert  forest  tundra
VOCAB = ["marine", "freshwater", "aquatic", "soil", "desert", "forest", "tundra"]
BACKOFF_ANCESTORS = {"marine": {"aquatic", "biome"}, "freshwater": {"aquatic", "biome"}, "aquatic": {"biome"},
                     **{t: {"terrestrial", "biome"} for t in ("soil", "desert", "forest", "tundra")},
                     "terrestrial": {"biome"}}


def scores(*best):
    """One row of scores over VOCAB: 1 for the named labels, 0 for the others."""
    return [1.0 if v in best else 0.0 for v in VOCAB]


def backoff_fold(first_row):
    """4 test samples of one fold: two the model gets right with certainty, and two on which it cannot
    tell marine from freshwater (equal scores; the top-1 is then marine, the first in VOCAB)."""
    return {"S": np.array([scores("soil"), scores("marine"), scores("marine", "freshwater"),
                           scores("marine", "freshwater")]),
            "vocab": VOCAB,
            "gold": ["soil", "marine", "marine", "freshwater"],
            "rows": list(range(first_row, first_row + 4))}


def backoff_args(**overrides):
    args = dict(backoff_metric="strict", backoff_target=1.0, backoff_targets=[0.75], features=["x.npz"],
                samples="samples.tsv", max_per_study=50, seed=22, fold_seed=0, fold_groups=None,
                prototype_alpha=0.5, prototype_beta=0.1)
    return SimpleNamespace(**{**args, **overrides})


def test_backoff_all_climbs_to_the_common_parent_when_unsure():
    folds = {("biome", "linear"): [backoff_fold(0), backoff_fold(4)]}
    reports, calibration, extra = ev.backoff_all(folds, BACKOFF_ANCESTORS, backoff_args(), n_samples=8)
    report = reports[("biome", "linear")]

    # The scores separate the gold label perfectly except for the ties, so the fitted temperature is tiny and
    # the probabilities are 1 / 0 for the certain rows and 0.5 / 0.5 for the marine-freshwater ties.
    assert extra[(0, "biome", "linear")] == {"prob": 1.0, "backoff": "soil", "backoff_q": 1.0}
    assert extra[(1, "biome", "linear")] == {"prob": 1.0, "backoff": "marine", "backoff_q": 1.0}
    # Target 100 % strict accuracy: answering "marine" on the ties (q = 0.5) is wrong for the freshwater one,
    # so tau must exceed 0.5; the first grid value above is 0.525. q(aquatic) = 0.5 + 0.5 = 1 >= tau.
    assert report["strict_1.0"]["tau_per_fold"] == [0.525, 0.525]
    assert extra[(2, "biome", "linear")] == {"prob": 0.5, "backoff": "aquatic", "backoff_q": 1.0}
    assert extra[(3, "biome", "linear")] == {"prob": 0.5, "backoff": "aquatic", "backoff_q": 1.0}
    assert report["strict_1.0"]["accuracy_strict"] == 1.0
    assert report["strict_1.0"]["exact"] == 0.5 and report["strict_1.0"]["coarser"] == 0.5

    # Target 75 %: the plain top-1 (lowest tau, 0.05) is already 3/4 right -> no back-off;
    # exact = 2 certain rows + the marine tie, per fold; the freshwater tie answered "marine" is another branch
    assert report["strict_0.75"]["tau_per_fold"] == [0.05, 0.05]
    assert report["strict_0.75"]["exact"] == 0.75 and report["strict_0.75"]["other_branch"] == 0.25


def test_backoff_all_calibration_entry_for_step_6():
    folds = {("biome", "linear"): [backoff_fold(0), backoff_fold(4)]}
    _, calibration, _ = ev.backoff_all(folds, BACKOFF_ANCESTORS, backoff_args(), n_samples=8)
    entry = calibration["biome"]["linear"]
    assert set(entry) == {"temperature", "curve", "p_top1_quantiles", "settings"}
    assert entry["temperature"] < 0.01  # near the lower bound: the scores are (almost) perfectly separating
    assert [row["tau"] for row in entry["curve"]][:3] == [0.05, 0.075, 0.1]
    # top-1 probabilities are 1, 1, 0.5, 0.5 in each fold -> the lower half is 0.5
    assert entry["p_top1_quantiles"] == {"0.1": 0.5, "0.2": 0.5, "0.3": 0.5, "0.4": 0.5, "0.5": 0.75}
    assert entry["settings"]["method"] == "linear" and entry["settings"]["n_samples"] == 8
    assert entry["settings"]["prototype_alpha"] == 0  # only reported for the prototype method
    json.dumps(calibration)


def test_backoff_all_needs_at_least_two_folds(capsys):
    # temperature and tau of a fold are fitted on the *other* folds: a slot scored in a single fold (labelled by
    # one study only, say) cannot be calibrated, so it is left out (its prob / backoff columns stay empty)
    folds = {("material", "linear"): [backoff_fold(0)], ("biome", "linear"): [backoff_fold(0), backoff_fold(4)]}
    reports, calibration, extra = ev.backoff_all(folds, BACKOFF_ANCESTORS, backoff_args(), n_samples=8)
    assert set(reports) == {("biome", "linear")}
    assert set(calibration) == {"biome"}
    assert {slot for _, slot, _ in extra} == {"biome"}
    assert "back-off skipped for material / linear" in capsys.readouterr().out


def test_backoff_all_with_nothing_to_calibrate():
    assert ev.backoff_all({}, BACKOFF_ANCESTORS, backoff_args(), n_samples=0) == ({}, {}, {})


# =============================================================================================== print_backoff
def test_print_backoff_prints_nothing_without_backoff(capsys):
    ev.print_backoff({}, {}, [0.9])
    assert capsys.readouterr().out == ""


def test_print_backoff_one_line_per_slot_and_method(capsys):
    folds = {("biome", "linear"): [backoff_fold(0), backoff_fold(4)]}
    reports, _, _ = ev.backoff_all(folds, BACKOFF_ANCESTORS, backoff_args(backoff_targets=[1.0]), n_samples=8)
    ev.print_backoff(reports, {"biome": {"linear": {"top1": 0.75}}}, [1.0])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.split()[:4] == ["biome", "linear", "top1", "0.750"]
    assert "@1.0: strict cov 1.00 exact 0.50" in line
