"""Integration tests for methods.py: the methods chained the way 5_evaluate.py chains them, on a toy
"ontology" and toy "samples" that a human can check by eye.

The toy world (all vectors are unit vectors in 2-D, so cosine similarity = closeness of angles)
---------------------------------------------------------------------------------------------
Ontology terms, one every 45 degrees. Only four of them are ever used as training labels (*):

                         90  sea water*
            135 lake water     |      45 sediment
                               |
            180 feces* --------+-------- 0 soil*
                               |
            225 skin           |      315 rock
                         270  saliva*

Training samples: three per used label, at the label's angle and +/- 5 degrees, each label from its
own study (S1 soil, S2 sea water, S3 feces, S4 saliva). Test samples sit a few degrees off a term.

So the "right answer" for a test sample is the used term nearest to it, which you can read off the
picture. Because both the terms and the samples are symmetric around the origin, their means are
zero and the centring inside prototype_model changes nothing.

`run_fold` below chains the methods.py functions the way 5_evaluate.evaluate_fold does for one slot
(without its bookkeeping: slots, records, majority baseline, --closed_only, back-off inputs). It is kept
as a minimal copy rather than calling evaluate_fold because the mutation harness runs these tests on a
copy of methods.py alone, and because the tests need the raw score matrices and per-method knobs.
evaluate_fold's own wiring is tested with the real function in test_5_evaluate_unit.py, and
test_methods_mutations.py::test_run_fold_matches_5_evaluate fails if this copy drifts from it.
"""

import numpy as np
import pytest
from scipy import sparse

import methods as M


def at(*degrees):
    """Unit vectors in 2-D at the given angles (degrees)."""
    rad = np.radians(degrees)
    return np.column_stack([np.cos(rad), np.sin(rad)])


# ----------------------------------------------------------------------------- the toy world
TERM_IDS = np.array(["soil", "sediment", "sea water", "lake water", "feces", "skin", "saliva", "rock"])
TERM_MATRIX = at(0, 45, 90, 135, 180, 225, 270, 315)

TRAIN_X = at(-5, 0, 5, 85, 90, 95, 175, 180, 185, 265, 270, 275)
TRAIN_Y = np.array(["soil"] * 3 + ["sea water"] * 3 + ["feces"] * 3 + ["saliva"] * 3)
TRAIN_STUDIES = np.array(["S1"] * 3 + ["S2"] * 3 + ["S3"] * 3 + ["S4"] * 3)


# ----------------------------------------------------------------------------- the workflow
def run_fold(train_x, train_y, train_studies, test_x, term_ids=TERM_IDS, term_matrix=TERM_MATRIX, *,
             k=3, knn_study_k=3, hybrid_weight=2.0, alpha=0.5, beta=0.1, unseen_bonus=0.0):
    """Every method on one fold of one slot, called as in 5_evaluate.evaluate_fold.

    -> (predictions: {method: (top-5 lists, confidences)} for the 11 methods of methods.py,
        backoff_scores: the full prototype score matrix (test samples x closed vocabulary), which
                        5_evaluate.py hands to hierarchy.py for the hierarchical back-off).

    Defaults differ from 5_evaluate.py where the toy is too small: k = 25 / knn_study_k = 50 would make
    every one of the 12 training samples a neighbour.
    """
    n = test_x.shape[0]
    # The "closed" vocabulary: the terms that occur as training labels in this fold.
    closed = np.isin(term_ids, train_y)

    # The linear classifier is fitted once; its scores are reused by `linear` and by `hybrid`.
    classes, linear = M.linear_scores(test_x, train_x, train_y)

    predictions = {
        "knn": M.knn(test_x, train_x, train_y, k),
        "knn_study": M.knn_study(test_x, train_x, train_y, train_studies, knn_study_k),
        "linear": M.rank(linear, classes),
    }

    # Cosine(sample, term) in row blocks (with 49k real terms the full matrix does not fit in memory).
    # Each method ranks each block on its own; join_ranks glues the blocks back together.
    blocks = [(rows, M.dense(test_x[rows] @ term_matrix.T)) for rows in M.row_blocks(n)]
    # retrieval: nearest term. Confidence = best cosine (margin=False), because near-synonym terms
    # would make the margin meaningless.
    predictions["retrieval"] = M.join_ranks(
        [M.rank(cos[:, closed], term_ids[closed], margin=False) for _, cos in blocks])
    predictions["retrieval_open"] = M.join_ranks(
        [M.rank(cos, term_ids, margin=False) for _, cos in blocks])
    # hybrid: linear score + weight x cosine. In the open version unseen terms start at -1.
    predictions["hybrid"] = M.join_ranks(
        [M.hybrid(classes, linear[rows], cos[:, closed], term_ids[closed], hybrid_weight) for rows, cos in blocks])
    predictions["hybrid_open"] = M.join_ranks(
        [M.hybrid(classes, linear[rows], cos, term_ids, hybrid_weight) for rows, cos in blocks])

    backoff_scores = None
    if not sparse.issparse(train_x):  # prototypes and label regression need dense embeddings
        # retrieval_prior = prototype with alpha 0 (term vectors + label-frequency prior only)
        model = M.prototype_model(train_x, train_y, term_matrix[closed], term_ids[closed], 0, beta)
        predictions["retrieval_prior"] = M.prototype(test_x, model, term_ids[closed])

        keep = []  # prototype() appends its full score matrix here
        model = M.prototype_model(train_x, train_y, term_matrix[closed], term_ids[closed], alpha, beta)
        predictions["prototype"] = M.prototype(test_x, model, term_ids[closed], keep)
        backoff_scores = keep[0]

        model = M.prototype_model(train_x, train_y, term_matrix, term_ids, alpha, beta, unseen_bonus)
        predictions["prototype_open"] = M.prototype(test_x, model, term_ids)

        # label_regression needs, for every training sample, the vector of its gold term (evaluate_fold
        # uses the raw term vectors; with one feature block they are the term matrix)
        row_of = {t: i for i, t in enumerate(term_ids)}
        gold_vectors = term_matrix[[row_of[t] for t in train_y]]
        predictions["label_reg"] = M.label_regression(test_x, train_x, gold_vectors,
                                                      term_matrix[closed], term_ids[closed])
    return predictions, backoff_scores


def top1(predictions, method):
    return [str(ranked[0]) for ranked in predictions[method][0]]


# ============================================================================= 1. the basic workflow
# One clear test sample per used label, a few degrees off its term.
CLEAR_TEST_X = at(3, 88, 183, 268)
CLEAR_EXPECTED = ["soil", "sea water", "feces", "saliva"]

ALL_METHODS = ["knn", "knn_study", "linear", "retrieval", "retrieval_open", "hybrid", "hybrid_open",
               "retrieval_prior", "prototype", "prototype_open", "label_reg"]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_workflow_every_method_finds_the_obvious_label(method):
    predictions, _ = run_fold(TRAIN_X, TRAIN_Y, TRAIN_STUDIES, CLEAR_TEST_X)
    assert top1(predictions, method) == CLEAR_EXPECTED


@pytest.mark.parametrize("method", ["linear", "retrieval", "hybrid", "prototype", "label_reg"])
def test_workflow_confidence_is_lower_for_an_ambiguous_sample(method):
    # 3 deg sits on 'soil'; 40 deg is almost half-way between soil (0) and sea water (90).
    # Both are answered 'soil', but the ambiguous one gets the lower confidence: confidence ranks samples.
    predictions, _ = run_fold(TRAIN_X, TRAIN_Y, TRAIN_STUDIES, at(3, 40))
    assert top1(predictions, method) == ["soil", "soil"]
    clear, ambiguous = predictions[method][1]
    assert clear > ambiguous


# ============================================================================= 2. edge cases
def test_edge_one_study_floods_the_neighbour_list():
    # Why knn_study exists. Terms: soil at 0 deg, sediment at 45 deg; test sample at 25 deg.
    #   study BIG : 5 near-identical 'soil' samples at 17-21 deg (4 to 8 deg from the test sample)
    #   study B   : 1 'sediment' sample at 30.5 deg (5.5 deg away)
    #   study C   : 1 'sediment' sample at 34 deg (9 deg away)
    # With k = 7 all seven are neighbours.
    train_x = at(21, 20, 19, 18, 17, 30.5, 34)
    train_y = np.array(["soil"] * 5 + ["sediment"] * 2)
    studies = np.array(["BIG"] * 5 + ["B", "C"])
    test_x = at(25)

    # knn counts samples: soil ~ 5 x 0.99 beats sediment ~ 2 x 0.99 -> one study decides
    ranked, _ = M.knn(test_x, train_x, train_y, k=7)
    assert ranked == [["soil", "sediment"]]

    # knn_study counts studies: soil = 1 (BIG), sediment = 1 + 1 (B, C) -> two curators outvote one
    ranked, confidence = M.knn_study(test_x, train_x, train_y, studies, k=7)
    assert ranked == [["sediment", "soil"]]
    assert confidence == pytest.approx([2 / 3])


def test_edge_gold_label_never_seen_in_training():
    # A test sample at 120 deg. Its true term is 'lake water' (135 deg), which no training sample uses.
    # The nearest *used* term is 'sea water' (90 deg, 30 deg away); lake water is 15 deg away.
    def first(**kwargs):
        predictions, _ = run_fold(TRAIN_X, TRAIN_Y, TRAIN_STUDIES, at(120), **kwargs)
        return {method: top1(predictions, method)[0] for method in predictions}

    answers = first()
    # closed-vocabulary methods cannot answer an unseen term
    for method in ["knn", "knn_study", "linear", "retrieval", "hybrid", "retrieval_prior", "prototype", "label_reg"]:
        assert answers[method] == "sea water", method
    # plain nearest-term retrieval over all terms finds it
    assert answers["retrieval_open"] == "lake water"
    # hybrid_open: lake water starts at -1 (the classifier never saw it); -1 + 2 x cos 15 = 0.93 is
    # less than sea water's (small positive linear score) + 2 x cos 30 = 1.97
    assert answers["hybrid_open"] == "sea water"
    # prototype_open: the frequency prior favours seen labels (beta x log(3.5/16) = -0.15 against
    # beta x log(0.5/16) = -0.35): sea water cos 30 - 0.15 = 0.71 beats lake water cos 15 - 0.35 = 0.62
    assert answers["prototype_open"] == "sea water"
    # ... unless unseen terms get a bonus: lake water 0.62 + 0.2 = 0.82 now wins (the documented trade-off)
    assert first(unseen_bonus=0.2)["prototype_open"] == "lake water"


def test_edge_slot_with_a_single_training_label():
    # Every training sample is 'soil' (e.g. a slot that one study filled). Each method can only say
    # 'soil', even for a sample at 180 deg that looks like feces; margins carry no information.
    train_x = at(-5, 0, 5)
    train_y = np.array(["soil"] * 3)
    predictions, _ = run_fold(train_x, train_y, np.array(["S1"] * 3), at(3, 180))

    for method in ["knn", "knn_study", "linear", "retrieval", "hybrid", "retrieval_prior",
                   "prototype", "label_reg"]:
        assert top1(predictions, method) == ["soil", "soil"], method
    # linear_scores returns a constant column of ones for a single class -> margin 0 for everyone
    assert list(predictions["linear"][1]) == [0.0, 0.0]
    # open methods can still escape to other terms: the 180 deg sample is nearest to the feces term
    assert top1(predictions, "retrieval_open")[1] == "feces"


def test_edge_sparse_tfidf_features():
    # With --features tfidf the samples (and term names) are sparse word vectors. 5_evaluate.py then
    # runs knn, knn_study, linear, retrieval and hybrid, and skips prototype and label regression.
    #               soil  sea  water  feces  fresh
    def tfidf(rows):
        return sparse.csr_matrix(np.array(rows, dtype=float) / np.linalg.norm(rows, axis=1, keepdims=True))

    term_ids = np.array(["soil", "sea water", "feces", "fresh water"])
    term_matrix = tfidf([[1, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 1]])
    train_x = tfidf([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0],    # "soil", "soil"
                     [0, 1, 1, 0, 0], [0, 1, 0, 0, 0],    # "sea water", "sea"
                     [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])   # "feces", "feces"
    train_y = np.array(["soil", "soil", "sea water", "sea water", "feces", "feces"])
    studies = np.array(["S1", "S2", "S3", "S4", "S5", "S6"])
    test_x = tfidf([[1, 0, 0, 0, 0],    # "soil"
                    [0, 1, 1, 0, 0],    # "sea water"
                    [0, 0, 0, 1, 0],    # "feces"
                    [0, 0, 1, 0, 1]])   # "fresh water": shares only 'water' with sea water

    predictions, backoff_scores = run_fold(train_x, train_y, studies, test_x, term_ids, term_matrix,
                                           k=2, knn_study_k=2)
    assert backoff_scores is None and "prototype" not in predictions  # skipped for sparse features
    for method in ["knn", "knn_study", "linear", "retrieval", "hybrid"]:
        assert top1(predictions, method)[:3] == ["soil", "sea water", "feces"], method
        assert top1(predictions, method)[3] == "sea water", method  # closed vocabulary: best it can do
    assert top1(predictions, "retrieval_open") == ["soil", "sea water", "feces", "fresh water"]


def test_edge_empty_test_set():
    # Zero test samples: every method returns no predictions instead of raising. (linear_scores,
    # prototype and label_regression used to raise inside sklearn. evaluate_fold now skips a slot
    # without test samples before calling them, but the functions themselves accept 0 rows.)
    predictions, backoff_scores = run_fold(TRAIN_X, TRAIN_Y, TRAIN_STUDIES, np.zeros((0, 2)))
    for method, (ranked, confidence) in predictions.items():
        assert ranked == [] and len(confidence) == 0, method
    assert backoff_scores.shape == (0, 4)  # 0 samples x 4 closed terms


def test_edge_many_test_samples_cross_every_internal_block_boundary():
    # knn/knn_study work in chunks of 2,000 rows, prototype/row_blocks in blocks of 1,000. 2,001 copies
    # of the four clear test samples must give 2,001 copies of the same answers.
    test_x = np.tile(CLEAR_TEST_X, (2001, 1))  # 8,004 rows
    predictions, backoff_scores = run_fold(TRAIN_X, TRAIN_Y, TRAIN_STUDIES, test_x)
    for method in ALL_METHODS:
        assert top1(predictions, method) == CLEAR_EXPECTED * 2001, method
    assert backoff_scores.shape == (8004, 4)
