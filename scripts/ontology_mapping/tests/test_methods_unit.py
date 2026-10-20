"""Unit tests for methods.py: one small, hand-checkable example per behaviour.

Conventions used throughout
---------------------------
* Most toy data lives in 2-D. `at(0, 10, 90)` returns unit vectors at those angles (degrees), so for
  unit vectors "most similar" (largest dot product = cosine) simply means "smallest angle difference"
  and you can read the nearest neighbours off the numbers.
* Every expected value is written out literally in the test, with the arithmetic in a comment.
* Tests whose name contains `quirk` pin down current behaviour that callers must know about (an input
  contract or a surprising corner); they are not a statement that the behaviour is desirable.
* Tests whose name contains `regression` cover bugs fixed on 2026-10-19 (rank with n=1 or a list of ids,
  knn with negative similarities or exact ties, empty test sets). test_methods_mutations.py re-introduces
  each of those bugs and checks that these tests catch it.

See test_methods_integration.py for how the functions are chained in 5_evaluate.py.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

import methods as M


def at(*degrees):
    """Unit vectors in 2-D at the given angles (degrees): at(0) = [[1, 0]], at(90) = [[0, 1]]."""
    rad = np.radians(degrees)
    return np.column_stack([np.cos(rad), np.sin(rad)])


def cos(deg):
    return np.cos(np.radians(deg))


# =============================================================================================== dense
def test_dense_converts_sparse_and_leaves_arrays_alone():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert isinstance(M.dense(sparse.csr_matrix(a)), np.ndarray)
    assert_allclose(M.dense(sparse.csr_matrix(a)), a)
    assert_allclose(M.dense(a), a)
    assert_allclose(M.dense([[1, 2]]), [[1, 2]])  # lists become arrays


# =============================================================================================== rank
IDS = np.array(["a", "b", "c"])


def test_rank_docstring_example():
    # scores of a, b, c = 0.1, 0.9, 0.5 -> order b, c, a; margin = best - second = 0.9 - 0.5 = 0.4
    ranked, confidence = M.rank(np.array([[0.1, 0.9, 0.5]]), IDS)
    assert ranked == [["b", "c", "a"]]
    assert confidence == pytest.approx([0.4])


def test_rank_without_margin_returns_the_best_score():
    ranked, confidence = M.rank(np.array([[0.1, 0.9, 0.5]]), IDS, margin=False)
    assert ranked == [["b", "c", "a"]]
    assert confidence == pytest.approx([0.9])


def test_rank_ranks_each_row_independently():
    scores = np.array([[0.1, 0.9, 0.5],   # b > c > a
                       [0.7, 0.0, 0.2]])  # a > c > b
    ranked, confidence = M.rank(scores, IDS)
    assert ranked == [["b", "c", "a"], ["a", "c", "b"]]
    assert confidence == pytest.approx([0.4, 0.5])


def test_rank_n_limits_the_list_and_is_capped_by_the_vocabulary():
    scores = np.array([[0.1, 0.9, 0.5]])
    assert M.rank(scores, IDS, n=2)[0] == [["b", "c"]]
    assert M.rank(scores, IDS, n=5)[0] == [["b", "c", "a"]]  # only 3 terms exist


def test_rank_is_the_same_across_its_internal_1000_row_blocks():
    # 2,500 rows: row i has its best score in column i % 3. Rows 999/1000 and 1999/2000 straddle blocks.
    n = 2500
    scores = np.zeros((n, 3))
    scores[np.arange(n), np.arange(n) % 3] = 1.0
    ranked, confidence = M.rank(scores, IDS)
    assert [r[0] for r in ranked] == [IDS[i % 3] for i in range(n)]
    assert_allclose(confidence, np.ones(n))  # margin 1 - 0


def test_rank_empty_input():
    ranked, confidence = M.rank(np.zeros((0, 3)), IDS)
    assert ranked == [] and len(confidence) == 0


def test_rank_single_term_has_zero_margin():
    # with one term there is no second-best score: margin = best - best = 0
    ranked, confidence = M.rank(np.array([[0.7]]), np.array(["only"]))
    assert ranked == [["only"]]
    assert confidence == pytest.approx([0.0])


def test_rank_regression_accepts_a_plain_list_of_ids():
    # the docstring example, literally (ids used to have to be a numpy array: a list raised TypeError)
    ranked, confidence = M.rank(np.array([[0.1, 0.9, 0.5]]), ["a", "b", "c"])
    assert ranked == [["b", "c", "a"]]
    assert confidence == pytest.approx([0.4])


def test_rank_regression_margin_with_n_equal_1():
    # only the top-1 id is returned, but the margin still compares best (0.9) and second-best (0.5).
    # (It used to keep only one column and return best - best = 0 for every row.)
    ranked, confidence = M.rank(np.array([[0.1, 0.9, 0.5]]), IDS, n=1)
    assert ranked == [["b"]]
    assert confidence == pytest.approx([0.4])


# ====================================================================================== row_blocks / join_ranks
def test_row_blocks():
    assert M.row_blocks(2500) == [slice(0, 1000), slice(1000, 2000), slice(2000, 3000)]
    assert M.row_blocks(2000) == [slice(0, 1000), slice(1000, 2000)]
    assert M.row_blocks(3, size=2) == [slice(0, 2), slice(2, 4)]
    # zero rows still give one (empty) block, so callers always get at least one part to join
    assert M.row_blocks(0) == [slice(0, 1000)]


def test_join_ranks_concatenates_blocks_in_order():
    part1 = ([["a", "b"]], np.array([0.4]))
    part2 = ([["c", "a"], ["b", "c"]], np.array([0.1, 0.2]))
    ranked, confidence = M.join_ranks([part1, part2])
    assert ranked == [["a", "b"], ["c", "a"], ["b", "c"]]
    assert_allclose(confidence, [0.4, 0.1, 0.2])


# =============================================================================================== knn
# Ten training samples on the unit circle. Picture (angle: label):
#   0 soil, 10 soil, 20 soil | 80 water, 90 water, 100 water | 170 feces, 180 feces, 190 feces | 270 skin
KNN_TRAIN = at(0, 10, 20, 80, 90, 100, 170, 180, 190, 270)
KNN_LABELS = ["soil"] * 3 + ["water"] * 3 + ["feces"] * 3 + ["skin"]


def test_knn_ten_points_on_a_circle():
    # query at 12 deg: the 3 nearest are 10 (2 deg away), 20 (8 deg), 0 (12 deg) -> all soil
    # query at 187 deg: 190 (3), 180 (7), 170 (17)                                   -> all feces
    ranked, confidence = M.knn(at(12, 187), KNN_TRAIN, KNN_LABELS, k=3)
    assert ranked == [["soil"], ["feces"]]
    assert confidence == pytest.approx([1.0, 1.0])  # one label holds the whole vote


def test_knn_mixed_neighbourhood_is_a_similarity_weighted_vote():
    # query at 52 deg, k=4: 80 water (28 deg away), 20 soil (32), 90 water (38), 10 soil (42)
    # water = cos 28 + cos 38 = 1.671 > soil = cos 32 + cos 42 = 1.591
    ranked, confidence = M.knn(at(52), KNN_TRAIN, KNN_LABELS, k=4)
    water, soil = cos(28) + cos(38), cos(32) + cos(42)
    assert ranked == [["water", "soil"]]
    assert confidence == pytest.approx([water / (water + soil)])


def test_knn_docstring_example():
    # knn does not normalise: vectors of length 0.9, 0.8, 0.7 along x give similarities 0.9, 0.8, 0.7
    train = np.array([[0.9, 0], [0.8, 0], [0.7, 0]])
    ranked, confidence = M.knn(np.array([[1.0, 0]]), train, ["soil", "soil", "sediment"], k=3)
    assert ranked == [["soil", "sediment"]]
    assert confidence == pytest.approx([1.7 / 2.4])  # soil 0.9 + 0.8, total 2.4


def test_knn_k_larger_than_training_set_uses_every_sample():
    ranked, confidence = M.knn(at(0), at(0, 90), ["a", "b"], k=25)
    assert ranked == [["a", "b"]]
    assert confidence == pytest.approx([1.0 / (1.0 + 0.0)])  # cos 0 = 1, cos 90 = 0
    # a single training sample (a tiny slot) is the only neighbour; at 180 deg (cos -1) it gives no support
    ranked, confidence = M.knn(at(0), at(180), ["a"], k=25)
    assert ranked == [["a"]]
    assert confidence == pytest.approx([0.0])


def test_knn_returns_at_most_five_labels():
    train = at(0, 1, 2, 3, 4, 5, 6)  # seven different labels, nearest first
    ranked, _ = M.knn(at(0), train, list("abcdefg"), k=7)
    assert ranked == [["a", "b", "c", "d", "e"]]


def test_knn_sparse_input_gives_the_same_result_as_dense():
    test = at(12, 52)
    dense_result = M.knn(test, KNN_TRAIN, KNN_LABELS, k=4)
    sparse_result = M.knn(sparse.csr_matrix(test), sparse.csr_matrix(KNN_TRAIN), KNN_LABELS, k=4)
    assert sparse_result[0] == dense_result[0]
    assert_allclose(sparse_result[1], dense_result[1])


def test_knn_is_the_same_across_its_internal_2000_row_chunks():  # see methods.neighbours
    test = at(*np.linspace(0, 359, 2001))  # row 2000 falls into a second chunk
    ranked, confidence = M.knn(test, KNN_TRAIN, KNN_LABELS, k=3)
    for i in [0, 1999, 2000]:
        one_ranked, one_confidence = M.knn(test[i:i + 1], KNN_TRAIN, KNN_LABELS, k=3)
        assert ranked[i] == one_ranked[0]
        assert confidence[i] == pytest.approx(one_confidence[0])


def test_knn_empty_test_set():
    ranked, confidence = M.knn(np.zeros((0, 2)), KNN_TRAIN, KNN_LABELS, k=3)
    assert ranked == [] and len(confidence) == 0


def test_knn_regression_negative_similarities_vote_zero():
    # Similarities a = +1.0, b = -0.5. A negative similarity counts as 0 (no support), not as a vote
    # against: a = 1.0, b = 0 -> confidence 1.0 / 1.0. (Raw similarities used to be summed, giving
    # 1.0 / (1.0 - 0.5) = 2.0: not a share.)
    train = np.array([[1.0, 0.0], [-0.5, 0.0]])
    ranked, confidence = M.knn(np.array([[1.0, 0.0]]), train, ["a", "b"], k=2)
    assert ranked == [["a", "b"]]
    assert confidence == pytest.approx([1.0])


def test_knn_regression_only_negative_neighbours_give_confidence_0():
    # query at 0 deg, both neighbours at 120 and 150 deg (cosines -0.5 and -0.87): no support at all
    ranked, confidence = M.knn(at(0), at(120, 150), ["a", "b"], k=2)
    assert ranked == [["a", "b"]]  # equal (zero) votes keep the closest-first order
    assert confidence == pytest.approx([0.0])


def test_knn_regression_identical_training_vectors_are_ordered_by_training_index():
    # Same rule as knn_study (methods.neighbours): with identical vectors the lower index counts as closer.
    # (knn used to leave exact ties in argpartition's arbitrary order.)
    identical = at(0, 0)
    assert M.knn(at(0), identical, ["x", "y"], k=1)[0] == [["x"]]
    assert M.knn(at(0), identical, ["y", "x"], k=1)[0] == [["y"]]
    # k=2: equal votes (1 and 1) -> the 'closer' (lower index) label first
    assert M.knn(at(0), identical, ["x", "y"], k=2)[0] == [["x", "y"]]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_neighbours_ties_among_many_identical_vectors_follow_training_index(dtype):
    # Real features: 50,100 Metalog samples share only 22,618 distinct keyword vectors (911 sub-biome
    # vectors), so many neighbours tie exactly. Five identical float32/float64 vectors 'a'..'e', k=3:
    # the three lowest indices are the neighbours, in index order (equal votes keep closest-first order).
    # With only 2 tied vectors argpartition happens to return index order anyway; with 5 it does not.
    train = np.repeat(at(0), 5, axis=0).astype(dtype)
    labels = ["a", "b", "c", "d", "e"]
    assert M.knn(at(0).astype(dtype), train, labels, k=3)[0] == [["a", "b", "c"]]
    assert M.knn_study(at(0).astype(dtype), train, labels, list("ABCDE"), k=3)[0] == [["a", "b", "c"]]


# =============================================================================================== knn_study
def test_knn_study_docstring_example_tie_goes_to_the_closest_neighbour():
    # k=4, query at 0 deg. Study A has 3 'soil' samples, study B 1 'sediment' sample.
    # votes: soil = 3 x 1/3 = 1, sediment = 1 x 1/1 = 1 -> a tie between two curators.
    train_labels = ["sediment", "soil", "soil", "soil"]
    studies = np.array(["B", "A", "A", "A"])

    # sediment is the closest neighbour (1 deg) -> sediment wins the tie
    ranked, confidence = M.knn_study(at(0), at(1, 2, 3, 4), train_labels, studies, k=4)
    assert ranked == [["sediment", "soil"]]
    assert confidence == pytest.approx([0.5])  # 1 of 2 votes

    # move sediment to 5 deg: now a soil sample (2 deg) is closest -> soil wins the tie
    ranked, _ = M.knn_study(at(0), at(5, 2, 3, 4), train_labels, studies, k=4)
    assert ranked == [["soil", "sediment"]]


def test_knn_study_votes_are_not_weighted_by_similarity():
    # k=3, query at 0: 'a' from study A at 1 deg (very close); 'b' from studies B and C at 60 and 70 deg.
    # knn_study: a = 1, b = 1 + 1 = 2 -> b wins (two curators against one), confidence 2/3
    train, labels, studies = at(1, 60, 70), ["a", "b", "b"], np.array(["A", "B", "C"])
    ranked, confidence = M.knn_study(at(0), train, labels, studies, k=3)
    assert ranked == [["b", "a"]]
    assert confidence == pytest.approx([2 / 3])
    # knn on the same data: a = cos 1 = 1.0 beats b = cos 60 + cos 70 = 0.84
    assert M.knn(at(0), train, labels, k=3)[0] == [["a", "b"]]


def test_knn_study_a_study_splits_its_single_vote_between_its_labels():
    # k=3: study A has soil (1 deg) and sediment (2 deg) -> 0.5 each; study B has soil (3 deg) -> 1
    # soil = 0.5 + 1 = 1.5, sediment = 0.5; confidence = 1.5 / (number of studies = 2) = 0.75
    ranked, confidence = M.knn_study(at(0), at(1, 2, 3), ["soil", "sediment", "soil"],
                                     np.array(["A", "A", "B"]), k=3)
    assert ranked == [["soil", "sediment"]]
    assert confidence == pytest.approx([0.75])


def test_knn_study_one_study_fills_every_neighbour_slot():
    # A dominating study (Metalog has studies of 1,500+ samples; 5_evaluate caps them at --max_per_study 50,
    # the same as knn_study_k) can supply all k neighbours. Its single vote is then split by its labels:
    # k=4, all from study A: soil 3 x 1/4 = 0.75, sediment 0.25 -> confidence 0.75 / 1
    ranked, confidence = M.knn_study(at(0), at(1, 2, 3, 4), ["soil", "sediment", "soil", "soil"],
                                     ["A"] * 4, k=4)
    assert ranked == [["soil", "sediment"]]
    assert confidence == pytest.approx([0.75])


def test_knn_study_float_noise_cannot_break_a_tie():
    # k=11, query at 0 deg: study A has 10 'soil' samples at 1..10 deg, study B one 'sediment' at 11 deg.
    # soil = 1/10 added ten times = 0.9999999999999999 in floating point, sediment = 1: a real tie between
    # two curators, so it goes to the closest neighbour's label (soil), not to sediment by float noise.
    train = at(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    ranked, confidence = M.knn_study(at(0), train, ["soil"] * 10 + ["sediment"], ["A"] * 10 + ["B"], k=11)
    assert ranked == [["soil", "sediment"]]
    assert confidence == pytest.approx([0.5])


def test_knn_study_identical_training_vectors_are_ordered_by_training_index():
    # Two training samples with the same text (identical vectors) from different studies.
    # The similarity tie is broken by training-sample index: the lower index counts as closer.
    identical = at(0, 0)
    studies = np.array(["A", "B"])
    assert M.knn_study(at(0), identical, ["x", "y"], studies, k=1)[0] == [["x"]]
    assert M.knn_study(at(0), identical, ["y", "x"], studies, k=1)[0] == [["y"]]  # swap order -> swap answer
    # k=2: one vote each -> tie -> the 'closer' (lower index) label first
    assert M.knn_study(at(0), identical, ["x", "y"], studies, k=2)[0] == [["x", "y"]]


def test_knn_study_empty_test_set():
    ranked, confidence = M.knn_study(np.zeros((0, 2)), at(0), ["a"], np.array(["A"]), k=1)
    assert ranked == [] and len(confidence) == 0


def test_knn_study_regression_accepts_a_plain_list_of_studies():
    # (train_studies used to have to be a numpy array: a list raised TypeError)
    ranked, confidence = M.knn_study(at(0), at(1, 2), ["a", "b"], ["A", "B"], k=2)
    assert ranked == [["a", "b"]]
    assert confidence == pytest.approx([0.5])


# =============================================================================================== prototypes
# Toy data whose means are both zero, so centring changes nothing and every number can be read off:
#   training samples  'a': (1, 1), (1, -1)      -> normalised centroid direction (1, 0)
#                     'b': (-1, 1), (-1, -1)    -> normalised centroid direction (-1, 0)
#   term vectors      'a': (0, 1)   'b': (0, -1)   'c': (1, 0)   'd': (-1, 0)   ('c', 'd' never used as labels)
P_TRAIN = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
P_LABELS = np.array(["a", "a", "b", "b"])
P_VOCAB = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
P_IDS = np.array(["a", "b", "c", "d"])
R = 1 / np.sqrt(2)


def test_prototype_model_alpha_0_keeps_the_term_vectors():
    # alpha = 0 is 'retrieval_prior' in 5_evaluate.py: prototypes are the (centred) term vectors
    mean, prototypes, _ = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0, beta=0)
    assert_allclose(mean, [0, 0])
    assert_allclose(prototypes, P_VOCAB)


def test_prototype_model_alpha_1_uses_the_training_centroids():
    # alpha = 1: a nearest-centroid classifier for labels with training samples
    _, prototypes, _ = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=1, beta=0)
    assert_allclose(prototypes, [[1, 0],     # a: centroid of (1, 1) and (1, -1)
                                 [-1, 0],    # b: centroid of (-1, 1) and (-1, -1)
                                 [1, 0],     # c: no training samples -> term vector
                                 [-1, 0]],   # d: no training samples -> term vector
                    atol=1e-12)


def test_prototype_model_alpha_half_is_the_bisector_of_centroid_and_term():
    # a: normalise(0.5 * (1, 0) + 0.5 * (0, 1)) = (1, 1) / sqrt 2
    _, prototypes, _ = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0.5, beta=0)
    assert_allclose(prototypes, [[R, R], [-R, -R], [1, 0], [-1, 0]])


def test_prototype_model_centres_the_term_vectors():
    # Term vectors (2, 1) and (0, 1) share an offset (1, 1) ("modality gap"); centring removes it:
    # (2, 1) - (1, 1) = (1, 0) and (0, 1) - (1, 1) = (-1, 0)
    vocab = np.array([[2.0, 1.0], [0.0, 1.0]])
    _, prototypes, _ = M.prototype_model(np.array([[1.0, 0], [0, 1.0]]), np.array(["x", "y"]), vocab,
                                         np.array(["x", "y"]), alpha=0, beta=0)
    assert_allclose(prototypes, [[1, 0], [-1, 0]])


def test_prototype_model_centres_the_training_samples():
    # Samples (3, 1) 'x' and (1, 1) 'y' share an offset: their mean is (2, 1). Centred: (1, 0) and (-1, 0),
    # so with alpha = 1 the prototypes point exactly along +x and -x (not towards (3, 1) and (1, 1)).
    vocab = np.array([[0.0, 1.0], [0.0, -1.0]])
    _, prototypes, _ = M.prototype_model(np.array([[3.0, 1.0], [1.0, 1.0]]), np.array(["x", "y"]), vocab,
                                         np.array(["x", "y"]), alpha=1, beta=0)
    assert_allclose(prototypes, [[1, 0], [-1, 0]], atol=1e-12)


def test_prototype_model_each_sample_counts_equally_in_the_centroid():
    # Samples are normalised before they are averaged, so a long vector does not pull the centroid.
    # 'a': (3, 1) and (1, -1); 'b': (-3, -1) and (-1, 1) -> mean (0, 0), centring changes nothing.
    # normalised: (3, 1)/sqrt 10 + (1, -1)/sqrt 2 = (1.656, -0.391) -> direction (0.973, -0.230)
    # (without normalising: (3, 1) + (1, -1) = (4, 0) -> (1, 0))
    train = np.array([[3.0, 1.0], [1.0, -1.0], [-3.0, -1.0], [-1.0, 1.0]])
    _, prototypes, _ = M.prototype_model(train, np.array(["a", "a", "b", "b"]), P_VOCAB[:2], P_IDS[:2],
                                         alpha=1, beta=0)
    a = np.array([3, 1]) / np.sqrt(10) + np.array([1, -1]) / np.sqrt(2)
    assert_allclose(prototypes, [a / np.linalg.norm(a), -a / np.linalg.norm(a)])


def test_prototype_model_returns_the_training_mean():
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, _, _ = M.prototype_model(train, np.array(["a", "b"]), P_VOCAB, P_IDS, alpha=0.5, beta=0)
    assert_allclose(mean, [2, 3])


def test_prototype_model_bias_is_the_log_label_frequency():
    # counts a=2, b=2, c=0, d=0 -> (n + 0.5) = 2.5, 2.5, 0.5, 0.5, sum 6
    _, _, bias = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0, beta=1.0)
    assert_allclose(bias, np.log([2.5 / 6, 2.5 / 6, 0.5 / 6, 0.5 / 6]))
    # beta scales it; beta = 0 switches the prior off
    _, _, bias = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0, beta=0.1)
    assert_allclose(bias, 0.1 * np.log([2.5 / 6, 2.5 / 6, 0.5 / 6, 0.5 / 6]))
    _, _, bias = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0, beta=0)
    assert_allclose(bias, [0, 0, 0, 0])


def test_prototype_model_unseen_bonus_goes_to_terms_without_training_samples_only():
    _, _, bias = M.prototype_model(P_TRAIN, P_LABELS, P_VOCAB, P_IDS, alpha=0, beta=0, unseen_bonus=0.3)
    assert_allclose(bias, [0, 0, 0.3, 0.3])  # c and d are unseen


def test_prototype_model_every_training_label_must_be_in_the_vocabulary():
    with pytest.raises(KeyError):
        M.prototype_model(P_TRAIN, np.array(["a", "a", "b", "not_a_term"]), P_VOCAB, P_IDS, alpha=0.5, beta=0)


def test_prototype_model_quirk_identical_training_samples_have_no_centroid():
    # When all training samples are the same vector, centring makes them (0, 0): there is no centroid
    # direction. alpha < 1 falls back to the term vector; alpha = 1 leaves a zero prototype
    # (its score is then the bias alone).
    train, labels = np.array([[3.0, 4.0], [3.0, 4.0]]), np.array(["a", "a"])
    vocab, ids = np.array([[0.0, 1.0], [0.0, -1.0]]), np.array(["a", "b"])
    _, prototypes, _ = M.prototype_model(train, labels, vocab, ids, alpha=0.5, beta=0)
    assert_allclose(prototypes[0], [0, 1])
    _, prototypes, _ = M.prototype_model(train, labels, vocab, ids, alpha=1, beta=0)
    assert_allclose(prototypes[0], [0, 0])


# A hand-made model: mean (1, 1), prototypes x = (1, 0) and y = (0, 1), bias x = 0, y = -0.5
HAND_MODEL = (np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, -0.5]))
HAND_IDS = np.array(["x", "y"])


def test_prototype_scores_is_cosine_of_the_centred_sample_plus_bias():
    # (3, 1) - mean = (2, 0) -> direction (1, 0): cos to x = 1, to y = 0 -> scores 1 + 0, 0 - 0.5
    # (1, 3) - mean = (0, 2) -> direction (0, 1): cos to x = 0, to y = 1 -> scores 0 + 0, 1 - 0.5
    scores = M.prototype_scores(np.array([[3.0, 1.0], [1.0, 3.0]]), HAND_MODEL)
    assert_allclose(scores, [[1.0, -0.5], [0.0, 0.5]])


def test_prototype_ranks_by_prototype_scores_with_margin_confidence():
    ranked, confidence = M.prototype(np.array([[3.0, 1.0], [1.0, 3.0]]), HAND_MODEL, HAND_IDS)
    assert ranked == [["x", "y"], ["y", "x"]]
    assert_allclose(confidence, [1.5, 0.5])  # 1 - (-0.5), 0.5 - 0


def test_prototype_scores_out_collects_the_full_score_matrix():
    # 5_evaluate.py passes a list to keep the scores for the hierarchical back-off
    keep = []
    test = np.array([[3.0, 1.0], [1.0, 3.0]])
    M.prototype(test, HAND_MODEL, HAND_IDS, scores_out=keep)
    assert len(keep) == 1
    assert_allclose(keep[0], [[1.0, -0.5], [0.0, 0.5]])


def test_prototype_is_the_same_across_its_internal_1000_row_blocks():
    test = np.tile([[3.0, 1.0], [1.0, 3.0]], (1001, 1))  # 2,002 rows -> 3 blocks
    keep = []
    ranked, _ = M.prototype(test, HAND_MODEL, HAND_IDS, scores_out=keep)
    assert keep[0].shape == (2002, 2)
    assert_allclose(keep[0], M.prototype_scores(test, HAND_MODEL))
    assert ranked[1000] == ["x", "y"] and ranked[2001] == ["y", "x"]


def test_prototype_regression_empty_test_set():
    # (an empty test set used to fail inside sklearn's normalize)
    keep = []
    ranked, confidence = M.prototype(np.zeros((0, 2)), HAND_MODEL, HAND_IDS, scores_out=keep)
    assert ranked == [] and len(confidence) == 0
    assert keep[0].shape == (0, 2)  # 0 samples x 2 terms


# =============================================================================================== linear_scores
def test_linear_scores_single_class_returns_a_constant_column():
    classes, scores = M.linear_scores(at(0, 90), at(10, 20), np.array(["soil", "soil"]))
    assert list(classes) == ["soil"]
    assert_allclose(scores, [[1.0], [1.0]])


def test_linear_scores_binary_returns_one_column_per_class():
    # 'left' samples at x < 0, 'right' at x > 0. sklearn returns one score for a binary problem;
    # linear_scores expands it to two columns [-s, s] so column j always belongs to classes[j].
    train = np.array([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]])
    labels = np.array(["left", "left", "right", "right"])
    classes, scores = M.linear_scores(np.array([[-3.0, 0], [3.0, 0]]), train, labels)
    assert list(classes) == ["left", "right"]
    assert scores.shape == (2, 2)
    assert_allclose(scores[:, 0], -scores[:, 1])
    assert list(classes[scores.argmax(axis=1)]) == ["left", "right"]


def test_linear_scores_three_classes():
    # three well separated clusters; classes come back sorted
    train = np.array([[-5.0, 0], [-5.0, 1], [5.0, 0], [5.0, 1], [0, 5.0], [1.0, 5.0]])
    labels = np.array(["west", "west", "east", "east", "north", "north"])
    test = np.array([[-6.0, 0], [6.0, 0], [0, 6.0]])
    classes, scores = M.linear_scores(test, train, labels)
    assert list(classes) == ["east", "north", "west"]
    assert list(classes[scores.argmax(axis=1)]) == ["west", "east", "north"]


def test_linear_scores_sparse_features_use_linearsvc(monkeypatch):
    # dense features -> RidgeClassifier, sparse (TF-IDF) -> LinearSVC
    used = []
    real_svc, real_ridge = M.LinearSVC, M.RidgeClassifier
    monkeypatch.setattr(M, "LinearSVC", lambda **kw: used.append("LinearSVC") or real_svc(**kw))
    monkeypatch.setattr(M, "RidgeClassifier", lambda **kw: used.append("Ridge") or real_ridge(**kw))
    train = np.array([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0], [0, 2.0], [0, 3.0]])
    labels = np.array(["left", "left", "right", "right", "up", "up"])
    test = np.array([[-3.0, 0], [3.0, 0], [0, 4.0]])

    classes, scores = M.linear_scores(sparse.csr_matrix(test), sparse.csr_matrix(train), labels)
    assert used == ["LinearSVC"]
    assert list(classes[scores.argmax(axis=1)]) == ["left", "right", "up"]

    M.linear_scores(test, train, labels)
    assert used == ["LinearSVC", "Ridge"]


def test_linear_scores_regression_empty_test_set():
    # (an empty test set used to fail inside sklearn)
    classes, scores = M.linear_scores(np.zeros((0, 2)), at(0, 90), np.array(["b", "a"]))
    assert list(classes) == ["a", "b"]
    assert scores.shape == (0, 2)


# =============================================================================================== hybrid
def test_hybrid_adds_weighted_cosine_to_the_linear_score():
    # The classifier knows a and b; the vocabulary also has c (never a training label) and lists the
    # terms in a different order than `classes`. hybrid matches them by name.
    classes = np.array(["a", "b"])
    linear = np.array([[1.0, -1.0]])         # columns follow classes: a, b
    vocab_ids = np.array(["c", "a", "b"])
    cosine = np.array([[0.9, 0.0, 0.0]])     # columns follow vocab_ids: c, a, b
    # a = 1 + 2 x 0 = 1;  b = -1 + 2 x 0 = -1;  c (unseen -> -1) = -1 + 2 x 0.9 = 0.8
    ranked, confidence = M.hybrid(classes, linear, cosine, vocab_ids, weight=2.0)
    assert ranked == [["a", "c", "b"]]
    assert confidence == pytest.approx([0.2])  # 1 - 0.8


def test_hybrid_an_unseen_term_wins_only_through_its_cosine():
    # a = 0.5 + 0;  c = -1 + 2 x 1.0 = 1.0 -> c beats every known class
    ranked, _ = M.hybrid(np.array(["a", "b"]), np.array([[0.5, -1.0]]), np.array([[1.0, 0.0, 0.0]]),
                         np.array(["c", "a", "b"]), weight=2.0)
    assert ranked[0][0] == "c"
    # weight 0 -> the linear score alone: c stays at -1 and cannot win
    ranked, _ = M.hybrid(np.array(["a", "b"]), np.array([[0.5, -1.0]]), np.array([[1.0, 0.0, 0.0]]),
                         np.array(["c", "a", "b"]), weight=0.0)
    assert ranked[0][0] == "a"


def test_hybrid_every_class_must_be_in_the_vocabulary():
    with pytest.raises(KeyError):
        M.hybrid(np.array(["a", "z"]), np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]),
                 np.array(["a", "b"]), weight=1.0)


# =============================================================================================== label_regression
def test_label_regression_can_land_between_training_labels():
    # Training: four compass samples, each labelled with the term in its own direction.
    # The vocabulary also has 'northeast', which is never a training label.
    # Ridge learns (approximately, shrunk) the identity map; after normalising, a sample at 45 deg
    # predicts the 45 deg direction -> nearest term 'northeast'. A sample at 80 deg -> 'north'.
    train = at(0, 90, 180, 270)
    gold_vectors = train.copy()  # the term vector of each sample's gold label
    vocab_vectors = at(0, 45, 90, 180, 270)
    vocab_ids = np.array(["east", "northeast", "north", "west", "south"])
    ranked, confidence = M.label_regression(at(45, 80), train, gold_vectors, vocab_vectors, vocab_ids)
    assert [r[0] for r in ranked] == ["northeast", "north"]
    # margin of the first sample: cos 0 - cos 45 (east and north tie for second)
    assert confidence[0] == pytest.approx(1 - cos(45))


def test_label_regression_regression_empty_test_set():
    # (an empty test set used to fail inside sklearn)
    ranked, confidence = M.label_regression(np.zeros((0, 2)), at(0, 90), at(0, 90), at(0, 90),
                                            np.array(["east", "north"]))
    assert ranked == [] and len(confidence) == 0
