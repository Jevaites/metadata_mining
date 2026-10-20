"""
Unit tests of the helper functions of 6_predict_atlas.py, each on a few hand-made numbers whose
correct answer can be read off directly.

  function                         what it does in the pipeline                                  tests
  knn_study_top2                   the knn_study vote (one vote per study) for a block of samples  TestKnnStudyTop2
  sub_biome_parts + chunk_scores   the scores of linear / prototype, split over the two blocks     TestScoreDecomposition
  knn_study_chunk                  knn_study for one keyword chunk (split similarities)            TestScoreDecomposition
  ridge_class_scores, top1_and_margin  one score row per class; best label and margin           TestLinearHelpers
  candidate_strings                the reranker's candidate list 'id:p;id:p;...'                   TestCandidateStrings
  backoff_columns                  calibrated probability + hierarchical back-off of one slot      TestBackoffColumns
  load_backoff                     temperature, tau and closure per slot from calibration.json     TestLoadBackoff
  concat_parts                     the chunk TSVs -> atlas_predictions.tsv.gz                      TestConcatParts

The file-reading steps (train_models, build_index) and main() are tested on a toy atlas in
test_6_predict_atlas_integration.py.

Notation used below, as in the script: a sample vector is x = w [kw, sb], with kw and sb the
unit-norm keyword and sub-biome embeddings and w = 1/sqrt(2) (BLOCK_WEIGHT), so ||x|| = 1. A sample
without sub-biome has x = w [kw, 0].
"""
import gzip
import json
import types

import numpy as np
import pandas as pd
import pytest

import hierarchy
import methods
from common import SLOTS
from step6_toy import predict, write_calibration  # noqa: F401  (predict: a fixture)

W = 1 / np.sqrt(2)  # BLOCK_WEIGHT


def unit(degrees):
    """2-d unit vector at this angle: the dot product of two of them is the cosine of the angle between them."""
    return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])


# ============================================================================= knn_study_top2
class TestKnnStudyTop2:
    """knn_study_top2(sim, labels, studies, n_classes, k) -> (best class index, winner's vote share).

    Ten 2-d training vectors on a half circle, every 20 degrees; the query sits at 25 degrees:

        index   angle  label     angle to the query  -> rank of closeness
          0       0    soil          25                  3rd
          1      20    soil           5                  1st
          2      40    soil          15                  2nd
          3      60    sediment      35                  4th
          4      80    sediment      55                  5th
          5..9  100..180 water     >= 75                 6th..10th

    sim = query . training vector = cos(angle difference), exactly what the pipeline passes in.
    """
    CLASSES = np.array(["sediment", "soil", "water"])  # label = index into this array
    SOIL, SEDIMENT, WATER = 1, 0, 2
    labels = np.array([SOIL, SOIL, SOIL, SEDIMENT, SEDIMENT, WATER, WATER, WATER, WATER, WATER])
    train = np.array([unit(a) for a in range(0, 181, 20)])
    sim = (unit(25) @ train.T)[None, :]  # 1 query x 10 training samples

    def top2(self, studies, k, sim=None):
        best, share = self.predict_fn(self.sim if sim is None else sim, self.labels, np.array(studies), 3, k)
        return self.CLASSES[best[0]], round(float(share[0]), 4)

    @pytest.fixture(autouse=True)
    def _bind(self, predict):
        self.predict_fn = predict.knn_study_top2

    def test_every_neighbour_from_its_own_study_is_a_plain_majority_vote(self):
        own_study = list(range(10))
        assert self.top2(own_study, k=3) == ("soil", 1.0)  # neighbours 1, 2, 0: all soil
        assert self.top2(own_study, k=5) == ("soil", 0.6)  # + 3 and 4: soil 3 votes, sediment 2

    def test_a_study_has_one_vote_in_total(self):
        # the three soil samples come from one study A, the two sediment samples from studies B and C
        studies = ["A", "A", "A", "B", "C", "D", "E", "F", "G", "H"]
        studies = np.unique(studies, return_inverse=True)[1]  # the script passes study *indices*
        # k=5: soil = 1/3 + 1/3 + 1/3 = 1 (one curator), sediment = 1 + 1 = 2 (two curators)
        assert self.top2(studies, k=5) == ("sediment", round(2 / 3, 4))

    def test_tie_goes_to_the_label_of_the_closest_neighbour(self):
        studies = np.unique(["A", "A", "A", "B", "C", "D", "E", "F", "G", "H"], return_inverse=True)[1]
        # k=4: soil 1 (study A), sediment 1 (study B): a tie; the closest neighbour (index 1) is soil
        assert self.top2(studies, k=4) == ("soil", 0.5)

    def test_unlabelled_training_samples_are_never_neighbours(self):
        # the script sets the similarity of training samples without a label in the slot to -inf;
        # here the three soil samples are unlabelled, so the 3 nearest are 3, 4 (sediment) and 5 (water)
        sim = self.sim.copy()
        sim[0, [0, 1, 2]] = -np.inf
        assert self.top2(list(range(10)), k=3, sim=sim) == ("sediment", round(2 / 3, 4))

    def test_fewer_labelled_samples_than_k_only_the_labelled_ones_vote(self):
        # REGRESSION (fixed 2026-10-19). Only samples 1 (soil) and 3 (sediment) are labelled in this slot;
        # the 8 others are unlabelled: class 0 = "" and similarity -inf, exactly as train_models and
        # knn_study_chunk prepare them. k=5 > 2 labelled samples: the two labelled ones vote, 1 each,
        # a tie won by the closest (soil, 5 degrees away). Before the fix, three unlabelled samples filled
        # the free neighbour slots and "" won with 3 votes of 5. 5_evaluate.knn_study never sees unlabelled
        # samples, so this is also what the evaluation scored.
        classes = np.array(["", "sediment", "soil"])
        labels = np.zeros(10, int)
        labels[1], labels[3] = 2, 1
        sim = np.where(labels > 0, self.sim, -np.inf)
        best, share = self.predict_fn(sim, labels, np.arange(10), 3, 5)
        assert (classes[best[0]], share[0]) == ("soil", 0.5)

    def test_k_larger_than_the_training_set_means_every_sample_votes(self):
        # REGRESSION (fixed 2026-10-19): k >= the number of training samples used to raise in
        # np.argpartition; now every training sample is a neighbour: water 5 votes of 10
        own_study = list(range(10))
        assert self.top2(own_study, k=10) == ("water", 0.5)
        assert self.top2(own_study, k=50) == ("water", 0.5)  # 50 = the --knn_study_k default


# ============================================================================= score decomposition
def toy_blocks():
    """Two distinct keyword vectors, two distinct sub-biome vectors (2-d, unit norm), and four atlas
    samples that combine them. In the script, kw holds the rows of the current keyword chunk, sb all
    distinct sub-biomes; kw_local / sb_local are each sample's row there, and sb row len(sb) = 2
    means 'no sub-biome'."""
    kw = W * np.array([unit(0), unit(90)])  # already multiplied by w, as main() does
    sb = W * np.array([unit(0), unit(45)])
    kw_local = np.array([0, 0, 1, 1])
    sb_local = np.array([0, 2, 0, 1])       # sample 1 has no sub-biome
    has_sb = (sb_local < len(sb)).astype(np.float32)
    # the same samples as full vectors x = [w kw, w sb] (zeros when there is no sub-biome)
    sb_or_zero = np.vstack([sb, np.zeros(2)])
    X = np.hstack([kw[kw_local], sb_or_zero[sb_local]])
    return kw, sb, kw_local, sb_local, has_sb, X


def same_params_for_every_slot(**params):
    """The functions loop over the three slots; the tests give every slot the same parameters."""
    return {"kw_dim": 2, **{f"{slot}_{name}": value for slot in SLOTS for name, value in params.items()}}


# linear model with three classes, weights chosen by hand: columns 0-1 act on the keywords, 2-3 on the sub-biome
LINEAR_COEF = np.array([[1.0, 0.0, 0.5, 0.0],
                        [0.0, 1.0, 0.0, 0.5],
                        [-1.0, -1.0, 0.0, 0.0]], dtype=np.float32)
LINEAR_INTERCEPT = np.array([0.0, 0.1, 0.2], dtype=np.float32)
LINEAR_MODEL = same_params_for_every_slot(coef=LINEAR_COEF, intercept=LINEAR_INTERCEPT)
# prototype model with three classes: training mean mu, unit-norm prototypes, label-frequency bias
PROTO_MEAN = np.array([0.1, 0.2, 0.0, 0.1], dtype=np.float32)
PROTO_P = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float32) / np.sqrt(2).astype(np.float32)
PROTO_BIAS = np.array([0.0, -0.1, -0.2], dtype=np.float32)
PROTO_MODEL = same_params_for_every_slot(mean=PROTO_MEAN, prototypes=PROTO_P, bias=PROTO_BIAS)


def decomposed_scores(predict, method, model):
    """The script's scores of the four toy_blocks() samples (sub_biome_parts + chunk_scores), and their
    full vectors X."""
    kw, sb, kw_local, sb_local, has_sb, X = toy_blocks()
    parts = predict.sub_biome_parts(method, model, sb)
    return predict.chunk_scores(method, "biome", model, parts, kw, kw_local, sb_local, has_sb), X


class TestScoreDecomposition:
    """The script never builds x = w [kw, sb] for the 3.4M atlas samples: it scores each distinct
    keyword text and each distinct sub-biome text once and adds the two parts. These tests check that
    the sum is exactly the score of the model applied to the full vector x."""

    def test_linear_scores_are_W_x_plus_b(self, predict):
        scores, X = decomposed_scores(predict, "linear", LINEAR_MODEL)
        np.testing.assert_allclose(scores, X @ LINEAR_COEF.T + LINEAR_INTERCEPT, atol=1e-6)
        # by hand, sample 0 = w [(1, 0), (1, 0)]: class 0 scores w (1 + 0.5) = 1.0607
        assert scores[0, 0] == pytest.approx(1.5 * W, abs=1e-6)
        # sample 1 has no sub-biome: only the keyword weights and the intercept count, w * 1 + 0
        assert scores[1, 0] == pytest.approx(W, abs=1e-6)

    def test_prototype_scores_are_cosine_to_centred_x_plus_bias(self, predict):
        scores, X = decomposed_scores(predict, "prototype", PROTO_MODEL)
        # the evaluated model (methods.prototype_scores = cos(x - mu, P) + bias) on the full vectors
        np.testing.assert_allclose(scores, methods.prototype_scores(X, (PROTO_MEAN, PROTO_P, PROTO_BIAS)), atol=1e-5)

    def test_knn_study_chunk_equals_methods_knn_study(self, predict):
        # six training samples: three near 0 degrees (soil, two studies), three near 90 degrees (water)
        train_kw = np.array([unit(0), unit(5), unit(10), unit(80), unit(85), unit(90)])
        train_sb = np.array([unit(0), unit(0), unit(0), unit(90), unit(90), unit(90)])
        train_x = (W * np.hstack([train_kw, train_sb])).astype(np.float32)
        studies = np.array(["A", "A", "B", "C", "C", "D"])
        y = np.array(["soil", "soil", "soil", "water", "water", "water"])
        classes, labels = np.unique(y, return_inverse=True)
        model = {"kw_dim": 2, "train_x": train_x, "train_studies": studies,
                 **{f"{slot}_classes": classes for slot in SLOTS}, **{f"{slot}_labels": labels for slot in SLOTS}}
        kw, sb, kw_local, sb_local, has_sb, X = toy_blocks()

        parts = predict.sub_biome_parts("knn_study", model, sb)
        best, confidence = predict.knn_study_chunk(model, parts, kw, kw_local, sb_local, k=3)

        ranked, expected_confidence = methods.knn_study(X, train_x, y, studies, k=3)
        assert list(classes[best["biome"]]) == [r[0] for r in ranked]
        np.testing.assert_allclose(confidence["biome"], expected_confidence)
        # readable check: sample 0 (keywords at 0 degrees, sub-biome at 0) -> soil, 3 neighbours from
        # studies A, A, B -> 1/2 + 1/2 + 1 = 2 votes for soil = all of them
        assert (classes[best["biome"][0]], confidence["biome"][0]) == ("soil", 1.0)

    def test_knn_study_chunk_ties_among_identical_training_vectors(self, predict):
        # Real training vectors repeat exactly (keywords: 50,100 samples share 22,618 distinct vectors;
        # sub-biomes: 911), so similarities tie and methods.TIE_BREAK decides which tied samples are among
        # the k nearest: the lowest training rows, in both scripts. Here rows 0-7 share one vector (keywords
        # and sub-biome at 0 degrees), row 1 is unlabelled, row 8 is elsewhere (90 degrees):
        #   row    0      1     2     3      4     5     6     7     8
        #   label  water  -     soil  water  soil  soil  soil  soil  sediment
        #   study  A      B     B     C      D     E     F     G     H
        # Atlas sample 0 (at 0 degrees), k = 4: the labelled rows 0, 2, 3, 4 -> water 2 (studies A, C),
        # soil 2 (B, D): a tie, won by the label of row 0 -> water, share 0.5. Any other choice of 4 tied
        # rows gives soil (np.argpartition without the tie-break picks rows 3-7: soil 1.0).
        train_kw = np.array([unit(0)] * 8 + [unit(90)])
        train_x = (W * np.hstack([train_kw, train_kw])).astype(np.float32)
        studies = np.array(list("ABBCDEFGH"))
        y = np.array(["water", "", "soil", "water", "soil", "soil", "soil", "soil", "sediment"])
        classes, labels = np.unique(y, return_inverse=True)
        model = {"kw_dim": 2, "train_x": train_x, "train_studies": studies,
                 **{f"{slot}_classes": classes for slot in SLOTS}, **{f"{slot}_labels": labels for slot in SLOTS}}
        kw, sb, kw_local, sb_local, has_sb, X = toy_blocks()

        parts = predict.sub_biome_parts("knn_study", model, sb)
        best, confidence = predict.knn_study_chunk(model, parts, kw, kw_local, sb_local, k=4)

        assert (classes[best["biome"][0]], confidence["biome"][0]) == ("water", 0.5)
        # the same as the evaluated knn_study, which only sees the labelled rows, on all four atlas samples
        labelled = y != ""
        ranked, expected_confidence = methods.knn_study(X, train_x[labelled], y[labelled], studies[labelled], k=4)
        assert list(classes[best["biome"]]) == [r[0] for r in ranked]
        np.testing.assert_allclose(confidence["biome"], expected_confidence)

    def test_knn_study_chunk_with_fewer_labelled_samples_than_k(self, predict):
        # REGRESSION (fixed 2026-10-19). Same six training samples, but only two are labelled in the slot:
        # sample 0 (soil, near 0 degrees) and sample 3 (water, near 90 degrees); k = 4 > 2.
        # 5_evaluate only ever sees the two labelled samples: each votes 1, a tie won by the closer one.
        # So every atlas sample gets the label of the closer labelled sample, with half of the vote.
        train_kw = np.array([unit(0), unit(5), unit(10), unit(80), unit(85), unit(90)])
        train_sb = np.array([unit(0), unit(0), unit(0), unit(90), unit(90), unit(90)])
        train_x = (W * np.hstack([train_kw, train_sb])).astype(np.float32)
        studies = np.array(["A", "A", "B", "C", "C", "D"])
        y = np.array(["soil", "", "", "water", "", ""])
        classes, labels = np.unique(y, return_inverse=True)  # classes = ["", "soil", "water"]
        model = {"kw_dim": 2, "train_x": train_x, "train_studies": studies,
                 **{f"{slot}_classes": classes for slot in SLOTS}, **{f"{slot}_labels": labels for slot in SLOTS}}
        kw, sb, kw_local, sb_local, has_sb, X = toy_blocks()

        parts = predict.sub_biome_parts("knn_study", model, sb)
        best, confidence = predict.knn_study_chunk(model, parts, kw, kw_local, sb_local, k=4)

        labelled = y != ""
        ranked, expected_confidence = methods.knn_study(X, train_x[labelled], y[labelled], studies[labelled], k=4)
        # (atlas sample 2 has its keywords at 90 degrees but its sub-biome at 0: similarity 0.50 to the
        # soil sample, 0.49 to the water sample)
        assert [r[0] for r in ranked] == ["soil", "soil", "soil", "water"]
        assert list(classes[best["biome"]]) == ["soil", "soil", "soil", "water"]
        np.testing.assert_allclose(confidence["biome"], expected_confidence)
        assert list(confidence["biome"]) == [0.5, 0.5, 0.5, 0.5]


# ============================================================================= linear: per-class scores, top-1
class TestLinearHelpers:
    """ridge_class_scores(clf) -> (coef, intercept) with one row per class; top1_and_margin(scores)."""

    def test_two_classes_get_two_opposite_score_rows(self, predict):
        # REGRESSION (fixed 2026-10-19): with two classes scikit-learn keeps one score s (for classes_[1]);
        # the script needs one row per class, so class 0 scores -s (as methods.linear_scores does)
        from sklearn.linear_model import RidgeClassifier
        X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        clf = RidgeClassifier(alpha=1.0).fit(X, ["soil", "soil", "water", "water"])
        coef, intercept = predict.ridge_class_scores(clf)
        assert coef.shape == (2, 2) and intercept.shape == (2,)
        scores = X @ coef.T + intercept
        np.testing.assert_allclose(scores[:, 1], clf.decision_function(X))  # class 1 (water): s
        np.testing.assert_allclose(scores[:, 0], -clf.decision_function(X))  # class 0 (soil): -s
        assert list(clf.classes_[scores.argmax(axis=1)]) == ["soil", "soil", "water", "water"]

    def test_three_classes_are_unchanged(self, predict):
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=1.0).fit(np.eye(3), ["a", "b", "c"])
        coef, intercept = predict.ridge_class_scores(clf)
        np.testing.assert_array_equal(coef, clf.coef_)
        np.testing.assert_array_equal(intercept, clf.intercept_)

    def test_top1_and_margin(self, predict):
        # the example of methods.rank: best = column 1 (0.9), margin 0.9 - 0.5 = 0.4
        best, margin = predict.top1_and_margin(np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]]))
        assert list(best) == [1, 0]
        np.testing.assert_allclose(margin, [0.4, 1.0])

    def test_a_single_class_wins_with_margin_0(self, predict):
        # a slot with one label in training: nothing to choose, so margin 0 (as methods.rank)
        best, margin = predict.top1_and_margin(np.array([[-1.0], [-1.0]]))
        assert list(best) == [0, 0] and list(margin) == [0.0, 0.0]


# ============================================================================= candidate_strings
class TestCandidateStrings:
    """candidate_strings(P, classes, k, broader, max_candidates): the k most probable classes, then the
    broader terms of those that are themselves classes, up to max_candidates."""
    classes = np.array(["ENVO_FOREST", "ENVO_GRASSLAND", "ENVO_TERRESTRIAL", "ENVO_MARINE"])
    broader = [[2], [2], [], []]  # forest and grassland are_a terrestrial (class index 2)
    P = np.array([[0.50, 0.10, 0.05, 0.35]])

    def test_top_k_then_their_broader_terms(self, predict):
        out = predict.candidate_strings(self.P, self.classes, 2, self.broader, max_candidates=10)
        # top-2 = forest (0.5), marine (0.35); then terrestrial, a broader term of forest
        assert out == ["ENVO_FOREST:0.5;ENVO_MARINE:0.35;ENVO_TERRESTRIAL:0.05"]

    def test_max_candidates_equal_to_k_means_no_broader_terms(self, predict):
        assert predict.candidate_strings(self.P, self.classes, 2, self.broader, 2) == ["ENVO_FOREST:0.5;ENVO_MARINE:0.35"]

    def test_k_larger_than_the_vocabulary_lists_every_class_once(self, predict):
        out = predict.candidate_strings(self.P, self.classes, 10, self.broader, 10)
        assert out == ["ENVO_FOREST:0.5;ENVO_MARINE:0.35;ENVO_GRASSLAND:0.1;ENVO_TERRESTRIAL:0.05"]

    def test_a_broader_term_already_in_the_top_k_is_not_repeated(self, predict):
        P = np.array([[0.50, 0.05, 0.40, 0.05]])  # terrestrial is itself in the top-2
        assert predict.candidate_strings(P, self.classes, 2, self.broader, 10) == ["ENVO_FOREST:0.5;ENVO_TERRESTRIAL:0.4"]


# ============================================================================= backoff_columns
BIOME_ROOT = "ENVO_00000428"  # 'biome': in hierarchy.NO_ANSWER, never an answer


class TestBackoffColumns:
    """backoff_columns(scores, (T, tau, closure, broader), classes, label_of, args) for one slot.

    Toy slot with five labels; forest biome is_a terrestrial biome, everything is_a 'biome' (the root):

                    biome (root: never an answer)
           /           |        |       \\
      terrestrial    marine    lake    river
          |
        forest

    A broad term is an answer only if it is a label of the slot and fewer than half of the labels sit
    at or below it (hierarchy.Closure): terrestrial covers 2 of 5 labels, so it can be an answer.

    The scores are log(P) and the temperature is 1, so the calibrated probabilities are exactly P
    (softmax(log P) = P): every probability below can be read from the test.
    """
    classes = np.array(["ENVO_FOREST", "ENVO_TERRESTRIAL", "ENVO_MARINE", "ENVO_LAKE", "ENVO_RIVER"])
    ancestors = {"ENVO_FOREST": {"ENVO_TERRESTRIAL", BIOME_ROOT}, "ENVO_TERRESTRIAL": {BIOME_ROOT},
                 "ENVO_MARINE": {BIOME_ROOT}, "ENVO_LAKE": {BIOME_ROOT}, "ENVO_RIVER": {BIOME_ROOT}}
    label_of = {"ENVO_FOREST": "forest biome", "ENVO_TERRESTRIAL": "terrestrial biome", "ENVO_MARINE": "marine biome"}
    P = np.array([
        # forest terr.  marine lake   river
        [0.90, 0.025, 0.025, 0.025, 0.025],  # sure of forest                      -> top1 forest (0.9)
        [0.50, 0.40,  0.05,  0.025, 0.025],  # forest or terrestrial: q(terr.) 0.9 -> coarser terrestrial
        [0.05, 0.05,  0.60,  0.15,  0.15],   # marine 0.6, above it only the root  -> abstain
    ])
    TAU = 0.8

    def columns(self, predict, T=1.0, topk=0):
        closure = hierarchy.Closure(self.classes, self.ancestors)
        args = types.SimpleNamespace(topk=topk, max_candidates=10)
        broader = [[1], [], [], [], []]
        return predict.backoff_columns(np.log(self.P), (T, self.TAU, closure, broader), self.classes, self.label_of, args)

    def test_top1_coarser_and_abstain(self, predict):
        c = self.columns(predict)
        assert list(c["p"]) == [0.9, 0.5, 0.6]                       # calibrated top-1 probability
        assert list(c["backoff"]) == ["ENVO_FOREST", "ENVO_TERRESTRIAL", ""]
        assert list(c["backoff_label"]) == ["forest biome", "terrestrial biome", ""]
        assert list(c["backoff_p"]) == [0.9, 0.9, 0.0]               # 0.5 + 0.4 for terrestrial
        assert list(c["backoff_kind"]) == ["top1", "coarser", "abstain"]
        assert "candidates" not in c                                 # topk = 0: no candidate column

    def test_a_higher_temperature_makes_the_model_less_sure(self, predict):
        # T = 2 flattens P (softmax(log P / 2) is proportional to sqrt(P)): for the first sample,
        # sqrt(0.9) = 0.949 and sqrt(0.025) = 0.158 (x 4) -> forest 0.6, every other label 0.1, so
        # forest + terrestrial = 0.7 < tau: the sample that was 'top1' at T = 1 now abstains (as do the others)
        c = self.columns(predict, T=2.0)
        assert c["p"][0] == 0.6
        assert list(c["backoff_kind"]) == ["abstain", "abstain", "abstain"]

    def test_candidates_column(self, predict):
        # with --topk the reranker's candidate list is added, from the same calibrated probabilities
        c = self.columns(predict, topk=2)
        assert c["candidates"][1] == "ENVO_FOREST:0.5;ENVO_TERRESTRIAL:0.4"  # (rows 0 and 2 have tied runners-up)


# ============================================================================= load_backoff
class TestLoadBackoff:
    """load_backoff(args, model, terms, out_dir) -> {slot: (temperature, tau, Closure, broader)}."""
    classes = np.array(["ENVO_FOREST", "ENVO_GRASSLAND", "ENVO_TERRESTRIAL", "ENVO_MARINE"])
    terms = pd.DataFrame({"term_id": ["ENVO_FOREST", "ENVO_GRASSLAND", "ENVO_TERRESTRIAL", "ENVO_MARINE", BIOME_ROOT],
                          "parents": ["ENVO_TERRESTRIAL", "ENVO_TERRESTRIAL", BIOME_ROOT, BIOME_ROOT, ""]})
    # out-of-fold accuracy at each tau: the lowest tau reaching the target accuracy is chosen
    CURVE = [(0.3, 0.80), (0.6, 0.92), (0.9, 0.99)]
    SETTINGS = {"samples": "/elsewhere/metalog_training_set.tsv.gz", "n_samples": 6, "max_per_study": 50, "seed": 22}

    def run(self, predict, tmp_path, calibrations, target=0.9, accuracy="strict", n_samples=6):
        args = types.SimpleNamespace(calibration=[str(c) for c in calibrations], method="linear", target_accuracy=target,
                                     accuracy=accuracy, topk=5, max_candidates=10, max_per_study=50, seed=22,
                                     samples="~/metalog/metalog_training_set.tsv.gz")
        model = {f"{slot}_classes": self.classes for slot in SLOTS}
        if n_samples is not None:  # None: a model.npz from before n_samples was saved (as atlas_backoff/model.npz)
            model["n_samples"] = n_samples
        tmp_path.mkdir(exist_ok=True)  # the output dir (run_settings.json is written there)
        return predict.load_backoff(args, model, self.terms, str(tmp_path))

    def test_temperature_and_tau_per_slot(self, predict, tmp_path):
        # tau = the lowest one reaching --target_accuracy (hierarchy.choose_tau, tested in test_hierarchy_unit)
        cal = write_calibration(tmp_path / "calibration.json", "linear", 0.2, self.CURVE, self.SETTINGS)
        backoff = self.run(predict, tmp_path, [cal], target=0.9)
        assert set(backoff) == set(SLOTS)
        temperature, tau, closure, broader = backoff["biome"]
        assert (temperature, tau) == (0.2, 0.6)

    def test_accuracy_option_chooses_the_curve_column(self, predict, tmp_path):
        # lenient accuracy (a more specific answer also counts) reaches 0.9 already at tau 0.3
        cal = write_calibration(tmp_path / "calibration.json", "linear", 0.2, self.CURVE, self.SETTINGS)
        data = json.load(open(cal))
        for slot in data:
            data[slot]["linear"]["curve"][0]["accuracy_lenient"] = 0.95
        json.dump(data, open(cal, "w"))
        assert self.run(predict, tmp_path / "s", [cal], accuracy="strict")["biome"][1] == 0.6
        assert self.run(predict, tmp_path / "l", [cal], accuracy="lenient")["biome"][1] == 0.3

    def test_model_without_n_samples_is_checked_against_the_first_calibration(self, predict, tmp_path):
        # older model.npz files have no n_samples: the calibrations must then at least agree with each other
        a = write_calibration(tmp_path / "a.json", "linear", 0.2, self.CURVE, self.SETTINGS)
        b = write_calibration(tmp_path / "b.json", "linear", 0.2, self.CURVE, {**self.SETTINGS, "n_samples": 7})
        assert self.run(predict, tmp_path / "one", [a], n_samples=None)["biome"][1] == 0.6
        with pytest.raises(SystemExit, match="not the same training set"):
            self.run(predict, tmp_path / "two", [a, b], n_samples=None)

    def test_broader_lists_the_classes_above_each_class(self, predict, tmp_path):
        cal = write_calibration(tmp_path / "calibration.json", "linear", 0.2, self.CURVE, self.SETTINGS)
        _, _, _, broader = self.run(predict, tmp_path, [cal])["biome"]
        # forest -> terrestrial (index 2), grassland -> terrestrial; the root is not a class
        assert broader == [[2], [2], [], []]

    def test_several_calibrations_are_pooled(self, predict, tmp_path):
        # fold seed 1 reaches 0.9 only at tau 0.9; fold seed 2 already at tau 0.6; their mean at tau 0.6
        # is (0.88 + 0.94) / 2 = 0.91 >= 0.9, so the pooled tau is 0.6. The temperature is the geometric mean.
        a = write_calibration(tmp_path / "a.json", "linear", 0.1, [(0.3, 0.70), (0.6, 0.88), (0.9, 0.99)], self.SETTINGS)
        b = write_calibration(tmp_path / "b.json", "linear", 0.4, [(0.3, 0.80), (0.6, 0.94), (0.9, 0.99)], self.SETTINGS)
        temperature, tau, _, _ = self.run(predict, tmp_path, [a, b])["biome"]
        assert tau == 0.6
        assert temperature == pytest.approx(np.sqrt(0.1 * 0.4))

    def test_writes_run_settings_json(self, predict, tmp_path):
        cal = write_calibration(tmp_path / "calibration.json", "linear", 0.2, self.CURVE, self.SETTINGS)
        self.run(predict, tmp_path, [cal])
        settings = json.load(open(tmp_path / "run_settings.json"))
        assert settings["target_accuracy"] == 0.9 and len(settings["calibration"]) == 1  # md5 of the file content


# ============================================================================= concat_parts
class TestConcatParts:
    def write_part(self, p, text):
        with gzip.open(p, "wt") as handle:
            handle.write(text)
        return str(p)

    def test_header_once_then_the_rows_in_order(self, predict, tmp_path):
        a = self.write_part(tmp_path / "rows_00000000.tsv.gz", "sample_id\tbiome\nS1\tforest\nS2\tmarine\n")
        b = self.write_part(tmp_path / "rows_00000002.tsv.gz", "sample_id\tbiome\nS3\tlake\n")
        empty = self.write_part(tmp_path / "rows_00000004.tsv.gz", "sample_id\tbiome\n")  # chunk without samples
        predict.concat_parts([a, b, empty], str(tmp_path / "final.tsv.gz"))
        assert gzip.open(tmp_path / "final.tsv.gz", "rt").read() == "sample_id\tbiome\nS1\tforest\nS2\tmarine\nS3\tlake\n"

    def test_parts_with_other_columns_are_refused(self, predict, tmp_path):
        a = self.write_part(tmp_path / "a.tsv.gz", "sample_id\tbiome\nS1\tforest\n")
        b = self.write_part(tmp_path / "b.tsv.gz", "sample_id\tbiome\tbiome_p\nS2\tmarine\t0.9\n")
        with pytest.raises(SystemExit, match="other columns"):
            predict.concat_parts([a, b], str(tmp_path / "final.tsv.gz"))
        assert not (tmp_path / "final.tsv.gz").exists()  # only .tmp: never a half-written final file
