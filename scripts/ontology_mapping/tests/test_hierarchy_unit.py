"""Unit tests for hierarchy.py: one class per function, each test checks one behaviour on tiny inputs.

Most tests use the toy biome ontology of toy_hierarchy.py (drawn again here for reading convenience):

    biome (root, never an answer; above it environmental system and BFO_0000040, as in ENVO)
    ├── aquatic          label     ├── marine  label  └── ocean  label
    │                              └── freshwater  label  └── lake  (not a label)
    └── terrestrial (not a label)  ├── soil, desert, forest, grassland, tundra  (labels)

VOCAB = [ocean, marine, freshwater, aquatic, soil, desert, forest, grassland, tundra] = the columns of P.
probs({OCEAN: .55, ...}) builds a probability row over VOCAB (missing labels = 0).

Functions under test, in the order the pipeline uses them:
    softmax(S, T)                   scores -> probabilities
    fit_temperature(blocks)         the T that makes softmax(S / T) match how often the gold label is right
    Closure(vocab, anc)             label x node matrix A; q = P @ A = summed probability at or below each node
    Closure.decode(P, tau)          climb from the top-1 label to the most specific allowed node with q >= tau
    relation(nodes, gold, anc)      EXACT / COARSER / TOO_SPECIFIC / OTHER for every (sample, node)
    outcome(rel, pick)              the code of the picked node (0 = abstain)
    summarise(out)                  coverage, accuracies, outcome shares
    choose_tau(curve, target)       lowest tau whose accuracy reaches the target
    oof_backoff(folds, anc)         all of the above, cross-fitted over folds (mainly in the integration tests)
    decode_candidates(...)          the same climb over a short candidate list (LLM reranker)
    merge_calibrations(entries)     pool calibration curves of several fold seeds

Tests named `..._quirk` pin down current behaviour that is surprising but harmless today.

Tests marked "regression test (fixed 2026-10-19)" pin down two bugs found while writing these tests
(answers="ontology" dropped PO_/FOODON_ labels; decode_candidates broke ties by set order).

See test_hierarchy_integration.py for how the functions are chained in 5_evaluate / 6_predict_atlas /
7_rerank_atlas, and test_hierarchy_mutations.py for the bugs these tests are proven to catch.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from toy_hierarchy import (ANC, AQUATIC, BFO, BIOME, DESERT, FOREST, FRESHWATER, GRASSLAND, LAKE, MARINE, OCEAN,
                           SCRIPTS_DIR, SOIL, SYSTEM, TERRESTRIAL, TUNDRA, VOCAB, log_scores, probs)

import hierarchy
from hierarchy import COARSER, EXACT, OTHER, TOO_SPECIFIC

LN3 = np.log(3)


# ----------------------------------------------------------------------------- helpers
def answer(closure, row, tau):
    """Decode ONE probability row (a dict) -> (answer term or None for abstain, its q rounded)."""
    pick, q = closure.decode(probs(row, vocab=closure.vocab), tau)
    return (str(closure.nodes[pick[0]]) if pick[0] >= 0 else None), round(float(q[0]), 4)


def q_of(closure, row):
    """{node: q} for ONE probability row (a dict): q = summed probability of the labels at or below the node."""
    q = probs(row, vocab=closure.vocab) @ closure.A
    return {str(node): round(float(v), 4) for node, v in zip(closure.nodes, q[0])}


def allowed(closure):
    """The nodes that may be an answer."""
    return {str(node) for node in closure.nodes[closure.informative]}


# The worked example of the module docstring: 55 % ocean, 15 % marine, 15 % freshwater, 15 % elsewhere (soil)
WORKED_EXAMPLE = {OCEAN: .55, MARINE: .15, FRESHWATER: .15, SOIL: .15}


# ============================================================================= softmax
class TestSoftmax:
    def test_each_row_separately_and_rows_sum_to_one(self):
        # exp(0) : exp(ln 3) = 1 : 3  ->  25 % : 75 %
        P = hierarchy.softmax(np.array([[0.0, LN3], [LN3, 0.0], [5.0, 5.0]]), T=1)
        assert P == pytest.approx(np.array([[0.25, 0.75], [0.75, 0.25], [0.5, 0.5]]))
        assert P.sum(1) == pytest.approx([1, 1, 1])

    def test_scores_are_divided_by_the_temperature(self):
        # (0, 2 ln 3) / T=2 = (0, ln 3)  ->  25 % : 75 %
        assert hierarchy.softmax(np.array([[0.0, 2 * LN3]]), T=2) == pytest.approx(np.array([[0.25, 0.75]]))

    def test_low_temperature_sharpens_high_temperature_flattens(self):
        S = np.array([[0.0, 1.0]])
        assert hierarchy.softmax(S, T=0.01)[0, 1] > 0.999   # almost one-hot
        assert hierarchy.softmax(S, T=100)[0, 1] < 0.51     # almost uniform

    def test_huge_scores_do_not_overflow(self):
        # exp(1000) overflows; subtracting the row max first keeps it finite
        assert hierarchy.softmax(np.array([[1000.0, 1000.0]]), T=1) == pytest.approx(np.array([[0.5, 0.5]]))


# ============================================================================= fit_temperature
def two_class_block(n_gold_1, n_gold_0, scores=(0.0, 1.0)):
    """Rows that all have the same scores; gold is column 1 in n_gold_1 rows, column 0 in n_gold_0 rows."""
    S = np.array([scores] * (n_gold_1 + n_gold_0))
    gold = np.array([1] * n_gold_1 + [0] * n_gold_0)
    return S, gold


class TestFitTemperature:
    def test_temperature_makes_probabilities_match_observed_frequencies(self):
        # Every row scores (0, 1) and column 1 is right 75 % of the time. The best T makes
        # softmax((0, 1) / T) = (25 %, 75 %): exp(1/T) = 3, i.e. T = 1 / ln 3 = 0.910.
        T = hierarchy.fit_temperature([two_class_block(75, 25)])
        assert T == pytest.approx(1 / LN3, rel=1e-3)

    def test_rows_whose_gold_is_not_in_the_vocabulary_are_ignored(self):
        # gold = -1: the model could not have predicted that label, so the row says nothing about T
        S, gold = two_class_block(75, 25)
        S = np.vstack([S, [[5.0, 0.0]] * 50])  # 50 more rows, scores that would change T if they counted
        gold = np.concatenate([gold, [-1] * 50])
        assert hierarchy.fit_temperature([(S, gold)]) == pytest.approx(1 / LN3, rel=1e-3)

    def test_blocks_with_different_vocabularies_are_pooled(self):
        # block 2 has 3 columns; column 1 scores -100 (never chosen), columns 0 / 2 behave like block 1
        block1 = two_class_block(75, 25)
        S2 = np.array([[1.0, -100.0, 0.0]] * 100)
        gold2 = np.array([0] * 75 + [2] * 25)
        assert hierarchy.fit_temperature([block1, (S2, gold2)]) == pytest.approx(1 / LN3, rel=1e-3)

    def test_always_right_pushes_T_towards_zero(self):
        # the top score is always the gold label: the sharper the better. The loss is ~0 (flat) once
        # T < ~0.005, so the optimiser stops somewhere there, above its bound exp(-8) = 3.4e-4
        assert hierarchy.fit_temperature([two_class_block(10, 0)]) < 0.01

    def test_always_wrong_pushes_T_to_its_upper_bound(self):
        # the top score is never the gold label: the flatter the better, until the bound exp(3) = 20
        assert hierarchy.fit_temperature([two_class_block(0, 10)]) > 19

    def test_no_usable_gold_at_all_quirk(self):
        # every gold is -1: the objective is flat, the returned T is arbitrary (but within the bounds)
        S, _ = two_class_block(5, 5)
        T = hierarchy.fit_temperature([(S, np.full(10, -1))])
        assert np.exp(-8) <= T <= np.exp(3)


# ============================================================================= Closure (construction)
class TestClosure:
    def test_nodes_are_the_labels_plus_all_their_ancestors_sorted(self):
        C = hierarchy.Closure(VOCAB, ANC)
        # lake: neither a label nor an ancestor of one; the root's own ancestors are nodes too
        assert C.nodes.tolist() == sorted(VOCAB + [TERRESTRIAL, BIOME, SYSTEM, BFO])
        assert C.nodes[C.label_col].tolist() == VOCAB  # label_col: column of each label in C.nodes

    def test_A_marks_each_label_and_its_ancestors(self):
        C = hierarchy.Closure(VOCAB, ANC)
        row_of_ocean = C.A[VOCAB.index(OCEAN)]
        assert set(C.nodes[row_of_ocean == 1]) == {OCEAN, MARINE, AQUATIC, BIOME, SYSTEM, BFO}
        row_of_soil = C.A[VOCAB.index(SOIL)]
        assert set(C.nodes[row_of_soil == 1]) == {SOIL, TERRESTRIAL, BIOME, SYSTEM, BFO}

    def test_n_below_counts_the_labels_at_or_below_each_node(self):
        C = hierarchy.Closure(VOCAB, ANC)
        n_below = dict(zip(C.nodes, C.n_below))
        assert n_below[OCEAN] == 1        # itself
        assert n_below[MARINE] == 2       # marine, ocean
        assert n_below[AQUATIC] == 4      # aquatic, marine, ocean, freshwater
        assert n_below[TERRESTRIAL] == 5  # its five labels (it is not a label itself)
        assert n_below[BIOME] == 9        # every label

    def test_q_sums_the_probability_at_or_below_each_node(self):
        C = hierarchy.Closure(VOCAB, ANC)
        assert q_of(C, WORKED_EXAMPLE) == {
            OCEAN: .55,
            MARINE: .70,         # .55 ocean + .15 marine
            FRESHWATER: .15,
            AQUATIC: .85,        # .55 + .15 + .15
            SOIL: .15, DESERT: 0, FOREST: 0, GRASSLAND: 0, TUNDRA: 0,
            TERRESTRIAL: .15,
            BIOME: 1.0, SYSTEM: 1.0, BFO: 1.0,
        }

    def test_label_with_two_parents_is_counted_once_higher_up(self):
        # DAG: estuary is_a marine AND is_a freshwater. Going up to aquatic, estuary is reached by two
        # paths but must be counted once, so q(aquatic) = 1, not 1.6.
        estuary = "ENVO_estuary"
        anc = {**ANC, estuary: {MARINE, FRESHWATER} | ANC[MARINE]}
        C = hierarchy.Closure([estuary, MARINE, FRESHWATER], anc)
        q = q_of(C, {estuary: .6, MARINE: .3, FRESHWATER: .1})
        assert q[MARINE] == .9       # .6 estuary + .3 marine
        assert q[FRESHWATER] == .7   # .6 estuary + .1 freshwater
        assert q[AQUATIC] == 1.0     # each label once

    def test_q_never_decreases_going_up(self):
        C = hierarchy.Closure(VOCAB, ANC)
        P = np.random.default_rng(0).dirichlet(np.ones(len(VOCAB)), size=50)  # 50 random probability rows
        q = P @ C.A
        col = {n: j for j, n in enumerate(C.nodes)}
        for label in VOCAB:
            for ancestor in ANC[label]:
                assert np.all(q[:, col[ancestor]] >= q[:, col[label]] - 1e-12)


# ============================================================================= Closure: allowed answers
class TestClosureAllowedAnswers:
    def test_labels_mode_every_label_can_be_an_answer_but_no_other_node(self):
        C = hierarchy.Closure(VOCAB, ANC)  # answers="labels" is the default
        assert allowed(C) == set(VOCAB)    # not terrestrial (not a label), not biome (slot root)

    def test_labels_mode_a_label_above_half_of_the_labels_is_too_vague(self):
        # 4 labels: aquatic has 3 at or below it, marine 2: both >= half (2) -> not answers
        C = hierarchy.Closure([AQUATIC, MARINE, OCEAN, SOIL], ANC)
        assert allowed(C) == {OCEAN, SOIL}

    def test_slot_root_is_never_an_answer(self):
        # biome is a label here and covers only 2 of the 5 labels (less than half): only NO_ANSWER excludes it
        vocab = [BIOME, OCEAN, "UBERON_feces", "UBERON_skin", "PO_leaf"]  # the last three: no ancestors
        C = hierarchy.Closure(vocab, ANC)
        assert BIOME in hierarchy.NO_ANSWER
        assert allowed(C) == {OCEAN, "UBERON_feces", "UBERON_skin", "PO_leaf"}

    def test_two_label_vocabulary_has_no_allowed_answer_quirk(self):
        # each label covers 1 of 2 labels = half -> too vague; a 1- or 2-label slot always abstains
        C = hierarchy.Closure([OCEAN, FRESHWATER], ANC)
        assert allowed(C) == set()
        assert answer(C, {OCEAN: 1.0}, tau=0.5) == (None, 0.0)

    def test_ontology_mode_adds_ancestors_that_are_specific_enough(self):
        # terrestrial (not a label) has 5 of the 9 labels below it
        default = hierarchy.Closure(VOCAB, ANC, answers="ontology")             # floor .25: 5 >= 2.25 -> no
        permissive = hierarchy.Closure(VOCAB, ANC, answers="ontology", floor=0.6)  # 5 < 5.4 -> yes
        assert allowed(default) == set(VOCAB)
        assert allowed(permissive) == set(VOCAB) | {TERRESTRIAL}
        # so a sample that is surely terrestrial but unsure which: abstain vs 'terrestrial'
        land = {SOIL: .4, DESERT: .3, FOREST: .3}
        assert answer(hierarchy.Closure(VOCAB, ANC), land, tau=0.9) == (None, 0.0)
        assert answer(permissive, land, tau=0.9) == (TERRESTRIAL, 1.0)

    def test_ontology_mode_keeps_labels_whatever_their_share(self):
        # same 4 labels as the 'too vague' test: in ontology mode the half rule does not apply to labels
        C = hierarchy.Closure([AQUATIC, MARINE, OCEAN, SOIL], ANC, answers="ontology")
        assert allowed(C) == {AQUATIC, MARINE, OCEAN, SOIL}

    def test_ontology_mode_keeps_labels_from_other_ontologies(self):
        # regression test (fixed 2026-10-19): ontology mode used to require EVERY answer to be ENVO_/UBERON_,
        # which dropped the slot's own PO_ / FOODON_ labels; the ENVO/UBERON condition is for added ancestors
        C = hierarchy.Closure(VOCAB + ["PO_leaf"], ANC, answers="ontology")
        assert "PO_leaf" in allowed(hierarchy.Closure(VOCAB + ["PO_leaf"], ANC))  # labels mode
        assert "PO_leaf" in allowed(C)                                             # ontology mode too
        assert answer(C, {"PO_leaf": .9, SOIL: .1}, tau=0.5) == ("PO_leaf", .9)

    def test_ontology_mode_adds_only_ENVO_or_UBERON_ancestors_below_the_root(self):
        # floor 1: any share, so only the prefix and NO_ANSWER decide. Never added: a PO parent of the PO
        # label, BFO_0000040 (not ENVO), environmental system (ENVO, but a slot root: 9 of the 10 labels)
        anc = {**ANC, "PO_leaf": {"PO_plant_organ"}}
        C = hierarchy.Closure(VOCAB + ["PO_leaf"], anc, answers="ontology", floor=1.0)
        assert {"PO_plant_organ", BFO, SYSTEM} <= set(C.nodes)
        assert allowed(C) == set(VOCAB) | {"PO_leaf", TERRESTRIAL}  # the ENVO non-label ancestor is added


# ============================================================================= Closure.decode
class TestDecode:
    def test_worked_example_of_the_module_docstring(self):
        # q(ocean) .55 < .8, q(marine) .70 < .8, q(aquatic) .85 >= .8 -> aquatic
        assert answer(hierarchy.Closure(VOCAB, ANC), WORKED_EXAMPLE, tau=0.8) == (AQUATIC, .85)

    @pytest.mark.parametrize("tau, expected", [
        (0.01, (OCEAN, .55)),    # tau ~ 0: the plain top-1 label
        (0.55, (OCEAN, .55)),    # q == tau counts as reaching tau
        (0.60, (MARINE, .70)),   # 0.80 -> aquatic: test_worked_example_of_the_module_docstring
        (0.90, (None, 0.0)),     # only biome (q = 1) is left, and it is never an answer -> abstain
    ])
    def test_higher_tau_climbs_higher(self, tau, expected):
        assert answer(hierarchy.Closure(VOCAB, ANC), WORKED_EXAMPLE, tau) == expected

    def test_climb_stays_on_the_path_of_the_top1_label(self):
        # top-1 is soil (.30), but 70 % of the probability is aquatic. The climb only looks at soil and
        # its ancestors (terrestrial: not a label; biome: root), so it abstains instead of answering aquatic.
        row = {SOIL: .30, OCEAN: .25, MARINE: .25, FRESHWATER: .20}
        C = hierarchy.Closure(VOCAB, ANC)
        assert q_of(C, row)[AQUATIC] == .70
        assert answer(C, row, tau=0.5) == (None, 0.0)

    def test_top1_label_that_is_too_vague_abstains(self):
        # 4 labels: aquatic covers 3 of them, so it is not an allowed answer (see TestClosureAllowedAnswers)
        C = hierarchy.Closure([AQUATIC, MARINE, OCEAN, SOIL], ANC)
        assert answer(C, {AQUATIC: .9, SOIL: .1}, tau=0.5) == (None, 0.0)
        assert answer(C, {OCEAN: .9, SOIL: .1}, tau=0.5) == (OCEAN, .9)

    def test_several_rows_at_once_pick_indexes_nodes_and_minus_1_abstains(self):
        C = hierarchy.Closure(VOCAB, ANC)
        P = probs({OCEAN: .9, MARINE: .1},   # sure: the top-1 itself
                  WORKED_EXAMPLE,            # unsure: climbs to aquatic
                  {SOIL: .6, DESERT: .4})    # unsure, nothing allowed above soil: abstain
        pick, q = C.decode(P, tau=0.8)
        assert pick[2] == -1
        assert C.nodes[pick[:2]].tolist() == [OCEAN, AQUATIC]  # indices into C.nodes, NOT into VOCAB
        assert q == pytest.approx([.9, .85, 0.0])

    # DAG: estuary is_a marine AND is_a freshwater (same depth). Closure.decode ranks the candidate nodes by
    # fewest labels at or below them, then label, then depth, then q (decode_candidates differs: see there).
    ESTUARY = "ENVO_estuary"
    DAG = {**ANC, ESTUARY: {MARINE, FRESHWATER} | ANC[MARINE]}

    def test_two_equally_specific_parents_the_more_probable_first(self):
        # no ocean label: marine and freshwater each have 2 labels at or below them (itself + estuary)
        C = hierarchy.Closure([self.ESTUARY, MARINE, FRESHWATER, SOIL, DESERT, FOREST, GRASSLAND, TUNDRA], self.DAG)
        # estuary .4 < .6; both parents reach .6: the more probable wins, whatever the id order
        assert answer(C, {self.ESTUARY: .4, MARINE: .35, FRESHWATER: .25}, tau=0.6) == (MARINE, .75)
        assert answer(C, {self.ESTUARY: .4, MARINE: .25, FRESHWATER: .35}, tau=0.6) == (FRESHWATER, .75)

    def test_parent_with_fewer_labels_below_first(self):
        # with ocean: marine covers 3 labels, freshwater 2 -> freshwater, although marine is more probable
        C = hierarchy.Closure(VOCAB + [self.ESTUARY], self.DAG)
        assert answer(C, {self.ESTUARY: .4, OCEAN: .35, FRESHWATER: .25}, tau=0.6) == (FRESHWATER, .65)


# ============================================================================= relation / outcome
NODES = np.array([OCEAN, MARINE, AQUATIC, FRESHWATER, SOIL, BIOME])


class TestRelation:
    def test_each_node_compared_with_the_gold_label(self):
        rel = hierarchy.relation(NODES, [MARINE], ANC)
        #                       ocean         marine  aquatic  freshwater  soil   biome
        assert rel[0].tolist() == [TOO_SPECIFIC, EXACT, COARSER, OTHER,      OTHER, COARSER]

    def test_gold_that_is_not_a_node(self):
        # lake is no label (so never an answer) but can be a gold label; its ancestors are still COARSER
        rel = hierarchy.relation(NODES, [LAKE], ANC)
        #                       ocean  marine  aquatic  freshwater  soil   biome
        assert rel[0].tolist() == [OTHER, OTHER, COARSER, COARSER,    OTHER, COARSER]

    def test_gold_with_two_parents(self):
        # DAG (25 of the 43 real biome labels have several parents): estuary is_a marine AND is_a freshwater,
        # so both are COARSER for an estuary gold, and estuary is TOO_SPECIFIC for a marine or freshwater gold
        estuary = "ENVO_estuary"
        dag = {**ANC, estuary: {MARINE, FRESHWATER} | ANC[MARINE]}
        rel = hierarchy.relation(np.array([estuary, MARINE, FRESHWATER, OCEAN]), [estuary, MARINE, FRESHWATER], dag)
        #                         estuary       marine   freshwater  ocean
        assert rel.tolist() == [[EXACT,        COARSER, COARSER,    OTHER],          # gold estuary
                                [TOO_SPECIFIC, EXACT,   OTHER,      TOO_SPECIFIC],   # gold marine
                                [TOO_SPECIFIC, OTHER,   EXACT,      OTHER]]          # gold freshwater

    def test_one_row_per_sample(self):
        rel = hierarchy.relation(NODES, [MARINE, SOIL, MARINE], ANC)
        assert rel.shape == (3, len(NODES))
        assert rel[0].tolist() == rel[2].tolist()            # same gold, same row
        assert rel[1, list(NODES).index(SOIL)] == EXACT

    def test_no_samples(self):
        assert hierarchy.relation(NODES, [], ANC).shape == (0, len(NODES))


class TestOutcome:
    def test_code_of_the_picked_node_and_0_for_abstain(self):
        rel = hierarchy.relation(NODES, [MARINE, SOIL], ANC)
        # sample 0 (gold marine) answered ocean (node 0); sample 1 (gold soil) abstained
        assert hierarchy.outcome(rel, np.array([0, -1])).tolist() == [TOO_SPECIFIC, 0]
        # sample 0 answered aquatic (node 2); sample 1 answered soil (node 4)
        assert hierarchy.outcome(rel, np.array([2, 4])).tolist() == [COARSER, EXACT]


# ============================================================================= summarise
class TestSummarise:
    def test_example_of_the_docstring(self):
        # exact, coarser, abstain, other branch
        out = np.array([EXACT, COARSER, 0, OTHER])
        assert hierarchy.summarise(out) == {
            "coverage": 0.75,            # 3 of 4 answered
            "accuracy_strict": 0.6667,   # exact + coarser among the 3 answered
            "accuracy_lenient": 0.6667,  # no too-specific answers here
            "exact": 0.25, "coarser": 0.25, "too_specific": 0.0, "other_branch": 0.25,  # shares of ALL 4
        }

    def test_lenient_accuracy_also_accepts_too_specific(self):
        out = np.array([EXACT, TOO_SPECIFIC, TOO_SPECIFIC, OTHER])
        s = hierarchy.summarise(out)
        assert (s["accuracy_strict"], s["accuracy_lenient"]) == (0.25, 0.75)

    def test_all_abstain_gives_no_accuracy(self):
        s = hierarchy.summarise(np.array([0, 0, 0]))
        assert s["coverage"] == 0.0
        assert s["accuracy_strict"] is None and s["accuracy_lenient"] is None

    def test_no_samples_quirk(self):
        # never happens in the pipeline; coverage is the mean of an empty array: nan, with a warning
        with pytest.warns(RuntimeWarning):
            s = hierarchy.summarise(np.array([], dtype=int))
        assert np.isnan(s["coverage"]) and s["accuracy_strict"] is None


# ============================================================================= choose_tau
CURVE = [
    {"tau": 0.5, "accuracy_strict": 0.70, "accuracy_lenient": 0.80},
    {"tau": 0.6, "accuracy_strict": 0.85, "accuracy_lenient": 0.90},
    {"tau": 0.7, "accuracy_strict": None, "accuracy_lenient": None},  # nothing answered at this tau
    {"tau": 0.8, "accuracy_strict": 0.95, "accuracy_lenient": 0.97},
    {"tau": 1.0, "accuracy_strict": None, "accuracy_lenient": None},
]


class TestChooseTau:
    @pytest.mark.parametrize("target, metric, expected", [
        (0.60, "accuracy_strict", 0.5),   # already reached at the lowest tau
        (0.85, "accuracy_strict", 0.6),   # reached exactly counts
        (0.90, "accuracy_strict", 0.8),   # the None row at 0.7 is skipped
        (0.90, "accuracy_lenient", 0.6),  # lenient accuracy is higher -> lower tau
        (0.99, "accuracy_strict", 1.0),   # never reached -> the highest tau (in practice: abstain on almost all)
    ])
    def test_lowest_tau_reaching_the_target(self, target, metric, expected):
        assert hierarchy.choose_tau(CURVE, target, metric) == expected


# ============================================================================= oof_backoff (shape only)
def tiny_fold():
    """Three samples; their scores are log-probabilities (see toy_hierarchy.log_scores)."""
    return {"S": log_scores({OCEAN: .9, MARINE: .1}, {SOIL: .8, DESERT: .2}, {FRESHWATER: .7, OCEAN: .3}),
            "vocab": VOCAB, "gold": [OCEAN, SOIL, FRESHWATER]}


class TestOofBackoff:
    # what the numbers mean is tested in test_hierarchy_integration.py; here only the shape of the output
    def test_report_and_fitted_models(self):
        report, fitted = hierarchy.oof_backoff([tiny_fold(), tiny_fold()], ANC, targets=(0.9,))
        assert set(report) == {"temperature_per_fold", "temperature_all", "curve", "strict_0.9", "lenient_0.9"}
        assert len(report["temperature_per_fold"]) == 2
        assert [row["tau"] for row in report["curve"]] == list(hierarchy.TAUS)
        assert set(report["strict_0.9"]) == {"tau_per_fold", "coverage", "accuracy_strict", "accuracy_lenient",
                                             "exact", "coarser", "too_specific", "other_branch"}
        assert len(fitted) == 2
        P, C = fitted[0]  # calibrated probabilities of fold 0 and the Closure to decode them with
        assert P.shape == (3, len(VOCAB)) and isinstance(C, hierarchy.Closure)

    def test_needs_at_least_two_folds(self):
        # the temperature and tau of a fold are chosen on the OTHER folds: with one fold there are none
        with pytest.raises(ValueError):
            hierarchy.oof_backoff([tiny_fold()], ANC)


# ============================================================================= decode_candidates
class TestDecodeCandidates:
    CANDS, P = [OCEAN, FRESHWATER, MARINE], [.55, .30, .15]

    @pytest.mark.parametrize("tau, expected", [
        (0.50, (OCEAN, .55)),
        (0.60, (MARINE, .70)),    # .55 ocean + .15 marine
        (0.90, (AQUATIC, 1.0)),   # all three; aquatic need not be a candidate, only an allowed label
    ])
    def test_climbs_like_closure_decode(self, tau, expected):
        node, q = hierarchy.decode_candidates(self.CANDS, self.P, tau, set(VOCAB), ANC)
        assert (node, round(q, 4)) == expected

    def test_starts_from_the_most_probable_candidate_not_the_first(self):
        node, _ = hierarchy.decode_candidates([MARINE, OCEAN], [.3, .7], 0.5, set(VOCAB), ANC)
        assert node == OCEAN

    def test_ancestors_that_are_not_allowed_are_skipped(self):
        # marine is not a label of this slot -> straight from ocean to aquatic
        node, q = hierarchy.decode_candidates(self.CANDS, self.P, 0.6, set(VOCAB) - {MARINE}, ANC)
        assert (node, round(q, 4)) == (AQUATIC, 1.0)

    def test_slot_root_is_never_an_answer(self):
        # terrestrial is not allowed and biome is the root, even when listed in `allowed`
        assert hierarchy.decode_candidates([SOIL, DESERT], [.6, .4], 0.8, set(VOCAB) | {BIOME}, ANC) == (None, 0.0)

    def test_probability_outside_the_candidates_is_not_counted(self):
        # the candidates hold 80 %; the other 20 % (classes not shown to the reranker) is not counted
        node, q = hierarchy.decode_candidates([OCEAN, FRESHWATER], [.5, .3], 0.85, set(VOCAB), ANC)
        assert (node, q) == (None, 0.0)

    def test_top_candidate_is_not_checked_against_allowed_quirk(self):
        # harmless in 7_rerank_atlas.py: candidates are model classes, which are all allowed
        node, _ = hierarchy.decode_candidates([LAKE, OCEAN], [.6, .4], 0.5, set(VOCAB), ANC)
        assert node == LAKE

    # Regression tests (fixed 2026-10-19). estuary is_a marine AND is_a freshwater, both at depth 2. When both
    # reach tau, the answer used to be whichever came first in anc[estuary], a set whose iteration order
    # changes with PYTHONHASHSEED. Now: the more probable parent first, then the smaller id.
    # Lists in `anc` make the iteration order explicit, so both orders are tried.
    ESTUARY = "ENVO_estuary"
    ANC_ORDERS = [{**ANC, ESTUARY: [MARINE, FRESHWATER, AQUATIC, BIOME, SYSTEM, BFO]},
                  {**ANC, ESTUARY: [FRESHWATER, MARINE, AQUATIC, BIOME, SYSTEM, BFO]}]

    def test_equal_depth_parents_the_more_probable_first(self):
        cands, p = [self.ESTUARY, OCEAN, FRESHWATER], [.4, .35, .25]
        # estuary .4 < .6; q(marine) = .4 + .35 (ocean) = .75; q(freshwater) = .4 + .25 = .65: both >= .6
        # (Closure.decode answers freshwater here: fewer labels below it; TestDecode::test_parent_with_fewer_labels_below_first)
        for anc in self.ANC_ORDERS:
            node, q = hierarchy.decode_candidates(cands, p, 0.6, set(VOCAB) | {self.ESTUARY}, anc)
            assert (node, round(q, 4)) == (MARINE, .75)

    def test_equal_depth_parents_equally_probable_by_id(self):
        cands, p = [self.ESTUARY, OCEAN, FRESHWATER], [.4, .3, .3]
        # q(marine) = q(freshwater) = .7: 'ENVO_freshwater' < 'ENVO_marine'
        for anc in self.ANC_ORDERS:
            node, _ = hierarchy.decode_candidates(cands, p, 0.6, set(VOCAB) | {self.ESTUARY}, anc)
            assert node == FRESHWATER

    def test_same_answer_whatever_the_hash_seed(self):
        # the real situation: anc values are sets, their order depends on PYTHONHASHSEED (before the fix,
        # seeds 0..5 gave marine 4 times and freshwater 2 times)
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import hierarchy\n"
                "B = 'ENVO_00000428'\n"
                "anc = {'e': {'ENVO_m', 'ENVO_f', 'ENVO_a', B}, 'o': {'ENVO_m', 'ENVO_a', B},\n"
                "       'ENVO_m': {'ENVO_a', B}, 'ENVO_f': {'ENVO_a', B}, 'ENVO_a': {B}}\n"
                "print(hierarchy.decode_candidates(['e', 'o', 'ENVO_f'], [.4, .3, .3], .6,\n"
                "      {'e', 'o', 'ENVO_m', 'ENVO_f', 'ENVO_a'}, anc)[0])")
        answers = {subprocess.run([sys.executable, "-c", code, str(SCRIPTS_DIR)], capture_output=True, text=True,
                                  env=dict(os.environ, PYTHONHASHSEED=str(seed)), check=True).stdout.strip()
                   for seed in range(6)}
        assert answers == {"ENVO_f"}


# ============================================================================= merge_calibrations
def curve_row(tau, exact, coarser, too_specific, other):
    """A calibration curve row; coverage and accuracies follow from the four shares."""
    coverage = exact + coarser + too_specific + other
    return {"tau": tau, "coverage": coverage, "exact": exact, "coarser": coarser, "too_specific": too_specific,
            "other_branch": other, "accuracy_strict": (exact + coarser) / coverage if coverage else None,
            "accuracy_lenient": (exact + coarser + too_specific) / coverage if coverage else None}


def calibration(temperature, curve, fold_seed):
    """One calibration.json entry (calibration[slot][method]) as 5_evaluate.py writes it."""
    return {"temperature": temperature, "curve": curve, "settings": {"method": "prototype", "fold_seed": fold_seed}}


class TestMergeCalibrations:
    def test_one_entry_is_returned_unchanged(self):
        entry = calibration(0.1, [curve_row(0.5, .5, .3, 0, .2)], fold_seed=0)
        assert hierarchy.merge_calibrations([entry]) is entry

    def test_shares_are_averaged_and_accuracy_recomputed_from_them(self):
        seed0 = calibration(0.1, [curve_row(0.5, exact=.5, coarser=.3, too_specific=0, other=.2)], fold_seed=0)
        seed1 = calibration(0.4, [curve_row(0.5, exact=.5, coarser=0, too_specific=0, other=0)], fold_seed=1)
        assert (seed0["curve"][0]["accuracy_strict"], seed1["curve"][0]["accuracy_strict"]) == (0.8, 1.0)
        merged = hierarchy.merge_calibrations([seed0, seed1])
        row = merged["curve"][0]
        assert (row["coverage"], row["exact"], row["coarser"], row["other_branch"]) == (.75, .5, .15, .1)
        # (.5 + .15) / .75 = .8667: pooled samples, NOT the mean of the two accuracies (.9)
        assert row["accuracy_strict"] == 0.8667

    def test_temperature_is_the_geometric_mean(self):
        merged = hierarchy.merge_calibrations([calibration(0.1, [curve_row(0.5, 1, 0, 0, 0)], 0),
                                               calibration(0.4, [curve_row(0.5, 1, 0, 0, 0)], 1)])
        assert merged["temperature"] == pytest.approx(0.2)  # sqrt(.1 * .4)

    def test_settings_of_the_first_entry_with_every_fold_seed(self):
        merged = hierarchy.merge_calibrations([calibration(0.1, [curve_row(0.5, 1, 0, 0, 0)], 0),
                                               calibration(0.1, [curve_row(0.5, 1, 0, 0, 0)], 7)])
        assert merged["settings"] == {"method": "prototype", "fold_seed": [0, 7]}

    def test_p_top1_quantiles_are_dropped_quirk(self):
        # 5_evaluate.py also writes p_top1_quantiles; a merged entry has none. Harmless: 6_predict_atlas.py
        # does not use them, and 7_rerank_atlas.py (its gate) reads one calibration.json without merging
        entries = [{**calibration(0.1, [curve_row(0.5, 1, 0, 0, 0)], seed), "p_top1_quantiles": {"0.3": 0.5}}
                   for seed in (0, 1)]
        assert set(hierarchy.merge_calibrations(entries)) == {"temperature", "curve", "settings"}

    def test_tau_where_nothing_is_answered_has_no_accuracy(self):
        merged = hierarchy.merge_calibrations([calibration(0.1, [curve_row(1.0, 0, 0, 0, 0)], 0),
                                               calibration(0.1, [curve_row(1.0, 0, 0, 0, 0)], 1)])
        assert merged["curve"][0]["accuracy_strict"] is None

    def test_different_tau_grids_are_refused(self):
        with pytest.raises(SystemExit):
            hierarchy.merge_calibrations([calibration(0.1, [curve_row(0.5, 1, 0, 0, 0)], 0),
                                          calibration(0.1, [curve_row(0.6, 1, 0, 0, 0)], 1)])
