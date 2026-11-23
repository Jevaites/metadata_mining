"""Unit tests of experiments/set_valued.py (set-valued decoding over the DAG): one class per function.

Most tests use the toy ontology of toy_hierarchy.py (aquatic > marine > ocean, aquatic > freshwater, and five
terrestrial labels under a non-label 'terrestrial'); set_table uses a hand-made closure matrix with one label under
two parents (a DAG), which the toy tree cannot show.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import set_valued as sv  # noqa: E402
from hierarchy import Closure  # noqa: E402
from toy_hierarchy import (ANC, AQUATIC, DESERT, FOREST, FRESHWATER, MARINE, OCEAN, SOIL, TERRESTRIAL,  # noqa: E402
                           VOCAB, probs)


def closure():
    C = Closure(VOCAB, ANC)
    return C, C.A > 0, list(C.nodes), np.asarray(C.informative)


def named(sets, nodes):
    return [tuple(nodes[k] for k in s) for s in sets]


# ----------------------------------------------------------------------------- oof_probabilities
class TestOofProbabilities:
    def test_temperature_of_a_fold_is_fitted_on_the_other_folds_only(self, monkeypatch):
        seen = []
        monkeypatch.setattr(sv, "fit_temperature", lambda blocks: seen.append(len(blocks)) or 0.5)
        rng = np.random.default_rng(0)
        samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "study_code": list("aabbcc"),
                                "biome": [OCEAN, SOIL] * 3, "feature": [""] * 6, "material": [""] * 6})
        x = rng.normal(size=(6, 4))
        vec = {OCEAN: np.eye(4)[0], SOIL: np.eye(4)[1]}
        folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])), (np.array([0, 1, 4, 5]), np.array([2, 3])),
                 (np.array([0, 1, 2, 3]), np.array([4, 5]))]
        out = sv.oof_probabilities(samples, x, vec, folds)
        assert seen[:3] == [2, 2, 2]  # biome: each of the 3 folds fitted on the other 2 folds' scores
        f = out["biome"][0]
        assert f["T"] == 0.5 and f["P"].shape == (2, 2) and np.allclose(f["P"].sum(1), 1)
        assert f["sample_id"].tolist() == ["s0", "s1"]
        assert [g["P"].shape for g in out["feature"]] == [(0, 0)] * 3  # an unlabelled slot gives empty folds


# ----------------------------------------------------------------------------- candidate_sets
class TestCandidateSets:
    def test_only_allowed_nodes_with_enough_mass(self):
        C, A, nodes, inf = closure()
        p = probs({OCEAN: .5, SOIL: .3, DESERT: .19, FOREST: .01})[0]
        sets = named(sv.candidate_sets(p @ C.A, A, inf, n_candidates=10, max_terms=1, min_mass=0.02), nodes)
        # terrestrial (not a label) and the root are never candidates; forest (0.01) is below min_mass
        assert set(sets) == {(OCEAN,), (MARINE,), (AQUATIC,), (SOIL,), (DESERT,)}
        assert sets[-2:] == [(SOIL,), (DESERT,)]  # most mass first (the three 0.5 nodes tie, then 0.3, 0.19)

    def test_n_candidates_keeps_the_heaviest(self):
        C, A, nodes, inf = closure()
        p = probs({OCEAN: .45, FRESHWATER: .05, SOIL: .3, DESERT: .2})[0]
        # aquatic .5 > marine = ocean .45 > soil .3: the 2 heaviest are aquatic and one of the .45 pair
        sets = named(sv.candidate_sets(p @ C.A, A, inf, n_candidates=2, max_terms=1), nodes)
        assert sets[0] == (AQUATIC,) and sets[1] in [(MARINE,), (OCEAN,)] and len(sets) == 2

    def test_sets_never_hold_a_term_and_its_ancestor(self):
        C, A, nodes, inf = closure()
        p = probs({OCEAN: .4, FRESHWATER: .3, SOIL: .3})[0]
        pairs = {frozenset(s) for s in named(sv.candidate_sets(p @ C.A, A, inf, max_terms=2), nodes) if len(s) == 2}
        for a, b in map(tuple, pairs):
            assert a not in ANC[b] and b not in ANC[a], (a, b)
        assert {frozenset({OCEAN, SOIL}), frozenset({OCEAN, FRESHWATER}), frozenset({AQUATIC, SOIL})} <= pairs
        assert frozenset({OCEAN, MARINE}) not in pairs and frozenset({AQUATIC, OCEAN}) not in pairs

    def test_min_mass_is_inclusive(self):
        C, A, nodes, inf = closure()
        p = probs({OCEAN: .5, SOIL: .48, DESERT: .02})[0]
        assert (DESERT,) in named(sv.candidate_sets(p @ C.A, A, inf, max_terms=1, min_mass=0.02), nodes)

    def test_max_terms(self):
        C, A, nodes, inf = closure()
        p = probs({OCEAN: .3, FRESHWATER: .3, SOIL: .2, DESERT: .2})[0]
        sizes = {len(s) for s in sv.candidate_sets(p @ C.A, A, inf, max_terms=3)}
        assert sizes == {1, 2, 3}
        assert {len(s) for s in sv.candidate_sets(p @ C.A, A, inf, max_terms=2)} == {1, 2}

    def test_no_candidate_when_mass_is_spread_thin(self):
        C, A, nodes, inf = closure()
        p = probs({t: 1 / 9 for t in VOCAB})[0]
        # every label has 0.11 >= 0.02: candidates exist; with min_mass 0.5 only aquatic (0.44) fails too
        assert sv.candidate_sets(p @ C.A, A, inf, max_terms=2, min_mass=0.5) == []


# ----------------------------------------------------------------------------- set_table
class TestSetTable:
    # labels x, y, z; node 0 = parent P1 of x and y, node 1 = parent P2 of x and z (x has two parents)
    A = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)

    def test_a_label_under_both_nodes_counts_once(self):
        p = np.array([.5, .3, .2])
        t = sv.set_table(p, [(0,), (1,), (0, 1)], self.A, np.array([.5, .25, .25]))
        assert t["mass"].tolist() == pytest.approx([.8, .7, 1.0])  # not .8 + .7
        assert t["n_labels"].tolist() == [2, 2, 3]
        assert t["n_terms"].tolist() == [1, 1, 2]

    def test_information_content_and_expected_credit(self):
        p, share = np.array([.5, .3, .2]), np.array([.5, .25, .25])
        t = sv.set_table(p, [(0,)], self.A, share)
        ic = -np.log(.75)
        assert t["ic"][0] == pytest.approx(ic)
        # E[credit] = sum over covered labels of p_y * min(1, IC(set) / IC(y))
        assert t["expected_credit"][0] == pytest.approx(.5 * ic / -np.log(.5) + .3 * ic / -np.log(.25))


# ----------------------------------------------------------------------------- decision rules
def table(mass, n_labels, n_terms, expected_credit=None):
    return {"mass": np.array(mass, float), "n_labels": np.array(n_labels), "n_terms": np.array(n_terms),
            "expected_credit": np.array(expected_credit if expected_credit is not None else mass, float)}


class TestConformalPick:
    def test_smallest_set_reaching_tau(self):
        t = table(mass=[.6, .85, .95, .9], n_labels=[1, 4, 9, 2], n_terms=[1, 1, 1, 2])
        assert sv.conformal_pick(t, .8, max_terms=2) == 3  # 2 labels beat 4 and 9

    def test_tie_on_size_goes_to_more_mass(self):
        t = table(mass=[.82, .9], n_labels=[2, 2], n_terms=[1, 2])
        assert sv.conformal_pick(t, .8, max_terms=2) == 1

    def test_max_terms_and_abstention(self):
        t = table(mass=[.6, .95], n_labels=[1, 2], n_terms=[1, 2])
        assert sv.conformal_pick(t, .8, max_terms=1) == -1
        assert sv.conformal_pick(t, .99, max_terms=2) == -1

    def test_tau_is_inclusive(self):
        assert sv.conformal_pick(table([.8], [1], [1]), .8, 1) == 0


class TestExpectedValuePick:
    def test_lambda_zero_takes_the_most_expected_credit(self):
        t = table(mass=[.5, .9], n_labels=[1, 2], n_terms=[1, 2], expected_credit=[.5, .6])
        assert sv.expected_value_pick(t, 0, 2) == 1

    def test_penalty_on_wrong_answers_favours_the_safer_set(self):
        # single: E[credit] .55, P(wrong) .45; pair: E[credit] .5, P(wrong) .05
        t = table(mass=[.55, .95], n_labels=[1, 2], n_terms=[1, 2], expected_credit=[.55, .5])
        assert sv.expected_value_pick(t, 0.05, 2) == 0
        assert sv.expected_value_pick(t, 1.0, 2) == 1
        assert sv.expected_value_pick(t, 1.0, 1) == 0  # singles only: .55 - .45 = .10 > 0, still answers

    def test_abstains_when_no_set_has_positive_value(self):
        t = table(mass=[.3, .5], n_labels=[1, 2], n_terms=[1, 2], expected_credit=[.3, .2])
        assert sv.expected_value_pick(t, 2.0, 2) == -1
        assert sv.expected_value_pick(table([0.0], [1], [1], [0.0]), 0.0, 1) == -1  # value exactly 0: abstain


# ----------------------------------------------------------------------------- cross-fitting
class TestCrossFit:
    def test_threshold_comes_from_the_other_folds(self):
        # 2 folds x 2 samples, grid of 2. Fold 0's own samples would allow grid 0, fold 1's do not.
        fold = np.array([0, 0, 1, 1])
        answered = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
        covered = np.array([[1, 0], [1, 1], [0, 0], [1, 1]])
        per_sample, chosen = sv.cross_fit(answered, covered, fold, target=0.9)
        # for fold 0, judged on fold 1: grid 0 has 1/2 right -> grid 1 (1/1); for fold 1, judged on fold 0: grid 0
        assert chosen == {0: 1, 1: 0}
        assert per_sample.tolist() == [1, 1, 0, 0]

    def test_most_answers_among_feasible_ties_to_the_first(self):
        fold = np.array([0, 1, 1, 1])
        answered = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]])
        covered = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0]])
        assert sv.cross_fit(answered, covered, fold, 0.9)[1][0] == 0

    def test_no_feasible_threshold_takes_the_last(self):
        fold = np.array([0, 1])
        answered, covered = np.array([[1, 1], [1, 1]]), np.array([[0, 0], [0, 0]])
        assert sv.cross_fit(answered, covered, fold, 0.9)[1] == {0: 1, 1: 1}


class TestFallback:
    def test_only_abstentions_change_and_threshold_from_other_folds_abstentions(self):
        fold = np.array([0, 0, 1, 1])
        primary = np.array([1, 0, 1, 0])
        answered = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
        covered = np.array([[0, 0], [1, 1], [0, 0], [1, 0]])  # rows 0 / 2 (answered by primary) are ignored
        per_sample, chosen = sv.fallback(primary, answered, covered, fold, 0.9)
        assert per_sample[[0, 2]].tolist() == [-1, -1]
        # fold 0 judged on fold 1's abstention (row 3): grid 0 right -> 0; fold 1 judged on row 1: both right -> 0
        assert chosen == {0: 0, 1: 0}
        assert per_sample[[1, 3]].tolist() == [0, 0]

    def test_infeasible_keeps_abstaining(self):
        fold = np.array([0, 1])
        per_sample, chosen = sv.fallback(np.array([0, 0]), np.array([[1], [1]]), np.array([[0], [0]]), fold, 0.9)
        assert chosen == {0: None, 1: None} and per_sample.tolist() == [-1, -1]


# ----------------------------------------------------------------------------- decode_slot on hand-made probabilities
class TestDecodeSlot:
    """2 folds x 20 samples: 14 confident ocean / soil samples (0.95 on their label), 6 split ones (ocean .5, soil
    .45; gold alternates). On a split sample the top-1 path (ocean, marine, aquatic) never reaches 0.5 + and
    'terrestrial' is not an answer, so the back-off can only answer it below the 90 % target; {ocean, soil} holds
    0.95 and always covers gold."""

    def folds(self):
        out = []
        for k in range(2):
            rows, gold = [], []
            for i in range(14):
                g = OCEAN if i % 2 else SOIL
                rows.append({g: .95, DESERT: .05})
                gold.append(g)
            for i in range(6):
                rows.append({OCEAN: .5, SOIL: .45, DESERT: .05})
                gold.append(OCEAN if i % 2 else SOIL)
            out.append({"P": probs(*rows), "vocab": np.array(VOCAB), "gold": np.array(gold),
                        "sample_id": np.array([f"f{k}_{i}" for i in range(20)]),
                        "study": np.array([f"study{k}_{i // 2}" for i in range(20)])})
        return out

    def run(self, **kw):
        args = type("Args", (), {"n_candidates": 10, "max_terms": 2, "min_mass": 0.02, "target": 0.9,
                                 "discounts": [0.7, 1.0], "answers": "labels", "flag_top": 5, **kw})
        return sv.decode_slot(self.folds(), ANC, args)[:4]

    def test_back_off_answers_confident_samples_and_abstains_on_split_ones(self):
        rows, answers, outcomes, thresholds = self.run()
        split = np.array([i % 20 >= 14 for i in range(40)])
        assert all(a == [] for a, s in zip(answers["backoff"], split) if s)
        assert all(len(a) == 1 for a, s in zip(answers["backoff"], split) if not s)

    def test_sets_answer_the_split_samples_and_keep_the_rest(self):
        rows, answers, outcomes, thresholds = self.run()
        split = np.array([i % 20 >= 14 for i in range(40)])
        for name in ["conformal_m2", "ev_m2", "backoff+set", "ev1+ev2set", "evr_d0.7"]:
            assert all(set(a) == {OCEAN, SOIL} for a, s in zip(answers[name], split) if s), name
            assert all(len(a) == 1 for a, s in zip(answers[name], split) if not s), name
        assert (outcomes["backoff+set"][:, 1] == 1).all()  # every answer covers gold
        assert [a for a, s in zip(answers["backoff+set"], split) if not s] == \
               [a for a, s in zip(answers["backoff"], split) if not s]

    def test_top1_and_thresholds(self):
        rows, answers, outcomes, thresholds = self.run()
        assert all(len(a) == 1 for a in answers["top1"])
        assert set(thresholds) == {"backoff", "conformal_m1", "ev_m1", "conformal_m2", "ev_m2", "backoff+set",
                                   "backoff+set@0.8", "ev1+ev2set", "evr_d0.7", "evr_d1"}
        assert set(thresholds["backoff"]) == {0, 1}

    def test_reviewer_score_without_discount_answers_sets_everywhere(self):
        # d = 1: an extra term never lowers the score, so even confident samples get {label, desert}
        rows, answers, outcomes, thresholds = self.run()
        assert all(len(a) == 2 for a in answers["evr_d1"])
        assert all(set(a) == {OCEAN, SOIL} for i, a in enumerate(answers["evr_d1"]) if i % 20 >= 14)

    def test_answers_are_ordered_by_mass(self):
        rows, answers, outcomes, thresholds = self.run()
        split = [a for i, a in enumerate(answers["conformal_m2"]) if i % 20 >= 14]
        assert split and all(a == [OCEAN, SOIL] for a in split)  # ocean .5 before soil .45

    def test_max_terms_one_has_no_sets(self):
        rows, answers, outcomes, thresholds = self.run(max_terms=1)
        assert set(answers) == {"top1", "backoff", "conformal_m1", "ev_m1"}
        assert all(len(a) <= 1 for v in answers.values() for a in v)


# ----------------------------------------------------------------------------- reviewer score
class TestReviewerCredit:
    A = TestSetTable.A  # x under nodes 0 and 1, y under 0, z under 1

    def test_a_set_is_worth_its_best_term_per_label(self):
        p, share = np.array([.5, .3, .2]), np.array([.5, .25, .25])
        t = sv.set_table(p, [(0, 1)], self.A, share)
        ic_node, ic_x, ic_y = -np.log(.75), -np.log(.5), -np.log(.25)
        # x: best of the two terms (both IC .29) .29/.69; y: only node 0 covers it; z: only node 1
        expected = .5 * ic_node / ic_x + .3 * ic_node / ic_y + .2 * ic_node / ic_y
        assert t["reviewer_credit"][0] == pytest.approx(expected)
        assert t["expected_credit"][0] == pytest.approx(0.0)  # the union covers every label: IC 0, credit 0

    def test_single_terms_score_as_the_credit_score(self):
        p, share = np.array([.5, .3, .2]), np.array([.5, .25, .25])
        t = sv.set_table(p, [(0,), (1,)], self.A, share)
        assert t["reviewer_credit"] == pytest.approx(t["expected_credit"])


class TestReviewerValuePick:
    # single: reviewer credit .5, mass .55; pair: reviewer credit .6, mass .95
    t = {"reviewer_credit": np.array([.5, .6]), "mass": np.array([.55, .95]), "n_terms": np.array([1, 2])}

    def test_discount_one_takes_the_pair(self):
        assert sv.reviewer_value_pick(self.t, 1.0, 2, 1.0) == 1  # .5 - .45 vs .6 - .05

    def test_small_discount_keeps_the_single_term(self):
        assert sv.reviewer_value_pick(self.t, 1.0, 2, 0.05) == 0  # pair .03 - .05 < 0, single .05

    def test_max_terms_and_abstention(self):
        assert sv.reviewer_value_pick(self.t, 1.0, 1, 1.0) == 0
        assert sv.reviewer_value_pick(self.t, 20.0, 2, 1.0) == -1


# ----------------------------------------------------------------------------- flags and masses
class TestUnrelatedRunnerUp:
    def test_skips_ancestors_and_labels_sharing_an_allowed_ancestor(self):
        vocab = np.array(VOCAB)
        p = probs({OCEAN: .45, MARINE: .25, FRESHWATER: .2, SOIL: .1})[0]
        # marine: ancestor of ocean; freshwater: shares the label 'aquatic' with ocean -> soil
        assert sv.unrelated_runner_up(p, vocab, ANC, {AQUATIC, MARINE, OCEAN, FRESHWATER, SOIL}) == (SOIL, pytest.approx(.1))
        # if aquatic is not an allowed answer, freshwater is a separate branch
        assert sv.unrelated_runner_up(p, vocab, ANC, {OCEAN, FRESHWATER, SOIL})[0] == FRESHWATER

    def test_none_within_top(self):
        vocab = np.array(VOCAB)
        p = probs({OCEAN: .6, MARINE: .3, AQUATIC: .1})[0]
        assert sv.unrelated_runner_up(p, vocab, ANC, set(), top=2) == ("", 0.0)


class TestAnswerMasses:
    def row(self):
        A = TestSetTable.A
        return {"p": np.array([.5, .3, .2]), "A": A, "col": {"N0": 0, "N1": 1}}

    def test_per_term_and_union(self):
        per_term, union = sv.answer_masses(self.row(), ["N1", "N0"])
        assert per_term == pytest.approx([.7, .8]) and union == pytest.approx(1.0)  # x counted once
        assert sv.answer_masses(self.row(), []) == ([], 0.0)

    def test_by_mass(self):
        assert sv.by_mass(self.row(), ["N1", "N0"]) == ["N0", "N1"]


class TestFlags:
    def rows(self):
        base = {"study": "s", "fold": 0, "gold_seen": True, "top1_p": .5}
        return [{**base, "sample_id": "a", "gold": OCEAN, "top1": OCEAN, "runner_up": SOIL, "runner_up_p": .2},
                {**base, "sample_id": "b", "gold": SOIL, "top1": OCEAN, "runner_up": SOIL, "runner_up_p": .4},
                {**base, "sample_id": "c", "gold": SOIL, "top1": SOIL, "runner_up": "", "runner_up_p": 0.0}]

    def test_table_and_threshold_is_inclusive(self):
        ft = sv.flag_table(self.rows(), [[], [OCEAN], [SOIL]], ANC, 0.2)
        assert ft["ambiguous_branches"].tolist() == [True, True, False]
        assert ft["backoff"].tolist() == ["", OCEAN, SOIL]

    def test_summary(self):
        ft = sv.flag_table(self.rows(), [[], [OCEAN], [SOIL]], ANC, 0.2)
        s = sv.flag_summary(ft, ANC)
        assert s["flagged"] == pytest.approx(2 / 3) and s["flagged_among_abstentions"] == 1.0
        assert s["top1_right_if_flagged"] == .5 and s["runner_up_right_if_flagged"] == .5
        assert s["either_right_if_flagged"] == 1.0


# ----------------------------------------------------------------------------- ontology terms as answers
class TestOntologyAnswers:
    """8 labels under a non-label ENVO_big (80 % of the labels: never an answer) and 2 labels under a non-label
    ENVO_small (20 % < Closure's 25 % floor: an answer in 'ontology' mode only). 6 split samples per fold put .5 / .45
    on the two small labels (half of their top-1s wrong, so answering them breaks the 90 % target): with labels only
    the back-off abstains on them, with ontology terms it answers ENVO_small."""
    BIG = [f"ENVO_b{i}" for i in range(8)]
    SMALL = ["ENVO_s1", "ENVO_s2"]
    VOC = np.array(sorted(BIG + SMALL))
    ANC2 = {**{b: {"ENVO_big"} for b in BIG}, **{t: {"ENVO_small"} for t in SMALL}, "ENVO_big": set(), "ENVO_small": set()}

    def folds(self):
        out = []
        for k in range(2):
            rows, gold = [], []
            for i in range(16):
                g = self.VOC[i % 10]
                rows.append({g: .95, self.VOC[(i + 1) % 10]: .05}); gold.append(g)
            for i in range(6):
                rows.append({"ENVO_s1": .5, "ENVO_s2": .45, "ENVO_b0": .05}); gold.append(self.SMALL[i % 2])
            P = np.array([[r.get(v, 0.0) for v in self.VOC] for r in rows])
            out.append({"P": P, "vocab": self.VOC, "gold": np.array(gold),
                        "sample_id": np.array([f"f{k}_{i}" for i in range(22)]),
                        "study": np.array([f"st{k}_{i // 2}" for i in range(22)])})
        return out

    def test_ontology_back_off_answers_the_shared_parent(self):
        args = type("Args", (), {"n_candidates": 10, "max_terms": 2, "min_mass": 0.02, "target": 0.9,
                                 "discounts": [1.0], "answers": "ontology", "flag_top": 5})
        rows, answers, outcomes, thresholds, source = sv.decode_slot(self.folds(), self.ANC2, args)
        split = [i % 22 >= 16 for i in range(44)]
        assert all(a == [] for a, s_ in zip(answers["backoff"], split) if s_)
        assert all(a == ["ENVO_small"] for a, s_ in zip(answers["backoff@onto"], split) if s_)
        assert "backoff+set@onto" in answers and source["backoff@onto"] is not rows
        assert "ENVO_small" not in {t for a in answers["evr_d1"] for t in a}  # other methods keep the labels


# ----------------------------------------------------------------------------- small helpers
class TestHelpers:
    def test_covers(self):
        assert sv.covers(MARINE, OCEAN, ANC) and sv.covers(OCEAN, OCEAN, ANC)
        assert not sv.covers(OCEAN, MARINE, ANC)  # more specific than gold
        assert not sv.covers(TERRESTRIAL, OCEAN, ANC)

    def test_study_pool(self):
        P = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        out = sv.study_pool(P, ["a", "a", "b"], 0.25)  # study a's mean row is [0.5, 0.5]
        assert out.tolist() == [[0.875, 0.125], [0.125, 0.875], [0.5, 0.5]]
        assert sv.study_pool(P, ["a", "a", "b"], 0.0) is P

    def test_bootstrap_resamples_groups(self):
        diff = np.array([1.0, 1.0, 0.0])
        lo, hi = sv.bootstrap_ci(diff, ["a", "a", "b"], n=200, seed=0)
        assert lo == 0.0 and hi == 1.0  # whole studies: a mean is 0, 1 or 2/3 -> never 1/3 from mixing rows
        assert sv.bootstrap_ci(np.ones(4), ["a", "b", "c", "d"], n=50) == (1.0, 1.0)

    def test_fold_groups(self, tmp_path):
        samples = pd.DataFrame({"sample_id": ["s1", "s2"], "study_code": ["A", "B"]})
        assert sv.fold_groups(samples).tolist() == ["A", "B"]
        p = tmp_path / "groups.tsv"
        pd.DataFrame({"sample_id": ["s1"], "project_group": ["P"]}).to_csv(p, sep="\t", index=False)
        assert sv.fold_groups(samples, str(p)).tolist() == ["P", "B"]


# ----------------------------------------------------------------------------- RuleScorer
class TestRuleScorer:
    """Biome labels X (river) and Y (lake) under a non-label parent PARENT; studies A, B, E (fold 0) have the same
    vector, A labels it X, B and E label it Y (2 study pairs vouch for X ~ Y); study C (fold 1) has another vector.
    Feature labels R (rhizosphere) / R2 (rhizosphere environment): synonyms by head word. X part_of Y2 (ENVO
    relation, a facet)."""
    X, Y, PARENT, Y2, R, R2 = "ENVO_x", "ENVO_y", "ENVO_parent", "ENVO_y2", "ENVO_r", "ENVO_r2"

    def scorer(self):
        X, Y, Y2, R, R2 = self.X, self.Y, self.Y2, self.R, self.R2
        samples = pd.DataFrame({"sample_id": ["a", "b", "e", "c"], "study_code": ["A", "B", "E", "C"],
                                "biome": [X, Y, Y, Y2], "feature": [R, R2, "", ""], "material": ["", "", "", ""]})
        x = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        fold = np.array([0, 0, 0, 1])
        anc = {X: {self.PARENT}, Y: {self.PARENT}, Y2: set(), self.PARENT: set(), R: set(), R2: set()}
        dist = {X: {self.PARENT: 1}, Y: {self.PARENT: 1}}
        label = {X: "river", Y: "lake", Y2: "lake water", R: "rhizosphere", R2: "rhizosphere environment"}
        vec = {t: v for t, v in zip([X, Y, Y2, R, R2], np.eye(5))}
        relations = {X: {("BFO:0000050", Y2)}}
        return sv.RuleScorer(samples, x, fold, anc, dist, label, vec, relations)

    def test_inputs(self):
        s = self.scorer()
        assert s.synonyms == {(self.R, self.R2)}
        assert s.n_pairs == 3  # a-b, a-e, b-e
        # co-labels of a fold come only from pairs of studies outside it: fold 1 sees A-B and A-E, fold 0 nothing
        assert s.co_labels["biome"][1] == {(self.X, self.Y)} and s.co_labels["biome"][0] == set()

    def test_one_term(self):
        s = self.scorer()
        rules, credit = s.term("biome", self.X, self.PARENT, 0)
        assert rules["broader"] == 1 and rules["exact"] == 0 and rules["all"] == 1
        c = s.slot["biome"]
        ic = lambda t: -np.log(c["counts"][t] / c["total"])  # noqa: E731
        assert credit["IC ratio"] == pytest.approx(ic(self.PARENT) / ic(self.X))
        assert credit["depth 0.5^k"] == 0.5 and credit["full credit"] == 1
        assert s.term("biome", self.X, self.Y, 1)[0]["co-labels"] == 1
        assert s.term("biome", self.X, self.Y, 0)[0]["co-labels"] == 0  # its own fold never vouches
        assert s.term("biome", self.X, self.Y2, 0)[0]["facets"] == 1
        assert s.term("feature", self.R, self.R2, 0)[0]["synonyms"] == 1

    def test_set_conventions_and_abstention(self):
        s = self.scorer()
        out = s.per_sample("biome", [self.X, self.X, self.X], [[self.X, self.Y2], [], [self.Y2]], ["a", "a", "a"])
        assert out[("any", "exact")] == [1.0, 0.0, 0.0]
        assert out[("split", "exact")] == [0.5, 0.0, 0.0]
        assert out[("any", "facets")] == [1.0, 0.0, 1.0] and out[("split", "facets")] == [1.0, 0.0, 1.0]

    def test_macro_and_learnable(self):
        s = self.scorer()
        rows = {(r["convention"], r["measure"]): r for r in
                s.score("biome", [self.X, self.Y, self.Y], [[self.X], [], [self.Y]], ["a", "b", "e"])}
        r = rows[("any", "exact")]
        assert r["micro"] == pytest.approx(2 / 3)
        assert r["macro"] == pytest.approx((1 + 0.5) / 2)  # X: 1/1, Y: 1/2
        assert r["macro_learnable"] == pytest.approx(0.5)  # only Y is used by >= 2 studies
