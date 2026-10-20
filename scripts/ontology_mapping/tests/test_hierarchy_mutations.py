"""Mutation tests: proof that test_hierarchy_unit.py and test_hierarchy_integration.py catch plausible bugs.

Each entry of MUTATIONS is a small, realistic mistake in hierarchy.py (a > for a >=, a forgotten
condition, a fold that leaks into its own calibration, ...) together with HOW the tests must catch it:
`caught_by` maps a test to a piece of its failure message, usually the wrong value next to the right one
(e.g. "assert ('ENVO_marine', 0.7) == ('ENVO_ocean', 0.55)": the mutant answered marine where ocean is right).

For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in hierarchy.py
     (if not, hierarchy.py was edited: update the snippet here);
  2. writes a copy of hierarchy.py with the snippet replaced, next to copies of the test files, in a
     temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real `hierarchy` imported by the other tests is never touched);
  4. asserts that each of them FAILS with an AssertionError whose message contains the expected text
     (a crash, or a test that is not run, does not count as catching the bug in the expected way).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness. EQUIVALENT lists changes that look like bugs but cannot change any
result: test_equivalent_change_passes_everything documents why no test catches them.

To read one case: `edits` (find -> replace) is the bug, `why` is what it would do to the pipeline,
`caught_by` is which assertion notices and what it sees.

~1-2 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation and not equivalent"
"""
from collections import namedtuple
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
HIERARCHY = HERE.parent / "hierarchy.py"
TEST_FILES = ["test_hierarchy_unit.py", "test_hierarchy_integration.py"]
HELPERS = ["toy_hierarchy.py"]

U, I = "test_hierarchy_unit.py::", "test_hierarchy_integration.py::"  # node-id prefixes

# edits: [(find, replace), ...]; caught_by: {test node id: text (or tuple of texts) its failure message must contain}

MUTATIONS = [
    # ------------------------------------------------------------------ softmax / temperature
    Mutation("softmax_without_max_shift",
             [("    Z = Z - Z.max(1, keepdims=True)\n    E = np.exp(Z)", "    E = np.exp(Z)")],
             "exp() of large scores / small temperatures overflows: probabilities become nan",
             {U + "TestSoftmax::test_huge_scores_do_not_overflow": "nan"}),
    Mutation("temperature_counts_unknown_gold",
             [("            m = g >= 0\n", "            m = g >= -1\n")],
             "rows whose gold label is not a model class (-1) are scored against the LAST class (here always "
             "wrong), which drives T to its upper bound 20",
             {U + "TestFitTemperature::test_rows_whose_gold_is_not_in_the_vocabulary_are_ignored": "Obtained: 20.08"}),
    Mutation("temperature_fitted_on_own_fold",
             [("T = fit_temperature([b for j, b in enumerate(blocks) if j != k])", "T = fit_temperature(blocks)")],
             "each fold's temperature also sees that fold's gold labels: optimistic out-of-fold estimates "
             "(fold 1's T becomes ~0.77, pulled by its own all-ocean labels, instead of 1)",
             {I + "TestEdgeCases::test_temperature_of_a_fold_is_fitted_on_the_other_folds": "Obtained: 0.76"}),
    # ------------------------------------------------------------------ Closure: q and allowed answers
    Mutation("closure_without_ancestors",
             [("self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] = 1", "self.A[i, col[v]] = 1")],
             "q(node) = P(node) only: probability no longer adds up going up, nothing ever climbs",
             {U + "TestClosure::test_q_sums_the_probability_at_or_below_each_node": "{'ENVO_aquatic': 0.0} != {'ENVO_aquatic': 0.85}",
              U + "TestDecode::test_worked_example_of_the_module_docstring": "assert (None, 0.0) == ('ENVO_aquatic', 0.85)"}),
    Mutation("half_rule_inclusive",
             [("(self.n_below < 0.5 * len(self.vocab))", "(self.n_below <= 0.5 * len(self.vocab))")],
             "a label covering exactly half of the slot's labels becomes an allowed (too vague) answer",
             {U + "TestClosureAllowedAnswers::test_labels_mode_a_label_above_half_of_the_labels_is_too_vague":
                 ("Extra items in the left set", "'ENVO_marine'")}),
    Mutation("root_can_be_an_answer",
             [("not_root = ~np.isin(self.nodes, list(NO_ANSWER))", "not_root = np.ones(len(nodes), dtype=bool)")],
             "'biome' / 'environmental material' become answers: true of nearly every sample, useless",
             {U + "TestClosureAllowedAnswers::test_slot_root_is_never_an_answer":
                 ("Extra items in the left set", "'ENVO_00000428'")}),
    Mutation("ontology_mode_drops_other_labels",  # the bug fixed on 2026-10-19, put back
             [("self.informative = ((onto & (self.n_below < floor * len(self.vocab))) | self.is_label) & not_root",
               "self.informative = onto & ((self.n_below < floor * len(self.vocab)) | self.is_label) & not_root")],
             "answers='ontology' never answers the slot's PO_ / FOODON_ labels",
             {U + "TestClosureAllowedAnswers::test_ontology_mode_keeps_labels_from_other_ontologies": "assert 'PO_leaf' in {"}),
    # ------------------------------------------------------------------ Closure.decode
    Mutation("tau_must_be_exceeded",
             [("(q >= tau - 1e-12)", "(q > tau)")],
             "a node whose q equals tau exactly is skipped (one more step up than intended)",
             {U + "TestDecode::test_higher_tau_climbs_higher[0.55-expected1]": "('ENVO_marine', 0.7) == ('ENVO_ocean', 0.55)"}),
    Mutation("climb_leaves_the_top1_path",
             [("ok = (q >= tau - 1e-12) & self.informative & on_path", "ok = (q >= tau - 1e-12) & self.informative")],
             "the answer can be a branch the model did not rank first (here marine, .25 + .25, for a soil top-1)",
             {U + "TestDecode::test_climb_stays_on_the_path_of_the_top1_label": "('ENVO_marine', 0.5) == (None, 0.0)"}),
    Mutation("most_general_first",
             [("self.rank_key = -self.n_below * 1e6", "self.rank_key = self.n_below * 1e6")],
             "among the nodes reaching tau the most GENERAL one is picked: always the vaguest answer",
             {U + "TestDecode::test_higher_tau_climbs_higher[0.01-expected0]": "('ENVO_aquatic', 0.85) == ('ENVO_ocean', 0.55)",
              I + "TestEvaluationWorkflow::test_answers_per_sample_at_the_chosen_tau":
                 "{'open ocean': {'ENVO_aquatic'}} != {'open ocean': {'ENVO_ocean'}}"}),
    # ------------------------------------------------------------------ outcomes and summaries
    Mutation("coarser_and_too_specific_swapped",
             [("COARSER if n in ga else TOO_SPECIFIC if g in anc.get(n, ())",
               "TOO_SPECIFIC if n in ga else COARSER if g in anc.get(n, ())")],
             "a true-but-broader answer is graded too specific (and vice versa): strict accuracy is wrong",
             {U + "TestRelation::test_each_node_compared_with_the_gold_label": "assert [2, 1, 3, 4, 4, 3] == [3, 1, 2, 4, 4, 2]",
              I + "TestEvaluationWorkflow::test_curve_matches_the_hand_computed_table[0.8-0.6667-0.96-1.0]":
                 "assert (0.6667, 0.5, 1.0) == (0.6667, 0.96, 1.0)"}),
    Mutation("abstain_graded_as_node_0",
             [("    return np.where(pick >= 0, out, 0)", "    return out")],
             "an abstention (-1) is graded as if it had answered node 0 (here ocean: OTHER for a soil sample)",
             {U + "TestOutcome::test_code_of_the_picked_node_and_0_for_abstain": "assert [3, 4] == [3, 0]"}),
    Mutation("strict_accuracy_exact_only",
             [('"accuracy_strict": acc((EXACT, COARSER))', '"accuracy_strict": acc((EXACT,))')],
             "true-but-broader answers count as errors: back-off can never raise strict accuracy",
             {U + "TestSummarise::test_example_of_the_docstring": "{'accuracy_strict': 0.3333} != {'accuracy_strict': 0.6667}"}),
    # ------------------------------------------------------------------ choosing tau
    Mutation("choose_tau_highest_instead_of_lowest",
             [("    for row in curve:\n        if row[metric]", "    for row in reversed(curve):\n        if row[metric]")],
             "the most cautious tau reaching the target is chosen: far fewer answers than needed",
             {U + "TestChooseTau::test_lowest_tau_reaching_the_target[0.6-accuracy_strict-0.5]": "assert 0.8 == 0.5"}),
    Mutation("tau_chosen_on_all_folds",
             [("**summarise(others[i])} for i, t in enumerate(TAUS)], target, metric)",
               "**summarise(pooled[i])} for i, t in enumerate(TAUS)], target, metric)")],
             "each fold's tau also sees that fold's outcomes: the out-of-fold estimate is no longer honest",
             {I + "TestEdgeCases::test_tau_of_a_fold_is_chosen_on_the_other_folds": "[0.7, 0.7] == [0.7, 0.925]"}),
    Mutation("lenient_targets_use_strict_accuracy",
             [("**summarise(others[i])} for i, t in enumerate(TAUS)], target, metric)",
               "**summarise(others[i])} for i, t in enumerate(TAUS)], target, \"accuracy_strict\")")],
             "the lenient_* estimates (and taus) silently repeat the strict ones",
             {I + "TestEvaluationWorkflow::test_tau_per_target_is_the_lowest_reaching_it[lenient_0.75-0.05]":
                 "[0.625, 0.625, 0.625] == [0.05, 0.05, 0.05]"}),
    # ------------------------------------------------------------------ decode_candidates (reranker)
    Mutation("candidates_shallowest_ancestor_first",
             [("key=lambda a: (-len(anc.get(a, ())), -summed(a), a)", "key=lambda a: (len(anc.get(a, ())), -summed(a), a)")],
             "the climb jumps to the broadest allowed term instead of the next one up",
             {U + "TestDecodeCandidates::test_climbs_like_closure_decode[0.6-expected1]": "('ENVO_aquatic', 1.0) == ('ENVO_marine', 0.7)"}),
    Mutation("candidate_ties_by_set_order",  # the bug fixed on 2026-10-19, put back
             [("key=lambda a: (-len(anc.get(a, ())), -summed(a), a)", "key=lambda a: -len(anc.get(a, ()))")],
             "between two parents of equal depth, the first in the ancestor SET wins: hash-seed dependent",
             {U + "TestDecodeCandidates::test_equal_depth_parents_the_more_probable_first":
                 "('ENVO_freshwater', 0.65) == ('ENVO_marine', 0.75)",
              U + "TestDecodeCandidates::test_same_answer_whatever_the_hash_seed": ("Extra items in the left set", "'ENVO_m'")}),
    Mutation("candidates_root_can_be_an_answer",
             [("broader = [a for a in anc.get(top, ()) if a in allowed and a not in NO_ANSWER]",
               "broader = [a for a in anc.get(top, ()) if a in allowed]"),
              ("        if node in NO_ANSWER:\n            continue\n", "")],
             "the reranker can answer 'biome'",
             {U + "TestDecodeCandidates::test_slot_root_is_never_an_answer": "('ENVO_00000428', 1.0) == (None, 0.0)"}),
    # ------------------------------------------------------------------ merge_calibrations
    Mutation("merged_accuracy_over_all_samples",
             [('round(sum(mean[k] for k in ok) / mean["coverage"], 4)', 'round(sum(mean[k] for k in ok), 4)')],
             "merged accuracy = share of ALL samples answered correctly, not of the answered ones",
             {U + "TestMergeCalibrations::test_shares_are_averaged_and_accuracy_recomputed_from_them": "0.65 == 0.8667"}),
    Mutation("merged_temperature_arithmetic_mean",
             [('np.exp(np.mean([np.log(e["temperature"]) for e in entries]))',
               'np.mean([e["temperature"] for e in entries])')],
             "the arithmetic mean over-weights the larger temperature",
             {U + "TestMergeCalibrations::test_temperature_is_the_geometric_mean": "assert 0.25 == 0.2"}),
]

Equivalent = namedtuple("Equivalent", "name edits why")

EQUIVALENT = [
    Equivalent("closure_plus_equals",
               [("self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] = 1",
                 "self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] += 1")],
               "looks like it could count a shared ancestor twice (DAG), but the indices come from a set, so "
               "each node appears once and += 1 on a zero matrix gives exactly the same A"),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the hierarchy.py tests (all, or only `node_ids`) against `source` as hierarchy.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {"hierarchy.py": source,
                                  **{f"tests/{name}": HERE / name for name in TEST_FILES + HELPERS}})
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES, verbose=True)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, HIERARCHY.read_text()), MUTATIONS, min_tests=80)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(HIERARCHY, mutation.edits), mutation.caught_by))


@pytest.mark.parametrize("change", EQUIVALENT, ids=[e.name for e in EQUIVALENT])
def test_equivalent_change_passes_everything(change, tmp_path):
    check_unmutated(run_on_copy(tmp_path, mutate(HIERARCHY, change.edits)), [], min_tests=80)
