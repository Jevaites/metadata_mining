"""Mutation tests: proof that test_set_valued_unit.py and test_set_valued_integration.py catch plausible bugs in
experiments/set_valued.py (subprocess style, as test_hierarchy_mutations.py; harness in mutation_harness.py).

Each Mutation is a one-line mistake (an ancestor allowed next to its descendant, a label counted twice under two
parents, a threshold chosen with the held-out fold, ...) with the tests that must fail on it and a piece of their
failure message. The script, its imports (_setup.py, acceptable_answers.py, common.py, hierarchy.py, methods.py) and
the test files are copied into a temporary folder; the mutant replaces the copy of set_valued.py, or of hierarchy.py for
the mutations in IN_HIERARCHY (candidate sets, set table, reviewer pick, runner-up and cross-fit live there since they
are shared with 5_evaluate.py / 6_predict_atlas.py; test_hierarchy_mutations.py covers the vectorised SetDecoder).

~2 s per mutation. Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SCRIPT = SCRIPTS / "experiments" / "set_valued.py"
HIERARCHY = SCRIPTS / "hierarchy.py"
TEST_FILES = ["test_set_valued_unit.py", "test_set_valued_integration.py"]
HELPERS = ["toy_hierarchy.py", "step5_toy.py"]
MODULES = ["common.py", "hierarchy.py", "methods.py"]
EXPERIMENT_MODULES = ["_setup.py", "acceptable_answers.py"]

U, I = "test_set_valued_unit.py::", "test_set_valued_integration.py::"

MUTATIONS = [
    # ------------------------------------------------------------------ candidate sets
    Mutation("candidates_allow_any_node",
             [("if informative[k] and q[k] >= min_mass]", "if q[k] >= min_mass]")],
             "the slot root and non-label nodes such as 'terrestrial' become answers (a set {biome} is always right "
             "and says nothing)",
             {U + "TestCandidateSets::test_only_allowed_nodes_with_enough_mass": "ENVO_terrestrial"}),
    Mutation("candidates_min_mass_exclusive",
             [("if informative[k] and q[k] >= min_mass]", "if informative[k] and q[k] > min_mass]")],
             "a node exactly at --min_mass is dropped (off by one at the boundary)",
             {U + "TestCandidateSets::test_min_mass_is_inclusive": "ENVO_desert"}),
    Mutation("sets_may_hold_an_ancestor",
             [("            if all(not comparable(a, b) for a, b in itertools.combinations(s, 2))]", "            ]")],
             "{ocean, marine} is offered as a 2-term answer: the same answer as 'marine', counted as a set",
             {U + "TestCandidateSets::test_sets_never_hold_a_term_and_its_ancestor": "assert"}),
    # ------------------------------------------------------------------ set table
    Mutation("union_counts_a_label_twice",
             [("cover = np.array([A[:, list(s)].any(1) for s in sets])",
               "cover = np.array([A[:, list(s)].sum(1) for s in sets])")],
             "a label under both nodes of a set (several parents) adds its probability twice: mass can exceed 1",
             {U + "TestSetTable::test_a_label_under_both_nodes_counts_once": "1.5"}),
    Mutation("expected_credit_is_plain_mass",
             [("(cover * p * np.minimum(1, ic[:, None] / ic_label[None, :])).sum(1)", "(cover * p).sum(1)")],
             "expected credit ignores how vague the set is: the expected-value rule then prefers the broadest set",
             {U + "TestSetTable::test_information_content_and_expected_credit": "0.8"}),
    # ------------------------------------------------------------------ decision rules
    Mutation("conformal_largest_first",
             [("np.lexsort((-table[\"mass\"][idx], table[\"n_labels\"][idx]))",
               "np.lexsort((-table[\"mass\"][idx], -table[\"n_labels\"][idx]))")],
             "the conformal rule returns the vaguest set that reaches tau instead of the smallest",
             {U + "TestConformalPick::test_smallest_set_reaching_tau": "assert 2 == 3"}),
    Mutation("conformal_ignores_max_terms",
             [("ok = (table[\"mass\"] >= tau - 1e-12) & (table[\"n_terms\"] <= max_terms)",
               "ok = table[\"mass\"] >= tau - 1e-12")],
             "conformal_m1 answers with pairs: the single-term baseline is no longer single-term",
             {U + "TestConformalPick::test_max_terms_and_abstention": "assert 1 == -1"}),
    Mutation("ev_answers_at_zero_value",
             [("    return k if value[k] > 0 else -1\n", "    return k if value[k] >= 0 else -1\n")],
             "a set worth nothing (no expected credit, no risk) is answered instead of abstaining",
             {U + "TestExpectedValuePick::test_abstains_when_no_set_has_positive_value": "assert 0 == -1"}),
    Mutation("ev_penalises_the_right_mass",
             [("table[\"expected_credit\"] - lam * (1 - table[\"mass\"])", "table[\"expected_credit\"] - lam * table[\"mass\"]")],
             "the penalty falls on P(right) instead of P(wrong): a larger lambda makes the rule bolder, not safer",
             {U + "TestExpectedValuePick::test_penalty_on_wrong_answers_favours_the_safer_set": "assert -1 == 1"}),
    # ------------------------------------------------------------------ thresholds
    Mutation("cross_fit_on_own_fold",
             [("        o = fold != k\n", "        o = fold == k\n")],
             "each fold's threshold is tuned on its own test samples: optimistic accuracy at the target",
             {U + "TestCrossFit::test_threshold_comes_from_the_other_folds": "{0: 0, 1: 1}"}),
    Mutation("cross_fit_infeasible_takes_the_first",
             [("chosen[int(k)] = int(ok[n_ans[ok].argmax()]) if len(ok) else answered.shape[1] - 1",
               "chosen[int(k)] = int(ok[n_ans[ok].argmax()]) if len(ok) else 0")],
             "when no threshold reaches the target, the most permissive one is used instead of the strictest",
             {U + "TestCrossFit::test_no_feasible_threshold_takes_the_last": "{0: 0, 1: 0}"}),
    Mutation("fallback_replaces_answered_samples",
             [("empty = np.asarray(primary_answered) == 0", "empty = np.ones(len(fold), bool)")],
             "back-off + set overwrites confident back-off answers with sets (and tunes the set threshold on them)",
             {U + "TestFallback::test_only_abstentions_change_and_threshold_from_other_folds_abstentions": "-1"}),
    Mutation("fallback_tuned_on_all_rows",
             [("o = (fold != k) & empty", "o = fold != k")],
             "the set threshold is tuned on every sample of the other folds, not on the abstentions it will serve",
             {U + "TestFallback::test_only_abstentions_change_and_threshold_from_other_folds_abstentions": "None"}),
    # ------------------------------------------------------------------ probabilities
    Mutation("temperature_fitted_on_all_folds",
             [("f[\"T\"] = fit_temperature([blk for j, blk in enumerate(blocks) if j != k])",
               "f[\"T\"] = fit_temperature(blocks)")],
             "each fold's temperature also sees its own gold labels: optimistic calibration",
             {U + "TestOofProbabilities::test_temperature_of_a_fold_is_fitted_on_the_other_folds_only": "[3, 3, 3]"}),
    Mutation("study_pool_weights_swapped",
             [("return (1 - weight) * P + weight * means", "return weight * P + (1 - weight) * means")],
             "--study_pool 0.3 keeps 30 % of the sample's own row instead of 70 %",
             {U + "TestHelpers::test_study_pool": "0.625"}),
    # ------------------------------------------------------------------ scoring
    Mutation("split_convention_takes_the_best_term",
             [("                out[(\"split\", m)].append(float(np.mean(v)) if v else 0.0)\n            for m in SCHEMES:",
               "                out[(\"split\", m)].append(max(v) if v else 0.0)\n            for m in SCHEMES:")],
             "'split' scores a set like 'any': the strict convention silently becomes the lenient one",
             {U + "TestRuleScorer::test_set_conventions_and_abstention": "0.5"}),
    Mutation("co_labels_from_the_test_fold",
             [("train = differ & (fold[I] != f) & (fold[J] != f)", "train = differ")],
             "a test study vouches for its own label pair: co-label credit leaks across folds",
             {U + "TestRuleScorer::test_inputs": "ENVO_x"}),
    Mutation("learnable_includes_single_study_labels",
             [("\"learnable\": set(n_studies[n_studies >= 2].index)", "\"learnable\": set(n_studies[n_studies >= 1].index)")],
             "macro over learnable labels includes labels used by one study (unlearnable under study folds)",
             {U + "TestRuleScorer::test_macro_and_learnable": "0.75"}),
    Mutation("covers_accepts_narrower_answers",
             [("return term == gold or term in anc.get(gold, ())",
               "return term == gold or term in anc.get(gold, ()) or gold in anc.get(term, ())")],
             "an answer more specific than gold counts as right: accuracy at the target is overstated",
             {U + "TestHelpers::test_covers": "assert not True"}),
    # ------------------------------------------------------------------ reviewer score
    Mutation("reviewer_credit_takes_the_worst_term",
             [("best = np.array([np.max([term_credit[n] for n in s], axis=0) for s in sets])",
               "best = np.array([np.min([term_credit[n] for n in s], axis=0) for s in sets])")],
             "a set is worth its least informative term per label: labels covered by one term only score 0",
             {U + "TestReviewerCredit::test_a_set_is_worth_its_best_term_per_label": "assert"}),
    Mutation("reviewer_discount_on_every_term",
             [("discount ** (table[\"n_terms\"] - 1.0)", "discount ** table[\"n_terms\"]")],
             "the discount also hits single terms: with a small d the rule abstains instead of giving the single term",
             {U + "TestReviewerValuePick::test_small_discount_keeps_the_single_term": "assert -1 == 0"}),
    # ------------------------------------------------------------------ flags and masses
    Mutation("runner_up_may_be_an_ancestor",
             [("        if v in anc.get(first, ()) or first in anc.get(v, ()):\n            continue\n", "")],
             "the 'unrelated' runner-up can be the top-1's own parent (ocean / marine biome): every confident sample "
             "looks ambiguous",
             {U + "TestUnrelatedRunnerUp::test_skips_ancestors_and_labels_sharing_an_allowed_ancestor": "ENVO_marine"}),
    Mutation("union_mass_counts_shared_labels_twice",
             [("return per_term, float(r[\"p\"] @ r[\"A\"][:, cols].any(1))",
               "return per_term, float(r[\"p\"] @ r[\"A\"][:, cols].sum(1))")],
             "set_mass adds a label under two of the terms twice (several parents): masses above 1",
             {U + "TestAnswerMasses::test_per_term_and_union": "1.5"}),
    Mutation("primary_answer_is_the_least_probable",
             [("key=lambda x: (-x[0], x[1])", "key=lambda x: (x[0], x[1])")],
             "the alternatives come first and the primary answer last",
             {U + "TestAnswerMasses::test_by_mass": "assert"}),
    Mutation("flag_threshold_exclusive",
             [("\"ambiguous_branches\": [r[\"runner_up_p\"] >= flag_mass for r in rows]",
               "\"ambiguous_branches\": [r[\"runner_up_p\"] > flag_mass for r in rows]")],
             "a runner-up exactly at --flag_mass is not flagged (off by one at the boundary)",
             {U + "TestFlags::test_table_and_threshold_is_inclusive": "assert"}),
    # ------------------------------------------------------------------ ontology answers
    Mutation("ontology_mode_never_used",
             [("onto = sample_rows(folds, anc, args, answers=\"ontology\", max_terms=min(2, args.max_terms))",
               "onto = sample_rows(folds, anc, args, answers=\"labels\", max_terms=min(2, args.max_terms))")],
             "--answers ontology silently reruns the labels-only back-off: backoff@onto equals backoff",
             {U + "TestOntologyAnswers::test_ontology_back_off_answers_the_shared_parent": "assert"}),
]


IN_HIERARCHY = {"candidates_allow_any_node", "candidates_min_mass_exclusive", "sets_may_hold_an_ancestor",
                "union_counts_a_label_twice", "expected_credit_is_plain_mass", "cross_fit_on_own_fold",
                "cross_fit_infeasible_takes_the_first", "reviewer_credit_takes_the_worst_term",
                "reviewer_discount_on_every_term", "runner_up_may_be_an_ancestor"}


def run_on_copy(tmp_path, source, node_ids=(), hierarchy=None):
    files = {"experiments/set_valued.py": source,
             **{f"experiments/{m}": SCRIPTS / "experiments" / m for m in EXPERIMENT_MODULES},
             **{m: SCRIPTS / m for m in MODULES},
             **({"hierarchy.py": hierarchy} if hierarchy is not None else {}),
             **{f"tests/{name}": HERE / name for name in TEST_FILES + HELPERS}}
    tests = build_copy(tmp_path, files)
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES, verbose=True)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, SCRIPT.read_text()), MUTATIONS, min_tests=40)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    if mutation.name in IN_HIERARCHY:
        outcomes = run_on_copy(tmp_path, SCRIPT.read_text(), mutation.caught_by, hierarchy=mutate(HIERARCHY, mutation.edits))
    else:
        outcomes = run_on_copy(tmp_path, mutate(SCRIPT, mutation.edits), mutation.caught_by)
    check_caught(mutation, outcomes)


def test_hierarchy_mutations_target_hierarchy():
    """Every IN_HIERARCHY snippet is in hierarchy.py and no longer in set_valued.py (so it really was moved)."""
    assert IN_HIERARCHY <= {m.name for m in MUTATIONS}
    for m in MUTATIONS:
        if m.name in IN_HIERARCHY:
            for find, _ in m.edits:
                assert HIERARCHY.read_text().count(find) == 1 and find not in SCRIPT.read_text(), m.name
