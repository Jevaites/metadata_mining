"""
Mutation tests for 6_predict_atlas.py: proof that the unit and integration tests catch plausible bugs,
and a record of what each bug would do to the atlas labels.

Each entry of MUTATIONS is one small change to the script (find -> replace), a sentence on its effect
on the pipeline (`why`), and the tests that must catch it (`caught_by`). The last four re-introduce
the two bugs fixed on 2026-10-19, so they also act as regression checks.

  test_mutation_is_caught[<name>]   runs the `caught_by` tests in a separate Python process against a
                                    mutated copy of the script (step6_toy loads the file named by
                                    $PREDICT_ATLAS_FILE) and asserts that every one of them FAILS
                                    (a test that does not run, e.g. because it was renamed, does not count).
  test_unmutated_copy_passes        the same harness on an unchanged copy: every test passes, so a
                                    failure above is due to the mutation and not to the harness.

~2 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from collections import namedtuple
from pathlib import Path

import pytest

from mutation_harness import check_caught, check_unmutated, mutate, named_tests, run_pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "6_predict_atlas.py"
U, I = "test_6_predict_atlas_unit.py::", "test_6_predict_atlas_integration.py::"  # node-id prefixes

Mutation = namedtuple("Mutation", "name find replace why caught_by")

MUTATIONS = [
    # ------------------------------------------------------------------ plausible slips
    Mutation("no_sub_biome_norm",
             "norm = np.sqrt(BLOCK_WEIGHT ** 2 * (1 + has_sb)", "norm = np.sqrt(BLOCK_WEIGHT ** 2 * 2",
             "prototype: ||x|| is taken as 1 also for samples without a sub-biome (really 1/sqrt 2): wrong cosines",
             [U + "TestScoreDecomposition::test_prototype_scores_are_cosine_to_centred_x_plus_bias"]),
    Mutation("no_study_weighting",
             "weight = np.where(real, 1 / np.maximum(per_study, 1), 0.0)", "weight = real.astype(float)",
             "knn_study: every neighbour votes 1, so one 50-sample study outvotes all other curators",
             [U + "TestKnnStudyTop2::test_a_study_has_one_vote_in_total"]),
    Mutation("tie_by_label_order",
             "best = np.where(rounded == rounded.max(axis=1, keepdims=True), first, k).argmin(axis=1)",
             "best = rounded.argmax(axis=1)",
             "knn_study: a tie goes to the alphabetically first label, not to the closest neighbour's",
             [U + "TestKnnStudyTop2::test_tie_goes_to_the_label_of_the_closest_neighbour"]),
    Mutation("no_broader_candidates",
             "row += sorted(extra, key=lambda c: -P[i, c])[:max_candidates - k]", "row += []",
             "the reranker's candidate lists lose the broader terms ('top-5 + ancestors' of the pilot)",
             [U + "TestCandidateStrings::test_top_k_then_their_broader_terms"]),
    Mutation("missing_text_row_0",
             "row_of.get(t, -1)", "row_of.get(t, 0)",
             "a text missing from the .h5 gets the vector of its first text: never-embedded samples get labels",
             [I + "TestWorkflowStages::test_stage2_index",
              I + "TestWorkflow::test_every_method_finds_the_obvious_answers[linear]"]),
    Mutation("parts_always_recomputed",
             "if os.path.exists(part):  # done in an earlier (interrupted) run", "if False:",
             "no resume: every rerun starts the 3.4M samples again (and --max_seconds never gets to the end)",
             [I + "TestChunksAndResume::test_a_finished_part_is_never_recomputed"]),
    Mutation("p_is_min",
             '"p": np.round(prob.max(axis=1), 4)', '"p": np.round(prob.min(axis=1), 4)',
             "<slot>_p is the probability of the least likely label instead of the top-1",
             [U + "TestBackoffColumns::test_top1_coarser_and_abstain",
              I + "TestBackoff::test_a_confident_model_answers_its_top1[prototype]"]),
    Mutation("same_sub_biome_for_all",
             "scores += parts[slot][sb_local]", "scores += parts[slot][0]",
             "linear: every sample gets the score part of the first sub-biome text, not of its own",
             [U + "TestScoreDecomposition::test_linear_scores_are_W_x_plus_b",
              I + "TestWorkflow::test_linear_confidence_is_a_margin_and_drops_without_sub_biome"]),
    Mutation("unlabelled_neighbours",
             'masked = np.where(parts["unlabelled"][slot][None, :], -np.inf, sim)', "masked = sim",
             "knn_study: training samples without a label in the slot become neighbours and vote for ''",
             [U + "TestScoreDecomposition::test_knn_study_chunk_with_fewer_labelled_samples_than_k",
              I + "TestSmallVocabulariesAndLargeK::test_knn_study_neighbours_are_only_labelled_samples"]),
    Mutation("no_sub_biome_offset_1",
             'np.zeros((1, model[f"{slot}_coef"].shape[0]))', 'np.ones((1, model[f"{slot}_coef"].shape[0]))',
             "linear: +1 for every label of samples without a sub-biome. Harmless for the output (argmax, margin "
             "and softmax do not change), but the scores are no longer W x + b",
             [U + "TestScoreDecomposition::test_linear_scores_are_W_x_plus_b"]),
    Mutation("resume_mixed_settings",
             "if previous != run:", "if False:",
             "a rerun with other back-off settings silently keeps chunks of the old run: mixed or missing columns",
             [I + "TestRefusedRuns::test_adding_calibration_to_an_output_dir_made_without",
              I + "TestChunksAndResume::test_resume_with_back_off_checks_the_settings"]),
    Mutation("no_tie_break",
             "sim = sim.astype(np.float64) - TIE_BREAK * np.arange(sim.shape[1])", "sim = sim.astype(np.float64)",
             "knn_study: among identical training vectors (common in Metalog) np.argpartition picks arbitrary rows, "
             "so the atlas vote can differ from the evaluated methods.knn_study",
             [U + "TestScoreDecomposition::test_knn_study_chunk_ties_among_identical_training_vectors"]),
    Mutation("accuracy_option_ignored",
             'f"accuracy_{args.accuracy}")', '"accuracy_strict")',
             "--accuracy lenient silently uses the strict curve: a higher tau, more back-off than asked for",
             [U + "TestLoadBackoff::test_accuracy_option_chooses_the_curve_column"]),
    # ------------------------------------------------------------------ the two bugs fixed on 2026-10-19
    Mutation("regression_binary_single_score",
             "if len(clf.classes_) == 2 and coef.shape[0] == 1:", "if False:",
             "linear: a 2-label slot keeps scikit-learn's single score row, so every sample gets the first "
             "label (before the fix, when top1_and_margin did not exist either, the run crashed)",
             [U + "TestLinearHelpers::test_two_classes_get_two_opposite_score_rows",
              I + "TestSmallVocabulariesAndLargeK::test_a_slot_with_two_labels[linear]"]),
    Mutation("regression_single_label_top2",
             "if scores.shape[1] == 1:", "if False:",
             "linear / prototype: a 1-label slot crashes in the top-2 step",
             [U + "TestLinearHelpers::test_a_single_class_wins_with_margin_0",
              I + "TestSmallVocabulariesAndLargeK::test_a_slot_with_one_label[linear]"]),
    Mutation("regression_unlabelled_fill_k",
             "real = np.isfinite(np.take_along_axis(sim, idx, axis=1))", "real = np.ones(idx.shape, bool)",
             "knn_study: with fewer labelled samples than k, unlabelled samples fill the list and '' can win",
             [U + "TestKnnStudyTop2::test_fewer_labelled_samples_than_k_only_the_labelled_ones_vote",
              U + "TestScoreDecomposition::test_knn_study_chunk_with_fewer_labelled_samples_than_k"]),
    Mutation("regression_k_not_capped",
             "    k = min(k, sim.shape[1])\n", "",
             "knn_study: k >= the number of training samples crashes in np.argpartition",
             [U + "TestKnnStudyTop2::test_k_larger_than_the_training_set_means_every_sample_votes",
              I + "TestSmallVocabulariesAndLargeK::test_knn_study_with_k_larger_than_the_training_set"]),
]


def run_tests_on(tmp_path, source, node_ids):
    """Run the given tests of this folder with `source` as the script (step6_toy loads $PREDICT_ATLAS_FILE)."""
    script = tmp_path / SCRIPT.name
    script.write_text(source)
    return run_pytest(tmp_path, HERE, node_ids, env={"PREDICT_ATLAS_FILE": str(script)}, args=["--rootdir", str(HERE)])


def test_unmutated_copy_passes(tmp_path):
    every_test = named_tests(MUTATIONS)
    check_unmutated(run_tests_on(tmp_path, SCRIPT.read_text(), every_test), MUTATIONS, min_tests=len(every_test))


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    source = mutate(SCRIPT, [(mutation.find, mutation.replace)])
    check_caught(mutation, run_tests_on(tmp_path, source, mutation.caught_by))
