"""Mutation tests: proof that test_methods_unit.py and test_methods_integration.py catch plausible bugs.

Each entry of MUTATIONS is a small, realistic mistake in methods.py (a forgotten centring, an
off-by-one in the neighbour list, a swapped sign, ...) together with the tests that must catch it.
For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in methods.py
     (if not, methods.py was edited: update the snippet here);
  2. writes a copy of methods.py with the snippet replaced, next to copies of the two test files,
     in a temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real `methods` imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run, e.g. because the mutant does not
     even import, does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness. test_run_fold_matches_5_evaluate (not a mutation) checks that the
workflow copy in test_methods_integration.py still matches the real 5_evaluate.evaluate_fold.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the pipeline,
`caught_by` names the tests whose assertions it breaks. Example: "study_votes_not_split" gives every
neighbour a full vote in knn_study, so one study with many near-identical samples outvotes the other
curators again, and test_edge_one_study_floods_the_neighbour_list fails.

The mutations marked "regression" put back a bug that was fixed on 2026-10-19 (the code as it was
before the fix), so the fix stays covered.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
METHODS = HERE.parent / "methods.py"
TEST_FILES = ["test_methods_unit.py", "test_methods_integration.py"]

U, I = "test_methods_unit.py::", "test_methods_integration.py::"  # node-id prefixes

MUTATIONS = [
    # ------------------------------------------------------------------ rank / blocks
    Mutation("rank_ascending",
             [("np.argsort(-scores[i:i + 1000]", "np.argsort(scores[i:i + 1000]")],
             "terms are ranked worst first",
             [U + "test_rank_docstring_example",
              I + "test_workflow_every_method_finds_the_obvious_label[retrieval]"]),
    Mutation("margin_is_best_score",
             [("top2[:, 0] - top2[:, -1] if margin", "top2[:, 0] if margin")],
             "a sample with two near-equal candidates looks as confident as a clear one",
             [U + "test_rank_docstring_example",
              U + "test_rank_ranks_each_row_independently"]),
    Mutation("rank_n1_margin_zero",  # regression
             [("[:, :max(n, 2)]", "[:, :n]")],
             "with n=1 only one column is kept, so every margin is best - best = 0",
             [U + "test_rank_regression_margin_with_n_equal_1"]),
    Mutation("rank_ids_must_be_array",  # regression
             [("    ids = np.asarray(ids)  # a plain list works too\n", "")],
             "a plain list of ids (as in the docstring example) raises TypeError",
             [U + "test_rank_regression_accepts_a_plain_list_of_ids"]),
    Mutation("row_blocks_drop_last",
             [("range(0, max(n, 1), size)", "range(0, max(n, 1) - size, size)")],
             "the last block of test samples is never scored",
             [U + "test_row_blocks",
              I + "test_edge_many_test_samples_cross_every_internal_block_boundary"]),
    # ------------------------------------------------------------------ neighbours: knn, knn_study
    Mutation("chunk_drops_a_row",
             [("test[start:start + rows] @ train.T", "test[start:start + rows - 1] @ train.T")],
             "the last test sample of every 2,000-row chunk gets no prediction (all later rows shift)",
             [U + "test_knn_is_the_same_across_its_internal_2000_row_chunks",
              I + "test_edge_many_test_samples_cross_every_internal_block_boundary"]),
    Mutation("one_neighbour_too_few",
             [("[:, :k]  # the k best", "[:, :k - 1]  # the k best")],
             "knn and knn_study look at k - 1 neighbours",
             [U + "test_knn_docstring_example",
              U + "test_knn_mixed_neighbourhood_is_a_similarity_weighted_vote"]),
    Mutation("tie_break_reversed",
             [("ordering = sim - offset", "ordering = sim + offset")],
             "identical texts are ordered last-index-first: results no longer match 6_predict_atlas.py",
             [U + "test_knn_study_identical_training_vectors_are_ordered_by_training_index",
              U + "test_knn_regression_identical_training_vectors_are_ordered_by_training_index",
              U + "test_neighbours_ties_among_many_identical_vectors_follow_training_index[float64]"]),
    Mutation("tie_break_missing",
             [("ordering = sim - offset", "ordering = sim")],
             "exact ties (identical texts) are left in argpartition's arbitrary order",
             [U + "test_neighbours_ties_among_many_identical_vectors_follow_training_index[float64]",
              U + "test_neighbours_ties_among_many_identical_vectors_follow_training_index[float32]"]),
    Mutation("tie_break_in_float32",
             [("sim = dense(test[start:start + rows] @ train.T).astype(np.float64)",
               "sim = dense(test[start:start + rows] @ train.T)"),
              ("ordering = sim - offset", "ordering = sim - offset.astype(sim.dtype)")],
             "with float32 embeddings (the real .npz/.h5 dtype) the 1e-13 offset is lost: ties unbroken",
             [U + "test_neighbours_ties_among_many_identical_vectors_follow_training_index[float32]"]),
    Mutation("knn_unweighted",
             [("votes[train_labels[i]] += max(s, 0.0)", "votes[train_labels[i]] += 1")],
             "knn counts neighbours instead of weighting them by similarity",
             [U + "test_knn_docstring_example",
              U + "test_knn_study_votes_are_not_weighted_by_similarity"]),
    Mutation("knn_negative_votes",  # regression
             [("votes[train_labels[i]] += max(s, 0.0)", "votes[train_labels[i]] += s")],
             "negative similarities subtract from the vote: the confidence can exceed 1",
             [U + "test_knn_regression_negative_similarities_vote_zero"]),
    Mutation("study_votes_not_split",
             [("votes[train_labels[i]] += 1 / per_study[study]", "votes[train_labels[i]] += 1")],
             "knn_study gives every neighbour a full vote: one large study outvotes the other curators",
             [U + "test_knn_study_docstring_example_tie_goes_to_the_closest_neighbour",
              U + "test_knn_study_a_study_splits_its_single_vote_between_its_labels",
              I + "test_edge_one_study_floods_the_neighbour_list"]),
    Mutation("studies_must_be_array",  # regression
             [("    train_studies = np.asarray(train_studies)  # a plain list works too\n", "")],
             "a plain list of study codes raises TypeError",
             [U + "test_knn_study_regression_accepts_a_plain_list_of_studies"]),
    Mutation("study_votes_not_rounded",
             [("key=lambda label: round(votes[label], 9)", "key=lambda label: votes[label]")],
             "a tie between two curators is decided by float noise (10 x 1/10 < 1) instead of closeness",
             [U + "test_knn_study_float_noise_cannot_break_a_tie"]),
    # ------------------------------------------------------------------ prototypes
    Mutation("terms_not_centred",
             [("prototypes = normalize(vocab_vectors - vocab_vectors.mean(axis=0))",
               "prototypes = normalize(vocab_vectors)")],
             "the offset between term names and sample texts (modality gap) is kept",
             [U + "test_prototype_model_centres_the_term_vectors"]),
    Mutation("samples_not_centred",
             [("np.add.at(sums, inverse, normalize(train - mean))", "np.add.at(sums, inverse, normalize(train))")],
             "class centroids are built from uncentred samples, while test samples are centred",
             [U + "test_prototype_model_centres_the_training_samples"]),
    Mutation("centroid_samples_not_normalised",
             [("np.add.at(sums, inverse, normalize(train - mean))", "np.add.at(sums, inverse, train - mean)")],
             "samples far from the training mean pull their class centroid more than the others",
             [U + "test_prototype_model_each_sample_counts_equally_in_the_centroid"]),
    Mutation("test_samples_not_centred",
             [("return normalize(test - mean) @ prototypes.T + bias", "return normalize(test) @ prototypes.T + bias")],
             "test samples are compared to centred prototypes without being centred themselves",
             [U + "test_prototype_scores_is_cosine_of_the_centred_sample_plus_bias"]),
    Mutation("alpha_ignored",
             [("    if alpha > 0:\n", "    if False:\n")],
             "prototype degenerates to retrieval_prior: training centroids are never used",
             [U + "test_prototype_model_alpha_1_uses_the_training_centroids",
              U + "test_prototype_model_alpha_half_is_the_bisector_of_centroid_and_term"]),
    Mutation("alpha_swapped",
             [("normalize(alpha * normalize(sums) + (1 - alpha) * prototypes[rows])",
               "normalize((1 - alpha) * normalize(sums) + alpha * prototypes[rows])")],
             "--prototype_alpha weights the term vector instead of the centroid",
             [U + "test_prototype_model_alpha_1_uses_the_training_centroids"]),
    Mutation("prior_smoothing_changed",
             [("(n + 0.5) / (n + 0.5).sum()", "(n + 1) / (n + 1).sum()")],
             "the label-frequency prior is no longer the documented one",
             [U + "test_prototype_model_bias_is_the_log_label_frequency"]),
    Mutation("prior_sign_flipped",
             [("beta * np.log(", "-beta * np.log(")],
             "frequent labels are penalised instead of favoured; unseen terms win",
             [U + "test_prototype_model_bias_is_the_log_label_frequency",
              I + "test_edge_gold_label_never_seen_in_training"]),
    Mutation("unseen_bonus_ignored",
             [("+ unseen_bonus * (n == 0)", "")],
             "--prototype_unseen_bonus has no effect",
             [U + "test_prototype_model_unseen_bonus_goes_to_terms_without_training_samples_only",
              I + "test_edge_gold_label_never_seen_in_training"]),
    Mutation("unseen_bonus_to_seen_terms",
             [("unseen_bonus * (n == 0)", "unseen_bonus * (n > 0)")],
             "the bonus goes to the terms that already have training samples",
             [U + "test_prototype_model_unseen_bonus_goes_to_terms_without_training_samples_only"]),
    Mutation("prototype_empty_test_set_crashes",  # regression
             [("    if test.shape[0] == 0:  # sklearn's normalize refuses 0 rows\n        return np.zeros((0, len(bias)))\n",
               "")],
             "a fold without labelled test samples crashes evaluate_fold",
             [U + "test_prototype_regression_empty_test_set",
              I + "test_edge_empty_test_set"]),
    # ------------------------------------------------------------------ linear methods
    Mutation("binary_columns_swapped",
             [("scores = np.column_stack([-scores, scores])", "scores = np.column_stack([scores, -scores])")],
             "in a two-label slot every prediction is the other label",
             [U + "test_linear_scores_binary_returns_one_column_per_class"]),
    Mutation("single_class_not_handled",
             [("if len(classes) < 2:", "if len(classes) < 1:")],
             "a slot with one training label crashes (sklearn needs two classes)",
             [U + "test_linear_scores_single_class_returns_a_constant_column",
              I + "test_edge_slot_with_a_single_training_label"]),
    Mutation("sparse_features_use_ridge",
             [("clf = LinearSVC(C=0.5, random_state=0) if sparse.issparse(train) else RidgeClassifier(alpha=1.0)",
               "clf = RidgeClassifier(alpha=1.0)")],
             "TF-IDF features no longer get the classifier chosen for them",
             [U + "test_linear_scores_sparse_features_use_linearsvc"]),
    Mutation("linear_empty_test_set_crashes",  # regression
             [("    if test.shape[0] == 0:  # nothing to score", "    if False:  # nothing to score")],
             "a fold without labelled test samples crashes evaluate_fold",
             [U + "test_linear_scores_regression_empty_test_set",
              I + "test_edge_empty_test_set"]),
    Mutation("hybrid_unseen_start_at_0",
             [("scores = np.full(cosine.shape, -1.0)", "scores = np.full(cosine.shape, 0.0)")],
             "unseen terms start level with a neutral class instead of at 'not this class' (-1)",
             [U + "test_hybrid_adds_weighted_cosine_to_the_linear_score"]),
    Mutation("hybrid_columns_by_position",
             [("scores[:, [column[c] for c in classes]] = linear", "scores[:, :len(classes)] = linear")],
             "linear scores land on the wrong terms when the vocabulary is ordered differently from classes",
             [U + "test_hybrid_adds_weighted_cosine_to_the_linear_score"]),
    Mutation("hybrid_weight_ignored",
             [("rank(scores + weight * cosine, vocab_ids)", "rank(scores + cosine, vocab_ids)")],
             "--hybrid_weight has no effect",
             [U + "test_hybrid_adds_weighted_cosine_to_the_linear_score",
              U + "test_hybrid_an_unseen_term_wins_only_through_its_cosine"]),
    Mutation("label_regression_not_normalised",
             [("predicted = normalize(Ridge(alpha=alpha).fit(train, train_gold_vectors).predict(test))",
               "predicted = Ridge(alpha=alpha).fit(train, train_gold_vectors).predict(test)")],
             "confidences shrink with the ridge penalty instead of being cosine margins",
             [U + "test_label_regression_can_land_between_training_labels"]),
    Mutation("label_regression_empty_test_set_crashes",  # regression
             [("    if test.shape[0] == 0:  # sklearn refuses 0 rows\n        return [], np.zeros(0)\n", "")],
             "a fold without labelled test samples crashes evaluate_fold",
             [U + "test_label_regression_regression_empty_test_set",
              I + "test_edge_empty_test_set"]),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the methods.py tests (all, or only `node_ids`) against `source` as methods.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {"methods.py": source, **{f"tests/{name}": HERE / name for name in TEST_FILES}})
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, METHODS.read_text()), MUTATIONS, min_tests=50)


@pytest.mark.parametrize("unseen_bonus", [0.0, 0.2])
def test_run_fold_matches_5_evaluate(unseen_bonus):
    """test_methods_integration.run_fold is a minimal copy of 5_evaluate.evaluate_fold (kept so the harness
    above can run it on a lone methods.py). Guard against drift: on its toy world, with clear, ambiguous and
    unseen-label test samples, every method must give the same top-5 lists and confidences as the real
    evaluate_fold (one labelled slot; evaluate_fold adds only 'majority')."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from step_loader import load_step
    import test_methods_integration as T

    ev = load_step("5_evaluate.py", "evaluate")
    test_x = T.at(3, 88, 183, 268, 40, 120)
    args = SimpleNamespace(k=3, knn_study_k=3, hybrid_weight=2.0, prototype_alpha=0.5, prototype_beta=0.1,
                           prototype_unseen_bonus=unseen_bonus, closed_only=False, backoff_methods=["prototype"])
    other_slots = {slot: "" for slot in ev.SLOTS if slot != "biome"}  # only one slot is labelled
    train = pd.DataFrame({"biome": T.TRAIN_Y, "study_code": T.TRAIN_STUDIES, **other_slots})
    test = pd.DataFrame({"biome": ["soil"] * len(test_x), "study_code": "S9", **other_slots})
    term_row = {t: i for i, t in enumerate(T.TERM_IDS)}
    records, backoff_inputs = ev.evaluate_fold(args, train, test, T.TRAIN_X, test_x, T.TERM_MATRIX, T.TERM_IDS,
                                               T.TERM_MATRIX, term_row)
    expected, backoff_scores = T.run_fold(T.TRAIN_X, T.TRAIN_Y, T.TRAIN_STUDIES, test_x, unseen_bonus=unseen_bonus)

    records = pd.DataFrame(records)
    assert set(records["method"]) == set(expected) | {"majority"}
    for method, (ranked, confidence) in expected.items():
        got = records[records["method"] == method].sort_values("row")
        assert [list(map(str, r)) for r in got["top5"]] == [list(map(str, r)) for r in ranked], method
        np.testing.assert_allclose(got["confidence"].to_numpy(float), confidence, err_msg=method)
    np.testing.assert_allclose(backoff_inputs[("biome", "prototype")]["S"], backoff_scores)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(METHODS, mutation.edits), mutation.caught_by))
