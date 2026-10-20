"""
Mutation tests for 5_evaluate.py: does the test suite notice when the code is wrong?

Each entry of MUTATIONS is one small, plausible bug, written as a replacement in the source of 5_evaluate.py:
`original` (a snippet of the real code) -> `mutant` (the broken version), plus the test of
test_5_evaluate_unit.py / test_5_evaluate_integration.py that is supposed to catch it.

test_mutation_is_caught runs, for every entry:
  1. `original` must occur exactly once in 5_evaluate.py. If the code is refactored and the snippet disappears,
     this fails and the entry must be updated (otherwise the mutation would silently test nothing).
  2. A broken copy of the module is built: the source with the one replacement, executed as a separate module
     (the real 5_evaluate module, and the file on disk, are never modified).
  3. The catching test is run against the real module: it passes.
  4. The same test is run against the broken copy (the test module's `ev` is pointed at it): it must fail,
     with the expected exception (AssertionError = a wrong result; other types = the bug makes it crash).

Read with `pytest tests/test_5_evaluate_mutations.py -v`: the test ids are the mutation names.
"""
import tempfile
import types
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

import step5_toy as toy
import test_5_evaluate_integration as integration
import test_5_evaluate_unit as unit
from step_loader import load_step
from test_5_evaluate_unit import fake_openai  # noqa: F401 (fixture, requested by name below)

ev = load_step("5_evaluate.py", "evaluate")
SOURCE_PATH = Path(ev.__file__)
SOURCE = SOURCE_PATH.read_text()


class Mutation(NamedTuple):
    name: str
    why: str           # what the mutation breaks, in words
    original: str      # exact snippet of 5_evaluate.py
    mutant: str        # its broken replacement
    check: Callable    # check(ctx) runs the catching test; ctx gives fresh folders, capsys and the fake OpenAI
    raises: type = AssertionError  # or a tuple of exception types


def basic_run(ctx):
    """What the `basic_run` fixture of the integration tests returns, computed with the module under test."""
    paths = toy.write_project(ctx.tmp())
    return integration.run_main(paths, *integration.embedding_options(paths))

MUTATIONS = [
    # ------------------------------------------------------------------------------------- scoring
    Mutation(
        "gold_or_ancestor_counts_only_direct_parents",
        "top1_gold_or_ancestor must accept any is_a ancestor of gold (a grandparent too), not only the parent",
        "p in ancestors.get(t, ())", "p in parents.get(t, ())",
        lambda c: unit.test_score_on_four_hand_made_predictions()),
    Mutation(
        "confident_half_takes_the_least_confident",
        "top1_confident_half must keep the most confident half (descending order), not the least confident",
        'hit[np.argsort(-confidence, kind="stable")[:len(g) // 2]]', 'hit[np.argsort(confidence, kind="stable")[:len(g) // 2]]',
        lambda c: unit.test_score_on_four_hand_made_predictions()),
    Mutation(
        "confident_half_of_one_sample_is_nan",
        "the fix for a single-sample group removed: the empty 'confident half' gives NaN instead of None",
        ".mean() if len(g) > 1 else None,", ".mean(),",
        lambda c: unit.test_score_single_sample_has_no_confident_half()),
    Mutation(
        "precision_cutoff_stops_at_the_first_dip",
        "precision_threshold must keep the largest confident prefix reaching the target, not the first one",
        "return round(float(confidence[order][ok[-1]]), 4), round((ok[-1] + 1) / len(order), 4)",
        "return round(float(confidence[order][ok[0]]), 4), round((ok[0] + 1) / len(order), 4)",
        lambda c: unit.test_precision_threshold_takes_the_lowest_cutoff_not_the_first_dip()),
    # ------------------------------------------------------------------------------------- features
    Mutation(
        "blocks_weighted_one_over_n",
        "with n feature blocks each block must be weighted 1/sqrt(n), so that sample . term = mean of block cosines",
        "weight = 1 / np.sqrt(len(args.features))", "weight = 1 / len(args.features)",
        lambda c: unit.test_two_blocks_dot_product_is_the_mean_of_block_cosines()),
    Mutation(
        "embedding_requests_repeat_duplicate_texts",
        "every distinct text must be sent to the embedding API once, however often it occurs",
        "todo = [t for t in dict.fromkeys(texts) if key[t] not in cache]", "todo = [t for t in texts if key[t] not in cache]",
        lambda c: unit.test_embedding_encoder_embeds_each_distinct_text_once(c.openai, c.tmp())),
    Mutation(
        "embedding_cache_never_saved",
        "new embeddings must be written to the cache file, so that later folds and runs reuse them",
        "        if todo:\n            np.savez(cache_path", "        if False:\n            np.savez(cache_path",
        lambda c: unit.test_embedding_cache_is_reused_by_a_later_run(c.openai, c.tmp())),
    # ------------------------------------------------------------------------------------- one fold
    Mutation(
        "unseen_flag_always_true",
        "gold_seen_in_train must say whether the gold label occurs among the training labels of the fold",
        "seen = np.isin(test[slot].to_numpy()[te], y)", "seen = np.ones(te.sum(), dtype=bool)",
        lambda c: unit.test_evaluate_fold_unseen_gold_label()),
    Mutation(
        "unlabelled_test_samples_are_scored",
        "only test samples labelled in a slot may be scored in that slot",
        'tr, te = train[slot].ne("").to_numpy(), test[slot].ne("").to_numpy()',
        'tr, te = train[slot].ne("").to_numpy(), np.ones(len(test), dtype=bool)',
        lambda c: unit.test_evaluate_fold_only_samples_labelled_in_a_slot_are_scored()),
    Mutation(
        "majority_ranks_the_rarest_first",
        "the majority baseline must rank training labels from the most to the least frequent",
        "Counter(y).most_common(5)", "Counter(y).most_common()[::-1][:5]",
        lambda c: unit.test_evaluate_fold_majority_ranks_training_label_counts()),
    Mutation(
        "closed_retrieval_ranks_every_term",
        "retrieval (closed) must rank only the slot's training labels, not every term of the table",
        'predictions["retrieval"] = M.join_ranks([M.rank(c[:, in_block], ids[in_block], margin=False)',
        'predictions["retrieval"] = M.join_ranks([M.rank(c, ids, margin=False)',
        lambda c: unit.test_evaluate_fold_closed_methods_only_rank_training_labels()),
    Mutation(
        "closed_only_ignored",
        "--closed_only must skip the open-vocabulary methods",
        'if not args.closed_only:\n                predictions["retrieval_open"]',
        'if True:\n                predictions["retrieval_open"]',
        lambda c: unit.test_evaluate_fold_closed_only_skips_open_vocabulary_methods()),
    Mutation(
        "dense_only_methods_run_on_sparse_features",
        "prototype / retrieval_prior need dense vectors: they must be skipped for sparse (TF-IDF) features",
        "if term_matrix is not None and not sparse.issparse(x_tr):\n            a, b",
        "if term_matrix is not None:\n            a, b",
        lambda c: unit.test_evaluate_fold_sparse_features_skip_the_dense_only_methods(),
        # recent scikit-learn refuses the sparse matrix (TypeError); older versions run and add the methods
        raises=(TypeError, AssertionError)),
    Mutation(
        "empty_slot_not_skipped__with_term_vectors",
        "the fix for empty slots removed: with term vectors, retrieval over an empty vocabulary crashes",
        "        if not tr.any() or not te.any():", "        if False:",
        lambda c: unit.test_evaluate_fold_slot_without_any_label_is_skipped(unit.TERM_VECTORS),
        raises=IndexError),
    Mutation(
        "empty_slot_not_skipped__without_term_vectors",
        "the fix for empty slots removed: without term vectors an empty back-off input is created for the slot",
        "        if not tr.any() or not te.any():", "        if False:",
        lambda c: unit.test_evaluate_fold_slot_without_any_label_is_skipped(None)),
    # ------------------------------------------------------------------------------------- back-off
    Mutation(
        "backoff_prob_inverted",
        "the prob column must be the calibrated probability of the top-1 label",
        '"prob": round(float(p), 4)', '"prob": round(float(1 - p), 4)',
        lambda c: unit.test_backoff_all_climbs_to_the_common_parent_when_unsure()),
    Mutation(
        "backoff_column_uses_the_wrong_target",
        "the backoff column must use --backoff_target, not the first of the reported --backoff_targets",
        'name = f"{args.backoff_metric}_{args.backoff_target}"',
        'name = f"{args.backoff_metric}_{min(args.backoff_targets)}"',
        lambda c: unit.test_backoff_all_climbs_to_the_common_parent_when_unsure()),
    Mutation(
        "backoff_with_a_single_fold",
        "the fix removed: a slot scored in one fold has no 'other folds' to fit temperature and tau on",
        "        if len(folds) < 2:", "        if False:",
        lambda c: unit.test_backoff_all_needs_at_least_two_folds(c.capsys),
        raises=ValueError),
    # ------------------------------------------------------------------------------------- main()
    Mutation(
        "gold_read_from_the_wrong_slot",
        "the gold term of each prediction row must come from that row's slot",
        'pred["gold"] = [samples.at[r, s] for r, s in zip(pred["row"], pred["slot"])]',
        'pred["gold"] = [samples.at[r, "biome"] for r, s in zip(pred["row"], pred["slot"])]',
        lambda c: integration.test_basic_gold_columns_come_from_the_training_set(basic_run(c))),
    Mutation(
        "samples_without_a_vector_kept",
        "samples without a vector in the .npz must be dropped when the samples are selected",
        "samples = select_samples(args.samples, required, args.max_per_study, args.seed)",
        "samples = select_samples(args.samples, [], args.max_per_study, args.seed)",
        lambda c: integration.test_samples_without_a_vector_are_dropped(c.tmp()),
        raises=KeyError),  # the missing sample is looked up in the .npz and is not there
    Mutation(
        "folds_split_by_sample_not_study",
        "cross-validation must keep every study on one side of a fold (samples of a study share most of their text)",
        'groups = samples["study_code"]\n    if args.fold_groups:', 'groups = samples["sample_id"]\n    if args.fold_groups:',
        lambda c: integration.test_unseen_gold_label(c.tmp())),
    Mutation(
        "tfidf_fitted_on_all_samples",
        "the TF-IDF vectorizer of a fold must be fitted on its training texts only, not on the held-out samples",
        'build_features(args, npz, list(train["text"]), term_texts', 'build_features(args, npz, list(samples["text"]), term_texts',
        lambda c: integration.test_tfidf_is_fitted_on_the_training_fold_only(c.tmp())),
    Mutation(
        "fold_groups_ignored",
        "--fold_groups must keep whole projects on one side of a fold",
        'groups = samples["sample_id"].map(project).fillna(samples["study_code"])',
        'groups = samples["study_code"]',
        lambda c: integration.test_fold_groups_keep_a_project_on_one_side(c.tmp())),
]


@pytest.fixture
def ctx(tmp_path, capsys, fake_openai):  # noqa: F811 (fake_openai is the imported fixture)
    """What a check may need: a fresh folder per call (the real and the mutant run must not share files such as
    the embedding cache), capsys, and the args of the fake OpenAI client."""
    return types.SimpleNamespace(tmp=lambda: Path(tempfile.mkdtemp(dir=tmp_path)), capsys=capsys, openai=fake_openai)


def build_mutant(mutation):
    """5_evaluate.py with the one replacement, executed as a separate module named evaluate_mutant."""
    module = types.ModuleType("evaluate_mutant")
    module.__file__ = str(SOURCE_PATH)
    code = compile(SOURCE.replace(mutation.original, mutation.mutant), f"5_evaluate.py [{mutation.name}]", "exec")
    exec(code, module.__dict__)
    return module


def run_check(mutation, module, ctx, monkeypatch):
    """Run the catching test with `ev` (in both test modules) pointing at `module`."""
    with monkeypatch.context() as patch:
        patch.setattr(unit, "ev", module)
        patch.setattr(integration, "ev", module)
        unit.FakeOpenAI.calls = []
        mutation.check(ctx)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # e.g. the mean of an empty slice in a mutant
@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, ctx, monkeypatch):
    assert SOURCE.count(mutation.original) == 1, f"update this mutation: its snippet is not (once) in {SOURCE_PATH.name}"
    run_check(mutation, ev, ctx, monkeypatch)  # the real code passes the test ...
    with pytest.raises(mutation.raises):  # ... the broken copy fails it
        run_check(mutation, build_mutant(mutation), ctx, monkeypatch)


def test_mutation_names_are_unique():
    assert len({m.name for m in MUTATIONS}) == len(MUTATIONS)
