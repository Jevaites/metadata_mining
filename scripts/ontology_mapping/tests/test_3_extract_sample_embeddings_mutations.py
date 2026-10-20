"""Mutation tests: proof that test_3_extract_sample_embeddings.py catches plausible bugs.

Same harness as test_6c_flag_controls_mutations.py. Each entry of MUTATIONS is a small, realistic
mistake in 3_extract_sample_embeddings.py together with the tests that must catch it.
For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in 3_extract_sample_embeddings.py
     (if not, the script was edited: update the snippet here, otherwise the mutation tests nothing);
  2. writes a copy of the script with the snippet replaced into a temporary folder, laid out like the
     repository (step 3 finds embed_subbiomes_keywords.py through ../embeddings, relative to itself):
         <tmp>/ontology_mapping/3_extract_sample_embeddings.py   <- the mutant
         <tmp>/ontology_mapping/common.py                        <- unchanged copies
         <tmp>/ontology_mapping/tests/test_3_extract_sample_embeddings.py
         <tmp>/embeddings/embed_subbiomes_keywords.py
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real module imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to
the mutation and not to the harness.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the .npz,
`caught_by` names the tests whose assertions it breaks. Example: "index_is_the_raw_h5_row" stores the
h5 row number (0, 2) instead of the position in `vectors` (0, 1). On the basic toy data the two are
the same (rows 0 and 1 are used), so only test_output_order_... notices, where rows 0 and 2 are used.

The last two mutations undo the fix of 2026-10-19 (clear message when nothing matches): they show
that the tests added with the fix would catch a regression.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent                      # scripts/ontology_mapping/tests
SCRIPT = HERE.parent / "3_extract_sample_embeddings.py"
COMMON = HERE.parent / "common.py"                          # used by the load_npz hand-off test
EMBED = HERE.parent.parent / "embeddings" / "embed_subbiomes_keywords.py"
TEST_FILE = "test_3_extract_sample_embeddings.py"

T = TEST_FILE + "::"  # node-id prefix

MUTATIONS = [
    # ------------------------------------------------------------------ read_first_column()
    Mutation("header_not_skipped",
             [('return ids - {"sample_id", ""}', "return ids")],
             "the 'sample_id' header (and '' from blank lines) are counted as requested samples",
             [T + "test_read_first_column_keeps_only_first_tsv_column_and_skips_header",
              T + "test_read_first_column_ignores_blank_lines_duplicates_spaces_and_crlf"]),
    Mutation("gzip_not_detected",
             [('if path.endswith(".gz")', 'if path.endswith(".gzip")')],
             "metalog_training_set.tsv.gz is read as plain text: garbage ids or a UnicodeDecodeError",
             [T + "test_read_first_column_reads_gzip_when_name_ends_in_gz"]),
    # ------------------------------------------------------------------ sample -> text
    Mutation("keywords_cleaning_inverted",
             [('args.kind == "keywords"', 'args.kind == "sub_biomes"')],
             "'{soil, forest floor}' keeps its braces and commas, so no keyword text matches the h5 file",
             [T + "test_basic_workflow_keywords",
              T + "test_no_text_found_in_embedding_file_stops_with_a_clear_message"]),
    # ------------------------------------------------------------------ text -> h5 row
    Mutation("h5_bytes_not_decoded",
             [("row_of = {t.decode() if isinstance(t, bytes) else t: i", "row_of = {t: i")],
             "h5py returns the texts as bytes: b'soil' != 'soil', nothing matches",
             [T + "test_basic_workflow",
              T + "test_non_ascii_texts_match"]),
    Mutation("h5_texts_decoded_as_latin1",
             [("row_of = {t.decode() if", 'row_of = {t.decode("latin-1") if')],
             "'vicuña feces' comes back as 'vicuÃ±a feces': every non-ASCII text is silently dropped",
             [T + "test_non_ascii_texts_match"]),
    Mutation("missing_text_raises",
             [("for sid, t in text_of.items() if t in row_of}", "for sid, t in text_of.items()}")],
             "one sample whose text was never embedded crashes the whole run with a KeyError",
             [T + "test_printed_summary_accounts_for_every_dropped_sample",
              T + "test_lookup_is_exact_after_cleaning"]),
    Mutation("rows_not_deduplicated",
             [("rows = np.array(sorted(set(found.values())))", "rows = np.array(sorted(found.values()))")],
             "S1 and S3 share 'soil': row 0 is asked twice and h5py refuses repeated indices",
             [T + "test_basic_workflow"]),
    # ------------------------------------------------------------------ the .npz layout
    Mutation("index_is_the_raw_h5_row",
             [("index=np.array([position[found[s]] for s in sample_ids])",
               "index=np.array([found[s] for s in sample_ids])")],
             "index holds h5 row numbers, not positions in `vectors`: wrong vector or out of range",
             [T + "test_output_order_sample_ids_sorted_vectors_in_h5_row_order"]),
    Mutation("sample_ids_not_sorted",
             [("sample_ids = np.array(sorted(found))", "sample_ids = np.array(list(found))")],
             "sample order follows the texts file, not the documented sorted order",
             [T + "test_output_order_sample_ids_sorted_vectors_in_h5_row_order"]),
    Mutation("output_tilde_not_expanded",
             [("np.savez(os.path.expanduser(args.output),", "np.savez(args.output,")],
             "--output ~/x.npz writes to a folder literally named '~' in the current directory",
             [T + "test_tilde_in_paths_is_expanded"]),
    # ------------------------------------------------------------------ regressions of the 2026-10-19 fix
    Mutation("no_text_guard_removed",
             [("    if not text_of:\n", "    if False:\n")],
             "wrong --sample_ids: the run stops with the h5 message instead of pointing at --sample_ids",
             [T + "test_no_requested_sample_has_a_text_stops_with_a_clear_message"]),
    Mutation("no_match_guard_removed",
             [("        if not found:", "        if False:")],
             "wrong --kind: back to the cryptic h5py 'Indexing arrays must have integer dtypes'",
             [T + "test_no_text_found_in_embedding_file_stops_with_a_clear_message"]),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the step-3 tests (all, or only `node_ids`) against `source` as 3_extract_sample_embeddings.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {"ontology_mapping/" + SCRIPT.name: source, "ontology_mapping/common.py": COMMON,
                                  "embeddings/" + EMBED.name: EMBED, "ontology_mapping/tests/" + TEST_FILE: HERE / TEST_FILE},
                       "ontology_mapping/tests")
    return run_pytest(tmp_path, tests, node_ids or [TEST_FILE])


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, SCRIPT.read_text()), MUTATIONS, min_tests=14)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(SCRIPT, mutation.edits), mutation.caught_by))
