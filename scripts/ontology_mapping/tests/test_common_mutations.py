"""Mutation tests: proof that test_common_unit.py and test_common_integration.py catch plausible bugs.

Each entry of MUTATIONS is a small, realistic mistake in common.py (an off-by-one, a forgotten
normalisation, a split by sample instead of by study, ...) together with the tests that must catch it.
For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in common.py
     (if not, common.py was edited: update the snippet here);
  2. writes a copy of common.py with the snippet replaced, next to copies of the two test files,
     in a temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real `common` imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run, e.g. because the mutant does not
     even import, does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the pipeline,
`caught_by` names the tests whose assertions it breaks. Example: "folds_per_sample" assigns folds per
sample instead of per study, so samples of one study land on both sides of a split (the leakage
study_folds exists to prevent), and TestStudyFolds::test_no_study_on_both_sides fails.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common.py"
TEST_FILES = ["test_common_unit.py", "test_common_integration.py"]

U, I = "test_common_unit.py::", "test_common_integration.py::"  # node-id prefixes

MUTATIONS = [
    # ------------------------------------------------------------------ ontology terms
    Mutation("term_text_separator",
             [('str.replace("||", "; "', 'str.replace("||", ", "')],
             "term texts no longer match the keys of the term-embedding .h5",
             [U + "TestTermText::test_label_then_synonyms_joined_by_semicolons",
              I + "test_term_table_to_term_vectors"]),
    Mutation("term_text_whitespace_cleaned",
             [('str.replace("||", "; ", regex=False)', r'str.replace("||", "; ", regex=False).str.replace(r"\s+", " ", regex=True)')],
             "a tidy-looking clean-up: the 2 multi-line FOODON texts no longer match their .h5 keys",
             [U + "TestTermText::test_newlines_and_spaces_inside_synonyms_are_kept_verbatim",
              U + "TestLoadTerms::test_reads_the_real_table_format"]),
    Mutation("na_strings_parsed",
             [("keep_default_na=False", "keep_default_na=True")],
             "'NA' / 'nan' cells become NaN: e.g. the synonym 'NA' of nasal artery turns the whole term text into NaN",
             [U + "TestReadTsv::test_every_cell_is_a_string_and_empty_cells_stay_empty",
              U + "TestLoadTerms::test_reads_the_real_table_format"]),
    Mutation("obsolete_kept",
             [('terms[terms["obsolete"] != "True"]', "terms")],
             "obsolete terms become possible labels",
             [U + "TestLoadTerms::test_drops_obsolete_terms_and_adds_text",
              I + "test_term_table_to_term_vectors"]),
    Mutation("term_vectors_not_normalised",
             [("return normalize(vectors[[row_of[t] for t in texts]])", "return vectors[[row_of[t] for t in texts]]")],
             "dot products are no longer cosines: long vectors win the nearest-term search",
             [U + "TestLoadTermVectors::test_rows_follow_the_requested_order_and_are_unit_length",
              I + "test_selected_samples_and_nearest_term"]),
    Mutation("ancestors_direct_parents_only",
             [("{p} | up(p) for p in", "{p} for p in")],
             "ancestor sets stop at the parents: closure-based scoring and back-off see no grandparents",
             [U + "TestAncestorSets::test_chain",
              U + "TestTermAncestors::test_reads_the_parents_column"]),
    Mutation("distance_off_by_one",
             [("distance[parent] = d + 1", "distance[parent] = d + 2")],
             "every ancestor looks twice as far as it is",
             [U + "TestAncestorDistances::test_shortest_number_of_steps"]),
    Mutation("coarsen_farthest_ancestor",
             [("up[0][1] if up", "up[-1][1] if up")],
             "rare labels jump to the most general qualifying ancestor instead of the closest",
             [U + "TestCoarsenLabels::test_docstring_example",
              I + "test_coarsen_rare_labels_to_ancestors"]),
    Mutation("coarsen_strict_threshold",
             [("if support.get(term, 0) >= min_support:", "if support.get(term, 0) > min_support:")],
             "a label with exactly min_support samples is coarsened",
             [U + "TestCoarsenLabels::test_exactly_min_support_is_enough",
              I + "test_coarsen_rare_labels_to_ancestors"]),
    Mutation("coarsen_no_descendant_support",
             [("support[a] = support.get(a, 0) + n", "support[a] = support.get(a, 0)")],
             "support counts only a term's own samples, not those below it",
             [U + "TestCoarsenLabels::test_docstring_example"]),
    Mutation("coarsen_ignores_known",
             [("            if a in known:\n                support[a]", "            if True:\n                support[a]"),
              ("if a in known and support.get(a, 0) >= min_support", "if support.get(a, 0) >= min_support")],
             "labels can be coarsened to terms outside the term table (e.g. a BFO upper class)",
             [U + "TestCoarsenLabels::test_ancestor_outside_known_is_never_used"]),
    # ------------------------------------------------------------------ metadata records
    Mutation("no_dedupe",
             [('"; ".join(dict.fromkeys(kept))', '"; ".join(kept)')],
             "sample and study copies of the same field are both kept",
             [U + "TestRecordToText::test_duplicate_lines_kept_once_in_first_order"]),
    Mutation("missing_not_stripped",
             [('value.lower().strip(" .") in MISSING', "value.lower() in MISSING")],
             "'missing.' or ' NA ' are kept as if they were values",
             [U + "TestRecordToText::test_missing_values_are_dropped[missing.]"]),
    Mutation("code_key_format",
             [('f"{m[1]}_{m[2]}"', 'f"{m[1]}:{m[2]}"')],
             "ontology codes are never found in code_to_label (keys use '_'), so never translated",
             [U + "TestRecordToText::test_ontology_codes_replaced_by_labels",
              I + "test_step2_metadata_records_become_clean_texts"]),
    Mutation("prefix_not_stripped",
             [('re.sub(r"^(sample|study)_", "", key.strip())', "key.strip()")],
             "keys keep their 'sample_' / 'study_' prefix, and anchored DROP_KEYS patterns stop matching",
             [U + "TestRecordToText::test_key_prefix_removed_and_key_lowercased",
              I + "test_step2_metadata_records_become_clean_texts"]),
    Mutation("date_keys_kept",
             [("|library|date|time|", "|library|time|")],
             "collection dates stay in the text (they identify the study, not the habitat)",
             [U + "TestRecordToText::test_identifier_date_and_coordinate_keys_are_dropped[sample_collection_date]",
              I + "test_step2_metadata_records_become_clean_texts"]),
    Mutation("last_record_lost",
             [("    if sample_id:\n        yield sample_id, lines\n\n\ndef record", "\n\ndef record")],
             "the last sample of sample.info is silently skipped",
             [U + "TestIterSampleInfo::test_plain_file",
              U + "TestIterSampleInfo::test_gzipped_file"]),
    Mutation("plain_file_strict_utf8",  # the bug fixed on 2026-10-19, kept as a regression check
             [(r'encoding="utf-8", errors="replace", newline="\n")', r'encoding="utf-8", newline="\n")')],
             "one non-UTF-8 byte stops the whole pass over sample.info",
             [U + "TestIterSampleInfo::test_invalid_utf8_is_replaced_in_plain_file"]),
    Mutation("universal_newlines",  # the behaviour before 2026-10-19
             [(r', newline="\n") as handle:', ") as handle:")],
             "a lone '\\r' inside a value splits the line: the rest of the value (it has no '=') is lost",
             [U + "TestIterSampleInfo::test_carriage_return_inside_a_value_becomes_a_space"]),
    Mutation("carriage_return_kept",
             [(r'.replace("\r", " ")', "")],
             "a '\\r' stays inside the value, and in the text sent to the embedding / reranking model",
             [U + "TestIterSampleInfo::test_carriage_return_inside_a_value_becomes_a_space"]),
    # ------------------------------------------------------------------ samples, folds
    Mutation("npz_index_ignored",
             [('normalize(saved["vectors"])[saved["index"]]', 'normalize(saved["vectors"])')],
             "one row per distinct text instead of per sample: samples get other samples' vectors",
             [U + "TestLoadNpz::test_rows_shared_vectors_and_normalisation",
              I + "test_selected_samples_and_nearest_term"]),
    Mutation("stable_key_format",
             [('f"{seed}:{v}"', 'f"{v}:{seed}"')],
             "a different (still valid) shuffle: published selections and folds would change silently",
             [U + "TestStableKey::test_is_md5_of_seed_colon_value",
              U + "TestStudyFolds::test_fold_of_a_study_is_md5_mod_n_folds"]),
    Mutation("no_shuffle_before_cap",
             [('samples = samples.iloc[np.argsort(stable_key(seed, samples["sample_id"]), kind="stable")]', "pass")],
             "the per-study cap keeps the first rows of the file, which depend on how the file was written",
             [U + "TestSelectSamples::test_caps_each_study_keeping_the_smallest_keys",
              U + "TestSelectSamples::test_rows_ordered_by_stable_key_with_a_fresh_index"]),
    Mutation("label_required_in_all_slots",
             [('.ne("").any(axis=1)', '.ne("").all(axis=1)')],
             "samples labelled in only one or two slots (e.g. every human sample: no biome) are dropped",
             [U + "TestSelectSamples::test_keeps_samples_with_at_least_one_slot_label"]),
    Mutation("require_ids_ignored",
             [("for ids in require_ids:", "for ids in ():")],
             "samples without an embedding are kept and later crash or get no vector",
             [U + "TestSelectSamples::test_keeps_samples_present_in_every_required_set",
              I + "test_selected_samples_and_nearest_term"]),
    Mutation("cap_ignored",
             [("    if max_per_study:\n", "    if False:\n")],
             "huge cohorts (1,679 infant-gut samples) dominate training and scores again",
             [U + "TestSelectSamples::test_caps_each_study_keeping_the_smallest_keys"]),
    Mutation("default_cap_changed",
             [("max_per_study=50, seed=22", "max_per_study=100, seed=22")],
             "every script that relies on the default selects other samples: earlier scores are no longer comparable",
             [U + "TestSelectSamples::test_defaults_cap_50_per_study_with_seed_22"]),
    Mutation("folds_per_sample",
             [("for k in stable_key(seed, studies)])[inverse]", "for k in stable_key(seed, range(len(codes)))])")],
             "folds per sample, not per study: near-copies of test rows end up in training (leakage)",
             [U + "TestStudyFolds::test_no_study_on_both_sides",
              I + "test_cross_validation_by_study"]),
    Mutation("train_includes_test",
             [("yield np.where(fold != f)[0], np.where(fold == f)[0]", "yield np.arange(len(codes)), np.where(fold == f)[0]")],
             "the model is trained on the samples it is tested on",
             [U + "TestStudyFolds::test_train_and_test_are_complementary",
              I + "test_cross_validation_by_study"]),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the common.py tests (all, or only `node_ids`) against `source` as common.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {"common.py": source, **{f"tests/{name}": HERE / name for name in TEST_FILES}})
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, COMMON.read_text()), MUTATIONS, min_tests=50)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(COMMON, mutation.edits), mutation.caught_by))
