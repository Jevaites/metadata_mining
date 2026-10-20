"""Mutation tests: proof that test_2_build_training_set_unit.py and _integration.py catch plausible bugs.

Same harness as test_common_mutations.py. Each entry of MUTATIONS is a small, realistic mistake in
2_build_training_set.py together with the tests that must catch it. For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in 2_build_training_set.py
     (if not, the script was edited: update the snippet here);
  2. writes a copy of the script with the snippet replaced, next to copies of common.py, step2_helpers.py
     and the two test files, in a temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real module imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run, e.g. because the mutant does not
     even import, does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the training set,
`caught_by` names the tests whose assertions it breaks. Example: "own_id_tried_last" tries the accessions
written inside a record before the record's own id, so a record that is itself a Metalog sample but also
names another one gets the other sample's labels; test_own_id_wins_over_accessions_inside_the_record fails.

The last three mutations undo the fixes of 2026-10-19 (empty output without a header, Metalog snapshots
mixed, oldest snapshot by default): they show that the tests added with each fix would catch a regression.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "2_build_training_set.py"
SUPPORT = [HERE.parent / "common.py"]                           # copied unchanged, next to the script
TEST_FILES = ["test_2_build_training_set_unit.py", "test_2_build_training_set_integration.py", "step2_helpers.py"]

U, I = "test_2_build_training_set_unit.py::", "test_2_build_training_set_integration.py::"  # node-id prefixes

MUTATIONS = [
    # ------------------------------------------------------------------ linking MicrobeAtlas -> Metalog
    Mutation("own_id_tried_last",
             [('accessions = [sample_id] + ACCESSION.findall(" ".join(lines))',
               'accessions = ACCESSION.findall(" ".join(lines)) + [sample_id]')],
             "a record that is itself a Metalog sample but names another accession gets the other one's labels",
             [I + "test_own_id_wins_over_accessions_inside_the_record"]),
    Mutation("unlinked_records_kept",
             [("        if spire is None:\n            continue",
               "        if spire is None:\n"
               "            spire = sample_id\n"
               '            by_accession[spire] = {"study_code": "", "domain": "", "biome": "", "feature": "", '
               '"material": ""}')],
             "records Metalog does not know get a row with empty labels: the training set fills with unlabelled rows",
             [I + "test_basic_workflow"]),                                             # SRS9000002 appears
    Mutation("accession_regex_accepts_runs_and_experiments",
             [(r"[SED]RS\d+)\b", r"[SED]R[SRX]\d+)\b")],
             "run (SRR) and experiment (SRX) ids are taken for sample accessions",
             [U + "test_accession_regex[SRR000001 SRX000001 SRP000001-expected8]"]),

    # ------------------------------------------------------------------ Metalog labels
    Mutation("obsolete_labels_kept",
             [('labels.loc[unusable, slot] = ""', "pass")],
             "obsolete / unknown terms (e.g. Metalog's human biome ENVO:00009003) become training labels",
             [U + "test_code_not_in_valid_terms_is_blanked",
              I + "test_basic_workflow"]),                                             # SRS9000001's biome
    Mutation("duplicate_accession_keeps_last",
             [('drop_duplicates("spire_sample_name")', 'drop_duplicates("spire_sample_name", keep="last")')],
             "an accession under two Metalog aliases takes the labels of the last alias instead of the first",
             [U + "test_accession_under_two_aliases_keeps_the_first_alias"]),
    Mutation("repeated_item_keeps_last",
             [('aggfunc="first"', 'aggfunc="last"')],
             "a metadata item given twice for one sample keeps its second value",
             [U + "test_repeated_item_of_one_sample_keeps_the_first_value"]),

    # ------------------------------------------------------------------ text
    Mutation("obsolete_codes_not_named_in_text",
             [('"text": record_to_text(lines, code_to_label, args.max_chars)',
               '"text": record_to_text(lines, {k: v for k, v in code_to_label.items() if k in valid_terms}, '
               'args.max_chars)')],
             "an obsolete code in the metadata stays a bare code instead of its (still descriptive) label",
             [I + "test_obsolete_term_is_no_label_but_still_named_in_the_text"]),
    Mutation("max_chars_ignored",
             [('"text": record_to_text(lines, code_to_label, args.max_chars)',
               '"text": record_to_text(lines, code_to_label, 2000)')],
             "--max_chars has no effect",
             [I + "test_text_is_truncated_to_max_chars"]),

    # ------------------------------------------------------------------ regressions of the 2026-10-19 fixes
    Mutation("empty_output_without_columns",
             [("out = pd.DataFrame(rows, columns=COLUMNS)", "out = pd.DataFrame(rows)")],
             "with no linked record the file has no header and the summary print raises KeyError",
             [I + "test_no_record_linked_writes_an_empty_table"]),
    Mutation("all_snapshots_mixed",
             [('chosen = [p for p in paths if p.endswith(f"_all_long_{date}.tsv.gz")]', "chosen = paths")],
             "every snapshot of the folder is read: a sample in two snapshots keeps the OLDER labels",
             [U + "test_snapshot_paths_default_is_the_latest_date",
              I + "test_latest_snapshot_is_used_by_default"]),
    Mutation("oldest_snapshot_by_default",
             [("date = date or dates[-1]", "date = date or dates[0]")],
             "without --date the oldest snapshot is used, while step 2b uses the latest",
             [U + "test_snapshot_paths_default_is_the_latest_date",
              I + "test_latest_snapshot_is_used_by_default"]),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the step-2 tests (all, or only `node_ids`) against `source` as 2_build_training_set.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {SCRIPT.name: source, "common.py": SUPPORT[0],
                                  **{f"tests/{name}": HERE / name for name in TEST_FILES}})
    return run_pytest(tmp_path, tests, node_ids or [name for name in TEST_FILES if name.startswith("test_")])


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, SCRIPT.read_text()), MUTATIONS, min_tests=40)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(SCRIPT, mutation.edits), mutation.caught_by))
