"""Mutation tests: proof that test_6c_flag_controls_unit.py and _integration.py catch plausible bugs.

Same harness as test_common_mutations.py. Each entry of MUTATIONS is a small, realistic mistake in
6c_flag_controls.py together with the tests that must catch it. For every mutation, test_mutation_is_caught:
  1. checks that the original snippet still occurs exactly once in 6c_flag_controls.py
     (if not, the script was edited: update the snippet here);
  2. writes a copy of the script with the snippet replaced, next to copies of common.py, step_loader.py
     and the two test files, in a temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real module imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run, e.g. because the mutant does not
     even import, does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the atlas flags,
`caught_by` names the tests whose assertions it breaks. Example: "lab_context_ignored" drops the
lab-context condition of the weak rule, so the keyword 'negative control' alone flags a sample;
TestWeakKeywordRule::test_weak_keyword_alone_is_not_enough[negative control] and the toy atlas
(sample mock2, a fermented food whose study mentions a mock community) fail.

The last six mutations undo the fixes of 2026-10-19 (word boundary, sub-biome-only samples, forced
gzip) and 2026-10-20 (underscores, quoting the text as written, 'control group'): they show that the
tests added with each fix would catch a regression.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "6c_flag_controls.py"
SUPPORT = [HERE.parent / "common.py", HERE / "step_loader.py"]   # copied unchanged
TEST_FILES = ["test_6c_flag_controls_unit.py", "test_6c_flag_controls_integration.py"]

U, I = "test_6c_flag_controls_unit.py::", "test_6c_flag_controls_integration.py::"  # node-id prefixes

MUTATIONS = [
    # ------------------------------------------------------------------ the three rules of evidence()
    Mutation("sub_biome_rule_off",
             [("if SUB_BIOME.search(sb):", "if False:")],
             "samples whose sub-biome is 'laboratory control', 'mock community' ... are no longer flagged",
             [U + "TestSubBiomeRule::test_control_sub_biomes_are_flagged[laboratory control]",
              I + "test_toy_atlas",                                                    # lab1
              I + "test_sample_only_in_the_sub_biome_file_is_judged_on_its_sub_biome"]),
    Mutation("sub_biome_case_sensitive",
             [(r'''|\b(blank|kit|reagent|extraction)[ -](control|blank)\b", re.I)''',
               r'''|\b(blank|kit|reagent|extraction)[ -](control|blank)\b")''')],
             "'Laboratory Control' (GPT capitalises freely) is no longer a control sub-biome",
             [U + "TestSubBiomeRule::test_case_is_ignored_and_the_original_text_is_reported"]),
    Mutation("strong_rule_off",
             [('strong = next((k for k, text in kws if STRONG.search(text)), "")', 'strong = ""')],
             "extraction blanks, NTCs, ZymoBIOMICS standards ... are no longer flagged by their keywords",
             [U + "TestStrongKeywordRule::test_strong_keywords_are_flagged[extraction blank]",
              I + "test_toy_atlas"]),                                                  # blank1, ntc1
    Mutation("reports_last_matching_keyword",
             [('strong = next((k for k, text in kws if STRONG.search(text)), "")',
               'strong = next((k for k, text in reversed(kws) if STRONG.search(text)), "")')],
             "same flags, but the evidence names another keyword than the first one of the GPT set",
             [U + "TestRulePrecedence::test_first_matching_keyword_is_reported"]),
    Mutation("lab_context_ignored",
             [("if weak and (LAB_SUB_BIOME.search(sb) or any(LAB_KEYWORD.match(text) for _, text in kws)):",
               "if weak:")],
             "'negative control' / 'mock community' alone flag a sample: untreated groups of real studies "
             "and studies that also sequenced a mock lose their habitat label",
             [U + "TestWeakKeywordRule::test_weak_keyword_alone_is_not_enough[negative control]",
              I + "test_toy_atlas"]),                                                  # mock2
    Mutation("control_alone_is_weak",
             [(r'WEAK = re.compile(r"^(negative|positive)[ -]controls?$',
               r'WEAK = re.compile(r"^((negative|positive)[ -])?controls?$')],
             "'control' alone becomes a weak keyword: a 'control' plot in a lab study is flagged",
             [U + "TestWeakKeywordRule::test_control_alone_is_not_even_a_weak_keyword"]),
    # ------------------------------------------------------------------ reading the GPT files
    Mutation("read_texts_keeps_first",
             [("out[sid] = text", "out.setdefault(sid, text)")],
             "a repeated sample id keeps its first, not its last GPT answer",
             [U + "TestReadTexts::test_repeated_sample_id_keeps_the_last_line",
              I + "test_repeated_sample_gives_one_row_from_its_last_line"]),
    Mutation("keywords_not_stripped",
             [('return [k.strip() for k in text.strip("{} ").split(",")]', 'return text.strip("{} ").split(",")')],
             "every keyword after the first keeps its leading space, so the ^...$ patterns never match it",
             [U + "TestSplitKeywords::test_braces_removed_and_keywords_stripped",
              I + "test_keyword_set_is_split_on_commas_and_stripped"]),
    Mutation("sub_biome_not_stripped",
             [('found = evidence(sub.get(sid, "").strip(),', 'found = evidence(sub.get(sid, ""),')],
             "a sub-biome with surrounding spaces escapes the ^-anchored sub-biome patterns",
             [I + "test_sub_biome_is_stripped"]),
    # ------------------------------------------------------------------ regressions of the 2026-10-19 fixes
    Mutation("no_word_boundary",
             [(r'(negative[ -]controls?|negatives?|blanks?|controls?)\b(?!',
               r'(negative[ -]controls?|negatives?|blanks?|controls?)(?!')],
             "'air blanket', 'kit controller' are flagged as an air blank, a kit control",
             [U + "TestStrongKeywordRule::test_control_word_must_end_at_a_word_boundary[air blanket]"]),
    Mutation("sub_biome_only_samples_dropped",
             [("for sid in list(keywords) + [s for s in sub if s not in keywords]:", "for sid in keywords:")],
             "a sample with a sub-biome but no keywords line is missing from the output, never flagged",
             [I + "test_sample_only_in_the_sub_biome_file_is_judged_on_its_sub_biome",
              I + "test_samples_follow_the_keywords_file_then_the_sub_biome_only_ones"]),
    Mutation("gzip_forced",
             [('out.to_csv(path(args.output), sep="\\t", index=False)',
               'out.to_csv(path(args.output), sep="\\t", index=False, compression="gzip")')],
             "an output named .tsv is a gzip file that read_tsv() cannot read",
             [I + "test_output_named_tsv_is_plain_text"]),
    # ------------------------------------------------------------------ regressions of the 2026-10-20 fixes
    Mutation("underscore_not_a_space",
             [('kws = [(k, k.replace("_", " ")) for k in keywords]', 'kws = [(k, k) for k in keywords]')],
             "snake_case keywords ('reagent_control', 'PCR_negative_control') are never matched",
             [U + "TestUnderscores::test_strong_keyword[reagent_control]",
              U + "TestUnderscores::test_weak_keyword_and_lab_context",
              I + "test_real_gpt_lines"]),                                             # SRS3000938
    Mutation("evidence_quotes_the_matched_text",
             [('strong = next((k for k, text in kws if STRONG.search(text)), "")',
               'strong = next((text for k, text in kws if STRONG.search(text)), "")')],
             "same flags, but the evidence quotes 'reagent control', a text GPT never wrote",
             [U + "TestUnderscores::test_strong_keyword[reagent_control]"]),
    Mutation("control_group_is_a_control",
             [("\\b(?![ -]groups?\\b)", "\\b")],
             "'buffer control group' (horses given a buffer) is flagged as a buffer control",
             [U + "TestStrongKeywordRule::test_a_control_group_is_a_group_of_real_samples[buffer control group]",
              I + "test_real_gpt_lines"]),                                             # SRS4129065
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the 6c tests (all, or only `node_ids`) against `source` as 6c_flag_controls.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {SCRIPT.name: source, "common.py": SUPPORT[0], "tests/step_loader.py": SUPPORT[1],
                                  **{f"tests/{name}": HERE / name for name in TEST_FILES}})
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, SCRIPT.read_text()), MUTATIONS, min_tests=100)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(SCRIPT, mutation.edits), mutation.caught_by))
