"""Mutation tests: proof that test_1_build_term_index_unit.py and _integration.py catch plausible bugs.

Same harness as test_common_mutations.py. Each entry of MUTATIONS is a small, realistic mistake in
1_build_term_index.py (a `find` -> `replace` edit of its source) together with the tests that must
catch it. For every mutation, test_mutation_is_caught:
  1. checks that the `find` snippet still occurs exactly once in 1_build_term_index.py
     (if not, the script was edited: update the snippet here);
  2. writes a copy of the script with the snippet replaced, next to copies of common.py, step_loader.py
     and the two test files, in a temporary folder;
  3. runs the tests listed in `caught_by` on that copy, in a separate Python process
     (so the real module imported by the other tests is never touched);
  4. asserts that every one of them FAILS (a test that is not run, e.g. because the mutant does not
     even import, does not count as catching it).
test_unmutated_copy_passes runs the same harness on an unchanged copy, so a failure above is due to the
mutation and not to the harness, and checks that every test named in `caught_by` exists.

To read one case: the `find` -> `replace` pair is the bug, `why` is what it would do to the term table,
`caught_by` names the tests whose assertions it breaks. Example: "part_of_as_parent" also reads
'relationship: BFO:0000050 ENVO:00000099' (part_of) lines as parents, so sediment would get BFO_0000050 as an
is_a parent; test_parents_are_the_is_a_lines_in_order_and_nothing_else and the toy ENVO table fail.
`caught_by` lists the most direct tests, not all of them (e.g. losing the last OBO stanza fails dozens).

The last three mutations undo the fixes of 2026-10-19 (local paths starting with 'http', \\" in OBO
synonyms, a run with no term at all): they show that the regression test added with each fix would
catch the bug coming back.

~1 s per mutation (a fresh Python each). Skip them with:  pytest -k "not mutation"
"""
from pathlib import Path

import pytest

from mutation_harness import Mutation, build_copy, check_caught, check_unmutated, mutate, run_pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "1_build_term_index.py"
SUPPORT = [HERE.parent / "common.py", HERE / "step_loader.py"]   # copied unchanged
TEST_FILES = ["test_1_build_term_index_unit.py", "test_1_build_term_index_integration.py"]

U, I = "test_1_build_term_index_unit.py::", "test_1_build_term_index_integration.py::"  # node-id prefixes
OBO, OWL = U + "TestParseObo::", U + "TestParseOwl::"
OWL_LABEL = OWL + "test_label_is_the_first_nonempty_english_label_else_the_class_is_dropped"
TOY_TABLE = I + "test_build_a_table_from_an_obo_and_an_owl_file"   # the toy ENVO + FOODON table, cell by cell

MUTATIONS = [
    # ------------------------------------------------------------------ reading files
    Mutation("gz_not_decompressed",
             [('return gzip.decompress(data) if source.endswith(".gz") else data', 'return data')],
             "envo.obo.gz / foodon.owl.gz are parsed as compressed bytes: no terms, or a decoding crash",
             [U + "TestReadBytes::test_local_gz_file_is_decompressed",
              U + "TestReadBytes::test_gz_url_is_downloaded_then_decompressed",
              TOY_TABLE]),
    Mutation("windows_line_ends_kept",
             [('.decode("utf-8").splitlines()', '.decode("utf-8").split("\\n")')],
             "an OBO file saved with Windows line ends keeps a '\\r' on every line ('[Term]\\r' is no stanza)",
             [U + "test_read_obo_decodes_utf8_and_splits_lines_also_windows_line_ends"]),
    Mutation("owl_gz_read_as_obo",
             [('owl = re.search(r"\\.owl(\\.gz)?$", source)', 'owl = source.endswith(".owl")')],
             "foodon.owl.gz (how FOODON is stored) goes to the OBO reader: 0 FOODON terms",
             [TOY_TABLE]),

    # ------------------------------------------------------------------ parse_owl (FOODON)
    Mutation("regional_english_rejected",
             [('lang == "en" or lang == "" or lang.startswith("en-")', 'lang == "en" or lang == ""')],
             "FOODON's en-gb / en-us texts are dropped ('yoghurt', 'cornflour')",
             [U + "test_is_english[en-gb-True]",
              OWL_LABEL + "[en-us]",
              OWL + "test_synonyms_of_every_kind_are_kept_if_english_and_not_empty"]),
    Mutation("language_tag_case_sensitive",
             [('lang = (lang or "").lower()', 'lang = (lang or "")')],
             "xml:lang='EN' or 'en-GB' is not recognised as English",
             [U + "test_is_english[EN-True]"]),
    Mutation("owl_prefix_without_underscore",
             [('if not term_id.startswith(prefix + "_"):', 'if not term_id.startswith(prefix):')],
             "PO=... would also take PORO_ (porifera) classes from an OWL file",
             [OWL + "test_prefix_must_match_the_whole_prefix_not_just_its_start"]),
    Mutation("owl_label_any_language",
             [('label = next((t for lang, t in labels if is_english(lang) and t), "")',
               'label = next((t for lang, t in labels if t), "")')],
             "a class whose French label comes first is labelled 'yaourt'; French-only classes become rows",
             [OWL_LABEL + "[fr-then-en]", OWL_LABEL + "[only-fr]", TOY_TABLE]),
    Mutation("owl_synonyms_any_language",
             [('and is_english(e.get(XML_LANG))]', ']')],
             "French / Spanish FOODON synonyms end up in the embedded term text",
             [OWL + "test_synonyms_of_every_kind_are_kept_if_english_and_not_empty", TOY_TABLE]),
    Mutation("owl_only_exact_synonyms",
             [('SYNONYM_TAGS = {OIO + t for t in ["hasExactSynonym", "hasSynonym", "hasNarrowSynonym", '
               '"hasBroadSynonym",\n                                  "hasRelatedSynonym"]}',
               'SYNONYM_TAGS = {OIO + t for t in ["hasExactSynonym"]}')],
             "narrow / broad / related FOODON synonyms are lost (the OBO reader keeps every scope)",
             [OWL + "test_synonyms_of_every_kind_are_kept_if_english_and_not_empty"]),
    Mutation("owl_alternative_term_ignored",
             [('} | {OBO + "IAO_0000118"}', '}')],
             "IAO_0000118 'alternative term' texts are lost",
             [OWL + "test_synonyms_of_every_kind_are_kept_if_english_and_not_empty"]),
    Mutation("owl_restrictions_as_parents",
             [('if e.get(RDF + "resource")]', 'if True]')],
             "an anonymous restriction ('derives from some Bos taurus') crashes or becomes a parent",
             [OWL + "test_parents_are_named_superclasses_only", TOY_TABLE]),
    Mutation("owl_deprecated_any_value",
             [('== "true" for e in cls', '!= "" for e in cls')],
             "owl:deprecated 'false' marks a valid class as obsolete: it disappears from the label set",
             [OWL + "test_obsolete_flag_is_owl_deprecated[false]"]),
    Mutation("owl_definition_dropped",
             [('definition = cls.find(OBO + "IAO_0000115")', 'definition = None')],
             "FOODON terms lose their definition; the OBO and OWL readers no longer agree",
             [OWL + "test_one_class_becomes_one_dict",
              I + "test_same_term_from_obo_or_owl_gives_the_same_row"]),

    # ------------------------------------------------------------------ parse_obo (ENVO, Uberon, PO)
    Mutation("last_stanza_lost",
             [('for line in lines + ["[End]"]:', 'for line in lines:')],
             "the last [Term] of every file is silently dropped (stanzas are closed by the next '[')",
             [OBO + "test_last_stanza_is_kept_when_the_file_ends_right_after_it",
              OBO + "test_one_term_stanza_becomes_one_dict"]),
    Mutation("typedefs_read_as_terms",
             [('if line == "[Term]" else None', '')],
             "a [Typedef] (relation) with an ENVO id becomes a term row",
             [OBO + "test_header_and_typedef_lines_do_not_leak_into_terms"]),
    Mutation("terms_without_name_kept",
             [('term["term_id"].startswith(prefix + "_") and term["label"]', 'term["term_id"].startswith(prefix + "_")')],
             "terms without a name become rows with an empty label (and an empty text to embed)",
             [OBO + "test_term_without_name_is_dropped"]),
    Mutation("every_synonym_type_is_a_language",
             [('if m and m[1] == m[2]}', 'if m}')],
             "any synonym type described as '... synonym' (e.g. SYSTEMATIC) is treated as a translation",
             [OBO + "test_which_synonym_types_count_as_translations[SYSTEMATIC]"]),
    Mutation("translations_kept",
             [('scope_and_type[1] in language_types', 'False')],
             "PO's Spanish and Japanese synonyms ('suelo', 'hoja') end up in the term text",
             [OBO + "test_synonyms[spanish]",
              OBO + "test_which_synonym_types_count_as_translations[Spanish]",
              TOY_TABLE]),
    Mutation("po_scope_suffix_kept",
             [('term["synonyms"].append(SCOPE_SUFFIX.sub("", unescape(match.group(1))))',
               'term["synonyms"].append(unescape(match.group(1)))')],
             "PO synonyms keep their scope: 'foliage leaf (exact)'",
             [OBO + "test_synonyms[po-scope-suffix]",
              I + "test_append_adds_a_new_ontology_and_keeps_the_existing_rows_unchanged"]),
    Mutation("part_of_as_parent",
             [('elif tag == "is_a":', 'elif tag in ("is_a", "relationship"):')],
             "part_of relationships become is_a parents: wrong ancestors for back-off and evaluation",
             [OBO + "test_parents_are_the_is_a_lines_in_order_and_nothing_else", TOY_TABLE]),
    Mutation("is_a_comment_kept",
             [('value.split()[0].replace(":", "_")', 'value.replace(":", "_")')],
             "parents read as 'ENVO_00000001 ! environmental material': no parent id matches a term id",
             [OBO + "test_one_term_stanza_becomes_one_dict",
              OBO + "test_parents_are_the_is_a_lines_in_order_and_nothing_else"]),
    Mutation("is_obsolete_false_read_as_true",
             [('elif tag == "is_obsolete" and value.strip() == "true":', 'elif tag == "is_obsolete":')],
             "'is_obsolete: false' marks a valid term as obsolete",
             [OBO + "test_obsolete_flag[false]"]),

    # ------------------------------------------------------------------ main(): the table and --append
    Mutation("lists_not_deduplicated",
             [('"||".join(dict.fromkeys(values))', '"||".join(values)')],
             "'dirt' given as EXACT and RELATED is written twice: 'dirt||earth||dirt'",
             [TOY_TABLE]),
    Mutation("prefix_case_kept",
             [('prefix.upper()) if owl else parse_obo(read_obo(source), prefix.upper())',
               'prefix) if owl else parse_obo(read_obo(source), prefix)')],
             "--obo envo=envo.obo finds no 'envo_' terms: 0 rows",
             [I + "test_prefix_is_case_insensitive"]),
    Mutation("append_keeps_old_rows_of_reimported_ontology",
             [('kept = kept[~kept["ontology"].isin(new_ontologies)]', 'pass')],
             "re-importing ENVO with --append keeps the old ENVO rows too: every ENVO term twice",
             [I + "test_append_replaces_the_rows_of_an_ontology_given_again"]),
    Mutation("append_reads_NA_as_missing",
             [('read_csv(args.append, sep="\\t", dtype=str, keep_default_na=False)',
               'read_csv(args.append, sep="\\t", dtype=str)')],
             "a kept synonym 'NA' (or 'null', 'None') is read as missing and written back empty",
             [I + "test_append_keeps_texts_that_pandas_would_read_as_missing"]),
    Mutation("append_new_rows_first",
             [('df = pd.concat([kept, df.astype({"obsolete": str})], ignore_index=True)',
               'df = pd.concat([df.astype({"obsolete": str}), kept], ignore_index=True)')],
             "same rows, other order: the documented order (kept rows, then new ones) is not respected",
             [I + "test_append_adds_a_new_ontology_and_keeps_the_existing_rows_unchanged",
              I + "test_append_replaces_the_rows_of_an_ontology_given_again"]),
    Mutation("append_read_without_quoting",
             [('read_csv(args.append, sep="\\t", dtype=str, keep_default_na=False)',
               'read_csv(args.append, sep="\\t", dtype=str, keep_default_na=False, quoting=3)')],
             "the kept table is read with QUOTE_NONE: FOODON's 697 multi-line definitions break into extra rows",
             [I + "test_multiline_owl_texts_stay_one_row_also_through_append"]),
    Mutation("missing_source_read_as_empty",
             [('        with open(source, "rb") as handle:\n            data = handle.read()',
               '        try:\n            data = open(source, "rb").read()\n        except FileNotFoundError:\n            data = b""')],
             "a mistyped path gives 0 terms and the run overwrites the table without that ontology",
             [I + "test_bad_input_stops_the_run_and_leaves_the_existing_table_alone[missing-source]"]),

    # ------------------------------------------------------------------ the fixes of 2026-10-19, undone
    Mutation("undo_fix_http_prefix",
             [('if source.startswith(("http://", "https://")):', 'if source.startswith("http"):')],
             "a local path like http_mirror/envo.obo is sent to urlopen: 'unknown url type'",
             [U + "TestReadBytes::test_local_path_that_starts_with_http_is_read_from_disk"]),
    Mutation("undo_fix_synonym_unescape",
             [('SCOPE_SUFFIX.sub("", unescape(match.group(1)))', 'SCOPE_SUFFIX.sub("", match.group(1))')],
             'an OBO synonym "5\\" end" keeps its backslash',
             [OBO + "test_escaped_quote_in_synonym_is_unescaped_like_in_definitions"]),
    Mutation("undo_fix_empty_table",
             [('df = pd.DataFrame(rows, columns=COLUMNS)', 'df = pd.DataFrame(rows)')],
             "a run where no source has terms (mistyped prefix) stops with KeyError: 'synonyms'",
             [I + "test_no_terms_at_all_writes_a_table_with_only_the_header"]),
]


def run_on_copy(tmp_path, source, node_ids=()):
    """Run the step-1 tests (all, or only `node_ids`) against `source` as 1_build_term_index.py.
    Returns {node id: Outcome}."""
    tests = build_copy(tmp_path, {SCRIPT.name: source, "common.py": SUPPORT[0], "tests/step_loader.py": SUPPORT[1],
                                  **{f"tests/{name}": HERE / name for name in TEST_FILES}})
    return run_pytest(tmp_path, tests, node_ids or TEST_FILES)


def test_unmutated_copy_passes(tmp_path):
    check_unmutated(run_on_copy(tmp_path, SCRIPT.read_text()), MUTATIONS, min_tests=70)


@pytest.mark.parametrize("mutation", MUTATIONS, ids=[m.name for m in MUTATIONS])
def test_mutation_is_caught(mutation, tmp_path):
    check_caught(mutation, run_on_copy(tmp_path, mutate(SCRIPT, mutation.edits), mutation.caught_by))
