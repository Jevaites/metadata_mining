"""Integration tests for 1_build_term_index.py: the command line, run on toy ontologies.

How the step is used (see its docstring and the README):
    python 1_build_term_index.py --obo ENVO=envo.obo FOODON=foodon.owl.gz --output ontology_terms.tsv.gz
    python 1_build_term_index.py --append old_table.tsv.gz --obo PO=po.obo --output ontology_terms.tsv.gz
Inside main(), for each PREFIX=source:
    read_bytes / read_obo  ->  parse_obo (or parse_owl for .owl / .owl.gz)  ->  list of term dicts
then all dicts become one table: list columns are de-duplicated and joined with "||", and with --append
the old table's rows are kept for every ontology not given again.

The toy ontologies below have a handful of terms, each there for a reason (see the tree above each
one), so the full expected output table can be written out by hand and compared cell by cell.
Output tables are read back as later steps read them (common.read_tsv): every cell a string, empty
cells "" (never NaN), obsolete as "True" / "False".

Tests whose docstring starts with "Regression (fixed 2026-10-19)" cover bugs found while writing these
tests; test_1_build_term_index_mutations.py re-introduces each bug and checks that its test catches it.
"""

import gzip
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from step_loader import SCRIPTS_DIR, fake_web, load_step  # noqa: F401 (fake_web is a fixture used by name)

bti = load_step("1_build_term_index.py", "build_term_index")

COLUMNS = ["ontology", "term_id", "label", "synonyms", "definition", "parents", "obsolete"]


# ============================================================================= toy ontologies
# Toy ENVO (OBO). Indentation = is_a:
#   ENVO_00000001 environmental material   is_a BFO_0000040 (outside ENVO: kept as a parent, not as a row)
#     ENVO_00000002 soil       synonyms 'dirt' (given twice -> written once), 'earth';
#                              'suelo' is Spanish -> dropped; definition with quotes and a comma
#     ENVO_00000003 sediment   is_a ENVO_00000001 given twice (asserted + inferred) -> written once;
#                              its part_of relationship (BFO:0000050) is not a parent
#   ENVO_00000009 obsolete mud (obsolete: kept, flagged True)
#   CHEBI_15377   water        (a CHEBI term inside the ENVO file -> no row)
TOY_ENVO_OBO = r'''
format-version: 1.2
synonymtypedef: Spanish "Spanish synonym (exact)" EXACT
ontology: envo

[Term]
id: ENVO:00000001
name: environmental material
is_a: BFO:0000040 ! material entity

[Term]
id: ENVO:00000002
name: soil
def: "A \"loose\", porous material at the surface." [ENVO:cjm]
synonym: "dirt" EXACT []
synonym: "earth" RELATED []
synonym: "dirt" RELATED []
synonym: "suelo" EXACT Spanish []
is_a: ENVO:00000001 ! environmental material

[Term]
id: ENVO:00000003
name: sediment
is_a: ENVO:00000001 ! environmental material
is_a: ENVO:00000001 {is_inferred="true"} ! environmental material
relationship: BFO:0000050 ENVO:00000099 ! part of sea floor

[Term]
id: ENVO:00000009
name: obsolete mud
is_obsolete: true

[Term]
id: CHEBI:15377
name: water

[Typedef]
id: part_of
name: part of
'''
ENVO_ROWS = [
    # ontology term_id         label                     synonyms       definition                                     parents         obsolete
    ("ENVO", "ENVO_00000001", "environmental material", "",            "",                                            "BFO_0000040",   "False"),
    ("ENVO", "ENVO_00000002", "soil",                   "dirt||earth", 'A "loose", porous material at the surface.', "ENVO_00000001", "False"),
    ("ENVO", "ENVO_00000003", "sediment",               "",            "",                                            "ENVO_00000001", "False"),
    ("ENVO", "ENVO_00000009", "obsolete mud",           "",            "",                                            "",              "True"),
]

# Toy FOODON (OWL, the format FOODON is released in). Indentation = rdfs:subClassOf:
#   FOODON_00000001 dairy food
#     FOODON_00000002 yogurt   French label listed first -> the English one is used;
#                              synonyms 'yoghurt' (en-gb, given twice -> once), 'yaourt' (fr -> dropped);
#                              'derives from some Bos taurus' is a restriction, not a parent
#   FOODON_00000009 obsolete curd (owl:deprecated)
#   CHEBI_15377     water    (not a FOODON class -> no row)
TOY_FOODON_OWL = '''<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:obo="http://purl.obolibrary.org/obo/"
         xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00000001">
        <rdfs:label xml:lang="en">dairy food</rdfs:label>
    </owl:Class>
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00000002">
        <rdfs:label xml:lang="fr">yaourt</rdfs:label>
        <rdfs:label xml:lang="en">yogurt</rdfs:label>
        <obo:IAO_0000115 xml:lang="en">A fermented milk food.</obo:IAO_0000115>
        <oboInOwl:hasExactSynonym xml:lang="en-gb">yoghurt</oboInOwl:hasExactSynonym>
        <oboInOwl:hasRelatedSynonym>yoghurt</oboInOwl:hasRelatedSynonym>
        <oboInOwl:hasExactSynonym xml:lang="fr">yaourt</oboInOwl:hasExactSynonym>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/FOODON_00000001"/>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0001000"/>
                <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/NCBITaxon_9913"/>
            </owl:Restriction>
        </rdfs:subClassOf>
    </owl:Class>
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00000009">
        <rdfs:label>obsolete curd</rdfs:label>
        <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
    </owl:Class>
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/CHEBI_15377">
        <rdfs:label>water</rdfs:label>
    </owl:Class>
</rdf:RDF>
'''
FOODON_ROWS = [
    ("FOODON", "FOODON_00000001", "dairy food",    "",        "",                       "",                "False"),
    ("FOODON", "FOODON_00000002", "yogurt",        "yoghurt", "A fermented milk food.", "FOODON_00000001", "False"),
    ("FOODON", "FOODON_00000009", "obsolete curd", "",        "",                       "",                "True"),
]

# Toy PO (OBO), added later with --append. PO writes the scope inside the synonym text ('(exact)'),
# which is removed, and has translated synonyms (Spanish, Japanese), which are dropped.
#   PO_0000001 plant structure
#     PO_0000002 leaf   synonyms 'foliage leaf'
TOY_PO_OBO = '''
format-version: 1.2
synonymtypedef: Spanish "Spanish synonym (exact)" EXACT
synonymtypedef: Japanese "Japanese synonym (exact)" EXACT

[Term]
id: PO:0000001
name: plant structure

[Term]
id: PO:0000002
name: leaf
synonym: "foliage leaf (exact)" EXACT []
synonym: "hoja (exact)" EXACT Spanish []
synonym: "葉 (exact)" EXACT Japanese []
is_a: PO:0000001 ! plant structure
'''
PO_ROWS = [
    ("PO", "PO_0000001", "plant structure", "",             "", "",           "False"),
    ("PO", "PO_0000002", "leaf",            "foliage leaf", "", "PO_0000001", "False"),
]

# A new ENVO release, to re-import over an existing table: 'sediment' is gone, 'sand' is new,
# the obsolete term is no longer shipped.
TOY_ENVO_V2_OBO = '''
[Term]
id: ENVO:00000001
name: environmental material

[Term]
id: ENVO:00000002
name: soil
synonym: "dirt" EXACT []
is_a: ENVO:00000001 ! environmental material

[Term]
id: ENVO:00000004
name: sand
is_a: ENVO:00000001 ! environmental material
'''
ENVO_V2_ROWS = [
    ("ENVO", "ENVO_00000001", "environmental material", "",     "", "",              "False"),
    ("ENVO", "ENVO_00000002", "soil",                   "dirt", "", "ENVO_00000001", "False"),
    ("ENVO", "ENVO_00000004", "sand",                   "",     "", "ENVO_00000001", "False"),
]


# ============================================================================= helpers
def write(path, text):
    """Write a toy ontology (gzipped when the name ends in .gz, like the real downloads); return its path."""
    data = textwrap.dedent(text).lstrip("\n").encode("utf-8")
    path.write_bytes(gzip.compress(data) if path.name.endswith(".gz") else data)
    return str(path)


@pytest.fixture
def run_step(monkeypatch):
    """run_step('--obo', 'ENVO=envo.obo', '--output', 'out.tsv') == running the script with those arguments."""
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["1_build_term_index.py", *map(str, args)])
        bti.main()
    return run


def read_table(path):
    """The output table as later steps read it (common.read_tsv): all strings, empty cells stay ""."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def assert_table(path, expected_rows):
    pd.testing.assert_frame_equal(read_table(path), pd.DataFrame(expected_rows, columns=COLUMNS))


# ============================================================================= basic workflow
def test_build_a_table_from_an_obo_and_an_owl_file(tmp_path, run_step, capsys):
    envo = write(tmp_path / "envo.obo", TOY_ENVO_OBO)
    foodon = write(tmp_path / "foodon.owl.gz", TOY_FOODON_OWL)  # .owl(.gz) -> parse_owl, anything else -> parse_obo
    out = tmp_path / "ontology_terms.tsv"

    run_step("--obo", f"ENVO={envo}", f"FOODON={foodon}", "--output", out)

    assert_table(out, ENVO_ROWS + FOODON_ROWS)  # one row per term, ontologies in the order given
    log = capsys.readouterr().out
    assert f"ENVO: 4 terms (1 obsolete) from {envo}" in log
    assert f"FOODON: 3 terms (1 obsolete) from {foodon}" in log
    assert "Wrote 7 terms" in log


def test_later_steps_can_load_the_table(tmp_path, run_step):
    # The hand-off to steps 2-7: common.load_terms() drops the obsolete rows (obsolete == "True")
    # and builds the text that gets embedded, 'label; synonym; synonym'.
    pytest.importorskip("sklearn")  # common.py imports it
    import common

    out = tmp_path / "ontology_terms.tsv.gz"
    run_step("--obo", f"ENVO={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}", "--output", out)

    terms = common.load_terms(str(out))
    assert list(terms["term_id"]) == ["ENVO_00000001", "ENVO_00000002", "ENVO_00000003"]  # no obsolete mud
    assert terms.loc[1, "text"] == "soil; dirt; earth"


def test_gz_output_is_gzip_compressed(tmp_path, run_step):
    out = tmp_path / "ontology_terms.tsv.gz"
    run_step("--obo", f"ENVO={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}", "--output", out)
    assert out.read_bytes()[:2] == b"\x1f\x8b"  # gzip magic number (pandas compresses by file extension)
    assert_table(out, ENVO_ROWS)


def test_sources_can_be_urls(tmp_path, run_step, fake_web):
    # No network: fake_web serves these bytes for these URLs (see step_loader.py)
    fake_web.pages["https://example.org/envo.obo"] = textwrap.dedent(TOY_ENVO_OBO).encode()
    fake_web.pages["https://example.org/foodon.owl.gz"] = gzip.compress(TOY_FOODON_OWL.encode())
    out = tmp_path / "ontology_terms.tsv"

    run_step("--obo", "ENVO=https://example.org/envo.obo", "FOODON=https://example.org/foodon.owl.gz",
             "--output", out)

    assert fake_web.requested == ["https://example.org/envo.obo", "https://example.org/foodon.owl.gz"]
    assert_table(out, ENVO_ROWS + FOODON_ROWS)  # same table as from local files


def test_command_line_as_documented(tmp_path):
    # The real script in a fresh Python process, exactly as in the README
    envo = write(tmp_path / "envo.obo", TOY_ENVO_OBO)
    out = tmp_path / "ontology_terms.tsv.gz"
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "1_build_term_index.py"),
                             "--obo", f"ENVO={envo}", "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Wrote 4 terms" in result.stdout
    assert_table(out, ENVO_ROWS)


# ============================================================================= --append workflow
def test_append_adds_a_new_ontology_and_keeps_the_existing_rows_unchanged(tmp_path, run_step, capsys):
    # Day 1: an ENVO table.  Day 2: add PO without re-downloading ENVO (the README's step 1).
    old = tmp_path / "ontology_terms_envo.tsv.gz"
    run_step("--obo", f"ENVO={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}", "--output", old)

    new = tmp_path / "ontology_terms.tsv.gz"
    run_step("--append", old, "--obo", f"PO={write(tmp_path / 'po.obo', TOY_PO_OBO)}", "--output", new)

    # the ENVO rows are exactly the old ones (empty cells still empty, obsolete still "True"/"False"),
    # and the PO rows come after them
    assert_table(new, ENVO_ROWS + PO_ROWS)
    assert "Kept 4 terms of ['ENVO']" in capsys.readouterr().out


def test_append_replaces_the_rows_of_an_ontology_given_again(tmp_path, run_step):
    # An existing ENVO + PO table, then a new ENVO release: its rows replace all the old ENVO rows
    # ('sediment' and 'obsolete mud' disappear, 'sand' appears) and PO is untouched.
    old = tmp_path / "old.tsv.gz"
    run_step("--obo", f"ENVO={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}",
             f"PO={write(tmp_path / 'po.obo', TOY_PO_OBO)}", "--output", old)
    assert_table(old, ENVO_ROWS + PO_ROWS)

    new = tmp_path / "new.tsv.gz"
    run_step("--append", old, "--obo", f"ENVO={write(tmp_path / 'envo_v2.obo', TOY_ENVO_V2_OBO)}", "--output", new)

    assert_table(new, PO_ROWS + ENVO_V2_ROWS)  # kept rows first, then the newly parsed ones


# ============================================================================= edge cases
def test_append_keeps_texts_that_pandas_would_read_as_missing(tmp_path, run_step):
    # pandas reads 'NA', 'null', 'None', 'nan', ... as NaN by default, and NaN is written back as an
    # empty cell. The old table is therefore read with keep_default_na=False, so an abbreviation
    # synonym 'NA' (real: UBERON_2005085 nasal artery) survives being carried over by --append.
    uberon = write(tmp_path / "uberon.obo", '''
        [Term]
        id: UBERON:2005085
        name: nasal artery
        synonym: "NA" EXACT []
        ''')
    old = tmp_path / "old.tsv"
    run_step("--obo", f"UBERON={uberon}", "--output", old)

    new = tmp_path / "new.tsv"
    run_step("--append", old, "--obo", f"PO={write(tmp_path / 'po.obo', TOY_PO_OBO)}", "--output", new)

    nasal_artery = ("UBERON", "UBERON_2005085", "nasal artery", "NA", "", "", "False")
    assert_table(new, [nasal_artery] + PO_ROWS)


def test_multiline_owl_texts_stay_one_row_also_through_append(tmp_path, run_step):
    # 697 FOODON definitions (and 2 synonyms) span several lines. The TSV quotes such cells, so the
    # table, and the table re-read by --append, still has one row per term with the text unchanged.
    foodon = write(tmp_path / "foodon.owl", '''<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
                 xmlns:owl="http://www.w3.org/2002/07/owl#"
                 xmlns:obo="http://purl.obolibrary.org/obo/">
            <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00001730">
                <rdfs:label xml:lang="en">chicory</rdfs:label>
                <obo:IAO_0000115 xml:lang="en">Blue daisy, coffeeweed.

Note: "Cornflower" is commonly applied to Centaurea cyanus.</obo:IAO_0000115>
            </owl:Class>
        </rdf:RDF>
        ''')
    old = tmp_path / "old.tsv.gz"
    run_step("--obo", f"FOODON={foodon}", "--output", old)
    new = tmp_path / "new.tsv.gz"
    run_step("--append", old, "--obo", f"PO={write(tmp_path / 'po.obo', TOY_PO_OBO)}", "--output", new)

    definition = 'Blue daisy, coffeeweed.\n\nNote: "Cornflower" is commonly applied to Centaurea cyanus.'
    chicory = ("FOODON", "FOODON_00001730", "chicory", "", definition, "", "False")
    assert_table(old, [chicory])
    assert_table(new, [chicory] + PO_ROWS)


@pytest.mark.parametrize("make_args, error", [
    (lambda d: ["--obo", f"ENVO={d / 'missing.obo'}"], FileNotFoundError),                  # mistyped path
    (lambda d: ["--obo", str(d / "envo.obo")], ValueError),                                 # no PREFIX=
    (lambda d: ["--append", d / "missing.tsv.gz", "--obo", f"ENVO={d / 'envo.obo'}"], FileNotFoundError),
    (lambda d: ["--obo", f"FOODON={d / 'cut.owl'}"], ET.ParseError),                       # truncated download
], ids=["missing-source", "pair-without-prefix", "missing-append-table", "truncated-owl"])
def test_bad_input_stops_the_run_and_leaves_the_existing_table_alone(tmp_path, run_step, make_args, error):
    # The README writes over the table the next steps use: a failed run must not replace it with a
    # partial one (e.g. without ENVO because envo.obo was mistyped).
    write(tmp_path / "envo.obo", TOY_ENVO_OBO)
    (tmp_path / "cut.owl").write_text(TOY_FOODON_OWL[:600])
    out = tmp_path / "ontology_terms.tsv"
    out.write_text("the table of the last good run\n")
    with pytest.raises(error):
        run_step(*make_args(tmp_path), "--output", out)
    assert out.read_text() == "the table of the last good run\n"


def test_prefix_is_case_insensitive(tmp_path, run_step):
    out = tmp_path / "ontology_terms.tsv"
    run_step("--obo", f"envo={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}", "--output", out)
    assert_table(out, ENVO_ROWS)  # ontology column 'ENVO', ids 'ENVO_...'


def test_same_term_from_obo_or_owl_gives_the_same_row(tmp_path, run_step):
    # The two readers are interchangeable: one term, written in both formats, gives one identical row
    as_obo = write(tmp_path / "yogurt.obo", '''
        [Term]
        id: FOODON:00000002
        name: yogurt
        def: "A fermented milk food." []
        synonym: "yoghurt" EXACT []
        is_a: FOODON:00000001 ! dairy food
        ''')
    as_owl = write(tmp_path / "yogurt.owl", '''<?xml version="1.0"?>
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
                 xmlns:owl="http://www.w3.org/2002/07/owl#"
                 xmlns:obo="http://purl.obolibrary.org/obo/"
                 xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">
            <owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_00000002">
                <rdfs:label>yogurt</rdfs:label>
                <obo:IAO_0000115>A fermented milk food.</obo:IAO_0000115>
                <oboInOwl:hasExactSynonym>yoghurt</oboInOwl:hasExactSynonym>
                <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/FOODON_00000001"/>
            </owl:Class>
        </rdf:RDF>
        ''')
    run_step("--obo", f"FOODON={as_obo}", "--output", tmp_path / "from_obo.tsv")
    run_step("--obo", f"FOODON={as_owl}", "--output", tmp_path / "from_owl.tsv")

    yogurt = [("FOODON", "FOODON_00000002", "yogurt", "yoghurt", "A fermented milk food.", "FOODON_00000001", "False")]
    assert_table(tmp_path / "from_obo.tsv", yogurt)
    assert_table(tmp_path / "from_owl.tsv", yogurt)


def test_a_source_without_terms_of_its_prefix_adds_no_rows(tmp_path, run_step, capsys):
    # e.g. a mistyped pair, UBERON=envo.obo: nothing fails, the log line '0 terms' is the only sign
    envo = write(tmp_path / "envo.obo", TOY_ENVO_OBO)
    out = tmp_path / "ontology_terms.tsv"
    run_step("--obo", f"ENVO={envo}", f"UBERON={envo}", "--output", out)
    assert_table(out, ENVO_ROWS)
    assert f"UBERON: 0 terms (0 obsolete) from {envo}" in capsys.readouterr().out


def test_no_terms_at_all_writes_a_table_with_only_the_header(tmp_path, run_step):
    """Regression (fixed 2026-10-19): with no term at all (e.g. a single mistyped prefix), pd.DataFrame([])
    had no columns and main() stopped with KeyError: 'synonyms'. The columns are now always given."""
    out = tmp_path / "ontology_terms.tsv"
    run_step("--obo", f"UBERON={write(tmp_path / 'envo.obo', TOY_ENVO_OBO)}", "--output", out)
    assert_table(out, [])
