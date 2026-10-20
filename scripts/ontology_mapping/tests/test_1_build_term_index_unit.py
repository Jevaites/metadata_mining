"""Unit tests for 1_build_term_index.py: each function on its own, with inputs small enough to read.

Functions under test, in the order main() uses them:
    read_bytes(source)        bytes of a local path or a URL, gunzipped when the name ends in .gz
    read_obo(source)          an OBO file as a list of text lines
    parse_obo(lines, prefix)  one dict per [Term] stanza whose id starts with PREFIX:
    is_english(lang)          is an OWL xml:lang tag English?
    parse_owl(data, prefix)   one dict per owl:Class whose IRI ends in PREFIX_..., same keys as parse_obo
main() (the command line, with the "||"-joined TSV it writes) is in test_1_build_term_index_integration.py.

Both parsers return a list of dicts like
    {"ontology": "ENVO", "term_id": "ENVO_00000002", "label": "soil", "synonyms": ["dirt"],
     "definition": "...", "parents": ["ENVO_00000001"], "obsolete": False}
The term() helper below builds that dict, so an expected value reads as term("ENVO_00000002", "soil", ...).

Tests whose docstring starts with "Regression (fixed 2026-10-19)" cover bugs found while writing these
tests; test_1_build_term_index_mutations.py re-introduces each bug and checks that its test catches it.
"""

import gzip
import textwrap

import pytest

from step_loader import fake_web, load_step  # noqa: F401 (fake_web is a fixture used by name)

bti = load_step("1_build_term_index.py", "build_term_index")


# ----------------------------------------------------------------------------- helpers
def term(term_id, label, synonyms=(), definition="", parents=(), obsolete=False):
    """The dict a parser returns for one term; the ontology is the id's prefix."""
    return {"ontology": term_id.split("_")[0], "term_id": term_id, "label": label, "synonyms": list(synonyms),
            "definition": definition, "parents": list(parents), "obsolete": obsolete}


def obo_lines(text):
    """OBO text written indented in a test -> the list of lines parse_obo receives from read_obo."""
    return textwrap.dedent(text).strip("\n").splitlines()


def owl(classes):
    """owl:Class elements written indented in a test -> the RDF/XML bytes parse_owl receives from read_bytes."""
    header = ('<?xml version="1.0"?>\n'
              '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
              '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n'
              '         xmlns:owl="http://www.w3.org/2002/07/owl#"\n'
              '         xmlns:obo="http://purl.obolibrary.org/obo/"\n'
              '         xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#">\n')
    return (header + textwrap.dedent(classes) + "</rdf:RDF>\n").encode("utf-8")


IRI = "http://purl.obolibrary.org/obo/"  # OBO class IRIs are IRI + "FOODON_00000002"


# ============================================================================= read_bytes / read_obo
class TestReadBytes:
    def test_local_file_is_returned_unchanged(self, tmp_path):
        path = tmp_path / "toy.obo"
        path.write_bytes(b"format-version: 1.2\n")
        assert bti.read_bytes(str(path)) == b"format-version: 1.2\n"

    def test_local_gz_file_is_decompressed(self, tmp_path):
        path = tmp_path / "toy.obo.gz"
        path.write_bytes(gzip.compress(b"format-version: 1.2\n"))
        assert bti.read_bytes(str(path)) == b"format-version: 1.2\n"

    def test_url_is_downloaded(self, fake_web):
        fake_web.pages["https://example.org/toy.obo"] = b"format-version: 1.2\n"
        assert bti.read_bytes("https://example.org/toy.obo") == b"format-version: 1.2\n"
        assert fake_web.requested == ["https://example.org/toy.obo"]

    def test_gz_url_is_downloaded_then_decompressed(self, fake_web):
        fake_web.pages["https://example.org/toy.owl.gz"] = gzip.compress(b"<rdf:RDF/>")
        assert bti.read_bytes("https://example.org/toy.owl.gz") == b"<rdf:RDF/>"

    def test_local_path_that_starts_with_http_is_read_from_disk(self, tmp_path, monkeypatch):
        """Regression (fixed 2026-10-19): every source starting with 'http' was sent to urlopen, so a local
        path like 'http_mirror/envo.obo' failed with 'unknown url type'. Now only http:// and https:// are URLs."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "http_mirror").mkdir()
        (tmp_path / "http_mirror" / "envo.obo").write_bytes(b"format-version: 1.2\n")
        assert bti.read_bytes("http_mirror/envo.obo") == b"format-version: 1.2\n"


def test_read_obo_decodes_utf8_and_splits_lines_also_windows_line_ends(tmp_path):
    path = tmp_path / "toy.obo"
    path.write_bytes("[Term]\r\nname: café\r\n".encode("utf-8"))
    assert bti.read_obo(str(path)) == ["[Term]", "name: café"]  # no '\r' left, 'é' decoded


# ============================================================================= parse_obo
class TestParseObo:
    def test_one_term_stanza_becomes_one_dict(self):
        # a real envo.obo stanza (2026-06-26 release), two xref lines left out
        lines = obo_lines('''
            [Term]
            id: ENVO:00000019
            name: saline lake
            def: "A lake whose water contains a considerable concentration of dissolved salts." [MA:ma]
            synonym: "salina" RELATED [USGS:SDTS]
            synonym: "salt lake" EXACT [USGS:SDTS]
            synonym: "soda lake" RELATED []
            xref: FTT:221
            xref: https://en.wikipedia.org/wiki/Saline_lake
            is_a: ENVO:00000020 ! lake
            is_a: ENVO:01001319 ! saline water body
            intersection_of: ENVO:00000020 ! lake
            intersection_of: RO:0002473 ENVO:00002010 ! composed primarily of saline water
            ''')
        assert bti.parse_obo(lines, "ENVO") == [
            term("ENVO_00000019", "saline lake", synonyms=["salina", "salt lake", "soda lake"],
                 definition="A lake whose water contains a considerable concentration of dissolved salts.",
                 parents=["ENVO_00000020", "ENVO_01001319"])]
        # ids: ':' -> '_'; '! comments', [xrefs], xref: and intersection_of: lines are dropped

    def test_only_terms_of_the_requested_ontology_are_kept_but_foreign_parents_are(self):
        # ENVO files also declare terms of other ontologies (here CHEBI water); only ENVO_ terms are rows.
        # A parent in another ontology (BFO) is still kept: the hierarchy needs it.
        lines = obo_lines('''
            [Term]
            id: ENVO:00000001
            name: environmental material
            is_a: BFO:0000040 ! material entity

            [Term]
            id: CHEBI:15377
            name: water
            ''')
        assert bti.parse_obo(lines, "ENVO") == [term("ENVO_00000001", "environmental material",
                                                     parents=["BFO_0000040"])]

    def test_prefix_must_match_the_whole_prefix_not_just_its_start(self):
        # PO (plant ontology) vs PORO (porifera ontology): 'PORO_...' starts with 'PO' but not with 'PO_'
        lines = obo_lines('''
            [Term]
            id: PO:0000002
            name: leaf

            [Term]
            id: PORO:0000001
            name: sponge
            ''')
        assert [t["term_id"] for t in bti.parse_obo(lines, "PO")] == ["PO_0000002"]

    def test_term_without_name_is_dropped(self):
        lines = obo_lines('''
            [Term]
            id: ENVO:00000005
            ''')
        assert bti.parse_obo(lines, "ENVO") == []

    def test_header_and_typedef_lines_do_not_leak_into_terms(self):
        # The header (before the first stanza) and [Typedef] stanzas (relations) also have 'id:' / 'name:'
        # lines: they must neither become terms, even with an ENVO id, nor overwrite the term before them.
        lines = obo_lines('''
            format-version: 1.2
            ontology: envo

            [Term]
            id: ENVO:00000002
            name: soil

            [Typedef]
            id: part_of
            name: part of
            is_a: overlaps

            [Typedef]
            id: ENVO:00000099
            name: has quality
            ''')
        assert bti.parse_obo(lines, "ENVO") == [term("ENVO_00000002", "soil")]

    def test_last_stanza_is_kept_when_the_file_ends_right_after_it(self):
        # stanzas are closed when the next '[...]' starts; the last one is closed by an '[End]' sentinel
        lines = ["[Term]", "id: ENVO:00000001", "name: environmental material",
                 "[Term]", "id: ENVO:00000002", "name: soil"]  # no blank line, no newline at the end
        assert [t["label"] for t in bti.parse_obo(lines, "ENVO")] == ["environmental material", "soil"]

    # obsolete terms are kept (flagged), not dropped: Metalog and submitters still use their codes
    @pytest.mark.parametrize("obsolete_line, expected", [
        ("is_obsolete: true", True),
        ("is_obsolete: false", False),
        ("", False),  # no is_obsolete line at all
    ], ids=["true", "false", "absent"])
    def test_obsolete_flag(self, obsolete_line, expected):
        lines = ["[Term]", "id: ENVO:00000009", "name: obsolete mud", obsolete_line]
        assert bti.parse_obo(lines, "ENVO")[0]["obsolete"] is expected

    @pytest.mark.parametrize("def_line, expected", [
        ('def: "A porous material." [ENVO:cjm]', "A porous material."),        # xrefs dropped
        (r'def: "A \"loose\" material." []', 'A "loose" material.'),            # \" unescaped
        ('def: "Soil, or dirt; see below." [] {comment="x"}', "Soil, or dirt; see below."),  # trailing {...} dropped
        ("def: no quotes at all", ""),                                          # malformed -> empty
        ("", ""),                                                               # no def line
    ], ids=["xrefs", "escaped-quotes", "trailing-modifier", "unquoted", "missing"])
    def test_definition(self, def_line, expected):
        lines = ["[Term]", "id: ENVO:00000002", "name: soil", def_line]
        assert bti.parse_obo(lines, "ENVO")[0]["definition"] == expected

    # The two synonym types of po.obo used below: 'Spanish' is a translation (its description is
    # "Spanish synonym"), 'OMO:0003004' (plural form, also in envo.obo) is not. Translations are skipped,
    # every other synonym is kept.
    SYNONYM_TYPEDEFS = ['synonymtypedef: Spanish "Spanish synonym (exact)" EXACT',
                        'synonymtypedef: OMO:0003004 "plural form"']

    @pytest.mark.parametrize("synonym_line, expected", [
        ('synonym: "dirt" EXACT []', ["dirt"]),                 # every scope is kept ...
        ('synonym: "earth" RELATED []', ["earth"]),
        ('synonym: "topsoil" NARROW []', ["topsoil"]),
        ('synonym: "ground" BROAD []', ["ground"]),
        ('synonym: "dirt" EXACT [ENVO:cjm, Wikipedia:Soil]', ["dirt"]),   # xrefs dropped
        ('synonym: "pingos" EXACT OMO:0003004 []', ["pingos"]),  # ... and typed synonyms that are not a language
        # envo.obo: a {comment} with escaped quotes after the synonym does not change it
        (r'synonym: "drift ice" BROAD [] {comment="refer to this class simply as \"drift ice\"."}', ["drift ice"]),
        ('synonym: "planta entera (Spanish, exact)" EXACT Spanish [POC:Maria_Alejandra_Gandolfo]', []),  # translation
        ('synonym: "suelo" EXACT Spanish', []),                 # also without the [xrefs] list
        # PO repeats the scope inside the text; that suffix is removed (only at the very end)
        ('synonym: "pollen sac wall (exact)" EXACT []', ["pollen sac wall"]),
        ('synonym: "frutices (narrow, plural)" NARROW OMO:0003004 [FNA:ec8c2064]', ["frutices"]),
        ('synonym: "leaf (narrow) tip" NARROW []', ["leaf (narrow) tip"]),
    ], ids=["exact", "related", "narrow", "broad", "xrefs", "plural-type", "trailing-comment", "spanish",
        "spanish-no-xrefs", "po-scope-suffix", "po-scope-suffix-with-comma", "scope-not-at-end"])
    def test_synonyms(self, synonym_line, expected):
        lines = self.SYNONYM_TYPEDEFS + ["[Term]", "id: ENVO:00000002", "name: soil", synonym_line]
        assert bti.parse_obo(lines, "ENVO")[0]["synonyms"] == expected

    @pytest.mark.parametrize("typedef_line, synonym_type, is_translation", [
        # rule: a synonym type is a language when its description starts with "<the type's own name> synonym"
        ('synonymtypedef: Spanish "Spanish synonym (exact)" EXACT', "Spanish", True),
        ('synonymtypedef: Japanese "Japanese synonym" EXACT', "Japanese", True),
        ('synonymtypedef: SYSTEMATIC "Systematic synonym" EXACT', "SYSTEMATIC", False),  # case differs
        ('synonymtypedef: OMO:0003004 "plural form"', "OMO:0003004", False),
    ], ids=["Spanish", "Japanese", "SYSTEMATIC", "OMO-plural"])
    def test_which_synonym_types_count_as_translations(self, typedef_line, synonym_type, is_translation):
        lines = [typedef_line, "[Term]", "id: PO:0000002", "name: leaf", f'synonym: "x" EXACT {synonym_type} []']
        expected = [] if is_translation else ["x"]
        assert bti.parse_obo(lines, "PO")[0]["synonyms"] == expected

    def test_known_quirk_untyped_translations_and_trailing_spaces_are_kept(self):
        """Known quirk (left as is, see the README): ~35 PO translations have no synonym type, only the
        language inside the text, so they are kept (32 PO terms in ontology_terms.tsv.gz); a double space
        before PO's scope suffix leaves a trailing space (64 PO synonyms). Real po.obo lines."""
        lines = ["[Term]", "id: PO:0000043", "name: crown root",
                 'synonym: "adventitious root  (broad)" BROAD []',
                 'synonym: "pedicelo (Spanish, broad)" BROAD [POC:Maria_Alejandra_Gandolfo]']
        assert bti.parse_obo(lines, "PO")[0]["synonyms"] == ["adventitious root ", "pedicelo (Spanish, broad)"]

    def test_parents_are_the_is_a_lines_in_order_and_nothing_else(self):
        lines = obo_lines('''
            [Term]
            id: ENVO:00000003
            name: sediment
            is_a: ENVO:00000001 ! environmental material
            is_a: ENVO:00000004 {is_inferred="true"} ! granular material
            relationship: BFO:0000050 ENVO:00000099 ! part of sea floor
            intersection_of: ENVO:00000001
            ''')
        # relationship (BFO:0000050 = part_of, as envo.obo writes it) and intersection_of are not is_a parents
        assert bti.parse_obo(lines, "ENVO")[0]["parents"] == ["ENVO_00000001", "ENVO_00000004"]

    def test_escaped_quote_in_synonym_is_unescaped_like_in_definitions(self):
        """Regression (fixed 2026-10-19): \\" was unescaped in definitions but not in synonyms, which
        kept the backslash ('5\\" end'). Both now go through unescape()."""
        lines = ["[Term]", "id: UBERON:0000001", "name: x", r'synonym: "5\" end" EXACT []']
        assert bti.parse_obo(lines, "UBERON")[0]["synonyms"] == ['5" end']


# ============================================================================= is_english
@pytest.mark.parametrize("lang, expected", [
    (None, True),            # no xml:lang attribute
    ("", True),
    ("en", True),
    ("EN", True),            # case-insensitive
    ("en-gb", True),         # regional English (FOODON: 'yoghurt', 'cornflour')
    ("en-US", True),
    ("en-scotland", True),
    ("en-br", True),         # FOODON's tag for British English ('yoghurt', 'claret')
    ("fr", False),
    ("ge", False),           # FOODON's (non-standard) tag for German
    ("es", False),
    ("de-en", False),        # German (as spoken in England): the language is the first part
])
def test_is_english(lang, expected):
    assert bti.is_english(lang) is expected


# ============================================================================= parse_owl
class TestParseOwl:
    def test_one_class_becomes_one_dict(self):
        data = owl(f'''
            <owl:Class rdf:about="{IRI}FOODON_00000002">
                <rdfs:label xml:lang="en">yogurt</rdfs:label>
                <obo:IAO_0000115>A fermented milk food.</obo:IAO_0000115>
                <oboInOwl:hasExactSynonym xml:lang="en-gb">yoghurt</oboInOwl:hasExactSynonym>
                <rdfs:subClassOf rdf:resource="{IRI}FOODON_00000001"/>
            </owl:Class>
            ''')
        assert bti.parse_owl(data, "FOODON") == [
            term("FOODON_00000002", "yogurt", synonyms=["yoghurt"], definition="A fermented milk food.",
                 parents=["FOODON_00000001"])]  # IRIs are cut to their last part: .../obo/FOODON_00000001

    def test_only_classes_of_the_requested_ontology_are_kept(self):
        data = owl(f'''
            <owl:Class rdf:about="{IRI}FOODON_00000001"><rdfs:label>dairy food</rdfs:label></owl:Class>
            <owl:Class rdf:about="{IRI}CHEBI_15377"><rdfs:label>water</rdfs:label></owl:Class>
            <owl:Class rdf:about="http://example.org/not-an-obo-iri"><rdfs:label>x</rdfs:label></owl:Class>
            ''')
        assert [t["term_id"] for t in bti.parse_owl(data, "FOODON")] == ["FOODON_00000001"]

    def test_prefix_must_match_the_whole_prefix_not_just_its_start(self):
        # as in parse_obo: 'PORO_0000001' (porifera ontology) starts with 'PO' but not with 'PO_'
        data = owl(f'''
            <owl:Class rdf:about="{IRI}PO_0000002"><rdfs:label>leaf</rdfs:label></owl:Class>
            <owl:Class rdf:about="{IRI}PORO_0000001"><rdfs:label>sponge</rdfs:label></owl:Class>
            ''')
        assert [t["term_id"] for t in bti.parse_owl(data, "PO")] == ["PO_0000002"]

    @pytest.mark.parametrize("label_elements, expected_label", [
        ('<rdfs:label>yogurt</rdfs:label>', "yogurt"),                                  # no language tag
        ('<rdfs:label xml:lang="en-us">yogurt</rdfs:label>', "yogurt"),                 # regional English
        ('<rdfs:label xml:lang="fr">yaourt</rdfs:label>'
         '<rdfs:label xml:lang="en">yogurt</rdfs:label>', "yogurt"),                    # first ENGLISH label
        ('<rdfs:label xml:lang="en"> </rdfs:label>'
         '<rdfs:label xml:lang="en">yogurt</rdfs:label>', "yogurt"),                    # first NON-EMPTY label
        ('<rdfs:label xml:lang="fr">yaourt</rdfs:label>', None),                        # no English label
        ('', None),                                                                     # no label at all
    ], ids=["no-lang", "en-us", "fr-then-en", "blank-then-en", "only-fr", "no-label"])
    def test_label_is_the_first_nonempty_english_label_else_the_class_is_dropped(self, label_elements,
                                                                                   expected_label):
        data = owl(f'<owl:Class rdf:about="{IRI}FOODON_00000002">{label_elements}</owl:Class>\n')
        labels = [t["label"] for t in bti.parse_owl(data, "FOODON")]
        assert labels == ([expected_label] if expected_label else [])

    def test_synonyms_of_every_kind_are_kept_if_english_and_not_empty(self):
        data = owl(f'''
            <owl:Class rdf:about="{IRI}FOODON_00000002">
                <rdfs:label>yogurt</rdfs:label>
                <oboInOwl:hasExactSynonym>yoghurt</oboInOwl:hasExactSynonym>
                <oboInOwl:hasSynonym>yoghourt</oboInOwl:hasSynonym>
                <oboInOwl:hasNarrowSynonym>greek yogurt</oboInOwl:hasNarrowSynonym>
                <oboInOwl:hasBroadSynonym>fermented milk</oboInOwl:hasBroadSynonym>
                <oboInOwl:hasRelatedSynonym>curd</oboInOwl:hasRelatedSynonym>
                <obo:IAO_0000118>yogourt</obo:IAO_0000118>
                <oboInOwl:hasExactSynonym xml:lang="en-au">yoghurt cup</oboInOwl:hasExactSynonym>
                <oboInOwl:hasExactSynonym xml:lang="fr">yaourt</oboInOwl:hasExactSynonym>
                <oboInOwl:hasExactSynonym>  </oboInOwl:hasExactSynonym>
                <rdfs:comment>not a synonym</rdfs:comment>
            </owl:Class>
            ''')
        # IAO_0000118 = 'alternative term'; 'yaourt' (fr), the blank one and the comment are left out
        assert bti.parse_owl(data, "FOODON")[0]["synonyms"] == [
            "yoghurt", "yoghourt", "greek yogurt", "fermented milk", "curd", "yogourt", "yoghurt cup"]

    def test_parents_are_named_superclasses_only(self):
        # FOODON_00000015 'cattle bull' as foodon.owl writes it (shortened): its equivalentClass holds an
        # anonymous owl:Class, it has two named parents in other ontologies and one restriction
        data = owl(f'''
            <owl:Class rdf:about="{IRI}FOODON_00000015">
                <owl:equivalentClass>
                    <owl:Class>
                        <owl:intersectionOf rdf:parseType="Collection">
                            <rdf:Description rdf:about="{IRI}NCBITaxon_9913"/>
                            <rdf:Description rdf:about="{IRI}FOODON_99999999"/>
                        </owl:intersectionOf>
                    </owl:Class>
                </owl:equivalentClass>
                <rdfs:subClassOf rdf:resource="{IRI}NCBITaxon_9913"/>
                <rdfs:subClassOf rdf:resource="{IRI}UBERON_0007023"/>
                <rdfs:subClassOf>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{IRI}RO_0000086"/>
                        <owl:someValuesFrom rdf:resource="{IRI}PATO_0002366"/>
                    </owl:Restriction>
                </rdfs:subClassOf>
                <obo:IAO_0000114 rdf:resource="{IRI}IAO_0000122"/>
                <obo:IAO_0000115 xml:lang="en">A cattle bull is an adult  "intact" (i.e. not castrated) male.</obo:IAO_0000115>
                <obo:IAO_0000119 xml:lang="en">https://en.wikipedia.org/wiki/Cattle#Terminology</obo:IAO_0000119>
                <oboInOwl:hasSynonym xml:lang="en">bull</oboInOwl:hasSynonym>
                <rdfs:label xml:lang="en">cattle bull</rdfs:label>
            </owl:Class>
            ''')
        # one row (the anonymous class is not one); parents = the named subClassOf only: not the restriction
        # ('has quality some intact'), not the equivalentClass members; parents in other ontologies are kept;
        # IAO_0000119 (definition source) is not a synonym
        assert bti.parse_owl(data, "FOODON") == [
            term("FOODON_00000015", "cattle bull", synonyms=["bull"],
                 definition='A cattle bull is an adult  "intact" (i.e. not castrated) male.',
                 parents=["NCBITaxon_9913", "UBERON_0007023"])]

    @pytest.mark.parametrize("deprecated_element, expected", [
        ('<owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>', True),
        ('<owl:deprecated>false</owl:deprecated>', False),
        ('', False),
    ], ids=["true", "false", "absent"])
    def test_obsolete_flag_is_owl_deprecated(self, deprecated_element, expected):
        data = owl(f'<owl:Class rdf:about="{IRI}FOODON_00000009"><rdfs:label>obsolete curd</rdfs:label>'
                   f'{deprecated_element}</owl:Class>\n')
        assert bti.parse_owl(data, "FOODON")[0]["obsolete"] is expected

    def test_missing_definition_is_empty_string(self):
        data = owl(f'<owl:Class rdf:about="{IRI}FOODON_00000001"><rdfs:label>dairy food</rdfs:label></owl:Class>\n')
        assert bti.parse_owl(data, "FOODON")[0]["definition"] == ""
