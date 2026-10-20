"""Unit tests for 6c_flag_controls.py: each function on its own, with inputs small enough to read.

Functions under test:
    evidence(sub_biome, keywords)   '' for a real habitat sample, else WHY it is a control / mock:
                                      "sub-biome: <sub-biome>"                (rule 1)
                                      "keyword: <keyword>"                    (rule 2, strong keyword)
                                      "keyword + lab context: <keyword>"      (rule 3, weak keyword + lab cue)
                                    The rules are tried in that order; the first one that fires wins.
                                    '_' is matched as a space; the evidence quotes the text as written.
    split_keywords(text)            GPT keyword set '{a, b}' -> ['a', 'b']
    read_texts(path)                {sample_id: text} of a GPT output file, one 'id<TAB>text' per line
main() (reading both GPT files, splitting the keyword sets, writing the TSV) is in
test_6c_flag_controls_integration.py.

evidence() receives what main() gives it: the sub-biome already stripped, and the keyword set split
into a list of stripped keywords, e.g.
    GPT keywords line   "s1<TAB>{extraction blank, water}"  ->  keywords = ["extraction blank", "water"]
    GPT sub-biome line  "s1<TAB>laboratory control"         ->  sub_biome = "laboratory control"

Conventions (as in the other test files of this folder):
  * Tests named `..._quirk` pin down current behaviour that is surprising; they document it, they do
    not endorse it. If the behaviour is changed on purpose, update the test.
  * Tests marked xfail(strict=True) are KNOWN BUGS: they fail today (reported as "xfailed"). Once the
    bug is fixed they pass, strict=True turns that into a failure ("XPASS(strict)"), and the marker must go.
"""

import pytest

from step_loader import load_step

fc = load_step("6c_flag_controls.py", "flag_controls")

NOT_A_CONTROL = ""  # what evidence() returns for a real habitat sample


# ============================================================================= evidence(): rule 1, sub-biome
class TestSubBiomeRule:
    """The GPT sub-biome alone is enough: no keyword needed (keywords=[] in every test below)."""

    @pytest.mark.parametrize("sub_biome", [
        "laboratory control",      # laboratory / synthetic / sterile  +  control / blank / mock / standard
        "laboratory blank",
        "laboratory-mock",         # space or hyphen
        "laboratory mock-community",
        "synthetic standard",
        "sterile blank",
        "mock community",          # mock community / mock DNA, anywhere in the sub-biome
        "zymo mock community",
        "ferret mock-community",   # real: the mock community sequenced in a ferret study
        "mock dna",
        "mockcommunity",           # ... also written as one word
        "mock",                    # 'mock' on its own
        "negative control",        # anywhere in the sub-biome
        "kit blank",               # blank / kit / reagent / extraction  +  control / blank
        "reagent control",
        "extraction blank",
    ])
    def test_control_sub_biomes_are_flagged(self, sub_biome):
        assert fc.evidence(sub_biome, []) == f"sub-biome: {sub_biome}"

    def test_case_is_ignored_and_the_original_text_is_reported(self):
        assert fc.evidence("Laboratory Control", []) == "sub-biome: Laboratory Control"

    @pytest.mark.parametrize("sub_biome", [
        "forest soil",             # an ordinary habitat
        "control",                 # 'control' alone is never enough ...
        "healthy control",         # ... a healthy subject is a real human sample
        "control soil",            # ... an untreated plot is a real soil sample
        "blank",                   # 'blank' alone as a sub-biome is not in the list (as a keyword it is: rule 2)
        "positive control",        # only 'negative control' is in the sub-biome list
        "mock sample",             # 'mock' + something other than community / DNA
        "laboratory reagent",      # 'laboratory' + a word that is not control / blank / mock / standard
        "lab control",             # 'lab' is not 'laboratory'
        "mockingbird gut",         # 'mock' must be a whole word
        "",                        # no sub-biome at all
    ])
    def test_habitat_sub_biomes_are_not_flagged(self, sub_biome):
        assert fc.evidence(sub_biome, []) == NOT_A_CONTROL

    def test_laboratory_must_be_the_first_word(self):
        assert fc.evidence("laboratory control", []) != NOT_A_CONTROL
        assert fc.evidence("my laboratory control", []) == NOT_A_CONTROL

    def test_negative_control_anywhere_in_the_sub_biome_flags_it_quirk(self):
        # As a KEYWORD, 'negative control' is weak (it can be an untreated group of a real study), but as
        # part of the SUB-BIOME it is enough on its own, wherever it appears.
        assert fc.evidence("negative control soil", []) == "sub-biome: negative control soil"


# ============================================================================= evidence(): rule 2, strong keyword
class TestStrongKeywordRule:
    """One strong keyword is enough, whatever the sub-biome (here an ordinary habitat: 'forest soil')."""

    @pytest.mark.parametrize("keyword", [
        # <technical step> + blank / control / negative
        "extraction blank",
        "DNA extraction blank",
        "pcr negative control",
        "pcr negative",
        "kit control",
        "reagent blanks",          # plural
        "buffer control",
        "air blank",
        "library blank",
        "sequencing controls",
        "isolation negative",
        "pcr negatives",
        "negative extraction control",
        "negative extraction controls",
        # blanks on their own
        "blank",
        "dna blank",
        "DNA blanks",
        "blank swab",
        "blank control",
        "control blank",           # the most frequent strong keyword in the atlas (1,278 samples)
        "extraction blank control",
        # no-template control
        "ntc",
        "NTC",
        "no template control",
        "no-template controls",
        # named technical controls
        "empty control",
        "technical control",
        "processing controls",
        # commercial mock communities / standards
        "zymobiomics",
        "ZymoBIOMICS gut standard",
        "community standard",
        "microbial community standards",
        "whole cell standard",
        "microbial dna standard",
    ])
    def test_strong_keywords_are_flagged(self, keyword):
        assert fc.evidence("forest soil", [keyword]) == f"keyword: {keyword}"

    @pytest.mark.parametrize("keyword", [
        "control",                 # 'control' alone is never enough
        "control group",
        "pcr",                     # a technical step without blank / control / negative
        "extraction kit",
        "air",
        "air sample",
        "blank soil",              # 'blank' + a word that is not sample / control / well / swab
        "blank wells",             # only 'well' singular is listed
        "blanket bog",             # 'blank' must be a whole word here
        "ntc samples",             # 'ntc' must be the whole keyword
        "technical replicate",
        "zymobiomicsx",
        # real atlas keywords of real samples
        "Blank Spring",            # a hot spring in California
        "blank pond",
        "mock-infected",           # sham-infected hosts: real tissue samples
        "mock control",            # ... their sham-treated control group
        "mock treatment",
    ])
    def test_ordinary_keywords_are_not_flagged(self, keyword):
        assert fc.evidence("forest soil", [keyword]) == NOT_A_CONTROL

    def test_any_one_keyword_of_the_set_is_enough(self):
        assert fc.evidence("forest soil", ["soil", "forest", "extraction blank"]) == "keyword: extraction blank"

    def test_pcr_negative_is_flagged_even_if_it_could_be_a_test_result_quirk(self):
        # 'pcr negative' is read as a PCR negative control; in a clinical study it could also mean
        # "this patient tested PCR-negative", i.e. a real human sample.
        assert fc.evidence("human gut", ["pcr negative"]) == "keyword: pcr negative"

    @pytest.mark.parametrize("keyword", ["air blanket", "kit controller", "pcr negativity",
                                         "negative extraction controller"])
    def test_control_word_must_end_at_a_word_boundary(self, keyword):
        # fixed 2026-10-19: these used to match as 'air blank', 'kit control', ...
        assert fc.evidence("forest soil", [keyword]) == NOT_A_CONTROL

    def test_more_words_after_the_control_word_are_allowed(self):
        assert fc.evidence("forest soil", ["pcr control water"]) == "keyword: pcr control water"

    @pytest.mark.parametrize("keyword", ["buffer control group", "buffer control groups", "air control group"])
    def test_a_control_group_is_a_group_of_real_samples(self, keyword):
        # fixed 2026-10-20: 28 horse gut samples ('buffer control group': horses given a buffer instead of a
        # starch overload) were flagged as buffer controls
        assert fc.evidence("horse cecum", [keyword]) == NOT_A_CONTROL

    @pytest.mark.parametrize("keyword", ["sterile water blank", "field blank", "filter blank", "water blank",
                                         "non-template control", "negative pcr control", "pcrblank"])
    def test_blanks_missing_from_the_vocabulary_quirk(self, keyword):
        # Real controls (about 330 atlas samples) that no rule knows: 'sterile water blank' (119 samples)
        # is only a LAB keyword, and 'non-template' is not 'no-template'. Not flagged today.
        assert fc.evidence("sterile water", [keyword]) == NOT_A_CONTROL


# ============================================================================= evidence(): rule 3, weak keyword
class TestWeakKeywordRule:
    """negative / positive control, mock community / sample / DNA: flagged ONLY with a laboratory cue,
    either in the sub-biome or in another keyword."""

    WEAK_KEYWORDS = ["negative control", "positive controls", "mock community", "mock communities",
                     "mock sample", "mock dna"]

    @pytest.mark.parametrize("keyword", WEAK_KEYWORDS)
    def test_weak_keyword_alone_is_not_enough(self, keyword):
        # a study of real human gut samples that also mentions its negative control group / mock community
        assert fc.evidence("human gut", [keyword]) == NOT_A_CONTROL

    @pytest.mark.parametrize("keyword", WEAK_KEYWORDS)
    def test_weak_keyword_with_a_lab_sub_biome_is_flagged(self, keyword):
        assert fc.evidence("laboratory", [keyword]) == f"keyword + lab context: {keyword}"

    @pytest.mark.parametrize("lab_sub_biome", [
        "laboratory",              # starts with laboratory / sterile / synthetic
        "sterile water",
        "synthetic community",
        "water blank",             # contains the word blank or mock
        "mock-up",
        "lab control",             # ends with 'control'
    ])
    def test_lab_sub_biomes(self, lab_sub_biome):
        assert fc.evidence(lab_sub_biome, ["negative control"]) == "keyword + lab context: negative control"

    @pytest.mark.parametrize("lab_keyword", [
        "synthetic metagenome",
        "sterile water",
        "sterile water blank",
        "nuclease-free water",
        "PCR-grade water",
        "pcr grade sterile water",
        "laboratory",
        "lab",
    ])
    def test_lab_keywords(self, lab_keyword):
        # the sub-biome is an ordinary habitat: the laboratory cue comes from another keyword of the set
        assert fc.evidence("human gut", ["negative control", lab_keyword]) == "keyword + lab context: negative control"

    def test_the_lab_keyword_may_come_before_the_weak_keyword(self):
        assert fc.evidence("human gut", ["laboratory", "mock community"]) == "keyword + lab context: mock community"

    @pytest.mark.parametrize("sub_biome, keywords", [
        ("human gut", ["negative control", "lab work"]),          # 'lab' must be the whole keyword
        ("labrador", ["negative control"]),                       # 'laboratory' must be a whole word
        ("control soil", ["negative control"]),                   # 'control' must END the sub-biome
        ("soil controls", ["negative control"]),                  # ... and be singular
        ("laboratory", ["negative control group"]),               # the weak keyword must be the whole keyword
    ])
    def test_near_misses_are_not_flagged(self, sub_biome, keywords):
        assert fc.evidence(sub_biome, keywords) == NOT_A_CONTROL

    def test_control_alone_is_not_even_a_weak_keyword(self):
        # even with the strongest lab cue, 'control' says nothing: 'control group', 'control plot', ...
        assert fc.evidence("laboratory", ["control"]) == NOT_A_CONTROL

    def test_any_sub_biome_ending_in_control_is_a_lab_context_quirk(self):
        # '... control' counts as lab-like, so a healthy-control subject whose keywords call the group a
        # 'negative control' is flagged, although 'healthy control' alone is a real human sample.
        assert fc.evidence("healthy control", []) == NOT_A_CONTROL
        assert fc.evidence("healthy control", ["negative control"]) == "keyword + lab context: negative control"

    def test_lab_is_a_lab_keyword_but_not_a_lab_sub_biome_quirk(self):
        assert fc.evidence("human gut", ["negative control", "lab"]) != NOT_A_CONTROL
        assert fc.evidence("lab", ["negative control"]) == NOT_A_CONTROL


# ============================================================================= evidence(): snake_case texts
class TestUnderscores:
    """GPT sometimes writes keywords in snake_case ('reagent_control', 'PCR_negative_control'). '_' is matched
    as a space, and the evidence quotes the text as written. Fixed 2026-10-20 (31 atlas samples were missed)."""

    @pytest.mark.parametrize("keyword", ["reagent_control", "PCR_negative_control", "kit-blank_control",
                                         "extraction_negative_control", "blank_well"])
    def test_strong_keyword(self, keyword):
        assert fc.evidence("laboratory reagent", [keyword]) == f"keyword: {keyword}"

    def test_weak_keyword_and_lab_context(self):
        assert fc.evidence("laboratory soil", ["soil metagenome", "negative_control"]) == \
            "keyword + lab context: negative_control"
        assert fc.evidence("human gut", ["Negative_Control", "nuclease_free_water"]) == \
            "keyword + lab context: Negative_Control"

    def test_sub_biome(self):
        assert fc.evidence("laboratory_control", []) == "sub-biome: laboratory_control"

    def test_snake_case_real_samples_are_not_flagged(self):
        assert fc.evidence("mouse feces", ["mouse_gi_tract", "antibiotic_study", "control_group"]) == NOT_A_CONTROL


# ============================================================================= evidence(): order of the rules
class TestRulePrecedence:
    def test_sub_biome_rule_wins_over_keywords(self):
        assert fc.evidence("laboratory control", ["extraction blank"]) == "sub-biome: laboratory control"

    def test_strong_keyword_wins_over_weak_keyword_even_when_listed_later(self):
        keywords = ["negative control", "laboratory", "extraction blank"]   # weak + lab cue, then strong
        assert fc.evidence("laboratory", keywords) == "keyword: extraction blank"

    def test_first_matching_keyword_is_reported(self):
        assert fc.evidence("forest soil", ["kit blank", "ntc"]) == "keyword: kit blank"
        assert fc.evidence("forest soil", ["ntc", "kit blank"]) == "keyword: ntc"


# ============================================================================= evidence(): inputs
class TestEvidenceInputs:
    def test_nothing_known_is_not_a_control(self):
        assert fc.evidence("", []) == NOT_A_CONTROL
        assert fc.evidence("", [""]) == NOT_A_CONTROL       # what main() passes for an empty keyword set

    def test_inputs_must_be_stripped_by_the_caller_quirk(self):
        # The patterns are anchored at ^ (and mostly $): surrounding spaces hide a match. main() strips
        # both the sub-biome and every keyword before calling evidence(), so this only matters for
        # other callers.
        assert fc.evidence("  laboratory control", []) == NOT_A_CONTROL
        assert fc.evidence("forest soil", [" ntc "]) == NOT_A_CONTROL
        assert fc.evidence("laboratory", [" negative control"]) == NOT_A_CONTROL


# ============================================================================= split_keywords()
class TestSplitKeywords:
    def test_braces_removed_and_keywords_stripped(self):
        assert fc.split_keywords("{extraction blank, water}") == ["extraction blank", "water"]
        assert fc.split_keywords(" { soil ,  ntc  } ") == ["soil", "ntc"]

    def test_empty_set_gives_one_empty_keyword(self):
        # never matches a rule, so harmless; evidence() is tested with [""] above
        assert fc.split_keywords("{}") == [""]
        assert fc.split_keywords("") == [""]

    def test_trailing_comma_gives_an_empty_keyword(self):
        assert fc.split_keywords("{soil,}") == ["soil", ""]

    def test_no_braces_is_fine(self):
        assert fc.split_keywords("soil, ntc") == ["soil", "ntc"]


# ============================================================================= read_texts()
class TestReadTexts:
    def test_one_entry_per_line(self, tmp_path):
        p = tmp_path / "GPT_sub_biomes.txt"
        p.write_text("s1\tforest soil\ns2\tlaboratory control\n")
        assert fc.read_texts(str(p)) == {"s1": "forest soil", "s2": "laboratory control"}

    def test_text_is_kept_as_is_after_the_first_tab(self, tmp_path):
        # Only the newline is removed: spaces, braces and later tabs stay (main() does the stripping).
        p = tmp_path / "GPT_keywords.txt"
        p.write_text("s1\t{ soil, forest }\ns2\t a\tb \n")
        assert fc.read_texts(str(p)) == {"s1": "{ soil, forest }", "s2": " a\tb "}

    def test_line_without_tab_gives_an_empty_text(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("s1\n")
        assert fc.read_texts(str(p)) == {"s1": ""}

    def test_windows_line_endings_are_removed(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_bytes(b"s1\tforest soil\r\n")
        assert fc.read_texts(str(p)) == {"s1": "forest soil"}

    def test_repeated_sample_id_keeps_the_last_line(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("s1\tforest soil\ns1\tlaboratory control\n")
        assert fc.read_texts(str(p)) == {"s1": "laboratory control"}

    def test_invalid_utf8_is_replaced_not_fatal(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_bytes(b"s1\tsoil \xff\n")
        assert fc.read_texts(str(p)) == {"s1": "soil �"}       # U+FFFD replacement character
