"""Integration tests for 6c_flag_controls.py: main() run end to end on two tiny GPT output files.

How the step is used in the pipeline (README, step 6c):
    python 6c_flag_controls.py --keywords_texts   GPT_keywords.txt      # 'id<TAB>{kw1, kw2, ...}' per line
                               --sub_biomes_texts GPT_sub_biomes.txt    # 'id<TAB>sub-biome'       per line
                               --output           atlas_controls.tsv.gz
and the output is later joined with the outputs of steps 6, 6b and 7 on sample_id.

What main() does, step by step:
    1. read_texts(keywords), read_texts(sub_biomes)  -> {sample_id: text} each (a repeated id: last line wins)
    2. for every sample of either file, once (keywords-file order, then sub-biome-only samples):
         keywords = split_keywords("{a, b}")                  -> ["a", "b"]  ([""] when missing)
         found    = evidence(sub_biome.strip(), keywords)     (sub-biome "" when missing; GPT writes 'NA'
                                                                when it has none)
    3. write a TSV, gzipped when the name ends in .gz: sample_id, control (True/False),
       control_evidence (found, '' if none)
    4. print a one-line summary counting the controls per rule ('sub-biome', 'keyword',
       'keyword + lab context')
Unit tests of evidence() and read_texts() are in test_6c_flag_controls_unit.py.
"""

import ast
import gzip
import sys

import pandas as pd
import pytest

from step_loader import load_step

fc = load_step("6c_flag_controls.py", "flag_controls")


# ----------------------------------------------------------------------------- helpers
def write_lines(p, lines):
    p.write_text("".join(line + "\n" for line in lines))
    return str(p)


def per_rule_counts(printed):
    """The {rule: count} dict at the end of the summary line main() prints."""
    return ast.literal_eval(printed[printed.index("({") + 1:printed.rindex("})") + 1])


@pytest.fixture
def run(tmp_path, monkeypatch):
    """run(keyword_lines, sub_biome_lines) -> list of (sample_id, control, control_evidence) rows written.

    Lines are given as they appear in the GPT files, e.g. "s1\t{soil, forest}" and "s1\tforest soil".
    The script is called exactly as from the command line (sys.argv), then its output is read back."""

    def _run(keyword_lines, sub_biome_lines, output_name="atlas_controls.tsv.gz"):
        keywords = write_lines(tmp_path / "GPT_keywords.txt", keyword_lines)
        sub_biomes = write_lines(tmp_path / "GPT_sub_biomes.txt", sub_biome_lines)
        output = tmp_path / output_name
        monkeypatch.setattr(sys, "argv", ["6c_flag_controls.py", "--keywords_texts", keywords,
                                          "--sub_biomes_texts", sub_biomes, "--output", str(output)])
        fc.main()
        table = pd.read_csv(output, sep="\t", dtype={"sample_id": str}, keep_default_na=False)  # .gz -> gunzipped
        assert list(table.columns) == ["sample_id", "control", "control_evidence"]
        return list(table.itertuples(index=False, name=None))

    return _run


# ============================================================================= the basic workflow
def test_toy_atlas(run, capsys):
    """Eight samples, one per kind of decision. Read across: sub-biome, keywords -> flagged?, because."""
    keyword_lines = [
        "soil1\t{soil, forest, pine}",                         # a real habitat
        "gut1\t{human gut, healthy control}",                  # 'control' alone is never enough
        "lab1\t{water}",                                       # rule 1: the sub-biome says it
        "blank1\t{extraction blank, water}",                   # rule 2: a strong keyword
        "ntc1\t{NTC}",                                         # rule 2: no-template control
        "mock1\t{mock community, synthetic metagenome}",       # rule 3: weak keyword + lab keyword
        "mock2\t{mock community, probiotic}",                  # weak keyword, no lab cue -> real sample
        "neg1\t{negative control}",                            # rule 3: weak keyword + lab sub-biome
    ]
    sub_biome_lines = [
        "soil1\tforest soil",
        "gut1\thealthy control",
        "lab1\tlaboratory blank",
        "blank1\tfreshwater",
        "ntc1\tNA",                                            # GPT's 'no sub-biome' (18,520 atlas samples)
        "mock1\tfecal sample",
        "mock2\tfermented food",
        "neg1\tsterile water",
    ]

    assert run(keyword_lines, sub_biome_lines) == [
        ("soil1",  False, ""),
        ("gut1",   False, ""),
        ("lab1",   True,  "sub-biome: laboratory blank"),
        ("blank1", True,  "keyword: extraction blank"),
        ("ntc1",   True,  "keyword: NTC"),
        ("mock1",  True,  "keyword + lab context: mock community"),
        ("mock2",  False, ""),
        ("neg1",   True,  "keyword + lab context: negative control"),
    ]
    # summary: 8 samples, 5 flagged = 1 by sub-biome + 2 by strong keyword + 2 by weak keyword in lab context
    printed = capsys.readouterr().out
    assert "8 samples, 5 controls / mocks" in printed
    assert per_rule_counts(printed) == {"sub-biome": 1, "keyword": 2, "keyword + lab context": 2}


def test_real_gpt_lines(run):
    """Shortened lines of the real GPT_keywords.txt / GPT_sub_biomes.txt (ids and texts as written)."""
    keyword_lines = [
        "ERS9001098\t{American Gut, control blank, sterile water, citizen science}",
        "SRS3000938\t{reagent_control, extraction_buffer, PCR_reagent_control, human gut metagenome, Brisbane}",
        "ERS10534175\t{synthetic metagenome, mock community, positive control, DNA extraction comparison}",
        "SRS3758574\t{hot spring, sediment, sorted cells, Blank Spring, metagenome}",
        "SRS4129065\t{horse, cecum, gut metagenome, buffer control group, starch overload}",
        "ERS10535561\t{mouse, small intestine, mock-infected, laboratory}",
        "SRS589016\t{groundwater, aquifer water, sandstone aquifer, CO2 injection, metagenome}",
    ]
    sub_biome_lines = [
        "ERS9001098\tsterile water",
        "SRS3000938\tlaboratory reagent",
        "ERS10534175\tlaboratory synthetic",
        "SRS3758574\thot spring sediment",
        "SRS4129065\thorse cecum",
        "ERS10535561\tmouse small intestine",
        "SRS589016\tNA",
    ]
    assert run(keyword_lines, sub_biome_lines) == [
        ("ERS9001098",  True,  "keyword: control blank"),
        ("SRS3000938",  True,  "keyword: reagent_control"),
        ("ERS10534175", True,  "keyword + lab context: mock community"),
        ("SRS3758574",  False, ""),                       # a place called Blank Spring
        ("SRS4129065",  False, ""),                       # horses given a buffer: a control GROUP
        ("ERS10535561", False, ""),                       # sham-infected mice ('laboratory' alone is no control)
        ("SRS589016",   False, ""),
    ]


# ============================================================================= edge cases: which samples are in the output
def test_samples_follow_the_keywords_file_then_the_sub_biome_only_ones(run):
    rows = run(["b\t{soil}", "a\t{soil}"], ["c\tsoil", "a\tsoil", "b\tsoil"])
    assert [sample_id for sample_id, _, _ in rows] == ["b", "a", "c"]


def test_sample_without_sub_biome_is_judged_on_its_keywords_alone(run):
    # 'x' has no sub-biome line: a strong keyword still flags it, a weak keyword cannot (no lab cue)
    rows = run(["x\t{ntc}", "y\t{negative control}"], [])
    assert rows == [("x", True, "keyword: ntc"), ("y", False, "")]


def test_sample_only_in_the_sub_biome_file_is_judged_on_its_sub_biome(run):
    # 'z' has no keywords line. Fixed 2026-10-19: it used to be left out of the output altogether,
    # so a control known only from its sub-biome was never flagged.
    rows = run(["a\t{soil}"], ["a\tforest soil", "z\tlaboratory control"])
    assert rows == [("a", False, ""), ("z", True, "sub-biome: laboratory control")]


def test_sample_id_matching_is_exact(run):
    # 'S1' in the keywords file is not 's1' of the sub-biome file: two different samples
    rows = run(["S1\t{water}"], ["s1\tlaboratory control"])
    assert rows == [("S1", False, ""), ("s1", True, "sub-biome: laboratory control")]


def test_repeated_sample_gives_one_row_from_its_last_line(run):
    # Same rule in both files (as in read_texts). Fixed 2026-10-19: a repeated id in the keywords file
    # used to give one row per line, which duplicates rows when the output is joined on sample_id.
    rows = run(["a\t{ntc}", "a\t{soil}"], ["a\tforest soil"])
    assert rows == [("a", False, "")]


# ============================================================================= edge cases: parsing the GPT texts
def test_keyword_set_is_split_on_commas_and_stripped(run):
    # spaces around keywords and around the braces do not matter; a trailing comma gives an empty keyword
    rows = run(["a\t { soil ,  ntc  , } "], ["a\tforest soil"])
    assert rows == [("a", True, "keyword: ntc")]


def test_sub_biome_is_stripped(run):
    rows = run(["a\t{water}"], ["a\t   laboratory control   "])
    assert rows == [("a", True, "sub-biome: laboratory control")]


def test_keywords_are_not_split_inside_a_sub_biome(run):
    # only the keyword set is split on commas: 'mock community' at the end of the sub-biome still matches
    rows = run(["a\t{water}"], ["a\tsoil, mock community"])
    assert rows == [("a", True, "sub-biome: soil, mock community")]


@pytest.mark.parametrize("keyword_line", ["a\t{}", "a\t", "a"])
def test_empty_or_missing_keyword_set(run, keyword_line):
    # '{}', nothing after the tab, or no tab at all: no keywords; the sub-biome is still used
    assert run([keyword_line], ["a\tforest soil"]) == [("a", False, "")]
    assert run([keyword_line], ["a\tmock community"]) == [("a", True, "sub-biome: mock community")]


# ============================================================================= edge cases: files
def test_empty_input_files_give_a_header_only_output(run, capsys):
    assert run([], []) == []
    printed = capsys.readouterr().out
    assert "0 samples, 0 controls / mocks" in printed
    assert per_rule_counts(printed) == {}


EXPECTED_TEXT = "sample_id\tcontrol\tcontrol_evidence\na\tTrue\tkeyword: ntc\n"


def test_output_named_gz_is_gzipped(run, tmp_path):
    run(["a\t{ntc}"], [], output_name="atlas_controls.tsv.gz")
    with gzip.open(tmp_path / "atlas_controls.tsv.gz", "rt") as handle:
        assert handle.read() == EXPECTED_TEXT


def test_output_named_tsv_is_plain_text(run, tmp_path):
    # fixed 2026-10-19: gzip used to be forced, so 'x.tsv' was a gzip file that read_tsv() could not read
    run(["a\t{ntc}"], [], output_name="atlas_controls.tsv")
    assert (tmp_path / "atlas_controls.tsv").read_text() == EXPECTED_TEXT


def test_paths_with_home_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_lines(tmp_path / "kw.txt", ["a\t{ntc}"])
    write_lines(tmp_path / "sb.txt", ["a\tsoil"])
    monkeypatch.setattr(sys, "argv", ["6c_flag_controls.py", "--keywords_texts", "~/kw.txt",
                                      "--sub_biomes_texts", "~/sb.txt", "--output", "~/out.tsv.gz"])
    fc.main()
    assert (tmp_path / "out.tsv.gz").exists()


def test_missing_input_file_writes_no_output(tmp_path, monkeypatch):
    write_lines(tmp_path / "kw.txt", ["a\t{ntc}"])
    output = tmp_path / "out.tsv.gz"
    monkeypatch.setattr(sys, "argv", ["6c_flag_controls.py", "--keywords_texts", str(tmp_path / "kw.txt"),
                                      "--sub_biomes_texts", str(tmp_path / "missing.txt"), "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        fc.main()
    assert not output.exists()


def test_missing_argument_stops_with_a_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["6c_flag_controls.py", "--keywords_texts", "kw.txt"])
    with pytest.raises(SystemExit):
        fc.main()
