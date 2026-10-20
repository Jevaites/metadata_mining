"""
Tests for 3_extract_sample_embeddings.py (step 3 of the ontology-mapping pipeline).

What step 3 does, on the toy data used by most tests below
-----------------------------------------------------------

    --texts  (sample -> text)        --unique_h5  (text -> vector)
        S1   soil                        row 0   "soil"       [ 1, 0]
        S2   seawater                    row 1   "seawater"   [ 0, 1]
        S3   soil                        row 2   "human gut"  [-1, 0]
        S4   human gut

    --sample_ids  (which samples we want: first column)
        S1  S2  S3                       (S4 is not requested)

                              |  step 3
                              v

    --output .npz
        sample_ids = [S1, S2, S3]        sorted
        vectors    = [[1, 0], [0, 1]]    every h5 row that is needed, once, in h5 row order
        index      = [0, 1, 0]           the vector of sample_ids[i] is vectors[index[i]]

    S1 and S3 share the text "soil", so its vector is stored once and both point at it.
    "human gut" is never needed, so it is not copied out of the h5 file.

How a sample gets matched: its text is cleaned with the same clean_text() that was used
when the texts were embedded (embed_subbiomes_keywords.py), then looked up by EXACT string
equality in the h5 "texts" column. A sample is silently dropped when
  (a) it has no line in --texts,  (b) its text is empty / NA,  (c) its cleaned text is not in the h5 file;
test_printed_summary_accounts_for_every_dropped_sample has one sample of each. If that leaves no sample
at all, step 3 stops with a message saying which input to check (last two tests).
clean_text() / iter_samples() themselves are tested in scripts/embeddings/tests; here only their use.

Conventions (shared with the other files in this folder): tests named `..._quirk` pin down current
behaviour that is surprising but harmless. test_3_extract_sample_embeddings_mutations.py breaks the
script on purpose, one plausible bug at a time, and checks that the tests below notice.

Layout of this file
  1. helpers that build tiny input files the way the real pipeline writes them
  2. unit tests: read_first_column()
  3. integration tests: run main() end to end, exactly like the command line
  4. edge cases

Run (from the repo root or from scripts/ontology_mapping):
    python -m pytest scripts/ontology_mapping/tests -v
Needs numpy, h5py, pytest (+ scikit-learn and pandas for the hand-off test to common.load_npz).
"""

import gzip
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ONTOLOGY_MAPPING = Path(__file__).resolve().parent.parent
STEP3_FILE = ONTOLOGY_MAPPING / "3_extract_sample_embeddings.py"

# Step 3 imports iter_samples() from scripts/embeddings/embed_subbiomes_keywords.py, and that module
# does `from openai import OpenAI` at import time. Step 3 never calls the API, so when openai is not
# installed a dummy stands in for it. (If it is installed, the real one is used.)
if importlib.util.find_spec("openai") is None:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=None)

# The file name starts with a digit, so `import 3_extract_sample_embeddings` is a syntax error:
# load it from its path instead.
_spec = importlib.util.spec_from_file_location("step3", STEP3_FILE)
step3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step3)

# Loading step3 put scripts/embeddings on sys.path, so the producer of the h5 files is importable.
import embed_subbiomes_keywords as embed  # noqa: E402


# =============================================================================================
# 1. helpers
# =============================================================================================

# 2-d vectors that are easy to tell apart by eye.
SOIL = [1.0, 0.0]
SEAWATER = [0.0, 1.0]
GUT = [-1.0, 0.0]


def write_lines(path, lines):
    """One string per line, newline-terminated, like every text file in the pipeline."""
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_unique_h5(path, text_to_vector):
    """Write a 'GPT_*_unique_embeddings__*.h5' file with the SAME function the embedding step uses
    (embed.h5_append): dataset 'texts' (utf-8 strings) and dataset 'embeddings' (float32), one row
    per distinct text, rows in the dict's order."""
    embed.h5_append(str(path), texts=list(text_to_vector),
                    embeddings=np.array(list(text_to_vector.values()), dtype=np.float32))
    return path


def run_step3(tmp_path, monkeypatch, *, kind, texts, unique_h5, sample_ids):
    """Run step3.main() exactly as the command line would, and return the .npz as a plain dict."""
    output = tmp_path / "out.npz"
    monkeypatch.setattr(sys, "argv", [
        "3_extract_sample_embeddings.py", "--kind", kind, "--texts", str(texts),
        "--unique_h5", str(unique_h5), "--sample_ids", str(sample_ids), "--output", str(output)])
    step3.main()
    with np.load(output) as saved:
        return {name: saved[name] for name in saved.files}


def vector_of(result):
    """Undo the index indirection: {sample_id: its vector as a plain list}."""
    return {s: result["vectors"][i].tolist() for s, i in zip(result["sample_ids"], result["index"])}


@pytest.fixture
def toy(tmp_path):
    """The toy data drawn in the module docstring, as the three input files step 3 reads."""
    return dict(
        texts=write_lines(tmp_path / "GPT_sub_biomes.txt", [
            "S1\tsoil",
            "S2\tseawater",
            "S3\tsoil",
            "S4\thuman gut",
        ]),
        unique_h5=write_unique_h5(tmp_path / "GPT_sub_biomes_unique_embeddings.h5", {
            "soil": SOIL,          # row 0
            "seawater": SEAWATER,  # row 1
            "human gut": GUT,      # row 2
        }),
        # Same shape as metalog_training_set.tsv.gz: header line, sample_id first, other columns ignored.
        sample_ids=write_lines(tmp_path / "training_set.tsv", [
            "sample_id\tbiome",
            "S1\tENVO_00001998",
            "S2\tENVO_00002149",
            "S3\tENVO_00001998",
        ]),
    )


# =============================================================================================
# 2. unit tests
# =============================================================================================

# ---- read_first_column(): which samples are requested -----------------------------------------

def test_read_first_column_keeps_only_first_tsv_column_and_skips_header(tmp_path):
    ids = write_lines(tmp_path / "ids.tsv", ["sample_id\tlabel", "S1\tsoil", "S2\tseawater"])
    assert step3.read_first_column(str(ids)) == {"S1", "S2"}


def test_read_first_column_reads_gzip_when_name_ends_in_gz(tmp_path):
    ids = tmp_path / "ids.tsv.gz"
    with gzip.open(ids, "wt") as handle:
        handle.write("sample_id\tlabel\nS1\tsoil\n")
    assert step3.read_first_column(str(ids)) == {"S1"}


def test_read_first_column_ignores_blank_lines_duplicates_spaces_and_crlf(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_bytes(b"S1\r\n\r\n  S2  \nS1\n\n")
    assert step3.read_first_column(str(ids)) == {"S1", "S2"}


# =============================================================================================
# 3. integration tests: main() end to end
# =============================================================================================

def test_basic_workflow(tmp_path, monkeypatch, toy):
    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", **toy)

    # The easy-to-read view: every requested sample got the vector of its text.
    assert vector_of(result) == {"S1": SOIL, "S2": SEAWATER, "S3": SOIL}

    # The stored layout (see the module docstring):
    assert result["sample_ids"].tolist() == ["S1", "S2", "S3"]  # S4 has a text but was not requested
    assert result["vectors"].tolist() == [SOIL, SEAWATER]       # "soil" stored once, "human gut" not at all
    assert result["index"].tolist() == [0, 1, 0]                # S1 and S3 both point at the "soil" row
    assert result["vectors"].dtype == np.float32                # copied from the h5 file unchanged
    assert result["index"].dtype.kind == "i"                    # integer positions into `vectors`


def test_basic_workflow_keywords(tmp_path, monkeypatch):
    # GPT_keywords.txt stores '{kw, kw}' (285 real lines are a bare 'kw,kw' list); the embedding step
    # embedded the cleaned 'kw kw', case kept. --kind keywords applies the same cleaning, so they match.
    texts = write_lines(tmp_path / "GPT_keywords.txt", [            # real lines, shortened
        "ERS218570\t{Short-tailed shearwater, chick, gut, faeces}",
        "ERS635136\twastewater treatment plant,activated sludge,Denmark",
    ])
    unique_h5 = write_unique_h5(tmp_path / "unique.h5", {
        "Short-tailed shearwater chick gut faeces": GUT,
        "wastewater treatment plant activated sludge Denmark": SEAWATER,
    })
    ids = write_lines(tmp_path / "ids.txt", ["ERS218570", "ERS635136"])

    result = run_step3(tmp_path, monkeypatch, kind="keywords", texts=texts, unique_h5=unique_h5, sample_ids=ids)

    assert vector_of(result) == {"ERS218570": GUT, "ERS635136": SEAWATER}


def test_output_is_what_common_load_npz_expects(tmp_path, monkeypatch, toy):
    """Hand-off to the next steps: 5_evaluate.py etc. read the .npz through common.load_npz(),
    which expands `index` back to one (L2-normalised) row per sample."""
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ONTOLOGY_MAPPING))
    from common import load_npz

    run_step3(tmp_path, monkeypatch, kind="sub_biomes", **toy)
    row_of, matrix = load_npz(str(tmp_path / "out.npz"))

    assert row_of == {"S1": 0, "S2": 1, "S3": 2}
    assert matrix.tolist() == [SOIL, SEAWATER, SOIL]  # the toy vectors already have length 1


def test_printed_summary_accounts_for_every_dropped_sample(tmp_path, monkeypatch, capsys):
    """The printed counts are the only diagnostics a real run gives. Here one sample is lost
    at each stage, so every number is easy to check by hand."""
    texts = write_lines(tmp_path / "texts.txt", [
        "S1\tsoil",
        "S2\tsoil",          # same text as S1
        "S3\tsediment",      # (c) has a text, but "sediment" was never embedded
        "S4\tNA",            # (b) NA text
    ])                       # (a) S5 has no line at all
    unique_h5 = write_unique_h5(tmp_path / "unique.h5", {"soil": SOIL})
    ids = write_lines(tmp_path / "ids.txt", ["S1", "S2", "S3", "S4", "S5"])

    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", texts=texts, unique_h5=unique_h5, sample_ids=ids)
    printed = capsys.readouterr().out

    assert "5 samples requested, 3 have a sub_biomes text" in printed       # S4 (NA) and S5 (no line) gone
    assert "2 samples found in unique.h5 (1 texts missing), 1 distinct vectors" in printed  # S3 gone
    assert vector_of(result) == {"S1": SOIL, "S2": SOIL}


# =============================================================================================
# 4. edge cases
# =============================================================================================

def test_lookup_is_exact_after_cleaning(tmp_path, monkeypatch, toy):
    write_lines(toy["texts"], [
        "S1\t  soil  ",   # surrounding spaces are stripped by clean_text -> "soil" -> match
        "S2\tSoil",       # case differs     -> no match
        "S3\tsoil.",      # punctuation      -> no match
    ])
    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", **toy)
    assert vector_of(result) == {"S1": SOIL}


def test_output_order_sample_ids_sorted_vectors_in_h5_row_order(tmp_path, monkeypatch):
    # h5 rows deliberately in a different order from the samples
    unique_h5 = write_unique_h5(tmp_path / "unique.h5", {
        "seawater": SEAWATER,  # row 0
        "human gut": GUT,      # row 1
        "soil": SOIL,          # row 2
    })
    texts = write_lines(tmp_path / "texts.txt", ["S2\tseawater", "S10\tsoil"])
    ids = write_lines(tmp_path / "ids.txt", ["S10", "S2"])

    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", texts=texts, unique_h5=unique_h5, sample_ids=ids)

    assert result["sample_ids"].tolist() == ["S10", "S2"]  # string sort: "S10" < "S2"
    assert result["vectors"].tolist() == [SEAWATER, SOIL]  # h5 rows 0 and 2, increasing
    assert result["index"].tolist() == [1, 0]              # S10 -> soil, S2 -> seawater
    assert vector_of(result) == {"S10": SOIL, "S2": SEAWATER}


def test_non_ascii_texts_match(tmp_path, monkeypatch):
    # Real sub-biomes (5 in GPT_sub_biomes.txt). h5py returns the utf-8 'texts' column as bytes:
    # step 3 must decode them as utf-8 for these to match.
    texts = write_lines(tmp_path / "GPT_sub_biomes.txt", ["SRS4848111\tvicuña feces", "SRS7682291\tBrandt’s vole gut"])
    unique_h5 = write_unique_h5(tmp_path / "unique.h5", {"vicuña feces": GUT, "Brandt’s vole gut": SOIL})
    ids = write_lines(tmp_path / "ids.txt", ["SRS4848111", "SRS7682291"])
    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", texts=texts, unique_h5=unique_h5, sample_ids=ids)
    assert vector_of(result) == {"SRS4848111": GUT, "SRS7682291": SOIL}


def test_sample_listed_twice_in_texts_file_keeps_last_line_quirk(tmp_path, monkeypatch, toy):
    # Quirk: dict(iter_samples(...)) silently keeps the last line; no warning about the duplicate.
    # (The real GPT_*.txt files have no duplicate sample ids.)
    write_lines(toy["texts"], ["S1\tsoil", "S1\tseawater"])
    write_lines(toy["sample_ids"], ["S1"])
    result = run_step3(tmp_path, monkeypatch, kind="sub_biomes", **toy)
    assert vector_of(result) == {"S1": SEAWATER}


def test_tilde_in_paths_is_expanded(tmp_path, monkeypatch, toy):
    monkeypatch.setenv("HOME", str(tmp_path))  # "~/x" now means tmp_path/x
    monkeypatch.setattr(sys, "argv", [
        "3_extract_sample_embeddings.py", "--kind", "sub_biomes",
        "--texts", "~/GPT_sub_biomes.txt", "--unique_h5", "~/GPT_sub_biomes_unique_embeddings.h5",
        "--sample_ids", "~/training_set.tsv", "--output", "~/out_from_tilde.npz"])
    step3.main()
    assert (tmp_path / "out_from_tilde.npz").exists()


def test_no_text_found_in_embedding_file_stops_with_a_clear_message(tmp_path, monkeypatch):
    """Fixed 2026-10-19 (was a cryptic h5py TypeError: an empty *float* `rows` array cannot index h5py).

    Most likely cause in practice: the wrong --kind. Here a keywords file is read with
    --kind sub_biomes, so the braces are kept ('{soil}') and nothing matches the h5 text 'soil'."""
    texts = write_lines(tmp_path / "GPT_keywords.txt", ["S1\t{soil}"])
    unique_h5 = write_unique_h5(tmp_path / "unique.h5", {"soil": SOIL})
    ids = write_lines(tmp_path / "ids.txt", ["S1"])

    with pytest.raises(SystemExit, match="check that --kind matches --texts"):
        run_step3(tmp_path, monkeypatch, kind="sub_biomes", texts=texts, unique_h5=unique_h5, sample_ids=ids)
    assert not (tmp_path / "out.npz").exists()  # no half-empty output left behind


def test_no_requested_sample_has_a_text_stops_with_a_clear_message(tmp_path, monkeypatch, toy):
    """Fixed 2026-10-19 (same crash as above). Typical cause: --sample_ids is about other samples."""
    write_lines(toy["sample_ids"], ["sample_id", "X1", "X2"])  # none of these is in --texts

    with pytest.raises(SystemExit, match="are --sample_ids and --texts about the same samples"):
        run_step3(tmp_path, monkeypatch, kind="sub_biomes", **toy)
    assert not (tmp_path / "out.npz").exists()
