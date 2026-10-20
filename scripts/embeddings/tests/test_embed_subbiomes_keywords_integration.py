"""Integration tests for embed_subbiomes_keywords.py: main() end to end on a 7-sample toy dataset.

How the script is used (and how these tests call it): one command line ->
    1. pick the samples           --full | --n N | --n_per_biome N | --sample_ids_file F
    2. read each target's file    GPT_sub_biomes.txt, GPT_keywords.txt  (id<TAB>text, no header)
    3. de-duplicate the texts     6 samples but only 3 distinct sub-biome texts -> 3 API inputs
    4. estimate cost and disk     then stop (--dry_run), ask (no --yes), or go on (--yes)
    5. stage 1, embed_unique      GPT_{target}_unique_embeddings__{tag}.h5   (texts, embeddings)
    6. stage 2, materialize       GPT_{target}_embeddings__{tag}.h5          (sample_ids, texts, embeddings)
   Re-running the same command resumes from what the two .h5 files already hold.

The toy dataset (the `data` fixture) - read the expected values in the tests against this table:

    sample  GPT_biomes  GPT_sub_biomes       GPT_keywords      keywords after clean_text()
    s1      soil        agricultural soil    {soil, wheat}     soil wheat
    s2      soil        agricultural soil    {soil, maize}     soil maize
    s3      soil        forest soil          {soil, pine}      soil pine
    s4      water       sea water            {seawater}        seawater
    s5      water       sea water            {seawater}        seawater
    s6      water       sea water            {}                (empty: skipped)
    s7      animal      NA  (skipped)        {gut}             gut

The fake API returns these 3-d vectors (the numbers are labels, chosen so each text's vector is easy to
recognise in the outputs; axes ~ soil / water / animal):
"""

import json
import sys
from types import SimpleNamespace

import h5py
import pytest

import embed_subbiomes_keywords as esk
from helpers import FakeOpenAI, read_h5, rows, write_lines

VECTORS = {
    # sub-biome texts
    "agricultural soil": [1, 0, 0],
    "forest soil":       [2, 0, 0],
    "sea water":         [0, 1, 0],
    # keyword texts (already cleaned)
    "soil wheat":        [3, 0, 0],
    "soil maize":        [4, 0, 0],
    "soil pine":         [5, 0, 0],
    "seawater":          [0, 2, 0],
    "gut":               [0, 0, 1],
    # only used by the tests that add samples to the source files
    "lake water":        [0, 3, 0],
}

# What a complete run over every sample must produce (in source-file order).
ALL_SUB_BIOME_ROWS = [
    ("s1", "agricultural soil", [1, 0, 0]),
    ("s2", "agricultural soil", [1, 0, 0]),  # same text as s1 -> same vector, embedded only once
    ("s3", "forest soil",       [2, 0, 0]),
    ("s4", "sea water",         [0, 1, 0]),
    ("s5", "sea water",         [0, 1, 0]),
    ("s6", "sea water",         [0, 1, 0]),
    # s7: NA, no row
]
ALL_KEYWORD_ROWS = [
    ("s1", "soil wheat", [3, 0, 0]),
    ("s2", "soil maize", [4, 0, 0]),
    ("s3", "soil pine",  [5, 0, 0]),
    ("s4", "seawater",   [0, 2, 0]),
    ("s5", "seawater",   [0, 2, 0]),
    # s6: '{}', no row
    ("s7", "gut",        [0, 0, 1]),
]

# Every test runs text-embedding-3-small cut down to 3 dimensions, so file names carry this prefix.
MODEL_DIM = "text-embedding-3-small__dim3"


# ----------------------------------------------------------------------------- fixtures and helpers
@pytest.fixture
def data(tmp_path):
    """The toy dataset above, an output folder and an API key file. Returns the folder paths."""
    input_dir, output_dir = tmp_path / "latest", tmp_path / "out"
    input_dir.mkdir()
    write_lines(input_dir / "GPT_biomes.txt", [
        "s1\tsoil", "s2\tsoil", "s3\tsoil", "s4\twater", "s5\twater", "s6\twater", "s7\tanimal"])
    write_lines(input_dir / "GPT_sub_biomes.txt", [
        "s1\tagricultural soil", "s2\tagricultural soil", "s3\tforest soil",
        "s4\tsea water", "s5\tsea water", "s6\tsea water", "s7\tNA"])
    write_lines(input_dir / "GPT_keywords.txt", [
        "s1\t{soil, wheat}", "s2\t{soil, maize}", "s3\t{soil, pine}",
        "s4\t{seawater}", "s5\t{seawater}", "s6\t{}", "s7\t{gut}"])
    (tmp_path / "api_key").write_text("sk-test\n")
    return {"input": input_dir, "output": output_dir, "api_key": tmp_path / "api_key", "tmp": tmp_path}


@pytest.fixture
def run(data, monkeypatch):
    """run(*flags, client=FakeOpenAI(VECTORS)) runs main() as the command line

        embed_subbiomes_keywords.py --input_dir <toy> --output_dir <out> --api_key_path <key>
                                    --embedding_dim 3 --yes  *flags

    with OpenAI(...) replaced by `client`. Returns the client, whose .calls show what was sent.
    Pass client=None to make constructing a client an error (for runs that must not reach the API).
    """
    def run_main(*flags, client="default", answer=None):
        if client == "default":
            client = FakeOpenAI(VECTORS)

        def make_client(api_key, max_retries):
            assert client is not None, "this run was not supposed to create an API client"
            client.api_key = api_key
            return client

        monkeypatch.setattr(esk, "OpenAI", make_client)
        argv = ["embed_subbiomes_keywords.py",
                "--input_dir", str(data["input"]), "--output_dir", str(data["output"]),
                "--api_key_path", str(data["api_key"]), "--embedding_dim", "3"]
        if answer is None:
            argv.append("--yes")
        else:  # no --yes: main() asks "Proceed? [y/N]" and gets `answer`
            monkeypatch.setattr("builtins.input", lambda prompt: answer)
        monkeypatch.setattr(sys, "argv", argv + list(flags))
        esk.main()
        return client
    return run_main


def out(data, name):
    """Path of an output file, e.g. out(data, f'GPT_sub_biomes_embeddings__{MODEL_DIM}__full.h5')."""
    return data["output"] / name


def unique_file(data, target, subset="full"):
    return out(data, f"GPT_{target}_unique_embeddings__{MODEL_DIM}__{subset}.h5")


def sample_file(data, target, subset="full"):
    return out(data, f"GPT_{target}_embeddings__{MODEL_DIM}__{subset}.h5")


def h5_files(data):
    return sorted(p.name for p in data["output"].glob("*.h5"))


# ============================================================================= the basic workflow
class TestFullRun:
    def test_per_sample_files(self, run, data):
        run("--full")
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS
        assert rows(sample_file(data, "keywords")) == ALL_KEYWORD_ROWS

    def test_unique_files_hold_each_distinct_text_once(self, run, data):
        run("--full")
        assert rows(unique_file(data, "sub_biomes")) == [
            ("agricultural soil", [1, 0, 0]), ("forest soil", [2, 0, 0]), ("sea water", [0, 1, 0])]
        assert rows(unique_file(data, "keywords")) == [
            ("soil wheat", [3, 0, 0]), ("soil maize", [4, 0, 0]), ("soil pine", [5, 0, 0]),
            ("seawater", [0, 2, 0]), ("gut", [0, 0, 1])]

    def test_api_is_called_once_per_distinct_text(self, run):
        client = run("--full")
        # one call per target (batch_size 2048 > number of texts), texts in first-seen order
        assert client.batches == [
            ["agricultural soil", "forest soil", "sea water"],                 # 6 sub-biome samples
            ["soil wheat", "soil maize", "soil pine", "seawater", "gut"],      # 6 keyword samples
        ]
        assert all(call["model"] == "text-embedding-3-small" and call["dimensions"] == 3 for call in client.calls)
        assert client.api_key == "sk-test"  # read from --api_key_path, stripped

    def test_output_files_and_manifest(self, run, data):
        run("--full")
        assert h5_files(data) == sorted([
            f"GPT_keywords_embeddings__{MODEL_DIM}__full.h5",
            f"GPT_keywords_unique_embeddings__{MODEL_DIM}__full.h5",
            f"GPT_sub_biomes_embeddings__{MODEL_DIM}__full.h5",
            f"GPT_sub_biomes_unique_embeddings__{MODEL_DIM}__full.h5",
        ])
        manifest = json.loads(out(data, f"run_manifest__{MODEL_DIM}__full.json").read_text())
        assert manifest["tag"] == f"{MODEL_DIM}__full"
        assert manifest["embedding_dim"] == 3
        assert manifest["subset_size"] is None  # None = every sample
        assert not list(data["output"].glob("subset_ids__*"))  # only written for subsets

    def test_one_target_only(self, run, data):
        client = run("--full", "--targets", "keywords")
        assert client.texts_sent == ["soil wheat", "soil maize", "soil pine", "seawater", "gut"]
        assert h5_files(data) == sorted([f"GPT_keywords_embeddings__{MODEL_DIM}__full.h5",
                                         f"GPT_keywords_unique_embeddings__{MODEL_DIM}__full.h5"])

    def test_small_batches_same_result(self, run, data):
        client = run("--full", "--targets", "sub_biomes", "--batch_size", "2")
        assert client.batches == [["agricultural soil", "forest soil"], ["sea water"]]
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS

    def test_unique_only_skips_per_sample_file(self, run, data):
        run("--full", "--unique_only")
        assert h5_files(data) == sorted([f"GPT_keywords_unique_embeddings__{MODEL_DIM}__full.h5",
                                         f"GPT_sub_biomes_unique_embeddings__{MODEL_DIM}__full.h5"])


# ============================================================================= subsets
class TestSubsets:
    def test_sample_ids_file(self, run, data):
        ids_file = write_lines(data["tmp"] / "pick.txt", ["s1", "s4"])
        client = run("--sample_ids_file", ids_file)
        assert client.texts_sent == ["agricultural soil", "sea water", "soil wheat", "seawater"]
        assert rows(sample_file(data, "sub_biomes", "ids-pick")) == [
            ("s1", "agricultural soil", [1, 0, 0]), ("s4", "sea water", [0, 1, 0])]
        assert rows(sample_file(data, "keywords", "ids-pick")) == [
            ("s1", "soil wheat", [3, 0, 0]), ("s4", "seawater", [0, 2, 0])]

    def test_subset_ids_are_saved_for_reuse(self, run, data):
        ids_file = write_lines(data["tmp"] / "pick.txt", ["s4", "s1"])
        run("--sample_ids_file", ids_file)
        saved = out(data, "subset_ids__ids-pick.txt")
        assert saved.read_text() == "s1\ns4\n"  # sorted, one per line: usable as --sample_ids_file

    def test_ids_without_usable_text_are_left_out(self, run, data):
        # s7: sub-biome NA, keywords 'gut'.  s6: sub-biome 'sea water', keywords '{}'.  s99: not in the data.
        ids_file = write_lines(data["tmp"] / "pick.txt", ["s4", "s6", "s7", "s99"])
        run("--sample_ids_file", ids_file)
        assert [r[0] for r in rows(sample_file(data, "sub_biomes", "ids-pick"))] == ["s4", "s6"]
        assert [r[0] for r in rows(sample_file(data, "keywords", "ids-pick"))] == ["s4", "s7"]

    def test_n_per_biome(self, run, data):
        run("--n_per_biome", "2", "--targets", "sub_biomes")
        chosen = set(out(data, "subset_ids__perbiome2_seed42.txt").read_text().split())
        assert len(chosen & {"s1", "s2", "s3"}) == 2   # soil: 2 of 3
        assert len(chosen & {"s4", "s5", "s6"}) == 2   # water: 2 of 3
        assert "s7" in chosen                          # animal: only 1 sample...
        written = {r[0] for r in rows(sample_file(data, "sub_biomes", "perbiome2_seed42"))}
        assert written == chosen - {"s7"}              # ...whose sub-biome is NA, so it has no row

    def test_different_subsets_never_share_files(self, run, data):
        run("--sample_ids_file", write_lines(data["tmp"] / "a.txt", ["s1"]), "--targets", "sub_biomes")
        run("--sample_ids_file", write_lines(data["tmp"] / "b.txt", ["s3"]), "--targets", "sub_biomes")
        assert rows(sample_file(data, "sub_biomes", "ids-a")) == [("s1", "agricultural soil", [1, 0, 0])]
        assert rows(sample_file(data, "sub_biomes", "ids-b")) == [("s3", "forest soil", [2, 0, 0])]


# ============================================================================= runs that stop before the API
class TestStopsBeforeApi:
    def test_dry_run_estimates_and_writes_no_embeddings(self, run, data, capsys):
        run("--full", "--dry_run", client=None)
        # tiktoken is blocked in tests, so tokens = characters // 4 per target:
        #   sub-biomes: 'agricultural soil' 17 + 'forest soil' 11 + 'sea water' 9 = 37 -> 9
        #   keywords:   10 + 10 + 9 + 8 + 3 = 40 -> 10
        assert "Estimated 19 tokens" in capsys.readouterr().out
        assert h5_files(data) == []
        assert out(data, f"run_manifest__{MODEL_DIM}__full.json").exists()  # the manifest is still written

    def test_answering_no_stops(self, run, data):
        run("--full", client=None, answer="n")
        assert h5_files(data) == []

    def test_answering_y_goes_on(self, run, data):
        run("--full", answer="y")
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS

    def test_not_enough_disk(self, run, data, monkeypatch):
        monkeypatch.setattr(esk.shutil, "disk_usage", lambda d: SimpleNamespace(free=0))  # a full disk
        with pytest.raises(SystemExit, match="Not enough free space"):
            run("--full", client=None)
        assert h5_files(data) == []

    def test_dimension_above_native_size(self, run):
        with pytest.raises(SystemExit, match="> text-embedding-3-small's native 1536"):
            run("--full", "--embedding_dim", "2000", client=None)

    def test_no_subset_chosen(self, run):
        with pytest.raises(SystemExit, match="--full"):
            run(client=None)


# ============================================================================= re-running: the files are the resume state
class TestResume:
    def test_second_identical_run_does_nothing(self, run, data):
        run("--full")
        client = run("--full")
        assert client.calls == []  # nothing paid for twice
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS  # no duplicate rows
        assert rows(sample_file(data, "keywords")) == ALL_KEYWORD_ROWS

    def test_api_failure_then_rerun_finishes_the_job(self, run, data):
        # Run 1: one text per call; the 2nd call fails (as if the connection dropped).
        with pytest.raises(ConnectionError):
            run("--full", "--targets", "sub_biomes", "--batch_size", "1",
                client=FakeOpenAI(VECTORS, fail_on_call=2))
        assert rows(unique_file(data, "sub_biomes")) == [("agricultural soil", [1, 0, 0])]  # batch 1 kept
        assert not sample_file(data, "sub_biomes").exists()  # stage 2 never started

        # Run 2: same command; only the two missing texts are sent.
        client = run("--full", "--targets", "sub_biomes", "--batch_size", "1")
        assert client.batches == [["forest soil"], ["sea water"]]
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS

    def test_new_samples_in_source_are_added(self, run, data):
        run("--full", "--targets", "sub_biomes")
        with open(data["input"] / "GPT_sub_biomes.txt", "a") as f:
            f.write("s8\tforest soil\n")  # known text: no API call needed
            f.write("s9\tlake water\n")   # new text
        client = run("--full", "--targets", "sub_biomes")
        assert client.texts_sent == ["lake water"]
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS + [
            ("s8", "forest soil", [2, 0, 0]),
            ("s9", "lake water", [0, 3, 0]),
        ]

    def test_half_written_row_is_repaired(self, run, data):
        run("--full", "--targets", "sub_biomes")
        # Simulate a crash inside h5_append while writing s6: 'sample_ids' got the row, the other columns not.
        with h5py.File(sample_file(data, "sub_biomes"), "a") as f:
            f["texts"].resize(5, axis=0)
            f["embeddings"].resize(5, axis=0)
        client = run("--full", "--targets", "sub_biomes")
        assert client.calls == []  # the texts are all in the unique file already
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS  # s6 cut, then rewritten

    def test_edited_source_text_is_not_picked_up_quirk(self, run, data):
        # Resume only asks "is this sample id in the file?", not "is its text the same?".
        # After regenerating the GPT outputs, use a new --output_dir (or delete the old files).
        run("--full", "--targets", "sub_biomes")
        source = data["input"] / "GPT_sub_biomes.txt"
        source.write_text(source.read_text().replace("s3\tforest soil", "s3\tsea water"))
        run("--full", "--targets", "sub_biomes")
        assert ("s3", "forest soil", [2, 0, 0]) in rows(sample_file(data, "sub_biomes"))  # still the old text

    def test_crash_during_first_write_is_repaired(self, run, data):
        # The very first h5_append of the per-sample file creates 'sample_ids', then fails while creating
        # 'embeddings' (the big column, so a full disk would fail exactly there). Left behind: a file with
        # only sample_ids s1..s3 (here, the first write was a chunk of 3 samples).
        run("--full", "--targets", "sub_biomes", "--unique_only")  # stage 1 complete
        with h5py.File(sample_file(data, "sub_biomes"), "w") as f:
            f.create_dataset("sample_ids", data=["s1", "s2", "s3"], dtype=h5py.string_dtype("utf-8"),
                             maxshape=(None,))
        # Re-running: the missing 'texts' / 'embeddings' count as 0 rows, so s1..s3 are dropped and rewritten.
        # (Before the fix, s1..s3 counted as done, and two runs later s1 was paired with the 'sea water' vector.)
        run("--full", "--targets", "sub_biomes")
        run("--full", "--targets", "sub_biomes")  # a further run changes nothing
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS

    def test_crash_during_first_write_of_unique_file_is_repaired(self, run, data):
        # Same crash in stage 1: the unique file has 'texts' = ['agricultural soil'] and no 'embeddings'.
        data["output"].mkdir()
        with h5py.File(unique_file(data, "sub_biomes"), "w") as f:
            f.create_dataset("texts", data=["agricultural soil"], dtype=h5py.string_dtype("utf-8"), maxshape=(None,))
        client = run("--full", "--targets", "sub_biomes")
        assert client.texts_sent == ["agricultural soil", "forest soil", "sea water"]  # embedded again, not skipped
        assert rows(sample_file(data, "sub_biomes")) == ALL_SUB_BIOME_ROWS


# ============================================================================= unusual source data
class TestEdgeCases:
    def test_duplicate_sample_id_gives_two_rows_quirk(self, run, data):
        with open(data["input"] / "GPT_sub_biomes.txt", "a") as f:
            f.write("s1\tforest soil\n")  # s1 a second time, with another text
        run("--full", "--targets", "sub_biomes")
        s1_rows = [r for r in rows(sample_file(data, "sub_biomes")) if r[0] == "s1"]
        assert s1_rows == [("s1", "agricultural soil", [1, 0, 0]), ("s1", "forest soil", [2, 0, 0])]

    def test_keyword_sets_with_same_words_share_a_vector_quirk(self, run, data):
        # '{forest soil, leaf}' and '{forest, soil leaf}' both clean to 'forest soil leaf' (see clean_text)
        write_lines(data["input"] / "GPT_keywords.txt", ["k1\t{forest soil, leaf}", "k2\t{forest, soil leaf}"])
        client = run("--full", "--targets", "keywords", client=FakeOpenAI({"forest soil leaf": [9, 9, 9]}))
        assert client.texts_sent == ["forest soil leaf"]
        assert rows(sample_file(data, "keywords")) == [("k1", "forest soil leaf", [9, 9, 9]),
                                                       ("k2", "forest soil leaf", [9, 9, 9])]

    def test_unicode_texts(self, run, data):
        write_lines(data["input"] / "GPT_sub_biomes.txt", ["u1\tsédiment de rivière", "u2\t土壌"])
        run("--full", "--targets", "sub_biomes",
            client=FakeOpenAI({"sédiment de rivière": [1, 1, 0], "土壌": [1, 0, 1]}))
        assert rows(sample_file(data, "sub_biomes")) == [("u1", "sédiment de rivière", [1, 1, 0]),
                                                         ("u2", "土壌", [1, 0, 1])]

    def test_target_without_any_usable_text_in_subset(self, run, data):
        # s6 has a sub-biome ('sea water') but its keywords are '{}'. The keywords target has nothing to
        # embed: it is skipped (no API call, no files) and the sub-biome target is done as usual.
        # (Before the fix, main() crashed opening a unique file that was never written.)
        ids_file = write_lines(data["tmp"] / "pick.txt", ["s6"])
        client = run("--sample_ids_file", ids_file)
        assert client.texts_sent == ["sea water"]
        assert rows(sample_file(data, "sub_biomes", "ids-pick")) == [("s6", "sea water", [0, 1, 0])]
        assert not unique_file(data, "keywords", "ids-pick").exists()
        assert not sample_file(data, "keywords", "ids-pick").exists()
