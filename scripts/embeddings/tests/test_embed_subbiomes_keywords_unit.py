"""Unit tests for embed_subbiomes_keywords.py: one class per function, each test checks one behaviour.

Functions under test, in the order main() uses them:
    clean_text(text, is_keywords)        sub-biome: strip.  keywords: '{a, b}' -> 'a b'
    iter_samples(path, keep_ids, is_kw)  stream (sample_id, cleaned text) from a 'id<TAB>text' file
    pick_subset(args)                    which sample ids to embed + a short tag for the file names
    estimate_tokens(texts, model)        token count for the cost estimate
    check_disk_space(out_dir, needed)    stop early if the outputs will not fit
    h5_append / h5_strings / h5_align    the append-only .h5 files that double as resume state
    embed_unique(...)                    stage 1: each distinct text -> one API vector
    materialize(...)                     stage 2: copy each text's vector to every sample with that text
main() and the command line are in test_embed_subbiomes_keywords_integration.py.

Conventions
  * Inputs are written inline, small enough that the expected output can be checked by eye.
    Vectors are 2-d (or 3-d) and their numbers are just labels: [1, 0] is "the soil vector", etc.
  * Files go to pytest's `tmp_path` (a fresh temporary directory per test).
  * Tests named `..._quirk` pin down current behaviour that is surprising but harmless today.
  * Lines copied from the real GPT_*.txt files are marked "real"; the rest are made up.
"""

import sys
from collections import namedtuple
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import embed_subbiomes_keywords as esk
from helpers import FakeOpenAI, read_h5, rows, write_lines


@pytest.fixture
def cli(monkeypatch):
    """cli('--n', '3', ...) -> the argparse Namespace main() would get for that command line."""
    def parse(*argv):
        monkeypatch.setattr(sys, "argv", ["embed_subbiomes_keywords.py", *argv])
        return esk.parse_args()
    return parse


# ============================================================================= reading the source files
class TestCleanText:
    # clean_text() is also imported by the ontology-mapping steps so that their texts match the
    # embedded ones byte for byte: these outputs are a contract, not just an implementation detail.

    def test_sub_biome_is_only_stripped(self):
        assert esk.clean_text("  agricultural soil \n", is_keywords=False) == "agricultural soil"

    def test_sub_biome_keeps_braces_and_commas(self):
        assert esk.clean_text("{soil, sand}", is_keywords=False) == "{soil, sand}"

    def test_keywords_braces_and_commas_removed(self):
        assert esk.clean_text("{soil, wheat, field}", is_keywords=True) == "soil wheat field"

    def test_keywords_whitespace_collapsed(self):
        assert esk.clean_text("  {soil ,  wheat,field }  ", is_keywords=True) == "soil wheat field"

    def test_keywords_without_braces(self):
        # real (shortened): 285 lines of GPT_keywords.txt are a bare comma list, often without spaces
        assert esk.clean_text("wastewater treatment plant,activated sludge,Denmark", is_keywords=True) == \
            "wastewater treatment plant activated sludge Denmark"

    def test_keywords_keep_case_and_hyphens(self):
        # real (shortened): the embedded texts keep the case; lower-casing here would match none of them
        assert esk.clean_text("{Short-tailed shearwater, chick, gut}", is_keywords=True) == \
            "Short-tailed shearwater chick gut"

    def test_keywords_unbalanced_brace_is_kept_quirk(self):
        # braces are only removed as a {...} pair
        assert esk.clean_text("{soil, wheat", is_keywords=True) == "{soil wheat"

    def test_keywords_empty_set_becomes_empty_string(self):
        assert esk.clean_text("{}", is_keywords=True) == ""

    def test_keyword_boundaries_are_lost_quirk(self):
        # Two different keyword sets give the same text, hence the same embedding.
        two_keywords = esk.clean_text("{forest soil, leaf}", is_keywords=True)
        also_two_keywords = esk.clean_text("{forest, soil leaf}", is_keywords=True)
        assert two_keywords == also_two_keywords == "forest soil leaf"


class TestIterSamples:
    def test_basic(self, tmp_path):
        p = write_lines(tmp_path / "sub.txt", ["s1\tagricultural soil", "s2\tsea water"])
        assert list(esk.iter_samples(p, None, False)) == [("s1", "agricultural soil"), ("s2", "sea water")]

    def test_keywords_are_cleaned(self, tmp_path):
        p = write_lines(tmp_path / "kw.txt", ["s1\t{soil, wheat}"])
        assert list(esk.iter_samples(p, None, True)) == [("s1", "soil wheat")]

    def test_keep_ids_filters(self, tmp_path):
        p = write_lines(tmp_path / "sub.txt", ["s1\tsoil", "s2\twater", "s3\tgut"])
        assert list(esk.iter_samples(p, {"s1", "s3"}, False)) == [("s1", "soil"), ("s3", "gut")]

    def test_na_values_skipped_case_insensitive(self, tmp_path):
        # real: 18,520 sub-biomes are 'NA' (the only NA-like value in the real files)
        p = write_lines(tmp_path / "sub.txt", [
            "s1\tsoil",
            "s2\tNA", "s3\tnan", "s4\tNone", "s5\tN/A", "s6\t   ",  # all skipped
        ])
        assert list(esk.iter_samples(p, None, False)) == [("s1", "soil")]

    def test_keywords_na_after_cleaning_skipped(self, tmp_path):
        p = write_lines(tmp_path / "kw.txt", ["s1\t{NA}", "s2\t{}", "s3\t{soil}"])
        assert list(esk.iter_samples(p, None, True)) == [("s3", "soil")]

    def test_malformed_lines_skipped(self, tmp_path):
        p = write_lines(tmp_path / "sub.txt", [
            "s1\tsoil",
            "no_tab_on_this_line",  # no text
            "\torphan text",        # no sample id
            "s2\t",                 # empty text
            "",                     # blank line
        ])
        assert list(esk.iter_samples(p, None, False)) == [("s1", "soil")]

    def test_only_first_tab_splits(self, tmp_path):
        p = write_lines(tmp_path / "sub.txt", ["s1\tsoil\tsand"])
        assert list(esk.iter_samples(p, None, False)) == [("s1", "soil\tsand")]

    def test_windows_line_endings(self, tmp_path):
        (tmp_path / "sub.txt").write_bytes(b"s1\tsoil\r\ns2\twater\r\n")
        assert list(esk.iter_samples(str(tmp_path / "sub.txt"), None, False)) == [("s1", "soil"), ("s2", "water")]

    def test_duplicate_sample_ids_are_all_yielded_quirk(self, tmp_path):
        # no de-duplication of ids: see the integration test for what that does to the outputs
        p = write_lines(tmp_path / "sub.txt", ["s1\tsoil", "s1\twater"])
        assert list(esk.iter_samples(p, None, False)) == [("s1", "soil"), ("s1", "water")]


# ============================================================================= which samples
class TestPickSubset:
    @pytest.fixture
    def input_dir(self, tmp_path):
        """7 samples. Sub-biomes: s7 is NA, so only 6 are usable. Biomes: 3 soil, 3 water, 1 animal."""
        write_lines(tmp_path / "GPT_sub_biomes.txt", [
            "s1\tagricultural soil", "s2\tagricultural soil", "s3\tforest soil",
            "s4\tsea water", "s5\tsea water", "s6\tsea water", "s7\tNA"])
        write_lines(tmp_path / "GPT_biomes.txt", [
            "s1\tsoil", "s2\tsoil", "s3\tsoil", "s4\twater", "s5\twater", "s6\twater", "s7\tanimal"])
        return str(tmp_path)

    def test_full(self, cli, input_dir):
        assert esk.pick_subset(cli("--input_dir", input_dir, "--full")) == (None, "full")  # None = every sample

    def test_no_subset_flag_and_no_full_stops(self, cli, input_dir):
        with pytest.raises(SystemExit, match="--full"):
            esk.pick_subset(cli("--input_dir", input_dir))

    def test_sample_ids_file(self, cli, input_dir, tmp_path):
        ids_file = write_lines(tmp_path / "my_subset.txt", ["s1", "  s4  ", "", "s9"])  # stripped; blank skipped
        ids, tag = esk.pick_subset(cli("--input_dir", input_dir, "--sample_ids_file", ids_file))
        assert ids == {"s1", "s4", "s9"}  # s9 is not in the data: not checked here (main() reports it)
        assert tag == "ids-my_subset"

    def test_n_random_samples(self, cli, input_dir):
        ids, tag = esk.pick_subset(cli("--input_dir", input_dir, "--n", "3"))
        assert len(ids) == 3
        assert ids <= {"s1", "s2", "s3", "s4", "s5", "s6"}  # drawn from usable sub-biome samples (not s7)
        assert tag == "n3_seed42"

    def test_n_same_seed_same_samples(self, cli, input_dir):
        first, _ = esk.pick_subset(cli("--input_dir", input_dir, "--n", "3", "--seed", "7"))
        again, _ = esk.pick_subset(cli("--input_dir", input_dir, "--n", "3", "--seed", "7"))
        assert first == again

    def test_n_larger_than_data_takes_everything_and_tag_says_so(self, cli, input_dir):
        ids, tag = esk.pick_subset(cli("--input_dir", input_dir, "--n", "100"))
        assert ids == {"s1", "s2", "s3", "s4", "s5", "s6"}
        assert tag == "n6_seed42"  # the actual size, not 100

    def test_n_per_biome(self, cli, input_dir):
        ids, tag = esk.pick_subset(cli("--input_dir", input_dir, "--n_per_biome", "2"))
        assert len(ids & {"s1", "s2", "s3"}) == 2  # soil: 2 of 3
        assert len(ids & {"s4", "s5", "s6"}) == 2  # water: 2 of 3
        assert "s7" in ids                         # animal: its only sample (pool smaller than 2)
        assert tag == "perbiome2_seed42"

    def test_n_per_biome_groups_are_case_sensitive_quirk(self, cli, tmp_path):
        write_lines(tmp_path / "GPT_biomes.txt", ["s1\tsoil", "s2\tSoil"])
        ids, _ = esk.pick_subset(cli("--input_dir", str(tmp_path), "--n_per_biome", "1"))
        assert ids == {"s1", "s2"}  # 'soil' and 'Soil' are two biomes, 1 sample each

    def test_subset_flag_wins_over_full_quirk(self, cli, input_dir):
        # --n with --full silently means --n (order: sample_ids_file > n_per_biome > n > full)
        ids, tag = esk.pick_subset(cli("--input_dir", input_dir, "--n", "2", "--full"))
        assert len(ids) == 2 and tag == "n2_seed42"


# ============================================================================= cost and disk
class TestEstimateTokens:
    def test_without_tiktoken_four_chars_per_token(self):
        # tiktoken is blocked in every test (conftest.py); 4 + 8 = 12 characters -> 3 tokens
        assert esk.estimate_tokens(["abcd", "abcdefgh"], "text-embedding-3-small") == (3, "~4 chars/token")

    def test_with_tiktoken(self, monkeypatch):
        # a stand-in tokenizer where every word is one token
        word_encoder = SimpleNamespace(encode=str.split)
        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(encoding_for_model=lambda model: word_encoder))
        assert esk.estimate_tokens(["sea water", "agricultural soil field"], "text-embedding-3-small") == (5, "tiktoken")

    def test_tiktoken_error_falls_back(self, monkeypatch):
        def unknown_model(model):
            raise KeyError(model)
        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(encoding_for_model=unknown_model))
        assert esk.estimate_tokens(["abcdefgh"], "some-model") == (2, "~4 chars/token")


class TestCheckDiskSpace:
    Usage = namedtuple("Usage", "total used free")

    def test_needs_5_percent_margin(self, monkeypatch, tmp_path):
        # 1000 bytes needed -> at least 1050 free
        monkeypatch.setattr(esk.shutil, "disk_usage", lambda d: self.Usage(0, 0, 1050))
        esk.check_disk_space(str(tmp_path), 1000)  # no error
        monkeypatch.setattr(esk.shutil, "disk_usage", lambda d: self.Usage(0, 0, 1049))
        with pytest.raises(SystemExit, match="Not enough free space"):
            esk.check_disk_space(str(tmp_path), 1000)


# ============================================================================= .h5 files
class TestH5Append:
    def test_first_call_creates_columns(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil", "water"], embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32))
        assert read_h5(p) == {"texts": ["soil", "water"], "embeddings": [[1, 0], [0, 1]]}

    def test_next_calls_append_rows(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil"], embeddings=np.array([[1, 0]], dtype=np.float32))
        esk.h5_append(p, texts=["water", "gut"], embeddings=np.array([[0, 1], [1, 1]], dtype=np.float32))
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1]), ("gut", [1, 1])]

    def test_column_types(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil"], embeddings=np.array([[1, 0, 0]], dtype=np.float32))
        with h5py.File(p, "r") as f:
            assert f["embeddings"].dtype == np.float32           # kept from the array
            assert f["embeddings"].maxshape == (None, 3)         # rows can grow, width is fixed
            assert h5py.check_string_dtype(f["texts"].dtype).encoding == "utf-8"

    def test_unicode_round_trip(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["sédiment de rivière", "土壌"])
        assert esk.h5_strings(p, "texts") == ["sédiment de rivière", "土壌"]

    def test_zero_rows_is_not_supported_quirk(self, tmp_path):
        # never happens in the script (it only appends non-empty batches), but beware when reusing h5_append
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil"])
        with pytest.raises(TypeError):
            esk.h5_append(p, texts=[])


class TestH5Strings:
    def test_missing_file_is_empty(self, tmp_path):
        assert esk.h5_strings(str(tmp_path / "does_not_exist.h5"), "texts") == []

    def test_reads_column(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, sample_ids=["s1", "s2"], texts=["soil", "water"])
        assert esk.h5_strings(p, "sample_ids") == ["s1", "s2"]

    def test_missing_column_is_empty(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, sample_ids=["s1"])
        assert esk.h5_strings(p, "texts") == []


class TestH5Align:
    # h5_align(path, columns): `columns` are the columns the file should have, e.g. ("texts", "embeddings")
    COLUMNS = ("texts", "embeddings")

    def test_missing_file_is_fine(self, tmp_path):
        esk.h5_align(str(tmp_path / "does_not_exist.h5"), self.COLUMNS)  # no error, nothing created
        assert not (tmp_path / "does_not_exist.h5").exists()

    def test_aligned_file_untouched(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil", "water"], embeddings=np.array([[1, 0], [0, 1]], dtype=np.float32))
        esk.h5_align(p, self.COLUMNS)
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1])]

    def test_ragged_columns_cut_to_shortest(self, tmp_path):
        # What a crash inside h5_append leaves: 'texts' got its 3rd row, 'embeddings' did not.
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil", "water", "gut"], embeddings=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
        with h5py.File(p, "a") as f:
            f["embeddings"].resize(2, axis=0)
        esk.h5_align(p, self.COLUMNS)
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1])]  # 'gut' dropped: it gets re-embedded on resume

    def test_column_missing_after_crash_on_first_write(self, tmp_path):
        # A crash (e.g. disk full) during the very FIRST h5_append, after 'texts' was created but before
        # 'embeddings' was. Judged only by the columns present, the file would look complete ('soil' counted
        # as done, but with no vector). The missing 'embeddings' counts as 0 rows, so 'texts' is cut to 0.
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil"])  # 'embeddings' never written
        esk.h5_align(p, self.COLUMNS)
        assert esk.h5_strings(p, "texts") == []  # the half-written row is dropped: 'soil' is embedded again

    def test_append_after_repair_of_missing_column(self, tmp_path):
        p = str(tmp_path / "f.h5")
        esk.h5_append(p, texts=["soil"])  # as above
        esk.h5_align(p, self.COLUMNS)
        esk.h5_append(p, texts=["soil"], embeddings=np.array([[1, 0]], dtype=np.float32))
        assert rows(p) == [("soil", [1, 0])]


# ============================================================================= stage 1: distinct texts
class TestEmbedUnique:
    VECTORS = {"soil": [1, 0], "water": [0, 1], "gut": [1, 1]}

    def embed(self, out_path, texts, client, dim=2, batch_size=10, model="text-embedding-3-small"):
        esk.embed_unique("test", texts, out_path, client, model, dim, batch_size)

    def test_each_text_embedded_once_and_written(self, tmp_path):
        p, client = str(tmp_path / "u.h5"), FakeOpenAI(self.VECTORS)
        self.embed(p, ["soil", "water", "gut"], client)
        assert client.texts_sent == ["soil", "water", "gut"]
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1]), ("gut", [1, 1])]
        with h5py.File(p, "r") as f:
            assert f["embeddings"].dtype == np.float32  # the API returns float64 lists; the real files are float32

    def test_batches(self, tmp_path):
        p, client = str(tmp_path / "u.h5"), FakeOpenAI(self.VECTORS)
        self.embed(p, ["soil", "water", "gut"], client, batch_size=2)
        assert client.batches == [["soil", "water"], ["gut"]]
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1]), ("gut", [1, 1])]

    def test_resume_skips_texts_already_in_file(self, tmp_path):
        p = str(tmp_path / "u.h5")
        self.embed(p, ["soil"], FakeOpenAI(self.VECTORS))
        client = FakeOpenAI(self.VECTORS)
        self.embed(p, ["soil", "water", "gut"], client)
        assert client.texts_sent == ["water", "gut"]  # 'soil' not paid for twice
        assert rows(p) == [("soil", [1, 0]), ("water", [0, 1]), ("gut", [1, 1])]

    def test_nothing_to_do_no_call_no_file(self, tmp_path):
        p, client = str(tmp_path / "u.h5"), FakeOpenAI(self.VECTORS)
        self.embed(p, [], client)
        assert client.calls == []
        assert not (tmp_path / "u.h5").exists()  # main() skips such a target, see the integration tests

    def test_dimensions_sent_when_smaller_than_native(self, tmp_path):
        client = FakeOpenAI(self.VECTORS)
        self.embed(str(tmp_path / "u.h5"), ["soil"], client, dim=2)  # text-embedding-3-small is natively 1536
        assert client.calls == [{"input": ["soil"], "model": "text-embedding-3-small", "dimensions": 2}]

    def test_no_dimensions_at_native_size(self, tmp_path, monkeypatch):
        monkeypatch.setitem(esk.NATIVE_DIM, "toy-model", 2)  # pretend a model whose native size is 2
        client = FakeOpenAI(self.VECTORS)
        self.embed(str(tmp_path / "u.h5"), ["soil"], client, dim=2, model="toy-model")
        assert client.calls == [{"input": ["soil"], "model": "toy-model"}]  # no 'dimensions' key

    def test_wrong_vector_size_from_api_stops_before_writing(self, tmp_path):
        p = str(tmp_path / "u.h5")
        with pytest.raises(AssertionError, match="expected dim 3"):
            self.embed(p, ["soil"], FakeOpenAI(self.VECTORS), dim=3)  # API returns 2-d vectors
        assert not (tmp_path / "u.h5").exists()

    def test_api_failure_keeps_finished_batches(self, tmp_path):
        p = str(tmp_path / "u.h5")
        with pytest.raises(ConnectionError):
            self.embed(p, ["soil", "water", "gut"], FakeOpenAI(self.VECTORS, fail_on_call=2), batch_size=1)
        assert rows(p) == [("soil", [1, 0])]  # batch 1 is safe on disk; a re-run starts at 'water'


# ============================================================================= stage 2: per-sample rows
class TestMaterialize:
    TEXT_TO_VECTOR = {"soil": np.array([1, 0], dtype=np.float32), "water": np.array([0, 1], dtype=np.float32)}

    def test_vector_copied_to_every_sample_with_that_text(self, tmp_path):
        p = str(tmp_path / "s.h5")
        samples = [("s1", "soil"), ("s2", "soil"), ("s3", "water")]
        esk.materialize("test", iter(samples), self.TEXT_TO_VECTOR, p)
        assert rows(p) == [("s1", "soil", [1, 0]), ("s2", "soil", [1, 0]), ("s3", "water", [0, 1])]

    def test_small_chunks_give_the_same_file(self, tmp_path):
        p = str(tmp_path / "s.h5")
        samples = [("s1", "soil"), ("s2", "soil"), ("s3", "water")]
        esk.materialize("test", iter(samples), self.TEXT_TO_VECTOR, p, chunk=2)  # writes 2 rows, then 1
        assert rows(p) == [("s1", "soil", [1, 0]), ("s2", "soil", [1, 0]), ("s3", "water", [0, 1])]

    def test_resume_skips_samples_already_in_file(self, tmp_path):
        p = str(tmp_path / "s.h5")
        esk.materialize("test", iter([("s1", "soil")]), self.TEXT_TO_VECTOR, p)
        esk.materialize("test", iter([("s1", "soil"), ("s2", "water")]), self.TEXT_TO_VECTOR, p)
        assert rows(p) == [("s1", "soil", [1, 0]), ("s2", "water", [0, 1])]  # s1 not written twice

    def test_sample_whose_text_has_no_vector_is_skipped_with_warning(self, tmp_path, capsys):
        p = str(tmp_path / "s.h5")
        esk.materialize("test", iter([("s1", "soil"), ("s2", "gut")]), self.TEXT_TO_VECTOR, p)  # no 'gut' vector
        assert rows(p) == [("s1", "soil", [1, 0])]
        assert "WARNING 1 samples had no embedded text" in capsys.readouterr().out

    def test_no_samples_no_file(self, tmp_path):
        esk.materialize("test", iter([]), self.TEXT_TO_VECTOR, str(tmp_path / "s.h5"))
        assert not (tmp_path / "s.h5").exists()
