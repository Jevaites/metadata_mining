"""Unit tests for common.py: one class per function, each test checks one behaviour on tiny inputs.

Conventions
  * Inputs are written inline, small enough that the expected output can be checked by eye.
  * Files are written to pytest's `tmp_path` (a fresh temporary directory per test).
  * Tests named `..._quirk` pin down current behaviour that is surprising but harmless today.

See test_common_mutations.py for the proof that these tests catch plausible bugs in common.py.

See test_common_integration.py for how the functions are chained in the pipeline.
"""

import gzip
import hashlib

import h5py
import numpy as np
import pandas as pd
import pytest

import common


# ----------------------------------------------------------------------------- helpers
def write_tsv(p, rows):
    """rows: list of dicts -> TSV file (all values written as given)."""
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


def write_term_h5(p, texts, vectors):
    """Same layout as 4_embed_terms.py output: 'texts' (utf-8 strings) and 'embeddings' (n x dim)."""
    with h5py.File(p, "w") as handle:
        handle.create_dataset("texts", data=texts, dtype=h5py.string_dtype("utf-8"))
        handle.create_dataset("embeddings", data=np.array(vectors, dtype=float))
    return str(p)


def md5_hex(s):
    return hashlib.md5(s.encode()).hexdigest()


# ============================================================================= files
class TestPath:
    def test_expands_home(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/alice")
        assert common.path("~/data/x.tsv") == "/home/alice/data/x.tsv"



class TestReadTsv:
    def test_every_cell_is_a_string_and_empty_cells_stay_empty(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("id\tcount\tnote\n007\t3\t\nx\tNA\tnan\n")
        df = common.read_tsv(str(p))
        assert df.to_dict("records") == [
            {"id": "007", "count": "3", "note": ""},   # leading zeros kept, number not parsed, "" not NaN
            {"id": "x", "count": "NA", "note": "nan"},  # 'NA' / 'nan' are not turned into NaN
        ]


class TestFileMd5:
    def test_known_value(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"hello")
        assert common.file_md5(str(p)) == "5d41402abc4b2a76b9719d911017c592"  # md5("hello")

    def test_file_larger_than_one_read_block(self, tmp_path):
        content = b"x" * (3 * (1 << 20) + 5)  # 3 MiB + 5 bytes: read in 4 blocks of <= 1 MiB
        p = tmp_path / "big.bin"
        p.write_bytes(content)
        assert common.file_md5(str(p)) == hashlib.md5(content).hexdigest()


# ============================================================================= ontology terms
class TestTermText:
    def test_label_then_synonyms_joined_by_semicolons(self):
        terms = pd.DataFrame({"label": ["fecal material"], "synonyms": ["droppings||frass||pellet"]})
        assert common.term_text(terms).tolist() == ["fecal material; droppings; frass; pellet"]

    def test_no_synonyms_leaves_a_trailing_separator(self):
        # Kept on purpose: the term embeddings (.h5) are keyed by this exact string, "soil; " included.
        terms = pd.DataFrame({"label": ["soil"], "synonyms": [""]})
        assert common.term_text(terms).tolist() == ["soil; "]

    def test_newlines_and_spaces_inside_synonyms_are_kept_verbatim(self):
        # Real cases (shortened): 2 FOODON synonyms span lines, 18 PO synonyms end with a space.
        # The .h5 is keyed by the exact text, and it holds these texts with their '\n' and spaces.
        terms = pd.DataFrame({"label": ["dry bean food product", "crown root"],
                              "synonyms": ["kidney bean (Ph. vulgaris); \nlima bean (Ph. lunatus)",
                                           "adventitious root ||nodal root"]})
        assert common.term_text(terms).tolist() == [
            "dry bean food product; kidney bean (Ph. vulgaris); \nlima bean (Ph. lunatus)",
            "crown root; adventitious root ; nodal root"]


class TestLoadTerms:
    def test_drops_obsolete_terms_and_adds_text(self, tmp_path):
        p = write_tsv(tmp_path / "terms.tsv", [
            {"term_id": "ENVO_1", "label": "soil", "synonyms": "dirt", "parents": "", "obsolete": "False"},
            {"term_id": "ENVO_2", "label": "old soil", "synonyms": "", "parents": "", "obsolete": "True"},
            {"term_id": "ENVO_3", "label": "water", "synonyms": "", "parents": "", "obsolete": "False"},
        ])
        terms = common.load_terms(p)
        assert terms["term_id"].tolist() == ["ENVO_1", "ENVO_3"]
        assert terms["text"].tolist() == ["soil; dirt", "water; "]
        assert terms.index.tolist() == [0, 1]  # index reset after the filter: row i = position i

    def test_reads_the_real_table_format(self, tmp_path):
        # As in ontology_terms.tsv.gz: gzipped; a synonym spanning lines is quoted, with "" for a quote;
        # 'NA' (nasal artery) and 'nan' (naan) are real synonyms and must stay text.
        p = tmp_path / "ontology_terms.tsv.gz"
        with gzip.open(p, "wt") as handle:
            handle.write("ontology\tterm_id\tlabel\tsynonyms\tdefinition\tparents\tobsolete\n"
                         'FOODON\tFOODON_00001730\tchicory food product\t"Blue daisy.\n\nNote: ""Cornflower"" '
                         'is Centaurea."\t\tFOODON_00001010\tFalse\n'
                         "UBERON\tUBERON_2005085\tnasal artery\tNA\tThe nasal arteries.\tUBERON_0001637\tFalse\n"
                         "FOODON\tFOODON_00002532\tnaan\tnaan bread||nan\t\tFOODON_00001010\tFalse\n"
                         "PO\tPO_0007514\tOBSOLETE B3 crown root meristem formation\t\tOBSOLETE.\t\tTrue\n")
        terms = common.load_terms(str(p))
        assert terms["term_id"].tolist() == ["FOODON_00001730", "UBERON_2005085", "FOODON_00002532"]
        assert terms["text"].tolist() == ['chicory food product; Blue daisy.\n\nNote: "Cornflower" is Centaurea.',
                                          "nasal artery; NA", "naan; naan bread; nan"]


class TestLoadTermVectors:
    def test_rows_follow_the_requested_order_and_are_unit_length(self, tmp_path):
        p = write_term_h5(tmp_path / "t.h5", ["soil; ", "water; "], [[3, 4], [0, 2]])
        vectors = common.load_term_vectors(str(p), ["water; ", "soil; "])  # reversed order
        np.testing.assert_allclose(vectors, [[0, 1],        # water (0, 2) / 2
                                             [0.6, 0.8]])   # soil  (3, 4) / 5

    def test_a_text_can_be_requested_twice(self, tmp_path):
        p = write_term_h5(tmp_path / "t.h5", ["soil; "], [[1, 0]])
        np.testing.assert_allclose(common.load_term_vectors(str(p), ["soil; ", "soil; "]), [[1, 0], [1, 0]])

    def test_missing_text_stops_with_a_hint_to_re_embed(self, tmp_path):
        p = write_term_h5(tmp_path / "t.h5", ["soil; "], [[1, 0]])
        with pytest.raises(SystemExit, match=r"1 term texts are not in .*'water; '.*re-run 4_embed_terms.py"):
            common.load_term_vectors(str(p), ["soil; ", "water; "])


class TestAncestorSets:
    def test_chain(self):
        # a -> b -> c   (arrow = "is_a")
        assert common.ancestor_sets({"a": {"b"}, "b": {"c"}}) == {"a": {"b", "c"}, "b": {"c"}, "c": set()}

    def test_several_parents_union_of_both_branches(self):
        #     a
        #    / \
        #   b   c
        #   |   |
        #   d   e
        result = common.ancestor_sets({"a": {"b", "c"}, "b": {"d"}, "c": {"e"}})
        assert result["a"] == {"b", "c", "d", "e"}

    def test_diamond_shared_ancestor_counted_once(self):
        # a -> b -> d  and  a -> c -> d
        result = common.ancestor_sets({"a": {"b", "c"}, "b": {"d"}, "c": {"d"}})
        assert result["a"] == {"b", "c", "d"}

    def test_parents_that_are_not_keys_appear_as_keys_without_ancestors(self):
        # 'BFO_1' is only a parent (an upper class outside the term table)
        result = common.ancestor_sets({"soil": {"BFO_1"}})
        assert result == {"soil": {"BFO_1"}, "BFO_1": set()}

    def test_cycle_terminates(self):
        # never seen in ENVO / Uberon; the only requirement is not to recurse forever
        result = common.ancestor_sets({"a": {"b"}, "b": {"a"}})
        assert "b" in result["a"] and "a" in result["b"]


class TestTermAncestors:
    def test_reads_the_parents_column(self):
        terms = pd.DataFrame({"term_id": ["lake", "freshwater", "aquatic"],
                              "parents": ["freshwater", "aquatic||BFO_1", ""]})  # '' = no parent
        result = common.term_ancestors(terms)
        assert result["lake"] == {"freshwater", "aquatic", "BFO_1"}
        assert result["freshwater"] == {"aquatic", "BFO_1"}
        assert result["aquatic"] == set()


class TestAncestorDistances:
    def test_shortest_number_of_steps(self):
        # a -> b -> c -> BFO_1, plus a shortcut a -> c
        terms = pd.DataFrame({"term_id": ["a", "b", "c"], "parents": ["b||c", "c", "BFO_1"]})
        result = common.ancestor_distances(terms)
        assert result == {
            "a": {"b": 1, "c": 1, "BFO_1": 2},  # c is 1 step (shortcut), not 2 (via b)
            "b": {"c": 1, "BFO_1": 2},
            "c": {"BFO_1": 1},
        }

    def test_only_table_terms_are_keys(self):
        # unlike term_ancestors, an outside parent (BFO_1) is a value but never a key
        terms = pd.DataFrame({"term_id": ["soil"], "parents": ["BFO_1"]})
        assert common.ancestor_distances(terms) == {"soil": {"BFO_1": 1}}


class TestCoarsenLabels:
    # The docstring example:   lake -> freshwater -> aquatic,  river -> freshwater,  marine -> aquatic,  soil
    DISTANCES = {"lake": {"freshwater": 1, "aquatic": 2}, "river": {"freshwater": 1, "aquatic": 2},
                 "freshwater": {"aquatic": 1}, "marine": {"aquatic": 1}, "aquatic": {}, "soil": {}}
    KNOWN = {"lake", "river", "freshwater", "marine", "aquatic", "soil"}
    LABELS = ["lake"] * 60 + ["river"] * 30 + ["freshwater"] * 20 + ["marine"] * 150 + ["soil"] * 5

    def test_docstring_example(self):
        mapping = common.coarsen_labels(self.LABELS, self.DISTANCES, 100, self.KNOWN)
        assert mapping == {
            "lake": "freshwater",        # 60 < 100; freshwater has 110 (closer than aquatic, 260)
            "river": "freshwater",       # 30 < 100
            "freshwater": "freshwater",  # 20 own samples, but 110 with lake + river below it
            "marine": "marine",          # 150 >= 100
            "soil": "soil",              # 5 < 100 and no ancestor: kept
        }

    def test_exactly_min_support_is_enough(self):
        # lake has exactly 5 samples: it qualifies (>= 5) and is not coarsened to freshwater (6)
        labels = ["lake"] * 5 + ["river"] * 1
        mapping = common.coarsen_labels(labels, self.DISTANCES, 5, self.KNOWN)
        assert mapping == {"lake": "lake", "river": "freshwater"}

    def test_min_support_1_changes_nothing(self):
        mapping = common.coarsen_labels(self.LABELS, self.DISTANCES, 1, self.KNOWN)
        assert all(old == new for old, new in mapping.items())

    def test_empty_label_is_ignored(self):
        mapping = common.coarsen_labels(["", "", "soil"], {"soil": {}}, 1, {"soil"})
        assert mapping == {"soil": "soil"}

    def test_ancestor_outside_known_is_never_used(self):
        # aquatic removed from the term table: lake can no longer climb to it, even if it would qualify
        mapping = common.coarsen_labels(self.LABELS, self.DISTANCES, 200, self.KNOWN - {"aquatic"})
        assert mapping["lake"] == "lake"

    def test_tie_between_two_parents_goes_to_the_smaller_id(self):
        # x has two parents at distance 1, both with 5 samples: "p" < "q" alphabetically
        labels = ["x"] + ["p"] * 5 + ["q"] * 5
        mapping = common.coarsen_labels(labels, {"x": {"p": 1, "q": 1}, "p": {}, "q": {}}, 5, {"x", "p", "q"})
        assert mapping["x"] == "p"


# ============================================================================= MicrobeAtlas metadata records
class TestIterSampleInfo:
    TEXT = (">SRS1\n"
            "sample_env_biome=human gut\n"
            "study_STUDY_TITLE=Infant gut  \n"   # trailing spaces are kept (only '\n' is removed)
            ">SRS2\n"
            ">SRS3\n"
            "sample_env_material=soil\n")
    EXPECTED = [("SRS1", ["sample_env_biome=human gut", "study_STUDY_TITLE=Infant gut  "]),
                ("SRS2", []),                             # a record without lines is still yielded
                ("SRS3", ["sample_env_material=soil"])]   # the last record (no '>' after it) too

    def test_plain_file(self, tmp_path):
        p = tmp_path / "sample.info"
        p.write_text(self.TEXT)
        assert list(common.iter_sample_info(str(p))) == self.EXPECTED

    def test_gzipped_file(self, tmp_path):
        p = tmp_path / "sample.info.gz"
        with gzip.open(p, "wt") as handle:
            handle.write(self.TEXT)
        assert list(common.iter_sample_info(str(p))) == self.EXPECTED

    def test_lines_before_the_first_header_are_ignored(self, tmp_path):
        p = tmp_path / "sample.info"
        p.write_text("orphan=line\n>SRS1\nkey=value\n")
        assert list(common.iter_sample_info(str(p))) == [("SRS1", ["key=value"])]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "sample.info"
        p.write_text("")
        assert list(common.iter_sample_info(str(p))) == []

    def test_header_without_id_is_skipped_quirk(self, tmp_path):
        # a bare '>' gives sample_id '' (falsy): its record is silently dropped
        p = tmp_path / "sample.info"
        p.write_text(">\nkey=lost\n>SRS2\nkey=kept\n")
        assert list(common.iter_sample_info(str(p))) == [("SRS2", ["key=kept"])]

    def test_real_layout_with_blank_lines(self, tmp_path):
        # sample.info.gz starts with an empty line and ends every record with one; the empty line is
        # kept as '' (record_to_text drops it)
        p = tmp_path / "sample.info.gz"
        with gzip.open(p, "wt") as handle:
            handle.write("\n>SRS7471636\nsample_center_name=\nruns=SRR12767728\n\n>SRS4278871\nsample_TITLE=x\n\n")
        assert list(common.iter_sample_info(str(p))) == [
            ("SRS7471636", ["sample_center_name=", "runs=SRR12767728", ""]),
            ("SRS4278871", ["sample_TITLE=x", ""])]

    def test_carriage_return_inside_a_value_becomes_a_space(self, tmp_path):
        # 20 real lines contain a lone '\r' (pasted text; shortened from SRS431655 and SRS000345). Read with
        # universal newlines, the line was split there and the part after '\r' was lost (it has no '=').
        p = tmp_path / "sample.info.gz"
        with gzip.open(p, "wb") as handle:
            handle.write(b">SRS431655\nsample_DESCRIPTION=eutrophication in Lake\r Geneva Switzerland-France\n"
                         b"study_STUDY_ABSTRACT=\r <p>\r Clostridium sp.\n")
        assert list(common.iter_sample_info(str(p))) == [
            ("SRS431655", ["sample_DESCRIPTION=eutrophication in Lake  Geneva Switzerland-France",
                           "study_STUDY_ABSTRACT=  <p>  Clostridium sp."])]

    def test_crlf_line_endings_are_removed(self, tmp_path):
        p = tmp_path / "sample.info"
        p.write_bytes(b">SRS1\r\nsample_env_biome=soil\r\n>SRS2\r\n")
        assert list(common.iter_sample_info(str(p))) == [("SRS1", ["sample_env_biome=soil"]), ("SRS2", [])]

    def test_invalid_utf8_is_replaced_in_gz_file(self, tmp_path):
        p = tmp_path / "sample.info.gz"
        with gzip.open(p, "wb") as handle:
            handle.write(b">SRS1\nsite=caf\xe9\n")  # latin-1 'é', invalid in utf-8
        assert list(common.iter_sample_info(str(p))) == [("SRS1", ["site=caf�"])]

    def test_invalid_utf8_is_replaced_in_plain_file(self, tmp_path):
        # was a bug (plain files were opened without errors="replace"); fixed 2026-10-19
        p = tmp_path / "sample.info"
        p.write_bytes(b">SRS1\nsite=caf\xe9\n")
        assert list(common.iter_sample_info(str(p))) == [("SRS1", ["site=caf�"])]


class TestRecordToText:
    def text(self, lines, code_to_label=None, max_chars=None):
        return common.record_to_text(lines, code_to_label or {}, max_chars)

    def test_docstring_example(self):
        lines = ["sample_env_material=feces [ENVO:00002003]", "sample_collection_date=2014-05-01",
                 "sample_host_age=missing"]
        assert self.text(lines, {"ENVO_00002003": "fecal material"}) == "env_material: feces [fecal material]"

    def test_real_record(self):
        # SRS7471636 of sample.info.gz, shortened: only 3 of its 17 lines carry habitat information
        lines = ["sample_center_name=", "sample_alias=AOA_enrichment_SS", "sample_TAXON_ID=2109671",
                 "sample_SCIENTIFIC_NAME=Candidatus Nitrosocosmicus sp.", "sample_XREF_LINK=bioproject: 667025)",
                 "sample_isolation_source=ammonia-oxidizing archaea enrichment inoculated with soil",
                 "sample_collection_date=2018-12-01", "sample_lat_lon=23.13 N 113.28 E",
                 "sample_sample_type=missing", "experiments=SRX9238413", "study=SRP286249",
                 "study_STUDY_TITLE=AOA_enrichment_SS", "study_STUDY_TYPE=", "study_STUDY_LINK=      ",
                 "experiment_LIBRARY_STRATEGY=WGS", "runs=SRR12767728", ""]
        assert self.text(lines) == ("scientific_name: Candidatus Nitrosocosmicus sp.; "
                                    "isolation_source: ammonia-oxidizing archaea enrichment inoculated with soil; "
                                    "study_title: AOA_enrichment_SS")

    def test_key_prefix_removed_and_key_lowercased(self):
        assert self.text(["sample_Env_Material=Soil", "study_STUDY_TITLE=Gut Study"]) == \
            "env_material: Soil; study_title: Gut Study"  # values keep their case

    def test_spaces_around_key_and_value_are_stripped(self):
        assert self.text(["  sample_depth = 5 m  "]) == "depth: 5 m"

    def test_only_the_first_equals_sign_splits(self):
        assert self.text(["sample_note=ratio=1:2"]) == "note: ratio=1:2"

    @pytest.mark.parametrize("key", [
        "geo_loc_name", "isolation_source", "host_body_site", "depth", "temperature", "title", "description",
    ])
    def test_habitat_keys_are_kept(self, key):
        assert self.text([f"{key}=x"]) == f"{key}: x"

    @pytest.mark.parametrize("raw_key", [
        "sample_experiment_title",   # ^experiment
        "sample_run_accession",      # ^run
        "study",                     # ^study$
        "sample name",               # ^sample name$
        "sample_collection_date",    # date
        "sample_first_public",       # public
        "sample_lat_lon",            # lat_lon
        "sample_geographic location (latitude)",  # latitude
        "sample_host_subject_id",    # subject, _id$
        "sample_biosample id",       # ' id$'
        "sample_INSDC center name",  # insdc / center (after lowercasing)
        "sample_library_strategy",   # library
        "sample_replicate",          # replicate
    ])
    def test_identifier_date_and_coordinate_keys_are_dropped(self, raw_key):
        assert self.text([f"{raw_key}=x"]) == ""

    @pytest.mark.parametrize("raw_key", ["sample_runoff_type", "sample_maritime_zone"])
    def test_substring_matching_also_drops_some_habitat_keys_quirk(self, raw_key):
        # DROP_KEYS matches substrings: 'runoff' starts with 'run', 'maritime' contains 'time'
        assert self.text([f"{raw_key}=x"]) == ""

    @pytest.mark.parametrize("value", ["", "NA", "n/a", "  Not Collected ", "missing.", "-", "unknown", "NULL"])
    def test_missing_values_are_dropped(self, value):
        assert self.text([f"sample_depth={value}"]) == ""

    @pytest.mark.parametrize("value", ["0", "none detected", "unknown soil", "NA12878"])
    def test_values_that_only_contain_a_missing_word_are_kept(self, value):
        # MISSING is compared with the whole value, not searched inside it
        assert self.text([f"sample_x={value}"]) == f"x: {value}"

    def test_line_without_equals_sign_is_dropped(self):
        assert self.text(["just some text"]) == ""   # value '' -> missing

    def test_ontology_codes_replaced_by_labels(self):
        code_to_label = {"ENVO_00002003": "fecal material", "UBERON_0001988": "feces"}
        assert self.text(["sample_a=ENVO:00002003", "sample_b=ENVO_00002003", "sample_c=UBERON:0001988"],
                         code_to_label) == "a: fecal material; b: fecal material; c: feces"

    def test_codes_not_replaced(self):
        code_to_label = {"ENVO_00002003": "fecal material"}
        assert self.text(["sample_a=ENVO:12345678",     # unknown code: kept as written
                          "sample_b=envo:00002003",     # lowercase prefix: not recognised
                          "sample_c=ENVO:000020031"],   # 9 digits: not a code
                         code_to_label) == "a: ENVO:12345678; b: envo:00002003; c: ENVO:000020031"

    def test_duplicate_lines_kept_once_in_first_order(self):
        # sample_env and study_env both become 'env: soil' once the prefix is removed
        assert self.text(["sample_env=soil", "sample_depth=5", "study_env=soil"]) == "env: soil; depth: 5"

    def test_truncated_to_max_chars(self):
        assert self.text(["sample_env_material=soil"], max_chars=10) == "env_materi"

    def test_max_chars_none_means_no_truncation(self):
        assert self.text(["sample_env_material=soil"], max_chars=None) == "env_material: soil"

    def test_empty_record(self):
        assert self.text([]) == ""


# ============================================================================= samples, folds
class TestLoadNpz:
    def test_rows_shared_vectors_and_normalisation(self, tmp_path):
        # S1 and S2 have the same GPT text, hence one shared vector (index 0)
        p = tmp_path / "kw.npz"
        np.savez(p, sample_ids=np.array(["S1", "S2", "S3"]), index=np.array([0, 0, 1]),
                 vectors=np.array([[3.0, 4.0], [0.0, 5.0]]))
        row_of, matrix = common.load_npz(str(p))
        assert row_of == {"S1": 0, "S2": 1, "S3": 2}
        np.testing.assert_allclose(matrix, [[0.6, 0.8],   # S1: (3, 4) / 5
                                            [0.6, 0.8],   # S2: same text as S1
                                            [0.0, 1.0]])  # S3: (0, 5) / 5

    def test_zero_vector_stays_zero(self, tmp_path):
        # sklearn's normalize leaves an all-zero row as zeros (no division by zero, no NaN)
        p = tmp_path / "kw.npz"
        np.savez(p, sample_ids=np.array(["S1"]), index=np.array([0]), vectors=np.zeros((1, 2)))
        np.testing.assert_array_equal(common.load_npz(str(p))[1], [[0.0, 0.0]])


class TestStableKey:
    def test_is_md5_of_seed_colon_value(self):
        assert common.stable_key(22, ["SRS1"]).tolist() == [md5_hex("22:SRS1")]


class TestSelectSamples:
    @staticmethod
    def sample(sample_id, study, material="ENVO_1", biome="", feature=""):
        return {"sample_id": sample_id, "study_code": study, "biome": biome, "feature": feature, "material": material}

    @staticmethod
    def smallest_keys(ids, n, seed=22):
        """The n ids select_samples keeps under a cap of n (smallest md5('seed:id'))."""
        return set(sorted(ids, key=lambda s: md5_hex(f"{seed}:{s}"))[:n])

    def test_keeps_samples_with_at_least_one_slot_label(self, tmp_path):
        p = write_tsv(tmp_path / "s.tsv", [
            self.sample("S1", "A", material="ENVO_1"),
            self.sample("S2", "A", material="", biome="ENVO_2"),   # biome only: kept
            self.sample("S3", "A", material=""),                   # no label at all: dropped
        ])
        assert set(common.select_samples(p)["sample_id"]) == {"S1", "S2"}

    def test_keeps_samples_present_in_every_required_set(self, tmp_path):
        p = write_tsv(tmp_path / "s.tsv", [self.sample(s, "A") for s in ["S1", "S2", "S3"]])
        with_keywords, with_sub_biomes = {"S1", "S2"}, {"S2", "S3"}
        assert common.select_samples(p, [with_keywords, with_sub_biomes])["sample_id"].tolist() == ["S2"]

    def test_caps_each_study_keeping_the_smallest_keys(self, tmp_path):
        # rows written in DEcreasing key order, so "the first 2 rows of the file" would be the wrong answer
        ids_a = sorted([f"A{i}" for i in range(5)], key=lambda s: md5_hex(f"22:{s}"), reverse=True)
        p = write_tsv(tmp_path / "s.tsv", [self.sample(s, "A") for s in ids_a] + [self.sample("B0", "B")])
        selected = common.select_samples(p, max_per_study=2)
        assert set(selected["sample_id"]) == self.smallest_keys(ids_a, 2) | {"B0"}

    def test_defaults_cap_50_per_study_with_seed_22(self, tmp_path):
        # 9 experiments scripts call select_samples with the defaults: changing them changes their samples
        ids = [f"SRS{i:07d}" for i in range(51)]
        p = write_tsv(tmp_path / "s.tsv", [self.sample(s, "Probst_2018_aquifer") for s in ids])
        assert set(common.select_samples(p)["sample_id"]) == self.smallest_keys(ids, 50, seed=22)

    def test_max_per_study_0_means_no_cap(self, tmp_path):
        p = write_tsv(tmp_path / "s.tsv", [self.sample(f"A{i}", "A") for i in range(5)])
        assert len(common.select_samples(p, max_per_study=0)) == 5

    def test_rows_ordered_by_stable_key_with_a_fresh_index(self, tmp_path):
        ids = ["S1", "S2", "S3", "S4"]
        p = write_tsv(tmp_path / "s.tsv", [self.sample(s, "A") for s in ids])
        selected = common.select_samples(p)
        assert selected["sample_id"].tolist() == sorted(ids, key=lambda s: md5_hex(f"22:{s}"))
        assert selected.index.tolist() == [0, 1, 2, 3]

    def test_seed_changes_which_samples_are_kept(self, tmp_path):
        ids = [f"A{i}" for i in range(20)]
        p = write_tsv(tmp_path / "s.tsv", [self.sample(s, "A") for s in ids])
        kept = {seed: set(common.select_samples(p, max_per_study=5, seed=seed)["sample_id"]) for seed in (1, 2)}
        assert kept[1] == self.smallest_keys(ids, 5, seed=1)
        assert kept[2] == self.smallest_keys(ids, 5, seed=2)
        assert kept[1] != kept[2]


class TestStudyFolds:
    STUDIES = ["A", "A", "B", "C", "C", "C", "D", "E", "E"]   # 9 samples, 5 studies

    def test_every_sample_is_tested_exactly_once(self):
        tested = np.concatenate([test for _, test in common.study_folds(self.STUDIES, n_folds=3)])
        assert sorted(tested.tolist()) == list(range(9))

    def test_train_and_test_are_complementary(self):
        for train, test in common.study_folds(self.STUDIES, n_folds=3):
            assert sorted(train.tolist() + test.tolist()) == list(range(9))

    def test_no_study_on_both_sides(self):
        codes = np.array(self.STUDIES)
        for train, test in common.study_folds(codes, n_folds=3):
            assert set(codes[train]).isdisjoint(codes[test])

    def test_fold_of_a_study_is_md5_mod_n_folds(self):
        folds = list(common.study_folds(["A", "B", "C"], n_folds=4, seed=0))
        for i, study in enumerate(["A", "B", "C"]):
            expected_fold = int(md5_hex(f"0:{study}"), 16) % 4
            assert i in folds[expected_fold][1]

    def test_study_keeps_its_fold_when_other_studies_are_added(self):
        def fold_of(codes):
            return {codes[i]: f for f, (_, test) in enumerate(common.study_folds(codes, 5)) for i in test}
        before = fold_of(["A", "B", "C"])
        after = fold_of(["A", "B", "C", "NEW1", "NEW2"])
        assert {s: after[s] for s in before} == before

    def test_more_folds_than_studies_gives_empty_test_folds(self):
        # 2 studies cannot fill 5 folds: at least 3 folds have no test sample (callers must cope)
        folds = list(common.study_folds(["A", "A", "B"], n_folds=5))
        assert len(folds) == 5
        assert sum(len(test) == 0 for _, test in folds) >= 3
