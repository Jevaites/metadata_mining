"""
Unit tests for 2_build_training_set.py: the two regular expressions, load_metalog_labels() and
snapshot_paths().

(The linking of MicrobeAtlas records to Metalog samples lives inside main(); it is tested end to end in
test_2_build_training_set_integration.py. test_2_build_training_set_mutations.py shows which tests fail
for which plausible bug.)

Ontology ids used throughout (real ENVO / UBERON ids, so that the values look like Metalog's):
    ENVO_00001998   soil
    ENVO_00002003   fecal material
    ENVO_00000447   marine biome
    ENVO_00002149   sea water
    UBERON_0001988  feces
"""

import os

import pytest

from step2_helpers import step2, write_metalog_long  # noqa: F401  (step2 is a pytest fixture)


# ============================================================================= GOLD_VALUE
# Extracts the ontology code from a Metalog value. Metalog writes a curated term as
# "<label> [<PREFIX>:<digits>]"; only ENVO and UBERON codes in square brackets count.
@pytest.mark.parametrize("value, expected", [
    ("soil [ENVO:00001998]",                        ("ENVO", "00001998")),
    ("feces [UBERON:0001988]",                      ("UBERON", "0001988")),
    # two codes in one value: the first one is used (no value of the 2026-09-20 snapshot has two; 2b uses the same rule)
    ("feces [UBERON:0001988] [ENVO:00002003]",      ("UBERON", "0001988")),
    # not a label: other ontologies, a code without brackets, the underscore form, nothing at all
    ("apple [FOODON:00002473]",                     None),
    ("ENVO:00001998",                               None),
    ("soil [ENVO_00001998]",                        None),
    ("soil",                                        None),
    ("",                                            None),
])
def test_gold_value_regex(step2, value, expected):
    match = step2.GOLD_VALUE.search(value)
    assert (match.groups() if match else None) == expected


# ============================================================================= ACCESSION
# Finds the sample accessions written inside a MicrobeAtlas record:
# BioSample (SAMN..., SAMEA..., SAMD...) and SRA / ENA / DDBJ sample ids (SRS..., ERS..., DRS...).
@pytest.mark.parametrize("text, expected", [
    ("sample_biosample=SAMN00000001",                   ["SAMN00000001"]),
    ("sample_biosample=SAMEA0000001",                   ["SAMEA0000001"]),
    ("sample_biosample=SAMD00000001",                   ["SAMD00000001"]),
    ("sample_secondary=SRS000001 ERS000002 DRS000003",  ["SRS000001", "ERS000002", "DRS000003"]),
    # several accessions: all of them, in the order they appear
    ("ERS000002; SAMEA0000001",                         ["ERS000002", "SAMEA0000001"]),
    # whole words only: not glued to letters, digits or '_' ...
    ("sample_alias=SAMEA0000001_rep1",                  []),
    ("xSRS000001",                                      []),
    # ... upper case only, and only the S(ample) kinds of SRA id (not runs SRR / experiments SRX / studies SRP)
    ("samea0000001",                                    []),
    ("SRR000001 SRX000001 SRP000001",                   []),
])
def test_accession_regex(step2, text, expected):
    assert step2.ACCESSION.findall(text) == expected


# ============================================================================= load_metalog_labels
# Input: the <domain>_all_long_<date>.tsv.gz files of one snapshot, one row per (sample_alias, metadata_item,
# curation_tier, value).
# Output: one row per Metalog sample, keyed by spire_sample_name (the accession MicrobeAtlas records are
# linked by), with study_code, domain and one term id per slot (biome / feature / material, "" if none).

def sample(alias, spire, study="", biome="", feature="", material=""):
    """The long-table rows of one Metalog sample, with all five items the function reads.
    sample("h1", "SAMEA1", material="feces [UBERON:0001988]") ->
        [("h1", "spire_sample_name", "SAMEA1"), ("h1", "study_code", ""), ("h1", "environment_biome", ""),
         ("h1", "environment_feature", ""), ("h1", "environment_material", "feces [UBERON:0001988]")]"""
    return [(alias, "spire_sample_name", spire), (alias, "study_code", study), (alias, "environment_biome", biome),
            (alias, "environment_feature", feature), (alias, "environment_material", material)]


def labels_of(step2, tmp_path, files, valid_terms):
    """Write `files` ({filename: rows}) to a folder and run load_metalog_labels on it.
    Returns {spire_sample_name: {column: value}} so that tests can compare plain dicts."""
    for filename, rows in files.items():
        write_metalog_long(tmp_path, filename, rows)
    labels = step2.load_metalog_labels(str(tmp_path), valid_terms)
    return labels.set_index("spire_sample_name")[["sample_alias", "study_code", "domain", "biome", "feature",
                                                  "material"]].to_dict("index")


VALID = {"ENVO_00001998", "ENVO_00002003", "ENVO_00000447", "ENVO_00002149", "UBERON_0001988"}


def test_long_table_becomes_one_row_per_sample(step2, tmp_path):
    """The basic case, with the raw long format spelled out: two samples -> two rows, term ids in the slots."""
    files = {"environmental_all_long_2026-09-20.tsv.gz": [
        # sample_alias  metadata_item           value
        ("sea_1",  "spire_sample_name",    "ERS000001"),
        ("sea_1",  "study_code",           "Ocean_2021"),
        ("sea_1",  "environment_biome",    "marine biome [ENVO:00000447]"),
        ("sea_1",  "environment_feature",  ""),
        ("sea_1",  "environment_material", "sea water [ENVO:00002149]"),
        ("soil_1", "spire_sample_name",    "SAMEA0000002"),
        ("soil_1", "study_code",           "Soil_2020"),
        ("soil_1", "environment_material", "soil [ENVO:00001998]"),
        ("soil_1", "host_age",             "42"),  # any other metadata item is ignored
    ]}
    assert labels_of(step2, tmp_path, files, VALID) == {
        "ERS000001":    {"sample_alias": "sea_1", "study_code": "Ocean_2021", "domain": "environmental",
                         "biome": "ENVO_00000447", "feature": "", "material": "ENVO_00002149"},
        # soil_1 has no biome / feature rows at all: its slots are "", not NaN
        "SAMEA0000002": {"sample_alias": "soil_1", "study_code": "Soil_2020", "domain": "environmental",
                         "biome": "", "feature": "", "material": "ENVO_00001998"},
    }


def test_domain_is_the_file_name_prefix_and_files_are_concatenated(step2, tmp_path):
    files = {
        "human_all_long_2026-09-20.tsv.gz":  sample("h1", "SAMEA0000001", material="feces [UBERON:0001988]"),
        "animal_all_long_2026-09-20.tsv.gz": sample("a1", "SAMEA0000002", material="fecal material [ENVO:00002003]"),
    }
    labels = labels_of(step2, tmp_path, files, VALID)
    assert {spire: row["domain"] for spire, row in labels.items()} == {"SAMEA0000002": "animal",
                                                                       "SAMEA0000001": "human"}
    assert labels["SAMEA0000001"]["material"] == "UBERON_0001988"  # UBERON codes are labels too


def test_other_files_in_the_folder_are_ignored(step2, tmp_path):
    """Only *_all_long_*.tsv.gz files are read. The real folder also holds the *_core_long_ / *_extended_long_
    subsets of the same rows and the *_core_wide_ tables used by step 2b."""
    files = {
        "human_all_long_2026-09-20.tsv.gz":      sample("h1", "SAMEA0000001"),
        "human_core_wide_2026-09-20.tsv.gz":     sample("h2", "SAMEA0000002"),
        "human_all_long_2026-09-20.tsv":         sample("h3", "SAMEA0000003"),  # not .gz
        "human_core_long_2026-09-20.tsv.gz":     sample("h4", "SAMEA0000004"),
        "human_extended_long_2026-09-20.tsv.gz": sample("h5", "SAMEA0000005"),
    }
    assert list(labels_of(step2, tmp_path, files, VALID)) == ["SAMEA0000001"]


def test_code_not_in_valid_terms_is_blanked(step2, tmp_path):
    """A well-formed code that is obsolete or missing from the term index is not a usable label -> ""."""
    files = {"human_all_long_2026-09-20.tsv.gz": sample(
        "h1", "SAMEA0000001",
        biome="human-associated habitat [ENVO:00009003]",  # obsolete: not in VALID -> ""
        material="feces [UBERON:0001988]")}                 # valid: kept
    row = labels_of(step2, tmp_path, files, VALID)["SAMEA0000001"]
    assert (row["biome"], row["material"]) == ("", "UBERON_0001988")


def test_values_without_a_bracketed_envo_or_uberon_code_give_no_label(step2, tmp_path):
    """The non-label values of the real snapshot: 'negative control' (the only value without a code) and
    PO / FOODON codes (~1,000 values). Step 2 gives no label for them; step 2b reads PO / FOODON itself."""
    files = {"animal_all_long_2026-09-20.tsv.gz": sample(
        "a1", "SAMEA0000001",
        biome="negative control",
        feature="leaf [PO:0025034]",
        material="fermented dairy food product [FOODON:00001053]")}
    row = labels_of(step2, tmp_path, files, VALID)["SAMEA0000001"]
    assert (row["biome"], row["feature"], row["material"]) == ("", "", "")


def test_first_code_is_used_even_when_it_is_invalid(step2, tmp_path):
    """Only the FIRST code is extracted, and only then checked against the term index: if it is obsolete,
    the slot becomes "" even though a valid second code follows (no fall-back to the second code)."""
    files = {"human_all_long_2026-09-20.tsv.gz": sample(
        "h1", "SAMEA0000001", material="old term [ENVO:00009003] | feces [UBERON:0001988]")}
    assert labels_of(step2, tmp_path, files, VALID)["SAMEA0000001"]["material"] == ""


def test_sample_without_spire_sample_name_is_dropped(step2, tmp_path):
    """Without an accession a Metalog sample cannot be linked to MicrobeAtlas, so it is not returned."""
    files = {"human_all_long_2026-09-20.tsv.gz": [
        *sample("h1", "SAMEA0000001"),
        *sample("h2", ""),                                     # empty spire_sample_name
        ("h3", "environment_material", "feces [UBERON:0001988]"),  # no spire_sample_name row at all
    ]}
    assert list(labels_of(step2, tmp_path, files, VALID)) == ["SAMEA0000001"]


def test_repeated_item_of_one_sample_keeps_the_first_value(step2, tmp_path):
    files = {"human_all_long_2026-09-20.tsv.gz": [
        *sample("h1", "SAMEA0000001", material="feces [UBERON:0001988]"),
        ("h1", "environment_material", "fecal material [ENVO:00002003]"),  # second value for h1: ignored
    ]}
    assert labels_of(step2, tmp_path, files, VALID)["SAMEA0000001"]["material"] == "UBERON_0001988"


def test_accession_under_two_aliases_keeps_the_first_alias(step2, tmp_path):
    """The same accession under two Metalog aliases (e.g. a sample re-used by two studies) -> one row.
    "First" = files in alphabetical order, then aliases in alphabetical order within a file
    (pivot_table sorts them), NOT the order of the rows in the file."""
    files = {
        "human_all_long_2026-09-20.tsv.gz": [
            *sample("study_B_h1", "SAMEA0000001", study="Study_B"),  # written first, but 'study_A' < 'study_B'
            *sample("study_A_h1", "SAMEA0000001", study="Study_A"),
        ],
        # 'animal_...' sorts before 'human_...': this file is read first, so its alias wins overall
        "animal_all_long_2026-09-20.tsv.gz": sample("zoo_1", "SAMEA0000001", study="Zoo_2018"),
    }
    assert labels_of(step2, tmp_path, files, VALID) == {
        "SAMEA0000001": {"sample_alias": "zoo_1", "study_code": "Zoo_2018", "domain": "animal",
                         "biome": "", "feature": "", "material": ""}}

    # without the animal file, study_A beats study_B
    (tmp_path / "animal_all_long_2026-09-20.tsv.gz").unlink()
    assert step2.load_metalog_labels(str(tmp_path), VALID)["study_code"].tolist() == ["Study_A"]


def test_slot_present_in_one_file_only(step2, tmp_path):
    """A file without any environment_biome row still gets a biome column ("") after concatenation."""
    files = {
        "human_all_long_2026-09-20.tsv.gz": [("h1", "spire_sample_name", "SAMEA0000001")],  # spire only
        "environmental_all_long_2026-09-20.tsv.gz": sample("e1", "ERS000002", biome="marine biome [ENVO:00000447]"),
    }
    labels = labels_of(step2, tmp_path, files, VALID)
    assert (labels["SAMEA0000001"]["biome"], labels["ERS000002"]["biome"]) == ("", "ENVO_00000447")


# ============================================================================= snapshot_paths
# A Metalog folder may hold several snapshots (one file per domain and date). Only one is read:
# the latest by default, or the one given by --date.

def touch(folder, *names):
    for name in names:
        (folder / name).write_bytes(b"")


def test_snapshot_paths_default_is_the_latest_date(step2, tmp_path):
    touch(tmp_path, "human_all_long_2026-01-15.tsv.gz", "animal_all_long_2026-01-15.tsv.gz",
          "human_all_long_2026-09-20.tsv.gz", "animal_all_long_2026-09-20.tsv.gz")
    date, paths = step2.snapshot_paths(str(tmp_path))
    assert date == "2026-09-20"
    assert [os.path.basename(p) for p in paths] == ["animal_all_long_2026-09-20.tsv.gz",
                                                    "human_all_long_2026-09-20.tsv.gz"]


def test_snapshot_paths_explicit_date(step2, tmp_path):
    touch(tmp_path, "human_all_long_2026-01-15.tsv.gz", "human_all_long_2026-09-20.tsv.gz")
    date, paths = step2.snapshot_paths(str(tmp_path), "2026-01-15")
    assert (date, [os.path.basename(p) for p in paths]) == ("2026-01-15", ["human_all_long_2026-01-15.tsv.gz"])


def test_snapshot_paths_ignores_files_without_a_date(step2, tmp_path):
    """'human_all_long_v2.tsv.gz' does not follow Metalog's naming: never read."""
    touch(tmp_path, "human_all_long_v2.tsv.gz", "human_all_long_2026-09-20.tsv.gz")
    assert [os.path.basename(p) for p in step2.snapshot_paths(str(tmp_path))[1]] == ["human_all_long_2026-09-20.tsv.gz"]


def test_snapshot_paths_unknown_date_stops_and_lists_the_dates(step2, tmp_path):
    touch(tmp_path, "human_all_long_2026-01-15.tsv.gz", "human_all_long_2026-09-20.tsv.gz")
    with pytest.raises(SystemExit, match="snapshots there: 2026-01-15, 2026-09-20"):
        step2.snapshot_paths(str(tmp_path), "2025-12-31")


# ----------------------------------------------------------------------------- failures on unusable input
# These document how the function fails, so that a change of behaviour is noticed.

def test_empty_folder_stops_with_a_clear_message(step2, tmp_path):
    """No *_all_long_<date>.tsv.gz file (e.g. a wrong --metalog_dir)."""
    with pytest.raises(SystemExit, match="no \\*_all_long_<YYYY-MM-DD>.tsv.gz files in"):
        step2.load_metalog_labels(str(tmp_path), VALID)


def test_slot_missing_from_every_file_raises(step2, tmp_path):
    """If no file has a single environment_feature row, the column does not exist and the slot loop fails
    (rather than silently writing an empty slot)."""
    files = {"human_all_long_2026-09-20.tsv.gz": [
        ("h1", "spire_sample_name",    "SAMEA0000001"),
        ("h1", "environment_biome",    "marine biome [ENVO:00000447]"),
        ("h1", "environment_material", "feces [UBERON:0001988]"),
    ]}
    with pytest.raises(KeyError, match="environment_feature"):
        labels_of(step2, tmp_path, files, VALID)
