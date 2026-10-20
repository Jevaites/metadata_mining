"""
Integration tests for 2_build_training_set.py: main() run end to end on small mock inputs, exactly as

    python 2_build_training_set.py --metalog_dir DIR --sample_info FILE --ontology_terms FILE --output FILE

What step 2 does, in three moves (each visible in test_basic_workflow):
  1. Metalog long tables  -> one label row per Metalog accession (spire_sample_name), term ids in the
                             biome / feature / material slots; obsolete or unknown terms blanked.
  2. MicrobeAtlas records -> each record is linked to a Metalog accession: its own id if Metalog knows it,
                             otherwise the first accession written inside the record that Metalog knows.
                             Records that link to nothing are skipped.
  3. One output row per linked record: its Metalog labels + `text`, the record cleaned into
                             'key: value; key: value' (common.record_to_text).

The mock world below (4 Metalog samples, 7 ontology terms) is shared by all tests but
test_real_sample_info_records, which runs on three shortened real records. Each test writes only the
MicrobeAtlas records it needs, so the expected output can be read off directly. Mock records name their
BioSample as SRA records do: 'experiment_external_id_BioSample=...' (an 'experiment...' key, so it links but
is dropped from the text).

test_2_build_training_set_mutations.py shows which of these tests fail for which plausible bug.
"""

from step2_helpers import run_step2, step2  # noqa: F401  (pytest fixtures)

# ----------------------------------------------------------------------------- the mock world
# Term table (output of step 1): term_id, label, obsolete
TERMS = [
    ("ENVO_00000446",  "terrestrial biome",        False),
    ("ENVO_00000114",  "agricultural field",       False),
    ("ENVO_00001998",  "soil",                     False),
    ("ENVO_00000447",  "marine biome",             False),
    ("ENVO_00002149",  "sea water",                False),
    ("UBERON_0001988", "feces",                    False),
    ("ENVO_00009003",  "human-associated habitat", True),   # obsolete: not a valid label
]


def metalog_sample(alias, spire, study, biome="", feature="", material=""):
    """The rows of one sample in a Metalog long table (sample_alias, metadata_item, value)."""
    return [(alias, "spire_sample_name", spire), (alias, "study_code", study),
            (alias, "environment_biome", biome), (alias, "environment_feature", feature),
            (alias, "environment_material", material)]


# Metalog: two domain files, four samples.
#   accession     study       domain         biome                    feature              material
#   SAMEA0000001  Shao_2019   human          (obsolete -> "")         -                    feces
#   SRS0000002    Smith_2020  environmental  terrestrial biome        agricultural field   soil
#   ERS0000003    Ocean_2021  environmental  marine biome             -                    sea water
#   SRS0000004    Lake_2022   environmental  -                        -                    -      (never linked below)
METALOG = {
    "human_all_long_2026-09-20.tsv.gz": [
        *metalog_sample("infant_01", "SAMEA0000001", "Shao_2019",
                        biome="human-associated habitat [ENVO:00009003]", material="feces [UBERON:0001988]"),
    ],
    "environmental_all_long_2026-09-20.tsv.gz": [
        *metalog_sample("field_01", "SRS0000002", "Smith_2020", biome="terrestrial biome [ENVO:00000446]",
                        feature="agricultural field [ENVO:00000114]", material="soil [ENVO:00001998]"),
        *metalog_sample("sea_01", "ERS0000003", "Ocean_2021", biome="marine biome [ENVO:00000447]",
                        material="sea water [ENVO:00002149]"),
        *metalog_sample("lake_01", "SRS0000004", "Lake_2022"),
    ],
}

COLUMNS = ["sample_id", "spire_sample_name", "study_code", "domain", "biome", "feature", "material", "text"]


def linked(out):
    """{MicrobeAtlas sample_id: Metalog accession it was linked to}: the result of the linking, at a glance."""
    return dict(zip(out["sample_id"], out["spire_sample_name"]))


# ============================================================================= basic workflow
def test_basic_workflow(run_step2):
    atlas = {
        # 1. the record's own id is a Metalog accession -> linked directly
        "SRS0000002": ["sample_env_material=soil [ENVO:00001998]",  # code in the text -> replaced by its label
                       "sample_collection_date=2019-05-01",         # date key -> dropped from the text
                       "study_STUDY_TITLE=Soil of a wheat field"],  # 'study_' prefix -> dropped, key lower-cased
        # 2. own id unknown to Metalog, but the record names its BioSample -> linked through it
        "SRS9000001": ["experiment_external_id_BioSample=SAMEA0000001",  # identifier key -> dropped from the text
                       "sample_host=Homo sapiens",
                       "sample_host_age=missing"],                  # missing value -> dropped from the text
        # 3. direct link again (an ENA id)
        "ERS0000003": ["sample_env_biome=marine biome"],
        # 4. nothing Metalog knows -> skipped
        "SRS9000002": ["sample_env_material=freshwater"],
    }
    out, log = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)

    assert list(out.columns) == COLUMNS
    assert out.to_dict("records") == [
        {"sample_id": "SRS0000002", "spire_sample_name": "SRS0000002", "study_code": "Smith_2020",
         "domain": "environmental",
         "biome": "ENVO_00000446", "feature": "ENVO_00000114", "material": "ENVO_00001998",
         "text": "env_material: soil [soil]; study_title: Soil of a wheat field"},

        {"sample_id": "SRS9000001", "spire_sample_name": "SAMEA0000001", "study_code": "Shao_2019",
         "domain": "human",
         "biome": "",  # Metalog's human biome ENVO:00009003 is obsolete -> blanked
         "feature": "", "material": "UBERON_0001988",
         "text": "host: Homo sapiens"},

        {"sample_id": "ERS0000003", "spire_sample_name": "ERS0000003", "study_code": "Ocean_2021",
         "domain": "environmental",
         "biome": "ENVO_00000447", "feature": "", "material": "ENVO_00002149",
         "text": "env_biome: marine biome"},
    ]
    # SRS9000002 (no link) and SRS0000004 (Metalog sample without a MicrobeAtlas record) are absent
    assert "4 MicrobeAtlas records, 3 linked to 3 Metalog samples" in log


# ============================================================================= linking edge cases
def test_own_id_wins_over_accessions_inside_the_record(run_step2):
    """SRS0000002 is itself a Metalog accession and also names another one (SAMEA0000001):
    the record's own id is tried first."""
    atlas = {"SRS0000002": ["experiment_external_id_BioSample=SAMEA0000001"]}
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert linked(out) == {"SRS0000002": "SRS0000002"}


def test_first_known_accession_inside_the_record_wins(run_step2):
    """Own id unknown; the record names three accessions. SRS7777777 is unknown to Metalog and skipped;
    ERS0000003 comes before SAMEA0000001 in the record, so it is the link."""
    atlas = {"SRS9000001": ["sample_secondary=SRS7777777",
                            "sample_ena=ERS0000003",
                            "experiment_external_id_BioSample=SAMEA0000001"]}
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert linked(out) == {"SRS9000001": "ERS0000003"}


def test_two_records_linked_to_the_same_metalog_sample_give_two_rows(run_step2):
    """Rows are per MicrobeAtlas record, not per Metalog sample: both rows carry the same labels."""
    atlas = {"SRS9000001": ["experiment_external_id_BioSample=SAMEA0000001"],
             "SRS9000002": ["experiment_external_id_BioSample=SAMEA0000001"]}
    out, log = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert linked(out) == {"SRS9000001": "SAMEA0000001", "SRS9000002": "SAMEA0000001"}
    assert out["material"].tolist() == ["UBERON_0001988", "UBERON_0001988"]
    assert "2 MicrobeAtlas records, 2 linked to 1 Metalog samples" in log


def test_accessions_are_only_found_as_whole_upper_case_words(run_step2):
    """'SAMEA0000001_rep1' (glued to '_rep1') and 'samea0000001' (lower case) are not read as accessions."""
    atlas = {"SRS9000001": ["sample_alias=SAMEA0000001_rep1"],
             "SRS9000002": ["sample_note=samea0000001"],
             "SRS9000003": ["sample_note=from SAMEA0000001."]}  # punctuation is a word boundary: found
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert linked(out) == {"SRS9000003": "SAMEA0000001"}


def test_linked_records_are_kept_even_without_usable_labels_or_text(run_step2):
    """Linking does not depend on the labels: SRS0000004 has no label in Metalog and its record has no
    metadata line at all, yet it gets a row (with "" everywhere). Filtering is left to step 2b."""
    atlas = {"SRS0000004": []}
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert out.to_dict("records") == [
        {"sample_id": "SRS0000004", "spire_sample_name": "SRS0000004", "study_code": "Lake_2022",
         "domain": "environmental", "biome": "", "feature": "", "material": "", "text": ""}]


# ============================================================================= text edge cases
def test_obsolete_term_is_no_label_but_still_named_in_the_text(run_step2):
    """The term table plays two roles. As LABELS only non-obsolete terms count (the biome below is blanked);
    as TEXT every code is replaced by its label, obsolete or not (it still describes the sample)."""
    atlas = {"SRS9000001": ["experiment_external_id_BioSample=SAMEA0000001",
                            "sample_env_biome=ENVO:00009003",     # obsolete, colon form
                            "sample_env_feature=ENVO_99999999"]}  # not in the term table: left as is
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert out.loc[0, "biome"] == ""
    assert out.loc[0, "text"] == "env_biome: human-associated habitat; env_feature: ENVO_99999999"


def test_text_is_truncated_to_max_chars(run_step2):
    atlas = {"ERS0000003": ["sample_env_biome=marine biome"]}  # full text: 'env_biome: marine biome' (23 chars)
    out, _ = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS, max_chars=9)
    assert out.loc[0, "text"] == "env_biome"


# ============================================================================= real record formats
# Three records of the real sample.info.gz, shortened, in its layout: an empty first line, a blank line
# between records. An SRA record names its BioSample in experiment_external_id_BioSample; an ENA record
# in sample_alias (and repeats its own ERS id in several fields). Run, experiment, study and BioProject ids
# (SRR / SRX / SRP / ERR / ERX / ERP / PRJNA) are written there too and must not be taken for samples.
REAL_SAMPLE_INFO = """
>SRS1337475
sample_center_name=
sample_alias=PNUSAE002469
sample_SCIENTIFIC_NAME=Escherichia coli
sample_collection_date=Aug-2013
sample_host=Missing
sample_isolation_source=stool
experiments=SRX1629903
study=SRP046387
study_STUDY_TITLE=PulseNet Escherichia coli and Shigella genome sequencing
study_XREF_LINK=pubmed: 25103754)
experiment=SRX1629903
experiment_external_id_BioProject=PRJNA218110
experiment_external_id_BioSample=SAMN04544282
runs=SRR3223145

>ERS132372
sample_center_name=
sample_alias=SAMEA1316539
sample_SCIENTIFIC_NAME=Streptococcus suis
sample_INSDC status=public
sample_SRA accession=ERS132372
sample_Sample Name=ERS132372
sample_specific_host=Pig
experiments=ERX096402
study=ERP000428
study_STUDY_TITLE=Streptococcus suis genome comparisons
experiment_DESIGN_DESCRIPTION=Illumina sequencing of library 3804130, constructed from sample accession ERS132372 for study accession ERP000428.
runs=ERR120130

>SRS7471636
sample_alias=AOA_enrichment_SS
sample_isolation_source=ammonia-oxidizing archaea enrichment inoculated with soil
experiments=SRX9238413
study=SRP286249
"""


def test_real_sample_info_records(run_step2):
    """The SRA record links through its SAMN BioSample, the ENA record through the SAMEA in sample_alias;
    the third names no Metalog accession. Identifier lines and 'Missing' values are left out of the text."""
    metalog = {"human_all_long_2026-09-20.tsv.gz": metalog_sample(
                   "PulseNet.S1", "SAMN04544282", "PulseNet_2016", material="feces [UBERON:0001988]"),
               "animal_all_long_2026-09-20.tsv.gz": metalog_sample(
                   "Suis.S1", "SAMEA1316539", "Suis_2012", material="feces [UBERON:0001988]")}
    out, log = run_step2(metalog=metalog, atlas=REAL_SAMPLE_INFO, terms=TERMS, sample_info_name="sample.info.gz")
    assert out[["sample_id", "spire_sample_name", "domain", "text"]].values.tolist() == [
        ["SRS1337475", "SAMN04544282", "human",
         "scientific_name: Escherichia coli; isolation_source: stool; "
         "study_title: PulseNet Escherichia coli and Shigella genome sequencing"],
        ["ERS132372", "SAMEA1316539", "animal",
         "scientific_name: Streptococcus suis; specific_host: Pig; study_title: Streptococcus suis genome comparisons"],
    ]
    assert "3 MicrobeAtlas records, 2 linked to 2 Metalog samples" in log


# ============================================================================= Metalog snapshots
# Two snapshots of the same Metalog sample in one folder: the old one says feces, the new one soil.
TWO_SNAPSHOTS = {
    "human_all_long_2026-01-15.tsv.gz": metalog_sample("infant_01", "SAMEA0000001", "Old_study_code",
                                                       material="feces [UBERON:0001988]"),
    "human_all_long_2026-09-20.tsv.gz": metalog_sample("infant_01", "SAMEA0000001", "New_study_code",
                                                       material="soil [ENVO:00001998]"),
}
INFANT_RECORD = {"SRS9000001": ["experiment_external_id_BioSample=SAMEA0000001"]}


def test_latest_snapshot_is_used_by_default(run_step2):
    """Only the latest snapshot is read (as in step 2b). Fixed bug: all snapshots used to be read and mixed,
    and the OLDER file won because it sorts first."""
    out, log = run_step2(metalog=TWO_SNAPSHOTS, atlas=INFANT_RECORD, terms=TERMS)
    assert (out.loc[0, "study_code"], out.loc[0, "material"]) == ("New_study_code", "ENVO_00001998")
    assert "Metalog snapshot 2026-09-20" in log


def test_date_selects_an_older_snapshot(run_step2):
    out, log = run_step2(metalog=TWO_SNAPSHOTS, atlas=INFANT_RECORD, terms=TERMS, date="2026-01-15")
    assert (out.loc[0, "study_code"], out.loc[0, "material"]) == ("Old_study_code", "UBERON_0001988")
    assert "Metalog snapshot 2026-01-15" in log


# ============================================================================= nothing linked
def test_no_record_linked_writes_an_empty_table(run_step2):
    """E.g. sample.info and Metalog from unrelated sources: an empty table with the usual header, and a log
    saying 0 records were linked. Fixed bug: the DataFrame had no columns, so the file had no header and
    the summary print crashed with KeyError: 'spire_sample_name'."""
    atlas = {"SRS9000001": ["sample_env_material=freshwater"]}
    out, log = run_step2(metalog=METALOG, atlas=atlas, terms=TERMS)
    assert list(out.columns) == COLUMNS and len(out) == 0
    assert "1 MicrobeAtlas records, 0 linked to 0 Metalog samples" in log
