"""Unit tests of 2b_clean_metalog.py, one section per function, in the order the script defines them.

Each test builds the smallest input that shows one behaviour, so the expected value can be read off
the input. The full workflow (main) is in test_2b_clean_metalog_integration.py.

    pytest scripts/ontology_mapping/tests/test_2b_clean_metalog_unit.py -v
"""

import os

import pandas as pd
import pytest

from step2b_helpers import CORE_COLUMNS, load_step, read_tsv, samples_frame, terms_table, write_long, write_tsv

clean = load_step("2b_clean_metalog")
# pandas warns about the capture groups of the AUDIT regexes; the script silences it, pytest re-enables it.
pytestmark = pytest.mark.filterwarnings("ignore:This pattern is interpreted as a regular expression")


# ============================================================================= audited_fields
# Input: a step-2 text 'key: value; key: value', sample fields first, then study fields.
# Output: only the sample fields whose key matches AUDIT_KEYS (free text that describes the sample).

def test_audited_fields_keeps_descriptive_keys_only():
    text = "title: river water sample; geo_loc_name: USA; isolation_source: river water; host_age: 3"
    assert clean.audited_fields(text) == "title: river water sample; isolation_source: river water"


def test_audited_fields_cuts_everything_from_the_study_part_on():
    # 'notes' would be audited, but it comes after study_title: it belongs to the study part.
    text = "isolation_source: river water; study_title: Controls of DOM addition; notes: control; abstract: mesocosm"
    assert clean.audited_fields(text) == "isolation_source: river water"


@pytest.mark.parametrize("study_key", ["study_title", "study_abstract", "study_description", "abstract"])
def test_audited_fields_recognises_every_study_key(study_key):
    assert clean.audited_fields(f"isolation_source: soil; {study_key}: negative control") == "isolation_source: soil"


def test_audited_fields_text_that_starts_with_the_study_part_gives_nothing():
    assert clean.audited_fields("study_title: Gut study; title: stool") == ""


def test_audited_fields_key_match_is_case_insensitive_and_accepts_variants():
    text = "Sample Type: stool; ENV_MEDIUM: feces; environment (biome): gut; host_sex: female"
    assert clean.audited_fields(text) == "Sample Type: stool; ENV_MEDIUM: feces; environment (biome): gut"


def test_audited_fields_drops_empty_values_and_fragments_of_values_that_contain_semicolons():
    # Pairs are split on "; ", so the value 'soil; pot 3' is cut: the fragment 'pot 3' has no key and is lost.
    assert clean.audited_fields("title: ; description: soil; pot 3; notes: dry") == "description: soil; notes: dry"


# ============================================================================= snapshot_files
def touch_snapshot(d, date, domains=clean.DOMAINS):
    for domain in domains:
        for kind in ("core_wide", "all_long"):
            open(os.path.join(d, f"{domain}_{kind}_{date}.tsv.gz"), "w").close()


def test_snapshot_files_defaults_to_the_latest_date(tmp_path):
    touch_snapshot(tmp_path, "2026-01-01")
    touch_snapshot(tmp_path, "2026-09-20")
    date, files = clean.snapshot_files(str(tmp_path), None)
    assert date == "2026-09-20"
    assert sorted(files) == sorted(clean.DOMAINS)
    assert files["human"] == (str(tmp_path / "human_core_wide_2026-09-20.tsv.gz"),
                              str(tmp_path / "human_all_long_2026-09-20.tsv.gz"))


def test_snapshot_files_explicit_date(tmp_path):
    touch_snapshot(tmp_path, "2026-01-01")
    touch_snapshot(tmp_path, "2026-09-20")
    assert clean.snapshot_files(str(tmp_path), "2026-01-01")[0] == "2026-01-01"


def test_snapshot_files_without_any_snapshot_exits(tmp_path):
    with pytest.raises(SystemExit, match="no \\*_core_wide"):
        clean.snapshot_files(str(tmp_path), None)


def test_snapshot_files_incomplete_latest_snapshot_exits_instead_of_falling_back(tmp_path):
    touch_snapshot(tmp_path, "2026-01-01")                       # complete
    touch_snapshot(tmp_path, "2026-09-20", domains=["human"])    # latest, but only one domain
    with pytest.raises(SystemExit, match="missing for snapshot 2026-09-20"):
        clean.snapshot_files(str(tmp_path), None)


# ============================================================================= load_core
def test_load_core_stacks_domains_and_fills_missing_columns(tmp_path):
    human = write_tsv(tmp_path / "h.tsv.gz", [
        {"sample_alias": "Gut.S1", "study_code": "Gut", "spire_sample_name": "SAMN01", "artificial": "",
         "environment_biome": "b", "environment_feature": "f", "environment_material": "m", "extra_column": "x"}])
    ocean = write_tsv(tmp_path / "o.tsv.gz", [  # no `artificial` column in this file
        {"sample_alias": "Sea.W1", "study_code": "Sea", "spire_sample_name": "SAMN20",
         "environment_biome": "b2", "environment_feature": "f2", "environment_material": "m2"}])
    core = clean.load_core({"human": (human, None), "ocean": (ocean, None)})

    assert list(core.columns) == ["domain", *CORE_COLUMNS]          # extra_column is dropped
    assert core["domain"].tolist() == ["human", "ocean"]
    assert core["sample_alias"].tolist() == ["Gut.S1", "Sea.W1"]
    assert core["artificial"].tolist() == ["", ""]                    # missing column -> ""


# ============================================================================= load_audit_text
def test_load_audit_text_is_alias_plus_audited_long_rows(tmp_path):
    long = write_long(tmp_path / "long.tsv.gz", [
        ("Gut.S1", "extended", "sample_title", "stool of subject 7"),   # kept
        ("Gut.S1", "all", "isolation_source", "feces"),                 # kept (tier 'all')
        ("Gut.S1", "core", "sample_description", "core tier"),          # dropped: core tier
        ("Gut.S1", "all", "atopy", "control"),                          # dropped: questionnaire key, not audited
        ("Gut.S1", "all", "study_abstract", "negative controls"),       # dropped: study-level key
        ("Gut.S1", "all", "notes", ""),                                 # dropped: empty value
        ("Other.X", "all", "notes", "not in samples"),                  # ignored: not one of `samples`
    ])
    samples = samples_frame([("Gut.S1", "Gut", "SAMN01", "", "", "", ""),
                             ("Gut.S2", "Gut", "SAMN02", "", "", "", "")])
    text = clean.load_audit_text({"human": (None, long)}, samples)

    assert text.tolist() == ["Gut.S1; sample_title: stool of subject 7; isolation_source: feces",
                             "Gut.S2; "]  # no long rows: the alias alone (it can hold 'BLANK', 'NTC', ...)


# ============================================================================= load_overrides
def test_load_overrides_without_file_is_empty():
    df = clean.load_overrides(None)
    assert len(df) == 0 and list(df.columns) == ["scope", "id", "pattern", "decision", "reason"]


def test_load_overrides_normalises_decisions_and_drops_undecided_rows(tmp_path):
    p = write_tsv(tmp_path / "o.tsv", [
        {"scope": "study", "id": "Gut", "pattern": "ntc", "decision": " Control "},
        {"scope": "study", "id": "Sea", "pattern": "", "decision": ""},       # not reviewed yet
        {"scope": "sample", "id": "Soil.P2", "pattern": "", "decision": "OK"}])
    df = clean.load_overrides(p)
    assert df["id"].tolist() == ["Gut", "Soil.P2"]
    assert df["decision"].tolist() == ["control", "ok"]
    assert df["reason"].tolist() == ["", ""]  # added when the file has no reason column


def test_load_overrides_rejects_unknown_decisions(tmp_path):
    p = write_tsv(tmp_path / "o.tsv", [{"scope": "study", "id": "Gut", "pattern": "", "decision": "maybe"}])
    with pytest.raises(SystemExit, match="unknown decisions \\['maybe'\\]"):
        clean.load_overrides(p)


def test_load_overrides_rejects_files_without_the_needed_columns(tmp_path):
    p = write_tsv(tmp_path / "o.tsv", [{"study": "Gut", "decision": "control"}])
    with pytest.raises(SystemExit, match="needs columns"):
        clean.load_overrides(p)


# ============================================================================= load_label_map
def test_load_label_map(tmp_path):
    p = write_tsv(tmp_path / "map.tsv", [
        {"slot": "biome", "from_id": "ENVO_00009003", "to_id": "ENVO_01001002", "reason": "obsolete"},
        {"slot": "*", "from_id": " ENVO_00000428 ", "to_id": "", "reason": "uninformative: blank it"}])
    assert clean.load_label_map(p) == {("biome", "ENVO_00009003"): "ENVO_01001002",
                                       ("*", "ENVO_00000428"): ""}
    assert clean.load_label_map(None) == {}


def test_load_label_map_study_rows_get_a_three_part_key(tmp_path):
    p = write_tsv(tmp_path / "map.tsv", [
        {"slot": "biome", "from_id": "ENVO_00002030", "to_id": "ENVO_01000252", "reason": "", "study_code": ""},
        {"slot": "biome", "from_id": "ENVO_00002030", "to_id": "ENVO_00000447", "reason": "", "study_code": " S2 "}])
    assert clean.load_label_map(p) == {("biome", "ENVO_00002030"): "ENVO_01000252",
                                       ("biome", "ENVO_00002030", "S2"): "ENVO_00000447"}


# ============================================================================= flag_labels
def flag(biome="", feature="", material="", label_map=None):
    """Run flag_labels on one sample with the toy ontology; return the sample's row as a dict."""
    samples = samples_frame([("A", "S", "SAMN1", "", biome, feature, material)])
    clean.flag_labels(samples, terms_table(), label_map or {})
    return samples.iloc[0].to_dict()


@pytest.mark.parametrize("raw_value, raw_id, status, clean_id", [
    # usable: an id of a kept ontology (ENVO, UBERON, PO, FOODON) that is in the term index
    ("fecal material [ENVO:00002003]",                      "ENVO_00002003",  "ok",             "ENVO_00002003"),
    ("feces [UBERON:0001988]",                              "UBERON_0001988", "ok",             "UBERON_0001988"),
    ("leaf [PO:0025034]",                                   "PO_0025034",     "ok",             "PO_0025034"),
    ("fermented dairy food product [FOODON:00001053]",      "FOODON_00001053", "ok",            "FOODON_00001053"),
    ("soil [envo:00001998]",                                "ENVO_00001998",  "ok",             "ENVO_00001998"),   # prefix case
    ("soil [ENVO:00001998] and sediment [ENVO:00002007]",   "ENVO_00001998",  "ok",             "ENVO_00001998"),   # first code wins
    # unusable: the raw id is kept in <slot>_raw, the clean id is empty
    ("obsolete human-associated habitat [ENVO:00009003]",   "ENVO_00009003",  "obsolete",       ""),
    ("made-up term [ENVO:99999999]",                        "ENVO_99999999",  "not_in_index",   ""),
    ("ciliated cell [CL:0000064]",                          "CL_0000064",     "other_ontology", ""),
    ("fecal material",                                      "",               "no_code",        ""),
    ("Negative Control",                                    "",               "control_value",  ""),
    ("blank",                                               "",               "control_value",  ""),
    ("",                                                    "",               "empty",          ""),
    ("   ",                                                 "",               "empty",          ""),
])
def test_flag_labels_status(raw_value, raw_id, status, clean_id):
    row = flag(material=raw_value)
    assert (row["material_raw"], row["material_status"], row["material_clean"]) == (raw_id, status, clean_id)


def test_flag_labels_flags_all_three_slots_and_their_ontology():
    row = flag(biome="marine pelagic biome [ENVO:01000023]", feature="", material="feces [UBERON:0001988]")
    assert (row["biome_clean"], row["feature_clean"], row["material_clean"]) == ("ENVO_01000023", "", "UBERON_0001988")
    assert (row["biome_ontology"], row["feature_ontology"], row["material_ontology"]) == ("ENVO", "", "UBERON")
    assert row["feature_status"] == "empty"


@pytest.mark.parametrize("biome, expected", [
    ("biome [ENVO:00000428]",                 "True"),   # the root itself
    ("marine pelagic biome [ENVO:01000023]",  "True"),   # two is_a steps below it
    ("animal-associated environment [ENVO:01001002]", "False"),  # not under 'biome'
    ("obsolete human-associated habitat [ENVO:00009003]", ""),   # no clean biome
    ("",                                      ""),
])
def test_flag_labels_biome_in_biome_subtree(biome, expected):
    assert flag(biome=biome)["biome_in_biome_subtree"] == expected


# The label map: {(slot, from_id): to_id}, slot '*' = every slot. It is looked up for every raw id,
# whatever its status, so it can also rescue obsolete or other-ontology ids.
def test_label_map_replaces_a_usable_id():
    row = flag(biome="biome [ENVO:00000428]", label_map={("biome", "ENVO_00000428"): "ENVO_00000447"})
    assert (row["biome_status"], row["biome_clean"]) == ("ok", "ENVO_00000447")


def test_label_map_empty_target_blanks_the_label_on_purpose():
    row = flag(biome="biome [ENVO:00000428]", label_map={("biome", "ENVO_00000428"): ""})
    assert (row["biome_status"], row["biome_clean"], row["biome_in_biome_subtree"]) == ("ok", "", "")


def test_label_map_rescues_an_obsolete_id():
    row = flag(biome="obsolete human-associated habitat [ENVO:00009003]",
               label_map={("biome", "ENVO_00009003"): "ENVO_01001002"})
    assert (row["biome_status"], row["biome_clean"]) == ("obsolete", "ENVO_01001002")  # status stays 'obsolete'


def test_label_map_slot_specific_entry_wins_over_star_and_only_applies_to_its_slot():
    label_map = {("material", "ENVO_00001998"): "ENVO_00002007",   # material soil -> sediment
                 ("*", "ENVO_00001998"): ""}                         # soil anywhere else -> blank
    row = flag(feature="soil [ENVO:00001998]", material="soil [ENVO:00001998]", label_map=label_map)
    assert row["material_clean"] == "ENVO_00002007"
    assert row["feature_clean"] == ""


def test_label_map_target_outside_the_index_is_used_but_warned_about(capsys):
    row = flag(material="soil [ENVO:00001998]", label_map={("material", "ENVO_00001998"): "ENVO_12345678"})
    assert row["material_clean"] == "ENVO_12345678"
    assert "WARNING: label map targets not valid in the term index: {'ENVO_12345678': 1}" in capsys.readouterr().out


def test_label_map_study_row_applies_to_its_study_only_and_wins_over_the_plain_row():
    label_map = {("biome", "ENVO_00000428"): "ENVO_00000447",          # every study
                 ("biome", "ENVO_00000428", "S"): "ENVO_01000023"}     # study S (the toy sample's study)
    assert flag(biome="biome [ENVO:00000428]", label_map=label_map)["biome_clean"] == "ENVO_01000023"
    other = {("biome", "ENVO_00000428"): "ENVO_00000447", ("biome", "ENVO_00000428", "T"): "ENVO_01000023"}
    assert flag(biome="biome [ENVO:00000428]", label_map=other)["biome_clean"] == "ENVO_00000447"


# ============================================================================= flag_duplicates
def duplicates(rows):
    """rows: (alias, accession, material) -> {alias: drop reason}. The other labels are equal everywhere."""
    samples = samples_frame([(a, "S", acc, "", "biome", "feature", m) for a, acc, m in rows])
    return dict(zip(samples["sample_alias"], clean.flag_duplicates(samples)))


def test_flag_duplicates_missing_accession():
    assert duplicates([("A", "", "soil"), ("B", "NA", "soil"), ("C", " none ", "soil"), ("D", "SAMN1", "soil")]) == \
        {"A": "no_accession", "B": "no_accession", "C": "no_accession", "D": ""}


def test_flag_duplicates_same_accession_same_labels_keeps_the_first_alias():
    assert duplicates([("A", "SAMN1", "soil"), ("B", "SAMN1", "soil"), ("C", "SAMN1", "soil")]) == \
        {"A": "", "B": "duplicate_alias", "C": "duplicate_alias"}


def test_flag_duplicates_same_accession_different_labels_drops_all_copies():
    # A and B agree, C disagrees: we cannot tell which label is right, so all three go.
    assert duplicates([("A", "SAMN1", "soil"), ("B", "SAMN1", "soil"), ("C", "SAMN1", "sediment"),
                       ("D", "SAMN2", "soil")]) == \
        {"A": "conflicting_duplicate", "B": "conflicting_duplicate", "C": "conflicting_duplicate", "D": ""}


def test_flag_duplicates_compares_raw_label_strings():
    # Same id, different wording of the raw value: counted as a conflict (labels are compared as raw text).
    assert duplicates([("A", "SAMN1", "soil [ENVO:00001998]"), ("B", "SAMN1", "Soil [ENVO:00001998]")]) == \
        {"A": "conflicting_duplicate", "B": "conflicting_duplicate"}


def test_flag_duplicates_when_no_sample_has_an_accession():
    # regression: this used to raise TypeError (agg on the empty frame of samples with an accession)
    assert duplicates([("A", "", "soil"), ("B", "NA", "soil")]) == {"A": "no_accession", "B": "no_accession"}


# ============================================================================= audit
def hits(text):
    samples = pd.DataFrame(index=[0])
    clean.audit(samples, pd.Series([text]))
    return samples.loc[0, "audit_hits"]


@pytest.mark.parametrize("text, expected", [
    # one example per category
    ("Gut.S2; sample_title: negative control",        "negative_control"),
    ("Gut.S2; sample_title: extraction blank",        "negative_control"),
    ("Gut.S2; sample_title: ZymoBIOMICS standard",    "mock_or_spike_in"),
    ("Soil.P4; sample_type: enrichment culture",      "cultivation"),
    ("Soil.P4; treatment: incubated 7 days",          "cultivation"),
    ("Sea.M1; description: mesocosm tank 3",          "mesocosm"),
    ("Sea.F1; description: FACS sorted cells",        "sorted_cells"),
    ("Gut.V1; description: VLP fraction",             "virome_enrichment"),
    ("Soil.P2; description: ancient permafrost",      "ancient_or_post_mortem"),
    ("Bone.B1; description: herbarium specimen",      "ancient_or_post_mortem"),
    # '_' and '.' become spaces first, so ids in aliases are seen as words
    ("Gut.S2_NTC",                                    "negative_control"),
    ("Wortelboer.D002_M009_VLP",                      "virome_enrichment"),
    # near misses that must NOT hit
    ("Soil.X; description: agriculture soil",         ""),   # 'cultur' inside a word
    ("Soil.X; description: DNA enrichment kit",       ""),   # enrichment kit is a lab product
    ("Gut.X; description: in vitro fertilization",    ""),
    ("Gut.X; description: paleodiet cohort",          ""),
    ("Gut.X; description: mockingbird gut",           ""),
    ("Gut.X; description: stool",                     ""),
    # a known false positive (the audit only makes suggestions for review)
    ("Spring.X; description: Blank Spring hot spring", "negative_control"),
])
def test_audit_categories(text, expected):
    assert hits(text) == expected


def test_audit_several_categories_are_joined_in_audit_order():
    assert hits("description: mesocosm with enrichment culture") == "cultivation;mesocosm"


def test_audit_returns_the_normalised_text():
    samples = pd.DataFrame(index=[0])
    assert clean.audit(samples, pd.Series(["Gut.S2_NTC__1"])).tolist() == ["Gut S2 NTC 1"]


# ============================================================================= apply_overrides
def override_rows(*rows):
    return pd.DataFrame(rows, columns=["scope", "id", "pattern", "decision", "reason"])


def run_overrides(samples, overrides, text=None):
    """text: the normalised audit text per sample (default: the alias with '_' / '.' as spaces)."""
    norm = pd.Series(text) if text is not None else samples["sample_alias"].str.replace(r"[_.]+", " ", regex=True)
    n = clean.apply_overrides(samples, norm, overrides)
    return n, dict(zip(samples["sample_alias"], zip(samples["artificial_bucket"], samples["artificial_source"],
                                                    samples["reviewed"])))


@pytest.mark.parametrize("artificial, bucket", [
    ("",                      "none"),
    ("negative control",      "control"),
    ("mock",                  "control"),
    ("marked as contaminated", "control"),
    (" Mesocosm ",            "perturbed"),   # case and whitespace do not matter
    ("sorted cells",          "perturbed"),
    ("paleosample",           "degraded"),
    ("museum specimen",       "degraded"),
])
def test_metalog_artificial_value_to_bucket(artificial, bucket):
    samples = samples_frame([("A", "S", "SAMN1", artificial, "", "", "")])
    _, result = run_overrides(samples, override_rows())
    assert result["A"] == (bucket, "metalog" if artificial.strip() else "", False)


def test_unknown_metalog_artificial_value_is_perturbed_with_a_warning(capsys):
    samples = samples_frame([("A", "S", "SAMN1", "freeze-dried", "", "", "")])
    _, result = run_overrides(samples, override_rows())
    assert result["A"][0] == "perturbed"
    assert "unknown Metalog `artificial` values treated as perturbed: ['freeze-dried']" in capsys.readouterr().out


def test_study_override_with_pattern_only_changes_the_matching_samples_of_that_study():
    samples = samples_frame([("Gut.S1", "Gut", "1", "", "", "", ""),
                             ("Gut.S2_NTC", "Gut", "2", "", "", "", ""),
                             ("Sea.W1_NTC", "Sea", "3", "", "", "", "")])   # matches, but other study
    n, result = run_overrides(samples, override_rows(("study", "Gut", r"\bntc\b", "control", "")))
    assert result == {"Gut.S1": ("none", "", False),
                      "Gut.S2_NTC": ("control", "override", True),
                      "Sea.W1_NTC": ("none", "", False)}
    assert n == 1


def test_study_override_without_pattern_changes_the_whole_study():
    samples = samples_frame([("Gut.S1", "Gut", "1", "", "", "", ""), ("Gut.S2", "Gut", "2", "", "", "", "")])
    n, result = run_overrides(samples, override_rows(("study", "Gut", "", "degraded", "")))
    assert [r[0] for r in result.values()] == ["degraded", "degraded"] and n == 2


def test_ok_decision_clears_a_metalog_flag():
    samples = samples_frame([("Soil.P1", "Soil", "1", "paleosample", "", "", "")])
    _, result = run_overrides(samples, override_rows(("sample", "Soil.P1", "", "ok", "modern soil, Metalog mistake")))
    assert result["Soil.P1"] == ("none", "override", True)


def test_sample_decision_wins_over_its_study_decision_whatever_the_row_order():
    samples = samples_frame([("Gut.S1", "Gut", "1", "", "", "", ""), ("Gut.S2", "Gut", "2", "", "", "", "")])
    overrides = override_rows(("sample", "Gut.S2", "", "ok", ""),            # listed first ...
                              ("study", "Gut", "", "control", ""))           # ... but applied after this one
    n, result = run_overrides(samples, overrides)
    assert result["Gut.S1"][0] == "control"
    assert result["Gut.S2"][0] == "none"
    assert n == 3  # sample updates: 2 (study row) + 1 (sample row)


def test_override_patterns_are_matched_on_the_normalised_text():
    # The text the patterns see has '_' and '.' replaced by spaces, so a pattern with '_' never matches.
    samples = samples_frame([("Gut.S2_NTC", "Gut", "2", "", "", "", "")])
    assert run_overrides(samples, override_rows(("study", "Gut", "S2_NTC", "control", "")))[0] == 0
    assert run_overrides(samples, override_rows(("study", "Gut", "S2 NTC", "control", "")))[0] == 1


def test_override_that_matches_nothing_is_warned_about(capsys):
    samples = samples_frame([("Gut.S1", "Gut", "1", "", "", "", "")])
    assert run_overrides(samples, override_rows(("study", "Typo_Study", "", "control", "")))[0] == 0
    assert "override matched no sample: study Typo_Study" in capsys.readouterr().out


def test_override_with_unknown_scope_exits():
    samples = samples_frame([("Gut.S1", "Gut", "1", "", "", "", "")])
    with pytest.raises(SystemExit, match="scope must be study or sample"):
        run_overrides(samples, override_rows(("project", "Gut", "", "control", "")))


# ============================================================================= review_table
def review(rows):
    """rows: (alias, study, artificial, material, audit_hits, audit_unreviewed, audit_text)."""
    samples = pd.DataFrame(rows, columns=["sample_alias", "study_code", "artificial", "environment_material",
                                          "audit_hits", "audit_unreviewed", "text"])
    return clean.review_table(samples, samples["text"])


def test_review_table_one_row_per_study_and_category_of_unreviewed_hits():
    table = review([
        ("Gut.S1",     "Gut", "",                 "feces", "",                    False, "Gut.S1"),
        ("Gut.S2_NTC", "Gut", "",                 "feces", "negative_control",    True,  "Gut.S2_NTC"),
        ("Gut.S3_NTC", "Gut", "",                 "blank", "negative_control",    True,  "Gut.S3_NTC"),
        ("Gut.S4_NTC", "Gut", "",                 "feces", "negative_control",    False, "Gut.S4_NTC"),  # reviewed: left out
        ("Gut.BLANK",  "Gut", "negative control", "blank", "negative_control",    False, "Gut.BLANK"),   # Metalog flag: left out
        ("Sea.M1",     "Sea", "",                 "water", "cultivation;mesocosm", True, "Sea.M1; description: mesocosm culture"),
    ])
    assert table[["id", "category", "suggested", "n_hit", "n_study"]].values.tolist() == [
        ["Gut", "negative_control", "control",   2, 5],   # most hits first
        ["Sea", "cultivation",      "perturbed", 1, 1],   # a sample with two categories is in both rows
        ["Sea", "mesocosm",         "perturbed", 1, 1],
    ]
    gut = table.iloc[0]
    assert gut["metalog_flags_in_study"] == "negative control=1"
    assert gut["labels"] == "feces"                      # most common material among the 2 hits (feces, blank)
    assert gut["pattern"] == clean.AUDIT["negative_control"][1]
    assert (gut["scope"], gut["decision"], gut["reason"]) == ("study", "", "")


def test_review_table_examples_are_snippets_around_the_match():
    text = "." * 100 + "blank" + "." * 100   # '.' counts as a space when matching, so 'blank' is a word
    table = review([("S.1", "S", "", "", "negative_control", True, text)])
    assert table.iloc[0]["examples"] == "." * 60 + "blank" + "." * 60  # 60 characters on each side


def test_review_table_at_most_three_examples():
    rows = [(f"S.{i}_NTC", "S", "", "", "negative_control", True, f"S.{i}_NTC") for i in range(5)]
    assert review(rows).iloc[0]["examples"] == "S.0_NTC || S.1_NTC || S.2_NTC"


def test_review_table_example_falls_back_to_the_start_when_the_match_is_not_found_again():
    # audit() collapses '__' into one space ('single cell' hits), review_table replaces characters one by one
    # ('single  cell' does not): the example is then the first 120 characters.
    table = review([("S.1", "S", "", "", "sorted_cells", True, "S.1; description: single__cell")])
    assert table.iloc[0]["examples"] == "S.1; description: single__cell"


def test_review_table_without_unreviewed_hits_is_empty_with_all_columns():
    table = review([("S.1", "S", "", "", "", False, "S.1")])
    assert len(table) == 0 and "decision" in table.columns


# ============================================================================= backup_review
def test_backup_review_no_file_no_backup(tmp_path):
    clean.backup_review(str(tmp_path / "audit_review.tsv"), None)
    assert os.listdir(tmp_path) == []


def test_backup_review_file_without_decisions_is_not_backed_up(tmp_path):
    p = write_tsv(tmp_path / "audit_review.tsv", [{"scope": "study", "id": "Gut", "decision": ""}])
    clean.backup_review(p, None)
    assert os.listdir(tmp_path) == ["audit_review.tsv"]


def test_backup_review_decisions_are_backed_up_silently_when_passed_as_overrides(tmp_path, capsys):
    p = write_tsv(tmp_path / "audit_review.tsv", [{"scope": "study", "id": "Gut", "decision": "control"}])
    clean.backup_review(p, p)
    assert read_tsv(tmp_path / "audit_review.previous.tsv")["decision"].tolist() == ["control"]
    assert "WARNING" not in capsys.readouterr().out


def test_backup_review_decisions_not_passed_as_overrides_are_backed_up_with_a_warning(tmp_path, capsys):
    p = write_tsv(tmp_path / "audit_review.tsv", [{"scope": "study", "id": "Gut", "decision": "control"},
                                                  {"scope": "study", "id": "Sea", "decision": ""}])
    clean.backup_review(p, None)
    assert os.path.exists(tmp_path / "audit_review.previous.tsv")
    assert "had 1 decisions that were not passed with --overrides" in capsys.readouterr().out


# ============================================================================= clean_training_set
def flags_frame(rows):
    """Flags as main() leaves them: (alias, study, accession, drop_reason, bucket, in_gold_eval, biome_clean)."""
    df = pd.DataFrame(rows, columns=["sample_alias", "study_code", "spire_sample_name", "drop_reason",
                                     "artificial_bucket", "in_gold_eval", "biome_clean"])
    df["artificial"] = ""
    df["audit_hits"] = ""
    df["audit_unreviewed"] = False
    df["feature_clean"] = ""
    df["material_clean"] = ""
    return df


def training_rows(tmp_path, rows):
    """Step-2 rows: (sample_id, accession, study, biome, text)."""
    return write_tsv(tmp_path / "ts.tsv.gz", [
        {"sample_id": s, "spire_sample_name": acc, "study_code": st, "domain": "human",
         "biome": b, "feature": "", "material": "", "text": t} for s, acc, st, b, t in rows])


def test_clean_training_set_swaps_in_the_clean_labels_and_drops_flagged_samples(tmp_path):
    flags = flags_frame([
        ("Gut.S1",    "Gut", "SAMN01", "",                   "none",    True,  "ENVO_01001002"),
        ("Gut.BLANK", "Gut", "SAMN03", "artificial_control", "control", False, ""),
        ("Gut.M1",    "Gut", "SAMN04", "",                   "perturbed", False, "ENVO_01001002"),
    ])
    ts = training_rows(tmp_path, [
        ("SRS1", "SAMN01", "Gut", "",              "isolation_source: stool"),   # step 2 blanks obsolete ids
        ("SRS3", "SAMN03", "Gut", "",              "isolation_source: blank"),
        ("SRS4", "SAMN04", "Gut", "ENVO_01001002", "isolation_source: mesocosm"),
    ])
    rows, gold, n_before = clean.clean_training_set(ts, flags)

    assert rows["sample_id"].tolist() == ["SRS1", "SRS4"]        # SRS3 is a control: dropped
    assert rows["biome"].tolist() == ["ENVO_01001002", "ENVO_01001002"]   # the clean id ...
    assert rows["biome_raw"].tolist() == ["", "ENVO_01001002"]   # ... and the step-2 id
    assert gold["sample_id"].tolist() == ["SRS1"]                 # SRS4 is perturbed: training only
    assert n_before == 3


def test_clean_training_set_accession_of_a_duplicate_alias_joins_the_kept_alias(tmp_path):
    flags = flags_frame([("Sea.W2a", "Sea", "SAMN21", "",                "none", True,  "ENVO_01000023"),
                         ("Sea.W2b", "Sea", "SAMN21", "duplicate_alias", "none", False, "ENVO_01000023")])
    rows, gold, _ = clean.clean_training_set(training_rows(tmp_path, [("SRS5", "SAMN21", "Sea", "", "t")]), flags)
    assert rows["sample_id"].tolist() == gold["sample_id"].tolist() == ["SRS5"]


def test_clean_training_set_repeated_text_in_a_study_is_gold_once(tmp_path):
    flags = flags_frame([("Sea.W1", "Sea", "SAMN20", "", "none", True, "ENVO_01000023")])
    ts = training_rows(tmp_path, [("SRS2", "SAMN20", "Sea", "", "isolation_source: sea water"),   # two runs of
                                  ("SRS9", "SAMN20", "Sea", "", "isolation_source: sea water")])  # one sample
    rows, gold, _ = clean.clean_training_set(ts, flags)
    assert rows["dup_text_in_study"].tolist() == [False, True]
    assert gold["sample_id"].tolist() == ["SRS2"]


def test_clean_training_set_same_text_in_two_studies_is_gold_in_both(tmp_path):
    flags = flags_frame([("Sea.W1", "Sea", "SAMN20", "", "none", True, "ENVO_01000023"),
                         ("Bay.W1", "Bay", "SAMN40", "", "none", True, "ENVO_01000023")])
    ts = training_rows(tmp_path, [("SRS2", "SAMN20", "Sea", "", "isolation_source: sea water"),
                                  ("SRS9", "SAMN40", "Bay", "", "isolation_source: sea water")])
    rows, gold, _ = clean.clean_training_set(ts, flags)
    assert rows["dup_text_in_study"].tolist() == [False, False]
    assert gold["sample_id"].tolist() == ["SRS2", "SRS9"]


def test_clean_training_set_first_copy_of_a_text_counts_even_when_it_is_not_gold(tmp_path):
    """Known quirk: dup_text_in_study is computed on all linked rows, before any filtering. When the first
    copy is a dropped control or a perturbed sample, the later in_gold_eval copy is still a duplicate, so
    that text has no gold row (real data: 24 rows, one text of Kim_2016_marine_virus)."""
    flags = flags_frame([("Sea.BLANK", "Sea", "SAMN03", "artificial_control", "control", False, ""),
                         ("Sea.M1",    "Sea", "SAMN23", "",                   "perturbed", False, "ENVO_01000023"),
                         ("Sea.W1",    "Sea", "SAMN20", "",                   "none",      True,  "ENVO_01000023"),
                         ("Sea.W2",    "Sea", "SAMN21", "",                   "none",      True,  "ENVO_01000023")])
    ts = training_rows(tmp_path, [("SRS3", "SAMN03", "Sea", "", "isolation_source: water"),       # dropped control
                                  ("SRS1", "SAMN20", "Sea", "", "isolation_source: water"),
                                  ("SRS6", "SAMN23", "Sea", "", "isolation_source: sea water"),   # perturbed
                                  ("SRS2", "SAMN21", "Sea", "", "isolation_source: sea water")])
    rows, gold, _ = clean.clean_training_set(ts, flags)
    assert rows[["sample_id", "dup_text_in_study"]].values.tolist() == [["SRS1", True], ["SRS6", False], ["SRS2", True]]
    assert len(gold) == 0


def test_clean_training_set_rows_without_a_metalog_sample_are_dropped_with_a_warning(tmp_path, capsys):
    flags = flags_frame([("Gut.S1", "Gut", "SAMN01", "", "none", True, "")])
    ts = training_rows(tmp_path, [("SRS1", "SAMN01", "Gut", "", "a"), ("SRS7", "SAMN99", "Gut", "", "b")])
    rows, _, n_before = clean.clean_training_set(ts, flags)
    assert rows["sample_id"].tolist() == ["SRS1"]
    assert n_before == 1  # counted after this drop
    assert "1 training rows have no Metalog sample in this snapshot" in capsys.readouterr().out


def test_clean_training_set_output_columns(tmp_path):
    flags = flags_frame([("Gut.S1", "Gut", "SAMN01", "", "none", True, "")])
    rows, _, _ = clean.clean_training_set(training_rows(tmp_path, [("SRS1", "SAMN01", "Gut", "", "a")]), flags)
    assert list(rows.columns) == ["sample_id", "spire_sample_name", "study_code", "domain",
                                  "biome", "feature", "material", "biome_raw", "feature_raw", "material_raw",
                                  "artificial", "artificial_bucket", "audit_hits", "audit_unreviewed",
                                  "in_gold_eval", "dup_text_in_study", "text"]


# ============================================================================= counts
def test_counts_markdown_rows_most_frequent_first():
    assert clean.counts(pd.Series(["", "", "no_accession", "no_accession", "no_accession", "artificial_control"])) == \
        "| no_accession | 3 |\n| (none) | 2 |\n| artificial_control | 1 |"
