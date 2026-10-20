"""Integration tests of 2b_clean_metalog.py: the whole script (main) on a mini Metalog snapshot.

Read this file top to bottom to see how the step is used:
  1. test_basic_run            one run: what every flag says about each of 16 toy samples
  2. test_review_loop          fill in audit_review.tsv, re-run with --overrides, re-run again (idempotent)
  3. test_with_training_set    --training_set and --label_map: the clean and gold training sets
  4. edge cases                snapshot selection, forgotten --overrides, hand-written overrides, ...

    pytest scripts/ontology_mapping/tests/test_2b_clean_metalog_integration.py -v
"""

import os
import sys
from types import SimpleNamespace

import pytest

from step2b_helpers import load_step, read_tsv, terms_table, write_snapshot, write_tsv

clean = load_step("2b_clean_metalog")
common = load_step("common")
pytestmark = pytest.mark.filterwarnings("ignore:This pattern is interpreted as a regular expression")

# ----------------------------------------------------------------------------- the mini snapshot
# Raw Metalog label values (as in the core_wide files)
ANIMAL = "animal-associated environment [ENVO:01001002]"
GUT = "intestine environment [ENVO:2100002]"
FECAL = "fecal material [ENVO:00002003]"
PELAGIC = "marine pelagic biome [ENVO:01000023]"
SEAWATER = "sea water [ENVO:00002149]"
SOIL = "soil [ENVO:00001998]"

# 16 samples in 4 studies, one per domain. Each sample shows one thing; the last column is what the
# script should conclude (checked in test_basic_run).
CORE = {
    "human": [
        # alias        study  accession  artificial          biome     feature  material
        ("Gut.S1",     "Gut", "SAMN01", "",                 ANIMAL,   GUT,     FECAL),               # normal: gold
        ("Gut.S2_NTC", "Gut", "SAMN02", "",                 ANIMAL,   GUT,     FECAL),               # 'NTC' in alias: audit hit
        ("Gut.BLANK_1", "Gut", "SAMN03", "negative control", ANIMAL,  "",      "negative control"),  # Metalog control: dropped
        ("Gut.S4",     "Gut", "",       "",                 ANIMAL,   GUT,     FECAL),               # no accession: dropped
    ],
    "animal": [
        ("Mouse.M1", "Mouse", "SAMN10", "", "obsolete human-associated habitat [ENVO:00009003]", GUT, "feces [UBERON:0001988]"),  # obsolete biome
        ("Mouse.M2", "Mouse", "SAMN11", "", ANIMAL, GUT, "ciliated cell [CL:0000064]"),                                        # CL material
    ],
    "ocean": [
        ("Sea.W1",    "Sea", "SAMN20", "",         PELAGIC, "", SEAWATER),                    # normal: gold
        ("Sea.W2a",   "Sea", "SAMN21", "",         PELAGIC, "", SEAWATER),                    # same accession, same labels:
        ("Sea.W2b",   "Sea", "SAMN21", "",         PELAGIC, "", SEAWATER),                    #   keep the first alias
        ("Sea.W3a",   "Sea", "SAMN22", "",         PELAGIC, "", SEAWATER),                    # same accession, labels
        ("Sea.W3b",   "Sea", "SAMN22", "",         PELAGIC, "", "sediment [ENVO:00002007]"),  #   disagree: drop both
        ("Sea.MESOCOSM_1", "Sea", "SAMN23", "mesocosm", PELAGIC, "", SEAWATER),               # perturbed: train, not gold
    ],
    "environmental": [
        ("Soil.P1", "Soil", "SAMN30", "paleosample", "", "", SOIL),   # degraded: train, not gold
        ("Soil.P2", "Soil", "SAMN31", "",            "", "", SOIL),   # 'ancient' in its description: audit hit
        ("Soil.P3", "Soil", "SAMN32", "",            "", "", SOIL),   # 'mesocosm' only in the study abstract: gold
        ("Soil.P4", "Soil", "SAMN33", "",            "", "", SOIL),   # normal (until the training-set text, test 3)
    ],
}
LONG = [
    # alias      tier        key                  value
    ("Soil.P2", "extended", "sample_description", "ancient permafrost soil"),
    ("Soil.P3", "extended", "sample_description", "fresh topsoil"),
    ("Soil.P3", "all",      "study_abstract",     "we compare fresh soil with mesocosm controls"),  # not audited
]


@pytest.fixture
def inputs(tmp_path):
    """Paths of the inputs of one run: the snapshot, the term index and an output directory."""
    return SimpleNamespace(metalog_dir=write_snapshot(tmp_path / "metalog", CORE, LONG),
                           terms=write_tsv(tmp_path / "terms.tsv.gz", terms_table()),
                           out=str(tmp_path / "clean"), tmp=tmp_path)


def run_step(monkeypatch, capsys, inputs, *extra):
    """Run `python 2b_clean_metalog.py --metalog_dir ... --ontology_terms ... --output_dir ... <extra>` and
    read its outputs. flags is indexed by sample alias."""
    monkeypatch.setattr(sys, "argv", ["2b_clean_metalog.py", "--metalog_dir", inputs.metalog_dir,
                                      "--ontology_terms", inputs.terms, "--output_dir", inputs.out, *extra])
    clean.main()
    out = lambda name: os.path.join(inputs.out, name)
    result = SimpleNamespace(stdout=capsys.readouterr().out,
                             flags=read_tsv(out("metalog_flags.tsv.gz")).set_index("sample_alias"),
                             review=read_tsv(out("audit_review.tsv")),
                             summary=open(out("summary.md")).read())
    if os.path.exists(out("training_set.clean.tsv.gz")):
        result.clean = read_tsv(out("training_set.clean.tsv.gz"))
        result.gold = read_tsv(out("training_set.gold.tsv.gz"))
    return result


def flag_view(flags, columns=("drop_reason", "artificial_bucket", "audit_hits", "in_train", "in_gold_eval")):
    """{alias: (the chosen flag columns)}, to compare against a table written by hand."""
    return {alias: tuple(row[c] for c in columns) for alias, row in flags.iterrows()}


# ============================================================================= 1. one run
def test_basic_run(monkeypatch, capsys, inputs):
    r = run_step(monkeypatch, capsys, inputs)

    # One row per Metalog sample (nothing is deleted), with the flags:
    assert flag_view(r.flags) == {
        # alias         drop_reason               bucket       audit_hits                in_train  in_gold_eval
        "Gut.S1":     ("",                      "none",      "",                       "True",   "True"),
        "Gut.S2_NTC": ("",                      "none",      "negative_control",       "True",   "False"),  # unreviewed hit
        "Gut.BLANK_1": ("artificial_control",   "control",   "negative_control",       "False",  "False"),
        "Gut.S4":     ("no_accession",          "none",      "",                       "False",  "False"),
        "Mouse.M1":   ("",                      "none",      "",                       "True",   "True"),
        "Mouse.M2":   ("",                      "none",      "",                       "True",   "True"),
        "Sea.W1":     ("",                      "none",      "",                       "True",   "True"),
        "Sea.W2a":    ("",                      "none",      "",                       "True",   "True"),
        "Sea.W2b":    ("duplicate_alias",       "none",      "",                       "False",  "False"),
        "Sea.W3a":    ("conflicting_duplicate", "none",      "",                       "False",  "False"),
        "Sea.W3b":    ("conflicting_duplicate", "none",      "",                       "False",  "False"),
        "Sea.MESOCOSM_1": ("",                  "perturbed", "mesocosm",               "True",   "False"),
        "Soil.P1":    ("",                      "degraded",  "",                       "True",   "False"),
        "Soil.P2":    ("",                      "none",      "ancient_or_post_mortem", "True",   "False"),  # unreviewed hit
        "Soil.P3":    ("",                      "none",      "",                       "True",   "True"),
        "Soil.P4":    ("",                      "none",      "",                       "True",   "True"),
    }
    # audit_unreviewed: an audit hit on a sample that Metalog did not flag (BLANK_1 and MESOCOSM_1 are flagged,
    # so their hits need no review)
    assert r.flags.index[r.flags["audit_unreviewed"] == "True"].tolist() == ["Gut.S2_NTC", "Soil.P2"]

    # Label flags (raw id -> status -> clean id), a few samples:
    labels = flag_view(r.flags, ["biome_raw", "biome_status", "biome_clean", "material_status", "material_clean"])
    assert labels["Gut.S1"] == ("ENVO_01001002", "ok", "ENVO_01001002", "ok", "ENVO_00002003")
    assert labels["Mouse.M1"] == ("ENVO_00009003", "obsolete", "", "ok", "UBERON_0001988")  # no label map yet
    assert labels["Mouse.M2"] == ("ENVO_01001002", "ok", "ENVO_01001002", "other_ontology", "")
    assert labels["Gut.BLANK_1"] == ("ENVO_01001002", "ok", "ENVO_01001002", "control_value", "")
    assert labels["Soil.P1"][1] == "empty"
    # 'biome' subtree: marine pelagic biome is under it, animal-associated environment is not
    assert r.flags.loc["Sea.W1", "biome_in_biome_subtree"] == "True"
    assert r.flags.loc["Gut.S1", "biome_in_biome_subtree"] == "False"
    # The blank still carries the study's default biome: worth knowing, it is a usable-looking label on a control
    assert r.flags.loc["Gut.BLANK_1", "control_has_habitat_label"] == "True"

    # audit_text: what the audit regexes were matched on (alias + audited long-file fields)
    assert r.flags.loc["Soil.P2", "audit_text"] == "Soil.P2; sample_description: ancient permafrost soil"
    assert r.flags.loc["Soil.P3", "audit_text"] == "Soil.P3; sample_description: fresh topsoil"  # no abstract

    # audit_review.tsv: one row per study x category of unreviewed hits, decision left empty
    assert r.review[["scope", "id", "category", "suggested", "n_hit", "n_study", "decision"]].values.tolist() == [
        ["study", "Gut",  "negative_control",       "control",  "1", "4", ""],
        ["study", "Soil", "ancient_or_post_mortem", "degraded", "1", "4", ""],
    ]
    assert r.review.loc[0, "metalog_flags_in_study"] == "negative control=1"   # Gut.BLANK_1
    assert r.review.loc[1, "examples"] == "Soil.P2; sample_description: ancient permafrost soil"

    # summary.md
    for line in ["16 Metalog samples in 4 studies.",
                 "| artificial_control | 1 |", "| conflicting_duplicate | 2 |", "| no_accession | 1 |",
                 "| duplicate_alias | 1 |", "| perturbed | 1 |", "| degraded | 1 |",
                 "2 unreviewed samples, 2 study x category rows",
                 "Clean biome labels under ENVO 'biome': 6 of 11.",      # the 6 Sea samples; Soil has no biome
                 "| in_train | 11 |", "| in_gold_eval | 7 |"]:
        assert line in r.summary, line


# ============================================================================= 2. review loop
def test_review_loop(monkeypatch, capsys, inputs):
    review_path = os.path.join(inputs.out, "audit_review.tsv")

    # Run 1: two rows to review (see test_basic_run).
    run_step(monkeypatch, capsys, inputs)

    # The reviewer opens audit_review.tsv: the Gut 'NTC' sample is a real control, 'ancient' in Soil is a
    # false alarm (ok = a normal sample).
    review = read_tsv(review_path)
    review.loc[review["id"] == "Gut", "decision"] = "control"
    review.loc[review["id"] == "Soil", ["decision", "reason"]] = ["ok", "ancient = old permafrost, modern DNA"]
    review.to_csv(review_path, sep="\t", index=False)

    # Run 2: the same command plus --overrides audit_review.tsv.
    r2 = run_step(monkeypatch, capsys, inputs, "--overrides", review_path)
    columns = ("drop_reason", "artificial_bucket", "artificial_source", "reviewed", "in_gold_eval")
    view = flag_view(r2.flags, columns)
    assert view["Gut.S2_NTC"] == ("artificial_control", "control", "override", "True", "False")  # now dropped
    assert view["Soil.P2"] == ("", "none", "override", "True", "True")                          # now gold
    assert view["Gut.S1"] == ("", "none", "", "False", "True")   # same study, pattern did not match: untouched
    assert r2.flags["audit_unreviewed"].eq("False").all()
    assert "| in_gold_eval | 8 |" in r2.summary
    # The new review file keeps the two decided rows (and has nothing left to review) ...
    assert r2.review[["id", "decision"]].values.tolist() == [["Gut", "control"], ["Soil", "ok"]]
    # ... and the file with the decisions was backed up before being overwritten.
    assert read_tsv(os.path.join(inputs.out, "audit_review.previous.tsv"))["decision"].tolist() == ["control", "ok"]
    assert "WARNING" not in r2.stdout

    # Run 3: the same command again gives the same outputs.
    r3 = run_step(monkeypatch, capsys, inputs, "--overrides", review_path)
    assert r3.flags.equals(r2.flags)
    assert r3.review.equals(r2.review)


# ============================================================================= 3. with the training set
# Step-2 rows: MicrobeAtlas records linked to a Metalog sample by accession. biome = the step-2 id
# (ENVO / UBERON only, obsolete ids already blanked by step 2: Mouse.M1's ENVO_00009003 arrives as "").
TRAINING = [
    # sample_id  accession study    biome            text
    ("SRS1",   "SAMN01", "Gut",   "ENVO_01001002", "isolation_source: stool"),                 # Gut.S1: gold
    ("SRS3",   "SAMN03", "Gut",   "ENVO_01001002", "isolation_source: blank"),                 # Gut.BLANK_1: dropped
    ("SRS10",  "SAMN10", "Mouse", "",              "isolation_source: mouse feces"),           # Mouse.M1: label map
    ("SRS2",   "SAMN20", "Sea",   "ENVO_01000023", "isolation_source: sea water"),             # Sea.W1: gold
    ("SRS9",   "SAMN20", "Sea",   "ENVO_01000023", "isolation_source: sea water"),             # 2nd run, same text
    ("SRS5",   "SAMN21", "Sea",   "ENVO_01000023", "isolation_source: surface water"),         # Sea.W2a (W2b is the alias)
    ("SRS12",  "SAMN22", "Sea",   "ENVO_01000023", "isolation_source: deep water"),            # Sea.W3a/b conflict: dropped
    ("SRS6",   "SAMN23", "Sea",   "ENVO_01000023", "isolation_source: tank"),                  # Sea.MESOCOSM_1: training only
    ("SRS8",   "SAMN33", "Soil",  "",              "sample_type: enrichment culture"),         # Soil.P4: new audit hit
    ("SRS11",  "SAMN32", "Soil",  "",              "isolation_source: topsoil; abstract: we compare with enrichment cultures"),
    ("SRS7",   "SAMN99", "Soil",  "",              "isolation_source: soil"),                  # not in Metalog
]


DOMAIN = {"Gut": "human", "Mouse": "animal", "Sea": "ocean", "Soil": "environmental"}


def test_with_training_set(monkeypatch, capsys, inputs):
    ts = write_tsv(inputs.tmp / "training_set.tsv.gz", [
        {"sample_id": s, "spire_sample_name": acc, "study_code": st, "domain": DOMAIN[st], "biome": b, "feature": "",
         "material": "", "text": t} for s, acc, st, b, t in TRAINING])
    label_map = write_tsv(inputs.tmp / "label_map.tsv", [
        {"slot": "biome", "from_id": "ENVO_00009003", "to_id": "ENVO_01001002", "reason": "obsolete human habitat"}])
    r = run_step(monkeypatch, capsys, inputs, "--training_set", ts, "--label_map", label_map)

    # The MicrobeAtlas text adds sample fields to the audit: Soil.P4 is now an unreviewed cultivation hit.
    # Soil.P3's record mentions enrichment cultures only in the study abstract, which is cut off.
    assert r.flags.loc["Soil.P4", "audit_hits"] == "cultivation"
    # audit_text = '<alias>; <audited long-file fields (none for P4)>; <audited fields of the step-2 text>'
    assert r.flags.loc["Soil.P4", "audit_text"] == "Soil.P4; ; sample_type: enrichment culture"
    assert r.flags.loc["Soil.P3", "in_gold_eval"] == "True"
    # The label map rescues the obsolete biome
    assert (r.flags.loc["Mouse.M1", "biome_status"], r.flags.loc["Mouse.M1", "biome_clean"]) == ("obsolete", "ENVO_01001002")

    # training_set.clean: every linked row without a drop reason, in input order
    assert r.clean["sample_id"].tolist() == ["SRS1", "SRS10", "SRS2", "SRS9", "SRS5", "SRS6", "SRS8", "SRS11"]
    #   SRS3 (blank), SRS12 (conflicting duplicate) and SRS7 (no Metalog sample) are gone
    assert "1 training rows have no Metalog sample" in r.stdout
    # biome = the clean id, biome_raw = the step-2 id
    srs10 = r.clean.set_index("sample_id").loc["SRS10"]
    assert (srs10["biome"], srs10["biome_raw"], srs10["domain"]) == ("ENVO_01001002", "", "animal")

    # training_set.gold: in_gold_eval, and the first of repeated texts in a study
    assert r.gold["sample_id"].tolist() == ["SRS1", "SRS10", "SRS2", "SRS5", "SRS11"]
    #   SRS9: same text as SRS2;  SRS6: perturbed;  SRS8: unreviewed audit hit
    assert "10 step-2 rows → 8 clean (training) → 5 gold" in r.summary


# ============================================================================= 4. edge cases
def test_date_selects_an_older_snapshot(monkeypatch, capsys, inputs):
    write_snapshot(inputs.metalog_dir, {"human": [("Old.S1", "Old", "SAMN90", "", ANIMAL, GUT, FECAL)]},
                   date="2025-01-01")
    assert len(run_step(monkeypatch, capsys, inputs).flags) == 16                         # latest by default
    old = run_step(monkeypatch, capsys, inputs, "--date", "2025-01-01")
    assert old.flags.index.tolist() == ["Old.S1"]
    assert "# Metalog cleaning flags, snapshot 2025-01-01" in old.summary


def test_forgetting_overrides_after_a_review_keeps_a_backup(monkeypatch, capsys, inputs):
    review_path = os.path.join(inputs.out, "audit_review.tsv")
    run_step(monkeypatch, capsys, inputs)
    review = read_tsv(review_path)
    review["decision"] = "control"
    review.to_csv(review_path, sep="\t", index=False)

    r = run_step(monkeypatch, capsys, inputs)  # no --overrides: the decisions are not applied ...
    assert r.review["decision"].tolist() == ["", ""]
    assert r.flags.loc["Gut.S2_NTC", "artificial_bucket"] == "none"
    # ... but they are not lost
    assert "had 2 decisions that were not passed with --overrides" in r.stdout
    assert read_tsv(os.path.join(inputs.out, "audit_review.previous.tsv"))["decision"].tolist() == ["control", "control"]


def test_hand_written_sample_override(monkeypatch, capsys, inputs):
    # A reviewer can flag any sample, audit hit or not. No `pattern`, no `category` column needed.
    overrides = write_tsv(inputs.tmp / "my_overrides.tsv", [
        {"scope": "sample", "id": "Sea.W1", "pattern": "", "decision": "perturbed", "reason": "filtered >3 um"}])
    r = run_step(monkeypatch, capsys, inputs, "--overrides", overrides)
    assert flag_view(r.flags, ["artificial_bucket", "in_train", "in_gold_eval"])["Sea.W1"] == ("perturbed", "True", "False")
    # carried into the new review file, above the rows still to review
    assert r.review[["scope", "id", "category", "decision"]].values.tolist() == [
        ["sample", "Sea.W1", "", "perturbed"],
        ["study", "Gut", "negative_control", ""],
        ["study", "Soil", "ancient_or_post_mortem", ""]]


def test_a_control_without_accession_has_both_drop_reasons(monkeypatch, capsys, inputs):
    write_snapshot(inputs.metalog_dir, {"human": [("Gut.S1", "Gut", "SAMN01", "", ANIMAL, GUT, FECAL),
                                                  ("Gut.BLANK9", "Gut", "", "mock", "", "", "mock")]})
    r = run_step(monkeypatch, capsys, inputs)
    assert r.flags.loc["Gut.BLANK9", "drop_reason"] == "no_accession;artificial_control"


def test_snapshot_with_nothing_to_review(monkeypatch, capsys, inputs):
    # Only one domain has samples (the other files have a header only) and nothing looks artificial.
    write_snapshot(inputs.metalog_dir, {"ocean": [("Sea.W1", "Sea", "SAMN20", "", PELAGIC, "", SEAWATER)]})
    r = run_step(monkeypatch, capsys, inputs)
    assert flag_view(r.flags)["Sea.W1"] == ("", "none", "", "True", "True")
    assert len(r.review) == 0 and "decision" in r.review.columns
    assert "0 unreviewed samples, 0 study x category rows" in r.summary


# ============================================================================= step 2 -> 2b contract
# sample.info doubles the study prefix ('study_STUDY_TITLE=...'). record_to_text strips the first 'study_',
# so the step-2 text still says 'study_title:', which is where audited_fields cuts the study part off.
def test_study_fields_of_a_real_step2_record_are_not_audited():
    record = ["sample_title=topsoil core", "sample_DESCRIPTION=plot 3",
              "study_STUDY_TITLE=Negative controls in soil sequencing", "study_STUDY_TYPE=Other",
              "study_STUDY_ABSTRACT=We also sequenced blanks."]
    text = common.record_to_text(record, {}, 1000)
    assert text == ("title: topsoil core; description: plot 3; study_title: Negative controls in soil sequencing; "
                    "study_type: Other; study_abstract: We also sequenced blanks.")
    assert clean.audited_fields(text) == "title: topsoil core; description: plot 3"
