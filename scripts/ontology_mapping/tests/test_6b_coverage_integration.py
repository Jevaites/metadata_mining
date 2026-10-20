"""Integration tests for 6b_coverage.py: main() run end to end on a toy Metalog and a toy atlas.

How the step is used in the pipeline (README, step 6b):
    python 6b_coverage.py --samples        training_set.clean.tsv.gz        # Metalog samples + labels (step 2/2b)
                          --train_vectors  keywords.npz sub_biomes.npz      # their vectors (step 3)
                          [--fold_groups   project_groups.tsv]              # sample_id -> project_group
                          --index          index.npz                        # atlas sample -> .h5 rows (step 6)
                          --keywords_h5    GPT_keywords_unique_....h5       # atlas keyword embeddings
                          --sub_biomes_h5  GPT_sub_biomes_unique_....h5     # atlas sub-biome embeddings
                          --output_dir     atlas_coverage/
and atlas_coverage.tsv.gz is later joined with the outputs of steps 6, 6c and 7 on sample_id.

What main() does, step by step:
    1. training samples = common.select_samples(...): labelled Metalog samples that have BOTH a keyword
       and a sub-biome vector (at most --max_per_study per study)
    2. calibration (skipped when coverage_calibration.json already exists with the same settings):
         held-out similarity of each training sample = similarity to the nearest training sample of
         ANOTHER study (or project, with --fold_groups); threshold = --quantile of these, rounded to 4 dp
         -> coverage_calibration.json, metalog_held_out_sim.tsv.gz
    3. for each chunk of --chunk_rows keyword .h5 rows that some atlas sample uses (and that has no
       part file yet): score the atlas samples of that chunk
         sim = 0.5 * (keyword cosine + sub-biome cosine), or the keyword cosine alone without sub-biome
         coverage_sim = mean of the --k largest sims (4 dp); nearest_study = study of the largest
         -> parts/rows_<first row>.tsv.gz
       (stops early, keeping what is done, after --max_seconds; rerunning continues)
    4. concatenate all parts, recompute in_coverage = coverage_sim >= threshold -> atlas_coverage.tsv.gz
Unit tests of top_k_mean() and calibrate() are in test_6b_coverage_unit.py.

Conventions (as in the other test files of this folder):
  * Tests named `..._quirk` pin down current behaviour that is surprising; they document it, they do
    not endorse it. If the behaviour is changed on purpose, update the test.
  * Tests marked xfail(strict=True) are KNOWN BUGS: they fail today (reported as "xfailed"). Once the
    bug is fixed they pass, strict=True turns that into a failure ("XPASS(strict)"), and the marker must go.
"""

import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from step_loader import load_step

cov = load_step("6b_coverage.py", "atlas_coverage")  # not "coverage": that name belongs to coverage.py / pytest-cov


# ============================================================================= the toy world
# A 4-D embedding space with named axes. Every vector has unit length, and every cosine that matters
# is a "nice" number (0.8 = 4/5, 0.6 = 3/5), so each expected value below can be checked by hand.
#             gut  soil  lab  water
GUT       = (1.0, 0.0, 0.0, 0.0)
GUT_WET   = (0.8, 0.0, 0.0, 0.6)   # cos(GUT, GUT_WET)  = 0.8
SOIL      = (0.0, 1.0, 0.0, 0.0)   # cos(GUT, SOIL)     = 0
SOIL_WET  = (0.0, 0.8, 0.0, 0.6)   # cos(SOIL, SOIL_WET) = 0.8;  cos(GUT_WET, SOIL_WET) = 0.36 (shared "water")
LAB       = (0.0, 0.0, 1.0, 0.0)   # orthogonal to every training vector
LAB_GUT   = (0.6, 0.0, 0.8, 0.0)   # cos with GUT 0.6, with GUT_WET 0.48
GUT_LAB   = (0.8, 0.0, 0.6, 0.0)   # cos with GUT 0.8, with GUT_WET 0.64

# --- Metalog: 4 studies x 2 samples are the training data; the 2 lab_study samples are dropped -------
METALOG = [
    # sample_id         study_code      biome label   keyword vec  sub-biome vec
    ("m_gut1a",         "gut_study_1",  "intestine",  GUT,         GUT),
    ("m_gut1b",         "gut_study_1",  "intestine",  GUT,         GUT),
    ("m_gut2a",         "gut_study_2",  "intestine",  GUT_WET,     GUT_WET),
    ("m_gut2b",         "gut_study_2",  "intestine",  GUT_WET,     GUT_WET),
    ("m_soil1a",        "soil_study_1", "soil",       SOIL,        SOIL),
    ("m_soil1b",        "soil_study_1", "soil",       SOIL,        SOIL),
    ("m_soil2a",        "soil_study_2", "soil",       SOIL_WET,    SOIL_WET),
    ("m_soil2b",        "soil_study_2", "soil",       SOIL_WET,    SOIL_WET),
    ("m_lab_unlabelled", "lab_study",   "",           LAB,         LAB),    # no label   -> not a training sample
    ("m_lab_no_sb",     "lab_study",    "laboratory", LAB,         None),   # no sub-biome vector -> not one either
]
TRAINING_IDS = {s for s, *_ in METALOG if not s.startswith("m_lab")}
# Calibration: every training sample's nearest sample in ANOTHER study is at 0.8
#   gut_study_1 <-> gut_study_2 (0.8),  soil_study_1 <-> soil_study_2 (0.8);  gut vs soil: 0 or 0.36
# so all 8 held-out similarities are 0.8 and the threshold (any quantile) is 0.8.
THRESHOLD = 0.8

# --- the atlas: embeddings are stored once per distinct text, samples point at rows (index.npz) ------
KEYWORD_H5 = [  # row: (text, vector)
    ("human gut",                    GUT),       # row 0
    ("agricultural soil",            SOIL),      # row 1
    ("lab culture of a gut isolate", LAB_GUT),   # row 2
    ("gut content, lab processed",   GUT_LAB),   # row 3
    ("a text no sample uses",        LAB),       # row 4 (no atlas sample points here)
]
SUB_BIOME_H5 = [
    ("gut",               GUT),       # row 0
    ("soil",              SOIL),      # row 1
    ("lab gut isolate",   LAB_GUT),   # row 2
    ("processed gut",     GUT_LAB),   # row 3
    ("laboratory",        LAB),       # row 4
]
ATLAS = [
    # sample_id       keyword row  sub-biome row (-1 = the sample has no sub-biome text)
    ("a_gut",          0,           0),
    ("a_gut_twin",     0,           0),     # same two texts as a_gut: scored once, both get the result
    ("a_soil",         1,           1),
    ("a_lab_gut",      2,           2),
    ("a_border",       3,           3),
    ("a_gut_no_sb",    0,          -1),
    ("a_gut_in_lab",   0,           4),
]
EXPECTED = {
    # sample_id:      (coverage_sim, nearest_study,  in_coverage)
    "a_gut":          (1.0,          "gut_study_1",  True),    # identical to m_gut1a/b
    "a_gut_twin":     (1.0,          "gut_study_1",  True),
    "a_soil":         (1.0,          "soil_study_1", True),
    "a_lab_gut":      (0.6,          "gut_study_1",  False),   # cos(LAB_GUT, GUT) = 0.6 < 0.8
    "a_border":       (0.8,          "gut_study_1",  True),    # cos(GUT_LAB, GUT) = 0.8: >= threshold counts as in
    "a_gut_no_sb":    (1.0,          "gut_study_1",  True),    # keyword cosine alone: 1
    "a_gut_in_lab":   (0.5,          "gut_study_1",  False),   # 0.5 * (keyword 1 + sub-biome cos(LAB, GUT) 0)
}


# ============================================================================= helpers
def write_npz(p, rows):
    """A step-3 .npz: sample_ids, index, vectors (vector of sample_ids[i] = vectors[index[i]])."""
    np.savez(p, sample_ids=np.array([s for s, _ in rows]), index=np.arange(len(rows)),
             vectors=np.array([v for _, v in rows]))
    return str(p)


def write_h5(p, rows, scale=1.0):
    """A unique-embeddings .h5: texts, embeddings (one row per distinct text)."""
    with h5py.File(p, "w") as f:
        f.create_dataset("texts", data=[t.encode() for t, _ in rows])
        f.create_dataset("embeddings", data=scale * np.array([v for _, v in rows], dtype=np.float32))
    return str(p)


@pytest.fixture
def world(tmp_path):
    """Writes the toy world above as the pipeline's input files; returns their paths."""
    d = tmp_path / "inputs"
    d.mkdir()
    pd.DataFrame([(s, study, biome, "", "") for s, study, biome, _, _ in METALOG],
                 columns=["sample_id", "study_code", "biome", "feature", "material"]
                 ).to_csv(d / "training_set.tsv", sep="\t", index=False)
    np.savez(d / "index.npz", sample_ids=np.array([s for s, _, _ in ATLAS]),
             kw_rows=np.array([k for _, k, _ in ATLAS]), sb_rows=np.array([b for _, _, b in ATLAS]))
    return SimpleNamespace(
        dir=d,
        samples=str(d / "training_set.tsv"),
        kw_npz=write_npz(d / "keywords.npz", [(s, kw) for s, _, _, kw, _ in METALOG]),
        sb_npz=write_npz(d / "sub_biomes.npz", [(s, sb) for s, _, _, _, sb in METALOG if sb is not None]),
        index=str(d / "index.npz"),
        keywords_h5=write_h5(d / "keywords.h5", KEYWORD_H5),
        sub_biomes_h5=write_h5(d / "sub_biomes.h5", SUB_BIOME_H5),
    )


@pytest.fixture
def run_6b(world, tmp_path, monkeypatch):
    """run_6b(*options, out="out") -> output dir. Calls main() exactly as from the command line.
    module=: another version of the script (test_6b_coverage_mutations.py passes deliberately broken ones)."""

    def _run(*options, out="out", module=cov):
        out_dir = tmp_path / out
        monkeypatch.setattr(sys, "argv", [
            "6b_coverage.py", "--samples", world.samples, "--train_vectors", world.kw_npz, world.sb_npz,
            "--index", world.index, "--keywords_h5", world.keywords_h5, "--sub_biomes_h5", world.sub_biomes_h5,
            "--output_dir", str(out_dir), *map(str, options)])
        module.main()
        return out_dir

    return _run


def read_result(out_dir):
    """atlas_coverage.tsv.gz as {sample_id: (coverage_sim, nearest_study, in_coverage)}."""
    table = pd.read_csv(out_dir / "atlas_coverage.tsv.gz", sep="\t")
    assert list(table.columns) == ["sample_id", "coverage_sim", "nearest_study", "in_coverage"]
    return {s: (sim, study, bool(flag)) for s, sim, study, flag in table.itertuples(index=False)}


def read_calibration(out_dir):
    return json.loads((out_dir / "coverage_calibration.json").read_text())


def read_held_out(out_dir):
    """metalog_held_out_sim.tsv.gz as {sample_id: (study_code, held_out_sim)}."""
    table = pd.read_csv(out_dir / "metalog_held_out_sim.tsv.gz", sep="\t")
    return {s: (study, sim) for s, study, sim in table.itertuples(index=False)}


def part_names(out_dir):
    return sorted(p.name for p in (out_dir / "parts").iterdir())


def write_groups(world, project_of):
    """project_groups.tsv in its real layout (sample_id, study_code, project_group) for the samples listed."""
    study = {s: st for s, st, *_ in METALOG}
    p = world.dir / "project_groups.tsv"
    pd.DataFrame([(s, study[s], g) for s, g in project_of.items()],
                 columns=["sample_id", "study_code", "project_group"]).to_csv(p, sep="\t", index=False)
    return p


# ============================================================================= the basic workflow
def test_toy_atlas(run_6b):
    """Read EXPECTED next to the vectors: one atlas sample per kind of decision."""
    out = run_6b()
    assert read_result(out) == EXPECTED


def test_calibration_outputs(run_6b):
    out = run_6b()
    calibration = read_calibration(out)
    assert calibration["threshold"] == THRESHOLD
    assert set(calibration["held_out_quantiles"].values()) == {0.8}    # all held-out values are 0.8
    assert calibration["settings"]["n_train"] == 8                     # the two lab_study samples are not used
    assert calibration["settings"]["k"] == 1 and calibration["settings"]["quantile"] == 0.05
    assert calibration["settings"]["chunk_rows"] == 5000               # a resume must use the same chunks
    # one row per TRAINING sample, with its study and held-out similarity
    assert read_held_out(out) == {s: (study, 0.8) for s, study, *_ in METALOG if s in TRAINING_IDS}


def test_files_written(run_6b):
    out = run_6b()
    assert sorted(p.name for p in out.iterdir()) == [
        "atlas_coverage.tsv.gz", "coverage_calibration.json", "metalog_held_out_sim.tsv.gz", "parts"]
    assert part_names(out) == ["rows_00000000.tsv.gz"]   # default --chunk_rows 5000: one chunk for 5 rows


# ============================================================================= training data and calibration
def test_unusable_metalog_samples_do_not_count_as_coverage(run_6b):
    # m_lab_unlabelled has the LAB vector but no label; m_lab_no_sb has a label but no sub-biome vector.
    # Neither is a training sample.
    out = run_6b()
    assert "lab_study" not in {study for study, _ in read_held_out(out).values()}
    # If m_lab_unlabelled counted, a_lab_gut would be at cos(LAB_GUT, LAB) = 0.8 from it (keyword and
    # sub-biome alike) and in coverage. It stays at 0.6 from gut_study_1, out of coverage.
    assert read_result(out)["a_lab_gut"] == (0.6, "gut_study_1", False)


def test_fold_groups_merge_studies_into_projects(world, run_6b):
    # Both gut studies are one project: a gut sample can no longer be compared with the other gut study.
    # The soil samples are not in the file: they fall back to their study code.
    groups = write_groups(world, dict.fromkeys(["m_gut1a", "m_gut1b", "m_gut2a", "m_gut2b"], "gut_consortium"))
    out = run_6b("--fold_groups", groups)

    held_out = {s: sim for s, (_, sim) in read_held_out(out).items()}
    assert held_out == {
        "m_gut1a": 0.0,  "m_gut1b": 0.0,     # GUT vs SOIL / SOIL_WET: 0
        "m_gut2a": 0.36, "m_gut2b": 0.36,    # GUT_WET vs SOIL_WET: 0.36
        "m_soil1a": 0.8, "m_soil1b": 0.8,    # soil studies are still two groups: 0.8 as before
        "m_soil2a": 0.8, "m_soil2b": 0.8,
    }
    # 5 % quantile of [0, 0, 0.36, 0.36, 0.8, 0.8, 0.8, 0.8] = 0: nothing is flagged any more
    assert read_calibration(out)["threshold"] == 0.0
    assert all(flag for _, _, flag in read_result(out).values())
    # only the flag depends on the calibration: similarities and nearest studies are unchanged
    assert {s: r[:2] for s, r in read_result(out).items()} == {s: e[:2] for s, e in EXPECTED.items()}


def test_samples_with_fewer_than_k_samples_outside_their_group_get_nan(world, run_6b):
    # Three studies are one project ("big", 6 samples); only soil_study_2 (2 samples) is outside it.
    # With --k 3, a "big" sample has only 2 samples outside its group: no held-out similarity ("nan" in the
    # file, left out of the threshold). soil_study_2 samples have 6 outside: their 3 nearest are
    # m_soil1a/b (0.8 each) and a GUT_WET sample (0.36) -> (0.8 + 0.8 + 0.36) / 3 = 0.6533.
    big = ["m_gut1a", "m_gut1b", "m_gut2a", "m_gut2b", "m_soil1a", "m_soil1b"]
    groups = write_groups(world, dict.fromkeys(big, "big"))
    out = run_6b("--fold_groups", groups, "--k", 3)

    held_out = {s: sim for s, (_, sim) in read_held_out(out).items()}
    assert all(np.isnan(held_out[s]) for s in big)
    assert held_out["m_soil2a"] == held_out["m_soil2b"] == 0.6533
    assert read_calibration(out)["threshold"] == 0.6533   # from the 2 samples that have a value
    # experiments/coverage_check.py reads the file with common.read_tsv (strings) and .astype(float)
    from common import read_tsv
    assert read_tsv(out / "metalog_held_out_sim.tsv.gz")["held_out_sim"].astype(float).isna().sum() == 6


def test_max_per_study_caps_the_training_samples(run_6b):
    # 1 sample per study is kept (the one with the smallest common.stable_key). Each study's two samples
    # are identical here, so only the counts change, not the scores.
    out = run_6b("--max_per_study", 1)
    assert read_calibration(out)["settings"]["n_train"] == 4
    assert sorted(study for study, _ in read_held_out(out).values()) == [
        "gut_study_1", "gut_study_2", "soil_study_1", "soil_study_2"]
    assert read_result(out) == EXPECTED


def test_k2_averages_the_two_nearest_training_samples(run_6b):
    # The k nearest may come from the same study: a_lab_gut's two nearest are m_gut1a and m_gut1b (0.6 each).
    # a_gut_in_lab: sims to m_gut1a/b are 0.5 each -> 0.5. Calibration: each sample's two nearest in
    # other studies are both at 0.8 -> threshold 0.8. Since every study has duplicates, nothing changes.
    out = run_6b("--k", 2)
    assert read_calibration(out)["threshold"] == THRESHOLD
    assert read_result(out) == EXPECTED


def test_k_larger_than_the_training_set_is_refused(run_6b):
    # 8 training samples: the mean of the 9 nearest does not exist
    with pytest.raises(SystemExit, match="--k 9: needs 1 <= k <= 8"):
        run_6b("--k", 9)


def test_keyword_and_sub_biome_vectors_are_kept_apart(world, run_6b):
    # In the toy world each Metalog sample has the same keyword and sub-biome vector, so keyword and
    # sub-biome columns could be swapped unnoticed. Real data: 22,618 distinct keyword vectors but 911
    # sub-biome ones. Here both gut studies share the sub-biome GUT ("human gut"): 4 distinct keyword
    # vectors, 3 distinct sub-biome vectors.
    rows = [(s, GUT if study.startswith("gut") else sb) for s, study, _, _, sb in METALOG if sb is not None]
    write_npz(world.sb_npz, rows)
    out = run_6b()
    # calibration: gut_study_1 <-> gut_study_2 = 0.5 * (keyword 0.8 + sub-biome 1) = 0.9; soil as before 0.8
    assert {s: sim for s, (_, sim) in read_held_out(out).items()} == {
        s: (0.9 if s.startswith("m_gut") else 0.8) for s in TRAINING_IDS}
    assert read_calibration(out)["threshold"] == THRESHOLD   # 5 % of [0.8 x 4, 0.9 x 4]
    # atlas: the nearest is still the identical study (gut_study_2 is at 0.9 from a_gut, 0.72 from a_border)
    assert read_result(out) == EXPECTED


def test_atlas_embeddings_need_not_be_unit_length(world, run_6b):
    # the .h5 vectors are normalised by the script (training .npz vectors too, by common.load_npz)
    write_h5(world.keywords_h5, KEYWORD_H5, scale=3.0)
    write_h5(world.sub_biomes_h5, SUB_BIOME_H5, scale=0.1)
    assert read_result(run_6b()) == EXPECTED


# ============================================================================= chunks, subsets, resume
def test_chunks_give_the_same_result_and_unused_chunks_get_no_part(run_6b):
    out = run_6b("--chunk_rows", 1)    # one keyword .h5 row per chunk
    assert read_result(out) == EXPECTED
    # rows 0-3 are used by atlas samples; row 4 by none, so chunk 4 is skipped and writes no part
    assert part_names(out) == ["rows_00000000.tsv.gz", "rows_00000001.tsv.gz",
                               "rows_00000002.tsv.gz", "rows_00000003.tsv.gz"]


def test_only_ids_scores_only_the_listed_samples(world, run_6b):
    ids = world.dir / "ids.txt"
    ids.write_text("a_soil\na_lab_gut\nnot_an_atlas_sample\n")    # unknown ids are ignored
    out = run_6b("--only_ids", ids)
    assert read_result(out) == {s: EXPECTED[s] for s in ["a_soil", "a_lab_gut"]}


def test_only_ids_matching_no_sample_writes_an_empty_table(world, run_6b):
    # No chunk has a sample, so no part is written: the output is the header line alone
    ids = world.dir / "ids.txt"
    ids.write_text("not_an_atlas_sample\n")
    out = run_6b("--only_ids", ids)
    assert part_names(out) == []
    assert read_result(out) == {}              # read_result also checks the 4 column names


def test_a_stopped_run_keeps_its_calibration_and_a_rerun_finishes(run_6b):
    # --max_seconds is checked before each chunk; calibration has already been written by then
    out = run_6b("--max_seconds", 1e-9)
    assert (out / "coverage_calibration.json").exists()
    assert part_names(out) == []
    assert not (out / "atlas_coverage.tsv.gz").exists()

    run_6b()                                   # same output dir, same settings: continues
    assert read_result(out) == EXPECTED


def test_a_rerun_reuses_finished_parts_and_reapplies_the_threshold(run_6b):
    out = run_6b()
    # Tamper with the finished part: if the rerun recomputed it, the tampering would disappear.
    part = out / "parts" / "rows_00000000.tsv.gz"
    table = pd.read_csv(part, sep="\t")
    table.loc[table["sample_id"] == "a_gut", "coverage_sim"] = 0.1234    # its in_coverage stays True
    table.to_csv(part, sep="\t", index=False, compression="gzip")

    result = read_result(run_6b())
    assert result["a_gut"] == (0.1234, "gut_study_1", False)   # part reused; flag recomputed: 0.1234 < 0.8
    assert result["a_gut_twin"] == EXPECTED["a_gut_twin"]


def test_a_rerun_with_other_settings_is_refused(run_6b):
    run_6b()
    with pytest.raises(SystemExit, match="use another --output_dir"):
        run_6b("--k", 2)                       # k is part of the saved settings
    run_6b("--k", 2, out="out_k2")             # a fresh output dir is fine


def test_a_resume_from_moved_inputs_continues(world, run_6b):
    # inputs are identified by content (md5), not path: the Cowork VM mounts them under a new path each session
    out = run_6b("--max_seconds", 1e-9)
    moved = world.dir / "moved"
    moved.mkdir()
    (moved / "training_set.tsv").write_bytes(open(world.samples, "rb").read())
    world.samples = str(moved / "training_set.tsv")
    run_6b()
    assert read_result(out) == EXPECTED


def test_a_resume_with_changed_training_samples_is_refused(world, run_6b):
    run_6b("--max_seconds", 1e-9)
    with open(world.samples, "a") as handle:
        handle.write("m_new\tgut_study_3\tintestine\t\t\n")   # one more row: another md5
    with pytest.raises(SystemExit, match="use another --output_dir"):
        run_6b()


def test_a_resume_with_another_chunk_size_is_refused(run_6b):
    # Parts are named by their first row only. Resuming a --chunk_rows 2 run (parts rows_00000000 = rows 0-1,
    # rows_00000002 = rows 2-3) with --chunk_rows 5000 would find "rows_00000000" done and skip rows 0-4,
    # silently losing a_lab_gut and a_border. --chunk_rows is therefore part of the saved settings.
    # (test_6b_coverage_mutations.py::test_chunk_rows_is_part_of_the_settings shows the loss without it.)
    run_6b("--chunk_rows", 2)
    with pytest.raises(SystemExit, match="use another --output_dir"):
        run_6b()
