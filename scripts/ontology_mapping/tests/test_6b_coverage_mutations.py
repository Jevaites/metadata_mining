"""Mutation tests for 6b_coverage.py: deliberate one-line bugs, and what each one breaks.

Why: a test suite can pass because the code is right, or because no test looks at the line that matters.
Each test below takes one line of 6b_coverage.py, breaks it in a plausible way (a "mutant": the source with
that line replaced, loaded as a separate module by load_mutant; the real script is not touched), runs the
same small scenario on the real module and on the mutant, and asserts:
    the real code gives the right answer, and the mutant gives a specific, documented wrong one.
So each test also documents why the line is written the way it is.

MUTANTS is the catalogue: name -> (the original text, its broken replacement). test_every_mutated_line_still_exists
fails when a line is rewritten, so the catalogue is kept in step with the script.

Scenarios reuse the toy data of the other two test files:
    X, STUDIES                    4 points in 2-D (test_6b_coverage_unit.py: s0/s1 study A, s2 B, s3 C)
    the toy Metalog / atlas       test_6b_coverage_integration.py (METALOG, ATLAS, EXPECTED, threshold 0.8)

One mutant SURVIVES on purpose (part_flag_strict): the flag written in each part file is recomputed when
the parts are merged, so that line cannot change the final output. Its test documents this.
"""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from step_loader import SCRIPTS_DIR, load_step
from test_6b_coverage_integration import (EXPECTED, GUT, METALOG, read_held_out, read_result, run_6b,  # noqa: F401
                                          world, write_groups, write_npz)
from test_6b_coverage_unit import STUDIES, X

cov = load_step("6b_coverage.py", "atlas_coverage")

MUTANTS = {
    # calibrate() / top_k_mean()
    "no_same_group_mask":          ("sim[g[s0:s0 + block, None] == g[None, :]] = -np.inf", "pass"),
    "minus_inf_kept":              ("held_out[~np.isfinite(held_out)] = np.nan", "pass"),
    "no_k_check":                  ("if not 1 <= k <= sim.shape[1]:", "if False:"),
    # main(): calibration
    "threshold_not_rounded":       ("threshold = round(threshold, 4)", "threshold = threshold"),
    "fold_groups_no_fallback":     ('.fillna(train["study_code"])', '.fillna("no_group")'),
    "chunk_rows_not_in_settings":  ('"n_train": len(train),\n                "chunk_rows": args.chunk_rows}',
                                    '"n_train": len(train)}'),
    # main(): scoring
    "no_sub_biome_not_special":    ("sim[has_sb] = 0.5 * (sim[has_sb] + sb_sim[sb_p[has_sb]][:, sb_col])",
                                    "sim = 0.5 * (sim + sb_sim[sb_p][:, sb_col])"),
    "sub_biomes_vs_keyword_vectors": ("sb_sim = normalize(handle[\"embeddings\"][:]).astype(np.float32) @ sb_u.T",
                                      "sb_sim = normalize(handle[\"embeddings\"][:]).astype(np.float32) @ kw_u.T"),
    "part_flag_strict":            ('"in_coverage": sim_out >= threshold}', '"in_coverage": sim_out > threshold}'),
    "finished_parts_recomputed":   ("if os.path.exists(part) or not in_chunk.any():", "if not in_chunk.any():"),
    # main(): merging the parts
    "final_flag_strict":           ('out["in_coverage"] = out["coverage_sim"].astype(float) >= threshold',
                                    'out["in_coverage"] = out["coverage_sim"].astype(float) > threshold'),
    "no_parts_not_handled":        ("if parts  # no part", "if True  # no part"),
}


def load_mutant(name):
    """6b_coverage.py with MUTANTS[name] applied, as a separate module (the real one is untouched)."""
    old, new = MUTANTS[name]
    source = (SCRIPTS_DIR / "6b_coverage.py").read_text()
    assert source.count(old) == 1, f"mutant {name!r}: {old!r} occurs {source.count(old)} times in 6b_coverage.py"
    module = types.ModuleType(f"atlas_coverage_{name}")
    module.__file__ = str(SCRIPTS_DIR / "6b_coverage.py")
    exec(compile(source.replace(old, new), f"6b_coverage.py ({name})", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_mutated_line_still_exists(name):
    load_mutant(name)  # asserts the original text occurs exactly once in 6b_coverage.py


# ============================================================================= calibrate() / top_k_mean()
def test_samples_of_the_same_study_are_masked():
    # Without the mask, each sample's nearest training sample is ITSELF (similarity 1): every held-out
    # similarity is 1 and the threshold would flag almost every atlas sample.
    real = cov.calibrate(X, X, STUDIES, k=1, quantile=0.05)[2]
    broken = load_mutant("no_same_group_mask").calibrate(X, X, STUDIES, k=1, quantile=0.05)[2]
    assert real == pytest.approx([0.8, 0.8, 0.8, 0.6])
    assert broken.tolist() == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_samples_without_k_neighbours_in_other_groups_are_left_out_of_the_quantiles():
    # k=3: s0 and s1 have only 2 samples in other studies -> their top 3 contains a masked -inf.
    # Kept as -inf, np.quantile interpolates with -inf and returns NaN; in main(), `coverage_sim >= NaN`
    # is False for every atlas sample: the whole atlas would be flagged out of coverage, without an error.
    real = cov.calibrate(X, X, STUDIES, k=3, quantile=0.0)[0]
    broken = load_mutant("minus_inf_kept").calibrate(X, X, STUDIES, k=3, quantile=0.0)[0]
    assert real == pytest.approx(0.2)
    assert np.isnan(broken)


def test_k_larger_than_the_number_of_training_samples_is_an_error():
    # np.partition(sim, n - k) accepts a negative kth: with 2 training samples and k=3, the "mean of the
    # 3 nearest" silently becomes the mean of the 2 that exist.
    sim = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="only 2 training samples"):
        cov.top_k_mean(sim, k=3)
    assert load_mutant("no_k_check").top_k_mean(sim, k=3)[0].tolist() == [1.5]


# ============================================================================= main(): calibration
def test_the_threshold_is_rounded_like_coverage_sim(run_6b):
    # coverage_sim is rounded to 4 decimals. The held-out similarity 0.8 is computed in float32 as
    # 0.800000011920929; unrounded, it puts a_border (coverage_sim 0.8, exactly as close to Metalog as the
    # worst held-out Metalog sample) just below the threshold.
    real = run_6b(out="real")
    broken = run_6b(out="broken", module=load_mutant("threshold_not_rounded"))
    assert json.loads((real / "coverage_calibration.json").read_text())["threshold"] == 0.8
    assert json.loads((broken / "coverage_calibration.json").read_text())["threshold"] > 0.8
    assert read_result(real)["a_border"] == (0.8, "gut_study_1", True)
    assert read_result(broken)["a_border"] == (0.8, "gut_study_1", False)


def test_samples_missing_from_fold_groups_keep_their_own_study(world, run_6b):
    # project_groups.tsv lists the gut samples only. The soil samples must fall back to their study codes;
    # with one shared fallback group, soil_study_1 and soil_study_2 merge and lose each other as neighbours.
    groups = write_groups(world, dict.fromkeys(["m_gut1a", "m_gut1b", "m_gut2a", "m_gut2b"], "gut_consortium"))
    real = run_6b("--fold_groups", groups, out="real")
    broken = run_6b("--fold_groups", groups, out="broken", module=load_mutant("fold_groups_no_fallback"))
    soil = ["m_soil1a", "m_soil2a"]
    assert [read_held_out(real)[s][1] for s in soil] == [0.8, 0.8]     # nearest: the other soil study
    assert [read_held_out(broken)[s][1] for s in soil] == [0.0, 0.36]  # nearest: a gut sample


def test_chunk_rows_is_part_of_the_settings(run_6b):
    # A --chunk_rows 2 run stopped after its first chunk (part rows_00000000 = keyword rows 0-1), resumed
    # with the default --chunk_rows 5000 (one chunk, rows 0-4, ALSO named rows_00000000).
    def stopped_then_resumed(out, module):
        out_dir = run_6b("--chunk_rows", 2, out=out, module=module)
        (out_dir / "parts" / "rows_00000002.tsv.gz").unlink()    # as if stopped before the 2nd chunk
        (out_dir / "atlas_coverage.tsv.gz").unlink()
        run_6b(out=out, module=module)
        return out_dir

    with pytest.raises(SystemExit, match="use another --output_dir"):
        stopped_then_resumed("real", cov)                         # real: the resume is refused
    broken = stopped_then_resumed("broken", load_mutant("chunk_rows_not_in_settings"))
    # mutant: "rows_00000000" exists, so the whole 0-4 chunk is skipped; rows 2 and 3 are never scored
    assert set(EXPECTED) - set(read_result(broken)) == {"a_lab_gut", "a_border"}


# ============================================================================= main(): scoring
def test_a_sample_without_sub_biome_uses_the_keyword_cosine_alone(run_6b):
    # sb_rows = -1 means "no sub-biome". Used as an index, -1 silently picks the LAST sub-biome row
    # ("laboratory"), so a plain gut sample is averaged with an unrelated sub-biome: 0.5 * (1 + 0).
    real = read_result(run_6b(out="real"))
    broken = read_result(run_6b(out="broken", module=load_mutant("no_sub_biome_not_special")))
    assert real["a_gut_no_sb"] == (1.0, "gut_study_1", True)
    assert broken["a_gut_no_sb"] == (0.5, "gut_study_1", False)


def test_atlas_sub_biomes_are_compared_with_training_sub_biomes(world, run_6b):
    # Both gut studies share the sub-biome GUT (as in test_keyword_and_sub_biome_vectors_are_kept_apart):
    # 4 distinct keyword, 3 distinct sub-biome vectors, sorted [SOIL_WET, SOIL, GUT_WET, GUT] and
    # [SOIL_WET, SOIL, GUT]. The mutant reads the gut sub-biome column (2) of the keyword table: GUT_WET,
    # so a_gut gets 0.5 * (1 + 0.8) = 0.9 instead of 1.
    write_npz(world.sb_npz, [(s, GUT if study.startswith("gut") else sb)
                             for s, study, _, _, sb in METALOG if sb is not None])
    real = read_result(run_6b(out="real"))
    broken = read_result(run_6b(out="broken", module=load_mutant("sub_biomes_vs_keyword_vectors")))
    assert real["a_gut"] == (1.0, "gut_study_1", True)
    assert broken["a_gut"][0] == 0.9


def test_the_flag_in_the_part_files_does_not_reach_the_output(run_6b):
    # SURVIVING MUTANT, on purpose: the merge recomputes in_coverage from coverage_sim, so a wrong flag in
    # a part file cannot change atlas_coverage.tsv.gz (the part file itself does differ).
    real = run_6b(out="real")
    broken = run_6b(out="broken", module=load_mutant("part_flag_strict"))
    assert read_result(real) == read_result(broken) == EXPECTED

    def part_flag(out_dir):
        part = pd.read_csv(out_dir / "parts" / "rows_00000000.tsv.gz", sep="\t").set_index("sample_id")
        return bool(part.loc["a_border", "in_coverage"])
    assert part_flag(real) is True
    assert part_flag(broken) is False


def test_finished_parts_are_not_recomputed(run_6b):
    # A part is overwritten by hand; a resumed run must keep it (resume = skip what is done). The mutant
    # recomputes it: still correct output here, but a resumed 3.4M-sample run would start over.
    def tampered_then_rerun(out, module):
        out_dir = run_6b(out=out, module=module)
        part = out_dir / "parts" / "rows_00000000.tsv.gz"
        table = pd.read_csv(part, sep="\t")
        table.loc[table["sample_id"] == "a_gut", "coverage_sim"] = 0.1234
        table.to_csv(part, sep="\t", index=False, compression="gzip")
        return read_result(run_6b(out=out, module=module))

    assert tampered_then_rerun("real", cov)["a_gut"][0] == 0.1234                                   # kept
    assert tampered_then_rerun("broken", load_mutant("finished_parts_recomputed"))["a_gut"][0] == 1.0  # redone


# ============================================================================= main(): merging the parts
def test_a_sample_exactly_at_the_threshold_is_in_coverage(run_6b):
    # a_border's coverage_sim is 0.8 = the threshold: ">=" keeps it, ">" flags it
    real = read_result(run_6b(out="real"))
    broken = read_result(run_6b(out="broken", module=load_mutant("final_flag_strict")))
    assert real["a_border"][2] is True
    assert broken["a_border"][2] is False


def test_no_atlas_sample_to_score_gives_an_empty_table(world, run_6b):
    ids = world.dir / "ids.txt"
    ids.write_text("not_an_atlas_sample\n")
    assert read_result(run_6b("--only_ids", ids, out="real")) == {}
    with pytest.raises(ValueError, match="No objects to concatenate"):   # pd.concat of zero part files
        run_6b("--only_ids", ids, out="broken", module=load_mutant("no_parts_not_handled"))
