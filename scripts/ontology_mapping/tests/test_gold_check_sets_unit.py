"""Unit tests for experiments/gold_check_sets.py: the set answers scored against coarse gold biomes."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import gold_check_sets as g  # noqa: E402

MAJORITY = {"terr": "soil", "animal": "animal", "aquatic": "water", "crop": "plant"}
COMPATIBLE = {"terr": {"soil", "plant"}, "animal": {"animal"}, "aquatic": {"water"}, "crop": {"plant", "soil"}}
MIX = {"animal": 0.5, "plant": 0.2, "soil": 0.1, "water": 0.1, "other": 0.1}


def test_answer_terms():
    assert g.answer_terms("", "") == []
    assert g.answer_terms("terr", "") == ["terr"]
    assert g.answer_terms("terr", "animal:0.3;aquatic:0.1") == ["terr", "animal", "aquatic"]


def test_term_scores_any_split_and_decisive():
    # a plant sample answered {terr, animal}: terr is compatible (lenient), not the majority (strict)
    assert g.term_scores(["terr", "animal"], "plant", MAJORITY, COMPATIBLE) == (True, 0.5, False, 0.0, False)
    # {terr, crop}: majorities soil / plant differ -> not decisive; both compatible with soil
    assert g.term_scores(["terr", "crop"], "soil", MAJORITY, COMPATIBLE) == (True, 1.0, True, 0.5, False)
    assert g.term_scores(["animal"], "animal", MAJORITY, COMPATIBLE)[4] is None  # single term: no decisiveness


def table():
    rows = [  # sample, gold, answer, alternatives, kind, backoff, flag, top-1, runner-up
        ("s1", "animal", "animal", "", "single", "animal", "False", "animal", ""),
        ("s2", "plant", "terr", "animal:0.4", "set", "", "True", "animal", "terr"),
        ("s3", "water", "", "", "none", "", "False", "aquatic", ""),
        ("s4", "soil", "animal", "", "single", "animal", "False", "animal", ""),
    ]
    cols = ["sample_id", "gold_biome", "biome_answer", "biome_alternatives", "biome_answer_kind", "biome_backoff",
            "biome_ambiguous_branches", "biome", "biome_runner_up"]
    return pd.DataFrame(rows, columns=cols)


def test_summary_of_one_slot():
    t = g.score_slot(table(), "biome", MAJORITY, COMPATIBLE)
    s = g.summarise(t, MIX)
    assert (s["answered"], s["backoff_answered"]) == (0.75, 0.5)
    assert s["lenient_any"] == pytest.approx(2 / 3, abs=1e-3)       # s1, s2 right; s4 wrong
    assert s["lenient_split"] == pytest.approx(0.5, abs=1e-3)       # (1 + .5 + 0) / 3
    assert s["backoff_lenient"] == 0.5                               # s1 right, s4 wrong
    assert s["set"] == 0.25 and s["set_lenient_any"] == 1.0 and s["set_decisive"] == 0.0
    assert s["on_backoff_abstentions"] == {"n": 2, "answered": 0.5, "lenient_any": 1.0, "lenient_split": 0.5}
    assert s["flagged"] == {"n": 1, "top1_lenient": 0.0, "runner_up_lenient": 1.0, "either": 1.0}
    # per class: animal 1 / 1, plant 1 / -, soil 0 / 0, water unanswered -> re-weighted with MIX
    assert s["reweighted"]["lenient_any"] == pytest.approx(0.5 * 1 + 0.2 * 1 + 0.1 * 0, abs=1e-3)
