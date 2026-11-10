"""main() of experiments/set_valued.py end to end on the toy world of 5_evaluate.py's tests (step5_toy.py): 15 terms,
marine / freshwater / soil / lake / human samples in 2-sample-study pairs, 2 folds, real on-disk formats, plus a
3-term ENVO OBO for the facet rule.

MIXED_SAMPLES (added here) sit halfway between sea and fresh water; two of their studies label them marine, two
freshwater (an estuary-like disagreement), so some probabilities are split and some answers are sets. Which
samples get which set depends on the 2-fold training data; the exact decision logic is pinned on hand-made
probabilities in test_set_valued_unit.py::TestDecodeSlot.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import set_valued as sv  # noqa: E402
from common import load_terms, term_ancestors  # noqa: E402
from step5_toy import HUMAN_SAMPLES, LAKE_SAMPLES, SAMPLES, write_project  # noqa: E402

MIXED_SAMPLES = [(f"mixed_{c}{k}_s{i}", f"mixed_{c}{k}", cls, (0.7, 0.7, 0.05 * i, 0.0, 0.0))
                 for c, cls in (("m", "marine"), ("f", "freshwater")) for k in (1, 2) for i in (1, 2, 3)]
MARINE, FRESHWATER = "ENVO_marine_biome", "ENVO_freshwater_biome"

OBO = """format-version: 1.2

[Term]
id: ENVO:river
name: river
relationship: RO:0001025 ENVO:fresh_water

[Term]
id: ENVO:fresh_water
name: fresh water

[Typedef]
id: RO:0001025
name: located in
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("set_valued")
    paths = write_project(tmp, SAMPLES + LAKE_SAMPLES + HUMAN_SAMPLES + MIXED_SAMPLES)
    (tmp / "envo.obo").write_text(OBO)
    out = tmp / "out"
    sv.main(["--ontology_terms", paths["ontology_terms.tsv"], "--keywords", paths["features.npz"],
             "--sub_biomes", paths["features.npz"], "--term_vectors", paths["term_vectors.h5"],
             "--samples", paths["training_set.tsv"], "--folds", "2", "--envo_obo", str(tmp / "envo.obo"),
             "--bootstrap", "20", "--output_dir", str(out)])
    terms = load_terms(paths["ontology_terms.tsv"])
    return {"out": out, "anc": term_ancestors(terms), "terms": set(terms["term_id"]),
            "summary": pd.read_csv(out / "summary.tsv", sep="\t"),
            "rules": pd.read_csv(out / "rule_scores.tsv", sep="\t"),
            "decisions": pd.read_csv(out / "decisions.tsv.gz", sep="\t", dtype=str, keep_default_na=False),
            "settings": json.loads((out / "settings.json").read_text())}


METHODS = ["top1", "backoff", "conformal_m1", "ev_m1", "conformal_m2", "ev_m2", "conformal_m3", "ev_m3",
           "backoff+set", "backoff+set@0.8", "ev1+ev2set"]


def test_every_method_and_slot_is_reported(run):
    s = run["summary"]
    assert sorted(s["slot"].unique()) == ["biome", "feature", "material"]
    for slot, g in s.groupby("slot"):
        assert g["method"].tolist() == METHODS, slot
    top1 = s[s["method"] == "top1"]
    assert (top1["answered"] == 1).all() and top1["set_credit"].isna().all()


def test_one_decision_per_labelled_sample_slot_and_method(run):
    d = run["decisions"]
    n_labelled = {"biome": 33, "feature": 39, "material": 39}  # human samples have no biome label
    for (slot, method), g in d.groupby(["slot", "method"]):
        assert len(g) == n_labelled[slot] and g["sample_id"].is_unique, (slot, method)
    assert set(d["method"]) == set(METHODS)


def test_answers_are_terms_and_never_hold_a_term_with_its_ancestor(run):
    anc = run["anc"]
    for answer in run["decisions"]["answer"]:
        terms = [t for t in answer.split("||") if t]
        assert set(terms) <= run["terms"]
        for a in terms:
            for b in terms:
                assert a == b or a not in anc.get(b, ()), answer


def test_some_answers_are_sets_and_only_from_set_methods(run):
    d = run["decisions"]
    is_set = d["answer"].str.contains("||", regex=False)
    assert is_set[(d["slot"] == "biome") & (d["method"] == "conformal_m2")].any()  # the toy does produce sets
    assert not is_set[d["method"].isin(["top1", "backoff", "conformal_m1", "ev_m1"])].any()
    assert d.loc[is_set, "answer"].str.count("[|][|]").max() <= 2  # at most 3 terms (--max_terms 3)


def test_summary_matches_the_decisions(run):
    d, s, anc = run["decisions"], run["summary"], run["anc"]
    for (slot, method), g in d.groupby(["slot", "method"]):
        answers = [a.split("||") if a else [] for a in g["answer"]]
        answered = np.array([bool(a) for a in answers])
        right = np.array([any(sv.covers(t, gold, anc) for t in a) for a, gold in zip(answers, g["gold"])])
        row = s[(s["slot"] == slot) & (s["method"] == method)].iloc[0]
        assert row["answered"] == pytest.approx(answered.mean(), abs=1e-4), (slot, method)
        if answered.any():
            assert row["accuracy"] == pytest.approx(right[answered].mean(), abs=1e-4), (slot, method)
        assert row["exact"] == pytest.approx(np.mean([a == [gold] for a, gold in zip(answers, g["gold"])]), abs=1e-4)
        assert row["sets"] == pytest.approx(np.mean([len(a) > 1 for a in answers]), abs=1e-4)


def test_rule_scores_of_top1_are_plain_accuracy(run):
    r, d = run["rules"], run["decisions"]
    for slot in ["biome", "feature", "material"]:
        g = d[(d["slot"] == slot) & (d["method"] == "top1")]
        row = r[(r["slot"] == slot) & (r["method"] == "top1") & (r["convention"] == "any") & (r["measure"] == "exact")]
        assert row["micro"].iloc[0] == pytest.approx((g["answer"] == g["gold"]).mean(), abs=1e-4)
        # single answers: both set conventions agree
        both = r[(r["slot"] == slot) & (r["method"] == "top1")].pivot(index="measure", columns="convention", values="micro")
        assert (both["any"] == both["split"]).all()


def test_split_never_exceeds_any(run):
    r = run["rules"].pivot_table(index=["slot", "method", "measure"], columns="convention", values="micro")
    assert (r["split"] <= r["any"] + 1e-9).all()


def test_settings_record_a_threshold_per_fold(run):
    st = run["settings"]
    assert set(st["temperatures"]) == {"biome", "feature", "material"}
    for slot, per_method in st["thresholds"].items():
        assert set(per_method) == set(METHODS) - {"top1"}
        for method, chosen in per_method.items():
            assert set(chosen) == {"0", "1"}, (slot, method)


def test_fallback_keeps_the_back_off_answers(run):
    d = run["decisions"].pivot_table(index=["slot", "sample_id"], columns="method", values="answer", aggfunc="first")
    answered = d["backoff"] != ""
    assert (d.loc[answered, "backoff+set"] == d.loc[answered, "backoff"]).all()
    assert (d.loc[d["ev_m1"] != "", "ev1+ev2set"] == d.loc[d["ev_m1"] != "", "ev_m1"]).all()


def test_no_rules_mode_needs_no_obo(tmp_path):
    paths = write_project(tmp_path, SAMPLES + LAKE_SAMPLES)
    sv.main(["--ontology_terms", paths["ontology_terms.tsv"], "--keywords", paths["features.npz"],
             "--sub_biomes", paths["features.npz"], "--term_vectors", paths["term_vectors.h5"],
             "--samples", paths["training_set.tsv"], "--folds", "2", "--envo_obo", str(tmp_path / "missing.obo"),
             "--no_rules", "--bootstrap", "0", "--max_terms", "1", "--output_dir", str(tmp_path / "out")])
    out = tmp_path / "out"
    assert not (out / "rule_scores.tsv").exists()
    s = pd.read_csv(out / "summary.tsv", sep="\t")
    # max_terms 1: no set methods, no fallbacks, no sets
    assert sorted(s["method"].unique()) == sorted(["top1", "backoff", "conformal_m1", "ev_m1"])
    assert (s["sets"] == 0).all()
