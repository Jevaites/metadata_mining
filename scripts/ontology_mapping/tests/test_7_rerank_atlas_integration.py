"""Integration tests for 7_rerank_atlas.py: the five commands, run on a toy world through main().

How step 7 is used (its docstring has the real paths):

    7_rerank_atlas.py fit     --pilot P.jsonl --responses R.jsonl --calibration C.json --samples TRAIN --output params.json
    7_rerank_atlas.py build   --params params.json --atlas_dir ATLAS --sample_info sample.info.gz --samples TRAIN
                              --train_vectors kw.npz sb.npz --keywords_h5 kw.h5 --sub_biomes_h5 sb.h5 --work_dir WORK
    7_rerank_atlas.py submit  --work_dir WORK [--limit N]           (OpenAI Batch API; here: FakeOpenAI)
    7_rerank_atlas.py collect --work_dir WORK                       (rerun until every batch is collected)
    7_rerank_atlas.py apply   --params params.json --atlas_dir ATLAS --work_dir WORK   -> WORK/atlas_final.tsv.gz

Files passed between the commands:
    fit     -> params.json          per slot: use_llm, w, T, tau, gate_p (+ the pilot evaluation)
    build   -> requests.jsonl       one LLM request per distinct (slot, metadata text, candidates)
               members.tsv.gz       request_id -> the atlas samples it stands for
    submit  -> batches.json         the batches sent, and whether they were collected
    collect -> responses.jsonl      one answer per request (the letter and its log-probabilities)
    apply   -> atlas_final.tsv.gz   <slot>_final / _final_label / _final_p / _final_source per sample

================================================================================ the toy world
Ontology: step7_toy.py (aquatic biome > marine biome > ocean biome; aquatic biome > freshwater biome > lake biome).

Training set (Metalog, curated). Only the keyword vector matters: every sub-biome vector is [1, 0].
    sample  biome       feature  text                       keyword vector
    T1      marine      sea      seawater, North Sea        [1, 0]       (0 degrees)
    T2      marine      sea      coral reef water           [0.8, 0.6]   (37 degrees)
    T3      freshwater  pond     lake water, Lake Geneva    [0, 1]       (90 degrees)
    T4      freshwater  river    river water                [0.6, 0.8]   (53 degrees)
    T5      aquatic     -        water of unknown origin    [0.7, 0.7]   (45 degrees)

sample.info.gz is written in the real layout (empty first line, empty line after each record); S2/S3 also carry
'Missing', date and accession lines that common.record_to_text drops.

Atlas (6_predict_atlas.py output, biome slot; the reranker's gate is biome_p < 0.6; an abstaining back-off is
"" with p 0.0, as in the real file):
    sample  biome_p  biome_candidates                         biome_backoff  metadata (sample.info)        keyword vector
    S1      0.90     marine:0.9;ocean:0.06;aquatic:0.04       marine 0.96    open ocean, 35 PSU            [1, 0]
    S2      0.50     marine:0.5;freshwater:0.25;aquatic:0.25  aquatic 1.0    lake shore, 0.1 PSU           [0, 1]
    S3      0.50     (as S2)                                  aquatic 1.0    (as S2: a replicate)          [0, 1]
    S4      0.50     (as S2)                                  aquatic 1.0    -- missing from sample.info --
    S5      0.40     marine:0.4;lake:0.3;aquatic:0.3          aquatic 1.0    estuary                       [1, 0]
    S6      0.60     marine:0.6;freshwater:0.3;aquatic:0.1    marine 0.6     beach, 35 PSU                 [0.6, 0.8]
What should happen to each:
    S1  confident (0.9 >= 0.6): not sent to the LLM, keeps its back-off answer
    S2  gated; the LLM reads "0.1 PSU" and says freshwater -> reranked
    S3  same text and candidates as S2: no request of its own, gets S2's answer
    S4  gated but no metadata text: no request -> backoff_no_llm_answer
    S5  gated; the fake LLM has no rule for "estuary" and the request fails -> backoff_no_llm_answer
    S6  exactly at the gate (0.6 is not < 0.6): not gated
The fake LLM (step7_toy.keyword_llm): "0.1 PSU" -> freshwater biome, "35 PSU" -> marine biome.
"""

import json
import os
import sys

import numpy as np
import pytest

from step7_toy import (FakeOpenAI, keyword_llm, load_rerank_atlas, output_file_client, pilot_response, read_jsonl,
                       read_tsv, write_h5, write_jsonl, write_sample_info, write_terms, write_tsv, write_vectors_npz, LABEL)

rr = load_rerank_atlas()

# The LLM of the stage tests: 0.6 on the label its rule picks, the rest split evenly over the other options.
ATLAS_LLM = keyword_llm([("0.1 PSU", "freshwater biome"), ("35 PSU", "marine biome")], p=0.6)

# Hand-written parameters for the stage tests (fit is tested on its own), chosen so that apply can be
# checked with pencil and paper: w = 1 and T = 1 make the fusion a plain product of probabilities.
PARAMS = {"slots": {"biome": {"use_llm": True, "w": 1.0, "T": 1.0, "tau": 0.4, "gate_p": 0.6},
                    "feature": {"use_llm": False, "w": 0, "T": 1.0, "tau": 0.5, "gate_p": 0.5}}}


# ============================================================================= the toy world on disk
TRAINING = [
    # sample_id, study, biome, feature, material, text, keyword vector
    ("T1", "studyA", "marine", "sea", "sea_water", "seawater, North Sea", [1, 0]),
    ("T2", "studyB", "marine", "sea", "sea_water", "coral reef water", [0.8, 0.6]),
    ("T3", "studyC", "freshwater", "pond", "fresh_water", "lake water, Lake Geneva", [0, 1]),
    ("T4", "studyD", "freshwater", "river", "fresh_water", "river water", [0.6, 0.8]),
    ("T5", "studyE", "aquatic", "", "", "water of unknown origin", [0.7, 0.7]),
]
S2_CANDIDATES = "marine:0.5;freshwater:0.25;aquatic:0.25"
ATLAS = [
    # sample_id, biome_p, biome_candidates, biome_backoff, biome_backoff_p, feature_backoff, keyword-h5 row, sub-biome-h5 row
    ("S1", 0.90, "marine:0.9;ocean:0.06;aquatic:0.04", "marine", 0.96, "sea", 0, -1),
    ("S2", 0.50, S2_CANDIDATES, "aquatic", 1.0, "pond", 1, 0),
    ("S3", 0.50, S2_CANDIDATES, "aquatic", 1.0, "pond", 1, 0),
    ("S4", 0.50, S2_CANDIDATES, "aquatic", 1.0, "pond", 1, -1),
    ("S5", 0.40, "marine:0.4;lake:0.3;aquatic:0.3", "aquatic", 1.0, "", 0, -1),  # feature: step 6 abstained
    ("S6", 0.60, "marine:0.6;freshwater:0.3;aquatic:0.1", "marine", 0.6, "sea", 2, -1),
]
KEYWORDS_H5 = [[1, 0], [0, 1], [0.6, 0.8]]  # rows of the atlas keyword file (ATLAS's keyword-h5 row)
SUB_BIOMES_H5 = [[1, 0]]                   # -1 in ATLAS = the sample has no sub-biome
S2_LINES = ["sample_env_biome=lake shore", "sample_salinity=0.1 PSU", "sample_host=Missing",  # real-style lines; the
            "sample_collection_date=Aug-2013", "experiments=SRX1629903", "study=SRP046387"]  # last 4 are dropped
SAMPLE_INFO = {
    "S1": ["sample_env_biome=open ocean", "sample_salinity=35 PSU"],
    "S2": S2_LINES,
    "S3": S2_LINES,
    "S5": ["sample_env_biome=estuary"],
    "S6": ["sample_env_biome=beach", "sample_salinity=35 PSU"],
}
S2_TEXT = "env_biome: lake shore; salinity: 0.1 PSU"  # what common.record_to_text makes of S2_LINES
S5_TEXT = "env_biome: estuary"


def make_world(d, atlas=ATLAS, sample_info=SAMPLE_INFO, topk=3):
    """Write every input of build / apply into the directory d; -> {name: path}."""
    d.mkdir(exist_ok=True)
    w = {"terms": write_terms(d / "ontology_terms.tsv")}
    w["samples"] = write_tsv(d / "training_set.tsv.gz", [
        {"sample_id": s, "study_code": st, "biome": b, "feature": f, "material": m, "text": t} for s, st, b, f, m, t, _ in TRAINING])
    w["kw_npz"] = write_vectors_npz(d / "keywords.npz", {s: v for s, *_, v in TRAINING})
    w["sb_npz"] = write_vectors_npz(d / "sub_biomes.npz", {s: [1, 0] for s, *_ in TRAINING})
    w["kw_h5"] = write_h5(d / "keywords.h5", KEYWORDS_H5)
    w["sb_h5"] = write_h5(d / "sub_biomes.h5", SUB_BIOMES_H5)
    w["sample_info"] = write_sample_info(d / "sample.info.gz", sample_info)
    atlas_dir = d / "atlas"  # 6_predict_atlas.py --output_dir
    atlas_dir.mkdir()
    w["atlas_dir"] = str(atlas_dir)
    write_tsv(atlas_dir / "atlas_predictions.tsv.gz", [
        {"sample_id": s, "biome_p": p, "biome_candidates": c, "biome_backoff": b, "biome_backoff_p": bp,
         "feature_p": 0.9, "feature_candidates": "", "feature_backoff": f, "feature_backoff_p": 0.9 if f else 0.0,
         "material_p": 0.8, "material_candidates": "", "material_backoff": "sea_water", "material_backoff_p": 0.8}
        for s, p, c, b, bp, f, _, _ in atlas])
    np.savez(atlas_dir / "index.npz", sample_ids=np.array([a[0] for a in atlas]),
             kw_rows=np.array([a[6] for a in atlas]), sb_rows=np.array([a[7] for a in atlas]))
    np.savez(atlas_dir / "model.npz", biome_classes=np.array(["aquatic", "freshwater", "lake", "marine", "ocean"]),
             feature_classes=np.array(["pond", "river", "sea"]), material_classes=np.array(["fresh_water", "sea_water"]))
    json.dump({"topk": topk}, open(atlas_dir / "run_settings.json", "w"))
    w["work"] = str(d / "work")
    return w


def write_params(d, params=PARAMS, **biome):
    """params.json as fit writes it (only the keys build and apply read); biome=... overrides biome settings."""
    params = json.loads(json.dumps(params))
    params["slots"]["biome"].update(biome)
    p = d / "params.json"
    json.dump(params, open(p, "w"))
    return str(p)


def build_argv(w, params, *extra):
    return ["build", "--params", params, "--atlas_dir", w["atlas_dir"], "--sample_info", w["sample_info"],
            "--samples", w["samples"], "--train_vectors", w["kw_npz"], w["sb_npz"], "--keywords_h5", w["kw_h5"],
            "--sub_biomes_h5", w["sb_h5"], "--ontology_terms", w["terms"], "--work_dir", w["work"], *extra]


def apply_argv(w, params):
    return ["apply", "--params", params, "--atlas_dir", w["atlas_dir"], "--work_dir", w["work"], "--ontology_terms", w["terms"]]


def final_table(w, slot="biome"):
    """atlas_final.tsv.gz as [(sample, <slot>_final, <slot>_final_p, <slot>_final_source)]."""
    df = read_tsv(os.path.join(w["work"], "atlas_final.tsv.gz"))
    return [(s, f, float(p), src) for s, f, p, src in
            zip(df["sample_id"], df[f"{slot}_final"], df[f"{slot}_final_p"], df[f"{slot}_final_source"])]


def requests_by_text(w):
    return {r["text"]: r for r in read_jsonl(os.path.join(w["work"], "requests.jsonl"))}


def members_by_text(w):
    """{request text: sorted samples it stands for}, from members.tsv.gz and requests.jsonl."""
    text_of = {r["request_id"]: r["text"] for r in read_jsonl(os.path.join(w["work"], "requests.jsonl"))}
    m = read_tsv(os.path.join(w["work"], "members.tsv.gz"))
    out = {}
    for rid, sid in zip(m["request_id"], m["sample_id"]):
        out.setdefault(text_of[rid], []).append(sid)
    return {t: sorted(s) for t, s in out.items()}


@pytest.fixture
def run(monkeypatch):
    """run("build", "--params", ...) = `python 7_rerank_atlas.py build --params ...`."""
    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["7_rerank_atlas.py", *[str(a) for a in argv]])
        rr.main()
    return run


@pytest.fixture
def openai(monkeypatch):
    """openai(llm) replaces the OpenAI client of submit / collect by a FakeOpenAI answering with llm."""
    def install(llm):
        fake = FakeOpenAI(llm)
        monkeypatch.setattr(rr, "client_for", lambda args: fake)
        return fake
    return install


# ============================================================================= 1. the basic workflow
def test_build_submit_collect_apply_on_the_toy_atlas(tmp_path, run, openai):
    w = make_world(tmp_path)
    params = write_params(tmp_path)

    # ---- build: which samples get an LLM request, with which candidates and examples
    run(*build_argv(w, params))
    assert members_by_text(w) == {S2_TEXT: ["S2", "S3"],  # S3 shares S2's request (same text and candidates)
                                  S5_TEXT: ["S5"]}        # S1, S6 not gated; S4 has no metadata text
    s2 = requests_by_text(w)[S2_TEXT]
    assert s2["slot"] == "biome" and s2["request_id"].startswith("biome-")
    assert s2["sample_id"] == s2["request_id"]  # the pilot's key, so the pilot's parse() works unchanged
    assert [c["term_id"] for c in s2["candidates"]] == ["marine", "freshwater", "aquatic"]  # S2_CANDIDATES' order
    assert [c["label"] for c in s2["candidates"]] == ["marine biome", "freshwater biome", "aquatic biome"]
    assert s2["base_probs"] == [0.5, 0.25, 0.25] and s2["topk_mass"] == 1.0
    assert sorted(s2["option_order"]) == [0, 1, 2]  # letters shuffled (seeded by the request id)
    # Example per candidate = the most similar training sample labelled with it. S2 points at 90 degrees:
    #   marine:     T2 (37 degrees) is closer than T1 (0 degrees)
    #   freshwater: T3 (90 degrees) is closer than T4 (53 degrees)
    #   aquatic:    T5, the only one
    assert {c["term_id"]: c["example"] for c in s2["candidates"]} == {
        "marine": "coral reef water", "freshwater": "lake water, Lake Geneva", "aquatic": "water of unknown origin"}
    # S5 points at 0 degrees: marine -> T1; 'lake' has no training sample at all -> no example
    s5 = requests_by_text(w)[S5_TEXT]
    assert {c["term_id"]: c["example"] for c in s5["candidates"]} == {
        "marine": "seawater, North Sea", "lake": "", "aquatic": "water of unknown origin"}

    # ---- submit: both requests go out in one batch
    api = openai(ATLAS_LLM)
    run("submit", "--work_dir", w["work"])
    assert len(api.sent) == 1
    assert sorted(line["custom_id"] for line in api.sent[0]) == sorted([s2["request_id"], s5["request_id"]])
    sent = {line["custom_id"]: line["body"] for line in api.sent[0]}
    assert sent[s2["request_id"]] == rr.body_for(s2, "gpt-4.1-mini")  # the v2 prompt of the request, as in the pilot

    # ---- collect: S2's request is answered, S5's fails (the fake LLM has no rule for an estuary)
    run("collect", "--work_dir", w["work"])
    responses = read_jsonl(os.path.join(w["work"], "responses.jsonl"))
    assert [r["request_id"] for r in responses] == [s2["request_id"]]
    letter_of = {c["term_id"]: "ABC"[s2["option_order"].index(i)] for i, c in enumerate(s2["candidates"])}
    assert responses[0]["content"] == letter_of["freshwater"]  # the letter under which freshwater was shown
    assert json.load(open(os.path.join(w["work"], "batches.json")))["batches"][0]["collected"] is True

    # ---- apply: S2 by hand, with w = 1, T = 1 (a plain product), tau = 0.4:
    #   candidates          marine   freshwater  aquatic
    #   base                0.5      0.25        0.25
    #   LLM                 0.2      0.6         0.2
    #   base x LLM          0.10     0.15        0.05     (sum 0.30)
    #   fused (x mass 1.0)  1/3      1/2         1/6
    #   back-off: top = freshwater, its summed probability 0.5 >= tau 0.4 -> 'freshwater', p 0.5
    run(*apply_argv(w, params))
    assert final_table(w) == [
        ("S1", "marine", 0.96, "backoff"),                  # not gated: step 6's back-off answer
        ("S2", "freshwater", 0.5, "rerank"),                # the LLM overturned the base model's 'marine'
        ("S3", "freshwater", 0.5, "rerank"),                # S2's answer
        ("S4", "aquatic", 1.0, "backoff_no_llm_answer"),    # no metadata text, no request
        ("S5", "aquatic", 1.0, "backoff_no_llm_answer"),    # request failed
        ("S6", "marine", 0.6, "backoff"),                   # at the gate, not below it
    ]
    df = read_tsv(os.path.join(w["work"], "atlas_final.tsv.gz"))
    assert list(df["biome_final_label"]) == [LABEL[t] for t in df["biome_final"]]
    # feature does not use the LLM: its back-off is copied, "" (abstain) included, with p 0
    assert list(zip(df["feature_final"], df["feature_final_p"], df["feature_final_source"])) == [
        ("sea", "0.9", "backoff"), ("pond", "0.9", "backoff"), ("pond", "0.9", "backoff"),
        ("pond", "0.9", "backoff"), ("", "0.0", "backoff"), ("sea", "0.9", "backoff")]

    # ---- the failed request is retried by the next submit (and only it)
    run("submit", "--work_dir", w["work"])
    assert [line["custom_id"] for line in api.sent[1]] == [s5["request_id"]]


# ============================================================================= 2. from the pilot to the final answers
# The pilot: 10 Metalog samples per slot, 2 per study (5 studies, for fit's study-grouped 5-fold CV).
#   biome:   the base model always says marine 0.8 / freshwater 0.2, but half the samples are freshwater;
#            the LLM reads the salinity and gives the right term 0.9  ->  the LLM should be switched ON.
#   feature: the base model is right (sea 0.7 / pond 0.3); the LLM calls every harbour a pond (wrong, 0.9)
#            ->  the LLM should stay OFF.
PILOT_LLM = keyword_llm([("35 PSU", "marine biome"), ("0.1 PSU", "freshwater biome"), ("harbour", "pond")], p=0.9)


def pilot_sample(slot, sample_id, study, gold, cands, base, text, order):
    return {"slot": slot, "sample_id": sample_id, "study": study, "gold": gold, "base_probs": base,
            "topk_mass": sum(base), "text": text, "option_order": order,
            "candidates": [{"term_id": t, "label": LABEL[t], "synonyms": [], "definition": "", "example": ""} for t in cands]}


def make_pilot(d):
    pilot = []
    for i in range(5):
        study = f"study{i}"
        pilot += [pilot_sample("biome", f"PM{i}", study, "marine", ["marine", "freshwater"], [0.8, 0.2], "salinity: 35 PSU", [0, 1]),
                  pilot_sample("biome", f"PF{i}", study, "freshwater", ["marine", "freshwater"], [0.8, 0.2], "salinity: 0.1 PSU", [1, 0]),
                  pilot_sample("feature", f"FA{i}", study, "sea", ["sea", "pond"], [0.7, 0.3], "site: harbour", [0, 1]),
                  pilot_sample("feature", f"FB{i}", study, "sea", ["sea", "pond"], [0.7, 0.3], "site: harbour", [1, 0])]
    return (write_jsonl(d / "pilot.jsonl", pilot),
            write_jsonl(d / "responses_pilot.jsonl", [pilot_response(o, PILOT_LLM) for o in pilot]),
            write_calibration(d))


def write_calibration(d):
    """calibration.json of ONE 5_evaluate.py run (slot -> method -> entry). The gate reads p_top1_quantiles, the
    top-1 probability below which that share of out-of-fold predictions falls; with --gate 0.3: biome 0.6,
    feature 0.55 (prototype). hierarchy.merge_calibrations drops p_top1_quantiles, so fit needs an unmerged file."""
    def entry(*q):
        return {"temperature": 1.0, "curve": [], "p_top1_quantiles": dict(zip(("0.1", "0.2", "0.3", "0.4", "0.5"), q)),
                "settings": {}}
    calibration = {"biome": {"linear": entry(0.45, 0.55, 0.65, 0.75, 0.85), "prototype": entry(0.4, 0.5, 0.6, 0.7, 0.8)},
                   "feature": {"linear": entry(0.4, 0.5, 0.6, 0.7, 0.8), "prototype": entry(0.35, 0.45, 0.55, 0.65, 0.75)},
                   "material": {"linear": entry(0.5, 0.6, 0.7, 0.8, 0.9), "prototype": entry(0.5, 0.6, 0.7, 0.8, 0.9)}}
    json.dump(calibration, open(d / "calibration.json", "w"))
    return str(d / "calibration.json")


def fit_argv(w, pilot, responses, calibration, output, *extra):
    return ["fit", "--pilot", pilot, "--responses", responses, "--calibration", calibration, "--samples", w["samples"],
            "--ontology_terms", w["terms"], "--output", output, *extra]


def test_fit_then_rerank_the_atlas(tmp_path, run, openai):
    w = make_world(tmp_path / "world")
    pilot, responses, calibration = make_pilot(tmp_path)
    params = str(tmp_path / "params.json")

    # ---- fit: the weight of the LLM per slot, from the pilot
    run(*fit_argv(w, pilot, responses, calibration, params))
    p = json.load(open(params))
    biome, feature = p["slots"]["biome"], p["slots"]["feature"]
    assert sorted(p["slots"]) == ["biome", "feature"]  # material: no pilot samples, no entry
    # biome: overturning 0.8 / 0.2 with a 0.9 / 0.1 LLM needs w > log 4 / log 9 = 0.63 -> w = 1, the only grid
    # value above; then every pilot answer is right, so T is very sharp and the lowest tau of the grid is enough.
    assert (biome["use_llm"], biome["w"], biome["tau"], biome["gate_p"], biome["n_pilot"]) == (True, 1.0, 0.05, 0.6, 10)
    assert biome["T"] < 0.05
    assert biome["cv_w_per_fold"] == [1.0] * 5
    # the honest (nested CV) estimate: without the LLM the base can only reach 90 % by answering the common
    # parent 'aquatic biome' (all coarser); with it, every held-out pilot sample gets its exact term
    alone, fused = biome["cv_estimate"]["base alone (w 0)"], biome["cv_estimate"]["base + LLM (w by CV)"]
    assert (alone["exact"], alone["coarser"]) == (0.0, 1.0)
    assert (fused["exact"], fused["accuracy_strict"]) == (1.0, 1.0)
    # feature: the base is right, the LLM wrong -> w = 0 and the LLM stays off
    assert (feature["use_llm"], feature["w"]) == (False, 0)
    assert [r["w"] for r in biome["per_w"]] == list(rr.W_GRID)  # one row per weight of the grid

    # ---- build / submit / collect / apply with the fitted parameters
    run(*build_argv(w, params))
    assert members_by_text(w) == {S2_TEXT: ["S2", "S3"], S5_TEXT: ["S5"]}  # gate 0.6, as with PARAMS
    openai(keyword_llm([("0.1 PSU", "freshwater biome")], p=0.9))
    run("submit", "--work_dir", w["work"])
    run("collect", "--work_dir", w["work"])
    run(*apply_argv(w, params))
    # S2: base 0.5 x LLM 0.9 for freshwater wins, and with T ~ 0.03 the fused distribution is ~one-hot
    table = final_table(w)
    assert table[1] == ("S2", "freshwater", pytest.approx(1.0, abs=1e-3), "rerank")
    assert [(s, f, src) for s, f, _, src in table] == [
        ("S1", "marine", "backoff"), ("S2", "freshwater", "rerank"), ("S3", "freshwater", "rerank"),
        ("S4", "aquatic", "backoff_no_llm_answer"), ("S5", "aquatic", "backoff_no_llm_answer"), ("S6", "marine", "backoff")]


# ============================================================================= 3. edge cases: fit
def test_fit_a_slot_left_out_of_slots_stays_off_even_when_the_llm_helps(tmp_path, run):
    w = make_world(tmp_path / "world")
    pilot, responses, calibration = make_pilot(tmp_path)
    run(*fit_argv(w, pilot, responses, calibration, tmp_path / "params.json", "--slots", "feature"))
    biome = json.load(open(tmp_path / "params.json"))["slots"]["biome"]
    assert (biome["w"], biome["use_llm"]) == (1.0, False)  # fitted, but not allowed to use the LLM


def test_fit_gate_is_the_quantile_of_the_one_calibration_json_given(tmp_path, run):
    # gate_p = calibration[slot][--method]["p_top1_quantiles"][str(--gate)] (see write_calibration)
    w = make_world(tmp_path / "world")
    pilot, responses, calibration = make_pilot(tmp_path)
    gate_p = {}
    for name, extra in [("gate 0.2", ["--gate", "0.2"]), ("linear", ["--method", "linear"])]:
        run(*fit_argv(w, pilot, responses, calibration, tmp_path / "params.json", *extra))
        gate_p[name] = json.load(open(tmp_path / "params.json"))["slots"]["biome"]["gate_p"]
    assert gate_p == {"gate 0.2": 0.5, "linear": 0.65}
    # a merged calibration (hierarchy.merge_calibrations over seeds) has no p_top1_quantiles: fit refuses it
    merged = json.load(open(calibration))
    for slot in merged:
        for entry in merged[slot].values():
            del entry["p_top1_quantiles"]
    json.dump(merged, open(calibration, "w"))
    with pytest.raises(KeyError, match="p_top1_quantiles"):
        run(*fit_argv(w, pilot, responses, calibration, tmp_path / "params.json"))


def test_fit_uses_only_pilot_samples_with_a_usable_answer(tmp_path, run):
    w = make_world(tmp_path / "world")
    pilot, responses, calibration = make_pilot(tmp_path)
    kept = [r for r in read_jsonl(responses) if r["sample_id"] not in ("PM0", "PF0")]  # 2 biome answers missing
    write_jsonl(responses, kept)
    run(*fit_argv(w, pilot, responses, calibration, tmp_path / "params.json"))
    assert json.load(open(tmp_path / "params.json"))["slots"]["biome"]["n_pilot"] == 8


def write_single_study_pilot(d):
    """A biome pilot where the LLM matters in one study only. study0 has a freshwater sample that the base model
    gets wrong and the LLM gets right; in study1-4 the base is right and the LLM cannot decide (0.5 / 0.5: it
    changes nothing). -> (pilot, responses, calibration) paths."""
    sure, unsure = keyword_llm([("0.1 PSU", "freshwater biome"), ("35 PSU", "marine biome")], p=0.9), \
        keyword_llm([("PSU", "marine biome")], p=0.5)
    pilot, responses = [], []
    for i in range(5):
        for j, (gold, text) in enumerate([("freshwater" if i == 0 else "marine", "salinity: 0.1 PSU" if i == 0 else "salinity: 35 PSU"),
                                          ("marine", "salinity: 35 PSU")]):
            o = pilot_sample("biome", f"P{i}{j}", f"study{i}", gold, ["marine", "freshwater"], [0.8, 0.2], text, [0, 1])
            pilot.append(o)
            responses.append(pilot_response(o, sure if i == 0 else unsure))
    return write_jsonl(d / "pilot.jsonl", pilot), write_jsonl(d / "resp.jsonl", responses), write_calibration(d)


def test_fit_cv_estimate_never_lets_a_study_choose_its_own_weight(tmp_path, run):
    w = make_world(tmp_path / "world")
    run(*fit_argv(w, *write_single_study_pilot(tmp_path), tmp_path / "params.json"))
    biome = json.load(open(tmp_path / "params.json"))["slots"]["biome"]
    # On the whole pilot, w = 1 fixes study0's freshwater sample: 10 of 10 exact (w = 0: 9 of 10).
    assert biome["w"] == 1.0 and next(r for r in biome["per_w"] if r["w"] == 1.0)["exact"] == 1.0
    # In the CV, the fold that holds out study0 fits w on study1-4 only, where the LLM never helps: w = 0 (ties go
    # to the smaller weight), so study0's freshwater sample stays wrong. The estimate is 9 of 10, not 10 of 10.
    assert sorted(biome["cv_w_per_fold"]) == [0, 1.0, 1.0, 1.0, 1.0]
    assert biome["cv_estimate"]["base + LLM (w by CV)"]["exact"] == 0.9


# ============================================================================= 4. edge cases: build
def test_build_streaming_the_keyword_file_in_chunks_gives_the_same_requests(tmp_path, run):
    # --chunk_rows: the atlas keyword .h5 (6 GB) is read a block of rows at a time
    w = make_world(tmp_path)
    params = write_params(tmp_path)
    run(*build_argv(w, params))
    whole = requests_by_text(w)
    run(*build_argv(w, params, "--chunk_rows", "1"))  # one row per block
    assert requests_by_text(w) == whole


def test_build_same_text_but_other_candidates_is_another_request(tmp_path, run):
    # S7: S2's metadata text, but step 6 ranked freshwater first -> its own request (the key includes the candidates)
    atlas = ATLAS + [("S7", 0.5, "freshwater:0.5;marine:0.3;aquatic:0.2", "aquatic", 1.0, "pond", 1, 0)]
    w = make_world(tmp_path, atlas=atlas, sample_info={**SAMPLE_INFO, "S7": SAMPLE_INFO["S2"]})
    run(*build_argv(w, write_params(tmp_path)))
    requests = read_jsonl(os.path.join(w["work"], "requests.jsonl"))
    assert sorted(r["text"] for r in requests) == sorted([S2_TEXT, S2_TEXT, S5_TEXT])  # S2/S3 and S7 differ
    assert len({r["request_id"] for r in requests}) == 3


def test_build_refuses_when_no_slot_uses_the_llm(tmp_path, run):
    w = make_world(tmp_path)
    with pytest.raises(SystemExit, match="no slot uses the LLM"):
        run(*build_argv(w, write_params(tmp_path, use_llm=False)))


def test_build_refuses_an_atlas_run_without_candidates(tmp_path, run):
    w = make_world(tmp_path, topk=0)  # 6_predict_atlas.py run without --calibration: no <slot>_candidates
    with pytest.raises(SystemExit, match="no <slot>_candidates"):
        run(*build_argv(w, write_params(tmp_path)))


def test_build_with_no_gated_sample_writes_empty_outputs_and_the_next_steps_still_run(tmp_path, run, openai):
    # Formerly a bug: with no request at all, build crashed with an IndexError (an empty list of sub-biome rows
    # became a float array, which numpy refuses as an index). Now it writes empty outputs and stops early.
    w = make_world(tmp_path)
    params = write_params(tmp_path, gate_p=0.1)  # every biome_p >= 0.1: nothing gated
    run(*build_argv(w, params))
    assert read_jsonl(os.path.join(w["work"], "requests.jsonl")) == []
    assert len(read_tsv(os.path.join(w["work"], "members.tsv.gz"))) == 0
    api = openai(ATLAS_LLM)
    run("submit", "--work_dir", w["work"])
    assert api.sent == []  # nothing to send
    run(*apply_argv(w, params))
    assert {src for *_, src in final_table(w)} == {"backoff"}  # every sample keeps step 6's back-off answer


# ============================================================================= 5. edge cases: submit / collect
@pytest.fixture
def built(tmp_path, run):
    """The toy atlas after build (2 requests: S2's and S5's)."""
    w = make_world(tmp_path)
    run(*build_argv(w, write_params(tmp_path)))
    return w


def test_submit_dry_run_sends_nothing_and_limit_sends_only_the_first_requests(built, run, openai):
    api = openai(ATLAS_LLM)
    run("submit", "--work_dir", built["work"], "--dry_run")
    assert api.sent == [] and not os.path.exists(os.path.join(built["work"], "batches.json"))
    run("submit", "--work_dir", built["work"], "--limit", "1")  # a trial: the first line of requests.jsonl
    first = read_jsonl(os.path.join(built["work"], "requests.jsonl"))[0]["request_id"]
    assert [[line["custom_id"] for line in batch] for batch in api.sent] == [[first]]


def test_submit_never_resends_a_request_that_is_still_in_flight(built, run, openai):
    api = openai(ATLAS_LLM)
    first, second = [r["request_id"] for r in read_jsonl(os.path.join(built["work"], "requests.jsonl"))]
    run("submit", "--work_dir", built["work"], "--limit", "1")  # batch-0 = [first], not collected yet
    run("submit", "--work_dir", built["work"])                  # room for more batches (--max_pending 3)
    assert [[line["custom_id"] for line in b] for b in api.sent] == [[first], [second]]  # `first` not sent twice


def test_submit_splits_into_batches_and_waits_when_max_pending_are_in_flight(built, run, openai):
    api = openai(ATLAS_LLM)
    first, second = [r["request_id"] for r in read_jsonl(os.path.join(built["work"], "requests.jsonl"))]
    run("submit", "--work_dir", built["work"], "--batch_requests", "1", "--max_pending", "1")
    assert [[line["custom_id"] for line in b] for b in api.sent] == [[first]]  # 2 batches of 1, only 1 allowed
    run("submit", "--work_dir", built["work"], "--batch_requests", "1", "--max_pending", "1")
    assert len(api.sent) == 1  # the first batch is still pending: nothing new, and `first` is not sent twice
    run("collect", "--work_dir", built["work"])
    run("submit", "--work_dir", built["work"], "--batch_requests", "1", "--max_pending", "1")
    assert [[line["custom_id"] for line in b] for b in api.sent] == [[first], [second]]


def test_collect_leaves_a_running_batch_for_the_next_collect(built, run, openai):
    api = openai(ATLAS_LLM)
    run("submit", "--work_dir", built["work"])
    api.still_running.add("batch-0")
    run("collect", "--work_dir", built["work"])
    assert not os.path.exists(os.path.join(built["work"], "responses.jsonl"))
    assert json.load(open(os.path.join(built["work"], "batches.json")))["batches"][0]["collected"] is False
    api.still_running.clear()
    run("collect", "--work_dir", built["work"])  # rerun: now it is finished
    assert len(read_jsonl(os.path.join(built["work"], "responses.jsonl"))) == 1  # S2's (S5's fails)


def test_collect_before_any_submit_stops(built, run, openai):
    openai(ATLAS_LLM)
    with pytest.raises(SystemExit, match="no batch submitted yet"):
        run("collect", "--work_dir", built["work"])


def test_collect_skips_an_expired_request_and_the_next_submit_resends_it(built, run, openai):
    # a request the Batch API never ran (24 h window over) is in the error file with "response": null
    api = openai(ATLAS_LLM)
    s2 = requests_by_text(built)[S2_TEXT]["request_id"]
    api.expired.add(s2)
    run("submit", "--work_dir", built["work"])
    run("collect", "--work_dir", built["work"])
    assert read_jsonl(os.path.join(built["work"], "responses.jsonl")) == []  # S2 expired, S5 failed
    api.expired.clear()
    run("submit", "--work_dir", built["work"])
    assert sorted(line["custom_id"] for line in api.sent[1]) == sorted(line["custom_id"] for line in api.sent[0])
    run("collect", "--work_dir", built["work"])
    assert [r["request_id"] for r in read_jsonl(os.path.join(built["work"], "responses.jsonl"))] == [s2]


# One line of a real Batch API output file (gpt-4.1-mini, rerank_atlas run; ids shortened, top_logprobs cut to 3),
# and what collect stores for it (the first line of the real responses.jsonl).
REAL_OUTPUT_LINE = {
    "id": "batch_req_68dd", "custom_id": "biome-b709b812855995e5",
    "response": {"status_code": 200, "request_id": "5b1f", "body": {
        "id": "chatcmpl-CLx", "object": "chat.completion", "created": 1759315200, "model": "gpt-4.1-mini-2025-04-14",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "A", "refusal": None, "annotations": []},
                     "logprobs": {"content": [{"token": "A", "logprob": -0.0486, "bytes": [65], "top_logprobs": [
                         {"token": "A", "logprob": -0.0486, "bytes": [65]}, {"token": "E", "logprob": -3.0486, "bytes": [69]},
                         {"token": " A", "logprob": -17.6736, "bytes": [32, 65]}]}], "refusal": None},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 979, "completion_tokens": 1, "total_tokens": 980,
                  "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                  "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0}},
        "service_tier": "default", "system_fingerprint": "fp_6f2e"}},
    "error": None}
REAL_STORED = {"request_id": "biome-b709b812855995e5", "model": "gpt-4.1-mini-2025-04-14", "content": "A",
               "top_logprobs_raw": [["A", -0.0486], ["E", -3.0486], [" A", -17.6736]], "usage": {"in": 979, "out": 1}}


def leading_newline_line():
    """REAL_OUTPUT_LINE answered "\nE": a blank first token, then the letter (its top_logprobs are the ones to keep)."""
    line = json.loads(json.dumps(REAL_OUTPUT_LINE))
    choice = line["response"]["body"]["choices"][0]
    choice["message"]["content"] = "\nE"
    choice["logprobs"]["content"] = [
        {"token": "\n", "logprob": -0.01, "bytes": [10], "top_logprobs": [{"token": "\n", "logprob": -0.01, "bytes": [10]}]},
        {"token": "E", "logprob": -0.2, "bytes": [69], "top_logprobs": [{"token": "E", "logprob": -0.2, "bytes": [69]},
                                                                         {"token": "A", "logprob": -1.8, "bytes": [65]}]}]
    return line


def collect_lines(module, run_module, work, lines, monkeypatch):
    """collect() of `module` on one finished batch with these output lines; -> responses.jsonl records."""
    client, state = output_file_client(lines)
    os.makedirs(work, exist_ok=True)
    json.dump(state, open(os.path.join(work, "batches.json"), "w"))
    monkeypatch.setattr(module, "client_for", lambda args: client)
    run_module(module, "collect", "--work_dir", work)
    return read_jsonl(os.path.join(work, "responses.jsonl"))


def test_collect_stores_a_real_batch_output_line(tmp_path, monkeypatch):
    def run_module(module, *argv):
        monkeypatch.setattr(sys, "argv", ["7_rerank_atlas.py", *argv])
        module.main()
    stored = collect_lines(rr, run_module, str(tmp_path / "work"), [REAL_OUTPUT_LINE, leading_newline_line()], monkeypatch)
    assert stored[0] == REAL_STORED
    # a blank first token is skipped: the log-probabilities are the letter's
    assert (stored[1]["content"], stored[1]["top_logprobs_raw"]) == ("\nE", [["E", -0.2], ["A", -1.8]])


# ============================================================================= 6. edge cases: apply
# apply on its own: requests.jsonl / responses.jsonl / members.tsv.gz written by hand for one gated sample, S2,
# so each test shows exactly what apply reads. S2's request: candidates marine, freshwater, aquatic in letter
# order A, B, C (option_order [0, 1, 2]).
def write_apply_inputs(w, base_probs, responses):
    os.makedirs(w["work"], exist_ok=True)
    cands = ["marine", "freshwater", "aquatic"]
    write_jsonl(os.path.join(w["work"], "requests.jsonl"), [{
        "slot": "biome", "request_id": "biome-s2", "sample_id": "biome-s2", "text": S2_TEXT, "option_order": [0, 1, 2],
        "candidates": [{"term_id": t, "label": LABEL[t], "synonyms": [], "definition": "", "example": ""} for t in cands],
        "base_probs": base_probs, "topk_mass": sum(base_probs)}])
    write_jsonl(os.path.join(w["work"], "responses.jsonl"), [{"request_id": "biome-s2", **r} for r in responses])
    write_tsv(os.path.join(w["work"], "members.tsv.gz"), [{"request_id": "biome-s2", "sample_id": "S2"}])


def says(**letter_probs):
    """A collected answer: says(B=0.6, A=0.2, C=0.2) = letter B, with these probabilities."""
    return {"content": max(letter_probs, key=letter_probs.get),
            "top_logprobs_raw": [[letter, float(np.log(p))] for letter, p in letter_probs.items()]}


def s2_final(w):
    return next(row for row in final_table(w) if row[0] == "S2")


def test_apply_backs_off_the_reranked_answer_when_tau_is_higher(tmp_path, run):
    # the fusion of the basic workflow (fused 1/3, 1/2, 1/6), but tau 0.6: freshwater (0.5) is not enough;
    # its parent aquatic sums every candidate below it: 1/3 + 1/2 + 1/6 = 1.0 -> 'aquatic biome'
    w = make_world(tmp_path)
    write_apply_inputs(w, [0.5, 0.25, 0.25], [says(A=0.2, B=0.6, C=0.2)])
    run(*apply_argv(w, write_params(tmp_path, tau=0.6)))
    assert s2_final(w) == ("S2", "aquatic", 1.0, "rerank")


def test_apply_abstains_when_tau_exceeds_the_candidates_total_mass(tmp_path, run):
    # 20 % of the base probability is outside the candidates (mass 0.8) and is never counted: even 'aquatic'
    # (all candidates, 0.8) stays below tau 0.9 -> "" (abstain), although the LLM was asked
    w = make_world(tmp_path)
    write_apply_inputs(w, [0.4, 0.2, 0.2], [says(A=0.2, B=0.6, C=0.2)])
    run(*apply_argv(w, write_params(tmp_path, tau=0.9)))
    assert s2_final(w) == ("S2", "", 0.0, "rerank")


def test_apply_the_latest_response_of_a_request_wins(tmp_path, run):
    # e.g. a request answered twice (re-collected): the last line of responses.jsonl is used
    w = make_world(tmp_path)
    write_apply_inputs(w, [0.5, 0.25, 0.25], [says(A=0.2, B=0.6, C=0.2), says(A=0.9, B=0.05, C=0.05)])
    run(*apply_argv(w, write_params(tmp_path)))
    assert s2_final(w)[1] == "marine"  # the second answer (A = marine); the first would give freshwater


def test_apply_an_unusable_response_counts_as_no_answer(tmp_path, run):
    w = make_world(tmp_path)
    write_apply_inputs(w, [0.5, 0.25, 0.25], [{"content": "Sorry, I cannot tell.", "top_logprobs_raw": None}])
    run(*apply_argv(w, write_params(tmp_path)))
    assert s2_final(w) == ("S2", "aquatic", 1.0, "backoff_no_llm_answer")


def test_apply_right_after_build_marks_every_gated_sample_as_waiting(built, tmp_path, run):
    # no responses.jsonl yet: apply still writes a complete table, with the back-off for everyone
    run(*apply_argv(built, write_params(tmp_path)))
    assert [(s, src) for s, _, _, src in final_table(built)] == [
        ("S1", "backoff"), ("S2", "backoff_no_llm_answer"), ("S3", "backoff_no_llm_answer"),
        ("S4", "backoff_no_llm_answer"), ("S5", "backoff_no_llm_answer"), ("S6", "backoff")]
