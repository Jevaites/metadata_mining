"""
Tests for the follow-up pieces of the Jev pilot: `jev_pilot.py run --questions` (a flat request without the
whole-vocabulary Choice or without the Nouls) and experiments/jev_followups.py (candidate insertion, compare,
score_open). The builders that need the training set and embeddings (`candidates`, `open`) are exercised on the real
data by the commands in README 7d, not here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import jev_followups as jf  # noqa: E402
import jev_pilot as jp  # noqa: E402
from test_jev_pilot import AQ, MA, OC, TE, Scripted, args, pilot_record, run, toy_inputs, write_jsonl  # noqa: E402


def test_flat_request_can_skip_questions():
    ctx = jp.Ctx(toy_inputs(), args(questions=["rerank"]))
    q, _ = jp.flat_questions(pilot_record(), ctx)
    assert set(q) == {"rerank_0"}
    rec = run(jp.ask_flat(Scripted(weights={"ocean biome": 5}), pilot_record(), ctx))
    assert "vocab" not in rec and rec["fits"] == [] and rec["content"] == "C"  # no Nouls: never "none"
    view = jp.arm_views(rec, pilot_record(), ctx)
    assert set(view) == {"jev_rerank"} and view["jev_rerank"][1] == OC


def test_decide_rerank_without_nouls_always_picks():
    pick, dist = jp.decide_rerank([0.2, 0.7, 0.1], [])
    assert pick == 1 and dist[-1] == 0


def test_many_candidates_skip_the_letter_format():
    o = pilot_record()
    o["candidates"] = o["candidates"] * 10  # 30 candidates: more than rerank_prompt's 22 letters
    o["option_order"] = list(range(30))
    rec = run(jp.ask_flat(Scripted(), o, jp.Ctx(toy_inputs(), args(questions=["rerank"]))))
    assert "content" not in rec and len(rec["rerank"]) == 30


def test_insert_new_keeps_the_old_order():
    rng = np.random.default_rng(0)
    order = jf.insert_new([2, 0, 1], 3, 3, rng)
    assert sorted(order) == [0, 1, 2, 3, 4, 5] and [i for i in order if i < 3] == [2, 0, 1]


def test_compare_and_gate(tmp_path):
    rows = [pilot_record("S1", "own", OC, (MA, OC, AQ)), pilot_record("S2", "s1", MA, (MA, OC, AQ)),
            pilot_record("S3", "s2", TE, (AQ, MA, TE)), pilot_record("S4", "s3", AQ, (MA, AQ, OC))]
    for r, conf in zip(rows, (0.2, 0.9, 0.3, 0.8)):
        r["base_maxp"] = conf
    pilot = tmp_path / "pilot.jsonl"
    write_jsonl(pilot, rows)
    # the Jev answers pick candidate 1 everywhere: right on S1 and S4, wrong on S2 and S3
    resp = tmp_path / "jev.jsonl"
    write_jsonl(resp, [{"slot": "biome", "sample_id": r["sample_id"], "arm": "flat", "rerank": [0.1, 0.8, 0.1],
                        "fits": [0.5, 0.9, 0.5]} for r in rows])
    out = tmp_path / "cmp.json"
    jf.main(["compare", "--gate", "0.5", "--output", str(out), "--runs", f"proto={pilot},-,prototype", f"jev={pilot},{resp},rerank"])
    res = json.loads(out.read_text())["biome"]
    assert res["proto"]["exact"] == 0.25 and res["jev"]["exact"] == 0.5  # prototype right on S2 only
    # gate 50 %: Jev answers S1 (0.2) and S3 (0.3), the prototype the rest: S1 right (Jev) + S2 right (prototype)
    assert res["jev"]["gated 50 %"] == 0.5 and res["proto"]["gated 50 %"] == 0.25


def test_score_open_splits_unseen_labels(tmp_path):
    inp = toy_inputs()
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(inp))
    orig = [pilot_record("S1", "own", "ENVO_00000010", (MA, OC, AQ)), pilot_record("S2", "s1", OC, (MA, OC, AQ))]
    pilot = tmp_path / "pilot.jsonl"
    write_jsonl(pilot, orig)
    opened = []
    for o in orig:
        ids = [OC, "ENVO_00000010", AQ]
        opened.append(dict(o, candidates=[{"term_id": t} for t in ids], gold_in_topk=o["gold"] in ids))
    open_pilot = tmp_path / "open.jsonl"
    write_jsonl(open_pilot, opened)
    resp = tmp_path / "resp.jsonl"  # the open Choice picks the hot spring term (an unseen label) for S1, ocean for S2
    write_jsonl(resp, [{"slot": "biome", "sample_id": "S1", "rerank": [0.1, 0.8, 0.1]},
                       {"slot": "biome", "sample_id": "S2", "rerank": [0.7, 0.2, 0.1]}])
    out = tmp_path / "open_score.json"
    jf.main(["score_open", "--pilot", str(pilot), "--open_pilot", str(open_pilot), "--inputs_open", str(inputs),
             "--responses", str(resp), "--output", str(out)])
    res = json.loads(out.read_text())["biome"]
    assert res["all"]["open Choice"]["exact"] == 1.0 and res["all"]["prototype"]["exact"] == 0.0
    assert res["unseen labels"]["n"] == 1 and res["unseen labels"]["open Choice"]["exact"] == 1.0
    assert res["unseen labels"]["prototype"]["coarser"] == 0.0  # marine biome is not an ancestor of the hot spring term


if __name__ == "__main__":
    pytest.main([__file__])


def test_similar_samples_go_into_the_state():
    import jev_questions as W
    o = pilot_record()
    o["similar"] = [{"text": "title: gut; host: human", "term": "intestine environment"}]
    client = Scripted()
    run(jp.ask_flat(client, o, jp.Ctx(toy_inputs(), args(questions=["rerank"]))))
    state = client.requests[0][0]
    ex = state["similar_samples_labelled_by_curators"]["sample_1"]
    assert ex["sample_metadata"] == {"title": "gut", "host": "human"}
    assert ex[f"{W.MIXS[o['slot']]}_assigned"] == "intestine environment"
    assert W.state("title: x") == {"sample_metadata": {"title": "x"}}  # no examples: the state is unchanged


def test_similar_samples_leave_the_study_out_and_cap_per_study():
    import pandas as pd
    from types import SimpleNamespace
    X = np.array([[1, 0], [0.99, 0.1], [0.98, 0.2], [0.97, 0.3], [0, 1]], np.float32)
    S = pd.DataFrame({"text": list("abcde"), "feature": ["T1", "T1", "T2", "T2", "T3"]})
    T = SimpleNamespace(S=S, X=X, studies=np.array(["own", "s1", "s1", "s1", "s2"]),
                        vec=lambda ids: np.array([[1, 0]], np.float32))
    info = {t: {"label": t.lower()} for t in ("T1", "T2", "T3")}
    a = SimpleNamespace(similar=3, similar_per_study=2, example_chars=300)
    out = jf.similar_samples(T, X / np.linalg.norm(X, axis=1, keepdims=True), "feature", {"sample_id": "q", "study": "own"}, info, a)
    assert [x["text"] for x in out] == ["b", "c", "e"]  # "a" is the sample's own study; s1 capped at 2
    assert [x["term"] for x in out] == ["t1", "t2", "t3"]
