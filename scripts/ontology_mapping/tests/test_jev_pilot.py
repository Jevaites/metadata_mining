"""
Tests for experiments/jev_pilot.py (the Jev pilot: one flat request per sample with a whole-vocabulary Choice, a
rerank Choice and a Noul per candidate; top-down beam search; the request-keyed cache; decisions in code).

No network: the arms run against a scripted client (probabilities chosen per option name) or the script's own
keyword-overlap FakeJev; one test drives the real typesafe-sdk client through an httpx2 MockTransport to check
that the requests we build are valid for the SDK (skipped when typesafe-sdk is not installed).

Toy biome ontology (ids shortened to ENVO_0000000x):
  biome (ENVO_00000428, a slot root: never an answer)
    aquatic biome (1) -> marine biome (2) -> ocean biome (3)
                      -> freshwater biome (4) -> lake biome (5)
    terrestrial biome (6)
  environmental system determined by an organism (9) -> animal-associated environment (7)
"""
import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import jev_pilot as jp  # noqa: E402
import jev_questions as W  # noqa: E402
from rerank_prompt import parse  # noqa: E402

B0, AQ, MA, OC, FW, LK, TE, AN, ORG, OWN = ("ENVO_00000428", "ENVO_00000001", "ENVO_00000002", "ENVO_00000003",
                                           "ENVO_00000004", "ENVO_00000005", "ENVO_00000006", "ENVO_00000007",
                                           "ENVO_00000009", "ENVO_00000010")
LABELS = {B0: "biome", AQ: "aquatic biome", MA: "marine biome", OC: "ocean biome", FW: "freshwater biome",
          LK: "lake biome", TE: "terrestrial biome", AN: "animal-associated environment",
          ORG: "environmental system determined by an organism", OWN: "hot spring biome"}
PARENTS = {B0: [], AQ: [B0], MA: [AQ], OC: [MA], FW: [AQ], LK: [FW], TE: [B0], AN: [ORG], ORG: [], OWN: [AQ]}


def toy_inputs(slot_use=None, parents=None, labels=None):
    parents = parents or PARENTS
    labels = labels or LABELS
    terms = {t: {"label": labels[t], "synonyms": [], "definition": f"definition of {labels[t]}", "parents": p}
             for t, p in parents.items()}
    use = slot_use or {AQ: {"s1": 3, "s2": 1}, MA: {"s1": 2, "s3": 2}, OC: {"s2": 5}, FW: {"s3": 1},
                       LK: {"s1": 1, "s4": 1}, TE: {"s2": 2}, AN: {"s4": 3}, OWN: {"own": 2}}
    return {"terms": terms, "slots": {"biome": use, "feature": {}, "material": {}}}


def args(**kw):
    d = dict(min_studies=1, orders=1, seed=0, beam=3, start="lca", model="jev-test")
    d.update(kw)
    return SimpleNamespace(**d)


def pilot_record(sample_id="S1", study="own", gold=OC, cands=(MA, OC, AQ), order=(2, 0, 1)):
    return {"slot": "biome", "sample_id": sample_id, "study": study, "fold": 0, "gold": gold,
            "gold_in_topk": gold in cands, "base_top1": cands[0],
            "base_probs": [0.5, 0.3, 0.1][:len(cands)], "topk_mass": 0.9, "base_maxp": 0.5,
            "text": "seawater from the open ocean, pelagic zone",
            "candidates": [{"term_id": c, "label": LABELS.get(c, c), "synonyms": [], "definition": "", "example": f"ex {c}"}
                           for c in cands],
            "option_order": list(order)}


class Scripted:
    """A Jev stand-in whose Choice probabilities come from {option name: weight} (missing names: 0.01, then
    normalised) and whose Nouls come from {question key: value}. Keeps every request."""
    model = "scripted"

    def __init__(self, weights=None, nouls=None):
        self.weights, self.nouls, self.requests = weights or {}, nouls or {}, []

    async def system_one(self, state, questions, **kw):
        self.requests.append((state, questions))
        answers = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                names = list(q["criteria"])
                w = np.array([self.weights.get(n, 0.01) for n in names], float)
                p = w / w.sum()
                answers[k] = SimpleNamespace(probabilities=dict(zip(names, p)), choice=names[int(p.argmax())],
                                             confidence=float(p.max()))
            else:
                answers[k] = SimpleNamespace(noul=self.nouls.get(k, 0.5))
        return SimpleNamespace(model=self.model, answers=answers, usage=SimpleNamespace(input_tokens=100))


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- helpers
def test_option_names_are_unique():
    inp = toy_inputs(labels={**LABELS, LK: "marine biome"})
    ctx = jp.Ctx(inp, args())
    assert jp.option_names([MA, LK, OC], ctx) == [f"marine biome ({MA})", f"marine biome ({LK})", "ocean biome"]


def test_vocab_excludes_the_samples_own_study_and_sorts_by_use():
    ctx = jp.Ctx(toy_inputs(), args())
    labels, freq = ctx.vocab("biome", "own")
    assert OWN not in labels  # only its own study uses it: an unseen label
    assert labels[0] == OC and freq[OC] == 5
    labels_s2, freq_s2 = ctx.vocab("biome", "s2")
    assert OC not in labels_s2 and TE not in labels_s2 and freq_s2[AQ] == 3
    ctx2 = jp.Ctx(toy_inputs(), args(min_studies=2))
    assert ctx2.vocab("biome", "own")[0] == [AQ, MA, LK]  # used by >= 2 other studies (ties by id)


def test_tree_structure_and_options():
    ctx = jp.Ctx(toy_inputs(), args())
    tree = ctx.tree("biome", "own")
    assert sorted(tree["children"][jp.ROOT]) == [ORG, B0]
    assert sorted(tree["children"][AQ]) == [MA, FW]
    assert B0 not in tree["answers"] and OC in tree["answers"]
    assert set(tree["below"][AQ]) == {AQ, MA, OC, FW, LK}
    opts = jp.node_options(AQ, tree, ctx)
    assert [o[1] for o in opts] == [MA, FW, None]  # None = 'aquatic biome itself'; marine (4 + 5 samples below) before freshwater (2)
    assert "ocean biome" in opts[0][2]["narrower terms include"]
    assert [o[1] for o in jp.node_options(B0, tree, ctx)] == [AQ, TE]  # a slot root: no "itself" option
    assert jp.node_options(OC, tree, ctx) == []  # a leaf



# ----------------------------------------------------------------------------- flat request
def test_flat_request_holds_vocab_rerank_and_fits_questions():
    ctx = jp.Ctx(toy_inputs(), args())
    o = pilot_record()
    q, (vids, vnames, cids, cnames) = jp.flat_questions(o, ctx)
    assert set(q) == {"vocab_0", "rerank_0", "fits_0", "fits_1", "fits_2"}
    assert set(q["vocab_0"]["criteria"]) == {LABELS[t] for t in ctx.vocab("biome", "own")[0]}
    assert list(q["rerank_0"]["criteria"]) == ["aquatic biome", "marine biome", "ocean biome"]  # = option_order (GPT's)
    assert q["rerank_0"]["instructions"] == W.which_term("biome")
    assert q["rerank_0"]["criteria"]["ocean biome"]["a similar sample the curators labelled with this term"] == {"text": f"ex {OC}"}
    assert q["fits_1"]["type"] == "noul" and q["fits_1"]["instructions"]["term"]["name"] == "ocean biome"
    q2, _ = jp.flat_questions(o, jp.Ctx(toy_inputs(), args(orders=2)))
    assert {"vocab_1", "rerank_1"} <= set(q2) and list(q2["rerank_0"]["criteria"]) == list(q["rerank_0"]["criteria"])


def test_flat_record_and_rerank_pilot_compatibility():
    ctx = jp.Ctx(toy_inputs(), args())
    o = pilot_record()
    client = Scripted(weights={"ocean biome": 6, "marine biome": 3, "aquatic biome": 1},
                      nouls={"fits_0": 0.6, "fits_1": 0.9, "fits_2": 0.8})
    rec = run(jp.ask_flat(client, o, ctx))
    assert len(client.requests) == 1 and client.requests[0][0] == {"sample_metadata": {"text": o["text"]}}
    assert max(rec["vocab"], key=rec["vocab"].get) == OC
    assert rec["rerank"] == pytest.approx([0.3, 0.6, 0.1], abs=1e-6) and rec["fits"] == [0.6, 0.9, 0.8]
    dist, pick = parse(o, rec)  # rerank_pilot.py reads the jev_rerank decision
    assert pick == 1 and rec["content"] == "C"  # ocean biome is shown third
    assert dist == pytest.approx([0.9 * 0.3, 0.9 * 0.6, 0.9 * 0.1, 0.1], abs=1e-5)


def test_decide_rerank_uses_the_choice_for_which_and_the_nouls_for_whether():
    pick, dist = jp.decide_rerank([0.2, 0.7, 0.1], [0.5, 0.25, 0.1])
    assert pick == 1 and dist == pytest.approx([0.1, 0.35, 0.05, 0.5])  # the Choice's top, although its Noul is low
    pick, _ = jp.decide_rerank([0.2, 0.7, 0.1], [0.2, 0.25, 0.1])
    assert pick == 3  # no candidate reaches 0.3: none


def test_decide_fits_table():
    anc = jp.Ctx(toy_inputs(), args()).anc
    ids = [MA, OC, AQ]
    pick, decision, dist = jp.decide_fits([0.6, 0.9, 0.3], ids, anc)
    raw = np.array([0.6, 0.9, 0.3, 0.4 * 0.1 * 0.7])
    assert (pick, decision) == (1, "yes") and dist == pytest.approx(raw / raw.sum())
    # all three true: the most specific (ocean, below marine and aquatic), not the highest Noul (aquatic)
    assert jp.decide_fits([0.8, 0.75, 0.95], ids, anc)[:2] == (1, "yes")
    assert jp.decide_fits([0.5, 0.4, 0.2], ids, anc)[:2] == (0, "unsettled")
    assert jp.decide_fits([0.2, 0.25, 0.1], ids, anc)[:2] == (3, "none")


def test_metadata_becomes_named_fields():
    text = "title: Rhizosphere; geo_loc_name: Brazil: Minas Gerais; note: a; b; host: x; host: y"
    assert W.metadata_fields(text) == {"title": "Rhizosphere", "geo_loc_name": "Brazil: Minas Gerais",
                                       "note": "a; b", "host": "x | y"}
    assert W.state("plain words") == {"sample_metadata": {"text": "plain words"}}


# ----------------------------------------------------------------------------- cache
def test_cache_replays_identical_requests_and_keeps_the_payload(tmp_path):
    inner = Scripted(weights={"a": 3})
    cache = jp.CachedClient(inner, str(tmp_path / "cache.jsonl"), "jev-test")
    q = {"x": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None}}, "y": {"type": "noul", "instructions": "?"}}
    r1 = run(cache.system_one({"s": 1}, q))
    r2 = run(cache.system_one({"s": 1}, q))
    assert len(inner.requests) == 1 and (cache.hits, cache.misses) == (1, 1)
    assert r2.answers["x"].probabilities == r1.answers["x"].probabilities and r2.answers["y"].noul == 0.5
    reordered = {"x": {"type": "choice", "instructions": "?", "criteria": {"b": None, "a": None}}, "y": q["y"]}
    run(cache.system_one({"s": 1}, reordered))  # option order is part of the question: a new request
    assert len(inner.requests) == 2
    again = jp.CachedClient(Scripted(), str(tmp_path / "cache.jsonl"), "jev-test")  # reloaded from the file
    assert run(again.system_one({"s": 1}, q)).answers["x"].probabilities == r1.answers["x"].probabilities
    logged = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert logged[0]["request"] == {"state": {"s": 1}, "questions": q} and len(logged) == 2
    other = jp.CachedClient(Scripted(), str(tmp_path / "cache.jsonl"), "fake-jev")  # another model: never replayed
    run(other.system_one({"s": 1}, q))
    assert other.misses == 1


# ----------------------------------------------------------------------------- top-down
TD_WEIGHTS = {"biome": 0.8, "environmental system determined by an organism": 0.2,
              "aquatic biome": 0.75, "terrestrial biome": 0.25,
              "marine biome": 0.6, "freshwater biome": 0.2, "aquatic biome itself, nothing narrower": 0.2,
              "ocean biome": 0.9, "marine biome itself, nothing narrower": 0.1,
              "lake biome": 0.6, "freshwater biome itself, nothing narrower": 0.4}


def test_topdown_beam_from_the_root():
    ctx = jp.Ctx(toy_inputs(), args(start="root"))
    client = Scripted(weights=TD_WEIGHTS)
    rec = run(jp.ask_topdown(client, pilot_record(), ctx))
    # ROOT -> biome .8 / organism .2 (single child: animal-associated, a leaf: finished at .2);
    # biome -> aquatic .6 / terrestrial .2 (leaf); aquatic -> marine .36 / freshwater .12 / itself .12
    # (beam 3 keeps marine .36, terrestrial .2, animal .2); marine -> ocean .324 / itself .036
    assert rec["terminals"] == pytest.approx({OC: 0.324, TE: 0.2, AN: 0.2}, abs=1e-6)
    assert rec["arm"] == "topdown_root" and rec["start"] == jp.ROOT and rec["gold_reachable"]
    assert rec["questions"] == 4 and rec["requests"] == 4  # ROOT, biome, aquatic, marine
    assert rec["best_path"] == [jp.ROOT, B0, AQ, MA, OC]
    assert client.requests[2][1]["n0_0"]["instructions"] == W.which_branch("biome", "aquatic biome")
    assert client.requests[0][1]["n0_0"]["instructions"] == W.which_branch("biome")


def test_topdown_lca_starts_below_the_root():
    ctx = jp.Ctx(toy_inputs(), args(start="lca"))
    client = Scripted(weights=TD_WEIGHTS)
    rec = run(jp.ask_topdown(client, pilot_record(cands=(OC, MA, AQ)), ctx))
    # aquatic -> marine .6 / freshwater .2 / itself .2; then marine and freshwater in one request:
    # ocean .54, marine itself .06, lake .12, freshwater itself .08 -> beam 3: ocean, aquatic itself, lake
    assert rec["start"] == AQ and rec["questions"] == 3 and rec["requests"] == 2
    assert rec["terminals"] == pytest.approx({OC: 0.54, AQ: 0.2, LK: 0.12}, abs=1e-6)


def test_topdown_sums_routes_through_several_parents():
    CO = "ENVO_00000008"
    parents = {B0: [], MA: [B0], TE: [B0], CO: [MA, TE]}
    labels = {B0: "biome", MA: "marine biome", TE: "terrestrial biome", CO: "coastal biome"}
    inp = toy_inputs(slot_use={MA: {"s1": 1}, TE: {"s1": 1}, CO: {"s2": 1}}, parents=parents, labels=labels)
    ctx = jp.Ctx(inp, args(start="root"))
    client = Scripted(weights={"marine biome": .5, "terrestrial biome": .5, "coastal biome": 1,
                               "marine biome itself, nothing narrower": .25,
                               "terrestrial biome itself, nothing narrower": 2 / 3})
    rec = run(jp.ask_topdown(client, pilot_record(gold=CO, cands=(CO, MA, TE)), ctx))
    # ROOT has one child (biome): passed without asking. Coastal via marine .4 and via terrestrial .3 are two beam
    # entries, summed at the end; beam 3 keeps them and terrestrial itself (.2), drops marine itself (.1)
    assert rec["terminals"] == pytest.approx({CO: 0.7, TE: 0.2}, abs=1e-6)
    assert rec["questions"] == 3 and rec["requests"] == 2  # biome, then marine + terrestrial in one request


def test_lca_start():
    ctx = jp.Ctx(toy_inputs(), args())
    tree = ctx.tree("biome", "own")
    assert jp.lca_start(pilot_record(cands=(OC, MA, AQ)), tree, ctx) == AQ
    assert jp.lca_start(pilot_record(cands=(OC, MA, LK)), tree, ctx) == AQ
    assert jp.lca_start(pilot_record(cands=(OC, TE, MA)), tree, ctx) == B0
    assert jp.lca_start(pilot_record(cands=(OC, AN, MA)), tree, ctx) == jp.ROOT


# ----------------------------------------------------------------------------- run / check_order / score through main()
def write_jsonl(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_run_replay_check_order_and_score(tmp_path, capsys):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(toy_inputs()))
    pilot = tmp_path / "pilot.jsonl"
    rows = [pilot_record("S1", "own", OC, (MA, OC, AQ)), pilot_record("S2", "s1", LK, (FW, LK, AQ)),
            pilot_record("S3", "s2", TE, (AQ, MA, OC)), pilot_record("S4", "s3", AN, (AN, TE, AQ))]
    rows[1]["text"] = "lake water, freshwater lake"
    write_jsonl(pilot, rows)
    common = ["--pilot", str(pilot), "--inputs", str(inputs), "--fake"]
    runs = [["--arm", "flat", "--output", str(tmp_path / "flat.jsonl")],
            ["--arm", "topdown", "--start", "lca", "--output", str(tmp_path / "td_lca.jsonl")],
            ["--arm", "topdown", "--start", "root", "--output", str(tmp_path / "td_root.jsonl")]]
    for r in runs:
        jp.main(["run", *common, *r])
    cache = tmp_path / "jev_cache.jsonl"
    n_cached = len(cache.read_text().splitlines())
    for r in runs:  # a rerun replays everything from the cache
        jp.main(["run", *common, *r])
    assert len(cache.read_text().splitlines()) == n_cached
    assert "0 new requests" in capsys.readouterr().out
    assert all(len((tmp_path / f).read_text().splitlines()) == 4 for f in ("flat.jsonl", "td_lca.jsonl", "td_root.jsonl"))
    jp.main(["run", *common, "--arm", "flat", "--output", str(tmp_path / "dry" / "flat.jsonl"), "--dry_run"])
    assert not (tmp_path / "dry").exists()
    jp.main(["check_order", *common, "--cache", str(cache)])
    order = json.loads((tmp_path / "check_order.json").read_text())["biome"]
    assert order["n"] == 4 and 0 <= order["agree"] <= 1
    ref = tmp_path / "gpt.jsonl"  # a GPT-style answer: letter B (= candidate 0 under option_order [2, 0, 1])
    write_jsonl(ref, [{"slot": "biome", "sample_id": r["sample_id"], "content": "B", "top_logprobs_raw": [["B", -0.1]]}
                      for r in rows])
    report = tmp_path / "score.json"
    jp.main(["score", "--pilot", str(pilot), "--inputs", str(inputs), "--responses", str(tmp_path / "flat.jsonl"),
             str(tmp_path / "td_lca.jsonl"), str(tmp_path / "td_root.jsonl"), "--reference", f"gpt={ref}",
             "--output", str(report)])
    res = json.loads(report.read_text())
    assert set(res["biome"]) == {"base", "gpt", "jev_vocab", "jev_rerank", "jev_fits", "topdown_lca", "topdown_root"}
    assert res["biome"]["base"]["exact"] == 0.25  # S4 (animal-associated) is the base's only hit
    assert res["biome"]["gpt"]["exact"] == 0.25  # the GPT file picks candidate 0 like the base
    assert len(res["biome"]["jev_vocab"]["recall@3 / @5 / @10"]) == 3
    assert "gold reachable from start" in res["biome"]["topdown_root"]
    assert len(res["biome"]["jev_rerank"]["delta exact vs base (95% CI)"]) == 3
    assert set(res["cost"]) == {"flat request", "topdown_lca", "topdown_root"}


def test_backoff_climbs_to_a_confident_ancestor():
    ctx = jp.Ctx(toy_inputs(), args())
    o = pilot_record(gold=OC)
    # 0.55 ocean + 0.15 marine + 0.15 freshwater: q(marine) = .70, q(aquatic) = .85 (+ lake .15 = 1.0)
    P = {OC: 0.55, MA: 0.15, FW: 0.15, LK: 0.15}
    res = jp.backoff_points([(o, P, "own")], ctx, "biome", targets=(1.0,))
    cov, strict, exact, too_specific = res[1.0]
    assert cov == 1.0 and strict == 1.0 and exact == 0.0  # the most specific node above tau(=0.85) is aquatic: coarser


# ----------------------------------------------------------------------------- prepare
def test_prepare_counts_slot_use_per_study(tmp_path):
    import pandas as pd
    terms = pd.DataFrame([{"ontology": "ENVO", "term_id": t, "label": LABELS[t], "synonyms": "a||b||c||d",
                           "definition": "x" * 300, "parents": "||".join(p), "obsolete": "False"} for t, p in PARENTS.items()])
    terms.to_csv(tmp_path / "terms.tsv.gz", sep="\t", index=False)
    samples = pd.DataFrame([{"sample_id": f"S{i}", "study_code": st, "domain": "", "biome": b, "feature": "", "material": "",
                             "text": "t"} for i, (st, b) in enumerate([("s1", OC), ("s1", AQ), ("s2", "OLD"), ("s3", "")])])
    samples.to_csv(tmp_path / "samples.tsv.gz", sep="\t", index=False)
    (tmp_path / "map.tsv").write_text(f"slot\tfrom_id\tto_id\nbiome\tOLD\t{MA}\n")
    pilot = tmp_path / "pilot.jsonl"
    write_jsonl(pilot, [pilot_record("S9", "s9", OC, (OC, MA))])
    out = tmp_path / "inputs.json"
    jp.main(["prepare", "--pilot", str(pilot), "--ontology_terms", str(tmp_path / "terms.tsv.gz"),
             "--samples", str(tmp_path / "samples.tsv.gz"), "--require_embeddings", "--label_map",
             str(tmp_path / "map.tsv"), "--output", str(out)])
    inp = json.loads(out.read_text())
    assert inp["slots"]["biome"] == {OC: {"s1": 1}, AQ: {"s1": 1}, MA: {"s2": 1}}  # OLD -> marine; empty skipped
    assert set(inp["terms"]) == {OC, AQ, MA, B0}  # the labels, the pilot's terms and their ancestors
    assert inp["terms"][OC]["synonyms"] == ["a", "b", "c"] and len(inp["terms"][OC]["definition"]) == 220


# ----------------------------------------------------------------------------- the real SDK
def test_requests_are_valid_for_the_typesafe_sdk():
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        answers = {}
        for k, q in body["questions"].items():
            if q["type"] == "choice":
                names = list(q["criteria"])
                p = {n: (0.7 if n == "ocean biome" else 0.3 / (len(names) - 1)) for n in names}
                answers[k] = {"type": "choice", "choice": "ocean biome", "probabilities": p, "confidence": 0.6}
            else:
                answers[k] = {"type": "noul", "noul": 0.9}
        return httpx2.Response(200, json={"model": "jev-1.14.0", "answers": answers,
                                          "usage": {"input_tokens": 321, "output_tokens": 0}})

    async def go():
        client = sdk.AsyncTypeSafeClient(api_key="test-key", model="jev-1.14.0", compression=False,
                                         transport=httpx2.MockTransport(handler))
        async with client:
            ctx = jp.Ctx(toy_inputs(), args())
            return await jp.ask_flat(client, pilot_record(), ctx)
    rec = asyncio.run(go())
    body = seen[0]
    assert body["state"] == {"sample_metadata": {"text": pilot_record()["text"]}} and body.get("model") == "jev-1.14.0"
    assert body["questions"]["rerank_0"]["type"] == "choice" and body["questions"]["fits_0"]["type"] == "noul"
    assert body["questions"]["fits_0"]["criteria"]["false"].startswith("The term names a different")
    assert rec["model"] == "jev-1.14.0" and rec["usage"]["in"] == 321 and rec["content"] == "C"
    assert math.isclose(sum(rec["rerank"]), 1, abs_tol=1e-5) and max(rec["vocab"], key=rec["vocab"].get) == OC
