"""
Tests for experiments/jev_open_topdown.py (top-down over open vocabularies). No network: a scripted client.

Toy ontology:
  anatomical entity (UBERON_0001062) -> gut
  environmental material (ENVO_00010483, a root) -> water -> sea water, lake water
                                                -> soil  -> garden soil
  environmental system (ENVO_01000254, a root) -> biome (ENVO_00000428) -> marine biome
The material slot uses water, sea water and soil (other studies); lake water and garden soil are unseen labels.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import jev_open_topdown as T  # noqa: E402
import jev_pilot as jp  # noqa: E402
from test_jev_pilot import Scripted, run, write_jsonl  # noqa: E402

AE, GUT = "UBERON_0001062", "UBERON_0000160"
EM, WA, SW, LW, SO, GS = "ENVO_00010483", "ENVO_00000001", "ENVO_00000002", "ENVO_00000003", "ENVO_00000004", "ENVO_00000005"
ES, BI, MB = "ENVO_01000254", "ENVO_00000428", "ENVO_00000006"
LABELS = {AE: "anatomical entity", GUT: "gut", EM: "environmental material", WA: "water", SW: "sea water",
          LW: "lake water", SO: "soil", GS: "garden soil", ES: "environmental system", BI: "biome", MB: "marine biome"}
PARENTS = {AE: [], GUT: [AE], EM: [], WA: [EM], SW: [WA], LW: [WA], SO: [EM], GS: [SO], ES: [], BI: [ES], MB: [BI]}


def inputs():
    terms = {t: {"label": LABELS[t], "synonyms": [], "definition": f"definition of {LABELS[t]}", "parents": p}
             for t, p in PARENTS.items()}
    use = {"biome": {MB: {"s1": 2}}, "feature": {}, "material": {WA: {"s1": 3}, SW: {"s2": 2}, SO: {"s1": 1}}}
    return {"terms": terms, "slots": use}


def forest():
    inp = inputs()
    return T.Forest(jp.Ctx(inp, SimpleNamespace(min_studies=1)), T.open_roots(inp["terms"]))


def record(gold=LW, slot="material"):
    return {"slot": slot, "sample_id": "S1", "study": "own", "gold": gold, "text": "water from a freshwater lake",
            "candidates": [{"term_id": t} for t in (SW, WA, SO)], "open5": [WA, SW, LW], "retrieved": [LW, SW, WA, GS, SO]}


def args(**kw):
    d = dict(tree="auto", start="root", beam=3, seed=0, model="jev-test", wide="split", finalists=3)
    d.update(kw)
    return SimpleNamespace(**d)


def test_roots_trees_and_answers():
    F = forest()
    assert set(F.roots) == {AE, EM, ES}
    tr = F.tree("auto", "material", "own")
    assert tr["nodes"] == set(PARENTS) and LW in tr["answers"]
    assert not {AE, EM, ES, BI} & tr["answers"]  # roots and NO_ANSWER are never answers
    b = F.tree("auto", "biome", "own")
    assert b["nodes"] == {BI, MB} and b["top"] == [MB] and b["answers"] == {MB}


def test_climb_cover_is_open_and_leave_one_study_out():
    F = forest()
    # from sea water: water (3 terms) <= 3, environmental material (6) > 3 -> water; soil stays soil
    assert F.climb_cover([WA, SW, SO], 3) == [WA, SO]
    tr = F.tree("climb3", "material", "own")
    assert tr["top"] == [WA, SO] and tr["nodes"] == {WA, SW, LW, SO, GS}  # lake water, garden soil: unseen but inside
    assert F.tree("climb3", "material", "s1")["top"] == [WA]  # s1's own labels (water, soil) are left out: sea water -> water


def test_lca_starts():
    F = forest()
    tr = F.tree("auto", "material", "own")
    assert T.start_node(F, record(), tr, "lca_proto3") == EM  # sea water, water, soil
    assert T.start_node(F, record(), tr, "lca_open5") == WA  # water, sea water, lake water
    assert T.start_node(F, record(), tr, "root") == jp.ROOT


def test_walk_reaches_an_unseen_label():
    F = forest()
    client = Scripted(weights={"environmental material": 5, "water": 5, "lake water": 5})
    rec = run(T.ask(client, record(), F, args()))
    assert max(rec["terminals"], key=rec["terminals"].get) == LW
    assert rec["gold_in_tree"] and rec["gold_reachable"] and rec["gold_alive_end"] and not rec["gold_cut"]
    assert rec["best_path"] == [jp.ROOT, EM, WA, LW] and rec["requests"] == rec["levels"] == 3
    # the 'water' node offers its children and 'water itself'
    names = set(client.requests[2][1]["n0"]["criteria"])
    assert {"sea water", "lake water"} <= names and any("itself" in n for n in names) and len(names) == 3


def test_width_limit_keeps_the_children_closest_to_the_sample(monkeypatch):
    F = forest()
    tr = F.tree("auto", "material", "own")
    monkeypatch.setattr(jp, "MAX_OPTIONS", 2)
    o = dict(record(gold=GS), retrieved=[GS, SW])
    opts, cut, q = T.fit_options(F, EM, tr, o, 0, (0,))
    assert cut == 0 and {x[1] for x in opts} == {WA, SO}  # 2 children fit: nothing cut
    opts, cut, _ = T.fit_options(F, WA, tr, o, 0, (0,))  # sea water, lake water + itself = 3 > 2
    assert cut == 1 and {x[1] for x in opts} == {SW, None}  # sea water is retrieved, lake water is not; itself stays


def test_a_wide_node_is_split_then_decided_in_a_final_round(monkeypatch):
    inp = inputs()
    extra = {"ENVO_00000007": "river water", "ENVO_00000008": "ground water"}
    for t, label in extra.items():
        inp["terms"][t] = {"label": label, "synonyms": [], "definition": "", "parents": [WA]}
    F = T.Forest(jp.Ctx(inp, SimpleNamespace(min_studies=1)), T.open_roots(inp["terms"]))
    monkeypatch.setattr(jp, "MAX_OPTIONS", 3)  # water: 4 children + itself = 5 options > 3
    opts, chunks = T.chunk_questions(F, WA, F.tree("auto", "material", "own"), record(), 0, (0,))
    assert len(chunks) == 2 and sorted(i for _, idx in chunks for i in idx) == [i for i, x in enumerate(opts) if x[1]]
    client = Scripted(weights={"environmental material": 5, "water": 5, "lake water": 5, "river water": 2})
    rec = run(T.ask(client, record(), F, args(finalists=1)))
    assert max(rec["terminals"], key=rec["terminals"].get) == LW and rec["nodes_cut"] == 1 and not rec["gold_cut"]
    final = client.requests[-1][1]
    assert list(final) == ["n0f"] and len(final["n0f"]["criteria"]) == 3  # one finalist per chunk + 'water itself'


def test_split_requests():
    qs = {f"n{i}": ({"q": i}, 25000) for i in range(5)}
    groups = T.split_requests(qs, 1000)
    assert [len(g) for g in groups] == [2, 2, 1]


def test_run_and_score(tmp_path):
    inp = tmp_path / "inputs.json"
    inp.write_text(json.dumps(dict(inputs(), open_roots=[AE, EM, ES])))
    pilot = tmp_path / "pilot.jsonl"
    write_jsonl(pilot, [record()])
    out = tmp_path / "td.jsonl"
    T.main(["run", "--pilot", str(pilot), "--inputs", str(inp), "--fake", "--output", str(out), "--cache",
            str(tmp_path / "cache.jsonl")])
    assert len(jp.read_jsonl(out)) == 1
    op = tmp_path / "open.jsonl"
    write_jsonl(op, [dict(record(), candidates=[{"term_id": t} for t in (WA, LW)])])
    resp = tmp_path / "open_resp.jsonl"
    write_jsonl(resp, [{"slot": "material", "sample_id": "S1", "rerank": [0.2, 0.8]}])
    score = tmp_path / "score.json"
    T.main(["score", "--pilot", str(pilot), "--inputs", str(inp), "--responses", str(out), "--open",
            f"open={op},{resp}", "--output", str(score)])
    res = json.loads(score.read_text())["material"]
    assert res["unseen labels"]["n"] == 1 and res["unseen labels"]["open"]["exact"] == 1.0
    assert res["all"]["prototype"]["other"] == 1.0  # sea water vs lake water: siblings


DS, CO = "UBERON_0001007", "UBERON_0001155"  # digestive system; colon (part_of digestive system, not is_a)


def curated_forest(links, tmp_path):
    inp = inputs()
    inp["terms"][DS] = {"label": "digestive system", "synonyms": [], "definition": "", "parents": [AE], "part_of": []}
    inp["terms"][CO] = {"label": "colon", "synonyms": [], "definition": "", "parents": [AE], "part_of": [DS]}
    cur = tmp_path / "curated.tsv"
    head = "slot\tgroup\tgroup_description\tcategory\tcategory_description\tanchor_id\tanchor_label\n"
    cur.write_text(head + f"material\tHost-derived\thost material\tGut\tgut material\t{DS}\tdigestive system\n"
                   f"material\tWater and liquids\twaters\tNatural water\tsea, lake water\t{WA}\twater\n")
    return T.Forest(jp.Ctx(inp, SimpleNamespace(min_studies=1)), T.open_roots(inp["terms"]), links=links,
                    curated=T.load_curated(cur))


def test_curated_top_layer_and_part_of_links(tmp_path):
    isa = curated_forest("isa", tmp_path).tree("curated", "material", "own")
    both_f = curated_forest("both", tmp_path)
    both = both_f.tree("curated", "material", "own")
    assert len(isa["top"]) == 2 and all(v.startswith("CUR:material:G:") for v in isa["top"])
    assert CO not in isa["nodes"] and CO in both["nodes"]  # colon hangs under the digestive system by part_of only
    assert both["name"] == "curated_both" and not set(both["virtual"]) & both["answers"]
    g = [v for v in both["top"] if v.endswith("Host-derived")][0]
    assert both_f.below(g, both) == {DS, CO} and both_f.children(DS, both) == [CO]
    client = Scripted(weights={"Water and liquids": 5, "lake water": 5})
    rec = run(T.ask(client, record(), both_f, args(tree="curated")))
    assert max(rec["terminals"], key=rec["terminals"].get) == LW
    assert rec["best_path"][-2:] == [WA, LW] and rec["requests"] == 2  # group, category, anchor: single options pass
    root_q = client.requests[0][1]["n0"]
    assert set(root_q["criteria"]) == {"Host-derived", "Water and liquids"}
    assert root_q["criteria"]["Host-derived"]["narrower terms include"] == "Gut"


def test_add_links_reads_part_of_from_obo(tmp_path):
    inp = inputs()
    inp["terms"][DS] = {"label": "digestive system", "synonyms": [], "definition": "", "parents": [AE]}
    inp["terms"][CO] = {"label": "colon", "synonyms": [], "definition": "", "parents": [AE]}
    f = tmp_path / "inputs.json"
    f.write_text(json.dumps(inp))
    obo = tmp_path / "u.obo"
    obo.write_text("[Term]\nid: UBERON:0001155\nname: colon\nrelationship: part_of UBERON:0001007 ! digestive system\n"
                   "[Term]\nid: ENVO:00000003\nrelationship: BFO:0000050 ENVO:00000001\n")
    T.main(["add_links", "--inputs", str(f), "--obo", str(obo)])
    terms = json.loads(f.read_text())["terms"]
    assert terms[CO]["part_of"] == [DS] and terms[LW]["part_of"] == [WA] and terms[SW]["part_of"] == []
