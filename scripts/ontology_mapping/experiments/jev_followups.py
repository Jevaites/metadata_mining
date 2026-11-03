#!/usr/bin/env python3
"""
Follow-up Jev experiments (README 7d, claude/jev-pilot-results.md; Vincent's questions of 2026-11-03). No API calls
here: these steps build the inputs and score the outputs; the API runs are `jev_pilot.py run` (and, for GPT,
`rerank_pilot.py run`) on the files written here.

  candidates  a pilot file in the format of `rerank_pilot.py build` (so `jev_pilot.py run` and `rerank_pilot.py run`
              both read it), with other candidates:
                --base prototype --source S --add 3   the prototype's candidates + the 3 best terms of source S that
                    are not already among them: does reranking turn the extra recall into right answers?
                    S = jev_vocab (Jev's whole-vocabulary Choice, from a flat run), prototype (the leave-one-study-out
                    prototype's next terms) or retrieval (zero-shot embedding retrieval over the same vocabulary),
                    exactly the sources of jev_baselines.py.
                --base none --source jev_vocab --add 3|5   Jev's own top k only: a second pass with the reranker's
                    richer descriptions. In `score`, "base" is then Jev's whole-vocabulary top 1.
              Every new candidate is described like the others: synonyms, definition, and the most similar sample
              from another study that the curators labelled with it. New candidates go to seeded random positions of
              the option order; the original candidates keep their order (the one GPT saw).
  open        retrieve, then choose, over the whole ontology. biome: the ENVO biome branch plus the Metalog biome
              labels outside it (one Choice holds them all, no retrieval); feature / material: the top --k (255) of
              embedding retrieval over every ENVO + UBERON term (label; synonyms, uncentred). Writes a pilot file
              (no example samples: most of these terms have none) and an inputs file that covers its terms. Run it
              with `jev_pilot.py run --arm flat --questions rerank`.
  compare     exact top-1 of several runs on the same samples (Jev rerank / Nouls / vocabulary, GPT files, the
              prototype), paired differences against the first, optionally as a gated reranker (--gate 0.3).
  score_open  the open Choice vs the prototype and Jev's Metalog-vocabulary Choice, on all samples and on the samples
              whose gold no other study uses (unseen labels): exact / correct but coarser / more specific / other
              branch, and how often the options held gold.

cd scripts/ontology_mapping; X=~/MicrobeAtlasProject/ontology_mapping/experiments; J=$X/jev
M=~/MicrobeAtlasProject/metalog/clean/previous_2026-10-01/biome_label_map.tsv
python3 experiments/jev_followups.py candidates --pilot $X/rerank/pilot_k5anc.jsonl --inputs $J/inputs.json --label_map $M \
    --base prototype --source jev_vocab --flat $J/flat.jsonl --add 3 --output $J/followups/gated_plus_jev3.jsonl
python3 experiments/jev_followups.py open --pilot $X/rerank/pilot_k5anc.jsonl --inputs $J/inputs.json --label_map $M \
    --output $J/followups/gated_open.jsonl --inputs_out $J/followups/inputs_gated_open.json
(the full list of commands is in README 7d)
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, path, term_ancestors  # noqa: E402
from hierarchy import NO_ANSWER  # noqa: E402
import jev_baselines as jb  # noqa: E402
import jev_pilot as jp  # noqa: E402

BIOME_ROOT = "ENVO_00000428"


def write_jsonl(rows, p):
    os.makedirs(os.path.dirname(path(p)) or ".", exist_ok=True)
    with open(path(p), "w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")


def term_entry(t, terms, example=""):
    """A candidate as rerank_pilot.py build writes it."""
    v = terms.get(t, {})
    return {"term_id": t, "label": v.get("label", t), "synonyms": v.get("synonyms", [])[:3],
            "definition": v.get("definition", "")[:220], "example": example}


def insert_new(order, n_old, n_new, rng):
    """Option order with the new candidates (indices n_old ... n_old + n_new - 1) at random positions; the old
    candidates keep their relative order."""
    order = list(order)
    for k in range(n_old, n_old + n_new):
        order.insert(int(rng.integers(0, len(order) + 1)), k)
    return order


# ----------------------------------------------------------------------------- candidates
def candidates(args):
    from sklearn.preprocessing import normalize
    pilot = jp.read_jsonl(args.pilot)
    inputs = json.load(open(path(args.inputs)))
    ctx = jp.Ctx(inputs, SimpleNamespace(min_studies=1))
    T = jb.load_training(args)
    Xn = normalize(T.X)
    flat = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(args.flat)} if args.flat else {}
    if args.source == "jev_vocab" and not flat:
        raise SystemExit("--source jev_vocab needs --flat (a jev_pilot.py run --arm flat output with the vocab question)")
    out, stats = [], {}
    for slot in SLOTS:
        ps = [o for o in pilot if o["slot"] == slot and (args.source != "jev_vocab" or (slot, o["sample_id"]) in flat)]
        if not ps:
            continue
        ranked = jb.loso_rankings(T, slot, ps, ctx)[args.source] if args.source != "jev_vocab" else None
        y = T.S[slot].to_numpy()
        for o in ps:
            sid = o["sample_id"]
            if ranked is None:
                P = flat[(slot, sid)]["vocab"]
                ranking = sorted(P, key=lambda t: -P[t])
            else:
                ranking = ranked[sid]
            base = list(o["candidates"]) if args.base == "prototype" else []
            have = {c["term_id"] for c in base}
            new = [t for t in ranking if t not in have][:args.add]
            x = normalize(T.vec([sid]))[0]

            def example(t):
                idx = np.where((y == t) & (T.studies != o["study"]))[0]
                return T.S["text"].iloc[int(idx[np.argmax(Xn[idx] @ x)])][:args.example_chars] if len(idx) else ""
            cands = base + [term_entry(t, inputs["terms"], example(t)) for t in new]
            rng = np.random.default_rng(jp.stable_seed(args.seed, slot, sid, args.source, args.add))
            ids = [c["term_id"] for c in cands]
            rec = dict(o, candidates=cands, gold_in_topk=o["gold"] in ids,
                       built={"base": args.base, "source": args.source, "add": args.add})
            if args.base == "prototype":
                rec.update(option_order=insert_new(o["option_order"], len(base), len(new), rng),
                           base_probs=list(o["base_probs"]) + [0.0] * len(new))
            else:  # Jev's own top k: its whole-vocabulary probabilities play the base model
                P = flat[(slot, sid)]["vocab"]
                probs = [float(P.get(t, 0.0)) for t in ids]
                rec.update(option_order=rng.permutation(len(cands)).tolist(), base_probs=probs, topk_mass=float(sum(probs)),
                           base_maxp=float(max(probs) if probs else 0.0), base_top1=ids[0] if ids else "")
            out.append(rec)
            stats.setdefault(slot, []).append((o["gold"] in [c["term_id"] for c in o["candidates"]], rec["gold_in_topk"], len(cands)))
    write_jsonl(out, args.output)
    print(f"wrote {len(out)} samples to {args.output} ({args.base} + {args.add} from {args.source})")
    for slot, v in stats.items():
        print(f"  {slot:8} gold among the candidates: before {np.mean([a for a, _, _ in v]):.3f}, after {np.mean([b for _, b, _ in v]):.3f}; "
              f"candidates per sample {np.mean([n for _, _, n in v]):.1f} (max {max(n for _, _, n in v)})")


# ----------------------------------------------------------------------------- open
def similar_samples(T, Xn, slot, o, info, args):
    """The args.similar training samples most similar to pilot sample o (cosine in the prototype's feature space),
    from other studies (at most args.similar_per_study per study), labelled for this slot. Their metadata is cut to
    args.example_chars. -> [{"text", "term_id", "term"}], most similar first."""
    y = T.S[slot].to_numpy()
    ok = np.where((y != "") & (T.studies != o["study"]))[0]
    x = T.vec([o["sample_id"]])[0]
    order = ok[np.argsort(-(Xn[ok] @ (x / np.linalg.norm(x))))]
    out, per = [], {}
    for i in order:
        st, t = T.studies[i], y[i]
        if per.get(st, 0) >= args.similar_per_study or t not in info:
            continue
        per[st] = per.get(st, 0) + 1
        out.append({"text": T.S["text"].iloc[int(i)][:args.example_chars], "term_id": t, "term": info[t]["label"]})
        if len(out) == args.similar:
            break
    return out


def open_vocabulary(args):
    pilot = jp.read_jsonl(args.pilot)
    inputs = json.load(open(path(args.inputs)))
    ctx = jp.Ctx(inputs, SimpleNamespace(min_studies=1))
    T = jb.load_training(args)
    terms = T.terms
    anc = term_ancestors(terms)
    info = {t: {"label": l, "synonyms": [s for s in syn.split("||") if s][:3], "definition": d[:220],
                "parents": sorted({p for p in par.split("||") if p})}
            for t, l, syn, d, par in zip(terms["term_id"], terms["label"], terms["synonyms"], terms["definition"], terms["parents"])}
    keep = np.array([t.startswith(("ENVO_", "UBERON_")) for t in terms["term_id"]])
    open_ids = terms["term_id"].to_numpy()[keep]
    TMo = T.TM[keep]
    branch = [t for t in info if t.startswith("ENVO_") and (t == BIOME_ROOT or BIOME_ROOT in anc.get(t, ())) and t not in NO_ANSWER]
    if args.similar:
        from sklearn.preprocessing import normalize
        Xn = normalize(T.X)
    out, needed, stats = [], set(), {}
    for o in pilot:
        slot, sid = o["slot"], o["sample_id"]
        similar = similar_samples(T, Xn, slot, o, info, args) if args.similar else []
        if slot == "biome":  # the branch fits one Choice; add the Metalog biome labels outside it (conventions)
            extra = [t for t in ctx.vocab(slot, o["study"])[0] if t not in set(branch) and t not in NO_ANSWER]
            ids = (branch + extra)[:jp.MAX_OPTIONS]
        else:
            sc = TMo @ T.vec([sid])[0]
            ids = [str(t) for t in open_ids[np.argsort(-sc)[:args.k]]]
        # the similar samples' terms are always options (in place of the last retrieved ones), so the examples
        # can be followed
        shown = [x["term_id"] for x in similar if x["term_id"] in info and x["term_id"] not in ids]
        shown = list(dict.fromkeys(shown))
        if shown:
            ids = ids[:jp.MAX_OPTIONS - len(shown)] + shown if len(ids) + len(shown) > jp.MAX_OPTIONS else ids + shown
        needed |= set(ids)
        rng = np.random.default_rng(jp.stable_seed(args.seed, slot, sid, "open"))
        rec = dict(o, candidates=[{"term_id": t, "label": info[t]["label"], "synonyms": info[t]["synonyms"],
                                   "definition": info[t]["definition"], "example": ""} for t in ids],
                   option_order=rng.permutation(len(ids)).tolist(), gold_in_topk=o["gold"] in ids,
                   prototype_top1=o["candidates"][0]["term_id"], base_probs=[0.0] * len(ids), topk_mass=0.0,
                   built={"open": "biome branch + Metalog biome labels" if slot == "biome" else f"retrieval top {args.k}",
                          "similar": args.similar})
        if similar:
            rec["similar"] = [{"text": x["text"], "term": x["term"]} for x in similar]
        out.append(rec)
        stats.setdefault(slot, []).append((rec["gold_in_topk"], len(ids), o["gold"] in ctx.vocab(slot, o["study"])[0]))
    needed |= set().union(*[anc.get(t, set()) for t in list(needed)])
    new_inputs = dict(inputs, terms=dict(inputs["terms"]))
    for t in needed:
        if t in info and t not in new_inputs["terms"]:
            new_inputs["terms"][t] = info[t]
    write_jsonl(out, args.output)
    os.makedirs(os.path.dirname(path(args.inputs_out)) or ".", exist_ok=True)
    json.dump(new_inputs, open(path(args.inputs_out), "w"))
    print(f"wrote {len(out)} samples to {args.output}, {len(new_inputs['terms'])} terms to {args.inputs_out}")
    for slot, v in stats.items():
        unseen = [g for g, _, inv in v if not inv]
        print(f"  {slot:8} options {np.mean([n for _, n, _ in v]):.0f} per sample; gold among them {np.mean([g for g, _, _ in v]):.3f} "
              f"(unseen-label samples: {len(unseen)}, gold among the options for {np.mean(unseen) if unseen else float('nan'):.3f})")


# ----------------------------------------------------------------------------- score_open
def cands_of(o):
    return [c["term_id"] for c in o["candidates"]]


def score_open(args):
    pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(args.pilot)}
    opened = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(args.open_pilot)}
    ctx = jp.Ctx(json.load(open(path(args.inputs_open))), SimpleNamespace(min_studies=1))
    resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(args.responses)}
    vocab = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(args.flat)} if args.flat else {}
    also = []  # other open-Choice variants on the same samples: name=open_pilot,responses
    for spec in args.also or []:
        name, rest = spec.split("=", 1)
        pf, rf = rest.split(",")
        also.append((name, {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pf)},
                     {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(rf)}))
    report = {}
    for slot in SLOTS:
        keys = [k for k in resp if k[0] == slot and k in opened and (not vocab or k in vocab)
                and all(k in a_resp for _, _, a_resp in also)]
        if not keys:
            continue
        picks = {"prototype": [], "open Choice": []}
        if vocab:
            picks["Jev, Metalog vocabulary"] = []
        for name, _, _ in also:
            picks[name] = []
        unseen, held, studies = [], [], []
        for k in keys:
            o, op = pilot[k], opened[k]
            ids = [c["term_id"] for c in op["candidates"]]
            picks["prototype"].append(o["candidates"][0]["term_id"])
            picks["open Choice"].append(ids[int(np.argmax(resp[k]["rerank"]))])
            if vocab:
                P = vocab[k]["vocab"]
                picks["Jev, Metalog vocabulary"].append(max(P, key=P.get) if P else None)
            for name, a_open, a_resp in also:
                picks[name].append(cands_of(a_open[k])[int(np.argmax(a_resp[k]["rerank"]))])
            unseen.append(o["gold"] not in ctx.vocab(slot, o["study"])[0])
            held.append(op["gold_in_topk"])
            studies.append(o["study"])
        gold = [pilot[k]["gold"] for k in keys]
        unseen, held = np.array(unseen), np.array(held)
        res = {}
        for subset, m in [("all", np.ones(len(keys), bool)), ("unseen labels", unseen), ("seen labels", ~unseen)]:
            if not m.any():
                continue
            row = {"n": int(m.sum()), "options hold gold": round(float(held[m].mean()), 3)}
            for name, pk in picks.items():
                rel = np.array([jp.relation(p, g, ctx.anc) for p, g in zip(pk, gold)])[m]
                row[name] = {x: round(float((rel == x).mean()), 3) for x in ("exact", "coarser", "too_specific", "other", "none")}
                row[name]["exact or coarser"] = round(row[name]["exact"] + row[name]["coarser"], 3)
            ex = {name: np.array([p == g for p, g in zip(pk, gold)]) for name, pk in picks.items()}
            for name in picks:
                if name != "open Choice":
                    row[f"open Choice - {name}, exact (95% CI)"] = jp.boot_diff(ex["open Choice"][m], ex[name][m], np.array(studies)[m])
            for name, _, _ in also:
                for ref in ("open Choice", "prototype"):
                    row[f"{name} - {ref}, exact (95% CI)"] = jp.boot_diff(ex[name][m], ex[ref][m], np.array(studies)[m])
            res[subset] = row
        report[slot] = res
        print(f"\n== {slot}")
        for subset, row in res.items():
            print(f"  [{subset}] n {row['n']}, options hold gold {row['options hold gold']}")
            for name in picks:
                print(f"    {name:26} " + ", ".join(f"{k} {v}" for k, v in row[name].items()))
            for k, v in row.items():
                if " - " in k and k.endswith("(95% CI)"):
                    print(f"    {k:60} {v}")
    if args.output:
        json.dump(report, open(path(args.output), "w"), indent=1)


# ----------------------------------------------------------------------------- compare
def compare(args):
    """Exact top-1 of several decisions on the same samples, side by side. Each run = name=pilot,responses,decision
    with decision rerank (Jev's rerank Choice + Nouls), fits (Jev's Nouls alone), vocab (Jev's whole-vocabulary
    Choice), gpt (a rerank_pilot.py file) or prototype (the pilot's first candidate; responses '-'). The first run is
    the reference for the paired differences; with --gate q, each run is also scored as a gated reranker: its pick
    on the reference pilot's q least-confident samples (by base_maxp), the reference's pick elsewhere."""
    from rerank_prompt import parse
    ctx = jp.Ctx(json.load(open(path(args.inputs))), SimpleNamespace(min_studies=1)) if args.inputs else None
    runs = []
    for spec in args.runs:
        name, rest = spec.split("=", 1)
        pf, rf, dec = rest.split(",")
        pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pf)}
        resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(rf)} if rf != "-" else None
        runs.append((name, pilot, resp, dec))

    def pick(o, rec, dec):
        c = [x["term_id"] for x in o["candidates"]]
        if dec == "prototype":
            return c[0] if c else None
        if dec == "vocab":
            P = rec["vocab"]
            return max(P, key=P.get) if P else None
        if dec == "gpt":
            d, p = parse(o, rec)
            return None if d is None or p == len(c) else c[p]
        if dec == "fits":
            p, _, _ = jp.decide_fits(rec["fits"], c, ctx.anc)
        else:
            p, _ = jp.decide_rerank(rec["rerank"], rec["fits"])
        return None if p == len(c) else c[p]
    report = {}
    for slot in SLOTS:
        keys = [k for k in runs[0][1] if k[0] == slot and all(k in p and (r is None or k in r) for _, p, r, _ in runs)]
        if not keys:
            continue
        ref_pilot = runs[0][1]
        gold = [ref_pilot[k]["gold"] for k in keys]
        st = [ref_pilot[k]["study"] for k in keys]
        conf = np.array([ref_pilot[k].get("base_maxp", 0.0) for k in keys])
        gated = np.zeros(len(keys), bool)
        if args.gate:
            gated[np.argsort(conf)[:int(round(args.gate * len(keys)))]] = True
        ex, rows = {}, {}
        for name, pilot, resp, dec in runs:
            picks = [pick(pilot[k], resp[k] if resp else None, dec) for k in keys]
            e = np.array([p == g for p, g in zip(picks, gold)])
            offered = np.array([g in [x["term_id"] for x in pilot[k]["candidates"]] for k, g in zip(keys, gold)])
            ex[name] = e
            rows[name] = {"exact": round(float(e.mean()), 3), "gold offered": round(float(offered.mean()), 3),
                          "exact | offered": round(float(e[offered].mean()), 3) if offered.any() else None}
            if ctx is not None:  # how the misses relate to gold (hierarchy.relation)
                rel = np.array([jp.relation(p, g, ctx.anc) for p, g in zip(picks, gold)])
                rows[name].update({x: round(float((rel == x).mean()), 3) for x in ("coarser", "too_specific", "none")})
        ref = runs[0][0]
        for name in rows:
            if name != ref:
                rows[name]["vs " + ref] = jp.boot_diff(ex[name], ex[ref], st)
            if args.gate:
                mix = np.where(gated, ex[name], ex[ref])
                rows[name][f"gated {int(args.gate * 100)} %"] = round(float(mix.mean()), 3)
                if name != ref:
                    rows[name][f"gated vs {ref}"] = jp.boot_diff(mix, ex[ref], st)
        report[slot] = rows
        print(f"\n== {slot} (n={len(keys)})")
        for name, r in rows.items():
            print(f"  {name:34} " + " | ".join(f"{k} {v}" for k, v in r.items()))
    if args.output:
        os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
        json.dump(report, open(path(args.output), "w"), indent=1)


def main(argv=None):
    from _setup import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)

    def training(x):
        for name in ["ontology_terms", "samples", "keywords", "sub_biomes", "term_vectors"]:
            x.add_argument(f"--{name}", default=DEFAULTS[name])
        x.add_argument("--label_map", default=None)
        x.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("candidates")
    training(c)
    c.add_argument("--pilot", required=True)
    c.add_argument("--inputs", required=True)
    c.add_argument("--base", choices=["prototype", "none"], default="prototype")
    c.add_argument("--source", choices=["jev_vocab", "prototype", "retrieval", "retrieval_prior"], required=True)
    c.add_argument("--flat", default=None, help="jev_pilot.py flat output (needed for --source jev_vocab)")
    c.add_argument("--add", type=int, default=3)
    c.add_argument("--example_chars", type=int, default=300)
    c.add_argument("--output", required=True)
    o = sub.add_parser("open")
    training(o)
    o.add_argument("--pilot", required=True)
    o.add_argument("--inputs", required=True)
    o.add_argument("--k", type=int, default=255, help="feature / material: options retrieved per sample")
    o.add_argument("--similar", type=int, default=0,
                   help="also give the n most similar labelled samples of other studies, with their term, in the state")
    o.add_argument("--similar_per_study", type=int, default=2)
    o.add_argument("--example_chars", type=int, default=300)
    o.add_argument("--output", required=True)
    o.add_argument("--inputs_out", required=True)
    s = sub.add_parser("score_open")
    s.add_argument("--pilot", required=True, help="the original pilot (the prototype's candidates)")
    s.add_argument("--open_pilot", required=True)
    s.add_argument("--inputs_open", required=True)
    s.add_argument("--responses", required=True, help="jev_pilot.py run --arm flat --questions rerank on the open pilot")
    s.add_argument("--flat", default=None, help="the original flat run (Jev's Metalog-vocabulary Choice)")
    s.add_argument("--also", nargs="*", default=None,
                   help="other open-Choice variants on the same samples, name=open_pilot,responses (e.g. with similar samples)")
    s.add_argument("--output", default=None)
    m = sub.add_parser("compare")
    m.add_argument("--runs", nargs="+", required=True, help="name=pilot,responses,decision (the first is the reference)")
    m.add_argument("--inputs", default=None, help="needed for the fits decision (ancestors)")
    m.add_argument("--gate", type=float, default=0.0, help="also score each run gated on the reference pilot's confidence")
    m.add_argument("--output", default=None)
    args = p.parse_args(argv)
    {"candidates": candidates, "open": open_vocabulary, "score_open": score_open, "compare": compare}[args.step](args)


if __name__ == "__main__":
    main()
