#!/usr/bin/env python3
"""
Step 7 (optional): LLM reranking of the least-confident atlas predictions, fused with the base
model, then hierarchical back-off. Design and pilot: claude/backoff-plus-rerank-design.md,
claude/rerank-pilot-results.md (round 3: top-5 + broader terms, gpt-4.1-mini, prompt v2).

  fit      (no API) from a Metalog pilot (experiments/rerank_pilot.py build --add_ancestors) and the
           LLM's responses to it, per slot: the fusion weight w, the temperature T2 of the fused
           distribution and the back-off threshold tau2 for --target_accuracy:
               p ∝ exp((log p_base + w · log p_llm) / T2) over the candidates (base mass outside
               the candidates kept aside), then hierarchy.decode_candidates at tau2.
           w is picked from a grid for the most exact answers (lenient: exact + too specific) at the
           target accuracy; w = 0 means the LLM does not help that slot. A nested study-grouped CV
           on the pilot gives an honest estimate, against the base model alone. -> rerank_params.json
           The gate (which samples get the LLM) is the pilot's: the --gate least-confident share of
           out-of-fold predictions, i.e. top-1 probability below that quantile of calibration.json.
  build    (no API) atlas samples of the slots that use the LLM (rerank_params.json) whose top-1
           probability is below the gate. One request per distinct (slot, metadata text,
           candidates); the candidates are 6_predict_atlas.py's <slot>_candidates (top-5 + broader
           slot labels), each shown with the most similar training sample curators labelled with it.
           -> requests.jsonl, members.tsv.gz (request -> samples), and the token count and cost.
  submit   (API) the requests through the OpenAI Batch API (half price, results within 24 h), at most
           --max_pending batches at a time. --limit N sends only the first N requests (a trial).
  collect  (API) checks the batches and appends finished results to responses.jsonl. Rerun until all
           are collected, then submit again for the remaining requests or failures.
  apply    (no API) final answer per sample and slot: the reranked back-off where an LLM answer
           exists, 6_predict_atlas.py's <slot>_backoff otherwise. -> atlas_final.tsv.gz:
             <slot>_final, <slot>_final_label, <slot>_final_p (summed probability of the answer),
             <slot>_final_source: backoff (not gated or slot without LLM), rerank, or
             backoff_no_llm_answer (gated, but no usable LLM response yet). "" = abstain.

The prompt and the parsing of the answer are experiments/rerank_pilot.py's (v2, letter
log-probabilities): the fitted w / T2 / tau2 are only valid for that exact prompt and model.

cd scripts/ontology_mapping; P=~/MicrobeAtlasProject; X=$P/ontology_mapping/experiments/rerank
R=$P/ontology_mapping/rerank_atlas; A=$P/ontology_mapping/atlas_backoff
python3 7_rerank_atlas.py fit --pilot $X/pilot_k5anc.jsonl --responses $X/resp_k5anc_gpt-4.1-mini.jsonl \
  --calibration $P/ontology_mapping/cv_backoff/calibration.json --samples $P/metalog/clean/training_set.clean.tsv.gz \
  --output $R/rerank_params.json
python3 7_rerank_atlas.py build --params $R/rerank_params.json --atlas_dir $A --sample_info $P/sample.info.gz \
  --samples $P/metalog/clean/training_set.clean.tsv.gz --train_vectors $P/metalog/keywords__large1024.npz \
  $P/metalog/sub_biomes__large1024.npz --keywords_h5 <keywords .h5> --sub_biomes_h5 <sub-biomes .h5> --work_dir $R
python3 7_rerank_atlas.py submit --work_dir $R --limit 2000      # trial; then without --limit
python3 7_rerank_atlas.py collect --work_dir $R                  # rerun until all collected
python3 7_rerank_atlas.py apply --params $R/rerank_params.json --atlas_dir $A --work_dir $R
"""
import argparse
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import hierarchy  # noqa: E402
from common import SLOTS, ancestor_sets, load_npz, load_terms, path, read_tsv, select_samples  # noqa: E402
from rerank_pilot import PRICES, count_tokens, parse, prompt  # noqa: E402  (the pilot's exact prompt)

PROMPT_VERSION = "v2"
W_GRID = (0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
EXACT, COARSER, TOO_SPECIFIC = hierarchy.EXACT, hierarchy.COARSER, hierarchy.TOO_SPECIFIC


# ----------------------------------------------------------------------------- fusion
def fuse(lb, D, w, T, mass):
    """Fused distribution over the candidates + 1 (base mass outside them)."""
    z = (lb + w * D) / T
    z = np.exp(z - z.max())
    return np.append(z / z.sum() * mass, max(1 - mass, 0.0))


def llm_logprob(o, responses):
    """Sum over LLMs of log p_llm(candidate), from parse() of each response; None if none usable."""
    k, total, n = len(o["candidates"]), np.zeros(len(o["candidates"])), 0
    for r in responses:
        dist, _ = parse(o, r)
        if dist is not None:
            total += np.log(np.maximum(dist[:k], 1e-6))
            n += 1
    return total if n else None


def fit_T(rows, w):
    def nll(logT):
        T = np.exp(logT)
        return -np.mean([np.log(fuse(r["lb"], r["D"], w, T, r["mass"])[r["gold_idx"]] + 1e-12) for r in rows])
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def outcomes(rows, w, T, tau, allowed, anc):
    """Outcome code per row (0 abstain, then hierarchy.EXACT / COARSER / TOO_SPECIFIC / OTHER)."""
    out = []
    for r in rows:
        p = fuse(r["lb"], r["D"], w, T, r["mass"])[:-1]
        node, _ = hierarchy.decode_candidates(r["cands"], p, tau, allowed, anc)
        g = r["gold"]
        out.append(0 if node is None else EXACT if node == g else COARSER if node in anc.get(g, ())
                   else TOO_SPECIFIC if g in anc.get(node, ()) else hierarchy.OTHER)
    return np.array(out, dtype=int)


def fit_slot(rows, allowed, anc, target, metric, grid=W_GRID):
    """-> best {w, T, tau, objective} and the per-w table; objective = exact share (strict) or
    exact + too specific (lenient) at the lowest tau reaching the target accuracy."""
    useful = (EXACT,) if metric == "strict" else (EXACT, TOO_SPECIFIC)
    table = []
    for w in grid:
        T = fit_T(rows, w)
        curve = [{"tau": float(t), **hierarchy.summarise(outcomes(rows, w, T, t, allowed, anc))} for t in hierarchy.TAUS]
        tau = hierarchy.choose_tau(curve, target, f"accuracy_{metric}")
        row = next(c for c in curve if c["tau"] == tau)
        objective = sum(row[{EXACT: "exact", TOO_SPECIFIC: "too_specific"}[u]] for u in useful)
        table.append({"w": w, "T": round(T, 4), "tau": tau, "objective": round(objective, 4), **row})
    best = max(table, key=lambda r: (r["objective"], -r["w"]))  # ties: the smaller weight
    return best, table


def pilot_rows(pilot_path, response_paths):
    pilot = [json.loads(l) for l in open(path(pilot_path))]
    responses = [{(r["slot"], r["sample_id"]): r for r in map(json.loads, open(path(f)))} for f in response_paths]
    rows = defaultdict(list)
    for o in pilot:
        key = (o["slot"], o["sample_id"])
        D = llm_logprob(o, [R[key] for R in responses if key in R])
        if D is None:
            continue
        cands = [c["term_id"] for c in o["candidates"]]
        rows[o["slot"]].append({"cands": cands, "lb": np.log(np.maximum(np.array(o["base_probs"]), 1e-9)), "D": D,
                                "mass": o["topk_mass"], "gold": o["gold"], "study": o["study"],
                                "gold_idx": cands.index(o["gold"]) if o["gold"] in cands else len(cands)})
    return rows


def fit(args):
    terms = load_terms(args.ontology_terms)
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p})
    samples = read_tsv(args.samples)
    calibration = json.load(open(path(args.calibration)))
    rows = pilot_rows(args.pilot, args.responses)
    params = {"pilot": args.pilot, "responses": args.responses, "models": [], "prompt_version": PROMPT_VERSION,
              "target_accuracy": args.target_accuracy, "accuracy": args.accuracy, "gate": args.gate,
              "method": args.method, "slots": {}}
    for f in args.responses:
        first = json.loads(open(path(f)).readline())
        params["models"].append(first.get("model", "?"))
    print(f"target accuracy {args.target_accuracy} ({args.accuracy}); w grid {W_GRID}; LLM: {params['models']}")
    for slot in SLOTS:
        R = rows.get(slot, [])
        if not R:
            continue
        allowed = set(samples[slot]) - {""}
        # honest estimate: nested CV over pilot studies (w, T2, tau2 fitted on 4/5, applied to 1/5)
        studies = np.array([r["study"] for r in R])
        fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(np.unique(studies)))}
        fold = np.array([fold_of[s] for s in studies])
        cv = {"base alone (w 0)": np.zeros(len(R), int), "base + LLM (w by CV)": np.zeros(len(R), int)}
        chosen = []
        for k in range(5):
            tr = [R[i] for i in np.where(fold != k)[0]]
            te_idx = np.where(fold == k)[0]
            te = [R[i] for i in te_idx]
            for name, grid in [("base alone (w 0)", (0,)), ("base + LLM (w by CV)", W_GRID)]:
                best, _ = fit_slot(tr, allowed, anc, args.target_accuracy, args.accuracy, grid)
                cv[name][te_idx] = outcomes(te, best["w"], best["T"], best["tau"], allowed, anc)
                if len(grid) > 1:
                    chosen.append(best["w"])
        best, table = fit_slot(R, allowed, anc, args.target_accuracy, args.accuracy)
        c = calibration[slot][args.method]
        gate_p = c["p_top1_quantiles"][str(args.gate)]
        use = slot in args.slots and best["w"] > 0
        estimate = {name: hierarchy.summarise(o) for name, o in cv.items()}
        params["slots"][slot] = {"use_llm": use, "w": best["w"], "T": best["T"], "tau": best["tau"], "gate_p": gate_p,
                                 "n_pilot": len(R), "cv_estimate": estimate, "cv_w_per_fold": chosen, "per_w": table}
        print(f"\n== {slot}: {len(R)} pilot samples; gate: top-1 probability < {gate_p} (the {args.gate:.0%} least confident)")
        print(f"   {'w':>5} {'T2':>7} {'tau2':>5} {'answered':>8} {'acc strict':>10} {'acc lenient':>11} {'exact':>6} {'too spec':>8}")
        for r in table:
            print(f"   {r['w']:>5} {r['T']:>7.3f} {r['tau']:>5} {r['coverage']:>8.2f} {r['accuracy_strict'] or 0:>10.3f} "
                  f"{r['accuracy_lenient'] or 0:>11.3f} {r['exact']:>6.3f} {r['too_specific']:>8.3f}")
        for name, e in estimate.items():
            print(f"   nested CV, {name:22}: answered {e['coverage']:.2f}, accuracy strict {e['accuracy_strict'] or 0:.3f} / "
                  f"lenient {e['accuracy_lenient'] or 0:.3f}, exact {e['exact']:.3f}, too specific {e['too_specific']:.3f}")
        print(f"   -> w {best['w']}, T2 {best['T']}, tau2 {best['tau']}; LLM {'ON' if use else 'off'} for {slot}"
              + ("" if slot in args.slots else " (not in --slots)") + (" (w = 0: no gain)" if best["w"] == 0 else ""))
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    json.dump(params, open(path(args.output), "w"), indent=1)
    print(f"\nwrote {args.output}")


# ----------------------------------------------------------------------------- build
def parse_candidates(s):
    out = []
    for item in s.split(";"):
        t, _, p = item.rpartition(":")
        out.append((t, float(p)))
    return out


def build(args):
    import h5py
    from sklearn.preprocessing import normalize
    build_set = importlib.import_module("2_build_training_set")  # the text cleaning of the training samples
    params = json.load(open(path(args.params)))
    slots = [s for s, p in params["slots"].items() if p["use_llm"]]
    if not slots:
        sys.exit("no slot uses the LLM in rerank_params.json: nothing to build")
    atlas_dir, work = path(args.atlas_dir), path(args.work_dir)
    os.makedirs(work, exist_ok=True)
    settings = json.load(open(os.path.join(atlas_dir, "run_settings.json")))
    if not settings.get("topk"):
        sys.exit(f"{atlas_dir} has no <slot>_candidates: run 6_predict_atlas.py with --calibration")
    terms = load_terms(args.ontology_terms)
    info = {t: {"label": l, "synonyms": [s for s in syn.split("||") if s][:3], "definition": d[:220]}
            for t, l, syn, d in zip(terms["term_id"], terms["label"], terms["synonyms"], terms["definition"])}
    code_to_label = dict(zip(read_tsv(args.ontology_terms)["term_id"], read_tsv(args.ontology_terms)["label"]))

    # 1. gated samples per slot
    cols = ["sample_id"] + [c for s in slots for c in (f"{s}_p", f"{s}_candidates")]
    atlas = pd.read_csv(os.path.join(atlas_dir, "atlas_predictions.tsv.gz"), sep="\t", dtype=str,
                        keep_default_na=False, usecols=cols)
    gated = {}
    for slot in slots:
        g = atlas[atlas[f"{slot}_p"].astype(float) < params["slots"][slot]["gate_p"]]
        gated[slot] = g[["sample_id", f"{slot}_candidates"]]
        print(f"{slot}: {len(g)} of {len(atlas)} atlas samples gated ({len(g) / len(atlas):.1%})", flush=True)
    del atlas
    needed = set().union(*[set(g["sample_id"]) for g in gated.values()])

    # 2. metadata texts (as 2_build_training_set.py writes them for the training samples)
    texts = {}
    for sample_id, lines in build_set.iter_sample_info(path(args.sample_info)):
        if sample_id in needed:
            texts[sample_id] = build_set.record_to_text(lines, code_to_label, 2000)[:args.text_chars]
    print(f"texts: {len(texts)} of {len(needed)} gated samples found in {args.sample_info}", flush=True)

    # 3. requests: one per distinct (slot, text, candidates); the first sample stands for the others
    requests, members = {}, []
    for slot in slots:
        for sid, cand in zip(gated[slot]["sample_id"], gated[slot][f"{slot}_candidates"]):
            if sid not in texts or not cand:
                continue
            key = hashlib.md5(f"{slot}\t{texts[sid]}\t{cand}".encode()).hexdigest()[:16]
            rid = f"{slot}-{key}"
            if rid not in requests:
                requests[rid] = {"slot": slot, "request_id": rid, "first_sample": sid, "text": texts[sid], "cand": cand}
            members.append((rid, sid))
    print(f"{len(members)} gated (sample, slot) pairs -> {len(requests)} distinct requests", flush=True)

    # 4. the example per candidate: the most similar training sample curators labelled with it
    (kw_row, kw), (sb_row, sb) = (load_npz(p) for p in args.train_vectors)
    train = select_samples(args.samples, [set(kw_row), set(sb_row)], args.max_per_study, args.seed)
    Xtr = normalize(np.hstack([kw[[kw_row[s] for s in train["sample_id"]]], sb[[sb_row[s] for s in train["sample_id"]]]]))
    rows_of = {slot: defaultdict(list) for slot in slots}
    for slot in slots:
        for i, t in enumerate(train[slot]):
            if t:
                rows_of[slot][t].append(i)
    rows_of = {slot: {t: np.array(v) for t, v in d.items()} for slot, d in rows_of.items()}
    index = dict(np.load(os.path.join(atlas_dir, "index.npz")))
    pos = {s: i for i, s in enumerate(index["sample_ids"])}
    reqs = list(requests.values())
    kw_rows = np.array([index["kw_rows"][pos[r["first_sample"]]] for r in reqs])
    sb_rows = np.array([index["sb_rows"][pos[r["first_sample"]]] for r in reqs])
    with h5py.File(path(args.sub_biomes_h5), "r") as h:
        SB = normalize(h["embeddings"][:])
    V = np.zeros((len(reqs), Xtr.shape[1]), dtype=np.float32)
    V[sb_rows >= 0, kw.shape[1]:] = SB[sb_rows[sb_rows >= 0]]
    with h5py.File(path(args.keywords_h5), "r") as h:  # streamed: rows of the 6 GB file in order
        n = h["embeddings"].shape[0]
        for first in range(0, n, args.chunk_rows):
            m = (kw_rows >= first) & (kw_rows < first + args.chunk_rows)
            if m.any():
                block = h["embeddings"][first:first + args.chunk_rows]
                V[m, :kw.shape[1]] = normalize(block[kw_rows[m] - first])
    V = normalize(V)
    with open(os.path.join(work, "requests.jsonl"), "w") as handle:
        for start in range(0, len(reqs), 2000):
            sim = V[start:start + 2000] @ Xtr.T
            for i, r in enumerate(reqs[start:start + 2000]):
                cands = parse_candidates(r["cand"])
                out = []
                for t, _ in cands:
                    idx = rows_of[r["slot"]].get(t)
                    example = train["text"].iloc[idx[np.argmax(sim[i, idx])]][:args.example_chars] if idx is not None else ""
                    out.append({"term_id": t, **info.get(t, {"label": t, "synonyms": [], "definition": ""}), "example": example})
                order = np.random.default_rng(int(r["request_id"].split("-")[1], 16)).permutation(len(out))
                rec = {"slot": r["slot"], "request_id": r["request_id"], "sample_id": r["request_id"], "text": r["text"],
                       "candidates": out, "base_probs": [p for _, p in cands], "topk_mass": float(sum(p for _, p in cands)),
                       "option_order": order.tolist()}
                handle.write(json.dumps(rec) + "\n")
    pd.DataFrame(members, columns=["request_id", "sample_id"]).to_csv(os.path.join(work, "members.tsv.gz"), sep="\t", index=False)
    estimate(os.path.join(work, "requests.jsonl"), args.model)


def estimate(requests_path, model, limit=None):
    n_in, n = 0, 0
    with open(requests_path) as handle:
        for i, line in enumerate(handle):
            if limit and i >= limit:
                break
            o = json.loads(line)
            n_in += sum(count_tokens(m["content"]) for m in prompt(o, "logprobs", PROMPT_VERSION))
            n += 1
    pin, pout = PRICES.get(model, (0, 0))
    cost = (n_in / 1e6 * pin + n * 4 / 1e6 * pout) * 0.5  # Batch API: half price
    print(f"{n} requests, ~{n_in:,} input tokens: ~${cost:,.2f} with {model} through the Batch API")
    return cost


# ----------------------------------------------------------------------------- batch API
def batch_paths(work):
    return (os.path.join(work, "requests.jsonl"), os.path.join(work, "responses.jsonl"),
            os.path.join(work, "batches.json"))


def client_for(args):
    from openai import OpenAI
    return OpenAI(api_key=open(path(args.api_key_path)).read().strip(), max_retries=8)


def body_for(o, model):
    body = {"model": model, "messages": prompt(o, "logprobs", PROMPT_VERSION), "max_completion_tokens": 16,
            "logprobs": True, "top_logprobs": 5 if model.startswith("gpt-5") else 10}
    if model.startswith("gpt-5.1"):
        body["reasoning_effort"] = "none"
    if not model.startswith(("gpt-5", "o")):
        body["temperature"] = 0
    return body


def submit(args):
    work = path(args.work_dir)
    req_path, resp_path, state_path = batch_paths(work)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"batches": []}
    pending = [b for b in state["batches"] if not b["collected"]]
    done = {json.loads(l)["request_id"] for l in open(resp_path)} if os.path.exists(resp_path) else set()
    in_flight = {rid for b in pending for rid in b["requests"]}
    todo = []
    with open(req_path) as handle:
        for i, line in enumerate(handle):
            if args.limit and i >= args.limit:
                break
            o = json.loads(line)
            if o["request_id"] not in done and o["request_id"] not in in_flight:
                todo.append(o)
    print(f"{len(done)} requests answered, {len(in_flight)} in {len(pending)} pending batch(es), {len(todo)} to send")
    slots_free = args.max_pending - len(pending)
    parts = [todo[i:i + args.batch_requests] for i in range(0, len(todo), args.batch_requests)][:max(slots_free, 0)]
    if not parts:
        print("nothing to submit now" + (" (pending batches: run collect)" if pending else ""))
        return
    cost = sum(estimate_part(p, args.model) for p in parts)
    print(f"-> {len(parts)} batch(es), {sum(len(p) for p in parts)} requests, ~${cost:,.2f}")
    if args.dry_run:
        return
    client = client_for(args)
    for part in parts:
        batch_input = os.path.join(work, f"batch_input_{len(state['batches'])}.jsonl")
        with open(batch_input, "w") as handle:
            for o in part:
                handle.write(json.dumps({"custom_id": o["request_id"], "method": "POST", "url": "/v1/chat/completions",
                                         "body": body_for(o, args.model)}) + "\n")
        upload = client.files.create(file=open(batch_input, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                      completion_window="24h", metadata={"job": "rerank_atlas"})
        state["batches"].append({"batch_id": batch.id, "model": args.model, "requests": [o["request_id"] for o in part],
                                 "collected": False})
        json.dump(state, open(state_path, "w"))  # after every batch: an interrupted submit is not lost
        print(f"submitted {batch.id}: {len(part)} requests ({os.path.getsize(batch_input) / 1e6:.0f} MB)")
        os.remove(batch_input)


def estimate_part(part, model):
    n_in = sum(count_tokens(m["content"]) for o in part for m in prompt(o, "logprobs", PROMPT_VERSION))
    pin, pout = PRICES.get(model, (0, 0))
    return (n_in / 1e6 * pin + len(part) * 4 / 1e6 * pout) * 0.5


def collect(args):
    work = path(args.work_dir)
    _, resp_path, state_path = batch_paths(work)
    if not os.path.exists(state_path):
        sys.exit("no batch submitted yet")
    state, client, waiting = json.load(open(state_path)), client_for(args), 0
    for b in state["batches"]:
        if b["collected"]:
            continue
        batch = client.batches.retrieve(b["batch_id"])
        c = batch.request_counts
        print(f"batch {batch.id}: {batch.status}" + (f" ({c.completed} done, {c.failed} failed of {c.total})" if c else ""))
        if batch.status in ("validating", "in_progress", "finalizing", "cancelling"):
            waiting += 1
            continue
        for e in (batch.errors.data if batch.errors and batch.errors.data else []):
            print(f"   error: {e.code}: {e.message}")
        n_ok = 0
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            with open(resp_path, "a") as handle:
                for line in client.files.content(file_id).text.splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    resp = r.get("response") or {}
                    if resp.get("status_code") != 200:
                        continue  # failed request: stays unanswered, the next submit retries it
                    body = resp["body"]
                    ch = body["choices"][0]
                    lp = None
                    content_lp = (ch.get("logprobs") or {}).get("content") or []
                    if content_lp:
                        first = next((t for t in content_lp if t["token"].strip()), content_lp[0])
                        lp = [[t["token"], t["logprob"]] for t in first.get("top_logprobs", [])]
                    u = body.get("usage") or {}
                    handle.write(json.dumps({"request_id": r["custom_id"], "model": body.get("model"),
                                             "content": ch["message"]["content"], "top_logprobs_raw": lp,
                                             "usage": {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}}) + "\n")
                    n_ok += 1
        print(f"   collected {n_ok} answers")
        b["collected"] = True
        json.dump(state, open(state_path, "w"))
    rows = [json.loads(l) for l in open(resp_path)] if os.path.exists(resp_path) else []
    pin, pout = PRICES.get(state["batches"][-1]["model"], (0, 0))
    cost = sum(r["usage"]["in"] * pin + r["usage"]["out"] * pout for r in rows) / 1e6 * 0.5
    print(f"{len(rows)} answers so far, ${cost:,.2f} measured" + (f"; {waiting} batch(es) still running" if waiting else ""))


# ----------------------------------------------------------------------------- apply
def apply(args):
    params = json.load(open(path(args.params)))
    work, atlas_dir = path(args.work_dir), path(args.atlas_dir)
    terms = load_terms(args.ontology_terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p})
    model = np.load(os.path.join(atlas_dir, "model.npz"))
    req_path, resp_path, _ = batch_paths(work)
    responses = {}
    if os.path.exists(resp_path):
        for line in open(resp_path):
            r = json.loads(line)
            responses[r["request_id"]] = r  # the latest answer wins
    answer = {}  # request_id -> (term or "", summed probability)
    if os.path.exists(req_path):
        for line in open(req_path):
            o = json.loads(line)
            r = responses.get(o["request_id"])
            if r is None:
                continue
            D = llm_logprob(o, [r])
            if D is None:
                continue
            p = params["slots"][o["slot"]]
            fused = fuse(np.log(np.maximum(np.array(o["base_probs"]), 1e-9)), D, p["w"], p["T"], o["topk_mass"])[:-1]
            allowed = set(model[f"{o['slot']}_classes"])
            node, q = hierarchy.decode_candidates([c["term_id"] for c in o["candidates"]], fused, p["tau"], allowed, anc)
            answer[o["request_id"]] = (node or "", q)
    members = read_tsv(os.path.join(work, "members.tsv.gz")) if os.path.exists(os.path.join(work, "members.tsv.gz")) \
        else pd.DataFrame(columns=["request_id", "sample_id"])
    cols = ["sample_id"] + [c for s in SLOTS for c in (f"{s}_p", f"{s}_backoff", f"{s}_backoff_p")]
    atlas = pd.read_csv(os.path.join(atlas_dir, "atlas_predictions.tsv.gz"), sep="\t", dtype=str,
                        keep_default_na=False, usecols=cols)
    out = pd.DataFrame({"sample_id": atlas["sample_id"]})
    for slot in SLOTS:
        final = np.array(atlas[f"{slot}_backoff"], dtype=object)
        final_p = np.array(atlas[f"{slot}_backoff_p"].replace("", "0"), dtype=float)
        source = np.full(len(atlas), "backoff", dtype=object)
        p = params["slots"].get(slot, {})
        if p.get("use_llm"):
            gated = atlas[f"{slot}_p"].astype(float).to_numpy() < p["gate_p"]
            source[gated] = "backoff_no_llm_answer"
            m = members[members["request_id"].str.startswith(slot + "-") & members["request_id"].isin(answer)]
            rows = pd.Series(np.arange(len(atlas)), index=atlas["sample_id"]).reindex(m["sample_id"]).to_numpy()
            ok = ~np.isnan(rows)
            rows, rids = rows[ok].astype(int), m["request_id"].to_numpy()[ok]
            final[rows] = [answer[r][0] for r in rids]
            final_p[rows] = [answer[r][1] for r in rids]
            source[rows] = "rerank"
        out[f"{slot}_final"] = final
        out[f"{slot}_final_label"] = [label_of.get(t, "") for t in final]
        out[f"{slot}_final_p"] = np.round(final_p, 4)
        out[f"{slot}_final_source"] = source
        counts = Counter(source)
        print(f"{slot}: " + ", ".join(f"{k} {v / len(out):.1%}" for k, v in counts.most_common())
              + f"; abstain {(final == '').mean():.1%}")
    target = os.path.join(work, "atlas_final.tsv.gz")
    out.to_csv(target, sep="\t", index=False, compression="gzip")
    print(f"wrote {target}")


# ----------------------------------------------------------------------------- main
def main():
    P = "~/MicrobeAtlasProject"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--pilot", required=True, help="experiments/rerank_pilot.py build --add_ancestors output")
    f.add_argument("--responses", nargs="+", required=True, help="rerank_pilot.py run output(s) for that pilot")
    f.add_argument("--calibration", required=True, help="5_evaluate.py calibration.json (for the gate)")
    f.add_argument("--samples", default=f"{P}/metalog/clean/training_set.clean.tsv.gz", help="the training set (slot labels)")
    f.add_argument("--method", default="prototype", help="base model of 6_predict_atlas.py")
    f.add_argument("--slots", nargs="+", default=["biome", "feature"],
                   help="slots that may use the LLM (pilot: no gain on material with the strict metric)")
    f.add_argument("--gate", type=float, default=0.3, help="share of least-confident predictions (as the pilot)")
    f.add_argument("--target_accuracy", type=float, default=0.9)
    f.add_argument("--accuracy", choices=["strict", "lenient"], default="strict")
    f.add_argument("--output", required=True)
    b = sub.add_parser("build")
    b.add_argument("--params", required=True)
    b.add_argument("--atlas_dir", required=True, help="6_predict_atlas.py --output_dir (run with --calibration)")
    b.add_argument("--sample_info", default=f"{P}/sample.info.gz")
    b.add_argument("--samples", default=f"{P}/metalog/clean/training_set.clean.tsv.gz",
                   help="the training set of 6_predict_atlas.py (examples)")
    b.add_argument("--train_vectors", nargs=2, required=True, help="keywords .npz, sub_biomes .npz (as 6_predict_atlas.py)")
    b.add_argument("--keywords_h5", required=True)
    b.add_argument("--sub_biomes_h5", required=True)
    b.add_argument("--max_per_study", type=int, default=50)
    b.add_argument("--seed", type=int, default=22)
    b.add_argument("--text_chars", type=int, default=1200)
    b.add_argument("--example_chars", type=int, default=300)
    b.add_argument("--chunk_rows", type=int, default=50_000)
    b.add_argument("--model", default="gpt-4.1-mini", help="for the cost estimate")
    b.add_argument("--work_dir", required=True)
    for name in ("submit", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--work_dir", required=True)
        s.add_argument("--api_key_path", default=f"{P}/my_api_key")
        if name == "submit":
            s.add_argument("--model", default="gpt-4.1-mini", help="must be the model the params were fitted for")
            s.add_argument("--limit", type=int, default=None, help="only the first N requests (trial)")
            s.add_argument("--batch_requests", type=int, default=20_000)
            s.add_argument("--max_pending", type=int, default=3, help="batches in flight (enqueued-token limits)")
            s.add_argument("--dry_run", action="store_true")
    a = sub.add_parser("apply")
    a.add_argument("--params", required=True)
    a.add_argument("--atlas_dir", required=True)
    a.add_argument("--work_dir", required=True)
    for s in (f, b, a):
        s.add_argument("--ontology_terms", default=f"{P}/ontology_terms.tsv.gz")
    args = p.parse_args()
    {"fit": fit, "build": build, "submit": submit, "collect": collect, "apply": apply}[args.step](args)


if __name__ == "__main__":
    main()
