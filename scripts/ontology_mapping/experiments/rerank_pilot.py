#!/usr/bin/env python3
"""
Real LLM reranker pilot (claude/backoff-plus-rerank-design.md, hierarchical-backoff-results.md).

The simulation in hierarchical_backoff.py says what a reranker would bring IF it reached accuracy r
inside the base top-5 with confidence AUROC a. This pilot measures r and a for real LLMs on the
samples the reranker would actually see: the least-confident share of each slot (out of fold).

Steps:
  build   (no API) out-of-fold prototype probabilities (fold seed 0, cross-fitted temperature, the
          biome label map applied), the gated least-confident --gate share per slot, a random
          --n_per_slot of them. Per sample: the metadata text, the top-5 candidate terms (label,
          synonyms, definition), and for each candidate the most similar *training* sample that
          Metalog labelled with it (shows the curators' convention). -> pilot.jsonl
  run     (API; run on your Mac) one request per sample: options A-E (shuffled) + F "none of these",
          answer = one letter. With --confidence logprobs the letter's token log-probabilities
          give the reranker's distribution over A-F; with verbal the model also states a 0-100
          confidence. Resumable; --dry_run prints the token count and cost.
  score   (no API) reranker accuracy, r = accuracy when gold is in the top-5, base accuracy on the
          same samples, "none" use, AUROC of the reranker's confidence, and the fused result
          (base^a * reranker^b, a and b fitted by study-grouped cross-validation on the pilot).
          Prints the hierarchical_backoff.py command that projects the measured r / AUROC onto
          the whole slot.

cd scripts/ontology_mapping; X=~/MicrobeAtlasProject/ontology_mapping/experiments/rerank
python3 experiments/rerank_pilot.py build --label_map ~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv --output $X/pilot.jsonl
python3 experiments/rerank_pilot.py run --pilot $X/pilot.jsonl --model gpt-4.1-mini --output $X/resp_gpt-4.1-mini.jsonl --dry_run
python3 experiments/rerank_pilot.py run --pilot $X/pilot.jsonl --model gpt-4.1-mini --output $X/resp_gpt-4.1-mini.jsonl
python3 experiments/rerank_pilot.py score --pilot $X/pilot.jsonl --responses $X/resp_gpt-4.1-mini.jsonl
python3 experiments/rerank_pilot.py vote --pilot $X/pilot.jsonl --responses $X/resp_v2_gpt-4.1-mini.jsonl $X/resp_v2_gpt-5.1.jsonl
  vote: base + any combination of rerankers, weights tuned for accuracy by study-grouped CV.
  Any OpenAI-compatible endpoint works for `run` via --base_url / --api_key_path (e.g. Qwen3 on a
  local vLLM server or a hosted provider), as long as it returns token log-probabilities.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common import SLOTS, path  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVW"  # up to 22 candidates + "none"
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4.1-nano": (0.10, 0.40),
          "gpt-5.1": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0)}  # $ / 1M input, output tokens
SLOT_TEXT = {
    "biome": "broad-scale environment (MIxS env_broad_scale): the biome or major environmental system the sample comes from",
    "feature": "local environment (MIxS env_local_scale): the environmental feature or host part the sample was taken from",
    "material": "environmental material (MIxS env_medium): the material that was sampled",
}


# ----------------------------------------------------------------------------- build
def build(args):
    import hierarchical_backoff as hb
    from sklearn.preprocessing import normalize
    from common import ancestor_sets, load_npz, load_terms, read_tsv, select_samples, study_folds
    terms = load_terms(args.ontology_terms)
    ids = terms["term_id"].to_numpy()
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(ids, terms["parents"]) if p})
    ROOTS = {"ENVO_00000428", "ENVO_00010483", "ENVO_01000254"}  # biome, environmental material, environmental system
    info = {t: {"label": l, "synonyms": [s for s in syn.split("||") if s][:3], "definition": d[:220]}
            for t, l, syn, d in zip(ids, terms["label"], terms["synonyms"], terms["definition"])}
    TM = hb.term_matrix(terms, args.term_vectors)
    trow = {t: i for i, t in enumerate(ids)}
    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    S = select_samples(args.samples, [set(kr), set(sr)])
    if args.label_map:
        lm = read_tsv(args.label_map)
        for slot in SLOTS:
            m = {r.from_id.strip(): r.to_id.strip() for r in lm.itertuples() if r.slot.strip() in (slot, "*")}
            S[slot] = S[slot].map(lambda t: m.get(t, t))
    X = np.hstack([K[[kr[s] for s in S["sample_id"]]], B[[sr[s] for s in S["sample_id"]]]]) / np.sqrt(2)
    Xn = normalize(X)
    studies = S["study_code"].to_numpy()
    rng = np.random.default_rng(args.seed)
    rng_order = np.random.default_rng(args.seed + 1)

    def order_for(n):
        # the main stream always draws a 5-permutation, so the sampled pilot samples are the same
        # whatever --k / --add_ancestors (paired comparisons); other sizes take a second stream
        p5 = rng.permutation(5)
        return p5.tolist() if n == 5 else rng_order.permutation(n).tolist()
    out = []
    for slot in args.slots:
        y = S[slot].to_numpy()
        F = []
        for fold, (tr, te) in enumerate(study_folds(studies, 5, args.fold_seed)):
            a, b = tr[y[tr] != ""], te[y[te] != ""]
            vocab = np.unique(y[a])
            col = {v: i for i, v in enumerate(vocab)}
            F.append({"fold": fold, "train": a, "test": b, "vocab": vocab,
                      "S": hb.base_scores("prototype", X[a], y[a], X[b], vocab, TM[[trow[v] for v in vocab]]),
                      "gcol": np.array([col.get(g, -1) for g in y[b]])})
        pool = []
        for k, f in enumerate(F):
            T = hb.fit_temperature([(g["S"], g["gcol"]) for j, g in enumerate(F) if j != k])
            f["P"] = hb.softmax(f["S"], T)
            gated = np.argsort(f["P"].max(1))[:int(round(args.gate * len(f["P"])))]  # same gate as the simulation
            pool += [(k, i) for i in gated]
        pick = rng.choice(len(pool), size=min(args.n_per_slot, len(pool)), replace=False)
        for k, i in sorted(pool[j] for j in pick):
            f = F[k]
            s = f["test"][i]
            order = list(np.argsort(-f["P"][i])[:args.k])
            if args.add_ancestors:  # also offer the candidates' broader terms that are labels in this slot
                col = {v: j for j, v in enumerate(f["vocab"])}
                extra = {col[a] for c in order for a in anc.get(f["vocab"][c], ()) if a in col and a not in ROOTS} - set(order)
                order += sorted(extra, key=lambda c: -f["P"][i, c])[:max(0, args.max_candidates - len(order))]
            order = np.array(order)
            cands = [f["vocab"][c] for c in order]
            examples = {}
            for c in cands:  # the most similar training sample curators labelled with this candidate
                idx = f["train"][y[f["train"]] == c]
                if len(idx):
                    j = idx[np.argmax(Xn[idx] @ Xn[s])]
                    examples[c] = S["text"].iloc[j][:args.example_chars]
            out.append({"slot": slot, "sample_id": S["sample_id"].iloc[s], "study": studies[s], "fold": int(f["fold"]),
                        "gold": y[s], "gold_in_topk": bool(y[s] in cands), "base_top1": cands[0],
                        "base_probs": [float(f["P"][i, c]) for c in order],
                        "topk_mass": float(f["P"][i, order].sum()), "base_maxp": float(f["P"][i].max()),
                        "text": S["text"].iloc[s][:args.text_chars],
                        "candidates": [{"term_id": c, **info[c], "example": examples.get(c, "")} for c in cands],
                        "option_order": order_for(len(cands))})
        print(f"{slot}: {len(F[0]['vocab'])}+ labels, gated pool {len(pool)}, picked {len(pick)}; "
              f"gold in top-{args.k}: {np.mean([o['gold_in_topk'] for o in out if o['slot'] == slot]):.2f}, "
              f"base top-1: {np.mean([o['gold'] == o['base_top1'] for o in out if o['slot'] == slot]):.2f}", flush=True)
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    with open(path(args.output), "w") as handle:
        for o in out:
            handle.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} samples to {args.output}")


# ----------------------------------------------------------------------------- run
GRANULARITY = ("Several options can be true at different levels of detail (e.g. 'sediment' and 'marine sediment'). "
               "Choose the level of detail the curators would use: follow the examples, and prefer the more general "
               "term unless the metadata explicitly supports the more specific one.")


def prompt(o, confidence, version="v2"):
    opts = [o["candidates"][i] for i in o["option_order"]]
    lines = [f"Sample metadata:\n{o['text']}\n",
             f"Choose the ontology term that best describes this sample's {SLOT_TEXT[o['slot']]}.",
             "Each option shows how expert curators used the term, with the most similar sample they labelled with it."]
    if version == "v2":
        lines.append(GRANULARITY)
    lines.append("")
    for letter, c in zip(LETTERS, opts):
        syn = f" (synonyms: {'; '.join(c['synonyms'])})" if c["synonyms"] else ""
        lines.append(f"{letter}. {c['label']}{syn}")
        if c["definition"]:
            lines.append(f"   definition: {c['definition']}")
        if c["example"]:
            lines.append(f"   example sample labelled with it: {c['example']}")
    lines.append(f"{LETTERS[len(opts)]}. none of these terms fits")
    if confidence == "logprobs":
        lines.append("\nAnswer with the single letter of the best option, nothing else.")
    else:
        lines.append("\nAnswer with the letter of the best option and your confidence from 0 to 100 that it is right, "
                     "e.g. 'B 70', nothing else.")
    system = ("You are an expert curator of microbiome sample metadata who annotates samples with ENVO and UBERON "
              "terms, following the conventions of the Metalog database.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def run(args):
    pilot = [json.loads(l) for l in open(path(args.pilot))]
    done = set()
    if os.path.exists(path(args.output)):
        done = {(r["slot"], r["sample_id"]) for r in map(json.loads, open(path(args.output)))}
    todo = [o for o in pilot if (o["slot"], o["sample_id"]) not in done]
    n_in = sum(count_tokens(m["content"]) for o in todo for m in prompt(o, args.confidence, args.prompt_version))
    pin, pout = PRICES.get(args.model, (args.price_in or 0, args.price_out or 0))
    n_out = len(todo) * (4 if args.confidence == "logprobs" else 8)
    print(f"{len(pilot)} samples, {len(done)} done, {len(todo)} to send: ~{n_in:,} input / ~{n_out:,} output tokens "
          f"= ${n_in / 1e6 * pin + n_out / 1e6 * pout:.2f} with {args.model} (${pin}/${pout} per 1M)")
    if args.dry_run or not todo:
        return
    from openai import OpenAI
    client = OpenAI(api_key=open(path(args.api_key_path)).read().strip(), base_url=args.base_url, max_retries=8)
    lock, state = threading.Lock(), {"logprobs": args.confidence == "logprobs"}

    def body_for(o, use_logprobs):
        mode = "logprobs" if use_logprobs else "verbal"
        body = {"model": args.model, "messages": prompt(o, mode, args.prompt_version),
                "max_completion_tokens": args.max_tokens or (16 if use_logprobs else 24)}
        if args.model.startswith("gpt-5.1"):  # no hidden reasoning tokens eating the budget
            body["reasoning_effort"] = "none"
        if not args.model.startswith(("gpt-5", "o")):
            body["temperature"] = 0
        if use_logprobs:  # gpt-5.x accepts at most 5
            body.update(logprobs=True, top_logprobs=5 if args.model.startswith("gpt-5") else 10)
        return body, mode

    def call(o):
        body, mode = body_for(o, state["logprobs"])
        try:
            try:
                r = client.chat.completions.create(**body)
            except Exception as e:
                if "max_tokens" not in str(e) and "output limit" not in str(e):
                    raise
                body["max_completion_tokens"] *= 8  # rare long answer: retry once with room
                r = client.chat.completions.create(**body)
        except Exception as e:
            if "logprobs" not in body or "logprob" not in str(e).lower():
                raise
            with lock:
                if state["logprobs"]:
                    print(f"  {args.model} rejects logprobs ({str(e)[:90]}): switching to a stated 0-100 confidence", flush=True)
                state["logprobs"] = False
            body, mode = body_for(o, False)
            r = client.chat.completions.create(**body)
        ch = r.choices[0]
        lp = None
        if ch.logprobs and ch.logprobs.content:
            first = next((t for t in ch.logprobs.content if t.token.strip()), ch.logprobs.content[0])
            # raw list: "B" and " B" are different tokens of the same letter (summed in score)
            lp = [[t.token, t.logprob] for t in first.top_logprobs]
        rec = {"slot": o["slot"], "sample_id": o["sample_id"], "model": r.model, "mode": mode, "prompt": args.prompt_version, "content": ch.message.content,
               "top_logprobs_raw": lp, "usage": {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}}
        with lock, open(path(args.output), "a") as handle:
            handle.write(json.dumps(rec) + "\n")

    errors = []

    def safe(o):  # one failed request must not stop the run; rerun the same command to retry it
        try:
            call(o)
        except Exception as e:
            with lock:
                errors.append(f"{o['slot']} {o['sample_id']}: {str(e)[:150]}")

    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(safe, todo), 1):
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    if errors:
        print(f"{len(errors)} requests failed (rerun the same command to retry them), e.g. {errors[0]}")
    rows = [json.loads(l) for l in open(path(args.output))]
    cost = sum(r["usage"]["in"] for r in rows) / 1e6 * pin + sum(r["usage"]["out"] for r in rows) / 1e6 * pout
    print(f"measured: {len(rows)} responses, ${cost:.2f}")


# ----------------------------------------------------------------------------- score
def parse(o, r):
    """-> reranker distribution over the k candidates + 'none' (in candidate order), chosen index."""
    k = len(o["candidates"])
    letters = LETTERS[:k + 1]
    order = o["option_order"] + [k]  # letter position -> candidate index (k = none)
    text = (r["content"] or "").strip()
    import re
    m = re.search(r"\b([A-Z])\b", text.upper())
    choice_letter = m.group(1) if m and m.group(1) in letters else None
    dist = np.zeros(k + 1)
    # log-probabilities of the answer token; tokens of the same letter (" B", "B") are summed.
    # Files written before 2026-10-01 kept a dict whose keys were stripped, so " B" overwrote "B":
    # those are not usable and only the chosen letter is kept.
    for tok, lp in (r.get("top_logprobs_raw") or []):
        t = tok.strip().upper().rstrip(".")
        if len(t) == 1 and t in letters:
            dist[order[letters.index(t)]] += math.exp(lp)
    if dist.sum() <= 0:  # no usable log-probabilities: stated confidence or plain choice
        if choice_letter is None:
            return None, None
        conf = 0.8
        nums = [int(x) for x in text.replace("%", " ").split()[1:] if x.isdigit()]
        if nums:
            conf = min(max(nums[0] / 100, 0.01), 0.99)
        dist[:] = (1 - conf) / k
        dist[order[letters.index(choice_letter)]] = conf
    dist = dist / dist.sum()
    # the pick is the letter the model wrote (greedy); the distribution only gives its confidence
    pick = order[letters.index(choice_letter)] if choice_letter else int(np.argmax(dist))
    return dist, pick


def auroc(score, label):
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def score(args):
    from common import ancestor_sets, load_terms
    terms = load_terms(args.ontology_terms)
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p})
    pilot = {(o["slot"], o["sample_id"]): o for o in map(json.loads, open(path(args.pilot)))}
    resp = {(r["slot"], r["sample_id"]): r for r in map(json.loads, open(path(args.responses)))}
    model = next(iter(resp.values()))["model"] if resp else "?"
    print(f"{args.responses}: {len(resp)} responses ({model}); usable log-probabilities in "
          f"{np.mean([bool(r.get('top_logprobs_raw')) for r in resp.values()]):.0%}"
          + ("; old-format log-probabilities ignored (choice only)" if any(r.get("top_logprobs") for r in resp.values()) else ""))
    report = {}
    for slot in SLOTS:
        rows = []
        for key, o in pilot.items():
            if key[0] != slot or key not in resp:
                continue
            dist, pick = parse(o, resp[key])
            if dist is None:
                continue
            k = len(o["candidates"])
            gold_idx = next((i for i, c in enumerate(o["candidates"]) if c["term_id"] == o["gold"]), k)  # k = not in top-k
            rows.append((o, dist, pick, gold_idx))
        if not rows:
            continue
        n = len(rows)
        in_k = np.array([g < len(o["candidates"]) for o, _, _, g in rows])
        llm_right = np.array([p == g for o, _, p, g in rows])  # 'none' is right when gold is not in the top-k
        llm_term_right = np.array([p == g and g < len(o["candidates"]) for o, _, p, g in rows])
        base_right = np.array([g == 0 for o, _, _, g in rows])
        conf = np.array([d[p] for o, d, p, g in rows])
        none = np.array([p == len(o["candidates"]) for o, _, p, g in rows])
        # fusion: p ∝ base^a * llm^b over the top-k (+ none: base mass outside the top-k), (a, b) by grouped CV
        def fused(o, d, a, b):
            base = np.array(o["base_probs"] + [max(1 - o["topk_mass"], 1e-6)])
            z = a * np.log(np.maximum(base, 1e-9)) + b * np.log(np.maximum(d, 1e-6))
            z = np.exp(z - z.max())
            return z / z.sum()
        grid = [(a, b) for a in (0, 0.25, 0.5, 0.75, 1, 1.5) for b in (0, 0.25, 0.5, 0.75, 1, 1.5, 2) if a or b]
        studies = np.array([o["study"] for o, *_ in rows])
        ust = np.unique(studies)
        fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(ust))}
        fold = np.array([fold_of[s] for s in studies])
        f_right = np.zeros(n, bool)
        f_conf = np.zeros(n)
        chosen = []
        for f in range(5):
            tr, te = fold != f, fold == f
            if not te.any():
                continue
            best = max(grid, key=lambda ab: sum(np.log(fused(o, d, *ab)[g] + 1e-9)
                                               for (o, d, _, g), m in zip(rows, tr) if m))
            chosen.append(best)
            for i in np.where(te)[0]:
                o, d, _, g = rows[i]
                q = fused(o, d, *best)
                kk = len(o["candidates"])  # the answer is the best *term*; 'none' only lowers its confidence
                f_right[i] = int(np.argmax(q[:kk])) == g
                f_conf[i] = q[:kk].max()
        def rel(o, i):  # the picked candidate vs gold: exact / ancestor (true, coarser) / descendant / other
            if i >= len(o["candidates"]):
                return "none"
            t = o["candidates"][i]["term_id"]
            return "exact" if t == o["gold"] else "coarser" if t in anc.get(o["gold"], ()) else \
                "too_specific" if o["gold"] in anc.get(t, ()) else "other"
        llm_rel = np.array([rel(o, p) for o, _, p, g in rows])
        base_rel = np.array([rel(o, 0) for o, *_ in rows])
        res = {"n": n, "gold_in_top5": round(float(in_k.mean()), 3),
               "base_top1": round(float(base_right.mean()), 3),
               "base_r (top1 | gold in top5)": round(float(base_right[in_k].mean()), 3),
               "llm_term_correct": round(float(llm_term_right.mean()), 3),
               "llm_r (pick | gold in top5)": round(float(llm_term_right[in_k].mean()), 3),
               "llm_says_none": round(float(none.mean()), 3),
               "none_when_gold_outside_top5": round(float(none[~in_k].mean()), 3) if (~in_k).any() else None,
               "llm_auroc": round(auroc(conf, llm_right), 3),
               "base_auroc": round(auroc([o["base_maxp"] for o, *_ in rows], base_right), 3),
               "base_exact/coarser/too_specific/other": [round(float((base_rel == x).mean()), 3) for x in ("exact", "coarser", "too_specific", "other")],
               "llm_exact/coarser/too_specific/other/none": [round(float((llm_rel == x).mean()), 3) for x in ("exact", "coarser", "too_specific", "other", "none")],
               "fused_top1": round(float(f_right.mean()), 3),
               "fused_auroc": round(auroc(f_conf, f_right), 3),
               "fusion_weights_base_llm": chosen}
        report[slot] = res
        print(f"\n== {slot} (n={n}, gated least-confident samples)")
        for kk, v in res.items():
            print(f"  {kk:32} {v}")
        r_meas, a_meas = res["llm_r (pick | gold in top5)"], res["llm_auroc"]
        print(f"  -> project onto the whole slot: python3 experiments/hierarchical_backoff.py --models prototype "
              f"--fold_seeds 3 --gates {args.gate} --rerank_acc {r_meas} --auroc {a_meas} --label_map <map> --output <json>")
    if args.output:
        json.dump(report, open(path(args.output), "w"), indent=1)


def vote(args):
    """Combine the base model with one or several rerankers: log p = log p_base + sum_m w_m log p_m
    over the top-k candidates, weights chosen for top-1 accuracy by study-grouped CV (small weights
    tame over-confident log-probabilities). Prints each reranker alone and every combination."""
    import itertools
    pilot = {(o["slot"], o["sample_id"]): o for o in map(json.loads, open(path(args.pilot)))}
    names = [os.path.basename(f).replace(".jsonl", "") for f in args.responses]
    resp = {n: {(r["slot"], r["sample_id"]): r for r in map(json.loads, open(path(f)))}
            for n, f in zip(names, args.responses)}
    grid = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1]
    report = {}
    for slot in SLOTS:
        keys = [k for k in pilot if k[0] == slot and all(k in resp[n] for n in names)]
        if not keys:
            continue
        base, gold, studies, D = [], [], [], {n: [] for n in names}
        for k in keys:
            o = pilot[k]
            kk = len(o["candidates"])
            base.append(np.log(np.maximum(np.array(o["base_probs"]), 1e-9)))
            gold.append(next((i for i, c in enumerate(o["candidates"]) if c["term_id"] == o["gold"]), kk))
            studies.append(o["study"])
            for n in names:
                d, _ = parse(o, resp[n][k])
                D[n].append(np.log(np.maximum(d[:kk] if d is not None else np.ones(kk) / kk, 1e-6)))
        base, gold, D = np.array(base), np.array(gold), {n: np.array(v) for n, v in D.items()}
        ust = np.unique(studies)
        fold_of = {s: i % 5 for i, s in enumerate(np.random.default_rng(0).permutation(ust))}
        fold = np.array([fold_of[s] for s in studies])

        def cv(models):
            right = np.zeros(len(gold), bool)
            for f in range(5):
                tr, te = fold != f, fold == f
                acc = lambda w, m: ((base[m] + sum(wi * D[n][m] for wi, n in zip(w, models))).argmax(1) == gold[m]).mean()
                best = max(itertools.product(grid, repeat=len(models)), key=lambda w: acc(w, tr))
                right[te] = (base[te] + sum(wi * D[n][te] for wi, n in zip(best, models))).argmax(1) == gold[te]
            return float(right.mean())
        res = {"n": len(keys), "base_top1": round(float((base.argmax(1) == gold).mean()), 3)}
        for n in names:
            res[f"{n} alone"] = round(float((D[n].argmax(1) == gold).mean()), 3)
        for r in range(1, len(names) + 1):
            for combo in itertools.combinations(names, r):
                res["base + " + " + ".join(combo)] = round(cv(list(combo)), 3)
        report[slot] = res
        print(f"\n== {slot} (n={len(keys)}, top-1 on the gated samples)")
        for kk, v in res.items():
            print(f"  {kk:60} {v}")
    if args.output:
        json.dump(report, open(path(args.output), "w"), indent=1)


def main():
    from _setup import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        b.add_argument(f"--{name}", default=DEFAULTS[name])
    b.add_argument("--term_vectors", default="~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5")
    b.add_argument("--label_map", default=None)
    b.add_argument("--slots", nargs="+", default=SLOTS)
    b.add_argument("--gate", type=float, default=0.3)
    b.add_argument("--n_per_slot", type=int, default=500)
    b.add_argument("--k", type=int, default=5, help="number of base candidates (top-k)")
    b.add_argument("--add_ancestors", action="store_true", help="also offer the candidates' broader terms (slot labels)")
    b.add_argument("--max_candidates", type=int, default=10, help="cap with --add_ancestors")
    b.add_argument("--fold_seed", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--text_chars", type=int, default=1200)
    b.add_argument("--example_chars", type=int, default=300)
    b.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--model", default="gpt-4.1-mini")
    r.add_argument("--confidence", choices=["logprobs", "verbal"], default="logprobs")
    r.add_argument("--prompt_version", choices=["v1", "v2"], default="v2",
                   help="v2 adds the granularity instruction (v1 answers were too specific vs Metalog)")
    r.add_argument("--max_tokens", type=int, default=None, help="default: 16 (logprobs) / 24 (verbal)")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--api_key_path", default="~/MicrobeAtlasProject/my_api_key")
    r.add_argument("--base_url", default=None, help="e.g. a local vLLM server for Qwen3 (OpenAI-compatible)")
    r.add_argument("--price_in", type=float, default=None)
    r.add_argument("--price_out", type=float, default=None)
    r.add_argument("--output", required=True)
    r.add_argument("--dry_run", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--pilot", required=True)
    s.add_argument("--responses", required=True)
    s.add_argument("--ontology_terms", default=DEFAULTS["ontology_terms"])
    s.add_argument("--gate", type=float, default=0.3)
    s.add_argument("--output", default=None)
    v = sub.add_parser("vote")
    v.add_argument("--pilot", required=True)
    v.add_argument("--responses", nargs="+", required=True)
    v.add_argument("--output", default=None)
    args = p.parse_args()
    {"build": build, "run": run, "score": score, "vote": vote}[args.step](args)


if __name__ == "__main__":
    main()
