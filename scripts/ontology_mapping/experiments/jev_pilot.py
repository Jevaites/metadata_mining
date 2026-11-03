#!/usr/bin/env python3
"""
Jev pilot (TypeSafe's System One model) for ontology mapping, run on the same 1,500 gated out-of-fold samples as
the GPT reranker pilot (rerank_pilot.py, claude/rerank-pilot-results.md), so every comparison is paired.
Design after TypeSafe's cookbooks (claude/jev-cookbook-refactor.md); Jev docs: https://docs.typesafe.ai
(Choice = one option with probabilities, <= 255 options; Noul = P(statement true); $0.042 / 1M input tokens).

Two kinds of request (--arm):
  flat     ONE request per sample with three kinds of question over the same state (TypeSafe's parallel-questions
           cookbook: questions are answered in isolation, so bundling does not change the answers):
             vocab_*   a Choice over the whole slot vocabulary (every term Metalog uses in the slot in another study:
                       37-42 biome, ~200 feature, ~180 material): "the whole ontology in the input" in the form that
                       fits (ENVO + UBERON have 18k+ terms: over 64k tokens and 255 options);
             rerank_*  a Choice over the prototype's candidates (top-5 + broader slot labels, exactly those GPT saw,
                       in GPT's order first): which candidate;
             fits_i    one Noul per candidate, "does this term describe the sample?": whether (a term and its broader
                       terms can all be true).
           The skill-suggestion cookbook's split: the Choice says which, the Nouls say whether.
  topdown  beam search down the slot's part of the ontology (the vocabulary and its ancestors), one request per
           level; at each node a Choice over its children plus "X itself" when X is a slot term; single-option nodes
           are passed without asking (TypeSafe's hierarchical-classification cookbook, beam 3). --start lca (the
           common ancestor of the base top-3; the default) or root.

Decisions are code, separate from the requests, and recomputed by `score` from the stored answers (so thresholds
can change without new calls):
  jev_vocab   the vocabulary Choice's top term (back-off on its distribution);
  jev_rerank  the rerank Choice's top candidate, or none when no candidate's Noul reaches NOUL_NO (0.3);
  jev_fits    the Nouls alone: the most specific candidate >= NOUL_YES (0.7), else the highest if > NOUL_NO, else none;
  topdown_*   the best finished path; back-off on the distribution over finished paths.
All the wording Jev sees is in jev_questions.py.

Every request goes through a request-keyed cache (<output folder>/jev_cache.jsonl: key, payload, answers), as in the
cookbooks: a rerun replays instead of paying, a crashed run resumes for free, the file is the log of exact payloads,
and topdown lca / root share their node calls.

Steps:
  prepare      (no API) -> inputs.json: the term-table subset and, per slot, how many samples of each study use each
               label (training set + label map). A sample's vocabulary = labels used by >= --min_studies other studies.
  check_order  (API) first-option bias on the rerank Choice (documented for jev-1.13; we use 1.14): pilot order vs
               reversed, in one request, for --limit samples per slot. Rerun `run` with --orders 2 only if it is real.
  run          (API; on your Mac) one arm -> one JSONL line per sample. --fake: keyword-overlap stand-in (no network);
               with --dry_run nothing is written and the token count and cost are printed.
  score        (no API) one table per slot: base, the Jev decisions above, GPT files (--reference name=file), paired on
               the samples all answered.
The flat lines also carry `content` / `top_logprobs_raw` for the jev_rerank decision, so rerank_pilot.py score / vote
read them like GPT files.

cd scripts/ontology_mapping
X=~/MicrobeAtlasProject/ontology_mapping/experiments/rerank; J=~/MicrobeAtlasProject/ontology_mapping/experiments/jev
python3 experiments/jev_pilot.py prepare --pilot $X/pilot_k5anc.jsonl \
    --label_map ~/MicrobeAtlasProject/metalog/clean/previous_2026-10-01/biome_label_map.tsv --output $J/inputs.json
python3 experiments/jev_pilot.py check_order --pilot $X/pilot_k5anc.jsonl --inputs $J/inputs.json --cache $J/jev_cache.jsonl --limit 30
python3 experiments/jev_pilot.py run --arm flat --pilot $X/pilot_k5anc.jsonl --inputs $J/inputs.json --output $J/flat.jsonl
python3 experiments/jev_pilot.py run --arm topdown --start lca --pilot $X/pilot_k5anc.jsonl --inputs $J/inputs.json --output $J/topdown_lca.jsonl
python3 experiments/jev_pilot.py run --arm topdown --start root --pilot $X/pilot_k5anc.jsonl --inputs $J/inputs.json --output $J/topdown_root.jsonl
python3 experiments/jev_pilot.py score --pilot $X/pilot_k5anc.jsonl --inputs $J/inputs.json \
    --responses $J/flat.jsonl $J/topdown_*.jsonl --reference gpt-4.1-mini=$X/resp_k5anc_gpt-4.1-mini.jsonl --output $J/score.json
(add --fake --dry_run to a run for the cost estimate). The API key is read from --api_key_path (default
~/MicrobeAtlasProject/jev_api_key), or $TYPESAFE_API_KEY if that file is missing.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, ancestor_sets, path  # noqa: E402
from hierarchy import NO_ANSWER, Closure  # noqa: E402
import jev_questions as W  # noqa: E402  (all the wording Jev sees)
from rerank_prompt import LETTERS, count_tokens, parse  # noqa: E402

PRICE_IN = 0.042  # $ per 1M input tokens (jev-1.13 / 1.14); output tokens are free
MAX_OPTIONS = 255  # per Choice question
ROOT = "ROOT"  # virtual root of the top-down tree
MAX_LEVELS = 30


# ----------------------------------------------------------------------------- small helpers
def read_jsonl(p):
    with open(path(p)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(*parts):
    return int(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def auroc(score, label):
    """Probability that a right answer has a higher confidence than a wrong one (ties count half)."""
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - label.sum() * (label.sum() + 1) / 2) / (label.sum() * (~label).sum()))


def relation(t, gold, anc):
    """'exact', 'coarser' (an ancestor of gold), 'too_specific' (a descendant), 'other', or 'none' (t None)."""
    if t is None:
        return "none"
    if t == gold:
        return "exact"
    if t in anc.get(gold, ()):
        return "coarser"
    if gold in anc.get(t, ()):
        return "too_specific"
    return "other"


class Ctx:
    """Everything a request builder needs: terms, ancestors, slot usage, cached vocabularies and trees."""

    def __init__(self, inputs, args):
        self.terms = inputs["terms"]
        self.slots = inputs["slots"]
        self.args = args
        self.anc = ancestor_sets({t: set(v["parents"]) for t, v in self.terms.items()})
        self._vocab, self._tree = {}, {}

    def label(self, t):
        return self.terms[t]["label"] if t in self.terms else t

    def vocab(self, slot, study):
        """Slot labels used by >= min_studies studies other than `study`, most used first (ties by id).
        -> (labels, {label: samples using it outside `study`})."""
        key = (slot, study)
        if key not in self._vocab:
            use = {t: {s: n for s, n in st.items() if s != study} for t, st in self.slots[slot].items()}
            freq = {t: sum(st.values()) for t, st in use.items() if len(st) >= self.args.min_studies}
            labels = sorted(freq, key=lambda t: (-freq[t], t))
            self._vocab[key] = (labels, freq)
        return self._vocab[key]

    def tree(self, slot, study):
        """The slot's part of the ontology for this vocabulary. -> {children, below (answer labels at or
        below a node, most used first), answers, freq}; children[ROOT] = the nodes without a parent inside."""
        key = (slot, study)
        if key not in self._tree:
            labels, freq = self.vocab(slot, study)
            answers = [v for v in labels if v not in NO_ANSWER]
            nodes = set(labels).union(*[self.anc.get(v, set()) for v in labels]) & set(self.terms)
            children = {n: [] for n in nodes}
            children[ROOT] = []
            for n in sorted(nodes):
                parents = [p for p in self.terms[n]["parents"] if p in nodes]
                for p in parents:
                    children[p].append(n)
                if not parents:
                    children[ROOT].append(n)
            below = {n: [] for n in nodes}
            for v in answers:  # answers are most used first, so every list is too
                for a in ({v} | self.anc.get(v, set())) & nodes:
                    below[a].append(v)
            below[ROOT] = list(answers)
            # drop branches without any answer below (only possible via NO_ANSWER roots that are leaves)
            children = {n: [c for c in cs if below[c]] for n, cs in children.items()}
            self._tree[key] = {"children": children, "below": below, "answers": set(answers), "freq": freq}
        return self._tree[key]


def option_names(ids, ctx):
    """Unique option names (Jev keys its probabilities by them): the label, + ' (id)' when two share a label."""
    labels = [ctx.label(t) for t in ids]
    dup = {l for l in labels if labels.count(l) > 1}
    return [f"{l} ({t})" if l in dup else l for l, t in zip(labels, ids)]


def describe(t, ctx, example=None, narrower=None):
    return W.describe(ctx.terms.get(t, {}), example=example, narrower=narrower)


def orders(n, k, seed_parts, first=None):
    """k option orders over n options: `first` (if given) then seeded permutations."""
    rng = np.random.default_rng(stable_seed(*seed_parts))
    out = [list(first)] if first is not None else []
    while len(out) < k:
        out.append(rng.permutation(n).tolist())
    return out[:max(k, 1)]


def choice_question(instr, names, descs, order):
    return {"type": "choice", "instructions": instr, "criteria": {names[i]: descs[i] for i in order}}


def mean_probs(answers, keys, names):
    """Average of the Choice probabilities over several questions (option orders), aligned with `names`."""
    p = np.zeros(len(names))
    for k in keys:
        probs = answers[k].probabilities
        p += np.array([float(probs.get(n, 0.0)) for n in names])
    p = p / max(len(keys), 1)
    return p / p.sum() if p.sum() > 0 else np.ones(len(names)) / len(names)


def usage_in(r):
    u = getattr(r, "usage", None)
    return int(getattr(u, "input_tokens", 0) or 0)


# ----------------------------------------------------------------------------- the stand-in for Jev
class FakeJev:
    """Keyword-overlap stand-in with Jev's interface (async system_one -> .model, .answers, .usage): a Choice
    option's probability is softmax(3 x the number of its words found in the state); a Noul is a logistic of
    the same overlap. Used for dry runs (token count, request walk) and tests. Counts the request size in
    tokens of the JSON (state counted once; also as if counted once per question)."""
    model = "fake-jev"

    def __init__(self):
        self.tokens, self.tokens_per_question, self.requests, self.questions = 0, 0, 0, 0

    @staticmethod
    def _words(x):
        return set(re.findall(r"[a-z]{4,}", json.dumps(x).lower()))

    async def system_one(self, state, questions, **kw):
        sw = self._words(state)
        n_state = count_tokens(json.dumps(state))
        n_q = {k: count_tokens(json.dumps(q)) for k, q in questions.items()}
        self.tokens += n_state + sum(n_q.values())
        self.tokens_per_question += sum(n_state + n for n in n_q.values())
        self.requests += 1
        self.questions += len(questions)
        answers = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                names = list(q["criteria"])
                s = np.array([3.0 * len(sw & self._words([n, q["criteria"][n]])) for n in names])
                p = np.exp(s - s.max())
                p = p / p.sum()
                answers[k] = SimpleNamespace(type="choice", choice=names[int(p.argmax())],
                                             probabilities=dict(zip(names, p.tolist())), confidence=float(p.max()))
            else:
                ov = len(sw & self._words(q["instructions"]))
                answers[k] = SimpleNamespace(type="noul", noul=float(1 / (1 + math.exp(-(ov - 3)))))
        await asyncio.sleep(0)
        return SimpleNamespace(model=self.model, answers=answers,
                               usage=SimpleNamespace(input_tokens=n_state + sum(n_q.values()), output_tokens=0))


def real_client(args):
    try:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    except ImportError:
        raise SystemExit(f"typesafe-sdk is not installed for this Python ({sys.executable}): "
                         f"run `{sys.executable} -m pip install typesafe-sdk==0.7.5`")
    if os.path.exists(path(args.api_key_path)):
        key = open(path(args.api_key_path)).read().strip()
    elif os.environ.get("TYPESAFE_API_KEY"):
        key = os.environ["TYPESAFE_API_KEY"]
    else:
        raise SystemExit(f"no Jev API key: put it in {args.api_key_path} (or set TYPESAFE_API_KEY)")
    return AsyncTypeSafeClient(api_key=key, model=args.model, timeout=120,
                               retry=RetryPolicy(max_retries=8, backoff_initial=1.0, backoff_max=30.0, timeout=600))



# ----------------------------------------------------------------------------- the request-keyed cache
def answer_to_json(a):
    if getattr(a, "noul", None) is not None:
        return {"type": "noul", "noul": float(a.noul)}
    return {"type": "choice", "probabilities": {k: float(v) for k, v in a.probabilities.items()}}


def answer_from_json(d):
    if d["type"] == "noul":
        return SimpleNamespace(type="noul", noul=d["noul"])
    return SimpleNamespace(type="choice", probabilities=d["probabilities"])


class CachedClient:
    """Request-keyed cache in front of Jev (or of FakeJev), as in TypeSafe's cookbooks. The key is a hash of (model,
    state, questions) in the order they are sent (option order is part of the question). Each new request is appended
    to a JSONL file with its payload and answers, so reruns replay, a crashed run resumes for free and the file is the
    log of exact payloads."""

    def __init__(self, client, cache_path, model):
        self.client, self.path, self.model = client, cache_path, model
        self.store, self.hits, self.misses, self.new_tokens = {}, 0, 0, 0
        if os.path.exists(cache_path):
            with open(cache_path) as handle:
                for line in handle:
                    if line.strip():
                        rec = json.loads(line)
                        self.store[rec["key"]] = rec["response"]

    def key(self, state, questions):
        return hashlib.sha256(json.dumps([self.model, state, questions]).encode()).hexdigest()

    async def system_one(self, state, questions, **kw):
        k = self.key(state, questions)
        if k not in self.store:
            r = await self.client.system_one(state, questions, **kw)
            resp = {"model": r.model, "answers": {q: answer_to_json(a) for q, a in r.answers.items()},
                    "usage": {"input_tokens": usage_in(r)}}
            self.store[k] = resp
            self.misses += 1
            self.new_tokens += usage_in(r)
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as handle:
                handle.write(json.dumps({"key": k, "model": self.model,
                                         "request": {"state": state, "questions": questions}, "response": resp}) + "\n")
        else:
            self.hits += 1
        resp = self.store[k]
        return SimpleNamespace(model=resp["model"], answers={q: answer_from_json(a) for q, a in resp["answers"].items()},
                               usage=SimpleNamespace(input_tokens=resp["usage"]["input_tokens"], output_tokens=0))


# ----------------------------------------------------------------------------- decisions (pure functions)
def decide_rerank(probs, fits):
    """The rerank Choice says which candidate, the Nouls say whether any fits (skill-suggestion cookbook): none when
    no candidate's Noul reaches NOUL_NO. -> (pick: candidate index, k = none; distribution over candidates + none,
    with P(some candidate fits) taken as the highest Noul)."""
    probs, fits = np.asarray(probs, float), np.asarray(fits, float)
    some = float(fits.max()) if len(fits) else 1.0  # no Nouls asked: no "none" decision
    pick = len(probs) if some < W.NOUL_NO else int(np.argmax(probs))
    return pick, np.append(some * probs, 1 - some)


def decide_fits(fits, ids, anc):
    """The Nouls alone: the most specific candidate judged true (>= NOUL_YES; among incomparable ones the highest
    Noul); if none is a settled yes, the highest Noul when it is above NOUL_NO; else none.
    -> (pick, k = none; decision 'yes' / 'unsettled' / 'none'; distribution: Nouls + prod(1 - Noul), normalised)."""
    n = np.asarray(fits, float)
    yes = [i for i in range(len(ids)) if n[i] >= W.NOUL_YES]
    if yes:  # drop a yes-term when a more specific yes-term lies below it
        specific = [i for i in yes if not any(ids[i] in anc.get(ids[j], ()) for j in yes if j != i)]
        pick, decision = max(specific, key=lambda i: n[i]), "yes"
    elif len(n) and n.max() > W.NOUL_NO:
        pick, decision = int(np.argmax(n)), "unsettled"
    else:
        pick, decision = len(ids), "none"
    dist = np.append(n, float(np.prod(1 - n)))
    return pick, decision, (dist / dist.sum() if dist.sum() > 0 else np.ones(len(dist)) / len(dist))


def letters_record(o, pick, dist):
    """The rerank_prompt.parse format (GPT files): the picked letter + a log-probability per letter."""
    k = len(o["candidates"])
    letter_of = {cand: LETTERS[pos] for pos, cand in enumerate(o["option_order"])}
    letter_of[k] = LETTERS[k]
    return {"content": letter_of[pick],
            "top_logprobs_raw": [[letter_of[i], float(math.log(max(dist[i], 1e-12)))] for i in range(k + 1)]}


# ----------------------------------------------------------------------------- flat request
def flat_questions(o, ctx):
    """-> (questions, (vocabulary ids, their names, candidate ids, their names)) for one sample."""
    slot, seed = o["slot"], (ctx.args.seed, o["slot"], o["sample_id"])
    vids = ctx.vocab(slot, o["study"])[0][:MAX_OPTIONS]
    vnames = option_names(vids, ctx)
    vdesc = [describe(t, ctx) for t in vids]
    cids = [c["term_id"] for c in o["candidates"]]
    cnames = option_names(cids, ctx)
    cdesc = [describe(c["term_id"], ctx, example=c.get("example")) for c in o["candidates"]]
    ask = set(getattr(ctx.args, "questions", None) or ("vocab", "rerank", "fits"))
    q = {}
    if "vocab" in ask:
        for j, order in enumerate(orders(len(vids), ctx.args.orders, seed + ("vocab",))):  # random order: no frequency prior
            q[f"vocab_{j}"] = choice_question(W.which_term(slot), vnames, vdesc, order)
    if "rerank" in ask:
        for j, order in enumerate(orders(len(cids), ctx.args.orders, seed + ("rerank",), first=o["option_order"])):
            q[f"rerank_{j}"] = choice_question(W.which_term(slot), cnames, cdesc, order)
    if "fits" in ask:
        for i, (name, d) in enumerate(zip(cnames, cdesc)):
            q[f"fits_{i}"] = W.term_fits(slot, {"name": name, **(d or {})})
    return q, (vids, vnames, cids, cnames)


async def ask_flat(client, o, ctx):
    q, (vids, vnames, cids, cnames) = flat_questions(o, ctx)
    r = await client.system_one(W.state(o["text"], o.get("similar"), o["slot"]), q)
    rec = {"slot": o["slot"], "sample_id": o["sample_id"], "arm": "flat", "model": r.model,
           "questions": len(q), "usage": {"in": usage_in(r), "out": 0}}
    if any(k.startswith("vocab_") for k in q):
        vocab = mean_probs(r.answers, [k for k in q if k.startswith("vocab_")], vnames)
        rec["vocab"] = {t: round(float(p), 6) for t, p in zip(vids, vocab) if p >= 1e-6}
        rec["n_vocab"] = len(vids)
    if any(k.startswith("rerank_") for k in q):
        rerank = mean_probs(r.answers, [k for k in q if k.startswith("rerank_")], cnames)
        fits = [float(r.answers[f"fits_{i}"].noul) for i in range(len(cids)) if f"fits_{i}" in q]
        rec.update(rerank=[round(float(p), 6) for p in rerank], fits=[round(f, 6) for f in fits])
        if len(cids) < len(LETTERS):  # rerank_pilot.py score / vote read the jev_rerank decision (<= 22 candidates)
            pick, dist = decide_rerank(rerank, fits)
            rec.update(letters_record(o, pick, dist))
    return rec


# ----------------------------------------------------------------------------- top-down
def lca_start(o, tree, ctx, n=3):
    """The most specific node that is (an ancestor of) each of the base top-n candidates, or ROOT."""
    nodes = set(tree["below"]) - {ROOT}
    common = None
    for c in o["candidates"][:n]:
        up = ({c["term_id"]} | ctx.anc.get(c["term_id"], set())) & nodes
        common = up if common is None else common & up
    if not common:
        return ROOT
    return min(common, key=lambda x: (len(tree["below"][x]), -len(ctx.anc.get(x, ())), x))


def node_options(node, tree, ctx, n_narrower=6):
    """-> [(name, child or None for 'node itself', description)]: the children (most used first, each described with
    a few slot terms below it), then 'node itself' if the node is a slot term."""
    kids = sorted(tree["children"].get(node, []), key=lambda c: (-sum(tree["freq"].get(v, 0) for v in tree["below"][c]), c))
    out = [(name, c, describe(c, ctx, narrower=[ctx.label(v) for v in tree["below"][c] if v != c][:n_narrower]))
           for c, name in zip(kids, option_names(kids, ctx))]
    if node != ROOT and node in tree["answers"] and kids:
        name, desc = W.itself_option(ctx.label(node))
        out.append((name, None, desc))
    return out


def advance(path, tree, ctx):
    """Follow nodes with a single option without asking. -> (path, finished: a leaf was reached)."""
    while True:
        opts = node_options(path[-1], tree, ctx)
        if not opts:
            return path, True
        if len(opts) > 1:
            return path, False
        path = path + (opts[0][1],)


async def ask_topdown(client, o, ctx):
    """Beam search: each beam entry is (probability, path, finished); a path's probability is the product of its
    edge probabilities. One request per level, one Choice per distinct open node. A node reached by two routes
    (several parents) is two entries; their probabilities are summed at the end."""
    tree = ctx.tree(o["slot"], o["study"])
    start = lca_start(o, tree, ctx) if ctx.args.start == "lca" else ROOT
    beam = [(1.0, *advance((start,), tree, ctx))]
    trace, n_q, tokens, model = [], 0, 0, ctx.args.model
    for level in range(MAX_LEVELS):
        nodes = sorted({p[-1] for _, p, done in beam if not done})
        if not nodes:
            break
        opts = {n: node_options(n, tree, ctx) for n in nodes}
        q = {}
        for i, n in enumerate(nodes):
            for j, order in enumerate(orders(len(opts[n]), ctx.args.orders, (ctx.args.seed, o["slot"], o["sample_id"], n))):
                q[f"n{i}_{j}"] = choice_question(W.which_branch(o["slot"], None if n == ROOT else ctx.label(n)),
                                                 [x[0] for x in opts[n]], [x[2] for x in opts[n]], order)
        r = await client.system_one(W.state(o["text"]), q)
        n_q, tokens, model = n_q + len(q), tokens + usage_in(r), r.model
        probs = {n: mean_probs(r.answers, [k for k in q if k.startswith(f"n{i}_")], [x[0] for x in opts[n]])
                 for i, n in enumerate(nodes)}
        trace.append({n: {x[0]: round(float(p), 5) for x, p in zip(opts[n], probs[n])} for n in nodes})
        new = [b for b in beam if b[2]]
        for p, route, done in beam:
            if not done:
                for (_, child, _), pc in zip(opts[route[-1]], probs[route[-1]]):
                    new.append((p * pc, route, True) if child is None else (p * pc, *advance(route + (child,), tree, ctx)))
        beam = sorted(new, key=lambda b: -b[0])[:ctx.args.beam]
    terminals = {}
    for p, route, done in beam:
        if done and route[-1] in tree["answers"]:
            terminals[route[-1]] = terminals.get(route[-1], 0.0) + p
    best = max((b for b in beam if b[2]), key=lambda b: b[0], default=(0.0, (start,), False))
    return {"slot": o["slot"], "sample_id": o["sample_id"], "arm": f"topdown_{ctx.args.start}", "model": model,
            "start": start, "gold_reachable": o["gold"] in tree["below"].get(start, []),
            "terminals": {t: round(p, 6) for t, p in terminals.items()}, "best_path": list(best[1]), "trace": trace,
            "questions": n_q, "requests": len(trace), "usage": {"in": tokens, "out": 0}}


# ----------------------------------------------------------------------------- prepare
def prepare(args):
    from common import load_npz, load_terms, read_tsv, select_samples
    pilot = read_jsonl(args.pilot)
    terms = load_terms(args.ontology_terms)
    rows = {t: (l, s, d, p) for t, l, s, d, p in zip(terms["term_id"], terms["label"], terms["synonyms"],
                                                      terms["definition"], terms["parents"])}
    require = [set(load_npz(f)[0]) for f in args.require_embeddings]
    S = select_samples(args.samples, require)
    if args.label_map:
        lm = read_tsv(args.label_map)
        for slot in SLOTS:
            m = {r.from_id.strip(): r.to_id.strip() for r in lm.itertuples() if r.slot.strip() in (slot, "*")}
            S[slot] = S[slot].map(lambda t: m.get(t, t))
    slots = {}
    for slot in SLOTS:
        use = {}
        for t, st in zip(S[slot], S["study_code"]):
            if t:
                use.setdefault(t, {})
                use[t][st] = use[t].get(st, 0) + 1
        slots[slot] = use
    needed = set().union(*[set(u) for u in slots.values()]) | {o["gold"] for o in pilot} | \
        {c["term_id"] for o in pilot for c in o["candidates"]}
    parents = {t: {p for p in v[3].split("||") if p} for t, v in rows.items()}
    anc = ancestor_sets(parents)
    needed |= set().union(*[anc.get(t, set()) for t in list(needed)])
    out = {"terms": {t: {"label": rows[t][0], "synonyms": [s for s in rows[t][1].split("||") if s][:3],
                         "definition": rows[t][2][:220], "parents": sorted(parents[t])}
                     for t in sorted(needed) if t in rows},
           "slots": slots,
           "source": {"pilot": args.pilot, "samples": args.samples, "label_map": args.label_map, "n_samples": len(S)}}
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    with open(path(args.output), "w") as handle:
        json.dump(out, handle)
    ctx = Ctx(out, SimpleNamespace(min_studies=args.min_studies))
    print(f"{len(S)} training samples, {len(out['terms'])} terms -> {args.output}")
    for slot in SLOTS:
        ps = [o for o in pilot if o["slot"] == slot]
        if not ps:
            continue
        sizes = [len(ctx.vocab(slot, o["study"])[0]) for o in ps]
        in_vocab = np.mean([o["gold"] in ctx.vocab(slot, o["study"])[0] for o in ps])
        print(f"  {slot}: {len(slots[slot])} labels; pilot vocabulary {min(sizes)}-{max(sizes)} terms "
              f"(> {MAX_OPTIONS}: {np.mean(np.array(sizes) > MAX_OPTIONS):.0%}); gold in the vocabulary {in_vocab:.1%}; "
              f"gold among the pilot candidates {np.mean([o['gold_in_topk'] for o in ps]):.1%}")


# ----------------------------------------------------------------------------- run / check_order
def load_pilot(args):
    pilot = [o for o in read_jsonl(args.pilot) if o["slot"] in args.slots]
    if args.limit:
        pilot = [o for slot in args.slots for o in [x for x in pilot if x["slot"] == slot][:args.limit]]
    return pilot


def make_client(args):
    """-> (the client to call, the underlying client to open / close or None, the cache or None)."""
    raw = FakeJev() if args.fake else real_client(args)
    if args.dry_run:
        return raw, (None if args.fake else raw), None
    cache = CachedClient(raw, path(args.cache or os.path.join(os.path.dirname(args.output), "jev_cache.jsonl")),
                         FakeJev.model if args.fake else args.model)
    return cache, (None if args.fake else raw), cache


def gather(items, fn, opener, workers):
    """Run the async fn over items with at most `workers` in flight. -> (results in item order, None for a failure;
    errors). A failed call is an error, never an answer."""
    errors = []

    async def main():
        sem = asyncio.Semaphore(workers)

        async def one(x):
            async with sem:
                try:
                    return await fn(x)
                except Exception as e:  # rerun the command to retry: the cache keeps everything that succeeded
                    errors.append(f"{x.get('slot')} {x.get('sample_id')}: {type(e).__name__}: {str(e)[:200]}")
                    return None
        if opener is not None:
            async with opener:
                return await asyncio.gather(*(one(x) for x in items))
        return await asyncio.gather(*(one(x) for x in items))
    return asyncio.run(main()), errors


def run(args):
    pilot = load_pilot(args)
    ctx = Ctx(json.load(open(path(args.inputs))), args)
    client, opener, cache = make_client(args)
    ask = ask_flat if args.arm == "flat" else ask_topdown
    t0 = time.time()
    results, errors = gather(pilot, lambda o: ask(client, o, ctx), opener, args.workers)
    records = [r for r in results if r]
    n_in = sum(r["usage"]["in"] for r in records)
    nq = sum(r["questions"] for r in records)
    name = args.arm if args.arm == "flat" else f"topdown_{args.start}"
    print(f"{name}: {len(records)} / {len(pilot)} samples in {time.time() - t0:.0f}s, {nq:,} questions "
          f"({nq / max(len(records), 1):.1f} per sample), {n_in:,} input tokens = ${n_in / 1e6 * PRICE_IN:.2f}")
    if args.fake and args.dry_run:
        print(f"  o200k estimate with fake answers (top-down paths and their cost will differ with Jev); if the state "
              f"is billed once per question: ${client.tokens_per_question / 1e6 * PRICE_IN:.2f}")
    if cache:
        print(f"  cache {cache.path}: {cache.hits} replayed, {cache.misses} new requests "
              f"({cache.new_tokens:,} tokens = ${cache.new_tokens / 1e6 * PRICE_IN:.2f} spent now)")
    if not args.dry_run:
        os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
        with open(path(args.output), "w") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")
        print(f"  wrote {args.output}")
    if errors:
        print(f"{len(errors)} samples failed (rerun the same command: answered requests come from the cache), "
              f"e.g. {errors[0]}")


def check_order(args):
    """First-option bias: the rerank Choice in the pilot order and reversed, two questions in one request.
    Prints, per slot: top-1 agreement, mean total-variation distance between the two distributions, and how often
    the pick is the first-listed option in each order (vs 1 / number of options by chance)."""
    pilot = load_pilot(args)
    ctx = Ctx(json.load(open(path(args.inputs))), args)
    args.output = args.output or os.path.join(os.path.dirname(path(args.cache)), "check_order.json")
    client, opener, cache = make_client(args)

    async def ask(o):
        cids = [c["term_id"] for c in o["candidates"]]
        names = option_names(cids, ctx)
        desc = [describe(c["term_id"], ctx, example=c.get("example")) for c in o["candidates"]]
        fwd, rev = list(o["option_order"]), list(o["option_order"])[::-1]
        q = {"rerank_0": choice_question(W.which_term(o["slot"]), names, desc, fwd),
             "reversed": choice_question(W.which_term(o["slot"]), names, desc, rev)}
        r = await client.system_one(W.state(o["text"]), q)
        a, b = mean_probs(r.answers, ["rerank_0"], names), mean_probs(r.answers, ["reversed"], names)
        return {"slot": o["slot"], "agree": int(a.argmax() == b.argmax()), "tv": 0.5 * float(np.abs(a - b).sum()),
                "first_fwd": int(a.argmax() == fwd[0]), "first_rev": int(b.argmax() == rev[0]), "chance": 1 / len(cids)}
    results, errors = gather(pilot, ask, opener, args.workers)
    report = {}
    for slot in SLOTS:
        rows = [x for x in results if x and x["slot"] == slot]
        if rows:
            report[slot] = {k: round(float(np.mean([x[k] for x in rows])), 3) for k in ("agree", "tv", "first_fwd", "first_rev", "chance")}
            report[slot]["n"] = len(rows)
            print(f"{slot}: n {len(rows)}, top-1 agreement {report[slot]['agree']}, mean TV distance {report[slot]['tv']}, "
                  f"pick = first-listed option {report[slot]['first_fwd']} / {report[slot]['first_rev']} (chance {report[slot]['chance']})")
    if cache:
        print(f"cache: {cache.hits} replayed, {cache.misses} new (${cache.new_tokens / 1e6 * PRICE_IN:.3f})")
    if errors:
        print(f"{len(errors)} failed, e.g. {errors[0]}")
    if not args.dry_run:
        json.dump(report, open(path(args.output), "w"), indent=1)


# ----------------------------------------------------------------------------- score
def backoff_points(rows, ctx, slot, targets=(0.8, 0.6)):
    """Climbing on each sample's P (hierarchy.Closure over its vocabulary): for each coverage target, the
    largest tau answering >= target of the samples -> {target: (coverage, strict, exact, too_specific)}."""
    from hierarchy import TAUS
    out_codes = []  # per sample: outcome per tau
    for o, P, study in rows:
        labels, _ = ctx.vocab(slot, study)
        key = ("closure", slot, study)
        if key not in ctx._tree:
            ctx._tree[key] = Closure(labels, ctx.anc)
        C = ctx._tree[key]
        p = np.array([P.get(v, 0.0) for v in C.vocab])
        if p.sum() <= 0:
            out_codes.append(["abstain"] * len(TAUS))
            continue
        q = p @ C.A
        on_path = C.A[int(p.argmax())] > 0
        codes = []
        for tau in TAUS:
            ok = (q >= tau - 1e-12) & C.informative & on_path
            if not ok.any():
                codes.append("abstain")
                continue
            node = C.nodes[int(np.argmax(np.where(ok, C.rank_key + q, -np.inf)))]
            codes.append(relation(node, o["gold"], ctx.anc))
        out_codes.append(codes)
    codes = np.array(out_codes)
    res = {}
    for target in targets:
        best = None
        for j in range(len(TAUS) - 1, -1, -1):
            ans = codes[:, j] != "abstain"
            if ans.mean() >= target:
                best = j
                break
        if best is None:
            res[target] = None
            continue
        c = codes[:, best]
        ans = c != "abstain"
        res[target] = (round(float(ans.mean()), 3), round(float(np.isin(c[ans], ["exact", "coarser"]).mean()), 3),
                       round(float((c == "exact").mean()), 3), round(float((c == "too_specific").mean()), 3))
    return res


def boot_diff(a, b, studies, reps=2000, seed=0):
    """Mean(a - b) with a 95 % interval from resampling studies."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    st = np.asarray(studies)
    u = np.unique(st)
    idx = {s: np.where(st == s)[0] for s in u}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        pick = np.concatenate([idx[s] for s in rng.choice(u, len(u))])
        d.append((a[pick] - b[pick]).mean())
    return round(float((a - b).mean()), 3), round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)


def arm_views(rec, o, ctx):
    """-> {decision name: (P {term: prob} over answers, may sum to < 1; picked term or None; confidence)} for one
    response line, recomputing every decision from the stored answers."""
    cands = [c["term_id"] for c in o["candidates"]]
    out = {}
    if rec.get("arm") == "flat":
        if "vocab" in rec:
            P = dict(rec["vocab"])
            pick = max(P, key=P.get) if P else None
            out["jev_vocab"] = (P, pick, P.get(pick, 0.0))
        if "rerank" in rec:
            k, dist = decide_rerank(rec["rerank"], rec["fits"])
            out["jev_rerank"] = (dict(zip(cands, dist[:-1])), None if k == len(cands) else cands[k], float(dist[k]))
        if rec.get("fits"):
            k, _, dist = decide_fits(rec["fits"], cands, ctx.anc)
            out["jev_fits"] = (dict(zip(cands, dist[:-1])), None if k == len(cands) else cands[k], float(dist[k]))
    elif "terminals" in rec:
        tot = sum(rec["terminals"].values())
        P = {t: p / tot for t, p in rec["terminals"].items()} if tot > 0 else {}
        pick = max(P, key=P.get) if P else None
        out[rec["arm"]] = (P, pick, P.get(pick, 0.0))
    return out


def score(args):
    pilot = {(o["slot"], o["sample_id"]): o for o in read_jsonl(args.pilot)}
    ctx = Ctx(json.load(open(path(args.inputs))), SimpleNamespace(min_studies=args.min_studies))
    views = {"base": {k: (dict(zip([c["term_id"] for c in o["candidates"]], o["base_probs"])), o["candidates"][0]["term_id"],
                          float(o["base_maxp"])) for k, o in pilot.items()}}
    usage, walk = {}, {}
    for f in args.responses:
        for rec in read_jsonl(f):
            key = (rec["slot"], rec["sample_id"])
            if key not in pilot:
                continue
            if "terminals" in rec:
                walk.setdefault(rec["arm"], {})[key] = (rec["gold_reachable"], rec["questions"], rec["requests"])
            for name, v in arm_views(rec, pilot[key], ctx).items():
                views.setdefault(name, {})[key] = v
                u = usage.setdefault("flat request" if rec.get("arm") == "flat" else name, [0, set()])
                if key not in u[1]:
                    u[0] += rec["usage"]["in"]
                    u[1].add(key)
    for ref in args.reference:
        name, f = ref.split("=", 1)
        for rec in read_jsonl(f):
            key = (rec["slot"], rec["sample_id"])
            if key in pilot:
                dist, pick = parse(pilot[key], rec)
                if dist is not None:
                    cands = [c["term_id"] for c in pilot[key]["candidates"]]
                    views.setdefault(name, {})[key] = (dict(zip(cands, dist[:-1])), None if pick == len(cands) else cands[pick],
                                                       float(dist[pick]))
    report = {"cost": {n: f"{t:,} tokens = ${t / 1e6 * PRICE_IN:.3f} for {len(s)} samples" for n, (t, s) in usage.items()}}
    for slot in SLOTS:
        keys = [k for k in pilot if k[0] == slot and all(k in v for v in views.values())]
        if not keys:
            continue
        studies = [pilot[k]["study"] for k in keys]
        offered = np.array([pilot[k]["gold_in_topk"] for k in keys])
        base_exact = np.array([views["base"][k][1] == pilot[k]["gold"] for k in keys])
        res = {}
        for name, v in views.items():
            rel = np.array([relation(v[k][1], pilot[k]["gold"], ctx.anc) for k in keys])
            exact = rel == "exact"
            r = {"n": len(keys), "exact": round(float(exact.mean()), 3),
                 "hier (exact|coarser)": round(float(np.isin(rel, ["exact", "coarser"]).mean()), 3),
                 "lenient (exact|too specific)": round(float(np.isin(rel, ["exact", "too_specific"]).mean()), 3),
                 "too_specific/other/none": [round(float((rel == x).mean()), 3) for x in ("too_specific", "other", "none")],
                 "r (exact | gold offered)": round(float(exact[offered].mean()), 3) if offered.any() else None,
                 "auroc": round(auroc([v[k][2] for k in keys], exact), 3),
                 "backoff ~80% / ~60% answered (coverage, strict, exact, too specific)":
                     list(backoff_points([(pilot[k], v[k][0], pilot[k]["study"]) for k in keys], ctx, slot).values()),
                 "delta exact vs base (95% CI)": None if name == "base" else boot_diff(exact, base_exact, studies)}
            if name == "jev_vocab":  # does the whole-vocabulary Choice find gold where the prototype's candidates miss it?
                top = [sorted(v[k][0], key=lambda t: -v[k][0][t]) for k in keys]
                gold = [pilot[k]["gold"] for k in keys]
                cand = [{c["term_id"] for c in pilot[k]["candidates"]} for k in keys]
                r["recall@3 / @5 / @10"] = [round(float(np.mean([g in t[:n] for g, t in zip(gold, top)])), 3) for n in (3, 5, 10)]
                r["rescue: gold in Jev top-3 when not offered"] = (
                    round(float(np.mean([g in t[:3] for g, t, m in zip(gold, top, ~offered) if m])), 3) if (~offered).any() else None)
                r["recall: candidates / + Jev top-3"] = [round(float(offered.mean()), 3),
                                                        round(float(np.mean([g in c or g in t[:3] for g, c, t in zip(gold, cand, top)])), 3)]
            if name in walk:
                r["gold reachable from start"] = round(float(np.mean([walk[name][k][0] for k in keys])), 3)
                r["questions / requests per sample"] = [round(float(np.mean([walk[name][k][i] for k in keys])), 1) for i in (1, 2)]
            res[name] = r
        report[slot] = res
        print(f"\n== {slot}: {len(keys)} samples answered by every arm; gold offered in {offered.mean():.0%}")
        for name, r in res.items():
            print(f"  [{name}]")
            for k, val in r.items():
                if val is not None:
                    print(f"    {k:70} {val}")
    print("\ncost: " + "; ".join(f"{n}: {c}" for n, c in report["cost"].items()))
    if args.output:
        os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
        json.dump(report, open(path(args.output), "w"), indent=1)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    from _setup import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--pilot", required=True)
    a.add_argument("--ontology_terms", default=DEFAULTS["ontology_terms"])
    a.add_argument("--samples", default=DEFAULTS["samples"])
    a.add_argument("--require_embeddings", nargs="*", default=[DEFAULTS["keywords"], DEFAULTS["sub_biomes"]],
                   help="keep training samples with these embeddings (as rerank_pilot.py build); none: no filter")
    a.add_argument("--label_map", default=None)
    a.add_argument("--min_studies", type=int, default=1)
    a.add_argument("--output", required=True)

    def api_options(x):
        x.add_argument("--pilot", required=True)
        x.add_argument("--inputs", required=True)
        x.add_argument("--slots", nargs="+", default=SLOTS)
        x.add_argument("--limit", type=int, default=0, help="first N samples per slot (0 = all)")
        x.add_argument("--model", default="jev-1.14.0")
        x.add_argument("--orders", type=int, default=1, help="option orders per Choice, probabilities averaged")
        x.add_argument("--min_studies", type=int, default=1)
        x.add_argument("--seed", type=int, default=0)
        x.add_argument("--workers", type=int, default=8, help="requests in flight (TypeSafe rate-limits per organisation)")
        x.add_argument("--api_key_path", default="~/MicrobeAtlasProject/jev_api_key")
        x.add_argument("--fake", action="store_true", help="keyword-overlap stand-in for Jev (no network)")
        x.add_argument("--dry_run", action="store_true", help="no cache, no output; with --fake: token count and cost")
    r = sub.add_parser("run")
    api_options(r)
    r.add_argument("--arm", choices=["flat", "topdown"], required=True)
    r.add_argument("--questions", nargs="+", choices=["vocab", "rerank", "fits"], default=["vocab", "rerank", "fits"],
                   help="flat: which questions to ask (e.g. rerank fits: reranking only, no whole-vocabulary Choice)")
    r.add_argument("--start", choices=["lca", "root"], default="lca", help="topdown: where the walk starts")
    r.add_argument("--beam", type=int, default=3)
    r.add_argument("--output", required=True)
    r.add_argument("--cache", default=None, help="default: jev_cache.jsonl next to --output")
    c = sub.add_parser("check_order")
    api_options(c)
    c.add_argument("--cache", required=True)
    c.add_argument("--output", default=None, help="default: check_order.json next to --cache")
    s = sub.add_parser("score")
    s.add_argument("--pilot", required=True)
    s.add_argument("--inputs", required=True)
    s.add_argument("--responses", nargs="*", default=[])
    s.add_argument("--reference", nargs="*", default=[], help="name=file of GPT responses on the same pilot")
    s.add_argument("--min_studies", type=int, default=1)
    s.add_argument("--output", default=None)
    args = p.parse_args(argv)
    {"prepare": prepare, "run": run, "check_order": check_order, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
