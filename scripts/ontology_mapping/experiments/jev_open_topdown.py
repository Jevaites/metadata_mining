#!/usr/bin/env python3
"""
Top-down over open vocabularies (README 7d, claude/jev-open-topdown.md): Jev's beam walk (jev_pilot.ask_topdown) on
trees that are not limited to the labels Metalog uses, so the walk can end at a label no other study used.

Trees (--tree):
  auto     biome: the ENVO biome branch only (ENVO_00000428 'biome' and its 127 descendants);
           feature / material: the whole open ontology, every ENVO + UBERON term under the 11 meaningful roots
           (anatomical entity, environmental material, environmental system, construction, layer, ...; 17,870 terms).
  climbM   feature / material, a restricted but open vocabulary (M = 300, 1000, ...): from each label the slot uses
           in other studies, climb to the broadest ancestors whose subtree has <= M terms; the walk starts with a
           Choice among these nodes and may end at any term below them. Built per study (leave-one-study-out).
           M = 300: ~56-58 start nodes, ~6k terms; M = 1000: ~28-34 start nodes, ~9-12k terms.
Starts (--start):
  root        the top of the tree;
  lca_proto3  the most specific common ancestor in the tree of the prototype's top 3 (jev_pilot's lca start);
  lca_open5   the same for the top 5 of retrieve-then-choose with similar samples (open_similar);
  without a common ancestor in the tree, the walk starts at the root.
Walk: as jev_pilot.ask_topdown (beam 3, product of edge probabilities, one Choice per open node, 'X itself' when X is
an answer with children, single-option nodes passed without asking). Options: the node's children in the tree, each
with synonyms, definition and up to 6 narrower terms (the slot's Metalog labels below it, most used in other studies
first, then its children). Answers: every tree term except the ontology roots and NO_ANSWER.
Limits: a node with more children than fit one Choice (255 options or ~30k tokens per question; organ part 347 and
regional part of brain 305 children) is split (--wide split, the default): its children go into balanced chunks (seeded
random), each chunk is a Choice in the level's request, and each chunk's top --finalists (3) children plus 'X itself'
meet in a final Choice (one more request); non-finalists get probability 0. --wide trim instead keeps the children
closest to the sample (best retrieval rank of a term in their subtree; `build` stores each sample's top --retrieve
terms). A level's questions are split into requests of <= ~60k tokens.

Steps:
  build  (no API) -> a pilot file (the pilot + open5 + retrieved terms) and an inputs file holding every open term.
  run    (API; on your Mac) one tree x start -> one JSONL line per sample. --fake --dry_run: token count and cost.
  score  (no API) vs the prototype and the retrieve-then-choose runs, on all / unseen-label / seen-label samples.

cd scripts/ontology_mapping; X=~/MicrobeAtlasProject/ontology_mapping/experiments; J=$X/jev; F=$J/followups; O=$F/results
python3 experiments/jev_open_topdown.py build --pilot $X/rerank/pilot_k5anc.jsonl --inputs $F/inputs_gated_open_similar.json \
    --open_pilot $F/gated_open_similar.jsonl --open_responses $O/gated_open_similar.jsonl \
    --label_map ~/MicrobeAtlasProject/metalog/clean/previous_2026-10-01/biome_label_map.tsv \
    --output $F/gated_topdown_open.jsonl --inputs_out $F/inputs_gated_topdown_open.json
python3 experiments/jev_open_topdown.py run --pilot $F/gated_topdown_open.jsonl --inputs $F/inputs_gated_topdown_open.json \
    --tree auto --start root --output $O/gated_td_auto_root.jsonl
(all commands: experiments/run_jev_followups.sh topdown_open)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, path  # noqa: E402
from hierarchy import NO_ANSWER  # noqa: E402
import jev_pilot as jp  # noqa: E402
import jev_questions as W  # noqa: E402
from rerank_prompt import count_tokens  # noqa: E402

BIOME_ROOT = "ENVO_00000428"
UBERON_ROOT = "UBERON_0001062"  # anatomical entity
ENVO_ROOT_LABELS = ("astronomical body part", "environmental material", "environmental system", "construction", "layer",
                    "manufactured product", "mass of environmental material", "environmental zone",
                    "fiat part of an astronomical object", "astronomical object")
Q_BUDGET = 30000    # tokens per question, state included (Jev: 32k)
REQ_BUDGET = 60000  # tokens per request (Jev: 64k)
N_NARROWER = 6


def is_open_term(t):
    return t.startswith(("ENVO_", "UBERON_"))


def open_roots(terms):
    """The 11 meaningful roots: anatomical entity + the parentless ENVO classes named in ENVO_ROOT_LABELS."""
    roots = [UBERON_ROOT] if UBERON_ROOT in terms else []
    for t, v in sorted(terms.items()):
        if t.startswith("ENVO_") and v["label"] in ENVO_ROOT_LABELS and not [p for p in v["parents"] if is_open_term(p)]:
            roots.append(t)
    return roots


# ----------------------------------------------------------------------------- the trees
class Forest:
    """ENVO + UBERON as children / descendant sets, and the trees the walk runs on (cached)."""

    def __init__(self, ctx, roots, links="isa", curated=None):
        """links: 'isa' (is_a only) or 'both' (is_a + part_of: a node's children also hold the terms that are part of
        it; needs the part_of lists of `add_links`). curated: rows of the curated top layer (load_curated)."""
        self.ctx, self.links, self.curated = ctx, links, curated or []
        keep = {t for t in ctx.terms if is_open_term(t)}
        self.kids = defaultdict(list)
        for t in sorted(keep):
            for p in ctx.terms[t]["parents"]:
                if p in keep:
                    self.kids[p].append(t)
        self.part_kids = defaultdict(list)
        for t in sorted(keep):
            for p in ctx.terms[t].get("part_of", []):
                if p in keep:
                    self.part_kids[p].append(t)
        self._desc_both = {}
        self.desc = defaultdict(set)
        for t in keep:
            for a in ctx.anc.get(t, ()):
                if a in keep:
                    self.desc[a].add(t)
        self.roots = [r for r in roots if r in keep]
        self.open = set(self.roots).union(*[self.desc[r] for r in self.roots])
        self.branch = {BIOME_ROOT} | self.desc[BIOME_ROOT]
        self._trees = {}

    def size(self, t):
        return len(self.desc[t]) + 1

    def kd(self, t):
        """Children by the chosen links."""
        if self.links == "both":
            return sorted(set(self.kids[t]) | set(self.part_kids[t]))
        return self.kids[t]

    def dsc(self, t):
        """Descendants by the chosen links."""
        if self.links != "both":
            return self.desc[t]
        if t not in self._desc_both:
            seen, stack = set(), list(self.kd(t))
            while stack:
                u = stack.pop()
                if u not in seen:
                    seen.add(u)
                    stack.extend(self.kd(u))
            self._desc_both[t] = seen
        return self._desc_both[t]

    def climb_cover(self, labels, M):
        """From each label (in the open vocabulary), climb while every next ancestor's subtree has <= M terms; keep
        the last nodes reached, then drop the nodes that lie inside another kept node."""
        c = set()
        for t in labels:
            if t not in self.open:
                continue
            best = frontier = {t}
            while frontier:
                nxt = {p for u in frontier for p in self.ctx.terms[u]["parents"]
                       if p in self.open and self.size(p) <= M}
                if not nxt:
                    break
                best = frontier = nxt
            c |= best
        return sorted(x for x in c if not (self.ctx.anc.get(x, set()) & c))

    def tree(self, name, slot, study):
        """-> {name, nodes, top (the root's children), answers}. biome always walks the biome branch."""
        if slot == "biome":
            key = ("branch",)
        elif name == "auto":
            key = ("open",)
        elif name.startswith("climb"):
            key = ("climb", int(name[5:]), slot, study)
        elif name == "curated":
            key = ("curated", slot)
        else:
            raise ValueError(f"unknown tree {name}")
        if key not in self._trees:
            virtual = {}
            if key[0] == "branch":
                nodes, top, tname = self.branch, [c for c in self.kids[BIOME_ROOT] if c in self.branch], "branch"
            elif key[0] == "open":
                nodes, top, tname = self.open, list(self.roots), "open"
            elif key[0] == "climb":
                top = self.climb_cover(self.ctx.vocab(slot, study)[0], key[1])
                nodes, tname = set(top).union(*[self.desc[x] for x in top]), f"climb{key[1]}"
            else:
                top, virtual = self.curated_top(slot)
                anchors = {a for v, kids in virtual.items() for a in kids if a not in virtual}
                nodes, tname = set(anchors).union(*[self.dsc(a) for a in anchors]), "curated"
            if self.links == "both" and key[0] != "curated":
                nodes = set(nodes).union(*[self.dsc(x) for x in top])
            answers = nodes - set(self.roots) - NO_ANSWER - {BIOME_ROOT}
            vbelow = {}
            for v in sorted(virtual, key=lambda x: x.count(":C:"), reverse=True):  # categories, then groups
                vbelow[v] = set().union(*[vbelow.get(k) or ({k} | self.dsc(k)) & nodes for k in virtual[v]])
            self._trees[key] = {"name": tname + ("_both" if self.links == "both" else ""), "nodes": nodes, "top": top,
                                "answers": answers, "virtual": virtual, "vbelow": vbelow}
        return self._trees[key]

    def curated_top(self, slot):
        """The curated top layer of one slot as virtual nodes: ROOT -> groups -> categories -> anchor terms. The
        virtual nodes are added to the term table (label = name, definition = description) so they are described like
        terms; they are never answers. -> (groups, {virtual node: children})."""
        rows = [r for r in self.curated if r["slot"] == slot and r["anchor_id"] in self.ctx.terms]
        if not rows:
            raise SystemExit(f"no curated top layer for {slot} (--curated)")
        top, virtual = [], {}
        for r in rows:
            g, c = f"CUR:{slot}:G:{r['group']}", f"CUR:{slot}:C:{r['category']}"
            for v, name, d in ((g, r["group"], r["group_description"]), (c, r["category"], r["category_description"])):
                if v not in self.ctx.terms:
                    self.ctx.terms[v] = {"label": name, "synonyms": [], "definition": d, "parents": []}
            if g not in virtual:
                top.append(g)
                virtual[g] = []
            if c not in virtual[g]:
                virtual[g].append(c)
            virtual.setdefault(c, [])
            if r["anchor_id"] not in virtual[c]:
                virtual[c].append(r["anchor_id"])
        return top, virtual

    def children(self, node, tr):
        if node == jp.ROOT:
            return list(tr["top"])
        if node in tr["virtual"]:
            return list(tr["virtual"][node])
        return [c for c in self.kd(node) if c in tr["nodes"]]

    def below(self, node, tr):
        """Tree terms at or below node."""
        if node == jp.ROOT:
            return tr["nodes"]
        if node in tr["vbelow"]:
            return tr["vbelow"][node]
        return ({node} | self.dsc(node)) & tr["nodes"]

    def lca(self, ids, tr):
        common = None
        for t in ids:
            up = ({t} | self.ctx.anc.get(t, set())) & tr["nodes"]
            common = up if common is None else common & up
        if not common:
            return jp.ROOT
        return min(common, key=lambda x: (self.size(x), -len(self.ctx.anc.get(x, ())), x))


# ----------------------------------------------------------------------------- options and the walk
def child_options(F, node, tr, o):
    """-> [(name, child or None for 'node itself', description)], children ordered by Metalog use below them (other
    studies), then label (the question shuffles them anyway)."""
    ctx, slot = F.ctx, o["slot"]
    labels, freq = ctx.vocab(slot, o["study"])
    kids = F.children(node, tr)
    below = {c: F.below(c, tr) for c in kids}
    used = {c: sum(freq[v] for v in labels if v in below[c]) for c in kids}
    kids = sorted(kids, key=lambda c: (-used[c], ctx.label(c), c))
    out = []
    for c, name in zip(kids, jp.option_names(kids, ctx)):
        if c in tr["virtual"]:  # a curated group / category: its own children (its description gives examples)
            nar = list(tr["virtual"][c])[:2 * N_NARROWER]
        else:
            nar = [v for v in labels if v in below[c] and v != c][:N_NARROWER]
            nar += [k for k in sorted(F.children(c, tr), key=ctx.label) if k not in nar][:N_NARROWER - len(nar)]
        out.append((name, c, jp.describe(c, ctx, narrower=[ctx.label(v) for v in nar])))
    if node != jp.ROOT and node in tr["answers"] and kids:
        name, desc = W.itself_option(ctx.label(node))
        out.append((name, None, desc))
    return out


def question(o, node, F, opts, order):
    return jp.choice_question(W.which_branch(o["slot"], None if node == jp.ROOT else F.ctx.label(node)),
                              [x[0] for x in opts], [x[2] for x in opts], order)


def fit_options(F, node, tr, o, state_tokens, seed):
    """The options of one node, cut to what fits one Choice: <= MAX_OPTIONS and <= Q_BUDGET tokens with the state.
    Children are kept by closeness to the sample (best rank of a term of their subtree among the retrieved terms),
    'X itself' always stays. -> (options, number of children cut, the question)."""
    opts = child_options(F, node, tr, o)
    order = jp.orders(len(opts), 1, seed)[0]
    q = question(o, node, F, opts, order)
    if len(opts) <= jp.MAX_OPTIONS and state_tokens + count_tokens(json.dumps(q)) <= Q_BUDGET:
        return opts, 0, q
    rank = {t: r for r, t in reversed(list(enumerate(o.get("retrieved", []))))}
    inf = len(rank) + 1

    def closeness(x):
        if x[1] is None:
            return -1
        return min([rank.get(t, inf) for t in F.below(x[1], tr)] or [inf])
    ranked = sorted(opts, key=closeness)
    n = min(len(ranked), jp.MAX_OPTIONS)
    while True:
        keep = [x for x in opts if x in ranked[:n]]  # back in the original order
        order = jp.orders(len(keep), 1, seed)[0]
        q = question(o, node, F, keep, order)
        if state_tokens + count_tokens(json.dumps(q)) <= Q_BUDGET or n <= 2:
            return keep, len(opts) - len(keep), q
        n = max(2, int(n * 0.85))


def chunk_questions(F, node, tr, o, state_tokens, seed):
    """The node's question, or, when it does not fit one Choice (> MAX_OPTIONS options or > Q_BUDGET tokens with the
    state), its children split into the fewest balanced chunks that each fit (seeded random assignment); 'X itself'
    then waits for the final round. -> (options, [(question, indices of its options)])."""
    opts = child_options(F, node, tr, o)
    q = question(o, node, F, opts, jp.orders(len(opts), 1, seed)[0])
    if len(opts) <= jp.MAX_OPTIONS and state_tokens + count_tokens(json.dumps(q)) <= Q_BUDGET:
        return opts, [(q, list(range(len(opts))))]
    kids = [i for i, x in enumerate(opts) if x[1] is not None]
    rng = np.random.default_rng(jp.stable_seed(*seed, "chunks"))
    kids = [kids[i] for i in rng.permutation(len(kids))]
    n = 2
    while True:
        parts = [sorted(kids[j::n]) for j in range(n)]
        qs = [question(o, node, F, [opts[i] for i in part], jp.orders(len(part), 1, seed + (f"chunk{j}",))[0])
              for j, part in enumerate(parts)]
        if all(len(part) <= jp.MAX_OPTIONS and state_tokens + count_tokens(json.dumps(q)) <= Q_BUDGET
               for part, q in zip(parts, qs)) or n >= len(kids) // 2:
            return opts, list(zip(qs, parts))
        n += 1


def advance(F, path, tr):
    """Follow nodes with a single option without asking. -> (path, finished)."""
    while True:
        kids = F.children(path[-1], tr)
        n = len(kids) + (1 if path[-1] != jp.ROOT and path[-1] in tr["answers"] and kids else 0)
        if n == 0:
            return path, True
        if n > 1:
            return path, False
        path = path + (kids[0],)


def split_requests(qs, state_tokens):
    """Greedy packing of {key: (question, tokens)} into requests of <= REQ_BUDGET tokens."""
    groups, cur, size = [], {}, state_tokens
    for k, (q, n) in qs.items():
        if cur and size + n > REQ_BUDGET:
            groups.append(cur)
            cur, size = {}, state_tokens
        cur[k] = q
        size += n
    if cur:
        groups.append(cur)
    return groups


def start_node(F, o, tr, start):
    if start == "root":
        return jp.ROOT
    ids = [c["term_id"] for c in o["candidates"][:3]] if start == "lca_proto3" else list(o.get("open5", []))[:5]
    return F.lca(ids, tr) if ids else jp.ROOT


async def ask(client, o, F, args):
    tr = F.tree(args.tree, o["slot"], o["study"])
    start = start_node(F, o, tr, args.start)
    gold = o["gold"]
    state = W.state(o["text"])
    st_tok = count_tokens(json.dumps(state))
    beam = [(1.0, *advance(F, (start,), tr))]
    trace, n_q, n_req, tokens, model, cut, gold_cut = [], 0, 0, 0, args.model, 0, False
    for level in range(jp.MAX_LEVELS):
        nodes = sorted({p[-1] for _, p, done in beam if not done})
        if not nodes:
            break
        plans, qs, finals = {}, {}, {}
        for i, n in enumerate(nodes):
            seed = (args.seed, o["slot"], o["sample_id"], n)
            if args.wide == "trim":
                o_n, c_n, q = fit_options(F, n, tr, o, st_tok, seed)
                plans[n] = (o_n, [(q, list(range(len(o_n))))])
                if c_n:
                    cut += 1
                    if gold in F.below(n, tr) and gold != n and not any(x[1] and gold in F.below(x[1], tr) for x in o_n):
                        gold_cut = True
            else:
                plans[n] = chunk_questions(F, n, tr, o, st_tok, seed)
            chunks = plans[n][1]
            if len(chunks) == 1:
                qs[f"n{i}"] = (chunks[0][0], count_tokens(json.dumps(chunks[0][0])))
            else:
                cut += 1
                for j, (q, _) in enumerate(chunks):
                    qs[f"n{i}c{j}"] = (q, count_tokens(json.dumps(q)))

        async def send(questions):
            nonlocal tokens, model, n_q, n_req
            groups = split_requests(questions, st_tok)
            rs = await asyncio.gather(*(client.system_one(state, g) for g in groups))
            out = {}
            for r in rs:
                out.update(r.answers)
                tokens += jp.usage_in(r)
                model = r.model
            n_q, n_req = n_q + len(questions), n_req + len(groups)
            return out
        answers = await send(qs)
        # split nodes: each chunk's top --finalists children (+ 'X itself') meet in one final Choice
        final_qs = {}
        for i, n in enumerate(nodes):
            o_n, chunks = plans[n]
            if len(chunks) > 1:
                names = [x[0] for x in o_n]
                fin = [k for k, x in enumerate(o_n) if x[1] is None]
                for j, (_, idx) in enumerate(chunks):
                    p = jp.mean_probs(answers, [f"n{i}c{j}"], [names[k] for k in idx])
                    fin += [idx[k] for k in np.argsort(-p, kind="stable")[:args.finalists]]
                fin = sorted(set(fin))
                finals[n] = fin
                q = question(o, n, F, [o_n[k] for k in fin], jp.orders(len(fin), 1, (args.seed, o["slot"], o["sample_id"], n, "final"))[0])
                final_qs[f"n{i}f"] = (q, count_tokens(json.dumps(q)))
                if gold in F.below(n, tr) and gold != n and not any(o_n[k][1] and gold in F.below(o_n[k][1], tr) for k in fin):
                    gold_cut = True
        if final_qs:
            answers.update(await send(final_qs))
        opts, probs = {n: plans[n][0] for n in nodes}, {}
        for i, n in enumerate(nodes):
            names = [x[0] for x in opts[n]]
            if n in finals:
                full = np.zeros(len(names))
                full[finals[n]] = jp.mean_probs(answers, [f"n{i}f"], [names[k] for k in finals[n]])
                probs[n] = full
            else:
                probs[n] = jp.mean_probs(answers, [f"n{i}"], names)
        trace.append({n: dict(sorted(((x[0], round(float(p), 4)) for x, p in zip(opts[n], probs[n])),
                                     key=lambda kv: -kv[1])[:5]) for n in nodes})
        new = [b for b in beam if b[2]]
        for p, route, done in beam:
            if not done:
                for (_, child, _), pc in zip(opts[route[-1]], probs[route[-1]]):
                    new.append((p * pc, route, True) if child is None else (p * pc, *advance(F, route + (child,), tr)))
        beam = sorted(new, key=lambda b: -b[0])[:args.beam]
    terminals = {}
    for p, route, done in beam:
        if done and route[-1] in tr["answers"]:
            terminals[route[-1]] = terminals.get(route[-1], 0.0) + p
    best = max((b for b in beam if b[2]), key=lambda b: b[0], default=(0.0, (start,), False))
    alive = any((done and r[-1] == gold) or (not done and gold in F.below(r[-1], tr)) for _, r, done in beam)
    return {"slot": o["slot"], "sample_id": o["sample_id"], "arm": f"td_{tr['name']}_{args.start}", "tree": tr["name"],
            "model": model, "start": start, "tree_size": len(tr["nodes"]), "start_size": len(F.below(start, tr)),
            "gold_in_tree": gold in tr["answers"], "gold_reachable": gold in F.below(start, tr) and gold in tr["answers"],
            "gold_alive_end": alive, "gold_cut": gold_cut, "nodes_cut": cut, "wide": args.wide,
            "terminals": {t: round(p, 6) for t, p in terminals.items()}, "best_path": list(best[1]), "trace": trace,
            "questions": n_q, "requests": n_req, "levels": len(trace), "usage": {"in": tokens, "out": 0}}


def load_curated(p):
    """The curated top layer (TSV: slot, group, group_description, category, category_description, anchor_id,
    anchor_label), one row per anchor term."""
    import csv
    with open(path(p)) as handle:
        return [r for r in csv.DictReader(handle, delimiter="\t") if r.get("slot") and not r["slot"].startswith("#")]


def load_forest(inputs_path, args):
    inputs = json.load(open(path(inputs_path)))
    ctx = jp.Ctx(inputs, SimpleNamespace(min_studies=getattr(args, "min_studies", 1)))
    links = getattr(args, "links", "isa")
    if links == "both" and not any("part_of" in v for v in inputs["terms"].values()):
        raise SystemExit(f"--links both needs part_of links in {inputs_path}: run `add_links` first")
    curated = load_curated(args.curated) if getattr(args, "tree", "") == "curated" else None
    return Forest(ctx, inputs.get("open_roots") or open_roots(inputs["terms"]), links=links, curated=curated)


def add_links(args):
    """Add part_of parents (ENVO BFO:0000050, UBERON part_of) from the OBO files to every open term of an inputs
    file (term['part_of'])."""
    inputs = json.load(open(path(args.inputs)))
    part = defaultdict(set)
    for f in args.obo:
        cur = None
        for line in open(path(f)):
            line = line.rstrip("\n")
            if line in ("[Term]", "[Typedef]"):
                cur = None
            elif line.startswith("id: "):
                cur = line[4:].strip().replace(":", "_")
            elif cur and line.startswith("relationship: "):
                f_ = line.split()
                if len(f_) > 2 and f_[1] in ("part_of", "BFO:0000050"):
                    part[cur].add(f_[2].replace(":", "_"))
    n = 0
    for t, v in inputs["terms"].items():
        if is_open_term(t):
            v["part_of"] = sorted(p for p in part.get(t, ()) if p in inputs["terms"])
            n += len(v["part_of"])
    json.dump(inputs, open(path(args.output or args.inputs), "w"))
    print(f"{n:,} part_of links on {sum(bool(v.get('part_of')) for v in inputs['terms'].values()):,} terms -> {args.output or args.inputs}")


def run(args):
    pilot = [o for o in jp.read_jsonl(args.pilot) if o["slot"] in args.slots]
    if args.tree != "auto":
        pilot = [o for o in pilot if o["slot"] != "biome"]  # biome only walks its branch (--tree auto)
    if args.unseen_only:
        pilot = [o for o in pilot if o["gold"] not in seen_labels(args, o)]
    if args.limit:
        pilot = [o for s in args.slots for o in [x for x in pilot if x["slot"] == s][:args.limit]]
    F = load_forest(args.inputs, args)
    client, opener, cache = jp.make_client(args)
    t0 = time.time()
    results, errors = jp.gather(pilot, lambda o: ask(client, o, F, args), opener, args.workers)
    records = [r for r in results if r]
    n_in = sum(r["usage"]["in"] for r in records)
    print(f"td_{args.tree}{'_both' if args.links == 'both' else ''}_{args.start}: {len(records)} / {len(pilot)} samples in {time.time() - t0:.0f}s, "
          f"{sum(r['questions'] for r in records):,} questions, {sum(r['requests'] for r in records):,} requests, "
          f"{n_in:,} input tokens = ${n_in / 1e6 * jp.PRICE_IN:.2f}")
    for slot in args.slots:
        rs = [r for r in records if r["slot"] == slot]
        if rs:
            print(f"  {slot:8} gold in tree {np.mean([r['gold_in_tree'] for r in rs]):.2f}, reachable from the start "
                  f"{np.mean([r['gold_reachable'] for r in rs]):.2f}, alive at the end {np.mean([r['gold_alive_end'] for r in rs]):.2f}; "
                  f"levels {np.mean([r['levels'] for r in rs]):.1f}, nodes too wide for one Choice {np.mean([r['nodes_cut'] for r in rs]):.2f} per sample")
    if args.fake and args.dry_run:
        print("  o200k estimate with fake answers: the real paths (and cost) will differ")
    if cache:
        print(f"  cache {cache.path}: {cache.hits} replayed, {cache.misses} new requests "
              f"({cache.new_tokens:,} tokens = ${cache.new_tokens / 1e6 * jp.PRICE_IN:.2f} spent now)")
    if not args.dry_run:
        os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
        with open(path(args.output), "w") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")
        print(f"  wrote {args.output}")
    if errors:
        print(f"{len(errors)} samples failed (rerun the same command: answered requests come from the cache), e.g. {errors[0]}")


_VOCAB = {}


def seen_labels(args, o):
    """The slot's labels used by other studies (for --unseen_only), from the inputs file."""
    if "ctx" not in _VOCAB:
        _VOCAB["ctx"] = jp.Ctx(json.load(open(path(args.inputs))), SimpleNamespace(min_studies=1))
    return set(_VOCAB["ctx"].vocab(o["slot"], o["study"])[0])


# ----------------------------------------------------------------------------- build
def build(args):
    import jev_baselines as jb
    pilot = jp.read_jsonl(args.pilot)
    inputs = json.load(open(path(args.inputs)))
    opened = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(args.open_pilot)} if args.open_pilot else {}
    resp = {(r["slot"], r["sample_id"]): r for r in jp.read_jsonl(args.open_responses)} if args.open_responses else {}
    T = jb.load_training(args)
    terms = T.terms
    keep = np.array([is_open_term(t) for t in terms["term_id"]])
    info = {t: {"label": l, "synonyms": [s for s in syn.split("||") if s][:3], "definition": d[:220],
                "parents": sorted({p for p in par.split("||") if p})}
            for t, l, syn, d, par, k in zip(terms["term_id"], terms["label"], terms["synonyms"], terms["definition"],
                                            terms["parents"], keep) if k}
    new_inputs = dict(inputs, terms={**inputs["terms"], **info}, open_roots=open_roots(info))
    open_ids = terms["term_id"].to_numpy()[keep]
    TMo = T.TM[keep]
    out, n_open5 = [], 0
    for o in pilot:
        k = (o["slot"], o["sample_id"])
        rec = dict(o)
        if k in resp and k in opened:
            ids = [c["term_id"] for c in opened[k]["candidates"]]
            rec["open5"] = [ids[i] for i in np.argsort(-np.array(resp[k]["rerank"]), kind="stable")[:5]]
            n_open5 += 1
        if o["slot"] != "biome":
            sc = TMo @ T.vec([o["sample_id"]])[0]
            rec["retrieved"] = [str(t) for t in open_ids[np.argsort(-sc)[:args.retrieve]]]
        out.append(rec)
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    with open(path(args.output), "w") as handle:
        for r in out:
            handle.write(json.dumps(r) + "\n")
    json.dump(new_inputs, open(path(args.inputs_out), "w"))
    F = Forest(jp.Ctx(new_inputs, SimpleNamespace(min_studies=1)), new_inputs["open_roots"])
    print(f"wrote {len(out)} samples ({n_open5} with the open_similar top 5) to {args.output}; {len(new_inputs['terms'])} "
          f"terms to {args.inputs_out}. Roots: {', '.join(F.ctx.label(r) for r in F.roots)}; open vocabulary "
          f"{len(F.open):,} terms, biome branch {len(F.branch)}")


# ----------------------------------------------------------------------------- score
def score(args):
    pilot = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(args.pilot)}
    F = load_forest(args.inputs, args)
    ctx = F.ctx
    td = {}
    for f in args.responses:
        for r in jp.read_jsonl(f):
            td.setdefault(r["arm"], {})[(r["slot"], r["sample_id"])] = r
    opens = {}
    for spec in args.open or []:
        name, rest = spec.split("=", 1)
        pf, rf = rest.split(",")
        op = {(o["slot"], o["sample_id"]): o for o in jp.read_jsonl(pf)}
        opens[name] = {}
        for r in jp.read_jsonl(rf):
            k = (r["slot"], r["sample_id"])
            if k in op:
                opens[name][k] = op[k]["candidates"][int(np.argmax(r["rerank"]))]["term_id"]
    report = {}
    for slot in SLOTS:
        arms = [a for a, v in td.items() if any(k[0] == slot for k in v)]
        if not arms:
            continue
        keys = [k for k in pilot if k[0] == slot and all(k in td[a] for a in arms) and all(k in v for v in opens.values())]
        if not keys:
            continue
        gold = [pilot[k]["gold"] for k in keys]
        studies = np.array([pilot[k]["study"] for k in keys])
        unseen = np.array([pilot[k]["gold"] not in ctx.vocab(slot, pilot[k]["study"])[0] for k in keys])
        picks = {"prototype": [pilot[k]["candidates"][0]["term_id"] for k in keys]}
        picks.update({name: [v[k] for k in keys] for name, v in opens.items()})
        for a in arms:
            picks[a] = []
            for k in keys:
                t = td[a][k]["terminals"]
                picks[a].append(max(t, key=t.get) if t else None)
        ex = {n: np.array([p == g for p, g in zip(pk, gold)]) for n, pk in picks.items()}
        res = {}
        for subset, m in [("all", np.ones(len(keys), bool)), ("unseen labels", unseen), ("seen labels", ~unseen)]:
            if not m.any():
                continue
            row = {"n": int(m.sum())}
            for n, pk in picks.items():
                rel = np.array([jp.relation(p, g, ctx.anc) for p, g in zip(pk, gold)])[m]
                r = {x: round(float((rel == x).mean()), 3) for x in ("exact", "coarser", "too_specific", "other", "none")}
                r["exact or coarser"] = round(r["exact"] + r["coarser"], 3)
                if n in td:
                    rs = [td[n][k] for k, mm in zip(keys, m) if mm]
                    finished = [g in r_["terminals"] and g != p for r_, g, p in
                                zip(rs, np.array(gold, object)[m], np.array(pk, object)[m])]
                    r.update({"gold in tree": round(float(np.mean([x["gold_in_tree"] for x in rs])), 3),
                              "gold reachable from start": round(float(np.mean([x["gold_reachable"] for x in rs])), 3),
                              "gold on the beam at the end": round(float(np.mean([x["gold_alive_end"] for x in rs])), 3),
                              "gold finished, not picked": round(float(np.mean(finished)), 3),
                              "gold cut by the width limit": round(float(np.mean([x["gold_cut"] for x in rs])), 3),
                              "start = root": round(float(np.mean([x["start"] == jp.ROOT for x in rs])), 3),
                              "levels / requests per sample": [round(float(np.mean([x[f] for x in rs])), 1) for f in ("levels", "requests")]})
                    for ref in ["prototype"] + list(opens):
                        r[f"vs {ref} (95% CI)"] = jp.boot_diff(ex[n][m], ex[ref][m], studies[m])
                row[n] = r
            res[subset] = row
        cost = {a: f"${sum(td[a][k]['usage']['in'] for k in keys) / 1e6 * jp.PRICE_IN:.2f}" for a in arms}
        res["cost"] = cost
        report[slot] = res
        print(f"\n== {slot} ({len(keys)} samples; cost {', '.join(f'{a} {c}' for a, c in cost.items())})")
        for subset, row in res.items():
            if subset == "cost":
                continue
            print(f"  [{subset}] n {row['n']}")
            for n in picks:
                r = row[n]
                print(f"    {n:26} exact {r['exact']:.3f} | coarser {r['coarser']:.3f} | too specific {r['too_specific']:.3f} "
                      f"| other {r['other']:.3f} | none {r['none']:.3f}")
                if n in td:
                    print("      " + "; ".join(f"{k} {v}" for k, v in r.items()
                                               if k not in ("exact", "coarser", "too_specific", "other", "none", "exact or coarser")))
    if args.output:
        os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
        json.dump(report, open(path(args.output), "w"), indent=1)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    from _setup import DEFAULTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes", "term_vectors"]:
        b.add_argument(f"--{name}", default=DEFAULTS[name])
    b.add_argument("--label_map", default=None)
    b.add_argument("--pilot", required=True)
    b.add_argument("--inputs", required=True, help="an inputs file of the same pilot (slot use per study)")
    b.add_argument("--open_pilot", default=None, help="jev_followups.py open --similar 5 output (for lca_open5)")
    b.add_argument("--open_responses", default=None, help="its jev_pilot.py run output")
    b.add_argument("--retrieve", type=int, default=2000, help="retrieved terms kept per sample (width limit ranking)")
    b.add_argument("--output", required=True)
    b.add_argument("--inputs_out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--inputs", required=True)
    r.add_argument("--tree", default="auto", help="auto (biome branch / whole open ontology), climbM (e.g. climb300) or "
                   "curated (the --curated top layer + the ontology below its anchor terms)")
    r.add_argument("--links", choices=["isa", "both"], default="isa", help="walk is_a links only, or is_a + part_of")
    r.add_argument("--curated", default=os.path.join(HERE, "curated_top_layer.tsv"))
    r.add_argument("--start", choices=["root", "lca_proto3", "lca_open5"], default="root")
    r.add_argument("--slots", nargs="+", default=SLOTS)
    r.add_argument("--unseen_only", action="store_true", help="only the samples whose gold no other study uses")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--beam", type=int, default=3)
    r.add_argument("--wide", choices=["split", "trim"], default="split",
                   help="a node too wide for one Choice: split its children into chunks, then a final Choice among each "
                        "chunk's top --finalists (+ 'X itself'); or trim to the children closest to the sample by retrieval")
    r.add_argument("--finalists", type=int, default=3)
    r.add_argument("--model", default="jev-1.14.0")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--api_key_path", default="~/MicrobeAtlasProject/jev_api_key")
    r.add_argument("--fake", action="store_true")
    r.add_argument("--dry_run", action="store_true")
    r.add_argument("--output", required=True)
    r.add_argument("--cache", default=None, help="default: jev_cache.jsonl next to --output")
    k = sub.add_parser("add_links")
    k.add_argument("--inputs", required=True)
    k.add_argument("--obo", nargs="+", default=["~/MicrobeAtlasProject/ontologies/envo.obo",
                                                "~/MicrobeAtlasProject/ontologies/uberon-basic.obo"])
    k.add_argument("--output", default=None, help="default: update --inputs in place")
    s = sub.add_parser("score")
    s.add_argument("--pilot", required=True, help="the original pilot (the prototype's candidates)")
    s.add_argument("--inputs", required=True, help="the build --inputs_out")
    s.add_argument("--responses", nargs="+", required=True, help="run outputs (one arm each)")
    s.add_argument("--open", nargs="*", default=None, help="retrieve-then-choose runs, name=open_pilot,responses")
    s.add_argument("--output", default=None)
    args = p.parse_args(argv)
    {"build": build, "run": run, "score": score, "add_links": add_links}[args.step](args)


if __name__ == "__main__":
    main()
