#!/usr/bin/env python3
"""
How redundant is the sub-biome vocabulary?

Claim to test: many distinct sub-biome strings are trivial variants of each other
(typos, word permutations, synonyms), so the effective vocabulary is far smaller
than the raw count of distinct strings.

Method: all-pairs cosine over the DISTINCT sub-biome texts (not samples), then
single-linkage merge at several thresholds, and count how many groups remain and
how many samples they cover. Each merged pair is also classified by string form,
to separate "same words reordered" and "one character off" from genuine synonymy -
only the last kind is a real semantic merge.

    python3 scripts/subbiome_redundancy.py --thresholds 0.98 0.95 0.90
"""

import argparse
import json
import os
import re
from collections import Counter

import h5py
import numpy as np

import compare_to_previous_embeddings as C
from compare_to_previous_embeddings import stream, unit


def load_new(target="sub_biomes"):
    path = C.new_unique(target)
    with h5py.File(path, "r") as f:
        texts = [t.decode() for t in f["texts"][:]]
        vecs = f["embeddings"][:].astype(np.float32)
    return texts, unit(vecs)


def load_old(target, max_texts, seed):
    """One vector per distinct text from Dany's per-sample file, sampled."""
    path = C.old_embeddings(target)
    with h5py.File(path, "r") as f:
        first = {}
        col = f["texts"]
        for i in range(0, col.shape[0], 200_000):
            for k, t in enumerate(col[i:i + 200_000]):
                first.setdefault(t.decode(), i + k)
        print(f"  Dany: {len(first):,} distinct texts", flush=True)
        keep = sorted(np.random.default_rng(seed).choice(
            len(first), min(max_texts, len(first)), replace=False))
        items = sorted(first.items(), key=lambda kv: kv[1])
        items = [items[i] for i in keep]
        rows = np.array([r for _, r in items])
        texts = [t for t, _ in items]
        vecs = np.empty((len(rows), f["embeddings"].shape[1]), np.float32)
        order = np.argsort(rows)
        block = 20_000
        srt = rows[order]
        for start in range(0, f["embeddings"].shape[0], block):
            lo, hi = np.searchsorted(srt, [start, start + block])
            if hi > lo:
                slab = f["embeddings"][start:start + block]
                vecs[order[lo:hi]] = slab[srt[lo:hi] - start]
    ok = np.isfinite(vecs).all(axis=1)
    return [t for t, g in zip(texts, ok) if g], unit(vecs[ok])


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def variant_kind(a, b):
    """Why are these two strings similar?"""
    na, nb = re.sub(r"[^a-z0-9 ]", " ", a.lower()), re.sub(r"[^a-z0-9 ]", " ", b.lower())
    ta, tb = na.split(), nb.split()
    if ta == tb:
        return "identical after normalising"
    if sorted(ta) == sorted(tb):
        return "same words, reordered"
    if set(ta) == set(tb):
        return "same word set"
    # cheap edit distance on the normalised strings
    if abs(len(na) - len(nb)) <= 2:
        d, prev = 0, None
        import difflib
        d = sum(1 for op in difflib.SequenceMatcher(None, na, nb).get_opcodes() if op[0] != "equal")
        if d <= 1 and abs(len(na) - len(nb)) <= 2:
            return "near-identical spelling"
    if set(ta) & set(tb):
        return "shares a word"
    return "different words"


def analyse(name, texts, vecs, counts, thresholds, block, examples):
    n = len(texts)
    print(f"\n{name}: {n:,} distinct texts, {sum(counts):,} samples", flush=True)
    unions = {t: Union(n) for t in thresholds}
    best = np.zeros(n, np.float32)
    pair_kinds = {t: Counter() for t in thresholds}
    shown = []

    for start in range(0, n, block):
        sims = vecs[start:start + block] @ vecs.T
        rows = np.arange(start, min(start + block, n))
        sims[np.arange(len(rows)), rows] = -np.inf
        best[rows] = sims.max(axis=1)
        for t in thresholds:
            ri, ci = np.nonzero(sims >= t)
            for r, c in zip(ri, ci):
                g = rows[r]
                if g < c:                                  # each pair once
                    unions[t].join(int(g), int(c))
                    if t == max(thresholds):
                        pair_kinds[t][variant_kind(texts[g], texts[c])] += 1
                        if len(shown) < examples and len(texts[g]) > 4:
                            shown.append((float(sims[r, c]), texts[g], texts[c]))
        del sims

    out = {"n_texts": n, "n_samples": int(sum(counts)),
           "nearest_neighbour_cosine": {q: float(np.percentile(best, p))
                                        for q, p in [("p10", 10), ("p50", 50), ("p90", 90)]},
           "frac_with_nn_above": {str(t): float((best >= t).mean()) for t in thresholds},
           "groups": {}}
    print("  nearest-neighbour cosine: p10 {p10:.3f}  median {p50:.3f}  p90 {p90:.3f}".format(
        **out["nearest_neighbour_cosine"]), flush=True)
    for t in sorted(thresholds, reverse=True):
        roots = [unions[t].find(i) for i in range(n)]
        groups = len(set(roots))
        by_root = Counter()
        for r, c in zip(roots, counts):
            by_root[r] += c
        top = sum(v for _, v in by_root.most_common(500))
        out["groups"][str(t)] = {"n_groups": groups,
                                 "reduction": round(n / groups, 2),
                                 "samples_in_top_500_groups": float(top / sum(counts))}
        print(f"  merge at cos>={t}: {n:,} -> {groups:,} groups ({n/groups:.2f}x); "
              f"top 500 groups cover {top/sum(counts):.1%} of samples", flush=True)
    out["frac_nn_above"] = out.pop("frac_with_nn_above")
    print(f"  texts whose nearest neighbour is >= {max(thresholds)}: "
          f"{(best >= max(thresholds)).mean():.1%}", flush=True)
    kinds = pair_kinds[max(thresholds)]
    if kinds:
        total = sum(kinds.values())
        print(f"  what the {total:,} pairs above {max(thresholds)} actually are:", flush=True)
        for kind, c in kinds.most_common():
            print(f"      {c/total:6.1%}  {kind}", flush=True)
        out["pair_kinds"] = {k: v for k, v in kinds.most_common()}
    for s, a, b in shown[:examples]:
        print(f"      {s:.3f}  {a!r}  <->  {b!r}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    p.add_argument("--block", type=int, default=2000)
    p.add_argument("--max_old_texts", type=int, default=30000)
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_old", action="store_true")
    args = p.parse_args()

    root = os.path.expanduser(args.root)
    C.OLD, C.NEW = f"{root}/sidequest", f"{root}/sidequest/latest"
    out_dir = f"{C.NEW}/embeddings/vs_previous"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    texts, vecs = load_new()
    freq = Counter(t for _, t in stream(f"{C.NEW}/GPT_sub_biomes.txt", False))
    summary["new"] = analyse("new (GPT-5, 3-large 1024d)", texts, vecs,
                             [freq.get(t, 0) for t in texts], args.thresholds, args.block, args.examples)

    if not args.skip_old:
        otexts, ovecs = load_old("sub_biomes", args.max_old_texts, args.seed)
        ofreq = Counter(t for _, t in stream(f"{C.OLD}/GPT_sub_biomes.txt", False))
        summary["dany"] = analyse("Dany (GPT-3.5, 3-small 1536d, sampled)", otexts, ovecs,
                                  [ofreq.get(t, 0) for t in otexts], args.thresholds,
                                  args.block, args.examples)

    path = os.path.join(out_dir, "subbiome_redundancy.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\nwritten: {path}", flush=True)


if __name__ == "__main__":
    main()
