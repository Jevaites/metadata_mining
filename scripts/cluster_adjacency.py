#!/usr/bin/env python3
"""
Which community clusters look alike in metadata space?

Two readings when a pair is distinct biologically but adjacent textually:
  (a) the clustering over-split something the metadata calls one habitat, or
  (b) the metadata is too coarse to separate a real biological difference.
You tell them apart by reading the strings, so the top pairs are printed.

Mirror case: one string spread across many clusters - the metadata merging things
the biology keeps apart. That is the 'agricultural soil' pattern from
scatter-arms-string-collisions.md, measured here against the clusters.

'High' cosine is defined against the distribution of all centroid pairs, not an
absolute number: embedding spaces are anisotropic and every pair scores high.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/cluster_adjacency.py [coarse]
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"
BLOCK, TOP = 512, 25


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
            self.p[ra] = rb


def load_join(tag):
    out = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            out[s] = c
    return out


def spread(join, path, is_kw, top=8):
    """For the commonest strings, how many distinct clusters do they cover?"""
    per = defaultdict(set)
    freq = Counter()
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        c = join.get(sid)
        if c is None or not raw.strip():
            continue
        t = clean_text(raw, is_kw)
        per[t].add(c)
        freq[t] += 1
    rows = [(t, n, len(per[t])) for t, n in freq.most_common(top)]
    allc = len({c for v in per.values() for c in v})
    covered = sum(1 for v in per.values() if len(v) > 1)
    return rows, allc, len(freq), covered


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    z = np.load(f"{OUT}/cluster_centroids_{tag}.npz", allow_pickle=True)
    C = z["centroids"].astype(np.float32)
    ids = [str(x) for x in z["cluster"]]
    meta = {r["cluster"]: r for r in json.load(open(f"{OUT}/cluster_coherence_{tag}.json"))["clusters"]}
    n = len(ids)
    print(f"{tag}: {n:,} cluster centroids, dim {C.shape[1]}", flush=True)

    best_j = np.zeros(n, dtype=np.int64)
    best_v = np.full(n, -2.0, dtype=np.float32)
    samp = []
    pairs = []
    rng = np.random.default_rng(0)
    for s in range(0, n, BLOCK):
        sims = C[s:s + BLOCK] @ C.T
        for r in range(sims.shape[0]):
            sims[r, s + r] = -2.0
        j = sims.argmax(axis=1)
        v = sims[np.arange(sims.shape[0]), j]
        best_j[s:s + len(j)], best_v[s:s + len(v)] = j, v
        samp.append(sims[:, rng.integers(0, n, 40)].ravel())
        hi = np.argwhere(sims > 0.90)
        for r, c in hi:
            if s + r < c:
                pairs.append((float(sims[r, c]), s + r, int(c)))
    samp = np.concatenate(samp)

    print(f"\n  centroid-pair cosine: median {np.median(samp):.3f}  p90 {np.percentile(samp, 90):.3f}"
          f"  p99 {np.percentile(samp, 99):.3f}  max {samp.max():.3f}", flush=True)
    print(f"  each cluster's NEAREST other cluster: median {np.median(best_v):.3f}  "
          f"p10 {np.percentile(best_v, 10):.3f}  p90 {np.percentile(best_v, 90):.3f}", flush=True)
    print(f"  pairs above 0.90: {len(pairs):,} of {n * (n - 1) // 2:,} "
          f"({len(pairs) / (n * (n - 1) / 2):.4%})", flush=True)

    print(f"\n  MOST textually adjacent cluster pairs")
    pairs.sort(reverse=True)
    for v, a, b in pairs[:12]:
        ma, mb = meta[ids[a]], meta[ids[b]]
        print(f"    cos {v:.3f}   #{ids[a]} ({ma['members']:,}) <-> #{ids[b]} ({mb['members']:,})")
        print(f"        {ma['modal_text'][:88]}")
        print(f"        {mb['modal_text'][:88]}")

    print(f"\n  how far would merging on metadata collapse the {n:,} clusters?", flush=True)
    for thr in (0.98, 0.96, 0.94, 0.92, 0.90):
        u = Union(n)
        for v, a, b in pairs:
            if v >= thr:
                u.join(a, b)
        comp = len({u.find(i) for i in range(n)})
        big = Counter(u.find(i) for i in range(n)).most_common(1)[0][1]
        print(f"    >= {thr:.2f}: {comp:,} groups ({n / comp:.2f}x fewer)  "
              f"largest group {big:,} clusters", flush=True)

    join = load_join(tag)
    spread_out = {}
    for target, is_kw in [("sub_biomes", False), ("keywords", True)]:
        rows, allc, ntexts, multi = spread(join, f"{NEW}/GPT_{target}.txt", is_kw)
        spread_out[target] = {"distinct_strings": ntexts, "clusters": allc,
                              "strings_spanning_multiple": multi,
                              "share_spanning_multiple": multi / ntexts,
                              "top": [{"text": t, "samples": c, "clusters": k} for t, c, k in rows]}
        print(f"\n  {target}: {ntexts:,} distinct strings over {allc:,} clusters; "
              f"{multi:,} strings ({multi / ntexts:.1%}) span more than one cluster", flush=True)
        print(f"    {'samples':>9} {'clusters':>9}  string")
        for t, cnt, k in rows:
            print(f"    {cnt:>9,} {k:>9,}  {t[:76]}", flush=True)

    json.dump({"tag": tag, "n_clusters": n,
               "centroid_cos": {"median": float(np.median(samp)),
                                "p90": float(np.percentile(samp, 90)),
                                "p99": float(np.percentile(samp, 99))},
               "nearest_other": {"median": float(np.median(best_v)),
                                 "p10": float(np.percentile(best_v, 10)),
                                 "p90": float(np.percentile(best_v, 90))},
               "pairs_above_0.90": len(pairs),
               "top_pairs": [{"cos": v, "a": ids[a], "b": ids[b],
                              "a_text": meta[ids[a]]["modal_text"],
                              "b_text": meta[ids[b]]["modal_text"]} for v, a, b in pairs[:TOP]],
               "spread": spread_out},
              open(f"{OUT}/cluster_adjacency_{tag}.json", "w"), indent=2, default=str)
    print(f"\nwrote {OUT}/cluster_adjacency_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
