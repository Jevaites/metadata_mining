#!/usr/bin/env python3
"""
Which community clusters does the metadata describe well, and which clusters
look alike in metadata space?

Uses the best configuration from cluster_agreement.py: new-run KEYWORDS embedded
with 3-large@1024 (purity@10 0.82 vs 0.47 for sub-biomes). No head-to-head
restriction is needed here - we are not comparing runs - so this runs on all
1.53M clustered samples that have keyword text, not the 734k overlap.

Per cluster, from M sampled members with unit vectors u_i:

    coherence  = mean pairwise cosine inside the cluster, = (||S||^2 - m) / m(m-1)
                 with S = sum u_i. Exact and streamable, no m x m matrix.
    modal share= fraction of ALL members carrying the single commonest string
    centroid   = S / ||S||, used for the cluster x cluster adjacency

Reads the unique-embedding table SEQUENTIALLY in slabs rather than fetching
scattered rows: that file is chunked (128, 64), and scattered reads measured ~96x
slower than sequential on this data.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_structure.py [fine]
"""
import json, os, sys
from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"
UNIQ = f"{NEW}/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5"

M, MIN_MEMBERS, SEED = 20, 20, 42       # members sampled per cluster / minimum to include
SLAB = 50_000


def rss():
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB"


def load_join(tag):
    out = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            out[s] = c
    return out


def scan(join, path, is_kw):
    """One pass: per-cluster text-hash counts and a reservoir of M members."""
    rng = np.random.default_rng(SEED)
    counts, seen, res = defaultdict(Counter), Counter(), defaultdict(list)
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        c = join.get(sid)
        if c is None or not raw.strip():
            continue
        counts[c][hash(clean_text(raw, is_kw))] += 1
        seen[c] += 1
        r = res[c]
        if len(r) < M:                                  # reservoir sampling
            r.append(sid)
        else:
            j = rng.integers(0, seen[c])
            if j < M:
                r[j] = sid
    return counts, seen, res


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "coarse"
    os.makedirs(OUT, exist_ok=True)
    join = load_join(tag)
    print(f"{tag}: {len(join):,} clustered samples", flush=True)

    counts, seen, res = scan(join, f"{NEW}/GPT_keywords.txt", True)
    keep = sorted(c for c in seen if seen[c] >= MIN_MEMBERS and len(res[c]) >= MIN_MEMBERS)
    print(f"  {len(seen):,} clusters with keyword text, {len(keep):,} with >= {MIN_MEMBERS} members "
          f"[rss {rss()}]", flush=True)

    # sampled members -> their text; then text -> row in the unique table
    wanted_ids = {s: c for c in keep for s in res[c]}
    txt = {}
    for line in open(f"{NEW}/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid in wanted_ids and raw.strip():
            txt[sid] = clean_text(raw, True)
    need = set(txt.values())
    print(f"  {len(wanted_ids):,} sampled members, {len(need):,} distinct texts [rss {rss()}]", flush=True)

    row_of = {}
    with h5py.File(UNIQ, "r") as f:
        d = f["texts"]
        for i in range(0, d.shape[0], 200_000):
            for k, v in enumerate(d[i:i + 200_000]):
                v = v.decode()
                if v in need:
                    row_of[v] = i + k
    print(f"  resolved {len(row_of):,} of {len(need):,} texts to rows [rss {rss()}]", flush=True)

    # cluster index, and which rows each cluster needs
    idx = {c: i for i, c in enumerate(keep)}
    # two parallel accumulations: every sampled member, and one per DISTINCT text.
    # A cluster where all members carry the same string has coherence 1.0 trivially
    # (identical vectors); the distinct-text version asks whether the different
    # things its members say are still close.
    rows, owners, first = [], [], []
    for c in keep:
        got = set()
        for s in res[c]:
            r = row_of.get(txt.get(s))
            if r is None:
                continue
            rows.append(r); owners.append(idx[c]); first.append(r not in got)
            got.add(r)
    rows, owners, first = np.array(rows), np.array(owners), np.array(first)
    order = np.argsort(rows)
    rows, owners, first = rows[order], owners[order], first[order]

    with h5py.File(UNIQ, "r") as f:
        d = f["embeddings"]
        dim, n_rows = d.shape[1], d.shape[0]
        S = np.zeros((len(keep), dim), dtype=np.float64)
        U = np.zeros((len(keep), dim), dtype=np.float64)
        n = np.zeros(len(keep), dtype=np.int64)
        nu = np.zeros(len(keep), dtype=np.int64)
        for start in range(0, n_rows, SLAB):            # sequential slabs, not scattered reads
            lo, hi = np.searchsorted(rows, [start, start + SLAB])
            if hi == lo:
                continue
            slab = d[start:start + SLAB]
            v = slab[rows[lo:hi] - start].astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            np.add.at(S, owners[lo:hi], v)
            np.add.at(n, owners[lo:hi], 1)
            f = first[lo:hi]
            np.add.at(U, owners[lo:hi][f], v[f])
            np.add.at(nu, owners[lo:hi][f], 1)
            if start % (SLAB * 8) == 0:
                print(f"    slab {start:>9,}/{n_rows:,} [rss {rss()}]", flush=True)

    ok = n >= MIN_MEMBERS
    print(f"  centroids for {int(ok.sum()):,} clusters [rss {rss()}]", flush=True)
    norm2 = (S ** 2).sum(axis=1)
    coherence = np.where(ok, (norm2 - n) / np.maximum(n * (n - 1), 1), np.nan)
    unorm2 = (U ** 2).sum(axis=1)
    coh_uniq = np.where(nu >= 3, (unorm2 - nu) / np.maximum(nu * (nu - 1), 1), np.nan)
    cent = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)

    # global reference: mean cosine between two random sampled members
    G = S.sum(axis=0); N = n.sum()
    global_cos = float((G @ G - N) / (N * (N - 1)))
    print(f"  mean cosine between two random samples: {global_cos:.3f}", flush=True)

    # resolve each cluster's modal text
    modal_hash = {c: counts[c].most_common(1)[0] for c in keep}
    want_hash = {h for h, _ in modal_hash.values()}
    hash_txt = {}
    for line in open(f"{NEW}/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if join.get(sid) and raw.strip():
            t = clean_text(raw, True)
            h = hash(t)
            if h in want_hash and h not in hash_txt:
                hash_txt[h] = t

    recs = []
    for c in keep:
        i = idx[c]
        if not ok[i]:
            continue
        h, cnt = modal_hash[c]
        recs.append({"cluster": c, "members": int(seen[c]), "sampled": int(n[i]),
                     "coherence": float(coherence[i]), "lift": float(coherence[i] - global_cos),
                     "distinct_texts": len(counts[c]),
                     "texts_per_member": len(counts[c]) / seen[c],
                     "modal_share": cnt / seen[c], "modal_text": hash_txt.get(h, "?"),
                     "coherence_distinct_texts": float(coh_uniq[i]),
                     "n_distinct_sampled": int(nu[i])})
    recs.sort(key=lambda r: -r["coherence"])
    json.dump({"tag": tag, "global_cos": global_cos, "clusters": recs},
              open(f"{OUT}/cluster_coherence_{tag}.json", "w"), indent=2, default=str)
    np.savez_compressed(f"{OUT}/cluster_centroids_{tag}.npz",
                        centroids=cent[ok].astype(np.float32),
                        cluster=np.array([c for c in keep if ok[idx[c]]]))

    co = np.array([r["coherence"] for r in recs])
    cu = np.array([r["coherence_distinct_texts"] for r in recs], dtype=float)
    ms = np.array([r["modal_share"] for r in recs])
    sz = np.array([r["members"] for r in recs], dtype=float)
    fin = np.isfinite(cu)
    print(f"\n  coherence, all sampled members : median {np.median(co):.3f}  "
          f"p10 {np.percentile(co, 10):.3f}  p90 {np.percentile(co, 90):.3f}", flush=True)
    print(f"  coherence, distinct texts only : median {np.median(cu[fin]):.3f}  "
          f"p10 {np.percentile(cu[fin], 10):.3f}  p90 {np.percentile(cu[fin], 90):.3f}  "
          f"(n={int(fin.sum()):,})", flush=True)
    print(f"  random-pair reference {global_cos:.3f}", flush=True)
    print(f"  Spearman(coherence, cluster size)  {spearmanr(co, sz)[0]:+.3f}"
          f"   Spearman(coherence, modal share) {spearmanr(co, ms)[0]:+.3f}", flush=True)
    print(f"  Spearman(distinct-text coherence, size) {spearmanr(cu[fin], sz[fin])[0]:+.3f}"
          f"   vs modal share {spearmanr(cu[fin], ms[fin])[0]:+.3f}", flush=True)
    print(f"\n  MOST coherent clusters")
    print(f"    {'cluster':>8} {'members':>8} {'coh':>6} {'coh-uq':>7} {'modal':>6}  modal keyword string")
    for r in recs[:10]:
        print(f"    {r['cluster']:>8} {r['members']:>8,} {r['coherence']:>6.3f} "
              f"{r['coherence_distinct_texts']:>7.3f} {r['modal_share']:>6.0%}  {r['modal_text'][:64]}")
    print(f"\n  LEAST coherent clusters")
    for r in recs[-10:]:
        print(f"    {r['cluster']:>8} {r['members']:>8,} {r['coherence']:>6.3f} "
              f"{r['coherence_distinct_texts']:>7.3f} {r['modal_share']:>6.0%}  {r['modal_text'][:64]}")
    print(f"\nwrote {OUT}/cluster_coherence_{tag}.json and cluster_centroids_{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
