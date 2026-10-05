#!/usr/bin/env python3
"""
Do metadata embeddings agree with MicrobeAtlas community clusters?

The clusters come from 16S community composition - the sequence data - so they
are the first criterion in this project that neither annotation run was fit to.
gold_dict was neutral too but tiny (820 usable); this is ~800k samples.

For Dany (GPT-3.5 -> 3-small 1536d) and new (GPT-5 -> 3-large 1024d), on the
same samples, at two cluster granularities, for sub-biomes and keywords:

    separation AUC    P(cosine of a same-cluster pair > cosine of a random pair)
    purity@k          of a sample's k nearest neighbours, the fraction sharing
                      its cluster - what clustering actually depends on
    exact-text purity P(same cluster | identical text string). Samples sharing a
                      string have IDENTICAL vectors, so no embedder can separate
                      them. This is the signal available from string matching
                      alone: where purity@k barely beats it, the embedding is
                      adding nothing and the ceiling is the text, not the model.

Sampling is stratified by CLUSTER, not uniform: with ~30k clusters in the
overlap a uniform draw leaves ~2 samples per cluster and purity@k is floored
near zero by construction, telling you nothing about the embedding.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_agreement.py
"""
import json, os, sys
from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_to_previous_embeddings import fetch, rows_for, texts_for, unit, cosines, stream

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
OLD, NEW = f"{ROOT}/sidequest", f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"

PER_CLUSTER, MAX_CLUSTERS = 40, 400      # -> 16,000 samples per granularity
N_SAME_PAIRS, KS, SEED = 150_000, (10, 20), 42


def old_h5(t):
    return f"{OLD}/GPT_{t}_embeddings{'_aligned' if t == 'sub_biomes' else ''}.h5"


def new_h5(t):
    return f"{NEW}/embeddings/GPT_{t}_unique_embeddings__text-embedding-3-large__dim1024__full.h5"


def load_join(tag):
    out = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            out[s] = c
    return out


def ids_in(path):
    return {line.split("\t", 1)[0] for line in open(path, encoding="utf-8", errors="replace") if line.strip()}


def auc(same, diff):
    """P(a same-cluster pair scores higher than a different-cluster pair)."""
    r = rankdata(np.concatenate([same, diff]))
    n1, n2 = len(same), len(diff)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def pair_purity(groups):
    """(P(same cluster | same group), pairs) over a dict group -> [cluster_id]."""
    same = tot = 0
    for v in groups.values():
        n = len(v)
        if n < 2:
            continue
        same += sum(c * (c - 1) // 2 for c in Counter(v).values())
        tot += n * (n - 1) // 2
    return (same / tot if tot else None), tot


def choose(head2head, join, rng):
    """Up to MAX_CLUSTERS clusters with enough samples, PER_CLUSTER samples each."""
    by_cluster = defaultdict(list)
    for s in head2head:
        by_cluster[join[s]].append(s)
    big = sorted((c for c, v in by_cluster.items() if len(v) >= PER_CLUSTER))
    if len(big) > MAX_CLUSTERS:
        big = [big[i] for i in sorted(rng.choice(len(big), MAX_CLUSTERS, replace=False))]
    ids, labels = [], []
    for c in big:
        v = sorted(by_cluster[c])
        pick = [v[i] for i in sorted(rng.choice(len(v), PER_CLUSTER, replace=False))]
        ids += pick
        labels += [c] * PER_CLUSTER
    return ids, np.array(labels), len(by_cluster), len(big)


def metrics(vec, labels, rng):
    """Separation AUC + purity@k, for one run's vectors on one cluster labelling."""
    n = len(vec)
    by = defaultdict(list)
    for i, c in enumerate(labels):
        by[c].append(i)
    members = [np.array(v) for v in by.values() if len(v) > 1]

    si, sj = [], []
    per = max(1, N_SAME_PAIRS // len(members))
    for m in members:
        a = m[rng.integers(0, len(m), per)]
        b = m[rng.integers(0, len(m), per)]
        keep = a != b
        si.append(a[keep]); sj.append(b[keep])
    si, sj = np.concatenate(si), np.concatenate(sj)
    di = rng.integers(0, n, len(si)); dj = rng.integers(0, n, len(si))
    keep = labels[di] != labels[dj]
    di, dj = di[keep], dj[keep]

    same, diff = cosines(vec, si, sj), cosines(vec, di, dj)
    out = {"auc": auc(same, diff), "cos_same": float(same.mean()), "cos_diff": float(diff.mean()),
           "n_same_pairs": int(len(same)), "n_diff_pairs": int(len(diff))}

    q = rng.choice(n, min(2000, n), replace=False)
    hits = {k: [] for k in KS}
    for s in range(0, len(q), 200):                      # blocked: n x n would not fit
        blk = q[s:s + 200]
        sims = vec[blk] @ vec.T
        sims[np.arange(len(blk)), blk] = -np.inf
        for k in KS:
            nb = np.argpartition(-sims, k, axis=1)[:, :k]
            hits[k].append((labels[nb] == labels[blk][:, None]).mean(axis=1))
    for k in KS:
        out[f"purity@{k}"] = float(np.concatenate(hits[k]).mean())
    out["baseline"] = float(sum(len(v) * (len(v) - 1) for v in by.values()) / (n * (n - 1)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    joins = {t: load_join(t) for t in ("fine", "coarse")}
    summary = {}

    for target in ("sub_biomes", "keywords"):
        is_kw = target == "keywords"
        print(f"\n{'=' * 78}\n{target}\n{'=' * 78}", flush=True)
        cand = ids_in(f"{OLD}/GPT_{target}.txt")
        head = {s for s in ids_in(f"{NEW}/GPT_{target}.txt") if s in cand}
        del cand
        print(f"samples with text in both runs: {len(head):,}", flush=True)

        chosen, wanted = {}, set()
        for tag, join in joins.items():
            h2h = [s for s in head if s in join]
            ids, labels, n_cl, n_used = choose(h2h, join, rng)
            chosen[tag] = (ids, labels)
            wanted |= set(ids)
            print(f"  {tag}: {len(h2h):,} head-to-head samples in {n_cl:,} clusters -> "
                  f"{len(ids):,} samples from {n_used} clusters of {PER_CLUSTER}", flush=True)

            # ceiling on the FULL head-to-head set, not the subset - far better powered
            for run, path in [("dany", f"{OLD}/GPT_{target}.txt"), ("new", f"{NEW}/GPT_{target}.txt")]:
                g = defaultdict(list)
                for s, txt in stream(path, is_kw):
                    if s in join and s in head:
                        g[hash(txt)].append(join[s])
                p, npairs = pair_purity(g)
                base = pair_purity({0: [join[s] for s in h2h]})[0]
                summary[f"{target}|{tag}|{run}|exact_text"] = {
                    "purity": p, "pairs": npairs, "distinct_texts": len(g),
                    "samples_with_a_twin": sum(len(v) for v in g.values() if len(v) > 1),
                    "base_rate": base, "lift": p / base if p and base else None}
                print(f"    exact-text purity ({run}): {p:.3f} over {npairs:,} pairs, "
                      f"{len(g):,} distinct texts | random-pair base {base:.4f} "
                      f"({p / base:.0f}x)", flush=True)

        old_txt = texts_for(f"{OLD}/GPT_{target}.txt", wanted, is_kw)
        new_txt = texts_for(f"{NEW}/GPT_{target}.txt", wanted, is_kw)
        old_row = rows_for(old_h5(target), "sample_ids", wanted)
        text_row = rows_for(new_h5(target), "texts", set(new_txt.values()))
        print(f"  loaded {len(old_row):,} old rows, {len(text_row):,} new text rows", flush=True)

        for tag, (ids, labels) in chosen.items():
            ok = [i for i, s in enumerate(ids) if s in old_row and new_txt.get(s) in text_row]
            ids2, lab2 = [ids[i] for i in ok], labels[ok]
            old_v = fetch(old_h5(target), [old_row[s] for s in ids2])
            new_v = fetch(new_h5(target), [text_row[new_txt[s]] for s in ids2])
            fin = np.isfinite(old_v).all(axis=1) & np.isfinite(new_v).all(axis=1)
            old_v, new_v, lab2 = unit(old_v[fin]), unit(new_v[fin]), lab2[fin]
            print(f"\n  {tag}: {len(lab2):,} samples, {len(set(lab2))} clusters "
                  f"({int((~fin).sum())} dropped for a non-finite vector)", flush=True)
            for run, vec in [("dany", old_v), ("new", new_v)]:
                m = metrics(vec, lab2, np.random.default_rng(SEED))
                summary[f"{target}|{tag}|{run}|embedding"] = m
                print(f"    {run:<5} AUC {m['auc']:.3f} | purity@10 {m['purity@10']:.3f} "
                      f"purity@20 {m['purity@20']:.3f} | baseline {m['baseline']:.4f} "
                      f"| cos same {m['cos_same']:.3f} vs diff {m['cos_diff']:.3f}", flush=True)
            del old_v, new_v
        json.dump(summary, open(f"{OUT}/cluster_agreement.json", "w"), indent=2, default=str)

    print(f"\nwrote {OUT}/cluster_agreement.json", flush=True)


if __name__ == "__main__":
    main()
