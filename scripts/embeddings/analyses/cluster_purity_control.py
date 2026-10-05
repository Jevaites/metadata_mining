#!/usr/bin/env python3
"""
Control for two problems in the first cluster-agreement run.

1. TIE-BREAKING BY ARRAY POSITION. choose() builds the sample list cluster by
   cluster, so same-cluster samples are CONTIGUOUS. When many vectors are exactly
   identical (83% of new sub-biome samples have >= 10 duplicates), every tied
   candidate has the same similarity and np.argpartition's choice among them is
   arbitrary - if it correlates with array position it silently returns the
   query's cluster-mates and inflates purity. Symptom: purity@10 = 0.48 against
   an exact-text ceiling of 0.12, which is impossible if the top-10 really is a
   fair draw from the tie group. Test: recompute under a random permutation.

2. UNPAIRED TIE-FREE COMPARISON. Comparing Dany's tie-free samples to new's
   compares different populations (n=6,679 vs 2,592). Restrict to samples that
   are tie-free in BOTH runs and compare on those.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_purity_control.py
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_agreement as A
from compare_to_previous_embeddings import fetch, rows_for, texts_for, unit

def rss():
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB"


K = 10
ONLY = os.environ.get("ONLY")          # e.g. "keywords:coarse" - replays the rng, skips other cells


def purity_at_k(vec, lab, k=K):
    """Per-sample fraction of the k nearest neighbours sharing the cluster."""
    hits = []
    for s in range(0, len(lab), 200):
        blk = np.arange(s, min(s + 200, len(lab)))
        sims = vec[blk] @ vec.T
        sims[np.arange(len(blk)), blk] = -np.inf
        nb = np.argpartition(-sims, k, axis=1)[:, :k]
        hits.append((lab[nb] == lab[blk][:, None]).mean(axis=1))
    return np.concatenate(hits)


def main():
    rng = np.random.default_rng(A.SEED)
    joins = {t: A.load_join(t) for t in ("fine", "coarse")}
    out = {}

    for target in ("sub_biomes", "keywords"):
        is_kw = target == "keywords"
        print(f"\n{'=' * 78}\n{target}\n{'=' * 78}", flush=True)
        cand = A.ids_in(f"{A.OLD}/GPT_{target}.txt")
        head = set()                                   # stream, never hold the 3.4M new id set
        with open(f"{A.NEW}/GPT_{target}.txt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.split("\t", 1)[0]
                if s in cand:
                    head.add(s)
        del cand
        print(f"  head-to-head {len(head):,}  [rss {rss()}]", flush=True)

        chosen, wanted = {}, set()
        for tag, join in joins.items():
            h2h = [s for s in head if s in join]
            ids, labels, _, _ = A.choose(h2h, join, rng)
            chosen[tag] = (ids, labels)
            wanted |= set(ids)

        print(f"  chosen {len(wanted):,} samples  [rss {rss()}]", flush=True)
        old_txt = texts_for(f"{A.OLD}/GPT_{target}.txt", wanted, is_kw)
        new_txt = texts_for(f"{A.NEW}/GPT_{target}.txt", wanted, is_kw)
        old_row = rows_for(A.old_h5(target), "sample_ids", wanted)
        print(f"  texts loaded  [rss {rss()}]", flush=True)
        old_row = old_row if False else old_row
        text_row = rows_for(A.new_h5(target), "texts", set(new_txt.values()))
        print(f"  row maps loaded: {len(old_row):,} old, {len(text_row):,} new texts  [rss {rss()}]", flush=True)

        for tag, (ids, labels) in chosen.items():
            if ONLY and ONLY != f"{target}:{tag}":
                continue
            ok = [i for i, s in enumerate(ids) if s in old_row and new_txt.get(s) in text_row]
            ids2, lab2 = [ids[i] for i in ok], labels[ok]
            old_v = fetch(A.old_h5(target), [old_row[s] for s in ids2])
            new_v = fetch(A.new_h5(target), [text_row[new_txt[s]] for s in ids2])
            fin = np.isfinite(old_v).all(axis=1) & np.isfinite(new_v).all(axis=1)
            ids2 = [s for s, g in zip(ids2, fin) if g]
            old_v, new_v, lab2 = unit(old_v[fin]), unit(new_v[fin]), lab2[fin]
            n = len(lab2)

            perm = np.random.default_rng(7).permutation(n)       # break position/cluster coupling
            print(f"\n  {tag}: {n:,} samples, {len(set(lab2))} clusters", flush=True)

            res, dup = {}, {}
            for run, vec, txt in [("dany", old_v, old_txt), ("new", new_v, new_txt)]:
                t = [txt[s] for s in ids2]
                sizes = Counter(t)
                dup[run] = np.array([sizes[x] - 1 for x in t])
                blocked = purity_at_k(vec, lab2)
                shuffled = purity_at_k(vec[perm], lab2[perm])
                back = np.empty(n); back[perm] = shuffled          # realign to original ids
                res[run] = back
                g = defaultdict(list)
                for x, c in zip(t, lab2):
                    g[x].append(c)
                ceil = A.pair_purity(g)[0]
                print(f"    {run:<5} purity@10  cluster-blocked {blocked.mean():.3f}"
                      f"  ->  shuffled {shuffled.mean():.3f}"
                      f"   (exact-text ceiling {ceil:.3f})", flush=True)
                out[f"{target}|{tag}|{run}"] = {
                    "purity@10_blocked_order": float(blocked.mean()),
                    "purity@10_shuffled_order": float(shuffled.mean()),
                    "ceiling_exact_text": ceil}

            both_free = (dup["dany"] < K) & (dup["new"] < K)
            d, w = res["dany"][both_free], res["new"][both_free]
            print(f"    paired, tie-free in BOTH runs (n={int(both_free.sum()):,}): "
                  f"dany {d.mean():.3f}  new {w.mean():.3f}  "
                  f"diff {w.mean() - d.mean():+.3f}", flush=True)
            if both_free.sum() > 30:
                delta = w - d
                boot = np.array([delta[np.random.default_rng(i).integers(0, len(delta), len(delta))].mean()
                                 for i in range(400)])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                print(f"      95% CI on the difference [{lo:+.3f}, {hi:+.3f}]", flush=True)
                out[f"{target}|{tag}|paired_tie_free"] = {
                    "n": int(both_free.sum()), "dany": float(d.mean()), "new": float(w.mean()),
                    "diff": float(w.mean() - d.mean()), "ci": [float(lo), float(hi)]}
            del old_v, new_v
        json.dump(out, open(f"{A.OUT}/cluster_purity_control{'_' + ONLY.replace(':', '_') if ONLY else ''}.json", "w"), indent=2, default=str)
    print(f"\nwrote {A.OUT}/cluster_purity_control.json", flush=True)


if __name__ == "__main__":
    main()
