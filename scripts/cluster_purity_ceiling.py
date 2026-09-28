#!/usr/bin/env python3
"""
Is the sub-biome purity@k measuring the embedding, or just tie-breaking?

cluster_agreement.py found new > Dany on separation AUC everywhere, but new
WORSE than Dany on purity@10 for sub-biomes while BETTER for keywords. The
suspect is degeneracy: samples sharing a sub-biome string have identical
vectors, so if a sample has >= k exact duplicates its top-k is an arbitrary
draw from that tie group and purity@k just reports the tie group's purity.

Replays cluster_agreement.py's exact subsets (same seed, same call order) and:

    tie size          how many samples share each sample's exact string, in-subset
    ceiling           exact-text purity ON THE SUBSET, so it is directly
                      comparable to purity@k (the run in cluster_agreement.py
                      computed it on the full 785k set, base rate 0.0001 vs
                      0.0024 here - not comparable, do not put them in one table)
    purity@10 by tie  split into samples with < 10 duplicates (the embedding is
                      really ranking) and >= 10 (it is not)

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/cluster_purity_ceiling.py
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_agreement as A
from compare_to_previous_embeddings import fetch, rows_for, texts_for, unit

K = 10


def tie_purity(texts, labels):
    """Purity of the exact-string partition, on these samples only."""
    g = defaultdict(list)
    for t, c in zip(texts, labels):
        g[t].append(c)
    return A.pair_purity(g), {t: len(v) for t, v in g.items()}


def main():
    os.makedirs(A.OUT, exist_ok=True)
    rng = np.random.default_rng(A.SEED)
    joins = {t: A.load_join(t) for t in ("fine", "coarse")}
    out = {}

    for target in ("sub_biomes", "keywords"):
        is_kw = target == "keywords"
        print(f"\n{'=' * 78}\n{target}\n{'=' * 78}", flush=True)
        cand = A.ids_in(f"{A.OLD}/GPT_{target}.txt")
        head = {s for s in A.ids_in(f"{A.NEW}/GPT_{target}.txt") if s in cand}
        del cand

        chosen, wanted = {}, set()
        for tag, join in joins.items():                       # same order as the main script
            h2h = [s for s in head if s in join]
            ids, labels, _, _ = A.choose(h2h, join, rng)
            chosen[tag] = (ids, labels)
            wanted |= set(ids)

        old_txt = texts_for(f"{A.OLD}/GPT_{target}.txt", wanted, is_kw)
        new_txt = texts_for(f"{A.NEW}/GPT_{target}.txt", wanted, is_kw)
        old_row = rows_for(A.old_h5(target), "sample_ids", wanted)
        text_row = rows_for(A.new_h5(target), "texts", set(new_txt.values()))

        for tag, (ids, labels) in chosen.items():
            ok = [i for i, s in enumerate(ids) if s in old_row and new_txt.get(s) in text_row]
            ids2, lab2 = [ids[i] for i in ok], labels[ok]
            old_v = fetch(A.old_h5(target), [old_row[s] for s in ids2])
            new_v = fetch(A.new_h5(target), [text_row[new_txt[s]] for s in ids2])
            fin = np.isfinite(old_v).all(axis=1) & np.isfinite(new_v).all(axis=1)
            ids2 = [s for s, g in zip(ids2, fin) if g]
            old_v, new_v, lab2 = unit(old_v[fin]), unit(new_v[fin]), lab2[fin]
            n = len(lab2)
            base = float(sum(v * (v - 1) for v in Counter(lab2).values()) / (n * (n - 1)))
            print(f"\n  {tag}: {n:,} samples, {len(set(lab2))} clusters, base rate {base:.4f}", flush=True)

            for run, vec, txt in [("dany", old_v, old_txt), ("new", new_v, new_txt)]:
                t = [txt[s] for s in ids2]
                (ceil, cpairs), sizes = tie_purity(t, lab2)
                tie = np.array([sizes[x] - 1 for x in t])     # duplicates, excluding self

                q = np.arange(n)
                hits = []
                for s in range(0, n, 200):
                    blk = q[s:s + 200]
                    sims = vec[blk] @ vec.T
                    sims[np.arange(len(blk)), blk] = -np.inf
                    nb = np.argpartition(-sims, K, axis=1)[:, :K]
                    hits.append((lab2[nb] == lab2[blk][:, None]).mean(axis=1))
                hits = np.concatenate(hits)

                free, stuck = tie < K, tie >= K
                row = {"distinct_texts": len(sizes), "ceiling_exact_text": ceil,
                       "ceiling_pairs": cpairs, "base_rate": base,
                       "median_duplicates": int(np.median(tie)),
                       "frac_with_ge_k_duplicates": float(stuck.mean()),
                       "purity@10_all": float(hits.mean()),
                       "purity@10_tie_free": float(hits[free].mean()) if free.any() else None,
                       "purity@10_tie_bound": float(hits[stuck].mean()) if stuck.any() else None,
                       "n_tie_free": int(free.sum())}
                out[f"{target}|{tag}|{run}"] = row
                print(f"    {run:<5} {len(sizes):>6,} distinct texts | median dups {row['median_duplicates']:>4}"
                      f" | >={K} dups: {stuck.mean():5.1%}", flush=True)
                print(f"          ceiling (exact-text, same subset) {ceil:.3f}"
                      f" | purity@10 all {hits.mean():.3f}"
                      f" | tie-free {row['purity@10_tie_free'] if free.any() else float('nan'):.3f}"
                      f" (n={int(free.sum()):,})"
                      f" | tie-bound {row['purity@10_tie_bound'] if stuck.any() else float('nan'):.3f}", flush=True)
            del old_v, new_v
        json.dump(out, open(f"{A.OUT}/cluster_purity_ceiling.json", "w"), indent=2, default=str)
    print(f"\nwrote {A.OUT}/cluster_purity_ceiling.json", flush=True)


if __name__ == "__main__":
    main()
