#!/usr/bin/env python3
"""
Full-scale neighbour agreement between Dany's embeddings and the new ones.

For each point, take its k nearest neighbours in Dany's space and in the new
space, and report the fraction shared. Unlike a global correlation this is a
LOCAL measure, which is what clustering actually depends on.

Run over the whole overlap rather than a 15k subsample, because "top-k among
1.6M candidates" is a different and much harder question than "top-k among 15k".

POINTS ARE DISTINCT (old text, new text) PAIRS, not samples. At sample level the
new sub-biome space has 1,641,958 samples spread over only 19,345 distinct
vectors - 85 samples share each one - so every top-20 list would be exact ties
broken arbitrarily, measuring nothing. Deduplicating on the text pair removes
points that carry no geometric information. Residual within-space ties are
reported alongside, so the numbers can be read honestly.

Memory-bounded: candidates are streamed from disk in blocks and only a running
top-k is kept, so this needs ~1 GB regardless of corpus size.

    python3 scripts/neighbour_overlap_full.py --n_queries 1000 --k 10 20 50
"""

import argparse
import json
import os
import numpy as np

from compare_to_previous_embeddings import (fetch, new_unique, old_embeddings, rows_for,
                                            stream, unit)
import compare_to_previous_embeddings as C

COLORS = {"Dany": "#2a78d6", "new": "#eb6834", "shared": "#1baf7a"}


def build_points(target, is_keywords):
    """-> (old_rows, new_rows) for each distinct (old text, new text) pair."""
    old = {sid: text for sid, text in stream(f"{C.OLD}/GPT_{target}.txt", is_keywords)}
    pairs = {}
    for sid, new_text in stream(f"{C.NEW}/GPT_{target}.txt", is_keywords):
        old_text = old.get(sid)
        if old_text is not None:
            pairs.setdefault((old_text, new_text), sid)
    print(f"  {len(pairs):,} distinct (old,new) text pairs", flush=True)

    sids = set(pairs.values())
    old_row = rows_for(old_embeddings(target), "sample_ids", sids)
    text_row = rows_for(new_unique(target), "texts", {t for _, t in pairs})
    keep = [(old_row[s], text_row[n]) for (o, n), s in pairs.items()
            if s in old_row and n in text_row]
    same = np.array([o == n for (o, n) in pairs if pairs[(o, n)] in old_row and n in text_row])
    rows = np.array(keep)
    print(f"  {len(rows):,} usable points ({int(same.sum()):,} with unchanged text)", flush=True)
    return rows[:, 0], rows[:, 1], same


def top_k(h5_path, rows, queries, k, block=20_000):
    """Top-k neighbours of each query among all rows, streaming the corpus.

    Reads the dataset in CONTIGUOUS slices and picks the wanted rows out of each
    slice, rather than fetching scattered rows. Dany's keywords file is chunked
    (313, 48) so one row spans 32 chunks: scattered reads cost ~1.2 ms/row while
    a sequential scan of the whole 12.6 GB runs at 1.4 GB/s - measured 96x
    faster, even though it reads rows we do not need.

    Only a running (n_queries, k) frontier is kept, so memory is independent of
    corpus size."""
    import time
    q_vec = unit(fetch(h5_path, rows[queries].tolist()).astype(np.float32))
    best_s = np.full((len(queries), k), -np.inf, np.float32)
    best_i = np.full((len(queries), k), -1, np.int64)

    order = np.argsort(rows)                 # point indices, ordered by dataset row
    sorted_rows = rows[order]
    t0 = time.time()
    with __import__("h5py").File(h5_path, "r") as f:
        dset = f["embeddings"]
        n_rows = dset.shape[0]
        for bi, start in enumerate(range(0, n_rows, block)):
            lo, hi = np.searchsorted(sorted_rows, [start, start + block])
            if hi == lo:
                continue
            if bi % 20 == 0:
                print(f"      row {start:,}/{n_rows:,}  ({time.time() - t0:.0f}s)", flush=True)
            slab = dset[start:start + block]
            cand = unit(slab[sorted_rows[lo:hi] - start].astype(np.float32))
            idx = order[lo:hi]
            sims = q_vec @ cand.T
            for qi, gi in enumerate(queries):     # never be your own neighbour
                local = np.flatnonzero(idx == gi)
                if len(local):
                    sims[qi, local[0]] = -np.inf
            kb = min(k, sims.shape[1])
            part = np.argpartition(-sims, kb - 1, axis=1)[:, :kb]
            merged_s = np.concatenate([best_s, np.take_along_axis(sims, part, 1)], axis=1)
            merged_i = np.concatenate([best_i, idx[part]], axis=1)
            pick = np.argpartition(-merged_s, k - 1, axis=1)[:, :k]
            best_s = np.take_along_axis(merged_s, pick, 1)
            best_i = np.take_along_axis(merged_i, pick, 1)
            del slab, cand, sims

    ties = sum(int((row == row.max()).sum() > 1) for row in best_s) / len(queries)
    return best_i, ties


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--targets", nargs="+", default=["sub_biomes", "keywords"])
    p.add_argument("--n_queries", type=int, default=1000)
    p.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--block", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    root = os.path.expanduser(args.root)
    C.OLD, C.NEW = f"{root}/sidequest", f"{root}/sidequest/latest"
    out_dir = args.output_dir or f"{C.NEW}/embeddings/vs_previous"
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for target in args.targets:
        print(f"\n{'=' * 66}\n{target}\n{'=' * 66}", flush=True)
        old_rows, new_rows, same_text = build_points(target, target == "keywords")
        n = len(old_rows)
        queries = np.sort(np.random.default_rng(args.seed).choice(n, min(args.n_queries, n), replace=False))
        kmax = max(args.k)

        print(f"  scanning Dany's space ({n:,} candidates) ...", flush=True)
        d_idx, d_ties = top_k(old_embeddings(target), old_rows, queries, kmax, args.block)
        print(f"  scanning the new space ...", flush=True)
        n_idx, n_ties = top_k(new_unique(target), new_rows, queries, kmax, args.block)

        summary[target] = {"n_points": int(n), "n_queries": int(len(queries)),
                           "tied_frontier_dany": d_ties, "tied_frontier_new": n_ties, "overlap": {}}
        for k in args.k:
            shared = [len(set(a[:k]) & set(b[:k])) / k for a, b in zip(d_idx, n_idx)]
            # same-text points isolate the embedding model; changed-text adds the LLM
            by_split = {}
            for label, mask in [("same text", same_text[queries]), ("changed text", ~same_text[queries])]:
                if mask.sum() >= 30:
                    by_split[label] = float(np.mean(np.array(shared)[mask]))
            summary[target]["overlap"][k] = {"all": float(np.mean(shared)), **by_split}
            extra = "  ".join(f"{lab} {v:.3f}" for lab, v in by_split.items())
            print(f"    overlap@{k:<3} {np.mean(shared):.3f}   {extra}", flush=True)
        print(f"    tied top-k frontier: Dany {d_ties:.1%} of queries, new {n_ties:.1%}", flush=True)

    path = os.path.join(out_dir, "neighbour_overlap_full.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\nwritten: {path}", flush=True)


if __name__ == "__main__":
    main()
