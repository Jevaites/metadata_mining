#!/usr/bin/env python3
"""
Print a few rows of any embeddings .h5 in this project, to eyeball the data.

Handles whatever columns a file happens to have (the older
GPT_sub_biomes_keywords_embeddings.h5 uses sub_texts/key_texts instead of
texts), and only ever reads the rows it prints - safe on a 14 GB file.

    python3 scripts/peek_h5.py                     # the default set of files
    python3 scripts/peek_h5.py a.h5 b.h5 --rows 5 --values 8
    python3 scripts/peek_h5.py a.h5 --random       # random rows instead of the head
"""

import argparse
import os
import random

import h5py
import numpy as np

BASE = os.path.expanduser("~/MicrobeAtlasProject/sidequest")
DEFAULTS = [
    f"{BASE}/GPT_sub_biomes_embeddings_aligned.h5",                                                    # old
    f"{BASE}/GPT_keywords_embeddings.h5",                                                              # old
    f"{BASE}/latest/embeddings/GPT_sub_biomes_embeddings__text-embedding-3-large__dim1024__full.h5",   # new
    f"{BASE}/latest/embeddings/GPT_keywords_embeddings__text-embedding-3-large__dim1024__full.h5",
    f"{BASE}/latest/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5",
]


def peek(path, n_rows, n_values, use_random, seed, text_width):
    print("\n" + "=" * 100)
    print(path)
    if not os.path.exists(path):
        print("  (missing)")
        return

    with h5py.File(path, "r") as f:
        columns = list(f)
        n_total, dim = f["embeddings"].shape
        size_gb = os.path.getsize(path) / 1024 ** 3
        print(f"{n_total:,} rows x {dim} dims ({f['embeddings'].dtype}) | columns: {columns} | {size_gb:.2f} GB")

        rows = (sorted(random.Random(seed).sample(range(n_total), min(n_rows, n_total)))
                if use_random else list(range(min(n_rows, n_total))))
        text_columns = [c for c in columns if c != "embeddings"]

        for i in rows:
            print(f"\n  [row {i}]")
            for column in text_columns:
                value = f[column][i]
                value = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                if len(value) > text_width:
                    value = value[:text_width] + f"... ({len(value)} chars)"
                print(f"    {column:<12} {value}")
            vector = f["embeddings"][i]
            head = ", ".join(f"{v:+.5f}" for v in vector[:n_values])
            print(f"    {'embedding':<12} [{head}, ...]  norm={np.linalg.norm(vector):.6f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", default=DEFAULTS, help="Files to peek at (default: the main ones).")
    p.add_argument("--rows", type=int, default=3, help="Rows to print per file.")
    p.add_argument("--values", type=int, default=6, help="Embedding components to show per row.")
    p.add_argument("--random", action="store_true", help="Random rows instead of the first ones.")
    p.add_argument("--text_width", type=int, default=100, help="Truncate text columns to this many chars.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for path in (args.paths or DEFAULTS):
        peek(os.path.expanduser(path), args.rows, args.values, args.random, args.seed, args.text_width)


if __name__ == "__main__":
    main()
