#!/usr/bin/env python3
"""
Sanity-check one run of embed_subbiomes_keywords.py.

  1. structure  - shapes, dtype, no NaN/Inf/zero rows, norms ~1
  2. coverage   - every ID in the subset list made it into the per-sample file,
                  and sample_ids are unique
  3. dedup      - samples sharing a text got the same vector (what stage 2
                  promises), on a random sample of rows
  4. reference  - optional: texts that also appear in an older embeddings file
                  should match it almost exactly, since the API is (near-)
                  deterministic for a fixed input string. Only meaningful if the
                  reference used the same model and dimension.

Embeddings are read in chunks and never held in memory whole, so this works on
a full-scale per-sample file (3.4M rows x 1024 dims is ~14 GB on disk).

    python3 scripts/verify_embeddings.py \
        --full_h5   .../GPT_sub_biomes_embeddings__text-embedding-3-large__dim1024__full.h5 \
        --unique_h5 .../GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5

    python3 scripts/verify_embeddings.py \
        --full_h5   .../GPT_sub_biomes_embeddings__text-embedding-3-small__dim1536__perbiome2250_seed42.h5 \
        --subset_ids_file .../subset_ids__perbiome2250_seed42.txt \
        --reference_h5 ~/MicrobeAtlasProject/sidequest/GPT_sub_biomes_embeddings_aligned.h5
"""

import argparse
import random
from collections import defaultdict

import h5py
import numpy as np

CHUNK = 250_000                    # rows per read: ~1 GB at 1024 dims
MAX_REFERENCE_ROWS_PER_TEXT = 5


def decode(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]


def check_structure(f, path):
    print(f"\n=== structure: {path}")
    embeddings = f["embeddings"]
    n_rows, dim = embeddings.shape
    n_bad = n_zero = 0
    lo, hi = np.inf, 0.0
    for i in range(0, n_rows, CHUNK):
        block = embeddings[i:i + CHUNK]
        n_bad += int(np.isnan(block).sum() + np.isinf(block).sum())
        norms = np.linalg.norm(block, axis=1)
        n_zero += int((norms == 0).sum())
        lo, hi = min(lo, norms.min()), max(hi, norms.max())
    print(f"{n_rows} rows x {dim} dims ({embeddings.dtype}), columns {list(f)}")
    print(f"norms: min={lo:.4f} max={hi:.4f}")
    print("OK" if n_bad == 0 and n_zero == 0 else f"!! {n_bad} NaN/Inf values, {n_zero} zero rows")


def check_coverage(f, subset_ids_path):
    print("\n=== coverage")
    present = decode(f["sample_ids"][:])
    unique = set(present)
    print(f"{len(present)} rows, {len(unique)} distinct sample_ids"
          + ("" if len(unique) == len(present) else "  !! duplicated sample_ids"))
    if not subset_ids_path:
        return
    expected = {line.strip() for line in open(subset_ids_path, encoding="utf-8") if line.strip()}
    missing, extra = expected - unique, unique - expected
    print(f"vs {subset_ids_path}: expected {len(expected)}, missing {len(missing)}, unexpected {len(extra)}")
    if missing:
        print(f"  missing are fine if those samples had no usable text, e.g. {sorted(missing)[:5]}")
    if extra:
        print(f"  !! not in the subset list, e.g. {sorted(extra)[:5]}")


def check_dedup(f, n_checks, seed):
    """Sampled, not exhaustive: at full scale ~100 samples share each text, so
    comparing every pair would be ~3.4M reads for no extra information."""
    print("\n=== same text -> same vector")
    texts = decode(f["texts"][:])
    first_row = {}
    for i, text in enumerate(texts):
        first_row.setdefault(text, i)
    shared = len(texts) - len(first_row)
    if not shared:
        print("no text is shared by more than one row, nothing to check")
        return
    rows = sorted(random.Random(seed).sample(range(len(texts)), min(n_checks, len(texts))))
    mismatches = sum(
        not np.array_equal(f["embeddings"][i], f["embeddings"][first_row[texts[i]]]) for i in rows
    )
    print(f"{len(first_row)} distinct texts over {len(texts)} rows; "
          f"{len(rows)} random rows checked against their text's first row; {mismatches} mismatched")


def check_reference(f, reference_path, threshold):
    print(f"\n=== cross-check vs {reference_path}")
    new = {}
    for text, row in zip(decode(f["texts"][:]), range(f["embeddings"].shape[0])):
        new.setdefault(text, row)

    with h5py.File(reference_path, "r") as ref:
        if ref["embeddings"].shape[1] != f["embeddings"].shape[1]:
            print(f"!! reference is {ref['embeddings'].shape[1]}-dim, this run is "
                  f"{f['embeddings'].shape[1]}-dim - not comparable, skipping")
            return
        rows = defaultdict(list)
        for i, text in enumerate(decode(ref["texts"][:])):
            # A popular text can have >100k reference rows; a handful is plenty
            # both to compare against and to spot the reference disagreeing with
            # itself, and it keeps the read below from blowing up memory.
            if text in new and len(rows[text]) < MAX_REFERENCE_ROWS_PER_TEXT:
                rows[text].append(i)
        if not rows:
            print("no overlapping texts")
            return
        wanted = sorted({i for idxs in rows.values() for i in idxs})
        ref_vectors = dict(zip(wanted, ref["embeddings"][wanted]))

    def unit(v):
        return v / np.linalg.norm(v)

    best, disagreeing_refs = [], []
    for text, idxs in rows.items():
        stack = np.stack([unit(ref_vectors[i]) for i in idxs])
        best.append(float((stack @ unit(f["embeddings"][new[text]])).max()))
        if len(idxs) > 1 and (stack @ stack[0]).min() < threshold:
            disagreeing_refs.append(text)

    best = np.array(best)
    print(f"{len(rows)} of {len(new)} texts also in the reference; "
          f"cosine min={best.min():.6f} mean={best.mean():.6f}")
    print(f"{(best < threshold).sum()} texts below {threshold}: "
          f"{[t for t, b in zip(rows, best) if b < threshold][:5]}")
    if disagreeing_refs:
        print(f"note: {len(disagreeing_refs)} of those texts already disagree with themselves "
              f"inside the reference file, e.g. {disagreeing_refs[:5]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full_h5", required=True, help="Per-sample H5 (sample_ids/texts/embeddings).")
    p.add_argument("--unique_h5", help="Per-distinct-text H5 (texts/embeddings).")
    p.add_argument("--subset_ids_file")
    p.add_argument("--reference_h5", help="Older embeddings file to cross-check against, matched on text.")
    p.add_argument("--similarity_threshold", type=float, default=0.999)
    p.add_argument("--n_dedup_checks", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.unique_h5:
        with h5py.File(args.unique_h5, "r") as f:
            check_structure(f, args.unique_h5)

    with h5py.File(args.full_h5, "r") as f:
        check_structure(f, args.full_h5)
        check_coverage(f, args.subset_ids_file)
        check_dedup(f, args.n_dedup_checks, args.seed)
        if args.reference_h5:
            check_reference(f, args.reference_h5, args.similarity_threshold)
    print("\nDone.")


if __name__ == "__main__":
    main()
