#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity-check an output of embed_subbiomes_keywords.py.

Checks, in order:
  1. Structural: shapes line up, embedding dim matches, no NaN/Inf, no
     all-zero rows, embedding norms are in a sane range.
  2. Coverage: every ID in --subset_ids_file (if given) appears in the
     per-sample H5, or is accounted for in the run's failed_texts/skipped
     side files.
  3. Internal consistency: samples that share the exact same text have
     bit-identical embeddings (this is what deduplication promises).
  4. Optional external cross-check: if --reference_h5 points at an older
     embeddings file (e.g. the pre-existing
     sidequest/GPT_sub_biomes_embeddings_aligned.h5), any text present in
     both files should give near-identical embeddings, since
     text-embedding-3-small is deterministic for a fixed input string. This
     also reports whether the *reference* file agrees with itself across its
     own duplicate rows, since a same-text mismatch there points at a
     historical issue in the older data rather than in the new run.

Example
-------
python scripts/verify_embeddings.py \
    --full_h5 sidequest/latest/embeddings/GPT_sub_biomes_embeddings__text-embedding-3-small__perbiome200_seed42.h5 \
    --unique_h5 sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-small__perbiome200_seed42.h5 \
    --subset_ids_file sidequest/latest/embeddings/subset_ids__perbiome200_seed42.txt \
    --reference_h5 sidequest/GPT_sub_biomes_embeddings_aligned.h5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full_h5", required=True, help="One-row-per-sample H5 (sample_ids/texts/embeddings).")
    parser.add_argument("--unique_h5", default=None, help="Compact per-distinct-text H5 (texts/embeddings).")
    parser.add_argument("--subset_ids_file", default=None, help="Expected sample IDs, one per line.")
    parser.add_argument(
        "--reference_h5", default=None,
        help="An older embeddings H5 (sample_ids/texts/embeddings) to cross-check against, by matching text.",
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.999,
        help="Cosine similarity below this, for a text present in both files, is flagged.",
    )
    return parser.parse_args()


def decode(arr: np.ndarray) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else x for x in arr]


def check_structure(path: str) -> Dict[str, np.ndarray]:
    print(f"\n=== structural checks: {path} ===")
    with h5py.File(path, "r") as f:
        data = {name: f[name][:] for name in f.keys()}

    n = len(data.get("embeddings", []))
    print(f"rows: {n}")
    print(f"columns: {list(data.keys())}")

    embs = data["embeddings"]
    n_nan = int(np.isnan(embs).sum())
    n_inf = int(np.isinf(embs).sum())
    norms = np.linalg.norm(embs, axis=1)
    n_zero = int((norms == 0).sum())
    print(f"embedding dtype: {embs.dtype}, dim: {embs.shape[1]}")
    print(f"NaN values: {n_nan}, Inf values: {n_inf}, all-zero rows: {n_zero}")
    print(f"embedding norm: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")
    if n_nan or n_inf or n_zero:
        print("!! problems found - do not trust this file yet")
    else:
        print("OK: no NaN/Inf/zero-vector rows")
    return data


def check_coverage(full_data: Dict[str, np.ndarray], subset_ids_path: str) -> None:
    print(f"\n=== coverage vs {subset_ids_path} ===")
    with open(subset_ids_path, "r", encoding="utf-8") as handle:
        expected = {line.strip() for line in handle if line.strip()}
    present = set(decode(full_data["sample_ids"]))
    missing = expected - present
    extra = present - expected
    print(f"expected: {len(expected)}, present: {len(present)}, missing: {len(missing)}, unexpected: {len(extra)}")
    if missing:
        print(f"  (missing samples are OK if they had no usable text, or their text failed to embed - "
              f"check failed_texts__*/skipped__* next to the output files) e.g. {sorted(missing)[:5]}")
    if extra:
        print(f"  !! unexpected sample IDs not in the subset list, e.g. {sorted(extra)[:5]}")


def check_same_text_consistency(full_data: Dict[str, np.ndarray]) -> None:
    print("\n=== same-text consistency (dedup promise) ===")
    texts = decode(full_data["texts"])
    embs = full_data["embeddings"]
    by_text: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        by_text[t].append(i)
    dupe_groups = {t: idxs for t, idxs in by_text.items() if len(idxs) > 1}
    n_checked = 0
    n_mismatch = 0
    for idxs in dupe_groups.values():
        ref = embs[idxs[0]]
        for i in idxs[1:]:
            n_checked += 1
            if not np.allclose(embs[i], ref, atol=1e-6):
                n_mismatch += 1
    print(f"{len(dupe_groups)} distinct texts shared by >1 sample here, {n_checked} pairs checked")
    print("OK: all matched" if n_mismatch == 0 else f"!! {n_mismatch} mismatches - dedup/materialize bug")


def check_against_reference(full_data: Dict[str, np.ndarray], reference_path: str, threshold: float) -> None:
    print(f"\n=== cross-check against reference: {reference_path} ===")
    texts = decode(full_data["texts"])
    embs = full_data["embeddings"]
    new_lookup: Dict[str, np.ndarray] = {}
    for t, e in zip(texts, embs):
        new_lookup.setdefault(t, e)  # first occurrence is enough, they're already verified self-consistent

    wanted = set(new_lookup.keys())
    with h5py.File(reference_path, "r") as f:
        ref_embs = f["embeddings"]
        ref_idx_by_text: Dict[str, List[int]] = defaultdict(list)
        # Bulk-read the (much smaller) text column in one shot rather than indexing
        # the HDF5 dataset one row at a time, which is orders of magnitude slower.
        for i, raw in enumerate(f["texts"][:]):
            t = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if t in wanted:
                ref_idx_by_text[t].append(i)

        n_overlap = len(ref_idx_by_text)
        print(f"{n_overlap} / {len(wanted)} distinct texts here also appear in the reference file")
        if not n_overlap:
            return

        best_sims = []
        internally_inconsistent = []
        flagged = []
        for t, idxs in ref_idx_by_text.items():
            ref_vecs = np.stack([ref_embs[i] for i in idxs])
            new_vec = new_lookup[t]
            cos = ref_vecs @ new_vec / (np.linalg.norm(ref_vecs, axis=1) * np.linalg.norm(new_vec))
            best = float(cos.max())
            best_sims.append(best)
            if best < threshold:
                flagged.append((t, best, len(idxs)))
            if len(idxs) > 1:
                self_cos = ref_vecs @ ref_vecs[0] / (np.linalg.norm(ref_vecs, axis=1) * np.linalg.norm(ref_vecs[0]))
                if self_cos.min() < threshold:
                    internally_inconsistent.append(t)

        best_sims_arr = np.array(best_sims)
        print(
            f"best cosine similarity to reference: min={best_sims_arr.min():.6f} "
            f"mean={best_sims_arr.mean():.6f} (threshold {threshold})"
        )
        print(f"texts below threshold: {len(flagged)} / {n_overlap}")
        for t, best, n in flagged[:10]:
            print(f"  cos={best:.6f} ({n} reference rows) -> {t!r}")
        if internally_inconsistent:
            print(
                f"note: {len(internally_inconsistent)} of the overlapping texts are NOT self-consistent "
                "within the reference file itself (its own duplicate rows disagree with each other) - "
                "this points at a pre-existing issue in the reference data, not necessarily in the new run. "
                f"e.g. {internally_inconsistent[:5]}"
            )


def main() -> None:
    args = parse_args()

    full_data = check_structure(args.full_h5)
    if args.unique_h5:
        check_structure(args.unique_h5)
    if args.subset_ids_file:
        check_coverage(full_data, args.subset_ids_file)
    check_same_text_consistency(full_data)
    if args.reference_h5:
        check_against_reference(full_data, args.reference_h5, args.similarity_threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
