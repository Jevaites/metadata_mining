#!/usr/bin/env python3
"""
Pull per-sample vectors out of a *unique* embedding file (one row per distinct
text, as written by embed_subbiomes_keywords.py) for a chosen set of samples.

sample -> text comes from the GPT source file (GPT_keywords.txt / GPT_sub_biomes.txt),
cleaned with the *same* clean_text() that produced the embedded texts, so the
lookup is an exact string match.

Output .npz: sample_ids (n,), index (n,), vectors (k, dim)
  -> the vector of sample_ids[i] is vectors[index[i]]  (k distinct texts <= n samples)

python scripts/extract_sample_embeddings.py --kind keywords \
  --texts ~/MicrobeAtlasProject/sidequest/latest/GPT_keywords.txt \
  --unique_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sample_ids ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --output ~/MicrobeAtlasProject/metalog/keywords__large1024.npz
"""

import argparse
import gzip
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_subbiomes_keywords import iter_samples  # same cleaning as when the texts were embedded


def read_first_column(path):
    """Sample ids = first tab-separated column (a 'sample_id' header line is skipped)."""
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as handle:
        ids = {line.split("\t", 1)[0].strip() for line in handle}
    return ids - {"sample_id", ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=["keywords", "sub_biomes"], required=True)
    parser.add_argument("--texts", required=True, help="GPT_keywords.txt or GPT_sub_biomes.txt")
    parser.add_argument("--unique_h5", required=True, help="GPT_*_unique_embeddings__*.h5 (texts, embeddings)")
    parser.add_argument("--sample_ids", required=True, help="File whose first column is sample_id")
    parser.add_argument("--output", required=True, help=".npz to write")
    args = parser.parse_args()

    wanted = read_first_column(os.path.expanduser(args.sample_ids))
    text_of = dict(iter_samples(os.path.expanduser(args.texts), wanted, args.kind == "keywords"))
    print(f"{len(wanted)} samples requested, {len(text_of)} have a {args.kind} text")

    with h5py.File(os.path.expanduser(args.unique_h5), "r") as h5:
        row_of = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(h5["texts"][:])}
        found = {sid: row_of[t] for sid, t in text_of.items() if t in row_of}
        rows = np.array(sorted(set(found.values())))  # h5py needs increasing indices
        vectors = h5["embeddings"][rows]
    print(f"{len(found)} samples found in {os.path.basename(args.unique_h5)} "
          f"({len(text_of) - len(found)} texts missing), {len(rows)} distinct vectors")

    position = {row: i for i, row in enumerate(rows)}
    sample_ids = np.array(sorted(found))
    np.savez(os.path.expanduser(args.output), sample_ids=sample_ids,
             index=np.array([position[found[s]] for s in sample_ids]), vectors=vectors)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
