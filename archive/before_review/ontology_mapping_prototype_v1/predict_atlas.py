#!/usr/bin/env python3
"""
Label every MicrobeAtlas sample with ENVO/Uberon terms (biome, feature, material)
from its GPT keyword + sub-biome embeddings, using the model that
map_samples_to_ontology.py evaluates as `linear` on `--features kw.npz sb.npz`.

Why this is cheap: a linear model's score is a sum over feature blocks,
    score(sample) = W_kw . kw(sample) + W_sb . sb(sample) + b,
so W_kw . kw is computed once per *distinct* keyword text (1.5M rows, streamed
from the unique .h5 in chunks) and W_sb . sb once per distinct sub-biome (32k).
No per-sample embedding file and no API call is needed.

Resumable: the model, the sample->row index and every finished chunk are
saved in --output_dir; with --max_seconds the script stops cleanly and the
next run continues where it stopped.

python scripts/predict_atlas.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --train_vectors ~/MicrobeAtlasProject/metalog/keywords__large1024.npz \
                  ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --keywords_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_keywords.txt \
  --keywords_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_sub_biomes.txt \
  --sub_biomes_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/atlas_kw_sb
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_subbiomes_keywords import iter_samples  # same text cleaning as the embedded texts
from map_samples_to_ontology import SLOTS, load_precomputed

BLOCK_WEIGHT = 1 / np.sqrt(2)  # two blocks, weighted exactly as in map_samples_to_ontology.stack()


def train_models(args, path):
    """Fit one RidgeClassifier per slot on [kw, sb] of the labelled samples; save coef/intercept/classes."""
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    (kw_row, kw), (sb_row, sb) = (load_precomputed(p) for p in args.train_vectors)
    samples = pd.read_csv(os.path.expanduser(args.samples), sep="\t", keep_default_na=False, dtype=str)
    samples = samples[samples[SLOTS].ne("").any(axis=1)]  # same filtering order as the evaluation
    samples = samples[samples["sample_id"].isin(kw_row) & samples["sample_id"].isin(sb_row)]
    samples = samples.sample(frac=1, random_state=22).groupby("study_code").head(args.max_per_study)
    X = BLOCK_WEIGHT * np.hstack([kw[[kw_row[s] for s in samples["sample_id"]]],
                                  sb[[sb_row[s] for s in samples["sample_id"]]]])
    model = {"kw_dim": kw.shape[1]}  # columns [0, kw_dim) of coef are keywords, the rest sub-biomes
    for slot in SLOTS:
        has = samples[slot].ne("").to_numpy()
        clf = RidgeClassifier(alpha=1.0).fit(X[has], samples[slot].to_numpy()[has])
        model[f"{slot}_coef"], model[f"{slot}_intercept"] = clf.coef_, clf.intercept_
        model[f"{slot}_classes"] = clf.classes_
        print(f"{slot}: trained on {has.sum()} samples, {len(clf.classes_)} labels")
    np.savez(path, **model)
    return model


def build_index(args, path):
    """For every atlas sample with a keyword text: its row in the keyword .h5 and sub-biome .h5 (-1 if none)."""
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    rows = {}
    for kind, texts, h5 in [("keywords", args.keywords_texts, args.keywords_h5),
                            ("sub_biomes", args.sub_biomes_texts, args.sub_biomes_h5)]:
        with h5py.File(os.path.expanduser(h5), "r") as handle:
            row_of = {t.decode(): i for i, t in enumerate(handle["texts"][:])}
        rows[kind] = {sid: row_of.get(t, -1)
                      for sid, t in iter_samples(os.path.expanduser(texts), None, kind == "keywords")}
        print(f"{kind}: {len(rows[kind])} samples with text")
    sample_ids = np.array([s for s, r in rows["keywords"].items() if r >= 0])
    index = {"sample_ids": sample_ids,
             "kw_rows": np.array([rows["keywords"][s] for s in sample_ids]),
             "sb_rows": np.array([rows["sub_biomes"].get(s, -1) for s in sample_ids])}
    np.savez(path, **index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True, help="metalog_training_set.tsv.gz (labels)")
    parser.add_argument("--train_vectors", nargs=2, required=True, help="keywords .npz, sub_biomes .npz")
    parser.add_argument("--keywords_texts", required=True)
    parser.add_argument("--keywords_h5", required=True)
    parser.add_argument("--sub_biomes_texts", required=True)
    parser.add_argument("--sub_biomes_h5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_study", type=int, default=50, help="Same cap as in the evaluation")
    parser.add_argument("--chunk_rows", type=int, default=20_000, help="Keyword .h5 rows per chunk (memory)")
    parser.add_argument("--max_seconds", type=float, default=None, help="Stop after this long (resume later)")
    args = parser.parse_args()
    start = time.time()
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)

    model = train_models(args, os.path.join(out_dir, "model.npz"))
    index = build_index(args, os.path.join(out_dir, "index.npz"))
    terms = pd.read_csv(os.path.expanduser(args.ontology_terms), sep="\t", keep_default_na=False)
    label_of = dict(zip(terms["term_id"], terms["label"]))

    # sub-biome contribution per distinct sub-biome text (+ intercept); last row = "no sub-biome"
    with h5py.File(os.path.expanduser(args.sub_biomes_h5), "r") as handle:
        sb = BLOCK_WEIGHT * normalize(handle["embeddings"][:])
    kw_dim = int(model["kw_dim"])
    coef = {slot: model[f"{slot}_coef"].astype(np.float32) for slot in SLOTS}  # float32 halves memory
    sb_scores = {slot: np.vstack([sb @ coef[slot][:, kw_dim:].T, np.zeros(len(model[f"{slot}_classes"]))])
                 + model[f"{slot}_intercept"] for slot in SLOTS}
    sb_rows = np.where(index["sb_rows"] >= 0, index["sb_rows"], len(sb))

    with h5py.File(os.path.expanduser(args.keywords_h5), "r") as handle:
        n_rows = handle["embeddings"].shape[0]
        for first in range(0, n_rows, args.chunk_rows):
            part = os.path.join(out_dir, "parts", f"rows_{first:08d}.tsv.gz")
            if os.path.exists(part):
                continue
            if args.max_seconds and time.time() - start > args.max_seconds:
                print(f"Stopping after {time.time() - start:.0f}s; rerun to continue")
                return
            kw = normalize(handle["embeddings"][first:first + args.chunk_rows]) * np.float32(BLOCK_WEIGHT)
            in_chunk = (index["kw_rows"] >= first) & (index["kw_rows"] < first + len(kw))
            out = pd.DataFrame({"sample_id": index["sample_ids"][in_chunk]})
            for slot in SLOTS:
                scores = (kw @ coef[slot][:, :kw_dim].T)[index["kw_rows"][in_chunk] - first]
                scores += sb_scores[slot][sb_rows[in_chunk]]
                top2 = np.argpartition(-scores, 1, axis=1)[:, :2]  # best two, unordered
                top2 = np.take_along_axis(top2, np.argsort(-np.take_along_axis(scores, top2, axis=1), axis=1), axis=1)
                best = model[f"{slot}_classes"][top2[:, 0]]
                out[slot] = best
                out[f"{slot}_label"] = [label_of.get(t, "") for t in best]
                out[f"{slot}_confidence"] = np.round(np.take_along_axis(scores, top2, axis=1) @ [1, -1], 4)
            out.to_csv(part + ".tmp", sep="\t", index=False, compression="gzip")
            os.replace(part + ".tmp", part)  # a chunk counts as done only once fully written
            print(f"rows {first}-{first + len(kw)}: {in_chunk.sum()} samples ({time.time() - start:.0f}s)")

    final = os.path.join(out_dir, "atlas_predictions.tsv.gz")
    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "rows_*.tsv.gz")))
    pd.concat(pd.read_csv(p, sep="\t", keep_default_na=False) for p in parts).to_csv(
        final, sep="\t", index=False, compression="gzip")
    print(f"Wrote {final}")


if __name__ == "__main__":
    main()
