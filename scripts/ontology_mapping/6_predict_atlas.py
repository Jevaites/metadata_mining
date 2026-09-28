#!/usr/bin/env python3
"""
Step 6: label every MicrobeAtlas sample with one ENVO/Uberon term per slot
(biome, feature, material) + a confidence, using the `linear` model of
5_evaluate.py on `--features keywords.npz sub_biomes.npz`.

Why it is cheap: a linear score is a sum over the two feature blocks,
    score(sample) = W_kw . kw(sample) + W_sb . sb(sample) + b,
so W_kw . kw is computed once per *distinct* keyword text (1.5M rows, streamed from
the unique .h5 in chunks) and W_sb . sb once per distinct sub-biome text (32k).
No per-sample embedding file and no API call is needed.

Training samples: exactly those of the evaluation (common.select_samples with the same
--max_per_study and --seed), so the atlas labels come from the model that was scored.
Pass --max_per_study 0 to train on every linked sample instead.

Confidence = best minus second-best score (as `linear` in 5_evaluate.py): use the
margin_for_90pct_precision values of metrics.json to decide which labels to trust.

Resumable: model.npz, index.npz and every finished chunk are kept in --output_dir, and
--max_seconds stops cleanly; rerun the same command to continue. Delete --output_dir
to retrain (for example after changing the training options).

python 6_predict_atlas.py \
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

from common import SLOTS, load_npz, path, read_tsv, select_samples

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
from embed_subbiomes_keywords import iter_samples  # same text cleaning as the embedded texts

BLOCK_WEIGHT = np.float32(1 / np.sqrt(2))  # two blocks, weighted exactly as in 5_evaluate.build_features()


def train_models(args, model_path):
    """One RidgeClassifier per slot on [kw, sb]; saves coef / intercept / classes."""
    if os.path.exists(model_path):
        print(f"Using existing {model_path} (delete the output dir to retrain)")
        return dict(np.load(model_path))
    (kw_row, kw), (sb_row, sb) = (load_npz(p) for p in args.train_vectors)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)], args.max_per_study, args.seed)
    ids = samples["sample_id"]
    X = BLOCK_WEIGHT * np.hstack([kw[[kw_row[s] for s in ids]], sb[[sb_row[s] for s in ids]]])
    model = {"kw_dim": kw.shape[1]}  # columns [0, kw_dim) of coef are keywords, the rest sub-biomes
    for slot in SLOTS:
        has = samples[slot].ne("").to_numpy()
        clf = RidgeClassifier(alpha=1.0).fit(X[has], samples[slot].to_numpy()[has])
        model[f"{slot}_coef"], model[f"{slot}_intercept"], model[f"{slot}_classes"] = \
            clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32), clf.classes_.astype(str)
        print(f"{slot}: trained on {has.sum()} samples, {len(clf.classes_)} labels")
    np.savez(model_path, **model)
    return model


def build_index(args, index_path):
    """For every atlas sample with a keyword text: its row in the keyword .h5 and in the
    sub-biome .h5 (-1 = no sub-biome)."""
    if os.path.exists(index_path):
        return dict(np.load(index_path))
    rows = {}
    for kind, texts, h5 in [("keywords", args.keywords_texts, args.keywords_h5),
                            ("sub_biomes", args.sub_biomes_texts, args.sub_biomes_h5)]:
        with h5py.File(path(h5), "r") as handle:
            row_of = {t.decode(): i for i, t in enumerate(handle["texts"][:])}
        rows[kind] = {sid: row_of.get(t, -1) for sid, t in iter_samples(path(texts), None, kind == "keywords")}
        print(f"{kind}: {len(rows[kind])} samples with text")
    sample_ids = np.array([s for s, r in rows["keywords"].items() if r >= 0])
    index = {"sample_ids": sample_ids,
             "kw_rows": np.array([rows["keywords"][s] for s in sample_ids]),
             "sb_rows": np.array([rows["sub_biomes"].get(s, -1) for s in sample_ids])}
    np.savez(index_path, **index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True, help="2_build_training_set.py output (labels)")
    parser.add_argument("--train_vectors", nargs=2, required=True, help="keywords .npz, sub_biomes .npz")
    parser.add_argument("--keywords_texts", required=True)
    parser.add_argument("--keywords_h5", required=True)
    parser.add_argument("--sub_biomes_texts", required=True)
    parser.add_argument("--sub_biomes_h5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_study", type=int, default=50, help="Same cap as the evaluation (0 = all)")
    parser.add_argument("--seed", type=int, default=22, help="Same seed as the evaluation")
    parser.add_argument("--chunk_rows", type=int, default=20_000, help="Keyword .h5 rows per chunk (memory)")
    parser.add_argument("--max_seconds", type=float, default=None, help="Stop after this long (resume later)")
    args = parser.parse_args()
    start = time.time()
    out_dir = path(args.output_dir)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)

    model = train_models(args, os.path.join(out_dir, "model.npz"))
    index = build_index(args, os.path.join(out_dir, "index.npz"))
    terms = read_tsv(args.ontology_terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    kw_dim = int(model["kw_dim"])
    coef = {slot: model[f"{slot}_coef"] for slot in SLOTS}

    # sub-biome part of the score (+ intercept) per distinct sub-biome text; extra last row = no sub-biome
    with h5py.File(path(args.sub_biomes_h5), "r") as handle:
        sb = BLOCK_WEIGHT * normalize(handle["embeddings"][:])
    sb_scores = {slot: np.vstack([sb @ coef[slot][:, kw_dim:].T, np.zeros((1, coef[slot].shape[0]))])
                 + model[f"{slot}_intercept"] for slot in SLOTS}
    sb_rows = np.where(index["sb_rows"] >= 0, index["sb_rows"], len(sb))

    with h5py.File(path(args.keywords_h5), "r") as handle:
        n_rows = handle["embeddings"].shape[0]
        for first in range(0, n_rows, args.chunk_rows):
            part = os.path.join(out_dir, "parts", f"rows_{first:08d}.tsv.gz")
            if os.path.exists(part):
                continue
            if args.max_seconds and time.time() - start > args.max_seconds:
                print(f"Stopping after {time.time() - start:.0f}s; rerun to continue")
                return
            kw = BLOCK_WEIGHT * normalize(handle["embeddings"][first:first + args.chunk_rows])
            in_chunk = (index["kw_rows"] >= first) & (index["kw_rows"] < first + len(kw))
            out = pd.DataFrame({"sample_id": index["sample_ids"][in_chunk]})
            for slot in SLOTS:
                scores = (kw @ coef[slot][:, :kw_dim].T)[index["kw_rows"][in_chunk] - first]
                scores += sb_scores[slot][sb_rows[in_chunk]]
                top2 = np.argpartition(-scores, 1, axis=1)[:, :2]  # best two, unordered
                top2 = np.take_along_axis(top2, np.argsort(-np.take_along_axis(scores, top2, axis=1), axis=1), axis=1)
                best = model[f"{slot}_classes"][top2[:, 0]]
                out[slot], out[f"{slot}_label"] = best, [label_of.get(t, "") for t in best]
                out[f"{slot}_confidence"] = np.round(np.take_along_axis(scores, top2, axis=1) @ [1, -1], 4)
            out.to_csv(part + ".tmp", sep="\t", index=False, compression="gzip")
            os.replace(part + ".tmp", part)  # a chunk counts as done only once fully written
            print(f"rows {first}-{first + len(kw)}: {in_chunk.sum()} samples ({time.time() - start:.0f}s)")

    final = os.path.join(out_dir, "atlas_predictions.tsv.gz")
    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "rows_*.tsv.gz")))
    pd.concat(read_tsv(p) for p in parts).to_csv(final, sep="\t", index=False, compression="gzip")
    print(f"Wrote {final}")


if __name__ == "__main__":
    main()
