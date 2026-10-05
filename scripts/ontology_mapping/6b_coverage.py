#!/usr/bin/env python3
"""
Step 6b: flag atlas samples that lie outside what the Metalog training data covers.

The atlas models (6_predict_atlas.py) always answer with a Metalog label, also for habitats Metalog
has no samples of (laboratory, air, food, plant tissue...): there they are confidently wrong
instead of abstaining (claude/gold-check-and-project-folds.md). This step measures, per atlas
sample, how close it is to the model's training samples:

  coverage_sim   cosine similarity, in the model's feature space x = [kw, sb] / sqrt 2, to the
                 nearest training sample (mean of the --k nearest). Samples without a sub-biome
                 text use the keyword cosine alone.
  nearest_study  the study of the nearest training sample
  in_coverage    coverage_sim >= threshold (both rounded to 4 decimals, so the flag can be
                 recomputed from the output file and is the same in every run, resumed or not)

The threshold is calibrated on Metalog itself: every training sample is compared with the training
samples of *other* projects (--fold_groups, else study codes), as a new study would be, and the
threshold is the --quantile of these similarities (default 0.05: an atlas sample is flagged when it
is farther from Metalog than 95 % of the held-out Metalog samples). coverage_calibration.json keeps
the reference distribution. It is a property of the inputs only, not of a method, so one run serves
linear, prototype and knn_study atlases (merge on sample_id).

Training samples: those of the atlas model (common.select_samples with the same --max_per_study and
--seed). Similarities are computed once per distinct (keyword, sub-biome) text pair (1.6M for 3.4M
samples) against the distinct training vectors. Resumable like step 6 (--max_seconds, parts/).

python 6b_coverage.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --train_vectors ~/MicrobeAtlasProject/metalog/keywords__large1024.npz \
                  ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --fold_groups ~/MicrobeAtlasProject/metalog/clean/project_groups.tsv \
  --index ~/MicrobeAtlasProject/ontology_mapping/atlas_backoff/index.npz \
  --keywords_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/atlas_coverage
"""

import argparse
import glob
import json
import os
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from common import load_npz, path, read_tsv, select_samples


def top_k_mean(sim, k):
    """-> (mean of the k largest values per row, column of the largest)."""
    best = sim.argmax(axis=1)
    if k == 1:
        return sim[np.arange(len(sim)), best], best
    top = np.partition(sim, sim.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1), best


def calibrate(kw, sb, groups, k, quantile, block=2000):
    """Similarity of every training sample to the training samples of the other groups."""
    _, g = np.unique(groups, return_inverse=True)
    held_out = np.zeros(len(kw))
    for s0 in range(0, len(kw), block):
        sim = 0.5 * (kw[s0:s0 + block] @ kw.T + sb[s0:s0 + block] @ sb.T)
        sim[g[s0:s0 + block, None] == g[None, :]] = -np.inf
        held_out[s0:s0 + block] = top_k_mean(sim, k)[0]
    qs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
    return float(np.quantile(held_out, quantile)), {str(q): round(float(np.quantile(held_out, q)), 4) for q in qs}, held_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="2_build_training_set.py / 2b_clean_metalog.py output")
    ap.add_argument("--train_vectors", nargs=2, required=True, help="keywords .npz, sub_biomes .npz")
    ap.add_argument("--fold_groups", default=None, help="project_groups.tsv (experiments/project_groups.py); default study_code")
    ap.add_argument("--index", required=True, help="index.npz written by 6_predict_atlas.py (atlas sample -> .h5 rows)")
    ap.add_argument("--keywords_h5", required=True)
    ap.add_argument("--sub_biomes_h5", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k", type=int, default=1, help="coverage_sim = mean similarity of the k nearest training samples")
    ap.add_argument("--quantile", type=float, default=0.05, help="Threshold = this quantile of the held-out Metalog similarities")
    ap.add_argument("--max_per_study", type=int, default=50, help="Same cap as the atlas model")
    ap.add_argument("--seed", type=int, default=22, help="Same seed as the atlas model")
    ap.add_argument("--only_ids", default=None, help="File with one sample id per line: score only these (tests)")
    ap.add_argument("--chunk_rows", type=int, default=5000, help="Keyword .h5 rows per chunk (memory)")
    ap.add_argument("--max_seconds", type=float, default=None, help="Stop after this long (resume later)")
    args = ap.parse_args()
    start = time.time()
    out_dir = path(args.output_dir)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)

    # training samples of the atlas model, as float32 [kw | sb] blocks
    (kw_row, kw), (sb_row, sb) = (load_npz(p) for p in args.train_vectors)
    train = select_samples(args.samples, [set(kw_row), set(sb_row)], args.max_per_study, args.seed)
    train_kw = kw[[kw_row[s] for s in train["sample_id"]]].astype(np.float32)
    train_sb = sb[[sb_row[s] for s in train["sample_id"]]].astype(np.float32)
    train_studies = train["study_code"].to_numpy()

    settings = {"samples": os.path.abspath(path(args.samples)), "fold_groups": args.fold_groups, "k": args.k,
                "quantile": args.quantile, "max_per_study": args.max_per_study, "seed": args.seed,
                "only_ids": args.only_ids, "n_train": len(train)}
    calibration_path = os.path.join(out_dir, "coverage_calibration.json")
    if os.path.exists(calibration_path):
        saved = json.load(open(calibration_path))
        if saved["settings"] != settings:
            raise SystemExit(f"{out_dir} was written with {saved['settings']}, not {settings}: use another --output_dir")
        threshold = saved["threshold"]
    else:
        groups = train["study_code"]
        if args.fold_groups:
            project = dict(read_tsv(args.fold_groups)[["sample_id", "project_group"]].values)
            groups = train["sample_id"].map(project).fillna(train["study_code"])
        threshold, quantiles, held_out = calibrate(train_kw, train_sb, groups.to_numpy(), args.k, args.quantile)
        threshold = round(threshold, 4)  # the value saved and used by every (resumed) run
        pd.DataFrame({"sample_id": train["sample_id"], "study_code": train_studies,
                      "held_out_sim": np.round(held_out, 4)}).to_csv(
            os.path.join(out_dir, "metalog_held_out_sim.tsv.gz"), sep="\t", index=False, compression="gzip")
        json.dump({"settings": settings, "threshold": threshold, "held_out_quantiles": quantiles},
                  open(calibration_path, "w"), indent=1)
        print(f"{len(train)} training samples, {groups.nunique()} groups; held-out similarity quantiles {quantiles}; "
              f"threshold {threshold:.4f} ({time.time() - start:.0f}s)")

    index = dict(np.load(path(args.index)))
    if args.only_ids:
        keep = np.isin(index["sample_ids"], open(path(args.only_ids)).read().split())
        index = {name: v[keep] for name, v in index.items()}
    # many training samples share a keyword or sub-biome text: one column per distinct vector, expanded per block
    kw_u, kw_col = np.unique(train_kw, axis=0, return_inverse=True)
    sb_u, sb_col = np.unique(train_sb, axis=0, return_inverse=True)
    kw_col, sb_col = kw_col.ravel(), sb_col.ravel()
    print(f"{len(kw_u)} distinct keyword and {len(sb_u)} distinct sub-biome vectors among the training samples")
    with h5py.File(path(args.sub_biomes_h5), "r") as handle:
        sb_sim = normalize(handle["embeddings"][:]).astype(np.float32) @ sb_u.T  # atlas sub-biomes x distinct train
    with h5py.File(path(args.keywords_h5), "r") as handle:
        n_rows = handle["embeddings"].shape[0]
        for first in range(0, n_rows, args.chunk_rows):
            part = os.path.join(out_dir, "parts", f"rows_{first:08d}.tsv.gz")
            in_chunk = (index["kw_rows"] >= first) & (index["kw_rows"] < first + args.chunk_rows)
            if os.path.exists(part) or not in_chunk.any():
                continue
            if args.max_seconds and time.time() - start > args.max_seconds:
                print(f"Stopping after {time.time() - start:.0f}s; rerun to continue")
                return
            rows, kw_local = np.unique(index["kw_rows"][in_chunk] - first, return_inverse=True)
            kw_chunk = handle["embeddings"][first:first + args.chunk_rows][rows]  # only rows used by samples
            kw_sim = normalize(kw_chunk).astype(np.float32) @ kw_u.T
            # one score per distinct (keyword, sub-biome) pair: many atlas samples share both texts
            pairs, sample_pair = np.unique(np.c_[kw_local, index["sb_rows"][in_chunk]], axis=0, return_inverse=True)
            sim_out, best_out = np.zeros(len(pairs), np.float32), np.zeros(len(pairs), int)
            for s0 in range(0, len(pairs), 2000):
                kw_p, sb_p = pairs[s0:s0 + 2000, 0], pairs[s0:s0 + 2000, 1]
                has_sb = sb_p >= 0
                sim = kw_sim[kw_p][:, kw_col]                                         # pairs x training samples
                sim[has_sb] = 0.5 * (sim[has_sb] + sb_sim[sb_p[has_sb]][:, sb_col])  # no sub-biome: keyword cosine
                sim_out[s0:s0 + 2000], best_out[s0:s0 + 2000] = top_k_mean(sim, args.k)
            sample_pair = sample_pair.ravel()
            sim_out, best_out = np.round(sim_out[sample_pair].astype(np.float64), 4), best_out[sample_pair]
            pd.DataFrame({"sample_id": index["sample_ids"][in_chunk], "coverage_sim": sim_out,
                          "nearest_study": train_studies[best_out],
                          "in_coverage": sim_out >= threshold}).to_csv(part + ".tmp", sep="\t", index=False, compression="gzip")
            os.replace(part + ".tmp", part)
            print(f"rows {first}-{first + args.chunk_rows}: {in_chunk.sum()} samples ({time.time() - start:.0f}s)")

    final = os.path.join(out_dir, "atlas_coverage.tsv.gz")
    out = pd.concat(read_tsv(p) for p in sorted(glob.glob(os.path.join(out_dir, "parts", "rows_*.tsv.gz"))))
    out["in_coverage"] = out["coverage_sim"].astype(float) >= threshold  # same rule for every part
    out.to_csv(final, sep="\t", index=False, compression="gzip")
    print(f"Wrote {final}: {len(out)} samples, {(~out['in_coverage']).mean():.1%} outside coverage "
          f"(threshold {threshold:.4f})")


if __name__ == "__main__":
    main()
