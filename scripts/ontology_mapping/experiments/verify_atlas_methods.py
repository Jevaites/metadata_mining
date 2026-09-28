#!/usr/bin/env python3
"""
Check that 6_predict_atlas.py (streamed, decomposed per feature block) gives the same top-1 and
confidence as 5_evaluate.py's functions applied directly to the full sample vectors, for every
--method. The "atlas" here is the ~50k Metalog-linked samples of the .npz files: their vectors are
written as unique-text .h5 files and index.npz is written directly, so no GPT text file is needed.

python experiments/verify_atlas_methods.py [--work_dir /tmp/atlas_check]
Expected: 100 % identical top-1 and |confidence difference| <= 0.0001 (4-decimal rounding).
"""
import argparse
import importlib
import os
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import SLOTS, load_npz, load_term_vectors, load_terms, path, read_tsv, select_samples  # noqa: E402
from _setup import DEFAULTS  # noqa: E402

evaluate = importlib.import_module("5_evaluate")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in DEFAULTS.items():
        p.add_argument(f"--{name}", default=default)
    p.add_argument("--work_dir", default="/tmp/atlas_method_check")
    p.add_argument("--methods", nargs="+", default=["linear", "prototype", "knn_study"])
    args = p.parse_args()
    work = path(args.work_dir)
    os.makedirs(work, exist_ok=True)

    # 1. mini atlas: .h5 files from the .npz unique vectors, and index.npz
    kw_npz, sb_npz = np.load(path(args.keywords)), np.load(path(args.sub_biomes))
    for name, z in [("kw", kw_npz), ("sb", sb_npz)]:
        with h5py.File(f"{work}/{name}.h5", "w") as h:
            h["embeddings"] = z["vectors"]
            h["texts"] = np.array([f"{name}{i}".encode() for i in range(len(z["vectors"]))])
    sb_row = dict(zip(sb_npz["sample_ids"], sb_npz["index"]))
    ids = kw_npz["sample_ids"]
    index = {"sample_ids": ids, "kw_rows": kw_npz["index"], "sb_rows": np.array([sb_row.get(s, -1) for s in ids])}
    print(f"mini atlas: {len(ids)} samples ({(index['sb_rows'] < 0).sum()} without sub-biome)")

    # 2. direct computation with 5_evaluate's functions
    terms = load_terms(args.ontology_terms)
    term_ids = terms["term_id"].to_numpy()
    tv = load_term_vectors(args.term_vectors, list(terms["text"]))
    term_matrix = np.hstack([tv, tv]) / np.sqrt(2)
    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kr), set(sr)])
    X = np.hstack([K[[kr[i] for i in samples["sample_id"]]], B[[sr[i] for i in samples["sample_id"]]]]) / np.sqrt(2)
    A = np.hstack([K[[kr[i] for i in ids]], np.array([B[sr[i]] if i in sr else np.zeros(B.shape[1]) for i in ids])]) / np.sqrt(2)

    failures = 0
    for method in args.methods:
        out_dir = f"{work}/{method}"
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(f"{out_dir}/atlas_predictions.tsv.gz"):
            np.savez(f"{out_dir}/index.npz", **index)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "6_predict_atlas.py"), "--method", method,
                            "--ontology_terms", args.ontology_terms, "--samples", args.samples,
                            "--train_vectors", args.keywords, args.sub_biomes, "--term_vectors", args.term_vectors,
                            "--keywords_texts", "unused", "--sub_biomes_texts", "unused",
                            "--keywords_h5", f"{work}/kw.h5", "--sub_biomes_h5", f"{work}/sb.h5",
                            "--chunk_rows", "5000", "--output_dir", out_dir], check=True, stdout=subprocess.DEVNULL)
        atlas = read_tsv(f"{out_dir}/atlas_predictions.tsv.gz").set_index("sample_id").loc[ids]
        for slot in SLOTS:
            has = samples[slot].ne("").to_numpy()
            y = samples[slot].to_numpy()[has]
            if method == "linear":
                classes, scores = evaluate.linear_scores(A, X[has], y)
                top5, confidence = evaluate.rank(scores, classes)
            elif method == "prototype":
                closed = np.isin(term_ids, y)
                model = evaluate.prototype_model(X[has], y, term_matrix[closed], term_ids[closed], 0.5, 0.1)
                top5, confidence = evaluate.prototype(A, model, term_ids[closed])
            else:
                top5, confidence = evaluate.knn_study(A, X[has], y, samples["study_code"].to_numpy()[has], 50)
            same = np.array([t[0] for t in top5]) == atlas[slot].to_numpy()
            conf_diff = np.abs(np.round(confidence, 4) - atlas[f"{slot}_confidence"].astype(float).to_numpy()).max()
            ok = same.all() and conf_diff <= 1.5e-4
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {method:10s} {slot:8s} identical top-1 {same.mean():.5f} "
                  f"({(~same).sum()} differ), max |confidence diff| {conf_diff:.4f}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
