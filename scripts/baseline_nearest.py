#!/usr/bin/env python3
"""
"Trivial" embedding baselines for mapping samples to ENVO/Uberon, with no training:

  nearest_term_open    the term (out of all 18.8k ENVO+Uberon terms) closest to the
                       sample's keyword embedding, by Euclidean distance
  nearest_term_closed  same, but only among the terms Metalog uses in that slot
  knn1 / knn25         label of the closest labelled sample / majority label of the 25
                       closest, by Euclidean distance (5-fold CV grouped by study)
  majority             most frequent label in the training folds

Same samples, cap and folds as map_samples_to_ontology.py (--features keywords.npz
sub_biomes.npz), so the numbers compare directly with the prototype's linear model.
Vectors are used as stored (not re-normalised); OpenAI embeddings have unit norm,
so Euclidean ranking equals cosine ranking; the script prints the norms to show it.

python scripts/baseline_nearest.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --keywords ~/MicrobeAtlasProject/metalog/keywords__large1024.npz \
  --same_samples_as ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --term_vectors ~/MicrobeAtlasProject/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/baseline_nearest
"""

import argparse
import json
import os
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_samples_to_ontology import SLOTS, score, term_text


def nearest(queries, items, k, chunk=2000):
    """Indices of the k items with the smallest Euclidean distance to each query, closest first."""
    item_sq = (items ** 2).sum(1)
    out = []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        dist_sq = (q ** 2).sum(1)[:, None] + item_sq[None, :] - 2 * q @ items.T  # ||q - x||^2
        idx = np.argpartition(dist_sq, min(k, items.shape[0] - 1), axis=1)[:, :k]
        out.append(np.take_along_axis(idx, np.argsort(np.take_along_axis(dist_sq, idx, 1), 1), 1))
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--keywords", required=True, help="keywords .npz from extract_sample_embeddings.py")
    parser.add_argument("--same_samples_as", nargs="*", default=[],
                        help=".npz files; keep only samples also in them (to match the prototype's sample set)")
    parser.add_argument("--term_vectors", required=True, help=".h5 from embed_ontology_terms.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_per_study", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # terms: ids, raw embeddings (as stored), parents for the parent/child metric
    terms = pd.read_csv(os.path.expanduser(args.ontology_terms), sep="\t", keep_default_na=False)
    terms = terms[terms["obsolete"].astype(str) != "True"].reset_index(drop=True)
    with h5py.File(os.path.expanduser(args.term_vectors), "r") as handle:
        row_of = {t.decode(): i for i, t in enumerate(handle["texts"][:])}
        term_vecs = handle["embeddings"][:][[row_of[t] for t in term_text(terms)]]
    term_ids = terms["term_id"].to_numpy()
    term_vec_of = dict(zip(term_ids, term_vecs / np.linalg.norm(term_vecs, axis=1, keepdims=True)))
    parents = {t: set(p.split("||")) for t, p in zip(term_ids, terms["parents"]) if p}

    # samples: same filtering / shuffling / per-study cap as map_samples_to_ontology.py
    saved = np.load(os.path.expanduser(args.keywords))
    kw_row = {s: i for i, s in enumerate(saved["sample_ids"])}
    kw_vecs = saved["vectors"][saved["index"]]
    samples = pd.read_csv(os.path.expanduser(args.samples), sep="\t", keep_default_na=False, dtype=str)
    samples = samples[samples[SLOTS].ne("").any(axis=1)]
    for path in [args.keywords, *args.same_samples_as]:
        samples = samples[samples["sample_id"].isin(set(np.load(os.path.expanduser(path))["sample_ids"]))]
    samples = samples.sample(frac=1, random_state=args.seed).groupby("study_code").head(args.max_per_study)
    samples = samples.reset_index(drop=True)
    X = kw_vecs[[kw_row[s] for s in samples["sample_id"]]]
    print(f"{len(samples)} samples, {samples.study_code.nunique()} studies; vector norms: samples "
          f"{np.linalg.norm(X, axis=1).min():.4f}-{np.linalg.norm(X, axis=1).max():.4f}, terms "
          f"{np.linalg.norm(term_vecs, axis=1).min():.4f}-{np.linalg.norm(term_vecs, axis=1).max():.4f}")

    rows = []  # (slot, method, row, top5)
    folds = list(GroupKFold(n_splits=args.folds).split(samples, groups=samples["study_code"]))
    for slot in SLOTS:
        gold = samples[slot].to_numpy()
        has = gold != ""
        # no training at all: nearest term, among all terms or among the terms Metalog uses in this slot
        closed = np.isin(term_ids, np.unique(gold[has]))
        for method, mask in [("nearest_term_open", np.ones(len(term_ids), bool)), ("nearest_term_closed", closed)]:
            idx = nearest(X[has], term_vecs[mask], 5)
            rows += [(slot, method, r, term_ids[mask][i].tolist()) for r, i in zip(np.where(has)[0], idx)]
        # trained on labelled samples of the other studies
        for train_idx, test_idx in folds:
            tr, te = train_idx[has[train_idx]], test_idx[has[test_idx]]
            neighbours = nearest(X[te], X[tr], 25)
            majority = [label for label, _ in Counter(gold[tr]).most_common(5)]
            for r, nb in zip(te, neighbours):
                votes = [label for label, _ in Counter(gold[tr][nb]).most_common(5)]  # ties: closest first
                rows += [(slot, "knn1", r, list(dict.fromkeys(gold[tr][nb]))[:5]),
                         (slot, "knn25", r, votes), (slot, "majority", r, majority)]
        print(f"{slot} done")

    pred = pd.DataFrame(rows, columns=["slot", "method", "row", "top5"])
    pred["gold"] = [samples.at[r, s] for r, s in zip(pred["row"], pred["slot"])]
    metrics = {slot: {m: score(g["top5"].tolist(), g["gold"].tolist(), parents, np.zeros(len(g)),
                               np.ones(len(g), bool), term_vec_of)
                      for m, g in by_slot.groupby("method")} for slot, by_slot in pred.groupby("slot")}
    for d in metrics.values():  # metrics that are meaningless here (no confidence, no train/unseen split)
        for v in d.values():
            for key in ["top1_confident_half", "n_unseen_label", "top1_unseen_label"]:
                v.pop(key)
    pred["pred"] = pred["top5"].str[0]
    pred["domain"] = samples.loc[pred["row"], "domain"].to_numpy()
    pred["sample_id"] = samples.loc[pred["row"], "sample_id"].to_numpy()
    by_domain = (pred["pred"] == pred["gold"]).groupby([pred["slot"], pred["method"], pred["domain"]]).mean()

    label_of = dict(zip(term_ids, terms["label"]))
    pred["gold_label"], pred["pred_label"] = pred["gold"].map(label_of), pred["pred"].map(label_of)
    pred["top5"] = pred["top5"].str.join("||")
    pred.drop(columns="row").to_csv(os.path.join(out_dir, "predictions.tsv.gz"), sep="\t", index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    pd.set_option("display.width", 200)
    print(pd.DataFrame({(s, m): v for s, d in metrics.items() for m, v in d.items()}).T)
    print("\ntop1 by domain:\n", by_domain.unstack().round(3))


if __name__ == "__main__":
    main()
