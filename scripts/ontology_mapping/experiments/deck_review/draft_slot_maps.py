#!/usr/bin/env python3
"""
Would feature and material label maps help, like the biome label map did? (report §8.8)

MIxS restricts each slot to a class of terms (env_local_scale: finer ENVO terms, UBERON / PO for hosts;
env_medium: ENVO environmental material or other mass nouns). Metalog labels outside these classes:
    feature   allowed: ENVO environmental system / astronomical body part, UBERON anatomical structure,
              PO plant anatomy (not materials, biomes or UBERON substances)
    material  allowed: ENVO environmental material, UBERON organism substance, FOODON food material
For every out-of-class label, two automatic sources propose an in-class Metalog label (the procedure of
draft_biome_label_map.py): (1) the production prototype trained on in-class labels only, out of fold,
majority over the label's samples; (2) the majority label of the 25 nearest in-class samples from other
studies. The draft map takes source 1. Then the prototype is re-evaluated with the mapped labels; the
fair comparison is top-1 on samples whose label the map did not touch.

Findings (fold seed 0): out of class: feature 17 % of samples (64 of 208 labels), material 11 % (84 of 185).
The sources agree on 37 / 64 and 57 / 84 labels; many proposals are visibly wrong (middle ear -> fecal
material, greenhouse soil -> wastewater treatment plant). Untouched samples: feature top-1 .715 -> .749,
macro .084 -> .130; material .797 -> .807, macro .102 -> .123 (the biome map gave +4.9). A real map needs
review: lateral moves (structure -> substance, material -> place), targets Metalog does not use, and
study-dependent meanings (surface water = ocean or lake).

python experiments/deck_review/draft_slot_maps.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Data, parser, path, prototype_fold_scores, save_json  # noqa: E402

BIOME, MATERIAL, SUBSTANCE = "ENVO_00000428", "ENVO_00010483", "UBERON_0000463"
ALLOWED = {"feature": {"ENVO_01000254", "ENVO_01000813", "UBERON_0000061", "PO_0025131"},
           "material": {MATERIAL, SUBSTANCE, "FOODON_00002403"}}
EXCLUDED = {"feature": {MATERIAL, SUBSTANCE, BIOME}, "material": set()}


def predict(d, slot, labels, train_idx, test_idx):
    S, vocab = prototype_fold_scores(d, slot, train_idx, test_idx, labels=labels)
    return vocab[S.argmax(1)] if len(test_idx) else np.array([], object)


def main():
    ap = parser(__doc__, "draft_slot_maps")
    ap.add_argument("--k", type=int, default=25, help="neighbours for source 2")
    args = ap.parse_args()
    d = Data(args)
    out = {}
    for slot in ["feature", "material"]:
        y = d.y(slot)
        has = y != ""
        A = lambda t: d.anc.get(t, set()) | {t}  # noqa: E731
        ok = np.array([bool(t) and bool(A(t) & ALLOWED[slot]) and not (A(t) & EXCLUDED[slot]) for t in y])
        outside = has & ~ok
        counts = pd.Series(y[outside]).value_counts()
        print(f"== {slot}: outside the allowed classes {outside.sum()} samples ({outside.sum() / has.sum():.1%}), "
              f"{len(counts)} of {len(set(y[has]))} labels", flush=True)
        s1 = np.full(len(y), "", object)
        for tr, te in d.folds:
            s1[te[outside[te]]] = predict(d, slot, y, tr[ok[tr]], te[outside[te]])
        s2 = np.full(len(y), "", object)
        cand, io = np.where(ok)[0], np.where(outside)[0]
        for a in range(0, len(io), 1000):
            b = io[a:a + 1000]
            S = d.X[b] @ d.X[cand].T
            S[d.study[b, None] == d.study[None, cand]] = -9
            nn = np.argsort(-S, 1)[:, :args.k]
            for r, i in enumerate(b):
                s2[i] = pd.Series(y[cand[nn[r]]]).value_counts().index[0]
        rows = []
        for t, n in counts.items():
            idx = np.where(y == t)[0]
            m1, m2 = pd.Series(s1[idx]).value_counts(), pd.Series(s2[idx]).value_counts()
            rows.append({"from_id": t, "from_label": d.label.get(t, t), "n_samples": int(n), "n_studies": len(set(d.study[idx])),
                         "to_id": m1.index[0], "model": d.label.get(m1.index[0]), "model_share": m1.iloc[0] / n,
                         "knn": d.label.get(m2.index[0]), "knn_share": m2.iloc[0] / n, "agree": m1.index[0] == m2.index[0]})
        R = pd.DataFrame(rows)
        tsv = os.path.join(os.path.dirname(path(args.output)), f"draft_map_{slot}.tsv")
        os.makedirs(os.path.dirname(tsv), exist_ok=True)
        R.to_csv(tsv, sep="\t", index=False)
        print(R.drop(columns=["from_id", "to_id"]).head(15).to_string(), "\n  ->", tsv, flush=True)
        mapping = dict(zip(R.from_id, R.to_id))
        y2 = np.array([mapping.get(t, t) for t in y], object)
        res = {"outside_samples": outside.sum() / has.sum(), "outside_labels": len(counts), "labels": len(set(y[has])),
               "sources_agree_labels": int(R.agree.sum()), "sources_agree_samples": R.n_samples[R.agree].sum() / R.n_samples.sum()}
        for name, labels in [("original", y), ("draft map", y2)]:
            pred = np.full(len(y), "", object)
            for tr, te in d.folds:
                pred[te[has[te]]] = predict(d, slot, labels, tr[has[tr]], te[has[te]])
            hit = pred == labels
            u = has & ok
            res[name] = {"top1_all": hit[has].mean(), "top1_untouched": hit[u].mean(),
                         "macro_untouched": pd.Series(hit[u]).groupby(labels[u]).mean().mean(), "labels": len(set(labels[has]))}
            print(slot, name, {k: round(float(v), 3) for k, v in res[name].items()}, flush=True)
        out[slot] = res
    save_json(out, args.output)


if __name__ == "__main__":
    main()
