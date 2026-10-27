#!/usr/bin/env python3
"""
Two model variants asked about in the deck review (report §8.2 and §4.6):

balance   Does class balancing pay off once back-off is counted? Biome, fold seeds 0-2: linear (ridge) with
          and without class_weight='balanced', and the prototype with label-prior weight beta 0.1
          (production), 0.05 and 0. Top-1, macro (all labels; learnable = used by >= 2 studies) and the
          back-off at 90 % (answered share, strict accuracy, exact share).
centroids Is one centroid per (study, label) better than knn_study? Vote of the k nearest centroids
          (k = 1, 3, 5, 10, 25, 50) against knn_study (k = 50), fold seed 0, all slots.

Findings: balance — top-1 .664 / .666 / .647 / .643 / .603 (linear, linear balanced, prototype beta 0.1 /
0.05 / 0); learnable macro .42 / .52 / .43 / .46 / .58; answered at 90 % 66 / 51 / 77 / 70 / 41 %: balancing's
macro gain costs about a quarter of the answered biomes. centroids — k 1 / 5 / 10: biome .623 / .651 / .659,
feature .602 / .634 / .638, material .660 / .713 / .704 vs knn_study .682 / .630 / .706: no gain.

python experiments/deck_review/balance_and_study_centroids.py balance
python experiments/deck_review/balance_and_study_centroids.py centroids
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (SLOTS, M, Data, oof_backoff_at, oof_outcomes, parser, save_json,  # noqa: E402
                     study_folds, summarise, unit)


def centroid_vote(Xte, Xtr, ytr, st_tr, k):
    key = pd.Series(list(zip(st_tr, ytr)))
    groups = key.groupby(key).indices
    keys = list(groups)
    C = unit(np.vstack([Xtr[groups[g]].mean(0) for g in keys]))
    lab = np.array([g[1] for g in keys])
    idx = np.argsort(-(Xte @ C.T), axis=1)[:, :k]
    pred = []
    for row in idx:
        v = defaultdict(float)
        for j in row:
            v[lab[j]] += 1
        best = max(v.values())
        tied = {l for l in v if v[l] == best}
        pred.append(next(lab[j] for j in row if lab[j] in tied))  # tie: the nearest centroid among the tied labels
    return np.array(pred)


def main():
    ap = parser(__doc__, "balance_and_study_centroids")
    ap.add_argument("mode", choices=["balance", "centroids"])
    args = ap.parse_args()
    args.output = args.output.replace(".json", f"_{args.mode}.json")
    from sklearn.linear_model import RidgeClassifier
    d = Data(args)
    res = []
    seeds = (0, 1, 2) if args.mode == "balance" else (args.fold_seed,)
    for seed in seeds:
        folds = list(study_folds(d.samples.study_code, 5, seed))
        for slot in (["biome"] if args.mode == "balance" else SLOTS):
            y = d.y(slot)
            has = y != ""
            nst = pd.Series(d.study[has]).groupby(y[has]).nunique()
            learnable = set(nst[nst >= 2].index)
            variants = (["linear", "linear_balanced", "prototype_beta0.1", "prototype_beta0.05", "prototype_beta0"]
                        if args.mode == "balance" else ["knn_study50", "centroids1", "centroids3", "centroids5",
                                                        "centroids10", "centroids25", "centroids50"])
            for name in variants:
                hits, golds, fl = [], [], []
                for tr, te in folds:
                    tr, te = tr[has[tr]], te[has[te]]
                    if name.startswith("linear"):
                        clf = RidgeClassifier(alpha=1.0, class_weight="balanced" if "balanced" in name else None).fit(d.X[tr], y[tr])
                        S = clf.decision_function(d.X[te])
                        vocab = clf.classes_
                        pred = vocab[S.argmax(1)]
                        fl.append({"S": S, "vocab": vocab, "gold": y[te]})
                    elif name.startswith("prototype"):
                        vocab = np.unique(y[tr])
                        model = M.prototype_model(d.X[tr], y[tr], d.term_matrix(vocab), vocab, 0.5, float(name.split("beta")[1]))
                        S = M.prototype_scores(d.X[te], model)
                        pred = vocab[S.argmax(1)]
                        fl.append({"S": S, "vocab": vocab, "gold": y[te]})
                    elif name == "knn_study50":
                        ranked, _ = M.knn_study(d.X[te], d.X[tr], y[tr], d.study[tr], 50)
                        pred = np.array([r[0] for r in ranked])
                    else:
                        pred = centroid_vote(d.X[te], d.X[tr], y[tr], d.study[tr], int(name.replace("centroids", "")))
                    hits += list(pred == y[te])
                    golds += list(y[te])
                h, g = pd.Series(hits), np.array(golds)
                L = np.isin(g, list(learnable))
                row = {"seed": seed, "slot": slot, "variant": name, "top1": float(h.mean()), "macro": float(h.groupby(g).mean().mean()),
                       "macro_learnable": float(h[L].groupby(g[L]).mean().mean())}
                if fl:
                    s = summarise(oof_backoff_at(oof_outcomes(fl, d.anc))[0])
                    row.update(backoff_answered=s["coverage"], backoff_strict=s["accuracy_strict"], backoff_exact=s["exact"])
                res.append(row)
                print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    save_json(res, args.output)


if __name__ == "__main__":
    main()
