#!/usr/bin/env python3
"""
Does the cap of 50 training samples per study matter? (deck slide "evaluation checks"; report §5.1)

The test samples stay fixed (the capped evaluation set, <= 50 per study, folds as 5_evaluate.py); only the
training side changes: at most N samples per training study, N = 5, 10, 20, 50, 100, 200, all. For each
cap: prototype and linear top-1, prototype macro, and the prototype's back-off at 90 % (tau out of fold).

Findings (fold seeds 0-2): top-1 moves by <= 2 points between cap 5 and no cap; macro (all labels) falls as
big studies dominate (feature .094 at cap 5 -> .076 without cap). The cap of 50 is harmless.

python experiments/deck_review/cap_sweep.py --fold_seed 0   (repeat with 1, 2; ~1 h each)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (SLOTS, M, Data, oof_backoff_at, oof_outcomes, parser, save_json,  # noqa: E402
                     select_samples, summarise)

CAPS = [5, 10, 20, 50, 100, 200, 0]  # 0 = no cap


def main():
    ap = parser(__doc__, "cap_sweep")
    args = ap.parse_args()
    args.output = args.output.replace(".json", f"_seed{args.fold_seed}.json")
    test = Data(args)  # the fixed evaluation set and its folds
    fold_of_study = {}
    for k, (_, te) in enumerate(test.folds):
        for s in set(test.study[te]):
            fold_of_study[s] = k
    from common import load_npz
    npz = [load_npz(args.keywords), load_npz(args.sub_biomes)]
    w = 1 / np.sqrt(2)
    results = []
    for cap in CAPS:
        pool = select_samples(args.samples, [set(r) for r, _ in npz], cap)
        pool = pool[pool.study_code.isin(fold_of_study)].reset_index(drop=True)
        Xp = np.hstack([w * v[[r[s] for s in pool.sample_id]] for r, v in npz]).astype(np.float32)
        fold_p = pool.study_code.map(fold_of_study).to_numpy()
        row = {"cap": cap, "n_train_pool": len(pool)}
        for slot in SLOTS:
            yp, yt = pool[slot].to_numpy(), test.y(slot)
            hits, hits_lin, gold, folds = [], [], [], []
            for k, (_, te) in enumerate(test.folds):
                a = (fold_p != k) & (yp != "")
                te = te[yt[te] != ""]
                vocab = np.unique(yp[a])
                model = M.prototype_model(Xp[a], yp[a], test.term_matrix(vocab), vocab, 0.5, 0.1)
                S = M.prototype_scores(test.X[te], model)
                hits += list(vocab[S.argmax(1)] == yt[te])
                cls, L = M.linear_scores(test.X[te], Xp[a], yp[a])
                hits_lin += list(cls[np.asarray(L).argmax(1)] == yt[te])
                gold += list(yt[te])
                folds.append({"S": S, "vocab": vocab, "gold": yt[te]})
            h = pd.Series(hits)
            bo = summarise(oof_backoff_at(oof_outcomes(folds, test.anc))[0])
            row.update({f"{slot}_top1": float(h.mean()), f"{slot}_macro": float(h.groupby(np.array(gold)).mean().mean()),
                        f"{slot}_linear_top1": float(np.mean(hits_lin)), f"{slot}_backoff_answered": bo["coverage"],
                        f"{slot}_backoff_strict": bo["accuracy_strict"]})
        results.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    save_json(results, args.output)


if __name__ == "__main__":
    main()
