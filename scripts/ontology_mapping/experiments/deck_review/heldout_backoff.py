#!/usr/bin/env python3
"""
How the production back-off behaves on held-out Metalog samples (deck slides "Back-off trades some exact
answers for 3-6x fewer errors", "How confident is the model", "atlas results by habitat"; report §7.5).

Re-runs the production prototype (alpha 0.5, beta 0.1) with study-grouped 5-fold CV, calibrates it and
applies the hierarchical back-off with tau chosen out of fold for 90 % strict accuracy (hierarchy.py), then:
1. answer everything (top-1 for every sample) vs back-off: exact / correct broader / wrong / abstain, and
   where the correct and the wrong top-1 answers end up after back-off;
2. a flat cut-off at the same 90 % (answer the top-1 only if p >= t, else abstain);
3. reliability: top-1 accuracy per probability bin, median probability;
4. by number of training studies of the gold label (why macro is low);
5. by GPT coarse biome (animal, water, soil, plant, other) and by density (similarity to the nearest
   training sample, quartiles): answered share and strict accuracy, and whether a tau per group helps.

Findings (fold seed 0): answering everything gives exact .653 / .627 / .726 and wrong .244 / .361 / .242
(biome / feature / material); back-off answers .792 / .595 / .808 with exact .481 / .502 / .672, correct
broader .230 / .035 / .057, wrong .081 / .057 / .078. Biome: of the correct top-1 answers 65 % stay exact,
20 % climb, 15 % abstain; of the wrong ones 17 % become exact (the curator's term was the broader one),
28 % correct broader, 31 % abstain, 23 % stay wrong. Flat cut-off at 90 %: biome answers 50 %. p >= 0.9 is
right 94-97 % of the time; 0.5-0.6 about half. Per-group tau (density or habitat) does not help.

python experiments/deck_review/heldout_backoff.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (SLOTS, TAUS, Data, choose_tau, oof_backoff_at, oof_outcomes, parser, path,  # noqa: E402
                     prototype_fold_scores, save_json, summarise)

EXACT, COARSER = 1, 2


def shares(o):
    return {"exact": float((o == 1).mean()), "correct_broader": float((o == 2).mean()), "wrong": float(np.isin(o, [3, 4]).mean()),
            "abstain": float((o == 0).mean())}


def curve(outs):
    return [{"tau": float(t), **summarise(outs[i])} for i, t in enumerate(TAUS)]


def group_tau(outs_all, fold_id, group, target=0.9, min_n=20):
    """tau chosen per group (on the other folds of that group) -> outcomes, taus."""
    res = np.zeros(outs_all.shape[1], int)
    taus = {}
    for g in sorted(set(group)):
        for k in sorted(set(fold_id)):
            o = (fold_id != k) & (group == g)
            tau = choose_tau(curve(outs_all[:, o]), target) if o.sum() > min_n else 1.0
            taus.setdefault(str(g), []).append(tau)
            sel = (fold_id == k) & (group == g)
            res[sel] = outs_all[list(TAUS).index(tau), sel]
    return res, taus


def main():
    ap = parser(__doc__, "heldout_backoff")
    ap.add_argument("--gpt_biomes", default="~/MicrobeAtlasProject/sidequest/latest/GPT_biomes.txt",
                    help="sample_id <tab> GPT coarse biome (no header)")
    ap.add_argument("--target", type=float, default=0.9)
    args = ap.parse_args()
    d = Data(args)
    gb = pd.read_csv(path(args.gpt_biomes), sep="\t", header=None, names=["sample_id", "gpt"], dtype=str, quoting=3)
    gpt_of = dict(zip(gb.sample_id, gb.gpt))
    out = {}
    for slot in SLOTS:
        y = d.y(slot)
        folds, meta = [], []
        for k, (tr, te) in enumerate(d.folds):
            tr, te = tr[y[tr] != ""], te[y[te] != ""]
            S, vocab = prototype_fold_scores(d, slot, tr, te)
            folds.append({"S": S, "vocab": vocab, "gold": y[te]})
            nstud = pd.Series(d.study[tr]).groupby(y[tr]).nunique().to_dict()
            meta.append(pd.DataFrame({"i": te, "fold": k, "nn_sim": (d.X[te] @ d.X[tr].T).max(1),
                                      "n_train_studies": [nstud.get(g, 0) for g in y[te]]}))
        outs = oof_outcomes(folds, d.anc)
        bo, taus = oof_backoff_at(outs, args.target)
        meta = pd.concat(meta, ignore_index=True)
        p1 = np.concatenate([f["p1"] for f in folds])
        t1 = np.concatenate([o[0] for o in outs])  # tau 0.05: the plain top-1 for (almost) every sample
        fold_id = meta.fold.to_numpy()
        ans = bo > 0
        r = {"n": len(bo), "taus": taus, "answer_everything": shares(t1), "backoff": shares(bo),
             "backoff_accuracy_strict": np.isin(bo[ans], [1, 2]).mean(),
             "flows": {"of_correct_top1": shares(bo[t1 == EXACT]), "of_wrong_top1": shares(bo[t1 != EXACT])}}
        # flat cut-off at the same target, threshold chosen on the other folds
        flat = np.zeros(len(bo), int)
        for k in sorted(set(fold_id)):
            o, thr = fold_id != k, 1.01
            for t in np.round(np.arange(0, 1.0001, 0.005), 3):
                m = o & (p1 >= t)
                if m.sum() and np.isin(t1[m], [1, 2]).mean() >= args.target:
                    thr = t
                    break
            sel = (fold_id == k) & (p1 >= thr)
            flat[sel] = t1[sel]
        r["flat_cutoff"] = {**shares(flat), "answered": (flat > 0).mean()}
        r["reliability"] = [{"p_bin": f"{lo:.1f}-{lo + .1:.1f}", "n": int(m.sum()), "top1_right": (t1[m] == 1).mean() if m.any() else None}
                            for lo in np.arange(0, 1, 0.1) for m in [(p1 >= lo) & (p1 < lo + 0.1 + 1e-9 * (lo > 0.85))]]
        r["p1_median"], r["p1_median_answered"] = float(np.median(p1)), float(np.median(p1[ans]))
        b = pd.cut(meta.n_train_studies, [-1, 0, 1, 2, 4, 9, 19, 1e9], labels=["0", "1", "2", "3-4", "5-9", "10-19", "20+"])
        gold = y[meta.i.to_numpy()]
        df = pd.DataFrame({"b": b, "hit": t1 == 1, "gold": gold})
        r["by_train_studies"] = df.groupby("b", observed=False).agg(n=("hit", "size"), top1=("hit", "mean"), labels=("gold", "nunique")).reset_index().to_dict("records")
        outs_all = np.concatenate(outs, axis=1)
        for name, group in [("gpt_coarse_biome", np.array([gpt_of.get(s, "NA") for s in d.samples.sample_id.to_numpy()[meta.i]])),
                            ("density_quartile", np.digitize(meta.nn_sim, np.quantile(meta.nn_sim, [.25, .5, .75])))]:
            tab = []
            gtau, gt = group_tau(outs_all, fold_id, group, args.target)
            for g in sorted(set(group)):
                m = group == g
                if m.sum() < 30:
                    continue
                tab.append({"group": str(g), "n": int(m.sum()), "top1": (t1[m] == 1).mean(), "answered": ans[m].mean(),
                            "strict": np.isin(bo[m & ans], [1, 2]).mean() if (m & ans).any() else None,
                            "answered_group_tau": (gtau[m] > 0).mean(),
                            "strict_group_tau": np.isin(gtau[m & (gtau > 0)], [1, 2]).mean() if (m & (gtau > 0)).any() else None})
            r[f"by_{name}"] = tab
            r[f"group_tau_{name}"] = {"answered": (gtau > 0).mean(), "strict": np.isin(gtau[gtau > 0], [1, 2]).mean(), "taus": gt}
        out[slot] = r
        print(f"== {slot}: taus {taus}")
        print("  answer everything", {k: round(v, 3) for k, v in r["answer_everything"].items()})
        print("  back-off         ", {k: round(v, 3) for k, v in r["backoff"].items()}, "strict", round(r["backoff_accuracy_strict"], 3))
        print("  flat cut-off     ", {k: round(v, 3) for k, v in r["flat_cutoff"].items()})
        print("  correct top-1 ->", {k: round(v, 3) for k, v in r["flows"]["of_correct_top1"].items()},
              " wrong top-1 ->", {k: round(v, 3) for k, v in r["flows"]["of_wrong_top1"].items()}, flush=True)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
