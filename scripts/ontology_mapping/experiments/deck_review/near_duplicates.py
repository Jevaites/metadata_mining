#!/usr/bin/env python3
"""
The label ceiling: how consistent are Metalog's curators on near-identical samples, and where do the
model's errors on those samples come from? (deck slide "On near-identical samples, most errors are
curator disagreements"; report §7.4.)

For every evaluation sample S, its most similar sample S2 from ANOTHER study (cosine of the kw + sb
vectors). S2 is a near-duplicate when cosine >= 0.9 (24 % of samples). For each slot:

1. nearest pair (as analyses.py label_ceiling): curators agree = S and S2 share the Metalog label;
   model right = the model's out-of-fold top-1 for S equals S's own label (from cv_backoff predictions);
   model same = the model gives S and S2 the same top-1. Reported by similarity band, for all pairs below
   0.9, and for random cross-study pairs.
2. near-duplicates only: model right when the curators agree / disagree, the share of the model's errors
   that fall on disagreeing pairs, and the best accuracy a model can reach if it gives both samples of a
   pair the same answer: (1 + agree) / 2.
3. robustness: (a) the nearest *labelled* neighbour at >= 0.9 instead of the nearest neighbour;
   (b) all neighbours at >= 0.9 instead of the nearest: all / mixed / none share S's label, agreement as
   the share of neighbours or of their studies; (c) "disagree" only if >= 30 % (or 50 %) of them differ.

Findings (linear, fold seed 0): near-duplicate pairs share the label 74 / 80 / 88 % (biome / feature /
material); the model is right on 89 / 96 / 99.5 % of the agreeing ones and 49 / 27 / 22 % of the others,
so 61 / 81 / 96 % of its errors there are curator disagreements; it gives both samples the same label
88 / 87 / 93 % of the time (more consistent than the curators). Variants (a)-(c) move these by <= 5 points.

python experiments/deck_review/near_duplicates.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SLOTS, Data, cv_predictions, parser, save_json  # noqa: E402

BANDS = [0, .75, .8, .85, .9, 1.01]
BAND_NAMES = ["<0.75", "0.75-0.80", "0.80-0.85", "0.85-0.90", ">=0.90"]


def decompose(agree, hit):
    a = agree.mean()
    err_agree, err_dis = ((~hit) & agree).sum(), ((~hit) & ~agree).sum()
    return {"n": int(len(hit)), "curators_agree": a, "model_right": hit.mean(),
            "model_right_if_agree": hit[agree].mean() if agree.any() else None,
            "model_right_if_disagree": hit[~agree].mean() if (~agree).any() else None,
            "share_of_errors_where_curators_disagree": err_dis / max(err_agree + err_dis, 1),
            "best_possible_one_answer_per_pair": (1 + a) / 2}


def main():
    ap = parser(__doc__, "near_duplicates", needs=["samples", "keywords", "sub_biomes", "predictions"])
    ap.add_argument("--methods", nargs="+", default=["linear", "prototype"])
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()
    d = Data(args)
    X, st, n = d.X, d.study, len(d.X)
    near, nsim, neigh = np.zeros(n, int), np.zeros(n), []
    for a in range(0, n, 2000):
        S = X[a:a + 2000] @ X.T
        S[st[a:a + 2000, None] == st[None, :]] = -1  # other studies only
        near[a:a + 2000], nsim[a:a + 2000] = S.argmax(1), S.max(1)
        for row in S:
            j = np.where(row >= args.threshold)[0]
            neigh.append((j, row[j]))
    dup = nsim >= args.threshold
    print(f"share with a near-duplicate in another study: {dup.mean():.3f}", flush=True)
    band = pd.cut(nsim, BANDS, labels=BAND_NAMES).astype(str)
    rng = np.random.default_rng(0)
    out = {"share_with_near_duplicate": dup.mean(), "threshold": args.threshold}
    for method in args.methods:
        pred = cv_predictions(args.predictions, method)
        for slot in SLOTS:
            g = d.y(slot)
            pr = pred[slot]["pred"].reindex(d.samples["sample_id"]).fillna("").to_numpy()
            hit = (pr == g) & (g != "")
            ok = (g != "") & (g[near] != "") & (pr != "") & (pr[near] != "")
            agree, same = g == g[near], pr == pr[near]
            r = {"bands": {}}
            for b in BAND_NAMES:
                m = ok & (band == b)
                r["bands"][b] = {"n": int(m.sum()), "curators_agree": agree[m].mean(), "model_right": hit[m].mean(),
                                 "model_same_label": same[m].mean()}
            m = ok & ~dup
            r["below_threshold"] = {"n": int(m.sum()), "curators_agree": agree[m].mean(),
                                    "model_right": hit[m].mean(), "model_same_label": same[m].mean()}
            idx = np.where((g != "") & (pr != ""))[0]
            i, j = rng.choice(idx, 200000), rng.choice(idx, 200000)
            k = st[i] != st[j]
            r["random_cross_study_pairs"] = {"curators_agree": (g[i] == g[j])[k].mean(), "model_same_label": (pr[i] == pr[j])[k].mean()}
            m = ok & dup
            r["near_duplicates"] = {**decompose(agree[m], hit[m]), "model_same_label": same[m].mean()}
            # robustness: all labelled neighbours >= threshold
            rows = []
            for s_ in np.where(g != "")[0]:
                j, sim = neigh[s_]
                keep = g[j] != ""
                if not keep.any():
                    continue
                j, sim = j[keep], sim[keep]
                lab = g[j]
                differ = (lab != g[s_]).mean()
                per_study = pd.Series(lab).groupby(st[j]).agg(lambda x: x.value_counts().index[0])
                rows.append((s_, g[j[sim.argmax()]] == g[s_], differ, (per_study == g[s_]).mean(), len(j)))
            R = pd.DataFrame(rows, columns=["i", "nearest_labelled_agree", "share_differ", "study_share_agree", "k"])
            h = hit[R.i.to_numpy()]
            kind = np.where(R.share_differ == 0, "all agree", np.where(R.share_differ == 1, "none agree", "mixed"))
            r["robustness"] = {
                "nearest_labelled_neighbour": decompose(R.nearest_labelled_agree.to_numpy(), h),
                "agreement_share_of_all_neighbours": 1 - R.share_differ.mean(),
                "agreement_share_of_studies": R.study_share_agree.mean(),
                "neighbours_median": float(R.k.median()),
                "kinds": pd.Series(kind).value_counts(normalize=True).to_dict(),
                "model_right_by_kind": pd.Series(h).groupby(kind).mean().to_dict(),
                **{f"disagree_if_{int(t * 100)}pct_differ": decompose((R.share_differ < t).to_numpy(), h) for t in (0.3, 0.5)},
            }
            out[f"{method}|{slot}"] = r
            nd = r["near_duplicates"]
            print(f"{method:9s} {slot:8s} agree {nd['curators_agree']:.3f} right {nd['model_right']:.3f} "
                  f"| agree {nd['model_right_if_agree']:.3f} disagree {nd['model_right_if_disagree']:.3f} "
                  f"| errors from disagreement {nd['share_of_errors_where_curators_disagree']:.3f} "
                  f"| model same {nd['model_same_label']:.3f} | bound {nd['best_possible_one_answer_per_pair']:.3f}", flush=True)
    save_json(out, args.output)


if __name__ == "__main__":
    main()
