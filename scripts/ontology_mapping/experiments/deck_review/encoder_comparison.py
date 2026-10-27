#!/usr/bin/env python3
"""
Did the new embedding run help? GPT-3.5 text + text-embedding-3-small (the previous run) against GPT-5.1
text + text-embedding-3-large (current), on the same samples and folds, with the linear (ridge) model
(deck slide "input text"; report §7.3).

Arms: keywords, sub-biome, kw + sb, for each run. Samples: the evaluation set restricted to samples with
vectors in both runs. Paired difference of kw + sb with a study bootstrap (2,000 resamples).

Findings (17,539 samples, fold seed 0): kw + sb top-1 .601 / .610 / .708 (previous) -> .668 / .622 / .712
(current): +6.7 points on biome, small on feature and material.

python experiments/deck_review/encoder_comparison.py \
    --old_keywords <previous run keywords .npz> --old_sub_biomes <previous run sub-biomes .npz>
(the .npz need `sample_ids` and `vectors`, one row per sample; default: the extract in _claude_tmp)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SLOTS, M, Data, parser, path, save_json, unit  # noqa: E402


def main():
    ap = parser(__doc__, "encoder_comparison", needs=["samples", "keywords", "sub_biomes"])
    ap.add_argument("--old_keywords", default="~/MicrobeAtlasProject/_claude_tmp/dany_kw.npz")
    ap.add_argument("--old_sub_biomes", default="~/MicrobeAtlasProject/_claude_tmp/dany_sb.npz")
    args = ap.parse_args()
    d = Data(args)
    old = {}
    for tag, p in [("kw", args.old_keywords), ("sb", args.old_sub_biomes)]:
        z = np.load(path(p), allow_pickle=True)
        old[tag] = dict(zip(z["sample_ids"], unit(np.nan_to_num(z["vectors"].astype(np.float32)))))
    keep = d.samples.sample_id.isin(old["kw"]).to_numpy() & d.samples.sample_id.isin(old["sb"]).to_numpy()
    ids = d.samples.sample_id.to_numpy()[keep]
    fold = np.zeros(len(d.X), int)
    for k, (_, te) in enumerate(d.folds):
        fold[te] = k
    fold = fold[keep]
    h = d.X.shape[1] // 2
    new_kw, new_sb = d.X[keep, :h] * np.sqrt(2), d.X[keep, h:] * np.sqrt(2)
    old_kw, old_sb = np.vstack([old["kw"][s] for s in ids]), np.vstack([old["sb"][s] for s in ids])
    w = 1 / np.sqrt(2)
    arms = {"previous: keywords": old_kw, "previous: sub-biome": old_sb, "previous: kw + sb": np.hstack([w * old_kw, w * old_sb]),
            "current: keywords": new_kw, "current: sub-biome": new_sb, "current: kw + sb": np.hstack([w * new_kw, w * new_sb])}
    print(f"{keep.sum()} samples with both runs, {len(set(d.study[keep]))} studies", flush=True)
    res, hits = {}, {}
    for name, X in arms.items():
        res[name] = {}
        for slot in SLOTS:
            y = d.y(slot)[keep]
            hit = np.zeros(len(y), bool)
            for k in range(5):
                tr, te = (fold != k) & (y != ""), (fold == k) & (y != "")
                cls, L = M.linear_scores(X[te], X[tr], y[tr])
                hit[te] = cls[np.asarray(L).argmax(1)] == y[te]
            m = y != ""
            hits[(name, slot)] = hit
            res[name][slot] = {"top1": float(hit[m].mean()), "macro": float(pd.Series(hit[m]).groupby(y[m]).mean().mean())}
        print(name, {s: round(v["top1"], 3) for s, v in res[name].items()}, flush=True)
    rng = np.random.default_rng(0)
    st = d.study[keep]
    for slot in SLOTS:
        m = d.y(slot)[keep] != ""
        g = pd.DataFrame({"s": st[m], "a": hits[("previous: kw + sb", slot)][m], "b": hits[("current: kw + sb", slot)][m]}).groupby("s").sum()
        g["n"] = pd.Series(st[m]).value_counts()
        diff = (g.b.sum() - g.a.sum()) / g.n.sum()
        bs = [(x.b.sum() - x.a.sum()) / x.n.sum() for x in (g.iloc[rng.integers(0, len(g), len(g))] for _ in range(2000))]
        res[f"diff|{slot}"] = {"points": 100 * diff, "ci95": list(100 * np.percentile(bs, [2.5, 97.5]))}
        print(slot, "current - previous (kw + sb):", round(100 * diff, 1), np.round(100 * np.percentile(bs, [2.5, 97.5]), 1))
    save_json(res, args.output)


if __name__ == "__main__":
    main()
