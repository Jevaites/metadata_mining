#!/usr/bin/env python3
"""
What does swapping 3-small@1536 for 3-large@1024 actually change?

Uses only the sub-biome strings GPT-5 left IDENTICAL to GPT-3.5 (5,070 distinct
texts). Same input string in both runs, so every difference is the embedding
model. Each text is unique in both spaces, so there are no ties.

Cosines from a 1536d and a 1024d model are not on the same scale, so raw
differences conflate "the model groups these more" with "the whole scale
shifted". Every pair is therefore converted to its PERCENTILE within its own
space, and the statistic is the change in percentile - unit-free.

Pairs are classified by how the two strings relate lexically, which is a proxy
for what varies between them:

    same head noun    'pig gut'        vs 'tick gut'         same part, different host
    same first word   'rice rhizoplane' vs 'rice endosphere'  same host, different part
    some shared word / no shared word                         controls

    python3 scripts/model_effect_same_text.py
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import compare_to_previous_embeddings as C
from compare_to_previous_embeddings import (classify, cosines, fetch, new_unique,
                                            old_embeddings, rows_for, texts_for, unit)

COLORS = {"same head noun": "#2a78d6", "same first word": "#eb6834",
          "some shared word": "#1baf7a", "no shared word": "#52514e"}


def load_same_text(seed):
    same, _ = classify("sub_biomes", False)
    txt = texts_for(f"{C.NEW}/GPT_sub_biomes.txt", set(same), False)
    by_text = {}
    for sid in same:                       # one representative sample per distinct text
        by_text.setdefault(txt.get(sid), sid)
    by_text.pop(None, None)
    texts = sorted(by_text)
    orow = rows_for(old_embeddings("sub_biomes"), "sample_ids", set(by_text.values()))
    trow = rows_for(new_unique("sub_biomes"), "texts", set(texts))
    texts = [t for t in texts if by_text[t] in orow and t in trow]
    O = fetch(old_embeddings("sub_biomes"), [orow[by_text[t]] for t in texts])
    N = fetch(new_unique("sub_biomes"), [trow[t] for t in texts])
    ok = np.isfinite(O).all(axis=1) & np.isfinite(N).all(axis=1)
    texts = [t for t, g in zip(texts, ok) if g]
    return texts, unit(O[ok]), unit(N[ok])


def sample_pairs(texts, n_per_class, seed):
    """-> {class: (i array, j array)}, stratified so each relation is populated."""
    rng = np.random.default_rng(seed)
    toks = [t.lower().split() for t in texts]
    heads, firsts = defaultdict(list), defaultdict(list)
    for i, tk in enumerate(toks):
        if tk:
            heads[tk[-1]].append(i)
            firsts[tk[0]].append(i)

    def within(groups, reject):
        out = []
        keys = [k for k, v in groups.items() if len(v) > 1]
        while len(out) < n_per_class and keys:
            k = keys[rng.integers(0, len(keys))]
            g = groups[k]
            a, b = g[rng.integers(0, len(g))], g[rng.integers(0, len(g))]
            if a != b and not reject(toks[a], toks[b]):
                out.append((a, b))
        return np.array(out)

    classes = {}
    classes["same head noun"] = within(heads, lambda x, y: x == y)
    classes["same first word"] = within(firsts, lambda x, y: x[-1] == y[-1] or x == y)

    n = len(texts)
    shared, none = [], []
    while len(shared) < n_per_class or len(none) < n_per_class:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a == b:
            continue
        sa, sb = set(toks[a]), set(toks[b])
        if sa & sb and toks[a][-1] != toks[b][-1] and toks[a][0] != toks[b][0]:
            if len(shared) < n_per_class:
                shared.append((a, b))
        elif not (sa & sb) and len(none) < n_per_class:
            none.append((a, b))
    classes["some shared word"] = np.array(shared)
    classes["no shared word"] = np.array(none)
    return classes


def percentile_maps(O, N, rng, n_ref=1_000_000):
    """Reference distribution of pairwise cosines in each space, for percentiles."""
    n = len(O)
    i = rng.integers(0, n, n_ref)
    j = rng.integers(0, n, n_ref)
    keep = i != j
    i, j = i[keep], j[keep]
    ref_o = np.sort(cosines(O, i, j))     # chunked: O[i] on 1M pairs is 6 GB
    ref_n = np.sort(cosines(N, i, j))
    return ref_o, ref_n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--n_per_class", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    root = os.path.expanduser(args.root)
    C.OLD, C.NEW = f"{root}/sidequest", f"{root}/sidequest/latest"
    out_dir = f"{C.NEW}/embeddings/vs_previous"

    texts, O, N = load_same_text(args.seed)
    print(f"{len(texts):,} sub-biome strings identical in both runs "
          f"(Dany {O.shape[1]}d, new {N.shape[1]}d)", flush=True)

    rng = np.random.default_rng(args.seed)
    ref_o, ref_n = percentile_maps(O, N, rng)
    print(f"overall pairwise cosine: Dany mean {ref_o.mean():.4f}, new mean {ref_n.mean():.4f} "
          f"(scale shift {ref_n.mean() - ref_o.mean():+.4f} - this is why percentiles are used)",
          flush=True)

    classes = sample_pairs(texts, args.n_per_class, args.seed)
    summary, dists = {}, {}
    print(f"\n{'relation':<18}{'n':>8}{'cos Dany':>10}{'cos new':>9}"
          f"{'pctile Dany':>13}{'pctile new':>12}{'shift':>9}", flush=True)
    for name, pr in classes.items():
        i, j = pr[:, 0], pr[:, 1]
        co, cn = cosines(O, i, j), cosines(N, i, j)
        po = np.searchsorted(ref_o, co) / len(ref_o)
        pn = np.searchsorted(ref_n, cn) / len(ref_n)
        d = pn - po
        se = d.std() / np.sqrt(len(d))
        dists[name] = d
        summary[name] = {"n": int(len(d)), "cos_dany": float(co.mean()), "cos_new": float(cn.mean()),
                         "pct_dany": float(po.mean()), "pct_new": float(pn.mean()),
                         "shift": float(d.mean()), "se": float(se)}
        print(f"{name:<18}{len(d):>8,}{co.mean():>10.3f}{cn.mean():>9.3f}"
              f"{po.mean():>13.3f}{pn.mean():>12.3f}{d.mean():>+9.3f}  (SE {se:.4f})", flush=True)

        order = np.argsort(d)
        for tag, sel in [("new separates most", order[:3]), ("new groups most", order[-3:])]:
            print(f"     {tag}:", flush=True)
            for k in sel:
                print(f"       {d[k]:+.2f} pctile | cos {co[k]:.2f} -> {cn[k]:.2f} | "
                      f"{texts[i[k]]!r} <-> {texts[j[k]]!r}", flush=True)

    json.dump(summary, open(f"{out_dir}/model_effect_same_text.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor="#fcfcfb")
        bins = np.linspace(-0.6, 0.6, 140)
        for name, d in dists.items():
            dens, edges = np.histogram(d, bins=bins, density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, dens, lw=2, color=COLORS[name],
                    label=f"{name}  (mean {d.mean():+.3f})")
        ax.axvline(0, color="#8a8a86", lw=1, ls="--")
        ax.set(xlabel="change in pairwise-similarity percentile  (new − Dany)", ylabel="density")
        ax.set_title("Same input text: what the embedding model alone changes", fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(alpha=0.22, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/model_effect_same_text.png", dpi=150, facecolor="#fcfcfb")
        print(f"\nfigure: {out_dir}/model_effect_same_text.png", flush=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
