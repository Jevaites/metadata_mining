#!/usr/bin/env python3
"""
One figure summarising the new-vs-Dany comparison, for sharing.

Row 1  pairwise cosine agreement per arm (hexbin + Pearson) - global structure
Row 2  left : top-k neighbour agreement at full scale       - local structure
       right: 5-NN accuracy against curated gold_dict       - which is better

    python3 scripts/summary_figure.py
"""

import argparse
import json
import os

import numpy as np

import compare_to_previous_embeddings as C
from compare_to_previous_embeddings import (classify, fetch, geometry, new_unique,
                                            old_embeddings, rows_for, texts_for, unit)

DANY, NEW, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
ARMS = [("sub_biomes", "same", "sub-biomes\nsame text (20%)"),
        ("sub_biomes", "changed", "sub-biomes\nchanged text (80%)"),
        ("keywords", "all", "keywords\n~all text changed")]


def arm_pairs(target, which, n_samples, n_pairs, seed):
    is_kw = target == "keywords"
    same, changed = classify(target, is_kw)
    pool = sorted({"same": same, "changed": changed, "all": same + changed}[which])
    rng = np.random.default_rng(seed)
    ids = sorted(np.array(pool)[rng.choice(len(pool), min(n_samples, len(pool)), replace=False)])
    new_txt = texts_for(f"{C.NEW}/GPT_{target}.txt", set(ids), is_kw)
    orow = rows_for(old_embeddings(target), "sample_ids", set(ids))
    trow = rows_for(new_unique(target), "texts", set(new_txt.values()))
    ids = [s for s in ids if s in orow and new_txt.get(s) in trow]
    O = fetch(old_embeddings(target), [orow[s] for s in ids])
    N = fetch(new_unique(target), [trow[new_txt[s]] for s in ids])
    ok = np.isfinite(O).all(axis=1) & np.isfinite(N).all(axis=1)
    _, _, a, b, st = geometry(unit(O[ok]), unit(N[ok]), np.random.default_rng(seed), n_pairs)
    return a, b, st["pearson"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--n_samples", type=int, default=12000)
    p.add_argument("--n_pairs", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = os.path.expanduser(args.root)
    C.OLD, C.NEW = f"{root}/sidequest", f"{root}/sidequest/latest"
    out_dir = f"{C.NEW}/embeddings/vs_previous"

    nof = json.load(open(f"{out_dir}/neighbour_overlap_full.json"))
    # gold_dict 5-NN accuracy, measured (McNemar in brackets)
    gold = [("sub-biomes, same text", 0.8350, 0.8400, "n=200, not significant"),
            ("sub-biomes, changed text", 0.7975, 0.8540, "n=815, p<0.001"),
            ("keywords", 0.7883, 0.8630, "n=1044, p<0.001")]

    pairs = []
    for target, which, label in ARMS:
        print(f"computing {label!r} ...", flush=True)
        pairs.append((label, *arm_pairs(target, which, args.n_samples, args.n_pairs, args.seed)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 8.6), facecolor=SURFACE)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 0.95], hspace=0.42, wspace=0.55)

    # ---- row 1: global structure ------------------------------------------
    for col, (label, a, b, r) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        ax.set_facecolor(SURFACE)
        ax.hexbin(a, b, gridsize=64, bins="log", cmap="Blues", mincnt=1, linewidths=0)
        lim = [-0.05, 1.03]
        ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--", zorder=3)
        ax.set(xlim=lim, ylim=lim)
        ax.set_xlabel("cosine — Dany", fontsize=9, color=MUTED)
        if col == 0:
            ax.set_ylabel("cosine — new", fontsize=9, color=MUTED)
        ax.set_title(label, fontsize=10.5, color=INK, pad=16)
        ax.text(0.04, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.5, 0.965, "Global structure survives when the text is unchanged — and only then",
             ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # ---- row 2 left: local structure --------------------------------------
    ax = fig.add_subplot(gs[1, 0:3])
    ax.set_facecolor(SURFACE)
    ks = [10, 20, 50]
    vals = [nof["keywords"]["overlap"][str(k)]["all"] for k in ks]
    bars = ax.bar([f"top-{k}" for k in ks], vals, width=0.5, color=DANY, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(-0.42, 1.03, "identical spaces would be 100%", fontsize=8.5, color=MUTED,
            va="bottom", ha="left")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("shared nearest neighbours", fontsize=9, color=MUTED)
    ax.set_title("…but locally the spaces barely agree\nkeywords, 1.25M points",
                 fontsize=10.5, color=INK, loc="left")
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.grid(axis="y", alpha=0.2, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- row 2 right: quality vs curated gold labels ----------------------
    ax = fig.add_subplot(gs[1, 3:6])
    ax.set_facecolor(SURFACE)
    y = np.arange(len(gold))[::-1]
    for yi, (label, d, n, note) in zip(y, gold):
        ax.plot([d, n], [yi, yi], color=MUTED, lw=1.5, zorder=1, alpha=0.5)
        ax.scatter([d], [yi], s=95, color=DANY, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([n], [yi], s=95, color=NEW, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(n + 0.005, yi + 0.02, f"+{(n - d) * 100:.1f} pp", fontsize=10.5,
                fontweight="bold", color=INK, va="center")
        ax.text(n + 0.005, yi - 0.27, note, fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([g[0] for g in gold], fontsize=9.5, color=INK)
    ax.set_xlim(0.755, 0.945)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlabel("5-NN accuracy vs curated gold_dict", fontsize=9, color=MUTED)
    ax.set_title("Quality only improves where GPT-5 changed the text",
                 fontsize=10.5, color=INK, loc="left", pad=16)
    ax.scatter([], [], s=95, color=DANY, label="Dany (GPT-3.5 · 3-small)")
    ax.scatter([], [], s=95, color=NEW, label="new (GPT-5 · 3-large)")
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(-0.01, 1.02))
    ax.tick_params(labelsize=9, colors=MUTED)
    ax.grid(axis="x", alpha=0.2, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    path = os.path.join(out_dir, "summary_for_slack.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwritten: {path}", flush=True)


if __name__ == "__main__":
    main()
