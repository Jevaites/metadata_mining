#!/usr/bin/env python3
"""
Figure for the cluster-agreement results.

    left    purity@10 against community clusters - keywords vs sub-biomes
    middle  the paired correction: new minus Dany on samples tie-free in BOTH
    right   why the raw numbers mislead - sub-biome vectors are mostly duplicates

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/cluster_agreement_figure.py
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
V = f"{ROOT}/sidequest/latest/embeddings/vs_clusters"
DANY, NEW, INK, FAINT = "#2a78d6", "#eb6834", "#23262b", "#d6d9dd"
CELLS = [("sub_biomes", "fine"), ("sub_biomes", "coarse"),
         ("keywords", "fine"), ("keywords", "coarse")]
NAME = {("sub_biomes", "fine"): "sub-biomes\nfine", ("sub_biomes", "coarse"): "sub-biomes\ncoarse",
        ("keywords", "fine"): "keywords\nfine", ("keywords", "coarse"): "keywords\ncoarse"}


def main():
    ctrl = json.load(open(f"{V}/cluster_purity_control.json"))
    ceil = json.load(open(f"{V}/cluster_purity_ceiling.json"))
    agree = json.load(open(f"{V}/cluster_agreement.json"))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.9), constrained_layout=True)
    x = np.arange(len(CELLS))

    # --- 1. purity@10 -------------------------------------------------------
    d = [ctrl[f"{t}|{g}|dany"]["purity@10_shuffled_order"] for t, g in CELLS]
    n = [ctrl[f"{t}|{g}|new"]["purity@10_shuffled_order"] for t, g in CELLS]
    base = np.mean([agree[f"{t}|{g}|new|embedding"]["baseline"] for t, g in CELLS])
    ax1.bar(x - .19, d, .36, color=DANY, label="Dany  (GPT-3.5 → 3-small 1536d)", zorder=2)
    ax1.bar(x + .19, n, .36, color=NEW, label="new  (GPT-5 → 3-large 1024d)", zorder=2)
    for xi, (a, b) in enumerate(zip(d, n)):
        ax1.annotate(f"{a:.2f}", (xi - .19, a), ha="center", va="bottom", fontsize=8.5, color=INK)
        ax1.annotate(f"{b:.2f}", (xi + .19, b), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax1.axhline(base, color=INK, linewidth=1, linestyle=":", zorder=3)
    ax1.set_xticks(x, [NAME[c] for c in CELLS], fontsize=9)
    ax1.set(ylim=(0, 1.0), ylabel=f"purity@10 — share of the 10 nearest neighbours\n"
        f"in the same cluster  (random-pair baseline {base:.4f})")
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax1.set_title("Metadata embeddings do track community clusters —\n"
                  "and keywords carry far more of that signal than sub-biomes",
                  fontsize=9.5, color=INK)

    # --- 2. the paired correction ------------------------------------------
    p = [ctrl[f"{t}|{g}|paired_tie_free"] for t, g in CELLS]
    diff = [r["diff"] for r in p]
    err = np.array([[r["diff"] - r["ci"][0] for r in p], [r["ci"][1] - r["diff"] for r in p]])
    y = np.arange(len(CELLS))[::-1]
    ax2.barh(y, diff, .5, color=NEW, zorder=2)
    ax2.errorbar(diff, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, zorder=3)
    ax2.axvline(0, color=INK, linewidth=1.1, zorder=3)
    for yi, r in zip(y, p):
        ax2.annotate(f"{r['diff']:+.3f}   (n={r['n']:,})", (r["ci"][1], yi), xytext=(7, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax2.set_yticks(y, [NAME[c].replace("\n", " ") for c in CELLS], fontsize=9)
    ax2.set(xlim=(-0.012, max(r["ci"][1] for r in p) * 1.75),
            xlabel="purity@10:  new − Dany,  95% CI")
    ax2.set_title("Compared like with like, the new run wins everywhere\n"
                  "(samples with <10 identical twins in BOTH runs)", fontsize=9.5, color=INK)
    ax2.grid(axis="x", color=FAINT, linewidth=.7, zorder=0)
    ax2.set_axisbelow(True)

    # --- 3. why the raw numbers mislead ------------------------------------
    cells3 = [("sub_biomes", "fine"), ("keywords", "fine")]
    dd = [ceil[f"{t}|{g}|dany"]["frac_with_ge_k_duplicates"] for t, g in cells3]
    nn = [ceil[f"{t}|{g}|new"]["frac_with_ge_k_duplicates"] for t, g in cells3]
    x3 = np.arange(2)
    ax3.bar(x3 - .19, dd, .36, color=DANY, zorder=2)
    ax3.bar(x3 + .19, nn, .36, color=NEW, zorder=2)
    for xi, (a, b, c) in enumerate(zip(dd, nn, cells3)):
        md = ceil[f"{c[0]}|{c[1]}|dany"]["median_duplicates"]
        mn = ceil[f"{c[0]}|{c[1]}|new"]["median_duplicates"]
        ax3.annotate(f"{a:.0%}\nmedian {md}", (xi - .19, a), ha="center", va="bottom",
                     fontsize=8.5, color=INK)
        ax3.annotate(f"{b:.0%}\nmedian {mn}", (xi + .19, b), ha="center", va="bottom",
                     fontsize=8.5, color=INK)
    ax3.set_xticks(x3, ["sub-biomes", "keywords"], fontsize=9)
    ax3.set(ylim=(0, 1.02),
            ylabel="share of samples with ≥10 identical-vector twins")
    ax3.set_title("83% of new sub-biome samples have ≥10 exact twins:\n"
                  "their neighbourhood is a tie, not a ranking", fontsize=9.5, color=INK)

    for ax in (ax1, ax3):
        ax.grid(axis="y", color=FAINT, linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
    for ax in (ax1, ax2, ax3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5, color=FAINT)
    path = f"{V}/cluster_agreement.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
