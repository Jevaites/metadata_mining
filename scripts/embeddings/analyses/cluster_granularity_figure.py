#!/usr/bin/env python3
"""
Figure for the nested granularity test.

    left    the ladder: how far down the hierarchy metadata still separates
    middle  depth is not uniform - level C across 201 coarse clusters
    right   and it runs BACKWARDS against coherence

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_granularity_figure.py
"""
import json, os

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
V = f"{ROOT}/sidequest/latest/embeddings/vs_clusters"
ACC, ALT, INK, FAINT, MUTED = "#eb6834", "#2a78d6", "#23262b", "#d6d9dd", "#8a8f98"
ORDER = ["A  coarse cluster, whole corpus",
         "B  coarse cluster, within a biome",
         "C  fine cluster, within a coarse cluster"]
SHORT = ["A\ncoarse clusters,\nwhole corpus",
         "B\ncoarse clusters,\nwithin one biome",
         "C\nfine clusters,\nwithin one cluster"]


def main():
    main_g = json.load(open(f"{V}/cluster_granularity_c8m25.json"))
    alt_g = json.load(open(f"{V}/cluster_granularity_c6m15.json"))
    coh = {r["cluster"]: r for r in json.load(open(f"{V}/cluster_coherence_coarse.json"))["clusters"]}
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 5.0), constrained_layout=True)

    # --- 1. the ladder ------------------------------------------------------
    x = np.arange(3)
    p = [main_g["levels"][k]["purity_mean"] for k in ORDER]
    sd = [main_g["levels"][k]["purity_sd"] for k in ORDER]
    a = [main_g["levels"][k]["auc_mean"] for k in ORDER]
    aa = [alt_g["levels"][k]["auc_mean"] for k in ORDER]
    base = main_g["baseline"]
    ax1.bar(x, p, .52, yerr=sd, color=ACC, capsize=5,
            error_kw=dict(ecolor=INK, elinewidth=1.2), zorder=2)
    ax1.plot(x, a, "o-", color=ALT, linewidth=2, markersize=8, zorder=4)
    ax1.plot(x, aa, "o--", color=ALT, linewidth=1.3, markersize=6, alpha=.55, zorder=4)
    ax1.annotate("separation AUC", (0, a[0]), xytext=(10, 8), textcoords="offset points",
                 fontsize=9, color=ALT, fontweight="bold")
    ax1.annotate("dashed: same test at 6 classes × 15\n(robustness — AUC has no baseline)",
                 (2, aa[2]), xytext=(-8, -30), textcoords="offset points", ha="right",
                 fontsize=8, color=ALT)
    ax1.axhline(base, color=INK, linewidth=1.1, linestyle=":", zorder=3)
    for xi, (v, av) in enumerate(zip(p, a)):
        ax1.annotate(f"{v:.2f}", (xi, v), xytext=(0, -16), textcoords="offset points",
                     ha="center", fontsize=9.5, color="white", fontweight="bold")
    ax1.annotate(f"random baseline {base:.2f}", (1.5, base), xytext=(0, 6),
                 textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax1.set_xticks(x, SHORT, fontsize=8.5)
    ax1.set(ylim=(0, 1.02), ylabel="purity@10 (bars)   /   separation AUC (line)")
    ax1.set_title("Every level is 8 classes × 25 samples, so the baseline is\n"
                  "the same: the drop is depth, not more classes", fontsize=9.5, color=INK)

    # --- 2. level C is not uniform ------------------------------------------
    lc = [k for k in alt_g["levels"] if k.startswith("C ")][0]
    pc = np.array(alt_g["levels"][lc]["purity_per_context"])
    bc = alt_g["levels"][lc]["baseline"]
    ax2.hist(pc, bins=32, color=ACC, zorder=2)
    ax2.axvline(bc, color=INK, linewidth=1.2, linestyle=":", zorder=3)
    top = ax2.get_ylim()[1]
    ax2.annotate(f"baseline\n{bc:.2f}", (bc, top * .98), xytext=(6, 0), textcoords="offset points",
                 ha="left", va="top", fontsize=8.5, color=INK)
    ax2.annotate(f"{np.mean(pc < 0.25):.0%} of coarse clusters have\n"
                 f"essentially no resolvable substructure", (0.97, 0.72),
                 xycoords="axes fraction", ha="right", fontsize=8.5, color=INK)
    ax2.annotate(f"{np.mean(pc > 0.70):.0%} resolve\nalmost fully", (0.97, 0.45),
                 xycoords="axes fraction", ha="right", fontsize=8.5, color=INK)
    ax2.set(xlabel="purity@10 inside one coarse cluster", ylabel=f"coarse clusters (n={len(pc)})")
    ax2.set_title("How deep you can go is cluster-specific —\n"
                  "median 0.44, but the spread runs baseline to 1.0", fontsize=9.5, color=INK)

    # --- 3. and it runs backwards against coherence -------------------------
    ctx = alt_g["levels"][lc]["context"]
    keep = [i for i, c in enumerate(ctx) if c in coh]
    co = np.array([coh[ctx[i]]["coherence"] for i in keep])
    pv = pc[keep]
    rho = spearmanr(pv, co)[0]
    ax3.scatter(co, pv, s=26, color=ACC, alpha=.55, linewidths=0, zorder=2)
    z = np.polyfit(co, pv, 1)
    xs = np.linspace(co.min(), co.max(), 50)
    ax3.plot(xs, np.polyval(z, xs), color=INK, linewidth=1.6, zorder=3)
    ax3.axhline(bc, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    o = sorted(keep, key=lambda i: -pc[i])
    for i, (dx, dy, ha) in zip([o[0], o[1], o[-2], o[-1]],
                               [(8, 2, "left"), (8, -10, "left"), (-8, 6, "right"), (-8, -4, "right")]):
        r = coh[ctx[i]]
        ax3.annotate(" ".join(r["modal_text"].split()[:2]), (r["coherence"], pc[i]),
                     xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK)
    ax3.set(xlabel="coherence of the coarse cluster (metadata uniformity)",
            ylabel="purity@10 of its fine structure")
    ax3.annotate(f"Spearman {rho:+.2f}", (0.03, 0.95), xycoords="axes fraction",
                 fontsize=9, color=INK, fontweight="bold")
    ax3.set_title("The more uniform a cluster's metadata, the LESS\n"
                  "of its internal structure the text can recover", fontsize=9.5, color=INK)

    for ax in (ax1, ax2, ax3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5, color=FAINT)
        ax.grid(axis="y", color=FAINT, linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
    path = f"{V}/cluster_granularity.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
