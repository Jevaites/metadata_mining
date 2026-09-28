#!/usr/bin/env python3
"""Figure for the study-blocking test."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
V = f"{ROOT}/sidequest/latest/embeddings/vs_clusters"
KW, SB, INK, FAINT = "#eb6834", "#2a78d6", "#23262b", "#d6d9dd"


def main():
    d = json.load(open(f"{V}/cluster_study_blocked.json"))
    base = d["exact_baseline"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)

    # --- study identification ---------------------------------------------
    vals = [d["keywords"]["study10"], d["sub_biomes"]["study10"]]
    ax1.bar([0, 1], vals, .5, color=[KW, SB], zorder=2)
    for x, v in zip([0, 1], vals):
        ax1.annotate(f"{v:.0%}", (x, v), xytext=(0, 5), textcoords="offset points",
                     ha="center", fontsize=11, color=INK, fontweight="bold")
    ax1.set_xticks([0, 1], ["keywords", "sub-biomes"], fontsize=10)
    ax1.set(ylim=(0, 1.0), ylabel="share of the 10 nearest neighbours from the SAME study")
    ax1.set_title("Keyword embeddings are largely a study fingerprint\n"
                  f"difference {d['diff_study10']['mean']:+.2f} "
                  f"[{d['diff_study10']['ci'][0]:+.2f}, {d['diff_study10']['ci'][1]:+.2f}]",
                  fontsize=9.5, color=INK)

    # --- cluster purity, plain vs study-blocked ----------------------------
    x = np.arange(2)
    plain = [d["keywords"]["purity"], d["sub_biomes"]["purity"]]
    blocked = [d["keywords"]["purity_blocked"], d["sub_biomes"]["purity_blocked"]]
    ax2.bar(x - .19, plain, .36, color=[KW, SB], zorder=2)
    ax2.bar(x + .19, blocked, .36, color=[KW, SB], alpha=.45, zorder=2,
            hatch="///", edgecolor="white", linewidth=0)
    for xi, (a, b) in enumerate(zip(plain, blocked)):
        ax2.annotate(f"{a:.3f}", (xi - .19, a), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, color=INK)
        ax2.annotate(f"{b:.3f}", (xi + .19, b), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, color=INK)
    ax2.axhline(base, color=INK, linewidth=1.1, linestyle=":", zorder=3)
    ax2.annotate(f"random baseline {base:.3f}", (1.45, base), xytext=(0, 5),
                 textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax2.set_xticks(x, ["keywords", "sub-biomes"], fontsize=10)
    ax2.set(ylim=(0, max(plain) * 1.35), ylabel="cluster purity@10")
    dp, db = d["diff_purity10"], d["diff_purity10_blocked"]
    ax2.annotate("solid: all neighbours\nhatched: same-study neighbours removed",
                 (0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                 fontsize=8.5, color=INK)
    ax2.set_title(f"The keyword advantage mostly goes with it:\n"
                  f"{dp['mean']:+.3f} → {db['mean']:+.3f} "
                  f"[{db['ci'][0]:+.3f}, {db['ci'][1]:+.3f}]", fontsize=9.5, color=INK)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=FAINT, linewidth=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5, color=FAINT)
    p = f"{V}/cluster_study_blocked.png"
    fig.savefig(p, dpi=150, facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
