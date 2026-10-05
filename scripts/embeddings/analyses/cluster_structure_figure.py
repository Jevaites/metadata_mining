#!/usr/bin/env python3
"""
Figure for the per-cluster coherence and cross-cluster adjacency results.

    left    how tight each community cluster is in metadata space
    middle  every cluster's nearest other cluster, against the all-pairs null
    right   the mirror - how many clusters one string covers, by field

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_structure_figure.py
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
V = f"{ROOT}/sidequest/latest/embeddings/vs_clusters"
SUB, KW, INK, FAINT, MUTED = "#2a78d6", "#eb6834", "#23262b", "#d6d9dd", "#8a8f98"


def main():
    coh = json.load(open(f"{V}/cluster_coherence_coarse.json"))
    adj = json.load(open(f"{V}/cluster_adjacency_coarse.json"))
    near = np.load(f"{V}/cluster_nearest_coarse.npz")
    z = np.load(f"{V}/cluster_centroids_coarse.npz", allow_pickle=True)
    recs = coh["clusters"]
    co = np.array([r["coherence"] for r in recs])
    g = coh["global_cos"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=True)

    # --- 1. coherence -------------------------------------------------------
    ax1.hist(co, bins=60, color=KW, zorder=2)
    ax1.axvline(g, color=INK, linewidth=1.3, linestyle="--", zorder=3)
    ax1.axvline(np.median(co), color=MUTED, linewidth=1.3, linestyle=":", zorder=3)
    top = ax1.get_ylim()[1]
    ax1.annotate(f"two random\nsamples {g:.2f}", (g, top * .99), xytext=(6, 0),
                 textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK)
    ax1.annotate(f"median cluster {np.median(co):.2f}", (np.median(co), top * .99), xytext=(-6, 0),
                 textcoords="offset points", ha="right", va="top", fontsize=8.5, color=MUTED)
    low = (co < 0.60).mean()
    s_low = sum(r["members"] for r in recs if r["coherence"] < 0.60)
    s_all = sum(r["members"] for r in recs)
    ax1.axvspan(co.min(), 0.60, color=INK, alpha=.05, zorder=1)
    ax1.annotate(f"{low:.0%} of clusters\n({s_low / s_all:.0%} of samples)\nbelow 0.60",
                 (0.415, top * .62), ha="left", fontsize=8.5, color=INK)
    ax1.set(xlabel="mean pairwise cosine inside the cluster (keywords, 3-large 1024d)",
            ylabel=f"clusters  (n={len(recs):,})")
    ax1.set_title("Most clusters are far tighter in metadata space\n"
                  "than two random samples — but a quarter of samples are not",
                  fontsize=9.5, color=INK)

    # --- 2. nearest other cluster vs the null -------------------------------
    C = z["centroids"].astype(np.float32)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(C), 200_000), rng.integers(0, len(C), 200_000)
    keep = i != j
    null = np.einsum("ij,ij->i", C[i[keep]], C[j[keep]])
    nb = near["nearest_cos"]
    for v, c, lab in [(null, MUTED, f"any two clusters (median {np.median(null):.2f})"),
                      (nb, KW, f"each cluster's nearest other (median {np.median(nb):.2f})")]:
        x = np.sort(v)
        ax2.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, linewidth=2, label=lab)
    ax2.set(xlim=(0, 1.005), ylim=(0, 1), xlabel="cosine between cluster centroids",
            ylabel="cumulative fraction")
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax2.annotate(f"{(nb >= 0.90).mean():.0%} of clusters have another cluster\n"
                 f"at cosine ≥ 0.90 — but only {near['same_modal'].mean():.0%} of those\n"
                 f"nearest pairs share a modal string",
                 (0.03, 0.55), xycoords="axes fraction", fontsize=8.5, color=INK)
    ax2.set_title("Half of all clusters have a textual near-twin,\n"
                  "and mostly it is not the same study repeated", fontsize=9.5, color=INK)

    # --- 3. one string, how many clusters? ---------------------------------
    rows, colors, labels = [], [], []
    for field, col in [("sub_biomes", SUB), ("keywords", KW)]:
        for r in adj["spread"][field]["top"][:5]:
            rows.append(r["clusters"]); colors.append(col)
            labels.append(r["text"][:38] + ("…" if len(r["text"]) > 38 else ""))
    y = np.arange(len(rows))[::-1]
    ax3.barh(y, rows, .68, color=colors, zorder=2)
    for yi, v in zip(y, rows):
        ax3.annotate(f" {v:,}", (v, yi), va="center", fontsize=8.5, color=INK)
    ax3.set_yticks(y, labels, fontsize=7.5)
    ax3.set_xscale("log")
    ax3.set(xlim=(0.8, max(rows) * 4), xlabel="distinct community clusters covered by that one string")
    sb, kw = adj["spread"]["sub_biomes"], adj["spread"]["keywords"]
    ax3.annotate(f"strings covering >1 cluster:\n"
                 f"sub-biomes {sb['share_spanning_multiple']:.0%}   "
                 f"keywords {kw['share_spanning_multiple']:.0%}",
                 (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8.5, color=INK)
    ax3.set_title("Why keywords win: a sub-biome string is spread over\n"
                  "hundreds of clusters, a keyword string usually over one",
                  fontsize=9.5, color=INK)
    ax3.grid(axis="x", color=FAINT, linewidth=.7, zorder=0)
    ax3.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (SUB, KW)]
    ax3.legend(handles, ["sub-biomes", "keywords"], fontsize=8.5, frameon=False, loc="lower right",
               bbox_to_anchor=(1.0, 0.13))
    for ax in (ax1, ax2, ax3):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5, color=FAINT)
    path = f"{V}/cluster_structure.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
