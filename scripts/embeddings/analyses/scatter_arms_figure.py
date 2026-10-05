#!/usr/bin/env python3
"""
Figure for the two arms of the sub-biomes / changed-text scatter.

    left   the arms, highlighted on the original 200k pairs
    middle where along each arm the points sit (the other run's cosine)
    right  what the biggest new strings absorb, over the whole overlap

    MAP_ROOT=... python3 scripts/embeddings/analyses/scatter_arms_figure.py
"""
import os, sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
OLD, NEW = f"{ROOT}/sidequest", f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_previous"
TOP, RIGHT, MUTED = "#eb6834", "#1baf7a", "#8a8f98"
INK, FAINT = "#23262b", "#d6d9dd"


def stream(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid and raw.strip():
            yield sid, clean_text(raw, False)


def ecdf(ax, v, color, label):
    x = np.sort(v)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, linewidth=2, label=label)


def main():
    d = np.load(f"{OUT}/scatter_arms_points.npz")
    a, b, top, right = d["cos_dany"], d["cos_new"], d["top"], d["right"]
    off = ~top & ~right
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)

    # --- 1. the scatter, arms highlighted -----------------------------------
    lim = [min(a.min(), b.min()) - 0.03, 1.03]
    ax1.hexbin(a[off], b[off], gridsize=70, bins="log", cmap="Greys", mincnt=1,
               linewidths=0, extent=(*lim, *lim))
    ax1.scatter(a[top], b[top], s=5, color=TOP, alpha=.55, linewidths=0, zorder=3)
    ax1.scatter(a[right], b[right], s=5, color=RIGHT, alpha=.55, linewidths=0, zorder=3)
    ax1.plot(lim, lim, color=FAINT, linewidth=1.4, linestyle="--", zorder=1)
    box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=.85)
    ax1.annotate(f"top arm — same new string\n{top.sum():,} pairs ({top.mean():.1%})",
                 xy=(0.03, 0.93), xycoords="axes fraction", color=TOP, fontsize=9,
                 fontweight="bold", va="top", bbox=box)
    ax1.annotate(f"right arm — same Dany string\n{right.sum():,} pairs ({right.mean():.1%})",
                 xy=(0.95, 0.10), xycoords="axes fraction", color=RIGHT, fontsize=9,
                 fontweight="bold", ha="right", va="bottom", bbox=box)
    ax1.set(xlim=lim, ylim=lim, xlabel="cosine — Dany (3-small, 1536d)",
            ylabel="cosine — new (3-large, 1024d)")
    ax1.set_title("Both arms are literal string collisions\n"
                  "(100% of arm pairs share a text, and vice versa)",
                  fontsize=9.5, color=INK)

    # --- 2. where along the arm -------------------------------------------
    ecdf(ax2, a[top], TOP, f"top arm: cosine in Dany's space  (median {np.median(a[top]):.2f})")
    ecdf(ax2, b[right], RIGHT, f"right arm: cosine in new space  (median {np.median(b[right]):.2f})")
    ecdf(ax2, np.concatenate([a[off], b[off]]), MUTED, "off-arm pairs, both spaces")
    ax2.axvline(0.5, color=FAINT, linewidth=1.2, linestyle=":")
    ax2.annotate(f"{np.mean(a[top] < 0.5):.0%} of the top arm and "
                 f"{np.mean(b[right] < 0.5):.0%} of the right arm\n"
                 "sit below 0.5 — the other run calls them unrelated",
                 xy=(0.5, 0.06), xytext=(0.54, 0.06), fontsize=8.5, color=INK, va="center")
    ax2.set(xlim=(-0.05, 1.02), ylim=(0, 1), xlabel="cosine in the other run's space",
            ylabel="cumulative fraction of arm pairs")
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.set_title("Most collisions join things the other run also\n"
                  "thought were close — but a real tail does not", fontsize=9.5, color=INK)

    # --- 3. what the big new strings absorb, full overlap ------------------
    old = dict(stream(f"{OLD}/GPT_sub_biomes.txt"))
    new = dict(stream(f"{NEW}/GPT_sub_biomes.txt"))
    both = set(old) & set(new)
    absorbed = defaultdict(set)
    n_samples = Counter()
    for s in both:
        absorbed[new[s]].add(old[s])
        n_samples[new[s]] += 1
    rows = sorted(((t, len(absorbed[t]), n) for t, n in n_samples.most_common(10)),
                  key=lambda r: r[1])
    y = np.arange(len(rows))
    ax3.barh(y, [r[1] for r in rows], color=TOP, height=.62, zorder=2)
    ax3.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    for i, (_, k, n) in enumerate(rows):
        ax3.annotate(f"{k:,}   ({n/1000:.0f}k samples)", xy=(k, i), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK)
    ax3.set(xlim=(0, max(r[1] for r in rows) * 1.55),
            xlabel="distinct Dany strings merged into it")
    ax3.set_title(f"Same 1.64M samples: Dany {len(set(old[s] for s in both)):,} distinct\n"
                  f"strings, the new run {len(set(new[s] for s in both)):,} — 4.5x smaller",
                  fontsize=9.5, color=INK)
    ax3.grid(axis="x", color=FAINT, linewidth=.7, zorder=0)
    ax3.set_axisbelow(True)

    for ax in (ax1, ax2, ax3):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8.5, color=FAINT)
    path = f"{OUT}/scatter_arms.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
