#!/usr/bin/env python3
"""
Does the way a keyword list is flattened into one string change its embedding?

The pipeline turns  {a b, c, d e}  into  "a b c d e" - braces off, commas to
spaces. Two things are thrown away: the list separators, and any guarantee that
the same keywords in a different order land in the same place. This measures
both, on the same samples, with the same model.

Three ways of writing the same keyword list:

    join     Sambar deer feces gut metagenome      what the pipeline does today
    commas   Sambar deer, feces, gut metagenome    separators kept
    sorted   Sambar deer feces gut metagenome      alphabetical, separators dropped

    commas vs join  isolates the SEPARATORS;  sorted vs join isolates the ORDER.

Three questions:

    1. agreement   one sample, three spellings - how far apart are its vectors?
    2. permutation shuffle one sample's keywords N ways - how far apart do they
                   land, *relative to* the distance between different samples?
                   That ratio is the answer to "does order matter", not the raw gap.
    3. quality     does any variant separate GPT_biomes labels better
                   (separation AUC, 5-NN cross-validated accuracy)?

    python3 scripts/keyword_style_experiment.py --n_per_biome 1000 --dry_run
    python3 scripts/keyword_style_experiment.py --n_per_biome 1000 --yes
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import h5py
import numpy as np
from openai import OpenAI

from embed_subbiomes_keywords import (DEFAULT_INPUT_DIR, NATIVE_DIM, PRICE_PER_1M_TOKENS,
                                      embed_unique, estimate_tokens, iter_samples)
from evaluate_embeddings import auc_same_over_diff, knn_cross_val, unit_rows

VARIANTS = {
    "join":   lambda kws: " ".join(kws),
    "commas": lambda kws: ", ".join(kws),
    "sorted": lambda kws: " ".join(sorted(kws, key=str.lower)),
}
# Categorical slots 1-3 of the validated palette (all-pairs CVD-safe).
COLORS = {"join": "#2a78d6", "commas": "#eb6834", "sorted": "#1baf7a"}


def keyword_lists(path, keep_ids):
    """-> {sample_id: [keyword, ...]}, splitting on the commas before they are lost."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            sid, _, raw = line.rstrip("\n").partition("\t")
            if sid not in keep_ids:
                continue
            kws = [k.strip() for k in raw.strip().strip("{}").split(",") if k.strip()]
            if kws:
                out[sid] = kws
    return out


def pick_subset(biomes_path, n_per_biome, seed):
    groups = defaultdict(list)
    for sid, biome in iter_samples(biomes_path, None, False):
        groups[biome].append(sid)
    rng = random.Random(seed)
    chosen = {}
    for biome in sorted(groups):
        pool = groups[biome]
        for sid in (pool if len(pool) <= n_per_biome else rng.sample(pool, n_per_biome)):
            chosen[sid] = biome
        print(f"  biome '{biome}': {min(len(pool), n_per_biome)} / {len(pool)}")
    return chosen


def vectors_for(texts, h5_path):
    """Map each text to its vector from a compact unique-text file."""
    with h5py.File(h5_path, "r") as f:
        lookup = dict(zip((t.decode("utf-8") for t in f["texts"][:]), f["embeddings"][:]))
    return np.stack([lookup[t] for t in texts])


def cosines(a, b):
    return np.einsum("ij,ij->i", unit_rows(a), unit_rows(b))


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output_dir", default=None, help="Default: <input_dir>/embeddings/keyword_style")
    p.add_argument("--api_key_path", default=os.path.expanduser("~/MicrobeAtlasProject/my_api_key_embeddings"))
    p.add_argument("--model", default="text-embedding-3-large", choices=sorted(NATIVE_DIM))
    p.add_argument("--embedding_dim", type=int, default=1024)
    p.add_argument("--n_per_biome", type=int, default=1000)
    p.add_argument("--n_perm_samples", type=int, default=200, help="Samples to shuffle for the order test.")
    p.add_argument("--n_perms", type=int, default=6, help="Shufflings per sample.")
    p.add_argument("--n_pairs", type=int, default=50000, help="Random pairs for the between-sample baseline.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    out_dir = os.path.expanduser(args.output_dir) if args.output_dir else \
        os.path.join(input_dir, "embeddings", "keyword_style")
    os.makedirs(out_dir, exist_ok=True)
    dim, tag = args.embedding_dim, f"{args.model}__dim{args.embedding_dim}__perbiome{args.n_per_biome}_seed{args.seed}"

    print(f"Selecting {args.n_per_biome} samples per biome ...")
    biome_of = pick_subset(os.path.join(input_dir, "GPT_biomes.txt"), args.n_per_biome, args.seed)
    kws_of = keyword_lists(os.path.join(input_dir, "GPT_keywords.txt"), set(biome_of))
    samples = sorted(kws_of)
    print(f"{len(samples)} of {len(biome_of)} selected samples have keywords "
          f"(mean {np.mean([len(kws_of[s]) for s in samples]):.1f} keywords each)")

    # text per sample per variant, plus one shuffled-order group per permutation sample
    texts_of = {name: [fn(kws_of[s]) for s in samples] for name, fn in VARIANTS.items()}

    rng = random.Random(args.seed)
    perm_samples = [s for s in samples if len(kws_of[s]) >= 4]
    perm_samples = rng.sample(perm_samples, min(args.n_perm_samples, len(perm_samples)))
    perm_texts = {}
    for s in perm_samples:
        seen = []
        for _ in range(args.n_perms):
            shuffled = kws_of[s][:]
            rng.shuffle(shuffled)
            text = " ".join(shuffled)
            if text not in seen:
                seen.append(text)
        perm_texts[s] = seen

    jobs = {name: list(dict.fromkeys(texts)) for name, texts in texts_of.items()}
    jobs["perm"] = list(dict.fromkeys(t for group in perm_texts.values() for t in group))

    total = 0
    for name, unique in jobs.items():
        tokens, how = estimate_tokens(unique, args.model)
        total += tokens
        print(f"  [{name}] {len(unique)} distinct texts, ~{tokens:,} tokens ({how})")
    cost = total / 1e6 * PRICE_PER_1M_TOKENS[args.model]
    print(f"\nEstimated {total:,} tokens = ${cost:.4f} with {args.model}")
    if args.dry_run:
        print("--dry_run: stopping before any API call.")
        return
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return

    client = OpenAI(api_key=open(os.path.expanduser(args.api_key_path)).read().strip(), max_retries=8)
    paths = {}
    for name, unique in jobs.items():
        paths[name] = os.path.join(out_dir, f"kwstyle_{name}__{tag}.h5")
        embed_unique(name, unique, paths[name], client, args.model, dim, 2048)

    analyse(samples, biome_of, texts_of, perm_texts, paths, args, out_dir, tag)


# --------------------------------------------------------------------------

def analyse(samples, biome_of, texts_of, perm_texts, paths, args, out_dir, tag):
    vectors = {name: vectors_for(texts_of[name], paths[name]) for name in VARIANTS}
    summary = {"n_samples": len(samples), "model": args.model, "dim": args.embedding_dim}

    # 1. agreement: the same sample written three ways
    print(f"\n{'=' * 64}\n1. same sample, different spelling\n{'=' * 64}")
    agreement = {}
    for a, b in [("join", "commas"), ("join", "sorted"), ("commas", "sorted")]:
        sims = cosines(vectors[a], vectors[b])
        agreement[f"{a} vs {b}"] = {"mean": float(sims.mean()), "min": float(sims.min()),
                                    "p5": float(np.percentile(sims, 5))}
        print(f"  {a:>7} vs {b:<7} cosine  mean {sims.mean():.4f}  p5 {np.percentile(sims, 5):.4f}  min {sims.min():.4f}")
    summary["agreement"] = agreement

    # 2. permutation spread, against the between-sample yardstick
    print(f"\n{'=' * 64}\n2. order: shuffling one sample's keywords\n{'=' * 64}")
    perm_vecs = vectors_for(sorted({t for g in perm_texts.values() for t in g}), paths["perm"])
    perm_index = {t: i for i, t in enumerate(sorted({t for g in perm_texts.values() for t in g}))}
    perm_sims = []
    for group in perm_texts.values():
        if len(group) < 2:
            continue
        block = unit_rows(perm_vecs[[perm_index[t] for t in group]])
        pair = block @ block.T
        perm_sims.extend(pair[np.triu_indices(len(group), k=1)])
    perm_sims = np.asarray(perm_sims)

    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, len(samples), args.n_pairs)
    j = rng.integers(0, len(samples), args.n_pairs)
    i, j = i[i != j], j[i != j]
    between = cosines(vectors["join"][i], vectors["join"][j])

    gap_order = 1 - perm_sims.mean()
    gap_between = 1 - between.mean()
    print(f"  same keywords, reshuffled : cosine mean {perm_sims.mean():.4f}  min {perm_sims.min():.4f}")
    print(f"  different samples         : cosine mean {between.mean():.4f}")
    print(f"  -> reordering moves a sample {gap_order / gap_between:.1%} as far as "
          f"swapping it for a different sample")
    summary["permutation"] = {"within_mean": float(perm_sims.mean()), "within_min": float(perm_sims.min()),
                              "between_mean": float(between.mean()),
                              "relative_displacement": float(gap_order / gap_between)}

    # 3. quality: does any spelling separate the biomes better?
    print(f"\n{'=' * 64}\n3. quality vs GPT_biomes\n{'=' * 64}")
    labels = [biome_of[s] for s in samples]
    quality = {}
    for name in VARIANTS:
        unit = unit_rows(vectors[name])
        sims = np.einsum("ij,ij->i", unit[i], unit[j])
        same = np.array(labels)[i] == np.array(labels)[j]
        auc = auc_same_over_diff(sims[same], sims[~same])
        knn = knn_cross_val(unit, labels, 5, 5, args.seed)
        quality[name] = {"separation_auc": auc, "knn_accuracy": knn.get("accuracy"),
                         "baseline": knn.get("baseline_majority_accuracy")}
        print(f"  {name:<7} AUC {auc:.4f}   5-NN {knn.get('accuracy'):.4f} "
              f"(baseline {knn.get('baseline_majority_accuracy'):.4f})")
    summary["quality"] = quality

    plot(perm_sims, cosines(vectors["join"], vectors["commas"]), between, quality, out_dir, tag)
    path = os.path.join(out_dir, f"keyword_style_summary__{tag}.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\nWritten to {path}")


def plot(perm_sims, sep_sims, between, quality, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing, skipping the figure)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    series = [(perm_sims, COLORS["sorted"], "same keywords, reshuffled"),
              (sep_sims, COLORS["commas"], "same keywords, commas added"),
              (between, COLORS["join"], "different samples")]
    # Range from the data: the within-sample curves crowd against 1.0, so a
    # fixed 0-1 axis would hide the only part worth looking at.
    lo = min(np.percentile(v, 0.5) for v, _, _ in series)
    bins = np.linspace(min(lo, 0.0), 1.0, 160)
    for values, color, label in series:
        density, edges = np.histogram(values, bins=bins, density=True)
        ax1.plot((edges[:-1] + edges[1:]) / 2, density, color=color, linewidth=2,
                 label=f"{label}  (mean {values.mean():.3f})")
    ax1.set_xlabel("cosine similarity")
    ax1.set_ylabel("density")
    ax1.set_title("How far does rewriting the same list move it?", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    names = list(quality)
    x = np.arange(len(names))
    for offset, key, marker, label in [(-0.09, "separation_auc", "o", "separation AUC"),
                                       (0.09, "knn_accuracy", "s", "5-NN accuracy")]:
        values = [quality[n][key] for n in names]
        for xi, name, value in zip(x + offset, names, values):
            ax2.plot(xi, value, marker, color=COLORS[name], markersize=9,
                     markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            ax2.annotate(f"{value:.3f}", xy=(xi, value), xytext=(0, 9), textcoords="offset points",
                         ha="center", fontsize=8, color="#52514e")
        ax2.plot([], [], marker, color="#52514e", markersize=7, linestyle="none", label=label)
    ax2.axhline(quality[names[0]]["baseline"], linestyle="--", linewidth=1,
                color="#8a8a86", label="majority baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Does the spelling change quality? (vs GPT_biomes)", fontsize=10)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.grid(alpha=0.25, linewidth=0.6, axis="y")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, f"keyword_style__{tag}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figure: {path}")


if __name__ == "__main__":
    main()
