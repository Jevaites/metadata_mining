#!/usr/bin/env python3
"""
Compare embedding quality across model / dimension configs.

Reads the compact "*_unique_embeddings*.h5" tables (one row per distinct text).
Using the per-sample files instead would flood every pairwise-similarity
distribution with a spurious cos=1 spike from duplicated texts.

All configs are cut down to the SAME texts in the SAME order, and the same
random pairs are scored in every config, so the comparison is paired.

Per config, and per label source, it reports:

  * the pairwise cosine distribution over random pairs (the plain one, and
    split into same-label / different-label pairs);
  * separation_auc = P(a same-label pair scores above a different-label pair).
    Prefer this over separation_gap when comparing across dimensions: raw
    cosines shrink as dimension grows, so the gap of the means drops even when
    the classes are separated exactly as well. AUC and Cohen's d are unit-free;
    the gap is not.
  * k-NN cross-validated accuracy against a majority-class baseline - the most
    decision-relevant number, since it approximates "would a nearest-neighbour
    lookup recover the right label";
  * a nearest-neighbour spot check on the same query texts in every config.

Label sources (at least one required):
  --label_file   sample_id<TAB>label, e.g. GPT_biomes.txt (3.4M samples, coarse,
                 LLM-derived: measures agreement with GPT, not ground truth)
  --gold_pkl     curated sample_id -> (reads, biome, subbiome, lat_lon, place),
                 ~1k samples. Use --gold_field biome on small subsets.
                 Note keyword texts are near-unique per sample, so almost none
                 of them pick up a gold label - gold is only practical for
                 sub_biomes.

A text can carry different labels on different samples; it takes the majority
label, dropped when that majority is below --label_purity_threshold.

    python3 scripts/evaluate_embeddings.py \
        --config small_1536=.../GPT_sub_biomes_unique_embeddings__text-embedding-3-small__dim1536__perbiome2000_seed42.h5 \
        --config large_3072=.../GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim3072__perbiome2000_seed42.h5 \
        --source_text_file ~/MicrobeAtlasProject/sidequest/latest/GPT_sub_biomes.txt \
        --label_file       ~/MicrobeAtlasProject/sidequest/latest/GPT_biomes.txt \
        --gold_pkl ~/MicrobeAtlasProject/gold_dict.pkl --gold_field biome \
        --output_dir ~/MicrobeAtlasProject/sidequest/latest/embeddings/eval_sub_biomes
"""

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from embed_subbiomes_keywords import clean_text

GOLD_FIELD_INDEX = {"biome": 1, "subbiome": 2}


def abspath(path):
    """~ is not expanded by the shell inside `--config label=~/x`, so do it here."""
    return os.path.abspath(os.path.expanduser(path))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_unique_embeddings(path):
    with h5py.File(path, "r") as f:
        texts = [t.decode("utf-8") if isinstance(t, bytes) else t for t in f["texts"][:]]
        return texts, f["embeddings"][:].astype(np.float32)


def unit_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1.0, norms)


def load_sample_labels(label_file, gold_pkl, gold_field):
    """-> {source_name: {sample_id: label}}"""
    sources = {}
    if label_file:
        labels = {}
        for line in open(label_file, encoding="utf-8", errors="replace"):
            sid, _, label = line.rstrip("\n").partition("\t")
            if sid and label.strip():
                labels[sid] = label.strip()
        sources["gpt_biome"] = labels
    if gold_pkl:
        index = GOLD_FIELD_INDEX[gold_field]
        gold = pickle.load(open(gold_pkl, "rb"))
        sources[f"gold_{gold_field}"] = {
            sid: str(row[index]).strip()
            for sid, row in gold.items() if len(row) > index and row[index]
        }
    return sources


def label_texts(source_text_file, texts, sample_to_label, is_keywords, purity, min_count):
    """Majority label per distinct text, from one pass over the big source file.

    Note this counts *every* sample carrying the text, including samples outside
    the embedded subset - that is what lets a ~1k-sample gold set label texts in
    a subset it barely overlaps."""
    wanted = set(texts)
    counts = defaultdict(Counter)
    for line in open(source_text_file, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        label = sample_to_label.get(sid)
        if label is None or not raw:
            continue
        text = clean_text(raw, is_keywords)
        if text in wanted:
            counts[text][label] += 1

    resolved = {}
    for text, counter in counts.items():
        label, n = counter.most_common(1)[0]
        if n / sum(counter.values()) >= purity:
            resolved[text] = label
    keep = {c for c, n in Counter(resolved.values()).items() if n >= min_count}
    dropped = {c: n for c, n in Counter(resolved.values()).items() if n < min_count}
    if dropped:
        print(f"    dropping classes with < {min_count} texts: {dropped}")
    return {t: l for t, l in resolved.items() if l in keep}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def summarize(values):
    if len(values) == 0:
        return {"n": 0}
    return {"n": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            "p5": float(np.percentile(values, 5)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def auc_same_over_diff(same, diff):
    """P(random same-label pair scores above a random different-label pair).

    0.5 = the embedding tells the two apart no better than chance, 1.0 = perfect.
    Unlike a difference of mean cosines, this does not change when the whole
    similarity scale shifts, so it is safe to compare across dimensions."""
    if len(same) == 0 or len(diff) == 0:
        return None
    order = np.argsort(np.concatenate([same, diff]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    n1, n2 = len(same), len(diff)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def pairwise_cosine(unit_embeddings, pairs, labels):
    """Cosine of the given index pairs, overall and split by label agreement."""
    i, j = pairs
    sims = np.einsum("ij,ij->i", unit_embeddings[i], unit_embeddings[j])
    labels = np.array([l or "" for l in labels], dtype=object)
    both = (labels[i] != "") & (labels[j] != "")
    same_sims = sims[both & (labels[i] == labels[j])]
    diff_sims = sims[both & (labels[i] != labels[j])]

    result = {"overall": summarize(sims), "same_label": summarize(same_sims),
              "diff_label": summarize(diff_sims), "separation_auc": auc_same_over_diff(same_sims, diff_sims)}
    if len(same_sims) and len(diff_sims):
        pooled = np.sqrt((same_sims.std() ** 2 + diff_sims.std() ** 2) / 2)
        result["separation_gap"] = float(same_sims.mean() - diff_sims.mean())
        result["separation_cohens_d"] = float(result["separation_gap"] / pooled)
    return result, sims, same_sims, diff_sims


def knn_cross_val(unit_embeddings, labels, k, folds, seed):
    mask = np.array([l is not None for l in labels])
    X, y = unit_embeddings[mask], np.array([l for l in labels if l is not None])
    if len(y) < 2 * folds or len(set(y)) < 2:
        return {"n": int(len(y)), "note": "not enough labelled data"}
    folds = min(folds, min(Counter(y).values()))
    if folds < 2:
        return {"n": int(len(y)), "note": "smallest class has < 2 members"}

    correct = baseline = total = 0
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        model = KNeighborsClassifier(n_neighbors=min(k, len(train)), metric="cosine").fit(X[train], y[train])
        correct += int((model.predict(X[test]) == y[test]).sum())
        baseline += int((y[test] == Counter(y[train]).most_common(1)[0][0]).sum())
        total += len(test)
    return {"n": int(len(y)), "n_classes": len(set(y)), "folds": folds,
            "accuracy": correct / total, "baseline_majority_accuracy": baseline / total}


def nearest_neighbours(texts, unit_embeddings, labels, queries, top_k):
    out = []
    index = {t: i for i, t in enumerate(texts)}
    for text in queries:
        q = index[text]
        sims = unit_embeddings @ unit_embeddings[q]
        best = [i for i in np.argsort(-sims)[:top_k + 1] if i != q][:top_k]
        out.append({"query_text": text, "query_label": labels[q],
                    "neighbours": [{"text": texts[i], "label": labels[i], "cosine": float(sims[i])} for i in best]})
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def get_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib missing, skipping plots)")
        return None


def plot_overlay(sims_by_config, output_dir):
    """The plain pairwise-cosine distribution of every config on one axis."""
    plt = get_pyplot()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-0.4, 1.0, 120)
    for label, sims in sims_by_config.items():
        density, edges = np.histogram(sims, bins=bins, density=True)
        ax.plot((edges[:-1] + edges[1:]) / 2, density, label=f"{label} (mean {sims.mean():.3f})")
    ax.set(xlabel="cosine similarity", ylabel="density", title="Pairwise cosine similarity, all configs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pairwise_cosine_overlay.png"), dpi=130)
    plt.close(fig)


def plot_separation(source, split_by_config, output_dir):
    """Same-label vs different-label pairs, one small panel per config."""
    plt = get_pyplot()
    if plt is None:
        return
    n = len(split_by_config)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.2), sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(-0.4, 1.0, 60)
    for ax, (label, (same, diff)) in zip(axes[0], split_by_config.items()):
        ax.hist(diff, bins=bins, density=True, alpha=0.55, color="tab:red", label="different")
        ax.hist(same, bins=bins, density=True, alpha=0.55, color="tab:blue", label="same")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("cosine")
    axes[0][0].set_ylabel("density")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Same-label vs different-label pairs ({source})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"separation__{source}.png"), dpi=130)
    plt.close(fig)


def plot_summary(source, rows, output_dir):
    """AUC and k-NN accuracy per config: the headline comparison."""
    plt = get_pyplot()
    if plt is None:
        return
    names = list(rows)
    fig, axes = plt.subplots(1, 2, figsize=(4 + 0.8 * len(names), 3.6))
    for ax, key, title in [(axes[0], "auc", "separation AUC (same > diff)"),
                           (axes[1], "knn", "k-NN CV accuracy")]:
        values = [rows[n][key] for n in names]
        ax.plot(range(len(names)), values, "o-")
        if key == "knn":
            ax.axhline(rows[names[0]]["baseline"], ls="--", c="grey", label="majority baseline")
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"label source: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"summary__{source}.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, dest="configs",
                   help="label=path_to_unique_embeddings.h5 ; repeat per config.")
    p.add_argument("--source_text_file", required=True, help="GPT_sub_biomes.txt or GPT_keywords.txt.")
    p.add_argument("--is_keywords", action="store_true", help="Set when --source_text_file is the keywords file.")
    p.add_argument("--label_file")
    p.add_argument("--gold_pkl")
    p.add_argument("--gold_field", default="biome", choices=sorted(GOLD_FIELD_INDEX))
    p.add_argument("--label_purity_threshold", type=float, default=0.8)
    p.add_argument("--min_label_count", type=int, default=20)
    p.add_argument("--max_points", type=int, default=20000, help="Cap on distinct texts scored per config.")
    p.add_argument("--n_pairs", type=int, default=50000)
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--knn_folds", type=int, default=5)
    p.add_argument("--n_spotcheck", type=int, default=5)
    p.add_argument("--spotcheck_top_k", type=int, default=5)
    p.add_argument("--output_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    configs = [(c.split("=", 1)[0], abspath(c.split("=", 1)[1])) for c in args.configs]
    output_dir = abspath(args.output_dir) if args.output_dir else os.path.dirname(configs[0][1])
    os.makedirs(output_dir, exist_ok=True)

    loaded = {}
    for label, path in configs:
        texts, embeddings = load_unique_embeddings(path)
        loaded[label] = (texts, embeddings)
        print(f"  [{label}] {len(texts)} distinct texts, dim={embeddings.shape[1]}")

    # Score every config on the same texts, in the same order, with the same
    # random pairs - otherwise differences between configs are partly sampling.
    shared = sorted(set.intersection(*(set(t) for t, _ in loaded.values())))
    if len(shared) > args.max_points:
        shared = sorted(random.Random(args.seed).sample(shared, args.max_points))
    print(f"\nScoring {len(shared)} texts common to all {len(loaded)} configs")

    points = {}
    for label, (texts, embeddings) in loaded.items():
        index = {t: i for i, t in enumerate(texts)}
        points[label] = unit_rows(embeddings[[index[t] for t in shared]])

    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, len(shared), args.n_pairs)
    j = rng.integers(0, len(shared), args.n_pairs)
    pairs = (i[i != j], j[i != j])

    sources = load_sample_labels(
        abspath(args.label_file) if args.label_file else None,
        abspath(args.gold_pkl) if args.gold_pkl else None, args.gold_field)
    if not sources:
        raise SystemExit("Pass --label_file and/or --gold_pkl.")

    text_labels = {}
    for name, sample_to_label in sources.items():
        print(f"\nResolving '{name}' labels per text ({len(sample_to_label)} labelled samples)...")
        resolved = label_texts(abspath(args.source_text_file), shared, sample_to_label,
                               args.is_keywords, args.label_purity_threshold, args.min_label_count)
        print(f"  {len(resolved)} / {len(shared)} texts labelled")
        if len(resolved) < args.min_label_count:
            print(f"  -> too few, skipping '{name}' entirely")
            continue
        text_labels[name] = resolved
    if not text_labels:
        raise SystemExit("No label source produced usable labels for these texts.")

    summary = {"n_texts_scored": len(shared), "configs": {}}
    all_sims, split, per_source_rows = {}, defaultdict(dict), defaultdict(dict)

    for config, unit_embeddings in points.items():
        print(f"\n{'=' * 64}\n{config}  (dim={unit_embeddings.shape[1]})\n{'=' * 64}")
        summary["configs"][config] = {"dim": int(unit_embeddings.shape[1]), "by_label_source": {}}

        for source, label_map in text_labels.items():
            labels = [label_map.get(t) for t in shared]
            cosine, sims, same, diff = pairwise_cosine(unit_embeddings, pairs, labels)
            all_sims[config] = sims  # identical pairs across sources, so recording once is enough
            split[source][config] = (same, diff)

            knn = knn_cross_val(unit_embeddings, labels, args.knn_k, args.knn_folds, args.seed)
            print(f"-- {source}: {sum(l is not None for l in labels)} labelled texts")
            print(f"   cosine overall mean {cosine['overall']['mean']:.4f} | "
                  f"same {cosine['same_label'].get('mean', float('nan')):.4f} vs "
                  f"diff {cosine['diff_label'].get('mean', float('nan')):.4f} | "
                  f"AUC {cosine['separation_auc']:.4f} | d {cosine.get('separation_cohens_d', float('nan')):.3f}")
            print(f"   {args.knn_k}-NN accuracy {knn.get('accuracy')} vs baseline "
                  f"{knn.get('baseline_majority_accuracy')} over {knn.get('n_classes')} classes")

            summary["configs"][config]["by_label_source"][source] = {"pairwise_cosine": cosine, "knn": knn}
            per_source_rows[source][config] = {"auc": cosine["separation_auc"],
                                               "knn": knn.get("accuracy"),
                                               "baseline": knn.get("baseline_majority_accuracy")}

        queries = random.Random(args.seed).sample(shared, min(args.n_spotcheck, len(shared)))
        main_source = next(iter(text_labels))
        spot = nearest_neighbours(shared, unit_embeddings,
                                  [text_labels[main_source].get(t) for t in shared],
                                  queries, args.spotcheck_top_k)
        summary["configs"][config]["spotcheck"] = spot
        print(f"\n   nearest neighbours ({main_source}):")
        for item in spot:
            print(f"   query {item['query_text']!r} [{item['query_label']}]")
            for nb in item["neighbours"]:
                print(f"       {nb['cosine']:.4f}  [{nb['label']}]  {nb['text']!r}")

    if not args.no_plots:
        plot_overlay(all_sims, output_dir)
        for source in text_labels:
            plot_separation(source, split[source], output_dir)
            plot_summary(source, per_source_rows[source], output_dir)

    print(f"\n{'=' * 64}\nSummary ({len(shared)} shared texts)\n{'=' * 64}")
    print(f"{'config':<14}{'dim':>6}" + "".join(f"{s + ' auc':>16}{s + ' knn':>16}" for s in text_labels))
    for config, cfg in summary["configs"].items():
        row = f"{config:<14}{cfg['dim']:>6}"
        for source in text_labels:
            got = cfg["by_label_source"][source]
            row += f"{got['pairwise_cosine']['separation_auc']:>16.4f}{got['knn'].get('accuracy', 0):>16.4f}"
        print(row)

    path = os.path.join(output_dir, "evaluation_summary.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2, default=str)
    print(f"\nWritten to {path} (plus PNGs in the same directory)")


if __name__ == "__main__":
    main()
