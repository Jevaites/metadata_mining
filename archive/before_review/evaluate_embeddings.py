#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare embedding-quality across different model / --embedding_dim configs
produced by embed_subbiomes_keywords.py.

Works on the compact "*_unique_embeddings*.h5" tables (one row per distinct
text) rather than the per-sample files: this is both cheaper (orders of
magnitude fewer rows) and methodologically correct, since scoring the
per-sample file would let identical-text duplicates flood any pairwise
similarity distribution with a spurious cos=1 spike.

Three complementary signals are computed per config, per label source:

  1. Pairwise cosine similarity distribution, split into "same coarse label"
     vs "different coarse label" pairs (random-sampled, not exhaustive - see
     --n_pairs). A good embedding space should show same-label pairs
     concentrated at higher cosine similarity than different-label pairs;
     the gap between the two distributions ("separation") is a single
     number to compare configs by. This directly extends the plain
     "pairwise cosine similarity distribution" check by removing the
     duplicate-text artifact and adding the label split, which the
     un-split version can't distinguish (a small-diameter embedding space
     with everything mutually similar looks identical to a well-clustered
     one under the un-split view).
  2. k-NN cross-validated classification accuracy against the same coarse
     labels, compared to a majority-class baseline. This is a much more
     direct, decision-relevant signal than either cosine number: it asks
     "if I stood at a point in this embedding and voted using its
     neighbors, would I recover the right label", which is close to what
     downstream ontology-mapping will actually do.
  3. Qualitative nearest-neighbor spot check: a handful of texts (same
     random texts across configs, for a fair look) with their top-k nearest
     neighbors printed side by side, to catch failure modes a summary
     statistic can hide (e.g. a config that's numerically fine but returns
     nonsense neighbors for rare/ambiguous inputs).

Two label sources are supported, since neither is perfect alone:
  - --label_file: sample_id<TAB>label, one label per sample (e.g. the
    GPT-derived GPT_biomes.txt - full scale, ~3.4M samples, but coarse and
    itself LLM-generated, so it can only measure "as good as GPT_biomes.txt
    agrees with itself").
  - --gold_pkl: a pickled dict sample_id -> (read_count, biome,
    subbiome_or_material, lat_lon, location), e.g. gold_dict.pkl - far
    smaller (~1k samples) but curated, so a config that wins here is more
    convincing even though the sample size gives it more noise.

A distinct text can be shared by samples with different labels (e.g. the
same free-text sub-biome string attached to samples someone else coded as
two different coarse biomes). Each text is assigned the majority label
among its samples, and --label_purity_threshold drops texts where that
majority isn't dominant enough to trust as a label for evaluation purposes
(this is a data-quality filter on the *label*, unrelated to embedding
quality).

Example
-------
python scripts/evaluate_embeddings.py \\
    --config small_1536=sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-small__perbiome200_seed42.h5 \\
    --config large_1024=sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__perbiome200_seed42.h5 \\
    --source_text_file ~/MicrobeAtlasProject/sidequest/latest/GPT_sub_biomes.txt \\
    --label_file ~/MicrobeAtlasProject/sidequest/latest/GPT_biomes.txt \\
    --gold_pkl ~/MicrobeAtlasProject/gold_dict.pkl --gold_field subbiome \\
    --output_dir ~/MicrobeAtlasProject/sidequest/latest/embeddings/eval
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for evaluate_embeddings.py. "
        "Install with: pip install scikit-learn"
    ) from exc

try:
    # Reuse the exact same keyword-text cleaning the embedding script used,
    # so texts line up between the raw source file and the *_unique_embeddings
    # H5 (which was built from cleaned text for the keywords target).
    from embed_subbiomes_keywords import clean_keyword_text
except ImportError:
    def clean_keyword_text(text: str, sep: str = " ", strip_commas: bool = True) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        if strip_commas:
            text = text.replace(",", sep)
        return " ".join(text.split())


GOLD_FIELD_INDEX = {"biome": 1, "subbiome": 2}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", action="append", required=True, dest="configs",
        help="label=path_to_unique_embeddings.h5 ; repeat for each config to compare.",
    )
    parser.add_argument(
        "--source_text_file", required=True,
        help="Raw sample_id<TAB>text file the embeddings were computed from "
             "(e.g. GPT_sub_biomes.txt or GPT_keywords.txt) - used to recover, "
             "for each distinct text, which samples (and hence which labels) "
             "share it.",
    )
    parser.add_argument("--is_keywords", action="store_true", help="Set if --source_text_file is a keywords file (brace-wrapped, comma-separated).")
    parser.add_argument("--keyword_sep", default=" ")
    parser.add_argument("--keep_keyword_commas", action="store_true")

    parser.add_argument("--label_file", default=None, help="sample_id<TAB>label file, e.g. GPT_biomes.txt.")
    parser.add_argument("--label_file_name", default="gpt_biome", help="Name for this label source in output.")
    parser.add_argument("--gold_pkl", default=None, help="Pickled sample_id -> (read_count, biome, subbiome, lat_lon, location) dict, e.g. gold_dict.pkl.")
    parser.add_argument("--gold_field", default="subbiome", choices=sorted(GOLD_FIELD_INDEX), help="Which gold_dict.pkl field to use as the label.")
    parser.add_argument("--gold_name", default=None, help="Name for the gold label source in output (default: 'gold_' + --gold_field).")

    parser.add_argument("--label_purity_threshold", type=float, default=0.8, help="Drop a text as unlabeled if its majority label covers less than this fraction of its samples.")
    parser.add_argument("--min_label_count", type=int, default=20, help="Drop a label class entirely if fewer than this many texts end up assigned to it (avoids 1-member classes breaking k-fold CV).")

    parser.add_argument("--max_points", type=int, default=20000, help="Cap on distinct texts used per (config, label source) for pairwise-cosine/kNN, for runtime/memory. Randomly subsampled.")
    parser.add_argument("--n_pairs", type=int, default=50000, help="Random pairs sampled for the pairwise cosine similarity distribution.")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--knn_folds", type=int, default=5)
    parser.add_argument("--n_spotcheck", type=int, default=5, help="Number of texts to print nearest-neighbor spot checks for.")
    parser.add_argument("--spotcheck_top_k", type=int, default=5)

    parser.add_argument("--output_dir", default=None, help="Where to write the JSON summary and histogram PNGs. Default: alongside the first --config file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true", help="Skip matplotlib histograms (summary/spot-check still printed and saved to JSON).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def decode(arr: np.ndarray) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else x for x in arr]


def load_unique_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    with h5py.File(path, "r") as f:
        texts = decode(f["texts"][:])
        embs = f["embeddings"][:].astype(np.float32)
    return texts, embs


def normalize_rows(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def resolve_path(path: str) -> str:
    """Expand ~ and resolve to an absolute path. Needed because the shell
    only auto-expands a leading ~ when it's the first character of an
    argument word - inside `--config label=~/x` the ~ comes after `label=`
    and is passed through to Python completely literally in some shells."""
    return os.path.abspath(os.path.expanduser(path))


def parse_config_args(config_strs: Sequence[str]) -> "list[Tuple[str, str]]":
    configs = []
    for c in config_strs:
        if "=" not in c:
            raise SystemExit(f"--config must be label=path, got: {c!r}")
        label, path = c.split("=", 1)
        configs.append((label, resolve_path(path)))
    return configs


def load_gpt_style_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sample_id, label = parts
            labels[sample_id] = label.strip()
    return labels


def load_gold_labels(path: str, field: str) -> Dict[str, str]:
    idx = GOLD_FIELD_INDEX[field]
    with open(path, "rb") as handle:
        gold_dict = pickle.load(handle)
    labels: Dict[str, str] = {}
    for sample_id, tup in gold_dict.items():
        if len(tup) > idx and tup[idx]:
            labels[sample_id] = str(tup[idx]).strip()
    return labels


def build_text_label_counts(
    source_text_file: str,
    wanted_texts: "set[str]",
    sample_to_label: Dict[str, str],
    is_keywords: bool,
    keyword_sep: str,
    keep_keyword_commas: bool,
) -> Dict[str, Counter]:
    """One pass over the (large) raw source file, building text -> Counter(label)
    only for texts we actually have embeddings for, and only counting samples
    that have a known label. Memory stays bounded by len(wanted_texts), not by
    the number of samples in the file."""
    counts: Dict[str, Counter] = defaultdict(Counter)
    with open(source_text_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            sample_id, raw_text = parts
            label = sample_to_label.get(sample_id)
            if label is None:
                continue
            text = clean_keyword_text(raw_text, keyword_sep, not keep_keyword_commas) if is_keywords else raw_text.strip()
            if text in wanted_texts:
                counts[text][label] += 1
    return counts


def resolve_text_labels(
    text_counts: Dict[str, Counter], purity_threshold: float, min_label_count: int
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for text, counter in text_counts.items():
        total = sum(counter.values())
        label, n = counter.most_common(1)[0]
        if total > 0 and (n / total) >= purity_threshold:
            resolved[text] = label
    # Drop tiny classes - StratifiedKFold and majority-baseline stats are
    # meaningless (or crash) with a class that has fewer texts than the
    # number of CV folds.
    class_counts = Counter(resolved.values())
    keep_classes = {c for c, n in class_counts.items() if n >= min_label_count}
    dropped = {c: n for c, n in class_counts.items() if n < min_label_count}
    if dropped:
        print(f"    (dropping {len(dropped)} label classes with < {min_label_count} texts: {dropped})")
    return {t: l for t, l in resolved.items() if l in keep_classes}


# --------------------------------------------------------------------------- #
# Metric 1: pairwise cosine similarity, same-label vs different-label
# --------------------------------------------------------------------------- #

def pairwise_cosine_by_label(
    embs: np.ndarray, labels: List[Optional[str]], n_pairs: int, seed: int
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = embs.shape[0]
    embs_n = normalize_rows(embs)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    sims = np.einsum("ij,ij->i", embs_n[i], embs_n[j])

    labels_arr = np.array([l if l is not None else "" for l in labels], dtype=object)
    has_label = labels_arr != ""
    both_labeled = has_label[i] & has_label[j]
    same = both_labeled & (labels_arr[i] == labels_arr[j])
    diff = both_labeled & (labels_arr[i] != labels_arr[j])

    def stats(mask: np.ndarray) -> Dict[str, float]:
        vals = sims[mask]
        if len(vals) == 0:
            return {"n": 0, "mean": None, "std": None, "p5": None, "p50": None, "p95": None}
        return {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p5": float(np.percentile(vals, 5)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
        }

    same_stats, diff_stats, overall_stats = stats(same), stats(diff), stats(np.ones_like(sims, dtype=bool))
    separation = (
        same_stats["mean"] - diff_stats["mean"]
        if same_stats["mean"] is not None and diff_stats["mean"] is not None
        else None
    )
    return {
        "overall": overall_stats,
        "same_label": same_stats,
        "diff_label": diff_stats,
        "separation_gap": separation,
        "_same_sims": sims[same],
        "_diff_sims": sims[diff],
    }


# --------------------------------------------------------------------------- #
# Metric 2: k-NN cross-validated classification vs majority baseline
# --------------------------------------------------------------------------- #

def knn_cross_val(
    embs: np.ndarray, labels: List[Optional[str]], k: int, folds: int, seed: int
) -> Dict[str, object]:
    mask = np.array([l is not None for l in labels])
    X = normalize_rows(embs[mask])
    y = np.array([l for l in labels if l is not None])
    n_classes = len(set(y))
    if len(y) < folds * 2 or n_classes < 2:
        return {"n": int(len(y)), "n_classes": n_classes, "accuracy": None, "baseline": None, "note": "not enough labeled data / classes for CV"}

    class_counts = Counter(y)
    usable_folds = min(folds, min(class_counts.values()))
    if usable_folds < 2:
        return {"n": int(len(y)), "n_classes": n_classes, "accuracy": None, "baseline": None, "note": "smallest class has <2 members, can't cross-validate"}

    skf = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=seed)
    correct = 0
    baseline_correct = 0
    total = 0
    for train_idx, test_idx in skf.split(X, y):
        clf = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)), metric="cosine")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        correct += int((preds == y[test_idx]).sum())
        majority_label = Counter(y[train_idx]).most_common(1)[0][0]
        baseline_correct += int((y[test_idx] == majority_label).sum())
        total += len(test_idx)

    return {
        "n": int(len(y)),
        "n_classes": n_classes,
        "folds_used": usable_folds,
        "accuracy": correct / total,
        "baseline_majority_accuracy": baseline_correct / total,
        "lift_over_baseline": (correct - baseline_correct) / total,
    }


# --------------------------------------------------------------------------- #
# Metric 3: qualitative nearest-neighbor spot check
# --------------------------------------------------------------------------- #

def nearest_neighbor_spotcheck(
    texts: List[str], embs: np.ndarray, labels: List[Optional[str]], n: int, top_k: int, seed: int,
    fixed_query_texts: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    embs_n = normalize_rows(embs)
    text_to_idx = {t: i for i, t in enumerate(texts)}
    if fixed_query_texts is not None:
        query_idxs = [text_to_idx[t] for t in fixed_query_texts if t in text_to_idx]
    else:
        rng = random.Random(seed)
        query_idxs = rng.sample(range(len(texts)), min(n, len(texts)))

    results = []
    for qi in query_idxs:
        sims = embs_n @ embs_n[qi]
        order = np.argsort(-sims)
        neighbors = []
        for oi in order:
            if oi == qi:
                continue
            neighbors.append({"text": texts[oi], "label": labels[oi], "cosine": float(sims[oi])})
            if len(neighbors) >= top_k:
                break
        results.append({"query_text": texts[qi], "query_label": labels[qi], "neighbors": neighbors})
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_histograms(config_label: str, label_source: str, same_sims: np.ndarray, diff_sims: np.ndarray, output_dir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not available, skipping histogram)")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 80)
    if len(diff_sims):
        ax.hist(diff_sims, bins=bins, alpha=0.55, density=True, label=f"different label (n={len(diff_sims)})", color="tab:red")
    if len(same_sims):
        ax.hist(same_sims, bins=bins, alpha=0.55, density=True, label=f"same label (n={len(same_sims)})", color="tab:blue")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"{config_label} - pairwise cosine similarity\n(label source: {label_source})")
    ax.legend()
    fig.tight_layout()
    fname = f"cosine_hist__{config_label}__{label_source}.png".replace(os.sep, "_")
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    configs = parse_config_args(args.configs)

    args.source_text_file = resolve_path(args.source_text_file)
    if args.label_file:
        args.label_file = resolve_path(args.label_file)
    if args.gold_pkl:
        args.gold_pkl = resolve_path(args.gold_pkl)

    output_dir = resolve_path(args.output_dir) if args.output_dir else os.path.dirname(configs[0][1])
    os.makedirs(output_dir, exist_ok=True)

    print("Loading embeddings for each config...")
    config_data: Dict[str, Tuple[List[str], np.ndarray]] = {}
    all_texts: "set[str]" = set()
    for label, path in configs:
        texts, embs = load_unique_embeddings(path)
        config_data[label] = (texts, embs)
        all_texts.update(texts)
        print(f"  [{label}] {path}: {len(texts)} distinct texts, dim={embs.shape[1]}")

    label_sources: Dict[str, Dict[str, str]] = {}
    if args.label_file:
        print(f"\nLoading label source '{args.label_file_name}' from {args.label_file} ...")
        label_sources[args.label_file_name] = load_gpt_style_labels(args.label_file)
        print(f"  {len(label_sources[args.label_file_name])} labeled samples")
    if args.gold_pkl:
        gold_name = args.gold_name or f"gold_{args.gold_field}"
        print(f"\nLoading label source '{gold_name}' from {args.gold_pkl} (field={args.gold_field}) ...")
        label_sources[gold_name] = load_gold_labels(args.gold_pkl, args.gold_field)
        print(f"  {len(label_sources[gold_name])} labeled samples")
    if not label_sources:
        raise SystemExit("Provide at least one of --label_file / --gold_pkl to evaluate against.")

    text_label_maps: Dict[str, Dict[str, str]] = {}
    for name, sample_to_label in label_sources.items():
        print(f"\nResolving per-text majority labels for '{name}' (this reads {args.source_text_file} once)...")
        counts = build_text_label_counts(
            args.source_text_file, all_texts, sample_to_label,
            args.is_keywords, args.keyword_sep, args.keep_keyword_commas,
        )
        resolved = resolve_text_labels(counts, args.label_purity_threshold, args.min_label_count)
        text_label_maps[name] = resolved
        print(f"  {len(counts)} of {len(all_texts)} distinct texts had >=1 labeled sample; "
              f"{len(resolved)} kept after purity>={args.label_purity_threshold} and min_label_count>={args.min_label_count} filters")

    # Fixed set of query texts for spot-checks, shared across configs so
    # they're comparable - drawn from the intersection of all configs' texts.
    common_texts = set.intersection(*(set(t) for t, _ in config_data.values())) if len(config_data) > 1 else all_texts
    rng = random.Random(args.seed)
    spotcheck_queries = rng.sample(sorted(common_texts), min(args.n_spotcheck, len(common_texts))) if common_texts else []

    summary: Dict[str, object] = {"configs": {}, "label_sources": {n: len(m) for n, m in label_sources.items()}}

    for config_label, (texts, embs) in config_data.items():
        print(f"\n{'=' * 70}\nConfig: {config_label}  (dim={embs.shape[1]}, {len(texts)} distinct texts)\n{'=' * 70}")

        if len(texts) > args.max_points:
            idx = np.array(sorted(random.Random(args.seed).sample(range(len(texts)), args.max_points)))
            texts_use = [texts[i] for i in idx]
            embs_use = embs[idx]
            print(f"  subsampled to {args.max_points} texts for pairwise/kNN (of {len(texts)})")
        else:
            texts_use, embs_use = texts, embs

        summary["configs"][config_label] = {"dim": int(embs.shape[1]), "n_unique_texts": len(texts), "n_used_for_eval": len(texts_use), "by_label_source": {}}

        for label_source_name, text_label_map in text_label_maps.items():
            labels_use = [text_label_map.get(t) for t in texts_use]
            n_labeled = sum(l is not None for l in labels_use)
            print(f"\n-- label source: {label_source_name} ({n_labeled}/{len(texts_use)} texts labeled) --")

            cos_result = pairwise_cosine_by_label(embs_use, labels_use, args.n_pairs, args.seed)
            same_sims, diff_sims = cos_result.pop("_same_sims"), cos_result.pop("_diff_sims")
            print(f"  pairwise cosine: same-label mean={cos_result['same_label']['mean']}, "
                  f"diff-label mean={cos_result['diff_label']['mean']}, "
                  f"separation_gap={cos_result['separation_gap']}")

            knn_result = knn_cross_val(embs_use, labels_use, args.knn_k, args.knn_folds, args.seed)
            print(f"  {args.knn_k}-NN CV accuracy: {knn_result.get('accuracy')} "
                  f"vs majority baseline {knn_result.get('baseline_majority_accuracy')} "
                  f"(lift {knn_result.get('lift_over_baseline')}), "
                  f"n={knn_result.get('n')}, n_classes={knn_result.get('n_classes')}")

            png_path = None
            if not args.no_plots:
                png_path = plot_histograms(config_label, label_source_name, same_sims, diff_sims, output_dir)
                if png_path:
                    print(f"  histogram saved: {png_path}")

            summary["configs"][config_label]["by_label_source"][label_source_name] = {
                "pairwise_cosine": cos_result,
                "knn": knn_result,
                "histogram_png": png_path,
            }

        if spotcheck_queries:
            main_label_source = next(iter(text_label_maps))
            labels_full = [text_label_maps[main_label_source].get(t) for t in texts]
            spot = nearest_neighbor_spotcheck(
                texts, embs, labels_full, n=args.n_spotcheck, top_k=args.spotcheck_top_k,
                seed=args.seed, fixed_query_texts=spotcheck_queries,
            )
            summary["configs"][config_label]["spotcheck"] = spot
            print(f"\n  -- nearest-neighbor spot check (label source: {main_label_source}) --")
            for item in spot:
                print(f"  query: {item['query_text']!r}  [{item['query_label']}]")
                for nb in item["neighbors"]:
                    print(f"      cos={nb['cosine']:.4f}  [{nb['label']}]  {nb['text']!r}")

    print(f"\n{'=' * 70}\nComparison summary\n{'=' * 70}")
    header = f"{'config':<20}{'dim':>6}" + "".join(f"{name + ' knn':>18}{name + ' gap':>14}" for name in text_label_maps)
    print(header)
    for config_label, cfg in summary["configs"].items():
        row = f"{config_label:<20}{cfg['dim']:>6}"
        for name in text_label_maps:
            by_src = cfg["by_label_source"].get(name, {})
            knn_acc = by_src.get("knn", {}).get("accuracy")
            gap = by_src.get("pairwise_cosine", {}).get("separation_gap")
            row += f"{('%.4f' % knn_acc) if knn_acc is not None else 'n/a':>18}{('%.4f' % gap) if gap is not None else 'n/a':>14}"
        print(row)

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\nFull results written to {summary_path}")


if __name__ == "__main__":
    main()
