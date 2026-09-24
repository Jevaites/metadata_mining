#!/usr/bin/env python3
"""
Compare the new embeddings against Dany's previous ones, on overlapping samples.

    Dany : GPT-3.5 text -> text-embedding-3-small, 1536d, 2,056,410 samples
    new  : GPT-5   text -> text-embedding-3-large, 1024d, 3,437,092 samples

The two runs live in different spaces (different model AND different dimension),
so their vectors cannot be compared to each other directly. What CAN be compared
is the geometry each induces: take the same pairs of samples, measure cosine in
each space, and ask whether the two agree.

THE CONFOUND. Two things changed at once - the annotating LLM and the embedding
model. Of the 1,655,663 overlapping samples only 19.9% have an identical
sub-biome string, and 41 (0.002%) an identical keyword string. So:

    sub_biomes / same text     identical input -> isolates the EMBEDDING MODEL
    sub_biomes / changed text  embedding model + LLM annotation together
    keywords                   no same-text subset exists; combined effect only

Agreement says how *different* the runs are, never which is *better*. For that
you need an external criterion, so each arm is also scored by 5-NN accuracy
against three label sources: gold_dict (curated, neutral to both runs) and each
era's own GPT_biomes (each circular in favour of its own run - a verdict that
holds on all three is the one to trust).

Memory-lean: no full-corpus text dictionary is ever held. Samples are classified
same/changed using hashes, then only the chosen few thousand are materialised.

    python3 scripts/compare_to_previous_embeddings.py --n_samples 15000
"""

import argparse
import json
import os
import pickle
from collections import Counter

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from embed_subbiomes_keywords import clean_text

DEFAULT_ROOT = "~/MicrobeAtlasProject"
OLD = NEW = GOLD = None                 # set from --root in main()
SEQ, ACCENT = "Blues", "#eb6834"


def new_unique(target):
    return f"{NEW}/embeddings/GPT_{target}_unique_embeddings__text-embedding-3-large__dim1024__full.h5"


def old_embeddings(target):
    return f"{OLD}/GPT_{target}_embeddings{'_aligned' if target == 'sub_biomes' else ''}.h5"


def stream(path, is_keywords):
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid and raw.strip():
            yield sid, clean_text(raw, is_keywords)


def classify(target, is_keywords):
    """Split the overlap into same-text and changed-text, holding only hashes."""
    old_hash = {sid: hash(text) for sid, text in stream(f"{OLD}/GPT_{target}.txt", is_keywords)}
    same, changed = [], []
    for sid, text in stream(f"{NEW}/GPT_{target}.txt", is_keywords):
        h = old_hash.get(sid)
        if h is not None:
            (same if h == hash(text) else changed).append(sid)
    return same, changed


def texts_for(path, wanted, is_keywords=False):
    return {sid: text for sid, text in stream(path, is_keywords) if sid in wanted}


def rows_for(h5_path, column, wanted, chunk=200_000):
    """value -> row index, reading a string column in chunks so it never all sits in RAM."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        dset = f[column]
        for i in range(0, dset.shape[0], chunk):
            for k, v in enumerate(dset[i:i + chunk]):
                v = v.decode()
                if v in wanted:
                    out[v] = i + k
    return out


def fetch(h5_path, rows, column="embeddings", batch=512):
    """Read these rows (duplicates and any order allowed) from a big HDF5 column.

    h5py fancy indexing demands strictly increasing indices, and rows repeat here
    because many samples share one text in the deduplicated new tables - so read
    each distinct row once, in order, then expand back.

    Read in small batches: Dany's keywords file is chunked (313, 48), so one row
    spans 32 chunks and a single fancy read of ~16k rows allocates several GB
    inside h5py (measured 0.46 GB for just 2000 rows) and gets OOM-killed. The
    sub-biome file is contiguous and does not care either way.
    """
    distinct, inverse = np.unique(np.asarray(rows), return_inverse=True)
    idx = distinct.tolist()
    with h5py.File(h5_path, "r") as f:
        dset = f[column]
        block = np.concatenate([dset[idx[s:s + batch]] for s in range(0, len(idx), batch)])
    return block[inverse]


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cosines(block, i, j, chunk=20_000):
    """Cosine of the given index pairs, in chunks - block[i] on 200k pairs would
    materialise a 200000 x dim copy (1.2 GB at 1536d) and blow up memory."""
    return np.concatenate([np.einsum("ij,ij->i", block[i[s:s + chunk]], block[j[s:s + chunk]])
                           for s in range(0, len(i), chunk)])


def geometry(old, new, rng, n_pairs):
    i = rng.integers(0, len(old), n_pairs)
    j = rng.integers(0, len(old), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    a, b = cosines(old, i, j), cosines(new, i, j)
    return i, j, a, b, {
        "pearson": float(pearsonr(a, b)[0]), "spearman": float(spearmanr(a, b)[0]),
        "dany_mean": float(a.mean()), "dany_sd": float(a.std()),
        "new_mean": float(b.mean()), "new_sd": float(b.std()), "n_pairs": int(len(a))}


def neighbour_overlap(old, new, rng, k, n_queries):
    """Of each sample's k nearest neighbours, what fraction are the same in both
    runs? Local agreement, which is what clustering actually depends on."""
    q = rng.choice(len(old), min(n_queries, len(old)), replace=False)
    picks = []
    for block in (old, new):
        sims = block[q] @ block.T
        sims[np.arange(len(q)), q] = -np.inf
        picks.append(np.argpartition(-sims, k, axis=1)[:, :k])
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(*picks)]))


def quality(vectors, labels, seed, k=5, folds=5, max_n=6000):
    """5-NN CV accuracy against a label source, plus the majority baseline.

    Capped at max_n samples: sklearn's cosine k-NN builds a dense float64
    distance matrix, so 16k x 1536 exhausts memory. The cap also makes arms
    directly comparable, since k-NN accuracy depends on point density."""
    mask = np.array([l is not None for l in labels])
    X, y = vectors[mask], np.array([l for l in labels if l is not None])
    if len(y) > max_n:
        pick = np.random.default_rng(seed).choice(len(y), max_n, replace=False)
        X, y = X[pick], y[pick]
    counts = Counter(y)
    keep = np.array([counts[v] >= folds for v in y], dtype=bool)
    X, y = X[keep], y[keep]
    if len(y) < 2 * folds or len(set(y)) < 2:
        return {"n": int(len(y)), "accuracy": None}
    correct = base = total = 0
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = KNeighborsClassifier(k, metric="cosine").fit(X[tr], y[tr])
        correct += int((m.predict(X[te]) == y[te]).sum())
        base += int((y[te] == Counter(y[tr]).most_common(1)[0][0]).sum())
        total += len(te)
    return {"n": int(len(y)), "n_classes": int(len(set(y))),
            "accuracy": correct / total, "baseline": base / total}


def plot(arms, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(arms), figsize=(4.7 * len(arms), 4.6), squeeze=False)
    for ax, (name, a, b, st) in zip(axes[0], arms):
        ax.hexbin(a, b, gridsize=70, bins="log", cmap=SEQ, mincnt=1, linewidths=0)
        lim = [min(a.min(), b.min()) - 0.02, 1.02]
        ax.plot(lim, lim, color=ACCENT, linewidth=1.5, linestyle="--", label="y = x")
        ax.set(xlim=lim, ylim=lim, xlabel="cosine - Dany (3-small, 1536d)",
               ylabel="cosine - new (3-large, 1024d)")
        ax.set_title(f"{name}\nPearson {st['pearson']:.3f} | Spearman {st['spearman']:.3f}"
                     f" | top-{st['k']} neighbours {st['neighbour_overlap']:.0%}", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nfigure: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="Directory holding sidequest/ and gold_dict.pkl.")
    p.add_argument("--n_samples", type=int, default=15000, help="Samples per arm.")
    p.add_argument("--n_pairs", type=int, default=200_000)
    p.add_argument("--n_queries", type=int, default=2000)
    p.add_argument("--k", type=int, default=20, help="Neighbourhood size.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    global OLD, NEW, GOLD
    root = os.path.expanduser(args.root)
    OLD, NEW, GOLD = f"{root}/sidequest", f"{root}/sidequest/latest", f"{root}/gold_dict.pkl"
    out_dir = args.output_dir or f"{NEW}/embeddings/vs_previous"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    gold_dict = pickle.load(open(GOLD, "rb"))
    gold_ids = set(gold_dict)

    summary, arms = {}, []
    for target in ("sub_biomes", "keywords"):
        is_kw = target == "keywords"
        print(f"\n{'=' * 70}\n{target}\n{'=' * 70}", flush=True)
        same, changed = classify(target, is_kw)
        overlap = len(same) + len(changed)
        print(f"overlap {overlap:,} | identical text {len(same):,} ({len(same)/overlap:.1%})", flush=True)

        splits = {"same text": same, "changed text": changed} if len(same) >= 500 \
            else {"all overlap": same + changed}
        chosen = {}
        for arm, ids in splits.items():
            ids = sorted(ids)
            gold_here = sorted(set(ids) & gold_ids)      # keep every gold sample
            if len(ids) > args.n_samples:
                ids = [ids[i] for i in sorted(rng.choice(len(ids), args.n_samples, replace=False))]
            chosen[arm] = sorted(set(ids) | set(gold_here))
            print(f"  {arm}: {len(chosen[arm])} samples ({len(gold_here)} of them gold)", flush=True)
        del same, changed, splits

        wanted = {s for ids in chosen.values() for s in ids}
        old_txt = texts_for(f"{OLD}/GPT_{target}.txt", wanted, is_kw)
        new_txt = texts_for(f"{NEW}/GPT_{target}.txt", wanted, is_kw)
        old_row = rows_for(old_embeddings(target), "sample_ids", wanted)
        text_row = rows_for(new_unique(target), "texts", set(new_txt.values()))
        labels = {"gpt_biome_dany": texts_for(f"{OLD}/GPT_biomes.txt", wanted),
                  "gpt_biome_new": texts_for(f"{NEW}/GPT_biomes.txt", wanted),
                  "gold_biome": {s: str(v[1]).strip()
                                 for s, v in gold_dict.items() if s in wanted and v[1]}}
        print(f"loaded: {len(old_row)} old rows, {len(text_row)} new text rows, "
              f"gold labels for {len(labels['gold_biome'])}", flush=True)

        for arm, ids in chosen.items():
            ids = [s for s in ids if s in old_row and new_txt.get(s) in text_row]
            old_v = fetch(old_embeddings(target), [old_row[s] for s in ids])
            new_v = fetch(new_unique(target), [text_row[new_txt[s]] for s in ids])
            ok = np.isfinite(old_v).all(axis=1) & np.isfinite(new_v).all(axis=1)
            if (~ok).sum():
                print(f"  dropped {int((~ok).sum())} samples with a non-finite vector in one run", flush=True)
            ids = [s for s, good in zip(ids, ok) if good]
            old_v, new_v = unit(old_v[ok]), unit(new_v[ok])

            name = f"{target} - {arm}"
            print(f"\n  {name}: {len(ids)} samples", flush=True)
            pi, pj, a, b, st = geometry(old_v, new_v, rng, args.n_pairs)
            st["k"] = args.k
            st["neighbour_overlap"] = neighbour_overlap(old_v, new_v, rng, args.k, args.n_queries)
            st["n_samples"] = len(ids)
            print(f"    Pearson {st['pearson']:.3f}  Spearman {st['spearman']:.3f}  "
                  f"top-{args.k} neighbour overlap {st['neighbour_overlap']:.1%}", flush=True)
            print(f"    mean cosine  Dany {st['dany_mean']:.3f} (sd {st['dany_sd']:.3f})  "
                  f"new {st['new_mean']:.3f} (sd {st['new_sd']:.3f})", flush=True)

            st["quality"] = {}
            for src, mapping in labels.items():
                y = [mapping.get(s) for s in ids]
                qd, qn = quality(old_v, y, args.seed), quality(new_v, y, args.seed)
                st["quality"][src] = {"dany": qd, "new": qn}
                if qd.get("accuracy") is not None:
                    print(f"    5-NN vs {src:<15} Dany {qd['accuracy']:.4f}  new {qn['accuracy']:.4f}"
                          f"   (n={qd['n']}, baseline {qd['baseline']:.3f})", flush=True)

            st["examples"] = {}
            for label, order in [("high for Dany, low for new", np.argsort(b - a)),
                                 ("high for new, low for Dany", np.argsort(a - b))]:
                rows, printed, seen = [], f"    {label}:", set()
                for idx in order:
                    if len(rows) >= 3:
                        break
                    s1, s2 = ids[pi[idx]], ids[pj[idx]]
                    key = tuple(sorted((new_txt[s1], new_txt[s2])))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"cos_dany": round(float(a[idx]), 3), "cos_new": round(float(b[idx]), 3),
                                 "dany": [old_txt[s1], old_txt[s2]], "new": [new_txt[s1], new_txt[s2]]})
                    printed += (f"\n      Dany {a[idx]:+.3f} / new {b[idx]:+.3f}"
                                f"\n        Dany: {old_txt[s1][:62]!r} <-> {old_txt[s2][:62]!r}"
                                f"\n        new : {new_txt[s1][:62]!r} <-> {new_txt[s2][:62]!r}")
                st["examples"][label] = rows
                print(printed, flush=True)

            summary[name] = st
            arms.append((name, a, b, st))

    plot(arms, os.path.join(out_dir, "cosine_agreement.png"))
    path = os.path.join(out_dir, "comparison_summary.json")
    json.dump(summary, open(path, "w", encoding="utf-8"), indent=2, default=str)
    print(f"summary: {path}", flush=True)


if __name__ == "__main__":
    main()
