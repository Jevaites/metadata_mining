#!/usr/bin/env python3
"""
Do the new keywords map to Metalog ontology labels better than the old ones?

Any number of keyword arms, same samples, TF-IDF + ridge, 5-fold CV GROUPED BY
STUDY so no study appears in both train and test (extraction-step-improvements.md
established TF-IDF is within 0.1 points of dense embeddings on keywords, so the
representation is held fixed and only the text differs).

Reports micro and macro top-1 for biome / feature / material. Macro is the one
that matters: the whole point is the long tail. Differences carry a paired
bootstrap over studies, and results are also broken out by Metalog domain so a
failure in a small stratum cannot hide inside an animal-dominated average.

    python3 scripts/arms_mapping_cv.py --arms mini_v3_9k mini_old_9k
"""
import argparse, gzip, os, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold

TARGETS = ["biome", "feature", "material"]


def kw(v):
    return " ".join(w.strip() for w in v.strip().strip("{}").split(",") if w.strip())


def boot_ci(correct_a, correct_b, groups, boot, seed=0):
    """Paired difference in accuracy, resampling studies."""
    keys = list(groups)
    a = np.array([correct_a[list(groups[k])].sum() for k in keys], float)
    b = np.array([correct_b[list(groups[k])].sum() for k in keys], float)
    n = np.array([len(groups[k]) for k in keys], float)
    obs = (a.sum() - b.sum()) / n.sum()
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(boot, len(keys)))
    d = (a[idx].sum(1) - b[idx].sum(1)) / n[idx].sum(1)
    return obs, *np.quantile(d, [.025, .975])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--arms", nargs="+", required=True, help="tags: GPT_keywords_<tag>.txt")
    p.add_argument("--baseline", default="GPT_keywords.txt",
                   help="production keywords, added as the 'old' arm; '' to skip")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min_class", type=int, default=5, help="drop labels rarer than this")
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    root = os.path.expanduser(a.root)
    lat = f"{root}/sidequest/latest"

    arms = {}
    for tag in a.arms:
        d = {}
        for line in open(f"{lat}/GPT_keywords_{tag}.txt", encoding="utf-8", errors="replace"):
            s, _, v = line.rstrip("\n").partition("\t")
            if v.strip():
                d[s] = kw(v)
        arms[tag] = d

    meta = {}
    with gzip.open(f"{root}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        ix = {c: h.index(c) for c in ["sample_id", "study_code", "domain"] + TARGETS}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(ix.values()):
                meta[f[ix["sample_id"]]] = {c: f[ix[c]] for c in ix}

    ids = sorted(set.intersection(*(set(d) for d in arms.values())) & set(meta))
    if a.baseline:
        base, want = {}, set(ids)
        for line in open(f"{lat}/{a.baseline}", encoding="utf-8", errors="replace"):
            s, _, v = line.rstrip("\n").partition("\t")
            if s in want and v.strip():
                base[s] = kw(v)
            if len(base) == len(want):
                break
        arms = {"old": base, **arms}
        ids = sorted(set(ids) & set(base))
    names = list(arms)
    print(f"{len(ids):,} samples, {len({meta[s]['study_code'] for s in ids})} studies, "
          f"{len(names)} arms: {', '.join(names)}\n")

    groups = defaultdict(set)
    for i, s in enumerate(ids):
        groups[meta[s]["study_code"]].add(i)
    study_of = np.array([meta[s]["study_code"] for s in ids])
    domain_of = np.array([meta[s]["domain"] or "unknown" for s in ids])

    results = {}
    for tgt in TARGETS:
        y_raw = np.array([meta[s][tgt] or "" for s in ids])
        keep = np.array([bool(v) for v in y_raw])
        cnt = Counter(y_raw[keep])
        keep &= np.array([cnt.get(v, 0) >= a.min_class for v in y_raw])
        idx = np.flatnonzero(keep)
        if len(set(y_raw[idx])) < 2:
            print(f"{tgt}: too few labels, skipped"); continue
        y, g = y_raw[idx], study_of[idx]
        n_splits = min(a.folds, len(set(g)))
        print(f"{tgt}: {len(idx):,} labelled samples, {len(set(y))} classes, "
              f"{n_splits}-fold grouped by study")
        for name in names:
            X_text = [arms[name][ids[i]] for i in idx]
            pred = np.empty(len(idx), dtype=object)
            for tr, te in GroupKFold(n_splits=n_splits).split(X_text, y, g):
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                Xtr = vec.fit_transform([X_text[i] for i in tr])
                Xte = vec.transform([X_text[i] for i in te])
                clf = RidgeClassifier().fit(Xtr, y[tr])
                pred[te] = clf.predict(Xte)
            ok = (pred == y).astype(float)
            micro = ok.mean()
            macro = np.mean([ok[y == c].mean() for c in sorted(set(y))])
            results.setdefault(tgt, {})[name] = (ok, y, idx)
            print(f"   {name:<16} micro {micro:6.3f}   macro {macro:6.3f}")
        if len(names) >= 2:
            x, z = names[-2], names[-1]
            okx, okz = results[tgt][x][0], results[tgt][z][0]
            sub = defaultdict(list)
            for pos, i in enumerate(idx):
                sub[study_of[i]].append(pos)
            obs, lo, hi = boot_ci(okx, okz, {k: v for k, v in sub.items()}, a.boot)
            flag = "" if lo <= 0 <= hi else "   <-- CI excludes zero"
            print(f"   {x} - {z}: micro {obs*100:+.1f} pp  [{lo*100:+.1f}, {hi*100:+.1f}]{flag}")
        print()

    print("BY DOMAIN (micro top-1)")
    doms = [d for d, c in Counter(domain_of).most_common() if c >= 50]
    print("%-10s %-9s %6s " % ("target", "domain", "n") + "".join("%16s" % n for n in names))
    for tgt in results:
        for dom in doms:
            rows = []
            for name in names:
                ok, y, idx = results[tgt][name]
                m = domain_of[idx] == dom
                rows.append(ok[m].mean() if m.sum() else float("nan"))
            ok, y, idx = results[tgt][names[0]]
            n = int((domain_of[idx] == dom).sum())
            if n < 30:
                continue
            print("%-10s %-9s %6d " % (tgt, dom[:9], n) + "".join("%15.3f " % r for r in rows))


if __name__ == "__main__":
    main()
