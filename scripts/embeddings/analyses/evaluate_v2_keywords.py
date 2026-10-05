#!/usr/bin/env python3
"""
Does the identity-stripped keyword text actually work better?

Runs on the 2,000-sample dev set produced by extract_keywords_v2.py, paired
against the same samples' original keywords. No API calls: TF-IDF is used for the
mapping model because extraction-step-improvements.md established TF-IDF is
within 0.1 points of dense embeddings on keywords, and the comparison here is
old-vs-new with the representation held fixed.

Three questions:

  1. mapping      micro and macro top-1 for biome/feature/material, 5-fold CV
                  GROUPED BY STUDY. Macro is the one that matters - the whole
                  point is the long tail.
  2. collisions   the new text is more degenerate (36.4% of samples now share a
                  string, up from 15.5%). Degeneracy is only bad if it merges
                  samples that differ. P(same label | same string) says which.
  3. damage       samples left with fewer than 5 keywords, or none at all.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/evaluate_v2_keywords.py
"""
import gzip, json, os, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
L = f"{ROOT}/sidequest/latest"
SLOTS = ("biome", "feature", "material")


def load(path, keys=None):
    d = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        s, _, v = line.rstrip("\n").partition("\t")
        if s and (keys is None or s in keys):
            d[s] = v.strip()
    return d


def metalog():
    rows = {}
    with gzip.open(f"{ROOT}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        ix = {c: h.index(c) for c in ("sample_id", "study_code", *SLOTS)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(ix.values()):
                rows[f[ix["sample_id"]]] = {c: f[ix[c]] for c in ix}
    return rows


def cluster_join():
    d = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_coarse.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            d[s] = c
    return d


def mapping(texts, labels, groups, folds=5):
    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(texts)
    y, g = np.asarray(labels, dtype=object), np.asarray(groups)
    keep = np.array([v not in ("", "NA", "nan", "None", None) for v in y])
    X, y, g = X[keep], y[keep], g[keep]
    c = Counter(y)
    keep = np.array([c[v] >= folds for v in y])
    X, y, g = X[keep], y[keep], g[keep]
    if len(y) < 50 or len(set(g)) < folds or len(set(y)) < 2:
        return None
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(folds).split(X, y, g):
        pred[te] = RidgeClassifier().fit(X[tr], y[tr]).predict(X[te])
    per = [float((pred[y == k] == k).mean()) for k in sorted(set(y))]
    return {"n": int(len(y)), "classes": len(per),
            "top1": float((pred == y).mean()), "macro": float(np.mean(per))}


def collisions(texts, labels):
    """P(two samples sharing a string share a label), and how many pairs that is."""
    g = defaultdict(list)
    for s, t in texts.items():
        if s in labels and t.strip():
            g[t].append(labels[s])
    same = tot = 0
    for v in g.values():
        if len(v) < 2:
            continue
        same += sum(k * (k - 1) // 2 for k in Counter(v).values())
        tot += len(v) * (len(v) - 1) // 2
    base_all = [l for s, l in labels.items() if s in texts]
    bc = Counter(base_all); bn = len(base_all)
    base = sum(k * (k - 1) for k in bc.values()) / (bn * (bn - 1)) if bn > 1 else None
    return {"pairs": tot, "purity": same / tot if tot else None, "base_rate": base}


def main():
    new_kw = load(f"{L}/GPT_keywords_v2.txt")
    ids = set(new_kw)
    old_kw = load(f"{L}/GPT_keywords.txt", ids)
    meta, clus = metalog(), cluster_join()
    both = sorted(ids & set(old_kw))
    print(f"paired on {len(both):,} samples\n")

    variants = {"old": {s: old_kw[s] for s in both}, "new": {s: new_kw[s] for s in both}}

    print("=== 1. mapping, TF-IDF + ridge, 5-fold CV grouped by study ===")
    print(f"  {'slot':<9} {'variant':<6} {'n':>6} {'classes':>8} {'top1':>8} {'macro':>8}")
    out = {}
    for slot in SLOTS:
        ev = [s for s in both if s in meta]
        for name, txt in variants.items():
            r = mapping([txt[s] for s in ev], [meta[s][slot] for s in ev],
                        [meta[s]["study_code"] for s in ev])
            if r:
                out[f"map|{slot}|{name}"] = r
                print(f"  {slot:<9} {name:<6} {r['n']:>6,} {r['classes']:>8} "
                      f"{r['top1']:>8.4f} {r['macro']:>8.4f}")
        a, b = out.get(f"map|{slot}|old"), out.get(f"map|{slot}|new")
        if a and b:
            print(f"  {'':<9} {'DIFF':<6} {'':>6} {'':>8} {b['top1']-a['top1']:>+8.4f} "
                  f"{b['macro']-a['macro']:>+8.4f}")

    print("\n=== 2. are the new collisions good ones? P(same label | same string) ===")
    print(f"  {'label':<22} {'variant':<6} {'pairs':>7} {'purity':>8} {'base':>8} {'lift':>7}")
    for lname, labels in [("Metalog biome", {s: meta[s]["biome"] for s in both if s in meta}),
                          ("Metalog material", {s: meta[s]["material"] for s in both if s in meta}),
                          ("community cluster", {s: clus[s] for s in both if s in clus})]:
        for name, txt in variants.items():
            r = collisions(txt, labels)
            out[f"col|{lname}|{name}"] = r
            if r["purity"] is None:
                continue
            print(f"  {lname:<22} {name:<6} {r['pairs']:>7,} {r['purity']:>8.3f} "
                  f"{r['base_rate']:>8.3f} {r['purity']/r['base_rate']:>6.1f}x")

    print("\n=== 3. damage check ===")
    cnt = lambda v: len([x for x in v.strip('{} ').split(',') if x.strip()])
    short = [s for s in both if cnt(new_kw[s]) < 5]
    empty = [s for s in both if cnt(new_kw[s]) == 0 or new_kw[s].strip('{} ').lower() in ("", "na")]
    print(f"  fewer than 5 keywords: {len(short)} ({len(short)/len(both):.1%})")
    print(f"  empty or NA          : {len(empty)}")
    for s in empty[:4]:
        print(f"    {s}  old: {old_kw[s][:80]}")
    print("  shortest non-empty examples:")
    for s in sorted([x for x in short if x not in empty], key=lambda x: cnt(new_kw[x]))[:4]:
        print(f"    {s}  new: {new_kw[s][:70]}")
        print(f"    {'':<12} old: {old_kw[s][:70]}")

    json.dump(out, open(f"{L}/embeddings/vs_clusters/evaluate_v2_keywords.json", "w"),
              indent=2, default=str)
    print(f"\nwrote {L}/embeddings/vs_clusters/evaluate_v2_keywords.json")


if __name__ == "__main__":
    main()
