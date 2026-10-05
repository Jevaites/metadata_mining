#!/usr/bin/env python3
"""
Is the keywords-beat-sub-biomes result real, or is it study identification?

cluster_agreement.py found keywords far ahead of sub-biomes against community
clusters (purity@10 0.82 vs 0.47). But a coarse cluster is largely a study -
among Metalog-linked samples the median cluster draws 74.8% of its members from
one study, and 37.8% of clusters are >=90% one study. That evaluation never
blocked study, while the ontology-mapping CV is grouped by study, where keywords
barely beat sub-biomes and LOSE on macro accuracy. This tests whether the gap is
study leakage.

On a study-balanced reference set (clusters spanning >= MIN_STUDIES studies, at
most PER_STUDY samples from each), three numbers per representation:

    purity@10          same-cluster share of the 10 nearest neighbours
    purity@10 blocked  same, with same-study candidates removed first
    study@10           same-STUDY share of the 10 nearest - identity leakage,
                       measured directly rather than inferred

If the hypothesis holds: keywords have far higher study@10, and their advantage
over sub-biomes shrinks once blocked.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_study_blocked.py
"""
import gzip, json, os, sys
from collections import Counter, defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"
UNIQ = {"keywords": f"{NEW}/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5",
        "sub_biomes": f"{NEW}/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5"}

MIN_STUDIES, PER_STUDY, MAX_PER_CLUSTER, K = 5, 8, 40, 10
TAG, SEED, SLAB = "coarse", 42, 50_000


def studies():
    st = {}
    with gzip.open(f"{ROOT}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        i_s, i_st = h.index("sample_id"), h.index("study_code")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(i_s, i_st):
                st[f[i_s]] = f[i_st]
    return st


def load_join(tag):
    d = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            d[s] = c
    return d


def texts_for(path, wanted, is_kw):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid in wanted and raw.strip():
            out[sid] = clean_text(raw, is_kw)
    return out


def vectors(path, ids, txt):
    """Resolve texts to rows, then read the table in sequential slabs."""
    need = {txt[s] for s in ids}
    row_of = {}
    with h5py.File(path, "r") as f:
        d = f["texts"]
        for i in range(0, d.shape[0], 200_000):
            for k, v in enumerate(d[i:i + 200_000]):
                v = v.decode()
                if v in need:
                    row_of[v] = i + k
    keep = [i for i, s in enumerate(ids) if txt[s] in row_of]
    rows = np.array([row_of[txt[ids[i]]] for i in keep])
    order = np.argsort(rows)
    srt = rows[order]
    with h5py.File(path, "r") as f:
        d = f["embeddings"]
        V = np.zeros((len(keep), d.shape[1]), dtype=np.float32)
        for start in range(0, d.shape[0], SLAB):
            lo, hi = np.searchsorted(srt, [start, start + SLAB])
            if hi == lo:
                continue
            slab = d[start:start + SLAB]
            V[order[lo:hi]] = slab[srt[lo:hi] - start]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return keep, V


def metrics(V, cl, stu):
    """Per-query cluster purity (plain and study-blocked) and study purity."""
    n = len(cl)
    plain, blocked, samestudy, usable = [], [], [], []
    for s in range(0, n, 256):
        blk = np.arange(s, min(s + 256, n))
        sims = V[blk] @ V.T
        sims[np.arange(len(blk)), blk] = -np.inf
        nb = np.argpartition(-sims, K, axis=1)[:, :K]
        plain.append((cl[nb] == cl[blk][:, None]).mean(axis=1))
        samestudy.append((stu[nb] == stu[blk][:, None]).mean(axis=1))
        m = sims.copy()
        m[stu[None, :] == stu[blk][:, None]] = -np.inf     # drop every same-study candidate
        nb2 = np.argpartition(-m, K, axis=1)[:, :K]
        blocked.append((cl[nb2] == cl[blk][:, None]).mean(axis=1))
        # a query is usable only if >=K same-cluster candidates survive the block
        usable.append(((cl[None, :] == cl[blk][:, None]) & (stu[None, :] != stu[blk][:, None])
                       ).sum(axis=1) >= K)
    return (np.concatenate(plain), np.concatenate(blocked),
            np.concatenate(samestudy), np.concatenate(usable))


def boot(diff, groups, n=600, seed=SEED):
    """Paired bootstrap over CLUSTERS - queries in one cluster are not independent."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for d, g in zip(diff, groups):
        by[g].append(d)
    keys = list(by)
    out = []
    for _ in range(n):
        pick = [by[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        flat = [x for v in pick for x in v]
        if flat:
            out.append(np.mean(flat))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def main():
    rng = np.random.default_rng(SEED)
    st, join = studies(), load_join(TAG)
    per = defaultdict(lambda: defaultdict(list))
    for s, study in st.items():
        if s in join:
            per[join[s]][study].append(s)

    ids, cl, stu = [], [], []
    for c, by_study in sorted(per.items()):
        if len(by_study) < MIN_STUDIES:
            continue
        chosen = []
        for study, v in sorted(by_study.items()):
            v = sorted(v)
            take = v if len(v) <= PER_STUDY else [v[i] for i in sorted(rng.choice(len(v), PER_STUDY, replace=False))]
            chosen += [(s, study) for s in take]
        if len(chosen) > MAX_PER_CLUSTER:
            chosen = [chosen[i] for i in sorted(rng.choice(len(chosen), MAX_PER_CLUSTER, replace=False))]
        if len(chosen) < 2 * K:
            continue
        for s, study in chosen:
            ids.append(s); cl.append(c); stu.append(study)
    print(f"reference set: {len(ids):,} samples, {len(set(cl))} clusters, "
          f"{len(set(stu))} studies", flush=True)

    wanted = set(ids)
    res = {}
    for target, is_kw in [("keywords", True), ("sub_biomes", False)]:
        txt = texts_for(f"{NEW}/GPT_{target}.txt", wanted, is_kw)
        sub = [i for i, s in enumerate(ids) if s in txt]
        keep, V = vectors(UNIQ[target], [ids[i] for i in sub], txt)
        idx = [sub[i] for i in keep]
        c = np.array([cl[i] for i in idx]); u = np.array([stu[i] for i in idx])
        p, b, ss, ok = metrics(V, c, u)
        res[target] = {"ids": [ids[i] for i in idx], "cluster": c, "study": u,
                       "plain": p, "blocked": b, "study10": ss, "usable": ok}
        print(f"  {target}: {len(idx):,} samples with a vector, "
              f"{int(ok.sum()):,} usable for the blocked metric [dim {V.shape[1]}]", flush=True)
        del V

    common = sorted(set(res["keywords"]["ids"]) & set(res["sub_biomes"]["ids"]))
    pos = {t: {s: i for i, s in enumerate(res[t]["ids"])} for t in res}
    sel = {t: np.array([pos[t][s] for s in common]) for t in res}
    use = res["keywords"]["usable"][sel["keywords"]] & res["sub_biomes"]["usable"][sel["sub_biomes"]]
    cgrp = res["keywords"]["cluster"][sel["keywords"]]
    print(f"\n  paired on {len(common):,} samples, {int(use.sum()):,} usable when blocked", flush=True)

    base = float(np.mean([np.mean(cgrp == x) for x in cgrp[:500]]))
    print(f"\n  {'representation':<14} {'purity@10':>11} {'blocked':>11} {'study@10':>11}")
    out = {"n_samples": len(common), "n_usable_blocked": int(use.sum()),
           "n_clusters": int(len(set(cgrp))), "approx_baseline": base}
    for t in ("keywords", "sub_biomes"):
        r = {k: res[t][k][sel[t]] for k in ("plain", "blocked", "study10")}
        print(f"  {t:<14} {r['plain'].mean():>11.3f} {r['blocked'][use].mean():>11.3f} "
              f"{r['study10'].mean():>11.3f}")
        out[t] = {"purity": float(r["plain"].mean()),
                  "purity_blocked": float(r["blocked"][use].mean()),
                  "study10": float(r["study10"].mean())}

    kw = {k: res["keywords"][k][sel["keywords"]] for k in ("plain", "blocked", "study10")}
    sb = {k: res["sub_biomes"][k][sel["sub_biomes"]] for k in ("plain", "blocked", "study10")}
    print(f"\n  keywords − sub-biomes, paired bootstrap over clusters:")
    for name, a, b, m in [("purity@10", kw["plain"], sb["plain"], np.ones(len(use), bool)),
                          ("purity@10 blocked", kw["blocked"], sb["blocked"], use),
                          ("study@10", kw["study10"], sb["study10"], np.ones(len(use), bool))]:
        d = (a - b)[m]
        lo, hi = boot(d, cgrp[m])
        print(f"    {name:<20} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={int(m.sum()):,})")
        out[f"diff_{name.replace(' ', '_').replace('@', '')}"] = {
            "mean": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(m.sum())}

    json.dump(out, open(f"{OUT}/cluster_study_blocked.json", "w"), indent=2, default=str)
    print(f"\nwrote {OUT}/cluster_study_blocked.json", flush=True)


if __name__ == "__main__":
    main()
