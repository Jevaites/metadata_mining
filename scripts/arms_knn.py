#!/usr/bin/env python3
"""
In EMBEDDING space, do the new keywords stop fingerprinting studies?

study-leakage-in-cluster-evaluation.md showed the old keyword embeddings put
~75% of a sample's ten nearest neighbours inside its own study, which inflated
every purity number computed on them. This measures, for two or more arms on the
identical sample set, model and dimension:

  study@k           share of the k nearest neighbours in the query's own study
  study@k norm      the same, divided by the ceiling min(k, peers)/k, because a
                    sample with only 3 same-study peers cannot score above 0.3
  purity@k          share of labelled neighbours sharing the query's Metalog label
  blocked purity@k  purity@k with same-study neighbours removed first - the only
                    purity figure that is not contaminated by study leakage

Differences carry a paired bootstrap over studies; everything is also broken out
by Metalog domain.

    python3 scripts/arms_knn.py \
        --arms v3=.../v3_9k/GPT_keywords_embeddings__...__full.h5 \
               v1=.../old_9k/GPT_keywords_embeddings__...__full.h5
"""
import argparse, gzip, os
from collections import Counter, defaultdict

import h5py
import numpy as np

TARGETS = ["biome", "feature", "material"]


def load(path):
    with h5py.File(os.path.expanduser(path), "r") as h:
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["sample_ids"][:]]
        X = np.asarray(h["embeddings"][:], dtype=np.float32)
    return ids, X


def topk(X, k, blocked_by=None, chunk=512):
    """Indices of the k nearest neighbours (cosine), excluding self.
    blocked_by: array of group labels; neighbours in the query's group are dropped."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = len(X)
    out = np.empty((n, k), dtype=np.int32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        S = X[a:b] @ X.T
        S[np.arange(b - a), np.arange(a, b)] = -2.0          # self
        if blocked_by is not None:
            same = blocked_by[a:b, None] == blocked_by[None, :]
            S[same] = -2.0
        out[a:b] = np.argpartition(-S, k, axis=1)[:, :k]
        rows = np.arange(b - a)[:, None]
        out[a:b] = out[a:b][rows, np.argsort(-S[rows, out[a:b]], axis=1)]
    return out


def boot(vals_a, vals_b, study, boot_n, seed=0):
    """Paired difference in a per-sample quantity, resampling studies."""
    g = defaultdict(list)
    for i, s in enumerate(study):
        g[s].append(i)
    keys = list(g)
    A = np.array([np.nansum(vals_a[g[k]]) for k in keys], float)
    B = np.array([np.nansum(vals_b[g[k]]) for k in keys], float)
    N = np.array([np.sum(~np.isnan(vals_a[g[k]])) for k in keys], float)
    obs = (A.sum() - B.sum()) / N.sum()
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(boot_n, len(keys)))
    d = (A[idx].sum(1) - B[idx].sum(1)) / np.maximum(N[idx].sum(1), 1)
    return obs, *np.quantile(d, [.025, .975])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--arms", nargs="+", required=True, help="name=path.h5")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    root = os.path.expanduser(a.root)

    arms = {}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        arms[name] = load(path)
        print(f"  {name}: {len(arms[name][0]):,} samples x {arms[name][1].shape[1]}d")

    meta = {}
    with gzip.open(f"{root}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        ix = {c: h.index(c) for c in ["sample_id", "study_code", "domain"] + TARGETS}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(ix.values()):
                meta[f[ix["sample_id"]]] = {c: f[ix[c]] for c in ix}

    ids = sorted(set.intersection(*(set(i) for i, _ in arms.values())) & set(meta))
    print(f"\n{len(ids):,} samples shared by {len(arms)} arms, "
          f"{len({meta[s]['study_code'] for s in ids})} studies, k={a.k}\n")
    study = np.array([meta[s]["study_code"] for s in ids])
    domain = np.array([meta[s]["domain"] or "unknown" for s in ids])
    peers = Counter(study)
    ceiling = np.array([min(a.k, peers[s] - 1) / a.k for s in study])

    lab = {t: np.array([meta[s][t] or "" for s in ids]) for t in TARGETS}
    names = list(arms)
    per = {}
    for name in names:
        aid, X = arms[name]
        pos = {s: i for i, s in enumerate(aid)}
        Z = X[[pos[s] for s in ids]]
        nn = topk(Z, a.k)
        nn_b = topk(Z, a.k, blocked_by=study)
        m = {}
        m["study@k"] = (study[nn] == study[:, None]).mean(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m["study@k norm"] = np.where(ceiling > 0, m["study@k"] / ceiling, np.nan)
        for t in TARGETS:
            y = lab[t]
            has = y != ""
            def pur(idxs):
                v = np.full(len(ids), np.nan)
                ok = has & (y[idxs] != "").any(1)
                sel = np.flatnonzero(ok)
                match = (y[idxs[sel]] == y[sel][:, None])
                valid = (y[idxs[sel]] != "")
                v[sel] = np.where(valid.sum(1) > 0, match.sum(1) / np.maximum(valid.sum(1), 1), np.nan)
                return v
            m[f"purity@k {t}"] = pur(nn)
            m[f"blocked purity@k {t}"] = pur(nn_b)
        per[name] = m

    keys = list(per[names[0]])
    print("%-24s" % "" + "".join("%12s" % n for n in names))
    for kk in keys:
        print("%-24s" % kk + "".join("%12.3f" % np.nanmean(per[n][kk]) for n in names))

    if len(names) >= 2:
        x, z = names[0], names[1]
        print(f"\nPAIRED DIFFERENCE  {x} minus {z}  (bootstrap over studies, 95% CI)")
        for kk in keys:
            obs, lo, hi = boot(per[x][kk], per[z][kk], study, a.boot)
            flag = "" if lo <= 0 <= hi else "   <-- CI excludes zero"
            print("  %-24s %+7.3f   [%+.3f, %+.3f]%s" % (kk, obs, lo, hi, flag))

    print("\nBY DOMAIN")
    doms = [d for d, c in Counter(domain).most_common() if c >= 100]
    for kk in ["study@k norm", "blocked purity@k biome", "blocked purity@k material"]:
        print(" ", kk)
        print("    %-12s %6s " % ("domain", "n") + "".join("%12s" % n for n in names))
        for d in doms:
            msk = domain == d
            print("    %-12s %6d " % (d[:12], msk.sum())
                  + "".join("%12.3f" % np.nanmean(per[n][kk][msk]) for n in names))


if __name__ == "__main__":
    main()
