#!/usr/bin/env python3
"""
Did the identity-stripped keywords stop fingerprinting studies, in EMBEDDING space?

study-leakage-in-cluster-evaluation.md showed old keyword embeddings put 75% of a
sample's ten nearest neighbours in its own study. The v2 prompt removed identity
from the text (country 28.6% -> 0.0%). This checks the vectors.

Same 2,000 dev samples, same model and dimension (3-large @ 1024) for both arms,
so the only difference is the text.

CEILING WARNING: the dev set was drawn at most 8 samples per study, ~3.7 on
average, so a query often has fewer than 10 same-study peers and study@10 cannot
reach the 0.84 measured earlier on a differently-built set. Absolute numbers here
are NOT comparable to that figure. The paired old-vs-new difference is, since both
arms face the identical ceiling, and a normalised version is reported too.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/v2_study_blocked.py
"""
import gzip, json, os, sys
from collections import Counter, defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
L = f"{ROOT}/sidequest/latest"
OLD_UNIQ = f"{L}/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5"
NEW_PS = (f"{L}/embeddings/v2_dev/GPT_keywords_embeddings__"
          f"text-embedding-3-large__dim1024__ids-dev2000_sample_ids.h5")
K = 10


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


def clusters():
    d = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_coarse.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            d[s] = c
    return d


def text_of(path, wanted):
    d = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        s, _, v = line.rstrip("\n").partition("\t")
        if s in wanted and v.strip():
            d[s] = clean_text(v, True)
    return d


def old_vectors(ids, txt):
    need = {txt[s] for s in ids}
    row = {}
    with h5py.File(OLD_UNIQ, "r") as f:
        d = f["texts"]
        for i in range(0, d.shape[0], 200_000):
            for k, v in enumerate(d[i:i + 200_000]):
                v = v.decode()
                if v in need:
                    row[v] = i + k
    keep = [s for s in ids if txt[s] in row]
    rows = np.array([row[txt[s]] for s in keep])
    order = np.argsort(rows); srt = rows[order]
    with h5py.File(OLD_UNIQ, "r") as f:
        d = f["embeddings"]
        V = np.zeros((len(keep), d.shape[1]), np.float32)
        for start in range(0, d.shape[0], 50_000):
            lo, hi = np.searchsorted(srt, [start, start + 50_000])
            if hi == lo:
                continue
            slab = d[start:start + 50_000]
            V[order[lo:hi]] = slab[srt[lo:hi] - start]
    return keep, V


def new_vectors(ids):
    with h5py.File(NEW_PS, "r") as f:
        sid = [x.decode() for x in f["sample_ids"][:]]
        E = f["embeddings"][:]
    pos = {s: i for i, s in enumerate(sid)}
    keep = [s for s in ids if s in pos]
    return keep, E[[pos[s] for s in keep]].astype(np.float32)


def metrics(V, cl, stu):
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    n = len(cl)
    out = {k: [] for k in ("study", "study_norm", "purity", "blocked", "usable")}
    for s in range(0, n, 256):
        blk = np.arange(s, min(s + 256, n))
        sims = V[blk] @ V.T
        sims[np.arange(len(blk)), blk] = -np.inf
        nb = np.argpartition(-sims, K, axis=1)[:, :K]
        same_stu = (stu[nb] == stu[blk][:, None]).mean(axis=1)
        avail = (stu[None, :] == stu[blk][:, None]).sum(axis=1) - 1
        out["study"].append(same_stu)
        out["study_norm"].append(same_stu / np.maximum(np.minimum(avail, K) / K, 1e-9))
        out["purity"].append((cl[nb] == cl[blk][:, None]).mean(axis=1))
        m = sims
        m[stu[None, :] == stu[blk][:, None]] = -np.inf
        nb2 = np.argpartition(-m, K, axis=1)[:, :K]
        out["blocked"].append((cl[nb2] == cl[blk][:, None]).mean(axis=1))
        out["usable"].append(((cl[None, :] == cl[blk][:, None]) &
                              (stu[None, :] != stu[blk][:, None])).sum(axis=1) >= K)
    return {k: np.concatenate(v) for k, v in out.items()}


def boot(diff, groups, n=600, seed=42):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for d, g in zip(diff, groups):
        by[g].append(d)
    keys = list(by)
    out = []
    for _ in range(n):
        flat = [x for j in rng.integers(0, len(keys), len(keys)) for x in by[keys[j]]]
        if flat:
            out.append(np.mean(flat))
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out)) - 1]


def main():
    st, cl_all = studies(), clusters()
    ids0 = [l.strip() for l in open(f"{L}/dev2000_sample_ids.txt") if l.strip()]
    ids0 = [s for s in ids0 if s in st and s in cl_all]
    old_txt = text_of(f"{L}/GPT_keywords.txt", set(ids0))
    ko, Vo = old_vectors([s for s in ids0 if s in old_txt], old_txt)
    kn, Vn = new_vectors(ids0)
    common = sorted(set(ko) & set(kn))
    po, pn = {s: i for i, s in enumerate(ko)}, {s: i for i, s in enumerate(kn)}
    Vo, Vn = Vo[[po[s] for s in common]], Vn[[pn[s] for s in common]]
    cl = np.array([cl_all[s] for s in common]); stu = np.array([st[s] for s in common])
    n = len(common)
    sizes = Counter(cl)
    base = sum(v * (v - 1) for v in sizes.values()) / (n * (n - 1))
    av = np.array([np.sum(stu == x) - 1 for x in stu])
    print(f"{n:,} samples with a study, a coarse cluster and both vectors")
    print(f"  {len(set(stu))} studies, {len(set(cl))} clusters, cluster baseline {base:.4f}")
    print(f"  same-study peers available: median {int(np.median(av))}, "
          f"max {av.max()} -> study@10 is capped near {np.mean(np.minimum(av, K) / K):.2f}\n")

    R = {"old": metrics(Vo, cl, stu), "new": metrics(Vn, cl, stu)}
    use = R["old"]["usable"] & R["new"]["usable"]
    print(f"  {'':<6} {'study@10':>10} {'study@10 norm':>15} {'purity@10':>11} {'blocked':>9}")
    for t in ("old", "new"):
        r = R[t]
        print(f"  {t:<6} {r['study'].mean():>10.3f} {r['study_norm'].mean():>15.3f} "
              f"{r['purity'].mean():>11.3f} {r['blocked'][use].mean():>9.3f}")
    print(f"\n  paired differences (new - old), bootstrap over studies, n_blocked={int(use.sum()):,}")
    out = {"n": n, "baseline": base}
    for name, key, mask in [("study@10", "study", None), ("study@10 normalised", "study_norm", None),
                            ("cluster purity@10", "purity", None),
                            ("purity@10 study-blocked", "blocked", use)]:
        m = np.ones(n, bool) if mask is None else mask
        d = (R["new"][key] - R["old"][key])[m]
        lo, hi = boot(d, stu[m])
        sig = "  *" if (lo > 0 or hi < 0) else ""
        print(f"    {name:<26} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
        out[name] = {"diff": float(d.mean()), "ci": [float(lo), float(hi)],
                     "old": float(R["old"][key][m].mean()), "new": float(R["new"][key][m].mean())}
    json.dump(out, open(f"{L}/embeddings/vs_clusters/v2_study_blocked.json", "w"), indent=2)
    print(f"\nwrote {L}/embeddings/vs_clusters/v2_study_blocked.json")


if __name__ == "__main__":
    main()
