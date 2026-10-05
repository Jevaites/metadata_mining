#!/usr/bin/env python3
"""
How deep can free-text metadata resolve? (ENVO depth, empirically)

Two clusterings give only two points, so instead of a flat fine-vs-coarse
comparison this is a NESTED test: given a sample is already in the right broad
group, can the metadata take you one level deeper?

    level A   whole corpus      classes = coarse clusters   (anywhere in the corpus)
    level B   inside one biome  classes = coarse clusters   (which habitat within soil?)
    level C   inside one coarse classes = fine clusters     (which community within it?)

A vs B isolates how much of the coarse-cluster signal is really just "soil vs gut";
B vs C is the depth question. Biomes are reported separately: there are only 5 of
them, too few for the C-class design, so that number has its own baseline.

Every level is measured at IDENTICAL difficulty - C classes x M samples per
class - so purity@10 is directly comparable and the random baseline is the same
number at all three. Without that, "more classes" and "harder" are confounded.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/cluster_granularity.py
"""
import json, os, sys
from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"
UNIQ = f"{NEW}/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5"

C_CLASSES = int(os.environ.get("C_CLASSES", 8))     # classes per context
M_PER = int(os.environ.get("M_PER", 25))           # samples per class
K = 10
DRAWS_A, DRAWS_B, CONTEXTS_C = 40, 15, 400
SLAB, SEED = 50_000, 42


def rss():
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB"


def load_join(tag):
    out = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            out[s] = c
    return out


def auc(same, diff):
    r = rankdata(np.concatenate([same, diff]))
    n1, n2 = len(same), len(diff)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def score(vec, lab, rng):
    """purity@k and separation AUC inside one context (200 samples)."""
    n = len(lab)
    sims = vec @ vec.T
    np.fill_diagonal(sims, -np.inf)
    nb = np.argpartition(-sims, K, axis=1)[:, :K]
    purity = float((lab[nb] == lab[:, None]).mean())
    iu, ju = np.triu_indices(n, 1)
    same_mask = lab[iu] == lab[ju]
    s = sims[iu, ju]
    if same_mask.sum() < 20 or (~same_mask).sum() < 20:
        return purity, None
    return purity, auc(s[same_mask], s[~same_mask])


def pick(rng, groups):
    """C classes with >= M members, M members each -> (ids, labels) or None."""
    big = sorted(g for g, v in groups.items() if len(v) >= M_PER)
    if len(big) < C_CLASSES:
        return None
    take = [big[i] for i in rng.choice(len(big), C_CLASSES, replace=False)]
    ids, lab = [], []
    for g in take:
        v = sorted(groups[g])
        ids += [v[i] for i in rng.choice(len(v), M_PER, replace=False)]
        lab += [g] * M_PER
    return ids, np.array(lab)


def main():
    rng = np.random.default_rng(SEED)
    fine, coarse = load_join("fine"), load_join("coarse")

    # one pass: members per (coarse, fine) pair, and each coarse cluster's biome
    members, biome_votes = defaultdict(list), defaultdict(Counter)
    for line in open(f"{NEW}/GPT_biomes.txt", encoding="utf-8", errors="replace"):
        sid, _, b = line.rstrip("\n").partition("\t")
        c, f = coarse.get(sid), fine.get(sid)
        if c is None or f is None or not b.strip():
            continue
        members[(c, f)].append(sid)
        biome_votes[c][b.strip()] += 1
    biome_of = {c: v.most_common(1)[0][0] for c, v in biome_votes.items()}
    print(f"samples in both clusterings with a biome: "
          f"{sum(len(v) for v in members.values()):,} [rss {rss()}]", flush=True)

    # --- is the hierarchy real? --------------------------------------------
    fine_spread, coarse_spread = defaultdict(Counter), defaultdict(Counter)
    for (c, f), v in members.items():
        fine_spread[f][c] += len(v)
        coarse_spread[c][biome_of[c]] += len(v)
    def purity_of(spread):
        vals = [max(v.values()) / sum(v.values()) for v in spread.values()]
        return np.mean(vals), np.mean([len(v) for v in spread.values()])
    fp, fn = purity_of(fine_spread)
    print(f"  nesting: a fine cluster sits {fp:.1%} in its main coarse cluster "
          f"(mean {fn:.2f} coarse clusters touched)", flush=True)
    cb = np.mean([max(v.values()) / sum(v.values()) for v in coarse_spread.values()])
    print(f"           a coarse cluster sits {cb:.1%} in its main biome", flush=True)

    # --- build the three levels' contexts ----------------------------------
    by_biome, by_biome_coarse, by_coarse_fine = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for (c, f), v in members.items():
        b = biome_of[c]
        by_biome[b] += v
        by_biome_coarse[b][c] += v
        by_coarse_fine[c][f] = v
    print(f"  {len(by_biome)} biomes, {len(by_biome_coarse)} biome contexts, "
          f"{len(by_coarse_fine):,} coarse contexts [rss {rss()}]", flush=True)

    by_coarse = defaultdict(list)
    for c, g in by_coarse_fine.items():
        for v in g.values():
            by_coarse[c] += v

    jobs = []                                   # (level, context_name, ids, labels)
    for _ in range(DRAWS_A):
        r = pick(rng, by_coarse)
        if r:
            jobs.append(("A  coarse cluster, whole corpus", "ALL", *r))
    for _ in range(DRAWS_A):                    # biomes: 5 classes, its own baseline
        big = sorted(b for b, v in by_biome.items() if len(v) >= M_PER)
        if len(big) < 2:
            break
        ids, lab = [], []
        for b in big:
            v = sorted(by_biome[b])
            ids += [v[i] for i in rng.choice(len(v), M_PER, replace=False)]
            lab += [b] * M_PER
        jobs.append((f"A0 biome ({len(big)} classes, own baseline)", "ALL", ids, np.array(lab)))
    for b, groups in sorted(by_biome_coarse.items()):
        for _ in range(DRAWS_B):
            r = pick(rng, groups)
            if r:
                jobs.append(("B  coarse cluster, within a biome", b, *r))
    cands = sorted(c for c, g in by_coarse_fine.items()
                   if sum(1 for v in g.values() if len(v) >= M_PER) >= C_CLASSES)
    if len(cands) > CONTEXTS_C:
        cands = [cands[i] for i in sorted(rng.choice(len(cands), CONTEXTS_C, replace=False))]
    for c in cands:
        r = pick(rng, by_coarse_fine[c])
        if r:
            jobs.append(("C  fine cluster, within a coarse cluster", c, *r))
    print(f"  contexts: " + ", ".join(f"{k} {v}" for k, v in
                                      Counter(j[0] for j in jobs).items()), flush=True)
    del members, by_biome, by_biome_coarse, by_coarse_fine, by_coarse

    # --- resolve texts -> rows, read sequentially --------------------------
    wanted = {s for j in jobs for s in j[2]}
    txt = {}
    for line in open(f"{NEW}/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid in wanted and raw.strip():
            txt[sid] = clean_text(raw, True)
    need = set(txt.values())
    row_of = {}
    with h5py.File(UNIQ, "r") as f:
        d = f["texts"]
        for i in range(0, d.shape[0], 200_000):
            for k, v in enumerate(d[i:i + 200_000]):
                v = v.decode()
                if v in need:
                    row_of[v] = i + k
    print(f"  {len(wanted):,} samples, {len(need):,} texts, {len(row_of):,} resolved "
          f"[rss {rss()}]", flush=True)

    ids = sorted({s for s in wanted if txt.get(s) in row_of})
    slot = {s: i for i, s in enumerate(ids)}
    rows = np.array([row_of[txt[s]] for s in ids])
    order = np.argsort(rows)
    with h5py.File(UNIQ, "r") as f:
        d = f["embeddings"]
        V = np.zeros((len(ids), d.shape[1]), dtype=np.float32)
        srt = rows[order]
        for start in range(0, d.shape[0], SLAB):
            lo, hi = np.searchsorted(srt, [start, start + SLAB])
            if hi == lo:
                continue
            slab = d[start:start + SLAB]
            V[order[lo:hi]] = slab[srt[lo:hi] - start]
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    print(f"  vectors loaded [rss {rss()}]", flush=True)

    # --- score every context ------------------------------------------------
    res = defaultdict(list)
    for level, ctx, jids, jlab in jobs:
        ok = [i for i, s in enumerate(jids) if s in slot]
        if len(ok) < len(jids) * 0.8:
            continue
        lab = jlab[ok]
        if len(set(lab)) < 2:
            continue
        p, a = score(V[[slot[jids[i]] for i in ok]], lab, rng)
        res[level].append((p, a, ctx))

    base = (M_PER - 1) / (C_CLASSES * M_PER - 1)
    print(f"\n  every level: {C_CLASSES} classes x {M_PER} samples, "
          f"random baseline purity@{K} = {base:.3f}\n")
    print(f"  {'level':<40} {'contexts':>8} {'purity@10':>18} {'AUC':>16}")
    summary = {"baseline": base, "classes": C_CLASSES, "per_class": M_PER, "levels": {}}
    for level in sorted(res):
        v = res[level]
        p = np.array([x[0] for x in v])
        a = np.array([x[1] for x in v if x[1] is not None])
        nc = len(set(np.concatenate([[x[2]] for x in v]))) if False else len(v)
        b = base if not level.startswith("A0") else (M_PER - 1) / (5 * M_PER - 1)
        print(f"  {level:<40} {len(v):>8} {p.mean():>10.3f} ± {p.std():.3f} "
              f"{a.mean():>10.3f} ± {a.std():.3f}   (baseline {b:.3f})")
        summary["levels"][level] = {"contexts": len(v), "baseline": float(b),
                                    "purity_mean": float(p.mean()),
                                    "purity_sd": float(p.std()), "auc_mean": float(a.mean()),
                                    "auc_sd": float(a.std()),
                                    "lift_over_baseline": float(p.mean() / base),
                                    "purity_per_context": [float(x) for x in p],
                                    "context": [x[2] for x in v]}
    summary["nesting"] = {"fine_in_coarse": float(fp), "coarse_in_biome": float(cb)}
    tagf = f"_c{C_CLASSES}m{M_PER}"
    json.dump(summary, open(f"{OUT}/cluster_granularity{tagf}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/cluster_granularity{tagf}.json", flush=True)


if __name__ == "__main__":
    main()
