#!/usr/bin/env python3
"""
Test 2: does removing identity content from keywords trade cluster purity for
mapping accuracy?

study-leakage-in-cluster-evaluation.md showed keyword embeddings put 75% of a
sample's nearest neighbours in its own study. The hypothesis is that keywords mix
two things: CATEGORY (what kind of place - transfers across studies, useful for
ENVO) and IDENTITY (which project, site, protocol - does not transfer). If so,
stripping identity should RAISE mapping macro accuracy and LOWER cluster purity.
A double dissociation; a single metric moving either way proves nothing.

Defining identity without leaking: a token is identity-like if it is tied to few
studies. Token statistics are computed on HALF the studies (fit half) and every
evaluation runs on the other half (eval half), so the rule never sees the studies
it is judged on. Tokens unseen in the fit half are KEPT - that makes the strip
conservative, so a dissociation we do see is not manufactured.

Rejected rule, recorded so it is not retried: dropping low corpus-frequency
tokens. It keeps 'California', 'USA', 'Illumina' (common corpus-wide) and drops
'Chelodina longicollis', 'gravesoil', 'marasmus' - exactly backwards.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/keyword_identity_strip.py
"""
import gzip, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
OUT = f"{NEW}/embeddings/vs_clusters"
MIN_STUDIES_KEEP, K, SEED = 3, 10, 42
KEEP_UNSEEN = os.environ.get("KEEP_UNSEEN", "1") == "1"   # unseen in the fit half
WORD = re.compile(r"[A-Za-z][A-Za-z\-']*|\d[\w\-]*")


def metalog():
    rows = {}
    with gzip.open(f"{ROOT}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        ix = {c: h.index(c) for c in ("sample_id", "study_code", "biome", "feature", "material")}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(ix.values()):
                rows[f[ix["sample_id"]]] = {c: f[ix[c]] for c in ix}
    return rows


def keywords_for(wanted):
    out = {}
    for line in open(f"{NEW}/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid in wanted and raw.strip():
            out[sid] = clean_text(raw, True)
    return out


def load_join(tag):
    d = {}
    with open(f"{ROOT}/clusters/sample_to_cluster_{tag}.tsv") as fh:
        next(fh)
        for line in fh:
            s, _, c = line.rstrip("\n").partition("\t")
            d[s] = c
    return d


def purity(texts, cl, stu):
    """Cluster purity@K, study-blocked purity@K, and study@K, on TF-IDF cosine."""
    X = normalize(TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(texts))
    cl, stu, n = np.asarray(cl), np.asarray(stu), len(cl)
    plain, blocked, same = [], [], []
    for s in range(0, n, 256):
        blk = np.arange(s, min(s + 256, n))
        sims = (X[blk] @ X.T).toarray()
        sims[np.arange(len(blk)), blk] = -np.inf
        nb = np.argpartition(-sims, K, axis=1)[:, :K]
        plain.append((cl[nb] == cl[blk][:, None]).mean(axis=1))
        same.append((stu[nb] == stu[blk][:, None]).mean(axis=1))
        m = sims
        m[stu[None, :] == stu[blk][:, None]] = -np.inf
        nb2 = np.argpartition(-m, K, axis=1)[:, :K]
        blocked.append((cl[nb2] == cl[blk][:, None]).mean(axis=1))
    return (float(np.concatenate(plain).mean()), float(np.concatenate(blocked).mean()),
            float(np.concatenate(same).mean()))


def mapping(texts, labels, groups, folds=5):
    """Study-grouped CV: micro top-1 and macro top-1 (mean per-class recall)."""
    X = TfidfVectorizer(min_df=2, sublinear_tf=True).fit_transform(texts)
    y, g = np.asarray(labels), np.asarray(groups)
    keep = np.array([v not in ("", "NA", "nan", "None") for v in y])
    X, y, g = X[keep], y[keep], g[keep]
    counts = Counter(y)
    keep = np.array([counts[v] >= folds for v in y])
    X, y, g = X[keep], y[keep], g[keep]
    if len(set(g)) < folds or len(y) < 50:
        return None
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(folds).split(X, y, g):
        m = RidgeClassifier().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    micro = float((pred == y).mean())
    per = [float((pred[y == c] == c).mean()) for c in sorted(set(y))]
    return {"n": int(len(y)), "classes": len(per), "top1": micro, "macro": float(np.mean(per))}


def main():
    rng = np.random.default_rng(SEED)
    meta = metalog()
    all_studies = sorted({r["study_code"] for r in meta.values()})
    rng.shuffle(all_studies)
    fit_half = set(all_studies[: len(all_studies) // 2])
    eval_half = set(all_studies[len(all_studies) // 2:])
    print(f"{len(meta):,} labelled samples, {len(all_studies)} studies "
          f"-> {len(fit_half)} fit / {len(eval_half)} eval", flush=True)

    kw = keywords_for(set(meta))
    print(f"  keyword text for {len(kw):,}", flush=True)

    # --- the strip rule, learned on the fit half only -----------------------
    tok_studies = defaultdict(set)
    for s, t in kw.items():
        st = meta[s]["study_code"]
        if st in fit_half:
            for w in set(WORD.findall(t)):
                tok_studies[w].add(st)
    eval_vocab = set()
    for s2, t in kw.items():
        if meta[s2]["study_code"] in eval_half:
            eval_vocab |= set(WORD.findall(t))
    keep_tokens = {w for w, v in tok_studies.items() if len(v) >= MIN_STUDIES_KEEP}
    print(f"  vocabulary: {len(tok_studies):,} tokens in the fit half, "
          f"{len(eval_vocab):,} in the eval half, "
          f"{len(eval_vocab & set(tok_studies)):,} shared", flush=True)
    print(f"  tokens in >= {MIN_STUDIES_KEEP} fit-half studies (kept): {len(keep_tokens):,}"
          f" | unseen in the fit half: "
          f"{'KEPT (conservative)' if KEEP_UNSEEN else 'DROPPED (strict)'}", flush=True)

    def strip(t):
        return " ".join(w for w in WORD.findall(t)
                        if w in keep_tokens or (KEEP_UNSEEN and w not in tok_studies))

    stripped = {s: strip(t) for s, t in kw.items()}
    keptfrac = np.mean([len(WORD.findall(stripped[s])) / max(1, len(WORD.findall(kw[s])))
                        for s in kw])
    print(f"  tokens kept per sample: {keptfrac:.1%}", flush=True)
    for s in list(kw)[:3]:
        print(f"    orig    : {kw[s][:100]}")
        print(f"    stripped: {stripped[s][:100]}")

    # second, blunter treatment: drop every capitalised token. This definitely
    # removes place, project and instrument names ("Tara Oceans", "California",
    # "Illumina"); it also damages binomial taxonomy ("Homo sapiens" -> "sapiens"),
    # which is category information, so it is a lower bound, not a clean knife.
    nocaps = {s2: " ".join(w for w in WORD.findall(t) if not w[:1].isupper())
              for s2, t in kw.items()}
    ncfrac = np.mean([len(WORD.findall(nocaps[s2])) / max(1, len(WORD.findall(kw[s2])))
                      for s2 in kw])
    print(f"  no-caps variant keeps {ncfrac:.1%} of tokens per sample", flush=True)
    for s2 in list(kw)[:2]:
        print(f"    no-caps : {nocaps[s2][:100]}")

    variants = {"original": kw, "study_tied_stripped": stripped, "no_caps": nocaps}
    res = {"min_studies_keep": MIN_STUDIES_KEEP, "keep_unseen": KEEP_UNSEEN,
           "tokens_kept_rule": len(keep_tokens),
           "token_kept_fraction_stripped": float(keptfrac),
           "token_kept_fraction_nocaps": float(ncfrac)}

    # --- (a) mapping, on the eval half only --------------------------------
    ev = [s for s in kw if meta[s]["study_code"] in eval_half]
    print(f"\n  MAPPING  (study-grouped CV on {len(ev):,} eval-half samples)", flush=True)
    print(f"    {'slot':<9} {'variant':<10} {'n':>7} {'classes':>8} {'top1':>8} {'macro':>8}")
    for slot in ("biome", "feature", "material"):
        for name, txt in variants.items():
            r = mapping([txt[s] for s in ev], [meta[s][slot] for s in ev],
                        [meta[s]["study_code"] for s in ev])
            if r:
                res[f"map|{slot}|{name}"] = r
                print(f"    {slot:<9} {name:<10} {r['n']:>7,} {r['classes']:>8} "
                      f"{r['top1']:>8.4f} {r['macro']:>8.4f}", flush=True)

    # --- (b) clusters, study-balanced, eval half only ----------------------
    join = load_join("coarse")
    per = defaultdict(lambda: defaultdict(list))
    for s in kw:
        st = meta[s]["study_code"]
        if st in eval_half and s in join:
            per[join[s]][st].append(s)
    ids, cl, stu = [], [], []
    for c, by in sorted(per.items()):
        if len(by) < 3:
            continue
        chosen = [(s, st) for st, v in sorted(by.items()) for s in sorted(v)[:8]]
        if len(chosen) < 2 * K:
            continue
        for s, st in chosen[:40]:
            ids.append(s); cl.append(c); stu.append(st)
    print(f"\n  CLUSTERS  ({len(ids):,} samples, {len(set(cl))} clusters, "
          f"{len(set(stu))} studies)", flush=True)
    sizes = Counter(cl)
    base = sum(v * (v - 1) for v in sizes.values()) / (len(cl) * (len(cl) - 1))
    print(f"    random-pair baseline {base:.4f}")
    print(f"    {'variant':<10} {'purity@10':>11} {'blocked':>10} {'study@10':>10}")
    for name, txt in variants.items():
        p, b, ss = purity([txt[s] for s in ids], cl, stu)
        res[f"clu|{name}"] = {"purity": p, "blocked": b, "study10": ss, "baseline": base}
        print(f"    {name:<10} {p:>11.3f} {b:>10.3f} {ss:>10.3f}", flush=True)

    print(f"\n  === the dissociation ===", flush=True)
    for name in ("study_tied_stripped", "no_caps"):
        print(f"\n    -- {name} minus original --")
        for slot in ("biome", "feature", "material"):
            a, b = res.get(f"map|{slot}|original"), res.get(f"map|{slot}|{name}")
            if a and b:
                print(f"    mapping {slot:<9} top1 {b['top1'] - a['top1']:+.4f}   "
                      f"macro {b['macro'] - a['macro']:+.4f}")
        co, cs = res["clu|original"], res[f"clu|{name}"]
        print(f"    cluster purity@10        {cs['purity'] - co['purity']:+.4f}")
        print(f"    cluster study@10         {cs['study10'] - co['study10']:+.4f}")

    suffix = "" if KEEP_UNSEEN else "_strict"
    json.dump(res, open(f"{OUT}/keyword_identity_strip{suffix}.json", "w"), indent=2, default=str)
    print(f"\nwrote {OUT}/keyword_identity_strip{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
