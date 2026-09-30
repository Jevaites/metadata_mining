#!/usr/bin/env python3
"""
Does the v3 MUST INCLUDE list actually make the model report stated attributes?

Compares keyword arms on the same samples with value-level matching: for each
attribute the record states, we take the VALUE from the metadata and ask whether
that value appears in the keyword string. This avoids grading the model against
a hand-written list of expected words.

  recall        of records that state the attribute, share whose keywords carry it
  invention     share of records whose keywords assert the attribute when the
                record never mentions it anywhere
  paired CI     bootstrap over studies, so a few large studies cannot carry a result

    python3 scripts/compare_include_list.py --arms mini_v3 mini_v3_nolist
"""
import argparse, gzip, os, pickle, random, re, sys
from collections import defaultdict

MISSING = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "unspecified",
           "not applicable", "not collected", "not provided", "not specified", "restricted access"}

def kws(v):
    return [w.strip().lower() for w in v.strip().strip("{}").split(",") if w.strip()]

def fields(text):
    """cleaned record -> list of (normalised key, raw value)"""
    out = []
    for line in text.split("\n"):
        k, has, v = line.partition("=")
        if not has:
            continue
        v = v.strip()
        if v.lower() in MISSING:
            continue
        k = k.strip().lower()
        k = re.sub(r"^sample[_ ]", "", k)
        out.append((k, v))
    return out

SYN = {}
for _g in [("feces", "faeces", "fecal", "faecal", "stool", "stools", "excrement", "dung"),
           ("intestine", "intestinal", "intestines", "bowel", "gut", "caecum", "cecum"),
           ("rhizosphere", "rhizospheric"), ("skin", "cutaneous", "dermal", "epidermis"),
           ("oral", "mouth", "buccal", "saliva", "salivary"), ("vagina", "vaginal"),
           ("nasal", "nose", "nasopharyngeal", "nasopharynx"), ("rumen", "ruminal"),
           ("leaf", "leaves", "phyllosphere", "foliar"), ("root", "roots"),
           ("lung", "pulmonary", "respiratory", "airway"), ("milk", "breastmilk"),
           ("blood", "serum", "plasma"), ("urine", "urinary"),
           ("sediment", "sediments"), ("soil", "soils"), ("water", "waters")]:
    for _w in _g:
        SYN[_w] = _g[0]

def canon(w):
    """fold spelling and morphological variants: faeces/stool/fecal -> feces"""
    w = w.lower()
    return SYN.get(w) or SYN.get(w.rstrip("s")) or w.rstrip("s")

def toks(text):
    return {canon(w) for w in re.split(r"[^a-z0-9.+-]+", text.lower()) if len(w) > 2}

def words(v):
    return [canon(w) for w in re.split(r"[^a-z0-9.+-]+", v.lower()) if len(w) > 3]

def nums(v):
    return re.findall(r"\d+(?:\.\d+)?", v)

def num_in(n, text):
    """9 matches '9 m depth'; 3.48 matches '3.48 c'; avoids matching inside 1993"""
    if "." in n:
        n = n.rstrip("0").rstrip(".") or "0"
        return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", text)
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d.])", text)

KEY = {
 "host sex":     lambda k: re.search(r"(^|[_ ])(sex|gender)([_ ]|$)", k),
 "host age":     lambda k: re.search(r"(^|[_ ])age([_ ]|$)", k) or "life stage" in k or "dev_stage" in k,
 "disease":      lambda k: re.search(r"disease|health|diagnos|phenotype|clinical", k),
 "depth":        lambda k: re.search(r"(^|[_ ])depth", k),
 "temperature":  lambda k: re.search(r"(^|[_ ])temp", k),
 "pH":           lambda k: re.fullmatch(r"ph|ph_value|soil[_ ]ph", k),
 "treatment":    lambda k: re.search(r"treatment|diet|amendment|exposure|antibiot", k),
 "host species": lambda k: k in ("host", "host_scientific_name", "host scientific name",
                                 "host_taxid", "host_common_name"),
 "body site":    lambda k: re.search(r"body[_ ](site|product)|organism part|tissue", k),
 # isolation_source is scored separately: many studies use it for a PLACE
 # ("Yellowstone National Park", "Magee Womens Hospital"), not a material, so a
 # prompt that correctly drops place names scores low here by construction.
 "isolation src": lambda k: re.search(r"isolation[_ ]source|env[_ ]medium", k),
}
NUMERIC = {"host age", "depth", "temperature", "pH"}
# a bare mention anywhere in the record, used to separate invention from recall
MENTION = {"host sex": re.compile(r"\b(male|female|man|woman)\b", re.I),
           "depth":    re.compile(r"depth", re.I),
           "temperature": re.compile(r"temp", re.I)}
ASSERT = {"host sex": re.compile(r"\b(male|female)\b", re.I),
          "depth":    re.compile(r"\bdepth\b|\b\d+(\.\d+)?\s*(m|cm|metre|meter)\b", re.I),
          "temperature": re.compile(r"\b\d+(\.\d+)?\s*(°|deg|c\b|celsius)", re.I)}

def carries(attr, value, kwtext):
    if attr in NUMERIC:
        ns = nums(value)
        return bool(ns) and any(num_in(n, kwtext) for n in ns)
    ws = words(value)
    if attr == "host species":                      # genus is enough
        ws = [w for w in ws if not w.isdigit()][:1] or ws
    return any(w in toks(kwtext) for w in ws)

def boot_diff(hx, hy, groups, keys, boot, seed=0):
    """Paired difference in rates, bootstrapping over groups. Returns (obs, lo, hi)."""
    import numpy as np
    x = np.array([sum(hx[s] for s in groups[k]) for k in keys], float)
    y = np.array([sum(hy[s] for s in groups[k]) for k in keys], float)
    n = np.array([len(groups[k]) for k in keys], float)
    obs = (x.sum() - y.sum()) / n.sum()
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(boot, len(keys)))
    d = (x[idx].sum(1) - y[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.quantile(d, [.025, .975])
    return obs, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/MicrobeAtlasProject")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--baseline", default="GPT_keywords.txt")
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    root = os.path.expanduser(a.root)
    lat = f"{root}/sidequest/latest"

    meta = pickle.load(open(f"{lat}/.metadata_devset.pkl", "rb"))
    arms = {}
    for tag in a.arms:
        d = {}
        for line in open(f"{lat}/GPT_keywords_{tag}.txt", encoding="utf-8", errors="replace"):
            s, _, v = line.rstrip("\n").partition("\t")
            if v.strip():
                d[s] = v
        arms[tag] = d
    ids = sorted(set.intersection(*(set(d) for d in arms.values())) & set(meta))
    base, idset = {}, set(ids)
    for line in open(f"{lat}/{a.baseline}", encoding="utf-8", errors="replace"):
        s, _, v = line.rstrip("\n").partition("\t")
        if s in idset and v.strip():
            base[s] = v
        if len(base) == len(idset):
            break
    arms = {"old": base, **arms}
    print(f"{len(ids):,} samples shared by {len(a.arms)} arms\n")

    study = {}
    with gzip.open(f"{root}/metalog/metalog_training_set.tsv.gz", "rt") as fh:
        h = next(fh).rstrip("\n").split("\t")
        i_s, i_st = h.index("sample_id"), h.index("study_code")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > max(i_s, i_st):
                study[f[i_s]] = f[i_st]

    names = list(arms)
    text = {t: {s: " ".join(kws(arms[t].get(s, ""))) for s in ids} for t in names}
    FLD = {s: fields(meta[s]) for s in ids}          # parse each record once

    # ---- recall, conditional on the record stating the attribute
    print("RECALL - of records stating the attribute, share whose keywords carry its value")
    print("%-14s %6s " % ("attribute", "n") + "".join("%16s" % t for t in names))
    rows = {}
    for attr, test in KEY.items():
        sub, vals = [], {}
        for s in ids:
            v = [val for k, val in FLD[s] if test(k)]
            if v:
                sub.append(s); vals[s] = max(v, key=len)
        if len(sub) < 15:
            print("%-14s %6d   (too few to report)" % (attr, len(sub))); continue
        hit = {t: {s: carries(attr, vals[s], text[t][s]) for s in sub} for t in names}
        rows[attr] = (sub, hit)
        print("%-14s %6d " % (attr, len(sub))
              + "".join("%15.1f%%" % (100 * sum(hit[t].values()) / len(sub)) for t in names))

    # ---- paired difference between the two v3 arms, bootstrapped over studies
    if len(a.arms) == 2:
        x, y = a.arms
        print(f"\nPAIRED DIFFERENCE  {x} minus {y}   (bootstrap over studies, "
              f"{a.boot} resamples, 95% CI)")
        for attr, (sub, hit) in rows.items():
            by = defaultdict(list)
            for s in sub:
                by[study.get(s, s)].append(s)
            keys = list(by)
            obs, lo, hi = boot_diff(hit[x], hit[y], by, keys, a.boot)
            flag = "" if lo <= 0 <= hi else "   <-- CI excludes zero"
            print("  %-14s %+6.1f pp   [%+.1f, %+.1f]%s"
                  % (attr, 100 * obs, 100 * lo, 100 * hi, flag))

    # ---- exclusion violations: things the prompt forbids by name
    EXCL = {"generic term": re.compile(r"\b(metagenom\w*|microbiom\w*|microbiota|"
                                      r"microbial communit\w*|environmental sample|bacteria|"
                                      r"sample|data)\b", re.I),
            "code / date":  re.compile(r"\b(replicate[\s\-_]*\d+|subject\s+\w+|patient\s+\w+|"
                                      r"\d{4}-\d{2}|\d{1,2}[a-z]{3}\d{4}|(19|20)\d{2})\b", re.I),
            "method term":  re.compile(r"\b(illumina|miseq|hiseq|novaseq|16s|shotgun|amplicon|"
                                      r"rnalater|primer|kit|sequencing)\b", re.I)}
    print("\nEXCLUSION VIOLATIONS - keywords contain what the prompt forbids by name")
    print("%-14s %6s " % ("violation", "n") + "".join("%16s" % t for t in names))
    excl = {}
    for label, pat in EXCL.items():
        hit = {t: {s: bool(pat.search(text[t][s])) for s in ids} for t in names}
        excl[label] = hit
        print("%-14s %6d " % (label, len(ids))
              + "".join("%15.1f%%" % (100 * sum(hit[t].values()) / len(ids)) for t in names))
    if len(a.arms) == 2:
        x, y = a.arms
        by = defaultdict(list)
        for s in ids:
            by[study.get(s, s)].append(s)
        keys = list(by)
        print(f"\nPAIRED DIFFERENCE  {x} minus {y}   (bootstrap over studies, 95% CI)")
        for label, hit in excl.items():
            obs, lo, hi = boot_diff(hit[x], hit[y], by, keys, a.boot)
            flag = "" if lo <= 0 <= hi else "   <-- CI excludes zero"
            print("  %-14s %+6.1f pp   [%+.1f, %+.1f]%s"
                  % (label, 100 * obs, 100 * lo, 100 * hi, flag))

    # ---- invention: asserted although the record never mentions it
    print("\nINVENTION - keywords assert the attribute, record never mentions it")
    print("%-14s %6s " % ("attribute", "n") + "".join("%16s" % t for t in names))
    for attr, pat in ASSERT.items():
        silent = [s for s in ids
                  if not MENTION[attr].search(meta[s]) and not any(
                      KEY[attr](k) for k, _ in FLD[s])]
        if len(silent) < 15:
            continue
        print("%-14s %6d " % (attr, len(silent))
              + "".join("%15.1f%%" % (100 * sum(bool(pat.search(text[t][s])) for s in silent)
                                     / len(silent)) for t in names))

    # ---- shape
    print("\nSHAPE")
    print("%-22s " % "" + "".join("%16s" % t for t in names))
    for label, fn in [("keywords / sample", lambda t: sum(len(kws(arms[t].get(s, ""))) for s in ids) / len(ids)),
                      ("distinct strings", lambda t: 100 * len({text[t][s] for s in ids}) / len(ids)),
                      ("repeated stem in str", lambda t: 100 * sum(
                          1 for s in ids if (w := [x for k in kws(arms[t].get(s, "")) for x in k.split()])
                          and len(set(w)) < len(w) - 1) / len(ids))]:
        print("%-22s " % label + "".join("%15.1f%s" % (fn(t), "%" if "sample" not in label else " ")
                                         for t in names))

if __name__ == "__main__":
    main()
