#!/usr/bin/env python3
"""
Corpus-wide document frequency of every keyword token.

The signal we want to strip is IDENTITY: site names, project codes, protocols,
instrument names. Those are rare by nature - a site name appears in one study and
nowhere else - while category words ("soil", "rhizosphere", "feces") recur across
thousands of unrelated samples. Document frequency separates them without using
any label or study id, so nothing leaks into the later evaluation.

One pass over all 3.4M keyword strings. Writes clusters/../keyword_token_df.tsv
(token<TAB>df) for tokens with df >= MIN_DF; anything absent is rarer than that.

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/keyword_token_df.py
"""
import os, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
NEW = f"{ROOT}/sidequest/latest"
MIN_DF = 3
WORD = re.compile(r"[A-Za-z][A-Za-z\-']*|\d[\w\-]*")


def tokens(text):
    return WORD.findall(text)


def main():
    df, n = Counter(), 0
    for line in open(f"{NEW}/GPT_keywords.txt", encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if not sid or not raw.strip():
            continue
        n += 1
        df.update(set(tokens(clean_text(raw, True))))          # set: document frequency
    print(f"{n:,} samples, {len(df):,} distinct tokens", flush=True)

    keep = {t: c for t, c in df.items() if c >= MIN_DF}
    out = f"{NEW}/keyword_token_df.tsv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("token\tdf\n")
        for t, c in sorted(keep.items(), key=lambda kv: -kv[1]):
            fh.write(f"{t}\t{c}\n")
    print(f"  kept {len(keep):,} tokens with df >= {MIN_DF} -> {out}", flush=True)

    v = sorted(df.values(), reverse=True)
    print(f"  df quantiles over all tokens: "
          + "  ".join(f"p{p} {v[int(len(v) * (1 - p / 100))]:,}" for p in (50, 75, 90, 99)))
    for thr in (10, 100, 1000, 10_000):
        toks = sum(1 for c in df.values() if c >= thr)
        mass = sum(c for c in df.values() if c >= thr) / sum(df.values())
        print(f"  df >= {thr:>6,}: {toks:>8,} tokens ({toks/len(df):5.2%} of vocabulary), "
              f"{mass:6.2%} of all token occurrences")


if __name__ == "__main__":
    main()
