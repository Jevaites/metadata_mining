#!/usr/bin/env python3
"""
Check the atlas predictions against the 1,091 hand-labelled gold samples (gold_dict.pkl),
which are mostly *not* in Metalog. No curator needed: the gold labels are a coarse biome
(animal / plant / soil / water / other) plus a free-text sub-biome, so every predicted
ENVO/Uberon term is first turned into coarse biomes automatically.

Term -> coarse biome, learned from the Metalog-linked training samples:
  for every labelled sample (any slot) and its GPT coarse biome, count that biome for the
  label and all its is_a ancestors. A term's coarse distribution is these counts, normalised.
  majority(term)   = the most frequent coarse biome
  compatible(term) = every coarse biome with a share >= --min_share (e.g. terrestrial biome
                     -> {soil, plant})

Scores per slot, for the top-1 answer, the back-off answer and (biome) the reranked answer:
  answered        share of samples that get an answer (back-off can abstain)
  coarse_strict   majority(answer) == gold coarse biome
  coarse_lenient  gold coarse biome in compatible(answer)
by confidence band, by gold class, and re-weighted to the atlas class mix. The same strict
metric on Metalog's own out-of-fold predictions (gold term mapped the same way) is the
reference: if the atlas is as reliable as Metalog CV says, the two should match per band.

python gold_check.py --dir ~/MicrobeAtlasProject/ontology_mapping/experiments/gold_check \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --training_set ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --cv_predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz
(inputs in --dir are extracted by gold_check_extract.sh)
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SLOTS, path, read_tsv  # noqa: E402

COARSE = ["animal", "plant", "soil", "water", "other"]
BANDS = [(0.9, 1.01), (0.75, 0.9), (0.5, 0.75), (0.0, 0.5)]


def ancestors_of(terms):
    parents = {t: [p for p in ps.split("||") if p] for t, ps in zip(terms["term_id"], terms["parents"])}
    memo = {}

    def up(t):
        if t not in memo:
            memo[t] = set()
            for p in parents.get(t, []):
                memo[t] |= {p} | up(p)
        return memo[t]
    return up


def coarse_map(train, gpt, up, min_share):
    counts = defaultdict(Counter)
    for row in train.itertuples():
        g = gpt.get(row.sample_id)
        if g is None:
            continue
        for slot in SLOTS:
            label = getattr(row, slot)
            if label:
                for t in {label} | up(label):
                    counts[t][g] += 1
    majority, compatible = {}, {}
    for t, c in counts.items():
        n = sum(c.values())
        majority[t] = c.most_common(1)[0][0]
        compatible[t] = {k for k, v in c.items() if v / n >= min_share}
    return majority, compatible, counts


def band_of(p):
    for lo, hi in BANDS:
        if lo <= p < hi:
            return f"{lo:.2f}-{min(hi, 1):.2f}"
    return "none"


def score(df, answer_col, p_col, gold_col, majority, compatible):
    """df rows: one sample. Returns summary dict + per-band + per-gold-class tables."""
    d = df.copy()
    d["answered"] = d[answer_col] != ""
    d["mapped"] = d[answer_col].map(lambda t: t in majority)
    d["strict"] = [a != "" and majority.get(a) == g for a, g in zip(d[answer_col], d[gold_col])]
    d["lenient"] = [a != "" and g in compatible.get(a, ()) for a, g in zip(d[answer_col], d[gold_col])]
    d["band"] = [band_of(float(p)) if a != "" and p != "" else "none" for a, p in zip(d[answer_col], d[p_col])]
    ans = d[d["answered"]]
    out = {"n": len(d), "answered": round(d["answered"].mean(), 3),
           "strict_among_answered": round(ans["strict"].mean(), 3) if len(ans) else None,
           "lenient_among_answered": round(ans["lenient"].mean(), 3) if len(ans) else None,
           "unmapped_answers": int((d["answered"] & ~d["mapped"]).sum())}
    by_band = ans.groupby("band").agg(n=("strict", "size"), strict=("strict", "mean"), lenient=("lenient", "mean")).round(3)
    by_class = d.groupby(gold_col).agg(n=("answered", "size"), answered=("answered", "mean"),
                                       strict=("strict", lambda s: s[d.loc[s.index, "answered"]].mean()),
                                       lenient=("lenient", lambda s: s[d.loc[s.index, "answered"]].mean())).round(3)
    return out, by_band, by_class, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ontology_terms", required=True)
    ap.add_argument("--training_set", required=True)
    ap.add_argument("--cv_predictions", required=True)
    ap.add_argument("--cv_method", default="prototype")
    ap.add_argument("--min_share", type=float, default=0.15)
    ap.add_argument("--atlas_mix", default="animal=0.620,water=0.130,soil=0.110,plant=0.094,other=0.045",
                    help="Coarse-biome mix of the atlas (GPT biomes), for re-weighting")
    args = ap.parse_args()
    D = path(args.dir)

    terms = read_tsv(args.ontology_terms)
    up = ancestors_of(terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    train = read_tsv(args.training_set)
    gpt = dict(pd.read_csv(f"{D}/gpt_biomes_gold_and_linked.tsv", sep="\t", header=None, dtype=str).values)
    majority, compatible, counts = coarse_map(train, gpt, up, args.min_share)
    gold = read_tsv(f"{D}/gold_labels.tsv")
    linked = set(train["sample_id"])
    report = {"gold_samples": len(gold), "gold_linked_to_metalog_excluded": int(gold["sample_id"].isin(linked).sum())}
    gold = gold[~gold["sample_id"].isin(linked)]

    atlas = read_tsv(f"{D}/atlas_backoff_gold.tsv").merge(gold, on="sample_id")
    final = read_tsv(f"{D}/atlas_final_gold.tsv")
    atlas = atlas.merge(final[["sample_id", "biome_final", "biome_final_p", "biome_final_source"]], on="sample_id", how="left").fillna("")
    atlas["gpt_biome"] = atlas["sample_id"].map(gpt).fillna("")
    report["gold_samples_with_prediction"] = len(atlas)
    g_ok = atlas[atlas["gpt_biome"] != ""]
    report["reference_gpt_biome_vs_gold"] = round(float((g_ok["gpt_biome"] == g_ok["gold_biome"]).mean()), 3)
    mix = {k: float(v) for k, v in (x.split("=") for x in args.atlas_mix.split(","))}

    tables, examples = {}, []
    for slot in SLOTS:
        variants = [("top1", slot, f"{slot}_p"), ("backoff", f"{slot}_backoff", f"{slot}_backoff_p")]
        if slot == "biome":
            variants.append(("rerank_final", "biome_final", "biome_final_p"))
        for name, col, pcol in variants:
            out, by_band, by_class, d = score(atlas, col, pcol, "gold_biome", majority, compatible)
            w = by_class.reindex(COARSE)
            out["strict_reweighted_to_atlas_mix"] = round(float(np.nansum([w.loc[c, "strict"] * mix[c] for c in COARSE])), 3)
            out["answered_reweighted_to_atlas_mix"] = round(float(np.nansum([w.loc[c, "answered"] * mix[c] for c in COARSE])), 3)
            report[f"{slot}/{name}"] = out
            tables[f"{slot}/{name}/by_band"] = by_band.reset_index().to_dict("records")
            tables[f"{slot}/{name}/by_gold_class"] = by_class.reset_index().to_dict("records")
            if name == "backoff":
                wrong = d[d["answered"] & ~d["lenient"]]
                for r in wrong.head(400).itertuples():
                    examples.append({"slot": slot, "sample_id": r.sample_id, "gold_biome": r.gold_biome,
                                     "gold_sub_biome": r.gold_sub_biome, "answer": label_of.get(getattr(r, col), ""),
                                     "answer_coarse": majority.get(getattr(r, col), "?"), "p": getattr(r, pcol)})

    # reference: Metalog out-of-fold, gold term mapped with the same majority map
    cv = read_tsv(args.cv_predictions)
    cv = cv[cv["method"] == args.cv_method].copy()
    cv["gold_coarse"] = cv["gold"].map(majority).fillna("")
    cv = cv[cv["gold_coarse"] != ""]
    for slot in SLOTS:
        c = cv[cv["slot"] == slot]
        for name, col, pcol in [("top1", "pred", "prob"), ("backoff", "backoff", "backoff_q")]:
            out, by_band, _, _ = score(c, col, pcol, "gold_coarse", majority, compatible)
            report[f"metalog_cv/{slot}/{name}"] = out
            tables[f"metalog_cv/{slot}/{name}/by_band"] = by_band.reset_index().to_dict("records")

    # answered share: gold vs whole atlas is in the atlas summary; here gold vs Metalog CV
    json.dump({"summary": report, "tables": tables}, open(f"{D}/gold_check.json", "w"), indent=1)
    pd.DataFrame(examples).to_csv(f"{D}/gold_check_disagreements.tsv", sep="\t", index=False)
    mp = pd.DataFrame([{"term_id": t, "label": label_of.get(t, ""), "majority": majority[t],
                        "compatible": "|".join(sorted(compatible[t])), "n": sum(counts[t].values()),
                        **{c: round(counts[t][c] / sum(counts[t].values()), 3) for c in COARSE}} for t in majority])
    mp.sort_values("n", ascending=False).to_csv(f"{D}/term_to_coarse_biome.tsv", sep="\t", index=False)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
