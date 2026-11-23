#!/usr/bin/env python3
"""
Set answers on the hand-labelled gold samples (§8's 1,021 samples outside Metalog, coarse labels animal / plant /
soil / water / other): do the several-answers columns of 6_predict_atlas.py (<slot>_answer + <slot>_alternatives,
hierarchy.SetDecoder; README §21) hold outside Metalog, and do they beat the back-off on the same samples?

Every term is mapped to coarse biomes as in gold_check.py (gold_check.coarse_map: the GPT coarse biomes of the Metalog
samples carrying the term or a term below it): majority(term) and compatible(term) (share >= --min_share). For an
answer of one or more terms, among answered samples:
  lenient_any     one of its terms is compatible with the gold coarse biome (a reviewer picks the right term)
  lenient_split   mean over its terms of 'compatible' (a random pick)
  strict_any / strict_split   the same with majority(term) == gold
  decisive        sets only: all terms have the same majority coarse biome (the set still pins the coarse biome)
Both the set answers and the back-off (<slot>_backoff of the same file) are scored on the same samples, overall, on the
back-off's abstentions, and per answer kind; 'reweighted' re-weights the per-class shares to the atlas class mix
(the gold set is balanced: ~210 samples per class). The ambiguous_branches flag: how often the top-1, the runner-up or
either is compatible with gold when flagged.

python experiments/gold_check_sets.py --dir ~/MicrobeAtlasProject/ontology_mapping/experiments/gold_check \
  --atlas_gold ~/MicrobeAtlasProject/ontology_mapping/experiments/gold_check/atlas_sets_gold.tsv \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --training_set ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz
(atlas_sets_gold.tsv = the gold samples' rows of atlas_sets_g/atlas_predictions.tsv.gz: gold_check_extract.sh)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SLOTS, path, read_tsv, term_ancestors  # noqa: E402
from gold_check import COARSE, coarse_map  # noqa: E402


def answer_terms(primary, alternatives):
    """[primary, alternative terms...] from the <slot>_answer and <slot>_alternatives ('id:p;id:p') columns."""
    if not primary:
        return []
    return [primary] + [a.rsplit(":", 1)[0] for a in alternatives.split(";") if a]


def term_scores(terms, gold, majority, compatible):
    """(lenient_any, lenient_split, strict_any, strict_split, decisive) of one answered sample. decisive: every term has
    the same majority coarse biome (None for a single term)."""
    lenient = [gold in compatible.get(t, ()) for t in terms]
    strict = [majority.get(t) == gold for t in terms]
    decisive = None if len(terms) < 2 else len({majority.get(t) for t in terms}) == 1
    return any(lenient), float(np.mean(lenient)), any(strict), float(np.mean(strict)), decisive


def score_slot(df, slot, majority, compatible):
    """Per-sample table of one slot: answered, kind and the term scores of the set answer and of the back-off."""
    rows = []
    for r in df.itertuples():
        terms = answer_terms(getattr(r, f"{slot}_answer"), getattr(r, f"{slot}_alternatives"))
        backoff = getattr(r, f"{slot}_backoff")
        s = term_scores(terms, r.gold_biome, majority, compatible) if terms else (False, 0.0, False, 0.0, None)
        b = term_scores([backoff], r.gold_biome, majority, compatible) if backoff else (False, 0.0, False, 0.0, None)
        rows.append({"gold": r.gold_biome, "kind": getattr(r, f"{slot}_answer_kind"), "answered": bool(terms),
                     "lenient_any": s[0], "lenient_split": s[1], "strict_any": s[2], "strict_split": s[3],
                     "decisive": s[4], "backoff_answered": bool(backoff), "backoff_lenient": b[0], "backoff_strict": b[2],
                     "flag": getattr(r, f"{slot}_ambiguous_branches") == "True",
                     "top1_lenient": r.gold_biome in compatible.get(getattr(r, slot), ()),
                     "runner_lenient": r.gold_biome in compatible.get(getattr(r, f"{slot}_runner_up"), ())})
    return pd.DataFrame(rows)


def summarise(t, mix):
    """Summary dict of one slot's per-sample table (see the module docstring)."""
    def among(mask, col):
        return round(float(t.loc[mask, col].mean()), 3) if mask.any() else None

    ans, bo = t["answered"], t["backoff_answered"]
    out = {"n": len(t), "answered": round(float(ans.mean()), 3), "backoff_answered": round(float(bo.mean()), 3),
           "single": round(float((t["kind"] == "single").mean()), 3), "set": round(float((t["kind"] == "set").mean()), 3)}
    for col in ("lenient_any", "lenient_split", "strict_any", "strict_split"):
        out[col] = among(ans, col)
    out["backoff_lenient"], out["backoff_strict"] = among(bo, "backoff_lenient"), among(bo, "backoff_strict")
    out["single_lenient"] = among(t["kind"] == "single", "lenient_any")
    sets = t["kind"] == "set"
    out["set_lenient_any"], out["set_lenient_split"] = among(sets, "lenient_any"), among(sets, "lenient_split")
    out["set_decisive"] = round(float(t.loc[sets, "decisive"].astype(bool).mean()), 3) if sets.any() else None
    abst = ~bo
    out["on_backoff_abstentions"] = {"n": int(abst.sum()), "answered": among(abst, "answered"),
                                     "lenient_any": among(abst & ans, "lenient_any"),
                                     "lenient_split": among(abst & ans, "lenient_split")}
    fl = t["flag"]
    out["flagged"] = {"n": int(fl.sum()), "top1_lenient": among(fl, "top1_lenient"),
                      "runner_up_lenient": among(fl, "runner_lenient"),
                      "either": round(float((t.loc[fl, "top1_lenient"] | t.loc[fl, "runner_lenient"]).mean()), 3)
                      if fl.any() else None}
    by = t.groupby("gold")
    cls = pd.DataFrame({"answered": by["answered"].mean(), "backoff_answered": by["backoff_answered"].mean(),
                        "lenient_any": t[ans].groupby("gold")["lenient_any"].mean(),
                        "backoff_lenient": t[bo].groupby("gold")["backoff_lenient"].mean()})
    w = cls.reindex(COARSE)
    out["reweighted"] = {c: round(float(np.nansum([w.loc[k, c] * mix[k] for k in COARSE])), 3) for c in cls.columns}
    out["by_gold_class"] = cls.round(3).reset_index().to_dict("records")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="gold_check folder (gold_labels.tsv, gpt_biomes_gold_and_linked.tsv)")
    ap.add_argument("--atlas_gold", required=True, help="the gold samples' rows of an atlas_predictions.tsv.gz with set answers")
    ap.add_argument("--ontology_terms", required=True)
    ap.add_argument("--training_set", required=True)
    ap.add_argument("--min_share", type=float, default=0.15, help="As gold_check.py")
    ap.add_argument("--atlas_mix", default="animal=0.620,water=0.130,soil=0.110,plant=0.094,other=0.045")
    ap.add_argument("--output", default=None, help="JSON report (default: <dir>/gold_check_sets.json)")
    args = ap.parse_args(argv)
    D = path(args.dir)
    terms = read_tsv(args.ontology_terms)
    train = read_tsv(args.training_set)
    gpt = dict(pd.read_csv(f"{D}/gpt_biomes_gold_and_linked.tsv", sep="\t", header=None, dtype=str).values)
    majority, compatible, _ = coarse_map(train, gpt, term_ancestors(terms), args.min_share)
    gold = read_tsv(f"{D}/gold_labels.tsv")
    gold = gold[~gold["sample_id"].isin(set(train["sample_id"]))]
    atlas = read_tsv(path(args.atlas_gold)).merge(gold[["sample_id", "gold_biome"]], on="sample_id")
    if "biome_answer" not in atlas.columns:
        raise SystemExit(f"{args.atlas_gold} has no set-answer columns: extract it from an atlas run with set answers")
    mix = {k: float(v) for k, v in (x.split("=") for x in args.atlas_mix.split(","))}
    report = {"gold_samples": len(atlas)}
    for slot in SLOTS:
        report[slot] = summarise(score_slot(atlas, slot, majority, compatible), mix)
    out = path(args.output) if args.output else f"{D}/gold_check_sets.json"
    json.dump(report, open(out, "w"), indent=1)
    for slot in SLOTS:
        r = report[slot]
        print(f"{slot:8} answered {r['answered']:.3f} (back-off {r['backoff_answered']:.3f}) | lenient any "
              f"{r['lenient_any']} split {r['lenient_split']} (back-off {r['backoff_lenient']}) | sets {r['set']:.3f}, "
              f"lenient any {r['set_lenient_any']}, decisive {r['set_decisive']} | reweighted {r['reweighted']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
