#!/usr/bin/env python3
"""
Are the base model's "too specific" answers wrong, or more precise than the curator? (deck slide
"scoring rules"; report §8.1)

sample     draw held-out samples whose prototype top-1 is a descendant of the curator's term (outcome
           TOO_SPECIFIC in the cv_backoff predictions): 40 per slot, at most 2 per study, with the
           metadata text, for a manual review.
summarise  count the judgements of a reviewed file (column `judgement`):
           stated    = the metadata states the more specific term
           plausible = likely from the metadata, not stated
           arguable  = defensible but uncertain
           wrong     = contradicted by the metadata

The reviewed sample used in the deck is too_specific_review_base120.tsv (120 answers, one reviewer:
Claude, 2026-10-22): stated 46 %, plausible 18 %, arguable 18 %, wrong 18 %. For comparison, the LLM
reranker's too-specific answers (gpt-5.1, 47 reviewed, rerank_pilot.py) were 68 / 15 / 11 / 6 %.
A new draw with `sample` gives a comparable, not identical, set (the original draw's seed was not kept).

python experiments/deck_review/too_specific_review.py sample --output too_specific_sample.tsv
python experiments/deck_review/too_specific_review.py summarise --reviewed experiments/deck_review/too_specific_review_base120.tsv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SLOTS, Data, parser, path  # noqa: E402


def main():
    ap = parser(__doc__, "too_specific_review")
    ap.add_argument("mode", choices=["sample", "summarise"])
    ap.add_argument("--reviewed", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "too_specific_review_base120.tsv"))
    ap.add_argument("--per_slot", type=int, default=40)
    ap.add_argument("--per_study", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "summarise":
        r = pd.read_csv(path(args.reviewed), sep="\t")
        print(r.judgement.value_counts(normalize=True).round(3).to_string())
        print(r.groupby("slot").judgement.value_counts().unstack(fill_value=0).to_string())
        return
    d = Data(args)
    P = pd.read_csv(path(args.predictions), sep="\t", dtype=str, keep_default_na=False)
    P = P[P.method == "prototype"]
    text = d.samples.set_index("sample_id").text
    rows = []
    for slot in SLOTS:
        p = P[P.slot == slot]
        too = np.array([g in d.anc.get(t, ()) for t, g in zip(p.pred, p.gold)])  # top-1 below the gold term
        c = p[too].sample(frac=1, random_state=args.seed).groupby("study_code").head(args.per_study).head(args.per_slot)
        rows.append(pd.DataFrame({"slot": slot, "sample_id": c.sample_id, "study_code": c.study_code,
                                  "gold_label": c.gold.map(d.label), "pred_label": c.pred.map(d.label), "prob": c.prob,
                                  "text": c.sample_id.map(text), "judgement": ""}))
        print(slot, "too specific:", too.mean().round(3), "drawn", len(c))
    out = args.output if args.output.endswith(".tsv") else args.output.replace(".json", ".tsv")
    pd.concat(rows).to_csv(path(out), sep="\t", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
