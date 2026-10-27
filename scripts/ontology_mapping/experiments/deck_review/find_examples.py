#!/usr/bin/env python3
"""
Find real samples to illustrate the deck and the report.

abstain_right  Held-out Metalog samples whose top-1 terms are all right for feature and material, but
               whose back-off abstains (probability below tau): the model is cautious where Metalog is thin.
               Reads the cv_backoff predictions of the production prototype. Used for the woodcock gut
               (ERS4537110, Youngblut_2020_animal: animal-associated environment .95 answered; intestine
               environment .46 and fecal material .50 abstained; all three equal the curator's labels).
facets         Most similar cross-study pairs whose labels are two facets of the same thing (e.g.
               ocean / marine photic zone, hot spring / microbial mat), for the scoring-rules discussion.

python experiments/deck_review/find_examples.py abstain_right --domain animal
python experiments/deck_review/find_examples.py facets
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Data, parser, path  # noqa: E402

FACET_PAIRS = [("feature", "hot spring", "microbial mat"), ("feature", "agricultural soil", "rhizosphere"),
               ("feature", "ocean", "marine photic zone")]


def main():
    ap = parser(__doc__, "find_examples")
    ap.add_argument("mode", choices=["abstain_right", "facets"])
    ap.add_argument("--domain", default="animal", help="Metalog domain filter for abstain_right")
    ap.add_argument("--method", default="prototype")
    args = ap.parse_args()
    d = Data(args)
    label = d.label
    if args.mode == "abstain_right":
        P = pd.read_csv(path(args.predictions), sep="\t", dtype=str)
        P = P[P.method == args.method]
        W = P.pivot_table(index="sample_id", columns="slot", values=["pred", "gold", "prob", "backoff"], aggfunc="first")
        W.columns = [f"{a}_{b}" for a, b in W.columns]
        W = W.join(d.samples.set_index("sample_id")[["study_code", "domain"]])
        ok = W.domain.eq(args.domain)
        for s in ["feature", "material"]:
            ok &= W[f"pred_{s}"].eq(W[f"gold_{s}"]) & W[f"backoff_{s}"].isna()
        for sid, r in W[ok].iterrows():
            print(sid, r.study_code, "|", " | ".join(f"{s}: {label.get(r[f'pred_{s}'])} {float(r[f'prob_{s}']):.2f} "
                                                      f"{'answered' if isinstance(r[f'backoff_{s}'], str) else 'abstain'}"
                                                      for s in ["biome", "feature", "material"]))
    else:
        s, X = d.samples, d.X
        for slot, a, b in FACET_PAIRS:
            names = s[slot].map(label)
            ia, ib = np.where(names.eq(a))[0], np.where(names.eq(b))[0]
            if not len(ia) or not len(ib):
                print("none:", a, b)
                continue
            S = X[ia] @ X[ib].T
            S[d.study[ia][:, None] == d.study[ib][None, :]] = -1
            for k in np.argsort(-S.ravel())[:2]:
                i, j = np.unravel_index(k, S.shape)
                print(f"{a} / {b}: cosine {S[i, j]:.3f}")
                for r in (s.iloc[ia[i]], s.iloc[ib[j]]):
                    print("   ", r.sample_id, r.study_code, "|", " | ".join(label.get(r[x], "") for x in ["biome", "feature", "material"]))
                    print("      ", str(r.text)[:300])


if __name__ == "__main__":
    main()
