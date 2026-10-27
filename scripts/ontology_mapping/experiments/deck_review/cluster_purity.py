#!/usr/bin/env python3
"""
Do the text embeddings carry habitat, not only study identity? Agreement with the 16S community clusters
(deck slide "The text embedding carries habitat, not only study identity"; report §8.7).

Metalog-linked samples with a coarse 16S cluster; clusters with >= 5 studies; a study-balanced draw
(<= 8 samples per study and cluster, <= 40 per cluster), 5 draws. For keywords, sub-biomes and kw + sb
(the mapping's input: the two vectors concatenated / sqrt 2, no new embeddings needed):
purity@10 = share of a sample's 10 nearest neighbours in its cluster, with any neighbour and with
same-study neighbours removed; own-study share of the 10 nearest; chance = random-pair baseline.

Findings (2,747 samples, 122 clusters): blocked purity@10 sub-biomes .111 (11x chance), keywords .137 (13x),
kw + sb .130 (13x); own-study neighbours 21 / 80 / 74 %.

python experiments/deck_review/cluster_purity.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Data, parser, path, save_json  # noqa: E402


def main():
    ap = parser(__doc__, "cluster_purity", needs=["samples", "keywords", "sub_biomes"])
    ap.add_argument("--clusters", default="~/MicrobeAtlasProject/ontology_mapping/atlas_coverage/sample_to_cluster_coarse.tsv")
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()
    d = Data(args, max_per_study=0)  # every linked sample; the draw below balances studies
    cl = pd.read_csv(path(args.clusters), sep="\t", dtype=str)
    cl = cl.rename(columns={cl.columns[0]: "sample_id", cl.columns[1]: "cluster"})[["sample_id", "cluster"]]
    s = d.samples[["sample_id", "study_code"]].reset_index().merge(cl, on="sample_id")
    ns = s.groupby("cluster").study_code.nunique()
    s = s[s.cluster.isin(ns[ns >= 5].index)]
    print("linked samples with a cluster in >= 5 studies:", len(s), flush=True)
    h = d.X.shape[1] // 2
    res = []
    for seed in range(args.draws):
        x = s.sample(frac=1, random_state=seed).groupby(["cluster", "study_code"]).head(8).groupby("cluster").head(40).reset_index(drop=True)
        X = d.X[x["index"].to_numpy()]
        reps = {"keywords": X[:, :h] * np.sqrt(2), "sub-biomes": X[:, h:] * np.sqrt(2), "keywords + sub-biomes": X}
        clu, st = x.cluster.to_numpy(), x.study_code.to_numpy()
        base = np.mean([(clu == clu[i]).sum() - 1 for i in range(len(x))]) / (len(x) - 1)
        row = {"seed": seed, "n": len(x), "clusters": x.cluster.nunique(), "studies": x.study_code.nunique(), "chance": float(base)}
        for name, V in reps.items():
            sim = V @ V.T
            np.fill_diagonal(sim, -9)
            nn = np.argsort(-(sim - 1e-9 * np.arange(len(x))), axis=1)[:, :10]
            blocked = sim.copy()
            blocked[st[:, None] == st[None, :]] = -9
            nnb = np.argsort(-blocked, axis=1)[:, :10]
            row[name] = {"purity": float((clu[nn] == clu[:, None]).mean()), "purity_blocked": float((clu[nnb] == clu[:, None]).mean()),
                         "own_study_share": float((st[nn] == st[:, None]).mean())}
        res.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)},
              {k: {a: round(b, 3) for a, b in v.items()} for k, v in row.items() if isinstance(v, dict)}, flush=True)
    mean = {name: {m: float(np.mean([r[name][m] for r in res])) for m in res[0][name]} for name in ["keywords", "sub-biomes", "keywords + sub-biomes"]}
    mean["chance"] = float(np.mean([r["chance"] for r in res]))
    print("mean over draws:", mean)
    save_json({"draws": res, "mean": mean}, args.output)


if __name__ == "__main__":
    main()
