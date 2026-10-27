#!/usr/bin/env python3
"""
Compare biome label maps by their effect on step 5 (study-grouped CV), biome slot only.

Arms: one step-5 run per (label map, fold seed), all else equal (same samples, folds, features).
Scored per arm, seed and method:
  unchanged   samples whose RAW Metalog biome is a real biome (not a from_id of any map) and whose label is
              the same in every arm. Their labels are not touched, so a change in score there is caused only
              by how the relabelled training samples teach the model (the fair comparison; label-map
              independent). top-1, top-5, macro top-1 (mean over gold labels), back-off coverage and strict
              accuracy at the run's 90 % target.
  remapped    samples whose raw biome IS a from_id: scored against the arm's own labels (learnability of the
              chosen terms; not comparable as "accuracy" between arms, since the labels differ).
  all         every biome-labelled sample, own labels (what 5_evaluate prints).
Differences are paired (same seed, same samples) against the reference arm.
flips.tsv: unchanged samples whose top-1 correctness differs from the reference arm, by (method, arm, gold,
reference prediction, arm prediction), summed over seeds, with the main studies: which relabelled terms pull
which genuinely labelled samples (e.g. a specific term used by one study attracting the samples of another
study of the same site that its curators labelled with the broader term).

Layout (experiments README §19): <runs>/<arm>/cv_s<seed>/predictions.tsv.gz from 5_evaluate.py --closed_only on
<X>/<arm>/clean/training_set.clean.tsv.gz (2b_clean_metalog.py --label_map <X>/maps/<arm>.tsv); the maps give the
from_ids (raw terms that some map relabels).

python experiments/biome_map_compare.py --runs $X/runs --arms current llm reviewed reviewed_v2 reviewed_v3 \
  --seeds 0 1 2 --training_set $X/current/clean/training_set.clean.tsv.gz --maps $X/maps \
  --ontology_terms $P/ontology_terms.tsv.gz --output $X/results
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ontology_mapping
from common import read_tsv, term_ancestors  # noqa: E402

METHODS = ["prototype", "linear", "knn_study"]


def strict_ok(answer, gold, anc):
    """Back-off answer is the gold term or one of its ancestors."""
    return answer == gold or answer in anc.get(gold, set())


def score(df, anc):
    if len(df) == 0:
        return {}
    top1 = (df["pred"] == df["gold"]).to_numpy()
    top5 = np.array([g in t.split("||") for g, t in zip(df["gold"], df["top5"])])
    macro = pd.Series(top1).groupby(df["gold"].to_numpy()).mean().mean()
    out = {"n": len(df), "top1": top1.mean(), "top5": top5.mean(), "macro_top1": macro}
    if "backoff" in df and df["backoff"].notna().any():
        b = df["backoff"].fillna("")
        answered = (b != "").to_numpy()
        ok = np.array([a != "" and strict_ok(a, g, anc) for a, g in zip(b, df["gold"])])
        out["backoff_coverage"] = answered.mean()
        out["backoff_strict_acc"] = ok[answered].mean() if answered.any() else np.nan
        out["backoff_right_share"] = ok.mean()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--reference", default="current")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--training_set", required=True)
    p.add_argument("--maps", required=True)
    p.add_argument("--ontology_terms", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    terms = read_tsv(args.ontology_terms)
    anc = term_ancestors(terms)
    froms = set()
    for arm in args.arms:
        m = read_tsv(os.path.join(args.maps, f"{arm}.tsv"))
        froms |= set(m.loc[m["slot"] == "biome", "from_id"])
    raw = read_tsv(args.training_set).set_index("sample_id")["biome_raw"]

    rows, by_term, flips = [], [], []
    for seed in args.seeds:
        preds = {}
        for arm in args.arms:
            f = os.path.join(args.runs, arm, f"cv_s{seed}", "predictions.tsv.gz")
            d = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[])
            d = d[(d["slot"] == "biome") & d["method"].isin(METHODS)].copy()
            d["backoff"] = d["backoff"].replace("", np.nan)
            preds[arm] = d
        # samples scored in every arm with the same gold label, raw biome not remapped
        gold = pd.concat({a: d.drop_duplicates("sample_id").set_index("sample_id")["gold"] for a, d in preds.items()}, axis=1)
        common_ids = gold.dropna().index
        same = gold.loc[common_ids].nunique(axis=1) == 1
        unchanged = set(i for i in common_ids[same.to_numpy()] if raw.get(i, "") not in froms)
        ref_pred = {m: preds[args.reference][preds[args.reference]["method"] == m].set_index("sample_id")
                    for m in METHODS}
        for arm, d in preds.items():
            for method in METHODS:
                dm = d[d["method"] == method]
                if arm != args.reference:
                    r = ref_pred[method]
                    for sid, gold_id, pred in zip(dm["sample_id"], dm["gold"], dm["pred"]):
                        if sid in unchanged and sid in r.index:
                            before = r.at[sid, "pred"]
                            if (before == gold_id) != (pred == gold_id):
                                flips.append({"method": method, "arm": arm, "seed": seed, "gold": gold_id,
                                              "reference_pred": before, "arm_pred": pred,
                                              "delta": int(pred == gold_id) - int(before == gold_id),
                                              "study": r.at[sid, "study_code"]})
                is_remapped = dm["sample_id"].map(lambda i: raw.get(i, "") in froms)
                for subset, sel in [("unchanged", dm["sample_id"].isin(unchanged)), ("remapped", is_remapped),
                                    ("all", pd.Series(True, index=dm.index))]:
                    s = score(dm[sel.to_numpy()], anc)
                    rows.append({"seed": seed, "arm": arm, "method": method, "subset": subset, **s})
                rem = dm[is_remapped.to_numpy()]
                for from_id, g in rem.groupby(rem["sample_id"].map(raw)):
                    by_term.append({"seed": seed, "arm": arm, "method": method, "from_id": from_id,
                                    "gold": g["gold"].iloc[0], "n": len(g), "top1": (g["pred"] == g["gold"]).mean()})
    res = pd.DataFrame(rows)
    labels = dict(zip(terms["term_id"], terms["label"]))
    if flips:
        fl = pd.DataFrame(flips)
        fl = fl.groupby(["method", "arm", "gold", "reference_pred", "arm_pred"]).agg(
            net=("delta", "sum"), n=("delta", "size"),
            studies=("study", lambda x: "; ".join(f"{k} ({v})" for k, v in x.value_counts().head(4).items()))).reset_index()
        for c in ["gold", "reference_pred", "arm_pred"]:
            fl[c] = fl[c].map(lambda t: labels.get(t, t))
        fl = fl.reindex(fl["net"].abs().sort_values(ascending=False).index)
        fl.to_csv(os.path.join(args.output, "flips.tsv"), sep="\t", index=False)
    bt = pd.DataFrame(by_term)
    bt = bt.groupby(["method", "from_id", "arm"]).agg(gold=("gold", "first"), n=("n", "mean"), top1=("top1", "mean")).reset_index()
    bt["from_label"], bt["gold_label"] = bt["from_id"].map(labels), bt["gold"].map(labels)
    bt.to_csv(os.path.join(args.output, "remapped_by_term.tsv"), sep="\t", index=False)
    res.to_csv(os.path.join(args.output, "scores_per_seed.tsv"), sep="\t", index=False)

    metrics = ["n", "top1", "top5", "macro_top1", "backoff_coverage", "backoff_strict_acc", "backoff_right_share"]
    mean = res.groupby(["subset", "method", "arm"])[metrics].mean().reset_index()
    mean.to_csv(os.path.join(args.output, "scores_mean.tsv"), sep="\t", index=False)
    # paired differences vs the reference arm
    ref = res[res["arm"] == args.reference].set_index(["seed", "method", "subset"])
    diffs = []
    for arm in args.arms:
        if arm == args.reference:
            continue
        a = res[res["arm"] == arm].set_index(["seed", "method", "subset"])
        dlt = (a[metrics[1:]] - ref[metrics[1:]]).reset_index()
        dlt["arm"] = arm
        diffs.append(dlt)
    diffs = pd.concat(diffs)
    summ = diffs.groupby(["subset", "method", "arm"])[metrics[1:]].agg(["mean", "min", "max"])
    summ.to_csv(os.path.join(args.output, "paired_diff_vs_reference.tsv"), sep="\t")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(mean.round(4).to_string(index=False))
    print("\npaired difference vs", args.reference, "(mean, min, max over seeds)")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
