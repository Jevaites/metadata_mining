#!/usr/bin/env python3
"""
Human biome labels: what changes when the obsolete human biome is mapped instead of blanked?
(project doc claude/human-biome-label.md; experiments README §18)

Every human sample in Metalog has the biome `obsolete human-associated habitat [ENVO:00009003]`. ENVO has no
replacement, so steps 2 / 2b blank it: the human domain (56 % of the clean training set) has no biome label, is
never trained on for biome and never scored. The variant maps it to `animal-associated environment
[ENVO:01001002]` (humans are animals; the animal domain already uses this term) with one extra label-map row,
then step 5 is re-run unchanged. This script compares the two step-5 runs.

  prepare   writes the variant label map: --label_map + the row ENVO_00009003 -> ENVO_01001002
  compare   reads both runs' predictions.tsv.gz (and calibration.json of every fold seed) and writes:
    check.tsv          feature / material rows identical between runs? (they must be: only biome labels changed)
    biome_scores.tsv   per method and subset: n, top-1, top-5, macro top-1 (mean per gold label)
                       subsets: before_non_human (the current score), after_non_human (same samples, model
                       trained with human biomes), after_human, after_all
    biome_by_domain.tsv  top-1 per domain, before / after
    backoff.tsv        back-off at --prediction_target (strict, 0.9): tau per fold, coverage and strict accuracy
                       on the same subsets. tau is chosen on the other folds' samples, so in the variant the easy
                       human samples take part in choosing tau for everyone.
    tau_pooled.tsv     the production tau per target (pooled curve of the fold-seed runs, as 6_predict_atlas.py)
                       for biome, before / after, with the pooled coverage and accuracy it was chosen on
    changed.tsv        non-human samples whose top-1 changed (production method), counted by (gold, before, after)
    summary.json       the headline numbers

prepare:
python experiments/human_biome.py prepare \
  --label_map ~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv \
  --output ~/MicrobeAtlasProject/ontology_mapping/experiments/human_biome/biome_label_map.human.tsv
then 2b with that label map into experiments/human_biome/clean and 5_evaluate.py on its training_set.clean.tsv.gz
(seed 0 all methods, seeds 1-2 --closed_only; commands in experiments/README.md §18), then:
python experiments/human_biome.py compare \
  --before ~/MicrobeAtlasProject/ontology_mapping/cv_backoff \
  --after ~/MicrobeAtlasProject/ontology_mapping/experiments/human_biome/cv_backoff \
  --before_seeds ~/MicrobeAtlasProject/ontology_mapping/cv_backoff_s1 ~/MicrobeAtlasProject/ontology_mapping/cv_backoff_s2 \
  --after_seeds ~/MicrobeAtlasProject/ontology_mapping/experiments/human_biome/cv_backoff_s1 \
                ~/MicrobeAtlasProject/ontology_mapping/experiments/human_biome/cv_backoff_s2 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/human_biome
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ontology_mapping
from common import load_terms, path, read_tsv, term_ancestors  # noqa: E402
import hierarchy  # noqa: E402

FROM_ID, TO_ID = "ENVO_00009003", "ENVO_01001002"
PRODUCTION = "prototype"


def prepare(args):
    m = read_tsv(args.label_map)
    if (m["from_id"] == FROM_ID).any():
        raise SystemExit(f"{args.label_map} already maps {FROM_ID}")
    row = {c: "" for c in m.columns}
    row.update(slot="biome", from_id=FROM_ID, to_id=TO_ID,
               reason="obsolete human-associated habitat (no replacement in ENVO); humans are animals -> "
                      "animal-associated environment")
    for column, value in (("from_label", "obsolete human-associated habitat"),
                          ("to_label", "animal-associated environment"), ("status", "experiment")):
        if column in row:
            row[column] = value
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    pd.concat([m, pd.DataFrame([row])]).to_csv(path(args.output), sep="\t", index=False)
    print(f"wrote {args.output} ({len(m) + 1} rows)")


def read_predictions(run_dir):
    p = read_tsv(os.path.join(path(run_dir), "predictions.tsv.gz"))
    p["top5"] = p["top5"].str.split(r"\|\|")
    p["pred"] = p["top5"].str[0]
    return p


def subsets(before, after):
    """{name: (frame, label)} for one slot x method; non-human samples are the same rows in both runs."""
    return {
        "before_non_human": before[before["domain"] != "human"],
        "after_non_human": after[after["domain"] != "human"],
        "after_human": after[after["domain"] == "human"],
        "after_all": after,
    }


def scores(g):
    hit = (g["pred"] == g["gold"]).to_numpy()
    return {"n": len(g), "top1": hit.mean() if len(g) else np.nan,
            "top5": np.mean([t in r for r, t in zip(g["top5"], g["gold"])]) if len(g) else np.nan,
            "macro_top1": pd.Series(hit).groupby(g["gold"].to_numpy()).mean().mean() if len(g) else np.nan}


def strict_hit(answer, gold, ancestors):
    return answer != "" and (answer == gold or answer in ancestors.get(gold, ()))


def backoff_scores(g, ancestors):
    answered = g["backoff"] != ""
    hit = np.array([strict_hit(a, t, ancestors) for a, t in zip(g["backoff"], g["gold"])])
    return {"coverage": answered.mean() if len(g) else np.nan,
            "accuracy_strict": hit[answered.to_numpy()].mean() if answered.any() else np.nan,
            "exact": (g["backoff"] == g["gold"]).mean() if len(g) else np.nan}


def pooled_tau(run_dirs, slot, method, target):
    """The tau 6_predict_atlas.py uses: lowest tau whose pooled strict accuracy reaches the target."""
    entries = [json.load(open(os.path.join(path(d), "calibration.json")))[slot][method] for d in run_dirs]
    pooled = hierarchy.merge_calibrations(entries)
    tau = hierarchy.choose_tau(pooled["curve"], target, "accuracy_strict")
    row = next(r for r in pooled["curve"] if r["tau"] == tau)
    return tau, row["coverage"], row["accuracy_strict"]


def compare(args):
    os.makedirs(path(args.output_dir), exist_ok=True)
    out = lambda name: os.path.join(path(args.output_dir), name)
    terms = load_terms(args.ontology_terms)
    ancestors = term_ancestors(terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    before, after = read_predictions(args.before), read_predictions(args.after)
    key = ["sample_id", "slot", "method"]

    # 1. feature / material untouched
    check = []
    for slot in ("feature", "material"):
        b = before[before["slot"] == slot].set_index(key).sort_index()
        a = after[after["slot"] == slot].set_index(key).sort_index()
        same_rows = b.index.equals(a.index)
        n_diff = int((b["top5"].str.join("|") != a["top5"].str.join("|")).sum()) if same_rows else -1
        check.append({"slot": slot, "rows_before": len(b), "rows_after": len(a), "same_rows": same_rows,
                      "top5_differs": n_diff})
    check = pd.DataFrame(check)
    check.to_csv(out("check.tsv"), sep="\t", index=False)
    print(check.to_string(index=False))

    # 2. biome scores per method and subset; 3. by domain; 4. back-off
    b_biome, a_biome = before[before["slot"] == "biome"], after[after["slot"] == "biome"]
    nh_before = set(b_biome.loc[b_biome["domain"] != "human", "sample_id"])
    nh_after = set(a_biome.loc[a_biome["domain"] != "human", "sample_id"])
    assert nh_before == nh_after, "non-human biome rows differ between runs"
    rows, domain_rows, backoff_rows = [], [], []
    calib_before = json.load(open(os.path.join(path(args.before), "metrics.json")))["biome"]
    calib_after = json.load(open(os.path.join(path(args.after), "metrics.json")))["biome"]
    for method in sorted(set(a_biome["method"])):
        b, a = b_biome[b_biome["method"] == method], a_biome[a_biome["method"] == method]
        for name, g in subsets(b, a).items():
            rows.append({"method": method, "subset": name, **scores(g)})
        for run, g in (("before", b), ("after", a)):
            for domain, d in g.groupby("domain"):
                domain_rows.append({"method": method, "run": run, "domain": domain, "n": len(d),
                                    "top1": (d["pred"] == d["gold"]).mean()})
        if "backoff" in calib_after.get(method, {}):  # predictions.tsv holds the answer at --backoff_target only
            target = args.prediction_target
            taus = {"before": calib_before[method]["backoff"][f"strict_{target}"]["tau_per_fold"],
                    "after": calib_after[method]["backoff"][f"strict_{target}"]["tau_per_fold"]}
            for name, g in subsets(b, a).items():
                backoff_rows.append({"method": method, "target": target, "subset": name,
                                     "tau_per_fold": " ".join(map(str, taus[name.split("_")[0]])),
                                     **backoff_scores(g, ancestors)})
    biome_scores = pd.DataFrame(rows).round(4)
    biome_scores.to_csv(out("biome_scores.tsv"), sep="\t", index=False)
    by_domain = pd.DataFrame(domain_rows).round(4)
    by_domain.to_csv(out("biome_by_domain.tsv"), sep="\t", index=False)
    backoff = pd.DataFrame(backoff_rows).round(4)
    backoff.to_csv(out("backoff.tsv"), sep="\t", index=False)

    # 5. the production tau (pooled over fold seeds)
    tau_rows = []
    for method in args.backoff_methods:
        for target in args.targets:
            row = {"method": method, "target": target}
            for run, dirs in (("before", [args.before, *args.before_seeds]), ("after", [args.after, *args.after_seeds])):
                tau, coverage, accuracy = pooled_tau(dirs, "biome", method, target)
                row.update({f"tau_{run}": tau, f"pooled_coverage_{run}": coverage, f"pooled_accuracy_{run}": accuracy})
            tau_rows.append(row)
    tau_pooled = pd.DataFrame(tau_rows)
    tau_pooled.to_csv(out("tau_pooled.tsv"), sep="\t", index=False)

    # 6. which non-human answers changed (production method)
    b = b_biome[(b_biome["method"] == PRODUCTION) & (b_biome["domain"] != "human")].set_index("sample_id")
    a = a_biome[(a_biome["method"] == PRODUCTION) & (a_biome["domain"] != "human")].set_index("sample_id")
    a = a.loc[b.index]
    moved = pd.DataFrame({"domain": b["domain"], "gold": b["gold"], "before": b["pred"], "after": a["pred"]})
    moved = moved[moved["before"] != moved["after"]]
    changed = (moved.groupby(["domain", "gold", "before", "after"]).size().rename("n").reset_index()
               .sort_values(["n", "domain", "gold", "before", "after"], ascending=[False, True, True, True, True],
                            kind="stable"))
    for column in ("gold", "before", "after"):
        changed.insert(changed.columns.get_loc(column) + 1, f"{column}_label", changed[column].map(label_of))
    changed.to_csv(out("changed.tsv"), sep="\t", index=False)

    def pick(method, subset, column, frame=biome_scores):
        return float(frame.loc[(frame["method"] == method) & (frame["subset"] == subset), column].iloc[0])

    summary = {
        "production_method": PRODUCTION,
        "biome_rows": {"before": int(len(b_biome[b_biome["method"] == PRODUCTION])),
                       "after": int(len(a_biome[a_biome["method"] == PRODUCTION]))},
        "top1": {s: pick(PRODUCTION, s, "top1") for s in ("before_non_human", "after_non_human", "after_human",
                                                          "after_all")},
        "macro_top1": {s: pick(PRODUCTION, s, "macro_top1") for s in ("before_non_human", "after_non_human",
                                                                      "after_human", "after_all")},
        "non_human_top1_changed": int(len(moved)),
        "non_human_changed_to_animal_associated": int((moved["after"] == TO_ID).sum()),
        "non_human_changed_from_animal_associated": int((moved["before"] == TO_ID).sum()),
        "backoff_target": args.prediction_target,
        "backoff": {s: {k: float(backoff.loc[(backoff["method"] == PRODUCTION) & (backoff["subset"] == s), k]
                                 .iloc[0]) for k in ("coverage", "accuracy_strict")}
                    for s in ("before_non_human", "after_non_human", "after_human", "after_all")},
        "tau_pooled": tau_pooled.to_dict("records"),
        "feature_material_unchanged": bool((check["same_rows"] & (check["top5_differs"] == 0)).all()),
    }
    json.dump(summary, open(out("summary.json"), "w"), indent=1, sort_keys=True)
    print(biome_scores[biome_scores["method"].isin([PRODUCTION, "linear", "knn", "majority"])].to_string(index=False))
    print(backoff[backoff["method"] == PRODUCTION].to_string(index=False))
    print(tau_pooled.to_string(index=False))
    print(changed.head(10).to_string(index=False))
    print(json.dumps(summary, indent=1, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--label_map", required=True, help="the current biome label map (2b --label_map)")
    pr.add_argument("--output", required=True)
    co = sub.add_parser("compare")
    co.add_argument("--before", required=True, help="5_evaluate.py output dir of the current training set")
    co.add_argument("--after", required=True, help="5_evaluate.py output dir of the variant training set")
    co.add_argument("--before_seeds", nargs="*", default=[], help="--closed_only runs of the other fold seeds")
    co.add_argument("--after_seeds", nargs="*", default=[])
    co.add_argument("--ontology_terms", default="~/MicrobeAtlasProject/ontology_terms.tsv.gz")
    co.add_argument("--backoff_methods", nargs="+", default=["prototype", "linear"])
    co.add_argument("--targets", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95])
    co.add_argument("--prediction_target", type=float, default=0.9, help="5_evaluate.py --backoff_target")
    co.add_argument("--output_dir", required=True)
    args = p.parse_args()
    prepare(args) if args.command == "prepare" else compare(args)


if __name__ == "__main__":
    main()
