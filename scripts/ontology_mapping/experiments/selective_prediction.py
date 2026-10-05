"""Coverage vs accuracy of `linear` when only its most confident predictions are kept (confidence =
margin, as in 5_evaluate.py / 6_predict_atlas.py). Same samples, folds and training set as
cleaning_effect.py (raw arm, fold seed 0). Reports, per slot, accuracy when keeping the top X %
most confident predictions, and the margin cut-off reaching 80/85/90/95 % accuracy (apply those to
the *_confidence columns of 6_predict_atlas.py output). ~2 min.

python experiments/selective_prediction.py --clean_dir ~/MicrobeAtlasProject/metalog/clean \
  --output ~/MicrobeAtlasProject/metalog/clean/experiments/selective_prediction.json
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleaning_effect as ce
from common import SLOTS, study_folds

p = ce.parser(__doc__); p.add_argument("--clean_dir", required=True); args = p.parse_args()
u, X = ce.load_universe(args); train_m, test_m = ce.masks(u)
raw_capped = u[u["capped"]]
fold_of = {}
for f, (_, te) in enumerate(study_folds(raw_capped["study_code"], 5, 0)):
    fold_of.update({s: f for s in raw_capped["study_code"].iloc[te]})
fold = u["study_code"].map(fold_of).to_numpy()
rows = ce.cap(u, test_m["test_raw"])
gold_rows = set(ce.cap(u, test_m["test_gold"]))
recs = []
for f in range(5):
    tr = ce.cap(u, train_m["raw"] & (fold != f)); te = rows[fold[rows] == f]
    for slot in SLOTS:
        t = tr[u[slot].to_numpy()[tr] != ""]; e = te[u[slot].to_numpy()[te] != ""]
        clf = RidgeClassifier(alpha=1.0).fit(X[t], u[slot].to_numpy()[t])
        s = clf.decision_function(X[e]); top2 = np.sort(s, axis=1)[:, -2:]
        pred = clf.classes_[s.argmax(1)]
        recs.append(pd.DataFrame({"row": e, "slot": slot, "conf": top2[:, 1] - top2[:, 0],
                                  "correct": pred == u[slot].to_numpy()[e]}))
d = pd.concat(recs, ignore_index=True)
r = u.loc[d["row"]]
d["group"] = np.select([r["drop_reason"].to_numpy() != "", r["artificial_bucket"].to_numpy() != "none",
                        r["audit_unreviewed"].to_numpy()], ["control", "artificial", "audit"], "normal")
d["study"] = r["study_code"].to_numpy(); d["gold"] = d["row"].isin(gold_rows)
out = {}
for slot in SLOTS:
    g = d[d.slot == slot].sort_values("conf", ascending=False).reset_index(drop=True)
    n = len(g); prec = g["correct"].cumsum() / np.arange(1, n + 1)
    res = {"n": n, "top1_all": round(g.correct.mean(), 4), "by_coverage": {}, "thresholds": {}}
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = int(round(cov * n)); res["by_coverage"][f"{int(cov*100)}%"] = round(float(g.correct[:k].mean()), 4)
    for target in [0.8, 0.85, 0.9, 0.95]:
        ok = np.where(prec >= target)[0]
        if len(ok):
            k = ok[-1] + 1; thr = float(g.conf[k - 1]); rest = g[k:]
            res["thresholds"][str(target)] = {
                "margin": round(thr, 4), "coverage": round(k / n, 4),
                "review_share": round(1 - k / n, 4),
                "errors_caught_in_review": round(float((~rest.correct).sum() / (~g.correct).sum()), 4),
                "review_accuracy": round(float(rest.correct.mean()), 4),
                "review_group_mix": rest.group.value_counts(normalize=True).round(3).to_dict(),
                "accepted_accuracy_gold_only": round(float(g[:k][g[:k].gold].correct.mean()), 4)}
    res["errors_by_group"] = (~g.correct).groupby(g.group).sum().astype(int).to_dict()
    res["samples_by_group"] = g.group.value_counts().to_dict()
    res["review_share_by_group_at_0.9"] = (g.conf < res["thresholds"]["0.9"]["margin"]).groupby(g.group).mean().round(3).to_dict() if "0.9" in res["thresholds"] else None
    out[slot] = res
json.dump(out, open(args.output, "w"), indent=1)
print(json.dumps(out, indent=1))
