#!/usr/bin/env python3
"""
Cheap upgrades to nearest-term retrieval (trivial-methods-upgrades.md, section 2).

Query = sample embedding, candidates = ontology-term embeddings ("label; synonyms"). Closed
vocabulary = the labels of the *training* fold for that slot (as `retrieval` in 5_evaluate.py).
Hyper-parameters in brackets are tuned by nested CV (3 study-grouped inner folds per outer fold).

  N-A  keywords -> nearest term (the old baseline)          no labels used
  N-B  sub-biome as the query                               no labels used
  N-C  keywords + sub-biome as the query                    no labels used
  N-D  + "modality-gap" centring of both sides              no labels used
  N-E  + ancestor smoothing of term vectors [gamma]         no labels used
  N-F  + log label frequency [beta, query, centring]        label counts only (-> retrieval_prior)
  N-G  + blend with the centroid of the term's training samples [alpha, beta, gamma, query, centring]
                                                            (-> prototype)
  N-P  nearest centroid only [query, centring]
  N-H  label_reg: ridge map from sample space to term space, then nearest term
  N-O1 / N-O2  open vocabulary (all 18.8k terms): keywords / kw+sb centred
  bonus sweep: prototype over all terms with a bonus for terms never seen in training
               (-> prototype_open --prototype_unseen_bonus); overall top-1 and top-1 on unseen labels

python experiments/trivial_nearest_term.py --output ~/MicrobeAtlasProject/ontology_mapping/experiments/trivial_nearest_term.json
(about 20-40 min)
"""
import itertools
import time

import numpy as np
from sklearn.linear_model import Ridge

from _setup import SLOTS, inner_folds, load, parser, save_json, unit


def main():
    args = parser(__doc__).parse_args()
    samples, kw, sb, terms, T, folds = load(args)
    study = samples["study_code"].to_numpy()
    tid = terms["term_id"].to_numpy()
    row = {t: i for i, t in enumerate(tid)}
    parents = {t: [p for p in ps.split("||") if p in row] for t, ps in zip(tid, terms["parents"])}

    def ancestors(t, seen=None):
        seen = set() if seen is None else seen
        for p in parents.get(t, []):
            if p not in seen:
                seen.add(p)
                ancestors(p, seen)
        return seen
    anc_mean = np.array([T[[row[a] for a in ancestors(t)]].mean(0) if ancestors(t) else T[i] for i, t in enumerate(tid)])
    Q = {"kw": kw, "sb": sb, "kwsb": unit(kw + sb)}

    def predict(y, tr, te, vocab, c):
        V = np.array([row[v] for v in vocab])
        X = Q[c["q"]]
        tv = unit(T[V] + c.get("gamma", 0) * anc_mean[V])
        xq_tr, xq_te = X[tr], X[te]
        if c.get("center"):
            mq = xq_tr.mean(0)
            xq_te, tv = unit(xq_te - mq), unit(tv - tv.mean(0))
            xq_tr = unit(xq_tr - mq)
        alpha = c.get("alpha", 0)
        pos = {v: j for j, v in enumerate(vocab)}
        n = np.zeros(len(vocab))
        for v, cnt in zip(*np.unique(y[tr], return_counts=True)):
            if v in pos:
                n[pos[v]] = cnt
                if alpha > 0:
                    cen = xq_tr[y[tr] == v].mean(0)
                    z = alpha * cen / np.linalg.norm(cen) + (1 - alpha) * tv[pos[v]]
                    tv[pos[v]] = z / np.linalg.norm(z)
        sc = xq_te @ tv.T
        if c.get("beta", 0) > 0:
            sc = sc + c["beta"] * np.log((n + 0.5) / (n + 0.5).sum())
        sc = sc + c.get("bonus", 0) * (n == 0)
        return np.asarray(vocab)[sc.argmax(1)]

    def label_reg(y, tr, te, vocab, c):
        V = np.array([row[v] for v in vocab])
        X = np.hstack([kw, sb]) / np.sqrt(2)
        W = Ridge(alpha=10).fit(X[tr], T[[row[v] for v in y[tr]]])
        return np.asarray(vocab)[(unit(W.predict(X[te])) @ T[V].T).argmax(1)]

    def run(fn, y, tr, te, grid, is_open):
        vocab_of = lambda idx: list(tid) if is_open else sorted(set(y[idx]))
        configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        best = configs[0]
        if len(configs) > 1:
            acc = np.zeros(len(configs))
            for a, b in inner_folds(study, tr):
                for j, c in enumerate(configs):
                    acc[j] += (fn(y, a, b, vocab_of(a), c) == y[b]).sum()
            best = configs[int(acc.argmax())]
        return fn(y, tr, te, vocab_of(tr), best), best

    B = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
    QQ = ["kw", "kwsb"]
    rows = [
        ("N-A keywords, closed vocab (old baseline)", predict, dict(q=["kw"]), False),
        ("N-B query = sub-biome only", predict, dict(q=["sb"]), False),
        ("N-C query = keywords + sub-biome", predict, dict(q=["kwsb"]), False),
        ("N-D + modality-gap centring", predict, dict(q=["kwsb"], center=[True]), False),
        ("N-E + ancestor smoothing of term vectors", predict, dict(q=["kwsb"], center=[True], gamma=[0, 0.5, 1]), False),
        ("N-F + log label frequency", predict, dict(q=QQ, center=[False, True], gamma=[0, 0.5, 1], beta=B), False),
        ("N-G + few-shot centroid blend", predict, dict(q=QQ, center=[False, True], gamma=[0, 1], beta=B, alpha=[0.25, 0.5, 0.75, 1.0]), False),
        ("N-P nearest centroid only", predict, dict(q=QQ, center=[False, True], alpha=[1.0]), False),
        ("N-H label_reg", label_reg, dict(q=["kwsb"]), False),
        ("N-O1 open vocab, keywords", predict, dict(q=["kw"]), True),
        ("N-O2 open vocab, kw+sb, centred", predict, dict(q=["kwsb"], center=[True]), True),
    ]
    results = {}
    for slot in SLOTS:
        y = samples[slot].to_numpy()
        has = y != ""
        for name, fn, grid, is_open in rows:
            t0, ok, chosen = time.time(), np.zeros(len(y), bool), []
            for tr, te in folds:
                a, b = tr[has[tr]], te[has[te]]
                pred, best = run(fn, y, a, b, grid, is_open)
                ok[b] = pred == y[b]
                chosen.append(best)
            results.setdefault(name, {})[slot] = round(float(ok[has].mean()), 4)
            results.setdefault(name + " | chosen", {})[slot] = chosen
            print(f"{slot:8s} {name:44s} {ok[has].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        # unseen-term bonus for the open prototype (fixed alpha 0.5, beta 0.1, centred kw+sb)
        sweep = {}
        for bonus in [0, 0.2, 0.4, 0.5, 0.6, 0.8]:
            ok, unseen_ok = [], []
            for tr, te in folds:
                a, b = tr[has[tr]], te[has[te]]
                p = predict(y, a, b, list(tid), dict(q="kwsb", center=True, alpha=0.5, beta=0.1, bonus=bonus))
                unseen = ~np.isin(y[b], y[a])
                ok.extend(p == y[b])
                unseen_ok.extend((p == y[b])[unseen])
            sweep[str(bonus)] = {"top1": round(float(np.mean(ok)), 4), "top1_unseen_label": round(float(np.mean(unseen_ok)), 4)}
        results.setdefault("prototype_open unseen-bonus sweep", {})[slot] = sweep
        print(f"{slot:8s} unseen-bonus sweep {sweep}", flush=True)
        save_json(results, args.output)


if __name__ == "__main__":
    main()
