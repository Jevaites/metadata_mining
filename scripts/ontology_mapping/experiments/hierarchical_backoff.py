#!/usr/bin/env python3
"""
Hierarchical back-off (+ simulated reranking) on out-of-fold predictions.

Idea (claude/hierarchical-prediction-literature.md, claude/backoff-plus-rerank-design.md):
  1. base model scores over the slot's training labels -> calibrated probabilities P (softmax with a
     temperature fitted on the *other* folds, i.e. cross-fitted);
  2. sum up the ontology: q(node) = sum of P over the labels that are the node or below it
     (q = P @ A, A = label x node ancestor-closure matrix; each label counted once, so q <= 1 and
     q never decreases going up, also with several parents);
  3. climbing: among the top-1 label and its ancestors, output the most specific node with
     q >= tau (fewest labels below it; ties -> a label, then deeper). --answers labels (default)
     allows only terms Metalog uses in the slot; --answers ontology also their ENVO/UBERON
     ancestors unless they sit above a quarter of the slot's labels; slot roots (biome,
     environmental material, environmental system) are never an answer. Nothing allowed -> abstain.
     tau -> 0 gives the plain top-1. A flat baseline (top-1 if max P >= tau, else abstain) is
     scored the same way, so the value of the hierarchy is hier vs flat at equal accuracy;
  4. tau per slot is chosen on the other folds for a target hierarchical accuracy
     (answer = gold or an ancestor of gold) and applied to the held-out fold.
Simulated reranker (no API call), see simulate_rerank(): the gated least-confident share of the
samples gets a reranker over the base top-k with accuracy r (when gold is in the top-k) and an
informative confidence (AUROC --auroc); it is fused with the base distribution, then the same
back-off runs. r = 1 is the oracle (upper bound). It tells how good a real reranker must be.

Metrics per (base model, slot): exact top-1; gold-or-ancestor; and at each target: coverage
(answered share), accuracy among answered (exact or ancestor), exact share, share answered at a
label, too specific, other branch.

cd scripts/ontology_mapping
python3 experiments/hierarchical_backoff.py --output ~/MicrobeAtlasProject/ontology_mapping/experiments/backoff/results.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import importlib  # noqa: E402
from common import SLOTS, ancestor_sets, load_npz, load_terms, path, select_samples, study_folds  # noqa: E402
from _setup import DEFAULTS  # noqa: E402

evaluate = importlib.import_module("5_evaluate")
TARGETS = (0.80, 0.85, 0.90, 0.95)
# slot roots / upper classes: true of nearly everything, so never an answer ("this is from a biome")
NO_ANSWER = {"ENVO_00000428",   # biome
             "ENVO_00010483",   # environmental material
             "ENVO_01000254"}   # environmental system


# ----------------------------------------------------------------------------- base models
def term_matrix(terms, vectors_path):
    """[T, T] / sqrt(2): term vectors in the space of the two-block sample vectors (label_syn text)."""
    import h5py
    with h5py.File(path(vectors_path), "r") as handle:
        row = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        V = handle["embeddings"][:]
    T = normalize(V[[row[t] for t in terms["text"]]])
    return np.hstack([T, T]) / np.sqrt(2)


def base_scores(model, x_tr, y_tr, x_te, vocab, TM_vocab, a=0.5, b=0.1):
    """(test x vocab) scores of a base model; vocab = the training labels of the slot."""
    if model == "prototype":
        mean, P, bias = evaluate.prototype_model(x_tr, y_tr, TM_vocab, vocab, a, b)
        return normalize(x_te - mean) @ P.T + bias
    if model == "linear":
        clf = RidgeClassifier(alpha=1.0).fit(x_tr, y_tr)
        col = {c: i for i, c in enumerate(clf.classes_)}
        return clf.decision_function(x_te)[:, [col[v] for v in vocab]]
    raise ValueError(model)


def softmax(S, T):
    Z = S / T
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_temperature(blocks):
    """Temperature minimising the NLL of the gold label over [(scores, gold column or -1)]."""
    def nll(logT):
        T, total, n = np.exp(logT), 0.0, 0
        for S, g in blocks:
            m = g >= 0
            if m.any():
                Z = S[m] / T
                Z = Z - Z.max(1, keepdims=True)
                total -= (Z[np.arange(m.sum()), g[m]] - np.log(np.exp(Z).sum(1))).sum()
                n += m.sum()
        return total / max(n, 1)
    return float(np.exp(minimize_scalar(nll, bounds=(-8, 3), method="bounded").x))


# ----------------------------------------------------------------------------- hierarchy
class Closure:
    """Label x node ancestor-closure matrix of one fold's vocabulary."""

    def __init__(self, vocab, anc, answers="labels", floor=0.25):
        self.vocab = list(vocab)
        nodes = sorted(set(self.vocab).union(*[anc.get(v, set()) for v in self.vocab]))
        self.nodes = np.array(nodes)
        col = {n: j for j, n in enumerate(nodes)}
        self.A = np.zeros((len(self.vocab), len(nodes)), dtype=np.float32)
        for i, v in enumerate(self.vocab):
            self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] = 1
        self.n_below = self.A.sum(0)  # labels at or below each node
        self.is_label = np.isin(self.nodes, self.vocab)
        self.depth = np.array([len(anc.get(n, ())) for n in nodes])  # number of ancestors
        # allowed answers: "labels" = only terms Metalog uses in this slot (any depth);
        # "ontology" = also their ENVO/UBERON ancestors, unless above `floor` of the slot's labels
        # (upper classes such as "environmental system" say nothing)
        onto = np.array([n.startswith(("ENVO_", "UBERON_")) for n in nodes])
        not_root = ~np.isin(self.nodes, list(NO_ANSWER))
        if answers == "labels":
            self.informative = self.is_label & (self.n_below < 0.5 * len(self.vocab)) & not_root
        else:
            self.informative = onto & ((self.n_below < floor * len(self.vocab)) | self.is_label) & not_root
        # tie-break among equally specific nodes: labels first, then deeper, then higher q
        self.rank_key = -self.n_below * 1e6 + self.is_label * 1e4 + self.depth * 10

    def decode(self, P, tau):
        """Climbing: among the top-1 label and its ancestors, the most specific node with q >= tau
        (so tau -> 0 gives the top-1 label). -> node index per sample, -1 = abstain."""
        q = P @ self.A
        ok = (q >= tau - 1e-12) & self.informative & (self.A[P.argmax(1)] > 0)
        key = np.where(ok, self.rank_key[None, :] + q, -np.inf)
        pick = key.argmax(1)
        return np.where(ok.any(1), pick, -1)


def relation(nodes, gold, anc):
    """(samples x nodes) category if that node were the answer: 1 exact, 2 ancestor of gold (true but
    coarser), 3 too specific (a descendant of gold), 4 other branch."""
    cache, rows = {}, []
    for g in gold:
        if g not in cache:
            ga = anc.get(g, set())
            cache[g] = np.array([1 if n == g else 2 if n in ga else 3 if g in anc.get(n, ()) else 4
                                 for n in nodes], dtype=np.int8)
        rows.append(cache[g])
    return np.vstack(rows)


def outcome(rel, pick):
    """Per-sample category (0 = abstain) for the picked node indices."""
    out = rel[np.arange(len(pick)), np.maximum(pick, 0)].astype(int)
    return np.where(pick >= 0, out, 0)


def summarise(out, at_label):
    answered = out > 0
    cov = answered.mean()
    return {"coverage": round(float(cov), 4),
            "accuracy_answered": round(float(np.isin(out[answered], (1, 2)).mean()), 4) if answered.any() else None,
            "exact": round(float((out == 1).mean()), 4),
            "gold_or_ancestor": round(float(np.isin(out, (1, 2)).mean()), 4),
            "at_label": round(float((answered & at_label).mean()), 4),
            "too_specific": round(float((out == 3).mean()), 4),
            "other_branch": round(float((out == 4).mean()), 4)}


# ----------------------------------------------------------------------------- reranker simulation
def simulate_rerank(P, gold_col, gate_frac, r, auroc, k=5, rng=None):
    """Gated samples (the `gate_frac` share with the lowest max P) get a simulated reranker over the
    base top-k:
      * its pick: the gold label with probability r when gold is in the top-k, otherwise the
        best-ranked non-gold candidate (so a wrong reranker can demote a correct top-1);
      * its confidence: an informative score z = mu * [pick correct] + N(0, 1), with mu set so that
        z separates right from wrong picks with the given AUROC (LLM option probabilities reach
        ~0.85, Plaut et al. 2024), turned into the Bayes-calibrated P(correct | z);
      * fusion: the pick gets that share of the base top-k mass, the other candidates share the rest
        in base proportions; base mass outside the top-k is untouched.
    -> fused P, gated indices, the reranker's accuracy on the gated samples."""
    from scipy.stats import norm
    P = P.copy()
    order = np.argsort(-P, axis=1)[:, :k]
    gated = np.argsort(P.max(1))[:int(round(gate_frac * len(P)))]
    in_k = (order[gated] == gold_col[gated, None]).any(1)
    right = in_k & (rng.random(len(gated)) < r)
    mu = np.sqrt(2) * norm.ppf(auroc)
    z = mu * right + rng.standard_normal(len(gated))
    pi = min(max(right.mean(), 1e-6), 1 - 1e-6)
    conf = pi * norm.pdf(z - mu) / (pi * norm.pdf(z - mu) + (1 - pi) * norm.pdf(z))
    for idx, s in enumerate(gated):
        cand = order[s]
        pick = gold_col[s] if right[idx] else next(cc for cc in cand if cc != gold_col[s])
        mass = P[s, cand].sum()
        others = P[s, cand] * (cand != pick)
        P[s, cand] = mass * ((1 - conf[idx]) * others / max(others.sum(), 1e-12) + conf[idx] * (cand == pick))
    return P, gated, float(right.mean()) if len(gated) else None


def flat_decode(C, P, tau):
    """Flat abstention baseline: the top-1 label if its probability >= tau, else abstain."""
    col = {n: j for j, n in enumerate(C.nodes)}
    lab = np.array([col[v] for v in C.vocab])
    top = P.argmax(1)
    return np.where(P.max(1) >= tau, lab[top], -1)


def evaluate_variant(F, decode, anc):
    taus = np.round(np.arange(0.05, 1.0001, 0.025), 3)
    per_fold = []
    for f in F:
        outs, labs = [], []
        for tau in taus:
            pick = decode(f["C"], f["Pv"], tau)
            outs.append(outcome(f["rel"], pick))
            labs.append(np.where(pick >= 0, f["C"].is_label[np.maximum(pick, 0)], False))
        per_fold.append((np.array(outs), np.array(labs)))
    res = {"curve": []}
    for t_i, tau in enumerate(taus):
        o = np.concatenate([pf[0][t_i] for pf in per_fold])
        l = np.concatenate([pf[1][t_i] for pf in per_fold])
        res["curve"].append({"tau": float(tau), **summarise(o, l)})
    for target in TARGETS:  # tau chosen on the other folds, applied to the held-out fold
        o_all, l_all, chosen = [], [], []
        for k in range(len(F)):
            accs = []
            for t_i in range(len(taus)):
                o = np.concatenate([per_fold[j][0][t_i] for j in range(len(F)) if j != k])
                accs.append(np.isin(o[o > 0], (1, 2)).mean() if (o > 0).any() else 0)
            ok = [t_i for t_i, acc in enumerate(accs) if acc >= target]
            t_i = ok[0] if ok else len(taus) - 1
            chosen.append(float(taus[t_i]))
            o_all.append(per_fold[k][0][t_i])
            l_all.append(per_fold[k][1][t_i])
        res[f"target_{target}"] = {"tau_per_fold": chosen, **summarise(np.concatenate(o_all), np.concatenate(l_all))}
    return res


# ----------------------------------------------------------------------------- main
def run(args):
    terms = load_terms(args.ontology_terms)
    term_ids = terms["term_id"].to_numpy()
    parents = {t: set(p.split("||")) for t, p in zip(term_ids, terms["parents"]) if p}
    anc = ancestor_sets(parents)
    TM = term_matrix(terms, args.term_vectors)
    trow = {t: i for i, t in enumerate(term_ids)}
    (kr, K), (sr, B) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kr), set(sr)])
    if args.label_map:  # same format as 2b_clean_metalog.py --label_map: slot, from_id, to_id
        from common import read_tsv
        lm = read_tsv(args.label_map)
        for slot in SLOTS:
            m = {r.from_id.strip(): r.to_id.strip() for r in lm.itertuples() if r.slot.strip() in (slot, "*")}
            before = samples[slot].copy()
            samples[slot] = samples[slot].map(lambda t: m.get(t, t))
            if m:
                print(f"label map, {slot}: {(before != samples[slot]).sum()} labels changed "
                      f"({((before != '') & (samples[slot] == '')).sum()} blanked)", flush=True)
    X = np.hstack([K[[kr[s] for s in samples["sample_id"]]], B[[sr[s] for s in samples["sample_id"]]]]) / np.sqrt(2)
    studies = samples["study_code"].to_numpy()
    print(f"{len(samples)} samples, {len(set(studies))} studies", flush=True)
    rng = np.random.default_rng(0)
    results = {}
    for seed in range(args.fold_seeds):
        for slot in SLOTS:
            y = samples[slot].to_numpy()
            folds = []  # per fold: test idx, vocab, scores, gold column
            for tr, te in study_folds(studies, 5, seed):
                a, b = tr[y[tr] != ""], te[y[te] != ""]
                vocab = np.unique(y[a])
                col = {v: i for i, v in enumerate(vocab)}
                for model in args.models:
                    S = base_scores(model, X[a], y[a], X[b], vocab, TM[[trow[v] for v in vocab]])
                    folds.append({"model": model, "test": b, "vocab": vocab, "S": S,
                                  "gold": y[b], "gcol": np.array([col.get(g, -1) for g in y[b]])})
            for model in args.models:
                F = [f for f in folds if f["model"] == model]
                # cross-fitted temperature and per-fold closures / probabilities
                for k, f in enumerate(F):
                    f["T"] = fit_temperature([(g["S"], g["gcol"]) for j, g in enumerate(F) if j != k])
                    f["P"] = softmax(f["S"], f["T"])
                    f["C"] = Closure(f["vocab"], anc, args.answers)
                    f["rel"] = relation(f["C"].nodes, f["gold"], anc)
                gold_all = np.concatenate([f["gold"] for f in F])
                variants = {"base": None}
                for gate in args.gates:
                    for r in args.rerank_acc:
                        variants[f"rerank_r{r}_gate{gate}"] = (gate, r)
                for name, spec in variants.items():
                    extra = {}
                    for f in F:
                        if spec is None:
                            f["Pv"] = f["P"]
                        else:
                            f["Pv"], _, acc = simulate_rerank(f["P"], f["gcol"], spec[0], spec[1], args.auroc, args.k, rng)
                            extra.setdefault("reranker_acc_on_gated", []).append(acc)
                    top = np.concatenate([np.argsort(-f["Pv"], 1)[:, :args.k] for f in F])
                    gcol = np.concatenate([f["gcol"] for f in F])
                    top1 = np.concatenate([f["vocab"][f["Pv"].argmax(1)] for f in F])
                    in_k = (top == gcol[:, None]).any(1)
                    res = {"top1_exact": round(float((top1 == gold_all).mean()), 4),
                           "top1_gold_or_ancestor": round(float(np.mean(
                               [p == g or p in anc.get(g, ()) for p, g in zip(top1, gold_all)])), 4),
                           f"top{args.k}_recall": round(float(in_k.mean()), 4),
                           f"top1_given_gold_in_top{args.k}": round(float((top1 == gold_all)[in_k].mean()), 4)}
                    if spec is None:  # how good the base model is inside its own top-k on the gated share
                        for gate in args.gates:
                            sub = []
                            for f in F:
                                g_idx = np.argsort(f["P"].max(1))[:int(round(gate * len(f["P"])))]
                                o = np.argsort(-f["P"][g_idx], 1)[:, :args.k]
                                ink = (o == f["gcol"][g_idx, None]).any(1)
                                sub.append((o[:, 0] == f["gcol"][g_idx])[ink])
                            res[f"base_top1_given_gold_in_top{args.k}_gate{gate}"] = round(float(np.concatenate(sub).mean()), 4)
                    if extra:
                        res["reranker_acc_on_gated"] = round(float(np.mean(extra["reranker_acc_on_gated"])), 4)
                    res["hier"] = evaluate_variant(F, lambda C, P, t: C.decode(P, t), anc)
                    res["flat"] = evaluate_variant(F, flat_decode, anc)
                    res["temperature"] = [round(f["T"], 4) for f in F]
                    results.setdefault(f"seed{seed}", {}).setdefault(slot, {}).setdefault(model, {})[name] = res
                print(f"seed {seed} {slot} done", flush=True)
    return results


def report(results):
    for seed, by_slot in results.items():
        for slot, by_model in by_slot.items():
            print(f"\n=== {seed} {slot}  (cov = answered share, exact = exact share of all samples; tau set on other folds)")
            print(f"{'model / variant':34} top1  g|anc  top5 | " + " | ".join(f"@{t} hier cov/exact  flat cov/exact" for t in TARGETS))
            for model, by_var in by_model.items():
                for name, r in by_var.items():
                    cells = " | ".join(f"     {r['hier'][f'target_{t}']['coverage']:.2f}/{r['hier'][f'target_{t}']['exact']:.2f}"
                                       f"        {r['flat'][f'target_{t}']['coverage']:.2f}/{r['flat'][f'target_{t}']['exact']:.2f}" for t in TARGETS)
                    k = [x for x in r if x.endswith("_recall")][0]
                    print(f"{model + ' ' + name:34} {r['top1_exact']:.3f} {r['top1_gold_or_ancestor']:.3f} {r[k]:.3f} | {cells}")
                base = by_var["base"]
                print("   base top-1 accuracy when gold is in its top-k:",
                      {kk: v for kk, v in base.items() if kk.startswith("base_top1_given") or kk.startswith("top1_given")})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "samples", "keywords", "sub_biomes"]:
        p.add_argument(f"--{name}", default=DEFAULTS[name])
    p.add_argument("--term_vectors", default="~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5",
                   help="h5 holding the label_syn term texts (term_text_variants.py embed output or 4_embed_terms.py)")
    p.add_argument("--models", nargs="+", default=["prototype", "linear"])
    p.add_argument("--label_map", default=None, help="TSV slot, from_id, to_id (2b_clean_metalog.py format), "
                   "applied to training and test labels, e.g. draft_biome_label_map.py output")
    p.add_argument("--fold_seeds", type=int, default=1)
    p.add_argument("--answers", choices=["labels", "ontology"], default="labels",
                   help="back off only to terms Metalog uses in the slot, or also to their ontology ancestors")
    p.add_argument("--k", type=int, default=5, help="reranker candidate list size")
    p.add_argument("--gates", nargs="+", type=float, default=[0.3, 1.0], help="share of samples sent to the reranker")
    p.add_argument("--rerank_acc", nargs="+", type=float, default=[0.6, 0.8, 1.0],
                   help="simulated reranker accuracy when gold is in the top-k (1.0 = oracle)")
    p.add_argument("--auroc", type=float, default=0.85, help="simulated reranker: how well its confidence separates right from wrong picks")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    t0 = time.time()
    results = run(args)
    os.makedirs(os.path.dirname(path(args.output)) or ".", exist_ok=True)
    json.dump(results, open(path(args.output), "w"), indent=1)
    report(results)
    print(f"\nwrote {args.output} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
