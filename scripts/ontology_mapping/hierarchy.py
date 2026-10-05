"""
Calibrated probabilities and hierarchical back-off, shared by 5_evaluate.py, 6_predict_atlas.py
and 7_rerank_atlas.py (imported, never run directly).

  1. calibration: a base model's scores over the slot's training labels -> probabilities P,
     softmax(scores / T), with the temperature T fitted by log-likelihood (fit_temperature);
  2. sum up the ontology: q(node) = sum of P over the labels at or below the node
     (q = P @ A, A = label x node closure matrix, each label counted once, so q <= 1 and q never
     decreases going up, also with several parents);
  3. climbing (Closure.decode): among the top-1 label and its ancestors, the most specific node with
     q >= tau. Only terms Metalog uses in the slot are answers (and never the slot roots below);
     nothing allowed -> abstain. tau -> 0 gives the plain top-1;
  4. tau per slot is chosen for a target accuracy on out-of-fold predictions (choose_tau).

Accuracy, strict: the answer is the gold term or one of its ancestors (true, maybe coarser).
Accuracy, lenient: also counts answers *below* gold ("too specific"): Metalog curators often stop at
a general term, and a manual review found most such answers true (claude/rerank-pilot-results.md).
Background: claude/hierarchical-prediction-literature.md, claude/hierarchical-backoff-results.md.
"""
import numpy as np
from scipy.optimize import minimize_scalar

# slot roots / upper classes: true of nearly everything, so never an answer ("this is from a biome")
NO_ANSWER = {"ENVO_00000428",   # biome
             "ENVO_00010483",   # environmental material
             "ENVO_01000254"}   # environmental system
TAUS = np.round(np.arange(0.05, 1.0001, 0.025), 3)
EXACT, COARSER, TOO_SPECIFIC, OTHER = 1, 2, 3, 4  # outcome codes, 0 = abstain


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


class Closure:
    """Label x node ancestor-closure matrix of one vocabulary (the training labels of a slot)."""

    def __init__(self, vocab, anc, answers="labels", floor=0.25):
        self.vocab = list(vocab)
        nodes = sorted(set(self.vocab).union(*[anc.get(v, set()) for v in self.vocab]))
        self.nodes = np.array(nodes)
        col = {n: j for j, n in enumerate(nodes)}
        self.label_col = np.array([col[v] for v in self.vocab])
        self.A = np.zeros((len(self.vocab), len(nodes)), dtype=np.float32)
        for i, v in enumerate(self.vocab):
            self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] = 1
        self.n_below = self.A.sum(0)  # labels at or below each node
        self.is_label = np.isin(self.nodes, self.vocab)
        self.depth = np.array([len(anc.get(n, ())) for n in nodes])  # number of ancestors
        # allowed answers: "labels" = only terms Metalog uses in this slot (any depth);
        # "ontology" = also their ENVO/UBERON ancestors, unless above `floor` of the slot's labels
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
        (so tau -> 0 gives the top-1 label). -> (node index per sample, -1 = abstain; q of that node)."""
        q = P @ self.A
        ok = (q >= tau - 1e-12) & self.informative & (self.A[P.argmax(1)] > 0)
        key = np.where(ok, self.rank_key[None, :] + q, -np.inf)
        pick = np.where(ok.any(1), key.argmax(1), -1)
        return pick, np.where(pick >= 0, q[np.arange(len(q)), np.maximum(pick, 0)], 0.0)


def relation(nodes, gold, anc):
    """(samples x nodes) outcome code if that node were the answer: EXACT, COARSER (an ancestor of
    gold: true but coarser), TOO_SPECIFIC (a descendant of gold), OTHER (another branch)."""
    cache, rows = {}, []
    for g in gold:
        if g not in cache:
            ga = anc.get(g, set())
            cache[g] = np.array([EXACT if n == g else COARSER if n in ga else TOO_SPECIFIC if g in anc.get(n, ())
                                 else OTHER for n in nodes], dtype=np.int8)
        rows.append(cache[g])
    return np.vstack(rows) if rows else np.zeros((0, len(nodes)), np.int8)


def outcome(rel, pick):
    """Per-sample outcome code (0 = abstain) of the picked node indices."""
    out = rel[np.arange(len(pick)), np.maximum(pick, 0)].astype(int)
    return np.where(pick >= 0, out, 0)


def summarise(out):
    """Shares of all samples, plus accuracy among the answered ones (strict and lenient)."""
    answered = out > 0
    acc = lambda ok: round(float(np.isin(out[answered], ok).mean()), 4) if answered.any() else None
    share = lambda code: round(float((out == code).mean()), 4)
    return {"coverage": round(float(answered.mean()), 4),
            "accuracy_strict": acc((EXACT, COARSER)), "accuracy_lenient": acc((EXACT, COARSER, TOO_SPECIFIC)),
            "exact": share(EXACT), "coarser": share(COARSER), "too_specific": share(TOO_SPECIFIC),
            "other_branch": share(OTHER)}


def choose_tau(curve, target, metric="accuracy_strict"):
    """The lowest tau (most answers) whose accuracy reaches `target`, from a list of
    {"tau", metric, ...} rows in increasing tau; the highest tau if none does."""
    for row in curve:
        if row[metric] is not None and row[metric] >= target:
            return row["tau"]
    return curve[-1]["tau"]


def oof_backoff(folds, anc, targets=(0.8, 0.85, 0.9, 0.95)):
    """Back-off on out-of-fold scores. folds: [{"S": test x vocab scores, "vocab", "gold"}].
    1. temperature per fold fitted on the *other* folds (cross-fitted), and on all folds (`T_all`,
       for the final model);
    2. the curve: outcome shares at every tau of TAUS, pooled over folds;
    3. per target and metric (strict / lenient), tau chosen on the other folds and applied to the
       held-out fold: an honest estimate of coverage and accuracy at that target.
    -> (report dict, per-fold list of (P, Closure)) ."""
    blocks = []
    for f in folds:
        col = {v: i for i, v in enumerate(f["vocab"])}
        blocks.append((f["S"], np.array([col.get(g, -1) for g in f["gold"]])))
    per_fold = []
    for k, f in enumerate(folds):
        T = fit_temperature([b for j, b in enumerate(blocks) if j != k])
        P = softmax(f["S"], T)
        C = Closure(f["vocab"], anc)
        rel = relation(C.nodes, f["gold"], anc)
        outs = np.array([outcome(rel, C.decode(P, tau)[0]) for tau in TAUS])  # taus x samples
        per_fold.append({"T": T, "P": P, "C": C, "outs": outs})
    pooled = np.concatenate([pf["outs"] for pf in per_fold], axis=1)
    report = {"temperature_per_fold": [round(pf["T"], 4) for pf in per_fold],
              "temperature_all": round(fit_temperature(blocks), 4),
              "curve": [{"tau": float(t), **summarise(pooled[i])} for i, t in enumerate(TAUS)]}
    for metric in ("accuracy_strict", "accuracy_lenient"):
        for target in targets:
            outs, chosen = [], []
            for k, pf in enumerate(per_fold):
                others = np.concatenate([p["outs"] for j, p in enumerate(per_fold) if j != k], axis=1)
                tau = choose_tau([{"tau": float(t), **summarise(others[i])} for i, t in enumerate(TAUS)], target, metric)
                chosen.append(tau)
                outs.append(pf["outs"][list(TAUS).index(tau)])
            name = f"{metric.split('_')[1]}_{target}"
            report[name] = {"tau_per_fold": chosen, **summarise(np.concatenate(outs))}
    return report, [(pf["P"], pf["C"]) for pf in per_fold]


def decode_candidates(cands, p, tau, allowed, anc):
    """Back-off over a candidate list (the LLM reranker's options) instead of the whole vocabulary:
    among the most probable candidate and its broader terms in `allowed` (the slot's labels; the
    slot roots never count), most specific first, the first whose summed probability (itself + the
    candidates below it) reaches tau. Probability outside the candidates is not counted, so this is
    slightly more cautious than Closure.decode. -> (term or None, its summed probability)."""
    top = cands[int(np.argmax(p))]
    chain = [top] + sorted((a for a in anc.get(top, ()) if a in allowed and a not in NO_ANSWER),
                           key=lambda a: len(anc.get(a, ())), reverse=True)
    for node in chain:
        if node in NO_ANSWER:
            continue
        q = sum(pi for c, pi in zip(cands, p) if c == node or node in anc.get(c, ()))
        if q >= tau - 1e-12:
            return node, float(q)
    return None, 0.0


def merge_calibrations(entries):
    """Pool the calibration of one method over several fold seeds (calibration.json entries of
    5_evaluate.py runs on the same samples): every run scores each sample once out of fold, so pooling
    the outcomes = averaging the outcome shares per tau; accuracies are recomputed from the pooled
    shares and the temperature is the geometric mean. One entry is returned unchanged."""
    if len(entries) == 1:
        return entries[0]
    taus = [[r["tau"] for r in e["curve"]] for e in entries]
    if any(t != taus[0] for t in taus):
        raise SystemExit("calibrations to merge have different tau grids")
    curve = []
    for rows in zip(*(e["curve"] for e in entries)):
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("coverage", "exact", "coarser", "too_specific", "other_branch")}
        acc = lambda ok: round(sum(mean[k] for k in ok) / mean["coverage"], 4) if mean["coverage"] > 0 else None
        curve.append({"tau": rows[0]["tau"], **{k: round(v, 4) for k, v in mean.items()},
                      "accuracy_strict": acc(("exact", "coarser")), "accuracy_lenient": acc(("exact", "coarser", "too_specific"))})
    return {"temperature": float(np.exp(np.mean([np.log(e["temperature"]) for e in entries]))), "curve": curve,
            "settings": {**entries[0]["settings"], "fold_seed": [e["settings"].get("fold_seed") for e in entries]}}
