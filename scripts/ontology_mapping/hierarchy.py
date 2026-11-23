"""
Calibrated probabilities and hierarchical back-off, shared by 5_evaluate.py, 6_predict_atlas.py and
7_rerank_atlas.py (imported, never run directly).

  1. calibration: a base model's scores over the slot's training labels -> probabilities P =
     softmax(scores / T), the temperature T fitted by log-likelihood on out-of-fold scores;
  2. sum up the ontology: q(node) = sum of P over the labels at or below the node
     (q = P @ A, A = label x node closure matrix; each label counted once, so q <= 1 and q never
     decreases going up, also with several parents);
  3. climbing (Closure.decode): among the top-1 label and its ancestors, the most specific node with
     q >= tau. Only terms Metalog uses in the slot are answers (never the slot roots); none -> abstain.
     tau -> 0 gives the plain top-1;
  4. tau per slot is the lowest one whose out-of-fold accuracy reaches a target (choose_tau);
  5. several answers (SetDecoder, oof_sets; second half of this file): instead of climbing, a small set of
     incomparable labels where the probability is split between unrelated branches, by expected value.

Worked example (biome slot, tau = 0.8; ENVO: ocean biome is_a marine biome is_a aquatic biome, freshwater
biome is_a aquatic biome): P(ocean biome) = 0.55, P(marine biome) = 0.15, P(freshwater biome) = 0.15, the rest
elsewhere. q(ocean biome) = 0.55 < 0.8; q(marine biome) = 0.55 + 0.15 = 0.70 < 0.8; q(aquatic biome) = 0.85
>= 0.8 -> the answer is 'aquatic biome' (true, coarser) instead of a 55 %-sure 'ocean biome'.

Outcome of an answer against the gold label: EXACT, COARSER (an ancestor of gold: true but less
specific), TOO_SPECIFIC (a descendant of gold), OTHER (another branch: wrong), 0 = abstain.
Accuracy, strict: EXACT or COARSER. Lenient: also TOO_SPECIFIC (Metalog curators often stop at a general
term, and a manual review found most "too specific" answers true: claude/rerank-pilot-results.md).
Background: claude/hierarchical-prediction-literature.md, claude/hierarchical-backoff-results.md.
"""
import itertools
from collections import Counter

import numpy as np
from scipy.optimize import minimize_scalar

# slot roots / upper classes: true of nearly everything, so never an answer ("this is from a biome")
NO_ANSWER = {"ENVO_00000428",   # biome
             "ENVO_00010483",   # environmental material
             "ENVO_01000254"}   # environmental system
TAUS = np.round(np.arange(0.05, 1.0001, 0.025), 3)  # the tau grid: 0.05, 0.075, ..., 1.0
EXACT, COARSER, TOO_SPECIFIC, OTHER = 1, 2, 3, 4  # outcome codes, 0 = abstain


def softmax(S, T):
    """Row-wise softmax of scores S / T (max subtracted first for numerical stability)."""
    Z = S / T
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def fit_temperature(blocks):
    """Temperature minimising the mean negative log-likelihood of the gold label.
    blocks: [(scores n x V, gold column per row or -1 when gold is not in that vocabulary)].
    Prototype scores are cosines (range ~0.2), so T comes out small (~0.09); ridge scores ~0.2."""
    def nll(logT):
        T, total, n = np.exp(logT), 0.0, 0
        for S, g in blocks:
            m = g >= 0
            if m.any():
                Z = S[m] / T
                Z = Z - Z.max(1, keepdims=True)
                total -= (Z[np.arange(m.sum()), g[m]] - np.log(np.exp(Z).sum(1))).sum()  # -log softmax(gold)
                n += m.sum()
        return total / max(n, 1)
    return float(np.exp(minimize_scalar(nll, bounds=(-8, 3), method="bounded").x))  # T in [3e-4, 20]


class Closure:
    """Label x node ancestor-closure matrix of one vocabulary (the training labels of a slot).

    nodes = the labels and all their ancestors; A[i, j] = 1 when node j is label i or one of its ancestors.
    Example: labels [marine biome, ocean biome] -> nodes [aquatic biome, biome, marine biome, ocean biome, ...]
    and q = P @ A gives q(marine biome) = P(marine biome) + P(ocean biome)."""

    def __init__(self, vocab, anc, answers="labels", floor=0.25):
        self.vocab = list(vocab)
        nodes = sorted(set(self.vocab).union(*[anc.get(v, set()) for v in self.vocab]))
        self.nodes = np.array(nodes)
        col = {n: j for j, n in enumerate(nodes)}
        self.label_col = np.array([col[v] for v in self.vocab])
        self.A = np.zeros((len(self.vocab), len(nodes)), dtype=np.float32)
        for i, v in enumerate(self.vocab):
            self.A[i, [col[n] for n in ({v} | anc.get(v, set()))]] = 1
        self.n_below = self.A.sum(0)  # labels at or below each node: fewer = more specific
        self.is_label = np.isin(self.nodes, self.vocab)
        self.depth = np.array([len(anc.get(n, ())) for n in nodes])  # number of ancestors (tie-break only)
        # allowed answers: "labels" = only terms Metalog uses in this slot (any depth), unless they sit above
        # half of the slot's labels; "ontology" = every label (any ontology, any share: PO / FOODON labels
        # included) plus their ENVO/UBERON ancestors that are below `floor` of the labels
        onto = np.array([n.startswith(("ENVO_", "UBERON_")) for n in nodes])
        not_root = ~np.isin(self.nodes, list(NO_ANSWER))
        if answers == "labels":
            self.informative = self.is_label & (self.n_below < 0.5 * len(self.vocab)) & not_root
        else:
            self.informative = ((onto & (self.n_below < floor * len(self.vocab))) | self.is_label) & not_root
        # order among candidate answers: most specific first (fewest labels below), then labels, then
        # deeper; q (< 1) breaks the remaining ties because the other terms are multiples of 10
        self.rank_key = -self.n_below * 1e6 + self.is_label * 1e4 + self.depth * 10

    def decode(self, P, tau):
        """Climbing: among the top-1 label and its ancestors, the most specific allowed node with q >= tau
        (so tau -> 0 gives the top-1 label). -> (node index per sample, -1 = abstain; q of that node)."""
        q = P @ self.A  # summed probability of every node
        on_path = self.A[P.argmax(1)] > 0  # the top-1 label and its ancestors
        ok = (q >= tau - 1e-12) & self.informative & on_path
        key = np.where(ok, self.rank_key[None, :] + q, -np.inf)
        pick = np.where(ok.any(1), key.argmax(1), -1)
        return pick, np.where(pick >= 0, q[np.arange(len(q)), np.maximum(pick, 0)], 0.0)


def relation(nodes, gold, anc):
    """(samples x nodes) outcome code if that node were the answer: EXACT, COARSER (an ancestor of
    gold: true but coarser), TOO_SPECIFIC (a descendant of gold), OTHER (another branch)."""
    cache, rows = {}, []
    for g in gold:
        if g not in cache:  # one row per distinct gold label
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
    """Shares of all samples, plus accuracy among the answered ones (strict and lenient).
    Example: outcomes [1, 2, 0, 4] -> coverage 0.75, strict accuracy 2/3, exact 0.25."""
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
    -> (report dict, per-fold list of (P, Closure))."""
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
            name = f"{metric.split('_')[1]}_{target}"  # e.g. "strict_0.9"
            report[name] = {"tau_per_fold": chosen, **summarise(np.concatenate(outs))}
    return report, [(pf["P"], pf["C"]) for pf in per_fold]


def decode_candidates(cands, p, tau, allowed, anc):
    """Back-off over a candidate list (the LLM reranker's options) instead of the whole vocabulary:
    among the most probable candidate and its broader terms in `allowed` (the slot's labels; the
    slot roots never count), most specific first, the first whose summed probability (itself + the
    candidates below it) reaches tau. Probability outside the candidates is not counted, so this is
    slightly more cautious than Closure.decode. Broader terms of equal depth (several parents in the
    DAG) are tried in a fixed order, the more probable first, then by id: the answer never depends on
    the iteration order of the ancestor set (which changes with PYTHONHASHSEED).
    -> (term or None, its summed probability)."""
    def summed(node):  # the node's probability: candidates equal to it or below it
        return sum(pi for c, pi in zip(cands, p) if c == node or node in anc.get(c, ()))

    top = cands[int(np.argmax(p))]
    broader = [a for a in anc.get(top, ()) if a in allowed and a not in NO_ANSWER]
    chain = [top] + sorted(broader, key=lambda a: (-len(anc.get(a, ())), -summed(a), a))  # deepest first
    for node in chain:
        if node in NO_ANSWER:
            continue
        q = summed(node)
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


# ============================================================================= several answers (set-valued)
# experiments/set_valued.py (README §21, claude/set-valued-answers.md) compared set-valued rules on out-of-fold
# probabilities; the one kept for the pipeline is evr_d0.7, the reviewer-score expected value: the answer is the set
# of <= 3 incomparable terms worth the most, d ** (terms - 1) * E[credit of the set's best term] - lambda * P(wrong),
# or nothing when no set is worth more than 0. A single term is the usual answer; sets appear where the probability
# is split between unrelated branches (host-associated vs terrestrial), where the back-off can only abstain.
# Per-sample reference functions (candidate_sets, set_table, reviewer_value_pick, unrelated_runner_up; used by
# set_valued.py) and their vectorised equivalent for 5_evaluate.py / 6_predict_atlas.py (SetDecoder, tested equal).
# the lambda grid: steps of 0.05 up to 5 (2026-11-12; before, 0.5 from 2 on: feature's lambda jumped from 2.0, at
# 89.98 % pooled accuracy, to 2.5 and lost ~4 points answered), then 0.5 up to 19.5
LAMBDAS = np.round(np.concatenate([np.arange(0, 5, 0.05), np.arange(5, 20, 0.5)]), 3)


def label_share(vocab, counts):
    """Training share of each label of `vocab` from {label: training samples}, + 0.5 each (a label with no count keeps
    a finite information content IC = -log share)."""
    cnt = np.array([counts.get(v, 0) for v in vocab], float) + 0.5
    return cnt / cnt.sum()


def candidate_sets(q, A, informative, n_candidates=10, max_terms=3, min_mass=0.02):
    """Antichains of <= max_terms node indices among the n_candidates allowed nodes with the most mass (q >= min_mass).
    A: labels x nodes closure (bool); two nodes are comparable when the labels below one include the other's."""
    order = np.argsort(-q, kind="stable")
    cand = [int(k) for k in order if informative[k] and q[k] >= min_mass][:n_candidates]
    below = {k: A[:, k] for k in cand}

    def comparable(a, b):
        return bool((below[a] <= below[b]).all() or (below[b] <= below[a]).all())

    return [s for m in range(1, max_terms + 1) for s in itertools.combinations(cand, m)
            if all(not comparable(a, b) for a, b in itertools.combinations(s, 2))]


def set_table(p, sets, A, share):
    """Per candidate set: covered-label mask (sets x labels), summed probability, labels covered, number of terms,
    IC = -log(training share of the covered labels), expected credit sum_y p_y min(1, IC(set) / IC(y)) (the whole
    set as one answer), and expected reviewer credit sum_y p_y max over the set's terms t covering y of
    min(1, IC(t) / IC(y)) (a reviewer keeps the best term). p: probability row over labels; share: training share of
    each label (> 0)."""
    cover = np.array([A[:, list(s)].any(1) for s in sets])
    ic_label = -np.log(share)
    ic = -np.log(cover @ share)
    term_credit = {}  # node -> credit of that single term for each label below it (0 elsewhere)
    for n in {n for s in sets for n in s}:
        below = A[:, n]
        term_credit[n] = np.where(below, np.minimum(1, -np.log(share[below].sum()) / ic_label), 0.0)
    best = np.array([np.max([term_credit[n] for n in s], axis=0) for s in sets])
    return {"cover": cover, "mass": cover @ p, "n_labels": cover.sum(1), "n_terms": np.array([len(s) for s in sets]),
            "ic": ic, "expected_credit": (cover * p * np.minimum(1, ic[:, None] / ic_label[None, :])).sum(1),
            "reviewer_credit": best @ p}


def reviewer_value_pick(table, lam, max_terms, discount):
    """Index of the set with <= max_terms terms maximising discount ** (n_terms - 1) * E[reviewer credit]
    - lam * (1 - mass); -1 (abstain) if that is <= 0."""
    worth = table["reviewer_credit"] * discount ** (table["n_terms"] - 1.0)
    value = np.where(table["n_terms"] <= max_terms, worth - lam * (1 - table["mass"]), -np.inf)
    k = int(value.argmax())
    return k if value[k] > 0 else -1


def unrelated_runner_up(p, vocab, anc, allowed, top=5):
    """(label, probability) of the most probable of the `top` labels after the top-1 that is neither an ancestor nor a
    descendant of the top-1 and has no ancestor in `allowed` (the slot's allowed answers) in common with it;
    ("", 0.0) if none. Example: top-1 animal-associated environment .48, terrestrial biome .41 -> (terrestrial, .41);
    ocean .5, marine biome .4 -> marine is an ancestor, skipped."""
    order = np.argsort(-p, kind="stable")
    first = vocab[order[0]]
    up_first = anc.get(first, set()) | {first}
    for j in order[1:top + 1]:
        v = vocab[j]
        if v in anc.get(first, ()) or first in anc.get(v, ()):
            continue
        if (anc.get(v, set()) | {v}) & up_first & allowed:
            continue
        return v, float(p[j])
    return "", 0.0


def cross_fit(answered, covered, fold, target):
    """answered, covered: samples x grid (0/1). Per held-out fold k, the grid index that answers most of the OTHER
    folds' samples with accuracy (covered / answered) >= target (ties: the first); the last grid index if none.
    -> (grid index per sample, {fold: grid index})."""
    chosen, per_sample = {}, np.zeros(len(fold), int)
    for k in np.unique(fold):
        o = fold != k
        n_ans = answered[o].sum(0)
        acc = covered[o].sum(0) / np.maximum(n_ans, 1)
        ok = np.where((acc >= target) & (n_ans > 0))[0]
        chosen[int(k)] = int(ok[n_ans[ok].argmax()]) if len(ok) else answered.shape[1] - 1
        per_sample[fold == k] = chosen[int(k)]
    return per_sample, chosen


class SetDecoder:
    """The reviewer-score set answers of many probability rows at once (same decisions as candidate_sets + set_table +
    reviewer_value_pick row by row: test_hierarchy_unit.py::TestSetDecoder).

    closure: the slot's Closure (labels mode: only labels Metalog uses in the slot are answers); share: training share
    of each closure.vocab label (label_share). For a set S of incomparable allowed nodes,
        mass(S)  = P of the labels at or below one of its terms (each label once, also with several parents)
        worth(S) = d ** (|S| - 1) * sum_y p_y max_{t in S covering y} c_t(y),  c_t(y) = min(1, IC(t) / IC(y))
        value(S) = worth(S) - lam * (1 - mass(S));  answer = the set of highest value (first on ties), none if <= 0.
    Computed by inclusion-exclusion over the candidates' pairwise overlaps (labels under two incomparable terms exist
    only with several parents, so the overlaps are mostly 0):
        mass(a, b)  = q_a + q_b - P(labels under a and b)
        worth(a, b) = d * (r_a + r_b - sum_{y under a and b} p_y min(c_a(y), c_b(y))),  r_t = sum_y p_y c_t(y)
    (max(x, y) = x + y - min(x, y); min(c_a, c_b) is the credit of the less informative of the two); triples
    likewise, the triple overlap computed only where every pair of the triple shares a label."""

    def __init__(self, closure, share, max_terms=3, discount=0.7, n_candidates=10, min_mass=0.02, batch=2000):
        self.closure, self.share, self.max_terms, self.discount = closure, share, max_terms, discount
        self.n_candidates, self.min_mass, self.batch = n_candidates, min_mass, batch
        if max_terms > 3:
            raise ValueError("SetDecoder handles up to 3 terms per answer")
        A = closure.A > 0                                      # labels x nodes
        n_nodes = A.shape[1]
        self.pad = n_nodes                                     # index of an all-zero padding node
        ic_label = -np.log(share)
        self.node_ic = np.append(-np.log(A.T.astype(float) @ share), np.inf)
        At = np.vstack([A.T, np.zeros((1, A.shape[0]), bool)]).astype(float)  # (nodes + pad) x labels
        self.At = At
        self.credit = At * np.minimum(1, self.node_ic[:, None] / ic_label[None, :])  # c_t(y), 0 off t's labels
        self.credit[self.pad] = 0.0
        inter = A.T.astype(int) @ A.astype(int)                # labels under both nodes
        n_below = np.diag(inter)
        comparable = (inter == n_below[:, None]) | (inter == n_below[None, :])
        self.comparable = np.ones((n_nodes + 1, n_nodes + 1), bool)  # the padding node is comparable to everything
        self.comparable[:n_nodes, :n_nodes] = comparable
        self.overlap = np.zeros((n_nodes + 1, n_nodes + 1), bool)
        self.overlap[:n_nodes, :n_nodes] = inter > 0
        self.allowed = np.asarray(closure.informative, bool)
        # templates: positions among the candidates, in the order of candidate_sets (size, then lexicographic)
        k = n_candidates
        self.templates = [s for m in range(1, max_terms + 1) for s in itertools.combinations(range(k), m)]
        self.size = np.array([len(s) for s in self.templates])
        self.pos = np.array([list(s) + [-1] * (3 - len(s)) for s in self.templates]).reshape(-1, 3)

    # ------------------------------------------------------------------ tables
    def tables(self, P):
        """-> (candidate node indices: samples x n_candidates, pad = no candidate; mass and worth: samples x
        templates, -inf where the template is not a valid set for that sample)."""
        P = np.asarray(P, float)
        Q = P @ self.closure.A
        key = np.where(self.allowed[None, :] & (Q >= self.min_mass), Q, -np.inf)
        order = np.argsort(-key, axis=1, kind="stable")[:, :self.n_candidates]
        cand = np.where(np.take_along_axis(key, order, 1) > -np.inf, order, self.pad)
        if cand.shape[1] < self.n_candidates:  # fewer nodes than candidates
            cand = np.hstack([cand, np.full((len(P), self.n_candidates - cand.shape[1]), self.pad)])
        M = self.At[cand]                                      # samples x K x labels
        PM = M * P[:, None, :]
        W = self.credit[cand] * P[:, None, :]
        q, r = PM.sum(2), W.sum(2)
        I = PM @ M.transpose(0, 2, 1)                          # I[a, b] = P(labels under a and b)
        O = W @ M.transpose(0, 2, 1)                           # O[a, b] = sum_{y under a and b} p_y c_a(y)
        ic = self.node_ic[cand]
        n, T = len(P), len(self.templates)
        mass, worth = np.zeros((n, T)), np.zeros((n, T))
        valid = np.ones((n, T), bool)
        rows = np.arange(n)
        for t, s in enumerate(self.templates):
            nodes = cand[:, list(s)]
            valid[:, t] = (nodes != self.pad).all(1)
            mass[:, t] = q[:, list(s)].sum(1)
            worth[:, t] = r[:, list(s)].sum(1)
            for a, b in itertools.combinations(s, 2):
                valid[:, t] &= ~self.comparable[cand[:, a], cand[:, b]]
                mass[:, t] -= I[rows, a, b]
                low = ic[:, a] <= ic[:, b]                     # the less informative term gives the min credit
                worth[:, t] -= np.where(low, O[rows, a, b], O[rows, b, a])
            if len(s) == 3:
                a, b, c = s
                need = valid[:, t] & self.overlap[cand[:, a], cand[:, b]] & self.overlap[cand[:, a], cand[:, c]] \
                    & self.overlap[cand[:, b], cand[:, c]]
                for i in np.where(need)[0]:                    # a label under three incomparable terms: rare
                    under = M[i, a] * M[i, b] * M[i, c]
                    lowest = s[int(np.argmin(ic[i, list(s)]))]
                    mass[i, t] += float(P[i] @ under)
                    worth[i, t] += float((W[i, lowest] * under).sum())
            worth[:, t] *= self.discount ** (len(s) - 1)
        mass[~valid], worth[~valid] = 0.0, -np.inf
        return cand, mass, worth

    def picks(self, mass, worth, lam):
        """Template index per sample at this lambda, -1 = no answer (best value <= 0)."""
        value = worth - lam * (1 - mass)
        k = value.argmax(1)
        return np.where(value[np.arange(len(k)), k] > 0, k, -1)

    def answer_nodes(self, cand, k):
        """samples x 3 node indices of the picked sets (-1 beyond the set's size or for no answer)."""
        pos = self.pos[np.maximum(k, 0)]
        nodes = np.where(pos >= 0, np.take_along_axis(cand, np.maximum(pos, 0), 1), -1)
        return np.where((k >= 0)[:, None], nodes, -1)

    def decode(self, P, lam):
        """Answers at one lambda, in batches. -> (terms: samples x 3 node indices ordered by summed probability, most
        probable first (the primary answer), ties by id, -1 = none; their summed probabilities (0 for none); the
        summed probability of the whole answer, each label once)."""
        P = np.asarray(P, float)
        terms, masses, total = np.full((len(P), 3), -1), np.zeros((len(P), 3)), np.zeros(len(P))
        for s0 in range(0, len(P), self.batch):
            b = slice(s0, s0 + self.batch)
            cand, mass, worth = self.tables(P[b])
            k = self.picks(mass, worth, lam)
            nodes = self.answer_nodes(cand, k)
            Q = P[b] @ self.closure.A
            m = np.where(nodes >= 0, np.take_along_axis(Q, np.maximum(nodes, 0), 1), -np.inf)
            order = np.lexsort((np.where(nodes >= 0, nodes, np.iinfo(int).max), -m), axis=1)  # -mass, then id
            terms[b] = np.take_along_axis(nodes, order, 1)
            masses[b] = np.maximum(np.take_along_axis(m, order, 1), 0)
            total[b] = np.where(k >= 0, mass[np.arange(len(k)), np.maximum(k, 0)], 0.0)
        return terms, masses, total

    def runner_up(self, P, top=5):
        """unrelated_runner_up for every row (allowed = closure.informative): (label column per row, -1 = none;
        its probability)."""
        A = self.closure.A > 0
        lab = self.closure.label_col
        related = (A & self.allowed[None, :]).astype(int) @ A.T.astype(int) > 0  # share an allowed ancestor (or self)
        related |= A[:, lab] | A[:, lab].T                     # one label is an ancestor of the other
        out, p_out = np.full(len(P), -1), np.zeros(len(P))
        order = np.argsort(-np.asarray(P), axis=1, kind="stable")[:, :top + 1]
        first = order[:, 0]
        for j in range(1, order.shape[1]):
            v = order[:, j]
            take = (out < 0) & ~related[first, v]
            out[take] = v[take]
        rows = np.arange(len(P))
        p_out = np.where(out >= 0, np.asarray(P)[rows, np.maximum(out, 0)], 0.0)
        return out, p_out


def choose_lambda(curve, target, metric="accuracy_strict"):
    """The lambda answering most samples with accuracy >= target (ties: the first) from the set curve
    [{"lambda", "answered", metric, ...}] in grid order; the last lambda if none reaches it (as cross_fit)."""
    ok = [r for r in curve if r[metric] is not None and r[metric] >= target and r["answered"] > 0]
    if not ok:
        return curve[-1]["lambda"]
    return max(ok, key=lambda r: (r["answered"], -curve.index(r)))["lambda"]


def set_outcomes(decoder, cand, mass, worth, gold, anc, lambdas=LAMBDAS):
    """Per sample and lambda: (answered, covered strict, covered lenient, exact, set) as samples x lambdas arrays.
    covered strict: a term of the set is gold or an ancestor of it; lenient: also a descendant of gold."""
    nodes = decoder.closure.nodes
    rel = relation(np.append(nodes, ""), gold, anc)            # (samples x nodes + pad) outcome codes
    rel[:, -1] = 0
    code = np.take_along_axis(rel, cand, 1)                    # samples x candidates
    shape = (len(gold), len(lambdas))
    out = {k: np.zeros(shape, int) for k in ("answered", "strict", "lenient", "exact", "set")}
    for j, lam in enumerate(lambdas):
        k = decoder.picks(mass, worth, lam)
        pos = decoder.pos[np.maximum(k, 0)]
        c = np.where(pos >= 0, np.take_along_axis(code, np.maximum(pos, 0), 1), 0)
        has = k >= 0
        out["answered"][:, j] = has
        out["strict"][:, j] = has & np.isin(c, (EXACT, COARSER)).any(1)
        out["lenient"][:, j] = has & np.isin(c, (EXACT, COARSER, TOO_SPECIFIC)).any(1)
        size = np.where(has, decoder.size[np.maximum(k, 0)], 0)
        out["exact"][:, j] = (size == 1) & (c[:, 0] == EXACT)
        out["set"][:, j] = size > 1
    return out


def set_curve(out):
    """Pooled shares per lambda of set_outcomes arrays (concatenated over folds): answered, covered strict / lenient,
    exact, sets; accuracies among answered."""
    curve = []
    for j, lam in enumerate(LAMBDAS[:out["answered"].shape[1]]):
        row = {"lambda": float(lam), **{k: round(float(out[k][:, j].mean()), 4) for k in out}}
        curve.append(_set_accuracies(row))
    return curve


def _set_accuracies(row):
    n = row["answered"]
    row["accuracy_strict"] = round(row["strict"] / n, 4) if n > 0 else None
    row["accuracy_lenient"] = round(row["lenient"] / n, 4) if n > 0 else None
    return row


def oof_sets(fitted, folds, anc, target, metric="strict", discount=0.7, max_terms=3, n_candidates=10, min_mass=0.02,
             flag_top=5):
    """Set answers on out-of-fold probabilities (5_evaluate.py): fitted = oof_backoff's per-fold (P, Closure), folds
    its inputs (gold per fold). The label shares of fold k are those of the other folds' gold (= fold k's training
    labels); lambda per fold is cross-fitted (the most answered at >= target accuracy on the other folds).
    -> (report, curve pooled over folds, per fold {terms, masses, set_mass, nodes, runner_up, runner_up_p,
    label_counts})."""
    per_fold, outs = [], []
    for k, ((P, C), f) in enumerate(zip(fitted, folds)):
        counts = Counter(g for j, other in enumerate(folds) if j != k for g in other["gold"])
        dec = SetDecoder(C, label_share(C.vocab, counts), max_terms, discount, n_candidates, min_mass)
        cand, mass, worth = [], [], []
        for s0 in range(0, len(P), dec.batch):
            c, m, w = dec.tables(P[s0:s0 + dec.batch])
            cand.append(c), mass.append(m), worth.append(w)
        cand, mass, worth = np.vstack(cand), np.vstack(mass), np.vstack(worth)
        outs.append(set_outcomes(dec, cand, mass, worth, f["gold"], anc))
        per_fold.append({"decoder": dec, "P": P, "label_counts": counts})
    fold = np.concatenate([np.full(len(f["gold"]), k) for k, f in enumerate(folds)])
    pooled = {key: np.vstack([o[key] for o in outs]) for key in outs[0]}
    g, chosen = cross_fit(pooled["answered"], pooled[metric], fold, target)
    curve = set_curve(pooled)
    at = {key: pooled[key][np.arange(len(fold)), g] for key in pooled}
    report = {"lambda_per_fold": [float(LAMBDAS[chosen[k]]) for k in sorted(chosen)],
              "lambda_all": choose_lambda(curve, target, f"accuracy_{metric}"),
              "settings": {"discount": discount, "max_terms": max_terms, "n_candidates": n_candidates,
                           "min_mass": min_mass, "target": target, "metric": metric, "flag_top": flag_top},
              **_set_accuracies({key: round(float(v.mean()), 4) for key, v in at.items()})}
    for k, pf in enumerate(per_fold):
        dec, P = pf.pop("decoder"), pf.pop("P")
        terms, masses, total = dec.decode(P, LAMBDAS[chosen.get(k, len(LAMBDAS) - 1)])
        ru, ru_p = dec.runner_up(P, flag_top)
        pf.update(nodes=dec.closure.nodes, terms=terms, masses=masses, set_mass=total,
                  runner_up=np.where(ru >= 0, np.asarray(dec.closure.vocab, dtype=object)[np.maximum(ru, 0)], ""),
                  runner_up_p=ru_p)
    return report, curve, per_fold


def answer_columns(terms, masses, set_mass, nodes, runner_up, runner_up_p, flag_mass):
    """Output columns of the set answers: answer ('||'-joined terms, primary first), answer_masses (same order),
    answer_mass (whole answer), answer_kind (single / set / none), runner_up, runner_up_p, ambiguous_branches
    (runner-up probability >= flag_mass)."""
    nodes = np.append(np.asarray(nodes, dtype=object), "")     # index -1 -> ""
    n_terms = (terms >= 0).sum(1)
    return {"answer": ["||".join(nodes[t] for t in row if t >= 0) for row in terms],
            "answer_masses": ["||".join(f"{m:.4f}" for t, m in zip(row, mr) if t >= 0) for row, mr in zip(terms, masses)],
            "answer_mass": np.round(set_mass, 4),
            "answer_kind": np.where(n_terms == 0, "none", np.where(n_terms == 1, "single", "set")),
            "runner_up": runner_up, "runner_up_p": np.round(runner_up_p, 4),
            "ambiguous_branches": np.asarray(runner_up_p) >= flag_mass}


def merge_set_calibrations(entries):
    """Pool the "sets" part of several calibration.json entries (fold seeds of the same samples): the shares per
    lambda are averaged and the accuracies recomputed; settings and label counts must agree. None if any entry has
    no "sets" (calibration.json from before the set answers)."""
    sets = [e.get("sets") for e in entries]
    if any(s is None for s in sets):
        return None
    if any(s["settings"] != sets[0]["settings"] for s in sets) or any(s["label_counts"] != sets[0]["label_counts"]
                                                                       for s in sets):
        raise SystemExit("calibrations to merge have different set settings or label counts")
    if len(sets) == 1:
        return sets[0]
    curve = []
    for rows in zip(*(s["curve"] for s in sets)):
        if any(r["lambda"] != rows[0]["lambda"] for r in rows):
            raise SystemExit("calibrations to merge have different lambda grids")
        mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("answered", "strict", "lenient", "exact", "set")}
        curve.append(_set_accuracies({"lambda": rows[0]["lambda"], **mean}))
    return {**sets[0], "curve": curve, "lambda_per_fold": [s["lambda_per_fold"] for s in sets]}
