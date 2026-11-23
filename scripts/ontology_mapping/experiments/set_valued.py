#!/usr/bin/env python3
"""
More than one answer: set-valued decoding over the ontology DAG (project doc claude/set-valued-answers.md;
experiments README §21).

The production back-off returns ONE node per slot (the most specific allowed node on the top-1 label's path with
summed probability q >= tau) or abstains. When the probability is split between unrelated branches (host-associated
vs terrestrial, waste water vs wastewater treatment plant) it can only abstain. This script tests answering with a
small set of terms instead, and better single-node decoding, on the same out-of-fold probabilities:

  top1            the top-1 label (answers everything)
  backoff         hierarchy.Closure.decode (production rule), tau chosen on the other folds
  conformal_m<k>  the smallest set of <= k incomparable nodes whose summed probability reaches tau (fewest labels
                  covered, then more mass): Zhang, Li & Bastani, "Conformal Structured Prediction", ICLR 2025
                  (tau cross-fitted here, not split-conformal; same decision rule)
  ev_m<k>         expected value: the set of <= k nodes maximising E[credit] - lambda * P(wrong), abstain if <= 0;
                  credit = min(1, IC(set) / IC(gold)). Bayes-optimal set-valued prediction under that utility
                  (Mortier et al., DAMI 2021; Mortier, Hüllermeier, Dembczyński & Waegeman 2022)
  backoff+set     back-off; where it abstains, a conformal_m2 set (its own tau, chosen on the other folds'
                  abstentions); backoff+set@0.8 the same with an 80 % target for the sets
  ev1+ev2set      ev_m1; where it abstains, an ev_m2 set (its own lambda)
  evr_d<d>        expected value with the reviewer score: a set of <= --max_terms nodes is worth the credit of its
                  best term covering gold, min(1, IC(term) / IC(gold)), times d for each term beyond the first
                  (--discounts); d = 1 does not penalise sets, d -> 0 approaches ev_m1
  backoff@onto,   with --answers ontology: back-off and back-off + set where ENVO / UBERON terms that no curator used
  backoff+set@onto  can also be answers (hierarchy.Closure(answers="ontology")); the other methods keep the labels

Every threshold (tau, lambda) is chosen on the other folds as the one answering most samples at >= --target accuracy
among answered (accuracy = one term of the set is the gold label or an ancestor of it), then applied to the held-out
fold.

Probabilities: the production prototype (alpha 0.5, beta 0.1) on keyword + sub-biome vectors, closed vocabulary (the
training fold's labels), softmax with a temperature fitted on the other folds (hierarchy.fit_temperature), the same
as 5_evaluate.py / hierarchy.oof_backoff. --study_pool w mixes each probability row with its study's mean row
((1 - w) P + w mean; test-fold study, no labels used).

Candidate sets per sample: antichains (no term an ancestor of another) of <= --max_terms nodes among the
--n_candidates allowed nodes (hierarchy.Closure.informative) with the most mass, each with q >= --min_mass.
Summed probability of a set = P of the labels at or below one of its nodes, each label counted once (DAG-safe).

Scores, all over every sample with a gold label (abstain = 0):
  summary.tsv      answered, accuracy among answered, exact (a single term equal to gold), share of sets, mean labels
                   covered, set credit = min(1, IC(set) / IC(gold)) if covered (IC from the training fold's label
                   counts, IC(set) = -log share of the labels it covers), and the credit difference to back-off with
                   a 95 % study bootstrap interval
  rule_scores.tsv  the acceptance rules and graded credits of acceptable_answers.py (README §15), micro / macro /
                   macro over learnable labels, for each method and two set conventions: 'any' (a set scores if one
                   of its terms does: a reviewer picks the right one) and 'split' (mean over its terms: a random pick)
  decisions.tsv.gz one row per sample, slot and method: the answer terms ('||'-joined, '' = abstain), each term's
                   summed probability ('masses', same order) and the set's summed probability ('set_mass'; a label
                   under two of the terms counted once). Terms are ordered by mass: the first is the primary
                   answer, the rest are the alternatives
  flags.tsv.gz     one row per sample and slot: top-1 label and probability, the unrelated runner-up (the most
                   probable of the next --flag_top labels that is neither an ancestor nor a descendant of the top-1
                   and shares no allowed ancestor with it) and its probability, ambiguous_branches (runner-up
                   probability >= --flag_mass), the back-off answer, gold and whether gold was a training label
  flags_summary.tsv per slot: share flagged, among all samples and among back-off abstentions, and how often the
                   top-1, the runner-up or either is right when flagged
  settings.json    the options, temperatures and the threshold chosen per fold

Example (~15 min, most of it the per-sample candidate sets and the rule scoring):
python experiments/set_valued.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --fold_groups ~/MicrobeAtlasProject/metalog/clean/project_groups.tsv \
  --envo_obo ~/MicrobeAtlasProject/ontologies/envo.obo --answers ontology \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/set_valued/seed0
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from _setup import DEFAULTS, SLOTS, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds
import methods as M  # noqa: E402  (importable once _setup has put scripts/ontology_mapping on the path)
from common import ancestor_distances, read_tsv, term_ancestors  # noqa: E402
from hierarchy import (LAMBDAS, TAUS, Closure, candidate_sets, cross_fit, fit_temperature,  # noqa: E402,F401
                       reviewer_value_pick, set_table, softmax, unrelated_runner_up)
from acceptable_answers import (GENERIC_HEADS, direct_relation, envo_relations, isa_relation,  # noqa: E402
                                near_duplicate_pairs)

CONFORMAL_TAUS = np.round(np.arange(0.3, 1.0001, 0.025), 3)
RULES = ["exact", "broader", "narrower", "synonyms", "co-labels", "facets", "synonyms + co-labels", "all"]
SCHEMES = ["exact", "depth 0.5^k", "IC ratio", "full credit", "hierarchical F"]


# ----------------------------------------------------------------------------- 1. out-of-fold probabilities
def fold_groups(samples, groups_path=None):
    """Fold grouping as 5_evaluate.py --fold_groups: the project group of a sample, else its study code."""
    if not groups_path:
        return samples["study_code"]
    project = dict(read_tsv(groups_path)[["sample_id", "project_group"]].values)
    return samples["sample_id"].map(project).fillna(samples["study_code"])


def study_pool(P, studies, weight):
    """(1 - weight) * P + weight * (mean row of the sample's study). weight 0 returns P unchanged."""
    if not weight:
        return P
    means = pd.DataFrame(P).groupby(np.asarray(studies)).transform("mean").to_numpy()
    return (1 - weight) * P + weight * means


def oof_probabilities(samples, x, vector_of, folds, alpha=0.5, beta=0.1, pool=0.0):
    """{slot: [per test fold: {P, vocab, gold, sample_id, study, T}]}: prototype scores of the test samples over the
    training fold's labels, softmax with a temperature fitted on the other folds' scores (as hierarchy.oof_backoff).
    vector_of: {term_id: term vector in the sample space}."""
    out = {}
    for slot in SLOTS:
        y_all = samples[slot].to_numpy()
        blocks, folds_out = [], []
        for train, test in folds:
            a, b = train[y_all[train] != ""], test[y_all[test] != ""]
            vocab = np.unique(y_all[a])
            if len(vocab):
                model = M.prototype_model(x[a], y_all[a], np.vstack([vector_of[t] for t in vocab]), vocab, alpha, beta)
                S = M.prototype_scores(x[b], model)
            else:  # no labelled training sample in this slot: nothing to predict (its test samples are skipped)
                S, b = np.zeros((0, 0)), b[:0]
            col = {v: i for i, v in enumerate(vocab)}
            blocks.append((S, np.array([col.get(g, -1) for g in y_all[b]])))
            folds_out.append({"S": S, "vocab": vocab, "gold": y_all[b], "rows": b,
                              "sample_id": samples["sample_id"].to_numpy()[b],
                              "study": samples["study_code"].to_numpy()[b]})
        for k, f in enumerate(folds_out):
            f["T"] = fit_temperature([blk for j, blk in enumerate(blocks) if j != k])
            S = f.pop("S")
            f["P"] = study_pool(softmax(S, f["T"]), f["study"], pool) if S.size else S
        out[slot] = folds_out
    return out


# ----------------------------------------------------------------------------- 2-3. candidate sets, decision rules
# candidate_sets, set_table, reviewer_value_pick, unrelated_runner_up and cross_fit live in hierarchy.py (shared with
# 5_evaluate.py / 6_predict_atlas.py, whose vectorised hierarchy.SetDecoder gives the same evr decisions)
def conformal_pick(table, tau, max_terms):
    """Index of the smallest set (fewest labels covered, then most mass, then first) with <= max_terms terms and mass
    >= tau; -1 if none."""
    ok = (table["mass"] >= tau - 1e-12) & (table["n_terms"] <= max_terms)
    if not ok.any():
        return -1
    idx = np.where(ok)[0]
    return int(idx[np.lexsort((-table["mass"][idx], table["n_labels"][idx]))[0]])


def expected_value_pick(table, lam, max_terms):
    """Index of the set with <= max_terms terms maximising E[credit] - lam * (1 - mass); -1 (abstain) if that is <= 0."""
    value = np.where(table["n_terms"] <= max_terms, table["expected_credit"] - lam * (1 - table["mass"]), -np.inf)
    k = int(value.argmax())
    return k if value[k] > 0 else -1


# ----------------------------------------------------------------------------- 4. thresholds chosen on other folds
def fallback(primary_answered, answered, covered, fold, target):
    """Keep the primary decision; where it abstains, use the grid index chosen on the other folds' ABSTENTIONS only
    (most answered at >= target). -> (grid index per sample, -1 = keep the primary decision; {fold: index or None})."""
    empty = np.asarray(primary_answered) == 0
    per_sample, chosen = np.full(len(fold), -1), {}
    for k in np.unique(fold):
        o = (fold != k) & empty
        n_ans = answered[o].sum(0)
        acc = covered[o].sum(0) / np.maximum(n_ans, 1)
        ok = np.where((acc >= target) & (n_ans > 0))[0]
        chosen[int(k)] = int(ok[n_ans[ok].argmax()]) if len(ok) else None
        if chosen[int(k)] is not None:
            per_sample[(fold == k) & empty] = chosen[int(k)]
    return per_sample, chosen


# ----------------------------------------------------------------------------- 5. scoring
def covers(term, gold, anc):
    """The answer term is the gold label or an is_a ancestor of it."""
    return term == gold or term in anc.get(gold, ())


def bootstrap_ci(diff, groups, n=500, seed=0):
    """95 % interval of mean(diff) resampling whole groups (studies) with replacement."""
    groups = np.asarray(groups)
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    rng, means = np.random.default_rng(seed), []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        means.append(diff[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class RuleScorer:
    """acceptable_answers.py's rules and graded credits for decisions that are lists of terms ([] = abstain).
    Built from the same inputs: synonym candidates among the labels in use, out-of-fold curator co-labels from
    cross-study near-duplicates, direct ENVO relations, IC from the slot's labelled samples."""

    def __init__(self, samples, x, fold, anc, dist, label, vector_of, relations, dup=0.9, syn_cosine=0.88,
                 min_study_pairs=2):
        self.anc, self.dist, self.relations = anc, dist, relations
        self.fold_of = dict(zip(samples["sample_id"], fold))
        studies = samples["study_code"].to_numpy()
        used = sorted(set().union(*[set(samples[s]) for s in SLOTS]) - {""})
        norm = {t: GENERIC_HEADS.sub("", label[t].lower()).strip() for t in used}
        self.synonyms = {(a, b) for i, a in enumerate(used) for b in used[i + 1:]
                         if float(vector_of[a] @ vector_of[b]) >= syn_cosine or (norm[a] == norm[b] and norm[a] != "")}
        I, J = near_duplicate_pairs(x, studies, dup)
        self.n_pairs, self.co_labels, self.slot = len(I), {}, {}
        for slot in SLOTS:
            g = samples[slot].to_numpy()
            differ = (g[I] != "") & (g[J] != "") & (g[I] != g[J])
            self.co_labels[slot] = {}
            for f in np.unique(fold):
                train = differ & (fold[I] != f) & (fold[J] != f)
                seen = defaultdict(set)
                for i, j in zip(I[train], J[train]):
                    seen[tuple(sorted((g[i], g[j])))].add(tuple(sorted((studies[i], studies[j]))))
                self.co_labels[slot][int(f)] = {k for k, s in seen.items() if len(s) >= min_study_pairs}
            counts = Counter()
            for t, n in samples.loc[samples[slot] != "", slot].value_counts().items():
                for a in {t} | anc.get(t, set()):
                    counts[a] += n
            n_studies = samples[samples[slot] != ""].groupby(slot)["study_code"].nunique()
            self.slot[slot] = {"counts": counts, "total": int((samples[slot] != "").sum()),
                               "learnable": set(n_studies[n_studies >= 2].index)}

    def term(self, slot, gold, term, fold):
        """({rule: 0/1}, {scheme: credit}) of one answer term."""
        anc, exact = self.anc, gold == term
        broader, narrower = term in anc.get(gold, ()), gold in anc.get(term, ())
        key = tuple(sorted((gold, term)))
        syn, co = exact or key in self.synonyms, exact or key in self.co_labels[slot][fold]
        facet = exact or (isa_relation(gold, term, anc, {}) in ("", "siblings")
                          and direct_relation(gold, term, self.relations) != "")
        rules = {"exact": exact, "broader": exact or broader, "narrower": exact or narrower, "synonyms": syn,
                 "co-labels": co, "facets": facet, "synonyms + co-labels": syn or co,
                 "all": broader or narrower or syn or co or facet}
        c = self.slot[slot]
        ic = lambda t: -np.log(max(c["counts"].get(t, 0), 0.5) / c["total"])  # noqa: E731
        up_a, up_b = {term} | anc.get(term, set()), {gold} | anc.get(gold, set())
        credit = {"exact": float(exact),
                  "depth 0.5^k": 1.0 if exact else (0.5 ** self.dist[gold][term] if broader else 0.0),
                  "IC ratio": 1.0 if exact else ((ic(term) / ic(gold) if ic(gold) > 0 else 0.0) if broader else 0.0),
                  "full credit": float(exact or broader),
                  "hierarchical F": 2 * len(up_a & up_b) / (len(up_a) + len(up_b))}
        return {k: float(v) for k, v in rules.items()}, credit

    def per_sample(self, slot, gold, answers, sample_ids):
        """{(convention, measure): per-sample values}; 'any' = max over the set's terms, 'split' = mean; [] -> 0."""
        out = {(conv, m): [] for conv in ("any", "split") for m in RULES + ["credit: " + s for s in SCHEMES]}
        for g, ans, sid in zip(gold, answers, sample_ids):
            scored = [self.term(slot, g, t, self.fold_of[sid]) for t in ans]
            for m in RULES:
                v = [r[m] for r, _ in scored]
                out[("any", m)].append(max(v) if v else 0.0)
                out[("split", m)].append(float(np.mean(v)) if v else 0.0)
            for m in SCHEMES:
                v = [s[m] for _, s in scored]
                out[("any", "credit: " + m)].append(max(v) if v else 0.0)
                out[("split", "credit: " + m)].append(float(np.mean(v)) if v else 0.0)
        return out

    def score(self, slot, gold, answers, sample_ids):
        """Rows (convention, measure, micro, macro, macro_learnable)."""
        gold, learnable, rows = np.asarray(gold), self.slot[slot]["learnable"], []
        for (conv, m), v in self.per_sample(slot, gold, answers, sample_ids).items():
            per_label = pd.Series(v).groupby(gold).mean()
            rows.append({"convention": conv, "measure": m, "micro": float(np.mean(v)), "macro": float(per_label.mean()),
                         "macro_learnable": float(per_label[per_label.index.isin(learnable)].mean())})
        return rows


# ----------------------------------------------------------------------------- 6. one slot, every method
def sample_rows(folds, anc, args, answers="labels", max_terms=None):
    """Per test sample: gold, top-1, back-off answer at every tau of TAUS, and the candidate sets with their table,
    coverage of gold, set credit and exactness (set credit uses the training fold's label counts). `answers` is
    hierarchy.Closure's answer mode ("labels" or "ontology"); max_terms overrides args.max_terms. Each row also keeps
    its probability row, the closure and the node columns, for the masses of any answer (answer_masses)."""
    rows = []
    max_terms = args.max_terms if max_terms is None else max_terms
    for k, f in enumerate(folds):
        P, vocab = f["P"], f["vocab"]
        if not len(P):
            continue
        C = Closure(vocab, anc, answers=answers)
        A, nodes, inf = C.A > 0, list(C.nodes), np.asarray(C.informative)
        counts = Counter(t for j, g in enumerate(folds) if j != k for t in g["gold"])  # fold k's training labels
        cnt = np.array([counts.get(v, 0) for v in vocab], float) + 0.5
        share = cnt / cnt.sum()
        ic_label, ic_unseen = dict(zip(vocab, -np.log(share))), -np.log(0.5 / cnt.sum())
        node_ic = -np.log(A.T.astype(float) @ share)  # IC of each single node
        node_labels = A.sum(0)
        Q = P @ C.A
        col = {n: j for j, n in enumerate(nodes)}
        allowed = set(np.asarray(nodes)[inf])
        bo = np.stack([C.decode(P, t)[0] for t in TAUS], 1)  # samples x taus, -1 = abstain
        for i in range(len(P)):
            gold = f["gold"][i]
            ic_gold = ic_label.get(gold, ic_unseen)
            r = {"fold": k, "gold": gold, "sample_id": f["sample_id"][i], "study": f["study"][i],
                 "top1": vocab[P[i].argmax()], "top1_p": float(P[i].max()), "p": P[i], "A": A, "col": col,
                 "gold_seen": bool(gold in ic_label)}
            r["runner_up"], r["runner_up_p"] = unrelated_runner_up(P[i], vocab, anc, allowed, args.flag_top)
            # back-off outcomes per tau: (answered, covered, set credit, exact, labels covered)
            r["backoff"] = [[] if n < 0 else [nodes[n]] for n in bo[i]]
            r["backoff_out"] = [(0, 0, 0.0, 0, 0) if n < 0 else
                                (1, int(covers(nodes[n], gold, anc)),
                                 min(1.0, node_ic[n] / ic_gold) if covers(nodes[n], gold, anc) else 0.0,
                                 int(nodes[n] == gold), int(node_labels[n])) for n in bo[i]]
            sets = candidate_sets(Q[i], A, inf, args.n_candidates, max_terms, args.min_mass)
            r["sets"] = [[nodes[n] for n in s] for s in sets]
            if sets:
                t = set_table(P[i], sets, A, share)
                t.pop("cover")  # sets x labels per sample: ~1.5 GB per slot if kept, and not needed after this
                ok = np.array([any(covers(n, gold, anc) for n in s) for s in r["sets"]])
                r.update(t, covered=ok, credit=np.where(ok, np.minimum(1, t["ic"] / ic_gold), 0.0),
                         exact=np.array([s == [gold] for s in r["sets"]]))
            rows.append(r)
    return rows


def answer_masses(r, terms):
    """(per-term summed probability, in the order given; summed probability of the whole answer, each label once)."""
    if not terms:
        return [], 0.0
    cols = [r["col"][t] for t in terms]
    per_term = [float(r["p"] @ r["A"][:, c]) for c in cols]
    return per_term, float(r["p"] @ r["A"][:, cols].any(1))


def by_mass(r, terms):
    """The answer's terms ordered by summed probability, most probable first (the primary answer), ties by id."""
    m, _ = answer_masses(r, terms)
    return [t for _, t in sorted(zip(m, terms), key=lambda x: (-x[0], x[1]))]


def set_outcome(r, k):
    """(answered, covered, set credit, exact, labels covered) of candidate set k of a sample row (-1 = abstain)."""
    if k < 0 or "mass" not in r:
        return (0, 0, 0.0, 0, 0)
    return (1, int(r["covered"][k]), float(r["credit"][k]), int(r["exact"][k]), int(r["n_labels"][k]))


def decode_slot(folds, anc, args):
    """Every method on one slot -> (rows, {method: answer-term lists}, {method: samples x 5 outcomes},
    {method: {fold: threshold}}, {method: row list the answers come from}). Answers are ordered by mass."""
    rows = sample_rows(folds, anc, args)
    fold = np.array([r["fold"] for r in rows])
    answers, outcomes, thresholds, grids = {}, {}, {}, {}
    answers["top1"] = [[r["top1"]] for r in rows]
    outcomes["top1"] = np.array([(1, int(covers(r["top1"], r["gold"], anc)), np.nan, int(r["top1"] == r["gold"]), 1)
                                 for r in rows])
    O = np.array([r["backoff_out"] for r in rows])  # samples x taus x 5
    g, chosen = cross_fit(O[:, :, 0], O[:, :, 1], fold, args.target)
    answers["backoff"] = [r["backoff"][gi] for r, gi in zip(rows, g)]
    outcomes["backoff"] = O[np.arange(len(rows)), g]
    thresholds["backoff"] = {k: float(TAUS[v]) for k, v in chosen.items()}

    def conf(m):
        return lambda r, t: conformal_pick(r, t, m) if "mass" in r else -1

    def ev(m):
        return lambda r, lam: expected_value_pick(r, lam, m) if "mass" in r else -1

    def answer(r, k):
        return [] if k < 0 else r["sets"][k]

    def run(name, pick, grid):
        picks = np.array([[pick(r, t) for t in grid] for r in rows])  # samples x grid
        O = np.array([[set_outcome(r, k) for k in p] for r, p in zip(rows, picks)])
        g, chosen = cross_fit(O[:, :, 0], O[:, :, 1], fold, args.target)
        answers[name] = [answer(r, p[gi]) for r, p, gi in zip(rows, picks, g)]
        outcomes[name] = O[np.arange(len(rows)), g]
        thresholds[name] = {k: float(grid[v]) for k, v in chosen.items()}
        grids[name] = (picks, O, grid)

    for m in range(1, args.max_terms + 1):
        run(f"conformal_m{m}", conf(m), CONFORMAL_TAUS)
        run(f"ev_m{m}", ev(m), LAMBDAS)

    def evr(d):
        return lambda r, lam: reviewer_value_pick(r, lam, args.max_terms, d) if "mass" in r else -1

    if args.max_terms >= 2:
        for d in args.discounts:
            run(f"evr_d{d:g}", evr(d), LAMBDAS)

    def run_fallback(name, primary, sets_from, target):
        picks, O, grid = grids[sets_from]
        g, chosen = fallback(outcomes[primary][:, 0], O[:, :, 0], O[:, :, 1], fold, target)
        answers[name] = [a if gi < 0 else answer(r, p[gi]) for r, a, p, gi in zip(rows, answers[primary], picks, g)]
        outcomes[name] = np.array([o if gi < 0 else O[i, gi] for i, (o, gi) in enumerate(zip(outcomes[primary], g))])
        thresholds[name] = {k: (None if v is None else float(grid[v])) for k, v in chosen.items()}

    if args.max_terms >= 2:
        run_fallback("backoff+set", "backoff", "conformal_m2", args.target)
        run_fallback("backoff+set@0.8", "backoff", "conformal_m2", 0.8)
        run_fallback("ev1+ev2set", "ev_m1", "ev_m2", args.target)
    source = {name: rows for name in answers}

    if args.answers == "ontology":  # back-off-based rules with ENVO / UBERON terms no curator used as answers too
        onto = sample_rows(folds, anc, args, answers="ontology", max_terms=min(2, args.max_terms))
        O = np.array([r["backoff_out"] for r in onto])
        g, chosen = cross_fit(O[:, :, 0], O[:, :, 1], fold, args.target)
        answers["backoff@onto"] = [r["backoff"][gi] for r, gi in zip(onto, g)]
        outcomes["backoff@onto"] = O[np.arange(len(onto)), g]
        thresholds["backoff@onto"] = {k: float(TAUS[v]) for k, v in chosen.items()}
        source["backoff@onto"] = onto
        if args.max_terms >= 2:
            picks = np.array([[conf(2)(r, t) for t in CONFORMAL_TAUS] for r in onto])
            Os = np.array([[set_outcome(r, k) for k in p] for r, p in zip(onto, picks)])
            g, chosen = fallback(outcomes["backoff@onto"][:, 0], Os[:, :, 0], Os[:, :, 1], fold, args.target)
            answers["backoff+set@onto"] = [a if gi < 0 else answer(r, p[gi])
                                           for r, a, p, gi in zip(onto, answers["backoff@onto"], picks, g)]
            outcomes["backoff+set@onto"] = np.array([o if gi < 0 else Os[i, gi]
                                                     for i, (o, gi) in enumerate(zip(outcomes["backoff@onto"], g))])
            thresholds["backoff+set@onto"] = {k: (None if v is None else float(CONFORMAL_TAUS[v]))
                                              for k, v in chosen.items()}
            source["backoff+set@onto"] = onto
    answers = {name: [by_mass(r, a) for r, a in zip(source[name], ans)] for name, ans in answers.items()}
    return rows, answers, outcomes, thresholds, source


def summarise(answers, outcomes, gold, studies, bootstrap=500):
    """summary rows of one slot (see the module docstring); set credit is not defined for top1 (NaN)."""
    rows, base = [], outcomes["backoff"][:, 2]
    for name, ans in answers.items():
        o = outcomes[name]
        answered = o[:, 0] == 1
        row = {"method": name, "answered": answered.mean(),
               "accuracy": o[answered, 1].mean() if answered.any() else np.nan,
               "exact": np.mean([a == [g] for a, g in zip(ans, gold)]),
               "sets": np.mean([len(a) > 1 for a in ans]),
               "mean_labels_covered": o[answered, 4].mean() if answered.any() else np.nan,
               "set_credit": np.nan if name == "top1" else o[:, 2].mean()}
        if name not in ("top1", "backoff") and bootstrap:
            row["credit_diff_lo"], row["credit_diff_hi"] = bootstrap_ci(o[:, 2] - base, studies, bootstrap)
        rows.append(row)
    return rows


def flag_table(rows, backoff_answers, anc, flag_mass):
    """flags.tsv rows of one slot (see the module docstring)."""
    return pd.DataFrame({
        "sample_id": [r["sample_id"] for r in rows], "study_code": [r["study"] for r in rows],
        "fold": [r["fold"] for r in rows], "gold": [r["gold"] for r in rows],
        "gold_seen": [r["gold_seen"] for r in rows],
        "top1": [r["top1"] for r in rows], "top1_p": [round(r["top1_p"], 4) for r in rows],
        "runner_up": [r["runner_up"] for r in rows], "runner_up_p": [round(r["runner_up_p"], 4) for r in rows],
        "ambiguous_branches": [r["runner_up_p"] >= flag_mass for r in rows],
        "backoff": ["||".join(a) for a in backoff_answers]})


def flag_summary(flags, anc):
    """Share flagged (all / among back-off abstentions) and, when flagged, how often top-1 / runner-up / either covers
    gold (is gold or an ancestor of it)."""
    fl = flags["ambiguous_branches"].to_numpy()
    abst = (flags["backoff"] == "").to_numpy()
    top_ok = np.array([covers(t, g, anc) for t, g in zip(flags["top1"], flags["gold"])])
    run_ok = np.array([bool(t) and covers(t, g, anc) for t, g in zip(flags["runner_up"], flags["gold"])])
    rate = lambda m: float(m.mean()) if len(m) else np.nan  # noqa: E731
    return {"flagged": rate(fl), "flagged_among_abstentions": rate(fl[abst]), "abstentions": rate(abst),
            "top1_right_if_flagged": rate(top_ok[fl]), "runner_up_right_if_flagged": rate(run_ok[fl]),
            "either_right_if_flagged": rate((top_ok | run_ok)[fl])}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ["ontology_terms", "keywords", "sub_biomes", "term_vectors"]:
        ap.add_argument(f"--{name}", default=DEFAULTS[name])
    ap.add_argument("--samples", default="~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz")
    ap.add_argument("--fold_groups", default=None, help="project_groups.tsv (as 5_evaluate.py --fold_groups)")
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_per_study", type=int, default=50)
    ap.add_argument("--envo_obo", default="~/MicrobeAtlasProject/ontologies/envo.obo")
    ap.add_argument("--target", type=float, default=0.9, help="Accuracy among answered that every threshold must reach")
    ap.add_argument("--max_terms", type=int, default=3)
    ap.add_argument("--n_candidates", type=int, default=10)
    ap.add_argument("--min_mass", type=float, default=0.02)
    ap.add_argument("--study_pool", type=float, default=0.0)
    ap.add_argument("--discounts", type=float, nargs="*", default=[0.7, 0.8, 0.9, 1.0],
                    help="Set discounts d of the reviewer-score expected value (one evr_d<d> method each)")
    ap.add_argument("--answers", choices=["labels", "ontology"], default="labels",
                    help="ontology: also run back-off and back-off + set with unused ENVO / UBERON terms as answers")
    ap.add_argument("--flag_mass", type=float, default=0.2, help="Runner-up probability that flags ambiguous branches")
    ap.add_argument("--flag_top", type=int, default=5, help="Labels after the top-1 searched for the runner-up")
    ap.add_argument("--bootstrap", type=int, default=500, help="Study bootstrap replicates (0 = none)")
    ap.add_argument("--no_rules", action="store_true", help="Skip rule_scores.tsv (no OBO needed)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)
    out_dir = path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    (kw_row, kw), (sb_row, sb) = load_npz(args.keywords), load_npz(args.sub_biomes)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)], args.max_per_study)
    w = 1 / np.sqrt(2)
    x = np.hstack([w * kw[[kw_row[s] for s in samples["sample_id"]]],
                   w * sb[[sb_row[s] for s in samples["sample_id"]]]]).astype(np.float32)
    terms = load_terms(args.ontology_terms)
    text, label = dict(zip(terms["term_id"], terms["text"])), dict(zip(terms["term_id"], terms["label"]))
    anc = term_ancestors(terms)
    used = sorted(set().union(*[set(samples[s]) for s in SLOTS]) - {""})
    raw = dict(zip(used, load_term_vectors(args.term_vectors, [text[t] for t in used])))
    vector_of = {t: np.concatenate([w * v, w * v]) for t, v in raw.items()}  # term side [t, t] / sqrt(2)
    groups = fold_groups(samples, args.fold_groups)
    folds = list(study_folds(groups, args.folds, args.fold_seed))
    fold = np.zeros(len(samples), int)
    for k, (_, test) in enumerate(folds):
        fold[test] = k
    print(f"{len(samples)} samples, {groups.nunique()} fold groups, fold seed {args.fold_seed}", flush=True)

    probs = oof_probabilities(samples, x, vector_of, folds, pool=args.study_pool)
    scorer = None
    if not args.no_rules:
        relations = envo_relations(args.envo_obo)
        scorer = RuleScorer(samples, x, fold, anc, ancestor_distances(terms), label,
                            {t: v / np.linalg.norm(v) for t, v in raw.items()}, relations)
        print(f"scorer: {len(scorer.synonyms)} synonym candidates, {scorer.n_pairs} near-duplicate pairs", flush=True)

    summary, rules, decisions, flags, flag_rows = [], [], [], [], []
    settings = {"args": vars(args), "temperatures": {}, "thresholds": {}}
    for slot in SLOTS:
        rows, answers, outcomes, thresholds, source = decode_slot(probs[slot], anc, args)
        gold = np.array([r["gold"] for r in rows])
        ids, studies = np.array([r["sample_id"] for r in rows]), np.array([r["study"] for r in rows])
        f_of = np.array([r["fold"] for r in rows])
        for row in summarise(answers, outcomes, gold, studies, args.bootstrap):
            summary.append({"slot": slot, **row})
        if scorer:
            for name, ans in answers.items():
                for row in scorer.score(slot, gold, ans, ids):
                    rules.append({"slot": slot, "method": name, **row})
        for name, ans in answers.items():
            masses = [answer_masses(r, a) for r, a in zip(source[name], ans)]
            decisions.append(pd.DataFrame({"sample_id": ids, "study_code": studies, "slot": slot, "fold": f_of,
                                           "gold": gold, "method": name, "answer": ["||".join(a) for a in ans],
                                           "masses": ["||".join(f"{v:.4f}" for v in m) for m, _ in masses],
                                           "set_mass": [round(u, 4) for _, u in masses]}))
        ft = flag_table(rows, answers["backoff"], anc, args.flag_mass)
        flags.append(ft.assign(slot=slot))
        flag_rows.append({"slot": slot, **flag_summary(ft, anc)})
        settings["temperatures"][slot] = [round(f["T"], 4) for f in probs[slot]]
        settings["thresholds"][slot] = thresholds
        print(pd.DataFrame([r for r in summary if r["slot"] == slot]).round(3).to_string(index=False), flush=True)

    pd.DataFrame(summary).round(4).to_csv(os.path.join(out_dir, "summary.tsv"), sep="\t", index=False)
    if rules:
        pd.DataFrame(rules).round(4).to_csv(os.path.join(out_dir, "rule_scores.tsv"), sep="\t", index=False)
    pd.concat(decisions).to_csv(os.path.join(out_dir, "decisions.tsv.gz"), sep="\t", index=False)
    flag_cols = ["sample_id", "study_code", "slot", "fold", "gold", "gold_seen", "top1", "top1_p", "runner_up",
                 "runner_up_p", "ambiguous_branches", "backoff"]
    pd.concat(flags)[flag_cols].to_csv(os.path.join(out_dir, "flags.tsv.gz"), sep="\t", index=False)
    pd.DataFrame(flag_rows).round(4).to_csv(os.path.join(out_dir, "flags_summary.tsv"), sep="\t", index=False)
    print(pd.DataFrame(flag_rows).round(3).to_string(index=False))
    with open(os.path.join(out_dir, "settings.json"), "w") as handle:
        json.dump(settings, handle, indent=1, default=str)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
