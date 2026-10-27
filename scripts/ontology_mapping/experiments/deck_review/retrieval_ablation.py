#!/usr/bin/env python3
"""
Zero-shot retrieval: what centring and the label prior each contribute, what happens over the whole
ontology, and how much restricting each slot to an allowed subset of the ontology helps
(deck slide "retrieval_prior"; report §4.6 and §8.8).

Score of term c for sample x (all vectors kw + sb, 2 x 1024, unit norm; terms as [t, t] / sqrt(2)):
    raw:          cos(x, t_c)
    centre:       cos(x - mu_x, t_c - mu_t)      mu_x = training-sample mean, mu_t = mean of the candidate terms
    prior:        + 0.1 * log((n_c + 0.5) / sum)  n_c = training count of c (the prototype's beta term)
Vocabularies: closed = the slot's training labels; open = every live term with a vector (49k);
subsets = MIxS-style allowed classes per slot (no Metalog labels used, so they apply to any sample):
    biome     ENVO biome (ENVO:00000428) descendants + animal- / plant-associated environment
    feature   ENVO environmental system (ENVO:01000254) or astronomical body part (ENVO:01000813),
              minus materials and biomes (+ variant with UBERON anatomical structure and PO anatomy)
    material  ENVO environmental material (ENVO:00010483) + UBERON organism substance (UBERON:0000463)
              (+ variant with FOODON food material)
Metrics: top-1 and macro top-1 (mean of per-label top-1) over the 5 study folds; for subsets also the
share of Metalog samples whose label lies in the subset and top-1 on those samples.
Also: geometry of the two clouds (mean cosines, mean-vector norms, angle between the means).

Findings (fold seed 0, top-1 b / f / m): closed raw .440 / .384 / .245; centring only .370 / .184 / .293;
prior only .484 / .511 / .615; both .559 / .580 / .670 (macro .211 / .055 / .059: the prior favours frequent
labels). Open raw .028 / .136 / .083, open centre + prior .543 / .581 / .675. Subsets, raw top-1: biome
(151 terms) .376, feature (1,603 ENVO terms) .182 (.236 centred), material (947 terms) .183. Clouds:
mean cosine sample-sample .40, term-term .32, sample-term .23; cos(mu_x, mu_t) .65 (modality gap).

python experiments/deck_review/retrieval_ablation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SLOTS, Data, parser, save_json, unit  # noqa: E402

BIOME, MATERIAL, SYSTEM, BODY_PART = "ENVO_00000428", "ENVO_00010483", "ENVO_01000254", "ENVO_01000813"
ANIMAL_ENV, PLANT_ENV = "ENVO_01001002", "ENVO_01001001"
SUBSTANCE, UBERON_STRUCTURE, PO_ANATOMY, FOOD = "UBERON_0000463", "UBERON_0000061", "PO_0025131", "FOODON_00002403"


def subsets(d):
    anc = d.anc
    A = {t: anc.get(t, set()) | {t} for t in d.term_ids}

    def under(roots, exclude=(), prefix=None):
        r, e = set(roots), set(exclude)
        return sorted(t for t in d.term_ids if A[t] & r and not A[t] & e and (prefix is None or t.startswith(prefix)))
    return {
        "biome": {"ENVO biome + animal/plant-associated env.": under([BIOME]) + [ANIMAL_ENV, PLANT_ENV]},
        "feature": {"ENVO system / body part": under([SYSTEM, BODY_PART], [MATERIAL, BIOME], "ENVO"),
                    "+ UBERON structure, PO anatomy": under([SYSTEM, BODY_PART, UBERON_STRUCTURE, PO_ANATOMY],
                                                           [MATERIAL, BIOME, SUBSTANCE])},
        "material": {"ENVO material + UBERON substance": under([MATERIAL, SUBSTANCE]),
                     "+ FOODON food material": under([MATERIAL, SUBSTANCE, FOOD])},
    }


def scores(A, V):
    """A: samples (2 x 1024 layout, unit), V: terms (1024, unit) -> cos(A, [V, V] / sqrt 2)."""
    h = A.shape[1] // 2
    return (A[:, :h] + A[:, h:]) @ V.T / np.sqrt(2)


def run(d, slot, vocab_fn, centre, prior):
    y = d.y(slot)
    pred, gold = [], []
    for tr, te in d.folds:
        tr, te = tr[y[tr] != ""], te[y[te] != ""]
        vocab = np.asarray(vocab_fn(y[tr]))
        V = d.term_vector(vocab)
        A = d.X[te]
        if centre:
            A, V = unit(A - d.X[tr].mean(0)), unit(V - V.mean(0))
        bias = 0
        if prior:
            labels, counts = np.unique(y[tr], return_counts=True)
            n = pd.Series(counts, index=labels).reindex(vocab).fillna(0).to_numpy()
            bias = 0.1 * np.log((n + 0.5) / (n + 0.5).sum())
        for a in range(0, len(A), 500):
            pred += list(vocab[(scores(A[a:a + 500], V) + bias).argmax(1)])
        gold += list(y[te])
    pred, gold = np.array(pred), np.array(gold)
    hit = pred == gold
    return pred, gold, {"top1": hit.mean(), "macro": pd.Series(hit).groupby(gold).mean().mean(), "n": len(gold)}


def main():
    ap = parser(__doc__, "retrieval_ablation")
    ap.add_argument("--skip_open", action="store_true", help="skip the 49k-term runs (slow, ~1.5 GB RAM)")
    args = ap.parse_args()
    d = Data(args)
    ids = d.term_ids[np.abs(d.term_vector(d.term_ids)).sum(1) > 0]  # terms with a vector
    out = {}

    def report(key, r):
        out[key] = r
        print(f"{key:70s} " + " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)

    closed = lambda y: np.unique(y)  # noqa: E731
    for slot in SLOTS:
        for name, c, p in [("raw", False, False), ("centre", True, False), ("prior", False, True), ("centre+prior", True, True)]:
            report(f"{slot}|closed {name}", run(d, slot, closed, c, p)[2])
        if not args.skip_open:
            for name, c, p in [("raw", False, False), ("centre", True, False), ("centre+prior", True, True)]:
                report(f"{slot}|open {name}", run(d, slot, lambda y: ids, c, p)[2])
    for slot, variants in subsets(d).items():
        y = d.y(slot)
        for name, vocab in variants.items():
            vocab = [t for t in vocab if t in set(ids)]
            inside = pd.Series(y[y != ""]).isin(set(vocab))
            report(f"{slot}|subset {name}|coverage", {"terms": len(vocab), "samples_in_subset": inside.mean(),
                                                      "labels_in_subset": pd.Series(np.unique(y[y != ""])).isin(set(vocab)).mean()})
            for c in (False, True):
                pred, gold, r = run(d, slot, lambda _y, v=np.array(vocab): v, c, False)
                m = np.isin(gold, vocab)
                h = pred[m] == gold[m]
                r.update(top1_in_subset=h.mean(), macro_in_subset=pd.Series(h).groupby(gold[m]).mean().mean())
                report(f"{slot}|subset {name}|{'centre' if c else 'raw'}", r)
    # geometry of the sample and term clouds (all Metalog labels of the three slots)
    rng = np.random.default_rng(0)
    labels = np.unique(np.concatenate([d.y(s)[d.y(s) != ""] for s in SLOTS]))
    T2 = d.term_matrix(labels)
    X = d.X
    i, j = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
    mx, mt = X.mean(0), T2.mean(0)
    report("geometry", {"cos_sample_sample": float((X[i] * X[j]).sum(1).mean()),
                        "cos_term_term": float((T2 @ T2.T)[np.triu_indices(len(T2), 1)].mean()),
                        "cos_sample_term": float((X[rng.integers(0, len(X), 3000)] @ T2.T).mean()),
                        "norm_mu_x": float(np.linalg.norm(mx)), "norm_mu_t": float(np.linalg.norm(mt)),
                        "cos_mu_x_mu_t": float(mx @ mt / np.linalg.norm(mx) / np.linalg.norm(mt))})
    save_json(out, args.output)


if __name__ == "__main__":
    main()
