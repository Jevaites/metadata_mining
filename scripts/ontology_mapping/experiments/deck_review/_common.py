"""Shared set-up for the deck-review analyses (experiments/deck_review, October 2026).

These scripts answer the questions raised while reviewing the presentation deck and the written report.
They use the pipeline's own sample selection, folds, methods and back-off (common.py, methods.py,
hierarchy.py), so their numbers are comparable with 5_evaluate.py on the cleaned labels + biome map
(metalog/clean/training_set.clean.tsv.gz, fold seed 0 unless stated).

Run every script from scripts/ontology_mapping, e.g.
    python experiments/deck_review/near_duplicates.py
Default paths are those of ~/MicrobeAtlasProject; every path is a command-line option. Results go to
~/MicrobeAtlasProject/ontology_mapping/experiments/deck_review/<script>.json unless --output is given.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # scripts/ontology_mapping
import methods as M  # noqa: E402,F401
from common import (SLOTS, load_npz, load_terms, path, select_samples,  # noqa: E402,F401
                    study_folds, term_ancestors)
from hierarchy import (TAUS, Closure, choose_tau, fit_temperature, outcome, relation,  # noqa: E402,F401
                       softmax, summarise)

P = "~/MicrobeAtlasProject"
DEFAULTS = {
    "ontology_terms": f"{P}/ontology_terms.tsv.gz",
    "samples": f"{P}/metalog/clean/training_set.clean.tsv.gz",
    "keywords": f"{P}/metalog/keywords__large1024.npz",
    "sub_biomes": f"{P}/metalog/sub_biomes__large1024.npz",
    "term_vectors": f"{P}/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5",
    "predictions": f"{P}/ontology_mapping/cv_backoff/predictions.tsv.gz",
}
OUT_DIR = f"{P}/ontology_mapping/experiments/deck_review"


def parser(description, script, needs=None):
    """argparse with the shared paths (all of DEFAULTS, or only `needs`), --fold_seed and --output."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in DEFAULTS.items():
        if needs is None or name in needs:
            p.add_argument(f"--{name}", default=default)
    p.add_argument("--fold_seed", type=int, default=0, help="study-to-fold assignment, as in 5_evaluate.py")
    p.add_argument("--max_per_study", type=int, default=50, help="as in 5_evaluate.py")
    p.add_argument("--output", default=f"{OUT_DIR}/{script}.json", help="JSON file for the results")
    return p


def save_json(results, output):
    os.makedirs(os.path.dirname(path(output)) or ".", exist_ok=True)
    with open(path(output), "w") as handle:
        json.dump(results, handle, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", output)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class Data:
    """The evaluation samples (same rows as 5_evaluate.py), their kw + sb vectors, the term table and
    term vectors. Loaded lazily, so a script only pays for what it uses.

    X[i] = [e(keywords), e(sub-biome)] / sqrt(2) for sample i (unit norm), as in 5_evaluate.py.
    term_matrix(ids) gives each term's vector in the same 2 x 1024 layout: [t, t] / sqrt(2)."""

    def __init__(self, args, max_per_study=None):
        self.args = args
        npz = [load_npz(args.keywords), load_npz(args.sub_biomes)]
        cap = args.max_per_study if max_per_study is None else max_per_study
        self.samples = select_samples(args.samples, [set(r) for r, _ in npz], cap)
        w = 1 / np.sqrt(2)
        self.X = np.hstack([w * v[[r[s] for s in self.samples["sample_id"]]] for r, v in npz]).astype(np.float32)
        self.study = self.samples["study_code"].to_numpy()
        self.folds = list(study_folds(self.samples["study_code"], 5, args.fold_seed))
        self._terms = None
        print(f"{len(self.samples)} samples, {self.samples['study_code'].nunique()} studies, "
              f"fold_seed {args.fold_seed}, cap {cap}", flush=True)

    def y(self, slot):
        return self.samples[slot].to_numpy()

    # --- terms -------------------------------------------------------------------------------
    def _load_terms(self):
        if self._terms is None:
            terms = load_terms(self.args.ontology_terms)
            vec = load_any_term_vectors(self.args.term_vectors, list(terms["text"]))
            self._terms = {"table": terms, "anc": term_ancestors(terms), "label": dict(zip(terms.term_id, terms.label)),
                           "ids": terms["term_id"].to_numpy(), "vectors": vec,
                           "row": {t: i for i, t in enumerate(terms["term_id"])}}
        return self._terms

    @property
    def anc(self):
        return self._load_terms()["anc"]

    @property
    def label(self):
        return self._load_terms()["label"]

    @property
    def term_ids(self):
        return self._load_terms()["ids"]

    def term_vector(self, ids):
        t = self._load_terms()
        return t["vectors"][[t["row"][i] for i in ids]]

    def term_matrix(self, ids):
        w = 1 / np.sqrt(2)
        V = self.term_vector(ids)
        return np.hstack([w * V, w * V]).astype(np.float32)


def load_any_term_vectors(p, texts):
    """Term vectors for `texts`: the pipeline's .h5 (4_embed_terms.py), or an .npz with `texts` and
    `vectors` arrays (an extract of it). Texts without a vector get a zero row (never a label in
    practice: every Metalog label and every live term is embedded)."""
    if str(p).endswith(".npz"):
        z = np.load(path(p), allow_pickle=True)
        row = {t: i for i, t in enumerate(z["texts"])}
        V = z["vectors"].astype(np.float32)
    else:
        import h5py
        with h5py.File(path(p), "r") as handle:
            row = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
            V = handle["embeddings"][:].astype(np.float32)
    out = np.zeros((len(texts), V.shape[1]), np.float32)
    have = [i for i, t in enumerate(texts) if t in row]
    out[have] = V[[row[texts[i]] for i in have]]
    missing = len(texts) - len(have)
    if missing:
        print(f"  {missing} of {len(texts)} term texts have no vector (zero rows)", flush=True)
    return unit(out)


def cv_predictions(p, method):
    """Out-of-fold predictions of 5_evaluate.py --backoff (cv_backoff/predictions.tsv.gz) for one method:
    {slot: DataFrame indexed by sample_id with pred (top-1), backoff ('' = abstain), prob, gold}."""
    P = pd.read_csv(path(p), sep="\t", dtype=str, keep_default_na=False)
    P = P[P.method == method]
    return {s: P[P.slot == s].set_index("sample_id") for s in SLOTS}


def oof_outcomes(fold_list, anc):
    """Back-off outcomes at every tau of the grid, temperature fitted on the other folds (as hierarchy.oof_backoff).
    fold_list: [{"S": scores n x V, "vocab": labels, "gold": gold labels}] -> list of (len(TAUS) x n) outcome codes."""
    blocks = []
    for f in fold_list:
        col = {v: i for i, v in enumerate(f["vocab"])}
        blocks.append((f["S"], np.array([col.get(g, -1) for g in f["gold"]])))
    outs = []
    for k, f in enumerate(fold_list):
        T = fit_temperature([b for j, b in enumerate(blocks) if j != k])
        P = softmax(f["S"], T)
        C = Closure(f["vocab"], anc)
        rel = relation(C.nodes, f["gold"], anc)
        outs.append(np.array([outcome(rel, C.decode(P, t)[0]) for t in TAUS]))
        f["T"], f["p1"] = T, P.max(1)
    return outs


def oof_backoff_at(outs, target=0.9):
    """Per fold, tau chosen on the other folds (lowest tau with strict accuracy >= target) -> outcomes
    of every sample at its fold's tau (concatenated) and the taus."""
    picked, taus = [], []
    for k in range(len(outs)):
        other = np.concatenate([o for j, o in enumerate(outs) if j != k], axis=1)
        tau = choose_tau([{"tau": float(t), **summarise(other[i])} for i, t in enumerate(TAUS)], target)
        taus.append(tau)
        picked.append(outs[k][list(TAUS).index(tau)])
    return np.concatenate(picked), taus


def prototype_fold_scores(data, slot, train_idx, test_idx, labels=None, beta=0.1, alpha=0.5):
    """Production prototype (methods.prototype_model, alpha 0.5, beta 0.1) trained on train_idx;
    -> (scores of test_idx over the training labels, vocab)."""
    y = data.y(slot) if labels is None else labels
    vocab = np.unique(y[train_idx])
    model = M.prototype_model(data.X[train_idx], y[train_idx], data.term_matrix(vocab), vocab, alpha, beta)
    S = np.vstack([M.prototype_scores(data.X[test_idx[a:a + 2000]], model) for a in range(0, len(test_idx), 2000)]) \
        if len(test_idx) else np.zeros((0, len(vocab)))
    return S, vocab
