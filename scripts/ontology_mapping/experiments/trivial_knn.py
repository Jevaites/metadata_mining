#!/usr/bin/env python3
"""
Cheap upgrades to k-NN (trivial-methods-upgrades.md, section 1).

Each row adds one idea to kNN; every hyper-parameter (k, similarity temperature, one vote per
study, class-prior exponent) is tuned by nested CV: 3 study-grouped inner folds inside each outer
training fold, so no row is tuned on its own test studies.

  A  kNN-25 majority vote, keywords only (the old baseline)
  B  + keywords and sub-biome as input
  C  + tuned k and similarity-weighted votes (softmax temperature tau)
  D  + one vote per study (-> knn_study in 5_evaluate.py)
  E  + class-prior correction (vote / frequency^alpha)
  F  E on centred embeddings           G  E on PCA-whitened embeddings (256 dims)
  H  E with hubness correction (CSLS)  I  E with Wilson editing of the training set
  J  kNN-50 with one vote per study in an LDA space (PCA 256 -> shrinkage LDA), + LDA itself
  ridge on the same folds, for reference

python experiments/trivial_knn.py --output ~/MicrobeAtlasProject/ontology_mapping/experiments/trivial_knn.json
(about 30-60 min; add --rows A,B,D to run a subset)
"""
import itertools
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier

from _setup import SLOTS, inner_folds, load, parser, save_json, unit

KMAX = 100
GRID = dict(k=[5, 10, 25, 50, 100], tau=[None, 0.1, 0.05, 0.02], study_vote=[False, True], alpha=[0, 0.25, 0.5, 0.75])


def transform(Xtr, Xte, how):
    if how == "raw":
        return Xtr, Xte
    mu = Xtr.mean(0)
    if how == "center":
        return unit(Xtr - mu), unit(Xte - mu)
    _, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)  # "whiten256"
    W = Vt[:256].T / S[:256]
    return unit((Xtr - mu) @ W), unit((Xte - mu) @ W)


def neighbours(Xtr, Xte, hub=False, chunk=2000):
    """The KMAX most similar training rows per test row, closest first (and their similarities)."""
    r = None
    if hub:  # CSLS: penalise training points close to everything (mean sim to their 10 nearest)
        r = np.concatenate([np.sort(Xtr[i:i + chunk] @ Xtr.T, 1)[:, -11:-1].mean(1) for i in range(0, len(Xtr), chunk)])
    I, Sm = [], []
    for i in range(0, len(Xte), chunk):
        S = Xte[i:i + chunk] @ Xtr.T
        if r is not None:
            S -= r[None, :] / 2
        idx = np.argpartition(-S, KMAX, axis=1)[:, :KMAX]
        sim = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sim, 1)
        I.append(np.take_along_axis(idx, o, 1))
        Sm.append(np.take_along_axis(sim, o, 1))
    return np.vstack(I), np.vstack(Sm)


def vote(idx, sim, ytr, gtr, C, freq, k, tau, study_vote, alpha):
    idx, sim = idx[:, :k], sim[:, :k]
    w = np.ones_like(sim) if tau is None else np.exp((sim - sim[:, :1]) / tau)
    if study_vote:  # each study's vote is shared by its neighbours
        g = gtr[idx]
        w = w / (g[:, :, None] == g[:, None, :]).sum(2)
    sc = np.zeros((len(idx), C))
    np.add.at(sc, (np.repeat(np.arange(len(idx)), k), ytr[idx].ravel()), w.ravel())
    return (sc / np.maximum(freq, 1) ** alpha).argmax(1)


def run(X, y, study, tr, te, grid, how="raw", hub=False, edit=False):
    """Tune on inner folds of tr, then predict te. -> (predictions, chosen config)."""
    classes = np.unique(y[tr])
    C = len(classes)

    def fit_predict(a, b, configs):
        Xa, Xb = transform(X[a], X[b], how)
        ya = np.searchsorted(classes, y[a])
        keep = np.ones(len(a), bool)
        if edit:  # Wilson editing: drop training points whose other-study neighbours disagree
            ia, _ = neighbours(Xa, Xa)
            lab = np.where(study[a][ia] == study[a][:, None], -1, ya[ia])[:, :25]
            maj = np.array([np.bincount(r[r >= 0], minlength=C).argmax() if (r >= 0).any() else -1 for r in lab])
            keep = (maj == ya) | (maj == -1)
            Xa, ya = Xa[keep], ya[keep]
        idx, sim = neighbours(Xa, Xb, hub)
        freq = np.bincount(ya, minlength=C)
        return [vote(idx, sim, ya, study[a][keep], C, freq, **c) for c in configs]

    configs = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    best = configs[0]
    if len(configs) > 1:
        acc = np.zeros(len(configs))
        for a, b in inner_folds(study, tr):
            for j, p in enumerate(fit_predict(a, b, configs)):
                acc[j] += (classes[p] == y[b]).sum()
        best = configs[int(acc.argmax())]
    return classes[fit_predict(tr, te, [best])[0]], best


def lda_rows(X, y, study, folds, has):
    ok_lda, ok_knn = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for tr, te in folds:
        a, b = tr[has[tr]], te[has[te]]
        pca = PCA(256, random_state=0).fit(X[a])  # LDA on 2048 dims is ill-conditioned with ~14k samples
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(pca.transform(X[a]), y[a])
        Za, Zb = unit(lda.transform(pca.transform(X[a]))), unit(lda.transform(pca.transform(X[b])))
        ok_lda[b] = lda.predict(pca.transform(X[b])) == y[b]
        classes = np.unique(y[a])
        idx, sim = neighbours(Za, Zb)
        p = vote(idx, sim, np.searchsorted(classes, y[a]), study[a], len(classes), np.ones(len(classes)), 50, None, True, 0)
        ok_knn[b] = classes[p] == y[b]
    return ok_lda[has].mean(), ok_knn[has].mean()


def main():
    p = parser(__doc__)
    p.add_argument("--rows", default="A,B,C,D,E,F,G,H,I,J")
    p.add_argument("--slots", default="biome,feature,material", help="e.g. material, to split a run over processes")
    args = p.parse_args()
    import warnings
    warnings.filterwarnings("ignore")  # LDA covariance warnings for single-sample classes
    samples, kw, sb, _, _, folds = load(args)
    KWSB = np.hstack([kw, sb]) / np.sqrt(2)
    study = samples["study_code"].to_numpy()
    fixed = dict(k=[25], tau=[None], study_vote=[False], alpha=[0])
    rows = {
        "A kNN-25 majority, keywords only (old baseline)": (kw, fixed, "raw", False, False),
        "B + keywords & sub-biome": (KWSB, fixed, "raw", False, False),
        "C + tuned k and similarity weights": (KWSB, dict(k=GRID["k"], tau=GRID["tau"], study_vote=[False], alpha=[0]), "raw", False, False),
        "D + one vote per study": (KWSB, dict(k=GRID["k"], tau=GRID["tau"], study_vote=[False, True], alpha=[0]), "raw", False, False),
        "E + class-prior correction": (KWSB, GRID, "raw", False, False),
        "F E on centred embeddings": (KWSB, GRID, "center", False, False),
        "G E on whitened embeddings (256 dims)": (KWSB, GRID, "whiten256", False, False),
        "H E with hubness correction (CSLS)": (KWSB, GRID, "raw", True, False),
        "I E with Wilson editing of training set": (KWSB, GRID, "raw", False, True),
    }
    wanted = set(args.rows.split(","))
    results = {}
    for slot in [s for s in SLOTS if s in args.slots.split(",")]:
        y = samples[slot].to_numpy()
        has = y != ""
        ok = np.zeros(len(y), bool)
        for tr, te in folds:
            a, b = tr[has[tr]], te[has[te]]
            ok[b] = RidgeClassifier(alpha=1.0).fit(KWSB[a], y[a]).predict(KWSB[b]) == y[b]
        results.setdefault("ridge (reference)", {})[slot] = round(float(ok[has].mean()), 4)
        for name, (X, grid, how, hub, edit) in rows.items():
            if name[0] not in wanted:
                continue
            t0, ok, chosen = time.time(), np.zeros(len(y), bool), []
            for tr, te in folds:
                a, b = tr[has[tr]], te[has[te]]
                pred, best = run(X, y, study, a, b, grid, how, hub, edit)
                ok[b] = pred == y[b]
                chosen.append(best)
            results.setdefault(name, {})[slot] = round(float(ok[has].mean()), 4)
            results.setdefault(name + " | chosen", {})[slot] = chosen
            print(f"{slot:8s} {name:48s} {ok[has].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if "J" in wanted:
            lda, knn_lda = lda_rows(KWSB, y, study, folds, has)
            results.setdefault("J LDA classifier", {})[slot] = round(float(lda), 4)
            results.setdefault("J kNN-50, one vote per study, in LDA space", {})[slot] = round(float(knn_lda), 4)
            print(f"{slot:8s} J LDA {lda:.3f} / kNN in LDA space {knn_lda:.3f}", flush=True)
        print(f"{slot:8s} ridge (reference) {results['ridge (reference)'][slot]:.3f}", flush=True)
        save_json(results, args.output)


if __name__ == "__main__":
    main()
