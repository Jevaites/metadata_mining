"""
The mapping methods: given sample vectors (and, for some, ontology-term vectors), rank the terms of a
slot for each test sample. Imported by 5_evaluate.py (cross-validation) and 6_predict_atlas.py (atlas).

Every method returns (top-5 term lists, confidence per sample). The confidence only *ranks* samples
(higher = more likely right); hierarchy.py turns scores into calibrated probabilities.

  family            method            needs                       idea
  baseline          majority          labels                      most frequent training label (in 5_evaluate)
  zero-shot         retrieval         term vectors                nearest term vector
  neighbours        knn, knn_study    labelled samples            vote of the most similar training samples
  linear            linear, hybrid    labelled samples (+ terms)  one-vs-rest ridge scores (+ term cosine)
  term space        label_regression  labelled samples + terms    regress the gold term's vector, then nearest term
  prototypes        prototype         labelled samples + terms    nearest class prototype (term vector + centroid)

Shapes used below: n = test samples, m = training samples, d = feature dimension, V = vocabulary size.
"""

from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

TIE_BREAK = 1e-13  # knn_study: equal similarities (identical texts) are ordered by training-sample index


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


# ----------------------------------------------------------------------------- ranking helpers
def rank(scores, ids, n=5, margin=True):
    """Top-n ids per row of an (n_samples x V) score matrix, and a confidence per row:
    best minus second-best score (margin=True) or the best score itself.

    Example: scores [[0.1, 0.9, 0.5]], ids ['a', 'b', 'c'] -> ([['b', 'c', 'a']], [0.4])."""
    # row blocks keep memory low (samples x 49k terms); argsort is per row, so the result is the same
    order = np.vstack([np.argsort(-scores[i:i + 1000], axis=1)[:, :n] for i in range(0, max(len(scores), 1), 1000)])
    top2 = np.take_along_axis(scores, order[:, :2], axis=1)
    return [ids[row].tolist() for row in order], top2[:, 0] - top2[:, -1] if margin else top2[:, 0]


def row_blocks(n, size=1000):
    """Row slices of at most `size` rows (at least one, possibly empty, slice)."""
    return [slice(start, start + size) for start in range(0, max(n, 1), size)]


def join_ranks(parts):
    """Concatenate the (top-n lists, confidences) of row blocks."""
    return [r for ranked, _ in parts for r in ranked], np.concatenate([c for _, c in parts])


# ----------------------------------------------------------------------------- neighbour methods
def knn(test, train, train_labels, k):
    """Similarity-weighted vote over the labels of the k most similar training samples;
    confidence = the winner's share of the vote.
    Example (k=3): neighbours (sim 0.9 'soil'), (0.8 'soil'), (0.7 'sediment') -> soil, 1.7 / 2.4 = 0.71."""
    ranked, confidence = [], []
    for start in range(0, test.shape[0], 2000):  # chunks keep the (2000 x m) similarity matrix small
        sim = dense(test[start:start + 2000] @ train.T)
        idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]  # the k best, unordered
        idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1), axis=1)
        for row_idx, row_sim in zip(idx, np.take_along_axis(sim, idx, axis=1)):  # closest first (tie-break)
            votes = defaultdict(float)
            for i, s in zip(row_idx, row_sim):
                votes[train_labels[i]] += s
            ranked.append(sorted(votes, key=votes.get, reverse=True)[:5])
            confidence.append(votes[ranked[-1][0]] / max(sum(votes.values()), 1e-9))
    return ranked, np.array(confidence)


def knn_study(test, train, train_labels, train_studies, k):
    """knn in which every training *study* has one vote. The k most similar training samples are found
    as in knn(); each neighbour then votes 1 / (number of the k neighbours from its study).

    Why: Metalog studies contribute up to 50 near-identical samples, so without this one study fills the
    neighbour list and outvotes every other curator. Example (k=4): 3 neighbours from study A say
    'soil', 1 from study B says 'sediment' -> soil 3 x 1/3 = 1, sediment 1: a tie between two curators,
    broken in favour of the closest neighbour's label.
    Votes are not weighted by similarity (nested CV preferred unweighted votes, experiments README §3).
    Many samples share an identical text, so similarities tie exactly; ties are broken by training-sample
    index (TIE_BREAK is far below float32 resolution), which makes the result deterministic and identical
    to 6_predict_atlas.py --method knn_study. Confidence = the winner's share of the vote."""
    ranked, confidence = [], []
    offset = TIE_BREAK * np.arange(train.shape[0])
    for start in range(0, test.shape[0], 2000):
        sim = dense(test[start:start + 2000] @ train.T).astype(np.float64) - offset
        idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]
        idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1), axis=1)
        for row_idx in idx:  # closest first, so ties go to the label met first
            studies = train_studies[row_idx]
            per_study = Counter(studies)  # how many of the k neighbours each study has
            votes = defaultdict(float)
            for i, study in zip(row_idx, studies):
                votes[train_labels[i]] += 1 / per_study[study]
            # rounded, so that float noise in sums like 1/3 + 1/3 + 1/3 cannot break a tie;
            # sorted() is stable, so equal votes keep the closest-first order
            ranked.append(sorted(votes, key=lambda label: round(votes[label], 9), reverse=True)[:5])
            confidence.append(votes[ranked[-1][0]] / sum(votes.values()))
    return ranked, np.array(confidence)


# ----------------------------------------------------------------------------- prototypes
def prototype_model(train, train_labels, vocab_vectors, vocab_ids, alpha, beta, unseen_bonus=0.0):
    """Class prototypes for prototype() -> (sample mean: d, prototypes: V x d, bias: V).

    1. Centre both sides: samples minus the training mean, terms minus the vocabulary mean. Keyword
       lists and short term names sit in different regions of the embedding space; centring removes
       that offset ("modality gap").
    2. prototype(term) = normalise(alpha * normalise(centroid of its training samples)
                                   + (1 - alpha) * normalise(centred term vector)).
       A term without training samples keeps its term vector alone.
       alpha = 0 is retrieval_prior, alpha = 1 a nearest-centroid classifier.
    3. bias(term) = beta * log((n_train(term) + 0.5) / sum): the label frequencies. Curators use a few
       conventional terms far more often than their names suggest (e.g. 'fecal material' for every gut
       sample); this prior is where most of the gain over plain retrieval comes from (+18 to +54 points).
       Example: beta 0.1, a label with 5,000 of 14,000 samples gets 0.1 * log(0.357) = -0.10, one with
       5 samples 0.1 * log(0.0004) = -0.78.
    4. + unseen_bonus for terms without training samples. A bare term vector scores lower than a
       centroid-blended prototype, so with 0 an unseen term (almost) never wins. Raising it trades
       accuracy on seen labels for recovering unseen ones (sweep in experiments/README.md).
    """
    mean = train.mean(axis=0)
    prototypes = normalize(vocab_vectors - vocab_vectors.mean(axis=0))
    column = {t: i for i, t in enumerate(vocab_ids)}
    labels, inverse, counts = np.unique(train_labels, return_inverse=True, return_counts=True)
    sums = np.zeros((len(labels), train.shape[1]))
    np.add.at(sums, inverse, normalize(train - mean))  # sum of the centred, normalised samples per label
    rows = [column[t] for t in labels]
    if alpha > 0:
        prototypes[rows] = normalize(alpha * normalize(sums) + (1 - alpha) * prototypes[rows])
    n = np.zeros(len(vocab_ids))
    n[rows] = counts
    return mean, prototypes, beta * np.log((n + 0.5) / (n + 0.5).sum()) + unseen_bonus * (n == 0)


def prototype_scores(test, model):
    """(n x V) cosine(centred sample, prototype) + bias (see prototype_model)."""
    mean, prototypes, bias = model
    return normalize(test - mean) @ prototypes.T + bias


def prototype(test, model, vocab_ids, scores_out=None):
    """Rank terms by prototype_scores. With a list as `scores_out`, the full score matrix is appended to
    it (closed vocabulary only: the back-off needs every score of the sample)."""
    ranked, confidence, blocks = [], [], []
    for start in range(0, max(test.shape[0], 1), 1000):  # row blocks keep memory low with 49k terms
        scores = prototype_scores(test[start:start + 1000], model)
        r, c = rank(scores, vocab_ids)
        ranked += r
        confidence.append(c)
        if scores_out is not None:
            blocks.append(scores)
    if scores_out is not None:
        scores_out.append(np.vstack(blocks))
    return ranked, np.concatenate(confidence)


# ----------------------------------------------------------------------------- linear methods
def linear_scores(test, train, train_labels):
    """-> (classes, decision scores n x classes) of a one-vs-rest linear classifier:
    RidgeClassifier on dense features (each class is a +1/-1 regression target; as accurate as
    logistic regression here and 100x faster than LinearSVC on 2048 dims), LinearSVC on sparse TF-IDF."""
    classes = np.unique(train_labels)
    if len(classes) < 2:
        return classes, np.ones((test.shape[0], 1))
    clf = LinearSVC(C=0.5, random_state=0) if sparse.issparse(train) else RidgeClassifier(alpha=1.0)
    scores = clf.fit(train, train_labels).decision_function(test)
    if scores.ndim == 1:  # binary problem: sklearn returns only the score of classes_[1]
        scores = np.column_stack([-scores, scores])
    return clf.classes_, scores


def hybrid(classes, linear, cosine, vocab_ids, weight):
    """linear score + weight x cosine(sample, term) over `vocab_ids`; a term the classifier never saw
    scores -1 (ridge's target for 'not this class'), so only its cosine can make it win (hybrid_open)."""
    column = {t: i for i, t in enumerate(vocab_ids)}
    scores = np.full(cosine.shape, -1.0)
    scores[:, [column[c] for c in classes]] = linear
    return rank(scores + weight * cosine, vocab_ids)


def label_regression(test, train, train_gold_vectors, vocab_vectors, vocab_ids, alpha=10.0):
    """Ridge regression sample vector -> embedding of its gold term, then rank terms by cosine to the
    predicted vector. Labels with similar meaning share signal instead of being unrelated classes
    (e.g. 'marine biome' and 'ocean biome' sit close in term space)."""
    predicted = normalize(Ridge(alpha=alpha).fit(train, train_gold_vectors).predict(test))
    return rank(predicted @ vocab_vectors.T, vocab_ids)
