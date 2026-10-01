#!/usr/bin/env python3
"""
Step 5: cross-validated comparison of methods that map a sample to one ENVO/Uberon
term per slot (biome, feature, material), scored against Metalog's curated labels.

Every method uses the same sample vectors (--features), so rows are comparable.

  majority        the most frequent training label (the floor to beat)
  knn             similarity-weighted vote of the k nearest *training samples*
  linear          one-vs-rest linear classifier on the sample vectors
                  (RidgeClassifier on dense vectors, LinearSVC on sparse TF-IDF)
Methods that also need term vectors (text encoders, or .npz features + --term_vectors):
  retrieval       zero-shot: the term whose vector is closest to the sample, among the
                  terms used as training labels for this slot ("closed" vocabulary)
  retrieval_open  same, among all 18.8k ENVO+Uberon terms
  hybrid          linear score + --hybrid_weight x cosine(sample, term), closed vocabulary
  hybrid_open     same over all terms; terms never seen in training get linear score -1,
                  so this is the only supervised method that can output an unseen term
  label_reg       (dense features, --term_vectors) ridge regression from the sample vector
                  to the embedding of its gold term, then the closest term (closed vocab)
Upgraded "trivial" methods (added 2026-09-28; see experiments/README.md for how they were chosen):
  knn_study       knn in which each training *study* has one vote, split over its samples among
                  the --knn_study_k nearest neighbours (so one big study cannot outvote the rest)
  retrieval_prior (dense) retrieval after centring both sides + --prototype_beta x log(label
                  frequency in the training fold), closed vocabulary
  prototype       (dense) nearest class prototype: each term's (centred) vector is blended with the
                  centroid of its training samples (--prototype_alpha), + the same log prior
  prototype_open  same over all 18.8k terms; terms without training samples keep their term
                  vector (+ --prototype_unseen_bonus), so this can output an unseen term

Evaluation: cross-validation over Metalog `study_code` (common.study_folds), so no
study is in both train and test (samples of a study share most of their text). At most
--max_per_study samples per study are kept (common.select_samples).

--features takes one or more blocks. Each block is L2-normalised and weighted by
1/sqrt(n_blocks) before concatenation, so a dot product = mean of the block cosines.
  tfidf       TF-IDF of the cleaned MicrobeAtlas text (fitted on the training fold only)
  <file>.npz  precomputed per-sample vectors (3_extract_sample_embeddings.py);
              samples without a vector are dropped; term side = --term_vectors
  <model>     any other value: an OpenAI-compatible embedding model applied to the text
              (--api_key_path, --base_url, --dimensions); every distinct text is cached

Calibrated probabilities and hierarchical back-off (--backoff_methods, default prototype and linear;
hierarchy.py): the out-of-fold scores become probabilities (softmax, temperature fitted on the other
folds); the answer climbs from the top-1 to the most specific broader term Metalog uses in the slot
whose summed probability reaches tau, or abstains. For each target accuracy (--backoff_targets),
tau is chosen on the other folds and applied to the held-out fold. Accuracy is reported strict (gold
or a coarser true term) and lenient (also a more specific term than Metalog's).

Outputs in --output_dir:
  metrics.json      per slot and method, see score(); back-off methods also have "backoff":
                    temperature, the curve (coverage / accuracy / exact / too specific per tau) and,
                    per target, strict_<t> / lenient_<t> (coverage, accuracies, outcome shares)
  calibration.json  per slot and back-off method: temperature fitted on all folds, the pooled curve,
                    the top-1 probability quantiles and the training settings: the input of
                    6_predict_atlas.py --calibration and 7_rerank_atlas.py fit
  predictions.tsv   one row per test sample x slot x method: gold, top-1, top-5, confidence; back-off
                    methods also: prob (calibrated top-1 probability), backoff (answer at
                    --backoff_target, "" = abstain), backoff_q (its summed probability)

python 5_evaluate.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --features ~/MicrobeAtlasProject/metalog/keywords__large1024.npz ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --term_vectors ~/MicrobeAtlasProject/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/cv_kw_sb
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

import hierarchy
from common import SLOTS, ancestor_sets, load_npz, load_term_vectors, load_terms, path, select_samples, study_folds


# ----------------------------------------------------------------------------- features
def text_encoder(name, fit_texts, args, cache_dir):
    """encode(texts) -> L2-normalised matrix. `tfidf` or an OpenAI-compatible embedding model."""
    if name == "tfidf":
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                     stop_words="english", max_features=300_000).fit(fit_texts)
        return lambda texts: normalize(vectorizer.transform(texts))

    from openai import OpenAI
    client = OpenAI(api_key=open(path(args.api_key_path)).read().strip(), base_url=args.base_url, max_retries=8)
    extra = {"dimensions": args.dimensions} if args.dimensions else {}
    cache_path = os.path.join(cache_dir, f"embedding_cache__{name.replace('/', '-')}__dim{args.dimensions}.npz")

    def encode(texts):
        """Embed each distinct text once; the cache (md5(text) -> vector) survives across runs."""
        cache = {}
        if os.path.exists(cache_path):
            saved = np.load(cache_path)
            cache = dict(zip(saved["keys"], saved["vectors"]))
        key = {t: hashlib.md5(t.encode()).hexdigest() for t in texts}
        todo = [t for t in dict.fromkeys(texts) if key[t] not in cache]
        for start in range(0, len(todo), 1000):
            batch = todo[start:start + 1000]
            response = client.embeddings.create(model=name, input=batch, **extra)
            cache.update({key[t]: np.asarray(d.embedding, dtype=np.float32) for t, d in zip(batch, response.data)})
            print(f"  embedded {start + len(batch)}/{len(todo)} new texts")
        if todo:
            np.savez(cache_path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
        return normalize(np.stack([cache[key[t]] for t in texts]))
    return encode


def stack(blocks):
    """Concatenate feature blocks column-wise (sparse if any block is sparse)."""
    if len(blocks) == 1:
        return blocks[0]
    if any(sparse.issparse(b) for b in blocks):
        return sparse.hstack([sparse.csr_matrix(b) for b in blocks]).tocsr()
    return np.hstack(blocks)


def build_features(args, npz, train_texts, term_texts, term_vectors, cache_dir):
    """-> (encode_samples(df) -> matrix, term matrix in the same space or None)."""
    weight = 1 / np.sqrt(len(args.features))
    sample_fns, term_blocks = [], []
    for spec in args.features:
        if spec in npz:
            row_of, vectors = npz[spec]
            sample_fns.append(lambda df, r=row_of, v=vectors: v[[r[s] for s in df["sample_id"]]])
            term_blocks.append(term_vectors)  # None without --term_vectors
        else:
            encode = text_encoder(spec, train_texts + term_texts, args, cache_dir)
            sample_fns.append(lambda df, e=encode: e(list(df["text"])))
            term_blocks.append(encode(term_texts))
    encode_samples = lambda df: stack([weight * f(df) for f in sample_fns])
    term_matrix = None if any(b is None for b in term_blocks) else stack([weight * b for b in term_blocks])
    return encode_samples, term_matrix


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


# ----------------------------------------------------------------------------- methods
# every method returns (list of top-5 id lists, confidence per sample); confidence only ranks samples
def rank(scores, ids, n=5, margin=True):
    """Top-n ids per row of a (samples x ids) score matrix; confidence = best minus second-best
    score (margin=True) or the best score itself."""
    # row blocks keep memory low (samples x 18.8k terms); argsort is per row, so the result is the same
    order = np.vstack([np.argsort(-scores[i:i + 1000], axis=1)[:, :n] for i in range(0, max(len(scores), 1), 1000)])
    top2 = np.take_along_axis(scores, order[:, :2], axis=1)
    return [ids[row].tolist() for row in order], top2[:, 0] - top2[:, -1] if margin else top2[:, 0]


def knn(test, train, train_labels, k):
    """Similarity-weighted vote over the labels of the k most similar training samples;
    confidence = the winner's share of the vote."""
    ranked, confidence = [], []
    for start in range(0, test.shape[0], 2000):  # chunks keep the similarity matrix small
        sim = dense(test[start:start + 2000] @ train.T)
        idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]
        idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1), axis=1)
        for row_idx, row_sim in zip(idx, np.take_along_axis(sim, idx, axis=1)):  # closest first (tie-break)
            votes = defaultdict(float)
            for i, s in zip(row_idx, row_sim):
                votes[train_labels[i]] += s
            ranked.append(sorted(votes, key=votes.get, reverse=True)[:5])
            confidence.append(votes[ranked[-1][0]] / max(sum(votes.values()), 1e-9))
    return ranked, np.array(confidence)


TIE_BREAK = 1e-13  # knn_study: equal similarities (identical texts) are ordered by training-sample index


def knn_study(test, train, train_labels, train_studies, k):
    """knn in which every training *study* has one vote. The k most similar training samples are
    found as in knn(); each neighbour then votes 1 / (number of the k neighbours from its study).
    Metalog studies contribute up to --max_per_study near-identical samples, so without this one
    study fills the neighbour list and outvotes every other curator. Votes are not weighted by
    similarity (nested CV preferred unweighted votes, see experiments/).
    Many samples share an identical text, so similarities tie exactly; ties are broken by training
    sample index (the TIE_BREAK offset is far below float32 resolution), which makes the result
    deterministic and identical to 6_predict_atlas.py --method knn_study.
    Confidence = the winner's share of the vote."""
    ranked, confidence = [], []
    offset = TIE_BREAK * np.arange(train.shape[0])
    for start in range(0, test.shape[0], 2000):
        sim = dense(test[start:start + 2000] @ train.T).astype(np.float64) - offset
        idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]
        idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1), axis=1)
        for row_idx in idx:  # closest first, so ties go to the label met first
            studies = train_studies[row_idx]
            per_study = Counter(studies)
            votes = defaultdict(float)
            for i, study in zip(row_idx, studies):
                votes[train_labels[i]] += 1 / per_study[study]
            # rounded, so that float noise in sums like 1/3 + 1/3 + 1/3 cannot break a tie;
            # sorted() is stable, so equal votes keep the closest-first order
            ranked.append(sorted(votes, key=lambda label: round(votes[label], 9), reverse=True)[:5])
            confidence.append(votes[ranked[-1][0]] / sum(votes.values()))
    return ranked, np.array(confidence)


def prototype_model(train, train_labels, vocab_vectors, vocab_ids, alpha, beta, unseen_bonus=0.0):
    """Class prototypes for prototype() -> (sample mean, prototypes: vocab x dim, bias: vocab).

    1. Centre both sides: samples minus the training mean, terms minus the vocabulary mean. Keyword
       lists and short term names sit in different regions of the embedding space; centring
       removes that offset ("modality gap").
    2. prototype(term) = normalise(alpha * normalise(centroid of its training samples)
                                   + (1 - alpha) * normalise(centred term vector)).
       A term without training samples keeps its term vector alone.
       alpha = 0 is retrieval_prior, alpha = 1 a nearest-centroid classifier.
    3. bias(term) = beta * log((n_train(term) + 0.5) / sum): the label frequencies. Curators use a
       few conventional terms far more often than their names suggest (fecal material, not
       intestine environment); this is where most of the gain over plain retrieval comes from.
    4. + unseen_bonus for terms without training samples. A bare term vector scores lower than a
       centroid-blended prototype, so with 0 an unseen term (almost) never wins. Raising it trades
       accuracy on seen labels for recovering unseen ones (sweep in experiments/README.md).
    """
    mean = train.mean(axis=0)
    prototypes = normalize(vocab_vectors - vocab_vectors.mean(axis=0))
    column = {t: i for i, t in enumerate(vocab_ids)}
    labels, inverse, counts = np.unique(train_labels, return_inverse=True, return_counts=True)
    sums = np.zeros((len(labels), train.shape[1]))
    np.add.at(sums, inverse, normalize(train - mean))
    rows = [column[t] for t in labels]
    if alpha > 0:
        prototypes[rows] = normalize(alpha * normalize(sums) + (1 - alpha) * prototypes[rows])
    n = np.zeros(len(vocab_ids))
    n[rows] = counts
    return mean, prototypes, beta * np.log((n + 0.5) / (n + 0.5).sum()) + unseen_bonus * (n == 0)


def prototype_scores(test, model):
    """(samples x vocab) cosine(centred sample, prototype) + bias (see prototype_model)."""
    mean, prototypes, bias = model
    return normalize(test - mean) @ prototypes.T + bias


def prototype(test, model, vocab_ids, scores_out=None):
    """Rank terms by prototype_scores. With a list as `scores_out`, the score matrix is appended to
    it (closed vocabulary only: the back-off needs every score of the sample)."""
    ranked, confidence, blocks = [], [], []
    for start in range(0, max(test.shape[0], 1), 1000):  # row blocks keep memory low with 18.8k terms
        scores = prototype_scores(test[start:start + 1000], model)
        r, c = rank(scores, vocab_ids)
        ranked += r
        confidence.append(c)
        if scores_out is not None:
            blocks.append(scores)
    if scores_out is not None:
        scores_out.append(np.vstack(blocks))
    return ranked, np.concatenate(confidence)


def linear_scores(test, train, train_labels):
    """-> (classes, decision scores test x classes) of a one-vs-rest linear classifier."""
    classes = np.unique(train_labels)
    if len(classes) < 2:
        return classes, np.ones((test.shape[0], 1))
    clf = LinearSVC(C=0.5, random_state=0) if sparse.issparse(train) else RidgeClassifier(alpha=1.0)
    scores = clf.fit(train, train_labels).decision_function(test)
    if scores.ndim == 1:  # binary problem: sklearn returns only the score of classes_[1]
        scores = np.column_stack([-scores, scores])
    return clf.classes_, scores


def hybrid(classes, linear, cosine, vocab_ids, weight):
    """linear score + weight x cosine over `vocab_ids`; a term the classifier never saw scores -1."""
    column = {t: i for i, t in enumerate(vocab_ids)}
    scores = np.full(cosine.shape, -1.0)
    scores[:, [column[c] for c in classes]] = linear
    return rank(scores + weight * cosine, vocab_ids)


def label_regression(test, train, train_gold_vectors, vocab_vectors, vocab_ids, alpha=10.0):
    """Ridge regression sample vector -> gold-term embedding, then rank terms by cosine to the prediction."""
    predicted = normalize(Ridge(alpha=alpha).fit(train, train_gold_vectors).predict(test))
    return rank(predicted @ vocab_vectors.T, vocab_ids)


# ----------------------------------------------------------------------------- scoring
def precision_threshold(confidence, correct, target=0.9):
    """Lowest confidence cut-off whose retained samples reach `target` precision, and the coverage
    at that cut-off (None if no cut-off reaches it). Use it to decide which predictions to keep."""
    order = np.argsort(-confidence)
    precision = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(precision >= target)[0]
    if not len(ok):
        return None, 0.0
    return round(float(confidence[order][ok[-1]]), 4), round((ok[-1] + 1) / len(order), 4)


def score(g, parents, term_vector_of, ancestors):
    """Metrics for one (slot, method) group of predictions.tsv rows:
      top1, top5                accuracy of the first / any of the five predictions
      top1_or_parent_child      top-1 counted as a hit if it is the gold term or one is_a step away
      top1_gold_or_ancestor     top-1 counted as a hit if it is the gold term or any is_a ancestor of it
                                (true but coarser)
      top1_near_synonym         top-1 counted as a hit if its term vector has cosine >= 0.8 with the
                                gold term's (near-synonyms such as "microbial mat" / "microbial mat material")
      macro_top1                mean of per-label top-1 (weights rare labels like frequent ones)
      top1_confident_half       top-1 on the 50 % most confident samples
      top1_unseen_label         top-1 on samples whose gold term never occurs in the training fold
      pred_gold_cosine          mean cosine between predicted and gold term vectors (1 = exact)
      margin_for_90pct_precision / coverage_at_90pct_precision   see precision_threshold()
      top1_by_domain            top-1 per Metalog domain (animal, environmental, human, ocean)"""
    gold, pred, top5 = g["gold"].to_numpy(), g["top5"].str[0].to_numpy(), g["top5"]
    hit = pred == gold
    near = np.array([p == t or p in parents.get(t, ()) or t in parents.get(p, ()) for p, t in zip(pred, gold)])
    confidence, unseen = g["confidence"].to_numpy(), ~g["gold_seen_in_train"].to_numpy()
    threshold, coverage = precision_threshold(confidence, hit)
    result = {
        "n": len(g), "top1": hit.mean(), "top5": np.mean([t in r for r, t in zip(top5, gold)]),
        "top1_or_parent_child": near.mean(),
        "top1_gold_or_ancestor": np.mean([p == t or p in ancestors.get(t, ()) for p, t in zip(pred, gold)]),
        "macro_top1": pd.Series(hit).groupby(gold).mean().mean(),
        "top1_confident_half": hit[np.argsort(-confidence)[:len(g) // 2]].mean(),
        "n_unseen_label": int(unseen.sum()), "top1_unseen_label": hit[unseen].mean() if unseen.any() else None,
        "margin_for_90pct_precision": threshold, "coverage_at_90pct_precision": coverage,
    }
    if term_vector_of is not None:
        cosine = np.array([term_vector_of[p] @ term_vector_of[t] for p, t in zip(pred, gold)])
        result["pred_gold_cosine"] = cosine.mean()
        result["top1_near_synonym"] = (cosine >= 0.8).mean()
    result = {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in result.items()}
    result["top1_by_domain"] = pd.Series(hit).groupby(g["domain"].to_numpy()).mean().round(4).to_dict()
    return result


# ----------------------------------------------------------------------------- back-off
def backoff_all(backoff_folds, ancestors, args, n_samples):
    """Calibrated probabilities + hierarchical back-off for every (slot, method) in --backoff_methods
    (hierarchy.oof_backoff). -> (report per (slot, method) for metrics.json,
    calibration.json content for 6_predict_atlas.py, per (row, slot, method): prob = calibrated top-1
    probability, backoff = the answer at --backoff_target (tau chosen on the other folds; "" =
    abstain), backoff_q = its summed probability)."""
    reports, calibration, extra = {}, {}, {}
    name = f"{args.backoff_metric}_{args.backoff_target}"
    targets = sorted(set(args.backoff_targets) | {args.backoff_target})
    for (slot, method), folds in sorted(backoff_folds.items()):
        report, fitted = hierarchy.oof_backoff(folds, ancestors, targets)
        reports[(slot, method)] = report
        p_top1 = []
        for f, (P, C), tau in zip(folds, fitted, report[name]["tau_per_fold"]):
            pick, q = C.decode(P, tau)
            p_top1.append(P.max(1))
            for r, p, k, qq in zip(f["rows"], P.max(1), pick, q):
                extra[(r, slot, method)] = {"prob": round(float(p), 4), "backoff": C.nodes[k] if k >= 0 else "",
                                            "backoff_q": round(float(qq), 4)}
        p_top1 = np.concatenate(p_top1)
        calibration.setdefault(slot, {})[method] = {
            "temperature": report["temperature_all"],
            "curve": report["curve"],  # pooled out-of-fold outcome shares per tau: pick tau for any target
            "p_top1_quantiles": {str(qt): round(float(np.quantile(p_top1, qt)), 4) for qt in (0.1, 0.2, 0.3, 0.4, 0.5)},
            "settings": {"method": method, "features": args.features, "samples": args.samples, "n_samples": n_samples,
                         "max_per_study": args.max_per_study, "seed": args.seed, "fold_seed": args.fold_seed,
                         "prototype_alpha": args.prototype_alpha if method == "prototype" else 0,
                         "prototype_beta": args.prototype_beta}}
    return reports, calibration, extra


def print_backoff(backoff, metrics, targets):
    if not backoff:
        return
    print("\nHierarchical back-off (tau chosen on the other folds; cov = answered share, exact / too specific = "
          "shares of all samples; strict: gold or a coarser true term, lenient: also too specific)")
    for (slot, method), r in sorted(backoff.items()):
        cells = []
        for t in targets:
            st, le = r[f"strict_{t}"], r[f"lenient_{t}"]
            cells.append(f"@{t}: strict cov {st['coverage']:.2f} exact {st['exact']:.2f} | lenient cov {le['coverage']:.2f}")
        print(f"  {slot:8} {method:15} top1 {metrics[slot][method]['top1']:.3f} T {r['temperature_all']:.3f} | " + " || ".join(cells))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True, help="1_build_term_index.py output")
    parser.add_argument("--samples", required=True, help="2_build_training_set.py output")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--features", nargs="+", default=["tfidf"], help="Feature blocks, see above")
    parser.add_argument("--term_vectors", default=None, help="4_embed_terms.py output (same space as the .npz)")
    parser.add_argument("--only_samples_in", nargs="*", default=[],
                        help=".npz files: also require a vector there (to compare runs on identical samples)")
    parser.add_argument("--k", type=int, default=25, help="Neighbours for knn")
    parser.add_argument("--hybrid_weight", type=float, default=2.0, help="Weight of the cosine in hybrid")
    parser.add_argument("--knn_study_k", type=int, default=50, help="Neighbours for knn_study")
    parser.add_argument("--prototype_alpha", type=float, default=0.5,
                        help="prototype: weight of the training centroid vs the term vector (0-1)")
    parser.add_argument("--prototype_beta", type=float, default=0.1,
                        help="retrieval_prior / prototype: weight of the log label frequency")
    parser.add_argument("--prototype_unseen_bonus", type=float, default=0.0,
                        help="prototype_open: score bonus for terms never seen in training (see prototype_model)")
    parser.add_argument("--backoff_methods", nargs="*", default=["prototype", "linear"],
                        help="Methods (prototype, linear, retrieval_prior) that also get calibrated probabilities and "
                             "hierarchical back-off (metrics.json 'backoff', calibration.json for step 6)")
    parser.add_argument("--backoff_targets", nargs="+", type=float, default=[0.8, 0.85, 0.9, 0.95],
                        help="Target accuracies reported for the back-off")
    parser.add_argument("--backoff_target", type=float, default=0.9,
                        help="Target accuracy of the backoff column in predictions.tsv.gz")
    parser.add_argument("--backoff_metric", choices=["strict", "lenient"], default="strict",
                        help="strict: the answer is gold or a coarser true term; lenient: also a more specific term")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=0, help="Which study-to-fold assignment (see study_folds)")
    parser.add_argument("--max_per_study", type=int, default=50, help="0 = no cap")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--api_key_path", default=None, help="For embedding-model features")
    parser.add_argument("--base_url", default=None, help="For embedding-model features (e.g. a local server)")
    parser.add_argument("--dimensions", type=int, default=None, help="For embedding-model features")
    args = parser.parse_args()
    out_dir = path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    terms = load_terms(args.ontology_terms)
    term_ids, term_texts = terms["term_id"].to_numpy(), list(terms["text"])
    term_vectors = load_term_vectors(args.term_vectors, term_texts) if args.term_vectors else None
    term_row = {t: i for i, t in enumerate(term_ids)}
    parents = {t: set(p.split("||")) for t, p in zip(term_ids, terms["parents"]) if p}

    npz = {spec: load_npz(spec) for spec in args.features if spec.endswith(".npz")}
    required = [set(row_of) for row_of, _ in npz.values()]
    required += [set(np.load(path(p))["sample_ids"]) for p in args.only_samples_in]
    samples = select_samples(args.samples, required, args.max_per_study, args.seed)
    print(f"{len(samples)} labelled samples from {samples['study_code'].nunique()} studies")

    records = []  # one per (test sample, slot, method)
    backoff_folds = defaultdict(list)  # (slot, method) -> per fold: closed-vocabulary scores of the test samples
    folds = study_folds(samples["study_code"], args.folds, args.fold_seed)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train, test = samples.iloc[train_idx], samples.iloc[test_idx]
        print(f"fold {fold}: {len(train)} train / {len(test)} test samples")
        encode_samples, term_matrix = build_features(args, npz, list(train["text"]), term_texts, term_vectors, out_dir)
        train_x, test_x = encode_samples(train), encode_samples(test)

        for slot in SLOTS:
            tr, te = train[slot].ne("").to_numpy(), test[slot].ne("").to_numpy()
            y, x_tr, x_te = train[slot].to_numpy()[tr], train_x[tr], test_x[te]
            closed = np.isin(term_ids, y)  # terms used as training labels for this slot
            classes, linear = linear_scores(x_te, x_tr, y)
            fold_info = {"gold": test[slot].to_numpy()[te], "rows": test.index[te]}
            if "linear" in args.backoff_methods:
                backoff_folds[(slot, "linear")].append({"S": dense(linear), "vocab": classes, **fold_info})
            predictions = {
                "majority": ([[label for label, _ in Counter(y).most_common(5)]] * te.sum(), np.zeros(te.sum())),
                "knn": knn(x_te, x_tr, y, args.k),
                "knn_study": knn_study(x_te, x_tr, y, train["study_code"].to_numpy()[tr], args.knn_study_k),
                "linear": rank(linear, classes),
            }
            if term_matrix is not None:
                cosine = dense(x_te @ term_matrix.T)  # samples x all terms
                # confidence = best cosine: near-synonym terms make the margin meaningless here
                predictions["retrieval"] = rank(cosine[:, closed], term_ids[closed], margin=False)
                predictions["retrieval_open"] = rank(cosine, term_ids, margin=False)
                predictions["hybrid"] = hybrid(classes, linear, cosine[:, closed], term_ids[closed], args.hybrid_weight)
                predictions["hybrid_open"] = hybrid(classes, linear, cosine, term_ids, args.hybrid_weight)
            if term_matrix is not None and not sparse.issparse(x_tr):
                a, b = args.prototype_alpha, args.prototype_beta
                vocab = term_matrix[closed]
                for method, alpha in [("retrieval_prior", 0), ("prototype", a)]:
                    keep = [] if method in args.backoff_methods else None
                    predictions[method] = prototype(x_te, prototype_model(x_tr, y, vocab, term_ids[closed], alpha, b),
                                                    term_ids[closed], keep)
                    if keep:
                        backoff_folds[(slot, method)].append({"S": keep[0], "vocab": term_ids[closed], **fold_info})
                predictions["prototype_open"] = prototype(x_te, prototype_model(x_tr, y, term_matrix, term_ids, a, b,
                                                                             args.prototype_unseen_bonus), term_ids)
            if term_vectors is not None and not sparse.issparse(x_tr):
                predictions["label_reg"] = label_regression(x_te, x_tr, term_vectors[[term_row[t] for t in y]],
                                                            term_vectors[closed], term_ids[closed])
            seen = np.isin(test[slot].to_numpy()[te], y)
            for method, (top5, confidence) in predictions.items():
                records += [{"row": r, "slot": slot, "method": method, "top5": t, "confidence": c,
                             "gold_seen_in_train": s} for r, t, c, s in zip(test.index[te], top5, confidence, seen)]

    pred = pd.DataFrame(records)
    pred["gold"] = [samples.at[r, s] for r, s in zip(pred["row"], pred["slot"])]
    ancestors = ancestor_sets(parents)
    backoff, calibration, extra = backoff_all(backoff_folds, ancestors, args, len(samples))
    for column in ["prob", "backoff", "backoff_q"]:
        pred[column] = [extra.get((r, s, m), {}).get(column, "") for r, s, m in zip(pred["row"], pred["slot"], pred["method"])]
    info = samples.loc[pred["row"], ["sample_id", "study_code", "domain"]].reset_index(drop=True)
    pred = pd.concat([info, pred.drop(columns="row")], axis=1)
    term_vector_of = dict(zip(term_ids, term_vectors)) if term_vectors is not None else None
    metrics = {slot: {method: score(g, parents, term_vector_of, ancestors) for method, g in by_slot.groupby("method")}
               for slot, by_slot in pred.groupby("slot")}
    for (slot, method), report in backoff.items():
        metrics[slot][method]["backoff"] = report
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    if calibration:
        with open(os.path.join(out_dir, "calibration.json"), "w") as handle:
            json.dump(calibration, handle, indent=1)

    label_of = dict(zip(term_ids, terms["label"]))
    pred["pred"] = pred["top5"].str[0]
    pred["gold_label"], pred["pred_label"] = pred["gold"].map(label_of), pred["pred"].map(label_of)
    pred["backoff_label"] = pred["backoff"].map(label_of).fillna("")
    pred["top5"] = pred["top5"].str.join("||")
    pred.to_csv(os.path.join(out_dir, "predictions.tsv.gz"), sep="\t", index=False)

    table = pd.DataFrame({(s, m): {k: v for k, v in d.items() if k != "top1_by_domain"}
                          for s, by_method in metrics.items() for m, d in by_method.items()}).T
    pd.set_option("display.width", 250)
    print(table[["n", "top1", "top5", "top1_or_parent_child", "macro_top1", "top1_confident_half"]])
    print_backoff(backoff, metrics, args.backoff_targets)


if __name__ == "__main__":
    main()
