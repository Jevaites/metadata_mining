#!/usr/bin/env python3
"""
Map sample metadata text to ENVO/Uberon terms, one prediction per slot
(biome, feature, material), and score it against Metalog labels.

The methods use the same features (see --features), so they are directly comparable:
  majority   always predict the most frequent training label (the floor to beat)
  knn        label transfer: labels of the k most similar *training samples*
  linear     a linear classifier trained on the sample vectors ("linear probe"):
             LinearSVC for sparse features (TF-IDF), RidgeClassifier for dense embeddings
             (as accurate, and ~100x faster than LinearSVC on dense data)
and, when every feature block has a term representation (text encoders, or .npz
blocks together with --term_vectors):
  retrieval  zero-shot: nearest ontology term to the sample (term text = label + synonyms)
  hybrid     linear score + --hybrid_weight x retrieval cosine; with --open_vocab it can
             also predict terms never seen in training (their linear score is -1)
  label_reg  (dense features only) ridge regression from the sample vector to the
             *embedding of its gold term*, then the nearest term to the predicted vector

Evaluation is k-fold cross-validation over *studies* (GroupKFold on study_code):
samples of one study share most of their text, so a random split would leak.
At most --max_per_study samples are kept per study, so a few huge cohorts do
not dominate the scores (most MicrobeAtlas studies are small).

--features takes one or more blocks; several blocks are concatenated (each block
L2-normalised and weighted 1/sqrt(n_blocks), so cosine = mean of the block cosines):
  tfidf              TF-IDF of the sample text (local; fitted on the training fold)
  st:<model>         sentence-transformers model on the sample text
  <model>            OpenAI-compatible embedding model on the sample text
                     (--api_key_path, optional --base_url), cached on disk
  <file>.npz         precomputed per-sample vectors (extract_sample_embeddings.py),
                     e.g. the GPT keyword / sub-biome embeddings
Samples without a vector in every .npz block are dropped. For .npz blocks the term
side comes from --term_vectors (embed_ontology_terms.py, same model and dimension).
With --term_vectors, metrics also report pred_gold_cosine (cosine between predicted
and gold term embeddings; 1 when exact): partial credit for near-misses.

python scripts/map_samples_to_ontology.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/tfidf \
  --features tfidf
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import LinearSVC

SLOTS = ["biome", "feature", "material"]


def term_text(terms):
    """The text that represents an ontology term: 'label; synonym; synonym' (definitions left out)."""
    return terms["label"] + "; " + terms["synonyms"].str.replace("||", "; ", regex=False)


# ----------------------------------------------------------------------------- encoders
def make_encoder(name, fit_texts, api_key_path=None, base_url=None, cache_dir="."):
    """Return encode(texts) -> L2-normalised matrix (rows = texts). Cosine = dot product."""
    if name == "tfidf":
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                     stop_words="english", max_features=300_000).fit(fit_texts)
        return lambda texts: normalize(vectorizer.transform(texts))

    if name.startswith("st:"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name[3:])
        return lambda texts: model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                          show_progress_bar=True)

    from openai import OpenAI
    client = OpenAI(api_key=open(os.path.expanduser(api_key_path)).read().strip(),
                    base_url=base_url, max_retries=8)
    cache_path = os.path.join(cache_dir, f"embedding_cache__{name.replace('/', '-')}.npz")

    def encode(texts):
        """Embed with the API, caching every distinct text (the same text is never paid twice)."""
        cache = {}
        if os.path.exists(cache_path):
            saved = np.load(cache_path)
            cache = dict(zip(saved["keys"], saved["vectors"]))
        key = {t: hashlib.md5(t.encode()).hexdigest() for t in texts}  # short, fixed-size cache keys
        todo = [t for t in dict.fromkeys(texts) if key[t] not in cache]
        for start in range(0, len(todo), 1000):
            batch = todo[start:start + 1000]
            response = client.embeddings.create(model=name, input=batch)
            cache.update({key[t]: np.asarray(d.embedding, dtype=np.float32) for t, d in zip(batch, response.data)})
            print(f"  embedded {min(start + 1000, len(todo))}/{len(todo)} new texts")
        if todo:
            np.savez(cache_path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
        return normalize(np.stack([cache[key[t]] for t in texts]))
    return encode


def load_precomputed(path):
    """{sample_id: row} and the L2-normalised per-sample matrix of an extract_sample_embeddings.py file."""
    saved = np.load(os.path.expanduser(path))
    return {s: i for i, s in enumerate(saved["sample_ids"])}, normalize(saved["vectors"])[saved["index"]]


def load_term_vectors(path, texts):
    """L2-normalised term embeddings (embed_ontology_terms.py output), one row per text in `texts`."""
    import h5py
    with h5py.File(os.path.expanduser(path), "r") as handle:
        row_of = {t.decode() if isinstance(t, bytes) else t: i for i, t in enumerate(handle["texts"][:])}
        missing = [t for t in texts if t not in row_of]
        if missing:
            raise SystemExit(f"{len(missing)} term texts are not in {path}, e.g. {missing[0]!r}: re-run embed_ontology_terms.py")
        vectors = handle["embeddings"][:]
    return normalize(vectors[[row_of[t] for t in texts]])


def stack(blocks):
    """Concatenate feature blocks column-wise (sparse if any block is sparse)."""
    if len(blocks) == 1:
        return blocks[0]
    if any(sparse.issparse(b) for b in blocks):
        return sparse.hstack([sparse.csr_matrix(b) for b in blocks]).tocsr()
    return np.hstack(blocks)


def build_features(specs, precomputed, fit_texts, args, out_dir, term_vectors=None):
    """Return encode_samples(df) and encode_terms() (None if a block has no term representation).
    .npz blocks use `term_vectors` (rows aligned with the term table) as their term side."""
    weight = 1 / np.sqrt(len(specs))
    sample_fns, text_fns = [], []
    for spec in specs:
        if spec in precomputed:
            row_of, vectors = precomputed[spec]
            sample_fns.append(lambda df, r=row_of, v=vectors: v[[r[s] for s in df["sample_id"]]])
            text_fns.append(None if term_vectors is None else (lambda texts: term_vectors))
        else:
            encode = make_encoder(spec, fit_texts, args.api_key_path, args.base_url, out_dir)
            sample_fns.append(lambda df, e=encode: e(list(df["text"])))
            text_fns.append(encode)
    encode_samples = lambda df: stack([weight * f(df) for f in sample_fns])
    encode_terms = None if None in text_fns else (lambda texts: stack([weight * f(texts) for f in text_fns]))
    return encode_samples, encode_terms


def top_k_similar(queries, items, k, chunk=2000):
    """Indices and cosine scores of the k most similar items for each query (both normalised)."""
    all_idx, all_sim = [], []
    for start in range(0, queries.shape[0], chunk):
        sim = queries[start:start + chunk] @ items.T
        sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
        idx = np.argpartition(-sim, min(k, sim.shape[1] - 1), axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        all_idx.append(idx)
        all_sim.append(np.take_along_axis(sim, idx, axis=1))
    return np.vstack(all_idx), np.vstack(all_sim)


# ----------------------------------------------------------------------------- methods
# each method returns (top-5 label lists, confidence per sample); confidence only ranks samples
def rank(scores, ids, n=5):
    """Top-n ids per row of a (samples x ids) score matrix; confidence = best minus second-best score."""
    order = np.argsort(-scores, axis=1)[:, :n]
    top2 = np.take_along_axis(scores, order[:, :2], axis=1)
    return [ids[row].tolist() for row in order], top2[:, 0] - top2[:, -1]


def predict_retrieval(sample_vecs, term_vecs, term_ids, n=5):
    idx, sim = top_k_similar(sample_vecs, term_vecs, n)
    return [[term_ids[i] for i in row] for row in idx], sim[:, 0]  # confidence: best cosine


def predict_knn(sample_vecs, train_vecs, train_labels, k=25, n=5):
    """Similarity-weighted vote over the labels of the k nearest training samples."""
    idx, sim = top_k_similar(sample_vecs, train_vecs, k)
    ranked, confidence = [], []
    for row_idx, row_sim in zip(idx, sim):
        votes = defaultdict(float)
        for i, s in zip(row_idx, row_sim):
            votes[train_labels[i]] += s
        ranked.append(sorted(votes, key=votes.get, reverse=True)[:n])
        confidence.append(votes[ranked[-1][0]] / max(sum(votes.values()), 1e-9))  # winning vote share
    return ranked, np.array(confidence)


def linear_scores(sample_vecs, train_vecs, train_labels):
    """One-vs-rest linear classifier -> (classes, decision scores samples x classes)."""
    classes = np.unique(train_labels)
    if len(classes) < 2:
        return classes, np.ones((sample_vecs.shape[0], 1))
    clf = LinearSVC(C=0.5) if sparse.issparse(train_vecs) else RidgeClassifier(alpha=1.0)
    clf.fit(train_vecs, train_labels)
    scores = clf.decision_function(sample_vecs)
    if scores.ndim == 1:  # binary problem: sklearn returns one column (score of classes_[1])
        scores = np.column_stack([-scores, scores])
    return clf.classes_, scores


def predict_linear(sample_vecs, train_vecs, train_labels):
    return rank(*reversed(linear_scores(sample_vecs, train_vecs, train_labels)))


def predict_hybrid(classes, linear, sample_vecs, term_vecs, term_ids, weight):
    """Linear score + weight x cosine(sample, term), over term_ids (terms unseen in training score -1)."""
    column = {t: i for i, t in enumerate(term_ids)}
    scores = np.full((linear.shape[0], len(term_ids)), -1.0)
    scores[:, [column[c] for c in classes]] = linear
    cosine = sample_vecs @ term_vecs.T
    return rank(scores + weight * (cosine.toarray() if sparse.issparse(cosine) else cosine), term_ids)


def predict_label_regression(sample_vecs, train_vecs, train_term_vecs, term_vecs, term_ids, alpha=10.0):
    """Regress sample vector -> gold-term embedding, then rank terms by cosine to the prediction."""
    predicted = normalize(Ridge(alpha=alpha).fit(train_vecs, train_term_vecs).predict(sample_vecs))
    return rank(predicted @ term_vecs.T, term_ids)


# ----------------------------------------------------------------------------- scoring
def score(ranked, gold, parents, confidence, seen, term_vec_of=None):
    """top1 / top5 accuracy, top1 counting a direct is_a parent or child as a hit, macro top1,
    top1 on the most confident half of the samples (what you get if you only label those),
    top1 on samples whose gold label never occurs in the training fold (only zero-shot can get
    those right), and, with term vectors, the mean cosine between predicted and gold terms."""
    top1 = np.array([bool(r) and r[0] == g for r, g in zip(ranked, gold)])
    near = np.array([bool(r) and (r[0] == g or r[0] in parents.get(g, ()) or g in parents.get(r[0], ()))
                     for r, g in zip(ranked, gold)])
    per_class = pd.Series(top1).groupby(np.asarray(gold)).mean()
    confident_half = np.argsort(-np.asarray(confidence))[:len(gold) // 2]
    unseen = ~np.asarray(seen)
    result = {"n": len(gold), "top1": round(top1.mean(), 4),
              "top1_confident_half": round(top1[confident_half].mean(), 4),
              "top5": round(np.mean([g in r for r, g in zip(ranked, gold)]), 4),
              "top1_or_parent_child": round(near.mean(), 4), "macro_top1": round(per_class.mean(), 4),
              "n_unseen_label": int(unseen.sum()),
              "top1_unseen_label": round(top1[unseen].mean(), 4) if unseen.any() else None}
    if term_vec_of is not None:
        result["pred_gold_cosine"] = round(float(np.mean(
            [float(term_vec_of[r[0]] @ term_vec_of[g]) if r else 0.0 for r, g in zip(ranked, gold)])), 4)
    return result


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True, help="TSV from build_metalog_training_set.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--features", nargs="+", default=["tfidf"], help="Feature blocks, see above")
    parser.add_argument("--only_samples_in", nargs="*", default=[],
                        help=".npz files: evaluate only samples present in all of them (for fair comparisons)")
    parser.add_argument("--api_key_path", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--k", type=int, default=25, help="Neighbours for knn")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds over studies")
    parser.add_argument("--max_per_study", type=int, default=50, help="Cap samples per study (0 = no cap)")
    parser.add_argument("--open_vocab", action="store_true",
                        help="retrieval/hybrid/label_reg search all ENVO+Uberon terms, not only labels seen in training")
    parser.add_argument("--term_vectors", default=None,
                        help="Term embeddings .h5 from embed_ontology_terms.py (same space as the .npz features)")
    parser.add_argument("--hybrid_weight", type=float, default=2.0, help="Weight of the cosine in `hybrid`")
    parser.add_argument("--predict", default=None,
                        help="Optional TSV (sample_id, text) to label with the linear model trained on all labelled samples; "
             "with .npz features its sample_ids must be in those files")
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
    out_dir = os.path.expanduser(args.output_dir)

    terms = pd.read_csv(os.path.expanduser(args.ontology_terms), sep="\t", keep_default_na=False)
    terms = terms[terms["obsolete"].astype(str) != "True"].reset_index(drop=True)
    terms["text"] = term_text(terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    parents = {t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p}
    term_ids = terms["term_id"].to_numpy()
    term_vectors = load_term_vectors(args.term_vectors, list(terms["text"])) if args.term_vectors else None
    term_vec_of = dict(zip(term_ids, term_vectors)) if term_vectors is not None else None
    term_row = {t: i for i, t in enumerate(term_ids)}

    samples = pd.read_csv(os.path.expanduser(args.samples), sep="\t", keep_default_na=False, dtype=str)
    samples = samples[samples[SLOTS].ne("").any(axis=1)]
    precomputed = {spec: load_precomputed(spec) for spec in args.features if spec.endswith(".npz")}
    for path in [*precomputed, *args.only_samples_in]:
        ids = precomputed[path][0] if path in precomputed else set(np.load(os.path.expanduser(path))["sample_ids"])
        samples = samples[samples["sample_id"].isin(ids)]
    if args.max_per_study:
        samples = samples.sample(frac=1, random_state=args.seed).groupby("study_code").head(args.max_per_study)
    samples = samples.reset_index(drop=True)
    print(f"{len(samples)} labelled samples from {samples.study_code.nunique()} studies")

    records = []  # one dict per (test sample, slot, method)
    folds = GroupKFold(n_splits=args.folds).split(samples, groups=samples["study_code"])
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train, test = samples.iloc[train_idx], samples.iloc[test_idx]
        print(f"fold {fold}: {len(train)} train / {len(test)} test samples")
        encode_samples, encode_terms = build_features(args.features, precomputed, list(train["text"]) +
                                                      list(terms["text"]), args, out_dir, term_vectors)
        train_vecs, test_vecs = encode_samples(train), encode_samples(test)
        term_vecs = encode_terms(list(terms["text"])) if encode_terms else None

        for slot in SLOTS:
            tr, te = train[slot].ne("").to_numpy(), test[slot].ne("").to_numpy()
            train_labels = train[slot].to_numpy()[tr]
            vocab = terms.index if args.open_vocab else terms.index[terms["term_id"].isin(set(train_labels))]
            majority = [label for label, _ in Counter(train_labels).most_common(5)]
            classes, linear = linear_scores(test_vecs[te], train_vecs[tr], train_labels)
            predictions = {
                "majority": ([majority] * te.sum(), np.zeros(te.sum())),
                "knn": predict_knn(test_vecs[te], train_vecs[tr], train_labels, k=args.k),
                "linear": rank(linear, classes),
            }
            if term_vecs is not None:
                predictions["retrieval"] = predict_retrieval(test_vecs[te], term_vecs[vocab], term_ids[vocab])
                predictions["hybrid"] = predict_hybrid(classes, linear, test_vecs[te], term_vecs[vocab],
                                                       term_ids[vocab], args.hybrid_weight)
            if term_vectors is not None and not sparse.issparse(train_vecs):
                predictions["label_reg"] = predict_label_regression(
                    test_vecs[te], train_vecs[tr], term_vectors[[term_row[t] for t in train_labels]],
                    term_vectors[vocab], term_ids[vocab])
            seen = test[slot].to_numpy()[te]
            seen = np.isin(seen, train_labels)
            for method, (top5_lists, confidence) in predictions.items():
                for row, top5, conf, is_seen in zip(test.index[te], top5_lists, confidence, seen):
                    records.append({"row": row, "slot": slot, "method": method, "top5": top5,
                                    "confidence": conf, "gold_seen_in_train": is_seen})

    pred = pd.DataFrame(records)
    pred["gold"] = [samples.at[row, slot] for row, slot in zip(pred["row"], pred["slot"])]
    metrics = {slot: {method: score(g["top5"].tolist(), g["gold"].tolist(), parents, g["confidence"].to_numpy(),
                                    g["gold_seen_in_train"].to_numpy(), term_vec_of)
                      for method, g in by_slot.groupby("method")}
               for slot, by_slot in pred.groupby("slot")}
    out = samples.loc[pred["row"], ["sample_id", "study_code", "domain"]].reset_index(drop=True)
    out = out.join(pred[["slot", "method", "gold", "confidence", "gold_seen_in_train"]])
    out["gold_label"] = out["gold"].map(label_of)
    out["pred"] = pred["top5"].str[0]
    out["pred_label"] = out["pred"].map(label_of)
    out["top5"] = pred["top5"].str.join("||")
    out.to_csv(os.path.join(out_dir, "predictions.tsv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(pd.DataFrame({(slot, m): v for slot, d in metrics.items() for m, v in d.items()}).T)

    if args.predict:  # production mode: train on every labelled sample, label new ones
        new = pd.read_csv(os.path.expanduser(args.predict), sep="\t", keep_default_na=False, dtype=str)
        encode_samples, _ = build_features(args.features, precomputed,
                                           list(samples["text"]) + list(terms["text"]), args, out_dir, term_vectors)
        all_vecs, new_vecs = encode_samples(samples), encode_samples(new)
        for slot in SLOTS:
            has = samples[slot].ne("").to_numpy()
            ranked, confidence = predict_linear(new_vecs, all_vecs[has], samples[slot].to_numpy()[has])
            new[f"{slot}_pred"] = [r[0] for r in ranked]
            new[f"{slot}_pred_label"] = [label_of.get(r[0], "") for r in ranked]
            new[f"{slot}_confidence"] = confidence
        new.drop(columns=["text"]).to_csv(os.path.join(out_dir, "new_sample_predictions.tsv"), sep="\t", index=False)
        print(f"Labelled {len(new)} new samples -> {out_dir}/new_sample_predictions.tsv")


if __name__ == "__main__":
    main()
