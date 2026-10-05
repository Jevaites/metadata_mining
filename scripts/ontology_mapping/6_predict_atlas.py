#!/usr/bin/env python3
"""
Step 6: label every MicrobeAtlas sample with one ENVO/Uberon term per slot
(biome, feature, material) + a confidence, using one model of 5_evaluate.py on
`--features keywords.npz sub_biomes.npz`:

  --method linear     (default) RidgeClassifier, as before
  --method prototype  prototype (closed vocabulary) with --prototype_alpha / --prototype_beta;
                      needs --term_vectors
  --method knn_study  knn_study with --knn_study_k; slower (a similarity to every training
                      sample), use a smaller --chunk_rows on small machines

Why it is cheap: a sample vector is x = w [kw, sb] (w = 1/sqrt 2), so every score splits over the
two blocks and is computed once per *distinct* keyword text (1.5M rows, streamed from the unique
.h5 in chunks) and once per distinct sub-biome text (32k):
    linear     W_kw . kw + W_sb . sb + b
    prototype  (x . P - mu . P) / ||x - mu|| + bias, with x . P = w (kw . P_kw + sb . P_sb) and
               ||x - mu||^2 = ||x||^2 - 2 w (kw . mu_kw + sb . mu_sb) + ||mu||^2
    knn_study  x . x_train = w^2 (kw . kw_train + sb . sb_train)
No per-sample embedding file and no API call is needed.

Training samples: exactly those of the evaluation (common.select_samples with the same
--max_per_study and --seed), so the atlas labels come from the model that was scored.
Pass --max_per_study 0 to train on every linked sample instead.

Confidence is the same as in 5_evaluate.py for the chosen method (margin for linear and prototype,
winner's vote share for knn_study): use that method's margin_for_90pct_precision in metrics.json to
decide which labels to trust.

Calibrated probabilities and hierarchical back-off (--calibration, linear and prototype): with the
calibration.json that 5_evaluate.py wrote for the same method and training settings (or several, one
per --fold_seed, pooled so that tau does not depend on one fold assignment), every slot also
gets (hierarchy.py; claude/hierarchical-backoff-results.md):
  <slot>_p              calibrated probability of the top-1 term (softmax of the scores / temperature)
  <slot>_backoff        the most specific term, among the top-1 and its broader terms that Metalog uses
                        in the slot, whose summed probability reaches the slot's tau; "" = abstain
  <slot>_backoff_label / _backoff_p   its label / summed probability
  <slot>_backoff_kind   top1 (no back-off needed), coarser, or abstain
  <slot>_candidates     the --topk best terms, then their broader terms that are labels of the slot
                        (up to --max_candidates), with probabilities, "id:p;id:p;...": the candidate
                        list of the LLM reranker (7_rerank_atlas.py; "top-5 + ancestors" in the pilot)
tau per slot = the lowest tau whose out-of-fold accuracy (--accuracy strict: the answer is gold or a
coarser true term; lenient: also a more specific one) reaches --target_accuracy in 5_evaluate's curve.
The accuracy is that of Metalog-like studies; check it on hand-labelled atlas samples before relying on it.

Resumable: model.npz, index.npz and every finished chunk are kept in --output_dir, and
--max_seconds stops cleanly; rerun the same command to continue. Delete --output_dir
to retrain (for example after changing the training options).

python 6_predict_atlas.py \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --samples ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz \
  --train_vectors ~/MicrobeAtlasProject/metalog/keywords__large1024.npz \
                  ~/MicrobeAtlasProject/metalog/sub_biomes__large1024.npz \
  --keywords_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_keywords.txt \
  --keywords_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_sub_biomes.txt \
  --sub_biomes_h5 ~/MicrobeAtlasProject/sidequest/latest/embeddings/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/atlas_kw_sb
"""

import argparse
import glob
import json
import os
import sys
import time

import importlib

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize

import hierarchy
from common import (SLOTS, ancestor_sets, ensure_settings, file_signature, load_npz, load_term_vectors,
                    load_terms, path, read_tsv, select_samples)

evaluate = importlib.import_module("5_evaluate")  # prototype_model: exactly the evaluated model
TIE_BREAK = evaluate.TIE_BREAK

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
from embed_subbiomes_keywords import iter_samples  # same text cleaning as the embedded texts

# Keep NumPy's float64 scalar here.  Step 5 uses the same expression; forcing this scalar to
# float32 made the streamed linear predictor differ slightly from the model that was evaluated.
BLOCK_WEIGHT = 1 / np.sqrt(2)


def train_models(args, model_path):
    """Per slot, the parameters of --method, trained on [kw, sb] of the evaluated samples:
      linear     coef / intercept / classes of a RidgeClassifier
      prototype  mean / prototypes / bias / classes of 5_evaluate.prototype_model (closed vocabulary)
      knn_study  the training matrix (shared by the slots), its study codes, and the labels per slot
    """
    if os.path.exists(model_path):
        model = dict(np.load(model_path))
        saved = str(model.get("method", "linear"))  # model.npz files from before --method are linear
        if saved != args.method:
            raise SystemExit(f"{model_path} holds a {saved} model, not {args.method}: use another --output_dir")
        print(f"Using existing {model_path} (delete the output dir to retrain)")
        return model
    (kw_row, kw), (sb_row, sb) = (load_npz(p) for p in args.train_vectors)
    samples = select_samples(args.samples, [set(kw_row), set(sb_row)], args.max_per_study, args.seed)
    ids = samples["sample_id"]
    X = BLOCK_WEIGHT * np.hstack([kw[[kw_row[s] for s in ids]], sb[[sb_row[s] for s in ids]]])
    model = {"method": args.method, "kw_dim": kw.shape[1]}  # columns [0, kw_dim) are keywords, the rest sub-biomes
    if args.method == "prototype":
        if not args.term_vectors:
            raise SystemExit("--method prototype needs --term_vectors")
        terms = load_terms(args.ontology_terms)
        used = set(np.concatenate([samples[slot].to_numpy() for slot in SLOTS])) - {""}
        terms = terms[terms["term_id"].isin(used)].reset_index(drop=True)  # only the labels: 49k terms do not fit
        term_row = {t: i for i, t in enumerate(terms["term_id"])}
        tv = load_term_vectors(args.term_vectors, list(terms["text"]))
        term_matrix = BLOCK_WEIGHT * np.hstack([tv, tv])  # term side, weighted as in 5_evaluate.build_features()
    if args.method == "knn_study":
        model["train_x"], model["train_studies"] = X.astype(np.float32), samples["study_code"].to_numpy().astype(str)
    for slot in SLOTS:
        has = samples[slot].ne("").to_numpy()
        y = samples[slot].to_numpy()[has]
        if args.method == "linear":
            clf = RidgeClassifier(alpha=1.0).fit(X[has], y)
            model[f"{slot}_coef"], model[f"{slot}_intercept"], model[f"{slot}_classes"] = \
                clf.coef_, clf.intercept_, clf.classes_.astype(str)
        elif args.method == "prototype":
            classes = np.unique(y)  # closed vocabulary = the training labels, as `prototype` in 5_evaluate
            mean, prototypes, bias = evaluate.prototype_model(X[has], y, term_matrix[[term_row[t] for t in classes]],
                                                              classes, args.prototype_alpha, args.prototype_beta)
            model[f"{slot}_mean"], model[f"{slot}_prototypes"] = mean.astype(np.float32), prototypes.astype(np.float32)
            model[f"{slot}_bias"], model[f"{slot}_classes"] = bias.astype(np.float32), classes.astype(str)
        else:
            classes, model[f"{slot}_labels"] = np.unique(np.where(has, samples[slot].to_numpy(), ""), return_inverse=True)
            model[f"{slot}_classes"] = classes.astype(str)  # "" = no label: those samples are not neighbours
        print(f"{slot}: trained on {has.sum()} samples, {len(np.unique(y))} labels")
    np.savez(model_path, **model)
    return model


def knn_study_top2(sim, labels, studies, n_classes, k):
    """knn_study for a block of samples: sim = samples x training samples (unlabelled training
    samples already set to -inf). Each of the k nearest neighbours votes 1 / (number of the k from
    its study); ties go to the label whose nearest neighbour is closest, as in 5_evaluate.knn_study.
    -> (best class, winner's vote share)."""
    # A slot can have fewer labelled samples than the requested k in a small prototype.  Masked
    # (-inf) rows must never enter the vote as the synthetic empty-label class.
    available = np.isfinite(sim).sum(axis=1)
    if not len(sim):
        return np.array([], dtype=int), np.array([], dtype=float)
    if available.min() == 0:
        raise ValueError("knn_study has no labelled training sample for this slot")
    k = min(k, int(available.min()))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1), axis=1)
    g = studies[idx]
    weight = 1 / (g[:, :, None] == g[:, None, :]).sum(axis=2)
    rows = np.repeat(np.arange(len(idx)), k)
    votes = np.zeros((len(idx), n_classes))
    np.add.at(votes, (rows, labels[idx].ravel()), weight.ravel())
    first = np.full((len(idx), n_classes), k)
    np.minimum.at(first, (rows, labels[idx].ravel()), np.tile(np.arange(k), len(idx)))
    rounded = np.round(votes, 9)  # as in 5_evaluate.knn_study: equal votes -> the closest label
    best = np.where(rounded == rounded.max(axis=1, keepdims=True), first, k).argmin(axis=1)
    return best, votes[np.arange(len(idx)), best] / votes.sum(axis=1)


def build_index(args, index_path):
    """For every atlas sample with a keyword text: its row in the keyword .h5 and in the
    sub-biome .h5 (-1 = no sub-biome)."""
    if os.path.exists(index_path):
        return dict(np.load(index_path))
    rows = {}
    for kind, texts, h5 in [("keywords", args.keywords_texts, args.keywords_h5),
                            ("sub_biomes", args.sub_biomes_texts, args.sub_biomes_h5)]:
        with h5py.File(path(h5), "r") as handle:
            row_of = {t.decode() if isinstance(t, bytes) else str(t): i
                      for i, t in enumerate(handle["texts"][:])}
        rows[kind] = {sid: row_of.get(t, -1) for sid, t in iter_samples(path(texts), None, kind == "keywords")}
        print(f"{kind}: {len(rows[kind])} samples with text")
    sample_ids = np.array([s for s, r in rows["keywords"].items() if r >= 0])
    index = {"sample_ids": sample_ids,
             "kw_rows": np.array([rows["keywords"][s] for s in sample_ids]),
             "sb_rows": np.array([rows["sub_biomes"].get(s, -1) for s in sample_ids])}
    np.savez(index_path, **index)
    return index


def load_backoff(args, model, terms):
    """Per slot: (temperature, tau, Closure over the model's labels), from 5_evaluate's calibration.json.
    The settings must match the model; the run manifest written in main() prevents stale cached
    models, indexes or prediction chunks from being mixed with changed inputs."""
    if not args.calibration:
        return None
    if args.method == "knn_study":
        raise SystemExit("--calibration needs --method linear or prototype (knn_study has no scores to calibrate)")
    calibrations = [json.load(open(path(p))) for p in args.calibration]
    anc = ancestor_sets({t: set(p.split("||")) for t, p in zip(terms["term_id"], terms["parents"]) if p})
    out = {}
    for slot in SLOTS:
        entries = []
        for p, calibration in zip(args.calibration, calibrations):
            if args.method not in calibration.get(slot, {}):
                raise SystemExit(f"{p} has no {args.method} calibration for {slot}: "
                                 f"run 5_evaluate.py with --backoff_methods {args.method}")
            st = calibration[slot][args.method]["settings"]
            mismatch = [f"{k}={st[k]} (here {v})" for k, v in
                        [("max_per_study", args.max_per_study), ("seed", args.seed)] +
                        ([("prototype_alpha", args.prototype_alpha), ("prototype_beta", args.prototype_beta)]
                         if args.method == "prototype" else []) if st.get(k) != v]
            if mismatch:
                raise SystemExit(f"{p} was fitted with other settings: {', '.join(mismatch)}")
            if st.get("n_samples") != calibrations[0][slot][args.method]["settings"].get("n_samples"):
                raise SystemExit(f"{p} was fitted on another number of samples than {args.calibration[0]}")
            if os.path.abspath(path(st["samples"])) != os.path.abspath(path(args.samples)):
                print(f"warning: calibration fitted on {st['samples']}, training on {args.samples}")
            entries.append(calibration[slot][args.method])
        c = hierarchy.merge_calibrations(entries)  # several fold seeds: pooled curve, one tau
        tau = hierarchy.choose_tau(c["curve"], args.target_accuracy, f"accuracy_{args.accuracy}")
        row = next(r for r in c["curve"] if r["tau"] == tau)
        print(f"{slot}: temperature {c['temperature']:.4f}, tau {tau} -> out-of-fold coverage {row['coverage']:.2f}, "
              f"accuracy {row['accuracy_' + args.accuracy]:.3f} ({args.accuracy}), exact {row['exact']:.2f}")
        classes = model[f"{slot}_classes"]
        col = {t: i for i, t in enumerate(classes)}
        broader = [[col[a] for a in anc.get(t, ()) if a in col and a not in hierarchy.NO_ANSWER] for t in classes]
        out[slot] = (c["temperature"], tau, hierarchy.Closure(classes, anc), broader)
    return out


def candidate_strings(P, classes, k, broader, max_candidates):
    """Per row 'id:p;id:p;...': the k most probable classes, then the classes that are broader terms of
    those (broader[c] = their indices), most probable first, up to max_candidates in all
    (as rerank_pilot.py build --add_ancestors)."""
    k = min(k, P.shape[1])
    idx = np.argpartition(-P, k - 1, axis=1)[:, :k]
    idx = np.take_along_axis(idx, np.argsort(-np.take_along_axis(P, idx, axis=1), axis=1), axis=1)
    out = []
    for i, row in enumerate(idx):
        row = list(row)
        if max_candidates > k:
            extra = {a for c in row for a in broader[c]} - set(row)
            row += sorted(extra, key=lambda c: -P[i, c])[:max_candidates - k]
        out.append(";".join(f"{classes[j]}:{P[i, j]:.4g}" for j in row))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", required=True)
    parser.add_argument("--samples", required=True, help="2_build_training_set.py output (labels)")
    parser.add_argument("--train_vectors", nargs=2, required=True, help="keywords .npz, sub_biomes .npz")
    parser.add_argument("--keywords_texts", required=True)
    parser.add_argument("--keywords_h5", required=True)
    parser.add_argument("--sub_biomes_texts", required=True)
    parser.add_argument("--sub_biomes_h5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["linear", "prototype", "knn_study"], default="linear")
    parser.add_argument("--term_vectors", default=None, help="4_embed_terms.py output (for --method prototype)")
    parser.add_argument("--prototype_alpha", type=float, default=0.5, help="As in 5_evaluate.py")
    parser.add_argument("--prototype_beta", type=float, default=0.1, help="As in 5_evaluate.py")
    parser.add_argument("--knn_study_k", type=int, default=50, help="As in 5_evaluate.py")
    parser.add_argument("--max_per_study", type=int, default=50, help="Same cap as the evaluation (0 = all)")
    parser.add_argument("--seed", type=int, default=22, help="Same seed as the evaluation")
    parser.add_argument("--calibration", nargs="+", default=None,
                        help="calibration.json of 5_evaluate.py (same method and settings): adds probabilities, "
                             "the back-off term and the top-k columns. Several files (runs with other "
                             "--fold_seed) are pooled: one tau from the mean curve (hierarchy.merge_calibrations)")
    parser.add_argument("--target_accuracy", type=float, default=0.9, help="Back-off: out-of-fold accuracy to reach")
    parser.add_argument("--accuracy", choices=["strict", "lenient"], default="strict",
                        help="strict: gold or a coarser true term; lenient: also a more specific term")
    parser.add_argument("--topk", type=int, default=5, help="With --calibration: top terms written in <slot>_candidates (0 = none)")
    parser.add_argument("--max_candidates", type=int, default=10,
                        help="<slot>_candidates: top-k + their broader slot labels, up to this many (= --topk: no broader terms)")
    parser.add_argument("--chunk_rows", type=int, default=20_000, help="Keyword .h5 rows per chunk (memory)")
    parser.add_argument("--max_seconds", type=float, default=None, help="Stop after this long (resume later)")
    args = parser.parse_args()
    start = time.time()
    out_dir = path(args.output_dir)
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)

    index_path = os.path.join(out_dir, "index.npz")
    inputs = [args.ontology_terms, args.samples, *args.train_vectors, args.keywords_h5, args.sub_biomes_h5]
    # Tests and advanced users may provide a prebuilt index without the original GPT text files.
    # Normal runs record the source texts, which lets resume safety catch a regenerated index.
    if all(os.path.exists(path(p)) for p in (args.keywords_texts, args.sub_biomes_texts)):
        inputs.extend([args.keywords_texts, args.sub_biomes_texts])
    elif os.path.exists(index_path):
        inputs.append(index_path)
    else:
        raise SystemExit("keyword/sub-biome text files are missing and no prebuilt index.npz was supplied")
    if args.term_vectors:
        inputs.append(args.term_vectors)
    if args.calibration:
        inputs.extend(args.calibration)
    run = {
        "inputs": [file_signature(p) for p in inputs],
        "method": args.method,
        "prototype_alpha": args.prototype_alpha,
        "prototype_beta": args.prototype_beta,
        "knn_study_k": args.knn_study_k,
        "max_per_study": args.max_per_study,
        "seed": args.seed,
        "target_accuracy": args.target_accuracy,
        "accuracy": args.accuracy,
        "topk": args.topk if args.calibration else 0,
        "max_candidates": args.max_candidates,
    }
    source_texts_exist = all(os.path.exists(path(p)) for p in (args.keywords_texts, args.sub_biomes_texts))
    cached = [os.path.join(out_dir, "model.npz"), *glob.glob(os.path.join(out_dir, "parts", "rows_*.tsv.gz"))]
    if source_texts_exist:  # without source texts, index.npz is an explicitly supplied input
        cached.append(index_path)
    ensure_settings(os.path.join(out_dir, "run_settings.json"), run, cached)

    model = train_models(args, os.path.join(out_dir, "model.npz"))
    index = build_index(args, index_path)
    terms = read_tsv(args.ontology_terms)
    label_of = dict(zip(terms["term_id"], terms["label"]))
    if not args.calibration:
        args.topk = 0
    backoff = load_backoff(args, model, terms)
    kw_dim = int(model["kw_dim"])
    with h5py.File(path(args.sub_biomes_h5), "r") as handle:
        sb = BLOCK_WEIGHT * normalize(handle["embeddings"][:])
    sb_rows = np.where(index["sb_rows"] >= 0, index["sb_rows"], len(sb))  # len(sb) = the "no sub-biome" row
    has_sb = (index["sb_rows"] >= 0).astype(np.float32)

    # per-slot parts that depend only on the sub-biome (one row per distinct sub-biome text + 1)
    if args.method == "linear":  # unchanged from the linear-only version
        coef = {slot: model[f"{slot}_coef"] for slot in SLOTS}
        sb_scores = {slot: np.vstack([sb @ coef[slot][:, kw_dim:].T, np.zeros((1, coef[slot].shape[0]))])
                     + model[f"{slot}_intercept"] for slot in SLOTS}
    elif args.method == "prototype":
        sb = np.vstack([sb, np.zeros((1, sb.shape[1]), dtype=sb.dtype)])
        P = {slot: model[f"{slot}_prototypes"] for slot in SLOTS}
        mu = {slot: model[f"{slot}_mean"] for slot in SLOTS}
        sb_dot = {slot: sb @ P[slot][:, kw_dim:].T for slot in SLOTS}            # w sb . P_sb
        sb_mu = {slot: sb @ mu[slot][kw_dim:] for slot in SLOTS}                 # w sb . mu_sb
        mu_P = {slot: P[slot] @ mu[slot] for slot in SLOTS}
        mu_mu = {slot: float(mu[slot] @ mu[slot]) for slot in SLOTS}
    else:
        sb = np.vstack([sb, np.zeros((1, sb.shape[1]), dtype=sb.dtype)])
        train_kw, train_sb = model["train_x"][:, :kw_dim], model["train_x"][:, kw_dim:]
        _, train_studies = np.unique(model["train_studies"], return_inverse=True)
        n_classes = {slot: len(model[f"{slot}_classes"]) for slot in SLOTS}
        unlabelled = {slot: model[f"{slot}_classes"][model[f"{slot}_labels"]] == "" for slot in SLOTS}
    # knn_study holds (distinct keyword rows x training samples): cap the chunk so it fits in memory
    chunk_rows = min(args.chunk_rows, 4000) if args.method == "knn_study" else args.chunk_rows
    with h5py.File(path(args.keywords_h5), "r") as handle:
        n_rows = handle["embeddings"].shape[0]
        for first in range(0, n_rows, chunk_rows):
            part = os.path.join(out_dir, "parts", f"rows_{first:08d}.tsv.gz")
            if os.path.exists(part):
                continue
            if args.max_seconds and time.time() - start > args.max_seconds:
                print(f"Stopping after {time.time() - start:.0f}s; rerun to continue")
                return
            kw = BLOCK_WEIGHT * normalize(handle["embeddings"][first:first + chunk_rows])
            in_chunk = (index["kw_rows"] >= first) & (index["kw_rows"] < first + len(kw))
            kw_local, sb_local = index["kw_rows"][in_chunk] - first, sb_rows[in_chunk]
            out = pd.DataFrame({"sample_id": index["sample_ids"][in_chunk]})
            best, confidence, extra = {}, {}, {}
            if args.method == "knn_study":
                kw_sim = kw @ train_kw.T                                  # distinct keyword rows x training
                sb_ids, sb_inverse = np.unique(sb_local, return_inverse=True)
                sb_sim = sb[sb_ids] @ train_sb.T                          # distinct sub-biomes x training
                for slot in SLOTS:
                    best[slot], confidence[slot] = np.zeros(len(out), int), np.zeros(len(out))
                for s0 in range(0, len(out), 2000):
                    sim = kw_sim[kw_local[s0:s0 + 2000]] + sb_sim[sb_inverse[s0:s0 + 2000]]
                    sim = sim.astype(np.float64) - TIE_BREAK * np.arange(sim.shape[1])  # see knn_study
                    for slot in SLOTS:
                        masked = np.where(unlabelled[slot][None, :], -np.inf, sim)
                        b, c = knn_study_top2(masked, model[f"{slot}_labels"], train_studies, n_classes[slot], args.knn_study_k)
                        best[slot][s0:s0 + 2000], confidence[slot][s0:s0 + 2000] = b, c
            else:
                for slot in SLOTS:
                    if args.method == "linear":
                        scores = (kw @ coef[slot][:, :kw_dim].T)[kw_local]
                        scores += sb_scores[slot][sb_local]  # in place (float32), as in the linear-only version
                    else:
                        dot = (kw @ P[slot][:, :kw_dim].T)[kw_local] + sb_dot[slot][sb_local] - mu_P[slot]
                        x_mu = (kw @ mu[slot][:kw_dim])[kw_local] + sb_mu[slot][sb_local]
                        norm = np.sqrt(BLOCK_WEIGHT ** 2 * (1 + has_sb[in_chunk]) - 2 * x_mu + mu_mu[slot])
                        scores = dot / norm[:, None] + model[f"{slot}_bias"]
                    top2 = np.argpartition(-scores, 1, axis=1)[:, :2]  # best two, unordered
                    top2 = np.take_along_axis(top2, np.argsort(-np.take_along_axis(scores, top2, axis=1), axis=1), axis=1)
                    best[slot] = top2[:, 0]
                    confidence[slot] = np.take_along_axis(scores, top2, axis=1) @ [1, -1]
                    if backoff:
                        T, tau, closure, broader = backoff[slot]
                        prob = hierarchy.softmax(scores.astype(np.float64), T)
                        pick, q = closure.decode(prob, tau)
                        node = np.where(pick >= 0, closure.nodes[np.maximum(pick, 0)], "")
                        extra[slot] = {"p": np.round(prob.max(axis=1), 4), "backoff": node,
                                       "backoff_label": [label_of.get(t, "") for t in node], "backoff_p": np.round(q, 4),
                                       "backoff_kind": np.where(pick < 0, "abstain", np.where(
                                           pick == closure.label_col[prob.argmax(axis=1)], "top1", "coarser"))}
                        if args.topk:
                            extra[slot]["candidates"] = candidate_strings(prob, model[f"{slot}_classes"], args.topk,
                                                                          broader, args.max_candidates)
            for slot in SLOTS:
                terms_out = model[f"{slot}_classes"][best[slot]]
                out[slot], out[f"{slot}_label"] = terms_out, [label_of.get(t, "") for t in terms_out]
                out[f"{slot}_confidence"] = np.round(confidence[slot], 4)
                for name, values in extra.get(slot, {}).items():
                    out[f"{slot}_{name}"] = values
            out.to_csv(part + ".tmp", sep="\t", index=False, compression="gzip")
            os.replace(part + ".tmp", part)  # a chunk counts as done only once fully written
            print(f"rows {first}-{first + len(kw)}: {in_chunk.sum()} samples ({time.time() - start:.0f}s)")

    final = os.path.join(out_dir, "atlas_predictions.tsv.gz")
    parts = sorted(glob.glob(os.path.join(out_dir, "parts", "rows_*.tsv.gz")))
    concat_parts(parts, final)
    print(f"Wrote {final}")


def concat_parts(parts, final):
    """Concatenate the chunk TSVs line by line (header once): the 3.4M-row table does not have to
    fit in memory, and the text is the same as the parts'."""
    import gzip
    import shutil
    header = None
    with gzip.open(final + ".tmp", "wt", compresslevel=3) as out:  # level 9 takes minutes for 3.4M rows
        for p in parts:
            with gzip.open(p, "rt") as part:
                first = part.readline()
                if header is None:
                    header = first
                    out.write(first)
                elif first != header:
                    raise SystemExit(f"{p} has other columns than {parts[0]}: use another --output_dir")
                shutil.copyfileobj(part, out)
    os.replace(final + ".tmp", final)


if __name__ == "__main__":
    main()
