#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid ontology mapper for ENVO/Uberon experiments.

This script supports:
- exact / synonym matching
- lexical retrieval with TF-IDF
- optional dense retrieval with OpenAI-compatible embeddings
- optional top-k reranking with an OpenAI-compatible chat model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
    from openai import OpenAI
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for map_metadata_to_ontology.py. "
    ) from exc

from ontology_mapping_utils import (
    build_term_text,
    load_api_key,
    load_tabular,
    normalize_text,
    safe_literal_list,
    save_tabular,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map metadata mentions to ontology terms.")
    parser.add_argument("--ontology_terms", required=True, help="TSV/CSV term table from build_ontology_term_index.py")
    parser.add_argument("--sample_mentions", required=True, help="TSV/CSV with mention_text/context_text")
    parser.add_argument("--output_dir", required=True, help="Directory for predictions and metrics")
    parser.add_argument("--candidate_k", type=int, default=20, help="How many candidates to keep per sample")
    parser.add_argument("--lexical_pool_size", type=int, default=50, help="How many candidates to gather before reranking")
    parser.add_argument("--word_weight", type=float, default=0.45)
    parser.add_argument("--char_weight", type=float, default=0.35)
    parser.add_argument("--dense_weight", type=float, default=0.20)
    parser.add_argument("--exact_boost", type=float, default=0.25)
    parser.add_argument("--embedding_api_key_path", default=None)
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--reranker_api_key_path", default=None)
    parser.add_argument("--reranker_model", default=None)
    parser.add_argument("--base_url", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--reranker_temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def validate_inputs(terms_df: pd.DataFrame, samples_df: pd.DataFrame) -> None:
    required_term_cols = {"ontology", "term_id", "label"}
    required_sample_cols = {"sample_id", "target_ontology"}
    missing_term = required_term_cols - set(terms_df.columns)
    missing_sample = required_sample_cols - set(samples_df.columns)
    if missing_term:
        raise ValueError(f"Ontology term table is missing columns: {sorted(missing_term)}")
    if missing_sample:
        raise ValueError(f"Sample mentions table is missing columns: {sorted(missing_sample)}")


def prepare_term_table(terms_df: pd.DataFrame) -> pd.DataFrame:
    terms_df = terms_df.copy()
    if "synonyms" not in terms_df.columns:
        terms_df["synonyms"] = ""
    if "definition" not in terms_df.columns:
        terms_df["definition"] = ""
    if "parent_labels" not in terms_df.columns:
        terms_df["parent_labels"] = ""

    terms_df["ontology"] = terms_df["ontology"].astype(str).str.upper()
    terms_df["synonym_list"] = terms_df["synonyms"].apply(safe_literal_list)
    terms_df["parent_label_list"] = terms_df["parent_labels"].apply(safe_literal_list)
    terms_df["text_for_embedding"] = terms_df.apply(
        lambda row: row["text_for_embedding"]
        if "text_for_embedding" in terms_df.columns and pd.notna(row.get("text_for_embedding", None)) and str(row["text_for_embedding"]).strip()
        else build_term_text(row["label"], row["synonym_list"], row["definition"], row["parent_label_list"]),
        axis=1,
    )
    terms_df["lexical_text"] = terms_df.apply(
        lambda row: " | ".join(
            piece
            for piece in [
                str(row["label"]).strip(),
                "; ".join(row["synonym_list"]),
                str(row["definition"]).strip(),
                "; ".join(row["parent_label_list"]),
            ]
            if piece
        ),
        axis=1,
    )
    terms_df["label_norm"] = terms_df["label"].apply(normalize_text)
    terms_df["synonym_norms"] = terms_df["synonym_list"].apply(lambda values: [normalize_text(value) for value in values if normalize_text(value)])
    return terms_df


def prepare_sample_table(samples_df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
    samples_df = samples_df.copy()
    if "mention_text" not in samples_df.columns:
        samples_df["mention_text"] = ""
    if "context_text" not in samples_df.columns:
        samples_df["context_text"] = ""
    samples_df["target_ontology"] = samples_df["target_ontology"].astype(str).str.upper()
    samples_df["mention_text"] = samples_df["mention_text"].fillna("").astype(str)
    samples_df["context_text"] = samples_df["context_text"].fillna("").astype(str)
    samples_df["query_text"] = samples_df.apply(
        lambda row: " | ".join(part for part in [row["mention_text"], row["context_text"]] if part.strip()),
        axis=1,
    )
    if max_samples is not None:
        samples_df = samples_df.head(max_samples).copy()
    return samples_df


def build_exact_index(term_rows: pd.DataFrame) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for row in term_rows.itertuples():
        normalized_strings = [row.label_norm] + list(row.synonym_norms)
        for normalized in normalized_strings:
            if not normalized:
                continue
            index.setdefault(normalized, []).append(row.Index)
    return index


def top_indices(scores: np.ndarray, limit: int) -> List[int]:
    if scores.size == 0:
        return []
    limit = min(limit, scores.size)
    partition = np.argpartition(-scores, limit - 1)[:limit]
    ordered = partition[np.argsort(-scores[partition])]
    return ordered.tolist()


def normalize_score_map(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    values = np.array(list(score_map.values()), dtype=float)
    max_val = float(values.max())
    min_val = float(values.min())
    if max_val == min_val:
        return {key: 1.0 for key in score_map}
    return {key: (float(value) - min_val) / (max_val - min_val) for key, value in score_map.items()}


def build_lexical_models(term_rows: pd.DataFrame) -> Dict[str, object]:
    lexical_texts = term_rows["lexical_text"].fillna("").astype(str).tolist()
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        strip_accents="unicode",
    )
    word_matrix = word_vectorizer.fit_transform(lexical_texts)
    char_matrix = char_vectorizer.fit_transform(lexical_texts)
    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "word_matrix": word_matrix,
        "char_matrix": char_matrix,
    }


def embed_texts(client: OpenAI, model: str, texts: Sequence[str], batch_size: int = 100) -> np.ndarray:
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        response = client.embeddings.create(input=chunk, model=model)
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def sanitize_model_name(model_name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in model_name)


def cache_path(output_dir: str, prefix: str, model_name: str, ontology: str) -> str:
    return os.path.join(output_dir, f"{prefix}__{ontology}__{sanitize_model_name(model_name)}.npz")


def get_or_create_term_embeddings(
    client: OpenAI,
    model_name: str,
    term_rows: pd.DataFrame,
    output_dir: str,
    ontology: str,
) -> np.ndarray:
    path = cache_path(output_dir, "term_embeddings", model_name, ontology)
    if os.path.exists(path):
        loaded = np.load(path)
        return loaded["embeddings"]

    embeddings = embed_texts(client, model_name, term_rows["text_for_embedding"].tolist())
    np.savez_compressed(path, embeddings=embeddings)
    return embeddings


def candidate_row_to_payload(row: pd.Series) -> dict:
    return {
        "term_id": row["term_id"],
        "label": row["label"],
        "definition": row.get("definition", ""),
        "parent_labels": safe_literal_list(row.get("parent_labels", "")),
        "score": round(float(row["combined_score"]), 6),
    }


def rerank_with_llm(
    client: OpenAI,
    model_name: str,
    sample_row: pd.Series,
    candidates_df: pd.DataFrame,
    temperature: float,
) -> Tuple[str | None, float | None, List[str]]:
    candidate_payload = [candidate_row_to_payload(row) for _, row in candidates_df.iterrows()]
    prompt = {
        "task": "Select the ontology term that best matches the sample mention.",
        "rules": [
            "Prefer the most specific term supported by the sample text and context.",
            "Do not invent ontology terms.",
            "If several candidates are plausible, rank them.",
            "Use only the provided candidates.",
        ],
        "sample": {
            "sample_id": sample_row["sample_id"],
            "target_ontology": sample_row["target_ontology"],
            "mention_text": sample_row["mention_text"],
            "context_text": sample_row["context_text"],
        },
        "candidates": candidate_payload,
        "output_schema": {
            "best_term_id": "string or null",
            "confidence": "number between 0 and 1",
            "ranked_term_ids": ["candidate ids in best-first order"],
        },
    }

    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an ontology reranker for biological sample metadata."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    best_term_id = data.get("best_term_id")
    confidence = data.get("confidence")
    ranked_term_ids = data.get("ranked_term_ids") or []
    ranked_term_ids = [str(term_id) for term_id in ranked_term_ids]
    if best_term_id is not None:
        best_term_id = str(best_term_id)
    if confidence is not None:
        confidence = float(confidence)
    return best_term_id, confidence, ranked_term_ids


def compute_metrics(predictions_df: pd.DataFrame) -> dict:
    scored_df = predictions_df[predictions_df["gold_term_id"].astype(str).str.strip().ne("")].copy()
    if scored_df.empty:
        return {}

    scored_df["exact_match"] = scored_df["predicted_term_id"] == scored_df["gold_term_id"]
    scored_df["topk_match"] = scored_df.apply(
        lambda row: row["gold_term_id"] in safe_literal_list(row["topk_term_ids"]),
        axis=1,
    )

    metrics = {
        "overall": {
            "n": int(len(scored_df)),
            "exact_accuracy": float(scored_df["exact_match"].mean()),
            "topk_recall": float(scored_df["topk_match"].mean()),
        },
        "by_ontology": {},
    }

    for ontology, ontology_df in scored_df.groupby("target_ontology"):
        metrics["by_ontology"][ontology] = {
            "n": int(len(ontology_df)),
            "exact_accuracy": float(ontology_df["exact_match"].mean()),
            "topk_recall": float(ontology_df["topk_match"].mean()),
        }

    return metrics


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    terms_df = prepare_term_table(load_tabular(os.path.abspath(os.path.expanduser(args.ontology_terms))))
    samples_df = prepare_sample_table(load_tabular(os.path.abspath(os.path.expanduser(args.sample_mentions))), args.max_samples)
    validate_inputs(terms_df, samples_df)

    embedding_client = None
    reranker_client = None

    if args.embedding_api_key_path:
        embedding_client = OpenAI(
            api_key=load_api_key(os.path.abspath(os.path.expanduser(args.embedding_api_key_path))),
            base_url=args.base_url or None,
        )
    if args.reranker_api_key_path and args.reranker_model:
        reranker_client = OpenAI(
            api_key=load_api_key(os.path.abspath(os.path.expanduser(args.reranker_api_key_path))),
            base_url=args.base_url or None,
        )

    all_predictions = []
    all_candidate_rows = []

    for ontology, ontology_terms in terms_df.groupby("ontology", sort=False):
        ontology_terms = ontology_terms.reset_index(drop=True)
        ontology_samples = samples_df[samples_df["target_ontology"] == ontology].copy()
        if ontology_samples.empty:
            continue

        exact_index = build_exact_index(ontology_terms)
        lexical_models = build_lexical_models(ontology_terms)

        dense_matrix = None
        if embedding_client is not None:
            dense_matrix = get_or_create_term_embeddings(
                embedding_client,
                args.embedding_model,
                ontology_terms,
                output_dir,
                ontology,
            )
            dense_norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
            dense_norms[dense_norms == 0] = 1.0
            dense_matrix = dense_matrix / dense_norms

        for sample_row in ontology_samples.itertuples(index=False):
            mention_norm = normalize_text(sample_row.mention_text)
            query_text = sample_row.query_text.strip() or sample_row.mention_text.strip() or sample_row.context_text.strip()
            query_norm = normalize_text(query_text)

            exact_scores = {}
            for normalized in [mention_norm, query_norm]:
                if normalized and normalized in exact_index:
                    for idx in exact_index[normalized]:
                        exact_scores[idx] = 1.0

            word_query = lexical_models["word_vectorizer"].transform([query_text])
            char_query = lexical_models["char_vectorizer"].transform([query_text])
            word_scores_arr = cosine_similarity(word_query, lexical_models["word_matrix"]).ravel()
            char_scores_arr = cosine_similarity(char_query, lexical_models["char_matrix"]).ravel()

            word_scores = {idx: float(word_scores_arr[idx]) for idx in top_indices(word_scores_arr, args.lexical_pool_size)}
            char_scores = {idx: float(char_scores_arr[idx]) for idx in top_indices(char_scores_arr, args.lexical_pool_size)}

            dense_scores = {}
            if embedding_client is not None and dense_matrix is not None:
                query_embedding = embed_texts(embedding_client, args.embedding_model, [query_text])[0]
                query_norm_value = np.linalg.norm(query_embedding)
                if query_norm_value != 0:
                    query_embedding = query_embedding / query_norm_value
                    dense_scores_arr = dense_matrix @ query_embedding
                    dense_scores = {
                        idx: float(dense_scores_arr[idx]) for idx in top_indices(dense_scores_arr, args.lexical_pool_size)
                    }

            word_scores = normalize_score_map(word_scores)
            char_scores = normalize_score_map(char_scores)
            dense_scores = normalize_score_map(dense_scores)

            candidate_indices = set(word_scores) | set(char_scores) | set(dense_scores) | set(exact_scores)
            if not candidate_indices:
                continue

            candidate_rows = []
            for idx in candidate_indices:
                row = ontology_terms.iloc[idx]
                combined_score = (
                    args.word_weight * word_scores.get(idx, 0.0) +
                    args.char_weight * char_scores.get(idx, 0.0) +
                    args.dense_weight * dense_scores.get(idx, 0.0) +
                    args.exact_boost * exact_scores.get(idx, 0.0)
                )
                candidate_rows.append(
                    {
                        "sample_id": sample_row.sample_id,
                        "target_ontology": sample_row.target_ontology,
                        "term_id": row["term_id"],
                        "label": row["label"],
                        "definition": row.get("definition", ""),
                        "parent_labels": row.get("parent_labels", ""),
                        "word_score": word_scores.get(idx, 0.0),
                        "char_score": char_scores.get(idx, 0.0),
                        "dense_score": dense_scores.get(idx, 0.0),
                        "exact_match": bool(exact_scores.get(idx, 0.0)),
                        "combined_score": float(combined_score),
                    }
                )

            candidates_df = pd.DataFrame(candidate_rows).sort_values("combined_score", ascending=False).head(args.candidate_k).reset_index(drop=True)
            reranked_term_ids: List[str] = candidates_df["term_id"].tolist()
            predicted_term_id = reranked_term_ids[0] if reranked_term_ids else ""
            reranker_confidence = None

            if reranker_client is not None and args.reranker_model and not candidates_df.empty:
                try:
                    best_term_id, confidence, ranked_term_ids = rerank_with_llm(
                        reranker_client,
                        args.reranker_model,
                        pd.Series(sample_row._asdict()),
                        candidates_df,
                        args.reranker_temperature,
                    )
                    if ranked_term_ids:
                        reranked_term_ids = ranked_term_ids
                    if best_term_id:
                        predicted_term_id = best_term_id
                    reranker_confidence = confidence
                except Exception as exc:
                    print(f"Warning: reranker failed for sample {sample_row.sample_id}: {exc}")

            for rank, candidate in enumerate(reranked_term_ids, start=1):
                match = candidates_df[candidates_df["term_id"] == candidate]
                if match.empty:
                    continue
                row_dict = match.iloc[0].to_dict()
                row_dict["rank"] = rank
                all_candidate_rows.append(row_dict)

            predicted_row = candidates_df[candidates_df["term_id"] == predicted_term_id]
            if predicted_row.empty and not candidates_df.empty:
                predicted_row = candidates_df.head(1)
                predicted_term_id = predicted_row.iloc[0]["term_id"]

            predicted_label = predicted_row.iloc[0]["label"] if not predicted_row.empty else ""
            top_score = float(predicted_row.iloc[0]["combined_score"]) if not predicted_row.empty else np.nan
            all_predictions.append(
                {
                    "sample_id": sample_row.sample_id,
                    "target_ontology": sample_row.target_ontology,
                    "mention_text": sample_row.mention_text,
                    "context_text": sample_row.context_text,
                    "gold_term_id": getattr(sample_row, "term_id", ""),
                    "predicted_term_id": predicted_term_id,
                    "predicted_label": predicted_label,
                    "top_score": top_score,
                    "reranker_confidence": reranker_confidence,
                    "topk_term_ids": "||".join(reranked_term_ids),
                }
            )

    prediction_columns = [
        "sample_id",
        "target_ontology",
        "mention_text",
        "context_text",
        "gold_term_id",
        "predicted_term_id",
        "predicted_label",
        "top_score",
        "reranker_confidence",
        "topk_term_ids",
    ]
    candidate_columns = [
        "sample_id",
        "target_ontology",
        "term_id",
        "label",
        "definition",
        "parent_labels",
        "word_score",
        "char_score",
        "dense_score",
        "exact_match",
        "combined_score",
        "rank",
    ]
    predictions_df = pd.DataFrame(all_predictions, columns=prediction_columns)
    candidate_df = pd.DataFrame(all_candidate_rows, columns=candidate_columns)

    predictions_path = os.path.join(output_dir, "ontology_mapping_predictions.tsv")
    candidates_path = os.path.join(output_dir, "ontology_mapping_candidates.tsv")
    save_tabular(predictions_df, predictions_path)
    save_tabular(candidate_df, candidates_path)

    metrics = compute_metrics(predictions_df)
    metrics_path = os.path.join(output_dir, "ontology_mapping_metrics.json")
    write_json(metrics_path, metrics)

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved candidate table to {candidates_path}")
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No gold_term_id values found; metrics file is empty.")


if __name__ == "__main__":
    main()
