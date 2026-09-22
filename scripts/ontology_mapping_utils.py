#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities shared by ontology mapping scripts."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for ontology mapping utilities. "
    ) from exc


DEFAULT_ENVO_FIELDS = [
    "env_biome",
    "env_feature",
    "env_material",
    "environment_biome",
    "environment_feature",
    "environment_material",
    "isolation_source",
    "sample_isolation_source",
    "habitat",
    "geo_loc_name",
    "geographic_location",
    "sample_scientific_name",
    "sample_title",
    "study_title",
    "study_abstract",
    "description",
    "title",
]

DEFAULT_UBERON_FIELDS = [
    "host",
    "host_body_habitat",
    "host_body_product",
    "host_body_site",
    "host_tissue_sampled",
    "body_site",
    "body_habitat",
    "body_product",
    "organism_part",
    "anatomical_site",
    "anatomical_region",
    "sample_scientific_name",
    "isolation_source",
    "sample_isolation_source",
    "sample_title",
    "study_title",
    "study_abstract",
    "description",
    "title",
]

GENERIC_CONTEXT_FIELDS = [
    "keywords",
    "sample_name",
    "sample_alias",
    "sample_biosamplemodel",
    "study",
    "study_title",
    "study_abstract",
    "description",
    "title",
]

ONTOLOGY_VALUE_PATTERN = re.compile(
    r"^(?P<label>.*?)\s*\[(?P<prefix>[A-Za-z]+):(?P<local_id>[^\]]+)\]\s*$"
)


def normalize_text(text: str) -> str:
    """Normalize text for robust lexical matching."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[_/|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def split_multivalue_text(value: str) -> List[str]:
    """Split text on common metadata separators while preserving useful phrases."""
    text = str(value).strip()
    if not text:
        return []
    if ";" in text:
        pieces = [part.strip() for part in text.split(";")]
    elif "||" in text:
        pieces = [part.strip() for part in text.split("||")]
    else:
        pieces = [text]
    return [piece for piece in pieces if piece]


def parse_ontology_annotated_value(value: str) -> Tuple[str, str, str] | None:
    """Parse values like 'soil [ENVO:00001998]'."""
    match = ONTOLOGY_VALUE_PATTERN.match(str(value).strip())
    if not match:
        return None
    label = match.group("label").strip()
    prefix = match.group("prefix").upper()
    local_id = match.group("local_id").strip()
    return label, f"{prefix}_{local_id}", prefix


def contains_ontology_annotation(value: str) -> bool:
    return parse_ontology_annotated_value(value) is not None


def parse_metadata_text(metadata_text: str) -> Dict[str, str]:
    """Parse key=value metadata text into a normalized dictionary."""
    parsed = {}
    for raw_line in str(metadata_text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = normalize_text(key)
        value = value.strip()
        if key and value:
            parsed[key] = value
    return parsed


def select_fields_for_ontology(target_ontology: Optional[str]) -> List[str]:
    target = normalize_text(target_ontology)
    if target == "envo":
        return DEFAULT_ENVO_FIELDS
    if target == "uberon":
        return DEFAULT_UBERON_FIELDS
    return unique_preserve_order(DEFAULT_ENVO_FIELDS + DEFAULT_UBERON_FIELDS)


def build_mention_and_context(
    metadata_dict: Dict[str, str],
    target_ontology: Optional[str] = None,
    preferred_fields: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """
    Build a compact mention text plus broader context from parsed metadata.
    """
    if preferred_fields:
        field_order = [normalize_text(field) for field in preferred_fields]
    else:
        field_order = [normalize_text(field) for field in select_fields_for_ontology(target_ontology)]

    generic_fields = [normalize_text(field) for field in GENERIC_CONTEXT_FIELDS]

    mention_parts = []
    used_keys = set()
    for field in field_order:
        if field in metadata_dict:
            mention_parts.append(metadata_dict[field])
            used_keys.add(field)

    if not mention_parts:
        fallback_keys = [
            key for key in metadata_dict
            if any(token in key for token in ("source", "site", "tissue", "habitat", "biome", "feature", "material", "host"))
        ]
        for key in fallback_keys[:6]:
            mention_parts.append(metadata_dict[key])
            used_keys.add(key)

    context_parts = list(mention_parts)
    for field in generic_fields + field_order:
        if field in metadata_dict and field not in used_keys:
            context_parts.append(metadata_dict[field])
            used_keys.add(field)

    if not context_parts:
        context_parts = [f"{key}={value}" for key, value in list(metadata_dict.items())[:12]]

    mention_text = " | ".join(unique_preserve_order(part.strip() for part in mention_parts if part.strip()))
    context_text = " | ".join(unique_preserve_order(part.strip() for part in context_parts if part.strip()))
    return mention_text, context_text


def sample_id_to_metadata_path(split_dir: str, sample_id: str) -> str:
    subdir = f"dir_{sample_id[-3:]}"
    return os.path.join(split_dir, subdir, f"{sample_id}_clean.txt")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_api_key(api_key_path: str) -> str:
    with open(api_key_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tabular(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def save_tabular(df: pd.DataFrame, path: str) -> None:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def stratified_sample(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    max_samples_per_group: Optional[int],
    max_total_samples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Sample rows while keeping term/ontology diversity."""
    rng = random.Random(seed)
    sampled_parts = []

    if max_samples_per_group is None:
        sampled = df.copy()
    else:
        for _, group_df in df.groupby(list(group_cols), dropna=False, sort=False):
            if len(group_df) <= max_samples_per_group:
                sampled_parts.append(group_df)
                continue
            chosen_idx = rng.sample(list(group_df.index), max_samples_per_group)
            sampled_parts.append(group_df.loc[chosen_idx])
        sampled = pd.concat(sampled_parts, ignore_index=False).sort_index()

    if max_total_samples is not None and len(sampled) > max_total_samples:
        chosen_idx = rng.sample(list(sampled.index), max_total_samples)
        sampled = sampled.loc[chosen_idx].sort_index()

    return sampled.reset_index(drop=True)


def build_term_text(label: str, synonyms: Sequence[str], definition: str, parent_labels: Sequence[str]) -> str:
    pieces = [str(label).strip()]
    if synonyms:
        pieces.append("synonyms: " + "; ".join(str(s).strip() for s in synonyms if str(s).strip()))
    if definition:
        pieces.append("definition: " + str(definition).strip())
    if parent_labels:
        pieces.append("parents: " + "; ".join(str(p).strip() for p in parent_labels if str(p).strip()))
    return " | ".join(piece for piece in pieces if piece and piece != "synonyms: " and piece != "parents: ")


def safe_literal_list(value: object) -> List[str]:
    """Accept JSON-list strings, pipe-separated strings, or iterables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    splitter = "||" if "||" in text else "|"
    return [part.strip() for part in text.split(splitter) if part.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv_rows(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
