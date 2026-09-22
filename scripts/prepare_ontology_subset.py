#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a small labeled subset for ontology mapping experiments.

Expected annotation columns:
- sample_id
- target_ontology
- term_id

Optional annotation columns:
- mention_text
- context_text
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for prepare_ontology_subset.py. "
    ) from exc

from ontology_mapping_utils import (
    build_mention_and_context,
    load_tabular,
    normalize_text,
    parse_metadata_text,
    sample_id_to_metadata_path,
    save_tabular,
    select_fields_for_ontology,
    stratified_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare small ontology mapping subsets.")
    parser.add_argument("--annotations", required=True, help="CSV/TSV with sample_id,target_ontology,term_id")
    parser.add_argument("--output_file", required=True, help="CSV/TSV file to write")
    parser.add_argument(
        "--split_metadata_dir",
        default=None,
        help="Directory containing *_clean.txt sample metadata files",
    )
    parser.add_argument(
        "--field_list",
        nargs="*",
        default=None,
        help="Optional preferred metadata fields to use for mention/context building",
    )
    parser.add_argument("--max_samples_per_term", type=int, default=25, help="Cap per ontology term")
    parser.add_argument("--max_total_samples", type=int, default=1000, help="Overall cap after stratification")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument(
        "--ontologies",
        nargs="*",
        default=None,
        help="Optional whitelist of ontologies to keep, e.g. ENVO UBERON",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    required = {"sample_id", "target_ontology", "term_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotations file is missing columns: {sorted(missing)}")


def build_text_columns(
    df: pd.DataFrame,
    split_metadata_dir: str,
    field_list: List[str] | None,
) -> pd.DataFrame:
    mention_values = []
    context_values = []
    metadata_paths = []
    missing_count = 0

    for row in df.itertuples(index=False):
        existing_mention = getattr(row, "mention_text", None)
        existing_context = getattr(row, "context_text", None)
        if pd.notna(existing_mention) and str(existing_mention).strip() and pd.notna(existing_context) and str(existing_context).strip():
            mention_values.append(str(existing_mention).strip())
            context_values.append(str(existing_context).strip())
            metadata_paths.append("")
            continue

        sample_id = str(row.sample_id).strip()
        metadata_path = sample_id_to_metadata_path(split_metadata_dir, sample_id)
        metadata_paths.append(metadata_path)

        if not os.path.exists(metadata_path):
            mention_values.append("")
            context_values.append("")
            missing_count += 1
            continue

        with open(metadata_path, "r", encoding="utf-8") as handle:
            parsed_metadata = parse_metadata_text(handle.read())

        preferred_fields = field_list or select_fields_for_ontology(getattr(row, "target_ontology"))
        mention_text, context_text = build_mention_and_context(
            parsed_metadata,
            target_ontology=getattr(row, "target_ontology"),
            preferred_fields=preferred_fields,
        )
        mention_values.append(mention_text)
        context_values.append(context_text)

    if missing_count:
        print(f"Warning: metadata not found for {missing_count} samples")

    out_df = df.copy()
    out_df["mention_text"] = mention_values
    out_df["context_text"] = context_values
    out_df["metadata_path"] = metadata_paths
    return out_df


def main() -> None:
    args = parse_args()
    df = load_tabular(os.path.abspath(os.path.expanduser(args.annotations)))
    validate_columns(df)

    if args.ontologies:
        allowed = {normalize_text(value).upper() for value in args.ontologies}
        df = df[df["target_ontology"].astype(str).str.upper().isin(allowed)].reset_index(drop=True)

    sampled_df = stratified_sample(
        df,
        group_cols=["target_ontology", "term_id"],
        max_samples_per_group=args.max_samples_per_term,
        max_total_samples=args.max_total_samples,
        seed=args.seed,
    )

    if args.split_metadata_dir:
        split_metadata_dir = os.path.abspath(os.path.expanduser(args.split_metadata_dir))
        sampled_df = build_text_columns(sampled_df, split_metadata_dir, args.field_list)
    else:
        if "mention_text" not in sampled_df.columns:
            sampled_df["mention_text"] = ""
        if "context_text" not in sampled_df.columns:
            sampled_df["context_text"] = ""

    sampled_df = sampled_df[
        sampled_df["mention_text"].astype(str).str.strip().ne("") |
        sampled_df["context_text"].astype(str).str.strip().ne("")
    ].reset_index(drop=True)

    output_file = os.path.abspath(os.path.expanduser(args.output_file))
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_tabular(sampled_df, output_file)

    summary = sampled_df.groupby("target_ontology").size().to_dict()
    print(f"Saved {len(sampled_df)} rows to {output_file}")
    print(f"Counts by ontology: {summary}")


if __name__ == "__main__":
    main()
