#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert Metalog long-format exports into sample-level ontology mapping examples.

The output is designed to feed directly into map_metadata_to_ontology.py.
It extracts ENVO and Uberon gold terms from curated Metalog fields while
building mention/context text from other metadata fields to reduce leakage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from collections import Counter, defaultdict
from glob import glob
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for prepare_metalog_for_ontology_mapping.py. "
        "Install the required packages in your Python environment and rerun the script."
    ) from exc

from ontology_mapping_utils import (
    parse_ontology_annotated_value,
    save_tabular,
    stratified_sample,
    unique_preserve_order,
    write_json,
)


TARGET_FIELDS = ["environment_biome", "environment_feature", "environment_material"]
FIELDS_TO_EXCLUDE_FROM_MENTIONS = set(TARGET_FIELDS) | {
    "unchanged_environment_biome",
    "environment_material_old",
    "environment_feature_old",
}

SLOT_FIELD_PRIORITIES = {
    "environment_biome": [
        "environmental_package",
        "sample_title",
        "sample_description",
        "description",
        "location",
        "geographic_location",
        "collection_site",
        "sampling_site",
        "site",
        "site_description",
        "host",
        "host_scientific_name",
    ],
    "environment_feature": [
        "collection_site",
        "collection_site_abbreviation",
        "sample_site",
        "collection_site_abbreviation",
        "collection_location",
        "sampling_site",
        "sampling_station",
        "sampling_campaign",
        "site",
        "site_name",
        "site_description",
        "location",
        "geographic_location",
        "sample_title",
        "sample_description",
    ],
    "environment_material": [
        "isolation_source",
        "specific_material",
        "environment_(material)",
        "sample_material_processing",
        "body_site",
        "bodysite",
        "body_site",
        "tissue",
        "tissue_type",
        "sample_title",
        "sample_description",
        "description",
    ],
}

ONTOLOGY_PRIORITIES = {
    "ENVO": [
        "isolation_source",
        "specific_material",
        "environment_(material)",
        "collection_site",
        "sampling_site",
        "site",
        "site_name",
        "location",
        "geographic_location",
        "sample_title",
        "sample_description",
        "description",
        "environmental_package",
        "host",
        "host_scientific_name",
    ],
    "UBERON": [
        "body_site",
        "bodysite",
        "sampling_site",
        "detailed_sampling_site",
        "skin_site_hmp",
        "skin_site_short",
        "skin_site_type",
        "tissue",
        "tissue_type",
        "sample_title",
        "sample_description",
        "description",
        "isolation_source",
        "host",
        "host_scientific_name",
    ],
}

GENERIC_FALLBACK_FIELDS = [
    "sample_title",
    "sample_description",
    "description",
    "location",
    "geographic_location",
    "host",
    "host_scientific_name",
    "isolation_source",
    "body_site",
    "bodysite",
    "tissue",
    "tissue_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Metalog data for ontology mapping.")
    parser.add_argument("--metalog_dir", default="metalog", help="Directory with Metalog long-format tables")
    parser.add_argument(
        "--file_patterns",
        nargs="+",
        default=["*_all_long_*.tsv", "*_all_long_*.tsv.gz"],
        help="Glob patterns relative to --metalog_dir",
    )
    parser.add_argument(
        "--output_file",
        default="metalog/metalog_ontology_examples.tsv",
        help="TSV/CSV file ready for map_metadata_to_ontology.py",
    )
    parser.add_argument(
        "--summary_file",
        default="metalog/metalog_ontology_examples_summary.json",
        help="JSON summary output",
    )
    parser.add_argument(
        "--allowed_ontologies",
        nargs="+",
        default=["ENVO", "UBERON"],
        help="Ontology prefixes to keep",
    )
    parser.add_argument("--max_samples_per_term", type=int, default=50, help="Optional per-term cap")
    parser.add_argument("--max_total_samples", type=int, default=5000, help="Optional overall cap")
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def open_maybe_gzip(path: str):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else open(path, "r", encoding="utf-8", newline="")


def normalize_field_name(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def discover_files(metalog_dir: str, patterns: Sequence[str]) -> List[str]:
    matches = []
    for pattern in patterns:
        matches.extend(glob(os.path.join(metalog_dir, pattern)))
    return sorted(set(matches))


def grouped_sample_rows(path: str) -> Iterator[Tuple[str, List[dict]]]:
    with open_maybe_gzip(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        current_alias = None
        current_rows: List[dict] = []
        for row in reader:
            sample_alias = row["sample_alias"]
            if current_alias is None:
                current_alias = sample_alias
            if sample_alias != current_alias:
                yield current_alias, current_rows
                current_alias = sample_alias
                current_rows = []
            current_rows.append(row)
        if current_alias is not None and current_rows:
            yield current_alias, current_rows


def collect_sample_metadata(rows: Sequence[dict]) -> Dict[str, List[str]]:
    metadata: DefaultDict[str, List[str]] = defaultdict(list)
    for row in rows:
        field = normalize_field_name(row["metadata_item"])
        value = str(row["value"]).strip()
        if value:
            metadata[field].append(value)
    return dict(metadata)


def unique_values(values: Iterable[str]) -> List[str]:
    return unique_preserve_order(value.strip() for value in values if str(value).strip())


def value_is_leaky(value: str, gold_term_id: str, gold_label: str) -> bool:
    value = str(value).strip()
    if not value:
        return True
    if parse_ontology_annotated_value(value) is not None:
        return True
    if value.lower() == gold_label.lower():
        return True
    if gold_label.lower() in value.lower() and len(value.split()) <= len(gold_label.split()) + 1:
        return True
    if gold_term_id.replace("_", ":") in value:
        return True
    return False


def pick_source_values(
    metadata: Dict[str, List[str]],
    field_order: Sequence[str],
    gold_term_id: str,
    gold_label: str,
    max_values: int,
) -> Tuple[List[str], List[str]]:
    selected_values = []
    selected_fields = []
    for field in field_order:
        if field in FIELDS_TO_EXCLUDE_FROM_MENTIONS:
            continue
        values = metadata.get(field, [])
        for value in values:
            if value_is_leaky(value, gold_term_id, gold_label):
                continue
            selected_values.append(value)
            selected_fields.append(field)
            if len(selected_values) >= max_values:
                return unique_values(selected_values), unique_values(selected_fields)
    return unique_values(selected_values), unique_values(selected_fields)


def build_mention_and_context_for_gold(
    metadata: Dict[str, List[str]],
    target_field: str,
    target_ontology: str,
    gold_term_id: str,
    gold_label: str,
) -> Tuple[str, str, List[str]]:
    slot_fields = [normalize_field_name(field) for field in SLOT_FIELD_PRIORITIES.get(target_field, [])]
    ontology_fields = [normalize_field_name(field) for field in ONTOLOGY_PRIORITIES.get(target_ontology, [])]
    fallback_fields = [normalize_field_name(field) for field in GENERIC_FALLBACK_FIELDS]

    primary_order = unique_values(slot_fields + ontology_fields)
    mention_values, mention_fields = pick_source_values(metadata, primary_order, gold_term_id, gold_label, max_values=4)

    remaining_order = unique_values(primary_order + fallback_fields + list(metadata.keys()))
    context_values, context_fields = pick_source_values(metadata, remaining_order, gold_term_id, gold_label, max_values=12)

    mention_text = " | ".join(mention_values)
    context_text = " | ".join(context_values)
    return mention_text, context_text, unique_values(mention_fields + context_fields)


def iter_gold_examples_for_sample(
    sample_id: str,
    sample_alias: str,
    domain: str,
    source_file: str,
    metadata: Dict[str, List[str]],
    allowed_ontologies: set[str],
) -> Iterator[dict]:
    spire_sample_name = metadata.get("spire_sample_name", [""])
    study_code = metadata.get("study_code", [""])
    geographic_location = metadata.get("geographic_location", [""])

    for target_field in TARGET_FIELDS:
        gold_values = unique_values(metadata.get(target_field, []))
        for gold_value in gold_values:
            parsed = parse_ontology_annotated_value(gold_value)
            if parsed is None:
                continue

            gold_label, term_id, ontology_prefix = parsed
            if ontology_prefix not in allowed_ontologies:
                continue

            mention_text, context_text, source_fields = build_mention_and_context_for_gold(
                metadata,
                target_field=target_field,
                target_ontology=ontology_prefix,
                gold_term_id=term_id,
                gold_label=gold_label,
            )

            if not mention_text and not context_text:
                continue

            yield {
                "sample_id": sample_id,
                "sample_alias": sample_alias,
                "spire_sample_name": spire_sample_name[0] if spire_sample_name else "",
                "study_code": study_code[0] if study_code else "",
                "domain": domain,
                "source_file": os.path.basename(source_file),
                "target_ontology": ontology_prefix,
                "target_field": target_field,
                "term_id": term_id,
                "term_label": gold_label,
                "gold_value": gold_value,
                "mention_text": mention_text,
                "context_text": context_text,
                "source_fields": "||".join(source_fields),
                "geographic_location": geographic_location[0] if geographic_location else "",
            }


def build_dataset(file_paths: Sequence[str], allowed_ontologies: set[str]) -> Tuple[pd.DataFrame, dict]:
    rows = []
    per_ontology = Counter()
    per_field = Counter()
    per_domain = Counter()
    per_term = Counter()
    files_seen = {}

    for path in file_paths:
        domain = os.path.basename(path).split("_", 1)[0]
        files_seen[os.path.basename(path)] = {"domain": domain, "samples": 0, "examples": 0}

        for sample_alias, sample_rows in grouped_sample_rows(path):
            files_seen[os.path.basename(path)]["samples"] += 1
            metadata = collect_sample_metadata(sample_rows)
            sample_id = metadata.get("spire_sample_name", [sample_alias])[0] or sample_alias

            for example in iter_gold_examples_for_sample(
                sample_id=sample_id,
                sample_alias=sample_alias,
                domain=domain,
                source_file=path,
                metadata=metadata,
                allowed_ontologies=allowed_ontologies,
            ):
                rows.append(example)
                per_ontology[example["target_ontology"]] += 1
                per_field[(example["target_ontology"], example["target_field"])] += 1
                per_domain[(example["domain"], example["target_ontology"])] += 1
                per_term[(example["target_ontology"], example["term_id"])] += 1
                files_seen[os.path.basename(path)]["examples"] += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["sample_id", "sample_alias", "target_field", "term_id", "mention_text", "context_text"]
        ).reset_index(drop=True)

    summary = {
        "n_examples": int(len(df)),
        "n_unique_samples": int(df["sample_id"].nunique()) if not df.empty else 0,
        "n_unique_terms": int(df["term_id"].nunique()) if not df.empty else 0,
        "by_ontology": dict(per_ontology),
        "by_field": {f"{ontology}:{field}": count for (ontology, field), count in per_field.items()},
        "by_domain": {f"{domain}:{ontology}": count for (domain, ontology), count in per_domain.items()},
        "top_terms": [
            {"ontology": ontology, "term_id": term_id, "count": count}
            for (ontology, term_id), count in per_term.most_common(100)
        ],
        "files": files_seen,
    }
    return df, summary


def main() -> None:
    args = parse_args()
    metalog_dir = os.path.abspath(os.path.expanduser(args.metalog_dir))
    output_file = os.path.abspath(os.path.expanduser(args.output_file))
    summary_file = os.path.abspath(os.path.expanduser(args.summary_file))
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    allowed_ontologies = {ontology.upper() for ontology in args.allowed_ontologies}
    file_paths = discover_files(metalog_dir, args.file_patterns)
    if not file_paths:
        raise SystemExit(f"No Metalog files found in {metalog_dir} for patterns: {args.file_patterns}")

    print("Processing Metalog files:")
    for path in file_paths:
        print(f"  - {path}")

    df, summary = build_dataset(file_paths, allowed_ontologies)
    if df.empty:
        raise SystemExit("No ENVO/Uberon ontology examples could be extracted from the Metalog files.")

    sampled_df = stratified_sample(
        df,
        group_cols=["target_ontology", "term_id"],
        max_samples_per_group=args.max_samples_per_term,
        max_total_samples=args.max_total_samples,
        seed=args.seed,
    )

    save_tabular(sampled_df, output_file)

    sampled_summary = {
        **summary,
        "sampled_n_examples": int(len(sampled_df)),
        "sampled_n_unique_samples": int(sampled_df["sample_id"].nunique()),
        "sampled_n_unique_terms": int(sampled_df["term_id"].nunique()),
        "sampled_by_ontology": sampled_df.groupby("target_ontology").size().to_dict(),
        "sampled_by_field": {
            f"{ontology}:{field}": int(count)
            for (ontology, field), count in sampled_df.groupby(["target_ontology", "target_field"]).size().to_dict().items()
        },
    }
    write_json(summary_file, sampled_summary)

    print(f"Saved {len(sampled_df)} sampled examples to {output_file}")
    print(f"Unique samples: {sampled_df['sample_id'].nunique()}")
    print(f"Unique terms: {sampled_df['term_id'].nunique()}")
    print(f"Counts by ontology: {sampled_df.groupby('target_ontology').size().to_dict()}")
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
