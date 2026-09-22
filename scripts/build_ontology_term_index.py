#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a flat ontology term table for retrieval experiments.

Example:
python scripts/build_ontology_term_index.py \
  --ontologies ENVO UBERON \
  --output_dir . \
  --output_prefix ontology_terms
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Dict, Iterable, List, Tuple

try:
    import pandas as pd
    import pronto
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for build_ontology_term_index.py. "
    ) from exc

from ontology_mapping_utils import build_term_text, save_tabular, unique_preserve_order


DEFAULT_SOURCES = {
    "ENVO": "http://purl.obolibrary.org/obo/envo.owl",
    "UBERON": "http://purl.obolibrary.org/obo/uberon.owl",
    "FOODON": "http://purl.obolibrary.org/obo/foodon.owl",
    "PO": "http://purl.obolibrary.org/obo/po.owl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ontology term index tables.")
    parser.add_argument("--ontologies", nargs="+", required=True, help="Ontology prefixes such as ENVO UBERON")
    parser.add_argument("--output_dir", default=".", help="Directory for output tables")
    parser.add_argument("--output_prefix", default="ontology_terms", help="Output prefix without extension")
    parser.add_argument(
        "--output_format",
        default="tsv",
        choices=["tsv", "csv"],
        help="Primary table output format",
    )
    parser.add_argument(
        "--ontology_source",
        action="append",
        default=[],
        help="Optional override of form PREFIX=/path/or/url",
    )
    return parser.parse_args()


def parse_source_overrides(pairs: Iterable[str]) -> Dict[str, str]:
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --ontology_source value: {pair}")
        prefix, source = pair.split("=", 1)
        overrides[prefix.strip().upper()] = source.strip()
    return overrides


def materialize_source(source: str) -> Tuple[str, str]:
    """Return local path plus temp path for cleanup."""
    if os.path.exists(os.path.expanduser(source)):
        return os.path.expanduser(source), ""

    response = requests.get(source, timeout=180)
    response.raise_for_status()

    suffix = ".owl" if source.endswith(".owl") else ".obo"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(response.content)
    handle.flush()
    handle.close()
    return handle.name, handle.name


def get_direct_parent_terms(term: pronto.Term) -> List[pronto.Term]:
    try:
        parents = list(term.superclasses(distance=1, with_self=False))
    except TypeError:
        parents = list(term.superclasses(distance=1))
        parents = [parent for parent in parents if str(parent.id) != str(term.id)]
    return [parent for parent in parents if getattr(parent, "name", None)]


def extract_rows(ontology_prefix: str, source: str) -> List[dict]:
    local_path, temp_path = materialize_source(source)
    try:
        ontology = pronto.Ontology(local_path)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    rows = []
    expected_prefix = ontology_prefix.upper() + ":"

    for term in ontology.terms():
        if term.obsolete or not term.name:
            continue

        term_id = str(term.id)
        if not term_id.startswith(expected_prefix):
            continue

        definition = str(term.definition).strip() if getattr(term, "definition", None) else ""
        synonyms = unique_preserve_order(str(syn.description).strip() for syn in term.synonyms if str(syn.description).strip())
        parents = get_direct_parent_terms(term)
        parent_ids = [str(parent.id).replace(":", "_") for parent in parents]
        parent_labels = [str(parent.name).strip() for parent in parents if str(parent.name).strip()]

        rows.append(
            {
                "ontology": ontology_prefix.upper(),
                "term_id": term_id.replace(":", "_"),
                "label": str(term.name).strip(),
                "definition": definition,
                "synonyms": "||".join(synonyms),
                "parent_ids": "||".join(parent_ids),
                "parent_labels": "||".join(parent_labels),
                "text_for_embedding": build_term_text(term.name, synonyms, definition, parent_labels),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    overrides = parse_source_overrides(args.ontology_source)
    all_rows = []

    for ontology_prefix in args.ontologies:
        ontology_prefix = ontology_prefix.upper()
        source = overrides.get(ontology_prefix, DEFAULT_SOURCES.get(ontology_prefix))
        if source is None:
            raise ValueError(f"No source configured for ontology '{ontology_prefix}'")
        print(f"Loading {ontology_prefix} from {source}")
        rows = extract_rows(ontology_prefix, source)
        print(f"  -> extracted {len(rows)} terms")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).sort_values(["ontology", "term_id"]).reset_index(drop=True)
    table_ext = "tsv" if args.output_format == "tsv" else "csv"
    output_table = os.path.join(output_dir, f"{args.output_prefix}.{table_ext}")
    save_tabular(df, output_table)
    print(f"Saved {len(df)} ontology terms to {output_table}")

    jsonl_path = os.path.join(output_dir, f"{args.output_prefix}.jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved JSONL copy to {jsonl_path}")


if __name__ == "__main__":
    main()
