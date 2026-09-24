#!/usr/bin/env python3
"""
Build the labelled data set: MicrobeAtlas free-text metadata (input) paired with
Metalog's curated ENVO/Uberon terms (labels), one row per MicrobeAtlas sample.

Linking: a MicrobeAtlas record is linked to a Metalog sample when the record id
(SRS/ERS/DRS) or any BioSample accession (SAMN/SAMEA/SAMD...) inside the record
equals Metalog's `spire_sample_name`.

Output columns: sample_id, spire_sample_name, study_code, domain,
                biome, feature, material, text

python scripts/build_metalog_training_set.py \
  --metalog_dir ~/MicrobeAtlasProject/metalog \
  --sample_info ~/MicrobeAtlasProject/sample.info.gz \
  --ontology_terms ~/MicrobeAtlasProject/ontology_terms.tsv.gz \
  --output ~/MicrobeAtlasProject/metalog/metalog_training_set.tsv.gz
"""

import argparse
import glob
import gzip
import os
import re

import pandas as pd

SLOTS = {"environment_biome": "biome", "environment_feature": "feature", "environment_material": "material"}
GOLD_VALUE = re.compile(r"\[(ENVO|UBERON):(\d+)\]")  # "soil [ENVO:00001998]"
CODE_IN_TEXT = re.compile(r"\b(ENVO|UBERON)[:_](\d{7,8})\b")  # ENVO:00001998 or ENVO_00001998
ACCESSION = re.compile(r"\b(SAM[END][A-Z]?\d+|[SED]RS\d+)\b")
MISSING = {"", "na", "n/a", "nan", "none", "null", "-", "missing", "unknown", "unspecified",
           "not applicable", "not collected", "not provided", "not available", "not determined"}
# keys that are identifiers, dates or coordinates: noise for text matching (and they leak study identity)
DROP_KEYS = re.compile(r"^(experiment|run)|^study$|^sample name$|alias|xref|link|insdc|accession|center|broker"
                       r"|submitter|checklist|library|date|time|update|public|latitude|longitude|lat_lon"
                       r"|taxon_id|_id$| id$|subject|patient|participant|replicate")


def load_metalog_labels(metalog_dir, valid_terms):
    """One row per Metalog sample: spire_sample_name, study_code, domain + one term id per slot."""
    frames = []
    for path in sorted(glob.glob(os.path.join(metalog_dir, "*_all_long_*.tsv.gz"))):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df = df[df["metadata_item"].isin(["spire_sample_name", "study_code", *SLOTS])]
        wide = df.pivot_table(index="sample_alias", columns="metadata_item", values="value", aggfunc="first")
        wide["domain"] = os.path.basename(path).split("_")[0]
        frames.append(wide.reset_index())
        print(f"{os.path.basename(path)}: {len(wide)} samples")
    labels = pd.concat(frames, ignore_index=True).fillna("")

    for metalog_field, slot in SLOTS.items():
        match = labels[metalog_field].str.extract(GOLD_VALUE)
        labels[slot] = (match[0] + "_" + match[1]).fillna("")
        unusable = labels[slot].ne("") & ~labels[slot].isin(valid_terms)
        print(f"{slot}: {labels[slot].ne('').sum()} ENVO/UBERON labels, "
              f"{unusable.sum()} dropped because obsolete or not in the term index")
        labels.loc[unusable, slot] = ""
    # a few accessions appear under two Metalog aliases (e.g. in two studies): keep the first
    return labels[labels["spire_sample_name"] != ""].drop_duplicates("spire_sample_name")


def iter_sample_info(path):
    """Yield (sample_id, lines) for each '>SAMPLE' record of sample.info(.gz)."""
    sample_id, lines = None, []
    with gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if sample_id:
                    yield sample_id, lines
                sample_id, lines = line[1:].strip(), []
            else:
                lines.append(line.rstrip("\n"))
    if sample_id:
        yield sample_id, lines


def record_to_text(lines, code_to_label, max_chars):
    """Clean one metadata record into 'key: value; key: value; ...'."""
    kept = []
    for line in lines:
        key, _, value = line.partition("=")
        key = re.sub(r"^(sample|study)_", "", key.strip()).lower()
        value = value.strip()
        if not key or DROP_KEYS.search(key) or value.lower().strip(" .") in MISSING:
            continue
        value = CODE_IN_TEXT.sub(lambda m: code_to_label.get(f"{m[1]}_{m[2]}", m[0]), value)
        kept.append(f"{key}: {value}")
    return "; ".join(dict.fromkeys(kept))[:max_chars]  # dict.fromkeys drops duplicate lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metalog_dir", required=True)
    parser.add_argument("--sample_info", required=True, help="MicrobeAtlas sample.info(.gz)")
    parser.add_argument("--ontology_terms", required=True, help="TSV from build_ontology_term_index.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate each sample text")
    parser.add_argument("--include_unlabeled", action="store_true",
                        help="Also write every non-linked MicrobeAtlas sample (empty labels), for prediction")
    args = parser.parse_args()

    terms = pd.read_csv(args.ontology_terms, sep="\t", keep_default_na=False)
    code_to_label = dict(zip(terms["term_id"], terms["label"]))  # obsolete included: it is input text
    valid_terms = set(terms.loc[terms["obsolete"].astype(str) != "True", "term_id"])

    labels = load_metalog_labels(os.path.expanduser(args.metalog_dir), valid_terms)
    by_accession = labels.set_index("spire_sample_name")[["study_code", "domain", *SLOTS.values()]].to_dict("index")

    rows, n_records = [], 0
    for sample_id, lines in iter_sample_info(os.path.expanduser(args.sample_info)):
        n_records += 1
        accessions = [sample_id] + ACCESSION.findall(" ".join(lines))
        spire = next((acc for acc in accessions if acc in by_accession), None)
        if spire is None and not args.include_unlabeled:
            continue
        label = by_accession.get(spire, {})
        rows.append({"sample_id": sample_id, "spire_sample_name": spire or "",
                     "study_code": label.get("study_code", ""), "domain": label.get("domain", ""),
                     **{slot: label.get(slot, "") for slot in SLOTS.values()},
                     "text": record_to_text(lines, code_to_label, args.max_chars)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.expanduser(args.output), sep="\t", index=False)
    linked = out["spire_sample_name"].ne("")
    print(f"{n_records} MicrobeAtlas records, {linked.sum()} linked to "
          f"{out.loc[linked, 'spire_sample_name'].nunique()} Metalog samples")
    print(out[linked].groupby("domain")[list(SLOTS.values())].agg(lambda s: s.ne("").sum()))
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
