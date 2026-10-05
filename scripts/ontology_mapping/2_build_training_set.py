#!/usr/bin/env python3
"""
Step 2: build the labelled data set: MicrobeAtlas free-text metadata (input) paired with
Metalog's curated ENVO/Uberon terms (labels), one row per MicrobeAtlas sample.

Linking: a MicrobeAtlas record is linked to a Metalog sample when the record id
(SRS/ERS/DRS) or any BioSample accession (SAMN/SAMEA/SAMD...) inside the record
equals Metalog's `spire_sample_name`.

Output columns: sample_id, spire_sample_name, study_code, domain,
                biome, feature, material (term ids, "" if none), text

`text` is the record as 'key: value; key: value', with sample_/study_ key prefixes
removed, identifier / date / coordinate keys and missing values dropped, ontology
codes replaced by their label, truncated to --max_chars. It is the input of the
`tfidf` features only; the GPT keywords / sub-biomes are extracted upstream.

python 2_build_training_set.py \
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


def snapshot_files(metalog_dir, date=None):
    """The four Metalog long tables from one snapshot (latest by default).

    Selecting a snapshot matters when old and new downloads share a directory: concatenating both
    would silently let the alphabetically first version win for duplicate sample accessions.
    """
    found = glob.glob(os.path.join(metalog_dir, "*_all_long_*.tsv.gz"))
    dates = sorted({m.group(1) for p in found if (m := re.search(r"_(\d{4}-\d{2}-\d{2})\.tsv\.gz$", p))})
    if not dates:
        raise SystemExit(f"no *_all_long_<date>.tsv.gz files in {metalog_dir}")
    date = date or dates[-1]
    files = [os.path.join(metalog_dir, f"{domain}_all_long_{date}.tsv.gz")
             for domain in ("animal", "environmental", "human", "ocean")]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"incomplete Metalog snapshot {date}; missing {missing}")
    return date, files


def load_metalog_labels(files, valid_terms):
    """One row per Metalog sample: accession, study, domain and one term id per slot."""
    frames = []
    for file_path in files:
        df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
        df = df[df["metadata_item"].isin(["spire_sample_name", "study_code", *SLOTS])]
        wide = df.pivot_table(index="sample_alias", columns="metadata_item", values="value", aggfunc="first")
        wide["domain"] = os.path.basename(file_path).split("_")[0]
        frames.append(wide.reset_index())
        print(f"{os.path.basename(file_path)}: {len(wide)} samples")
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
    parser.add_argument("--ontology_terms", required=True, help="TSV from 1_build_term_index.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--date", help="Metalog snapshot date YYYY-MM-DD (default: latest complete snapshot)")
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate each sample text")
    args = parser.parse_args()

    terms = pd.read_csv(args.ontology_terms, sep="\t", keep_default_na=False)
    code_to_label = dict(zip(terms["term_id"], terms["label"]))  # obsolete included: it is input text
    valid_terms = set(terms.loc[terms["obsolete"].astype(str) != "True", "term_id"])

    snapshot_date, files = snapshot_files(os.path.expanduser(args.metalog_dir), args.date)
    print(f"Metalog snapshot {snapshot_date}")
    labels = load_metalog_labels(files, valid_terms)
    by_accession = labels.set_index("spire_sample_name")[["study_code", "domain", *SLOTS.values()]].to_dict("index")

    rows, n_records = [], 0
    for sample_id, lines in iter_sample_info(os.path.expanduser(args.sample_info)):
        n_records += 1
        accessions = [sample_id] + ACCESSION.findall(" ".join(lines))
        spire = next((acc for acc in accessions if acc in by_accession), None)
        if spire is None:
            continue
        label = by_accession[spire]
        rows.append({"sample_id": sample_id, "spire_sample_name": spire,
                     "study_code": label["study_code"], "domain": label["domain"],
                     **{slot: label[slot] for slot in SLOTS.values()},
                     "text": record_to_text(lines, code_to_label, args.max_chars)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.expanduser(args.output), sep="\t", index=False)
    print(f"{n_records} MicrobeAtlas records, {len(out)} linked to {out['spire_sample_name'].nunique()} Metalog samples")
    print("non-empty labels per domain:")
    print(out.groupby("domain")[list(SLOTS.values())].agg(lambda s: s.ne("").sum()))
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
