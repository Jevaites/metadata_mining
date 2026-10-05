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
import os
import re

import pandas as pd

from common import iter_sample_info, record_to_text

SLOTS = {"environment_biome": "biome", "environment_feature": "feature", "environment_material": "material"}
GOLD_VALUE = re.compile(r"\[(ENVO|UBERON):(\d+)\]")  # "soil [ENVO:00001998]" -> ENVO, 00001998
ACCESSION = re.compile(r"\b(SAM[END][A-Z]?\d+|[SED]RS\d+)\b")  # BioSample (SAMN/SAMEA/SAMD) or SRA sample


def load_metalog_labels(metalog_dir, valid_terms):
    """One row per Metalog sample: spire_sample_name, study_code, domain + one term id per slot.
    Example row: SAMEA5617776, Shao_2019_infants, human, biome "", feature ENVO_2100002 (intestine environment),
    material ENVO_00002003 (fecal material); the human biome is blanked: Metalog's ENVO:00009003 is obsolete."""
    frames = []
    for path in sorted(glob.glob(os.path.join(metalog_dir, "*_all_long_*.tsv.gz"))):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df = df[df["metadata_item"].isin(["spire_sample_name", "study_code", *SLOTS])]
        # long table (sample_alias, metadata_item, value) -> one row per sample, one column per item
        wide = df.pivot_table(index="sample_alias", columns="metadata_item", values="value", aggfunc="first")
        wide["domain"] = os.path.basename(path).split("_")[0]
        frames.append(wide.reset_index())
        print(f"{os.path.basename(path)}: {len(wide)} samples")
    labels = pd.concat(frames, ignore_index=True).fillna("")

    for metalog_field, slot in SLOTS.items():
        match = labels[metalog_field].str.extract(GOLD_VALUE)  # the first ENVO / UBERON code of the value
        labels[slot] = (match[0] + "_" + match[1]).fillna("")
        unusable = labels[slot].ne("") & ~labels[slot].isin(valid_terms)
        print(f"{slot}: {labels[slot].ne('').sum()} ENVO/UBERON labels, "
              f"{unusable.sum()} dropped because obsolete or not in the term index")
        labels.loc[unusable, slot] = ""
    # a few accessions appear under two Metalog aliases (e.g. in two studies): keep the first
    return labels[labels["spire_sample_name"] != ""].drop_duplicates("spire_sample_name")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metalog_dir", required=True)
    parser.add_argument("--sample_info", required=True, help="MicrobeAtlas sample.info(.gz)")
    parser.add_argument("--ontology_terms", required=True, help="TSV from 1_build_term_index.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_chars", type=int, default=2000, help="Truncate each sample text")
    args = parser.parse_args()

    terms = pd.read_csv(args.ontology_terms, sep="\t", keep_default_na=False)
    code_to_label = dict(zip(terms["term_id"], terms["label"]))  # obsolete included: it is input text
    valid_terms = set(terms.loc[terms["obsolete"].astype(str) != "True", "term_id"])

    labels = load_metalog_labels(os.path.expanduser(args.metalog_dir), valid_terms)
    by_accession = labels.set_index("spire_sample_name")[["study_code", "domain", *SLOTS.values()]].to_dict("index")

    rows, n_records = [], 0
    for sample_id, lines in iter_sample_info(os.path.expanduser(args.sample_info)):
        n_records += 1
        # the record's own id, then every BioSample / SRA accession written inside it (e.g. sample_biosample=SAMN...)
        accessions = [sample_id] + ACCESSION.findall(" ".join(lines))
        spire = next((acc for acc in accessions if acc in by_accession), None)  # first one Metalog knows
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
