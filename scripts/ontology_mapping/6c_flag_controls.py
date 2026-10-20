#!/usr/bin/env python3
"""
Step 6c: flag atlas samples that are technical controls or mock communities, which have no habitat
and should not get an environment label (Metalog drops them from training: 2b_clean_metalog.py,
artificial bucket `control`).

A sample is flagged when its GPT sub-biome or its GPT keywords say so:
  sub-biome        'laboratory control / blank / mock / standard', 'mock community', 'negative control' ...
  strong keyword   a blank or a technical control: extraction / PCR / kit / reagent / buffer / air
                   blank or control, (DNA) blank(s), no-template control (NTC), empty / technical /
                   processing control, or a commercial mock (ZymoBIOMICS, community or whole-cell standard)
  weak keyword     negative / positive control, mock community / sample / DNA: these words also appear
                   in the study context of real samples (a study that sequenced a mock community, an
                   untreated 'negative control' group), so they count only with a lab-like sub-biome
                   ('laboratory ...', 'sterile ...', '... control', '... blank') or a lab keyword
                   (synthetic metagenome, sterile / nuclease-free water, laboratory)
'control' alone is never enough: 'control soil', 'healthy control' or 'control group' are real samples
of a habitat (an untreated plot, a healthy subject); so is a '<step> control group' ('buffer control group':
horses given a buffer). '_' counts as a space ('reagent_control').

Output TSV: sample_id, control (True/False), control_evidence (the sub-biome or keyword that matched),
one row per sample of either GPT file (a sample without keywords is judged on its sub-biome, and the
other way round); gzipped when the output name ends in .gz.
Checked against Metalog's own control flags on the linked samples: experiments/README.md §12.

python 6c_flag_controls.py \
  --keywords_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_keywords.txt \
  --sub_biomes_texts ~/MicrobeAtlasProject/sidequest/latest/GPT_sub_biomes.txt \
  --output ~/MicrobeAtlasProject/ontology_mapping/atlas_controls.tsv.gz
"""

import argparse
import re

import pandas as pd

from common import path

SUB_BIOME = re.compile(r"^(laboratory|synthetic|sterile)[ -](control|blank|mock|standard)\b|\bmock[ -]?(community|dna)\b"
                       r"|^mock$|\bnegative[ -]control\b|\b(blank|kit|reagent|extraction)[ -](control|blank)\b", re.I)
STRONG = re.compile(
    r"^((dna[ -])?extraction|pcr|kit|reagent|buffer|air|library|sequencing|isolation)[ -](negative[ -]controls?|negatives?|blanks?|controls?)\b(?![ -]groups?\b)"
    r"|^negative[ -]extraction[ -]controls?\b|^(dna[ -])?blanks?([ -](sample|control|well|swab))?$|^ntc$|^no[ -]template[ -]controls?$"
    r"|^(empty|technical|processing)[ -]controls?$|^zymobiomics\b|^(microbial[ -])?community[ -]standards?$"
    r"|^whole[ -]cell[ -]standards?$|^microbial[ -]dna[ -]standard$|^control[ -]blank$", re.I)
WEAK = re.compile(r"^(negative|positive)[ -]controls?$|^mock[ -](community|communities|sample|dna)$", re.I)
LAB_SUB_BIOME = re.compile(r"^(laboratory|sterile|synthetic)\b|\b(blank|mock)\b|\bcontrol$", re.I)
LAB_KEYWORD = re.compile(r"^(synthetic metagenome|sterile water( blank)?|nuclease[ -]free water|pcr[ -]grade.*water|laboratory|lab)$", re.I)


def evidence(sub_biome, keywords):
    """'' when the sample looks like a real habitat sample, else what makes it a control or mock (the text as
    GPT wrote it). '_' counts as a space: GPT sometimes writes snake_case keywords ('reagent_control')."""
    sb = sub_biome.replace("_", " ")
    if SUB_BIOME.search(sb):
        return f"sub-biome: {sub_biome}"
    kws = [(k, k.replace("_", " ")) for k in keywords]  # (as written, as matched)
    strong = next((k for k, text in kws if STRONG.search(text)), "")
    if strong:
        return f"keyword: {strong}"
    weak = next((k for k, text in kws if WEAK.match(text)), "")
    if weak and (LAB_SUB_BIOME.search(sb) or any(LAB_KEYWORD.match(text) for _, text in kws)):
        return f"keyword + lab context: {weak}"
    return ""


def split_keywords(text):
    """GPT keyword set '{a, b , c}' -> ['a', 'b', 'c'] (stripped; '' for an empty set or a trailing comma)."""
    return [k.strip() for k in text.strip("{} ").split(",")]


def read_texts(p):
    """{sample_id: text} of a GPT output file ('id<TAB>text')."""
    out = {}
    with open(path(p), errors="replace") as handle:
        for line in handle:
            sid, _, text = line.rstrip("\n").partition("\t")
            out[sid] = text
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keywords_texts", required=True)
    ap.add_argument("--sub_biomes_texts", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    keywords = read_texts(args.keywords_texts)
    sub = read_texts(args.sub_biomes_texts)
    rows = []
    for sid in list(keywords) + [s for s in sub if s not in keywords]:  # every sample of either file, once
        found = evidence(sub.get(sid, "").strip(), split_keywords(keywords.get(sid, "")))
        rows.append((sid, bool(found), found))
    out = pd.DataFrame(rows, columns=["sample_id", "control", "control_evidence"])
    out.to_csv(path(args.output), sep="\t", index=False)  # compression from the name: .gz -> gzip
    hits = out.loc[out["control"], "control_evidence"].str.split(":").str[0].value_counts().to_dict()
    print(f"Wrote {path(args.output)}: {len(out)} samples, {out['control'].sum()} controls / mocks ({hits})")


if __name__ == "__main__":
    main()
