#!/usr/bin/env python3
"""
Flatten ENVO / Uberon OBO files into one term table (one row per term).

Output columns: ontology, term_id, label, synonyms, definition, parents, obsolete
(list columns are joined with "||"; ids use "_" as in ENVO_00001998).

python scripts/build_ontology_term_index.py \
  --obo ENVO=https://raw.githubusercontent.com/EnvironmentOntology/envo/master/envo.obo \
        UBERON=https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo \
  --output ~/MicrobeAtlasProject/ontology_terms.tsv.gz
"""

import argparse
import re
import urllib.request

import pandas as pd

QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')  # first "..." on a line, allowing \" inside


def read_obo(source):
    """Return the OBO file as a list of lines, from a local path or a URL."""
    if source.startswith("http"):
        with urllib.request.urlopen(source) as response:
            return response.read().decode("utf-8").splitlines()
    with open(source, encoding="utf-8") as handle:
        return handle.read().splitlines()


def parse_obo(lines, prefix):
    """Minimal OBO reader: keep [Term] stanzas whose id starts with PREFIX:."""
    terms, term = [], None
    for line in lines + ["[End]"]:
        if line.startswith("["):  # a new stanza starts -> close the previous one
            if term and term["term_id"].startswith(prefix + "_") and term["label"]:
                terms.append(term)
            term = {"ontology": prefix, "term_id": "", "label": "", "synonyms": [],
                    "definition": "", "parents": [], "obsolete": False} if line == "[Term]" else None
            continue
        if term is None or ": " not in line:
            continue
        tag, value = line.split(": ", 1)
        if tag == "id":
            term["term_id"] = value.strip().replace(":", "_")
        elif tag == "name":
            term["label"] = value.strip()
        elif tag == "def":
            match = QUOTED.match(value)
            term["definition"] = match.group(1).replace('\\"', '"') if match else ""
        elif tag == "synonym":
            match = QUOTED.match(value)
            if match:
                term["synonyms"].append(match.group(1))
        elif tag == "is_a":
            term["parents"].append(value.split()[0].replace(":", "_"))
        elif tag == "is_obsolete" and value.strip() == "true":
            term["obsolete"] = True
    return terms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obo", nargs="+", required=True, help="PREFIX=path_or_url, e.g. ENVO=envo.obo")
    parser.add_argument("--output", required=True, help="TSV to write")
    args = parser.parse_args()

    rows = []
    for pair in args.obo:
        prefix, source = pair.split("=", 1)
        terms = parse_obo(read_obo(source), prefix.upper())
        print(f"{prefix}: {len(terms)} terms ({sum(t['obsolete'] for t in terms)} obsolete) from {source}")
        rows.extend(terms)

    df = pd.DataFrame(rows)
    for column in ["synonyms", "parents"]:
        df[column] = df[column].apply(lambda values: "||".join(dict.fromkeys(values)))  # dedupe, keep order
    df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(df)} terms to {args.output}")


if __name__ == "__main__":
    main()
