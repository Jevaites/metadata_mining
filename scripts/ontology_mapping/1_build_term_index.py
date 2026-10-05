#!/usr/bin/env python3
"""
Step 1: flatten ontology files (ENVO, Uberon, PO, FOODON) into one term table (one row per term).

Obsolete terms are kept and flagged (obsolete=True): they are not valid labels, but
their codes still appear in submitter text and in Metalog, so later steps need them.

Output columns: ontology, term_id, label, synonyms, definition, parents, obsolete
(list columns are joined with "||"; ids use "_" as in ENVO_00001998).

Sources are OBO files, or OWL (RDF/XML) files for ontologies without an OBO release (FOODON),
as paths (.gz allowed) or URLs. Only the ontology's own terms (id prefix) are kept; their
parents can be in another ontology. Synonyms: every synonym scope; parents: named is_a only.

python 1_build_term_index.py \
  --obo ENVO=https://raw.githubusercontent.com/EnvironmentOntology/envo/master/envo.obo \
        UBERON=https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo \
        PO=https://raw.githubusercontent.com/Planteome/plant-ontology/master/po.obo \
        FOODON=https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl \
  --output ~/MicrobeAtlasProject/ontology_terms.tsv.gz

--append keeps the rows of an existing table for the ontologies not given in --obo, so new
ontologies can be added without re-downloading (and changing) the others:
python 1_build_term_index.py --append ~/MicrobeAtlasProject/ontology_terms_envo_uberon.tsv.gz \
  --obo PO=po.obo FOODON=foodon.owl.gz --output ~/MicrobeAtlasProject/ontology_terms.tsv.gz
"""

import argparse
import gzip
import io
import re
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')  # first "..." on a line, allowing \" inside
SCOPE_SUFFIX = re.compile(r" \((exact|narrow|broad|related)(, [^)]*)?\)$")  # PO repeats the scope in the text


def read_bytes(source):
    """The file's bytes, from a local path or a URL (gunzipped when it ends with .gz)."""
    if source.startswith("http"):
        with urllib.request.urlopen(source) as response:
            data = response.read()
    else:
        with open(source, "rb") as handle:
            data = handle.read()
    return gzip.decompress(data) if source.endswith(".gz") else data


def read_obo(source):
    """Return the OBO file as a list of lines."""
    return read_bytes(source).decode("utf-8").splitlines()


RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OBO = "{http://purl.obolibrary.org/obo/}"
OIO = "{http://www.geneontology.org/formats/oboInOwl#}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
SYNONYM_TAGS = {OIO + t for t in ["hasExactSynonym", "hasSynonym", "hasNarrowSynonym", "hasBroadSynonym",
                                  "hasRelatedSynonym"]} | {OBO + "IAO_0000118"}  # IAO_0000118: alternative term


def parse_owl(data, prefix):
    """Minimal OWL (RDF/XML) reader: the owl:Class elements whose IRI is .../obo/PREFIX_...,
    with the same fields as parse_obo (definition = IAO_0000115, obsolete = owl:deprecated)."""
    terms = []
    root = ET.parse(io.BytesIO(data)).getroot()
    for cls in root.findall("{http://www.w3.org/2002/07/owl#}Class"):
        iri = cls.get(RDF + "about", "")
        term_id = iri.rsplit("/", 1)[-1]
        if not term_id.startswith(prefix + "_"):
            continue
        labels = [(e.get(XML_LANG) or "", (e.text or "").strip()) for e in cls.findall(RDFS + "label")]
        label = next((t for lang, t in labels if lang in ("", "en") and t), "")
        if not label:
            continue
        definition = cls.find(OBO + "IAO_0000115")
        terms.append({
            "ontology": prefix, "term_id": term_id, "label": label,
            "synonyms": [(e.text or "").strip() for e in cls if e.tag in SYNONYM_TAGS and (e.text or "").strip()
                         and e.get(XML_LANG, "en") == "en"],
            "definition": (definition.text or "").strip() if definition is not None else "",
            "parents": [e.get(RDF + "resource").rsplit("/", 1)[-1] for e in cls.findall(RDFS + "subClassOf")
                        if e.get(RDF + "resource")],  # named parents; restrictions are skipped
            "obsolete": any((e.text or "").strip() == "true" for e in cls.findall("{http://www.w3.org/2002/07/owl#}deprecated")),
        })
    return terms


def parse_obo(lines, prefix):
    """Minimal OBO reader: keep [Term] stanzas whose id starts with PREFIX:."""
    # synonym types that are translations (PO: 'synonymtypedef: Spanish "Spanish synonym (exact)" EXACT'):
    # their synonyms are skipped, the term texts stay English like ENVO / Uberon
    language_types = {m[1] for m in (re.match(r'synonymtypedef: (\S+) "(\S+) synonym', l) for l in lines) if m and m[1] == m[2]}
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
            scope_and_type = value[match.end():].split() if match else []
            if match and not (len(scope_and_type) > 1 and scope_and_type[1] in language_types):
                term["synonyms"].append(SCOPE_SUFFIX.sub("", match.group(1)))
        elif tag == "is_a":
            term["parents"].append(value.split()[0].replace(":", "_"))
        elif tag == "is_obsolete" and value.strip() == "true":
            term["obsolete"] = True
    return terms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obo", nargs="+", required=True,
                        help="PREFIX=path_or_url of an .obo or .owl file (.gz allowed), e.g. ENVO=envo.obo")
    parser.add_argument("--append", default=None,
                        help="Existing term table: keep its rows for the ontologies not given in --obo")
    parser.add_argument("--output", required=True, help="TSV to write")
    args = parser.parse_args()

    rows = []
    for pair in args.obo:
        prefix, source = pair.split("=", 1)
        owl = re.search(r"\.owl(\.gz)?$", source)
        terms = parse_owl(read_bytes(source), prefix.upper()) if owl else parse_obo(read_obo(source), prefix.upper())
        print(f"{prefix}: {len(terms)} terms ({sum(t['obsolete'] for t in terms)} obsolete) from {source}")
        rows.extend(terms)

    df = pd.DataFrame(rows)
    for column in ["synonyms", "parents"]:
        df[column] = df[column].apply(lambda values: "||".join(dict.fromkeys(values)))  # dedupe, keep order
    if args.append:
        new_ontologies = set(df["ontology"])
        kept = pd.read_csv(args.append, sep="\t", dtype=str, keep_default_na=False)
        kept = kept[~kept["ontology"].isin(new_ontologies)]
        print(f"Kept {len(kept)} terms of {sorted(set(kept['ontology']))} from {args.append}")
        df = pd.concat([kept, df.astype({"obsolete": str})], ignore_index=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(df)} terms to {args.output}")


if __name__ == "__main__":
    main()
