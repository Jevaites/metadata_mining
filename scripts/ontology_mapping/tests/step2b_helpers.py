"""Toy data and input writers for the tests of 2b_clean_metalog.py (imported by its two test modules:
`from step2b_helpers import ...`).

Run from the repository root (or from scripts/ontology_mapping):
    python -m pytest scripts/ontology_mapping/tests/test_2b_clean_metalog_unit.py \
                     scripts/ontology_mapping/tests/test_2b_clean_metalog_integration.py -v

The step scripts start with a digit ('2b_clean_metalog.py'), so they cannot be imported with a plain
`import`; `load_step` imports them by file name.
"""

import importlib
import os
import sys

import pandas as pd

ONTOLOGY_MAPPING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ONTOLOGY_MAPPING)  # common.py and the step scripts


def load_step(name):
    """Import a step script by its file name, e.g. load_step('2b_clean_metalog')."""
    return importlib.import_module(name)


# ----------------------------------------------------------------------------- toy ontology
# A thirteen-term ontology, the minimum needed to see every label status. Real ids, so the raw
# Metalog values below look like the real ones, but the hierarchy is cut down to:
#
#   biome (ENVO_00000428)                 <- root of the 'biome' subtree
#     └── marine biome (ENVO_00000447)
#           └── marine pelagic biome (ENVO_01000023)
#
# Every other term has no parent; ENVO_00009003 is obsolete.
TERM_ROWS = [
    # term_id            label                                parents          obsolete
    ("ENVO_00000428",  "biome",                             "",              False),
    ("ENVO_00000447",  "marine biome",                      "ENVO_00000428", False),
    ("ENVO_01000023",  "marine pelagic biome",              "ENVO_00000447", False),
    ("ENVO_01001002",  "animal-associated environment",     "",              False),
    ("ENVO_2100002",   "intestine environment",             "",              False),
    ("ENVO_00002003",  "fecal material",                    "",              False),
    ("ENVO_00002149",  "sea water",                         "",              False),
    ("ENVO_00001998",  "soil",                              "",              False),
    ("ENVO_00002007",  "sediment",                          "",              False),
    ("ENVO_00009003",  "obsolete human-associated habitat", "",              True),
    ("UBERON_0001988", "feces",                             "",              False),
    ("PO_0025034",     "leaf",                              "",              False),
    ("FOODON_00001053", "fermented dairy food product",     "",              False),
]


def terms_table():
    """The toy ontology in the format of 1_build_term_index.py (all strings, '||'-joined lists)."""
    return pd.DataFrame([{"ontology": t.split("_")[0], "term_id": t, "label": label, "synonyms": "",
                          "definition": "", "parents": parents, "obsolete": str(obsolete)}
                         for t, label, parents, obsolete in TERM_ROWS])


# ----------------------------------------------------------------------------- files
# Tuple order used by the tests. The files are written with the real files' column order (FILE_*_COLUMNS,
# the real core_wide tables have ~40 columns), so the script must select columns by name.
CORE_COLUMNS = ["sample_alias", "study_code", "spire_sample_name", "artificial",
                "environment_biome", "environment_feature", "environment_material"]
LONG_COLUMNS = ["sample_alias", "curation_tier", "metadata_item", "value"]
FILE_CORE_COLUMNS = ["study_code", "sample_alias", "spire_sample_name", "longitude", "environment_material",
                     "environment_biome", "environment_feature", "artificial"]
FILE_LONG_COLUMNS = ["sample_alias", "metadata_item", "curation_tier", "value"]


def write_tsv(p, rows, columns=None):
    """Write rows (dicts, or tuples in `columns` order) as a TSV; '.gz' paths are gzipped."""
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(p, sep="\t", index=False)
    return str(p)


def write_core(p, rows):
    """Write a core_wide table from (alias, study, accession, artificial, biome, feature, material) tuples."""
    return write_tsv(p, pd.DataFrame(rows, columns=CORE_COLUMNS).reindex(columns=FILE_CORE_COLUMNS, fill_value=""))


def write_long(p, rows):
    """Write an all_long table from (alias, curation_tier, metadata_item, value) tuples."""
    return write_tsv(p, pd.DataFrame(list(rows), columns=LONG_COLUMNS)[FILE_LONG_COLUMNS])


def read_tsv(p):
    """Read a TSV as strings, empty cells as "" (as common.read_tsv)."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def write_snapshot(metalog_dir, core, long=(), date="2026-09-20", domains=("human", "animal", "ocean", "environmental")):
    """Write one Metalog snapshot: <domain>_core_wide_<date>.tsv.gz and <domain>_all_long_<date>.tsv.gz.

    core: {domain: [ (alias, study, accession, artificial, biome, feature, material), ... ]}
    long: [ (alias, curation_tier, metadata_item, value), ... ]  (all written to the first domain's file;
          the script concatenates the long files of all domains, so which file holds a row does not matter)
    A domain missing from `core` gets files with a header only."""
    os.makedirs(metalog_dir, exist_ok=True)
    for i, domain in enumerate(domains):
        write_core(os.path.join(metalog_dir, f"{domain}_core_wide_{date}.tsv.gz"), core.get(domain, []))
        write_long(os.path.join(metalog_dir, f"{domain}_all_long_{date}.tsv.gz"), long if i == 0 else [])
    return str(metalog_dir)


def samples_frame(rows, columns=CORE_COLUMNS):
    """A samples frame as load_core returns it (strings, "" for empty), from tuples in `columns` order."""
    return pd.DataFrame(rows, columns=columns).fillna("")
