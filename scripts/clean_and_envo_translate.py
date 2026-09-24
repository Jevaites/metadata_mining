#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean the per-sample metadata files and translate ontology codes to labels.

For every dir_*/<sample>.txt under --metadata_dirs, writes <sample>_clean.txt:
  - drops lines whose value is empty / a missing-value word ("NA", "not collected", ...)
  - drops experiment* and run* lines (sequencing details, not sample context)
  - replaces codes such as ENVO:00001998 or UBERON_0001988 by 'label (definition: ...)'

Originally by dgaio (2023); rewritten 2026-09 to fix the matching bugs listed in
docs (value-based missing check, exact code matching, one log file per directory).

python ~/github/metadata_mining/scripts/clean_and_envo_translate.py \
    --path_to_dir ~/MicrobeAtlasProject \
    --ontology_dict ontologies_dict.pkl \
    --metadata_dirs sample_info_split_dirs \
    --max_processes 8
"""

import argparse
import glob
import os
import pickle
import re
import time
from functools import partial
from multiprocessing import Pool

MISSING = {"", "na", "n/a", "nan", "none", "null", "missing", "unknown", "unspecified",
           "not applicable", "not collected", "not provided"}
CODE = re.compile(r"\[?\b(ENVO|UBERON|FOODON|PO)[:_](\d{7,8})\b\]?", re.IGNORECASE)  # optional [ ]


def clean_file(path, ontology_dict, log):
    kept = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key, has_value, value = line.rstrip("\n").partition("=")
            if key.lower().startswith(("experiment", "run")) or (has_value and value.strip().lower() in MISSING):
                log.write(f"Rejected line: {line.rstrip()}\n")
                continue

            def translate(match):
                label = ontology_dict.get(f"{match[1].upper()}_{match[2]}")
                if label is None:
                    return match[0]
                log.write(f"Converting '{match[0]}' in line '{line.strip()}' to '{label}'\n")
                return f"'{label}'"

            kept.append(key + "=" + CODE.sub(translate, value) if has_value else key)
    with open(path.replace(".txt", "_clean.txt"), "w", encoding="utf-8") as out:
        out.write("\n".join(kept) + "\n")


def clean_directory(dir_path, ontology_dict, log_prefix):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"{log_prefix}_log_{os.path.basename(dir_path)}_{stamp}.txt", "a") as log:
        for path in glob.glob(os.path.join(dir_path, "*.txt")):
            if not path.endswith("_clean.txt"):
                log.write(f"Processing file {os.path.basename(path)}...\n")
                clean_file(path, ontology_dict, log)


def main():
    parser = argparse.ArgumentParser(description="Clean metadata and translate ontology codes to labels")
    parser.add_argument("--path_to_dir", default=".", help="Working directory")
    parser.add_argument("--ontology_dict", required=True, help="Pickle {ENVO_00000001: 'label (definition: ...)'}")
    parser.add_argument("--metadata_dirs", required=True, help="Directory containing dir_* sub-directories")
    parser.add_argument("--max_processes", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    root = os.path.expanduser(args.path_to_dir)
    with open(os.path.join(root, args.ontology_dict), "rb") as handle:
        ontology_dict = pickle.load(handle)

    base_dir = os.path.join(root, args.metadata_dirs)
    dirs = sorted(glob.glob(os.path.join(base_dir, "dir_*")))
    worker = partial(clean_directory, ontology_dict=ontology_dict,
                     log_prefix=os.path.join(base_dir, "log_clean_and_envo_translate"))
    with Pool(min(args.max_processes, os.cpu_count())) as pool:
        pool.map(worker, dirs)
    print(f"Cleaned {len(dirs)} directories in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
