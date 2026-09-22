#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract just the sample.info records whose sample ID is a key in a gold_dict
pickle, so downstream splitting/cleaning only ever touches a small, relevant
subset instead of the full sample.info file.

Adapted from scripts/temp/make_subset_large_file.py's subset_large_file(),
generalized to take an arbitrary gold_dict.pkl instead of Janko's biome TSV,
and made gzip-aware on both input and output.
"""

# run as:
# python ~/github/metadata_mining/scripts/prepare_gold_sample_info_subset.py \
#     --large_file ~/MicrobeAtlasProject/sample.info.gz \
#     --gold_dict ~/MicrobeAtlasProject/gold_dict.pkl \
#     --output_subset ~/MicrobeAtlasProject/sample.info_gold.gz

import argparse
import gzip
import os
import pickle
import time


def open_maybe_gzip(path, mode):
    return gzip.open(path, mode + 't') if path.endswith('.gz') else open(path, mode)


def subset_large_file(input_file_path, output_file_path, valid_samples_set):
    found = set()
    writing_sample = False
    with open_maybe_gzip(input_file_path, 'r') as input_file, \
         open_maybe_gzip(output_file_path, 'w') as output_file:
        for line in input_file:
            if line.startswith('>'):
                sample_name = line[1:].strip()
                writing_sample = sample_name in valid_samples_set
                if writing_sample:
                    found.add(sample_name)

            if writing_sample:
                output_file.write(line)
    return found


def main():
    parser = argparse.ArgumentParser(
        description='Subset a sample.info(.gz) file down to the sample IDs present in a gold_dict.pkl'
    )
    parser.add_argument('--large_file', required=True, help='Path to sample.info or sample.info.gz')
    parser.add_argument('--gold_dict', required=True, help='Path to gold_dict.pkl')
    parser.add_argument('--output_subset', required=True, help='Path to write the subset (.gz optional)')
    args = parser.parse_args()

    large_file = os.path.expanduser(args.large_file)
    gold_dict_path = os.path.expanduser(args.gold_dict)
    output_subset = os.path.expanduser(args.output_subset)

    with open(gold_dict_path, 'rb') as f:
        gold_dict = pickle.load(f)
    valid_samples_set = set(gold_dict.keys())
    print(f"gold_dict has {len(valid_samples_set)} sample IDs")

    start_time = time.time()
    found = subset_large_file(large_file, output_subset, valid_samples_set)
    elapsed = time.time() - start_time

    missing = valid_samples_set - found
    print(f"Found {len(found)}/{len(valid_samples_set)} gold_dict samples in {large_file} ({elapsed:.1f}s)")
    if missing:
        print(f"WARNING: {len(missing)} gold_dict sample IDs were not found, e.g.: {list(missing)[:10]}")
    print(f"Subset written to {output_subset}")


if __name__ == "__main__":
    main()
