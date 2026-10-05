#!/usr/bin/env python3
"""
Group Metalog study codes that belong to the same sequencing project, for project-level CV folds.

For every training-set sample, read its MicrobeAtlas record (sample.info.gz) and collect the
project accessions it carries: the SRA/ENA/DDBJ study (`study=SRP/ERP/DRP…`) and any BioProject
(`PRJNA/PRJEB/PRJDB…`). Two study codes are merged when any of their samples share an accession
(connected components over study_code <-> accession), so e.g. Stewart_2018_cow_rumen and
Stewart_2019_cow_rumen end up in one group. Accessions miss projects that re-sequence the same
physical samples under separate accessions: TARA_Oceans_prokaryote and TARA_Oceans_protists share
63 of 64 stations. `--merge` adds such groups by pattern (default: all TARA_* codes).

Output TSV: sample_id, study_code, project_group (the alphabetically first study code of the
group). Pass it to `5_evaluate.py --fold_groups` so whole projects stay on one side of a fold;
the per-study cap is unchanged (still by study_code), so the evaluated samples are identical.

python experiments/project_groups.py \
  --sample_info ~/MicrobeAtlasProject/sample.info.gz \
  --training_set ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --output ~/MicrobeAtlasProject/metalog/clean/project_groups.tsv
"""

import argparse
import fnmatch
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import iter_sample_info, path, read_tsv  # noqa: E402

STUDY = re.compile(r"^study=([SED]RP\d+)")
BIOPROJECT = re.compile(r"\b(PRJ(?:NA|EB|DB|DA|EA)\d+)\b")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample_info", required=True)
    ap.add_argument("--training_set", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--merge", nargs="*", default=["TARA_*"],
                    help="Shell patterns: all study codes matching one pattern form one group")
    args = ap.parse_args()

    train = read_tsv(args.training_set)
    study_of = dict(zip(train["sample_id"], train["study_code"]))
    accessions = defaultdict(set)  # study_code -> project accessions
    for sid, lines in iter_sample_info(path(args.sample_info)):
        if sid not in study_of:
            continue
        acc = {m[1] for m in (STUDY.match(x) for x in lines) if m} | set(BIOPROJECT.findall("\n".join(lines)))
        accessions[study_of[sid]] |= acc

    # union-find over study codes sharing an accession
    parent = {s: s for s in set(study_of.values())}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    owner = {}
    for study, accs in accessions.items():
        for a in accs:
            if a in owner:
                ra, rb = find(owner[a]), find(study)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                owner[a] = study
    for pattern in args.merge:
        codes = sorted(s for s in parent if fnmatch.fnmatch(s, pattern))
        for s in codes[1:]:
            ra, rb = find(codes[0]), find(s)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for s in parent:
        groups[find(s)].append(s)
    merged = {g: sorted(v) for g, v in groups.items() if len(v) > 1}

    with open(path(args.output), "w") as out:
        out.write("sample_id\tstudy_code\tproject_group\n")
        for sid, study in study_of.items():
            out.write(f"{sid}\t{study}\t{find(study)}\n")
    print(f"{len(parent)} study codes -> {len(groups)} project groups; {len(merged)} groups merge several codes:")
    for g, v in sorted(merged.items(), key=lambda kv: -len(kv[1])):
        n = sum(1 for s in study_of.values() if s in v)
        print(f"  {n:6d} samples  {' + '.join(v)}")


if __name__ == "__main__":
    main()
