#!/usr/bin/env python3
"""
Turn the MicrobeAtlas cluster dumps into a clean sample_id -> cluster_id join.

The dumps key on <run_id>.<sample_id> and we want the sample. Three things have
to be decided, and they are decided here once so every downstream script agrees:

    cluster -1        HDBSCAN noise. Dropped - it is not a cluster.
    several runs      One sample can be sequenced more than once. Rows that say
                      -1 are ignored rather than treated as a vote, so a sample
                      clustered in any run keeps that cluster.
    runs disagree     ~1% of samples have runs landing in DIFFERENT clusters.
                      Dropped: there is no non-arbitrary way to pick one.

Writes clusters/sample_to_cluster_{fine,coarse}.tsv (sample_id<TAB>cluster_id).

    MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/build_cluster_join.py
"""
import os
from collections import Counter

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
C = f"{ROOT}/clusters"
SOURCES = {"fine": "cclusts_msamp5_UMAP15d_100nbr.tsv",
           "coarse": "cclusts_msamp50_UMAP15d_100nbr.tsv"}


def build(path):
    keep, conflicted, rows, noise = {}, set(), 0, 0
    with open(path) as fh:
        header = next(fh)
        assert header.startswith("SID"), header
        for line in fh:
            sid, _, cid = line.rstrip("\n").partition("\t")
            rows += 1
            if cid == "-1":
                noise += 1
                continue
            s = sid.split(".", 1)[1] if "." in sid else sid
            prev = keep.get(s)
            if prev is None:
                keep[s] = cid
            elif prev != cid:
                conflicted.add(s)
    for s in conflicted:
        del keep[s]
    return keep, rows, noise, len(conflicted)


def main():
    for tag, fname in SOURCES.items():
        keep, rows, noise, conflicts = build(f"{C}/{fname}")
        sizes = Counter(keep.values())
        v = sorted(sizes.values(), reverse=True)
        out = f"{C}/sample_to_cluster_{tag}.tsv"
        with open(out, "w") as fh:
            fh.write("sample_id\tcluster_id\n")
            for s, c in sorted(keep.items()):
                fh.write(f"{s}\t{c}\n")
        print(f"{tag}: {rows:,} rows -> {len(keep):,} samples in {len(sizes):,} clusters")
        print(f"   dropped {noise:,} noise rows, {conflicts:,} samples whose runs disagreed")
        print(f"   cluster sizes: max {v[0]:,}  median {v[len(v) // 2]}  min {v[-1]}  "
              f"| >=40 samples: {sum(1 for x in v if x >= 40):,}")
        print(f"   wrote {out}\n")


if __name__ == "__main__":
    main()
