#!/usr/bin/env bash
# Inputs for gold_check.py: the gold samples' rows of the atlas outputs, and GPT coarse biomes for
# the gold + Metalog-linked samples. ~30 s on the Mac.
#   bash experiments/gold_check_extract.sh
#   ATLAS_SETS=<atlas_predictions.tsv.gz with set answers> bash experiments/gold_check_extract.sh
# gzip -dc, not zcat: macOS's zcat only reads .Z files. pipefail + .tmp files: a failed read stops the script and
# leaves the earlier outputs as they were.
set -eo pipefail
unz() { gzip -dc "$1"; }
P=${P:-~/MicrobeAtlasProject}; O=$P/ontology_mapping/experiments/gold_check; L=$P/sidequest/latest
# set answers (6_predict_atlas.py with a calibration that has "sets"), for gold_check_sets.py; ATLAS_SETS to override
S=${ATLAS_SETS:-$P/ontology_mapping/atlas_sets_g/atlas_predictions.tsv.gz}
if [ -n "$ATLAS_SETS" ] && [ ! -f "$S" ]; then echo "ATLAS_SETS: no such file: $S" >&2; exit 1; fi
mkdir -p "$O"
python3 - "$P/gold_dict.pkl" "$O" <<'EOF'
import pickle, sys
g = pickle.load(open(sys.argv[1], "rb")); o = sys.argv[2]
open(f"{o}/gold_ids.txt", "w").write("\n".join(g) + "\n")
with open(f"{o}/gold_labels.tsv", "w") as f:
    f.write("sample_id\tpmid\tgold_biome\tgold_sub_biome\tcoords\tlocation\n")
    for k, v in g.items():
        f.write(k + "\t" + "\t".join(str(x).replace("\t", " ") for x in v) + "\n")
EOF
pick() { awk -F'\t' 'NR==FNR{ids[$1]=1;next} FNR==1||($1 in ids)' "$1" -; }
extract() { unz "$1" | pick "$O/gold_ids.txt" > "$2.tmp"; mv "$2.tmp" "$2"; }  # the gold rows of a .tsv.gz
extract "$P/ontology_mapping/atlas_backoff/atlas_predictions.tsv.gz" "$O/atlas_backoff_gold.tsv"
extract "$P/ontology_mapping/rerank_atlas/atlas_final.tsv.gz" "$O/atlas_final_gold.tsv"
if [ -f "$S" ]; then extract "$S" "$O/atlas_sets_gold.tsv"; fi
ids=$(mktemp)
{ cat "$O/gold_ids.txt"; unz "$P/metalog/clean/training_set.clean.tsv.gz" | cut -f1; } > "$ids"
awk -F'\t' 'NR==FNR{ids[$1]=1;next} ($1 in ids)' "$ids" "$L/GPT_biomes.txt" > "$O/gpt_biomes_gold_and_linked.tsv.tmp"
mv "$O/gpt_biomes_gold_and_linked.tsv.tmp" "$O/gpt_biomes_gold_and_linked.tsv"
rm -f "$ids"
wc -l "$O"/*.tsv
