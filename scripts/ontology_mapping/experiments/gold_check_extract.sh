#!/usr/bin/env bash
# Inputs for gold_check.py: the gold samples' rows of the atlas outputs, and GPT coarse biomes for
# the gold + Metalog-linked samples. ~30 s on the Mac.
#   bash experiments/gold_check_extract.sh
set -e
P=${P:-~/MicrobeAtlasProject}; O=$P/ontology_mapping/experiments/gold_check; L=$P/sidequest/latest
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
zcat "$P/ontology_mapping/atlas_backoff/atlas_predictions.tsv.gz" | pick "$O/gold_ids.txt" > "$O/atlas_backoff_gold.tsv"
zcat "$P/ontology_mapping/rerank_atlas/atlas_final.tsv.gz" | pick "$O/gold_ids.txt" > "$O/atlas_final_gold.tsv"
ids=$(mktemp)
{ cat "$O/gold_ids.txt"; zcat "$P/metalog/clean/training_set.clean.tsv.gz" | cut -f1; } > "$ids"
awk -F'\t' 'NR==FNR{ids[$1]=1;next} ($1 in ids)' "$ids" "$L/GPT_biomes.txt" > "$O/gpt_biomes_gold_and_linked.tsv"
rm -f "$ids"
wc -l "$O"/*.tsv
