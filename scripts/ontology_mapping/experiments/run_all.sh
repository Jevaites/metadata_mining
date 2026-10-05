#!/usr/bin/env bash
# Reproduces every number of trivial-methods-upgrades.md. Run from scripts/ontology_mapping:
#   bash experiments/run_all.sh
# Override the data locations with environment variables (defaults = the pipeline README paths).
# Runtime on a laptop: roughly 1.5-3 h in total (the two nested-CV ladders dominate).
set -euo pipefail
P=${P:-~/MicrobeAtlasProject}
TERMS=${TERMS:-$P/ontology_terms.tsv.gz}
TRAIN=${TRAIN:-$P/metalog/metalog_training_set.tsv.gz}
KW=${KW:-$P/metalog/keywords__large1024.npz}
SB=${SB:-$P/metalog/sub_biomes__large1024.npz}
TV=${TV:-$P/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5}
OUT=${OUT:-$P/ontology_mapping/experiments}
mkdir -p "$OUT"
DATA=(--ontology_terms "$TERMS" --samples "$TRAIN" --keywords "$KW" --sub_biomes "$SB")

# sections 3-4: every 5_evaluate.py method (old and new) on 5 study-to-fold assignments,
# and on coarser labels
for seed in 0 1 2 3 4; do
  [ -f "$OUT/cv_seed$seed/metrics.json" ] || python 5_evaluate.py --ontology_terms "$TERMS" --samples "$TRAIN" \
    --features "$KW" "$SB" --term_vectors "$TV" --fold_seed $seed --output_dir "$OUT/cv_seed$seed"
done
python experiments/coarsen_labels.py "${DATA[@]}" --min_support 100 --output "$OUT/training_set__coarse100.tsv.gz"
[ -f "$OUT/cv_coarse100/metrics.json" ] || python 5_evaluate.py --ontology_terms "$TERMS" --samples "$OUT/training_set__coarse100.tsv.gz" \
  --features "$KW" "$SB" --term_vectors "$TV" --output_dir "$OUT/cv_coarse100"
python experiments/summarise_runs.py --runs "$OUT"/cv_seed{0,1,2,3,4} --coarse "$OUT/cv_coarse100" --output "$OUT/summary.json" \
  | tee "$OUT/summary.txt"

# sections 1-2: nested-CV ladders (fold_seed 0)
python experiments/trivial_knn.py "${DATA[@]}" --term_vectors "$TV" --output "$OUT/trivial_knn.json" | tee "$OUT/trivial_knn.log"
python experiments/trivial_nearest_term.py "${DATA[@]}" --term_vectors "$TV" --output "$OUT/trivial_nearest_term.json" \
  | tee "$OUT/trivial_nearest_term.log"
