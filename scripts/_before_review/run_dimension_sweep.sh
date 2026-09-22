#!/usr/bin/env bash
# Run the full embed -> verify -> evaluate pipeline for a sweep of
# model/embedding_dim configs, in one command.
#
# Configs (edit the CONFIGS array below to change these):
#   text-embedding-3-small @ 256, 1024, 1536
#   text-embedding-3-large @ 1024, 1536, 2048, 3072
#
# All 7 configs embed the SAME --n_per_biome sample subset (same --seed),
# so the comparison at the end is apples-to-apples. embed_subbiomes_keywords.py
# tags its output files by model but NOT by --embedding_dim, so without an
# explicit --run_tag per config, e.g. small@256 and small@1024 would collide
# on the exact same output/state files. This script assigns each config its
# own --run_tag to keep them fully separate.
#
# Usage:
#   ./run_dimension_sweep.sh            # dry-run only: prints token/cost
#                                        # estimates for all 7 configs, calls
#                                        # no API, writes nothing.
#   ./run_dimension_sweep.sh --yes      # actually runs the full sweep:
#                                        # embed (real API calls) -> verify
#                                        # -> evaluate/compare, for all 7
#                                        # configs.
#   ./run_dimension_sweep.sh --yes --skip-verify   # skip the verify step.
#
# Safe to re-run: embed_subbiomes_keywords.py resumes from its own
# per-run-tag "done" files, so re-running after an interruption (or to add
# more configs later) does not re-embed text already embedded.
#
# NOTE: the underlying embed script's run manifest file
# (run_manifest__perbiome<N>_seed<SEED>.json) is keyed by the sample subset,
# not by model/dim, so it gets overwritten by each config's run - this is
# harmless (the subset itself is identical and deterministic across all
# configs), it just means only the *last* config's exact argv ends up
# recorded in that particular file; each config's own H5 filenames already
# encode its model/dim/run_tag.

set -euo pipefail

# ---- fixed pipeline parameters (edit as needed) ----
N_PER_BIOME=2000
SEED=42
INPUT_DIR="$HOME/MicrobeAtlasProject/sidequest/latest"
EMBED_DIR="$INPUT_DIR/embeddings"
GOLD_PKL="$HOME/MicrobeAtlasProject/gold_dict.pkl"
LABEL_FILE="$INPUT_DIR/GPT_biomes.txt"
GOLD_FIELD="biome"   # "subbiome" needs a much bigger sample to have enough texts/class
LEGACY_REFERENCE="$HOME/MicrobeAtlasProject/sidequest/GPT_sub_biomes_embeddings_aligned.h5"
BATCH_SIZE=2048

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- configs to sweep: "model:dim" ----
CONFIGS=(
  "text-embedding-3-small:256"
  "text-embedding-3-small:1024"
  "text-embedding-3-small:1536"
  "text-embedding-3-large:1024"
  "text-embedding-3-large:1536"
  "text-embedding-3-large:2048"
  "text-embedding-3-large:3072"
)

DO_RUN=false
SKIP_VERIFY=false
for arg in "$@"; do
  case "$arg" in
    --yes) DO_RUN=true ;;
    --skip-verify) SKIP_VERIFY=true ;;
    -h|--help)
      echo "Usage: $0 [--yes] [--skip-verify]"
      echo "  (no args)       Dry-run only: print token/cost estimates for all configs, call no API."
      echo "  --yes           Actually run the full sweep (real API calls, real cost)."
      echo "  --skip-verify   Skip the verify_embeddings.py sanity-check step for each config."
      exit 0
      ;;
  esac
done

label_for() {  # "text-embedding-3-small" 256 -> "small_256"
  local model="$1" dim="$2"
  case "$model" in
    text-embedding-3-small) echo "small_${dim}" ;;
    text-embedding-3-large) echo "large_${dim}" ;;
    *) echo "${model}_${dim}" ;;
  esac
}

run_tag_for() {  # unique per (model, dim) so output files never collide
  local model="$1" dim="$2"
  echo "${model}__dim${dim}__perbiome${N_PER_BIOME}_seed${SEED}"
}

echo "=================================================================="
echo " Step 1/3: token/cost estimate for all ${#CONFIGS[@]} configs (dry-run, no API calls)"
echo "=================================================================="
COST_LOG="$(mktemp)"
trap 'rm -f "$COST_LOG"' EXIT

for cfg in "${CONFIGS[@]}"; do
  model="${cfg%%:*}"; dim="${cfg##*:}"
  tag="$(run_tag_for "$model" "$dim")"
  echo
  echo "--- $model @ ${dim} dims (run_tag: $tag) ---"
  out="$(python3 "$SCRIPT_DIR/embed_subbiomes_keywords.py" \
    --n_per_biome "$N_PER_BIOME" --seed "$SEED" --targets sub_biomes keywords \
    --model "$model" --embedding_dim "$dim" --run_tag "$tag" \
    --input_dir "$INPUT_DIR" --batch_size "$BATCH_SIZE" --dry_run)"
  echo "$out"
  echo "$out" | grep -oE '^Estimated cost @ \$[0-9.]+ / 1M tokens: \$[0-9.]+' \
    | grep -oE '\$[0-9.]+$' | tr -d '$' >> "$COST_LOG" || true
done

TOTAL_COST="$(awk '{s+=$1} END{printf "%.4f", s}' "$COST_LOG")"
echo
echo "=================================================================="
echo " Estimated total cost across all ${#CONFIGS[@]} configs: \$${TOTAL_COST}"
echo "=================================================================="

if ! $DO_RUN; then
  echo
  echo "Dry-run only - no API calls were made. Re-run with --yes to actually embed, verify, and evaluate."
  exit 0
fi

echo
echo "=================================================================="
echo " Step 2/3: embedding + verifying each of ${#CONFIGS[@]} configs"
echo "=================================================================="
SUB_BIOME_CONFIG_ARGS=()
KEYWORDS_CONFIG_ARGS=()

for cfg in "${CONFIGS[@]}"; do
  model="${cfg%%:*}"; dim="${cfg##*:}"
  tag="$(run_tag_for "$model" "$dim")"
  label="$(label_for "$model" "$dim")"
  echo
  echo "------------------------------------------------------------"
  echo " $label   ($model, $dim dims, run_tag=$tag)"
  echo "------------------------------------------------------------"

  python3 "$SCRIPT_DIR/embed_subbiomes_keywords.py" \
    --n_per_biome "$N_PER_BIOME" --seed "$SEED" --targets sub_biomes keywords \
    --model "$model" --embedding_dim "$dim" --run_tag "$tag" \
    --input_dir "$INPUT_DIR" --batch_size "$BATCH_SIZE" --yes

  SUB_UNIQUE_H5="$EMBED_DIR/GPT_sub_biomes_unique_embeddings__${tag}.h5"
  SUB_FULL_H5="$EMBED_DIR/GPT_sub_biomes_embeddings__${tag}.h5"
  KW_UNIQUE_H5="$EMBED_DIR/GPT_keywords_unique_embeddings__${tag}.h5"
  KW_FULL_H5="$EMBED_DIR/GPT_keywords_embeddings__${tag}.h5"
  SUBSET_IDS="$EMBED_DIR/subset_ids__perbiome${N_PER_BIOME}_seed${SEED}.txt"

  SUB_BIOME_CONFIG_ARGS+=(--config "${label}=${SUB_UNIQUE_H5}")
  KEYWORDS_CONFIG_ARGS+=(--config "${label}=${KW_UNIQUE_H5}")

  if ! $SKIP_VERIFY; then
    echo "-- verifying sub_biomes ($label) --"
    VERIFY_ARGS=(--full_h5 "$SUB_FULL_H5" --unique_h5 "$SUB_UNIQUE_H5" --subset_ids_file "$SUBSET_IDS")
    # Only text-embedding-3-small @ 1536 matches the legacy reference file's
    # model/dimension, so only cross-check that one config against it.
    if [[ "$model" == "text-embedding-3-small" && "$dim" == "1536" && -f "$LEGACY_REFERENCE" ]]; then
      VERIFY_ARGS+=(--reference_h5 "$LEGACY_REFERENCE")
    fi
    python3 "$SCRIPT_DIR/verify_embeddings.py" "${VERIFY_ARGS[@]}"

    echo "-- verifying keywords ($label) --"
    python3 "$SCRIPT_DIR/verify_embeddings.py" \
      --full_h5 "$KW_FULL_H5" --unique_h5 "$KW_UNIQUE_H5" --subset_ids_file "$SUBSET_IDS"
  fi
done

echo
echo "=================================================================="
echo " Step 3/3: evaluating/comparing all ${#CONFIGS[@]} configs"
echo "=================================================================="
mkdir -p "$EMBED_DIR/eval_sweep_sub_biomes" "$EMBED_DIR/eval_sweep_keywords"

echo "-- sub_biomes --"
python3 "$SCRIPT_DIR/evaluate_embeddings.py" \
  "${SUB_BIOME_CONFIG_ARGS[@]}" \
  --source_text_file "$INPUT_DIR/GPT_sub_biomes.txt" \
  --label_file "$LABEL_FILE" \
  --gold_pkl "$GOLD_PKL" --gold_field "$GOLD_FIELD" \
  --output_dir "$EMBED_DIR/eval_sweep_sub_biomes"

echo
echo "-- keywords --"
python3 "$SCRIPT_DIR/evaluate_embeddings.py" \
  "${KEYWORDS_CONFIG_ARGS[@]}" \
  --source_text_file "$INPUT_DIR/GPT_keywords.txt" --is_keywords \
  --label_file "$LABEL_FILE" \
  --gold_pkl "$GOLD_PKL" --gold_field "$GOLD_FIELD" \
  --output_dir "$EMBED_DIR/eval_sweep_keywords"

echo
echo "=================================================================="
echo " All done."
echo "   $EMBED_DIR/eval_sweep_sub_biomes/evaluation_summary.json"
echo "   $EMBED_DIR/eval_sweep_keywords/evaluation_summary.json"
echo "=================================================================="
