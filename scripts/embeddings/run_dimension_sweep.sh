#!/usr/bin/env bash
# Embed -> verify -> compare, for a sweep of model/dimension configs.
#
# Every config embeds the SAME sample subset (same --n_per_biome and --seed),
# so the comparison at the end is apples-to-apples. Output filenames carry
# model + dim + subset, so configs never overwrite each other and a re-run
# resumes instead of re-paying for text already embedded.
#
#   ./run_dimension_sweep.sh          # dry run: token/cost estimate only
#   ./run_dimension_sweep.sh --yes    # real API calls, then verify + compare
#   ./run_dimension_sweep.sh --yes --skip-verify

set -eo pipefail   # not -u: bash 3.2 on macOS errors on empty "${array[@]}"

N_PER_BIOME=2250
SEED=42
INPUT_DIR="$HOME/MicrobeAtlasProject/sidequest/latest"
EMBED_DIR="$INPUT_DIR/embeddings"
GOLD_PKL="$HOME/MicrobeAtlasProject/gold_dict.pkl"
# Same model and dimension as the new small@1536 config, so texts present in
# both should come back near-identical.
LEGACY_REFERENCE="$HOME/MicrobeAtlasProject/sidequest/GPT_sub_biomes_embeddings_aligned.h5"

CONFIGS=(
  "text-embedding-3-small:256"
  "text-embedding-3-small:1024"
  "text-embedding-3-small:1536"
  "text-embedding-3-large:1024"
  "text-embedding-3-large:1536"
  "text-embedding-3-large:2048"
  "text-embedding-3-large:3072"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSET_IDS="$EMBED_DIR/subset_ids__perbiome${N_PER_BIOME}_seed${SEED}.txt"
DO_RUN=false
SKIP_VERIFY=false
for arg in "$@"; do
  case "$arg" in
    --yes) DO_RUN=true ;;
    --skip-verify) SKIP_VERIFY=true ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  esac
done

embed() {  # model dim [extra flags...]
  python3 "$SCRIPT_DIR/embed_subbiomes_keywords.py" \
    --n_per_biome "$N_PER_BIOME" --seed "$SEED" --targets sub_biomes keywords \
    --input_dir "$INPUT_DIR" --model "$1" --embedding_dim "$2" "${@:3}"
}

SUB_ARGS=()
KW_ARGS=()
for cfg in "${CONFIGS[@]}"; do
  model="${cfg%%:*}"; dim="${cfg##*:}"
  tag="${model}__dim${dim}__perbiome${N_PER_BIOME}_seed${SEED}"
  short="${model#text-embedding-3-}_${dim}"
  SUB_ARGS+=(--config "${short}=$EMBED_DIR/GPT_sub_biomes_unique_embeddings__${tag}.h5")
  KW_ARGS+=(--config "${short}=$EMBED_DIR/GPT_keywords_unique_embeddings__${tag}.h5")

  echo; echo "=== $short ==="
  if $DO_RUN; then
    embed "$model" "$dim" --yes
    if ! $SKIP_VERIFY; then
      verify=(--full_h5 "$EMBED_DIR/GPT_sub_biomes_embeddings__${tag}.h5"
              --unique_h5 "$EMBED_DIR/GPT_sub_biomes_unique_embeddings__${tag}.h5"
              --subset_ids_file "$SUBSET_IDS")
      # The legacy file is small@1536; cross-checking any other config is meaningless.
      if [[ "$model" == "text-embedding-3-small" && "$dim" == 1536 && -f "$LEGACY_REFERENCE" ]]; then
        verify+=(--reference_h5 "$LEGACY_REFERENCE")
      fi
      python3 "$SCRIPT_DIR/verify_embeddings.py" "${verify[@]}"
      python3 "$SCRIPT_DIR/verify_embeddings.py" \
        --full_h5 "$EMBED_DIR/GPT_keywords_embeddings__${tag}.h5" \
        --unique_h5 "$EMBED_DIR/GPT_keywords_unique_embeddings__${tag}.h5" \
        --subset_ids_file "$SUBSET_IDS"
    fi
  else
    embed "$model" "$dim" --dry_run
  fi
done

if ! $DO_RUN; then
  echo; echo "Dry run only - no API calls. Re-run with --yes to embed, verify and compare."
  exit 0
fi

echo; echo "=== comparing all ${#CONFIGS[@]} configs ==="
python3 "$SCRIPT_DIR/evaluate_embeddings.py" "${SUB_ARGS[@]}" \
  --source_text_file "$INPUT_DIR/GPT_sub_biomes.txt" \
  --label_file "$INPUT_DIR/GPT_biomes.txt" --gold_pkl "$GOLD_PKL" --gold_field biome \
  --output_dir "$EMBED_DIR/eval_sweep_sub_biomes"

python3 "$SCRIPT_DIR/evaluate_embeddings.py" "${KW_ARGS[@]}" \
  --source_text_file "$INPUT_DIR/GPT_keywords.txt" --is_keywords \
  --label_file "$INPUT_DIR/GPT_biomes.txt" --gold_pkl "$GOLD_PKL" --gold_field biome \
  --output_dir "$EMBED_DIR/eval_sweep_keywords"

echo; echo "Results: $EMBED_DIR/eval_sweep_{sub_biomes,keywords}/"
