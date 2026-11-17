#!/usr/bin/env bash
# Jev follow-up experiments (README 7d, experiments/jev_followups.py). Run from scripts/ontology_mapping:
#   bash experiments/run_jev_followups.sh            # everything: API runs (Jev + gpt-4.1-mini), then the scores
#   bash experiments/run_jev_followups.sh score      # the scores only (no API calls)
#   bash experiments/run_jev_followups.sh gpt_pass2  # GPT second pass over Jev's top 5 (both sets, gpt-4.1-mini and
#                                                    # gpt-5.1, ~$4.5), then the scores
#   bash experiments/run_jev_followups.sh open_similar  # retrieve-then-choose with 5 similar labelled samples of
#                                                    # other studies in the request (both sets, Jev, ~$1.4), then the scores
#   bash experiments/run_jev_followups.sh pass2_open # a second pass over the top 5 of retrieve-then-choose with
#                                                    # similar samples, with / without an example per option (~$0.55)
#   bash experiments/run_jev_followups.sh topdown_open  # top-down over open vocabularies (jev_open_topdown.py): biome
#                                                    # branch / whole ENVO+UBERON / climb-M covers, 3 starts (~$4)
#   bash experiments/run_jev_followups.sh topdown_curated  # top-down from a curated top layer
#                                                    # (experiments/curated_top_layer.tsv), is_a vs is_a + part_of links,
#                                                    # and the whole ontology with is_a + part_of (feature, material; ~$2)
# The inputs in $J/followups/ were built with `jev_followups.py candidates / open` (commands in README 7d).
# Resumable: Jev answers are cached in $J/jev_cache.jsonl and rerank_pilot.py run skips answered samples, so a rerun
# after an interruption only sends what is missing. Jev key: ~/MicrobeAtlasProject/jev_api_key; OpenAI key:
# ~/MicrobeAtlasProject/my_api_key. Estimated cost: Jev ~$3, gpt-4.1-mini ~$2.8.
set -euo pipefail
X=~/MicrobeAtlasProject/ontology_mapping/experiments; R=$X/rerank; J=$X/jev; F=$J/followups; O=$F/results
mkdir -p $O
LM=~/MicrobeAtlasProject/metalog/clean/previous_2026-10-01/biome_label_map.tsv
jev() { python3 experiments/jev_pilot.py run --arm flat --cache $J/jev_cache.jsonl "$@"; }

if [ "${1:-all}" = "gpt_pass2" ]; then
  # 3b. the second pass over Jev's whole-vocabulary top 5 by GPT instead of Jev (same candidate files)
  for SET in gated random; do for M in gpt-4.1-mini gpt-5.1; do
    python3 experiments/rerank_pilot.py run --pilot $F/${SET}_jevtop5.jsonl --model $M --output $O/${M}_${SET}_jevtop5.jsonl
  done; done
fi

if [ "${1:-all}" = "open_similar" ]; then
  # 4c. the open Choice with similar labelled samples (inputs: jev_followups.py open --similar 5, README 7d)
  for SET in gated random; do
    jev --pilot $F/${SET}_open_similar.jsonl --inputs $F/inputs_${SET}_open_similar.json --questions rerank \
      --output $O/${SET}_open_similar.jsonl
  done
fi

if [ "${1:-all}" = "pass2_open" ]; then
  # 4f. second pass over the open Choice's top 5 (inputs: jev_followups.py pass2, README 7d); the similar samples stay in
  # the state; ex = each option with an example sample of another study (none for a term no study used), noex = none
  for SET in gated random; do for V in ex noex; do
    jev --pilot $F/${SET}_open_similar_pass2_$V.jsonl --inputs $F/inputs_${SET}_open_similar.json --questions rerank fits \
      --output $O/${SET}_open_similar_pass2_$V.jsonl
  done; done
fi

if [ "${1:-all}" = "topdown_open" ]; then
  # 5. top-down over open vocabularies (README 7d): build (no API; the training set and term vectors), then the walks
  for SET in gated random; do
    P=$R/pilot_k5anc.jsonl; [ $SET = random ] && P=$J/pilot_all_k5anc.jsonl
    [ -f $F/${SET}_topdown_open.jsonl ] || python3 experiments/jev_open_topdown.py build --pilot $P \
      --inputs $F/inputs_${SET}_open_similar.json --open_pilot $F/${SET}_open_similar.jsonl \
      --open_responses $O/${SET}_open_similar.jsonl --label_map $LM \
      --output $F/${SET}_topdown_open.jsonl --inputs_out $F/inputs_${SET}_topdown_open.json
    for A in "auto root" "auto lca_proto3" "auto lca_open5" "climb300 root" "climb1000 root"; do
      set -- $A
      python3 experiments/jev_open_topdown.py run --pilot $F/${SET}_topdown_open.jsonl \
        --inputs $F/inputs_${SET}_topdown_open.json --tree $1 --start $2 --cache $J/jev_cache.jsonl \
        --output $O/${SET}_td_$1_$2.jsonl
    done
  done
  set -- topdown_open
fi

if [ "${1:-all}" = "topdown_curated" ]; then
  # 5b. curated top layer + ontology trees below its anchors; part_of links from ENVO and UBERON (uberon-basic.obo)
  for SET in gated random; do
    python3 experiments/jev_open_topdown.py add_links --inputs $F/inputs_${SET}_topdown_open.json
    for A in "curated isa" "curated both" "auto both"; do
      set -- $A
      N=$1; [ $2 = both ] && N=${1}_both
      python3 experiments/jev_open_topdown.py run --pilot $F/${SET}_topdown_open.jsonl \
        --inputs $F/inputs_${SET}_topdown_open.json --tree $1 --links $2 --start root --slots feature material \
        --cache $J/jev_cache.jsonl --output $O/${SET}_td_${N}_root.jsonl
    done
  done
  set -- topdown_curated
fi

if [ "${1:-all}" = "all" ]; then
  # 1. Jev on exactly the top-5 candidates GPT-5.1 saw (round 2)
  jev --pilot $R/pilot.jsonl --inputs $J/inputs.json --questions rerank fits --output $O/gated_top5.jsonl
  for SET in gated random; do
    I=$J/inputs.json; [ $SET = random ] && I=$J/inputs_all.json
    # 2. reranking the prototype's candidates + 3 new terms from each source
    for S in jev_vocab prototype retrieval; do
      jev --pilot $F/${SET}_plus_${S}3.jsonl --inputs $I --questions rerank fits --output $O/${SET}_plus_${S}3.jsonl
    done
    # 3. a second pass over Jev's own whole-vocabulary top 3 / 5
    for K in 3 5; do
      jev --pilot $F/${SET}_jevtop${K}.jsonl --inputs $I --questions rerank fits --output $O/${SET}_jevtop${K}.jsonl
    done
    # 4. retrieve, then choose, over the whole ontology
    jev --pilot $F/${SET}_open.jsonl --inputs $F/inputs_${SET}_open.json --questions rerank --output $O/${SET}_open.jsonl
  done
  # 2b. gpt-4.1-mini on the same augmented candidates (gated samples, as in round 3)
  for S in jev_vocab prototype retrieval; do
    python3 experiments/rerank_pilot.py run --pilot $F/gated_plus_${S}3.jsonl --model gpt-4.1-mini --output $O/gpt-4.1-mini_gated_plus_${S}3.jsonl
  done
fi

cmp() { python3 experiments/jev_followups.py compare "$@"; }
echo; echo "######## 1. Jev vs GPT on the same candidates (gated samples)"
cmp --output $O/compare_vs_gpt.json --runs \
  prototype=$R/pilot.jsonl,-,prototype \
  jev_top5=$R/pilot.jsonl,$O/gated_top5.jsonl,rerank \
  gpt-5.1_top5=$R/pilot.jsonl,$R/resp_v2_gpt-5.1.jsonl,gpt \
  gpt-4.1-mini_top5=$R/pilot.jsonl,$R/resp_v2_gpt-4.1-mini.jsonl,gpt \
  jev_top5+anc=$R/pilot_k5anc.jsonl,$J/flat.jsonl,rerank \
  gpt-4.1-mini_top5+anc=$R/pilot_k5anc.jsonl,$R/resp_k5anc_gpt-4.1-mini.jsonl,gpt
echo; echo "######## 1b. The same, with Jev as the reference (paired Jev - GPT differences)"
cmp --output $O/compare_jev_vs_gpt.json --runs \
  jev_top5=$R/pilot.jsonl,$O/gated_top5.jsonl,rerank \
  gpt-5.1_top5=$R/pilot.jsonl,$R/resp_v2_gpt-5.1.jsonl,gpt \
  gpt-4.1-mini_top5=$R/pilot.jsonl,$R/resp_v2_gpt-4.1-mini.jsonl,gpt

echo; echo "######## 2. Reranking the augmented candidates, gated samples (Jev and gpt-4.1-mini)"
cmp --inputs $J/inputs.json --output $O/compare_augmented_gated.json --runs \
  prototype=$R/pilot_k5anc.jsonl,-,prototype \
  jev=$R/pilot_k5anc.jsonl,$J/flat.jsonl,rerank \
  jev+jev3=$F/gated_plus_jev_vocab3.jsonl,$O/gated_plus_jev_vocab3.jsonl,rerank \
  jev+prototype3=$F/gated_plus_prototype3.jsonl,$O/gated_plus_prototype3.jsonl,rerank \
  jev+retrieval3=$F/gated_plus_retrieval3.jsonl,$O/gated_plus_retrieval3.jsonl,rerank \
  gpt=$R/pilot_k5anc.jsonl,$R/resp_k5anc_gpt-4.1-mini.jsonl,gpt \
  gpt+jev3=$F/gated_plus_jev_vocab3.jsonl,$O/gpt-4.1-mini_gated_plus_jev_vocab3.jsonl,gpt \
  gpt+prototype3=$F/gated_plus_prototype3.jsonl,$O/gpt-4.1-mini_gated_plus_prototype3.jsonl,gpt \
  gpt+retrieval3=$F/gated_plus_retrieval3.jsonl,$O/gpt-4.1-mini_gated_plus_retrieval3.jsonl,gpt

echo; echo "######## 2. Reranking the augmented candidates, random samples (overall, and gated at 30 %)"
cmp --inputs $J/inputs_all.json --gate 0.3 --output $O/compare_augmented_random.json --runs \
  prototype=$J/pilot_all_k5anc.jsonl,-,prototype \
  jev=$J/pilot_all_k5anc.jsonl,$J/all/flat.jsonl,rerank \
  jev+jev3=$F/random_plus_jev_vocab3.jsonl,$O/random_plus_jev_vocab3.jsonl,rerank \
  jev+prototype3=$F/random_plus_prototype3.jsonl,$O/random_plus_prototype3.jsonl,rerank \
  jev+retrieval3=$F/random_plus_retrieval3.jsonl,$O/random_plus_retrieval3.jsonl,rerank

for SET in gated random; do
  P=$R/pilot_k5anc.jsonl; FL=$J/flat.jsonl; I=$J/inputs.json
  [ $SET = random ] && { P=$J/pilot_all_k5anc.jsonl; FL=$J/all/flat.jsonl; I=$J/inputs_all.json; }
  echo; echo "######## 3. A second pass over Jev's own top 3 / 5 ($SET samples)"
  cmp --inputs $I --output $O/compare_jevtop_$SET.json --runs \
    jev_vocab_top1=$P,$FL,vocab \
    prototype=$P,-,prototype \
    jevtop3_rerank=$F/${SET}_jevtop3.jsonl,$O/${SET}_jevtop3.jsonl,rerank \
    jevtop5_rerank=$F/${SET}_jevtop5.jsonl,$O/${SET}_jevtop5.jsonl,rerank \
    jevtop5_fits=$F/${SET}_jevtop5.jsonl,$O/${SET}_jevtop5.jsonl,fits \
    $(for M in gpt-4.1-mini gpt-5.1; do [ -f $O/${M}_${SET}_jevtop5.jsonl ] && echo "jevtop5_$M=$F/${SET}_jevtop5.jsonl,$O/${M}_${SET}_jevtop5.jsonl,gpt"; done)
  if [ -f $O/gpt-5.1_${SET}_jevtop5.jsonl ]; then
    echo; echo "######## 3b. The second pass: GPT - Jev, paired ($SET samples)"
    cmp --inputs $I --output $O/compare_jevtop5_jev_vs_gpt_$SET.json --runs \
      jevtop5_jev=$F/${SET}_jevtop5.jsonl,$O/${SET}_jevtop5.jsonl,rerank \
      jevtop5_gpt-4.1-mini=$F/${SET}_jevtop5.jsonl,$O/gpt-4.1-mini_${SET}_jevtop5.jsonl,gpt \
      jevtop5_gpt-5.1=$F/${SET}_jevtop5.jsonl,$O/gpt-5.1_${SET}_jevtop5.jsonl,gpt
  fi
  echo; echo "######## 4. Retrieve, then choose, over the whole ontology ($SET samples)"
  python3 experiments/jev_followups.py score_open --pilot $P --open_pilot $F/${SET}_open.jsonl \
    --inputs_open $F/inputs_${SET}_open.json --responses $O/${SET}_open.jsonl --flat $FL --output $O/score_open_$SET.json \
    $([ -f $O/${SET}_open_similar.jsonl ] && echo "--also with_similar_samples=$F/${SET}_open_similar.jsonl,$O/${SET}_open_similar.jsonl")
done
echo; echo "######## 4b. The open Choice as a fallback for unseen labels, and its most frequent errors"
python3 experiments/jev_analyses.py open_hybrid
echo; echo "######## 4d. Which samples should get the open Choice's pick: coverage flag or the prototype's low confidence"
python3 experiments/jev_analyses.py open_detector
if [ -f $O/gated_open_similar.jsonl ]; then
  echo; echo "######## 4e. The same two analyses for the open Choice with similar samples"
  python3 experiments/jev_analyses.py open_hybrid --open_variant open_similar
  python3 experiments/jev_analyses.py open_detector --open_variant open_similar
  python3 experiments/jev_analyses.py open_examples
fi
if [ -f $O/gated_open_similar_pass2_ex.jsonl ]; then
  for SET in gated random; do
    P=$R/pilot_k5anc.jsonl; FL=$J/flat.jsonl; G=""
    [ $SET = random ] && { P=$J/pilot_all_k5anc.jsonl; FL=$J/all/flat.jsonl; G="--gate 0.3"; }
    echo; echo "######## 4f. Second pass over the top 5 of retrieve-then-choose with similar samples ($SET samples)"
    cmp --inputs $F/inputs_${SET}_open_similar.json $G --output $O/compare_open_pass2_$SET.json --runs \
      open_similar=$F/${SET}_open_similar.jsonl,$O/${SET}_open_similar.jsonl,rerank \
      pass2_examples=$F/${SET}_open_similar_pass2_ex.jsonl,$O/${SET}_open_similar_pass2_ex.jsonl,rerank \
      pass2_examples_fits=$F/${SET}_open_similar_pass2_ex.jsonl,$O/${SET}_open_similar_pass2_ex.jsonl,fits \
      pass2_no_examples=$F/${SET}_open_similar_pass2_noex.jsonl,$O/${SET}_open_similar_pass2_noex.jsonl,rerank \
      prototype=$P,-,prototype \
      jev_rerank=$P,$FL,rerank
    echo; echo "######## 4g. The same, split by labels seen / unseen in other studies ($SET samples)"
    python3 experiments/jev_followups.py score_open --pilot $P --open_pilot $F/${SET}_open_similar.jsonl \
      --inputs_open $F/inputs_${SET}_open_similar.json --responses $O/${SET}_open_similar.jsonl --open_name open_similar \
      --output $O/score_open_pass2_$SET.json --also \
      pass2_examples=$F/${SET}_open_similar_pass2_ex.jsonl,$O/${SET}_open_similar_pass2_ex.jsonl \
      pass2_no_examples=$F/${SET}_open_similar_pass2_noex.jsonl,$O/${SET}_open_similar_pass2_noex.jsonl
  done
  echo; echo "######## 4h. Does the second pass trade new labels for seen ones?"
  python3 experiments/jev_analyses.py open_pass2
fi
if [ -f $O/gated_td_auto_root.jsonl ]; then
  for SET in gated random; do
    P=$R/pilot_k5anc.jsonl; [ $SET = random ] && P=$J/pilot_all_k5anc.jsonl
    echo; echo "######## 5. Top-down over open vocabularies vs retrieve-then-choose ($SET samples; unseen = gold no other study uses)"
    python3 experiments/jev_open_topdown.py score --pilot $P --inputs $F/inputs_${SET}_topdown_open.json \
      --responses $O/${SET}_td_*.jsonl --output $O/score_topdown_open_$SET.json --open \
      open=$F/${SET}_open.jsonl,$O/${SET}_open.jsonl open_similar=$F/${SET}_open_similar.jsonl,$O/${SET}_open_similar.jsonl
  done
fi
