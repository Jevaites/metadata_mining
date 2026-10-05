# Embeddings of the GPT sub-biomes and keywords

Gives every MicrobeAtlas sample a vector for its GPT sub-biome text and one for its GPT keyword text
(text-embedding-3-large, 1024 dimensions). These vectors are the input of the ontology mapping
(`../ontology_mapping`) and of the cluster analyses below.

| script | what it does |
|---|---|
| `embed_subbiomes_keywords.py` | Embeds each *distinct* text once (32k sub-biome, 1.5M keyword texts for 3.4M samples) and writes `GPT_{target}_unique_embeddings__{model}__dim{d}__{subset}.h5` (texts, embeddings), optionally a per-sample file. The `.h5` is its own resume state. `clean_text()` is imported by the ontology-mapping steps so that texts match byte for byte. |
| `verify_embeddings.py` | Sanity checks of one run: shapes, NaN / zero vectors, norms, coverage of the subset, identical vectors for identical texts, agreement with an older file. |
| `evaluate_embeddings.py` | Compares model / dimension configurations on the same texts: same-vs-different-label AUC and 5-NN accuracy against GPT biomes or the gold set. |
| `run_dimension_sweep.sh` | The 7-configuration sweep (small @ 256/1024/1536, large @ 1024–3072): embed → verify → compare. |
| `peek_h5.py` | Prints a few rows of any embeddings `.h5`. |

```bash
python3 scripts/embeddings/embed_subbiomes_keywords.py --n_per_biome 2000 --dry_run            # cost, no API call
python3 scripts/embeddings/embed_subbiomes_keywords.py --full --model text-embedding-3-large --embedding_dim 1024 --yes
```

Choice (sweep, two independent subsets): large beats small on sub-biomes (+0.035 AUC, 5× the noise); nothing
above 1024 dimensions is measurable. Details: project doc `embedding-pipeline-scripts.md`.

## analyses/

One-off analyses of the GPT texts and their embeddings (2026-09-22 → 09-30). Each script's docstring says what
it measures; the results are in the project docs named below. Run from the repository root with
`MAP_ROOT=~/MicrobeAtlasProject python3 scripts/embeddings/analyses/<script>.py`.

| question | scripts | project doc |
|---|---|---|
| Does the way a keyword list is written change its embedding? | `keyword_style_experiment.py` | `keyword-style-experiment.md` |
| New (GPT-5 + 3-large) vs previous (GPT-3.5 + 3-small) embeddings | `compare_to_previous_embeddings.py`, `neighbour_overlap_full.py`, `model_effect_same_text.py`, `summary_figure.py` | `new-vs-dany-embeddings.md` |
| Why some pairs have cosine 1 in one run only; how noisy the vocabularies are | `scatter_arms.py`, `scatter_arms_figure.py`, `collapse_purity.py`, `vocab_noise.py` | `scatter-arms-string-collisions.md` |
| Is the sub-biome vocabulary full of near-duplicates? | `subbiome_redundancy.py` | `subbiome-vocabulary-redundancy.md` |
| Do metadata embeddings agree with the 16S community clusters? | `build_cluster_join.py`, `cluster_agreement.py`, `cluster_agreement_figure.py`, `cluster_purity_ceiling.py`, `cluster_purity_control.py` | `cluster-agreement.md` |
| Which clusters are coherent / adjacent; how deep can text resolve? | `cluster_structure.py`, `cluster_structure_figure.py`, `cluster_adjacency.py`, `cluster_granularity.py`, `cluster_granularity_figure.py` | `cluster-coherence-and-adjacency.md`, `cluster-granularity-depth.md` |
| Is cluster agreement just study identity? | `cluster_study_blocked.py`, `cluster_study_blocked_figure.py` | `study-leakage-in-cluster-evaluation.md` |
| Can study identity be removed from the keywords? | `keyword_identity_strip.py`, `keyword_token_df.py` | `keyword-identity-strip-test.md` |
| New extraction prompts (v2 identity-stripped, v3 with include list) | `extract_keywords_v2.py`, `evaluate_v2_keywords.py`, `v2_study_blocked.py`, `compare_include_list.py`, `prepare_gold_sample_info_subset.py` | `identity-stripped-prompt-v2-results.md`, `include-list-and-model-choice.md` |
| Did the v3 keywords help downstream; does canonicalising them help? | `arms_knn.py`, `arms_mapping_cv.py`, `canonicalise_keywords.py` | `v3-downstream-results-and-canonicalisation-test.md` |

Main lesson for the mapping: almost all of the embedding gain came from the GPT-5 text, not the embedding
model; keyword embeddings fingerprint the study (75 % of a sample's 10 nearest neighbours share its study),
so every evaluation must be grouped or blocked by study.
