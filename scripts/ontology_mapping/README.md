# Ontology mapping prototype (MicrobeAtlas → ENVO / Uberon)

Gives every MicrobeAtlas sample one ontology term for each of the three Metalog slots, plus a
confidence score. Terms are ENVO / Uberon, and PO (plant anatomy) / FOODON (food) where Metalog
uses them (leaf, fermented food products, milk):

| slot | Metalog field | example |
|---|---|---|
| `biome` | `environment_biome` | `ENVO_00000446` terrestrial biome |
| `feature` | `environment_feature` | `ENVO_2100002` intestine environment |
| `material` | `environment_material` | `ENVO_00002003` fecal material |

**The approach is supervised label transfer.** About 58k MicrobeAtlas samples are also in
Metalog, which curated their slots by hand. Each of these samples gets a vector: by default,
the embedding of the keywords and sub-biome that GPT extracted from its metadata. A linear
classifier learns to map that vector to Metalog's labels. It is then applied to all 3.4M
atlas samples. Zero-shot retrieval (the ontology term closest to the sample) is also
evaluated, and used inside the `hybrid` method.

## Pipeline

```
ENVO/Uberon/PO/FOODON ─1─> ontology_terms.tsv.gz ─────────────┬──4──> term embeddings (.h5)
                                                              │                │
Metalog *_all_long_*.tsv.gz ─┐                                │                │
MicrobeAtlas sample.info.gz ─┴──2──> metalog_training_set.tsv.gz              │
                                          │  (sample_id, study, domain,        │
                                          │   3 labels, cleaned text)          │
GPT_{keywords,sub_biomes}.txt ─┐          │                                    │
unique GPT embeddings (.h5) ───┴──3──> keywords.npz, sub_biomes.npz            │
                                          │                                    │
                                          ├──5──> metrics.json, predictions, ◄─┘   (evaluation)
                                          │       calibration.json
                                          ├──6──> atlas_predictions.tsv.gz         (all 3.4M samples:
                                          │                                         term, probability,
                                          │                                         back-off term, candidates)
                                          └──7──> atlas_final.tsv.gz               (optional: LLM reranking
                                                                                    of the least-confident)
```

| step | script | what it does | time |
|---|---|---|---|
| 1 | `1_build_term_index.py` | Parses the ontology files (OBO; OWL for FOODON) into one row per term: label, synonyms, definition, `is_a` parents, obsolete flag. `--append` adds ontologies to an existing table. | < 1 min |
| 2 | `2_build_training_set.py` | Links MicrobeAtlas records to Metalog samples by accession, and writes the labels and a cleaned text for each linked sample. | ~5 min |
| 2b | `2b_clean_metalog.py` | Flags every Metalog sample (controls, perturbed and ancient samples, duplicate accessions, unusable labels) and writes a clean and a gold version of the training set. Nothing is deleted. | < 1 min |
| 3 | `3_extract_sample_embeddings.py` | Looks up the GPT keyword or sub-biome embedding of each labelled sample (run once per kind). | < 1 min |
| 4 | `4_embed_terms.py` | Embeds every term text (`label; synonyms`) with the model used for the GPT texts; resumable (only new texts). ~49k texts with FOODON (783k tokens): about $0.10. | ~12 min (new texts only) |
| 5 | `5_evaluate.py` | Runs cross-validation grouped by study for every method on the same features. Writes metrics and per-sample predictions. | 5–15 min |
| 6 | `6_predict_atlas.py` | Trains `linear` (or `--method prototype / knn_study`) on the evaluated samples and labels every atlas sample, streaming the 6.4 GB keyword `.h5`. With `--calibration` (from step 5): calibrated probabilities, the back-off term for a target accuracy, and the reranker's candidates. Resumable. | ~2 min (knn_study: longer; back-off: a few min more) |
| 6b | `6b_coverage.py` | Flags atlas samples outside what Metalog covers: similarity to the nearest training sample in the model's feature space, against a threshold calibrated on held-out Metalog projects. Method-independent. Resumable. | ~25 min |
| 6c | `6c_flag_controls.py` | Flags atlas samples that are technical controls (blanks, negative controls) or mock communities, from their GPT sub-biome and keywords: they have no habitat and should not get a label. | < 1 min |
| 7 | `7_rerank_atlas.py` | Optional. LLM reranking of the least-confident predictions, fused with the base model and backed off again: `fit` on the Metalog pilot, `build` the requests, `submit` / `collect` through the Batch API, `apply`. | API: ≤ 24 h per batch |
| – | `analyses.py` | Optional. Computes the extra analyses behind the findings report: bootstrap CIs, hierarchy of errors, label ceiling, coarser labels. | ~3 min |
| – | `common.py` | Shared helpers: term loading, sample selection, study folds. | |
| – | `hierarchy.py` | Shared calibration and hierarchical back-off (steps 5–7). | |

Steps 3, 4 and 6 import `iter_samples`, `embed_unique` and related helpers from
`../embed_subbiomes_keywords.py`. Those helpers produced the GPT embeddings, so the texts
match byte for byte.

## Run it

```bash
cd ~/github/metadata_mining/scripts/ontology_mapping
P=~/MicrobeAtlasProject; L=$P/sidequest/latest; E=$L/embeddings
TERMS=$P/ontology_terms.tsv.gz; TRAIN=$P/metalog/metalog_training_set.tsv.gz
TV=$P/ontology_mapping/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5

python 1_build_term_index.py --output $TERMS \
  --obo ENVO=https://raw.githubusercontent.com/EnvironmentOntology/envo/master/envo.obo \
        UBERON=https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo \
        PO=https://raw.githubusercontent.com/Planteome/plant-ontology/master/po.obo \
        FOODON=https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl
# the current table = the 2026-09-23 ENVO/Uberon table + PO / FOODON (2026-10-05 copies in $P/ontologies):
#   python 1_build_term_index.py --append $P/ontology_terms_envo_uberon.tsv.gz \
#     --obo PO=$P/ontologies/po.obo FOODON=$P/ontologies/foodon.owl.gz --output $TERMS
python 2_build_training_set.py --metalog_dir $P/metalog --sample_info $P/sample.info.gz \
  --ontology_terms $TERMS --output $TRAIN
python 2b_clean_metalog.py --metalog_dir $P/metalog --ontology_terms $TERMS --training_set $TRAIN \
  --output_dir $P/metalog/clean        # then review clean/audit_review.tsv and re-run with --overrides
for kind in keywords sub_biomes; do
  python 3_extract_sample_embeddings.py --kind $kind --texts $L/GPT_$kind.txt --sample_ids $TRAIN \
    --unique_h5 $E/GPT_${kind}_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
    --output $P/metalog/${kind}__large1024.npz
done
python 4_embed_terms.py --ontology_terms $TERMS          # needs ~/MicrobeAtlasProject/my_api_key_embeddings

# evaluation: main configuration (keywords + sub-biomes), then e.g. a TF-IDF baseline on the same samples
KW=$P/metalog/keywords__large1024.npz; SB=$P/metalog/sub_biomes__large1024.npz
python 5_evaluate.py --ontology_terms $TERMS --samples $TRAIN --features $KW $SB --term_vectors $TV \
  --output_dir $P/ontology_mapping/cv_kw_sb
python 5_evaluate.py --ontology_terms $TERMS --samples $TRAIN --features tfidf --only_samples_in $KW $SB \
  --output_dir $P/ontology_mapping/cv_tfidf

python 6_predict_atlas.py --ontology_terms $TERMS --samples $TRAIN --train_vectors $KW $SB \
  --keywords_texts $L/GPT_keywords.txt --sub_biomes_texts $L/GPT_sub_biomes.txt \
  --keywords_h5 $E/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_h5 $E/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --output_dir $P/ontology_mapping/atlas_kw_sb
# same with another model (use a separate --output_dir per method):
#   ... --method prototype --term_vectors $TV --output_dir $P/ontology_mapping/atlas_kw_sb_prototype
#   ... --method knn_study --output_dir $P/ontology_mapping/atlas_kw_sb_knn_study

# experiments behind knn_study / prototype (≈ 1.5-3 h), and the atlas consistency check
bash experiments/run_all.sh
python experiments/verify_atlas_methods.py
```

**Recommended configuration (2026-10-01): cleaned labels, prototype, calibrated back-off, optional
LLM reranking.** See *Calibration and back-off* and *Step 7* below for what each option does.

```bash
# labels: the biome label map (best-effort draft, still to be reviewed) + the cleaning flags
python 2b_clean_metalog.py --metalog_dir $P/metalog --ontology_terms $TERMS --training_set $TRAIN \
  --label_map $P/metalog/clean/biome_label_map.tsv --output_dir $P/metalog/clean
CLEAN=$P/metalog/clean/training_set.clean.tsv.gz; O=$P/ontology_mapping
python 5_evaluate.py --ontology_terms $TERMS --samples $CLEAN --features $KW $SB --term_vectors $TV \
  --output_dir $O/cv_backoff                                  # writes calibration.json
for s in 1 2; do                                              # more fold seeds for a pooled calibration
  python 5_evaluate.py --ontology_terms $TERMS --samples $CLEAN --features $KW $SB --term_vectors $TV \
    --fold_seed $s --closed_only --output_dir $O/cv_backoff_s$s
done
python 6_predict_atlas.py --ontology_terms $TERMS --samples $CLEAN --train_vectors $KW $SB \
  --keywords_texts $L/GPT_keywords.txt --sub_biomes_texts $L/GPT_sub_biomes.txt \
  --keywords_h5 $E/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_h5 $E/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --method prototype --term_vectors $TV \
  --calibration $O/cv_backoff/calibration.json $O/cv_backoff_s1/calibration.json $O/cv_backoff_s2/calibration.json \
  --target_accuracy 0.9 --accuracy strict --output_dir $O/atlas_backoff   # 4 GB RAM: add --chunk_rows 2000
# coverage flag (one run serves every atlas method; merge on sample_id), then its validation
python 6b_coverage.py --samples $CLEAN --train_vectors $KW $SB --fold_groups $P/metalog/clean/project_groups.tsv \
  --index $O/atlas_backoff/index.npz \
  --keywords_h5 $E/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_h5 $E/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --output_dir $O/atlas_coverage                              # rerun until atlas_coverage.tsv.gz is written
python experiments/coverage_check.py --coverage_dir $O/atlas_coverage --gold_dir $O/experiments/gold_check \
  --training_set $CLEAN --ontology_terms $TERMS --cv_predictions $O/cv_backoff/predictions.tsv.gz
# controls and mock communities (no habitat: leave them unlabelled)
python 6c_flag_controls.py --keywords_texts $L/GPT_keywords.txt --sub_biomes_texts $L/GPT_sub_biomes.txt \
  --output $O/atlas_controls.tsv.gz
# optional LLM reranking (needs the Metalog pilot: experiments/rerank_pilot.py, README 7c)
X=$O/experiments/rerank; R=$O/rerank_atlas
python 7_rerank_atlas.py fit --pilot $X/pilot_k5anc.jsonl --responses $X/resp_k5anc_gpt-4.1-mini.jsonl \
  --calibration $O/cv_backoff/calibration.json --samples $CLEAN --target_accuracy 0.9 --output $R/rerank_params.json
python 7_rerank_atlas.py build --params $R/rerank_params.json --atlas_dir $O/atlas_backoff \
  --sample_info $P/sample.info.gz --samples $CLEAN --train_vectors $KW $SB --work_dir $R \
  --keywords_h5 $E/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5 \
  --sub_biomes_h5 $E/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5
python 7_rerank_atlas.py submit --work_dir $R --limit 2000   # trial; then without --limit
python 7_rerank_atlas.py collect --work_dir $R               # rerun until collected; submit again for the rest
python 7_rerank_atlas.py apply --params $R/rerank_params.json --atlas_dir $O/atlas_backoff --work_dir $R
python experiments/verify_atlas_methods.py --methods linear prototype --calibration $O/cv_backoff/calibration.json
```

## How it works

### Step 1: terms

`parse_obo` is a minimal reader for OBO files. It keeps only `[Term]` stanzas whose ID has
the right prefix, and reads `id`, `name`, `def`, `synonym` (all scopes), `is_a` and
`is_obsolete`. IDs are written `ENVO_00001998`. List columns are joined with `||`.

- **FOODON** has no OBO release: `parse_owl` reads its OWL (RDF/XML) file the same way: `rdfs:label`
  (English), every synonym property, `IAO_0000115` (definition), named `rdfs:subClassOf` parents
  (restrictions are skipped) and `owl:deprecated`. Only `FOODON_` classes are kept (not the
  NCBITaxon or other imports); their parents can be in other ontologies.
- **PO** synonyms that are translations (Spanish, Japanese, German synonym types) are skipped, and
  the scope PO repeats inside the text ("radix (exact)") is removed, so term texts stay English.
- **`--append`** keeps the rows of an existing table for the ontologies not rebuilt. The current
  table is the 2026-09-23 ENVO / Uberon table plus PO (1,793 terms) and FOODON (29,255), so the
  ENVO / Uberon rows, and every result that depends only on them, are unchanged.
Obsolete terms are kept but flagged. That lets step 2 report obsolete gold labels, and still
translate obsolete codes that appear in submitter text.

**Term text** (`common.term_text`) is `label; synonym; synonym`. It is used for term
embeddings and for TF-IDF retrieval. Definitions are left out on purpose: they bring in
generic words, such as country names, that match noise in the metadata.

### Step 2: labelled samples

- **Labels.** Each Metalog long table is pivoted to one row per sample. The first
  `[ENVO:…]` or `[UBERON:…]` code of each slot becomes the label (these step-2 labels become the
  `*_raw` columns of the clean training set; step 2b also keeps PO and FOODON labels). A code that is obsolete
  or not in the term index is blanked, and the number blanked is printed.
  - This blanks the biome of **every human sample**: Metalog uses the obsolete
    `ENVO:00009003`, which has no replacement.
  - The `human` domain therefore has feature and material labels only.
- **Linking.** A MicrobeAtlas record is linked when its own ID, or any BioSample/SRA
  accession it contains, equals Metalog's `spire_sample_name`. Result: 58,365 records linked
  to 57,874 Metalog samples in 552 studies.
- **Text** (`record_to_text`). The text is used only by the `tfidf` features.
  - The `sample_` and `study_` prefixes are removed from the keys.
  - Identifier, date and coordinate keys are dropped (`DROP_KEYS`). They carry no meaning
    about the environment, and they identify the study.
  - Values that mean "missing" are dropped.
  - `ENVO:…` codes are replaced by their label.
  - The result is written as `key: value; …` and truncated to 2,000 characters.

### Step 2b: cleaning flags

Flag, don't delete. `metalog_flags.tsv.gz` has one row per Metalog sample (all ~159k, not only
the linked ones) with the raw values and these flags:

- **`artificial_bucket`**, from Metalog's `artificial` field:
  - `control`: negative control, mock, spike-in, marked as contaminated. These are hard drops:
    the label is not a habitat, and 247 of them carry the study's default habitat label.
  - `perturbed`: cultivation, (virome) enrichment, sorted cells, mesocosm. The label is the
    source environment, so they stay in training, but they are excluded from gold evaluation.
  - `degraded`: paleosample, post-mortem, museum specimen. Same treatment as `perturbed`.
- **Audit.** Metalog's flag misses some samples (e.g. `Gaffney_2019_marine_Doggerland`
  negative controls labelled marine sediment). Regexes run over the alias and a short list of
  sample-level free-text keys (`AUDIT_KEYS`; study abstracts and questionnaires are excluded).
  Hits on unflagged samples go to `audit_review.tsv`, one row per study × category. They count
  as `audit_unreviewed` (kept for training, out of gold) until someone fills `decision`
  (`control` / `perturbed` / `degraded` / `ok`) and the script is re-run with
  `--overrides audit_review.tsv`. Decided rows are carried into the rewritten file, so re-running the same
  command is idempotent, and a review file with decisions is backed up to
  `audit_review.previous.tsv` before it is overwritten.
- **`drop_reason`**: `no_accession`, `conflicting_duplicate` (same accession under two
  aliases with different labels, e.g. `Pascelli_2020_sponge_virus`), `duplicate_alias`,
  `artificial_control`.
- **Labels.** `<slot>_status` is `ok`, `obsolete` (the human biome `ENVO:00009003`),
  `other_ontology` (CL: 34 materials), `not_in_index`, `no_code`, `control_value` or `empty`.
  ENVO, Uberon, PO and FOODON ids are usable (`KEEP_ONTOLOGIES`; PO / FOODON since 2026-10-05:
  leaf, bark, fermented and dairy food products, milk, lettuce: 465 feature and 128 material labels
  of linked samples). `<slot>_clean` is the usable id. `--label_map` (TSV: slot, from_id, to_id,
  reason) remaps ids, including obsolete and other-ontology ones; raw ids stay in `<slot>_raw`.
  Besides the biome rows, `biome_label_map.tsv` has two material rules: *rhizosphere* (an ENVO
  ecosystem) and *rhizoplane* (a surface layer) → *soil*, the material the other half of the
  rhizosphere studies use (130 linked samples).
  `biome_in_biome_subtree` says whether the biome is under ENVO *biome*.

With `--training_set`, it also writes:

- `training_set.clean.tsv.gz`: step-2 rows without hard drops, with cleaned labels in
  `biome/feature/material`. A drop-in replacement for `$TRAIN`.
- `training_set.gold.tsv.gz`: rows with `in_gold_eval` (bucket `none`, no unreviewed audit
  hit), first copy of each text per study. Use it for evaluation numbers.

`summary.md` has the counts.

### Step 3: sample vectors

Maps each labelled sample to its GPT text, using the same cleaning as when the texts were
embedded, and reads that text's row from the *unique* embedding file. Many samples share a
text, so the `.npz` stores each distinct vector once: `vectors[index[i]]` is the vector of
`sample_ids[i]`.

### Step 5: evaluation

**Samples** (`common.select_samples`):

1. Keep samples with at least one label.
2. Keep samples that have a vector in every `.npz` block.
3. Keep at most 50 samples per study: those with the smallest seeded hash of the sample id
   (`common.stable_key`). Without this cap, one cohort (for example 1,679 infant-gut samples)
   would dominate the scores. A sample's place does not depend on the other rows, so adding labels
   keeps every previously selected sample (before 2026-10-05: a seeded shuffle of the whole table,
   which redrew every study when rows were added).

The result is 17,720 samples from 539 studies (training_set.clean, 2026-10-05).

**Folds** (`common.study_folds`): 5 folds, and all samples of a study go to the same fold,
because samples of one study share most of their text. A study's fold is its seeded hash mod 5, so
it does not move when other studies change, and the folds are identical on every machine. Folds are
not size-balanced (±20 %). `--fold_seed` picks a different split. The split alone moves the biome
back-off coverage by up to 6 points (0.73–0.79 over 3 seeds), so the atlas calibration pools
several seeds (see *Calibration and back-off*). Before 2026-10-05 studies were balanced greedily as
in `GroupKFold`.
`--fold_groups project_groups.tsv` (from `experiments/project_groups.py`) groups by *project*
instead: study codes that share a sequencing project (TARA ×4, Stewart 2018/2019, Alneberg
2018/2020) stay on one side. The evaluated samples are unchanged. With it, biome top-1 of the
supervised methods is about 1–1.5 points lower, because those projects were leaking across folds
(experiments README, section 9).

**Features.** One or more blocks. Each block is L2-normalised and scaled by
1/√(number of blocks), then the blocks are concatenated. So the dot product of two samples
is the mean of their per-block cosines.

| block | sample side | term side (for retrieval) |
|---|---|---|
| `tfidf` | TF-IDF (word 1–2-grams) of `text`, fitted on the training fold and the term texts | TF-IDF of the term text |
| `x.npz` | precomputed vectors (step 3) | `--term_vectors` (step 4) |
| a model name | OpenAI-compatible embeddings of `text` (`--api_key_path`, `--base_url`, `--dimensions`); every distinct text is cached | same model on the term text |

**Methods.** Each method returns a top-5 list and a confidence per sample, per slot. Only the
training fold is used.

| method | idea | confidence |
|---|---|---|
| `majority` | most frequent training label | none |
| `knn` | vote of the k = 25 most similar training samples, weighted by similarity | winner's vote share |
| `knn_study` | k-NN where each training *study* has one vote: of the k = 50 most similar training samples (`--knn_study_k`), each votes 1 / (number of them from its study) | winner's vote share |
| `linear` | `RidgeClassifier` on dense features (`LinearSVC` on sparse TF-IDF) | margin between the best and second-best score |
| `retrieval` / `retrieval_open` | nearest term vector, among the slot's training labels / among all 18.8k terms | best cosine |
| `hybrid` / `hybrid_open` | linear score + 2 × cosine to the term. With `_open`, terms never seen in training get linear score −1, so they can still win on cosine | margin |
| `label_reg` | ridge regression from the sample vector to the embedding of its gold term, then the nearest term | margin |
| `retrieval_prior` | `retrieval` after centring both sides, + β·log(label frequency in the training fold) (`--prototype_beta`, 0.1) | margin |
| `prototype` | nearest class prototype: each term's centred vector blended with the centroid of its training samples (`--prototype_alpha`, 0.5), + the same log prior | margin |
| `prototype_open` | `prototype` over all 18.8k terms; a term without training samples keeps its term vector, + `--prototype_unseen_bonus` (default 0) | margin |

`knn_study`, `retrieval_prior` and `prototype(_open)` were added on 2026-09-28: they are the
"trivial" methods with the cheap upgrades that made them as accurate as `linear` (see
[experiments/README.md](experiments/README.md) for how they were chosen and every number). All
other methods are unchanged, and their predictions are byte-identical to the previous version.

**Metrics** (see `score()` in `metrics.json`):

| metric | meaning |
|---|---|
| `top1`, `top5` | accuracy of the first prediction, and of any of the five |
| `macro_top1` | mean per-label top-1. Rare labels count as much as frequent ones. |
| `top1_or_parent_child` | a hit if the prediction is the gold term or one `is_a` step away from it |
| `top1_gold_or_ancestor` | a hit if the prediction is the gold term or any `is_a` ancestor of it (true but coarser) |
| `top1_near_synonym` | a hit if the prediction's term vector has cosine ≥ 0.8 with the gold term's (near-synonyms) |
| `top1_confident_half` | top-1 on the 50 % of samples with the highest confidence |
| `top1_unseen_label` | top-1 on test samples whose gold term never occurs in the training fold (14–17 % of samples). Only retrieval-based methods can get these right. |
| `pred_gold_cosine` | cosine between the predicted and gold term embeddings. Gives partial credit for near-misses. |
| `margin_for_90pct_precision`, `coverage_at_90pct_precision` | the lowest confidence that keeps precision ≥ 90 %, and the share of samples kept at that threshold. Apply this threshold to the step 6 output. |
| `top1_by_domain` | top-1 per Metalog domain (animal / environmental / human / ocean) |

`predictions.tsv.gz` has one row per (test sample, slot, method), with gold, prediction,
top-5, confidence, domain and study. Every metric can be recomputed from it.

### Step 6: atlas labels

`--method` picks the model: `linear` (default, as before), `prototype` (needs `--term_vectors`)
or `knn_study`. Each is trained exactly as in step 5 and gives the same top-1 and confidence as
step 5 would on the same vectors (checked by `experiments/verify_atlas_methods.py`).

The score splits over the two blocks, e.g. `W_kw·kw + W_sb·sb + b` for `linear`. The script therefore
computes the keyword part once per distinct keyword text, streaming the unique `.h5` in chunks of
20k rows, and the sub-biome part once per distinct sub-biome. It then adds the two for each sample.
`prototype` also splits (the centring norm is expanded, see the script docstring). `knn_study` needs
a similarity to every training sample, so it is slower (chunks are capped at 4k rows).

- The model is trained on exactly the samples that were evaluated (same cap and seed), so the
  atlas labels come from the model that was scored.
- Samples without a sub-biome get only the keyword part of the score.
- The output has one row per sample: the term, label and confidence (margin) for each slot.

### Step 6b: coverage flag

The atlas models always answer with a Metalog label, including for habitats Metalog has no samples
of (food, laboratory, insects, plant tissue). `6b_coverage.py` adds, per atlas sample:

- `coverage_sim`: cosine similarity to the nearest of the model's training samples, in the model's
  feature space `[kw, sb] / √2` (samples without a sub-biome: keyword cosine);
- `nearest_study`: that training sample's study;
- `in_coverage`: `coverage_sim` ≥ threshold.

The threshold is the 5th percentile (`--quantile`) of the similarity of each Metalog training sample
to the training samples of *other* projects (`--fold_groups`), i.e. of how close a new Metalog study
is to the rest: 0.611. Both are rounded to 4 decimals, so the flag can be recomputed from the output
file. The flag depends only on the inputs, not on the method.

Cost: one similarity per distinct (keyword, sub-biome) text pair (1.6M for 3.4M samples) against the
distinct training vectors. Results (`experiments/README.md` §10): 8.6 % of the atlas is flagged;
flagged feature / material answers are far less reliable, while within Metalog the calibrated
probability already carries most of the signal.

### Step 6c: controls and mock communities

Blanks, negative controls and mock communities have no habitat. Metalog drops the ones it knows
(2b, bucket `control`), but the atlas models label every sample, so `6c_flag_controls.py` flags them
from the GPT texts:

- **sub-biome** `laboratory control / blank / mock / standard`, `mock community`, `negative control`;
- **strong keyword**: an extraction / PCR / kit / reagent / buffer blank or control, (DNA) blanks,
  no-template / empty / technical / processing control, ZymoBIOMICS or another community standard;
- **weak keyword**: negative / positive control, mock community / sample. These also appear in the
  study context of real samples (an untreated control group, a study that also sequenced a mock),
  so they count only with a lab-like sub-biome or a lab keyword (synthetic metagenome, sterile water).

`control` alone never counts (`control soil`, `healthy control`). Output: `sample_id, control,
control_evidence`; leave flagged samples unlabelled. 13,204 atlas samples (0.4 %) are flagged.
On the linked samples it finds half of Metalog's 58 controls (the others have nothing in their
metadata that says control), and every other hit is a control Metalog missed (audit hits, probiotic
positive controls); 37 of 40 random atlas hits are clear controls or mocks (experiments/README.md §12).

### Calibration and back-off (steps 5 and 6)

Metalog labels sit at different depths of ENVO/Uberon, and a wrong specific term is worse than a
correct general one. So instead of always giving the top-1, the answer can climb the ontology
(`hierarchy.py`; background in the project docs `hierarchical-prediction-literature.md` and
`hierarchical-backoff-results.md`):

1. **Probabilities.** The scores of `prototype` (or `linear`) over the slot's labels go through a
   softmax with a temperature fitted by log-likelihood on the other folds (cross-fitted).
2. **Summing up the ontology.** For every term, q = the summed probability of the labels at or below it.
   q never decreases going up, also with several parents.
3. **Climbing.** Among the top-1 label and its broader terms that Metalog uses in the slot, the
   answer is the most specific one with q ≥ τ. The slot roots (*biome*, *environmental material*,
   *environmental system*) are never an answer. Nothing qualifies → abstain.
4. **τ per slot** is the lowest one whose out-of-fold accuracy reaches the target. Strict accuracy
   counts the gold term or a coarser true term. Lenient accuracy also counts a more specific term
   than Metalog's (a manual review found most of those true). Step 5 reports both with τ chosen on
   the other folds. Step 6 reads τ from `calibration.json` (`--target_accuracy`, `--accuracy`).
5. **Several fold seeds.** τ and the temperature depend on the fold split (biome coverage at 90 %:
   0.73 / 0.79 / 0.79 for fold seeds 1 / 0 / 2). Step 6 accepts several `calibration.json` files of
   the same method and samples (`5_evaluate.py --fold_seed 1 --closed_only`, ...) and pools them
   (`hierarchy.merge_calibrations`): mean outcome shares per τ, accuracies recomputed, geometric-mean
   temperature, then one τ. The atlas uses fold seeds 0–2.

Step 6 refuses a calibration fitted with other training settings, and a resumed run whose chunks
were written with other back-off options (`run_settings.json`). Its new columns per slot:
`_p` (calibrated top-1 probability), `_backoff`, `_backoff_label`, `_backoff_p` (summed probability),
`_backoff_kind` (`top1`, `coarser`, `abstain`) and `_candidates` (top-5 + their broader slot labels
with probabilities, for step 7).

Out-of-fold results, prototype, raw labels, fold seed 0 (coverage = answered share; exact = share of
all samples answered with the gold term):

| slot | top-1 | 80 %: coverage / exact | 90 %: coverage / exact | 95 %: coverage / exact |
|---|---|---|---|---|
| biome | 0.509 | 0.68 / 0.43 | 0.46 / 0.28 | 0.23 / 0.16 |
| feature | 0.639 | 0.79 / 0.60 | 0.59 / 0.51 | 0.49 / 0.45 |
| material | 0.723 | 0.94 / 0.71 | 0.79 / 0.64 | 0.67 / 0.55 |

With the biome label map, biome back-off reaches 95 % at 66 % coverage (`hierarchical-backoff-results.md`).

### Step 7: LLM reranking

For the least-confident predictions (top-1 probability below the 30 % quantile of the out-of-fold
ones), an LLM chooses among the candidates: top-5 + their broader slot labels, each shown with its
synonyms, definition and the most similar training sample curators labelled with it, plus "none".
The prompt and answer parsing are `experiments/rerank_pilot.py`'s (v2, letter log-probabilities).

- **Fusion:** p ∝ exp((log p_base + w · log p_LLM) / T2) over the candidates, then back-off at τ2
  over the candidates (`hierarchy.decode_candidates`).
- **`fit`** picks w, T2 and τ2 per slot on the Metalog pilot (500 gated samples per slot): w gives the
  most exact answers (lenient: exact + too specific) at the target accuracy on the gated samples.
  w = 0 switches the LLM off for that slot. A nested study-grouped CV prints an honest estimate
  against the base model alone.
- **`build`** makes one request per distinct (slot, text, candidates) and prints the cost.
  **`submit` / `collect`** use the Batch API (half price), at most `--max_pending` batches in flight.
  **`apply`** writes `<slot>_final`, `_final_label`, `_final_p` and `_final_source` (`backoff`,
  `rerank`, or `backoff_no_llm_answer`).

Pilot (gpt-4.1-mini, top-5 + broader terms, nested CV on the gated samples, strict accuracy):

| target | slot | base alone: answered / accuracy / exact | + LLM: answered / accuracy / exact |
|---|---|---|---|
| 0.85 | biome | 0.60 / 0.805 / 0.164 | 0.68 / 0.873 / 0.268 |
| 0.90 | biome | 0.43 / 0.902 / 0.130 | 0.51 / 0.891 / 0.162 |
| 0.80–0.90 | feature | 0.00 (target unreachable on these samples) | ≤ 0.04 |

So far the LLM only pays off on biome. Feature's gated samples cannot reach 80 % accuracy with or
without it: the gold term is among the candidates for only half of them. Material has no gain on
the strict metric but gains on the lenient one (`--slots` decides which slots may use it).
Cost: about $0.22 per 1,000 requests with gpt-4.1-mini through the Batch API (~1,100 input tokens
each); for biome, about 30 % of the atlas samples, fewer after deduplication.

## Extending it

- **A new feature block** (for example 16S composition): write an `.npz` with `sample_ids`,
  `index` and `vectors` (the same layout as step 3), then pass it to `--features`. Nothing
  else changes.
- **Experiments** that motivated the methods (nested-CV ladders, 5 fold assignments, label
  processing) are in `experiments/`; `bash experiments/run_all.sh` reproduces all of them.
- **A new method**: write a function that returns `(top5_lists, confidence)` and add it to the
  `predictions` dict in `5_evaluate.main`. All metrics and outputs pick it up automatically.
- **A new metric**: add it to `score()`. It receives every prediction row of one
  (slot, method).
- **Different labels** (label cleanup, coarser granularity): write a new training set with
  the same columns. `analyses.py` (`granularity`) shows how to remap labels to ancestors.

## Known issues and limits

- **Human biome.** Metalog's human biome label is obsolete, so no human sample has a biome
  label. The atlas model predicts *animal-associated environment* for humans. Decide the
  convention before relying on it.
- **Distribution shift.** Every score comes from Metalog-linked samples, which are shotgun
  metagenomes and 56 % human. Most of the atlas is amplicon data from small studies. The atlas
  labels have not been checked by hand on unlinked studies. `6b_coverage.py` flags the samples
  farthest from Metalog (8.6 % of the atlas).
- **`hybrid` weight.** The weight 2 was chosen on this same cross-validation, so the gain of
  `hybrid` is slightly optimistic.
- **Linked studies.** A few Metalog study codes describe the same project, for example
  `TARA_Oceans_prokaryote` / `_protists` and `Stewart_2018` / `2019_cow_rumen`. They can end up
  in different folds. This affects < 2 % of samples.
- **Folds before 2026-09-24.** Earlier runs used sklearn's `GroupKFold`. Its unstable tie-break
  assigns studies to folds differently on different machines, and that moved scores by up to
  2.6 points (see the report). `study_folds` removes this.
- **Results before 2026-10-05 used another sample selection and other folds** (seeded shuffle,
  size-balanced folds). Re-running an older experiment with the current `common.py` gives slightly
  different numbers; the old code is in git history. With the old selection, adding the PO / FOODON
  labels redrew the subsample and moved the biome back-off coverage by 6 points (experiments/README.md
  §13–14).
- **Memory on a 4 GB machine.** Run `6_predict_atlas.py` with `--chunk_rows 2000` (the default 20k
  rows can map to ~250k samples per chunk). The final file is concatenated line by line from the parts.
- **Resume files.** `6_predict_atlas.py` (and `6b_coverage.py`) reuses `model.npz`, `index.npz` and `parts/` from its
  output directory. Delete the directory after changing any training option.
- **Accuracy targets are Metalog's.** τ, the reranker's weights and the gate are fitted on
  Metalog-linked studies. Check the back-off accuracy on a few hundred hand-labelled atlas samples
  (stratified by `_p`) before quoting it for the atlas.
- **Biome label map.** `biome_label_map.tsv` is a best-effort draft (23 rows marked `review`);
  re-run 2b, 5 and 6 after the review.
- **Pilot vs production base model.** The pilot's base probabilities came from the raw training set +
  the biome label map; production uses `training_set.clean`. The difference is small, but refit
  (`7_rerank_atlas.py fit`) on a pilot built from the same training set when the labels change much.
