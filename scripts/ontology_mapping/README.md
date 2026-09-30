# Ontology mapping prototype (MicrobeAtlas → ENVO / Uberon)

Gives every MicrobeAtlas sample one ENVO/Uberon term for each of the three Metalog
slots, plus a confidence score:

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
ENVO + Uberon .obo ──1──> ontology_terms.tsv.gz ─────────────┬──4──> term embeddings (.h5)
                                                              │                │
Metalog *_all_long_*.tsv.gz ─┐                                │                │
MicrobeAtlas sample.info.gz ─┴──2──> metalog_training_set.tsv.gz              │
                                          │  (sample_id, study, domain,        │
                                          │   3 labels, cleaned text)          │
GPT_{keywords,sub_biomes}.txt ─┐          │                                    │
unique GPT embeddings (.h5) ───┴──3──> keywords.npz, sub_biomes.npz            │
                                          │                                    │
                                          ├──5──> metrics.json, predictions ◄──┘   (evaluation)
                                          └──6──> atlas_predictions.tsv.gz         (all 3.4M samples)
```

| step | script | what it does | time |
|---|---|---|---|
| 1 | `1_build_term_index.py` | Parses the OBO files into one row per term: label, synonyms, definition, `is_a` parents, obsolete flag. | < 1 min |
| 2 | `2_build_training_set.py` | Links MicrobeAtlas records to Metalog samples by accession, and writes the labels and a cleaned text for each linked sample. | ~5 min |
| 2b | `2b_clean_metalog.py` | Flags every Metalog sample (controls, perturbed and ancient samples, duplicate accessions, unusable labels) and writes a clean and a gold version of the training set. Nothing is deleted. | < 1 min |
| 3 | `3_extract_sample_embeddings.py` | Looks up the GPT keyword or sub-biome embedding of each labelled sample (run once per kind). | < 1 min |
| 4 | `4_embed_terms.py` | Embeds every term text (`label; synonyms`) with the model used for the GPT texts. Costs about $0.01. | ~2 min |
| 5 | `5_evaluate.py` | Runs cross-validation grouped by study for every method on the same features. Writes metrics and per-sample predictions. | 5–15 min |
| 6 | `6_predict_atlas.py` | Trains `linear` (or `--method prototype / knn_study`) on the evaluated samples and labels every atlas sample, streaming the 6.4 GB keyword `.h5`. Resumable. | ~2 min (knn_study: longer) |
| – | `analyses.py` | Optional. Computes the extra analyses behind the findings report: bootstrap CIs, hierarchy of errors, label ceiling, coarser labels. | ~3 min |
| – | `common.py` | Shared helpers: term loading, sample selection, study folds. | |

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
        UBERON=https://raw.githubusercontent.com/obophenotype/uberon/master/uberon.obo
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

## How it works

### Step 1: terms

`parse_obo` is a minimal reader for OBO files. It keeps only `[Term]` stanzas whose ID has
the right prefix, and reads `id`, `name`, `def`, `synonym` (all scopes), `is_a` and
`is_obsolete`. IDs are written `ENVO_00001998`. List columns are joined with `||`.
Obsolete terms are kept but flagged. That lets step 2 report obsolete gold labels, and still
translate obsolete codes that appear in submitter text.

**Term text** (`common.term_text`) is `label; synonym; synonym`. It is used for term
embeddings and for TF-IDF retrieval. Definitions are left out on purpose: they bring in
generic words, such as country names, that match noise in the metadata.

### Step 2: labelled samples

- **Labels.** Each Metalog long table is pivoted to one row per sample. The first
  `[ENVO:…]` or `[UBERON:…]` code of each slot becomes the label. A code that is obsolete
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
  `other_ontology` (PO, FOODON, CL), `not_in_index`, `no_code`, `control_value` or `empty`.
  `<slot>_clean` is the usable id. `--label_map` (TSV: slot, from_id, to_id, reason) remaps
  ids, including obsolete and other-ontology ones; raw ids stay in `<slot>_raw`.
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
3. Shuffle with a fixed seed, then keep at most 50 samples per study. Without this cap, one
   cohort (for example 1,679 infant-gut samples) would dominate the scores.

The result is 17,723 samples from 539 studies.

**Folds** (`common.study_folds`): 5 folds, and all samples of a study go to the same fold,
because samples of one study share most of their text. The balancing is the same as
sklearn's `GroupKFold`, but ties are broken with a fixed seed, so the folds are identical on
every machine (see *Known issues*). `--fold_seed` picks a different split, which is how the
noise band in the report was measured.

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
  labels have not been checked by hand on unlinked studies.
- **`hybrid` weight.** The weight 2 was chosen on this same cross-validation, so the gain of
  `hybrid` is slightly optimistic.
- **Linked studies.** A few Metalog study codes describe the same project, for example
  `TARA_Oceans_prokaryote` / `_protists` and `Stewart_2018` / `2019_cow_rumen`. They can end up
  in different folds. This affects < 2 % of samples.
- **Folds before 2026-09-24.** Earlier runs used sklearn's `GroupKFold`. Its unstable tie-break
  assigns studies to folds differently on different machines, and that moved scores by up to
  2.6 points (see the report). `study_folds` removes this.
- **Resume files.** `6_predict_atlas.py` reuses `model.npz`, `index.npz` and `parts/` from its
  output directory. Delete the directory after changing any training option.
