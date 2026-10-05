# Ontology mapping prototype (MicrobeAtlas → ENVO / Uberon)

Gives every MicrobeAtlas sample one ontology term for each of the three Metalog slots, with a calibrated
probability, a "back-off" answer that is allowed to be coarser when the model is unsure, and quality flags.

| slot | Metalog field | MIxS equivalent | example (a human stool sample) |
|---|---|---|---|
| `biome` | `environment_biome` | env_broad_scale | `ENVO_01001002` animal-associated environment |
| `feature` | `environment_feature` | env_local_scale | `ENVO_2100002` intestine environment |
| `material` | `environment_material` | env_medium | `ENVO_00002003` fecal material |

Terms come from ENVO and Uberon, plus PO (plant anatomy) and FOODON (food), which Metalog uses for leaves,
fermented food products and milk.

**Approach: supervised label transfer.** 58,365 MicrobeAtlas samples are also in Metalog, whose curators
labelled the three slots by hand. Each sample is represented by the embedding of the keywords and the
sub-biome that GPT extracted from its free-text metadata (text-embedding-3-large, 1024 dims). A model
learns the mapping embedding → Metalog label on the linked samples, and is applied to all 3.4M atlas
samples. Zero-shot matching of the text to term names (no labels) was tried first and is far worse
(top-1 ≤ 0.14 over the whole ontology): the curators' choice of term is a convention that the term names
do not carry (see the findings report and `experiments/README.md`).

## Pipeline

```
ENVO / Uberon / PO / FOODON ─1─> ontology_terms.tsv.gz ──────────────┬──4──> term embeddings (.h5)
                                                                     │                 │
Metalog *_all_long / *_core_wide ─┐                                  │                 │
MicrobeAtlas sample.info.gz ──────┴──2──> metalog_training_set ─2b─> training_set.clean (labels cleaned)
                                                                     │                 │
GPT_{keywords,sub_biomes}.txt ─┐                                     │                 │
unique GPT embeddings (.h5) ───┴──3──> keywords.npz, sub_biomes.npz ─┤                 │
                                                                     ├──5──> metrics, predictions, calibration.json  (cross-validation)
                                                                     ├──6──> atlas_predictions.tsv.gz  (3.4M samples: term, p, back-off, candidates)
                                                                     ├──6b─> atlas_coverage.tsv.gz     (outside what Metalog covers?)
                                                                     ├──6c─> atlas_controls.tsv.gz     (blank / mock community?)
                                                                     └──7──> atlas_final.tsv.gz        (optional LLM reranking of the unsure ones)
```

| step | script | what it does | time (Mac VM) |
|---|---|---|---|
| 1 | `1_build_term_index.py` | Parses OBO (OWL for FOODON) into one row per term: label, synonyms, definition, `is_a` parents, obsolete flag | < 1 min |
| 2 | `2_build_training_set.py` | Links MicrobeAtlas records to Metalog samples by accession; writes the labels and a cleaned metadata text | ~5 min |
| 2b | `2b_clean_metalog.py` | Flags controls, perturbed / ancient samples, duplicates, unusable labels; applies the label map; writes the clean and gold training sets | < 1 min |
| 3 | `3_extract_sample_embeddings.py` | Looks up each labelled sample's GPT keyword / sub-biome embedding (run once per kind) | < 1 min |
| 4 | `4_embed_terms.py` | Embeds every term text with the model of the GPT texts (resumable; API, ~$0.09 for 49k terms) | ~12 min |
| 5 | `5_evaluate.py` | Study-grouped cross-validation of every method; calibration and back-off curves | 5–30 min |
| 6 | `6_predict_atlas.py` | Trains one method on the evaluated samples and labels the 3.4M atlas samples (streams the 6.4 GB keyword .h5) | 2–10 min |
| 6b | `6b_coverage.py` | Similarity to the nearest training sample; flags samples farther than 95 % of held-out Metalog samples | ~25 min |
| 6c | `6c_flag_controls.py` | Flags blanks, negative controls and mock communities from the GPT texts | < 1 min |
| 7 | `7_rerank_atlas.py` | Optional: an LLM chooses among the candidates of the least-confident samples (Batch API) | ≤ 24 h |

Steps 6, 6b and 7 are resumable (`--max_seconds`; finished chunks are kept).

### Code layout

| module | content |
|---|---|
| `common.py` | file helpers; term table, term vectors, ancestors, label coarsening; MicrobeAtlas record reader and text cleaning; **sample selection and study folds** (every script uses these, so all see the same rows) |
| `methods.py` | the mapping methods (`knn`, `knn_study`, `linear`, `hybrid`, `label_regression`, `prototype`, ...) used by steps 5 and 6 |
| `hierarchy.py` | calibration (temperature), ontology closure, climbing back-off, tau choice; used by steps 5, 6, 7 |
| `rerank_prompt.py` | the LLM reranker's prompt and answer parsing (pilot and step 7) |
| `experiments/` | every experiment behind a design choice, with its results (`experiments/README.md`) |
| `../embeddings/embed_subbiomes_keywords.py` | wrote the GPT embeddings; steps 3, 4 and 6 import its text cleaning so texts match byte for byte |

## Run it

```bash
cd ~/github/metadata_mining/scripts/ontology_mapping
P=~/MicrobeAtlasProject; L=$P/sidequest/latest; E=$L/embeddings; O=$P/ontology_mapping
TERMS=$P/ontology_terms.tsv.gz; TRAIN=$P/metalog/metalog_training_set.tsv.gz; CLEAN=$P/metalog/clean/training_set.clean.tsv.gz
KW=$P/metalog/keywords__large1024.npz; SB=$P/metalog/sub_biomes__large1024.npz
TV=$O/ontology_terms_unique_embeddings__text-embedding-3-large__dim1024.h5
KWH5=$E/GPT_keywords_unique_embeddings__text-embedding-3-large__dim1024__full.h5
SBH5=$E/GPT_sub_biomes_unique_embeddings__text-embedding-3-large__dim1024__full.h5

# 1. terms (current table = the 2026-09-23 ENVO / Uberon table + PO / FOODON appended)
python 1_build_term_index.py --append $P/ontology_terms_envo_uberon.tsv.gz \
  --obo PO=$P/ontologies/po.obo FOODON=$P/ontologies/foodon.owl.gz --output $TERMS
# 2. labelled samples, 2b. cleaning (+ the biome label map)
python 2_build_training_set.py --metalog_dir $P/metalog --sample_info $P/sample.info.gz --ontology_terms $TERMS --output $TRAIN
python 2b_clean_metalog.py --metalog_dir $P/metalog --ontology_terms $TERMS --training_set $TRAIN \
  --label_map $P/metalog/clean/biome_label_map.tsv --output_dir $P/metalog/clean
# 3. sample vectors, 4. term vectors (API key in ~/MicrobeAtlasProject/my_api_key_embeddings)
for kind in keywords sub_biomes; do
  python 3_extract_sample_embeddings.py --kind $kind --texts $L/GPT_$kind.txt --sample_ids $TRAIN \
    --unique_h5 $E/GPT_${kind}_unique_embeddings__text-embedding-3-large__dim1024__full.h5 --output $P/metalog/${kind}__large1024.npz
done
python 4_embed_terms.py --ontology_terms $TERMS
# 5. evaluation (fold seed 0 with every method; seeds 1-2 closed-vocabulary only, for a pooled calibration)
python 5_evaluate.py --ontology_terms $TERMS --samples $CLEAN --features $KW $SB --term_vectors $TV --output_dir $O/cv_backoff
for s in 1 2; do
  python 5_evaluate.py --ontology_terms $TERMS --samples $CLEAN --features $KW $SB --term_vectors $TV \
    --fold_seed $s --closed_only --output_dir $O/cv_backoff_s$s
done
# 6. atlas labels: prototype, back-off at 90 % strict accuracy (4 GB RAM: --chunk_rows 2000)
python 6_predict_atlas.py --ontology_terms $TERMS --samples $CLEAN --train_vectors $KW $SB \
  --keywords_texts $L/GPT_keywords.txt --sub_biomes_texts $L/GPT_sub_biomes.txt --keywords_h5 $KWH5 --sub_biomes_h5 $SBH5 \
  --method prototype --term_vectors $TV --target_accuracy 0.9 --accuracy strict \
  --calibration $O/cv_backoff/calibration.json $O/cv_backoff_s1/calibration.json $O/cv_backoff_s2/calibration.json \
  --output_dir $O/atlas_backoff
# 6b. coverage flag, 6c. controls
python 6b_coverage.py --samples $CLEAN --train_vectors $KW $SB --fold_groups $P/metalog/clean/project_groups.tsv \
  --index $O/atlas_backoff/index.npz --keywords_h5 $KWH5 --sub_biomes_h5 $SBH5 --output_dir $O/atlas_coverage
python 6c_flag_controls.py --keywords_texts $L/GPT_keywords.txt --sub_biomes_texts $L/GPT_sub_biomes.txt --output $O/atlas_controls.tsv.gz
# 7. optional LLM reranking: fit on the Metalog pilot (experiments/rerank_pilot.py), build, submit, collect, apply
X=$O/experiments/rerank; R=$O/rerank_atlas
python 7_rerank_atlas.py fit --pilot $X/pilot_k5anc.jsonl --responses $X/resp_k5anc_gpt-4.1-mini.jsonl \
  --calibration $O/cv_backoff/calibration.json --samples $CLEAN --output $R/rerank_params.json
python 7_rerank_atlas.py build --params $R/rerank_params.json --atlas_dir $O/atlas_backoff --sample_info $P/sample.info.gz \
  --samples $CLEAN --train_vectors $KW $SB --keywords_h5 $KWH5 --sub_biomes_h5 $SBH5 --work_dir $R
python 7_rerank_atlas.py submit --work_dir $R --limit 2000      # a trial; then without --limit
python 7_rerank_atlas.py collect --work_dir $R                  # rerun until all batches are collected
python 7_rerank_atlas.py apply --params $R/rerank_params.json --atlas_dir $O/atlas_backoff --work_dir $R
```

Join the outputs of 6, 6b, 6c and 7 on `sample_id`.

## How it works

### 1. Terms
A minimal OBO reader keeps `[Term]` stanzas with the ontology's prefix (`ENVO_00001998` style ids, list
columns joined with `||`); FOODON is read from OWL (`rdfs:label`, synonym properties, `IAO_0000115`
definition, named `rdfs:subClassOf` parents). PO's translated synonyms are skipped so term texts stay English.
Obsolete terms are kept and flagged: they are not valid labels but their codes appear in metadata.
**Term text** (`common.term_text`) = `label; synonym; ...`, e.g. `fecal material; droppings; frass; pellet`.
Definitions are left out: they embed far from keyword lists and add noise words (country names).

### 2. Labelled samples
- Each Metalog long table is pivoted to one row per sample; the first `[ENVO:…]` / `[UBERON:…]` code of each
  slot is the label. Obsolete codes are blanked: this removes the biome of **every human sample** (Metalog's
  `ENVO:00009003` is obsolete with no replacement).
- A MicrobeAtlas record is linked when its id, or any BioSample / SRA accession written inside it, equals
  Metalog's `spire_sample_name`: 58,365 records → 57,874 Metalog samples in 552 studies.
- `text` (`common.record_to_text`), used only by the TF-IDF baseline and the LLM reranker:
  `sample_env_material=feces [ENVO:00002003]` → `env_material: feces [fecal material]`; identifier, date and
  coordinate keys and missing values are dropped (they identify the study, not the habitat).

### 2b. Cleaning: flag, don't delete
`metalog_flags.tsv.gz` keeps every Metalog sample (159k) with its raw values and flags:
- **artificial bucket** from Metalog's `artificial` field: `control` (negative control, mock, spike-in →
  dropped: not a habitat), `perturbed` (cultivation, enrichment, sorted cells, mesocosm) and `degraded`
  (ancient, post-mortem, museum) → kept for training (the label is the source habitat), excluded from evaluation;
- **audit**: regexes over sample-level text find artificial samples Metalog did not flag (e.g. 40 negative
  controls labelled marine sediment); they go to `audit_review.tsv` for a human decision;
- **label status** per slot and the **label map** (`slot, from_id, to_id`): `biome_label_map.tsv` maps the
  54 non-biome terms used in the biome slot (e.g. *lentic water body* → *freshwater lake biome*) and the
  material *rhizosphere* / *rhizoplane* → *soil*. Raw ids stay in `<slot>_raw`.
Outputs `training_set.clean.tsv.gz` (training) and `training_set.gold.tsv.gz` (evaluation subset).

### 3–4. Vectors
A sample's vector is `[keywords, sub-biome] / √2` (two unit vectors, 2048 dims), so a dot product is the mean
of the two cosines. Terms are embedded with the same model; with two blocks the term vector `[t, t] / √2`
is comparable with sample vectors. `.npz` files store each distinct vector once (`vectors[index[i]]`).

### 5. Evaluation
- **Samples** (`common.select_samples`): ≥ 1 label, a vector in every block, at most 50 per study (the
  smallest seeded hash of the sample id): 17,720 samples, 539 studies. Without the cap one infant-gut study
  (1,679 samples) would dominate.
- **Folds** (`common.study_folds`): 5 folds by study (hash of the study code mod 5), so no study is in both
  train and test (samples of a study share most of their text). `--fold_groups` keeps whole sequencing
  projects together (TARA ×4, Stewart 2018/2019, Alneberg 2018/2020): supervised biome top-1 drops by
  ~1.3 points from that leak (≈ 3 points in total over 3 seeds, the rest is fold noise).
- **Methods** (`methods.py`, all with the same vectors; top-5 + a confidence per sample):

| method | idea | confidence |
|---|---|---|
| `majority` | most frequent training label | none |
| `retrieval` / `retrieval_open` | nearest term vector, among the slot's training labels / among all 49k terms | best cosine |
| `retrieval_prior` | centred retrieval + 0.1·log(label frequency) | margin |
| `knn` | similarity-weighted vote of the 25 nearest training samples | winner's vote share |
| `knn_study` | k = 50, each training *study* has one vote | winner's vote share |
| `linear` | one-vs-rest `RidgeClassifier` (LinearSVC on TF-IDF) | margin best − second |
| `hybrid` / `hybrid_open` | linear score + 2·cosine(sample, term); `_open`: unseen terms score −1 | margin |
| `label_reg` | ridge regression to the gold term's vector, then the nearest term | margin |
| `prototype` / `prototype_open` | prototype = ½ centred term vector + ½ centroid of its training samples; score = cosine + 0.1·log(frequency) | margin |

- **Metrics** (`5_evaluate.score`): top-1, top-5, macro top-1 (mean per label), top-1 counting the gold term
  *or an ancestor* (true but coarser), near-synonym credit (term cosine ≥ 0.8), top-1 on labels unseen in
  training, per domain, and the confidence cut-off for 90 % precision.
- **Calibration and back-off** (`hierarchy.py`), for `prototype` and `linear`:
  1. scores → probabilities: softmax(scores / T), T fitted on the other folds (T ≈ 0.09 prototype, 0.2 linear);
  2. q(node) = summed probability of the labels at or below the node (ontology closure, DAG-safe);
  3. answer = the most specific of {top-1, its broader slot labels} with q ≥ τ; none → abstain;
  4. τ per slot = the lowest τ whose out-of-fold accuracy reaches the target (strict: gold or a coarser true
     term; lenient: also a more specific one).

  Example: P(ocean biome) 0.55, P(marine biome) 0.15, P(freshwater biome) 0.15, τ = 0.8 → *aquatic biome*
  (q = 0.85) rather than a 55 %-sure *ocean biome*.

### 6. Atlas labels
The model is trained on exactly the evaluated samples (same cap and seed). Scores are split by block, e.g.
for `linear` `W_kw·kw + W_sb·sb + b`, so the 6.4 GB keyword file is streamed once in chunks and each distinct
keyword text (1.5M) and sub-biome text (32k) is scored once. `prototype` splits the same way (the centring norm
is expanded algebraically); `knn_study` needs similarities to every training sample (slower). With
`--calibration` (several fold seeds are pooled, `hierarchy.merge_calibrations`) each slot gets `_p`, `_backoff`,
`_backoff_label`, `_backoff_p`, `_backoff_kind` (`top1` / `coarser` / `abstain`) and `_candidates` (top-5 +
their broader slot labels, the reranker's options).

### 6b. Coverage flag
`coverage_sim` = cosine to the nearest training sample in the model's space; threshold = 5th percentile of the
same similarity for Metalog training samples against *other* projects (0.612). 8.5 % of the atlas is outside:
mostly plant tissue, food, laboratory and non-mammal animals.

### 6c. Controls and mocks
Rules over the GPT sub-biome and keywords (`laboratory control`, `mock community`, extraction / PCR / kit
blanks, ZymoBIOMICS ...); weak cues (`negative control`) count only with a laboratory context, and `control`
alone never counts (`healthy control`). 13,204 atlas samples (0.4 %); 37 of 40 random hits are real controls.

### 7. LLM reranking (optional)
For samples whose top-1 probability is in the least-confident 30 % (out of fold), an LLM chooses among the
candidates (prompt in `rerank_prompt.py`); its letter log-probabilities are fused with the base model,
p ∝ exp((log p_base + w·log p_LLM) / T2), and backed off again at τ2. `fit` chooses w, T2, τ2 per slot on the
Metalog pilot (w = 0 switches the LLM off for a slot); so far only biome benefits.

## Results (fold seed 0, `training_set.clean`, 17,720 samples)

| | biome | feature | material |
|---|---|---|---|
| majority | 0.285 | 0.471 | 0.445 |
| zero-shot nearest term, all terms | 0.028 | 0.136 | 0.083 |
| `knn_study` | 0.682 | 0.630 | 0.706 |
| `linear` | 0.671 | 0.624 | 0.713 |
| `prototype` (production) | 0.653 | 0.627 | 0.726 |
| `prototype` top-5 | 0.920 | 0.742 | 0.806 |
| `prototype` back-off at 90 % strict: answered / exact | 0.79 / 0.48 | 0.60 / 0.50 | 0.81 / 0.67 |

Biome numbers exclude human samples (no usable label) and use the biome label map. The trained methods are
within ~2 points of each other; the remaining errors are mostly label conventions (near-identical samples
from different studies share a biome label only 55 % of the time). Full tables, all experiments and the
fold-to-fold noise: `experiments/README.md` and the findings report.

## Extending it
- **A feature block** (e.g. 16S composition): write an `.npz` with `sample_ids`, `index`, `vectors` (as step 3)
  and add it to `--features`.
- **A method**: a function in `methods.py` returning `(top5 lists, confidence)`, added to `evaluate_fold` in
  `5_evaluate.py`; every metric picks it up.
- **A metric**: add it to `5_evaluate.score()` (it receives all prediction rows of one slot × method).
- **Other labels**: write a training set with the same columns (e.g. through `2b --label_map`).

## Known limits and open decisions
- **Human biome**: Metalog's human biome label is obsolete, so humans get *animal-associated environment*;
  the convention needs a decision.
- **Accuracy targets are Metalog's**: τ, the reranker weights and the gate are fitted on Metalog-linked
  studies (shotgun, 56 % human); the atlas is mostly amplicon data from small studies. The coarse gold check
  (1,021 hand-labelled atlas samples) supports the 90 % target for animal / soil / water samples only;
  check a few hundred atlas samples at the term level before quoting it.
- **The biome label map is a draft** (23 rows to review), and part of its gain is by construction.
- **Pending decisions**: strict vs lenient accuracy (are "more specific than Metalog" answers acceptable?),
  and the review of `audit_review.tsv`.
- **Linked samples get a prediction, not their curated label**: for the 58k Metalog-linked atlas samples the
  curated label could be published instead.
- **`rerank_atlas/` is from the 2026-10-01 atlas** (only a 2,000-request trial ran); rebuild it for the
  current `atlas_backoff`. 53 % of the gated samples have no record in `sample.info.gz`.
- **Memory**: on a 4 GB machine run step 6 with `--chunk_rows 2000`.
- **Resume files**: steps 6 and 6b reuse `model.npz`, `index.npz` and `parts/`; delete the output directory
  after changing a training option (the scripts refuse mismatching calibration or settings).
