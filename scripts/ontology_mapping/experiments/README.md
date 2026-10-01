# Experiments: upgrading the "trivial" embedding methods

These scripts reproduce every number behind `knn_study`, `retrieval_prior` and
`prototype(_open)` in `5_evaluate.py`, and behind the project note
*trivial-methods-upgrades.md*. They use the pipeline's own sample selection and folds
(`common.select_samples`, `common.study_folds`), so their numbers match `5_evaluate.py`.

Run everything from `scripts/ontology_mapping` (≈ 1.5–3 h on a laptop, most of it in the two
nested-CV ladders; finished `5_evaluate` runs are skipped when you re-run):

```bash
bash experiments/run_all.sh                  # writes ~/MicrobeAtlasProject/ontology_mapping/experiments/
python experiments/verify_atlas_methods.py   # step 6 == step 5 for every --method (~3 min)
```

| script | reproduces | output |
|---|---|---|
| `run_all.sh` | everything below, in order | `experiments/` in the output dir |
| `5_evaluate.py --fold_seed 0..4` (called by `run_all.sh`) | all methods on 5 study-to-fold assignments | `cv_seed0..4/` |
| `coarsen_labels.py` | a training set whose labels are replaced by their closest ancestor with ≥ 100 samples | `training_set__coarse100.tsv.gz`, then `cv_coarse100/` |
| `summarise_runs.py` | the tables of sections 1–2 below | `summary.txt`, `summary.json` |
| `trivial_knn.py` | the k-NN ladder (section 3) | `trivial_knn.json`, `.log` |
| `trivial_nearest_term.py` | the nearest-term ladder and the unseen-term bonus sweep (section 4) | `trivial_nearest_term.json`, `.log` |
| `term_text_variants.py` | build / embed / evaluate six term-text variants (section 5) | `variants.tsv.gz`, `term_variants.h5`, `results.json` |
| `term_keywords.py` | idea B: LLM-written, sample-style keywords for every term (section 5b) | `term_keywords_raw.jsonl`, `llm_variants.tsv.gz` |
| `verify_atlas_methods.py` | `6_predict_atlas.py --method X` gives the same top-1 and confidence as step 5's functions | prints OK / FAIL per method and slot |
| `cleaning_effect.py` | effect of `2b_clean_metalog.py` on training and evaluation, and an artificial-sample detector (see the end of this file) | `cleaning_effect.json`, `audit_review_with_detector.tsv` |
| `_setup.py` | shared loading (paths, samples, folds) | |

All numbers are top-1 accuracy on the 17,723 capped Metalog-linked samples (539 studies), with
keyword + sub-biome embeddings, 5 folds grouped by study.

## 1. All methods over 5 study-to-fold assignments (`summary.txt`)

Top-1, mean ± sd over `--fold_seed 0..4`; the difference to `linear` is paired (same test rows).

| method | biome | feature | material | vs linear (points): biome / feature / material, mean [min, max] |
|---|---|---|---|---|
| majority | 0.274 ± 0.000 | 0.477 ± 0.000 | 0.448 ± 0.000 | -24.2 [-24.6, -23.8] / -16.0 [-16.4, -15.6] / -26.4 [-27.3, -25.7] |
| retrieval | 0.181 ± 0.011 | 0.393 ± 0.007 | 0.251 ± 0.002 | -33.6 [-35.0, -31.7] / -24.4 [-25.1, -23.4] / -46.0 [-46.6, -45.3] |
| retrieval_open | 0.050 ± 0.000 | 0.141 ± 0.000 | 0.095 ± 0.000 | -46.6 [-47.0, -46.2] / -49.7 [-50.1, -49.3] / -61.7 [-62.6, -61.0] |
| **retrieval_prior** (new) | 0.475 ± 0.006 | 0.596 ± 0.003 | 0.677 ± 0.004 | -4.1 [-4.9, -2.8] / -4.2 [-4.6, -3.5] / -3.5 [-4.4, -3.0] |
| knn | 0.492 ± 0.009 | 0.616 ± 0.005 | 0.682 ± 0.007 | -2.4 [-3.8, -1.5] / -2.2 [-2.5, -1.5] / -2.9 [-3.6, -2.5] |
| **knn_study** (new) | 0.524 ± 0.008 | 0.639 ± 0.004 | 0.696 ± 0.003 | +0.8 [-0.3, +1.9] / +0.2 [-0.1, +0.7] / -1.6 [-2.3, -0.8] |
| linear | 0.516 ± 0.003 | 0.637 ± 0.003 | 0.711 ± 0.006 | — / — / — |
| hybrid | 0.514 ± 0.006 | 0.646 ± 0.003 | 0.714 ± 0.005 | -0.3 [-0.7, +0.6] / +0.8 [+0.2, +1.3] / +0.3 [+0.0, +0.5] |
| hybrid_open | 0.510 ± 0.005 | 0.646 ± 0.003 | 0.718 ± 0.006 | -0.7 [-0.9, -0.1] / +0.9 [+0.1, +1.3] / +0.7 [+0.5, +1.0] |
| **prototype** (new) | 0.507 ± 0.004 | 0.636 ± 0.003 | 0.723 ± 0.003 | -1.0 [-1.8, +0.1] / -0.1 [-0.7, +0.5] / +1.1 [+0.6, +1.5] |
| **prototype_open** (new) | 0.505 ± 0.002 | 0.634 ± 0.004 | 0.724 ± 0.003 | -1.2 [-1.6, -0.4] / -0.3 [-0.7, +0.3] / +1.2 [+0.7, +1.8] |
| label_reg | 0.530 ± 0.006 | 0.637 ± 0.003 | 0.726 ± 0.006 | +1.4 [+0.6, +2.7] / -0.1 [-0.5, +0.3] / +1.5 [+1.2, +1.9] |

- **`knn_study` fixes k-NN**: +3.2 / +2.3 / +1.4 points over `knn`, level with `linear` on biome
  and feature, 1.6 points behind on material.
- **`prototype` reaches `linear`** (−1.0 / −0.1 / +1.1), using only the term vectors, the label
  counts and the class centroids.
- **`retrieval_prior`**: the label frequencies alone lift nearest-term retrieval from
  0.18 / 0.39 / 0.25 to 0.48 / 0.60 / 0.68.
- Nothing beats `label_reg` / `hybrid` by more than ~1.5 points; all methods share the ceiling set
  by the labels (near-duplicates from different studies share a label 55 / 81 / 89 % of the time).

## 2. Label processing (`--fold_seed 0`; biome / feature / material)

| method | exact | near-synonym (cos ≥ 0.8) | gold or ancestor | coarser labels (≥ 100 samples, retrained) |
|---|---|---|---|---|
| majority | 0.274 / 0.477 / 0.448 | 0.274 / 0.477 / 0.448 | 0.297 / 0.477 / 0.448 | 0.274 / 0.477 / 0.448 |
| retrieval | 0.183 / 0.389 / 0.251 | 0.223 / 0.413 / 0.266 | 0.195 / 0.396 / 0.266 | 0.366 / 0.571 / 0.404 |
| retrieval_open | 0.050 / 0.141 / 0.095 | 0.090 / 0.196 / 0.112 | 0.067 / 0.176 / 0.099 | 0.030 / 0.114 / 0.059 |
| retrieval_prior | 0.474 / 0.599 / 0.678 | 0.492 / 0.618 / 0.686 | 0.552 / 0.603 / 0.701 | 0.521 / 0.609 / 0.712 |
| knn | 0.476 / 0.610 / 0.672 | 0.500 / 0.624 / 0.688 | 0.577 / 0.623 / 0.702 | 0.516 / 0.630 / 0.720 |
| knn_study | 0.511 / 0.634 / 0.693 | 0.535 / 0.652 / 0.704 | 0.614 / 0.656 / 0.727 | 0.558 / 0.666 / 0.736 |
| linear | 0.514 / 0.634 / 0.708 | 0.532 / 0.646 / 0.730 | 0.622 / 0.652 / 0.748 | 0.564 / 0.653 / 0.742 |
| hybrid | 0.511 / 0.647 / 0.713 | 0.536 / 0.664 / 0.733 | 0.597 / 0.663 / 0.751 | 0.574 / 0.675 / 0.750 |
| hybrid_open | 0.507 / 0.647 / 0.714 | 0.533 / 0.664 / 0.733 | 0.587 / 0.662 / 0.751 | 0.568 / 0.668 / 0.744 |
| prototype | 0.509 / 0.639 / 0.723 | 0.525 / 0.657 / 0.734 | 0.609 / 0.650 / 0.752 | 0.560 / 0.661 / 0.747 |
| prototype_open | 0.504 / 0.636 / 0.726 | 0.525 / 0.655 / 0.737 | 0.615 / 0.648 / 0.755 | 0.555 / 0.660 / 0.750 |
| label_reg | 0.525 / 0.636 / 0.727 | 0.555 / 0.653 / 0.741 | 0.629 / 0.647 / 0.762 | 0.569 / 0.663 / 0.751 |

- Near-synonym credit, ancestor credit and coarser labels lift every trained method by similar
  amounts; the ranking does not change. Coarser labels: biome 95 → 43, feature 202 → 85,
  material 179 → 50 labels (ancestors outside the term index, e.g. BFO roots, are not used).
- Human samples (7,546 of 17,723) have no biome label at all (obsolete ENVO:00009003), so every
  biome number here is on non-human samples only.

## 3. k-NN ladder (`trivial_knn.py`, `--fold_seed 0`, nested CV)

Each row adds one idea; every hyper-parameter is chosen by nested CV (3 study-grouped inner folds
inside each outer training fold): k ∈ {5, 10, 25, 50, 100}, similarity temperature τ ∈ {none, 0.1,
0.05, 0.02}, one vote per study yes/no, class-prior exponent ∈ {0, 0.25, 0.5, 0.75}.

| row | biome | feature | material |
|---|---|---|---|
| A kNN-25 majority, keywords only (old baseline) | 0.462 | 0.590 | 0.648 |
| B + keywords & sub-biome | 0.477 | 0.609 | 0.672 |
| C + tuned k and similarity weights | 0.485 | 0.614 | 0.676 |
| D + one vote per study | 0.481 | 0.640 | 0.701 |
| E + class-prior correction | 0.481 | 0.638 | 0.701 |
| F E on centred embeddings | 0.491 | 0.634 | 0.707 |
| G E on whitened embeddings (256 dims) | 0.466 | 0.614 | 0.665 |
| H E with hubness correction (CSLS) | 0.501 | 0.629 | 0.702 |
| I E with Wilson editing of training set | 0.481 | 0.624 | 0.709 |
| J LDA classifier | 0.504 | 0.635 | 0.686 |
| J kNN-50, one vote per study, in LDA space | 0.520 | 0.639 | 0.692 |
| ridge (reference) | 0.514 | 0.634 | 0.708 |

For comparison, the fixed `knn_study` (k = 50, unweighted, one vote per study, *not* tuned) on the
same folds: 0.511 / 0.634 / 0.693 (section 2).

- **Input (B) and one vote per study (D) are the two upgrades that matter** for feature and
  material (+1.9 / +2.4 and +2.6 / +2.5).
- **On biome the nested tuning is noisy**: it picks k between 5 and 100 from fold to fold and ends
  at 0.481, below the fixed `knn_study` (0.511). With ~10k labelled biome samples in ~400 studies,
  the inner folds are too small to tune a large grid reliably. This is why `knn_study` uses a fixed
  default, checked over 5 splits in section 1, rather than per-run tuning.
- **Standard retrieval tricks** (similarity weights, prior correction, centring, hubness
  correction, editing) move results by ±1–2 points without a consistent direction; PCA whitening
  hurts everywhere. LDA metric learning ≈ `knn_study`.


## 4. Nearest-term ladder (`trivial_nearest_term.py`, `--fold_seed 0`, nested CV)

Closed vocabulary = the training fold's labels for that slot.

| row | uses labels? | biome | feature | material |
|---|---|---|---|---|
| N-A keywords → nearest term (old baseline) | no | 0.259 | 0.453 | 0.106 |
| N-B sub-biome as the query | no | 0.101 | 0.151 | **0.317** |
| N-C keywords + sub-biome as the query | no | 0.183 | 0.389 | 0.251 |
| N-D + modality-gap centring | no | 0.152 | 0.193 | 0.296 |
| N-E + ancestor smoothing of term vectors | no | 0.158 | 0.193 | 0.296 |
| N-F + log label frequency (→ `retrieval_prior`) | counts | 0.484 | 0.610 | 0.682 |
| N-G + blend with training centroids (→ `prototype`) | yes | 0.509 | 0.640 | 0.718 |
| N-P nearest centroid only | yes | 0.435 | 0.584 | 0.634 |
| N-H `label_reg` | yes | 0.525 | 0.636 | 0.727 |
| N-O1 open vocabulary, keywords | no | 0.049 | 0.161 | 0.063 |
| N-O2 open vocabulary, kw+sb, centred | no | 0.038 | 0.092 | 0.092 |

- Without labels nothing helps except, for material, querying with the short sub-biome
  (0.106 → 0.317): a one-phrase query matches a material's name much better than a keyword list.
- The label counts are the big step (+23 / +16 / +58 points over N-A).

**Unseen-term bonus for `prototype_open`** (centred kw+sb, α = 0.5, β = 0.1): top-1 overall /
top-1 on test samples whose label never occurs in the training fold (14–17 % of samples):

| bonus | biome | feature | material |
|---|---|---|---|
| 0 (default) | 0.506 / 0.000 | 0.638 / 0.000 | 0.723 / 0.000 |
| 0.4 | 0.482 / 0.019 | 0.635 / 0.014 | 0.731 / 0.069 |
| 0.5 | 0.462 / 0.057 | 0.628 / 0.055 | **0.733 / 0.114** |
| 0.6 | 0.411 / 0.097 | 0.606 / 0.066 | 0.721 / 0.132 |
| 0.8 | 0.204 / 0.153 | 0.474 / 0.080 | 0.529 / 0.195 |

A bare term vector scores lower than a centroid-blended prototype, so with no bonus an unseen term
never wins. For material, a bonus of 0.4–0.5 recovers 7–11 % of unseen labels *and* slightly raises
overall accuracy; for biome it only costs accuracy. The bonus is tuned on this same CV, so treat
these gains as optimistic (like `--hybrid_weight`).

## 5. What text represents a term? (`term_text_variants.py`)

Six texts per term, embedded with text-embedding-3-large @ 1024 (about $0.36), scored with the same
models as `5_evaluate.py` (`label_syn`, the current text, reproduces its numbers exactly). Top-1
biome / feature / material, mean over fold seeds 0–4:

| method | `label` | `label_exact` | `label_syn` | `label_syn_def` | `label_syn_parents` | `label_syn_def_parents` |
|---|---|---|---|---|---|---|
| retrieval (slot vocabulary) | 0.204 / 0.419 / 0.442 | 0.207 / 0.403 / 0.320 | 0.181 / 0.393 / 0.251 | 0.176 / 0.264 / 0.189 | 0.138 / 0.493 / 0.172 | 0.184 / 0.158 / 0.260 |
| retrieval, sub-biome query | 0.136 / 0.223 / 0.462 | 0.107 / 0.229 / 0.346 | 0.097 / 0.167 / 0.320 | 0.093 / 0.156 / 0.303 | 0.075 / 0.270 / 0.200 | 0.088 / 0.129 / 0.339 |
| retrieval_open (all terms) | 0.033 / 0.128 / 0.339 | 0.051 / 0.192 / 0.142 | 0.050 / 0.141 / 0.095 | 0.035 / 0.080 / 0.051 | 0.047 / 0.261 / 0.094 | 0.035 / 0.087 / 0.091 |
| retrieval_open, unseen labels | 0.143 / 0.122 / 0.213 | 0.149 / 0.097 / 0.194 | 0.134 / 0.101 / 0.186 | 0.112 / 0.080 / 0.092 | 0.122 / 0.106 / 0.148 | 0.132 / 0.128 / 0.121 |
| retrieval_prior | 0.441 / 0.598 / 0.680 | 0.466 / 0.600 / 0.680 | 0.475 / 0.596 / 0.677 | 0.458 / 0.599 / 0.680 | 0.467 / 0.586 / 0.671 | 0.451 / 0.596 / 0.680 |
| prototype | 0.503 / 0.640 / 0.720 | 0.506 / 0.639 / 0.722 | 0.507 / 0.636 / 0.723 | 0.504 / 0.638 / 0.724 | 0.502 / 0.637 / 0.720 | 0.504 / 0.637 / 0.723 |
| prototype_open (bonus 0.5) | 0.463 / 0.635 / 0.729 | 0.478 / 0.636 / 0.731 | 0.477 / 0.633 / 0.732 | 0.482 / 0.634 / 0.729 | 0.479 / 0.632 / 0.725 | 0.484 / 0.635 / 0.730 |
| prototype_open, unseen labels | 0.068 / 0.067 / 0.123 | 0.077 / 0.063 / 0.116 | 0.034 / 0.052 / 0.110 | 0.022 / 0.025 / 0.068 | 0.017 / 0.023 / 0.048 | 0.018 / 0.024 / 0.069 |
| label_reg | 0.531 / 0.636 / 0.723 | 0.530 / 0.637 / 0.720 | 0.530 / 0.637 / 0.726 | 0.538 / 0.638 / 0.719 | 0.541 / 0.638 / 0.718 | 0.539 / 0.638 / 0.720 |

- **Trained methods do not care** (prototype, retrieval_prior within ±1 point; label_reg +1 on
  biome with definitions or parents, −0.5 on material). The class centroids and label counts carry
  the signal, not the term text.
- **Zero-shot retrieval swings a lot, but mostly on one decision.** 45 % of material and 48 % of
  feature labels are *fecal material* / *intestine environment*, and the variants mainly change which
  of the gut terms wins. With `label`, *fecal material* is predicted for 29 % of material samples;
  with `label_syn` (its synonyms are "droppings; frass; pellet") only 12 %, and *intestine
  environment* takes over. Macro top-1 (mean per label), which is not dominated by that one term,
  moves only a few points (slot vocabulary: biome 0.29–0.32, feature 0.31–0.33, material 0.32–0.38).
- **Definitions hurt zero-shot** (feature 0.393 → 0.264, material 0.251 → 0.189): a sentence-long
  definition embeds far from a keyword list. **Parents help feature** (0.393 → 0.493: "Is a:
  digestive tract environment" pulls gut samples to *intestine environment*) and hurt the others.
- **The plain label is the best zero-shot text** overall, and for terms never seen in training:
  open retrieval material 0.095 → 0.339, prototype_open unseen-label recovery 3.4 / 5.2 / 11.0 % →
  6.8 / 6.7 / 12.3 %.

### 5b. Terms described like samples (`term_keywords.py`, idea B)

For each term, gpt-5.1 imagines 3 microbial samples that would carry the term and writes the same
fields the production prompt extracts from sample metadata (5-8 keywords in curly brackets and a
sub-biome; prompt: `term_keywords_prompt.txt`; production sampling settings). Three new variants:
`llm_kw` (average of the imagined keyword lists), `llm_kw_sb` (keyword block + sub-biome block, like
the samples) and `label_llm_kw_parents` (label + imagined keywords + parent labels).

```bash
X=~/MicrobeAtlasProject/ontology_mapping/experiments/term_text
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl --dry_run      # cost estimate
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl --max_terms 50 # pilot, measured cost
python3 experiments/term_keywords.py generate --raw $X/term_keywords_raw.jsonl                # all terms (resumable)
#   or through the Batch API (about half the price, results within 24 h; terms already in --raw are skipped):
python3 experiments/term_keywords.py batch_submit --raw $X/term_keywords_raw_v2.jsonl --dry_run
python3 experiments/term_keywords.py batch_submit --raw $X/term_keywords_raw_v2.jsonl
python3 experiments/term_keywords.py batch_collect --raw $X/term_keywords_raw_v2.jsonl  # rerun until all collected
#   state in <raw>.batch.json; submit/generate refuse to run while a batch is uncollected (no double paying);
#   failed requests or batches just leave their terms missing, and another batch_submit retries only those.
#   A batch rejected for the enqueued-token limit: collect it, then batch_submit --batch_requests 300
python3 experiments/term_keywords.py build --raw $X/term_keywords_raw.jsonl --output $X/llm_variants.tsv.gz
python3 experiments/term_text_variants.py embed --variants $X/llm_variants.tsv.gz --output $X/term_variants.h5
python3 experiments/term_text_variants.py evaluate --variants $X/variants.tsv.gz $X/llm_variants.tsv.gz \
  --vectors $X/term_variants.h5 --only label label_syn llm_kw llm_kw_sb label_llm_kw_parents --output $X/results_llm.json
```

`evaluate` now also reports macro top-1 (`| macro`), because micro top-1 of zero-shot retrieval is
dominated by the *fecal material* / *intestine environment* decision.

## 6. Verification

| check | result |
|---|---|
| old methods of `5_evaluate.py` (all metrics and every prediction row), new code vs commit b6710e6, fold_seed 0 | identical |
| new methods, two runs (before / after the memory change in `rank`) | identical |
| `verify_atlas_methods.py`: step 6 vs step 5 functions on the 50,100 linked samples, 3 methods × 3 slots, cloud and Mac | 100 % identical top-1, confidence within 0.0001 |
| `6_predict_atlas.py --method linear` on the real 3.44M atlas, new code vs commit b6710e6 | byte-identical output |
| `--method prototype` / `knn_study` on the real atlas (Mac VM, 4 cores) | 1.5 min / ~30 min (resumable); pairwise agreement with linear 80–83 %, all three agree on 72–74 % |
| `coarsen_labels.py`, `trivial_knn.py --rows A,B`: Mac vs cloud | identical output |

## Differences from the first version of the note

The first analysis (trivial-methods-upgrades.md, 26 Sep) used sklearn `GroupKFold` and its own
random fold assignments; these scripts use the pipeline's `study_folds`, so single numbers differ by
up to ~2 points (fold noise), but every conclusion holds. Two corrections:

- the old nearest-term baseline built its closed vocabulary from all labels, test folds included;
  with the training-fold vocabulary it is 0.259 / 0.453 / 0.106 (was 0.281 / 0.480 / 0.148);
- `prototype_open` does **not** recover unseen labels unless `--prototype_unseen_bonus` is raised
  (table above).

## Does the Metalog cleaning help? (`cleaning_effect.py`, 2026-09-29)

```bash
python experiments/cleaning_effect.py --clean_dir ~/MicrobeAtlasProject/metalog/clean \
  --output ~/MicrobeAtlasProject/metalog/clean/experiments/cleaning_effect.json   # ~4 min
```

`linear`, keyword + sub-biome embeddings, the pipeline's 50-per-study cap and folds (the `raw` arm
on the raw test set is the 0.514 / 0.634 / 0.708 of fold seed 0 above). Fold seeds 0–2; CIs are a
paired bootstrap over test studies (seed 0), in points vs `raw`.

**A. Cleaning the training data does nothing measurable.** Fixed gold test set; accuracy is the
mean over the 3 seeds, the difference [95 % CI] is seed 0. Only 3 controls survive the cap in the
linked set (hence `no_controls` ≈ `raw`); the perturbed samples (~930) are what the arms remove.

| training arm | biome | feature | material |
|---|---|---|---|
| raw | 0.521 | 0.641 | 0.733 |
| no_controls | 0.520 (+0.0 [−0.1, 0.1]) | 0.641 (+0.0 [−0.0, 0.1]) | 0.733 (+0.0 [0.0, 0.0]) |
| no_artificial | 0.520 (−0.7 [−2.4, 0.8]) | 0.639 (−0.3 [−0.9, 0.2]) | 0.735 (+0.2 [−0.1, 0.6]) |
| no_audit | 0.522 (−0.6 [−2.4, 0.8]) | 0.639 (−0.3 [−1.0, 0.3]) | 0.736 (+0.3 [−0.1, 0.6]) |
| dedup | 0.523 (−0.8 [−3.2, 1.6]) | 0.631 (−0.8 [−1.6, 0.0]) | 0.734 (−0.2 [−1.0, 0.5]) |

Perturbed samples cost nothing and carry rare real labels (biome macro 0.101 → 0.093 without
them), so they stay in training. Deduplicating *training* texts slightly hurts; dedupe the test only.

**B. It changes the measurement.** The raw-trained model, by group of test sample:

| group | biome | feature | material |
|---|---|---|---|
| gold | 0.521 | 0.642 | 0.734 |
| perturbed + degraded | 0.361 | 0.312 | 0.430 |
| unreviewed audit hits | 0.254 | 0.190 | 0.395 |
| repeated text in study | 0.607 | 0.734 | 0.706 |

Raw → gold test: +0.6 / +0.5 / +2.4 points. The pending review can move the gold numbers by
≤ 0.5 points (`test_gold` vs `test_gold_audit_ok`).

**C. An artificial-sample detector works across studies.** Logistic regression on the same vectors,
study-grouped CV, 1,879 artificial samples from 58 studies: ROC AUC 0.82, AP 0.35 (base rate
0.055); precision 0.78 at recall 0.10 (score ≥ 0.9); controls: recall 0.77 at 0.5. Never trained on
the audit hits, it scores them 0.42 on average, like Metalog-flagged samples (0.38) and far above
gold (0.08). `audit_review_with_detector.tsv` adds its mean score per review row.

## 7. Hierarchical back-off + simulated reranking (`hierarchical_backoff.py`, 2026-10-01)

Calibrated base probabilities (cross-fitted temperature) -> sum up the ontology (q = P @ A) ->
climb from the top-1 label to the most specific node with q >= tau; tau chosen on the other folds
for a target hierarchical accuracy. A flat baseline (abstain below tau) is scored the same way.
The reranker is simulated (accuracy r inside the top-5, confidence AUROC 0.85) on the gated
least-confident share, so no API call is needed. Results: project doc `hierarchical-backoff-results.md`.

```bash
X=~/MicrobeAtlasProject/ontology_mapping/experiments
python3 experiments/hierarchical_backoff.py --fold_seeds 3 --models prototype linear \
  --gates 0.2 0.3 0.5 1.0 --rerank_acc 0.7 0.85 1.0 --output $X/backoff/results.json   # ~10 min
```

### 7b. Biome-slot label map (`draft_biome_label_map.py`, 2026-10-01)

45 % of Metalog biome labels are not ENVO biomes. This drafts, for each of the 54 such terms, the
biome to use instead (cross-study kNN votes + a biome-only prototype + closest ENVO biome term by
label), drops the slot root 'biome', and keeps animal- / plant-associated environment as
conventions. Output = the `2b_clean_metalog.py --label_map` format plus evidence columns for
reviewers (`review_decision`, `review_notes`). `hierarchical_backoff.py --label_map` applies it.

```bash
python3 experiments/draft_biome_label_map.py --output ~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv
python3 experiments/hierarchical_backoff.py --fold_seeds 3 --models prototype --gates 0.3 0.5 --rerank_acc 0.85 1.0 \
  --label_map ~/MicrobeAtlasProject/metalog/clean/biome_label_map.tsv --output $X/backoff/results_mapped.json
```

### 7c. Real reranker pilot (`rerank_pilot.py`, 2026-10-01)

Measures the reranker accuracy inside the top-5 (r) and its confidence AUROC on 1,500 gated
least-confident out-of-fold samples (500 per slot; fold seed 0, biome label map applied), which the
simulation in section 7 needs. `pilot.jsonl` is already built; run the API step on the Mac:

```bash
R=~/MicrobeAtlasProject/ontology_mapping/experiments/rerank
python3 experiments/rerank_pilot.py run --pilot $R/pilot.jsonl --model gpt-4.1-mini --output $R/resp_gpt-4.1-mini.jsonl --dry_run
python3 experiments/rerank_pilot.py run --pilot $R/pilot.jsonl --model gpt-4.1-mini --output $R/resp_gpt-4.1-mini.jsonl
python3 experiments/rerank_pilot.py run --pilot $R/pilot.jsonl --model gpt-5.1 --output $R/resp_gpt-5.1.jsonl
python3 experiments/rerank_pilot.py score --pilot $R/pilot.jsonl --responses $R/resp_gpt-4.1-mini.jsonl
```

## 8. Gold-set check of the atlas back-off (`gold_check.py`, 2026-10-01)

Checks `atlas_backoff/` (prototype, 90 % strict target) on the 1,091 hand-labelled gold samples
(`gold_dict.pkl`), outside Metalog except 24 (excluded). Gold labels are coarse (animal / plant /
soil / water / other, ~210 each), so each predicted term is turned into coarse biomes
automatically. For each term, take the GPT coarse biome of the Metalog-linked samples labelled
with it or a descendant; compatible = every coarse biome with a share ≥ 0.15
(`term_to_coarse_biome.tsv`). No curator is needed.

```bash
bash experiments/gold_check_extract.sh            # ~30 s
P=~/MicrobeAtlasProject
python3 experiments/gold_check.py --dir $P/ontology_mapping/experiments/gold_check \
  --ontology_terms $P/ontology_terms.tsv.gz --training_set $P/metalog/clean/training_set.clean.tsv.gz \
  --cv_predictions $P/ontology_mapping/cv_backoff/predictions.tsv.gz
```

Back-off answers, 1,021 gold samples with a prediction. *Coarse-consistent* = gold coarse biome
compatible with the answer:

| slot | answered (gold) | answered (Metalog CV) | coarse-consistent: animal/soil/water | plant | other |
|---|---|---|---|---|---|
| biome | 0.64 | 0.79 | 0.97 (81 % answered) | 0.94 (41 %) | 0.64 (38 %) |
| feature | 0.18 | 0.61 | 0.98 (26 %) | 0.00 (n = 5) | 0.38 (11 %) |
| material | 0.43 | 0.78 | 0.97 (60 %) | 0.03 (17 %) | 0.22 (20 %) |

**Reading the coarse metric:**
- It is noisy. On the 24 linked gold samples, Metalog's *own* labels are coarse-consistent only
  50 / 83 / 74 % of the time under the strict (majority) map. Gold's classes follow the GigaScience
  paper's conventions, e.g. rhizosphere → plant, wastewater → water, food / bioreactor / lab → other.
- So the 151 answered-but-inconsistent cases were reviewed by rule (`gold_check_disagreements_reviewed.tsv`,
  one reviewer, Claude). 66 were defensible (bioreactor → sludge / activated sludge / wastewater
  treatment plant; rhizosphere / compost / litter → soil; sponge, coral → marine biome), 30
  arguable, 46 wrong, 9 with uninformative gold.
- After review, answered back-off terms are coarse-correct 97 / 95 / 92 % (arguable counted half),
  or 97 / 93 / 91 % (arguable counted wrong).

**Conclusions:**
1. **The 90 % target holds at the coarse level outside Metalog.** Coarse correctness is necessary
   but not sufficient for the exact term, so this does not prove 90 % strict.
2. **On unfamiliar samples the model abstains more than on Metalog:** biome 64 vs 79 % answered,
   feature 18 vs 61 %, material 43 vs 78 %. Plant and other are answered least. Coverage, not
   precision, is the cost out of domain.
3. **Confidence ranks well on gold too.** Top-1 coarse consistency rises with the calibrated p:
   biome 0.65 → 0.77 → 0.88 → 0.94 for p < 0.5 / 0.5–0.75 / 0.75–0.9 / ≥ 0.9.
4. **Real gaps:**
   - plant anatomy: leaf / root / whole plant → material *soil* or *organic material*;
   - "other" habitats Metalog barely has (laboratory, air, food, urban) get forced into
     marine / water / intestine terms.
5. **Re-weighted to the atlas mix** (62 % animal), coarse-strict top-1 is 0.81 / 0.88 / 0.81.

Not used: `atlas_final` (rerank). Only the 2,000-request trial was run, so 6 of the gold samples
were reranked.

## 9. Project-level folds (`project_groups.py`, `5_evaluate.py --fold_groups`, 2026-10-01)

```bash
python3 experiments/project_groups.py --sample_info ~/MicrobeAtlasProject/sample.info.gz \
  --training_set ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --output ~/MicrobeAtlasProject/metalog/clean/project_groups.tsv        # ~1.5 min
python3 5_evaluate.py ... --fold_groups ~/MicrobeAtlasProject/metalog/clean/project_groups.tsv
```

**How groups are formed.** Study codes are merged when their samples share an SRA study or a
BioProject accession: Stewart 2018/2019 and Alneberg 2018/2020. `--merge 'TARA_*'` adds the four
TARA codes, because prokaryote and protist size fractions come from the same stations (63 of 64
shared). Generic shared titles ("human gut metagenome Metagenome") are *not* merged. The result is
552 → 547 groups; after the per-study cap, 378 evaluated samples are in merged projects.

`training_set.clean`, kw+sb, fold seeds 0–2, top-1:

| slot | method | study folds | project folds | merged-project samples: study → project folds |
|---|---|---|---|---|
| biome | linear | 0.680 | 0.650 | 0.99 → 0.64 |
| biome | label_reg | 0.681 | 0.652 | 0.99 → 0.62 |
| biome | knn_study | 0.680 | 0.652 | 0.95 → 0.36 |
| biome | prototype | 0.641 | 0.642 | 0.74 → 0.66 |
| feature | linear | 0.641 | 0.635 | 0.41 → 0.10 |
| material | linear | 0.703 | 0.715 | 0.53 → 0.67 |

**Conclusions:**
- **The leak is real but small.** Merged-project samples (3.8 % of biome rows, 2.2 % of
  feature / material rows) lose 30–60 points on biome once whole projects are held out. That is
  worth about −1.3 / −0.7 points of biome / feature top-1 for the supervised methods.
- **The rest of the gap is fold noise.** The remaining biome difference on other samples
  (−1.7 for linear) is within the fold-to-fold spread, which is ±1.4 points with the cleaned
  labels (linear: 0.656–0.683 over 3 study-fold seeds).
- **prototype is barely affected.** It leans on the term vector and the label prior, not on
  near-copies of the test study.
- **Recommendation:** report with `--fold_groups` from now on, and use ≥ 5 fold seeds for biome
  comparisons.
- Also fixed: `top1_confident_half` and `margin_for_90pct_precision` now sort confidences
  stably. Ties (majority, kNN vote shares) used to rank differently on the Mac and on Linux, so
  those two metrics differed slightly between machines; top-1 never did.

