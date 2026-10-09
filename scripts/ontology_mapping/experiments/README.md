# Experiments behind the ontology-mapping pipeline

Every design choice of the pipeline was tested by a script in this folder. They use the pipeline's own
sample selection and folds (`common.select_samples`, `common.study_folds`), so their numbers are comparable
with `5_evaluate.py`. Run them from `scripts/ontology_mapping` (e.g. `python experiments/trivial_knn.py ...`);
default paths are those of `~/MicrobeAtlasProject` (`_setup.DEFAULTS`). Outputs go to
`~/MicrobeAtlasProject/ontology_mapping/experiments/` unless stated.

Unless stated: top-1 accuracy, biome / feature / material, keyword + sub-biome embeddings, 5 study-grouped
folds, ≤ 50 samples per study (17.7k samples, 539 studies). Biome excludes human samples (no usable label).

## Index

| § | question | script | answer |
|---|---|---|---|
| 1 | How do all methods compare, robustly? | `5_evaluate.py --fold_seed 0..4`, `summarise_runs.py` (`run_all.sh`) | Trained methods tie within ~1.5 points; zero-shot is far behind |
| 2 | Do hierarchy-aware metrics or coarser labels change the ranking? | `coarsen_labels.py`, `analyses.py` | They lift every method alike; ranking unchanged |
| 3 | Can cheap upgrades make k-NN competitive? | `trivial_knn.py` | Yes: both GPT fields + one vote per study (→ `knn_study`) |
| 4 | Can cheap upgrades make nearest-term retrieval competitive? | `trivial_nearest_term.py` | Only with labels: log label prior + training centroids (→ `prototype`) |
| 5 | What text should represent a term? | `term_text_variants.py`, `term_keywords.py` | Trained methods don't care; plain label is best zero-shot; LLM-imagined samples fail |
| 6 | Does cleaning Metalog help? | `cleaning_effect.py`, `selective_prediction.py` | Not the model; it changes the measurement. Confidence triage works |
| 7 | Does hierarchical back-off (+ reranking) help? | `hierarchical_backoff.py`, `draft_biome_label_map.py` | Back-off: +20–30 % coverage at equal accuracy on biome; the biome label map is the larger gain |
| 7c | Does a real LLM reranker help? | `rerank_pilot.py` | Biome and feature, a little (+2 points overall top-1); material no |
| 8 | Does the atlas back-off hold outside Metalog? | `gold_check_extract.sh`, `gold_check.py` | At the coarse level, yes for animal / soil / water; abstains more out of domain |
| 9 | Do sister studies leak across folds? | `project_groups.py`, `5_evaluate.py --fold_groups` | A little: ~1.3 biome points |
| 10 | Can we flag samples Metalog does not cover? | `6b_coverage.py`, `coverage_check.py` | Yes (8.5 % of the atlas); within Metalog the probability already holds this signal |
| 11–13 | How are plant / food samples labelled; does keeping PO / FOODON help? | `5_evaluate.py` on old vs new labels | Plant feature +10 points, plant material back-off answers 23 → 49 % |
| 12 | Can we flag blanks and mock communities? | `6c_flag_controls.py` | 13,204 atlas samples; 37 / 40 checked hits are real |
| 14 | Is the sample selection stable? | `common.select_samples` / `study_folds` | Now yes; fold choice still moves biome back-off coverage by ±3 points |
| 15 | How many top-1 “errors” are defensible alternative terms, and how should they be scored? | `acceptable_answers.py` (+ `label_pair_kinds_draft.tsv`) | Many: all rules together +23 / +8 / +6 points micro, +40 / +18 / +16 macro on learnable labels; broader answers need partial credit |
| 16 | Why is macro low, and does class balancing help? | `macro_and_balance.py` | 65 % of feature / material labels come from one study (unlearnable under study folds). Balanced weights: biome +10 macro on learnable labels at no accuracy cost; feature 1/√n weights +6, accuracy unchanged; material: a trade |
| 17 | Is one probability threshold enough? Can zero-shot take over when the model abstains? | `confidence_checks.py` | Yes: margin, runner-up, entropy add ≤ 0.004 AUC. Abstention catches 67–86 % of unseen-label feature / material samples; zero-shot alone gets 11–25 % of them (plain label), so it needs a reranker |
| – | Is step 6 the model of step 5? | `verify_atlas_methods.py` | Identical top-1 for every method and slot |

`run_all.sh` reproduces §1–4 (≈ 1.5–3 h). `_setup.py` holds the shared loading code.

## 1. All methods over 5 fold assignments (raw labels; `summary.txt`)

Mean ± sd over `--fold_seed 0..4`; the difference to `linear` is paired (same test rows), in points.

| method | biome | feature | material | vs linear: mean [min, max] |
|---|---|---|---|---|
| majority | 0.274 ± 0.000 | 0.477 ± 0.000 | 0.448 ± 0.000 | −24.2 / −16.0 / −26.4 |
| retrieval (slot vocabulary) | 0.181 ± 0.011 | 0.393 ± 0.007 | 0.251 ± 0.002 | −33.6 / −24.4 / −46.0 |
| retrieval_open (all terms) | 0.050 ± 0.000 | 0.141 ± 0.000 | 0.095 ± 0.000 | −46.6 / −49.7 / −61.7 |
| retrieval_prior | 0.475 ± 0.006 | 0.596 ± 0.003 | 0.677 ± 0.004 | −4.1 / −4.2 / −3.5 |
| knn | 0.492 ± 0.009 | 0.616 ± 0.005 | 0.682 ± 0.007 | −2.4 / −2.2 / −2.9 |
| **knn_study** | 0.524 ± 0.008 | 0.639 ± 0.004 | 0.696 ± 0.003 | +0.8 [−0.3, +1.9] / +0.2 / −1.6 |
| linear | 0.516 ± 0.003 | 0.637 ± 0.003 | 0.711 ± 0.006 | — |
| hybrid | 0.514 ± 0.006 | 0.646 ± 0.003 | 0.714 ± 0.005 | −0.3 / +0.8 [+0.2, +1.3] / +0.3 |
| hybrid_open | 0.510 ± 0.005 | 0.646 ± 0.003 | 0.718 ± 0.006 | −0.7 / +0.9 / +0.7 |
| **prototype** | 0.507 ± 0.004 | 0.636 ± 0.003 | 0.723 ± 0.003 | −1.0 / −0.1 / +1.1 [+0.6, +1.5] |
| prototype_open | 0.505 ± 0.002 | 0.634 ± 0.004 | 0.724 ± 0.003 | −1.2 / −0.3 / +1.2 |
| label_reg | 0.530 ± 0.006 | 0.637 ± 0.003 | 0.726 ± 0.006 | +1.4 [+0.6, +2.7] / −0.1 / +1.5 |

With the cleaned labels and the biome label map (fold seed 0, `cv_backoff`): knn_study 0.682 / 0.630 /
0.706, linear 0.671 / 0.624 / 0.713, prototype 0.653 / 0.627 / 0.726, label_reg 0.670 / 0.623 / 0.730.
The `hybrid` weight and the `prototype_open` bonus were tuned on this same CV (slightly optimistic).

## 2. Label processing (fold seed 0, raw labels)

| method | exact | near-synonym (cos ≥ 0.8) | gold or ancestor | coarser labels (≥ 100 samples, retrained) |
|---|---|---|---|---|
| retrieval | 0.183 / 0.389 / 0.251 | 0.223 / 0.413 / 0.266 | 0.195 / 0.396 / 0.266 | 0.366 / 0.571 / 0.404 |
| knn_study | 0.511 / 0.634 / 0.693 | 0.535 / 0.652 / 0.704 | 0.614 / 0.656 / 0.727 | 0.558 / 0.666 / 0.736 |
| linear | 0.514 / 0.634 / 0.708 | 0.532 / 0.646 / 0.730 | 0.622 / 0.652 / 0.748 | 0.564 / 0.653 / 0.742 |
| prototype | 0.509 / 0.639 / 0.723 | 0.525 / 0.657 / 0.734 | 0.609 / 0.650 / 0.752 | 0.560 / 0.661 / 0.747 |
| label_reg | 0.525 / 0.636 / 0.727 | 0.555 / 0.653 / 0.741 | 0.629 / 0.647 / 0.762 | 0.569 / 0.663 / 0.751 |

Coarser labels: biome 95 → 43, feature 202 → 85, material 179 → 50 labels. Every trained method gains
similarly, so the ranking does not change. `analyses.py` also measures the **label ceiling**: for samples
whose nearest neighbour *from another study* is a near-duplicate (cosine ≥ 0.9), the two share the label
55 / 81 / 89 % of the time (raw labels), and the model is right about as often on them.

## 3. k-NN ladder (`trivial_knn.py`, fold seed 0, nested CV)

Each row adds one idea; hyper-parameters are tuned by 3 inner study-grouped folds.

| row | biome | feature | material |
|---|---|---|---|
| A kNN-25 majority, keywords only | 0.462 | 0.590 | 0.648 |
| B + keywords & sub-biome | 0.477 | 0.609 | 0.672 |
| C + tuned k and similarity weights | 0.485 | 0.614 | 0.676 |
| D + one vote per study | 0.481 | 0.640 | 0.701 |
| E + class-prior correction | 0.481 | 0.638 | 0.701 |
| F/G/H/I centring / whitening / hubness (CSLS) / Wilson editing | 0.466–0.501 | 0.614–0.634 | 0.665–0.709 |
| J kNN-50, one vote per study, in an LDA space | 0.520 | 0.639 | 0.692 |
| ridge (reference) | 0.514 | 0.634 | 0.708 |

B (use both GPT fields) and D (one vote per study) matter; the standard retrieval tricks move ±1–2 points
without a direction. Nested tuning is noisy on biome (it picks k from 5 to 100), so `knn_study` uses fixed
k = 50, unweighted votes (0.511 / 0.634 / 0.693 on the same folds).

## 4. Nearest-term ladder (`trivial_nearest_term.py`, fold seed 0, nested CV)

| row | uses labels? | biome | feature | material |
|---|---|---|---|---|
| N-A keywords → nearest slot term | no | 0.259 | 0.453 | 0.106 |
| N-B sub-biome as the query | no | 0.101 | 0.151 | **0.317** |
| N-C keywords + sub-biome | no | 0.183 | 0.389 | 0.251 |
| N-D + centring ("modality gap") | no | 0.152 | 0.193 | 0.296 |
| N-F + log label frequency (→ `retrieval_prior`) | counts | 0.484 | 0.610 | 0.682 |
| N-G + blend with training centroids (→ `prototype`) | yes | 0.509 | 0.640 | 0.718 |
| N-P nearest centroid only | yes | 0.435 | 0.584 | 0.634 |
| N-H `label_reg` | yes | 0.525 | 0.636 | 0.727 |
| N-O1 open vocabulary (all terms), keywords | no | 0.049 | 0.161 | 0.063 |

Without labels nothing helps except a short query for material. The label counts are the big step
(+23 / +16 / +58 points). **Unseen-term bonus** for `prototype_open` (overall / unseen-label top-1): 0 →
0.506 / 0.000, 0.638 / 0.000, 0.723 / 0.000; 0.5 → 0.462 / 0.057, 0.628 / 0.055, **0.733 / 0.114**. Only
material gains; tuned on the same CV.

## 5. Term text (`term_text_variants.py`, `term_keywords.py`; fold seeds 0–4)

| method | `label` | `label_syn` (current) | `label_syn_def` | `label_syn_parents` | LLM-imagined samples (`llm_kw_sb`) |
|---|---|---|---|---|---|
| retrieval, slot vocabulary | 0.204 / 0.419 / 0.442 | 0.181 / 0.393 / 0.251 | 0.176 / 0.264 / 0.189 | 0.138 / 0.493 / 0.172 | 0.318 / 0.246 / 0.455 |
| retrieval_open, unseen labels | **0.143 / 0.122 / 0.213** | 0.134 / 0.101 / 0.186 | – | – | 0.017 / 0.106 / 0.127 |
| prototype | 0.503 / 0.640 / 0.720 | 0.507 / 0.636 / 0.723 | 0.504 / 0.638 / 0.724 | 0.502 / 0.637 / 0.720 | 0.502 / 0.631 / 0.723 |
| label_reg | 0.531 / 0.636 / 0.723 | 0.530 / 0.637 / 0.726 | 0.538 / 0.638 / 0.719 | 0.541 / 0.638 / 0.718 | 0.519 / 0.634 / 0.725 |

Trained methods move ±1 point. Zero-shot swings mostly on one decision (*fecal material* vs *intestine
environment*, 45–48 % of feature / material labels): the synonyms of *fecal material* ("droppings; frass;
pellet") pull it away from human-gut keywords. The plain label is the best zero-shot text; LLM-written sample
descriptions of every term ($17.50) make thousands of rare terms compete with real samples and fail.

## 6. Cleaning Metalog (`cleaning_effect.py`, `selective_prediction.py`; linear, fold seeds 0–2)

- **Training on cleaner data changes nothing** (all arms within ±1 point, CIs include 0); perturbed samples
  carry rare real labels and stay.
- **Evaluation does change**: by group of test sample, gold 0.521 / 0.642 / 0.734, perturbed + degraded
  0.361 / 0.312 / 0.430, unreviewed audit hits 0.254 / 0.190 / 0.395. Report gold-set numbers.
- **A detector** (logistic regression on the vectors) separates artificial samples with AUC 0.82; it scores
  the never-seen audit hits like Metalog-flagged samples (0.42 vs 0.38, gold 0.08).
- **Confidence triage**: keeping the 50 % most confident predictions gives 0.75 / 0.94 / 0.97 accuracy.

## 7. Hierarchical back-off and simulated reranking (`hierarchical_backoff.py`; prototype, 3 fold seeds)

| slot | top-1 | 80 % target: back-off / flat coverage | 90 %: back-off / flat | 95 %: back-off / flat |
|---|---|---|---|---|
| biome (raw labels) | 0.508 | **0.72** / 0.52 | 0.45 / 0.34 | not reached / 0.27 |
| biome (label map) | 0.650 | – | **0.76** / 0.46 | **0.66** / – |
| feature | 0.635 | 0.78 / 0.77 | 0.60 / 0.59 | 0.49 / 0.50 |
| material | 0.722 | 0.94 / 0.93 | **0.79** / 0.72 | **0.68** / 0.56 |

"Flat" = answer the top-1 or abstain. Back-off pays where labels nest (biome, material). A simulated
reranker (accuracy 0.85 inside the top-5, AUROC 0.85, on the 30 % least confident) adds +9 / +4 / +2
top-1 points; reranking everything *lowers* feature and material. The **biome label map**
(`draft_biome_label_map.py`: 54 non-biome terms used as biomes → an ENVO biome by cross-study kNN votes, a
biome-only prototype and label similarity) lifts top-1 on samples it did *not* relabel by +4.9 points (all 3
seeds +4.5 to +5.2): part of the overall gain is by construction, this part is not.

## 7c. LLM reranker pilot (`rerank_pilot.py`; 500 gated samples per slot)

| slot | gold in candidates | base right when gold offered | gpt-4.1-mini v2 | gpt-5.1 v2 |
|---|---|---|---|---|
| biome | 81 % | 0.41 | 0.44 | 0.44 |
| feature | 48 % | 0.46 | 0.60 | 0.62 |
| material | 51 % | **0.69** | 0.52 | 0.46 |

LLM confidence is badly over-confident (AUROC 0.55–0.61); fusing it requires recalibration (T2 = 1.7–15).
With recalibrated fusion and back-off, at ~80 % answered: biome strict accuracy 0.743 → 0.789, feature
0.281 → 0.390, material no gain. A manual review of 47 "more specific than Metalog" answers found 68 % stated
in the metadata and 6 % wrong, so the strict metric understates the LLM. Top-5 + broader terms is the
candidate set used by step 7.

## 8. Gold-set check (`gold_check.py`; 1,021 hand-labelled samples outside Metalog, coarse labels)

| back-off answers | biome | feature | material |
|---|---|---|---|
| answered, Metalog CV (2026-10-01) | 0.79 | 0.61 | 0.78 |
| answered, gold samples (2026-10-01 atlas) | 0.64 | 0.18 | 0.43 |
| coarse-consistent among answered (2026-10-01) | 0.925 | 0.875 | 0.82 |
| … after a rule review of the 151 inconsistent answers (arguable = ½) | 0.97 | 0.95 | 0.92 |
| answered / coarse-consistent, current atlas (2026-10-05) | 0.58 / 0.93 | 0.16 / 0.89 | 0.50 / 0.74 |

Each predicted term is mapped to compatible coarse biomes (animal / plant / soil / water / other) from the GPT
biomes of the Metalog samples carrying it. The metric is noisy (Metalog's own labels are only 50–83 %
consistent on the 24 linked gold samples). The 2026-10-05 material drop is rhizosphere → *soil* (Metalog's
convention, counted as "not plant"). Out of domain the model abstains more rather than erring more; weak spots:
plant anatomy, food, laboratory, air.

## 9. Project-level folds (`project_groups.py`)

Study codes sharing an SRA study / BioProject accession are merged (Stewart 2018/2019, Alneberg 2018/2020), plus
all four TARA codes (same stations): 552 → 547 groups, 378 evaluated samples affected. Their biome accuracy falls
from 0.99 to 0.64 once the sister study cannot be in training: about −1.3 points of overall biome top-1 for the
supervised methods; `prototype` is barely affected.

## 10. Coverage flag (`6b_coverage.py`, `coverage_check.py`)

| slot | flagged (Metalog OOF) | top-1 unflagged / flagged | back-off answers unflagged / flagged |
|---|---|---|---|
| biome | 6.9 % | 0.66 / 0.51 | 81 % / 43 % |
| feature | 5.0 % | 0.67 / 0.09 | 64 % / 5 % |
| material | 4.7 % | 0.75 / 0.15 | 82 % / 12 % |

Adding coverage to the calibrated probability does not raise the AUC for top-1 correctness (0.83 / 0.92 / 0.88).
Atlas: 8.5 % flagged (plant 25 %, other 29 %, soil / water 3 %); flagged samples are ≈ 5 / 0.5 / 1.4 % of the
atlas's back-off answers. Gold samples: 19 % flagged.

## 11–13. Plant and food labels (PO / FOODON kept, rhizosphere / rhizoplane material → soil)

Same 17,640 samples and folds, old → new labels: feature on GPT-plant samples prototype 0.19 → 0.29, linear
0.11 → 0.22; material on GPT-plant samples 0.54 → 0.64 (prototype); plant material back-off answers 23 → 49 %
with strict accuracy 0.64 → 0.93. Biome unchanged; food samples barely move (FOODON labels from 5 small studies).

## 12. Controls (`6c_flag_controls.py`)

13,204 atlas samples (0.4 %): 7,458 by sub-biome, 4,129 by a strong keyword, 1,617 by a weak keyword in a lab
context. On the linked samples it finds 29 of Metalog's 58 controls; its 27 other hits are defensible (audit hits,
probiotic positive controls). 37 of 40 random atlas hits are clear controls or mocks.

## 14. Stable selection and folds

Since 2026-10-05 a sample's selection and a study's fold depend only on their own id, so adding labels keeps
every previously selected sample (all 17,640 kept, +80) and every study's fold. Before, a seeded shuffle
redrew 22 % of the samples and moved biome back-off coverage by 6 points with no label change. The fold split
itself still moves biome coverage at 90 % between 0.73 and 0.79 (fold seeds 1 / 0 / 2), so the atlas pools the
calibration of seeds 0–2 (τ = 0.825, coverage 0.76).

## 15. Acceptable answers: synonyms, hierarchy, facets, curator co-labels (`acceptable_answers.py`)

Prototype top-1 of the production CV run (cleaned labels + biome map, fold seed 0). Inputs beyond the pipeline's:
ENVO's OBO file (for the non-is_a relations; release 2026-06-26, the one the term table was built from) and
`label_pair_kinds_draft.tsv`, a one-pass classification of the 122 disagreeing label pairs (Claude, **not reviewed**).

```bash
# once: ENVO OBO of the term table's release (the GitHub copy is the same file, data-version releases/2026-06-26)
curl -L -o ~/MicrobeAtlasProject/ontologies/envo.obo http://purl.obolibrary.org/obo/envo/releases/2026-06-26/envo.obo
#   (or https://raw.githubusercontent.com/EnvironmentOntology/envo/master/envo.obo while its header says data-version: releases/2026-06-26;
#    the file used here has md5 b568c065f8bb96af5075ed57cf981875)
python experiments/acceptable_answers.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --envo_obo ~/MicrobeAtlasProject/ontologies/envo.obo \
  --pair_kinds experiments/label_pair_kinds_draft.tsv \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/acceptable_answers   # ~1-2 min
```

**Curator disagreement.** 39,975 pairs of samples from different studies have near-identical vectors (cosine ≥ 0.9);
they share the label 76 / 85 / 94 % of the time. Disagreeing label pairs 23 / 70 / 29, by kind (share of the
disagreeing sample pairs; `composition.tsv`):

| slot | equivalent | broader / narrower | two facets | different facts |
|---|---|---|---|---|
| biome | 0 % | 93 % | 0.1 % | 7 % |
| feature | 8 % | 9 % | 61 % | 22 % |
| material | 18 % | 36 % | 44 % | 2 % |

**Acceptance rules** (`rule_scores.tsv`), each alone vs exact, in points: micro / macro / macro over learnable labels
(used by ≥ 2 studies). Combined rows are per-sample ORs.

| rule | biome | feature | material |
|---|---|---|---|
| exact (level) | 0.653 / 0.262 / 0.439 | 0.627 / 0.079 / 0.228 | 0.726 / 0.074 / 0.214 |
| broader (any depth) | +10.4 / +45.5 / +32.4 | +1.3 / +3.7 / +2.3 | +3.2 / +6.6 / +11.5 |
| narrower (any depth) | +11.4 / +4.3 / +7.2 | +1.5 / +2.5 / +3.1 | +1.8 / +2.8 / +3.6 |
| automatic synonyms | +0.6 / +4.8 / +8.0 | +0.7 / +1.2 / +2.2 | +0.4 / +1.1 / +0.2 |
| curator co-labels (out of fold, ≥ 2 study pairs) | +7.4 / +5.2 / +8.8 | +2.1 / +1.4 / +4.0 | +1.9 / +1.5 / +4.3 |
| synonyms + co-labels | +8.0 / +10.0 / +16.8 | +2.7 / +2.6 / +6.1 | +2.3 / +2.6 / +4.4 |
| facets (direct ENVO relation) | 0 / 0 / 0 | +3.7 / +4.6 / +9.3 | +0.6 / +2.2 / +0.2 |
| all | +22.9 / +50.3 / +40.4 | +8.0 / +12.0 / +18.1 | +5.7 / +11.8 / +15.5 |

Synonyms and co-labels rescue disjoint samples (0 / 0 / 3 shared), so they add up; the sum of all single rules
overstates "all" (micro +29.8 / +9.2 / +8.0 vs +22.9 / +8.0 / +5.7; `overlap.tsv`).

**Graded credit for broader answers** (`graded_credit.tsv`), micro / macro:

| scheme | biome | feature | material |
|---|---|---|---|
| exact | 0.653 / 0.262 | 0.627 / 0.079 | 0.726 / 0.074 |
| depth (0.5 per step) | 0.696 / 0.423 | 0.633 / 0.096 | 0.741 / 0.105 |
| IC(answer) / IC(gold) | 0.704 / 0.445 | 0.637 / 0.104 | 0.743 / 0.105 |
| full credit | 0.756 / 0.717 | 0.639 / 0.116 | 0.758 / 0.140 |
| hierarchical F (ancestor sets) | 0.910 / 0.842 | 0.734 / 0.321 | 0.842 / 0.387 |

Recommended: report exact Metalog, the IC-weighted acceptable score, and the share of broader answers, always naming
the rules. Hierarchical F is too generous (generic upper classes are shared by every pair).

Limits: automatic synonyms are candidates (siblings and narrower terms pass the filter); facets use every ENVO relation
type in both directions, direct links only, and no Uberon / PO relations; co-labels mean "the text cannot tell them
apart", not "equivalent"; one fold seed.

## 16. Macro by label support and class balancing (`macro_and_balance.py`)

```bash
python experiments/macro_and_balance.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --fold_seeds 0 1 2 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/macro_and_balance   # ~4 min on the Mac; resumable
```

**Why macro is low** (`support_summary.tsv`, `support_buckets.tsv`; prototype, fold seed 0). A label used by one study
is never in training when that study is tested, so it scores 0 by construction:

| slot | labels | from 1 study | their samples | macro all | macro ≥ 2 / ≥ 5 / ≥ 10 studies | drop 10 % rarest |
|---|---|---|---|---|---|---|
| biome | 42 | 40 % | 3 % | 0.262 | 0.439 / 0.583 / 0.616 | 0.297 |
| feature | 208 | 65 % | 16 % | 0.079 | 0.228 / 0.547 / 0.731 | 0.088 |
| material | 185 | 65 % | 12 % | 0.074 | 0.214 / 0.549 / 0.852 | 0.083 |

Feature accuracy per label by number of studies: 1: 0.00 (136 labels), 2: 0.07, 3–4: 0.11, 5–9: 0.46, 10–19: 0.69,
20+: 0.99. Report macro over learnable labels (≥ 2 studies) next to the share of samples they cover.

**Class balancing** (`balance_summary.tsv`; linear unless stated, mean over fold seeds 0–2, micro / macro over learnable
labels):

| variant | biome | feature | material |
|---|---|---|---|
| plain (pipeline `linear`) | 0.663 / 0.424 | 0.630 / 0.264 | 0.715 / 0.210 |
| balanced (n / K n_c) | 0.666 / **0.523** | 0.591 / 0.348 | 0.639 / 0.253 |
| tempered, weights ∝ n_c^-0.5 | 0.664 / 0.472 | **0.631 / 0.320** | 0.693 / 0.227 |
| tempered, n_c^-0.25 | 0.662 / 0.437 | 0.632 / 0.293 | 0.705 / 0.221 |
| balanced, dominant class weight 1 | 0.664 / 0.525 | 0.618 / 0.341 | 0.644 / 0.253 |
| plain answer when it is the dominant class | 0.667 / 0.523 | 0.621 / 0.340 | 0.650 / 0.243 |
| logit adjustment 0.05 | 0.663 / 0.434 | 0.632 / 0.286 | 0.711 / 0.225 |
| prototype β 0.1 (production) | 0.647 / 0.428 | 0.633 / 0.243 | 0.725 / 0.204 |
| prototype β 0.05 | 0.643 / 0.460 | 0.622 / 0.303 | 0.681 / 0.266 |
| prototype β 0 (no prior) | 0.603 / 0.585 | 0.486 / 0.367 | 0.485 / 0.343 |

- Biome: balancing is free on all three seeds (macro on learnable labels +10 points, worst seed 0.494 vs best plain 0.435).
- Feature: 1/√n weights keep accuracy and add +6 points of macro on learnable labels. Full balancing loses 1,163
  answers vs 458 gained (fold seed 0), 60 % of the losses on *intestine environment*; a dominant-class rule recovers
  only part of it (`balance_lost_gained.tsv`).
- Material: every variant trades accuracy for macro; losses spread over *soil*, *fecal material*, *fresh water* ….
- Not yet tested: the effect on calibration and back-off coverage, and a balanced variant of the prototype itself.

## 17. Confidence checks: one threshold, zero-shot fallback (`confidence_checks.py`)

```bash
python experiments/confidence_checks.py \
  --samples ~/MicrobeAtlasProject/metalog/clean/training_set.clean.tsv.gz \
  --predictions ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/predictions.tsv.gz \
  --calibration ~/MicrobeAtlasProject/ontology_mapping/cv_backoff/calibration.json \
  --plain_label_vectors ~/MicrobeAtlasProject/ontology_mapping/experiments/term_text/term_variants.h5 \
  --output_dir ~/MicrobeAtlasProject/ontology_mapping/experiments/confidence_checks   # ~1 min
```

**One probability threshold is enough** (`reliability.tsv`, `shape_auc.tsv`, `shape_band.tsv`; prototype, fold seed 0).
Probabilities are roughly calibrated, a little overconfident mid-range (feature top-1 with p in (0.3, 0.4]: 20 %
right). Study-grouped CV AUC for "the top-1 is right":

| features | biome | feature | material |
|---|---|---|---|
| p1 | 0.808 | 0.920 | 0.884 |
| p1 + runner-up p2 | 0.807 | 0.919 | 0.885 |
| p1 + entropy | 0.805 | 0.920 | 0.888 |
| p1 + mass outside the top 5 | 0.806 | 0.920 | 0.885 |

(the score margin added to the CV probability: −0.001 / −0.001 / 0.000). At p1 ≈ 0.30, material top-1s are right 35 %
of the time when the alternatives sit in the top 5 and 7 % when they are spread out; biome and feature show no pattern.
The shape matters for what to answer instead (back-off, a candidate list), not for trusting the top-1.

**Zero-shot fallback** (`zero_shot_fallback.tsv`): abstention as a detector of labels never seen in training, and
zero-shot retrieval on those samples.

| | biome | feature | material |
|---|---|---|---|
| samples whose gold is unseen in training | 3.3 % | 16.8 % | 13.6 % |
| of those, abstained on | 11 % | 86 % | 67 % |
| of the abstained, gold unseen | 2 % | 36 % | 47 % |
| zero-shot on unseen gold, label; synonyms, all 49k terms: exact / in top 5 | 0.02 / 0.19 | 0.09 / 0.27 | 0.18 / 0.32 |
| zero-shot on unseen gold, plain label, 18.9k ENVO / Uberon terms: exact / in top 5 | 0.02 / 0.06 | 0.11 / 0.31 | 0.25 / 0.39 |

Abstention detects the impossible cases for feature and material; zero-shot alone is too weak to answer them, but its
top 5 (plain label) holds the curator's term for a third of them: a candidate list for a reranker.

Reproducibility note: the AUC cross-validation uses `common.study_folds`, not sklearn's `GroupKFold`, whose
tie-breaking between equal-size studies gave different folds (and AUCs differing in the 3rd decimal) on the Mac and in
the cloud.

## Verification (2026-10-05 review)

Every pipeline step was re-run with the reviewed code and compared with the production outputs:

| check | result |
|---|---|
| step 1 term table, step 2b flags / clean / gold sets / review / summary, step 3 `.npz`, step 6c controls | byte-identical |
| step 2 training set (58,365 rows) | byte-identical |
| step 5, fold seed 0, all methods: reviewed vs original code (same machine) | metrics and predictions byte-identical |
| step 5, original code: this machine vs the production run | top-1 identical; 3 tie-dependent knn_study metrics differ in the 4th decimal; confidences within 2e-15 |
| step 6 atlas (prototype + back-off, 3.44M samples) | byte-identical content |
| step 6b on 50,000 random atlas samples + the held-out calibration | identical |
| step 7 `fit`; `rerank_pilot.py score` / `vote` | identical to the stored `rerank_params.json` / original code |
| `verify_atlas_methods.py` (step 6 streaming == step 5 functions, 50,100-sample mini atlas) | OK for linear / prototype / knn_study on all slots: identical top-1 and back-off, differences ≤ 1e-4 (4-decimal rounding of the outputs) |
| experiments, reviewed vs original code: `gold_check`, `coverage_check`, `coarsen_labels`, `project_groups`, `hierarchical_backoff` | identical (`term_to_coarse_biome.tsv`: same rows, tie order differed; the sort is now deterministic) |
| `analyses.py` | identical except `granularity` for feature / material at min_support 300 (and material at 100): the bug fix keeps coarsening inside ENVO / Uberon, e.g. material @300: 42 labels, top-1 0.779 (was 31 labels, 0.782) |
| `draft_biome_label_map.py` | not comparable: needs plain-label term vectors (`term_text/term_variants.h5`), which were not re-staged; the map itself is a reviewed, versioned input |
