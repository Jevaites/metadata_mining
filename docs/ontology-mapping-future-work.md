# Ontology mapping: future work

Priorities are ordered by expected scientific value, not implementation novelty.

## 1. Build a defensible benchmark

- Curate an external, project-held-out set with exact ENVO/Uberon/PO/FOODON terms for all three
  slots. The current 1,021-sample external set has only coarse biome classes, so it cannot establish
  exact-term accuracy.
- Double-annotate a stratified sample and report inter-annotator agreement. Oversample plant,
  food, laboratory, air, urban, engineered and rare-animal habitats.
- Treat Metalog as a high-quality **silver** source. Preserve raw labels, curator/source provenance,
  ontology release, remaps and adjudications.
- Freeze a versioned benchmark manifest: sample IDs, project groups, ontology versions, label-map
  version and hashes of every input.

## 2. Improve coverage before model complexity

- Add curated examples for the habitats the coverage detector identifies as weak: plant tissues,
  food/fermentation, laboratory/controls, air, urban/engineered environments and non-mammal hosts.
- Review the 32 pending Metalog audit groups and the draft biome-label map with a domain curator.
- Keep PO and FOODON; investigate further anatomy/material ontologies only when the benchmark shows
  a concrete gap. Do not expand the candidate space without measuring open-vocabulary precision.
- Separate “no habitat” samples (blank, mock, technical control) from genuine environments before
  ontology mapping.

## 3. Make the prediction target explicit

- Decide whether success means exact curator agreement, a biologically compatible term, or a
  calibrated ancestor. Report all three, but choose one primary endpoint.
- Define slot constraints and invalid combinations (for example, a material term used as a biome).
  Validate outputs with ontology domains/ranges or a small hand-reviewed rule set.
- Replace the draft rhizosphere and obsolete-human mappings with reviewed, versioned rules.
- Publish both the predicted term and its evidence: top candidates, calibrated probability,
  coverage flag, selected ancestor, model/version and whether the sample was suppressed as control.

## 4. Strengthen evaluation

- Use project-level folds by default; run at least five fold seeds and report study-bootstrap
  confidence intervals.
- Make macro accuracy, rare-label recall and per-habitat coverage co-primary metrics. Micro accuracy
  is dominated by fecal/intestine labels.
- Tune prototype weights, priors, thresholds and reranker gates in nested folds or a development set.
  Evaluate once on the frozen external benchmark.
- Add temporal validation: train on an older Metalog snapshot, test on newly curated studies.
- Measure calibration with reliability diagrams, ECE/Brier score and risk-coverage curves, including
  separate curves for in-coverage and out-of-coverage samples.

## 5. Improve open-vocabulary mapping

- Train a dual encoder or metric-learning model on sample–term pairs with hard ontology negatives.
  The current zero-shot/open methods recover very few unseen labels.
- Generate term-side descriptions in the same style as sample keywords, but validate generated text
  for factuality and leakage before embedding it.
- Retrieve a constrained candidate set first (slot, ontology branch and lexical matches), then use a
  cross-encoder or small LLM to rerank. Include ancestors and siblings as hard alternatives.
- Test character/lexical retrieval alongside embeddings for exact ontology names, abbreviations and
  codes; fuse scores only after nested validation.

## 6. Use the ontology more fully

- Parse pinned OWL releases with a standards-compliant library and retain replacement terms,
  equivalent classes, alternative IDs and relevant relations—not only named `is_a` parents.
- Add automated ontology-release diffs: new, obsolete, merged and reparented terms; block a run when
  a label map points to an invalid term.
- Investigate hierarchy-aware losses and evaluation distances, while keeping the current transparent
  ancestor back-off as the baseline.

## 7. Scale and monitor safely

- Turn the new run manifests into a single pipeline manifest and record package versions, Git commit
  and random seeds. Store summaries, not millions of intermediate part files, after verification.
- Add CI for synthetic unit tests plus a small end-to-end fixture covering every step and resume.
- Monitor Atlas drift by project/date, label distribution, abstention, coverage and control rate.
- Route low-confidence, out-of-coverage and high-impact rare classes to active-learning review;
  retrain only after adjudication.

## Recommended next experiment

Curate 1,000 exact-term samples from entirely unseen projects, stratified across the weak habitat
groups above. Compare: (1) prototype + prior, (2) constrained lexical/embedding retrieval, and
(3) retrieval + cross-encoder reranking. Use project-grouped nested tuning, then one frozen test.
This resolves the largest current uncertainty: whether the system predicts ontology terms beyond
Metalog's dominant conventions, rather than merely transferring those conventions reliably.
