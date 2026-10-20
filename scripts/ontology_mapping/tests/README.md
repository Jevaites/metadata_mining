# Tests of the ontology-mapping pipeline (`scripts/ontology_mapping`)

One section per script, in pipeline order. Each script has up to three test files:

- `test_<script>_unit.py`: one class or block per function, inputs small enough to check by eye.
- `test_<script>_integration.py`: `main()` (the CLI) end to end on a toy world written in the real on-disk
  formats (column order, compression, `.npz` / `.h5` layout), plus edge cases and failure paths.
- `test_<script>_mutations.py`: plausible one-line bugs ("mutants") with the tests that must catch them. This
  shows the tests check behaviour rather than restate the code, and makes regressions of fixed bugs visible.

Nothing calls the OpenAI API or the network: fakes replace `openai.OpenAI`, `urlopen` and the Batch API.
Tests never read `~/MicrobeAtlasProject`; where the real data has a peculiar format, a few real lines are
copied (shortened) into the test as literals.

## Running

```bash
pip install pytest h5py          # the rest (numpy, pandas, scikit-learn, scipy, openai) is in metadmin_env.yml
cd scripts/ontology_mapping
python -m pytest tests -k "not mutation"          # ~1,070 tests, ~15 s
python -m pytest tests/test_methods_mutations.py  # one mutation file at a time: 10-55 s each, ~5 min in all
python -m pytest tests ../embeddings/tests        # both test folders in one run also works
```

Checked with Python 3.10, numpy 2.2, pandas 2.3, scikit-learn 1.7, scipy 1.15. `metadmin_env.yml` pins
scikit-learn 1.1.1 with scipy 1.15.2: RidgeClassifier fails with that pair (scipy removed `sym_pos`), so
use scipy < 1.11 or a newer scikit-learn.

## Shared helpers

| file | what |
|---|---|
| `conftest.py` | only puts `scripts/ontology_mapping` and `scripts/embeddings` on `sys.path`. Helpers live in importable modules, never here: `from conftest import ...` is ambiguous when both test folders run together. |
| `step_loader.py` | `load_step("1_build_term_index.py", alias)` imports a numbered script; `fake_web` fixture replaces `urlopen`. |
| `mutation_harness.py` | shared harness of the subprocess-style mutation files (1, 2, 3, 6, 6c, common, hierarchy, methods): `Mutation(name, edits, why, caught_by)`, `mutate`, `build_copy` (temporary copy of the script + its tests), `run_pytest` (fresh Python, outcomes collected by a plugin), `check_unmutated` (the copy passes and every named test exists), `check_caught` (every named test fails; with a dict `caught_by`, on an AssertionError containing the given text). |
| `step2_helpers.py`, `step2b_helpers.py`, `step4_helpers.py`, `step5_toy.py`, `step6_toy.py`, `step7_toy.py`, `toy_hierarchy.py` | toy worlds, fixtures and file writers of one step (see its section). |

Two mutation styles coexist. Subprocess files (above) write each mutant into a copy of the script and rerun the
named regular tests against it. The in-process files (4, 5, 6b, 7) load each mutant as a separate module and
assert the specific wrong result it gives next to the right one.

## Outputs to regenerate after these fixes

- `atlas_controls.tsv.gz` (6c, written 2026-10-05): rerun 6c (~1 min): +31 snake_case controls, -28 horse
  samples ('buffer control group'), plus the 2026-10-19 fixes.
- `rerank_atlas/` (7, written 2026-10-01): older than `cv_backoff/calibration.json` and `atlas_backoff/`
  (2026-10-05); `apply` does not check this, so rerun fit / build / ... before using `atlas_final.tsv.gz`.
- `atlas_backoff/` and `atlas_coverage/` are valid, but their saved settings predate the md5 / `chunk_rows`
  checks, so resuming into them is refused: use a new `--output_dir` for a rerun.

---

## common.py

Shared helpers: term table and term texts (`load_terms`, `term_text`, `load_term_vectors`), ancestors and
coarsening, sample.info parsing (`iter_sample_info`, `record_to_text`, `DROP_KEYS`), sample selection and folds
(`load_npz`, `stable_key`, `select_samples`, `study_folds`). Function-level tests of these live here; step
tests only test their own use of them.

| file | covers |
|---|---|
| `test_common_unit.py` (98) | one class per function; includes shortened real lines (SRS7471636 record, `\r` inside values, FOODON multi-line synonyms, `NA` / `nan` synonyms, the real sample.info layout) and the `select_samples` defaults (50 per study, seed 22) used by experiment scripts |
| `test_common_integration.py` (11) | the helpers chained as steps 2 and 5/6 chain them: record -> text, terms -> term vectors, selection -> nearest term, study folds, label coarsening, adding a study keeps the earlier selection/folds |
| `test_common_mutations.py` (29 + 1) | 29 bugs written into a copy of common.py (~40 s) |

Toy world: 6 live ENVO terms + 1 obsolete under a BFO root with 2-D "compass" embeddings; 8 samples in 4 studies
(one unlabelled, one without embedding); a 3-record sample.info in the real layout (empty first line, empty line
after each record).

Bugs fixed:
- `iter_sample_info` opened plain (non-.gz) files with the locale encoding and without `errors="replace"`, so one
  non-UTF-8 byte stopped the pass. Both paths now use utf-8 with replace (the pipeline reads `.gz`: no change).
- `iter_sample_info` read with universal newlines: a lone `\r` inside a value (20 lines of sample.info.gz,
  16 samples, e.g. SRS431655 "...in Lake\r Geneva") split the line and the text after it was lost. It now reads
  with `newline="\n"` and turns `\r` into a space. None of the 16 samples is in the training set or the rerank
  requests, so existing results are unchanged.

Quirks pinned (fixing them would change training texts or `.h5` keys):
- `term_text` leaves a trailing `"; "` when a term has no synonyms, and keeps newlines and edge spaces inside
  synonyms (2 FOODON texts span lines). The term `.h5` is keyed by this exact text.
- `DROP_KEYS` matches substrings: `runoff_*` ("run"), `maritime_*`, `sludge_retent_time`, `reaction_time`
  ("time"), `subject is affected` are dropped.
- A header that is just `>` (empty id) silently loses its record.
- `study_folds` folds are not size-balanced; with more folds than studies some test folds are empty.

## 1_build_term_index.py

Parses OBO / OWL ontologies into `ontology_terms.tsv.gz` (ontology, term_id, label, synonyms, definition,
parents, obsolete; ids `ENVO_...`, lists joined with `||`), read by every later step via `common.load_terms`.

| file | covers |
|---|---|
| `test_1_build_term_index_unit.py` (66) | read_bytes, read_obo, parse_obo, is_english, parse_owl; fixtures are real stanzas, shortened (ENVO 'saline lake', PO synonyms with scope suffixes and Spanish / OMO:0003004 types, FOODON 'cattle bull' with equivalentClass, restriction and foreign parents) |
| `test_1_build_term_index_integration.py` (17) | the CLI on a toy ENVO (OBO), FOODON (OWL.gz) and PO (OBO) drawn as trees in comments; `--append` (add an ontology, replace one with a new release, keep 'NA' synonyms and multi-line FOODON definitions); URLs (faked); gzip output; hand-off to `common.load_terms`; bad input (missing file, no `PREFIX=`, missing `--append` table, truncated OWL) raises and leaves the existing table untouched |
| `test_1_build_term_index_mutations.py` (32 + 1) | 32 bugs (~30 s) |

Re-parsing the real `envo.obo` / `po.obo` reproduces the 4,317 ENVO and 1,793 PO rows of the real table cell by cell.

Bugs fixed:
1. Any source starting with "http" was treated as a URL (a local `http_mirror/envo.obo` failed); now only `http://` / `https://`.
2. `\"` was unescaped in OBO definitions but not in synonyms; both now use `unescape()`.
3. A run where no source had any term (mistyped prefix) crashed with `KeyError: 'synonyms'`; a header-only table is now written.

Quirks pinned:
- A source whose prefix matches no term only prints "0 terms" (no error).
- ~35 PO translations without a synonym type ("pedicelo (Spanish, broad)") are kept (32 PO terms), and a double
  space before PO's scope suffix leaves a trailing space (64 synonyms). Fixing either changes term texts, so the
  term embeddings would need recomputing.
- FOODON upstream texts are kept as given (2 synonyms that are IRIs, multi-line definitions / synonyms).

## 2_build_training_set.py and 2b_clean_metalog.py

Step 2 links Metalog labels to MicrobeAtlas `sample.info` records (`metalog_training_set.tsv.gz`); 2b flags
Metalog samples (controls, perturbed, degraded, duplicates, label status) and writes `training_set.clean` /
`.gold`, `metalog_flags`, `audit_review.tsv`, `summary.md`.

| file | covers |
|---|---|
| `test_2_build_training_set_unit.py` (33) | `GOLD_VALUE` / `ACCESSION` regexes (real non-label values: `negative control`, PO, FOODON), `load_metalog_labels`, `snapshot_paths` |
| `test_2_build_training_set_integration.py` (12) | `main()` on a 4-sample Metalog / 7-term mock, plus 3 shortened real sample.info records (SRA: SAMN in `experiment_external_id_BioSample`; ENA: SAMEA in `sample_alias`; SRX/SRR/SRP/ERX/PRJNA ids are not taken for sample ids) |
| `test_2_build_training_set_mutations.py` (11 + 1) | 11 bugs (~15 s) |
| `test_2b_clean_metalog_unit.py` (104) | one block per function: audited_fields, snapshot_files, loaders, flag_labels (every status incl. PO/FOODON), duplicates, audit regexes, overrides, review table, backup, clean_training_set |
| `test_2b_clean_metalog_integration.py` (9) | `main()` on 16 toy samples in 4 studies / 4 domains: one run, the review loop (idempotent), `--training_set` + `--label_map`, edge cases, the step-2 text contract |
| `step2_helpers.py`, `step2b_helpers.py` | writers of the real formats (all_long: `sample_alias, metadata_item, curation_tier, value`; core_wide selected by column name) |

2b has no mutation file; 32 hand-made mutants were checked once: 29 caught, 2 equivalent, 1 implausible.

Bugs fixed in step 2:
- Nothing linked -> crash (no header, KeyError in the summary print). The output now has explicit columns.
- Metalog snapshots were mixed: every `*_all_long_*` file was read, and a sample in two snapshots kept the OLDER
  labels while 2b used the latest. `snapshot_paths()` and `--date` now select one snapshot (default the latest,
  as in 2b); an unknown date stops with the list of dates; files without a YYYY-MM-DD date are ignored.

Quirks pinned:
- Only the first code of a Metalog value is used, in both steps (the 2026-09-20 snapshot has no value with two codes).
- Step 2 keeps only ENVO/UBERON and blanks obsolete ids, so `*_raw` in training_set.clean holds the step-2 id
  (human biome `""`, PO/FOODON `""`); 2b's clean id keeps PO/FOODON.
- A record links to its own id first, then to the first Metalog-known accession in record order.
- An accession under two aliases: step 2 keeps the alphabetically first file/alias, 2b the first in domain order
  (the real snapshot has none).
- `dup_text_in_study` is computed before filtering: if the first copy of a text is a dropped control or perturbed
  sample, the text has no gold row (real data: 24 rows, 1 text).
- Audit and override patterns match a normalised text (`_` and `.` become spaces); "Blank Spring" stays a false positive.
- The obsolete human biome ENVO:00009003 has no replacement in `biome_label_map.tsv`, so every human sample has an empty clean biome.

## 3_extract_sample_embeddings.py

Pulls the vectors of the labelled samples out of a GPT `*_unique_embeddings__*.h5` (one row per distinct text)
into an `.npz` (`sample_ids`, `index`, `vectors`: the vector of `sample_ids[i]` is `vectors[index[i]]`).

| file | covers |
|---|---|
| `test_3_extract_sample_embeddings.py` (14) | `read_first_column` (header, gzip, blank lines, CRLF); `main()` for sub-biomes and keywords, hand-off to `common.load_npz`, printed counts for each way a sample is dropped; exact match after cleaning, sorted ids with vectors in h5 row order, non-ASCII texts ("vicuña feces"), `~` paths, the two "nothing matched" stops |
| `test_3_extract_sample_embeddings_mutations.py` (12 + 1) | 12 bugs (~10 s) |

Toy world: 4 samples / 3 texts, 2-d vectors (soil=[1,0], seawater=[0,1], human gut=[-1,0]); keyword tests use
real shortened lines (`{Short-tailed shearwater, ...}`, brace-less `a,b,c`). `clean_text` / `iter_samples`
are tested in `scripts/embeddings/tests`.

Bug fixed: when nothing matched (wrong `--kind`, or `--sample_ids` about other samples) the script crashed with
h5py's "Indexing arrays must have integer dtypes"; it now stops with a message naming the input to check, before
writing any output.

Quirk pinned: a sample listed twice in the texts file keeps its last line (the real GPT files have no duplicate ids).

## 4_embed_terms.py

Embeds every distinct non-obsolete term text (`common.term_text`) into
`ontology_terms_unique_embeddings__<model>__dim<dim>.h5` (`texts`, `embeddings`) with `embed_unique` from
`embed_subbiomes_keywords.py`; a rerun resumes from the file. The script has no function of its own besides
`main()`; the functions it calls are tested in `test_common_unit.py` and `scripts/embeddings/tests`.

| file | covers |
|---|---|
| `test_4_embed_terms_integration.py` (15) | 3 live terms + 1 obsolete through `main()`: dry run (exact cost line, no client, no file), full run (one call, `dimensions`), read-back with `load_term_vectors` + nearest term, free rerun. Edge cases: ontology update (only new texts embedded; stale file refused), same text in two ontologies embedded once, API outage then rerun, kill during a write and during the first write, one file per model/dim, wrong dimension from the API, only obsolete terms, invalid model, `~` paths, multi-line FOODON synonyms stored verbatim |
| `test_4_embed_terms_mutations.py` (8) | 8 bugs (4 in step 4, 4 in `embed_unique` / `h5_align`), in-process; each docstring names the regular tests that fail |
| `step4_helpers.py` | `FakeEmbeddingClient`; `concept_vector` = [water, soil, gut] word counts, so every vector can be checked by eye |

Bug fixed: a term table with only obsolete terms used to exit quietly with no `.h5`, and later steps failed on a
missing file; it now stops with "has no non-obsolete terms: nothing to embed".

Kept on purpose: the trailing "; " of a term without synonyms ('soil; ') is the key of the existing `.h5` files.

## methods.py

The mapping methods used by 5_evaluate (cross-validation) and 6_predict_atlas (atlas): rank, row_blocks,
neighbours / knn / knn_study (TIE_BREAK), prototype_model / prototype_scores / prototype, linear_scores, hybrid,
label_regression.

| file | covers |
|---|---|
| `test_methods_unit.py` (59) | one example per behaviour, mostly 2-D unit vectors given as angles (`at(0, 10, 90)`: nearest = smallest angle difference): margins, row blocks, similarity-weighted vs per-study votes, a study filling every neighbour slot, exact ties (2 and 5 identical vectors, float64 and float32), float-noise ties, centring / normalisation / alpha / bias / unseen_bonus of prototypes, ridge vs LinearSVC, hybrid, label regression onto an unseen term, empty inputs |
| `test_methods_integration.py` (22) | `run_fold()` chains the methods as `5_evaluate.evaluate_fold` does for one slot, on 8 terms every 45 deg (4 used as labels). Edge cases: one study flooding the neighbour list, gold label unseen in training (closed vs open methods), single-label slot, sparse TF-IDF, empty test set, inputs crossing the internal block sizes |
| `test_methods_mutations.py` (35 + 1 + 2) | 35 bugs ("regression" ones put back fixed bugs); `test_run_fold_matches_5_evaluate` checks that `run_fold` gives the same top-5 lists, confidences and back-off scores as the real `evaluate_fold` (~50 s) |

`run_fold` is a minimal copy of `evaluate_fold` because the mutation harness runs these tests on a lone copy of
methods.py; the drift guard above keeps it honest, and `evaluate_fold`'s own wiring is tested in `test_5_evaluate_unit.py`.

Bugs fixed:
- `rank(..., n=1)` returned margin 0 for every row; now keeps >= 2 columns (the pipeline uses n=5: no change).
- `rank` and `knn_study` crashed on plain lists of ids / study codes (now `np.asarray`).
- knn: a neighbour with negative similarity subtracted from the vote, so confidence could exceed 1; it now votes 0.
- knn: exact ties were left in argpartition's arbitrary order; knn and knn_study now share `neighbours()`, ties
  ordered by training index (TIE_BREAK), similarities in float64. knn top-5 lists change only where exact ties or
  negative-similarity neighbours occur.
- linear_scores, prototype_scores and label_regression raised inside sklearn on an empty test set; they now return empty results.

Quirks pinned:
- prototype_model with all training samples of a label identical: alpha < 1 falls back to the term vector,
  alpha = 1 gives a zero prototype (score = bias).
- prototype_model and hybrid raise KeyError when a training label is not in the vocabulary (input contract).
- knn confidence divides by max(total vote, 1e-9): a lone neighbour at ~90 deg gives ~6e-8, not 0.
- Real features are float32, |v| = 1 +/- 6e-4, with many exact duplicates (50,100 samples, 22,618 distinct
  keyword vectors): exact ties are common, so the tie-break matters.

## hierarchy.py

Calibrated probabilities and hierarchical back-off (softmax, fit_temperature, Closure / decode, relation, outcome,
summarise, choose_tau, oof_backoff, decode_candidates, merge_calibrations), used by 5, 6 and 7.

| file | covers |
|---|---|
| `toy_hierarchy.py` | toy biome ontology: 9 labels (`terrestrial` and `lake` are terms but not labels), real slot root ENVO_00000428 with non-ENVO ancestors above it (simplified) |
| `test_hierarchy_unit.py` (69) | one class per function; DAG cases (a gold with two parents; two equally specific parents), the half rule, slot roots, ontology mode, merge_calibrations pooling |
| `test_hierarchy_integration.py` (25) | the 5 / 6 / 7 workflows, chaining only hierarchy functions, on a perfectly calibrated toy model (scores = log-probabilities, gold labels in exact proportion), so T = 1 and every coverage / accuracy / tau is in the hand-computed table of the module docstring. Edge cases: tau and temperature cross-fitted on the other folds, a gold label never seen, an unreachable target, folds with different vocabularies, strict vs lenient taus |
| `test_hierarchy_mutations.py` (21 + 1 + 1) | 21 bugs, each must fail the named tests with an AssertionError containing a given wrong-vs-right value; plus 1 documented equivalent change no test can catch (~20 s) |

Bugs fixed:
1. `Closure(answers="ontology")` required every answer to be ENVO_/UBERON_, so a slot's own PO_/FOODON_ labels
   were never answers; now every label plus ENVO/UBERON ancestors below `floor`. Only affects
   `experiments/hierarchical_backoff.py --answers ontology` (results before 2026-10-19 excluded PO/FOODON answers).
2. `decode_candidates` (7): between two broader terms of equal depth (DAG) the answer depended on PYTHONHASHSEED;
   now ordered by depth, summed probability, then id. Reranked answers made before 2026-10-19 can differ in such ties.

Quirks pinned (`_quirk` tests): slots with 1-2 labels always abstain (half rule); decode_candidates does not
check the top candidate against `allowed` (candidates are model classes); fit_temperature with no usable gold
returns an arbitrary T within its bounds; summarise of 0 samples gives nan coverage; merge_calibrations drops
`p_top1_quantiles` (7 reads them from one unmerged calibration.json).

## 5_evaluate.py

Study-grouped cross-validation of every mapping method per slot on the Metalog samples; writes
`predictions.tsv.gz`, `metrics.json` and `calibration.json` (read by 6 and 7).

| file | covers |
|---|---|
| `test_5_evaluate_unit.py` (45) | `stack`, `precision_threshold`, `score`, `text_encoder` (TF-IDF; fake OpenAI: dedup, batches, md5 cache), `build_features` (1/sqrt(n) block weights, lookup by id, sparse), `evaluate_fold` (closed/open vocabulary, sparse features skip dense-only methods, unseen gold, single label, empty slot skipped, back-off inputs), `backoff_all`, `print_backoff` |
| `test_5_evaluate_integration.py` (25) | `main()` on the toy world: output files, exact predictions header, gold columns, calibration keys read by 6/7, TF-IDF fitted on the training fold only (leakage guard), two `.npz` blocks as in production, samples without a biome (human), unseen label, `--max_per_study`, missing vectors, `--only_samples_in`, `--fold_groups` |
| `test_5_evaluate_mutations.py` (23 + 1) | 23 bugs (scoring, features, folds split by sample, TF-IDF leakage, back-off, main), in-process |
| `step5_toy.py` | 5-d vectors (sea, freshwater, soil, lake, gut), 15 terms, 18 samples = 3 classes x 2 studies x 3, plus optional lake (unseen label) and human (no biome) studies; 2 folds; real formats (gzipped 17-column training set, `.npz` with shared vectors via `index`, `.h5` keyed by term text) |

Bugs fixed:
- A slot with no labelled sample on one side of a fold crashed; it is now skipped in that fold, with a message.
- A slot scored in a single fold crashed the back-off; it is now skipped.
- `top1_confident_half` of a one-sample group was NaN (invalid JSON); it is now None.

Quirks pinned: with term vectors every training label must be a term of the table (true on the real training set);
majority confidence is 0; ties in `precision_threshold` keep input order; the embedding cache is keyed by model +
dimensions, not `--base_url`; calibration settings report `prototype_alpha` only for the prototype method.

## 6_predict_atlas.py

Trains one method (linear / prototype / knn_study) on the evaluated Metalog samples and labels every atlas sample
by streaming the keyword / sub-biome vectors in chunks. Writes model.npz, index.npz (read by 6b), parts,
run_settings.json and atlas_predictions.tsv.gz (read by 7); with `--calibration` each slot also gets `_p`,
`_backoff`, `_backoff_label`, `_backoff_p`, `_backoff_kind`, `_candidates`.

| file | covers |
|---|---|
| `test_6_predict_atlas_unit.py` (30) | knn_study_top2 on 10 points on a half circle (one vote per study, ties, unlabelled rows, k > n); the score split (linear, prototype, knn_study) against the full vector and against `methods.knn_study`, incl. exact ties among 8 identical training vectors; ridge_class_scores / top1_and_margin (2 and 1 classes); candidate_strings; backoff_columns; load_backoff (wiring, `--accuracy` column, pooling, a model.npz without n_samples); concat_parts |
| `test_6_predict_atlas_integration.py` (35) | `main()` on 3 one-hot environments (forest / grassland / ocean), 7 training samples in 3 studies, 9 atlas samples (missing / `NA` / unknown sub-biome, brace-less keywords, unknown keywords, keywords `N/A`, sub-biome only): stages, back-off at T -> 0 and T -> inf, chunk-size invariance, resume (md5 of the calibration, changed settings refused), the stale-cache pitfall, every SystemExit, 1- and 2-label slots, k larger than the training set |
| `test_6_predict_atlas_mutations.py` (17 + 1) | 17 bugs run against the real tests with `$PREDICT_ATLAS_FILE` pointing at the mutant (~35 s). `no_sub_biome_offset_1` changes no output; only the score-identity test catches it |
| `step6_toy.py` | the `predict` fixture and writers of every input in its real format |

Bugs fixed:
1. `--method linear` on a 2-label slot crashed (RidgeClassifier keeps one score); `ridge_class_scores` now builds
   [-s, s] like `methods.linear_scores`. A 1-label slot crashed in the top-2 step; it now returns that label with margin 0.
2. `knn_study_top2`: with fewer labelled training samples than k, unlabelled samples voted for "" (which could
   win), and k >= n raised. Now only labelled samples vote and k is capped, matching `methods.knn_study`.

Pinned: ties among identical training vectors go to the lowest training rows (as in `methods.knn_study`); an
unknown or `NA` sub-biome counts as none; unknown or `N/A` keywords leave the sample out; the model cache is never
invalidated by new inputs (delete `--output_dir` to retrain).

## 6b_coverage.py

Flags atlas samples farther from the Metalog training samples than 95 % of held-out Metalog samples (threshold
calibrated with fold groups); chunked and resumable.

| file | covers |
|---|---|
| `test_6b_coverage_unit.py` (20) | `top_k_mean` and `calibrate` on 2-D points (cosines 1 / 0.8 / 0.6 / 0): same-group masking, quantiles, k, fold groups, NaN with fewer than k samples outside the group, a single group refused |
| `test_6b_coverage_integration.py` (20) | `main()`: 4 studies x 2 samples on gut / soil / lab / water axes (every held-out similarity 0.8, so threshold 0.8), 7 atlas samples (one per decision); fold groups (real 3-column `project_groups.tsv`), keyword and sub-biome vectors kept apart, max_per_study, k, chunks, only_ids, stop / resume, resume from moved inputs (md5), refusal when settings or content change |
| `test_6b_coverage_mutations.py` (12) | 12 bugs, in-process; `part_flag_strict` survives on purpose (the merge recomputes `in_coverage`) |

Bugs fixed:
1. `--chunk_rows` was not in the saved settings: resuming with another chunk size silently lost samples; now refused.
2. NaN threshold: a sample with fewer than k samples outside its group made the quantile NaN and the whole atlas
   was flagged; such samples are now NaN ("nan" in `metalog_held_out_sim.tsv.gz`) and left out; ValueError if none has a value.
3. k larger than the training set silently averaged all columns; now a clear error.
4. `--only_ids` matching no atlas sample crashed in `pd.concat`; a header-only table is now written.

## 6c_flag_controls.py

Flags blank / mock-community / negative-control samples from the GPT keyword and sub-biome texts ->
`atlas_controls.tsv.gz` (`sample_id`, `control`, `control_evidence`).

| file | covers |
|---|---|
| `test_6c_flag_controls_unit.py` (156) | `evidence()` rule by rule (sub-biome; strong keyword; weak keyword only with lab context), rule order, underscores, near misses with real phrasings (Blank Spring, mock-infected, mock control, buffer control group); `split_keywords`, `read_texts` |
| `test_6c_flag_controls_integration.py` (19) | `main()` on an 8-sample toy atlas (missing sub-biome written `NA`) and 7 shortened real GPT lines; row set and order, repeated ids, sub-biome-only samples, empty and missing files, `.gz` vs plain output |
| `test_6c_flag_controls_mutations.py` (15 + 1) | 15 bugs (~20 s) |

Bugs fixed (rerun `atlas_controls.tsv.gz`):
1. Control words matched inside longer words ('air blanket'); now word-bounded.
2. Samples present only in the sub-biome file were left out of the output.
3. gzip was forced even for a `.tsv` output name.
4. A repeated id gave duplicate rows.
5. snake_case keywords (`reagent_control`, `PCR_negative_control`) were never matched: +31 samples. `_` now counts
   as a space; the evidence quotes the text as written.
6. '<treatment> control group' was a strong keyword: 28 horse samples ('buffer control group') were flagged; now excluded.

Quirks pinned: 'pcr negative' is flagged even when it is a test result; any sub-biome ending in 'control' counts as
lab context; 'lab' is a lab keyword but not a lab sub-biome; inputs must be stripped by the caller; ~330 real blanks
are not in the vocabulary ('sterile water blank' 119, 'field blank', 'non-template control', 'pcrblank', ...):
adding them is a design decision, flip the quirk test when you do.

## 7_rerank_atlas.py and rerank_prompt.py

Optional LLM reranking of the unsure atlas samples (commands fit / build / submit / collect / apply, OpenAI Batch
API) and the pilot's exact prompt and answer parsing.

| file | covers |
|---|---|
| `test_7_rerank_atlas_unit.py` (41) | fuse, llm_logprob, fit_T, outcomes (incl. a two-parent term), fit_slot, pilot_rows, parse_candidates, estimate, body_for |
| `test_7_rerank_atlas_integration.py` (23) | the five commands through `main()` on a 6-sample toy atlas (S1-S6: confident / reranked / replicate / no metadata / failed request / at the gate): the full pilot -> fit -> build -> submit -> collect -> apply chain; gate = quantile of one unmerged calibration.json; build refusals and nothing gated; dry run, `--limit`, in-flight, `--max_pending`, an expired request, a real Batch output line; apply (back-off, abstain, latest answer wins, unusable answer) |
| `test_7_rerank_atlas_mutations.py` (18 x 2) | 18 bugs (catalogue `MUTANTS`), in-process; a parametrized test also fails if a mutated line no longer exists |
| `test_rerank_prompt_unit.py` (92) | prompt text per slot / version / confidence, 22-candidate limit, bad arguments; count_tokens; parse with log-probs (candidate order, summed spellings incl. real tokens `"\tC"`, `'"C'`, Cyrillic) and without (stated / clamped confidence, old files), real answer lines; answer_letter |
| `test_rerank_prompt_integration.py` (12) | prompt -> fake LLM -> parse round trip; the exact v2 prompt text (a failure = prompt changed: refit step 7); option orders; "none"; verbal mode; 1 and 22 candidates |
| `step7_toy.py` | toy ontology (aquatic > marine > ocean; aquatic > freshwater > lake), writers (sample.info in the real layout), keyword-rule fake LLM, fake Batch API with the real result-line layout (failed and expired requests), `load_mutant()` |

`rerank_prompt.py` has no mutation file: 12 hand-made mutants were checked once, all caught.

Fixed:
- build crashed with IndexError when no request was built (empty float row arrays); it now writes empty
  requests.jsonl / members.tsv.gz and returns.
- `outcomes` graded answers with an inline copy of `hierarchy.relation`; it now calls `hierarchy.relation` /
  `outcome` (identical output on the real pilot), so steps 5-7 cannot drift apart.

Quirks pinned: the LLM's "none of these" probability is dropped in llm_logprob, so a "none" vote fuses like an
undecided LLM (changing it would invalidate the fitted w / T2 / tau2); fit needs one unmerged calibration.json
(a merged one has no quantiles: KeyError); apply does not check that requests / members match the current atlas;
parse reads only a whole number after the letter as a stated confidence.
