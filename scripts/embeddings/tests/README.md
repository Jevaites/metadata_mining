# Tests of `embed_subbiomes_keywords.py` (`scripts/embeddings`)

`embed_subbiomes_keywords.py` embeds the GPT sub-biome / keyword text of every MicrobeAtlas sample. Each distinct
text is embedded once (`GPT_{target}_unique_embeddings__{tag}.h5`: `texts`, `embeddings`), then the vectors are
copied to each sample (`GPT_{target}_embeddings__{tag}.h5`: `sample_ids`, `texts`, `embeddings`). Both files
are the resume state.

Its `clean_text` / `iter_samples` are imported by ontology-mapping steps 3 and 6, and `embed_unique` /
`estimate_tokens` / the h5 helpers by step 4, so their outputs are a contract: texts must match the embedded ones
byte for byte. The function-level tests of these live here; `scripts/ontology_mapping/tests` only tests how the
steps use them.

## Files

| file | covers |
|---|---|
| `test_embed_subbiomes_keywords_unit.py` (57) | one class per function. `clean_text`: braces, commas, whitespace, case and hyphens kept, brace-less real lines. `iter_samples`: NA values, malformed lines, CRLF, first tab only. `pick_subset`: `--full`, `--n`, `--n_per_biome`, `--sample_ids_file`, seeds, tags. `estimate_tokens`: with, without and failing tiktoken. `check_disk_space`: 5 % margin. `h5_append` / `h5_strings` / `h5_align`: growable columns, utf-8, ragged-file repair, a column never created. `embed_unique`: batches, resume, `dimensions` only below the native size, float32 output, wrong vector size, API failure. `materialize`: broadcast, chunks, resume, missing vector |
| `test_embed_subbiomes_keywords_integration.py` (29) | `main()` end to end: full run and file names, manifest, one target, `--unique_only`; subsets (id file, saved subset ids, ids without text, per-biome, no file collisions); stops before any API call (`--dry_run` cost, the prompt, full disk, dim above native, no subset); resume (free rerun, API failure then rerun, new samples, half-written row, crash during the first write of either file); duplicate ids, keyword sets that clean to the same text, unicode, a target with no usable text |
| `helpers.py` | `FakeOpenAI` (returns the vector given per text, records calls, can fail on call N), `read_h5`, `rows`, `write_lines` |
| `conftest.py` | puts `scripts/embeddings` on `sys.path`, stubs `openai` if it is missing, blocks tiktoken in every test (token estimate = len // 4, checkable by hand) |

The mutants of `embed_unique` / `h5_align` are in `scripts/ontology_mapping/tests/test_4_embed_terms_mutations.py`
(their docstrings name the tests here that catch them).

## Toy world

- Unit tests: inline 2-3 line inputs with 2-d vectors ([1,0] = soil, ...).
- Integration: 7 samples (s1-s3 soil, s4-s6 water, s7 animal); s7's sub-biome is `NA` and s6's keywords are `{}`;
  3 distinct sub-biome and 5 distinct keyword texts; 3-d vectors, `text-embedding-3-small` at dim 3.
- Real-format cases copied from `sidequest/latest/GPT_*.txt`: `{Short-tailed shearwater, chick, gut}`,
  brace-less `a,b,c` (285 real keyword lines), `NA` (the only NA form; 18,520 sub-biomes).

## Running

```bash
cd scripts/embeddings && python -m pytest tests -q     # < 1 s; no network, no API key
```

## Bugs fixed (pinned by tests)

- A target with no usable text in the chosen subset crashed opening a unique file that was never written. It is
  now skipped with no API call and no files (`test_target_without_any_usable_text_in_subset`).
- A crash during the very first write of either `.h5` file left a file with only some columns, which looked
  complete on resume: texts counted as done with no vector, and later samples were paired with the wrong vector.
  `h5_align` now counts a missing column as 0 rows (`test_column_missing_after_crash_on_first_write`,
  `test_crash_during_first_write_*`).

## Known quirks (`..._quirk` tests)

- Keyword boundaries are lost: `{forest soil, leaf}` and `{forest, soil leaf}` give the same text and vector.
- An unbalanced brace is kept.
- `--n_per_biome` groups are case-sensitive.
- A subset flag silently wins over `--full`.
- `h5_append` of 0 rows raises.
- A duplicate sample id gives two rows (the real files have none).
- Resume only checks sample ids, so an edited source text is not picked up: use a new `--output_dir` after
  regenerating the GPT files.
