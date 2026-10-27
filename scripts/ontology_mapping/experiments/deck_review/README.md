# Deck-review analyses (October 2026)

The questions raised while reviewing the presentation deck and the written report (three feedback
rounds, 2026-10-21 to 2026-10-27), each answered by one script. They reuse the pipeline's sample
selection, folds, methods and back-off (`common.py`, `methods.py`, `hierarchy.py`) through
`_common.py`, so the numbers are comparable with `5_evaluate.py` on the cleaned labels + biome map.

Run from `scripts/ontology_mapping`, e.g. `python experiments/deck_review/near_duplicates.py`. Default
paths are those of `~/MicrobeAtlasProject`; every path is an option (`--help`). Results go to
`~/MicrobeAtlasProject/ontology_mapping/experiments/deck_review/<script>.json`.

Unless stated: production prototype, fold seed 0, ≤ 50 samples per study (17,720 samples, 539 studies),
top-1 for biome / feature / material, biome without human samples.

## Index

| question | script | answer | runtime |
|---|---|---|---|
| On near-identical samples from different studies, how consistent are the curators, and where do the model's errors come from? | `near_duplicates.py` | 24 % of samples have one (cosine ≥ 0.9). Curators agree 74 / 80 / 88 %; the model (linear) is right on 89 / 96 / 99.5 % of agreeing pairs, 49 / 27 / 22 % of the others, so 61 / 81 / 96 % of its errors there are curator disagreements. The model gives both samples the same label 88 / 87 / 93 % of the time. Using all neighbours ≥ 0.9, one vote per study, or a 30 % disagreement threshold moves these by ≤ 5 points | 3 min |
| What do centring and the label prior each add to zero-shot retrieval? What happens over all 49k terms, and with per-slot term subsets? | `retrieval_ablation.py` | Closed vocabulary: raw .440 / .384 / .245, centring only .370 / .184 / .293, prior only .484 / .511 / .615, both .559 / .580 / .670 (macro .211 / .055 / .059). All terms: raw .028 / .136 / .083, centre + prior .543 / .581 / .675. Subsets (raw): biome 151 terms .376, feature 1,603 ENVO terms .182, material 947 terms .183 | 25 min, 1.5 GB |
| Would feature / material label maps help like the biome map? | `draft_slot_maps.py` | Out-of-class labels: feature 17 % of samples, material 11 %. Automatic sources agree on 37 / 64 and 57 / 84 labels, with visible errors. Untouched samples: feature .715 → .749 (macro .084 → .130), material .797 → .807 (macro .102 → .123). Worth doing, with review | 10 min |
| Back-off vs answering everything; flat cut-off; reliability; by habitat and density | `heldout_backoff.py` | Answer everything: exact .653 / .627 / .726, wrong .244 / .361 / .242. Back-off at 90 %: answered .792 / .595 / .808, exact .481 / .502 / .672, broader .230 / .035 / .057, wrong .081 / .057 / .078. Flat cut-off answers 50 / 58 / 76 %. A tau per habitat or density group does not help | 5 min |
| Does the per-study training cap matter? | `cap_sweep.py` | Top-1 within ≤ 2 points from cap 5 to no cap; macro falls without a cap (feature .094 → .076) | ~1 h per seed |
| Does class balancing pay off after back-off? Are study centroids better than knn_study? | `balance_and_study_centroids.py` | Balanced linear: learnable biome macro .42 → .52 but answered at 90 % 66 → 51 %. Centroids k = 1 / 5 / 10 do not beat knn_study | 30 min / 10 min |
| Did the new embedding run help? | `encoder_comparison.py` | kw + sb, linear: .601 / .610 / .708 (GPT-3.5 + 3-small) → .668 / .622 / .712 (GPT-5.1 + 3-large) | 10 min |
| Do the text embeddings carry habitat beyond study identity? | `cluster_purity.py` | Study-blocked purity@10 against 16S clusters: sub-biomes .111, keywords .137, kw + sb .130 (11–13 × chance); own-study neighbours 22 / 80 / 75 % | 1 min |
| What does the hybrid do to the decision regions? | `hybrid_toy.py` | Toy, computed: each region grows toward its own term; 5 % of the plane changes label; the sketch's x stays A | seconds |
| Are "too specific" answers wrong? | `too_specific_review.py` + `too_specific_review_base120.tsv` | 120 reviewed base-model answers: stated 46 %, plausible 18 %, arguable 18 %, wrong 18 % | seconds |
| Examples for the deck | `find_examples.py` | woodcock gut ERS4537110 (held out, all top-1 right, abstains on feature / material); facet pairs | 1 min |
| Atlas: confidence, out-of-coverage flags, human samples, study overlap, MIxS fields | `atlas_summary.py` (`hist`, `flags`, `human`, `study_overlap`, `mixs_fields`) | Median top-1 p 0.90 / 0.67 / 0.79; flagged: lab culture 87 %, frog skin 80 %, lab mock 77 %; 114 / 132 frequent sub-biomes never flagged; human sub-biome 1.2 M, top-1 animal-associated 99.99 %, answered 93.4 %. See the JSON for study overlap and MIxS fields | 3 min; 30–60 min for the last two (reads sample.info) |

## Notes

- Several analyses were first run as ad-hoc scripts during the review; these files are the cleaned,
  re-run versions, and every number in the table was reproduced with them on 2026-10-27 (cap sweep:
  fold seed 0 re-run, top-1 .658 / .641 / .711 at cap 5 vs .653 / .627 / .726 at cap 50; seeds 1-2 from
  the original runs).
- `atlas_summary.py study_overlap` counts 9,619 SRA study accessions behind the 552 linked Metalog
  studies (the deck first said 9,815, from an ad-hoc count whose definition was not kept); 94.7 % of
  atlas samples with a known SRA study come from a study with no linked sample.
- `near_duplicates.py` reads the out-of-fold predictions of `5_evaluate.py --backoff` (`cv_backoff/`);
  run that first. `heldout_backoff.py` re-runs the prototype itself.
- `draft_slot_maps.py` writes `draft_map_feature.tsv` / `draft_map_material.tsv` next to its JSON: a
  starting point for a reviewed map, not a map to apply as is.
