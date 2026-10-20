"""
Integration tests: 5_evaluate.py run end to end (its main(), exactly as from the command line) on the toy
world of step5_toy.py, then the three output files are checked.

How step 5 is used in the pipeline
    1_build_term_index.py   -> ontology_terms.tsv.gz    (--ontology_terms)
    2_build_training_set.py -> training_set.tsv.gz      (--samples: sample text + Metalog's gold term per slot)
    3_extract_sample_embeddings.py -> <features>.npz    (--features)
    4_embed_terms.py        -> term embeddings .h5      (--term_vectors)
    5_evaluate.py           -> metrics.json, calibration.json, predictions.tsv.gz   (--output_dir)
    6_predict_atlas.py / 7_rerank_atlas.py read calibration.json

What main() does
    1. load terms (obsolete ones dropped), term vectors, the per-sample vectors; select the samples
       (labelled, with a vector, at most --max_per_study per study);
    2. split the samples by study into --folds folds; for each fold, run every method of methods.py on the
       held-out studies, per slot (biome, feature, material);
    3. score each (slot, method), calibrate + back off the --backoff_methods, write the outputs.

The first block of tests runs the basic workflow once and reads it from several angles; the second block
changes one input or option at a time (edge cases). Toy-world reminders: 18 samples = 3 classes (marine,
freshwater, soil) x 2 studies x 3 samples, 2 folds, each fold tests one study per class; the extra lake and
human studies are added only by the tests that use them.
"""
import gzip
import json
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")  # needed to write the toy term vectors (and by 5_evaluate.py to read them)

import step5_toy as toy  # noqa: E402
from common import study_folds  # noqa: E402
from step_loader import load_step  # noqa: E402

ev = load_step("5_evaluate.py", "evaluate")

CLOSED_METHODS = ["knn", "knn_study", "linear", "retrieval", "hybrid", "retrieval_prior", "prototype", "label_reg"]
OPEN_METHODS = ["retrieval_open", "hybrid_open", "prototype_open"]
ALL_METHODS = ["majority"] + CLOSED_METHODS + OPEN_METHODS
# predictions.tsv.gz header, as in the real ontology_mapping/cv_backoff/predictions.tsv.gz
PREDICTION_COLUMNS = ["sample_id", "study_code", "domain", "slot", "method", "top5", "confidence", "gold_seen_in_train",
                      "gold", "prob", "backoff", "backoff_q", "pred", "gold_label", "pred_label", "backoff_label"]


def run_main(paths, *options):
    """Run `python 5_evaluate.py --ontology_terms ... --samples ... --output_dir ... <options>` and read the
    outputs back: (metrics dict, calibration dict or None, predictions DataFrame)."""
    argv = ["5_evaluate.py", "--ontology_terms", paths["ontology_terms.tsv"], "--samples", paths["training_set.tsv"],
            "--output_dir", paths["output_dir"], "--folds", "2", "--k", "3", "--knn_study_k", "3", *options]
    with mock.patch.object(sys, "argv", argv):
        ev.main()
    out = paths["output_dir"]
    with open(os.path.join(out, "metrics.json")) as handle:
        metrics = json.load(handle)
    calibration = None
    if os.path.exists(os.path.join(out, "calibration.json")):
        with open(os.path.join(out, "calibration.json")) as handle:
            calibration = json.load(handle)
    with gzip.open(os.path.join(out, "predictions.tsv.gz"), "rt") as handle:
        predictions = pd.read_csv(handle, sep="\t", dtype=str, keep_default_na=False)
    predictions["top5"] = predictions["top5"].str.split("||", regex=False)
    predictions["gold_seen_in_train"] = predictions["gold_seen_in_train"] == "True"
    return metrics, calibration, predictions


def embedding_options(paths):
    return ["--features", paths["features.npz"], "--term_vectors", paths["term_vectors.h5"]]


def test_toy_world_folds():
    """The fold split that the comments in step5_toy.py and below rely on (common.study_folds, seed 0)."""
    studies = np.array([study for _, study, _, _ in toy.SAMPLES + toy.LAKE_SAMPLES + toy.HUMAN_SAMPLES])
    for fold, (_, test_idx) in enumerate(study_folds(studies, n_folds=2, seed=0), start=1):
        assert set(studies[test_idx]) == toy.FOLD_TEST_STUDIES[fold]


# =============================================================================================== basic workflow
@pytest.fixture(scope="module")
def basic_run(tmp_path_factory):
    """The basic workflow: toy embeddings + toy term vectors, all methods, default back-off settings."""
    paths = toy.write_project(tmp_path_factory.mktemp("basic"))
    return run_main(paths, *embedding_options(paths))


def test_basic_outputs_are_written(basic_run):
    metrics, calibration, predictions = basic_run
    assert set(metrics) == {"biome", "feature", "material"}
    assert set(metrics["biome"]) == set(ALL_METHODS)
    assert calibration is not None
    assert list(predictions.columns) == PREDICTION_COLUMNS


def test_basic_every_sample_is_tested_exactly_once_per_slot_and_method(basic_run):
    _, _, predictions = basic_run
    # cross-validation: each sample is in the test part of exactly one fold
    assert len(predictions) == 18 * 3 * len(ALL_METHODS)
    assert not predictions.duplicated(["sample_id", "slot", "method"]).any()
    assert set(predictions["sample_id"]) == {s for s, *_ in toy.SAMPLES}


def test_basic_gold_columns_come_from_the_training_set(basic_run):
    _, _, predictions = basic_run
    row = predictions[(predictions["sample_id"] == "marine_study1_s1") & (predictions["method"] == "knn")]
    row = row.set_index("slot")
    assert row.loc["biome", "gold"] == "ENVO_marine_biome" and row.loc["biome", "gold_label"] == "marine biome"
    assert row.loc["feature", "gold"] == "ENVO_ocean"
    assert row.loc["material", "gold"] == "ENVO_sea_water"
    assert row.loc["biome", "study_code"] == "marine_study1" and row.loc["biome", "domain"] == "ocean"


def test_basic_closed_methods_are_always_right(basic_run):
    # every sample sits next to its class axis and its class is in the training fold (the other study):
    # every method that ranks the training labels finds the gold term, in every slot
    metrics, _, predictions = basic_run
    closed = predictions[predictions["method"].isin(CLOSED_METHODS)]
    wrong = closed[closed["pred"] != closed["gold"]]
    assert wrong.empty, wrong[["sample_id", "slot", "method", "gold", "pred"]]
    for slot in ev.SLOTS:
        for method in CLOSED_METHODS:
            m = metrics[slot][method]
            assert (m["n"], m["top1"], m["macro_top1"], m["pred_gold_cosine"]) == (18, 1.0, 1.0, 1.0), (slot, method)
            assert m["top1_by_domain"] == {"environmental": 1.0, "ocean": 1.0}


def test_basic_majority_is_right_on_one_class_in_three(basic_run):
    # each training fold has 3 samples of every class -> the majority answer is one of the three classes for
    # all 9 test samples of the fold: right on 3 of 9, whichever class it picks
    metrics, _, predictions = basic_run
    assert all(metrics[slot]["majority"]["top1"] == round(1 / 3, 4) for slot in ev.SLOTS)
    majority = predictions[(predictions["method"] == "majority") & (predictions["slot"] == "biome")]
    assert majority["pred"].nunique() <= 2  # one answer per fold
    assert (majority["confidence"].astype(float) == 0).all()


def test_basic_knn_confidence_is_one_when_all_neighbours_agree(basic_run):
    _, _, predictions = basic_run
    knn = predictions[predictions["method"] == "knn"]
    # k = 3 and each class has 3 training samples: the 3 nearest neighbours are always the own class
    assert np.allclose(knn["confidence"].astype(float), 1.0)


def test_basic_obsolete_terms_are_never_predicted(basic_run):
    # ENVO_old_soil has the same vector as ENVO_soil, but it is obsolete: load_terms drops it, so not even the
    # open-vocabulary methods (which rank every term of the table) can return it
    _, _, predictions = basic_run
    assert not predictions["top5"].apply(lambda top5: "ENVO_old_soil" in top5).any()


def test_basic_open_methods_rank_terms_of_other_slots(basic_run):
    # *_open methods rank all 14 terms, not only the slot's labels: for a marine sample the biome term
    # "marine biome", the feature term "ocean" and the material term "sea water" point the same way (tie),
    # so any of them can come first. Real data has the same effect with near-synonyms across slots.
    _, _, predictions = basic_run
    marine = predictions[(predictions["sample_id"] == "marine_study1_s3") & (predictions["method"] == "retrieval_open")]
    for top5 in marine["top5"]:
        assert set(top5[:3]) == {"ENVO_marine_biome", "ENVO_ocean", "ENVO_sea_water"}


def test_basic_backoff_never_needs_to_back_off(basic_run):
    # back-off (default --backoff_methods prototype linear): the classifier is always right, so the lowest tau
    # already reaches the 90 % target and the answer is the top-1 label itself, with probability ~1
    metrics, _, predictions = basic_run
    backed = predictions[predictions["method"].isin(["prototype", "linear"])]
    assert (backed["backoff"] == backed["gold"]).all()
    assert (backed["backoff_label"] == backed["gold_label"]).all()
    assert (backed["prob"].astype(float) > 0.5).all()
    report = metrics["biome"]["linear"]["backoff"]
    assert report["strict_0.9"]["coverage"] == 1.0 and report["strict_0.9"]["exact"] == 1.0
    # methods without back-off leave the three columns empty
    others = predictions[~predictions["method"].isin(["prototype", "linear"])]
    assert (others[["prob", "backoff", "backoff_q"]] == "").all().all()


def test_basic_calibration_for_step_6(basic_run):
    _, calibration, _ = basic_run
    assert {slot: set(methods) for slot, methods in calibration.items()} == \
           {slot: {"linear", "prototype"} for slot in ev.SLOTS}
    settings = calibration["biome"]["prototype"]["settings"]
    assert settings["n_samples"] == 18 and settings["max_per_study"] == 50 and settings["fold_seed"] == 0
    assert settings["seed"] == 22 and settings["prototype_beta"] == 0.1
    assert os.path.basename(settings["samples"]) == "training_set.tsv.gz"  # step 6 compares the file name
    assert settings["prototype_alpha"] == 0.5  # reported for the prototype method only
    assert calibration["biome"]["linear"]["settings"]["prototype_alpha"] == 0
    # what 6_predict_atlas.py (hierarchy.choose_tau on the curve, softmax temperature) and 7_rerank_atlas.py
    # (gate = p_top1_quantiles[str(--gate)]) read from every entry
    for slot in ev.SLOTS:
        for entry in calibration[slot].values():
            assert entry["temperature"] > 0
            assert {"tau", "coverage", "accuracy_strict", "accuracy_lenient", "exact"} <= set(entry["curve"][0])
            assert set(entry["p_top1_quantiles"]) == {"0.1", "0.2", "0.3", "0.4", "0.5"}


# =============================================================================================== edge cases
def test_without_term_vectors_only_label_based_methods_run(tmp_path):
    # No --term_vectors: nothing to compare samples with terms, so only majority / knn / knn_study / linear
    paths = toy.write_project(tmp_path)
    metrics, calibration, _ = run_main(paths, "--features", paths["features.npz"])
    assert set(metrics["biome"]) == {"majority", "knn", "knn_study", "linear"}
    assert "pred_gold_cosine" not in metrics["biome"]["knn"]
    assert metrics["biome"]["linear"]["top1"] == 1.0
    assert set(calibration["biome"]) == {"linear"}  # prototype (default back-off method) was not computed


@pytest.mark.parametrize("with_term_vectors", [True, False])
def test_slot_nobody_labelled_is_left_out(tmp_path, with_term_vectors):
    # Only biome is labelled: the feature and material slots simply do not appear in the outputs, with the
    # default back-off too. (This used to crash: an empty slot was still handed to the methods and the back-off.)
    paths = toy.write_project(tmp_path, slots=("biome",))
    options = embedding_options(paths) if with_term_vectors else ["--features", paths["features.npz"]]
    metrics, calibration, predictions = run_main(paths, *options)
    assert set(metrics) == set(calibration) == set(predictions["slot"]) == {"biome"}
    assert metrics["biome"]["knn"]["top1"] == 1.0


def test_backoff_switched_off(tmp_path):
    # `--backoff_methods` with no value: no calibrated probabilities, and no calibration.json is written
    paths = toy.write_project(tmp_path)
    metrics, calibration, predictions = run_main(paths, *embedding_options(paths), "--backoff_methods")
    assert calibration is None
    assert "backoff" not in metrics["biome"]["linear"]
    assert (predictions["backoff"] == "").all()


def test_slot_labelled_by_one_study_only(tmp_path, capsys):
    # Only soil_study2 has material labels. It is tested in fold 1, where no training sample has a material
    # label (nothing to learn from: skipped, with a message); in fold 2 it is training data but no test sample
    # has a material label (nothing to score: skipped). So material is never scored; the other slots are.
    paths = toy.write_project(tmp_path)
    samples = pd.read_csv(paths["training_set.tsv"], sep="\t", dtype=str, keep_default_na=False)
    samples.loc[samples["study_code"] != "soil_study2", "material"] = ""
    samples.to_csv(paths["training_set.tsv"], sep="\t", index=False)
    metrics, _, _ = run_main(paths, *embedding_options(paths))
    assert set(metrics) == {"biome", "feature"}
    assert "material: no labelled training sample in this fold, 3 test samples not scored" in capsys.readouterr().out


def test_tfidf_features_from_the_sample_text(tmp_path):
    # default --features tfidf: vectors from the `text` column (fitted on each training fold), no .npz needed.
    # Each class has its own words ("sea water ocean sample", "river fresh water sample", "field soil crop
    # sample"). The dense-only methods (prototype, retrieval_prior, label_reg) are skipped.
    paths = toy.write_project(tmp_path)
    metrics, calibration, _ = run_main(paths)
    assert set(metrics["biome"]) == {"majority", "knn", "knn_study", "linear",
                                     "retrieval", "hybrid", "retrieval_open", "hybrid_open"}
    assert metrics["biome"]["knn"]["top1"] == 1.0 and metrics["biome"]["linear"]["top1"] == 1.0
    assert set(calibration["biome"]) == {"linear"}


def test_tfidf_is_fitted_on_the_training_fold_only(tmp_path):
    # Leakage guard: a fold's TF-IDF vocabulary and idf weights come from its training texts (plus the term
    # texts), never from the held-out samples. Each toy text ends with its sample id, so the fitted texts tell
    # which samples they came from.
    paths = toy.write_project(tmp_path)
    ids = {s for s, *_ in toy.SAMPLES}
    real, fitted = ev.text_encoder, []

    def spy(name, fit_texts, *rest):
        fitted.append({t.split()[-1] for t in fit_texts if t.split() and t.split()[-1] in ids})
        return real(name, fit_texts, *rest)

    with mock.patch.object(ev, "text_encoder", spy):
        run_main(paths)
    assert len(fitted) == 2  # one fit per fold
    for fold, fitted_ids in enumerate(fitted, start=1):
        assert fitted_ids == {s for s, study, *_ in toy.SAMPLES if study not in toy.FOLD_TEST_STUDIES[fold]}


def test_two_feature_blocks_as_in_the_real_command(tmp_path):
    # The production run: --features keywords.npz sub_biomes.npz --term_vectors <terms>.h5. The real sub-biome
    # .npz has 30 samples fewer than the keyword one: a sample needs a vector in every block to be used.
    paths = toy.write_project(tmp_path)
    sub_biomes = str(tmp_path / "sub_biomes.npz")
    toy.write_npz(sub_biomes, [s for s in toy.SAMPLES if s[0] != "freshwater_study2_s1"])
    metrics, calibration, predictions = run_main(paths, "--features", paths["features.npz"], sub_biomes,
                                                 "--term_vectors", paths["term_vectors.h5"])
    assert "freshwater_study2_s1" not in set(predictions["sample_id"])
    assert metrics["biome"]["knn"]["n"] == 17 and calibration["biome"]["prototype"]["settings"]["n_samples"] == 17
    closed = predictions[predictions["method"].isin(CLOSED_METHODS)]
    assert (closed["pred"] == closed["gold"]).all()
    # sample . term = mean of the two block cosines (both blocks hold the same vectors here): retrieval's
    # confidence (the best cosine) for marine_study1_s3 = (1, 0, 0, 0, 0) and marine biome is (1 + 1) / 2 = 1
    row = predictions[(predictions["sample_id"] == "marine_study1_s3") & (predictions["slot"] == "biome")
                      & (predictions["method"] == "retrieval")]
    assert float(row["confidence"].iloc[0]) == pytest.approx(1.0)


def test_human_samples_have_no_biome(tmp_path):
    # As in the real training set (no human sample has a biome label), human samples are labelled in feature
    # and material only: they are scored in those two slots, never enter the biome training data, and still
    # count in n_samples (step 6 checks it against its own training set).
    paths = toy.write_project(tmp_path, samples=toy.SAMPLES + toy.HUMAN_SAMPLES)
    metrics, calibration, predictions = run_main(paths, *embedding_options(paths))
    human = predictions[predictions["sample_id"].str.startswith("human")]
    assert set(human["slot"]) == {"feature", "material"}
    assert metrics["biome"]["knn"]["n"] == 18 and metrics["feature"]["knn"]["n"] == 24
    assert set(metrics["feature"]["knn"]["top1_by_domain"]) == {"environmental", "human", "ocean"}
    assert "human" not in metrics["biome"]["knn"]["top1_by_domain"]
    closed = human[human["method"].isin(CLOSED_METHODS)]
    assert (closed["pred"] == closed["gold"]).all()  # intestine environment / fecal material
    assert calibration["biome"]["prototype"]["settings"]["n_samples"] == 24


def test_unseen_gold_label(tmp_path):
    # Add a lake study (lake biome, a child of freshwater biome). It is tested in fold 1, and no other sample
    # is a lake: when it is tested, no training sample carries its label.
    paths = toy.write_project(tmp_path, samples=toy.SAMPLES + toy.LAKE_SAMPLES)
    metrics, _, predictions = run_main(paths, *embedding_options(paths))
    lake = predictions[predictions["sample_id"].str.startswith("lake") & (predictions["slot"] == "biome")]
    assert not lake["gold_seen_in_train"].any()
    pred = lake.groupby("method")["pred"].agg(set)
    # closed vocabulary: the best training label is freshwater biome (true, but coarser than lake biome)
    assert pred["knn"] == pred["retrieval"] == {"ENVO_freshwater_biome"}
    # open vocabulary: the lake axis is closest -> lake biome, although no training sample has it
    assert pred["retrieval_open"] == {"ENVO_lake_biome"}
    assert metrics["biome"]["retrieval_open"]["n_unseen_label"] == 3
    assert metrics["biome"]["retrieval_open"]["top1_unseen_label"] == 1.0
    assert metrics["biome"]["knn"]["top1_unseen_label"] == 0.0
    # their feature (river) and material (fresh water) labels are seen
    assert predictions[predictions["sample_id"].str.startswith("lake") & (predictions["slot"] != "biome")][
        "gold_seen_in_train"].all()


def test_max_per_study_caps_each_study(tmp_path):
    paths = toy.write_project(tmp_path)
    metrics, _, predictions = run_main(paths, *embedding_options(paths), "--max_per_study", "2")
    knn = predictions[(predictions["method"] == "knn") & (predictions["slot"] == "biome")]
    assert knn.groupby("study_code").size().tolist() == [2] * 6  # 6 studies x 2 samples
    assert metrics["biome"]["knn"]["n"] == 12


def test_samples_without_a_vector_are_dropped(tmp_path):
    # soil_study1_s3 is labelled but has no vector in the .npz (e.g. its text could not be embedded)
    has_vector = [s for s in toy.SAMPLES if s[0] != "soil_study1_s3"]
    paths = toy.write_project(tmp_path, npz_samples=has_vector)
    metrics, _, predictions = run_main(paths, *embedding_options(paths))
    assert "soil_study1_s3" not in set(predictions["sample_id"])
    assert metrics["biome"]["knn"]["n"] == 17


def test_only_samples_in_restricts_to_another_npz(tmp_path):
    # --only_samples_in: also require a vector in another .npz, to compare two runs on identical samples
    paths = toy.write_project(tmp_path)
    other = str(tmp_path / "other.npz")
    toy.write_npz(other, [s for s in toy.SAMPLES if s[1] != "marine_study1"])  # no vectors for one study
    _, _, predictions = run_main(paths, *embedding_options(paths), "--only_samples_in", other)
    assert "marine_study1" not in set(predictions["study_code"])
    assert predictions["sample_id"].nunique() == 15


def test_fold_groups_keep_a_project_on_one_side(tmp_path):
    # --fold_groups: declare the two marine studies one project. Whole projects go to one fold, so both marine
    # studies are tested together, and while they are tested no marine sample is left for training: the marine
    # labels become unseen. Samples missing from the file keep their study_code as group.
    paths = toy.write_project(tmp_path)
    groups = str(tmp_path / "project_groups.tsv")
    marine = [s for s, study, *_ in toy.SAMPLES if study.startswith("marine")]
    pd.DataFrame({"sample_id": marine, "project_group": "marine_project"}).to_csv(groups, sep="\t", index=False)
    metrics, _, predictions = run_main(paths, *embedding_options(paths), "--fold_groups", groups)
    is_marine = predictions["sample_id"].isin(marine)
    assert not predictions.loc[is_marine, "gold_seen_in_train"].any()
    assert predictions.loc[~is_marine, "gold_seen_in_train"].all()
    marine_knn = predictions[is_marine & (predictions["method"] == "knn")]
    assert (marine_knn["pred"] != marine_knn["gold"]).all()  # a label absent from training cannot be voted for
    assert metrics["biome"]["knn"]["top1"] == round(12 / 18, 4)  # the 12 freshwater / soil samples stay right
