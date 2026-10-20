"""
Integration tests: 6_predict_atlas.py run end to end (main(), with real files) on a toy atlas.

THE TOY WORLD ------------------------------------------------------------------------------------
Embeddings are 3-d one-hot vectors, one direction per environment (the same for keywords, sub-biomes
and ontology terms), so every sample sits exactly on "its" environment:

    direction   keyword text (.h5)    sub-biome text (.h5)   biome            feature     material
    e0 forest   "forest trees"        "woodland"             forest biome     woodland    leaf litter
    e1 grass    "meadow grass"        "pasture"              grassland biome  meadow      soil
    e2 ocean    "ocean seawater"      "open sea"             marine biome     ocean       sea water

Ontology (is_a):  forest biome, grassland biome -> terrestrial biome -> biome (root)
                  marine biome -> biome;   leaf litter, soil, sea water -> environmental material (root)

Training set (Metalog), 7 samples in 3 studies; M7 has no feature label:
    M1, M2  STUDY_FOREST   forest      M3, M4  STUDY_GRASS   grassland
    M5, M6, M7  STUDY_OCEAN  ocean

Atlas (GPT_keywords.txt / GPT_sub_biomes.txt), one sample per situation the script must handle:
    sample               keywords            sub-biome            expected
    A_FOREST             {forest, trees}     woodland             forest row
    A_GRASS              {meadow, grass}     pasture              grassland row
    A_OCEAN              {ocean, seawater}   open sea             ocean row
    A_FOREST_NO_SB       {forest, trees}     (no line)            forest row, from the keywords alone
    A_GRASS_SB_NA        meadow,grass (**)   NA                   grassland row, from the keywords alone
    A_OCEAN_SB_UNKNOWN   {ocean, seawater}   abyssal plain (*)    ocean row, sub-biome ignored
    A_KW_UNKNOWN         {volcano, ash} (*)  -                    not in the output
    A_KW_NA              N/A                 -                    not in the output
    A_SB_ONLY            (no line)           woodland             not in the output (keywords are required)
    (*) text missing from the .h5 (never embedded)
    (**) no braces, as on some real lines: cleaned to 'meadow grass' like '{meadow, grass}'
    'N/A' (keywords) and 'NA' (sub-biomes) are how the real GPT_*.txt files write "none".
The keyword .h5 has a 4th row, "unused text", that no sample refers to (a chunk can be empty).

WORKFLOW (what main() does, and what the tests below check stage by stage) -----------------------
    1. train_models   -> <out>/model.npz        one model per slot on the evaluated training samples
    2. build_index    -> <out>/index.npz        each atlas sample's row in the keyword / sub-biome .h5
    3. per chunk of --chunk_rows keyword .h5 rows -> <out>/parts/rows_<first row>.tsv.gz
    4. concat_parts   -> <out>/atlas_predictions.tsv.gz
Stages 1-3 are cached: rerunning the same command resumes (and never recomputes a finished part).
"""
import gzip
import os

import numpy as np
import pandas as pd
import pytest

from common import load_terms
from step6_toy import (predict, read_predictions, run_predict,  # noqa: F401  (predict: a fixture)
                        write_calibration, write_sample_vectors, write_samples,
                       write_terms, write_text_embeddings, write_texts)

E0, E1, E2 = np.eye(3)  # forest, grassland, ocean
BIOME_ROOT, MATERIAL_ROOT = "ENVO_00000428", "ENVO_00010483"  # in hierarchy.NO_ANSWER: never an answer

TERMS = [  # (term_id, label, synonyms, parents)
    (BIOME_ROOT, "biome", [], []),
    ("ENVO_TERRESTRIAL", "terrestrial biome", [], [BIOME_ROOT]),
    ("ENVO_FOREST", "forest biome", ["woodland biome"], ["ENVO_TERRESTRIAL"]),
    ("ENVO_GRASSLAND", "grassland biome", ["prairie biome"], ["ENVO_TERRESTRIAL"]),
    ("ENVO_MARINE", "marine biome", ["ocean biome"], [BIOME_ROOT]),
    ("ENVO_WOODLAND", "woodland", [], []),
    ("ENVO_MEADOW", "meadow", [], []),
    ("ENVO_OCEAN", "ocean", [], []),
    (MATERIAL_ROOT, "environmental material", [], []),
    ("ENVO_LITTER", "leaf litter", [], [MATERIAL_ROOT]),
    ("ENVO_SOIL", "soil", [], [MATERIAL_ROOT]),
    ("ENVO_SEAWATER", "sea water", [], [MATERIAL_ROOT]),
]
FOREST = ("ENVO_FOREST", "ENVO_WOODLAND", "ENVO_LITTER")  # (biome, feature, material)
GRASS = ("ENVO_GRASSLAND", "ENVO_MEADOW", "ENVO_SOIL")
OCEAN = ("ENVO_MARINE", "ENVO_OCEAN", "ENVO_SEAWATER")
TERM_DIRECTION = {"ENVO_FOREST": E0, "ENVO_WOODLAND": E0, "ENVO_LITTER": E0, "ENVO_GRASSLAND": E1,
                  "ENVO_MEADOW": E1, "ENVO_SOIL": E1, "ENVO_MARINE": E2, "ENVO_OCEAN": E2, "ENVO_SEAWATER": E2}

TRAINING = [  # (sample_id, study_code, labels, direction of its keyword and sub-biome vectors)
    ("M1", "STUDY_FOREST", FOREST, E0), ("M2", "STUDY_FOREST", FOREST, E0),
    ("M3", "STUDY_GRASS", GRASS, E1), ("M4", "STUDY_GRASS", GRASS, E1),
    ("M5", "STUDY_OCEAN", OCEAN, E2), ("M6", "STUDY_OCEAN", OCEAN, E2),
    ("M7", "STUDY_OCEAN", ("ENVO_MARINE", "", "ENVO_SEAWATER"), E2),  # no feature label
]
KEYWORD_H5 = {"forest trees": E0, "meadow grass": E1, "ocean seawater": E2, "unused text": (E0 + E1) / 2}
SUB_BIOME_H5 = {"woodland": E0, "pasture": E1, "open sea": E2}
KEYWORD_LINES = [("A_FOREST", "{forest, trees}"), ("A_GRASS", "{meadow, grass}"), ("A_OCEAN", "{ocean, seawater}"),
                 ("A_FOREST_NO_SB", "{forest, trees}"), ("A_GRASS_SB_NA", "meadow,grass"),
                 ("A_OCEAN_SB_UNKNOWN", "{ocean, seawater}"), ("A_KW_UNKNOWN", "{volcano, ash}"), ("A_KW_NA", "N/A")]
SUB_BIOME_LINES = [("A_FOREST", "woodland"), ("A_GRASS", "pasture"), ("A_OCEAN", "open sea"),
                   ("A_GRASS_SB_NA", "NA"), ("A_OCEAN_SB_UNKNOWN", "abyssal plain"), ("A_SB_ONLY", "woodland")]

EXPECTED = {  # the answer for every atlas sample that must appear in the output
    "A_FOREST": FOREST, "A_GRASS": GRASS, "A_OCEAN": OCEAN,
    "A_FOREST_NO_SB": FOREST, "A_GRASS_SB_NA": GRASS, "A_OCEAN_SB_UNKNOWN": OCEAN,
}


class Toy:
    """The toy world written to tmp_path; argv() gives the command line of 6_predict_atlas.py."""

    def __init__(self, root, training=TRAINING, keyword_lines=KEYWORD_LINES):
        self.root = root
        self.terms = write_terms(root / "ontology_terms.tsv.gz", TERMS)
        # the training file has the name of the real one: load_backoff compares it with calibration.json
        self.samples = write_samples(root / "metalog_training_set.tsv.gz",
                                     [(s, study, *labels) for s, study, labels, _ in training])
        self.train_kw = write_sample_vectors(root / "keywords.npz", {s: d for s, _, _, d in training})
        self.train_sb = write_sample_vectors(root / "sub_biomes.npz", {s: d for s, _, _, d in training})
        self.kw_texts = write_texts(root / "GPT_keywords.txt", keyword_lines)
        self.sb_texts = write_texts(root / "GPT_sub_biomes.txt", SUB_BIOME_LINES)
        self.kw_h5 = write_text_embeddings(root / "keywords_unique.h5", KEYWORD_H5)
        self.sb_h5 = write_text_embeddings(root / "sub_biomes_unique.h5", SUB_BIOME_H5)
        texts = load_terms(self.terms).set_index("term_id")["text"]  # 'forest biome; woodland biome'
        self.term_vectors = write_text_embeddings(root / "term_vectors.h5",
                                                  {texts[t]: d for t, d in TERM_DIRECTION.items()})

    def argv(self, out="out", method="linear", *extra):
        options = ["--ontology_terms", self.terms, "--samples", self.samples,
                   "--train_vectors", self.train_kw, self.train_sb,
                   "--keywords_texts", self.kw_texts, "--keywords_h5", self.kw_h5,
                   "--sub_biomes_texts", self.sb_texts, "--sub_biomes_h5", self.sb_h5,
                   "--output_dir", self.root / out, "--method", method]
        if method == "prototype":
            options += ["--term_vectors", self.term_vectors]
        if method == "knn_study":
            options += ["--knn_study_k", 2]  # 2 neighbours = the 2 samples of the right study: vote shares of 1.0
        return [*options, *extra]

    def calibration(self, name="calibration.json", method="linear", temperature=0.1, **settings):
        """A calibration.json of 5_evaluate.py for this training set; tau 0.5 reaches the 0.9 target."""
        settings = {"samples": "/anywhere/metalog_training_set.tsv.gz", "n_samples": len(TRAINING),
                    "max_per_study": 50, "seed": 22, "prototype_alpha": 0.5, "prototype_beta": 0.1, **settings}
        return write_calibration(self.root / name, method, temperature, [(0.5, 0.95), (0.8, 0.99)], settings)


@pytest.fixture
def toy(tmp_path):
    return Toy(tmp_path)


def answers(table):
    """{sample_id: (biome, feature, material)} of a prediction table."""
    return {s: (r["biome"], r["feature"], r["material"]) for s, r in table.iterrows()}


# ============================================================================= the basic workflow
class TestWorkflow:
    @pytest.mark.parametrize("method", ["linear", "prototype", "knn_study"])
    def test_every_method_finds_the_obvious_answers(self, predict, toy, method):
        run_predict(predict, *toy.argv("out", method))
        table = read_predictions(toy.root / "out")
        assert answers(table) == EXPECTED  # also: exactly these 6 samples, no others
        # one term id, its label and a confidence per slot
        assert list(table.columns) == ["biome", "biome_label", "biome_confidence", "feature", "feature_label",
                                       "feature_confidence", "material", "material_label", "material_confidence"]
        assert table.loc["A_FOREST", "biome_label"] == "forest biome"

    def test_linear_confidence_is_a_margin_and_drops_without_sub_biome(self, predict, toy):
        run_predict(predict, *toy.argv())
        confidence = read_predictions(toy.root / "out")["biome_confidence"].astype(float)
        assert (confidence > 0).all()  # margin = best - second best score
        # keywords + sub-biome point to forest: a larger margin than the keywords alone
        assert confidence["A_FOREST"] > confidence["A_FOREST_NO_SB"]
        # an unknown sub-biome text, or 'NA', is the same as no sub-biome
        assert confidence["A_OCEAN"] > confidence["A_OCEAN_SB_UNKNOWN"]
        assert confidence["A_GRASS"] > confidence["A_GRASS_SB_NA"]
        assert confidence["A_GRASS_SB_NA"] == confidence["A_FOREST_NO_SB"]  # forest and grassland are symmetric in the toy

    def test_knn_study_confidence_is_the_vote_share(self, predict, toy):
        run_predict(predict, *toy.argv("out", "knn_study"))
        table = read_predictions(toy.root / "out")
        # k = 2: both neighbours of every sample are from its own environment's study -> 100 % of the vote
        assert set(table["biome_confidence"]) == {"1.0"}


class TestWorkflowStages:
    """What stages 1-3 leave in --output_dir (chunk size 2: keyword .h5 rows 0-1 and 2-3); stage 4:
    TestChunksAndResume and test_6_predict_atlas_unit.py::TestConcatParts."""

    @pytest.fixture
    def out(self, predict, toy):
        run_predict(predict, *toy.argv("out", "knn_study", "--chunk_rows", 2))
        return toy.root / "out"

    def test_stage1_model(self, out):
        model = np.load(out / "model.npz")
        assert str(model["method"]) == "knn_study" and int(model["n_samples"]) == 7 and int(model["kw_dim"]) == 3
        assert list(model["biome_classes"]) == ["ENVO_FOREST", "ENVO_GRASSLAND", "ENVO_MARINE"]
        # knn_study keeps "" as a class: training samples without a label in the slot (M7's feature),
        # never used as neighbours
        assert list(model["feature_classes"]) == ["", "ENVO_MEADOW", "ENVO_OCEAN", "ENVO_WOODLAND"]
        assert (model["feature_classes"][model["feature_labels"]] == "").sum() == 1

    def test_stage2_index(self, out):
        index = np.load(out / "index.npz")
        rows = {s: (int(k), int(b)) for s, k, b in zip(index["sample_ids"], index["kw_rows"], index["sb_rows"])}
        # (row in the keyword .h5, row in the sub-biome .h5 or -1); '{forest, trees}' was cleaned to
        # 'forest trees' before the lookup, as when the texts were embedded
        assert rows == {"A_FOREST": (0, 0), "A_GRASS": (1, 1), "A_OCEAN": (2, 2),
                        "A_FOREST_NO_SB": (0, -1), "A_GRASS_SB_NA": (1, -1), "A_OCEAN_SB_UNKNOWN": (2, -1)}

    def test_stage3_one_part_per_chunk(self, out):
        assert sorted(os.listdir(out / "parts")) == ["rows_00000000.tsv.gz", "rows_00000002.tsv.gz"]
        part = pd.read_csv(out / "parts" / "rows_00000000.tsv.gz", sep="\t")
        assert sorted(part["sample_id"]) == ["A_FOREST", "A_FOREST_NO_SB", "A_GRASS", "A_GRASS_SB_NA"]  # keyword rows 0, 1


# ============================================================================= back-off (--calibration)
class TestBackoff:
    """With --calibration, every slot gets <slot>_p, _backoff, _backoff_label, _backoff_p, _backoff_kind
    and _candidates. The toy calibration has tau 0.5; only the temperature is varied here (the
    top1 / coarser / abstain logic itself is shown on explicit probabilities in the unit tests)."""

    @pytest.mark.parametrize("method", ["linear", "prototype"])  # the real atlas_backoff run is prototype
    def test_a_confident_model_answers_its_top1(self, predict, toy, method):
        # T = 0.001: softmax(scores / T) is ~1 for the best label -> no back-off needed
        run_predict(predict, *toy.argv("out", method, "--calibration", toy.calibration(method=method, temperature=0.001)))
        table = read_predictions(toy.root / "out")
        assert set(table["biome_backoff_kind"]) == {"top1"}
        assert (table["biome_backoff"] == table["biome"]).all() and set(table["biome_p"]) == {"1.0"}
        assert table.loc["A_FOREST", "biome_backoff_label"] == "forest biome"
        # the column order of the real atlas_backoff/atlas_predictions.tsv.gz (7_rerank_atlas.py reads them by name)
        assert list(table.columns[:10]) == ["biome", "biome_label", "biome_confidence", "biome_p", "biome_backoff",
                                            "biome_backoff_label", "biome_backoff_p", "biome_backoff_kind",
                                            "biome_candidates", "feature"]
        assert len(table.columns) == 27  # 9 per slot
        # candidates: the 3 labels (--topk 5 > 3), most probable first, as 'id:p;id:p;id:p'
        assert table.loc["A_FOREST", "biome_candidates"].split(";")[0] == "ENVO_FOREST:1"
        assert len(table.loc["A_FOREST", "biome_candidates"].split(";")) == 3

    def test_an_unsure_model_abstains_rather_than_answering_the_root(self, predict, toy):
        # T = 1e6: every label gets 1/3 (scores / T are all ~0). No label reaches tau = 0.5, terrestrial biome is not a
        # label of the slot, and the root 'biome' (q = 1) is never an answer -> abstain everywhere
        run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration(temperature=1e6)))
        table = read_predictions(toy.root / "out")
        assert set(table["biome_backoff_kind"]) == {"abstain"}
        assert set(table["biome_backoff"]) == {""} and set(table["biome_p"]) == {"0.3333"}
        assert answers(table) == EXPECTED  # the plain top-1 columns are unchanged

    def test_topk_0_drops_the_candidates_column(self, predict, toy):
        run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration(), "--topk", 0))
        assert "biome_candidates" not in read_predictions(toy.root / "out").columns


# ============================================================================= chunking and resuming
class TestChunksAndResume:
    def test_the_chunk_size_does_not_change_the_result(self, predict, toy):
        run_predict(predict, *toy.argv("big", "linear", "--chunk_rows", 1000))  # one chunk
        run_predict(predict, *toy.argv("small", "linear", "--chunk_rows", 1))   # one chunk per .h5 row (row 3: empty)
        big, small = read_predictions(toy.root / "big"), read_predictions(toy.root / "small")
        pd.testing.assert_frame_equal(big.sort_index(), small.sort_index())   # row order follows the chunks

    def test_stopped_by_max_seconds_then_resumed(self, predict, toy, capsys):
        run_predict(predict, *toy.argv("out", "linear", "--max_seconds", 1e-9))  # stops before the first chunk
        assert "Stopping after" in capsys.readouterr().out
        out = toy.root / "out"
        assert (out / "model.npz").exists() and (out / "index.npz").exists()
        assert os.listdir(out / "parts") == [] and not (out / "atlas_predictions.tsv.gz").exists()
        run_predict(predict, *toy.argv("out", "linear"))  # the same command again finishes the job
        assert answers(read_predictions(out)) == EXPECTED

    def test_a_finished_part_is_never_recomputed(self, predict, toy):
        run_predict(predict, *toy.argv("out", "linear", "--chunk_rows", 2))
        out = toy.root / "out"
        # simulate a run interrupted after the first chunk: part 2 missing; and mark part 1 by hand
        first = out / "parts" / "rows_00000000.tsv.gz"
        text = gzip.open(first, "rt").read().replace("forest biome", "MARKED BY THE TEST")
        with gzip.open(first, "wt") as handle:
            handle.write(text)
        os.remove(out / "parts" / "rows_00000002.tsv.gz")
        run_predict(predict, *toy.argv("out", "linear", "--chunk_rows", 2))
        table = read_predictions(out)
        assert table.loc["A_FOREST", "biome_label"] == "MARKED BY THE TEST"  # part 1 kept as it was
        assert table.loc["A_OCEAN", "biome"] == "ENVO_MARINE"                # part 2 recomputed

    def test_resume_with_back_off_checks_the_settings(self, predict, toy):
        # run_settings.json keeps the md5 of each calibration.json: the same file at another path resumes,
        # another --target_accuracy is refused (its chunks would mix two taus)
        calibration = toy.calibration()
        run_predict(predict, *toy.argv("out", "linear", "--calibration", calibration, "--chunk_rows", 2))
        os.remove(toy.root / "out" / "parts" / "rows_00000002.tsv.gz")
        moved = toy.root / "moved.json"
        os.rename(calibration, moved)
        run_predict(predict, *toy.argv("out", "linear", "--calibration", moved, "--chunk_rows", 2))
        assert answers(read_predictions(toy.root / "out")) == EXPECTED
        with pytest.raises(SystemExit, match="has chunks written with"):
            run_predict(predict, *toy.argv("out", "linear", "--calibration", moved, "--target_accuracy", 0.95))

    def test_the_cache_is_not_invalidated_by_new_inputs(self, predict, toy):
        # PITFALL (documented in the docstring: "delete --output_dir to retrain"): model.npz, index.npz
        # and the parts are reused as they are, even if the input files changed in between
        run_predict(predict, *toy.argv())
        toy.kw_texts = write_texts(toy.root / "GPT_keywords.txt", KEYWORD_LINES + [("A_NEW", "{meadow, grass}")])
        run_predict(predict, *toy.argv())
        assert "A_NEW" not in read_predictions(toy.root / "out").index


# ============================================================================= training options
class TestTrainingOptions:
    def test_max_per_study_caps_the_training_samples(self, predict, toy):
        # one sample per study: the one with the smallest common.stable_key(22, sample_id), here M1, M4, M7.
        # M7 has no feature label, so the feature slot keeps only 2 labels (woodland, meadow)
        run_predict(predict, *toy.argv("out", "linear", "--max_per_study", 1))
        model = np.load(toy.root / "out" / "model.npz")
        assert int(model["n_samples"]) == 3
        table = read_predictions(toy.root / "out")
        assert dict(table["biome"]) == {s: labels[0] for s, labels in EXPECTED.items()}

    def test_max_per_study_0_means_no_cap(self, predict, toy):
        run_predict(predict, *toy.argv("out", "linear", "--max_per_study", 0))
        assert int(np.load(toy.root / "out" / "model.npz")["n_samples"]) == 7

    def test_a_sample_without_any_label_is_not_a_training_sample(self, predict, tmp_path):
        toy = Toy(tmp_path, training=TRAINING + [("M8", "STUDY_OCEAN", ("", "", ""), E2)])
        run_predict(predict, *toy.argv())
        assert int(np.load(tmp_path / "out" / "model.npz")["n_samples"]) == 7


# ============================================================================= refused runs
class TestRefusedRuns:
    """Situations in which the script stops with a message instead of writing inconsistent output."""

    def test_another_method_in_the_same_output_dir(self, predict, toy):
        run_predict(predict, *toy.argv("out", "linear"))
        with pytest.raises(SystemExit, match="holds a linear model, not prototype"):
            run_predict(predict, *toy.argv("out", "prototype"))

    def test_adding_calibration_to_an_output_dir_made_without(self, predict, toy):
        run_predict(predict, *toy.argv("out", "linear"))
        with pytest.raises(SystemExit, match="has chunks written with"):
            run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration()))

    def test_prototype_without_term_vectors(self, predict, toy):
        argv = toy.argv("out", "prototype")
        argv = argv[:argv.index("--term_vectors")]
        with pytest.raises(SystemExit, match="needs --term_vectors"):
            run_predict(predict, *argv)

    def test_calibration_with_knn_study(self, predict, toy):
        with pytest.raises(SystemExit, match="--calibration needs --method linear or prototype"):
            run_predict(predict, *toy.argv("out", "knn_study", "--calibration", toy.calibration(method="knn_study")))

    def test_calibration_of_another_method(self, predict, toy):
        with pytest.raises(SystemExit, match="has no linear calibration"):
            run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration(method="prototype")))

    def test_calibration_fitted_with_another_seed(self, predict, toy):
        with pytest.raises(SystemExit, match=r"other settings: seed=7 \(here 22\)"):
            run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration(seed=7)))

    def test_calibration_fitted_on_another_number_of_samples(self, predict, toy):
        with pytest.raises(SystemExit, match="not the same training set"):
            run_predict(predict, *toy.argv("out", "linear", "--calibration", toy.calibration(n_samples=99)))


# ============================================================================= small vocabularies, large k
class TestSmallVocabulariesAndLargeK:
    """Cases that real Metalog data does not hit, but small or filtered training sets can. Before
    2026-10-19 the first three crashed and the last gave wrong answers (see the mutation tests)."""

    @staticmethod
    def training_with_material(material_of_study):
        """TRAINING with the material label of each study replaced."""
        return [(s, st, (labels[0], labels[1], material_of_study[st]), d) for s, st, labels, d in TRAINING]

    @pytest.mark.parametrize("method", ["linear", "prototype", "knn_study"])
    def test_a_slot_with_two_labels(self, predict, tmp_path, method):
        # grassland samples get 'leaf litter' too: material has only 2 labels (leaf litter, sea water).
        # With --method linear, scikit-learn keeps a single score for two classes (ridge_class_scores).
        toy = Toy(tmp_path, training=self.training_with_material(
            {"STUDY_FOREST": "ENVO_LITTER", "STUDY_GRASS": "ENVO_LITTER", "STUDY_OCEAN": "ENVO_SEAWATER"}))
        run_predict(predict, *toy.argv("out", method))
        material = read_predictions(tmp_path / "out")["material"]
        assert dict(material) == {"A_FOREST": "ENVO_LITTER", "A_GRASS": "ENVO_LITTER", "A_OCEAN": "ENVO_SEAWATER",
                                  "A_FOREST_NO_SB": "ENVO_LITTER", "A_GRASS_SB_NA": "ENVO_LITTER",
                                  "A_OCEAN_SB_UNKNOWN": "ENVO_SEAWATER"}

    @pytest.mark.parametrize("method", ["linear", "prototype", "knn_study"])
    def test_a_slot_with_one_label(self, predict, tmp_path, method):
        # every training sample has material 'soil': every atlas sample gets soil; for linear and
        # prototype the margin over the second best label is 0 (there is none)
        toy = Toy(tmp_path, training=self.training_with_material(
            {"STUDY_FOREST": "ENVO_SOIL", "STUDY_GRASS": "ENVO_SOIL", "STUDY_OCEAN": "ENVO_SOIL"}))
        run_predict(predict, *toy.argv("out", method))
        table = read_predictions(tmp_path / "out")
        assert set(table["material"]) == {"ENVO_SOIL"}
        assert set(table["material_confidence"]) == ({"1.0"} if method == "knn_study" else {"0.0"})
        assert answers(table.assign(material="")) == {s: (b, f, "") for s, (b, f, _) in EXPECTED.items()}

    def test_knn_study_with_k_larger_than_the_training_set(self, predict, toy):
        # the default --knn_study_k 50 with 7 training samples: every training sample is a neighbour,
        # each study has one vote (1/3 each), and the tie goes to the closest neighbour's label
        argv = toy.argv("out", "knn_study")
        argv[argv.index("--knn_study_k") + 1] = 50
        run_predict(predict, *argv)
        table = read_predictions(toy.root / "out")
        assert answers(table) == EXPECTED
        assert set(table["biome_confidence"]) == {"0.3333"}

    def test_knn_study_neighbours_are_only_labelled_samples(self, predict, tmp_path):
        # only M1 (forest) and M5 (ocean) have a material label; k = 2: for an ocean sample the two
        # neighbours are M5 and nothing else (not an unlabelled ocean sample, which would vote for "")
        training = [(s, st, (labels[0], labels[1], labels[2] if s in ("M1", "M5") else ""), d)
                    for s, st, labels, d in TRAINING]
        toy = Toy(tmp_path, training=training)
        run_predict(predict, *toy.argv("out", "knn_study"))
        material = read_predictions(tmp_path / "out")["material"]
        assert material["A_OCEAN"] == "ENVO_SEAWATER" and material["A_FOREST"] == "ENVO_LITTER"
        assert "" not in set(material)
