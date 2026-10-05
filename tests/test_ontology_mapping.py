"""Fast regression tests for the ontology-mapping pipeline's reusable logic.

Run from the repository root with:
    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE = Path(__file__).parents[1] / "scripts" / "ontology_mapping"
sys.path.insert(0, str(PIPELINE))

common = importlib.import_module("common")
build_training = importlib.import_module("2_build_training_set")
clean_metalog = importlib.import_module("2b_clean_metalog")
predict_atlas = importlib.import_module("6_predict_atlas")
coverage = importlib.import_module("6b_coverage")


class SnapshotTests(unittest.TestCase):
    def test_latest_complete_snapshot_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for date in ("2025-01-01", "2026-01-01"):
                for domain in ("animal", "environmental", "human", "ocean"):
                    (root / f"{domain}_all_long_{date}.tsv.gz").touch()

            date, files = build_training.snapshot_files(str(root))

            self.assertEqual(date, "2026-01-01")
            self.assertTrue(all("2026-01-01" in p for p in files))

    def test_incomplete_requested_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "animal_all_long_2026-01-01.tsv.gz").touch()
            with self.assertRaises(SystemExit):
                build_training.snapshot_files(str(root), "2026-01-01")


class ResumeSafetyTests(unittest.TestCase):
    def test_changed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "run_settings.json"
            common.ensure_settings(str(manifest), {"method": "prototype"})
            common.ensure_settings(str(manifest), {"method": "prototype"})
            with self.assertRaises(SystemExit):
                common.ensure_settings(str(manifest), {"method": "linear"})
            self.assertEqual(json.loads(manifest.read_text()), {"method": "prototype"})

    def test_orphaned_cached_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.npz"
            artifact.touch()
            with self.assertRaises(SystemExit):
                common.ensure_settings(root / "run_settings.json", {"method": "prototype"}, [artifact])

    def test_prediction_weight_matches_evaluation_precision(self):
        block = np.ones((1, 2), dtype=np.float32)
        self.assertEqual((predict_atlas.BLOCK_WEIGHT * block).dtype, np.float64)


class NeighbourTests(unittest.TestCase):
    def test_knn_study_ignores_masked_unlabelled_rows_and_clamps_k(self):
        # Only columns 0 and 2 are labelled. A requested k larger than that must not allow the
        # masked columns to vote for the synthetic empty-label class (class 0).
        sim = np.array([[0.9, -np.inf, 0.8, -np.inf]])
        labels = np.array([1, 0, 2, 0])
        studies = np.array([0, 1, 2, 3])

        best, confidence = predict_atlas.knn_study_top2(sim, labels, studies, 3, k=50)

        np.testing.assert_array_equal(best, [1])
        np.testing.assert_allclose(confidence, [0.5])

    def test_top_k_mean_rejects_impossible_k(self):
        with self.assertRaises(ValueError):
            coverage.top_k_mean(np.ones((2, 1)), 2)


class LabelMapTests(unittest.TestCase):
    def test_invalid_label_map_target_fails_fast(self):
        samples = pd.DataFrame({
            "environment_biome": ["soil [ENVO:00001998]"],
            "environment_feature": [""],
            "environment_material": [""],
        })
        terms = pd.DataFrame({
            "term_id": ["ENVO_00001998"],
            "obsolete": ["False"],
            "parents": [""],
        })
        with self.assertRaises(SystemExit):
            clean_metalog.flag_labels(samples, terms, {("biome", "ENVO_00001998"): "ENVO_99999999"})


if __name__ == "__main__":
    unittest.main()
