from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_core import ROOT, binary_metrics, features, infer, load_packaged_model, prepare_inputs, read_csv_bytes, reference_labels, replay_review


class InferenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_packaged_model()
        cls.sample = read_csv_bytes((ROOT / "demo/sample_inputs.csv").read_bytes())
        cls.expected = pd.read_csv(ROOT / "demo/expected_predictions.csv")

    def test_original_sample_predictions(self):
        output, audit = infer(self.sample, self.model)
        merged = output.merge(self.expected, on="source_id", validate="one_to_one")
        actual = merged.model_prediction.map({"benign": 0, "attack": 1}).to_numpy()
        np.testing.assert_array_equal(actual, merged.expected_prediction.to_numpy())
        np.testing.assert_allclose(merged.model_attack_score, merged.expected_attack_score, atol=1e-12, rtol=0)
        self.assertEqual(audit["rows_with_device_name_used_in_model_training"], 0)

    def test_feature_order_and_label_exclusion(self):
        frame = self.sample.loc[:, list(reversed(self.sample.columns))].copy()
        original, _ = infer(self.sample, self.model)
        frame["label1"] = "attack"
        np.testing.assert_array_equal(prepare_inputs(self.sample)[0], prepare_inputs(frame)[0])
        reordered, _ = infer(frame, self.model)
        np.testing.assert_array_equal(original.model_prediction, reordered.model_prediction)
        # Parallel tree-score summation can differ at machine precision.
        np.testing.assert_allclose(original.model_attack_score, reordered.model_attack_score, atol=1e-12, rtol=0)

    def test_all_zero_rows_retained_and_flagged(self):
        output, audit = infer(self.sample, self.model)
        self.assertEqual(len(output), len(self.sample))
        self.assertGreater(audit["all_zero_predictor_rows"], 0)
        self.assertTrue(output.loc[output.all_zero_predictors, "review_flag"].str.startswith("All-zero").all())

    def test_missing_field_blocked(self):
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            prepare_inputs(self.sample.drop(columns=features()[0]))

    def test_bad_feature_values_blocked(self):
        for value in [np.nan, np.inf, "not_a_number", 1e100]:
            with self.subTest(value=value):
                frame = self.sample.head(2).copy().astype({features()[0]: object})
                frame.loc[0, features()[0]] = value
                _, audit = prepare_inputs(frame)
                self.assertGreater(audit["invalid_rows_after_float32_conversion"], 0)
                with self.assertRaisesRegex(ValueError, "Prediction blocked"):
                    infer(frame, self.model)

    def test_duplicate_headers_and_empty_input_rejected(self):
        for data in [b"", b"x,x\n1,2\n", b"x,y\n"]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    read_csv_bytes(data)

    def test_absent_or_invalid_labels_do_not_create_false_metrics(self):
        for frame in [self.sample.drop(columns="label1"), self.sample.assign(label1="unresolved")]:
            output, audit = infer(frame, self.model)
            self.assertIsNone(audit["supplied_label_metrics"])
            self.assertEqual(len(output), len(frame))

    def test_metrics_reconcile_confusion_counts(self):
        metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual([metrics[k] for k in ["TN", "FP", "FN", "TP"]], [1, 1, 1, 1])
        self.assertEqual(metrics["accuracy_pct"], 50)
        self.assertEqual(metrics["attack_f1_pct"], 50)
        self.assertIsNone(binary_metrics(np.array([1]), np.array([1]))["false_positive_rate_pct"])

    def test_original_replay_evidence(self):
        record = json.loads((ROOT / "verification/inference_replay.json").read_text())
        self.assertEqual(record["held_out_predictions_checked"], 56443)
        self.assertEqual(record["held_out_prediction_mismatches"], 0)
        self.assertTrue(record["original_archive_digest_matches"])


class ReviewChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = pd.read_csv(ROOT / "results/training_review_candidates.csv.gz")
        cls.expected = pd.read_csv(ROOT / "results/review_budget_results.csv")

    def test_every_recorded_method_and_budget_reproduces(self):
        for row in self.expected.itertuples(index=False):
            with self.subTest(method=row.method, budget=row.budget_pct):
                queue, stats = replay_review(row.method, row.budget_pct, self.candidates)
                self.assertEqual(len(queue), row.reviewed_rows)
                self.assertEqual(queue.source_id.nunique(), row.reviewed_rows)
                self.assertEqual(stats["recovered_injected_flips"], row.recovered_injected_flips)
                self.assertAlmostEqual(stats["injected_flip_recovery_pct"], row.injected_flip_recovery_pct, places=10)
                self.assertNotIn("known_injected_flip", queue.columns)

    def test_flip_truth_does_not_enter_selection(self):
        altered = self.candidates.copy()
        altered["known_injected_flip"] = np.roll(altered.known_injected_flip.to_numpy(), 123)
        for method in self.expected.method.unique():
            left, _ = replay_review(method, 5, self.candidates)
            right, _ = replay_review(method, 5, altered)
            self.assertEqual(left.source_id.tolist(), right.source_id.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
