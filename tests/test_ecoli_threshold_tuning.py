import unittest

import numpy as np
import torch

from gpm.scripts.tune_ecoli_thresholds import (
    _extract_model_state_dict,
    build_report,
    build_threshold_grid,
    find_best_threshold,
)


class EcoliThresholdTuningTests(unittest.TestCase):
    def test_build_threshold_grid_includes_bounds(self):
        grid = build_threshold_grid(0.0, 1.0, 5)
        self.assertTrue(np.allclose(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0])))

    def test_extract_model_state_dict_from_payload_dict(self):
        payload = {
            "model": {"layer.weight": torch.ones(2, 2)},
            "epoch": 3,
            "step": 42,
        }
        state_dict, meta = _extract_model_state_dict(payload)
        self.assertIn("layer.weight", state_dict)
        self.assertEqual(meta["epoch"], 3)
        self.assertEqual(meta["step"], 42)

    def test_extract_model_state_dict_from_raw_state_dict(self):
        payload = {"layer.weight": torch.ones(2, 2)}
        state_dict, meta = _extract_model_state_dict(payload)
        self.assertIn("layer.weight", state_dict)
        self.assertEqual(meta, {})

    def test_find_best_threshold_improves_f1_over_baseline(self):
        logits = torch.tensor([-1.2, -0.4, -0.2, 0.1, 0.3, 1.5], dtype=torch.float32).reshape(-1, 1, 1)
        targets = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32).reshape(-1, 1, 1)
        grid = build_threshold_grid(0.1, 0.9, 81)

        baseline = find_best_threshold(
            logits=logits,
            targets=targets,
            thresholds=[0.5],
            objective="f1",
            baseline_threshold=0.5,
        )
        tuned = find_best_threshold(
            logits=logits,
            targets=targets,
            thresholds=grid,
            objective="f1",
            baseline_threshold=0.5,
        )

        self.assertGreaterEqual(float(tuned["f1"]), float(baseline["f1"]))
        self.assertNotAlmostEqual(float(tuned["threshold"]), 0.5, places=6)

    def test_build_report_contains_expected_keys(self):
        logits = torch.tensor([-1.2, -0.4, -0.2, 0.1, 0.3, 1.5], dtype=torch.float32).reshape(-1, 1, 1)
        targets = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32).reshape(-1, 1, 1)
        grid = build_threshold_grid(0.1, 0.9, 81)
        report = build_report(
            logits=logits,
            targets=targets,
            threshold_grid=grid,
            selected_objective="f1",
            baseline_threshold=0.5,
        )

        self.assertIn("baseline_threshold", report)
        self.assertIn("baseline_metrics", report)
        self.assertIn("best_threshold", report)
        self.assertIn("best_metrics", report)
        self.assertIn("best_by_objective", report)
        self.assertIn("n_validation_samples", report)
        self.assertIn("f1", report["best_by_objective"])
        self.assertIn("balanced_accuracy", report["best_by_objective"])
        self.assertIn("accuracy", report["best_by_objective"])

    def test_find_best_threshold_rejects_unknown_objective(self):
        logits = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1, 1)
        targets = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1, 1)
        with self.assertRaises(ValueError):
            find_best_threshold(
                logits=logits,
                targets=targets,
                thresholds=[0.5],
                objective="unknown_metric",
                baseline_threshold=0.5,
            )

    def test_build_report_includes_test_metrics_for_baseline_and_selected_threshold(self):
        logits = torch.tensor([-1.2, -0.4, -0.2, 0.1, 0.3, 1.5], dtype=torch.float32).reshape(-1, 1, 1)
        targets = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32).reshape(-1, 1, 1)
        test_logits = torch.tensor([-0.6, -0.2, 0.2, 0.8], dtype=torch.float32).reshape(-1, 1, 1)
        test_targets = torch.tensor([0, 1, 1, 1], dtype=torch.float32).reshape(-1, 1, 1)
        grid = build_threshold_grid(0.1, 0.9, 81)

        report = build_report(
            logits=logits,
            targets=targets,
            threshold_grid=grid,
            selected_objective="f1",
            baseline_threshold=0.5,
            test_logits=test_logits,
            test_targets=test_targets,
        )

        self.assertIn("test", report)
        test_section = report["test"]
        self.assertEqual(float(test_section["baseline_threshold"]), float(report["baseline_threshold"]))
        self.assertEqual(float(test_section["selected_threshold"]), float(report["best_threshold"]))
        self.assertIn("baseline_metrics", test_section)
        self.assertIn("selected_threshold_metrics", test_section)
        self.assertEqual(int(test_section["baseline_metrics"]["n_samples"]), int(test_targets.numel()))
        self.assertEqual(int(test_section["selected_threshold_metrics"]["n_samples"]), int(test_targets.numel()))
        self.assertAlmostEqual(
            float(test_section["baseline_metrics"]["bce_loss"]),
            float(test_section["selected_threshold_metrics"]["bce_loss"]),
            places=8,
        )

        required_metric_keys = {
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "n_samples",
            "n_positive",
            "n_negative",
            "bce_loss",
        }
        self.assertTrue(required_metric_keys.issubset(set(test_section["baseline_metrics"].keys())))
        self.assertTrue(required_metric_keys.issubset(set(test_section["selected_threshold_metrics"].keys())))

    def test_build_report_omits_test_section_when_test_logits_are_not_provided(self):
        logits = torch.tensor([-1.2, -0.4, -0.2, 0.1, 0.3, 1.5], dtype=torch.float32).reshape(-1, 1, 1)
        targets = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32).reshape(-1, 1, 1)
        grid = build_threshold_grid(0.1, 0.9, 81)

        report = build_report(
            logits=logits,
            targets=targets,
            threshold_grid=grid,
            selected_objective="f1",
            baseline_threshold=0.5,
        )
        self.assertNotIn("test", report)


if __name__ == "__main__":
    unittest.main()
