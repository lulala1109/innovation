"""Offline tests for waveform and optional perceptual metrics."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.perceptual import (
    OptionalMetricDependencyError,
    compute_perceptual_metrics,
    compute_pesq,
    compute_perturbation_metrics,
    compute_stoi,
    perturbation_l2,
    perturbation_linf,
    perturbation_rms,
    signal_to_noise_ratio_db,
)


class PerceptualMetricTests(unittest.TestCase):
    def test_known_torch_perturbation_metrics(self):
        reference = torch.tensor([1.0, -1.0])
        degraded = torch.tensor([1.5, -0.5])

        self.assertAlmostEqual(perturbation_linf(reference, degraded), 0.5)
        self.assertAlmostEqual(perturbation_l2(reference, degraded), math.sqrt(0.5))
        self.assertAlmostEqual(perturbation_rms(reference, degraded), 0.5)
        self.assertAlmostEqual(
            signal_to_noise_ratio_db(reference, degraded),
            10.0 * math.log10(4.0),
        )

        metrics = compute_perturbation_metrics(reference, degraded)
        self.assertEqual(
            set(metrics),
            {
                "perturbation_linf",
                "perturbation_l2",
                "perturbation_rms",
                "snr_db",
            },
        )
        self.assertAlmostEqual(metrics["snr_db"], 10.0 * math.log10(4.0))

    def test_numpy_inputs_and_zero_noise_edges(self):
        reference = np.array([[0.25, -0.5, 0.75]], dtype=np.float32)
        identical = reference.copy()
        metrics = compute_perturbation_metrics(reference, identical)
        self.assertEqual(metrics["perturbation_linf"], 0.0)
        self.assertEqual(metrics["perturbation_l2"], 0.0)
        self.assertEqual(metrics["perturbation_rms"], 0.0)
        self.assertEqual(metrics["snr_db"], math.inf)

        silence = np.zeros(3, dtype=np.float32)
        noisy_silence = np.ones(3, dtype=np.float32)
        self.assertEqual(
            signal_to_noise_ratio_db(silence, noisy_silence), -math.inf
        )
        self.assertEqual(signal_to_noise_ratio_db(silence, silence), math.inf)

    def test_shape_empty_and_nonfinite_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_perturbation_metrics(torch.zeros(4), torch.zeros(5))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compute_perturbation_metrics(np.array([]), np.array([]))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_perturbation_metrics(
                np.array([0.0, np.nan]), np.array([0.0, 1.0])
            )

    def test_base_metrics_never_import_optional_dependencies(self):
        reference = np.zeros(16, dtype=np.float32)
        with patch("evaluation.perceptual.import_module") as importer:
            metrics = compute_perceptual_metrics(reference, reference)
        importer.assert_not_called()
        self.assertNotIn("pesq", metrics)
        self.assertNotIn("stoi", metrics)

    def test_missing_pesq_and_stoi_raise_clear_dependency_errors(self):
        reference = np.zeros(160, dtype=np.float32)
        with patch(
            "evaluation.perceptual.import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(OptionalMetricDependencyError, "'pesq'"):
                compute_pesq(reference, reference, 16_000)
            with self.assertRaisesRegex(OptionalMetricDependencyError, "'pystoi'"):
                compute_stoi(reference, reference, 16_000)

    def test_optional_adapters_call_real_backend_contracts(self):
        calls = {}

        def fake_pesq(sample_rate, reference, degraded, mode):
            calls["pesq"] = (sample_rate, reference.shape, degraded.shape, mode)
            return 3.75

        def fake_stoi(reference, degraded, sample_rate, *, extended):
            calls["stoi"] = (
                reference.shape,
                degraded.shape,
                sample_rate,
                extended,
            )
            return 0.82

        def fake_import(name):
            if name == "pesq":
                return SimpleNamespace(pesq=fake_pesq)
            if name == "pystoi":
                return SimpleNamespace(stoi=fake_stoi)
            raise AssertionError(name)

        reference = torch.zeros(1, 160)
        degraded = torch.full((1, 160), 0.01)
        with patch("evaluation.perceptual.import_module", side_effect=fake_import):
            metrics = compute_perceptual_metrics(
                reference,
                degraded,
                sample_rate=16_000,
                include_pesq=True,
                include_stoi=True,
                extended_stoi=True,
            )
        self.assertEqual(metrics["pesq"], 3.75)
        self.assertEqual(metrics["stoi"], 0.82)
        self.assertEqual(calls["pesq"], (16_000, (160,), (160,), "wb"))
        self.assertEqual(calls["stoi"], ((160,), (160,), 16_000, True))

    def test_optional_metric_validation_does_not_guess_invalid_settings(self):
        reference = np.zeros(80, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            compute_perceptual_metrics(reference, reference, include_stoi=True)
        with self.assertRaisesRegex(ValueError, "only 8000 Hz or 16000 Hz"):
            compute_pesq(reference, reference, 44_100)
        with self.assertRaisesRegex(ValueError, "narrow-band"):
            compute_pesq(reference, reference, 8_000, mode="wb")


if __name__ == "__main__":
    unittest.main()
