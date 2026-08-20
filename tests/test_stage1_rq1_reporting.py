"""Tests for the Stage-1 RQ1 Markdown/figure reporting boundary."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reporting.generate_stage1_rq1_report import (
    OptionalRQ1ReportingDependencyError,
    generate_stage1_rq1_report,
)


class Stage1RQ1ReportingTests(unittest.TestCase):
    def _analysis(self):
        cell_rows = []
        phase_rows = []
        for metric in ("H_probe", "R_probe"):
            for layer in (1, 2):
                for step, progress in enumerate((0.0, 0.5, 1.0)):
                    cell_rows.append(
                        {
                            "metric": metric,
                            "layer": layer,
                            "step": step,
                            "progress": progress,
                            "mean": 0.2 + 0.1 * layer + 0.05 * step,
                            "delta_mean": 0.05 * step,
                        }
                    )
                for phase, mean in (("early", 0.3), ("middle", 0.4), ("late", 0.5)):
                    phase_rows.append(
                        {
                            "metric": metric,
                            "layer": layer,
                            "phase": phase,
                            "mean": mean,
                            "ci_low": mean - 0.05,
                            "ci_high": mean + 0.05,
                        }
                    )
        return {
            "format": "stage1-rq1-analysis",
            "version": 1,
            "axes": {
                "case_ids": ["c1", "c2"],
                "layers": [1, 2],
                "steps": [0, 1, 2],
            },
            "cell_statistics": cell_rows,
            "phase_profiles": phase_rows,
            "mixed_effects": {
                metric: {
                    "layer_effect": {"likelihood_ratio": 2.0, "p_value": 0.1},
                    "attack_progress_effect": {"likelihood_ratio": 3.0, "p_value": 0.05},
                    "layer_by_progress_interaction": {
                        "likelihood_ratio": 4.0,
                        "p_value": 0.02,
                    },
                }
                for metric in ("H_probe", "R_probe")
            }
            | {
                "H_vs_R": {
                    "state_by_progress": {"likelihood_ratio": 5.0, "p_value": 0.01},
                    "state_by_layer_by_progress": {
                        "likelihood_ratio": 6.0,
                        "p_value": 0.005,
                    },
                }
            },
        }

    def test_markdown_report_works_without_plot_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            output = generate_stage1_rq1_report(
                self._analysis(), directory, make_plots=False
            )
            report = Path(output["report"])
            self.assertTrue(report.is_file())
            text = report.read_text(encoding="utf-8")
            self.assertIn("RQ1-a", text)
            self.assertIn("RQ1-b", text)
            self.assertIn("RQ1-c", text)
            self.assertIn("not used as a filter", text)
            self.assertIn("not asserted", text)
            self.assertNotIn("remains 80 train", text)
            self.assertEqual(output["figures"], {})

    def test_report_derives_verified_split_and_hides_invalid_lrt(self):
        analysis = self._analysis()
        analysis["axes"]["pair_ids"] = [f"held-{index}" for index in range(20)]
        analysis["axes"]["case_ids"] = [f"case-{index}" for index in range(20)]
        analysis["metadata"] = {
            "source_metadata": {
                "probe_training_num_pairs": 80,
                "probe_training_stage1_provenance_verified": True,
                "probe_training_measurement_split": "measurement_train",
                "probe_training_stage1_role": "probe_candidate",
                "trajectory_num_pairs": 20,
                "trajectory_measurement_split": "measurement_val",
                "trajectory_stage1_role": "trajectory_candidate",
            }
        }
        analysis["mixed_effects"]["H_probe"]["layer_effect"] = {
            "status": "invalid",
            "reason": "full model did not converge",
            "likelihood_ratio": None,
            "p_value": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = generate_stage1_rq1_report(
                analysis, directory, make_plots=False
            )
            text = Path(output["report"]).read_text(encoding="utf-8")
        self.assertIn("80 probe-training pairs / 20 held-out", text)
        self.assertIn("matches the planned 80/20 split", text)
        self.assertIn("not reported (full model did not converge)", text)

    def test_requested_plots_raise_clear_optional_dependency_error(self):
        module = importlib.import_module("reporting.generate_stage1_rq1_report")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                module.importlib,
                "import_module",
                side_effect=ImportError("missing matplotlib"),
            ):
                with self.assertRaises(OptionalRQ1ReportingDependencyError):
                    generate_stage1_rq1_report(
                        self._analysis(), directory, make_plots=True
                    )


if __name__ == "__main__":
    unittest.main()
