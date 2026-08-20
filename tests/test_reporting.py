"""Offline report-table tests; matplotlib remains optional."""

import tempfile
import unittest
import importlib
from pathlib import Path
from unittest import mock

from reporting.generate_report import (
    OptionalReportingDependencyError,
    build_report_tables,
    generate_report,
    plot_safety_heatmap,
)


class ReportingTests(unittest.TestCase):
    def _inputs(self):
        analysis = {
            "layer_step_table": [
                {
                    "step": 0, "layer": 0, "refusal": 0.9,
                    "harmfulness": 0.8, "refusal_degradation": 0.0,
                    "layer_weight": 0.5, "is_bottleneck": True,
                },
                {
                    "step": 1, "layer": 0, "refusal": 0.4,
                    "harmfulness": 0.8, "refusal_degradation": 0.5,
                    "layer_weight": 0.8, "is_bottleneck": True,
                },
            ],
            "bottleneck_path": [
                {"step": 0, "layer": 0, "refusal_degradation": 0.0},
                {"step": 1, "layer": 0, "refusal_degradation": 0.5},
            ],
        }
        patching = {
            "baseline_scores": {"refusal": 0.1},
            "trials": [
                {
                    "condition": "critical", "layer": 0,
                    "scores": {"refusal": 0.7},
                    "delta": {"refusal": 0.6},
                }
            ],
        }
        statistics = {
            "baseline_method": "standard",
            "metric": "score",
            "comparisons": {
                "adaptive": {
                    "status": "ok", "n_pairs": 2, "n_observations": 4,
                    "mean_baseline": 0.2, "mean_candidate": 0.6,
                    "mean_raw_difference": 0.4, "mean_improvement": 0.4,
                    "bootstrap_ci": {
                        "low": 0.3, "high": 0.5, "confidence": 0.95,
                    },
                    "effect_size_paired_cohens_dz": 2.0,
                    "permutation_test": {"p_value": 0.25},
                }
            },
        }
        quality = {
            "cases": [
                {
                    "case_id": "c1", "pair_id": "p1",
                    "method": "adaptive", "seed": 1,
                    "attack_success": True,
                    "metrics": {"spr_db": 20.0, "stoi": 0.9},
                    "budget_verification": {"valid": True},
                }
            ]
        }
        return analysis, patching, statistics, quality

    def test_all_document_tables_generate_without_plot_dependencies(self):
        analysis, patching, statistics, quality = self._inputs()
        tables = build_report_tables(
            safety_analysis=analysis,
            patching=patching,
            method_statistics=statistics,
            audio_quality=quality,
        )
        self.assertEqual(
            set(tables),
            {
                "safety_heatmap", "bottleneck_path", "activation_patching",
                "method_comparison", "quality_tradeoff",
            },
        )
        self.assertEqual(
            tables["activation_patching"][0]["delta_refusal"], 0.6
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate_report(
                directory,
                safety_analysis=analysis,
                patching=patching,
                method_statistics=statistics,
                audio_quality=quality,
                make_plots=False,
            )
            self.assertEqual(len(manifest["tables"]), 5)
            self.assertEqual(manifest["figures"], {})
            self.assertTrue(Path(manifest["report"]).is_file())

    def test_requested_plot_has_clear_optional_dependency_error(self):
        analysis, _, _, _ = self._inputs()
        rows = analysis["layer_step_table"]
        report_module = importlib.import_module("reporting.generate_report")
        with mock.patch.object(
            report_module.importlib,
            "import_module",
            side_effect=ImportError("missing"),
        ):
            with self.assertRaisesRegex(
                OptionalReportingDependencyError, "matplotlib"
            ):
                plot_safety_heatmap(rows, "unused.png")

    def test_real_qwen_batch_and_behavior_evaluation_schemas_are_supported(self):
        patching_batch = {
            "format": "qwen-activation-patching-batch",
            "cases": [
                {
                    "pair_id": "p1",
                    "seed": 11,
                    "baseline_scores": {"refusal": 0.2},
                    "trials": [
                        {
                            "condition": "critical",
                            "layer": 7,
                            "response": {"scores": {"refusal": 0.8}},
                            "delta": {"refusal": 0.6},
                        }
                    ],
                },
                {"pair_id": "p2", "status": "failed", "message": "boom"},
            ],
        }
        quality = {
            "cases": [
                {
                    "case_id": "c1",
                    "pair_id": "p1",
                    "method": "adaptive",
                    "seed": 11,
                    "attack_success": False,
                    "metrics": {"spr_db": 22.0},
                    "budget_verification": {"valid": True},
                }
            ]
        }
        behavior = [
            {
                "pair_id": "p1",
                "method": "adaptive",
                "budget": {"seed": 11},
                "reported_attack_success": True,
                "strongreject": {"avg_score": 0.75, "runs": []},
            }
        ]

        tables = build_report_tables(
            patching=patching_batch,
            audio_quality=quality,
            behavior_cases=behavior,
        )
        patch_row = tables["activation_patching"][0]
        self.assertEqual(patch_row["pair_id"], "p1")
        self.assertEqual(patch_row["patched_refusal"], 0.8)
        quality_row = tables["quality_tradeoff"][0]
        self.assertTrue(quality_row["attack_success"])
        self.assertEqual(quality_row["behavior_score"], 0.75)


if __name__ == "__main__":
    unittest.main()
