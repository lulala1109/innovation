"""Offline tests for cross-sample Stage-1 RQ1 analysis."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.analyze_stage1_rq1 import (
    _lrt,
    analyze_stage1_rq1,
    phase_for_progress,
    validate_score_payload,
    write_rq1_outputs,
)


class Stage1RQ1AnalysisTests(unittest.TestCase):
    def _payload(self, *, cases: int = 6):
        layers = [3, 7]
        steps = torch.arange(4, dtype=torch.int64)
        progress = steps.float() / 3.0
        h = torch.empty(cases, len(layers), len(steps))
        r = torch.empty_like(h)
        for case in range(cases):
            offset = (case - cases / 2) * 0.01
            h[case, 0] = 0.30 + 0.10 * progress + offset
            h[case, 1] = 0.70 + 0.20 * progress - offset
            r[case, 0] = 0.85 - 0.15 * progress - offset
            r[case, 1] = 0.65 - 0.35 * progress + offset
        behavior = []
        for case in range(cases):
            behavior.append(
                [
                    {
                        "label_status": "ok",
                        "refusal_label": step < 2,
                        "compliance_label": step >= 3,
                    }
                    for step in range(4)
                ]
            )
        return {
            "format": "stage1-trajectory-scores",
            "version": 1,
            "case_ids": [f"case-{index}" for index in range(cases)],
            "pair_ids": [f"pair-{index}" for index in range(cases)],
            "layers": layers,
            "steps": steps,
            "scores": {
                "H_probe": h,
                "R_probe": r,
                "H_direction": torch.logit(h.clamp(0.01, 0.99)),
                "R_direction": torch.logit(r.clamp(0.01, 0.99)),
            },
            "attack_loss": torch.stack(
                [1.0 - progress + case * 0.01 for case in range(cases)]
            ),
            "behavior": behavior,
            "metadata": {
                "probe_version": 2,
                "directions_missing": False,
                "unverified_provenance_allowed": False,
                "probe_model_fingerprint_verified": True,
                "probe_source_payload_sha256_verified": True,
                "trajectory_measurement_split": "measurement_val",
                "trajectory_stage1_role": "trajectory_candidate",
                "trajectory_num_pairs": cases,
            },
        }

    def test_phase_boundaries_are_explicit(self):
        self.assertEqual(phase_for_progress(0.0), "early")
        self.assertEqual(phase_for_progress(1.0 / 3.0), "middle")
        self.assertEqual(phase_for_progress(2.0 / 3.0), "late")
        self.assertEqual(phase_for_progress(1.0), "late")

    def test_analysis_builds_complete_heatmap_delta_phase_and_events(self):
        result = analyze_stage1_rq1(
            self._payload(),
            bootstrap_replicates=30,
            seed=7,
            include_mixed_effects=False,
        )
        self.assertEqual(result["format"], "stage1-rq1-analysis")
        self.assertEqual(len(result["cell_statistics"]), 4 * 2 * 4)
        baseline = [
            row
            for row in result["cell_statistics"]
            if row["metric"] == "H_probe" and row["step"] == 0
        ]
        self.assertTrue(baseline)
        self.assertTrue(all(abs(row["delta_mean"]) < 1e-12 for row in baseline))
        self.assertEqual(
            {row["phase"] for row in result["phase_profiles"]},
            {"early", "middle", "late"},
        )
        self.assertTrue(
            all(row["first_non_refusal_step"] == 2 for row in result["behavior_events"])
        )
        self.assertTrue(
            all(row["first_compliance_step"] == 3 for row in result["behavior_events"])
        )
        self.assertTrue(result["event_aligned_statistics"])
        self.assertEqual(len(result["attack_loss_statistics"]), 4)
        self.assertEqual(len(result["loss_state_correlations"]), 4 * 2)
        reproducibility = result["profile_reproducibility"]
        self.assertTrue(
            all(
                row.get("loo_spearman_bootstrap_method")
                == "pair-profile-resample-recompute-loo"
                for row in reproducibility
                if row["correlation_count"]
            )
        )
        self.assertTrue(
            all(0.0 <= row["fdr_q_value"] <= 1.0 for row in result["layer_slopes"])
        )

    def test_mixed_effects_reports_layer_step_interaction_and_state_difference(self):
        result = analyze_stage1_rq1(
            self._payload(cases=5),
            bootstrap_replicates=10,
            seed=3,
            include_mixed_effects=True,
        )
        mixed = result["mixed_effects"]
        self.assertEqual(set(mixed), {"H_probe", "R_probe", "H_vs_R"})
        for metric in ("H_probe", "R_probe"):
            interaction = mixed[metric]["layer_by_progress_interaction"]
            self.assertIn(interaction["status"], {"ok", "invalid"})
            if interaction["status"] == "ok":
                self.assertGreaterEqual(interaction["likelihood_ratio"], 0.0)
                self.assertTrue(0.0 <= interaction["p_value"] <= 1.0)
            else:
                self.assertIsNone(interaction["likelihood_ratio"])
                self.assertIsNone(interaction["p_value"])
                self.assertTrue(interaction["reason"])
        self.assertIn("state_by_progress", mixed["H_vs_R"])
        self.assertIn("state_by_layer_by_progress", mixed["H_vs_R"])
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_rq1_outputs(result, directory)
            with Path(artifacts["mixed_effects"]).open(encoding="utf-8") as handle:
                serialized = json.load(handle)
        self.assertEqual(set(serialized), {"H_probe", "R_probe", "H_vs_R"})

    def test_rejects_sparse_steps_and_writes_atomic_tables(self):
        payload = self._payload()
        payload["steps"] = torch.tensor([0, 1, 3, 4])
        with self.assertRaisesRegex(ValueError, "complete 0..T"):
            validate_score_payload(payload)

        result = analyze_stage1_rq1(
            self._payload(),
            bootstrap_replicates=5,
            include_mixed_effects=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_rq1_outputs(result, directory)
            self.assertTrue(Path(artifacts["summary"]).is_file())
            self.assertTrue(Path(artifacts["cell_statistics"]).is_file())
            self.assertTrue(Path(artifacts["attack_loss_statistics"]).is_file())
            with Path(artifacts["summary"]).open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["format"], "stage1-rq1-analysis")
            with Path(artifacts["cell_statistics"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4 * 2 * 4)

    def test_legacy_direction_compatibility_artifact_is_explicitly_rejected(self):
        payload = self._payload()
        payload["metadata"]["directions_missing"] = True
        payload["scores"]["H_direction"].fill_(float("nan"))
        payload["scores"]["R_direction"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "v2 probe|legacy compatibility"):
            validate_score_payload(payload)

    def test_invalid_mixed_model_lrt_never_reports_a_p_value(self):
        invalid = _lrt(
            {
                "converged": False,
                "variance_parameters_at_boundary": False,
                "fixed_effect_count": 1,
                "log_likelihood": -10.0,
            },
            {
                "converged": True,
                "variance_parameters_at_boundary": False,
                "fixed_effect_count": 2,
                "log_likelihood": -9.0,
            },
        )
        self.assertEqual(invalid["status"], "invalid")
        self.assertIsNone(invalid["likelihood_ratio"])
        self.assertIsNone(invalid["p_value"])
        self.assertIn("did not converge", invalid["reason"])

    def test_empty_event_alignment_still_materializes_a_csv(self):
        payload = self._payload()
        for case in payload["behavior"]:
            for row in case:
                row.update(
                    {
                        "label_status": "unknown",
                        "refusal_label": None,
                        "compliance_label": None,
                    }
                )
        result = analyze_stage1_rq1(
            payload,
            bootstrap_replicates=5,
            weakening_threshold=100.0,
            include_mixed_effects=False,
        )
        self.assertEqual(result["event_aligned_statistics"], [])
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_rq1_outputs(result, directory)
            event_path = Path(artifacts["event_aligned_statistics"])
            self.assertTrue(event_path.is_file())
            with event_path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertIn("event", header)

    def test_single_layer_requires_explicitly_skipping_layer_mixed_models(self):
        payload = self._payload()
        payload["layers"] = payload["layers"][:1]
        for name in tuple(payload["scores"]):
            payload["scores"][name] = payload["scores"][name][:, :1, :]
        with self.assertRaisesRegex(ValueError, "at least two layers|skip-mixed"):
            analyze_stage1_rq1(payload, bootstrap_replicates=5)
        result = analyze_stage1_rq1(
            payload,
            bootstrap_replicates=5,
            include_mixed_effects=False,
        )
        self.assertIsNone(result["mixed_effects"])


if __name__ == "__main__":
    unittest.main()
