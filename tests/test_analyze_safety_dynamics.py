"""Offline trajectory-analysis regression tests."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from core.artifacts import TrajectoryArtifactStore
from experiments.analyze_safety_dynamics import analyze_trajectory, write_analysis


class SafetyDynamicsAnalysisTests(unittest.TestCase):
    def _store(self, root: Path, refusals: list[torch.Tensor]) -> None:
        store = TrajectoryArtifactStore(root)
        harmfulness = [0.8, 0.7, 0.6]
        weights = [0.2, 0.3, 0.5]
        for step, refusal in enumerate(refusals):
            store.save_checkpoint(
                step,
                {
                    "refusal": refusal,
                    "harmfulness": torch.tensor(harmfulness),
                    "layer_weights": torch.tensor(weights),
                },
            )

    def test_dynamic_bottleneck_path_and_json_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            self._store(
                root,
                [
                    torch.tensor([1.0, 1.0, 1.0]),
                    torch.tensor([0.2, 0.8, 0.9]),
                    torch.tensor([0.8, 0.1, 0.7]),
                ],
            )
            result = analyze_trajectory(root, require_all_states=True)

            self.assertEqual(
                [entry["layer"] for entry in result["bottleneck_path"]],
                [0, 0, 1],
            )
            self.assertEqual(result["go_no_go"]["decision"], "go_layer_adaptive")
            self.assertEqual(result["go_no_go"]["bottleneck_switches"], 1)
            self.assertEqual(len(result["layer_step_table"]), 9)
            self.assertEqual(result["reference_refusal_source"], "earliest_checkpoint_step_0")

            json_path = Path(directory) / "analysis.json"
            csv_path = Path(directory) / "analysis.csv"
            write_analysis(result, json_path=json_path, csv_path=csv_path)
            with json_path.open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["go_no_go"]["decision"], "go_layer_adaptive")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)

    def test_stable_path_degrades_to_critical_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            self._store(
                root,
                [
                    torch.tensor([1.0, 1.0, 1.0]),
                    torch.tensor([0.1, 0.8, 0.9]),
                    torch.tensor([0.0, 0.7, 0.8]),
                ],
            )
            result = analyze_trajectory(root)
            decision = result["go_no_go"]
            self.assertEqual(decision["decision"], "no_go_critical_window")
            self.assertFalse(decision["dynamic_migration"])
            self.assertEqual(decision["critical_window"]["layers"], [0])
            self.assertEqual(decision["critical_window"]["start_layer"], 0)

    def test_refusal_is_required_and_other_states_can_be_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            store = TrajectoryArtifactStore(root)
            store.save_checkpoint(0, {"refusal": torch.ones(2)})
            permissive = analyze_trajectory(root)
            self.assertIsNone(permissive["layer_step_table"][0]["harmfulness"])
            with self.assertRaisesRegex(ValueError, "harmfulness and layer_weights"):
                analyze_trajectory(root, require_all_states=True)

    def test_safety_gaps_recover_reference_for_random_initialized_step_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            store = TrajectoryArtifactStore(root)
            reference = torch.tensor([0.9, 0.8])
            refusals = (
                torch.tensor([0.7, 0.6]),
                torch.tensor([0.4, 0.7]),
            )
            for step, refusal in enumerate(refusals):
                gaps = reference - refusal
                store.save_checkpoint(
                    step,
                    {"refusal": refusal},
                    metadata={
                        "safety_gaps": {
                            str(layer): float(gap)
                            for layer, gap in enumerate(gaps)
                        }
                    },
                )

            result = analyze_trajectory(root)

            self.assertEqual(
                result["reference_refusal_source"],
                "safety_gaps_plus_refusal",
            )
            step_zero = result["layer_step_table"][:2]
            self.assertEqual(
                [round(row["reference_refusal"], 6) for row in step_zero],
                [0.9, 0.8],
            )
            self.assertEqual(
                [round(row["refusal_degradation"], 6) for row in step_zero],
                [0.2, 0.2],
            )

    def test_inconsistent_safety_gap_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            store = TrajectoryArtifactStore(root)
            store.save_checkpoint(
                0,
                {"refusal": torch.tensor([0.7])},
                metadata={"safety_gaps": {"0": 0.2}},
            )
            store.save_checkpoint(
                1,
                {"refusal": torch.tensor([0.6])},
                metadata={"safety_gaps": {"0": 0.4}},
            )
            with self.assertRaisesRegex(ValueError, "derived reference_refusal"):
                analyze_trajectory(root)

    def test_explicit_and_gap_derived_references_must_agree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "case"
            store = TrajectoryArtifactStore(root)
            store.save_checkpoint(
                0,
                {"refusal": torch.tensor([0.7])},
                metadata={
                    "reference_refusal": {"0": 0.9},
                    "safety_gaps": {"0": 0.3},
                },
            )
            with self.assertRaisesRegex(
                ValueError, "explicit and safety_gaps-derived"
            ):
                analyze_trajectory(root)


if __name__ == "__main__":
    unittest.main()
