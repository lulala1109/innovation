"""Offline causal patching coordinator tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from experiments.activation_patching import (
    build_patch_plan,
    coordinate_activation_patching,
    main,
)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = nn.Identity()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(self.layer0(value)))


class ActivationPatchingExperimentTests(unittest.TestCase):
    def test_plan_is_reproducible_and_excludes_critical_layers(self) -> None:
        first = build_patch_plan(range(6), [2], random_control_count=2, seed=17)
        second = build_patch_plan(range(6), [2], random_control_count=2, seed=17)
        self.assertEqual(first, second)
        controls = [trial["layer"] for trial in first if trial["condition"] == "random_control"]
        self.assertNotIn(2, controls)
        self.assertEqual(len(controls), 2)

    def test_coordinator_patches_critical_and_explicit_control_layers(self) -> None:
        model = ToyModel()
        value = torch.tensor([[1.0]])
        result = coordinate_activation_patching(
            model=model,
            layer_modules={0: model.layer0, 1: model.layer1, 2: model.layer2},
            source_activations={0: torch.tensor([[5.0]]), 1: torch.tensor([[7.0]])},
            forward_fn=lambda: model(value),
            score_fn=lambda output: {"refusal": output.mean(), "asr": -output.mean()},
            critical_layers=[0],
            random_control_count=1,
            random_layers=[1],
            seed=9,
        )

        self.assertEqual(result["baseline_scores"], {"refusal": 1.0, "asr": -1.0})
        self.assertEqual(result["trials"][0]["scores"]["refusal"], 5.0)
        self.assertEqual(result["trials"][1]["scores"]["refusal"], 7.0)
        self.assertAlmostEqual(
            result["aggregates"]["critical_minus_random"]["refusal"], -2.0
        )
        # Context exit must remove hooks; an ordinary forward is unchanged.
        self.assertEqual(float(model(value).item()), 1.0)

    def test_cli_accepts_injected_factory_and_writes_output(self) -> None:
        model = ToyModel()
        value = torch.tensor([[1.0]])

        def factory(_args):
            return {
                "model": model,
                "layer_modules": {0: model.layer0, 1: model.layer1},
                "source_activations": {0: torch.tensor([[3.0]]), 1: torch.tensor([[4.0]])},
                "forward_fn": lambda: model(value),
                "score_fn": lambda output: float(output.item()),
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "patching.json"
            exit_code = main(
                [
                    "--critical-layers",
                    "0",
                    "--random-layers",
                    "1",
                    "--random-control-count",
                    "1",
                    "--output",
                    str(output),
                ],
                experiment_factory=factory,
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.is_file())

    def test_explicit_controls_must_match_requested_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal"):
            build_patch_plan(
                range(4), [0], random_control_count=2, random_layers=[1], seed=1
            )

    def test_random_control_count_must_be_an_integer_not_bool(self) -> None:
        for invalid in (True, False, 1.5, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    build_patch_plan(
                        range(4), [0], random_control_count=invalid, seed=1
                    )


if __name__ == "__main__":
    unittest.main()
