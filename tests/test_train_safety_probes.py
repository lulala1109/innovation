"""CPU-only tests for leakage-free dual safety probe training."""

from __future__ import annotations

import builtins
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.batch_safety_attack import _default_probe_loader
from experiments.train_safety_probes import (
    load_training_payload,
    split_pair_groups,
    train_probe_checkpoint,
    train_safety_probes,
    validate_training_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _training_payload(groups: int = 6):
    harmfulness = []
    refusal = []
    pair_ids = []
    layer_two = []
    layer_five = []
    for group in range(groups):
        # Every pair contributes both classes. Thus any valid group split keeps
        # both probe targets trainable without leaking a pair across partitions.
        for harmful_label in (0.0, 1.0):
            refusal_label = 1.0 - harmful_label
            harmfulness.append(harmful_label)
            refusal.append(refusal_label)
            pair_ids.append(f"pair-{group}")
            harmful_signal = -2.0 if harmful_label == 0 else 2.0
            refusal_signal = -2.0 if refusal_label == 0 else 2.0
            layer_two.append([harmful_signal, refusal_signal])
            layer_five.append(
                [harmful_signal, refusal_signal, (group - groups / 2) * 0.01]
            )
    return {
        "hidden_states": {
            2: torch.tensor(layer_two, dtype=torch.float32),
            5: torch.tensor(layer_five, dtype=torch.float32),
        },
        "harmfulness_labels": torch.tensor(harmfulness),
        "refusal_labels": torch.tensor(refusal),
        "pair_ids": pair_ids,
    }


class TrainSafetyProbeTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_has_no_pair_leakage(self):
        pair_ids = ["p0", "p0", "p1", "p2", "p2", "p3"]
        first = split_pair_groups(
            pair_ids, validation_fraction=0.4, seed=17
        )
        second = split_pair_groups(
            pair_ids, validation_fraction=0.4, seed=17
        )
        self.assertEqual(first, second)
        self.assertTrue(set(first.train_pair_ids).isdisjoint(first.validation_pair_ids))
        train_rows = {pair_ids[index] for index in first.train_indices}
        validation_rows = {pair_ids[index] for index in first.validation_indices}
        self.assertEqual(train_rows, set(first.train_pair_ids))
        self.assertEqual(validation_rows, set(first.validation_pair_ids))

    def test_checkpoint_is_batch_loader_compatible_and_states_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "states.pt"
            output = root / "probes.pt"
            torch.save(_training_payload(), source)

            checkpoint = train_probe_checkpoint(
                source,
                output,
                validation_fraction=0.34,
                seed=9,
                epochs=60,
                learning_rate=0.05,
            )
            scorer = _default_probe_loader(output)

        self.assertEqual(checkpoint["hidden_sizes"], {2: 2, 5: 3})
        self.assertEqual(
            set(checkpoint["metrics"]), {"harmfulness", "refusal"}
        )
        self.assertNotIn("combined", checkpoint["metrics"])
        optimization = checkpoint["metadata"]["optimization"]
        self.assertIn("harmfulness", optimization)
        self.assertIn("refusal", optimization)
        self.assertNotIn("combined", optimization)
        self.assertIsNot(
            scorer.harmfulness_probe.probe_for(2),
            scorer.refusal_probe.probe_for(2),
        )
        for direction in ("harmfulness", "refusal"):
            for partition in ("train", "validation"):
                for layer_metrics in checkpoint["metrics"][direction][partition].values():
                    self.assertGreaterEqual(layer_metrics["accuracy"], 0.99)

    def test_fixed_seed_reproduces_all_probe_parameters(self):
        options = dict(
            validation_fraction=0.34,
            seed=123,
            epochs=8,
            learning_rate=0.03,
        )
        first = train_safety_probes(_training_payload(), **options)
        second = train_safety_probes(_training_payload(), **options)
        self.assertEqual(first["metadata"]["split"], second["metadata"]["split"])
        for name, first_value in first["state_dict"].items():
            self.assertTrue(torch.equal(first_value, second["state_dict"][name]), name)

    def test_shape_and_sample_count_mismatches_are_rejected(self):
        payload = _training_payload()
        payload["hidden_states"][2] = torch.zeros(3, 2, 1)
        with self.assertRaisesRegex(ValueError, "shape \[N, D\]"):
            validate_training_payload(payload)

        payload = _training_payload()
        payload["pair_ids"] = payload["pair_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "pair_ids must have length"):
            validate_training_payload(payload)

        payload = _training_payload()
        payload["hidden_states"][5] = payload["hidden_states"][5][:-1]
        with self.assertRaisesRegex(ValueError, "same sample count"):
            validate_training_payload(payload)

    def test_invalid_labels_and_nonfinite_states_are_rejected(self):
        payload = _training_payload()
        payload["harmfulness_labels"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary values"):
            validate_training_payload(payload)

        payload = _training_payload()
        payload["refusal_labels"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_training_payload(payload)

        payload = _training_payload()
        payload["hidden_states"][2][0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_training_payload(payload)

    def test_group_validation_rejects_unsplittable_or_bad_ids(self):
        with self.assertRaisesRegex(ValueError, "at least two unique"):
            split_pair_groups(["only", "only"])
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            split_pair_groups(["one", " "])
        with self.assertRaisesRegex(ValueError, "within \(0, 1\)"):
            split_pair_groups(["one", "two"], validation_fraction=1.0)

    def test_loader_requests_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.pt"
            torch.save(_training_payload(), path)
            real_load = torch.load
            with mock.patch.object(torch, "load", wraps=real_load) as loader:
                loaded = load_training_payload(path)
        self.assertIn("hidden_states", loaded)
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")

    def test_cli_help_does_not_import_torch(self):
        source = textwrap.dedent(
            """
            import builtins
            import runpy
            import sys

            real_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError("--help imported torch")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import
            sys.argv = ["train_safety_probes.py", "--help"]
            runpy.run_module("experiments.train_safety_probes", run_name="__main__")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--validation-fraction", completed.stdout)


if __name__ == "__main__":
    unittest.main()
