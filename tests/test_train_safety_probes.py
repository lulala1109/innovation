"""CPU-only tests for leakage-free dual safety probe training."""

from __future__ import annotations

import builtins
import hashlib
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
    build_parser,
    load_training_payload,
    split_pair_group_folds,
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
    def test_group_kfold_is_deterministic_complete_and_pair_safe(self):
        pair_ids = [f"p{index // 3}" for index in range(24)]
        first = split_pair_group_folds(pair_ids, folds=5, seed=91)
        second = split_pair_group_folds(pair_ids, folds=5, seed=91)
        self.assertEqual(first, second)
        validation_rows = []
        for fold in first:
            self.assertTrue(
                set(fold.train_pair_ids).isdisjoint(fold.validation_pair_ids)
            )
            validation_rows.extend(fold.validation_indices)
            for pair_id in set(pair_ids):
                assignments = {
                    index in fold.validation_indices
                    for index, value in enumerate(pair_ids)
                    if value == pair_id
                }
                self.assertEqual(len(assignments), 1)
        self.assertEqual(sorted(validation_rows), list(range(len(pair_ids))))

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

    def test_v2_group_oof_metrics_directions_and_final_probe(self):
        payload = _training_payload(groups=10)
        checkpoint = train_safety_probes(
            payload,
            cv_folds=5,
            seed=7,
            epochs=30,
            learning_rate=0.05,
            bootstrap_replicates=40,
        )
        self.assertEqual(checkpoint["version"], 2)
        self.assertEqual(
            checkpoint["metadata"]["split"]["mode"], "pair_group_kfold_oof"
        )
        assignments = checkpoint["oof_predictions"]["fold_assignments"]
        self.assertEqual(assignments.shape, (20,))
        self.assertEqual(set(assignments.tolist()), set(range(5)))
        for pair_id in set(payload["pair_ids"]):
            pair_folds = {
                int(assignments[index])
                for index, value in enumerate(payload["pair_ids"])
                if value == pair_id
            }
            self.assertEqual(len(pair_folds), 1)

        for direction in ("harmfulness", "refusal"):
            for layer in (2, 5):
                means = checkpoint["class_means"][direction][layer]
                expected = means["positive"] - means["negative"]
                self.assertTrue(
                    torch.allclose(checkpoint["directions"][direction][layer], expected)
                )
                probabilities = checkpoint["oof_predictions"]["probabilities"][
                    direction
                ][layer]
                projections = checkpoint["oof_predictions"][
                    "direction_projections"
                ][direction][layer]
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue(torch.isfinite(projections).all())
                layer_metrics = checkpoint["metrics"][direction]["oof"][str(layer)]
                for metric in ("auroc", "f1", "balanced_accuracy"):
                    self.assertGreaterEqual(layer_metrics[metric], 0.99)
                    interval = layer_metrics["bootstrap_95_ci"][metric]
                    self.assertEqual(interval["requested_replicates"], 40)
                    self.assertEqual(interval["method"], "pair_cluster_percentile")
                self.assertEqual(
                    layer_metrics["confusion_matrix"], [[10, 0], [0, 10]]
                )
                alignment = layer_metrics["probe_direction_alignment"]
                self.assertGreaterEqual(alignment["spearman"], 0.8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe-v2.pt"
            legacy_path = Path(directory) / "probe-v1.pt"
            unsupported_path = Path(directory) / "probe-v3.pt"
            torch.save(checkpoint, path)
            legacy = {
                "version": 1,
                "hidden_sizes": checkpoint["hidden_sizes"],
                "state_dict": checkpoint["state_dict"],
            }
            torch.save(legacy, legacy_path)
            unsupported = dict(legacy)
            unsupported["version"] = 3
            torch.save(unsupported, unsupported_path)
            scorer = _default_probe_loader(path)
            legacy_scorer = _default_probe_loader(legacy_path)
            with self.assertRaisesRegex(ValueError, "version must be 1 or 2"):
                _default_probe_loader(unsupported_path)
        self.assertFalse(scorer.training)
        self.assertFalse(legacy_scorer.training)

    def test_explicit_measurement_provenance_rejects_held_out_or_wrong_role(self):
        payload = _training_payload()
        payload["row_metadata"] = [
            {
                "measurement_split": "measurement_train",
                "stage1_role": "probe_candidate",
            }
            for _ in payload["pair_ids"]
        ]
        validate_training_payload(payload)

        payload["row_metadata"][3]["measurement_split"] = "measurement_val"
        with self.assertRaisesRegex(ValueError, "only measurement_train"):
            validate_training_payload(payload)
        payload["row_metadata"][3]["measurement_split"] = "measurement_train"
        payload["row_metadata"][3]["stage1_role"] = "trajectory_candidate"
        with self.assertRaisesRegex(ValueError, "only probe_candidate"):
            validate_training_payload(payload)

        # Legacy payloads may have row metadata without the new provenance keys.
        payload["row_metadata"] = [{} for _ in payload["pair_ids"]]
        validate_training_payload(payload)
        payload["row_metadata"][0]["measurement_split"] = "measurement_train"
        with self.assertRaisesRegex(ValueError, "present on every row"):
            validate_training_payload(payload)

    def test_source_payload_hash_and_model_provenance_are_preserved(self):
        payload = _training_payload()
        payload["metadata"] = {
            "pooling": "mean",
            "token_span": "audio",
            "layers": [2, 5],
            "model_provenance": {
                "source_model": "qwen-3b",
                "model_id": "local/model",
                "model_fingerprint": "a" * 64,
                "dtype": "bfloat16",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "payload.pt"
            output = Path(directory) / "probe.pt"
            torch.save(payload, source)
            expected_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            checkpoint = train_probe_checkpoint(
                source,
                output,
                cv_folds=3,
                epochs=4,
                bootstrap_replicates=5,
            )
        provenance = checkpoint["metadata"]["provenance"]
        self.assertEqual(provenance["source_payload_sha256"], expected_digest)
        self.assertEqual(provenance["source_model"], "qwen-3b")
        self.assertEqual(provenance["model_fingerprint"], "a" * 64)
        self.assertEqual(provenance["pooling"], "mean")
        self.assertEqual(provenance["token_span"], "audio")
        self.assertEqual(
            provenance["training_payload_metadata"], payload["metadata"]
        )

    def test_production_cli_defaults_to_five_group_folds(self):
        args = build_parser().parse_args(["--input", "states.pt", "--output", "p.pt"])
        self.assertEqual(args.cv_folds, 5)
        self.assertEqual(args.bootstrap_replicates, 1000)

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
