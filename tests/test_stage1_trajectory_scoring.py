"""Offline tests for the Stage-1 hidden-state scoring bridge."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from core.safety_state import DualSafetyStateScorer
from experiments.score_stage1_trajectories import (
    DELTA_KEYS,
    SCORE_KEYS,
    Stage1ScoringError,
    score_stage1_trajectories,
    sha256_file,
    validate_score_payload,
)


MODEL_PROVENANCE = {
    "model_name": "toy",
    "model_id": "/models/toy",
    "dtype": "float32",
}
MODEL_FINGERPRINT = hashlib.sha256(
    json.dumps(
        MODEL_PROVENANCE,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
MODEL_PROVENANCE["model_fingerprint"] = MODEL_FINGERPRINT
SOURCE_MANIFEST_SHA256 = hashlib.sha256(b"source-manifest").hexdigest()
BEHAVIOR_SIDECARS = [
    {"path": "/immutable/behavior.jsonl", "sha256": hashlib.sha256(b"behavior").hexdigest()}
]
REPLAY_CONFIG = {
    "manifest_sha256": SOURCE_MANIFEST_SHA256,
    "behavior_sidecars": BEHAVIOR_SIDECARS,
    "model_provenance": MODEL_PROVENANCE,
    "model_fingerprint": MODEL_FINGERPRINT,
    "device": "cpu",
    "dtype": "float32",
    "layers": [2, 5],
    "token_span": "audio",
    "pooling": "mean",
    "sequence_has_embedding": False,
}
REPLAY_FINGERPRINT = hashlib.sha256(
    json.dumps(
        REPLAY_CONFIG,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _probe_checkpoint(path: Path, *, version: int = 2) -> None:
    hidden_sizes = OrderedDict([(2, 2), (5, 3)])
    scorer = DualSafetyStateScorer(hidden_size=hidden_sizes, trainable=False)
    with torch.no_grad():
        # Learned probe weights deliberately differ from the independent mean
        # directions below; this catches accidental reuse of probe directions.
        scorer.harmfulness_probe.probe_for(2).linear.weight.copy_(
            torch.tensor([[0.25, 0.75]])
        )
        scorer.refusal_probe.probe_for(2).linear.weight.copy_(
            torch.tensor([[-0.5, 0.5]])
        )
        scorer.harmfulness_probe.probe_for(5).linear.weight.copy_(
            torch.tensor([[0.25, 0.75, 0.5]])
        )
        scorer.refusal_probe.probe_for(5).linear.weight.copy_(
            torch.tensor([[-0.5, 0.5, -0.25]])
        )
        for parameter in (
            scorer.harmfulness_probe.probe_for(2).linear.bias,
            scorer.refusal_probe.probe_for(2).linear.bias,
            scorer.harmfulness_probe.probe_for(5).linear.bias,
            scorer.refusal_probe.probe_for(5).linear.bias,
        ):
            parameter.zero_()
    checkpoint = {
        "format": "dual-safety-state-layerwise-linear-probes",
        "version": version,
        "hidden_sizes": hidden_sizes,
        "state_dict": OrderedDict(
            (name, value.detach().cpu()) for name, value in scorer.state_dict().items()
        ),
        "metadata": {
            "num_samples": 6,
            "num_pairs": 2,
            "measurement_split": "measurement_train",
            "stage1_role": "probe_candidate",
            "stage1_provenance_verified": True,
            "provenance": {
                "source_payload_sha256": _digest("training-payload"),
                "training_payload_metadata": {
                    "pooling": "mean",
                    "token_span": "audio",
                    "sequence_has_embedding": False,
                    "layers": [2, 5],
                    "model_provenance": {
                        "model_fingerprint": MODEL_FINGERPRINT,
                    },
                },
            }
        },
    }
    if version == 2:
        checkpoint["directions"] = {
            "harmfulness": {
                2: torch.tensor([2.0, 0.0]),
                5: torch.tensor([2.0, 0.0, 0.0]),
            },
            "refusal": {
                2: torch.tensor([0.0, 4.0]),
                5: torch.tensor([0.0, 4.0, 0.0]),
            },
        }
        checkpoint["class_means"] = {
            "harmfulness": {
                2: {"negative": torch.tensor([-1.0, 0.0]), "positive": torch.tensor([1.0, 0.0])},
                5: {"negative": torch.tensor([-1.0, 0.0, 0.0]), "positive": torch.tensor([1.0, 0.0, 0.0])},
            },
            "refusal": {
                2: {"negative": torch.tensor([0.0, -2.0]), "positive": torch.tensor([0.0, 2.0])},
                5: {"negative": torch.tensor([0.0, -2.0, 0.0]), "positive": torch.tensor([0.0, 2.0, 0.0])},
            },
        }
    torch.save(checkpoint, path)


def _case_payload(case_index: int) -> dict:
    case_id = f"case-{case_index}"
    pair_id = f"pair-{case_index}"
    experiment_fingerprint = _digest(f"experiment-{case_index}")
    checkpoints = [_digest(f"checkpoint-{case_index}-{step}") for step in range(4)]
    layer_two = torch.tensor(
        [
            [0.0 + case_index, 0.0],
            [1.0 + case_index, -1.0],
            [2.0 + case_index, -1.5],
            [3.0 + case_index, -2.0],
        ],
        dtype=torch.float32,
    )
    layer_five = torch.cat(
        [layer_two, torch.full((4, 1), float(case_index))], dim=1
    )
    attack_loss = torch.tensor([4.0, 3.0, 2.0, 1.0]) + case_index
    behavior = []
    row_metadata = []
    for step in range(4):
        status = "unknown" if step == 1 else "ok"
        response = f"response-{case_index}-{step}"
        behavior.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "step": step,
                "label_status": status,
                "generation_status": "ok",
                "refusal_label": None if status == "unknown" else bool(step < 2),
                "compliance_label": None if status == "unknown" else bool(step >= 2),
                "jailbreak_success": None if status == "unknown" else bool(step >= 2),
                "response": response,
                "response_sha256": _digest(response),
            }
        )
        row_metadata.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "step": step,
                "total_steps": 3,
                "progress": step / 3,
                "phase": ("early" if step == 0 else "middle" if step == 1 else "late"),
                "attack_loss": float(attack_loss[step]),
                "checkpoint_sha256": checkpoints[step],
                "experiment_fingerprint": experiment_fingerprint,
                "response_sha256": _digest(f"response-{case_index}-{step}"),
                "label_status": status,
                "generation_status": "ok",
            }
        )
    return {
        "format": "stage1-trajectory-hidden-states",
        "version": 1,
        "case_id": case_id,
        "pair_id": pair_id,
        "steps": torch.arange(4, dtype=torch.int64),
        "hidden_states": OrderedDict([(2, layer_two), (5, layer_five)]),
        "layers": [2, 5],
        "attack_loss": attack_loss.float(),
        "behavior": behavior,
        "row_metadata": row_metadata,
        "checkpoint_sha256": checkpoints,
        "experiment_fingerprint": experiment_fingerprint,
        "metadata": {
            "total_steps": 3,
            "model_name": "toy",
            "model_id": "/models/toy",
            "model_fingerprint": MODEL_FINGERPRINT,
            "model_provenance": dict(MODEL_PROVENANCE),
            "replay_fingerprint": REPLAY_FINGERPRINT,
            "pooling": "mean",
            "token_span": "audio",
            "sequence_has_embedding": False,
            "measurement_split": "measurement_val",
            "stage1_role": "trajectory_candidate",
            "experiment_fingerprint": experiment_fingerprint,
        },
    }


def _write_replay(root: Path, cases: list[dict] | None = None) -> Path:
    replay = root / "replay"
    replay.mkdir(parents=True)
    case_payloads = cases if cases is not None else [_case_payload(0), _case_payload(1)]
    descriptors = []
    for payload in case_payloads:
        path = replay / f"{payload['case_id']}.pt"
        torch.save(payload, path)
        descriptors.append(
            {
                "case_id": payload["case_id"],
                "pair_id": payload["pair_id"],
                "path": path.name,
                "sha256": sha256_file(path),
                "num_steps": len(payload["steps"]),
                "total_steps": payload["metadata"]["total_steps"],
                "experiment_fingerprint": payload["experiment_fingerprint"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
    with (replay / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "stage1-trajectory-hidden-state-index",
                "version": 1,
                "replay_fingerprint": REPLAY_FINGERPRINT,
                "model_fingerprint": MODEL_FINGERPRINT,
                "model_provenance": MODEL_PROVENANCE,
                "config": REPLAY_CONFIG,
                "source_manifest": {
                    "path": "/immutable/manifest.csv",
                    "sha256": SOURCE_MANIFEST_SHA256,
                },
                "behavior_sidecars": BEHAVIOR_SIDECARS,
                "complete": True,
                "num_cases": len(descriptors),
                "cases": descriptors,
            },
            handle,
        )
    return replay


class Stage1TrajectoryScoringTests(unittest.TestCase):
    def test_scores_full_grid_with_independent_directions_and_atomic_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = _write_replay(root)
            probe = root / "probe.pt"
            output = root / "scores"
            _probe_checkpoint(probe)

            payload = score_stage1_trajectories(replay, probe, output)
            self.assertEqual(validate_score_payload(payload), (2, 2, 4))
            from experiments.analyze_stage1_rq1 import (
                validate_score_payload as validate_analysis_payload,
            )

            self.assertEqual(validate_analysis_payload(payload).shape, (2, 2, 4))
            self.assertEqual(tuple(payload["scores"]), SCORE_KEYS)
            self.assertEqual(tuple(payload["deltas"]), DELTA_KEYS)
            self.assertEqual(payload["scores"]["H_probe"].shape, (2, 2, 4))
            self.assertTrue(
                torch.equal(
                    payload["deltas"]["delta_H_probe"][:, :, 0],
                    torch.zeros((2, 2)),
                )
            )
            # Mean-difference H projection is the first hidden coordinate; it
            # is not the learned probe probability or the probe projection.
            self.assertAlmostEqual(payload["scores"]["H_direction"][0, 0, 3].item(), 3.0)
            self.assertNotAlmostEqual(payload["scores"]["H_probe"][0, 0, 3].item(), 3.0)
            self.assertEqual(payload["behavior"][0][1]["label_status"], "unknown")
            self.assertTrue(payload["metadata"]["probe_model_fingerprint_verified"])
            self.assertEqual(payload["metadata"]["probe_training_num_pairs"], 2)
            self.assertTrue((output / "state_scores.pt").is_file())
            self.assertTrue((output / "state_scores_long.csv").is_file())
            self.assertEqual(list(output.glob("*.tmp")), [])
            loaded = torch.load(output / "state_scores.pt", map_location="cpu", weights_only=True)
            self.assertEqual(validate_score_payload(loaded), (2, 2, 4))
            with (output / "state_scores_long.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * 2 * 4)
            self.assertEqual(
                [row["phase"] for row in rows if row["case_id"] == "case-0" and row["layer"] == "2"],
                ["early", "middle", "late", "late"],
            )
            unknown = next(
                row for row in rows
                if row["case_id"] == "case-0" and row["layer"] == "2" and row["step"] == "1"
            )
            self.assertEqual(unknown["label_status"], "unknown")
            self.assertEqual(unknown["refusal_label"], "")

    def test_v1_requires_both_explicit_compatibility_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = _write_replay(root)
            probe = root / "probe-v1.pt"
            _probe_checkpoint(probe, version=1)
            with self.assertRaisesRegex(Stage1ScoringError, "Probe v1"):
                score_stage1_trajectories(replay, probe, root / "rejected")
            with self.assertRaisesRegex(Stage1ScoringError, "source provenance"):
                score_stage1_trajectories(
                    replay,
                    probe,
                    root / "still-rejected",
                    allow_missing_directions=True,
                )
            payload = score_stage1_trajectories(
                replay,
                probe,
                root / "compatible",
                allow_missing_directions=True,
                allow_unverified_provenance=True,
            )
            self.assertTrue(payload["metadata"]["directions_missing"])
            self.assertTrue(torch.isnan(payload["scores"]["H_direction"]).all())
            self.assertTrue(torch.isfinite(payload["scores"]["H_probe"]).all())

    def test_incomplete_step_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case_payload(0)
            case["steps"] = torch.tensor([0, 2, 3], dtype=torch.int64)
            case["hidden_states"] = OrderedDict(
                (layer, states[:3]) for layer, states in case["hidden_states"].items()
            )
            case["attack_loss"] = case["attack_loss"][:3]
            case["behavior"] = case["behavior"][:3]
            case["row_metadata"] = case["row_metadata"][:3]
            case["checkpoint_sha256"] = case["checkpoint_sha256"][:3]
            replay = _write_replay(root, [case])
            probe = root / "probe.pt"
            _probe_checkpoint(probe)
            with self.assertRaisesRegex(Stage1ScoringError, "exactly 0..3|complete 0..T"):
                score_stage1_trajectories(replay, probe, root / "scores")

    def test_probe_model_provenance_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = _write_replay(root)
            probe = root / "probe.pt"
            _probe_checkpoint(probe)
            checkpoint = torch.load(probe, map_location="cpu", weights_only=True)
            checkpoint["metadata"]["provenance"]["training_payload_metadata"][
                "model_provenance"
            ]["model_fingerprint"] = _digest("wrong-model")
            torch.save(checkpoint, probe)
            with self.assertRaisesRegex(Stage1ScoringError, "model_fingerprint mismatch"):
                score_stage1_trajectories(replay, probe, root / "scores")

    def test_v2_missing_source_hash_requires_explicit_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = _write_replay(root)
            probe = root / "probe.pt"
            _probe_checkpoint(probe)
            checkpoint = torch.load(probe, map_location="cpu", weights_only=True)
            checkpoint["metadata"]["provenance"]["source_payload_sha256"] = None
            torch.save(checkpoint, probe)
            with self.assertRaisesRegex(Stage1ScoringError, "source_payload_sha256"):
                score_stage1_trajectories(replay, probe, root / "strict")
            payload = score_stage1_trajectories(
                replay,
                probe,
                root / "compatible-v2",
                allow_unverified_provenance=True,
            )
            self.assertFalse(
                payload["metadata"]["probe_source_payload_sha256_verified"]
            )
            from experiments.analyze_stage1_rq1 import (
                validate_score_payload as validate_analysis_payload,
            )

            with self.assertRaisesRegex(ValueError, "exploratory only"):
                validate_analysis_payload(payload)

    def test_direction_must_equal_positive_minus_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = _write_replay(root)
            probe = root / "probe.pt"
            _probe_checkpoint(probe)
            checkpoint = torch.load(probe, map_location="cpu", weights_only=True)
            checkpoint["directions"]["harmfulness"][2] = torch.tensor([9.0, 0.0])
            torch.save(checkpoint, probe)
            with self.assertRaisesRegex(Stage1ScoringError, "positive-negative"):
                score_stage1_trajectories(replay, probe, root / "scores")


if __name__ == "__main__":
    unittest.main()
