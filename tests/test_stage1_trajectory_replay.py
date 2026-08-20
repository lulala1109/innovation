"""Offline tests for complete held-out Stage-1 hidden-state replay."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from experiments.replay_stage1_trajectories import (
    INDEX_FORMAT,
    OUTPUT_FORMAT,
    Stage1TrajectoryReplayError,
    replay_stage1_trajectories,
    validate_replay_payload,
)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Forward:
    def __init__(self, hidden_states, attention_mask, token_spans):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self.token_spans = token_spans


class _ToyModel:
    def __init__(self):
        self.calls: list[tuple[float, str, bool, bool]] = []
        self.training = True

    def eval(self):
        self.training = False
        return self

    def forward_attack(self, waveform, target_text, *, output_hidden_states=False):
        self.calls.append(
            (
                float(waveform.mean()),
                str(target_text),
                bool(output_hidden_states),
                bool(torch.is_grad_enabled()),
            )
        )
        value = waveform.float().mean()
        positions = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        first = torch.cat(
            (
                value.expand(1, 4, 1) + positions,
                (value * 2).expand(1, 4, 1),
            ),
            dim=-1,
        )
        second = first + 10.0
        embedding = torch.zeros_like(first)
        return _Forward(
            (embedding, first, second),
            torch.ones(1, 4, dtype=torch.long),
            {"audio": (0, 3), "target": (3, 4)},
        )


class Stage1TrajectoryReplayTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        split: str = "measurement_val",
        role: str = "trajectory_candidate",
        nonzero_step_zero: bool = False,
        omit_index_step: int | None = None,
        tamper_label_sha: bool = False,
    ) -> tuple[Path, Path]:
        case = root / "runs" / "case-alpha"
        trajectory = case / "trajectory"
        trajectory.mkdir(parents=True)
        experiment_config = {
            "method": "standard",
            "steps": 2,
            "target_text": "fixed attack target",
        }
        experiment_fingerprint = hashlib.sha256(
            json.dumps(
                experiment_config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        checkpoint_paths: dict[int, Path] = {}
        index_rows = []
        for step in range(3):
            delta_value = 0.25 if step == 0 and nonzero_step_zero else step * 0.1
            delta = torch.full((1, 8), delta_value)
            waveform = torch.full((1, 8), float(step))
            checkpoint = trajectory / f"step_{step:06d}.pt"
            torch.save({"delta": delta, "adversarial_wav": waveform}, checkpoint)
            checkpoint_paths[step] = checkpoint
            if step == omit_index_step:
                continue
            index_rows.append(
                {
                    "step": step,
                    "path": f"trajectory/{checkpoint.name}",
                    "tensors": {
                        "delta": {"shape": [1, 8], "dtype": "torch.float32"},
                        "adversarial_wav": {
                            "shape": [1, 8],
                            "dtype": "torch.float32",
                        },
                    },
                    "metadata": {
                        "case_id": "case-alpha",
                        "pair_id": "pair-alpha",
                        "step": step,
                        "state": step,
                        "loss": 3.0 - step,
                        "delta_linf": abs(delta_value),
                        "delta_rms": abs(delta_value),
                        "snr_db": None if step == 0 else 20.0 - step,
                        "target_text": "fixed attack target",
                        "harmful_text": "harmful request",
                        "experiment_fingerprint": experiment_fingerprint,
                    },
                }
            )
        trajectory_index = trajectory / "index.json"
        trajectory_index.write_text(
            json.dumps(
                {
                    "format": "safety-state-trajectory",
                    "version": 1,
                    "checkpoints": index_rows,
                }
            ),
            encoding="utf-8",
        )
        (case / "run.json").write_text(
            json.dumps(
                {
                    "case_id": "case-alpha",
                    "pair_id": "pair-alpha",
                    "method": "standard",
                    "model": "qwen-3b",
                    "harmful_text": "harmful request",
                    "budget": {
                        "steps": 2,
                        "init_mode": "zero",
                        "experiment_config": experiment_config,
                        "experiment_fingerprint": experiment_fingerprint,
                    },
                    "artifacts": {"trajectory": "trajectory/index.json"},
                }
            ),
            encoding="utf-8",
        )
        labels = root / "runs" / "behavior_labels.jsonl"
        with labels.open("w", encoding="utf-8") as handle:
            for step in (0, 1):
                response = "I refuse" if step == 0 else "uncertain response"
                checkpoint_sha = hashlib.sha256(
                    checkpoint_paths[step].read_bytes()
                ).hexdigest()
                if tamper_label_sha and step == 1:
                    checkpoint_sha = "0" * 64
                record = {
                    "format": "stage1-behavior-label",
                    "version": 1,
                    "case_id": "case-alpha",
                    "pair_id": "pair-alpha",
                    "step": step,
                    "checkpoint_path": str(checkpoint_paths[step]),
                    "checkpoint_sha256": checkpoint_sha,
                    "experiment_fingerprint": experiment_fingerprint,
                    "generation_status": "ok",
                    "label_status": "ok" if step == 0 else "unknown",
                    "response": response,
                    "response_sha256": _hash_text(response),
                }
                if step == 0:
                    record.update(
                        {
                            "refusal_label": True,
                            "compliance_label": False,
                            "jailbreak_success": False,
                        }
                    )
                handle.write(json.dumps(record) + "\n")
        manifest = root / "derived" / "trajectory-attached.csv"
        manifest.parent.mkdir()
        pd.DataFrame(
            [
                {
                    "case_id": "case-alpha",
                    "pair_id": "pair-alpha",
                    "harmful_text": "harmful request",
                    "measurement_split": split,
                    "stage1_role": role,
                    "trajectory_path": str(trajectory_index),
                    "behavior_labels_path": str(labels),
                    "experiment_fingerprint": experiment_fingerprint,
                }
            ]
        ).to_csv(manifest, index=False)
        return manifest, labels

    @staticmethod
    def _load_case(index_path: Path):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        path = index_path.parent / index["cases"][0]["path"]
        return index, torch.load(path, map_location="cpu", weights_only=True)

    def test_replay_preserves_complete_grid_unknown_and_missing_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root)
            model = _ToyModel()
            output = root / "hidden"

            summary = replay_stage1_trajectories(
                manifest,
                output,
                model=model,
                model_id="toy/model",
                device="cpu",
                dtype="float32",
                layers=(0, 1),
                project_root=root,
            )

            self.assertEqual((summary["replayed"], summary["skipped"]), (1, 0))
            index, payload = self._load_case(output / "index.json")
            self.assertEqual(index["format"], INDEX_FORMAT)
            self.assertTrue(index["complete"])
            self.assertEqual(
                index["model_provenance"],
                payload["metadata"]["model_provenance"],
            )
            self.assertEqual(
                index["model_fingerprint"],
                index["model_provenance"]["model_fingerprint"],
            )
            self.assertEqual(index["model_provenance"]["dtype"], "float32")
            self.assertEqual(payload["format"], OUTPUT_FORMAT)
            self.assertEqual(validate_replay_payload(payload), 3)
            torch.testing.assert_close(payload["steps"], torch.tensor([0, 1, 2]))
            torch.testing.assert_close(
                payload["attack_loss"], torch.tensor([3.0, 2.0, 1.0])
            )
            self.assertEqual(list(payload["hidden_states"]), [0, 1])
            self.assertEqual(payload["hidden_states"][0].shape, (3, 2))
            self.assertEqual(
                [row["label_status"] for row in payload["behavior"]],
                ["ok", "unknown", "missing"],
            )
            self.assertIsNone(payload["behavior"][1]["refusal_label"])
            self.assertIsNone(payload["behavior"][2]["jailbreak_success"])
            self.assertEqual(
                [row["forward_target_source"] for row in payload["row_metadata"]],
                ["response", "response", "harmful_text"],
            )
            self.assertFalse(model.training)
            self.assertEqual(len(model.calls), 3)
            self.assertTrue(all(call[2] for call in model.calls))
            self.assertTrue(all(not call[3] for call in model.calls))

            # Exercise the real downstream bridge, not a duplicated fixture
            # validator, so producer/consumer schema drift fails immediately.
            from experiments.score_stage1_trajectories import load_replay_artifact

            loaded = load_replay_artifact(output)
            self.assertEqual(len(loaded["cases"]), 1)
            self.assertEqual(loaded["cases"][0]["steps"].tolist(), [0, 1, 2])

    def test_resume_validates_outputs_without_forwarding_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root)
            output = root / "hidden"
            replay_stage1_trajectories(
                manifest,
                output,
                model=_ToyModel(),
                model_id="toy/model",
                device="cpu",
                dtype="float32",
                project_root=root,
            )
            resumed_model = _ToyModel()
            summary = replay_stage1_trajectories(
                manifest,
                output,
                model=resumed_model,
                model_id="toy/model",
                device="cpu",
                dtype="float32",
                project_root=root,
            )
            self.assertEqual((summary["replayed"], summary["skipped"]), (0, 1))
            self.assertEqual(resumed_model.calls, [])

            with self.assertRaisesRegex(
                Stage1TrajectoryReplayError, "Replay fingerprint changed"
            ):
                replay_stage1_trajectories(
                    manifest,
                    output,
                    model=_ToyModel(),
                    model_id="toy/model-v2",
                    device="cpu",
                    dtype="float32",
                    project_root=root,
                )

    def test_nonzero_t0_is_rejected_before_hidden_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, nonzero_step_zero=True)
            model = _ToyModel()
            with self.assertRaisesRegex(
                Stage1TrajectoryReplayError, "t=0.*zero delta"
            ):
                replay_stage1_trajectories(
                    manifest,
                    root / "hidden",
                    model=model,
                    model_id="toy/model",
                    device="cpu",
                    dtype="float32",
                    project_root=root,
                )
            self.assertEqual(model.calls, [])

    def test_only_held_out_trajectory_candidates_are_accepted(self):
        for split, role in (
            ("measurement_train", "trajectory_candidate"),
            ("measurement_val", "probe_candidate"),
        ):
            with self.subTest(split=split, role=role), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _ = self._fixture(root, split=split, role=role)
                model = _ToyModel()
                with self.assertRaisesRegex(
                    Stage1TrajectoryReplayError,
                    "only measurement_val/trajectory_candidate",
                ):
                    replay_stage1_trajectories(
                        manifest,
                        root / "hidden",
                        model=model,
                        model_id="toy/model",
                        device="cpu",
                        dtype="float32",
                        project_root=root,
                    )
                self.assertEqual(model.calls, [])

    def test_incomplete_grid_and_behavior_checkpoint_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, omit_index_step=1)
            with self.assertRaisesRegex(
                Stage1TrajectoryReplayError, "Invalid complete trajectory"
            ):
                replay_stage1_trajectories(
                    manifest,
                    root / "hidden",
                    model=_ToyModel(),
                    model_id="toy/model",
                    device="cpu",
                    dtype="float32",
                    project_root=root,
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._fixture(root, tamper_label_sha=True)
            with self.assertRaisesRegex(
                Stage1TrajectoryReplayError, "Behavior checkpoint SHA mismatch"
            ):
                replay_stage1_trajectories(
                    manifest,
                    root / "hidden",
                    model=_ToyModel(),
                    model_id="toy/model",
                    device="cpu",
                    dtype="float32",
                    project_root=root,
                )

    def test_cli_help_is_lightweight(self):
        result = subprocess.run(
            [sys.executable, "-m", "experiments.replay_stage1_trajectories", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--token-span", result.stdout)
        self.assertIn("--pooling", result.stdout)


if __name__ == "__main__":
    unittest.main()
