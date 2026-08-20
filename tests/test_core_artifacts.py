"""Offline tests for tensor trajectory artifact storage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.artifacts import TrajectoryArtifactStore


class TrajectoryArtifactStoreTests(unittest.TestCase):
    def test_tensor_checkpoint_index_and_recovery_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TrajectoryArtifactStore(temporary)
            delta = torch.tensor([0.1, -0.2], requires_grad=True)
            hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)

            record = store.save_checkpoint(
                10,
                {"delta": delta, "pooled_hidden": hidden},
                metadata={"attack_success": False, "layer": 7},
            )

            self.assertEqual(store.list_steps(), [10])
            self.assertEqual(store.latest_step(), 10)
            self.assertEqual(record.metadata["layer"], 7)
            self.assertEqual(record.tensors["pooled_hidden"]["shape"], [2, 3])

            index = json.loads(store.index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["checkpoints"][0]["step"], 10)
            self.assertEqual(
                index["checkpoints"][0]["tensors"]["delta"]["dtype"],
                "torch.float32",
            )
            # JSON stores descriptors, never tensor values.
            self.assertNotIn("0.100000001", store.index_path.read_text())

            recovered = store.recover_checkpoint()
            self.assertEqual(recovered.step, 10)
            self.assertTrue(torch.equal(recovered.tensors["delta"], delta.detach()))
            self.assertEqual(recovered.tensors["delta"].device.type, "cpu")
            self.assertFalse(recovered.tensors["delta"].requires_grad)

    def test_lists_orphaned_atomic_tensor_file_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TrajectoryArtifactStore(temporary)
            orphan = store.trajectory_dir / "step_000025.pt"
            torch.save({"delta": torch.tensor([0.25])}, orphan)

            self.assertEqual(store.list_steps(), [25])
            recovered = store.recover_checkpoint()
            self.assertEqual(recovered.step, 25)
            self.assertEqual(recovered.record.metadata, {})
            self.assertAlmostEqual(recovered.tensors["delta"].item(), 0.25)

    def test_duplicate_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TrajectoryArtifactStore(temporary)
            store.save_checkpoint(0, {"delta": torch.tensor([0.0])})
            with self.assertRaises(FileExistsError):
                store.save_checkpoint(0, {"delta": torch.tensor([1.0])})
            store.save_checkpoint(
                0, {"delta": torch.tensor([1.0])}, overwrite=True
            )
            self.assertEqual(store.load_checkpoint(0)["delta"].item(), 1.0)
            self.assertEqual(store.list_steps(), [0])

    def test_metadata_rejects_tensors_in_json_and_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TrajectoryArtifactStore(temporary)
            with self.assertRaisesRegex(TypeError, "metadata only"):
                store.save_checkpoint(
                    0,
                    {"delta": torch.zeros(1)},
                    metadata={"hidden": torch.zeros(2)},
                )
            with self.assertRaisesRegex(TypeError, "metadata only"):
                store.append_event({"hidden": torch.zeros(2)})

            store.write_run_metadata({"case_id": "case-1", "seed": 42})
            store.append_event({"step": 0, "loss": 1.25})
            self.assertEqual(
                json.loads((Path(temporary) / "run.json").read_text())["seed"],
                42,
            )
            event = json.loads(store.events_path.read_text().splitlines()[0])
            self.assertEqual(event["step"], 0)
            self.assertIn("timestamp", event)

    def test_failed_tensor_save_does_not_publish_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TrajectoryArtifactStore(temporary)
            with patch("core.artifacts.torch.save", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    store.save_checkpoint(3, {"delta": torch.zeros(1)})

            self.assertEqual(store.list_steps(), [])
            self.assertFalse(
                (store.trajectory_dir / "step_000003.pt").exists()
            )
            self.assertEqual(list(store.trajectory_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
