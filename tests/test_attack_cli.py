"""Offline tests for the migrated single-case entry point."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import attack


class SingleAttackEntryPointTests(unittest.TestCase):
    def test_run_single_builds_manifest_and_delegates_to_batch_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input.wav"
            audio.write_bytes(b"offline-audio")
            output = root / "run"
            expected = {
                "counts": {"completed": 1, "failed": 0, "skipped": 0}
            }
            with mock.patch.object(
                attack, "run_batch", return_value=expected
            ) as runner:
                result = attack.run_single(
                    audio,
                    output,
                    harmful_text="test harmful request",
                    target_text="target response",
                    stratum="test",
                    method="standard",
                    model_name="qwen-3b",
                )

            self.assertEqual(result, expected)
            manifest = output / "single_manifest.json"
            with manifest.open("r", encoding="utf-8") as handle:
                rows = json.load(handle)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["pair_id"].startswith("pair_"))
            self.assertEqual(rows[0]["clean_audio_path"], str(audio.resolve()))
            self.assertEqual(runner.call_args.kwargs["method"], "standard")

    def test_pair_id_is_content_stable_and_changes_with_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input.wav"
            audio.write_bytes(b"same-audio")
            with mock.patch.object(
                attack,
                "run_batch",
                return_value={"counts": {"completed": 1, "failed": 0, "skipped": 0}},
            ):
                attack.run_single(
                    audio,
                    root / "a",
                    harmful_text="request one",
                    target_text="target",
                )
                attack.run_single(
                    audio,
                    root / "b",
                    harmful_text="request one",
                    target_text="target",
                )
                attack.run_single(
                    audio,
                    root / "c",
                    harmful_text="request two",
                    target_text="target",
                )
            manifests = []
            for name in ("a", "b", "c"):
                with (root / name / "single_manifest.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    manifests.append(json.load(handle)[0])
            self.assertEqual(manifests[0]["pair_id"], manifests[1]["pair_id"])
            self.assertNotEqual(manifests[0]["pair_id"], manifests[2]["pair_id"])

    def test_blank_required_text_and_missing_audio_fail_before_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "input.wav"
            audio.write_bytes(b"x")
            with mock.patch.object(attack, "run_batch") as runner:
                with self.assertRaisesRegex(ValueError, "harmful_text"):
                    attack.run_single(
                        audio,
                        root / "run",
                        harmful_text=" ",
                        target_text="target",
                    )
                with self.assertRaises(FileNotFoundError):
                    attack.run_single(
                        root / "missing.wav",
                        root / "run",
                        harmful_text="request",
                        target_text="target",
                    )
                runner.assert_not_called()

    def test_main_removes_no_early_stop_and_returns_batch_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "input.wav"
            audio.write_bytes(b"x")
            summary = {
                "counts": {"completed": 1, "failed": 0, "skipped": 0}
            }
            with mock.patch.object(
                attack, "run_single", return_value=summary
            ) as runner:
                code = attack.main(
                    [
                        "--wav",
                        str(audio),
                        "--output-dir",
                        str(Path(directory) / "out"),
                        "--harmful-text",
                        "request",
                        "--target-text",
                        "target",
                        "--no-early-stop",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertNotIn("no_early_stop", runner.call_args.kwargs)
            self.assertFalse(runner.call_args.kwargs["early_stop"])


if __name__ == "__main__":
    unittest.main()
