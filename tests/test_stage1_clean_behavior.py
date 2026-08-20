"""Offline tests for hash-bound Stage-1 clean baseline evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import evaluate_stage1_clean as clean


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Model:
    sample_rate = 16_000

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, waveform, *, max_tokens, temperature, do_sample):
        assert max_tokens == 23
        assert temperature == 1.0
        assert do_sample is False
        self.calls.append(str(waveform))
        return f"response:{waveform}"


class _Judge:
    provider = "offline"
    model = "fake-clean-judge"
    base_url = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def public_config(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def evaluate(self, prompt, response):
        self.calls.append((prompt, response))
        refused = int("harmful" in prompt)
        return {"refusal": refused, "score": 0.0 if refused else 0.8}


class Stage1CleanBehaviorTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        files = {
            "benign": root / "benign.wav",
            "harmful": root / "harmful.wav",
            "legacy": root / "legacy-clean.wav",
        }
        files["benign"].write_bytes(b"benign-audio")
        files["harmful"].write_bytes(b"harmful-audio")
        files["legacy"].write_bytes(b"legacy-audio")
        manifest = root / "jbb_probe_candidates.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "pair_id",
                    "measurement_split",
                    "stage1_role",
                    "benign_text",
                    "harmful_text",
                    "benign_audio_path",
                    "harmful_audio_path",
                    "clean_audio_path",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair_id": "pair-alpha",
                    "measurement_split": "measurement_train",
                    "stage1_role": "probe_candidate",
                    "benign_text": "benign request",
                    "harmful_text": "harmful request",
                    "benign_audio_path": str(files["benign"]),
                    "harmful_audio_path": str(files["harmful"]),
                    "clean_audio_path": str(files["legacy"]),
                }
            )
        return manifest, files

    @staticmethod
    def _read(path: Path):
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _loader(path, *, target_sr):
        assert target_sr == 16_000
        return Path(path).stem

    def test_generates_and_labels_both_hash_bound_clean_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, files = self._fixture(root)
            output = root / "clean_labels.jsonl"
            model = _Model()
            judge = _Judge()

            summary = clean.evaluate_clean_baselines(
                manifest,
                output,
                model_id="local/model",
                device="cpu",
                dtype="float32",
                max_tokens=23,
                model_factory=lambda **_kwargs: model,
                audio_loader=self._loader,
                evaluator=judge,
            )

            self.assertEqual(summary["generated"], 2)
            self.assertEqual(summary["judged"], 2)
            self.assertEqual(summary["unknown"], 0)
            rows = self._read(output)
            self.assertEqual([(row["pair_id"], row["state"]) for row in rows], [
                ("pair-alpha", "X_B"),
                ("pair-alpha", "X_H"),
            ])
            by_state = {row["state"]: row for row in rows}
            self.assertIs(by_state["X_B"]["refusal_label"], False)
            self.assertIs(by_state["X_H"]["refusal_label"], True)
            self.assertEqual(Path(by_state["X_H"]["audio_path"]), files["harmful"])
            self.assertNotEqual(Path(by_state["X_H"]["audio_path"]), files["legacy"])
            for row in rows:
                self.assertEqual(row["label_status"], "ok")
                self.assertEqual(row["score"], row["strongreject_score"])
                self.assertEqual(
                    row["source_manifest_sha256"], _sha_file(manifest)
                )
                self.assertEqual(
                    row["prompt_sha256"],
                    hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                )
                self.assertEqual(row["audio_sha256"], _sha_file(Path(row["audio_path"])))
                self.assertEqual(
                    row["response_sha256"],
                    hashlib.sha256(row["response"].encode("utf-8")).hexdigest(),
                )

    def test_output_is_directly_attachable_to_probe_candidates(self):
        from data.prepare_stage1_manifests import attach_clean_outputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, files = self._fixture(root)
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0])
            rows[0]["clean_audio_path"] = str(files["harmful"])
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            labels = root / "clean_labels.jsonl"
            clean.evaluate_clean_baselines(
                manifest,
                labels,
                model_id="local/model",
                device="cpu",
                dtype="float32",
                max_tokens=23,
                model_factory=lambda **_kwargs: _Model(),
                audio_loader=self._loader,
                evaluator=_Judge(),
            )
            attached = attach_clean_outputs(
                manifest,
                labels,
                root / "measurement_manifest.csv",
                project_root=root,
            )
            self.assertEqual(len(attached), 1)
            self.assertFalse(bool(attached.loc[0, "benign_refused"]))
            self.assertTrue(bool(attached.loc[0, "clean_refused"]))

    def test_resume_skips_completed_without_loading_model_or_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, files = self._fixture(root)
            output = root / "clean_labels.jsonl"
            kwargs = {
                "model_id": "local/model",
                "device": "cpu",
                "dtype": "float32",
                "max_tokens": 23,
                "audio_loader": self._loader,
            }
            clean.evaluate_clean_baselines(
                manifest,
                output,
                model_factory=lambda **_kwargs: _Model(),
                evaluator=_Judge(),
                **kwargs,
            )

            def must_not_load(**_kwargs):
                raise AssertionError("completed resume loaded model/judge")

            summary = clean.evaluate_clean_baselines(
                manifest,
                output,
                model_factory=must_not_load,
                evaluator_factory=must_not_load,
                **kwargs,
            )
            self.assertEqual(summary["skipped"], 2)
            self.assertEqual(len(self._read(output)), 2)

            files["harmful"].write_bytes(b"changed-harmful-audio")
            with self.assertRaisesRegex(ValueError, "audio_sha256"):
                clean.evaluate_clean_baselines(
                    manifest,
                    output,
                    model_factory=must_not_load,
                    evaluator_factory=must_not_load,
                    **kwargs,
                )

    def test_judge_initialization_failure_is_unknown_and_resume_reuses_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _files = self._fixture(root)
            output = root / "clean_labels.jsonl"
            model = _Model()

            def failing_judge(**_kwargs):
                raise RuntimeError("judge init failed")

            summary = clean.evaluate_clean_baselines(
                manifest,
                output,
                model_id="local/model",
                device="cpu",
                dtype="float32",
                max_tokens=23,
                model_factory=lambda **_kwargs: model,
                audio_loader=self._loader,
                evaluator_factory=failing_judge,
            )
            self.assertEqual(summary["unknown"], 2)
            self.assertTrue(
                all(row["refusal_label"] is None for row in self._read(output))
            )

            def must_not_regenerate(**_kwargs):
                raise AssertionError("judge retry regenerated clean responses")

            recovered = clean.evaluate_clean_baselines(
                manifest,
                output,
                model_id="local/model",
                device="cpu",
                dtype="float32",
                max_tokens=23,
                model_factory=must_not_regenerate,
                audio_loader=self._loader,
                evaluator=_Judge(),
            )
            self.assertEqual(recovered["generated"], 0)
            self.assertEqual(recovered["judged"], 2)
            self.assertTrue(
                all(row["label_status"] == "ok" for row in self._read(output))
            )

    def test_model_initialization_failure_writes_unknown_for_every_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _files = self._fixture(root)
            output = root / "clean_labels.jsonl"
            calls = []

            def failing_model(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("model init failed")

            summary = clean.evaluate_clean_baselines(
                manifest,
                output,
                model_factory=failing_model,
                evaluator=_Judge(),
            )
            self.assertEqual(summary["unknown"], 2)
            self.assertEqual(len(calls), 2)
            rows = self._read(output)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["generation_status"] == "error" for row in rows))
            self.assertTrue(all(row["label_status"] == "unknown" for row in rows))
            self.assertTrue(all(row["refusal_label"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
