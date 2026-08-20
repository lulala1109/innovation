"""Offline tests for the generic, resumable safety-attack batch runner."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments import batch_safety_attack as batch


class FakeModel:
    sample_rate = 16_000
    device = "cpu"


class FakeAttacker:
    def __init__(self, *, fail: bool = False, references: list | None = None) -> None:
        self.fail = fail
        self.references = references

    def attack(self, wav, target_text, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic attack failure")
        if self.references is not None:
            self.references.append(kwargs.get("reference_refusal"))
        callback = kwargs["state_callback"]
        for step in kwargs["checkpoint_steps"]:
            callback(
                step,
                wav + 0.01,
                torch.full_like(wav, 0.01),
                {"step": step, "loss": float(step)},
            )
        return SimpleNamespace(
            adversarial_wav=wav + 0.01,
            adversarial_output=f"response to {target_text}",
            success=True,
        )


class BatchSafetyAttackTests(unittest.TestCase):
    def _manifest(self, root: Path, *, row_reference: str = "") -> Path:
        (root / "input.wav").touch()
        path = root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "case_id",
                    "pair_id",
                    "stratum",
                    "harmful_text",
                    "clean_audio_path",
                    "target_text",
                    "reference_refusal_path",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "case_id": "case-alpha",
                    "pair_id": "pair-alpha",
                    "stratum": "cyber",
                    "harmful_text": "harmful prompt",
                    "clean_audio_path": "input.wav",
                    "target_text": "target",
                    "reference_refusal_path": row_reference,
                }
            )
        return path

    def _jbb_manifest(self, root: Path) -> tuple[Path, Path]:
        manifest = root / "dataset" / "processed" / "stage1" / "jbb_pairs_audio.csv"
        manifest.parent.mkdir(parents=True)
        audio = (
            root
            / "dataset"
            / "derived"
            / "jbb_audio"
            / "harmful_clean"
            / "jbb_000.wav"
        )
        audio.parent.mkdir(parents=True)
        audio.touch()
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "pair_id",
                    "category",
                    "harmful_text",
                    "harmful_target",
                    "harmful_audio_path",
                    "measurement_split",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "pair_id": "jbb_000",
                    "category": "harassment",
                    "harmful_text": "harmful prompt",
                    "harmful_target": "harmful target",
                    "harmful_audio_path": (
                        "dataset/derived/jbb_audio/harmful_clean/jbb_000.wav"
                    ),
                    "measurement_split": "train",
                }
            )
        return manifest, audio

    @staticmethod
    def _model_factory(**_kwargs):
        return FakeModel()

    @staticmethod
    def _audio_loader(_path, *, target_sr):
        assert target_sr == 16_000
        return torch.zeros(1, 12)

    @staticmethod
    def _audio_saver(_wav, path, *, sample_rate):
        assert sample_rate == 16_000
        Path(path).write_bytes(b"offline-test-audio")

    def test_standard_run_writes_generic_schema_checkpoints_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output = root / "runs"
            factory_calls: list[dict] = []

            def attacker_factory(**kwargs):
                factory_calls.append(kwargs)
                return FakeAttacker()

            summary = batch.run_batch(
                manifest,
                output,
                method="standard",
                model_id="local/qwen",
                steps=2,
                checkpoint_steps=(0, 2),
                determinism="off",
                model_factory=self._model_factory,
                attacker_factory=attacker_factory,
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
            )
            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertEqual(len(factory_calls), 1)
            case_dirs = [path for path in output.iterdir() if path.is_dir()]
            self.assertEqual(len(case_dirs), 1)
            with (case_dirs[0] / "run.json").open("r", encoding="utf-8") as handle:
                run = json.load(handle)
            self.assertEqual(set(run), set(batch.RUN_FIELDS))
            self.assertEqual(run["pair_id"], "pair-alpha")
            self.assertEqual(run["method"], "standard")
            self.assertEqual(run["budget"]["checkpoint_steps"], [0, 2])
            with (case_dirs[0] / "trajectory" / "index.json").open(
                "r", encoding="utf-8"
            ) as handle:
                index = json.load(handle)
            self.assertEqual(
                [item["step"] for item in index["checkpoints"]], [0, 2]
            )

            def must_not_reload(**_kwargs):
                raise AssertionError("completed resume should not reload the model")

            resumed = batch.run_batch(
                manifest,
                output,
                method="standard",
                model_id="local/qwen",
                steps=2,
                checkpoint_steps=(0, 2),
                determinism="off",
                model_factory=must_not_reload,
                attacker_factory=attacker_factory,
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
            )
            self.assertEqual(resumed["counts"]["skipped"], 1)

    def test_jbb_aliases_and_project_root_audio_path_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, expected_audio = self._jbb_manifest(root)
            records = batch._read_records(manifest)
            records[0]["clean_audio_path"] = "wrong.wav"
            self.assertEqual(
                batch._audio_path_for_row(
                    records[0],
                    row_number=0,
                ),
                records[0]["harmful_audio_path"],
            )
            self.assertEqual(records[0]["stratum"], "harassment")
            self.assertEqual(records[0]["target_text"], "harmful target")

            loaded_paths: list[Path] = []

            def audio_loader(path, *, target_sr):
                self.assertEqual(target_sr, 16_000)
                loaded_paths.append(Path(path))
                return torch.zeros(1, 12)

            with mock.patch.object(batch, "PROJECT_ROOT", root):
                summary = batch.run_batch(
                    manifest,
                    root / "runs",
                    method="standard",
                    steps=0,
                    checkpoint_steps=(0,),
                    determinism="off",
                    model_factory=self._model_factory,
                    attacker_factory=lambda **_kwargs: FakeAttacker(),
                    audio_loader=audio_loader,
                    audio_saver=self._audio_saver,
                )

            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertEqual(loaded_paths, [expected_audio.resolve()])
            run_path = next(
                (root / "runs").glob("*/run.json")
            )
            with run_path.open("r", encoding="utf-8") as handle:
                run = json.load(handle)
            self.assertEqual(run["stratum"], "harassment")
            self.assertIn("harmful target", run["adversarial_response"])
            self.assertEqual(run["artifacts"]["input_audio"], str(expected_audio))
            self.assertEqual(
                run["budget"]["experiment_config"]["input_audio"]["path"],
                str(expected_audio),
            )

    def test_resume_reexecutes_when_experiment_fingerprint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output = root / "runs"
            calls: list[dict] = []

            def model_factory(**kwargs):
                calls.append(kwargs)
                return FakeModel()

            common = {
                "method": "standard",
                "determinism": "off",
                "model_factory": model_factory,
                "attacker_factory": lambda **_kwargs: FakeAttacker(),
                "audio_loader": self._audio_loader,
                "audio_saver": self._audio_saver,
                "steps": 1,
                "checkpoint_steps": (0, 1),
            }
            batch.run_batch(manifest, output, alpha=0.01, **common)
            unchanged = batch.run_batch(
                manifest, output, alpha=0.01, **common
            )
            self.assertEqual(unchanged["counts"]["skipped"], 1)

            changed = batch.run_batch(
                manifest, output, alpha=0.02, **common
            )
            self.assertEqual(changed["counts"]["skipped"], 0)
            self.assertEqual(changed["counts"]["completed"], 1)
            self.assertEqual(len(calls), 2)
            case_dir = next(path for path in output.iterdir() if path.is_dir())
            with (case_dir / "run.json").open("r", encoding="utf-8") as handle:
                run = json.load(handle)
            self.assertEqual(run["budget"]["alpha"], 0.02)
            self.assertEqual(
                len(run["budget"]["experiment_fingerprint"]), 64
            )

    def test_failure_is_isolated_in_error_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runs"
            summary = batch.run_batch(
                self._manifest(root),
                output,
                method="standard",
                determinism="off",
                model_factory=self._model_factory,
                attacker_factory=lambda **_kwargs: FakeAttacker(fail=True),
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
            )
            self.assertEqual(summary["counts"]["failed"], 1)
            case_dir = next(path for path in output.iterdir() if path.is_dir())
            self.assertTrue((case_dir / "error.json").is_file())
            self.assertFalse((case_dir / "run.json").exists())

    def test_pair_keyed_reference_is_selected_for_each_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references: list = []
            batch.run_batch(
                self._manifest(root),
                root / "runs",
                method="uniform",
                probe_checkpoint="trained-probe.pt",
                reference_refusal_path="references.pt",
                steps=0,
                checkpoint_steps=(0,),
                determinism="off",
                model_factory=self._model_factory,
                attacker_factory=lambda **_kwargs: FakeAttacker(references=references),
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
                probe_loader=lambda _path: object(),
                reference_loader=lambda _path: {"pair-alpha": {0: 0.9, 1: 0.8}},
            )
            self.assertEqual(references, [{0: 0.9, 1: 0.8}])
    def test_row_reference_works_without_global_and_is_pair_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references: list = []
            loader_paths: list[Path] = []

            def reference_loader(path):
                loader_paths.append(Path(path))
                return {
                    "by_pair_id": {
                        "pair-alpha": {0: 0.7, 1: 0.6},
                        "another-pair": {0: -99.0, 1: -99.0},
                    }
                }

            summary = batch.run_batch(
                self._manifest(root, row_reference="row-reference.pt"),
                root / "runs",
                method="uniform",
                probe_checkpoint="trained-probe.pt",
                reference_refusal_path=None,
                steps=0,
                checkpoint_steps=(0,),
                determinism="off",
                model_factory=self._model_factory,
                attacker_factory=lambda **_kwargs: FakeAttacker(
                    references=references
                ),
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
                probe_loader=lambda _path: object(),
                reference_loader=reference_loader,
            )

            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertEqual(references, [{0: 0.7, 1: 0.6}])
            self.assertEqual(loader_paths, [(root / "row-reference.pt").resolve()])

    def test_row_reference_missing_pair_is_rejected_without_cross_pair_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = batch.run_batch(
                self._manifest(root, row_reference="row-reference.pt"),
                root / "runs",
                method="uniform",
                probe_checkpoint="trained-probe.pt",
                steps=0,
                checkpoint_steps=(0,),
                determinism="off",
                model_factory=self._model_factory,
                attacker_factory=lambda **_kwargs: FakeAttacker(),
                audio_loader=self._audio_loader,
                audio_saver=self._audio_saver,
                probe_loader=lambda _path: object(),
                reference_loader=lambda _path: {
                    "another-pair": {0: 0.1, 1: 0.2}
                },
            )

            self.assertEqual(summary["counts"]["failed"], 1)
            self.assertIn("missing 'pair-alpha'", summary["cases"][0]["error"])

    def test_internal_method_without_any_reference_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def must_not_load(**_kwargs):
                raise AssertionError("manifest validation must precede model loading")

            with self.assertRaisesRegex(ValueError, "every manifest row"):
                batch.run_batch(
                    self._manifest(root),
                    root / "runs",
                    method="uniform",
                    probe_checkpoint="trained-probe.pt",
                    reference_refusal_path=None,
                    model_factory=must_not_load,
                )


    def test_method_inputs_and_cli_parameter_forwarding(self) -> None:
        with self.assertRaisesRegex(ValueError, "already-trained"):
            batch.validate_method_inputs(
                "safety_state_adaptive",
                probe_checkpoint=None,
                reference_refusal_path="reference.pt",
            )
        with self.assertRaisesRegex(ValueError, "static-topk-layers"):
            batch.validate_method_inputs(
                "static_topk",
                probe_checkpoint="probe.pt",
                reference_refusal_path="reference.pt",
                static_topk_layers=None,
            )
        with self.assertRaisesRegex(ValueError, "exactly top_k"):
            batch.validate_method_inputs(
                "static_topk",
                probe_checkpoint="probe.pt",
                reference_refusal_path="reference.pt",
                top_k=2,
                static_topk_layers=(3,),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            batch.validate_method_inputs(
                "static_topk",
                probe_checkpoint="probe.pt",
                reference_refusal_path="reference.pt",
                top_k=2,
                static_topk_layers=(3, 3),
            )
        batch.validate_method_inputs(
            "static_topk",
            probe_checkpoint="probe.pt",
            reference_refusal_path=None,
            top_k=2,
            static_topk_layers=(3, 7),
        )

        fake_summary = {"counts": {"failed": 0}}
        with mock.patch.object(batch, "run_batch", return_value=fake_summary) as runner:
            exit_code = batch.main(
                [
                    "--manifest",
                    "manifest.csv",
                    "--output-dir",
                    "runs",
                    "--model-id",
                    "local/qwen",
                    "--no-early-stop",
                ]
            )
        self.assertEqual(exit_code, 0)
        kwargs = runner.call_args.kwargs
        self.assertNotIn("no_early_stop", kwargs)
        self.assertFalse(kwargs["early_stop"])
        self.assertEqual(kwargs["model_id"], "local/qwen")


if __name__ == "__main__":
    unittest.main()
