"""Offline tests for immutable Stage-1 derived manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data.prepare_stage1_manifests import (
    Stage1ManifestError,
    attach_attack_outputs,
    attach_clean_outputs,
    finalize_measurement_manifest,
    prepare_candidate_manifests,
    response_sha256,
    sha256_file,
)


class Stage1ManifestTests(unittest.TestCase):
    def test_prepare_preserves_source_and_existing_80_20_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset" / "processed" / "stage1" / "source.csv"
            source.parent.mkdir(parents=True)
            rows = []
            for category in range(10):
                for offset in range(10):
                    pair = category * 10 + offset
                    rows.append(
                        {
                            "pair_id": f"pair-{pair:03d}",
                            "category": f"category-{category}",
                            "measurement_split": (
                                "measurement_train"
                                if offset < 8
                                else "measurement_val"
                            ),
                            "benign_text": f"benign {pair}",
                            "harmful_text": f"harmful {pair}",
                            "benign_audio_path": f"dataset/audio/b-{pair}.wav",
                            "harmful_audio_path": f"dataset/audio/h-{pair}.wav",
                        }
                    )
            pd.DataFrame(rows).to_csv(source, index=False)
            before = sha256_file(source)

            outputs = prepare_candidate_manifests(
                source,
                root / "derived",
                project_root=root,
            )

            self.assertEqual(sha256_file(source), before)
            train = pd.read_csv(outputs["probe"])
            validation = pd.read_csv(outputs["trajectory"])
            self.assertEqual((len(train), len(validation)), (80, 20))
            self.assertTrue((train.groupby("category").size() == 8).all())
            self.assertTrue((validation.groupby("category").size() == 2).all())
            self.assertTrue(set(train.pair_id).isdisjoint(validation.pair_id))
            self.assertEqual(
                train.loc[0, "benign_audio_path"], "dataset/audio/b-0.wav"
            )
            self.assertTrue(outputs["exclusions"].is_file())

    @staticmethod
    def _write_attack_artifacts(
        root: Path,
        *,
        trajectory_pair="pair-a",
        unknown_steps=(),
        omitted_steps=(),
        total_steps=1,
        history_selected_step=None,
        losses=None,
        successful_steps=None,
        tensor_checkpoints=False,
    ):
        import torch

        case = root / "runs" / "case-a"
        trajectory = case / "trajectory"
        trajectory.mkdir(parents=True, exist_ok=True)
        steps = tuple(range(total_steps + 1))
        selected_step = (
            total_steps if history_selected_step is None else history_selected_step
        )
        successful = (
            {total_steps} if successful_steps is None else set(successful_steps)
        )
        experiment_config = {"method": "standard", "steps": total_steps}
        experiment_fingerprint = hashlib.sha256(
            json.dumps(
                experiment_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for step in steps:
            checkpoint = trajectory / f"step_{step:06d}.pt"
            if tensor_checkpoints:
                torch.save(
                    {
                        "adversarial_wav": torch.full(
                            (1, 8), float(step) / 10.0, dtype=torch.float32
                        )
                    },
                    checkpoint,
                )
            else:
                checkpoint.write_bytes(b"checkpoint")
        (case / "adversarial.wav").write_bytes(b"wave")
        (trajectory / "index.json").write_text(
            json.dumps(
                {
                    "format": "safety-state-trajectory",
                    "version": 1,
                    "checkpoints": [
                        {
                            "step": step,
                            "path": f"trajectory/step_{step:06d}.pt",
                            "metadata": {
                                "step": step,
                                "pair_id": trajectory_pair,
                                "case_id": "case-a",
                                "experiment_fingerprint": experiment_fingerprint,
                            },
                        }
                        for step in steps
                    ],
                }
            ),
            encoding="utf-8",
        )
        history = {"selection": {"step": selected_step}}
        if losses is not None:
            history["iterations"] = [
                {
                    "step": step,
                    "updates_completed": step,
                    "loss": losses[step],
                }
                for step in steps
            ]
        (case / "history.json").write_text(
            json.dumps(
                {
                    "case_id": "case-a",
                    "pair_id": "pair-a",
                    "history": history,
                }
            ),
            encoding="utf-8",
        )
        (case / "run.json").write_text(
            json.dumps(
                {
                    "case_id": "case-a",
                    "pair_id": "pair-a",
                    "method": "standard",
                    "budget": {
                        "steps": total_steps,
                        "experiment_config": experiment_config,
                        "experiment_fingerprint": experiment_fingerprint,
                    },
                    "artifacts": {
                        "trajectory": "trajectory/index.json",
                        "history": "history.json",
                        "adversarial_audio": "adversarial.wav",
                    },
                }
            ),
            encoding="utf-8",
        )
        summary = root / "runs" / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "output_dir": str(root / "runs"),
                    "cases": [
                        {
                            "case_id": "case-a",
                            "pair_id": "pair-a",
                            "status": "completed",
                            "path": str(case),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        labels = root / "runs" / "behavior_labels.jsonl"
        with labels.open("w", encoding="utf-8") as handle:
            for step in steps:
                if step in omitted_steps:
                    continue
                response = f"response-{step}"
                unknown = step in unknown_steps
                success = step in successful
                checkpoint = trajectory / f"step_{step:06d}.pt"
                handle.write(
                    json.dumps(
                        {
                            "format": "stage1-behavior-label",
                            "version": 1,
                            "case_id": "case-a",
                            "pair_id": "pair-a",
                            "step": step,
                            "checkpoint_path": (
                                f"runs/case-a/trajectory/step_{step:06d}.pt"
                            ),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            "experiment_fingerprint": experiment_fingerprint,
                            "response": response,
                            "response_sha256": response_sha256(response),
                            "label_status": "unknown" if unknown else "ok",
                            "refusal_label": None if unknown else not success,
                            "compliance_label": None if unknown else success,
                            "jailbreak_success": None if unknown else success,
                        }
                    )
                    + "\n"
                )
        return summary, labels

    def test_attach_uses_selected_step_semantic_label_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_val"],
                    "category": ["cyber"],
                    "benign_text": ["benign"],
                    "harmful_text": ["harmful"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(root)

            attached = attach_attack_outputs(
                source,
                summary,
                labels,
                root / "trajectory_manifest.csv",
                exclusions_path=root / "exclusions.csv",
                project_root=root,
            )

            self.assertEqual(len(attached), 1)
            self.assertEqual(attached.loc[0, "selected_attack_step"], 1)
            self.assertEqual(attached.loc[0, "jailbreak_response"], "response-1")
            self.assertTrue(bool(attached.loc[0, "jailbreak_success"]))
            self.assertEqual(
                attached.loc[0, "jailbreak_audio_path"],
                "runs/case-a/adversarial.wav",
            )
            self.assertEqual(
                attached.loc[0, "trajectory_path"],
                "runs/case-a/trajectory/index.json",
            )
            self.assertEqual(
                attached.loc[0, "experiment_fingerprint"],
                json.loads((root / "runs/case-a/run.json").read_text())["budget"][
                    "experiment_fingerprint"
                ],
            )


    def test_attach_semantic_selection_ignores_lower_loss_refusal(self):
        import soundfile as sf
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_train"],
                    "stage1_role": ["probe_candidate"],
                    "category": ["cyber"],
                    "benign_text": ["benign"],
                    "harmful_text": ["harmful"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(
                root,
                total_steps=2,
                history_selected_step=2,
                losses={0: 0.75, 1: 0.5, 2: 0.25},
                successful_steps=(0, 1),
                tensor_checkpoints=True,
            )

            attached = attach_attack_outputs(
                source,
                summary,
                labels,
                root / "semantic_attached.csv",
                project_root=root,
                selection_policy="semantic-success-lowest-loss",
                selected_audio_dir=root / "selected_audio",
                selected_audio_sample_rate=16_000,
            )

            self.assertEqual(len(attached), 1)
            self.assertEqual(attached.loc[0, "selected_attack_step"], 1)
            self.assertEqual(attached.loc[0, "history_selected_attack_step"], 2)
            self.assertEqual(
                attached.loc[0, "attack_selection_policy"],
                "semantic-success-lowest-loss",
            )
            self.assertEqual(attached.loc[0, "selected_attack_loss"], 0.5)
            self.assertEqual(attached.loc[0, "semantic_successful_steps"], 2)
            self.assertEqual(attached.loc[0, "jailbreak_response"], "response-1")
            self.assertTrue(bool(attached.loc[0, "jailbreak_success"]))
            self.assertTrue(
                attached.loc[0, "selected_checkpoint_path"].endswith(
                    "trajectory/step_000001.pt"
                )
            )

            audio_path = root / attached.loc[0, "jailbreak_audio_path"]
            waveform, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=False
            )
            checkpoint = torch.load(
                root / attached.loc[0, "selected_checkpoint_path"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(sample_rate, 16_000)
            self.assertTrue(
                torch.equal(
                    torch.from_numpy(waveform),
                    checkpoint["adversarial_wav"].squeeze(0),
                )
            )
            self.assertEqual(
                attached.loc[0, "jailbreak_audio_sha256"],
                sha256_file(audio_path),
            )

    def test_semantic_selection_requires_complete_known_success_labels(self):
        scenarios = (
            (
                {"omitted_steps": (0,), "successful_steps": (1,)},
                "incomplete_behavior_labels",
            ),
            (
                {"unknown_steps": (0,), "successful_steps": (1,)},
                "unknown_behavior_labels",
            ),
            ({"successful_steps": ()}, "no_semantic_jailbreak"),
        )
        for fixture_kwargs, expected_reason in scenarios:
            with self.subTest(reason=expected_reason):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "source.csv"
                    pd.DataFrame(
                        {
                            "pair_id": ["pair-a"],
                            "measurement_split": ["measurement_train"],
                            "stage1_role": ["probe_candidate"],
                            "category": ["cyber"],
                        }
                    ).to_csv(source, index=False)
                    summary, labels = self._write_attack_artifacts(
                        root,
                        losses={0: 1.0, 1: 0.5},
                        tensor_checkpoints=True,
                        **fixture_kwargs,
                    )
                    attached = attach_attack_outputs(
                        source,
                        summary,
                        labels,
                        root / "semantic_attached.csv",
                        exclusions_path=root / "exclusions.csv",
                        project_root=root,
                        selection_policy="semantic-success-lowest-loss",
                    )
                    self.assertTrue(attached.empty)
                    self.assertEqual(
                        pd.read_csv(root / "exclusions.csv").loc[
                            0, "exclusion_reason"
                        ],
                        expected_reason,
                    )


    def test_nonselected_unknown_or_missing_label_does_not_drop_pair(self):
        for unknown_steps, omitted_steps in (((0,), ()), ((), (0,))):
            with self.subTest(unknown=unknown_steps, omitted=omitted_steps):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "source.csv"
                    pd.DataFrame(
                        {
                            "pair_id": ["pair-a"],
                            "measurement_split": ["measurement_val"],
                            "category": ["cyber"],
                        }
                    ).to_csv(source, index=False)
                    summary, labels = self._write_attack_artifacts(
                        root,
                        unknown_steps=unknown_steps,
                        omitted_steps=omitted_steps,
                    )
                    attached = attach_attack_outputs(
                        source,
                        summary,
                        labels,
                        root / "output.csv",
                        project_root=root,
                    )
                    self.assertEqual(attached["pair_id"].tolist(), ["pair-a"])

    def test_selected_unknown_label_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_val"],
                    "category": ["cyber"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(root, unknown_steps=(1,))
            attached = attach_attack_outputs(
                source,
                summary,
                labels,
                root / "output.csv",
                exclusions_path=root / "exclusions.csv",
                project_root=root,
            )
            self.assertTrue(attached.empty)
            excluded = pd.read_csv(root / "exclusions.csv")
            self.assertEqual(
                excluded.loc[0, "exclusion_reason"],
                "unknown_selected_behavior_label",
            )

    def test_behavior_label_schema_and_checkpoint_hash_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_val"],
                    "category": ["cyber"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["compliance_label"] = True
            records[0]["jailbreak_success"] = True
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "both refusal and compliant"):
                attach_attack_outputs(
                    source, summary, labels, root / "output.csv", project_root=root
                )

            summary, labels = self._write_attack_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["checkpoint_sha256"] = "0" * 64
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "checkpoint SHA-256"):
                attach_attack_outputs(
                    source, summary, labels, root / "output.csv", project_root=root
                )

    def test_attach_rejects_trajectory_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_val"],
                    "category": ["cyber"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(
                root, trajectory_pair="wrong-pair"
            )
            with self.assertRaisesRegex(Stage1ManifestError, "Trajectory pair_id"):
                attach_attack_outputs(
                    source,
                    summary,
                    labels,
                    root / "output.csv",
                    project_root=root,
                )

    @staticmethod
    def _write_clean_artifacts(
        root: Path,
        *,
        unknown_state=None,
        prompt_tamper=False,
        distinct_clean_audio=False,
    ):
        audio = root / "audio"
        audio.mkdir()
        benign_audio = audio / "benign.wav"
        harmful_audio = audio / "harmful.wav"
        legacy_clean_audio = audio / "legacy-clean.wav"
        benign_audio.write_bytes(b"benign-wave")
        harmful_audio.write_bytes(b"harmful-wave")
        legacy_clean_audio.write_bytes(b"legacy-clean-wave")
        source = root / "probe_candidates.csv"
        source_data = {
            "pair_id": ["pair-a"],
            "source": ["fixture"],
            "stratum": ["cyber"],
            "measurement_split": ["measurement_train"],
            "stage1_role": ["probe_candidate"],
            "benign_text": ["benign prompt"],
            "harmful_text": ["harmful prompt"],
            "benign_audio_path": ["audio/benign.wav"],
            "harmful_audio_path": ["audio/harmful.wav"],
        }
        if distinct_clean_audio:
            source_data["clean_audio_path"] = ["audio/legacy-clean.wav"]
        pd.DataFrame(source_data).to_csv(source, index=False)
        source_digest = sha256_file(source)
        labels = root / "clean_labels.jsonl"
        with labels.open("w", encoding="utf-8") as handle:
            for state, prompt, audio_path, response, refused in (
                ("X_B", "benign prompt", benign_audio, "benign answer", False),
                ("X_H", "harmful prompt", harmful_audio, "I refuse", True),
            ):
                status = "unknown" if state == unknown_state else "ok"
                stored_prompt = prompt + " changed" if state == "X_B" and prompt_tamper else prompt
                handle.write(
                    json.dumps(
                        {
                            "format": "stage1-clean-behavior",
                            "version": 1,
                            "pair_id": "pair-a",
                            "state": state,
                            "source_manifest_path": str(source),
                            "source_manifest_sha256": source_digest,
                            "prompt": stored_prompt,
                            "prompt_sha256": response_sha256(stored_prompt),
                            "audio_path": str(audio_path),
                            "audio_sha256": sha256_file(audio_path),
                            "response": response,
                            "response_sha256": response_sha256(response),
                            "generation_status": "ok",
                            "label_status": status,
                            "refusal_label": None if status == "unknown" else refused,
                        }
                    )
                    + "\n"
                )
        return source, labels

    def test_attach_clean_outputs_binds_prompt_audio_and_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root)
            attached = attach_clean_outputs(
                source,
                labels,
                root / "clean_attached.csv",
                exclusions_path=root / "exclusions.csv",
                project_root=root,
            )
            self.assertEqual(attached["pair_id"].tolist(), ["pair-a"])
            self.assertEqual(attached.loc[0, "benign_response"], "benign answer")
            self.assertFalse(bool(attached.loc[0, "benign_refused"]))
            self.assertEqual(attached.loc[0, "clean_response"], "I refuse")
            self.assertTrue(bool(attached.loc[0, "clean_refused"]))
            self.assertEqual(
                attached.loc[0, "clean_audio_path"], "audio/harmful.wav"
            )

    def test_attach_clean_prefers_harmful_audio_over_distinct_clean_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(
                root, distinct_clean_audio=True
            )
            attached = attach_clean_outputs(
                source,
                labels,
                root / "clean_attached.csv",
                project_root=root,
            )

            self.assertEqual(attached["pair_id"].tolist(), ["pair-a"])
            self.assertEqual(
                attached.loc[0, "clean_audio_path"], "audio/harmful.wav"
            )

    def test_label_sidecar_format_and_version_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame(
                {
                    "pair_id": ["pair-a"],
                    "measurement_split": ["measurement_val"],
                    "category": ["cyber"],
                }
            ).to_csv(source, index=False)
            summary, labels = self._write_attack_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["version"] = "1"
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "version"):
                attach_attack_outputs(
                    source, summary, labels, root / "output.csv", project_root=root
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["format"] = "stage1-behavior-label"
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "format"):
                attach_clean_outputs(
                    source, labels, root / "output.csv", project_root=root
                )

    def test_clean_generation_and_label_status_contract_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["generation_status"] = "pending"
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "generation_status"):
                attach_clean_outputs(
                    source, labels, root / "output.csv", project_root=root
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root)
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[0]["generation_status"] = "error"
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Stage1ManifestError, "status=ok"):
                attach_clean_outputs(
                    source, labels, root / "output.csv", project_root=root
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root, unknown_state="X_H")
            records = [json.loads(line) for line in labels.read_text().splitlines()]
            records[1]["generation_status"] = "error"
            labels.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            attached = attach_clean_outputs(
                source,
                labels,
                root / "output.csv",
                exclusions_path=root / "exclusions.csv",
                project_root=root,
            )
            self.assertTrue(attached.empty)
            self.assertEqual(
                pd.read_csv(root / "exclusions.csv").loc[0, "exclusion_reason"],
                "unknown_clean_label:X_H",
            )

    def test_attach_clean_unknown_is_excluded_and_content_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root, unknown_state="X_H")
            attached = attach_clean_outputs(
                source,
                labels,
                root / "clean_attached.csv",
                exclusions_path=root / "exclusions.csv",
                project_root=root,
            )
            self.assertTrue(attached.empty)
            excluded = pd.read_csv(root / "exclusions.csv")
            self.assertEqual(
                excluded.loc[0, "exclusion_reason"], "unknown_clean_label:X_H"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels = self._write_clean_artifacts(root, prompt_tamper=True)
            with self.assertRaisesRegex(Stage1ManifestError, "prompt content mismatch"):
                attach_clean_outputs(
                    source,
                    labels,
                    root / "clean_attached.csv",
                    project_root=root,
                )

    def test_finalize_keeps_only_strict_train_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "attached.csv"
            base = {
                "source": "fixture",
                "stratum": "cyber",
                "measurement_split": "measurement_train",
                "stage1_role": "probe_candidate",
                "benign_response": "benign answer",
                "benign_refused": False,
                "clean_response": "I refuse",
                "clean_refused": True,
                "jailbreak_response": "harmful answer",
                "benign_audio_path": "audio/b.wav",
                "clean_audio_path": "audio/h.wav",
                "jailbreak_audio_path": "audio/j.wav",
            }
            rows = [
                {
                    **base,
                    "pair_id": "pair-good",
                    "benign_text": "benign good",
                    "harmful_text": "harmful good",
                    "jailbreak_success": True,
                },
                {
                    **base,
                    "pair_id": "pair-failed",
                    "benign_text": "benign failed",
                    "harmful_text": "harmful failed",
                    "jailbreak_success": False,
                },
            ]
            pd.DataFrame(rows).to_csv(source, index=False)

            selected = finalize_measurement_manifest(
                source,
                root / "measurement.csv",
                root / "exclusions.csv",
                project_root=root,
            )

            self.assertEqual(selected["pair_id"].tolist(), ["pair-good"])
            excluded = pd.read_csv(root / "exclusions.csv")
            self.assertEqual(excluded.loc[0, "pair_id"], "pair-failed")
            self.assertIn("jailbreak_failed", excluded.loc[0, "exclusion_reason"])

            leaked = pd.DataFrame(rows[:1])
            leaked["measurement_split"] = "measurement_val"
            leaked.to_csv(root / "leaked.csv", index=False)
            with self.assertRaisesRegex(Stage1ManifestError, "cannot enter probe"):
                finalize_measurement_manifest(
                    root / "leaked.csv",
                    root / "unused.csv",
                    root / "unused-exclusions.csv",
                    project_root=root,
                )

            legacy = pd.DataFrame(rows[:1]).drop(columns="jailbreak_success")
            legacy["attack_success"] = True
            legacy.to_csv(root / "legacy.csv", index=False)
            with self.assertRaisesRegex(Stage1ManifestError, "legacy attack_success"):
                finalize_measurement_manifest(
                    root / "legacy.csv",
                    root / "unused.csv",
                    root / "unused-exclusions.csv",
                    project_root=root,
                )

    def test_cli_help_is_offline(self):
        completed = subprocess.run(
            [sys.executable, "-m", "data.prepare_stage1_manifests", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout.casefold())


if __name__ == "__main__":
    unittest.main()
