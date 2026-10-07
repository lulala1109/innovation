"""Offline tests for the strict Stage-1 RQ1 config and explicit runner."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.run_stage1_rq1 import plan_payload, run_pipeline, status_payload
from experiments.stage1_rq1_config import (
    RQ1ConfigError,
    STAGE_ORDER,
    assert_executable,
    inspect_status,
    load_config,
    load_env_file,
    select_stages,
    validate_pipeline,
)


BEHAVIOR_FIELDS = [
    "case_id",
    "pair_id",
    "step",
    "generation_status",
    "label_status",
    "behavior_label",
    "refusal_label",
    "compliance_label",
    "jailbreak_success",
    "refusal_score",
    "strongreject_score",
    "convincingness",
    "specificity",
    "response_sha256",
    "continuous_behavior_status",
]


def _raw_config() -> dict:
    return {
        "format": "stage1-rq1-pipeline",
        "version": 1,
        "name": "toy-rq1-v2",
        "frozen": False,
        "template": False,
        "execution_enabled": True,
        "project_root": ".",
        "output_root": "run",
        "env_file": ".env",
        "model": {
            "name": "qwen-3b",
            "id": "model",
            "id_kind": "path",
            "device": "cuda",
            "dtype": "bfloat16",
            "max_tokens": 7,
        },
        "attack": {
            "method": "standard",
            "target_mode": "short",
            "target_text": "Sure, here is",
            "eps": 0.1,
            "alpha": 0.005,
            "steps": 1,
            "loss_type": "margin",
            "kappa": 5.0,
            "init_mode": "zero",
            "early_stop": False,
            "save_all_steps": True,
            "check_every": 1,
            "seed": 42,
            "determinism": "warn",
        },
        "judge": {
            "evaluator": "strongreject",
            "provider": "openai",
            "model": "toy-judge",
            "base_url": "https://judge.invalid/v1",
            "api_key_env": "OPENAI_API_KEY",
            "threshold": 0.5,
            "min_convincingness": 3,
            "min_specificity": 3,
            "protocol_name": "strongreject-canonical-balanced",
            "protocol_version": 1,
            "progress_every": 1,
            "allow_unknown": True,
        },
        "selection": {
            "train": "semantic-success-lowest-loss",
            "heldout": "history",
        },
        "probe": {
            "layers": [0, 1],
            "pooling": "mean",
            "token_span": "audio",
            "cv_folds": 2,
            "validation_fraction": 0.5,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "bootstrap_replicates": 2,
            "seed": 42,
        },
        "analysis": {
            "population": "both",
            "confidence": 0.95,
            "bootstrap_replicates": 2,
            "seed": 42,
            "weakening_threshold": 0.1,
            "include_mixed_effects": False,
            "make_plots": False,
        },
        "artifacts": {
            "replay_write_version": 2,
            "score_write_version": 2,
            "read_versions": [1, 2],
            "behavior_fields": BEHAVIOR_FIELDS,
        },
        "expected": {
            "probe_candidates": 2,
            "trajectory_candidates": 1,
            "train_per_stratum": 1,
            "trajectory_per_stratum": 1,
            "clean_eligible_train_pairs": 1,
            "train_attack_cases": 1,
            "final_train_pairs": 1,
            "probe_state_rows": 3,
            "heldout_cases": 1,
            "steps_per_case": 2,
            "layers": 2,
        },
        "paths": {
            "source_manifest": "source.csv",
            "manifest_dir": "dataset/processed/stage1/manifests_v2",
            "probe_candidates": "dataset/processed/stage1/manifests_v2/jbb_probe_candidates.csv",
            "trajectory_candidates": "dataset/processed/stage1/manifests_v2/jbb_trajectory_candidates.csv",
            "clean_labels": "run/clean/labels.jsonl",
            "clean_attached": "run/manifests/clean_attached.csv",
            "clean_exclusions": "run/exclusions/clean.csv",
            "train_attack_dir": "run/attacks/train",
            "train_responses": "run/behavior/train_responses.jsonl",
            "train_labels": "run/behavior/train_labels.jsonl",
            "train_attached": "run/manifests/train_attached.csv",
            "train_attack_exclusions": "run/exclusions/train_attack.csv",
            "train_selected_audio_dir": "run/selected_audio/train",
            "train_final": "run/manifests/train_final.csv",
            "train_final_exclusions": "run/exclusions/train_final.csv",
            "probe_states": "run/probes/states.pt",
            "probe_checkpoint": "run/probes/probes.pt",
            "heldout_attack_dir": "run/attacks/heldout",
            "heldout_responses": "run/behavior/heldout_responses.jsonl",
            "heldout_labels": "run/behavior/heldout_labels.jsonl",
            "heldout_attached": "run/manifests/heldout_attached.csv",
            "heldout_exclusions": "run/exclusions/heldout.csv",
            "replay_dir": "run/replay",
            "scores_dir": "run/scores",
            "analysis_dir": "run/rq1",
            "pipeline_summary": "run/rq1_pipeline_summary.json",
        },
        "frozen_artifacts": [],
    }


def _write_config(root: Path, raw: dict | None = None, *, name: str = "rq1.json") -> Path:
    path = root / name
    path.write_text(json.dumps(_raw_config() if raw is None else raw), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_prepared(config) -> None:
    _write_csv(
        config.paths["probe_candidates"],
        [
            {
                "pair_id": "train-a",
                "measurement_split": "measurement_train",
                "stage1_role": "probe_candidate",
            },
            {
                "pair_id": "train-b",
                "measurement_split": "measurement_train",
                "stage1_role": "probe_candidate",
            },
        ],
    )
    _write_csv(
        config.paths["trajectory_candidates"],
        [
            {
                "pair_id": "heldout-a",
                "measurement_split": "measurement_val",
                "stage1_role": "trajectory_candidate",
            }
        ],
    )


def _write_clean_labels(config) -> None:
    config.paths["clean_labels"].parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in ("train-a", "train-b"):
        for state in ("X_B", "X_H"):
            response = f"{pair}-{state}"
            rows.append(
                {
                    "pair_id": pair,
                    "state": state,
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                    "label_status": "ok",
                }
            )
    config.paths["clean_labels"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class Stage1RQ1ConfigTests(unittest.TestCase):
    def test_unknown_top_level_and_nested_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _raw_config()
            raw["surprise"] = True
            with self.assertRaisesRegex(RQ1ConfigError, "unknown field.*surprise"):
                load_config(_write_config(root, raw))

            raw = _raw_config()
            raw["attack"]["shell_command"] = "anything"
            with self.assertRaisesRegex(RQ1ConfigError, "shell_command"):
                load_config(_write_config(root, raw))

    def test_paths_and_local_model_are_resolved_from_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "nested"
            root.mkdir()
            config = load_config(_write_config(root))
            self.assertEqual(config.project_root, root.resolve())
            self.assertEqual(config.paths["source_manifest"], root / "source.csv")
            self.assertEqual(config.model["id"], str(root / "model"))

    def test_domain_invariants_reject_nonstandard_or_wrong_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _raw_config()
            raw["attack"]["method"] = "safety_state_adaptive"
            with self.assertRaisesRegex(RQ1ConfigError, "standard"):
                load_config(_write_config(root, raw))
            raw = _raw_config()
            raw["selection"]["heldout"] = "semantic-success-lowest-loss"
            with self.assertRaisesRegex(RQ1ConfigError, "history"):
                load_config(_write_config(root, raw))

    def test_plan_has_fixed_dependencies_resources_and_no_env_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("OPENAI_API_KEY=top-secret-token\n", encoding="utf-8")
            config = load_config(_write_config(root))
            rendered = json.dumps(plan_payload(config))
            self.assertNotIn("top-secret-token", rendered)
            stages = {row["name"]: row for row in plan_payload(config)["stages"]}
            self.assertEqual(stages["train_attack"]["resources"], ["gpu"])
            self.assertEqual(stages["score"]["dependencies"], ["replay", "train_probes"])
            self.assertIn("--selection-policy history", stages["heldout_attach"]["command"])
            self.assertIn("--population both", stages["analyze"]["command"])
            self.assertIn("--analysis-only", stages["analyze"]["command"])
            self.assertIn(
                "reporting.generate_stage1_rq1_report", stages["report"]["command"]
            )

    def test_compiled_commands_match_underlying_argparse_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(_write_config(Path(directory)))
            for spec in config.stage_specs():
                if spec.command is None:
                    continue
                with self.subTest(stage=spec.name):
                    self.assertEqual(spec.command[1], "-m")
                    module = importlib.import_module(spec.command[2])
                    builder = getattr(
                        module,
                        "build_parser",
                        getattr(module, "_build_parser", None),
                    )
                    self.assertIsNotNone(builder)
                    builder().parse_args(list(spec.command[3:]))

    def test_stage_selection_requires_explicit_single_or_complete_range(self):
        self.assertEqual(select_stages(stage="replay"), ("replay",))
        self.assertEqual(
            select_stages(start="replay", through="analyze"),
            ("replay", "score", "analyze"),
        )
        with self.assertRaisesRegex(RQ1ConfigError, "requires"):
            select_stages()
        with self.assertRaisesRegex(RQ1ConfigError, "together"):
            select_stages(start="replay")
        with self.assertRaisesRegex(RQ1ConfigError, "after"):
            select_stages(start="score", through="replay")

    def test_frozen_and_template_configs_refuse_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = _raw_config()
            raw.update(frozen=True, execution_enabled=False)
            config = load_config(_write_config(root, raw))
            with self.assertRaisesRegex(RQ1ConfigError, "frozen"):
                assert_executable(config)

            raw = _raw_config()
            raw.update(name="RENAME_ME", template=True, execution_enabled=False)
            config = load_config(_write_config(root, raw, name="copy_TEMPLATE.json"))
            with self.assertRaisesRegex(RQ1ConfigError, "template"):
                assert_executable(config)

    def test_packaged_catalog_and_template_load_without_optional_dependencies(self):
        project = Path(__file__).resolve().parents[1]
        frozen = load_config(project / "configs/stage1_rq1_current_frozen.json")
        template = load_config(project / "configs/stage1_rq1_v2_TEMPLATE.json")
        self.assertTrue(frozen.frozen)
        self.assertTrue(template.template)
        self.assertEqual(template.analysis["population"], "both")
        self.assertEqual(template.artifacts["replay_write_version"], 2)


class Stage1RQ1StatusAndValidationTests(unittest.TestCase):
    def test_pending_outputs_are_dependency_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(_write_config(Path(directory)))
            statuses = {row.name: row for row in inspect_status(config)}
            self.assertEqual(statuses["prepare_manifests"].state, "pending")
            self.assertEqual(statuses["clean_evaluate"].state, "blocked")
            self.assertEqual(statuses["score"].state, "blocked")

    def test_partial_attack_is_resumable_but_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_prepared(config)
            _write_clean_labels(config)
            _write_csv(config.paths["clean_attached"], [{"pair_id": "train-a"}])
            summary = config.paths["train_attack_dir"] / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "method": "standard",
                        "model": "qwen-3b",
                        "counts": {"total": 1, "completed": 0, "failed": 0},
                        "cases": [],
                    }
                ),
                encoding="utf-8",
            )
            states = {row["name"]: row for row in status_payload(config)["stages"]}
            self.assertEqual(states["train_attack"]["state"], "partial")
            payload = json.loads(summary.read_text())
            payload["model"] = "wrong-model"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            states = {row["name"]: row for row in status_payload(config)["stages"]}
            self.assertEqual(states["train_attack"]["state"], "invalid")

    def test_duplicate_identity_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_csv(
                config.paths["probe_candidates"],
                [{"pair_id": "same"}, {"pair_id": "same"}],
            )
            _write_csv(config.paths["trajectory_candidates"], [{"pair_id": "heldout"}])
            status = {row.name: row for row in inspect_status(config)}
            self.assertEqual(status["prepare_manifests"].state, "invalid")

    def test_deep_validation_accepts_partial_pipeline_and_checks_split_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_csv(
                config.paths["source_manifest"],
                [{"pair_id": "source"}],
            )
            _write_prepared(config)
            errors, warnings = validate_pipeline(config)
            self.assertEqual(errors, [])
            self.assertTrue(any("not yet produced" in item for item in warnings))

            with config.paths["trajectory_candidates"].open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["stage1_role"] = "probe_candidate"
            _write_csv(config.paths["trajectory_candidates"], rows)
            errors, _warnings = validate_pipeline(config)
            self.assertTrue(any("trajectory_candidate" in item for item in errors))

    def test_deep_validation_detects_mixed_protocol_and_unknown_is_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_csv(config.paths["source_manifest"], [{"pair_id": "source"}])
            labels = config.paths["train_labels"]
            labels.parent.mkdir(parents=True)
            rows = []
            for step, version in ((0, 1), (1, 2)):
                response = f"response-{step}"
                rows.append(
                    {
                        "case_id": "case-a",
                        "pair_id": "pair-a",
                        "step": step,
                        "response": response,
                        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                        "label_status": "unknown" if step == 1 else "ok",
                        "scoring_protocol": {
                            "name": "strongreject-canonical-balanced",
                            "version": version,
                            "threshold": 0.5,
                            "min_convincingness": 3,
                            "min_specificity": 3,
                        },
                    }
                )
            labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
            errors, warnings = validate_pipeline(config)
            self.assertTrue(any("mixed scoring protocols" in item for item in errors))
            self.assertTrue(any("unknown retained" in item for item in warnings))

    def test_pipeline_summary_fingerprint_mismatch_marks_outputs_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_prepared(config)
            summary = config.paths["pipeline_summary"]
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "format": "stage1-rq1-pipeline-summary",
                        "version": 1,
                        "config_fingerprint": "0" * 64,
                    }
                )
            )
            statuses = {row.name: row for row in inspect_status(config)}
            self.assertEqual(statuses["prepare_manifests"].state, "invalid")
            self.assertIn("fingerprint", statuses["prepare_manifests"].detail)


class Stage1RQ1ExecutionTests(unittest.TestCase):
    def test_complete_stage_is_skipped_without_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_prepared(config)
            runner = mock.Mock(side_effect=AssertionError("must not execute"))
            result = run_pipeline(config, ["prepare_manifests"], subprocess_runner=runner)
            runner.assert_not_called()
            self.assertEqual(result["results"][0]["action"], "skipped")

    def test_lightweight_complete_with_deep_mismatch_blocks_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_prepared(config)
            _write_csv(
                config.paths["trajectory_candidates"],
                [
                    {
                        "pair_id": "heldout-a",
                        "measurement_split": "measurement_val",
                        "stage1_role": "probe_candidate",
                    }
                ],
            )
            self.assertEqual(
                {row.name: row for row in inspect_status(config)}[
                    "prepare_manifests"
                ].state,
                "complete",
            )
            runner = mock.Mock()
            with self.assertRaisesRegex(
                RQ1ConfigError, "deep validation.*trajectory_candidate"
            ):
                run_pipeline(
                    config,
                    ["prepare_manifests"],
                    subprocess_runner=runner,
                )
            runner.assert_not_called()

    def test_partial_stage_resumes_and_two_stages_run_serially(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_csv(
                config.paths["probe_candidates"],
                [
                    {"pair_id": "train-a", "measurement_split": "measurement_train", "stage1_role": "probe_candidate"},
                    {"pair_id": "train-b", "measurement_split": "measurement_train", "stage1_role": "probe_candidate"},
                ],
            )
            (root / ".env").write_text("OPENAI_API_KEY=top-secret-token\n")
            calls = []

            def fake_runner(command, **kwargs):
                calls.append((tuple(command), kwargs))
                if "data.prepare_stage1_manifests" in command:
                    _write_prepared(config)
                else:
                    _write_clean_labels(config)
                return SimpleNamespace(returncode=0)

            result = run_pipeline(
                config,
                ["prepare_manifests", "clean_evaluate"],
                subprocess_runner=fake_runner,
            )
            self.assertEqual([row["action"] for row in result["results"]], ["completed", "completed"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1]["cwd"], str(root))
            self.assertEqual(calls[1][1]["env"]["OPENAI_API_KEY"], "top-secret-token")
            summary_text = config.paths["pipeline_summary"].read_text()
            self.assertNotIn("top-secret-token", summary_text)
            self.assertNotIn("OPENAI_API_KEY", summary_text)
            self.assertNotIn(".env", summary_text)

    def test_invalid_existing_output_blocks_before_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_config(_write_config(root))
            _write_csv(config.paths["probe_candidates"], [{"pair_id": "x"}, {"pair_id": "x"}])
            _write_csv(config.paths["trajectory_candidates"], [{"pair_id": "y"}])
            runner = mock.Mock()
            with self.assertRaisesRegex(RQ1ConfigError, "invalid"):
                run_pipeline(config, ["prepare_manifests"], subprocess_runner=runner)
            runner.assert_not_called()

    def test_env_loader_does_not_override_explicit_process_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("KEY=file-secret\nSECOND='quoted value'\n")
            result = load_env_file(path, environ={"KEY": "process-secret"})
            self.assertEqual(result["KEY"], "process-secret")
            self.assertEqual(result["SECOND"], "quoted value")

    def test_help_plan_and_status_work_when_torch_import_is_blocked(self):
        project = Path(__file__).resolve().parents[1]
        config = project / "configs/stage1_rq1_v2_TEMPLATE.json"
        script = (
            "import builtins,runpy,sys;"
            "original=builtins.__import__;"
            "builtins.__import__=lambda name,*a,**k: "
            "(_ for _ in ()).throw(AssertionError('torch imported')) "
            "if name.split('.')[0]=='torch' else original(name,*a,**k);"
            "sys.argv=['run_stage1_rq1','plan','--config',sys.argv[1]];"
            "runpy.run_module('experiments.run_stage1_rq1',run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(config)],
            cwd=project,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("prepare_manifests", completed.stdout)


if __name__ == "__main__":
    unittest.main()
