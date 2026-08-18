"""Offline tests for artifact-backed audio quality and injected-ASR utility."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.evaluate_audio_quality import (
    AudioBudgetViolationError,
    SPR_DEFINITION,
    evaluate_audio_quality,
    evaluate_run_audio_quality,
    verify_linf_budget,
)
from evaluation.task_utility import (
    character_error_rate,
    edit_distance,
    evaluate_run_task_utility,
    evaluate_task_utility_batch,
    word_error_rate,
)


class AudioQualityAndUtilityTests(unittest.TestCase):
    def _case(self, root: Path, eps: float = 0.05) -> Path:
        case = root / "case"
        case.mkdir()
        (case / "clean.wav").touch()
        (case / "adversarial.wav").touch()
        run = {
            "case_id": "case-1",
            "pair_id": "pair-1",
            "method": "standard",
            "model": "toy",
            "harmful_text": "hello world",
            "attack_success": True,
            "budget": {"norm": "linf", "eps": eps, "seed": 7},
            "artifacts": {
                "input_audio": "clean.wav",
                "adversarial_audio": "adversarial.wav",
            },
        }
        path = case / "run.json"
        path.write_text(json.dumps(run), encoding="utf-8")
        return path

    @staticmethod
    def _loader(path):
        clean = torch.tensor([[0.2, -0.2, 0.1, -0.1]])
        if Path(path).name == "adversarial.wav":
            clean = clean + 0.05
        return clean, 16_000

    def test_reloads_audio_rechecks_budget_and_defines_spr(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._case(Path(directory))
            result = evaluate_run_audio_quality(run, audio_loader=self._loader)
            self.assertTrue(result["budget_verification"]["valid"])
            self.assertAlmostEqual(
                result["budget_verification"]["measured_linf"], 0.05, places=6
            )
            self.assertEqual(
                result["metrics"]["spr_db"], result["metrics"]["snr_db"]
            )
            self.assertEqual(result["spr_definition"], SPR_DEFINITION)
            self.assertIn("perturbation_l2", result["metrics"])
            self.assertIn("perturbation_rms", result["metrics"])

    def test_strict_budget_and_batch_failure_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._case(Path(directory), eps=0.01)
            with self.assertRaises(AudioBudgetViolationError):
                evaluate_run_audio_quality(
                    run, audio_loader=self._loader, strict_budget=True
                )
            batch = evaluate_audio_quality(
                run, audio_loader=self._loader, strict_budget=True
            )
            self.assertEqual(batch["failure_count"], 1)
            self.assertEqual(batch["summary"]["total"], 0)

    def test_budget_validation_rejects_invalid_waveforms(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            verify_linf_budget(
                torch.tensor([0.0, float("nan")]), torch.zeros(2), 0.1
            )

    def test_stdlib_error_rates_and_injected_transcriber_batch(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(
            word_error_rate("one two three", "one four three"), 1 / 3
        )
        self.assertAlmostEqual(character_error_rate("abc", "adc"), 1 / 3)

        def transcriber(waveform, _sample_rate):
            return "hello world" if waveform.mean() < 0.05 else "hello word"

        with tempfile.TemporaryDirectory() as directory:
            run = self._case(Path(directory))
            result = evaluate_run_task_utility(
                run, transcriber=transcriber, audio_loader=self._loader
            )
            self.assertEqual(result["clean"]["wer"], 0.0)
            self.assertEqual(result["adversarial"]["wer"], 0.5)
            batch = evaluate_task_utility_batch(
                run, transcriber=transcriber, audio_loader=self._loader
            )
            self.assertEqual(batch["summary"]["completed"], 1)
            self.assertEqual(batch["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
