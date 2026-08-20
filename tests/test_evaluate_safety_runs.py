"""Offline tests for generic safety-run loading and aggregation."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_safety_runs import (
    evaluate_safety_runs,
    load_safety_cases,
)


PUBLIC_CONFIG = {
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
}
SECRET = "must-not-be-written"


class _FakeStrongReject:
    def __init__(self):
        self.calls = []

    def public_config(self):
        return dict(PUBLIC_CONFIG)

    def evaluate(self, target_text, response):
        self.calls.append((target_text, response))
        score = {
            "response one": 0.9,
            "response two": 0.1,
            "response three": 0.5,
        }[response]
        return {
            "overall_reasoning": "offline",
            "refusal": float(score < 0.25),
            "convincingness": 4.0,
            "specificity": 4.0,
            "score": score,
            "raw_response": "offline",
        }


class _FakeLlamaGuard:
    def evaluate(self, target_text, response):
        return {"unsafe": response != "response two", "category": None}


class _FakeJailbreakEval:
    preset = "offline-preset"

    def evaluate(self, target_text, response):
        return {"jailbroken": response == "response one"}


def _write_case(root: Path, directory: str, filename: str, payload):
    case_dir = root / directory
    case_dir.mkdir()
    (case_dir / filename).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


class SafetyRunEvaluationTests(unittest.TestCase):
    def _make_batch(self, root: Path):
        _write_case(root, "case_001", "run.json", {
            "case_id": "case-001",
            "pair_id": "pair-001",
            "method": "standard_pgd",
            "model": "qwen-3b",
            "stratum": "weapons",
            "harmful_text": "request one",
            "adversarial_response": "response one",
            "attack_success": True,
            "budget": {"eps": 0.1},
            "artifacts": {"audio": "adversarial.wav"},
        })
        _write_case(root, "case_002", "config.json", {
            "case_id": "case-002",
            "pair_id": "pair-002",
            "method": "layer_adaptive_pgd",
            "model": "qwen-3b",
            "stratum": "weapons",
            "prompt": "request two",
            "response": "response two",
            "attack_success": False,
        })
        _write_case(root, "case_003", "run.json", {
            "case_id": "case-003",
            "pair_id": "pair-003",
            "method": "standard_pgd",
            "stratum": "privacy",
            "prompt": "request three",
            "adversarial_response": "response three",
        })

    def test_generic_aliases_and_grouped_summaries(self):
        evaluator = _FakeStrongReject()
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            self._make_batch(batch)

            results = evaluate_safety_runs(
                batch,
                runs=2,
                strongreject_evaluator=evaluator,
                llamaguard_evaluator=_FakeLlamaGuard(),
                jailbreakeval_evaluator=_FakeJailbreakEval(),
            )

        self.assertEqual(results["metadata"]["num_cases"], 3)
        self.assertEqual(
            results["metadata"]["strongreject_backend"],
            PUBLIC_CONFIG,
        )
        self.assertNotIn(SECRET, json.dumps(results))
        self.assertEqual(len(evaluator.calls), 6)
        self.assertEqual(
            results["summary"]["by_method"]["standard_pgd"]["count"],
            2,
        )
        self.assertEqual(
            results["summary"]["by_stratum"]["weapons"]["count"],
            2,
        )
        self.assertAlmostEqual(
            results["summary"]["overall"]["strongreject"]["mean_score"],
            0.5,
        )
        self.assertEqual(
            results["summary"]["overall"]["strongreject"]
            ["attack_success_count"],
            2,
        )
        self.assertEqual(
            results["summary"]["overall"]["llamaguard"]["unsafe_count"],
            2,
        )
        self.assertEqual(
            results["summary"]["overall"]["jailbreakeval"]
            ["jailbroken_count"],
            1,
        )
        self.assertEqual(results["cases"][1]["harmful_text"], "request two")
        self.assertEqual(results["cases"][1]["adversarial_response"], "response two")

    def test_partial_evaluation_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            self._make_batch(batch)
            results = evaluate_safety_runs(
                batch,
                max_samples=1,
                strongreject_evaluator=_FakeStrongReject(),
            )

        self.assertEqual(results["metadata"]["num_cases"], 1)
        self.assertEqual(results["metadata"]["total_cases_in_batch"], 3)
        self.assertTrue(results["metadata"]["is_partial"])

    def test_invalid_json_and_missing_required_fields_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            case_dir = batch / "case_bad"
            case_dir.mkdir()
            (case_dir / "run.json").write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_safety_cases(batch)

        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _write_case(batch, "case_bad", "run.json", {
                "case_id": "case-bad",
                "method": "standard_pgd",
                "stratum": "test",
                "harmful_text": "request",
                "adversarial_response": "response",
            })
            with self.assertRaisesRegex(ValueError, "pair_id"):
                load_safety_cases(batch)

    def test_conflicting_aliases_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _write_case(batch, "case_bad", "run.json", {
                "case_id": "case-bad",
                "pair_id": "pair-bad",
                "method": "standard_pgd",
                "stratum": "test",
                "harmful_text": "request A",
                "prompt": "request B",
                "adversarial_response": "response",
            })
            with self.assertRaisesRegex(ValueError, "conflicting alias"):
                load_safety_cases(batch)

    def test_evaluator_errors_propagate(self):
        class FailingStrongReject(_FakeStrongReject):
            def evaluate(self, target_text, response):
                raise RuntimeError("judge unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            self._make_batch(batch)
            with self.assertRaisesRegex(RuntimeError, "judge unavailable"):
                evaluate_safety_runs(
                    batch,
                    strongreject_evaluator=FailingStrongReject(),
                )


if __name__ == "__main__":
    unittest.main()
