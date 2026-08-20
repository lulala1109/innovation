"""Offline tests for attack-agnostic behavioral evaluators."""

import json
import unittest
from unittest.mock import patch

from core.llm_backend import LLMBackendError
from evaluation import behavior


class _FakeBackend:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.client = object()
        self.provider = kwargs.get("provider") or "openai"
        self.model = kwargs.get("model") or "gpt-4o-mini"
        self.base_url = kwargs.get("base_url")
        self.response = json.dumps({
            "overall_reasoning": "The response addresses the request.",
            "refusal": 0,
            "convincingness": 4,
            "specificity": 5,
        })
        self.error = None
        self.last_request = None

    def create_json_completion(self, messages, *, max_tokens=500):
        self.last_request = {"messages": messages, "max_tokens": max_tokens}
        if self.error is not None:
            raise self.error
        return self.response

    def public_config(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }


class StrongRejectEvaluatorTests(unittest.TestCase):
    def setUp(self):
        backend_patch = patch.object(behavior, "LLMBackend", _FakeBackend)
        backend_patch.start()
        self.addCleanup(backend_patch.stop)

    def test_valid_json_uses_original_score_formula(self):
        evaluator = behavior.StrongRejectEvaluator()

        result = evaluator.evaluate("request", "response")

        self.assertEqual(result["refusal"], 0.0)
        self.assertEqual(result["convincingness"], 4.0)
        self.assertEqual(result["specificity"], 5.0)
        self.assertAlmostEqual(result["score"], 0.9)
        self.assertEqual(evaluator.backend.last_request["max_tokens"], 800)
        prompt = evaluator.backend.last_request["messages"][0]["content"]
        self.assertIn("valid JSON object", prompt)

    def test_invalid_json_schema_fails_fast(self):
        evaluator = behavior.StrongRejectEvaluator()
        invalid_responses = [
            "",
            "not json",
            json.dumps({
                "overall_reasoning": "wrong type",
                "refusal": 0,
                "convincingness": "4",
                "specificity": 5,
            }),
            json.dumps({
                "overall_reasoning": "extra key",
                "refusal": 0,
                "convincingness": 4,
                "specificity": 5,
                "unexpected": True,
            }),
        ]

        for response in invalid_responses:
            with self.subTest(response=response):
                evaluator.backend.response = response
                with self.assertRaisesRegex(ValueError, "invalid JSON"):
                    evaluator.evaluate("request", "response")

    def test_provider_error_propagates(self):
        evaluator = behavior.StrongRejectEvaluator()
        evaluator.backend.error = LLMBackendError("offline API failure")

        with self.assertRaises(LLMBackendError):
            evaluator.evaluate("request", "response")

    def test_public_config_contains_no_credentials(self):
        evaluator = behavior.StrongRejectEvaluator(
            api_key="private-test-key",
            provider="qwen",
            model="qwen-plus",
            base_url="https://example.invalid/compatible-mode/v1",
        )

        self.assertEqual(evaluator.public_config(), {
            "provider": "qwen",
            "model": "qwen-plus",
            "base_url": "https://example.invalid/compatible-mode/v1",
        })
        self.assertNotIn("api_key", evaluator.public_config())
        self.assertEqual(
            evaluator.backend.init_kwargs["provider_env"],
            "STRONGREJECT_PROVIDER",
        )


class EvaluateResponseTests(unittest.TestCase):
    def test_optional_evaluator_errors_propagate(self):
        class StrongReject:
            def evaluate(self, target_text, response):
                return {"score": 0.5}

        class FailingEvaluator:
            def evaluate(self, target_text, response):
                raise RuntimeError("local evaluator failed")

        with self.assertRaisesRegex(RuntimeError, "local evaluator failed"):
            behavior.evaluate_response(
                "request",
                "response",
                strongreject=StrongReject(),
                llamaguard=FailingEvaluator(),
            )


if __name__ == "__main__":
    unittest.main()
