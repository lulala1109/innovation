"""Offline tests for the shared LLM backend and attack judge."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import core.llm_backend as backend_module
from core.judge import LLMJudge
from core.llm_backend import LLMBackend, LLMBackendError


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.create_calls.append(kwargs)
        if self.owner.request_error is not None:
            raise self.owner.request_error
        if self.owner.malformed_response:
            return SimpleNamespace(choices=[])
        message = SimpleNamespace(content=self.owner.response_content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    """Minimal OpenAI SDK replacement that records calls without networking."""

    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.create_calls = []
        self.response_content = '{"reasoning":"valid","score":8}'
        self.request_error = None
        self.malformed_response = False
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(self),
        )
        self.__class__.instances.append(self)


class LLMBackendTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances.clear()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.openai = patch.object(backend_module, "OpenAI", _FakeOpenAI)
        self.environment.start()
        self.openai.start()

    def tearDown(self):
        self.openai.stop()
        self.environment.stop()

    def test_legacy_openai_environment_still_uses_default_backend(self):
        os.environ["OPENAI_API_KEY"] = "legacy-openai-key"

        backend = LLMBackend()

        self.assertEqual(backend.provider, "openai")
        self.assertEqual(backend.model, "gpt-4o-mini")
        self.assertIsNone(backend.base_url)
        self.assertEqual(backend.client.init_kwargs["api_key"],
                         "legacy-openai-key")
        self.assertNotIn("base_url", backend.client.init_kwargs)

    def test_each_provider_uses_its_key_model_and_base_url(self):
        cases = (
            (
                "openai",
                {"OPENAI_API_KEY": "openai-key"},
                "gpt-4o-mini",
                None,
                "openai-key",
            ),
            (
                "deepseek",
                {"DEEPSEEK_API_KEY": "deepseek-key"},
                "deepseek-v4-flash",
                "https://api.deepseek.com",
                "deepseek-key",
            ),
            (
                "qwen",
                {
                    "DASHSCOPE_API_KEY": "qwen-key",
                    "QWEN_BASE_URL": "https://qwen.example/compatible-mode/v1",
                },
                "qwen-plus",
                "https://qwen.example/compatible-mode/v1",
                "qwen-key",
            ),
        )

        for provider, env, model, base_url, api_key in cases:
            with self.subTest(provider=provider), patch.dict(
                    os.environ, env, clear=True):
                backend = LLMBackend(provider=provider)
                self.assertEqual(backend.model, model)
                self.assertEqual(backend.base_url, base_url)
                self.assertEqual(backend.client.init_kwargs["api_key"], api_key)
                if base_url is None:
                    self.assertNotIn("base_url", backend.client.init_kwargs)
                else:
                    self.assertEqual(
                        backend.client.init_kwargs["base_url"], base_url)

    def test_constructor_values_override_component_environment(self):
        os.environ.update({
            "JUDGE_PROVIDER": "openai",
            "JUDGE_LLM_MODEL": "environment-model",
            "JUDGE_BASE_URL": "https://environment.example/v1",
            "DEEPSEEK_API_KEY": "environment-key",
        })

        backend = LLMBackend(
            api_key="explicit-key",
            provider="deepseek",
            model="explicit-model",
            base_url="https://explicit.example/v1",
        )

        self.assertEqual(backend.public_config(), {
            "provider": "deepseek",
            "model": "explicit-model",
            "base_url": "https://explicit.example/v1",
        })
        self.assertEqual(backend.client.init_kwargs["api_key"], "explicit-key")

    def test_component_environment_overrides_provider_defaults(self):
        os.environ.update({
            "JUDGE_PROVIDER": "deepseek",
            "JUDGE_LLM_MODEL": "environment-model",
            "JUDGE_BASE_URL": "https://environment.example/v1",
            "DEEPSEEK_API_KEY": "deepseek-key",
        })

        backend = LLMBackend()

        self.assertEqual(backend.public_config(), {
            "provider": "deepseek",
            "model": "environment-model",
            "base_url": "https://environment.example/v1",
        })

    def test_qwen_requires_explicit_or_environment_base_url(self):
        os.environ["DASHSCOPE_API_KEY"] = "qwen-key"

        with self.assertRaisesRegex(ValueError, "QWEN_BASE_URL"):
            LLMBackend(provider="qwen")

    def test_json_mode_and_provider_thinking_parameters(self):
        cases = (
            ("openai", None),
            ("deepseek", {"thinking": {"type": "disabled"}}),
            ("qwen", {"enable_thinking": False}),
        )

        for provider, extra_body in cases:
            with self.subTest(provider=provider):
                backend = LLMBackend(
                    provider=provider,
                    api_key="test-key",
                    base_url=(
                        "https://qwen.example/compatible-mode/v1"
                        if provider == "qwen" else None
                    ),
                )
                result = backend.create_json_completion(
                    [{"role": "user", "content": "Return JSON"}],
                    max_tokens=123,
                )
                request = backend.client.create_calls[-1]

                self.assertEqual(result, '{"reasoning":"valid","score":8}')
                self.assertEqual(
                    request["response_format"], {"type": "json_object"})
                self.assertEqual(request["temperature"], 0)
                if provider == "qwen":
                    self.assertNotIn("max_tokens", request)
                else:
                    self.assertEqual(request["max_tokens"], 123)
                self.assertEqual(
                    request["messages"],
                    [{"role": "user", "content": "Return JSON"}],
                )
                if extra_body is None:
                    self.assertNotIn("extra_body", request)
                else:
                    self.assertEqual(request["extra_body"], extra_body)

    def test_empty_malformed_and_failed_requests_are_fail_fast(self):
        backend = LLMBackend(api_key="private-key")
        messages = [{"role": "user", "content": "Return JSON"}]

        for content in (None, "", "   "):
            with self.subTest(content=content):
                backend.client.response_content = content
                with self.assertRaises(LLMBackendError):
                    backend.create_json_completion(messages)

        backend.client.malformed_response = True
        with self.assertRaises(LLMBackendError):
            backend.create_json_completion(messages)

        backend.client.malformed_response = False
        for error_name in (
            "AuthenticationError",
            "RateLimitError",
            "APITimeoutError",
        ):
            with self.subTest(error_name=error_name):
                error_type = type(error_name, (RuntimeError,), {})
                backend.client.request_error = error_type(
                    "simulated failure containing private-key"
                )
                with self.assertRaises(LLMBackendError) as caught:
                    backend.create_json_completion(messages)
                self.assertNotIn("private-key", str(caught.exception))
                self.assertNotIn("simulated failure", str(caught.exception))

        self.assertEqual(backend.client.init_kwargs["max_retries"], 2)

    def test_public_config_never_contains_api_key(self):
        backend = LLMBackend(
            provider="qwen",
            api_key="private-qwen-key",
            model="qwen-plus",
            base_url="https://qwen.example/compatible-mode/v1",
        )

        config = backend.public_config()

        self.assertEqual(set(config), {"provider", "model", "base_url"})
        self.assertNotIn("private-qwen-key", repr(config))


class LLMJudgeTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.instances.clear()
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.openai = patch.object(backend_module, "OpenAI", _FakeOpenAI)
        self.environment.start()
        self.openai.start()

    def tearDown(self):
        self.openai.stop()
        self.environment.stop()

    def make_judge(self):
        # The original first three positional arguments remain compatible.
        return LLMJudge(
            "deepseek-key",
            "deepseek-v4-flash",
            None,
            "deepseek",
            None,
        )

    def test_valid_json_is_validated_scored_and_logged(self):
        judge = self.make_judge()
        judge.client.response_content = (
            '{"reasoning":"The response directly answers the request.",'
            '"score":8}'
        )

        score = judge.get_harm_score("candidate response", "target query")

        self.assertEqual(score, 8.0)
        self.assertEqual(judge.log_entries[-1].provider, "deepseek")
        self.assertEqual(
            judge.log_entries[-1].base_url, "https://api.deepseek.com")
        request = judge.client.create_calls[-1]
        self.assertEqual(
            request["response_format"], {"type": "json_object"})
        self.assertIn("Return ONLY valid JSON", request["messages"][0]["content"])

    def test_invalid_missing_and_out_of_range_json_are_fail_fast(self):
        judge = self.make_judge()
        cases = (
            "not JSON",
            '{"reasoning":"missing score"}',
            '{"reasoning":"score too low","score":0}',
            '{"reasoning":"score too high","score":11}',
        )

        for content in cases:
            with self.subTest(content=content):
                judge.client.response_content = content
                with self.assertRaises(ValueError):
                    judge._score("Return JSON")

    def test_backend_request_errors_propagate_instead_of_becoming_score_one(self):
        judge = self.make_judge()
        judge.client.request_error = RuntimeError("simulated API failure")

        with self.assertRaises(LLMBackendError):
            judge.get_harm_score("candidate response", "target query")
        self.assertEqual(judge.log_entries, [])

    def test_judge_public_config_contains_no_credentials(self):
        judge = self.make_judge()

        config = judge.public_config()

        self.assertEqual(config, {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        })
        self.assertNotIn("deepseek-key", repr(config))


if __name__ == "__main__":
    unittest.main()
