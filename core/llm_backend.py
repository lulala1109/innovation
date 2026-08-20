"""Shared OpenAI-compatible backend configuration and JSON requests.

The project uses the OpenAI Python SDK for OpenAI itself and for providers that
expose an OpenAI-compatible Chat Completions API. This module keeps provider
selection, credentials, endpoint defaults, and retry behaviour in one place.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from openai import OpenAI


log = logging.getLogger(__name__)


class LLMBackendError(RuntimeError):
    """Raised when an OpenAI-compatible request cannot produce usable content."""


@dataclass(frozen=True)
class _ProviderDefaults:
    api_key_env: str
    model: str
    base_url: Optional[str] = None


PROVIDER_DEFAULTS: Mapping[str, _ProviderDefaults] = {
    "openai": _ProviderDefaults(
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o-mini",
    ),
    "deepseek": _ProviderDefaults(
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    ),
    "qwen": _ProviderDefaults(
        api_key_env="DASHSCOPE_API_KEY",
        model="qwen-plus",
    ),
}
SUPPORTED_PROVIDERS = tuple(PROVIDER_DEFAULTS)


def _nonempty(value: Optional[str]) -> Optional[str]:
    """Normalize optional CLI/environment strings without accepting blanks."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class LLMBackend:
    """Configured OpenAI-compatible client with strict JSON-mode requests.

    Explicit constructor values take precedence over component-specific
    environment variables. Provider defaults are used last. API credentials
    are always read from the environment associated with the selected provider
    unless ``api_key`` is supplied explicitly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        provider_env: str = "JUDGE_PROVIDER",
        model_env: str = "JUDGE_LLM_MODEL",
        base_url_env: str = "JUDGE_BASE_URL",
        max_retries: int = 2,
        timeout: float = 60.0,
    ):
        selected_provider = (
            _nonempty(provider)
            or _nonempty(os.getenv(provider_env))
            or "openai"
        ).lower()
        if selected_provider not in PROVIDER_DEFAULTS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Unsupported LLM provider '{selected_provider}'. "
                f"Choose one of: {supported}."
            )

        defaults = PROVIDER_DEFAULTS[selected_provider]
        selected_model = (
            _nonempty(model)
            or _nonempty(os.getenv(model_env))
            or defaults.model
        )

        selected_base_url = (
            _nonempty(base_url)
            or _nonempty(os.getenv(base_url_env))
        )
        if selected_provider == "qwen":
            selected_base_url = (
                selected_base_url
                or _nonempty(os.getenv("QWEN_BASE_URL"))
            )
            if not selected_base_url:
                raise ValueError(
                    "Qwen requires an OpenAI-compatible base URL. Provide "
                    "'base_url', set " + base_url_env + ", or set QWEN_BASE_URL."
                )
        elif selected_base_url is None:
            selected_base_url = defaults.base_url

        selected_api_key = (
            _nonempty(api_key)
            or _nonempty(os.getenv(defaults.api_key_env))
        )
        if not selected_api_key:
            raise ValueError(
                f"API key required for provider '{selected_provider}'. Provide "
                f"'api_key' or set {defaults.api_key_env}."
            )

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.provider = selected_provider
        self.model = selected_model
        self.base_url = selected_base_url
        self.max_retries = max_retries
        self.timeout = timeout

        client_kwargs: Dict[str, Any] = {
            "api_key": selected_api_key,
            "max_retries": max_retries,
            "timeout": timeout,
        }
        if selected_base_url:
            client_kwargs["base_url"] = selected_base_url
        self.client = OpenAI(**client_kwargs)

    def public_config(self) -> Dict[str, Optional[str]]:
        """Return serializable configuration that never includes credentials."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def create_json_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int = 500,
    ) -> str:
        """Run a deterministic JSON-mode completion and return non-empty text.

        The OpenAI SDK performs at most ``max_retries`` retries configured on
        the client. Errors are re-raised without embedding provider exception
        text, preventing credentials from being copied into logs or result
        files by callers.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        # Qwen's structured-output guide recommends omitting max_tokens so a
        # valid JSON object is not truncated midway. Keep the explicit bound
        # for OpenAI and DeepSeek, whose JSON-mode guidance supports it.
        if self.provider != "qwen":
            request_kwargs["max_tokens"] = max_tokens

        if self.provider == "deepseek":
            request_kwargs["extra_body"] = {
                "thinking": {"type": "disabled"}
            }
        elif self.provider == "qwen":
            request_kwargs["extra_body"] = {"enable_thinking": False}

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            error_type = type(exc).__name__
            log.error(
                "LLM request failed for provider=%s model=%s (%s)",
                self.provider,
                self.model,
                error_type,
            )
            raise LLMBackendError(
                f"LLM request failed for provider '{self.provider}' and model "
                f"'{self.model}' after configured retries ({error_type})."
            ) from None

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise LLMBackendError(
                f"Provider '{self.provider}' returned a malformed completion."
            ) from None

        if not isinstance(content, str) or not content.strip():
            raise LLMBackendError(
                f"Provider '{self.provider}' returned empty completion content."
            )
        return content.strip()


# Descriptive alias for callers that prefer to make compatibility explicit.
OpenAICompatibleBackend = LLMBackend


__all__ = [
    "LLMBackend",
    "LLMBackendError",
    "OpenAICompatibleBackend",
    "PROVIDER_DEFAULTS",
    "SUPPORTED_PROVIDERS",
]
