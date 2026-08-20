"""Supported audio-model adapters for waveform-bounded experiments.

Qwen exposes the differentiable ``forward_attack`` path with aligned hidden
states used by the safety-state methods. Incompatible exploratory adapters
live under :mod:`models.optional` and are absent from the default factory.
"""

from importlib import import_module

import torch
from typing import Optional

from models.base import AttackForwardOutput, BaseAudioModel

__all__ = [
    "AttackForwardOutput",
    "BaseAudioModel",
    "QwenModel",
    "create_model",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL_IDS",
]

SUPPORTED_MODELS = ("qwen-3b", "qwen-7b")
DEFAULT_MODEL_IDS = {
    "qwen-3b": "Qwen/Qwen2.5-Omni-3B",
    "qwen-7b": "Qwen/Qwen2.5-Omni-7B",
}


def create_model(
    model_type: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    token: Optional[str] = None,
    model_id: Optional[str] = None,
) -> BaseAudioModel:
    """
    Factory function to create audio model instances.

    Args:
        model_type: One of :data:`SUPPORTED_MODELS`.
        device: Device to run on (e.g., "cuda", "cuda:0")
        dtype: Model dtype (default: torch.bfloat16)
        token: HuggingFace token (optional, uses env var if not provided)
        model_id: Optional checkpoint path/ID overriding the portable default.

    Returns:
        BaseAudioModel instance

    Raises:
        ValueError: If model_type is not supported

    Example:
        model = create_model("qwen-3b", device="cuda:0")
        loss = model.compute_loss(wav, "target text")
    """
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Supported models: {list(SUPPORTED_MODELS)}"
        )
    checkpoint = model_id or DEFAULT_MODEL_IDS[model_type]

    from models.qwen import QwenModel

    return QwenModel(
        model_id=checkpoint,
        device=device,
        dtype=dtype,
        token=token,
    )



# Lazy imports for direct class access
def __getattr__(name):
    """Lazy import for model classes."""
    modules = {
        "AttackForwardOutput": "models.base",
        "QwenModel": "models.qwen",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
