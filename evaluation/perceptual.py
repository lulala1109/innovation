"""Offline waveform perturbation and optional perceptual-quality metrics.

The core metrics depend only on PyTorch and NumPy. PESQ and STOI are optional:
their third-party implementations are imported only when explicitly requested,
and a missing dependency raises a clear error instead of emitting a placeholder
or fabricated score.
"""

from __future__ import annotations

import math
from importlib import import_module
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


AudioArray = Union[torch.Tensor, np.ndarray]


class OptionalMetricDependencyError(ImportError):
    """Raised when an explicitly requested optional metric is unavailable."""


def _as_float_tensor(value: AudioArray, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            raise TypeError(f"{name} must contain real-valued audio samples")
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{name} must contain real-valued audio samples")
        # ``asarray`` also normalizes non-contiguous and non-native-endian input
        # before conversion to a standalone evaluation tensor.
        tensor = torch.from_numpy(np.asarray(value, dtype=np.float64).copy())
    else:
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite samples")
    return tensor


def _validate_pair(
    reference: AudioArray,
    degraded: AudioArray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_tensor = _as_float_tensor(reference, name="reference")
    degraded_tensor = _as_float_tensor(degraded, name="degraded")
    if reference_tensor.shape != degraded_tensor.shape:
        raise ValueError(
            "reference and degraded audio must have identical shapes; got "
            f"{tuple(reference_tensor.shape)} and {tuple(degraded_tensor.shape)}"
        )
    return reference_tensor, degraded_tensor


def perturbation_linf(reference: AudioArray, degraded: AudioArray) -> float:
    """Return ``max(abs(degraded - reference))`` over all samples."""

    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)
    return float((degraded_tensor - reference_tensor).abs().max().item())


def perturbation_l2(reference: AudioArray, degraded: AudioArray) -> float:
    """Return the unnormalized Euclidean norm of the waveform perturbation."""

    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)
    return float(torch.linalg.vector_norm(degraded_tensor - reference_tensor).item())


def perturbation_rms(reference: AudioArray, degraded: AudioArray) -> float:
    """Return root-mean-square perturbation amplitude over all samples."""

    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)
    noise = degraded_tensor - reference_tensor
    return float(torch.sqrt(torch.mean(noise.square())).item())


def signal_to_noise_ratio_db(reference: AudioArray, degraded: AudioArray) -> float:
    """Return power SNR in decibels without silently adding an epsilon.

    Identical waveforms have ``+inf`` SNR. A zero reference with non-zero noise
    has ``-inf`` SNR. These explicit edge values preserve the mathematical
    distinction and callers can encode them as ``null`` when writing strict JSON.
    """

    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)
    signal_power = torch.mean(reference_tensor.square()).item()
    noise_power = torch.mean((degraded_tensor - reference_tensor).square()).item()
    if noise_power == 0.0:
        return math.inf
    if signal_power == 0.0:
        return -math.inf
    return float(10.0 * math.log10(signal_power / noise_power))


# Short compatibility name used by the attack code and result schemas.
snr_db = signal_to_noise_ratio_db
signal_to_perturbation_ratio_db = signal_to_noise_ratio_db
spr_db = signal_to_perturbation_ratio_db


def compute_perturbation_metrics(
    reference: AudioArray,
    degraded: AudioArray,
) -> Dict[str, float]:
    """Compute JSON-ready scalar perturbation metrics in one validation pass."""

    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)
    noise = degraded_tensor - reference_tensor
    noise_power = torch.mean(noise.square()).item()
    signal_power = torch.mean(reference_tensor.square()).item()
    if noise_power == 0.0:
        snr = math.inf
    elif signal_power == 0.0:
        snr = -math.inf
    else:
        snr = float(10.0 * math.log10(signal_power / noise_power))
    return {
        "perturbation_linf": float(noise.abs().max().item()),
        "perturbation_l2": float(torch.linalg.vector_norm(noise).item()),
        "perturbation_rms": float(math.sqrt(noise_power)),
        "snr_db": snr,
    }


def _as_mono_numpy_pair(
    reference: AudioArray,
    degraded: AudioArray,
) -> Tuple[np.ndarray, np.ndarray]:
    reference_tensor, degraded_tensor = _validate_pair(reference, degraded)

    def mono(tensor: torch.Tensor, name: str) -> np.ndarray:
        if tensor.ndim == 1:
            flattened = tensor
        elif tensor.ndim == 2 and tensor.shape[0] == 1:
            flattened = tensor[0]
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            flattened = tensor[:, 0]
        else:
            raise ValueError(
                f"{name} must be mono with shape [samples], [1, samples], "
                "or [samples, 1] for PESQ/STOI"
            )
        return flattened.numpy().astype(np.float32, copy=False)

    return mono(reference_tensor, "reference"), mono(degraded_tensor, "degraded")


def _validate_sample_rate(sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return sample_rate


def compute_pesq(
    reference: AudioArray,
    degraded: AudioArray,
    sample_rate: int,
    *,
    mode: Optional[str] = None,
) -> float:
    """Compute PESQ using the optional ``pesq`` package.

    PESQ accepts 8 kHz narrow-band or 16 kHz wide-band audio. Dependency and
    backend computation errors are never converted into made-up scores.
    """

    sample_rate = _validate_sample_rate(sample_rate)
    if sample_rate not in (8_000, 16_000):
        raise ValueError("PESQ supports only 8000 Hz or 16000 Hz audio")
    resolved_mode = (mode or ("wb" if sample_rate == 16_000 else "nb")).lower()
    if resolved_mode not in ("nb", "wb"):
        raise ValueError("PESQ mode must be 'nb' or 'wb'")
    if sample_rate == 8_000 and resolved_mode != "nb":
        raise ValueError("8000 Hz PESQ requires narrow-band mode 'nb'")
    reference_np, degraded_np = _as_mono_numpy_pair(reference, degraded)
    try:
        module = import_module("pesq")
    except ImportError as exc:
        raise OptionalMetricDependencyError(
            "PESQ was requested but the optional 'pesq' package is not "
            "installed; install it to compute a real PESQ score"
        ) from exc
    score = float(module.pesq(sample_rate, reference_np, degraded_np, resolved_mode))
    if not math.isfinite(score):
        raise ValueError("PESQ backend returned a non-finite score")
    return score


def compute_stoi(
    reference: AudioArray,
    degraded: AudioArray,
    sample_rate: int,
    *,
    extended: bool = False,
) -> float:
    """Compute STOI using the optional ``pystoi`` package."""

    sample_rate = _validate_sample_rate(sample_rate)
    reference_np, degraded_np = _as_mono_numpy_pair(reference, degraded)
    try:
        module = import_module("pystoi")
    except ImportError as exc:
        raise OptionalMetricDependencyError(
            "STOI was requested but the optional 'pystoi' package is not "
            "installed; install it to compute a real STOI score"
        ) from exc
    score = float(
        module.stoi(
            reference_np,
            degraded_np,
            sample_rate,
            extended=bool(extended),
        )
    )
    if not math.isfinite(score):
        raise ValueError("STOI backend returned a non-finite score")
    return score


def compute_perceptual_metrics(
    reference: AudioArray,
    degraded: AudioArray,
    *,
    sample_rate: Optional[int] = None,
    include_pesq: bool = False,
    include_stoi: bool = False,
    pesq_mode: Optional[str] = None,
    extended_stoi: bool = False,
) -> Dict[str, float]:
    """Compute base metrics and explicitly requested optional quality scores."""

    metrics = compute_perturbation_metrics(reference, degraded)
    if (include_pesq or include_stoi) and sample_rate is None:
        raise ValueError("sample_rate is required for PESQ or STOI")
    if include_pesq:
        metrics["pesq"] = compute_pesq(
            reference,
            degraded,
            sample_rate,
            mode=pesq_mode,
        )
    if include_stoi:
        metrics["stoi"] = compute_stoi(
            reference,
            degraded,
            sample_rate,
            extended=extended_stoi,
        )
    return metrics


__all__ = [
    "OptionalMetricDependencyError",
    "compute_perceptual_metrics",
    "compute_pesq",
    "compute_perturbation_metrics",
    "compute_stoi",
    "perturbation_l2",
    "perturbation_linf",
    "perturbation_rms",
    "signal_to_noise_ratio_db",
    "signal_to_perturbation_ratio_db",
    "snr_db",
    "spr_db",
]
