"""Differentiable audio EoT transforms and honest evaluation-only adapters."""

from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


class EvaluationBackendRequiredError(RuntimeError):
    """A codec/playback evaluation was requested without a real backend."""


class NonDifferentiableTransformError(RuntimeError):
    """An evaluation-only transform was used in a gradient EoT path."""


def _waveform(value: Any, *, name: str = "waveform") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.is_complex() or not value.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if value.ndim < 1 or value.numel() == 0:
        raise ValueError(f"{name} must end in a non-empty sample dimension")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def add_noise(
    waveform: torch.Tensor,
    *,
    snr_db: float,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Add RMS-normalized noise at a requested signal-to-noise ratio."""

    waveform = _waveform(waveform)
    snr_db = float(snr_db)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if noise is None:
        noise = torch.randn(
            waveform.shape,
            device=waveform.device,
            dtype=waveform.dtype,
            generator=generator,
        )
    else:
        noise = _waveform(noise, name="noise").to(
            device=waveform.device, dtype=waveform.dtype
        )
        try:
            noise = torch.broadcast_to(noise, waveform.shape)
        except RuntimeError as exc:
            raise ValueError("noise cannot be broadcast to waveform shape") from exc
    signal_rms = waveform.square().mean(dim=-1, keepdim=True).sqrt()
    noise_rms = noise.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    return waveform + noise * (target_noise_rms / noise_rms)


additive_noise = add_noise
add_white_noise = add_noise


def resample_waveform(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """Differentiable linear resampling along the final dimension."""

    waveform = _waveform(waveform)
    for value, name in (
        (original_sample_rate, "original_sample_rate"),
        (target_sample_rate, "target_sample_rate"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if original_sample_rate == target_sample_rate:
        return waveform.clone()
    output_length = max(
        1, int(round(waveform.shape[-1] * target_sample_rate / original_sample_rate))
    )
    leading_shape = waveform.shape[:-1]
    flat = waveform.reshape(-1, 1, waveform.shape[-1])
    output = F.interpolate(
        flat, size=output_length, mode="linear", align_corners=False
    )
    return output.reshape(*leading_shape, output_length)


resample = resample_waveform


def apply_rir(
    waveform: torch.Tensor,
    rir: torch.Tensor,
    *,
    mode: str = "same",
    normalize_rir: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply causal room impulse response convolution using torch operations."""

    waveform = _waveform(waveform)
    rir = _waveform(rir, name="rir").to(
        device=waveform.device, dtype=waveform.dtype
    ).reshape(-1)
    if mode not in {"same", "full"}:
        raise ValueError("mode must be 'same' or 'full'")
    if normalize_rir:
        rir = rir / rir.square().sum().sqrt().clamp_min(eps)
    flat = waveform.reshape(-1, 1, waveform.shape[-1])
    # conv1d implements correlation, hence the flipped impulse response.
    convolved = F.conv1d(
        flat, rir.flip(0).reshape(1, 1, -1), padding=rir.numel() - 1
    )
    if mode == "same":
        convolved = convolved[..., : waveform.shape[-1]]
    return convolved.reshape(*waveform.shape[:-1], convolved.shape[-1])


rir_convolution = apply_rir
apply_room_impulse_response = apply_rir


@dataclass(frozen=True)
class NoiseTransform:
    snr_db: float
    noise: Optional[torch.Tensor] = None

    def __call__(
        self, waveform: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return add_noise(
            waveform, snr_db=self.snr_db, noise=self.noise, generator=generator
        )


@dataclass(frozen=True)
class ResampleTransform:
    original_sample_rate: int
    target_sample_rate: int
    roundtrip: bool = False

    def __call__(
        self, waveform: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        del generator
        output = resample_waveform(
            waveform, self.original_sample_rate, self.target_sample_rate
        )
        if self.roundtrip:
            output = resample_waveform(
                output, self.target_sample_rate, self.original_sample_rate
            )
            target_length = waveform.shape[-1]
            if output.shape[-1] > target_length:
                output = output[..., :target_length]
            elif output.shape[-1] < target_length:
                output = F.pad(output, (0, target_length - output.shape[-1]))
        return output


@dataclass(frozen=True)
class RIRTransform:
    rir: torch.Tensor
    mode: str = "same"
    normalize_rir: bool = True

    def __call__(
        self, waveform: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        del generator
        return apply_rir(
            waveform,
            self.rir,
            mode=self.mode,
            normalize_rir=self.normalize_rir,
        )


class EoTCompose:
    """Compose differentiable transforms sharing one torch generator."""

    differentiable = True

    def __init__(self, transforms: Iterable[Callable[..., torch.Tensor]]) -> None:
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError("transforms cannot be empty")
        for transform in self.transforms:
            if not callable(transform):
                raise TypeError("every transform must be callable")
            if getattr(transform, "differentiable", True) is False:
                raise NonDifferentiableTransformError(
                    "evaluation-only adapters cannot enter EoTCompose"
                )

    def __call__(
        self, waveform: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        output = waveform
        for transform in self.transforms:
            output = _call_transform(transform, output, generator)
        return output


def _call_transform(
    transform: Callable[..., torch.Tensor],
    waveform: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Support both simple f(wav) and stochastic f(wav, generator=...) APIs."""

    try:
        parameters = inspect.signature(transform).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_generator = "generator" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    output = (
        transform(waveform, generator=generator)
        if accepts_generator
        else transform(waveform)
    )
    return _waveform(output, name="transform output")


def expectation_over_transforms(
    waveform: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    transform: Callable[..., torch.Tensor],
    *,
    samples: int = 4,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Average a differentiable loss over stochastic transform draws."""

    waveform = _waveform(waveform)
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("samples must be an integer")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if getattr(transform, "differentiable", True) is False:
        raise NonDifferentiableTransformError(
            "evaluation-only transform cannot be used in attack-time EoT"
        )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=waveform.device)
        generator.manual_seed(int(seed))
    losses = []
    for _ in range(samples):
        transformed = _call_transform(transform, waveform, generator)
        loss = loss_fn(transformed)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError("loss_fn must return one scalar tensor")
        losses.append(loss.reshape(()))
    return torch.stack(losses).mean()


eot_loss = expectation_over_transforms
EOTCompose = EoTCompose


class EvaluationTransformAdapter:
    """Adapter for a real non-differentiable evaluation backend.

    The backend must be injected and return either a waveform, a
    (waveform, sample_rate) tuple, or a mapping with waveform/audio and
    sample_rate/sr. No codec or playback result is simulated.
    """

    differentiable = False

    def __init__(
        self,
        backend: Callable[..., Any],
        *,
        kind: str,
        name: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable")
        if kind not in {"codec", "playback"}:
            raise ValueError("kind must be codec or playback")
        self.backend = backend
        self.kind = kind
        self.name = str(name)
        self.options = dict(options or {})

    def evaluate(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> tuple[torch.Tensor, int]:
        waveform = _waveform(waveform)
        result = self.backend(
            waveform.detach().cpu(), sample_rate, **self.options
        )
        if isinstance(result, Mapping):
            output = result.get("waveform", result.get("audio"))
            output_rate = result.get("sample_rate", result.get("sr", sample_rate))
        elif isinstance(result, tuple) and len(result) == 2:
            output, output_rate = result
        else:
            output, output_rate = result, sample_rate
        output = torch.as_tensor(output)
        _waveform(output, name=f"{self.kind} backend output")
        if isinstance(output_rate, bool) or not isinstance(output_rate, int):
            raise TypeError("evaluation backend sample rate must be an integer")
        if output_rate <= 0:
            raise ValueError("evaluation backend sample rate must be positive")
        return output.detach().cpu(), output_rate

    def __call__(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise NonDifferentiableTransformError(
            f"{self.kind} adapter is evaluation-only; call evaluate(wav, sr)"
        )


class CodecEvaluationAdapter(EvaluationTransformAdapter):
    def __init__(
        self, backend: Callable[..., Any], codec: str, **options: Any
    ) -> None:
        if not codec:
            raise ValueError("codec cannot be empty")
        super().__init__(
            backend, kind="codec", name=str(codec),
            options={"codec": codec, **options},
        )


class PlaybackEvaluationAdapter(EvaluationTransformAdapter):
    def __init__(
        self, backend: Callable[..., Any], name: str = "playback", **options: Any
    ) -> None:
        super().__init__(
            backend, kind="playback", name=name, options=options
        )


def codec_roundtrip(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    backend: Optional[Callable[..., Any]] = None,
    codec: str = "opus",
    **options: Any,
) -> tuple[torch.Tensor, int]:
    if backend is None:
        raise EvaluationBackendRequiredError(
            "codec evaluation requires a real encode/decode backend"
        )
    return CodecEvaluationAdapter(backend, codec, **options).evaluate(
        waveform, sample_rate
    )


def playback_capture(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    backend: Optional[Callable[..., Any]] = None,
    **options: Any,
) -> tuple[torch.Tensor, int]:
    if backend is None:
        raise EvaluationBackendRequiredError(
            "playback evaluation requires a real playback/capture backend"
        )
    return PlaybackEvaluationAdapter(backend, **options).evaluate(
        waveform, sample_rate
    )


__all__ = [
    "CodecEvaluationAdapter",
    "EoTCompose",
    "EOTCompose",
    "EvaluationBackendRequiredError",
    "EvaluationTransformAdapter",
    "NoiseTransform",
    "NonDifferentiableTransformError",
    "PlaybackEvaluationAdapter",
    "RIRTransform",
    "ResampleTransform",
    "add_noise",
    "additive_noise",
    "add_white_noise",
    "apply_rir",
    "apply_room_impulse_response",
    "codec_roundtrip",
    "eot_loss",
    "expectation_over_transforms",
    "playback_capture",
    "resample",
    "resample_waveform",
    "rir_convolution",
]
