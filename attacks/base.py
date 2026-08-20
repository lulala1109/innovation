"""
Base class for WAV-level adversarial attacks.
"""

import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from models.base import BaseAudioModel
from core.audio import lowpass_filter_gradient


@dataclass
class AttackResult:
    """Result of an adversarial attack."""

    original_wav: torch.Tensor      # Original audio [1, T]
    adversarial_wav: torch.Tensor   # Perturbed audio [1, T]
    perturbation: torch.Tensor      # Delta [1, T]

    original_output: str            # Model output on original
    adversarial_output: str         # Model output on adversarial
    target_text: str                # Target we were trying to achieve

    success: bool                   # Whether target was achieved
    final_loss: float              # Final loss value
    steps_taken: int               # Number of attack steps

    history: Dict[str, Any]        # Structured optimization history


class BaseWavAttacker(ABC):
    """
    Base class for WAV-level adversarial attacks.

    Provides common functionality:
    - Perturbation initialization and clamping
    - Gradient computation and filtering
    - Progress logging

    Subclasses implement standard and safety-state-aware PGD strategies.
    """

    def __init__(
        self,
        model: BaseAudioModel,
        eps: float = 0.1,
        alpha: float = 0.005,
        use_lowpass: bool = False,
        lowpass_cutoff: float = 2000.0,
        verbose: bool = True
    ):
        """
        Initialize the attacker.

        Args:
            model: Audio model to attack
            eps: Maximum L-infinity perturbation (in waveform amplitude)
            alpha: Step size for gradient updates
            use_lowpass: Whether to lowpass filter gradients
            lowpass_cutoff: Cutoff frequency for lowpass filter (Hz)
            verbose: Whether to print progress
        """
        if not math.isfinite(float(eps)) or eps < 0:
            raise ValueError("eps must be finite and non-negative")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(float(lowpass_cutoff)) or lowpass_cutoff <= 0:
            raise ValueError("lowpass_cutoff must be finite and positive")
        if use_lowpass and lowpass_cutoff >= float(model.sample_rate) / 2.0:
            raise ValueError(
                "lowpass_cutoff must be below the model sample-rate Nyquist "
                "frequency"
            )
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.use_lowpass = use_lowpass
        self.lowpass_cutoff = lowpass_cutoff
        self.verbose = verbose

    def init_perturbation(
        self,
        wav: torch.Tensor,
        random_init: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Initialize and project the perturbation tensor.

        Args:
            wav: Original waveform [1, T]
            random_init: Whether to sample uniformly from the epsilon ball
            generator: Optional device-local RNG for reproducible random starts

        Returns:
            Projected delta tensor [1, T] with requires_grad=True
        """
        if random_init:
            delta = torch.empty_like(wav)
            delta.uniform_(
                -self.eps,
                self.eps,
                generator=generator
            )
        else:
            delta = torch.zeros_like(wav)

        # The first loss must already see a valid epsilon-bounded waveform.
        delta = self.clamp_perturbation(delta, wav)
        return delta.detach().requires_grad_(True)

    def clamp_perturbation(
        self,
        delta: torch.Tensor,
        wav: torch.Tensor
    ) -> torch.Tensor:
        """
        Clamp perturbation to eps-ball and ensure valid audio.

        Args:
            delta: Perturbation tensor
            wav: Original waveform

        Returns:
            Clamped delta
        """
        with torch.no_grad():
            # Project to the L-infinity epsilon ball.
            projected = delta.clamp(-self.eps, self.eps)

            # Keep the adversarial waveform in the valid audio range.
            projected = torch.clamp(
                wav + projected, -1.0, 1.0
            ) - wav

            delta.copy_(projected)

        return delta

    def process_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Process a gradient, optionally applying the configured low-pass filter."""

        if self.use_lowpass:
            grad = lowpass_filter_gradient(
                grad,
                cutoff_hz=self.lowpass_cutoff,
                sample_rate=self.model.sample_rate,
            )
        return grad

    def validate_resume_delta(
        self,
        value: torch.Tensor,
        wav: torch.Tensor,
        *,
        name: str = "delta",
    ) -> torch.Tensor:
        """Validate and clone a persisted perturbation without silently clipping it."""

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"resume {name} must be a tensor")
        restored = value.to(device=wav.device, dtype=wav.dtype).detach().clone()
        if restored.shape != wav.shape:
            raise ValueError(
                f"resume {name} shape {tuple(restored.shape)} does not match "
                f"waveform shape {tuple(wav.shape)}"
            )
        if not bool(torch.isfinite(restored).all()):
            raise ValueError(f"resume {name} contains NaN or infinite values")
        tolerance = max(1e-7, abs(float(self.eps)) * 1e-6)
        if restored.abs().max().item() > float(self.eps) + tolerance:
            raise ValueError(f"resume {name} exceeds the L-infinity budget")
        adversarial = wav + restored
        if (
            adversarial.amin().item() < -1.0 - tolerance
            or adversarial.amax().item() > 1.0 + tolerance
        ):
            raise ValueError(f"resume {name} produces audio outside [-1, 1]")
        return restored

    def compute_snr(
        self,
        original: torch.Tensor,
        adversarial: torch.Tensor
    ) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.

        Args:
            original: Original audio
            adversarial: Perturbed audio

        Returns:
            SNR in dB
        """
        noise = adversarial - original
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power < 1e-10:
            return float('inf')

        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()

    def log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    @abstractmethod
    def attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        steps: int = 100,
        **kwargs
    ) -> AttackResult:
        """
        Perform the adversarial attack.

        Args:
            wav: Original audio waveform [1, T]
            target_text: Target text to force
            steps: Number of attack iterations
            **kwargs: Additional attack-specific parameters

        Returns:
            AttackResult with adversarial audio and metadata
        """
        pass
