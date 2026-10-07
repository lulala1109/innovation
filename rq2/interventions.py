"""Pure tensor interventions used by the RQ2 causal experiments.

The functions in this module know nothing about Qwen or generation.  Keeping
the intervention algebra here makes every manipulation independently testable
and prevents hook lifecycle details from changing the causal estimand.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import torch


INTERVENTION_KINDS = (
    "full_state",
    "self_patch",
    "sham",
    "r_direction",
    "h_direction_control",
    "random_direction_control",
    "reverse_suppression",
    "reverse_h_control",
    "reverse_random_control",
    "reverse_sham",
    "subspace_restoration",
    "subspace_h_control",
    "subspace_random_control",
)


class InterventionError(ValueError):
    """Raised when an intervention is mathematically or structurally invalid."""


@dataclass(frozen=True)
class InterventionSpec:
    kind: str
    layer: int
    dose: float = 1.0
    token_scope: str = "audio"
    seed: int = 0
    replicate: int = 0

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise InterventionError(f"unsupported intervention kind: {self.kind}")
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise InterventionError("layer must be a non-negative integer")
        if not math.isfinite(float(self.dose)):
            raise InterventionError("dose must be finite")
        if not self.token_scope.strip():
            raise InterventionError("token_scope must be non-blank")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InterventionError("seed must be an integer")
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate < 0
        ):
            raise InterventionError("replicate must be non-negative")


@dataclass(frozen=True)
class InterventionAudit:
    kind: str
    layer: int
    dose: float
    token_scope: str
    token_count: int
    token_indices: tuple[int, ...]
    apply_count: int
    shift_l2: float
    shift_linf: float
    mean_shift_l2: float
    projection_before: Optional[float]
    projection_after: Optional[float]
    projection_reference: Optional[float]
    projection_delta: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _indices(
    sequence_length: int,
    selection: slice | tuple[int, int] | Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(selection, slice):
        result = torch.arange(sequence_length, device=device)[selection]
    elif isinstance(selection, torch.Tensor):
        if selection.ndim != 1:
            raise InterventionError("token indices must be one-dimensional")
        if selection.dtype == torch.bool:
            if selection.numel() != sequence_length:
                raise InterventionError("boolean token mask has the wrong length")
            result = selection.to(device=device).nonzero(as_tuple=True)[0]
        else:
            result = selection.to(device=device, dtype=torch.long)
    else:
        result = torch.as_tensor(tuple(selection), device=device, dtype=torch.long)
    if result.numel() == 0:
        raise InterventionError("token selection is empty")
    result = torch.where(result < 0, result + sequence_length, result).long()
    if bool(((result < 0) | (result >= sequence_length)).any()):
        raise InterventionError("token selection contains an out-of-range index")
    if result.unique().numel() != result.numel():
        raise InterventionError("token selection contains duplicates")
    return result


def _unit(vector: torch.Tensor, width: int, name: str) -> torch.Tensor:
    value = torch.as_tensor(vector)
    if tuple(value.shape) != (width,):
        raise InterventionError(f"{name} must have shape ({width},)")
    value = value.float()
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise InterventionError(f"{name} must have finite non-zero norm")
    return value / norm


def _selected_mean(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.index_select(-2, indices).float().mean(dim=tuple(range(value.ndim - 2)) + (-2,))


def _scalar_projection(mean: torch.Tensor, direction: Optional[torch.Tensor]) -> Optional[float]:
    if direction is None:
        return None
    return float(torch.dot(mean, direction).item())


def apply_intervention(
    activation: torch.Tensor,
    token_selection: slice | tuple[int, int] | Sequence[int] | torch.Tensor,
    spec: InterventionSpec,
    *,
    source_activation: Optional[torch.Tensor] = None,
    source_token_selection: Optional[
        slice | tuple[int, int] | Sequence[int] | torch.Tensor
    ] = None,
    target_reference_selection: Optional[
        slice | tuple[int, int] | Sequence[int] | torch.Tensor
    ] = None,
    application_scale: float = 1.0,
    refusal_direction: Optional[torch.Tensor] = None,
    harmfulness_direction: Optional[torch.Tensor] = None,
    subspace: Optional[torch.Tensor] = None,
    refusal_sigma: Optional[float] = None,
) -> tuple[torch.Tensor, InterventionAudit]:
    """Apply one intervention and return a detached numerical audit.

    Direction restoration changes the selected-token mean by the reference
    projection gap.  H and seeded-random controls receive exactly the same
    per-token shift norm as the corresponding R restoration.
    """

    if not isinstance(activation, torch.Tensor) or activation.ndim < 2:
        raise InterventionError("activation must have shape [..., tokens, hidden]")
    if not torch.is_floating_point(activation):
        raise InterventionError("activation must be floating point")
    width = activation.shape[-1]
    if not math.isfinite(float(application_scale)) or float(application_scale) <= 0:
        raise InterventionError("application_scale must be finite and positive")
    target_indices = _indices(activation.shape[-2], token_selection, activation.device)
    current = activation.index_select(-2, target_indices)
    current_mean = _selected_mean(activation, target_indices)
    reference_indices = (
        target_indices
        if target_reference_selection is None
        else _indices(
            activation.shape[-2], target_reference_selection, activation.device
        )
    )
    reference_current_mean = _selected_mean(activation, reference_indices)

    source: Optional[torch.Tensor] = None
    source_mean: Optional[torch.Tensor] = None
    if source_activation is not None:
        if source_activation.ndim != activation.ndim or source_activation.shape[-1] != width:
            raise InterventionError("source and target activation ranks/widths differ")
        selection = token_selection if source_token_selection is None else source_token_selection
        source_indices = _indices(
            source_activation.shape[-2], selection, source_activation.device
        )
        source = source_activation.index_select(-2, source_indices)
        source = source.to(device=activation.device, dtype=activation.dtype)
        source_mean = source.float().mean(
            dim=tuple(range(source.ndim - 2)) + (-2,)
        )

    r_unit = None
    if refusal_direction is not None:
        r_unit = _unit(refusal_direction, width, "refusal_direction").to(
            device=activation.device
        )
    h_unit = None
    if harmfulness_direction is not None:
        h_unit = _unit(harmfulness_direction, width, "harmfulness_direction").to(
            device=activation.device
        )

    shift_vector: Optional[torch.Tensor] = None
    kind = spec.kind
    if kind in {"full_state", "self_patch"}:
        if source is None:
            raise InterventionError(f"{kind} requires source_activation")
        if source.shape != current.shape:
            raise InterventionError(
                "full-state source and target selections must have identical shapes"
            )
        replacement = current + float(spec.dose) * (source - current)
    elif kind in {"sham", "reverse_sham"}:
        replacement = current.clone()
    elif kind in {"r_direction", "h_direction_control", "random_direction_control"}:
        if source_mean is None or r_unit is None:
            raise InterventionError(f"{kind} requires source_activation and refusal_direction")
        r_gap = torch.dot(source_mean - reference_current_mean, r_unit)
        magnitude = abs(float(spec.dose)) * torch.abs(r_gap)
        if kind == "r_direction":
            shift_vector = float(spec.dose) * r_gap * r_unit
        elif kind == "h_direction_control":
            if h_unit is None:
                raise InterventionError("h_direction_control requires harmfulness_direction")
            h_gap = torch.dot(source_mean - reference_current_mean, h_unit)
            sign = torch.where(h_gap < 0, -torch.ones_like(h_gap), torch.ones_like(h_gap))
            shift_vector = magnitude * sign * h_unit
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(spec.seed) + 1_000_003 * int(spec.replicate))
            random = torch.randn(width, generator=generator, dtype=torch.float32)
            random = _unit(random, width, "random_direction").to(activation.device)
            shift_vector = magnitude * random
        replacement = current + float(application_scale) * shift_vector.to(dtype=activation.dtype)
    elif kind in {"reverse_suppression", "reverse_h_control", "reverse_random_control"}:
        if r_unit is None:
            raise InterventionError(f"{kind} requires refusal_direction")
        if refusal_sigma is None or not math.isfinite(float(refusal_sigma)) or refusal_sigma <= 0:
            raise InterventionError(f"{kind} requires positive finite refusal_sigma")
        magnitude = float(spec.dose) * float(refusal_sigma)
        if kind == "reverse_suppression":
            shift_vector = -magnitude * r_unit
        elif kind == "reverse_h_control":
            if h_unit is None:
                raise InterventionError("reverse_h_control requires harmfulness_direction")
            shift_vector = -magnitude * h_unit
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(spec.seed) + 1_000_003 * int(spec.replicate))
            random = _unit(
                torch.randn(width, generator=generator, dtype=torch.float32),
                width,
                "random_direction",
            ).to(activation.device)
            shift_vector = magnitude * random
        replacement = current + float(application_scale) * shift_vector.to(dtype=activation.dtype)
    elif kind in {"subspace_restoration", "subspace_h_control", "subspace_random_control"}:
        if source_mean is None or subspace is None:
            raise InterventionError(f"{kind} requires source_activation and subspace")
        basis = torch.as_tensor(subspace).float()
        if basis.ndim != 2:
            raise InterventionError("subspace must be a rank-2 tensor")
        if basis.shape[0] == width:
            basis = basis.T
        if basis.shape[1] != width:
            raise InterventionError("subspace must have hidden width on one axis")
        gram = basis @ basis.T
        identity = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device)
        if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
            raise InterventionError("subspace rows must be orthonormal")
        basis = basis.to(activation.device)
        gap = source_mean - reference_current_mean
        projected = (gap @ basis.T) @ basis
        magnitude = abs(float(spec.dose)) * torch.linalg.vector_norm(projected)
        if kind == "subspace_restoration":
            shift_vector = float(spec.dose) * projected
        elif kind == "subspace_h_control":
            if h_unit is None:
                raise InterventionError("subspace_h_control requires harmfulness_direction")
            h_gap = torch.dot(gap, h_unit)
            sign = torch.where(h_gap < 0, -torch.ones_like(h_gap), torch.ones_like(h_gap))
            shift_vector = magnitude * sign * h_unit
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(spec.seed) + 1_000_003 * int(spec.replicate))
            random = _unit(
                torch.randn(width, generator=generator, dtype=torch.float32),
                width,
                "random_direction",
            ).to(activation.device)
            shift_vector = magnitude * random
        replacement = current + float(application_scale) * shift_vector.to(dtype=activation.dtype)
    else:  # guarded by InterventionSpec, retained for defensive callers
        raise InterventionError(f"unsupported intervention kind: {kind}")

    patched = activation.clone()
    patched.index_copy_(-2, target_indices, replacement)
    difference = (patched.float() - activation.float()).detach()
    selected_after_mean = _selected_mean(patched, target_indices)
    reference_projection = (
        _scalar_projection(source_mean, r_unit)
        if source_mean is not None
        else None
    )
    before_projection = _scalar_projection(current_mean, r_unit)
    after_projection = _scalar_projection(selected_after_mean, r_unit)
    audit = InterventionAudit(
        kind=kind,
        layer=spec.layer,
        dose=float(spec.dose),
        token_scope=spec.token_scope,
        token_count=int(target_indices.numel()),
        token_indices=tuple(int(value) for value in target_indices.detach().cpu().tolist()),
        apply_count=1,
        shift_l2=float(torch.linalg.vector_norm(difference).item()),
        shift_linf=float(difference.abs().max().item()),
        mean_shift_l2=float(
            torch.linalg.vector_norm(selected_after_mean - current_mean).item()
        ),
        projection_before=before_projection,
        projection_after=after_projection,
        projection_reference=reference_projection,
        projection_delta=(
            None
            if before_projection is None or after_projection is None
            else after_projection - before_projection
        ),
    )
    return patched, audit


class PrefillOnlyTransform:
    """Callable hook transform that applies exactly once to the prompt prefill."""

    def __init__(
        self,
        *,
        expected_sequence_length: int,
        token_selection: slice | tuple[int, int] | Sequence[int] | torch.Tensor,
        spec: InterventionSpec,
        source_activation: Optional[torch.Tensor] = None,
        source_token_selection: Optional[
            slice | tuple[int, int] | Sequence[int] | torch.Tensor
        ] = None,
        target_reference_selection: Optional[
            slice | tuple[int, int] | Sequence[int] | torch.Tensor
        ] = None,
        application_scale: float = 1.0,
        refusal_direction: Optional[torch.Tensor] = None,
        harmfulness_direction: Optional[torch.Tensor] = None,
        subspace: Optional[torch.Tensor] = None,
        refusal_sigma: Optional[float] = None,
    ) -> None:
        if expected_sequence_length <= 1:
            raise InterventionError("prefill sequence length must be greater than one")
        self.expected_sequence_length = int(expected_sequence_length)
        self.token_selection = token_selection
        self.spec = spec
        self.kwargs = {
            "source_activation": source_activation,
            "source_token_selection": source_token_selection,
            "target_reference_selection": target_reference_selection,
            "application_scale": application_scale,
            "refusal_direction": refusal_direction,
            "harmfulness_direction": harmfulness_direction,
            "subspace": subspace,
            "refusal_sigma": refusal_sigma,
        }
        self.prefill_hits = 0
        self.cache_skips = 0
        self.audit: Optional[InterventionAudit] = None

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        sequence_length = activation.shape[-2]
        if sequence_length == 1:
            self.cache_skips += 1
            return activation
        if sequence_length != self.expected_sequence_length:
            raise InterventionError(
                "unexpected non-cache sequence length: "
                f"{sequence_length} != {self.expected_sequence_length}"
            )
        self.prefill_hits += 1
        if self.prefill_hits > 1:
            raise InterventionError("intervention hit the prefill more than once")
        patched, audit = apply_intervention(
            activation,
            self.token_selection,
            self.spec,
            **self.kwargs,
        )
        self.audit = audit
        return patched

    def assert_applied_once(self) -> InterventionAudit:
        if self.prefill_hits != 1 or self.audit is None:
            raise InterventionError(
                f"expected exactly one prefill application, observed {self.prefill_hits}"
            )
        return self.audit


__all__ = [
    "INTERVENTION_KINDS",
    "InterventionAudit",
    "InterventionError",
    "InterventionSpec",
    "PrefillOnlyTransform",
    "apply_intervention",
]
