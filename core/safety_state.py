"""Explicit, differentiable harmfulness and refusal safety states.

Harm recognition and refusal execution are intentionally represented by two
independent probes. This module never collapses them into an implicit scalar
"safety score". Callers must choose the state that defines a gap or loss (the
dynamic bottleneck proposed by the project is based on refusal degradation).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn


LayerKey = Hashable
StateValues = Union[
    torch.Tensor,
    Mapping[LayerKey, torch.Tensor],
    Sequence[torch.Tensor],
]


@dataclass(frozen=True)
class DualSafetyScores:
    """Separate harmfulness-recognition and refusal-execution scores."""

    harmfulness: StateValues
    refusal: StateValues

    def as_dict(self) -> Mapping[str, StateValues]:
        """Return the two named states without synthesizing a combined score."""

        return {
            "harmfulness": self.harmfulness,
            "refusal": self.refusal,
        }


class LinearSafetyProbe(nn.Module):
    """One binary linear probe whose weight also defines a state direction."""

    def __init__(
        self,
        hidden_size: int,
        *,
        direction: Optional[torch.Tensor] = None,
        bias: Union[float, torch.Tensor] = 0.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("hidden_size must be an integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.linear = nn.Linear(hidden_size, 1)
        if direction is not None:
            direction = torch.as_tensor(direction)
            if tuple(direction.shape) != (hidden_size,):
                raise ValueError(
                    f"direction must have shape ({hidden_size},), got "
                    f"{tuple(direction.shape)}"
                )
            with torch.no_grad():
                self.linear.weight.copy_(direction.reshape(1, -1))
        with torch.no_grad():
            self.linear.bias.copy_(torch.as_tensor(bias).reshape(1))
        self.requires_grad_(trainable)

    @property
    def direction(self) -> torch.Tensor:
        """The raw probe direction, with shape ``[hidden]``."""

        return self.linear.weight.squeeze(0)

    def logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.shape[-1] != self.linear.in_features:
            raise ValueError(
                f"expected hidden dimension {self.linear.in_features}, got "
                f"{hidden_state.shape[-1]}"
            )
        return self.linear(hidden_state).squeeze(-1)

    def forward(
        self,
        hidden_state: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> torch.Tensor:
        logits = self.logits(hidden_state)
        return logits if return_logits else torch.sigmoid(logits)

    def project(
        self,
        hidden_state: torch.Tensor,
        *,
        center: Optional[torch.Tensor] = None,
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Project a state on the probe direction without applying sigmoid."""

        if hidden_state.shape[-1] != self.linear.in_features:
            raise ValueError(
                f"expected hidden dimension {self.linear.in_features}, got "
                f"{hidden_state.shape[-1]}"
            )
        centered = hidden_state if center is None else hidden_state - center
        direction = self.direction
        if normalize:
            direction = direction / direction.norm().clamp_min(eps)
        return torch.einsum("...d,d->...", centered, direction)


class LayerwiseLinearSafetyProbe(nn.Module):
    """Independent linear probe directions for explicitly named layers."""

    def __init__(
        self,
        hidden_sizes: Mapping[LayerKey, int],
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must name at least one layer")
        self.layer_keys = tuple(hidden_sizes.keys())
        self._module_names = {
            key: f"probe_{position}" for position, key in enumerate(self.layer_keys)
        }
        self.probes = nn.ModuleDict(
            {
                self._module_names[key]: LinearSafetyProbe(
                    hidden_sizes[key], trainable=trainable
                )
                for key in self.layer_keys
            }
        )

    def probe_for(self, layer: LayerKey) -> LinearSafetyProbe:
        if layer not in self._module_names:
            raise KeyError(f"no safety probe was configured for layer {layer!r}")
        return self.probes[self._module_names[layer]]

    def direction(self, layer: LayerKey) -> torch.Tensor:
        return self.probe_for(layer).direction

    def forward(
        self,
        hidden_states: Mapping[LayerKey, torch.Tensor],
        *,
        return_logits: bool = False,
    ) -> "OrderedDict[LayerKey, torch.Tensor]":
        if not isinstance(hidden_states, Mapping):
            raise TypeError("a layerwise probe requires a layer-to-tensor mapping")
        result: "OrderedDict[LayerKey, torch.Tensor]" = OrderedDict()
        for layer, hidden_state in hidden_states.items():
            result[layer] = self.probe_for(layer)(
                hidden_state, return_logits=return_logits
            )
        return result

    def project(
        self,
        hidden_states: Mapping[LayerKey, torch.Tensor],
        *,
        centers: Optional[Mapping[LayerKey, torch.Tensor]] = None,
        normalize: bool = True,
    ) -> "OrderedDict[LayerKey, torch.Tensor]":
        result: "OrderedDict[LayerKey, torch.Tensor]" = OrderedDict()
        for layer, hidden_state in hidden_states.items():
            center = None if centers is None else centers[layer]
            result[layer] = self.probe_for(layer).project(
                hidden_state, center=center, normalize=normalize
            )
        return result


def _apply_shared_probe(
    probe: LinearSafetyProbe,
    hidden_states: StateValues,
    *,
    return_logits: bool,
) -> StateValues:
    if isinstance(hidden_states, torch.Tensor):
        return probe(hidden_states, return_logits=return_logits)
    if isinstance(hidden_states, Mapping):
        return OrderedDict(
            (
                layer,
                probe(hidden_state, return_logits=return_logits),
            )
            for layer, hidden_state in hidden_states.items()
        )
    if isinstance(hidden_states, tuple):
        return tuple(
            probe(hidden_state, return_logits=return_logits)
            for hidden_state in hidden_states
        )
    if isinstance(hidden_states, list):
        return [
            probe(hidden_state, return_logits=return_logits)
            for hidden_state in hidden_states
        ]
    raise TypeError("hidden_states must be a tensor, mapping, tuple, or list")


def _project_shared_probe(
    probe: LinearSafetyProbe,
    hidden_states: StateValues,
    *,
    centers: Optional[StateValues],
    normalize: bool,
) -> StateValues:
    if isinstance(hidden_states, torch.Tensor):
        if centers is not None and not isinstance(centers, torch.Tensor):
            raise TypeError("tensor states require a tensor center")
        return probe.project(hidden_states, center=centers, normalize=normalize)
    if isinstance(hidden_states, Mapping):
        if centers is not None and not isinstance(centers, Mapping):
            raise TypeError("mapping states require mapping centers")
        return OrderedDict(
            (
                layer,
                probe.project(
                    hidden_state,
                    center=None if centers is None else centers[layer],
                    normalize=normalize,
                ),
            )
            for layer, hidden_state in hidden_states.items()
        )
    if isinstance(hidden_states, (tuple, list)):
        if centers is not None and not isinstance(centers, (tuple, list)):
            raise TypeError("sequence states require sequence centers")
        values = [
            probe.project(
                hidden_state,
                center=None if centers is None else centers[index],
                normalize=normalize,
            )
            for index, hidden_state in enumerate(hidden_states)
        ]
        return tuple(values) if isinstance(hidden_states, tuple) else values
    raise TypeError("hidden_states must be a tensor, mapping, tuple, or list")


class DualSafetyStateScorer(nn.Module):
    """Score harmfulness and refusal with two strictly independent probes.

    ``hidden_size`` may be a single dimension (one shared probe direction per
    state) or ``layer -> dimension`` (one direction per state and layer).
    Alternatively, callers may inject two already-trained probes. Reusing the
    same module for both states is rejected because it would erase the intended
    harmfulness/refusal distinction.
    """

    def __init__(
        self,
        hidden_size: Optional[Union[int, Mapping[LayerKey, int]]] = None,
        *,
        harmfulness_probe: Optional[
            Union[LinearSafetyProbe, LayerwiseLinearSafetyProbe]
        ] = None,
        refusal_probe: Optional[
            Union[LinearSafetyProbe, LayerwiseLinearSafetyProbe]
        ] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        supplied = harmfulness_probe is not None or refusal_probe is not None
        if supplied:
            if harmfulness_probe is None or refusal_probe is None:
                raise ValueError("both harmfulness_probe and refusal_probe are required")
            if hidden_size is not None:
                raise ValueError("hidden_size cannot be combined with supplied probes")
        else:
            if hidden_size is None:
                raise ValueError("hidden_size or two explicit probes are required")
            if isinstance(hidden_size, Mapping):
                harmfulness_probe = LayerwiseLinearSafetyProbe(
                    hidden_size, trainable=trainable
                )
                refusal_probe = LayerwiseLinearSafetyProbe(
                    hidden_size, trainable=trainable
                )
            else:
                harmfulness_probe = LinearSafetyProbe(
                    hidden_size, trainable=trainable
                )
                refusal_probe = LinearSafetyProbe(
                    hidden_size, trainable=trainable
                )
        if harmfulness_probe is refusal_probe:
            raise ValueError("harmfulness and refusal must use distinct probe modules")
        self.harmfulness_probe = harmfulness_probe
        self.refusal_probe = refusal_probe

    @staticmethod
    def _score(
        probe: Union[LinearSafetyProbe, LayerwiseLinearSafetyProbe],
        hidden_states: StateValues,
        *,
        return_logits: bool,
    ) -> StateValues:
        if isinstance(probe, LayerwiseLinearSafetyProbe):
            if not isinstance(hidden_states, Mapping):
                raise TypeError("layerwise safety probes require mapping hidden states")
            return probe(hidden_states, return_logits=return_logits)
        return _apply_shared_probe(
            probe, hidden_states, return_logits=return_logits
        )

    def score_harmfulness(
        self,
        hidden_states: StateValues,
        *,
        return_logits: bool = False,
    ) -> StateValues:
        return self._score(
            self.harmfulness_probe,
            hidden_states,
            return_logits=return_logits,
        )

    def score_refusal(
        self,
        hidden_states: StateValues,
        *,
        return_logits: bool = False,
    ) -> StateValues:
        return self._score(
            self.refusal_probe,
            hidden_states,
            return_logits=return_logits,
        )

    def forward(
        self,
        hidden_states: StateValues,
        *,
        return_logits: bool = False,
    ) -> DualSafetyScores:
        return DualSafetyScores(
            harmfulness=self.score_harmfulness(
                hidden_states, return_logits=return_logits
            ),
            refusal=self.score_refusal(
                hidden_states, return_logits=return_logits
            ),
        )

    @staticmethod
    def _project(
        probe: Union[LinearSafetyProbe, LayerwiseLinearSafetyProbe],
        hidden_states: StateValues,
        *,
        centers: Optional[StateValues],
        normalize: bool,
    ) -> StateValues:
        if isinstance(probe, LayerwiseLinearSafetyProbe):
            if not isinstance(hidden_states, Mapping):
                raise TypeError("layerwise safety probes require mapping hidden states")
            if centers is not None and not isinstance(centers, Mapping):
                raise TypeError("layerwise direction centers must be a mapping")
            return probe.project(
                hidden_states, centers=centers, normalize=normalize
            )
        return _project_shared_probe(
            probe,
            hidden_states,
            centers=centers,
            normalize=normalize,
        )

    def direction_scores(
        self,
        hidden_states: StateValues,
        *,
        harmfulness_centers: Optional[StateValues] = None,
        refusal_centers: Optional[StateValues] = None,
        normalize: bool = True,
    ) -> DualSafetyScores:
        """Return two named direction projections, still without aggregation."""

        return DualSafetyScores(
            harmfulness=self._project(
                self.harmfulness_probe,
                hidden_states,
                centers=harmfulness_centers,
                normalize=normalize,
            ),
            refusal=self._project(
                self.refusal_probe,
                hidden_states,
                centers=refusal_centers,
                normalize=normalize,
            ),
        )


def _reject_dual(value: Any, argument: str) -> None:
    if isinstance(value, DualSafetyScores):
        raise TypeError(
            f"{argument} must be an explicitly selected state; use "
            f"{argument}.refusal or {argument}.harmfulness"
        )


def compute_safety_gaps(
    reference_refusal: StateValues,
    current_refusal: StateValues,
) -> StateValues:
    """Compute refusal degradation ``R_reference - R_current`` per layer.

    The argument names make the intended state explicit. Passing a complete
    :class:`DualSafetyScores` object is rejected rather than silently blending
    harmfulness and refusal.
    """

    _reject_dual(reference_refusal, "reference_refusal")
    _reject_dual(current_refusal, "current_refusal")
    if isinstance(current_refusal, torch.Tensor):
        if not isinstance(reference_refusal, torch.Tensor):
            raise TypeError("tensor refusal states require a tensor reference")
        return reference_refusal - current_refusal
    if isinstance(current_refusal, Mapping):
        if not isinstance(reference_refusal, Mapping):
            raise TypeError("mapping refusal states require a mapping reference")
        if set(reference_refusal) != set(current_refusal):
            raise KeyError("reference and current refusal layers do not match")
        return OrderedDict(
            (
                layer,
                reference_refusal[layer] - current,
            )
            for layer, current in current_refusal.items()
        )
    if isinstance(current_refusal, (tuple, list)):
        if not isinstance(reference_refusal, (tuple, list)):
            raise TypeError("sequence refusal states require a sequence reference")
        if len(reference_refusal) != len(current_refusal):
            raise ValueError("reference and current refusal lengths do not match")
        values = [
            reference - current
            for reference, current in zip(reference_refusal, current_refusal)
        ]
        return tuple(values) if isinstance(current_refusal, tuple) else values
    raise TypeError("refusal states must be a tensor, mapping, tuple, or list")


def _stack_layers(
    values: StateValues,
    *,
    layer_dim: int,
) -> Tuple[torch.Tensor, Optional[Tuple[LayerKey, ...]], Optional[type]]:
    _reject_dual(values, "values")
    if isinstance(values, torch.Tensor):
        if values.ndim == 0:
            raise ValueError("a layer tensor must have at least one dimension")
        return values, None, None
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("layer mapping cannot be empty")
        keys = tuple(values.keys())
        try:
            stacked = torch.stack(tuple(values.values()), dim=layer_dim)
        except RuntimeError as exc:
            raise ValueError("all layer values must have matching shapes") from exc
        return stacked, keys, type(values)
    if isinstance(values, (tuple, list)):
        if not values:
            raise ValueError("layer sequence cannot be empty")
        try:
            stacked = torch.stack(tuple(values), dim=layer_dim)
        except RuntimeError as exc:
            raise ValueError("all layer values must have matching shapes") from exc
        return stacked, None, type(values)
    raise TypeError("layer values must be a tensor, mapping, tuple, or list")


def compute_layer_weights(
    gaps: StateValues,
    temperature: Union[float, torch.Tensor] = 1.0,
    *,
    layer_dim: int = -1,
) -> StateValues:
    """Compute differentiable dynamic weights ``softmax(gap / temperature)``."""

    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1 or bool((temperature.detach() <= 0).item()):
            raise ValueError("temperature must be a positive scalar")
    elif temperature <= 0:
        raise ValueError("temperature must be positive")
    stacked, keys, container_type = _stack_layers(gaps, layer_dim=layer_dim)
    weights = torch.softmax(stacked / temperature, dim=layer_dim)
    if keys is not None:
        values = weights.unbind(dim=layer_dim)
        return OrderedDict(zip(keys, values))
    if container_type in (tuple, list):
        values = weights.unbind(dim=layer_dim)
        return tuple(values) if container_type is tuple else list(values)
    return weights


@dataclass(frozen=True)
class BottleneckResult:
    """Argmax result for a dynamic safety bottleneck.

    ``indices`` and ``gaps`` retain any batch dimensions. For a scalar result,
    ``layer`` resolves the index back to the original mapping key.
    """

    indices: torch.Tensor
    gaps: torch.Tensor
    layer_keys: Optional[Tuple[LayerKey, ...]] = None

    @property
    def layer(self) -> LayerKey:
        if self.indices.numel() != 1:
            raise ValueError("batched bottlenecks do not have one scalar layer")
        index = int(self.indices.item())
        return self.layer_keys[index] if self.layer_keys is not None else index

    @property
    def gap(self) -> torch.Tensor:
        if self.gaps.numel() != 1:
            raise ValueError("batched bottlenecks do not have one scalar gap")
        return self.gaps.reshape(())

    def resolved_layers(self) -> Tuple[LayerKey, ...]:
        """Flatten batched indices and resolve them to mapping keys if present."""

        raw = self.indices.detach().reshape(-1).tolist()
        if self.layer_keys is None:
            return tuple(int(index) for index in raw)
        return tuple(self.layer_keys[int(index)] for index in raw)


def dynamic_bottleneck(
    gaps: StateValues,
    *,
    layer_dim: int = -1,
) -> BottleneckResult:
    """Locate the maximum refusal-degradation layer for the current attack state."""

    stacked, keys, _ = _stack_layers(gaps, layer_dim=layer_dim)
    values, indices = torch.max(stacked, dim=layer_dim)
    return BottleneckResult(indices=indices, gaps=values, layer_keys=keys)


def _weighted_layer_aggregate(
    values: StateValues,
    layer_weights: Optional[StateValues],
    *,
    layer_dim: int,
) -> torch.Tensor:
    stacked, keys, _ = _stack_layers(values, layer_dim=layer_dim)
    if layer_weights is None:
        return stacked.mean(dim=layer_dim)
    weight_tensor, weight_keys, _ = _stack_layers(
        layer_weights, layer_dim=layer_dim
    )
    if keys != weight_keys:
        raise KeyError("state values and layer weights must use the same layer keys")
    if stacked.shape != weight_tensor.shape:
        try:
            weight_tensor = torch.broadcast_to(weight_tensor, stacked.shape)
        except RuntimeError as exc:
            raise ValueError("layer weights cannot be broadcast to state values") from exc
    return (stacked * weight_tensor).sum(dim=layer_dim)


def differentiable_state_loss(
    current_refusal: StateValues,
    layer_weights: Optional[StateValues] = None,
    *,
    layer_dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return a refusal-suppression objective for gradient minimization.

    Minimizing this loss lowers the explicitly supplied refusal state. Dynamic
    weights from :func:`compute_layer_weights` remain attached to the graph, so
    callers may choose whether to detach them depending on their optimization
    design. Harmfulness is never folded into this objective implicitly.
    """

    aggregated = _weighted_layer_aggregate(
        current_refusal, layer_weights, layer_dim=layer_dim
    )
    if reduction == "none":
        return aggregated
    if reduction == "mean":
        return aggregated.mean()
    if reduction == "sum":
        return aggregated.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def safety_gap_attack_loss(
    gaps: StateValues,
    layer_weights: Optional[StateValues] = None,
    *,
    layer_dim: int = -1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Return ``-weighted_gap`` so minimization increases refusal degradation."""

    loss = differentiable_state_loss(
        gaps,
        layer_weights,
        layer_dim=layer_dim,
        reduction=reduction,
    )
    return -loss


# Semantic alias for callers that prefer the attack action in the name.
refusal_suppression_loss = differentiable_state_loss


__all__ = [
    "BottleneckResult",
    "DualSafetyScores",
    "DualSafetyStateScorer",
    "LayerwiseLinearSafetyProbe",
    "LinearSafetyProbe",
    "compute_layer_weights",
    "compute_safety_gaps",
    "differentiable_state_loss",
    "dynamic_bottleneck",
    "refusal_suppression_loss",
    "safety_gap_attack_loss",
]
