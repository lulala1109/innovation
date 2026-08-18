"""Utilities for selecting, pooling, collecting, and patching activations.

The functions in this module do not depend on a particular Transformers model.
They operate on ordinary tensors and :class:`torch.nn.Module` hooks, which keeps
the Qwen integration thin and makes the same analysis code reusable elsewhere.
Unless ``detach=True`` is requested, collection and patching preserve autograd so
the resulting states can participate in an input-space attack loss.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn


LayerKey = Hashable
LayerSelection = Optional[Union[int, slice, str, Iterable[int]]]
TokenSelection = Optional[
    Union[int, slice, Tuple[Optional[int], Optional[int]], Sequence[int], torch.Tensor]
]


def resolve_layer_indices(
    num_layers: int,
    layers: LayerSelection = None,
) -> Tuple[int, ...]:
    """Resolve a layer selector to unique non-negative indices.

    Supported string forms include ``"all"``, ``"last"``, comma-separated
    indices (``"1,4,-1"``), and Python-style slices (``"2:10:2"``). Negative
    indices are resolved relative to ``num_layers``.
    """

    if isinstance(num_layers, bool) or not isinstance(num_layers, int):
        raise TypeError("num_layers must be an integer")
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")

    raw_indices: Iterable[int]
    if layers is None or layers == "all":
        raw_indices = range(num_layers)
    elif isinstance(layers, bool):
        raise TypeError("boolean values are not valid layer selectors")
    elif isinstance(layers, int):
        raw_indices = (layers,)
    elif isinstance(layers, slice):
        raw_indices = range(num_layers)[layers]
    elif isinstance(layers, str):
        selector = layers.strip().lower()
        if selector == "last":
            raw_indices = (-1,)
        elif ":" in selector:
            if "," in selector:
                raise ValueError("a layer slice cannot be mixed with commas")
            pieces = selector.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"invalid layer slice: {layers!r}")
            parsed = [int(piece) if piece else None for piece in pieces]
            while len(parsed) < 3:
                parsed.append(None)
            raw_indices = range(num_layers)[slice(*parsed)]
        elif selector:
            raw_indices = tuple(int(piece.strip()) for piece in selector.split(","))
        else:
            raise ValueError("layer selector cannot be empty")
    else:
        raw_indices = layers

    result = []
    seen = set()
    for index in raw_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("layer indices must be integers")
        normalized = index + num_layers if index < 0 else index
        if normalized < 0 or normalized >= num_layers:
            raise IndexError(
                f"layer index {index} is out of range for {num_layers} layers"
            )
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def _token_indices(
    sequence_length: int,
    selection: TokenSelection,
    device: torch.device,
) -> torch.Tensor:
    if selection is None:
        indices = torch.arange(sequence_length, device=device)
    elif isinstance(selection, bool):
        raise TypeError("boolean values are not valid token selectors")
    elif isinstance(selection, int):
        index = selection + sequence_length if selection < 0 else selection
        indices = torch.tensor([index], device=device)
    elif isinstance(selection, slice):
        indices = torch.arange(sequence_length, device=device)[selection]
    elif isinstance(selection, tuple) and len(selection) == 2 and all(
        item is None or isinstance(item, int) for item in selection
    ):
        indices = torch.arange(sequence_length, device=device)[
            slice(selection[0], selection[1])
        ]
    elif isinstance(selection, torch.Tensor):
        if selection.dtype == torch.bool:
            if selection.ndim != 1 or selection.numel() != sequence_length:
                raise ValueError("a boolean token selector must match sequence length")
            indices = selection.to(device=device).nonzero(as_tuple=True)[0]
        else:
            if selection.ndim != 1:
                raise ValueError("token index tensor must be one-dimensional")
            indices = selection.to(device=device, dtype=torch.long)
    else:
        indices = torch.tensor(tuple(selection), device=device, dtype=torch.long)

    if indices.numel() == 0:
        raise ValueError("token selection is empty")
    indices = indices.to(dtype=torch.long)
    indices = torch.where(indices < 0, indices + sequence_length, indices)
    if bool(((indices < 0) | (indices >= sequence_length)).any()):
        raise IndexError("token selector contains an out-of-range index")
    return indices


def pool_tokens(
    hidden_state: torch.Tensor,
    *,
    pooling: str = "mean",
    attention_mask: Optional[torch.Tensor] = None,
    token_selection: TokenSelection = None,
) -> torch.Tensor:
    """Pool the sequence axis of a hidden-state tensor.

    ``hidden_state`` may be ``[tokens, hidden]`` or any batched shape ending in
    ``[..., tokens, hidden]``. The mask, when supplied, must have the matching
    ``[..., tokens]`` shape. Supported pooling modes are ``mean``, ``max``,
    ``first``, ``last``, and ``none``. The operation remains differentiable.
    """

    if not isinstance(hidden_state, torch.Tensor) or hidden_state.ndim < 2:
        raise ValueError("hidden_state must have shape [..., tokens, hidden]")
    sequence_length = hidden_state.shape[-2]
    indices = _token_indices(
        sequence_length, token_selection, hidden_state.device
    )
    selected = hidden_state.index_select(-2, indices)

    selected_mask: Optional[torch.Tensor] = None
    if attention_mask is not None:
        expected_shape = hidden_state.shape[:-1]
        if tuple(attention_mask.shape) != tuple(expected_shape):
            raise ValueError(
                "attention_mask must have shape hidden_state.shape[:-1]; "
                f"got {tuple(attention_mask.shape)} and {tuple(expected_shape)}"
            )
        selected_mask = attention_mask.to(device=hidden_state.device, dtype=torch.bool)
        selected_mask = selected_mask.index_select(-1, indices)
        if bool((selected_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every sample must retain at least one unmasked token")

    method = pooling.lower()
    if method == "none":
        return selected
    if method == "mean":
        if selected_mask is None:
            return selected.mean(dim=-2)
        weights = selected_mask.to(dtype=selected.dtype).unsqueeze(-1)
        return (selected * weights).sum(dim=-2) / weights.sum(dim=-2)
    if method == "max":
        if selected_mask is None:
            return selected.max(dim=-2).values
        masked = selected.masked_fill(
            ~selected_mask.unsqueeze(-1), torch.finfo(selected.dtype).min
        )
        return masked.max(dim=-2).values
    if method not in ("first", "last"):
        raise ValueError(
            "pooling must be one of: mean, max, first, last, none"
        )

    if selected_mask is None:
        token_index = 0 if method == "first" else selected.shape[-2] - 1
        return selected.select(-2, token_index)

    positions = torch.arange(selected.shape[-2], device=selected.device)
    view_shape = (1,) * (selected_mask.ndim - 1) + (selected.shape[-2],)
    positions = positions.view(view_shape).expand_as(selected_mask)
    if method == "first":
        positions = positions.masked_fill(~selected_mask, selected.shape[-2])
        chosen = positions.min(dim=-1).values
    else:
        positions = positions.masked_fill(~selected_mask, -1)
        chosen = positions.max(dim=-1).values
    gather_index = chosen.unsqueeze(-1).unsqueeze(-1).expand(
        *chosen.shape, 1, selected.shape[-1]
    )
    return selected.gather(-2, gather_index).squeeze(-2)


# A descriptive alias for callers that think in hidden-state terminology.
pool_hidden_state = pool_tokens


def collect_hidden_states(
    hidden_states: Union[Mapping[LayerKey, torch.Tensor], Sequence[torch.Tensor]],
    *,
    layers: Optional[Iterable[LayerKey]] = None,
    pooling: str = "mean",
    attention_mask: Optional[torch.Tensor] = None,
    token_selection: TokenSelection = None,
    sequence_has_embedding: bool = False,
) -> "OrderedDict[LayerKey, torch.Tensor]":
    """Select and pool hidden states into a stable ``layer -> tensor`` mapping.

    For Hugging Face-style sequences, set ``sequence_has_embedding=True`` to
    exclude element zero (the embedding output) and number transformer layers
    from zero. Mapping inputs preserve their existing layer identifiers.
    """

    if isinstance(hidden_states, Mapping):
        available: "OrderedDict[LayerKey, torch.Tensor]" = OrderedDict(hidden_states)
        selected_keys = tuple(available.keys()) if layers is None else tuple(layers)
    elif isinstance(hidden_states, Sequence) and not isinstance(
        hidden_states, (str, bytes)
    ):
        values = tuple(hidden_states)
        if sequence_has_embedding:
            values = values[1:]
        available = OrderedDict(enumerate(values))
        if layers is None:
            selected_keys = tuple(available.keys())
        else:
            requested = tuple(layers)
            selected_keys = resolve_layer_indices(len(values), requested)
    else:
        raise TypeError("hidden_states must be a mapping or a tensor sequence")

    result: "OrderedDict[LayerKey, torch.Tensor]" = OrderedDict()
    for key in selected_keys:
        if key not in available:
            raise KeyError(f"hidden state for layer {key!r} is unavailable")
        result[key] = pool_tokens(
            available[key],
            pooling=pooling,
            attention_mask=attention_mask,
            token_selection=token_selection,
        )
    return result


@dataclass(frozen=True)
class HiddenStateCollector:
    """Reusable configuration for :func:`collect_hidden_states`."""

    layers: Optional[Tuple[LayerKey, ...]] = None
    pooling: str = "mean"
    token_selection: TokenSelection = None
    sequence_has_embedding: bool = False
    detach: bool = False

    def __call__(
        self,
        hidden_states: Union[
            Mapping[LayerKey, torch.Tensor], Sequence[torch.Tensor]
        ],
        *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> "OrderedDict[LayerKey, torch.Tensor]":
        collected = collect_hidden_states(
            hidden_states,
            layers=self.layers,
            pooling=self.pooling,
            attention_mask=attention_mask,
            token_selection=self.token_selection,
            sequence_has_embedding=self.sequence_has_embedding,
        )
        if self.detach:
            return OrderedDict(
                (key, value.detach()) for key, value in collected.items()
            )
        return collected


def _extract_activation(
    output: Any,
    selector: Optional[Union[int, str]],
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """Extract one tensor and return a function that rebuilds the output."""

    if isinstance(output, torch.Tensor):
        if selector not in (None, 0):
            raise KeyError("a tensor output only supports selector None or 0")
        return output, lambda replacement: replacement

    if isinstance(output, tuple):
        index = 0 if selector is None else selector
        if not isinstance(index, int):
            raise TypeError("tuple outputs require an integer output selector")
        activation = output[index]
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"tuple output {index} is not a tensor")

        def rebuild_tuple(replacement: torch.Tensor) -> Any:
            values = list(output)
            values[index] = replacement
            if hasattr(output, "_fields"):
                return type(output)(*values)
            return tuple(values)

        return activation, rebuild_tuple

    if isinstance(output, list):
        index = 0 if selector is None else selector
        if not isinstance(index, int):
            raise TypeError("list outputs require an integer output selector")
        activation = output[index]
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"list output {index} is not a tensor")

        def rebuild_list(replacement: torch.Tensor) -> Any:
            values = list(output)
            values[index] = replacement
            return values

        return activation, rebuild_list

    if isinstance(output, Mapping):
        if selector is None:
            tensor_keys = [
                key for key, value in output.items() if isinstance(value, torch.Tensor)
            ]
            if len(tensor_keys) != 1:
                raise KeyError(
                    "mapping output requires output_selector when it does not "
                    "contain exactly one tensor"
                )
            key: Union[int, str] = tensor_keys[0]
        else:
            key = selector
        activation = output[key]
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"mapping output {key!r} is not a tensor")

        def rebuild_mapping(replacement: torch.Tensor) -> Any:
            try:
                values: MutableMapping[Any, Any] = output.copy()
            except AttributeError:
                values = dict(output)
            values[key] = replacement
            # Plain mappings are enough for nn.Module composition. Hugging Face
            # ModelOutput instances generally implement copy while preserving
            # their type; fall back to the copied mapping if reconstruction is
            # not supported by a custom mapping implementation.
            return values

        return activation, rebuild_mapping

    if isinstance(selector, str) and hasattr(output, selector):
        activation = getattr(output, selector)
        if not isinstance(activation, torch.Tensor):
            raise TypeError(f"output attribute {selector!r} is not a tensor")

        def rebuild_attribute(replacement: torch.Tensor) -> Any:
            # Arbitrary result objects cannot be reconstructed safely. Mutating
            # the designated attribute is the least surprising generic behavior
            # and matches how downstream modules access such containers.
            setattr(output, selector, replacement)
            return output

        return activation, rebuild_attribute

    raise TypeError(
        "module output must be a tensor/container, or output_selector must name "
        "a tensor attribute"
    )


PatchValue = Union[
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]


@dataclass(frozen=True)
class ActivationPatch:
    """Description of one module-output activation replacement."""

    replacement: PatchValue
    token_selection: TokenSelection = None
    output_selector: Optional[Union[int, str]] = None
    detach_replacement: bool = False


def _apply_activation_patch(
    activation: torch.Tensor,
    patch: ActivationPatch,
) -> torch.Tensor:
    replacement = (
        patch.replacement(activation)
        if callable(patch.replacement)
        else patch.replacement
    )
    if not isinstance(replacement, torch.Tensor):
        raise TypeError("activation replacement callable must return a tensor")
    replacement = replacement.to(
        device=activation.device, dtype=activation.dtype
    )
    if patch.detach_replacement:
        replacement = replacement.detach()

    if patch.token_selection is None:
        if replacement.shape != activation.shape:
            try:
                replacement = torch.broadcast_to(replacement, activation.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "replacement cannot be broadcast to the activation shape"
                ) from exc
        return replacement

    if activation.ndim < 2:
        raise ValueError("token-selective patching requires [..., tokens, hidden]")
    indices = _token_indices(
        activation.shape[-2], patch.token_selection, activation.device
    )
    target = activation.index_select(-2, indices)
    if replacement.shape == activation.shape:
        replacement = replacement.index_select(-2, indices)
    elif replacement.shape != target.shape:
        try:
            replacement = torch.broadcast_to(replacement, target.shape)
        except RuntimeError as exc:
            raise ValueError(
                "replacement cannot be broadcast to the selected activation"
            ) from exc
    patched = activation.clone()
    patched.index_copy_(-2, indices, replacement)
    return patched


class ActivationPatcher(AbstractContextManager):
    """Temporarily patch module outputs while the remainder of forward runs.

    ``patches`` maps either module objects or module names to
    :class:`ActivationPatch` objects (a raw tensor/callable is accepted as a
    shorthand). String module names require ``model`` and are resolved through
    ``model.named_modules()``. Hooks are always removed on context exit.
    """

    def __init__(
        self,
        patches: Mapping[
            Union[str, nn.Module], Union[ActivationPatch, PatchValue]
        ],
        *,
        model: Optional[nn.Module] = None,
    ) -> None:
        if not patches:
            raise ValueError("at least one activation patch is required")
        named_modules = dict(model.named_modules()) if model is not None else {}
        resolved: Dict[nn.Module, ActivationPatch] = {}
        for target, value in patches.items():
            if isinstance(target, str):
                if model is None:
                    raise ValueError("model is required for named module patches")
                if target not in named_modules:
                    raise KeyError(f"model has no module named {target!r}")
                module = named_modules[target]
            elif isinstance(target, nn.Module):
                module = target
            else:
                raise TypeError("patch targets must be modules or module names")
            patch = value if isinstance(value, ActivationPatch) else ActivationPatch(value)
            if module in resolved:
                raise ValueError("each module may have only one active patch")
            resolved[module] = patch
        self.patches = resolved
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationPatcher":
        if self._handles:
            raise RuntimeError("ActivationPatcher context is already active")
        for module, patch in self.patches.items():

            def hook(
                _module: nn.Module,
                _inputs: Tuple[Any, ...],
                output: Any,
                *,
                current_patch: ActivationPatch = patch,
            ) -> Any:
                activation, rebuild = _extract_activation(
                    output, current_patch.output_selector
                )
                return rebuild(_apply_activation_patch(activation, current_patch))

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        while self._handles:
            self._handles.pop().remove()
        return False


@contextmanager
def patch_activations(
    patches: Mapping[Union[str, nn.Module], Union[ActivationPatch, PatchValue]],
    *,
    model: Optional[nn.Module] = None,
) -> Iterator[ActivationPatcher]:
    """Context-manager convenience wrapper around :class:`ActivationPatcher`."""

    with ActivationPatcher(patches, model=model) as patcher:
        yield patcher


class ForwardActivationCollector(AbstractContextManager):
    """Collect output tensors from named layers during an ordinary forward pass.

    This hook-based collector is useful when a model cannot return
    ``output_hidden_states`` directly. By default it stores graph-connected
    tensors so a safety-state loss can differentiate back to the input.
    """

    def __init__(
        self,
        modules: Mapping[LayerKey, nn.Module],
        *,
        pooling: str = "mean",
        attention_mask: Optional[torch.Tensor] = None,
        token_selection: TokenSelection = None,
        output_selector: Optional[Union[int, str]] = None,
        detach: bool = False,
    ) -> None:
        if not modules:
            raise ValueError("at least one module must be collected")
        self.modules = OrderedDict(modules)
        self.pooling = pooling
        self.attention_mask = attention_mask
        self.token_selection = token_selection
        self.output_selector = output_selector
        self.detach = detach
        self.activations: "OrderedDict[LayerKey, torch.Tensor]" = OrderedDict()
        self._handles: list[Any] = []

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        module_names: Iterable[str],
        **kwargs: Any,
    ) -> "ForwardActivationCollector":
        named = dict(model.named_modules())
        modules: "OrderedDict[str, nn.Module]" = OrderedDict()
        for name in module_names:
            if name not in named:
                raise KeyError(f"model has no module named {name!r}")
            modules[name] = named[name]
        return cls(modules, **kwargs)

    def clear(self) -> None:
        self.activations.clear()

    def __enter__(self) -> "ForwardActivationCollector":
        if self._handles:
            raise RuntimeError("ForwardActivationCollector is already active")
        self.clear()
        for key, module in self.modules.items():

            def hook(
                _module: nn.Module,
                _inputs: Tuple[Any, ...],
                output: Any,
                *,
                current_key: LayerKey = key,
            ) -> None:
                activation, _ = _extract_activation(output, self.output_selector)
                pooled = pool_tokens(
                    activation,
                    pooling=self.pooling,
                    attention_mask=self.attention_mask,
                    token_selection=self.token_selection,
                )
                self.activations[current_key] = (
                    pooled.detach() if self.detach else pooled
                )

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        while self._handles:
            self._handles.pop().remove()
        return False


__all__ = [
    "ActivationPatch",
    "ActivationPatcher",
    "ForwardActivationCollector",
    "HiddenStateCollector",
    "collect_hidden_states",
    "patch_activations",
    "pool_hidden_state",
    "pool_tokens",
    "resolve_layer_indices",
]
