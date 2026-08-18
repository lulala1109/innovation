#!/usr/bin/env python3
"""Train independent layer-wise harmfulness and refusal linear probes.

The input is a local tensor checkpoint with this schema::

    {
        "hidden_states": {layer: Tensor[N, D]},
        "harmfulness_labels": Tensor[N],
        "refusal_labels": Tensor[N],
        "pair_ids": [str, ...],
    }

Rows sharing a ``pair_id`` are kept in the same train/validation partition.
Harmfulness and refusal use separate probe modules, optimizers, BCE-with-logits
losses, histories, and metrics; this script never creates a combined target or
combined safety score.

PyTorch is imported only inside runtime functions. Consequently ``--help`` can
be inspected without importing the model or tensor stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class GroupSplit:
    """Row indices and pair groups for a leakage-free partition."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_pair_ids: tuple[str, ...]
    validation_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedProbeData:
    """Validated CPU tensors ready for layer-wise probe training."""

    hidden_states: Mapping[Any, Any]
    harmfulness_labels: Any
    refusal_labels: Any
    pair_ids: tuple[str, ...]
    hidden_sizes: Mapping[Any, int]

    @property
    def num_samples(self) -> int:
        return len(self.pair_ids)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def split_pair_groups(
    pair_ids: Sequence[str],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> GroupSplit:
    """Deterministically split rows by whole ``pair_id`` groups.

    At least one group is assigned to each partition. Pair identifiers are
    sorted before seeded shuffling, so the result is independent of dictionary
    or set iteration order.
    """

    _validate_seed(seed)
    if isinstance(pair_ids, (str, bytes)) or not isinstance(pair_ids, Sequence):
        raise TypeError("pair_ids must be a sequence of strings")
    normalized: list[str] = []
    for index, pair_id in enumerate(pair_ids):
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"pair_ids[{index}] must be a non-empty string")
        normalized.append(pair_id.strip())
    if not normalized:
        raise ValueError("pair_ids must not be empty")
    if isinstance(validation_fraction, bool):
        raise TypeError("validation_fraction must be a number")
    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError) as exc:
        raise TypeError("validation_fraction must be a number") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be finite and within (0, 1)")

    groups = sorted(set(normalized))
    if len(groups) < 2:
        raise ValueError("pair_id group split requires at least two unique groups")
    random.Random(seed).shuffle(groups)
    validation_count = int(math.floor(len(groups) * fraction + 0.5))
    validation_count = max(1, min(len(groups) - 1, validation_count))
    validation_groups = frozenset(groups[:validation_count])
    train_groups = frozenset(groups[validation_count:])
    train_indices = tuple(
        index for index, pair_id in enumerate(normalized)
        if pair_id in train_groups
    )
    validation_indices = tuple(
        index for index, pair_id in enumerate(normalized)
        if pair_id in validation_groups
    )
    if not train_indices or not validation_indices:
        raise ValueError("group split produced an empty partition")
    if train_groups & validation_groups:
        raise AssertionError("pair_id leakage across train and validation splits")
    return GroupSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_pair_ids=tuple(sorted(train_groups)),
        validation_pair_ids=tuple(sorted(validation_groups)),
    )


def load_training_payload(path: str | Path) -> Mapping[str, Any]:
    """Load a tensor payload on CPU using PyTorch's restricted loader."""

    import torch

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Probe training payload not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe probe loading requires a PyTorch version supporting "
            "torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Probe training payload must contain one mapping")
    return payload


def _validate_binary_labels(value: Any, *, name: str, count: int) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.shape[0] != count:
        raise ValueError(f"{name} must have shape [{count}]")
    if value.is_complex():
        raise TypeError(f"{name} must be real-valued")
    labels = value.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(labels).all()):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    if labels.unique().numel() != 2:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def validate_training_payload(payload: Mapping[str, Any]) -> ValidatedProbeData:
    """Strictly validate and normalize the documented training schema."""

    import torch

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    required = {
        "hidden_states",
        "harmfulness_labels",
        "refusal_labels",
        "pair_ids",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Probe training payload is missing: " + ", ".join(missing))
    raw_states = payload["hidden_states"]
    if not isinstance(raw_states, Mapping) or not raw_states:
        raise ValueError("hidden_states must be a non-empty layer-to-tensor mapping")

    states: "OrderedDict[Any, Any]" = OrderedDict()
    hidden_sizes: "OrderedDict[Any, int]" = OrderedDict()
    sample_count: Optional[int] = None
    for layer, value in raw_states.items():
        if isinstance(layer, bool) or not isinstance(layer, (int, str)):
            raise TypeError("hidden-state layer keys must be integers or strings")
        if isinstance(layer, str) and not layer.strip():
            raise ValueError("hidden-state string layer keys must not be blank")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"hidden_states[{layer!r}] must be a torch.Tensor")
        if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
            raise ValueError(
                f"hidden_states[{layer!r}] must have non-empty shape [N, D]"
            )
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"hidden_states[{layer!r}] must be floating point")
        state = value.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(state).all()):
            raise ValueError(f"hidden_states[{layer!r}] contains non-finite values")
        if sample_count is None:
            sample_count = int(state.shape[0])
        elif state.shape[0] != sample_count:
            raise ValueError("all hidden-state layers must use the same sample count")
        states[layer] = state.contiguous()
        hidden_sizes[layer] = int(state.shape[1])
    assert sample_count is not None

    raw_pair_ids = payload["pair_ids"]
    if isinstance(raw_pair_ids, (str, bytes)) or not isinstance(
        raw_pair_ids, Sequence
    ):
        raise TypeError("pair_ids must be a sequence of strings")
    if len(raw_pair_ids) != sample_count:
        raise ValueError(f"pair_ids must have length {sample_count}")
    pair_ids: list[str] = []
    for index, value in enumerate(raw_pair_ids):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pair_ids[{index}] must be a non-empty string")
        pair_ids.append(value.strip())

    harmfulness = _validate_binary_labels(
        payload["harmfulness_labels"],
        name="harmfulness_labels",
        count=sample_count,
    )
    refusal = _validate_binary_labels(
        payload["refusal_labels"],
        name="refusal_labels",
        count=sample_count,
    )
    return ValidatedProbeData(
        hidden_states=states,
        harmfulness_labels=harmfulness,
        refusal_labels=refusal,
        pair_ids=tuple(pair_ids),
        hidden_sizes=hidden_sizes,
    )


def _validate_hyperparameters(
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    _validate_seed(seed)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    for name, value, allow_zero in (
        ("learning_rate", learning_rate, False),
        ("weight_decay", weight_decay, True),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a number") from exc
        bound_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
        if not math.isfinite(numeric) or not bound_ok:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")


def _selected_states(data: ValidatedProbeData, indices: Any) -> Mapping[Any, Any]:
    return OrderedDict(
        (layer, state.index_select(0, indices))
        for layer, state in data.hidden_states.items()
    )


def _direction_epoch(
    *,
    scorer: Any,
    direction: str,
    states: Mapping[Any, Any],
    labels: Any,
    optimizer: Any,
    loss_function: Any,
) -> tuple[float, Mapping[str, float]]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    if direction == "harmfulness":
        logits = scorer.score_harmfulness(states, return_logits=True)
    elif direction == "refusal":
        logits = scorer.score_refusal(states, return_logits=True)
    else:
        raise ValueError("direction must be harmfulness or refusal")
    layer_losses = OrderedDict(
        (str(layer), loss_function(layer_logits, labels))
        for layer, layer_logits in logits.items()
    )
    # This sum combines layers for one explicitly named state only. H and R are
    # optimized in separate calls with disjoint modules and optimizers.
    objective = torch.stack(tuple(layer_losses.values())).sum()
    objective.backward()
    optimizer.step()
    return (
        float(objective.detach().item()),
        {layer: float(loss.detach().item()) for layer, loss in layer_losses.items()},
    )


def _direction_metrics(
    *,
    scorer: Any,
    direction: str,
    states: Mapping[Any, Any],
    labels: Any,
    loss_function: Any,
) -> Mapping[str, Mapping[str, float | int]]:
    import torch

    with torch.no_grad():
        if direction == "harmfulness":
            logits = scorer.score_harmfulness(states, return_logits=True)
        elif direction == "refusal":
            logits = scorer.score_refusal(states, return_logits=True)
        else:
            raise ValueError("direction must be harmfulness or refusal")
        metrics: "OrderedDict[str, Mapping[str, float | int]]" = OrderedDict()
        for layer, layer_logits in logits.items():
            predictions = (layer_logits >= 0).to(dtype=labels.dtype)
            metrics[str(layer)] = {
                "loss": float(loss_function(layer_logits, labels).item()),
                "accuracy": float((predictions == labels).float().mean().item()),
                "predicted_positive_rate": float(predictions.mean().item()),
                "label_positive_rate": float(labels.mean().item()),
                "sample_count": int(labels.numel()),
            }
    return metrics


def train_safety_probes(
    payload: Mapping[str, Any],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
    source_path: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Train two independent layer-wise probe families and return a checkpoint."""

    import torch
    from torch import nn

    from core.safety_state import DualSafetyStateScorer

    _validate_hyperparameters(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    data = validate_training_payload(payload)
    split = split_pair_groups(
        data.pair_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_indices = torch.tensor(split.train_indices, dtype=torch.long)
    validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
    train_harmfulness = data.harmfulness_labels.index_select(0, train_indices)
    train_refusal = data.refusal_labels.index_select(0, train_indices)
    for name, labels in (
        ("harmfulness", train_harmfulness),
        ("refusal", train_refusal),
    ):
        if labels.unique().numel() != 2:
            raise ValueError(
                f"group split leaves the {name} training partition without "
                "both binary classes; choose another seed/fraction or add groups"
            )

    random.seed(seed)
    torch.manual_seed(seed)
    scorer = DualSafetyStateScorer(
        hidden_size=data.hidden_sizes,
        trainable=True,
    ).to(device="cpu", dtype=torch.float32)
    harmfulness_optimizer = torch.optim.Adam(
        scorer.harmfulness_probe.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    refusal_optimizer = torch.optim.Adam(
        scorer.refusal_probe.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_states = _selected_states(data, train_indices)
    validation_states = _selected_states(data, validation_indices)
    histories: dict[str, dict[str, Any]] = {
        "harmfulness": {"objective": [], "by_layer": {}},
        "refusal": {"objective": [], "by_layer": {}},
    }

    for _ in range(epochs):
        harmfulness_objective, harmfulness_layers = _direction_epoch(
            scorer=scorer,
            direction="harmfulness",
            states=train_states,
            labels=train_harmfulness,
            optimizer=harmfulness_optimizer,
            loss_function=loss_function,
        )
        refusal_objective, refusal_layers = _direction_epoch(
            scorer=scorer,
            direction="refusal",
            states=train_states,
            labels=train_refusal,
            optimizer=refusal_optimizer,
            loss_function=loss_function,
        )
        histories["harmfulness"]["objective"].append(harmfulness_objective)
        histories["refusal"]["objective"].append(refusal_objective)
        for layer, value in harmfulness_layers.items():
            histories["harmfulness"]["by_layer"].setdefault(layer, []).append(value)
        for layer, value in refusal_layers.items():
            histories["refusal"]["by_layer"].setdefault(layer, []).append(value)

    validation_harmfulness = data.harmfulness_labels.index_select(
        0, validation_indices
    )
    validation_refusal = data.refusal_labels.index_select(0, validation_indices)
    metrics = {
        "harmfulness": {
            "train": _direction_metrics(
                scorer=scorer,
                direction="harmfulness",
                states=train_states,
                labels=train_harmfulness,
                loss_function=loss_function,
            ),
            "validation": _direction_metrics(
                scorer=scorer,
                direction="harmfulness",
                states=validation_states,
                labels=validation_harmfulness,
                loss_function=loss_function,
            ),
        },
        "refusal": {
            "train": _direction_metrics(
                scorer=scorer,
                direction="refusal",
                states=train_states,
                labels=train_refusal,
                loss_function=loss_function,
            ),
            "validation": _direction_metrics(
                scorer=scorer,
                direction="refusal",
                states=validation_states,
                labels=validation_refusal,
                loss_function=loss_function,
            ),
        },
    }
    state_dict = OrderedDict(
        (name, value.detach().cpu())
        for name, value in scorer.state_dict().items()
    )
    return {
        "format": "dual-safety-state-layerwise-linear-probes",
        "version": 1,
        # This exact key is consumed by batch_safety_attack._default_probe_loader.
        "hidden_sizes": dict(data.hidden_sizes),
        "state_dict": state_dict,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source_path": (
                None if source_path is None
                else str(Path(source_path).expanduser().resolve())
            ),
            "num_samples": data.num_samples,
            "num_layers": len(data.hidden_sizes),
            "seed": seed,
            "validation_fraction": float(validation_fraction),
            "split": {
                "train_indices": list(split.train_indices),
                "validation_indices": list(split.validation_indices),
                "train_pair_ids": list(split.train_pair_ids),
                "validation_pair_ids": list(split.validation_pair_ids),
            },
            "optimization": {
                "epochs": epochs,
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "loss": "BCEWithLogitsLoss",
                "harmfulness": histories["harmfulness"],
                "refusal": histories["refusal"],
            },
        },
        "metrics": metrics,
    }


def save_probe_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save a tensor-and-primitive checkpoint for weights-only loading."""

    import torch

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(checkpoint), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def train_probe_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    **training_options: Any,
) -> Mapping[str, Any]:
    """Safely load, train, and atomically save one compatible checkpoint."""

    payload = load_training_payload(input_path)
    checkpoint = train_safety_probes(
        payload,
        source_path=input_path,
        **training_options,
    )
    save_probe_checkpoint(checkpoint, output_path)
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local .pt training payload")
    parser.add_argument("--output", required=True, help="Output probe checkpoint")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = train_probe_checkpoint(
        args.input,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    summary = {
        "output": str(Path(args.output).expanduser().resolve()),
        "hidden_sizes": checkpoint["hidden_sizes"],
        "metrics": checkpoint["metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GroupSplit",
    "ValidatedProbeData",
    "build_parser",
    "load_training_payload",
    "main",
    "save_probe_checkpoint",
    "split_pair_groups",
    "train_probe_checkpoint",
    "train_safety_probes",
    "validate_training_payload",
]
