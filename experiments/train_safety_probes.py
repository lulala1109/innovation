#!/usr/bin/env python3
"""Train independent layer-wise harmfulness and refusal linear probes.

The input is a local tensor checkpoint with this schema::

    {
        "hidden_states": {layer: Tensor[N, D]},
        "harmfulness_labels": Tensor[N],
        "refusal_labels": Tensor[N],
        "pair_ids": [str, ...],
    }

Rows sharing a ``pair_id`` are kept in the same hold-out partition. The CLI
uses five-fold out-of-fold evaluation, pair-cluster confidence intervals, and
then fits the serialized probes on all eligible measurement-train pairs.
Harmfulness and refusal use separate probe modules, class-mean directions,
optimizers, losses, histories, and metrics; this script never creates a
combined target or combined safety score.

PyTorch is imported only inside runtime functions. Consequently ``--help`` can
be inspected without importing the model or tensor stack.
"""

from __future__ import annotations

import argparse
import hashlib
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
class GroupFold:
    """One deterministic pair-grouped out-of-fold partition."""

    fold: int
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
    row_metadata: Optional[tuple[Mapping[str, Any], ...]]
    payload_metadata: Mapping[str, Any]

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


def split_pair_group_folds(
    pair_ids: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 42,
) -> tuple[GroupFold, ...]:
    """Return deterministic K-fold partitions without pair leakage.

    Unique pair identifiers are seeded-shuffled and distributed round-robin,
    which keeps fold group counts within one of each other. Every row appears
    in exactly one validation fold and all rows for one pair stay together.
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
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise TypeError("folds must be an integer")
    groups = sorted(set(normalized))
    if not 2 <= folds <= len(groups):
        raise ValueError(
            f"folds must be within [2, {len(groups)}] for pair-group CV"
        )
    random.Random(seed).shuffle(groups)
    validation_groups: list[list[str]] = [[] for _ in range(folds)]
    for index, pair_id in enumerate(groups):
        validation_groups[index % folds].append(pair_id)

    result: list[GroupFold] = []
    all_groups = frozenset(groups)
    for fold, raw_validation in enumerate(validation_groups):
        validation_set = frozenset(raw_validation)
        train_set = all_groups - validation_set
        train_indices = tuple(
            index
            for index, pair_id in enumerate(normalized)
            if pair_id in train_set
        )
        validation_indices = tuple(
            index
            for index, pair_id in enumerate(normalized)
            if pair_id in validation_set
        )
        if not train_indices or not validation_indices:
            raise ValueError(f"group CV fold {fold} produced an empty partition")
        result.append(
            GroupFold(
                fold=fold,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_pair_ids=tuple(sorted(train_set)),
                validation_pair_ids=tuple(sorted(validation_set)),
            )
        )
    validation_rows = sorted(
        index for item in result for index in item.validation_indices
    )
    if validation_rows != list(range(len(normalized))):
        raise AssertionError("group CV did not assign every row exactly once")
    return tuple(result)


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
    normalized_row_metadata: Optional[tuple[Mapping[str, Any], ...]] = None
    if "row_metadata" in payload:
        raw_metadata = payload["row_metadata"]
        if isinstance(raw_metadata, (str, bytes)) or not isinstance(
            raw_metadata, Sequence
        ):
            raise TypeError("row_metadata must be a sequence of mappings")
        if len(raw_metadata) != sample_count:
            raise ValueError(f"row_metadata must have length {sample_count}")
        if any(
            isinstance(row, Mapping) and "measurement_split" in row
            for row in raw_metadata
        ) and not all(
            isinstance(row, Mapping) and "measurement_split" in row
            for row in raw_metadata
        ):
            raise ValueError(
                "row_metadata measurement_split provenance must be present on every row"
            )
        if any(
            isinstance(row, Mapping) and "stage1_role" in row
            for row in raw_metadata
        ) and not all(
            isinstance(row, Mapping) and "stage1_role" in row
            for row in raw_metadata
        ):
            raise ValueError(
                "row_metadata stage1_role provenance must be present on every row"
            )
        rows: list[Mapping[str, Any]] = []
        for index, row in enumerate(raw_metadata):
            if not isinstance(row, Mapping):
                raise TypeError(f"row_metadata[{index}] must be a mapping")
            has_split = "measurement_split" in row
            has_role = "stage1_role" in row
            measurement_split = str(row.get("measurement_split", "")).strip()
            stage1_role = str(row.get("stage1_role", "")).strip()
            if has_split and measurement_split != "measurement_train":
                raise ValueError(
                    "probe training accepts only measurement_train rows; "
                    f"row_metadata[{index}] has {measurement_split!r}"
                )
            if has_role and stage1_role != "probe_candidate":
                raise ValueError(
                    "probe training accepts only probe_candidate rows; "
                    f"row_metadata[{index}] has {stage1_role!r}"
                )
            rows.append(dict(row))
        normalized_row_metadata = tuple(rows)

    raw_payload_metadata = payload.get("metadata", {})
    if raw_payload_metadata is None:
        raw_payload_metadata = {}
    if not isinstance(raw_payload_metadata, Mapping):
        raise TypeError("payload metadata must be a mapping when provided")
    return ValidatedProbeData(
        hidden_states=states,
        harmfulness_labels=harmfulness,
        refusal_labels=refusal,
        pair_ids=tuple(pair_ids),
        hidden_sizes=hidden_sizes,
        row_metadata=normalized_row_metadata,
        payload_metadata=dict(raw_payload_metadata),
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


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average ranks for ties, using zero-based ranks."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _spearman(values_a: Sequence[float], values_b: Sequence[float]) -> Optional[float]:
    if len(values_a) != len(values_b):
        raise ValueError("Spearman inputs must have equal length")
    if len(values_a) < 2:
        return None
    ranks_a = _rankdata(values_a)
    ranks_b = _rankdata(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    centered_a = [value - mean_a for value in ranks_a]
    centered_b = [value - mean_b for value in ranks_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return None
    return sum(
        first * second for first, second in zip(centered_a, centered_b)
    ) / denominator


def _binary_metric_values(
    labels: Sequence[int], probabilities: Sequence[float]
) -> Mapping[str, Any]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("binary metrics require non-empty, equal-length inputs")
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    positive_count = tp + fn
    negative_count = tn + fp
    accuracy = (tp + tn) / len(labels)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    specificity = tn / negative_count if negative_count else None
    sensitivity = tp / positive_count if positive_count else None
    balanced_accuracy = (
        None
        if sensitivity is None or specificity is None
        else (sensitivity + specificity) / 2.0
    )
    positives = [
        probability for label, probability in zip(labels, probabilities) if label == 1
    ]
    negatives = [
        probability for label, probability in zip(labels, probabilities) if label == 0
    ]
    auroc: Optional[float]
    if not positives or not negatives:
        auroc = None
    else:
        comparisons = 0.0
        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    comparisons += 1.0
                elif positive == negative:
                    comparisons += 0.5
        auroc = comparisons / (len(positives) * len(negatives))
    epsilon = 1.0e-7
    loss = -sum(
        label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1 - label)
        * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
        for label, probability in zip(labels, probabilities)
    ) / len(labels)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "auroc": auroc,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sample_count": len(labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "predicted_positive_rate": sum(predictions) / len(predictions),
        "label_positive_rate": positive_count / len(labels),
        "threshold": 0.5,
    }


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _resample_pair_indices(
    pair_ids: Sequence[str], *, rng: random.Random
) -> list[int]:
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, pair_id in enumerate(pair_ids):
        groups.setdefault(pair_id, []).append(index)
    names = list(groups)
    sampled: list[int] = []
    for _ in names:
        sampled.extend(groups[rng.choice(names)])
    return sampled


def _pair_cluster_metric_intervals(
    labels: Sequence[int],
    probabilities: Sequence[float],
    pair_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    metric_names = ("auroc", "f1", "balanced_accuracy", "accuracy")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices = _resample_pair_indices(pair_ids, rng=rng)
        values = _binary_metric_values(
            [labels[index] for index in indices],
            [probabilities[index] for index in indices],
        )
        for name in metric_names:
            value = values[name]
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    return {
        name: {
            "low": _percentile(values, 0.025),
            "high": _percentile(values, 0.975),
            "confidence_level": 0.95,
            "method": "pair_cluster_percentile",
            "requested_replicates": replicates,
            "valid_replicates": len(values),
            "seed": seed,
        }
        for name, values in samples.items()
    }


def _pair_cluster_spearman_interval(
    probabilities: Sequence[float],
    projections: Sequence[float],
    pair_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    samples: list[float] = []
    rng = random.Random(seed)
    for _ in range(replicates):
        indices = _resample_pair_indices(pair_ids, rng=rng)
        correlation = _spearman(
            [probabilities[index] for index in indices],
            [projections[index] for index in indices],
        )
        if correlation is not None and math.isfinite(correlation):
            samples.append(correlation)
    return {
        "low": _percentile(samples, 0.025),
        "high": _percentile(samples, 0.975),
        "confidence_level": 0.95,
        "method": "pair_cluster_percentile",
        "requested_replicates": replicates,
        "valid_replicates": len(samples),
        "seed": seed,
    }


def _score_probabilities(
    *, scorer: Any, direction: str, states: Mapping[Any, Any]
) -> Mapping[Any, Any]:
    import torch

    with torch.no_grad():
        if direction == "harmfulness":
            logits = scorer.score_harmfulness(states, return_logits=True)
        elif direction == "refusal":
            logits = scorer.score_refusal(states, return_logits=True)
        else:
            raise ValueError("direction must be harmfulness or refusal")
        return OrderedDict(
            (layer, torch.sigmoid(value).detach().cpu())
            for layer, value in logits.items()
        )


def _class_mean_geometry(
    states: Mapping[Any, Any], labels: Any
) -> tuple[Mapping[Any, Any], Mapping[Any, Mapping[str, Any]], Mapping[Any, Any]]:
    directions: "OrderedDict[Any, Any]" = OrderedDict()
    class_means: "OrderedDict[Any, Mapping[str, Any]]" = OrderedDict()
    centers: "OrderedDict[Any, Any]" = OrderedDict()
    negative_mask = labels == 0
    positive_mask = labels == 1
    if not bool(negative_mask.any()) or not bool(positive_mask.any()):
        raise ValueError("class-mean directions require both binary classes")
    for layer, state in states.items():
        negative = state[negative_mask].mean(dim=0).detach().cpu()
        positive = state[positive_mask].mean(dim=0).detach().cpu()
        direction = (positive - negative).contiguous()
        center = ((positive + negative) / 2.0).contiguous()
        directions[layer] = direction
        class_means[layer] = {
            "negative": negative.contiguous(),
            "positive": positive.contiguous(),
        }
        centers[layer] = center
    return directions, class_means, centers


def _direction_projections(
    states: Mapping[Any, Any],
    directions: Mapping[Any, Any],
    centers: Mapping[Any, Any],
) -> Mapping[Any, Any]:
    import torch

    projections: "OrderedDict[Any, Any]" = OrderedDict()
    for layer, state in states.items():
        vector = directions[layer]
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) == 0.0:
            projection = torch.zeros(state.shape[0], dtype=torch.float32)
        else:
            projection = ((state - centers[layer]) @ vector) / norm
        projections[layer] = projection.detach().cpu().to(dtype=torch.float32)
    return projections


def _metrics_for_probabilities(
    probabilities: Mapping[Any, Any],
    labels: Any,
    pair_ids: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    label_values = [int(value) for value in labels.tolist()]
    metrics: "OrderedDict[str, Mapping[str, Any]]" = OrderedDict()
    for layer_index, (layer, values) in enumerate(probabilities.items()):
        probability_values = [float(value) for value in values.tolist()]
        layer_metrics = dict(_binary_metric_values(label_values, probability_values))
        layer_metrics["bootstrap_95_ci"] = _pair_cluster_metric_intervals(
            label_values,
            probability_values,
            pair_ids,
            replicates=bootstrap_replicates,
            seed=seed + layer_index * 1009,
        )
        metrics[str(layer)] = layer_metrics
    return metrics


def _alignment_for_predictions(
    probabilities: Mapping[Any, Any],
    projections: Mapping[Any, Any],
    pair_ids: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> Mapping[str, Mapping[str, Any]]:
    result: "OrderedDict[str, Mapping[str, Any]]" = OrderedDict()
    for layer_index, (layer, values) in enumerate(probabilities.items()):
        probability_values = [float(value) for value in values.tolist()]
        projection_values = [float(value) for value in projections[layer].tolist()]
        interval_seed = seed + layer_index * 1009
        result[str(layer)] = {
            "spearman": _spearman(probability_values, projection_values),
            "sample_count": len(probability_values),
            "pair_count": len(set(pair_ids)),
            "bootstrap_95_ci": _pair_cluster_spearman_interval(
                probability_values,
                projection_values,
                pair_ids,
                replicates=bootstrap_replicates,
                seed=interval_seed,
            ),
        }
    return result


def _fit_probe_scorer(
    data: ValidatedProbeData,
    indices: Any,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[Any, Mapping[str, Mapping[str, Any]]]:
    import torch
    from torch import nn

    from core.safety_state import DualSafetyStateScorer

    harmfulness = data.harmfulness_labels.index_select(0, indices)
    refusal = data.refusal_labels.index_select(0, indices)
    for name, labels in (("harmfulness", harmfulness), ("refusal", refusal)):
        if labels.unique().numel() != 2:
            raise ValueError(
                f"probe training partition lacks both {name} classes; "
                "choose another pair split or add groups"
            )
    random.seed(seed)
    torch.manual_seed(seed)
    scorer = DualSafetyStateScorer(
        hidden_size=data.hidden_sizes, trainable=True
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
    states = _selected_states(data, indices)
    histories: dict[str, dict[str, Any]] = {
        "harmfulness": {"objective": [], "by_layer": {}},
        "refusal": {"objective": [], "by_layer": {}},
    }
    for _ in range(epochs):
        for direction, labels, optimizer in (
            ("harmfulness", harmfulness, harmfulness_optimizer),
            ("refusal", refusal, refusal_optimizer),
        ):
            objective, layer_losses = _direction_epoch(
                scorer=scorer,
                direction=direction,
                states=states,
                labels=labels,
                optimizer=optimizer,
                loss_function=loss_function,
            )
            histories[direction]["objective"].append(objective)
            for layer, value in layer_losses.items():
                histories[direction]["by_layer"].setdefault(layer, []).append(value)
    scorer.eval()
    return scorer, histories


def _safe_metadata(value: Any) -> Any:
    """Convert payload provenance to weights-only-loadable primitive values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_metadata(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_safe_metadata(item) for item in value]
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_provenance(
    data: ValidatedProbeData, *, source_path: Optional[str | Path]
) -> Mapping[str, Any]:
    source = None if source_path is None else Path(source_path).expanduser().resolve()
    payload_metadata = _safe_metadata(data.payload_metadata)
    model_provenance = data.payload_metadata.get("model_provenance", {})
    if not isinstance(model_provenance, Mapping):
        model_provenance = {}

    def lookup(name: str) -> Any:
        value = model_provenance.get(name)
        if value is None:
            value = data.payload_metadata.get(name)
        return _safe_metadata(value)

    return {
        "source_payload_sha256": (
            _file_sha256(source) if source is not None and source.is_file() else None
        ),
        "source_model": lookup("source_model") or lookup("model_name"),
        "model_id": lookup("model_id"),
        "model_fingerprint": lookup("model_fingerprint"),
        "dtype": lookup("dtype"),
        "token_span": lookup("token_span"),
        "pooling": lookup("pooling"),
        "layers": _safe_metadata(
            data.payload_metadata.get("layers", list(data.hidden_states))
        ),
        "training_payload_metadata": payload_metadata,
    }


def train_safety_probes(
    payload: Mapping[str, Any],
    *,
    validation_fraction: float = 0.2,
    cv_folds: Optional[int] = None,
    seed: int = 42,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
    bootstrap_replicates: int = 200,
    source_path: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """Train H/R probes, optionally evaluating them by pair-grouped OOF CV.

    ``cv_folds=None`` (or ``1``) retains the original single grouped hold-out
    API. The production CLI passes five folds by default. In both modes the
    serialized probes and class-mean directions are finally fit on every
    eligible measurement-train row; held-out measurement-val rows are rejected
    by :func:`validate_training_payload` when provenance is available.
    """

    import torch

    _validate_hyperparameters(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates <= 0
    ):
        raise ValueError("bootstrap_replicates must be a positive integer")
    if cv_folds is not None and (
        isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 1
    ):
        raise ValueError("cv_folds must be None or a positive integer")

    data = validate_training_payload(payload)
    all_indices = torch.arange(data.num_samples, dtype=torch.long)
    directions_by_target: dict[str, Mapping[Any, Any]] = {}
    class_means_by_target: dict[str, Mapping[Any, Mapping[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {
        "harmfulness": {},
        "refusal": {},
    }
    split_metadata: Mapping[str, Any]
    oof_payload: Optional[Mapping[str, Any]] = None

    if cv_folds is not None and cv_folds >= 2:
        folds = split_pair_group_folds(data.pair_ids, folds=cv_folds, seed=seed)
        fold_assignments = torch.full(
            (data.num_samples,), -1, dtype=torch.int64
        )
        oof_probabilities: dict[str, OrderedDict[Any, Any]] = {
            direction: OrderedDict(
                (layer, torch.empty(data.num_samples, dtype=torch.float32))
                for layer in data.hidden_states
            )
            for direction in ("harmfulness", "refusal")
        }
        oof_projections: dict[str, OrderedDict[Any, Any]] = {
            direction: OrderedDict(
                (layer, torch.empty(data.num_samples, dtype=torch.float32))
                for layer in data.hidden_states
            )
            for direction in ("harmfulness", "refusal")
        }
        fold_records: list[Mapping[str, Any]] = []
        for fold in folds:
            train_indices = torch.tensor(fold.train_indices, dtype=torch.long)
            validation_indices = torch.tensor(
                fold.validation_indices, dtype=torch.long
            )
            fold_assignments.index_fill_(0, validation_indices, fold.fold)
            fold_scorer, _ = _fit_probe_scorer(
                data,
                train_indices,
                seed=seed + fold.fold,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            train_states = _selected_states(data, train_indices)
            validation_states = _selected_states(data, validation_indices)
            fold_record: dict[str, Any] = {
                "fold": fold.fold,
                "train_indices": list(fold.train_indices),
                "validation_indices": list(fold.validation_indices),
                "train_pair_ids": list(fold.train_pair_ids),
                "validation_pair_ids": list(fold.validation_pair_ids),
                "class_counts": {},
            }
            for direction, labels in (
                ("harmfulness", data.harmfulness_labels),
                ("refusal", data.refusal_labels),
            ):
                train_labels = labels.index_select(0, train_indices)
                validation_probabilities = _score_probabilities(
                    scorer=fold_scorer,
                    direction=direction,
                    states=validation_states,
                )
                fold_directions, _, fold_centers = _class_mean_geometry(
                    train_states, train_labels
                )
                validation_projections = _direction_projections(
                    validation_states, fold_directions, fold_centers
                )
                for layer in data.hidden_states:
                    oof_probabilities[direction][layer].index_copy_(
                        0, validation_indices, validation_probabilities[layer]
                    )
                    oof_projections[direction][layer].index_copy_(
                        0, validation_indices, validation_projections[layer]
                    )
                fold_record["class_counts"][direction] = {
                    "negative": int((train_labels == 0).sum().item()),
                    "positive": int((train_labels == 1).sum().item()),
                }
            fold_records.append(fold_record)
        if bool((fold_assignments < 0).any()):
            raise AssertionError("OOF predictions are incomplete")

        for direction_index, (direction, labels) in enumerate(
            (
                ("harmfulness", data.harmfulness_labels),
                ("refusal", data.refusal_labels),
            )
        ):
            metric_seed = seed + direction_index * 100_003
            oof_metrics = _metrics_for_probabilities(
                oof_probabilities[direction],
                labels,
                data.pair_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=metric_seed,
            )
            alignment = _alignment_for_predictions(
                oof_probabilities[direction],
                oof_projections[direction],
                data.pair_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=metric_seed + 50_021,
            )
            for layer, layer_metrics in oof_metrics.items():
                layer_metrics["probe_direction_alignment"] = alignment[layer]
            metrics[direction]["oof"] = oof_metrics
            metrics[direction]["direction_alignment"] = alignment
        split_metadata = {
            "mode": "pair_group_kfold_oof",
            "fold_count": cv_folds,
            "folds": fold_records,
        }
        oof_payload = {
            "pair_ids": list(data.pair_ids),
            "fold_assignments": fold_assignments,
            "probabilities": oof_probabilities,
            "direction_projections": oof_projections,
        }
    else:
        split = split_pair_groups(
            data.pair_ids,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        train_indices = torch.tensor(split.train_indices, dtype=torch.long)
        validation_indices = torch.tensor(split.validation_indices, dtype=torch.long)
        evaluation_scorer, _ = _fit_probe_scorer(
            data,
            train_indices,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        train_states = _selected_states(data, train_indices)
        validation_states = _selected_states(data, validation_indices)
        train_pair_ids = tuple(data.pair_ids[index] for index in split.train_indices)
        validation_pair_ids = tuple(
            data.pair_ids[index] for index in split.validation_indices
        )
        for direction_index, (direction, labels) in enumerate(
            (
                ("harmfulness", data.harmfulness_labels),
                ("refusal", data.refusal_labels),
            )
        ):
            train_labels = labels.index_select(0, train_indices)
            validation_labels = labels.index_select(0, validation_indices)
            train_probabilities = _score_probabilities(
                scorer=evaluation_scorer, direction=direction, states=train_states
            )
            validation_probabilities = _score_probabilities(
                scorer=evaluation_scorer,
                direction=direction,
                states=validation_states,
            )
            split_directions, _, split_centers = _class_mean_geometry(
                train_states, train_labels
            )
            train_projections = _direction_projections(
                train_states, split_directions, split_centers
            )
            validation_projections = _direction_projections(
                validation_states, split_directions, split_centers
            )
            metric_seed = seed + direction_index * 100_003
            metrics[direction]["train"] = _metrics_for_probabilities(
                train_probabilities,
                train_labels,
                train_pair_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=metric_seed,
            )
            metrics[direction]["validation"] = _metrics_for_probabilities(
                validation_probabilities,
                validation_labels,
                validation_pair_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=metric_seed + 10_007,
            )
            metrics[direction]["direction_alignment"] = {
                "train": _alignment_for_predictions(
                    train_probabilities,
                    train_projections,
                    train_pair_ids,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=metric_seed + 20_011,
                ),
                "validation": _alignment_for_predictions(
                    validation_probabilities,
                    validation_projections,
                    validation_pair_ids,
                    bootstrap_replicates=bootstrap_replicates,
                    seed=metric_seed + 30_013,
                ),
            }
        split_metadata = {
            "mode": "single_pair_group_holdout",
            "train_indices": list(split.train_indices),
            "validation_indices": list(split.validation_indices),
            "train_pair_ids": list(split.train_pair_ids),
            "validation_pair_ids": list(split.validation_pair_ids),
        }

    # Evaluation never trains on held-out folds. Once it is complete, fit the
    # deployable probes and the serialized direction geometry on all 80
    # measurement-train rows.
    final_scorer, histories = _fit_probe_scorer(
        data,
        all_indices,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    all_states = _selected_states(data, all_indices)
    for direction_index, (direction, labels) in enumerate(
        (
            ("harmfulness", data.harmfulness_labels),
            ("refusal", data.refusal_labels),
        )
    ):
        directions, class_means, _ = _class_mean_geometry(all_states, labels)
        directions_by_target[direction] = directions
        class_means_by_target[direction] = class_means
        final_probabilities = _score_probabilities(
            scorer=final_scorer, direction=direction, states=all_states
        )
        metrics[direction]["final_train"] = _metrics_for_probabilities(
            final_probabilities,
            labels,
            data.pair_ids,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + direction_index * 100_003 + 40_009,
        )

    state_dict = OrderedDict(
        (name, value.detach().cpu())
        for name, value in final_scorer.state_dict().items()
    )
    provenance = _probe_provenance(data, source_path=source_path)
    fingerprint_input = {
        "source_payload_sha256": provenance["source_payload_sha256"],
        "hidden_sizes": {str(key): value for key, value in data.hidden_sizes.items()},
        "pair_ids": list(data.pair_ids),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "evaluation_mode": split_metadata["mode"],
    }
    training_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    provenance_verified = bool(data.row_metadata) and all(
        row.get("measurement_split") == "measurement_train"
        and row.get("stage1_role") == "probe_candidate"
        for row in data.row_metadata or ()
    )
    checkpoint: dict[str, Any] = {
        "format": "dual-safety-state-layerwise-linear-probes",
        "version": 2,
        # This exact key remains compatible with batch_safety_attack.
        "hidden_sizes": dict(data.hidden_sizes),
        "state_dict": state_dict,
        "directions": directions_by_target,
        "class_means": class_means_by_target,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source_path": (
                None
                if source_path is None
                else str(Path(source_path).expanduser().resolve())
            ),
            "num_samples": data.num_samples,
            "num_pairs": len(set(data.pair_ids)),
            "num_layers": len(data.hidden_sizes),
            "measurement_split": (
                "measurement_train" if provenance_verified else None
            ),
            "stage1_role": "probe_candidate" if provenance_verified else None,
            "stage1_provenance_verified": provenance_verified,
            "seed": seed,
            "validation_fraction": float(validation_fraction),
            "bootstrap": {
                "unit": "pair_id",
                "replicates": bootstrap_replicates,
                "confidence_level": 0.95,
                "seed": seed,
            },
            "split": split_metadata,
            "final_fit": "all_eligible_rows",
            "direction_projection": {
                "direction": "positive_class_mean-minus-negative_class_mean",
                "center": "midpoint_of_class_means",
                "formula": "dot(x-center,d)/l2_norm(d)",
                "zero_norm_value": 0.0,
            },
            "provenance": provenance,
            "training_fingerprint": training_fingerprint,
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
    if oof_payload is not None:
        checkpoint["oof_predictions"] = oof_payload
    return checkpoint


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
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Pair-grouped OOF folds (production default: 5); use 1 for the "
            "legacy single grouped hold-out evaluation"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Fixed-seed pair-cluster bootstrap replicates",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = train_probe_checkpoint(
        args.input,
        args.output,
        validation_fraction=args.validation_fraction,
        cv_folds=args.cv_folds,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bootstrap_replicates=args.bootstrap_replicates,
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
    "GroupFold",
    "GroupSplit",
    "ValidatedProbeData",
    "build_parser",
    "load_training_payload",
    "main",
    "save_probe_checkpoint",
    "split_pair_groups",
    "split_pair_group_folds",
    "train_probe_checkpoint",
    "train_safety_probes",
    "validate_training_payload",
]
