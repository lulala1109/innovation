#!/usr/bin/env python3
"""Cross-sample Stage-1 RQ1 analysis for Layer x PGD-step H/R scores.

This module is deliberately separate from ``analyze_safety_dynamics.py``.
The latter studies Dynamic Safety Bottlenecks for later research questions;
this module answers RQ1-a/b/c using held-out Standard-PGD trajectories.

The input is the tensor artifact produced by
``experiments.score_stage1_trajectories``.  All uncertainty calculations
resample complete ``pair_id`` clusters, never individual layer/step rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import optimize, stats


SCORE_FORMAT = "stage1-trajectory-scores"
SCORE_VERSION = 1
ANALYSIS_FORMAT = "stage1-rq1-analysis"
ANALYSIS_VERSION = 1
PRIMARY_METRICS = ("H_probe", "R_probe")
EXPECTED_SCORE_KEYS = (
    "H_probe",
    "R_probe",
    "H_direction",
    "R_direction",
)
PHASE_BOUNDS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


@dataclass(frozen=True)
class ValidatedScoreData:
    case_ids: tuple[str, ...]
    pair_ids: tuple[str, ...]
    layers: tuple[Any, ...]
    steps: np.ndarray
    progress: np.ndarray
    scores: Mapping[str, np.ndarray]
    attack_loss: np.ndarray
    behavior: tuple[tuple[Mapping[str, Any], ...], ...]
    metadata: Mapping[str, Any]

    @property
    def shape(self) -> tuple[int, int, int]:
        first = next(iter(self.scores.values()))
        return tuple(int(value) for value in first.shape)


def _safe_torch_load(path: str | Path) -> Mapping[str, Any]:
    import torch

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 score artifact not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old torch only
        raise RuntimeError(
            "Safe RQ1 analysis requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 score artifact must contain one mapping")
    return payload


def load_score_payload(path: str | Path) -> Mapping[str, Any]:
    """Safely load one scorer artifact on CPU."""

    return _safe_torch_load(path)


def _sequence_of_text(value: Any, *, name: str, count: Optional[int] = None) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must contain non-blank strings")
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must contain {count} values")
    return result


def _numpy_tensor(value: Any, *, name: str, ndim: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(result)


def _normalize_behavior(
    value: Any, *, cases: int, steps: int
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    if value is None:
        return tuple(tuple({"label_status": "missing"} for _ in range(steps)) for _ in range(cases))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("behavior must be a case-by-step sequence")
    if len(value) != cases:
        raise ValueError(f"behavior must contain {cases} case rows")
    normalized = []
    for case_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"behavior[{case_index}] must be a sequence")
        if len(row) != steps:
            raise ValueError(f"behavior[{case_index}] must contain {steps} steps")
        case_values = []
        for step_index, item in enumerate(row):
            if item is None:
                item = {"label_status": "missing"}
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"behavior[{case_index}][{step_index}] must be an object"
                )
            case_values.append(dict(item))
        normalized.append(tuple(case_values))
    return tuple(normalized)


def validate_score_payload(payload: Mapping[str, Any]) -> ValidatedScoreData:
    """Validate axes, complete 0..T grids, score tensors, and behavior rows."""

    if not isinstance(payload, Mapping):
        raise TypeError("score payload must be a mapping")
    if payload.get("format") != SCORE_FORMAT or payload.get("version") != SCORE_VERSION:
        raise ValueError(
            f"score payload must use format={SCORE_FORMAT!r}, version={SCORE_VERSION}"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    if metadata.get("probe_version") != 2 or metadata.get("directions_missing") is True:
        raise ValueError(
            "RQ1 analysis requires a provenance-verified v2 probe with independent "
            "class-mean directions; legacy compatibility scores cannot be analyzed"
        )
    if (
        metadata.get("unverified_provenance_allowed") is True
        or metadata.get("probe_model_fingerprint_verified") is not True
        or metadata.get("probe_source_payload_sha256_verified") is not True
    ):
        raise ValueError(
            "RQ1 analysis requires verified probe source and model provenance; "
            "scores produced with --allow-unverified-provenance are exploratory only"
        )
    if (
        metadata.get("trajectory_measurement_split") != "measurement_val"
        or metadata.get("trajectory_stage1_role") != "trajectory_candidate"
    ):
        raise ValueError(
            "RQ1 analysis accepts only measurement_val/trajectory_candidate scores"
        )
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping):
        raise ValueError("score payload requires a scores mapping")
    missing = [key for key in EXPECTED_SCORE_KEYS if key not in raw_scores]
    if missing:
        raise ValueError("score payload is missing scores: " + ", ".join(missing))

    scores: "OrderedDict[str, np.ndarray]" = OrderedDict()
    shape: Optional[tuple[int, int, int]] = None
    for key in EXPECTED_SCORE_KEYS:
        tensor = _numpy_tensor(raw_scores[key], name=f"scores[{key}]", ndim=3)
        if shape is None:
            shape = tuple(int(value) for value in tensor.shape)
        elif tensor.shape != shape:
            raise ValueError("all score tensors must have identical [N,L,S] shape")
        scores[key] = tensor
    assert shape is not None
    case_count, layer_count, step_count = shape
    if min(shape) < 1:
        raise ValueError("score tensors must have non-empty [N,L,S] dimensions")

    case_ids = _sequence_of_text(payload.get("case_ids"), name="case_ids", count=case_count)
    pair_ids = _sequence_of_text(payload.get("pair_ids"), name="pair_ids", count=case_count)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be unique")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("held-out RQ1 payload requires one case per pair_id")

    raw_layers = payload.get("layers")
    if isinstance(raw_layers, (str, bytes)) or not isinstance(raw_layers, Sequence):
        raise ValueError("layers must be a sequence")
    layers = tuple(raw_layers)
    if len(layers) != layer_count or len(set(map(str, layers))) != layer_count:
        raise ValueError(f"layers must contain {layer_count} unique values")

    raw_steps = payload.get("steps")
    if hasattr(raw_steps, "detach"):
        raw_steps = raw_steps.detach().cpu().numpy()
    steps = np.asarray(raw_steps)
    if steps.ndim != 1 or len(steps) != step_count:
        raise ValueError(f"steps must have shape [{step_count}]")
    if not np.issubdtype(steps.dtype, np.integer):
        if not np.all(np.equal(steps, np.floor(steps))):
            raise ValueError("steps must contain integers")
    steps = steps.astype(np.int64, copy=False)
    expected_steps = np.arange(step_count, dtype=np.int64)
    if not np.array_equal(steps, expected_steps):
        raise ValueError(
            f"RQ1 requires the complete 0..T grid; got {steps.tolist()}"
        )
    progress = steps.astype(np.float64) / max(1, int(steps[-1]))

    attack_loss = _numpy_tensor(
        payload.get("attack_loss"), name="attack_loss", ndim=2
    )
    if attack_loss.shape != (case_count, step_count):
        raise ValueError(
            f"attack_loss must have shape [{case_count},{step_count}]"
        )
    behavior = _normalize_behavior(
        payload.get("behavior"), cases=case_count, steps=step_count
    )
    return ValidatedScoreData(
        case_ids=case_ids,
        pair_ids=pair_ids,
        layers=layers,
        steps=steps,
        progress=progress,
        scores=scores,
        attack_loss=attack_loss,
        behavior=behavior,
        metadata=dict(metadata),
    )


def phase_for_progress(progress: float) -> str:
    """Assign the documented explicit early/middle/late progress phase."""

    value = float(progress)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("progress must be finite and within [0,1]")
    if value < PHASE_BOUNDS[1]:
        return "early"
    if value < PHASE_BOUNDS[2]:
        return "middle"
    return "late"


def _cluster_map(pair_ids: Sequence[str]) -> "OrderedDict[str, np.ndarray]":
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, pair_id in enumerate(pair_ids):
        groups.setdefault(pair_id, []).append(index)
    return OrderedDict(
        (key, np.asarray(indices, dtype=np.int64)) for key, indices in groups.items()
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    pair_ids: Sequence[str],
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(pair_ids):
        raise ValueError("bootstrap values and pair_ids must be aligned vectors")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    groups = _cluster_map(pair_ids)
    keys = tuple(groups)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(keys), size=len(keys))
        selected = np.concatenate([groups[keys[item]] for item in sampled])
        estimates[index] = float(np.mean(values[selected]))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _summary_row(
    values: np.ndarray,
    pair_ids: Sequence[str],
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    count = int(vector.size)
    variance = float(np.var(vector, ddof=1)) if count > 1 else 0.0
    low, high = _bootstrap_mean_ci(
        vector,
        pair_ids,
        replicates=replicates,
        confidence=confidence,
        rng=rng,
    )
    return {
        "mean": float(np.mean(vector)),
        "variance": variance,
        "standard_error": math.sqrt(variance / count) if count else None,
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "sample_count": count,
        "pair_count": len(set(pair_ids)),
    }


def _cell_statistics(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, values in data.scores.items():
        baseline = values[:, :, [0]]
        deltas = values - baseline
        for layer_index, layer in enumerate(data.layers):
            for step_index, step in enumerate(data.steps):
                row = {
                    "metric": metric,
                    "layer": layer,
                    "step": int(step),
                    "progress": float(data.progress[step_index]),
                    "phase": phase_for_progress(data.progress[step_index]),
                }
                row.update(
                    _summary_row(
                        values[:, layer_index, step_index],
                        data.pair_ids,
                        replicates=replicates,
                        confidence=confidence,
                        rng=rng,
                    )
                )
                delta = _summary_row(
                    deltas[:, layer_index, step_index],
                    data.pair_ids,
                    replicates=replicates,
                    confidence=confidence,
                    rng=rng,
                )
                row.update({f"delta_{key}": value for key, value in delta.items()})
                rows.append(row)
    return rows


def _phase_profiles(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    phases = tuple(phase_for_progress(value) for value in data.progress)
    rows: list[dict[str, Any]] = []
    for metric, values in data.scores.items():
        deltas = values - values[:, :, [0]]
        for phase in ("early", "middle", "late"):
            step_indices = np.asarray(
                [index for index, value in enumerate(phases) if value == phase],
                dtype=np.int64,
            )
            for layer_index, layer in enumerate(data.layers):
                per_case = values[:, layer_index, :][:, step_indices].mean(axis=1)
                per_case_delta = deltas[:, layer_index, :][:, step_indices].mean(axis=1)
                row = {"metric": metric, "layer": layer, "phase": phase}
                row.update(
                    _summary_row(
                        per_case,
                        data.pair_ids,
                        replicates=replicates,
                        confidence=confidence,
                        rng=rng,
                    )
                )
                delta = _summary_row(
                    per_case_delta,
                    data.pair_ids,
                    replicates=replicates,
                    confidence=confidence,
                    rng=rng,
                )
                row.update({f"delta_{key}": value for key, value in delta.items()})
                rows.append(row)
    return rows


def _attack_loss_statistics(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for step_index, step in enumerate(data.steps):
        row = {
            "step": int(step),
            "progress": float(data.progress[step_index]),
            "phase": phase_for_progress(data.progress[step_index]),
        }
        row.update(
            _summary_row(
                data.attack_loss[:, step_index],
                data.pair_ids,
                replicates=replicates,
                confidence=confidence,
                rng=rng,
            )
        )
        rows.append(row)
    return rows


def _loss_state_correlations(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for metric, values in data.scores.items():
        for layer_index, layer in enumerate(data.layers):
            coefficients = []
            pairs = []
            for case_index, pair_id in enumerate(data.pair_ids):
                loss = data.attack_loss[case_index]
                score = values[case_index, layer_index]
                if math.isclose(float(np.std(loss)), 0.0, abs_tol=1e-14) or math.isclose(
                    float(np.std(score)), 0.0, abs_tol=1e-14
                ):
                    continue
                coefficient = float(stats.spearmanr(loss, score).statistic)
                if math.isfinite(coefficient):
                    coefficients.append(coefficient)
                    pairs.append(pair_id)
            row: dict[str, Any] = {
                "metric": metric,
                "layer": layer,
                "correlation": "spearman(attack_loss, state_score)",
                "correlation_count": len(coefficients),
            }
            if coefficients:
                row.update(
                    _summary_row(
                        np.asarray(coefficients, dtype=np.float64),
                        pairs,
                        replicates=replicates,
                        confidence=confidence,
                        rng=rng,
                    )
                )
            rows.append(row)
    return rows


def _icc_two_way_average(matrix: np.ndarray) -> Optional[float]:
    """Return ICC(2,k) for layer profiles rated by held-out cases."""

    values = np.asarray(matrix, dtype=np.float64)
    # Rows are layers (targets), columns are cases (raters).
    if values.ndim != 2 or min(values.shape) < 2:
        return None
    targets, raters = values.shape
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = raters * float(np.sum((row_means - grand) ** 2))
    ss_columns = targets * float(np.sum((column_means - grand) ** 2))
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ss_error = float(np.sum(residual**2))
    ms_rows = ss_rows / (targets - 1)
    ms_columns = ss_columns / (raters - 1)
    ms_error = ss_error / ((targets - 1) * (raters - 1))
    denominator = ms_rows + (ms_columns - ms_error) / targets
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return float((ms_rows - ms_error) / denominator)


def _profile_reproducibility(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    def loo_correlations(profiles: np.ndarray) -> np.ndarray:
        coefficients = []
        for case_index in range(len(profiles)):
            others = np.delete(profiles, case_index, axis=0)
            if not len(others):
                continue
            profile = profiles[case_index]
            reference = others.mean(axis=0)
            if (
                profile.size < 2
                or math.isclose(float(np.std(profile)), 0.0, abs_tol=1e-14)
                or math.isclose(float(np.std(reference)), 0.0, abs_tol=1e-14)
            ):
                continue
            coefficient = float(stats.spearmanr(profile, reference).statistic)
            if math.isfinite(coefficient):
                coefficients.append(coefficient)
        return np.asarray(coefficients, dtype=np.float64)

    def bootstrap_summary(profiles: np.ndarray) -> Mapping[str, Any]:
        coefficients = loo_correlations(profiles)
        if not len(coefficients):
            return {}
        estimates = []
        case_count = len(profiles)
        for _ in range(replicates):
            # Resample complete held-out pair profiles, then recompute every LOO
            # correlation inside the bootstrap sample.  Resampling precomputed
            # LOO coefficients would ignore their shared-reference dependence.
            sampled = rng.integers(0, case_count, size=case_count)
            replicate_coefficients = loo_correlations(profiles[sampled])
            if len(replicate_coefficients):
                estimates.append(float(np.mean(replicate_coefficients)))
        if not estimates:
            return {
                "mean": float(np.mean(coefficients)),
                "variance": (
                    float(np.var(coefficients, ddof=1))
                    if len(coefficients) > 1
                    else 0.0
                ),
                "standard_error": None,
                "ci_low": None,
                "ci_high": None,
                "confidence": float(confidence),
                "sample_count": int(len(coefficients)),
                "pair_count": case_count,
                "bootstrap_valid_replicates": 0,
                "bootstrap_method": "pair-profile-resample-recompute-loo",
            }
        estimate_array = np.asarray(estimates, dtype=np.float64)
        alpha = (1.0 - confidence) / 2.0
        return {
            "mean": float(np.mean(coefficients)),
            "variance": (
                float(np.var(coefficients, ddof=1))
                if len(coefficients) > 1
                else 0.0
            ),
            "standard_error": (
                float(np.std(estimate_array, ddof=1))
                if len(estimate_array) > 1
                else 0.0
            ),
            "ci_low": float(np.quantile(estimate_array, alpha)),
            "ci_high": float(np.quantile(estimate_array, 1.0 - alpha)),
            "confidence": float(confidence),
            "sample_count": int(len(coefficients)),
            "pair_count": case_count,
            "bootstrap_valid_replicates": int(len(estimate_array)),
            "bootstrap_method": "pair-profile-resample-recompute-loo",
        }

    rows: list[dict[str, Any]] = []
    for metric, values in data.scores.items():
        for step_index, step in enumerate(data.steps):
            profiles = values[:, :, step_index]
            summary = bootstrap_summary(profiles)
            row: dict[str, Any] = {
                "metric": metric,
                "step": int(step),
                "progress": float(data.progress[step_index]),
                "phase": phase_for_progress(data.progress[step_index]),
                "icc_2k": _icc_two_way_average(profiles.T),
                "correlation_count": int(summary.get("sample_count", 0)),
            }
            if summary:
                row.update(
                    {
                        f"loo_spearman_{key}": value
                        for key, value in summary.items()
                    }
                )
            rows.append(row)
    return rows


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("FDR p-values must be a finite vector")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(values[original_index]) * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def _layer_slopes(
    data: ValidatedScoreData,
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    def one_sample_p_value(values: np.ndarray) -> float:
        if len(values) < 2:
            return 1.0
        if math.isclose(float(np.std(values, ddof=1)), 0.0, abs_tol=1e-14):
            return 1.0 if math.isclose(float(np.mean(values)), 0.0, abs_tol=1e-14) else 0.0
        return float(stats.ttest_1samp(values, 0.0).pvalue)

    rows: list[dict[str, Any]] = []
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for metric, values in data.scores.items():
        for layer_index, layer in enumerate(data.layers):
            slopes = np.asarray(
                [
                    np.polyfit(data.progress, values[case, layer_index], 1)[0]
                    for case in range(values.shape[0])
                ],
                dtype=np.float64,
            )
            row = {
                "metric": metric,
                "layer": layer,
                "p_value": one_sample_p_value(slopes),
            }
            row.update(
                {
                    f"slope_{key}": value
                    for key, value in _summary_row(
                        slopes,
                        data.pair_ids,
                        replicates=replicates,
                        confidence=confidence,
                        rng=rng,
                    ).items()
                }
            )
            grouped_indices[metric].append(len(rows))
            rows.append(row)

    difference = data.scores["H_probe"] - data.scores["R_probe"]
    metric = "H_probe_minus_R_probe"
    for layer_index, layer in enumerate(data.layers):
        slopes = np.asarray(
            [
                np.polyfit(data.progress, difference[case, layer_index], 1)[0]
                for case in range(difference.shape[0])
            ],
            dtype=np.float64,
        )
        row = {
            "metric": metric,
            "layer": layer,
            "p_value": one_sample_p_value(slopes),
        }
        row.update(
            {
                f"slope_{key}": value
                for key, value in _summary_row(
                    slopes,
                    data.pair_ids,
                    replicates=replicates,
                    confidence=confidence,
                    rng=rng,
                ).items()
            }
        )
        grouped_indices[metric].append(len(rows))
        rows.append(row)

    for indices in grouped_indices.values():
        adjusted = _bh_adjust([rows[index]["p_value"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            rows[index]["fdr_q_value"] = q_value
    return rows


def _group_indices(groups: Sequence[str]) -> tuple[np.ndarray, ...]:
    mapping: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, group in enumerate(groups):
        mapping.setdefault(str(group), []).append(index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in mapping.values())


def _random_intercept_components(
    X: np.ndarray,
    y: np.ndarray,
    group_rows: Sequence[np.ndarray],
    log_variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    residual_variance = float(np.exp(log_variances[0]))
    random_variance = float(np.exp(log_variances[1]))
    xt_vinv_x = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    xt_vinv_y = np.zeros(X.shape[1], dtype=np.float64)
    log_determinant = 0.0
    for indices in group_rows:
        group_x = X[indices]
        group_y = y[indices]
        count = len(indices)
        coefficient = random_variance / (
            residual_variance * (residual_variance + count * random_variance)
        )
        vinv_x = group_x / residual_variance - coefficient * group_x.sum(axis=0)
        vinv_y = group_y / residual_variance - coefficient * group_y.sum()
        xt_vinv_x += group_x.T @ vinv_x
        xt_vinv_y += group_x.T @ vinv_y
        log_determinant += (count - 1) * math.log(residual_variance)
        log_determinant += math.log(residual_variance + count * random_variance)
    beta = np.linalg.pinv(xt_vinv_x, rcond=1e-10) @ xt_vinv_y
    residual = y - X @ beta
    quadratic = 0.0
    for indices in group_rows:
        group_residual = residual[indices]
        count = len(indices)
        coefficient = random_variance / (
            residual_variance * (residual_variance + count * random_variance)
        )
        quadratic += float(group_residual @ group_residual) / residual_variance
        quadratic -= coefficient * float(group_residual.sum() ** 2)
    log_likelihood = -0.5 * (
        len(y) * math.log(2.0 * math.pi) + log_determinant + quadratic
    )
    return log_likelihood, beta, xt_vinv_x, residual_variance, random_variance


def fit_random_intercept_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    *,
    column_names: Sequence[str],
) -> dict[str, Any]:
    """Fit a Gaussian random-intercept model by maximum likelihood."""

    design = np.asarray(X, dtype=np.float64)
    outcome = np.asarray(y, dtype=np.float64)
    if design.ndim != 2 or outcome.ndim != 1 or len(design) != len(outcome):
        raise ValueError("mixed-model design and outcome shapes do not align")
    if design.shape[1] != len(column_names):
        raise ValueError("mixed-model column_names do not match design columns")
    if len(groups) != len(outcome):
        raise ValueError("mixed-model groups do not match observations")
    if len(set(groups)) < 2:
        raise ValueError("mixed model requires at least two pair_id groups")
    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("mixed model requires finite values")
    group_rows = _group_indices(groups)
    variance = max(float(np.var(outcome)), 1e-6)

    def objective(parameters: np.ndarray) -> float:
        log_likelihood, *_ = _random_intercept_components(
            design, outcome, group_rows, parameters
        )
        return -log_likelihood

    result = optimize.minimize(
        objective,
        np.log([variance * 0.75, variance * 0.25]),
        method="L-BFGS-B",
        bounds=((-20.0, 10.0), (-20.0, 10.0)),
    )
    log_likelihood, beta, information, residual_variance, random_variance = (
        _random_intercept_components(design, outcome, group_rows, result.x)
    )
    covariance = np.linalg.pinv(information, rcond=1e-10)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    information_rank = int(np.linalg.matrix_rank(information, tol=1e-10))
    information_singular = bool(
        information_rank < design.shape[1]
        or not np.isfinite(standard_errors).all()
        or np.any(standard_errors <= 0.0)
    )
    coefficients = []
    for name, estimate, standard_error in zip(column_names, beta, standard_errors):
        z_value = (
            float(estimate / standard_error)
            if standard_error > 0
            else float("inf") if estimate != 0 else 0.0
        )
        coefficients.append(
            {
                "term": name,
                "estimate": float(estimate),
                "standard_error": float(standard_error),
                "z_value": z_value,
                "p_value": float(2.0 * stats.norm.sf(abs(z_value))),
            }
        )
    variance_boundary = bool(
        any(
            math.isclose(float(value), bound, rel_tol=0.0, abs_tol=1e-4)
            for value in result.x
            for bound in (-20.0, 10.0)
        )
    )
    converged = bool(result.success) and math.isfinite(float(log_likelihood))
    inference_status = (
        "ok"
        if converged and not variance_boundary and not information_singular
        else "invalid_variance_boundary"
        if variance_boundary
        else "invalid_singular_information"
        if information_singular
        else "invalid_nonconvergence"
    )
    for coefficient in coefficients:
        coefficient["inference_status"] = inference_status
        if not math.isfinite(float(coefficient["z_value"])):
            coefficient["z_value"] = None
        if inference_status != "ok":
            coefficient["p_value"] = None
    return {
        "converged": converged,
        "inference_status": inference_status,
        "optimizer_message": str(result.message),
        "n_observations": int(len(outcome)),
        "n_pairs": int(len(group_rows)),
        "fixed_effect_count": int(design.shape[1]),
        "log_likelihood": float(log_likelihood),
        "aic": float(-2.0 * log_likelihood + 2.0 * (design.shape[1] + 2)),
        "residual_variance": residual_variance,
        "random_intercept_variance": random_variance,
        "information_rank": information_rank,
        "information_singular": information_singular,
        "variance_parameters_at_boundary": variance_boundary,
        "log_variance_parameters": [float(value) for value in result.x],
        "coefficients": coefficients,
    }


def _lrt(reduced: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    degrees = int(full["fixed_effect_count"]) - int(reduced["fixed_effect_count"])
    if degrees <= 0:
        raise ValueError("likelihood-ratio models are not properly nested")
    issues = []
    for name, model in (("reduced", reduced), ("full", full)):
        if model.get("converged") is not True:
            issues.append(f"{name} model did not converge")
        if model.get("variance_parameters_at_boundary") is True:
            issues.append(f"{name} model variance estimate is at an optimizer boundary")
        if model.get("information_singular") is True:
            issues.append(f"{name} model fixed-effect information is singular")
    try:
        reduced_likelihood = float(reduced["log_likelihood"])
        full_likelihood = float(full["log_likelihood"])
    except (KeyError, TypeError, ValueError):
        reduced_likelihood = float("nan")
        full_likelihood = float("nan")
    if not math.isfinite(reduced_likelihood) or not math.isfinite(full_likelihood):
        issues.append("model log-likelihood is not finite")
    tolerance = 1e-7 * max(
        1.0,
        abs(reduced_likelihood) if math.isfinite(reduced_likelihood) else 1.0,
        abs(full_likelihood) if math.isfinite(full_likelihood) else 1.0,
    )
    if (
        math.isfinite(reduced_likelihood)
        and math.isfinite(full_likelihood)
        and full_likelihood < reduced_likelihood - tolerance
    ):
        issues.append("full model log-likelihood is below the nested reduced model")
    if issues:
        return {
            "status": "invalid",
            "reason": "; ".join(issues),
            "likelihood_ratio": None,
            "degrees_of_freedom": degrees,
            "p_value": None,
        }
    statistic = max(0.0, 2.0 * (full_likelihood - reduced_likelihood))
    return {
        "status": "ok",
        "reason": None,
        "likelihood_ratio": statistic,
        "degrees_of_freedom": degrees,
        "p_value": float(stats.chi2.sf(statistic, degrees)),
    }


def _layer_design(
    values: np.ndarray,
    progress: np.ndarray,
    pair_ids: Sequence[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    cases, layers, steps = values.shape
    y = values.reshape(-1)
    progress_column = np.tile(progress, cases * layers)
    layer_index = np.tile(np.repeat(np.arange(layers), steps), cases)
    group_values = np.repeat(np.asarray(pair_ids, dtype=object), layers * steps).tolist()
    dummies = np.column_stack(
        [(layer_index == layer).astype(np.float64) for layer in range(1, layers)]
    ) if layers > 1 else np.empty((len(y), 0), dtype=np.float64)
    intercept = np.ones((len(y), 1), dtype=np.float64)
    progress_matrix = progress_column[:, None]
    layer_only = np.column_stack((intercept, dummies))
    progress_only = np.column_stack((intercept, progress_matrix))
    additive = np.column_stack((intercept, dummies, progress_matrix))
    interactions = dummies * progress_matrix if layers > 1 else dummies
    full = np.column_stack((additive, interactions))
    layer_names = [f"layer[{layer}]" for layer in range(1, layers)]
    full_names = ["intercept", *layer_names, "progress"] + [
        f"layer[{layer}]:progress" for layer in range(1, layers)
    ]
    return y, layer_only, progress_only, additive, full, group_values, full_names


def _mixed_effects_for_metric(
    values: np.ndarray, progress: np.ndarray, pair_ids: Sequence[str]
) -> dict[str, Any]:
    y, layer_only, progress_only, additive, full, groups, full_names = _layer_design(
        values, progress, pair_ids
    )
    layers = values.shape[1]
    layer_names = ["intercept"] + [f"layer[{layer}]" for layer in range(1, layers)]
    progress_names = ["intercept", "progress"]
    additive_names = [*layer_names, "progress"]
    fitted_layer = fit_random_intercept_model(
        layer_only, y, groups, column_names=layer_names
    )
    fitted_progress = fit_random_intercept_model(
        progress_only, y, groups, column_names=progress_names
    )
    fitted_additive = fit_random_intercept_model(
        additive, y, groups, column_names=additive_names
    )
    fitted_full = fit_random_intercept_model(
        full, y, groups, column_names=full_names
    )
    return {
        "model_formula": "score ~ C(layer) * progress + (1 | pair_id)",
        "layer_effect": _lrt(fitted_progress, fitted_additive),
        "attack_progress_effect": _lrt(fitted_layer, fitted_additive),
        "layer_by_progress_interaction": _lrt(fitted_additive, fitted_full),
        "full_model": fitted_full,
    }


def _state_comparison_model(data: ValidatedScoreData) -> dict[str, Any]:
    harmfulness = data.scores["H_probe"]
    refusal = data.scores["R_probe"]
    cases, layers, steps = harmfulness.shape
    base_values = np.concatenate((harmfulness.reshape(-1), refusal.reshape(-1)))
    observations = cases * layers * steps
    progress = np.tile(data.progress, cases * layers)
    layer_index = np.tile(np.repeat(np.arange(layers), steps), cases)
    groups_single = np.repeat(
        np.asarray(data.pair_ids, dtype=object), layers * steps
    ).tolist()
    groups = groups_single + groups_single
    state = np.concatenate((np.zeros(observations), np.ones(observations)))
    progress = np.concatenate((progress, progress))
    layer_index = np.concatenate((layer_index, layer_index))
    layer_dummies = np.column_stack(
        [(layer_index == layer).astype(np.float64) for layer in range(1, layers)]
    ) if layers > 1 else np.empty((2 * observations, 0), dtype=np.float64)
    intercept = np.ones((2 * observations, 1), dtype=np.float64)
    state_column = state[:, None]
    progress_column = progress[:, None]
    state_layer = layer_dummies * state_column
    layer_progress = layer_dummies * progress_column
    state_progress = state_column * progress_column
    triple = layer_dummies * state_progress
    base = np.column_stack(
        (intercept, layer_dummies, progress_column, state_column, state_layer, layer_progress)
    )
    with_state_progress = np.column_stack((base, state_progress))
    with_triple = np.column_stack((with_state_progress, triple))
    layer_names = [f"layer[{layer}]" for layer in range(1, layers)]
    base_names = [
        "intercept",
        *layer_names,
        "progress",
        "state[R-vs-H]",
        *[f"state:layer[{layer}]" for layer in range(1, layers)],
        *[f"layer[{layer}]:progress" for layer in range(1, layers)],
    ]
    state_progress_names = [*base_names, "state:progress"]
    triple_names = [*state_progress_names] + [
        f"state:layer[{layer}]:progress" for layer in range(1, layers)
    ]
    fitted_base = fit_random_intercept_model(
        base, base_values, groups, column_names=base_names
    )
    fitted_state_progress = fit_random_intercept_model(
        with_state_progress,
        base_values,
        groups,
        column_names=state_progress_names,
    )
    fitted_triple = fit_random_intercept_model(
        with_triple, base_values, groups, column_names=triple_names
    )
    return {
        "model_formula": (
            "score ~ state_type * C(layer) * progress + (1 | pair_id)"
        ),
        "state_by_progress": _lrt(fitted_base, fitted_state_progress),
        "state_by_layer_by_progress": _lrt(
            fitted_state_progress, fitted_triple
        ),
        "full_model": fitted_triple,
    }


def _behavior_events(
    data: ValidatedScoreData, *, weakening_threshold: float
) -> list[dict[str, Any]]:
    if not math.isfinite(weakening_threshold) or weakening_threshold < 0.0:
        raise ValueError("weakening_threshold must be finite and non-negative")
    mean_refusal = data.scores["R_probe"].mean(axis=1)
    refusal_delta = mean_refusal - mean_refusal[:, [0]]
    rows = []
    for case_index, (case_id, pair_id) in enumerate(
        zip(data.case_ids, data.pair_ids)
    ):
        non_refusal: Optional[int] = None
        compliance: Optional[int] = None
        for step_index, label in enumerate(data.behavior[case_index]):
            status = str(label.get("label_status", "")).strip().casefold()
            if status != "ok":
                continue
            if non_refusal is None and label.get("refusal_label") is False:
                non_refusal = int(data.steps[step_index])
            if compliance is None and label.get("compliance_label") is True:
                compliance = int(data.steps[step_index])
        weakening_candidates = np.flatnonzero(
            refusal_delta[case_index] <= -float(weakening_threshold)
        )
        weakening = (
            int(data.steps[int(weakening_candidates[0])])
            if len(weakening_candidates)
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "pair_id": pair_id,
                "first_refusal_weakening_step": weakening,
                "first_non_refusal_step": non_refusal,
                "first_compliance_step": compliance,
                "weakening_threshold": float(weakening_threshold),
            }
        )
    return rows


def _event_aligned_statistics(
    data: ValidatedScoreData,
    events: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    confidence: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, Any, int], list[tuple[float, str]]] = defaultdict(list)
    event_fields = (
        "first_refusal_weakening_step",
        "first_non_refusal_step",
        "first_compliance_step",
    )
    for case_index, event_row in enumerate(events):
        for event in event_fields:
            event_step = event_row.get(event)
            if event_step is None:
                continue
            for step_index, step in enumerate(data.steps):
                relative_step = int(step) - int(event_step)
                for metric in PRIMARY_METRICS:
                    values = data.scores[metric]
                    for layer_index, layer in enumerate(data.layers):
                        buckets[(event, metric, layer, relative_step)].append(
                            (
                                float(values[case_index, layer_index, step_index]),
                                data.pair_ids[case_index],
                            )
                        )
    rows = []
    total_steps = max(1, int(data.steps[-1]))
    for (event, metric, layer, relative_step), observations in sorted(
        buckets.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]), item[0][3])
    ):
        values = np.asarray([item[0] for item in observations], dtype=np.float64)
        pairs = [item[1] for item in observations]
        row = {
            "event": event,
            "metric": metric,
            "layer": layer,
            "relative_step": relative_step,
            "relative_progress": relative_step / total_steps,
        }
        row.update(
            _summary_row(
                values,
                pairs,
                replicates=replicates,
                confidence=confidence,
                rng=rng,
            )
        )
        rows.append(row)
    return rows


def analyze_stage1_rq1(
    payload: Mapping[str, Any],
    *,
    confidence: float = 0.95,
    bootstrap_replicates: int = 1000,
    seed: int = 42,
    weakening_threshold: float = 0.1,
    include_mixed_effects: bool = True,
) -> Mapping[str, Any]:
    """Return RQ1-a/b/c tables and preregistered statistical tests."""

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be within (0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    data = validate_score_payload(payload)
    if include_mixed_effects and len(data.layers) < 2:
        raise ValueError(
            "Layer-effect mixed models require at least two layers; provide a "
            "multi-layer score artifact or pass --skip-mixed-effects"
        )
    rng = np.random.default_rng(seed)
    cell_statistics = _cell_statistics(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    phase_profiles = _phase_profiles(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    attack_loss_statistics = _attack_loss_statistics(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    loss_state_correlations = _loss_state_correlations(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    reproducibility = _profile_reproducibility(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    layer_slopes = _layer_slopes(
        data,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    behavior_events = _behavior_events(
        data, weakening_threshold=weakening_threshold
    )
    event_aligned = _event_aligned_statistics(
        data,
        behavior_events,
        replicates=bootstrap_replicates,
        confidence=float(confidence),
        rng=rng,
    )
    mixed_effects: Optional[Mapping[str, Any]] = None
    if include_mixed_effects:
        mixed_effects = {
            "H_probe": _mixed_effects_for_metric(
                data.scores["H_probe"], data.progress, data.pair_ids
            ),
            "R_probe": _mixed_effects_for_metric(
                data.scores["R_probe"], data.progress, data.pair_ids
            ),
            "H_vs_R": _state_comparison_model(data),
        }
    return {
        "format": ANALYSIS_FORMAT,
        "version": ANALYSIS_VERSION,
        "axes": {
            "case_ids": list(data.case_ids),
            "pair_ids": list(data.pair_ids),
            "layers": list(data.layers),
            "steps": data.steps.tolist(),
            "progress": data.progress.tolist(),
        },
        "phase_definition": {
            "early": "0 <= progress < 1/3",
            "middle": "1/3 <= progress < 2/3",
            "late": "2/3 <= progress <= 1",
            "bounds": list(PHASE_BOUNDS),
        },
        "cell_statistics": cell_statistics,
        "phase_profiles": phase_profiles,
        "attack_loss_statistics": attack_loss_statistics,
        "loss_state_correlations": loss_state_correlations,
        "profile_reproducibility": reproducibility,
        "layer_slopes": layer_slopes,
        "behavior_events": behavior_events,
        "event_aligned_statistics": event_aligned,
        "mixed_effects": mixed_effects,
        "metadata": {
            "confidence": float(confidence),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_unit": "pair_id",
            "seed": seed,
            "weakening_threshold": float(weakening_threshold),
            "source_metadata": dict(data.metadata),
            "interpretation_guardrail": (
                "H-stable/R-decreasing is a tested hypothesis, not a selection rule"
            ),
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = ("status",),
) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = list(empty_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_rq1_outputs(
    result: Mapping[str, Any], output_dir: str | Path
) -> Mapping[str, str]:
    """Atomically materialize analysis tables plus a compact JSON summary."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    table_names = (
        "cell_statistics",
        "phase_profiles",
        "attack_loss_statistics",
        "loss_state_correlations",
        "profile_reproducibility",
        "layer_slopes",
        "behavior_events",
        "event_aligned_statistics",
    )
    artifacts: dict[str, str] = {}
    for name in table_names:
        rows = result.get(name)
        if not isinstance(rows, Sequence):
            raise ValueError(f"analysis result {name} must be a sequence")
        path = output / f"{name}.csv"
        _atomic_csv(
            path,
            rows,
            empty_fields=(
                "event",
                "metric",
                "layer",
                "relative_step",
                "relative_progress",
                "mean",
                "variance",
                "standard_error",
                "ci_low",
                "ci_high",
                "confidence",
                "sample_count",
                "pair_count",
            )
            if name == "event_aligned_statistics"
            else ("status",),
        )
        artifacts[name] = str(path)
    mixed_path = output / "mixed_effects.json"
    _atomic_json(mixed_path, result.get("mixed_effects"))
    artifacts["mixed_effects"] = str(mixed_path)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {*table_names, "mixed_effects"}
    }
    summary["artifacts"] = artifacts
    summary_path = output / "rq1_summary.json"
    _atomic_json(summary_path, summary)
    artifacts["summary"] = str(summary_path)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weakening-threshold", type=float, default=0.1)
    parser.add_argument("--skip-mixed-effects", action="store_true")
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also render optional matplotlib H/R figures",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_score_payload(args.scores)
    result = analyze_stage1_rq1(
        payload,
        confidence=args.confidence,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        weakening_threshold=args.weakening_threshold,
        include_mixed_effects=not args.skip_mixed_effects,
    )
    artifacts = write_rq1_outputs(result, args.output_dir)
    from reporting.generate_stage1_rq1_report import generate_stage1_rq1_report

    report = generate_stage1_rq1_report(
        result,
        args.output_dir,
        make_plots=args.make_plots,
    )
    artifacts = {**artifacts, **report}
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_FORMAT",
    "ANALYSIS_VERSION",
    "ValidatedScoreData",
    "analyze_stage1_rq1",
    "build_parser",
    "fit_random_intercept_model",
    "load_score_payload",
    "main",
    "phase_for_progress",
    "validate_score_payload",
    "write_rq1_outputs",
]
