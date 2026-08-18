#!/usr/bin/env python3
"""Pair- and seed-aligned method comparisons with dependency-free inference."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union


Metric = Union[str, Callable[[Mapping[str, Any]], Any]]


def _finite_metric(record: Mapping[str, Any], metric: Metric) -> float:
    if callable(metric):
        value = metric(record)
    else:
        value: Any = record
        for piece in metric.split("."):
            if not isinstance(value, Mapping) or piece not in value:
                raise KeyError(metric)
            value = value[piece]
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise TypeError("metric must resolve to a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("metric must be finite")
    return value


def _seed(record: Mapping[str, Any]) -> Any:
    if "seed" in record:
        return record["seed"]
    budget = record.get("budget")
    return budget.get("seed", 0) if isinstance(budget, Mapping) else 0


def _failed(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "")).casefold()
    error_marker = record.get("error")
    explicit_error = (
        isinstance(error_marker, str) and bool(error_marker.strip())
    ) or isinstance(error_marker, (Mapping, list, tuple))
    return status in {"failed", "error"} or explicit_error


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    differences: Sequence[float] | Mapping[Any, Sequence[float]],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Cluster bootstrap by pair ID; sequences are treated as singleton pairs."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if isinstance(differences, Mapping):
        clusters = {
            key: tuple(float(value) for value in values)
            for key, values in differences.items()
        }
    else:
        clusters = {
            index: (float(value),) for index, value in enumerate(differences)
        }
    if not clusters or any(not values for values in clusters.values()):
        raise ValueError("differences must contain non-empty pair clusters")
    if any(not math.isfinite(value) for values in clusters.values() for value in values):
        raise ValueError("differences must be finite")
    keys = tuple(clusters)
    cluster_means = {
        key: statistics.fmean(clusters[key]) for key in keys
    }
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(n_resamples):
        sampled = [rng.choice(keys) for _ in keys]
        bootstrapped.append(
            statistics.fmean(cluster_means[key] for key in sampled)
        )
    tail = (1.0 - confidence) / 2.0
    return percentile(bootstrapped, tail), percentile(bootstrapped, 1.0 - tail)


def paired_cohens_dz(differences: Sequence[float]) -> Optional[float]:
    """Paired standardized mean difference, using sample SD of differences."""

    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("differences cannot be empty")
    if len(values) == 1:
        return None
    deviation = statistics.stdev(values)
    if deviation == 0:
        mean = statistics.fmean(values)
        return 0.0 if mean == 0 else math.copysign(math.inf, mean)
    return statistics.fmean(values) / deviation


def paired_permutation_test(
    differences: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 42,
    exact_max_pairs: int = 18,
) -> dict[str, Any]:
    """Two-sided paired sign-flip permutation test."""

    values = tuple(float(value) for value in differences)
    if not values:
        raise ValueError("differences cannot be empty")
    observed = abs(statistics.fmean(values))
    tolerance = 1e-15
    if len(values) <= exact_max_pairs:
        total = 2 ** len(values)
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            permuted = abs(
                statistics.fmean(sign * value for sign, value in zip(signs, values))
            )
            extreme += permuted + tolerance >= observed
        return {"p_value": extreme / total, "exact": True, "permutations": total}
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        permuted = abs(
            statistics.fmean(
                value if rng.getrandbits(1) else -value for value in values
            )
        )
        extreme += permuted + tolerance >= observed
    return {
        "p_value": (extreme + 1) / (n_resamples + 1),
        "exact": False,
        "permutations": n_resamples,
    }


def compare_paired_methods(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_method: str,
    metric: Metric,
    methods: Optional[Iterable[str]] = None,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare each method to baseline on matching (pair_id, seed) observations.

    Bootstrap resampling is clustered by pair_id so multiple seeds do not
    masquerade as independent prompts. The permutation test uses one
    seed-averaged difference per pair for the same reason.
    """

    rows = list(records)
    if not rows:
        raise ValueError("records cannot be empty")
    values: dict[tuple[str, Any, str], float] = {}
    failure_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    all_methods: set[str] = set()
    failure_reasons: Counter[str] = Counter()
    for record in rows:
        if not isinstance(record, Mapping):
            raise TypeError("each record must be a mapping")
        pair_id = record.get("pair_id")
        method = record.get("method")
        if pair_id is None or method is None:
            raise ValueError("each record requires pair_id and method")
        pair_id, method = str(pair_id), str(method)
        all_methods.add(method)
        total_counts[method] += 1
        if _failed(record):
            failure_counts[method] += 1
            failure_reasons[f"{method}:explicit_failure"] += 1
            continue
        try:
            value = _finite_metric(record, metric)
        except (KeyError, TypeError, ValueError):
            failure_counts[method] += 1
            failure_reasons[f"{method}:missing_or_invalid_metric"] += 1
            continue
        key = (pair_id, _seed(record), method)
        if key in values:
            raise ValueError(
                "duplicate observation for "
                f"pair_id={pair_id!r}, seed={key[1]!r}, method={method!r}"
            )
        values[key] = value

    if baseline_method not in all_methods:
        raise ValueError(f"baseline method {baseline_method!r} is absent")
    candidates = (
        sorted(set(methods))
        if methods is not None
        else sorted(all_methods - {baseline_method})
    )
    comparisons: dict[str, Any] = {}
    baseline_keys = {
        (pair_id, run_seed): value
        for (pair_id, run_seed, method), value in values.items()
        if method == baseline_method
    }
    direction = 1.0 if higher_is_better else -1.0
    for method in candidates:
        method_keys = {
            (pair_id, run_seed): value
            for (pair_id, run_seed, candidate), value in values.items()
            if candidate == method
        }
        common = sorted(
            set(baseline_keys) & set(method_keys),
            key=lambda item: (item[0], repr(item[1])),
        )
        missing_baseline = len(set(method_keys) - set(baseline_keys))
        missing_candidate = len(set(baseline_keys) - set(method_keys))
        if not common:
            comparisons[method] = {
                "status": "no_paired_observations",
                "n_pairs": 0,
                "n_observations": 0,
                "missing_baseline": missing_baseline,
                "missing_candidate": missing_candidate,
            }
            continue
        raw_differences = [
            method_keys[key] - baseline_keys[key] for key in common
        ]
        improvements = [direction * difference for difference in raw_differences]
        clusters: dict[str, list[float]] = defaultdict(list)
        for (pair_id, _), improvement in zip(common, improvements):
            clusters[pair_id].append(improvement)
        pair_means = [
            statistics.fmean(clusters[pair_id]) for pair_id in sorted(clusters)
        ]
        ci_low, ci_high = paired_bootstrap_ci(
            clusters,
            confidence=confidence,
            n_resamples=bootstrap_samples,
            seed=seed,
        )
        permutation = paired_permutation_test(
            pair_means, n_resamples=permutation_samples, seed=seed
        )
        effect_size = paired_cohens_dz(pair_means)
        comparisons[method] = {
            "status": "ok",
            "baseline_method": baseline_method,
            "candidate_method": method,
            "metric": metric if isinstance(metric, str) else "<callable>",
            "higher_is_better": bool(higher_is_better),
            "n_pairs": len(clusters),
            "n_observations": len(common),
            "seeds_per_pair": {
                pair_id: len(clusters[pair_id]) for pair_id in sorted(clusters)
            },
            "mean_baseline": statistics.fmean(
                baseline_keys[key] for key in common
            ),
            "mean_candidate": statistics.fmean(
                method_keys[key] for key in common
            ),
            "mean_raw_difference": statistics.fmean(raw_differences),
            "mean_improvement": statistics.fmean(pair_means),
            "mean_observation_improvement": statistics.fmean(improvements),
            "bootstrap_ci": {
                "confidence": confidence,
                "low": ci_low,
                "high": ci_high,
                "samples": bootstrap_samples,
                "cluster_unit": "pair_id",
            },
            "effect_size_paired_cohens_dz": (
                effect_size
                if effect_size is None or math.isfinite(effect_size)
                else None
            ),
            "effect_size_degenerate": (
                effect_size is not None and not math.isfinite(effect_size)
            ),
            "permutation_test": {
                **permutation,
                "alternative": "two-sided",
                "unit": "pair_id_seed_mean",
            },
            "missing_baseline": missing_baseline,
            "missing_candidate": missing_candidate,
        }
    return {
        "format": "paired-method-statistics",
        "version": 1,
        "baseline_method": baseline_method,
        "metric": metric if isinstance(metric, str) else "<callable>",
        "failure_counts": {
            method: {
                "total_records": total_counts[method],
                "failed_records": failure_counts[method],
                "valid_records": total_counts[method] - failure_counts[method],
            }
            for method in sorted(all_methods)
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "comparisons": comparisons,
    }


paired_method_statistics = compare_paired_methods
paired_statistics = compare_paired_methods


def _read_records(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.casefold() == ".jsonl":
            values = [json.loads(line) for line in handle if line.strip()]
        else:
            values = json.load(handle)
    if isinstance(values, Mapping):
        values = values.get("cases", values.get("records", values.get("results")))
    if not isinstance(values, list) or any(not isinstance(v, Mapping) for v in values):
        raise ValueError("input must contain a list of result records")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_paired_methods(
        _read_records(Path(args.input)),
        baseline_method=args.baseline_method,
        metric=args.metric,
        higher_is_better=not args.lower_is_better,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compare_paired_methods",
    "paired_bootstrap_ci",
    "paired_cohens_dz",
    "paired_method_statistics",
    "paired_permutation_test",
    "paired_statistics",
    "percentile",
]
