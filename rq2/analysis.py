"""Paired causal-effect estimation and formal RQ2 statistical outputs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from experiments.analyze_stage1_rq1 import fit_random_intercept_model
from rq2.preregistration import LAYER_DOSE_VERSION
from rq2.formal_statistics import (
    EVENT_NAMES, FORMAL_STATISTICS_VERSION, MINIMUM_CONFIRMATORY_PAIRS,
    apply_event_families, apply_formal_families, family_policy,
)
from rq2.artifacts import (
    BehaviorLabel,
    LABEL_FORMAT,
    SIDECAR_VERSION,
    atomic_csv,
    atomic_json,
    read_jsonl,
    records_by_id,
    validate_trial_record,
)


class RQ2AnalysisError(ValueError):
    """Raised when causal analysis inputs are incomplete or inconsistent."""


RESTORATION_INTERVENTIONS = {"r_direction", "subspace_restoration"}
BEHAVIOR_ENDPOINT_VERSION = 2
EFFECT_DEFINITIONS = {
    "utility_effect": {
        "restoration": "S_base - S_intervention",
        "suppression": "S_intervention - S_base_clean",
    },
    "refusal_effect": {
        "restoration": "R_intervention - R_base",
        "suppression": "R_base_clean - R_intervention",
    },
    "compliance_effect": {
        "restoration": "C_base - C_intervention",
        "suppression": "C_intervention - C_base_clean",
    },
}


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RQ2AnalysisError("FDR p-values must be a finite vector")
    if len(values) == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = int(order[position])
        running = min(running, float(values[index]) * len(values) / (position + 1))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def pair_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        raise RQ2AnalysisError("bootstrap replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise RQ2AnalysisError("confidence must be within (0,1)")
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None or not math.isfinite(float(value)):
            continue
        by_pair[str(row["pair_id"])].append(float(value))
    if not by_pair:
        raise RQ2AnalysisError(f"no finite values for {value_field}")
    pair_ids = sorted(by_pair)
    pair_means = np.asarray(
        [np.mean(by_pair[pair_id]) for pair_id in pair_ids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draw = rng.integers(0, len(pair_ids), size=len(pair_ids))
        estimates[replicate] = float(np.mean(pair_means[draw]))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(pair_means)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "pair_count": len(pair_ids),
        "observation_count": sum(len(values) for values in by_pair.values()),
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "pair_id",
        "confidence": confidence,
    }


def _label(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("format") != LABEL_FORMAT or record.get("version") != SIDECAR_VERSION:
        raise RQ2AnalysisError("unsupported behavior-label format/version")
    if record.get("label_status") not in {"ok", "unknown"}:
        raise RQ2AnalysisError("invalid label_status")
    forbidden = {"response", "reasoning", "raw_response", "overall_reasoning"}
    if forbidden.intersection(record):
        raise RQ2AnalysisError("labels sidecar contains response or judge text")
    fields = {
        name: record.get(name)
        for name in (
            "trial_id", "label_status", "behavior_label", "refusal_label",
            "compliance_label", "jailbreak_success", "refusal_score",
            "strongreject_score", "refusal_orientation", "convincingness",
            "specificity", "response_sha256", "judge_fingerprint",
            "error_type", "retryable",
        )
    }
    try:
        return BehaviorLabel(**fields).to_record()
    except (TypeError, ValueError) as exc:
        raise RQ2AnalysisError("malformed behavior label") from exc


def _profile_value(record: Mapping[str, Any], metric: str, layer: int) -> Optional[float]:
    diagnostic = record.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return None
    profile = diagnostic.get("profile")
    if not isinstance(profile, Mapping):
        return None
    values = profile.get(metric)
    if not isinstance(values, Mapping):
        return None
    value = values.get(str(layer), values.get(layer))
    return None if value is None else float(value)


def join_effect_rows(
    trial_records: Iterable[Mapping[str, Any]],
    label_records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    trials = records_by_id(validate_trial_record(record) for record in trial_records)
    labels = records_by_id(_label(record) for record in label_records)
    unknown = sum(record.get("label_status") == "unknown" for record in labels.values())
    rows: list[dict[str, Any]] = []
    for trial_id, trial in trials.items():
        if trial["intervention"] == "baseline":
            continue
        baseline_id = trial.get("baseline_trial_id")
        if not isinstance(baseline_id, str) or baseline_id not in trials:
            raise RQ2AnalysisError(f"trial {trial_id} lacks its declared baseline")
        label = labels.get(trial_id)
        baseline_label = labels.get(baseline_id)
        if label is None or baseline_label is None:
            raise RQ2AnalysisError(f"trial {trial_id} or its baseline is unlabeled")
        if label["response_sha256"] != trial["response_sha256"]:
            raise RQ2AnalysisError(f"stale behavior label for trial {trial_id}")
        if baseline_label["response_sha256"] != trials[baseline_id]["response_sha256"]:
            raise RQ2AnalysisError(f"stale baseline label for trial {trial_id}")
        if label["label_status"] != "ok" or baseline_label["label_status"] != "ok":
            continue
        s_patch = float(label["strongreject_score"])
        s_base = float(baseline_label["strongreject_score"])
        base_refusal = bool(baseline_label["refusal_label"])
        intervention_refusal = bool(label["refusal_label"])
        base_compliance = bool(baseline_label["compliance_label"])
        intervention_compliance = bool(label["compliance_label"])
        reverse = str(trial["intervention"]).startswith("reverse_")
        direction = -1 if reverse else 1
        effect = direction * (s_base - s_patch)
        refusal_effect = direction * (int(intervention_refusal) - int(base_refusal))
        compliance_effect = direction * (int(base_compliance) - int(intervention_compliance))
        denominator = 1.0 - s_base if reverse else s_base
        normalized_effect = effect / denominator if denominator > 1e-12 else None
        layer = int(trial["layer"])
        r_patch = _profile_value(trial, "refusal_projection", layer)
        r_base = _profile_value(trials[baseline_id], "refusal_projection", layer)
        h_patch = _profile_value(trial, "harmfulness_projection", layer)
        h_base = _profile_value(trials[baseline_id], "harmfulness_projection", layer)
        rows.append(
            {
                "trial_id": trial_id,
                "baseline_trial_id": baseline_id,
                "pair_id": trial["pair_id"],
                "rq2_role": trial["rq2_role"],
                "state_key": trial["state_key"],
                "step": trial["step"],
                "layer": layer,
                "intervention": trial["intervention"],
                "dose": float(trial["dose"]),
                "token_scope": trial["token_scope"],
                "replicate": trial["replicate"],
                "generation_shift_l2": (
                    trial["audit"].get("shift_l2")
                    if isinstance(trial.get("audit"), Mapping) else None
                ),
                "effect_direction": "suppression" if reverse else "restoration",
                "base_strongreject_score": s_base,
                "intervention_strongreject_score": s_patch,
                "utility_effect": effect,
                "refusal_effect": refusal_effect,
                "compliance_effect": compliance_effect,
                "normalized_utility_effect": normalized_effect,
                # Legacy aliases describe utility, not refusal probability.
                "y_base": 1.0 - s_base,
                "y_intervention": 1.0 - s_patch,
                "causal_effect": effect,
                "ce_restore": None if reverse else effect,
                "ce_suppress": effect if reverse else None,
                "behavior_recovery_rate": normalized_effect,
                "delta_r": None if r_patch is None or r_base is None else r_patch - r_base,
                "delta_h": None if h_patch is None or h_base is None else h_patch - h_base,
                "base_refusal": base_refusal,
                "intervention_refusal": intervention_refusal,
                "base_compliance": base_compliance,
                "intervention_compliance": intervention_compliance,
                "base_convincingness": baseline_label["convincingness"],
                "intervention_convincingness": label["convincingness"],
                "base_specificity": baseline_label["specificity"],
                "intervention_specificity": label["specificity"],
            }
        )
    return rows, {
        "trial_count": len(trials),
        "label_count": len(labels),
        "unknown_label_count": int(unknown),
        "analyzed_effect_count": len(rows),
    }


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["layer"],
        row["state_key"],
        row["intervention"],
        row["dose"],
        row["token_scope"],
    )


def _is_primary_dose(
    row: Mapping[str, Any], restoration: float = 1.0, suppression: float = 1.0,
) -> bool:
    target = suppression if str(row["intervention"]).startswith("reverse_") else restoration
    return math.isclose(float(row["dose"]), target, rel_tol=0.0, abs_tol=1e-12)


def _pair_weighted_mean(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            by_pair[str(row["pair_id"])].append(float(value))
    return (
        float(np.mean([np.mean(values) for values in by_pair.values()]))
        if by_pair else None
    )


def _one_sample_p(values: Sequence[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) < 2:
        return 1.0
    deviation = float(np.std(vector, ddof=1))
    if math.isclose(deviation, 0.0, abs_tol=1e-14):
        return 1.0 if math.isclose(float(np.mean(vector)), 0.0, abs_tol=1e-14) else 0.0
    return float(stats.ttest_1samp(vector, 0.0).pvalue)


def summarize_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
    binary_tests: Optional[Sequence[Mapping[str, Any]]] = None,
    primary_restoration_dose: float = 1.0,
    primary_suppression_dose: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        estimates = {
            endpoint: pair_cluster_bootstrap(
                group,
                value_field=f"{endpoint}_effect",
                replicates=replicates,
                confidence=confidence,
                seed=seed + index,
            )
            for endpoint in ("utility", "refusal", "compliance")
        }
        utility = estimates["utility"]
        pair_values: dict[str, list[float]] = defaultdict(list)
        for row in group:
            pair_values[str(row["pair_id"])].append(float(row["utility_effect"]))
        means = [float(np.mean(values)) for values in pair_values.values()]
        standard_deviation = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        summary = {
            "layer": key[0],
            "state_key": key[1],
            "intervention": key[2],
            "dose": key[3],
            "token_scope": key[4],
            "effect_direction": (
                "suppression" if str(key[2]).startswith("reverse_") else "restoration"
            ),
            "dose_role": (
                "primary" if _is_primary_dose(group[0], primary_restoration_dose, primary_suppression_dose)
                else "sensitivity"
            ),
            "layer_role": group[0].get("layer_role", "unspecified"),
            "causal_effect_mean": utility["mean"],
            "ci_low": utility["ci_low"],
            "ci_high": utility["ci_high"],
            "pair_count": utility["pair_count"],
            "observation_count": utility["observation_count"],
            "sign_consistency": float(np.mean(np.asarray(means) > 0.0)),
            "standardized_effect": (
                float(np.mean(means)) / standard_deviation
                if standard_deviation > 1e-12 else None
            ),
            "p_value": _one_sample_p(means),
        }
        for endpoint, estimate in estimates.items():
            summary.update({
                f"{endpoint}_effect_mean": estimate["mean"],
                f"{endpoint}_ci_low": estimate["ci_low"],
                f"{endpoint}_ci_high": estimate["ci_high"],
                f"{endpoint}_pair_count": estimate["pair_count"],
            })
        for field in (
            "normalized_utility_effect", "delta_r", "delta_h",
            "base_strongreject_score", "intervention_strongreject_score",
            "base_refusal", "intervention_refusal",
            "base_compliance", "intervention_compliance",
            "base_convincingness", "intervention_convincingness",
            "base_specificity", "intervention_specificity",
        ):
            summary[f"{field}_mean"] = _pair_weighted_mean(group, field)
        # Retained for old consumers; this is a normalized utility-score change.
        summary["behavior_recovery_rate_mean"] = summary["normalized_utility_effect_mean"]
        summaries.append(summary)
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(summaries):
        primary = (
            row["intervention"] in RESTORATION_INTERVENTIONS
            and row["token_scope"] == "audio"
        ) or row["intervention"] == "reverse_suppression"
        family = ("primary" if primary else "specificity_control") if row["dose_role"] == "primary" else "sensitivity"
        row["fdr_family"] = family
        families[family].append(index)
    for family, indices in families.items():
        adjusted = bh_fdr([float(summaries[index]["p_value"]) for index in indices])
        for index, q_value in zip(indices, adjusted):
            summaries[index]["fdr_q_value"] = q_value
            summaries[index]["utility_p_value"] = summaries[index]["p_value"]
            summaries[index]["utility_fdr_q_value"] = q_value
            summaries[index]["utility_fdr_family"] = family

    binary = (
        paired_binary_tests(
            rows, primary_restoration_dose=primary_restoration_dose,
            primary_suppression_dose=primary_suppression_dose,
        )
        if binary_tests is None else binary_tests
    )
    binary_by_key = {(_group_key(row), row["outcome"]): row for row in binary}
    for row in summaries:
        for endpoint in ("refusal", "compliance"):
            evidence = binary_by_key.get((_group_key(row), endpoint))
            if evidence is None or evidence["pair_count"] != row[f"{endpoint}_pair_count"]:
                raise RQ2AnalysisError("binary evidence must match the paired effect population")
            row.update({
                f"{endpoint}_inference_status": evidence["inference_status"],
                f"{endpoint}_p_value": evidence["mcnemar_exact_p_value"],
                f"{endpoint}_fdr_q_value": evidence["fdr_q_value"],
                f"{endpoint}_fdr_family": evidence["fdr_family"],
                f"{endpoint}_false_to_true_count": evidence["baseline_false_to_intervention_true"],
                f"{endpoint}_true_to_false_count": evidence["baseline_true_to_intervention_false"],
            })
    return summaries


def paired_binary_tests(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_restoration_dose: float = 1.0,
    primary_suppression_dose: float = 1.0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_pair[str(row["pair_id"])].append(row)
        for outcome in ("refusal", "compliance"):
            left = f"base_{outcome}"
            right = f"intervention_{outcome}"
            paired = []
            for pair_rows in by_pair.values():
                baseline_values = {bool(row[left]) for row in pair_rows}
                if len(baseline_values) != 1:
                    raise RQ2AnalysisError("replicates disagree on their baseline binary label")
                intervention = float(np.mean([bool(row[right]) for row in pair_rows]))
                paired.append((next(iter(baseline_values)), intervention))
            # Fractional replicate means are retained in the pair bootstrap.
            # They are not binary observations for an exact McNemar test.
            binary_pairs = sum(intervention in (0.0, 1.0) for _, intervention in paired)
            is_binary = binary_pairs == len(paired)
            b_to_i = (
                sum(not base and intervention == 1.0 for base, intervention in paired)
                if is_binary else None
            )
            i_to_b = (
                sum(base and intervention == 0.0 for base, intervention in paired)
                if is_binary else None
            )
            discordant = b_to_i + i_to_b if is_binary else None
            p_value = (
                float(stats.binomtest(min(b_to_i, i_to_b), discordant, 0.5).pvalue)
                if discordant else (1.0 if is_binary else None)
            )
            reverse = str(key[2]).startswith("reverse_")
            direction = (-1 if reverse else 1) * (1 if outcome == "refusal" else -1)
            output.append({
                "layer": key[0],
                "state_key": key[1],
                "intervention": key[2],
                "dose": key[3],
                "token_scope": key[4],
                "outcome": outcome,
                "paired_effect_mean": float(np.mean([
                    direction * (intervention - int(base)) for base, intervention in paired
                ])),
                "baseline_false_to_intervention_true": b_to_i,
                "baseline_true_to_intervention_false": i_to_b,
                "discordant_count": discordant,
                "pair_count": len(paired),
                "binary_pair_count": binary_pairs,
                "replicate_aggregation": "within-pair-mean",
                "inference_status": "ok" if is_binary else "non_binary_replicate_mean",
                "mcnemar_exact_p_value": p_value,
                "fdr_family": (
                    "paired_binary" if _is_primary_dose(group[0], primary_restoration_dose, primary_suppression_dose)
                    else "paired_binary_sensitivity"
                ),
                "fdr_q_value": None,
            })
    # Keep unavailable comparisons in the correction family without reporting
    # an invented test result or changing the mean-based endpoint.
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        families[row["fdr_family"]].append(row)
    for family in families.values():
        q_values = bh_fdr([
            1.0 if row["mcnemar_exact_p_value"] is None else row["mcnemar_exact_p_value"]
            for row in family
        ])
        for row, value in zip(family, q_values):
            if row["mcnemar_exact_p_value"] is not None:
                row["fdr_q_value"] = value
    return output


def fit_layer_state_model(
    rows: Sequence[Mapping[str, Any]], *, primary_restoration_dose: float = 1.0,
) -> Mapping[str, Any]:
    primary = [
        row for row in rows
        if row["intervention"] in RESTORATION_INTERVENTIONS and row["token_scope"] == "audio"
        and _is_primary_dose(row, primary_restoration_dose)
    ]
    pairs = sorted({str(row["pair_id"]) for row in primary})
    layers = sorted({int(row["layer"]) for row in primary})
    states = sorted({str(row["state_key"]) for row in primary})
    if len(pairs) < 2 or len(primary) < 4:
        return {"inference_status": "insufficient_data", "n_observations": len(primary)}
    reference_layer = layers[0]
    reference_state = states[0]
    layer_terms = layers[1:]
    state_terms = states[1:]
    columns = ["intercept"]
    columns.extend(f"layer[{layer}]" for layer in layer_terms)
    columns.extend(f"state[{state}]" for state in state_terms)
    columns.extend(
        f"layer[{layer}]:state[{state}]"
        for layer in layer_terms
        for state in state_terms
    )
    design = []
    reduced_design = []
    layer_only_design = []
    for row in primary:
        layer_indicators = [float(int(row["layer"]) == layer) for layer in layer_terms]
        state_indicators = [float(row["state_key"] == state) for state in state_terms]
        interactions = [left * right for left in layer_indicators for right in state_indicators]
        reduced_row = [1.0, *layer_indicators, *state_indicators]
        layer_only_design.append([1.0, *layer_indicators])
        reduced_design.append(reduced_row)
        design.append([*reduced_row, *interactions])
    try:
        outcomes = np.asarray([float(row["causal_effect"]) for row in primary])
        pair_groups = [str(row["pair_id"]) for row in primary]
        fitted = fit_random_intercept_model(
            np.asarray(design),
            outcomes,
            pair_groups,
            column_names=columns,
        )
        def likelihood_ratio(reduced: Mapping[str, Any]) -> dict[str, Any]:
            degrees = int(fitted["fixed_effect_count"]) - int(
                reduced["fixed_effect_count"]
            )
            full_likelihood = float(fitted["log_likelihood"])
            reduced_likelihood = float(reduced["log_likelihood"])
            tolerance = 1e-7 * max(1.0, abs(full_likelihood), abs(reduced_likelihood))
            valid = (
                degrees > 0
                and fitted.get("inference_status") == "ok"
                and reduced.get("inference_status") == "ok"
                and full_likelihood >= reduced_likelihood - tolerance
            )
            statistic = max(0.0, 2.0 * (full_likelihood - reduced_likelihood))
            return {
                "status": "ok" if valid else "invalid_model_fit",
                "likelihood_ratio": statistic if valid else None,
                "degrees_of_freedom": degrees,
                "p_value": (
                    float(stats.chi2.sf(statistic, degrees)) if valid else None
                ),
                "reduced_inference_status": reduced.get("inference_status"),
                "full_inference_status": fitted.get("inference_status"),
            }
        if not layer_terms or not state_terms:
            interaction_lrt = {
                "status": "insufficient_design",
                "degrees_of_freedom": 0,
                "p_value": None,
            }
        else:
            reduced = fit_random_intercept_model(
                np.asarray(reduced_design),
                outcomes,
                pair_groups,
                column_names=columns[: len(reduced_design[0])],
            )
            interaction_lrt = likelihood_ratio(reduced)
        if not state_terms:
            state_profile_lrt = {
                "status": "insufficient_design",
                "degrees_of_freedom": 0,
                "p_value": None,
            }
        else:
            layer_only = fit_random_intercept_model(
                np.asarray(layer_only_design),
                outcomes,
                pair_groups,
                column_names=columns[: len(layer_only_design[0])],
            )
            state_profile_lrt = likelihood_ratio(layer_only)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return {
            "inference_status": "fit_error",
            "error_type": type(exc).__name__,
            "n_observations": len(primary),
        }
    return {
        **fitted,
        "layer_encoding": "categorical",
        "reference_layer": reference_layer,
        "reference_state": reference_state,
        "interaction_likelihood_ratio_test": interaction_lrt,
        "state_profile_likelihood_ratio_test": state_profile_lrt,
    }


def compare_degradation_to_causality(
    rq1_slopes_path: str | Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    candidate_layers: Sequence[int] = (),
    primary_restoration_dose: float = 1.0,
    rq2_trajectory_trials_path: Optional[str | Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare frozen RQ1 rankings and within-AdvBench R decline with formal utility effects."""
    base = Path(rq1_slopes_path).expanduser().resolve()
    rq1_root = base.parent.parent if base.parent.name in {"all", "baseline_refused"} else None
    slope_sources = (
        [(population, rq1_root / population / "layer_slopes.csv")
         for population in ("all", "baseline_refused")]
        if rq1_root is not None else [("configured", base)]
    )
    predictors: dict[str, dict[int, float]] = {}
    source_details: dict[str, dict[str, Any]] = {}
    for population, source in slope_sources:
        if not source.is_file():
            raise RQ2AnalysisError(f"frozen RQ1 slope source is missing: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            slope_rows = list(csv.DictReader(handle))
        for metric in ("R_probe", "R_direction"):
            values = {
                int(row["layer"]): -float(row["slope_mean"])
                for row in slope_rows if row.get("metric") == metric
            }
            if not values:
                raise RQ2AnalysisError(f"RQ1 {population} lacks {metric} slopes")
            key = f"rq1:{population}:{metric}_slope"
            predictors[key] = values
            source_details[key] = {
                "population": population, "metric": metric,
                "source_path": str(source),
                "interpretation": "cross-dataset observational RQ1 slope ranking",
            }
        if rq1_root is None:
            continue
        event_path = rq1_root / population / "event_aligned_statistics.csv"
        if not event_path.is_file():
            raise RQ2AnalysisError(f"frozen RQ1 event source is missing: {event_path}")
        with event_path.open("r", encoding="utf-8-sig", newline="") as handle:
            event_rows = list(csv.DictReader(handle))
        event_means: dict[tuple[str, int, int], float] = {}
        event_counts: dict[tuple[str, int, int], int] = {}
        for row in event_rows:
            if row.get("metric") != "R_probe":
                continue
            offset = int(row["relative_step"])
            if offset in (-2, 3):
                key = (str(row["event"]), int(row["layer"]), offset)
                event_means[key] = float(row["mean"])
                event_counts[key] = int(row["pair_count"])
        for event in sorted({item[0] for item in event_means}):
            values = {
                layer: event_means[(event, layer, -2)] - event_means[(event, layer, 3)]
                for layer in sorted({item[1] for item in event_means if item[0] == event})
                if (event, layer, -2) in event_means
                and (event, layer, 3) in event_means
                and event_counts[(event, layer, -2)] > 0
                and event_counts[(event, layer, 3)] > 0
            }
            key = f"rq1:{population}:event:{event}:R_probe_drop"
            predictors[key] = values
            source_details[key] = {
                "population": population, "metric": "R_probe_event_drop",
                "event": event, "source_path": str(event_path),
                "interpretation": "cross-dataset descriptive aggregate -2 minus +3; event offset populations may differ",
            }
    if rq2_trajectory_trials_path is not None:
        trajectory_path = Path(rq2_trajectory_trials_path).expanduser().resolve()
        if not trajectory_path.is_file():
            raise RQ2AnalysisError(f"RQ2 no-intervention trajectory source is missing: {trajectory_path}")
        by_pair: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
        for trial in read_jsonl(trajectory_path):
            if trial.get("rq2_role") != "rq2_causal_test":
                continue
            state = str(trial.get("state_key"))
            if state not in {"scan:0", "scan:100"}:
                continue
            profile = trial.get("diagnostic", {}).get("profile", {}).get("refusal_projection", {})
            for layer, value in profile.items():
                by_pair[(str(trial["pair_id"]), int(layer))][int(state[5:])] = float(value)
        local_values: dict[int, list[float]] = defaultdict(list)
        for (_, layer), values in by_pair.items():
            if 0 in values and 100 in values:
                local_values[layer].append(values[0] - values[100])
        key = "rq2:causal_test:scan_0_to_100:R_projection_drop"
        predictors[key] = {
            layer: float(np.mean(values)) for layer, values in local_values.items()
        }
        source_details[key] = {
            "population": "rq2_causal_test", "metric": "R_projection_drop",
            "source_path": str(trajectory_path),
            "pair_counts_by_layer": {str(layer): len(values) for layer, values in local_values.items()},
            "interpretation": "same-AdvBench paired no-intervention t0 minus t100 R projection",
        }
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in summaries:
        if (
            row["intervention"] in RESTORATION_INTERVENTIONS and row["token_scope"] == "audio"
            and _is_primary_dose(row, primary_restoration_dose)
            and row.get("utility_inference_status", "ok") == "ok"
        ):
            grouped[int(row["layer"])].append(float(row["causal_effect_mean"]))
    causal = {layer: float(np.mean(values)) for layer, values in grouped.items()}
    primary_source = f"rq1:{base.parent.name}:R_probe_slope" if rq1_root is not None else "rq1:configured:R_probe_slope"
    output: list[dict[str, Any]] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    candidates = set(map(int, candidate_layers))
    for source, predictor in predictors.items():
        shared = sorted(set(predictor).intersection(causal))
        correlation = (
            stats.spearmanr([predictor[layer] for layer in shared], [causal[layer] for layer in shared])
            if len(shared) >= 2 else None
        )
        rho = None if correlation is None or not math.isfinite(float(correlation.statistic)) else float(correlation.statistic)
        p_value = None if correlation is None or not math.isfinite(float(correlation.pvalue)) else float(correlation.pvalue)
        count = min(max(int(top_k), 1), len(shared)) if shared else 0
        source_top = sorted(shared, key=lambda layer: (-predictor[layer], layer))[:count]
        rq2_top = sorted(shared, key=lambda layer: (-causal[layer], layer))[:count]
        overlap = sorted(set(source_top).intersection(rq2_top))
        metrics = {
            **source_details[source],
            "source": source,
            "shared_layer_count": len(shared),
            "spearman_rho": rho, "spearman_p_value": p_value,
            "top_k": count,
            "rq1_top_layers": source_top, "rq2_top_layers": rq2_top,
            "top_k_overlap": len(overlap), "top_k_overlap_layers": overlap,
            "candidate_precision": (
                len(candidates.intersection(rq2_top)) / len(candidates) if candidates else None
            ),
        }
        source_metrics[source] = metrics
        for layer in shared:
            output.append({
                "source": source, "comparison_role": "primary" if source == primary_source else "descriptive",
                "population": metrics["population"], "metric": metrics["metric"],
                "event": metrics.get("event"), "layer": layer,
                "predictor_drop": predictor[layer], "rq1_r_drop": predictor[layer],
                "rq2_causal_effect": causal[layer],
                **{field: metrics[field] for field in (
                    "shared_layer_count", "spearman_rho", "spearman_p_value",
                    "top_k", "rq1_top_layers", "rq2_top_layers",
                    "top_k_overlap", "top_k_overlap_layers", "candidate_precision",
                )},
            })
    primary = source_metrics.get(primary_source, {})
    return output, {
        **primary,
        "primary_source": primary_source,
        "sources": source_metrics,
        "interpretation": "ranking agreement is descriptive; cross-dataset correlations are not causal evidence",
    }


def paired_specificity_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_layers: Sequence[int],
    neighbor_layers: Sequence[int] = (),
    distant_control_layers: Sequence[int] = (),
    depth_control_layers: Sequence[int] = (),
    replicates: int,
    confidence: float,
    seed: int,
    primary_restoration_dose: float = 1.0,
    primary_suppression_dose: float = 1.0,
) -> list[dict[str, Any]]:
    """Separate same-dose, same-layer specificity from descriptive layer profiles."""
    if distant_control_layers and depth_control_layers:
        raise RQ2AnalysisError("use distant_control_layers or its v1 alias, not both")
    distant = set(distant_control_layers or depth_control_layers)
    candidates, neighbors = set(candidate_layers), set(neighbor_layers)
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["pair_id"]), str(row["state_key"]), int(row["layer"]),
            str(row["intervention"]), float(row["dose"]), str(row["token_scope"]),
        )
        buckets[key].append(float(row["causal_effect"]))
    averaged = {key: float(np.mean(values)) for key, values in buckets.items()}
    shift_norms: dict[tuple[Any, ...], list[float | None]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["pair_id"]), str(row["state_key"]), int(row["layer"]),
            str(row["intervention"]), float(row["dose"]), str(row["token_scope"]),
        )
        try:
            value = float(row.get("generation_shift_l2"))
        except (TypeError, ValueError):
            value = float("nan")
        shift_norms[key].append(value if math.isfinite(value) and value >= 0 else None)
    paired: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    def add(
        name: str, key: tuple[Any, ...], control_key: tuple[Any, ...],
        *, scope: str = "same_layer_specificity", control_role: str = "same_layer",
    ) -> None:
        if control_key not in averaged:
            return
        pair_id, state, layer, intervention, dose, _ = key
        comparison = (name, layer, state, control_key[2], intervention, dose, scope, control_role)
        requires_zero_norm = (
            scope == "same_layer_specificity"
            and control_key[3] in {"sham", "reverse_sham"}
        )
        requires_equal_norm = scope == "same_layer_specificity" and not requires_zero_norm
        norm_error = None
        if requires_equal_norm:
            primary_norms = shift_norms.get(key, [])
            control_norms = shift_norms.get(control_key, [])
            if (
                primary_norms and control_norms
                and all(value is not None and value > 0 for value in (*primary_norms, *control_norms))
            ):
                norm_error = max(
                    abs(primary_norm - control_norm) / max(primary_norm, control_norm)
                    for primary_norm in primary_norms
                    for control_norm in control_norms
                )
        paired[comparison].append({
            "pair_id": pair_id, "contrast_effect": averaged[key] - averaged[control_key],
            "norm_audit_required": requires_equal_norm or requires_zero_norm,
            "norm_relative_error": norm_error,
            "sham_shift_l2": (
                max(shift_norms[control_key])
                if requires_zero_norm and shift_norms.get(control_key)
                and all(value is not None for value in shift_norms[control_key])
                else None
            ) if requires_zero_norm else None,
        })

    for key in averaged:
        pair_id, state, layer, intervention, dose, token_scope = key
        if token_scope != "audio":
            continue
        if intervention in RESTORATION_INTERVENTIONS:
            prefix = "subspace" if intervention == "subspace_restoration" else "r"
            h_control = "subspace_h_control" if prefix == "subspace" else "h_direction_control"
            random_control = "subspace_random_control" if prefix == "subspace" else "random_direction_control"
            for name, control in (
                (f"{prefix}_vs_h", h_control),
                (f"{prefix}_vs_random", random_control),
                (f"{prefix}_vs_sham", "sham"),
            ):
                add(name, key, (pair_id, state, layer, control, dose, "audio"))
            add("audio_vs_position", key, (pair_id, state, layer, intervention, dose, "position_control"))
            if layer in candidates:
                for control_layer in sorted((candidates | neighbors) - {layer}):
                    if abs(control_layer - layer) == 1:
                        add(
                            "candidate_vs_neighbor", key,
                            (pair_id, state, control_layer, intervention, dose, "audio"),
                            scope="layer_profile",
                            control_role="candidate" if control_layer in candidates else "neighbor",
                        )
                for control_layer in sorted(distant):
                    add(
                        "candidate_vs_distant", key,
                        (pair_id, state, control_layer, intervention, dose, "audio"),
                        scope="layer_profile", control_role="distant",
                    )
        elif intervention == "reverse_suppression":
            for name, control in (
                ("reverse_r_vs_h", "reverse_h_control"),
                ("reverse_r_vs_random", "reverse_random_control"),
                ("reverse_r_vs_sham", "reverse_sham"),
            ):
                add(name, key, (pair_id, state, layer, control, dose, "audio"))

    output: list[dict[str, Any]] = []
    for index, (key, values) in enumerate(sorted(paired.items(), key=lambda item: str(item[0]))):
        estimate = pair_cluster_bootstrap(
            values, value_field="contrast_effect", replicates=replicates,
            confidence=confidence, seed=seed + index,
        )
        pair_effects: dict[str, list[float]] = defaultdict(list)
        for item in values:
            pair_effects[str(item["pair_id"])].append(float(item["contrast_effect"]))
        row = {
            "contrast": key[0], "layer": key[1], "state_key": key[2],
            "control_layer": key[3], "primary_intervention": key[4], "dose": key[5],
            "token_scope": "audio", "contrast_scope": key[6], "control_layer_role": key[7],
            "contrast_effect_mean": estimate["mean"], "ci_low": estimate["ci_low"],
            "ci_high": estimate["ci_high"], "pair_count": estimate["pair_count"],
            "p_value": _one_sample_p([float(np.mean(group)) for group in pair_effects.values()]),
        }
        required_norms = [item for item in values if item["norm_audit_required"]]
        sham_audit = key[0] in {"r_vs_sham", "subspace_vs_sham", "reverse_r_vs_sham"}
        errors = [item["sham_shift_l2" if sham_audit else "norm_relative_error"] for item in required_norms]
        row["norm_audit_pair_count"] = len(required_norms)
        row["norm_audit_missing_pair_count"] = sum(value is None for value in errors)
        row["norm_max_relative_error"] = (
            None if sham_audit else max((value for value in errors if value is not None), default=None)
        )
        row["sham_max_shift_l2"] = (
            max((value for value in errors if value is not None), default=None)
            if sham_audit else None
        )
        row["norm_audit_status"] = (
            "not_applicable" if not required_norms else
            "missing" if any(value is None for value in errors) else
            "mismatch" if (row["sham_max_shift_l2"] > 0 if sham_audit
                           else row["norm_max_relative_error"] > 0.05) else "pass"
        )
        primary_dose = _is_primary_dose(
            {"intervention": key[4], "dose": key[5]}, primary_restoration_dose, primary_suppression_dose
        )
        row["dose_role"] = "primary" if primary_dose else "sensitivity"
        base_family = "specificity_contrast" if key[6] == "same_layer_specificity" else "layer_profile"
        row["fdr_family"] = base_family if primary_dose else base_family + "_sensitivity"
        output.append(row)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        families[row["fdr_family"]].append(row)
    for family in families.values():
        for row, q_value in zip(family, bh_fdr([float(item["p_value"]) for item in family])):
            row["fdr_q_value"] = q_value
    return output


def evaluate_causal_gate(
    summaries: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    minimum_sign_consistency: float,
    primary_restoration_dose: float = 1.0,
    primary_suppression_dose: float = 1.0,
    minimum_valid_pairs: int = MINIMUM_CONFIRMATORY_PAIRS,
) -> dict[str, Any]:
    def utility_supported(row: Mapping[str, Any]) -> bool:
        return (
            row.get("utility_inference_status") == "ok"
            and row.get("utility_fdr_family") == (
                "F4_reverse" if row["intervention"] == "reverse_suppression"
                else "F1_fixed_utility"
            )
            and int(row.get("utility_pair_count", row["pair_count"])) >= minimum_valid_pairs
            and row.get("utility_fdr_q_value") is not None
            and float(row["utility_effect_mean"]) > 0
            and float(row["utility_ci_low"]) > 0
            and float(row["utility_fdr_q_value"]) <= alpha
            and float(row["sign_consistency"]) >= minimum_sign_consistency
        )

    def refusal_confirmed(row: Mapping[str, Any]) -> bool:
        fields = ("refusal_effect_mean", "refusal_ci_low", "refusal_fdr_q_value")
        if (
            row.get("refusal_inference_status") != "ok"
            or row.get("refusal_fdr_family") != (
                "F4_reverse" if row["intervention"] == "reverse_suppression"
                else "F2_candidate_refusal"
            )
            or int(row.get("refusal_pair_count") or 0) < minimum_valid_pairs
            or row.get("refusal_pair_count") != row["pair_count"]
            or any(row.get(field) is None for field in fields)
        ):
            return False
        return (
            float(row["refusal_effect_mean"]) > 0
            and float(row["refusal_ci_low"]) > 0
            and float(row["refusal_fdr_q_value"]) <= alpha
        )

    def region(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "layer": int(row["layer"]),
            "state_key": row["state_key"],
            "intervention": row["intervention"],
            "dose": float(row["dose"]),
            "token_scope": row["token_scope"],
        }

    primary = [
        row for row in summaries
        if row["intervention"] in RESTORATION_INTERVENTIONS
        and row["token_scope"] == "audio"
        and _is_primary_dose(row, primary_restoration_dose, primary_suppression_dose)
        and utility_supported(row)
    ]
    # Binary evidence is joined on layer/state/intervention/dose/token scope
    # before this gate; utility improvement alone cannot confirm refusal.
    refusal_restoration = [row for row in primary if refusal_confirmed(row)]
    restoration_specificity_regions = []
    for row in refusal_restoration:
        prefix = "subspace" if row["intervention"] == "subspace_restoration" else "r"
        required = {
            f"{prefix}_vs_h", f"{prefix}_vs_random", f"{prefix}_vs_sham",
            "audio_vs_position",
        }
        names = {
            str(item["contrast"])
            for item in contrasts
            if int(item["layer"]) == int(row["layer"])
            and str(item["state_key"]) == str(row["state_key"])
            and item.get("contrast_scope") == "same_layer_specificity"
            and int(item.get("control_layer", -1)) == int(row["layer"])
            and item.get("primary_intervention") == row["intervention"]
            and item.get("token_scope") == row["token_scope"]
            and math.isclose(float(item.get("dose", float("nan"))), float(row["dose"]), rel_tol=0.0, abs_tol=1e-12)
            and item.get("formal_inference_status") == "ok"
            and item.get("norm_audit_status") == "pass"
            and item.get("fdr_family") == "F3_same_layer_specificity"
            and int(item.get("pair_count", 0)) >= minimum_valid_pairs
            and item.get("ci_low") is not None
            and float(item["ci_low"]) > 0
            and item.get("fdr_q_value") is not None
            and float(item["contrast_effect_mean"]) > 0
            and float(item["fdr_q_value"]) <= alpha
        }
        if required.issubset(names):
            restoration_specificity_regions.append(region(row))
    reverse = [
        row for row in summaries
        if row["intervention"] == "reverse_suppression"
        and row["token_scope"] == "audio"
        and row["state_key"] == "clean"
        and _is_primary_dose(row, primary_restoration_dose, primary_suppression_dose)
        and utility_supported(row)
        and refusal_confirmed(row)
    ]
    reverse_required = {"reverse_r_vs_h", "reverse_r_vs_random", "reverse_r_vs_sham"}
    reverse_layers = set()
    for row in reverse:
        names = {
            str(item["contrast"])
            for item in contrasts
            if int(item["layer"]) == int(row["layer"])
            and str(item["state_key"]) == "clean"
            and item.get("contrast_scope") == "same_layer_specificity"
            and int(item.get("control_layer", -1)) == int(row["layer"])
            and item.get("primary_intervention") == row["intervention"]
            and item.get("token_scope") == row["token_scope"]
            and math.isclose(float(item.get("dose", float("nan"))), float(row["dose"]), rel_tol=0.0, abs_tol=1e-12)
            and item.get("formal_inference_status") == "ok"
            and item.get("norm_audit_status") == "pass"
            and item.get("fdr_family") == "F4_reverse"
            and int(item.get("pair_count", 0)) >= minimum_valid_pairs
            and item.get("ci_low") is not None
            and float(item["ci_low"]) > 0
            and item.get("fdr_q_value") is not None
            and float(item["contrast_effect_mean"]) > 0
            and float(item["fdr_q_value"]) <= alpha
        }
        if reverse_required.issubset(names):
            reverse_layers.add(int(row["layer"]))
    bidirectional_layers = sorted(
        {item["layer"] for item in restoration_specificity_regions}.intersection(reverse_layers)
    )
    supported_regions = [
        item for item in restoration_specificity_regions
        if item["layer"] in bidirectional_layers
    ]
    all_required = {
        "restore": ["*_vs_h", "*_vs_random", "*_vs_sham", "audio_vs_position"],
        "suppress": sorted(reverse_required),
    }
    return {
        "behavior_endpoint_version": BEHAVIOR_ENDPOINT_VERSION,
        "layer_dose_version": LAYER_DOSE_VERSION,
        "formal_statistics_version": FORMAL_STATISTICS_VERSION,
        "minimum_valid_pairs": minimum_valid_pairs,
        "layer_profiles_used_as_mechanism_gate": False,
        "measured_equal_norm_required": True,
        "maximum_shift_norm_relative_error": 0.05,
        "primary_restoration_dose": primary_restoration_dose,
        "primary_suppression_dose": primary_suppression_dose,
        "supported": bool(supported_regions and bidirectional_layers),
        "utility_reduction_supported": bool(primary),
        "refusal_restoration_supported": bool(refusal_restoration),
        "reverse_refusal_suppression_supported": bool(reverse),
        "utility_reduction_regions": [region(row) for row in primary],
        "refusal_restoration_regions": [region(row) for row in refusal_restoration],
        "reverse_refusal_suppression_regions": [region(row) for row in reverse],
        "alpha": float(alpha),
        "minimum_sign_consistency": float(minimum_sign_consistency),
        "binary_confirmation_required": True,
        "binary_confirmation_criteria": [
            "same layer/state/intervention/dose/token scope and paired population",
            "positive oriented refusal effect",
            "pair-bootstrap CI lower bound > 0",
            "exact McNemar BH-FDR q <= alpha",
        ],
        "restoration_specificity_regions": restoration_specificity_regions,
        "supported_regions": supported_regions,
        "bidirectional_layers": bidirectional_layers,
        "required_specificity_contrasts": all_required,
    }


def causal_prior(
    summaries: Sequence[Mapping[str, Any]],
    *,
    gate: Optional[Mapping[str, Any]] = None,
    protocol_lock_sha256: Optional[str] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    primary_restoration_dose: float = 1.0,
) -> dict[str, Any]:
    if protocol is not None:
        primary_restoration_dose = float(protocol["primary_restoration_dose"])
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in summaries:
        if (
            row["intervention"] in RESTORATION_INTERVENTIONS and row["token_scope"] == "audio"
            and _is_primary_dose(row, primary_restoration_dose)
            and row.get("utility_inference_status", "ok") == "ok"
        ):
            by_layer[int(row["layer"])].append(float(row["causal_effect_mean"]))
    allowed = None if gate is None else set(int(value) for value in gate.get("bidirectional_layers", []))
    scores = {
        layer: max(0.0, float(np.mean(values)))
        for layer, values in by_layer.items()
        if allowed is None or layer in allowed
    }
    total = sum(scores.values())
    weights = {
        str(layer): (score / total if total > 0 else 1.0 / len(scores))
        for layer, score in scores.items()
    } if scores else {}
    return {
        "format": "rq2-causal-layer-prior",
        "version": 2,
        "layer_dose_version": LAYER_DOSE_VERSION,
        "behavior_endpoint_version": BEHAVIOR_ENDPOINT_VERSION,
        "outcome": "strongreject_score",
        "effect": "utility_effect",
        "confirmation_outcome": "refusal_label",
        "status": "available" if gate is None or gate.get("supported") is True else "withheld",
        "gate": None if gate is None else dict(gate),
        "protocol_lock_sha256": protocol_lock_sha256,
        "primary_intervention": None if protocol is None else protocol.get("primary_intervention"),
        "lambda_restore": None if protocol is None else protocol.get("primary_restoration_dose"),
        "eta_suppress": None if protocol is None else protocol.get("primary_suppression_dose"),
        "state_bins": [],
        "severity_proxy": {
            "status": "not_provided",
            "reason": "fixed PGD states are not mild/moderate/severe degradation bins",
        },
        "fixed_state_keys": sorted({
            str(row["state_key"])
            for row in summaries
            if row["intervention"] in RESTORATION_INTERVENTIONS
            and row["token_scope"] == "audio"
            and _is_primary_dose(row, primary_restoration_dose)
        }),
        "supported_regions": [] if gate is None else list(gate.get("supported_regions", [])),
        "weights": weights,
        "ranked_layers": sorted(scores, key=lambda layer: scores[layer], reverse=True),
    }


def paired_event_offset_tests(
    effect_rows: Sequence[Mapping[str, Any]],
    trial_records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: Sequence[str],
    candidate_layers: Sequence[int],
    primary_intervention: str,
    restoration_dose: float,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Complete-case, within-pair post-minus-pre event contrasts."""
    offsets = (-2, -1, 0, 1, 3)
    expected = set(map(str, expected_pair_ids))
    populations: dict[str, set[str]] = {name: set() for name in EVENT_NAMES}
    by_offset: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    for trial in trial_records:
        pair_id = str(trial["pair_id"])
        state = str(trial.get("state_key", ""))
        if pair_id not in expected:
            continue
        for name in EVENT_NAMES:
            if state.startswith(f"event:{name}:"):
                populations[name].add(pair_id)
    for row in effect_rows:
        if (
            str(row["pair_id"]) not in expected
            or row["intervention"] != primary_intervention
            or row["token_scope"] != "audio"
            or not math.isclose(float(row["dose"]), restoration_dose, rel_tol=0.0, abs_tol=1e-12)
        ):
            continue
        state = str(row["state_key"])
        for name in EVENT_NAMES:
            prefix = f"event:{name}:"
            if state.startswith(prefix):
                offset = int(state[len(prefix):])
                if offset in offsets and int(row["layer"]) in candidate_layers:
                    by_offset[(name, int(row["layer"]), str(row["pair_id"]), offset)].append(
                        float(row["utility_effect"])
                    )
                break
    tests: list[dict[str, Any]] = []
    family_counts: dict[str, dict[str, int]] = {}
    for name in EVENT_NAMES:
        family = f"E1_paired_offset_{name}"
        population = sorted(populations[name])
        family_rows: list[dict[str, Any]] = []
        for layer in candidate_layers:
            pair_differences: list[dict[str, Any]] = []
            excluded: dict[str, str] = {}
            for pair_id in population:
                values = {
                    offset: by_offset.get((name, int(layer), pair_id, offset), [])
                    for offset in offsets
                }
                missing = [offset for offset, items in values.items() if not items]
                if missing:
                    excluded[pair_id] = "missing_or_unlabeled_offsets:" + ",".join(
                        f"{offset:+d}" for offset in missing
                    )
                    continue
                pre = float(np.mean([np.mean(values[offset]) for offset in (-2, -1)]))
                post = float(np.mean([np.mean(values[offset]) for offset in (0, 1, 3)]))
                pair_differences.append({"pair_id": pair_id, "difference": post - pre})
            valid_count = len(pair_differences)
            estimate = (
                pair_cluster_bootstrap(
                    pair_differences, value_field="difference", replicates=replicates,
                    confidence=confidence, seed=seed + len(tests),
                )
                if valid_count >= 2 else None
            )
            p_value = (
                _one_sample_p([item["difference"] for item in pair_differences])
                if valid_count >= MINIMUM_CONFIRMATORY_PAIRS else None
            )
            status = (
                "no_eligible_events" if not population else
                "insufficient_data" if valid_count < MINIMUM_CONFIRMATORY_PAIRS else
                "inference_unavailable" if p_value is None or not math.isfinite(p_value) else "ok"
            )
            family_rows.append({
                "family": family, "event": name, "layer": int(layer),
                "test_id": f"{family}:L{layer}", "pre_offsets": "-2,-1",
                "post_offsets": "0,+1,+3",
                "raw_formal_pair_count": len(expected),
                "event_population_count": len(population),
                "valid_pair_count": valid_count,
                "excluded_pair_count": len(population) - valid_count,
                "excluded_pair_fraction": (
                    (len(population) - valid_count) / len(population) if population else None
                ),
                "excluded_pair_reasons": json.dumps(excluded, sort_keys=True),
                "valid_pair_ids": json.dumps([item["pair_id"] for item in pair_differences]),
                "minimum_valid_pairs": MINIMUM_CONFIRMATORY_PAIRS,
                "paired_post_minus_pre_mean": None if estimate is None else estimate["mean"],
                "ci_low": None if estimate is None else estimate["ci_low"],
                "ci_high": None if estimate is None else estimate["ci_high"],
                "p_value": p_value, "fdr_q_value": None,
                "inference_status": status,
            })
        q_values = bh_fdr([
            row["p_value"] if row["inference_status"] == "ok" else 1.0
            for row in family_rows
        ])
        for row, q_value in zip(family_rows, q_values):
            if row["inference_status"] == "ok":
                row["fdr_q_value"] = q_value
        family_counts[family] = {
            status: sum(row["inference_status"] == status for row in family_rows)
            for status in sorted({row["inference_status"] for row in family_rows})
        }
        tests.extend(family_rows)
    return tests, {
        "formal_statistics_version": FORMAL_STATISTICS_VERSION,
        "scope": "same-event complete-case paired offset contrast",
        "required_offsets": list(offsets),
        "pre_offsets": [-2, -1], "post_offsets": [0, 1, 3],
        "minimum_valid_pairs": MINIMUM_CONFIRMATORY_PAIRS,
        "event_population_counts": {name: len(populations[name]) for name in EVENT_NAMES},
        "planned_test_counts": {f"E1_paired_offset_{name}": len(candidate_layers) for name in EVENT_NAMES},
        "family_status_counts": family_counts,
    }


def analyze_trial_set(
    trials_path: str | Path,
    labels_path: str | Path,
    *,
    output_dir: str | Path,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    rq1_slopes_path: Optional[str | Path] = None,
    rq2_trajectory_trials_path: Optional[str | Path] = None,
    top_k: int = 5,
    candidate_layers: Sequence[int] = (),
    depth_control_layers: Sequence[int] = (),
    neighbor_layers: Sequence[int] = (),
    distant_control_layers: Sequence[int] = (),
    primary_restoration_dose: float = 1.0,
    primary_suppression_dose: float = 1.0,
    fdr_alpha: float = 0.05,
    minimum_sign_consistency: float = 0.6,
    protocol_lock_sha256: Optional[str] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    event: bool = False,
    pilot: bool = False,
    event_pair_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    event_pilot = pilot and event
    formal_event = event and not pilot
    if event_pilot and (not event_pair_ids or len(set(event_pair_ids)) != len(event_pair_ids)):
        raise RQ2AnalysisError("event pilot requires its explicit unique event population")
    if event_pair_ids is not None and not event_pilot:
        raise RQ2AnalysisError("event_pair_ids is only valid for a dev event pilot")
    if not pilot and protocol is None:
        raise RQ2AnalysisError("formal or event analysis requires a locked protocol")
    if distant_control_layers and depth_control_layers:
        raise RQ2AnalysisError("use distant_control_layers or its v1 alias, not both")
    distant_control_layers = tuple(distant_control_layers or depth_control_layers)
    dose_kwargs = {
        "primary_restoration_dose": primary_restoration_dose,
        "primary_suppression_dose": primary_suppression_dose,
    }
    if protocol is not None and any(
        not math.isclose(float(protocol[key]), value, rel_tol=0.0, abs_tol=1e-12)
        for key, value in dose_kwargs.items()
    ):
        raise RQ2AnalysisError("analysis primary doses must match the protocol lock")
    formal_family_policy: dict[str, Any] = {}
    if not pilot:
        if protocol.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION:
            raise RQ2AnalysisError("protocol lock lacks the registered formal family policy")
        if (
            list(candidate_layers) != list(protocol["candidate_layers"])
            or list(neighbor_layers) != list(protocol["neighbor_layers"])
            or list(distant_control_layers) != list(protocol["distant_control_layers"])
        ):
            raise RQ2AnalysisError("analysis layer roles differ from the protocol lock")
        formal_family_policy = family_policy(
            formal_layers=protocol["formal_layers"],
            candidate_layers=candidate_layers,
            neighbor_layers=neighbor_layers,
            distant_layers=distant_control_layers,
            fixed_steps=protocol["fixed_steps"],
        )
        if protocol.get("formal_statistics_policy") != formal_family_policy:
            raise RQ2AnalysisError("formal family slots differ from the protocol lock")
        if len(protocol.get("formal_pair_ids", ())) != 40:
            raise RQ2AnalysisError("formal protocol must identify exactly 40 frozen causal-test pairs")
    trial_records = read_jsonl(trials_path)
    label_records = read_jsonl(labels_path)
    if event_pilot:
        allowed_states = {"clean", *(f"event:first_non_refusal_step:{offset:+d}" for offset in (-2,-1,0,1,3))}
        if any(row.get("rq2_role") != "rq2_dev" or row.get("pair_id") not in event_pair_ids
               or row.get("state_key") not in allowed_states for row in trial_records):
            raise RQ2AnalysisError("event pilot contains an unexpected role, pair or state")
    rows, counts = join_effect_rows(trial_records, label_records)
    for row in rows:
        row["dose_role"] = (
            "primary" if _is_primary_dose(row, primary_restoration_dose, primary_suppression_dose)
            else "sensitivity"
        )
        row["layer_role"] = (
            "candidate" if row["layer"] in candidate_layers else
            "neighbor" if row["layer"] in neighbor_layers else
            "distant" if row["layer"] in distant_control_layers else "other_scan_layer"
        )
    endpoint_metadata = {
        "analysis_scope": "dev_event_pilot" if event_pilot else "dev_pilot" if pilot else ("event" if event else "formal"),
        "event_pair_ids": list(event_pair_ids) if event_pilot else None,
        "causal_evidence": False if pilot else None,
        "pilot_inference_status": "diagnostic_only" if pilot else None,
        "formal_statistics_version": FORMAL_STATISTICS_VERSION if not pilot else None,
        "layer_dose_version": LAYER_DOSE_VERSION,
        **dose_kwargs,
        "layer_roles": {
            "candidate_layers": list(candidate_layers),
            "neighbor_layers": list(neighbor_layers),
            "distant_control_layers": list(distant_control_layers),
        },
        "layer_profiles_used_as_mechanism_gate": False,
        "behavior_endpoint_version": BEHAVIOR_ENDPOINT_VERSION,
        "primary_outcome": "strongreject_score",
        "primary_effect": "utility_effect",
        "confirmation_outcome": "refusal_label",
        "secondary_outcome": "compliance_label",
        "effect_definitions": EFFECT_DEFINITIONS,
        "legacy_aliases": {
            "causal_effect": "utility_effect",
            "ce_restore": "restoration utility_effect",
            "ce_suppress": "suppression utility_effect",
            "refusal_orientation": "1 - strongreject_score; not a refusal probability",
            "behavior_recovery_rate": "normalized_utility_effect; not a fraction of refusals restored",
        },
        "replicate_aggregation": "within-pair-mean",
        "bootstrap_unit": "pair_id",
        "bootstrap_replicates": replicates,
        "confidence": confidence,
        "binary_test": "exact McNemar; unavailable for fractional replicate means",
        "binary_fdr_unavailable_policy": "p=1 correction slot; reported p/q remain missing",
        "binary_fdr_families": (
            ["paired_binary", "paired_binary_sensitivity"] if pilot else
            [] if event else ["F2_candidate_refusal", "F4_reverse"]
        ),
    }
    if not rows and formal_event:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        event_tests, event_manifest = apply_event_families(
            [], [], trial_records, label_records,
            expected_pair_ids=protocol["formal_pair_ids"],
            candidate_layers=candidate_layers,
            event_offsets=protocol["event_offsets"],
            primary_intervention=protocol["primary_intervention"],
            restoration_dose=primary_restoration_dose,
        )
        atomic_csv(output / "rq2_event_formal_tests.csv", event_tests)
        atomic_json(output / "rq2_event_family_manifest.json", event_manifest)
        paired_tests, paired_manifest = paired_event_offset_tests(
            [], trial_records, expected_pair_ids=protocol["formal_pair_ids"],
            candidate_layers=candidate_layers,
            primary_intervention=protocol["primary_intervention"],
            restoration_dose=primary_restoration_dose,
            replicates=replicates, confidence=confidence, seed=seed + 20_000,
        )
        atomic_csv(output / "rq2_event_paired_offsets.csv", paired_tests)
        atomic_json(output / "rq2_event_paired_offset_manifest.json", paired_manifest)
        atomic_csv(
            output / "rq2_event_causal_profiles.csv",
            [],
            fieldnames=(
                "layer", "state_key", "intervention", "dose", "token_scope",
                "causal_effect_mean", "ci_low", "ci_high", "pair_count",
                "fdr_q_value",
                *(
                    f"{endpoint}_{field}"
                    for endpoint in ("utility", "refusal", "compliance")
                    for field in ("effect_mean", "ci_low", "ci_high", "pair_count", "fdr_q_value")
                ),
            ),
        )
        atomic_csv(
            output / "rq2_paired_binary_tests.csv",
            [],
            fieldnames=(
                "layer", "state_key", "intervention", "dose", "token_scope",
                "outcome", "paired_effect_mean",
                "baseline_false_to_intervention_true", "baseline_true_to_intervention_false",
                "discordant_count", "pair_count", "binary_pair_count",
                "replicate_aggregation", "inference_status",
                "mcnemar_exact_p_value", "fdr_family", "fdr_q_value",
            ),
        )
        summary = {
            "format": "rq2-causal-analysis-summary",
            "version": 2,
            **endpoint_metadata,
            "event_centered": True,
            **counts,
            "group_count": 0,
            "inference_status": "no_eligible_events",
            "fdr_alpha": float(fdr_alpha),
            "formal_family_manifest": event_manifest,
            "formal_test_count": len(event_tests),
            "event_paired_offset_manifest": paired_manifest,
            "event_paired_offset_test_count": len(paired_tests),
        }
        atomic_json(output / "event_analysis.json", summary)
        return summary
    if not rows and pilot:
        raise RQ2AnalysisError("no fully labeled paired pilot trials are available")
    binary = paired_binary_tests(rows, **dose_kwargs) if rows else []
    summaries = summarize_effects(
        rows, replicates=replicates, confidence=confidence, seed=seed,
        binary_tests=binary, **dose_kwargs,
    ) if rows else []
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        output / "rq2_interventions.csv", rows,
        fieldnames=("pair_id", "layer", "state_key", "intervention", "dose",
                    "token_scope", "utility_effect", "refusal_effect", "compliance_effect"),
    )
    formal_tests: list[dict[str, Any]] = []
    formal_manifest: dict[str, Any] = {}
    if formal_event:
        formal_tests, formal_manifest = apply_event_families(
            summaries, rows, trial_records, label_records,
            expected_pair_ids=protocol["formal_pair_ids"],
            candidate_layers=candidate_layers,
            event_offsets=protocol["event_offsets"],
            primary_intervention=protocol["primary_intervention"],
            restoration_dose=primary_restoration_dose,
        )
        for item in binary:
            item["fdr_family"] = "unregistered_exploratory"
            item["fdr_q_value"] = None
        atomic_csv(output / "rq2_event_formal_tests.csv", formal_tests)
        atomic_json(output / "rq2_event_family_manifest.json", formal_manifest)
        paired_tests, paired_manifest = paired_event_offset_tests(
            rows, trial_records, expected_pair_ids=protocol["formal_pair_ids"],
            candidate_layers=candidate_layers,
            primary_intervention=protocol["primary_intervention"],
            restoration_dose=primary_restoration_dose,
            replicates=replicates, confidence=confidence, seed=seed + 20_000,
        )
        atomic_csv(output / "rq2_event_paired_offsets.csv", paired_tests)
        atomic_json(output / "rq2_event_paired_offset_manifest.json", paired_manifest)
    else:
        controls = paired_specificity_contrasts(
            rows,
            candidate_layers=candidate_layers,
            neighbor_layers=neighbor_layers,
            distant_control_layers=distant_control_layers,
            **dose_kwargs,
            replicates=replicates,
            confidence=confidence,
            seed=seed + 10_000,
        )
        if not pilot:
            formal_tests, formal_manifest = apply_formal_families(
                summaries, binary, controls, rows, trial_records, label_records,
                expected_pair_ids=protocol["formal_pair_ids"],
                primary_intervention=protocol["primary_intervention"],
                restoration_dose=primary_restoration_dose,
                suppression_dose=primary_suppression_dose,
                formal_layers=protocol["formal_layers"],
                candidate_layers=candidate_layers,
                neighbor_layers=neighbor_layers,
                distant_layers=distant_control_layers,
                fixed_steps=protocol["fixed_steps"],
            )
            atomic_csv(output / "rq2_formal_tests.csv", formal_tests)
            atomic_json(output / "rq2_formal_family_manifest.json", formal_manifest)
        contrast_fields = (
            "contrast", "layer", "state_key", "control_layer", "control_layer_role",
            "primary_intervention", "dose", "dose_role", "token_scope", "contrast_scope",
            "contrast_effect_mean", "ci_low", "ci_high", "pair_count",
            "p_value", "fdr_family", "fdr_q_value",
        )
        for filename, scope in (
            ("rq2_specificity_controls.csv", "same_layer_specificity"),
            ("rq2_layer_profile_contrasts.csv", "layer_profile"),
        ):
            atomic_csv(
                output / filename, [row for row in controls if row["contrast_scope"] == scope],
                fieldnames=contrast_fields,
            )
        bidirectional = [row for row in summaries if row["intervention"] in {
            "r_direction", "subspace_restoration", "reverse_suppression"
        }]
        atomic_csv(
            output / "rq2_bidirectional_intervention.csv",
            bidirectional,
            fieldnames=tuple(summaries[0]) if summaries else (
                "layer", "state_key", "intervention", "dose", "token_scope",
                "utility_effect_mean", "refusal_effect_mean", "pair_count",
            ),
        )
    atomic_csv(
        output / "rq2_paired_binary_tests.csv", binary,
        fieldnames=("layer", "state_key", "intervention", "dose", "token_scope",
                    "outcome", "pair_count", "mcnemar_exact_p_value",
                    "fdr_family", "fdr_q_value", "formal_inference_status"),
    )
    atomic_csv(
        output / "rq2_dose_sensitivity.csv",
        [row for row in summaries if row["dose_role"] == "sensitivity"],
        fieldnames=tuple(summaries[0]) if summaries else (
            "layer", "state_key", "intervention", "dose", "token_scope",
            "utility_effect_mean", "pair_count", "fdr_family", "fdr_q_value",
        ),
    )
    if formal_event:
        atomic_csv(output / "rq2_event_causal_profiles.csv", summaries)
    else:
        atomic_csv(
            output / "rq2_causal_map_mean_ci.csv", summaries,
            fieldnames=tuple(summaries[0]) if summaries else (
                "layer", "state_key", "intervention", "dose", "token_scope",
                "utility_effect_mean", "utility_ci_low", "utility_ci_high",
                "pair_count", "utility_pair_count", "fdr_family", "fdr_q_value",
                "formal_inference_status",
            ),
        )

    comparison: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    if rq1_slopes_path is not None and not event:
        comparison_rows, comparison = compare_degradation_to_causality(
            rq1_slopes_path,
            summaries,
            top_k=top_k,
            candidate_layers=candidate_layers,
            primary_restoration_dose=primary_restoration_dose,
            rq2_trajectory_trials_path=rq2_trajectory_trials_path,
        )
    if not event:
        atomic_csv(
            output / "rq2_degradation_vs_causality.csv",
            comparison_rows,
            fieldnames=(
                "layer", "rq1_r_drop", "rq2_causal_effect", "shared_layer_count",
                "spearman_rho", "spearman_p_value", "top_k", "rq1_top_layers",
                "rq2_top_layers", "top_k_overlap", "top_k_overlap_layers",
                "candidate_precision",
            ),
        )
    complete_f1_grid = (
        pilot or event or all(
            slot["inference_status"] == "ok"
            for slot in formal_tests if slot["family"] == "F1_fixed_utility"
        )
    )
    model = (
        {"inference_status": "not_applicable_to_dev_pilot"} if pilot
        else {"inference_status": "paired_event_offset_tests_reported_separately"} if event
        else (
            fit_layer_state_model(
                [row for row in rows if row["intervention"] == protocol["primary_intervention"]],
                primary_restoration_dose=primary_restoration_dose,
            )
            if complete_f1_grid else {
                "inference_status": "insufficient_data",
                "reason": "incomplete registered F1 layer-by-state grid",
            }
        )
    )
    gate = None if event or pilot else evaluate_causal_gate(
        summaries,
        controls,
        alpha=fdr_alpha,
        minimum_sign_consistency=minimum_sign_consistency,
        **dose_kwargs,
    )
    prior = None if pilot or event else causal_prior(
        summaries,
        gate=gate,
        protocol_lock_sha256=protocol_lock_sha256,
        protocol=protocol,
        primary_restoration_dose=primary_restoration_dose,
    )
    if not event and not pilot:
        atomic_json(output / "rq2_causal_prior.json", prior)
    summary = {
        "format": "rq2-causal-analysis-summary",
        "version": 2,
        **endpoint_metadata,
        "event_centered": event,
        **counts,
        "group_count": len(summaries),
        "fdr_alpha": float(fdr_alpha),
        "random_intercept_model": model,
        "rq1_vs_rq2": comparison,
        "formal_family_manifest": formal_manifest if not pilot else None,
        "formal_test_count": len(formal_tests),
        "event_paired_offset_manifest": paired_manifest if formal_event else None,
        "event_paired_offset_test_count": len(paired_tests) if formal_event else 0,
        "causal_claim_gate": gate,
        "causal_prior": prior,
    }
    atomic_json(output / ("event_analysis.json" if formal_event else "rq2_summary.json"), summary)
    return summary


__all__ = [
    "BEHAVIOR_ENDPOINT_VERSION",
    "EFFECT_DEFINITIONS",
    "RQ2AnalysisError",
    "analyze_trial_set",
    "bh_fdr",
    "causal_prior",
    "compare_degradation_to_causality",
    "fit_layer_state_model",
    "join_effect_rows",
    "pair_cluster_bootstrap",
    "paired_binary_tests",
    "paired_event_offset_tests",
    "paired_specificity_contrasts",
    "evaluate_causal_gate",
    "summarize_effects",
]
