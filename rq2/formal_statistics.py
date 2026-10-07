"""Frozen formal RQ2 test families, paired populations, and missingness audit."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


FORMAL_STATISTICS_VERSION = 3
MINIMUM_CONFIRMATORY_PAIRS = 20
EVENT_NAMES = (
    "first_refusal_weakening_step",
    "first_non_refusal_step",
    "first_compliance_step",
)


class RQ2FormalStatisticsError(ValueError):
    """Raised when the formal population or registered test grid is inconsistent."""


def family_policy(
    *, formal_layers: Sequence[int], candidate_layers: Sequence[int],
    neighbor_layers: Sequence[int], distant_layers: Sequence[int],
    fixed_steps: Sequence[int],
) -> dict[str, Any]:
    local = sum(
        abs(candidate - other) == 1
        for candidate in candidate_layers
        for other in set(candidate_layers).union(neighbor_layers)
        if candidate != other
    )
    states = len(fixed_steps)
    return {
        "formal_statistics_version": FORMAL_STATISTICS_VERSION,
        "minimum_confirmatory_pairs": MINIMUM_CONFIRMATORY_PAIRS,
        "fdr_families": {
            "F1_fixed_utility": len(formal_layers) * states,
            "F2_candidate_refusal": len(candidate_layers) * states,
            "F3_same_layer_specificity": len(candidate_layers) * states * 4,
            "F4_reverse": len(candidate_layers) * 5,
            "F5_neighbor_profile": local * states,
            "F5_distant_profile": len(candidate_layers) * len(distant_layers) * states,
            **{
                f"F5_event_{name}": len(candidate_layers) * 5
                for name in EVENT_NAMES
            },
        },
        "incomplete_test_correction": "p=1 slot; reported q remains missing",
        "F4_components": "reverse utility, explicit refusal, and H/random/sham contrasts",
        "F5_status": "descriptive; never a mechanism specificity gate",
        "event_paired_offset_families": {
            f"E1_paired_offset_{name}": len(candidate_layers) for name in EVENT_NAMES
        },
        "event_paired_offset_contrast": {
            "required_offsets": [-2, -1, 0, 1, 3],
            "pre_offsets": [-2, -1], "post_offsets": [0, 1, 3],
            "effect": "within-pair mean(post) minus mean(pre)",
            "minimum_valid_pairs": MINIMUM_CONFIRMATORY_PAIRS,
            "test": "two-sided one-sample paired-difference t-test",
            "correction": "BH by event type across all candidate layers; incomplete slots use p=1",
        },
    }


def _bh(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        index = order[rank]
        running = min(running, min(1.0, values[index] * len(values) / (rank + 1)))
        adjusted[index] = running
    return adjusted


def _value(row: Mapping[str, Any] | None, field: str) -> float | None:
    if row is None:
        return None
    try:
        value = float(row[field])
    except (KeyError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _key(layer: int, state: str, intervention: str, dose: float, scope: str = "audio") -> tuple[Any, ...]:
    return (int(layer), str(state), str(intervention), float(dose), str(scope))


def _paired_population(
    *,
    primary: tuple[Any, ...], comparator: tuple[Any, ...] | None,
    expected_ids: Sequence[str],
    valid: Mapping[tuple[Any, ...], set[str]],
    trials: Mapping[tuple[Any, ...], list[Mapping[str, Any]]],
    labels: Mapping[str, Mapping[str, Any]],
) -> tuple[int, dict[str, int], dict[str, str]]:
    def arm_reason(pair_id: str, arm: tuple[Any, ...]) -> str:
        records = trials.get((pair_id, *arm), ())
        if not records:
            return "missing_trial"
        reasons = []
        for trial in records:
            label = labels.get(str(trial["trial_id"]))
            baseline = labels.get(str(trial.get("baseline_trial_id")))
            if label is None or baseline is None:
                reasons.append("missing_label")
            elif label.get("label_status") == "unknown":
                reasons.append("unknown_intervention_label")
            elif baseline.get("label_status") == "unknown":
                reasons.append("unknown_baseline_label")
            else:
                reasons.append("not_analyzed")
        return reasons[0]

    valid_ids = valid.get(primary, set()).copy()
    if comparator is not None:
        valid_ids.intersection_update(valid.get(comparator, set()))
    valid_ids.intersection_update(expected_ids)
    reasons: dict[str, str] = {}
    for pair_id in expected_ids:
        if pair_id in valid_ids:
            continue
        reason = arm_reason(pair_id, primary)
        if comparator is not None and pair_id in valid.get(primary, set()):
            reason = "control_" + arm_reason(pair_id, comparator)
        reasons[pair_id] = reason
    return len(valid_ids), dict(sorted(Counter(reasons.values()).items())), reasons


def apply_formal_families(
    summaries: list[dict[str, Any]],
    binary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    trial_records: Sequence[Mapping[str, Any]],
    label_records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: Sequence[str],
    primary_intervention: str,
    restoration_dose: float,
    suppression_dose: float,
    formal_layers: Sequence[int],
    candidate_layers: Sequence[int],
    neighbor_layers: Sequence[int],
    distant_layers: Sequence[int],
    fixed_steps: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply BH to fixed slots; missing or underpowered slots remain in the denominator."""
    expected = tuple(sorted(set(map(str, expected_pair_ids))))
    if len(expected) != len(expected_pair_ids) or not expected:
        raise RQ2FormalStatisticsError("formal expected pair IDs must be non-empty and unique")
    if primary_intervention not in {"r_direction", "subspace_restoration"}:
        raise RQ2FormalStatisticsError("unsupported locked formal primary intervention")
    policy = family_policy(
        formal_layers=formal_layers, candidate_layers=candidate_layers,
        neighbor_layers=neighbor_layers, distant_layers=distant_layers,
        fixed_steps=fixed_steps,
    )
    summary_by_key = {
        _key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"]): row
        for row in summaries
    }
    binary_by_key = {
        (*_key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"]), row["outcome"]): row
        for row in binary
    }
    contrast_by_key = {
        (row["contrast"], *_key(row["layer"], row["state_key"], row["primary_intervention"], row["dose"]), int(row["control_layer"])): row
        for row in contrasts
    }
    valid: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in effect_rows:
        valid[_key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"])].add(str(row["pair_id"]))
    trials: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in trial_records:
        if row.get("intervention") != "baseline":
            trials[(str(row["pair_id"]), *_key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"]))].append(row)
    labels = {str(row["trial_id"]): row for row in label_records}

    # Old broad-family q values must not survive into formal reports.
    for row in summaries:
        row.update(fdr_family="unregistered_exploratory", fdr_q_value=None,
                   utility_fdr_family="unregistered_exploratory", utility_fdr_q_value=None,
                   refusal_fdr_family="unregistered_exploratory", refusal_fdr_q_value=None,
                   compliance_fdr_family="unregistered_exploratory", compliance_fdr_q_value=None)
    for row in (*binary, *contrasts):
        row.update(fdr_family="unregistered_exploratory", fdr_q_value=None)

    slots: list[dict[str, Any]] = []
    sources: list[dict[str, Any] | None] = []
    def add(
        family: str, endpoint: str, layer: int, state: str,
        primary: tuple[Any, ...], source: dict[str, Any] | None,
        *, comparator: tuple[Any, ...] | None = None, contrast: str = "",
        minimum: int = MINIMUM_CONFIRMATORY_PAIRS,
    ) -> None:
        count, reasons, excluded = _paired_population(
            primary=primary, comparator=comparator, expected_ids=expected,
            valid=valid, trials=trials, labels=labels,
        )
        if source is not None and int(source["pair_count"]) != count:
            raise RQ2FormalStatisticsError(
                f"paired population count disagrees for {family} L{layer} {state} {endpoint}"
            )
        p_field = "mcnemar_exact_p_value" if endpoint == "refusal" else "p_value"
        p_value = _value(source, p_field)
        status = (
            "insufficient_data" if count < minimum else
            "inference_unavailable" if p_value is None else "ok"
        )
        slots.append({
            "family": family,
            "test_id": f"{family}:L{layer}:{state}:{contrast or endpoint}",
            "endpoint": endpoint,
            "layer": layer, "state_key": state,
            "primary_intervention": primary[2], "dose": primary[3],
            "contrast": contrast, "control_layer": comparator[0] if comparator else None,
            "raw_pair_count": len(expected), "valid_pair_count": count,
            "excluded_pair_count": len(expected) - count,
            "excluded_pair_fraction": (len(expected) - count) / len(expected),
            "exclusion_reasons": json.dumps(reasons, sort_keys=True),
            "excluded_pair_reasons": json.dumps(excluded, sort_keys=True),
            "minimum_valid_pairs": minimum,
            "inference_status": status,
            "p_value": p_value,
            "fdr_q_value": None,
        })
        sources.append(source)

    states = tuple(f"fixed:{step}" for step in fixed_steps)
    for layer in formal_layers:
        for state in states:
            primary = _key(layer, state, primary_intervention, restoration_dose)
            add("F1_fixed_utility", "utility", layer, state, primary, summary_by_key.get(primary))
    for layer in candidate_layers:
        for state in states:
            primary = _key(layer, state, primary_intervention, restoration_dose)
            add("F2_candidate_refusal", "refusal", layer, state, primary,
                binary_by_key.get((*primary, "refusal")))
            prefix = "subspace" if primary_intervention == "subspace_restoration" else "r"
            controls = (
                (f"{prefix}_vs_h", "subspace_h_control" if prefix == "subspace" else "h_direction_control", "audio"),
                (f"{prefix}_vs_random", "subspace_random_control" if prefix == "subspace" else "random_direction_control", "audio"),
                (f"{prefix}_vs_sham", "sham", "audio"),
                ("audio_vs_position", primary_intervention, "position_control"),
            )
            for name, intervention, scope in controls:
                comparator = _key(layer, state, intervention, restoration_dose, scope)
                add("F3_same_layer_specificity", "contrast", layer, state, primary,
                    contrast_by_key.get((name, *primary, layer)),
                    comparator=comparator, contrast=name)
    for layer in candidate_layers:
        state = "clean"
        primary = _key(layer, state, "reverse_suppression", suppression_dose)
        add("F4_reverse", "utility", layer, state, primary, summary_by_key.get(primary))
        add("F4_reverse", "refusal", layer, state, primary,
            binary_by_key.get((*primary, "refusal")))
        for name, intervention in (
            ("reverse_r_vs_h", "reverse_h_control"),
            ("reverse_r_vs_random", "reverse_random_control"),
            ("reverse_r_vs_sham", "reverse_sham"),
        ):
            comparator = _key(layer, state, intervention, suppression_dose)
            add("F4_reverse", "contrast", layer, state, primary,
                contrast_by_key.get((name, *primary, layer)),
                comparator=comparator, contrast=name)
    local_layers = set(candidate_layers).union(neighbor_layers)
    for layer in candidate_layers:
        for state in states:
            primary = _key(layer, state, primary_intervention, restoration_dose)
            for other in sorted(local_layers - {layer}):
                if abs(other - layer) == 1:
                    name = "candidate_vs_neighbor"
                    add("F5_neighbor_profile", "contrast", layer, state, primary,
                        contrast_by_key.get((name, *primary, other)),
                        comparator=_key(other, state, primary_intervention, restoration_dose),
                        contrast=f"{name}:L{other}", minimum=2)
            for other in distant_layers:
                name = "candidate_vs_distant"
                add("F5_distant_profile", "contrast", layer, state, primary,
                    contrast_by_key.get((name, *primary, other)),
                    comparator=_key(other, state, primary_intervention, restoration_dose),
                    contrast=f"{name}:L{other}", minimum=2)

    # A missing planned slot occupies p=1 during BH, without a reported q value.
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, slot in enumerate(slots):
        by_family[slot["family"]].append(index)
    for family, indices in by_family.items():
        adjusted = _bh([
            slots[i]["p_value"] if slots[i]["inference_status"] == "ok" else 1.0
            for i in indices
        ])
        for index, q_value in zip(indices, adjusted):
            slot, source = slots[index], sources[index]
            if slot["inference_status"] == "ok":
                slot["fdr_q_value"] = q_value
            if source is None:
                continue
            source["fdr_family"] = family
            source["fdr_q_value"] = slot["fdr_q_value"]
            source["formal_inference_status"] = slot["inference_status"]
            source["raw_pair_count"] = slot["raw_pair_count"]
            source["excluded_pair_count"] = slot["excluded_pair_count"]
            source["exclusion_reasons"] = slot["exclusion_reasons"]
            if slot["endpoint"] == "utility":
                source["utility_fdr_family"] = family
                source["utility_fdr_q_value"] = slot["fdr_q_value"]
                source["utility_inference_status"] = slot["inference_status"]
            elif slot["endpoint"] == "refusal":
                summary = summary_by_key.get(_key(slot["layer"], slot["state_key"], slot["primary_intervention"], slot["dose"]))
                if summary is not None:
                    summary["refusal_fdr_family"] = family
                    summary["refusal_fdr_q_value"] = slot["fdr_q_value"]
                    summary["refusal_inference_status"] = (
                        slot["inference_status"] if slot["inference_status"] != "ok" else source["inference_status"]
                    )
                    summary["refusal_exclusion_reasons"] = slot["exclusion_reasons"]

    actual_counts = {family: len(indices) for family, indices in by_family.items()}
    for family, planned in policy["fdr_families"].items():
        if family.startswith("F5_event_"):
            continue
        if actual_counts.get(family, 0) != planned:
            raise RQ2FormalStatisticsError(f"formal family {family} has an incomplete planned grid")
    policy.update({
        "raw_pair_count": len(expected),
        "raw_pair_ids": list(expected),
        "family_status_counts": {
            family: dict(sorted(Counter(slots[i]["inference_status"] for i in indices).items()))
            for family, indices in by_family.items()
        },
    })
    return slots, policy


def apply_event_families(
    summaries: list[dict[str, Any]],
    effect_rows: Sequence[Mapping[str, Any]],
    trial_records: Sequence[Mapping[str, Any]],
    label_records: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: Sequence[str],
    candidate_layers: Sequence[int],
    event_offsets: Sequence[int],
    primary_intervention: str,
    restoration_dose: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep event types in separate descriptive F5 correction families."""
    if tuple(event_offsets) != (-2, -1, 0, 1, 3):
        raise RQ2FormalStatisticsError("event offsets differ from the frozen protocol")
    expected = set(map(str, expected_pair_ids))
    summary_by_key = {
        _key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"]): row
        for row in summaries
    }
    valid: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in effect_rows:
        valid[_key(row["layer"], row["state_key"], row["intervention"], row["dose"], row["token_scope"])].add(str(row["pair_id"]))
    trials: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    event_populations: dict[str, set[str]] = {name: set() for name in EVENT_NAMES}
    for trial in trial_records:
        state = str(trial.get("state_key", ""))
        for name in EVENT_NAMES:
            if state.startswith(f"event:{name}:") and str(trial["pair_id"]) in expected:
                event_populations[name].add(str(trial["pair_id"]))
        if trial.get("intervention") != "baseline":
            trials[(str(trial["pair_id"]), *_key(trial["layer"], state, trial["intervention"], trial["dose"], trial["token_scope"]))].append(trial)
    labels = {str(row["trial_id"]): row for row in label_records}
    for row in summaries:
        row.update(fdr_family="unregistered_exploratory", fdr_q_value=None,
                   utility_fdr_family="unregistered_exploratory", utility_fdr_q_value=None)
    output: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for name in EVENT_NAMES:
        family = f"F5_event_{name}"
        population = tuple(sorted(event_populations[name]))
        family_slots: list[dict[str, Any]] = []
        family_sources: list[dict[str, Any] | None] = []
        for layer in candidate_layers:
            for offset in event_offsets:
                state = f"event:{name}:{offset:+d}"
                primary = _key(layer, state, primary_intervention, restoration_dose)
                source = summary_by_key.get(primary)
                count, reasons, excluded = _paired_population(
                    primary=primary, comparator=None, expected_ids=population,
                    valid=valid, trials=trials, labels=labels,
                )
                if source is not None and int(source["pair_count"]) != count:
                    raise RQ2FormalStatisticsError(f"event population count disagrees at {state} L{layer}")
                p_value = _value(source, "p_value")
                status = (
                    "no_eligible_events" if not population else
                    "insufficient_data" if count < 2 else
                    "inference_unavailable" if p_value is None else "ok"
                )
                family_slots.append({
                    "family": family, "test_id": f"{family}:L{layer}:{offset:+d}",
                    "endpoint": "utility", "layer": layer, "state_key": state,
                    "primary_intervention": primary_intervention,
                    "dose": restoration_dose,
                    "raw_formal_pair_count": len(expected),
                    "event_population_count": len(population),
                    "valid_pair_count": count,
                    "excluded_pair_count": len(population) - count,
                    "excluded_pair_fraction": (
                        (len(population) - count) / len(population) if population else None
                    ),
                    "exclusion_reasons": json.dumps(reasons, sort_keys=True),
                    "excluded_pair_reasons": json.dumps(excluded, sort_keys=True),
                    "minimum_valid_pairs": 2,
                    "inference_status": status,
                    "p_value": p_value, "fdr_q_value": None,
                })
                family_sources.append(source)
        adjusted = _bh([
            slot["p_value"] if slot["inference_status"] == "ok" else 1.0
            for slot in family_slots
        ])
        for slot, source, q in zip(family_slots, family_sources, adjusted):
            if slot["inference_status"] == "ok":
                slot["fdr_q_value"] = q
            if source is not None:
                source.update(
                    fdr_family=family, fdr_q_value=slot["fdr_q_value"],
                    utility_fdr_family=family,
                    utility_fdr_q_value=slot["fdr_q_value"],
                    utility_inference_status=slot["inference_status"],
                    raw_pair_count=len(population),
                    excluded_pair_count=slot["excluded_pair_count"],
                    exclusion_reasons=slot["exclusion_reasons"],
                )
        counts[family] = dict(sorted(Counter(slot["inference_status"] for slot in family_slots).items()))
        output.extend(family_slots)
    return output, {
        "formal_statistics_version": FORMAL_STATISTICS_VERSION,
        "scope": "descriptive event F5; separate event-type populations",
        "formal_pair_count": len(expected),
        "event_population_counts": {name: len(value) for name, value in event_populations.items()},
        "family_status_counts": counts,
        "planned_test_counts": {f"F5_event_{name}": len(candidate_layers) * len(event_offsets) for name in EVENT_NAMES},
    }


__all__ = [
    "EVENT_NAMES", "FORMAL_STATISTICS_VERSION", "MINIMUM_CONFIRMATORY_PAIRS",
    "RQ2FormalStatisticsError", "apply_event_families",
    "apply_formal_families", "family_policy",
]
