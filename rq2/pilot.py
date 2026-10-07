"""Pre-formal dev-pilot stability and advancement checks."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PILOT_GATE_VERSION = 2
MINIMUM_DEV_PAIRS = 16
MINIMUM_SIGN_CONSISTENCY = 0.60


class RQ2PilotError(ValueError):
    """Raised when dev-pilot evidence cannot be interpreted safely."""


def read_pilot_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def evaluate_dev_pilot(
    summaries: Sequence[Mapping[str, Any]],
    *,
    intervention: str,
    dose: float,
    candidate_layers: Sequence[int],
    neighbor_layers: Sequence[int],
    fixed_steps: Sequence[int],
    minimum_effect: float,
    controls: Sequence[Mapping[str, Any]] = (),
    require_mechanism_checks: bool = False,
    event_state: str | None = None,
) -> dict[str, Any]:
    """Decide whether a fixed dev intervention merits an independent formal run."""
    if not candidate_layers or (not fixed_steps and event_state is None) or minimum_effect < 0:
        raise RQ2PilotError("pilot needs frozen candidate layers, fixed steps, and a non-negative threshold")
    if event_state is not None and (fixed_steps or event_state != "event:first_non_refusal_step:+0"):
        raise RQ2PilotError("event pilot requires only the frozen first-non-refusal center")
    states = (event_state,) if event_state else tuple(f"fixed:{step}" for step in fixed_steps)
    layers = set(candidate_layers) | set(neighbor_layers)
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in summaries:
        if row.get("intervention") != intervention or row.get("token_scope") != "audio":
            continue
        layer = _number(row, "layer")
        row_dose = _number(row, "dose")
        state = str(row.get("state_key", ""))
        if layer is None or not layer.is_integer() or int(layer) not in layers or state not in states:
            continue
        if row_dose is None or not math.isclose(row_dose, dose, rel_tol=0, abs_tol=1e-12):
            continue
        key = (int(layer), state)
        if key in by_key:
            raise RQ2PilotError(f"duplicate pilot summary for {intervention} at {key}")
        by_key[key] = row

    control_names = (
        ("subspace_vs_h", "subspace_vs_random", "subspace_vs_sham")
        if intervention == "subspace_restoration"
        else ("r_vs_h", "r_vs_random", "r_vs_sham")
    )

    def stable_direction(row: Mapping[str, Any] | None) -> bool:
        if row is None:
            return False
        effect = _number(row, "utility_effect_mean")
        count = _number(row, "pair_count")
        sign = _number(row, "sign_consistency")
        return (
            effect is not None and effect > 0
            and count is not None and count >= MINIMUM_DEV_PAIRS
            and sign is not None and sign >= MINIMUM_SIGN_CONSISTENCY
        )

    def control_supported(item: Mapping[str, Any], name: str, layer: int, state: str) -> bool:
        control_dose = _number(item, "dose")
        count = _number(item, "pair_count")
        effect = _number(item, "contrast_effect_mean")
        return (
            item.get("contrast") == name
            and item.get("contrast_scope") == "same_layer_specificity"
            and item.get("primary_intervention") == intervention
            and item.get("token_scope") == "audio"
            and _number(item, "layer") == layer
            and item.get("state_key") == state
            and _number(item, "control_layer") == layer
            and control_dose is not None
            and math.isclose(control_dose, dose, rel_tol=0, abs_tol=1e-12)
            and count is not None and count >= MINIMUM_DEV_PAIRS
            and effect is not None and effect > 0
            and item.get("norm_audit_status") == "pass"
        )

    regions: list[dict[str, Any]] = []
    for layer in candidate_layers:
        for state in states:
            row = by_key.get((layer, state))
            effect = _number(row, "utility_effect_mean") if row is not None else None
            pair_count = _number(row, "pair_count") if row is not None else None
            sign = _number(row, "sign_consistency") if row is not None else None
            partners = [
                {"layer": other_layer, "state_key": other_state,
                 "kind": "same_layer_other_state" if other_layer == layer else "adjacent_layer"}
                for other_layer, other_state in sorted(by_key)
                if (other_layer == layer and other_state != state)
                or (other_state == state and abs(other_layer - layer) == 1)
                if stable_direction(by_key[(other_layer, other_state)])
            ]
            refusal = _number(row, "refusal_effect_mean") if row is not None else None
            refusal_count = _number(row, "refusal_pair_count") if row is not None else None
            refusal_ok = (
                refusal is not None and refusal > 0
                and refusal_count is not None and refusal_count >= MINIMUM_DEV_PAIRS
            ) if require_mechanism_checks else None
            matched_controls = {
                name: any(
                    control_supported(item, name, layer, state)
                    for item in controls
                ) for name in control_names
            } if require_mechanism_checks else {}
            stable = stable_direction(row)
            eligible = (
                stable and effect is not None and effect >= minimum_effect
                and bool(partners)
                and (not require_mechanism_checks or (
                    refusal_ok is True and all(matched_controls.values())
                ))
            )
            regions.append({
                "layer": layer, "state_key": state, "intervention": intervention,
                "dose": dose, "utility_effect_mean": effect,
                "pair_count": int(pair_count) if pair_count is not None else 0,
                "sign_consistency": sign,
                "utility_ci_low": _number(row, "utility_ci_low") if row is not None else None,
                "utility_ci_high": _number(row, "utility_ci_high") if row is not None else None,
                "refusal_effect_mean": refusal,
                "refusal_pair_count": int(refusal_count) if refusal_count is not None else None,
                "refusal_direction_ok": refusal_ok,
                "controls_positive": matched_controls,
                "corroborating_regions": partners,
                "stable_direction": stable,
                "eligible": eligible,
            })
    qualifying = [
        {"layer": row["layer"], "state_key": row["state_key"]}
        for row in regions if row["eligible"]
    ]
    return {
        "pilot_gate_version": PILOT_GATE_VERSION,
        "pilot_coordinate": "event" if event_state else "fixed",
        "purpose": "dev advancement only; not a causal conclusion",
        "intervention": intervention,
        "dose": dose,
        "minimum_effect": minimum_effect,
        "minimum_valid_pairs": MINIMUM_DEV_PAIRS,
        "minimum_sign_consistency": MINIMUM_SIGN_CONSISTENCY,
        "require_mechanism_checks": require_mechanism_checks,
        "qualified": bool(qualifying),
        "qualifying_regions": qualifying,
        "regions": regions,
    }
