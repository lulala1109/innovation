"""Static, fail-closed reachability for Qwen audio-only prefill patching.

No model, tensor, API, or file writes. A reachable operation is not necessarily
effective. Existing layer roles and historical results are never rewritten.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class ReachabilityError(ValueError):
    pass


CONTRACT = "qwen_audio_only_prefill_block_output_v1"
KINDS = frozenset({
    "full_state", "self_patch", "sham", "r_direction", "h_direction_control",
    "random_direction_control", "reverse_suppression", "reverse_h_control",
    "reverse_random_control", "reverse_sham", "subspace_restoration",
    "subspace_h_control", "subspace_random_control",
})


def validated_layer_map(layer_map: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if (layer_map.get("format") != "rq2-qwen-layer-map" or layer_map.get("version") != 1
            or layer_map.get("passed") is not True):
        raise ReachabilityError("requires a passed Qwen layer-map artifact")
    count, rows = layer_map.get("layer_count"), layer_map.get("layers")
    if type(count) is not int or count < 1 or not isinstance(rows, list) or len(rows) != count:
        raise ReachabilityError("layer map has incomplete layer coverage")
    indexed = {}
    for row in rows:
        layer = row.get("decoder_layer")
        if (type(layer) is not int or not 0 <= layer < count or layer in indexed
                or row.get("hidden_state_index") != layer + 1 or row.get("passed") is not True):
            raise ReachabilityError("invalid, duplicate, or unverified layer-map row")
        site = row.get("activation_site")
        if site not in {"decoder_block_output", "final_output_norm"}:
            raise ReachabilityError("unknown activation site requires a new reachability contract")
        if site == "final_output_norm" and layer != count - 1:
            raise ReachabilityError("final norm is not at the terminal layer")
        # Even the terminal block output BEFORE final norm has no later
        # attention layer to transport an audio-only change to the readout.
        reachable = layer < count - 1 and site == "decoder_block_output"
        declared = row.get("audio_only_prefill_behaviorally_reachable")
        if type(declared) is not bool or declared != reachable:
            raise ReachabilityError("reachability flag is missing or contradicts activation topology")
        indexed[layer] = {"layer": layer, "activation_site": site,
                          "audio_only_prefill_behaviorally_reachable": reachable,
                          "reason": "later_attention_available" if reachable else "no_downstream_attention_after_audio_patch"}
    return indexed


def _field(item, name):
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def audit_plan_reachability(plan: Sequence[Any], layer_map: Mapping[str, Any],
                            *, execution_phase: str = "prefill_only") -> dict[str, Any]:
    """Audit actual planned operations, never silently prune an invalid layer."""
    layers = validated_layer_map(layer_map)
    grouped = defaultdict(list)
    for item in plan:
        layer, scope, kind, dose = (_field(item, name) for name in ("layer", "token_scope", "intervention", "dose"))
        if type(layer) is not int or layer not in layers or kind not in KINDS:
            raise ReachabilityError("plan contains an unknown layer or intervention")
        if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose):
            raise ReachabilityError("plan dose must be finite")
        if not isinstance(scope, str) or not scope:
            raise ReachabilityError("plan lacks token_scope")
        grouped[(layer, scope, kind, float(dose))].append(item)
    operations = []
    for (layer, scope, kind, dose), items in sorted(grouped.items()):
        topology = layers[layer]
        noop = kind in {"sham", "reverse_sham", "self_patch"} or dose == 0
        if scope != "audio" or execution_phase != "prefill_only":
            status = "unknown_contract"
            reason = "different_token_scope_or_execution_phase_requires_new_contract"
        elif noop:
            status, reason = "expected_noop", "identity_control_not_positive_behavioral_evidence"
        elif topology["audio_only_prefill_behaviorally_reachable"]:
            status, reason = "reachable", topology["reason"]
        else:
            status, reason = "unreachable", topology["reason"]
        operations.append({**topology, "token_scope": scope, "execution_phase": execution_phase,
                           "intervention": kind, "dose": dose, "status": status, "reason": reason,
                           "trial_count": len(items),
                           "pair_count": len({_field(i, "pair_id") for i in items})})
    blocked = [r for r in operations if r["status"] in {"unreachable", "unknown_contract"}]
    return {"format": "rq2-operation-reachability", "version": 1, "contract": CONTRACT,
            "passed": bool(operations) and not blocked, "planned_trial_count": len(plan),
            "blocked_trial_count": sum(r["trial_count"] for r in blocked),
            "operations": operations, "blocked_operations": blocked,
            "empty_plan": not bool(operations), "automatically_removed_layers": [],
            "proves_behavioral_effect": False}


def require_plan_reachable(plan, layer_map, *, execution_phase="prefill_only"):
    audit = audit_plan_reachability(plan, layer_map, execution_phase=execution_phase)
    if not audit["passed"]:
        blocked = sorted({r["layer"] for r in audit["blocked_operations"]})
        raise ReachabilityError(
            f"behavioral generation blocked: unreachable/unverified operation layers={blocked}; "
            "preserve frozen candidates/results and define a new protocol/run; do not silently drop layers")
    return audit
