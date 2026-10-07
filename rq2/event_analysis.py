"""Descriptive event-window statistics; never a formal claim gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from rq2.analysis import pair_cluster_bootstrap
from rq2.event_config import EVENT_NAME, EVENT_OFFSETS


def window_summary(
    effects: Sequence[Mapping[str, Any]], *, event_pair_ids: Sequence[str],
    full_window_pair_ids: Sequence[str], candidate_layers: Sequence[int],
    replicates: int, confidence: float, seed: int,
) -> dict[str, Any]:
    """Use the identical complete-case pairs at every offset within a layer."""
    values = defaultdict(list)
    for row in effects:
        if (row["pair_id"] not in event_pair_ids or row["intervention"] not in {"r_direction", "sham"}
            or row["token_scope"] != "audio" or float(row["dose"]) != 1.0):
            continue
        prefix = f"event:{EVENT_NAME}:"
        if not str(row["state_key"]).startswith(prefix):
            continue
        offset = int(row["state_key"][len(prefix):])
        values[(row["pair_id"], int(row["layer"]), row["intervention"], offset)].append(float(row["utility_effect"]))
    groups = []
    for layer in candidate_layers:
        # Shared population also across R and sham, so their profiles compare fairly.
        valid = [pair for pair in full_window_pair_ids if all(
            values[(pair, layer, intervention, offset)]
            for intervention in ("r_direction", "sham") for offset in EVENT_OFFSETS
        )]
        for intervention in ("r_direction", "sham"):
            profiles = []
            for offset in EVENT_OFFSETS:
                rows = [{"pair_id": pair, "value": float(np.mean(values[(pair, layer, intervention, offset)]))} for pair in valid]
                estimate = pair_cluster_bootstrap(rows, value_field="value", replicates=replicates,
                    confidence=confidence, seed=seed + layer + offset) if len(rows) >= 2 else None
                profiles.append({"offset": offset, "valid_pair_count": len(valid), "estimate": estimate})
            differences = []
            for pair in valid:
                by_offset = {offset: float(np.mean(values[(pair, layer, intervention, offset)])) for offset in EVENT_OFFSETS}
                difference = float(np.mean([by_offset[o] for o in (0, 1, 3)]) - np.mean([by_offset[o] for o in (-2, -1)]))
                differences.append({"pair_id": pair, "value": difference})
            contrast = pair_cluster_bootstrap(differences, value_field="value", replicates=replicates,
                confidence=confidence, seed=seed + 1000 + layer) if len(differences) >= 2 else None
            groups.append({"layer": layer, "intervention": intervention, "valid_pair_ids": valid,
                           "excluded_pair_ids": sorted(set(event_pair_ids) - set(valid)),
                           "profiles": profiles, "paired_post_minus_pre": contrast})
    return {"format": "rq2-event-dev-window", "version": 3, "scope": "descriptive_dev_only",
            "event_population_count": len(event_pair_ids),
            "structural_full_window_count": len(full_window_pair_ids),
            "required_offsets": list(EVENT_OFFSETS), "pre_offsets": [-2,-1], "post_offsets": [0,1,3],
            "groups": groups, "causal_evidence": False,
            "interpretation": "same-checkpoint intervention effects; event selection alone is not evidence of state dependence",
            "p_values": None, "fdr": None}
