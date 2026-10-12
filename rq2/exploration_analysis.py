"""Descriptive, paired dev analysis; deliberately no protocol lock or formal claim."""
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from rq2.analysis import EFFECT_DEFINITIONS, join_effect_rows, pair_cluster_bootstrap

ENDPOINTS = ("utility_effect", "refusal_effect", "compliance_effect")
GROUP = ("layer", "state_key", "intervention", "dose", "token_scope")


def estimate(rows, field, seed=42):
    result = pair_cluster_bootstrap(rows, value_field=field, replicates=2000,
                                    confidence=.95, seed=seed)
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(float(row[field]))
    values = [float(np.mean(v)) for v in by_pair.values()]
    result.update(positive_pairs=sum(v > 0 for v in values),
                  tied_pairs=sum(v == 0 for v in values),
                  negative_pairs=sum(v < 0 for v in values),
                  positive_fraction=sum(v > 0 for v in values) / len(values))
    return result


def paired_contrast(left, right):
    """Average random repetitions WITHIN pair, then contrast identical pairs."""
    def means(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["pair_id"]].append(row)
        return {p: {f: float(np.mean([r[f] for r in g])) for f in ENDPOINTS}
                for p, g in grouped.items()}
    a, b = means(left), means(right)
    if not a or set(a) != set(b):
        raise ValueError("Paired contrast populations must be identical and nonempty")
    return [{"pair_id": p, **{f: a[p][f] - b[p][f] for f in ENDPOINTS}} for p in sorted(a)]


def analyze(trials, labels):
    rows, coverage = join_effect_rows(trials, labels)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP)].append(row)
    summaries, contrasts = [], []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        item = {**dict(zip(GROUP, key)), "endpoints": {f: estimate(group, f) for f in ENDPOINTS}}
        # Exact paired binary test only for a single deterministic treatment per pair.
        if len({r["pair_id"] for r in group}) == len(group):
            plus = sum(r["refusal_effect"] > 0 for r in group)
            minus = sum(r["refusal_effect"] < 0 for r in group)
            item["descriptive_refusal_mcnemar_exact_p"] = (
                float(binomtest(plus, plus + minus, .5).pvalue) if plus + minus else 1.)
        summaries.append(item)
        main = key[2]
        if main not in {"r_direction", "reverse_suppression"} or key[3] != 1.:
            continue
        controls = (("h_direction_control", "random_direction_control", "sham") if main == "r_direction"
                    else ("reverse_h_control", "reverse_random_control", "reverse_sham"))
        for control in controls:
            other = groups.get((key[0], key[1], control, key[3], key[4]))
            if other:
                difference = paired_contrast(group, other)
                contrasts.append({"layer": key[0], "state_key": key[1],
                                  "contrast": main + " minus " + control,
                                  "endpoints": {f: estimate(difference, f) for f in ENDPOINTS}})
    return rows, {"scope": "post_oracle_exploratory_dev", "causal_evidence": False,
                  "protocol_locked": False, "formal_allowed": False,
                  "inference": "descriptive_only_no_confirmatory_FDR_or_pass_declaration",
                  "effect_definitions": EFFECT_DEFINITIONS, "coverage": coverage,
                  "summaries": summaries, "paired_control_contrasts": contrasts}


def window_contrasts(center_rows, window_rows, pair_ids):
    """Compare offsets on the same predeclared complete-window population."""
    expected = set(pair_ids)
    groups = defaultdict(list)
    for row in (*center_rows, *window_rows):
        if row["pair_id"] in expected and row["dose"] == 1. and row["intervention"] == "r_direction":
            groups[(row["layer"], row["state_key"])].append(row)
    result = []
    for layer in (19, 24, 26):
        center = groups[(layer, "event:first_non_refusal_step:+0")]
        if {r["pair_id"] for r in center} != expected:
            raise ValueError("Incomplete center coverage for window comparison")
        for offset in (-2, -1, 1, 3):
            other = groups[(layer, f"event:first_non_refusal_step:{offset:+d}")]
            if {r["pair_id"] for r in other} != expected:
                raise ValueError("Incomplete paired offset coverage")
            diff = paired_contrast(other, center)
            result.append({"layer": layer, "offset": offset, "reference_offset": 0,
                           "pair_ids": sorted(expected),
                           "contrast": "offset_effect minus center_effect",
                           "endpoints": {f: estimate(diff, f) for f in ENDPOINTS}})
    return {"comparisons": result,
            "limitation": "Baseline score floors and transient events prevent interpreting effect differences alone as mechanism state dependence."}
