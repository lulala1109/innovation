"""Generate the RQ2 causal report and optional diagnostic figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rq2.artifacts import atomic_json, atomic_text
from rq2.pilot import PILOT_GATE_VERSION
from rq2.formal_statistics import FORMAL_STATISTICS_VERSION


RESTORATION_INTERVENTIONS = {"r_direction", "subspace_restoration"}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], limit: int = 20) -> str:
    if not rows:
        return "_No eligible rows._"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, divider]
    for row in rows[:limit]:
        values = []
        for field in fields:
            value = row.get(field, "")
            numeric = _number(value)
            values.append(f"{numeric:.4g}" if numeric is not None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _endpoint_table_rows(
    rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[str] = ("utility", "refusal", "compliance"),
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        for endpoint in endpoints:
            base_field = "strongreject_score" if endpoint == "utility" else endpoint
            result.append({
                "layer": row["layer"],
                "state_key": row["state_key"],
                "intervention": row["intervention"],
                "dose": row["dose"],
                "token_scope": row["token_scope"],
                "endpoint": endpoint,
                "direction": row.get("effect_direction"),
                "base_mean": row.get(f"base_{base_field}_mean"),
                "intervention_mean": row.get(f"intervention_{base_field}_mean"),
                "effect_mean": row.get(f"{endpoint}_effect_mean"),
                "ci_low": row.get(f"{endpoint}_ci_low"),
                "ci_high": row.get(f"{endpoint}_ci_high"),
                "pair_count": row.get(f"{endpoint}_pair_count"),
                "fdr_q_value": row.get(f"{endpoint}_fdr_q_value"),
                "inference_status": (
                    "paired t-test" if endpoint == "utility"
                    else row.get(f"{endpoint}_inference_status")
                ),
            })
    return result


def _plot_heatmap(rows: Sequence[Mapping[str, Any]], destination: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    primary = [
        row for row in rows
        if row.get("intervention") in RESTORATION_INTERVENTIONS and row.get("token_scope") == "audio"
    ]
    layers = sorted({int(row["layer"]) for row in primary})
    states = sorted({row["state_key"] for row in primary})
    if not layers or not states:
        return False
    matrix = np.full((len(states), len(layers)), np.nan)
    for row in primary:
        matrix[states.index(row["state_key"]), layers.index(int(row["layer"]))] = float(
            row["utility_effect_mean"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(8, len(layers) * 0.3), max(3, len(states) * 0.6)))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(range(len(layers)), labels=layers, rotation=90)
    axis.set_yticks(range(len(states)), labels=states)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Attack state")
    axis.set_title("RQ2 harmful-answer utility reduction")
    figure.colorbar(image, ax=axis, label="S_base - S_intervention")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def _plot_controls(rows: Sequence[Mapping[str, Any]], destination: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    eligible = [row for row in rows if _number(row.get("contrast_effect_mean")) is not None]
    if not eligible:
        return False
    eligible = sorted(eligible, key=lambda row: float(row["contrast_effect_mean"]), reverse=True)[:30]
    labels = [f"L{row['layer']} {row['state_key']} {row['contrast']}" for row in eligible]
    values = [float(row["contrast_effect_mean"]) for row in eligible]
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.25)))
    axis.barh(range(len(labels)), values)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Paired primary-minus-control utility effect")
    axis.set_title("Registered specificity contrasts")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def _plot_bidirectional(rows: Sequence[Mapping[str, Any]], destination: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    eligible = [row for row in rows if _number(row.get("utility_effect_mean")) is not None]
    if not eligible:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for intervention in sorted({str(row["intervention"]) for row in eligible}):
        subset = [row for row in eligible if row["intervention"] == intervention]
        axis.scatter(
            [int(row["layer"]) for row in subset],
            [float(row["utility_effect_mean"]) for row in subset],
            label=intervention,
            alpha=0.75,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Oriented harmful-answer utility change")
    axis.set_title("Restoration and reverse suppression")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def _plot_ranking(rows: Sequence[Mapping[str, Any]], destination: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    eligible = [
        row for row in rows
        if row.get("comparison_role") == "primary"
        and _number(row.get("rq1_r_drop")) is not None
        and _number(row.get("rq2_causal_effect")) is not None
    ]
    if not eligible:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        [float(row["rq1_r_drop"]) for row in eligible],
        [float(row["rq2_causal_effect"]) for row in eligible],
    )
    for row in eligible:
        axis.annotate(str(row["layer"]), (float(row["rq1_r_drop"]), float(row["rq2_causal_effect"])))
    axis.set_xlabel("RQ1 R-drop")
    axis.set_ylabel("RQ2 harmful-answer utility reduction")
    axis.set_title("Degradation versus causality ranking")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def _plot_event(rows: Sequence[Mapping[str, Any]], destination: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    eligible = [row for row in rows if _number(row.get("utility_effect_mean")) is not None]
    if not eligible:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in eligible:
        pieces = str(row["state_key"]).split(":")
        if len(pieces) != 3:
            continue
        try:
            offset = int(pieces[-1])
        except ValueError:
            continue
        groups.setdefault((str(row["layer"]), pieces[1]), []).append(
            (offset, float(row["utility_effect_mean"]))
        )
    if not groups:
        plt.close(figure)
        return False
    for (layer, event), values in sorted(groups.items()):
        values.sort()
        axis.plot([item[0] for item in values], [item[1] for item in values], marker="o", label=f"L{layer} {event}")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Event-relative PGD offset")
    axis.set_ylabel("Harmful-answer utility reduction")
    axis.set_title("Event-centered utility-effect profile")
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def _claim_status(summary: Mapping[str, Any]) -> tuple[bool, str]:
    gate = summary.get("causal_claim_gate")
    if (
        summary.get("behavior_endpoint_version") != 2
        or summary.get("layer_dose_version") != 2
        or summary.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
        or not isinstance(gate, Mapping)
        or gate.get("behavior_endpoint_version") != 2
        or gate.get("layer_dose_version") != 2
        or gate.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
        or gate.get("supported") is not True
        or gate.get("refusal_restoration_supported") is not True
        or gate.get("reverse_refusal_suppression_supported") is not True
    ):
        return False, (
            "Paired utility and explicit-refusal evidence, specificity controls, "
            "and reverse suppression did not all meet the current criteria."
        )
    return True, (
        "At least one layer×state region has paired utility and explicit-refusal "
        "evidence, the required same-layer/same-dose utility-effect specificity contrasts, and "
        "same-layer reverse utility/refusal evidence."
    )


def generate_rq2_report(
    output_root: str | Path,
    *,
    title: str = "RQ2 Causal Layer×State Analysis",
) -> Path:
    root = Path(output_root).expanduser().resolve()
    formal = root / "formal"
    event = root / "event"
    summary_path = formal / "rq2_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    if (
        summary.get("behavior_endpoint_version") != 2
        or summary.get("layer_dose_version") != 2
        or summary.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
    ):
        raise ValueError("RQ2 report requires current endpoint, layer/dose and formal family analysis")
    formal_manifest = _read_json(formal / "rq2_formal_family_manifest.json")
    if (
        formal_manifest.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
        or summary.get("formal_family_manifest") != formal_manifest
    ):
        raise ValueError("formal family manifest does not match the analysis summary")
    formal_tests = _read_csv(formal / "rq2_formal_tests.csv")
    planned_formal_tests = sum(
        count for family, count in formal_manifest["fdr_families"].items()
        if not family.startswith("F5_event_")
    )
    if formal_manifest.get("raw_pair_count") != 40 or len(formal_tests) != planned_formal_tests:
        raise ValueError("formal family audit is incomplete or has the wrong frozen population")
    incomplete_tests = [row for row in formal_tests if row.get("inference_status") != "ok"]
    family_rows = [
        {
            "family": family, "planned_tests": planned,
            "status_counts": formal_manifest.get("family_status_counts", {}).get(family),
        }
        for family, planned in formal_manifest.get("fdr_families", {}).items()
        if not family.startswith("F5_event_")
    ]
    causal = [
        row for row in _read_csv(formal / "rq2_causal_map_mean_ci.csv")
        if row.get("dose_role") == "primary"
    ]
    controls = [
        row for row in _read_csv(formal / "rq2_specificity_controls.csv")
        if row.get("dose_role") == "primary"
    ]
    layer_profiles = [
        row for row in _read_csv(formal / "rq2_layer_profile_contrasts.csv")
        if row.get("dose_role") == "primary"
    ]
    formal_sensitivity = _read_csv(formal / "rq2_dose_sensitivity.csv")
    pilot_sensitivity = _read_csv(root / "mechanism_pilot" / "rq2_dose_sensitivity.csv")
    pilot_decisions = []
    for stage in ("oracle_analyze", "mechanism_analyze", "subspace_analyze"):
        source = root / stage.replace("_analyze", "_pilot") / "analysis.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        pilot_summary = _read_json(source)
        if pilot_summary.get("pilot_gate_version") != PILOT_GATE_VERSION:
            raise ValueError("RQ2 report requires current dev pilot decisions")
        if stage == "subspace_analyze" and pilot_summary.get("status") == "not_triggered":
            continue
        decision = pilot_summary.get("pilot_decision")
        if not isinstance(decision, Mapping):
            raise ValueError(f"missing dev pilot decision: {source}")
        for region in decision.get("regions", ()):
            pilot_decisions.append({
                "stage": stage,
                **region,
                "corroborating_count": len(region.get("corroborating_regions", ())),
            })
    bidirectional = [
        row for row in _read_csv(formal / "rq2_bidirectional_intervention.csv")
        if row.get("dose_role") == "primary"
    ]
    comparison = _read_csv(formal / "rq2_degradation_vs_causality.csv")
    comparison_sources = list((summary.get("rq1_vs_rq2") or {}).get("sources", {}).values())
    event_rows = [
        row for row in _read_csv(event / "rq2_event_causal_profiles.csv")
        if row.get("dose_role") == "primary"
    ]
    event_tests = _read_csv(event / "rq2_event_formal_tests.csv")
    event_paired = _read_csv(event / "rq2_event_paired_offsets.csv")
    event_summary_path = event / "event_analysis.json"
    event_summary = (
        _read_json(event_summary_path) if event_summary_path.is_file() else {}
    )
    if (event_summary or event_rows) and (
        event_summary.get("behavior_endpoint_version") != 2
        or event_summary.get("layer_dose_version") != 2
        or event_summary.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
    ):
        raise ValueError("RQ2 event report requires current endpoint and layer/dose analysis")
    if event_paired and not event_summary:
        raise ValueError("event paired-offset rows exist without a matching event analysis")
    if event_summary:
        paired_manifest = _read_json(event / "rq2_event_paired_offset_manifest.json")
        if (
            event_summary.get("event_paired_offset_manifest") != paired_manifest
            or paired_manifest.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
            or len(event_paired) != sum(paired_manifest["planned_test_counts"].values())
        ):
            raise ValueError("event paired-offset family audit is incomplete")
    claim, claim_reason = _claim_status(summary)
    gate = summary.get("causal_claim_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    gate_current = (
        gate.get("behavior_endpoint_version") == 2
        and gate.get("layer_dose_version") == 2
        and gate.get("formal_statistics_version") == FORMAL_STATISTICS_VERSION
    )
    utility_claim = gate_current and gate.get("utility_reduction_supported") is True
    refusal_claim = utility_claim and gate.get("refusal_restoration_supported") is True
    mechanism_claim = refusal_claim and bool(gate.get("restoration_specificity_regions"))
    bidirectional_claim = mechanism_claim and claim
    interaction = summary.get("random_intercept_model", {})
    interaction_lrt = (
        interaction.get("interaction_likelihood_ratio_test", {})
        if isinstance(interaction, Mapping)
        else {}
    )
    supported_layers = set(gate.get("bidirectional_layers", ()))
    qualified_event_pairs = [
        row for row in event_paired
        if row.get("inference_status") == "ok"
        and _number(row.get("layer")) in supported_layers
        and _number(row.get("fdr_q_value")) is not None
        and float(row["fdr_q_value"]) <= float(event_summary.get("fdr_alpha", 0.05))
        and _number(row.get("ci_low")) is not None
        and _number(row.get("ci_high")) is not None
        and (float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0)
        and int(row["valid_pair_count"]) >= 20
    ]
    event_reproduced = bool(qualified_event_pairs)
    # A global interaction can be driven outside supported layers; E1 is
    # pre-registered and restricted to a bidirectionally supported layer.
    state_claim = bidirectional_claim and event_reproduced
    claim_levels = {
        "utility_reduction": bool(utility_claim),
        "refusal_restoration": bool(refusal_claim),
        "r_specific_mechanism": bool(mechanism_claim),
        "bidirectional_evidence": bool(bidirectional_claim),
        "state_dependent_effect": bool(state_claim),
        "bottleneck_migration": False,
    }

    heatmap = root / "figures" / "rq2_causal_heatmap.png"
    plotted = _plot_heatmap(causal, heatmap)
    controls_plot = _plot_controls(controls, root / "figures" / "rq2_specificity_controls.png")
    bidirectional_plot = _plot_bidirectional(
        bidirectional, root / "figures" / "rq2_restore_suppress.png"
    )
    ranking_plot = _plot_ranking(comparison, root / "figures" / "rq2_ranking_comparison.png")
    event_plot = _plot_event(event_rows, root / "figures" / "rq2_event_profile.png")
    ranked = sorted(
        [row for row in causal
         if row.get("intervention") in RESTORATION_INTERVENTIONS
         and row.get("utility_inference_status") == "ok"],
        key=lambda row: (
            float(row["utility_effect_mean"])
            if _number(row.get("utility_effect_mean")) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    lines = [
        f"# {title}",
        "",
        "## Scope and decision",
        "",
        f"- 1. 效用下降 / Utility reduction: **{'supported' if utility_claim else 'not supported'}**.",
        f"- 2. 拒答恢复 / Refusal restoration: **{'supported' if refusal_claim else 'not supported'}**.",
        f"- 3. R 相关机制特异性 / R-specific mechanism: **{'supported' if mechanism_claim else 'not supported'}**.",
        f"- 4. 双向证据 / Bidirectional evidence: **{'supported' if bidirectional_claim else 'not supported'}**. {claim_reason}",
        f"- 5. 状态依赖效应 / State-dependent effect: **{'supported' if state_claim else 'not supported'}**. "
        + ("A same-event complete-case paired-offset contrast at a bidirectionally supported layer qualifies." if state_claim else "Requires all earlier levels plus a qualifying same-layer E1 event contrast."),
        "- 跨层瓶颈迁移 / Cross-layer bottleneck migration: **not established** by this protocol.",
        f"- Exploratory global Layer×State interaction LRT: status={interaction_lrt.get('status', 'unavailable')}, p={interaction_lrt.get('p_value')}; this alone cannot qualify level 5.",
        f"- Qualified same-event paired-offset tests: {len(qualified_event_pairs)}.",
        f"- Frozen formal population: {formal_manifest.get('raw_pair_count')} pairs; "
        f"planned formal tests: {len(formal_tests)}; tests without qualified inference: {len(incomplete_tests)}.",
        f"- Fully analyzed paired effects: {summary.get('analyzed_effect_count', 0)}; unknown labels excluded: {summary.get('unknown_label_count', 0)}.",
        "",
        "S is the StrongREJECT harmful-answer utility score; R is the explicit-refusal "
        "label and C is the frozen-threshold effective-compliance label. "
        "Restoration uses ΔU=S_base-S_patch, ΔR=R_patch-R_base, ΔC=C_base-C_patch. "
        "Reverse suppression uses ΔU=S_suppress-S_base_clean, "
        "ΔR=R_base_clean-R_suppress, ΔC=C_suppress-C_base_clean.",
        "",
        f"Primary restoration dose: {summary.get('primary_restoration_dose')}; "
        f"primary suppression dose: {summary.get('primary_suppression_dose')}. "
        "Primary maps, rankings and decision gates exclude sensitivity doses.",
        "",
        "Neighbor layers describe local profiles; distant layers are early/middle references, "
        "not depth-matched controls. Layer-profile comparisons are reported separately and "
        "cannot satisfy the same-layer specificity gate.",
        "",
        "The heatmap shows continuous utility reduction. A positive utility effect alone "
        "does not establish refusal restoration: the same comparison must also have a "
        "positive explicit-refusal effect, a pair-bootstrap CI above zero, and a "
        "qualifying exact McNemar FDR result.",
        "",
        "For Qwen layer 27, the frozen RQ1 representation is after the final decoder "
        "norm. RQ2 hooks that norm to keep the representation aligned. Audio-only "
        "prefill changes at this site cannot reach the answer-position logits or "
        "earlier KV cache, so its behavioral null is a structural limitation of "
        "this operator, not evidence that the R component is absent.",
        "",
        "The legacy `refusal_orientation=1-S` is not a refusal probability. "
        "`causal_effect`, `ce_restore`, and `ce_suppress` are utility-effect aliases. "
        "`behavior_recovery_rate` is a normalized utility-score change, not the fraction "
        "of samples whose refusal was restored. Internal `delta_r` remains an R-projection "
        "change and is distinct from the binary refusal effect.",
        "",
        "Protocol deviation T0.5: the AdvBench pool used conservative automated screening; ",
        "manual semantic-independence and audio-content-fidelity review was not performed. ",
        "This limitation applies to all resulting RQ2 claims.",
        "",
        "No mild/moderate/severe refusal-degradation proxy is provided. ",
        "Fixed PGD states are not severity bins.",
        "",
        "Replicates are averaged within each pair. Fractional binary-label replicate "
        "means retain pair-bootstrap estimates but have no exact McNemar p/q value "
        "(`non_binary_replicate_mean`); they cannot supply binary confirmation.",
        "",
    ]
    if plotted:
        lines.extend(["![Harmful-answer utility reduction by layer and state](figures/rq2_causal_heatmap.png)", ""])
    for available, caption, path in (
        (controls_plot, "Specificity controls", "figures/rq2_specificity_controls.png"),
        (bidirectional_plot, "Restore/suppress", "figures/rq2_restore_suppress.png"),
        (ranking_plot, "RQ1 vs RQ2 ranking", "figures/rq2_ranking_comparison.png"),
        (event_plot, "Event-centered profile", "figures/rq2_event_profile.png"),
    ):
        if available:
            lines.extend([f"![{caption}]({path})", ""])
    lines.extend(
        [
            "## Formal family and effective-pair audit",
            "",
            "F1–F4 require at least 20 independent paired observations per comparison. "
            "Missing or insufficient comparisons retain a p=1 correction slot and have no reported q value. "
            "F5 families are descriptive and do not satisfy the causal gate.",
            "",
            _markdown_table(family_rows, ("family", "planned_tests", "status_counts")),
            "",
            _markdown_table(
                incomplete_tests,
                ("family", "test_id", "raw_pair_count", "valid_pair_count",
                 "excluded_pair_count", "excluded_pair_fraction",
                 "exclusion_reasons", "inference_status"),
                limit=80,
            ),
            "",
            "Full per-test details, including excluded pair IDs and reasons, are in "
            "`formal/rq2_formal_tests.csv`.",
            "",
            "## Largest harmful-answer utility reductions",
            "",
            _markdown_table(
                ranked,
                ("layer", "state_key", "intervention", "dose", "token_scope", "utility_effect_mean", "utility_ci_low", "utility_ci_high", "utility_pair_count", "utility_fdr_q_value"),
            ),
            "",
            "## Explicit refusal and effective compliance",
            "",
            _markdown_table(
                _endpoint_table_rows(ranked, ("refusal", "compliance")),
                ("layer", "state_key", "intervention", "dose", "token_scope", "endpoint",
                 "base_mean", "intervention_mean", "effect_mean", "ci_low", "ci_high",
                 "pair_count", "fdr_q_value", "inference_status"),
                limit=40,
            ),
            "",
            "Binary means are rates; an effect of 0.20 is a net change of 20 percentage points.",
            "",
            "## Answer-quality diagnostics",
            "",
            _markdown_table(
                ranked,
                ("layer", "state_key", "intervention", "dose", "token_scope",
                 "base_convincingness_mean", "intervention_convincingness_mean",
                 "base_specificity_mean", "intervention_specificity_mean"),
            ),
            "",
            "## Utility-effect specificity controls",
            "",
            _markdown_table(
                controls,
                ("layer", "state_key", "primary_intervention", "dose", "contrast", "control_layer", "contrast_effect_mean", "ci_low", "ci_high", "pair_count", "fdr_q_value", "norm_audit_pair_count", "norm_audit_missing_pair_count", "norm_max_relative_error", "sham_max_shift_l2", "norm_audit_status"),
            ),
            "",
            "Actual generation shift L2 is audited within each paired same-layer comparison. Non-sham H/random and token-position controls must be within 5% relative error; sham must have exactly zero shift. Missing, zero primary/control, or mismatched audits cannot support mechanism specificity.",
            "",
            "## Neighbor and distant layer profiles",
            "",
            _markdown_table(
                layer_profiles,
                ("layer", "state_key", "primary_intervention", "dose", "contrast",
                 "control_layer", "control_layer_role", "contrast_effect_mean",
                 "ci_low", "ci_high", "pair_count", "fdr_q_value"),
            ),
            "",
            "## Dev pilot advancement decisions",
            "",
            "These are dev go/no-go diagnostics, not formal causal conclusions.",
            "",
            _markdown_table(
                pilot_decisions,
                ("stage", "layer", "state_key", "intervention", "dose",
                 "utility_effect_mean", "pair_count", "sign_consistency",
                 "utility_ci_low", "utility_ci_high", "refusal_effect_mean",
                 "corroborating_count", "controls_positive", "eligible"),
                limit=60,
            ),
            "",
            "## Dev pilot dose sensitivity (separate from formal evidence)",
            "",
            _markdown_table(
                _endpoint_table_rows(pilot_sensitivity),
                ("layer", "state_key", "intervention", "dose", "endpoint",
                 "effect_mean", "ci_low", "ci_high", "pair_count"),
                limit=60,
            ),
            "",
            "Sensitivity conditions without same-dose controls describe dose response only.",
            "",
            "## Formal dose sensitivity (if separately generated)",
            "",
            _markdown_table(
                _endpoint_table_rows(formal_sensitivity),
                ("layer", "state_key", "intervention", "dose", "endpoint",
                 "effect_mean", "ci_low", "ci_high", "pair_count"),
                limit=60,
            ),
            "",
            "## Bidirectional restore/suppress test",
            "",
            _markdown_table(
                _endpoint_table_rows(bidirectional),
                ("layer", "state_key", "intervention", "dose", "token_scope", "endpoint",
                 "direction", "effect_mean", "ci_low", "ci_high", "pair_count", "fdr_q_value"),
                limit=60,
            ),
            "",
            "## RQ1 degradation versus RQ2 causality",
            "",
            "RQ1 all/baseline-refused R_probe and R_direction slopes, plus event-aligned "
            "R_probe drops, are cross-dataset descriptive rankings. Event statistics use "
            "aggregate −2 and +3 means that may have different available pairs. The RQ2 "
            "scan:0-to-scan:100 R-projection drop is paired within the same AdvBench "
            "causal-test population. None of these rankings enters layer selection or the causal gate.",
            "",
            _markdown_table(
                comparison_sources,
                ("source", "population", "metric", "event", "shared_layer_count",
                 "spearman_rho", "top_k_overlap", "candidate_precision", "interpretation"),
                limit=25,
            ),
            "",
            _markdown_table(
                [row for row in comparison if row.get("comparison_role") == "primary"],
                ("source", "layer", "predictor_drop", "rq2_causal_effect"),
                limit=30,
            ),
            "",
            "Full per-layer multi-source alignment is in `formal/rq2_degradation_vs_causality.csv`.",
            "",
            "## Event-centered profile",
            "",
            "Event-type F5 offset summaries remain descriptive. E1 compares the mean of 0/+1/+3 "
            "with −2/−1 within the same event, layer and complete-case pair. Each event type "
            "has a separate four-layer BH family; at least 20 complete pairs, q≤alpha and a "
            "pair-bootstrap CI excluding zero are required for state-dependent evidence.",
            "",
            _markdown_table(
                event_paired,
                ("family", "event", "layer", "event_population_count", "valid_pair_count",
                 "excluded_pair_count", "paired_post_minus_pre_mean", "ci_low", "ci_high",
                 "fdr_q_value", "inference_status"),
                limit=60,
            ),
            "",
            "Complete-case pair IDs and missing-offset reasons are in `event/rq2_event_paired_offsets.csv`.",
            "",
            _markdown_table(
                event_tests,
                ("family", "test_id", "event_population_count", "valid_pair_count",
                 "excluded_pair_count", "inference_status", "fdr_q_value"),
                limit=60,
            ),
            "",
            _markdown_table(
                _endpoint_table_rows(event_rows),
                ("layer", "state_key", "intervention", "dose", "token_scope", "endpoint",
                 "effect_mean", "ci_low", "ci_high", "pair_count", "fdr_q_value"),
                limit=60,
            ),
            "",
            "## Interpretation guardrails",
            "",
            "Missing event offsets remain missing and are never substituted by another PGD step. Unknown Judge outcomes are excluded rather than converted to zero or `false`. Pilot results are not included in formal estimates, and all response/Judge text remains outside these analysis tables.",
            "",
        ]
    )
    report = root / "rq2_report.md"
    atomic_text(report, "\n".join(lines))
    atomic_json(
        root / "rq2_summary.json",
        {
            **dict(summary),
            "utility_reduction_supported": utility_claim,
            "refusal_restoration_supported": refusal_claim,
            "causal_refusal_bottleneck_supported": bidirectional_claim,
            "attack_state_dependent_supported": state_claim,
            "bottleneck_migration_supported": False,
            "claim_levels": claim_levels,
            "qualified_event_paired_offsets": [row["test_id"] for row in qualified_event_pairs],
            "report": str(report),
        },
    )
    prior_source = formal / "rq2_causal_prior.json"
    if prior_source.is_file():
        atomic_json(root / "rq2_causal_prior.json", _read_json(prior_source))
    return report


__all__ = ["generate_rq2_report"]
