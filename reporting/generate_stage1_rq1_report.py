#!/usr/bin/env python3
"""Generate the Stage-1 RQ1 report and optional cross-sample figures."""

from __future__ import annotations

import importlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class OptionalRQ1ReportingDependencyError(RuntimeError):
    """Raised only when optional figure output is explicitly requested."""


def _pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise OptionalRQ1ReportingDependencyError(
            "RQ1 figure output requires optional matplotlib; CSV, JSON, and "
            "Markdown outputs remain available without it"
        ) from exc


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            if not value.endswith("\n"):
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


def _save_figure(figure: Any, path: Path, pyplot: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    pyplot.close(figure)
    return path


def _ordered(values: Sequence[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def plot_layer_step_heatmap(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    metric: str,
    value_field: str = "mean",
) -> Path:
    """Plot one cross-sample mean or delta Layer x PGD-step heatmap."""

    selected = [row for row in rows if row.get("metric") == metric]
    if not selected:
        raise ValueError(f"cell statistics contain no metric {metric!r}")
    layers = _ordered([row["layer"] for row in selected])
    steps = sorted({int(row["step"]) for row in selected})
    lookup = {
        (str(row["layer"]), int(row["step"])): float(row[value_field])
        for row in selected
    }
    matrix = [
        [lookup.get((str(layer), step), float("nan")) for step in steps]
        for layer in layers
    ]
    plt = _pyplot()
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(steps) * 0.09), max(4.0, len(layers) * 0.22))
    )
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    tick_stride = max(1, len(steps) // 10)
    positions = list(range(0, len(steps), tick_stride))
    if positions[-1] != len(steps) - 1:
        positions.append(len(steps) - 1)
    axis.set_xticks(positions, labels=[steps[index] for index in positions])
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("PGD step")
    axis.set_ylabel("Layer")
    label = metric if value_field == "mean" else f"Delta {metric}"
    axis.set_title(f"{label}: cross-sample Layer x PGD-step")
    figure.colorbar(image, ax=axis, label=label)
    return _save_figure(figure, Path(path).expanduser().resolve(), plt)


def plot_phase_profiles(
    rows: Sequence[Mapping[str, Any]], path: str | Path, *, metric: str
) -> Path:
    """Plot early/middle/late layer-wise profiles with bootstrap intervals."""

    selected = [row for row in rows if row.get("metric") == metric]
    if not selected:
        raise ValueError(f"phase profiles contain no metric {metric!r}")
    layers = _ordered([row["layer"] for row in selected])
    lookup = {(str(row["layer"]), row["phase"]): row for row in selected}
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(max(7.0, len(layers) * 0.28), 4.5))
    x_values = list(range(len(layers)))
    for phase in ("early", "middle", "late"):
        phase_rows = [lookup[(str(layer), phase)] for layer in layers]
        means = [float(row["mean"]) for row in phase_rows]
        lower = [mean - float(row["ci_low"]) for mean, row in zip(means, phase_rows)]
        upper = [float(row["ci_high"]) - mean for mean, row in zip(means, phase_rows)]
        axis.errorbar(
            x_values,
            means,
            yerr=[lower, upper],
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=phase,
        )
    axis.set_xticks(x_values, labels=layers, rotation=45)
    axis.set_xlabel("Layer")
    axis.set_ylabel(metric)
    axis.set_title(f"{metric}: early/middle/late layer profiles")
    axis.legend()
    return _save_figure(figure, Path(path).expanduser().resolve(), plt)


def plot_layer_trajectories(
    rows: Sequence[Mapping[str, Any]], path: str | Path, *, metric: str
) -> Path:
    """Plot each layer's cross-sample optimization-time trajectory."""

    selected = [row for row in rows if row.get("metric") == metric]
    if not selected:
        raise ValueError(f"cell statistics contain no metric {metric!r}")
    layers = _ordered([row["layer"] for row in selected])
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for layer in layers:
        layer_rows = sorted(
            [row for row in selected if str(row["layer"]) == str(layer)],
            key=lambda row: float(row["progress"]),
        )
        axis.plot(
            [float(row["progress"]) for row in layer_rows],
            [float(row["mean"]) for row in layer_rows],
            linewidth=1.0,
            alpha=0.75,
            label=str(layer),
        )
    axis.set_xlabel("Normalized PGD progress")
    axis.set_ylabel(metric)
    axis.set_title(f"{metric}: layer-wise optimization trajectories")
    if len(layers) <= 12:
        axis.legend(title="Layer", ncol=2, fontsize="small")
    return _save_figure(figure, Path(path).expanduser().resolve(), plt)


def _effect_line(name: str, value: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(value, Mapping):
        return f"- {name}: not computed"
    if value.get("status") == "invalid" or value.get("p_value") is None:
        reason = value.get("reason") or "model inference was not valid"
        return f"- {name}: not reported ({reason})"
    statistic = value.get("likelihood_ratio")
    p_value = value.get("p_value")
    return f"- {name}: likelihood-ratio={statistic}, p={p_value}"


def _split_lines(analysis: Mapping[str, Any]) -> list[str]:
    axes = analysis.get("axes", {})
    pair_ids = axes.get("pair_ids", []) if isinstance(axes, Mapping) else []
    held_out = len(pair_ids) if isinstance(pair_ids, Sequence) else 0
    analysis_metadata = analysis.get("metadata", {})
    source = (
        analysis_metadata.get("source_metadata", {})
        if isinstance(analysis_metadata, Mapping)
        else {}
    )
    if not isinstance(source, Mapping):
        source = {}
    train_pairs = source.get("probe_training_num_pairs")
    train_verified = source.get("probe_training_stage1_provenance_verified") is True
    train_role_ok = (
        source.get("probe_training_measurement_split") == "measurement_train"
        and source.get("probe_training_stage1_role") == "probe_candidate"
    )
    held_out_verified = (
        source.get("trajectory_measurement_split") == "measurement_val"
        and source.get("trajectory_stage1_role") == "trajectory_candidate"
        and source.get("trajectory_num_pairs") == held_out
    )
    if (
        isinstance(train_pairs, int)
        and not isinstance(train_pairs, bool)
        and train_verified
        and train_role_ok
        and held_out_verified
    ):
        status = (
            "matches the planned 80/20 split"
            if train_pairs == 80 and held_out == 20
            else "does not match the planned 80/20 split"
        )
        return [
            f"- Provenance-recorded split: {train_pairs} probe-training pairs / "
            f"{held_out} held-out trajectory pairs ({status})."
        ]
    return [
        f"- Held-out trajectory pairs present: {held_out}. The complete 80/20 "
        "split is not asserted because training/trajectory provenance is incomplete."
    ]


def build_rq1_markdown(
    analysis: Mapping[str, Any], *, figures: Mapping[str, str]
) -> str:
    """Build a result-only report that does not predeclare H/R decoupling."""

    axes = analysis.get("axes", {})
    mixed = analysis.get("mixed_effects")
    lines = [
        "# Stage-1 RQ1: Optimization-Time Safety-State Dynamics",
        "",
        "## Analysis population",
        "",
        f"- Held-out cases: {len(axes.get('case_ids', []))}",
        f"- Layers: {len(axes.get('layers', []))}",
        f"- PGD states per case: {len(axes.get('steps', []))}",
        *_split_lines(analysis),
        "",
        "## RQ1-a: layer structure and cross-sample reproducibility",
        "",
        "See `cell_statistics.csv` and `profile_reproducibility.csv`. Layer main "
        "effects and profile reliability must both be considered; a numerical "
        "layer difference alone is not sufficient.",
        "",
        "## RQ1-b: optimization-time dynamics",
        "",
    ]
    if isinstance(mixed, Mapping):
        for metric in ("H_probe", "R_probe"):
            model = mixed.get(metric, {})
            lines.extend(
                [
                    f"### {metric}",
                    "",
                    _effect_line("Layer effect", model.get("layer_effect")),
                    _effect_line(
                        "Attack-progress effect", model.get("attack_progress_effect")
                    ),
                    _effect_line(
                        "Layer × progress interaction",
                        model.get("layer_by_progress_interaction"),
                    ),
                    "",
                ]
            )
    else:
        lines.extend(["Mixed-effects analysis was skipped.", ""])
    lines.extend(
        [
            "## RQ1-c: H/R trajectory comparison",
            "",
            "See `layer_slopes.csv`, `event_aligned_statistics.csv`, and the "
            "H-vs-R mixed model. The hypothesis H≈stable/R↓ is not used as a "
            "filter and may only be claimed if supported by these results.",
            "",
        ]
    )
    if isinstance(mixed, Mapping):
        comparison = mixed.get("H_vs_R", {})
        lines.extend(
            [
                _effect_line(
                    "State type × progress", comparison.get("state_by_progress")
                ),
                _effect_line(
                    "State type × layer × progress",
                    comparison.get("state_by_layer_by_progress"),
                ),
                "",
            ]
        )
    if figures:
        lines.extend(["## Figures", ""])
        for name, path in figures.items():
            lines.append(f"- {name}: `{path}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "Stage 1 characterizes Layer × Attack-Step safety-state dynamics. "
            "It does not identify a causal critical layer and does not perform "
            "activation patching; those belong to later research stages.",
        ]
    )
    return "\n".join(lines)


def generate_stage1_rq1_report(
    analysis: Mapping[str, Any],
    output_dir: str | Path,
    *,
    make_plots: bool = False,
) -> Mapping[str, Any]:
    """Write Markdown and, when requested, H/R RQ1 figures."""

    if analysis.get("format") != "stage1-rq1-analysis":
        raise ValueError("analysis must use format 'stage1-rq1-analysis'")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    if make_plots:
        figure_dir = output / "figures"
        cell_rows = analysis.get("cell_statistics", [])
        phase_rows = analysis.get("phase_profiles", [])
        for metric in ("H_probe", "R_probe"):
            for field, suffix in (("mean", "heatmap"), ("delta_mean", "delta_heatmap")):
                name = f"{metric}_{suffix}"
                path = plot_layer_step_heatmap(
                    cell_rows,
                    figure_dir / f"{name}.png",
                    metric=metric,
                    value_field=field,
                )
                figures[name] = str(path)
            profile_name = f"{metric}_phase_profiles"
            figures[profile_name] = str(
                plot_phase_profiles(
                    phase_rows,
                    figure_dir / f"{profile_name}.png",
                    metric=metric,
                )
            )
            trajectory_name = f"{metric}_layer_trajectories"
            figures[trajectory_name] = str(
                plot_layer_trajectories(
                    cell_rows,
                    figure_dir / f"{trajectory_name}.png",
                    metric=metric,
                )
            )
    report_path = output / "rq1_report.md"
    _atomic_text(report_path, build_rq1_markdown(analysis, figures=figures))
    return {"report": str(report_path), "figures": figures}


__all__ = [
    "OptionalRQ1ReportingDependencyError",
    "build_rq1_markdown",
    "generate_stage1_rq1_report",
    "plot_layer_step_heatmap",
    "plot_layer_trajectories",
    "plot_phase_profiles",
]
