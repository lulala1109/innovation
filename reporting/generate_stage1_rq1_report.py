#!/usr/bin/env python3
"""Generate the Stage-1 RQ1 report and optional cross-sample figures."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
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
    population_counts = source.get("population_audit_counts")
    dual_population = (
        source.get("population_policy") == "symmetric-no-primary-population"
        and isinstance(population_counts, Mapping)
    )
    source_held_out = (
        population_counts.get("all", held_out)
        if dual_population
        else held_out
    )
    train_verified = source.get("probe_training_stage1_provenance_verified") is True
    train_role_ok = (
        source.get("probe_training_measurement_split") == "measurement_train"
        and source.get("probe_training_stage1_role") == "probe_candidate"
    )
    held_out_verified = (
        source.get("trajectory_measurement_split") == "measurement_val"
        and source.get("trajectory_stage1_role") == "trajectory_candidate"
        and source.get("trajectory_num_pairs") == source_held_out
    )
    if (
        isinstance(train_pairs, int)
        and not isinstance(train_pairs, bool)
        and train_verified
        and train_role_ok
        and held_out_verified
    ):
        if dual_population:
            return [
                f"- Provenance-recorded source populations: {train_pairs} "
                f"post-selection probe-training pairs / {source_held_out} "
                f"held-out trajectory pairs; this symmetric view contains "
                f"{held_out} pair(s) (split/role provenance checks passed). "
                "Candidate allocation and the post-selection probe population "
                "are distinct quantities."
            ]
        return [
            f"- Provenance-recorded analysis populations: {train_pairs} "
            f"post-selection probe-training pairs / {held_out} held-out "
            "trajectory pairs (measurement split/role provenance checks passed). "
            "Candidate allocation and the post-selection probe population are "
            "distinct quantities."
        ]
    return [
        f"- Held-out trajectory pairs present: {held_out}. Full training/held-out "
        "analysis-population provenance is not asserted because source provenance "
        "is incomplete."
    ]


def _population_disclosure_lines(analysis: Mapping[str, Any]) -> list[str]:
    metadata = analysis.get("metadata", {})
    source = (
        metadata.get("source_metadata", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    if not isinstance(source, Mapping):
        return []
    if source.get("population_policy") != "symmetric-no-primary-population":
        return []
    counts = source.get("population_audit_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    view = source.get("analysis_population", "unknown")
    trajectory = analysis.get("behavior_trajectory", [])
    available = sum(
        1
        for row in trajectory
        if isinstance(row, Mapping)
        and row.get("continuous_behavior_status") == "available"
    )
    unavailable = sum(
        1
        for row in trajectory
        if isinstance(row, Mapping)
        and row.get("continuous_behavior_status") == "unavailable"
    )
    return [
        "",
        "## Baseline eligibility disclosure",
        "",
        f"- Population view: {view}.",
        "- The all and baseline_refused populations are symmetric analysis "
        "views; neither is designated as the primary analysis.",
        f"- Source held-out cases: {counts.get('all', 0)}; baseline-refused "
        f"eligible: {counts.get('baseline_refused', 0)}.",
        f"- At t=0: unknown={counts.get('t0_unknown', 0)}, "
        f"missing={counts.get('t0_missing', 0)}, "
        f"non-refusal={counts.get('t0_non_refusal', 0)}, "
        f"compliance={counts.get('t0_compliance', 0)}. These categories are "
        "reported separately and may overlap where logically applicable.",
        f"- Continuous behavior trajectory rows: available={available}, "
        f"unavailable={unavailable}. Unavailable v1 values are never inferred "
        "or fabricated.",
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
        *_population_disclosure_lines(analysis),
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


_SENSITIVE_REPORT_FIELDS = frozenset(
    {
        "response",
        "reasoning",
        "overall_reasoning",
        "raw_response",
        "judge_result",
        "judge_error",
        "judge_config",
    }
)


_BEHAVIOR_TRAJECTORY_FIELDS = (
    "population",
    "case_id",
    "pair_id",
    "step",
    "progress",
    "behavior_label",
    "refusal_label",
    "compliance_label",
    "jailbreak_success",
    "refusal_score",
    "strongreject_score",
    "convincingness",
    "specificity",
    "generation_status",
    "label_status",
    "response_sha256",
    "continuous_behavior_status",
)


def _json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return dict(value)


def _csv_mappings(
    path: Path,
    *,
    name: str,
    forbid_sensitive: bool = False,
    expected_fields: Optional[Sequence[str]] = None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if not fields:
                raise ValueError(f"{name} has no header: {path}")
            if expected_fields is not None and fields != tuple(expected_fields):
                raise ValueError(
                    f"{name} fields changed: {path}"
                )
            if forbid_sensitive:
                forbidden = _SENSITIVE_REPORT_FIELDS.intersection(fields)
                if forbidden:
                    raise ValueError(
                        f"{name} contains forbidden sensitive fields: "
                        + ", ".join(sorted(forbidden))
                    )
            return [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"cannot read {name}: {path}") from exc


def load_rq1_analysis_outputs(output_dir: str | Path) -> Mapping[str, Any]:
    """Load the persisted subset needed to render one RQ1 report."""

    output = Path(output_dir).expanduser().resolve()
    analysis = _json_mapping(output / "rq1_summary.json", name="RQ1 summary")
    if (
        analysis.get("format") != "stage1-rq1-analysis"
        or analysis.get("version") != 1
    ):
        raise ValueError(f"unsupported RQ1 analysis summary: {output}")
    axes = analysis.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError(f"RQ1 analysis summary has no axes mapping: {output}")
    for table in ("cell_statistics", "phase_profiles"):
        analysis[table] = _csv_mappings(
            output / f"{table}.csv",
            name=table,
        )
    behavior_rows = _csv_mappings(
        output / "behavior_trajectory.csv",
        name="behavior_trajectory",
        forbid_sensitive=True,
        expected_fields=_BEHAVIOR_TRAJECTORY_FIELDS,
    )
    analysis["behavior_trajectory"] = behavior_rows

    case_values = axes.get("case_ids")
    pair_values = axes.get("pair_ids")
    step_values = axes.get("steps")
    for name, values in (
        ("case_ids", case_values),
        ("pair_ids", pair_values),
        ("steps", step_values),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"RQ1 axes.{name} must be a sequence: {output}")
    case_ids = tuple(str(value) for value in case_values)
    pair_ids = tuple(str(value) for value in pair_values)
    steps = tuple(int(value) for value in step_values)
    if len(case_ids) != len(pair_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError(f"RQ1 case/pair axes are inconsistent: {output}")
    pair_by_case = dict(zip(case_ids, pair_ids))
    expected_grid = {(case_id, step) for case_id in case_ids for step in steps}
    seen: set[tuple[str, int]] = set()
    metadata = analysis.get("metadata")
    source = (
        metadata.get("source_metadata")
        if isinstance(metadata, Mapping)
        else None
    )
    expected_population = (
        source.get("analysis_population")
        if isinstance(source, Mapping)
        else None
    )
    for row in behavior_rows:
        case_id = row["case_id"]
        try:
            step = int(row["step"])
            progress = float(row["progress"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"behavior trajectory has invalid step/progress: {output}"
            ) from exc
        identity = (case_id, step)
        if identity in seen:
            raise ValueError(f"duplicate behavior trajectory identity: {identity!r}")
        seen.add(identity)
        if (
            pair_by_case.get(case_id) != row["pair_id"]
            or row["label_status"] not in {"ok", "unknown", "missing"}
            or row["continuous_behavior_status"] not in {
                "available",
                "unavailable",
            }
            or not math.isfinite(progress)
            or (
                expected_population is not None
                and row["population"] != expected_population
            )
        ):
            raise ValueError(
                f"behavior trajectory identity/status mismatch: {identity!r}"
            )
    if seen != expected_grid:
        raise ValueError(f"behavior trajectory is not a complete case×step grid: {output}")
    mixed_path = output / "mixed_effects.json"
    if mixed_path.is_file():
        with mixed_path.open("r", encoding="utf-8") as handle:
            analysis["mixed_effects"] = json.load(handle)
    else:
        analysis["mixed_effects"] = None
    return analysis


def generate_stage1_rq1_reports_from_outputs(
    analysis_dir: str | Path,
    *,
    population: str = "all",
    make_plots: bool = False,
) -> Mapping[str, Any]:
    """Render reports from completed analysis artifacts without recomputing them."""

    root = Path(analysis_dir).expanduser().resolve()
    if population == "all":
        return generate_stage1_rq1_report(
            load_rq1_analysis_outputs(root),
            root,
            make_plots=make_plots,
        )
    if population != "both":
        raise ValueError("population must be 'all' or 'both'")

    audit_path = root / "population_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    index_path = root / "population_index.json"
    index = _json_mapping(index_path, name="population index")
    policy = index.get("population_policy")
    entries = index.get("populations")
    if (
        index.get("format") != "stage1-rq1-population-analysis"
        or index.get("version") != 1
    ):
        raise ValueError("unsupported population index format/version")
    if (
        not isinstance(policy, Mapping)
        or policy.get("primary_population") is not None
        or policy.get("comparison") != "symmetric-no-primary-population"
    ):
        raise ValueError("population index must keep symmetric no-primary policy")
    if not isinstance(entries, Mapping) or set(entries) != {
        "all",
        "baseline_refused",
    }:
        raise ValueError("population index must contain both symmetric views")

    updated_entries: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for name in ("all", "baseline_refused"):
        entry = entries[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"population index entry {name!r} must be an object")
        destination = root / name
        declared_output = Path(str(entry.get("output_dir", ""))).expanduser()
        if not declared_output.is_absolute():
            declared_output = root / declared_output
        if declared_output.resolve() != destination:
            raise ValueError(f"population index output path mismatch for {name}")
        analysis = load_rq1_analysis_outputs(destination)
        axes = analysis.get("axes", {})
        case_ids = list(axes.get("case_ids", []))
        pair_ids = list(axes.get("pair_ids", []))
        if (
            entry.get("case_count") != len(case_ids)
            or list(entry.get("case_ids", [])) != case_ids
            or list(entry.get("pair_ids", [])) != pair_ids
        ):
            raise ValueError(f"population index identity mismatch for {name}")
        generated = generate_stage1_rq1_report(
            analysis,
            destination,
            make_plots=(
                make_plots and bool(analysis.get("cell_statistics"))
            ),
        )
        artifacts = entry.get("artifacts")
        merged_artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
        merged_artifacts.update(generated)
        updated_entry = dict(entry)
        updated_entry["artifacts"] = merged_artifacts
        updated_entries[name] = updated_entry
        reports[name] = generated

    updated_index = dict(index)
    updated_index["populations"] = updated_entries
    _atomic_text(
        index_path,
        json.dumps(
            updated_index,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return {
        "population_index": str(index_path),
        "populations": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument(
        "--population",
        choices=("all", "both"),
        default="all",
        help="Render one legacy all-case report or both symmetric population reports",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also render optional matplotlib H/R figures",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = generate_stage1_rq1_reports_from_outputs(
        args.analysis_dir,
        population=args.population,
        make_plots=args.make_plots,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OptionalRQ1ReportingDependencyError",
    "build_parser",
    "build_rq1_markdown",
    "generate_stage1_rq1_report",
    "generate_stage1_rq1_reports_from_outputs",
    "load_rq1_analysis_outputs",
    "main",
    "plot_layer_step_heatmap",
    "plot_layer_trajectories",
    "plot_phase_profiles",
]
