#!/usr/bin/env python3
"""Generate paper-ready data tables and optional safety experiment figures."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


class OptionalReportingDependencyError(ImportError):
    """Raised only when a requested figure backend is unavailable."""


def _rows(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return list(value)


def build_heatmap_table(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Layer x PGD-step safety-state table used directly by heatmaps."""

    rows = _rows(analysis.get("layer_step_table"), "layer_step_table")
    required = {"step", "layer", "refusal_degradation"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("heatmap rows require step/layer/refusal_degradation")
    return [
        {
            "step": row["step"],
            "layer": row["layer"],
            "refusal": row.get("refusal"),
            "harmfulness": row.get("harmfulness"),
            "refusal_degradation": row.get("refusal_degradation"),
            "layer_weight": row.get("layer_weight"),
            "is_bottleneck": bool(row.get("is_bottleneck", False)),
        }
        for row in sorted(rows, key=lambda row: (row["step"], str(row["layer"])))
    ]


def build_bottleneck_path_table(
    analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _rows(analysis.get("bottleneck_path"), "bottleneck_path")
    if any("step" not in row or "layer" not in row for row in rows):
        raise ValueError("bottleneck rows require step and layer")
    return [dict(row) for row in sorted(rows, key=lambda row: row["step"])]


def _patching_cases(patching: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Accept generic/Qwen single-case results and Qwen batch summaries."""

    if "trials" in patching:
        return [patching]
    cases = _rows(patching.get("cases"), "patching.cases")
    completed = [case for case in cases if case.get("status") != "failed"]
    if any("trials" not in case for case in completed):
        raise ValueError("completed patching cases require trials")
    return completed


def build_patching_table(patching: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten generic or Qwen activation-patching trials, one trial per row."""

    result = []
    for case in _patching_cases(patching):
        trials = _rows(case.get("trials"), "patching.trials")
        baseline = case.get("baseline_scores", {})
        if not isinstance(baseline, Mapping):
            raise ValueError("baseline_scores must be an object")
        for trial in trials:
            scores = trial.get("scores")
            if scores is None:
                response = trial.get("response")
                scores = (
                    response.get("scores")
                    if isinstance(response, Mapping)
                    else None
                )
            deltas = trial.get("delta")
            if not isinstance(scores, Mapping) or not isinstance(deltas, Mapping):
                raise ValueError(
                    "patching trials require scores (directly or under response) "
                    "and a delta object"
                )
            row = {
                "pair_id": case.get("pair_id"),
                "seed": case.get("seed"),
                "condition": trial.get("condition"),
                "layer": trial.get("layer"),
            }
            for metric, value in baseline.items():
                row[f"baseline_{metric}"] = value
            for metric, value in scores.items():
                row[f"patched_{metric}"] = value
            for metric, value in deltas.items():
                row[f"delta_{metric}"] = value
            result.append(row)
    return result


def build_method_comparison_table(
    statistics_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comparisons = statistics_result.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("statistics result requires comparisons")
    rows = []
    for method in sorted(comparisons):
        comparison = comparisons[method]
        if not isinstance(comparison, Mapping):
            raise ValueError("each comparison must be an object")
        bootstrap = comparison.get("bootstrap_ci", {})
        permutation = comparison.get("permutation_test", {})
        rows.append(
            {
                "baseline_method": comparison.get(
                    "baseline_method", statistics_result.get("baseline_method")
                ),
                "candidate_method": method,
                "metric": comparison.get(
                    "metric", statistics_result.get("metric")
                ),
                "status": comparison.get("status"),
                "n_pairs": comparison.get("n_pairs"),
                "n_observations": comparison.get("n_observations"),
                "mean_baseline": comparison.get("mean_baseline"),
                "mean_candidate": comparison.get("mean_candidate"),
                "mean_raw_difference": comparison.get("mean_raw_difference"),
                "mean_improvement": comparison.get("mean_improvement"),
                "ci_low": bootstrap.get("low"),
                "ci_high": bootstrap.get("high"),
                "confidence": bootstrap.get("confidence"),
                "paired_cohens_dz": comparison.get(
                    "effect_size_paired_cohens_dz"
                ),
                "permutation_p_value": permutation.get("p_value"),
                "missing_baseline": comparison.get("missing_baseline"),
                "missing_candidate": comparison.get("missing_candidate"),
            }
        )
    return rows


def build_quality_tradeoff_table(
    quality_result: Mapping[str, Any],
    *,
    behavior_cases: Optional[Iterable[Mapping[str, Any]]] = None,
) -> list[dict[str, Any]]:
    cases = _rows(quality_result.get("cases"), "quality.cases")
    behavior_index: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for row in behavior_cases or ():
        budget = row.get("budget")
        behavior_seed = (
            row.get("seed")
            if "seed" in row
            else budget.get("seed") if isinstance(budget, Mapping) else None
        )
        key = (row.get("pair_id"), row.get("method"), behavior_seed)
        if key in behavior_index:
            raise ValueError(f"duplicate behavior result for {key!r}")
        behavior_index[key] = row
    rows = []
    for case in cases:
        metrics = case.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("quality case requires metrics")
        key = (case.get("pair_id"), case.get("method"), case.get("seed"))
        behavior = behavior_index.get(key, {})
        strongreject = behavior.get("strongreject")
        nested_score = (
            strongreject.get("avg_score")
            if isinstance(strongreject, Mapping)
            else None
        )
        row = {
            "case_id": case.get("case_id"),
            "pair_id": case.get("pair_id"),
            "method": case.get("method"),
            "seed": case.get("seed"),
            "attack_success": behavior.get(
                "attack_success",
                behavior.get(
                    "reported_attack_success", case.get("attack_success")
                ),
            ),
            "behavior_score": behavior.get(
                "score", behavior.get("strongreject_score", nested_score)
            ),
            "budget_valid": case.get("budget_verification", {}).get("valid"),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def build_report_tables(
    *,
    safety_analysis: Optional[Mapping[str, Any]] = None,
    patching: Optional[Mapping[str, Any]] = None,
    method_statistics: Optional[Mapping[str, Any]] = None,
    audio_quality: Optional[Mapping[str, Any]] = None,
    behavior_cases: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    if safety_analysis is not None:
        tables["safety_heatmap"] = build_heatmap_table(safety_analysis)
        tables["bottleneck_path"] = build_bottleneck_path_table(safety_analysis)
    if patching is not None:
        tables["activation_patching"] = build_patching_table(patching)
    if method_statistics is not None:
        tables["method_comparison"] = build_method_comparison_table(
            method_statistics
        )
    if audio_quality is not None:
        tables["quality_tradeoff"] = build_quality_tradeoff_table(
            audio_quality, behavior_cases=behavior_cases
        )
    if not tables:
        raise ValueError("at least one report input is required")
    return tables


def write_csv_table(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    if not rows:
        raise ValueError("cannot write an empty table")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def _pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise OptionalReportingDependencyError(
            "Figure output requires optional matplotlib; table generation "
            "remains available without it"
        ) from exc


def plot_safety_heatmap(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    plt = _pyplot()
    steps = sorted({row["step"] for row in rows})
    layers = sorted({row["layer"] for row in rows}, key=str)
    lookup = {
        (row["layer"], row["step"]): row.get("refusal_degradation")
        for row in rows
    }
    matrix = [
        [lookup.get((layer, step), float("nan")) for step in steps]
        for layer in layers
    ]
    figure, axis = plt.subplots(figsize=(max(5, len(steps) * 0.6), 4))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set_xticks(range(len(steps)), labels=steps)
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("PGD step")
    axis.set_ylabel("Layer")
    axis.set_title("Refusal degradation: Layer x PGD step")
    figure.colorbar(image, ax=axis, label="Refusal degradation")
    return _save_figure(figure, path, plt)


def plot_bottleneck_path(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    plt = _pyplot()
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    positions = {layer: index for index, layer in enumerate(layers)}
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [row["step"] for row in rows],
        [positions[row["layer"]] for row in rows],
        marker="o",
    )
    axis.set_yticks(range(len(layers)), labels=layers)
    axis.set_xlabel("PGD step")
    axis.set_ylabel("Dynamic bottleneck layer")
    return _save_figure(figure, path, plt)


def plot_activation_patching(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    plt = _pyplot()
    delta_fields = [field for field in rows[0] if field.startswith("delta_")]
    if not delta_fields:
        raise ValueError("patching table has no delta metric")
    metric = delta_fields[0]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        [f"{row['condition']}:{row['layer']}" for row in rows],
        [row[metric] for row in rows],
    )
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=45)
    return _save_figure(figure, path, plt)


def plot_method_comparison(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    plt = _pyplot()
    usable = [row for row in rows if row.get("mean_improvement") is not None]
    if not usable:
        raise ValueError("method comparison table has no completed comparison")
    means = [row["mean_improvement"] for row in usable]
    lower = [mean - row["ci_low"] for mean, row in zip(means, usable)]
    upper = [row["ci_high"] - mean for mean, row in zip(means, usable)]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        [row["candidate_method"] for row in usable],
        means,
        yerr=[lower, upper],
        capsize=4,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Paired improvement")
    axis.tick_params(axis="x", rotation=30)
    return _save_figure(figure, path, plt)


def plot_quality_tradeoff(
    rows: Sequence[Mapping[str, Any]], path: str | Path
) -> Path:
    plt = _pyplot()
    x_field = "spr_db" if "spr_db" in rows[0] else "snr_db"
    y_field = (
        "behavior_score"
        if any(row.get("behavior_score") is not None for row in rows)
        else "attack_success"
    )
    usable = [
        row for row in rows
        if row.get(x_field) is not None and row.get(y_field) is not None
    ]
    if not usable:
        raise ValueError("quality table lacks paired quality/behavior values")
    figure, axis = plt.subplots(figsize=(6, 4))
    for method in sorted({row["method"] for row in usable}):
        selected = [row for row in usable if row["method"] == method]
        axis.scatter(
            [row[x_field] for row in selected],
            [float(row[y_field]) for row in selected],
            label=method,
        )
    axis.set_xlabel(x_field)
    axis.set_ylabel(y_field)
    axis.legend()
    return _save_figure(figure, path, plt)


def _save_figure(figure: Any, path: str | Path, plt: Any) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def generate_report(
    output_dir: str | Path,
    *,
    safety_analysis: Optional[Mapping[str, Any]] = None,
    patching: Optional[Mapping[str, Any]] = None,
    method_statistics: Optional[Mapping[str, Any]] = None,
    audio_quality: Optional[Mapping[str, Any]] = None,
    behavior_cases: Optional[Iterable[Mapping[str, Any]]] = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Write CSV tables, a Markdown index, and requested optional PNG figures."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = build_report_tables(
        safety_analysis=safety_analysis,
        patching=patching,
        method_statistics=method_statistics,
        audio_quality=audio_quality,
        behavior_cases=behavior_cases,
    )
    table_paths = {
        name: str(write_csv_table(rows, output / f"{name}.csv"))
        for name, rows in tables.items()
    }
    figures: dict[str, str] = {}
    if make_plots:
        plotters = {
            "safety_heatmap": plot_safety_heatmap,
            "bottleneck_path": plot_bottleneck_path,
            "activation_patching": plot_activation_patching,
            "method_comparison": plot_method_comparison,
            "quality_tradeoff": plot_quality_tradeoff,
        }
        for name, rows in tables.items():
            figures[name] = str(plotters[name](rows, output / f"{name}.png"))
    markdown = ["# Safety-state experiment report", "", "## Data tables", ""]
    markdown.extend(
        f"- [{name}]({Path(path).name})" for name, path in table_paths.items()
    )
    if figures:
        markdown.extend(["", "## Figures", ""])
        markdown.extend(
            f"![{name}]({Path(path).name})" for name, path in figures.items()
        )
    report_path = output / "report.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    manifest = {
        "format": "safety-state-report",
        "version": 1,
        "tables": table_paths,
        "figures": figures,
        "report": str(report_path),
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


create_report = generate_report


def _load_json(path: Optional[str]) -> Any:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--safety-analysis")
    parser.add_argument("--patching")
    parser.add_argument("--method-statistics")
    parser.add_argument("--audio-quality")
    parser.add_argument("--behavior-cases")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    behavior = _load_json(args.behavior_cases)
    if isinstance(behavior, Mapping):
        behavior = behavior.get("cases", behavior.get("results"))
    generate_report(
        args.output_dir,
        safety_analysis=_load_json(args.safety_analysis),
        patching=_load_json(args.patching),
        method_statistics=_load_json(args.method_statistics),
        audio_quality=_load_json(args.audio_quality),
        behavior_cases=behavior,
        make_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OptionalReportingDependencyError",
    "build_bottleneck_path_table",
    "build_parser",
    "build_heatmap_table",
    "build_method_comparison_table",
    "build_patching_table",
    "build_quality_tradeoff_table",
    "build_report_tables",
    "generate_report",
    "create_report",
    "main",
    "plot_activation_patching",
    "plot_bottleneck_path",
    "plot_method_comparison",
    "plot_quality_tradeoff",
    "plot_safety_heatmap",
    "write_csv_table",
]
