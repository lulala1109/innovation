#!/usr/bin/env python3
"""Analyze Layer x PGD-step safety-state trajectories without plotting deps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


STATE_ALIASES = {
    "refusal": ("refusal", "refusal_scores", "refusal_state"),
    "harmfulness": ("harmfulness", "harmfulness_scores", "harmfulness_state"),
    "layer_weights": ("layer_weights", "weights", "safety_layer_weights"),
    "degradation": (
        "safety_gaps",
        "refusal_degradation",
        "refusal_degradations",
        "degradation",
    ),
    "reference_refusal": (
        "reference_refusal",
        "reference_refusal_scores",
        "clean_refusal",
    ),
}
TABLE_FIELDS = (
    "step",
    "layer",
    "refusal",
    "harmfulness",
    "layer_weight",
    "reference_refusal",
    "refusal_degradation",
    "is_bottleneck",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _resolve_index(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).resolve()
    if candidate.is_file():
        if candidate.name != "index.json":
            raise ValueError("Trajectory file must be named index.json")
        index_path = candidate
    elif (candidate / "index.json").is_file():
        index_path = candidate / "index.json"
    elif (candidate / "trajectory" / "index.json").is_file():
        index_path = candidate / "trajectory" / "index.json"
    else:
        raise FileNotFoundError(f"No trajectory/index.json below {candidate}")
    trajectory_dir = index_path.parent
    case_root = trajectory_dir.parent if trajectory_dir.name == "trajectory" else trajectory_dir
    return index_path, case_root


def _read_index(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    checkpoints = value.get("checkpoints") if isinstance(value, dict) else None
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"{path}: missing non-empty checkpoints list")
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for item in checkpoints:
        if not isinstance(item, dict) or isinstance(item.get("step"), bool):
            raise ValueError(f"{path}: malformed checkpoint entry")
        step = int(item["step"])
        if step < 0 or step in seen:
            raise ValueError(f"{path}: invalid or duplicate step {step}")
        seen.add(step)
        result.append(item)
    return sorted(result, key=lambda item: int(item["step"]))


def _load_tensor_payload(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: checkpoint is not a tensor mapping")
    return value


def _layer_sort_key(layer: Any) -> tuple[int, Any]:
    if isinstance(layer, bool):
        return (2, str(layer))
    if isinstance(layer, int):
        return (0, layer)
    text = str(layer)
    if re.fullmatch(r"-?\d+", text):
        return (0, int(text))
    return (1, text)


def _normalize_layer(layer: Any) -> Any:
    if isinstance(layer, bool):
        return str(layer)
    if isinstance(layer, int):
        return layer
    text = str(layer)
    return int(text) if re.fullmatch(r"-?\d+", text) else text


def _finite_float(value: Any, *, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _as_layer_map(value: Any, *, name: str) -> dict[Any, float]:
    """Convert mapping/vector/batched-vector state values to layer scalars."""

    if isinstance(value, Mapping):
        result = {
            _normalize_layer(layer): _finite_float(score, name=name)
            for layer, score in value.items()
        }
    elif hasattr(value, "detach") and hasattr(value, "ndim"):
        tensor = value.detach().to(device="cpu", dtype=getattr(value, "dtype", None))
        if not (tensor.is_floating_point() or tensor.is_complex()):
            tensor = tensor.float()
        if tensor.is_complex():
            raise ValueError(f"{name} must be real-valued")
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim > 1:
            tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
        result = {
            index: _finite_float(score, name=name)
            for index, score in enumerate(tensor.reshape(-1))
        }
    elif isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            width = len(value[0])
            if any(len(row) != width for row in value):
                raise ValueError(f"{name} has ragged batched values")
            result = {
                index: sum(_finite_float(row[index], name=name) for row in value)
                / len(value)
                for index in range(width)
            }
        else:
            result = {
                index: _finite_float(score, name=name)
                for index, score in enumerate(value)
            }
    else:
        result = {0: _finite_float(value, name=name)}
    if not result:
        raise ValueError(f"{name} is empty")
    return dict(sorted(result.items(), key=lambda item: _layer_sort_key(item[0])))


def _prefixed_layer_values(
    source: Mapping[str, Any], aliases: Iterable[str], *, name: str
) -> Optional[dict[Any, float]]:
    values: dict[Any, float] = {}
    for alias in aliases:
        expression = re.compile(
            rf"^{re.escape(alias)}(?:[./:_-]+(?:layer)?[./:_-]*)?(-?\d+)$",
            flags=re.IGNORECASE,
        )
        for key, value in source.items():
            match = expression.match(str(key))
            if match:
                values[int(match.group(1))] = _finite_float(value, name=name)
    return dict(sorted(values.items())) if values else None


def _extract_state(
    payload: Mapping[str, Any], metadata: Mapping[str, Any], name: str
) -> Optional[dict[Any, float]]:
    aliases = STATE_ALIASES[name]
    for source in (payload, metadata):
        for alias in aliases:
            if alias in source and source[alias] is not None:
                return _as_layer_map(source[alias], name=name)
        prefixed = _prefixed_layer_values(source, aliases, name=name)
        if prefixed is not None:
            return prefixed
    return None


def _require_layer_maps_close(
    expected: Mapping[Any, float],
    actual: Mapping[Any, float],
    *,
    name: str,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-7,
) -> None:
    """Validate one reference reconstructed at multiple checkpoints."""

    if set(expected) != set(actual):
        raise ValueError(f"{name} layers change across checkpoints")
    inconsistent = [
        layer
        for layer in expected
        if not math.isclose(
            expected[layer], actual[layer], rel_tol=rel_tol, abs_tol=abs_tol
        )
    ]
    if inconsistent:
        raise ValueError(
            f"{name} changes across checkpoints at layer(s) "
            f"{sorted(inconsistent, key=_layer_sort_key)}"
        )


def _checkpoint_path(
    item: Mapping[str, Any], *, index_path: Path, case_root: Path
) -> Path:
    raw = item.get("path")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = case_root / path
    else:
        path = index_path.parent / f"step_{int(item['step']):06d}.pt"
    if not path.is_file():
        # Some external writers make paths relative to the trajectory dir.
        fallback = index_path.parent / Path(str(raw or path.name)).name
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Checkpoint listed in index does not exist: {path}")
    return path


def analyze_trajectory(
    trajectory: str | Path,
    *,
    min_unique_bottlenecks: int = 2,
    min_bottleneck_switches: int = 1,
    min_degradation_range: float = 0.0,
    require_all_states: bool = False,
) -> dict[str, Any]:
    """Return a Layer x Step table, bottleneck path, and Go/No-Go decision.

    Refusal degradation is ``reference_refusal - current_refusal``.  An explicit
    ``reference_refusal`` tensor is preferred.  Otherwise recorded ``safety_gaps``
    reconstruct it as ``current_refusal + gap``; the earliest checkpoint is used
    only when neither source is available.
    """

    if min_unique_bottlenecks < 1 or min_bottleneck_switches < 0:
        raise ValueError("Go/No-Go count thresholds must be non-negative")
    if min_degradation_range < 0:
        raise ValueError("min_degradation_range must be non-negative")
    index_path, case_root = _resolve_index(trajectory)
    checkpoint_items = _read_index(index_path)

    states: list[dict[str, Any]] = []
    explicit_reference: Optional[dict[Any, float]] = None
    gap_derived_reference: Optional[dict[Any, float]] = None
    for item in checkpoint_items:
        path = _checkpoint_path(item, index_path=index_path, case_root=case_root)
        payload = _load_tensor_payload(path)
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"step {item['step']}: metadata must be an object")
        refusal = _extract_state(payload, metadata, "refusal")
        harmfulness = _extract_state(payload, metadata, "harmfulness")
        layer_weights = _extract_state(payload, metadata, "layer_weights")
        recorded_degradation = _extract_state(payload, metadata, "degradation")
        reference = _extract_state(payload, metadata, "reference_refusal")
        if refusal is None:
            raise ValueError(f"step {item['step']}: refusal state is required")
        if require_all_states and (harmfulness is None or layer_weights is None):
            raise ValueError(
                f"step {item['step']}: harmfulness and layer_weights are required"
            )
        if reference is not None:
            if explicit_reference is not None:
                _require_layer_maps_close(
                    explicit_reference,
                    reference,
                    name="reference_refusal",
                )
            explicit_reference = reference
        if recorded_degradation is not None:
            if set(recorded_degradation) != set(refusal):
                raise ValueError(
                    f"step {item['step']}: safety_gaps and refusal layers do not match"
                )
            reconstructed = {
                layer: refusal[layer] + recorded_degradation[layer]
                for layer in refusal
            }
            if gap_derived_reference is not None:
                _require_layer_maps_close(
                    gap_derived_reference,
                    reconstructed,
                    name="safety_gaps-derived reference_refusal",
                )
            gap_derived_reference = reconstructed
        states.append(
            {
                "step": int(item["step"]),
                "refusal": refusal,
                "harmfulness": harmfulness or {},
                "layer_weights": layer_weights or {},
            }
        )

    reference_source = "checkpoint_tensor"
    if explicit_reference is not None and gap_derived_reference is not None:
        _require_layer_maps_close(
            explicit_reference,
            gap_derived_reference,
            name="explicit and safety_gaps-derived reference_refusal",
        )
    if explicit_reference is None:
        if gap_derived_reference is not None:
            explicit_reference = gap_derived_reference
            reference_source = "safety_gaps_plus_refusal"
        else:
            explicit_reference = dict(states[0]["refusal"])
            reference_source = f"earliest_checkpoint_step_{states[0]['step']}"
    reference = explicit_reference

    table: list[dict[str, Any]] = []
    bottleneck_path: list[dict[str, Any]] = []
    all_degradations: list[float] = []
    for state in states:
        refusal = state["refusal"]
        missing = set(refusal) - set(reference)
        if missing:
            raise ValueError(
                f"step {state['step']}: reference refusal lacks layer(s) {sorted(missing, key=_layer_sort_key)}"
            )
        degradations = {
            layer: reference[layer] - score for layer, score in refusal.items()
        }
        bottleneck_layer = max(degradations, key=degradations.__getitem__)
        # Python's max keeps the first equal value; refusal maps are sorted, so
        # ties deterministically select the shallowest/natural-first layer.
        max_gap = degradations[bottleneck_layer]
        bottleneck_path.append(
            {
                "step": state["step"],
                "layer": bottleneck_layer,
                "refusal_degradation": max_gap,
                "refusal": refusal[bottleneck_layer],
                "harmfulness": state["harmfulness"].get(bottleneck_layer),
                "layer_weight": state["layer_weights"].get(bottleneck_layer),
            }
        )
        layers = sorted(
            set(refusal) | set(state["harmfulness"]) | set(state["layer_weights"]),
            key=_layer_sort_key,
        )
        for layer in layers:
            gap = degradations.get(layer)
            if gap is not None:
                all_degradations.append(gap)
            table.append(
                {
                    "step": state["step"],
                    "layer": layer,
                    "refusal": refusal.get(layer),
                    "harmfulness": state["harmfulness"].get(layer),
                    "layer_weight": state["layer_weights"].get(layer),
                    "reference_refusal": reference.get(layer),
                    "refusal_degradation": gap,
                    "is_bottleneck": layer == bottleneck_layer,
                }
            )

    path_layers = [item["layer"] for item in bottleneck_path]
    unique_layers = sorted(set(path_layers), key=_layer_sort_key)
    switches = sum(left != right for left, right in zip(path_layers, path_layers[1:]))
    degradation_range = max(all_degradations) - min(all_degradations)
    is_dynamic = (
        len(unique_layers) >= min_unique_bottlenecks
        and switches >= min_bottleneck_switches
        and degradation_range >= min_degradation_range
    )
    frequencies = Counter(path_layers)
    numeric_layers = [layer for layer in unique_layers if isinstance(layer, int)]
    critical_window = {
        "layers": unique_layers,
        "start_layer": min(numeric_layers) if len(numeric_layers) == len(unique_layers) else None,
        "end_layer": max(numeric_layers) if len(numeric_layers) == len(unique_layers) else None,
        "frequency": {str(layer): frequencies[layer] for layer in unique_layers},
    }
    decision = "go_layer_adaptive" if is_dynamic else "no_go_critical_window"
    return {
        "format": "safety-state-dynamics-analysis",
        "version": 1,
        "trajectory_index": str(index_path),
        "reference_refusal_source": reference_source,
        "steps": [state["step"] for state in states],
        "layers": sorted(reference, key=_layer_sort_key),
        "layer_step_table": table,
        "bottleneck_path": bottleneck_path,
        "go_no_go": {
            "decision": decision,
            "dynamic_migration": is_dynamic,
            "unique_bottleneck_layers": unique_layers,
            "bottleneck_switches": switches,
            "switch_rate": switches / max(1, len(path_layers) - 1),
            "degradation_range": degradation_range,
            "thresholds": {
                "min_unique_bottlenecks": min_unique_bottlenecks,
                "min_bottleneck_switches": min_bottleneck_switches,
                "min_degradation_range": min_degradation_range,
            },
            "critical_window": None if is_dynamic else critical_window,
        },
    }


def write_analysis(
    result: Mapping[str, Any], *, json_path: str | Path, csv_path: str | Path
) -> tuple[Path, Path]:
    json_output = Path(json_path).resolve()
    csv_output = Path(csv_path).resolve()
    rows = result.get("layer_step_table")
    if not isinstance(rows, list):
        raise ValueError("analysis result lacks layer_step_table")
    _atomic_json(json_output, result)
    _atomic_csv(csv_output, rows)
    return json_output, csv_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="Case dir, trajectory dir, or index.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-unique-bottlenecks", type=int, default=2)
    parser.add_argument("--min-bottleneck-switches", type=int, default=1)
    parser.add_argument("--min-degradation-range", type=float, default=0.0)
    parser.add_argument("--require-all-states", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_trajectory(
        args.trajectory,
        min_unique_bottlenecks=args.min_unique_bottlenecks,
        min_bottleneck_switches=args.min_bottleneck_switches,
        min_degradation_range=args.min_degradation_range,
        require_all_states=args.require_all_states,
    )
    write_analysis(result, json_path=args.output_json, csv_path=args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STATE_ALIASES",
    "TABLE_FIELDS",
    "analyze_trajectory",
    "build_parser",
    "main",
    "write_analysis",
]
