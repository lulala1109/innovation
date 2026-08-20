#!/usr/bin/env python3
"""Coordinate causal activation-patching trials and random-layer controls.

The pure coordinator accepts an already-loaded model, forward closure, source
activations, and scorer.  The CLI obtains those objects from an injected factory
(``module:callable``), so this generic module never guesses how to load a model
or how to continue a model-specific forward pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import numbers
import os
import random
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Sequence


LayerKey = Hashable


def _unique(values: Iterable[LayerKey], *, name: str) -> tuple[LayerKey, ...]:
    result: list[LayerKey] = []
    seen: set[LayerKey] = set()
    for value in values:
        try:
            duplicate = value in seen
        except TypeError as exc:
            raise TypeError(f"{name} values must be hashable") from exc
        if not duplicate:
            result.append(value)
            seen.add(value)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return tuple(result)


def build_patch_plan(
    available_layers: Iterable[LayerKey],
    critical_layers: Iterable[LayerKey],
    *,
    random_control_count: int = 1,
    seed: int = 42,
    random_layers: Optional[Iterable[LayerKey]] = None,
) -> tuple[dict[str, Any], ...]:
    """Build a reproducible, non-overlapping critical/control trial plan."""

    available = _unique(available_layers, name="available_layers")
    critical = _unique(critical_layers, name="critical_layers")
    missing = [layer for layer in critical if layer not in available]
    if missing:
        raise KeyError(f"Critical layer(s) are unavailable: {missing}")
    if (
        isinstance(random_control_count, bool)
        or not isinstance(random_control_count, int)
        or random_control_count < 0
    ):
        raise ValueError("random_control_count must be a non-negative integer")

    candidates = [layer for layer in available if layer not in critical]
    if random_layers is None:
        if random_control_count > len(candidates):
            raise ValueError(
                f"Requested {random_control_count} random controls but only "
                f"{len(candidates)} non-critical layers are available"
            )
        controls = tuple(random.Random(seed).sample(candidates, random_control_count))
    else:
        controls = _unique(random_layers, name="random_layers")
        invalid = [layer for layer in controls if layer not in candidates]
        if invalid:
            raise ValueError(
                f"Random controls must be available non-critical layers: {invalid}"
            )
        if random_control_count != len(controls):
            raise ValueError(
                "random_control_count must equal the number of explicit random_layers"
            )

    plan = [
        {"condition": "critical", "layer": layer} for layer in critical
    ]
    plan.extend(
        {"condition": "random_control", "layer": layer} for layer in controls
    )
    return tuple(plan)


def _score_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        raw = value
    else:
        raw = {"score": value}
    result: dict[str, float] = {}
    for key, score in raw.items():
        if isinstance(score, bool):
            numeric = float(score)
        elif isinstance(score, numbers.Real):
            numeric = float(score)
        elif hasattr(score, "detach") and getattr(score, "numel", lambda: 0)() == 1:
            numeric = float(score.detach().cpu().item())
        else:
            raise TypeError(f"score {key!r} must be a real scalar")
        if not math.isfinite(numeric):
            raise ValueError(f"score {key!r} is not finite")
        result[str(key)] = numeric
    if not result:
        raise ValueError("score_fn returned no metrics")
    return result


def coordinate_activation_patching(
    *,
    model: Any,
    layer_modules: Mapping[LayerKey, Any],
    source_activations: Mapping[LayerKey, Any],
    forward_fn: Callable[[], Any],
    score_fn: Callable[[Any], Any],
    critical_layers: Iterable[LayerKey],
    random_control_count: int = 1,
    seed: int = 42,
    random_layers: Optional[Iterable[LayerKey]] = None,
    token_selection: Any = None,
    output_selector: Optional[int | str] = None,
    detach_replacement: bool = True,
    patcher_class: Optional[type] = None,
) -> dict[str, Any]:
    """Patch one layer per trial and compare scores with an unpatched forward.

    ``forward_fn`` must be a zero-argument closure that performs the jailbreak
    forward/generation through ``model``. ``source_activations[layer]`` should be
    the paired refused-harmful activation for that same layer.
    """

    if not isinstance(layer_modules, Mapping) or not layer_modules:
        raise ValueError("layer_modules must map layer identifiers to modules/names")
    if not isinstance(source_activations, Mapping):
        raise TypeError("source_activations must be a mapping")
    plan = build_patch_plan(
        layer_modules.keys(),
        critical_layers,
        random_control_count=random_control_count,
        seed=seed,
        random_layers=random_layers,
    )
    missing_sources = [
        trial["layer"] for trial in plan if trial["layer"] not in source_activations
    ]
    if missing_sources:
        raise KeyError(f"Missing source activation(s) for layer(s): {missing_sources}")

    if patcher_class is None:
        from core.activations import ActivationPatcher

        patcher_class = ActivationPatcher
    from core.activations import ActivationPatch

    baseline = _score_mapping(score_fn(forward_fn()))
    trials: list[dict[str, Any]] = []
    for trial in plan:
        layer = trial["layer"]
        patch = ActivationPatch(
            replacement=source_activations[layer],
            token_selection=token_selection,
            output_selector=output_selector,
            detach_replacement=detach_replacement,
        )
        with patcher_class({layer_modules[layer]: patch}, model=model):
            patched_scores = _score_mapping(score_fn(forward_fn()))
        if set(patched_scores) != set(baseline):
            raise ValueError(
                "score_fn must return the same metric keys for baseline and patches"
            )
        trials.append(
            {
                "condition": trial["condition"],
                "layer": layer,
                "scores": patched_scores,
                "delta": {
                    key: patched_scores[key] - baseline[key] for key in baseline
                },
            }
        )

    aggregates: dict[str, Any] = {}
    for condition in ("critical", "random_control"):
        selected = [trial for trial in trials if trial["condition"] == condition]
        aggregates[condition] = {
            "count": len(selected),
            "mean_delta": {
                metric: (
                    fmean(trial["delta"][metric] for trial in selected)
                    if selected
                    else None
                )
                for metric in baseline
            },
        }
    aggregates["critical_minus_random"] = {
        metric: (
            aggregates["critical"]["mean_delta"][metric]
            - aggregates["random_control"]["mean_delta"][metric]
            if aggregates["critical"]["mean_delta"][metric] is not None
            and aggregates["random_control"]["mean_delta"][metric] is not None
            else None
        )
        for metric in baseline
    }
    return {
        "format": "activation-patching-controls",
        "version": 1,
        "seed": seed,
        "baseline_scores": baseline,
        "trials": trials,
        "aggregates": aggregates,
    }


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


def _parse_layers(value: str) -> tuple[LayerKey, ...]:
    result: list[LayerKey] = []
    for piece in value.split(","):
        text = piece.strip()
        if text:
            try:
                result.append(int(text))
            except ValueError:
                result.append(text)
    if not result:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return tuple(result)


def _load_factory(specification: str) -> Callable[[argparse.Namespace], Mapping[str, Any]]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use the form 'module:callable'")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{specification!r} does not resolve to a callable")
    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory",
        default=None,
        help="module:callable returning model/layer_modules/source_activations/forward_fn/score_fn",
    )
    parser.add_argument("--critical-layers", type=_parse_layers, required=True)
    parser.add_argument("--random-control-count", type=int, default=1)
    parser.add_argument("--random-layers", type=_parse_layers, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-start", type=int, default=None)
    parser.add_argument("--token-end", type=int, default=None)
    parser.add_argument("--output-selector", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    experiment_factory: Optional[
        Callable[[argparse.Namespace], Mapping[str, Any]]
    ] = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if experiment_factory is None:
        if args.factory is None:
            parser.error(
                "--factory is required: generic activation patching does not guess "
                "how to load a model or continue its forward pass"
            )
        experiment_factory = _load_factory(args.factory)
    context = experiment_factory(args)
    if not isinstance(context, Mapping):
        raise TypeError("experiment factory must return a mapping of coordinator inputs")
    kwargs = dict(context)
    kwargs.update(
        {
            "critical_layers": args.critical_layers,
            "random_control_count": args.random_control_count,
            "seed": args.seed,
            "random_layers": args.random_layers,
            "token_selection": (
                None
                if args.token_start is None and args.token_end is None
                else (args.token_start, args.token_end)
            ),
        }
    )
    if args.output_selector is not None:
        try:
            kwargs["output_selector"] = int(args.output_selector)
        except ValueError:
            kwargs["output_selector"] = args.output_selector
    result = coordinate_activation_patching(**kwargs)
    _atomic_json(Path(args.output).resolve(), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "build_patch_plan",
    "coordinate_activation_patching",
    "main",
]
