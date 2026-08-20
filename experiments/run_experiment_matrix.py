#!/usr/bin/env python3
"""Expand and optionally execute fair white-box audio attack matrices.

The runner is JSON-configured and dry-runs by default. It expands the six
canonical attack methods over deterministic seeds and named ablation sweeps,
then validates that every job has the same optimization budget. Real execution
delegates each job to experiments.batch_safety_attack.run_batch.

Ablations are one-factor-at-a-time by default. Cartesian expansion is also
available for deliberately small grids. Checkpoint sweeps may vary only the
observation schedule; they cannot alter the optimization budget.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


METHODS = (
    "standard",
    "fixed",
    "uniform",
    "static_topk",
    "gradient_adaptive",
    "safety_state_adaptive",
)
ABLATION_AXES = (
    "measurement",
    "dynamic",
    "layer",
    "tau",
    "lambda",
    "checkpoint",
)
FAIR_BUDGET_FIELDS = (
    "eps",
    "alpha",
    "steps",
    "loss_type",
    "kappa",
    "init_mode",
    "check_every",
    "early_stop",
)
CONTROLLED_FIELDS = frozenset(
    {"manifest_path", "output_dir", "method", "seed"}
)
RUN_BATCH_FIELDS = frozenset(
    {
        "model_name",
        "model_id",
        "target_text",
        "device",
        "dtype",
        "eps",
        "alpha",
        "steps",
        "loss_type",
        "kappa",
        "init_mode",
        "check_every",
        "early_stop",
        "checkpoint_steps",
        "determinism",
        "probe_checkpoint",
        "reference_refusal_path",
        "layers",
        "fixed_layer",
        "top_k",
        "static_topk_layers",
        "state_loss_weight",
        "temperature",
        "token_span",
        "pooling",
        "verbose",
        "fail_fast",
    }
)
METHOD_OVERRIDE_FIELDS = frozenset(
    {
        "layers",
        "fixed_layer",
        "top_k",
        "static_topk_layers",
        "state_loss_weight",
        "temperature",
        "token_span",
        "pooling",
    }
)
AXIS_FIELDS = {
    "measurement": frozenset(
        {"token_span", "pooling", "probe_checkpoint"}
    ),
    "dynamic": frozenset(
        {"method", "fixed_layer", "top_k", "static_topk_layers", "layers"}
    ),
    "layer": frozenset(
        {"layers", "fixed_layer", "top_k", "static_topk_layers"}
    ),
    "tau": frozenset({"temperature"}),
    "lambda": frozenset({"state_loss_weight"}),
    "checkpoint": frozenset({"checkpoint_steps"}),
}
PATH_OPTION_FIELDS = frozenset(
    {"probe_checkpoint", "reference_refusal_path"}
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
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


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key.strip()] = item
    return result


def _unknown_keys(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    name: str,
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{name} has unsupported field(s): {unknown}")


def _finite_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    valid = numeric >= minimum if inclusive else numeric > minimum
    if not math.isfinite(numeric) or not valid:
        qualifier = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be finite and {qualifier} {minimum}")
    return numeric


def load_matrix_config(path: str | Path) -> dict[str, Any]:
    """Load a local JSON matrix configuration without importing model code."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Matrix config not found: {config_path}")
    if config_path.suffix.casefold() != ".json":
        raise ValueError("Matrix configuration must be JSON")
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Matrix configuration must contain one JSON object")
    return value


def _validate_budget(value: Any) -> dict[str, Any]:
    budget = _mapping(value, name="budget")
    missing = sorted(set(FAIR_BUDGET_FIELDS) - set(budget))
    if missing:
        raise ValueError(f"budget is missing fair-budget field(s): {missing}")
    _unknown_keys(budget, FAIR_BUDGET_FIELDS, name="budget")

    budget["eps"] = _finite_number(
        budget["eps"],
        name="budget.eps",
        minimum=0.0,
        inclusive=True,
    )
    budget["alpha"] = _finite_number(
        budget["alpha"],
        name="budget.alpha",
        minimum=0.0,
        inclusive=False,
    )
    steps = budget["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("budget.steps must be a positive integer")
    if (
        not isinstance(budget["loss_type"], str)
        or budget["loss_type"] not in {"ce", "margin"}
    ):
        raise ValueError("budget.loss_type must be ce or margin")
    budget["kappa"] = _finite_number(
        budget["kappa"],
        name="budget.kappa",
        minimum=0.0,
        inclusive=True,
    )
    if (
        not isinstance(budget["init_mode"], str)
        or budget["init_mode"] not in {"zero", "random"}
    ):
        raise ValueError("budget.init_mode must be zero or random")
    check_every = budget["check_every"]
    if (
        isinstance(check_every, bool)
        or not isinstance(check_every, int)
        or check_every < 0
    ):
        raise ValueError("budget.check_every must be a non-negative integer")
    if budget["early_stop"] is not False:
        raise ValueError(
            "budget.early_stop must be false so every method receives the "
            "same number of optimization updates"
        )
    return budget


def _validate_seeds(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError("seeds must be a non-empty JSON array")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed >= 2**32
        ):
            raise ValueError(
                f"seeds[{index}] must be an integer within [0, 2**32)"
            )
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    return tuple(seeds)


def _validate_methods(value: Any) -> dict[str, dict[str, Any]]:
    methods = _mapping(value, name="methods")
    missing = sorted(set(METHODS) - set(methods))
    extra = sorted(set(methods) - set(METHODS))
    if missing or extra:
        raise ValueError(
            "methods must define exactly the six canonical methods; "
            f"missing={missing}, extra={extra}"
        )

    result: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        overrides = _mapping(
            methods[method],
            name=f"methods.{method}",
        )
        locked = sorted(set(overrides) & set(FAIR_BUDGET_FIELDS))
        if locked:
            raise ValueError(
                f"methods.{method} overrides locked fair-budget fields: {locked}"
            )
        _unknown_keys(
            overrides,
            METHOD_OVERRIDE_FIELDS,
            name=f"methods.{method}",
        )
        result[method] = overrides
    return result


def _validate_apply_to(value: Any, *, axis: str) -> tuple[str, ...]:
    if value is None:
        return METHODS
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"sweeps.axes.{axis}.apply_to must be a non-empty array")
    methods = tuple(_text(item, name=f"{axis}.apply_to") for item in value)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"{axis}.apply_to has unknown methods: {unknown}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"{axis}.apply_to contains duplicates")
    return methods


def _validate_sweeps(value: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    sweeps = _mapping(value, name="sweeps")
    _unknown_keys(sweeps, {"mode", "axes"}, name="sweeps")
    mode = sweeps.get("mode", "one_factor")
    if mode not in {"one_factor", "cartesian"}:
        raise ValueError("sweeps.mode must be one_factor or cartesian")
    axes = _mapping(sweeps.get("axes", {}), name="sweeps.axes")
    unknown_axes = sorted(set(axes) - set(ABLATION_AXES))
    if unknown_axes:
        raise ValueError(f"Unknown ablation axes: {unknown_axes}")

    normalized: dict[str, dict[str, Any]] = {}
    for axis in ABLATION_AXES:
        if axis not in axes:
            continue
        spec = _mapping(axes[axis], name=f"sweeps.axes.{axis}")
        _unknown_keys(
            spec,
            {"apply_to", "variants"},
            name=f"sweeps.axes.{axis}",
        )
        apply_to = _validate_apply_to(spec.get("apply_to"), axis=axis)
        variants = spec.get("variants")
        if (
            not isinstance(variants, Sequence)
            or isinstance(variants, (str, bytes))
            or not variants
        ):
            raise ValueError(
                f"sweeps.axes.{axis}.variants must be a non-empty array"
            )
        seen: set[str] = set()
        normalized_variants: list[dict[str, Any]] = []
        for index, raw_variant in enumerate(variants):
            variant = _mapping(
                raw_variant,
                name=f"sweeps.axes.{axis}.variants[{index}]",
            )
            _unknown_keys(
                variant,
                {"name", "description", "overrides"},
                name=f"sweeps.axes.{axis}.variants[{index}]",
            )
            name = _text(
                variant.get("name"),
                name=f"sweeps.axes.{axis}.variants[{index}].name",
            )
            if name in seen:
                raise ValueError(f"Duplicate {axis} variant name: {name!r}")
            seen.add(name)
            overrides = _mapping(
                variant.get("overrides", {}),
                name=f"sweeps.axes.{axis}.{name}.overrides",
            )
            allowed = AXIS_FIELDS[axis]
            unsupported = sorted(set(overrides) - set(allowed))
            if unsupported:
                locked = sorted(
                    set(unsupported) & set(FAIR_BUDGET_FIELDS)
                )
                if locked:
                    raise ValueError(
                        f"{axis}.{name} overrides locked fair-budget fields: "
                        f"{locked}"
                    )
                raise ValueError(
                    f"{axis}.{name} has unsupported override fields: "
                    f"{unsupported}; allowed={sorted(allowed)}"
                )
            normalized_variants.append(
                {
                    "name": name,
                    "description": str(variant.get("description", "")).strip(),
                    "overrides": overrides,
                }
            )
        normalized[axis] = {
            "apply_to": apply_to,
            "variants": tuple(normalized_variants),
        }
    return mode, normalized


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(config, name="config")
    _unknown_keys(
        value,
        {
            "schema_version",
            "name",
            "manifest",
            "output_dir",
            "include_core",
            "seeds",
            "budget",
            "shared",
            "methods",
            "sweeps",
        },
        name="config",
    )
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("schema_version must be 1")
    name = _text(value.get("name"), name="name")
    manifest = _text(value.get("manifest"), name="manifest")
    output_dir = _text(value.get("output_dir"), name="output_dir")
    include_core = value.get("include_core", True)
    if not isinstance(include_core, bool):
        raise TypeError("include_core must be boolean")

    budget = _validate_budget(value.get("budget"))
    shared = _mapping(value.get("shared", {}), name="shared")
    _unknown_keys(shared, RUN_BATCH_FIELDS, name="shared")
    controlled = sorted(set(shared) & CONTROLLED_FIELDS)
    if controlled:
        raise ValueError(f"shared overrides matrix-controlled fields: {controlled}")
    locked = sorted(set(shared) & set(FAIR_BUDGET_FIELDS))
    if locked:
        raise ValueError(
            f"shared duplicates locked fair-budget fields: {locked}"
        )
    methods = _validate_methods(value.get("methods"))
    sweep_mode, sweeps = _validate_sweeps(value.get("sweeps", {}))
    if not include_core and not sweeps:
        raise ValueError("matrix has neither core runs nor sweeps")

    return {
        "name": name,
        "manifest": manifest,
        "output_dir": output_dir,
        "include_core": include_core,
        "seeds": _validate_seeds(value.get("seeds")),
        "budget": budget,
        "shared": shared,
        "methods": methods,
        "sweep_mode": sweep_mode,
        "sweeps": sweeps,
    }


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolve_options(
    options: Mapping[str, Any],
    *,
    config_dir: Path,
) -> dict[str, Any]:
    resolved = dict(options)
    for field in PATH_OPTION_FIELDS:
        value = resolved.get(field)
        if isinstance(value, str) and value.strip():
            resolved[field] = _resolve_path(config_dir, value.strip())
    return resolved


def _validate_layer_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"{name} must be a non-empty array")
    normalized: list[int] = []
    for index, layer in enumerate(value):
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
        ):
            raise ValueError(
                f"{name}[{index}] must be a non-negative integer"
            )
        normalized.append(layer)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must contain unique layers")
    return tuple(normalized)


def _validate_checkpoint_steps(options: Mapping[str, Any]) -> None:
    values = options.get("checkpoint_steps")
    if values is None:
        return
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
    ):
        raise ValueError("checkpoint_steps must be a non-empty array")
    steps = options["steps"]
    normalized: list[int] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > steps
        ):
            raise ValueError(
                f"checkpoint_steps must be integers within [0, {steps}]"
            )
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError("checkpoint_steps must be unique")
    if normalized != sorted(normalized):
        raise ValueError("checkpoint_steps must be in ascending order")


def _validate_job_options(options: Mapping[str, Any]) -> None:
    unknown = sorted(set(options) - set(RUN_BATCH_FIELDS) - {"method", "seed"})
    if unknown:
        raise ValueError(f"Expanded job has unsupported run_batch fields: {unknown}")

    method = options["method"]
    if method not in METHODS:
        raise ValueError(f"Expanded job has unknown method {method!r}")

    for field in ("model_name", "device"):
        if field in options:
            _text(options[field], name=field)
    for field in ("model_id", "target_text"):
        if field in options and options[field] is not None:
            _text(options[field], name=field)
    for field in PATH_OPTION_FIELDS:
        if field in options and options[field] is not None:
            _text(options[field], name=field)

    choices = {
        "dtype": {"float32", "float16", "bfloat16"},
        "determinism": {"off", "warn", "strict"},
        "token_span": {"audio", "target", "all"},
        "pooling": {"mean", "max", "first", "last"},
    }
    for field, allowed in choices.items():
        if field in options and (
            not isinstance(options[field], str)
            or options[field] not in allowed
        ):
            raise ValueError(
                f"{field} must be one of {sorted(allowed)}"
            )
    for field in ("verbose", "fail_fast"):
        if field in options and not isinstance(options[field], bool):
            raise TypeError(f"{field} must be boolean")

    fixed_layer = options.get("fixed_layer")
    if fixed_layer is not None and (
        isinstance(fixed_layer, bool)
        or not isinstance(fixed_layer, int)
        or fixed_layer < 0
    ):
        raise ValueError("fixed_layer must be a non-negative integer")
    layers = _validate_layer_sequence(options.get("layers"), name="layers")
    static_layers = _validate_layer_sequence(
        options.get("static_topk_layers"),
        name="static_topk_layers",
    )
    top_k = options.get("top_k", 1)
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer")

    if method != "standard" and not options.get("probe_checkpoint"):
        raise ValueError(f"method {method!r} requires probe_checkpoint")
    if method == "fixed" and fixed_layer is None:
        raise ValueError("method 'fixed' requires fixed_layer")
    if method == "static_topk" and len(static_layers) != top_k:
        raise ValueError(
            "static_topk requires exactly top_k unique static_topk_layers"
        )
    if method in {"uniform", "gradient_adaptive", "safety_state_adaptive"}:
        if "layers" in options and not layers:
            raise ValueError(f"method {method!r} requires non-empty layers")

    if "temperature" in options:
        _finite_number(
            options["temperature"],
            name="temperature",
            minimum=0.0,
            inclusive=False,
        )
    if "state_loss_weight" in options:
        _finite_number(
            options["state_loss_weight"],
            name="state_loss_weight",
            minimum=0.0,
            inclusive=True,
        )
    _validate_checkpoint_steps(options)


def _variant_groups(
    method: str,
    *,
    mode: str,
    sweeps: Mapping[str, Mapping[str, Any]],
) -> list[tuple[tuple[str, Mapping[str, Any]], ...]]:
    applicable = [
        (
            axis,
            tuple(sweeps[axis]["variants"]),
        )
        for axis in ABLATION_AXES
        if axis in sweeps and method in sweeps[axis]["apply_to"]
    ]
    if mode == "one_factor":
        return [
            ((axis, variant),)
            for axis, variants in applicable
            for variant in variants
        ]
    if not applicable:
        return []
    choices = [
        tuple((axis, variant) for variant in variants)
        for axis, variants in applicable
    ]
    return [tuple(group) for group in itertools.product(*choices)]


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return result[:72] or "run"


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def _job(
    *,
    name: str,
    manifest: str,
    output_root: Path,
    config_dir: Path,
    base_method: str,
    seed: int,
    group: str,
    variants: Sequence[tuple[str, Mapping[str, Any]]],
    shared: Mapping[str, Any],
    budget: Mapping[str, Any],
    methods: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    options.update(shared)
    options.update(budget)
    options.update(methods[base_method])
    options["method"] = base_method
    options["seed"] = seed

    for axis, variant in variants:
        overrides = dict(variant["overrides"])
        effective = overrides.get("method", options["method"])
        if effective != options["method"]:
            if effective not in methods:
                raise ValueError(
                    f"{axis}.{variant['name']} selects unknown method {effective!r}"
                )
            options.update(methods[effective])
        options.update(overrides)
    options = _resolve_options(options, config_dir=config_dir)
    _validate_job_options(options)

    labels = [
        {
            "axis": axis,
            "variant": variant["name"],
            "description": variant["description"],
        }
        for axis, variant in variants
    ]
    identity = {
        "matrix": name,
        "manifest": manifest,
        "base_method": base_method,
        "effective_method": options["method"],
        "seed": seed,
        "group": group,
        "sweeps": labels,
        "options": options,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    if group == "core":
        human = f"core__{base_method}__seed-{seed}"
    else:
        sweep_name = "__".join(
            f"{item['axis']}-{item['variant']}" for item in labels
        )
        human = (
            f"ablation__{base_method}__{sweep_name}__"
            f"seed-{seed}"
        )
    job_id = f"{_slug(human)}__{digest}"
    return {
        "job_id": job_id,
        "group": group,
        "base_method": base_method,
        "effective_method": options["method"],
        "seed": seed,
        "sweeps": labels,
        "manifest_path": manifest,
        "output_dir": str((output_root / job_id).resolve()),
        "options": _json_copy(options),
        "fair_budget": {
            field: _json_copy(options[field])
            for field in FAIR_BUDGET_FIELDS
        },
    }


def validate_fair_budget(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one identical optimization budget across every expanded job."""

    if not jobs:
        raise ValueError("experiment matrix expanded to zero jobs")
    canonical = dict(jobs[0]["fair_budget"])
    expected = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for job in jobs[1:]:
        actual = json.dumps(
            job["fair_budget"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if actual != expected:
            raise ValueError(
                f"Fair-budget mismatch in job {job.get('job_id')!r}"
            )
    return canonical


def expand_experiment_matrix(
    config: Mapping[str, Any],
    *,
    config_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Validate a config and deterministically expand all matrix jobs."""

    normalized = _validate_config(config)
    config_file = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else None
    )
    config_dir = config_file.parent if config_file is not None else Path.cwd()
    manifest = _resolve_path(config_dir, normalized["manifest"])
    root_value = (
        str(output_dir)
        if output_dir is not None
        else normalized["output_dir"]
    )
    output_root = Path(root_value).expanduser()
    if not output_root.is_absolute():
        output_root = config_dir / output_root
    output_root = output_root.resolve()

    jobs: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in normalized["seeds"]:
            if normalized["include_core"]:
                jobs.append(
                    _job(
                        name=normalized["name"],
                        manifest=manifest,
                        output_root=output_root,
                        config_dir=config_dir,
                        base_method=method,
                        seed=seed,
                        group="core",
                        variants=(),
                        shared=normalized["shared"],
                        budget=normalized["budget"],
                        methods=normalized["methods"],
                    )
                )
            for variants in _variant_groups(
                method,
                mode=normalized["sweep_mode"],
                sweeps=normalized["sweeps"],
            ):
                jobs.append(
                    _job(
                        name=normalized["name"],
                        manifest=manifest,
                        output_root=output_root,
                        config_dir=config_dir,
                        base_method=method,
                        seed=seed,
                        group="ablation",
                        variants=variants,
                        shared=normalized["shared"],
                        budget=normalized["budget"],
                        methods=normalized["methods"],
                    )
                )

    identifiers = [job["job_id"] for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("experiment expansion produced duplicate job IDs")
    fair_budget = validate_fair_budget(jobs)
    return {
        "format": "safety-experiment-matrix-plan",
        "version": 1,
        "name": normalized["name"],
        "config": None if config_file is None else str(config_file),
        "manifest": manifest,
        "output_dir": str(output_root),
        "methods": list(METHODS),
        "seeds": list(normalized["seeds"]),
        "sweep_mode": normalized["sweep_mode"],
        "fair_budget_fields": list(FAIR_BUDGET_FIELDS),
        "fair_budget": fair_budget,
        "jobs": jobs,
    }


def _default_run_batch(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    from experiments.batch_safety_attack import run_batch

    return run_batch(*args, **kwargs)


def _summary_from_plan(plan: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    jobs = []
    for planned in plan["jobs"]:
        job = dict(planned)
        job["status"] = "planned"
        jobs.append(job)
    return {
        "format": "safety-experiment-matrix-summary",
        "version": 1,
        "name": plan["name"],
        "config": plan["config"],
        "manifest": plan["manifest"],
        "output_dir": plan["output_dir"],
        "dry_run": dry_run,
        "methods": plan["methods"],
        "seeds": plan["seeds"],
        "sweep_mode": plan["sweep_mode"],
        "fair_budget_fields": plan["fair_budget_fields"],
        "fair_budget": plan["fair_budget"],
        "counts": {
            "total": len(jobs),
            "planned": len(jobs),
            "completed": 0,
            "failed": 0,
        },
        "jobs": jobs,
    }


def run_experiment_matrix(
    config_path: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    summary_path: Optional[str | Path] = None,
    dry_run: bool = True,
    fail_fast: bool = False,
    run_batch_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Expand a matrix and optionally dispatch every job through run_batch."""

    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be boolean")
    if not isinstance(fail_fast, bool):
        raise TypeError("fail_fast must be boolean")
    config_file = Path(config_path).expanduser().resolve()
    config = load_matrix_config(config_file)
    plan = expand_experiment_matrix(
        config,
        config_path=config_file,
        output_dir=output_dir,
    )
    summary_file = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else Path(plan["output_dir"]) / "matrix_summary.json"
    )
    summary = _summary_from_plan(plan, dry_run=dry_run)
    _atomic_json(summary_file, summary)
    if dry_run:
        return summary

    runner = run_batch_fn or _default_run_batch
    for job in summary["jobs"]:
        job["status"] = "running"
        _atomic_json(summary_file, summary)
        try:
            result = runner(
                job["manifest_path"],
                job["output_dir"],
                **dict(job["options"]),
            )
            if not isinstance(result, Mapping):
                raise TypeError("run_batch must return a summary mapping")
            counts = result.get("counts")
            batch_counts = dict(counts) if isinstance(counts, Mapping) else {}
            batch_failed = int(batch_counts.get("failed", 0) or 0)
            job["batch_counts"] = _json_copy(batch_counts)
            job["batch_summary_path"] = str(
                Path(job["output_dir"]) / "summary.json"
            )
            summary["counts"]["planned"] -= 1
            if batch_failed:
                job["status"] = "failed"
                summary["counts"]["failed"] += 1
            else:
                job["status"] = "completed"
                summary["counts"]["completed"] += 1
        except Exception as exc:
            job["status"] = "failed"
            job["error_type"] = type(exc).__name__
            job["message"] = str(exc)
            summary["counts"]["planned"] -= 1
            summary["counts"]["failed"] += 1
            _atomic_json(summary_file, summary)
            if fail_fast:
                raise
        _atomic_json(summary_file, summary)
        if fail_fast and job["status"] == "failed":
            break
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary", dest="summary_path", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the expanded jobs; omission performs a dry-run only",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    run_batch_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> int:
    args = build_parser().parse_args(argv)
    summary = run_experiment_matrix(
        args.config,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        dry_run=not args.execute,
        fail_fast=args.fail_fast,
        run_batch_fn=run_batch_fn,
    )
    return 1 if summary["counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABLATION_AXES",
    "FAIR_BUDGET_FIELDS",
    "METHODS",
    "build_parser",
    "expand_experiment_matrix",
    "load_matrix_config",
    "main",
    "run_experiment_matrix",
    "validate_fair_budget",
]
