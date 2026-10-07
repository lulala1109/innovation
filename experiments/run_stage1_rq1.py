"""Unified, explicit-stage runner for the Stage-1 RQ1 experiment pipeline.

Examples::

    python -m experiments.run_stage1_rq1 plan --config configs/my_rq1.json
    python -m experiments.run_stage1_rq1 status --config configs/my_rq1.json
    python -m experiments.run_stage1_rq1 validate --config configs/my_rq1.json
    python -m experiments.run_stage1_rq1 run --config configs/my_rq1.json --stage replay
    python -m experiments.run_stage1_rq1 run --config configs/my_rq1.json \
        --from heldout_attack --through report

The runner never runs all expensive stages implicitly.  Each executable stage
is launched in a fresh, serial subprocess so GPU/model and API work cannot
overlap or retain resources across stage boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from experiments.stage1_rq1_config import (
    PIPELINE_SUMMARY_FORMAT,
    PIPELINE_SUMMARY_VERSION,
    RQ1ConfigError,
    RQ1PipelineConfig,
    STAGE_ORDER,
    StageStatus,
    assert_executable,
    command_text,
    inspect_status,
    load_config,
    load_env_file,
    select_stages,
    validate_completed_stage,
    validate_pipeline,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _status_map(config: RQ1PipelineConfig) -> dict[str, StageStatus]:
    return {status.name: status for status in inspect_status(config)}


def plan_payload(config: RQ1PipelineConfig) -> Mapping[str, Any]:
    """Build a secret-free, non-executing pipeline plan."""

    statuses = _status_map(config)
    return {
        "config": str(config.source_path),
        "name": config.name,
        "frozen": config.frozen,
        "template": config.template,
        "execution_enabled": config.execution_enabled,
        "config_fingerprint": config.fingerprint,
        "stages": [
            {
                "name": spec.name,
                "dependencies": list(spec.dependencies),
                "resources": list(spec.resources),
                "status": statuses[spec.name].state,
                "status_detail": statuses[spec.name].detail,
                "command": command_text(spec),
                "outputs": [str(path) for path in spec.outputs],
            }
            for spec in config.stage_specs()
        ],
    }


def status_payload(config: RQ1PipelineConfig) -> Mapping[str, Any]:
    statuses = inspect_status(config)
    counts = {state: 0 for state in ("complete", "partial", "pending", "blocked", "invalid")}
    for status in statuses:
        counts[status.state] = counts.get(status.state, 0) + 1
    return {
        "config": str(config.source_path),
        "name": config.name,
        "counts": counts,
        "stages": [status.as_dict() for status in statuses],
    }


def _load_summary(config: RQ1PipelineConfig) -> dict[str, Any]:
    path = config.paths["pipeline_summary"]
    if not path.exists():
        return {
            "format": PIPELINE_SUMMARY_FORMAT,
            "version": PIPELINE_SUMMARY_VERSION,
            "pipeline_name": config.name,
            "config_path": str(config.source_path),
            "config_fingerprint": config.fingerprint,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            # Deliberately records neither the .env path/keys nor any values.
            "environment": {"loaded_from_env_file": bool(config.env_file)},
            "events": [],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RQ1ConfigError("pipeline summary is not a JSON object")
    if value.get("config_fingerprint") != config.fingerprint:
        raise RQ1ConfigError("pipeline summary fingerprint does not match config")
    events = value.get("events")
    if not isinstance(events, list):
        raise RQ1ConfigError("pipeline summary events must be an array")
    return value


def _record_event(
    config: RQ1PipelineConfig,
    summary: dict[str, Any],
    *,
    stage: str,
    action: str,
    returncode: Optional[int] = None,
    detail: Optional[str] = None,
) -> None:
    event: dict[str, Any] = {"timestamp": _utc_now(), "stage": stage, "action": action}
    if returncode is not None:
        event["returncode"] = returncode
    if detail is not None:
        event["detail"] = detail
    summary["events"].append(event)
    summary["updated_at"] = _utc_now()
    summary["last_status"] = [status.as_dict() for status in inspect_status(config)]
    _atomic_json(config.paths["pipeline_summary"], summary)


def _check_dependencies(stage: str, statuses: Mapping[str, StageStatus]) -> None:
    missing = [
        dependency
        for dependency in next(spec for spec in _SPECS_CACHE if spec.name == stage).dependencies
        if statuses[dependency].state != "complete"
    ]
    if missing:
        details = ", ".join(
            f"{name}={statuses[name].state}" for name in missing
        )
        raise RQ1ConfigError(f"stage {stage} is blocked by dependency state: {details}")


# This is assigned for the duration of run_pipeline.  Keeping dependency lookup
# module-local avoids accepting user-defined dependency graphs in configuration.
_SPECS_CACHE = ()


def run_pipeline(
    config: RQ1PipelineConfig,
    stages: Sequence[str],
    *,
    subprocess_runner: Any = subprocess.run,
) -> Mapping[str, Any]:
    """Run explicitly selected stages serially, skipping verified completion."""

    assert_executable(config)
    selected = tuple(stages)
    if not selected:
        raise RQ1ConfigError("at least one explicit stage is required")
    unknown = sorted(set(selected) - set(STAGE_ORDER))
    if unknown:
        raise RQ1ConfigError(f"unknown stage(s): {', '.join(unknown)}")

    global _SPECS_CACHE
    specs = config.stage_specs()
    _SPECS_CACHE = specs
    by_name = {spec.name: spec for spec in specs}
    summary = _load_summary(config)
    environment = load_env_file(config.env_file)
    results: list[Mapping[str, Any]] = []

    for stage in selected:
        statuses = _status_map(config)
        current = statuses[stage]
        if current.state == "complete":
            validation_errors = validate_completed_stage(config, stage)
            if validation_errors:
                raise RQ1ConfigError(
                    f"stage {stage} failed deep validation: "
                    + "; ".join(validation_errors[:3])
                )
            results.append({"stage": stage, "action": "skipped", "reason": "complete"})
            _record_event(config, summary, stage=stage, action="skipped_complete")
            continue
        if current.state == "invalid":
            raise RQ1ConfigError(f"stage {stage} is invalid: {current.detail}")
        _check_dependencies(stage, statuses)
        spec = by_name[stage]
        if spec.command is None:
            # report is intentionally audit-only because analyze already creates
            # it.  Missing report products therefore indicate an invalid analyze
            # result, never permission to overwrite/recompute implicitly.
            raise RQ1ConfigError(
                f"stage {stage} has no standalone command and its expected artifact is absent"
            )
        if "api" in spec.resources:
            key_name = str(config.judge["api_key_env"])
            if not environment.get(key_name):
                raise RQ1ConfigError(
                    f"stage {stage} requires environment variable {key_name!r}; value is not logged"
                )

        _record_event(config, summary, stage=stage, action="started")
        completed = subprocess_runner(
            list(spec.command),
            cwd=str(config.project_root),
            env=environment,
            check=False,
        )
        returncode = int(completed.returncode)
        if returncode != 0:
            _record_event(
                config,
                summary,
                stage=stage,
                action="failed",
                returncode=returncode,
            )
            raise RQ1ConfigError(f"stage {stage} failed with exit code {returncode}")
        after = _status_map(config)[stage]
        if after.state != "complete":
            _record_event(
                config,
                summary,
                stage=stage,
                action="incomplete",
                returncode=returncode,
                detail=f"{after.state}: {after.detail}",
            )
            raise RQ1ConfigError(
                f"stage {stage} exited successfully but artifact status is "
                f"{after.state}: {after.detail}"
            )
        _record_event(
            config, summary, stage=stage, action="completed", returncode=returncode
        )
        results.append({"stage": stage, "action": "completed", "returncode": returncode})

    return {
        "config": str(config.source_path),
        "config_fingerprint": config.fingerprint,
        "results": results,
        "pipeline_summary": str(config.paths["pipeline_summary"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "Show stages, dependencies, resources, commands, and outputs"),
        ("status", "Inspect lightweight artifact completion state"),
        ("validate", "Deeply validate identity, grids, hashes, protocol, and provenance"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    run = subparsers.add_parser(
        "run", help="Run one explicit stage or one explicit inclusive stage range"
    )
    run.add_argument("--config", required=True, type=Path)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=STAGE_ORDER)
    selection.add_argument("--from", dest="start", choices=STAGE_ORDER)
    run.add_argument("--through", choices=STAGE_ORDER)
    run.add_argument("--json", action="store_true", help="Emit final machine-readable JSON")
    return parser


def _print_plan(payload: Mapping[str, Any]) -> None:
    flags = []
    if payload["frozen"]:
        flags.append("frozen")
    if payload["template"]:
        flags.append("template")
    if not payload["execution_enabled"]:
        flags.append("execution-disabled")
    suffix = f" ({', '.join(flags)})" if flags else ""
    print(f"RQ1 pipeline: {payload['name']}{suffix}")
    for stage in payload["stages"]:
        dependencies = ",".join(stage["dependencies"]) or "-"
        resources = ",".join(stage["resources"])
        print(
            f"[{stage['status']:<8}] {stage['name']} "
            f"deps={dependencies} resources={resources}"
        )
        print(f"  {stage['command']}")


def _print_status(payload: Mapping[str, Any]) -> None:
    print(f"RQ1 pipeline: {payload['name']}")
    for stage in payload["stages"]:
        print(f"[{stage['state']:<8}] {stage['name']}: {stage['detail']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "plan":
            payload = plan_payload(config)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                _print_plan(payload)
            return 0
        if args.command == "status":
            payload = status_payload(config)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                _print_status(payload)
            return 1 if payload["counts"].get("invalid", 0) else 0
        if args.command == "validate":
            errors, warnings = validate_pipeline(config)
            payload = {
                "config": str(config.source_path),
                "valid": not errors,
                "errors": errors,
                "warnings": warnings,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                print("VALID" if not errors else "INVALID")
                for warning in warnings:
                    print(f"WARNING: {warning}")
                for error in errors:
                    print(f"ERROR: {error}")
            return 1 if errors else 0

        stages = select_stages(stage=args.stage, start=args.start, through=args.through)
        result = run_pipeline(config, stages)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            for row in result["results"]:
                print(f"{row['stage']}: {row['action']}")
        return 0
    except (OSError, RQ1ConfigError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "main",
    "plan_payload",
    "run_pipeline",
    "status_payload",
]
