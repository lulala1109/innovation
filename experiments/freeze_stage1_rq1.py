"""Build a read-only, hash-bound catalog for one completed Stage-1/RQ1 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from experiments.collect_safety_states import safe_torch_load
from experiments.stage1_rq1_config import (
    RQ1ConfigError,
    inspect_status,
    load_config,
    validate_pipeline,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RQ1ConfigError(f"expected a JSON object: {path}")
    return value


def _relative(path: Path, *, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _require_equal(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise RQ1ConfigError(f"{name} mismatch: {actual} != {expected}")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _add_existing(
    selected: set[Path],
    paths: Iterable[Path],
    *,
    required: bool,
) -> None:
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            selected.add(path)
        elif required:
            raise RQ1ConfigError(f"freeze input is missing: {path}")


def _artifact_allowlist(config: Any) -> tuple[Path, ...]:
    """Return files that bind inputs, labels, learned states, replay, and reports."""

    selected: set[Path] = set()
    required_path_fields = (
        "source_manifest",
        "probe_candidates",
        "trajectory_candidates",
        "clean_labels",
        "clean_attached",
        "train_responses",
        "train_labels",
        "train_attached",
        "train_final",
        "probe_states",
        "probe_checkpoint",
        "heldout_responses",
        "heldout_labels",
        "heldout_attached",
    )
    optional_path_fields = (
        "clean_exclusions",
        "train_attack_exclusions",
        "train_final_exclusions",
        "heldout_exclusions",
    )
    _add_existing(
        selected,
        (config.paths[field] for field in required_path_fields),
        required=True,
    )
    _add_existing(
        selected,
        (config.paths[field] for field in optional_path_fields),
        required=False,
    )
    _add_existing(
        selected,
        (
            config.source_path,
            config.paths["pipeline_summary"],
            config.paths["train_attack_dir"] / "summary.json",
            config.paths["heldout_attack_dir"] / "summary.json",
            config.paths["replay_dir"] / "index.json",
            config.paths["scores_dir"] / "state_scores.pt",
            config.paths["scores_dir"] / "state_scores_long.csv",
        ),
        required=True,
    )
    analysis_root = config.paths["analysis_dir"]
    if not analysis_root.is_dir():
        raise RQ1ConfigError(f"analysis directory is missing: {analysis_root}")
    analysis_files = tuple(
        path for path in analysis_root.rglob("*") if path.is_file()
    )
    if not analysis_files:
        raise RQ1ConfigError(f"analysis directory is empty: {analysis_root}")
    _add_existing(selected, analysis_files, required=True)
    return tuple(sorted(selected))


def _actual_expected(config: Any) -> Mapping[str, int]:
    train_summary = _json(config.paths["train_attack_dir"] / "summary.json")
    heldout_summary = _json(config.paths["heldout_attack_dir"] / "summary.json")
    train_counts = train_summary.get("counts")
    heldout_counts = heldout_summary.get("counts")
    if not isinstance(train_counts, Mapping) or not isinstance(heldout_counts, Mapping):
        raise RQ1ConfigError("attack summaries do not contain count objects")

    states = safe_torch_load(config.paths["probe_states"])
    checkpoint = safe_torch_load(config.paths["probe_checkpoint"])
    hidden_states = states.get("hidden_states")
    hidden_sizes = checkpoint.get("hidden_sizes")
    if not isinstance(hidden_states, Mapping) or not hidden_states:
        raise RQ1ConfigError("probe training states do not contain hidden states")
    if not isinstance(hidden_sizes, Mapping) or not hidden_sizes:
        raise RQ1ConfigError("probe checkpoint does not contain hidden sizes")
    first_matrix = next(iter(hidden_states.values()))
    if not hasattr(first_matrix, "shape") or len(first_matrix.shape) != 2:
        raise RQ1ConfigError("probe hidden-state matrices are malformed")

    probe_candidates = _csv_count(config.paths["probe_candidates"])
    trajectory_candidates = _csv_count(config.paths["trajectory_candidates"])
    clean_pairs = _csv_count(config.paths["clean_attached"])
    final_pairs = _csv_count(config.paths["train_final"])
    heldout_pairs = _csv_count(config.paths["heldout_attached"])
    train_cases = int(train_counts.get("total", -1))
    heldout_cases = int(heldout_counts.get("total", -1))
    probe_rows = int(first_matrix.shape[0])
    layers = len(hidden_sizes)

    _require_equal("clean/train attack case count", clean_pairs, train_cases)
    _require_equal("final-pair/probe-row triplet count", 3 * final_pairs, probe_rows)
    _require_equal("trajectory/heldout manifest count", trajectory_candidates, heldout_pairs)
    _require_equal("heldout manifest/attack count", heldout_pairs, heldout_cases)
    _require_equal("probe-state/probe-checkpoint layer count", len(hidden_states), layers)

    return {
        "probe_candidates": probe_candidates,
        "trajectory_candidates": trajectory_candidates,
        "clean_eligible_train_pairs": clean_pairs,
        "train_attack_cases": train_cases,
        "final_train_pairs": final_pairs,
        "probe_state_rows": probe_rows,
        "heldout_cases": heldout_cases,
        "steps_per_case": int(config.attack["steps"]) + 1,
        "layers": layers,
    }


def _orphan_payload(config: Any) -> Mapping[str, Any]:
    behavior_root = config.output_root / "behavior"
    canonical = config.paths["train_labels"]
    records: list[Mapping[str, Any]] = []
    if behavior_root.is_dir():
        for path in sorted(behavior_root.glob(".*.tmp")):
            if not path.is_file():
                continue
            records.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                    "line_count": _line_count(path),
                    "status": "orphaned_atomic_temp",
                    "disposition": "retained_in_place",
                    "excluded_from_frozen_artifacts": True,
                    "canonical_replacement": {
                        "path": str(canonical),
                        "sha256": _sha256(canonical),
                        "size_bytes": canonical.stat().st_size,
                        "line_count": _line_count(canonical),
                    },
                }
            )
    return {
        "format": "stage1-rq1-orphan-artifact-audit",
        "version": 1,
        "created_at": _utc_now(),
        "source_run": config.name,
        "artifacts": records,
    }


def build_catalog(
    source_path: Path,
    output_path: Path,
    catalog_root: Path,
    *,
    force: bool,
) -> int:
    source = load_config(source_path)
    if source.frozen:
        raise RQ1ConfigError("source configuration is already frozen")
    output_path = output_path.expanduser().resolve()
    catalog_root = catalog_root.expanduser().resolve()
    if output_path.exists() and not force:
        raise RQ1ConfigError(f"output config already exists: {output_path}")

    actual = _actual_expected(source)
    raw = deepcopy(dict(source.raw))
    raw["name"] = f"{source.name}_frozen"
    raw["frozen"] = True
    raw["template"] = False
    raw["execution_enabled"] = False
    raw["output_root"] = _relative(catalog_root, base=output_path.parent)
    expected = dict(raw["expected"])
    expected.update(actual)
    raw["expected"] = expected
    paths = dict(raw["paths"])
    paths["pipeline_summary"] = _relative(
        catalog_root / "rq1_pipeline_summary.json",
        base=output_path.parent,
    )
    raw["paths"] = paths

    frozen_records = [
        {
            "path": _relative(path, base=output_path.parent),
            "sha256": _sha256(path),
        }
        for path in _artifact_allowlist(source)
    ]
    raw["frozen_artifacts"] = frozen_records
    _atomic_json(output_path, raw)

    frozen = load_config(output_path)
    errors, warnings = validate_pipeline(frozen)
    status_rows = [row.as_dict() for row in inspect_status(frozen)]
    status_counts: dict[str, int] = {}
    for row in status_rows:
        status_counts[row["state"]] = status_counts.get(row["state"], 0) + 1

    catalog_root.mkdir(parents=True, exist_ok=True)
    orphan_path = catalog_root / "orphan_artifacts.json"
    validation_path = catalog_root / "p0_validation.json"
    checksum_path = catalog_root / "frozen_catalog.sha256"
    _atomic_json(orphan_path, _orphan_payload(source))
    _atomic_json(
        validation_path,
        {
            "format": "stage1-rq1-p0-validation",
            "version": 1,
            "created_at": _utc_now(),
            "source_config": str(source.source_path),
            "source_config_sha256": _sha256(source.source_path),
            "source_config_fingerprint": source.fingerprint,
            "frozen_config": str(frozen.source_path),
            "frozen_config_fingerprint": frozen.fingerprint,
            "actual_expected": dict(actual),
            "frozen_artifact_count": len(frozen_records),
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "status_counts": status_counts,
            "stages": status_rows,
            "metadata_corrections": [
                {
                    "field": "expected.layers",
                    "recorded": source.expected["layers"],
                    "actual": actual["layers"],
                    "reason": "Qwen2.5-Omni-7B text decoder has 28 layers",
                },
                {
                    "field": "expected.clean_eligible_train_pairs",
                    "recorded": source.expected["clean_eligible_train_pairs"],
                    "actual": actual["clean_eligible_train_pairs"],
                    "reason": "closed from the completed derived manifest",
                },
                {
                    "field": "expected.train_attack_cases",
                    "recorded": source.expected["train_attack_cases"],
                    "actual": actual["train_attack_cases"],
                    "reason": "closed from the completed attack summary",
                },
                {
                    "field": "expected.final_train_pairs",
                    "recorded": source.expected["final_train_pairs"],
                    "actual": actual["final_train_pairs"],
                    "reason": "closed from the finalized training manifest",
                },
                {
                    "field": "expected.probe_state_rows",
                    "recorded": source.expected["probe_state_rows"],
                    "actual": actual["probe_state_rows"],
                    "reason": "closed from the frozen training-state tensor",
                },
            ],
        },
    )
    checksum_targets = (output_path, validation_path, orphan_path)
    _atomic_text(
        checksum_path,
        "".join(
            f"{_sha256(path)}  {_relative(path, base=source.project_root)}\n"
            for path in checksum_targets
        ),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"frozen config: {output_path}")
    print(f"catalog root: {catalog_root}")
    print(f"frozen artifacts: {len(frozen_records)}")
    print(f"stage states: {status_counts}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return build_catalog(
            args.source,
            args.output,
            args.catalog_root,
            force=args.force,
        )
    except (OSError, ValueError, RQ1ConfigError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
