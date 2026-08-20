#!/usr/bin/env python3
"""Prepare immutable Stage-1 candidates and attach verified attack outputs.

The source JBB/TTS CSV is read-only input.  This module deliberately writes
derived manifests instead of adding experiment results to that source file.
Paths below the repository are stored relative to the repository root so the
outputs can be consumed consistently from different working directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from data.build_safety_pairs import (
    COMPLETION_COLUMNS,
    STATE_AUDIO_COLUMNS,
    ManifestValidationError,
    build_manifest,
    validate_manifest,
)
from data.datasets import read_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPLIT = "measurement_train"
VALIDATION_SPLIT = "measurement_val"
PROBE_ROLE = "probe_candidate"
TRAJECTORY_ROLE = "trajectory_candidate"
PATH_COLUMNS = (
    "benign_audio_path",
    "harmful_audio_path",
    "clean_audio_path",
    "jailbreak_audio_path",
    "trajectory_path",
    "behavior_labels_path",
    "clean_behavior_labels_path",
    "selected_checkpoint_path",
)
EXCLUSION_REASON_COLUMN = "exclusion_reason"
TRAJECTORY_LABEL_FORMAT = "stage1-behavior-label"
TRAJECTORY_LABEL_VERSION = 1
CLEAN_LABEL_FORMAT = "stage1-clean-behavior"
CLEAN_LABEL_VERSION = 1


class Stage1ManifestError(ValueError):
    """Raised when Stage-1 provenance or split invariants are violated."""


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """One checkpoint whose identity is bound to the run fingerprint."""

    path: Path
    experiment_fingerprint: str


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without changing ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_sha256(response: str) -> str:
    """Return the canonical digest used by the behavior sidecar."""

    return hashlib.sha256(response.encode("utf-8")).hexdigest()


def _sha256_digest(value: Any, *, name: str) -> str:
    digest = _text(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Stage1ManifestError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage1ManifestError("experiment_config must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def project_relative_path(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> str:
    """Store project-internal paths as POSIX relative paths.

    External artifacts remain absolute; silently rewriting them as relative to
    the manifest would make later replay depend on the current working
    directory.
    """

    root = Path(project_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_project_path(
    value: Any,
    *,
    base: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    must_exist: bool = True,
) -> Path:
    """Resolve absolute, project-relative, or legacy manifest-relative paths."""

    if _is_blank(value):
        raise Stage1ManifestError("A non-empty artifact path is required")
    raw = Path(str(value).strip()).expanduser()
    if raw.is_absolute():
        candidates = (raw,)
    else:
        candidates = (
            Path(project_root).expanduser().resolve() / raw,
            Path(base).expanduser().resolve() / raw,
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if not must_exist or resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve artifact path {value!r}; tried: "
        + ", ".join(str(path.resolve()) for path in candidates)
    )


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _text(value: Any, *, name: str) -> str:
    if _is_blank(value):
        raise Stage1ManifestError(f"{name} must be non-empty")
    return str(value).strip()


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    if normalized in {"unknown", "none", "null", "na", "n/a"}:
        return None
    raise Stage1ManifestError(f"{name} must be true, false, or unknown")


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise Stage1ManifestError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage1ManifestError(
            f"{name} must be a non-negative integer"
        ) from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise Stage1ManifestError(f"{name} must be a non-negative integer")
    return parsed


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def _read_json_object(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ManifestError(f"Cannot read {name}: {path}") from exc
    if not isinstance(value, Mapping):
        raise Stage1ManifestError(f"{name} must be a JSON object: {path}")
    return value


def _read_jsonl_objects(
    path: Path,
    *,
    allow_empty: bool = False,
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise Stage1ManifestError(
                        f"{path}:{line_number}: expected one JSON object"
                    )
                records.append(value)
    except json.JSONDecodeError as exc:
        raise Stage1ManifestError(
            f"{path}:{exc.lineno}: invalid behavior JSON"
        ) from exc
    if not records and not allow_empty:
        raise Stage1ManifestError(f"Behavior label file is empty: {path}")
    return records


def _stratum_column(frame: pd.DataFrame) -> str:
    for name in ("stratum", "category"):
        if name in frame.columns:
            return name
    raise Stage1ManifestError("Source manifest requires stratum or category")


def _validate_pair_ids(frame: pd.DataFrame, *, name: str) -> pd.Series:
    if "pair_id" not in frame.columns:
        raise Stage1ManifestError(f"{name} requires pair_id")
    ids = frame["pair_id"].fillna("").astype(str).str.strip()
    if (ids == "").any():
        raise Stage1ManifestError(f"{name} contains blank pair_id values")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise Stage1ManifestError(
            f"{name} contains duplicate pair_id values: {duplicates[:5]}"
        )
    return ids


def _normalise_path_columns(
    frame: pd.DataFrame,
    *,
    source_path: Path,
    project_root: Path,
) -> pd.DataFrame:
    result = frame.copy()
    for column in PATH_COLUMNS:
        if column not in result.columns:
            continue

        def normalize(value: Any) -> Any:
            if _is_blank(value):
                return ""
            resolved = resolve_project_path(
                value,
                base=source_path.parent,
                project_root=project_root,
                must_exist=False,
            )
            return project_relative_path(resolved, project_root=project_root)

        result[column] = result[column].map(normalize)
    return result


def prepare_candidate_manifests(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    expected_train: int | None = 80,
    expected_validation: int | None = 20,
    expected_train_per_stratum: int | None = 8,
    expected_validation_per_stratum: int | None = 2,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Path]:
    """Derive immutable probe and held-out candidate CSVs.

    No new split is sampled here: the existing ``measurement_split`` values
    are validated and copied.  Thus rerunning this function cannot move a pair
    between the probe pool and the held-out RQ1 trajectory pool.
    """

    source = Path(source_manifest).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source}")
    before_digest = sha256_file(source)
    frame = read_table(source)
    if frame.empty:
        raise Stage1ManifestError("Source manifest has no rows")
    ids = _validate_pair_ids(frame, name="Source manifest")
    if "measurement_split" not in frame.columns:
        raise Stage1ManifestError("Source manifest requires measurement_split")
    splits = frame["measurement_split"].fillna("").astype(str).str.strip()
    unknown_splits = sorted(set(splits) - {TRAIN_SPLIT, VALIDATION_SPLIT})
    if unknown_splits:
        raise Stage1ManifestError(
            "Unexpected measurement_split values: " + ", ".join(unknown_splits)
        )

    counts = splits.value_counts().to_dict()
    expected_counts = {
        TRAIN_SPLIT: expected_train,
        VALIDATION_SPLIT: expected_validation,
    }
    for split, expected in expected_counts.items():
        if expected is not None and counts.get(split, 0) != expected:
            raise Stage1ManifestError(
                f"Expected {expected} {split} rows, found {counts.get(split, 0)}"
            )

    stratum = _stratum_column(frame)
    if frame[stratum].fillna("").astype(str).str.strip().eq("").any():
        raise Stage1ManifestError(f"Source manifest contains blank {stratum} values")
    crosstab = pd.crosstab(frame[stratum], splits)
    per_stratum = {
        TRAIN_SPLIT: expected_train_per_stratum,
        VALIDATION_SPLIT: expected_validation_per_stratum,
    }
    for split, expected in per_stratum.items():
        if expected is None:
            continue
        actual = crosstab.get(split, pd.Series(0, index=crosstab.index))
        invalid = actual[actual != expected]
        if not invalid.empty:
            raise Stage1ManifestError(
                f"Expected {expected} {split} rows per {stratum}; mismatches: "
                + ", ".join(f"{key}={value}" for key, value in invalid.items())
            )

    root = Path(project_root).expanduser().resolve()
    normalized = _normalise_path_columns(
        frame,
        source_path=source,
        project_root=root,
    )
    if "stratum" not in normalized.columns:
        normalized["stratum"] = normalized[stratum]
    normalized["pair_id"] = ids
    normalized["stage1_role"] = splits.map(
        {TRAIN_SPLIT: PROBE_ROLE, VALIDATION_SPLIT: TRAJECTORY_ROLE}
    )
    probe = normalized.loc[splits == TRAIN_SPLIT].reset_index(drop=True)
    trajectory = normalized.loc[splits == VALIDATION_SPLIT].reset_index(drop=True)
    if set(probe["pair_id"]) & set(trajectory["pair_id"]):
        raise AssertionError("Probe and trajectory pair_id sets overlap")

    destination = Path(output_dir).expanduser().resolve()
    outputs = {
        "probe": destination / "jbb_probe_candidates.csv",
        "trajectory": destination / "jbb_trajectory_candidates.csv",
        "exclusions": destination / "jbb_stage1_exclusions.csv",
    }
    if any(path.resolve() == source for path in outputs.values()):
        raise Stage1ManifestError("A derived output cannot overwrite the source manifest")
    _atomic_csv(probe, outputs["probe"])
    _atomic_csv(trajectory, outputs["trajectory"])
    if not outputs["exclusions"].exists():
        _atomic_csv(
            pd.DataFrame(columns=[*normalized.columns, EXCLUSION_REASON_COLUMN]),
            outputs["exclusions"],
        )
    if sha256_file(source) != before_digest:
        raise RuntimeError("Source manifest changed while deriving candidates")
    return outputs


def _artifact_path(
    value: Any,
    *,
    bases: Sequence[Path],
    name: str,
    expect_directory: bool = False,
) -> Path:
    if _is_blank(value):
        raise Stage1ManifestError(f"{name} path is missing")
    raw = Path(str(value).strip()).expanduser()
    candidates = (raw,) if raw.is_absolute() else tuple(base / raw for base in bases)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() if expect_directory else resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve {name} {value!r}; tried: "
        + ", ".join(str(path.resolve()) for path in candidates)
    )


def _trajectory_records(
    index_path: Path,
    *,
    case_dir: Path,
    case_id: str,
    pair_id: str,
    experiment_fingerprint: str,
) -> dict[int, TrajectoryCheckpoint]:
    payload = _read_json_object(index_path, name="trajectory index")
    records = payload.get("checkpoints")
    if not isinstance(records, list) or not records:
        raise Stage1ManifestError(f"Trajectory has no checkpoints: {index_path}")
    result: dict[int, TrajectoryCheckpoint] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Stage1ManifestError(
                f"Trajectory checkpoint {position} must be an object"
            )
        step = _integer(record.get("step"), name="trajectory step")
        if step in result:
            raise Stage1ManifestError(f"Trajectory contains duplicate step {step}")
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise Stage1ManifestError("Trajectory metadata must be an object")
        owner = _text(
            metadata.get("pair_id"), name=f"trajectory step {step}.pair_id"
        )
        if owner != pair_id:
            raise Stage1ManifestError(
                f"Trajectory pair_id {owner!r} does not match {pair_id!r}"
            )
        owner_case = _text(
            metadata.get("case_id"), name=f"trajectory step {step}.case_id"
        )
        if owner_case != case_id:
            raise Stage1ManifestError(
                f"Trajectory case_id {owner_case!r} does not match {case_id!r}"
            )
        metadata_step = _integer(
            metadata.get("step"), name=f"trajectory step {step}.metadata.step"
        )
        if metadata_step != step:
            raise Stage1ManifestError(
                f"Trajectory metadata step {metadata_step} does not match {step}"
            )
        metadata_fingerprint = _sha256_digest(
            metadata.get("experiment_fingerprint"),
            name=f"trajectory step {step}.experiment_fingerprint",
        )
        if metadata_fingerprint != experiment_fingerprint:
            raise Stage1ManifestError(
                f"Trajectory experiment_fingerprint mismatch at step {step}"
            )
        checkpoint = _artifact_path(
            record.get("path"),
            bases=(index_path.parent, case_dir),
            name=f"trajectory checkpoint step {step}",
        )
        result[step] = TrajectoryCheckpoint(
            path=checkpoint,
            experiment_fingerprint=metadata_fingerprint,
        )
    return result


def _validate_trajectory_label(
    record: Mapping[str, Any],
    *,
    position: int,
) -> str:
    if record.get("format") != TRAJECTORY_LABEL_FORMAT:
        raise Stage1ManifestError(
            f"label[{position}].format must be exactly "
            f"{TRAJECTORY_LABEL_FORMAT!r}"
        )
    version = record.get("version")
    if type(version) is not int or version != TRAJECTORY_LABEL_VERSION:
        raise Stage1ManifestError(
            f"label[{position}].version must be exactly "
            f"{TRAJECTORY_LABEL_VERSION}"
        )
    status = record.get("label_status")
    if status not in {"ok", "unknown"}:
        raise Stage1ManifestError(
            f"label[{position}].label_status must be exactly 'ok' or 'unknown'"
        )
    label_fields = ("refusal_label", "compliance_label", "jailbreak_success")
    if status == "unknown":
        invalid = [
            field
            for field in label_fields
            if field not in record or record.get(field) is not None
        ]
        if invalid:
            raise Stage1ManifestError(
                f"label[{position}] unknown labels must explicitly be null: {invalid}"
            )
        return status

    labels: dict[str, bool] = {}
    for field in label_fields:
        value = record.get(field)
        if field not in record or not isinstance(value, bool):
            raise Stage1ManifestError(
                f"label[{position}].{field} must be an explicit boolean when status=ok"
            )
        labels[field] = value
    if labels["compliance_label"] != labels["jailbreak_success"]:
        raise Stage1ManifestError(
            f"label[{position}] compliance_label and jailbreak_success disagree"
        )
    if labels["refusal_label"] and labels["compliance_label"]:
        raise Stage1ManifestError(
            f"label[{position}] cannot be both refusal and compliant"
        )
    return status


def _behavior_index(
    behavior_labels: Path,
    *,
    project_root: Path,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for position, record in enumerate(
        _read_jsonl_objects(behavior_labels, allow_empty=True)
    ):
        case_id = _text(record.get("case_id"), name=f"label[{position}].case_id")
        pair_id = _text(record.get("pair_id"), name=f"label[{position}].pair_id")
        step = _integer(record.get("step"), name=f"label[{position}].step")
        response = record.get("response")
        if not isinstance(response, str):
            raise Stage1ManifestError(f"label[{position}].response must be a string")
        recorded_digest = _text(
            record.get("response_sha256"),
            name=f"label[{position}].response_sha256",
        )
        if recorded_digest != response_sha256(response):
            raise Stage1ManifestError(
                f"Behavior response hash mismatch for {case_id}/{pair_id}/step {step}"
            )
        _validate_trajectory_label(record, position=position)
        experiment_fingerprint = _sha256_digest(
            record.get("experiment_fingerprint"),
            name=f"label[{position}].experiment_fingerprint",
        )
        checkpoint_digest = _sha256_digest(
            record.get("checkpoint_sha256"),
            name=f"label[{position}].checkpoint_sha256",
        )
        checkpoint = _artifact_path(
            record.get("checkpoint_path"),
            bases=(project_root, behavior_labels.parent),
            name=f"behavior checkpoint step {step}",
        )
        if sha256_file(checkpoint) != checkpoint_digest:
            raise Stage1ManifestError(
                f"Behavior checkpoint SHA-256 mismatch for {case_id}/{pair_id}/step {step}"
            )
        key = (case_id, pair_id, step)
        if key in index:
            raise Stage1ManifestError(
                f"Duplicate behavior label for {case_id}/{pair_id}/step {step}"
            )
        normalized = dict(record)
        normalized["experiment_fingerprint"] = experiment_fingerprint
        normalized["checkpoint_sha256"] = checkpoint_digest
        normalized["_checkpoint_path"] = checkpoint
        index[key] = normalized
    return index


def _clean_label_index(
    labels_path: Path,
    *,
    source_manifest: Path,
    project_root: Path,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    source_digest = sha256_file(source_manifest)
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for position, record in enumerate(
        _read_jsonl_objects(labels_path, allow_empty=True)
    ):
        if record.get("format") != CLEAN_LABEL_FORMAT:
            raise Stage1ManifestError(
                f"clean_label[{position}].format must be exactly "
                f"{CLEAN_LABEL_FORMAT!r}"
            )
        version = record.get("version")
        if type(version) is not int or version != CLEAN_LABEL_VERSION:
            raise Stage1ManifestError(
                f"clean_label[{position}].version must be exactly "
                f"{CLEAN_LABEL_VERSION}"
            )
        pair_id = _text(
            record.get("pair_id"), name=f"clean_label[{position}].pair_id"
        )
        state = record.get("state")
        if state not in {"X_B", "X_H"}:
            raise Stage1ManifestError(
                f"clean_label[{position}].state must be exactly 'X_B' or 'X_H'"
            )
        key = (pair_id, state)
        if key in index:
            raise Stage1ManifestError(
                f"Duplicate clean baseline label for {pair_id}/{state}"
            )

        provenance_path = _artifact_path(
            record.get("source_manifest_path"),
            bases=(project_root, labels_path.parent),
            name=f"clean_label[{position}].source_manifest_path",
        )
        if provenance_path != source_manifest:
            raise Stage1ManifestError(
                f"Clean label source manifest mismatch for {pair_id}/{state}"
            )
        provenance_digest = _sha256_digest(
            record.get("source_manifest_sha256"),
            name=f"clean_label[{position}].source_manifest_sha256",
        )
        if provenance_digest != source_digest:
            raise Stage1ManifestError(
                f"Clean label source manifest SHA-256 mismatch for {pair_id}/{state}"
            )

        prompt = record.get("prompt")
        response = record.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise Stage1ManifestError(
                f"Clean label prompt/response must be strings for {pair_id}/{state}"
            )
        if _sha256_digest(
            record.get("prompt_sha256"),
            name=f"clean_label[{position}].prompt_sha256",
        ) != response_sha256(prompt):
            raise Stage1ManifestError(
                f"Clean prompt SHA-256 mismatch for {pair_id}/{state}"
            )
        if _sha256_digest(
            record.get("response_sha256"),
            name=f"clean_label[{position}].response_sha256",
        ) != response_sha256(response):
            raise Stage1ManifestError(
                f"Clean response SHA-256 mismatch for {pair_id}/{state}"
            )

        audio = _artifact_path(
            record.get("audio_path"),
            bases=(project_root, labels_path.parent),
            name=f"clean_label[{position}].audio_path",
        )
        audio_digest = _sha256_digest(
            record.get("audio_sha256"),
            name=f"clean_label[{position}].audio_sha256",
        )
        if sha256_file(audio) != audio_digest:
            raise Stage1ManifestError(
                f"Clean audio SHA-256 mismatch for {pair_id}/{state}"
            )

        status = record.get("label_status")
        if status not in {"ok", "unknown"}:
            raise Stage1ManifestError(
                f"clean_label[{position}].label_status must be exactly 'ok' or 'unknown'"
            )
        generation_status = record.get("generation_status")
        if generation_status not in {"ok", "error"}:
            raise Stage1ManifestError(
                f"clean_label[{position}].generation_status must be exactly "
                "'ok' or 'error'"
            )
        if status == "ok":
            if generation_status != "ok":
                raise Stage1ManifestError(
                    f"Clean label status=ok requires generation_status=ok for {pair_id}/{state}"
                )
            if "refusal_label" not in record or not isinstance(
                record.get("refusal_label"), bool
            ):
                raise Stage1ManifestError(
                    f"Clean refusal_label must be an explicit boolean for {pair_id}/{state}"
                )
        elif "refusal_label" not in record or record.get("refusal_label") is not None:
            raise Stage1ManifestError(
                f"Unknown clean refusal_label must explicitly be null for {pair_id}/{state}"
            )
        if generation_status == "error" and status != "unknown":
            raise Stage1ManifestError(
                f"Clean generation_status=error requires label_status=unknown "
                f"for {pair_id}/{state}"
            )

        normalized = dict(record)
        normalized["_audio_path"] = audio
        normalized["audio_sha256"] = audio_digest
        normalized["source_manifest_sha256"] = provenance_digest
        index[key] = normalized
    return index


def _candidate_audio_path(
    row: Mapping[str, Any],
    *,
    state: str,
    source_manifest: Path,
    project_root: Path,
) -> Path:
    if state == "X_B":
        values = [("benign_audio_path", row.get("benign_audio_path"))]
    else:
        harmful_audio = row.get("harmful_audio_path")
        if not _is_blank(harmful_audio):
            values = [("harmful_audio_path", harmful_audio)]
        else:
            values = [("clean_audio_path", row.get("clean_audio_path"))]
    resolved: list[tuple[str, Path]] = []
    for field, value in values:
        if _is_blank(value):
            continue
        resolved.append(
            (
                field,
                resolve_project_path(
                    value,
                    base=source_manifest.parent,
                    project_root=project_root,
                    must_exist=True,
                ),
            )
        )
    if not resolved:
        raise Stage1ManifestError(f"Candidate is missing audio for {state}")
    return resolved[0][1]


def attach_clean_outputs(
    source_manifest: str | Path,
    clean_labels: str | Path,
    output_manifest: str | Path,
    *,
    exclusions_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Attach hash-bound X_B/X_H responses and explicit refusal labels.

    Both states must be present and judged successfully.  A pair is retained
    only when its benign response is non-refusing and its clean harmful
    response refuses, as required by the Stage-1 measurement design.
    """

    source = Path(source_manifest).expanduser().resolve()
    labels_path = Path(clean_labels).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    before_digest = sha256_file(source)
    frame = read_table(source)
    pair_ids = _validate_pair_ids(frame, name="Clean candidate manifest")
    labels = _clean_label_index(
        labels_path,
        source_manifest=source,
        project_root=root,
    )
    unknown_pairs = sorted(
        {pair_id for pair_id, _ in labels} - set(pair_ids)
    )
    if unknown_pairs:
        raise Stage1ManifestError(
            "Clean labels contain pair_id values absent from candidates: "
            + ", ".join(unknown_pairs[:5])
        )

    attached_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    state_fields = {
        "X_B": ("benign_text", "benign_response", "benign_refused"),
        "X_H": ("harmful_text", "clean_response", "clean_refused"),
    }
    for _, source_row in frame.iterrows():
        row = source_row.to_dict()
        pair_id = str(row["pair_id"]).strip()
        pair_labels: dict[str, Mapping[str, Any]] = {}
        reasons: list[str] = []
        for state in ("X_B", "X_H"):
            label = labels.get((pair_id, state))
            if label is None:
                reasons.append(f"missing_clean_label:{state}")
                continue
            prompt_field, _, _ = state_fields[state]
            expected_prompt = _text(
                row.get(prompt_field), name=f"{pair_id}.{prompt_field}"
            )
            if label["prompt"] != expected_prompt:
                raise Stage1ManifestError(
                    f"Clean prompt content mismatch for {pair_id}/{state}"
                )
            if label["prompt_sha256"] != response_sha256(expected_prompt):
                raise Stage1ManifestError(
                    f"Clean prompt hash does not match candidate for {pair_id}/{state}"
                )
            expected_audio = _candidate_audio_path(
                row,
                state=state,
                source_manifest=source,
                project_root=root,
            )
            if label["_audio_path"] != expected_audio:
                raise Stage1ManifestError(
                    f"Clean audio path mismatch for {pair_id}/{state}"
                )
            if label["audio_sha256"] != sha256_file(expected_audio):
                raise Stage1ManifestError(
                    f"Clean audio hash does not match candidate for {pair_id}/{state}"
                )
            if label["label_status"] == "unknown":
                reasons.append(f"unknown_clean_label:{state}")
                continue
            pair_labels[state] = label

        if not reasons and pair_labels["X_B"]["refusal_label"]:
            reasons.append("benign_refused")
        if not reasons and not pair_labels["X_H"]["refusal_label"]:
            reasons.append("clean_not_refused")
        if reasons:
            row[EXCLUSION_REASON_COLUMN] = ";".join(reasons)
            exclusion_rows.append(row)
            continue

        benign_label = pair_labels["X_B"]
        harmful_label = pair_labels["X_H"]
        clean_audio = harmful_label["_audio_path"]
        row.update(
            {
                "benign_response": benign_label["response"],
                "benign_response_sha256": benign_label["response_sha256"],
                "benign_refused": benign_label["refusal_label"],
                "benign_audio_sha256": benign_label["audio_sha256"],
                "clean_audio_path": project_relative_path(
                    clean_audio, project_root=root
                ),
                "clean_audio_sha256": harmful_label["audio_sha256"],
                "clean_response": harmful_label["response"],
                "clean_response_sha256": harmful_label["response_sha256"],
                "clean_refused": harmful_label["refusal_label"],
                "clean_behavior_labels_path": project_relative_path(
                    labels_path, project_root=root
                ),
                "clean_source_manifest_sha256": before_digest,
            }
        )
        attached_rows.append(row)

    attached = pd.DataFrame(attached_rows)
    if attached.empty:
        attached = pd.DataFrame(columns=list(frame.columns))
    attached = _normalise_path_columns(
        attached,
        source_path=source,
        project_root=root,
    )
    _atomic_csv(attached, output_manifest)
    exclusion_destination = (
        None if exclusions_path is None else Path(exclusions_path).expanduser().resolve()
    )
    _append_exclusions(pd.DataFrame(exclusion_rows), exclusion_destination)
    if sha256_file(source) != before_digest:
        raise RuntimeError("Clean candidate manifest changed while attaching outputs")
    return attached


def _append_exclusions(frame: pd.DataFrame, path: Path | None) -> None:
    if path is None or frame.empty:
        return
    combined = frame.copy()
    if path.is_file():
        existing = read_table(path)
        if not existing.empty:
            combined = pd.concat([existing, combined], ignore_index=True, sort=False)
    keys = [
        column
        for column in ("pair_id", "case_id", EXCLUSION_REASON_COLUMN)
        if column in combined.columns
    ]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    _atomic_csv(combined, path)


def attach_attack_outputs(
    source_manifest: str | Path,
    batch_summary: str | Path,
    behavior_labels: str | Path,
    output_manifest: str | Path,
    *,
    exclusions_path: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    require_method: str | None = "standard",
    require_full_trajectory: bool = True,
) -> pd.DataFrame:
    """Attach only provenance-verified attack and semantic-judge results.

    The selected PGD state comes from ``history.selection.step``.  Its
    response/success label comes from the offline behavior sidecar, never from
    ``run.json.attack_success`` (which may be a target-substring heuristic).
    """

    source = Path(source_manifest).expanduser().resolve()
    summary_path = Path(batch_summary).expanduser().resolve()
    labels_path = Path(behavior_labels).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    frame = read_table(source)
    pair_ids = _validate_pair_ids(frame, name="Source manifest")
    summary = _read_json_object(summary_path, name="batch summary")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise Stage1ManifestError("Batch summary requires a cases list")
    label_index = _behavior_index(labels_path, project_root=root)
    source_pair_set = set(pair_ids)
    unknown_label_pairs = sorted(
        {pair_id for _, pair_id, _ in label_index if pair_id not in source_pair_set}
    )
    if unknown_label_pairs:
        raise Stage1ManifestError(
            "Behavior labels contain pair_id values absent from source: "
            + ", ".join(unknown_label_pairs[:5])
        )

    cases_by_pair: dict[str, Mapping[str, Any]] = {}
    for position, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise Stage1ManifestError(f"summary case {position} must be an object")
        pair_id = _text(case.get("pair_id"), name=f"summary case {position}.pair_id")
        if pair_id not in source_pair_set:
            raise Stage1ManifestError(
                f"Batch summary contains pair_id absent from source: {pair_id}"
            )
        if pair_id in cases_by_pair:
            raise Stage1ManifestError(f"Duplicate batch case for pair_id {pair_id}")
        cases_by_pair[pair_id] = case
    invalid_label_cases = sorted(
        {
            (label_case, label_pair)
            for label_case, label_pair, _ in label_index
            if label_pair in cases_by_pair
            and label_case
            != str(cases_by_pair[label_pair].get("case_id", "")).strip()
        }
    )
    if invalid_label_cases:
        raise Stage1ManifestError(
            "Behavior label case_id does not match batch summary: "
            + ", ".join(f"{case}/{pair}" for case, pair in invalid_label_cases[:5])
        )

    attached_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    summary_output = summary.get("output_dir")
    output_base = (
        Path(str(summary_output)).expanduser().resolve()
        if not _is_blank(summary_output)
        else summary_path.parent
    )
    for _, source_row in frame.iterrows():
        row = source_row.to_dict()
        pair_id = str(row["pair_id"]).strip()
        case = cases_by_pair.get(pair_id)
        if case is None:
            row[EXCLUSION_REASON_COLUMN] = "missing_attack_case"
            exclusion_rows.append(row)
            continue
        status = str(case.get("status", "")).strip().casefold()
        if status == "failed":
            row["case_id"] = case.get("case_id", "")
            row[EXCLUSION_REASON_COLUMN] = "attack_failed"
            exclusion_rows.append(row)
            continue
        if status not in {"completed", "skipped"}:
            raise Stage1ManifestError(
                f"Unsupported attack status {status!r} for pair_id {pair_id}"
            )
        case_id = _text(case.get("case_id"), name=f"case_id for {pair_id}")
        case_dir = _artifact_path(
            case.get("path"),
            bases=(summary_path.parent, output_base, root),
            name=f"case directory for {pair_id}",
            expect_directory=True,
        )
        run_path = case_dir / "run.json"
        run = _read_json_object(run_path, name="case run")
        if str(run.get("pair_id", "")).strip() != pair_id:
            raise Stage1ManifestError(
                f"run.json pair_id does not match source pair_id {pair_id!r}"
            )
        if str(run.get("case_id", "")).strip() != case_id:
            raise Stage1ManifestError(
                f"run.json case_id does not match summary case_id {case_id!r}"
            )
        if require_method is not None and run.get("method") != require_method:
            raise Stage1ManifestError(
                f"Expected method={require_method!r}, found {run.get('method')!r}"
            )
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise Stage1ManifestError(f"run.json artifacts missing for {pair_id}")
        trajectory_index = _artifact_path(
            artifacts.get("trajectory"),
            bases=(case_dir,),
            name=f"trajectory index for {pair_id}",
        )
        budget = run.get("budget", {})
        if not isinstance(budget, Mapping):
            raise Stage1ManifestError(f"run.json budget missing for {pair_id}")
        experiment_fingerprint = _sha256_digest(
            budget.get("experiment_fingerprint"),
            name=f"run experiment_fingerprint for {pair_id}",
        )
        experiment_config = budget.get("experiment_config")
        if not isinstance(experiment_config, Mapping):
            raise Stage1ManifestError(
                f"run experiment_config must be an object for {pair_id}"
            )
        if _mapping_fingerprint(experiment_config) != experiment_fingerprint:
            raise Stage1ManifestError(
                f"run experiment_config fingerprint mismatch for {pair_id}"
            )
        checkpoints = _trajectory_records(
            trajectory_index,
            case_dir=case_dir,
            case_id=case_id,
            pair_id=pair_id,
            experiment_fingerprint=experiment_fingerprint,
        )
        if require_full_trajectory:
            total_steps = _integer(budget.get("steps"), name="run budget steps")
            expected_steps = set(range(total_steps + 1))
            if set(checkpoints) != expected_steps:
                raise Stage1ManifestError(
                    f"Trajectory for {pair_id} is incomplete: expected steps "
                    f"0...{total_steps}, found {sorted(checkpoints)}"
                )

        history_path = _artifact_path(
            artifacts.get("history"),
            bases=(case_dir,),
            name=f"attack history for {pair_id}",
        )
        history_payload = _read_json_object(history_path, name="attack history")
        if str(history_payload.get("pair_id", "")).strip() != pair_id:
            raise Stage1ManifestError(
                f"history.json pair_id does not match source pair_id {pair_id!r}"
            )
        if str(history_payload.get("case_id", "")).strip() != case_id:
            raise Stage1ManifestError(
                f"history.json case_id does not match summary case_id {case_id!r}"
            )
        history = history_payload.get("history")
        selection = history.get("selection") if isinstance(history, Mapping) else None
        if not isinstance(selection, Mapping):
            raise Stage1ManifestError(f"history selection missing for {pair_id}")
        selected_step = _integer(
            selection.get("step"), name=f"selected attack step for {pair_id}"
        )
        if selected_step not in checkpoints:
            raise Stage1ManifestError(
                f"Selected step {selected_step} is absent from {pair_id} trajectory"
            )

        matching_labels = {
            step: label_index[(case_id, pair_id, step)]
            for step in checkpoints
            if (case_id, pair_id, step) in label_index
        }
        if selected_step not in matching_labels:
            row["case_id"] = case_id
            row[EXCLUSION_REASON_COLUMN] = "missing_selected_behavior_label"
            exclusion_rows.append(row)
            continue
        unexpected_steps = sorted(
            step
            for label_case, label_pair, step in label_index
            if label_case == case_id
            and label_pair == pair_id
            and step not in checkpoints
        )
        if unexpected_steps:
            raise Stage1ManifestError(
                f"Behavior labels contain steps absent from trajectory for {pair_id}: "
                f"{unexpected_steps}"
            )
        mismatch = []
        fingerprint_mismatch = []
        for step, label in matching_labels.items():
            if label["_checkpoint_path"] != checkpoints[step].path:
                mismatch.append(step)
            if label["experiment_fingerprint"] != experiment_fingerprint:
                fingerprint_mismatch.append(step)
        if mismatch:
            raise Stage1ManifestError(
                f"Behavior checkpoint path mismatch for {pair_id} step(s): {mismatch}"
            )
        if fingerprint_mismatch:
            raise Stage1ManifestError(
                f"Behavior experiment_fingerprint mismatch for {pair_id} step(s): "
                f"{fingerprint_mismatch}"
            )

        selected_label = matching_labels[selected_step]
        if selected_label["label_status"] == "unknown":
            row["case_id"] = case_id
            row[EXCLUSION_REASON_COLUMN] = "unknown_selected_behavior_label"
            exclusion_rows.append(row)
            continue

        jailbreak_success = selected_label["jailbreak_success"]
        adversarial_audio = _artifact_path(
            artifacts.get("adversarial_audio"),
            bases=(case_dir,),
            name=f"selected adversarial audio for {pair_id}",
        )
        row.update(
            {
                "case_id": case_id,
                "selected_attack_step": selected_step,
                "experiment_fingerprint": experiment_fingerprint,
                "selected_checkpoint_path": project_relative_path(
                    checkpoints[selected_step].path, project_root=root
                ),
                "selected_checkpoint_sha256": selected_label[
                    "checkpoint_sha256"
                ],
                "behavior_labeled_steps": len(matching_labels),
                "behavior_unknown_steps": sum(
                    label["label_status"] == "unknown"
                    for label in matching_labels.values()
                ),
                "behavior_missing_steps": len(checkpoints) - len(matching_labels),
                "trajectory_path": project_relative_path(
                    trajectory_index, project_root=root
                ),
                "behavior_labels_path": project_relative_path(
                    labels_path, project_root=root
                ),
                "jailbreak_audio_path": project_relative_path(
                    adversarial_audio, project_root=root
                ),
                "jailbreak_response": selected_label["response"],
                "jailbreak_response_sha256": selected_label["response_sha256"],
                "jailbreak_success": jailbreak_success,
            }
        )
        attached_rows.append(row)

    attached = pd.DataFrame(attached_rows, columns=list(frame.columns))
    if attached_rows:
        attached = pd.DataFrame(attached_rows)
    attached = _normalise_path_columns(
        attached,
        source_path=source,
        project_root=root,
    )
    _atomic_csv(attached, output_manifest)
    exclusions = pd.DataFrame(exclusion_rows)
    exclusion_destination = (
        None if exclusions_path is None else Path(exclusions_path).expanduser().resolve()
    )
    _append_exclusions(exclusions, exclusion_destination)
    return attached


def _canonicalize_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    raw_booleans: dict[str, pd.Series] = {}
    temporary = frame.copy()
    for name in ("benign_refused", "clean_refused", "jailbreak_success"):
        if name not in temporary.columns:
            raw_booleans[name] = pd.Series(None, index=temporary.index)
        else:
            raw_booleans[name] = temporary[name].map(
                lambda value, field=name: _optional_bool(value, name=field)
            )
        temporary[name] = pd.NA
    canonical = build_manifest(temporary)
    for name, values in raw_booleans.items():
        canonical[name] = pd.Series(values.tolist(), dtype="boolean")
    return canonical


def finalize_measurement_manifest(
    source_manifest: str | Path,
    output_manifest: str | Path,
    exclusions_path: str | Path,
    *,
    required_split: str = TRAIN_SPLIT,
    project_root: str | Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Keep only complete, leakage-free X_B/X_H/X_J probe triplets."""

    source = Path(source_manifest).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    raw = read_table(source)
    _validate_pair_ids(raw, name="Measurement source")
    explicit_labels = ("benign_refused", "clean_refused", "jailbreak_success")
    missing_explicit = [name for name in explicit_labels if name not in raw.columns]
    if missing_explicit:
        legacy_note = (
            "; legacy attack_success is not accepted as semantic jailbreak_success"
            if "jailbreak_success" in missing_explicit and "attack_success" in raw.columns
            else ""
        )
        raise Stage1ManifestError(
            "Measurement source requires explicit semantic label column(s): "
            + ", ".join(missing_explicit)
            + legacy_note
        )
    if "measurement_split" not in raw.columns:
        raise Stage1ManifestError("Measurement source requires measurement_split")
    split = raw["measurement_split"].fillna("").astype(str).str.strip()
    if (split != required_split).any():
        leaked = raw.loc[split != required_split, "pair_id"].astype(str).tolist()
        raise Stage1ManifestError(
            f"Held-out/non-{required_split} pair(s) cannot enter probe payload: "
            + ", ".join(leaked[:5])
        )
    if "stage1_role" not in raw.columns:
        raise Stage1ManifestError(
            "Measurement source requires stage1_role provenance"
        )
    roles = raw["stage1_role"].fillna("").astype(str).str.strip()
    if (roles != PROBE_ROLE).any():
        raise Stage1ManifestError(
            "Only probe_candidate rows may enter the Stage-1 probe payload"
        )
    canonical = _canonicalize_outcomes(raw)
    canonical = _normalise_path_columns(
        canonical,
        source_path=source,
        project_root=root,
    )

    valid_rows: list[int] = []
    exclusions: list[dict[str, Any]] = []
    required_text = (*STATE_AUDIO_COLUMNS, *COMPLETION_COLUMNS)
    for index, row in canonical.iterrows():
        reasons: list[str] = []
        benign_refused = _optional_bool(
            row["benign_refused"], name="benign_refused"
        )
        clean_refused = _optional_bool(row["clean_refused"], name="clean_refused")
        jailbreak_success = _optional_bool(
            row["jailbreak_success"], name="jailbreak_success"
        )
        if benign_refused is None:
            reasons.append("benign_refused_unknown")
        elif benign_refused:
            reasons.append("benign_refused")
        if clean_refused is None:
            reasons.append("clean_refused_unknown")
        elif not clean_refused:
            reasons.append("clean_not_refused")
        if jailbreak_success is None:
            reasons.append("jailbreak_success_unknown")
        elif not jailbreak_success:
            reasons.append("jailbreak_failed")
        reasons.extend(
            f"missing_{name}" for name in required_text if _is_blank(row[name])
        )
        if reasons:
            excluded = row.to_dict()
            excluded[EXCLUSION_REASON_COLUMN] = ";".join(reasons)
            exclusions.append(excluded)
        else:
            valid_rows.append(index)

    selected = canonical.loc[valid_rows].reset_index(drop=True)
    if selected.empty:
        raise Stage1ManifestError("No complete X_B/X_H/X_J triplets remain")
    try:
        validate_manifest(selected, require_state_triplets=True)
    except ManifestValidationError as exc:
        raise Stage1ManifestError(str(exc)) from exc
    _atomic_csv(selected, output_manifest)
    _append_exclusions(
        pd.DataFrame(exclusions),
        Path(exclusions_path).expanduser().resolve(),
    )
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and verify Stage-1 measurement manifests."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="derive immutable 80/20 candidates")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--expected-train", type=int, default=80)
    prepare.add_argument("--expected-validation", type=int, default=20)
    prepare.add_argument("--expected-train-per-stratum", type=int, default=8)
    prepare.add_argument("--expected-validation-per-stratum", type=int, default=2)

    attach = commands.add_parser("attach", help="attach verified attack/behavior outputs")
    attach.add_argument("--source", type=Path, required=True)
    attach.add_argument("--summary", type=Path, required=True)
    attach.add_argument("--behavior-labels", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--exclusions", type=Path)

    clean = commands.add_parser(
        "attach-clean", help="attach verified clean X_B/X_H baseline labels"
    )
    clean.add_argument("--source", type=Path, required=True)
    clean.add_argument("--clean-labels", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    clean.add_argument("--exclusions", type=Path)

    finalize = commands.add_parser("finalize", help="select strict probe triplets")
    finalize.add_argument("--source", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--exclusions", type=Path, required=True)
    finalize.add_argument("--required-split", default=TRAIN_SPLIT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        outputs = prepare_candidate_manifests(
            args.source,
            args.output_dir,
            expected_train=args.expected_train,
            expected_validation=args.expected_validation,
            expected_train_per_stratum=args.expected_train_per_stratum,
            expected_validation_per_stratum=args.expected_validation_per_stratum,
        )
        print(", ".join(f"{key}={value}" for key, value in outputs.items()))
    elif args.command == "attach":
        attached = attach_attack_outputs(
            args.source,
            args.summary,
            args.behavior_labels,
            args.output,
            exclusions_path=args.exclusions,
        )
        print(f"Attached {len(attached)} verified attack row(s) to {args.output}")
    elif args.command == "attach-clean":
        attached = attach_clean_outputs(
            args.source,
            args.clean_labels,
            args.output,
            exclusions_path=args.exclusions,
        )
        print(f"Attached {len(attached)} clean baseline row(s) to {args.output}")
    else:
        selected = finalize_measurement_manifest(
            args.source,
            args.output,
            args.exclusions,
            required_split=args.required_split,
        )
        print(f"Saved {len(selected)} complete triplet row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROBE_ROLE",
    "PROJECT_ROOT",
    "Stage1ManifestError",
    "TRAIN_SPLIT",
    "TRAJECTORY_ROLE",
    "VALIDATION_SPLIT",
    "attach_attack_outputs",
    "attach_clean_outputs",
    "finalize_measurement_manifest",
    "prepare_candidate_manifests",
    "project_relative_path",
    "resolve_project_path",
    "response_sha256",
    "sha256_file",
]
