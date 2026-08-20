#!/usr/bin/env python3
"""Collect paired X_B/X_H/X_J and optional PGD trajectory hidden states.

The resulting checkpoint is directly consumable by
``experiments.train_safety_probes``. Heavy model, audio, and PyTorch imports are
kept behind runtime functions so ``--help`` remains offline and lightweight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


COLLECTION_FORMAT = "safety-state-probe-training"
COLLECTION_VERSION = 1
BEHAVIOR_LABEL_FORMAT = "stage1-behavior-label"
BEHAVIOR_LABEL_VERSION = 1
BASE_STATES = ("X_B", "X_H", "X_J")
TRAJECTORY_STATE = "trajectory"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAJECTORY_COLUMNS = (
    "trajectory_path",
    "trajectory_index_path",
    "trajectory_index",
    "trajectory",
)
_CHECKPOINT_KEYS = (
    "checkpoint_path",
    "checkpoint",
    "tensor_path",
    "artifact_path",
    "path",
)
_WAVEFORM_KEYS = ("adversarial_wav", "waveform", "wav", "audio")
_RESPONSE_KEYS = (
    "response",
    "adversarial_response",
    "generated_response",
    "generated_text",
    "output_text",
)
_REFUSAL_KEYS = ("refusal_label", "refused", "is_refused", "refusal")
_SUCCESS_KEYS = ("jailbreak_success", "attack_success", "is_success", "jailbroken")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StateCollectionError(ValueError):
    """Raised when collected states would violate the training schema."""


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    """One resolved, pair-owned trajectory checkpoint."""

    step: int
    path: Path
    metadata: Mapping[str, Any]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        comparison = value != value
        return comparison if isinstance(comparison, bool) else bool(comparison)
    except (TypeError, ValueError):
        return False


def _optional_bool(value: Any, *, name: str) -> Optional[bool]:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "t", "yes", "y", "1", "refused", "success"}:
        return True
    if text in {"false", "f", "no", "n", "0", "complied", "failed", "failure"}:
        return False
    raise StateCollectionError(f"{name} must be a boolean value, got {value!r}")


def _json_safe(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StateCollectionError(
            f"{name} must contain only finite JSON-compatible metadata"
        ) from exc


def _required_sha256(value: Any, *, name: str) -> str:
    """Return one canonical lowercase SHA-256 digest or reject it."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise StateCollectionError(
            f"{name} must be a 64-character lowercase SHA-256 digest"
        )
    return value


def build_model_provenance(
    model_name: str,
    model_id: Optional[str],
    dtype: str,
) -> Mapping[str, str]:
    """Return the canonical, lightweight model identity used by Stage 1.

    Hashing multi-gigabyte model shards for every command would be needlessly
    expensive. Stage 1 instead binds artifacts to the resolved model
    arguments. The replay entry point uses the same canonical JSON digest.
    """

    from models import DEFAULT_MODEL_IDS

    normalized_name = str(model_name).strip()
    normalized_dtype = str(dtype).strip()
    if not normalized_name:
        raise StateCollectionError("model_name must be non-blank")
    if not normalized_dtype:
        raise StateCollectionError("dtype must be non-blank")
    resolved_model_id = (
        DEFAULT_MODEL_IDS.get(normalized_name, normalized_name)
        if model_id is None
        else str(model_id).strip()
    )
    if not resolved_model_id:
        raise StateCollectionError("model_id must be non-blank")
    path = Path(resolved_model_id).expanduser()
    if path.exists():
        resolved_model_id = str(path.resolve())
    canonical = {
        "model_name": normalized_name,
        "model_id": resolved_model_id,
        "dtype": normalized_dtype,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **canonical,
        "model_fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _normalize_model_provenance(value: Mapping[str, Any]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise StateCollectionError("model_provenance must be a mapping")
    normalized = build_model_provenance(
        str(value.get("model_name", "")),
        None if value.get("model_id") is None else str(value.get("model_id")),
        str(value.get("dtype", "")),
    )
    supplied = value.get("model_fingerprint")
    if supplied is not None and _required_sha256(
        supplied, name="model_provenance.model_fingerprint"
    ) != normalized["model_fingerprint"]:
        raise StateCollectionError("model_provenance fingerprint mismatch")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save one tensor-and-primitive checkpoint."""

    import torch

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
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


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old supported torch only
        raise RuntimeError(
            "Safe state collection requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StateCollectionError(f"Trajectory checkpoint is not a mapping: {path}")
    return payload


def _resolve_local_path(base: Path, value: Any, *, name: str) -> Path:
    if _is_blank(value):
        raise StateCollectionError(f"{name} is required")
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    manifest_relative = (base / path).resolve()
    project_relative = (PROJECT_ROOT / path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    if project_relative.exists():
        return project_relative
    return manifest_relative


def _load_behavior_labels(
    path: Path,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """Load hash-bound explicit labels keyed by case, pair, and step."""

    labels: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for line_number, row in enumerate(_read_json_lines(path), start=1):
        if row.get("format") != BEHAVIOR_LABEL_FORMAT:
            raise StateCollectionError(
                f"{path}:{line_number}: format must be "
                f"{BEHAVIOR_LABEL_FORMAT!r}"
            )
        version = row.get("version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != BEHAVIOR_LABEL_VERSION
        ):
            raise StateCollectionError(
                f"{path}:{line_number}: version must be "
                f"{BEHAVIOR_LABEL_VERSION}"
            )
        case_id = str(row.get("case_id", "")).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        raw_step = row.get("step")
        if not case_id or not pair_id:
            raise StateCollectionError(
                f"{path}:{line_number}: case_id and pair_id are required"
            )
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
            raise StateCollectionError(
                f"{path}:{line_number}: step must be a non-negative integer"
            )
        response = row.get("response")
        if not isinstance(response, str):
            raise StateCollectionError(
                f"{path}:{line_number}: response must be a string"
            )
        response_digest = _required_sha256(
            row.get("response_sha256"),
            name=f"{path}:{line_number}: response_sha256",
        )
        if response_digest != hashlib.sha256(response.encode("utf-8")).hexdigest():
            raise StateCollectionError(
                f"{path}:{line_number}: response_sha256 does not match response"
            )
        experiment_fingerprint = _required_sha256(
            row.get("experiment_fingerprint"),
            name=f"{path}:{line_number}: experiment_fingerprint",
        )
        checkpoint_digest = _required_sha256(
            row.get("checkpoint_sha256"),
            name=f"{path}:{line_number}: checkpoint_sha256",
        )
        if _is_blank(row.get("checkpoint_path")):
            raise StateCollectionError(
                f"{path}:{line_number}: checkpoint_path is required"
            )
        generation_status = str(row.get("generation_status", "")).strip().casefold()
        if generation_status not in {"ok", "error"}:
            raise StateCollectionError(
                f"{path}:{line_number}: generation_status must be 'ok' or 'error'"
            )
        status = str(row.get("label_status", "")).strip().casefold()
        if status not in {"ok", "unknown"}:
            raise StateCollectionError(
                f"{path}:{line_number}: label_status must be 'ok' or 'unknown'"
            )
        if status == "ok" and generation_status != "ok":
            raise StateCollectionError(
                f"{path}:{line_number}: label_status='ok' requires "
                "generation_status='ok'"
            )
        if generation_status == "error" and status != "unknown":
            raise StateCollectionError(
                f"{path}:{line_number}: generation_status='error' requires "
                "label_status='unknown'"
            )
        if status == "ok":
            refusal = _optional_bool(
                row.get("refusal_label"), name="behavior refusal_label"
            )
            compliance = _optional_bool(
                row.get("compliance_label"), name="behavior compliance_label"
            )
            success = _optional_bool(
                row.get("jailbreak_success"), name="behavior jailbreak_success"
            )
            if refusal is None or compliance is None or success is None:
                raise StateCollectionError(
                    f"{path}:{line_number}: ok labels require explicit refusal, "
                    "compliance, and jailbreak values"
                )
            if compliance != success or (refusal and compliance):
                raise StateCollectionError(
                    f"{path}:{line_number}: inconsistent explicit behavior labels"
                )
        identity = (case_id, pair_id, raw_step)
        if identity in labels:
            raise StateCollectionError(
                f"{path}:{line_number}: duplicate behavior-label identity {identity}"
            )
        normalized = dict(row)
        normalized["format"] = BEHAVIOR_LABEL_FORMAT
        normalized["version"] = BEHAVIOR_LABEL_VERSION
        normalized["generation_status"] = generation_status
        normalized["label_status"] = status
        normalized["response_sha256"] = response_digest
        normalized["experiment_fingerprint"] = experiment_fingerprint
        normalized["checkpoint_sha256"] = checkpoint_digest
        labels[identity] = normalized
    if not labels:
        raise StateCollectionError(f"Behavior-label sidecar is empty: {path}")
    return labels


def _read_manifest(
    manifest: Any,
    *,
    manifest_base: str | Path | None,
) -> tuple[Any, Path, Optional[Path]]:
    import pandas as pd

    from data.build_safety_pairs import build_manifest, validate_manifest
    from data.datasets import read_table

    source_path: Optional[Path]
    if isinstance(manifest, (str, Path)):
        source_path = Path(manifest).expanduser().resolve()
        frame = read_table(source_path)
        base = source_path.parent
    elif isinstance(manifest, pd.DataFrame):
        source_path = None
        frame = manifest.copy()
        base = Path(manifest_base or Path.cwd()).expanduser().resolve()
    else:
        raise TypeError("manifest must be a local path or pandas.DataFrame")
    canonical = build_manifest(frame)
    validate_manifest(canonical)
    return canonical, base, source_path


def _normalize_states(states: Iterable[str]) -> tuple[str, ...]:
    if isinstance(states, (str, bytes)):
        states = (str(states),)
    normalized: list[str] = []
    for state in states:
        value = str(state).strip().upper()
        if value not in BASE_STATES:
            raise ValueError(f"states must be selected from {BASE_STATES}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("states must not be empty")
    return tuple(normalized)


def _state_rows(
    manifest: Any,
    *,
    states: tuple[str, ...],
    require_triplets: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    from data.build_safety_pairs import select_state_sets

    views = select_state_sets(manifest)
    triplet_ids = set(views["X_J"]["pair_id"].astype(str))
    if require_triplets and not triplet_ids:
        raise StateCollectionError(
            "No complete X_B/X_H/X_J pair is available for state collection"
        )

    indexed = {
        state: {
            str(row["pair_id"]): row
            for row in views[state].to_dict(orient="records")
        }
        for state in states
    }
    examples: list[dict[str, Any]] = []
    selected_pairs: set[str] = set()
    for row in manifest.to_dict(orient="records"):
        pair_id = str(row["pair_id"])
        if require_triplets and pair_id not in triplet_ids:
            continue
        available = [state for state in states if pair_id in indexed[state]]
        if require_triplets and len(available) != len(states):
            raise StateCollectionError(
                f"pair_id {pair_id!r} is missing a requested base state"
            )
        for state in available:
            example = dict(indexed[state][pair_id])
            example["origin"] = "manifest"
            example["step"] = None
            examples.append(example)
        if available:
            selected_pairs.add(pair_id)
    if not examples:
        raise StateCollectionError("No rows match the requested safety states")
    return examples, selected_pairs


def _read_json_lines(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateCollectionError(
                    f"{path}:{line_number}: invalid trajectory JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise StateCollectionError(
                    f"{path}:{line_number}: expected one JSON object"
                )
            rows.append(value)
    return rows


def _read_trajectory_table(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        rows = _read_json_lines(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, Mapping):
            value = value.get("checkpoints", value.get("records", value.get("data")))
        if not isinstance(value, list) or any(
            not isinstance(row, Mapping) for row in value
        ):
            raise StateCollectionError(
                f"{path}: expected a checkpoint list or checkpoints object"
            )
        rows = list(value)
    elif suffix in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(
                csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            )
    else:
        raise StateCollectionError(
            f"Unsupported trajectory index format {suffix!r}: {path}"
        )
    if not rows:
        raise StateCollectionError(f"Trajectory index has no checkpoints: {path}")
    return rows


def _resolve_trajectory_source(source: str | Path) -> tuple[Optional[Path], list[Path]]:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        if path.suffix.casefold() == ".pt":
            return None, [path]
        return path, []
    if not path.is_dir():
        raise FileNotFoundError(f"Trajectory source does not exist: {path}")
    for candidate in (
        path / "trajectory_index.jsonl",
        path / "trajectory" / "trajectory_index.jsonl",
        path / "index.json",
        path / "trajectory" / "index.json",
    ):
        if candidate.is_file():
            return candidate, []
    checkpoints = sorted(path.glob("checkpoint_*.pt"))
    if not checkpoints:
        checkpoints = sorted((path / "trajectory").glob("checkpoint_*.pt"))
    if not checkpoints:
        checkpoints = sorted(path.glob("step_*.pt"))
    if not checkpoints:
        checkpoints = sorted((path / "trajectory").glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No trajectory index or checkpoints below {path}")
    return None, checkpoints


def _record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise StateCollectionError("trajectory metadata must be an object")
    result = dict(metadata)
    for key, value in record.items():
        if key not in {"metadata", *_CHECKPOINT_KEYS}:
            result.setdefault(str(key), value)
    return result


def _checkpoint_value(record: Mapping[str, Any]) -> Any:
    for key in _CHECKPOINT_KEYS:
        value = record.get(key)
        if isinstance(value, Mapping):
            for nested in ("tensor", "tensors", "checkpoint", "path"):
                if not _is_blank(value.get(nested)):
                    return value[nested]
        elif not _is_blank(value):
            return value
    files = record.get("files")
    if isinstance(files, Mapping):
        for key in ("checkpoint", "tensor", "tensors"):
            if not _is_blank(files.get(key)):
                return files[key]
    return None


def _step_from_name(path: Path) -> Optional[int]:
    match = re.search(r"(?:checkpoint|step)[_-]?(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _resolve_checkpoint_path(index_path: Path, raw: Any, step: int) -> Path:
    candidates: list[Path] = []
    if not _is_blank(raw):
        value = Path(str(raw).strip()).expanduser()
        if value.is_absolute():
            candidates.append(value)
        else:
            candidates.append(index_path.parent / value)
            if index_path.parent.name == "trajectory":
                candidates.append(index_path.parent.parent / value)
    candidates.extend(
        (
            index_path.parent / f"checkpoint_{step:06d}.pt",
            index_path.parent / f"checkpoint_{step}.pt",
            index_path.parent / f"step_{step:06d}.pt",
            index_path.parent / f"step_{step}.pt",
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Trajectory step {step} points to a missing checkpoint below {index_path.parent}"
    )


def load_trajectory_checkpoints(
    source: str | Path,
    *,
    pair_id: str,
) -> tuple[TrajectoryCheckpoint, ...]:
    """Resolve new JSONL, legacy JSON, or simple tabular trajectory indices."""

    index_path, direct_paths = _resolve_trajectory_source(source)
    checkpoints: list[TrajectoryCheckpoint] = []
    if index_path is None:
        for position, path in enumerate(direct_paths):
            step = _step_from_name(path)
            checkpoints.append(
                TrajectoryCheckpoint(
                    step=position if step is None else step,
                    path=path.resolve(),
                    metadata={},
                )
            )
    else:
        for position, record in enumerate(_read_trajectory_table(index_path)):
            metadata = _record_metadata(record)
            raw_step = record.get("step", metadata.get("step"))
            if _is_blank(raw_step):
                raise StateCollectionError(
                    f"{index_path}: checkpoint row {position} is missing step"
                )
            if isinstance(raw_step, bool):
                raise StateCollectionError("trajectory step must be an integer")
            try:
                step = int(raw_step)
            except (TypeError, ValueError) as exc:
                raise StateCollectionError("trajectory step must be an integer") from exc
            if step < 0 or str(raw_step).strip() not in {str(step), f"{step}.0"}:
                raise StateCollectionError(
                    f"trajectory step must be a non-negative integer, got {raw_step!r}"
                )
            checkpoint_path = _resolve_checkpoint_path(
                index_path, _checkpoint_value(record), step
            )
            checkpoints.append(
                TrajectoryCheckpoint(step=step, path=checkpoint_path, metadata=metadata)
            )

    seen: set[int] = set()
    normalized: list[TrajectoryCheckpoint] = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.step):
        if checkpoint.step in seen:
            raise StateCollectionError(
                f"pair_id {pair_id!r} has duplicate trajectory step {checkpoint.step}"
            )
        seen.add(checkpoint.step)
        owner = checkpoint.metadata.get("pair_id")
        if not _is_blank(owner) and str(owner).strip() != pair_id:
            raise StateCollectionError(
                f"trajectory pair_id {owner!r} does not match manifest pair_id {pair_id!r}"
            )
        normalized.append(checkpoint)
    if not normalized:
        raise StateCollectionError(f"pair_id {pair_id!r} has an empty trajectory")
    return tuple(normalized)


def _metadata_lookup(metadata: Mapping[str, Any], keys: Sequence[str]) -> Any:
    sources = [metadata]
    for nested_key in ("behavior", "labels", "result", "run"):
        nested = metadata.get(nested_key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        for key in keys:
            if key in source and not _is_blank(source[key]):
                return source[key]
    return None


def _trajectory_example(
    checkpoint: TrajectoryCheckpoint,
    *,
    pair_row: Mapping[str, Any],
    behavior_label: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, Any], Any]:
    import torch

    payload = _safe_torch_load(checkpoint.path)
    tensors = payload.get("tensors", payload)
    if not isinstance(tensors, Mapping):
        raise StateCollectionError(
            f"trajectory tensors must be a mapping: {checkpoint.path}"
        )
    waveform = None
    for key in _WAVEFORM_KEYS:
        value = tensors.get(key)
        if isinstance(value, torch.Tensor):
            waveform = value
            break
    if waveform is None:
        raise StateCollectionError(
            f"{checkpoint.path} does not contain any of {_WAVEFORM_KEYS}"
        )
    payload_metadata = payload.get("metadata", {})
    if payload_metadata is None:
        payload_metadata = {}
    if not isinstance(payload_metadata, Mapping):
        raise StateCollectionError(
            f"checkpoint metadata must be an object: {checkpoint.path}"
        )
    metadata = dict(checkpoint.metadata)
    metadata.update(payload_metadata)
    owner = _metadata_lookup(metadata, ("pair_id",))
    pair_id = str(pair_row["pair_id"])
    if not _is_blank(owner) and str(owner).strip() != pair_id:
        raise StateCollectionError(
            f"checkpoint pair_id {owner!r} does not match manifest pair_id {pair_id!r}"
        )
    case_value = _metadata_lookup(metadata, ("case_id",))
    if _is_blank(case_value):
        case_value = pair_row.get("case_id")
    case_id = "" if _is_blank(case_value) else str(case_value).strip()
    if behavior_label is None:
        response_value = _metadata_lookup(metadata, _RESPONSE_KEYS)
        response = "" if _is_blank(response_value) else str(response_value).strip()
        refusal_value = _metadata_lookup(metadata, _REFUSAL_KEYS)
        refusal = _optional_bool(refusal_value, name="trajectory refusal")
        success_value = _metadata_lookup(metadata, _SUCCESS_KEYS)
        success = _optional_bool(success_value, name="trajectory success")
        response_sha256 = None
        experiment_fingerprint = None
        checkpoint_sha256 = None
        compliance = None
    else:
        response = str(behavior_label["response"])
        refusal = _optional_bool(
            behavior_label.get("refusal_label"), name="behavior refusal_label"
        )
        success = _optional_bool(
            behavior_label.get("jailbreak_success"), name="behavior jailbreak_success"
        )
        compliance = _optional_bool(
            behavior_label.get("compliance_label"), name="behavior compliance_label"
        )
        response_sha256 = str(behavior_label["response_sha256"])
        experiment_fingerprint = str(behavior_label["experiment_fingerprint"])
        checkpoint_sha256 = str(behavior_label["checkpoint_sha256"])
    example = {
        "case_id": case_id,
        "pair_id": pair_id,
        "source": str(pair_row.get("source", "")),
        "stratum": str(pair_row.get("stratum", "")),
        "state": TRAJECTORY_STATE,
        "text": str(pair_row["harmful_text"]),
        "audio_path": str(checkpoint.path),
        "response": response,
        "harmfulness_label": 1,
        "refusal_label": refusal,
        "jailbreak_success": success,
        "compliance_label": compliance,
        "response_sha256": response_sha256,
        "experiment_fingerprint": experiment_fingerprint,
        "checkpoint_sha256": checkpoint_sha256,
        "origin": "trajectory",
        "step": checkpoint.step,
        "checkpoint_path": str(checkpoint.path),
        "measurement_split": pair_row.get("measurement_split"),
        "stage1_role": pair_row.get("stage1_role"),
        "probe_split": pair_row.get("probe_split"),
    }
    return example, waveform


def _judge_response(judge: Any, prompt: str, response: str) -> tuple[bool, Mapping[str, Any]]:
    """Require an explicit refusal decision from a judge result."""

    if hasattr(judge, "evaluate"):
        result = judge.evaluate(prompt, response)
    elif callable(judge):
        result = judge(prompt, response)
    else:
        raise TypeError("judge must be callable or expose evaluate(prompt, response)")
    if hasattr(result, "as_dict") and callable(result.as_dict):
        result = result.as_dict()
    elif isinstance(result, bool):
        raise StateCollectionError(
            "a boolean judge result is ambiguous; return an explicit refusal field"
        )
    if not isinstance(result, Mapping):
        raise StateCollectionError(
            "judge must return a mapping or behavior decision"
        )
    normalized = dict(result)
    refusal_value = _metadata_lookup(normalized, _REFUSAL_KEYS)
    success_value = _metadata_lookup(normalized, _SUCCESS_KEYS)
    refusal = _optional_bool(refusal_value, name="judge refusal")
    success = _optional_bool(success_value, name="judge attack_success")
    if refusal is None:
        raise StateCollectionError(
            "judge result must contain an explicit refusal label; "
            "attack_success cannot infer refusal"
        )
    normalized.setdefault("refusal", refusal)
    if success is not None:
        normalized.setdefault("attack_success", success)
    return bool(refusal), _json_safe(normalized, name="judge result")


def _generate_response(model: Any, waveform: Any, *, max_tokens: int) -> str:
    import torch

    with torch.no_grad():
        response = model.generate(
            waveform,
            max_tokens=max_tokens,
            temperature=1.0,
            do_sample=False,
        )
    if not isinstance(response, str):
        raise StateCollectionError("model.generate must return a string")
    return response.strip()


def _pool_forward_states(
    model: Any,
    waveform: Any,
    *,
    target_text: str,
    layers: Optional[Sequence[int]],
    pooling: str,
    token_span: str,
    sequence_has_embedding: bool,
) -> "OrderedDict[Any, Any]":
    import torch

    from core.activations import collect_hidden_states

    if not isinstance(waveform, torch.Tensor):
        raise TypeError("audio_loader and trajectory checkpoints must provide tensors")
    with torch.no_grad():
        forward = model.forward_attack(
            waveform, target_text, output_hidden_states=True
        )
    hidden_states = (
        forward.get("hidden_states")
        if isinstance(forward, Mapping)
        else getattr(forward, "hidden_states", None)
    )
    if hidden_states is None:
        raise StateCollectionError("model.forward_attack returned no hidden_states")
    attention_mask = (
        forward.get("attention_mask")
        if isinstance(forward, Mapping)
        else getattr(forward, "attention_mask", None)
    )
    token_spans = (
        forward.get("token_spans", {})
        if isinstance(forward, Mapping)
        else getattr(forward, "token_spans", {})
    )
    selection = None
    if token_span != "all":
        if not isinstance(token_spans, Mapping) or token_span not in token_spans:
            raise StateCollectionError(
                f"model forward does not expose the requested {token_span!r} token span"
            )
        span = token_spans[token_span]
        if (
            not isinstance(span, Sequence)
            or isinstance(span, (str, bytes))
            or len(span) != 2
        ):
            raise StateCollectionError(f"invalid {token_span!r} token span: {span!r}")
        selection = (int(span[0]), int(span[1]))
    pooled = collect_hidden_states(
        hidden_states,
        layers=layers,
        pooling=pooling,
        attention_mask=attention_mask,
        token_selection=selection,
        sequence_has_embedding=sequence_has_embedding,
    )
    result: "OrderedDict[Any, Any]" = OrderedDict()
    for layer, value in pooled.items():
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise StateCollectionError(
                f"pooled hidden state for layer {layer!r} must have shape [D]"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise StateCollectionError(
                f"pooled hidden state for layer {layer!r} contains non-finite values"
            )
        result[layer] = tensor.contiguous()
    if not result:
        raise StateCollectionError("no hidden-state layers were selected")
    return result


def _default_audio_loader(path: str | Path, *, target_sr: int) -> Any:
    from core.audio import load_audio

    return load_audio(str(path), target_sr=target_sr)


def _trajectory_source_for_row(
    row: Mapping[str, Any],
    *,
    pair_id: str,
    base: Path,
    trajectory_paths: Optional[Mapping[str, str | Path]],
    trajectory_column: str,
) -> Optional[Path]:
    value = None
    if trajectory_paths is not None and pair_id in trajectory_paths:
        value = trajectory_paths[pair_id]
    else:
        for column in (trajectory_column, *_TRAJECTORY_COLUMNS):
            if column in row and not _is_blank(row[column]):
                value = row[column]
                break
    if _is_blank(value):
        return None
    return _resolve_local_path(base, value, name=f"trajectory for pair_id {pair_id!r}")


def _append_collected_row(
    *,
    example: MutableMapping[str, Any],
    waveform: Any,
    model: Any,
    judge: Any,
    generate_missing_responses: bool,
    max_tokens: int,
    layers: Optional[Sequence[int]],
    pooling: str,
    token_span: str,
    sequence_has_embedding: bool,
    layer_rows: MutableMapping[Any, list[Any]],
    row_metadata: list[dict[str, Any]],
    pair_ids: list[str],
    states: list[str],
    steps: list[int],
    responses: list[str],
    harmfulness_labels: list[int],
    refusal_labels: list[int],
    behavior_labels: list[dict[str, Any]],
) -> None:
    response = str(example.get("response") or "")
    if not response.strip() and generate_missing_responses:
        response = _generate_response(model, waveform, max_tokens=max_tokens)
    refusal = _optional_bool(example.get("refusal_label"), name="refusal_label")
    judge_result: Optional[Mapping[str, Any]] = None
    if judge is not None:
        if not response:
            raise StateCollectionError("a non-empty response is required by the judge")
        refusal, judge_result = _judge_response(
            judge, str(example["text"]), response
        )
    if refusal is None:
        raise StateCollectionError(
            f"pair_id {example['pair_id']!r} state {example['state']!r} has no "
            "refusal label; store one in trajectory metadata or inject a judge"
        )
    target_text = response or str(example["text"])
    pooled = _pool_forward_states(
        model,
        waveform,
        target_text=target_text,
        layers=layers,
        pooling=pooling,
        token_span=token_span,
        sequence_has_embedding=sequence_has_embedding,
    )
    if not layer_rows:
        layer_rows.update((layer, []) for layer in pooled)
    if tuple(layer_rows) != tuple(pooled):
        raise StateCollectionError("selected hidden-state layers changed across rows")
    for layer, tensor in pooled.items():
        if layer_rows[layer] and tensor.shape != layer_rows[layer][0].shape:
            raise StateCollectionError(
                f"hidden size for layer {layer!r} changed across rows"
            )
        layer_rows[layer].append(tensor)

    pair_id = str(example["pair_id"]).strip()
    state = str(example["state"])
    step_value = example.get("step")
    step = -1 if step_value is None else int(step_value)
    harmfulness = int(example["harmfulness_label"])
    success = _optional_bool(
        example.get("state_jailbreak_success", example.get("jailbreak_success")),
        name="jailbreak_success"
    )
    if judge_result is not None:
        judged_success = _optional_bool(
            _metadata_lookup(judge_result, _SUCCESS_KEYS),
            name="judge attack_success",
        )
        if judged_success is not None:
            success = judged_success
    behavior = {
        "harmfulness": harmfulness,
        "refusal": int(refusal),
        "jailbreak_success": success,
    }
    compliance = _optional_bool(
        example.get("compliance_label"), name="compliance_label"
    )
    if compliance is not None:
        behavior["compliance"] = compliance
    if judge_result is not None:
        behavior["judge"] = judge_result
    metadata = {
        "row_index": len(row_metadata),
        "pair_id": pair_id,
        "case_id": str(example.get("case_id", "")),
        "state": state,
        "step": None if step < 0 else step,
        "source": str(example.get("source", "")),
        "stratum": str(example.get("stratum", "")),
        "prompt": str(example["text"]),
        "audio_path": str(example["audio_path"]),
        "response": response,
        "origin": str(example.get("origin", "manifest")),
        "harmfulness_label": harmfulness,
        "refusal_label": int(refusal),
        "jailbreak_success": success,
    }
    for provenance_name in ("measurement_split", "stage1_role", "probe_split"):
        provenance_value = example.get(provenance_name)
        if not _is_blank(provenance_value):
            metadata[provenance_name] = str(provenance_value).strip()
    response_sha256 = example.get("response_sha256")
    if not _is_blank(response_sha256):
        actual_digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
        if str(response_sha256) != actual_digest:
            raise StateCollectionError("behavior-label response hash changed during merge")
        metadata["response_sha256"] = actual_digest
        behavior["response_sha256"] = actual_digest
    for provenance_name in ("experiment_fingerprint", "checkpoint_sha256"):
        provenance_value = example.get(provenance_name)
        if not _is_blank(provenance_value):
            digest = _required_sha256(
                provenance_value,
                name=f"behavior-label {provenance_name}",
            )
            metadata[provenance_name] = digest
            behavior[provenance_name] = digest
    if example.get("checkpoint_path"):
        metadata["checkpoint_path"] = str(example["checkpoint_path"])
    if judge_result is not None:
        metadata["judge"] = judge_result
    row_metadata.append(_json_safe(metadata, name="row metadata"))
    pair_ids.append(pair_id)
    states.append(state)
    steps.append(step)
    responses.append(response)
    harmfulness_labels.append(harmfulness)
    refusal_labels.append(int(refusal))
    behavior_labels.append(_json_safe(behavior, name="behavior labels"))


def validate_collection_payload(
    payload: Mapping[str, Any],
    *,
    require_both_classes: bool = True,
) -> int:
    """Strictly validate dimensions, row identity, and pair/state ownership."""

    import torch

    required = {
        "hidden_states",
        "harmfulness_labels",
        "refusal_labels",
        "pair_ids",
        "states",
        "steps",
        "layers",
        "responses",
        "behavior_labels",
        "row_metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise StateCollectionError("collection payload is missing: " + ", ".join(missing))
    hidden = payload["hidden_states"]
    if not isinstance(hidden, Mapping) or not hidden:
        raise StateCollectionError("hidden_states must be a non-empty layer mapping")
    sample_count: Optional[int] = None
    for layer, tensor in hidden.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise StateCollectionError(
                f"hidden_states[{layer!r}] must have shape [N, D]"
            )
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise StateCollectionError(
                f"hidden_states[{layer!r}] must be finite floating point"
            )
        if sample_count is None:
            sample_count = int(tensor.shape[0])
        elif tensor.shape[0] != sample_count:
            raise StateCollectionError("all layers must use the same sample count")
    assert sample_count is not None
    if list(payload["layers"]) != list(hidden.keys()):
        raise StateCollectionError("layers must exactly match hidden_states keys")
    for name in (
        "pair_ids",
        "states",
        "responses",
        "behavior_labels",
        "row_metadata",
    ):
        value = payload[name]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise StateCollectionError(f"{name} must be a sequence")
        if len(value) != sample_count:
            raise StateCollectionError(f"{name} must have length {sample_count}")
    for name in ("harmfulness_labels", "refusal_labels", "steps"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.shape != (sample_count,):
            raise StateCollectionError(f"{name} must have shape [{sample_count}]")
    for name in ("harmfulness_labels", "refusal_labels"):
        labels = payload[name].detach().cpu()
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise StateCollectionError(f"{name} must contain only 0 and 1")
        if require_both_classes and labels.unique().numel() != 2:
            raise StateCollectionError(f"{name} must contain both binary classes")
    if payload["steps"].is_floating_point() or payload["steps"].is_complex():
        raise StateCollectionError("steps must use an integer tensor dtype")

    seen: set[tuple[str, str, int]] = set()
    base_by_pair: dict[str, set[str]] = {}
    for index in range(sample_count):
        pair_id = payload["pair_ids"][index]
        state = payload["states"][index]
        response = payload["responses"][index]
        behavior = payload["behavior_labels"][index]
        harmfulness = int(payload["harmfulness_labels"][index])
        refusal = int(payload["refusal_labels"][index])
        step = int(payload["steps"][index])
        metadata = payload["row_metadata"][index]
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise StateCollectionError(f"pair_ids[{index}] must be non-empty")
        if state not in (*BASE_STATES, TRAJECTORY_STATE):
            raise StateCollectionError(f"states[{index}] is invalid: {state!r}")
        if not isinstance(response, str):
            raise StateCollectionError(f"responses[{index}] must be a string")
        if not isinstance(behavior, Mapping):
            raise StateCollectionError(f"behavior_labels[{index}] must be an object")
        if not isinstance(metadata, Mapping):
            raise StateCollectionError("row_metadata entries must be objects")
        if state in BASE_STATES:
            expected_labels = {
                "X_B": (0, 0),
                "X_H": (1, 1),
                "X_J": (1, 0),
            }[state]
            if (harmfulness, refusal) != expected_labels:
                raise StateCollectionError(
                    f"state {state} at row {index} has inconsistent H/R labels"
                )
            if step != -1:
                raise StateCollectionError("base states must use step=-1")
        elif step < 0:
            raise StateCollectionError("trajectory states need non-negative steps")
        identity = (pair_id, state, -1 if state in BASE_STATES else step)
        if identity in seen:
            raise StateCollectionError(f"duplicate collected row identity: {identity}")
        seen.add(identity)
        if state in BASE_STATES:
            base_by_pair.setdefault(pair_id, set()).add(state)
        if metadata.get("pair_id") != pair_id or metadata.get("state") != state:
            raise StateCollectionError(
                f"row_metadata[{index}] does not own its pair/state row"
            )
        if metadata.get("row_index") != index:
            raise StateCollectionError(f"row_metadata[{index}] has wrong row_index")
        if metadata.get("response") != response:
            raise StateCollectionError(f"row_metadata[{index}] response disagrees")
        try:
            metadata_labels = (
                int(metadata.get("harmfulness_label")),
                int(metadata.get("refusal_label")),
            )
            behavior_values = (
                int(behavior.get("harmfulness")),
                int(behavior.get("refusal")),
            )
        except (TypeError, ValueError) as exc:
            raise StateCollectionError(
                f"row {index} metadata contains invalid behavior labels"
            ) from exc
        if metadata_labels != (harmfulness, refusal):
            raise StateCollectionError(f"row_metadata[{index}] labels disagree")
        if behavior_values != (harmfulness, refusal):
            raise StateCollectionError(f"behavior_labels[{index}] labels disagree")
    payload_metadata = payload.get("metadata", {})
    if not isinstance(payload_metadata, Mapping):
        raise StateCollectionError("metadata must be a mapping")
    model_provenance = payload_metadata.get("model_provenance")
    if model_provenance is not None:
        normalized_provenance = _normalize_model_provenance(model_provenance)
        for name, expected in normalized_provenance.items():
            if payload_metadata.get(name) != expected:
                raise StateCollectionError(
                    f"metadata.{name} must agree with metadata.model_provenance"
                )
    declared = payload_metadata.get("require_triplets", False)
    requested = set(payload_metadata.get("requested_states", ()))
    if declared:
        for pair_id in dict.fromkeys(payload["pair_ids"]):
            present = base_by_pair.get(pair_id, set())
            if present != requested:
                raise StateCollectionError(
                    f"pair_id {pair_id!r} does not contain every requested base state"
                )
    return sample_count


def collect_safety_states(
    manifest: Any,
    *,
    model: Any,
    output_path: str | Path | None = None,
    audio_loader: Optional[Callable[..., Any]] = None,
    judge: Any = None,
    behavior_labels: str | Path | None = None,
    manifest_base: str | Path | None = None,
    states: Iterable[str] = BASE_STATES,
    require_triplets: bool = True,
    layers: Optional[Sequence[int]] = None,
    pooling: str = "mean",
    token_span: str = "audio",
    sequence_has_embedding: bool = True,
    include_trajectories: bool = True,
    trajectory_paths: Optional[Mapping[str, str | Path]] = None,
    trajectory_column: str = "trajectory_path",
    generate_missing_responses: bool = True,
    max_tokens: int = 100,
    require_existing_audio: bool = True,
    shard_size: Optional[int] = None,
    model_provenance: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Collect pooled states into a leakage-safe dual-probe training payload."""

    import torch

    if model is None or not hasattr(model, "forward_attack"):
        raise TypeError("model must expose forward_attack(..., output_hidden_states=True)")
    if not hasattr(model, "sample_rate"):
        raise TypeError("model must expose sample_rate")
    selected_states = _normalize_states(states)
    if pooling not in {"mean", "max", "first", "last"}:
        raise ValueError("pooling must be mean, max, first, or last")
    if token_span not in {"audio", "target", "all"}:
        raise ValueError("token_span must be audio, target, or all")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if layers is not None:
        normalized_layers: Optional[tuple[int, ...]] = tuple(layers)
        if not normalized_layers or any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in normalized_layers
        ):
            raise ValueError("layers must contain one or more integers")
    else:
        normalized_layers = None
    if shard_size is not None and (
        isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0
    ):
        raise ValueError("shard_size must be a positive integer")
    normalized_model_provenance = (
        None
        if model_provenance is None
        else _normalize_model_provenance(model_provenance)
    )

    canonical, base, source_path = _read_manifest(
        manifest, manifest_base=manifest_base
    )
    behavior_labels_path: Optional[Path] = None
    explicit_behavior_labels: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if behavior_labels is not None:
        behavior_labels_path = _resolve_local_path(
            base, behavior_labels, name="behavior_labels"
        )
        explicit_behavior_labels = _load_behavior_labels(behavior_labels_path)
    examples, selected_pair_ids = _state_rows(
        canonical, states=selected_states, require_triplets=require_triplets
    )
    if trajectory_paths is not None:
        unknown = set(trajectory_paths) - set(canonical["pair_id"].astype(str))
        if unknown:
            raise StateCollectionError(
                "trajectory_paths contains pair_id values absent from the manifest: "
                + ", ".join(sorted(unknown)[:5])
            )
    manifest_rows = {
        str(row["pair_id"]): row for row in canonical.to_dict(orient="records")
    }
    load_audio = audio_loader or _default_audio_loader

    layer_rows: "OrderedDict[Any, list[Any]]" = OrderedDict()
    row_metadata: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    collected_states: list[str] = []
    steps: list[int] = []
    responses: list[str] = []
    harmfulness_labels: list[int] = []
    refusal_labels: list[int] = []
    collected_behavior_labels: list[dict[str, Any]] = []
    trajectory_label_counts = {"ok": 0, "unknown_skipped": 0}

    for example in examples:
        pair_id = str(example["pair_id"])
        audio_path = _resolve_local_path(
            base,
            example.get("audio_path"),
            name=f"{example['state']} audio_path for pair_id {pair_id!r}",
        )
        if require_existing_audio and not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        example["audio_path"] = str(audio_path)
        waveform = load_audio(audio_path, target_sr=int(model.sample_rate))
        _append_collected_row(
            example=example,
            waveform=waveform,
            model=model,
            judge=judge,
            generate_missing_responses=generate_missing_responses,
            max_tokens=max_tokens,
            layers=normalized_layers,
            pooling=pooling,
            token_span=token_span,
            sequence_has_embedding=sequence_has_embedding,
            layer_rows=layer_rows,
            row_metadata=row_metadata,
            pair_ids=pair_ids,
            states=collected_states,
            steps=steps,
            responses=responses,
            harmfulness_labels=harmfulness_labels,
            refusal_labels=refusal_labels,
            behavior_labels=collected_behavior_labels,
        )

    if include_trajectories:
        for pair_id in canonical["pair_id"].astype(str):
            if pair_id not in selected_pair_ids:
                continue
            pair_row = manifest_rows[pair_id]
            source = _trajectory_source_for_row(
                pair_row,
                pair_id=pair_id,
                base=base,
                trajectory_paths=trajectory_paths,
                trajectory_column=trajectory_column,
            )
            if source is None:
                continue
            for checkpoint in load_trajectory_checkpoints(source, pair_id=pair_id):
                checkpoint_case = _metadata_lookup(
                    checkpoint.metadata, ("case_id",)
                )
                manifest_case = pair_row.get("case_id")
                if not _is_blank(manifest_case):
                    if _is_blank(checkpoint_case) or str(checkpoint_case).strip() != str(
                        manifest_case
                    ).strip():
                        raise StateCollectionError(
                            f"trajectory case_id {checkpoint_case!r} does not match "
                            f"manifest case_id {str(manifest_case).strip()!r} for "
                            f"pair_id {pair_id!r} step {checkpoint.step}"
                        )
                explicit_label = None
                if explicit_behavior_labels:
                    case_value = checkpoint_case
                    if _is_blank(case_value):
                        case_value = pair_row.get("case_id")
                    if _is_blank(case_value):
                        raise StateCollectionError(
                            f"pair_id {pair_id!r} step {checkpoint.step} has no case_id "
                            "for behavior-label joining"
                        )
                    identity = (str(case_value).strip(), pair_id, checkpoint.step)
                    explicit_label = explicit_behavior_labels.get(identity)
                    if explicit_label is None:
                        raise StateCollectionError(
                            f"No behavior label for case_id={identity[0]!r}, "
                            f"pair_id={pair_id!r}, step={checkpoint.step}"
                        )
                    raw_checkpoint = explicit_label["checkpoint_path"]
                    label_checkpoint = Path(str(raw_checkpoint)).expanduser()
                    if not label_checkpoint.is_absolute():
                        assert behavior_labels_path is not None
                        label_checkpoint = behavior_labels_path.parent / label_checkpoint
                    if label_checkpoint.resolve() != checkpoint.path.resolve():
                        raise StateCollectionError(
                            f"Behavior label checkpoint does not match trajectory "
                            f"for {identity}"
                        )

                    label_fingerprint = str(
                        explicit_label["experiment_fingerprint"]
                    )
                    checkpoint_fingerprint = _required_sha256(
                        _metadata_lookup(
                            checkpoint.metadata, ("experiment_fingerprint",)
                        ),
                        name=(
                            "trajectory checkpoint metadata experiment_fingerprint "
                            f"for {identity}"
                        ),
                    )
                    if label_fingerprint != checkpoint_fingerprint:
                        raise StateCollectionError(
                            "Behavior label experiment_fingerprint does not match "
                            f"trajectory checkpoint metadata for {identity}"
                        )

                    expected_checkpoint_digest = str(
                        explicit_label["checkpoint_sha256"]
                    )
                    actual_checkpoint_digest = _file_sha256(checkpoint.path)
                    if expected_checkpoint_digest != actual_checkpoint_digest:
                        raise StateCollectionError(
                            "Behavior label checkpoint_sha256 does not match the "
                            f"checkpoint contents for {identity}"
                        )

                    if (
                        str(explicit_label.get("label_status", ""))
                        .strip()
                        .casefold()
                        == "unknown"
                    ):
                        trajectory_label_counts["unknown_skipped"] += 1
                        continue
                    trajectory_label_counts["ok"] += 1
                example, waveform = _trajectory_example(
                    checkpoint,
                    pair_row=pair_row,
                    behavior_label=explicit_label,
                )
                _append_collected_row(
                    example=example,
                    waveform=waveform,
                    model=model,
                    judge=None if explicit_label is not None else judge,
                    generate_missing_responses=generate_missing_responses,
                    max_tokens=max_tokens,
                    layers=normalized_layers,
                    pooling=pooling,
                    token_span=token_span,
                    sequence_has_embedding=sequence_has_embedding,
                    layer_rows=layer_rows,
                    row_metadata=row_metadata,
                    pair_ids=pair_ids,
                    states=collected_states,
                    steps=steps,
                    responses=responses,
                    harmfulness_labels=harmfulness_labels,
                    refusal_labels=refusal_labels,
                    behavior_labels=collected_behavior_labels,
                )

    hidden_states = OrderedDict(
        (layer, torch.stack(rows, dim=0).contiguous())
        for layer, rows in layer_rows.items()
    )
    payload: Mapping[str, Any] = {
        "format": COLLECTION_FORMAT,
        "version": COLLECTION_VERSION,
        "hidden_states": hidden_states,
        "harmfulness_labels": torch.tensor(
            harmfulness_labels, dtype=torch.float32
        ),
        "refusal_labels": torch.tensor(refusal_labels, dtype=torch.float32),
        "pair_ids": pair_ids,
        "states": collected_states,
        "steps": torch.tensor(steps, dtype=torch.int64),
        "layers": list(hidden_states),
        "responses": responses,
        "behavior_labels": collected_behavior_labels,
        "row_metadata": row_metadata,
        "metadata": {
            "manifest": None if source_path is None else str(source_path),
            "num_pairs": len(set(pair_ids)),
            "num_rows": len(pair_ids),
            "requested_states": list(selected_states),
            "require_triplets": bool(require_triplets),
            "include_trajectories": bool(include_trajectories),
            "pooling": pooling,
            "token_span": token_span,
            "sequence_has_embedding": bool(sequence_has_embedding),
            "layers": list(hidden_states),
            "measurement_splits": sorted(
                {
                    str(row["measurement_split"])
                    for row in row_metadata
                    if not _is_blank(row.get("measurement_split"))
                }
            ),
            "stage1_roles": sorted(
                {
                    str(row["stage1_role"])
                    for row in row_metadata
                    if not _is_blank(row.get("stage1_role"))
                }
            ),
            "behavior_labels": (
                None if behavior_labels_path is None else str(behavior_labels_path)
            ),
            "trajectory_label_counts": trajectory_label_counts,
            **(
                {}
                if normalized_model_provenance is None
                else {
                    "model_provenance": dict(normalized_model_provenance),
                    **dict(normalized_model_provenance),
                }
            ),
        },
    }
    validate_collection_payload(payload)
    if output_path is not None:
        save_collection_payload(payload, output_path, shard_size=shard_size)
    return payload


def _select_payload_rows(
    payload: Mapping[str, Any], indices: Sequence[int], *, shard_index: int
) -> Mapping[str, Any]:
    import torch

    selected = torch.tensor(tuple(indices), dtype=torch.long)
    metadata = dict(payload.get("metadata", {}))
    shard_pair_ids = list(dict.fromkeys(payload["pair_ids"][index] for index in indices))
    metadata.update(
        {
            "parent_require_triplets": bool(metadata.get("require_triplets", False)),
            "require_triplets": bool(metadata.get("require_triplets", False)),
            "num_rows": len(indices),
            "num_pairs": len(shard_pair_ids),
            "shard_index": shard_index,
            "pair_ids": shard_pair_ids,
        }
    )
    shard_metadata = []
    for row_index, source_index in enumerate(indices):
        row = dict(payload["row_metadata"][source_index])
        row["source_row_index"] = source_index
        row["row_index"] = row_index
        shard_metadata.append(row)
    return {
        **{
            key: value
            for key, value in payload.items()
            if key
            not in {
                "hidden_states",
                "harmfulness_labels",
                "refusal_labels",
                "pair_ids",
                "states",
                "steps",
                "responses",
                "behavior_labels",
                "row_metadata",
                "metadata",
            }
        },
        "hidden_states": OrderedDict(
            (layer, tensor.index_select(0, selected))
            for layer, tensor in payload["hidden_states"].items()
        ),
        "harmfulness_labels": payload["harmfulness_labels"].index_select(0, selected),
        "refusal_labels": payload["refusal_labels"].index_select(0, selected),
        "pair_ids": [payload["pair_ids"][index] for index in indices],
        "states": [payload["states"][index] for index in indices],
        "steps": payload["steps"].index_select(0, selected),
        "responses": [payload["responses"][index] for index in indices],
        "behavior_labels": [payload["behavior_labels"][index] for index in indices],
        "row_metadata": shard_metadata,
        "metadata": metadata,
    }


def _pair_shards(pair_ids: Sequence[str], shard_size: int) -> tuple[tuple[int, ...], ...]:
    """Group whole pairs into shards whose target size is ``shard_size`` rows."""

    grouped: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, pair_id in enumerate(pair_ids):
        grouped.setdefault(pair_id, []).append(index)
    shards: list[tuple[int, ...]] = []
    pending: list[int] = []
    for indices in grouped.values():
        if pending and len(pending) + len(indices) > shard_size:
            shards.append(tuple(pending))
            pending = []
        pending.extend(indices)
        if len(pending) >= shard_size:
            shards.append(tuple(pending))
            pending = []
    if pending:
        shards.append(tuple(pending))
    return tuple(shards)

def save_collection_payload(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    shard_size: Optional[int] = None,
) -> tuple[Path, ...]:
    """Atomically save a complete training payload and optional pair-safe shards.

    ``shard_size`` is a target row count: all rows sharing a ``pair_id`` stay in
    the same shard, even when one pair alone exceeds the requested size.
    """

    sample_count = validate_collection_payload(payload)
    if shard_size is not None and (
        isinstance(shard_size, bool)
        or not isinstance(shard_size, int)
        or shard_size <= 0
    ):
        raise ValueError("shard_size must be a positive integer")
    destination = atomic_torch_save(payload, output_path)
    written = [destination]
    if shard_size is None:
        return tuple(written)
    shard_dir = destination.parent / f"{destination.stem}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    for shard_index, indices in enumerate(
        _pair_shards(payload["pair_ids"], shard_size)
    ):
        shard = _select_payload_rows(payload, indices, shard_index=shard_index)
        validate_collection_payload(shard, require_both_classes=False)
        path = atomic_torch_save(
            shard, shard_dir / f"part-{shard_index:05d}.pt"
        )
        written.append(path)
        shard_records.append(
            {
                "index": shard_index,
                "path": str(path.relative_to(destination.parent)),
                "row_indices": list(indices),
                "pair_ids": list(dict.fromkeys(shard["pair_ids"])),
                "num_rows": len(indices),
            }
        )
    index_path = destination.with_suffix(destination.suffix + ".shards.json")
    _atomic_json(
        index_path,
        {
            "format": "safety-state-probe-shards",
            "version": 1,
            "complete_payload": destination.name,
            "num_rows": sample_count,
            "shard_size": shard_size,
            "pair_safe": True,
            "shards": shard_records,
        },
    )
    written.append(index_path)
    return tuple(written)


def _parse_layers(value: str) -> Optional[tuple[int, ...]]:
    text = value.strip().casefold()
    if text == "all":
        return None
    try:
        layers = tuple(int(piece.strip()) for piece in text.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be 'all' or comma-separated integers"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("layers must not be empty")
    return layers


def _load_object(specification: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use module:attribute syntax")
    return getattr(importlib.import_module(module_name), attribute)


def _default_model(
    *, model_name: str, model_id: Optional[str], device: str, dtype: str
) -> Any:
    import torch

    from models import create_model

    dtype_value = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    return create_model(
        model_name,
        model_id=model_id,
        device=device,
        dtype=dtype_value,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=("qwen-3b", "qwen-7b"), default="qwen-3b")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--layers", type=_parse_layers, default=None)
    parser.add_argument(
        "--pooling", choices=("mean", "max", "first", "last"), default="mean"
    )
    parser.add_argument("--token-span", choices=("audio", "target", "all"), default="audio")
    parser.add_argument("--states", default="X_B,X_H,X_J")
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--trajectory-column", default="trajectory_path")
    parser.add_argument(
        "--behavior-labels",
        type=Path,
        default=None,
        help="Explicit hash-bound trajectory labels from evaluate_stage1_behavior",
    )
    parser.add_argument("--no-generate-missing-responses", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--allow-missing-audio-files", action="store_true")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument(
        "--model-factory",
        help="Optional local module:callable receiving model_name/model_id/device/dtype",
    )
    parser.add_argument(
        "--audio-loader",
        help="Optional local module:callable with (path, target_sr=...) signature",
    )
    parser.add_argument(
        "--judge-factory",
        help="Optional local module:zero-argument-callable returning a refusal judge",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    factory = _default_model if args.model_factory is None else _load_object(args.model_factory)
    model = factory(
        model_name=args.model,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    audio_loader = (
        None if args.audio_loader is None else _load_object(args.audio_loader)
    )
    judge = None
    if args.judge_factory is not None:
        judge_factory = _load_object(args.judge_factory)
        judge = judge_factory()
    states = tuple(
        state.strip() for state in args.states.split(",") if state.strip()
    )
    payload = collect_safety_states(
        args.manifest,
        model=model,
        output_path=args.output,
        audio_loader=audio_loader,
        judge=judge,
        behavior_labels=args.behavior_labels,
        states=states,
        require_triplets=not args.allow_unpaired,
        layers=args.layers,
        pooling=args.pooling,
        token_span=args.token_span,
        include_trajectories=not args.no_trajectories,
        trajectory_column=args.trajectory_column,
        generate_missing_responses=not args.no_generate_missing_responses,
        max_tokens=args.max_tokens,
        require_existing_audio=not args.allow_missing_audio_files,
        shard_size=args.shard_size,
        model_provenance=build_model_provenance(
            args.model, args.model_id, args.dtype
        ),
    )
    print(
        f"Saved {len(payload['pair_ids'])} state rows across "
        f"{payload['metadata']['num_pairs']} pair(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_STATES",
    "COLLECTION_FORMAT",
    "COLLECTION_VERSION",
    "StateCollectionError",
    "TrajectoryCheckpoint",
    "atomic_torch_save",
    "build_model_provenance",
    "build_parser",
    "collect_safety_states",
    "load_trajectory_checkpoints",
    "save_collection_payload",
    "validate_collection_payload",
]
