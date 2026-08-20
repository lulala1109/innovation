#!/usr/bin/env python3
"""Replay held-out Stage-1 PGD checkpoints into pooled hidden states.

This entry point is deliberately separate from ``collect_safety_states``:
probe-training rows may omit unusable behavior labels, whereas an RQ1
trajectory must preserve the complete ``t=0..T`` grid.  Behavior decisions
are therefore attached as nullable metadata and never control whether a
checkpoint is forwarded through the model.

Each case is committed atomically to its own ``.pt`` file.  ``index.json`` is
updated after every completed case and binds resume to the manifest, behavior
sidecars, model identity, and replay configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


OUTPUT_FORMAT = "stage1-trajectory-hidden-states"
INDEX_FORMAT = "stage1-trajectory-hidden-state-index"
OUTPUT_VERSION = 1
INDEX_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POOLING = frozenset(("mean", "max", "first", "last"))
_TOKEN_SPANS = frozenset(("audio", "target", "all"))


class Stage1TrajectoryReplayError(ValueError):
    """Raised when replay inputs or outputs violate Stage-1 invariants."""


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    pair_id: str
    row: Mapping[str, Any]
    manifest_path: Path
    manifest_sha256: str
    trajectory_path: Path
    trajectory_sha256: str
    run_path: Path
    run_sha256: str
    behavior_path: Path
    behavior_sha256: str
    experiment_fingerprint: str
    harmful_text: str
    target_text: str
    total_steps: int
    checkpoints: tuple[Any, ...]
    checkpoint_metadata: Mapping[int, Mapping[str, Any]]
    behavior_by_step: Mapping[int, Mapping[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage1TrajectoryReplayError(
            "Replay provenance must be finite JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: Any, *, name: str) -> str:
    if value is None or not str(value).strip():
        raise Stage1TrajectoryReplayError(f"{name} must be non-blank")
    return str(value).strip()


def _required_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage1TrajectoryReplayError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage1TrajectoryReplayError(
            f"{name} must be a non-negative integer"
        )
    return value


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise Stage1TrajectoryReplayError(f"Invalid JSON in {name}: {path}") from exc
    if not isinstance(value, dict):
        raise Stage1TrajectoryReplayError(f"{name} must be one JSON object: {path}")
    return value


def _json_safe(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise Stage1TrajectoryReplayError(
            f"{name} must contain finite JSON-compatible values"
        ) from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _resolve_path(
    value: Any,
    *,
    manifest_base: Path,
    project_root: Path,
    name: str,
) -> Path:
    raw = Path(_required_text(value, name=name)).expanduser()
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (project_root / raw, manifest_base / raw)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve {name} {value!r}; tried "
        + ", ".join(str(candidate.resolve()) for candidate in candidates)
    )


def _resolve_label_checkpoint(
    value: Any,
    *,
    sidecar: Path,
    manifest_base: Path,
    project_root: Path,
) -> Path:
    raw = Path(_required_text(value, name="behavior checkpoint_path")).expanduser()
    candidates = (
        (raw,)
        if raw.is_absolute()
        else (sidecar.parent / raw, project_root / raw, manifest_base / raw)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return candidates[0].resolve()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    from data.datasets import read_table

    frame = read_table(path)
    if frame.empty:
        raise Stage1TrajectoryReplayError("Stage-1 trajectory manifest is empty")
    required = {
        "case_id",
        "pair_id",
        "harmful_text",
        "measurement_split",
        "stage1_role",
        "trajectory_path",
        "behavior_labels_path",
        "experiment_fingerprint",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage1TrajectoryReplayError(
            "Attached trajectory manifest is missing: " + ", ".join(missing)
        )
    rows = frame.to_dict(orient="records")
    identities: set[tuple[str, str]] = set()
    for position, row in enumerate(rows):
        split = _required_text(
            row.get("measurement_split"), name=f"row {position} measurement_split"
        )
        role = _required_text(
            row.get("stage1_role"), name=f"row {position} stage1_role"
        )
        if split != "measurement_val" or role != "trajectory_candidate":
            raise Stage1TrajectoryReplayError(
                "Replay accepts only measurement_val/trajectory_candidate rows; "
                f"row {position} has {split!r}/{role!r}"
            )
        identity = (
            _required_text(row.get("case_id"), name=f"row {position} case_id"),
            _required_text(row.get("pair_id"), name=f"row {position} pair_id"),
        )
        if identity in identities:
            raise Stage1TrajectoryReplayError(
                f"Duplicate attached trajectory identity: {identity}"
            )
        identities.add(identity)
    return rows


def _find_run_path(trajectory_path: Path) -> Path:
    candidates = [
        trajectory_path.parent / "run.json",
        trajectory_path.parent.parent / "run.json",
    ]
    for parent in trajectory_path.parents:
        candidate = parent / "run.json"
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= 5:
            break
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find run.json for trajectory index {trajectory_path}"
    )


def _artifact_path(case_dir: Path, value: Any) -> Path:
    raw = Path(_required_text(value, name="run artifacts.trajectory")).expanduser()
    return (raw if raw.is_absolute() else case_dir / raw).resolve()


def _load_label_index(path: Path) -> Mapping[tuple[str, str, int], Mapping[str, Any]]:
    from experiments.collect_safety_states import _load_behavior_labels

    try:
        return _load_behavior_labels(path)
    except (ValueError, FileNotFoundError) as exc:
        raise Stage1TrajectoryReplayError(
            f"Invalid explicit behavior-label sidecar {path}: {exc}"
        ) from exc


def _metadata_by_step(index: Mapping[str, Any], path: Path) -> dict[int, Mapping[str, Any]]:
    if index.get("format") != "safety-state-trajectory" or index.get("version") != 1:
        raise Stage1TrajectoryReplayError(
            f"Unsupported trajectory format/version: {path}"
        )
    records = index.get("checkpoints")
    if not isinstance(records, list) or not records:
        raise Stage1TrajectoryReplayError(f"Trajectory index has no checkpoints: {path}")
    result: dict[int, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Stage1TrajectoryReplayError(
                f"Trajectory checkpoint {position} is not an object: {path}"
            )
        step = _non_negative_int(
            record.get("step"), name=f"trajectory checkpoint {position} step"
        )
        if step in result:
            raise Stage1TrajectoryReplayError(f"Duplicate trajectory step {step}: {path}")
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise Stage1TrajectoryReplayError(
                f"Trajectory checkpoint {step} metadata must be an object"
            )
        result[step] = dict(metadata)
    return result


def _case_spec(
    row: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    behavior_override: Optional[Path],
    project_root: Path,
    label_cache: dict[Path, Mapping[tuple[str, str, int], Mapping[str, Any]]],
) -> _CaseSpec:
    from experiments.evaluate_stage1_behavior import enumerate_trajectory_checkpoints

    manifest_base = manifest_path.parent
    case_id = _required_text(row.get("case_id"), name="manifest case_id")
    pair_id = _required_text(row.get("pair_id"), name="manifest pair_id")
    harmful_text = _required_text(row.get("harmful_text"), name="manifest harmful_text")
    trajectory_path = _resolve_path(
        row.get("trajectory_path"),
        manifest_base=manifest_base,
        project_root=project_root,
        name=f"trajectory_path for {case_id}/{pair_id}",
    )
    behavior_path = (
        behavior_override
        if behavior_override is not None
        else _resolve_path(
            row.get("behavior_labels_path"),
            manifest_base=manifest_base,
            project_root=project_root,
            name=f"behavior_labels_path for {case_id}/{pair_id}",
        )
    ).resolve()
    if not behavior_path.is_file():
        raise FileNotFoundError(behavior_path)
    run_path = _find_run_path(trajectory_path)
    run = _json_object(run_path, name="attack run")
    if _required_text(run.get("case_id"), name="run case_id") != case_id:
        raise Stage1TrajectoryReplayError(f"run case_id does not match manifest: {case_id}")
    if _required_text(run.get("pair_id"), name="run pair_id") != pair_id:
        raise Stage1TrajectoryReplayError(f"run pair_id does not match manifest: {pair_id}")
    if run.get("method") != "standard":
        raise Stage1TrajectoryReplayError(
            f"Stage-1 replay requires method='standard', got {run.get('method')!r}"
        )
    if _required_text(run.get("harmful_text"), name="run harmful_text") != harmful_text:
        raise Stage1TrajectoryReplayError(
            f"run harmful_text does not match manifest for {pair_id}"
        )
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Stage1TrajectoryReplayError("run artifacts must be an object")
    if _artifact_path(run_path.parent, artifacts.get("trajectory")) != trajectory_path:
        raise Stage1TrajectoryReplayError(
            f"manifest trajectory_path is not the trajectory declared by {run_path}"
        )
    budget = run.get("budget")
    if not isinstance(budget, Mapping):
        raise Stage1TrajectoryReplayError("run budget must be an object")
    total_steps = _non_negative_int(budget.get("steps"), name="run budget.steps")
    if budget.get("init_mode") != "zero":
        raise Stage1TrajectoryReplayError(
            "Stage-1 replay requires a zero-initialized Standard PGD run"
        )
    experiment_fingerprint = _required_sha256(
        budget.get("experiment_fingerprint"), name="run experiment_fingerprint"
    )
    experiment_config = budget.get("experiment_config")
    if not isinstance(experiment_config, Mapping):
        raise Stage1TrajectoryReplayError("run experiment_config must be an object")
    if _fingerprint(experiment_config) != experiment_fingerprint:
        raise Stage1TrajectoryReplayError("run experiment_config fingerprint mismatch")
    if experiment_config.get("method") != "standard":
        raise Stage1TrajectoryReplayError("experiment_config method must be standard")
    manifest_fingerprint = _required_sha256(
        row.get("experiment_fingerprint"), name="manifest experiment_fingerprint"
    )
    if manifest_fingerprint != experiment_fingerprint:
        raise Stage1TrajectoryReplayError(
            f"manifest experiment_fingerprint mismatch for {pair_id}"
        )

    index = _json_object(trajectory_path, name="trajectory index")
    checkpoint_metadata = _metadata_by_step(index, trajectory_path)
    try:
        checkpoints = enumerate_trajectory_checkpoints(
            run_path.parent, require_all_steps=True
        )
    except (ValueError, FileNotFoundError) as exc:
        raise Stage1TrajectoryReplayError(
            f"Invalid complete trajectory for {case_id}/{pair_id}: {exc}"
        ) from exc
    checkpoints = tuple(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.case_id == case_id and checkpoint.pair_id == pair_id
    )
    if tuple(checkpoint.step for checkpoint in checkpoints) != tuple(
        range(total_steps + 1)
    ):
        raise Stage1TrajectoryReplayError(
            f"Trajectory for {case_id}/{pair_id} is not exactly 0..{total_steps}"
        )
    if set(checkpoint_metadata) != set(range(total_steps + 1)):
        raise Stage1TrajectoryReplayError(
            f"Trajectory metadata for {case_id}/{pair_id} is incomplete"
        )

    label_index = label_cache.get(behavior_path)
    if label_index is None:
        label_index = _load_label_index(behavior_path)
        label_cache[behavior_path] = label_index
    behavior_by_step: dict[int, Mapping[str, Any]] = {}
    unexpected = []
    for identity, label in label_index.items():
        label_case, label_pair, label_step = identity
        if label_case == case_id and label_pair == pair_id:
            if label_step > total_steps:
                unexpected.append(label_step)
            else:
                behavior_by_step[label_step] = label
    if unexpected:
        raise Stage1TrajectoryReplayError(
            f"Behavior sidecar has steps outside 0..{total_steps} for "
            f"{case_id}/{pair_id}: {sorted(unexpected)}"
        )
    for checkpoint in checkpoints:
        label = behavior_by_step.get(checkpoint.step)
        if label is None:
            continue
        if label["experiment_fingerprint"] != experiment_fingerprint:
            raise Stage1TrajectoryReplayError(
                f"Behavior fingerprint mismatch for {case_id}/{pair_id}/"
                f"{checkpoint.step}"
            )
        if label["checkpoint_sha256"] != checkpoint.checkpoint_sha256:
            raise Stage1TrajectoryReplayError(
                f"Behavior checkpoint SHA mismatch for {case_id}/{pair_id}/"
                f"{checkpoint.step}"
            )
        label_checkpoint = _resolve_label_checkpoint(
            label.get("checkpoint_path"),
            sidecar=behavior_path,
            manifest_base=manifest_base,
            project_root=project_root,
        )
        if label_checkpoint != checkpoint.checkpoint_path.resolve():
            raise Stage1TrajectoryReplayError(
                f"Behavior checkpoint path mismatch for {case_id}/{pair_id}/"
                f"{checkpoint.step}"
            )

    target_text = checkpoints[0].target_text
    if not target_text:
        target_text = harmful_text
    return _CaseSpec(
        case_id=case_id,
        pair_id=pair_id,
        row=dict(row),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        trajectory_path=trajectory_path,
        trajectory_sha256=_file_sha256(trajectory_path),
        run_path=run_path,
        run_sha256=_file_sha256(run_path),
        behavior_path=behavior_path,
        behavior_sha256=_file_sha256(behavior_path),
        experiment_fingerprint=experiment_fingerprint,
        harmful_text=harmful_text,
        target_text=target_text,
        total_steps=total_steps,
        checkpoints=checkpoints,
        checkpoint_metadata=checkpoint_metadata,
        behavior_by_step=behavior_by_step,
    )


def _model_provenance(
    model_name: str,
    model_id: Optional[str],
    dtype: str,
) -> Mapping[str, Any]:
    """Bind replay to canonical model arguments without hashing 3B weights."""

    from experiments.collect_safety_states import build_model_provenance

    try:
        return build_model_provenance(model_name, model_id, dtype)
    except ValueError as exc:
        raise Stage1TrajectoryReplayError(
            f"Invalid model provenance: {exc}"
        ) from exc


def _normalized_behavior(
    spec: _CaseSpec,
    step: int,
) -> tuple[dict[str, Any], str, str]:
    label = spec.behavior_by_step.get(step)
    if label is None:
        return (
            {
                "case_id": spec.case_id,
                "pair_id": spec.pair_id,
                "step": step,
                "generation_status": "missing",
                "label_status": "missing",
                "response": "",
                "response_sha256": None,
                "refusal_label": None,
                "compliance_label": None,
                "jailbreak_success": None,
            },
            spec.harmful_text,
            "harmful_text",
        )
    status = str(label["label_status"]).strip().casefold()
    response = str(label.get("response", ""))
    ok = status == "ok"
    behavior = {
        "case_id": spec.case_id,
        "pair_id": spec.pair_id,
        "step": step,
        "generation_status": str(label["generation_status"]).strip().casefold(),
        "label_status": status,
        "response": response,
        "response_sha256": label["response_sha256"],
        "refusal_label": bool(label["refusal_label"]) if ok else None,
        "compliance_label": bool(label["compliance_label"]) if ok else None,
        "jailbreak_success": bool(label["jailbreak_success"]) if ok else None,
    }
    target = response if response.strip() else spec.harmful_text
    source = "response" if response.strip() else "harmful_text"
    return behavior, target, source


def _phase(progress: float) -> str:
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def _checkpoint_tensors(path: Path, *, step: int) -> tuple[Any, Any]:
    import torch

    from experiments.collect_safety_states import _safe_torch_load

    payload = _safe_torch_load(path)
    tensors = payload.get("tensors", payload)
    if not isinstance(tensors, Mapping):
        raise Stage1TrajectoryReplayError(f"Checkpoint tensors are malformed: {path}")
    waveform = tensors.get("adversarial_wav")
    delta = tensors.get("delta")
    if not isinstance(waveform, torch.Tensor) or not isinstance(delta, torch.Tensor):
        raise Stage1TrajectoryReplayError(
            f"Checkpoint {path} must contain adversarial_wav and delta tensors"
        )
    if waveform.shape != delta.shape or waveform.numel() == 0:
        raise Stage1TrajectoryReplayError(
            f"Checkpoint {path} waveform and delta shapes must match and be non-empty"
        )
    if not bool(torch.isfinite(waveform).all()) or not bool(torch.isfinite(delta).all()):
        raise Stage1TrajectoryReplayError(f"Checkpoint {path} has non-finite tensors")
    if step == 0 and bool(torch.count_nonzero(delta).item()):
        raise Stage1TrajectoryReplayError(
            "Stage-1 t=0 must contain an exactly zero delta"
        )
    return waveform, delta


def _attack_loss(metadata: Mapping[str, Any], *, step: int) -> float:
    value = metadata.get("loss")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1TrajectoryReplayError(
            f"Trajectory step {step} metadata.loss must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise Stage1TrajectoryReplayError(
            f"Trajectory step {step} metadata.loss must be finite"
        )
    return result


def _case_filename(case_id: str, pair_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._-") or "case"
    digest = hashlib.sha256(f"{case_id}\0{pair_id}".encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}.pt"


def validate_replay_payload(
    payload: Mapping[str, Any],
    *,
    expected_replay_fingerprint: Optional[str] = None,
) -> int:
    """Validate one per-case RQ1 hidden-state payload and return ``S``."""

    import torch

    if not isinstance(payload, Mapping):
        raise Stage1TrajectoryReplayError("Replay payload must be a mapping")
    if payload.get("format") != OUTPUT_FORMAT or payload.get("version") != OUTPUT_VERSION:
        raise Stage1TrajectoryReplayError("Unsupported replay payload format/version")
    case_id = _required_text(payload.get("case_id"), name="payload case_id")
    pair_id = _required_text(payload.get("pair_id"), name="payload pair_id")
    fingerprint = _required_sha256(
        payload.get("experiment_fingerprint"), name="experiment_fingerprint"
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Stage1TrajectoryReplayError("payload metadata must be an object")
    replay_fingerprint = _required_sha256(
        metadata.get("replay_fingerprint"), name="metadata replay_fingerprint"
    )
    if (
        expected_replay_fingerprint is not None
        and replay_fingerprint != expected_replay_fingerprint
    ):
        raise Stage1TrajectoryReplayError("Replay fingerprint changed")
    _required_sha256(metadata.get("model_fingerprint"), name="model_fingerprint")
    model_provenance = metadata.get("model_provenance")
    if not isinstance(model_provenance, Mapping):
        raise Stage1TrajectoryReplayError("metadata model_provenance must be an object")
    provenance_fingerprint = _required_sha256(
        model_provenance.get("model_fingerprint"),
        name="model_provenance model_fingerprint",
    )
    canonical_provenance = {
        "model_name": _required_text(
            model_provenance.get("model_name"), name="model_provenance model_name"
        ),
        "model_id": _required_text(
            model_provenance.get("model_id"), name="model_provenance model_id"
        ),
        "dtype": _required_text(
            model_provenance.get("dtype"), name="model_provenance dtype"
        ),
    }
    if _fingerprint(canonical_provenance) != provenance_fingerprint:
        raise Stage1TrajectoryReplayError("model_provenance fingerprint mismatch")
    if metadata.get("model_fingerprint") != provenance_fingerprint:
        raise Stage1TrajectoryReplayError("metadata model fingerprint mismatch")
    if metadata.get("experiment_fingerprint") != fingerprint:
        raise Stage1TrajectoryReplayError("metadata experiment fingerprint mismatch")
    if metadata.get("measurement_split") != "measurement_val":
        raise Stage1TrajectoryReplayError(
            "replay payload must come from measurement_val"
        )
    if metadata.get("stage1_role") != "trajectory_candidate":
        raise Stage1TrajectoryReplayError(
            "replay payload must have stage1_role='trajectory_candidate'"
        )
    total_steps = _non_negative_int(metadata.get("total_steps"), name="total_steps")
    steps = payload.get("steps")
    if not isinstance(steps, torch.Tensor) or steps.dtype != torch.int64:
        raise Stage1TrajectoryReplayError("steps must be an int64 tensor")
    expected_steps = torch.arange(total_steps + 1, dtype=torch.int64)
    if steps.shape != expected_steps.shape or not bool(torch.equal(steps.cpu(), expected_steps)):
        raise Stage1TrajectoryReplayError(f"steps must be exactly 0..{total_steps}")
    sample_count = total_steps + 1
    hidden = payload.get("hidden_states")
    layers = payload.get("layers")
    if not isinstance(hidden, Mapping) or not hidden:
        raise Stage1TrajectoryReplayError("hidden_states must be a non-empty mapping")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        raise Stage1TrajectoryReplayError("layers must be a sequence")
    if list(layers) != list(hidden):
        raise Stage1TrajectoryReplayError("layers must match hidden_states keys")
    for layer, tensor in hidden.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] != sample_count
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise Stage1TrajectoryReplayError(
                f"hidden_states[{layer!r}] must be finite floating [S,D]"
            )
    attack_loss = payload.get("attack_loss")
    if (
        not isinstance(attack_loss, torch.Tensor)
        or attack_loss.shape != (sample_count,)
        or not attack_loss.is_floating_point()
        or not bool(torch.isfinite(attack_loss).all())
    ):
        raise Stage1TrajectoryReplayError("attack_loss must be finite floating [S]")
    behaviors = payload.get("behavior")
    row_metadata = payload.get("row_metadata")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    for name, values in (
        ("behavior", behaviors),
        ("row_metadata", row_metadata),
        ("checkpoint_sha256", checkpoint_sha256),
    ):
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != sample_count
        ):
            raise Stage1TrajectoryReplayError(f"{name} must have length {sample_count}")
    for step in range(sample_count):
        digest = _required_sha256(
            checkpoint_sha256[step], name=f"checkpoint_sha256[{step}]"
        )
        behavior = behaviors[step]
        row = row_metadata[step]
        if not isinstance(behavior, Mapping) or not isinstance(row, Mapping):
            raise Stage1TrajectoryReplayError("behavior and row_metadata rows must be objects")
        for name, value in (("behavior", behavior), ("row_metadata", row)):
            if (
                value.get("case_id") != case_id
                or value.get("pair_id") != pair_id
                or value.get("step") != step
            ):
                raise Stage1TrajectoryReplayError(f"{name} identity mismatch at step {step}")
        status = behavior.get("label_status")
        if status not in {"ok", "unknown", "missing"}:
            raise Stage1TrajectoryReplayError(f"invalid label_status at step {step}")
        decisions = (
            behavior.get("refusal_label"),
            behavior.get("compliance_label"),
            behavior.get("jailbreak_success"),
        )
        if status == "ok":
            if any(not isinstance(value, bool) for value in decisions):
                raise Stage1TrajectoryReplayError(
                    f"ok behavior requires explicit booleans at step {step}"
                )
            if decisions[0] and decisions[1] or decisions[1] != decisions[2]:
                raise Stage1TrajectoryReplayError(
                    f"inconsistent behavior decisions at step {step}"
                )
        elif any(value is not None for value in decisions):
            raise Stage1TrajectoryReplayError(
                f"non-ok behavior decisions must be null at step {step}"
            )
        if row.get("checkpoint_sha256") != digest:
            raise Stage1TrajectoryReplayError(
                f"row checkpoint SHA mismatch at step {step}"
            )
        if row.get("experiment_fingerprint") != fingerprint:
            raise Stage1TrajectoryReplayError(
                f"row experiment fingerprint mismatch at step {step}"
            )
        if float(row.get("attack_loss")) != float(attack_loss[step].item()):
            raise Stage1TrajectoryReplayError(f"row attack_loss mismatch at step {step}")
    return sample_count


def _replay_case(
    spec: _CaseSpec,
    *,
    model: Any,
    model_name: str,
    model_id: str,
    model_fingerprint: str,
    model_provenance: Mapping[str, Any],
    replay_fingerprint: str,
    layers: Optional[tuple[int, ...]],
    pooling: str,
    token_span: str,
    sequence_has_embedding: bool,
) -> Mapping[str, Any]:
    import torch

    from experiments.collect_safety_states import _pool_forward_states

    layer_rows: "OrderedDict[Any, list[Any]]" = OrderedDict()
    attack_losses: list[float] = []
    behavior_rows: list[dict[str, Any]] = []
    row_metadata: list[dict[str, Any]] = []
    checkpoint_digests: list[str] = []
    for checkpoint in spec.checkpoints:
        step = checkpoint.step
        metadata = spec.checkpoint_metadata[step]
        waveform, delta = _checkpoint_tensors(checkpoint.checkpoint_path, step=step)
        loss = _attack_loss(metadata, step=step)
        behavior, target_text, target_source = _normalized_behavior(spec, step)
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
            raise Stage1TrajectoryReplayError(
                f"Selected layers changed during replay for {spec.case_id}"
            )
        for layer, tensor in pooled.items():
            if layer_rows[layer] and tensor.shape != layer_rows[layer][0].shape:
                raise Stage1TrajectoryReplayError(
                    f"Hidden size changed for layer {layer!r} in {spec.case_id}"
                )
            layer_rows[layer].append(tensor)
        progress = 0.0 if spec.total_steps == 0 else step / spec.total_steps
        digest = checkpoint.checkpoint_sha256
        response_sha = behavior.get("response_sha256")
        row = {
            "case_id": spec.case_id,
            "pair_id": spec.pair_id,
            "step": step,
            "total_steps": spec.total_steps,
            "progress": progress,
            "phase": _phase(progress),
            "attack_loss": loss,
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "checkpoint_sha256": digest,
            "experiment_fingerprint": spec.experiment_fingerprint,
            "generation_status": behavior["generation_status"],
            "label_status": behavior["label_status"],
            "response_sha256": response_sha,
            "forward_target_source": target_source,
            "delta_linf": float(delta.detach().abs().max().item()),
            "delta_rms": float(delta.detach().float().square().mean().sqrt().item()),
            "snr_db": metadata.get("snr_db"),
        }
        attack_losses.append(loss)
        behavior_rows.append(_json_safe(behavior, name="behavior row"))
        row_metadata.append(_json_safe(row, name="row metadata"))
        checkpoint_digests.append(digest)
        del pooled, waveform, delta

    hidden_states = OrderedDict(
        (layer, torch.stack(values, dim=0).to(dtype=torch.float32).contiguous())
        for layer, values in layer_rows.items()
    )
    payload: Mapping[str, Any] = {
        "format": OUTPUT_FORMAT,
        "version": OUTPUT_VERSION,
        "case_id": spec.case_id,
        "pair_id": spec.pair_id,
        "steps": torch.arange(spec.total_steps + 1, dtype=torch.int64),
        "hidden_states": hidden_states,
        "layers": list(hidden_states),
        "attack_loss": torch.tensor(attack_losses, dtype=torch.float32),
        "behavior": behavior_rows,
        "row_metadata": row_metadata,
        "checkpoint_sha256": checkpoint_digests,
        "experiment_fingerprint": spec.experiment_fingerprint,
        "metadata": {
            "total_steps": spec.total_steps,
            "model_name": model_name,
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "model_provenance": dict(model_provenance),
            "replay_fingerprint": replay_fingerprint,
            "pooling": pooling,
            "token_span": token_span,
            "sequence_has_embedding": bool(sequence_has_embedding),
            "requested_layers": None if layers is None else list(layers),
            "trajectory_path": str(spec.trajectory_path),
            "trajectory_sha256": spec.trajectory_sha256,
            "run_path": str(spec.run_path),
            "run_sha256": spec.run_sha256,
            "behavior_labels_path": str(spec.behavior_path),
            "behavior_labels_sha256": spec.behavior_sha256,
            "manifest_path": str(spec.manifest_path),
            "manifest_sha256": spec.manifest_sha256,
            "measurement_split": "measurement_val",
            "stage1_role": "trajectory_candidate",
            "harmful_text": spec.harmful_text,
            "target_text": spec.target_text,
            "experiment_fingerprint": spec.experiment_fingerprint,
        },
    }
    validate_replay_payload(
        payload, expected_replay_fingerprint=replay_fingerprint
    )
    return payload


def _safe_output_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old supported torch only
        raise RuntimeError(
            "Safe replay resume requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage1TrajectoryReplayError(f"Replay output is not a mapping: {path}")
    return payload


def _case_output_matches(
    payload: Mapping[str, Any],
    spec: _CaseSpec,
    replay_fingerprint: str,
) -> None:
    validate_replay_payload(
        payload, expected_replay_fingerprint=replay_fingerprint
    )
    if payload.get("case_id") != spec.case_id or payload.get("pair_id") != spec.pair_id:
        raise Stage1TrajectoryReplayError("Existing replay case identity changed")
    if payload.get("experiment_fingerprint") != spec.experiment_fingerprint:
        raise Stage1TrajectoryReplayError("Existing attack fingerprint changed")
    expected = [checkpoint.checkpoint_sha256 for checkpoint in spec.checkpoints]
    if list(payload.get("checkpoint_sha256", ())) != expected:
        raise Stage1TrajectoryReplayError("Existing checkpoint contents changed")
    metadata = payload["metadata"]
    if metadata.get("behavior_labels_sha256") != spec.behavior_sha256:
        raise Stage1TrajectoryReplayError("Existing behavior labels changed")


def replay_stage1_trajectories(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    model: Any,
    behavior_labels: str | Path | None = None,
    model_name: str = "qwen-3b",
    model_id: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    layers: Optional[Sequence[int]] = None,
    token_span: str = "audio",
    pooling: str = "mean",
    sequence_has_embedding: bool = True,
    resume: bool = True,
    project_root: str | Path = PROJECT_ROOT,
) -> Mapping[str, Any]:
    """Replay an attached held-out manifest and atomically save every case."""

    from experiments.collect_safety_states import atomic_torch_save

    if model is None or not callable(getattr(model, "forward_attack", None)):
        raise TypeError("model must expose forward_attack(..., output_hidden_states=True)")
    if pooling not in _POOLING:
        raise ValueError(f"pooling must be one of {sorted(_POOLING)}")
    if token_span not in _TOKEN_SPANS:
        raise ValueError(f"token_span must be one of {sorted(_TOKEN_SPANS)}")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    normalized_layers = None if layers is None else tuple(layers)
    if normalized_layers is not None and (
        not normalized_layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in normalized_layers)
        or len(set(normalized_layers)) != len(normalized_layers)
    ):
        raise ValueError("layers must contain unique integer layer indices")
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    root = Path(project_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "index.json"
    if not resume and (index_path.exists() or any(destination.glob("*.pt"))):
        raise FileExistsError(
            f"Replay output already exists and resume is disabled: {destination}"
        )
    rows = _read_manifest(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    behavior_override = None
    if behavior_labels is not None:
        behavior_override = _resolve_path(
            behavior_labels,
            manifest_base=manifest_path.parent,
            project_root=root,
            name="behavior_labels override",
        )
    label_cache: dict[Path, Mapping[tuple[str, str, int], Mapping[str, Any]]] = {}
    specs = tuple(
        _case_spec(
            row,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            behavior_override=behavior_override,
            project_root=root,
            label_cache=label_cache,
        )
        for row in rows
    )
    model_provenance = _model_provenance(model_name, model_id, dtype)
    resolved_model_id = str(model_provenance["model_id"])
    model_fingerprint = str(model_provenance["model_fingerprint"])
    behavior_sources = sorted(
        {
            (str(spec.behavior_path), spec.behavior_sha256)
            for spec in specs
        }
    )
    replay_config = {
        "manifest_sha256": manifest_sha256,
        "behavior_sidecars": [
            {"path": path, "sha256": digest} for path, digest in behavior_sources
        ],
        "model_provenance": model_provenance,
        "model_fingerprint": model_fingerprint,
        "device": device,
        "dtype": dtype,
        "layers": None if normalized_layers is None else list(normalized_layers),
        "token_span": token_span,
        "pooling": pooling,
        "sequence_has_embedding": bool(sequence_has_embedding),
    }
    replay_fingerprint = _fingerprint(replay_config)
    existing_cases: dict[tuple[str, str], Mapping[str, Any]] = {}
    if index_path.is_file():
        existing = _json_object(index_path, name="replay index")
        if existing.get("format") != INDEX_FORMAT or existing.get("version") != INDEX_VERSION:
            raise Stage1TrajectoryReplayError("Unsupported existing replay index")
        if existing.get("replay_fingerprint") != replay_fingerprint:
            raise Stage1TrajectoryReplayError(
                "Replay fingerprint changed; use a new output directory"
            )
        records = existing.get("cases")
        if not isinstance(records, list):
            raise Stage1TrajectoryReplayError("Existing replay index cases must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise Stage1TrajectoryReplayError("Existing case index is malformed")
            key = (str(record.get("case_id", "")), str(record.get("pair_id", "")))
            if key in existing_cases:
                raise Stage1TrajectoryReplayError(f"Duplicate case in replay index: {key}")
            existing_cases[key] = record

    if callable(getattr(model, "eval", None)):
        model.eval()
    completed_records: list[dict[str, Any]] = []
    skipped = 0
    replayed = 0
    for spec in specs:
        key = (spec.case_id, spec.pair_id)
        filename = _case_filename(*key)
        output_path = destination / filename
        record = existing_cases.get(key)
        if record is not None:
            if record.get("path") != filename:
                raise Stage1TrajectoryReplayError(
                    f"Existing replay path changed for {spec.case_id}/{spec.pair_id}"
                )
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            if record.get("sha256") != _file_sha256(output_path):
                raise Stage1TrajectoryReplayError(
                    f"Existing replay output SHA changed: {output_path}"
                )
            _case_output_matches(
                _safe_output_load(output_path), spec, replay_fingerprint
            )
            completed_records.append(dict(record))
            skipped += 1
            continue
        if output_path.exists():
            if not resume:
                raise FileExistsError(output_path)
            payload = _safe_output_load(output_path)
            _case_output_matches(payload, spec, replay_fingerprint)
        else:
            payload = _replay_case(
                spec,
                model=model,
                model_name=model_name,
                model_id=resolved_model_id,
                model_fingerprint=model_fingerprint,
                model_provenance=model_provenance,
                replay_fingerprint=replay_fingerprint,
                layers=normalized_layers,
                pooling=pooling,
                token_span=token_span,
                sequence_has_embedding=sequence_has_embedding,
            )
            atomic_torch_save(payload, output_path)
            replayed += 1
        case_record = {
            "case_id": spec.case_id,
            "pair_id": spec.pair_id,
            "path": filename,
            "sha256": _file_sha256(output_path),
            "num_steps": spec.total_steps + 1,
            "total_steps": spec.total_steps,
            "experiment_fingerprint": spec.experiment_fingerprint,
            "checkpoint_sha256": [
                checkpoint.checkpoint_sha256 for checkpoint in spec.checkpoints
            ],
        }
        completed_records.append(case_record)
        _atomic_json(
            index_path,
            {
                "format": INDEX_FORMAT,
                "version": INDEX_VERSION,
                "replay_fingerprint": replay_fingerprint,
                "model_fingerprint": model_fingerprint,
                "model_provenance": model_provenance,
                "config": replay_config,
                "source_manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest_sha256,
                },
                "behavior_sidecars": [
                    {"path": path, "sha256": digest}
                    for path, digest in behavior_sources
                ],
                "complete": False,
                "cases": completed_records,
            },
        )
    final_index = {
        "format": INDEX_FORMAT,
        "version": INDEX_VERSION,
        "replay_fingerprint": replay_fingerprint,
        "model_fingerprint": model_fingerprint,
        "model_provenance": model_provenance,
        "config": replay_config,
        "source_manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "behavior_sidecars": [
            {"path": path, "sha256": digest} for path, digest in behavior_sources
        ],
        "complete": True,
        "num_cases": len(completed_records),
        "cases": completed_records,
    }
    _atomic_json(index_path, final_index)
    return {
        "index_path": str(index_path),
        "replay_fingerprint": replay_fingerprint,
        "model_fingerprint": model_fingerprint,
        "num_cases": len(specs),
        "replayed": replayed,
        "skipped": skipped,
    }


def _parse_layers(value: str) -> Optional[tuple[int, ...]]:
    if value.strip().casefold() == "all":
        return None
    try:
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be 'all' or comma-separated integers"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("layers must not be empty")
    return layers


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
        model_name, model_id=model_id, device=device, dtype=dtype_value
    )


def _load_object(specification: str) -> Any:
    module, separator, attribute = specification.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("factory must use module:attribute syntax")
    return getattr(importlib.import_module(module), attribute)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--behavior-labels", type=Path, default=None)
    parser.add_argument("--model", dest="model_name", choices=("qwen-3b", "qwen-7b"), default="qwen-3b")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--layers", type=_parse_layers, default=None)
    parser.add_argument("--token-span", choices=tuple(sorted(_TOKEN_SPANS)), default="audio")
    parser.add_argument("--pooling", choices=tuple(sorted(_POOLING)), default="mean")
    parser.add_argument(
        "--no-sequence-embedding",
        action="store_true",
        help="Treat hidden-state sequence index 0 as decoder layer 0",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--model-factory",
        help="Optional local module:callable receiving model_name/model_id/device/dtype",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    factory: Callable[..., Any] = (
        _default_model if args.model_factory is None else _load_object(args.model_factory)
    )
    model = factory(
        model_name=args.model_name,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    summary = replay_stage1_trajectories(
        args.manifest,
        args.output_dir,
        model=model,
        behavior_labels=args.behavior_labels,
        model_name=args.model_name,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        layers=args.layers,
        token_span=args.token_span,
        pooling=args.pooling,
        sequence_has_embedding=not args.no_sequence_embedding,
        resume=not args.no_resume,
    )
    print(
        f"Saved {summary['replayed']} case(s), resumed {summary['skipped']} case(s) "
        f"at {summary['index_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INDEX_FORMAT",
    "INDEX_VERSION",
    "OUTPUT_FORMAT",
    "OUTPUT_VERSION",
    "Stage1TrajectoryReplayError",
    "build_parser",
    "replay_stage1_trajectories",
    "validate_replay_payload",
]
