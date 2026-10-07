"""Versioned RQ2 trial artifacts and crash-safe local persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


TRIAL_FORMAT = "rq2-causal-intervention-trial"
TRIAL_VERSION = 1
RESPONSE_FORMAT = "rq2-intervention-response"
LABEL_FORMAT = "rq2-behavior-label"
SIDECAR_VERSION = 1
_SHA256_CHARS = frozenset("0123456789abcdef")


class RQ2ArtifactError(ValueError):
    """Raised when an RQ2 artifact is malformed or unsafe to resume."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrialKey:
    """Stable identity for one intervention condition."""

    run_fingerprint: str
    pair_id: str
    state_key: str
    layer: Optional[int]
    intervention: str
    dose: float = 0.0
    token_scope: str = "audio"
    replicate: int = 0

    def __post_init__(self) -> None:
        for name in ("run_fingerprint", "pair_id", "state_key", "intervention", "token_scope"):
            if not str(getattr(self, name)).strip():
                raise RQ2ArtifactError(f"{name} must be non-blank")
        if self.layer is not None and (
            isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0
        ):
            raise RQ2ArtifactError("layer must be a non-negative integer or None")
        if not math.isfinite(float(self.dose)):
            raise RQ2ArtifactError("dose must be finite")
        if isinstance(self.replicate, bool) or not isinstance(self.replicate, int) or self.replicate < 0:
            raise RQ2ArtifactError("replicate must be non-negative")

    @property
    def trial_id(self) -> str:
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trial_id": self.trial_id}


@dataclass(frozen=True)
class TrialRecord:
    """Public, text-free result for one generated intervention trial."""

    key: TrialKey
    rq2_role: str
    step: int
    baseline_trial_id: Optional[str]
    audit: Mapping[str, Any]
    diagnostic: Mapping[str, Any]
    response_sha256: str

    def __post_init__(self) -> None:
        if self.rq2_role not in {"rq2_dev", "rq2_causal_test", "rq1_reference"}:
            raise RQ2ArtifactError("invalid RQ2 data role")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise RQ2ArtifactError("step must be a non-negative integer")
        if self.baseline_trial_id == self.key.trial_id:
            raise RQ2ArtifactError("a trial cannot use itself as its baseline")
        _validate_sha256(self.response_sha256, "response_sha256")

    def to_record(self) -> dict[str, Any]:
        record = {
            "format": TRIAL_FORMAT,
            "version": TRIAL_VERSION,
            **self.key.to_dict(),
            "rq2_role": self.rq2_role,
            "step": self.step,
            "baseline_trial_id": self.baseline_trial_id,
            "audit": dict(self.audit),
            "diagnostic": dict(self.diagnostic),
            "response_sha256": self.response_sha256,
        }
        return validate_trial_record(record)


@dataclass(frozen=True)
class BehaviorLabel:
    trial_id: str
    label_status: str
    behavior_label: Optional[str]
    refusal_label: Optional[bool]
    compliance_label: Optional[bool]
    jailbreak_success: Optional[bool]
    refusal_score: Optional[float]
    strongreject_score: Optional[float]
    refusal_orientation: Optional[float]
    convincingness: Optional[int]
    specificity: Optional[int]
    response_sha256: str
    judge_fingerprint: str
    error_type: Optional[str] = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.label_status not in {"ok", "unknown"}:
            raise RQ2ArtifactError("label_status must be ok or unknown")
        if not self.trial_id:
            raise RQ2ArtifactError("trial_id is required")
        _validate_sha256(self.response_sha256, "response_sha256")
        _validate_sha256(self.judge_fingerprint, "judge_fingerprint")
        decisions = (
            self.refusal_label,
            self.compliance_label,
            self.jailbreak_success,
            self.refusal_score,
            self.strongreject_score,
            self.refusal_orientation,
            self.convincingness,
            self.specificity,
        )
        if self.label_status == "unknown" and any(value is not None for value in decisions):
            raise RQ2ArtifactError("unknown labels must not contain decisions or scores")
        if self.label_status == "ok" and (self.error_type is not None or self.retryable):
            raise RQ2ArtifactError("ok labels cannot carry an error")
        if self.label_status == "ok":
            if any(value is None for value in decisions):
                raise RQ2ArtifactError("ok labels require complete decisions and scores")
            if self.compliance_label != self.jailbreak_success:
                raise RQ2ArtifactError("compliance_label must equal jailbreak_success")
            if self.refusal_label and self.compliance_label:
                raise RQ2ArtifactError("refusal and compliance cannot both be true")
            if not math.isclose(
                float(self.refusal_orientation),
                1.0 - float(self.strongreject_score),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RQ2ArtifactError("refusal_orientation must equal 1-strongreject_score")

    def to_record(self) -> dict[str, Any]:
        return {"format": LABEL_FORMAT, "version": SIDECAR_VERSION, **asdict(self)}


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_CHARS for char in value):
        raise RQ2ArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _find_private_key(value: Any, path: str = "record") -> Optional[str]:
    forbidden = {"response", "reasoning", "judge_result", "raw_response", "overall_reasoning"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                return f"{path}.{key}"
            found = _find_private_key(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_private_key(item, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_trial_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("format") != TRIAL_FORMAT or record.get("version") != TRIAL_VERSION:
        raise RQ2ArtifactError("unsupported trial record format/version")
    key_fields = {
        name: record.get(name)
        for name in (
            "run_fingerprint",
            "pair_id",
            "state_key",
            "layer",
            "intervention",
            "dose",
            "token_scope",
            "replicate",
        )
    }
    key = TrialKey(**key_fields)
    if record.get("trial_id") != key.trial_id:
        raise RQ2ArtifactError("trial_id does not bind the declared trial key")
    _validate_sha256(record.get("response_sha256"), "response_sha256")
    private_path = _find_private_key(record)
    if private_path:
        raise RQ2ArtifactError(f"analysis trial record contains private field: {private_path}")
    json.dumps(dict(record), ensure_ascii=False, allow_nan=False)
    return dict(record)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    return descriptor, Path(name)


def atomic_json(path: str | Path, value: Any) -> Path:
    output = Path(path).expanduser().resolve()
    descriptor, temporary = _atomic_writer(output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def atomic_text(path: str | Path, value: str) -> Path:
    output = Path(path).expanduser().resolve()
    descriptor, temporary = _atomic_writer(output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def atomic_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    output = Path(path).expanduser().resolve()
    descriptor, temporary = _atomic_writer(output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                json.dump(dict(record), handle, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def append_jsonl_fsync(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one journal record durably; truncated tails are recoverable on resume."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(record), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    with output.open("a+b") as handle:
        handle.seek(0)
        existing = handle.read()
        if existing and not existing.endswith(b"\n"):
            boundary = existing.rfind(b"\n") + 1
            tail = existing[boundary:]
            try:
                parsed = json.loads(tail.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                handle.seek(boundary)
                handle.truncate()
            else:
                if not isinstance(parsed, dict):
                    raise RQ2ArtifactError("JSONL journal tail must be an object")
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        else:
            handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output.parent)
    return output


def atomic_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    output = Path(path).expanduser().resolve()
    fields: list[str] = list(fieldnames or ())
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise RQ2ArtifactError("empty CSV requires explicit fieldnames")
    descriptor, temporary = _atomic_writer(output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return output


def read_jsonl(
    path: str | Path,
    *,
    missing_ok: bool = False,
    recover_truncated: bool = False,
) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if missing_ok and not source.exists():
        return []
    if not source.is_file():
        raise FileNotFoundError(source)
    result: list[dict[str, Any]] = []
    lines = source.read_text(encoding="utf-8").splitlines()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if recover_truncated and line_number == len(lines):
                    break
                raise RQ2ArtifactError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RQ2ArtifactError(f"expected object at {source}:{line_number}")
            result.append(value)
    return result


def records_by_id(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "trial_id",
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        identity = str(record.get(id_field, "")).strip()
        if not identity:
            raise RQ2ArtifactError(f"record {position} has blank {id_field}")
        if identity in result:
            raise RQ2ArtifactError(f"duplicate {id_field}: {identity}")
        result[identity] = dict(record)
    return result


__all__ = [
    "BehaviorLabel",
    "LABEL_FORMAT",
    "RESPONSE_FORMAT",
    "RQ2ArtifactError",
    "SIDECAR_VERSION",
    "TRIAL_FORMAT",
    "TRIAL_VERSION",
    "TrialKey",
    "TrialRecord",
    "atomic_csv",
    "atomic_json",
    "atomic_jsonl",
    "append_jsonl_fsync",
    "atomic_text",
    "canonical_sha256",
    "file_sha256",
    "read_jsonl",
    "records_by_id",
    "validate_trial_record",
]
