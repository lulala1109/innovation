"""Crash-resilient storage for optimization-time tensor trajectories.

Tensor payloads and human-readable metadata deliberately use different files:

* ``trajectory/step_XXXXXX.pt`` contains tensors only.
* ``trajectory/index.json`` describes the available checkpoints.
* ``events.jsonl`` contains optional scalar/text event metadata.

Keeping tensors out of JSON avoids very large manifests and, more importantly,
gives callers a small index that can be read without loading model activations.
Checkpoint files are written to a temporary sibling and atomically renamed so a
process interruption cannot expose a partially written ``.pt`` file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch


_INDEX_VERSION = 1
_CHECKPOINT_RE = re.compile(r"^step_(\d+)\.pt$")


class ArtifactError(RuntimeError):
    """Raised when an artifact store is malformed or cannot be recovered."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata for one tensor checkpoint.

    ``path`` is absolute for convenient loading. ``metadata`` is guaranteed to
    be JSON-compatible and never contains a tensor payload.
    """

    step: int
    path: Path
    tensors: Mapping[str, Mapping[str, Any]]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RecoveredCheckpoint:
    """A checkpoint payload together with its index record."""

    record: CheckpointRecord
    tensors: Mapping[str, torch.Tensor]

    @property
    def step(self) -> int:
        return self.record.step


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    return step


def _json_copy(value: Any, *, name: str) -> Any:
    """Validate and detach metadata from mutable caller-owned containers."""

    def contains_tensor(item: Any) -> bool:
        if isinstance(item, torch.Tensor):
            return True
        if isinstance(item, Mapping):
            return any(
                contains_tensor(key) or contains_tensor(nested)
                for key, nested in item.items()
            )
        if isinstance(item, (list, tuple)):
            return any(contains_tensor(nested) for nested in item)
        return False

    if contains_tensor(value):
        raise TypeError(
            f"{name} must contain metadata only; save tensors in the "
            "checkpoint payload"
        )
    try:
        # Round-tripping both validates and prevents later caller mutation from
        # silently changing the in-memory index representation.
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite, JSON-serializable metadata") from exc


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic rename."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


class TrajectoryArtifactStore:
    """Store and recover tensor checkpoints for a single experiment case.

    Args:
        root: Case directory. It may safely contain other result files.
        trajectory_dirname: Subdirectory reserved for tensor checkpoints.

    The store is intentionally independent of the old Stage-1/Stage-2 result
    schema. A checkpoint payload is a flat mapping from meaningful tensor names
    (for example ``delta`` or ``refusal_states``) to tensors.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        trajectory_dirname: str = "trajectory",
    ) -> None:
        self.root = Path(root)
        self.trajectory_dir = self.root / trajectory_dirname
        self.index_path = self.trajectory_dir / "index.json"
        self.events_path = self.root / "events.jsonl"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, step: int) -> Path:
        return self.trajectory_dir / f"step_{step:06d}.pt"

    def _empty_index(self) -> Dict[str, Any]:
        return {
            "format": "safety-state-trajectory",
            "version": _INDEX_VERSION,
            "checkpoints": [],
        }

    def _read_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index()
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read artifact index: {self.index_path}") from exc
        if not isinstance(index, dict) or not isinstance(
            index.get("checkpoints"), list
        ):
            raise ArtifactError(f"malformed artifact index: {self.index_path}")
        if index.get("version") != _INDEX_VERSION:
            raise ArtifactError(
                f"unsupported artifact index version: {index.get('version')!r}"
            )
        return index

    def _indexed_records(self) -> Dict[int, Mapping[str, Any]]:
        records: Dict[int, Mapping[str, Any]] = {}
        for item in self._read_index()["checkpoints"]:
            if not isinstance(item, dict):
                raise ArtifactError("checkpoint index entries must be objects")
            try:
                step = _validate_step(item["step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError("checkpoint index contains an invalid step") from exc
            if step in records:
                raise ArtifactError(f"duplicate step {step} in checkpoint index")
            records[step] = item
        return records

    def list_checkpoints(self) -> Tuple[CheckpointRecord, ...]:
        """List complete checkpoint files in ascending step order.

        Files are also scanned directly. Consequently, a tensor file that was
        atomically committed immediately before an interruption is recoverable
        even if the process did not get a chance to update ``index.json``.
        Such an orphan has empty metadata until it is saved/indexed again.
        """

        indexed = self._indexed_records()
        discovered: Dict[int, Path] = {}
        for path in self.trajectory_dir.glob("step_*.pt"):
            match = _CHECKPOINT_RE.match(path.name)
            if match is not None and path.is_file():
                discovered[int(match.group(1))] = path

        records: List[CheckpointRecord] = []
        for step in sorted(discovered):
            item = indexed.get(step, {})
            records.append(
                CheckpointRecord(
                    step=step,
                    path=discovered[step],
                    tensors=item.get("tensors", {}),
                    metadata=item.get("metadata", {}),
                )
            )
        return tuple(records)

    def list_steps(self) -> List[int]:
        """Return recoverable checkpoint steps in ascending order."""

        return [record.step for record in self.list_checkpoints()]

    def latest_step(self) -> Optional[int]:
        """Return the newest recoverable step, or ``None`` for an empty store."""

        steps = self.list_steps()
        return steps[-1] if steps else None

    def checkpoint_record(self, step: int) -> CheckpointRecord:
        """Return metadata for ``step`` without loading any tensor."""

        step = _validate_step(step)
        for record in self.list_checkpoints():
            if record.step == step:
                return record
        raise KeyError(f"checkpoint step {step} does not exist")

    def save_checkpoint(
        self,
        step: int,
        tensors: Mapping[str, torch.Tensor],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        overwrite: bool = False,
    ) -> CheckpointRecord:
        """Atomically save a tensor-only checkpoint and update its JSON index."""

        step = _validate_step(step)
        if not isinstance(tensors, Mapping) or not tensors:
            raise ValueError("tensors must be a non-empty mapping")

        cpu_tensors: Dict[str, torch.Tensor] = {}
        tensor_index: Dict[str, Mapping[str, Any]] = {}
        for name, value in tensors.items():
            if not isinstance(name, str) or not name:
                raise TypeError("tensor names must be non-empty strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"checkpoint value {name!r} is not a tensor")
            tensor = value.detach().to(device="cpu")
            cpu_tensors[name] = tensor
            tensor_index[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "requires_grad": bool(value.requires_grad),
            }

        safe_metadata = _json_copy(metadata or {}, name="metadata")
        if not isinstance(safe_metadata, dict):
            raise TypeError("metadata must be a JSON object")

        final_path = self._checkpoint_path(step)
        if final_path.exists() and not overwrite:
            raise FileExistsError(f"checkpoint step {step} already exists")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            dir=str(self.trajectory_dir),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(cpu_tensors, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, final_path)
            _fsync_directory(self.trajectory_dir)
        except BaseException:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

        index = self._read_index()
        new_item = {
            "step": step,
            "path": str(final_path.relative_to(self.root)),
            "saved_at": _utc_now(),
            "tensors": tensor_index,
            "metadata": safe_metadata,
        }
        old_items = [
            item for item in index["checkpoints"] if item.get("step") != step
        ]
        old_items.append(new_item)
        index["checkpoints"] = sorted(old_items, key=lambda item: item["step"])
        _atomic_json_dump(self.index_path, index)
        return self.checkpoint_record(step)

    def load_checkpoint(
        self,
        step: Optional[int] = None,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Mapping[str, torch.Tensor]:
        """Load ``step`` (or the latest step) and validate its tensor payload."""

        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError("artifact store contains no checkpoints")
        record = self.checkpoint_record(step)
        try:
            # ``weights_only`` prevents arbitrary pickled objects from being
            # instantiated when loading a checkpoint we define as tensor-only.
            payload = torch.load(
                record.path, map_location=map_location, weights_only=True
            )
        except TypeError:  # Compatibility with older supported PyTorch builds.
            payload = torch.load(record.path, map_location=map_location)
        if not isinstance(payload, dict) or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in payload.items()
        ):
            raise ArtifactError(f"checkpoint {record.path} is not tensor-only")
        return payload

    def recover_checkpoint(
        self,
        step: Optional[int] = None,
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> RecoveredCheckpoint:
        """Recover tensors and metadata for resuming an interrupted run."""

        if step is None:
            step = self.latest_step()
            if step is None:
                raise FileNotFoundError("artifact store contains no checkpoints")
        record = self.checkpoint_record(step)
        return RecoveredCheckpoint(
            record=record,
            tensors=self.load_checkpoint(step, map_location=map_location),
        )

    def write_run_metadata(self, metadata: Mapping[str, Any]) -> Path:
        """Atomically write scalar/text run metadata to ``run.json``."""

        safe_metadata = _json_copy(metadata, name="run metadata")
        if not isinstance(safe_metadata, dict):
            raise TypeError("run metadata must be a JSON object")
        path = self.root / "run.json"
        _atomic_json_dump(path, safe_metadata)
        return path

    def append_event(self, metadata: Mapping[str, Any]) -> Path:
        """Append a metadata-only event with a durable JSONL write."""

        safe_metadata = _json_copy(metadata, name="event metadata")
        if not isinstance(safe_metadata, dict):
            raise TypeError("event metadata must be a JSON object")
        event = dict(safe_metadata)
        event.setdefault("timestamp", _utc_now())
        encoded = json.dumps(
            event, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(self.events_path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644
        )
        try:
            os.write(descriptor, (encoded + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self.events_path


__all__ = [
    "ArtifactError",
    "CheckpointRecord",
    "RecoveredCheckpoint",
    "TrajectoryArtifactStore",
]
