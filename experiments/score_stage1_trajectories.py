#!/usr/bin/env python3
"""Score held-out Stage-1 hidden-state trajectories with H/R probes.

The replay artifact and the probe checkpoint are treated as immutable inputs.
This bridge keeps learned probe probabilities separate from projections on the
independently estimated class-mean directions.  In particular, probe weights
are never used as a substitute for a missing mean-difference direction.

PyTorch is imported lazily so ``--help`` remains lightweight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPLAY_INDEX_FORMAT = "stage1-trajectory-hidden-state-index"
REPLAY_CASE_FORMAT = "stage1-trajectory-hidden-states"
REPLAY_VERSION = 1
PROBE_FORMAT = "dual-safety-state-layerwise-linear-probes"
SCORE_FORMAT = "stage1-trajectory-scores"
SCORE_VERSION = 1
PHASE_THRESHOLDS = {
    "early": [0.0, 1.0 / 3.0],
    "middle": [1.0 / 3.0, 2.0 / 3.0],
    "late": [2.0 / 3.0, 1.0],
}
SCORE_KEYS = ("H_probe", "R_probe", "H_direction", "R_direction")
DELTA_KEYS = (
    "delta_H_probe",
    "delta_R_probe",
    "delta_H_direction",
    "delta_R_direction",
)
LONG_FIELDS = (
    "case_id",
    "pair_id",
    "step",
    "total_steps",
    "progress",
    "phase",
    "layer",
    "attack_loss",
    "H_probe",
    "R_probe",
    "H_direction",
    "R_direction",
    "delta_H_probe",
    "delta_R_probe",
    "delta_H_direction",
    "delta_R_direction",
    "label_status",
    "generation_status",
    "refusal_label",
    "compliance_label",
    "jailbreak_success",
    "response_sha256",
    "checkpoint_sha256",
    "experiment_fingerprint",
    "model_fingerprint",
    "replay_fingerprint",
    "probe_checkpoint_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage1ScoringError(ValueError):
    """Raised when a trajectory/probe artifact violates the scoring contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage1ScoringError(f"{name} must be a non-empty string")
    return value.strip()


def _required_sha256(value: Any, *, name: str) -> str:
    text = _required_text(value, name=name).lower()
    if not _SHA256_RE.fullmatch(text):
        raise Stage1ScoringError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise Stage1ScoringError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise Stage1ScoringError(f"{name} must be an integer") from exc
    if result != value or result < minimum:
        raise Stage1ScoringError(f"{name} must be an integer >= {minimum}")
    return result


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe scoring requires PyTorch with torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage1ScoringError(f"{path}: expected a mapping checkpoint")
    return payload


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise Stage1ScoringError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise Stage1ScoringError(f"{path}: expected one JSON object")
    return payload


def _resolve_replay_index(value: str | Path) -> Path:
    candidate = Path(value).expanduser().resolve()
    index = candidate if candidate.is_file() else candidate / "index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Replay index not found: {index}")
    if index.name != "index.json":
        raise Stage1ScoringError("Replay index file must be named index.json")
    return index


def _resolve_child(root: Path, value: Any, *, name: str) -> Path:
    text = _required_text(value, name=name)
    candidate = Path(text).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise Stage1ScoringError(f"{name} escapes the replay directory: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _layer_token(layer: Any) -> str:
    if isinstance(layer, bool) or not isinstance(layer, (int, str)):
        raise Stage1ScoringError("layer keys must be integers or non-empty strings")
    if isinstance(layer, str) and not layer.strip():
        raise Stage1ScoringError("layer string keys must not be blank")
    return str(layer).strip()


def _layer_mapping(source: Sequence[Any], target: Sequence[Any]) -> OrderedDict[Any, Any]:
    """Map replay keys to checkpoint keys without silently merging aliases."""

    target_by_token: dict[str, Any] = {}
    for key in target:
        token = _layer_token(key)
        if token in target_by_token:
            raise Stage1ScoringError(f"ambiguous checkpoint layer alias {token!r}")
        target_by_token[token] = key
    result: "OrderedDict[Any, Any]" = OrderedDict()
    for key in source:
        token = _layer_token(key)
        if token not in target_by_token:
            raise Stage1ScoringError(f"replay layer {key!r} has no probe")
        result[key] = target_by_token.pop(token)
    if target_by_token:
        raise Stage1ScoringError(
            "probe layers are absent from replay: " + ", ".join(target_by_token)
        )
    return result


def _as_steps(value: Any, *, name: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise Stage1ScoringError(f"{name} must be an int64 Tensor[S]")
    if value.dtype != torch.int64:
        raise Stage1ScoringError(f"{name} must use torch.int64")
    return value.detach().cpu().contiguous()


def _as_float_vector(value: Any, *, count: int, name: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor) or value.shape != (count,):
        raise Stage1ScoringError(f"{name} must be a Tensor[{count}]")
    if not value.is_floating_point() or value.is_complex():
        raise Stage1ScoringError(f"{name} must be real floating point")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise Stage1ScoringError(f"{name} contains non-finite values")
    return result


def _case_model_fingerprint(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Stage1ScoringError("replay case metadata must be a mapping")
    return _required_sha256(
        metadata.get("model_fingerprint"), name="case.metadata.model_fingerprint"
    )


def _validate_case(
    payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    path: Path,
    index: Mapping[str, Any],
) -> Mapping[str, Any]:
    import torch

    # Keep this bridge pinned to the replay producer's public validator, then
    # apply scoring-specific descriptor/provenance checks below.
    from experiments.replay_stage1_trajectories import (
        Stage1TrajectoryReplayError,
        validate_replay_payload,
    )

    try:
        validate_replay_payload(
            payload,
            expected_replay_fingerprint=index.get("replay_fingerprint"),
        )
    except Stage1TrajectoryReplayError as exc:
        raise Stage1ScoringError(f"{path}: invalid replay payload: {exc}") from exc

    if payload.get("format") != REPLAY_CASE_FORMAT or payload.get("version") != REPLAY_VERSION:
        raise Stage1ScoringError(f"{path}: unsupported replay case format/version")
    case_id = _required_text(payload.get("case_id"), name=f"{path}: case_id")
    pair_id = _required_text(payload.get("pair_id"), name=f"{path}: pair_id")
    if case_id != _required_text(descriptor.get("case_id"), name="index case_id"):
        raise Stage1ScoringError(f"{path}: case_id disagrees with index")
    if pair_id != _required_text(descriptor.get("pair_id"), name="index pair_id"):
        raise Stage1ScoringError(f"{path}: pair_id disagrees with index")

    steps = _as_steps(payload.get("steps"), name=f"{path}: steps")
    count = int(steps.numel())
    total_steps = _integer(
        payload.get("metadata", {}).get("total_steps"),
        name=f"{path}: metadata.total_steps",
    )
    expected = torch.arange(total_steps + 1, dtype=torch.int64)
    if count != total_steps + 1 or not torch.equal(steps, expected):
        raise Stage1ScoringError(f"{path}: steps must be the complete 0..T grid")
    if _integer(descriptor.get("num_steps"), name="index num_steps", minimum=1) != count:
        raise Stage1ScoringError(f"{path}: num_steps disagrees with index")
    if _integer(descriptor.get("total_steps"), name="index total_steps") != total_steps:
        raise Stage1ScoringError(f"{path}: total_steps disagrees with index")

    hidden = payload.get("hidden_states")
    layers = payload.get("layers")
    if not isinstance(hidden, Mapping) or not hidden:
        raise Stage1ScoringError(f"{path}: hidden_states must be a non-empty mapping")
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise Stage1ScoringError(f"{path}: layers must be a sequence")
    if list(layers) != list(hidden):
        raise Stage1ScoringError(f"{path}: layers must exactly match hidden_states")
    normalized_hidden: "OrderedDict[Any, Any]" = OrderedDict()
    for layer, states in hidden.items():
        _layer_token(layer)
        if not isinstance(states, torch.Tensor) or states.ndim != 2:
            raise Stage1ScoringError(f"{path}: hidden_states[{layer!r}] needs [S,D]")
        if states.shape[0] != count or states.shape[1] < 1:
            raise Stage1ScoringError(f"{path}: hidden_states[{layer!r}] shape disagrees")
        if not states.is_floating_point() or states.is_complex():
            raise Stage1ScoringError(f"{path}: hidden_states[{layer!r}] must be float")
        state = states.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(state).all()):
            raise Stage1ScoringError(f"{path}: hidden_states[{layer!r}] is non-finite")
        normalized_hidden[layer] = state

    attack_loss = _as_float_vector(
        payload.get("attack_loss"), count=count, name=f"{path}: attack_loss"
    )
    behavior = payload.get("behavior")
    row_metadata = payload.get("row_metadata")
    if not isinstance(behavior, list) or len(behavior) != count:
        raise Stage1ScoringError(f"{path}: behavior must contain S mappings")
    if not isinstance(row_metadata, list) or len(row_metadata) != count:
        raise Stage1ScoringError(f"{path}: row_metadata must contain S mappings")

    hashes = payload.get("checkpoint_sha256")
    if not isinstance(hashes, list) or len(hashes) != count:
        raise Stage1ScoringError(f"{path}: checkpoint_sha256 must contain S digests")
    hashes = [
        _required_sha256(value, name=f"{path}: checkpoint_sha256[{position}]")
        for position, value in enumerate(hashes)
    ]
    descriptor_hashes = descriptor.get("checkpoint_sha256")
    if descriptor_hashes != hashes:
        raise Stage1ScoringError(f"{path}: checkpoint_sha256 disagrees with index")
    fingerprint = _required_sha256(
        payload.get("experiment_fingerprint"), name=f"{path}: experiment_fingerprint"
    )
    if fingerprint != _required_sha256(
        descriptor.get("experiment_fingerprint"), name="index experiment_fingerprint"
    ):
        raise Stage1ScoringError(f"{path}: experiment_fingerprint disagrees with index")

    normalized_behavior: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for position, step in enumerate(range(count)):
        behavior_row = behavior[position]
        metadata_row = row_metadata[position]
        if not isinstance(behavior_row, Mapping) or not isinstance(metadata_row, Mapping):
            raise Stage1ScoringError(f"{path}: behavior/row_metadata entries must map")
        for field, expected_value in (
            ("case_id", case_id), ("pair_id", pair_id), ("step", step)
        ):
            if metadata_row.get(field) != expected_value:
                raise Stage1ScoringError(f"{path}: row_metadata[{step}].{field} disagrees")
            if behavior_row.get(field) != expected_value:
                raise Stage1ScoringError(f"{path}: behavior[{step}].{field} disagrees")
        if metadata_row.get("total_steps") != total_steps:
            raise Stage1ScoringError(f"{path}: total_steps mismatch at step {step}")
        expected_progress = step / total_steps if total_steps else 0.0
        try:
            row_progress = float(metadata_row.get("progress"))
        except (TypeError, ValueError) as exc:
            raise Stage1ScoringError(f"{path}: invalid progress at step {step}") from exc
        if not math.isclose(row_progress, expected_progress, rel_tol=0.0, abs_tol=1e-12):
            raise Stage1ScoringError(f"{path}: progress mismatch at step {step}")
        if metadata_row.get("phase") != _phase(expected_progress):
            raise Stage1ScoringError(f"{path}: phase mismatch at step {step}")
        row_hash = _required_sha256(
            metadata_row.get("checkpoint_sha256"),
            name=f"{path}: row_metadata[{step}].checkpoint_sha256",
        )
        if row_hash != hashes[position]:
            raise Stage1ScoringError(f"{path}: checkpoint digest mismatch at step {step}")
        row_fingerprint = _required_sha256(
            metadata_row.get("experiment_fingerprint"),
            name=f"{path}: row_metadata[{step}].experiment_fingerprint",
        )
        if row_fingerprint != fingerprint:
            raise Stage1ScoringError(f"{path}: fingerprint mismatch at step {step}")
        response = behavior_row.get("response")
        if not isinstance(response, str):
            raise Stage1ScoringError(f"{path}: behavior response must be text at step {step}")
        response_digest = behavior_row.get("response_sha256")
        if response_digest is None:
            if response:
                raise Stage1ScoringError(
                    f"{path}: non-empty response lacks SHA-256 at step {step}"
                )
        else:
            response_digest = _required_sha256(
                response_digest, name=f"{path}: behavior[{step}].response_sha256"
            )
            if response_digest != hashlib.sha256(response.encode("utf-8")).hexdigest():
                raise Stage1ScoringError(
                    f"{path}: response SHA-256 mismatch at step {step}"
                )
        if metadata_row.get("response_sha256") != response_digest:
            raise Stage1ScoringError(
                f"{path}: response provenance mismatch at step {step}"
            )
        row_loss = float(metadata_row.get("attack_loss"))
        if not math.isfinite(row_loss) or not math.isclose(
            row_loss, float(attack_loss[position]), rel_tol=1e-6, abs_tol=1e-7
        ):
            raise Stage1ScoringError(f"{path}: attack_loss mismatch at step {step}")
        normalized_behavior.append(dict(behavior_row))
        normalized_rows.append(dict(metadata_row))

    model_fingerprint = _case_model_fingerprint(payload)
    if model_fingerprint != _required_sha256(
        index.get("model_fingerprint"), name="index.model_fingerprint"
    ):
        raise Stage1ScoringError(f"{path}: model_fingerprint disagrees with index")
    replay_fingerprint = _required_sha256(
        payload.get("metadata", {}).get("replay_fingerprint"),
        name=f"{path}: metadata.replay_fingerprint",
    )
    if replay_fingerprint != _required_sha256(
        index.get("replay_fingerprint"), name="index.replay_fingerprint"
    ):
        raise Stage1ScoringError(f"{path}: replay_fingerprint disagrees with index")
    return {
        "case_id": case_id,
        "pair_id": pair_id,
        "steps": steps,
        "total_steps": total_steps,
        "layers": list(layers),
        "hidden_states": normalized_hidden,
        "attack_loss": attack_loss,
        "behavior": normalized_behavior,
        "row_metadata": normalized_rows,
        "checkpoint_sha256": hashes,
        "experiment_fingerprint": fingerprint,
        "model_fingerprint": model_fingerprint,
        "replay_fingerprint": replay_fingerprint,
        "metadata": dict(payload["metadata"]),
        "path": path,
    }


def load_replay_artifact(value: str | Path) -> Mapping[str, Any]:
    """Load and validate the full index plus every immutable case shard."""

    index_path = _resolve_replay_index(value)
    index = _read_json(index_path)
    if index.get("format") != REPLAY_INDEX_FORMAT or index.get("version") != REPLAY_VERSION:
        raise Stage1ScoringError("unsupported replay index format/version")
    if index.get("complete") is not True:
        raise Stage1ScoringError("replay index is incomplete")
    _required_sha256(index.get("model_fingerprint"), name="index.model_fingerprint")
    replay_fingerprint = _required_sha256(
        index.get("replay_fingerprint"), name="index.replay_fingerprint"
    )
    replay_config = index.get("config")
    if not isinstance(replay_config, Mapping):
        raise Stage1ScoringError("index.config must be a mapping")
    canonical_replay_digest = hashlib.sha256(
        json.dumps(
            replay_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if replay_fingerprint != canonical_replay_digest:
        raise Stage1ScoringError("index replay_fingerprint does not bind config")
    source_manifest = index.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise Stage1ScoringError("index.source_manifest must be a mapping")
    source_manifest_sha = _required_sha256(
        source_manifest.get("sha256"), name="index.source_manifest.sha256"
    )
    _required_text(source_manifest.get("path"), name="index.source_manifest.path")
    if replay_config.get("manifest_sha256") != source_manifest_sha:
        raise Stage1ScoringError("replay config/source manifest provenance mismatch")
    behavior_sidecars = index.get("behavior_sidecars")
    if not isinstance(behavior_sidecars, list) or not behavior_sidecars:
        raise Stage1ScoringError("index.behavior_sidecars must be a non-empty list")
    normalized_sidecars = []
    for position, sidecar in enumerate(behavior_sidecars):
        if not isinstance(sidecar, Mapping):
            raise Stage1ScoringError(f"behavior_sidecars[{position}] must be a mapping")
        normalized_sidecars.append(
            {
                "path": _required_text(
                    sidecar.get("path"), name=f"behavior_sidecars[{position}].path"
                ),
                "sha256": _required_sha256(
                    sidecar.get("sha256"), name=f"behavior_sidecars[{position}].sha256"
                ),
            }
        )
    if replay_config.get("behavior_sidecars") != normalized_sidecars:
        raise Stage1ScoringError("replay config/behavior sidecar provenance mismatch")
    model_provenance = index.get("model_provenance")
    if not isinstance(model_provenance, Mapping):
        raise Stage1ScoringError("index.model_provenance must be a mapping")
    canonical_model = {
        "model_name": _required_text(
            model_provenance.get("model_name"), name="index model_name"
        ),
        "model_id": _required_text(
            model_provenance.get("model_id"), name="index model_id"
        ),
        "dtype": _required_text(model_provenance.get("dtype"), name="index dtype"),
    }
    canonical_digest = hashlib.sha256(
        json.dumps(
            canonical_model,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if model_provenance.get("model_fingerprint") != canonical_digest:
        raise Stage1ScoringError("index.model_provenance fingerprint mismatch")
    if index.get("model_fingerprint") != canonical_digest:
        raise Stage1ScoringError("index model_fingerprint disagrees with provenance")
    if replay_config.get("model_provenance") != model_provenance:
        raise Stage1ScoringError("replay config/model provenance mismatch")
    if replay_config.get("model_fingerprint") != canonical_digest:
        raise Stage1ScoringError("replay config/model fingerprint mismatch")
    descriptors = index.get("cases")
    if not isinstance(descriptors, list) or not descriptors:
        raise Stage1ScoringError("replay index cases must be a non-empty list")
    if _integer(index.get("num_cases"), name="index.num_cases", minimum=1) != len(descriptors):
        raise Stage1ScoringError("index.num_cases disagrees with cases")
    cases = []
    seen_cases: set[str] = set()
    seen_pairs: set[str] = set()
    for position, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise Stage1ScoringError(f"index.cases[{position}] must be a mapping")
        path = _resolve_child(
            index_path.parent, descriptor.get("path"), name=f"index.cases[{position}].path"
        )
        digest = _required_sha256(
            descriptor.get("sha256"), name=f"index.cases[{position}].sha256"
        )
        if sha256_file(path) != digest:
            raise Stage1ScoringError(f"{path}: case checkpoint SHA-256 mismatch")
        case = _validate_case(_safe_torch_load(path), descriptor, path=path, index=index)
        if case["case_id"] in seen_cases:
            raise Stage1ScoringError(f"duplicate case_id {case['case_id']!r}")
        if case["pair_id"] in seen_pairs:
            raise Stage1ScoringError(f"duplicate pair_id {case['pair_id']!r}")
        seen_cases.add(case["case_id"])
        seen_pairs.add(case["pair_id"])
        cases.append(case)
    reference = cases[0]
    for case in cases[1:]:
        if not __import__("torch").equal(case["steps"], reference["steps"]):
            raise Stage1ScoringError("all replay cases must share the complete step grid")
        if [_layer_token(x) for x in case["layers"]] != [
            _layer_token(x) for x in reference["layers"]
        ]:
            raise Stage1ScoringError("all replay cases must use identical layers")
        for layer, reference_layer in zip(case["layers"], reference["layers"]):
            if case["hidden_states"][layer].shape[1] != reference["hidden_states"][reference_layer].shape[1]:
                raise Stage1ScoringError("hidden size changes across replay cases")
        for field in ("model_fingerprint", "replay_fingerprint"):
            if case[field] != reference[field]:
                raise Stage1ScoringError(f"all replay cases must share {field}")
        for field in ("pooling", "token_span", "sequence_has_embedding"):
            if case["metadata"].get(field) != reference["metadata"].get(field):
                raise Stage1ScoringError(f"all replay cases must share {field}")
    return {"index_path": index_path, "index": index, "cases": cases}


def _probe_source_provenance(
    checkpoint: Mapping[str, Any],
    *,
    allow_unverified_provenance: bool,
) -> Mapping[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise Stage1ScoringError("probe metadata must be a mapping")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage1ScoringError("probe v2 metadata.provenance is required")
    source_digest = provenance.get("source_payload_sha256")
    if source_digest is None:
        if not allow_unverified_provenance:
            raise Stage1ScoringError(
                "probe source provenance is missing source_payload_sha256; save the "
                "training payload before training or explicitly pass "
                "--allow-unverified-provenance"
            )
    else:
        _required_sha256(
            source_digest,
            name="probe metadata.provenance.source_payload_sha256",
        )
    training = provenance.get("training_payload_metadata", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise Stage1ScoringError("probe training_payload_metadata must be a mapping")
    # Normalized fields may be copied to provenance; original collector fields
    # remain nested.  Prefer explicit normalized values.
    merged = dict(training)
    merged.update({key: value for key, value in provenance.items() if key != "training_payload_metadata"})
    return merged


def _lookup_layer(mapping: Mapping[Any, Any], layer: Any, *, name: str) -> Any:
    matches = [value for key, value in mapping.items() if _layer_token(key) == _layer_token(layer)]
    if len(matches) != 1:
        raise Stage1ScoringError(f"{name} must contain exactly one entry for layer {layer!r}")
    return matches[0]


def _direction_bundle(
    checkpoint: Mapping[str, Any],
    *,
    probe_layers: Sequence[Any],
    hidden_sizes: Mapping[Any, int],
    allow_missing: bool,
) -> Optional[Mapping[str, Mapping[Any, tuple[Any, Any]]]]:
    import torch

    directions = checkpoint.get("directions")
    class_means = checkpoint.get("class_means")
    if not isinstance(directions, Mapping) or not isinstance(class_means, Mapping):
        if allow_missing:
            return None
        raise Stage1ScoringError(
            "Probe checkpoint has no independent directions/class_means; use a v2 "
            "checkpoint or explicitly pass --allow-missing-directions"
        )
    result: dict[str, OrderedDict[Any, tuple[Any, Any]]] = {}
    for state in ("harmfulness", "refusal"):
        state_directions = directions.get(state)
        state_means = class_means.get(state)
        if not isinstance(state_directions, Mapping) or not isinstance(state_means, Mapping):
            raise Stage1ScoringError(f"probe {state} directions/class_means must map layers")
        by_layer: "OrderedDict[Any, tuple[Any, Any]]" = OrderedDict()
        for layer in probe_layers:
            size = int(hidden_sizes[layer])
            raw_direction = _lookup_layer(state_directions, layer, name=f"directions.{state}")
            raw_means = _lookup_layer(state_means, layer, name=f"class_means.{state}")
            if not isinstance(raw_direction, torch.Tensor) or raw_direction.shape != (size,):
                raise Stage1ScoringError(f"directions.{state}[{layer!r}] needs Tensor[{size}]")
            if not isinstance(raw_means, Mapping):
                raise Stage1ScoringError(f"class_means.{state}[{layer!r}] must map classes")
            negative = raw_means.get("negative")
            positive = raw_means.get("positive")
            if not isinstance(negative, torch.Tensor) or negative.shape != (size,):
                raise Stage1ScoringError(f"negative mean for {state}/{layer!r} has wrong shape")
            if not isinstance(positive, torch.Tensor) or positive.shape != (size,):
                raise Stage1ScoringError(f"positive mean for {state}/{layer!r} has wrong shape")
            direction = raw_direction.detach().to(device="cpu", dtype=torch.float32)
            negative = negative.detach().to(device="cpu", dtype=torch.float32)
            positive = positive.detach().to(device="cpu", dtype=torch.float32)
            if not all(bool(torch.isfinite(item).all()) for item in (direction, negative, positive)):
                raise Stage1ScoringError(f"non-finite direction/class mean for {state}/{layer!r}")
            expected = positive - negative
            if not torch.allclose(direction, expected, rtol=1e-5, atol=1e-6):
                raise Stage1ScoringError(
                    f"directions.{state}[{layer!r}] is not positive-negative"
                )
            by_layer[layer] = (direction.contiguous(), ((positive + negative) / 2).contiguous())
        result[state] = by_layer
    return result


def _validate_probe_and_build(
    path: str | Path,
    replay: Mapping[str, Any],
    *,
    allow_missing_directions: bool,
    allow_unverified_provenance: bool,
) -> Mapping[str, Any]:
    import torch
    from core.safety_state import DualSafetyStateScorer

    probe_path = Path(path).expanduser().resolve()
    checkpoint = _safe_torch_load(probe_path)
    if checkpoint.get("format") != PROBE_FORMAT:
        raise Stage1ScoringError("unsupported probe checkpoint format")
    version = _integer(checkpoint.get("version", 1), name="probe version", minimum=1)
    if version not in (1, 2):
        raise Stage1ScoringError(f"unsupported probe checkpoint version {version}")
    if version < 2 and not allow_missing_directions:
        raise Stage1ScoringError(
            "Probe v1 has no independent mean-difference directions; train a v2 "
            "checkpoint or explicitly pass --allow-missing-directions"
        )
    hidden_sizes = checkpoint.get("hidden_sizes")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(hidden_sizes, Mapping) or not hidden_sizes:
        raise Stage1ScoringError("probe hidden_sizes must be a non-empty mapping")
    if not isinstance(state_dict, Mapping):
        raise Stage1ScoringError("probe state_dict must be a mapping")
    probe_layers = list(hidden_sizes)
    replay_layers = replay["cases"][0]["layers"]
    layer_map = _layer_mapping(replay_layers, probe_layers)
    for replay_layer, probe_layer in layer_map.items():
        declared = _integer(hidden_sizes[probe_layer], name=f"hidden_sizes[{probe_layer!r}]", minimum=1)
        actual = int(replay["cases"][0]["hidden_states"][replay_layer].shape[1])
        if actual != declared:
            raise Stage1ScoringError(
                f"hidden size mismatch for layer {replay_layer!r}: replay={actual}, probe={declared}"
            )
    scorer = DualSafetyStateScorer(hidden_size=hidden_sizes, trainable=False)
    try:
        scorer.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise Stage1ScoringError("probe state_dict does not match hidden_sizes") from exc
    scorer.eval()
    directions = _direction_bundle(
        checkpoint,
        probe_layers=probe_layers,
        hidden_sizes=hidden_sizes,
        allow_missing=allow_missing_directions,
    )

    provenance = (
        _probe_source_provenance(
            checkpoint,
            allow_unverified_provenance=allow_unverified_provenance,
        )
        if version >= 2
        else {}
    )
    replay_meta = replay["cases"][0]["metadata"]
    for field in ("pooling", "token_span", "sequence_has_embedding"):
        if field not in provenance:
            if allow_unverified_provenance:
                continue
            raise Stage1ScoringError(f"probe source provenance is missing {field}")
        if provenance[field] != replay_meta.get(field):
            raise Stage1ScoringError(f"probe/replay provenance mismatch for {field}")
    if "layers" in provenance:
        if [_layer_token(x) for x in provenance["layers"]] != [_layer_token(x) for x in replay_layers]:
            raise Stage1ScoringError("probe/replay provenance mismatch for layers")
    # A model fingerprint is required when the training collector recorded it.
    # Its absence is explicit in output metadata rather than fabricated.
    source_model_fingerprint = provenance.get("model_fingerprint")
    model_provenance = provenance.get("model_provenance")
    if source_model_fingerprint is None and isinstance(model_provenance, Mapping):
        source_model_fingerprint = model_provenance.get("model_fingerprint")
    if source_model_fingerprint is not None:
        if _required_sha256(source_model_fingerprint, name="probe source model_fingerprint") != replay["cases"][0]["model_fingerprint"]:
            raise Stage1ScoringError("probe/replay model_fingerprint mismatch")
    elif not allow_unverified_provenance:
        raise Stage1ScoringError(
            "probe source provenance is missing model_fingerprint; regenerate the "
            "probe from a provenance-aware collection or explicitly pass "
            "--allow-unverified-provenance"
        )
    return {
        "path": probe_path,
        "sha256": sha256_file(probe_path),
        "checkpoint": checkpoint,
        "version": version,
        "scorer": scorer,
        "directions": directions,
        "layer_map": layer_map,
        "provenance": provenance,
        "model_fingerprint_verified": source_model_fingerprint is not None,
    }


def _phase(progress: float) -> str:
    if not 0.0 <= progress <= 1.0:
        raise Stage1ScoringError("progress must be within [0, 1]")
    if progress < 1.0 / 3.0:
        return "early"
    if progress < 2.0 / 3.0:
        return "middle"
    return "late"


def _behavior_value(behavior: Mapping[str, Any], row: Mapping[str, Any], *names: str) -> Any:
    for source in (behavior, row):
        for name in names:
            if name in source:
                return source[name]
    return None


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS, extrasaction="raise")
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


def validate_score_payload(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    """Validate the public ``[N,L,S]`` score payload contract."""

    import torch

    if payload.get("format") != SCORE_FORMAT or payload.get("version") != SCORE_VERSION:
        raise Stage1ScoringError("unsupported score payload format/version")
    case_ids = payload.get("case_ids")
    pair_ids = payload.get("pair_ids")
    layers = payload.get("layers")
    steps = payload.get("steps")
    if not isinstance(case_ids, list) or not case_ids:
        raise Stage1ScoringError("case_ids must be a non-empty list")
    if not isinstance(pair_ids, list) or len(pair_ids) != len(case_ids):
        raise Stage1ScoringError("pair_ids must align with case_ids")
    if len(set(case_ids)) != len(case_ids) or len(set(pair_ids)) != len(pair_ids):
        raise Stage1ScoringError("case_ids and pair_ids must each be unique")
    if not isinstance(layers, list) or not layers:
        raise Stage1ScoringError("layers must be a non-empty list")
    steps = _as_steps(steps, name="score steps")
    n, ell, s = len(case_ids), len(layers), int(steps.numel())
    if not torch.equal(steps, torch.arange(s, dtype=torch.int64)):
        raise Stage1ScoringError("score steps must be the complete 0..T grid")
    scores = payload.get("scores")
    deltas = payload.get("deltas")
    if not isinstance(scores, Mapping) or tuple(scores) != SCORE_KEYS:
        raise Stage1ScoringError("scores keys/order do not match the v1 contract")
    if not isinstance(deltas, Mapping) or tuple(deltas) != DELTA_KEYS:
        raise Stage1ScoringError("deltas keys/order do not match the v1 contract")
    allow_nan = bool(payload.get("metadata", {}).get("directions_missing"))
    for name, tensor in (*scores.items(), *deltas.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (n, ell, s):
            raise Stage1ScoringError(f"{name} must have shape [{n},{ell},{s}]")
        if not tensor.is_floating_point() or tensor.is_complex():
            raise Stage1ScoringError(f"{name} must be real floating point")
        if not allow_nan or "direction" not in name.lower():
            if not bool(torch.isfinite(tensor).all()):
                raise Stage1ScoringError(f"{name} contains non-finite values")
    attack_loss = payload.get("attack_loss")
    if not isinstance(attack_loss, torch.Tensor) or attack_loss.shape != (n, s):
        raise Stage1ScoringError(f"attack_loss must have shape [{n},{s}]")
    behavior = payload.get("behavior")
    if not isinstance(behavior, list) or len(behavior) != n or any(
        not isinstance(rows, list) or len(rows) != s for rows in behavior
    ):
        raise Stage1ScoringError("behavior must have the [N][S] grid")
    return n, ell, s


def score_stage1_trajectories(
    replay_input: str | Path,
    probe_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    allow_missing_directions: bool = False,
    allow_unverified_provenance: bool = False,
) -> Mapping[str, Any]:
    """Score every held-out case/layer/step and atomically write PT plus CSV."""

    import torch

    replay = load_replay_artifact(replay_input)
    probe = _validate_probe_and_build(
        probe_checkpoint,
        replay,
        allow_missing_directions=allow_missing_directions,
        allow_unverified_provenance=allow_unverified_provenance,
    )
    cases = replay["cases"]
    replay_layers = list(cases[0]["layers"])
    steps = cases[0]["steps"].clone()
    n, ell, s = len(cases), len(replay_layers), int(steps.numel())
    score_storage = OrderedDict(
        (name, torch.empty((n, ell, s), dtype=torch.float32)) for name in SCORE_KEYS
    )
    attack_loss = torch.stack([case["attack_loss"] for case in cases], dim=0)

    with torch.no_grad():
        for case_index, case in enumerate(cases):
            probe_states = OrderedDict(
                (probe_layer, case["hidden_states"][replay_layer])
                for replay_layer, probe_layer in probe["layer_map"].items()
            )
            dual = probe["scorer"](probe_states)
            for layer_index, replay_layer in enumerate(replay_layers):
                probe_layer = probe["layer_map"][replay_layer]
                score_storage["H_probe"][case_index, layer_index] = dual.harmfulness[probe_layer]
                score_storage["R_probe"][case_index, layer_index] = dual.refusal[probe_layer]
                if probe["directions"] is None:
                    score_storage["H_direction"][case_index, layer_index].fill_(float("nan"))
                    score_storage["R_direction"][case_index, layer_index].fill_(float("nan"))
                    continue
                states = probe_states[probe_layer]
                for state, output_name in (
                    ("harmfulness", "H_direction"), ("refusal", "R_direction")
                ):
                    direction, center = probe["directions"][state][probe_layer]
                    norm = direction.norm()
                    projection = torch.zeros(s, dtype=torch.float32) if float(norm) == 0.0 else torch.einsum(
                        "sd,d->s", states - center, direction / norm
                    )
                    score_storage[output_name][case_index, layer_index] = projection

    delta_storage = OrderedDict()
    for score_name, delta_name in zip(SCORE_KEYS, DELTA_KEYS):
        values = score_storage[score_name]
        delta_storage[delta_name] = values - values[:, :, :1]
    metadata = {
        "replay_index": str(replay["index_path"]),
        "replay_index_sha256": sha256_file(replay["index_path"]),
        "replay_fingerprint": cases[0]["replay_fingerprint"],
        "model_fingerprint": cases[0]["model_fingerprint"],
        "probe_checkpoint": str(probe["path"]),
        "probe_checkpoint_sha256": probe["sha256"],
        "probe_format": PROBE_FORMAT,
        "probe_version": probe["version"],
        "probe_source_payload_sha256": probe["provenance"].get("source_payload_sha256"),
        "probe_source_payload_sha256_verified": bool(
            probe["provenance"].get("source_payload_sha256")
        ),
        "probe_model_fingerprint_verified": probe["model_fingerprint_verified"],
        "probe_training_num_samples": probe["checkpoint"].get("metadata", {}).get(
            "num_samples"
        ),
        "probe_training_num_pairs": probe["checkpoint"].get("metadata", {}).get(
            "num_pairs"
        ),
        "probe_training_measurement_split": probe["checkpoint"].get(
            "metadata", {}
        ).get("measurement_split"),
        "probe_training_stage1_role": probe["checkpoint"].get("metadata", {}).get(
            "stage1_role"
        ),
        "probe_training_stage1_provenance_verified": bool(
            probe["checkpoint"].get("metadata", {}).get(
                "stage1_provenance_verified", False
            )
        ),
        "unverified_provenance_allowed": bool(allow_unverified_provenance),
        "directions_missing": probe["directions"] is None,
        "direction_definition": "dot(x-(negative+positive)/2, positive-negative)/||positive-negative||; zero norm -> 0",
        "direction_source": "independent class-mean difference (never probe weights)",
        "pooling": cases[0]["metadata"].get("pooling"),
        "token_span": cases[0]["metadata"].get("token_span"),
        "sequence_has_embedding": cases[0]["metadata"].get("sequence_has_embedding"),
        "phase_thresholds": PHASE_THRESHOLDS,
        "shape_axes": ["case", "layer", "step"],
        "trajectory_num_pairs": n,
        "trajectory_measurement_split": "measurement_val",
        "trajectory_stage1_role": "trajectory_candidate",
    }
    payload: Mapping[str, Any] = {
        "format": SCORE_FORMAT,
        "version": SCORE_VERSION,
        "case_ids": [case["case_id"] for case in cases],
        "pair_ids": [case["pair_id"] for case in cases],
        "layers": replay_layers,
        "steps": steps,
        "total_steps": int(steps[-1]),
        "scores": score_storage,
        "deltas": delta_storage,
        "attack_loss": attack_loss,
        "behavior": [case["behavior"] for case in cases],
        "row_metadata": [case["row_metadata"] for case in cases],
        "experiment_fingerprints": [case["experiment_fingerprint"] for case in cases],
        "checkpoint_sha256": [case["checkpoint_sha256"] for case in cases],
        "metadata": metadata,
    }
    validate_score_payload(payload)

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for step_index, step in enumerate(steps.tolist()):
            progress = step / int(steps[-1]) if int(steps[-1]) else 0.0
            behavior = case["behavior"][step_index]
            row_meta = case["row_metadata"][step_index]
            for layer_index, layer in enumerate(replay_layers):
                row: dict[str, Any] = {
                    "case_id": case["case_id"],
                    "pair_id": case["pair_id"],
                    "step": step,
                    "total_steps": int(steps[-1]),
                    "progress": progress,
                    "phase": _phase(progress),
                    "layer": layer,
                    "attack_loss": float(attack_loss[case_index, step_index]),
                    "label_status": _behavior_value(behavior, row_meta, "label_status"),
                    "generation_status": _behavior_value(behavior, row_meta, "generation_status"),
                    "refusal_label": _behavior_value(behavior, row_meta, "refusal_label", "refusal"),
                    "compliance_label": _behavior_value(behavior, row_meta, "compliance_label", "compliance"),
                    "jailbreak_success": _behavior_value(behavior, row_meta, "jailbreak_success"),
                    "response_sha256": _behavior_value(behavior, row_meta, "response_sha256"),
                    "checkpoint_sha256": case["checkpoint_sha256"][step_index],
                    "experiment_fingerprint": case["experiment_fingerprint"],
                    "model_fingerprint": case["model_fingerprint"],
                    "replay_fingerprint": case["replay_fingerprint"],
                    "probe_checkpoint_sha256": probe["sha256"],
                }
                for name in SCORE_KEYS:
                    row[name] = float(score_storage[name][case_index, layer_index, step_index])
                for name in DELTA_KEYS:
                    row[name] = float(delta_storage[name][case_index, layer_index, step_index])
                rows.append(row)

    destination = Path(output_dir).expanduser().resolve()
    _atomic_torch_save(payload, destination / "state_scores.pt")
    _atomic_csv(rows, destination / "state_scores_long.csv")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-input", required=True, help="Replay output directory or index.json"
    )
    parser.add_argument("--probe-checkpoint", required=True, help="Local v2 probe .pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-missing-directions",
        action="store_true",
        help="Allow a legacy v1 probe; direction tensors are explicitly written as NaN",
    )
    parser.add_argument(
        "--allow-unverified-provenance",
        action="store_true",
        help="Explicit compatibility mode for legacy probes without model provenance",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = score_stage1_trajectories(
        args.replay_input,
        args.probe_checkpoint,
        args.output_dir,
        allow_missing_directions=args.allow_missing_directions,
        allow_unverified_provenance=args.allow_unverified_provenance,
    )
    n, ell, s = validate_score_payload(payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir).expanduser().resolve()),
                "shape": [n, ell, s],
                "directions_missing": payload["metadata"]["directions_missing"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DELTA_KEYS",
    "LONG_FIELDS",
    "PHASE_THRESHOLDS",
    "SCORE_FORMAT",
    "SCORE_KEYS",
    "SCORE_VERSION",
    "Stage1ScoringError",
    "build_parser",
    "load_replay_artifact",
    "score_stage1_trajectories",
    "sha256_file",
    "validate_score_payload",
]
