#!/usr/bin/env python3
"""Build clean X_H refusal references from collected states and trained probes."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from experiments.collect_safety_states import (
    StateCollectionError,
    atomic_torch_save,
    validate_collection_payload,
)


REFERENCE_FORMAT = "pair-clean-refusal-references"
REFERENCE_VERSION = 1


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe reference loading requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Tensor checkpoint must contain a mapping: {path}")
    return payload


def load_probe_scorer(path: str | Path) -> Any:
    """Safely reconstruct a dual safety scorer from a local probe checkpoint."""

    checkpoint = _safe_torch_load(Path(path).expanduser().resolve())
    if "state_dict" not in checkpoint:
        raise ValueError("Probe checkpoint must contain state_dict")
    hidden_sizes = checkpoint.get("hidden_sizes", checkpoint.get("hidden_size"))
    if hidden_sizes is None:
        raise ValueError("Probe checkpoint must declare hidden_sizes or hidden_size")
    from core.safety_state import DualSafetyStateScorer

    scorer = DualSafetyStateScorer(hidden_size=hidden_sizes, trainable=False)
    scorer.load_state_dict(checkpoint["state_dict"], strict=True)
    scorer.eval()
    return scorer


def _load_collection(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        payload = value
    elif isinstance(value, (str, Path)):
        payload = _safe_torch_load(Path(value).expanduser().resolve())
    else:
        raise TypeError("collection must be a mapping or local .pt path")
    validate_collection_payload(payload, require_both_classes=False)
    return payload


def _normalize_layer_scores(
    value: Any,
    *,
    expected_layers: Sequence[Any],
    pair_id: str,
) -> "OrderedDict[Any, Any]":
    import torch

    if hasattr(value, "refusal"):
        value = value.refusal
    if not isinstance(value, Mapping):
        if isinstance(value, torch.Tensor) and value.numel() == len(expected_layers):
            value = {
                layer: score
                for layer, score in zip(expected_layers, value.reshape(-1))
            }
        else:
            raise StateCollectionError(
                "refusal scorer must return a layer-to-score mapping"
            )
    if set(value) != set(expected_layers):
        raise StateCollectionError(
            f"refusal scorer layers do not match collected layers for {pair_id!r}"
        )
    result: "OrderedDict[Any, Any]" = OrderedDict()
    for layer in expected_layers:
        score = value[layer]
        if not isinstance(score, torch.Tensor):
            score = torch.as_tensor(score, dtype=torch.float32)
        score = score.detach().to(device="cpu", dtype=torch.float32)
        if score.numel() != 1 or not bool(torch.isfinite(score).all()):
            raise StateCollectionError(
                f"reference refusal for pair {pair_id!r}, layer {layer!r} "
                "must be one finite scalar"
            )
        result[layer] = score.reshape(())
    return result


def validate_reference_checkpoint(checkpoint: Mapping[str, Any]) -> int:
    """Validate pair ownership, layer agreement, and finite reference values."""

    import torch

    if checkpoint.get("format") != REFERENCE_FORMAT:
        raise StateCollectionError("invalid refusal-reference checkpoint format")
    if checkpoint.get("version") != REFERENCE_VERSION:
        raise StateCollectionError("unsupported refusal-reference checkpoint version")
    wrapper = checkpoint.get("reference_refusal")
    if not isinstance(wrapper, Mapping):
        raise StateCollectionError("reference_refusal must be a mapping")
    references = wrapper.get("by_pair_id")
    if not isinstance(references, Mapping) or not references:
        raise StateCollectionError("reference_refusal.by_pair_id must be non-empty")
    pair_ids = checkpoint.get("pair_ids")
    layers = checkpoint.get("layers")
    if not isinstance(pair_ids, Sequence) or isinstance(pair_ids, (str, bytes)):
        raise StateCollectionError("pair_ids must be a sequence")
    if list(pair_ids) != list(references):
        raise StateCollectionError("pair_ids must exactly match reference pair ordering")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers:
        raise StateCollectionError("layers must be a non-empty sequence")
    for pair_id, layer_scores in references.items():
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise StateCollectionError("reference pair_id must be non-empty")
        if not isinstance(layer_scores, Mapping) or list(layer_scores) != list(layers):
            raise StateCollectionError(
                f"reference layers for pair_id {pair_id!r} do not match layers"
            )
        for layer, score in layer_scores.items():
            if not isinstance(score, torch.Tensor) or score.numel() != 1:
                raise StateCollectionError(
                    f"reference {pair_id!r}/{layer!r} must be a scalar tensor"
                )
            if not score.is_floating_point() or not bool(torch.isfinite(score).all()):
                raise StateCollectionError(
                    f"reference {pair_id!r}/{layer!r} must be finite floating point"
                )
    return len(references)


def build_pair_references(
    collection: str | Path | Mapping[str, Any],
    *,
    scorer: Any,
    output_path: str | Path | None = None,
    per_pair_dir: str | Path | None = None,
) -> Mapping[str, Any]:
    """Score exactly one clean X_H row per pair and atomically save references."""

    import torch

    payload = _load_collection(collection)
    if scorer is None:
        raise TypeError("scorer is required")
    hidden = payload["hidden_states"]
    layers = list(hidden)
    x_h_rows: "OrderedDict[str, int]" = OrderedDict()
    all_pairs: list[str] = []
    for index, pair_id in enumerate(payload["pair_ids"]):
        if pair_id not in all_pairs:
            all_pairs.append(pair_id)
        if payload["states"][index] != "X_H":
            continue
        if int(payload["steps"][index]) != -1:
            raise StateCollectionError("clean X_H reference rows must use step=-1")
        if pair_id in x_h_rows:
            raise StateCollectionError(
                f"pair_id {pair_id!r} has more than one clean X_H row"
            )
        metadata = payload["row_metadata"][index]
        if metadata.get("pair_id") != pair_id or metadata.get("state") != "X_H":
            raise StateCollectionError("X_H row metadata does not own its pair")
        x_h_rows[pair_id] = index
    missing = [pair_id for pair_id in all_pairs if pair_id not in x_h_rows]
    if missing:
        raise StateCollectionError(
            "Every collected pair needs exactly one clean X_H row; missing: "
            + ", ".join(missing[:5])
        )

    references: "OrderedDict[str, OrderedDict[Any, Any]]" = OrderedDict()
    if hasattr(scorer, "eval"):
        scorer.eval()
    for pair_id, index in x_h_rows.items():
        states = OrderedDict(
            (layer, tensor[index : index + 1]) for layer, tensor in hidden.items()
        )
        with torch.no_grad():
            if hasattr(scorer, "score_refusal"):
                values = scorer.score_refusal(states)
            elif callable(scorer):
                values = scorer(states)
            else:
                raise TypeError("scorer must be callable or expose score_refusal")
        references[pair_id] = _normalize_layer_scores(
            values, expected_layers=layers, pair_id=pair_id
        )

    checkpoint: Mapping[str, Any] = {
        "format": REFERENCE_FORMAT,
        "version": REFERENCE_VERSION,
        "reference_refusal": {"by_pair_id": references},
        "pair_ids": list(references),
        "layers": layers,
        "metadata": {
            "source_collection": (
                str(Path(collection).expanduser().resolve())
                if isinstance(collection, (str, Path))
                else None
            ),
            "state": "X_H",
            "num_pairs": len(references),
            "score_space": "refusal_probability",
        },
    }
    validate_reference_checkpoint(checkpoint)
    if output_path is not None:
        atomic_torch_save(checkpoint, output_path)
    if per_pair_dir is not None:
        save_per_pair_references(checkpoint, per_pair_dir)
    return checkpoint


def _safe_pair_filename(pair_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pair_id).strip("._")
    if not safe:
        raise StateCollectionError(f"pair_id cannot form a safe filename: {pair_id!r}")
    return safe + ".pt"


def _atomic_reference_index(path: Path, payload: Mapping[str, Any]) -> None:
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


def save_per_pair_references(
    checkpoint: Mapping[str, Any], directory: str | Path
) -> Mapping[str, Path]:
    """Write one pair-keyed, batch-loader-compatible checkpoint per pair."""

    validate_reference_checkpoint(checkpoint)
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    references = checkpoint["reference_refusal"]["by_pair_id"]
    paths: "OrderedDict[str, Path]" = OrderedDict()
    filenames: set[str] = set()
    for pair_id, scores in references.items():
        filename = _safe_pair_filename(pair_id)
        if filename.casefold() in filenames:
            raise StateCollectionError("pair_id filenames collide after sanitization")
        filenames.add(filename.casefold())
        pair_checkpoint = {
            "format": REFERENCE_FORMAT,
            "version": REFERENCE_VERSION,
            "reference_refusal": {"by_pair_id": {pair_id: scores}},
            "pair_ids": [pair_id],
            "layers": list(checkpoint["layers"]),
            "metadata": {
                **dict(checkpoint.get("metadata", {})),
                "num_pairs": 1,
                "pair_id": pair_id,
            },
        }
        path = atomic_torch_save(pair_checkpoint, destination / filename)
        paths[pair_id] = path
    _atomic_reference_index(
        destination / "index.json",
        {
            "format": "pair-clean-refusal-reference-index",
            "version": 1,
            "pairs": [
                {"pair_id": pair_id, "path": path.name}
                for pair_id, path in paths.items()
            ],
        },
    )
    return paths


def attach_reference_paths(
    manifest_path: str | Path,
    reference_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> Any:
    """Write a manifest copy with leakage-safe per-row reference paths."""

    from data.build_safety_pairs import build_manifest
    from data.datasets import read_table, write_table

    source = Path(manifest_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    manifest = build_manifest(read_table(source))
    manifest_ids = set(manifest["pair_id"].astype(str))
    if set(reference_paths) != manifest_ids:
        missing = manifest_ids - set(reference_paths)
        extra = set(reference_paths) - manifest_ids
        raise StateCollectionError(
            "reference paths and manifest pair_ids differ "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    values = []
    for pair_id in manifest["pair_id"].astype(str):
        path = Path(reference_paths[pair_id]).expanduser().resolve()
        try:
            values.append(str(path.relative_to(destination.parent)))
        except ValueError:
            values.append(str(path))
    manifest["reference_refusal_path"] = values
    write_table(manifest, destination)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path, help="Collected state .pt")
    parser.add_argument("--probe", required=True, type=Path, help="Trained dual-probe .pt")
    parser.add_argument("--output", required=True, type=Path, help="Aggregate references .pt")
    parser.add_argument("--per-pair-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.manifest is None) != (args.manifest_output is None):
        raise ValueError("--manifest and --manifest-output must be used together")
    if args.manifest is not None and args.per_pair_dir is None:
        raise ValueError("manifest attachment requires --per-pair-dir")
    scorer = load_probe_scorer(args.probe)
    checkpoint = build_pair_references(
        args.states,
        scorer=scorer,
        output_path=args.output,
    )
    reference_paths = None
    if args.per_pair_dir is not None:
        reference_paths = save_per_pair_references(checkpoint, args.per_pair_dir)
    if args.manifest is not None:
        assert reference_paths is not None
        attach_reference_paths(args.manifest, reference_paths, args.manifest_output)
    print(
        f"Saved {checkpoint['metadata']['num_pairs']} clean X_H pair reference(s) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REFERENCE_FORMAT",
    "REFERENCE_VERSION",
    "attach_reference_paths",
    "build_pair_references",
    "load_probe_scorer",
    "save_per_pair_references",
    "validate_reference_checkpoint",
]
