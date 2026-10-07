"""Read-only adapter for frozen RQ1 probes, directions, and calibration states."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from rq2.artifacts import atomic_json, file_sha256


class RQ1BundleError(ValueError):
    """Raised when a frozen RQ1 artifact cannot safely seed RQ2."""


def _layer_map(value: Mapping[Any, Any], name: str) -> "OrderedDict[int, Any]":
    result: "OrderedDict[int, Any]" = OrderedDict()
    for key, item in value.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise RQ1BundleError(f"{name} layer key is not integer-like: {key!r}") from exc
        if layer in result:
            raise RQ1BundleError(f"duplicate normalized layer {layer} in {name}")
        result[layer] = item
    return OrderedDict(sorted(result.items()))


def _metadata_provenance(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RQ1BundleError("probe metadata is required")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RQ1BundleError("probe metadata.provenance is required")
    training = provenance.get("training_payload_metadata")
    merged = dict(training) if isinstance(training, Mapping) else {}
    merged.update({key: value for key, value in provenance.items() if key != "training_payload_metadata"})
    return merged


@dataclass(frozen=True)
class FrozenRQ1Bundle:
    probe_path: Path
    training_states_path: Path
    probe_sha256: str
    training_states_sha256: str
    scorer: Any
    hidden_sizes: Mapping[int, int]
    directions: Mapping[str, Mapping[int, Any]]
    unit_directions: Mapping[str, Mapping[int, Any]]
    centers: Mapping[str, Mapping[int, Any]]
    class_means: Mapping[str, Mapping[int, Mapping[str, Any]]]
    refusal_sigma: Mapping[int, float]
    refusal_sigma_population: str
    refusal_sigma_count: int
    model_fingerprint: Optional[str]
    provenance: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "format": "rq2-rq1-source-bundle",
            "version": 1,
            "probe_path": str(self.probe_path),
            "probe_sha256": self.probe_sha256,
            "training_states_path": str(self.training_states_path),
            "training_states_sha256": self.training_states_sha256,
            "layers": list(self.hidden_sizes),
            "hidden_sizes": {str(key): value for key, value in self.hidden_sizes.items()},
            "refusal_sigma": {str(key): value for key, value in self.refusal_sigma.items()},
            "refusal_sigma_population": self.refusal_sigma_population,
            "refusal_sigma_count": self.refusal_sigma_count,
            "model_fingerprint": self.model_fingerprint,
            "pooling": self.provenance.get("pooling"),
            "token_span": self.provenance.get("token_span"),
            "sequence_has_embedding": self.provenance.get("sequence_has_embedding"),
            "direction_definition": "positive_class_mean-minus-negative_class_mean",
        }


def load_frozen_rq1_bundle(
    probe_path: str | Path,
    training_states_path: str | Path,
    *,
    expected_probe_sha256: Optional[str] = None,
    expected_training_sha256: Optional[str] = None,
    expected_model_fingerprint: Optional[str] = None,
    expected_layers: Optional[int] = None,
) -> FrozenRQ1Bundle:
    import torch

    from core.safety_state import DualSafetyStateScorer
    from experiments.collect_safety_states import safe_torch_load

    probe_file = Path(probe_path).expanduser().resolve()
    states_file = Path(training_states_path).expanduser().resolve()
    probe_digest = file_sha256(probe_file)
    states_digest = file_sha256(states_file)
    if expected_probe_sha256 and probe_digest != expected_probe_sha256:
        raise RQ1BundleError("probe SHA-256 does not match the frozen configuration")
    if expected_training_sha256 and states_digest != expected_training_sha256:
        raise RQ1BundleError("training-state SHA-256 does not match the frozen configuration")
    checkpoint = safe_torch_load(probe_file)
    if checkpoint.get("format") != "dual-safety-state-layerwise-linear-probes" or checkpoint.get("version") != 2:
        raise RQ1BundleError("RQ2 requires the frozen v2 dual safety-state probe")
    hidden_raw = checkpoint.get("hidden_sizes")
    state_dict = checkpoint.get("state_dict")
    raw_directions = checkpoint.get("directions")
    raw_means = checkpoint.get("class_means")
    if not isinstance(hidden_raw, Mapping) or not isinstance(state_dict, Mapping):
        raise RQ1BundleError("probe hidden_sizes/state_dict are malformed")
    if not isinstance(raw_directions, Mapping) or not isinstance(raw_means, Mapping):
        raise RQ1BundleError("probe directions/class_means are required")
    hidden_sizes = _layer_map(hidden_raw, "hidden_sizes")
    if not hidden_sizes or tuple(hidden_sizes) != tuple(range(len(hidden_sizes))):
        raise RQ1BundleError(
            "probe must contain a non-empty contiguous decoder-layer range starting at 0"
        )
    if expected_layers is not None:
        if (
            isinstance(expected_layers, bool)
            or not isinstance(expected_layers, int)
            or expected_layers < 1
        ):
            raise ValueError("expected_layers must be a positive integer or None")
        if len(hidden_sizes) != expected_layers:
            raise RQ1BundleError(
                f"probe contains {len(hidden_sizes)} layers; expected {expected_layers}"
            )
    scorer = DualSafetyStateScorer(hidden_size=hidden_sizes, trainable=False)
    try:
        scorer.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise RQ1BundleError("probe state_dict is incompatible with hidden_sizes") from exc
    scorer.eval()

    directions: dict[str, Mapping[int, Any]] = {}
    unit_directions: dict[str, Mapping[int, Any]] = {}
    centers: dict[str, Mapping[int, Any]] = {}
    class_means: dict[str, Mapping[int, Mapping[str, Any]]] = {}
    for state in ("harmfulness", "refusal"):
        state_directions = raw_directions.get(state)
        state_means = raw_means.get(state)
        if not isinstance(state_directions, Mapping) or not isinstance(state_means, Mapping):
            raise RQ1BundleError(f"missing {state} directions/class means")
        by_direction = _layer_map(state_directions, f"directions.{state}")
        by_means = _layer_map(state_means, f"class_means.{state}")
        normalized: "OrderedDict[int, Any]" = OrderedDict()
        state_centers: "OrderedDict[int, Any]" = OrderedDict()
        verified_means: "OrderedDict[int, Mapping[str, Any]]" = OrderedDict()
        for layer, width in hidden_sizes.items():
            direction = by_direction.get(layer)
            means = by_means.get(layer)
            if not isinstance(direction, torch.Tensor) or tuple(direction.shape) != (width,):
                raise RQ1BundleError(f"direction {state}/{layer} has wrong shape")
            if not isinstance(means, Mapping):
                raise RQ1BundleError(f"class means {state}/{layer} are malformed")
            negative = means.get("negative")
            positive = means.get("positive")
            if not all(isinstance(item, torch.Tensor) and tuple(item.shape) == (width,) for item in (negative, positive)):
                raise RQ1BundleError(f"class means {state}/{layer} have wrong shape")
            vector = direction.detach().cpu().float().contiguous()
            negative = negative.detach().cpu().float().contiguous()
            positive = positive.detach().cpu().float().contiguous()
            if not torch.allclose(vector, positive - negative, rtol=1e-5, atol=1e-6):
                raise RQ1BundleError(f"direction {state}/{layer} is not positive-negative")
            norm = torch.linalg.vector_norm(vector)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
                raise RQ1BundleError(f"direction {state}/{layer} has zero/non-finite norm")
            normalized[layer] = (vector / norm).contiguous()
            state_centers[layer] = ((positive + negative) / 2.0).contiguous()
            verified_means[layer] = {"negative": negative, "positive": positive}
        directions[state] = by_direction
        unit_directions[state] = normalized
        centers[state] = state_centers
        class_means[state] = verified_means

    provenance = _metadata_provenance(checkpoint)
    if provenance.get("pooling") != "mean" or provenance.get("token_span") != "audio":
        raise RQ1BundleError("RQ2 requires RQ1 mean pooling over the audio token span")
    if provenance.get("sequence_has_embedding") is not True:
        raise RQ1BundleError("RQ2 requires the RQ1 embedding-offset layer convention")
    model_fingerprint = provenance.get("model_fingerprint")
    model_provenance = provenance.get("model_provenance")
    if model_fingerprint is None and isinstance(model_provenance, Mapping):
        model_fingerprint = model_provenance.get("model_fingerprint")
    if expected_model_fingerprint and model_fingerprint != expected_model_fingerprint:
        raise RQ1BundleError("probe model fingerprint differs from the RQ2 model")

    training = safe_torch_load(states_file)
    source_digest = provenance.get("source_payload_sha256")
    if source_digest and source_digest != states_digest:
        raise RQ1BundleError("probe provenance does not bind the supplied training states")
    states = training.get("hidden_states")
    refusal_labels = training.get("refusal_labels")
    if not isinstance(states, Mapping) or not isinstance(refusal_labels, torch.Tensor):
        raise RQ1BundleError("training states lack hidden_states/refusal_labels")
    state_layers = _layer_map(states, "training.hidden_states")
    state_names = training.get("states")
    if not isinstance(state_names, (list, tuple)) or len(state_names) != refusal_labels.numel():
        raise RQ1BundleError("training states must identify every calibration row")
    labels = refusal_labels.detach().cpu().reshape(-1)
    mask = torch.tensor(
        [bool(float(label) >= 0.5) and str(state) == "X_H" for label, state in zip(labels, state_names)],
        dtype=torch.bool,
    )
    calibration_count = int(mask.sum().item())
    if calibration_count < 2:
        raise RQ1BundleError("refusal sigma calibration needs at least two refusal-positive X_H rows")
    refusal_sigma: "OrderedDict[int, float]" = OrderedDict()
    for layer, width in hidden_sizes.items():
        matrix = state_layers.get(layer)
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.shape[1] != width:
            raise RQ1BundleError(f"training hidden_states[{layer}] has wrong shape")
        if matrix.shape[0] != len(state_names):
            raise RQ1BundleError(f"training hidden_states[{layer}] row count mismatch")
        unit = unit_directions["refusal"][layer]
        center = centers["refusal"][layer]
        projection = (matrix.detach().cpu().float()[mask] - center) @ unit
        sigma = float(torch.std(projection, unbiased=True).item())
        if not math_is_finite_positive(sigma):
            raise RQ1BundleError(f"refusal projection sigma is invalid at layer {layer}")
        refusal_sigma[layer] = sigma

    return FrozenRQ1Bundle(
        probe_path=probe_file,
        training_states_path=states_file,
        probe_sha256=probe_digest,
        training_states_sha256=states_digest,
        scorer=scorer,
        hidden_sizes=hidden_sizes,
        directions=directions,
        unit_directions=unit_directions,
        centers=centers,
        class_means=class_means,
        refusal_sigma=refusal_sigma,
        refusal_sigma_population="refusal-positive X_H rows from frozen RQ1 probe training states",
        refusal_sigma_count=calibration_count,
        model_fingerprint=None if model_fingerprint is None else str(model_fingerprint),
        provenance=provenance,
    )


def math_is_finite_positive(value: float) -> bool:
    import math

    return math.isfinite(value) and value > 0.0


def write_bundle_summary(bundle: FrozenRQ1Bundle, output_path: str | Path) -> Path:
    return atomic_json(output_path, bundle.summary())


__all__ = [
    "FrozenRQ1Bundle",
    "RQ1BundleError",
    "load_frozen_rq1_bundle",
    "write_bundle_summary",
]
