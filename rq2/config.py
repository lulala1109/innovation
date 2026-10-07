"""Strict, domain-specific configuration for the Stage-2/RQ2 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rq2.artifacts import canonical_sha256
from rq2.formal_statistics import FORMAL_STATISTICS_VERSION, family_policy
from rq2.pilot import PILOT_GATE_VERSION
from rq2.preregistration import (
    RQ2PreregistrationError,
    assert_config_matches_preregistration,
    load_preregistration,
)


SCHEMA_VERSION = 4
STAGE_ORDER = (
    "sources",
    "trajectory",
    "trajectory_behavior_generate",
    "trajectory_behavior_judge",
    "events",
    "state_index",
    "layer_map",
    "identity",
    "oracle_generate",
    "oracle_judge",
    "oracle_analyze",
    "mechanism_generate",
    "mechanism_judge",
    "mechanism_analyze",
    "subspace_generate",
    "subspace_judge",
    "subspace_analyze",
    "protocol_lock",
    "formal_generate",
    "formal_judge",
    "formal_analyze",
    "event_generate",
    "event_judge",
    "event_analyze",
    "report",
)
STAGE_RESOURCES = {
    "sources": ("cpu",),
    "trajectory": ("gpu",),
    "trajectory_behavior_generate": ("gpu",),
    "trajectory_behavior_judge": ("api",),
    "events": ("cpu",),
    "state_index": ("cpu",),
    "layer_map": ("gpu",),
    "identity": ("gpu",),
    "oracle_generate": ("gpu",),
    "oracle_judge": ("api",),
    "oracle_analyze": ("cpu",),
    "mechanism_generate": ("gpu",),
    "mechanism_judge": ("api",),
    "mechanism_analyze": ("cpu",),
    "subspace_generate": ("gpu",),
    "subspace_judge": ("api",),
    "subspace_analyze": ("cpu",),
    "protocol_lock": ("cpu",),
    "formal_generate": ("gpu",),
    "formal_judge": ("api",),
    "formal_analyze": ("cpu",),
    "event_generate": ("gpu",),
    "event_judge": ("api",),
    "event_analyze": ("cpu",),
    "report": ("cpu",),
}
SMOKE_STAGE_ORDER = STAGE_ORDER[: STAGE_ORDER.index("mechanism_analyze") + 1]
DEV_SCREEN_STAGE_ORDER = STAGE_ORDER[: STAGE_ORDER.index("trajectory_behavior_judge") + 1]
STAGE_DEPENDENCIES = {
    stage: (() if index == 0 else (STAGE_ORDER[index - 1],))
    for index, stage in enumerate(STAGE_ORDER)
}


class RQ2ConfigError(ValueError):
    """Raised when an RQ2 configuration violates its closed schema."""


@dataclass(frozen=True)
class RQ2Config:
    path: Path
    project_root: Path
    name: str
    template: bool
    execution_enabled: bool
    frozen: bool
    output_root: Path
    manifest: Path
    preregistration_path: Path
    preregistration_sha256: str
    statistical_preregistration_path: Optional[Path]
    statistical_preregistration_sha256: Optional[str]
    statistical_preregistration_record: Optional[Mapping[str, Any]]
    preregistration_record: Mapping[str, Any]
    trajectory: Mapping[str, Any]
    model: Mapping[str, Any]
    rq1_sources: Mapping[str, Any]
    sampling: Mapping[str, Any]
    pilot: Mapping[str, Any]
    formal: Mapping[str, Any]
    smoke: Mapping[str, Any]
    dev_screen: Mapping[str, Any]
    judge: Mapping[str, Any]
    statistics: Mapping[str, Any]
    raw: Mapping[str, Any]
    fingerprint: str

    def stage_path(self, stage: str) -> Path:
        if stage not in STAGE_ORDER:
            raise RQ2ConfigError(f"unknown RQ2 stage: {stage}")
        names = {
            "sources": "provenance/rq1_sources.json",
            "trajectory": "trajectory/summary.json",
            "trajectory_behavior_generate": "trajectory_behavior/responses.jsonl",
            "trajectory_behavior_judge": "trajectory_behavior/labels.jsonl",
            "events": "state_index/behavior_events.json",
            "state_index": "state_index/index.json",
            "layer_map": "provenance/layer_map.json",
            "identity": "identity_tests.json",
            "oracle_generate": "oracle_pilot/responses.jsonl",
            "oracle_judge": "oracle_pilot/labels.jsonl",
            "oracle_analyze": "oracle_pilot/analysis.json",
            "mechanism_generate": "mechanism_pilot/responses.jsonl",
            "mechanism_judge": "mechanism_pilot/labels.jsonl",
            "mechanism_analyze": "mechanism_pilot/analysis.json",
            "subspace_generate": "subspace_pilot/responses.jsonl",
            "subspace_judge": "subspace_pilot/labels.jsonl",
            "subspace_analyze": "subspace_pilot/analysis.json",
            "protocol_lock": "protocol_lock.json",
            "formal_generate": "formal/responses.jsonl",
            "formal_judge": "formal/labels.jsonl",
            "formal_analyze": "formal/rq2_summary.json",
            "event_generate": "event/responses.jsonl",
            "event_judge": "event/labels.jsonl",
            "event_analyze": "event/event_analysis.json",
            "report": "rq2_report.md",
        }
        return self.output_root / names[stage]


_TOP_LEVEL = {
    "schema_version",
    "name",
    "template",
    "execution_enabled",
    "frozen",
    "output_root",
    "manifest",
    "preregistration",
    "statistical_preregistration",
    "trajectory",
    "model",
    "rq1_sources",
    "sampling",
    "pilot",
    "formal",
    "judge",
    "statistics",
    "smoke",
    "dev_screen",
}
_OPTIONAL_TOP_LEVEL = {"smoke", "dev_screen", "statistical_preregistration"}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RQ2ConfigError(f"{name} must be an object")
    return dict(value)


def _text(value: Any, name: str) -> str:
    if value is None or not str(value).strip():
        raise RQ2ConfigError(f"{name} must be non-blank")
    return str(value).strip()


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RQ2ConfigError(f"{name} must be boolean")
    return value


def _resolve(base: Path, value: Any, name: str) -> Path:
    path = Path(_text(value, name)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _integer_sequence(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RQ2ConfigError(f"{name} must be an integer list")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise RQ2ConfigError(f"{name} must contain only integers")
    if len(set(result)) != len(result):
        raise RQ2ConfigError(f"{name} contains duplicates")
    return result


def _closed_fields(value: Mapping[str, Any], name: str, allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RQ2ConfigError(f"unknown {name} fields: {unknown}")


def load_rq2_config(path: str | Path) -> RQ2Config:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RQ2ConfigError(f"invalid JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise RQ2ConfigError("RQ2 config must be a JSON object")
    unknown = sorted(set(raw) - _TOP_LEVEL)
    missing = sorted((_TOP_LEVEL - _OPTIONAL_TOP_LEVEL) - set(raw))
    if unknown or missing:
        raise RQ2ConfigError(f"config fields changed; unknown={unknown}, missing={missing}")
    if raw.get("schema_version") not in (3, SCHEMA_VERSION):
        raise RQ2ConfigError(f"schema_version must be 3 or {SCHEMA_VERSION}")

    project_root = config_path.parent.parent.resolve()
    name = _text(raw["name"], "name")
    template = _bool(raw["template"], "template")
    execution_enabled = _bool(raw["execution_enabled"], "execution_enabled")
    frozen = _bool(raw["frozen"], "frozen")
    output_root = _resolve(project_root, raw["output_root"], "output_root")
    manifest = _resolve(project_root, raw["manifest"], "manifest")
    preregistration = _mapping(raw["preregistration"], "preregistration")
    _closed_fields(preregistration, "preregistration", {"path", "sha256"})
    preregistration_path = _resolve(
        project_root, preregistration.get("path"), "preregistration.path"
    )
    preregistration_sha256 = _text(
        preregistration.get("sha256"), "preregistration.sha256"
    )
    try:
        preregistration_record = load_preregistration(
            preregistration_path,
            expected_sha256=preregistration_sha256,
        )
    except RQ2PreregistrationError as exc:
        raise RQ2ConfigError(str(exc)) from exc
    v2 = raw["schema_version"] == 4
    if preregistration_record["version"] != (2 if v2 else 1):
        raise RQ2ConfigError("schema v3 requires preregistration v1; schema v4 requires v2")
    statistical_preregistration_path: Optional[Path] = None
    statistical_preregistration_sha256: Optional[str] = None
    statistical_preregistration_record: Optional[Mapping[str, Any]] = None
    if v2 and execution_enabled and "statistical_preregistration" not in raw:
        raise RQ2ConfigError("executable schema v4 requires statistical_preregistration")
    if "statistical_preregistration" in raw:
        if not v2:
            raise RQ2ConfigError("statistical_preregistration is only valid for schema v4")
        reference = _mapping(raw["statistical_preregistration"], "statistical_preregistration")
        _closed_fields(reference, "statistical_preregistration", {"path", "sha256"})
        statistical_preregistration_path = _resolve(
            project_root, reference.get("path"), "statistical_preregistration.path"
        )
        statistical_preregistration_sha256 = _text(
            reference.get("sha256"), "statistical_preregistration.sha256"
        )
        if (
            not statistical_preregistration_path.is_file()
            or hashlib.sha256(statistical_preregistration_path.read_bytes()).hexdigest()
            != statistical_preregistration_sha256
        ):
            raise RQ2ConfigError("statistical preregistration is missing or changed")
        try:
            statistical_preregistration_record = _mapping(
                json.loads(statistical_preregistration_path.read_text(encoding="utf-8")),
                "statistical_preregistration record",
            )
        except json.JSONDecodeError as exc:
            raise RQ2ConfigError("statistical preregistration is invalid JSON") from exc
        parent = statistical_preregistration_record.get("parent_layer_dose_record")
        if (
            statistical_preregistration_record.get("format") != "rq2-v2-statistical-preregistration-addendum"
            or statistical_preregistration_record.get("version") != 1
            or statistical_preregistration_record.get("event_artifact_version") != 3
            or statistical_preregistration_record.get("pilot_gate_version") != PILOT_GATE_VERSION
            or statistical_preregistration_record.get("formal_statistics_version") != FORMAL_STATISTICS_VERSION
            or not isinstance(parent, Mapping)
            or parent.get("sha256") != preregistration_sha256
        ):
            raise RQ2ConfigError("statistical preregistration parent or version is invalid")
    distant_field = "distant_control_layers" if v2 else "depth_control_layers"
    protected = (
        project_root / "outputs" / "stage1",
        project_root / "dataset" / "processed" / "stage1",
    )
    if any(_is_within(output_root, root.resolve()) for root in protected):
        raise RQ2ConfigError("RQ2 output_root cannot be inside a frozen RQ1 namespace")
    expected_root = (project_root / "outputs" / "stage2_rq2").resolve()
    if not _is_within(output_root, expected_root):
        raise RQ2ConfigError("output_root must be inside outputs/stage2_rq2")

    trajectory = _mapping(raw["trajectory"], "trajectory")
    model = _mapping(raw["model"], "model")
    sources = _mapping(raw["rq1_sources"], "rq1_sources")
    sampling = _mapping(raw["sampling"], "sampling")
    pilot = _mapping(raw["pilot"], "pilot")
    formal = _mapping(raw["formal"], "formal")
    judge = _mapping(raw["judge"], "judge")
    statistics = _mapping(raw["statistics"], "statistics")
    smoke = _mapping(raw["smoke"], "smoke") if "smoke" in raw else {}
    dev_screen = _mapping(raw["dev_screen"], "dev_screen") if "dev_screen" in raw else {}
    if smoke and dev_screen:
        raise RQ2ConfigError("smoke and dev_screen modes are mutually exclusive")
    if _is_within(output_root, (expected_root / "dev_screen").resolve()) and not dev_screen:
        raise RQ2ConfigError("the dev_screen output namespace requires dev_screen mode")

    _closed_fields(trajectory, "trajectory", {
        "mode", "eps", "alpha", "steps", "kappa", "seed", "determinism",
        "refusal_weakening_delta", "capture_profiles",
    })
    _closed_fields(model, "model", {
        "name", "model_id", "device", "dtype", "max_tokens", "temperature",
        "do_sample", "output_selector", "layer_map_atol", "layer_map_rtol",
        "identity_atol",
    })
    _closed_fields(sources, "rq1_sources", {
        "probe_path", "training_states_path", "probe_sha256",
        "training_states_sha256", "rq1_manifests", "layer_slopes_path",
    })
    _closed_fields(sampling, "sampling", {"fixed_steps", "event_offsets"})
    pilot_fields = {
        "candidate_layers", "neighbor_layers", distant_field,
        "restoration_doses", "suppression_doses", "random_replicates", "seed",
        "allow_rq1_overlap", "oracle_min_effect", "mechanism_min_effect",
        "subspace_enabled", "subspace_path",
        "subspace_operator", "subspace_protocol_path", "subspace_protocol_sha256",
    }
    if v2:
        pilot_fields.update({"primary_restoration_dose", "primary_suppression_dose", "sensitivity_doses"})
    _closed_fields(pilot, "pilot", pilot_fields)
    _closed_fields(formal, "formal", {"layers", "minimum_pairs"})
    _closed_fields(smoke, "smoke", {
        "enabled", "source_manifest", "source_manifest_sha256",
        "smoke_pair_ids", "smoke_layers",
    })
    _closed_fields(dev_screen, "dev_screen", {
        "enabled", "source_manifest", "source_manifest_sha256",
        "dev_pair_ids", "screening_protocol", "screening_protocol_sha256",
    })
    _closed_fields(judge, "judge", {
        "protocol", "provider", "model", "base_url", "schema_retries",
        "success_threshold", "min_convincingness", "min_specificity",
        "keep_unknown", "retry_unknown",
    })
    _closed_fields(statistics, "statistics", {
        "bootstrap_unit", "bootstrap_replicates", "confidence", "seed", "top_k",
        "fdr_alpha", "minimum_sign_consistency",
    })

    mode = trajectory.get("mode")
    if mode not in {"existing", "generate"}:
        raise RQ2ConfigError("trajectory.mode must be existing or generate")
    if isinstance(trajectory.get("steps"), bool) or trajectory.get("steps") != 100:
        raise RQ2ConfigError("RQ2 trajectories must use exactly 100 PGD steps")
    for field in ("eps", "alpha", "kappa"):
        value = trajectory.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RQ2ConfigError(f"trajectory.{field} must be finite and positive")
    if float(trajectory["alpha"]) > float(trajectory["eps"]):
        raise RQ2ConfigError("trajectory.alpha cannot exceed trajectory.eps")
    if isinstance(trajectory.get("seed"), bool) or not isinstance(trajectory.get("seed"), int):
        raise RQ2ConfigError("trajectory.seed must be an integer")
    if trajectory.get("determinism") not in {"off", "warn", "error"}:
        raise RQ2ConfigError("trajectory.determinism must be off, warn, or error")
    weakening_raw = trajectory.get("refusal_weakening_delta", 0.2)
    if isinstance(weakening_raw, bool) or not isinstance(weakening_raw, (int, float)):
        raise RQ2ConfigError(
            "trajectory.refusal_weakening_delta must be numeric"
        )
    weakening = float(weakening_raw)
    if not 0.0 <= weakening <= 1.0:
        raise RQ2ConfigError("trajectory.refusal_weakening_delta must be within [0,1]")
    _bool(trajectory.get("capture_profiles"), "trajectory.capture_profiles")
    do_sample = _bool(model.get("do_sample"), "model.do_sample")
    if do_sample:
        raise RQ2ConfigError("RQ2 requires deterministic generation (model.do_sample=false)")
    if not all(model.get(field) not in (None, "") for field in ("name", "model_id", "dtype")):
        raise RQ2ConfigError("model.name, model_id, and dtype are required")
    if model.get("dtype") not in {"float32", "float16", "bfloat16"}:
        raise RQ2ConfigError("model.dtype is unsupported")
    if (
        isinstance(model.get("max_tokens"), bool)
        or not isinstance(model.get("max_tokens"), int)
        or model["max_tokens"] < 1
    ):
        raise RQ2ConfigError("model.max_tokens must be a positive integer")
    if (
        isinstance(model.get("temperature"), bool)
        or not isinstance(model.get("temperature"), (int, float))
        or not math.isfinite(float(model["temperature"]))
        or float(model["temperature"]) < 0.0
    ):
        raise RQ2ConfigError("model.temperature must be finite and non-negative")
    for field in ("layer_map_atol", "layer_map_rtol", "identity_atol"):
        value = model.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RQ2ConfigError(f"model.{field} must be finite and positive")
    if (
        isinstance(model.get("output_selector"), bool)
        or (
            model.get("output_selector") is not None
            and not isinstance(model.get("output_selector"), (int, str))
        )
    ):
        raise RQ2ConfigError("model.output_selector must be null, integer, or string")

    for field in ("probe_path", "training_states_path", "probe_sha256", "training_states_sha256"):
        _text(sources.get(field), f"rq1_sources.{field}")
    _text(sources.get("layer_slopes_path"), "rq1_sources.layer_slopes_path")
    for field in ("probe_sha256", "training_states_sha256"):
        digest = str(sources[field])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RQ2ConfigError(f"rq1_sources.{field} must be lowercase SHA-256")
    manifests = sources.get("rq1_manifests")
    if (
        not isinstance(manifests, Sequence)
        or isinstance(manifests, (str, bytes))
        or not manifests
        or any(not isinstance(value, str) or not value.strip() for value in manifests)
    ):
        raise RQ2ConfigError("rq1_sources.rq1_manifests must be a non-empty path list")

    fixed_steps = _integer_sequence(sampling.get("fixed_steps"), "sampling.fixed_steps")
    if len(fixed_steps) < 2:
        raise RQ2ConfigError("sampling.fixed_steps must contain at least two states")
    if tuple(sorted(fixed_steps)) != fixed_steps:
        raise RQ2ConfigError("sampling.fixed_steps must be strictly increasing")
    if any(step < 0 or step > int(trajectory["steps"]) for step in fixed_steps):
        raise RQ2ConfigError(
            "sampling.fixed_steps must stay within the configured trajectory"
        )
    event_offsets = _integer_sequence(sampling.get("event_offsets"), "sampling.event_offsets")
    if event_offsets != (-2, -1, 0, 1, 3):
        raise RQ2ConfigError("sampling.event_offsets must be exactly [-2, -1, 0, 1, 3]")
    layer_sets: dict[str, set[int]] = {}
    for field in ("candidate_layers", "neighbor_layers", distant_field):
        layers = _integer_sequence(pilot.get(field, ()), f"pilot.{field}")
        if any(layer < 0 for layer in layers):
            raise RQ2ConfigError(f"pilot.{field} cannot contain negative layers")
        layer_sets[field] = set(layers)
    if not pilot.get("candidate_layers"):
        raise RQ2ConfigError("pilot.candidate_layers cannot be empty")
    if layer_sets["candidate_layers"].intersection(layer_sets[distant_field]):
        raise RQ2ConfigError("candidate and distant-reference layers must be disjoint")
    if v2 and (
        layer_sets["neighbor_layers"].intersection(layer_sets["candidate_layers"])
        or layer_sets["neighbor_layers"].intersection(layer_sets[distant_field])
    ):
        raise RQ2ConfigError("v2 layer role sets must be disjoint")
    for field in ("restoration_doses", "suppression_doses"):
        values = pilot.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise RQ2ConfigError(f"pilot.{field} must be a non-empty numeric list")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in values
        ):
            raise RQ2ConfigError(f"pilot.{field} must contain finite positive numbers")
    if v2:
        for field in ("primary_restoration_dose", "primary_suppression_dose"):
            value = pilot.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 1.0:
                raise RQ2ConfigError(f"pilot.{field} must be explicitly fixed at 1.0")
        if pilot.get("sensitivity_doses") != [0.5]:
            raise RQ2ConfigError("v2 pilot.sensitivity_doses must be [0.5]")
        for field in ("restoration_doses", "suppression_doses"):
            if list(pilot[field]) != [1.0, 0.5]:
                raise RQ2ConfigError(f"v2 pilot.{field} must be [1.0, 0.5]; 1.5 is excluded")
    if (
        isinstance(pilot.get("random_replicates"), bool)
        or not isinstance(pilot.get("random_replicates"), int)
        or pilot["random_replicates"] < 1
    ):
        raise RQ2ConfigError("pilot.random_replicates must be a positive integer")
    if isinstance(pilot.get("seed"), bool) or not isinstance(pilot.get("seed"), int):
        raise RQ2ConfigError("pilot.seed must be an integer")
    _bool(pilot.get("allow_rq1_overlap"), "pilot.allow_rq1_overlap")
    subspace_enabled = _bool(pilot.get("subspace_enabled"), "pilot.subspace_enabled")
    if subspace_enabled:
        if not str(pilot.get("subspace_path") or "").strip():
            raise RQ2ConfigError("pilot.subspace_path is required when subspace is enabled")
        if "subspace" not in name.casefold():
            raise RQ2ConfigError("conditional subspace work requires a separately named new run")
        if pilot.get("subspace_operator") != "pooled_gap_shift_v1":
            raise RQ2ConfigError("current code only implements the explicit pooled_gap_shift_v1 operator")
        protocol_path = _resolve(project_root, pilot.get("subspace_protocol_path"), "pilot.subspace_protocol_path")
        expected_protocol_sha = _text(pilot.get("subspace_protocol_sha256"), "pilot.subspace_protocol_sha256")
        if not protocol_path.is_file() or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != expected_protocol_sha:
            raise RQ2ConfigError("subspace calibration/operator protocol is missing or changed")
    for field in ("oracle_min_effect", "mechanism_min_effect"):
        value = pilot.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RQ2ConfigError(f"pilot.{field} must be finite and non-negative")
    if formal.get("layers") != "all":
        raise RQ2ConfigError(
            "formal.layers must be 'all' to scan every frozen decoder layer"
        )
    if (
        isinstance(formal.get("minimum_pairs"), bool)
        or not isinstance(formal.get("minimum_pairs"), int)
        or formal["minimum_pairs"] < 20
    ):
        raise RQ2ConfigError("formal.minimum_pairs must be an integer of at least 20")
    if judge.get("protocol") != "strongreject-canonical-balanced":
        raise RQ2ConfigError("judge.protocol must be strongreject-canonical-balanced")
    _text(judge.get("provider"), "judge.provider")
    _text(judge.get("model"), "judge.model")
    if (
        isinstance(judge.get("schema_retries"), bool)
        or not isinstance(judge.get("schema_retries"), int)
        or judge["schema_retries"] < 0
    ):
        raise RQ2ConfigError("judge.schema_retries must be a non-negative integer")
    success_threshold = judge.get("success_threshold")
    if (
        isinstance(success_threshold, bool)
        or not isinstance(success_threshold, (int, float))
        or not math.isfinite(float(success_threshold))
        or not 0.0 <= float(success_threshold) <= 1.0
    ):
        raise RQ2ConfigError("judge.success_threshold must be within [0,1]")
    for field in ("min_convincingness", "min_specificity"):
        value = judge.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise RQ2ConfigError(f"judge.{field} must be an integer within [1,5]")
    _bool(judge.get("keep_unknown"), "judge.keep_unknown")
    _bool(judge.get("retry_unknown"), "judge.retry_unknown")
    if statistics.get("bootstrap_unit") != "pair_id":
        raise RQ2ConfigError("statistics.bootstrap_unit must be pair_id")
    if (
        isinstance(statistics.get("bootstrap_replicates"), bool)
        or not isinstance(statistics.get("bootstrap_replicates"), int)
        or statistics["bootstrap_replicates"] < 1
    ):
        raise RQ2ConfigError("statistics.bootstrap_replicates must be positive")
    for field in ("confidence", "fdr_alpha"):
        value = statistics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise RQ2ConfigError(f"statistics.{field} must be within (0,1)")
    sign_consistency = statistics.get("minimum_sign_consistency")
    if (
        isinstance(sign_consistency, bool)
        or not isinstance(sign_consistency, (int, float))
        or not math.isfinite(float(sign_consistency))
        or not 0.0 < float(sign_consistency) <= 1.0
    ):
        raise RQ2ConfigError(
            "statistics.minimum_sign_consistency must be within (0,1]"
        )
    if (
        isinstance(statistics.get("top_k"), bool)
        or not isinstance(statistics.get("top_k"), int)
        or statistics["top_k"] < 1
    ):
        raise RQ2ConfigError("statistics.top_k must be positive")
    if isinstance(statistics.get("seed"), bool) or not isinstance(statistics.get("seed"), int):
        raise RQ2ConfigError("statistics.seed must be an integer")

    if statistical_preregistration_record is not None:
        expected_design = {
            "candidate_layers": list(pilot["candidate_layers"]),
            "neighbor_layers": list(pilot["neighbor_layers"]),
            "distant_control_layers": list(pilot["distant_control_layers"]),
            "formal_layers": "all_0_to_27",
            "fixed_steps": list(sampling["fixed_steps"]),
            "event_offsets": list(sampling["event_offsets"]),
            "primary_restoration_dose": float(pilot["primary_restoration_dose"]),
            "primary_suppression_dose": float(pilot["primary_suppression_dose"]),
            "sensitivity_doses": list(pilot["sensitivity_doses"]),
            "random_replicates": int(pilot["random_replicates"]),
            "minimum_valid_pairs": int(formal["minimum_pairs"]),
            "bootstrap_unit": statistics["bootstrap_unit"],
            "bootstrap_replicates": int(statistics["bootstrap_replicates"]),
            "confidence": float(statistics["confidence"]),
            "fdr_alpha": float(statistics["fdr_alpha"]),
            "minimum_sign_consistency": float(statistics["minimum_sign_consistency"]),
            "top_k": int(statistics["top_k"]),
        }
        if statistical_preregistration_record.get("design") != expected_design:
            raise RQ2ConfigError("v2 configuration differs from statistical preregistration design")
        policy = family_policy(
            formal_layers=range(28), candidate_layers=pilot["candidate_layers"],
            neighbor_layers=pilot["neighbor_layers"],
            distant_layers=pilot["distant_control_layers"],
            fixed_steps=sampling["fixed_steps"],
        )
        expected_families = {
            **policy["fdr_families"], **policy["event_paired_offset_families"]
        }
        if statistical_preregistration_record.get("formal_family_policy") != expected_families:
            raise RQ2ConfigError("v2 statistical preregistration family slots have changed")

    if smoke:
        if not v2 or _bool(smoke.get("enabled"), "smoke.enabled") is not True:
            raise RQ2ConfigError("smoke requires schema v4 and smoke.enabled=true")
        if "smoke" not in name.casefold():
            raise RQ2ConfigError("smoke run name must contain smoke")
        smoke_root = (expected_root / "smoke").resolve()
        if not _is_within(output_root, smoke_root) or output_root == smoke_root:
            raise RQ2ConfigError("smoke output_root must be a dedicated child of outputs/stage2_rq2/smoke")
        if subspace_enabled:
            raise RQ2ConfigError("smoke cannot enable the conditional subspace branch")
        smoke_ids = smoke.get("smoke_pair_ids")
        if not isinstance(smoke_ids, list) or not 2 <= len(smoke_ids) <= 3:
            raise RQ2ConfigError("smoke.smoke_pair_ids must contain 2–3 pair IDs")
        if any(not isinstance(item, str) or not item.strip() for item in smoke_ids) or len(set(smoke_ids)) != len(smoke_ids):
            raise RQ2ConfigError("smoke.smoke_pair_ids must be distinct nonblank strings")
        smoke_layers = _integer_sequence(smoke.get("smoke_layers"), "smoke.smoke_layers")
        if not 2 <= len(smoke_layers) <= 3 or not set(smoke_layers).issubset(layer_sets["candidate_layers"]):
            raise RQ2ConfigError("smoke.smoke_layers must contain 2–3 preregistered candidate layers")
        source_manifest = _resolve(project_root, smoke.get("source_manifest"), "smoke.source_manifest")
        if manifest == source_manifest:
            raise RQ2ConfigError("smoke manifest must be separate from the frozen formal manifest")
        source_digest = _text(smoke.get("source_manifest_sha256"), "smoke.source_manifest_sha256")
        if len(source_digest) != 64 or any(char not in "0123456789abcdef" for char in source_digest):
            raise RQ2ConfigError("smoke source manifest SHA must be lowercase SHA-256")
        if not source_manifest.is_file() or hashlib.sha256(source_manifest.read_bytes()).hexdigest() != source_digest:
            raise RQ2ConfigError("smoke source manifest is missing or changed")
        if not manifest.is_file():
            raise RQ2ConfigError("smoke subset manifest is missing")
        source_rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        subset_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_rows = sorted(
            (row for row in source_rows if row.get("rq2_role") == "rq2_dev"),
            key=lambda row: int(row["split_rank"]),
        )
        if len(dev_rows) != 20 or sum(row.get("rq2_role") == "rq2_causal_test" for row in source_rows) != 40:
            raise RQ2ConfigError("smoke source must have the frozen 20-dev/40-formal role split")
        chosen = dev_rows[:len(smoke_ids)]
        if [row["pair_id"] for row in chosen] != smoke_ids or subset_rows != chosen:
            raise RQ2ConfigError("smoke manifest must exactly copy the first 2–3 frozen dev rows by split_rank")
    if dev_screen:
        if not v2 or _bool(dev_screen.get("enabled"), "dev_screen.enabled") is not True:
            raise RQ2ConfigError("dev_screen requires schema v4 and dev_screen.enabled=true")
        if "dev_screen" not in name.casefold():
            raise RQ2ConfigError("dev_screen run name must contain dev_screen")
        screen_root = (expected_root / "dev_screen").resolve()
        if not _is_within(output_root, screen_root) or output_root == screen_root:
            raise RQ2ConfigError("dev_screen output_root must be a dedicated child of outputs/stage2_rq2/dev_screen")
        if subspace_enabled:
            raise RQ2ConfigError("dev_screen cannot enable subspace")
        source_manifest = _resolve(project_root, dev_screen.get("source_manifest"), "dev_screen.source_manifest")
        source_digest = _text(dev_screen.get("source_manifest_sha256"), "dev_screen.source_manifest_sha256")
        if manifest == source_manifest or not source_manifest.is_file():
            raise RQ2ConfigError("dev_screen requires a separate subset manifest")
        if hashlib.sha256(source_manifest.read_bytes()).hexdigest() != source_digest:
            raise RQ2ConfigError("dev_screen source manifest SHA mismatch")
        protocol = _resolve(project_root, dev_screen.get("screening_protocol"), "dev_screen.screening_protocol")
        protocol_digest = _text(dev_screen.get("screening_protocol_sha256"), "dev_screen.screening_protocol_sha256")
        if not protocol.is_file() or hashlib.sha256(protocol.read_bytes()).hexdigest() != protocol_digest:
            raise RQ2ConfigError("dev_screen screening protocol SHA mismatch")
        if not manifest.is_file():
            raise RQ2ConfigError("dev_screen subset manifest is missing")
        source_rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        subset_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        dev_rows = sorted(
            (row for row in source_rows if row.get("rq2_role") == "rq2_dev"),
            key=lambda row: int(row["split_rank"]),
        )
        ids = dev_screen.get("dev_pair_ids")
        if len(source_rows) != 60 or len(dev_rows) != 20 or sum(row.get("rq2_role") == "rq2_causal_test" for row in source_rows) != 40:
            raise RQ2ConfigError("dev_screen source must have the frozen 20-dev/40-formal split")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids) or len(set(ids)) != 20 or ids != [row["pair_id"] for row in dev_rows]:
            raise RQ2ConfigError("dev_screen IDs must include all 20 frozen dev pairs by split_rank")
        if subset_rows != dev_rows:
            raise RQ2ConfigError("dev_screen manifest must exactly copy all 20 frozen dev rows by split_rank")
    try:
        assert_config_matches_preregistration(raw, preregistration_record)
    except RQ2PreregistrationError as exc:
        raise RQ2ConfigError(str(exc)) from exc

    fingerprint = canonical_sha256(raw)
    return RQ2Config(
        path=config_path,
        project_root=project_root,
        name=name,
        template=template,
        execution_enabled=execution_enabled,
        frozen=frozen,
        output_root=output_root,
        manifest=manifest,
        preregistration_path=preregistration_path,
        preregistration_sha256=preregistration_sha256,
        preregistration_record=preregistration_record,
        statistical_preregistration_path=statistical_preregistration_path,
        statistical_preregistration_sha256=statistical_preregistration_sha256,
        statistical_preregistration_record=statistical_preregistration_record,
        trajectory=trajectory,
        model=model,
        rq1_sources=sources,
        sampling=sampling,
        pilot=pilot,
        formal=formal,
        smoke=smoke,
        dev_screen=dev_screen,
        judge=judge,
        statistics=statistics,
        raw=raw,
        fingerprint=fingerprint,
    )


def assert_executable(config: RQ2Config) -> None:
    if config.template:
        raise RQ2ConfigError("template config cannot execute; copy and rename it first")
    if not config.execution_enabled:
        raise RQ2ConfigError("execution_enabled is false")
    if config.frozen:
        raise RQ2ConfigError("frozen config is read-only")
    encoded = json.dumps(config.raw, ensure_ascii=False)
    if "RENAME_ME" in encoded or "REQUIRED" in encoded:
        raise RQ2ConfigError("config still contains template placeholders")


def select_stages(
    *,
    stage: Optional[str] = None,
    from_stage: Optional[str] = None,
    through_stage: Optional[str] = None,
) -> tuple[str, ...]:
    supplied = sum(value is not None for value in (stage, from_stage, through_stage))
    if stage is not None and supplied != 1:
        raise RQ2ConfigError("--stage cannot be combined with --from/--through")
    if stage is not None:
        if stage not in STAGE_ORDER:
            raise RQ2ConfigError(f"unknown stage {stage!r}")
        return (stage,)
    if from_stage is None and through_stage is None:
        raise RQ2ConfigError("run requires --stage or --from/--through")
    start = 0 if from_stage is None else STAGE_ORDER.index(from_stage)
    end = len(STAGE_ORDER) - 1 if through_stage is None else STAGE_ORDER.index(through_stage)
    if start > end:
        raise RQ2ConfigError("--from stage occurs after --through stage")
    return STAGE_ORDER[start : end + 1]


__all__ = [
    "RQ2Config",
    "RQ2ConfigError",
    "SCHEMA_VERSION",
    "SMOKE_STAGE_ORDER",
    "DEV_SCREEN_STAGE_ORDER",
    "STAGE_DEPENDENCIES",
    "STAGE_ORDER",
    "STAGE_RESOURCES",
    "assert_executable",
    "load_rq2_config",
    "select_stages",
]
