"""Frozen, dev-only configuration and provenance for the v3 event pilot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from rq2.artifacts import atomic_json, canonical_sha256, file_sha256, read_jsonl
from rq2.config import RQ2Config, load_rq2_config
from rq2.dev_screen import summarize_dev_screen

EVENT_NAME = "first_non_refusal_step"
EVENT_CENTER = f"event:{EVENT_NAME}:+0"
EVENT_OFFSETS = (-2, -1, 0, 1, 3)
EVENT_STAGES = (
    "sources", "events", "state_index", "layer_map", "identity",
    "oracle_generate", "oracle_judge", "oracle_analyze",
    "mechanism_generate", "mechanism_judge", "mechanism_analyze",
    "window_generate", "window_judge", "window_analyze", "report",
)
SOURCE_FILES = (
    "pipeline_state.json", "baseline_screen/summary.json",
    "trajectory/summary.json", "trajectory/resolved_manifest.json",
    "trajectory_behavior/scan_index.json", "trajectory_behavior/trials.jsonl",
    "trajectory_behavior/responses.jsonl", "trajectory_behavior/labels.jsonl",
    "provenance/rq1_sources.json", "provenance/reference_statistics.json",
)


class EventPilotError(ValueError):
    pass


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EventPilotError(f"expected JSON object: {path}")
    return value


def event_design(base: RQ2Config) -> dict[str, Any]:
    """The complete dev design; changing it requires a new protocol/run."""
    return {
        "scope": "dev_only_event_pilot",
        "event": EVENT_NAME, "search_start": 1,
        "require_clean_refusal": True, "require_step0_refusal": True,
        "unresolved_prior_missing": "exclude_event_report_reason",
        "require_positive_baseline_score": False,
        "primary_state": EVENT_CENTER, "event_offsets": list(EVENT_OFFSETS),
        "window_population": "same_event_complete_case_within_layer",
        "window_pre": [-2, -1], "window_post": [0, 1, 3],
        "candidate_layers": list(base.pilot["candidate_layers"]),
        "neighbor_layers": list(base.pilot["neighbor_layers"]),
        "distant_control_layers": list(base.pilot["distant_control_layers"]),
        "primary_restoration_dose": 1.0, "primary_suppression_dose": 1.0,
        "sensitivity_doses": [0.5], "random_replicates": 3,
        "minimum_dev_pairs": 16, "minimum_sign_consistency": 0.60,
        "oracle_min_effect": 0.05, "mechanism_min_effect": 0.03,
        "corroboration": "same_event_center_adjacent_layer",
        "baseline_replay": "exact_response_sha_for_sham_self_patch_and_zero_dose",
        "norm_relative_tolerance": 0.05,
        "missing_effects": "never_improve_positive_fraction_denominator",
        "formal_minimum_valid_pairs": 20, "formal_source_pair_count": 40,
        "formal_families": {
            "F1": "event_center_utility", "F2": "event_center_binary_refusal",
            "F3": "same_layer_specificity", "F4": "clean_reverse",
            "E1": "same_event_complete_case_window_separate_family",
        },
        "formal_correction": "BH separately within each frozen family",
        "formal_execution": "separate_run_and_event_formal_executor_required",
        "bootstrap_unit": "pair_id", "random_aggregation": "within_pair_mean",
        "pilot_causal_claims": False, "auto_formal": False, "subspace": False,
    }


def _resolve(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EventPilotError("path must be a nonempty string")
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _check_ref(root: Path, value: Mapping[str, Any]) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise EventPilotError("artifact reference requires exactly path and sha256")
    path = _resolve(root, value["path"])
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise EventPilotError(f"bound artifact missing or changed: {path}")
    return path


@dataclass(frozen=True)
class EventPilotConfig:
    path: Path
    raw: Mapping[str, Any]
    source: RQ2Config
    protocol_path: Path
    preregistration_path: Path
    fingerprint: str
    output_root: Path

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(self.source.dev_screen["dev_pair_ids"])

    @property
    def runtime_config(self) -> RQ2Config:
        # Reuse model/trajectory/runtime plumbing, never the old stage scheduler.
        return replace(
            self.source, path=self.path, name=str(self.raw["name"]),
            output_root=self.output_root, dev_screen={}, smoke={},
            manifest=self.source.output_root / "trajectory/resolved_manifest.json",
            raw=self.raw, fingerprint=self.fingerprint,
        )

    def verify_source(self) -> dict[str, Any]:
        _check_ref(self.source.project_root, self.raw["source_config"])
        _check_ref(self.source.project_root, self.raw["protocol"])
        _check_ref(self.source.project_root, self.raw["preregistration"])
        for name, digest in self.raw["source_artifacts"].items():
            path = self.source.output_root / name
            if not path.is_file() or file_sha256(path) != digest:
                raise EventPilotError(f"source screen artifact changed: {name}")
        summary = summarize_dev_screen(self.source.path)
        if summary["decision"] != "fixed_state_infeasible" or summary["unknown_scan_labels"]:
            raise EventPilotError("event v3 requires the complete negative fixed-state screen")
        saved = read_object(self.source.output_root / "baseline_screen/summary.json")
        if saved != summary:
            raise EventPilotError("saved screen summary differs from verified sidecars")
        state = read_object(self.source.output_root / "pipeline_state.json")
        required_stages = ("sources", "trajectory", "trajectory_behavior_generate", "trajectory_behavior_judge")
        if set(state["stages"]) != set(required_stages):
            raise EventPilotError("source must contain only the four completed dev-screen stages")
        bound = {}
        for stage in required_stages:
            bound.update(state["stages"][stage]["artifacts"])
        for name in SOURCE_FILES:
            if name in {"pipeline_state.json", "baseline_screen/summary.json"}:
                continue
            path = self.source.output_root / name
            if bound.get(str(path)) != self.raw["source_artifacts"][name]:
                raise EventPilotError(f"source stage does not bind {name}")
        resolved = read_object(self.source.output_root / "trajectory/resolved_manifest.json")["records"]
        frozen = {row["pair_id"]: row for row in read_jsonl(self.source.manifest)}
        if len(resolved) != 20 or {r["pair_id"] for r in resolved} != set(self.pair_ids):
            raise EventPilotError("source resolved manifest is not exactly the 20 dev pairs")
        for row in resolved:
            if row.get("rq2_role") != "rq2_dev":
                raise EventPilotError("causal-test is forbidden in the dev event run")
            parent = frozen[row["pair_id"]]
            for key in ("content_group", "clean_audio_sha256", "harmful_text"):
                if row.get(key) != parent.get(key):
                    raise EventPilotError(f"resolved manifest changed {key}")
        return summary


def load_event_config(path: str | Path) -> EventPilotConfig:
    path = Path(path).resolve()
    root = path.parent.parent
    raw = read_object(path)
    fields = {"format", "version", "name", "execution_enabled", "output_root",
              "source_config", "source_artifacts", "protocol", "preregistration"}
    if set(raw) != fields or raw["format"] != "rq2-event-dev-pilot" or raw["version"] != 3:
        raise EventPilotError("unsupported event dev config; model/Judge/population overrides are forbidden")
    if not isinstance(raw["execution_enabled"], bool):
        raise EventPilotError("execution_enabled must be boolean")
    name = raw["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) or "event" not in name or "dev" not in name:
        raise EventPilotError("event dev run name must be a lowercase event/dev identifier")
    output = _resolve(root, raw["output_root"])
    if output != (root / "outputs/stage2_rq2/event_dev" / name).resolve():
        raise EventPilotError("event dev output must use its own event_dev/name namespace")
    source_path = _check_ref(root, raw["source_config"])
    source = load_rq2_config(source_path)
    if source.project_root != root or not source.dev_screen or source.smoke:
        raise EventPilotError("source must be the isolated dev-screen config in this project")
    if set(raw["source_artifacts"]) != set(SOURCE_FILES):
        raise EventPilotError("source artifact binding is incomplete")
    for field, expected in (("candidate_layers", [19,24,26,27]), ("neighbor_layers", [18,20,23,25]),
                            ("distant_control_layers", [2,12]), ("random_replicates", 3),
                            ("oracle_min_effect", 0.05), ("mechanism_min_effect", 0.03)):
        if source.pilot[field] != expected:
            raise EventPilotError(f"v3 requires unchanged frozen {field}")
    protocol = _check_ref(root, raw["protocol"])
    prereg = _check_ref(root, raw["preregistration"])
    record = read_object(prereg)
    expected = {
        "format": "rq2-v3-event-dev-preregistration", "version": 3,
        "development_status": "after_dev_baselines_before_event_interventions",
        "source_config": raw["source_config"], "protocol": raw["protocol"],
        "source_artifacts": raw["source_artifacts"],
        "source_manifest_sha256": source.dev_screen["source_manifest_sha256"],
        "dev_pair_ids": list(source.dev_screen["dev_pair_ids"]),
        "inherited_layer_dose_sha256": source.preregistration_sha256,
        "supersedes_fixed_state_primary_analysis": True,
        "design": event_design(source),
    }
    if record != expected:
        raise EventPilotError("v3 machine protocol differs from its frozen design or provenance")
    if source.pilot.get("subspace_enabled"):
        raise EventPilotError("subspace is not part of the event dev protocol")
    return EventPilotConfig(path, raw, source, protocol, prereg,
                            canonical_sha256(raw), output)


def prepare_event_config(source_path: str | Path, *, name: str, protocol: str | Path) -> dict[str, Any]:
    source = load_rq2_config(source_path)
    if not source.dev_screen:
        raise EventPilotError("prepare requires an isolated dev-screen config")
    root = source.project_root
    if not re.fullmatch(r"[a-z0-9_]+", name) or "event" not in name or "dev" not in name:
        raise EventPilotError("invalid event dev run name")
    protocol = Path(protocol).resolve()
    def reference(path: Path) -> dict[str, str]:
        return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
    artifacts = {name: file_sha256(source.output_root / name) for name in SOURCE_FILES}
    record = {
        "format": "rq2-v3-event-dev-preregistration", "version": 3,
        "development_status": "after_dev_baselines_before_event_interventions",
        "source_config": reference(source.path), "protocol": reference(protocol),
        "source_artifacts": artifacts,
        "source_manifest_sha256": source.dev_screen["source_manifest_sha256"],
        "dev_pair_ids": list(source.dev_screen["dev_pair_ids"]),
        "inherited_layer_dose_sha256": source.preregistration_sha256,
        "supersedes_fixed_state_primary_analysis": True, "design": event_design(source),
    }
    prereg_path = root / "configs" / f"rq2_event_preregistration_{name}.json"
    config_path = root / "configs" / f"stage2_rq2_{name}.json"
    # Encode once to bind the exact bytes atomic_json will emit.
    import hashlib
    prereg_sha = hashlib.sha256((json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    config = {
        "format": "rq2-event-dev-pilot", "version": 3, "name": name,
        "execution_enabled": True,
        "output_root": f"outputs/stage2_rq2/event_dev/{name}",
        "source_config": record["source_config"], "source_artifacts": artifacts,
        "protocol": record["protocol"],
        "preregistration": {"path": str(prereg_path.relative_to(root)), "sha256": prereg_sha},
    }
    for path, value in ((prereg_path, record), (config_path, config)):
        if path.exists() and read_object(path) != value:
            raise EventPilotError(f"refusing to overwrite different frozen preparation: {path}")
    for path, value in ((prereg_path, record), (config_path, config)):
        if not path.exists():
            atomic_json(path, value)
    spec = load_event_config(config_path)
    spec.verify_source()
    return {"config": str(config_path), "preregistration": str(prereg_path),
            "config_fingerprint": spec.fingerprint, "output_root": str(spec.output_root),
            "dev_pair_count": 20, "formal_pair_count": 0, "allowed_stages": list(EVENT_STAGES)}
