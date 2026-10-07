"""RQ2 manifest roles, split isolation, and attack-state indexing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from rq2.artifacts import canonical_sha256, file_sha256


RQ2_ROLES = ("rq2_dev", "rq2_causal_test", "rq1_reference")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RQ2DataError(ValueError):
    """Raised when RQ2 data roles or trajectory identities are invalid."""


@dataclass(frozen=True)
class RQ2Pair:
    pair_id: str
    role: str
    clean_audio_path: Path
    trajectory_path: Path
    harmful_text: str
    clean_refused: bool
    clean_audio_sha256: str
    trajectory_index_sha256: Optional[str]
    content_group: str
    row: Mapping[str, Any]


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            value = json.loads(text)
            if isinstance(value, Mapping):
                value = value.get("rows", value.get("records"))
            if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
                raise RQ2DataError(f"{path} must contain a list of objects")
            return [dict(row) for row in value]
        result = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise RQ2DataError(f"{path}:{number} must be an object")
            result.append(dict(value))
        return result
    raise RQ2DataError(f"unsupported manifest format: {path.suffix}")


def _text(value: Any, field: str) -> str:
    if value is None or not str(value).strip():
        raise RQ2DataError(f"{field} must be non-blank")
    return str(value).strip()


def _bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise RQ2DataError(f"{field} must be boolean")


def _path(base: Path, value: Any, field: str) -> Path:
    candidate = Path(_text(value, field)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_relative = (base / candidate).resolve()
    project_relative = (PROJECT_ROOT / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    if project_relative.exists():
        return project_relative
    return manifest_relative


def read_pair_ids(paths: Iterable[str | Path]) -> set[str]:
    result: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        for position, row in enumerate(_read_table(path)):
            pair_id = _text(row.get("pair_id"), f"{path}[{position}].pair_id")
            result.add(pair_id)
    return result


def read_pair_identities(paths: Iterable[str | Path]) -> dict[str, set[str]]:
    identities = {"pair_ids": set(), "content_groups": set(), "audio_sha256": set()}
    for value in paths:
        path = Path(value).expanduser().resolve()
        for position, row in enumerate(_read_table(path)):
            pair_id = _text(row.get("pair_id"), f"{path}[{position}].pair_id")
            identities["pair_ids"].add(pair_id)
            harmful = row.get("harmful_text")
            if harmful is not None and str(harmful).strip():
                normalized = " ".join(str(harmful).casefold().split())
                semantic_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                identities["content_groups"].add(f"text:{semantic_sha}")
                group = row.get("content_group")
                if group is not None and str(group).strip():
                    identities["content_groups"].add(f"group:{str(group).strip()}")
            audio = row.get("clean_audio_path") or row.get("harmful_audio_path")
            if audio:
                audio_path = _path(path.parent, audio, f"{path}[{position}].clean_audio_path")
                if audio_path.is_file():
                    identities["audio_sha256"].add(file_sha256(audio_path))
    return identities


def load_rq2_manifest(
    path: str | Path,
    *,
    rq1_pair_ids: Iterable[str] = (),
    rq1_content_groups: Iterable[str] = (),
    rq1_audio_sha256: Iterable[str] = (),
    allow_rq1_overlap_for_pilot: bool = False,
    require_paths: bool = True,
) -> tuple[RQ2Pair, ...]:
    manifest = Path(path).expanduser().resolve()
    rows = _read_table(manifest)
    if not rows:
        raise RQ2DataError("RQ2 manifest is empty")
    rq1_ids = set(rq1_pair_ids)
    rq1_content = set(rq1_content_groups)
    rq1_audio = set(rq1_audio_sha256)
    seen: set[str] = set()
    result: list[RQ2Pair] = []
    role_ids = {role: set() for role in RQ2_ROLES}
    role_content = {role: set() for role in RQ2_ROLES}
    role_audio = {role: set() for role in RQ2_ROLES}
    for position, row in enumerate(rows):
        prefix = f"{manifest}[{position}]"
        pair_id = _text(row.get("pair_id"), f"{prefix}.pair_id")
        role = _text(row.get("rq2_role"), f"{prefix}.rq2_role")
        if role not in RQ2_ROLES:
            raise RQ2DataError(f"{prefix}.rq2_role must be one of {RQ2_ROLES}")
        if pair_id in seen:
            raise RQ2DataError(f"duplicate pair_id in RQ2 manifest: {pair_id}")
        seen.add(pair_id)
        role_ids[role].add(pair_id)
        clean_audio = _path(
            manifest.parent,
            row.get("clean_audio_path") or row.get("harmful_audio_path"),
            f"{prefix}.clean_audio_path",
        )
        trajectory = _path(
            manifest.parent,
            row.get("trajectory_path") or row.get("trajectory_index"),
            f"{prefix}.trajectory_path",
        )
        if trajectory.is_dir():
            trajectory = trajectory / "index.json"
        if require_paths and not clean_audio.is_file():
            raise FileNotFoundError(clean_audio)
        if require_paths and not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        clean_audio_sha = file_sha256(clean_audio) if clean_audio.is_file() else ""
        if clean_audio_sha:
            if clean_audio_sha in role_audio[role]:
                raise RQ2DataError(f"duplicate clean audio within {role}: {pair_id}")
            role_audio[role].add(clean_audio_sha)
        clean_refused = _bool(row.get("clean_refused"), f"{prefix}.clean_refused")
        if not clean_refused:
            raise RQ2DataError(f"{pair_id} is not an eligible harmful-but-refused X_H")
        normalized_harm = " ".join(_text(row.get("harmful_text"), f"{prefix}.harmful_text").casefold().split())
        semantic_sha = hashlib.sha256(normalized_harm.encode("utf-8")).hexdigest()
        raw_group = row.get("content_group")
        content_group = str(raw_group).strip() if raw_group is not None and str(raw_group).strip() else semantic_sha
        content_identities = {f"text:{semantic_sha}", f"group:{content_group}"}
        overlap = content_identities.intersection(role_content[role])
        if overlap:
            raise RQ2DataError(
                f"duplicate harmful content within {role}: {sorted(overlap)}"
            )
        role_content[role].update(content_identities)
        result.append(
            RQ2Pair(
                pair_id=pair_id,
                role=role,
                clean_audio_path=clean_audio,
                trajectory_path=trajectory,
                harmful_text=_text(row.get("harmful_text"), f"{prefix}.harmful_text"),
                clean_refused=True,
                clean_audio_sha256=clean_audio_sha,
                trajectory_index_sha256=file_sha256(trajectory) if trajectory.is_file() else None,
                content_group=content_group,
                row=dict(row),
            )
        )

    if role_ids["rq2_dev"].intersection(role_ids["rq2_causal_test"]):
        raise RQ2DataError("rq2_dev and rq2_causal_test must be disjoint")
    content_overlap = role_content["rq2_dev"].intersection(role_content["rq2_causal_test"])
    if content_overlap:
        raise RQ2DataError("rq2_dev and rq2_causal_test overlap by content_group")
    audio_overlap = role_audio["rq2_dev"].intersection(role_audio["rq2_causal_test"])
    if audio_overlap:
        raise RQ2DataError("rq2_dev and rq2_causal_test overlap by clean-audio SHA-256")
    formal_overlap = role_ids["rq2_causal_test"].intersection(rq1_ids)
    if formal_overlap:
        raise RQ2DataError(
            "formal rq2_causal_test overlaps RQ1 pair IDs: "
            + ", ".join(sorted(formal_overlap)[:8])
        )
    formal_pairs = [pair for pair in result if pair.role == "rq2_causal_test"]
    if any(
        {
            f"group:{pair.content_group}",
            "text:" + hashlib.sha256(
                " ".join(pair.harmful_text.casefold().split()).encode("utf-8")
            ).hexdigest(),
        }.intersection(rq1_content)
        for pair in formal_pairs
    ):
        raise RQ2DataError("formal rq2_causal_test overlaps RQ1 by content identity")
    if any(pair.clean_audio_sha256 in rq1_audio for pair in formal_pairs):
        raise RQ2DataError("formal rq2_causal_test overlaps RQ1 by clean-audio SHA-256")
    pilot_overlap = role_ids["rq2_dev"].intersection(rq1_ids)
    if pilot_overlap and not allow_rq1_overlap_for_pilot:
        raise RQ2DataError(
            "rq2_dev overlaps RQ1 pair IDs without explicit pilot permission"
        )
    return tuple(result)


def trajectory_inventory(
    pair: RQ2Pair,
    *,
    terminal_step: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Resolve and hash every checkpoint in the required complete 0..100 trajectory."""

    from experiments.collect_safety_states import load_trajectory_checkpoints

    checkpoints = load_trajectory_checkpoints(pair.trajectory_path, pair_id=pair.pair_id)
    steps = tuple(item.step for item in checkpoints)
    expected = tuple(range(int(terminal_step) + 1))
    if steps != expected:
        raise RQ2DataError(
            f"trajectory for {pair.pair_id} must contain exactly 0..{terminal_step}"
        )
    return tuple(
        {
            "step": item.step,
            "checkpoint_path": str(item.path),
            "checkpoint_sha256": file_sha256(item.path),
            "experiment_fingerprint": item.metadata.get("experiment_fingerprint"),
            "metadata_pair_id": item.metadata.get("pair_id"),
        }
        for item in checkpoints
    )


def validate_trajectory_contract(
    pair: RQ2Pair,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an existing trajectory against the frozen Standard-PGD contract."""

    inventory = trajectory_inventory(pair, terminal_step=int(expected.get("steps", 100)))
    run_path = pair.trajectory_path.parent.parent / "run.json"
    if not run_path.is_file():
        raise RQ2DataError(f"trajectory for {pair.pair_id} lacks run.json provenance")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(run, Mapping):
        raise RQ2DataError(f"run.json for {pair.pair_id} must be an object")
    budget = run.get("budget")
    experiment = budget.get("experiment_config") if isinstance(budget, Mapping) else None
    if not isinstance(experiment, Mapping):
        raise RQ2DataError(f"run.json for {pair.pair_id} lacks experiment_config")
    required = {
        "method": "standard",
        "loss_type": "margin",
        "steps": int(expected.get("steps", 100)),
        "init_mode": "zero",
        "early_stop": False,
        "save_all_steps": True,
    }
    for key in ("model", "dtype", "target_text", "harmful_text"):
        if key in expected:
            required[key] = expected[key]
    mismatches = {
        key: {"expected": value, "observed": experiment.get(key)}
        for key, value in required.items()
        if experiment.get(key) != value
    }
    if "model_id" in expected:
        observed_model_id = experiment.get("model_id")
        expected_model_id = expected.get("model_id")
        same_model_id = observed_model_id == expected_model_id
        if not same_model_id and all(
            isinstance(value, str) and value.strip()
            for value in (observed_model_id, expected_model_id)
        ):
            observed_path = Path(str(observed_model_id)).expanduser()
            expected_path = Path(str(expected_model_id)).expanduser()
            observed_path = (
                observed_path.resolve()
                if observed_path.is_absolute()
                else (PROJECT_ROOT / observed_path).resolve()
            )
            expected_path = (
                expected_path.resolve()
                if expected_path.is_absolute()
                else (PROJECT_ROOT / expected_path).resolve()
            )
            same_model_id = observed_path == expected_path
        if not same_model_id:
            mismatches["model_id"] = {
                "expected": expected_model_id,
                "observed": observed_model_id,
            }
    for key in ("eps", "alpha", "kappa"):
        if key not in expected:
            continue
        try:
            matches = math.isclose(
                float(experiment.get(key, float("nan"))),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            mismatches[key] = {
                "expected": float(expected[key]),
                "observed": experiment.get(key),
            }
    input_audio = experiment.get("input_audio")
    input_sha = input_audio.get("sha256") if isinstance(input_audio, Mapping) else None
    if input_sha != pair.clean_audio_sha256:
        mismatches["input_audio.sha256"] = {
            "expected": pair.clean_audio_sha256,
            "observed": input_sha,
        }
    if str(run.get("pair_id")) != pair.pair_id:
        mismatches["pair_id"] = {"expected": pair.pair_id, "observed": run.get("pair_id")}
    if not isinstance(budget, Mapping) or budget.get("norm") != "linf":
        mismatches["norm"] = {"expected": "linf", "observed": None if not isinstance(budget, Mapping) else budget.get("norm")}
    experiment_fingerprint = budget.get("experiment_fingerprint")
    if (
        not isinstance(experiment_fingerprint, str)
        or len(experiment_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in experiment_fingerprint)
    ):
        mismatches["experiment_fingerprint"] = {
            "expected": "64-character SHA-256",
            "observed": experiment_fingerprint,
        }
    for item in inventory:
        if item["experiment_fingerprint"] != experiment_fingerprint:
            mismatches[f"checkpoint[{item['step']}].experiment_fingerprint"] = {
                "expected": experiment_fingerprint,
                "observed": item["experiment_fingerprint"],
            }
        if item["metadata_pair_id"] != pair.pair_id:
            mismatches[f"checkpoint[{item['step']}].pair_id"] = {
                "expected": pair.pair_id,
                "observed": item["metadata_pair_id"],
            }
    if mismatches:
        raise RQ2DataError(
            f"trajectory provenance mismatch for {pair.pair_id}: {mismatches}"
        )
    return {
        "pair_id": pair.pair_id,
        "clean_audio_path": str(pair.clean_audio_path),
        "clean_audio_sha256": pair.clean_audio_sha256,
        "run_path": str(run_path),
        "run_sha256": file_sha256(run_path),
        "trajectory_index_path": str(pair.trajectory_path),
        "trajectory_index_sha256": file_sha256(pair.trajectory_path),
        "experiment_fingerprint": experiment_fingerprint,
        "checkpoint_count": len(inventory),
        "terminal_step": inventory[-1]["step"],
        "checkpoint_inventory_sha256": canonical_sha256(inventory),
        "checkpoints": list(inventory),
    }


def load_behavior_events(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    result: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(_read_table(source)):
        pair_id = _text(row.get("pair_id"), f"{source}[{position}].pair_id")
        if pair_id in result:
            raise RQ2DataError(f"duplicate behavior event pair_id: {pair_id}")
        events: dict[str, Any] = {}
        judged_steps = row.get("judged_steps")
        if (
            not isinstance(judged_steps, list)
            or not judged_steps
            or any(isinstance(step, bool) or not isinstance(step, int) or step < 0
                   for step in judged_steps)
            or len(judged_steps) != len(set(judged_steps))
            or 0 not in judged_steps
        ):
            raise RQ2DataError(f"invalid judged_steps for {pair_id}")
        events["judged_steps"] = judged_steps
        for field in (
            "first_refusal_weakening_step",
            "first_non_refusal_step",
            "first_compliance_step",
        ):
            raw = row.get(field)
            if raw in (None, "") or (isinstance(raw, float) and math.isnan(raw)):
                events[field] = None
            else:
                numeric = float(raw)
                if not numeric.is_integer() or numeric < 0:
                    raise RQ2DataError(f"invalid {field} for {pair_id}")
                events[field] = int(numeric)
        result[pair_id] = events
    return result


def build_state_index(
    pairs: Sequence[RQ2Pair],
    *,
    fixed_steps: Sequence[int] = (2, 10, 100),
    event_offsets: Sequence[int] = (-2, -1, 0, 1, 3),
    behavior_events: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    if not fixed_steps or any(isinstance(step, bool) or step < 0 for step in fixed_steps):
        raise RQ2DataError("fixed_steps must contain non-negative integers")
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        inventory = trajectory_inventory(pair)
        by_step = {int(item["step"]): item for item in inventory}
        steps = tuple(by_step)
        available = set(steps)
        trajectory_sha = file_sha256(pair.trajectory_path)
        for step in fixed_steps:
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "rq2_role": pair.role,
                    "state_key": f"fixed:{step}",
                    "coordinate": "fixed",
                    "step": int(step) if step in available else None,
                    "available": step in available,
                    "trajectory_path": str(pair.trajectory_path),
                    "trajectory_sha256": trajectory_sha,
                    "checkpoint_path": None if step not in by_step else by_step[step]["checkpoint_path"],
                    "checkpoint_sha256": None if step not in by_step else by_step[step]["checkpoint_sha256"],
                    "clean_audio_sha256": pair.clean_audio_sha256,
                    "content_group": pair.content_group,
                }
            )
        event_map = (behavior_events or {}).get(pair.pair_id, {})
        judged_steps = event_map.get("judged_steps")
        if judged_steps is None:
            if event_map:
                raise RQ2DataError(
                    f"event map for {pair.pair_id} lacks judged_steps"
                )
            judged_available = available
        elif (
            not isinstance(judged_steps, list)
            or not judged_steps
            or any(isinstance(step, bool) or not isinstance(step, int) or step < 0
                   for step in judged_steps)
            or len(judged_steps) != len(set(judged_steps))
            or 0 not in judged_steps
        ):
            raise RQ2DataError(f"invalid judged_steps for {pair.pair_id}")
        else:
            judged_available = set(judged_steps)
        for event_name in (
            "first_refusal_weakening_step",
            "first_non_refusal_step",
            "first_compliance_step",
        ):
            event_step = event_map.get(event_name)
            for offset in event_offsets:
                selected = None if event_step is None else int(event_step) + int(offset)
                is_available = (
                    selected is not None
                    and selected in available
                    and selected in judged_available
                )
                rows.append(
                    {
                        "pair_id": pair.pair_id,
                        "rq2_role": pair.role,
                        "state_key": f"event:{event_name}:{offset:+d}",
                        "coordinate": "event",
                        "event": event_name,
                        "event_step": event_step,
                        "relative_step": int(offset),
                        "step": selected if is_available else None,
                        "available": is_available,
                        "trajectory_path": str(pair.trajectory_path),
                        "trajectory_sha256": trajectory_sha,
                        "checkpoint_path": None if not is_available else by_step[selected]["checkpoint_path"],
                        "checkpoint_sha256": None if not is_available else by_step[selected]["checkpoint_sha256"],
                        "clean_audio_sha256": pair.clean_audio_sha256,
                        "content_group": pair.content_group,
                    }
                )
    payload = {
        "format": "rq2-state-index",
        "version": 1,
        "fixed_steps": list(fixed_steps),
        "event_offsets": list(event_offsets),
        "pair_count": len(pairs),
        "records": rows,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def build_trajectory_scan_index(pairs: Sequence[RQ2Pair]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        records.append(
            {
                "pair_id": pair.pair_id,
                "rq2_role": pair.role,
                "state_key": "scan:clean",
                "step": None,
                "target_kind": "clean",
                "clean_audio_sha256": pair.clean_audio_sha256,
            }
        )
        for item in trajectory_inventory(pair):
            records.append(
                {
                    "pair_id": pair.pair_id,
                    "rq2_role": pair.role,
                    "state_key": f"scan:{item['step']}",
                    "step": item["step"],
                    "target_kind": "trajectory",
                    "checkpoint_path": item["checkpoint_path"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "clean_audio_sha256": pair.clean_audio_sha256,
                }
            )
    payload = {"format": "rq2-trajectory-scan-index", "version": 1, "records": records}
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


def derive_behavior_events(
    scan_trials: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    refusal_weakening_delta: float,
    event_offsets: Sequence[int] = (-2, -1, 0, 1, 3),
) -> dict[str, Any]:
    """Derive within-trajectory events from the judged PGD step-0 baseline."""

    if not 0.0 <= float(refusal_weakening_delta) <= 1.0:
        raise RQ2DataError("refusal_weakening_delta must be within [0,1]")
    label_by_id = {str(row.get("trial_id")): row for row in labels}
    by_pair: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    unknown_steps_by_pair: dict[str, set[int]] = {}
    for trial in scan_trials:
        pair_id = str(trial["pair_id"])
        by_pair.setdefault(pair_id, [])
        label = label_by_id.get(str(trial.get("trial_id")))
        if label is None or label.get("label_status") != "ok":
            if trial["state_key"] not in {"scan:clean", "scan:0"}:
                unknown_steps_by_pair.setdefault(pair_id, set()).add(int(trial["step"]))
            continue
        by_pair[pair_id].append((trial, label))
    records = []
    available_steps_by_pair: dict[str, set[int]] = {}
    for pair_id, values in sorted(by_pair.items()):
        clean = next((label for trial, label in values if trial["state_key"] == "scan:clean"), None)
        if clean is None:
            raise RQ2DataError(f"pair {pair_id} lacks a judged clean baseline")
        t0 = next((label for trial, label in values if trial["state_key"] == "scan:0"), None)
        if t0 is None:
            raise RQ2DataError(f"pair {pair_id} lacks a judged PGD step-0 baseline")
        clean_refused = bool(clean["refusal_label"])
        t0_refused = bool(t0["refusal_label"])
        clean_orientation = float(clean["refusal_orientation"])
        t0_orientation = float(t0["refusal_orientation"])
        stepped = sorted(
            (
                (int(trial["step"]), label)
                for trial, label in values
                if trial["state_key"].startswith("scan:")
                and trial["state_key"] != "scan:clean"
                and int(trial["step"]) >= 1
            ),
            key=lambda item: item[0],
        )
        unknown_steps = unknown_steps_by_pair.get(pair_id, set())
        first_unknown = min(unknown_steps) if unknown_steps else None
        available_steps_by_pair[pair_id] = {0, *(step for step, _ in stepped)}
        observed_weakening = (
            next(
                (
                    step
                    for step, label in stepped
                    if t0_orientation - float(label["refusal_orientation"])
                    >= refusal_weakening_delta
                ),
                None,
            )
            if t0_refused
            else None
        )
        observed_non_refusal = (
            next((step for step, label in stepped if not bool(label["refusal_label"])), None)
            if t0_refused
            else None
        )
        observed_compliance = (
            next((step for step, label in stepped if bool(label["compliance_label"])), None)
            if t0_refused
            else None
        )
        def resolve_first(observed: Optional[int]) -> tuple[Optional[int], str]:
            if not t0_refused:
                return None, "ineligible_t0_non_refusal"
            if first_unknown is not None and (observed is None or first_unknown < observed):
                return None, "unresolved_prior_missing_step"
            return observed, "observed" if observed is not None else "not_observed"

        weakening, weakening_status = resolve_first(observed_weakening)
        non_refusal, non_refusal_status = resolve_first(observed_non_refusal)
        compliance, compliance_status = resolve_first(observed_compliance)
        records.append(
            {
                "pair_id": pair_id,
                "clean_refused": clean_refused,
                "t0_refused": t0_refused,
                "baseline_refusing": t0_refused,
                "clean_refusal_orientation": clean_orientation,
                "t0_refusal_orientation": t0_orientation,
                "first_refusal_weakening_step": weakening,
                "first_refusal_weakening_step_status": weakening_status,
                "first_non_refusal_step": non_refusal,
                "first_non_refusal_step_status": non_refusal_status,
                "first_compliance_step": compliance,
                "first_compliance_step_status": compliance_status,
                "judged_step_count": len(stepped) + 1,
                "unknown_step_count": len(unknown_steps),
                "judged_steps": sorted(available_steps_by_pair[pair_id]),
            }
        )
    eligible = [row for row in records if row["clean_refused"] and row["t0_refused"]]
    event_fields = (
        "first_refusal_weakening_step",
        "first_non_refusal_step",
        "first_compliance_step",
    )
    population_counts = {
        "all_pairs": len(records),
        "clean_eligible_pairs": sum(row["clean_refused"] for row in records),
        "t0_refused_pairs": sum(row["t0_refused"] for row in records),
        "t0_non_refusal_pairs": sum(not row["t0_refused"] for row in records),
        "events": {
            field: {
                "eligible_pairs": len(eligible),
                "occurred_pairs": sum(row[field] is not None for row in eligible),
                "unresolved_pairs": sum(
                    row[field + "_status"] == "unresolved_prior_missing_step"
                    for row in eligible
                ),
                "available_by_offset": {
                    str(offset): sum(
                        row[field] is not None
                        and row[field] + offset in available_steps_by_pair[row["pair_id"]]
                        for row in eligible
                    )
                    for offset in event_offsets
                },
            }
            for field in event_fields
        },
    }
    payload = {
        "format": "rq2-behavior-events",
        "version": 3,
        "refusal_weakening_delta": float(refusal_weakening_delta),
        "event_offsets": list(event_offsets),
        "population_counts": population_counts,
        "records": records,
    }
    payload["fingerprint"] = canonical_sha256(payload)
    return payload


__all__ = [
    "RQ2DataError",
    "RQ2Pair",
    "RQ2_ROLES",
    "build_state_index",
    "build_trajectory_scan_index",
    "derive_behavior_events",
    "load_behavior_events",
    "load_rq2_manifest",
    "read_pair_ids",
    "read_pair_identities",
    "trajectory_inventory",
    "validate_trajectory_contract",
]
