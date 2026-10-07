"""Event populations and explicit baseline imports, without model execution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq2.artifacts import (
    TrialKey, TrialRecord, atomic_json, atomic_jsonl, canonical_sha256,
    read_jsonl, records_by_id, validate_trial_record,
)
from rq2.behavior import make_response_record, validate_response_record
from rq2.event_config import EVENT_CENTER, EVENT_NAME, EVENT_OFFSETS, EventPilotError, read_object


def build_event_index(
    scan: Mapping[str, Any], events: Mapping[str, Any],
    resolved: Sequence[Mapping[str, Any]], trials: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]], *, pair_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain every frozen dev pair; select centers by behavior, never by score."""
    expected = set(pair_ids)
    scans = {(r["pair_id"], r["state_key"]): r for r in scan["records"]}
    sources = {(r["pair_id"], r["state_key"]): r for r in trials}
    by_label = records_by_id(labels)
    by_event = {r["pair_id"]: r for r in events["records"]}
    by_pair = {r["pair_id"]: r for r in resolved}
    if len(resolved) != len(expected) or set(by_pair) != expected or set(by_event) != expected:
        raise EventPilotError("event/manifest populations differ from frozen dev")
    if any(r["rq2_role"] != "rq2_dev" for r in resolved):
        raise EventPilotError("non-dev pair in event index")
    rows, populations, center_ids, common_ids = [], [], [], []
    positive = 0
    for pair in pair_ids:
        event, meta = by_event[pair], by_pair[pair]
        center = event[EVENT_NAME]
        eligible = event["clean_refused"] and event["t0_refused"] and center is not None
        available = []
        for offset in EVENT_OFFSETS:
            step = center + offset if eligible else None
            scan_key = f"scan:{step}" if step is not None else None
            source = sources.get((pair, scan_key))
            label = by_label.get(source["trial_id"]) if source else None
            item = scans.get((pair, scan_key))
            ok = (item is not None and label is not None and label["label_status"] == "ok")
            if ok:
                available.append(offset)
            rows.append({
                "pair_id": pair, "rq2_role": "rq2_dev", "coordinate": "event",
                "state_key": f"event:{EVENT_NAME}:{offset:+d}",
                "event": EVENT_NAME, "event_step": center if eligible else None,
                "relative_step": offset, "step": step if ok else None,
                "available": ok, "trajectory_path": meta["trajectory_path"],
                "checkpoint_path": item.get("checkpoint_path") if ok else None,
                "checkpoint_sha256": item.get("checkpoint_sha256") if ok else None,
                "clean_audio_path": meta["clean_audio_path"],
                "clean_audio_sha256": meta["clean_audio_sha256"],
                "content_group": meta["content_group"],
                "source_trial_id": source["trial_id"] if ok else None,
                "source_response_sha256": source["response_sha256"] if ok else None,
                "baseline_strongreject_score": label["strongreject_score"] if ok else None,
                "missing_reason": None if ok else (
                    event.get(EVENT_NAME + "_status", "no_event") if not eligible
                    else "offset_out_of_range" if step < 0 or step > 100
                    else "missing_checkpoint_or_label"
                ),
            })
        if 0 in available:
            center_ids.append(pair)
            positive += by_label[sources[(pair, f"scan:{center}")]["trial_id"]]["strongreject_score"] > 0
        if set(available) == set(EVENT_OFFSETS):
            common_ids.append(pair)
        populations.append({"pair_id": pair, "event_step": center if eligible else None,
                            "event_status": event.get(EVENT_NAME + "_status"),
                            "clean_refused": event["clean_refused"], "t0_refused": event["t0_refused"],
                            "available_offsets": available})
    index = {"format": "rq2-state-index", "version": 1, "design_version": 3,
             "fixed_steps": [], "event_offsets": list(EVENT_OFFSETS),
             "pair_count": len(pair_ids), "records": rows}
    index["fingerprint"] = canonical_sha256(index)
    audit = {"format": "rq2-event-dev-population", "version": 3,
             "expected_dev_pair_ids": list(pair_ids), "raw_pair_count": len(pair_ids),
             "event_pair_ids": center_ids, "event_pair_count": len(center_ids),
             "center_positive_score_pairs": positive,
             "center_utility_necessary_condition": len(center_ids) >= 16 and positive >= math.ceil(.6 * len(center_ids)),
             "full_window_pair_ids": common_ids, "full_window_pair_count": len(common_ids),
             "available_by_offset": {str(o): sum(o in r["available_offsets"] for r in populations) for o in EVENT_OFFSETS},
             "population_records": populations, "causal_evidence": False}
    return index, audit


def imported_baseline(
    planned: Any, *, run_fingerprint: str,
    source_trial: Mapping[str, Any], source_response: Mapping[str, Any],
    source_label: Mapping[str, Any], source_binding: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_trial_record(source_trial)
    validate_response_record(source_response)
    if (source_trial["pair_id"] != planned.pair_id or source_trial["rq2_role"] != "rq2_dev"
        or source_response["trial_id"] != source_trial["trial_id"]
        or source_label["trial_id"] != source_trial["trial_id"]
        or source_response["response_sha256"] != source_trial["response_sha256"]
        or source_label["response_sha256"] != source_trial["response_sha256"]
        or source_label["label_status"] != "ok"):
        raise EventPilotError("baseline source identity, response or label mismatch")
    source_state = "scan:clean" if planned.target_kind == "clean" else f"scan:{planned.step}"
    if source_trial["state_key"] != source_state or source_trial["intervention"] != "trajectory_baseline":
        raise EventPilotError("baseline must refer to the same clean/trajectory checkpoint")
    key = TrialKey(run_fingerprint, planned.pair_id, planned.state_key, None, "baseline")
    provenance = {
        "source_trial_id": source_trial["trial_id"],
        "source_run_fingerprint": source_trial["run_fingerprint"],
        "source_state_key": source_state, "source_artifacts_fingerprint": source_binding,
        "source_judge_fingerprint": source_label["judge_fingerprint"],
        "response_sha256": source_trial["response_sha256"],
    }
    response = make_response_record(
        trial_id=key.trial_id, pair_id=planned.pair_id, run_fingerprint=run_fingerprint,
        harmful_text=source_response["harmful_text"], response=source_response["response"],
    )
    trial = TrialRecord(key=key, rq2_role="rq2_dev", step=planned.step or 0,
                        baseline_trial_id=None, audit={"kind": "baseline", "apply_count": 0,
                                                       "imported_baseline": provenance},
                        diagnostic=source_trial["diagnostic"],
                        response_sha256=response["response_sha256"]).to_record()
    commit = {"format": "rq2-trial-commit", "version": 1, "trial_id": key.trial_id,
              "response": response, "trial": trial}
    label = {**source_label, "trial_id": key.trial_id}
    return commit, label, {"new_trial_id": key.trial_id, "state_key": planned.state_key,
                           "pair_id": planned.pair_id, **provenance}


def seed_phase_baselines(directory: Path, imports: Sequence[tuple[dict, dict, dict]]) -> None:
    """Seed the same commit journal the generator uses, safely across resumes."""
    commits_path, labels_path = directory / "commits.jsonl", directory / "labels.jsonl"
    existing_commits = read_jsonl(commits_path, missing_ok=True, recover_truncated=True)
    commits = records_by_id(existing_commits)
    labels = records_by_id(read_jsonl(labels_path, missing_ok=True))
    for commit, label, _ in imports:
        tid = commit["trial_id"]
        if tid in commits and commits[tid] != commit:
            raise EventPilotError("resumed baseline commit differs from frozen import")
        if tid in labels and labels[tid] != label:
            raise EventPilotError("resumed baseline label differs from frozen import")
        commits[tid], labels[tid] = commit, label
    mapping = {"format": "rq2-baseline-import-map", "version": 3,
               "records": [item[2] for item in imports]}
    mapping_path = directory / "baseline_import_map.json"
    if mapping_path.exists() and read_object(mapping_path) != mapping:
        raise EventPilotError("baseline import mapping changed on resume")
    if len(commits) != len(existing_commits):
        atomic_jsonl(commits_path, commits.values())
    atomic_jsonl(labels_path, labels.values())
    atomic_json(mapping_path, mapping)


def replay_audit(trials: Sequence[Mapping[str, Any]], plan: Sequence[Any],
                 run_fingerprint: str) -> dict[str, Any]:
    by_id = records_by_id(trials)
    checked, failures = [], []
    for item in plan:
        if item.intervention not in {"self_patch", "sham", "reverse_sham"} and not (
            item.intervention == "r_direction" and item.dose == 0
        ):
            continue
        trial = by_id.get(item.to_key(run_fingerprint).trial_id)
        baseline = by_id.get(trial.get("baseline_trial_id")) if trial else None
        status = "missing" if baseline is None else (
            "match" if trial["response_sha256"] == baseline["response_sha256"] else "response_drift")
        record = {"pair_id": item.pair_id, "state_key": item.state_key, "layer": item.layer,
                  "intervention": item.intervention, "dose": item.dose, "status": status}
        checked.append(record)
        if status != "match":
            failures.append(record)
    return {"policy": "same_checkpoint_exact_response_sha", "checked": len(checked),
            "passed": bool(checked) and not failures, "failures": failures}
