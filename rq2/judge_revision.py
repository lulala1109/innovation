"""Isolated, append-only revision of a completed v3 Oracle's Judge labels.

No GPU path exists here. Preparation/audit never instantiate an API client.
Historical protocol equivalence remains unverified: derived results are
diagnostic and cannot authorize a downstream experiment.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rq2.artifacts import (
    atomic_json, atomic_jsonl, atomic_text, canonical_sha256, file_sha256,
    read_jsonl, records_by_id, RQ2ArtifactError,
)
from rq2.behavior import label_from_judge_result, unknown_label, validate_response_record
from rq2.event_config import EVENT_CENTER, load_event_config, read_object
from rq2.judge_consistency import (
    ENDPOINTS, RUBRIC, checked_label, scoring_consistency_audit,
    scoring_contract, scoring_key, signature,
)

POLICY = {
    "version": 1,
    "scope": "post_observation_dev_oracle_reanalysis_only",
    "authority_order": ["exact_frozen_imported_baseline", "unanimous_source_labels", "unanimous_original_oracle_labels"],
    "agreement": "all_canonical_rubric_and_endpoint_fields",
    "source_conflict": "unresolved_even_if_oracle_labels_agree",
    "authority_conflict": "block_not_adjudicate_event_anchor_in_place",
    "conflict_review": "one_blinded_evaluation_per_unique_input_no_outcome_based_retry",
    "unknown": "unresolved_never_zero_never_shrink_sign_denominator",
    "legacy_compatibility": "unverified_historical_prompt_and_endpoint",
    "allow_downstream_execution": False, "allow_formal": False,
}


def _immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_object(path) != value:
            raise RQ2ArtifactError(f"refusing to overwrite frozen revision artifact: {path.name}")
    else:
        atomic_json(path, value)


def _immutable_jsonl(path: Path, rows) -> None:
    rows = list(rows)
    if path.exists():
        if read_jsonl(path) != rows:
            raise RQ2ArtifactError(f"refusing to overwrite frozen revision artifact: {path.name}")
    else:
        atomic_jsonl(path, rows)


def resolve_groups(responses, labels, source_responses, source_labels, baseline_map,
                   *, contract, legacy_fingerprint, judge):
    """Pure migration: no favorable-label selection and no network side effects."""
    fp = canonical_sha256(contract)
    original = records_by_id(labels)
    source_by_id = records_by_id(validate_response_record(r) for r in source_responses)
    source_label_by_id = records_by_id(source_labels)
    response_by_id = records_by_id(validate_response_record(r) for r in responses)
    if set(original) != set(response_by_id) or set(source_by_id) != set(source_label_by_id):
        raise RQ2ArtifactError("revision requires exact response/label coverage")
    groups, source_groups, anchors = defaultdict(list), defaultdict(list), defaultdict(list)
    for tid, r in response_by_id.items():
        checked_label(original[tid], r, judge, legacy_fingerprint)
        groups[scoring_key(r, fp)].append(tid)
    for tid, r in source_by_id.items():
        checked_label(source_label_by_id[tid], r, judge, legacy_fingerprint)
        source_groups[scoring_key(r, fp)].append(tid)
    for mapping in baseline_map:
        tid, sid = mapping["new_trial_id"], mapping["source_trial_id"]
        r, src = response_by_id[tid], source_by_id[sid]
        if (r["harmful_text"] != src["harmful_text"] or r["response"] != src["response"]
                or r["pair_id"] != src["pair_id"]
                or original[tid] != {**source_label_by_id[sid], "trial_id": tid}):
            raise RQ2ArtifactError("frozen baseline import does not match its exact source label")
        anchors[scoring_key(r, fp)].append(sid)
    migrated, bindings, cache, conflicts, pending = [], [], [], [], []
    for key, tids in sorted(groups.items()):
        tids = sorted(tids)
        source_ids = sorted(source_groups.get(key, []))
        anchor_ids = sorted(set(anchors.get(key, [])))
        oracle_rows = [original[t] for t in tids]
        source_rows = [source_label_by_id[t] for t in source_ids]
        source_conflict = len({signature(l) for l in source_rows if l["label_status"] == "ok"}) > 1
        endpoint_conflict = len({signature(l, ENDPOINTS) for l in oracle_rows}) > 1
        rubric_conflict = len({signature(l, RUBRIC) for l in oracle_rows}) > 1
        pool = ([source_label_by_id[s] for s in anchor_ids] if anchor_ids else
                source_rows if source_rows else oracle_rows)
        origin = "frozen_baseline" if anchor_ids else "source_consensus" if source_rows else "oracle_consensus"
        known = [l for l in pool if l["label_status"] == "ok"]
        chosen = known[0] if len(known) == len(pool) and len({signature(l) for l in known}) == 1 else None
        if chosen is None:
            origin = "anchor_conflict_blocked" if anchor_ids else "unresolved"
            pending.append({"scoring_key": key, "trial_ids": tids,
                            "review_allowed": not bool(anchor_ids),
                            "reason": origin, "endpoint_conflict": endpoint_conflict,
                            "rubric_conflict": rubric_conflict, "source_rubric_conflict": source_conflict})
        conflict = {"scoring_key": key, "trial_ids": tids, "source_trial_ids": source_ids,
                    "anchor_source_trial_ids": anchor_ids,
                    "oracle_endpoint_conflict": endpoint_conflict,
                    "oracle_rubric_conflict": rubric_conflict, "source_conflict": source_conflict,
                    "authority_disagrees_with_oracle": bool(chosen and any(signature(l) != signature(chosen) for l in oracle_rows)),
                    "resolution": origin}
        if endpoint_conflict or rubric_conflict or source_conflict or conflict["authority_disagrees_with_oracle"]:
            conflicts.append(conflict)
        representative = response_by_id[tids[0]]
        if chosen is not None:
            entry = {**chosen, "trial_id": key, "judge_fingerprint": fp}
            cache.append({"scoring_key": key, "origin": origin,
                          "legacy_origin_trial_ids": anchor_ids or source_ids or tids,
                          "legacy_judge_fingerprint": legacy_fingerprint, "label": entry})
        for tid in tids:
            label = ({**chosen, "trial_id": tid, "judge_fingerprint": fp} if chosen else
                     unknown_label(tid, representative["response_sha256"], judge_fingerprint=fp,
                                   error_type="UnresolvedScoringConflict", retryable=False).to_record())
            migrated.append(label)
            bindings.append({"trial_id": tid, "scoring_key": key, "origin": origin,
                             "legacy_migration": True, "historical_contract_verified": False})
    return {"labels": migrated, "bindings": bindings, "cache": cache,
            "conflicts": conflicts, "pending": pending}


def _source_data(spec):
    from rq2.event_pipeline import EventPilotPipeline
    pipeline = EventPilotPipeline(spec)
    spec.verify_source()
    state = pipeline._state()["stages"]
    for stage in ("sources", "events", "state_index", "layer_map", "identity", "oracle_generate", "oracle_judge", "oracle_analyze"):
        if stage not in state or not pipeline._stage_is_fresh(stage, state[stage]):
            raise RQ2ArtifactError(f"revision requires the unchanged, completed original {stage}")
    pipeline._validate_legacy_judge_inputs("oracle_pilot")
    directory = spec.output_root / "oracle_pilot"
    trials = read_jsonl(directory / "trials.jsonl")
    responses = read_jsonl(directory / "responses.jsonl")
    labels = read_jsonl(directory / "labels.jsonl")
    planned = {p.to_key(pipeline._run_fingerprint("oracle_pilot")).trial_id for p in pipeline._phase_plan("oracle_pilot")}
    baselines = {t["trial_id"] for t in trials if t["intervention"] == "baseline"}
    population = pipeline.population()
    if (len(population["event_pair_ids"]) != 20 or set(population["event_pair_ids"]) != set(spec.pair_ids)
            or set(records_by_id(trials)) != planned | baselines or len(baselines) != 20):
        raise RQ2ArtifactError("revision must preserve the exact 20-pair Oracle plan")
    baseline_map = read_object(directory / "baseline_import_map.json")["records"]
    if {m["new_trial_id"] for m in baseline_map} != baselines or len(baseline_map) != len(baselines):
        raise RQ2ArtifactError("baseline import map coverage mismatch")
    _, source_responses, source_labels = pipeline._source_data()
    return pipeline, trials, responses, labels, list(source_responses.values()), list(source_labels.values()), baseline_map


def prepare_revision(event_config, *, name: str, protocol: str | Path):
    spec = load_event_config(event_config)
    root = spec.source.project_root
    if not re.fullmatch(r"[a-z0-9_]+", name) or "judge_revision" not in name:
        raise RQ2ArtifactError("revision name must be lowercase and contain judge_revision")
    output = root / "outputs/stage2_rq2/judge_revision" / name
    if output.is_symlink() or output.resolve() != output:
        raise RQ2ArtifactError("revision output must not alias an existing run")
    pipeline, trials, responses, labels, sr, sl, mapping = _source_data(spec)
    contract = scoring_contract(spec.source.judge)
    bound = {spec.path, spec.source.path, spec.protocol_path, spec.preregistration_path,
             spec.output_root / "pipeline_state.json", Path(protocol).resolve()}
    for record in pipeline._state()["stages"].values():
        bound.update(Path(p) for p in record["artifacts"])
    bound.update(spec.source.output_root / p for p in spec.raw["source_artifacts"])
    # Freeze scoring/analysis implementation in the derived namespace, not v3.
    bound.update(root / p for p in ("rq2/judge_revision.py", "rq2/judge_consistency.py",
        "rq2/analysis.py", "rq2/pilot.py", "rq2/behavior.py", "evaluation/behavior.py", "core/llm_backend.py"))
    manifest = {"format": "rq2-judge-revision", "version": 1, "name": name,
                "event_config": str(spec.path), "event_config_fingerprint": spec.fingerprint,
                "output_root": str(output), "policy": POLICY, "scoring_contract": contract,
                "legacy_judge_fingerprint": canonical_sha256({k:v for k,v in spec.source.judge.items() if k != "base_url"}),
                "bound_inputs": {str(p): file_sha256(p) for p in sorted(bound)},
                "dev_pair_ids": list(spec.pair_ids), "event_population": pipeline.population(),
                "legacy_compatibility_status": "unverified", "causal_evidence": False}
    config = root / "configs" / f"rq2_{name}.json"
    _immutable_json(config, manifest)
    return audit_revision(config)


def _load(config):
    config = Path(config).resolve()
    m = read_object(config)
    if m.get("format") != "rq2-judge-revision" or m.get("version") != 1 or m.get("policy") != POLICY:
        raise RQ2ArtifactError("unknown/modified scoring revision policy")
    root = config.parent.parent
    if not re.fullmatch(r"[a-z0-9_]+", str(m.get("name", ""))):
        raise RQ2ArtifactError("invalid revision name")
    output = (root / "outputs/stage2_rq2/judge_revision" / m["name"])
    if Path(m["output_root"]) != output or output.resolve() != output:
        raise RQ2ArtifactError("revision output escapes the isolated namespace")
    for p, digest in m["bound_inputs"].items():
        if not Path(p).is_file() or file_sha256(p) != digest:
            raise RQ2ArtifactError(f"frozen revision input changed: {Path(p).name}; use a new revision")
    spec = load_event_config(m["event_config"])
    if (m["event_config_fingerprint"] != spec.fingerprint or m["dev_pair_ids"] != list(spec.pair_ids)
            or scoring_contract(spec.source.judge) != m["scoring_contract"]
            or m["legacy_compatibility_status"] != "unverified"):
        raise RQ2ArtifactError("revision contract/population changed")
    return m, output, spec


def _migration(manifest, spec):
    data = _source_data(spec)
    pipeline, trials, responses, labels, sr, sl, mapping = data
    result = resolve_groups(responses, labels, sr, sl, mapping,
                            contract=manifest["scoring_contract"],
                            legacy_fingerprint=manifest["legacy_judge_fingerprint"], judge=spec.source.judge)
    return result, data


def audit_revision(config):
    manifest, output, spec = _load(config)
    result, (_, trials, responses, labels, sr, sl, _) = _migration(manifest, spec)
    audit = scoring_consistency_audit(trials, responses, labels)
    for name in ("labels", "bindings", "cache", "conflicts", "pending"):
        _immutable_jsonl(output / "offline" / f"{name}.jsonl", result[name])
    _immutable_json(output / "offline/original_scoring_audit.json", audit)
    fp = canonical_sha256(manifest["scoring_contract"])
    unique_responses = {scoring_key(r, fp): r for r in responses}
    # Deliberately exclude layers, conditions, trial IDs, old scores and reasons.
    tasks = [{"scoring_key": p["scoring_key"], "harmful_text": unique_responses[p["scoring_key"]]["harmful_text"],
              "response": unique_responses[p["scoring_key"]]["response"]}
             for p in result["pending"] if p["review_allowed"]]
    _immutable_jsonl(output / "private/review_inputs.jsonl", tasks)
    source_groups = defaultdict(list)
    source_labels = records_by_id(sl)
    for r in sr:
        source_groups[scoring_key(r, fp)].append(source_labels[r["trial_id"]])
    summary = {"revision_config": str(Path(config).resolve()), "output_root": str(output),
               "original_trials": len(trials), "unique_scoring_inputs": audit["unique_scoring_inputs"],
               "original_endpoint_conflict_groups": audit["endpoint_conflict_groups"],
               "original_rubric_conflict_groups": audit["rubric_conflict_groups"],
               "source_endpoint_conflict_groups": sum(len({signature(l, ENDPOINTS) for l in g}) > 1 for g in source_groups.values()),
               "source_rubric_conflict_groups": sum(len({signature(l, RUBRIC) for l in g}) > 1 for g in source_groups.values()),
               "resolved_unique_inputs": len(result["cache"]),
               "resolved_trials": sum(l["label_status"] == "ok" for l in result["labels"]),
               "pending_unique_inputs": len(result["pending"]), "blinded_review_inputs": len(tasks),
               "resolution_counts": dict(Counter(b["origin"] for b in result["bindings"])),
               "legacy_compatibility_status": "unverified", "eligible_for_analysis": not result["pending"],
               "allow_downstream_execution": False, "gpu_started": False, "api_started": False}
    _immutable_json(output / "offline/summary.json", summary)
    _immutable_json(output / "provenance.json", {"revision_manifest_sha256": file_sha256(config), **manifest})
    return summary


def review_revision(config, *, allow_api=False, evaluator_factory=None):
    """One evaluation per pending unique input; crashes/failures are not retried.

    Backend transport/schema retries are bounded by the frozen contract. They
    are not repeated attempts to obtain a favorable valid judgment.
    """
    if not allow_api:
        raise RQ2ArtifactError("review requires explicit --allow-api; audit never spends API quota")
    manifest, output, spec = _load(config)
    audit_revision(config)
    tasks = read_jsonl(output / "private/review_inputs.jsonl")
    fp = canonical_sha256(manifest["scoring_contract"])
    results = output / "review"
    evaluator = None
    for task in tasks:
        key = task["scoring_key"]
        path = results / f"{key}.json"
        if path.exists():
            continue
        if evaluator is None:
            if evaluator_factory is None:
                from evaluation.behavior import StrongRejectEvaluator
                evaluator = StrongRejectEvaluator(provider=spec.source.judge["provider"], model=spec.source.judge["model"],
                    base_url=spec.source.judge["base_url"], schema_retries=spec.source.judge.get("schema_retries", 2))
            else:
                evaluator = evaluator_factory()
            if hasattr(evaluator, "public_config"):
                public = evaluator.public_config()
                resolved = {**spec.source.judge, **public}
                if scoring_contract(resolved) != manifest["scoring_contract"]:
                    raise RQ2ArtifactError("actual scoring backend differs from frozen contract")
        # Exclusive creation before calling the API also prevents racing runs
        # or an interrupted request from silently obtaining a second judgment.
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = {"scoring_key": key, "status": "started", "contract_sha256": fp,
                   "revision_sha256": file_sha256(config)}
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(pending, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except FileExistsError:
            continue
        try:
            result = evaluator.evaluate(task["harmful_text"], task["response"])
            import hashlib
            label = label_from_judge_result(key, hashlib.sha256(task["response"].encode()).hexdigest(), result,
                judge_fingerprint=fp, success_threshold=manifest["scoring_contract"]["success_threshold"],
                min_convincingness=manifest["scoring_contract"]["min_convincingness"],
                min_specificity=manifest["scoring_contract"]["min_specificity"]).to_record()
            value = {**pending, "status": "complete", "label": label}
        except Exception as exc:
            value = {**pending, "status": "failed", "error_type": type(exc).__name__}
        atomic_json(path, value)
        print(f"[judge_revision] {key[:12]} {value['status']}", flush=True)
    return {"review_inputs": len(tasks), "completed": sum(read_object(results / f"{t['scoring_key']}.json").get("status") == "complete" for t in tasks),
            "automatic_retry": False, "allow_downstream_execution": False}


def event_definition_audit(source_trials, source_responses, source_labels,
                           responses, labels, baseline_map, fingerprint):
    """Check overlapping inputs; do not redefine or independently revalidate t*."""
    old_labels = records_by_id(source_labels)
    old_responses = records_by_id(source_responses)
    new_labels = records_by_id(labels)
    keyed = {scoring_key(r, fingerprint): new_labels[r["trial_id"]] for r in responses}
    centers = {m["pair_id"]: int(m["source_state_key"].split(":")[1]) for m in baseline_map}
    conflicts, checked = [], 0
    for trial in source_trials:
        state = trial["state_key"]
        if trial["pair_id"] not in centers or (state != "scan:clean" and trial["step"] > centers[trial["pair_id"]]):
            continue
        tid = trial["trial_id"]
        row = keyed.get(scoring_key(old_responses[tid], fingerprint))
        if row is None:
            continue
        checked += 1
        if row["refusal_label"] != old_labels[tid]["refusal_label"]:
            conflicts.append({"source_trial_id": tid, "pair_id": trial["pair_id"], "state_key": state,
                              "old_refusal": old_labels[tid]["refusal_label"], "revised_refusal": row["refusal_label"]})
    return {"passed": not conflicts, "checked_overlapping_event_predicates": checked,
            "conflicts": conflicts, "independent_event_revalidation": False}


def analyze_revision(config):
    manifest, output, spec = _load(config)
    audit_revision(config)
    migration, (pipeline, trials, responses, old_labels, sr, sl, baseline_map) = _migration(manifest, spec)
    fp = canonical_sha256(manifest["scoring_contract"])
    response_by_id = records_by_id(responses)
    labels = records_by_id(migration["labels"])
    review_hashes, unresolved = {}, []
    for pending in migration["pending"]:
        key = pending["scoring_key"]
        path = output / "review" / f"{key}.json"
        if not pending["review_allowed"] or not path.is_file():
            unresolved.append(key)
            continue
        reviewed = read_object(path)
        if (reviewed.get("status") != "complete" or reviewed.get("scoring_key") != key
                or reviewed.get("contract_sha256") != fp or reviewed.get("revision_sha256") != file_sha256(config)):
            unresolved.append(key)
            continue
        if reviewed.get("label", {}).get("trial_id") != key:
            raise RQ2ArtifactError("review label does not bind its unique scoring input")
        for tid in pending["trial_ids"]:
            row = {**reviewed["label"], "trial_id": tid}
            labels[tid] = checked_label(row, response_by_id[tid], spec.source.judge, fp)
        review_hashes[str(path)] = file_sha256(path)
    if unresolved:
        return {"status": "blocked", "reason": "unresolved_scoring_inputs",
                "unresolved_unique_inputs": len(unresolved), "qualified": False,
                "analysis_started": False, "allow_downstream_execution": False}
    audit = scoring_consistency_audit(trials, responses, list(labels.values()))
    if not audit["passed"]:
        raise RQ2ArtifactError("revised scoring consistency audit failed; analysis blocked")
    event_audit = event_definition_audit(pipeline._source_data()[0], sr, sl, responses,
                                       list(labels.values()), baseline_map, fp)
    _immutable_json(output / "event_definition_audit.json", event_audit)
    if not event_audit["passed"]:
        return {"status": "blocked", "reason": "event_predicate_changed_requires_new_protocol",
                "qualified": False, "analysis_started": False, "allow_downstream_execution": False,
                "event_definition_audit": event_audit}
    # One immutable analysis snapshot; no overwrites after a changed adjudication.
    target = output / "analysis"
    binding = {"revision_sha256": file_sha256(config), "review_artifacts": review_hashes,
               "labels_fingerprint": canonical_sha256(list(labels.values()))}
    _immutable_json(target / "input_binding.json", binding)
    if (target / "result.json").is_file():
        return read_object(target / "result.json")
    _immutable_jsonl(target / "labels.jsonl", labels.values())
    final_bindings = [dict(b) for b in migration["bindings"]]
    reviewed_keys = {p["scoring_key"] for p in migration["pending"]}
    for binding_row in final_bindings:
        if binding_row["scoring_key"] in reviewed_keys:
            binding_row.update(origin="independent_blinded_review", legacy_migration=False,
                               review_path=str(output / "review" / f"{binding_row['scoring_key']}.json"))
    _immutable_jsonl(target / "bindings.jsonl", final_bindings)
    final_cache = {}
    for row in final_bindings:
        key = row["scoring_key"]
        final_cache.setdefault(key, {"scoring_key": key, "origin": row["origin"],
                                    "label": {**labels[row["trial_id"]], "trial_id": key}})
    _immutable_jsonl(target / "cache.jsonl", final_cache.values())
    _immutable_json(target / "scoring_consistency_audit.json", audit)
    from rq2.analysis import analyze_trial_set
    from rq2.pilot import evaluate_dev_pilot, read_pilot_csv
    summary = analyze_trial_set(spec.output_root / "oracle_pilot/trials.jsonl", target / "labels.jsonl",
        output_dir=target, pilot=True, event=True, event_pair_ids=manifest["dev_pair_ids"], **pipeline._analysis_kwargs())
    rows = read_pilot_csv(target / "rq2_causal_map_mean_ci.csv")
    for row in rows:
        if row["state_key"] == EVENT_CENTER:
            row["sign_consistency"] = round(float(row["sign_consistency"]) * int(row["pair_count"])) / len(manifest["dev_pair_ids"])
    decision = evaluate_dev_pilot(rows, intervention="full_state", dose=1.,
        candidate_layers=spec.source.pilot["candidate_layers"], neighbor_layers=spec.source.pilot["neighbor_layers"],
        fixed_steps=(), event_state=EVENT_CENTER, minimum_effect=.05)
    old = read_object(spec.output_root / "oracle_pilot/analysis.json")
    result = {"status": "diagnostic_complete", "scoring_consistency_passed": True,
              "conditional_pilot_decision": decision, "qualified": False,
              "blocked_reason": "legacy_scoring_contract_not_historically_verified",
              "causal_evidence": False, "allow_downstream_execution": False,
              "event_definition_audit": event_audit,
              "original_pilot_decision": old["pilot_decision"],
              "analysis_summary": summary, "population_size": len(manifest["dev_pair_ids"]),
              "label_changes": {field: sum(labels[l["trial_id"]][field] != l[field] for l in old_labels) for field in ENDPOINTS},
              "input_binding": binding}
    _immutable_json(target / "result.json", result)
    atomic_text(target / "report.md", "# Oracle 评分修订诊断\n\n"
        "原始文件未覆盖；20 个 dev pair、候选层及 60% 门槛保持不变。\n\n"
        f"评分一致性通过。条件性统计门槛：{decision['qualified']}。\n\n"
        "历史评分模板与实际服务端点未完整记录，兼容性未核验；qualified=false，禁止据此自动推进机制或 formal。\n")
    return result
