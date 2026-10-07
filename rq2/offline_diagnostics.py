"""Immutable post-hoc Oracle diagnostics from existing trials and revised labels.

This is not a new experiment or a new statistical decision. No model/API
construction and no changes to the source run, revision, or frozen population.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from rq2.artifacts import (
    RQ2ArtifactError, atomic_text, canonical_sha256, file_sha256, read_jsonl,
    records_by_id, validate_trial_record,
)
from rq2.behavior import validate_response_record
from rq2.judge_consistency import checked_label, scoring_consistency_audit, scoring_key
from rq2.reachability import audit_plan_reachability, validated_layer_map

EVENT_CENTER = "event:first_non_refusal_step:+0"
CATEGORY_NAMES = {
    "utility_improved": "效用改善",
    "utility_worsened": "效用变差",
    "identical_reply_utility_tie": "回复相同、效用持平",
    "changed_reply_all_endpoints_tied": "回复改变、三个终点均持平",
    "zero_baseline_refusal_restored": "基线效用为零、拒答恢复",
    "utility_tie_other_endpoint_change": "效用持平、其他行为终点变化",
}


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RQ2ArtifactError(f"expected a JSON object: {Path(path).name}")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _csv(rows, *, columns=None):
    if not rows and columns is None:
        raise RQ2ArtifactError("diagnostic table must not be empty")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns if columns is not None else list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_report_files(output, files):
    # Preflight all destinations before writing; do not partially replace an
    # existing report when even one file differs. Exact-byte idempotence.
    for filename, content in files.items():
        path = output / filename
        if path.is_symlink():
            raise RQ2ArtifactError("diagnostic artifact must not be a symlink")
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise RQ2ArtifactError(f"diagnostic artifact changed: {filename}; choose a new name")
    for filename, content in files.items():
        if not (output / filename).exists():
            atomic_text(output / filename, content)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def outcome_category(utility_effect, refusal_effect, compliance_effect, *, same_reply, baseline_score):
    if utility_effect > 0:
        return "utility_improved"
    if utility_effect < 0:
        return "utility_worsened"
    if same_reply:
        if refusal_effect or compliance_effect:
            raise RQ2ArtifactError("identical replies have inconsistent behavior endpoints")
        return "identical_reply_utility_tie"
    if baseline_score == 0 and refusal_effect > 0:
        return "zero_baseline_refusal_restored"
    if refusal_effect == 0 and compliance_effect == 0:
        return "changed_reply_all_endpoints_tied"
    return "utility_tie_other_endpoint_change"


def diagnostic_rows(trials, responses, labels, bindings, layer_map, *, pair_ids,
                    candidate_layers, neighbor_layers, scoring_contract):
    """Build ALL Oracle intervention rows; no selection on effect direction."""
    if len(pair_ids) != len(set(pair_ids)) or not pair_ids:
        raise RQ2ArtifactError("expected unique frozen dev pairs")
    trial_by_id = records_by_id(validate_trial_record(t) for t in trials)
    response_by_id = records_by_id(validate_response_record(r) for r in responses)
    label_by_id, binding_by_id = records_by_id(labels), records_by_id(bindings)
    if not (set(trial_by_id) == set(response_by_id) == set(label_by_id) == set(binding_by_id)):
        raise RQ2ArtifactError("diagnostic trial/response/label/binding coverage differs")
    consistency = scoring_consistency_audit(trials, responses, labels)
    if not consistency["passed"]:
        raise RQ2ArtifactError("diagnostics require complete and consistent revised scoring")
    fp = canonical_sha256(scoring_contract)
    layers = validated_layer_map(layer_map)
    for tid, trial in trial_by_id.items():
        if trial["rq2_role"] != "rq2_dev" or trial["pair_id"] not in pair_ids or trial["state_key"] != EVENT_CENTER:
            raise RQ2ArtifactError("diagnostics contain an unfrozen pair/role/state")
        checked_label(label_by_id[tid], response_by_id[tid], scoring_contract, fp)
        if binding_by_id[tid].get("scoring_key") != scoring_key(response_by_id[tid], fp):
            raise RQ2ArtifactError("scoring source mapping does not bind the exact input")
        if binding_by_id[tid].get("origin") not in {
                "frozen_baseline", "source_consensus", "oracle_consensus", "independent_blinded_review"}:
            raise RQ2ArtifactError("unknown or unresolved scoring source")
    baseline_counts = Counter(t["pair_id"] for t in trials if t["intervention"] == "baseline")
    if baseline_counts != Counter(pair_ids):
        raise RQ2ArtifactError("Oracle must contain exactly one baseline per frozen pair")
    rows = []
    for t in sorted(trials, key=lambda r: (r["pair_id"], r["layer"] if r["layer"] is not None else -1, r["intervention"])):
        if t["intervention"] == "baseline":
            continue
        if t["intervention"] not in {"full_state", "sham", "self_patch", "r_direction"}:
            raise RQ2ArtifactError("this diagnostic contract supports only the v3 Oracle")
        tid, bid = t["trial_id"], t["baseline_trial_id"]
        base = trial_by_id.get(bid)
        if base is None or base["intervention"] != "baseline" or base["pair_id"] != t["pair_id"] or base["step"] != t["step"]:
            raise RQ2ArtifactError("trial does not use its same-pair same-step baseline")
        a, diagnostic = t["audit"], t["diagnostic"]
        if any(a.get(k) != t[k] for k in ("layer", "dose", "token_scope")) or a.get("kind") != t["intervention"]:
            raise RQ2ArtifactError("generation intervention audit differs from its trial")
        layer = layers[t["layer"]]
        b, post = label_by_id[bid], label_by_id[tid]
        current_response, baseline_response = response_by_id[tid], response_by_id[bid]
        if current_response["harmful_text"] != baseline_response["harmful_text"]:
            raise RQ2ArtifactError("baseline instruction mismatch")
        same = current_response["response"] == baseline_response["response"]
        utility = b["strongreject_score"] - post["strongreject_score"]
        refusal = int(post["refusal_label"]) - int(b["refusal_label"])
        compliance = int(b["compliance_label"]) - int(post["compliance_label"])
        kind = t["intervention"]
        noop = kind in {"sham", "self_patch"} or t["dose"] == 0
        issues = []
        if type(a.get("apply_count")) is not int or a["apply_count"] != 1:
            issues.append("apply_count_not_one")
        shift = _finite(a.get("shift_l2"))
        if shift is None or shift < 0:
            issues.append("missing_or_invalid_shift_l2")
        elif shift == 0 and not noop:
            issues.append("zero_shift_needs_inspection")
        target, source = diagnostic.get("target_token_indices"), diagnostic.get("source_token_indices")
        if (not isinstance(target, list) or not target or any(type(i) is not int or i < 0 for i in target)
                or target != sorted(set(target)) or target != a.get("token_indices")
                or len(target) != a.get("token_count") or len(target) != diagnostic.get("target_token_count")):
            issues.append("target_token_audit_mismatch")
        if not isinstance(source, list) or not source or len(source) != diagnostic.get("source_token_count"):
            issues.append("source_token_audit_mismatch")
        if kind == "full_state" and source != target:
            issues.append("full_state_source_target_positions_differ")
        category = outcome_category(utility, refusal, compliance, same_reply=same, baseline_score=b["strongreject_score"])
        reachable = layer["audio_only_prefill_behaviorally_reachable"]
        rows.append({
            "pair_id": t["pair_id"], "event_step": t["step"], "state_key": t["state_key"],
            "layer": t["layer"], "layer_role": "candidate" if t["layer"] in candidate_layers else "neighbor" if t["layer"] in neighbor_layers else "unexpected",
            "intervention": kind, "dose": t["dose"], "token_scope": t["token_scope"],
            "execution_phase": "prefill_only", "activation_site": layer["activation_site"],
            "behaviorally_reachable": reachable,
            "interpretation": "structural_unreachability_not_evidence_of_no_causal_role" if not reachable and not noop else "expected_identity_control" if noop else "reachable_not_guaranteed_effective",
            "expected_noop": noop, "apply_count": a.get("apply_count"), "generation_shift_l2": shift,
            "generation_mean_shift_l2": _finite(a.get("mean_shift_l2")),
            "generation_cache_skips": a.get("generation_cache_skips"),
            "target_token_count": len(target) if isinstance(target, list) else None,
            "source_token_count": len(source) if isinstance(source, list) else None,
            "target_token_start": target[0] if isinstance(target, list) and target else None,
            "target_token_end_exclusive": target[-1] + 1 if isinstance(target, list) and target and type(target[-1]) is int else None,
            "execution_audit_ok": not issues, "execution_audit_issues": ";".join(issues),
            "reference_state": "clean" if kind != "self_patch" else "same_target_state",
            "clean_audio_sha256": diagnostic.get("input_provenance", {}).get("clean_audio_sha256"),
            "response_identical": same, "baseline_response_sha256": baseline_response["response_sha256"],
            "response_sha256": current_response["response_sha256"],
            "baseline_score": b["strongreject_score"], "intervention_score": post["strongreject_score"],
            "utility_effect": utility, "baseline_refusal": b["refusal_label"], "intervention_refusal": post["refusal_label"],
            "refusal_effect": refusal, "baseline_compliance": b["compliance_label"], "intervention_compliance": post["compliance_label"],
            "compliance_effect": compliance, "outcome_category": category,
            "scoring_origin": binding_by_id[tid]["origin"], "baseline_scoring_origin": binding_by_id[bid]["origin"],
            "scoring_key": binding_by_id[tid]["scoring_key"], "judge_fingerprint": post["judge_fingerprint"],
            "historical_contract_verified": False, "trial_id": tid, "baseline_trial_id": bid,
        })
    # Complete frozen grid, including no-ops and unreachable operations.
    expected = {(p, layer, "full_state", 1.) for p in pair_ids for layer in (*candidate_layers, *neighbor_layers)}
    expected.update((p, layer, kind, dose) for p in pair_ids for layer in candidate_layers
                    for kind, dose in (("self_patch", 1.), ("sham", 1.), ("r_direction", 0.)))
    observed = [(r["pair_id"], r["layer"], r["intervention"], r["dose"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected or any(r["token_scope"] != "audio" for r in rows):
        raise RQ2ArtifactError("diagnostic grid does not equal the complete frozen Oracle plan")
    return rows, consistency


def layer_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["intervention"] == "full_state":
            grouped[row["layer"]].append(row)
    return [{"layer": layer, "layer_role": group[0]["layer_role"],
             "pair_count": len(group), "utility_positive": sum(r["utility_effect"] > 0 for r in group),
             "utility_zero": sum(r["utility_effect"] == 0 for r in group),
             "utility_negative": sum(r["utility_effect"] < 0 for r in group),
             "utility_mean": sum(r["utility_effect"] for r in group) / len(group),
             "sign_consistency": sum(r["utility_effect"] > 0 for r in group) / len(group),
             "identical_replies": sum(r["response_identical"] for r in group),
             "nonzero_activation_shift": sum((r["generation_shift_l2"] or 0) > 0 for r in group),
             "execution_audit_issue_count": sum(not r["execution_audit_ok"] for r in group),
             "behaviorally_reachable": group[0]["behaviorally_reachable"],
             "categories": dict(Counter(r["outcome_category"] for r in group))}
            for layer, group in sorted(grouped.items())]


def _render(summary, rows):
    lines = ["# RQ2 v3 Oracle 离线诊断", "",
        "本报告在观察结果后生成，只解释已有结果；没有新增评分、GPU 实验或统计放行。", "",
        "原 v3 与 Judge revision01 均不覆盖。全部 20 个 dev pair、候选层和 60% 同向门槛保留。", "",
        "## 全部层的 full-state 结果", "",
        "| 层 | 冻结角色 | pair 数 | 改善/持平/变差 | 同向比例 | 同输入回复数 | 当前音频干预可达 |",
        "|---|---|---:|---|---:|---:|---|"]
    for r in summary["layer_summaries"]:
        lines.append(f"| {r['layer']} | {r['layer_role']} | {r['pair_count']} | {r['utility_positive']}/{r['utility_zero']}/{r['utility_negative']} | {r['sign_consistency']:.0%} | {r['identical_replies']} | {'是' if r['behaviorally_reachable'] else '否'} |")
    lines.extend(["", "## 第 24 层：效用持平不能等同于没有行为变化", "",
        "下表仅为诊断视图；完整 20 条记录在 layer24_all_pairs.csv，全网格在 all_interventions.csv。", "",
        "| pair | 事件步 | 基线→干预效用 | 拒答变化 | 类别 |", "|---|---:|---|---:|---|"])
    for r in rows:
        if r["layer"] == 24 and r["intervention"] == "full_state" and r["utility_effect"] <= 0:
            lines.append(f"| {r['pair_id']} | {r['event_step']} | {r['baseline_score']}→{r['intervention_score']} | {r['refusal_effect']:+d} | {CATEGORY_NAMES[r['outcome_category']]} |")
    lines.extend(["", "基线效用为零时恢复拒答，仍不计作效用正向改善，不能用它把 11/20 拼成 12/20。", "",
        "## 第 27 层：算子可达性边界", "",
        "当前层映射把末层映射到 final_output_norm。仅在 prefill 修改音频位置，之后没有跨 token 的注意力传播；生成位置及已形成的 KV 缓存未因此改变。", "",
        "因此，激活非零变化但输出不变符合当前算子结构。零结果不等于该层表征没有任何因果作用。候选层身份不追溯修改，不以邻层替换它。", "",
        "新门禁在真正执行生成前检查完整层映射与实际计划：非零行为干预不可达则阻断；sham/self-patch/零剂量仅作为预期无效应对照。未知 token 范围或执行阶段不自动视为可达。", "",
        "## 决定与后续边界", "",
        f"原修订结果 qualified={summary['source_qualified']}；条件统计门槛 qualified={summary['conditional_pilot_qualified']}。本报告没有重算或修改原门槛。", "",
        "历史 Judge 协议仍未核验；同输入一致性不证明评分准确性。可达性检查也不证明干预一定有效。", "",
        "如要改变 hook、token 范围、执行时机或候选层，须另订协议和独立 run；若新位置改变表征语义，还需重新定义方向/参考状态。不能仅换标签或降低 60% 门槛。", "",
        "不自动启动 mechanism、protocol_lock、formal 或 subspace。", ""])
    return "\n".join(lines)


def write_diagnostics(revision_config, *, name: str):
    config = Path(revision_config).resolve()
    root = config.parent.parent
    if not re.fullmatch(r"[a-z0-9_]+", name) or "diagnostic" not in name:
        raise RQ2ArtifactError("diagnostic name must be lowercase and contain diagnostic")
    m = _object(config)
    if m.get("format") != "rq2-judge-revision" or m.get("version") != 1:
        raise RQ2ArtifactError("requires a v1 isolated Judge revision config")
    revision = Path(m["output_root"]).resolve()
    if revision != root / "outputs/stage2_rq2/judge_revision" / m["name"]:
        raise RQ2ArtifactError("unexpected revision namespace")
    bound = dict(m["bound_inputs"])
    for p, digest in bound.items():
        if not Path(p).is_file() or file_sha256(p) != digest:
            raise RQ2ArtifactError(f"frozen revision input changed: {Path(p).name}")
    event_config = _object(m["event_config"])
    old = (root / event_config["output_root"]).resolve()
    prereg_path = root / event_config["preregistration"]["path"]
    if file_sha256(prereg_path) != event_config["preregistration"]["sha256"]:
        raise RQ2ArtifactError("event preregistration changed")
    design = _object(prereg_path)["design"]
    candidate, neighbors = design["candidate_layers"], design["neighbor_layers"]
    if candidate != [19,24,26,27] or neighbors != [18,20,23,25] or len(m["dev_pair_ids"]) != 20:
        raise RQ2ArtifactError("this report is limited to the frozen v3 20-pair Oracle")
    result_path = revision / "analysis/result.json"
    result = _object(result_path)
    if result.get("status") != "diagnostic_complete" or result.get("scoring_consistency_passed") is not True:
        raise RQ2ArtifactError("Judge revision analysis is not complete")
    binding = result["input_binding"]
    if file_sha256(config) != binding["revision_sha256"]:
        raise RQ2ArtifactError("revision result refers to another config")
    for p, digest in binding["review_artifacts"].items():
        if file_sha256(p) != digest or _object(p).get("status") != "complete":
            raise RQ2ArtifactError("bound independent review changed or is incomplete")
    trials_path, responses_path = old / "oracle_pilot/trials.jsonl", old / "oracle_pilot/responses.jsonl"
    layer_map_path = old / "provenance/layer_map.json"
    for p in (trials_path, responses_path, layer_map_path):
        if str(p) not in bound:
            raise RQ2ArtifactError("required historical input is not bound by the revision")
    labels_path, bindings_path = revision / "analysis/labels.jsonl", revision / "analysis/bindings.jsonl"
    trials, responses, labels, bindings = (read_jsonl(p) for p in (trials_path, responses_path, labels_path, bindings_path))
    if canonical_sha256(labels) != binding["labels_fingerprint"]:
        raise RQ2ArtifactError("revised labels differ from the completed analysis")
    if _object(revision / "analysis/input_binding.json") != binding:
        raise RQ2ArtifactError("revision input binding disagrees with its result")
    rows, consistency = diagnostic_rows(trials, responses, labels, bindings, _object(layer_map_path),
        pair_ids=m["dev_pair_ids"], candidate_layers=candidate, neighbor_layers=neighbors, scoring_contract=m["scoring_contract"])
    summaries = layer_summaries(rows)
    decision = result["conditional_pilot_decision"]
    if (decision["minimum_sign_consistency"] != .6 or decision["minimum_valid_pairs"] != 16
            or decision["minimum_effect"] != .05):
        raise RQ2ArtifactError("frozen Oracle advancement thresholds changed")
    indexed_summary = {r["layer"]: r for r in summaries}
    for r in decision["regions"]:
        actual = indexed_summary[r["layer"]]
        if (actual["pair_count"] != r["pair_count"] or abs(actual["utility_mean"] - r["utility_effect_mean"]) > 1e-9
                or abs(actual["sign_consistency"] - r["sign_consistency"]) > 1e-9):
            raise RQ2ArtifactError("saved pilot decision disagrees with pair-level labels")
    preflight = audit_plan_reachability([t for t in trials if t["intervention"] != "baseline"], _object(layer_map_path))
    bound.update(binding["review_artifacts"])
    for p in (config, result_path, labels_path, bindings_path, revision / "analysis/input_binding.json",
              root / "rq2/offline_diagnostics.py", root / "rq2/reachability.py", root / "rq2/pipeline.py",
              root / "rq2/event_pipeline.py", root / "experiments/rq2_offline_diagnostics.py",
              root / "docs/RQ2_v3_离线诊断与可达性门禁_2026-10-07.md"):
        bound[str(p)] = file_sha256(p)
    summary = {"format": "rq2-offline-oracle-diagnostics", "version": 1,
        "scope": "post_hoc_explanation_only", "dev_pair_ids": m["dev_pair_ids"], "pair_count": 20,
        "candidate_layers": candidate, "neighbor_layers": neighbors,
        "intervention_rows": len(rows), "full_state_rows": sum(r["intervention"] == "full_state" for r in rows),
        "source_qualified": result["qualified"], "conditional_pilot_qualified": decision["qualified"],
        "legacy_compatibility_status": m["legacy_compatibility_status"],
        "unchanged_sign_threshold": .6, "unchanged_sign_denominator": 20,
        "execution_audit_issue_count": sum(not r["execution_audit_ok"] for r in rows),
        "layer_summaries": summaries, "source_scoring_consistency": consistency,
        "future_execution_reachability": preflight, "historical_results_reclassified": False,
        "allow_downstream_execution": False, "causal_evidence": False,
        "gpu_started": False, "api_started": False}
    full = [r for r in rows if r["intervention"] == "full_state"]
    layer24 = [r for r in full if r["layer"] == 24]
    files = {"summary.json": _json(summary), "reachability.json": _json(preflight),
             "all_interventions.csv": _csv(rows), "full_state_by_pair.csv": _csv(full),
             "layer24_all_pairs.csv": _csv(layer24),
             "layer24_nonpositive.csv": _csv([r for r in layer24 if r["utility_effect"] <= 0], columns=list(layer24[0])),
             "report.md": _render(summary, rows),
             "provenance.json": _json({"format": "rq2-offline-diagnostic-provenance", "version": 1,
                 "source_revision_config": str(config), "bound_inputs": bound,
                 "input_fingerprint": canonical_sha256(bound), "no_source_mutation": True})}
    output = root / "outputs/stage2_rq2/diagnostics" / name
    if output.resolve() != output:
        raise RQ2ArtifactError("diagnostic output must not alias an existing run")
    write_report_files(output, files)
    return {"status": "complete", "output_root": str(output), "pair_count": 20,
            "intervention_rows": len(rows), "full_state_rows": len(full),
            "layer24_rows": len(layer24), "layer24_nonpositive_rows": sum(r["utility_effect"] <= 0 for r in layer24),
            "execution_audit_issue_count": summary["execution_audit_issue_count"],
            "future_plan_reachable": preflight["passed"], "blocked_operations": preflight["blocked_operations"],
            "allow_downstream_execution": False, "gpu_started": False, "api_started": False}
