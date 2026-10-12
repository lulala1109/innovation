"""Dev-only fixed-request-service scoring; returned model changes are warnings.

This does NOT assert alias equivalence or lock serving weights. Existing runs
are immutable. Only paid receipts/labels are imported; no GPU or baseline API.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import copy
import fcntl
import json
from pathlib import Path

from rq2.alias_resume import (AliasConfig, AliasPipeline, IMPORTED_STAGES, RUN_STAGES,
    copy_exact, load_alias_config, receipt_label, request_body, source_snapshot)
from rq2.artifacts import atomic_json, atomic_jsonl, atomic_text, canonical_sha256, file_sha256, read_jsonl
from rq2.judge_consistency import checked_label, scoring_key
from rq2.replication_config import ROOT, BUDGET, DESIGN, read_object
from rq2.replication_judge import ReplicationStop, utc_now
from rq2.replication_pipeline import ReplicationPipeline
from rq2.replication_rejudge import MigrationJudge, remaining_budget

SOURCE_CONFIG = "configs/stage2_rq2_qwen7b_advbench_v41_alias_resume01.json"
NAME = "qwen7b_advbench_replication_dev_v4_run01_judge_v41_warn01"
PROTOCOL = "docs/RQ2_V41_Judge返回标识仅告警协议_2026-10-10.md"
IMPLEMENTATION = ("rq2/dev_judge_resume.py", "experiments/rq2_dev_judge_resume.py", PROTOCOL)
POLICY = {
    "accepted_by": "user", "accepted_on": "2026-10-10",
    "scope": "this_dev_run_only", "request_model": "DeepSeek-V4.1-Flash",
    "request_endpoint": "unchanged_and_hash_bound",
    "returned_model_policy": "record_and_warn_never_stop_for_identity_alone",
    "missing_returned_model": "record_unknown_and_warn",
    "model_version_locked": False, "alias_equivalence_asserted": False,
    "allows_formal_or_causal_claim": False,
    "cache_policy": "first_valid_paid_receipt_per_exact_input_and_service_protocol",
}


def warning_contract(source):
    value = copy.deepcopy(source.raw["scoring_contract"])
    value.pop("alias_acceptance", None)
    value.update(version=5, returned_model_policy=POLICY["returned_model_policy"],
                 identity_policy=POLICY,
                 parent_contract_sha256=canonical_sha256(source.raw["scoring_contract"]))
    value["implementation_sha256"]["rq2/dev_judge_resume.py"] = file_sha256(Path(__file__))
    return value


def identity_summary(attempts):
    counts, unknown, changes = Counter(), [], []
    previous = None
    observed = False
    for row in attempts:
        if "provider_model" not in row:
            continue  # An uncertain HTTP outcome has no claimed model receipt.
        model = row["provider_model"]
        if isinstance(model, str) and model:
            counts[model] += 1
        else:
            model = None
            unknown.append(row["number"])
        if observed and model != previous:
            changes.append({"attempt": row["number"], "from": previous, "to": model})
        observed, previous = True, model
    return {"model_version_locked": False, "alias_equivalence_asserted": False,
            "count_basis": "HTTP receipts including inherited retries, not independent pairs",
            "returned_model_counts": dict(sorted(counts.items())),
            "unknown_model_attempts": unknown, "model_transitions": changes}


class DevWarningJudge(MigrationJudge):
    """Retain original transport/reservation/cache/retry code; decouple identity."""
    def save_identity_audit(self):
        atomic_json(self.root / "model_identity_audit.json",
                    {"identity_policy": POLICY, **identity_summary(self.state["attempts"])})

    def _settle(self, row, payload):
        raw_model = payload.get("model") if isinstance(payload, dict) else None
        prior = next((a.get("provider_model") for a in reversed(self.state["attempts"][:-1])
                      if "provider_model" in a), None)
        model = raw_model if isinstance(raw_model, str) and raw_model else None
        row.update(provider_model=model, returned_model_raw=raw_model,
                   returned_model_field_present=isinstance(payload, dict) and "model" in payload,
                   identity_policy_sha256=canonical_sha256(POLICY))
        self.state.update(identity_policy=POLICY, provider_model_is_pinned=False)
        if self.state.get("provider_model") is None and model is not None:
            self.state["provider_model"] = model  # First observed, NOT a pin.
        if model is None or model != prior:
            row["identity_warning"] = "unknown_returned_model" if model is None else "returned_model_changed"
            print("[judge identity warning] " + json.dumps({
                "attempt": row["number"], "returned_model": model,
                "action": "continue_dev_only", "model_version_locked": False}, ensure_ascii=True), flush=True)

        def persist():
            self.save()
            self.save_identity_audit()

        usage = payload.get("usage") if isinstance(payload, dict) else None
        def integer(v):
            return isinstance(v, int) and not isinstance(v, bool) and v >= 0
        if not isinstance(usage, dict) or not all(integer(usage.get(k)) for k in
                                                  ("prompt_tokens", "completion_tokens", "total_tokens")):
            row.update(status="uncertain", error="missing_or_invalid_usage")
            persist()
            raise ReplicationStop("Provider omitted valid usage; reservation retained")
        p, c = usage["prompt_tokens"], usage["completion_tokens"]
        if usage["total_tokens"] != p + c:
            row.update(status="uncertain", error="usage_total_mismatch")
            persist()
            raise ReplicationStop("Usage totals inconsistent; reservation retained")
        row.update(status="settled", finished_at=utc_now(), prompt_tokens=p, completion_tokens=c,
                   cost_micro_cny=p * self.budget["input_cny_per_million"] + c * self.budget["output_cny_per_million"])
        if p > row["prompt_token_reservation"] or c > 800:
            self.state["fatal_error"] = "usage_exceeded_reservation"
            persist()
            raise ReplicationStop("Usage exceeded reservation; actual usage recorded; identity warning did not override it")
        persist()
        if self.summary()["accounted_cny_at_peak_rates"] >= self.budget["api_limit_cny"]:
            raise ReplicationStop("API budget reached")

    def summary(self):
        return {**super().summary(), "model_version_locked": False,
                "returned_model_policy": POLICY["returned_model_policy"]}


def audit_scoring(root, judge, old_contract, new_contract, responses_by_phase, history):
    """Validate *every* charged request and cached result; never select a better score."""
    old_fp, new_fp = canonical_sha256(old_contract), canonical_sha256(new_contract)
    old_judge = MigrationJudge(root / "scoring", judge, old_contract, BUDGET, history)
    ledger = old_judge.state
    if ledger.get("fatal_error") != "provider_model_or_reservation_violation":
        raise ReplicationStop("Source is not the recorded model-identifier stop")
    inputs = {scoring_key(r, old_fp): r for phase in responses_by_phase.values() for r in phase.values()}
    successes, attempts_by_key = {}, Counter()
    corrected_by_key = Counter()
    receipts = {}
    for attempt in ledger["attempts"]:
        key = attempt["key"]
        if key not in inputs or attempt["status"] != "settled":
            raise ReplicationStop("Unresolved or unbound source request; manual reconciliation required")
        attempts_by_key[key] += 1
        response = inputs[key]
        bodies = [request_body(judge, response, corrected=flag) for flag in (False, True)]
        matches = [i for i, body in enumerate(bodies) if canonical_sha256(body) == attempt["request_sha256"]]
        if len(matches) != 1:
            raise ReplicationStop("Receipt request hash is not the frozen input/model/protocol")
        corrected_by_key[key] += matches[0]
        if attempts_by_key[key] > 3 or corrected_by_key[key] > 1:
            raise ReplicationStop("Source request limits were exceeded")
        body = bodies[matches[0]]
        bound = len(json.dumps(body["messages"], ensure_ascii=False).encode()) + 1024
        path = root / "scoring/receipts" / f"{attempt['number']:06d}.json"
        payload = read_object(path)
        receipts[attempt["number"]] = path
        usage = payload.get("usage", {})
        values = [usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
            raise ReplicationStop("Receipt usage is incomplete; do not release its reservation")
        p, c, total = values
        if (total != p + c or p > bound or c > 800
                or attempt.get("prompt_token_reservation") != bound
                or attempt.get("reserved_micro_cny") != bound * 2 + 800 * 9
                or attempt.get("prompt_tokens") != p or attempt.get("completion_tokens") != c
                or attempt["cost_micro_cny"] != p * 2 + c * 9
                or payload.get("model") != attempt.get("provider_model")):
            raise ReplicationStop("Source receipt/model metadata, usage or budget does not match the ledger")
        try:
            label = receipt_label(payload, response, old_fp)
        except (ValueError, TypeError, KeyError, IndexError):
            continue  # Accounted format/empty retries; not eligible for a label.
        if key in successes:
            raise ReplicationStop("Multiple successful receipts for one input; do not choose a score")
        successes[key] = {"response": response, "label": label, "number": attempt["number"],
                          "receipt": path, "receipt_sha256": file_sha256(path)}
    receipt_files = set((root / "scoring/receipts").glob("*.json"))
    if receipt_files != set(receipts.values()):
        raise ReplicationStop("Receipt inventory differs from the charged attempt ledger")
    cached = set()
    for path in (root / "scoring/cache").glob("*.json"):
        entry = read_object(path)
        key = path.stem
        good = successes.get(key)
        if (good is None or entry.get("key") != key or entry.get("status") != "ok"
                or entry.get("contract_sha256") != old_fp
                or Path(entry.get("receipt", "")).resolve() != good["receipt"].resolve()
                or entry.get("receipt_sha256") != good["receipt_sha256"]
                or {**entry.get("label", {}), "trial_id": good["response"]["trial_id"]} != good["label"]):
            raise ReplicationStop("Source cache conflicts with its bound receipt")
        cached.add(key)
    recovered = set(successes) - cached
    if recovered != {ledger["attempts"][-1]["key"]}:
        raise ReplicationStop("Expected exactly the final paid, uncached receipt to recover")
    for phase, responses in responses_by_phase.items():
        labels = read_jsonl(root / phase / "labels.jsonl")
        seen = set()
        for label in labels:
            tid = label["trial_id"]
            if tid not in responses or tid in seen:
                raise ReplicationStop("Source label IDs are duplicated or outside the generated data")
            seen.add(tid)
            response = responses[tid]
            checked_label(label, response, judge, old_fp)
            good = successes.get(scoring_key(response, old_fp))
            if good is None or label != {**good["label"], "trial_id": tid}:
                raise ReplicationStop("Source labels disagree with their paid receipts")
        if phase == "trajectory_behavior" and seen != set(responses):
            raise ReplicationStop("Source baseline labels are incomplete")
    key_map = {key: scoring_key(response, new_fp) for key, response in inputs.items()}
    oracle_keys = {scoring_key(r, old_fp) for r in responses_by_phase["oracle_pilot"].values()}
    return {"ledger": ledger, "successes": successes, "key_map": key_map,
            "responses": responses_by_phase, "summary": {
                "charged_attempts_preserved": len(ledger["attempts"]),
                "charged_micro_cny_preserved": sum(a["cost_micro_cny"] for a in ledger["attempts"]),
                "cached_inputs_preserved": len(cached), "recovered_receipt_numbers": [successes[k]["number"] for k in sorted(recovered)],
                "remaining_oracle_unique_inputs": len(oracle_keys - set(successes)),
                "baseline_labels_preserved": len(responses_by_phase["trajectory_behavior"]),
                "provider_confirmed": False, "model_version_locked": False}}


@dataclass(frozen=True)
class DevWarningConfig(AliasConfig):
    @property
    def runtime_config(self):
        return replace(self.source.runtime_config, path=self.path, name=NAME, raw=self.raw,
                       fingerprint=self.fingerprint, output_root=self.output_root)

    def inspect(self):
        self.source.verify()
        for name, digest in self.raw["bound_inputs"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or file_sha256(path) != digest:
                raise ReplicationStop(f"Frozen dev warning input changed: {name}")
        if warning_contract(self.source) != self.raw["scoring_contract"]:
            raise ReplicationStop("Dev warning scoring contract changed")
        parent = AliasPipeline(self.source)
        inventory = source_snapshot(parent)
        if canonical_sha256(inventory) != self.raw["source_snapshot_sha256"]:
            raise ReplicationStop("Frozen alias source changed; do not run both versions")
        responses = {p: parent._phase_data(p)[1] for p in ("trajectory_behavior", "oracle_pilot")}
        audit = audit_scoring(self.source.output_root, self.source.runtime_config.judge,
                              self.source.raw["scoring_contract"], self.raw["scoring_contract"],
                              responses, self.raw["historical_accounting"])
        if audit["summary"] != self.raw["offline_audit"]:
            raise ReplicationStop("Paid receipt reconciliation changed")
        return inventory, audit

    def verify(self):
        _, audit = self.inspect()
        return {"scope": "dev_fixed_request_service_unlocked_version", "identity_policy": POLICY,
                "offline_audit": audit["summary"], "gpu_or_api_started": False}


def draft_config():
    source = load_alias_config(ROOT / SOURCE_CONFIG)
    source.verify()
    parent = AliasPipeline(source)
    contract = warning_contract(source)
    inventory = source_snapshot(parent)
    responses = {p: parent._phase_data(p)[1] for p in ("trajectory_behavior", "oracle_pilot")}
    audit = audit_scoring(source.output_root, source.runtime_config.judge, source.raw["scoring_contract"],
                          contract, responses, source.raw["historical_accounting"])
    return {"format": "rq2-dev-judge-warning-recovery", "version": 1, "name": NAME,
            "source_config": SOURCE_CONFIG, "source_snapshot_sha256": canonical_sha256(inventory),
            "output_root": "outputs/stage2_rq2/independent_replication/" + NAME,
            "identity_policy": POLICY, "scoring_contract": contract, "budget": BUDGET, "design": DESIGN,
            "historical_accounting": source.raw["historical_accounting"], "offline_audit": audit["summary"],
            "generation_fingerprints": dict(source.raw["generation_fingerprints"]),
            "bound_inputs": {p: file_sha256(ROOT / p) for p in (SOURCE_CONFIG, *IMPLEMENTATION)}}


def load_warning_config(path):
    path = Path(path).resolve()
    raw = read_object(path)
    fields = {"format", "version", "name", "source_config", "source_snapshot_sha256", "output_root",
              "identity_policy", "scoring_contract", "budget", "design", "historical_accounting",
              "offline_audit", "generation_fingerprints", "bound_inputs"}
    if (set(raw) != fields or raw["format"] != "rq2-dev-judge-warning-recovery" or raw["version"] != 1
            or raw["name"] != NAME or raw["source_config"] != SOURCE_CONFIG
            or raw["identity_policy"] != POLICY or raw["budget"] != BUDGET or raw["design"] != DESIGN
            or raw["output_root"] != "outputs/stage2_rq2/independent_replication/" + NAME
            or set(raw["bound_inputs"]) != {SOURCE_CONFIG, *IMPLEMENTATION}):
        raise ReplicationStop("Not the authorized dev-only warning policy")
    out = ROOT / raw["output_root"]
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ReplicationStop("Recovery output must be an independent non-symlink namespace")
    for name, digest in raw["bound_inputs"].items():
        if file_sha256(ROOT / name) != digest:
            raise ReplicationStop(f"Frozen dev warning input changed: {name}")
    source = load_alias_config(ROOT / SOURCE_CONFIG)
    if (raw["historical_accounting"] != source.raw["historical_accounting"]
            or raw["generation_fingerprints"] != source.raw["generation_fingerprints"]
            or raw["scoring_contract"] != warning_contract(source)):
        raise ReplicationStop("Historical accounting, generation or scoring contract changed")
    return DevWarningConfig(path, raw, source, canonical_sha256(raw))


class DevWarningPipeline(AliasPipeline):
    def _judge(self):
        return DevWarningJudge(self.config.output_root / "scoring", self.config.judge,
                               self.spec.raw["scoring_contract"], BUDGET,
                               self.spec.raw["historical_accounting"], transport=self.transport)

    def model(self):
        raise ReplicationStop("GPU execution is forbidden in dev Judge recovery")

    @contextmanager
    def source_locks(self):
        with ExitStack() as stack:
            source = self.spec.source
            while source is not None:
                handle = stack.enter_context((source.output_root / ".run.lock").open("r"))
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ReplicationStop("A source experiment is active; recovery is blocked") from None
                source = getattr(source, "source", None)
            yield

    def _stage_artifacts(self, stage):
        paths = (*ReplicationPipeline._stage_artifacts(self, stage),
                 self.config.output_root / "provenance/dev_judge_recovery_audit.json")
        if stage == "oracle_judge":
            paths += (self.config.output_root / "scoring/model_identity_audit.json",)
        return paths

    def _prepare(self):
        print("[prepare] validating source hashes and paid receipts; no API/GPU", flush=True)
        inventory, audit = self.spec.inspect()
        root, source = self.config.output_root, self.spec.source.output_root
        root.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in root.rglob("*")):
            raise ReplicationStop("Recovery output contains symlinks")
        with (root / ".run.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReplicationStop("Another process owns the recovery") from None
            state = self._state()
            if state["stages"]:
                if any(s not in state["stages"] or not self._stage_is_fresh(s, state["stages"][s]) for s in IMPORTED_STAGES):
                    raise ReplicationStop("Prepared import changed")
                return {"status": "already_prepared", "gpu_or_api_started": False}
            imported_ledger = root / "scoring/ledger.json"
            if imported_ledger.exists():
                value = read_object(imported_ledger)
                if (value.get("recovery_source_sha256") != self.spec.raw["source_snapshot_sha256"]
                        or len(value.get("attempts", [])) != len(audit["ledger"]["attempts"])):
                    raise ReplicationStop("Scoring progressed without import state; do not reset it")
            parent_state = read_object(source / "pipeline_state.json")
            skip = {"trajectory_behavior/labels.jsonl", "trajectory_behavior/scoring_consistency_audit.json",
                    "provenance/scoring_contract.json", "provenance/rq1_sources.json"}
            relative = {str(Path(p).relative_to(source)) for row in parent_state["stages"].values()
                        for p in row["artifacts"] if Path(p).is_relative_to(source)}
            for rel in sorted(relative - skip):
                copy_exact(source / rel, root / rel)
            print("[prepare] importing receipts into separate dev warning run", flush=True)
            for attempt in audit["ledger"]["attempts"]:
                rel = f"scoring/receipts/{attempt['number']:06d}.json"
                copy_exact(source / rel, root / rel)
                if attempt["number"] % 250 == 0:
                    print(f"[prepare] receipts {attempt['number']}/{len(audit['ledger']['attempts'])}", flush=True)
            ledger = copy.deepcopy(audit["ledger"])
            ledger.pop("fatal_error", None)  # Only in the new namespace, after complete receipt audit.
            ledger.update(binding=canonical_sha256({"contract": self.spec.raw["scoring_contract"],
                "budget": remaining_budget(BUDGET, self.spec.raw["historical_accounting"])}),
                recovery_source_sha256=self.spec.raw["source_snapshot_sha256"],
                resolved_source_error="provider_model_or_reservation_violation",
                identity_policy=POLICY, provider_model_is_pinned=False)
            ledger["source_alias_acceptance"] = ledger.pop("alias_acceptance", None)
            for row in ledger["attempts"]:
                row["previous_scoring_keys"] = [*row.get("previous_scoring_keys", []), row["key"]]
                row["key"] = audit["key_map"][row["key"]]
            atomic_json(imported_ledger, ledger)
            new_fp = canonical_sha256(self.spec.raw["scoring_contract"])
            print("[prepare] rebuilding content cache without rescoring", flush=True)
            for index, (key, item) in enumerate(audit["successes"].items(), 1):
                new_key = audit["key_map"][key]
                receipt = root / "scoring/receipts" / f"{item['number']:06d}.json"
                atomic_json(root / "scoring/cache" / (new_key + ".json"), {
                    "key": new_key, "contract_sha256": new_fp, "status": "ok",
                    "label": {**item["label"], "judge_fingerprint": new_fp},
                    "receipt": str(receipt), "receipt_sha256": item["receipt_sha256"]})
                if index % 250 == 0:
                    print(f"[prepare] cache {index}/{len(audit['successes'])}", flush=True)
            imported_counts = {}
            old_fp = canonical_sha256(self.spec.source.raw["scoring_contract"])
            for phase, responses in audit["responses"].items():
                labels = []
                for tid, response in responses.items():
                    good = audit["successes"].get(scoring_key(response, old_fp))
                    if good:
                        labels.append({**good["label"], "trial_id": tid, "judge_fingerprint": new_fp})
                atomic_jsonl(root / phase / "labels.jsonl", labels)
                imported_counts[phase] = len(labels)
            self._phase_data("trajectory_behavior")
            self._phase_data("oracle_pilot")
            self._require_consistent_scoring(root / "trajectory_behavior")
            atomic_json(root / "trajectory_behavior/scoring_consistency_audit.json", self._scoring_audit(root / "trajectory_behavior"))
            metadata = read_object(source / "provenance/rq1_sources.json")
            metadata.update(config_fingerprint=self.spec.fingerprint, source_config_fingerprint=self.spec.source.fingerprint,
                            judge_protocol_fingerprint=new_fp, identity_policy=POLICY,
                            labels_rebound_without_rescoring=True)
            metadata["source_alias_acceptance"] = metadata.pop("alias_acceptance", None)
            atomic_json(root / "provenance/rq1_sources.json", metadata)
            atomic_json(root / "provenance/scoring_contract.json", self.spec.raw["scoring_contract"])
            atomic_json(root / "provenance/source_inventory.json", inventory)
            atomic_json(root / "provenance/dev_judge_recovery_audit.json", {
                "format": "rq2-dev-judge-warning-recovery", "version": 1, "identity_policy": POLICY,
                "source_snapshot_sha256": self.spec.raw["source_snapshot_sha256"],
                "source_contract_sha256": old_fp, "effective_contract_sha256": new_fp,
                "generation_baseline_import_references": "historical_source_records_preserved_unchanged",
                "label_change": "fingerprint_only_no_score_changes", "raw_receipts_changed": False,
                "offline_audit": audit["summary"], "imported_labels": imported_counts,
                "gpu_or_api_started": False})
            state["stages"] = {s: {"output": str(self.stage_path(s)), "resources": ["cpu"],
                                   "mode": "imported_under_unlocked_service_version_policy",
                                   "artifacts": {str(p): file_sha256(p) for p in self._stage_artifacts(s)}}
                               for s in IMPORTED_STAGES}
            self._judge().save_identity_audit()
            self._write_state(state)
            return {"status": "prepared", "imported_labels": imported_counts,
                    "offline_audit": audit["summary"], "gpu_or_api_started": False}


    def run(self, stages, *, acknowledge_cost=False, accept_unlocked_judge=False):
        if (not stages or any(s not in RUN_STAGES for s in stages)
                or [RUN_STAGES.index(s) for s in stages] != sorted({RUN_STAGES.index(s) for s in stages})):
            raise ReplicationStop("Only Oracle Judge, analysis and report are permitted; no formal or GPU")
        if not acknowledge_cost or not accept_unlocked_judge:
            raise ReplicationStop("Explicitly acknowledge cost and the unlocked dev Judge version")
        with self.source_locks():
            self._prepare()
            return ReplicationPipeline.run(self, stages, acknowledge_cost=True, confirm_no_external_rq2_use=True)

    def _run_oracle_analyze(self):
        ReplicationPipeline._run_oracle_analyze(self)
        path = self.stage_path("oracle_analyze")
        value = read_object(path)
        value.update(identity_policy=POLICY, model_version_locked=False,
                     result_scope="dev_fixed_request_service_not_version_controlled",
                     model_identity_audit=identity_summary(self._judge().state["attempts"]),
                     allow_downstream_execution=False)
        value["pilot_decision"]["qualification_scope"] = "dev_service_level_only_not_formal"
        atomic_json(path, value)
        self._run_report()

    def _run_report(self):
        ReplicationPipeline._run_report(self)
        path = self.stage_path("report")
        audit = identity_summary(self._judge().state["attempts"])
        warning = ("# 重要限制：固定请求入口，实际 Judge 服务版本未锁定\n\n"
                   f"固定请求模型：{self.config.judge['model']}；请求地址与评分参数沿用冻结配置。\n\n"
                   "返回模型标识变化仅告警并保留原始回执；不声称这些名称对应完全相同的模型或推理实现。"
                   "以下结果仅属于当前 dev 服务级实验，不能自动进入 formal、mechanism 或 subspace，也不产生因果结论。\n\n"
                   "已观察的返回标识（计数为HTTP回执，含历史重试，不是独立pair）："
                   + json.dumps(audit["returned_model_counts"], ensure_ascii=False)
                   + f"；标识缺失/无效回执数：{len(audit['unknown_model_attempts'])}。\n\n")
        atomic_text(path, warning + path.read_text(encoding="utf-8"))

    def estimate(self):
        result = AliasPipeline.estimate(self)
        result.pop("alias_acceptance", None)
        result["identity_policy"] = POLICY
        return result
