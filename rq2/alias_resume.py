"""Audited, conditional Oracle continuation for exactly two accepted model IDs.

The source is immutable. This is an operator assumption, not evidence that the
provider served identical weights. No GPU stages or baseline rejudging exist
in this entry. Raw receipts and all charged attempts survive the migration.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import copy
import fcntl
import json
from pathlib import Path

from evaluation.behavior import StrongRejectEvaluator, StrongRejectScore
from rq2.artifacts import atomic_json, atomic_jsonl, atomic_text, canonical_sha256, file_sha256, read_jsonl
from rq2.behavior import label_from_judge_result
from rq2.judge_consistency import scoring_key, checked_label
from rq2.replication_config import ROOT, BUDGET, DESIGN, STAGES, read_object
from rq2.replication_judge import ReplicationStop
from rq2.replication_pipeline import ReplicationPipeline
from rq2.replication_rejudge import NAME as SOURCE_NAME, MigrationJudge, RejudgePipeline, load_rejudge_config, remaining_budget

NAME = SOURCE_NAME + "_alias01"
SOURCE_CONFIG = "configs/stage2_rq2_qwen7b_advbench_replication_dev_v4_judge_v41_run01.json"
PROTOCOL = "docs/RQ2_V41_Judge别名人工接受与恢复协议_2026-10-10.md"
IMPLEMENTATION = ("rq2/alias_resume.py", "experiments/rq2_alias_resume.py", PROTOCOL)
IMPORTED_STAGES = STAGES[:9]
RUN_STAGES = ("oracle_judge", "oracle_analyze", "report")
ALIASES = ("deepseek-flash", "deepseek-v4-1-flash-260910")
POLICY = {"accepted_by": "user", "accepted_on": "2026-10-10",
          "basis": "explicit_user_acceptance_without_provider_confirmation",
          "provider_confirmed": False, "allowed_returned_models": list(ALIASES),
          "request_model": "DeepSeek-V4.1-Flash", "scope": "this_dev_run_only",
          "unknown_models": "stop", "allows_formal_or_causal_claim": False}


def alias_contract(source):
    value = copy.deepcopy(source.raw["scoring_contract"])
    value.update(version=4, returned_model_policy="explicit_unverified_two_alias_allowlist",
                 alias_acceptance=POLICY,
                 parent_contract_sha256=canonical_sha256(source.raw["scoring_contract"]))
    value["implementation_sha256"]["rq2/alias_resume.py"] = file_sha256(Path(__file__))
    return value


def request_body(judge, response, *, corrected=False):
    messages = [{"role": "user", "content": StrongRejectEvaluator.JUDGE_PROMPT.format(
        target_text=response["harmful_text"], response_to_evaluate=response["response"])}]
    if corrected:
        messages.append({"role": "user", "content": StrongRejectEvaluator.SCHEMA_RETRY_INSTRUCTION})
    return {"model": judge["model"], "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}, "max_tokens": 800,
            "thinking": {"type": "disabled"}}


def receipt_label(payload, response, fingerprint):
    parsed = StrongRejectScore.model_validate_json(payload["choices"][0]["message"]["content"]).model_dump()
    return label_from_judge_result(response["trial_id"], response["response_sha256"], parsed,
                                   judge_fingerprint=fingerprint).to_record()


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
                or payload.get("model") not in ALIASES
                or payload.get("model") != attempt.get("provider_model")):
            raise ReplicationStop("Source has a non-alias model/usage/budget anomaly")
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
                "provider_confirmed": False}}


def source_snapshot(parent):
    root = parent.config.output_root
    state = parent._state()
    if set(state["stages"]) != set(IMPORTED_STAGES):
        raise ReplicationStop("Source must have completed Oracle generation but not Oracle Judge")
    expected = {}
    for stage in IMPORTED_STAGES:
        expected.update(state["stages"][stage]["artifacts"])
    files = {Path(p) for p in expected}
    for p in root.rglob("*"):
        if p.is_symlink() or p.name.endswith(".tmp"):
            raise ReplicationStop("Source contains a symlink or incomplete temporary artifact")
        if p.is_file() and p.name != ".run.lock":
            files.add(p)
    inventory = {}
    for path in sorted(files):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ReplicationStop("Source artifact is outside the project")
        digest = file_sha256(path)
        if str(path) in expected and digest != expected[str(path)]:
            raise ReplicationStop("Completed source artifact changed")
        inventory[str(path.relative_to(ROOT))] = digest
    return inventory


class AliasJudge(MigrationJudge):
    def _settle(self, row, payload):
        raw_model = payload.get("model") if isinstance(payload, dict) else None
        normalized = {**payload, "model": ALIASES[0]} if raw_model in ALIASES else payload
        try:
            super()._settle(row, normalized)
        finally:
            # Never rewrite the raw receipt or erase which returned ID was used.
            row["provider_model"] = raw_model if isinstance(raw_model, str) else None
            row["alias_policy_sha256"] = canonical_sha256(POLICY)
            self.save()
        if raw_model not in ALIASES:
            self.state["fatal_error"] = "unapproved_returned_model"
            self.save()
            raise ReplicationStop("Returned model is outside the two explicitly accepted aliases")

    def label(self, response):
        path = self.root / "cache" / (scoring_key(response, self.fingerprint) + ".json")
        if path.is_file():
            cache = read_object(path)
            receipt = Path(cache["receipt"])
            if (not receipt.resolve().is_relative_to((self.root / "receipts").resolve())
                    or read_object(receipt).get("model") not in ALIASES):
                raise ReplicationStop("Cached receipt uses a model outside the accepted aliases")
        return super().label(response)


@dataclass(frozen=True)
class AliasConfig:
    path: Path
    raw: dict
    source: object
    fingerprint: str

    @property
    def manifest(self):
        return self.source.manifest

    @property
    def pair_ids(self):
        return self.source.pair_ids

    @property
    def output_root(self):
        return ROOT / self.raw["output_root"]

    @property
    def runtime_config(self):
        return replace(self.source.runtime_config, path=self.path, name=NAME, raw=self.raw,
                       fingerprint=self.fingerprint, output_root=self.output_root)

    def inspect(self):
        self.source.verify()
        for name, digest in self.raw["bound_inputs"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or file_sha256(path) != digest:
                raise ReplicationStop(f"Alias recovery input changed: {name}")
        if alias_contract(self.source) != self.raw["scoring_contract"]:
            raise ReplicationStop("Alias contract changed")
        parent = RejudgePipeline(self.source)
        inventory = source_snapshot(parent)
        if canonical_sha256(inventory) != self.raw["source_snapshot_sha256"]:
            raise ReplicationStop("Frozen source changed; do not run source and recovery together")
        responses = {phase: parent._phase_data(phase)[1] for phase in ("trajectory_behavior", "oracle_pilot")}
        audit = audit_scoring(self.source.output_root, self.source.judge,
                              self.source.raw["scoring_contract"], self.raw["scoring_contract"],
                              responses, self.raw["historical_accounting"])
        if audit["summary"] != self.raw["offline_audit"]:
            raise ReplicationStop("Source receipt reconciliation changed")
        return inventory, audit

    def verify(self):
        _, audit = self.inspect()
        return {"scope": "conditional_dev_oracle_alias_recovery", "alias_acceptance": POLICY,
                "request_model": self.runtime_config.judge["model"],
                "offline_audit": audit["summary"], "gpu_or_api_started": False}


def draft_config():
    source = load_rejudge_config(ROOT / SOURCE_CONFIG)
    parent = RejudgePipeline(source)
    contract = alias_contract(source)
    inventory = source_snapshot(parent)
    responses = {phase: parent._phase_data(phase)[1] for phase in ("trajectory_behavior", "oracle_pilot")}
    audit = audit_scoring(source.output_root, source.judge, source.raw["scoring_contract"], contract,
                          responses, source.raw["historical_accounting"])
    return {"format": "rq2-v41-alias-recovery", "version": 1, "name": NAME,
            "source_config": SOURCE_CONFIG, "source_snapshot_sha256": canonical_sha256(inventory),
            "output_root": "outputs/stage2_rq2/independent_replication/" + NAME,
            "alias_acceptance": POLICY, "scoring_contract": contract, "budget": BUDGET, "design": DESIGN,
            "historical_accounting": source.raw["historical_accounting"], "offline_audit": audit["summary"],
            "generation_fingerprints": {phase: parent._run_fingerprint(phase) for phase in responses},
            "bound_inputs": {p: file_sha256(ROOT / p) for p in (SOURCE_CONFIG, *IMPLEMENTATION)}}


def load_alias_config(path):
    path = Path(path).resolve()
    raw = read_object(path)
    fields = {"format", "version", "name", "source_config", "source_snapshot_sha256", "output_root",
              "alias_acceptance", "scoring_contract", "budget", "design", "historical_accounting",
              "offline_audit", "generation_fingerprints", "bound_inputs"}
    if (set(raw) != fields or raw["format"] != "rq2-v41-alias-recovery" or raw["version"] != 1
            or raw["name"] != NAME or raw["source_config"] != SOURCE_CONFIG
            or raw["alias_acceptance"] != POLICY or raw["budget"] != BUDGET or raw["design"] != DESIGN
            or raw["output_root"] != "outputs/stage2_rq2/independent_replication/" + NAME
            or set(raw["bound_inputs"]) != {SOURCE_CONFIG, *IMPLEMENTATION}):
        raise ReplicationStop("Not the explicitly accepted two-alias recovery configuration")
    out = ROOT / raw["output_root"]
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ReplicationStop("Recovery output must be a separate non-symlink namespace")
    source = load_rejudge_config(ROOT / SOURCE_CONFIG)
    if raw["historical_accounting"] != source.raw["historical_accounting"]:
        raise ReplicationStop("Original costs cannot be reset")
    parent = RejudgePipeline(source)
    if raw["generation_fingerprints"] != {p: parent._run_fingerprint(p) for p in ("trajectory_behavior", "oracle_pilot")}:
        raise ReplicationStop("Generation identities changed")
    return AliasConfig(path, raw, source, canonical_sha256(raw))


def copy_exact(source, target):
    digest = file_sha256(source)
    if target.exists():
        if file_sha256(target) != digest:
            raise ReplicationStop(f"Partial import differs: {target.name}")
    else:
        atomic_text(target, source.read_text(encoding="utf-8"))
        if file_sha256(target) != digest:
            raise ReplicationStop("Imported bytes differ from source")


class AliasPipeline(ReplicationPipeline):
    def _judge(self):
        return AliasJudge(self.config.output_root / "scoring", self.config.judge,
                          self.spec.raw["scoring_contract"], BUDGET,
                          self.spec.raw["historical_accounting"], transport=self.transport)

    def _run_fingerprint(self, phase):
        return self.spec.raw["generation_fingerprints"][phase]

    def model(self):
        raise ReplicationStop("GPU execution is forbidden in alias recovery")

    @contextmanager
    def source_locks(self):
        with ExitStack() as stack:
            for root in (self.spec.source.output_root, self.spec.source.source.output_root):
                handle = stack.enter_context((root / ".run.lock").open("r"))
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ReplicationStop("A source experiment is active; recovery is blocked") from None
            yield

    def _stage_artifacts(self, stage):
        return (*super()._stage_artifacts(stage), self.config.output_root / "provenance/alias_recovery_audit.json")

    def prepare(self):
        with self.source_locks():
            return self._prepare()

    def _prepare(self):
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
            for attempt in audit["ledger"]["attempts"]:
                rel = f"scoring/receipts/{attempt['number']:06d}.json"
                copy_exact(source / rel, root / rel)
            ledger = copy.deepcopy(audit["ledger"])
            ledger.pop("fatal_error", None)  # Only in the new namespace, after complete receipt audit.
            ledger.update(binding=canonical_sha256({"contract": self.spec.raw["scoring_contract"],
                "budget": remaining_budget(BUDGET, self.spec.raw["historical_accounting"])}),
                provider_model=ALIASES[0], recovery_source_sha256=self.spec.raw["source_snapshot_sha256"],
                resolved_source_error="provider_model_or_reservation_violation", alias_acceptance=POLICY)
            for row in ledger["attempts"]:
                row["source_key"] = row["key"]
                row["key"] = audit["key_map"][row["key"]]
            atomic_json(imported_ledger, ledger)
            new_fp = canonical_sha256(self.spec.raw["scoring_contract"])
            for key, item in audit["successes"].items():
                new_key = audit["key_map"][key]
                receipt = root / "scoring/receipts" / f"{item['number']:06d}.json"
                atomic_json(root / "scoring/cache" / (new_key + ".json"), {
                    "key": new_key, "contract_sha256": new_fp, "status": "ok",
                    "label": {**item["label"], "judge_fingerprint": new_fp},
                    "receipt": str(receipt), "receipt_sha256": item["receipt_sha256"]})
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
                            judge_protocol_fingerprint=new_fp, alias_acceptance=POLICY,
                            labels_rebound_without_rescoring=True)
            atomic_json(root / "provenance/rq1_sources.json", metadata)
            atomic_json(root / "provenance/scoring_contract.json", self.spec.raw["scoring_contract"])
            atomic_json(root / "provenance/source_inventory.json", inventory)
            atomic_json(root / "provenance/alias_recovery_audit.json", {
                "format": "rq2-alias-recovery-audit", "version": 1, "alias_acceptance": POLICY,
                "source_snapshot_sha256": self.spec.raw["source_snapshot_sha256"],
                "source_contract_sha256": old_fp, "effective_contract_sha256": new_fp,
                "generation_baseline_import_references": "historical_source_records_preserved_unchanged",
                "label_change": "fingerprint_only_no_score_changes", "raw_receipts_changed": False,
                "offline_audit": audit["summary"], "imported_labels": imported_counts,
                "gpu_or_api_started": False})
            state["stages"] = {s: {"output": str(self.stage_path(s)), "resources": ["cpu"],
                                   "mode": "imported_under_unverified_alias_assumption",
                                   "artifacts": {str(p): file_sha256(p) for p in self._stage_artifacts(s)}}
                               for s in IMPORTED_STAGES}
            self._write_state(state)
            return {"status": "prepared", "imported_labels": imported_counts,
                    "offline_audit": audit["summary"], "gpu_or_api_started": False}

    def run(self, stages, *, acknowledge_cost=False, accept_unverified_aliases=False):
        if (not stages or any(s not in RUN_STAGES for s in stages)
                or [RUN_STAGES.index(s) for s in stages] != sorted({RUN_STAGES.index(s) for s in stages})):
            raise ReplicationStop("Only Oracle Judge, analysis and report are permitted")
        if not acknowledge_cost or not accept_unverified_aliases:
            raise ReplicationStop("Acknowledge cost and explicitly accept the unverified alias assumption")
        with self.source_locks():
            self._prepare()
            # Reuse the prior operator's dev-only authorization for the same
            # frozen population; this entry cannot generate or introduce data.
            return super().run(stages, acknowledge_cost=True, confirm_no_external_rq2_use=True)

    def _run_oracle_analyze(self):
        super()._run_oracle_analyze()
        path = self.stage_path("oracle_analyze")
        value = read_object(path)
        value.update(alias_acceptance=POLICY, result_scope="conditional_dev_evidence_unverified_judge_alias",
                     requires_provider_confirmation_for_version_equivalence=True)
        value["pilot_decision"]["qualification_is_conditional_on_alias_assumption"] = True
        atomic_json(path, value)
        self._run_report()

    def _run_report(self):
        super()._run_report()
        path = self.stage_path("report")
        warning = ("# 重要限制：Judge 别名等价未经平台确认\n\n"
                   "用户于2026-10-10接受将 deepseek-flash 与 deepseek-v4-1-flash-260910 视为同版本的人工假设。"
                   "本报告只作附带该假设的开发结果；不能声称模型版本已获证实，也不能据此进入formal或产生因果结论。\n\n")
        atomic_text(path, warning + path.read_text(encoding="utf-8"))

    def estimate(self):
        audit = self.spec.raw["offline_audit"]
        average = audit["charged_micro_cny_preserved"] / audit["charged_attempts_preserved"] / 1_000_000
        return {"remaining_unique_inputs_at_recovery": audit["remaining_oracle_unique_inputs"],
                "estimated_first_attempt_cost_cny": round(average * audit["remaining_oracle_unique_inputs"], 4),
                "estimate_basis": "prior_average_usage_and_accounting_rates_not_a_provider_invoice",
                "cumulative_limit_cny": 30, "new_gpu_hours": 0, "alias_acceptance": POLICY}
