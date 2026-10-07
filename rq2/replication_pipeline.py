"""Independent baseline -> manual checkpoint -> Oracle scheduler, no formal path."""
from __future__ import annotations

import fcntl
import gc
import json
from pathlib import Path

from rq2.artifacts import (atomic_json, atomic_jsonl, atomic_text, canonical_sha256,
                           file_sha256, read_jsonl, records_by_id, validate_trial_record)
from rq2.behavior import validate_response_record
from rq2.data import derive_behavior_events
from rq2.event_config import EVENT_CENTER
from rq2.event_data import build_event_index, imported_baseline, replay_audit, seed_phase_baselines
from rq2.event_pipeline import EventPilotPipeline
from rq2.experiment import build_oracle_plan
from rq2.judge_consistency import checked_label, scoring_consistency_audit
from rq2.pipeline import RQ2Pipeline
from rq2.replication_config import STAGES, BASELINE_STAGES, ORACLE_STAGES, read_object
from rq2.replication_judge import ContentJudge, ReplicationStop, utc_now
from rq2.replication_progress import StageProgress, line_count


class ReplicationPipeline(RQ2Pipeline):
    def __init__(self, spec, *, transport=None, **kwargs):
        self.spec, self.transport = spec, transport
        super().__init__(spec.runtime_config, **kwargs)

    def stage_path(self, stage):
        if stage not in STAGES:
            raise ReplicationStop("Replication forbids mechanism/window/subspace/protocol_lock/formal")
        return self.config.stage_path(stage)

    @property
    def population_path(self):
        return self.config.output_root / "state_index/event_population.json"

    def population(self):
        return read_object(self.population_path)

    @staticmethod
    def resources(stage):
        if stage in ("trajectory", "layer_map", "identity") or stage.endswith("_generate"):
            return ["gpu"]
        return ["api"] if stage.endswith("_judge") else ["cpu"]

    def _stage_is_fresh(self, stage, record):
        artifacts = record.get("artifacts", {})
        return str(self.stage_path(stage)) in artifacts and all(
            Path(p).is_file() and file_sha256(p) == h for p, h in artifacts.items())

    def plan(self):
        completed = self._state()["stages"]
        return [{"stage": s, "resources": self.resources(s),
                 "dependencies": list(STAGES[:i][-1:]), "output": str(self.stage_path(s)),
                 "status": "complete" if s in completed and self._stage_is_fresh(s, completed[s]) else "pending"}
                for i, s in enumerate(STAGES)]

    def validate(self, **kwargs):
        result = self.spec.verify()
        result.update(protocol_locked=False, actual_experiment_started=self.state_path.exists(),
                      baseline_stop="state_index", oracle_requires_separate_command=True)
        return result

    def status(self):
        result = {"run_name": self.config.name, "stages": self.plan(),
                  "allow_downstream_execution": False}
        ledger = self.config.output_root / "scoring/ledger.json"
        if ledger.exists():
            result["api_budget"] = self._judge().summary()
        for name in ("progress.json", "stop_reason.json"):
            path = self.config.output_root / name
            if path.exists():
                result[name] = read_object(path)
        hours = {"gpu": 0.0, "api": 0.0, "cpu": 0.0}
        for path in (self.config.output_root / "timings").glob("*.json"):
            item = read_object(path)
            resource = self.resources(item["stage"])[0]
            hours[resource] += item["elapsed_seconds"] / 3600
        result["recorded_stage_elapsed_hours"] = hours
        result["timing_note"] = "Includes completed/stopped attempts; not provider rental invoice or GPU utilization time"
        return result

    def _judge(self):
        return ContentJudge(self.config.output_root / "scoring", self.config.judge,
                            self.spec.raw["scoring_contract"], self.spec.raw["budget"],
                            transport=self.transport)

    def pairs(self):
        pairs = super().pairs()
        if len(pairs) != 20 or {p.pair_id for p in pairs} != set(self.spec.pair_ids) or any(p.role != "rq2_dev" for p in pairs):
            raise ReplicationStop("Resolved population is not exactly the selected 20 dev pairs")
        return pairs

    def model(self):
        if self._model_cache is None:
            from core.reproducibility import configure_reproducibility
            settings = configure_reproducibility(int(self.config.trajectory["seed"]), "warn")
            path = self.config.output_root / "provenance/runtime_reproducibility.json"
            if path.exists() and read_object(path) != settings:
                raise ReplicationStop("Reproducibility settings changed")
            if not path.exists():
                atomic_json(path, settings)
        return super().model()

    def _run_sources(self):
        super()._run_sources()
        path = self.stage_path("sources")
        payload = read_object(path)
        payload["inherited_rq1_layer_reference"] = payload.pop("candidate_layer_preregistration")
        payload.update(design_version=4, design=self.spec.raw["design"],
                       selection_manifest=self.spec.raw["selection_manifest"],
                       selected_pair_ids=list(self.spec.pair_ids),
                       judge_protocol_fingerprint=canonical_sha256(self.spec.raw["scoring_contract"]))
        atomic_json(path, payload)
        atomic_json(self.config.output_root / "provenance/scoring_contract.json", self.spec.raw["scoring_contract"])

    def _write_resolved_manifest(self, summary):
        # Batch resume labels completed trajectories as 'skipped'. They must not vanish.
        normalized = {**summary, "cases": [
            {**row, "status": "completed"} if row["status"] == "skipped" else row
            for row in summary["cases"]]}
        super()._write_resolved_manifest(normalized)

    def _run_fingerprint(self, phase):
        payload = {"config": self.spec.fingerprint, "phase": phase, "design_version": 4}
        if phase == "oracle_pilot":
            payload["state_index_sha256"] = file_sha256(self.stage_path("state_index"))
        return canonical_sha256(payload)

    def _phase_data(self, phase):
        directory = self.config.output_root / phase
        trials = [validate_trial_record(r) for r in read_jsonl(directory / "trials.jsonl")]
        responses = records_by_id(validate_response_record(r) for r in read_jsonl(directory / "responses.jsonl"))
        by_trial = records_by_id(trials)
        if set(by_trial) != set(responses):
            raise ReplicationStop("Trial/response coverage differs")
        instructions = {p.pair_id: p.harmful_text for p in self.pairs()}
        for row in trials:
            response = responses[row["trial_id"]]
            if (row["pair_id"] not in self.spec.pair_ids or row["rq2_role"] != "rq2_dev"
                or row["run_fingerprint"] != self._run_fingerprint(phase)
                or response["pair_id"] != row["pair_id"]
                or response["run_fingerprint"] != row["run_fingerprint"]
                or response["harmful_text"] != instructions[row["pair_id"]]
                or response["response_sha256"] != row["response_sha256"]):
                raise ReplicationStop("Generated response provenance differs from frozen population")
        if phase == "trajectory_behavior":
            scan = read_object(directory / "scan_index.json")
            expected = {(r["pair_id"], r["state_key"]) for r in scan["records"]}
            if len(trials) != 2040 or len(expected) != 2040 or {(r["pair_id"], r["state_key"]) for r in trials} != expected:
                raise ReplicationStop("Baseline scan does not cover exactly 20 x 102 states")
            if any(r["intervention"] != "trajectory_baseline" for r in trials):
                raise ReplicationStop("Non-baseline record in scan")
        else:
            plan = self._phase_plan(phase)
            planned = {p.to_key(self._run_fingerprint(phase)).trial_id for p in plan}
            baselines = {r["trial_id"] for r in trials if r["intervention"] == "baseline"}
            if set(responses) != planned | baselines or len(baselines) != self.population()["event_pair_count"]:
                raise ReplicationStop("Oracle grid coverage differs from frozen plan")
        return trials, responses

    def _source_data(self):
        trials, responses = self._phase_data("trajectory_behavior")
        labels = records_by_id(read_jsonl(self.config.output_root / "trajectory_behavior/labels.jsonl"))
        if set(labels) != set(responses):
            raise ReplicationStop("Incomplete baseline labels")
        fp = canonical_sha256(self.spec.raw["scoring_contract"])
        for tid, response in responses.items():
            checked_label(labels[tid], response, self.config.judge, fp)
            if labels[tid]["label_status"] != "ok":
                raise ReplicationStop("Unresolved baseline label; event definition cannot skip it")
        return trials, responses, labels

    def _run_judge_phase(self, phase):
        if phase not in {"trajectory_behavior", "oracle_pilot"}:
            raise ReplicationStop("Judge phase outside replication scope")
        _, responses = self._phase_data(phase)
        directory = self.config.output_root / phase
        labels = records_by_id(read_jsonl(directory / "labels.jsonl", missing_ok=True))
        if not set(labels) <= set(responses):
            raise ReplicationStop("Labels contain unexpected trial IDs")
        judge = self._judge()
        for tid, response in responses.items():
            expected = judge.label(response)  # Reuses content, across all stages and trial IDs.
            if tid in labels and labels[tid] != expected:
                raise ReplicationStop("Existing label conflicts with content cache")
            if tid not in labels:
                labels[tid] = expected
                atomic_jsonl(directory / "labels.jsonl", labels.values())
            if len(labels) % 25 == 0:
                print(f"[judge budget] {json.dumps(judge.summary(), ensure_ascii=False)}", flush=True)
        self._require_consistent_scoring(directory)
        atomic_json(directory / "scoring_consistency_audit.json", self._scoring_audit(directory))

    _scoring_audit = EventPilotPipeline._scoring_audit
    def _require_consistent_scoring(self, directory):
        if not self._scoring_audit(directory)["passed"]:
            raise ReplicationStop("Replication scoring unresolved/inconsistent; preserve cache and receipts, do not use v3 revision")

    def _run_events(self):
        trials, _, labels = self._source_data()
        events = derive_behavior_events(trials, list(labels.values()),
                                        refusal_weakening_delta=0.2, event_offsets=(0,))
        atomic_json(self.stage_path("events"), events)

    def _run_state_index(self):
        trials, _, labels = self._source_data()
        index, population = build_event_index(
            read_object(self.config.output_root / "trajectory_behavior/scan_index.json"),
            read_object(self.stage_path("events")), read_object(self._effective_manifest())["records"],
            trials, list(labels.values()), pair_ids=self.spec.pair_ids)
        index.update(design_version=4, event_offsets=[0],
                     records=[r for r in index["records"] if r["state_key"] == EVENT_CENTER])
        index.pop("fingerprint", None)
        index["fingerprint"] = canonical_sha256(index)
        population.update(version=4, minimum_event_pairs=16, allow_downstream_execution=False,
                          oracle_population_ready=population["event_pair_count"] >= 16)
        atomic_json(self.stage_path("state_index"), index)
        atomic_json(self.population_path, population)
        print(f"[population] {population['event_pair_count']}/20 eligible event pairs; Oracle requires >=16", flush=True)

    _sample_planned = EventPilotPipeline._sample_planned

    def _phase_plan(self, phase):
        if phase != "oracle_pilot":
            raise ReplicationStop("Only Oracle is implemented/authorized in this entry")
        plan = build_oracle_plan(self._state_index(), candidate_layers=[19,24,26],
                                neighbor_layers=[18,20,23,25], coordinate="event", state_keys=(EVENT_CENTER,))
        if len(plan) != 16 * self.population()["event_pair_count"] or len(plan) > 320:
            raise ReplicationStop("Oracle plan exceeds or differs from frozen grid")
        return plan

    def _run_oracle_generate(self):
        plan = self._phase_plan("oracle_pilot")
        self._assert_plan_reachable(plan)
        trials, responses, labels = self._source_data()
        sources = {(r["pair_id"], r["state_key"]): r for r in trials}
        unique = {p.pair_id: p for p in plan}
        binding = file_sha256(self.stage_path("trajectory_behavior_judge"))
        imports = []
        for item in unique.values():
            source = sources[(item.pair_id, f"scan:{item.step}")]
            imports.append(imported_baseline(item, run_fingerprint=self._run_fingerprint("oracle_pilot"),
                source_trial=source, source_response=responses[source["trial_id"]],
                source_label=labels[source["trial_id"]], source_binding=binding))
        directory = self.config.output_root / "oracle_pilot"
        seed_phase_baselines(directory, imports)
        super()._run_generation_phase("oracle_pilot", plan)
        self._phase_data("oracle_pilot")
        audit = replay_audit(read_jsonl(directory / "trials.jsonl"), plan, self._run_fingerprint("oracle_pilot"))
        atomic_json(directory / "baseline_replay_audit.json", audit)
        if not audit["passed"]:
            raise ReplicationStop("No-op replay drift; do not spend on Judge")

    _analyze_phase = EventPilotPipeline._analyze_phase
    _pilot_decision = EventPilotPipeline._pilot_decision

    def _run_oracle_analyze(self):
        self._phase_data("oracle_pilot")
        super()._run_oracle_analyze()
        path = self.stage_path("oracle_analyze")
        result = read_object(path)
        result.update(design_version=4, causal_evidence=False, allow_downstream_execution=False,
                      subspace_eligible=False, protocol_locked=False, population=self.population(),
                      budget=self._judge().summary(), decision_scope="independent_dev_oracle_only")
        atomic_json(path, result)
        self._run_report()

    def _run_report(self):
        result = read_object(self.stage_path("oracle_analyze"))
        decision = result["pilot_decision"]
        atomic_text(self.stage_path("report"), "# RQ2 独立开发复验 v4\n\n"
                    f"事件人口：{self.population()['event_pair_count']}/20。\n\n"
                    f"Oracle 门槛：{'通过，仅可讨论下一轮机制开发' if decision['qualified'] else '未通过，本轮结束'}。\n\n"
                    "未执行机制、窗口、subspace 或 formal；不产生因果结论。\n\n"
                    f"API 预算账本（高峰价、未计平台缓存优惠）：{json.dumps(result['budget'], ensure_ascii=False)}\n")

    def _stage_artifacts(self, stage):
        if stage == "report":
            return (self.stage_path(stage), self.stage_path("oracle_analyze"))
        if stage == "sources":
            return (self.stage_path(stage), self.spec.path, self.config.manifest,
                    self.config.output_root / "provenance/reference_statistics.json",
                    self.config.output_root / "provenance/scoring_contract.json")
        paths = list(super()._stage_artifacts(stage))
        if stage == "state_index":
            paths.append(self.population_path)
        if stage.endswith("_judge"):
            paths.append(self.stage_path(stage).parent / "scoring_consistency_audit.json")
        if stage == "oracle_generate":
            paths.extend(self.stage_path(stage).parent / p for p in ("baseline_import_map.json", "baseline_replay_audit.json"))
        return tuple(dict.fromkeys(paths))

    def _counter(self, stage):
        root = self.config.output_root
        if stage == "trajectory":
            return lambda: sum(1 for _ in (root / "trajectory/attacks").glob("*/trajectory/step_*.pt")), 2020
        if stage == "trajectory_behavior_generate":
            return lambda: line_count(root / "trajectory_behavior/commits.jsonl"), 2040
        if stage == "oracle_generate":
            return lambda: line_count(root / "oracle_pilot/commits.jsonl"), 17 * self.population()["event_pair_count"]
        if stage.endswith("_judge"):
            total = 2040 if stage.startswith("trajectory") else 17 * self.population()["event_pair_count"]
            return lambda: line_count(self.stage_path(stage)), total
        return lambda: int(self.stage_path(stage).is_file()), 1

    def run(self, stages, *, acknowledge_cost=False, confirm_no_external_rq2_use=False):
        if not stages or any(s not in STAGES for s in stages):
            raise ReplicationStop("Only the baseline and Oracle stages are allowed")
        positions = [STAGES.index(s) for s in stages]
        if positions != sorted(set(positions)):
            raise ReplicationStop("Stages must be unique and ordered")
        if any(s in BASELINE_STAGES for s in stages) and any(s in ORACLE_STAGES for s in stages):
            raise ReplicationStop("Stop at state_index first; Oracle requires a separate command")
        if not acknowledge_cost or not confirm_no_external_rq2_use:
            raise ReplicationStop("Run requires cost acknowledgement and confirmation of no external RQ2 use")
        self.spec.verify()
        root = self.config.output_root
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".run.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReplicationStop("Another replication process owns this run") from None
            atomic_json(root / "execution_acknowledgement.json", {
                "config_fingerprint": self.spec.fingerprint, "at": utc_now(),
                "cost_acknowledged": True, "no_external_rq2_use_confirmed_by_operator": True,
                "budget": self.spec.raw["budget"], "stages": list(stages)})
            state = self._state()
            completed = dict(state["stages"])
            fresh = set()
            try:
                for stage in stages:
                    for dependency in STAGES[:STAGES.index(stage)]:
                        if dependency in fresh:
                            continue
                        if dependency not in completed or not self._stage_is_fresh(dependency, completed[dependency]):
                            raise ReplicationStop(f"Missing/changed predecessor: {dependency}")
                        fresh.add(dependency)
                    if stage in completed:
                        if not self._stage_is_fresh(stage, completed[stage]):
                            raise ReplicationStop(f"Completed artifact changed: {stage}")
                        fresh.add(stage)
                        print(f"[{stage}] already complete; verified, skipped", flush=True)
                        continue
                    if stage in ORACLE_STAGES and self.population()["event_pair_count"] < 16:
                        raise ReplicationStop("Fewer than 16 eligible event pairs; no Oracle, no replacement")
                    counter, total = self._counter(stage)
                    with StageProgress(root, stage, counter, total):
                        getattr(self, f"_run_{stage}")()
                    artifacts = self._stage_artifacts(stage)
                    if any(not p.is_file() for p in artifacts):
                        raise ReplicationStop("Stage omitted a required artifact")
                    completed[stage] = {"output": str(self.stage_path(stage)), "resources": self.resources(stage),
                                        "artifacts": {str(p): file_sha256(p) for p in artifacts}}
                    self._write_state({**state, "stages": completed})
                    fresh.add(stage)
                    if "gpu" in self.resources(stage):
                        self._runtime_cache = self._model_cache = None
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                atomic_json(root / "stop_reason.json", {"status": "requested_stages_complete", "at": utc_now(),
                                                       "auto_launch_next_stage": False})
            except BaseException as exc:
                # Exceptions raised here are domain messages; never write HTTP headers/body/errors.
                atomic_json(root / "stop_reason.json", {"status": "stopped", "at": utc_now(),
                    "exception_type": type(exc).__name__,
                    "reason": str(exc) if isinstance(exc, ReplicationStop) else "Inspect terminal; raw error omitted for privacy",
                    "auto_retry": False})
                raise
        return self.status()
