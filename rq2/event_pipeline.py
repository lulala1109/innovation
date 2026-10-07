"""Independent v3 dev scheduler; no path to the v2 protocol lock or formal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rq2.analysis import analyze_trial_set, join_effect_rows
from rq2.artifacts import (
    BehaviorLabel, atomic_json, atomic_text, canonical_sha256, file_sha256,
    read_jsonl, records_by_id, validate_trial_record,
)
from rq2.behavior import validate_response_record
from rq2.judge_consistency import scoring_consistency_audit
from rq2.data import derive_behavior_events
from rq2.event_analysis import window_summary
from rq2.event_config import (
    EVENT_CENTER, EVENT_NAME, EVENT_OFFSETS, EVENT_STAGES, SOURCE_FILES,
    EventPilotConfig, EventPilotError, read_object,
)
from rq2.event_data import build_event_index, imported_baseline, replay_audit, seed_phase_baselines
from rq2.experiment import PlannedTrial, build_mechanism_plan, build_oracle_plan
from rq2.pilot import evaluate_dev_pilot, read_pilot_csv
from rq2.pipeline import RQ2Pipeline


class EventPilotPipeline(RQ2Pipeline):
    def __init__(self, spec: EventPilotConfig, **kwargs: Any) -> None:
        self.spec = spec
        super().__init__(spec.runtime_config, **kwargs)
        self._source_cache = None

    def stage_path(self, stage: str) -> Path:
        if stage not in EVENT_STAGES:
            raise EventPilotError("v3 dev forbids protocol_lock, formal, subspace and unknown stages")
        if stage.startswith("window_"):
            return self.config.output_root / "window_pilot" / {
                "window_generate": "responses.jsonl", "window_judge": "labels.jsonl",
                "window_analyze": "analysis.json",
            }[stage]
        if stage == "report":
            return self.config.output_root / "event_dev_report.md"
        return self.config.stage_path(stage)

    @staticmethod
    def resources(stage: str) -> list[str]:
        return ["gpu"] if stage.endswith("_generate") or stage in {"layer_map", "identity"} else ["api"] if stage.endswith("_judge") else ["cpu"]

    @staticmethod
    def dependencies(stage: str) -> tuple[str, ...]:
        if stage == "report":
            return ("oracle_analyze",)
        return EVENT_STAGES[:EVENT_STAGES.index(stage)]

    def plan(self) -> list[dict[str, Any]]:
        completed = self._state()["stages"]
        return [{"stage": stage, "resources": self.resources(stage),
                 "dependencies": list(self.dependencies(stage)[-1:]),
                 "output": str(self.stage_path(stage)),
                 "status": "complete" if stage in completed and self._stage_is_fresh(stage, completed[stage]) else "pending"}
                for stage in EVENT_STAGES]

    def validate(self, *, require_protocol: bool = False) -> dict[str, Any]:
        if require_protocol:
            raise EventPilotError("dev validation cannot require a formal lock")
        screen = self.spec.verify_source()
        _, audit = self._derive_index()
        return {"scope": "event_dev_static", "design_version": 3,
                "config_fingerprint": self.spec.fingerprint,
                "source_decision": screen["decision"], "source_responses": 2040,
                "dev_pair_count": len(self.spec.pair_ids), "formal_pair_count": 0,
                "event_pair_count": audit["event_pair_count"],
                "center_positive_score_pairs": audit["center_positive_score_pairs"],
                "full_window_pair_count": audit["full_window_pair_count"],
                "center_utility_necessary_condition": audit["center_utility_necessary_condition"],
                "allowed_stages": list(EVENT_STAGES), "protocol_locked": False,
                "gpu_or_api_started": False}

    def _effective_manifest(self) -> Path:
        return self.spec.source.output_root / "trajectory/resolved_manifest.json"

    def model(self):
        if self._model_cache is None:
            from core.reproducibility import configure_reproducibility
            mode = str(self.config.trajectory["determinism"])
            settings = configure_reproducibility(int(self.config.trajectory["seed"]),
                                                  "strict" if mode == "error" else mode)
            settings_path = self.config.output_root / "provenance/runtime_reproducibility.json"
            if settings_path.exists() and read_object(settings_path) != settings:
                raise EventPilotError("runtime reproducibility settings changed across processes")
            if not settings_path.exists():
                atomic_json(settings_path, settings)
        return super().model()

    def pairs(self):
        pairs = super().pairs()
        if {p.pair_id for p in pairs} != set(self.spec.pair_ids) or any(p.role != "rq2_dev" for p in pairs):
            raise EventPilotError("resolved runtime pairs must remain exactly the frozen dev population")
        return pairs

    def _source_data(self):
        if self._source_cache is None:
            root = self.spec.source.output_root / "trajectory_behavior"
            trials = [validate_trial_record(r) for r in read_jsonl(root / "trials.jsonl")]
            responses = records_by_id(validate_response_record(r) for r in read_jsonl(root / "responses.jsonl"))
            labels = records_by_id(read_jsonl(root / "labels.jsonl"))
            judge_fp = canonical_sha256({k: v for k, v in self.config.judge.items() if k != "base_url"})
            if set(responses) != {r["trial_id"] for r in trials} or set(labels) != set(responses):
                raise EventPilotError("source response/trial/label coverage differs")
            for trial in trials:
                response, label = responses[trial["trial_id"]], labels[trial["trial_id"]]
                BehaviorLabel(**{k: v for k, v in label.items() if k not in {"format", "version"}})
                if (trial["pair_id"] not in self.spec.pair_ids or trial["rq2_role"] != "rq2_dev"
                    or response["pair_id"] != trial["pair_id"]
                    or response["run_fingerprint"] != trial["run_fingerprint"]
                    or response["response_sha256"] != trial["response_sha256"]
                    or label["response_sha256"] != trial["response_sha256"]
                    or label["judge_fingerprint"] != judge_fp or label["label_status"] != "ok"):
                    raise EventPilotError("source baseline provenance or Judge configuration mismatch")
            self._source_cache = (trials, responses, labels)
        return self._source_cache

    def _derive_events(self) -> dict[str, Any]:
        trials, _, labels = self._source_data()
        return derive_behavior_events(trials, list(labels.values()),
            refusal_weakening_delta=float(self.config.trajectory["refusal_weakening_delta"]),
            event_offsets=EVENT_OFFSETS)

    def _derive_index(self):
        trials, _, labels = self._source_data()
        return build_event_index(
            read_object(self.spec.source.output_root / "trajectory_behavior/scan_index.json"),
            self._derive_events(), read_object(self._effective_manifest())["records"],
            trials, list(labels.values()), pair_ids=self.spec.pair_ids)

    @property
    def population_path(self) -> Path:
        return self.config.output_root / "state_index/event_population.json"

    def population(self) -> dict[str, Any]:
        return read_object(self.population_path)

    def _run_sources(self) -> None:
        source = self.spec.source.output_root
        payload = read_object(source / "provenance/rq1_sources.json")
        payload.update(config_fingerprint=self.config.fingerprint, design_version=3,
                       inherited_source_config_fingerprint=self.spec.source.fingerprint,
                       event_preregistration_sha256=self.spec.raw["preregistration"]["sha256"])
        atomic_json(self.stage_path("sources"), payload)
        atomic_json(self.config.output_root / "provenance/reference_statistics.json",
                    read_object(source / "provenance/reference_statistics.json"))
        atomic_json(self.config.output_root / "provenance/source_import.json", {
            "format": "rq2-event-source-import", "version": 3,
            "source_config": self.spec.raw["source_config"],
            "source_artifacts": self.spec.raw["source_artifacts"],
            "source_output_root": str(source), "dev_pair_ids": list(self.spec.pair_ids),
            "immutable_baselines": True, "new_run_fingerprint": self.spec.fingerprint,
        })

    def _run_events(self) -> None:
        atomic_json(self.stage_path("events"), self._derive_events())

    def _run_state_index(self) -> None:
        index, audit = self._derive_index()
        source_state = read_object(self.spec.source.output_root / "pipeline_state.json")
        bound = source_state["stages"]["trajectory"]["artifacts"]
        # Validate only the frozen event windows, plus each trajectory index/clean.
        selected = {}
        for row in index["records"]:
            selected[row["clean_audio_path"]] = row["clean_audio_sha256"]
            trajectory = row["trajectory_path"]
            if trajectory not in bound:
                raise EventPilotError("trajectory index missing from original source provenance")
            selected[trajectory] = bound[trajectory]
            if row["available"]:
                selected[row["checkpoint_path"]] = row["checkpoint_sha256"]
        for path, digest in selected.items():
            if not Path(path).is_file() or file_sha256(path) != digest:
                raise EventPilotError(f"frozen event input changed: {path}")
        atomic_json(self.stage_path("state_index"), index)
        atomic_json(self.population_path, audit)

    def _sample_planned(self) -> PlannedTrial:
        rows = [r for r in self._state_index()["records"] if r["state_key"] == EVENT_CENTER and r["available"]]
        if not rows:
            raise EventPilotError("no eligible event center for identity checks")
        row = rows[0]
        return PlannedTrial(pair_id=row["pair_id"], rq2_role="rq2_dev", state_key=EVENT_CENTER,
                            step=row["step"], layer=self.config.pilot["candidate_layers"][0],
                            intervention="sham", dose=1.0)

    def _phase_plan(self, phase: str):
        index = self._state_index()
        common = dict(candidate_layers=self.config.pilot["candidate_layers"],
                      neighbor_layers=self.config.pilot["neighbor_layers"],
                      coordinate="event", state_keys=(EVENT_CENTER,))
        if phase == "oracle_pilot":
            return build_oracle_plan(index, **common)
        if phase == "mechanism_pilot":
            return build_mechanism_plan(index, **common,
                restoration_doses=self.config.pilot["restoration_doses"],
                reverse_doses=self.config.pilot["suppression_doses"],
                primary_restoration_dose=1.0, random_replicates=3,
                seed=int(self.config.pilot["seed"]), enable_subspace=False)
        if phase == "window_pilot":
            return tuple(PlannedTrial(pair_id=r["pair_id"], rq2_role="rq2_dev", state_key=r["state_key"],
                step=r["step"], layer=layer, intervention=intervention, dose=1.0)
                for r in index["records"] if r["available"]
                for layer in self.config.pilot["candidate_layers"]
                for intervention in ("r_direction", "sham"))
        raise EventPilotError("unknown dev phase")

    def _run_fingerprint(self, phase: str) -> str:
        return canonical_sha256({"config_fingerprint": self.spec.fingerprint, "phase": phase,
                                 "state_index_sha256": file_sha256(self.stage_path("state_index")),
                                 "source_artifacts": self.spec.raw["source_artifacts"]})

    def _run_generation_phase(self, phase, plan):
        if not plan or any(t.pair_id not in self.population()["event_pair_ids"] or t.rq2_role != "rq2_dev" for t in plan):
            raise EventPilotError("empty plan or non-event/non-dev pair in generation")
        self._assert_plan_reachable(plan)  # Before seeding labels/commits or loading GPU.
        trials, responses, labels = self._source_data()
        sources = {(r["pair_id"], r["state_key"]): r for r in trials}
        unique = {}
        for trial in plan:
            unique.setdefault((trial.pair_id, trial.state_key), trial)
        imports = []
        for item in unique.values():
            key = "scan:clean" if item.target_kind == "clean" else f"scan:{item.step}"
            source = sources[(item.pair_id, key)]
            imports.append(imported_baseline(item, run_fingerprint=self._run_fingerprint(phase),
                source_trial=source, source_response=responses[source["trial_id"]],
                source_label=labels[source["trial_id"]],
                source_binding=canonical_sha256(self.spec.raw["source_artifacts"])))
        seed_phase_baselines(self.config.output_root / phase, imports)
        print(f"[{phase}] imported {len(imports)} frozen baselines; planned {len(plan)} interventions", flush=True)
        super()._run_generation_phase(phase, plan)
        directory = self.config.output_root / phase
        audit = replay_audit(read_jsonl(directory / "trials.jsonl"), plan, self._run_fingerprint(phase))
        atomic_json(directory / "baseline_replay_audit.json", audit)
        if not audit["passed"]:
            raise EventPilotError("baseline response drift detected before Judge; preserve this run and inspect baseline_replay_audit.json")

    def _run_oracle_generate(self) -> None:
        if not self.population()["center_utility_necessary_condition"]:
            raise EventPilotError("event center has insufficient population/restoration opportunity")
        self._run_generation_phase("oracle_pilot", self._phase_plan("oracle_pilot"))

    def _require_qualified(self, stage: str) -> None:
        # Recheck actual labels: a cached positive decision cannot bypass this.
        self._require_consistent_scoring(self.stage_path(stage).parent)
        value = read_object(self.stage_path(stage))
        if value.get("event_pilot_version") != 3 or value.get("pilot_decision", {}).get("qualified") is not True:
            raise EventPilotError(f"{stage} did not pass the v3 event pilot gate; see its decision")
        if value.get("historical_scoring_contract_verified") is not True:
            raise EventPilotError("legacy scoring contract is unverified; independent Judge revision is diagnostic only")

    def _scoring_audit(self, directory: Path):
        return scoring_consistency_audit(
            read_jsonl(directory / "trials.jsonl"),
            read_jsonl(directory / "responses.jsonl"),
            read_jsonl(directory / "labels.jsonl"))

    def _require_consistent_scoring(self, directory: Path) -> None:
        if not self._scoring_audit(directory)["passed"]:
            raise EventPilotError(
                "Judge labels are inconsistent/unresolved; preserve this run and use "
                "experiments/rq2_judge_revision.py for isolated reanalysis")

    def _run_mechanism_generate(self) -> None:
        self._require_qualified("oracle_analyze")
        self._run_generation_phase("mechanism_pilot", self._phase_plan("mechanism_pilot"))

    def _run_window_generate(self) -> None:
        self._require_qualified("mechanism_analyze")
        self._run_generation_phase("window_pilot", self._phase_plan("window_pilot"))

    def _run_judge_phase(self, phase: str) -> None:
        if phase not in {"oracle_pilot", "mechanism_pilot", "window_pilot"}:
            raise EventPilotError("Judge phase outside v3 dev scope")
        # v3 binds only the legacy fingerprint. Never rewrite its frozen labels
        # or spend API quota on new trial-ID-only judgments in this namespace.
        raise EventPilotError(
            "legacy v3 Judge execution is frozen; use experiments/rq2_judge_revision.py "
            "for independent content-keyed scoring, not in-place migration")

    def _validate_legacy_judge_inputs(self, phase: str) -> None:
        directory = self.config.output_root / phase
        planned = {p.to_key(self._run_fingerprint(phase)).trial_id for p in self._phase_plan(phase)}
        responses = records_by_id(read_jsonl(directory / "responses.jsonl"))
        trials = read_jsonl(directory / "trials.jsonl")
        baselines = {r["trial_id"] for r in trials if r["intervention"] == "baseline"}
        if set(responses) != planned | baselines or len(trials) != len(responses):
            raise EventPilotError("generated trials/responses do not cover the exact dev plan")
        for r in trials:
            validate_trial_record(r)
            if (r["rq2_role"] != "rq2_dev" or r["pair_id"] not in self.population()["event_pair_ids"]
                or r["run_fingerprint"] != self._run_fingerprint(phase)
                or responses[r["trial_id"]]["response_sha256"] != r["response_sha256"]):
                raise EventPilotError("Judge inputs violate dev identity or provenance")

    def _run_window_judge(self) -> None:
        self._run_judge_phase("window_pilot")

    def _analyze_phase(self, phase: str, *, event: bool = False):
        directory = self.config.output_root / phase
        self._require_consistent_scoring(directory)
        return analyze_trial_set(directory / "trials.jsonl", directory / "labels.jsonl",
            output_dir=directory, pilot=True, event=True,
            event_pair_ids=self.population()["event_pair_ids"], **self._analysis_kwargs())

    def _pilot_decision(self, phase, intervention, *, minimum_effect, require_mechanism_checks=False):
        directory = self.config.output_root / phase
        summaries = read_pilot_csv(directory / "rq2_causal_map_mean_ci.csv")
        population_count = self.population()["event_pair_count"]
        for row in summaries:
            if row["state_key"] == EVENT_CENTER:
                # A missing Judge result cannot improve the sign denominator.
                row["sign_consistency"] = round(float(row["sign_consistency"]) * int(row["pair_count"])) / population_count
        decision = evaluate_dev_pilot(summaries, intervention=intervention, dose=1.0,
            candidate_layers=self.config.pilot["candidate_layers"], neighbor_layers=self.config.pilot["neighbor_layers"],
            fixed_steps=(), event_state=EVENT_CENTER, minimum_effect=minimum_effect,
            controls=read_pilot_csv(directory / "rq2_specificity_controls.csv") if require_mechanism_checks else (),
            require_mechanism_checks=require_mechanism_checks)
        audit = replay_audit(read_jsonl(directory / "trials.jsonl"), self._phase_plan(phase), self._run_fingerprint(phase))
        scoring_audit = self._scoring_audit(directory)
        decision.update(event_pair_count=population_count, baseline_replay_audit=audit,
                        scoring_consistency_audit=scoring_audit,
                        sign_denominator="all_eligible_event_pairs", causal_evidence=False)
        if not audit["passed"] or not scoring_audit["passed"]:
            decision.update(qualified=False, qualifying_regions=[], blocked_reason=(
                "baseline_replay_mismatch" if not audit["passed"] else "scoring_consistency_failed"))
            for region in decision["regions"]:
                region["eligible"] = False
        return decision

    def _finish_analysis(self, stage: str) -> None:
        payload = read_object(self.stage_path(stage))
        payload.update(event_pilot_version=3, event_centered=True, causal_evidence=False,
                       config_fingerprint=self.spec.fingerprint,
                       event_preregistration_sha256=self.spec.raw["preregistration"]["sha256"],
                       population=self.population(), subspace_eligible=False,
                       automatically_launches_formal=False)
        atomic_json(self.stage_path(stage), payload)
        self._run_report()

    def _run_oracle_analyze(self) -> None:
        super()._run_oracle_analyze()
        self._finish_analysis("oracle_analyze")

    def _run_mechanism_analyze(self) -> None:
        super()._run_mechanism_analyze()
        self._finish_analysis("mechanism_analyze")

    def _run_window_analyze(self) -> None:
        summary = dict(self._analyze_phase("window_pilot"))
        directory = self.config.output_root / "window_pilot"
        trials = read_jsonl(directory / "trials.jsonl")
        effects, _ = join_effect_rows(trials, read_jsonl(directory / "labels.jsonl"))
        population = self.population()
        window = window_summary(effects, event_pair_ids=population["event_pair_ids"],
            full_window_pair_ids=population["full_window_pair_ids"],
            candidate_layers=self.config.pilot["candidate_layers"],
            replicates=int(self.config.statistics["bootstrap_replicates"]),
            confidence=float(self.config.statistics["confidence"]), seed=int(self.config.statistics["seed"]))
        window["baseline_replay_audit"] = replay_audit(trials, self._phase_plan("window_pilot"), self._run_fingerprint("window_pilot"))
        atomic_json(directory / "rq2_event_window_descriptive.json", window)
        summary.update(window=window)
        atomic_json(self.stage_path("window_analyze"), summary)
        self._finish_analysis("window_analyze")

    def _run_report(self) -> None:
        population = self.population()
        lines = ["# RQ2 v3 事件中心 dev pilot", "", "仅用于开发阶段准入判断；未进行 formal，也不产生因果结论。", "",
                 f"事件中心人口：{population['event_pair_count']} / 20；完整窗口人口：{population['full_window_pair_count']}。",
                 f"中心正分数人数：{population['center_positive_score_pairs']}；零分事件 pair 保留。", ""]
        for stage, name in (("oracle_analyze", "Oracle"), ("mechanism_analyze", "R 机制")):
            path = self.stage_path(stage)
            if not path.is_file():
                lines.append(f"{name}：尚未完成。\n")
                continue
            result = read_object(path)
            decision = result["pilot_decision"]
            qualified = decision["qualified"]
            audit = decision["baseline_replay_audit"]
            lines.append(f"{name}：{'满足下一阶段准备门槛' if qualified else '未满足推进门槛'}；"
                         f"重放一致性 {'通过' if audit['passed'] else '失败'}，检查 {audit['checked']} 条。\n")
            if not qualified:
                lines.append("查看 analysis.json 的 regions：样本数、固定事件人口同向比例、邻层支持、拒答和对照审计分别列出。\n")
        lines.extend(["formal：本入口禁止执行。独立 40-pair run 尚需事件版正式执行器、冻结检验网格和有效事件 N≥20 门禁。", "",
                      "窗口：属于同事件共同人口的描述分析；事件后自然恢复和事件定义本身不构成干预或状态依赖证据。", "",
                      "subspace：未授权执行；Oracle 有效而 R 弱时需另订 calibration/算子协议和新 run。", ""])
        atomic_text(self.stage_path("report"), "\n".join(lines))

    def _stage_artifacts(self, stage: str):
        primary = self.stage_path(stage)
        paths = [primary]
        if stage == "sources":
            paths.extend([self.spec.path, self.spec.source.path, self.spec.protocol_path, self.spec.preregistration_path,
                          self.spec.source.manifest, self.config.output_root / "provenance/reference_statistics.json",
                          self.config.output_root / "provenance/source_import.json"])
            paths.extend(self.spec.source.output_root / name for name in SOURCE_FILES)
        if stage in {"layer_map", "identity"} or stage.endswith("_generate"):
            paths.append(self.config.output_root / "provenance/runtime_reproducibility.json")
        if stage == "state_index":
            paths.append(self.population_path)
            for row in self._state_index()["records"]:
                paths.extend([Path(row["trajectory_path"]), Path(row["clean_audio_path"])])
                if row["available"]:
                    paths.append(Path(row["checkpoint_path"]))
        if stage.endswith("_generate"):
            paths.extend(primary.parent / name for name in ("trials.jsonl", "commits.jsonl", "baseline_import_map.json", "baseline_replay_audit.json"))
        if stage.endswith("_analyze"):
            paths.extend(sorted(primary.parent.glob("rq2_*")))
        return tuple(dict.fromkeys(paths))

    def _stage_is_fresh(self, stage, record):
        artifacts = record.get("artifacts", {})
        if str(self.stage_path(stage)) not in artifacts or not all(Path(p).is_file() and file_sha256(p) == h for p, h in artifacts.items()):
            return False
        if stage.endswith("_analyze"):
            value = read_object(self.stage_path(stage))
            return value.get("event_pilot_version") == 3 and value.get("analysis_scope") == "dev_event_pilot"
        return True

    def run(self, stages: Sequence[str]) -> dict[str, Any]:
        if not stages or any(s not in EVENT_STAGES for s in stages):
            raise EventPilotError("v3 dev forbids protocol_lock, formal, subspace and unknown stages")
        if not self.spec.raw["execution_enabled"]:
            raise EventPilotError("execution is disabled")
        self.spec.verify_source()
        state = self._state()
        completed = dict(state["stages"])
        fresh = {}
        for stage in stages:
            for dependency in self.dependencies(stage):
                if dependency not in fresh:
                    fresh[dependency] = dependency in completed and self._stage_is_fresh(dependency, completed[dependency])
                if not fresh[dependency]:
                    raise EventPilotError(f"{stage} requires completed, unchanged {dependency}")
            if stage in completed and stage != "report":
                if not self._stage_is_fresh(stage, completed[stage]):
                    raise EventPilotError(f"completed stage changed: {stage}")
                fresh[stage] = True
                continue
            print(f"[event_dev] {stage} starting ({','.join(self.resources(stage))})", flush=True)
            getattr(self, f"_run_{stage}")()
            paths = self._stage_artifacts(stage)
            if any(not p.is_file() for p in paths):
                raise EventPilotError(f"{stage} did not produce all required artifacts")
            completed[stage] = {"output": str(self.stage_path(stage)),
                                "sha256": file_sha256(self.stage_path(stage)),
                                "artifacts": {str(p): file_sha256(p) for p in paths},
                                "resources": self.resources(stage)}
            self._write_state({**state, "stages": completed})
            fresh[stage] = True
            print(f"[event_dev] {stage} complete", flush=True)
        return self.status()
