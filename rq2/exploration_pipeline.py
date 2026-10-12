"""Isolated exploratory mechanism/window execution on the existing twenty dev pairs."""
import fcntl
import gc
import json
from pathlib import Path
import shutil

from rq2.artifacts import (atomic_csv, atomic_json, atomic_jsonl, atomic_text,
                           canonical_sha256, file_sha256, read_jsonl, records_by_id,
                           TrialKey, validate_trial_record)
from rq2.behavior import validate_response_record
from rq2.dev_judge_resume import DevWarningPipeline, identity_summary
from rq2.event_data import imported_baseline, replay_audit, seed_phase_baselines
from rq2.experiment import QwenPromptCache, run_planned_trials
from rq2.exploration_analysis import analyze, window_contrasts
from rq2.exploration_config import (DESIGN, ROOT, STAGES, event_index, inventory,
                                    phase_plan, source_locks)
from rq2.judge_consistency import checked_label, scoring_consistency_audit, scoring_key
from rq2.pipeline import RQ2Pipeline
from rq2.reachability import require_plan_reachable
from rq2.replication_config import read_object
from rq2.replication_judge import ReplicationStop, utc_now
from rq2.replication_progress import StageProgress, line_count
from rq2.replication_rejudge import MigrationJudge


class FrozenEventPromptCache(QwenPromptCache):
    """Use only checkpoint paths/hashes already verified for this event window."""
    def __init__(self, model, pairs, index):
        super().__init__(model, pairs)
        for pair in pairs:
            selected = [r for r in index["records"] if r["pair_id"] == pair.pair_id and r["available"]]
            self._checkpoints[pair.pair_id] = {r["step"]: Path(r["checkpoint_path"]) for r in selected}
            self._checkpoint_hashes.update({(pair.pair_id, r["step"]): r["checkpoint_sha256"] for r in selected})

    def _checkpoint_paths(self, pair):
        if pair.pair_id not in self._checkpoints:
            raise ReplicationStop("Pair is outside the verified checkpoint window")
        return self._checkpoints[pair.pair_id]


class ExplorationJudge(MigrationJudge):
    """New ledger; old cache is read-only and its original paid receipts stay bound."""
    def __init__(self, root, judge, contract, budget, history, *, source_judge, transport=None):
        self.source_judge = source_judge
        super().__init__(root, judge, contract, budget, history, transport=transport)

    def _settle(self, row, payload):
        model = payload.get("model") if isinstance(payload, dict) else None
        row["provider_model"] = model if isinstance(model, str) and model else None
        row["returned_model_raw"] = model
        usage = payload.get("usage") if isinstance(payload, dict) else None
        valid = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                  for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if not valid or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            row.update(status="uncertain", error="invalid_or_missing_usage")
            self.save()
            raise ReplicationStop("Invalid usage: keep reservation, no automatic retry")
        p, c = usage["prompt_tokens"], usage["completion_tokens"]
        row.update(status="settled", finished_at=utc_now(), prompt_tokens=p, completion_tokens=c,
                   cost_micro_cny=p * self.budget["input_cny_per_million"] + c * self.budget["output_cny_per_million"])
        previous = self.state.get("provider_model")
        if not row["provider_model"] or row["provider_model"] != previous:
            row["identity_warning"] = "unlocked_service_model_identity"
            print("[judge identity warning] " + json.dumps({"attempt": row["number"],
                  "returned_model": row["provider_model"], "action": "record_continue_exploration_only"}), flush=True)
        self.state.update(provider_model=row["provider_model"], provider_model_is_pinned=False)
        if p > row["prompt_token_reservation"] or c > 800:
            self.state["fatal_error"] = "usage_exceeded_reservation"
        self.save()
        atomic_json(self.root / "model_identity_audit.json", identity_summary(self.state["attempts"]))
        if self.state.get("fatal_error"):
            raise ReplicationStop("Actual usage exceeded reservation; accounted and stopped")
        if self.summary()["accounted_cny_at_peak_rates"] >= self.budget["api_limit_cny"]:
            raise ReplicationStop("Cumulative API allowance reached")

    def imported_label(self, response):
        old_key = scoring_key(response, self.source_judge.fingerprint)
        old_path = self.source_judge.root / "cache" / (old_key + ".json")
        if not old_path.is_file():
            return None
        # ContentJudge.label on an existing cache only validates its original receipt.
        old = self.source_judge.label(response)
        label = {**old, "judge_fingerprint": self.fingerprint}
        return checked_label(label, response, self.judge, self.fingerprint)

    def label(self, response):
        if self.state.get("fatal_error") or any(a["status"] != "settled" for a in self.state["attempts"]):
            raise ReplicationStop("Unresolved local API/accounting state; no automatic continuation")
        imported = self.imported_label(response)
        if imported is not None:
            key = scoring_key(response, self.fingerprint)
            old_key = scoring_key(response, self.source_judge.fingerprint)
            path = self.source_judge.root / "cache" / (old_key + ".json")
            audit = {"key": key, "source_key": old_key, "source_cache": str(path),
                     "source_cache_sha256": file_sha256(path),
                     "source_contract_sha256": self.source_judge.fingerprint,
                     "new_contract_sha256": self.fingerprint,
                     "policy": "verified_same_service_request_rubric_content_no_regrading"}
            target = self.root / "imported_cache" / (key + ".json")
            if target.exists() and read_object(target) != audit:
                raise ReplicationStop("Imported scoring provenance changed")
            if not target.exists():
                atomic_json(target, audit)
            return imported
        return super().label(response)


class ExplorationPipeline(RQ2Pipeline):
    def __init__(self, spec, *, transport=None, **kwargs):
        self.spec, self.transport = spec, transport
        # Supplying a transport that always fails is a second barrier against source writes/API.
        def forbidden(body):
            raise ReplicationStop("The historical source cannot issue API calls")
        self.parent = DevWarningPipeline(spec.source, transport=forbidden)
        self.index = self.population = None
        super().__init__(spec.runtime_config, **kwargs)

    def _effective_manifest(self):
        return self.parent._effective_manifest()

    def pairs(self):
        pairs = super().pairs()
        if (len(pairs) != 20 or {p.pair_id for p in pairs} != set(self.spec.raw["pair_ids"])
                or any(p.role != "rq2_dev" for p in pairs)):
            raise ReplicationStop("Exploration must use exactly the original twenty dev pairs")
        return pairs

    def verify(self):
        frozen_source = inventory(self.parent)
        if (canonical_sha256(frozen_source) != self.spec.raw["source_inventory_sha256"]
                or len(frozen_source) != self.spec.raw["source_inventory_count"]):
            raise ReplicationStop("Completed Oracle/source cache changed; do not overwrite or silently rebind")
        if read_object(self.parent.config.stage_path("oracle_analyze")).get("oracle_effective") is not False:
            raise ReplicationStop("This continuation is bound to the recorded negative Oracle result")
        self.index, self.population = event_index(self.parent)
        if (canonical_sha256(self.index) != self.spec.raw["event_index_sha256"]
                or self.population["full_window_pair_ids"] != self.spec.raw["window_pair_ids"]):
            raise ReplicationStop("Event/window population changed")
        self.pairs()
        summary = self.parent._judge().summary()
        if (summary["unresolved_attempts"] or self.parent._judge().state.get("fatal_error")
                or self.spec.raw["history"] != {
                    "cost_micro_cny": round(summary["cumulative_accounted_cny_at_peak_rates"] * 1e6),
                    "http_attempts": summary["cumulative_http_attempts"]}):
            raise ReplicationStop("Historical accounting changed/unresolved")
        mapping = read_object(self.parent.config.stage_path("layer_map"))
        if read_object(self.parent.config.stage_path("identity")).get("passed") is not True:
            raise ReplicationStop("Source identity did not pass")
        for phase in ("mechanism", "window"):
            require_plan_reachable(self.plan_phase(phase), mapping)
        self.verified = {"scope": "post_oracle_exploratory_dev", "pair_count": 20,
                "window_pair_count": self.population["full_window_pair_count"],
                "causal_test_pairs": 0, "original_oracle_decision_unchanged": True,
                "formal_allowed": False, "protocol_locked": False,
                "scoring_service_compatible": True, "model_version_locked": False,
                "estimate": self.spec.raw["estimate"]}
        return self.verified

    def _judge(self):
        return ExplorationJudge(self.config.output_root / "scoring", self.config.judge,
                   self.spec.raw["scoring_contract"], self.spec.raw["budget"], self.spec.raw["history"],
                   source_judge=self.parent._judge(), transport=self.transport)

    def plan_phase(self, phase):
        if self.index is None:
            self.index, self.population = event_index(self.parent)
        return phase_plan(self.index, self.population, phase)

    def _run_fingerprint(self, phase):
        return canonical_sha256({"exploration_config": self.spec.fingerprint, "phase": phase})

    def stage_path(self, stage):
        if stage not in STAGES:
            raise ReplicationStop("No PGD, Oracle regeneration, protocol_lock, formal or subspace in this entry")
        root = self.config.output_root
        if stage == "prepare":
            return root / "provenance/preparation.json"
        if stage == "report":
            return root / "rq2_exploratory_report.md"
        phase, action = stage.split("_")
        return root / phase / {"generate": "responses.jsonl", "judge": "labels.jsonl", "analyze": "analysis.json"}[action]

    def artifacts(self, stage):
        target = self.stage_path(stage)
        if stage == "prepare":
            return [target, target.parent / "state_index.json", target.parent / "population.json",
                    target.parent / "scoring_contract.json"]
        if stage.endswith("_generate"):
            return [target.parent / name for name in ("responses.jsonl", "trials.jsonl", "commits.jsonl",
                                                       "baseline_import_map.json", "baseline_replay_audit.json")]
        if stage.endswith("_judge"):
            return [target, target.parent / "scoring_consistency_audit.json"]
        if stage.endswith("_analyze"):
            return [target, target.parent / "pair_effects.csv"]
        return [target]

    def _fresh(self, stage, record):
        expected = {str(p) for p in self.artifacts(stage)}
        hashes = record.get("artifacts", {})
        return set(hashes) == expected and all(Path(p).is_file() and file_sha256(p) == h for p, h in hashes.items())

    def plan(self):
        state = self._state()["stages"]
        return [{"stage": s, "resource": "gpu" if s.endswith("_generate") else "api" if s.endswith("_judge") else "cpu",
                 "status": "complete" if s in state and self._fresh(s, state[s]) else "pending"} for s in STAGES]

    def status(self):
        return {"run_name": self.config.name, "stages": self.plan(), "budget": self._judge().summary(),
                "formal_allowed": False, "protocol_locked": False}

    def _checked_phase(self, phase, require_labels=False):
        directory = self.config.output_root / phase
        trials = [validate_trial_record(r) for r in read_jsonl(directory / "trials.jsonl")]
        responses = records_by_id(validate_response_record(r) for r in read_jsonl(directory / "responses.jsonl"))
        planned = {p.to_key(self._run_fingerprint(phase)).trial_id: p for p in self.plan_phase(phase)}
        imports = read_object(directory / "baseline_import_map.json")["records"]
        baselines = {r["new_trial_id"] for r in imports}
        expected_baselines = {TrialKey(self._run_fingerprint(phase), p.pair_id,
                              p.state_key, None, "baseline").trial_id for p in planned.values()}
        if baselines != expected_baselines or len(imports) != len(baselines):
            raise ReplicationStop("Imported baseline population differs from the planned pairs/states")
        by_trial = records_by_id(trials)
        if set(by_trial) != set(planned) | baselines or set(responses) != set(by_trial):
            raise ReplicationStop("Incomplete or unplanned generated grid")
        instructions = {p.pair_id: p.harmful_text for p in self.pairs()}
        for tid, row in by_trial.items():
            response = responses[tid]
            if (row["pair_id"] not in instructions or row["rq2_role"] != "rq2_dev"
                    or row["run_fingerprint"] != self._run_fingerprint(phase)
                    or response["run_fingerprint"] != row["run_fingerprint"]
                    or response["pair_id"] != row["pair_id"]
                    or response["harmful_text"] != instructions[row["pair_id"]]
                    or response["response_sha256"] != row["response_sha256"]):
                raise ReplicationStop("Generated trial provenance changed")
            if tid in planned:
                p = planned[tid]
                expected_base = TrialKey(self._run_fingerprint(phase), p.pair_id,
                                        p.state_key, None, "baseline").trial_id
                if row["step"] != (p.step or 0) or row["baseline_trial_id"] != expected_base:
                    raise ReplicationStop("Trial is linked to a different checkpoint or baseline")
                if row["audit"].get("apply_count") != 1 or row["audit"].get("token_count", 0) <= 0:
                    raise ReplicationStop("Intervention did not apply once to audio tokens")
                if row["baseline_trial_id"] not in baselines:
                    raise ReplicationStop("Missing imported baseline link")
        labels = read_jsonl(directory / "labels.jsonl", missing_ok=not require_labels)
        if require_labels:
            if set(records_by_id(labels)) != set(by_trial):
                raise ReplicationStop("Label coverage is incomplete")
            for label in labels:
                checked_label(label, responses[label["trial_id"]], self.config.judge,
                              canonical_sha256(self.spec.raw["scoring_contract"]))
            audit = scoring_consistency_audit(trials, list(responses.values()), labels)
            if not audit["passed"]:
                raise ReplicationStop("Scoring conflict or no-op drift; no selective regrading")
        return trials, responses, labels

    def _run_prepare(self):
        checked = self.verified  # run() verified under source locks immediately before this stage.
        directory = self.config.output_root / "provenance"
        for name, payload in (("preparation.json", checked), ("state_index.json", self.index),
                              ("population.json", self.population),
                              ("scoring_contract.json", self.spec.raw["scoring_contract"])):
            atomic_json(directory / name, payload)

    def _generate(self, phase):
        from core.reproducibility import configure_reproducibility
        configure_reproducibility(42, "warn")
        plan = self.plan_phase(phase)
        trials, responses, labels = self.parent._exploration_source_data
        sources = {(r["pair_id"], r["state_key"]): r for r in trials}
        unique = {(p.pair_id, p.state_key): p for p in plan}
        imports, judge = [], self._judge()
        binding = file_sha256(self.parent.config.output_root / "trajectory_behavior/labels.jsonl")
        for p in unique.values():
            source_key = "scan:clean" if p.target_kind == "clean" else f"scan:{p.step}"
            original = sources[(p.pair_id, source_key)]
            commit, label, meta = imported_baseline(p, run_fingerprint=self._run_fingerprint(phase),
                source_trial=original, source_response=responses[original["trial_id"]],
                source_label=labels[original["trial_id"]], source_binding=binding)
            imported = judge.imported_label(commit["response"])
            if imported is None:
                raise ReplicationStop("Baseline cache/receipt absent; do not regrade baseline")
            if any(imported[k] != label[k] for k in label if k != "judge_fingerprint"):
                raise ReplicationStop("Imported baseline differs from its paid source receipt")
            imports.append((commit, imported, meta))
        directory = self.config.output_root / phase
        seed_phase_baselines(directory, imports)
        runtime = self.runtime()
        try:
            # Bound prompt/reference residency to ONE pair (at most four window states).
            # The commit journal still spans the whole phase and preserves resume order.
            for pair in self.pairs():
                subset = tuple(p for p in plan if p.pair_id == pair.pair_id)
                if not subset:
                    continue
                runtime.clear_activation_cache()
                run_planned_trials(subset, runtime=runtime,
                                   prompt_cache=FrozenEventPromptCache(self.model(), (pair,), self.index),
                                   run_fingerprint=self._run_fingerprint(phase),
                                   responses_path=directory / "responses.jsonl", trials_path=directory / "trials.jsonl",
                                   generation=self._generation())
                runtime.clear_activation_cache()
        finally:
            runtime.clear_activation_cache()
            self._runtime_cache = self._model_cache = None
            del runtime
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        data, _, _ = self._checked_phase(phase)
        audit = replay_audit(data, plan, self._run_fingerprint(phase))
        atomic_json(directory / "baseline_replay_audit.json", audit)
        if not audit["passed"]:
            raise ReplicationStop("No-op replay drift against frozen baseline; stop before API")

    def _score(self, phase):
        trials, responses, _ = self._checked_phase(phase)
        directory = self.config.output_root / phase
        labels = records_by_id(read_jsonl(directory / "labels.jsonl", missing_ok=True))
        if not set(labels) <= set(responses):
            raise ReplicationStop("Unexpected preexisting label")
        judge = self._judge()
        for tid, response in responses.items():
            expected = judge.label(response)
            if tid in labels and labels[tid] != expected:
                raise ReplicationStop("Label differs from first valid content receipt")
            if tid not in labels:
                labels[tid] = expected
                atomic_jsonl(directory / "labels.jsonl", labels.values())
            if len(labels) % 100 == 0:
                print("[judge budget] " + json.dumps(judge.summary()), flush=True)
        self._checked_phase(phase, require_labels=True)
        atomic_json(directory / "scoring_consistency_audit.json",
                    scoring_consistency_audit(trials, list(responses.values()), list(labels.values())))

    def _analyze(self, phase):
        trials, _, labels = self._checked_phase(phase, require_labels=True)
        rows, summary = analyze(trials, labels)
        summary.update(original_oracle_qualified=False, budget=self._judge().summary(),
                       model_version_locked=False, minimum_positive_fraction_reference=.60)
        if phase == "window":
            center_trials, _, center_labels = self._checked_phase("mechanism", require_labels=True)
            centers, _ = analyze(center_trials, center_labels)
            summary["paired_window_contrasts"] = window_contrasts(centers, rows, self.spec.raw["window_pair_ids"])
        directory = self.config.output_root / phase
        atomic_csv(directory / "pair_effects.csv", rows)
        atomic_json(directory / "analysis.json", summary)

    def _run_report(self):
        lines = ["# RQ2 Oracle 后探索性延续报告", "",
                 "当前 20 条 dev；原 Oracle 未通过判定保持不变。没有使用 40 条 causal-test，没有 protocol_lock 或 formal。",
                 "", "Judge 请求名 DeepSeek-V4.1-Flash，实际服务版本未锁定；返回标识仅记录和告警。",
                 "旧标签仅在请求、评分模板、解析和内容一致且原付费回执有效时复用。",
                 "", "所有 CI、配对检验及对照差均为描述性结果，不产生预注册通过或正式因果结论。",
                 "随机重复先在 pair 内平均；事件窗口只比较同一事件、同一批完整窗口 pair。",
                 "事件前分数地板与事件本身短暂，均限制状态依赖的解释。生成上限沿用 100 新 token，未补做截断实验。", ""]
        for phase in ("mechanism", "window"):
            summary = read_object(self.config.output_root / phase / "analysis.json")
            lines += ["## " + phase, "", "| 层 | 状态 | 干预 | 剂量 | 有效 pair | 平均效用变化 | 改善/持平/变差 | 拒答变化均值 |",
                      "|---|---|---|---|---|---|---|---|"]
            for row in summary["summaries"]:
                u, r = row["endpoints"]["utility_effect"], row["endpoints"]["refusal_effect"]
                lines.append(f"| {row['layer']} | {row['state_key']} | {row['intervention']} | {row['dose']} | {u['pair_count']} | {u['mean']:.5f} | {u['positive_pairs']}/{u['tied_pairs']}/{u['negative_pairs']} | {r['mean']:.3f} |")
            lines += ["", "恢复效应：S_base−S_patch，R_patch−R_base，C_base−C_patch。反向抑制的三个方向全部反转。",
                      "正反两个方向的正数具有不同含义，不混合求均值；0.5 只作敏感性分析。",
                      f"完整 CI 与同 pair 的 H/random/sham 差值见 [{phase}/analysis.json]({phase}/analysis.json)。", ""]
        lines += ["## 开销与后续边界", "", json.dumps(self._judge().summary(), ensure_ascii=False), "",
                  "上述费用继承旧运行费用；不是供应商发票，不含 GPU 租金。",
                  "本探索分支到此停止。Subspace、正式测试和可部署防御需新协议，不自动进入。", ""]
        atomic_text(self.stage_path("report"), "\n".join(lines))

    def run(self, stages, *, acknowledge_cost=False, accept_exploratory=False):
        if not stages or any(s not in STAGES for s in stages):
            raise ReplicationStop("Requested stage is outside exploration scope")
        positions = [STAGES.index(s) for s in stages]
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ReplicationStop("Stages must be a contiguous forward sequence")
        if stages != ("prepare",) and not (acknowledge_cost and accept_exploratory):
            raise ReplicationStop("Acknowledge costs and exploratory/unlocked-Judge limits before execution")
        root = self.config.output_root
        with source_locks(self.spec.source):
            self.verify()
            if shutil.disk_usage(ROOT).free < 1024 ** 3:
                raise ReplicationStop("Less than 1 GiB free; resolve storage before new outputs")
            root.mkdir(parents=True, exist_ok=True)
            if any(p.is_symlink() for p in root.rglob("*")):
                raise ReplicationStop("No symlink outputs in exploration")
            with (root / ".run.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ReplicationStop("This exploration is already running") from None
                state = self._state()
                try:
                    for stage in stages:
                        for dependency in STAGES[:STAGES.index(stage)]:
                            if dependency not in state["stages"] or not self._fresh(dependency, state["stages"][dependency]):
                                raise ReplicationStop(f"Incomplete or changed prerequisite: {dependency}")
                        if stage in state["stages"]:
                            if not self._fresh(stage, state["stages"][stage]):
                                raise ReplicationStop("Completed exploration stage changed; do not overwrite")
                            print(f"[{stage}] already complete; hashes verified", flush=True)
                            continue
                        if stage in {"prepare", "report"}:
                            counter, total = lambda: int(self.stage_path(stage).is_file()), 1
                        else:
                            phase, action = stage.split("_")
                            planned = self.plan_phase(phase)
                            total = len(planned) + len({(p.pair_id, p.state_key) for p in planned})
                            counted = "commits.jsonl" if action == "generate" else "labels.jsonl"
                            counter = lambda: line_count(root / phase / counted)
                            if action == "analyze":
                                counter, total = lambda: int(self.stage_path(stage).is_file()), 1
                        with StageProgress(root, stage, counter, total):
                            if stage in {"prepare", "report"}:
                                getattr(self, "_run_" + stage)()
                            else:
                                getattr(self, {"generate": "_generate", "judge": "_score", "analyze": "_analyze"}[action])(phase)
                        state["stages"][stage] = {"completed_at": utc_now(),
                            "artifacts": {str(p): file_sha256(p) for p in self.artifacts(stage)}}
                        self._write_state(state)
                    atomic_json(root / "stop_reason.json", {"status": "requested_stages_complete", "at": utc_now(),
                                "formal_allowed": False, "protocol_locked": False})
                except BaseException as exc:
                    try:
                        atomic_json(root / "stop_reason.json", {"status": "stopped", "at": utc_now(),
                                    "stage": stage, "error_type": type(exc).__name__, "auto_retry": False})
                    except OSError:
                        pass  # Preserve original quota error; never mask it with logging failure.
                    raise
        return self.status()
