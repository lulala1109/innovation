"""Independent calibration A: numerical gate, fresh generation, paired dev analysis."""
from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import shutil

from rq2.artifacts import (TrialKey, append_jsonl_fsync, atomic_csv, atomic_json, atomic_jsonl,
                          atomic_text, canonical_sha256, file_sha256, read_jsonl, records_by_id,
                          validate_trial_record)
from rq2.behavior import validate_response_record
from rq2.experiment import run_planned_trials
from rq2.exploration_analysis import ENDPOINTS, analyze, estimate, paired_contrast
from rq2.exploration_config import source_locks
from rq2.exploration_network_resume import NetworkResumePipeline, forbidden_transport
from rq2.exploration_pipeline import ExplorationJudge, ExplorationPipeline, FrozenEventPromptCache
from rq2.judge_consistency import checked_label, scoring_consistency_audit, scoring_key
from rq2.orthogonal_config import CENTER, ROOT, STAGES, plan_trials, source_snapshot
from rq2.orthogonal_interventions import (KINDS, OrthogonalRuntime, check_matched_audits, numeric_metrics)
from rq2.pipeline import RQ2Pipeline
from rq2.reachability import validated_layer_map
from rq2.replication_config import read_object
from rq2.replication_judge import ContentJudge, ReplicationStop, utc_now
from rq2.replication_progress import StageProgress, line_count


def require(condition, message):
    if not condition:
        raise ReplicationStop(message)


def recover_journal(path):
    """Remove only an interrupted trailing record before the next append."""
    path = Path(path)
    rows = read_jsonl(path, missing_ok=True, recover_truncated=True)
    if path.exists():
        try:
            complete = read_jsonl(path)
        except ValueError:
            complete = None
        if complete is None or path.read_bytes() and not path.read_bytes().endswith(b"\n"):
            atomic_jsonl(path, rows)
    return rows


class CalibrationJudge(ExplorationJudge):
    def __init__(self, *args, readers, uncertain_history, **kwargs):
        self.readers, self.uncertain_history = readers, uncertain_history
        super().__init__(*args, source_judge=None, **kwargs)
        for reader in readers:
            for key in ("provider", "model", "endpoint_sha256", "prompt_sha256", "schema_retry_prompt_sha256",
                        "temperature", "response_format", "max_tokens", "thinking", "success_threshold",
                        "min_convincingness", "min_specificity"):
                require(reader.contract.get(key) == self.contract.get(key), "Historical scoring service differs")
            for key in ("evaluation/behavior.py", "core/llm_backend.py", "rq2/behavior.py"):
                require(reader.contract["implementation_sha256"][key] == self.contract["implementation_sha256"][key],
                        "Historical scoring implementation differs")

    def summary(self):
        return {**super().summary(), **self.uncertain_history}

    def label(self, response):
        require(not self.state.get("fatal_error") and all(a["status"] == "settled" for a in self.state["attempts"]),
                "Unresolved calibration API/accounting state; no automatic continuation")
        imported = self.imported_label(response)
        return imported if imported is not None else ContentJudge.label(self, response)

    def imported_label(self, response):
        found, provenance, attempted = [], [], False
        for reader in self.readers:
            key = scoring_key(response, reader.fingerprint)
            path = reader.root / "cache" / (key + ".json")
            attempted |= any(a["key"] == key for a in reader.state["attempts"])
            if not path.is_file():
                continue
            # Invoke the frozen cache-only path directly, not an importing subclass
            # that could create a provenance file in a historical directory.
            old = ContentJudge.label(reader, response)
            found.append(checked_label({**old, "judge_fingerprint": self.fingerprint}, response,
                                       self.judge, self.fingerprint))
            provenance.append({"cache": str(path), "sha256": file_sha256(path),
                               "contract_sha256": reader.fingerprint})
        require(not found or all(label == found[0] for label in found), "Historical exact-input labels conflict")
        require(found or not attempted, "Historical request lacks a valid cache; do not silently pay again")
        if not found:
            return None
        key = scoring_key(response, self.fingerprint)
        path = self.root / "imported_cache" / (key + ".json")
        audit = {"key": key, "sources": provenance, "policy": "same_service_exact_content_paid_receipts_read_only"}
        require(not path.exists() or read_object(path) == audit, "Historical cache provenance changed")
        if not path.exists():
            atomic_json(path, audit)
        return found[0]


class OrthogonalPipeline(ExplorationPipeline):
    def __init__(self, spec, *, transport=None, **kwargs):
        self.spec, self.transport = spec, transport
        self.parent = NetworkResumePipeline(spec.source, spec.raw["source_recovery_attempt"], transport=forbidden_transport)
        self.index = self.population = self.source_baselines = None
        RQ2Pipeline.__init__(self, spec.runtime_config, **kwargs)

    def verify(self):
        inputs, history, uncertain = source_snapshot(self.parent)
        require(canonical_sha256(inputs) == self.spec.raw["source_inventory_sha256"]
                and len(inputs) == self.spec.raw["source_inventory_count"], "Source exploration artifacts changed")
        require(history == self.spec.raw["history"] and uncertain == self.spec.raw["historical_uncertain"],
                "Historical accounting changed")
        self.index, self.population = self.parent.index, self.parent.population
        require(canonical_sha256(self.index) == self.spec.raw["event_index_sha256"], "Event index changed")
        pairs, plan = self.pairs(), self.plan_phase("calibration")
        require({p.pair_id for p in pairs} == set(self.spec.raw["pair_ids"]), "Pair population differs")
        mapping = validated_layer_map(read_object(self.parent.parent.config.stage_path("layer_map")))
        require(all(p.intervention in KINDS and p.layer in (19, 24) and p.token_scope == "audio"
                    and p.dose == 1 and mapping[p.layer]["audio_only_prefill_behaviorally_reachable"] for p in plan),
                "Unreachable or unplanned calibration operation")
        trials, responses, _ = self.parent._checked_phase("mechanism", require_labels=True)
        self.source_baselines = {t["pair_id"]: responses[t["trial_id"]] for t in trials
                                 if t["intervention"] == "baseline" and t["state_key"] == CENTER}
        require(set(self.source_baselines) == set(self.spec.raw["pair_ids"]), "Source event baselines incomplete")
        require(self._generation() == {"max_tokens": 100, "temperature": 1., "do_sample": False},
                "Frozen generation settings changed")
        self.verified = {"scope": "orthogonal_calibration_a_exploratory_dev", "pair_count": 20,
                         "interventions": 240, "fresh_baselines": 20, "total_responses": 260,
                         "numeric_preflight_conditions": 240, "new_pgd_runs": 0, "causal_test_pairs": 0,
                         "formal_allowed": False, "protocol_locked": False, "gpu_or_api_started": False,
                         "original_oracle_decision_unchanged": True, "history": history}
        return self.verified

    def plan_phase(self, phase):
        require(phase == "calibration", "Only calibration A is supported")
        if self.index is None:
            self.index = read_object(self.spec.source.output_root / "provenance/state_index.json")
        return plan_trials(self.index, self.spec.raw["pair_ids"])

    def _run_fingerprint(self, phase):
        require(phase == "calibration", "Unknown calibration phase")
        return canonical_sha256({"orthogonal_config": self.spec.fingerprint, "phase": phase})

    def runtime(self):
        if self._runtime_cache is None:
            self._runtime_cache = OrthogonalRuntime(self.model(), self.bundle(),
                                                    output_selector=self.config.model.get("output_selector"))
        return self._runtime_cache

    def _release_runtime(self):
        if self._runtime_cache is not None:
            self._runtime_cache.clear_activation_cache()
        self._runtime_cache = self._model_cache = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _judge(self):
        current = self.parent._judge()
        readers = (self.parent.parent._judge(), self.parent.prior, current)
        return CalibrationJudge(self.config.output_root / "scoring", self.config.judge,
                                self.spec.raw["scoring_contract"], self.spec.raw["budget"], self.spec.raw["history"],
                                readers=readers, uncertain_history=self.spec.raw["historical_uncertain"],
                                transport=self.transport)

    def stage_path(self, stage):
        require(stage in STAGES, "No reverse, window, PGD, subspace or formal in calibration A")
        root = self.config.output_root
        if stage == "prepare":
            return root / "provenance/preparation.json"
        if stage == "numeric_preflight":
            return root / "numeric_preflight/summary.json"
        if stage == "report":
            return root / "rq2_orthogonal_calibration_report.md"
        return root / "calibration" / {"calibration_generate": "responses.jsonl",
                                       "calibration_judge": "labels.jsonl", "calibration_analyze": "analysis.json"}[stage]

    def artifacts(self, stage):
        target = self.stage_path(stage)
        if stage == "prepare":
            return [target, target.parent / "state_index.json", target.parent / "population.json", target.parent / "scoring_contract.json"]
        if stage == "numeric_preflight":
            return [target, target.parent / "audits.jsonl"]
        if stage == "calibration_generate":
            return [target.parent / n for n in ("responses.jsonl", "trials.jsonl", "commits.jsonl", "generation_audit.json")]
        if stage == "calibration_judge":
            return [target, target.parent / "scoring_consistency_audit.json",
                    *sorted(p for p in (self.config.output_root / "scoring").rglob("*") if p.is_file())]
        if stage == "calibration_analyze":
            return [target, target.parent / "pair_effects.csv"]
        return [target]

    def plan(self):
        state = self._state()["stages"]
        return [{"stage": s, "resource": "gpu" if s in ("numeric_preflight", "calibration_generate")
                 else "api" if s == "calibration_judge" else "cpu",
                 "status": ("complete" if self._fresh(s, state[s]) else "changed") if s in state else "pending"}
                for s in STAGES]

    def status(self):
        return {"run_name": self.config.name, "stages": self.plan(), "budget": self._judge().summary(),
                "formal_allowed": False, "protocol_locked": False}

    def _preflight_rows(self):
        records = records_by_id(recover_journal(self.config.output_root / "numeric_preflight/audits.jsonl"))
        expected = {p.to_key(self._run_fingerprint("calibration")).trial_id: p for p in self.plan_phase("calibration")}
        require(set(records) <= set(expected), "Unplanned preflight record")
        for tid, row in records.items():
            p = expected[tid]
            require(all(row[k] == getattr(p, k) for k in ("pair_id", "layer", "intervention", "replicate")),
                    "Preflight metadata mismatch")
            require(row["audit"]["kind"] == p.intervention and row["audit"]["layer"] == p.layer
                    and row["audit"]["dose"] == p.dose, "Preflight audit identity mismatch")
            numeric_metrics(row["audit"])
        return records

    def _numeric_preflight(self):
        from core.reproducibility import configure_reproducibility
        configure_reproducibility(42, "warn")
        rows, plan = self._preflight_rows(), self.plan_phase("calibration")
        path = self.config.output_root / "numeric_preflight/audits.jsonl"
        try:
            for pair in self.pairs():
                missing = [p for p in plan if p.pair_id == pair.pair_id
                           and p.to_key(self._run_fingerprint("calibration")).trial_id not in rows]
                if not missing:
                    continue
                runtime = self.runtime()
                runtime.clear_activation_cache()
                cache = FrozenEventPromptCache(self.model(), (pair,), self.index)
                for p in missing:
                    prepared = cache.prepare(p)
                    _, audit = runtime.diagnostic_trial(prepared.source_prompt, prepared.target_prompt,
                                                       p.to_spec(), source_cache_key=prepared.source_cache_key)
                    tid = p.to_key(self._run_fingerprint("calibration")).trial_id
                    row = {"trial_id": tid, "pair_id": p.pair_id, "layer": p.layer,
                           "intervention": p.intervention, "replicate": p.replicate, "audit": audit.to_dict()}
                    append_jsonl_fsync(path, row)
                    rows[tid] = row
                runtime.clear_activation_cache()
        finally:
            self._release_runtime()
        require(len(rows) == 240, "Incomplete numeric preflight")
        summary = check_matched_audits(list(rows.values()))
        summary.update(generation_responses=0, api_calls=0, formal_allowed=False,
                       scope="local_manipulation_check_not_behavioral_effectiveness")
        atomic_json(self.stage_path("numeric_preflight"), summary)

    def _checked_phase(self, phase, require_labels=False):
        require(phase == "calibration", "Unknown phase")
        root = self.config.output_root / phase
        trials = records_by_id(validate_trial_record(t) for t in read_jsonl(root / "trials.jsonl"))
        responses = records_by_id(validate_response_record(r) for r in read_jsonl(root / "responses.jsonl"))
        fingerprint = self._run_fingerprint(phase)
        plan = self.plan_phase(phase)
        planned = {p.to_key(fingerprint).trial_id: p for p in plan}
        baselines = {TrialKey(fingerprint, p.pair_id, CENTER, None, "baseline").trial_id for p in plan}
        require(set(trials) == set(responses) == set(planned) | baselines, "Generated coverage differs from 240+20")
        pairs = {p.pair_id: p for p in self.pairs()}
        centers = {r["pair_id"]: r for r in self.index["records"] if r["state_key"] == CENTER and r["available"]}
        audits, diagnostics = [], []
        for tid, trial in trials.items():
            response, pair = responses[tid], pairs[trial["pair_id"]]
            center = centers[pair.pair_id]
            provenance = trial["diagnostic"]["input_provenance"]
            require(trial["run_fingerprint"] == response["run_fingerprint"] == fingerprint
                    and trial["rq2_role"] == "rq2_dev" and trial["state_key"] == CENTER
                    and trial["step"] == center["step"] and response["pair_id"] == pair.pair_id
                    and response["harmful_text"] == pair.harmful_text
                    and response["response_sha256"] == trial["response_sha256"], "Generated provenance mismatch")
            require(provenance == {"pair_id": pair.pair_id, "content_group": pair.content_group,
                    "clean_audio_sha256": pair.clean_audio_sha256, "target": {"kind": "trajectory",
                    "step": center["step"], "checkpoint_path": center["checkpoint_path"],
                    "checkpoint_sha256": center["checkpoint_sha256"]}}, "Prompt/checkpoint provenance differs")
            if tid in baselines:
                require(trial["audit"].get("kind") == "baseline" and trial["audit"].get("apply_count") == 0,
                        "Baseline was not generated unpatched")
                require(response["response_sha256"] == self.source_baselines[pair.pair_id]["response_sha256"],
                        "Fresh baseline differs from the frozen event response; stop before API")
                continue
            p = planned[tid]
            bid = TrialKey(fingerprint, pair.pair_id, CENTER, None, "baseline").trial_id
            require(trial["baseline_trial_id"] == bid, "Wrong paired baseline")
            for audit in (trial["audit"], trial["audit"]["diagnostic"]):
                require(audit["kind"] == p.intervention and audit["layer"] == p.layer and audit["dose"] == p.dose,
                        "Generation/diagnostic operator identity mismatch")
                numeric_metrics(audit)
                require(audit["token_indices"] == trial["diagnostic"]["target_token_indices"]
                        == trial["diagnostic"]["source_token_indices"]
                        == trial["diagnostic"]["target_reference_token_indices"], "Audio span mismatch")
            audits.append(trial)
            diagnostics.append({**trial, "audit": trial["audit"]["diagnostic"]})
            if p.intervention == "sham":
                require(trial["response_sha256"] == trials[bid]["response_sha256"], "Sham changed the response")
        check_matched_audits(audits)
        check_matched_audits(diagnostics)
        labels = read_jsonl(root / "labels.jsonl", missing_ok=not require_labels)
        require(set(records_by_id(labels)) <= set(trials), "Unexpected label")
        if require_labels:
            require(set(records_by_id(labels)) == set(trials), "Incomplete label coverage")
            for label in labels:
                checked_label(label, responses[label["trial_id"]], self.config.judge,
                              canonical_sha256(self.spec.raw["scoring_contract"]))
            require(scoring_consistency_audit(list(trials.values()), list(responses.values()), labels)["passed"],
                    "Scoring consistency/no-op audit failed")
        return list(trials.values()), responses, labels

    def _generate(self, phase):
        from core.reproducibility import configure_reproducibility
        configure_reproducibility(42, "warn")
        require(read_object(self.stage_path("numeric_preflight"))["passed"] is True, "Numerical preflight not passed")
        root, plan = self.config.output_root / phase, self.plan_phase(phase)
        completed = self._generation_progress()
        try:
            for pair in self.pairs():
                subset = tuple(p for p in plan if p.pair_id == pair.pair_id)
                expected = {p.to_key(self._run_fingerprint(phase)).trial_id for p in subset}
                expected.add(TrialKey(self._run_fingerprint(phase), pair.pair_id, CENTER, None, "baseline").trial_id)
                if expected <= completed:
                    continue
                runtime = self.runtime()
                runtime.clear_activation_cache()
                run_planned_trials(subset, runtime=runtime,
                                   prompt_cache=FrozenEventPromptCache(self.model(), (pair,), self.index),
                                   run_fingerprint=self._run_fingerprint(phase),
                                   responses_path=root / "responses.jsonl", trials_path=root / "trials.jsonl",
                                   generation=self._generation())
                runtime.clear_activation_cache()
                completed = self._generation_progress()
        finally:
            self._release_runtime()
        self._checked_phase(phase)
        atomic_json(root / "generation_audit.json", {"passed": True, "fresh_baselines": 20,
                    "interventions": 240, "sham_replays": 40, "frozen_baseline_matches": 20,
                    "diagnostic_and_generation_numeric_checks": True,
                    "generation_limit": 100, "stop_reason_available": False,
                    "limitation": "Frozen model API returns text only; actual generated token count and finish reason are unavailable."})

    def _generation_progress(self):
        root = self.config.output_root / "calibration"
        commits = records_by_id(recover_journal(root / "commits.jsonl"))
        fingerprint = self._run_fingerprint("calibration")
        plan = self.plan_phase("calibration")
        expected = {p.to_key(fingerprint).trial_id for p in plan}
        baselines = {TrialKey(fingerprint, p.pair_id, CENTER, None, "baseline").trial_id: p for p in plan}
        require(set(commits) <= expected | set(baselines), "Unplanned generation commit; stop before model loading")
        responses, trials = [], []
        for tid, commit in commits.items():
            require(commit.get("format") == "rq2-trial-commit" and commit.get("version") == 1, "Invalid generation commit")
            r, t = validate_response_record(commit["response"]), validate_trial_record(commit["trial"])
            require(r["trial_id"] == t["trial_id"] == tid and t["run_fingerprint"] == r["run_fingerprint"] == fingerprint
                    and r["response_sha256"] == t["response_sha256"], "Generation commit identity differs")
            if tid in baselines:
                pair = baselines[tid].pair_id
                require(r["response_sha256"] == self.source_baselines[pair]["response_sha256"],
                        "Fresh baseline differs from the frozen event response; stop before further generation")
            else:
                numeric_metrics(t["audit"])
                numeric_metrics(t["audit"]["diagnostic"])
            responses.append(r)
            trials.append(t)
        if commits:
            # The fsynced commit is authoritative; projections may be stale after a crash.
            atomic_jsonl(root / "responses.jsonl", responses)
            atomic_jsonl(root / "trials.jsonl", trials)
        else:
            require(not any(read_jsonl(root / name, missing_ok=True, recover_truncated=True)
                            for name in ("responses.jsonl", "trials.jsonl")), "Generated projections exist without commits")
        return set(commits)

    def _analyze(self, phase):
        trials, _, labels = self._checked_phase(phase, require_labels=True)
        rows, summary = analyze(trials, labels)
        groups = defaultdict(list)
        for row in rows:
            groups[row["layer"], row["intervention"]].append(row)
        contrasts = []
        for layer in (19, 24):
            for control in ("h_orthogonal_control", "random_orthogonal_control", "sham"):
                diff = paired_contrast(groups[layer, "r_direction"], groups[layer, control])
                contrasts.append({"layer": layer, "contrast": "r_direction minus " + control,
                                  "endpoints": {f: estimate(diff, f) for f in ENDPOINTS}})
        summary.update(scope="orthogonal_calibration_a_exploratory_dev", paired_control_contrasts=contrasts,
                       original_oracle_qualified=False, budget=self._judge().summary(), model_version_locked=False,
                       selection="L19/L24 selected after prior dev results; not confirmatory")
        atomic_csv(self.config.output_root / phase / "pair_effects.csv", rows)
        atomic_json(self.stage_path("calibration_analyze"), summary)

    def _run_report(self):
        summary = read_object(self.stage_path("calibration_analyze"))
        lines = ["# RQ2 Orthogonal Calibration A", "", "Exploratory only: 20 existing dev pairs, L19/L24, event center.",
                 "240 interventions plus 20 freshly generated, frozen-response-matched baselines. No new PGD or causal-test.",
                 "The original negative Oracle decision is unchanged; no formal authorization.", "",
                 "| Layer | Contrast | Utility difference | Descriptive 95% CI | Pairs |",
                 "|---|---|---:|---|---:|"]
        for row in summary["paired_control_contrasts"]:
            u = row["endpoints"]["utility_effect"]
            lines.append(f"| {row['layer']} | {row['contrast']} | {u['mean']:.6f} | [{u['ci_low']:.6f}, {u['ci_high']:.6f}] | {u['pair_count']} |")
        lines += ["", "Random repetitions are averaged within pair; intervals are descriptive, not confirmatory.",
                  "See calibration/analysis.json for all three endpoints and calibration/pair_effects.csv for paired effects.",
                  "H orthogonalization tests geometric overlap, not semantic disentanglement.",
                  "No reverse, window, subspace, deployment defense, or human semantic review was run.",
                  "Generation remains greedy with a 100-new-token cap; the frozen text-only API exposes no finish reason.",
                  "The Judge serving version is not pinned. Costs carry previous spending and the uncertain reservation, not a provider invoice.",
                  "", json.dumps(self._judge().summary(), ensure_ascii=False), ""]
        atomic_text(self.stage_path("report"), "\n".join(lines))

    def run(self, stages, *, acknowledge_cost=False, accept_exploratory=False):
        require(bool(stages) and all(s in STAGES for s in stages), "Unsupported calibration stage")
        positions = [STAGES.index(s) for s in stages]
        require(positions == list(range(positions[0], positions[-1]+1)), "Stages must be contiguous and forward")
        require(tuple(stages) == ("prepare",) or acknowledge_cost and accept_exploratory,
                "Both cost and exploratory acknowledgements are required")
        root = self.config.output_root
        with source_locks(self.spec.source):
            self.verify()
            require(shutil.disk_usage(ROOT).free >= 1024**3, "Less than 1 GiB free")
            root.mkdir(parents=True, exist_ok=True)
            require(not any(p.is_symlink() for p in root.rglob("*")), "Output contains a symlink")
            with (root / ".run.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ReplicationStop("Calibration A is already running") from None
                state = self._state()
                try:
                    for stage in stages:
                        for dependency in STAGES[:STAGES.index(stage)]:
                            require(dependency in state["stages"] and self._fresh(dependency, state["stages"][dependency]),
                                    f"Incomplete or changed prerequisite: {dependency}")
                        if stage in state["stages"]:
                            require(self._fresh(stage, state["stages"][stage]), "Completed stage changed; do not overwrite")
                            print(f"[{stage}] already complete; hashes verified", flush=True)
                            continue
                        if stage == "numeric_preflight":
                            count, total = lambda: line_count(root / "numeric_preflight/audits.jsonl"), 240
                        elif stage in ("calibration_generate", "calibration_judge"):
                            path = root / "calibration" / ("commits.jsonl" if stage.endswith("generate") else "labels.jsonl")
                            count, total = lambda: line_count(path), 260
                        else:
                            count, total = lambda: int(self.stage_path(stage).is_file()), 1
                        with StageProgress(root, stage, count, total):
                            if stage == "prepare":
                                self._run_prepare()
                                self._judge().save()
                            elif stage == "numeric_preflight":
                                self._numeric_preflight()
                            elif stage == "report":
                                self._run_report()
                            else:
                                getattr(self, {"calibration_generate": "_generate", "calibration_judge": "_score",
                                               "calibration_analyze": "_analyze"}[stage])("calibration")
                        state["stages"][stage] = {"completed_at": utc_now(),
                            "artifacts": {str(p): file_sha256(p) for p in self.artifacts(stage)}}
                        self._write_state(state)
                    atomic_json(root / "stop_reason.json", {"status": "requested_stages_complete", "at": utc_now(),
                                                            "formal_allowed": False, "protocol_locked": False})
                except BaseException as exc:
                    atomic_json(root / "stop_reason.json", {"status": "stopped", "stage": stage,
                                "error_type": type(exc).__name__, "auto_retry": False, "at": utc_now()})
                    raise
        return self.status()
