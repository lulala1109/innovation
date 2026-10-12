"""Versioned V4.1 Judge migration; the original v4 run remains read-only.

Only immutable generation artifacts are imported. Labels, scoring caches,
events and decisions are never imported. Both new Judge phases share one
contract/cache/budget; old spending is deducted from the remaining allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
from pathlib import Path

from rq2.artifacts import atomic_json, atomic_text, canonical_sha256, file_sha256
from rq2.judge_consistency import scoring_key
from rq2.replication_config import (
    ROOT, NAME as SOURCE_NAME, STAGES, BASELINE_STAGES, ORACLE_STAGES,
    BUDGET, DESIGN, load_replication_config, read_object,
)
from rq2.replication_judge import ContentJudge, ReplicationStop, contract_for
from rq2.replication_pipeline import ReplicationPipeline

NAME = SOURCE_NAME + "_judge_v41_run01"
IMPORTED_STAGES = STAGES[:3]
RUN_STAGES = STAGES[3:]
PROFILE = "configs/rq2_judge_deepseek_v41.json"
PROTOCOL = "docs/RQ2_v4_Judge_V41迁移协议_2026-10-08.md"
IMPLEMENTATION = (
    "rq2/replication_rejudge.py", "experiments/rq2_replication_rejudge.py",
    PROFILE, PROTOCOL,
)
COPY_FILES = (
    "provenance/reference_statistics.json",
    "trajectory/summary.json", "trajectory/resolved_manifest.json",
    "trajectory_behavior/responses.jsonl", "trajectory_behavior/trials.jsonl",
    "trajectory_behavior/scan_index.json", "trajectory_behavior/commits.jsonl",
)


def revision_contract(judge):
    result = contract_for(judge)
    result.update(version=3, migration="v4_independent_rejudge_v41",
                  returned_model_policy="pin_first_nonempty_exact_no_alias_merging")
    result["implementation_sha256"]["rq2/replication_rejudge.py"] = file_sha256(Path(__file__))
    return result


def _check_hashes(bindings):
    for relative, digest in bindings.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_sha256(path) != digest:
            raise ReplicationStop(f"Frozen migration input changed: {relative}")


def _archive_paths(root):
    paths = [p for p in (root / "scoring").rglob("*") if p.is_file()]
    labels = root / "trajectory_behavior/labels.jsonl"
    if labels.is_file():
        paths.append(labels)
    return sorted(paths)


def historical_accounting(root, source):
    """Carry money/attempts only, never the old fatal flag or any old label."""
    judge = ContentJudge(root / "scoring", source.runtime_config.judge,
                         source.raw["scoring_contract"], source.raw["budget"])
    state = judge.state
    if state.get("fatal_error") != "provider_model_or_reservation_violation":
        raise ReplicationStop("This migration requires the recorded model-drift stop")
    for attempt in state["attempts"]:
        if attempt["status"] != "settled":
            raise ReplicationStop("Unresolved old charges require reconciliation before migration")
        receipt = read_object(root / "scoring/receipts" / f"{attempt['number']:06d}.json")
        usage = receipt.get("usage", {})
        p, c = attempt.get("prompt_tokens"), attempt.get("completion_tokens")
        if (not isinstance(p, int) or isinstance(p, bool) or p < 0
                or not isinstance(c, int) or isinstance(c, bool) or c < 0
                or usage.get("prompt_tokens") != p or usage.get("completion_tokens") != c
                or usage.get("total_tokens") != p + c
                or attempt["cost_micro_cny"] != p * BUDGET["input_cny_per_million"] + c * BUDGET["output_cny_per_million"]
                or p > attempt["prompt_token_reservation"] or c > 800
                or receipt.get("model") != attempt.get("provider_model")):
            raise ReplicationStop("Old receipt/usage/accounting mismatch; not a model-name-only stop")
    models = {a.get("provider_model") for a in state["attempts"]}
    if len(models) < 2 or any(not isinstance(m, str) or not m for m in models):
        raise ReplicationStop("Source does not document a change between valid model identifiers")
    return {"cost_micro_cny": sum(a["cost_micro_cny"] for a in state["attempts"]),
            "http_attempts": len(state["attempts"]),
            "ledger_sha256": file_sha256(root / "scoring/ledger.json")}


def remaining_budget(budget, history):
    remaining = round(budget["api_limit_cny"] * 1_000_000) - history["cost_micro_cny"]
    attempts = budget["http_attempt_limit"] - history["http_attempts"]
    if remaining <= 0 or attempts <= 0:
        raise ReplicationStop("Original cumulative budget is exhausted; no new allowance")
    return {**budget, "api_limit_cny": remaining / 1_000_000,
            "http_attempt_limit": attempts, "historical_accounting": history}


class MigrationJudge(ContentJudge):
    def __init__(self, root, judge, contract, budget, history, *, transport=None):
        self.history, self.total_limit = history, budget["api_limit_cny"]
        super().__init__(root, judge, contract, remaining_budget(budget, history), transport=transport)

    def summary(self):
        result = super().summary()
        # Preserve the parent's per-revision keys: _settle compares these to
        # the remaining allowance. Report the cumulative accounting separately.
        result.update(historical_accounted_cny=self.history["cost_micro_cny"] / 1_000_000,
                      historical_http_attempts=self.history["http_attempts"],
                      cumulative_accounted_cny_at_peak_rates=(
                          self.history["cost_micro_cny"] + sum(a["cost_micro_cny"] for a in self.state["attempts"])) / 1_000_000,
                      cumulative_limit_cny=self.total_limit,
                      cumulative_http_attempts=self.history["http_attempts"] + result["http_attempts"])
        return result


@dataclass(frozen=True)
class RejudgeConfig:
    path: Path
    raw: dict
    source: object
    judge: dict
    fingerprint: str

    @property
    def output_root(self):
        return ROOT / self.raw["output_root"]

    @property
    def manifest(self):
        return self.source.manifest

    @property
    def pair_ids(self):
        return self.source.pair_ids

    @property
    def runtime_config(self):
        return replace(self.source.runtime_config, path=self.path, name=NAME,
                       output_root=self.output_root, raw=self.raw,
                       fingerprint=self.fingerprint, judge=self.judge)

    def verify(self):
        self.source.verify()
        _check_hashes(self.raw["bound_inputs"])
        _check_hashes(self.raw["source_archive"])
        if revision_contract(self.judge) != self.raw["scoring_contract"]:
            raise ReplicationStop("New scoring contract changed")
        parent = ReplicationPipeline(self.source)
        source_state = parent._state()
        if set(source_state["stages"]) != set(IMPORTED_STAGES):
            raise ReplicationStop("Source must stop after baseline generation, before completed Judge")
        if canonical_sha256(source_state) != self.raw["source_state_fingerprint"]:
            raise ReplicationStop("Source run progressed or changed after migration was frozen")
        # All inherited tensors/audio/generation JSONs are verified, but never
        # copied into the revision's trajectory directory or loaded onto GPU.
        for stage in IMPORTED_STAGES:
            if not parent._stage_is_fresh(stage, source_state["stages"][stage]):
                raise ReplicationStop(f"Source generation artifact changed: {stage}")
        actual_paths = {str(p.relative_to(ROOT)) for p in _archive_paths(self.source.output_root)}
        if actual_paths != set(self.raw["source_archive"]):
            raise ReplicationStop("Source scoring archive changed; do not run both versions")
        history = historical_accounting(self.source.output_root, self.source)
        if history != self.raw["historical_accounting"]:
            raise ReplicationStop("Historical spending changed")
        remaining_budget(self.raw["budget"], history)
        _, responses = parent._phase_data("trajectory_behavior")
        unique = len({scoring_key(r, canonical_sha256(self.raw["scoring_contract"])) for r in responses.values()})
        if len(responses) != 2040 or unique != self.raw["baseline_unique_inputs"]:
            raise ReplicationStop("Baseline scoring population changed")
        return {"scope": "v4_judge_migration", "request_model": self.judge["model"],
                "config_fingerprint": self.fingerprint, "source_responses": len(responses),
                "unique_baseline_api_inputs": unique, "gpu_rerun_required": False,
                "old_labels_reused": False, "historical_accounting": history,
                "cumulative_api_limit_cny": BUDGET["api_limit_cny"],
                "baseline_stop": "state_index", "oracle_requires_separate_command": True,
                "gpu_or_api_started": False}


def draft_config(source_path):
    """Return a no-write frozen config proposal for review/installation."""
    source = load_replication_config(source_path)
    parent = ReplicationPipeline(source)
    profile = read_object(ROOT / PROFILE)
    contract = revision_contract(profile)
    _, responses = parent._phase_data("trajectory_behavior")
    bound = {str(source.path.relative_to(ROOT)): file_sha256(source.path)}
    bound.update({name: file_sha256(ROOT / name) for name in IMPLEMENTATION})
    return {"format": "rq2-replication-judge-migration", "version": 1, "name": NAME,
            "source_config": str(source.path.relative_to(ROOT)),
            "source_state_fingerprint": canonical_sha256(parent._state()),
            "source_archive": {str(p.relative_to(ROOT)): file_sha256(p) for p in _archive_paths(source.output_root)},
            "output_root": "outputs/stage2_rq2/independent_replication/" + NAME,
            "judge_profile": PROFILE, "protocol": PROTOCOL,
            "design": DESIGN, "budget": BUDGET, "scoring_contract": contract,
            "historical_accounting": historical_accounting(source.output_root, source),
            "baseline_unique_inputs": len({scoring_key(r, canonical_sha256(contract)) for r in responses.values()}),
            "bound_inputs": bound}


def load_rejudge_config(path):
    path = Path(path).resolve()
    raw = read_object(path)
    fields = {"format", "version", "name", "source_config", "source_state_fingerprint",
              "source_archive", "output_root", "judge_profile", "protocol", "design",
              "budget", "scoring_contract", "historical_accounting", "baseline_unique_inputs", "bound_inputs"}
    if (set(raw) != fields or raw["format"] != "rq2-replication-judge-migration"
            or raw["version"] != 1 or raw["name"] != NAME
            or raw["design"] != DESIGN or raw["budget"] != BUDGET
            or raw["judge_profile"] != PROFILE or raw["protocol"] != PROTOCOL):
        raise ReplicationStop("Not the frozen V4.1 Judge migration")
    expected = ROOT / "outputs/stage2_rq2/independent_replication" / NAME
    output = ROOT / raw["output_root"]
    if output != expected or any(p.is_symlink() for p in (output, *output.parents)):
        raise ReplicationStop("Migration requires its own non-symlink output namespace")
    required = {raw["source_config"], *IMPLEMENTATION}
    if set(raw["bound_inputs"]) != required:
        raise ReplicationStop("Incomplete migration implementation bindings")
    _check_hashes(raw["bound_inputs"])
    source = load_replication_config(ROOT / raw["source_config"])
    judge = read_object(ROOT / PROFILE)
    if judge != {**source.runtime_config.judge, "model": "DeepSeek-V4.1-Flash"}:
        raise ReplicationStop("Only the request model may change; keep endpoint/rubric/thresholds")
    spec = RejudgeConfig(path, raw, source, judge, canonical_sha256(raw))
    spec.verify()
    return spec


class RejudgePipeline(ReplicationPipeline):
    def _judge(self):
        return MigrationJudge(self.config.output_root / "scoring", self.config.judge,
                              self.spec.raw["scoring_contract"], self.spec.raw["budget"],
                              self.spec.raw["historical_accounting"], transport=self.transport)

    def _run_fingerprint(self, phase):
        if phase == "trajectory_behavior":
            # Preserve identity of the original generation, not its Judge.
            return canonical_sha256({"config": self.spec.source.fingerprint,
                                     "phase": phase, "design_version": 4})
        return super()._run_fingerprint(phase)

    def _stage_artifacts(self, stage):
        return (*super()._stage_artifacts(stage), self.config.output_root / "provenance/generation_import.json")

    @contextmanager
    def _source_lock(self):
        # Read-open: never create, truncate, or edit the source lock file.
        with (self.spec.source.output_root / ".run.lock").open("r") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReplicationStop("Original v4 run is active; migration cannot run concurrently") from None
            yield

    def _prepare_locked(self):
        self.spec.verify()
        root, old = self.config.output_root, self.spec.source.output_root
        root.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in root.rglob("*")):
            raise ReplicationStop("Migrated output must not contain symlinks")
        with (root / ".run.lock").open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReplicationStop("Another process owns the migrated run") from None
            state = self._state()
            if state["stages"]:
                if any(s not in state["stages"] or not self._stage_is_fresh(s, state["stages"][s]) for s in IMPORTED_STAGES):
                    raise ReplicationStop("Imported generation artifacts changed or are incomplete")
                return
            if (root / "scoring").exists() or (root / "trajectory_behavior/labels.jsonl").exists():
                raise ReplicationStop("Scoring exists without import state; do not silently reset")
            imported = {}
            for rel in COPY_FILES:
                source, target = old / rel, root / rel
                digest = file_sha256(source)
                if target.exists():
                    if file_sha256(target) != digest:
                        raise ReplicationStop(f"Partial import conflicts: {rel}")
                else:
                    atomic_text(target, source.read_text(encoding="utf-8"))
                    if file_sha256(target) != digest:
                        raise ReplicationStop(f"Import byte identity failed: {rel}")
                imported[rel] = {"source": str(source), "sha256": digest}
            source_meta = read_object(old / "provenance/rq1_sources.json")
            source_meta.update(config_fingerprint=self.spec.fingerprint,
                               source_config_fingerprint=self.spec.source.fingerprint,
                               source_judge_protocol_fingerprint=source_meta.get("judge_protocol_fingerprint"),
                               judge_protocol_fingerprint=canonical_sha256(self.spec.raw["scoring_contract"]),
                               judge_migration=NAME, old_labels_reused=False)
            atomic_json(root / "provenance/rq1_sources.json", source_meta)
            atomic_json(root / "provenance/scoring_contract.json", self.spec.raw["scoring_contract"])
            atomic_json(root / "provenance/generation_import.json", {
                "format": "rq2-generation-import", "version": 1,
                "config_fingerprint": self.spec.fingerprint,
                "source_config": str(self.spec.source.path),
                "source_state_fingerprint": self.spec.raw["source_state_fingerprint"],
                "files": imported, "gpu_executed": False, "old_labels_reused": False,
                "historical_accounting": self.spec.raw["historical_accounting"]})
            self._phase_data("trajectory_behavior")
            state["stages"] = {s: {"output": str(self.stage_path(s)), "resources": ["cpu"],
                                   "mode": "imported_not_executed",
                                   "artifacts": {str(p): file_sha256(p) for p in self._stage_artifacts(s)}}
                               for s in IMPORTED_STAGES}
            self._write_state(state)

    def prepare(self):
        with self._source_lock():
            self._prepare_locked()
        return {"status": "prepared", "gpu_or_api_started": False, "imported_stages": list(IMPORTED_STAGES)}

    def run(self, stages, *, acknowledge_cost=False, confirm_no_external_rq2_use=False):
        if (not stages or any(s not in RUN_STAGES for s in stages)
                or [STAGES.index(s) for s in stages] != sorted({STAGES.index(s) for s in stages})):
            raise ReplicationStop("Migration starts at Judge; baseline GPU stages cannot be run")
        if any(s in BASELINE_STAGES for s in stages) and any(s in ORACLE_STAGES for s in stages):
            raise ReplicationStop("Stop at state_index before a separate Oracle command")
        if not acknowledge_cost or not confirm_no_external_rq2_use:
            raise ReplicationStop("Both execution acknowledgement flags are required")
        with self._source_lock():
            self._prepare_locked()
            return super().run(stages, acknowledge_cost=acknowledge_cost,
                               confirm_no_external_rq2_use=confirm_no_external_rq2_use)

    def plan(self):
        result = super().plan()
        for row in result:
            if row["stage"] in IMPORTED_STAGES:
                row.update(resources=["cpu"], mode="import_only_no_gpu")
        return result

    def estimate(self):
        history = self.spec.raw["historical_accounting"]
        average = history["cost_micro_cny"] / max(1, history["http_attempts"]) / 1_000_000
        return {"request_model": self.config.judge["model"],
                "baseline_unique_scoring_inputs": self.spec.raw["baseline_unique_inputs"],
                "baseline_first_attempt_estimate_cny": round(average * self.spec.raw["baseline_unique_inputs"], 4),
                "oracle_max_new_unique_inputs": 320,
                "oracle_first_attempt_reference_cny": round(average * 320, 4),
                "estimate_basis": "old_5_requests_average_usage_at_prior_peak_rates_not_guaranteed_for_new_model",
                "accounting_rates_cny_per_million": {"input": BUDGET["input_cny_per_million"], "output": BUDGET["output_cny_per_million"]},
                "historical_accounting": history,
                "cumulative_api_limit_cny": self.spec.raw["budget"]["api_limit_cny"],
                "new_baseline_gpu_hours": 0,
                "oracle_gpu_hours_rough_estimate": [0.5, 1.5],
                "note": "No API validation performed. Estimates exclude retries and are not provider invoices. Baseline imports only; Oracle requires a separate GPU command."}

    def _run_report(self):
        super()._run_report()
        path = self.stage_path("report")
        atomic_text(path, path.read_text(encoding="utf-8")
                    + "\nJudge 请求模型：" + self.config.judge["model"]
                    + "。本报告只使用新契约评分；旧评分仅归档，GPU基线沿用原生成。\n")

    def _run_sources(self):
        raise ReplicationStop("Generation sources must be imported")

    _run_trajectory = _run_sources
    _run_trajectory_behavior_generate = _run_sources
