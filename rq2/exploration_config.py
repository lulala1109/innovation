"""Frozen, dev-only continuation of the completed v4 Oracle; no old files edited."""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import copy
import fcntl
from pathlib import Path

from rq2.artifacts import canonical_sha256, file_sha256, read_jsonl
from rq2.dev_judge_resume import DevWarningPipeline, load_warning_config
from rq2.event_config import EVENT_CENTER
from rq2.event_data import build_event_index
from rq2.experiment import PlannedTrial, build_mechanism_plan
from rq2.replication_config import ROOT, STAGES as SOURCE_STAGES, read_object
from rq2.replication_judge import ReplicationStop, contract_for

NAME = "qwen7b_advbench_post_oracle_explore_v1_run01"
SOURCE = "configs/stage2_rq2_qwen7b_advbench_v41_dev_warn01.json"
CONFIG = "configs/stage2_rq2_qwen7b_advbench_post_oracle_explore_v1_run01.json"
PROTOCOL = "docs/RQ2_Oracle后探索性延续协议_v1_2026-10-10.md"
STAGES = ("prepare", "mechanism_generate", "mechanism_judge", "mechanism_analyze",
          "window_generate", "window_judge", "window_analyze", "report")
DESIGN = {"candidate_layers": [19, 24, 26], "neighbor_layers": [18, 20, 23, 25],
          "center": EVENT_CENTER, "offsets": [-2, -1, 0, 1, 3],
          "primary_dose": 1., "sensitivity_dose": .5, "random_replicates": 3,
          "seed": 42, "source_pairs": 20, "formal_pairs": 0,
          "token_scope": "audio", "application": "prefill_once",
          "maximum_new_responses": 2360, "bootstrap_replicates": 2000,
          "minimum_complete_window_pairs": 16, "model_version_locked": False,
          "protocol_locked": False, "causal_evidence": False,
          "oracle_failure_override": "exploration_only_original_decision_unchanged",
          "allow_subspace": False, "allow_formal": False}
IMPLEMENTATION = ("rq2/exploration_config.py", "rq2/exploration_pipeline.py",
                  "rq2/exploration_analysis.py", "experiments/rq2_exploration.py", PROTOCOL)


@contextmanager
def source_locks(source):
    with ExitStack() as stack:
        while source is not None:
            handle = stack.enter_context((source.output_root / ".run.lock").open("r"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ReplicationStop("A source run is active; do not continue concurrently") from None
            source = getattr(source, "source", None)
        yield


def inventory(parent):
    """Bind finished stage outputs, scoring receipts/cache and checkpoint identities."""
    root = parent.config.output_root
    state = parent._state()
    if set(state["stages"]) != set(SOURCE_STAGES):
        raise ReplicationStop("Source Oracle/report is not complete")
    for stage, record in state["stages"].items():
        if stage == "trajectory":
            # Retain/hash the entire completed stage record, but only consume the
            # resolved manifest and summary here. Selected checkpoint bytes are
            # verified separately in event_index; no 2,020-checkpoint re-scan.
            for relative in ("trajectory/summary.json", "trajectory/resolved_manifest.json"):
                path = root / relative
                if record["artifacts"].get(str(path)) != file_sha256(path):
                    raise ReplicationStop("Source trajectory metadata changed")
            continue
        if not parent._stage_is_fresh(stage, record):
            raise ReplicationStop(f"Source stage changed: {stage}")
    paths = {p for p in root.rglob("*") if p.is_file() and p.name != ".run.lock"}
    for stage, record in state["stages"].items():
        if stage == "trajectory":
            continue
        paths.update(Path(p) for p in record["artifacts"])
    result = {}
    for p in sorted(paths):
        if p.is_symlink() or p.name.endswith(".tmp") or not p.resolve().is_relative_to(ROOT):
            raise ReplicationStop("Source contains unbound/symlink/temporary artifacts")
        result[str(p.resolve().relative_to(ROOT))] = file_sha256(p)
    return result


def event_index(parent):
    if not hasattr(parent, "_exploration_source_data"):
        parent._exploration_source_data = parent._source_data()
    trials, _, labels = parent._exploration_source_data
    root = parent.config.output_root
    index, population = build_event_index(
        read_object(root / "trajectory_behavior/scan_index.json"),
        read_object(root / "state_index/behavior_events.json"),
        read_object(parent._effective_manifest())["records"], trials, list(labels.values()),
        pair_ids=parent.spec.pair_ids)
    original = read_object(root / "state_index/index.json")
    old_centers = {(r["pair_id"], r["step"]) for r in original["records"] if r["available"]}
    new_centers = {(r["pair_id"], r["step"]) for r in index["records"]
                   if r["available"] and r["state_key"] == EVENT_CENTER}
    if old_centers != new_centers or population["event_pair_count"] != 20:
        raise ReplicationStop("Do not redefine events or drop existing zero-score pairs")
    if population["full_window_pair_count"] < 16:
        raise ReplicationStop("Fewer than 16 complete-window pairs; keep data, revisit scope")
    # Hash only the selected existing checkpoints, not every PGD checkpoint again.
    for row in index["records"]:
        if row["available"]:
            path = Path(row["checkpoint_path"])
            if not path.is_file() or file_sha256(path) != row["checkpoint_sha256"]:
                raise ReplicationStop("Selected event checkpoint changed/missing")
    return index, population


def phase_plan(index, population, phase):
    if phase == "mechanism":
        plan = list(build_mechanism_plan(index, candidate_layers=[19, 24, 26],
                    neighbor_layers=[18, 20, 23, 25], restoration_doses=[1., .5],
                    reverse_doses=[1.], random_replicates=3, seed=42,
                    primary_restoration_dose=1., coordinate="event", state_keys=(EVENT_CENTER,)))
        # Sensitivity only; the primary matched reverse controls remain dose 1.
        plan.extend(replace(p, dose=.5) for p in list(plan) if p.intervention == "reverse_suppression")
        if len(plan) != 920:
            raise ReplicationStop("Mechanism grid must retain exactly 20 dev pairs")
    elif phase == "window":
        common = set(population["full_window_pair_ids"])
        plan = []
        for row in index["records"]:
            if row["pair_id"] not in common or row["relative_step"] == 0:
                continue
            if not row["available"]:
                raise ReplicationStop("Complete window contains an unavailable offset")
            for layer in (19, 24, 26):
                for kind, repeat in (("r_direction", 0), ("h_direction_control", 0), ("sham", 0),
                                     *(("random_direction_control", i) for i in range(3))):
                    plan.append(PlannedTrial(row["pair_id"], "rq2_dev", row["state_key"],
                                row["step"], layer, kind, 1., seed=42, replicate=repeat))
        if len(plan) != len(common) * 72:
            raise ReplicationStop("Window grid coverage changed")
    else:
        raise ReplicationStop("Only mechanism/window exploration is supported")
    identities = [p.to_key("a" * 64).trial_id for p in plan]
    if len(set(identities)) != len(plan) or any(p.rq2_role != "rq2_dev" or p.layer == 27 for p in plan):
        raise ReplicationStop("Duplicate, formal, or unreachable trial in exploration")
    return tuple(plan)


def service_contract(source):
    # Request/rubric/transport stay fixed; execution scope has its OWN contract.
    current = contract_for(source.runtime_config.judge)
    old = source.raw["scoring_contract"]
    for key, value in current.items():
        if key in {"version", "implementation_sha256"}:
            continue
        if old.get(key) != value:
            raise ReplicationStop(f"Cannot reuse labels: scoring service field changed: {key}")
    for name, digest in current["implementation_sha256"].items():
        if old["implementation_sha256"].get(name) != digest:
            raise ReplicationStop("Historical rubric or transport implementation changed")
    result = copy.deepcopy(current)
    result.update(version=6, scope="post_oracle_exploratory_dev",
                  returned_model_policy="record_and_warn_never_stop_for_identity_alone",
                  model_version_locked=False, allows_formal_or_causal_claim=False,
                  source_contract_sha256=canonical_sha256(old),
                  reuse_policy="same_request_rubric_transport_exact_content_first_valid_receipt",
                  implementation_sha256={**current["implementation_sha256"],
                    "rq2/exploration_pipeline.py": file_sha256(ROOT / "rq2/exploration_pipeline.py")})
    return result


def estimate(parent, population):
    ledger = parent._judge().summary()
    timing_root = parent.spec.source.source.output_root / "timings"
    timings = [read_object(p) for p in timing_root.glob("oracle_generate_*.json")]
    seconds = sum(t["elapsed_seconds"] for t in timings)
    count = 920 + 72 * population["full_window_pair_count"]
    average = ledger["accounted_cny_at_peak_rates"] / max(ledger["http_attempts"], 1)
    nominal = count * seconds / 320 / 3600 if seconds else None
    return {"new_pgd_runs": 0, "new_mechanism_responses": 920,
            "new_window_responses": count - 920, "maximum_new_responses": count,
            "baseline_generation": 0, "historical_oracle_seconds_for_320": seconds,
            "gpu_hours_linear_reference": nominal,
            "gpu_hours_planning_range": [round(nominal * .75, 1), round(nominal * 1.5 + .5, 1)] if nominal else None,
            "api_first_attempt_reference_cny_no_cache_saving": round(count * average, 3),
            "api_cumulative_spent_cny": ledger["cumulative_accounted_cny_at_peak_rates"],
            "api_cumulative_pause_cny": 30,
            "api_remaining_allowance_cny": round(30 - ledger["cumulative_accounted_cny_at_peak_rates"], 6),
            "basis": "historical throughput and accounting prices; not a guarantee or provider invoice; GPU rental price unknown"}


def draft_config():
    source = load_warning_config(ROOT / SOURCE)
    parent = DevWarningPipeline(source, transport=lambda body: (_ for _ in ()).throw(ReplicationStop("Source API forbidden")))
    with source_locks(source):
        # load_warning_config already checks the frozen config/code/environment
        # chain. Do not call its recovery-specific recursive paid-history audit:
        # it re-hashes ALL trajectories and is unrelated to this selected replay.
        inputs = inventory(parent)
        index, population = event_index(parent)
        ledger = parent._judge().summary()
        if ledger["unresolved_attempts"] or parent._judge().state.get("fatal_error"):
            raise ReplicationStop("Source accounting is unresolved")
        return {"format": "rq2-post-oracle-exploration", "version": 1, "name": NAME,
                "source_config": SOURCE, "output_root": "outputs/stage2_rq2/exploratory/" + NAME,
                "design": DESIGN, "source_inventory_sha256": canonical_sha256(inputs),
                "source_inventory_count": len(inputs),
                "pair_ids": list(source.pair_ids), "window_pair_ids": population["full_window_pair_ids"],
                "event_index_sha256": canonical_sha256(index),
                "scoring_contract": service_contract(source),
                "budget": {"api_limit_cny": 30, "http_attempt_limit": 7080,
                           "input_cny_per_million": 2, "output_cny_per_million": 9},
                "history": {"cost_micro_cny": round(ledger["cumulative_accounted_cny_at_peak_rates"] * 1e6),
                            "http_attempts": ledger["cumulative_http_attempts"]},
                "estimate": estimate(parent, population),
                "bound_inputs": {p: file_sha256(ROOT / p) for p in (SOURCE, *IMPLEMENTATION)}}


@dataclass(frozen=True)
class ExplorationConfig:
    path: Path
    raw: dict
    source: object

    @property
    def output_root(self):
        return ROOT / self.raw["output_root"]

    @property
    def fingerprint(self):
        return canonical_sha256(self.raw)

    @property
    def runtime_config(self):
        return replace(self.source.runtime_config, path=self.path, name=NAME, raw=self.raw,
                       fingerprint=self.fingerprint, output_root=self.output_root,
                       sampling={"fixed_steps": [], "event_offsets": [-2, -1, 0, 1, 3]})


def load_config(path):
    path = Path(path).resolve()
    raw = read_object(path)
    if (raw.get("format") != "rq2-post-oracle-exploration" or raw.get("version") != 1
            or raw.get("name") != NAME or raw.get("source_config") != SOURCE
            or raw.get("design") != DESIGN
            or raw.get("output_root") != "outputs/stage2_rq2/exploratory/" + NAME
            or raw.get("budget") != {"api_limit_cny": 30, "http_attempt_limit": 7080,
                                    "input_cny_per_million": 2, "output_cny_per_million": 9}
            or set(raw.get("bound_inputs", {})) != {SOURCE, *IMPLEMENTATION}):
        raise ReplicationStop("Not the frozen dev-only exploration config")
    for name, digest in raw["bound_inputs"].items():
        if file_sha256(ROOT / name) != digest:
            raise ReplicationStop(f"Exploration input changed: {name}; do not rebind an active run")
    out = ROOT / raw["output_root"]
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ReplicationStop("Exploration output cannot be symlinked")
    source = load_warning_config(ROOT / SOURCE)
    if service_contract(source) != raw["scoring_contract"] or list(source.pair_ids) != raw["pair_ids"]:
        raise ReplicationStop("Source population/scoring scope changed")
    return ExplorationConfig(path, raw, source)
