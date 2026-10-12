"""Frozen, independent calibration A configuration and read-only source binding."""
from dataclasses import dataclass, replace
import copy
from pathlib import Path

from rq2.artifacts import canonical_sha256, file_sha256
from rq2.exploration_config import CONFIG as SOURCE, ROOT, STAGES as SOURCE_STAGES, load_config as load_source, source_locks
from rq2.exploration_network_resume import NetworkResumePipeline, forbidden_transport
from rq2.orthogonal_interventions import OrthogonalTrial, TOLERANCES
from rq2.replication_config import read_object
from rq2.replication_judge import ReplicationStop

NAME = "qwen7b_advbench_orthogonal_calibration_a_v1_run01"
CONFIG = "configs/stage2_rq2_" + NAME + ".json"
PROTOCOL = "docs/RQ2_正交对照校准A协议_v1_2026-10-12.md"
STAGES = ("prepare", "numeric_preflight", "calibration_generate", "calibration_judge", "calibration_analyze", "report")
CENTER = "event:first_non_refusal_step:+0"
DESIGN = {"layers": [19, 24], "pair_count": 20, "center": CENTER,
          "dose": 1., "seed": 42, "random_replicates": 3,
          "token_scope": "audio", "application": "prefill_once",
          "intervention_responses": 240, "baseline_responses": 20, "total_responses": 260,
          "numeric_preflight_conditions": 240, "tolerances": TOLERANCES,
          "seed_policy": "existing_pair_state_layer_replicate_hash_then_plus_1000003_times_replicate",
          "bootstrap_replicates": 2000, "formal_allowed": False, "protocol_locked": False,
          "model_version_locked": False, "allow_reverse": False, "allow_subspace": False,
          "new_pgd_runs": 0, "causal_test_pairs": 0}
BUDGET = {"api_limit_cny": 30, "http_attempt_limit": 7080,
          "input_cny_per_million": 2, "output_cny_per_million": 9}
IMPLEMENTATION = ("rq2/orthogonal_interventions.py", "rq2/orthogonal_config.py",
                  "rq2/orthogonal_pipeline.py", "experiments/rq2_orthogonal_calibration.py", PROTOCOL)


def plan_trials(index, pair_ids):
    rows = [r for r in index["records"] if r["state_key"] == CENTER and r["available"]]
    if (len(pair_ids) != 20 or len(set(pair_ids)) != 20 or len(rows) != 20
            or {r["pair_id"] for r in rows} != set(pair_ids)
            or any(r["rq2_role"] != "rq2_dev" for r in rows)):
        raise ReplicationStop("Calibration A requires the exact twenty frozen event-center dev pairs")
    by_pair = {r["pair_id"]: r for r in rows}
    return tuple(OrthogonalTrial(pair, "rq2_dev", CENTER, by_pair[pair]["step"], layer, kind, 1.,
                                seed=42, replicate=rep)
                 for pair in pair_ids for layer in (19, 24)
                 for kind, rep in (("r_direction", 0), ("h_orthogonal_control", 0),
                                   *(("random_orthogonal_control", i) for i in range(3)), ("sham", 0)))


def source_snapshot(parent):
    parent.verify()
    state = parent._state()["stages"]
    if set(state) != set(SOURCE_STAGES) or any(not parent._fresh(s, state[s]) for s in SOURCE_STAGES):
        raise ReplicationStop("The source exploration must be fully complete and unchanged")
    judge = parent._judge()
    summary = judge.summary()
    if summary["unresolved_attempts"] or judge.state.get("fatal_error"):
        raise ReplicationStop("The completed source recovery has unresolved accounting")
    for row in judge.state["attempts"]:
        payload = read_object(judge.root / "receipts" / f"{row['number']:06d}.json")
        usage = payload.get("usage", {})
        p, c = row.get("prompt_tokens"), row.get("completion_tokens")
        if (type(p) is not int or type(c) is not int or min(p, c) < 0 or c > 800
                or p > row["prompt_token_reservation"] or usage.get("prompt_tokens") != p
                or usage.get("completion_tokens") != c or usage.get("total_tokens") != p+c
                or row["cost_micro_cny"] != p*2+c*9):
            raise ReplicationStop("Source recovery receipt/accounting mismatch")
    paths = sorted(p for p in parent.config.output_root.rglob("*") if p.is_file() and p.name != ".run.lock")
    if any(p.is_symlink() for p in parent.config.output_root.rglob("*")):
        raise ReplicationStop("Source contains a symlink")
    inventory = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    history = {"cost_micro_cny": round(summary["cumulative_accounted_cny_at_peak_rates"] * 1e6),
               "http_attempts": summary["cumulative_http_attempts"]}
    uncertain = {k: summary[k] for k in ("historical_uncertain_attempts", "historical_uncertain_reserved_cny",
                                        "historical_uncertain_charge_policy")}
    return inventory, history, uncertain


def service_contract(source):
    contract = copy.deepcopy(source.raw["scoring_contract"])
    contract.update(version=7, scope="orthogonal_calibration_a_exploratory_dev",
                    source_contract_sha256=canonical_sha256(source.raw["scoring_contract"]))
    contract["implementation_sha256"]["rq2/orthogonal_pipeline.py"] = file_sha256(ROOT / "rq2/orthogonal_pipeline.py")
    return contract


def draft_config():
    source = load_source(ROOT / SOURCE)
    parent = NetworkResumePipeline(source, 234, transport=forbidden_transport)
    with source_locks(source):
        inputs, history, uncertain = source_snapshot(parent)
        plan_trials(parent.index, source.raw["pair_ids"])
        return {"format": "rq2-orthogonal-calibration-a", "version": 1, "name": NAME,
                "source_config": SOURCE, "source_recovery_attempt": 234,
                "output_root": "outputs/stage2_rq2/calibration/" + NAME, "design": DESIGN,
                "pair_ids": source.raw["pair_ids"], "source_inventory_sha256": canonical_sha256(inputs),
                "source_inventory_count": len(inputs), "event_index_sha256": canonical_sha256(parent.index),
                "scoring_contract": service_contract(source), "budget": BUDGET,
                "history": history, "historical_uncertain": uncertain,
                "bound_inputs": {p: file_sha256(ROOT / p) for p in (SOURCE, *IMPLEMENTATION)}}


@dataclass(frozen=True)
class OrthogonalConfig:
    path: Path
    raw: dict
    source: object

    @property
    def fingerprint(self):
        return canonical_sha256(self.raw)

    @property
    def output_root(self):
        return ROOT / self.raw["output_root"]

    @property
    def runtime_config(self):
        return replace(self.source.runtime_config, path=self.path, name=NAME, raw=self.raw,
                       fingerprint=self.fingerprint, output_root=self.output_root,
                       sampling={"fixed_steps": [], "event_offsets": [0]})


def load_config(path):
    path = Path(path).resolve()
    raw = read_object(path)
    if (raw.get("format") != "rq2-orthogonal-calibration-a" or raw.get("version") != 1
            or raw.get("name") != NAME or raw.get("source_config") != SOURCE
            or raw.get("source_recovery_attempt") != 234 or raw.get("design") != DESIGN
            or raw.get("budget") != BUDGET or raw.get("output_root") != "outputs/stage2_rq2/calibration/" + NAME
            or set(raw.get("bound_inputs", {})) != {SOURCE, *IMPLEMENTATION}):
        raise ReplicationStop("Not the frozen calibration A configuration")
    for name, digest in raw["bound_inputs"].items():
        if file_sha256(ROOT / name) != digest:
            raise ReplicationStop(f"Calibration input changed: {name}; do not rebind an active run")
    source = load_source(ROOT / SOURCE)
    if raw["pair_ids"] != source.raw["pair_ids"] or raw["scoring_contract"] != service_contract(source):
        raise ReplicationStop("Population or scoring service changed")
    spec = OrthogonalConfig(path, raw, source)
    if any(p.is_symlink() for p in (spec.output_root, *spec.output_root.parents)):
        raise ReplicationStop("Calibration output cannot be symlinked")
    return spec
