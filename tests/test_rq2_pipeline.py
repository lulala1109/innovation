import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pydantic")

from rq2.artifacts import atomic_json, file_sha256
from rq2.config import STAGE_ORDER, STAGE_RESOURCES, load_rq2_config, select_stages
from rq2.data import (
    RQ2DataError,
    RQ2Pair,
    build_state_index,
    derive_behavior_events,
    load_behavior_events,
    load_rq2_manifest,
    validate_trajectory_contract,
)
from rq2.experiment import RQ2ExperimentError, build_formal_plan, build_mechanism_plan
from rq2.pipeline import RQ2Pipeline, RQ2PipelineError
from rq2.rq1_bundle import RQ1BundleError, load_frozen_rq1_bundle
from core.safety_state import DualSafetyStateScorer


def test_first_event_is_unresolved_if_an_earlier_judge_step_is_missing():
    trials = [
        {"pair_id": pair, "trial_id": f"{pair}:{state}", "state_key": f"scan:{state}",
         "step": 0 if state == "clean" else int(state)}
        for pair in ("p", "q") for state in ("clean", "0", "1", "2")
    ]
    def label(pair, state, refused):
        return {"trial_id": f"{pair}:{state}", "label_status": "ok",
                "refusal_label": refused, "compliance_label": not refused,
                "refusal_orientation": 1.0 if refused else 0.0}
    labels = [
        label("p", "clean", True), label("p", "0", True),
        label("p", "2", False),
        label("q", "clean", True), label("q", "0", True),
        label("q", "1", False),
    ]
    events = derive_behavior_events(trials, labels, refusal_weakening_delta=0.2)
    by_pair = {row["pair_id"]: row for row in events["records"]}
    assert events["version"] == 3
    assert by_pair["p"]["first_non_refusal_step"] is None
    assert by_pair["p"]["first_non_refusal_step_status"] == "unresolved_prior_missing_step"
    assert by_pair["q"]["first_non_refusal_step"] == 1
    assert by_pair["q"]["first_non_refusal_step_status"] == "observed"
    assert events["population_counts"]["events"]["first_non_refusal_step"]["unresolved_pairs"] == 1


def test_stage_graph_contains_rq2_event_and_subspace_gates():
    assert STAGE_ORDER.index("events") < STAGE_ORDER.index("state_index")
    assert STAGE_ORDER.index("subspace_analyze") < STAGE_ORDER.index("protocol_lock")
    assert STAGE_RESOURCES["trajectory_behavior_judge"] == ("api",)


def test_contiguous_stage_selection():
    selected = select_stages(from_stage="oracle_generate", through_stage="oracle_analyze")
    assert selected == ("oracle_generate", "oracle_judge", "oracle_analyze")


def _state_index(role="rq2_dev"):
    return {"records": [{
        "pair_id": "p", "rq2_role": role, "state_key": "fixed:2",
        "coordinate": "fixed", "step": 2, "available": True,
    }]}


def test_mechanism_plan_includes_norm_matched_reverse_controls():
    plan = build_mechanism_plan(
        _state_index(), candidate_layers=(4,), restoration_doses=(1.0,),
        reverse_doses=(0.5,), random_replicates=2, seed=7,
    )
    reverse = [item for item in plan if item.state_key == "clean"]
    assert {item.intervention for item in reverse} == {
        "reverse_suppression", "reverse_h_control", "reverse_random_control",
        "reverse_sham",
    }
    assert sum(item.intervention == "reverse_random_control" for item in reverse) == 2


def test_formal_subspace_plan_uses_subspace_specific_controls():
    protocol = {
        "format": "rq2-protocol-lock", "version": 1, "locked": True,
        "primary_intervention": "subspace_restoration",
        "primary_restoration_dose": 1.0, "primary_suppression_dose": 0.5,
        "formal_layers": [4], "control_layers": [4], "candidate_layers": [4],
        "random_replicates": 1, "seed": 9,
    }
    plan = build_formal_plan(_state_index("rq2_causal_test"), protocol)
    kinds = {item.intervention for item in plan}
    assert "subspace_h_control" in kinds
    assert "subspace_random_control" in kinds
    assert "h_direction_control" not in kinds


class _DummyConfig:
    def __init__(self, primary: Path):
        self.primary = primary

    def stage_path(self, stage):
        return self.primary


def test_stage_freshness_checks_every_recorded_artifact(tmp_path):
    primary = tmp_path / "responses.jsonl"
    trials = tmp_path / "trials.jsonl"
    primary.write_text("response\n", encoding="utf-8")
    trials.write_text("trial\n", encoding="utf-8")
    pipeline = RQ2Pipeline(_DummyConfig(primary))
    record = {"artifacts": {
        str(primary): file_sha256(primary),
        str(trials): file_sha256(trials),
    }}
    assert pipeline._stage_is_fresh("formal_generate", record) is True
    trials.write_text("changed\n", encoding="utf-8")
    assert pipeline._stage_is_fresh("formal_generate", record) is False


def _trajectory_pair(tmp_path: Path):
    case = tmp_path / "case"
    trajectory = case / "trajectory"
    trajectory.mkdir(parents=True)
    fingerprint = "a" * 64
    checkpoints = []
    for step in range(101):
        checkpoint = trajectory / f"step_{step:06d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        checkpoints.append({
            "step": step,
            "path": checkpoint.name,
            "metadata": {
                "step": step, "pair_id": "p",
                "experiment_fingerprint": fingerprint,
            },
        })
    index = trajectory / "index.json"
    index.write_text(json.dumps({
        "format": "safety-state-trajectory", "version": 1,
        "checkpoints": checkpoints,
    }), encoding="utf-8")
    clean = tmp_path / "clean.wav"
    clean.write_bytes(b"clean")
    clean_sha = hashlib.sha256(b"clean").hexdigest()
    experiment = {
        "method": "standard", "loss_type": "margin", "steps": 100,
        "init_mode": "zero", "early_stop": False, "save_all_steps": True,
        "eps": 0.1, "alpha": 0.005, "kappa": 5.0,
        "model": "qwen-7b", "model_id": "weights", "dtype": "bfloat16",
        "target_text": "Sure, here is", "harmful_text": "harmful",
        "input_audio": {"sha256": clean_sha},
    }
    run = {
        "pair_id": "p",
        "budget": {
            "experiment_config": experiment,
            "experiment_fingerprint": fingerprint,
            "norm": "linf",
        },
    }
    (case / "run.json").write_text(json.dumps(run), encoding="utf-8")
    pair = RQ2Pair(
        pair_id="p", role="rq2_dev", clean_audio_path=clean,
        trajectory_path=index, harmful_text="harmful", clean_refused=True,
        clean_audio_sha256=clean_sha, trajectory_index_sha256=file_sha256(index),
        content_group="g", row={"target_text": "Sure, here is"},
    )
    expected = {
        "steps": 100, "eps": 0.1, "alpha": 0.005, "kappa": 5.0,
        "model": "qwen-7b", "model_id": "weights", "dtype": "bfloat16",
        "target_text": "Sure, here is", "harmful_text": "harmful",
    }
    return pair, expected, case / "run.json", run


def test_trajectory_contract_binds_run_and_all_checkpoints(tmp_path):
    pair, expected, run_path, run = _trajectory_pair(tmp_path)
    result = validate_trajectory_contract(pair, expected=expected)
    assert result["checkpoint_count"] == 101
    assert len(result["checkpoints"]) == 101
    run["budget"]["experiment_config"]["alpha"] = 0.01
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RQ2DataError, match="provenance mismatch"):
        validate_trajectory_contract(pair, expected=expected)


def test_content_groups_cannot_hide_identical_harmful_text(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [
        {
            "pair_id": "dev", "rq2_role": "rq2_dev",
            "clean_audio_path": "dev.wav", "trajectory_path": "dev.json",
            "harmful_text": "  Same   Harmful Prompt ", "clean_refused": True,
            "content_group": "group-a",
        },
        {
            "pair_id": "test", "rq2_role": "rq2_causal_test",
            "clean_audio_path": "test.wav", "trajectory_path": "test.json",
            "harmful_text": "same harmful prompt", "clean_refused": True,
            "content_group": "group-b",
        },
    ]}), encoding="utf-8")
    with pytest.raises(RQ2DataError, match="content_group"):
        load_rq2_manifest(manifest, require_paths=False)


def test_missing_event_offset_is_not_replaced_by_neighbor(tmp_path):
    pair, _, _, _ = _trajectory_pair(tmp_path)
    index = build_state_index(
        [pair],
        behavior_events={"p": {
            "judged_steps": list(range(101)),
            "first_refusal_weakening_step": None,
            "first_non_refusal_step": 1,
            "first_compliance_step": None,
        }},
    )
    row = next(
        item for item in index["records"]
        if item["state_key"] == "event:first_non_refusal_step:-2"
    )
    assert row["available"] is False
    assert row["step"] is None
    assert row["checkpoint_path"] is None


def test_event_offset_with_missing_scan_judge_stays_unavailable(tmp_path):
    pair, _, _, _ = _trajectory_pair(tmp_path)
    trials = [
        {"pair_id": "p", "trial_id": f"p:{state}", "state_key": f"scan:{state}",
         "step": 0 if state == "clean" else int(state)}
        for state in ("clean", "0", "1", "2")
    ]
    labels = [
        {"trial_id": f"p:{state}", "label_status": "ok",
         "refusal_label": refused, "compliance_label": not refused,
         "refusal_orientation": 1.0 if refused else 0.0}
        for state, refused in (("clean", True), ("0", True), ("1", False))
    ]
    events = derive_behavior_events(
        trials, labels, refusal_weakening_delta=0.2, event_offsets=(0, 1)
    )
    events_path = tmp_path / "events.json"
    atomic_json(events_path, events)
    recorded = load_behavior_events(events_path)
    assert recorded["p"]["judged_steps"] == [0, 1]
    index = build_state_index(
        [pair], fixed_steps=(2,), event_offsets=(0, 1), behavior_events=recorded
    )
    rows = {item["state_key"]: item for item in index["records"]}
    assert rows["fixed:2"]["available"] is True
    assert rows["event:first_non_refusal_step:+0"]["available"] is True
    assert rows["event:first_non_refusal_step:+1"]["available"] is False
    assert rows["event:first_non_refusal_step:+1"]["step"] is None


def test_pipeline_rejects_layers_outside_frozen_bundle():
    class LayerConfig:
        pilot = {
            "candidate_layers": [27, 28],
            "neighbor_layers": [26],
            "depth_control_layers": [2],
        }

    bundle = type(
        "Bundle",
        (),
        {"hidden_sizes": {layer: 3584 for layer in range(28)}},
    )()
    with pytest.raises(RQ2PipelineError, match=r"0\.\.27.*28"):
        RQ2Pipeline(LayerConfig())._validate_configured_layers(bundle)


def test_frozen_rq1_bundle_checks_sha_model_direction_and_sigma(tmp_path):
    layer_count = 28
    hidden_width = 3584
    hidden_sizes = {layer: hidden_width for layer in range(layer_count)}
    states_path = tmp_path / "states.pt"
    calibration = torch.zeros(4, hidden_width)
    calibration[0, 0] = 1.0
    calibration[1, 0] = 2.0
    calibration[3, 1] = 1.0
    hidden = {
        layer: calibration.clone()
        for layer in range(layer_count)
    }
    torch.save({
        "hidden_states": hidden,
        "refusal_labels": torch.tensor([1, 1, 0, 0]),
        "states": ["X_H", "X_H", "X_H", "X_J"],
    }, states_path)
    states_sha = file_sha256(states_path)
    scorer = DualSafetyStateScorer(hidden_size=hidden_sizes, trainable=False)
    negative = torch.zeros(hidden_width)
    positive = torch.zeros(hidden_width)
    positive[0] = 1.0
    means = {
        state: {
            layer: {
                "negative": negative.clone(),
                "positive": positive.clone(),
            }
            for layer in range(layer_count)
        }
        for state in ("harmfulness", "refusal")
    }
    directions = {
        state: {layer: positive.clone() for layer in range(layer_count)}
        for state in ("harmfulness", "refusal")
    }
    checkpoint = {
        "format": "dual-safety-state-layerwise-linear-probes",
        "version": 2,
        "hidden_sizes": hidden_sizes,
        "state_dict": scorer.state_dict(),
        "directions": directions,
        "class_means": means,
        "metadata": {"provenance": {"training_payload_metadata": {
            "pooling": "mean", "token_span": "audio",
            "sequence_has_embedding": True,
            "model_fingerprint": "model-fingerprint",
            "source_payload_sha256": states_sha,
        }}},
    }
    probe_path = tmp_path / "probe.pt"
    torch.save(checkpoint, probe_path)
    bundle = load_frozen_rq1_bundle(
        probe_path, states_path,
        expected_probe_sha256=file_sha256(probe_path),
        expected_training_sha256=states_sha,
        expected_model_fingerprint="model-fingerprint",
    )
    assert len(bundle.hidden_sizes) == 28
    assert set(bundle.hidden_sizes.values()) == {3584}
    assert bundle.refusal_sigma_count == 2
    with pytest.raises(RQ1BundleError, match="probe SHA"):
        load_frozen_rq1_bundle(
            probe_path, states_path, expected_probe_sha256="0" * 64
        )
    wrong_direction = torch.zeros(hidden_width)
    wrong_direction[1] = 1.0
    checkpoint["directions"]["refusal"][0] = wrong_direction
    bad_probe = tmp_path / "bad_probe.pt"
    torch.save(checkpoint, bad_probe)
    with pytest.raises(RQ1BundleError, match="positive-negative"):
        load_frozen_rq1_bundle(
            bad_probe, states_path,
            expected_training_sha256=states_sha,
            expected_model_fingerprint="model-fingerprint",
        )


def test_sources_provenance_binds_preregistration_and_inputs(tmp_path, monkeypatch):
    config = load_rq2_config("configs/stage2_rq2_TEMPLATE.json")
    config = replace(config, output_root=tmp_path / "rq2")
    pipeline = RQ2Pipeline(config)
    bundle = SimpleNamespace(
        refusal_sigma={0: 1.0},
        refusal_sigma_population="refusal-positive-intersection-X_H",
        refusal_sigma_count=1,
    )
    monkeypatch.setattr(pipeline, "bundle", lambda: bundle)
    monkeypatch.setattr(pipeline, "_model_provenance", lambda: {})
    monkeypatch.setattr(
        "rq2.pipeline.write_bundle_summary",
        lambda _bundle, output: atomic_json(output, {
            "format": "frozen-rq1-bundle-summary",
            "version": 1,
        }),
    )
    pipeline._run_sources()
    summary = json.loads(config.stage_path("sources").read_text(encoding="utf-8"))
    prereg = summary["candidate_layer_preregistration"]
    assert prereg["sha256"] == config.preregistration_sha256
    assert prereg["candidate_layers"] == [19, 24, 26, 27]
    assert len(prereg["input_artifacts"]) == len(
        config.preregistration_record["inputs"]
    )
    artifacts = pipeline._stage_artifacts("sources")
    assert config.preregistration_path in artifacts
    assert all(
        Path(item["path"]).resolve() in artifacts
        for item in prereg["input_artifacts"]
    )


def test_mechanism_plan_uses_explicit_primary_dose_for_controls():
    plan = build_mechanism_plan(
        _state_index(), candidate_layers=(19,), restoration_doses=(0.5, 1.0),
        reverse_doses=(1.0, 0.5), primary_restoration_dose=1.0,
        random_replicates=1, seed=7,
    )
    restoration = [item for item in plan if item.state_key == "fixed:2"]
    assert {item.dose for item in restoration if item.intervention == "r_direction"} == {0.5, 1.0}
    assert {item.dose for item in restoration if item.intervention != "r_direction"} == {1.0}


def test_formal_v2_plan_excludes_sensitivity_and_rejects_changed_primary_dose():
    protocol = {
        "format": "rq2-protocol-lock", "version": 2, "locked": True,
        "layer_dose_version": 2, "primary_intervention": "r_direction",
        "primary_restoration_dose": 1.0, "primary_suppression_dose": 1.0,
        "sensitivity_doses": [0.5], "formal_layers": [19, 20],
        "control_layers": [19, 20], "candidate_layers": [19],
        "random_replicates": 1, "seed": 9,
    }
    plan = build_formal_plan(_state_index("rq2_causal_test"), protocol)
    assert plan
    assert {item.dose for item in plan} == {1.0}
    for field in ("primary_restoration_dose", "primary_suppression_dose"):
        with pytest.raises(RQ2ExperimentError, match="fixed at 1.0"):
            build_formal_plan(
                _state_index("rq2_causal_test"), {**protocol, field: 0.5},
            )
