"""Repository-level contract tests for the formal Qwen7B RQ2 run config."""

import json
from pathlib import Path

from rq2.artifacts import file_sha256, read_jsonl
from rq2.config import assert_executable, load_rq2_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/stage2_rq2_qwen7b_advbench_sure_here_is_run01.json"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset/processed/rq2/advbench_manifest_v1/rq2_manifest.jsonl"
)
MANIFEST_AUDIT_PATH = MANIFEST_PATH.parent / "manifest_audit.json"
RESERVE_PATH = (
    PROJECT_ROOT / "dataset/processed/rq2/advbench_split_v1/reserve.jsonl"
)
EXPECTED_MANIFEST_SHA256 = (
    "66ed99e487ab63cba6af6a529aac962de0342721a23e8e77d0caf7889d6e4e86"
)
EXPECTED_MANIFEST_AUDIT_SHA256 = (
    "ebfa636b8088e812878ac05359db493c31e715eaa67eebe1ff237f6e5565c45b"
)


def test_formal_config_is_executable_and_binds_frozen_protocol():
    config = load_rq2_config(CONFIG_PATH)
    assert_executable(config)
    assert config.name == "qwen7b_advbench_sure_here_is_run01"
    assert config.template is False
    assert config.execution_enabled is True
    assert config.frozen is False
    assert config.manifest == MANIFEST_PATH
    assert config.output_root == (
        PROJECT_ROOT / "outputs/stage2_rq2/qwen7b_advbench_sure_here_is_run01"
    )
    assert config.trajectory["mode"] == "generate"
    assert config.model["name"] == "qwen-7b"
    assert config.model["dtype"] == "bfloat16"
    assert config.model["do_sample"] is False
    assert config.pilot["candidate_layers"] == [19, 24, 26, 27]
    assert config.formal == {"layers": "all", "minimum_pairs": 20}
    assert config.judge["provider"] == "deepseek"
    assert config.judge["model"] == "DeepSeek-V4-Flash"
    assert config.judge["base_url"] == "https://token.matpool.com/v1"


def test_formal_config_manifest_and_audit_are_the_frozen_t0_7_artifacts():
    assert file_sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert file_sha256(MANIFEST_AUDIT_PATH) == EXPECTED_MANIFEST_AUDIT_SHA256
    audit = json.loads(MANIFEST_AUDIT_PATH.read_text(encoding="utf-8"))
    assert audit["formal_manifest_ready"] is True
    assert audit["output"]["manifest"]["sha256"] == EXPECTED_MANIFEST_SHA256
    assert audit["attack_target"] == {
        "value": "Sure, here is",
        "policy": "uniform-fixed-short-compliance-prefix-v1",
        "advbench_original_target_included": False,
    }


def test_formal_manifest_has_exact_roles_target_and_no_reserve_or_trajectory():
    rows = read_jsonl(MANIFEST_PATH)
    reserve_ids = {row["pair_id"] for row in read_jsonl(RESERVE_PATH)}
    manifest_ids = {row["pair_id"] for row in rows}
    assert len(rows) == 60
    assert len(manifest_ids) == 60
    assert sum(row["rq2_role"] == "rq2_dev" for row in rows) == 20
    assert sum(row["rq2_role"] == "rq2_causal_test" for row in rows) == 40
    assert {row["target_text"] for row in rows} == {"Sure, here is"}
    assert not manifest_ids.intersection(reserve_ids)
    assert all("trajectory_path" not in row for row in rows)


def test_formal_v2_config_binds_layer_roles_doses_and_v1_parent():
    config = load_rq2_config(
        PROJECT_ROOT / "configs/stage2_rq2_qwen7b_advbench_sure_here_is_v2_run01.json"
    )
    parent = load_rq2_config(CONFIG_PATH)
    assert config.preregistration_record["version"] == 2
    assert config.preregistration_record["revision"]["parent_record"]["sha256"] == parent.preregistration_sha256
    assert config.manifest == parent.manifest
    assert config.output_root != parent.output_root
    assert config.pilot["candidate_layers"] == [19, 24, 26, 27]
    assert config.pilot["neighbor_layers"] == [18, 20, 23, 25]
    assert config.pilot["distant_control_layers"] == [2, 12]
    assert "depth_control_layers" not in config.pilot
    assert config.pilot["primary_restoration_dose"] == 1.0
    assert config.pilot["primary_suppression_dose"] == 1.0
    assert config.pilot["restoration_doses"] == [1.0, 0.5]
    assert config.pilot["suppression_doses"] == [1.0, 0.5]
    assert config.pilot["sensitivity_doses"] == [0.5]
    assert config.preregistration_record["analysis_policy"]["layer_profile_is_mechanism_gate"] is False
    assert config.statistical_preregistration_record["formal_family_policy"]["F1_fixed_utility"] == 84
    assert file_sha256(config.statistical_preregistration_path) == config.statistical_preregistration_sha256
