import json
from pathlib import Path

import pytest

from rq2.artifacts import file_sha256
from rq2.config import RQ2ConfigError, assert_executable, load_rq2_config


PREREGISTRATION_PATH = Path(
    "configs/rq2_candidate_layer_preregistration_qwen7b_v1.json"
).resolve()


def _config(tmp_path: Path, **updates):
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["preregistration"] = {
        "path": str(PREREGISTRATION_PATH),
        "sha256": file_sha256(PREREGISTRATION_PATH),
    }
    raw.update({"name": "unit", "output_root": "outputs/stage2_rq2/unit", **updates})
    path = configs / "unit.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_rq2_config(path)


def test_template_is_not_executable(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(RQ2ConfigError, match="template"):
        assert_executable(config)


def test_frozen_config_is_not_executable(tmp_path):
    config = _config(
        tmp_path, template=False, execution_enabled=True, frozen=True
    )
    with pytest.raises(RQ2ConfigError, match="frozen"):
        assert_executable(config)


def test_rq1_output_namespace_is_protected(tmp_path):
    with pytest.raises(RQ2ConfigError, match="RQ1 namespace"):
        _config(tmp_path, output_root="outputs/stage1/rq2")


def test_sampling_must_be_deterministic(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["model"]["do_sample"] = True
    with pytest.raises(RQ2ConfigError, match="deterministic"):
        _config(tmp_path, model=raw["model"])


def test_registered_state_coordinates_cannot_drift_from_preregistration(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["sampling"]["fixed_steps"] = [1, 10, 100]
    with pytest.raises(RQ2ConfigError, match="sampling.fixed_steps"):
        _config(tmp_path, sampling=raw["sampling"])


@pytest.mark.parametrize("fixed_steps", ([10], [10, 2], [2, 101]))
def test_registered_state_coordinates_must_fit_trajectory(tmp_path, fixed_steps):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["sampling"]["fixed_steps"] = fixed_steps
    with pytest.raises(RQ2ConfigError, match="fixed_steps"):
        _config(tmp_path, sampling=raw["sampling"])


def test_candidate_layers_cannot_drift_from_preregistration(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["pilot"]["candidate_layers"] = [32]
    with pytest.raises(RQ2ConfigError, match="pilot.candidate_layers"):
        _config(tmp_path, pilot=raw["pilot"])


def test_negative_layer_is_rejected_without_loading_bundle(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["pilot"]["candidate_layers"] = [-1]
    with pytest.raises(RQ2ConfigError, match="negative"):
        _config(tmp_path, pilot=raw["pilot"])


def test_existing_trajectory_also_requires_one_hundred_steps(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["trajectory"]["steps"] = 99
    with pytest.raises(RQ2ConfigError, match="exactly 100"):
        _config(tmp_path, trajectory=raw["trajectory"])


def test_event_analysis_contract_uses_actual_summary_name(tmp_path):
    config = _config(tmp_path)
    assert config.stage_path("event_analyze").name == "event_analysis.json"


def test_formal_scan_cannot_be_downgraded_to_smoke(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["formal"] = {"layers": [4], "minimum_pairs": 3}
    with pytest.raises(RQ2ConfigError, match="formal.layers"):
        _config(tmp_path, formal=raw["formal"])


def test_schema_v3_binds_preregistration_record(tmp_path):
    config = _config(tmp_path)
    assert config.raw["schema_version"] == 3
    assert config.preregistration_path == PREREGISTRATION_PATH
    assert config.preregistration_record["selection_results"]["candidate_layers"] == [
        19, 24, 26, 27,
    ]


def test_preregistration_sha_mismatch_is_rejected(tmp_path):
    with pytest.raises(RQ2ConfigError, match="preregistration SHA mismatch"):
        _config(tmp_path, preregistration={
            "path": str(PREREGISTRATION_PATH),
            "sha256": "0" * 64,
        })


def test_registered_dose_order_cannot_drift(tmp_path):
    raw = json.loads(Path("configs/stage2_rq2_TEMPLATE.json").read_text())
    raw["pilot"]["restoration_doses"] = [0.5, 1.0, 1.5]
    with pytest.raises(RQ2ConfigError, match="pilot.restoration_doses"):
        _config(tmp_path, pilot=raw["pilot"])
