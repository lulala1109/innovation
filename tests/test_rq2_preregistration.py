import csv
import json
import os
from pathlib import Path

import pytest

from rq2.artifacts import file_sha256
from rq2.preregistration import (
    RQ2PreregistrationError,
    build_preregistration,
    check_preregistration,
    serialize_record,
    validate_record_inputs,
    write_preregistration,
)


METRICS = (
    "H_direction",
    "H_probe",
    "H_probe_minus_R_probe",
    "R_direction",
    "R_probe",
)


def _write_slopes(path: Path, population: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_order = [24, 19, 27, 26, 25] + [layer for layer in range(28) if layer not in {24, 19, 27, 26, 25}]
    refused_order = [27, 19, 24, 17, 26] + [layer for layer in range(28) if layer not in {27, 19, 24, 17, 26}]
    order = all_order if population == "all" else refused_order
    refusal_slopes = {
        layer: -0.5 + 0.01 * rank for rank, layer in enumerate(order)
    }
    pair_count = 20 if population == "all" else 17
    fields = [
        "metric", "layer", "p_value", "slope_mean", "slope_variance",
        "slope_standard_error", "slope_ci_low", "slope_ci_high",
        "slope_confidence", "slope_sample_count", "slope_pair_count",
        "fdr_q_value",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            for layer in range(28):
                if metric == "R_probe":
                    slope = refusal_slopes[layer]
                elif metric == "R_direction":
                    slope = -float(layer + 1)
                elif metric == "H_direction":
                    slope = float(layer + 1)
                elif metric == "H_probe_minus_R_probe":
                    slope = 0.3
                else:
                    slope = 0.02
                writer.writerow({
                    "metric": metric,
                    "layer": layer,
                    "p_value": 0.001,
                    "slope_mean": slope,
                    "slope_variance": 0.01,
                    "slope_standard_error": 0.01,
                    "slope_ci_low": slope - 0.01,
                    "slope_ci_high": slope + 0.01,
                    "slope_confidence": 0.95,
                    "slope_sample_count": pair_count,
                    "slope_pair_count": pair_count,
                    "fdr_q_value": 0.01,
                })


def _write_profiles(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric", "step", "progress", "phase", "icc_2k",
        "correlation_count", "loo_spearman_mean", "loo_spearman_variance",
        "loo_spearman_standard_error", "loo_spearman_ci_low",
        "loo_spearman_ci_high", "loo_spearman_confidence",
        "loo_spearman_sample_count", "loo_spearman_pair_count",
        "loo_spearman_bootstrap_valid_replicates",
        "loo_spearman_bootstrap_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in ("H_probe", "R_probe", "R_direction"):
            for step in (2, 10, 100):
                writer.writerow({
                    "metric": metric,
                    "step": step,
                    "progress": step / 100,
                    "phase": "late" if step == 100 else "early",
                    "icc_2k": 0.8,
                    "correlation_count": 20,
                    "loo_spearman_mean": 0.9,
                    "loo_spearman_variance": 0.01,
                    "loo_spearman_standard_error": 0.01,
                    "loo_spearman_ci_low": 0.8,
                    "loo_spearman_ci_high": 0.95,
                    "loo_spearman_confidence": 0.95,
                    "loo_spearman_sample_count": 20,
                    "loo_spearman_pair_count": 20,
                    "loo_spearman_bootstrap_valid_replicates": 1000,
                    "loo_spearman_bootstrap_method": "test",
                })


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    project = tmp_path / "project"
    configs = project / "configs"
    analysis = project / "outputs" / "run" / "rq1"
    configs.mkdir(parents=True)
    files = {
        "population_index": analysis / "population_index.json",
        "probe": project / "outputs" / "run" / "probes" / "probe.pt",
        "states": project / "outputs" / "run" / "probes" / "states.pt",
    }
    files["probe"].parent.mkdir(parents=True)
    files["probe"].write_bytes(b"probe")
    files["states"].write_bytes(b"states")
    files["population_index"].parent.mkdir(parents=True)
    files["population_index"].write_text(json.dumps({
        "population_counts": {"all": 20, "baseline_refused": 17}
    }), encoding="utf-8")
    for population in ("all", "baseline_refused"):
        root = analysis / population
        files[f"{population}_layer_slopes"] = root / "layer_slopes.csv"
        files[f"{population}_phase_profiles"] = root / "phase_profiles.csv"
        files[f"{population}_profile_reproducibility"] = root / "profile_reproducibility.csv"
        _write_slopes(files[f"{population}_layer_slopes"], population)
        files[f"{population}_phase_profiles"].write_text("metric,layer,phase\n", encoding="utf-8")
        _write_profiles(files[f"{population}_profile_reproducibility"])
    frozen = configs / "frozen.json"
    artifacts = [
        {
            "path": Path(os.path.relpath(path, configs)).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in files.values()
    ]
    frozen.write_text(json.dumps({
        "frozen": True,
        "model": {"name": "qwen-7b", "id": "weights"},
        "expected": {"layers": 28},
        "paths": {
            "analysis_dir": Path(os.path.relpath(analysis, configs)).as_posix(),
            "probe_checkpoint": Path(os.path.relpath(files["probe"], configs)).as_posix(),
            "probe_states": Path(os.path.relpath(files["states"], configs)).as_posix(),
        },
        "frozen_artifacts": artifacts,
    }), encoding="utf-8")
    return frozen, files


def _refresh_catalog(frozen: Path, target: Path) -> None:
    raw = json.loads(frozen.read_text(encoding="utf-8"))
    resolved = target.resolve()
    for item in raw["frozen_artifacts"]:
        path = (frozen.parent / item["path"]).resolve()
        if path == resolved:
            item["sha256"] = file_sha256(target)
            break
    else:
        raise AssertionError(target)
    frozen.write_text(json.dumps(raw), encoding="utf-8")


def test_build_is_deterministic_and_reproduces_registered_layers(tmp_path):
    frozen, _ = _fixture(tmp_path)
    first = build_preregistration(frozen, date="2026-10-02")
    second = build_preregistration(frozen, date="2026-10-02")
    assert serialize_record(first) == serialize_record(second)
    assert first["selection_results"]["candidate_layers"] == [19, 24, 26, 27]
    assert first["selection_results"]["neighbor_layers"] == [18, 20, 23, 25]
    assert first["selection_results"]["depth_control_layers"] == [2, 12]
    record = tmp_path / "record.json"
    document = tmp_path / "record.md"
    write_preregistration(first, record_path=record, document_path=document)
    before = (record.read_bytes(), document.read_bytes())
    result = check_preregistration(
        frozen_config_path=frozen,
        record_path=record,
        document_path=document,
    )
    assert result["status"] == "VALID"
    assert before == (record.read_bytes(), document.read_bytes())


def test_build_rejects_frozen_sha_mismatch(tmp_path):
    frozen, files = _fixture(tmp_path)
    files["all_layer_slopes"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(RQ2PreregistrationError, match="SHA mismatch"):
        build_preregistration(frozen, date="2026-10-02")


@pytest.mark.parametrize("mutation,match", [("missing", "layer grid"), ("duplicate", "duplicate")])
def test_build_rejects_missing_or_duplicate_layer(tmp_path, mutation, match):
    frozen, files = _fixture(tmp_path)
    path = files["all_layer_slopes"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if mutation == "missing":
        rows = [
            row for row in rows
            if not (row["metric"] == "R_probe" and row["layer"] == "27")
        ]
    else:
        rows.append(next(
            dict(row) for row in rows
            if row["metric"] == "R_probe" and row["layer"] == "27"
        ))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _refresh_catalog(frozen, path)
    with pytest.raises(RQ2PreregistrationError, match=match):
        build_preregistration(frozen, date="2026-10-02")


def test_build_rejects_too_few_ci_fdr_eligible_layers(tmp_path):
    frozen, files = _fixture(tmp_path)
    path = files["baseline_refused_layer_slopes"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        if row["metric"] == "R_probe" and int(row["layer"]) >= 4:
            row["fdr_q_value"] = "0.5"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _refresh_catalog(frozen, path)
    with pytest.raises(RQ2PreregistrationError, match="eligible"):
        build_preregistration(frozen, date="2026-10-02")


def test_build_rejects_missing_population(tmp_path):
    frozen, files = _fixture(tmp_path)
    files["population_index"].write_text(json.dumps({
        "population_counts": {"all": 20}
    }), encoding="utf-8")
    _refresh_catalog(frozen, files["population_index"])
    with pytest.raises(RQ2PreregistrationError, match="population counts"):
        build_preregistration(frozen, date="2026-10-02")

def test_input_validation_rejects_a_self_consistent_but_modified_record(tmp_path):
    frozen, _ = _fixture(tmp_path)
    record = build_preregistration(frozen, date="2026-10-02")
    record["selection_results"]["candidate_layers"] = [19, 24, 26]
    record["frozen_parameters"]["pilot"]["candidate_layers"] = [19, 24, 26]

    with pytest.raises(RQ2PreregistrationError, match="stale.*frozen inputs"):
        validate_record_inputs(record, project_root=frozen.parent.parent)
