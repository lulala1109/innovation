"""Tests for the frozen, trajectory-pending T0.7 RQ2 manifest."""

import hashlib
import json
from pathlib import Path

import pytest

from data.build_rq2_manifest import (
    ATTACK_TARGET,
    OUTPUT_FIELDS,
    RQ2ManifestError,
    check_rq2_manifest,
    derive_rq2_manifest,
    write_rq2_manifest,
)
from data.build_rq2_split import write_rq2_split
from experiments.batch_safety_attack import (
    _audio_path_for_row,
    _read_records,
    _target_for_row,
    _validate_unique_cases,
)
from rq2.artifacts import atomic_json, atomic_jsonl, file_sha256, read_jsonl


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _refresh_split_output(split_audit: Path, role: str, path: Path) -> None:
    payload = json.loads(split_audit.read_text(encoding="utf-8"))
    rows = read_jsonl(path)
    payload["counts"][role] = len(rows)
    payload["outputs"][role]["rows"] = len(rows)
    payload["outputs"][role]["sha256"] = file_sha256(path)
    atomic_json(split_audit, payload)


def _refresh_rq1_source(split_audit: Path, rq1: Path) -> None:
    payload = json.loads(split_audit.read_text(encoding="utf-8"))
    payload["inputs"]["rq1_manifests"] = [
        {"path": str(rq1.resolve()), "sha256": file_sha256(rq1)}
    ]
    atomic_json(split_audit, payload)


def _fixture(tmp_path: Path) -> dict:
    pool = tmp_path / "conservative_candidate_pool.jsonl"
    review_audit = tmp_path / "summary.json"
    rows = []
    for index in range(8):
        audio = tmp_path / f"candidate_{index}.wav"
        audio.write_bytes(f"audio-{index}".encode("utf-8"))
        rows.append(
            {
                "pair_id": f"advbench_{index:04d}",
                "source_row_index": index,
                "harmful_text": f"candidate harmful text {index}",
                "content_group": _sha(f"content-{index}"),
                "goal_sha256": _sha(f"goal-{index}"),
                "clean_audio_path": str(audio),
                "clean_audio_sha256": file_sha256(audio),
                "clean_refused": True,
                "candidate_pool_status": "unassigned",
                "semantic_independence_review_status": (
                    "not_performed_conservative_exclusion"
                ),
                "audio_content_fidelity_review_status": "not_performed",
                "target_text": ATTACK_TARGET,
            }
        )
    atomic_jsonl(pool, rows)
    atomic_json(
        review_audit,
        {
            "format": "rq2-conservative-candidate-pool",
            "version": 1,
            "pool_is_unassigned": True,
            "formal_role_assignment_ready": False,
            "counts": {"candidate_pool": len(rows)},
            "outputs": {
                "conservative_candidate_pool": {
                    "rows": len(rows),
                    "sha256": file_sha256(pool),
                }
            },
            "protocol_deviation": {
                "manual_semantic_review_performed": False,
                "audio_content_fidelity_review_performed": False,
                "claims_not_allowed": [
                    "manual semantic independence confirmed",
                    "audio content fidelity confirmed",
                ],
            },
        },
    )
    rq1_audio = tmp_path / "rq1.wav"
    rq1_audio.write_bytes(b"rq1-audio")
    rq1 = tmp_path / "rq1.jsonl"
    atomic_jsonl(
        rq1,
        [
            {
                "pair_id": "jbb_000",
                "harmful_text": "different rq1 content",
                "content_group": "rq1-content",
                "harmful_audio_path": str(rq1_audio),
            }
        ],
    )
    split_dir = tmp_path / "split"
    write_rq2_split(
        candidate_pool_path=pool,
        candidate_summary_path=review_audit,
        rq1_manifest_paths=[rq1],
        output_dir=split_dir,
        decision_date="2026-10-02",
        seed=42,
        dev_count=2,
        causal_test_count=3,
        expected_candidate_count=8,
        minimum_causal_test_count=1,
        stratum="advbench",
    )
    return {
        "dev_path": split_dir / "rq2_dev.jsonl",
        "causal_test_path": split_dir / "rq2_causal_test.jsonl",
        "reserve_path": split_dir / "reserve.jsonl",
        "split_audit_path": split_dir / "split_audit.json",
        "review_audit_path": review_audit,
        "rq1_manifest_paths": [rq1],
        "output_dir": tmp_path / "manifest",
        "decision_date": "2026-10-02",
        "expected_dev_count": 2,
        "expected_causal_test_count": 3,
        "expected_reserve_count": 3,
        "minimum_causal_test_count": 1,
        "expected_split_audit_sha256": None,
        "expected_review_audit_sha256": None,
    }


def _derive_kwargs(kwargs: dict) -> dict:
    return {key: value for key, value in kwargs.items() if key != "output_dir"}


def test_build_check_and_batch_contract_are_deterministic(tmp_path):
    kwargs = _fixture(tmp_path)
    source_paths = [
        Path(kwargs["dev_path"]),
        Path(kwargs["causal_test_path"]),
        Path(kwargs["reserve_path"]),
        Path(kwargs["split_audit_path"]),
    ]
    sources_before = {path: path.read_bytes() for path in source_paths}
    built = write_rq2_manifest(**kwargs)
    output = Path(kwargs["output_dir"])
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    rebuilt = write_rq2_manifest(**kwargs)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    checked = check_rq2_manifest(**kwargs)
    third = {path.name: path.read_bytes() for path in output.iterdir()}

    assert built["counts"] == {
        "rq2_dev": 2,
        "rq2_causal_test": 3,
        "total": 5,
    }
    assert built == rebuilt
    assert checked["status"] == "VALID"
    assert first == second == third
    assert {path: path.read_bytes() for path in source_paths} == sources_before

    manifest = output / "rq2_manifest.jsonl"
    rows = read_jsonl(manifest)
    assert len(rows) == 5
    assert [row["rq2_role"] for row in rows] == [
        "rq2_dev",
        "rq2_dev",
        "rq2_causal_test",
        "rq2_causal_test",
        "rq2_causal_test",
    ]
    assert all(set(row) == set(OUTPUT_FIELDS) for row in rows)
    assert all(row["target_text"] == ATTACK_TARGET for row in rows)
    assert all("trajectory_path" not in row for row in rows)
    assert all(row["case_id"] == row["pair_id"] for row in rows)

    batch_rows = _read_records(manifest)
    assert len(_validate_unique_cases(batch_rows)) == 5
    for index, row in enumerate(batch_rows):
        assert Path(_audio_path_for_row(row, row_number=index)).is_file()
        assert _target_for_row(row, None, row_number=index) == ATTACK_TARGET

    audit = json.loads((output / "manifest_audit.json").read_text(encoding="utf-8"))
    assert audit["attack_target"]["value"] == ATTACK_TARGET
    assert audit["attack_target"]["advbench_original_target_included"] is False
    assert audit["trajectory_contract"]["initial_manifest_has_trajectory_path"] is False
    assert audit["inputs"]["reserve"]["included_in_manifest"] is False
    assert all(report["passed"] for report in audit["identity_isolation"].values())


def test_rejects_frozen_split_audit_sha_mismatch(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["expected_split_audit_sha256"] = "0" * 64
    with pytest.raises(RQ2ManifestError, match="split audit SHA-256 mismatch"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_frozen_review_audit_sha_mismatch(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["expected_review_audit_sha256"] = "0" * 64
    with pytest.raises(RQ2ManifestError, match="review audit SHA-256 mismatch"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_attack_target_drift(tmp_path):
    kwargs = _fixture(tmp_path)
    dev = Path(kwargs["dev_path"])
    rows = read_jsonl(dev)
    rows[0]["target_text"] = "different target"
    atomic_jsonl(dev, rows)
    _refresh_split_output(Path(kwargs["split_audit_path"]), "rq2_dev", dev)
    with pytest.raises(RQ2ManifestError, match="attack target drift"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_advbench_original_target_field(tmp_path):
    kwargs = _fixture(tmp_path)
    dev = Path(kwargs["dev_path"])
    rows = read_jsonl(dev)
    rows[0]["target"] = "original dataset completion"
    atomic_jsonl(dev, rows)
    _refresh_split_output(Path(kwargs["split_audit_path"]), "rq2_dev", dev)
    with pytest.raises(RQ2ManifestError, match="forbidden source fields"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_reserve_pair_leak(tmp_path):
    kwargs = _fixture(tmp_path)
    dev = Path(kwargs["dev_path"])
    reserve = Path(kwargs["reserve_path"])
    dev_rows = read_jsonl(dev)
    leaked = dict(read_jsonl(reserve)[0])
    leaked.pop("split_role")
    leaked.pop("reason")
    leaked["rq2_role"] = "rq2_dev"
    leaked["candidate_pool_status"] = "role_assigned"
    dev_rows[0] = leaked
    atomic_jsonl(dev, dev_rows)
    _refresh_split_output(Path(kwargs["split_audit_path"]), "rq2_dev", dev)
    with pytest.raises(RQ2ManifestError, match="reserve pair leaked"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_clean_audio_sha_mismatch(tmp_path):
    kwargs = _fixture(tmp_path)
    row = read_jsonl(Path(kwargs["dev_path"]))[0]
    Path(row["clean_audio_path"]).write_bytes(b"modified-audio")
    with pytest.raises(RQ2ManifestError, match="clean-audio SHA-256 mismatch"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


@pytest.mark.parametrize("overlap_kind", ["pair", "content", "audio"])
def test_rejects_rq1_identity_overlap(tmp_path, overlap_kind):
    kwargs = _fixture(tmp_path)
    selected = read_jsonl(Path(kwargs["causal_test_path"]))[0]
    rq1 = Path(kwargs["rq1_manifest_paths"][0])
    rq1_audio = tmp_path / "independent_rq1.wav"
    rq1_audio.write_bytes(b"independent-rq1-audio")
    row = {
        "pair_id": "jbb_independent",
        "harmful_text": "independent rq1 text",
        "content_group": "independent-rq1-content",
        "harmful_audio_path": str(rq1_audio),
    }
    if overlap_kind == "pair":
        row["pair_id"] = selected["pair_id"]
    elif overlap_kind == "content":
        row["harmful_text"] = selected["harmful_text"]
    else:
        row["harmful_audio_path"] = selected["clean_audio_path"]
    atomic_jsonl(rq1, [row])
    _refresh_rq1_source(Path(kwargs["split_audit_path"]), rq1)
    with pytest.raises(RQ2ManifestError, match="identity isolation failed"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_rejects_duplicate_assigned_pair(tmp_path):
    kwargs = _fixture(tmp_path)
    dev = Path(kwargs["dev_path"])
    rows = read_jsonl(dev)
    rows[1] = dict(rows[0])
    atomic_jsonl(dev, rows)
    _refresh_split_output(Path(kwargs["split_audit_path"]), "rq2_dev", dev)
    with pytest.raises(RQ2ManifestError, match="split ranks|duplicate pair_id"):
        derive_rq2_manifest(**_derive_kwargs(kwargs))


def test_check_rejects_modified_manifest_artifact(tmp_path):
    kwargs = _fixture(tmp_path)
    write_rq2_manifest(**kwargs)
    manifest = Path(kwargs["output_dir"]) / "rq2_manifest.jsonl"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(RQ2ManifestError, match="stale, or modified"):
        check_rq2_manifest(**kwargs)
