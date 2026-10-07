"""Tests for deterministic, outcome-blind RQ2 dataset splitting."""

import hashlib
import json
from pathlib import Path

import pytest

from data.build_rq2_split import (
    RESERVE_REASON,
    RQ2SplitError,
    SPLIT_ALGORITHM,
    check_rq2_split,
    derive_rq2_split,
    write_rq2_split,
)
from rq2.artifacts import atomic_json, atomic_jsonl, file_sha256, read_jsonl


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _refresh_summary(pool: Path, summary: Path) -> None:
    rows = read_jsonl(pool)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["counts"]["candidate_pool"] = len(rows)
    payload["outputs"]["conservative_candidate_pool"]["rows"] = len(rows)
    payload["outputs"]["conservative_candidate_pool"]["sha256"] = file_sha256(pool)
    atomic_json(summary, payload)


def _fixture(tmp_path: Path) -> dict:
    pool = tmp_path / "conservative_candidate_pool.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "split"
    rows = []
    for index in range(8):
        audio = tmp_path / f"candidate_{index}.wav"
        audio.write_bytes(f"audio-{index}".encode("utf-8"))
        rows.append(
            {
                "pair_id": f"advbench_{index:04d}",
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
                "target_text": "Sure, here is",
            }
        )
    atomic_jsonl(pool, rows)
    atomic_json(
        summary,
        {
            "format": "rq2-conservative-candidate-pool",
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
                "claims_not_allowed": ["manual semantic independence confirmed"]
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
    return {
        "candidate_pool_path": pool,
        "candidate_summary_path": summary,
        "rq1_manifest_paths": [rq1],
        "output_dir": output,
        "decision_date": "2026-10-02",
        "seed": 42,
        "dev_count": 2,
        "causal_test_count": 3,
        "expected_candidate_count": 8,
        "minimum_causal_test_count": 1,
        "stratum": "advbench",
    }


def _derive_kwargs(kwargs: dict) -> dict:
    return {key: value for key, value in kwargs.items() if key != "output_dir"}


def test_build_and_check_are_deterministic_and_complete(tmp_path):
    kwargs = _fixture(tmp_path)
    source_before = Path(kwargs["candidate_pool_path"]).read_bytes()
    built = write_rq2_split(**kwargs)
    output = Path(kwargs["output_dir"])
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    rebuilt = write_rq2_split(**kwargs)
    second = {path.name: path.read_bytes() for path in output.iterdir()}
    checked = check_rq2_split(**kwargs)
    third = {path.name: path.read_bytes() for path in output.iterdir()}

    assert built["counts"] == {
        "candidate_pool": 8,
        "rq2_dev": 2,
        "rq2_causal_test": 3,
        "reserve": 3,
    }
    assert rebuilt == built
    assert checked["status"] == "VALID"
    assert first == second == third
    assert Path(kwargs["candidate_pool_path"]).read_bytes() == source_before

    assignments = read_jsonl(output / "split_assignments.jsonl")
    dev = read_jsonl(output / "rq2_dev.jsonl")
    causal = read_jsonl(output / "rq2_causal_test.jsonl")
    reserve = read_jsonl(output / "reserve.jsonl")
    assert [row["split_rank"] for row in assignments] == list(range(1, 9))
    assert [row["split_role"] for row in assignments] == (
        ["rq2_dev"] * 2 + ["rq2_causal_test"] * 3 + ["reserve"] * 3
    )
    assert all(row["rq2_role"] == "rq2_dev" for row in dev)
    assert all(row["rq2_role"] == "rq2_causal_test" for row in causal)
    assert all("rq2_role" not in row for row in reserve)
    assert all(row["reason"] == RESERVE_REASON for row in reserve)
    audit = json.loads((output / "split_audit.json").read_text(encoding="utf-8"))
    assert audit["algorithm"] == SPLIT_ALGORITHM
    assert audit["seed"] == 42
    assert audit["roles_locked"] is True
    assert all(report["passed"] for report in audit["identity_isolation"].values())


def test_rejects_candidate_pool_sha_mismatch(tmp_path):
    kwargs = _fixture(tmp_path)
    summary = Path(kwargs["candidate_summary_path"])
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["outputs"]["conservative_candidate_pool"]["sha256"] = "0" * 64
    atomic_json(summary, payload)
    with pytest.raises(RQ2SplitError, match="SHA-256 does not match"):
        derive_rq2_split(**_derive_kwargs(kwargs))


@pytest.mark.parametrize("duplicate_field", ["content_group", "clean_audio"])
def test_rejects_duplicate_candidate_identity(tmp_path, duplicate_field):
    kwargs = _fixture(tmp_path)
    pool = Path(kwargs["candidate_pool_path"])
    rows = read_jsonl(pool)
    if duplicate_field == "content_group":
        rows[1]["content_group"] = rows[0]["content_group"]
        expected = "duplicate candidate harmful content"
    else:
        rows[1]["clean_audio_path"] = rows[0]["clean_audio_path"]
        rows[1]["clean_audio_sha256"] = rows[0]["clean_audio_sha256"]
        expected = "duplicate candidate clean audio"
    atomic_jsonl(pool, rows)
    _refresh_summary(pool, Path(kwargs["candidate_summary_path"]))
    with pytest.raises(RQ2SplitError, match=expected):
        derive_rq2_split(**_derive_kwargs(kwargs))


@pytest.mark.parametrize("overlap_kind", ["pair", "content", "audio"])
def test_rejects_causal_test_overlap_with_rq1(tmp_path, overlap_kind):
    kwargs = _fixture(tmp_path)
    _, _, causal, _, _ = derive_rq2_split(**_derive_kwargs(kwargs))
    selected = causal[0]
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
    with pytest.raises(RQ2SplitError, match="identity isolation failed"):
        derive_rq2_split(**_derive_kwargs(kwargs))


def test_rejects_formal_count_below_minimum(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["minimum_causal_test_count"] = 4
    with pytest.raises(RQ2SplitError, match="below the frozen minimum"):
        derive_rq2_split(**_derive_kwargs(kwargs))


def test_check_rejects_modified_split_artifact(tmp_path):
    kwargs = _fixture(tmp_path)
    write_rq2_split(**kwargs)
    dev = Path(kwargs["output_dir"]) / "rq2_dev.jsonl"
    dev.write_text(dev.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RQ2SplitError, match="stale, or modified"):
        check_rq2_split(**kwargs)
