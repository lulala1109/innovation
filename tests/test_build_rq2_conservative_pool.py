"""Tests for the conservative RQ2 candidate-pool derivation."""

import hashlib
import json
from pathlib import Path

import pytest

from data.build_rq2_conservative_pool import (
    ConservativePoolError,
    EXCLUSION_REASON,
    POLICY,
    check_conservative_pool,
    derive_conservative_pool,
    write_conservative_pool,
)
from rq2.artifacts import atomic_json, atomic_jsonl, read_jsonl


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture(tmp_path: Path):
    eligible = tmp_path / "eligible_pool.jsonl"
    review = tmp_path / "rq1_advbench_source_review.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "derived"
    candidates = [
        {
            "pair_id": f"advbench_{index:04d}",
            "source_row_index": index,
            "goal_sha256": _sha(f"goal-{index}"),
            "clean_audio_sha256": _sha(f"audio-{index}"),
            "clean_refused": True,
            "semantic_independence_review_status": "pending",
            "harmful_text": f"candidate {index}",
        }
        for index in range(4)
    ]
    reviews = [
        {
            "harmful_source": "AdvBench",
            "review_status": "pending",
            "rq1_pair_id": "jbb_001",
            "top_matches": [
                {"pair_id": "advbench_0001", "similarity": 0.9},
                {"pair_id": "already_excluded", "similarity": 0.8},
                {"pair_id": "advbench_0002", "similarity": 0.7},
            ],
        },
        {
            "harmful_source": "AdvBench",
            "review_status": "pending",
            "rq1_pair_id": "jbb_002",
            "top_matches": [
                {"pair_id": "advbench_0001", "similarity": 0.6},
                {"pair_id": "outside_pool_a", "similarity": 0.5},
                {"pair_id": "outside_pool_b", "similarity": 0.4},
            ],
        },
    ]
    atomic_jsonl(eligible, candidates)
    atomic_jsonl(review, reviews)
    atomic_json(summary, {
        "format": "advbench-rq2-clean-screening",
        "screening_complete": True,
        "eligible_count": 4,
        "rq1_advbench_source_review_rows": 2,
        "pool_is_unassigned": True,
        "formal_role_assignment_ready": False,
        "finalize_fingerprint": _sha("finalize"),
    })
    return eligible, review, summary, output


def _kwargs(tmp_path: Path):
    eligible, review, summary, output = _fixture(tmp_path)
    return {
        "eligible_pool_path": eligible,
        "rq1_review_path": review,
        "screening_summary_path": summary,
        "output_dir": output,
        "decision_date": "2026-10-02",
        "expected_input_count": 4,
        "expected_exclusion_count": 2,
        "expected_output_count": 2,
    }


def test_build_and_check_are_deterministic_and_preserve_original_pool(tmp_path):
    kwargs = _kwargs(tmp_path)
    source_before = Path(kwargs["eligible_pool_path"]).read_bytes()
    result = write_conservative_pool(**kwargs)
    assert result["candidate_pool_rows"] == 2
    assert result["excluded_rows"] == 2
    output = Path(kwargs["output_dir"])
    pool = read_jsonl(output / "conservative_candidate_pool.jsonl")
    exclusions = read_jsonl(output / "conservative_exclusions.jsonl")
    assert [row["pair_id"] for row in pool] == ["advbench_0000", "advbench_0003"]
    assert all(row["candidate_pool_policy"] == POLICY for row in pool)
    assert all(
        row["semantic_independence_review_status"]
        == "not_performed_conservative_exclusion"
        for row in pool
    )
    assert [row["pair_id"] for row in exclusions] == [
        "advbench_0001", "advbench_0002"
    ]
    assert all(row["reason"] == EXCLUSION_REASON for row in exclusions)
    assert len(exclusions[0]["rq1_matches"]) == 2
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    checked = check_conservative_pool(**kwargs)
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert checked["status"] == "VALID"
    assert before == after
    assert Path(kwargs["eligible_pool_path"]).read_bytes() == source_before


def test_builder_rejects_duplicate_eligible_pair_id(tmp_path):
    kwargs = _kwargs(tmp_path)
    eligible = Path(kwargs["eligible_pool_path"])
    rows = read_jsonl(eligible)
    rows.append(dict(rows[0]))
    atomic_jsonl(eligible, rows)
    with pytest.raises(ConservativePoolError, match="duplicate pair_id"):
        derive_conservative_pool(**{
            key: value for key, value in kwargs.items() if key != "output_dir"
        })


def test_builder_rejects_repeated_top_match_within_rq1_row(tmp_path):
    kwargs = _kwargs(tmp_path)
    review = Path(kwargs["rq1_review_path"])
    rows = read_jsonl(review)
    rows[0]["top_matches"][1] = dict(rows[0]["top_matches"][0])
    atomic_jsonl(review, rows)
    with pytest.raises(ConservativePoolError, match="repeats matched pair"):
        derive_conservative_pool(**{
            key: value for key, value in kwargs.items() if key != "output_dir"
        })


def test_check_rejects_modified_derived_artifact(tmp_path):
    kwargs = _kwargs(tmp_path)
    write_conservative_pool(**kwargs)
    output = Path(kwargs["output_dir"])
    pool_path = output / "conservative_candidate_pool.jsonl"
    pool_path.write_text(pool_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ConservativePoolError, match="stale, or modified"):
        check_conservative_pool(**kwargs)
