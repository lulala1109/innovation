import hashlib

import pytest

from rq2.artifacts import (
    RQ2ArtifactError,
    TrialKey,
    append_jsonl_fsync,
    atomic_csv,
    read_jsonl,
    validate_trial_record,
)


def test_trial_id_binds_condition():
    base = TrialKey("run", "pair", "fixed:2", 4, "r_direction", 1.0)
    changed = TrialKey("run", "pair", "fixed:2", 5, "r_direction", 1.0)
    assert base.trial_id != changed.trial_id


def test_analysis_record_rejects_nested_response_text():
    key = TrialKey("run", "pair", "fixed:2", 4, "sham", 1.0)
    digest = hashlib.sha256(b"x").hexdigest()
    record = {
        "format": "rq2-causal-intervention-trial", "version": 1,
        **key.to_dict(), "rq2_role": "rq2_dev", "step": 2,
        "baseline_trial_id": None, "audit": {},
        "diagnostic": {"nested": {"response": "private"}},
        "response_sha256": digest,
    }
    with pytest.raises(RQ2ArtifactError, match="private field"):
        validate_trial_record(record)


def test_empty_csv_has_explicit_schema(tmp_path):
    output = atomic_csv(tmp_path / "empty.csv", [], fieldnames=("layer", "effect"))
    assert output.read_text(encoding="utf-8") == "layer,effect\n"


def test_truncated_jsonl_recovery(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"trial_id":"ok"}\n{"trial_id":', encoding="utf-8")
    assert read_jsonl(path, recover_truncated=True) == [{"trial_id": "ok"}]


def test_append_journal_discards_truncated_tail_before_new_commit(tmp_path):
    path = tmp_path / "commits.jsonl"
    path.write_text('{"trial_id":"old"}\n{"trial_id":', encoding="utf-8")
    append_jsonl_fsync(path, {"trial_id": "new"})
    assert read_jsonl(path) == [
        {"trial_id": "old"},
        {"trial_id": "new"},
    ]
