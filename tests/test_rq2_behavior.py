import hashlib

import pytest

pytest.importorskip("pydantic")

from rq2.artifacts import RQ2ArtifactError, atomic_jsonl, read_jsonl
from rq2.behavior import (
    judge_response_sidecar,
    label_from_judge_result,
    make_response_record,
    unknown_label,
)


def test_canonical_balanced_label_and_identity():
    response = make_response_record(
        trial_id="trial", harmful_text="target", response="answer",
        pair_id="pair", run_fingerprint="run",
    )
    judge = hashlib.sha256(b"judge").hexdigest()
    label = label_from_judge_result(
        "trial", response["response_sha256"],
        {"refusal": 0, "convincingness": 5, "specificity": 5, "score": 1.0},
        judge_fingerprint=judge,
    )
    assert label.compliance_label is True
    assert label.refusal_orientation == pytest.approx(0.0)


def test_unknown_remains_unknown():
    digest = hashlib.sha256(b"response").hexdigest()
    judge = hashlib.sha256(b"judge").hexdigest()
    label = unknown_label(
        "trial", digest, judge_fingerprint=judge,
        error_type="TimeoutError", retryable=True,
    )
    assert label.label_status == "unknown"
    assert label.refusal_label is None


class _FlakyEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, target_text, response):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary")
        return {"refusal": 0, "convincingness": 5, "specificity": 5, "score": 1.0}


def test_unknown_label_is_retried_without_leaking_text(tmp_path):
    responses = tmp_path / "responses.jsonl"
    labels = tmp_path / "labels.jsonl"
    atomic_jsonl(responses, [make_response_record(
        trial_id="trial", harmful_text="target", response="answer",
        pair_id="pair", run_fingerprint="run",
    )])
    evaluator = _FlakyEvaluator()
    config = {"protocol": "strongreject-canonical-balanced", "model": "judge"}
    first = judge_response_sidecar(
        responses, labels, evaluator=evaluator, judge_config=config
    )
    assert first["unknown"] == 1
    assert read_jsonl(labels)[0]["label_status"] == "unknown"
    second = judge_response_sidecar(
        responses, labels, evaluator=evaluator, judge_config=config
    )
    row = read_jsonl(labels)[0]
    assert second["judged"] == 1
    assert row["label_status"] == "ok"
    assert "response" not in row and "reasoning" not in row


def test_changed_response_rejects_stale_label(tmp_path):
    responses = tmp_path / "responses.jsonl"
    labels = tmp_path / "labels.jsonl"
    evaluator = _FlakyEvaluator()
    evaluator.calls = 1
    config = {"protocol": "strongreject-canonical-balanced", "model": "judge"}
    original = make_response_record(
        trial_id="trial", harmful_text="target", response="answer",
        pair_id="pair", run_fingerprint="run",
    )
    atomic_jsonl(responses, [original])
    judge_response_sidecar(responses, labels, evaluator=evaluator, judge_config=config)
    changed = make_response_record(
        trial_id="trial", harmful_text="target", response="changed",
        pair_id="pair", run_fingerprint="run",
    )
    atomic_jsonl(responses, [changed])
    with pytest.raises(RQ2ArtifactError, match="stale label"):
        judge_response_sidecar(responses, labels, evaluator=evaluator, judge_config=config)
