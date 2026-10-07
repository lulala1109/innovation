"""Small CPU fixtures: no GPU model, network, or research-data resampling."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rq2.artifacts import atomic_json, atomic_jsonl, canonical_sha256, file_sha256, read_jsonl, RQ2ArtifactError
from rq2.behavior import make_response_record, label_from_judge_result
from rq2.event_config import EventPilotError
from rq2.event_pipeline import EventPilotPipeline
from rq2.judge_consistency import scoring_key, scoring_consistency_audit, scoring_contract, ENDPOINTS
from rq2.judge_revision import resolve_groups, review_revision, analyze_revision, event_definition_audit, _immutable_json


FP = "a" * 64
CONTRACT = {"protocol": "fixture-v1", "success_threshold": .5, "min_convincingness": 3, "min_specificity": 3}


def record(tid, *, instruction="fixture instruction", response="fixture answer", refusal=0, convincingness=5, specificity=5):
    r = make_response_record(trial_id=tid, pair_id="dev", run_fingerprint="b" * 64,
                             harmful_text=instruction, response=response)
    label = label_from_judge_result(tid, r["response_sha256"],
        {"refusal": refusal, "convincingness": convincingness, "specificity": specificity},
        judge_fingerprint=FP).to_record()
    return r, label


def resolve(oracle, sources=(), mappings=()):
    return resolve_groups([x[0] for x in oracle], [x[1] for x in oracle],
        [x[0] for x in sources], [x[1] for x in sources], mappings,
        contract=CONTRACT, legacy_fingerprint=FP, judge={})


def trial(r, intervention, baseline=None, dose=1.):
    return {"trial_id": r["trial_id"], "pair_id": r["pair_id"], "run_fingerprint": r["run_fingerprint"],
            "response_sha256": r["response_sha256"], "intervention": intervention,
            "baseline_trial_id": baseline, "dose": dose}


class ContentIdentityTests(unittest.TestCase):
    def test_identity_binds_instruction_reply_protocol_not_trial_id(self):
        a, b = record("a")[0], record("b")[0]
        self.assertEqual(scoring_key(a, FP), scoring_key(b, FP))
        for other in (record("c", instruction="different instruction")[0],
                      record("d", response="fixture answer ")[0]):
            self.assertNotEqual(scoring_key(a, FP), scoring_key(other, FP))
        self.assertNotEqual(scoring_key(a, FP), scoring_key(a, "c" * 64))

    def test_contract_includes_endpoint_and_rejects_secret_urls(self):
        judge = {"provider": "deepseek", "model": "fixture", "base_url": "https://example.invalid/v1", "protocol": "fixture"}
        first = scoring_contract(judge)
        self.assertNotEqual(first, scoring_contract({**judge, "base_url": "https://other.invalid/v1"}))
        self.assertIn("prompt_sha256", first)
        self.assertNotIn("base_url", first)
        with self.assertRaises(RQ2ArtifactError):
            scoring_contract({**judge, "base_url": "https://example.invalid?api_key=fixture"})

    def test_anchor_reuse_applies_to_full_state_and_noops_without_input_mutation(self):
        source = record("src")
        baseline = record("base")
        oracle = [baseline, record("sham", refusal=1), record("full", refusal=1), record("zero", convincingness=4)]
        before = copy.deepcopy((oracle, source))
        result = resolve(oracle, [source], [{"new_trial_id": "base", "source_trial_id": "src"}])
        self.assertEqual((oracle, source), before)
        self.assertEqual(len(result["cache"]), 1)
        self.assertEqual(result["pending"], [])
        self.assertEqual({l["strongreject_score"] for l in result["labels"]}, {1.})
        trials = [trial(baseline[0], "baseline"), trial(oracle[1][0], "sham", "base"),
                  trial(oracle[2][0], "full_state", "base"), trial(oracle[3][0], "r_direction", "base", 0.)]
        original = scoring_consistency_audit(trials, [x[0] for x in oracle], [x[1] for x in oracle])
        revised = scoring_consistency_audit(trials, [x[0] for x in oracle], result["labels"])
        self.assertFalse(original["passed"])
        self.assertTrue(revised["passed"])
        self.assertEqual(revised["identical_to_baseline"], 3)

    def test_other_source_consensus_has_predeclared_priority(self):
        result = resolve([record("a", refusal=0)], [record("source", refusal=1)])
        self.assertEqual(result["bindings"][0]["origin"], "source_consensus")
        self.assertTrue(result["labels"][0]["refusal_label"])
        self.assertTrue(result["conflicts"][0]["authority_disagrees_with_oracle"])

    def test_endpoint_equal_rubric_conflict_is_unresolved(self):
        result = resolve([record("a", refusal=1), record("b", refusal=1, convincingness=1)])
        self.assertEqual(len(result["pending"]), 1)
        self.assertFalse(result["pending"][0]["endpoint_conflict"])
        self.assertTrue(result["pending"][0]["rubric_conflict"])
        self.assertTrue(all(l["strongreject_score"] is None for l in result["labels"]))

    def test_source_conflict_does_not_fall_back_to_favorable_oracle(self):
        result = resolve([record("a", refusal=1)], [record("s1"), record("s2", refusal=1)])
        self.assertEqual(len(result["pending"]), 1)
        self.assertEqual(result["labels"][0]["label_status"], "unknown")

    def test_conflicting_frozen_anchors_cannot_be_adjudicated_in_place(self):
        result = resolve([record("b1"), record("b2", refusal=1)],
            [record("s1"), record("s2", refusal=1)],
            [{"new_trial_id":"b1","source_trial_id":"s1"},{"new_trial_id":"b2","source_trial_id":"s2"}])
        self.assertFalse(result["pending"][0]["review_allowed"])
        self.assertEqual(result["pending"][0]["reason"], "anchor_conflict_blocked")

    def test_audit_rejects_missing_and_invalid_labels(self):
        r, l = record("base")
        t = trial(r, "baseline")
        self.assertFalse(scoring_consistency_audit([t], [r], [])["passed"])
        broken = {**l, "strongreject_score": 0.}
        self.assertFalse(scoring_consistency_audit([t], [r], [broken])["passed"])


class RevisionControlTests(unittest.TestCase):
    def test_original_positive_flag_cannot_bypass_new_gate_or_call_gpu(self):
        pipeline = object.__new__(EventPilotPipeline)
        pipeline.stage_path = lambda _: Path("/fixture/analysis.json")
        pipeline._scoring_audit = Mock(return_value={"passed": False})
        pipeline._run_generation_phase = Mock()
        with self.assertRaises(EventPilotError):
            pipeline._run_mechanism_generate()
        pipeline._run_generation_phase.assert_not_called()
        with self.assertRaises(EventPilotError):
            pipeline._run_judge_phase("oracle_pilot")

    def test_api_requires_explicit_opt_in_and_frozen_artifact_cannot_change(self):
        evaluator = Mock()
        with self.assertRaises(RQ2ArtifactError):
            review_revision("/not_loaded", evaluator_factory=evaluator)
        evaluator.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            _immutable_json(path, {"version": 1})
            digest = file_sha256(path)
            _immutable_json(path, {"version": 1})
            with self.assertRaises(RQ2ArtifactError):
                _immutable_json(path, {"version": 2})
            self.assertEqual(file_sha256(path), digest)

    def test_review_once_per_key_and_no_retry_after_failure_or_resume(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / "config.json"
                atomic_json(config, {"fixture": True})
                response, _ = record("a")
                key = scoring_key(response, canonical_sha256(CONTRACT))
                atomic_jsonl(root / "private/review_inputs.jsonl", [{"scoring_key": key,
                    "harmful_text": response["harmful_text"], "response": response["response"]}])
                evaluate = Mock(side_effect=TimeoutError("fixture") if fail else None,
                                return_value={"refusal": 1, "convincingness": 1, "specificity": 1})
                evaluator = SimpleNamespace(evaluate=evaluate)
                manifest = {"scoring_contract": CONTRACT}
                with patch("rq2.judge_revision._load", return_value=(manifest, root, SimpleNamespace())), \
                        patch("rq2.judge_revision.audit_revision", return_value={}):
                    first = review_revision(config, allow_api=True, evaluator_factory=lambda: evaluator)
                    second = review_revision(config, allow_api=True, evaluator_factory=Mock(side_effect=AssertionError("resume called API")))
                self.assertEqual(evaluate.call_count, 1)
                self.assertEqual(first["completed"], 0 if fail else 1)
                self.assertEqual(first, second)

    def test_revised_pre_event_refusal_cannot_silently_move_event(self):
        r, old = record("pre", refusal=1)
        nr, changed = record("new", refusal=0)
        source_trial = {**trial(r, "trajectory_baseline"), "state_key": "scan:2", "step": 2}
        mapping = [{"pair_id":"dev", "source_state_key":"scan:7"}]
        audit = event_definition_audit([source_trial], [r], [old], [nr], [changed], mapping, FP)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["checked_overlapping_event_predicates"], 1)
        later = {**source_trial, "state_key": "scan:9", "step": 9}
        self.assertTrue(event_definition_audit([later], [r], [old], [nr], [changed], mapping, FP)["passed"])

    def test_unresolved_blocks_analysis_instead_of_dropping_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r, _ = record("a")
            migration = resolve([record("a"), record("b", refusal=1)])
            manifest = {"scoring_contract": CONTRACT}
            data = (None, [], [r, record("b")[0]], [], [], [], [])
            with patch("rq2.judge_revision._load", return_value=(manifest, root, None)), \
                    patch("rq2.judge_revision.audit_revision", return_value={}), \
                    patch("rq2.judge_revision._migration", return_value=(migration, data)):
                result = analyze_revision(root / "config.json")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["unresolved_unique_inputs"], 1)
            self.assertFalse((root / "analysis").exists())


if __name__ == "__main__":
    unittest.main()
