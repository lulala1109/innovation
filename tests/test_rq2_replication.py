"""Targeted CPU tests. No network, CUDA, model weights, or live experiment."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rq2.artifacts import atomic_json, canonical_sha256, records_by_id
from rq2.behavior import make_response_record
from rq2.event_config import EVENT_CENTER
from rq2.replication_config import BUDGET, STAGES
from rq2.replication_judge import ContentJudge, ReplicationStop
from rq2.replication_pipeline import ReplicationPipeline
from rq2.pipeline import RQ2Pipeline
from tests.test_rq2_event_pilot import population_fixture


JUDGE = {"provider": "deepseek", "model": "fixture", "base_url": "https://fixture.invalid/v1"}
CONTRACT = {"protocol": "cpu-test"}


def response(tid="one", instruction="fixture instruction", reply="fixture reply"):
    return make_response_record(trial_id=tid, harmful_text=instruction, response=reply,
                                pair_id="p", run_fingerprint="a"*64)


def completion(content=None):
    return {"model": "fixture-version", "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "choices": [{"message": {"content": content if content is not None else json.dumps({
                "overall_reasoning": "fixture", "refusal": 1, "convincingness": 1, "specificity": 1})}}]}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = Mock(return_value=completion())

    def judge(self, **kwargs):
        return ContentJudge(self.root, JUDGE, CONTRACT, kwargs.pop("budget", BUDGET),
                            transport=kwargs.pop("transport", self.transport))

    def test_same_content_across_trial_ids_and_resume_pays_once(self):
        first = self.judge().label(response())
        resumed = self.judge().label(response("other"))
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(first["strongreject_score"], resumed["strongreject_score"])
        self.assertEqual(resumed["trial_id"], "other")
        self.assertAlmostEqual(self.judge().summary()["accounted_cny_at_peak_rates"], .00038)

    def test_instruction_part_of_key(self):
        judge = self.judge()
        judge.label(response())
        judge.label(response("two", instruction="different instruction"))
        self.assertEqual(self.transport.call_count, 2)

    def test_budget_stops_before_sending(self):
        judge = self.judge(budget={**BUDGET, "api_limit_cny": .001})
        with self.assertRaises(ReplicationStop):
            judge.label(response())
        self.transport.assert_not_called()
        self.assertEqual(judge.summary()["http_attempts"], 0)

    def test_uncertain_request_remains_reserved_across_restart(self):
        self.transport.side_effect = TimeoutError("fixture")
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.judge().summary()["unresolved_attempts"], 1)
        self.transport.side_effect = None
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("two", reply="new reply"))
        self.assertEqual(self.transport.call_count, 1)

    def test_missing_usage_never_treated_as_free(self):
        payload = completion(); payload.pop("usage")
        self.transport.return_value = payload
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertGreater(self.judge().summary()["accounted_cny_at_peak_rates"], 0)
        self.assertEqual(self.judge().summary()["unresolved_attempts"], 1)

    def test_format_retry_is_bounded_and_accounted(self):
        self.transport.side_effect = [completion("invalid json"), completion()]
        label = self.judge().label(response())
        self.assertEqual(label["label_status"], "ok")
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(self.judge().summary()["http_attempts"], 2)
        self.assertEqual(len(self.transport.call_args.args[0]["messages"]), 2)

    def test_three_empty_responses_stop_without_fourth_attempt(self):
        self.transport.return_value = completion("")
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.transport.call_count, 3)

    def test_cache_receipt_tamper_and_contract_drift_block(self):
        self.judge().label(response())
        receipt = next((self.root / "receipts").glob("*.json"))
        atomic_json(receipt, completion("changed"))
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("other"))
        with self.assertRaises(ReplicationStop):
            ContentJudge(self.root, JUDGE, {"protocol": "different"}, BUDGET, transport=self.transport)

    def test_actual_usage_over_reservation_is_recorded_then_stops(self):
        payload = completion()
        payload["usage"] = {"prompt_tokens": 999999, "completion_tokens": 20, "total_tokens": 1000019}
        self.transport.return_value = payload
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.judge().summary()["known_prompt_tokens"], 999999)


class SchedulerTests(unittest.TestCase):
    def test_insufficient_population_blocks_before_gpu_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = object.__new__(ReplicationPipeline)
            pipeline.config = SimpleNamespace(output_root=Path(tmp), fingerprint="f"*64)
            pipeline.spec = SimpleNamespace(verify=Mock(), fingerprint="f"*64, raw={"budget": BUDGET})
            pipeline._state = lambda: {"stages": {s: {} for s in STAGES[:6]}}
            pipeline._stage_is_fresh = lambda *args: True
            pipeline.population = lambda: {"event_pair_count": 15}
            pipeline._run_layer_map = Mock(side_effect=AssertionError("GPU must not run"))
            with self.assertRaisesRegex(ReplicationStop, "Fewer than 16"):
                pipeline.run(["layer_map"], acknowledge_cost=True, confirm_no_external_rq2_use=True)
            pipeline._run_layer_map.assert_not_called()

    def test_forbidden_stages_crossing_and_ack_gate_before_any_write(self):
        pipeline = object.__new__(ReplicationPipeline)
        for stages in [("formal_generate",), ("mechanism_generate",), ("sources", "oracle_generate"), ("sources",)]:
            with self.assertRaises(ReplicationStop):
                pipeline.run(stages)

    def test_resumed_batch_skipped_trajectories_retained(self):
        pipeline = object.__new__(ReplicationPipeline)
        with patch.object(RQ2Pipeline, "_write_resolved_manifest") as writer:
            pipeline._write_resolved_manifest({"cases": [{"pair_id": "p", "status": "skipped"}]})
        self.assertEqual(writer.call_args.args[0]["cases"][0]["status"], "completed")

    def test_event_center_plan_keeps_zero_utility_pair_and_only_allowed_layers(self):
        ids, trials, labels, scan, resolved, events = population_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = object.__new__(ReplicationPipeline)
            pipeline.config = SimpleNamespace(output_root=root,
                stage_path=lambda stage: root / {"events": "events.json", "state_index": "index.json"}[stage])
            pipeline.spec = SimpleNamespace(pair_ids=ids)
            pipeline._source_data = lambda: (trials, {}, records_by_id(labels))
            pipeline._effective_manifest = lambda: root / "resolved.json"
            atomic_json(root / "trajectory_behavior/scan_index.json", scan)
            atomic_json(root / "events.json", events)
            atomic_json(root / "resolved.json", {"records": resolved})
            pipeline._run_state_index()
            self.assertEqual(pipeline.population()["event_pair_count"], 20)
            plan = pipeline._phase_plan("oracle_pilot")
            self.assertEqual(len(plan), 320)
            self.assertEqual({p.layer for p in plan}, {18,19,20,23,24,25,26})
            self.assertEqual({p.state_key for p in plan}, {EVENT_CENTER})
            self.assertIn("dev_1", {p.pair_id for p in plan})


if __name__ == "__main__":
    unittest.main()
