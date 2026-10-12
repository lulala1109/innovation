"""Focused CPU/mock checks for operator-accepted alias recovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rq2.alias_resume import (ALIASES, POLICY, IMPORTED_STAGES, AliasJudge, AliasPipeline,
                              audit_scoring, receipt_label)
from rq2.artifacts import atomic_json, atomic_jsonl, canonical_sha256, file_sha256
from rq2.replication_config import BUDGET
from rq2.replication_judge import ReplicationStop
from rq2.replication_rejudge import MigrationJudge
from tests.test_rq2_replication import completion, response

JUDGE = {"provider": "deepseek", "model": "DeepSeek-V4.1-Flash", "base_url": "https://fixture.invalid/v1"}
OLD = {"model": JUDGE["model"], "policy": "strict"}
NEW = {"model": JUDGE["model"], "alias_acceptance": POLICY}
HISTORY = {"cost_micro_cny": 10578, "http_attempts": 5, "ledger_sha256": "a" * 64}


def payload(model):
    result = completion()
    result["model"] = model
    return result


class AliasScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = Mock(side_effect=[payload(ALIASES[0]), payload(ALIASES[1])])

    def judge(self):
        return AliasJudge(self.root, JUDGE, NEW, BUDGET, HISTORY, transport=self.transport)

    def test_exact_two_aliases_preserve_raw_receipts_and_charges(self):
        self.judge().label(response())
        self.judge().label(response("two", reply="second reply"))
        judge = self.judge()
        self.assertEqual([a["provider_model"] for a in judge.state["attempts"]], list(ALIASES))
        self.assertEqual(json.loads((self.root / "receipts/000002.json").read_text())["model"], ALIASES[1])
        self.assertEqual(judge.state["provider_model"], ALIASES[0])
        self.assertAlmostEqual(judge.summary()["cumulative_accounted_cny_at_peak_rates"], .010578 + 2 * .00038)
        self.assertEqual(judge.summary()["cumulative_http_attempts"], 7)
        self.assertEqual(self.transport.call_args.args[0]["model"], JUDGE["model"])

    def test_unknown_model_stops_even_on_first_call_and_after_restart(self):
        self.transport.side_effect = [payload("deepseek-flash-unapproved")]
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("two"))
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(self.judge().state["attempts"][0]["status"], "settled")

    def test_token_overrun_is_not_excused_by_an_accepted_alias(self):
        result = payload(ALIASES[1])
        result["usage"] = {"prompt_tokens": 100, "completion_tokens": 801, "total_tokens": 901}
        self.transport.side_effect = [result]
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.judge().state["attempts"][0]["completion_tokens"], 801)
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("again"))
        self.assertEqual(self.transport.call_count, 1)

    def test_timeout_retains_reservation(self):
        self.transport.side_effect = TimeoutError()
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("again"))
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(self.judge().summary()["unresolved_attempts"], 1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old, self.new = self.root / "source", self.root / "derived"
        transport = Mock(side_effect=[payload(ALIASES[0]), payload(ALIASES[1])])
        original = MigrationJudge(self.old / "scoring", JUDGE, OLD, BUDGET, HISTORY, transport=transport)
        self.responses = {"trajectory_behavior": {"base": response("base")},
                          "oracle_pilot": {"oracle": response("oracle", reply="oracle reply")}}
        label = original.label(self.responses["trajectory_behavior"]["base"])
        with self.assertRaises(ReplicationStop):
            original.label(self.responses["oracle_pilot"]["oracle"])
        atomic_jsonl(self.old / "trajectory_behavior/labels.jsonl", [label])
        atomic_jsonl(self.old / "oracle_pilot/labels.jsonl", [])

    def audit(self):
        return audit_scoring(self.old, JUDGE, OLD, NEW, self.responses, HISTORY)

    def test_paid_uncached_receipt_recovered_without_calling_http(self):
        before = {p: file_sha256(p) for p in self.old.rglob("*") if p.is_file()}
        audit = self.audit()
        self.assertEqual(audit["summary"]["recovered_receipt_numbers"], [2])
        self.assertEqual(audit["summary"]["remaining_oracle_unique_inputs"], 0)
        self.assertEqual(audit["summary"]["charged_attempts_preserved"], 2)
        self.assertEqual(before, {p: file_sha256(p) for p in before})

    def test_request_hash_or_usage_mismatch_is_not_reconciled(self):
        p = self.old / "scoring/ledger.json"
        original = json.loads(p.read_text())
        for field, value in [("request_sha256", "bad"), ("cost_micro_cny", 1), ("status", "uncertain")]:
            altered = copy.deepcopy(original)
            altered["attempts"][-1][field] = value
            atomic_json(p, altered)
            with self.assertRaises(ReplicationStop):
                self.audit()
        atomic_json(p, original)

    def make_pipeline(self):
        paths = {"sources": "provenance/rq1_sources.json", "trajectory": "trajectory/summary.json",
                 "trajectory_behavior_generate": "trajectory_behavior/responses.jsonl",
                 "trajectory_behavior_judge": "trajectory_behavior/labels.jsonl", "events": "state_index/behavior_events.json",
                 "state_index": "state_index/index.json", "layer_map": "provenance/layer_map.json",
                 "identity": "identity_tests.json", "oracle_generate": "oracle_pilot/responses.jsonl",
                 "oracle_judge": "oracle_pilot/labels.jsonl", "report": "rq2_report.md",
                 "oracle_analyze": "oracle_pilot/analysis.json"}
        extra = ["provenance/reference_statistics.json", "trajectory/resolved_manifest.json",
                 "trajectory_behavior/trials.jsonl", "trajectory_behavior/scan_index.json", "trajectory_behavior/commits.jsonl",
                 "state_index/event_population.json", "oracle_pilot/trials.jsonl", "oracle_pilot/commits.jsonl",
                 "oracle_pilot/baseline_import_map.json", "oracle_pilot/baseline_replay_audit.json"]
        for rel in {*[paths[s] for s in IMPORTED_STAGES], *extra}:
            if not (self.old / rel).exists():
                atomic_json(self.old / rel, {"fixture": rel})
        manifest = self.old / "manifest.json"
        atomic_json(manifest, {"records": []})
        # Use a complete synthetic import manifest; validation itself is tested
        # by audit_scoring and the real, offline config preflight.
        artifacts = {str(self.old / rel): file_sha256(self.old / rel)
                     for rel in {*[paths[s] for s in IMPORTED_STAGES], *extra}}
        atomic_json(self.old / "pipeline_state.json", {"stages": {s: {"artifacts": artifacts} for s in IMPORTED_STAGES}})
        (self.old / ".run.lock").touch()
        ancestor = self.root / "ancestor"
        ancestor.mkdir()
        (ancestor / ".run.lock").touch()
        config = SimpleNamespace(output_root=self.new, name="fixture", fingerprint="b" * 64,
                                 manifest=manifest, judge=JUDGE, trajectory={"mode": "generate"},
                                 stage_path=lambda s: self.new / paths[s])
        spec = SimpleNamespace(runtime_config=config, fingerprint=config.fingerprint, path=self.root / "config.json",
            source=SimpleNamespace(output_root=self.old, fingerprint="c" * 64,
                source=SimpleNamespace(output_root=ancestor), raw={"scoring_contract": OLD}),
            raw={"scoring_contract": NEW, "historical_accounting": HISTORY, "source_snapshot_sha256": "d" * 64})
        atomic_json(spec.path, {"fixture": True})
        spec.inspect = Mock(return_value=({}, self.audit()))
        pipeline = AliasPipeline(spec, transport=Mock(side_effect=AssertionError("No API during prepare")))
        pipeline._phase_data = Mock()
        pipeline._scoring_audit = Mock(return_value={"passed": True})
        return pipeline

    def test_prepare_is_readonly_source_idempotent_and_preserves_attempts(self):
        pipeline = self.make_pipeline()
        before = {p: file_sha256(p) for p in self.old.rglob("*") if p.is_file()}
        first = pipeline.prepare()
        second = pipeline.prepare()
        self.assertEqual(first["status"], "prepared")
        self.assertEqual(second["status"], "already_prepared")
        judge = pipeline._judge()
        recovered = judge.label(self.responses["oracle_pilot"]["oracle"])
        expected = receipt_label(json.loads((self.old / "scoring/receipts/000002.json").read_text()),
                                 self.responses["oracle_pilot"]["oracle"], canonical_sha256(NEW))
        self.assertEqual(recovered, expected)
        self.assertEqual(judge.summary()["cumulative_http_attempts"], 7)
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        pipeline.transport.assert_not_called()

    def test_gpu_and_unacknowledged_run_are_blocked(self):
        pipeline = object.__new__(AliasPipeline)
        with self.assertRaises(ReplicationStop):
            pipeline.model()
        for stages, accept in [(["oracle_generate"], True), (["trajectory_behavior_judge"], True), (["oracle_judge"], False)]:
            with self.assertRaises(ReplicationStop):
                pipeline.run(stages, acknowledge_cost=True, accept_unverified_aliases=accept)

    def test_report_prominently_states_unverified_assumption(self):
        pipeline = self.make_pipeline()
        atomic_json(self.new / "oracle_pilot/analysis.json", {"pilot_decision": {"qualified": True}, "budget": {}})
        pipeline.population = lambda: {"event_pair_count": 20}
        pipeline._run_report()
        report = (self.new / "rq2_report.md").read_text()
        self.assertTrue(report.startswith("# 重要限制"))
        self.assertIn("未经平台确认", report)


if __name__ == "__main__":
    unittest.main()
