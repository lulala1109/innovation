"""Small CPU-only migration tests; no real HTTP, CUDA or project data."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rq2.artifacts import atomic_json, canonical_sha256, file_sha256
from rq2.replication_config import BUDGET
from rq2.replication_judge import ContentJudge, ReplicationStop
from rq2.replication_rejudge import (
    COPY_FILES, IMPORTED_STAGES, MigrationJudge, RejudgePipeline, remaining_budget,
)
from tests.test_rq2_replication import completion, response

JUDGE = {"provider": "deepseek", "model": "DeepSeek-V4.1-Flash", "base_url": "https://fixture.invalid/v1"}
CONTRACT = {"model": "DeepSeek-V4.1-Flash", "protocol": "cpu-test"}
HISTORY = {"cost_micro_cny": 10578, "http_attempts": 5, "ledger_sha256": "f" * 64}


class JudgeMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = Mock(return_value=completion())

    def judge(self, history=None):
        return MigrationJudge(self.root / "new", JUDGE, CONTRACT, BUDGET, history or HISTORY,
                              transport=self.transport)

    def test_old_cache_not_reused_and_baseline_oracle_share_new_contract(self):
        old_transport = Mock(return_value=completion())
        ContentJudge(self.root / "old", {**JUDGE, "model": "DeepSeek-V4-Flash"},
                     {"model": "DeepSeek-V4-Flash"}, BUDGET, transport=old_transport).label(response())
        old_hashes = {p: file_sha256(p) for p in (self.root / "old").rglob("*.json")}
        first = self.judge().label(response())
        second = self.judge().label(response("oracle"))
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(self.transport.call_args.args[0]["model"], "DeepSeek-V4.1-Flash")
        self.assertEqual(first["judge_fingerprint"], second["judge_fingerprint"])
        self.assertEqual(old_hashes, {p: file_sha256(p) for p in old_hashes})
        summary = self.judge().summary()
        self.assertAlmostEqual(summary["limit_cny"], 29.989422)
        self.assertAlmostEqual(summary["cumulative_accounted_cny_at_peak_rates"], .010958)
        self.assertEqual(summary["cumulative_http_attempts"], 6)

    def test_cumulative_money_and_request_limit_block_before_http(self):
        with self.assertRaises(ReplicationStop):
            self.judge({**HISTORY, "cost_micro_cny": 29999999}).label(response())
        with self.assertRaises(ReplicationStop):
            remaining_budget(BUDGET, {**HISTORY, "http_attempts": 7080})
        self.transport.assert_not_called()

    def test_model_drift_still_blocks_and_cannot_be_reset_by_restart(self):
        other = completion(); other["model"] = "different-model"
        self.transport.side_effect = [completion(), other]
        self.judge().label(response())
        with self.assertRaisesRegex(ReplicationStop, "model drift"):
            self.judge().label(response("two", reply="different reply"))
        with self.assertRaises(ReplicationStop):
            self.judge().label(response("three", reply="another reply"))
        self.assertEqual(self.transport.call_count, 2)

    def test_uncertain_call_keeps_reservation_and_no_automatic_retry(self):
        self.transport.side_effect = TimeoutError()
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.transport.call_count, 1)
        self.assertGreater(self.judge().summary()["cumulative_accounted_cny_at_peak_rates"], .010578)


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old, self.new = self.root / "old", self.root / "new"
        self.old.mkdir()
        (self.old / ".run.lock").touch()
        self.manifest = self.old / "manifest.json"
        atomic_json(self.manifest, {"records": []})
        atomic_json(self.old / "provenance/rq1_sources.json", {"judge_protocol_fingerprint": "old"})
        for relative in COPY_FILES:
            atomic_json(self.old / relative, {"fixture": relative})
        atomic_json(self.old / "trajectory_behavior/labels.jsonl", {"old_label": True})
        atomic_json(self.old / "scoring/ledger.json", {"historical": True})
        self.paths = {"sources": "provenance/rq1_sources.json", "trajectory": "trajectory/summary.json",
                      "trajectory_behavior_generate": "trajectory_behavior/responses.jsonl",
                      "trajectory_behavior_judge": "trajectory_behavior/labels.jsonl",
                      "oracle_judge": "oracle_pilot/labels.jsonl",
                      "state_index": "state_index/index.json"}
        config = SimpleNamespace(output_root=self.new, fingerprint="b" * 64, manifest=self.manifest,
                                 judge=JUDGE, trajectory={"mode": "generate"},
                                 stage_path=lambda s: self.new / self.paths[s])
        self.spec = SimpleNamespace(runtime_config=config, path=self.root / "new_config.json", fingerprint="b" * 64,
                                    source=SimpleNamespace(output_root=self.old, path=self.old / "config.json", fingerprint="a" * 64),
                                    verify=Mock(), raw={"scoring_contract": CONTRACT, "budget": BUDGET,
                                        "historical_accounting": HISTORY, "source_state_fingerprint": "c" * 64})
        atomic_json(self.spec.path, {"fixture": True})
        self.pipeline = RejudgePipeline(self.spec, transport=Mock(return_value=completion()))
        self.pipeline._phase_data = Mock()

    def test_import_is_idempotent_readonly_source_and_has_no_old_labels(self):
        before = {p: file_sha256(p) for p in self.old.rglob("*") if p.is_file()}
        with patch.object(self.pipeline, "model", side_effect=AssertionError("No GPU")):
            self.pipeline.prepare()
            self.pipeline.prepare()
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        self.assertFalse((self.new / "scoring").exists())
        self.assertFalse((self.new / "trajectory_behavior/labels.jsonl").exists())
        self.assertEqual(set(self.pipeline._state()["stages"]), set(IMPORTED_STAGES))
        for relative in COPY_FILES:
            self.assertEqual(file_sha256(self.old / relative), file_sha256(self.new / relative))

    def test_import_tampering_and_old_labels_without_state_fail_closed(self):
        self.pipeline.prepare()
        atomic_json(self.new / "trajectory_behavior/responses.jsonl", {"changed": True})
        with self.assertRaisesRegex(ReplicationStop, "changed"):
            self.pipeline.prepare()
        self.new = self.root / "another"
        self.pipeline.config.output_root = self.new
        atomic_json(self.new / "trajectory_behavior/labels.jsonl", {"stale": True})
        with self.assertRaisesRegex(ReplicationStop, "without import state"):
            self.pipeline.prepare()

    def test_no_baseline_gpu_or_cross_group_or_unacknowledged_run(self):
        for stages, flags in [(["trajectory"], True), (["trajectory_behavior_judge", "oracle_judge"], True),
                              (["trajectory_behavior_judge"], False)]:
            with self.assertRaises(ReplicationStop):
                self.pipeline.run(stages, acknowledge_cost=flags, confirm_no_external_rq2_use=flags)
        self.assertFalse(self.new.exists())

    def test_generation_identity_retained_but_oracle_identity_new(self):
        expected = canonical_sha256({"config": "a" * 64, "phase": "trajectory_behavior", "design_version": 4})
        self.assertEqual(self.pipeline._run_fingerprint("trajectory_behavior"), expected)
        atomic_json(self.new / "state_index/index.json", {"new_labels": True})
        oracle = self.pipeline._run_fingerprint("oracle_pilot")
        self.assertEqual(oracle, canonical_sha256({"config": "b" * 64, "phase": "oracle_pilot",
            "design_version": 4, "state_index_sha256": file_sha256(self.new / "state_index/index.json")}))

    def test_both_judge_phases_use_new_model_and_reuse_content(self):
        self.pipeline.prepare()
        self.pipeline._phase_data.return_value = ([], {"one": response()})
        self.pipeline._scoring_audit = Mock(return_value={"passed": True})
        self.pipeline._run_judge_phase("trajectory_behavior")
        self.pipeline._phase_data.return_value = ([], {"oracle": response("oracle")})
        self.pipeline._run_judge_phase("oracle_pilot")
        self.assertEqual(self.pipeline.transport.call_count, 1)
        self.assertEqual(self.pipeline.transport.call_args.args[0]["model"], "DeepSeek-V4.1-Flash")
        self.assertTrue((self.new / "trajectory_behavior/labels.jsonl").is_file())
        self.assertTrue((self.new / "oracle_pilot/labels.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
