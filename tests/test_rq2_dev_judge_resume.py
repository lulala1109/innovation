"""Focused offline/mock tests; never contacts the provider or loads a GPU."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from rq2.artifacts import atomic_json, canonical_sha256, file_sha256
from rq2.dev_judge_resume import DevWarningJudge, DevWarningPipeline, POLICY, audit_scoring
from rq2.replication_config import BUDGET
from rq2.replication_judge import ReplicationStop
import tests.test_rq2_alias_resume as legacy
from tests.test_rq2_replication import response

JUDGE, OLD, HISTORY, payload = legacy.JUDGE, legacy.OLD, legacy.HISTORY, legacy.payload
NEW = {"model": JUDGE["model"], "identity_policy": POLICY}


class WarningJudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.transport = Mock()
        self.output = io.StringIO()
        self.capture = redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def judge(self, *, budget=None, contract=None):
        return DevWarningJudge(self.root, JUDGE, contract or NEW, budget or BUDGET,
                               HISTORY, transport=self.transport)

    def test_names_and_missing_identity_warn_without_stopping_or_rescoring(self):
        names = ["deepseek-flash", "deepseek-v4-1-flash-260910", "deepseek-v4.1-flash", "unexpected-new-name"]
        replies = [payload(m) for m in names]
        missing = payload("remove")
        missing.pop("model")
        replies.append(missing)
        self.transport.side_effect = replies
        for i in range(5):
            self.judge().label(response(str(i), reply=f"reply {i}"))
        judge = self.judge()
        self.assertFalse(judge.state.get("fatal_error"))
        self.assertEqual([r["provider_model"] for r in judge.state["attempts"]], [*names, None])
        self.assertFalse(judge.state["attempts"][-1]["returned_model_field_present"])
        self.assertNotIn("model", json.loads((self.root / "receipts/000005.json").read_text()))
        for i in range(5):
            self.judge().label(response(str(i), reply=f"reply {i}"))
        self.assertEqual(self.transport.call_count, 5)
        self.assertTrue(all(c.args[0]["model"] == JUDGE["model"] for c in self.transport.call_args_list))
        self.assertEqual(judge.summary()["cumulative_http_attempts"], 10)
        audit = json.loads((self.root / "model_identity_audit.json").read_text())
        self.assertEqual(audit["unknown_model_attempts"], [5])
        self.assertEqual(len(audit["model_transitions"]), 4)
        self.assertIn("judge identity warning", self.output.getvalue())
        self.assertFalse(audit["model_version_locked"])

    def test_input_and_output_overrun_still_stop_and_record_actual_charge(self):
        for index, (p, c) in enumerate([(100, 801), (100000, 20)]):
            with self.subTest(p=p, c=c):
                directory = self.root / str(index)
                result = payload("unrecognized")
                result["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p+c}
                transport = Mock(return_value=result)
                def judge():
                    return DevWarningJudge(directory, JUDGE, NEW, BUDGET, HISTORY, transport=transport)
                with self.assertRaisesRegex(ReplicationStop, "Usage exceeded"):
                    judge().label(response())
                with self.assertRaises(ReplicationStop):
                    judge().label(response("again", reply="again"))
                self.assertEqual(transport.call_count, 1)
                row = judge().state["attempts"][0]
                self.assertEqual(row["cost_micro_cny"], p*2+c*9)
                self.assertEqual(row["provider_model"], "unrecognized")
                self.assertEqual(judge().state["fatal_error"], "usage_exceeded_reservation")

    def test_missing_usage_and_timeout_keep_reservation_and_no_repeat(self):
        for index, mode in enumerate(["usage", "timeout", "sum"]):
            with self.subTest(mode=mode):
                transport = Mock()
                result = payload("new-name")
                if mode == "usage":
                    result.pop("usage")
                    transport.return_value = result
                elif mode == "sum":
                    result["usage"]["total_tokens"] += 1
                    transport.return_value = result
                else:
                    transport.side_effect = TimeoutError()
                def judge():
                    return DevWarningJudge(self.root / str(index), JUDGE, NEW, BUDGET, HISTORY, transport=transport)
                with self.assertRaises(ReplicationStop):
                    judge().label(response())
                with self.assertRaises(ReplicationStop):
                    judge().label(response("again", reply="again"))
                self.assertEqual(transport.call_count, 1)
                row = judge().state["attempts"][0]
                self.assertEqual(row["status"], "uncertain")
                self.assertEqual(row["cost_micro_cny"], row["reserved_micro_cny"])

    def test_bad_scoring_schema_stops_after_bounded_repair(self):
        result = payload("new-name")
        result["choices"][0]["message"]["content"] = "invalid json"
        self.transport.return_value = result
        with self.assertRaisesRegex(ReplicationStop, "unresolved"):
            self.judge().label(response())
        self.assertEqual(self.transport.call_count, 2)
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(len(self.transport.call_args_list[1].args[0]["messages"]), 2)

    def test_empty_results_keep_three_attempt_limit(self):
        result = payload("new-name")
        result["choices"][0]["message"]["content"] = ""
        self.transport.return_value = result
        with self.assertRaises(ReplicationStop):
            self.judge().label(response())
        self.assertEqual(self.transport.call_count, 3)

    def test_budget_http_cap_and_contract_binding_cannot_be_bypassed(self):
        for budget in [{**BUDGET, "api_limit_cny": .011}, {**BUDGET, "http_attempt_limit": 5}]:
            with self.assertRaises(ReplicationStop):
                self.judge(budget=budget).label(response())
        self.transport.assert_not_called()
        self.transport.return_value = payload("new-name")
        self.judge().label(response())
        with self.assertRaisesRegex(ReplicationStop, "contract drift"):
            self.judge(contract={**NEW, "model": "different-request"})


class WarningRecoveryTests(unittest.TestCase):
    def setUp(self):
        legacy.RecoveryTests.setUp(self)
        receipt = self.old / "scoring/receipts/000002.json"
        value = json.loads(receipt.read_text())
        value["model"] = "deepseek-v4.1-flash"
        atomic_json(receipt, value)
        ledger = self.old / "scoring/ledger.json"
        value = json.loads(ledger.read_text())
        value["attempts"][-1]["provider_model"] = "deepseek-v4.1-flash"
        atomic_json(ledger, value)

    def audit(self):
        return audit_scoring(self.old, JUDGE, OLD, NEW, self.responses, HISTORY)

    def pipeline(self):
        old = legacy.RecoveryTests.make_pipeline(self)
        old.spec.raw["scoring_contract"] = NEW
        pipeline = DevWarningPipeline(old.spec, transport=old.transport)
        pipeline._phase_data = old._phase_data
        pipeline._scoring_audit = old._scoring_audit
        return pipeline

    def test_offline_recovery_keeps_sources_scores_receipts_and_accounting(self):
        pipeline = self.pipeline()
        before = {p: file_sha256(p) for p in self.old.rglob("*") if p.is_file()}
        with redirect_stdout(io.StringIO()):
            first = pipeline.prepare()
            second = pipeline.prepare()
        self.assertEqual(first["offline_audit"]["recovered_receipt_numbers"], [2])
        self.assertEqual(second["status"], "already_prepared")
        judge = pipeline._judge()
        label = judge.label(self.responses["oracle_pilot"]["oracle"])
        self.assertEqual(label["judge_fingerprint"], canonical_sha256(NEW))
        self.assertEqual(judge.summary()["cumulative_http_attempts"], 7)
        self.assertAlmostEqual(judge.summary()["cumulative_accounted_cny_at_peak_rates"], .010578+.00076)
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        self.assertEqual(file_sha256(self.old / "scoring/receipts/000002.json"),
                         file_sha256(self.new / "scoring/receipts/000002.json"))
        pipeline.transport.assert_not_called()

    def test_other_audit_errors_are_not_waived(self):
        path = self.old / "scoring/ledger.json"
        original = json.loads(path.read_text())
        for field, value in [("status", "uncertain"), ("request_sha256", "bad"), ("cost_micro_cny", 1),
                             ("provider_model", "different-from-receipt")]:
            row = copy.deepcopy(original)
            row["attempts"][-1][field] = value
            atomic_json(path, row)
            with self.assertRaises(ReplicationStop):
                self.audit()
        atomic_json(path, original)

    def test_only_dev_judge_phases_and_explicit_acknowledgements(self):
        pipeline = object.__new__(DevWarningPipeline)
        with self.assertRaises(ReplicationStop):
            pipeline.model()
        for stages, accept in [(["formal"], True), (["oracle_generate"], True),
                               (["trajectory_behavior_judge"], True), (["oracle_judge"], False)]:
            with self.assertRaises(ReplicationStop):
                pipeline.run(stages, acknowledge_cost=True, accept_unlocked_judge=accept)

    def test_report_and_machine_analysis_qualify_service_level_results(self):
        pipeline = self.pipeline()
        with redirect_stdout(io.StringIO()):
            pipeline.prepare()
        pipeline.population = lambda: {"event_pair_count": 20}
        result = {"pilot_decision": {"qualified": True}, "budget": {}}
        atomic_json(self.new / "oracle_pilot/analysis.json", result)
        pipeline._run_report()
        report = (self.new / "rq2_report.md").read_text()
        self.assertTrue(report.startswith("# 重要限制"))
        self.assertIn("实际 Judge 服务版本未锁定", report)
        self.assertIn("deepseek-v4.1-flash", report)
        with patch("rq2.dev_judge_resume.ReplicationPipeline._run_oracle_analyze"):
            pipeline._run_oracle_analyze()
        result = json.loads((self.new / "oracle_pilot/analysis.json").read_text())
        self.assertFalse(result["model_version_locked"])
        self.assertFalse(result["allow_downstream_execution"])
        self.assertEqual(result["pilot_decision"]["qualification_scope"], "dev_service_level_only_not_formal")


if __name__ == "__main__":
    unittest.main()
