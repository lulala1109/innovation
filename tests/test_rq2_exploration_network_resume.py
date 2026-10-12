"""Offline recovery tests: preserve uncertain charges and never retry implicitly."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import proxy_bypass

from experiments.rq2_exploration_network_resume import main
from rq2.artifacts import atomic_json, file_sha256
from rq2.exploration_network_resume import (
    NetworkResumeJudge, NetworkResumePipeline, check_endpoint, configure_direct_connection,
    endpoint_connection, inspect_failure,
)
from rq2.exploration_pipeline import ExplorationJudge
from rq2.replication_judge import ContentJudge, ReplicationStop
from tests.test_rq2_replication import JUDGE, completion, response


BUDGET = {"api_limit_cny": 30, "http_attempt_limit": 100,
          "input_cny_per_million": 2, "output_cny_per_million": 9}
HISTORY = {"cost_micro_cny": 1000, "http_attempts": 2}
CONTRACT = {"recovery_fixture": True}


class NetworkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = ContentJudge(self.root / "source", JUDGE, {"old": True}, BUDGET, transport=Mock())
        self.transport = Mock(return_value=completion())
        self.prior = ExplorationJudge(self.root / "prior", JUDGE, CONTRACT, BUDGET, HISTORY,
                                      source_judge=self.source, transport=self.transport)
        self.original_response = response()
        self.failed_response = response("failed", reply="different fixture reply")

    def fail(self, preceding_empty=False):
        self.prior.label(self.original_response)
        self.transport.side_effect = ([completion(""), URLError("fixture DNS failure")]
                                     if preceding_empty else URLError("fixture DNS failure"))
        with self.assertRaises(ReplicationStop):
            self.prior.label(self.failed_response)
        self.before = {str(p): file_sha256(p) for p in self.prior.root.rglob("*") if p.is_file()}
        return inspect_failure(self.prior, len(self.prior.state["attempts"]))

    def recovered(self, transport=None, budget=None):
        failed, counts = inspect_failure(self.prior, len(self.prior.state["attempts"]))
        history = {"cost_micro_cny": HISTORY["cost_micro_cny"]
                     + sum(a["cost_micro_cny"] for a in self.prior.state["attempts"]),
                   "http_attempts": HISTORY["http_attempts"] + len(self.prior.state["attempts"])}
        # Existing cache validation uses the original ContentJudge method, which
        # checks its receipt without allowing this historical ledger to send.
        prior = ContentJudge(self.prior.root, JUDGE, CONTRACT, self.prior.budget, transport=Mock())
        return NetworkResumeJudge(self.root / "recovered", JUDGE, CONTRACT, budget or BUDGET, history,
                    source_judge=self.source, prior=prior, failed=failed, historical_counts=counts,
                    transport=transport if transport is not None else Mock(return_value=completion()))

    def test_retry_and_cache_reuse_preserve_original_bytes_and_cost(self):
        failed, _ = self.fail()
        transport = Mock(return_value=completion())
        judge = self.recovered(transport)
        judge.label(self.original_response)
        transport.assert_not_called()
        first = judge.label(self.failed_response)
        self.assertEqual(first, judge.label(self.failed_response))
        self.recovered(transport).label(self.failed_response)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(self.before, {str(p): file_sha256(p) for p in self.prior.root.rglob("*") if p.is_file()})
        self.assertEqual(self.prior.state["attempts"][-1]["status"], "uncertain")
        summary = judge.summary()
        self.assertEqual(summary["historical_uncertain_attempts"], 1)
        self.assertEqual(summary["cumulative_http_attempts"], HISTORY["http_attempts"] + 3)
        self.assertAlmostEqual(summary["cumulative_accounted_cny_at_peak_rates"],
                               (HISTORY["cost_micro_cny"] + 380 + failed["reserved_micro_cny"] + 380) / 1e6)

    def test_second_network_failure_stays_blocked_across_restart(self):
        self.fail()
        transport = Mock(side_effect=URLError("fixture connection failure"))
        with self.assertRaises(ReplicationStop):
            self.recovered(transport).label(self.failed_response)
        with self.assertRaisesRegex(ReplicationStop, "Recovery request unresolved"):
            self.recovered(transport).label(self.failed_response)
        self.assertEqual(transport.call_count, 1)

    def test_three_attempt_cap_includes_the_uncertain_original(self):
        self.fail(preceding_empty=True)
        transport = Mock(return_value=completion(""))
        with self.assertRaisesRegex(ReplicationStop, "across ledgers"):
            self.recovered(transport).label(self.failed_response)
        self.assertEqual(transport.call_count, 1)

    def test_existing_receipt_blocks_retry(self):
        self.fail()
        number = len(self.prior.state["attempts"])
        atomic_json(self.prior.root / "receipts" / f"{number:06d}.json", completion())
        with self.assertRaisesRegex(ReplicationStop, "receipt/cache exists"):
            inspect_failure(self.prior, number)

    def test_wrong_attempt_or_missing_usage_is_not_accepted(self):
        self.fail()
        with self.assertRaises(ReplicationStop):
            inspect_failure(self.prior, 1)
        self.prior.state["attempts"][-1]["error"] = "invalid_or_missing_usage"
        with self.assertRaises(ReplicationStop):
            inspect_failure(self.prior, 2)

    def test_receipt_usage_mismatch_blocks_recovery(self):
        self.fail()
        payload = completion()
        payload["usage"]["prompt_tokens"] += 1
        atomic_json(self.prior.root / "receipts/000001.json", payload)
        with self.assertRaisesRegex(ReplicationStop, "receipt/usage/accounting mismatch"):
            inspect_failure(self.prior, 2)

    def test_cumulative_budget_is_not_reset(self):
        self.fail()
        transport = Mock()
        with self.assertRaisesRegex(ReplicationStop, "CNY pause threshold"):
            self.recovered(transport, {**BUDGET, "api_limit_cny": .02}).label(self.failed_response)
        transport.assert_not_called()

    def test_tampered_previous_cache_does_not_cause_regrading(self):
        self.fail()
        judge = self.recovered()
        atomic_json(self.prior.root / "receipts/000001.json", completion("changed"))
        with self.assertRaisesRegex(ReplicationStop, "receipt changed/missing"):
            judge.label(self.original_response)
        judge.transport.assert_not_called()

    def test_scope_and_acknowledgement_are_required_before_writes(self):
        pipe = object.__new__(NetworkResumePipeline)
        with self.assertRaises(ReplicationStop):
            pipe.run(("window_judge", "window_analyze", "report"))
        with self.assertRaises(ReplicationStop):
            pipe.run(("window_generate",), accept_uncertain_charge_and_retry=True)
        with patch("experiments.rq2_exploration_network_resume.load_config") as loader:
            self.assertEqual(main(["--config", "unused", "--attempt", "234", "run"]), 2)
            loader.assert_not_called()

    def test_probe_401_only_means_network_reachable_and_sends_no_secret(self):
        opener = Mock()
        opener.open.side_effect = HTTPError("https://fixture.invalid/v1/models", 401, "Unauthorized", {}, None)
        with patch("rq2.exploration_network_resume.build_opener", return_value=opener):
            result = check_endpoint(JUDGE)
        self.assertTrue(result["https_reachable"])
        self.assertFalse(result["api_key_or_balance_verified"])
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertIsNone(request.data)

    def test_probe_reports_network_reason_and_does_not_retry(self):
        opener = Mock()
        opener.open.side_effect = URLError("fixture DNS failure")
        with patch("rq2.exploration_network_resume.build_opener", return_value=opener):
            with self.assertRaisesRegex(ReplicationStop, "fixture DNS failure"):
                check_endpoint(JUDGE)
        self.assertEqual(opener.open.call_count, 1)

    def test_direct_mode_bypasses_dead_proxy_and_preserves_other_exclusions(self):
        with patch.dict(os.environ, {"https_proxy": "http://127.0.0.1:1",
                         "NO_PROXY": "localhost", "no_proxy": "existing.invalid"}, clear=True):
            self.assertEqual(endpoint_connection(JUDGE)["connection_mode"], "environment_proxy")
            configure_direct_connection(JUDGE)
            self.assertEqual(endpoint_connection(JUDGE)["connection_mode"], "direct")
            self.assertTrue(proxy_bypass("fixture.invalid"))
            self.assertTrue(proxy_bypass("fixture.invalid:443"))
            self.assertTrue(proxy_bypass("existing.invalid"))
            self.assertTrue(proxy_bypass("localhost"))
            self.assertEqual(os.environ["https_proxy"], "http://127.0.0.1:1")

    def test_network_diagnostics_never_show_proxy_credentials(self):
        with patch.dict(os.environ, {"https_proxy": "http://private-user:private-password@127.0.0.1:7890"}, clear=True):
            result = endpoint_connection(JUDGE)
            self.assertEqual(result["proxy"]["host"], "127.0.0.1")
            self.assertEqual(result["proxy"]["port"], 7890)
            self.assertTrue(result["proxy"]["credentials_present"])
            self.assertNotIn("private-user", str(result))
            self.assertNotIn("private-password", str(result))

    def test_probe_empty_exception_still_reports_type_and_route(self):
        opener = Mock()
        opener.open.side_effect = URLError(TimeoutError())
        with patch("rq2.exploration_network_resume.build_opener", return_value=opener):
            with self.assertRaisesRegex(ReplicationStop, r"Endpoint probe failed .*TimeoutError"):
                check_endpoint(JUDGE)


if __name__ == "__main__":
    unittest.main()
