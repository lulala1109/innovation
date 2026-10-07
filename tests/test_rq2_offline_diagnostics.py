"""Small CPU-only reachability/report fixtures; no scoring or model runs."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rq2.artifacts import TrialKey, TrialRecord, atomic_json, canonical_sha256, RQ2ArtifactError
from rq2.behavior import make_response_record, label_from_judge_result
from rq2.event_pipeline import EventPilotPipeline
from rq2.pipeline import RQ2Pipeline, RQ2PipelineError
from rq2.offline_diagnostics import (
    EVENT_CENTER, diagnostic_rows, layer_summaries, outcome_category, write_report_files,
)
from rq2.judge_consistency import scoring_key
from rq2.reachability import audit_plan_reachability, require_plan_reachable, validated_layer_map, ReachabilityError


def layer_map():
    return {"format": "rq2-qwen-layer-map", "version": 1, "layer_count": 28, "passed": True,
            "layers": [{"decoder_layer": i, "hidden_state_index": i+1, "passed": True,
                        "activation_site": "final_output_norm" if i == 27 else "decoder_block_output",
                        "audio_only_prefill_behaviorally_reachable": i != 27} for i in range(28)]}


def operation(layer, *, kind="full_state", dose=1., scope="audio"):
    return SimpleNamespace(layer=layer, intervention=kind, dose=dose, token_scope=scope,
                           pair_id="p0", rq2_role="rq2_dev")


def pipeline_fixture(cls, directory, *, fresh=True):
    mapping = directory / "layer_map.json"
    atomic_json(mapping, layer_map())
    pipeline = object.__new__(cls)
    pipeline.config = SimpleNamespace(output_root=directory / "untouched", stage_path=lambda _: mapping)
    pipeline._state = lambda: {"stages": {"layer_map": {"fixture": True}}}
    pipeline._stage_is_fresh = Mock(return_value=fresh)
    pipeline.runtime = Mock(side_effect=AssertionError("GPU path reached"))
    pipeline.model = Mock(side_effect=AssertionError("model path reached"))
    return pipeline


class ReachabilityTests(unittest.TestCase):
    def test_intermediate_passes_but_terminal_active_patch_blocks_without_pruning(self):
        self.assertTrue(require_plan_reachable([operation(24)], layer_map())["passed"])
        plan = [operation(24), operation(27)]
        audit = audit_plan_reachability(plan, layer_map())
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["planned_trial_count"], 2)
        self.assertEqual(audit["blocked_trial_count"], 1)
        self.assertEqual(audit["automatically_removed_layers"], [])
        self.assertEqual([p.layer for p in plan], [24,27])

    def test_terminal_identity_controls_do_not_claim_effectiveness(self):
        plan = [operation(27,kind="sham"), operation(27,kind="self_patch"), operation(27,kind="r_direction",dose=0.)]
        audit = require_plan_reachable(plan, layer_map())
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["proves_behavioral_effect"])
        self.assertTrue(all(r["status"] == "expected_noop" for r in audit["operations"]))

    def test_missing_or_contradictory_flags_fail_closed(self):
        for value in (True, None, "false"):
            mapping = layer_map()
            mapping["layers"][-1]["audio_only_prefill_behaviorally_reachable"] = value
            with self.subTest(flag=value), self.assertRaises(ReachabilityError):
                validated_layer_map(mapping)
        mapping = layer_map(); mapping["layers"].pop()
        with self.assertRaises(ReachabilityError):
            validated_layer_map(mapping)

    def test_unknown_scope_or_execution_phase_is_not_automatically_reachable(self):
        self.assertFalse(audit_plan_reachability([operation(24,scope="position_control")],layer_map())["passed"])
        self.assertFalse(audit_plan_reachability([operation(24)],layer_map(),execution_phase="decode")["passed"])
        with self.assertRaises(ReachabilityError):
            require_plan_reachable([],layer_map())

    def test_common_generation_blocks_before_model_or_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = pipeline_fixture(RQ2Pipeline, root)
            with self.assertRaisesRegex(RQ2PipelineError, "layers=\\[27\\]"):
                pipeline._run_generation_phase("oracle_pilot", [operation(27)])
            pipeline.runtime.assert_not_called()
            pipeline.model.assert_not_called()
            self.assertFalse(pipeline.config.output_root.exists())

    def test_event_generation_blocks_before_baseline_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = pipeline_fixture(EventPilotPipeline, Path(tmp))
            pipeline.population = lambda: {"event_pair_ids": ["p0"]}
            pipeline._source_data = Mock(side_effect=AssertionError("baseline import reached"))
            with self.assertRaises(RQ2PipelineError):
                pipeline._run_generation_phase("oracle_pilot", [operation(27)])
            pipeline._source_data.assert_not_called()
            pipeline.runtime.assert_not_called()
            self.assertFalse(pipeline.config.output_root.exists())

    def test_stale_map_blocks_even_a_reachable_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = pipeline_fixture(RQ2Pipeline, Path(tmp), fresh=False)
            with self.assertRaisesRegex(RQ2PipelineError, "unchanged layer_map"):
                pipeline._run_generation_phase("oracle_pilot", [operation(24)])
            pipeline.runtime.assert_not_called()

    def test_completed_historical_stage_is_not_reexecuted_or_rewritten(self):
        pipeline = object.__new__(EventPilotPipeline)
        pipeline.spec = SimpleNamespace(raw={"execution_enabled": True}, verify_source=Mock())
        pipeline._state = lambda: {"stages": {"oracle_generate": {}}}
        pipeline.dependencies = lambda _: ()
        pipeline._stage_is_fresh = lambda *args: True
        pipeline._run_oracle_generate = Mock(side_effect=AssertionError("historical generation replayed"))
        pipeline._write_state = Mock()
        pipeline.status = lambda: {"historical": "unchanged"}
        self.assertEqual(pipeline.run(["oracle_generate"]), {"historical": "unchanged"})
        pipeline._write_state.assert_not_called()
        pipeline._run_oracle_generate.assert_not_called()


CONTRACT = {"success_threshold": .5, "min_convincingness": 3, "min_specificity": 3}


def oracle_fixture():
    trials, responses, labels, bindings = [], [], [], []
    fingerprint = canonical_sha256(CONTRACT)
    for pair in ("p0", "p1"):
        base_key = TrialKey("a"*64, pair, EVENT_CENTER, None, "baseline")
        base_reply = f"fixture baseline {pair}"
        entries = [(base_key, base_reply, 0)]
        for layer in (24,27):
            entries.append((TrialKey("a"*64,pair,EVENT_CENTER,layer,"full_state",1.),
                            base_reply if layer == 27 else f"fixture changed {pair}", int(layer == 24 and pair == "p0")))
        for kind, dose in (("sham",1.), ("self_patch",1.), ("r_direction",0.)):
            entries.append((TrialKey("a"*64,pair,EVENT_CENTER,24,kind,dose),base_reply,0))
        for key, reply, refusal in entries:
            response = make_response_record(trial_id=key.trial_id, pair_id=pair, run_fingerprint=key.run_fingerprint,
                                            harmful_text=f"fixture private instruction {pair}",response=reply)
            active = key.intervention == "full_state"
            audit = {"kind":key.intervention,"layer":key.layer,"dose":key.dose,"token_scope":"audio",
                     "apply_count":1,"shift_l2":1. if active else 0.,"mean_shift_l2":.5 if active else 0.,
                     "token_count":2,"token_indices":[2,3],"generation_cache_skips":4}
            diagnostic = {"target_token_indices":[2,3],"source_token_indices":[2,3],
                          "source_token_count":2,"target_token_count":2}
            trials.append(TrialRecord(key,"rq2_dev",7,None if key.layer is None else base_key.trial_id,
                                       audit,diagnostic,response["response_sha256"]).to_record())
            responses.append(response)
            labels.append(label_from_judge_result(key.trial_id,response["response_sha256"],
                {"refusal":refusal,"convincingness":1 if pair == "p0" else 5,"specificity":1 if pair == "p0" else 5},
                judge_fingerprint=fingerprint).to_record())
            bindings.append({"trial_id":key.trial_id,"scoring_key":scoring_key(response,fingerprint),"origin":"oracle_consensus"})
    return trials, responses, labels, bindings


class DiagnosticTests(unittest.TestCase):
    def test_complete_grid_keeps_structural_nulls_and_separates_zero_baseline_refusal(self):
        data = oracle_fixture()
        before = copy.deepcopy(data)
        rows, audit = diagnostic_rows(*data,layer_map(),pair_ids=("p0","p1"),candidate_layers=(24,),neighbor_layers=(27,),scoring_contract=CONTRACT)
        self.assertEqual(data, before)
        self.assertTrue(audit["passed"])
        self.assertEqual(len(rows), 10)
        summary = {r["layer"]:r for r in layer_summaries(rows)}
        self.assertEqual(summary[24]["utility_positive"], 0)
        self.assertEqual(summary[24]["categories"]["zero_baseline_refusal_restored"], 1)
        self.assertEqual(summary[24]["categories"]["changed_reply_all_endpoints_tied"], 1)
        self.assertEqual(summary[27]["pair_count"], 2)
        self.assertFalse(summary[27]["behaviorally_reachable"])
        self.assertTrue(all(r["execution_audit_ok"] for r in rows))
        self.assertNotIn("fixture private instruction", str(rows))

    def test_missing_pair_or_changed_label_is_not_silently_excluded(self):
        data = oracle_fixture()
        with self.assertRaises(RQ2ArtifactError):
            diagnostic_rows(*data,layer_map(),pair_ids=("p0","p1","p2"),candidate_layers=(24,),neighbor_layers=(27,),scoring_contract=CONTRACT)
        data[2][0]["strongreject_score"] = .5
        with self.assertRaises(RQ2ArtifactError):
            diagnostic_rows(*data,layer_map(),pair_ids=("p0","p1"),candidate_layers=(24,),neighbor_layers=(27,),scoring_contract=CONTRACT)

    def test_worsening_is_distinct_from_unchanged_or_floor_effect(self):
        self.assertEqual(outcome_category(-.375,0,-1,same_reply=False,baseline_score=.5),"utility_worsened")
        self.assertEqual(outcome_category(0,0,0,same_reply=True,baseline_score=.5),"identical_reply_utility_tie")
        with self.assertRaises(RQ2ArtifactError):
            outcome_category(0,1,0,same_reply=True,baseline_score=0)

    def test_report_is_byte_idempotent_and_checks_all_files_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {"pairs.csv":"a,b\r\n1,2\r\n","report.md":"fixture report\n"}
            write_report_files(root,files)
            before = {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()}
            write_report_files(root,files)
            self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in root.iterdir()})
            with self.assertRaises(RQ2ArtifactError):
                write_report_files(root,{"new.json":"{}", "report.md":"changed"})
            self.assertFalse((root/"new.json").exists())
            self.assertEqual((root/"report.md").read_text(),files["report.md"])


if __name__ == "__main__":
    unittest.main()
