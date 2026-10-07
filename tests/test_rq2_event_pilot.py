import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rq2.analysis import analyze_trial_set, RQ2AnalysisError
from rq2.artifacts import TrialKey, TrialRecord, atomic_jsonl, read_jsonl, canonical_sha256
from rq2.behavior import label_from_judge_result, make_response_record, judge_response_sidecar
from rq2.data import derive_behavior_events
from rq2.event_analysis import window_summary
from rq2.event_config import EVENT_CENTER, EVENT_NAME, EVENT_OFFSETS, EventPilotError
from rq2.event_data import build_event_index, imported_baseline, replay_audit, seed_phase_baselines
from rq2.event_pipeline import EventPilotPipeline
from rq2.experiment import PlannedTrial, build_oracle_plan, build_mechanism_plan, run_planned_trials
from rq2.pilot import evaluate_dev_pilot


def source_record(pair="dev_0", step=7, *, refused=False, zero=False):
    state = "scan:clean" if step is None else f"scan:{step}"
    key = TrialKey("a" * 64, pair, state, None, "trajectory_baseline")
    response = make_response_record(trial_id=key.trial_id, pair_id=pair, run_fingerprint="a" * 64,
                                    harmful_text="fixture instruction", response="fixture reply")
    trial = TrialRecord(key, "rq2_dev", step or 0, None, {"apply_count": 0}, {}, response["response_sha256"]).to_record()
    label = label_from_judge_result(key.trial_id, response["response_sha256"],
        {"refusal": int(refused), "convincingness": 1 if zero else 5, "specificity": 1 if zero else 5},
        judge_fingerprint=canonical_sha256({"protocol":"fixture"})).to_record()
    return trial, response, label


def population_fixture():
    trials, labels, scans, resolved = [], [], [], []
    ids = [f"dev_{i}" for i in range(20)]
    for i, pair in enumerate(ids):
        for step in [None, *range(101)]:
            center = 1 if i == 0 else 7
            trial, _, label = source_record(pair, step, refused=step != center, zero=i == 1)
            trials.append(trial); labels.append(label)
            scans.append({"pair_id": pair, "state_key": trial["state_key"],
                          "checkpoint_path": f"/fixture/{pair}/{step}.pt", "checkpoint_sha256": "c" * 64})
        resolved.append({"pair_id": pair, "rq2_role": "rq2_dev", "trajectory_path": f"/fixture/{pair}/index.json",
                         "clean_audio_path": f"/fixture/{pair}.wav", "clean_audio_sha256": "d" * 64,
                         "content_group": pair})
    events = derive_behavior_events(trials, labels, refusal_weakening_delta=.2)
    return ids, trials, labels, {"records": scans}, resolved, events


class EventPopulationTests(unittest.TestCase):
    def test_center_zero_score_kept_and_window_boundary_not_clamped(self):
        ids, trials, labels, scan, resolved, events = population_fixture()
        index, audit = build_event_index(scan, events, resolved, trials, labels, pair_ids=ids)
        self.assertEqual(audit["event_pair_count"], 20)
        self.assertEqual(audit["center_positive_score_pairs"], 19)
        self.assertEqual(audit["full_window_pair_count"], 19)
        missing = next(r for r in index["records"] if r["pair_id"] == "dev_0" and r["relative_step"] == -2)
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["step"])
        plan = build_oracle_plan(index, candidate_layers=(19,), neighbor_layers=(18,),
                                 coordinate="event", state_keys=(EVENT_CENTER,))
        self.assertEqual(len(plan), 20 * 5)
        self.assertEqual({p.state_key for p in plan}, {EVENT_CENTER})
        self.assertIn("dev_1", {p.pair_id for p in plan})
        self.assertEqual(build_oracle_plan(index, candidate_layers=(19,)), ())
        mechanism = build_mechanism_plan(index, candidate_layers=(19,), neighbor_layers=(18,),
            restoration_doses=(1,.5), reverse_doses=(1,.5), random_replicates=3, seed=42,
            coordinate="event", state_keys=(EVENT_CENTER,))
        self.assertEqual({p.state_key for p in mechanism}, {"clean", EVENT_CENTER})
        self.assertEqual(len([p for p in mechanism if p.intervention == "reverse_suppression"]), 40)

    def test_prior_unknown_makes_first_event_unresolved(self):
        ids, trials, labels, scan, resolved, _ = population_fixture()
        target = next(t for t in trials if t["pair_id"] == "dev_2" and t["state_key"] == "scan:2")
        labels = [l for l in labels if l["trial_id"] != target["trial_id"]]
        events = derive_behavior_events(trials, labels, refusal_weakening_delta=.2)
        _, audit = build_event_index(scan, events, resolved, trials, labels, pair_ids=ids)
        self.assertEqual(audit["raw_pair_count"], 20)
        self.assertEqual(audit["event_pair_count"], 19)
        row = next(r for r in audit["population_records"] if r["pair_id"] == "dev_2")
        self.assertEqual(row["event_status"], "unresolved_prior_missing_step")

    def test_window_profiles_share_population_and_average_replicates(self):
        rows = []
        for pair in ("a", "b", "c"):
            for offset in EVENT_OFFSETS:
                for intervention in ("r_direction", "sham"):
                    if pair == "c" and offset == -2:
                        continue
                    for _ in range(3):
                        rows.append({"pair_id": pair, "layer": 19, "state_key": f"event:{EVENT_NAME}:{offset:+d}",
                                     "intervention": intervention, "token_scope": "audio", "dose": 1,
                                     "utility_effect": .5 if intervention == "r_direction" and offset == 0 else 0})
        result = window_summary(rows, event_pair_ids=("a","b","c"), full_window_pair_ids=("a","b","c"),
                                 candidate_layers=(19,), replicates=20, confidence=.95, seed=42)
        for group in result["groups"]:
            self.assertEqual(group["valid_pair_ids"], ["a","b"])
            self.assertTrue(all(r["valid_pair_count"] == 2 for r in group["profiles"]))
        self.assertAlmostEqual(result["groups"][0]["paired_post_minus_pre"]["mean"], 1/6)
        self.assertFalse(result["causal_evidence"])


class BaselineImportTests(unittest.TestCase):
    def setUp(self):
        self.plan = PlannedTrial("dev_0", "rq2_dev", EVENT_CENTER, 7, 19, "sham", 1.0)
        self.source = source_record()
        self.imported = imported_baseline(self.plan, run_fingerprint="e" * 64,
            source_trial=self.source[0], source_response=self.source[1], source_label=self.source[2], source_binding="f" * 64)

    def test_import_has_new_identity_and_immutable_source_mapping(self):
        commit, label, mapping = self.imported
        self.assertNotEqual(commit["trial_id"], self.source[0]["trial_id"])
        self.assertEqual(mapping["source_trial_id"], self.source[0]["trial_id"])
        self.assertEqual(label["strongreject_score"], self.source[2]["strongreject_score"])
        self.assertEqual(label["judge_fingerprint"], self.source[2]["judge_fingerprint"])
        bad = copy.deepcopy(self.source[0]); bad["rq2_role"] = "rq2_causal_test"
        with self.assertRaises(EventPilotError):
            imported_baseline(self.plan, run_fingerprint="e"*64, source_trial=bad,
                              source_response=self.source[1], source_label=self.source[2], source_binding="f"*64)

    def test_resume_uses_imported_baseline_without_generation_or_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            seed_phase_baselines(directory, [self.imported])
            model = SimpleNamespace(generate_from_prepared_prompt=Mock(side_effect=AssertionError("baseline regenerated")))
            audit = SimpleNamespace(to_dict=lambda: {"apply_count": 0, "shift_l2": 0})
            runtime = SimpleNamespace(model=model, run_trial=Mock(return_value=SimpleNamespace(
                response="fixture reply", generation_audit=audit, diagnostic_audit=audit,
                generation_cache_skips=0, diagnostic={})))
            cache = SimpleNamespace(prepare=lambda _: SimpleNamespace(source_prompt={}, target_prompt={},
                harmful_text="fixture instruction", source_cache_key="fixture", input_provenance={}))
            args = dict(runtime=runtime, prompt_cache=cache, run_fingerprint="e"*64,
                        responses_path=directory/"responses.jsonl", trials_path=directory/"trials.jsonl", generation={"do_sample":False})
            run_planned_trials([self.plan], **args)
            seed_phase_baselines(directory, [self.imported])
            run_planned_trials([self.plan], **args)
            self.assertEqual(runtime.run_trial.call_count, 1)
            self.assertEqual(len(read_jsonl(directory/"commits.jsonl")), 2)
            evaluator = SimpleNamespace(evaluate=Mock(return_value={"refusal":0,"convincingness":5,"specificity":5}))
            judged = judge_response_sidecar(directory/"responses.jsonl", directory/"labels.jsonl",
                                            evaluator=evaluator, judge_config={"protocol":"fixture"})
            self.assertEqual(judged["reused"], 1)
            self.assertEqual(judged["judged"], 1)
            self.assertEqual(evaluator.evaluate.call_count, 1)
            trials = read_jsonl(directory/"trials.jsonl")
            self.assertTrue(replay_audit(trials, [self.plan], "e"*64)["passed"])
            trials[-1]["response_sha256"] = "0"*64
            self.assertFalse(replay_audit(trials, [self.plan], "e"*64)["passed"])


class PilotGateTests(unittest.TestCase):
    def summary(self, layer, state=EVENT_CENTER, count=20, sign=.7):
        return {"layer":layer,"state_key":state,"intervention":"full_state","dose":1,
                "token_scope":"audio","pair_count":count,"utility_effect_mean":.2,"sign_consistency":sign}

    def test_event_corroboration_cannot_come_from_other_offsets(self):
        kwargs = dict(intervention="full_state", dose=1, candidate_layers=(19,), neighbor_layers=(18,),
                      fixed_steps=(), event_state=EVENT_CENTER, minimum_effect=.05)
        self.assertTrue(evaluate_dev_pilot([self.summary(19),self.summary(18)],**kwargs)["qualified"])
        self.assertFalse(evaluate_dev_pilot([self.summary(19),self.summary(18,f"event:{EVENT_NAME}:+1")],**kwargs)["qualified"])

    def test_missing_labels_cannot_improve_sign_denominator(self):
        pipeline = object.__new__(EventPilotPipeline)
        pipeline.config = SimpleNamespace(output_root=Path('/fixture'),pilot={"candidate_layers":[19],"neighbor_layers":[18]})
        pipeline.population = lambda: {"event_pair_count":20}
        pipeline._phase_plan = lambda phase: []
        pipeline._run_fingerprint = lambda phase: "e"*64
        pipeline._scoring_audit = lambda directory: {"passed": True}
        for positive, expected in ((11,False),(12,True)):
            with patch('rq2.event_pipeline.read_pilot_csv',return_value=[self.summary(19,count=16,sign=positive/16),self.summary(18,count=16,sign=positive/16)]), patch('rq2.event_pipeline.read_jsonl',return_value=[]), patch('rq2.event_pipeline.replay_audit',return_value={"passed":True}):
                result = pipeline._pilot_decision("oracle_pilot","full_state",minimum_effect=.05)
                self.assertEqual(result["qualified"],expected)

    def test_replay_mismatch_overrides_a_positive_oracle_gate(self):
        pipeline = object.__new__(EventPilotPipeline)
        pipeline.config = SimpleNamespace(output_root=Path('/fixture'),pilot={"candidate_layers":[19],"neighbor_layers":[18]})
        pipeline.population = lambda: {"event_pair_count":20}
        pipeline._phase_plan = lambda phase: []
        pipeline._run_fingerprint = lambda phase: "e"*64
        pipeline._scoring_audit = lambda directory: {"passed": True}
        with patch('rq2.event_pipeline.read_pilot_csv',return_value=[self.summary(19),self.summary(18)]), patch('rq2.event_pipeline.read_jsonl',return_value=[]), patch('rq2.event_pipeline.replay_audit',return_value={"passed":False}):
            result = pipeline._pilot_decision("oracle_pilot","full_state",minimum_effect=.05)
            self.assertFalse(result["qualified"])
            self.assertEqual(result["qualifying_regions"],[])
            self.assertTrue(all(not r["eligible"] for r in result["regions"]))

    def test_forbidden_stages_block_before_source_or_gpu_access(self):
        pipeline = object.__new__(EventPilotPipeline)
        for stage in ("protocol_lock","formal_generate","subspace_generate","trajectory"):
            with self.assertRaises(EventPilotError):
                pipeline.run([stage])


class EventAnalysisTests(unittest.TestCase):
    def test_event_pilot_outputs_controls_without_formal_claims(self):
        trials, labels = [], []
        for pair in ("a","b"):
            source = source_record(pair)
            plan = PlannedTrial(pair,"rq2_dev",EVENT_CENTER,7,19,"full_state",1.0)
            baseline, label, _ = imported_baseline(plan,run_fingerprint="e"*64,
                source_trial=source[0],source_response=source[1],source_label=source[2],source_binding="f"*64)
            trials.append(baseline["trial"]); labels.append(label)
            key = plan.to_key("e"*64)
            trial = TrialRecord(key,"rq2_dev",7,baseline["trial_id"],{"shift_l2":1.0},{},"c"*64).to_record()
            trials.append(trial)
            labels.append(label_from_judge_result(key.trial_id,"c"*64,{"refusal":1,"convincingness":1,"specificity":1},judge_fingerprint="b"*64).to_record())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_jsonl(root/'trials.jsonl',trials); atomic_jsonl(root/'labels.jsonl',labels)
            result = analyze_trial_set(root/'trials.jsonl',root/'labels.jsonl',output_dir=root,
                pilot=True,event=True,event_pair_ids=("a","b"),candidate_layers=(19,),replicates=20)
            self.assertEqual(result["analysis_scope"],"dev_event_pilot")
            self.assertIsNone(result["causal_claim_gate"])
            self.assertEqual(result["formal_test_count"],0)
            self.assertTrue((root/'rq2_specificity_controls.csv').is_file())
            self.assertFalse((root/'rq2_event_formal_tests.csv').exists())
            with self.assertRaises(RQ2AnalysisError):
                analyze_trial_set(root/'trials.jsonl',root/'labels.jsonl',output_dir=root,pilot=True,event=True,event_pair_ids=("a",),replicates=20)


if __name__ == '__main__':
    unittest.main()
