"""CPU tensors, actual toy-model hooks, synthetic receipts; no live GPU/API work."""
from collections import OrderedDict, Counter
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rq2.artifacts import atomic_json, atomic_text, canonical_sha256, file_sha256, read_jsonl
from rq2.behavior import make_response_record
from rq2.experiment import PlannedTrial, PreparedTrial
from rq2.interventions import InterventionError, InterventionSpec, apply_intervention
from rq2.orthogonal_config import CENTER, DESIGN, STAGES, plan_trials
from rq2.orthogonal_interventions import (
    OrthogonalSpec, OrthogonalPrefillTransform, OrthogonalRuntime, apply_orthogonal,
    check_matched_audits, numeric_metrics,
)
from rq2.orthogonal_pipeline import CalibrationJudge, OrthogonalPipeline, recover_journal
from rq2.pipeline import RQ2Pipeline
from rq2.replication_config import BUDGET
from rq2.replication_judge import ContentJudge, ReplicationStop
from tests.test_rq2_replication import JUDGE, completion, response


def grid():
    pairs = [f"p{i:02d}" for i in range(20)]
    index = {"records": [{"pair_id": p, "rq2_role": "rq2_dev", "state_key": CENTER,
                           "available": True, "step": i+2, "checkpoint_path": f"/fixture/{p}.pt",
                           "checkpoint_sha256": "c"*64} for i, p in enumerate(pairs)]}
    return pairs, index


def tensors(dtype=torch.float32):
    event = torch.zeros(1, 5, 4, dtype=dtype)
    clean = event.clone()
    clean[:, 1:4, 0] = 4
    clean[:, 1:4, 1] = 1
    return event, {"source_activation": clean, "source_token_selection": (1, 2, 3),
                   "refusal_direction": torch.tensor([1., 0, 0, 0]),
                   "harmfulness_direction": torch.tensor([.8, .6, 0, 0])}


class OperatorTests(unittest.TestCase):
    def test_grid_population_seed_and_unsupported_conditions(self):
        pairs, index = grid()
        plan = plan_trials(index, pairs)
        self.assertEqual(len(plan), 240)
        self.assertEqual(len({p.to_key("a"*64).trial_id for p in plan}), 240)
        self.assertEqual(Counter(p.intervention for p in plan),
                         {"r_direction": 40, "h_orthogonal_control": 40,
                          "random_orthogonal_control": 120, "sham": 40})
        self.assertEqual([p.to_spec() for p in plan], [p.to_spec() for p in plan_trials(index, pairs)])
        for kind, layer, dose, rep in (("reverse_suppression", 19, 1, 0), ("r_direction", 26, 1, 0),
                                      ("r_direction", 19, .5, 0), ("random_orthogonal_control", 19, 1, 3)):
            with self.assertRaises(InterventionError):
                OrthogonalSpec(kind, layer, dose=dose, replicate=rep)
        with self.assertRaises(InterventionError):
            InterventionSpec("h_orthogonal_control", 19)
        with self.assertRaises(ReplicationStop):
            plan_trials(index, pairs[:-1])
        index["records"][0]["rq2_role"] = "rq2_causal_test"
        with self.assertRaises(ReplicationStop):
            plan_trials(index, pairs)

    def test_R_and_sham_preserve_frozen_algebra(self):
        x, kwargs = tensors()
        for kind in ("r_direction", "sham"):
            y, a = apply_orthogonal(x, (1, 2, 3), OrthogonalSpec(kind, 19), **kwargs)
            old, b = apply_intervention(x, (1, 2, 3), InterventionSpec(kind, 19), **kwargs)
            self.assertTrue(torch.equal(y, old))
            self.assertEqual(a, b)

    def test_orthogonal_controls_preserve_R_and_match_norms_in_each_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x, kwargs = tensors(dtype)
            records = []
            for kind, rep in (("r_direction", 0), ("h_orthogonal_control", 0),
                              *(("random_orthogonal_control", i) for i in range(3)), ("sham", 0)):
                spec = OrthogonalSpec(kind, 19, seed=42, replicate=rep)
                y, a = apply_orthogonal(x, (1, 2, 3), spec, **kwargs)
                self.assertTrue(torch.equal(x[:, (0, 4)], y[:, (0, 4)]))
                self.assertTrue(torch.equal(x, torch.zeros_like(x)))
                if "orthogonal" in kind:
                    self.assertEqual(a.projection_delta, 0.)
                    self.assertAlmostEqual(a.mean_shift_l2, 4., delta=.03)
                records.append({"pair_id": "p", "layer": 19, "intervention": kind,
                                "replicate": rep, "audit": a.to_dict()})
            self.assertTrue(check_matched_audits(records)["passed"])

    def test_degenerate_direction_and_zero_gap_fail_without_fallback(self):
        x, kwargs = tensors()
        kwargs["harmfulness_direction"] = kwargs["refusal_direction"].clone()
        with self.assertRaisesRegex(InterventionError, "degenerate"):
            apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("h_orthogonal_control", 19), **kwargs)
        kwargs["source_activation"] = x.clone()
        with self.assertRaisesRegex(InterventionError, "near-zero"):
            apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("r_direction", 19), **kwargs)

    def test_H_orthogonal_sign_points_toward_clean(self):
        x, kwargs = tensors()
        kwargs["source_activation"][:, 1:4, 1] = -1
        y, _ = apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("h_orthogonal_control", 19), **kwargs)
        self.assertTrue(torch.all(y[:, 1:4, 1] < 0))

    def test_same_seed_repeats_and_different_seed_changes_random_direction(self):
        x, kwargs = tensors()
        fn = lambda seed: apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("random_orthogonal_control", 19, seed=seed), **kwargs)[0]
        self.assertTrue(torch.equal(fn(42), fn(42)))
        self.assertFalse(torch.equal(fn(42), fn(43)))

    def test_numeric_tampering_and_wrong_position_fail(self):
        x, kwargs = tensors()
        _, a = apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("r_direction", 19), **kwargs)
        for name, value in (("apply_count", 2), ("mean_shift_l2", float("nan")), ("shift_l2", -1)):
            with self.assertRaises(InterventionError):
                numeric_metrics({**a.to_dict(), name: value})
        with self.assertRaisesRegex(InterventionError, "selections"):
            apply_orthogonal(x, (1, 2, 3), OrthogonalSpec("r_direction", 19),
                             target_reference_selection=(0,), **kwargs)

    def test_transform_skips_decode_and_rejects_second_prefill(self):
        x, kwargs = tensors()
        transform = OrthogonalPrefillTransform(expected_sequence_length=5, token_selection=(1, 2, 3),
                    spec=OrthogonalSpec("h_orthogonal_control", 19), **kwargs)
        transform(x)
        transform(x[:, :1])
        self.assertEqual(transform.assert_applied_once().apply_count, 1)
        self.assertEqual(transform.cache_skips, 1)
        with self.assertRaises(InterventionError):
            transform(x)


class ToyModel:
    def __init__(self):
        self.layers = OrderedDict((i, torch.nn.Identity()) for i in (19, 24))
        self.generations = 0

    def get_transformer_layer_modules(self):
        return self.layers

    def forward_prepared_prompt(self, prompt, **kwargs):
        hidden = prompt["inputs_embeds"]
        for layer in self.layers.values():
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)

    def generate_from_prepared_prompt(self, prompt, **kwargs):
        self.generations += 1
        hidden = self.forward_prepared_prompt(prompt).logits
        self.forward_prepared_prompt({"inputs_embeds": prompt["inputs_embeds"][:, :1]})
        return "fixture refusal" if float(hidden[..., 0].mean()) > 1 else "fixture reply"


class ToyScorer(torch.nn.Module):
    def forward(self, pooled):
        return SimpleNamespace(refusal={k: torch.sigmoid(v[..., 0]) for k, v in pooled.items()},
                               harmfulness={k: torch.sigmoid(v[..., 1]) for k, v in pooled.items()})


def bundle():
    _, kwargs = tensors()
    directions = {"refusal": {k: kwargs["refusal_direction"] for k in (19, 24)},
                  "harmfulness": {k: kwargs["harmfulness_direction"] for k in (19, 24)}}
    return SimpleNamespace(hidden_sizes=OrderedDict((i, 4) for i in (19, 24)), unit_directions=directions,
                           centers={s: {k: torch.zeros(4) for k in (19, 24)} for s in directions},
                           refusal_sigma={19: 1., 24: 1.}, scorer=ToyScorer())


class ToyPromptCache:
    def __init__(self, model, pairs, index):
        self.pair = pairs[0]
        self.index = index

    def prepare(self, trial):
        x, kwargs = tensors()
        def prompt(t):
            return {"inputs_embeds": t, "attention_mask": torch.ones(1, 5), "token_spans": {"audio": (1, 4)}}
        p = self.pair
        center = next(r for r in self.index["records"] if r["pair_id"] == p.pair_id)
        return PreparedTrial(prompt(kwargs["source_activation"]), prompt(x), p.harmful_text, p.pair_id,
                             {"pair_id": p.pair_id, "content_group": p.content_group,
                              "clean_audio_sha256": p.clean_audio_sha256, "target": {"kind": "trajectory",
                              "step": center["step"], "checkpoint_path": center["checkpoint_path"],
                              "checkpoint_sha256": center["checkpoint_sha256"]}})


class PipelineTests(unittest.TestCase):
    def test_actual_runtime_hooks_and_exception_cleanup(self):
        ids, index = grid()
        p = SimpleNamespace(pair_id=ids[0], content_group="g", clean_audio_sha256="b"*64, harmful_text="fixture")
        prepared = ToyPromptCache(None, [p], index).prepare(plan_trials(index, ids)[0])
        model = ToyModel()
        runtime = OrthogonalRuntime(model, bundle())
        for kind in ("r_direction", "h_orthogonal_control", "random_orthogonal_control", "sham"):
            result = runtime.run_trial(prepared.source_prompt, prepared.target_prompt, OrthogonalSpec(kind, 19))
            self.assertEqual(result.generation_audit.apply_count, 1)
            self.assertEqual(result.generation_cache_skips, 1)
            self.assertEqual(result.generation_audit, result.diagnostic_audit)
            self.assertTrue(all(not layer._forward_hooks for layer in model.layers.values()))
        with patch.object(model, "generate_from_prepared_prompt", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                runtime.run_trial(prepared.source_prompt, prepared.target_prompt, OrthogonalSpec("r_direction", 19))
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers.values()))

    def test_invalid_stage_and_missing_ack_stop_before_io(self):
        pipe = object.__new__(OrthogonalPipeline)
        for stages in (("reverse_generate",), ("formal",), ("numeric_preflight",)):
            with self.assertRaises(ReplicationStop):
                pipe.run(stages)

    def test_truncated_journal_is_repaired_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            atomic_text(path, '{"ok": 1}\n{"partial":')
            self.assertEqual(recover_journal(path), [{"ok": 1}])
            self.assertEqual(read_jsonl(path), [{"ok": 1}])
            atomic_text(path, '{"ok": 1}')
            recover_journal(path)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_full_cpu_grid_score_report_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=False):
            root = Path(tmp)
            ids, index = grid()
            source = SimpleNamespace(output_root=root / "source", source=None)
            atomic_text(source.output_root / ".run.lock", "")
            pipe = object.__new__(OrthogonalPipeline)
            config = SimpleNamespace(output_root=root / "out", fingerprint="a"*64, name="toy",
                                     model={"max_tokens": 100, "temperature": 1., "do_sample": False}, judge=JUDGE)
            models = []
            attempted = {"count": 0, "interrupted": False}
            def model_factory(_):
                model = ToyModel()
                generate = model.generate_from_prepared_prompt
                def interrupt_once(*args, **kwargs):
                    attempted["count"] += 1
                    if attempted["count"] == 21 and not attempted["interrupted"]:
                        attempted["interrupted"] = True
                        raise RuntimeError("simulated interrupted generation")
                    return generate(*args, **kwargs)
                model.generate_from_prepared_prompt = interrupt_once
                models.append(model)
                return model
            RQ2Pipeline.__init__(pipe, config, model_factory=model_factory)
            contract = {"fixture": True}
            pipe.spec = SimpleNamespace(fingerprint="a"*64, source=source,
                                        raw={"pair_ids": ids, "scoring_contract": contract})
            pipe.index, pipe.population = index, {"expected_dev_pair_ids": ids}
            pipe.verified = {"scope": "toy", "formal_allowed": False}
            pipe.verify = Mock(return_value=pipe.verified)
            pipe.bundle = Mock(return_value=bundle())
            pairs = tuple(SimpleNamespace(pair_id=p, role="rq2_dev", content_group=p,
                          clean_audio_sha256="b"*64, harmful_text="fixture instruction " + p) for p in ids)
            pipe.pairs = lambda: pairs
            pipe.source_baselines = {p: response(p)["response_sha256"] for p in ids}
            pipe.source_baselines = {p: {"response_sha256": h} for p, h in pipe.source_baselines.items()}
            def payload(body):
                refused = "fixture refusal" in body["messages"][0]["content"]
                return completion(json.dumps({"overall_reasoning": "fixture", "refusal": int(refused),
                                              "convincingness": 1 if refused else 5,
                                              "specificity": 1 if refused else 5}))
            transport = Mock(side_effect=payload)
            pipe._judge = lambda: CalibrationJudge(root / "out/scoring", JUDGE, contract, BUDGET,
                {"http_attempts": 0, "cost_micro_cny": 0}, readers=(), uncertain_history={}, transport=transport)
            with patch("rq2.orthogonal_pipeline.FrozenEventPromptCache", ToyPromptCache):
                pipe.run(("prepare",))
                pipe.run(("numeric_preflight",), acknowledge_cost=True, accept_exploratory=True)
                self.assertEqual(sum(m.generations for m in models), 0)
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                    pipe.run(STAGES[2:], acknowledge_cost=True, accept_exploratory=True)
                self.assertEqual(transport.call_count, 0)
                result = pipe.run(STAGES[2:], acknowledge_cost=True, accept_exploratory=True)
                self.assertEqual(sum(m.generations for m in models), 260)
                self.assertEqual(attempted["count"], 261)
                self.assertTrue(all(s["status"] == "complete" for s in result["stages"]))
                analysis = json.loads(pipe.stage_path("calibration_analyze").read_text())
                self.assertEqual(analysis["coverage"]["analyzed_effect_count"], 240)
                self.assertEqual(len(analysis["paired_control_contrasts"]), 6)
                for c in analysis["paired_control_contrasts"]:
                    self.assertEqual(c["endpoints"]["utility_effect"]["pair_count"], 20)
                count = transport.call_count
                pipe.run(STAGES, acknowledge_cost=True, accept_exploratory=True)
                self.assertEqual(sum(m.generations for m in models), 260)
                self.assertEqual(transport.call_count, count)


def scoring_contract(scope):
    return {"scope": scope, "implementation_sha256": {k: "a"*64 for k in
            ("evaluation/behavior.py", "core/llm_backend.py", "rq2/behavior.py")}}


class JudgeTests(unittest.TestCase):
    def test_cache_reuse_is_read_only_and_history_is_carried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = ContentJudge(root / "old", JUDGE, scoring_contract("old"), BUDGET,
                               transport=Mock(return_value=completion()))
            old.label(response())
            before = {p: file_sha256(p) for p in old.root.rglob("*") if p.is_file()}
            transport = Mock(side_effect=AssertionError("new API forbidden"))
            new = CalibrationJudge(root / "new", JUDGE, scoring_contract("new"), BUDGET,
                     {"cost_micro_cny": 6047301, "http_attempts": 2654}, readers=(old,),
                     uncertain_history={"historical_uncertain_reserved_cny": .014016}, transport=transport)
            label = new.label(response("new-id"))
            self.assertEqual(label["judge_fingerprint"], new.fingerprint)
            self.assertEqual(new.summary()["cumulative_accounted_cny_at_peak_rates"], 6.047301)
            self.assertEqual(new.summary()["cumulative_http_attempts"], 2654)
            self.assertEqual(before, {p: file_sha256(p) for p in old.root.rglob("*") if p.is_file()})
            transport.assert_not_called()

    def test_uncertain_new_request_never_retries_or_uses_another_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport = Mock(side_effect=TimeoutError("fixture"))
            judge = CalibrationJudge(Path(tmp), JUDGE, scoring_contract("new"), BUDGET,
                {"cost_micro_cny": 6047301, "http_attempts": 2654}, readers=(), uncertain_history={}, transport=transport)
            for _ in range(2):
                with self.assertRaises(ReplicationStop):
                    judge.label(response())
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(judge.summary()["unresolved_attempts"], 1)

    def test_historical_attempt_without_cache_is_not_repaid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = ContentJudge(root / "old", JUDGE, scoring_contract("old"), BUDGET,
                               transport=Mock(side_effect=TimeoutError("fixture")))
            with self.assertRaises(ReplicationStop):
                old.label(response())
            transport = Mock()
            new = CalibrationJudge(root / "new", JUDGE, scoring_contract("new"), BUDGET,
                {"cost_micro_cny": 10000, "http_attempts": 1}, readers=(old,), uncertain_history={}, transport=transport)
            with self.assertRaisesRegex(ReplicationStop, "lacks a valid cache"):
                new.label(response())
            transport.assert_not_called()

    def test_conflicting_paid_sources_stop_without_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readers = []
            for i, refused in enumerate((True, False)):
                payload = completion(json.dumps({"overall_reasoning": "fixture", "refusal": int(refused),
                    "convincingness": 1 if refused else 5, "specificity": 1 if refused else 5}))
                reader = ContentJudge(root / str(i), JUDGE, scoring_contract(str(i)), BUDGET,
                                      transport=Mock(return_value=payload))
                reader.label(response())
                readers.append(reader)
            transport = Mock()
            judge = CalibrationJudge(root / "new", JUDGE, scoring_contract("new"), BUDGET,
                {"cost_micro_cny": 760, "http_attempts": 2}, readers=readers, uncertain_history={}, transport=transport)
            with self.assertRaisesRegex(ReplicationStop, "labels conflict"):
                judge.label(response())
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
