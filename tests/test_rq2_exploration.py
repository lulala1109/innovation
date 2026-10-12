"""Small CPU-only checks for the new boundary, grid, pairing and cache policy."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rq2.exploration_analysis import ENDPOINTS, paired_contrast, window_contrasts
from rq2.exploration_config import phase_plan
from rq2.exploration_pipeline import ExplorationJudge, ExplorationPipeline, FrozenEventPromptCache
from rq2.replication_judge import ContentJudge, ReplicationStop
from tests.test_rq2_replication import JUDGE, completion, response


def grid():
    ids = [f"dev_{i}" for i in range(20)]
    rows = [{"pair_id": p, "rq2_role": "rq2_dev", "coordinate": "event",
             "available": True, "step": 20 + o, "relative_step": o,
             "state_key": f"event:first_non_refusal_step:{o:+d}"}
            for p in ids for o in (-2, -1, 0, 1, 3)]
    return {"records": rows}, {"full_window_pair_ids": ids}


class ExplorationTests(unittest.TestCase):
    def test_fixed_grid_and_no_formal_or_unreachable_layer(self):
        index, population = grid()
        a, b = phase_plan(index, population, "mechanism"), phase_plan(index, population, "window")
        self.assertEqual((len(a), len(b)), (920, 1440))
        self.assertEqual(sum(p.target_kind == "clean" for p in a), 420)
        self.assertFalse(any(p.layer == 27 or p.rq2_role != "rq2_dev" for p in (*a, *b)))
        self.assertEqual({p.replicate for p in b if p.intervention == "random_direction_control"}, {0, 1, 2})
        self.assertFalse(any(p.state_key.endswith(":+0") for p in b))

    def test_random_repeats_average_inside_pair(self):
        row = lambda p, n: {"pair_id": p, **{f: n for f in ENDPOINTS}}
        result = paired_contrast([row("a", 1), row("b", .5)],
                                 [row("a", 0), row("a", .5), row("a", 1), row("b", .2)])
        self.assertAlmostEqual(result[0]["utility_effect"], .5)
        self.assertAlmostEqual(result[1]["utility_effect"], .3)
        with self.assertRaises(ValueError):
            paired_contrast([row("a", 1)], [row("b", 1)])

    def test_window_subset_does_not_filter_center_population(self):
        index, population = grid()
        population["full_window_pair_ids"] = population["full_window_pair_ids"][:18]
        self.assertEqual(len(phase_plan(index, population, "mechanism")), 920)
        self.assertEqual(len(phase_plan(index, population, "window")), 18 * 72)

    def test_prompt_cache_only_exposes_frozen_selected_checkpoints(self):
        pair = SimpleNamespace(pair_id="dev")
        index = {"records": [{"pair_id": "dev", "available": True, "step": 14,
                  "checkpoint_path": "/fixture/step_14.pt", "checkpoint_sha256": "a" * 64}]}
        cache = FrozenEventPromptCache(object(), (pair,), index)
        self.assertEqual(cache._checkpoint_paths(pair), {14: Path("/fixture/step_14.pt")})
        with self.assertRaises(ReplicationStop):
            cache._checkpoint_paths(SimpleNamespace(pair_id="formal"))

    def test_window_population_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            window_contrasts([], [], ["missing"])

    def test_forbidden_stage_and_missing_ack_before_any_write(self):
        pipe = object.__new__(ExplorationPipeline)
        with self.assertRaises(ReplicationStop):
            pipe.run(("formal_generate",), acknowledge_cost=True, accept_exploratory=True)
        with self.assertRaises(ReplicationStop):
            pipe.run(("mechanism_generate",))

    def test_receipt_backed_old_content_reused_without_payment_and_new_identity_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budget = {"api_limit_cny": 30, "http_attempt_limit": 100,
                      "input_cny_per_million": 2, "output_cny_per_million": 9}
            old_transport = Mock(return_value=completion())
            old = ContentJudge(root / "old", JUDGE, {"old": True}, budget, transport=old_transport)
            first = old.label(response())
            old_transport.reset_mock()
            transport = Mock(return_value={**completion(), "model": "any-new-serving-name"})
            judge = ExplorationJudge(root / "new", JUDGE, {"new": True}, budget,
                       {"cost_micro_cny": 380, "http_attempts": 1}, source_judge=old, transport=transport)
            imported = judge.label(response("new-id"))
            self.assertEqual(imported["strongreject_score"], first["strongreject_score"])
            self.assertEqual(imported["judge_fingerprint"], judge.fingerprint)
            old_transport.assert_not_called()
            transport.assert_not_called()
            judge.label(response("different", reply="another fixture reply"))
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(judge.summary()["cumulative_http_attempts"], 2)
            judge.label(response("same-again", reply="another fixture reply"))
            self.assertEqual(transport.call_count, 1)

    def test_unknown_usage_keeps_reservation_and_never_auto_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            budget = {"api_limit_cny": 30, "http_attempt_limit": 100,
                      "input_cny_per_million": 2, "output_cny_per_million": 9}
            old = ContentJudge(root / "old", JUDGE, {}, budget, transport=Mock())
            transport = Mock(return_value={"model": "fixture"})
            judge = ExplorationJudge(root / "new", JUDGE, {"new": True}, budget,
                       {"cost_micro_cny": 0, "http_attempts": 0}, source_judge=old, transport=transport)
            with self.assertRaises(ReplicationStop):
                judge.label(response())
            with self.assertRaises(ReplicationStop):
                judge.label(response())
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(judge.summary()["unresolved_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
