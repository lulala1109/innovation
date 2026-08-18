import unittest

from core.behavior_decision import BehaviorDecision, coerce_behavior_decision, evaluate_behavior


class BehaviorDecisionTests(unittest.TestCase):
    def test_default_behavior_decision_preserves_legacy_target_match(self):
        decision = evaluate_behavior("target phrase", "Here is the TARGET PHRASE.")
        self.assertEqual(decision, BehaviorDecision(attack_success=True))

    def test_mapping_behavior_decision_is_normalized(self):
        decision = coerce_behavior_decision(
            {"jailbroken": True, "refusal": False, "score": 0.9, "provider": "test"}
        )
        self.assertTrue(decision.attack_success)
        self.assertFalse(decision.refusal)
        self.assertAlmostEqual(decision.score, 0.9)
        self.assertEqual(decision.details, {"provider": "test"})

    def test_behavior_decision_requires_explicit_success_key(self):
        with self.assertRaisesRegex(ValueError, "attack_success"):
            coerce_behavior_decision({"score": 0.5})


if __name__ == "__main__":
    unittest.main()
