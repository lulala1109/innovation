"""Offline tests for separate harmfulness/refusal states and bottlenecks."""

import unittest
from collections import OrderedDict

import torch

from core.safety_state import (
    DualSafetyScores,
    DualSafetyStateScorer,
    compute_layer_weights,
    compute_safety_gaps,
    differentiable_state_loss,
    dynamic_bottleneck,
    safety_gap_attack_loss,
)


class SafetyStateTests(unittest.TestCase):
    def test_dual_scorer_keeps_probes_and_directions_separate(self):
        scorer = DualSafetyStateScorer(hidden_size=2)
        with torch.no_grad():
            scorer.harmfulness_probe.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            scorer.harmfulness_probe.linear.bias.zero_()
            scorer.refusal_probe.linear.weight.copy_(torch.tensor([[0.0, 2.0]]))
            scorer.refusal_probe.linear.bias.zero_()

        hidden = torch.tensor([[2.0, -1.0]], requires_grad=True)
        scores = scorer(hidden)
        self.assertIsInstance(scores, DualSafetyScores)
        self.assertFalse(hasattr(scores, "score"))
        self.assertGreater(scores.harmfulness.item(), 0.5)
        self.assertLess(scores.refusal.item(), 0.5)
        self.assertTrue(
            torch.equal(
                scorer.harmfulness_probe.direction,
                torch.tensor([1.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(scorer.refusal_probe.direction, torch.tensor([0.0, 2.0]))
        )

        projections = scorer.direction_scores(hidden)
        self.assertAlmostEqual(projections.harmfulness.item(), 2.0)
        self.assertAlmostEqual(projections.refusal.item(), -1.0)

    def test_layerwise_dual_probes_are_independent_per_layer(self):
        scorer = DualSafetyStateScorer(hidden_size={2: 3, 5: 2})
        self.assertIsNot(
            scorer.harmfulness_probe.probe_for(2),
            scorer.refusal_probe.probe_for(2),
        )
        states = OrderedDict(
            [(2, torch.ones(1, 3)), (5, torch.ones(1, 2))]
        )
        scores = scorer(states)
        self.assertEqual(tuple(scores.harmfulness), (2, 5))
        self.assertEqual(scores.refusal[2].shape, (1,))

    def test_gap_weights_and_bottleneck_match_documented_formulas(self):
        reference = OrderedDict(
            [(1, torch.tensor(0.9)), (4, torch.tensor(0.8)), (7, torch.tensor(0.7))]
        )
        current = OrderedDict(
            [
                (1, torch.tensor(0.8)),
                (4, torch.tensor(0.2)),
                (7, torch.tensor(0.5)),
            ]
        )
        gaps = compute_safety_gaps(reference, current)
        self.assertAlmostEqual(gaps[4].item(), 0.6)

        weights = compute_layer_weights(gaps, temperature=0.5)
        self.assertAlmostEqual(sum(value.item() for value in weights.values()), 1.0)
        self.assertGreater(weights[4], weights[7])
        bottleneck = dynamic_bottleneck(gaps)
        self.assertEqual(bottleneck.layer, 4)
        self.assertAlmostEqual(bottleneck.gap.item(), 0.6)

        later = compute_safety_gaps(
            reference,
            OrderedDict(
                [(1, torch.tensor(0.1)), (4, torch.tensor(0.7)), (7, torch.tensor(0.6))]
            ),
        )
        self.assertEqual(dynamic_bottleneck(later).layer, 1)

    def test_dynamic_weights_and_state_loss_remain_differentiable(self):
        reference = OrderedDict(
            [(0, torch.tensor([0.9])), (1, torch.tensor([0.8]))]
        )
        refusal_0 = torch.tensor([0.7], requires_grad=True)
        refusal_1 = torch.tensor([0.3], requires_grad=True)
        current = OrderedDict([(0, refusal_0), (1, refusal_1)])
        gaps = compute_safety_gaps(reference, current)
        weights = compute_layer_weights(gaps, temperature=0.25)

        loss = differentiable_state_loss(current, weights)
        loss.backward(retain_graph=True)
        self.assertIsNotNone(refusal_0.grad)
        self.assertIsNotNone(refusal_1.grad)
        self.assertGreater(refusal_0.grad.abs().item(), 0.0)
        self.assertGreater(refusal_1.grad.abs().item(), 0.0)

        refusal_0.grad.zero_()
        refusal_1.grad.zero_()
        attack_loss = safety_gap_attack_loss(gaps, weights)
        attack_loss.backward()
        self.assertGreater(refusal_0.grad.abs().item(), 0.0)
        self.assertGreater(refusal_1.grad.abs().item(), 0.0)

    def test_full_dual_state_is_rejected_where_one_state_is_required(self):
        dual = DualSafetyScores(
            harmfulness=torch.tensor([0.9, 0.8]),
            refusal=torch.tensor([0.7, 0.6]),
        )
        with self.assertRaisesRegex(TypeError, "explicitly selected state"):
            compute_safety_gaps(dual, dual)
        with self.assertRaisesRegex(TypeError, "explicitly selected state"):
            compute_layer_weights(dual)

    def test_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            compute_layer_weights(torch.tensor([0.1, 0.2]), temperature=0.0)


if __name__ == "__main__":
    unittest.main()
