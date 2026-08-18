"""Offline tests for canonical safety-state-aware layer-adaptive PGD."""

import unittest
from collections import OrderedDict

import torch

from attacks.layer_adaptive_pgd import (
    SUPPORTED_METHODS,
    LayerAdaptivePGDAttacker,
)
from core.safety_state import DualSafetyStateScorer
from models.base import AttackForwardOutput, BaseAudioModel


class ToyAudioModel(BaseAudioModel):
    """Small differentiable model with embedding + three transformer states."""

    def __init__(self, output_weight=1.0):
        self.output_weight = float(output_weight)
        self.forward_calls = 0
        self.compute_loss_calls = 0
        self.generate_calls = 0

    @property
    def sample_rate(self):
        return 16000

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return torch.float32

    def generate(
        self,
        wav,
        max_tokens=100,
        temperature=1.0,
        do_sample=False,
    ):
        self.generate_calls += 1
        return "target" if wav.mean().item() < -0.5 else "refused"

    def compute_loss(self, wav, target_text):
        self.compute_loss_calls += 1
        mean = wav.mean()
        return self.output_weight * (mean + 0.8).square()

    def compute_margin_loss(
        self, wav, target_text, kappa=5.0, early_weight=5.0
    ):
        return self.compute_loss(wav, target_text)

    @staticmethod
    def _layer(mean, coefficient, curvature, offset):
        positions = torch.linspace(
            -0.1, 0.1, 6, device=mean.device, dtype=mean.dtype
        )
        first = (
            coefficient * mean
            + curvature * mean.square()
            + offset * positions
        )
        second = (0.25 * mean + positions).expand_as(first)
        return torch.stack((first, second), dim=-1).unsqueeze(0)

    def forward_attack(
        self, wav, target_text, *, output_hidden_states=False
    ):
        self.forward_calls += 1
        mean = wav.mean()
        loss = self.output_weight * (mean + 0.8).square()
        hidden_states = None
        if output_hidden_states:
            embedding = self._layer(mean, 0.0, 0.0, 0.0)
            hidden_states = (
                embedding,
                self._layer(mean, 2.0, 0.0, 0.1),
                self._layer(mean, -1.0, 0.0, -0.1),
                self._layer(mean, 0.3, 2.0, 0.05),
            )
        return AttackForwardOutput(
            loss=loss,
            hidden_states=hidden_states,
            attention_mask=torch.ones(1, 6, dtype=torch.long),
            token_spans={"audio": (1, 3), "target": (4, 6)},
        )


def make_scorer():
    scorer = DualSafetyStateScorer(hidden_size=2, trainable=True)
    with torch.no_grad():
        scorer.refusal_probe.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        scorer.refusal_probe.linear.bias.zero_()
        scorer.harmfulness_probe.linear.weight.copy_(
            torch.tensor([[0.0, 1.0]])
        )
        scorer.harmfulness_probe.linear.bias.zero_()
    return scorer


REFERENCE = OrderedDict([(0, 0.92), (1, 0.88), (2, 0.84)])


class LayerAdaptivePGDTests(unittest.TestCase):
    def _attacker(self, method, model=None, scorer=None, **overrides):
        model = ToyAudioModel() if model is None else model
        if scorer is None and method != "standard":
            scorer = make_scorer()
        kwargs = dict(
            model=model,
            safety_scorer=scorer,
            method=method,
            eps=0.3,
            alpha=0.1,
            loss_type="ce",
            state_loss_weight=0.2,
            temperature=0.2,
            verbose=False,
        )
        if method == "fixed":
            kwargs["fixed_layer"] = 1
        elif method == "static_topk":
            kwargs["top_k"] = 2
            kwargs["static_layer_weights"] = {0: 0.1, 1: 0.9, 2: 0.8}
        kwargs.update(overrides)
        return LayerAdaptivePGDAttacker(**kwargs), model, scorer

    def test_all_six_methods_share_budget_steps_and_history_schema(self):
        self.assertEqual(
            SUPPORTED_METHODS,
            (
                "standard",
                "fixed",
                "uniform",
                "static_topk",
                "gradient_adaptive",
                "safety_state_adaptive",
            ),
        )
        wav = torch.full((1, 4), 0.2)
        results = {}
        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                attacker, model, scorer = self._attacker(method)
                attack_kwargs = {}
                if method != "standard":
                    attack_kwargs["reference_refusal"] = REFERENCE
                result = attacker.attack(
                    wav,
                    "target",
                    steps=3,
                    check_every=0,
                    early_stop=False,
                    **attack_kwargs,
                )
                results[method] = result

                self.assertEqual(result.steps_taken, 3)
                self.assertEqual(len(result.history["iterations"]), 4)
                self.assertEqual(
                    [item["state"] for item in result.history["iterations"]],
                    [0, 1, 2, 3],
                )
                self.assertLessEqual(
                    result.perturbation.abs().max().item(), 0.300001
                )
                self.assertLessEqual(result.adversarial_wav.max().item(), 1.0)
                self.assertGreaterEqual(result.adversarial_wav.min().item(), -1.0)
                config = result.history["config"]
                self.assertEqual(config["eps"], 0.3)
                self.assertEqual(config["alpha"], 0.1)
                self.assertEqual(config["steps"], 3)
                self.assertEqual(config["loss_type"], "ce")

                required = {
                    "state",
                    "loss",
                    "target_loss",
                    "output_loss",
                    "state_loss",
                    "layer_weights",
                    "safety_gaps",
                    "bottleneck_layer",
                    "linf",
                }
                for record in result.history["iterations"]:
                    self.assertTrue(required.issubset(record))

                if method == "standard":
                    self.assertEqual(model.compute_loss_calls, 4)
                    self.assertEqual(model.forward_calls, 0)
                    self.assertIsNone(
                        result.history["iterations"][0]["layer_weights"]
                    )
                else:
                    self.assertEqual(model.forward_calls, 4)
                    for parameter in scorer.parameters():
                        self.assertIsNone(parameter.grad)
                    for record in result.history["iterations"]:
                        self.assertAlmostEqual(
                            sum(record["layer_weights"].values()), 1.0, places=6
                        )

        fixed = results["fixed"].history["iterations"][0]["layer_weights"]
        self.assertEqual(fixed, {0: 0.0, 1: 1.0, 2: 0.0})
        uniform = results["uniform"].history["iterations"][0]["layer_weights"]
        for weight in uniform.values():
            self.assertAlmostEqual(weight, 1.0 / 3.0, places=6)
        static = results["static_topk"].history["iterations"][0]["layer_weights"]
        self.assertEqual(static, {0: 0.0, 1: 0.5, 2: 0.5})

    def test_safety_weights_are_recomputed_from_current_refusal_gap(self):
        attacker, _, _ = self._attacker("safety_state_adaptive")
        result = attacker.attack(
            torch.full((1, 4), 0.2),
            "target",
            steps=3,
            check_every=0,
            early_stop=False,
            reference_refusal=REFERENCE,
        )
        records = result.history["iterations"]
        first = records[0]["layer_weights"]
        last = records[-1]["layer_weights"]
        self.assertTrue(
            any(abs(first[layer] - last[layer]) > 1e-5 for layer in first)
        )
        for record in records:
            gaps = torch.tensor(list(record["safety_gaps"].values()))
            expected = torch.softmax(gaps / 0.2, dim=0)
            actual = torch.tensor(list(record["layer_weights"].values()))
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertNotEqual(
            records[0]["bottleneck_layer"], records[-1]["bottleneck_layer"]
        )

    def test_gradient_adaptive_weights_do_not_depend_on_safety_gap(self):
        reference_a = OrderedDict([(0, 0.99), (1, 0.50), (2, 0.10)])
        reference_b = OrderedDict([(0, 0.10), (1, 0.50), (2, 0.99)])
        runs = []
        for reference in (reference_a, reference_b):
            attacker, _, _ = self._attacker("gradient_adaptive")
            runs.append(
                attacker.attack(
                    torch.full((1, 4), 0.2),
                    "target",
                    steps=2,
                    check_every=0,
                    early_stop=False,
                    reference_refusal=reference,
                )
            )

        self.assertTrue(
            torch.allclose(runs[0].perturbation, runs[1].perturbation)
        )
        for record_a, record_b in zip(
            runs[0].history["iterations"], runs[1].history["iterations"]
        ):
            self.assertEqual(record_a["layer_weights"], record_b["layer_weights"])
            self.assertIsNotNone(record_a["gradient_scores"])
        self.assertNotEqual(
            runs[0].history["iterations"][0]["safety_gaps"],
            runs[1].history["iterations"][0]["safety_gaps"],
        )
        first = runs[0].history["iterations"][0]["layer_weights"]
        last = runs[0].history["iterations"][-1]["layer_weights"]
        self.assertTrue(
            any(abs(first[layer] - last[layer]) > 1e-5 for layer in first)
        )

    def test_state_loss_has_nonzero_input_gradient_and_not_probe_gradient(self):
        model = ToyAudioModel(output_weight=0.0)
        scorer = make_scorer()
        attacker, _, _ = self._attacker(
            "uniform",
            model=model,
            scorer=scorer,
            state_loss_weight=1.0,
        )
        result = attacker.attack(
            torch.full((1, 4), 0.2),
            "target",
            steps=1,
            check_every=0,
            early_stop=False,
            reference_refusal=REFERENCE,
        )
        self.assertGreater(result.perturbation.abs().max().item(), 0.0)
        self.assertNotEqual(
            result.history["iterations"][0]["state_loss"],
            result.history["iterations"][1]["state_loss"],
        )
        for parameter in scorer.parameters():
            self.assertIsNone(parameter.grad)

    def test_callbacks_use_update_aligned_checkpoint_states(self):
        attacker, _, _ = self._attacker("standard")
        snapshots = []

        def callback(state, adversarial_wav, delta, record):
            snapshots.append((state, delta.clone(), record["updates_completed"]))

        result = attacker.attack(
            torch.full((1, 4), 0.2),
            "target",
            steps=2,
            check_every=0,
            early_stop=False,
            state_callback=callback,
            checkpoint_steps={0, 2},
        )
        self.assertEqual([item[0] for item in snapshots], [0, 2])
        self.assertEqual([item[2] for item in snapshots], [0, 2])
        self.assertTrue(
            torch.equal(snapshots[0][1], torch.zeros_like(snapshots[0][1]))
        )
        self.assertGreater(snapshots[1][1].abs().max().item(), 0.0)
        self.assertEqual(len(result.history["iterations"]), 3)

    def test_nonstandard_methods_report_missing_state_inputs(self):
        model = ToyAudioModel()
        attacker = LayerAdaptivePGDAttacker(
            model,
            method="uniform",
            loss_type="ce",
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "requires safety_scorer"):
            attacker.attack(torch.zeros(1, 4), "target", steps=0)

        attacker = LayerAdaptivePGDAttacker(
            model,
            safety_scorer=make_scorer(),
            method="gradient_adaptive",
            loss_type="ce",
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "requires reference_refusal"):
            attacker.attack(torch.zeros(1, 4), "target", steps=0)

        with self.assertRaisesRegex(ValueError, "requires fixed_layer"):
            LayerAdaptivePGDAttacker(model, method="fixed", verbose=False)
        with self.assertRaisesRegex(ValueError, "static_topk requires"):
            LayerAdaptivePGDAttacker(model, method="static_topk", verbose=False)


if __name__ == "__main__":
    unittest.main()
