"""CPU-only tests for canonical PGD state checkpoint callbacks."""

import unittest

import torch

from attacks.pgd import PGDAttacker
from models.base import BaseAudioModel


class _ToyAudioModel(BaseAudioModel):
    """Small differentiable adapter that never loads an external checkpoint."""

    def __init__(self):
        self.generate_calls = 0

    @property
    def sample_rate(self) -> int:
        return 16_000

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        self.generate_calls += 1
        return "ordinary response"

    def compute_loss(self, wav: torch.Tensor, target_text: str) -> torch.Tensor:
        # Minimization gives a stable negative sign-gradient update from the
        # positive test waveform.
        return wav.square().mean()


class PGDStateCallbackTests(unittest.TestCase):
    def _attacker(self):
        model = _ToyAudioModel()
        attacker = PGDAttacker(
            model,
            eps=0.5,
            alpha=0.05,
            loss_type="ce",
            verbose=False,
        )
        return model, attacker

    def test_explicit_checkpoints_include_initialized_and_final_states(self):
        _, attacker = self._attacker()
        waveform = torch.full((1, 4), 0.4)
        snapshots = []

        def callback(step, adversarial_wav, delta, metadata):
            snapshots.append(
                (
                    step,
                    adversarial_wav.clone(),
                    delta.clone(),
                    dict(metadata),
                )
            )

        result = attacker.attack(
            waveform,
            "target text",
            steps=3,
            check_every=0,
            early_stop=False,
            state_callback=callback,
            checkpoint_steps={0, 3},
        )

        self.assertEqual([snapshot[0] for snapshot in snapshots], [0, 3])
        self.assertTrue(torch.equal(snapshots[0][2], torch.zeros_like(waveform)))
        self.assertTrue(
            torch.allclose(snapshots[3 // 3][2], torch.full_like(waveform, -0.15))
        )
        for step, adversarial, delta, metadata in snapshots:
            self.assertEqual(metadata["step"], step)
            self.assertEqual(metadata["updates_completed"], step)
            self.assertFalse(adversarial.requires_grad)
            self.assertFalse(delta.requires_grad)
            self.assertTrue(torch.allclose(adversarial, waveform + delta))
        self.assertEqual(result.steps_taken, 3)
        self.assertEqual(len(result.history["iterations"]), 4)

    def test_callback_without_filter_observes_all_n_plus_one_states(self):
        _, attacker = self._attacker()
        observed = []
        attacker.attack(
            torch.full((1, 2), 0.25),
            "target text",
            steps=2,
            check_every=0,
            early_stop=False,
            state_callback=lambda step, *_: observed.append(step),
        )
        self.assertEqual(observed, [0, 1, 2])

    def test_zero_update_run_emits_state_zero_once_as_final_state(self):
        _, attacker = self._attacker()
        observed = []
        result = attacker.attack(
            torch.full((1, 2), 0.25),
            "target text",
            steps=0,
            check_every=0,
            early_stop=False,
            state_callback=lambda step, *_: observed.append(step),
            checkpoint_steps={0},
        )
        self.assertEqual(observed, [0])
        self.assertEqual(result.steps_taken, 0)
        self.assertEqual(
            [item["step"] for item in result.history["iterations"]], [0]
        )

    def test_out_of_range_checkpoint_steps_are_rejected_before_generation(self):
        for invalid_steps in ({-1}, {3}, {-2, 7}):
            with self.subTest(checkpoint_steps=invalid_steps):
                model, attacker = self._attacker()
                with self.assertRaisesRegex(
                    ValueError, "checkpoint_steps must be within"
                ):
                    attacker.attack(
                        torch.full((1, 2), 0.25),
                        "target text",
                        steps=2,
                        state_callback=lambda *_: None,
                        checkpoint_steps=invalid_steps,
                    )
                self.assertEqual(model.generate_calls, 0)


if __name__ == "__main__":
    unittest.main()
