"""Offline differentiability and evaluation-adapter tests for audio EoT."""

import unittest

import torch

from transforms.eot import (
    CodecEvaluationAdapter,
    EoTCompose,
    EvaluationBackendRequiredError,
    NoiseTransform,
    NonDifferentiableTransformError,
    add_noise,
    apply_rir,
    codec_roundtrip,
    expectation_over_transforms,
    playback_capture,
    resample_waveform,
)


class EoTTransformTests(unittest.TestCase):
    def test_noise_resample_and_rir_are_differentiable(self):
        waveform = torch.ones(1, 8, requires_grad=True)
        noisy = add_noise(
            waveform, snr_db=20.0, noise=torch.ones_like(waveform)
        )
        noise_power = (noisy - waveform).square().mean()
        signal_power = waveform.square().mean()
        measured = 10 * torch.log10(signal_power / noise_power)
        self.assertAlmostEqual(measured.item(), 20.0, places=4)
        resampled = resample_waveform(noisy, 8_000, 16_000)
        self.assertEqual(resampled.shape[-1], 16)
        convolved = apply_rir(resampled, torch.tensor([1.0]))
        self.assertTrue(torch.allclose(convolved, resampled))
        convolved.sum().backward()
        self.assertGreater(waveform.grad.abs().sum().item(), 0.0)

    def test_composition_eot_supports_simple_and_stochastic_callables(self):
        waveform = torch.ones(1, 6, requires_grad=True)
        compose = EoTCompose(
            [
                lambda value: value * 2,
                NoiseTransform(30.0, noise=torch.ones(1, 6)),
            ]
        )
        loss = expectation_over_transforms(
            waveform, lambda value: value.square().mean(),
            compose, samples=3, seed=12,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(waveform.grad.abs().sum().item(), 0.0)

    def test_codec_and_playback_require_real_evaluation_backends(self):
        waveform = torch.ones(1, 4)
        with self.assertRaises(EvaluationBackendRequiredError):
            codec_roundtrip(waveform, 16_000)
        with self.assertRaises(EvaluationBackendRequiredError):
            playback_capture(waveform, 16_000)

        calls = []

        def backend(value, sample_rate, **options):
            calls.append((sample_rate, options))
            return value * 0.5, sample_rate

        adapter = CodecEvaluationAdapter(backend, "opus", bitrate="32k")
        output, sample_rate = adapter.evaluate(waveform, 16_000)
        self.assertTrue(torch.equal(output, waveform * 0.5))
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(calls[0][1]["codec"], "opus")
        with self.assertRaises(NonDifferentiableTransformError):
            adapter(waveform)
        with self.assertRaises(NonDifferentiableTransformError):
            EoTCompose([adapter])


if __name__ == "__main__":
    unittest.main()
