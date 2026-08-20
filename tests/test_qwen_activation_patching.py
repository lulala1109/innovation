"""Offline tests for the Qwen activation-patching adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.qwen_activation_patching import (
    align_token_selections,
    build_case_specs,
    coordinate_qwen_activation_patching,
    run_qwen_activation_patching,
)
from models.qwen import QwenModel


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.prefill_inputs: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-2] > 1:
            self.prefill_inputs.append(hidden.detach().clone())
        return hidden * self.value


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((_Scale(1.0), _Scale(2.0), _Scale(1.0)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _FakeThinker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeTextModel()
        self.audio_tower = SimpleNamespace(
            layers=nn.ModuleList((nn.Identity(), nn.Identity()))
        )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **_kwargs,
    ):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.model(inputs_embeds))

    def generate(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        del attention_mask
        prefill = self.model(inputs_embeds)
        # Simulate a cached decode call. A static full-prompt replacement would
        # fail here because this activation has only one token.
        self.model(torch.zeros_like(inputs_embeds[:, :1]))
        token = 1 if float(prefill.mean()) > 2.0 else 0
        return torch.tensor([[token]], dtype=torch.long)


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "target payload" if int(ids[0]) == 1 else "I cannot assist"


class _FakeQwen(QwenModel):
    def __init__(self) -> None:
        # Intentionally do not call QwenModel.__init__: no Transformers package
        # or checkpoint should be needed by these tests.
        self._device = "cpu"
        self._dtype = torch.float32
        self.thinker = _FakeThinker()
        self.tokenizer = _FakeTokenizer()
        self.eos_token_id = 2

    def prepare_audio_prompt(self, wav: torch.Tensor) -> dict:
        value = float(wav.reshape(-1)[0])
        embeds = torch.full((1, 4, 2), value, dtype=torch.float32)
        return {
            "inputs_embeds": embeds,
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "token_spans": {"audio": (1, 3)},
        }


class QwenActivationPatchingTests(unittest.TestCase):
    def test_stable_mapping_targets_text_decoder_not_audio_tower(self) -> None:
        model = _FakeQwen()
        mapping = model.transformer_layer_modules

        self.assertEqual(tuple(mapping), (0, 1, 2))
        self.assertIs(mapping[0], model.thinker.model.layers[0])
        self.assertIsNot(mapping[0], model.thinker.audio_tower.layers[0])
        selected = model.get_transformer_layer_modules((-1, 0))
        self.assertEqual(tuple(selected), (2, 0))

    def test_source_collection_is_unpooled(self) -> None:
        model = _FakeQwen()
        prompt = model.prepare_audio_prompt(torch.tensor([5.0]))
        activations = model.collect_layer_activations(
            layers=(0, 1),
            prepared_prompt=prompt,
            detach=True,
        )

        self.assertEqual(tuple(activations), (0, 1))
        self.assertEqual(tuple(activations[0].shape), (1, 4, 2))
        self.assertTrue(torch.equal(activations[0], torch.full((1, 4, 2), 5.0)))
        self.assertTrue(torch.equal(activations[1], torch.full((1, 4, 2), 10.0)))
        self.assertFalse(activations[0].requires_grad)

    def test_prefill_patch_changes_downstream_generation_and_skips_cache(self) -> None:
        model = _FakeQwen()
        result = coordinate_qwen_activation_patching(
            model=model,
            source_wav=torch.tensor([5.0]),
            jailbreak_wav=torch.tensor([0.0]),
            pair_id="pair-alpha",
            critical_layers=(0,),
            random_control_count=1,
            random_layers=(2,),
            token_selection="audio",
            output_selector=0,
            target_text="target payload",
        )

        self.assertEqual(result["baseline"]["text"], "I cannot assist")
        self.assertTrue(result["baseline"]["behavior"]["refused"])
        critical = result["trials"][0]
        self.assertEqual(critical["condition"], "critical")
        self.assertEqual(critical["response"]["text"], "target payload")
        self.assertTrue(critical["response"]["behavior"]["target_matched"])
        self.assertEqual(critical["patch"]["prefill_apply_count"], 1)
        # Layer 1 is downstream from the layer-0 hook and must receive the
        # replaced audio-token values during at least one prefill.
        downstream = model.thinker.model.layers[1].prefill_inputs
        self.assertTrue(
            any(float(value[:, 1:3].mean()) == 5.0 for value in downstream)
        )

    def test_injected_one_argument_score_function_is_supported(self) -> None:
        result = coordinate_qwen_activation_patching(
            model=_FakeQwen(),
            source_wav=torch.tensor([5.0]),
            jailbreak_wav=torch.tensor([0.0]),
            pair_id="pair-alpha",
            critical_layers=(0,),
            random_control_count=0,
            score_fn=lambda text: {"custom_success": "target" in text},
        )

        self.assertEqual(result["baseline_scores"], {"custom_success": 0.0})
        self.assertEqual(
            result["trials"][0]["response"]["scores"],
            {"custom_success": 1.0},
        )

    def test_token_span_lengths_must_align(self) -> None:
        source = {
            "inputs_embeds": torch.zeros(1, 5, 2),
            "token_spans": {"audio": (1, 4)},
        }
        jailbreak = {
            "inputs_embeds": torch.zeros(1, 4, 2),
            "token_spans": {"audio": (1, 3)},
        }
        with self.assertRaisesRegex(ValueError, "differ in length"):
            align_token_selections(source, jailbreak, "audio")

    def test_case_pair_mismatch_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.wav").touch()
            case = root / "case"
            case.mkdir()
            (case / "adversarial.wav").touch()
            (case / "run.json").write_text(
                json.dumps(
                    {
                        "pair_id": "pair-beta",
                        "artifacts": {"adversarial_audio": "adversarial.wav"},
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "pair_id": "pair-alpha",
                            "clean_audio_path": "source.wav",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            loaded = False

            def model_factory(**_kwargs):
                nonlocal loaded
                loaded = True
                return _FakeQwen()

            with self.assertRaisesRegex(ValueError, "absent from manifest"):
                run_qwen_activation_patching(
                    manifest,
                    root / "result.json",
                    critical_layers=(0,),
                    case_paths=(case,),
                    model_factory=model_factory,
                )
            self.assertFalse(loaded)

    def test_trajectory_case_runs_and_writes_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.wav").touch()
            case = root / "case"
            trajectory = case / "trajectory"
            trajectory.mkdir(parents=True)
            checkpoint = trajectory / "step_000003.pt"
            checkpoint.touch()
            (trajectory / "index.json").write_text(
                json.dumps(
                    {
                        "format": "safety-state-trajectory",
                        "version": 1,
                        "checkpoints": [
                            {
                                "step": 3,
                                "path": "trajectory/step_000003.pt",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (case / "run.json").write_text(
                json.dumps(
                    {
                        "pair_id": "pair-alpha",
                        "budget": {
                            "experiment_config": {
                                "target_text": "target payload",
                            }
                        },
                        "artifacts": {"adversarial_audio": "adversarial.wav"},
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "pair_id": "pair-alpha",
                            "clean_audio_path": "source.wav",
                            "harmful_text": "harmful",
                            "stratum": "test",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            specs = build_case_specs(manifest, case_paths=(trajectory,))
            self.assertEqual(specs[0].trajectory_step, 3)
            self.assertEqual(specs[0].target_text, "target payload")

            output = root / "patching.json"
            result = run_qwen_activation_patching(
                manifest,
                output,
                critical_layers=(0,),
                case_paths=(trajectory,),
                random_control_count=1,
                random_layers=(1,),
                model_factory=lambda **_kwargs: _FakeQwen(),
                audio_loader=lambda _path, target_sr: torch.tensor([5.0]),
                checkpoint_loader=lambda _path: torch.tensor([0.0]),
            )

            self.assertEqual(result["counts"], {"total": 1, "completed": 1, "failed": 0})
            self.assertTrue(output.is_file())
            self.assertEqual(list(root.glob(".patching.json.*.tmp")), [])
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["cases"][0]["pair_id"], "pair-alpha")
            self.assertEqual(
                written["cases"][0]["artifacts"]["jailbreak_kind"],
                "trajectory",
            )


if __name__ == "__main__":
    unittest.main()
