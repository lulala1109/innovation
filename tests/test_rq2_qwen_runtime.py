from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from core.activations import collect_hidden_states
from rq2.interventions import InterventionSpec, PrefillOnlyTransform
from rq2.qwen_runtime import QwenCausalRuntime, QwenRuntimeError


class ToyLayer(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, hidden):
        return hidden + self.delta


class ToyQwen:
    def __init__(self):
        self.layers = OrderedDict((i, ToyLayer(float(i + 1))) for i in range(3))

    def get_transformer_layer_modules(self):
        return self.layers

    def forward_prepared_prompt(self, prompt, *, output_hidden_states=False, use_cache=False):
        hidden = prompt["inputs_embeds"]
        states = [hidden]
        for layer in self.layers.values():
            hidden = layer(hidden)
            states.append(hidden)
        return SimpleNamespace(logits=hidden, hidden_states=tuple(states) if output_hidden_states else None)

    def generate_from_prepared_prompt(self, prompt, **kwargs):
        self.forward_prepared_prompt(prompt)
        return "deterministic"


def test_prefill_transform_skips_cache_token_and_hits_once():
    transform = PrefillOnlyTransform(
        expected_sequence_length=4,
        token_selection=(1, 2),
        spec=InterventionSpec("sham", 0),
    )
    assert transform(torch.zeros(1, 1, 3)).shape[-2] == 1
    transform(torch.zeros(1, 4, 3))
    assert transform.assert_applied_once().apply_count == 1
    assert transform.cache_skips == 1


def test_layer_map_checks_embedding_offset():
    bundle = SimpleNamespace(hidden_sizes=OrderedDict((i, 4) for i in range(3)))
    runtime = QwenCausalRuntime(ToyQwen(), bundle)
    prompt = {
        "inputs_embeds": torch.zeros(1, 5, 4),
        "attention_mask": torch.ones(1, 5),
        "token_spans": {"audio": (1, 4)},
    }
    result = runtime.map_layers(prompt)
    assert result["passed"] is True
    assert [row["hidden_state_index"] for row in result["layers"]] == [1, 2, 3]


def test_layer_map_checks_independent_rq1_replay_pooling():
    bundle = SimpleNamespace(hidden_sizes=OrderedDict((i, 4) for i in range(3)))
    runtime = QwenCausalRuntime(ToyQwen(), bundle)
    prompt = {
        "inputs_embeds": torch.zeros(1, 5, 4),
        "attention_mask": torch.ones(1, 5),
        "token_spans": {"audio": (1, 4)},
    }
    replay = {
        0: torch.full((4,), 1.0),
        1: torch.full((4,), 3.0),
        2: torch.full((4,), 6.0),
    }
    result = runtime.map_layers(prompt, rq1_pooled=replay)
    assert result["rq1_replay_checked"] is True
    assert all(row["rq1_replay_passed"] for row in result["layers"])
    replay[1] = torch.zeros(4)
    with pytest.raises(QwenRuntimeError, match="RQ1 replay"):
        runtime.map_layers(prompt, rq1_pooled=replay)


def test_layer_map_uses_final_norm_and_rq1_bfloat16_pooling():
    class NormToyQwen(ToyQwen):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4, elementwise_affine=False)

        def get_final_decoder_norm_module(self):
            return self.norm

        def forward_prepared_prompt(self, prompt, *, output_hidden_states=False, use_cache=False):
            hidden = prompt["inputs_embeds"]
            states = []
            for layer in self.layers.values():
                states.append(hidden)
                hidden = layer(hidden)
            hidden = self.norm(hidden)
            states.append(hidden)
            return SimpleNamespace(logits=hidden, hidden_states=tuple(states))

    torch.manual_seed(42)
    model = NormToyQwen()
    prompt = {
        "inputs_embeds": torch.randn(1, 17, 4).to(torch.bfloat16),
        "attention_mask": torch.ones(1, 17, dtype=torch.long),
        "token_spans": {"audio": (1, 16)},
    }
    states = model.forward_prepared_prompt(prompt, output_hidden_states=True).hidden_states
    replay = collect_hidden_states(
        states,
        layers=(0, 1, 2),
        pooling="mean",
        attention_mask=prompt["attention_mask"],
        token_selection=(1, 16),
        sequence_has_embedding=True,
    )
    runtime = QwenCausalRuntime(model, SimpleNamespace(
        hidden_sizes=OrderedDict((i, 4) for i in range(3))
    ))
    result = runtime.map_layers(prompt, rq1_pooled=replay)
    assert result["passed"] is True
    assert result["layers"][-1]["activation_site"] == "final_output_norm"
    assert result["layers"][-1]["audio_only_prefill_behaviorally_reachable"] is False
    assert all(row["rq1_replay_passed"] for row in result["layers"])


def _runtime_bundle():
    directions = {
        state: OrderedDict((i, torch.ones(4)) for i in range(3))
        for state in ("refusal", "harmfulness")
    }
    return SimpleNamespace(
        hidden_sizes=OrderedDict((i, 4) for i in range(3)),
        unit_directions=directions,
        refusal_sigma=OrderedDict((i, 1.0) for i in range(3)),
    )


def test_generation_exception_removes_patch_hook():
    class FailingToy(ToyQwen):
        def generate_from_prepared_prompt(self, prompt, **kwargs):
            self.forward_prepared_prompt(prompt)
            raise RuntimeError("generation failed")

    model = FailingToy()
    runtime = QwenCausalRuntime(model, _runtime_bundle())
    prompt = {
        "inputs_embeds": torch.zeros(1, 5, 4),
        "attention_mask": torch.ones(1, 5),
        "token_spans": {"audio": (1, 4)},
    }
    with pytest.raises(RuntimeError, match="generation failed"):
        runtime.generate_trial(
            prompt, prompt, InterventionSpec("sham", 1), max_tokens=2
        )
    assert all(not layer._forward_hooks for layer in model.layers.values())
    runtime.capture(prompt)


def test_audio_token_count_mismatch_is_blocked():
    runtime = QwenCausalRuntime(ToyQwen(), _runtime_bundle())
    source = {
        "inputs_embeds": torch.zeros(1, 5, 4),
        "attention_mask": torch.ones(1, 5),
        "token_spans": {"audio": (1, 3)},
    }
    target = {
        "inputs_embeds": torch.zeros(1, 5, 4),
        "attention_mask": torch.ones(1, 5),
        "token_spans": {"audio": (1, 4)},
    }
    with pytest.raises(ValueError, match="differ in length"):
        runtime.generate_trial(source, target, InterventionSpec("sham", 1))
