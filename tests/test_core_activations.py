"""Offline tests for model-agnostic hidden-state utilities."""

import unittest
from collections import OrderedDict

import torch
from torch import nn

from core.activations import (
    ActivationPatch,
    ActivationPatcher,
    ForwardActivationCollector,
    HiddenStateCollector,
    collect_hidden_states,
    pool_tokens,
    resolve_layer_indices,
)


class ActivationUtilityTests(unittest.TestCase):
    def test_resolve_configurable_layer_selection(self):
        self.assertEqual(resolve_layer_indices(6, "all"), (0, 1, 2, 3, 4, 5))
        self.assertEqual(resolve_layer_indices(6, "1,3,-1"), (1, 3, 5))
        self.assertEqual(resolve_layer_indices(8, "2:7:2"), (2, 4, 6))
        self.assertEqual(resolve_layer_indices(4, slice(1, None)), (1, 2, 3))
        with self.assertRaises(IndexError):
            resolve_layer_indices(3, (3,))

    def test_masked_token_pooling_is_differentiable(self):
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [50.0, 60.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ],
            requires_grad=True,
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        pooled = pool_tokens(hidden, pooling="mean", attention_mask=mask)
        expected = torch.tensor([[2.0, 3.0], [7.0, 8.0]])
        self.assertTrue(torch.allclose(pooled, expected))
        pool_tokens(
            hidden,
            pooling="last",
            attention_mask=mask,
            token_selection=(0, 3),
        ).sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertEqual(hidden.grad[0, 2].abs().sum().item(), 0.0)

    def test_collect_huggingface_hidden_states_excludes_embedding(self):
        states = tuple(
            torch.full((1, 2, 3), float(index)) for index in range(4)
        )
        collected = collect_hidden_states(
            states,
            layers=(0, 2),
            pooling="last",
            sequence_has_embedding=True,
        )
        self.assertEqual(tuple(collected), (0, 2))
        self.assertTrue(torch.equal(collected[0], torch.ones(1, 3)))
        self.assertTrue(torch.equal(collected[2], torch.full((1, 3), 3.0)))

        configured = HiddenStateCollector(
            layers=(1,), pooling="mean", sequence_has_embedding=True
        )
        self.assertEqual(configured(states)[1].shape, (1, 3))

    def test_activation_patch_is_consumed_by_remaining_forward(self):
        model = nn.Sequential(
            OrderedDict(
                [
                    ("first", nn.Linear(2, 2, bias=False)),
                    ("second", nn.Linear(2, 2, bias=False)),
                ]
            )
        )
        with torch.no_grad():
            model.first.weight.copy_(torch.eye(2))
            model.second.weight.copy_(2 * torch.eye(2))
        inputs = torch.tensor([[4.0, -3.0]])

        with ActivationPatcher(
            {"first": lambda activation: torch.ones_like(activation)},
            model=model,
        ):
            patched_output = model(inputs)

        self.assertTrue(torch.equal(patched_output, torch.tensor([[2.0, 2.0]])))
        self.assertTrue(torch.equal(model(inputs), torch.tensor([[8.0, -6.0]])))

    def test_token_selective_patch_preserves_other_tokens_and_gradients(self):
        module = nn.Identity()
        inputs = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        replacement = torch.tensor([[[20.0, 30.0]]], requires_grad=True)
        with ActivationPatcher(
            {
                module: ActivationPatch(
                    replacement=replacement,
                    token_selection=(1, 2),
                )
            }
        ):
            output = module(inputs)
        self.assertTrue(torch.equal(output[:, 0], inputs[:, 0]))
        self.assertTrue(torch.equal(output[:, 1], replacement.detach()[:, 0]))
        self.assertTrue(torch.equal(output[:, 2], inputs[:, 2]))
        output.sum().backward()
        self.assertTrue(torch.equal(replacement.grad, torch.ones_like(replacement)))

    def test_forward_collector_can_keep_graph_connected_states(self):
        model = nn.Sequential(nn.Identity(), nn.Linear(3, 1, bias=False))
        inputs = torch.randn(2, 4, 3, requires_grad=True)
        with ForwardActivationCollector(
            {0: model[0]}, pooling="mean", detach=False
        ) as collector:
            output = model(inputs)
        state_loss = collector.activations[0].square().mean()
        (output.mean() + state_loss).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
