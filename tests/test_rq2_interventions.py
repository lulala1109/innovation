import math

import pytest
import torch

from rq2.interventions import InterventionError, InterventionSpec, apply_intervention


def test_direction_restore_and_position_total_norm_matching():
    target = torch.zeros(1, 4, 3)
    source = torch.zeros(1, 4, 3)
    source[:, :2, 0] = 2.0
    spec = InterventionSpec("r_direction", layer=0, dose=1.0, token_scope="position_control")
    patched, audit = apply_intervention(
        target, (2, 3), spec,
        source_activation=source,
        source_token_selection=(0, 1),
        target_reference_selection=(0, 1),
        application_scale=math.sqrt(2 / 2),
        refusal_direction=torch.tensor([1.0, 0.0, 0.0]),
    )
    assert torch.allclose(patched[0, 2:, 0], torch.tensor([2.0, 2.0]))
    assert audit.token_indices == (2, 3)
    assert audit.shift_l2 == pytest.approx(math.sqrt(8.0))


def test_norm_matched_direction_controls():
    activation = torch.zeros(1, 2, 3)
    source = torch.ones(1, 2, 3)
    kwargs = dict(
        source_activation=source,
        refusal_direction=torch.tensor([1.0, 0.0, 0.0]),
        harmfulness_direction=torch.tensor([0.0, 1.0, 0.0]),
    )
    _, r = apply_intervention(activation, (0, 1), InterventionSpec("r_direction", 0), **kwargs)
    _, h = apply_intervention(activation, (0, 1), InterventionSpec("h_direction_control", 0), **kwargs)
    _, random = apply_intervention(
        activation, (0, 1), InterventionSpec("random_direction_control", 0, seed=7), **kwargs
    )
    assert h.shift_l2 == pytest.approx(r.shift_l2)
    assert random.shift_l2 == pytest.approx(r.shift_l2)


def test_subspace_and_reverse_controls():
    activation = torch.zeros(1, 2, 3)
    source = torch.ones(1, 2, 3)
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    patched, _ = apply_intervention(
        activation, (0, 1), InterventionSpec("subspace_restoration", 0),
        source_activation=source, subspace=basis,
    )
    assert torch.allclose(patched[..., :2], torch.ones(1, 2, 2))
    assert torch.allclose(patched[..., 2], torch.zeros(1, 2))
    with pytest.raises(InterventionError, match="refusal_sigma"):
        apply_intervention(
            activation, (0,), InterventionSpec("reverse_suppression", 0),
            refusal_direction=torch.tensor([1.0, 0.0, 0.0]), refusal_sigma=0.0,
        )


def test_subspace_controls_match_actual_projected_shift_norm():
    activation = torch.zeros(1, 2, 3)
    source = torch.tensor([[[3.0, 4.0, 9.0], [3.0, 4.0, 9.0]]])
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    kwargs = dict(
        source_activation=source,
        subspace=basis,
        harmfulness_direction=torch.tensor([0.0, 0.0, 1.0]),
    )
    _, restored = apply_intervention(
        activation, (0, 1), InterventionSpec("subspace_restoration", 0), **kwargs
    )
    _, h_control = apply_intervention(
        activation, (0, 1), InterventionSpec("subspace_h_control", 0), **kwargs
    )
    _, random_control = apply_intervention(
        activation,
        (0, 1),
        InterventionSpec("subspace_random_control", 0, seed=7),
        **kwargs,
    )
    assert restored.mean_shift_l2 == pytest.approx(5.0)
    assert h_control.shift_l2 == pytest.approx(restored.shift_l2)
    assert random_control.shift_l2 == pytest.approx(restored.shift_l2)


def test_reverse_controls_are_norm_matched():
    activation = torch.zeros(1, 2, 3)
    kwargs = dict(
        refusal_direction=torch.tensor([1.0, 0.0, 0.0]),
        harmfulness_direction=torch.tensor([0.0, 1.0, 0.0]),
        refusal_sigma=2.5,
    )
    audits = []
    for kind in ("reverse_suppression", "reverse_h_control", "reverse_random_control"):
        _, audit = apply_intervention(
            activation, (0, 1), InterventionSpec(kind, 0, dose=0.5, seed=4), **kwargs
        )
        audits.append(audit)
    assert audits[1].shift_l2 == pytest.approx(audits[0].shift_l2)
    assert audits[2].shift_l2 == pytest.approx(audits[0].shift_l2)


def test_intervention_replicate_requires_integer():
    with pytest.raises(InterventionError, match="replicate"):
        InterventionSpec("sham", 0, replicate="1")
