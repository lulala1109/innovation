"""Independent audio-prefill orthogonal controls; frozen RQ2 operators are unchanged."""
from dataclasses import dataclass, replace
import math

import torch

from rq2.experiment import PlannedTrial
from rq2.interventions import (
    InterventionAudit, InterventionError, InterventionSpec, PrefillOnlyTransform,
    _indices, _selected_mean, _unit, apply_intervention,
)
from rq2.qwen_runtime import QwenCausalRuntime

ORTHOGONAL_KINDS = ("h_orthogonal_control", "random_orthogonal_control")
KINDS = ("r_direction", *ORTHOGONAL_KINDS, "sham")
TOLERANCES = {"minimum_gap": 1e-6, "minimum_orthogonal_norm": 1e-6,
              "projection_relative": .02, "norm_relative": .05, "absolute": 1e-4}


@dataclass(frozen=True)
class OrthogonalSpec(InterventionSpec):
    def __post_init__(self):
        if self.kind not in KINDS or self.layer not in (19, 24) or self.dose != 1 or self.token_scope != "audio":
            raise InterventionError("orthogonal calibration supports only L19/L24, audio, dose 1")
        mapped = "h_direction_control" if self.kind in ORTHOGONAL_KINDS else self.kind
        InterventionSpec(mapped, self.layer, self.dose, self.token_scope, self.seed, self.replicate)
        if (self.kind == "random_orthogonal_control" and self.replicate not in range(3)
                or self.kind != "random_orthogonal_control" and self.replicate != 0):
            raise InterventionError("invalid orthogonal-control replicate")


@dataclass(frozen=True)
class OrthogonalTrial(PlannedTrial):
    def to_spec(self):
        # Keep the existing pair/state/layer/replicate seed derivation.
        old = PlannedTrial.to_spec(replace(self, intervention="sham"))
        return OrthogonalSpec(self.intervention, old.layer, old.dose, old.token_scope, old.seed, old.replicate)


def orthogonal_unit(vector, r_unit):
    unit = _unit(vector, r_unit.numel(), "control direction").to(r_unit.device)
    residual = unit - torch.dot(unit, r_unit) * r_unit
    norm = torch.linalg.vector_norm(residual)
    if not torch.isfinite(norm) or float(norm) < TOLERANCES["minimum_orthogonal_norm"]:
        raise InterventionError("control direction is degenerate after R projection removal")
    return residual / norm


def apply_orthogonal(activation, token_selection, spec, **kwargs):
    if not isinstance(spec, OrthogonalSpec):
        raise InterventionError("an independent OrthogonalSpec is required")
    if kwargs.get("application_scale", 1.) != 1 or kwargs.get("subspace") is not None:
        raise InterventionError("position rescaling and subspace are outside calibration A")
    indices = _indices(activation.shape[-2], token_selection, activation.device)
    reference = kwargs.get("target_reference_selection")
    if reference is not None and not torch.equal(indices, _indices(activation.shape[-2], reference, activation.device)):
        raise InterventionError("target and reference audio selections must match")
    if spec.kind not in ORTHOGONAL_KINDS:
        result, audit = apply_intervention(activation, token_selection, spec, **kwargs)
    else:
        source = kwargs.get("source_activation")
        if not isinstance(source, torch.Tensor) or source.ndim != activation.ndim or source.shape[-1] != activation.shape[-1]:
            raise InterventionError("clean and event activation ranks/widths differ")
        source_selection = kwargs.get("source_token_selection")
        source_indices = _indices(source.shape[-2], token_selection if source_selection is None else source_selection, source.device)
        if len(source_indices) != len(indices):
            raise InterventionError("clean and event audio token counts differ")
        source_mean = source.index_select(-2, source_indices).to(device=activation.device, dtype=activation.dtype).float()
        source_mean = source_mean.mean(dim=tuple(range(source.ndim - 2)) + (-2,))
        current_mean = _selected_mean(activation, indices)
        r = _unit(kwargs["refusal_direction"], activation.shape[-1], "R").to(activation.device)
        gap = torch.dot(source_mean - current_mean, r)
        if spec.kind == "h_orthogonal_control":
            direction = orthogonal_unit(kwargs["harmfulness_direction"], r)
            sign = -1. if float(torch.dot(source_mean-current_mean, direction)) < 0 else 1.
            direction = direction * sign
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(spec.seed + 1_000_003 * spec.replicate)
            direction = orthogonal_unit(torch.randn(r.numel(), generator=generator), r)
        shift = abs(float(gap)) * direction
        result = activation.clone()
        result.index_copy_(-2, indices, activation.index_select(-2, indices) + shift.to(activation.dtype))
        difference = (result.float() - activation.float()).detach()
        after_mean = _selected_mean(result, indices)
        before, after = float(torch.dot(current_mean, r)), float(torch.dot(after_mean, r))
        audit = InterventionAudit(spec.kind, spec.layer, spec.dose, spec.token_scope, len(indices),
                                  tuple(indices.cpu().tolist()), 1,
                                  float(torch.linalg.vector_norm(difference)), float(difference.abs().max()),
                                  float(torch.linalg.vector_norm(after_mean-current_mean)), before, after,
                                  float(torch.dot(source_mean, r)), after-before)
    numeric_metrics(audit.to_dict())
    return result, audit


def numeric_metrics(audit):
    kind = audit.get("kind")
    if kind not in KINDS or audit.get("apply_count") != 1 or audit.get("token_scope") != "audio":
        raise InterventionError("invalid prefill application identity")
    indices = audit.get("token_indices")
    if not indices or len(indices) != audit.get("token_count") or list(indices) != sorted(set(indices)):
        raise InterventionError("invalid audio token coverage")
    fields = ("projection_before", "projection_after", "projection_reference", "projection_delta",
              "mean_shift_l2", "shift_l2", "shift_linf")
    if any(isinstance(audit.get(k), bool) or not isinstance(audit.get(k), (int, float))
           or not math.isfinite(audit[k]) for k in fields):
        raise InterventionError("missing/nonfinite manipulation audit")
    gap = audit["projection_reference"] - audit["projection_before"]
    delta, norm = audit["projection_delta"], audit["mean_shift_l2"]
    if abs(audit["projection_after"] - audit["projection_before"] - delta) > 1e-6:
        raise InterventionError("projection delta disagrees with before/after")
    if any(audit[k] < 0 for k in ("mean_shift_l2", "shift_l2", "shift_linf")):
        raise InterventionError("negative manipulation norm")
    if kind == "sham":
        if delta != 0 or norm != 0 or audit["shift_l2"] != 0 or audit["shift_linf"] != 0:
            raise InterventionError("sham is not an identity")
    else:
        if abs(gap) < TOLERANCES["minimum_gap"]:
            raise InterventionError("near-zero R gap; stop without dropping or replacing the pair")
        error = abs(delta-gap) if kind == "r_direction" else abs(delta)
        if error > TOLERANCES["absolute"] + TOLERANCES["projection_relative"] * abs(gap):
            raise InterventionError("R target/orthogonality tolerance failed; no behavioral tuning")
        if abs(norm-abs(gap)) > TOLERANCES["absolute"] + TOLERANCES["norm_relative"] * abs(gap):
            raise InterventionError("realized mean displacement does not match the planned R gap")
    return {"R_gap": gap, "R_delta": delta, "mean_shift_l2": norm,
            "projection_error": abs(delta-gap) if kind == "r_direction" else abs(delta)}


def check_matched_audits(records):
    """Check matched realized norms, not just theoretically equal shift vectors."""
    lookup = {(r["pair_id"], r["layer"], r["intervention"], r["replicate"]): r for r in records}
    if len(lookup) != len(records):
        raise InterventionError("duplicate numeric audit condition")
    for row in records:
        a = row["audit"]
        numeric_metrics(a)
        ref = lookup[row["pair_id"], row["layer"], "r_direction", 0]["audit"]
        if a["token_indices"] != ref["token_indices"]:
            raise InterventionError("matched controls use different token positions")
        for k in ("projection_before", "projection_reference"):
            if not math.isclose(a[k], ref[k], rel_tol=1e-6, abs_tol=1e-4):
                raise InterventionError("matched controls use different unpatched references")
        if row["intervention"] != "sham":
            for k in ("mean_shift_l2", "shift_l2"):
                if abs(a[k] - ref[k]) > TOLERANCES["absolute"] + TOLERANCES["norm_relative"] * ref[k]:
                    raise InterventionError("matched realized control/R norm differs by more than 5 percent")
    return {"passed": True, "conditions": len(records), "tolerances": TOLERANCES}


class OrthogonalPrefillTransform(PrefillOnlyTransform):
    def __call__(self, activation):
        length = activation.shape[-2]
        if length == 1:
            self.cache_skips += 1
            return activation
        if length != self.expected_sequence_length:
            raise InterventionError("unexpected non-cache sequence length")
        self.prefill_hits += 1
        if self.prefill_hits != 1:
            raise InterventionError("intervention hit prefill more than once")
        result, self.audit = apply_orthogonal(activation, self.token_selection, self.spec, **self.kwargs)
        return result


class OrthogonalRuntime(QwenCausalRuntime):
    def _transform(self, **kwargs):
        original = super()._transform(**kwargs)
        return OrthogonalPrefillTransform(expected_sequence_length=original.expected_sequence_length,
                    token_selection=original.token_selection, spec=original.spec, **original.kwargs)
