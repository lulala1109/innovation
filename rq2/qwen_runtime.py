"""Qwen-specific, prefill-only execution for RQ2 interventions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from core.activations import ActivationPatch, ActivationPatcher, ForwardActivationCollector, pool_tokens
from experiments.qwen_activation_patching import align_token_selections
from rq2.artifacts import atomic_json
from rq2.interventions import (
    InterventionAudit,
    InterventionError,
    InterventionSpec,
    PrefillOnlyTransform,
)
from rq2.rq1_bundle import FrozenRQ1Bundle


class QwenRuntimeError(RuntimeError):
    """Raised when hook placement or execution violates the RQ2 protocol."""


@dataclass(frozen=True)
class RuntimeTrialResult:
    response: str
    diagnostic: Mapping[str, Any]
    diagnostic_audit: InterventionAudit
    generation_audit: InterventionAudit
    generation_cache_skips: int


def _audio_indices(prompt: Mapping[str, Any]) -> tuple[int, ...]:
    spans = prompt.get("token_spans")
    embeds = prompt.get("inputs_embeds")
    if not isinstance(spans, Mapping) or "audio" not in spans:
        raise QwenRuntimeError("prepared prompt lacks an audio token span")
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
        raise QwenRuntimeError("prepared prompt lacks [B,T,D] inputs_embeds")
    start, end = spans["audio"]
    if not all(isinstance(value, int) for value in (start, end)):
        raise QwenRuntimeError("audio span endpoints must be integers")
    if start < 0 or end <= start or end > embeds.shape[-2]:
        raise QwenRuntimeError("audio span is outside the prepared prompt")
    return tuple(range(start, end))


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise QwenRuntimeError(f"{name} is not a tensor")
    return value


def _maximum_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max().item())


class QwenCausalRuntime:
    """Execute RQ2 diagnostics and generation without modifying RQ1 wrappers."""

    def __init__(
        self,
        model: Any,
        bundle: FrozenRQ1Bundle,
        *,
        output_selector: Optional[int | str] = None,
        source_cache_size: int = 48,
    ) -> None:
        required = (
            "get_transformer_layer_modules",
            "forward_prepared_prompt",
            "generate_from_prepared_prompt",
        )
        missing = [name for name in required if not callable(getattr(model, name, None))]
        if missing:
            raise TypeError(f"Qwen runtime model is missing methods: {missing}")
        self.model = model
        self.bundle = bundle
        self.output_selector = output_selector
        self.modules = OrderedDict(model.get_transformer_layer_modules())
        self.final_output_norm_layer: Optional[int] = None
        norm_getter = getattr(model, "get_final_decoder_norm_module", None)
        if callable(norm_getter):
            final_layer = next(reversed(self.modules))
            norm = norm_getter()
            if not isinstance(norm, torch.nn.Module):
                raise QwenRuntimeError("final decoder norm is not a torch module")
            self.modules[final_layer] = norm
            self.final_output_norm_layer = final_layer
        if source_cache_size < 1:
            raise ValueError("source_cache_size must be positive")
        self.source_cache_size = int(source_cache_size)
        self._source_cache: "OrderedDict[tuple[str, int], torch.Tensor]" = OrderedDict()
        if tuple(self.modules) != tuple(bundle.hidden_sizes):
            raise QwenRuntimeError(
                "decoder layers do not match the frozen RQ1 probe layers"
            )

    def _source_activation(
        self,
        prompt: Mapping[str, Any],
        layer: int,
        *,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        if cache_key is None:
            return self.capture(prompt, layers=(layer,))[layer]
        key = (str(cache_key), layer)
        if key in self._source_cache:
            value = self._source_cache.pop(key)
            self._source_cache[key] = value
            return value
        value = self.capture(prompt, layers=(layer,))[layer]
        self._source_cache[key] = value
        while len(self._source_cache) > self.source_cache_size:
            self._source_cache.popitem(last=False)
        return value

    def clear_activation_cache(self) -> None:
        self._source_cache.clear()

    def capture(
        self,
        prompt: Mapping[str, Any],
        *,
        layers: Optional[Sequence[int]] = None,
        detach: bool = True,
    ) -> "OrderedDict[int, torch.Tensor]":
        selected = self.modules if layers is None else OrderedDict(
            (layer, self.modules[layer]) for layer in layers
        )
        collector = ForwardActivationCollector(
            selected,
            pooling="none",
            output_selector=self.output_selector,
            detach=detach,
        )
        context = torch.no_grad() if detach else torch.enable_grad()
        with context, collector:
            self.model.forward_prepared_prompt(
                prompt, output_hidden_states=False, use_cache=False
            )
        missing = [layer for layer in selected if layer not in collector.activations]
        if missing:
            raise QwenRuntimeError(f"forward missed decoder layers: {missing}")
        return OrderedDict(collector.activations)

    def map_layers(
        self,
        prompt: Mapping[str, Any],
        *,
        rq1_pooled: Optional[Mapping[int, torch.Tensor]] = None,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        output_path: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        """Prove hook layer i matches both HF states and optional RQ1 replay pooling."""

        collector = ForwardActivationCollector(
            self.modules,
            pooling="none",
            output_selector=self.output_selector,
            detach=True,
        )
        with torch.no_grad(), collector:
            output = self.model.forward_prepared_prompt(
                prompt, output_hidden_states=True, use_cache=False
            )
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(hidden_states, (tuple, list)):
            raise QwenRuntimeError("Qwen did not return a hidden_states sequence")
        if len(hidden_states) != len(self.modules) + 1:
            raise QwenRuntimeError(
                "hidden_states must contain embedding output plus every decoder layer"
            )
        records: list[dict[str, Any]] = []
        all_passed = True
        audio_indices = torch.tensor(_audio_indices(prompt), dtype=torch.long)
        attention_mask = _tensor(prompt.get("attention_mask"), "attention_mask")
        for layer in self.modules:
            hook = collector.activations.get(layer)
            returned = hidden_states[layer + 1]
            if not isinstance(hook, torch.Tensor) or not isinstance(returned, torch.Tensor):
                raise QwenRuntimeError(f"layer {layer} output is not tensor-valued")
            shape_equal = tuple(hook.shape) == tuple(returned.shape)
            max_abs = _maximum_difference(hook, returned)
            hook_flat = hook.detach().float().reshape(-1)
            returned_flat = returned.detach().float().reshape(-1)
            cosine = (
                float(torch.nn.functional.cosine_similarity(hook_flat, returned_flat, dim=0).item())
                if shape_equal
                else None
            )
            hidden_state_passed = shape_equal and bool(
                torch.allclose(hook.float(), returned.float(), atol=atol, rtol=rtol)
            )
            rq1_max_abs = None
            rq1_cosine = None
            rq1_passed = True
            if rq1_pooled is not None:
                reference = rq1_pooled.get(layer)
                if not isinstance(reference, torch.Tensor):
                    raise QwenRuntimeError(f"RQ1 replay lacks layer {layer}")
                pooled = pool_tokens(
                    hook.detach(),
                    pooling="mean",
                    attention_mask=attention_mask,
                    token_selection=audio_indices,
                ).float().reshape(-1).cpu()
                reference = reference.detach().float().reshape(-1).cpu()
                rq1_max_abs = _maximum_difference(pooled, reference)
                rq1_cosine = (
                    float(torch.nn.functional.cosine_similarity(pooled, reference, dim=0).item())
                    if pooled.shape == reference.shape else None
                )
                rq1_passed = pooled.shape == reference.shape and bool(
                    torch.allclose(pooled, reference, atol=atol, rtol=rtol)
                )
            passed = hidden_state_passed and rq1_passed
            all_passed = all_passed and passed
            records.append(
                {
                    "decoder_layer": layer,
                    "hidden_state_index": layer + 1,
                    "activation_site": (
                        "final_output_norm" if layer == self.final_output_norm_layer
                        else "decoder_block_output"
                    ),
                    "audio_only_prefill_behaviorally_reachable": (
                        layer != self.final_output_norm_layer
                    ),
                    "shape": list(hook.shape),
                    "shape_equal": shape_equal,
                    "max_abs_error": max_abs,
                    "cosine_similarity": cosine,
                    "hidden_state_passed": hidden_state_passed,
                    "rq1_replay_max_abs_error": rq1_max_abs,
                    "rq1_replay_cosine_similarity": rq1_cosine,
                    "rq1_replay_passed": rq1_passed,
                    "passed": passed,
                }
            )
        payload = {
            "format": "rq2-qwen-layer-map",
            "version": 1,
            "sequence_has_embedding": True,
            "rq1_replay_checked": rq1_pooled is not None,
            "layer_count": len(records),
            "atol": atol,
            "rtol": rtol,
            "passed": all_passed,
            "layers": records,
        }
        if output_path is not None:
            atomic_json(output_path, payload)
        if not all_passed:
            raise QwenRuntimeError("decoder hooks do not match HF states and the RQ1 replay convention")
        return payload

    def safety_profile(
        self,
        activations: Mapping[int, torch.Tensor],
        token_indices: Sequence[int],
    ) -> dict[str, dict[str, float]]:
        """Recompute all layerwise probe probabilities and direction projections."""

        pooled: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        index_cache: dict[torch.device, torch.Tensor] = {}
        for layer, activation in activations.items():
            if activation.ndim != 3 or activation.shape[0] != 1:
                raise QwenRuntimeError("diagnostic activations must have shape [1,T,D]")
            if activation.device not in index_cache:
                index_cache[activation.device] = torch.tensor(
                    token_indices, device=activation.device, dtype=torch.long
                )
            pooled[layer] = pool_tokens(
                activation,
                pooling="mean",
                attention_mask=torch.ones(
                    activation.shape[:-1], device=activation.device, dtype=torch.long
                ),
                token_selection=index_cache[activation.device],
            ).float().detach().cpu()
        expected = tuple(self.bundle.hidden_sizes)
        if tuple(pooled) != expected:
            raise QwenRuntimeError(
                f"a complete {len(expected)}-layer activation profile is required"
            )
        scorer = self.bundle.scorer.cpu().eval()
        with torch.no_grad():
            scores = scorer(pooled)
        result: dict[str, dict[str, float]] = {
            "harmfulness_probability": {},
            "refusal_probability": {},
            "harmfulness_projection": {},
            "refusal_projection": {},
        }
        for layer, hidden in pooled.items():
            result["harmfulness_probability"][str(layer)] = float(
                scores.harmfulness[layer].reshape(-1)[0].item()
            )
            result["refusal_probability"][str(layer)] = float(
                scores.refusal[layer].reshape(-1)[0].item()
            )
            for state in ("harmfulness", "refusal"):
                unit = self.bundle.unit_directions[state][layer]
                center = self.bundle.centers[state][layer]
                value = torch.dot(hidden.reshape(-1) - center, unit)
                result[f"{state}_projection"][str(layer)] = float(value.item())
        return result

    def _transform(
        self,
        *,
        prompt: Mapping[str, Any],
        source_activation: torch.Tensor,
        source_indices: Sequence[int],
        target_indices: Sequence[int],
        target_reference_indices: Optional[Sequence[int]],
        application_scale: float,
        spec: InterventionSpec,
        subspace: Optional[torch.Tensor] = None,
    ) -> PrefillOnlyTransform:
        embeds = _tensor(prompt.get("inputs_embeds"), "inputs_embeds")
        layer = spec.layer
        kwargs: dict[str, Any] = {
            "expected_sequence_length": int(embeds.shape[-2]),
            "token_selection": tuple(target_indices),
            "spec": spec,
            "source_activation": source_activation,
            "source_token_selection": tuple(source_indices),
            "target_reference_selection": (
                None if target_reference_indices is None else tuple(target_reference_indices)
            ),
            "application_scale": float(application_scale),
            "refusal_direction": self.bundle.unit_directions["refusal"][layer],
            "harmfulness_direction": self.bundle.unit_directions["harmfulness"][layer],
            "refusal_sigma": self.bundle.refusal_sigma[layer],
            "subspace": subspace,
        }
        return PrefillOnlyTransform(**kwargs)

    def diagnostic_trial(
        self,
        source_prompt: Mapping[str, Any],
        target_prompt: Mapping[str, Any],
        spec: InterventionSpec,
        *,
        token_selection: str | Sequence[int] = "audio",
        reference_token_selection: Optional[str | Sequence[int]] = None,
        application_scale: float = 1.0,
        subspace: Optional[torch.Tensor] = None,
        source_cache_key: Optional[str] = None,
    ) -> tuple[dict[str, Any], InterventionAudit]:
        if reference_token_selection is None:
            source_indices, target_indices = align_token_selections(
                source_prompt, target_prompt, token_selection
            )
            target_reference_indices = target_indices
        else:
            source_indices, target_reference_indices = align_token_selections(
                source_prompt, target_prompt, reference_token_selection
            )
            target_indices = tuple(int(value) for value in token_selection)
        activation_prompt = target_prompt if spec.kind == "self_patch" else source_prompt
        source_activation = self._source_activation(
            activation_prompt,
            spec.layer,
            cache_key=None if spec.kind == "self_patch" else source_cache_key,
        )
        transform = self._transform(
            prompt=target_prompt,
            source_activation=source_activation,
            source_indices=source_indices,
            target_indices=target_indices,
            target_reference_indices=target_reference_indices,
            application_scale=application_scale,
            spec=spec,
            subspace=subspace,
        )
        collector = ForwardActivationCollector(
            self.modules,
            pooling="none",
            output_selector=self.output_selector,
            detach=True,
        )
        patch = ActivationPatch(transform, output_selector=self.output_selector)
        with torch.no_grad(), ActivationPatcher({self.modules[spec.layer]: patch}), collector:
            self.model.forward_prepared_prompt(
                target_prompt, output_hidden_states=False, use_cache=False
            )
        audit = transform.assert_applied_once()
        profile = self.safety_profile(collector.activations, target_reference_indices)
        return {
            "profile": profile,
            "target_token_count": len(target_indices),
            "source_token_count": len(source_indices),
            "source_token_indices": list(source_indices),
            "target_token_indices": list(target_indices),
            "target_reference_token_indices": list(target_reference_indices),
            "application_scale": float(application_scale),
        }, audit

    def generate_trial(
        self,
        source_prompt: Mapping[str, Any],
        target_prompt: Mapping[str, Any],
        spec: InterventionSpec,
        *,
        token_selection: str | Sequence[int] = "audio",
        reference_token_selection: Optional[str | Sequence[int]] = None,
        application_scale: float = 1.0,
        subspace: Optional[torch.Tensor] = None,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        source_cache_key: Optional[str] = None,
    ) -> tuple[str, InterventionAudit, int]:
        if reference_token_selection is None:
            source_indices, target_indices = align_token_selections(
                source_prompt, target_prompt, token_selection
            )
            target_reference_indices = target_indices
        else:
            source_indices, target_reference_indices = align_token_selections(
                source_prompt, target_prompt, reference_token_selection
            )
            target_indices = tuple(int(value) for value in token_selection)
        activation_prompt = target_prompt if spec.kind == "self_patch" else source_prompt
        source_activation = self._source_activation(
            activation_prompt,
            spec.layer,
            cache_key=None if spec.kind == "self_patch" else source_cache_key,
        )
        transform = self._transform(
            prompt=target_prompt,
            source_activation=source_activation,
            source_indices=source_indices,
            target_indices=target_indices,
            target_reference_indices=target_reference_indices,
            application_scale=application_scale,
            spec=spec,
            subspace=subspace,
        )
        patch = ActivationPatch(transform, output_selector=self.output_selector)
        with ActivationPatcher({self.modules[spec.layer]: patch}):
            response = self.model.generate_from_prepared_prompt(
                target_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        audit = transform.assert_applied_once()
        return str(response), audit, transform.cache_skips

    def run_trial(
        self,
        source_prompt: Mapping[str, Any],
        target_prompt: Mapping[str, Any],
        spec: InterventionSpec,
        **generation: Any,
    ) -> RuntimeTrialResult:
        token_selection = generation.pop("token_selection", "audio")
        reference_token_selection = generation.pop("reference_token_selection", None)
        application_scale = float(generation.pop("application_scale", 1.0))
        subspace = generation.pop("subspace", None)
        source_cache_key = generation.pop("source_cache_key", None)
        diagnostic, diagnostic_audit = self.diagnostic_trial(
            source_prompt, target_prompt, spec,
            token_selection=token_selection,
            reference_token_selection=reference_token_selection,
            application_scale=application_scale,
            subspace=subspace,
            source_cache_key=source_cache_key,
        )
        response, generation_audit, cache_skips = self.generate_trial(
            source_prompt,
            target_prompt,
            spec,
            token_selection=token_selection,
            reference_token_selection=reference_token_selection,
            application_scale=application_scale,
            subspace=subspace,
            source_cache_key=source_cache_key,
            **generation,
        )
        return RuntimeTrialResult(
            response=response,
            diagnostic=diagnostic,
            diagnostic_audit=diagnostic_audit,
            generation_audit=generation_audit,
            generation_cache_skips=cache_skips,
        )

    def identity_tests(
        self,
        prompt: Mapping[str, Any],
        *,
        layer: int,
        atol: float = 1e-6,
        max_tokens: int = 32,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Run no-hook/capture/self/sham/lambda-zero identity checks."""

        indices = _audio_indices(prompt)
        with torch.no_grad():
            baseline = self.model.forward_prepared_prompt(
                prompt, output_hidden_states=False, use_cache=False
            )
        baseline_logits = _tensor(getattr(baseline, "logits", None), "baseline logits")
        baseline_activations = self.capture(prompt)
        baseline_profile = self.safety_profile(baseline_activations, indices)
        baseline_generation = str(self.model.generate_from_prepared_prompt(
            prompt, max_tokens=max_tokens, temperature=temperature, do_sample=False
        ))
        collector = ForwardActivationCollector(
            {layer: self.modules[layer]},
            pooling="none",
            output_selector=self.output_selector,
            detach=True,
        )
        with torch.no_grad(), collector:
            captured_output = self.model.forward_prepared_prompt(
                prompt, output_hidden_states=False, use_cache=False
            )
        captured_logits = _tensor(getattr(captured_output, "logits", None), "capture logits")
        source = collector.activations[layer]
        logit_tests = {
            "no_hook_vs_capture": _maximum_difference(baseline_logits, captured_logits)
        }
        profile_tests: dict[str, float] = {}
        generation_tests: dict[str, bool] = {}
        for name, kind, dose in (
            ("self", "self_patch", 1.0),
            ("sham", "sham", 1.0),
            ("lambda_zero", "r_direction", 0.0),
        ):
            spec = InterventionSpec(kind=kind, layer=layer, dose=dose)
            transform = self._transform(
                prompt=prompt,
                source_activation=source,
                source_indices=indices,
                target_indices=indices,
                target_reference_indices=indices,
                application_scale=1.0,
                spec=spec,
            )
            patched_collector = ForwardActivationCollector(
                self.modules,
                pooling="none",
                output_selector=self.output_selector,
                detach=True,
            )
            with torch.no_grad(), ActivationPatcher(
                {
                    self.modules[layer]: ActivationPatch(
                        transform, output_selector=self.output_selector
                    )
                }
            ), patched_collector:
                output = self.model.forward_prepared_prompt(
                    prompt, output_hidden_states=False, use_cache=False
                )
            transform.assert_applied_once()
            logit_tests[name] = _maximum_difference(
                baseline_logits, _tensor(getattr(output, "logits", None), f"{name} logits")
            )
            patched_profile = self.safety_profile(patched_collector.activations, indices)
            profile_tests[name] = max(
                abs(float(patched_profile[metric][str(index)]) - float(baseline_profile[metric][str(index)]))
                for metric in baseline_profile
                for index in self.modules
            )
            generation_transform = self._transform(
                prompt=prompt,
                source_activation=source,
                source_indices=indices,
                target_indices=indices,
                target_reference_indices=indices,
                application_scale=1.0,
                spec=spec,
            )
            with ActivationPatcher({
                self.modules[layer]: ActivationPatch(
                    generation_transform, output_selector=self.output_selector
                )
            }):
                generated = str(self.model.generate_from_prepared_prompt(
                    prompt, max_tokens=max_tokens, temperature=temperature, do_sample=False
                ))
            generation_transform.assert_applied_once()
            generation_tests[name] = generated == baseline_generation
        with torch.no_grad():
            post = self.model.forward_prepared_prompt(
                prompt, output_hidden_states=False, use_cache=False
            )
        hook_cleanup_error = _maximum_difference(
            baseline_logits, _tensor(getattr(post, "logits", None), "post-identity logits")
        )
        passed = (
            all(value <= atol for value in logit_tests.values())
            and all(value <= atol for value in profile_tests.values())
            and all(generation_tests.values())
            and hook_cleanup_error <= atol
        )
        return {
            "format": "rq2-identity-tests",
            "version": 1,
            "layer": layer,
            "atol": atol,
            "passed": passed,
            "max_abs_logit_differences": logit_tests,
            "max_abs_profile_differences": profile_tests,
            "generation_exact_matches": generation_tests,
            "hook_cleanup_max_abs_logit_difference": hook_cleanup_error,
        }


__all__ = [
    "QwenCausalRuntime",
    "QwenRuntimeError",
    "RuntimeTrialResult",
]
