"""Canonical waveform PGD with layer-wise safety-state objectives.

This module deliberately keeps the waveform update identical across all
ablations.  Every method minimizes its objective with a sign-gradient step and
then projects the perturbation onto the same L-infinity ball.  The only thing
that changes between the layer-aware methods is how their layer weights are
chosen.

State ``0`` is the initialized perturbation.  State ``N`` is the perturbation
after ``N`` completed updates, so an attack with ``N`` updates records exactly
``N + 1`` aligned states.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from attacks.base import AttackResult, BaseWavAttacker
from core.activations import collect_hidden_states, resolve_layer_indices
from core.behavior_decision import BehaviorEvaluator, evaluate_behavior
from core.safety_state import (
    compute_layer_weights,
    compute_safety_gaps,
    differentiable_state_loss,
    dynamic_bottleneck,
)
from models.base import AttackForwardOutput, BaseAudioModel


LayerKey = Hashable
LayerSelector = Optional[Union[int, slice, str, Iterable[int]]]
LayerTensorMap = "OrderedDict[LayerKey, torch.Tensor]"

LayerMethod = Literal[
    "standard",
    "fixed",
    "uniform",
    "static_topk",
    "gradient_adaptive",
    "safety_state_adaptive",
]

SUPPORTED_METHODS: Tuple[str, ...] = (
    "standard",
    "fixed",
    "uniform",
    "static_topk",
    "gradient_adaptive",
    "safety_state_adaptive",
)


@dataclass
class _StateEvaluation:
    total_loss: torch.Tensor
    target_loss: torch.Tensor
    state_loss: torch.Tensor
    utility_loss: torch.Tensor
    refusal_scores: Optional[LayerTensorMap] = None
    safety_gaps: Optional[LayerTensorMap] = None
    layer_weights: Optional[LayerTensorMap] = None
    gradient_scores: Optional[LayerTensorMap] = None
    bottleneck_layer: Optional[LayerKey] = None
    bottleneck_gap: Optional[torch.Tensor] = None
    reference_refusal: Optional[LayerTensorMap] = None
    static_weights: Optional[LayerTensorMap] = None
    hidden_states: Optional[LayerTensorMap] = None
    harmfulness_scores: Optional[LayerTensorMap] = None
    harmfulness_directions: Optional[LayerTensorMap] = None
    refusal_directions: Optional[LayerTensorMap] = None


def _as_scalar_layer_map(
    values: Mapping[LayerKey, torch.Tensor],
    *,
    name: str,
) -> LayerTensorMap:
    """Reduce batch/token axes while retaining one differentiable value/layer."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a layer-to-tensor mapping")
    if not values:
        raise ValueError(f"{name} cannot be empty")
    result: LayerTensorMap = OrderedDict()
    for layer, value in values.items():
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.numel() == 0:
            raise ValueError(f"{name} for layer {layer!r} is empty")
        result[layer] = value.mean()
    return result


def _float_map(
    values: Optional[Mapping[LayerKey, torch.Tensor]],
) -> Optional[Dict[LayerKey, float]]:
    if values is None:
        return None
    return {
        layer: float(value.detach().mean().item())
        for layer, value in values.items()
    }


class LayerAdaptivePGDAttacker(BaseWavAttacker):
    """Output-only and layer-aware PGD ablations under one fair update rule.

    ``standard`` needs only the ordinary :class:`BaseAudioModel` loss API.
    Every other method requires a safety scorer with ``score_refusal`` and an
    explicit ``reference_refusal`` passed to :meth:`attack`.  Requiring the
    reference even for gradient-based selection makes the logged safety gaps
    and bottlenecks comparable across all layer-aware ablations; importantly,
    gradient-adaptive *weights* are still computed only from per-layer input
    gradient norms and never from those gaps.

    The static baselines are defined as follows:

    * ``fixed``: one-hot weight on ``fixed_layer``;
    * ``uniform``: equal weight on every selected layer;
    * ``static_topk``: equal weight on a fixed, externally supplied top-k set;
    * ``gradient_adaptive``: online softmax of per-layer delta-gradient norms;
    * ``safety_state_adaptive``: online softmax of refusal degradation gaps.

    Probe and model parameters are never optimizer variables.  Gradients are
    requested with :func:`torch.autograd.grad` only for ``delta``, preventing
    parameter ``.grad`` accumulation even when a trainable probe is supplied.
    """

    def __init__(
        self,
        model: BaseAudioModel,
        safety_scorer: Optional[Any] = None,
        *,
        eps: float = 0.1,
        alpha: float = 0.005,
        loss_type: Literal["ce", "margin"] = "margin",
        kappa: float = 5.0,
        method: LayerMethod = "safety_state_adaptive",
        state_loss_weight: float = 1.0,
        utility_loss_weight: float = 0.0,
        utility_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        lambda_state: Optional[float] = None,
        temperature: float = 1.0,
        layers: LayerSelector = None,
        token_span: Literal["audio", "target", "all"] = "audio",
        pooling: Literal["mean", "max", "first", "last", "none"] = "mean",
        sequence_has_embedding: bool = True,
        fixed_layer: Optional[LayerKey] = None,
        top_k: int = 1,
        static_layer_weights: Optional[
            Union[Mapping[LayerKey, float], Sequence[float], torch.Tensor]
        ] = None,
        static_topk_layers: Optional[Iterable[LayerKey]] = None,
        probe_return_logits: bool = False,
        detach_dynamic_weights: bool = True,
        use_lowpass: bool = False,
        lowpass_cutoff: float = 2000.0,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            eps=eps,
            alpha=alpha,
            use_lowpass=use_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            verbose=verbose,
        )
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}; got {method!r}"
            )
        if loss_type not in {"ce", "margin"}:
            raise ValueError("loss_type must be 'ce' or 'margin'")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lambda_state is not None:
            if state_loss_weight != 1.0 and not math.isclose(
                float(state_loss_weight), float(lambda_state)
            ):
                raise ValueError(
                    "state_loss_weight and lambda_state specify different values"
                )
            state_loss_weight = float(lambda_state)
        if not math.isfinite(float(state_loss_weight)) or state_loss_weight < 0:
            raise ValueError("state_loss_weight must be finite and non-negative")
        if not math.isfinite(float(utility_loss_weight)) or utility_loss_weight < 0:
            raise ValueError("utility_loss_weight must be finite and non-negative")

        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if token_span not in {"audio", "target", "all"}:
            raise ValueError("token_span must be one of: audio, target, all")
        if pooling not in {"mean", "max", "first", "last", "none"}:
            raise ValueError(
                "pooling must be one of: mean, max, first, last, none"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if method == "fixed" and fixed_layer is None:
            raise ValueError("fixed method requires fixed_layer")
        if (
            method == "static_topk"
            and static_layer_weights is None
            and static_topk_layers is None
        ):
            raise ValueError(
                "static_topk requires static_layer_weights or static_topk_layers"
            )

        self.safety_scorer = safety_scorer
        self.loss_type = loss_type
        self.kappa = float(kappa)
        self.method = method
        self.state_loss_weight = float(state_loss_weight)
        self.utility_loss_weight = float(utility_loss_weight)
        self.utility_loss_fn = utility_loss_fn
        self.temperature = float(temperature)
        self.layers = layers
        self.token_span = token_span
        self.pooling = pooling
        self.sequence_has_embedding = bool(sequence_has_embedding)
        self.fixed_layer = fixed_layer
        self.top_k = top_k
        self.static_layer_weights = static_layer_weights
        self.static_topk_layers = (
            None if static_topk_layers is None else tuple(static_topk_layers)
        )
        self.probe_return_logits = bool(probe_return_logits)
        self.detach_dynamic_weights = bool(detach_dynamic_weights)

    @property
    def uses_safety_states(self) -> bool:
        return self.method != "standard"

    def _utility_loss(
        self,
        clean_wav: torch.Tensor,
        adversarial_wav: torch.Tensor,
    ) -> torch.Tensor:
        if self.utility_loss_weight == 0:
            return adversarial_wav.new_zeros(())
        if self.utility_loss_fn is None:
            value = (adversarial_wav - clean_wav).square().mean()
        else:
            value = self.utility_loss_fn(clean_wav, adversarial_wav)
            if not isinstance(value, torch.Tensor):
                value = adversarial_wav.new_tensor(value)
            if value.numel() != 1:
                value = value.mean()
        if not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("utility loss must be finite")
        return value

    def _standard_target_loss(
        self, wav: torch.Tensor, target_text: str
    ) -> torch.Tensor:
        if self.loss_type == "margin":
            return self.model.compute_margin_loss(
                wav, target_text, kappa=self.kappa
            )
        return self.model.compute_loss(wav, target_text)

    def _target_loss_from_forward(
        self, forward: AttackForwardOutput
    ) -> torch.Tensor:
        if self.loss_type == "ce":
            return forward.loss
        if forward.logits is None or forward.labels is None:
            raise ValueError(
                "margin loss in a layer-aware attack requires forward_attack() "
                "to return logits and labels"
            )

        shift_logits = forward.logits[:, :-1, :].contiguous()
        shift_labels = forward.labels[:, 1:].contiguous()
        per_sample = []
        for batch_index in range(shift_labels.shape[0]):
            valid = shift_labels[batch_index] != -100
            if not bool(valid.any()):
                continue
            logits = shift_logits[batch_index, valid]
            labels = shift_labels[batch_index, valid]
            target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other_mask = torch.ones_like(logits, dtype=torch.bool)
            other_mask.scatter_(1, labels.unsqueeze(1), False)
            top_other = logits.masked_fill(
                ~other_mask, float("-inf")
            ).max(dim=-1).values
            margins = torch.clamp(
                top_other - target_logits + self.kappa, min=0
            )
            token_weights = torch.ones_like(margins)
            token_weights[: min(3, len(token_weights))] = 5.0
            per_sample.append(
                (margins * token_weights).sum() / token_weights.sum()
            )
        if not per_sample:
            return forward.loss
        return torch.stack(per_sample).mean()

    def _selected_hidden_states(
        self, forward: AttackForwardOutput
    ) -> LayerTensorMap:
        hidden_states = forward.hidden_states
        if hidden_states is None:
            raise ValueError(
                "layer-aware methods require forward_attack(..., "
                "output_hidden_states=True) to return hidden_states"
            )

        token_selection = None
        if self.token_span != "all":
            if self.token_span not in forward.token_spans:
                raise KeyError(
                    f"forward_attack did not provide the requested "
                    f"{self.token_span!r} token span"
                )
            token_selection = forward.token_spans[self.token_span]

        if isinstance(hidden_states, Mapping):
            available_keys = tuple(hidden_states.keys())
            if self.layers is None or self.layers == "all":
                selected_keys = available_keys
            elif self.layers == "last":
                if not available_keys:
                    raise ValueError("hidden_states cannot be empty")
                selected_keys = (available_keys[-1],)
            elif isinstance(self.layers, (int, str)):
                selected_keys = (self.layers,)
            elif isinstance(self.layers, slice):
                selected_keys = available_keys[self.layers]
            else:
                selected_keys = tuple(self.layers)
            collected = collect_hidden_states(
                hidden_states,
                layers=selected_keys,
                pooling=self.pooling,
                attention_mask=forward.attention_mask,
                token_selection=token_selection,
            )
        elif isinstance(hidden_states, Sequence) and not isinstance(
            hidden_states, (str, bytes)
        ):
            num_layers = len(hidden_states) - int(self.sequence_has_embedding)
            if num_layers <= 0:
                raise ValueError("forward_attack returned no transformer layers")
            selected_indices = resolve_layer_indices(num_layers, self.layers)
            collected = collect_hidden_states(
                hidden_states,
                layers=selected_indices,
                pooling=self.pooling,
                attention_mask=forward.attention_mask,
                token_selection=token_selection,
                sequence_has_embedding=self.sequence_has_embedding,
            )
        else:
            raise TypeError(
                "forward_attack hidden_states must be a mapping or sequence"
            )
        if not collected:
            raise ValueError("layer selection retained no hidden states")
        return OrderedDict(collected)

    def _score_refusal(self, states: LayerTensorMap) -> LayerTensorMap:
        if self.safety_scorer is None:
            raise ValueError(
                f"method {self.method!r} requires safety_scorer"
            )
        score_method = getattr(self.safety_scorer, "score_refusal", None)
        if score_method is None or not callable(score_method):
            raise TypeError("safety_scorer must provide score_refusal()")
        scores = score_method(
            states, return_logits=self.probe_return_logits
        )
        return _as_scalar_layer_map(scores, name="refusal scores")

    def _auxiliary_state_scores(
        self,
        states: LayerTensorMap,
    ) -> Tuple[
        Optional[LayerTensorMap],
        Optional[LayerTensorMap],
        Optional[LayerTensorMap],
    ]:
        """Collect non-objective H scores and named direction projections."""

        if self.safety_scorer is None:
            return None, None, None
        detached_states = OrderedDict(
            (layer, value.detach()) for layer, value in states.items()
        )
        harmfulness = None
        harmfulness_direction = None
        refusal_direction = None
        with torch.no_grad():
            score_method = getattr(self.safety_scorer, "score_harmfulness", None)
            if callable(score_method):
                harmfulness = _as_scalar_layer_map(
                    score_method(
                        detached_states,
                        return_logits=self.probe_return_logits,
                    ),
                    name="harmfulness scores",
                )
            direction_method = getattr(self.safety_scorer, "direction_scores", None)
            if callable(direction_method):
                directions = direction_method(detached_states)
                harmfulness_direction = _as_scalar_layer_map(
                    directions.harmfulness, name="harmfulness directions"
                )
                refusal_direction = _as_scalar_layer_map(
                    directions.refusal, name="refusal directions"
                )
        return harmfulness, harmfulness_direction, refusal_direction


    @staticmethod
    def _normalize_reference(
        reference: Any,
        current: LayerTensorMap,
    ) -> LayerTensorMap:
        keys = tuple(current.keys())
        if isinstance(reference, Mapping):
            if set(reference.keys()) != set(keys):
                raise KeyError(
                    "reference_refusal and selected refusal layers do not match"
                )
            raw_values = [reference[key] for key in keys]
        elif isinstance(reference, torch.Tensor):
            if reference.numel() != len(keys):
                raise ValueError(
                    "reference_refusal tensor must contain one value per layer"
                )
            raw_values = tuple(reference.reshape(-1))
        elif isinstance(reference, Sequence) and not isinstance(
            reference, (str, bytes)
        ):
            if len(reference) != len(keys):
                raise ValueError(
                    "reference_refusal sequence must contain one value per layer"
                )
            raw_values = reference
        else:
            raise TypeError(
                "reference_refusal must explicitly contain refusal values as a "
                "mapping, tensor, or sequence (pass DualSafetyScores.refusal, "
                "not the complete dual state)"
            )

        normalized: LayerTensorMap = OrderedDict()
        for key, raw in zip(keys, raw_values):
            value = torch.as_tensor(
                raw,
                device=current[key].device,
                dtype=current[key].dtype,
            )
            if value.numel() == 0:
                raise ValueError(
                    f"reference_refusal for layer {key!r} is empty"
                )
            normalized[key] = value.detach().mean()
        return normalized

    @staticmethod
    def _uniform_weights(
        values: LayerTensorMap,
    ) -> LayerTensorMap:
        count = len(values)
        return OrderedDict(
            (
                layer,
                value.new_tensor(1.0 / count),
            )
            for layer, value in values.items()
        )

    @staticmethod
    def _resolve_named_layer(
        layer: LayerKey, keys: Tuple[LayerKey, ...]
    ) -> LayerKey:
        if layer in keys:
            return layer
        if isinstance(layer, int) and layer < 0:
            try:
                return keys[layer]
            except IndexError as exc:
                raise KeyError(
                    f"layer {layer!r} is not among selected layers {keys!r}"
                ) from exc
        raise KeyError(f"layer {layer!r} is not among selected layers {keys!r}")

    def _fixed_weights(self, values: LayerTensorMap) -> LayerTensorMap:
        keys = tuple(values.keys())
        layer = self._resolve_named_layer(self.fixed_layer, keys)
        return OrderedDict(
            (key, value.new_tensor(1.0 if key == layer else 0.0))
            for key, value in values.items()
        )

    def _build_static_topk_weights(
        self, values: LayerTensorMap
    ) -> LayerTensorMap:
        keys = tuple(values.keys())
        if self.static_topk_layers is not None:
            chosen = tuple(
                self._resolve_named_layer(layer, keys)
                for layer in self.static_topk_layers
            )
            chosen = tuple(dict.fromkeys(chosen))
            if not chosen:
                raise ValueError("static_topk_layers cannot be empty")
            if len(chosen) != self.top_k:
                raise ValueError(
                    "static_topk_layers must contain exactly top_k unique layers"
                )
        else:
            supplied = self.static_layer_weights
            if isinstance(supplied, Mapping):
                if set(supplied.keys()) != set(keys):
                    raise KeyError(
                        "static_layer_weights and selected layers do not match"
                    )
                scores = [float(torch.as_tensor(supplied[key]).item()) for key in keys]
            elif isinstance(supplied, torch.Tensor):
                if supplied.numel() != len(keys):
                    raise ValueError(
                        "static_layer_weights must have one score per selected layer"
                    )
                scores = [float(item) for item in supplied.detach().reshape(-1).tolist()]
            elif isinstance(supplied, Sequence) and not isinstance(
                supplied, (str, bytes)
            ):
                if len(supplied) != len(keys):
                    raise ValueError(
                        "static_layer_weights must have one score per selected layer"
                    )
                scores = [float(torch.as_tensor(item).item()) for item in supplied]
            else:
                raise TypeError(
                    "static_layer_weights must be a mapping, tensor, or sequence"
                )
            if self.top_k > len(keys):
                raise ValueError("top_k cannot exceed the selected layer count")
            ranked = sorted(
                range(len(keys)), key=lambda index: scores[index], reverse=True
            )
            chosen = tuple(keys[index] for index in ranked[: self.top_k])

        weight = 1.0 / len(chosen)
        chosen_set = set(chosen)
        return OrderedDict(
            (
                key,
                value.new_tensor(weight if key in chosen_set else 0.0),
            )
            for key, value in values.items()
        )

    def _gradient_adaptive_weights(
        self,
        refusal_scores: LayerTensorMap,
        delta: torch.Tensor,
    ) -> Tuple[LayerTensorMap, LayerTensorMap]:
        gradient_scores: LayerTensorMap = OrderedDict()
        for layer, score in refusal_scores.items():
            if not score.requires_grad:
                gradient = None
            else:
                gradient = torch.autograd.grad(
                    score,
                    delta,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
            norm = (
                delta.new_zeros(())
                if gradient is None
                else gradient.detach().reshape(-1).norm(p=2)
            )
            gradient_scores[layer] = norm
        weights = compute_layer_weights(
            gradient_scores, temperature=self.temperature
        )
        return OrderedDict(weights), gradient_scores

    def _evaluate_state(
        self,
        adversarial_wav: torch.Tensor,
        clean_wav: torch.Tensor,
        target_text: str,
        delta: torch.Tensor,
        reference_refusal: Optional[Any],
        normalized_reference: Optional[LayerTensorMap],
        static_weights: Optional[LayerTensorMap],
    ) -> _StateEvaluation:
        if self.method == "standard":
            target_loss = self._standard_target_loss(
                adversarial_wav, target_text
            )
            zero = target_loss.new_zeros(())
            utility_loss = self._utility_loss(clean_wav, adversarial_wav)
            return _StateEvaluation(
                total_loss=target_loss + self.utility_loss_weight * utility_loss,
                target_loss=target_loss,
                state_loss=zero,
                utility_loss=utility_loss,
            )

        forward = self.model.forward_attack(
            adversarial_wav,
            target_text,
            output_hidden_states=True,
        )
        target_loss = self._target_loss_from_forward(forward)
        states = self._selected_hidden_states(forward)
        refusal_scores = self._score_refusal(states)
        (
            harmfulness_scores,
            harmfulness_directions,
            refusal_directions,
        ) = self._auxiliary_state_scores(states)
        if normalized_reference is None:
            normalized_reference = self._normalize_reference(
                reference_refusal, refusal_scores
            )
        gaps = OrderedDict(
            compute_safety_gaps(normalized_reference, refusal_scores)
        )
        bottleneck = dynamic_bottleneck(gaps)

        gradient_scores = None
        if self.method == "fixed":
            layer_weights = self._fixed_weights(refusal_scores)
        elif self.method == "uniform":
            layer_weights = self._uniform_weights(refusal_scores)
        elif self.method == "static_topk":
            if static_weights is None:
                static_weights = self._build_static_topk_weights(refusal_scores)
            layer_weights = static_weights
        elif self.method == "gradient_adaptive":
            layer_weights, gradient_scores = self._gradient_adaptive_weights(
                refusal_scores, delta
            )
        elif self.method == "safety_state_adaptive":
            layer_weights = OrderedDict(
                compute_layer_weights(gaps, temperature=self.temperature)
            )
            if self.detach_dynamic_weights:
                layer_weights = OrderedDict(
                    (layer, weight.detach())
                    for layer, weight in layer_weights.items()
                )
        else:  # guarded by constructor; keeps type checkers exhaustive
            raise RuntimeError(f"unhandled layer method: {self.method}")

        state_loss = differentiable_state_loss(
            refusal_scores,
            layer_weights,
        )
        utility_loss = self._utility_loss(clean_wav, adversarial_wav)
        total_loss = (
            target_loss
            + self.state_loss_weight * state_loss
            + self.utility_loss_weight * utility_loss
        )
        return _StateEvaluation(
            total_loss=total_loss,
            target_loss=target_loss,
            state_loss=state_loss,
            utility_loss=utility_loss,
            refusal_scores=refusal_scores,
            safety_gaps=gaps,
            layer_weights=layer_weights,
            gradient_scores=gradient_scores,
            bottleneck_layer=bottleneck.layer,
            bottleneck_gap=bottleneck.gap,
            reference_refusal=normalized_reference,
            hidden_states=states,
            harmfulness_scores=harmfulness_scores,
            harmfulness_directions=harmfulness_directions,
            refusal_directions=refusal_directions,
            static_weights=static_weights,
        )

    def attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        steps: int = 100,
        *,
        init_mode: Literal["zero", "random"] = "zero",
        seed: int = 42,
        check_every: int = 20,
        early_stop: bool = True,
        state_callback: Optional[Callable[..., None]] = None,
        checkpoint_steps: Optional[Collection[int]] = None,
        behavior_evaluator: Optional[BehaviorEvaluator] = None,
        reference_refusal: Optional[Any] = None,
        callback: Optional[Callable[..., None]] = None,
        checkpoints: Optional[Collection[int]] = None,
    ) -> AttackResult:
        """Run canonical sign-gradient waveform PGD.

        Callback arguments are ``(state, adversarial_wav, delta, record)``.
        When checkpoints are supplied, only those exact states trigger the
        callback.  State 0 is before any update and the final state is after
        ``steps`` updates (unless early stopping ends the attack sooner).
        """

        if callback is not None:
            if state_callback is not None and state_callback is not callback:
                raise ValueError("callback and state_callback disagree")
            state_callback = callback
        if checkpoints is not None:
            if (
                checkpoint_steps is not None
                and set(checkpoint_steps) != set(checkpoints)
            ):
                raise ValueError("checkpoints and checkpoint_steps disagree")
            checkpoint_steps = checkpoints

        if steps < 0:
            raise ValueError("steps must be non-negative")
        if check_every < 0:
            raise ValueError("check_every must be non-negative")
        if init_mode not in {"zero", "random"}:
            raise ValueError("init_mode must be 'zero' or 'random'")
        if self.uses_safety_states:
            if self.safety_scorer is None:
                raise ValueError(
                    f"method {self.method!r} requires safety_scorer"
                )
            if reference_refusal is None:
                raise ValueError(
                    f"method {self.method!r} requires reference_refusal"
                )

        checkpoint_set = None
        if checkpoint_steps is not None:
            checkpoint_set = {int(step) for step in checkpoint_steps}
            invalid = sorted(
                step for step in checkpoint_set if step < 0 or step > steps
            )
            if invalid:
                raise ValueError(
                    "checkpoint steps must lie within [0, steps]; "
                    f"invalid values: {invalid}"
                )

        wav = wav.to(self.model.device)
        if not wav.is_floating_point():
            raise TypeError("Input waveform must be floating point")
        if not bool(torch.isfinite(wav).all()):
            raise ValueError("Input waveform contains NaN or infinite values")
        if wav.amin().item() < -1.0 or wav.amax().item() > 1.0:
            raise ValueError("Input waveform must be within [-1, 1]")

        if self.safety_scorer is not None and hasattr(
            self.safety_scorer, "zero_grad"
        ):
            self.safety_scorer.zero_grad(set_to_none=True)

        original_output = self.model.generate(wav, do_sample=False)
        generator = None
        if init_mode == "random":
            generator = torch.Generator(device=wav.device)
            generator.manual_seed(seed)
        delta = self.init_perturbation(
            wav,
            random_init=(init_mode == "random"),
            generator=generator,
        )

        history: Dict[str, Any] = {
            "method": self.method,
            "config": {
                "method": self.method,
                "eps": self.eps,
                "alpha": self.alpha,
                "steps": steps,
                "init_mode": init_mode,
                "seed": seed,
                "loss_type": self.loss_type,
                "kappa": self.kappa,
                "state_loss_weight": self.state_loss_weight,
                "utility_loss_weight": self.utility_loss_weight,
                "temperature": self.temperature,
                "layers": self.layers,
                "token_span": self.token_span,
                "pooling": self.pooling,
            },
            "iterations": [],
            "generation_checks": [],
            "selection": None,
            "reference_refusal": None,
            "first_success_step": None,
        }

        best_loss = float("inf")
        best_delta = None
        best_state = None
        best_success_loss = float("inf")
        best_success_delta = None
        best_success_state = None
        normalized_reference = None
        static_weights = None
        updates_done = 0

        for state in range(steps + 1):
            adversarial_wav = wav + delta
            evaluation = self._evaluate_state(
                adversarial_wav,
                wav,
                target_text,
                delta,
                reference_refusal,
                normalized_reference,
                static_weights,
            )
            normalized_reference = evaluation.reference_refusal
            static_weights = evaluation.static_weights

            loss_value = float(evaluation.total_loss.detach().item())
            target_value = float(evaluation.target_loss.detach().item())
            state_value = float(evaluation.state_loss.detach().item())
            utility_value = float(evaluation.utility_loss.detach().item())
            current_delta = delta.detach().clone()
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_delta = current_delta
                best_state = state

            linf = float(current_delta.abs().max().item())
            rms = float(current_delta.square().mean().sqrt().item())
            snr = self.compute_snr(wav, adversarial_wav.detach())
            record: Dict[str, Any] = {
                "state": state,
                "step": state,
                "updates_completed": state,
                "loss": loss_value,
                "target_loss": target_value,
                "output_loss": target_value,
                "state_loss": state_value,
                "utility_loss": utility_value,
                "layer_weights": _float_map(evaluation.layer_weights),
                "refusal_scores": _float_map(evaluation.refusal_scores),
                "harmfulness_scores": _float_map(evaluation.harmfulness_scores),
                "harmfulness_directions": _float_map(evaluation.harmfulness_directions),
                "refusal_directions": _float_map(evaluation.refusal_directions),
                "safety_gaps": _float_map(evaluation.safety_gaps),
                "gradient_scores": _float_map(evaluation.gradient_scores),
                "bottleneck_layer": evaluation.bottleneck_layer,
                "bottleneck_gap": (
                    None
                    if evaluation.bottleneck_gap is None
                    else float(evaluation.bottleneck_gap.detach().item())
                ),
                "linf": linf,
                "delta_linf": linf,
                "delta_rms": rms,
                "snr_db": float(snr) if math.isfinite(snr) else None,
                "is_best": is_best,
                "best_loss": best_loss,
            }
            history["iterations"].append(record)
            if normalized_reference is not None:
                history["reference_refusal"] = _float_map(normalized_reference)

            if state_callback is not None and (
                checkpoint_set is None or state in checkpoint_set
            ):
                callback_record = dict(record)
                snapshot_tensors: Dict[str, torch.Tensor] = {}
                for namespace, values in (
                    ("hidden_states", evaluation.hidden_states),
                    ("harmfulness_scores", evaluation.harmfulness_scores),
                    ("refusal_scores", evaluation.refusal_scores),
                    ("harmfulness_directions", evaluation.harmfulness_directions),
                    ("refusal_directions", evaluation.refusal_directions),
                ):
                    if values is None:
                        continue
                    for layer, value in values.items():
                        snapshot_tensors[f"{namespace}.{layer}"] = value.detach()
                if snapshot_tensors:
                    callback_record["_snapshot_tensors"] = snapshot_tensors
                state_callback(
                    state,
                    (wav + current_delta).detach(),
                    current_delta,
                    callback_record,
                )

            should_check = (
                state == 0
                or state == steps
                or (check_every > 0 and state > 0 and state % check_every == 0)
            )
            if should_check:
                if state == 0 and init_mode == "zero":
                    output = original_output
                else:
                    with torch.no_grad():
                        output = self.model.generate(
                            wav + current_delta, do_sample=False
                        )
                decision_now = evaluate_behavior(
                    target_text, output, behavior_evaluator
                )
                success_now = decision_now.attack_success
                history["generation_checks"].append(
                    {
                        "step": state,
                        "output": output,
                        "success": success_now,
                        "behavior": decision_now.as_dict(),
                    }
                )
                if success_now and history["first_success_step"] is None:
                    history["first_success_step"] = state
                if success_now and loss_value < best_success_loss:
                    best_success_loss = loss_value
                    best_success_delta = current_delta
                    best_success_state = state
                if success_now and early_stop:
                    updates_done = state
                    break

            if state == steps:
                updates_done = state
                break

            if not evaluation.total_loss.requires_grad:
                raise RuntimeError("attack loss is not differentiable with respect to delta")
            gradient = torch.autograd.grad(
                evaluation.total_loss,
                delta,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if gradient is None:
                raise RuntimeError("PGD perturbation gradient is missing")
            gradient = self.process_gradient(gradient.detach())
            with torch.no_grad():
                delta -= self.alpha * gradient.sign()
            delta = self.clamp_perturbation(delta, wav)
            delta.grad = None
            updates_done = state + 1

        if best_success_delta is not None:
            selected_delta = best_success_delta
            selected_state = best_success_state
            selected_loss = best_success_loss
            selection_reason = "successful_candidate"
        else:
            selected_delta = best_delta
            selected_state = best_state
            selected_loss = best_loss
            selection_reason = "lowest_loss"
        if selected_delta is None or selected_state is None:
            raise RuntimeError("attack did not produce a selectable candidate")

        final_wav = wav + selected_delta
        with torch.no_grad():
            final_output = self.model.generate(final_wav, do_sample=False)
        final_decision = evaluate_behavior(
            target_text, final_output, behavior_evaluator
        )
        success = final_decision.attack_success
        history["selection"] = {
            "state": selected_state,
            "step": selected_state,
            "reason": selection_reason,
            "recorded_loss": selected_loss,
            "final_loss": selected_loss,
            "success": success,
            "behavior": final_decision.as_dict(),
        }

        if self.safety_scorer is not None and hasattr(
            self.safety_scorer, "zero_grad"
        ):
            self.safety_scorer.zero_grad(set_to_none=True)

        return AttackResult(
            original_wav=wav.detach().cpu(),
            adversarial_wav=final_wav.detach().cpu(),
            perturbation=selected_delta.detach().cpu(),
            original_output=original_output,
            adversarial_output=final_output,
            target_text=target_text,
            success=success,
            final_loss=float(selected_loss),
            steps_taken=updates_done,
            history=history,
        )


# A descriptive alias for experiment code that names the proposed method rather
# than the shared ablation engine.
SafetyStateAdaptivePGDAttacker = LayerAdaptivePGDAttacker


__all__ = [
    "LayerAdaptivePGDAttacker",
    "SafetyStateAdaptivePGDAttacker",
    "SUPPORTED_METHODS",
]
