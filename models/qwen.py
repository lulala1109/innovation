"""
Qwen 2.5 Omni audio model wrapper for adversarial attacks.

Uses the EMBEDDINGS-BASED approach (same as safety-bypass/utils/mel_attacks.py):
1. WAV -> MEL spectrogram (differentiable)
2. MEL -> compressed audio features via get_audio_features() (~4x smaller)
3. Audio features + text embeddings -> inputs_embeds
4. Forward pass with inputs_embeds (NOT input_features)

This approach is much more memory-efficient than passing raw MEL directly:
- No fixed 30000 frame padding needed
- Smaller computation graph
- Works with gradient checkpointing for ~50% memory reduction
"""

import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Iterable, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from models.base import AttackForwardOutput, BaseAudioModel


class QwenMelTransform(torch.nn.Module):
    """
    Differentiable Whisper/Qwen log-MEL frontend.

    It uses the feature extractor configuration and MEL filterbank shipped with
    the selected Qwen checkpoint. Short audio is processed without constructing
    the full 300-second differentiable graph, while retaining the same real MEL
    frames and feature lengths as the official processor.
    """

    def __init__(
        self,
        processor,
        device: str = "cuda"
    ):
        super().__init__()

        feature_extractor = processor.feature_extractor
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.max_samples = feature_extractor.n_samples
        self.max_frames = feature_extractor.nb_max_frames
        self.padding_value = feature_extractor.padding_value

        if feature_extractor.dither != 0.0:
            raise ValueError(
                "Differentiable Qwen MEL currently requires dither=0.0"
            )

        # Buffers stay fixed while gradients flow through them back to the WAV.
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(feature_extractor.mel_filters).float()
        )  # (201, 128)
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft)
        )
        self.to(device)

    def forward(
        self,
        waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert one waveform to official-normalized log-MEL features.

        Args:
            waveform: Audio tensor shaped [T] or [1, T].

        Returns:
            mel: Log-MEL tensor [1, 128, real_frames].
            feature_attention_mask: Ones shaped [1, real_frames].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError("waveform must have shape [T] or [1, T]")
        if waveform.shape[0] != 1:
            raise ValueError(
                "The current Qwen embedding path supports batch_size=1 only"
            )

        waveform = waveform.to(
            device=self.mel_filters.device,
            dtype=torch.float32
        )
        num_samples = waveform.shape[-1]
        if num_samples < 1:
            raise ValueError("waveform must contain at least one sample")
        if num_samples > self.max_samples:
            raise ValueError(
                f"waveform has {num_samples} samples, exceeding the "
                f"processor limit of {self.max_samples}"
            )

        # Official rescaled attention-mask length: ceil(samples / hop_length).
        real_frames = (
            num_samples + self.hop_length - 1
        ) // self.hop_length

        # Official preprocessing pads short WAVs with zeros before STFT. One
        # n_fft tail covers all windows that can overlap real samples without
        # constructing the full 300-second differentiable graph.
        if num_samples < self.max_samples:
            stft_waveform = F.pad(
                waveform,
                (0, self.n_fft),
                mode="constant",
                value=self.padding_value
            )
        else:
            stft_waveform = waveform

        stft = torch.stft(
            stft_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True
        )

        # Whisper removes the final STFT frame and uses a power spectrogram.
        magnitudes = stft[..., :-1].abs().square()  # [1, 201, frames]

        # mel_filters is [frequency, mel].
        mel = torch.einsum(
            "fm,bft->bmt",
            self.mel_filters,
            magnitudes
        )

        # Official Whisper log-MEL normalization, independently per sample.
        log_spec = mel.clamp_min(1e-10).log10()
        max_value = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_value - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Qwen discards masked padding before the audio tower. Keeping only
        # real frames is equivalent and substantially cheaper for PGD.
        log_spec = log_spec[..., :real_frames]
        feature_attention_mask = torch.ones(
            (waveform.shape[0], real_frames),
            dtype=torch.long,
            device=waveform.device
        )

        return log_spec, feature_attention_mask


class QwenModel(BaseAudioModel):
    """
    Wrapper for Qwen 2.5 Omni model with audio capabilities.

    Qwen takes MEL spectrograms through its model-specific audio frontend.

    Implements the BaseAudioModel interface for use with attack algorithms.
    """

    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
    SAMPLE_RATE = 16000

    # Default system prompt for Qwen
    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, "
        "Alibaba Group, capable of perceiving auditory and visual inputs, "
        "as well as generating text and speech."
    )

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: Optional[str] = None
    ):
        """
        Initialize the Qwen model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run on
            dtype: Model dtype
            token: HuggingFace token (or uses env var)
        """
        self.model_id = model_id
        self._device = device
        self._dtype = dtype

        # Get HF token
        if token is None:
            token = os.getenv(
                "HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

        print(f"Loading Qwen model: {model_id}")
        print(f"Device: {device}, Dtype: {dtype}")

        # Load model and processor
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id, token=token)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=token
        ).eval()

        # Disable talker to save memory (we only need text output)
        if hasattr(self.model, 'disable_talker'):
            self.model.disable_talker()

        # PGD冻结模型参数， only optimizes the audio perturbation, not model weights
        self.model.requires_grad_(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Enable gradient checkpointing to reduce memory usage
        # Trades compute for memory by recomputing activations during backward
        self.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (memory optimization)")

        # Get the thinker for inference (text generation)
        self.thinker = self.model.thinker
        self.tokenizer = self.processor.tokenizer

        # Initialize differentiable MEL transform using processor's exact filterbank
        self.mel_transform = QwenMelTransform(
            processor=self.processor,
            device=device
        )

        # Special token IDs for embeddings approach
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.audio_bos_id = self.tokenizer.convert_tokens_to_ids(
            "<|audio_bos|>")
        self.audio_eos_id = self.tokenizer.convert_tokens_to_ids(
            "<|audio_eos|>")

        print("Qwen model loaded successfully!")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @staticmethod
    def _resolve_module_path(root: Any, path: str) -> Any:
        value = root
        for component in path.split("."):
            if not hasattr(value, component):
                return None
            value = getattr(value, component)
        return value

    @property
    def transformer_layer_modules(self) -> "OrderedDict[int, torch.nn.Module]":
        """Return the Thinker text-decoder blocks with stable zero-based keys.

        Qwen2.5-Omni also contains audio-encoder and (unless disabled) talker
        layers. Activation patching must target the Thinker text decoder, not
        whichever ModuleList happens to be found first by introspection.
        The first path is the layout used by current Transformers releases;
        the remaining paths cover older compatible wrappers explicitly.
        """

        candidate_paths = (
            "model.layers",
            "model.language_model.layers",
            "model.language_model.model.layers",
            "language_model.layers",
            "language_model.model.layers",
            "model.model.layers",
        )
        for path in candidate_paths:
            layers = self._resolve_module_path(self.thinker, path)
            if layers is None:
                continue
            try:
                values = tuple(layers)
            except TypeError:
                continue
            if values and all(isinstance(layer, torch.nn.Module) for layer in values):
                return OrderedDict(enumerate(values))
        raise RuntimeError(
            "Could not locate Qwen Thinker text-decoder layers. Expected one "
            "of: " + ", ".join(f"thinker.{path}" for path in candidate_paths)
        )

    def get_transformer_layer_modules(
        self,
        layers: Optional[Iterable[int]] = None,
    ) -> "OrderedDict[int, torch.nn.Module]":
        """Select Thinker decoder modules using the project's layer semantics."""

        available = self.transformer_layer_modules
        if layers is None:
            return available
        from core.activations import resolve_layer_indices

        selected = resolve_layer_indices(len(available), tuple(layers))
        return OrderedDict((index, available[index]) for index in selected)

    def wav_to_mel_exact(
        self,
        wav: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert WAV to differentiable processor-aligned MEL and frame mask.
        """
        wav = wav.to(self._device, dtype=torch.float32)
        return self.mel_transform(wav)

    def prepare_audio_prompt(self, wav: torch.Tensor) -> dict[str, Any]:
        """Build one reusable prompt for forward/generation and hook patching.

        token_spans contains half-open prompt indices. Keeping this beside
        the embeddings lets a patching caller align X_H and X_J audio tokens
        before installing a hook, instead of assuming equal sequence lengths.
        """

        wav = wav.to(self._device, dtype=torch.float32)
        mel, feature_mask = self.wav_to_mel_exact(wav)
        inputs_embeds, token_spans = self._create_embeddings_from_mel(
            mel,
            feature_mask,
            return_metadata=True,
        )
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "token_spans": dict(token_spans),
        }

    @staticmethod
    def _validate_prepared_prompt(prompt: Mapping[str, Any]) -> None:
        if not isinstance(prompt, Mapping):
            raise TypeError("prepared prompt must be a mapping")
        embeds = prompt.get("inputs_embeds")
        mask = prompt.get("attention_mask")
        spans = prompt.get("token_spans")
        if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
            raise ValueError("prepared prompt inputs_embeds must have shape [B, T, D]")
        if (
            not isinstance(mask, torch.Tensor)
            or tuple(mask.shape) != tuple(embeds.shape[:2])
        ):
            raise ValueError("prepared prompt attention_mask must have shape [B, T]")
        if not isinstance(spans, Mapping):
            raise TypeError("prepared prompt token_spans must be a mapping")

    def forward_prepared_prompt(
        self,
        prompt: Mapping[str, Any],
        *,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> Any:
        """Run the Thinker on a prepared prompt while ordinary hooks stay live."""

        self._validate_prepared_prompt(prompt)
        return self.thinker(
            inputs_embeds=prompt["inputs_embeds"],
            attention_mask=prompt["attention_mask"],
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
        )

    def collect_layer_activations(
        self,
        wav: Optional[torch.Tensor] = None,
        *,
        layers: Optional[Iterable[int]] = None,
        prepared_prompt: Optional[Mapping[str, Any]] = None,
        output_selector: Optional[Union[int, str]] = None,
        detach: bool = True,
    ) -> "OrderedDict[int, torch.Tensor]":
        """Collect unpooled [batch, tokens, hidden] decoder activations.

        This deliberately captures module outputs instead of using a pooled
        safety-state view: causal patching needs the original token axis. A
        caller may pass a prepared prompt so collection and subsequent span
        alignment are guaranteed to refer to the same embeddings.
        """

        if prepared_prompt is None:
            if wav is None:
                raise ValueError("wav is required when prepared_prompt is omitted")
            prepared_prompt = self.prepare_audio_prompt(wav)
        elif wav is not None:
            raise ValueError("provide either wav or prepared_prompt, not both")
        self._validate_prepared_prompt(prepared_prompt)

        from core.activations import ForwardActivationCollector

        modules = self.get_transformer_layer_modules(layers)
        collector = ForwardActivationCollector(
            modules,
            pooling="none",
            output_selector=output_selector,
            detach=detach,
        )
        gradient_context = torch.no_grad() if detach else nullcontext()
        with gradient_context, collector:
            self.forward_prepared_prompt(
                prepared_prompt,
                output_hidden_states=False,
                use_cache=False,
            )
        missing = [layer for layer in modules if layer not in collector.activations]
        if missing:
            raise RuntimeError(f"Qwen forward did not visit decoder layer(s): {missing}")
        return OrderedDict(collector.activations)

    def generate_from_prepared_prompt(
        self,
        prompt: Mapping[str, Any],
        *,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """Generate from a prepared prompt without bypassing decoder hooks."""

        self._validate_prepared_prompt(prompt)
        with torch.no_grad():
            gen_ids = self.thinker.generate(
                inputs_embeds=prompt["inputs_embeds"],
                attention_mask=prompt["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True
            ).strip()

    def _create_embeddings_from_mel(
        self,
        mel: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        return_metadata: bool = False,
    ):
        """
        Create input embeddings from MEL spectrogram using get_audio_features().

        This is the KEY memory optimization:
        - Instead of passing raw MEL (30000 frames) directly as input_features
        - We compress to audio features (~4x smaller) using get_audio_features()
        - Then concatenate with text embeddings

        Args:
            mel: Log-mel spectrogram [1, n_mels, time_frames]
            feature_attention_mask: Valid MEL frames [1, time_frames]

        Returns:
            Combined embeddings [1, seq_len, hidden_dim]
        """
        if mel.shape[0] != 1:
            raise ValueError(
                "The current Qwen embedding path supports batch_size=1 only"
            )
        if feature_attention_mask.shape != (mel.shape[0], mel.shape[2]):
            raise ValueError(
                "feature_attention_mask must match MEL batch/time dimensions"
            )
        feature_attention_mask = feature_attention_mask.to(
            device=self._device,
            dtype=torch.long
        )

        # Get compressed audio features (KEY STEP!)
        # This compresses the MEL to ~4x smaller representation
        audio_features = self.thinker.get_audio_features(
            input_features=mel.to(self._dtype),
            feature_attention_mask=feature_attention_mask
        ).unsqueeze(0)  # Shape: (1, 1, seq_len, hidden_dim)

        # Squeeze the extra dimension if needed
        if audio_features.dim() == 4:
            audio_features = audio_features.squeeze(
                1)  # (1, seq_len, hidden_dim)

        # Create text parts using Qwen chat format
        system_part = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
        user_part_prefix = f"<|im_start|>user\n"
        user_part_suffix = f"<|im_end|>\n"
        assistant_cue = "<|im_start|>assistant\n"

        txt_embeds = self.thinker.get_input_embeddings()

        # Tokenize text parts
        sys_ids = self.tokenizer(
            [system_part], return_tensors="pt").input_ids.to(self._device)
        user_prefix_ids = self.tokenizer(
            [user_part_prefix], return_tensors="pt").input_ids.to(self._device)
        user_suffix_ids = self.tokenizer(
            [user_part_suffix], return_tensors="pt").input_ids.to(self._device)
        assist_cue_ids = self.tokenizer(
            [assistant_cue], return_tensors="pt").input_ids.to(self._device)

        # Create audio token embeddings
        audio_bos_embed = txt_embeds(torch.tensor(
            [[self.audio_bos_id]], device=self._device))
        audio_eos_embed = txt_embeds(torch.tensor(
            [[self.audio_eos_id]], device=self._device))

        # Concatenate all embeddings
        inputs_embeds = torch.cat([
            txt_embeds(sys_ids),
            txt_embeds(user_prefix_ids),
            audio_bos_embed,
            audio_features,
            audio_eos_embed,
            txt_embeds(user_suffix_ids),
            txt_embeds(assist_cue_ids),
        ], dim=1)

        if return_metadata:
            audio_start = (
                sys_ids.shape[1] + user_prefix_ids.shape[1] + 1
            )
            return inputs_embeds, {
                "audio": (audio_start, audio_start + audio_features.shape[1]),
            }

        return inputs_embeds

    def _create_inputs(
        self,
        mel: torch.Tensor,
        target_text: str = "",
        audio_length: Optional[int] = None
    ) -> dict:
        """
        Create model inputs from MEL spectrogram.

        NOTE: This method is DEPRECATED for loss computation.
        Use _create_embeddings_from_mel() + forward with inputs_embeds instead.
        Kept for backward compatibility with generation code.

        Args:
            mel: [1, n_mels, time] MEL spectrogram (padded)
            target_text: Target text to append for loss computation
            audio_length: Actual audio length in frames (for attention mask)

        Returns:
            dict with input_ids, attention_mask, input_features, etc.
        """
        mel_length = mel.shape[2]
        if audio_length is None:
            audio_length = mel_length

        # Calculate number of audio tokens (4 frames per token)
        num_audio_tokens = audio_length // 4

        # Create conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio_url": "placeholder"}]
            },
        ]

        # Apply chat template
        chat_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # Expand <|AUDIO|> token based on audio length
        audio_token = "<|AUDIO|>"
        expanded_audio = audio_token * num_audio_tokens
        chat_text = chat_text.replace(audio_token, expanded_audio)

        # Add target text if provided (for loss computation)
        full_text = chat_text + target_text

        # Tokenize
        tokens = self.processor.tokenizer(full_text, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.model.device)
        attention_mask = tokens.attention_mask.to(self.model.device)

        # Create feature attention mask (1 for real audio, 0 for padding)
        feature_attention_mask = torch.zeros(
            1, mel_length, dtype=torch.int32, device=self.model.device
        )
        feature_attention_mask[:, :audio_length] = 1

        # Create labels if target text provided
        labels = None
        if target_text:
            labels = input_ids.clone()
            target_tokens = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids
            target_len = target_tokens.shape[1]
            # Mask everything except target tokens
            labels[:, :-target_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": mel.to(self._dtype),
            "feature_attention_mask": feature_attention_mask,
            "labels": labels
        }

    def compute_loss(
        self,
        wav: torch.Tensor,
        target_text: str
    ) -> torch.Tensor:
        """Compute differentiable target-text cross entropy."""
        return self.forward_attack(wav, target_text).loss

    def forward_attack(
        self,
        wav: torch.Tensor,
        target_text: str,
        *,
        output_hidden_states: bool = False,
    ) -> AttackForwardOutput:
        """Return output loss and optional layer states for one WAV forward."""
        wav = wav.to(self._device, dtype=torch.float32)
        mel, feature_mask = self.wav_to_mel_exact(wav)
        prompt_embeds, token_spans = self._create_embeddings_from_mel(
            mel,
            feature_mask,
            return_metadata=True,
        )

        txt_embeds = self.thinker.get_input_embeddings()
        target_ids = self.tokenizer(
            target_text,
            return_tensors="pt",
        ).input_ids.to(self._device)
        target_embeds = txt_embeds(target_ids)
        target_start = prompt_embeds.shape[1]
        inputs_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)

        labels = torch.full(
            (1, inputs_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=self._device,
        )
        labels[:, target_start:] = target_ids
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            device=self._device,
            dtype=torch.long,
        )
        token_spans["target"] = (
            target_start,
            target_start + target_ids.shape[1],
        )

        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = (
            tuple(outputs.hidden_states)
            if output_hidden_states and outputs.hidden_states is not None
            else None
        )
        return AttackForwardOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels,
            token_spans=token_spans,
            raw_output=outputs,
        )

    def compute_margin_loss(
        self,
        wav: torch.Tensor,
        target_text: str,
        kappa: float = 5.0,
        early_weight: float = 5.0
    ) -> torch.Tensor:
        """Compute a token-level Carlini-Wagner margin loss."""
        forward = self.forward_attack(wav, target_text)
        if forward.logits is None or forward.labels is None:
            return forward.loss

        shift_logits = forward.logits[:, :-1, :].contiguous()
        shift_labels = forward.labels[:, 1:].contiguous()
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return forward.loss

        valid_positions = valid_mask[0].nonzero(as_tuple=True)[0]
        valid_logits = shift_logits[0, valid_positions]
        valid_labels = shift_labels[0, valid_positions]
        target_logits = valid_logits.gather(
            1,
            valid_labels.unsqueeze(1),
        ).squeeze(1)

        label_mask = torch.ones_like(valid_logits, dtype=torch.bool)
        label_mask.scatter_(1, valid_labels.unsqueeze(1), False)
        top_other_logits = valid_logits.masked_fill(
            ~label_mask,
            float("-inf"),
        ).max(dim=-1).values
        margin_losses = torch.clamp(
            top_other_logits - target_logits + kappa,
            min=0,
        )

        weights = torch.ones_like(margin_losses)
        weights[:min(3, len(weights))] = early_weight
        return (margin_losses * weights).sum() / weights.sum()

    def generate(
        self,
        wav: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from audio using Qwen with EMBEDDINGS approach.

        Args:
            wav: Audio waveform tensor [T] or [1, T]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            Generated text string
        """
        with torch.no_grad():
            prompt = self.prepare_audio_prompt(wav)
        return self.generate_from_prepared_prompt(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
