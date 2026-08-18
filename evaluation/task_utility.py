"""Speech task utility with dependency-free WER/CER and injected ASR."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from evaluation.evaluate_audio_quality import (
    _artifact_path,
    _default_audio_loader,
    _normalize_loaded_audio,
    _read_run,
    discover_run_files,
)


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""

    left, right = tuple(reference), tuple(hypothesis)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("transcript must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def word_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"\w+|[^\w\s]", normalize_text(text), flags=re.UNICODE)
    )


def character_tokens(
    text: str, *, ignore_whitespace: bool = True
) -> tuple[str, ...]:
    normalized = normalize_text(text)
    if ignore_whitespace:
        normalized = "".join(normalized.split())
    return tuple(normalized)


def _rate(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float:
    # Explicit empty-reference convention: empty/empty=0, otherwise insertions.
    return edit_distance(reference, hypothesis) / max(1, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    return _rate(word_tokens(reference), word_tokens(hypothesis))


def character_error_rate(
    reference: str, hypothesis: str, *, ignore_whitespace: bool = True
) -> float:
    return _rate(
        character_tokens(reference, ignore_whitespace=ignore_whitespace),
        character_tokens(hypothesis, ignore_whitespace=ignore_whitespace),
    )


wer = word_error_rate
cer = character_error_rate
compute_wer = word_error_rate
compute_cer = character_error_rate


def compute_transcript_metrics(
    reference: str,
    hypothesis: str,
    *,
    ignore_whitespace_for_cer: bool = True,
) -> dict[str, Any]:
    reference_words, hypothesis_words = word_tokens(reference), word_tokens(hypothesis)
    reference_chars = character_tokens(
        reference, ignore_whitespace=ignore_whitespace_for_cer
    )
    hypothesis_chars = character_tokens(
        hypothesis, ignore_whitespace=ignore_whitespace_for_cer
    )
    word_edits = edit_distance(reference_words, hypothesis_words)
    character_edits = edit_distance(reference_chars, hypothesis_chars)
    return {
        "wer": word_edits / max(1, len(reference_words)),
        "cer": character_edits / max(1, len(reference_chars)),
        "word_edits": word_edits,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "character_edits": character_edits,
        "reference_characters": len(reference_chars),
        "hypothesis_characters": len(hypothesis_chars),
        "empty_reference_convention": "denominator=max(1, reference_length)",
    }


def _transcribe(transcriber: Any, waveform: torch.Tensor, sample_rate: int) -> str:
    if transcriber is None:
        raise ValueError("A real transcriber must be injected")
    method = getattr(transcriber, "transcribe", None)
    if callable(method):
        result = method(waveform, sample_rate=sample_rate)
    elif callable(transcriber):
        result = transcriber(waveform, sample_rate)
    else:
        raise TypeError("transcriber must be callable or expose transcribe()")
    if isinstance(result, Mapping):
        if "text" not in result:
            raise ValueError("transcriber mapping must contain 'text'")
        result = result["text"]
    if not isinstance(result, str):
        raise TypeError("transcriber must return text or {'text': text}")
    return result


def evaluate_task_utility(
    reference_text: str,
    clean_audio: torch.Tensor,
    adversarial_audio: torch.Tensor,
    *,
    sample_rate: int,
    transcriber: Any,
) -> dict[str, Any]:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    clean_text = _transcribe(transcriber, clean_audio, sample_rate)
    adversarial_text = _transcribe(transcriber, adversarial_audio, sample_rate)
    return {
        "reference_text": reference_text,
        "clean_transcript": clean_text,
        "adversarial_transcript": adversarial_text,
        "clean": compute_transcript_metrics(reference_text, clean_text),
        "adversarial": compute_transcript_metrics(reference_text, adversarial_text),
        "adversarial_vs_clean": compute_transcript_metrics(
            clean_text, adversarial_text
        ),
    }


def evaluate_run_task_utility(
    run_artifact: str | Path,
    *,
    transcriber: Any,
    audio_loader: Optional[Callable[[str | Path], Any]] = None,
    reference_field: str = "harmful_text",
) -> dict[str, Any]:
    run_path = Path(run_artifact).resolve()
    run = _read_run(run_path)
    reference = run.get(reference_field)
    if not isinstance(reference, str):
        raise ValueError(f"{run_path}: {reference_field!r} must be text")
    clean_path = _artifact_path(
        run_path, run["artifacts"],
        ("input_audio", "clean_audio", "reference_audio"),
    )
    adversarial_path = _artifact_path(
        run_path, run["artifacts"],
        ("adversarial_audio", "adversarial"),
    )
    loader = audio_loader or _default_audio_loader
    clean, clean_rate = _normalize_loaded_audio(loader(clean_path), path=clean_path)
    adversarial, adversarial_rate = _normalize_loaded_audio(
        loader(adversarial_path), path=adversarial_path
    )
    if clean_rate != adversarial_rate:
        raise ValueError("clean/adversarial sample rates differ")
    result = evaluate_task_utility(
        reference, clean, adversarial,
        sample_rate=clean_rate, transcriber=transcriber,
    )
    result.update(
        case_id=str(run["case_id"]),
        pair_id=str(run["pair_id"]),
        method=str(run["method"]),
        seed=run["budget"].get("seed"),
        source_path=str(run_path),
    )
    return result


def evaluate_task_utility_batch(
    runs: str | Path | Iterable[str | Path],
    *,
    transcriber: Any,
    audio_loader: Optional[Callable[[str | Path], Any]] = None,
    reference_field: str = "harmful_text",
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Batch API with explicit failed-transcription accounting."""

    paths = (
        discover_run_files(runs)
        if isinstance(runs, (str, Path))
        else [Path(path).resolve() for path in runs]
    )
    if not paths:
        raise ValueError("runs cannot be empty")
    cases, failures = [], []
    for path in paths:
        try:
            cases.append(
                evaluate_run_task_utility(
                    path, transcriber=transcriber, audio_loader=audio_loader,
                    reference_field=reference_field,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            failures.append(
                {"source_path": str(path), "error_type": type(exc).__name__,
                 "error": str(exc)}
            )
    wers = [float(case["adversarial"]["wer"]) for case in cases]
    cers = [float(case["adversarial"]["cer"]) for case in cases]
    return {
        "format": "task-utility-evaluation",
        "version": 1,
        "cases": cases,
        "failures": failures,
        "failure_count": len(failures),
        "summary": {
            "completed": len(cases),
            "failed": len(failures),
            "mean_adversarial_wer": sum(wers) / len(wers) if wers else None,
            "mean_adversarial_cer": sum(cers) / len(cers) if cers else None,
        },
    }


batch_evaluate_task_utility = evaluate_task_utility_batch


__all__ = [
    "cer",
    "compute_cer",
    "compute_wer",
    "character_error_rate",
    "character_tokens",
    "compute_transcript_metrics",
    "edit_distance",
    "evaluate_run_task_utility",
    "evaluate_task_utility",
    "evaluate_task_utility_batch",
    "batch_evaluate_task_utility",
    "normalize_text",
    "wer",
    "word_error_rate",
    "word_tokens",
]
