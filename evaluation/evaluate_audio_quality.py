#!/usr/bin/env python3
"""Recompute waveform quality and L-infinity validity from run artifacts.

SPR is signal-to-perturbation ratio:
10*log10(mean(clean**2)/mean((adversarial-clean)**2)). It is the same
waveform quantity as SNR, so spr_db is an explicit alias of snr_db.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from evaluation.perceptual import compute_perceptual_metrics


SPR_DEFINITION = (
    "10*log10(mean(clean^2)/mean((adversarial-clean)^2)); "
    "signal-to-perturbation ratio, identical to waveform SNR"
)


class AudioBudgetViolationError(ValueError):
    """Strict budget verification failure."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(value), handle, ensure_ascii=False, indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _default_audio_loader(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load stored amplitude without the project loader's peak normalization."""

    try:
        import soundfile as sf
    except ImportError:
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "Install soundfile/torchaudio or inject audio_loader"
            ) from exc
        waveform, sample_rate = torchaudio.load(str(path))
    else:
        samples, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(samples.T.copy())
    return waveform, int(sample_rate)


def _normalize_loaded_audio(value: Any, *, path: Path) -> tuple[torch.Tensor, int]:
    if isinstance(value, Mapping):
        waveform = value.get("waveform", value.get("audio"))
        sample_rate = value.get("sample_rate", value.get("sr"))
    elif isinstance(value, tuple) and len(value) == 2:
        waveform, sample_rate = value
    else:
        raise TypeError(
            f"{path}: loader must return (waveform, sample_rate) or a mapping"
        )
    waveform = torch.as_tensor(waveform)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"{path}: waveform must have [channels, samples] shape")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"{path}: waveform must be non-empty and finite")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError(f"{path}: sample rate must be an integer")
    if sample_rate <= 0:
        raise ValueError(f"{path}: sample rate must be positive")
    return waveform.detach().cpu(), sample_rate


def discover_run_files(path: str | Path) -> list[Path]:
    root = Path(path).resolve()
    if root.is_file():
        if root.name not in {"run.json", "config.json"}:
            raise ValueError("Run artifact must be run.json or config.json")
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    direct = root / "run.json"
    if direct.is_file():
        return [direct]
    paths = sorted(root.glob("*/run.json")) or sorted(root.rglob("run.json"))
    if not paths:
        raise ValueError(f"No run.json artifacts found below {root}")
    return paths


def _read_run(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        run = json.load(handle)
    if not isinstance(run, Mapping):
        raise ValueError(f"{path}: run must be an object")
    for field in ("case_id", "pair_id", "method", "budget", "artifacts"):
        if field not in run:
            raise ValueError(f"{path}: missing {field!r}")
    if not isinstance(run["budget"], Mapping) or not isinstance(
        run["artifacts"], Mapping
    ):
        raise ValueError(f"{path}: budget and artifacts must be objects")
    return run


def _artifact_path(
    run_path: Path, artifacts: Mapping[str, Any], aliases: Sequence[str]
) -> Path:
    values = [artifacts[name] for name in aliases if artifacts.get(name)]
    if not values:
        raise ValueError(f"{run_path}: missing {'/'.join(aliases)}")
    if len({str(value) for value in values}) != 1:
        raise ValueError(f"{run_path}: conflicting artifact aliases")
    path = Path(str(values[0])).expanduser()
    if not path.is_absolute():
        path = run_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio artifact not found: {path}")
    return path


def verify_linf_budget(
    clean: Any, adversarial: Any, eps: float, *, atol: float = 1e-6
) -> dict[str, Any]:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a real scalar")
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")
    if not math.isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    clean = torch.as_tensor(clean, dtype=torch.float64)
    adversarial = torch.as_tensor(adversarial, dtype=torch.float64)
    if clean.shape != adversarial.shape:
        raise ValueError("clean and adversarial shapes differ")
    if clean.numel() == 0:
        raise ValueError("waveforms cannot be empty")
    if not bool(torch.isfinite(clean).all()) or not bool(
        torch.isfinite(adversarial).all()
    ):
        raise ValueError("waveforms must be finite")
    measured = float((adversarial - clean).abs().max().item())
    return {
        "norm": "linf",
        "eps": eps,
        "atol": float(atol),
        "measured_linf": measured,
        "valid": measured <= eps + atol,
        "violation": max(0.0, measured - eps),
    }


def evaluate_run_audio_quality(
    run_artifact: str | Path,
    *,
    audio_loader: Optional[Callable[[str | Path], Any]] = None,
    include_pesq: bool = False,
    include_stoi: bool = False,
    pesq_mode: Optional[str] = None,
    extended_stoi: bool = False,
    budget_atol: float = 1e-6,
    strict_budget: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_artifact).resolve()
    run = _read_run(run_path)
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
    if clean.shape != adversarial.shape:
        raise ValueError("clean/adversarial waveform shapes differ")
    budget = run["budget"]
    norm = str(budget.get("norm", "linf")).casefold()
    if norm not in {"linf", "l_inf", "l-infinity", "infinity"}:
        raise ValueError(f"Expected L-infinity budget, got {norm!r}")
    if "eps" not in budget:
        raise ValueError(f"{run_path}: budget lacks eps")
    verification = verify_linf_budget(
        clean, adversarial, budget["eps"], atol=budget_atol
    )
    if strict_budget and not verification["valid"]:
        raise AudioBudgetViolationError(
            f"measured L-inf {verification['measured_linf']:.9g} exceeds "
            f"eps {verification['eps']:.9g}"
        )
    metrics = compute_perceptual_metrics(
        clean,
        adversarial,
        sample_rate=clean_rate,
        include_pesq=include_pesq,
        include_stoi=include_stoi,
        pesq_mode=pesq_mode,
        extended_stoi=extended_stoi,
    )
    metrics["spr_db"] = metrics["snr_db"]
    return {
        "case_id": str(run["case_id"]),
        "pair_id": str(run["pair_id"]),
        "method": str(run["method"]),
        "model": str(run.get("model", "unknown")),
        "seed": budget.get("seed"),
        "attack_success": run.get("attack_success"),
        "sample_rate": clean_rate,
        "num_samples": int(clean.shape[-1]),
        "clean_audio": str(clean_path),
        "adversarial_audio": str(adversarial_path),
        "metrics": metrics,
        "spr_definition": SPR_DEFINITION,
        "budget_verification": verification,
        "source_path": str(run_path),
    }


def _summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted({name for case in cases for name in case["metrics"]})
    aggregates = {}
    for name in names:
        values = [
            float(case["metrics"][name]) for case in cases
            if name in case["metrics"]
            and math.isfinite(float(case["metrics"][name]))
        ]
        aggregates[name] = {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "total": len(cases),
        "budget_valid": sum(c["budget_verification"]["valid"] for c in cases),
        "budget_violations": sum(
            not c["budget_verification"]["valid"] for c in cases
        ),
        "metrics": aggregates,
    }


def evaluate_audio_quality(
    runs: str | Path | Iterable[str | Path],
    *,
    audio_loader: Optional[Callable[[str | Path], Any]] = None,
    include_pesq: bool = False,
    include_stoi: bool = False,
    pesq_mode: Optional[str] = None,
    extended_stoi: bool = False,
    budget_atol: float = 1e-6,
    strict_budget: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
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
                evaluate_run_audio_quality(
                    path,
                    audio_loader=audio_loader,
                    include_pesq=include_pesq,
                    include_stoi=include_stoi,
                    pesq_mode=pesq_mode,
                    extended_stoi=extended_stoi,
                    budget_atol=budget_atol,
                    strict_budget=strict_budget,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            failures.append(
                {"source_path": str(path), "error_type": type(exc).__name__,
                 "error": str(exc)}
            )
    return {
        "format": "audio-quality-evaluation",
        "version": 1,
        "spr_definition": SPR_DEFINITION,
        "cases": cases,
        "failures": failures,
        "failure_count": len(failures),
        "summary": _summary(cases),
    }


evaluate_audio_quality_run = evaluate_run_audio_quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-pesq", action="store_true")
    parser.add_argument("--include-stoi", action="store_true")
    parser.add_argument("--pesq-mode", choices=("nb", "wb"))
    parser.add_argument("--extended-stoi", action="store_true")
    parser.add_argument("--budget-atol", type=float, default=1e-6)
    parser.add_argument("--strict-budget", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_audio_quality(
        args.runs,
        include_pesq=args.include_pesq,
        include_stoi=args.include_stoi,
        pesq_mode=args.pesq_mode,
        extended_stoi=args.extended_stoi,
        budget_atol=args.budget_atol,
        strict_budget=args.strict_budget,
        fail_fast=args.fail_fast,
    )
    _atomic_json(Path(args.output).resolve(), result)
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AudioBudgetViolationError",
    "SPR_DEFINITION",
    "build_parser",
    "discover_run_files",
    "evaluate_audio_quality",
    "evaluate_audio_quality_run",
    "evaluate_run_audio_quality",
    "main",
    "verify_linf_budget",
]
