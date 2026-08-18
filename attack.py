#!/usr/bin/env python3
"""Run one reproducible waveform-bounded safety-state experiment.

The single-case entry point writes a one-row manifest and delegates execution
to experiments.batch_safety_attack, so single and batch runs share exactly the
same attack implementation, artifact schema, checkpoint semantics, and resume
rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from experiments.batch_safety_attack import METHOD_CHOICES, run_batch


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            int(piece.strip()) for piece in value.split(",") if piece.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
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


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_pair_id(
    wav_path: Path, harmful_text: str, stratum: str
) -> str:
    digest = hashlib.sha256()
    digest.update(_file_digest(wav_path).encode("ascii"))
    digest.update(b"\0")
    digest.update(harmful_text.strip().encode("utf-8"))
    digest.update(b"\0")
    digest.update(stratum.strip().encode("utf-8"))
    return f"pair_{digest.hexdigest()[:20]}"


def run_single(
    wav_path: str | Path,
    output_dir: str | Path,
    *,
    harmful_text: str,
    target_text: str,
    pair_id: Optional[str] = None,
    case_id: Optional[str] = None,
    stratum: str = "single",
    **batch_options: Any,
) -> dict[str, Any]:
    """Write a reproducible one-case manifest and run the shared batch engine."""

    audio = Path(wav_path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {audio}")
    if not harmful_text.strip():
        raise ValueError("harmful_text cannot be blank")
    if not target_text.strip():
        raise ValueError("target_text cannot be blank")
    if not stratum.strip():
        raise ValueError("stratum cannot be blank")

    resolved_pair_id = (
        pair_id.strip()
        if pair_id is not None and pair_id.strip()
        else _derive_pair_id(audio, harmful_text, stratum)
    )
    resolved_case_id = (
        case_id.strip()
        if case_id is not None and case_id.strip()
        else f"single_{resolved_pair_id.removeprefix('pair_')}"
    )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "single_manifest.json"
    _atomic_json(
        manifest_path,
        [
            {
                "case_id": resolved_case_id,
                "pair_id": resolved_pair_id,
                "stratum": stratum.strip(),
                "harmful_text": harmful_text.strip(),
                "clean_audio_path": str(audio),
                "target_text": target_text.strip(),
            }
        ],
    )
    return run_batch(
        manifest_path,
        output,
        target_text=target_text.strip(),
        **batch_options,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, help="Normalized input audio")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--harmful-text", required=True)
    parser.add_argument("--target-text", required=True)
    parser.add_argument("--pair-id")
    parser.add_argument("--case-id")
    parser.add_argument("--stratum", default="single")

    parser.add_argument("--method", choices=METHOD_CHOICES, default="standard")
    parser.add_argument("--model", dest="model_name", default="qwen-3b")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )

    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--loss-type", choices=("ce", "margin"), default="margin"
    )
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument(
        "--init-mode", choices=("zero", "random"), default="zero"
    )
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--checkpoint-steps", type=_comma_ints)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--determinism",
        choices=("off", "warn", "strict"),
        default="warn",
    )

    parser.add_argument("--probe-checkpoint")
    parser.add_argument(
        "--reference-refusal", dest="reference_refusal_path"
    )
    parser.add_argument("--layers", type=_comma_ints)
    parser.add_argument("--fixed-layer", type=int)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--static-topk-layers", type=_comma_ints)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--token-span",
        choices=("audio", "target", "all"),
        default="audio",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "max", "first", "last"),
        default="mean",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = vars(build_parser().parse_args(argv))
    wav_path = args.pop("wav")
    output_dir = args.pop("output_dir")
    harmful_text = args.pop("harmful_text")
    target_text = args.pop("target_text")
    pair_id = args.pop("pair_id")
    case_id = args.pop("case_id")
    stratum = args.pop("stratum")
    args["early_stop"] = not args.pop("no_early_stop")

    summary = run_single(
        wav_path,
        output_dir,
        harmful_text=harmful_text,
        target_text=target_text,
        pair_id=pair_id,
        case_id=case_id,
        stratum=stratum,
        **args,
    )
    counts = summary["counts"]
    print(
        f"completed={counts['completed']} "
        f"failed={counts['failed']} skipped={counts['skipped']}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_single"]
