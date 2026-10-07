"""Fetch the pinned AdvBench source and prepare unmodified harmful-goal audio.

The output is an *audio inventory*, not an RQ2 manifest or an eligibility
decision. In particular, the AdvBench ``target`` column is retained only as
source metadata and is never passed to TTS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import urllib.request
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADV_BENCH_COMMIT = "098262edf85f807224e70ecd87b9d83716bf6b73"
ADV_BENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    f"{ADV_BENCH_COMMIT}/data/advbench/harmful_behaviors.csv"
)
ADV_BENCH_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
EXPECTED_ROWS = 520
SOURCE_CSV = PROJECT_ROOT / "dataset/raw/advbench/harmful_behaviors.csv"
AUDIO_DIR = PROJECT_ROOT / "dataset/derived/advbench_audio/harmful_clean"
INVENTORY = PROJECT_ROOT / "dataset/processed/rq2/advbench_audio_inventory.jsonl"
SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 0.25
MAX_DURATION_SECONDS = 120.0
MIN_PEAK = 0.003
MIN_RMS = 0.0005


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "advbench-rq2-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_source(data: bytes, *, expected_sha256: str = ADV_BENCH_SHA256) -> list[dict[str, object]]:
    actual = sha256_bytes(data)
    if actual != expected_sha256:
        raise ValueError(f"AdvBench source SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("AdvBench source is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != ["goal", "target"]:
        raise ValueError(f"AdvBench columns must be exactly goal,target; got {reader.fieldnames!r}")
    records = list(reader)
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"AdvBench source must contain {EXPECTED_ROWS} records; got {len(records)}")
    result: list[dict[str, object]] = []
    for index, record in enumerate(records):
        goal, target = record.get("goal"), record.get("target")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"AdvBench goal is empty at source row {index}")
        if not isinstance(target, str):
            raise ValueError(f"AdvBench target is missing at source row {index}")
        result.append(
            {
                "pair_id": f"advbench_{index:04d}",
                "source_row_index": index,
                # CSV record number (header is record one), not a byte offset.
                "source_csv_line": index + 2,
                "harmful_text": goal,
                "source_target": target,
                "goal_sha256": sha256_bytes(goal.encode("utf-8")),
                "source_csv_sha256": actual,
            }
        )
    return result


def fetch_source(
    destination: Path = SOURCE_CSV,
    *,
    downloader: Callable[[str], bytes] = _download,
    expected_sha256: str = ADV_BENCH_SHA256,
) -> Path:
    """Download once, refuse to replace an existing mismatched source."""

    if destination.exists():
        parse_source(destination.read_bytes(), expected_sha256=expected_sha256)
        return destination
    data = downloader(ADV_BENCH_URL)
    parse_source(data, expected_sha256=expected_sha256)
    _atomic_bytes(destination, data)
    return destination


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"AdvBench output must remain inside project root: {path}") from exc


def _tts_version() -> str:
    try:
        return version("gTTS")
    except PackageNotFoundError:
        return "unavailable"


def validate_audio_qa(path: Path) -> dict[str, object]:
    """Check container, duration, finite samples and audible energy."""

    import numpy as np
    import soundfile as sf

    if not path.is_file():
        raise FileNotFoundError(f"Audio file missing: {path}")
    info = sf.info(str(path))
    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ValueError(f"Expected WAV PCM_16, got {info.format}/{info.subtype}")
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz mono, got {info.samplerate} Hz/{info.channels} ch")
    duration = float(info.duration)
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(f"Duration {duration:.3f}s outside [{MIN_DURATION_SECONDS}, {MAX_DURATION_SECONDS}]s")
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or len(data) != info.frames or data.ndim != 1:
        raise ValueError("Decoded audio dimensions do not match WAV header")
    if not bool(np.isfinite(data).all()):
        raise ValueError("Decoded audio contains non-finite samples")
    peak = float(abs(data).max())
    rms = float(math.sqrt(float((data.astype("float64") ** 2).mean())))
    if peak < MIN_PEAK or rms < MIN_RMS:
        raise ValueError(f"Audio is silent or near-silent (peak={peak:.6f}, rms={rms:.6f})")
    return {
        "audio_sample_rate": info.samplerate,
        "audio_channels": info.channels,
        "audio_subtype": info.subtype,
        "duration_seconds": round(duration, 6),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
    }


def _default_generator(text: str, output_path: Path) -> None:
    # Lazy import: fetching a CSV and inspecting --help need no PyTorch/CUDA.
    from data.generate_jbb_tts import generate_one_audio, validate_audio

    generate_one_audio(text, output_path)
    validate_audio(output_path)


def _load_inventory(path: Path, source_rows: list[dict[str, object]]) -> list[dict[str, object]] | None:
    if not path.exists():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid AdvBench inventory: {path}") from exc
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"AdvBench inventory must contain {EXPECTED_ROWS} rows; got {len(rows)}")
    immutable_fields = (
        "pair_id", "source_row_index", "source_csv_line", "harmful_text", "source_target",
        "goal_sha256", "source_csv_sha256",
    )
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        if not isinstance(row, dict) or any(row.get(key) != source[key] for key in immutable_fields):
            raise ValueError(f"AdvBench source/inventory drift at row {index}; refusing resume")
    return rows


def _write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_bytes(path, payload.encode("utf-8"))


def _clear_pending_binding(row: dict[str, object]) -> None:
    row.pop("pending_audio_sha256", None)
    row.pop("pending_goal_sha256", None)
    row.pop("pending_tts_version", None)


def _mark_audio_failure(row: dict[str, object], reason: str) -> None:
    row["audio_status"] = "failed"
    row["qa_status"] = "failed"
    row["qa_reason"] = reason[:500]
    row["clean_audio_sha256"] = ""
    _clear_pending_binding(row)


def _mark_audio_ready(row: dict[str, object], *, audio_sha256: str, qa: dict[str, object]) -> None:
    row.update(qa)
    row["clean_audio_sha256"] = audio_sha256
    row["audio_status"] = "ready"
    row["qa_status"] = "ok"
    row["qa_reason"] = ""
    row["tts_version"] = row.get("pending_tts_version", row.get("tts_version", ""))
    _clear_pending_binding(row)


def prepare_tts(
    source_csv: Path = SOURCE_CSV,
    inventory_path: Path = INVENTORY,
    audio_dir: Path = AUDIO_DIR,
    *,
    project_root: Path = PROJECT_ROOT,
    limit: int | None = None,
    generator: Callable[[str, Path], None] = _default_generator,
    expected_sha256: str = ADV_BENCH_SHA256,
) -> dict[str, int]:
    """Create or resume all source rows, updating the inventory atomically per row."""

    if limit is not None and not 1 <= limit <= EXPECTED_ROWS:
        raise ValueError(f"--limit must be between 1 and {EXPECTED_ROWS}")
    source_rows = parse_source(source_csv.read_bytes(), expected_sha256=expected_sha256)
    old_rows = _load_inventory(inventory_path, source_rows)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(source_rows):
        audio_path = audio_dir / f"{source['pair_id']}.wav"
        relative = _project_relative(audio_path, project_root)
        if old_rows is not None:
            row = old_rows[index]
            if row.get("clean_audio_path") != relative:
                raise ValueError(f"AdvBench audio path drift for {source['pair_id']}; refusing resume")
        else:
            row = {
                **source,
                "clean_audio_path": relative,
                "clean_audio_sha256": "",
                "audio_status": "pending",
                "qa_status": "pending",
                "qa_reason": "",
                "tts_model": "gTTS-en",
                "tts_version": "",
                "audio_sample_rate": SAMPLE_RATE,
                "audio_channels": 1,
                "audio_subtype": "PCM_16",
                "duration_seconds": None,
                "peak": None,
                "rms": None,
            }
        rows.append(row)

    # A completed row is content-addressed. Never silently accept an edited or
    # deleted WAV, even when the current invocation limits work to other rows.
    for row in rows:
        if row.get("audio_status") == "ready":
            path = project_root / str(row["clean_audio_path"])
            stored_hash = row.get("clean_audio_sha256")
            if path.is_symlink() or not path.is_file() or not stored_hash or sha256_file(path) != stored_hash:
                raise ValueError(f"Audio SHA-256 drift for {row['pair_id']}; refusing resume")
            validate_audio_qa(path)

    _write_inventory(inventory_path, rows)
    selected = rows if limit is None else rows[:limit]
    for processed, row in enumerate(selected, start=1):
        if row.get("audio_status") == "ready" and row.get("qa_status") == "ok":
            if processed % 25 == 0 or processed == len(selected):
                print(f"AdvBench TTS progress {processed}/{len(selected)}; "
                      f"statuses={dict(Counter(str(item['audio_status']) for item in rows))}", flush=True)
            continue
        path = project_root / str(row["clean_audio_path"])
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.wav")
        if path.exists() or path.is_symlink():
            # A valid WAV at the expected filename is not proof that it was
            # synthesized from this goal. Only a preceding, durably recorded
            # pending transaction permits recovery after a crash during commit.
            pending_hash = row.get("pending_audio_sha256")
            bound = (
                not path.is_symlink()
                and isinstance(pending_hash, str)
                and len(pending_hash) == 64
                and row.get("pending_goal_sha256") == row["goal_sha256"]
                and path.is_file()
                and sha256_file(path) == pending_hash
            )
            if bound:
                try:
                    qa = validate_audio_qa(path)
                except Exception as exc:
                    _mark_audio_failure(row, f"Bound WAV failed QA: {type(exc).__name__}: {exc}")
                else:
                    _mark_audio_ready(row, audio_sha256=pending_hash, qa=qa)
            else:
                _mark_audio_failure(row, "Unbound pre-existing WAV; refusing to adopt or overwrite it")
            _write_inventory(inventory_path, rows)
        else:
            # If a previous transaction has no destination, it stopped before
            # the no-clobber link. It is safe to synthesize a fresh WAV.
            _clear_pending_binding(row)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.unlink(missing_ok=True)
            try:
                generator(str(row["harmful_text"]), temporary)
                qa = validate_audio_qa(temporary)
                staged_hash = sha256_file(temporary)
            except Exception as exc:
                _mark_audio_failure(row, f"{type(exc).__name__}: {exc}")
                _write_inventory(inventory_path, rows)
            else:
                # Write the goal/audio binding BEFORE the target WAV appears.
                # A crash after the link but before the ready write is then
                # recoverable without trusting an arbitrary existing file.
                row["pending_goal_sha256"] = row["goal_sha256"]
                row["pending_audio_sha256"] = staged_hash
                row["pending_tts_version"] = _tts_version()
                row["audio_status"] = "pending"
                row["qa_status"] = "pending"
                row["qa_reason"] = ""
                _write_inventory(inventory_path, rows)
                try:
                    # Same-directory hard link is atomic and never replaces a
                    # file another process placed at this target meanwhile.
                    os.link(temporary, path)
                except OSError as exc:
                    _mark_audio_failure(row, f"Could not commit WAV without overwrite: {type(exc).__name__}: {exc}")
                    _write_inventory(inventory_path, rows)
                else:
                    _mark_audio_ready(row, audio_sha256=staged_hash, qa=qa)
                    _write_inventory(inventory_path, rows)
            finally:
                temporary.unlink(missing_ok=True)
        if processed % 25 == 0 or processed == len(selected):
            print(f"AdvBench TTS progress {processed}/{len(selected)}; "
                  f"statuses={dict(Counter(str(item['audio_status']) for item in rows))}", flush=True)
    return dict(Counter(str(row["audio_status"]) for row in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare pinned AdvBench clean audio for RQ2 screening")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch", help="Download and verify the pinned official CSV")
    tts_parser = subparsers.add_parser("tts", help="Generate/resume the 520 clean harmful WAVs")
    tts_parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke run")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        path = fetch_source()
        print(f"AdvBench source verified: {path} ({EXPECTED_ROWS} rows; SHA-256 {ADV_BENCH_SHA256})")
        return 0
    counts = prepare_tts(limit=args.limit)
    print(f"AdvBench audio inventory: {INVENTORY}")
    print(f"Audio status: {counts}")
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
