"""Offline source/TTS tests; neither the network nor a real GPU is needed."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import data.prepare_advbench_rq2 as prep
from data.prepare_advbench_rq2 import (
    ADV_BENCH_COMMIT,
    ADV_BENCH_SHA256,
    EXPECTED_ROWS,
    fetch_source,
    parse_source,
    prepare_tts,
)


def _source_bytes() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["goal", "target"])
    for index in range(EXPECTED_ROWS):
        writer.writerow([f"Goal {index}: explain a prohibited act", f"TARGET_ONLY_{index}"])
    return stream.getvalue().encode("utf-8")


def _source(tmp_path: Path) -> tuple[Path, str]:
    data = _source_bytes()
    path = tmp_path / "raw" / "harmful_behaviors.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _fake_wav(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(16_000, 0.25, dtype=np.float32), 16_000, subtype="PCM_16")


def _inventory(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pinned_source_contract_and_all_520_rows(tmp_path: Path) -> None:
    assert ADV_BENCH_COMMIT == "098262edf85f807224e70ecd87b9d83716bf6b73"
    assert ADV_BENCH_SHA256 == "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
    data = _source_bytes()
    digest = hashlib.sha256(data).hexdigest()
    path = fetch_source(tmp_path / "harmful_behaviors.csv", downloader=lambda url: data, expected_sha256=digest)
    records = parse_source(path.read_bytes(), expected_sha256=digest)
    assert len(records) == EXPECTED_ROWS
    assert records[0]["pair_id"] == "advbench_0000"
    assert records[-1]["pair_id"] == "advbench_0519"
    assert records[0]["source_row_index"] == 0
    assert records[-1]["source_csv_line"] == 521
    assert records[0]["source_target"] == "TARGET_ONLY_0"
    assert records[0]["source_csv_sha256"] == digest
    assert records[0]["goal_sha256"] == hashlib.sha256(str(records[0]["harmful_text"]).encode()).hexdigest()


def test_source_rejects_bad_hash_columns_and_count(tmp_path: Path) -> None:
    data = _source_bytes()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_source(data)
    bad_columns = data.replace(b"goal,target", b"prompt,target", 1)
    with pytest.raises(ValueError, match="columns"):
        parse_source(bad_columns, expected_sha256=hashlib.sha256(bad_columns).hexdigest())
    short_data = b"goal,target\njust one,answer\n"
    with pytest.raises(ValueError, match="520 records"):
        parse_source(short_data, expected_sha256=hashlib.sha256(short_data).hexdigest())
    destination = tmp_path / "official.csv"
    destination.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_source(destination, downloader=lambda url: data, expected_sha256=hashlib.sha256(data).hexdigest())
    assert destination.read_bytes() == b"wrong"


def test_tts_inventory_preserves_pending_rows_and_never_reads_target(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    manifest = tmp_path / "inventory.jsonl"
    generated: list[str] = []

    def fake_generator(text: str, path: Path) -> None:
        generated.append(text)
        _fake_wav(text, path)

    counts = prepare_tts(
        source, manifest, tmp_path / "audio", project_root=tmp_path,
        limit=2, generator=fake_generator, expected_sha256=digest,
    )
    rows = _inventory(manifest)
    assert counts == {"ready": 2, "pending": 518}
    assert len(rows) == EXPECTED_ROWS
    assert generated == ["Goal 0: explain a prohibited act", "Goal 1: explain a prohibited act"]
    assert all("TARGET_ONLY" not in text for text in generated)
    assert rows[0]["clean_audio_path"] == "audio/advbench_0000.wav"
    assert rows[0]["audio_status"] == "ready"
    assert rows[0]["qa_status"] == "ok"
    assert rows[0]["clean_audio_sha256"]
    assert rows[0]["duration_seconds"] == 1.0
    assert rows[2]["audio_status"] == "pending"
    assert rows[2]["clean_audio_sha256"] == ""
    # A second run is a no-op for previously verified audio.
    prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path, limit=2,
                generator=fake_generator, expected_sha256=digest)
    assert len(generated) == 2


def test_failed_and_silent_tts_are_recorded_then_retried(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    manifest = tmp_path / "inventory.jsonl"

    def failing_generator(text: str, path: Path) -> None:
        if text.startswith("Goal 0:"):
            raise RuntimeError("temporary TTS failure")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")

    counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                         limit=2, generator=failing_generator, expected_sha256=digest)
    rows = _inventory(manifest)
    assert counts == {"failed": 2, "pending": 518}
    assert rows[0]["qa_reason"].startswith("RuntimeError: temporary TTS failure")
    assert "silent" in rows[1]["qa_reason"]
    assert not (tmp_path / "audio/advbench_0001.wav").exists()
    counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                         limit=2, generator=_fake_wav, expected_sha256=digest)
    assert counts == {"ready": 2, "pending": 518}


def test_audio_or_source_drift_refuses_resume(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    manifest = tmp_path / "inventory.jsonl"
    prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                limit=1, generator=_fake_wav, expected_sha256=digest)
    original = tmp_path / "audio/advbench_0000.wav"
    sf.write(original, np.full(16_000, 0.5, dtype=np.float32), 16_000, subtype="PCM_16")
    with pytest.raises(ValueError, match="Audio SHA-256 drift"):
        prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                    limit=1, generator=_fake_wav, expected_sha256=digest)
    original.unlink()
    # A changed source is never merged with an old inventory, even if its new
    # SHA is passed explicitly by a caller.
    data = source.read_bytes().replace(b"Goal 0:", b"Goal X:", 1)
    source.write_bytes(data)
    with pytest.raises(ValueError, match="source/inventory drift"):
        prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                    limit=1, generator=_fake_wav, expected_sha256=hashlib.sha256(data).hexdigest())


def test_unknown_invalid_existing_wav_is_not_overwritten(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    audio = tmp_path / "audio/advbench_0000.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"user-owned invalid content")
    manifest = tmp_path / "inventory.jsonl"
    called = False

    def generator(text: str, path: Path) -> None:
        nonlocal called
        called = True
        _fake_wav(text, path)

    counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                         limit=1, generator=generator, expected_sha256=digest)
    assert counts == {"failed": 1, "pending": 519}
    assert audio.read_bytes() == b"user-owned invalid content"
    assert not called
    # The operator can explicitly remove this exact bad file, then resume.
    audio.unlink()
    counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                         limit=1, generator=generator, expected_sha256=digest)
    assert called and counts == {"ready": 1, "pending": 519}


def test_valid_existing_wav_without_goal_audio_binding_is_not_adopted(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    audio = tmp_path / "audio/advbench_0000.wav"
    _fake_wav("unrelated speech", audio)
    original_bytes = audio.read_bytes()
    manifest = tmp_path / "inventory.jsonl"
    called = False

    def generator(text: str, path: Path) -> None:
        nonlocal called
        called = True
        _fake_wav(text, path)

    for _ in range(2):
        counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                             limit=1, generator=generator, expected_sha256=digest)
        assert counts == {"failed": 1, "pending": 519}
        assert audio.read_bytes() == original_bytes
        assert not called
        row = _inventory(manifest)[0]
        assert row["audio_status"] == "failed"
        assert "Unbound pre-existing WAV" in row["qa_reason"]
        assert row["clean_audio_sha256"] == ""


def test_goal_audio_bound_wav_recovers_after_commit_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, digest = _source(tmp_path)
    manifest = tmp_path / "inventory.jsonl"
    real_write = prep._write_inventory
    write_count = 0

    def crash_before_ready_write(path: Path, rows: list[dict[str, object]]) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise RuntimeError("simulated crash after WAV link")
        real_write(path, rows)

    monkeypatch.setattr(prep, "_write_inventory", crash_before_ready_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                    limit=1, generator=_fake_wav, expected_sha256=digest)
    audio = tmp_path / "audio/advbench_0000.wav"
    assert audio.is_file()
    staged = _inventory(manifest)[0]
    assert staged["audio_status"] == "pending"
    assert staged["pending_goal_sha256"] == staged["goal_sha256"]
    assert staged["pending_audio_sha256"] == hashlib.sha256(audio.read_bytes()).hexdigest()
    monkeypatch.setattr(prep, "_write_inventory", real_write)

    def fail_if_regenerated(text: str, path: Path) -> None:
        raise AssertionError("a durably bound WAV should be recovered without TTS")

    counts = prepare_tts(source, manifest, tmp_path / "audio", project_root=tmp_path,
                         limit=1, generator=fail_if_regenerated, expected_sha256=digest)
    assert counts == {"ready": 1, "pending": 519}
    recovered = _inventory(manifest)[0]
    assert recovered["clean_audio_sha256"] == staged["pending_audio_sha256"]
    assert recovered["qa_status"] == "ok"
    assert "pending_audio_sha256" not in recovered
