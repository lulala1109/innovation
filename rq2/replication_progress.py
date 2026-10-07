"""Human-readable heartbeat and durable timing for long GPU/API stages."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from rq2.artifacts import atomic_json
from rq2.replication_judge import utc_now


class StageProgress:
    def __init__(self, root, stage, counter, total, interval=15):
        self.root, self.stage = Path(root), stage
        self.counter, self.total, self.interval = counter, total, interval
        self.stop = threading.Event()
        self.thread = None

    def emit(self, status="running"):
        count = self.counter()
        elapsed = time.monotonic() - self.started
        advance = count - self.initial
        eta = None if advance <= 0 else max(0, self.total - count) * elapsed / advance
        payload = {"stage": self.stage, "status": status, "started_at": self.started_at,
                   "updated_at": utc_now(), "completed_units": count, "total_units": self.total,
                   "resumed_units": self.initial, "elapsed_seconds": round(elapsed, 1),
                   "eta_seconds": None if eta is None else round(eta, 1),
                   "eta_basis": "current_attempt_observed_throughput_not_guaranteed"}
        atomic_json(self.root / "progress.json", payload)
        remaining = "estimating/loading" if eta is None else f"{eta / 60:.1f} min"
        print(f"[{self.stage}] {count}/{self.total}; elapsed={elapsed / 60:.1f} min; ETA={remaining}; {status}", flush=True)
        return payload

    def __enter__(self):
        self.started, self.started_at = time.monotonic(), utc_now()
        self.initial = self.counter()
        self.emit()
        def work():
            while not self.stop.wait(self.interval):
                try:
                    self.emit()
                except Exception:
                    print(f"[{self.stage}] heartbeat unavailable; main stage still running", flush=True)
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, trace):
        self.stop.set()
        self.thread.join(timeout=2)
        payload = self.emit("complete" if kind is None else "stopped")
        directory = self.root / "timings"
        # A distinct attempt entry preserves previous stopped-stage timings.
        name = self.stage + "_" + self.started_at.replace(":", "-") + ".json"
        atomic_json(directory / name, payload)


def line_count(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(line.endswith(b"\n") for line in stream)
