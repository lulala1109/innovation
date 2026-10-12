#!/usr/bin/env python3
"""Resume only dev Oracle scoring; model identity changes are recorded warnings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.dev_judge_resume import DevWarningPipeline, RUN_STAGES, load_warning_config
from rq2.replication_judge import ReplicationStop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "status", "estimate"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    run.add_argument("--from", dest="from_stage", choices=RUN_STAGES, default="oracle_judge")
    run.add_argument("--through", choices=RUN_STAGES, required=True)
    run.add_argument("--acknowledge-cost", action="store_true")
    run.add_argument("--accept-unlocked-judge", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if not args.acknowledge_cost or not args.accept_unlocked_judge:
                raise ReplicationStop("Both execution acknowledgement flags are required; nothing started")
            if RUN_STAGES.index(args.from_stage) > RUN_STAGES.index(args.through):
                parser.error("--from must not follow --through")
        print("[preflight] checking frozen source files; no API/GPU during validation", flush=True)
        spec = load_warning_config(args.config)
        pipeline = DevWarningPipeline(spec)
        if args.command == "run":
            print(json.dumps(pipeline.estimate(), ensure_ascii=False, indent=2), flush=True)
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            start, end = RUN_STAGES.index(args.from_stage), RUN_STAGES.index(args.through) + 1
            result = pipeline.run(RUN_STAGES[start:end], acknowledge_cost=True, accept_unlocked_judge=True)
        elif args.command in {"status", "estimate"}:
            spec.verify()
            result = getattr(pipeline, args.command)()
        else:
            result = getattr(pipeline, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ReplicationStop as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
