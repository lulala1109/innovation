#!/usr/bin/env python3
"""Post-Oracle dev exploration only; never runs PGD, formal, or protocol_lock."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.exploration_config import STAGES, load_config
from rq2.exploration_pipeline import ExplorationPipeline
from rq2.replication_judge import ReplicationStop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate", "estimate", "status", "prepare"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    run.add_argument("--from", dest="from_stage", choices=STAGES, default="mechanism_generate")
    run.add_argument("--through", choices=STAGES, required=True)
    run.add_argument("--acknowledge-cost", action="store_true")
    run.add_argument("--accept-exploratory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if not args.acknowledge_cost or not args.accept_exploratory:
                raise ReplicationStop("Both acknowledgement flags are required; no execution started")
            if STAGES.index(args.from_stage) > STAGES.index(args.through):
                raise ReplicationStop("--from must not follow --through")
        print("[preflight] checking frozen dev sources; no GPU/API during validation", flush=True)
        spec = load_config(args.config)
        pipeline = ExplorationPipeline(spec)
        if args.command == "run":
            print(json.dumps(spec.raw["estimate"], ensure_ascii=False, indent=2), flush=True)
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            stages = STAGES[STAGES.index(args.from_stage):STAGES.index(args.through) + 1]
            result = pipeline.run(stages, acknowledge_cost=True, accept_exploratory=True)
        elif args.command == "prepare":
            result = pipeline.run(("prepare",))
        elif args.command == "validate":
            result = pipeline.verify()
        elif args.command == "estimate":
            result = spec.raw["estimate"]
        else:
            result = getattr(pipeline, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ReplicationStop as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
