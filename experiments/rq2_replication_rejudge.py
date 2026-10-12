#!/usr/bin/env python3
"""Resume v4 with a new V4.1 Judge, without rerunning baseline GPU generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.replication_judge import ReplicationStop
from rq2.replication_rejudge import RUN_STAGES, RejudgePipeline, load_rejudge_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate", "status", "prepare", "estimate"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--stage", choices=RUN_STAGES)
    selection.add_argument("--from", dest="from_stage", choices=RUN_STAGES)
    run.add_argument("--through", choices=RUN_STAGES)
    run.add_argument("--acknowledge-cost", action="store_true")
    run.add_argument("--confirm-no-external-rq2-use", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_rejudge_config(args.config)
        pipeline = RejudgePipeline(spec)
        if args.command == "run":
            if args.stage:
                if args.through:
                    parser.error("--stage cannot combine with --through")
                stages = (args.stage,)
            else:
                if not args.through:
                    parser.error("Explicit --through or --stage is required")
                start = RUN_STAGES.index(args.from_stage) if args.from_stage else 0
                end = RUN_STAGES.index(args.through) + 1
                if start >= end:
                    parser.error("--from must not follow --through")
                stages = RUN_STAGES[start:end]
            print(json.dumps(pipeline.estimate(), ensure_ascii=False, indent=2), flush=True)
            if not args.acknowledge_cost or not args.confirm_no_external_rq2_use:
                raise ReplicationStop("Review the estimate and provide both flags; nothing started")
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            result = pipeline.run(stages, acknowledge_cost=args.acknowledge_cost,
                                  confirm_no_external_rq2_use=args.confirm_no_external_rq2_use)
        else:
            result = getattr(pipeline, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ReplicationStop as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
