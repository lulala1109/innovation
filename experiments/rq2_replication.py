#!/usr/bin/env python3
"""Plan/validate or explicitly run the isolated independent dev replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.replication_config import STAGES, load_replication_config
from rq2.replication_judge import ReplicationStop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate", "status"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--stage", choices=STAGES)
    selection.add_argument("--from", dest="from_stage", choices=STAGES)
    run.add_argument("--through", choices=STAGES)
    run.add_argument("--acknowledge-cost", action="store_true",
                     help="Confirm the printed time estimate and CNY 30 API pause threshold")
    run.add_argument("--confirm-no-external-rq2-use", action="store_true",
                     help="Confirm these selected pairs were not used for PGD/intervention/tuning elsewhere")
    args = parser.parse_args(argv)
    try:
        spec = load_replication_config(args.config)
        from rq2.replication_pipeline import ReplicationPipeline
        pipeline = ReplicationPipeline(spec)
        if args.command == "run":
            if args.stage:
                if args.through:
                    parser.error("--stage cannot combine with --through")
                stages = (args.stage,)
            else:
                if not args.through:
                    parser.error("Specify --through or --stage; no implicit full experiment")
                start = STAGES.index(args.from_stage) if args.from_stage else 0
                end = STAGES.index(args.through) + 1
                if start >= end:
                    parser.error("--from must not follow --through")
                stages = STAGES[start:end]
            print(json.dumps({"before_execution_estimate": spec.raw["budget"],
                "note": "Peak API prices are conservative accounting, not actual invoice. GPU rent is separate."},
                ensure_ascii=False, indent=2), flush=True)
            if not args.acknowledge_cost or not args.confirm_no_external_rq2_use:
                raise ReplicationStop("Review the estimate and confirm both required flags; nothing started")
            # Match the user's .env workflow. Never print environment values.
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            result = pipeline.run(stages, acknowledge_cost=args.acknowledge_cost,
                                  confirm_no_external_rq2_use=args.confirm_no_external_rq2_use)
        else:
            result = getattr(pipeline, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ReplicationStop as exc:
        print(json.dumps({"status": "stopped_before_further_work", "reason": str(exc),
                          "auto_retry": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
