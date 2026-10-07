#!/usr/bin/env python3
"""Prepare and execute the isolated v3 event-centered dev pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.event_config import EVENT_STAGES, load_event_config, prepare_event_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="v3 event-dev configuration, not a v2 config")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Bind existing dev baselines and write a new config/protocol")
    prepare.add_argument("--source-config", default=str(ROOT / "configs/stage2_rq2_qwen7b_advbench_dev_screen01.json"))
    prepare.add_argument("--name", default="qwen7b_advbench_event_dev_v3_run01")
    prepare.add_argument("--protocol", default=str(ROOT / "docs/RQ2_7B_v3_事件中心_dev协议_2026-10-06.md"))
    for name in ("plan", "validate", "status"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--stage", choices=EVENT_STAGES)
    selection.add_argument("--from", dest="from_stage", choices=EVENT_STAGES)
    run.add_argument("--through", choices=EVENT_STAGES)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_event_config(args.source_config, name=args.name, protocol=args.protocol)
    else:
        if not args.config:
            parser.error("--config is required except for prepare")
        spec = load_event_config(args.config)
        from rq2.event_pipeline import EventPilotPipeline
        pipeline = EventPilotPipeline(spec)
        if args.command == "run":
            if args.stage:
                if args.through:
                    parser.error("--stage cannot be combined with --through")
                stages = (args.stage,)
            else:
                if not args.through:
                    parser.error("run requires --stage or --through; no implicit full run")
                start = EVENT_STAGES.index(args.from_stage) if args.from_stage else 0
                stop = EVENT_STAGES.index(args.through) + 1
                if start >= stop:
                    parser.error("--from must precede --through")
                stages = EVENT_STAGES[start:stop]
            result = pipeline.run(stages)
        else:
            result = getattr(pipeline, args.command)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
