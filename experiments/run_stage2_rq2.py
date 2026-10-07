#!/usr/bin/env python3
"""CLI for the isolated, stage-gated RQ2 causal pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rq2.config import STAGE_ORDER, load_rq2_config, select_stages


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a frozen-schema RQ2 JSON config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Show stages, resources, dependencies, and status")
    subparsers.add_parser("status", help="Show resumable pipeline status")
    validate = subparsers.add_parser("validate", help="Validate inputs and frozen provenance")
    validate.add_argument("--require-protocol", action="store_true")
    run = subparsers.add_parser("run", help="Run one stage or a contiguous stage range")
    selection = run.add_mutually_exclusive_group(required=False)
    selection.add_argument("--stage", choices=STAGE_ORDER)
    selection.add_argument("--from", dest="from_stage", choices=STAGE_ORDER)
    run.add_argument("--through", choices=STAGE_ORDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parser().parse_args(argv)
    config = load_rq2_config(options.config)
    # Heavy model/Judge dependencies stay out of --help and config parsing.
    from rq2.pipeline import RQ2Pipeline

    pipeline = RQ2Pipeline(config)
    result: Any
    if options.command == "plan":
        result = pipeline.plan()
    elif options.command == "status":
        result = pipeline.status()
    elif options.command == "validate":
        result = pipeline.validate(require_protocol=options.require_protocol)
    else:
        if options.stage is None and options.from_stage is None and options.through is None:
            raise SystemExit("run requires --stage or --from/--through")
        stages = select_stages(
            stage=options.stage,
            from_stage=options.from_stage,
            through_stage=options.through,
        )
        result = pipeline.run(stages)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
