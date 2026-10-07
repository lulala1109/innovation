#!/usr/bin/env python3
"""Versioned, GPU-free Judge revision for a completed v3 dev Oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.judge_revision import prepare_revision, audit_revision, review_revision, analyze_revision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="independent rq2-judge-revision config")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze inputs and produce an offline conflict inventory")
    prepare.add_argument("--event-config", default=str(ROOT / "configs/stage2_rq2_qwen7b_advbench_event_dev_v3_run01.json"))
    prepare.add_argument("--name", default="qwen7b_advbench_event_dev_v3_run01_judge_revision01")
    prepare.add_argument("--protocol", default=str(ROOT / "docs/RQ2_v3_Judge评分修订协议_2026-10-07.md"))
    commands.add_parser("audit", help="idempotent CPU-only inventory; no API client")
    review = commands.add_parser("review", help="one blinded API evaluation per unresolved unique input")
    review.add_argument("--allow-api", action="store_true", help="explicitly permit this bounded API batch")
    commands.add_parser("analyze", help="CPU-only diagnostic reanalysis; blocks until all inputs are resolved")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_revision(args.event_config, name=args.name, protocol=args.protocol)
    else:
        if not args.config:
            parser.error("--config is required except for prepare")
        if args.command == "review":
            if not args.allow_api:
                parser.error("review requires --allow-api; use audit for a no-cost inventory")
            result = review_revision(args.config, allow_api=True)
        elif args.command == "audit":
            result = audit_revision(args.config)
        else:
            result = analyze_revision(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
