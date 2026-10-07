#!/usr/bin/env python3
"""Prepare an isolated 20-dev baseline run or summarize its text-free sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rq2.artifacts import atomic_json
from rq2.config import RQ2ConfigError, load_rq2_config
from rq2.dev_screen import DevScreenError, prepare_dev_screen, summarize_dev_screen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Derive config/manifest only; no GPU or API")
    prepare.add_argument("--source-config", default=str(PROJECT_ROOT / "configs/stage2_rq2_qwen7b_advbench_sure_here_is_v2_run01.json"))
    prepare.add_argument("--name", default="qwen7b_advbench_dev_screen01")
    prepare.add_argument("--protocol", default=str(PROJECT_ROOT / "docs/RQ2_7B_dev_only_基线可恢复性筛查协议_2026-10-06.md"))
    summary = commands.add_parser("summarize", help="Read trial/Judge sidecars without model or API calls")
    summary.add_argument("--config", required=True)
    summary.add_argument("--save", action="store_true", help="Save the derived report in this dev-only run; inputs remain unchanged")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_dev_screen(args.source_config, name=args.name, protocol=args.protocol)
        else:
            result = summarize_dev_screen(args.config)
            if args.save:
                config = load_rq2_config(args.config)
                output = config.output_root / "baseline_screen" / "summary.json"
                atomic_json(output, result)
                result["saved_report"] = str(output)
            # Per-pair audit is preserved in the saved report, not echoed into terminal logs.
            result.pop("pair_audit", None)
    except (DevScreenError, RQ2ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
