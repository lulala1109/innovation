#!/usr/bin/env python3
"""Produce immutable pair-level Oracle diagnostics; no GPU or API stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.offline_diagnostics import write_diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-config", default=str(ROOT / "configs/rq2_qwen7b_advbench_event_dev_v3_run01_judge_revision01.json"))
    parser.add_argument("--name", default="qwen7b_advbench_event_dev_v3_run01_diagnostic01")
    args = parser.parse_args(argv)
    result = write_diagnostics(args.revision_config, name=args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
