#!/usr/bin/env python3
"""Build or check the frozen Qwen2.5-Omni-7B RQ2 layer preregistration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rq2.preregistration import (
    build_preregistration,
    build_preregistration_v2,
    check_preregistration,
    write_preregistration,
)


DEFAULT_FROZEN = PROJECT_ROOT / "configs" / "stage1_rq1_qwen7b_short_sure_here_is_run01_frozen.json"
DEFAULT_RECORD = PROJECT_ROOT / "configs" / "rq2_candidate_layer_preregistration_qwen7b_v1.json"
DEFAULT_DOCUMENT = PROJECT_ROOT / "docs" / "RQ2候选层预注册_Qwen2.5-Omni-7B_2026-10-02.md"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", default=str(DEFAULT_FROZEN))
    parser.add_argument("--record")
    parser.add_argument("--document")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build deterministic JSON and Markdown records")
    build.add_argument("--date", default="2026-10-02")
    build_v2 = subparsers.add_parser("build-v2", help="Create a v2 revision bound to the unchanged v1 parent")
    build_v2.add_argument("--date", default="2026-10-04")
    build_v2.add_argument("--parent-record", default=str(DEFAULT_RECORD))
    subparsers.add_parser("check", help="Check records without writing files")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = _parser().parse_args(argv)
    v2 = options.command == "build-v2"
    record_path = options.record or str(
        PROJECT_ROOT / "configs/rq2_candidate_layer_preregistration_qwen7b_v2.json"
        if v2 else DEFAULT_RECORD
    )
    document_path = options.document or str(
        PROJECT_ROOT / "docs/RQ2候选层与剂量预注册_Qwen2.5-Omni-7B_v2_2026-10-04.md"
        if v2 else DEFAULT_DOCUMENT
    )
    if options.command in {"build", "build-v2"}:
        record = (
            build_preregistration_v2(
                options.frozen_config, parent_record_path=options.parent_record, date=options.date
            )
            if v2 else build_preregistration(options.frozen_config, date=options.date)
        )
        result = write_preregistration(record, record_path=record_path, document_path=document_path)
    else:
        result = check_preregistration(
            frozen_config_path=options.frozen_config,
            record_path=record_path,
            document_path=document_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
