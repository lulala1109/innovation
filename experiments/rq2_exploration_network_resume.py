#!/usr/bin/env python3
"""Inspect/check an uncertain request, or explicitly resume window scoring only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rq2.exploration_config import STAGES, load_config
from rq2.exploration_network_resume import (
    NetworkResumePipeline, check_endpoint, configure_direct_connection, endpoint_connection,
)
from rq2.replication_judge import ReplicationStop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "check", "status"):
        command = commands.add_parser(name)
        if name == "check":
            command.add_argument("--direct", action="store_true",
                                 help="Bypass environment proxies for the configured endpoint")
    run = commands.add_parser("run")
    run.add_argument("--direct", action="store_true",
                     help="Bypass environment proxies for the probe and all scoring requests")
    run.add_argument("--accept-uncertain-charge-and-retry", action="store_true")
    run.add_argument("--acknowledge-cost", action="store_true")
    run.add_argument("--accept-exploratory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run" and not (args.accept_uncertain_charge_and_retry
                and args.acknowledge_cost and args.accept_exploratory):
            raise ReplicationStop("All three acknowledgement flags are required before execution")
        print("[preflight] verifying preserved sources and uncertain request; no GPU/API", flush=True)
        spec = load_config(args.config)
        pipeline = NetworkResumePipeline(spec, args.attempt)
        result = pipeline.verify()
        if args.command in {"check", "run"}:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
            if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
                raise ReplicationStop("DEEPSEEK_API_KEY is not loaded; no request sent")
            if args.direct:
                configure_direct_connection(spec.runtime_config.judge)
            print("[network] " + json.dumps({"api_key_loaded": True,
                  **endpoint_connection(spec.runtime_config.judge)}, ensure_ascii=False), flush=True)
            result["endpoint_probe"] = check_endpoint(spec.runtime_config.judge)
        if args.command == "run":
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            result = pipeline.run(STAGES[5:], acknowledge_cost=True, accept_exploratory=True,
                                  accept_uncertain_charge_and_retry=True)
        elif args.command == "status":
            result = pipeline.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ReplicationStop, OSError, ValueError) as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
