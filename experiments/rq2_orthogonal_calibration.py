#!/usr/bin/env python3
"""Independent calibration A only: no new PGD, reverse, subspace, or causal-test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2.artifacts import RQ2ArtifactError, atomic_json  # noqa: E402
from rq2.exploration_config import source_locks  # noqa: E402
from rq2.exploration_network_resume import configure_direct_connection, check_endpoint  # noqa: E402
from rq2.orthogonal_config import CONFIG, STAGES, draft_config, load_config  # noqa: E402
from rq2.orthogonal_pipeline import OrthogonalPipeline  # noqa: E402
from rq2.replication_judge import ReplicationStop  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "plan", "status"):
        commands.add_parser(name)
    check = commands.add_parser("check-endpoint")
    check.add_argument("--direct", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--from", dest="from_stage", choices=STAGES, default="numeric_preflight")
    run.add_argument("--through", choices=STAGES, default="report")
    run.add_argument("--acknowledge-cost", action="store_true")
    run.add_argument("--accept-exploratory", action="store_true")
    run.add_argument("--direct", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if not args.acknowledge_cost or not args.accept_exploratory:
                raise ReplicationStop("Both acknowledgement flags are required before execution")
            if STAGES.index(args.from_stage) > STAGES.index(args.through):
                raise ReplicationStop("--from must not follow --through")
        print("[preflight] verifying frozen dev inputs; no model construction or API requests", flush=True)
        config = Path(args.config).resolve()
        if args.command == "prepare" and not config.exists():
            if config != ROOT / CONFIG:
                raise ReplicationStop("Create calibration A only at its dedicated configuration path")
            atomic_json(config, draft_config())
        spec = load_config(config)
        pipeline = OrthogonalPipeline(spec)
        if args.command == "prepare":
            result = pipeline.run(("prepare",))
        elif args.command == "run":
            stages = STAGES[STAGES.index(args.from_stage):STAGES.index(args.through)+1]
            if "calibration_judge" in stages:
                from dotenv import load_dotenv
                load_dotenv(ROOT / ".env", override=False)
                if args.direct:
                    configure_direct_connection(pipeline.config.judge)
            result = pipeline.run(stages, acknowledge_cost=True, accept_exploratory=True)
        else:
            with source_locks(spec.source):
                verified = pipeline.verify()
                if args.command == "validate":
                    result = verified
                elif args.command == "check-endpoint":
                    from dotenv import load_dotenv
                    load_dotenv(ROOT / ".env", override=False)
                    if args.direct:
                        configure_direct_connection(pipeline.config.judge)
                    result = check_endpoint(pipeline.config.judge)
                else:
                    result = getattr(pipeline, args.command)()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ReplicationStop, RQ2ArtifactError, OSError, ValueError, RuntimeError) as exc:
        # Do not print arbitrary model/provider payloads or credentials.
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
