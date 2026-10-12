"""CPU-only direction/dose diagnostics; never run generation or Judge requests."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rq2.artifacts import RQ2ArtifactError  # noqa: E402
from rq2.direction_diagnostics import build, write_report  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.name):
        parser.error("--name must be a simple alphanumeric diagnostic name")
    try:
        summary, files = build(args.config)
        output = ROOT / "outputs/stage2_rq2/diagnostics" / args.name
        write_report(output, files, source_root=summary["source_root"], inputs=summary["source_files_sha256"])
    except (RQ2ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "stopped", "reason": str(exc), "auto_retry": False}))
        return 1
    print(json.dumps({"status": "complete", "output": str(output), "api_calls": 0,
                      "gpu_used": False, "formal_allowed": False, "rows": summary["trial_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
