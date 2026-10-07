"""Build the user-approved conservative RQ2 candidate pool.

The policy is deliberately mechanical: start from the finalized AdvBench
``eligible_pool.jsonl`` and exclude every unique candidate that appears in any
top-match entry of ``rq1_advbench_source_review.jsonl``.  It does not claim
that semantic-independence or audio-content-fidelity review was performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq2.artifacts import atomic_text, file_sha256, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREENING_ROOT = (
    PROJECT_ROOT / "outputs/stage2_rq2/advbench_clean_screening_run01"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SCREENING_ROOT / "conservative_candidate_pool"
POOL_FORMAT = "rq2-conservative-candidate-pool"
POOL_VERSION = 1
POLICY = "exclude-all-rq1-advbench-source-review-top-matches-v1"
EXCLUSION_REASON = "conservative_rq1_top_match_exclusion"


class ConservativePoolError(ValueError):
    """Raised when conservative-pool inputs or artifacts violate the contract."""


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConservativePoolError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise ConservativePoolError(f"{name} must be a JSON object: {path}")
    return value


def _records_by_pair_id(
    records: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ConservativePoolError(f"{name}[{position}] has a blank pair_id")
        if pair_id in result:
            raise ConservativePoolError(f"{name} has duplicate pair_id: {pair_id}")
        result[pair_id] = dict(record)
    return result


def _serialize_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )


def _serialize_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_screening_summary(
    summary: Mapping[str, Any], *, eligible_count: int, review_rows: int
) -> None:
    if summary.get("format") != "advbench-rq2-clean-screening":
        raise ConservativePoolError("screening summary has an unexpected format")
    if summary.get("screening_complete") is not True:
        raise ConservativePoolError("screening must be complete before pool derivation")
    if summary.get("eligible_count") != eligible_count:
        raise ConservativePoolError(
            "screening summary eligible_count does not match eligible_pool"
        )
    if summary.get("rq1_advbench_source_review_rows") != review_rows:
        raise ConservativePoolError(
            "screening summary review-row count does not match source review"
        )
    if summary.get("pool_is_unassigned") is not True:
        raise ConservativePoolError("screening source pool must remain unassigned")
    if summary.get("formal_role_assignment_ready") is not False:
        raise ConservativePoolError(
            "screening summary unexpectedly permits formal role assignment"
        )


def derive_conservative_pool(
    *,
    eligible_pool_path: str | Path,
    rq1_review_path: str | Path,
    screening_summary_path: str | Path,
    decision_date: str,
    expected_input_count: int | None = None,
    expected_exclusion_count: int | None = None,
    expected_output_count: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Derive a deterministic candidate pool and its exclusion audit."""

    eligible_path = Path(eligible_pool_path).expanduser().resolve()
    review_path = Path(rq1_review_path).expanduser().resolve()
    summary_path = Path(screening_summary_path).expanduser().resolve()
    if not isinstance(decision_date, str) or not decision_date.strip():
        raise ConservativePoolError("decision_date must be non-blank")

    try:
        eligible_rows = read_jsonl(eligible_path)
        review_rows = read_jsonl(review_path)
    except (OSError, ValueError) as exc:
        raise ConservativePoolError(str(exc)) from exc
    screening_summary = _json_object(summary_path, name="screening summary")
    eligible = _records_by_pair_id(eligible_rows, name="eligible_pool")
    _validate_screening_summary(
        screening_summary,
        eligible_count=len(eligible_rows),
        review_rows=len(review_rows),
    )
    if expected_input_count is not None and len(eligible_rows) != expected_input_count:
        raise ConservativePoolError(
            f"eligible_pool count mismatch: {len(eligible_rows)} != {expected_input_count}"
        )

    required = {
        "pair_id",
        "source_row_index",
        "goal_sha256",
        "clean_audio_sha256",
        "clean_refused",
        "semantic_independence_review_status",
    }
    for pair_id, row in eligible.items():
        missing = sorted(required - set(row))
        if missing:
            raise ConservativePoolError(
                f"eligible candidate {pair_id} lacks fields: {missing}"
            )
        if row["clean_refused"] is not True:
            raise ConservativePoolError(
                f"eligible candidate {pair_id} is not clean-refused"
            )
        if row["semantic_independence_review_status"] != "pending":
            raise ConservativePoolError(
                f"eligible candidate {pair_id} has unexpected semantic review status"
            )
        if not _valid_sha256(row["goal_sha256"]) or not _valid_sha256(
            row["clean_audio_sha256"]
        ):
            raise ConservativePoolError(
                f"eligible candidate {pair_id} has an invalid identity SHA"
            )
        if "rq2_role" in row:
            raise ConservativePoolError(
                f"eligible candidate {pair_id} is already role-assigned"
            )

    evidence: dict[str, list[dict[str, Any]]] = {}
    seen_rq1: set[str] = set()
    for position, row in enumerate(review_rows):
        rq1_pair_id = row.get("rq1_pair_id")
        if not isinstance(rq1_pair_id, str) or not rq1_pair_id.strip():
            raise ConservativePoolError(
                f"rq1 source review row {position} has a blank rq1_pair_id"
            )
        if rq1_pair_id in seen_rq1:
            raise ConservativePoolError(
                f"rq1 source review has duplicate rq1_pair_id: {rq1_pair_id}"
            )
        seen_rq1.add(rq1_pair_id)
        if row.get("review_status") != "pending":
            raise ConservativePoolError(
                f"rq1 source review {rq1_pair_id} has unexpected review_status"
            )
        matches = row.get("top_matches")
        if not isinstance(matches, list) or len(matches) != 3:
            raise ConservativePoolError(
                f"rq1 source review {rq1_pair_id} must contain exactly three top_matches"
            )
        match_ids: set[str] = set()
        for match in matches:
            if not isinstance(match, Mapping):
                raise ConservativePoolError(
                    f"rq1 source review {rq1_pair_id} has a malformed top_match"
                )
            pair_id = match.get("pair_id")
            similarity = match.get("similarity")
            if not isinstance(pair_id, str) or not pair_id.strip():
                raise ConservativePoolError(
                    f"rq1 source review {rq1_pair_id} has a blank matched pair_id"
                )
            if pair_id in match_ids:
                raise ConservativePoolError(
                    f"rq1 source review {rq1_pair_id} repeats matched pair {pair_id}"
                )
            match_ids.add(pair_id)
            if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
                raise ConservativePoolError(
                    f"rq1 source review {rq1_pair_id}/{pair_id} has invalid similarity"
                )
            similarity_float = float(similarity)
            if not math.isfinite(similarity_float) or not 0.0 <= similarity_float <= 1.0:
                raise ConservativePoolError(
                    f"rq1 source review {rq1_pair_id}/{pair_id} has invalid similarity"
                )
            if pair_id in eligible:
                evidence.setdefault(pair_id, []).append(
                    {
                        "rq1_pair_id": rq1_pair_id,
                        "similarity": similarity_float,
                    }
                )

    excluded_ids = set(evidence)
    if expected_exclusion_count is not None and len(excluded_ids) != expected_exclusion_count:
        raise ConservativePoolError(
            f"conservative exclusion count mismatch: {len(excluded_ids)} "
            f"!= {expected_exclusion_count}"
        )

    ordered = sorted(
        eligible.values(),
        key=lambda row: (int(row["source_row_index"]), str(row["pair_id"])),
    )
    candidate_pool: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in ordered:
        pair_id = str(row["pair_id"])
        if pair_id in excluded_ids:
            matches = sorted(
                evidence[pair_id],
                key=lambda item: (-float(item["similarity"]), str(item["rq1_pair_id"])),
            )
            exclusions.append(
                {
                    "pair_id": pair_id,
                    "goal_sha256": row["goal_sha256"],
                    "clean_audio_sha256": row["clean_audio_sha256"],
                    "decision": "exclude",
                    "reason": EXCLUSION_REASON,
                    "rq1_matches": matches,
                }
            )
            continue
        candidate_pool.append(
            {
                **row,
                "candidate_pool_policy": POLICY,
                "candidate_pool_status": "unassigned",
                "semantic_independence_review_status": (
                    "not_performed_conservative_exclusion"
                ),
                "audio_content_fidelity_review_status": "not_performed",
            }
        )

    if expected_output_count is not None and len(candidate_pool) != expected_output_count:
        raise ConservativePoolError(
            f"conservative candidate count mismatch: {len(candidate_pool)} "
            f"!= {expected_output_count}"
        )
    if len(candidate_pool) + len(exclusions) != len(eligible_rows):
        raise ConservativePoolError("candidate/exclusion partition is incomplete")

    pool_text = _serialize_jsonl(candidate_pool)
    exclusions_text = _serialize_jsonl(exclusions)
    summary = {
        "format": POOL_FORMAT,
        "version": POOL_VERSION,
        "decision_date": decision_date,
        "policy": POLICY,
        "inputs": {
            "eligible_pool": {
                "path": _portable_path(eligible_path),
                "sha256": file_sha256(eligible_path),
                "rows": len(eligible_rows),
            },
            "rq1_advbench_source_review": {
                "path": _portable_path(review_path),
                "sha256": file_sha256(review_path),
                "rows": len(review_rows),
            },
            "screening_summary": {
                "path": _portable_path(summary_path),
                "sha256": file_sha256(summary_path),
                "finalize_fingerprint": screening_summary.get(
                    "finalize_fingerprint"
                ),
            },
        },
        "counts": {
            "input_candidates": len(eligible_rows),
            "conservatively_excluded": len(exclusions),
            "candidate_pool": len(candidate_pool),
        },
        "outputs": {
            "conservative_candidate_pool": {
                "file": "conservative_candidate_pool.jsonl",
                "sha256": _text_sha256(pool_text),
                "rows": len(candidate_pool),
            },
            "conservative_exclusions": {
                "file": "conservative_exclusions.jsonl",
                "sha256": _text_sha256(exclusions_text),
                "rows": len(exclusions),
            },
        },
        "protocol_deviation": {
            "manual_semantic_review_performed": False,
            "audio_content_fidelity_review_performed": False,
            "selection_basis": (
                "user-approved conservative exclusion of all RQ1 AdvBench "
                "source-review top matches remaining in the eligible pool"
            ),
            "claims_not_allowed": [
                "manual semantic independence confirmed",
                "audio content fidelity confirmed",
            ],
        },
        "semantic_independence_review_status": (
            "not_performed_conservative_exclusion"
        ),
        "audio_content_fidelity_review_status": "not_performed",
        "pool_is_unassigned": True,
        "formal_role_assignment_ready": False,
    }
    return candidate_pool, exclusions, summary


def write_conservative_pool(
    *,
    eligible_pool_path: str | Path,
    rq1_review_path: str | Path,
    screening_summary_path: str | Path,
    output_dir: str | Path,
    decision_date: str,
    expected_input_count: int | None = None,
    expected_exclusion_count: int | None = None,
    expected_output_count: int | None = None,
) -> dict[str, Any]:
    pool, exclusions, summary = derive_conservative_pool(
        eligible_pool_path=eligible_pool_path,
        rq1_review_path=rq1_review_path,
        screening_summary_path=screening_summary_path,
        decision_date=decision_date,
        expected_input_count=expected_input_count,
        expected_exclusion_count=expected_exclusion_count,
        expected_output_count=expected_output_count,
    )
    destination = Path(output_dir).expanduser().resolve()
    pool_text = _serialize_jsonl(pool)
    exclusions_text = _serialize_jsonl(exclusions)
    summary_text = _serialize_json(summary)
    atomic_text(destination / "conservative_candidate_pool.jsonl", pool_text)
    atomic_text(destination / "conservative_exclusions.jsonl", exclusions_text)
    atomic_text(destination / "summary.json", summary_text)
    return {
        "status": "BUILT",
        "output_dir": str(destination),
        "candidate_pool_rows": len(pool),
        "excluded_rows": len(exclusions),
        "candidate_pool_sha256": _text_sha256(pool_text),
        "exclusions_sha256": _text_sha256(exclusions_text),
        "summary_sha256": _text_sha256(summary_text),
    }


def check_conservative_pool(
    *,
    eligible_pool_path: str | Path,
    rq1_review_path: str | Path,
    screening_summary_path: str | Path,
    output_dir: str | Path,
    decision_date: str,
    expected_input_count: int | None = None,
    expected_exclusion_count: int | None = None,
    expected_output_count: int | None = None,
) -> dict[str, Any]:
    pool, exclusions, summary = derive_conservative_pool(
        eligible_pool_path=eligible_pool_path,
        rq1_review_path=rq1_review_path,
        screening_summary_path=screening_summary_path,
        decision_date=decision_date,
        expected_input_count=expected_input_count,
        expected_exclusion_count=expected_exclusion_count,
        expected_output_count=expected_output_count,
    )
    destination = Path(output_dir).expanduser().resolve()
    expected = {
        "conservative_candidate_pool.jsonl": _serialize_jsonl(pool),
        "conservative_exclusions.jsonl": _serialize_jsonl(exclusions),
        "summary.json": _serialize_json(summary),
    }
    for name, content in expected.items():
        path = destination / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ConservativePoolError(
                f"conservative pool artifact is missing, stale, or modified: {path}"
            )
    return {
        "status": "VALID",
        "output_dir": str(destination),
        "candidate_pool_rows": len(pool),
        "excluded_rows": len(exclusions),
        "candidate_pool_sha256": summary["outputs"][
            "conservative_candidate_pool"
        ]["sha256"],
        "exclusions_sha256": summary["outputs"]["conservative_exclusions"][
            "sha256"
        ],
        "summary_sha256": _text_sha256(expected["summary.json"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument(
        "--eligible-pool",
        default=str(DEFAULT_SCREENING_ROOT / "eligible_pool.jsonl"),
    )
    parser.add_argument(
        "--rq1-review",
        default=str(DEFAULT_SCREENING_ROOT / "rq1_advbench_source_review.jsonl"),
    )
    parser.add_argument(
        "--screening-summary",
        default=str(DEFAULT_SCREENING_ROOT / "summary.json"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--decision-date", default="2026-10-02")
    parser.add_argument("--expected-input-count", type=int, default=480)
    parser.add_argument("--expected-exclusion-count", type=int, default=25)
    parser.add_argument("--expected-output-count", type=int, default=455)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "eligible_pool_path": args.eligible_pool,
        "rq1_review_path": args.rq1_review,
        "screening_summary_path": args.screening_summary,
        "output_dir": args.output_dir,
        "decision_date": args.decision_date,
        "expected_input_count": args.expected_input_count,
        "expected_exclusion_count": args.expected_exclusion_count,
        "expected_output_count": args.expected_output_count,
    }
    try:
        result = (
            write_conservative_pool(**kwargs)
            if args.mode == "build"
            else check_conservative_pool(**kwargs)
        )
    except ConservativePoolError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConservativePoolError",
    "EXCLUSION_REASON",
    "POLICY",
    "POOL_FORMAT",
    "POOL_VERSION",
    "check_conservative_pool",
    "derive_conservative_pool",
    "write_conservative_pool",
]
