"""Freeze the RQ2 dev, causal-test, and reserve split.

The split is intentionally independent of all RQ2 outcomes.  Candidates are
ordered by a SHA-256 key over their immutable identities, a preregistered seed,
and one stable AdvBench stratum.  The first ``dev_count`` rows become
``rq2_dev``, the next ``causal_test_count`` become ``rq2_causal_test``, and all
remaining rows are held as an outcome-blind reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq2.artifacts import atomic_text, canonical_sha256, file_sha256, read_jsonl
from rq2.data import RQ2DataError, read_pair_identities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_DIR = (
    PROJECT_ROOT
    / "outputs/stage2_rq2/advbench_clean_screening_run01/conservative_candidate_pool"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/rq2/advbench_split_v1"
DEFAULT_RQ1_MANIFESTS = (
    PROJECT_ROOT / "dataset/processed/stage1/jbb_pairs_split.csv",
    PROJECT_ROOT / "dataset/processed/stage1/jbb_pairs_audio.csv",
)
SPLIT_FORMAT = "rq2-dataset-split"
SPLIT_VERSION = 1
SPLIT_ALGORITHM = "sha256-canonical-identity-sort-v1"
DEFAULT_SEED = 42
DEFAULT_STRATUM = "advbench"
RESERVE_REASON = "not_selected_for_current_rq2_run"


class RQ2SplitError(ValueError):
    """Raised when split inputs or frozen artifacts violate the contract."""


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
        raise RQ2SplitError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise RQ2SplitError(f"{name} must be a JSON object: {path}")
    return value


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


def _resolve_audio_path(candidate_pool_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RQ2SplitError("clean_audio_path must be non-blank")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_relative = (PROJECT_ROOT / path).resolve()
    if project_relative.is_file():
        return project_relative
    return (candidate_pool_path.parent / path).resolve()


def _content_identities(row: Mapping[str, Any]) -> set[str]:
    harmful_text = row.get("harmful_text")
    if not isinstance(harmful_text, str) or not harmful_text.strip():
        raise RQ2SplitError("harmful_text must be non-blank")
    normalized = " ".join(harmful_text.casefold().split())
    text_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    content_group = row.get("content_group")
    if not isinstance(content_group, str) or not content_group.strip():
        raise RQ2SplitError("content_group must be non-blank")
    return {f"text:{text_sha}", f"group:{content_group.strip()}"}


def _validate_pool_summary(
    summary: Mapping[str, Any],
    *,
    candidate_pool_path: Path,
    candidate_count: int,
) -> None:
    if summary.get("format") != "rq2-conservative-candidate-pool":
        raise RQ2SplitError("candidate-pool summary has an unexpected format")
    if summary.get("pool_is_unassigned") is not True:
        raise RQ2SplitError("candidate pool must be unassigned before splitting")
    if summary.get("formal_role_assignment_ready") is not False:
        raise RQ2SplitError(
            "candidate-pool summary unexpectedly permits formal role assignment"
        )
    counts = summary.get("counts")
    output = summary.get("outputs", {}).get("conservative_candidate_pool")
    if not isinstance(counts, Mapping) or counts.get("candidate_pool") != candidate_count:
        raise RQ2SplitError("candidate-pool summary row count does not match input")
    if not isinstance(output, Mapping):
        raise RQ2SplitError("candidate-pool summary lacks its authoritative output")
    expected_sha = output.get("sha256")
    if not _valid_sha256(expected_sha) or expected_sha != file_sha256(candidate_pool_path):
        raise RQ2SplitError("candidate-pool SHA-256 does not match its summary")
    if output.get("rows") != candidate_count:
        raise RQ2SplitError("candidate-pool output row count does not match input")


def _overlap_report(
    left: Mapping[str, set[str]], right: Mapping[str, set[str]]
) -> dict[str, int | bool]:
    result: dict[str, int | bool] = {}
    for key in ("pair_ids", "content_groups", "audio_sha256"):
        result[f"{key}_overlap"] = len(left[key].intersection(right[key]))
    result["passed"] = all(value == 0 for value in result.values())
    return result


def _role_identities(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    identities = {"pair_ids": set(), "content_groups": set(), "audio_sha256": set()}
    for row in rows:
        identities["pair_ids"].add(str(row["pair_id"]))
        identities["content_groups"].update(_content_identities(row))
        identities["audio_sha256"].add(str(row["clean_audio_sha256"]))
    return identities


def derive_rq2_split(
    *,
    candidate_pool_path: str | Path,
    candidate_summary_path: str | Path,
    rq1_manifest_paths: Sequence[str | Path],
    decision_date: str,
    seed: int = DEFAULT_SEED,
    dev_count: int = 20,
    causal_test_count: int = 40,
    expected_candidate_count: int | None = 455,
    minimum_causal_test_count: int = 20,
    stratum: str = DEFAULT_STRATUM,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Derive the deterministic split and its machine-readable audit."""

    pool_path = Path(candidate_pool_path).expanduser().resolve()
    summary_path = Path(candidate_summary_path).expanduser().resolve()
    rq1_paths = tuple(Path(value).expanduser().resolve() for value in rq1_manifest_paths)
    if not rq1_paths:
        raise RQ2SplitError("at least one complete RQ1 manifest is required")
    if not isinstance(decision_date, str) or not decision_date.strip():
        raise RQ2SplitError("decision_date must be non-blank")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RQ2SplitError("seed must be a non-negative integer")
    if not isinstance(stratum, str) or not stratum.strip():
        raise RQ2SplitError("stratum must be non-blank")
    counts_to_check = {
        "dev_count": dev_count,
        "causal_test_count": causal_test_count,
        "minimum_causal_test_count": minimum_causal_test_count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts_to_check.values()):
        raise RQ2SplitError("split counts must be non-negative integers")
    if dev_count == 0:
        raise RQ2SplitError("rq2_dev must contain at least one pair")
    if causal_test_count < minimum_causal_test_count:
        raise RQ2SplitError(
            "rq2_causal_test count is below the frozen minimum: "
            f"{causal_test_count} < {minimum_causal_test_count}"
        )

    try:
        candidates = read_jsonl(pool_path)
    except (OSError, ValueError) as exc:
        raise RQ2SplitError(str(exc)) from exc
    summary = _json_object(summary_path, name="candidate-pool summary")
    _validate_pool_summary(
        summary,
        candidate_pool_path=pool_path,
        candidate_count=len(candidates),
    )
    if expected_candidate_count is not None and len(candidates) != expected_candidate_count:
        raise RQ2SplitError(
            f"candidate count mismatch: {len(candidates)} != {expected_candidate_count}"
        )
    if dev_count + causal_test_count > len(candidates):
        raise RQ2SplitError("dev and causal-test counts exceed the candidate pool")

    required = {
        "pair_id",
        "harmful_text",
        "content_group",
        "goal_sha256",
        "clean_audio_path",
        "clean_audio_sha256",
        "clean_refused",
        "candidate_pool_status",
        "semantic_independence_review_status",
        "audio_content_fidelity_review_status",
    }
    seen_pair_ids: set[str] = set()
    seen_content: set[str] = set()
    seen_audio: set[str] = set()
    validated: list[dict[str, Any]] = []
    for position, raw_row in enumerate(candidates):
        row = dict(raw_row)
        missing = sorted(required - set(row))
        if missing:
            raise RQ2SplitError(f"candidate[{position}] lacks fields: {missing}")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise RQ2SplitError(f"candidate[{position}] has a blank pair_id")
        if pair_id in seen_pair_ids:
            raise RQ2SplitError(f"duplicate candidate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        if row.get("clean_refused") is not True:
            raise RQ2SplitError(f"candidate {pair_id} is not clean-refused")
        if row.get("candidate_pool_status") != "unassigned" or "rq2_role" in row:
            raise RQ2SplitError(f"candidate {pair_id} is already role-assigned")
        if row.get("semantic_independence_review_status") != "not_performed_conservative_exclusion":
            raise RQ2SplitError(f"candidate {pair_id} has an unexpected semantic-review status")
        if row.get("audio_content_fidelity_review_status") != "not_performed":
            raise RQ2SplitError(f"candidate {pair_id} has an unexpected audio-review status")
        if not _valid_sha256(row.get("goal_sha256")) or not _valid_sha256(row.get("clean_audio_sha256")):
            raise RQ2SplitError(f"candidate {pair_id} has an invalid identity SHA")
        content_ids = _content_identities(row)
        content_overlap = seen_content.intersection(content_ids)
        if content_overlap:
            raise RQ2SplitError(
                f"duplicate candidate harmful content: {sorted(content_overlap)}"
            )
        seen_content.update(content_ids)
        audio_sha = str(row["clean_audio_sha256"])
        if audio_sha in seen_audio:
            raise RQ2SplitError(f"duplicate candidate clean audio: {pair_id}")
        seen_audio.add(audio_sha)
        audio_path = _resolve_audio_path(pool_path, row["clean_audio_path"])
        if not audio_path.is_file():
            raise RQ2SplitError(f"candidate clean audio is missing: {audio_path}")
        if file_sha256(audio_path) != audio_sha:
            raise RQ2SplitError(f"candidate clean-audio SHA mismatch: {pair_id}")
        split_key = canonical_sha256(
            {
                "algorithm": SPLIT_ALGORITHM,
                "seed": seed,
                "stratum": stratum,
                "pair_id": pair_id,
                "goal_sha256": row["goal_sha256"],
                "clean_audio_sha256": audio_sha,
            }
        )
        validated.append({**row, "_split_key": split_key})

    ordered = sorted(validated, key=lambda row: (row["_split_key"], row["pair_id"]))
    assignments: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    for zero_rank, internal in enumerate(ordered):
        rank = zero_rank + 1
        row = {key: value for key, value in internal.items() if key != "_split_key"}
        split_key = str(internal["_split_key"])
        if zero_rank < dev_count:
            role = "rq2_dev"
        elif zero_rank < dev_count + causal_test_count:
            role = "rq2_causal_test"
        else:
            role = "reserve"
        assignment = {
            "pair_id": row["pair_id"],
            "split_role": role,
            "split_rank": rank,
            "split_key": split_key,
            "stratum": stratum,
            "goal_sha256": row["goal_sha256"],
            "content_group": row["content_group"],
            "clean_audio_sha256": row["clean_audio_sha256"],
        }
        assignments.append(assignment)
        common = {
            **row,
            "candidate_pool_status": "role_assigned" if role != "reserve" else "reserve",
            "split_algorithm": SPLIT_ALGORITHM,
            "split_seed": seed,
            "split_key": split_key,
            "split_rank": rank,
            "stratum": stratum,
        }
        if role == "rq2_dev":
            dev_rows.append({**common, "rq2_role": role})
        elif role == "rq2_causal_test":
            causal_rows.append({**common, "rq2_role": role})
        else:
            reserve_rows.append({**common, "split_role": role, "reason": RESERVE_REASON})

    dev_identities = _role_identities(dev_rows)
    causal_identities = _role_identities(causal_rows)
    reserve_identities = _role_identities(reserve_rows)
    try:
        rq1_identities = read_pair_identities(rq1_paths)
    except (OSError, ValueError, RQ2DataError) as exc:
        raise RQ2SplitError(f"cannot read RQ1 identities: {exc}") from exc
    isolation = {
        "rq2_dev_vs_rq2_causal_test": _overlap_report(dev_identities, causal_identities),
        "rq2_dev_vs_rq1": _overlap_report(dev_identities, rq1_identities),
        "rq2_causal_test_vs_rq1": _overlap_report(causal_identities, rq1_identities),
        "assigned_vs_reserve": _overlap_report(
            {
                key: dev_identities[key].union(causal_identities[key])
                for key in dev_identities
            },
            reserve_identities,
        ),
    }
    failed = [name for name, report in isolation.items() if report["passed"] is not True]
    if failed:
        raise RQ2SplitError("identity isolation failed: " + ", ".join(failed))

    assignments_text = _serialize_jsonl(assignments)
    dev_text = _serialize_jsonl(dev_rows)
    causal_text = _serialize_jsonl(causal_rows)
    reserve_text = _serialize_jsonl(reserve_rows)
    audit = {
        "format": SPLIT_FORMAT,
        "version": SPLIT_VERSION,
        "decision_date": decision_date,
        "frozen": True,
        "roles_locked": True,
        "selection_outcome_blind": True,
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "stratification": {
            "field": "stratum",
            "rule": "single-stable-stratum",
            "values": [stratum],
            "reason": "AdvBench source table has no preregistered category field",
        },
        "ordering_key_fields": [
            "algorithm",
            "seed",
            "stratum",
            "pair_id",
            "goal_sha256",
            "clean_audio_sha256",
        ],
        "assignment_rule": {
            "rq2_dev": f"sorted ranks 1..{dev_count}",
            "rq2_causal_test": (
                f"sorted ranks {dev_count + 1}..{dev_count + causal_test_count}"
            ),
            "reserve": f"all sorted ranks after {dev_count + causal_test_count}",
            "post_outcome_reassignment_allowed": False,
        },
        "minimum_causal_test_count": minimum_causal_test_count,
        "counts": {
            "candidate_pool": len(candidates),
            "rq2_dev": len(dev_rows),
            "rq2_causal_test": len(causal_rows),
            "reserve": len(reserve_rows),
        },
        "inputs": {
            "candidate_pool": {
                "path": _portable_path(pool_path),
                "sha256": file_sha256(pool_path),
                "rows": len(candidates),
            },
            "candidate_pool_summary": {
                "path": _portable_path(summary_path),
                "sha256": file_sha256(summary_path),
            },
            "rq1_manifests": [
                {"path": _portable_path(path), "sha256": file_sha256(path)}
                for path in rq1_paths
            ],
        },
        "outputs": {
            "split_assignments": {
                "file": "split_assignments.jsonl",
                "sha256": _text_sha256(assignments_text),
                "rows": len(assignments),
            },
            "rq2_dev": {
                "file": "rq2_dev.jsonl",
                "sha256": _text_sha256(dev_text),
                "rows": len(dev_rows),
            },
            "rq2_causal_test": {
                "file": "rq2_causal_test.jsonl",
                "sha256": _text_sha256(causal_text),
                "rows": len(causal_rows),
            },
            "reserve": {
                "file": "reserve.jsonl",
                "sha256": _text_sha256(reserve_text),
                "rows": len(reserve_rows),
                "reason": RESERVE_REASON,
            },
        },
        "identity_isolation": isolation,
        "inherited_review_limitations": {
            "manual_semantic_review_performed": False,
            "audio_content_fidelity_review_performed": False,
            "claims_not_allowed": summary.get("protocol_deviation", {}).get(
                "claims_not_allowed", []
            ),
        },
        "formal_role_assignment_ready": True,
    }
    return assignments, dev_rows, causal_rows, reserve_rows, audit


def _expected_artifacts(**kwargs: Any) -> tuple[dict[str, str], dict[str, Any]]:
    assignments, dev_rows, causal_rows, reserve_rows, audit = derive_rq2_split(**kwargs)
    contents = {
        "split_assignments.jsonl": _serialize_jsonl(assignments),
        "rq2_dev.jsonl": _serialize_jsonl(dev_rows),
        "rq2_causal_test.jsonl": _serialize_jsonl(causal_rows),
        "reserve.jsonl": _serialize_jsonl(reserve_rows),
        "split_audit.json": _serialize_json(audit),
    }
    return contents, audit


def write_rq2_split(*, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    contents, audit = _expected_artifacts(**kwargs)
    destination = Path(output_dir).expanduser().resolve()
    for name, content in contents.items():
        atomic_text(destination / name, content)
    return {
        "status": "BUILT",
        "output_dir": str(destination),
        "counts": audit["counts"],
        "candidate_pool_sha256": audit["inputs"]["candidate_pool"]["sha256"],
        "split_audit_sha256": _text_sha256(contents["split_audit.json"]),
    }


def check_rq2_split(*, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    contents, audit = _expected_artifacts(**kwargs)
    destination = Path(output_dir).expanduser().resolve()
    for name, content in contents.items():
        path = destination / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RQ2SplitError(f"split artifact is missing, stale, or modified: {path}")
    return {
        "status": "VALID",
        "output_dir": str(destination),
        "counts": audit["counts"],
        "candidate_pool_sha256": audit["inputs"]["candidate_pool"]["sha256"],
        "split_audit_sha256": _text_sha256(contents["split_audit.json"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument(
        "--candidate-pool",
        default=str(DEFAULT_POOL_DIR / "conservative_candidate_pool.jsonl"),
    )
    parser.add_argument(
        "--candidate-summary", default=str(DEFAULT_POOL_DIR / "summary.json")
    )
    parser.add_argument(
        "--rq1-manifest",
        nargs="+",
        default=[str(path) for path in DEFAULT_RQ1_MANIFESTS],
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--decision-date", default="2026-10-02")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-count", type=int, default=20)
    parser.add_argument("--causal-test-count", type=int, default=40)
    parser.add_argument("--expected-candidate-count", type=int, default=455)
    parser.add_argument("--minimum-causal-test-count", type=int, default=20)
    parser.add_argument("--stratum", default=DEFAULT_STRATUM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "candidate_pool_path": args.candidate_pool,
        "candidate_summary_path": args.candidate_summary,
        "rq1_manifest_paths": args.rq1_manifest,
        "output_dir": args.output_dir,
        "decision_date": args.decision_date,
        "seed": args.seed,
        "dev_count": args.dev_count,
        "causal_test_count": args.causal_test_count,
        "expected_candidate_count": args.expected_candidate_count,
        "minimum_causal_test_count": args.minimum_causal_test_count,
        "stratum": args.stratum,
    }
    try:
        result = (
            write_rq2_split(**kwargs)
            if args.mode == "build"
            else check_rq2_split(**kwargs)
        )
    except RQ2SplitError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_STRATUM",
    "RESERVE_REASON",
    "RQ2SplitError",
    "SPLIT_ALGORITHM",
    "SPLIT_FORMAT",
    "SPLIT_VERSION",
    "check_rq2_split",
    "derive_rq2_split",
    "write_rq2_split",
]
