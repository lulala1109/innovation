"""Build the frozen, trajectory-pending RQ2 source manifest.

This is an offline T0.7 step.  It combines the frozen RQ2 dev and causal-test
roles, excludes the reserve, fixes the attack target to ``Sure, here is``, and
binds the resulting 60-row manifest to the T0.5 and T0.6 audit records.  It
does not create trajectories or invoke a model/Judge API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq2.artifacts import atomic_text, file_sha256, read_jsonl
from rq2.data import RQ2DataError, read_pair_identities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "dataset/processed/rq2/advbench_split_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset/processed/rq2/advbench_manifest_v1"
DEFAULT_REVIEW_AUDIT = (
    PROJECT_ROOT
    / "outputs/stage2_rq2/advbench_clean_screening_run01"
    / "conservative_candidate_pool/summary.json"
)
DEFAULT_RQ1_MANIFESTS = (
    PROJECT_ROOT / "dataset/processed/stage1/jbb_pairs_split.csv",
    PROJECT_ROOT / "dataset/processed/stage1/jbb_pairs_audio.csv",
)

MANIFEST_AUDIT_FORMAT = "rq2-source-manifest-audit"
MANIFEST_AUDIT_VERSION = 1
ATTACK_TARGET = "Sure, here is"
STRATUM = "advbench"
SPLIT_SEED = 42
SPLIT_ALGORITHM = "sha256-canonical-identity-sort-v1"
FROZEN_SPLIT_AUDIT_SHA256 = (
    "0cc3f20ed40a654d59aa5953c8c450917618033360393e2844b11f34302b8c48"
)
FROZEN_REVIEW_AUDIT_SHA256 = (
    "dba85273e4e2ee7237e23987288ec07d66c1036361462107a27effd9743e0594"
)
RESERVE_REASON = "not_selected_for_current_rq2_run"
OUTPUT_FIELDS = (
    "case_id",
    "pair_id",
    "rq2_role",
    "harmful_text",
    "clean_audio_path",
    "clean_audio_sha256",
    "clean_refused",
    "content_group",
    "goal_sha256",
    "stratum",
    "target_text",
    "split_rank",
    "split_key",
    "split_seed",
    "split_algorithm",
)
FORBIDDEN_SOURCE_FIELDS = frozenset(
    {
        "target",
        "harmful_target",
        "response",
        "raw_response",
        "judge_result",
        "reasoning",
        "overall_reasoning",
    }
)


class RQ2ManifestError(ValueError):
    """Raised when T0.7 inputs or artifacts violate the frozen contract."""


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
        raise RQ2ManifestError(f"cannot read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise RQ2ManifestError(f"{name} must be a JSON object: {path}")
    return value


def _rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RQ2ManifestError(f"cannot read {name}: {path}: {exc}") from exc
    return [dict(row) for row in rows]


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


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RQ2ManifestError(f"{name} must be non-blank")
    if value != value.strip():
        raise RQ2ManifestError(f"{name} must not contain surrounding whitespace")
    return value


def _resolve_audio_path(source_path: Path, value: Any) -> Path:
    raw = _required_text(value, name="clean_audio_path")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    project_relative = (PROJECT_ROOT / candidate).resolve()
    if project_relative.is_file():
        return project_relative
    return (source_path.parent / candidate).resolve()


def _content_identities(row: Mapping[str, Any]) -> set[str]:
    harmful = _required_text(row.get("harmful_text"), name="harmful_text")
    normalized = " ".join(harmful.casefold().split())
    semantic_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    group = _required_text(row.get("content_group"), name="content_group")
    return {f"text:{semantic_sha}", f"group:{group}"}


def _role_identities(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    identities = {"pair_ids": set(), "content_groups": set(), "audio_sha256": set()}
    for row in rows:
        identities["pair_ids"].add(str(row["pair_id"]))
        identities["content_groups"].update(_content_identities(row))
        identities["audio_sha256"].add(str(row["clean_audio_sha256"]))
    return identities


def _overlap_report(
    left: Mapping[str, set[str]], right: Mapping[str, set[str]]
) -> dict[str, int | bool]:
    report: dict[str, int | bool] = {}
    for key in ("pair_ids", "content_groups", "audio_sha256"):
        report[f"{key}_overlap"] = len(left[key].intersection(right[key]))
    report["passed"] = all(value == 0 for value in report.values())
    return report


def _validate_expected_sha(path: Path, expected: str | None, *, name: str) -> str:
    observed = file_sha256(path)
    if expected is not None:
        if not _valid_sha256(expected):
            raise RQ2ManifestError(f"expected {name} SHA-256 is invalid")
        if observed != expected:
            raise RQ2ManifestError(
                f"{name} SHA-256 mismatch: {observed} != {expected}"
            )
    return observed


def _validate_split_audit(
    audit: Mapping[str, Any],
    *,
    dev_path: Path,
    causal_path: Path,
    reserve_path: Path,
    expected_dev_count: int,
    expected_causal_test_count: int,
    expected_reserve_count: int,
) -> None:
    if audit.get("format") != "rq2-dataset-split" or audit.get("version") != 1:
        raise RQ2ManifestError("split audit has an unexpected format/version")
    for field in (
        "frozen",
        "roles_locked",
        "selection_outcome_blind",
        "formal_role_assignment_ready",
    ):
        if audit.get(field) is not True:
            raise RQ2ManifestError(f"split audit requires {field}=true")
    if audit.get("seed") != SPLIT_SEED:
        raise RQ2ManifestError("split audit seed drifted from 42")
    if audit.get("algorithm") != SPLIT_ALGORITHM:
        raise RQ2ManifestError("split audit algorithm drifted")
    stratification = audit.get("stratification")
    if not isinstance(stratification, Mapping) or stratification.get("values") != [STRATUM]:
        raise RQ2ManifestError("split audit stratum drifted from advbench")
    counts = audit.get("counts")
    expected_counts = {
        "rq2_dev": expected_dev_count,
        "rq2_causal_test": expected_causal_test_count,
        "reserve": expected_reserve_count,
    }
    if not isinstance(counts, Mapping):
        raise RQ2ManifestError("split audit lacks counts")
    for role, expected in expected_counts.items():
        if counts.get(role) != expected:
            raise RQ2ManifestError(
                f"split audit {role} count mismatch: {counts.get(role)} != {expected}"
            )
    outputs = audit.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RQ2ManifestError("split audit lacks outputs")
    for role, path, expected in (
        ("rq2_dev", dev_path, expected_dev_count),
        ("rq2_causal_test", causal_path, expected_causal_test_count),
        ("reserve", reserve_path, expected_reserve_count),
    ):
        record = outputs.get(role)
        if not isinstance(record, Mapping):
            raise RQ2ManifestError(f"split audit lacks output {role}")
        if record.get("file") != path.name:
            raise RQ2ManifestError(f"split audit {role} filename mismatch")
        if record.get("rows") != expected:
            raise RQ2ManifestError(f"split audit {role} row count mismatch")
        if record.get("sha256") != file_sha256(path):
            raise RQ2ManifestError(f"split audit {role} SHA-256 mismatch")


def _validate_review_audit(
    review: Mapping[str, Any],
    *,
    review_path: Path,
    split_audit: Mapping[str, Any],
) -> None:
    if review.get("format") != "rq2-conservative-candidate-pool":
        raise RQ2ManifestError("review/protocol-deviation audit has an unexpected format")
    protocol = review.get("protocol_deviation")
    if not isinstance(protocol, Mapping):
        raise RQ2ManifestError("review audit lacks protocol_deviation")
    if protocol.get("manual_semantic_review_performed") is not False:
        raise RQ2ManifestError("review audit must retain manual review not performed")
    if protocol.get("audio_content_fidelity_review_performed") is not False:
        raise RQ2ManifestError("review audit must retain audio review not performed")
    recorded = split_audit.get("inputs", {}).get("candidate_pool_summary")
    if not isinstance(recorded, Mapping):
        raise RQ2ManifestError("split audit lacks candidate_pool_summary input")
    if recorded.get("sha256") != file_sha256(review_path):
        raise RQ2ManifestError("split audit does not bind the review audit SHA-256")
    if recorded.get("path") != _portable_path(review_path):
        raise RQ2ManifestError("split audit does not bind the review audit path")


def _validate_rq1_sources(
    *, split_audit: Mapping[str, Any], rq1_paths: Sequence[Path]
) -> dict[str, set[str]]:
    recorded = split_audit.get("inputs", {}).get("rq1_manifests")
    if not isinstance(recorded, list) or len(recorded) != len(rq1_paths):
        raise RQ2ManifestError("split audit RQ1 manifest list mismatch")
    actual = [
        {"path": _portable_path(path), "sha256": file_sha256(path)}
        for path in rq1_paths
    ]
    if recorded != actual:
        raise RQ2ManifestError("RQ1 manifest provenance drifted from split audit")
    try:
        return read_pair_identities(rq1_paths)
    except (OSError, ValueError, RQ2DataError) as exc:
        raise RQ2ManifestError(f"cannot read RQ1 identities: {exc}") from exc


def _validate_reserve_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int
) -> set[str]:
    if len(rows) != expected_count:
        raise RQ2ManifestError(
            f"reserve count mismatch: {len(rows)} != {expected_count}"
        )
    pair_ids: set[str] = set()
    for position, row in enumerate(rows):
        pair_id = _required_text(row.get("pair_id"), name=f"reserve[{position}].pair_id")
        if pair_id in pair_ids:
            raise RQ2ManifestError(f"reserve has duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if row.get("split_role") != "reserve" or row.get("rq2_role") is not None:
            raise RQ2ManifestError(f"reserve row is role-assigned: {pair_id}")
        if row.get("candidate_pool_status") != "reserve":
            raise RQ2ManifestError(f"reserve row has invalid pool status: {pair_id}")
        if row.get("reason") != RESERVE_REASON:
            raise RQ2ManifestError(f"reserve row has invalid reason: {pair_id}")
    return pair_ids


def _manifest_row(
    row: Mapping[str, Any],
    *,
    position: int,
    expected_role: str,
    source_path: Path,
    reserve_ids: set[str],
) -> dict[str, Any]:
    forbidden = sorted(FORBIDDEN_SOURCE_FIELDS.intersection(row))
    if forbidden:
        raise RQ2ManifestError(
            f"{expected_role}[{position}] contains forbidden source fields: {forbidden}"
        )
    required = {
        "pair_id",
        "rq2_role",
        "harmful_text",
        "clean_audio_path",
        "clean_audio_sha256",
        "clean_refused",
        "content_group",
        "goal_sha256",
        "stratum",
        "target_text",
        "split_rank",
        "split_key",
        "split_seed",
        "split_algorithm",
        "candidate_pool_status",
    }
    missing = sorted(required - set(row))
    if missing:
        raise RQ2ManifestError(
            f"{expected_role}[{position}] lacks required fields: {missing}"
        )
    pair_id = _required_text(
        row.get("pair_id"), name=f"{expected_role}[{position}].pair_id"
    )
    if pair_id in reserve_ids:
        raise RQ2ManifestError(f"reserve pair leaked into the formal manifest: {pair_id}")
    if row.get("rq2_role") != expected_role:
        raise RQ2ManifestError(f"role mismatch for {pair_id}: {row.get('rq2_role')}")
    if row.get("candidate_pool_status") != "role_assigned":
        raise RQ2ManifestError(f"pair is not frozen as role_assigned: {pair_id}")
    if row.get("clean_refused") is not True:
        raise RQ2ManifestError(f"pair is not clean-refused: {pair_id}")
    if row.get("target_text") != ATTACK_TARGET:
        raise RQ2ManifestError(
            f"attack target drift for {pair_id}: expected {ATTACK_TARGET!r}"
        )
    if row.get("stratum") != STRATUM:
        raise RQ2ManifestError(f"stratum drift for {pair_id}")
    if row.get("split_seed") != SPLIT_SEED:
        raise RQ2ManifestError(f"split seed drift for {pair_id}")
    if row.get("split_algorithm") != SPLIT_ALGORITHM:
        raise RQ2ManifestError(f"split algorithm drift for {pair_id}")
    rank = row.get("split_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise RQ2ManifestError(f"invalid split_rank for {pair_id}")
    split_key = row.get("split_key")
    audio_sha = row.get("clean_audio_sha256")
    goal_sha = row.get("goal_sha256")
    if not _valid_sha256(split_key):
        raise RQ2ManifestError(f"invalid split_key for {pair_id}")
    if not _valid_sha256(audio_sha) or not _valid_sha256(goal_sha):
        raise RQ2ManifestError(f"invalid identity SHA for {pair_id}")
    harmful_text = _required_text(
        row.get("harmful_text"), name=f"{expected_role}[{position}].harmful_text"
    )
    content_group = _required_text(
        row.get("content_group"), name=f"{expected_role}[{position}].content_group"
    )
    audio_path = _resolve_audio_path(source_path, row.get("clean_audio_path"))
    if not audio_path.is_file():
        raise RQ2ManifestError(f"clean audio is missing for {pair_id}: {audio_path}")
    if file_sha256(audio_path) != audio_sha:
        raise RQ2ManifestError(f"clean-audio SHA-256 mismatch for {pair_id}")
    if row.get("trajectory_path") is not None or row.get("trajectory_index") is not None:
        raise RQ2ManifestError(
            f"initial generate-mode manifest must not carry a trajectory path: {pair_id}"
        )
    return {
        "case_id": pair_id,
        "pair_id": pair_id,
        "rq2_role": expected_role,
        "harmful_text": harmful_text,
        "clean_audio_path": _portable_path(audio_path),
        "clean_audio_sha256": audio_sha,
        "clean_refused": True,
        "content_group": content_group,
        "goal_sha256": goal_sha,
        "stratum": STRATUM,
        "target_text": ATTACK_TARGET,
        "split_rank": rank,
        "split_key": split_key,
        "split_seed": SPLIT_SEED,
        "split_algorithm": SPLIT_ALGORITHM,
    }


def derive_rq2_manifest(
    *,
    dev_path: str | Path,
    causal_test_path: str | Path,
    reserve_path: str | Path,
    split_audit_path: str | Path,
    review_audit_path: str | Path,
    rq1_manifest_paths: Sequence[str | Path],
    decision_date: str,
    expected_dev_count: int = 20,
    expected_causal_test_count: int = 40,
    expected_reserve_count: int = 395,
    minimum_causal_test_count: int = 20,
    expected_split_audit_sha256: str | None = FROZEN_SPLIT_AUDIT_SHA256,
    expected_review_audit_sha256: str | None = FROZEN_REVIEW_AUDIT_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive the minimal source manifest and its audit without writing files."""

    if not isinstance(decision_date, str) or not decision_date.strip():
        raise RQ2ManifestError("decision_date must be non-blank")
    for name, value in (
        ("expected_dev_count", expected_dev_count),
        ("expected_causal_test_count", expected_causal_test_count),
        ("expected_reserve_count", expected_reserve_count),
        ("minimum_causal_test_count", minimum_causal_test_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RQ2ManifestError(f"{name} must be a non-negative integer")
    if expected_dev_count < 1 or expected_causal_test_count < minimum_causal_test_count:
        raise RQ2ManifestError(
            "manifest requires dev>=1 and causal-test above the frozen minimum"
        )

    dev = Path(dev_path).expanduser().resolve()
    causal = Path(causal_test_path).expanduser().resolve()
    reserve = Path(reserve_path).expanduser().resolve()
    split_audit_file = Path(split_audit_path).expanduser().resolve()
    review_audit_file = Path(review_audit_path).expanduser().resolve()
    rq1_paths = tuple(Path(value).expanduser().resolve() for value in rq1_manifest_paths)
    if not rq1_paths:
        raise RQ2ManifestError("at least one complete RQ1 manifest is required")

    split_audit_sha = _validate_expected_sha(
        split_audit_file,
        expected_split_audit_sha256,
        name="split audit",
    )
    review_audit_sha = _validate_expected_sha(
        review_audit_file,
        expected_review_audit_sha256,
        name="review audit",
    )
    split_audit = _json_object(split_audit_file, name="split audit")
    review_audit = _json_object(review_audit_file, name="review audit")
    _validate_split_audit(
        split_audit,
        dev_path=dev,
        causal_path=causal,
        reserve_path=reserve,
        expected_dev_count=expected_dev_count,
        expected_causal_test_count=expected_causal_test_count,
        expected_reserve_count=expected_reserve_count,
    )
    _validate_review_audit(
        review_audit,
        review_path=review_audit_file,
        split_audit=split_audit,
    )
    rq1_identities = _validate_rq1_sources(
        split_audit=split_audit,
        rq1_paths=rq1_paths,
    )

    dev_rows = _rows(dev, name="rq2_dev split")
    causal_rows = _rows(causal, name="rq2_causal_test split")
    reserve_rows = _rows(reserve, name="reserve split")
    if len(dev_rows) != expected_dev_count:
        raise RQ2ManifestError(
            f"rq2_dev count mismatch: {len(dev_rows)} != {expected_dev_count}"
        )
    if len(causal_rows) != expected_causal_test_count:
        raise RQ2ManifestError(
            "rq2_causal_test count mismatch: "
            f"{len(causal_rows)} != {expected_causal_test_count}"
        )
    reserve_ids = _validate_reserve_rows(
        reserve_rows, expected_count=expected_reserve_count
    )
    manifest_rows = [
        *(
            _manifest_row(
                row,
                position=index,
                expected_role="rq2_dev",
                source_path=dev,
                reserve_ids=reserve_ids,
            )
            for index, row in enumerate(dev_rows)
        ),
        *(
            _manifest_row(
                row,
                position=index,
                expected_role="rq2_causal_test",
                source_path=causal,
                reserve_ids=reserve_ids,
            )
            for index, row in enumerate(causal_rows)
        ),
    ]
    manifest_rows.sort(key=lambda row: (int(row["split_rank"]), str(row["pair_id"])))

    total = expected_dev_count + expected_causal_test_count
    ranks = [row["split_rank"] for row in manifest_rows]
    if ranks != list(range(1, total + 1)):
        raise RQ2ManifestError("assigned split ranks must be exactly 1..dev+causal-test")
    pair_ids = [str(row["pair_id"]) for row in manifest_rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise RQ2ManifestError("formal manifest contains duplicate pair_id values")
    case_ids = [str(row["case_id"]) for row in manifest_rows]
    if len(set(case_ids)) != len(case_ids):
        raise RQ2ManifestError("formal manifest contains duplicate case_id values")

    dev_output = [row for row in manifest_rows if row["rq2_role"] == "rq2_dev"]
    causal_output = [
        row for row in manifest_rows if row["rq2_role"] == "rq2_causal_test"
    ]
    dev_identities = _role_identities(dev_output)
    causal_identities = _role_identities(causal_output)
    all_identities = {
        key: dev_identities[key].union(causal_identities[key])
        for key in dev_identities
    }
    expected_unique = len(manifest_rows)
    if (
        len(all_identities["pair_ids"]) != expected_unique
        or len(all_identities["audio_sha256"]) != expected_unique
        or len(all_identities["content_groups"]) != expected_unique * 2
    ):
        raise RQ2ManifestError("formal manifest contains duplicate pair/content/audio identity")
    isolation = {
        "rq2_dev_vs_rq2_causal_test": _overlap_report(
            dev_identities, causal_identities
        ),
        "rq2_dev_vs_rq1": _overlap_report(dev_identities, rq1_identities),
        "rq2_causal_test_vs_rq1": _overlap_report(
            causal_identities, rq1_identities
        ),
    }
    failed = [name for name, report in isolation.items() if report["passed"] is not True]
    if failed:
        raise RQ2ManifestError("identity isolation failed: " + ", ".join(failed))

    manifest_text = _serialize_jsonl(manifest_rows)
    review_protocol = review_audit["protocol_deviation"]
    audit = {
        "format": MANIFEST_AUDIT_FORMAT,
        "version": MANIFEST_AUDIT_VERSION,
        "decision_date": decision_date,
        "frozen": True,
        "manifest_role_counts": {
            "rq2_dev": len(dev_output),
            "rq2_causal_test": len(causal_output),
            "total": len(manifest_rows),
        },
        "minimum_causal_test_count": minimum_causal_test_count,
        "attack_target": {
            "value": ATTACK_TARGET,
            "policy": "uniform-fixed-short-compliance-prefix-v1",
            "advbench_original_target_included": False,
        },
        "trajectory_contract": {
            "mode": "generate",
            "initial_manifest_has_trajectory_path": False,
            "resolved_manifest_created_by_pipeline": True,
        },
        "inputs": {
            "rq2_dev": {
                "path": _portable_path(dev),
                "sha256": file_sha256(dev),
                "rows": len(dev_rows),
            },
            "rq2_causal_test": {
                "path": _portable_path(causal),
                "sha256": file_sha256(causal),
                "rows": len(causal_rows),
            },
            "reserve": {
                "path": _portable_path(reserve),
                "sha256": file_sha256(reserve),
                "rows": len(reserve_rows),
                "included_in_manifest": False,
                "reason": RESERVE_REASON,
            },
            "split_audit": {
                "path": _portable_path(split_audit_file),
                "sha256": split_audit_sha,
            },
            "review_protocol_deviation_audit": {
                "path": _portable_path(review_audit_file),
                "sha256": review_audit_sha,
            },
            "rq1_manifests": [
                {"path": _portable_path(path), "sha256": file_sha256(path)}
                for path in rq1_paths
            ],
        },
        "output": {
            "manifest": {
                "file": "rq2_manifest.jsonl",
                "sha256": _text_sha256(manifest_text),
                "rows": len(manifest_rows),
            }
        },
        "identity_isolation": isolation,
        "privacy_contract": {
            "output_fields": list(OUTPUT_FIELDS),
            "forbidden_source_fields": sorted(FORBIDDEN_SOURCE_FIELDS),
            "contains_model_response_text": False,
            "contains_judge_reasoning": False,
            "contains_advbench_original_target": False,
        },
        "inherited_review_limitations": {
            "manual_semantic_review_performed": review_protocol[
                "manual_semantic_review_performed"
            ],
            "audio_content_fidelity_review_performed": review_protocol[
                "audio_content_fidelity_review_performed"
            ],
            "claims_not_allowed": review_protocol.get("claims_not_allowed", []),
        },
        "formal_manifest_ready": True,
        "requires_trajectory_generation": True,
    }
    return manifest_rows, audit


def _expected_artifacts(**kwargs: Any) -> tuple[dict[str, str], dict[str, Any]]:
    rows, audit = derive_rq2_manifest(**kwargs)
    contents = {
        "rq2_manifest.jsonl": _serialize_jsonl(rows),
        "manifest_audit.json": _serialize_json(audit),
    }
    return contents, audit


def write_rq2_manifest(*, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    contents, audit = _expected_artifacts(**kwargs)
    destination = Path(output_dir).expanduser().resolve()
    for name, content in contents.items():
        atomic_text(destination / name, content)
    return {
        "status": "BUILT",
        "output_dir": str(destination),
        "counts": audit["manifest_role_counts"],
        "attack_target": ATTACK_TARGET,
        "manifest_sha256": audit["output"]["manifest"]["sha256"],
        "manifest_audit_sha256": _text_sha256(contents["manifest_audit.json"]),
    }


def check_rq2_manifest(*, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    contents, audit = _expected_artifacts(**kwargs)
    destination = Path(output_dir).expanduser().resolve()
    for name, content in contents.items():
        path = destination / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RQ2ManifestError(
                f"manifest artifact is missing, stale, or modified: {path}"
            )
    return {
        "status": "VALID",
        "output_dir": str(destination),
        "counts": audit["manifest_role_counts"],
        "attack_target": ATTACK_TARGET,
        "manifest_sha256": audit["output"]["manifest"]["sha256"],
        "manifest_audit_sha256": _text_sha256(contents["manifest_audit.json"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument("--dev", default=str(DEFAULT_SPLIT_DIR / "rq2_dev.jsonl"))
    parser.add_argument(
        "--causal-test", default=str(DEFAULT_SPLIT_DIR / "rq2_causal_test.jsonl")
    )
    parser.add_argument(
        "--reserve", default=str(DEFAULT_SPLIT_DIR / "reserve.jsonl")
    )
    parser.add_argument(
        "--split-audit", default=str(DEFAULT_SPLIT_DIR / "split_audit.json")
    )
    parser.add_argument("--review-audit", default=str(DEFAULT_REVIEW_AUDIT))
    parser.add_argument(
        "--rq1-manifest",
        nargs="+",
        default=[str(path) for path in DEFAULT_RQ1_MANIFESTS],
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--decision-date", default="2026-10-02")
    parser.add_argument("--expected-dev-count", type=int, default=20)
    parser.add_argument("--expected-causal-test-count", type=int, default=40)
    parser.add_argument("--expected-reserve-count", type=int, default=395)
    parser.add_argument("--minimum-causal-test-count", type=int, default=20)
    parser.add_argument(
        "--expected-split-audit-sha256", default=FROZEN_SPLIT_AUDIT_SHA256
    )
    parser.add_argument(
        "--expected-review-audit-sha256", default=FROZEN_REVIEW_AUDIT_SHA256
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "dev_path": args.dev,
        "causal_test_path": args.causal_test,
        "reserve_path": args.reserve,
        "split_audit_path": args.split_audit,
        "review_audit_path": args.review_audit,
        "rq1_manifest_paths": args.rq1_manifest,
        "output_dir": args.output_dir,
        "decision_date": args.decision_date,
        "expected_dev_count": args.expected_dev_count,
        "expected_causal_test_count": args.expected_causal_test_count,
        "expected_reserve_count": args.expected_reserve_count,
        "minimum_causal_test_count": args.minimum_causal_test_count,
        "expected_split_audit_sha256": args.expected_split_audit_sha256,
        "expected_review_audit_sha256": args.expected_review_audit_sha256,
    }
    try:
        result = (
            write_rq2_manifest(**kwargs)
            if args.mode == "build"
            else check_rq2_manifest(**kwargs)
        )
    except RQ2ManifestError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTACK_TARGET",
    "FROZEN_REVIEW_AUDIT_SHA256",
    "FROZEN_SPLIT_AUDIT_SHA256",
    "MANIFEST_AUDIT_FORMAT",
    "MANIFEST_AUDIT_VERSION",
    "OUTPUT_FIELDS",
    "RQ2ManifestError",
    "check_rq2_manifest",
    "derive_rq2_manifest",
    "write_rq2_manifest",
]
