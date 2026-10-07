"""Read-only, outcome-blind preparation for ONE independent RQ2 dev replication.

Prints an audit and a *non-executable* identity-only selection to stdout. Never
writes files, imports a model, reads credentials, or opens behavior labels /
responses. This is repository evidence, not proof of absence of external use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path("dataset/processed/rq2/advbench_split_v1")
FROZEN_AUDIT_SHA = "0cc3f20ed40a654d59aa5953c8c450917618033360393e2844b11f34302b8c48"
PAIR_RE = re.compile(r"advbench_\d{4}(?!\d)")
HASH_RE = re.compile(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])")
METADATA_NAMES = frozenset({
    "index.json", "summary.json", "resolved_manifest.json", "scan_index.json",
    "pipeline_state.json", "trials.jsonl", "commits.jsonl", "plan.json",
    "manifest.json", "manifest.jsonl", "generation_status.jsonl",
})
KNOWN_OUTPUT_ROOTS = {"stage1", "stage1_v2", "stage2_rq2"}
KNOWN_RQ2_ROOTS = {"advbench_clean_screening_run01", "smoke", "dev_screen",
                   "event_dev", "judge_revision", "diagnostics"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode())


def normalized_text_sha(text):
    return sha_bytes(" ".join(text.casefold().split()).encode())


def identities(rows):
    result = {k: set() for k in ("pair_ids", "content_groups", "audio_sha256")}
    for row in rows:
        result["pair_ids"].add(row["pair_id"])
        result["content_groups"].add("text:" + normalized_text_sha(row["harmful_text"]))
        if row.get("content_group"):
            result["content_groups"].add("group:" + row["content_group"])
        if row.get("clean_audio_sha256"):
            result["audio_sha256"].add(row["clean_audio_sha256"])
    return result


def overlap(left, right):
    counts = {key: len(left[key] & right[key]) for key in left}
    return {**counts, "passed": not any(counts.values())}


def protected_path(path):
    return any("formal" in part.casefold() or "causal_test" in part.casefold()
               or "causal-test" in part.casefold() for part in Path(path).parts)


def usage_tokens(value, identity_lookup):
    """Conservatively treat identity references in non-clean metadata as use."""
    hits = set()
    if isinstance(value, dict):
        for key, item in value.items():
            hits.update(usage_tokens(key, identity_lookup))
            if key == "harmful_text" and isinstance(item, str):
                hits.update(identity_lookup.get(normalized_text_sha(item), ()))
            hits.update(usage_tokens(item, identity_lookup))
    elif isinstance(value, list):
        for item in value:
            hits.update(usage_tokens(item, identity_lookup))
    elif isinstance(value, str):
        hits.update(identity_lookup.get(value, ()))
        for token in (*PAIR_RE.findall(value), *HASH_RE.findall(value)):
            hits.update(identity_lookup.get(token, ()))
    return hits


def select_rows(rows, used_ids, count=20):
    if not isinstance(count, int) or isinstance(count, bool) or count != 20:
        raise ValueError("This draft fixes the source population to 20; no adaptive size.")
    ordered = sorted(rows, key=lambda r: (r["split_rank"], r["pair_id"]))
    eligible = [r for r in ordered if r["pair_id"] not in used_ids]
    if len(eligible) < count:
        raise ValueError("Insufficient unused reserve; do not borrow causal-test.")
    return eligible[:count]


def audit(root=ROOT):
    root = Path(root).resolve()
    bound = {}

    def read(rel, expected=None):
        path = root / rel
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Input escapes repository: {rel}")
        data = path.read_bytes()
        digest = sha_bytes(data)
        if expected is not None and digest != expected:
            raise ValueError(f"Frozen input changed: {rel}")
        bound[str(rel)] = digest
        return data

    split = json.loads(read(SPLIT / "split_audit.json", FROZEN_AUDIT_SHA))
    assert split["frozen"] and split["roles_locked"] and split["selection_outcome_blind"]
    assert split["counts"] == {"candidate_pool": 455, "reserve": 395,
                              "rq2_causal_test": 40, "rq2_dev": 20}
    sources = {}
    for role in ("reserve", "rq2_dev", "rq2_causal_test", "split_assignments"):
        entry = split["outputs"][role]
        raw = read(SPLIT / entry["file"], entry["sha256"])
        sources[role] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        assert len(sources[role]) == entry["rows"]
    reserve = sources["reserve"]
    assert len({r["pair_id"] for r in reserve}) == 395
    assignments = {r["pair_id"]: r for r in sources["split_assignments"]}
    lookup = defaultdict(set)
    for row in reserve:
        pid = row["pair_id"]
        assert row["split_role"] == "reserve" and "rq2_role" not in row
        assert row["candidate_pool_status"] == "reserve" and row["clean_refused"] is True
        assert row["split_seed"] == 42 and row["split_algorithm"] == split["algorithm"]
        expected_key = canonical_sha({"algorithm": split["algorithm"], "seed": 42,
            "stratum": row["stratum"], "pair_id": pid, "goal_sha256": row["goal_sha256"],
            "clean_audio_sha256": row["clean_audio_sha256"]})
        assert expected_key == row["split_key"]
        for field in ("split_role", "split_rank", "split_key", "content_group",
                      "goal_sha256", "clean_audio_sha256"):
            assert row[field] == assignments[pid][field]
        read(Path(row["clean_audio_path"]), row["clean_audio_sha256"])
        for token in (pid, row["goal_sha256"], row["content_group"],
                      row["clean_audio_sha256"], normalized_text_sha(row["harmful_text"])):
            lookup[token].add(pid)
    assert [r["split_rank"] for r in sorted(reserve, key=lambda r: r["split_key"])] == list(range(61, 456))

    rq1_rows = []
    for entry in split["inputs"]["rq1_manifests"]:
        raw = read(Path(entry["path"]), entry["sha256"]).decode("utf-8-sig")
        for source in csv.DictReader(raw.splitlines()):
            row = dict(source)
            audio = row.get("clean_audio_path") or row.get("harmful_audio_path")
            if audio:
                row["clean_audio_sha256"] = sha_bytes(read(Path(audio)))
            rq1_rows.append(row)
    isolations = {"reserve_vs_rq1": overlap(identities(reserve), identities(rq1_rows))}
    for role in ("rq2_dev", "rq2_causal_test"):
        isolations["reserve_vs_" + role] = overlap(identities(reserve), identities(sources[role]))
    if not all(item["passed"] for item in isolations.values()):
        raise ValueError("Reserve identity overlap; stop before selection.")

    # Validate inherited conservative-exclusion evidence without opening responses.
    pool_info = split["inputs"]["candidate_pool_summary"]
    pool = json.loads(read(Path(pool_info["path"]), pool_info["sha256"]))
    candidate_info = split["inputs"]["candidate_pool"]
    candidate_rows = [json.loads(line) for line in read(Path(candidate_info["path"]),
                      candidate_info["sha256"]).decode().splitlines() if line.strip()]
    assert set(r["pair_id"] for r in reserve) <= set(r["pair_id"] for r in candidate_rows)

    inventory = sorted(p for p in (root / "outputs").rglob("*") if p.is_file())
    unknown_roots = sorted(p.name for p in (root / "outputs").iterdir()
                           if p.is_dir() and p.name not in KNOWN_OUTPUT_ROOTS)
    unknown_rq2 = sorted(p.name for p in (root / "outputs/stage2_rq2").iterdir()
                         if p.is_dir() and p.name not in KNOWN_RQ2_ROOTS)
    blocked_paths, hits, scanned = [], defaultdict(set), []
    clean_count, skipped_content = 0, 0
    for path in inventory:
        rel = path.relative_to(root)
        if path.is_symlink():
            blocked_paths.append(str(rel))
            continue
        if protected_path(rel):
            blocked_paths.append(str(rel))
            continue
        if str(rel).startswith("outputs/stage2_rq2/advbench_clean_screening_run01/"):
            clean_count += 1
            continue
        for pid in usage_tokens(str(rel), lookup):
            hits[pid].add(str(rel) + "#path_identity")
        if path.name not in METADATA_NAMES:
            skipped_content += 1
            continue
        raw = read(rel).decode()
        values = [json.loads(line) for line in raw.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(raw)
        for pid in usage_tokens(values, lookup):
            hits[pid].add(str(rel))
        scanned.append(str(rel))
    blockers = []
    if unknown_roots or unknown_rq2:
        blockers.append("unclassified_output_namespaces")
    if blocked_paths:
        blockers.append("protected_or_symlink_paths_require_separate_inventory_review")
    selected = [] if blockers else select_rows(reserve, set(hits))
    manifest = []
    for row in selected:
        manifest.append({
            "format": "rq2-replication-selection-draft", "version": 1,
            "execution_enabled": False, "proposed_role": "independent_replication_dev",
            "source_role": "reserve", "source_manifest": str(SPLIT / "reserve.jsonl"),
            "source_manifest_sha256": split["outputs"]["reserve"]["sha256"],
            **{k: row[k] for k in ("pair_id", "split_rank", "split_key", "goal_sha256",
                                   "content_group", "clean_audio_path", "clean_audio_sha256")},
            "normalized_text_sha256": normalized_text_sha(row["harmful_text"]),
        })
    record = {
        "format": "rq2-replication-reserve-audit", "version": 1,
        "status": "blocked" if blockers else "offline_checks_passed_draft_only",
        "execution_enabled": False, "new_experiment_authorized": False,
        "source_population": 395, "selected_count": len(manifest),
        "selection_rule": "first_20_remaining_in_original_frozen_split_rank_order",
        "selection_seed": 42, "sample_replacement_after_selection_allowed": False,
        "exclusion_rule": "any_reserve_identity_reference_in_non_clean_output_paths_or_allowlisted_metadata",
        "identity_isolation": isolations, "reserve_audio_hashes_verified": len(reserve),
        "selected_pair_ids": [r["pair_id"] for r in manifest],
        "selected_split_ranks": [r["split_rank"] for r in manifest],
        "prior_nonclean_usage": {pid: sorted(paths) for pid, paths in sorted(hits.items())},
        "inventory": {"scope": "current_repository_outputs_only",
            "file_count": len(inventory),
            "relative_paths_sha256": canonical_sha([str(p.relative_to(root)) for p in inventory]),
            "metadata_files_read": scanned, "metadata_file_count": len(scanned),
            "metadata_name_allowlist": sorted(METADATA_NAMES),
            "clean_screening_files_exempt": clean_count,
            "other_file_contents_not_read": skipped_content,
            "unknown_output_roots": unknown_roots, "unknown_rq2_roots": unknown_rq2,
            "protected_or_symlink_paths_not_read": blocked_paths},
        "behavior_labels_or_responses_read": False,
        "causal_test_access": "identity_manifest_only_no_behavior",
        "limitations": ["repository_metadata_and_path_audit_not_external_usage_proof",
            "reserve_was_previously_clean_screened_not_behavior_naive",
            "manual_semantic_independence_not_verified",
            "audio_content_fidelity_not_verified",
            "unindexed_or_alias_only_usage_may_be_missed",
            "data_use_history_confirmation_required_before_execution"],
        "inherited_pool_format": pool.get("format"),
        "blockers": blockers,
        "bound_inputs": dict(sorted(bound.items())),
    }
    return {"audit": record, "selection_manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if result["audit"]["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
