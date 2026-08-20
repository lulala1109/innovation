"""Build and validate paired X_B / X_H / X_J safety-state manifests.

The canonical representation is one wide row per semantic pair.  This keeps
the benign input (X_B), clean harmful/refused input (X_H), and successful
jailbreak input (X_J) tied to one immutable ``pair_id`` and makes accidental
cross-pair joins detectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from data.datasets import (
    assign_stratum,
    load_advbench,
    load_jbb,
    normalize_prompt_table,
    read_table,
    write_table,
)


MANIFEST_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "source",
    "stratum",
    "benign_text",
    "harmful_text",
    "benign_response",
    "benign_refused",
    "clean_response",
    "clean_refused",
    "jailbreak_response",
    "jailbreak_success",
    "benign_audio_path",
    "clean_audio_path",
    "jailbreak_audio_path",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "source",
    "stratum",
    "benign_text",
    "harmful_text",
)

COMPLETION_COLUMNS: tuple[str, ...] = (
    "benign_response",
    "clean_response",
    "jailbreak_response",
    "clean_audio_path",
    "jailbreak_audio_path",
)

STATE_AUDIO_COLUMNS: tuple[str, ...] = (
    "benign_audio_path",
    "clean_audio_path",
    "jailbreak_audio_path",
)

BOOLEAN_COLUMNS: tuple[str, ...] = (
    "benign_refused",
    "clean_refused",
    "jailbreak_success",
)

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "pair_id": ("pair_id", "id"),
    "source": ("source", "dataset"),
    "stratum": ("stratum", "category"),
    "benign_text": (
        "benign_text",
        "benign_prompt",
        "benign_goal",
        "benign",
    ),
    "harmful_text": (
        "harmful_text",
        "harmful_prompt",
        "harmful_goal",
        "prompt",
        "goal",
    ),
    "benign_response": (
        "benign_response",
        "benign_clean_response",
        "x_b_response",
    ),
    "benign_refused": (
        "benign_refused",
        "benign_is_refused",
        "x_b_refused",
    ),
    "clean_response": (
        "clean_response",
        "harmful_clean_response",
        "original_response",
    ),
    "clean_refused": ("clean_refused", "is_refused", "refused"),
    "jailbreak_response": (
        "jailbreak_response",
        "adversarial_response",
        "attack_response",
    ),
    "jailbreak_success": (
        "jailbreak_success",
        "attack_success",
        "is_success",
    ),
    "benign_audio_path": (
        "benign_audio_path",
        "benign_wav_path",
        "benign_audio",
        "x_b_audio_path",
        "x_b_path",
    ),
    "clean_audio_path": (
        "clean_audio_path",
        "original_audio_path",
        "harmful_audio_path",
        "x_h_audio_path",
    ),
    "jailbreak_audio_path": (
        "jailbreak_audio_path",
        "adversarial_audio_path",
        "attack_audio_path",
        "x_j_audio_path",
    ),
}


class ManifestValidationError(ValueError):
    """Raised when a safety-state manifest violates pairing invariants."""


def _normalize_identity_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip()


def stable_pair_id(
    *,
    source: object,
    stratum: object,
    benign_text: object,
    harmful_text: object,
) -> str:
    """Derive a stable content-addressed pair identifier."""

    identity = [
        _normalize_identity_text(source),
        _normalize_identity_text(stratum),
        _normalize_identity_text(benign_text),
        _normalize_identity_text(harmful_text),
    ]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"pair_{digest}"


def _resolve_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    casefolded = {str(column).casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        resolved = casefolded.get(alias.casefold())
        if resolved is not None:
            return resolved
    return None


def _string_series(
    frame: pd.DataFrame,
    column: str | None,
    *,
    default: str = "",
) -> pd.Series:
    if column is None:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).map(_normalize_identity_text)


def _parse_optional_bool(value: object) -> bool | pd._libs.missing.NAType:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Real) and value in {0, 1}:
        return bool(value)

    text = str(value).strip().casefold()
    if text in {"true", "t", "yes", "y", "1", "refused", "success"}:
        return True
    if text in {"false", "f", "no", "n", "0", "complied", "failure", "failed"}:
        return False
    if text in {"", "na", "n/a", "none", "null", "unknown"}:
        return pd.NA
    raise ValueError(f"not a boolean value: {value!r}")


def _boolean_series(
    frame: pd.DataFrame,
    column: str | None,
    *,
    canonical_name: str,
) -> pd.Series:
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    parsed: list[bool | pd._libs.missing.NAType] = []
    for index, value in frame[column].items():
        try:
            parsed.append(_parse_optional_bool(value))
        except ValueError as exc:
            raise ManifestValidationError(
                f"Invalid {canonical_name} value at row {index}: {value!r}"
            ) from exc
    return pd.Series(parsed, index=frame.index, dtype="boolean")


def build_manifest(
    records: pd.DataFrame,
    *,
    default_source: str = "local",
    regenerate_pair_ids: bool = False,
    require_complete: bool = False,
    require_state_triplets: bool = False,
) -> pd.DataFrame:
    """Canonicalize paired records and validate their state relationships."""

    if records.empty:
        raise ManifestValidationError("Cannot build a manifest from an empty table")

    resolved = {
        canonical: _resolve_column(records, aliases)
        for canonical, aliases in COLUMN_ALIASES.items()
    }
    manifest = pd.DataFrame(index=records.index)
    for name in MANIFEST_COLUMNS:
        if name in BOOLEAN_COLUMNS:
            manifest[name] = _boolean_series(
                records, resolved[name], canonical_name=name
            )
        else:
            manifest[name] = _string_series(records, resolved[name])

    manifest.loc[manifest["source"] == "", "source"] = _normalize_identity_text(
        default_source
    )
    missing_strata = manifest["stratum"] == ""
    manifest.loc[missing_strata, "stratum"] = manifest.loc[
        missing_strata, "harmful_text"
    ].map(assign_stratum)

    if regenerate_pair_ids:
        manifest["pair_id"] = ""
    missing_ids = manifest["pair_id"] == ""
    manifest.loc[missing_ids, "pair_id"] = manifest.loc[missing_ids].apply(
        lambda row: stable_pair_id(
            source=row["source"],
            stratum=row["stratum"],
            benign_text=row["benign_text"],
            harmful_text=row["harmful_text"],
        ),
        axis=1,
    )

    # Preserve provenance/experiment columns that do not alias canonical fields.
    alias_names = {
        alias.casefold()
        for aliases in COLUMN_ALIASES.values()
        for alias in aliases
    }
    for column in records.columns:
        if str(column).casefold() not in alias_names and column not in manifest.columns:
            manifest[column] = records[column].values

    manifest = manifest.reset_index(drop=True)
    validate_manifest(
        manifest,
        require_complete=require_complete,
        require_state_triplets=require_state_triplets,
    )
    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    require_complete: bool = False,
    require_state_triplets: bool = False,
) -> None:
    """Validate schema, unique pairing, and X_B/X_H/X_J consistency.

    A valid safety-state triplet has the following independently observed
    labels: ``X_B = H0/R0``, ``X_H = H1/R1``, and ``X_J = H1/R0``.  In
    particular, a benign response is never assumed to be non-refusing merely
    because its prompt is benign.
    """

    missing_columns = [name for name in MANIFEST_COLUMNS if name not in manifest.columns]
    if missing_columns:
        raise ManifestValidationError(
            "Manifest is missing required column(s): " + ", ".join(missing_columns)
        )
    if manifest.empty:
        raise ManifestValidationError("Manifest has no rows")

    for name in IDENTITY_COLUMNS:
        blank = manifest[name].map(_normalize_identity_text) == ""
        if blank.any():
            rows = manifest.index[blank].tolist()[:5]
            raise ManifestValidationError(f"{name} is blank at row(s): {rows}")

    ids = manifest["pair_id"].map(_normalize_identity_text)
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ManifestValidationError(
            "pair_id must be unique; duplicate(s): " + ", ".join(duplicates[:5])
        )

    identity_keys = manifest.apply(
        lambda row: stable_pair_id(
            source=row["source"],
            stratum=row["stratum"],
            benign_text=row["benign_text"],
            harmful_text=row["harmful_text"],
        ),
        axis=1,
    )
    duplicate_identities = identity_keys[
        identity_keys.duplicated(keep=False)
    ]
    if not duplicate_identities.empty:
        raise ManifestValidationError(
            "Semantic pairs must be unique even when pair_id values differ; "
            "duplicate row(s): "
            + str(duplicate_identities.index.tolist()[:5])
        )

    same_text = manifest.apply(
        lambda row: _normalize_identity_text(row["benign_text"])
        == _normalize_identity_text(row["harmful_text"]),
        axis=1,
    )
    if same_text.any():
        rows = manifest.index[same_text].tolist()[:5]
        raise ManifestValidationError(
            f"benign_text and harmful_text are identical at row(s): {rows}"
        )

    parsed_booleans: dict[str, pd.Series] = {}
    for name in BOOLEAN_COLUMNS:
        parsed_booleans[name] = _boolean_series(
            manifest, name, canonical_name=name
        )

    if require_complete:
        for name in COMPLETION_COLUMNS:
            blank = manifest[name].map(_normalize_identity_text) == ""
            if blank.any():
                rows = manifest.index[blank].tolist()[:5]
                raise ManifestValidationError(f"{name} is blank at row(s): {rows}")
        for name, values in parsed_booleans.items():
            if values.isna().any():
                rows = manifest.index[values.isna()].tolist()[:5]
                raise ManifestValidationError(f"{name} is unknown at row(s): {rows}")

    benign_refused = parsed_booleans["benign_refused"]
    clean_refused = parsed_booleans["clean_refused"]
    jailbreak_success = parsed_booleans["jailbreak_success"]
    valid_benign = benign_refused.eq(False).fillna(False)
    valid_clean_harmful = clean_refused.eq(True).fillna(False)
    valid_jailbreak = jailbreak_success.eq(True).fillna(False)
    orphan_jailbreak = valid_jailbreak & ~(
        valid_benign & valid_clean_harmful
    )
    if orphan_jailbreak.any():
        rows = manifest.index[orphan_jailbreak].tolist()[:5]
        raise ManifestValidationError(
            "X_J must be paired with a non-refusing X_B and a refused X_H "
            "state; violation at row(s): "
            f"{rows}"
        )

    if require_state_triplets:
        not_triplet = ~(
            valid_benign & valid_clean_harmful & valid_jailbreak
        )
        if not_triplet.any():
            rows = manifest.index[not_triplet].tolist()[:5]
            raise ManifestValidationError(
                "Every row must form an X_B/X_H/X_J triplet; incomplete row(s): "
                f"{rows}"
            )
        for name in (*STATE_AUDIO_COLUMNS, *COMPLETION_COLUMNS):
            blank = manifest[name].map(_normalize_identity_text) == ""
            if blank.any():
                rows = manifest.index[blank].tolist()[:5]
                raise ManifestValidationError(
                    f"Every state triplet requires {name}; blank at row(s): {rows}"
                )


def _state_view(
    rows: pd.DataFrame,
    *,
    state: str,
    text_column: str,
    audio_column: str,
    response_column: str | None,
    harmfulness_label: int,
    refusal_label: int,
    state_jailbreak_success: bool,
) -> pd.DataFrame:
    """Attach a common state-level interface while preserving manifest fields."""

    view = rows.copy().reset_index(drop=True)
    view["state"] = state
    view["text"] = view[text_column]
    view["audio_path"] = view[audio_column]
    view["response"] = "" if response_column is None else view[response_column]
    view["harmfulness_label"] = harmfulness_label
    view["state_jailbreak_success"] = state_jailbreak_success
    view["refusal_label"] = refusal_label
    return view


def select_state_sets(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return uniquely paired X_B, X_H, and X_J state-level views.

    Every view retains the original wide manifest columns and adds the common
    ``state``, ``text``, ``audio_path``, ``response``,
    ``harmfulness_label``, ``refusal_label``, and
    ``state_jailbreak_success`` fields. This lets downstream
    collectors iterate states without guessing which wide-table column belongs
    to a particular state.
    """

    validate_manifest(manifest)
    benign_refused = _boolean_series(
        manifest, "benign_refused", canonical_name="benign_refused"
    )
    clean_refused = _boolean_series(
        manifest, "clean_refused", canonical_name="clean_refused"
    )
    jailbreak_success = _boolean_series(
        manifest, "jailbreak_success", canonical_name="jailbreak_success"
    )
    valid_benign = benign_refused.eq(False).fillna(False)
    valid_clean_harmful = valid_benign & clean_refused.eq(True).fillna(False)
    valid_jailbreak = (
        valid_clean_harmful & jailbreak_success.eq(True).fillna(False)
    )
    return {
        "X_B": _state_view(
            manifest.loc[valid_benign],
            state="X_B",
            text_column="benign_text",
            audio_column="benign_audio_path",
            response_column="benign_response",
            state_jailbreak_success=False,
            harmfulness_label=0,
            refusal_label=0,
        ),
        "X_H": _state_view(
            manifest.loc[valid_clean_harmful],
            state="X_H",
            text_column="harmful_text",
            audio_column="clean_audio_path",
            response_column="clean_response",
            state_jailbreak_success=False,
            harmfulness_label=1,
            refusal_label=1,
        ),
        "X_J": _state_view(
            manifest.loc[valid_jailbreak],
            state="X_J",
            text_column="harmful_text",
            audio_column="jailbreak_audio_path",
            state_jailbreak_success=True,
            response_column="jailbreak_response",
            harmfulness_label=1,
            refusal_label=0,
        ),
    }


def pair_prompt_tables(
    harmful: pd.DataFrame,
    benign: pd.DataFrame,
    *,
    pair_on: str | None = None,
    pair_by_row: bool = False,
    default_source: str = "local",
) -> pd.DataFrame:
    """Pair canonical prompt tables using a unique key or explicit row order."""

    if pair_on is not None and pair_by_row:
        raise ValueError("Choose pair_on or pair_by_row, not both")
    if pair_on is None and not pair_by_row:
        raise ValueError(
            "Pairing must be explicit: provide pair_on or set pair_by_row=True"
        )

    for label, frame in (("harmful", harmful), ("benign", benign)):
        if "prompt" not in frame.columns:
            raise ManifestValidationError(f"{label} table is not a canonical prompt table")

    if pair_by_row:
        if len(harmful) != len(benign):
            raise ManifestValidationError(
                "Row pairing requires harmful and benign tables with equal length"
            )
        harmful_ordered = harmful.reset_index(drop=True)
        benign_ordered = benign.reset_index(drop=True)
    else:
        assert pair_on is not None
        for label, frame in (("harmful", harmful), ("benign", benign)):
            if pair_on not in frame.columns:
                raise ManifestValidationError(
                    f"{label} table is missing pairing column: {pair_on}"
                )
            if frame[pair_on].isna().any() or (
                frame[pair_on].astype(str).str.strip() == ""
            ).any():
                raise ManifestValidationError(
                    f"{label} pairing column {pair_on!r} contains blank values"
                )
            if frame[pair_on].duplicated().any():
                raise ManifestValidationError(
                    f"{label} pairing column {pair_on!r} must be unique"
                )
        harmful_keys = set(harmful[pair_on].tolist())
        benign_keys = set(benign[pair_on].tolist())
        if harmful_keys != benign_keys:
            missing_benign = harmful_keys - benign_keys
            missing_harmful = benign_keys - harmful_keys
            raise ManifestValidationError(
                "Pairing keys differ between tables "
                f"(missing benign={len(missing_benign)}, "
                f"missing harmful={len(missing_harmful)})"
            )
        benign_indexed = benign.set_index(pair_on, drop=False)
        harmful_ordered = harmful.reset_index(drop=True)
        benign_ordered = benign_indexed.loc[harmful_ordered[pair_on]].reset_index(
            drop=True
        )

    records = pd.DataFrame(
        {
            "source": harmful_ordered.get(
                "source", pd.Series(default_source, index=harmful_ordered.index)
            ),
            "stratum": harmful_ordered.get(
                "stratum",
                harmful_ordered["prompt"].map(assign_stratum),
            ),
            "benign_text": benign_ordered["prompt"].values,
            "harmful_text": harmful_ordered["prompt"].values,
        }
    )
    records.loc[records["source"].astype(str).str.strip() == "", "source"] = (
        default_source
    )
    return build_manifest(records, default_source=default_source)


def _load_local_prompt_table(
    path: Path,
    *,
    kind: str,
    split: str,
    default_source: str,
) -> pd.DataFrame:
    if kind == "advbench":
        return load_advbench(path)
    if kind == "jbb":
        return load_jbb(path, split=split)
    return normalize_prompt_table(
        read_table(path), default_source=default_source, infer_strata=True
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate a local X_B/X_H/X_J safety-pair manifest. "
            "Hugging Face access occurs only when --hf-dataset is selected."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Local paired manifest, or harmful prompt table with --benign-table",
    )
    source.add_argument(
        "--hf-dataset",
        choices=("advbench", "jbb"),
        help="Optionally load the harmful prompt table from Hugging Face",
    )
    parser.add_argument("--benign-table", type=Path)
    parser.add_argument(
        "--local-kind", choices=("generic", "advbench", "jbb"), default="generic"
    )
    pairing = parser.add_mutually_exclusive_group()
    pairing.add_argument("--pair-on", help="Unique column shared by prompt tables")
    pairing.add_argument(
        "--pair-by-row",
        action="store_true",
        help="Explicitly pair equally sized tables by row order",
    )
    parser.add_argument("--source", default="local", help="Fallback source label")
    parser.add_argument("--output", type=Path, help="Canonical manifest output")
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Also write x_b.csv, x_h.csv, and x_j.csv views",
    )
    parser.add_argument("--regenerate-pair-ids", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-state-triplets", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --input without writing a new manifest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        if args.input is None or args.benign_table is not None:
            raise ValueError("--validate-only requires a single local --input manifest")
        manifest = build_manifest(
            read_table(args.input),
            default_source=args.source,
            regenerate_pair_ids=False,
            require_complete=args.require_complete,
            require_state_triplets=args.require_state_triplets,
        )
    elif args.benign_table is None and args.input is not None:
        manifest = build_manifest(
            read_table(args.input),
            default_source=args.source,
            regenerate_pair_ids=args.regenerate_pair_ids,
            require_complete=args.require_complete,
            require_state_triplets=args.require_state_triplets,
        )
    else:
        if args.hf_dataset == "advbench":
            harmful = load_advbench()
        elif args.hf_dataset == "jbb":
            harmful = load_jbb(split="harmful")
        elif args.input is not None:
            harmful = _load_local_prompt_table(
                args.input,
                kind=args.local_kind,
                split="harmful",
                default_source=args.source,
            )
        else:  # pragma: no cover - guarded by argparse
            raise RuntimeError("No harmful input selected")

        if args.benign_table is not None:
            benign = _load_local_prompt_table(
                args.benign_table,
                kind=args.local_kind,
                split="benign",
                default_source=args.source,
            )
        elif args.hf_dataset == "jbb":
            benign = load_jbb(split="benign")
        else:
            raise ValueError("A benign table is required for this harmful dataset")

        manifest = pair_prompt_tables(
            harmful,
            benign,
            pair_on=args.pair_on,
            pair_by_row=args.pair_by_row,
            default_source=args.source,
        )
        if args.regenerate_pair_ids:
            manifest = build_manifest(manifest, regenerate_pair_ids=True)
        validate_manifest(
            manifest,
            require_complete=args.require_complete,
            require_state_triplets=args.require_state_triplets,
        )

    if not args.validate_only:
        if args.output is None:
            raise ValueError("--output is required unless --validate-only is used")
        write_table(manifest, args.output)
        print(f"Saved {len(manifest)} paired rows to {args.output}")

    states = select_state_sets(manifest)
    if args.state_dir is not None:
        args.state_dir.mkdir(parents=True, exist_ok=True)
        for state_name, state_frame in states.items():
            write_table(state_frame, args.state_dir / f"{state_name.lower()}.csv")
    print(
        "Validated states: "
        + ", ".join(f"{name}={len(frame)}" for name, frame in states.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOLEAN_COLUMNS",
    "COMPLETION_COLUMNS",
    "IDENTITY_COLUMNS",
    "MANIFEST_COLUMNS",
    "STATE_AUDIO_COLUMNS",
    "ManifestValidationError",
    "build_manifest",
    "pair_prompt_tables",
    "select_state_sets",
    "stable_pair_id",
    "validate_manifest",
]
