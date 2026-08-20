"""Deterministic stratified sampling utilities.

This module contains the reusable statistical logic from the legacy
``sample_advbench.py`` script without coupling it to one dataset or CSV-only
I/O.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Hashable

import pandas as pd

from data.datasets import read_table, write_table


CONFIDENCE_Z: dict[float, float] = {
    0.80: 1.282,
    0.85: 1.440,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Backwards-friendly name for callers migrating from the old script.
CONF_Z = CONFIDENCE_Z


def fpc_sample_size(population: int, p: float, margin: float, z: float) -> int:
    """Return a finite-population-corrected sample size for one proportion."""

    if population < 0:
        raise ValueError("population must be non-negative")
    if population == 0:
        return 0
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    if margin <= 0:
        raise ValueError("margin must be positive")
    if z <= 0:
        raise ValueError("z must be positive")

    variance = z**2 * p * (1 - p)
    numerator = variance * population
    denominator = margin**2 * (population - 1) + variance
    return min(population, math.ceil(numerator / denominator))


def fpc_n(N: int, p: float, e: float, z: float) -> int:
    """Compatibility alias using the argument names from the legacy utility."""

    return fpc_sample_size(N, p=p, margin=e, z=z)


def proportional_allocation(
    frame: pd.DataFrame,
    total: int,
    *,
    stratum_column: str = "stratum",
) -> dict[Hashable, int]:
    """Allocate ``total`` samples exactly with the largest-remainder method."""

    if stratum_column not in frame.columns:
        raise ValueError(f"Missing stratum column: {stratum_column}")
    if total < 0:
        raise ValueError("total must be non-negative")
    if total > len(frame):
        raise ValueError(
            f"Cannot sample {total} rows without replacement from {len(frame)} rows"
        )
    if frame[stratum_column].isna().any():
        raise ValueError(f"{stratum_column} contains missing values")

    sizes = frame.groupby(stratum_column, sort=False, dropna=False).size()
    if total == 0:
        return {name: 0 for name in sizes.index}

    raw = sizes.astype(float) * total / len(frame)
    allocation = {
        name: min(int(math.floor(raw[name])), int(sizes[name]))
        for name in sizes.index
    }
    remaining = total - sum(allocation.values())
    priority = sorted(
        sizes.index,
        key=lambda name: (-(raw[name] - math.floor(raw[name])), str(name)),
    )

    while remaining:
        progressed = False
        for name in priority:
            if allocation[name] < int(sizes[name]):
                allocation[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:  # Defensive: should be unreachable when total <= len(frame).
            raise RuntimeError("Could not allocate the requested sample size")
    return allocation


def allocate(frame: pd.DataFrame, n_tot: int) -> dict[Hashable, int]:
    """Compatibility alias for proportional allocation on ``stratum``."""

    return proportional_allocation(frame, n_tot, stratum_column="stratum")


def stratified_sample(
    frame: pd.DataFrame,
    total: int,
    seed: int,
    *,
    stratum_column: str = "stratum",
) -> tuple[pd.DataFrame, dict[Hashable, int]]:
    """Sample without replacement according to proportional allocation."""

    allocation = proportional_allocation(
        frame, total, stratum_column=stratum_column
    )
    if total == 0:
        return frame.iloc[0:0].copy(), allocation

    rng = random.Random(seed)
    parts: list[pd.DataFrame] = []
    for stratum, count in allocation.items():
        if count == 0:
            continue
        group = frame[frame[stratum_column] == stratum]
        parts.append(
            group.sample(
                n=count,
                replace=False,
                random_state=rng.randrange(0, 2**32),
            )
        )
    return pd.concat(parts, ignore_index=True), allocation


def balanced_sample(
    frame: pd.DataFrame,
    per_stratum: int,
    seed: int,
    *,
    stratum_column: str = "stratum",
    strict: bool = False,
) -> pd.DataFrame:
    """Take up to ``per_stratum`` rows from every stratum deterministically."""

    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    if stratum_column not in frame.columns:
        raise ValueError(f"Missing stratum column: {stratum_column}")
    if frame[stratum_column].isna().any():
        raise ValueError(f"{stratum_column} contains missing values")

    sizes = frame.groupby(stratum_column, sort=False).size()
    undersized = sizes[sizes < per_stratum]
    if strict and not undersized.empty:
        details = ", ".join(f"{name}={size}" for name, size in undersized.items())
        raise ValueError(
            f"Requested {per_stratum} rows per stratum, but found: {details}"
        )

    rng = random.Random(seed)
    parts = []
    for stratum in sizes.index:
        group = frame[frame[stratum_column] == stratum]
        parts.append(
            group.sample(
                n=min(per_stratum, len(group)),
                replace=False,
                random_state=rng.randrange(0, 2**32),
            )
        )
    return pd.concat(parts, ignore_index=True)


def exclude_rows(
    frame: pd.DataFrame,
    excluded: pd.DataFrame,
    *,
    key_column: str = "prompt",
) -> pd.DataFrame:
    """Remove rows whose key occurs in an already-tested table."""

    if key_column not in frame.columns:
        raise ValueError(f"Input table is missing exclusion key: {key_column}")
    if key_column not in excluded.columns:
        raise ValueError(f"Exclusion table is missing key: {key_column}")
    keys = set(excluded[key_column].dropna().tolist())
    return frame.loc[~frame[key_column].isin(keys)].reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically stratify a local safety-prompt table."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input table")
    parser.add_argument("--output", type=Path, required=True, help="Output table")
    parser.add_argument("--exclude", type=Path, help="Optional table of tested rows")
    parser.add_argument("--key-column", default="prompt", help="Exclusion key column")
    parser.add_argument("--stratum-column", default="stratum")
    parser.add_argument("--confidence", type=float, choices=sorted(CONFIDENCE_Z), default=0.85)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--prop", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, help="Exact sample size override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    frame = read_table(args.input)
    if args.exclude is not None:
        frame = exclude_rows(
            frame, read_table(args.exclude), key_column=args.key_column
        )
    if frame.empty:
        raise ValueError("No rows remain after exclusion")
    if args.stratum_column not in frame.columns:
        frame[args.stratum_column] = "default"

    if args.n is None:
        requested = fpc_sample_size(
            len(frame),
            p=args.prop,
            margin=args.margin,
            z=CONFIDENCE_Z[args.confidence],
        )
    else:
        requested = min(max(args.n, 0), len(frame))
    sampled, allocation = stratified_sample(
        frame,
        requested,
        args.seed,
        stratum_column=args.stratum_column,
    )
    write_table(sampled, args.output)
    print(f"Saved {len(sampled)} rows to {args.output}")
    print("Allocation: " + ", ".join(f"{key}={value}" for key, value in allocation.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIDENCE_Z",
    "CONF_Z",
    "allocate",
    "balanced_sample",
    "exclude_rows",
    "fpc_n",
    "fpc_sample_size",
    "proportional_allocation",
    "stratified_sample",
]
