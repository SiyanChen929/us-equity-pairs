"""Fail-closed contract for point-in-time equity-universe research data.

The existing Yahoo/current-Wikipedia snapshots deliberately do not satisfy this
contract.  A passing dataset must use permanent security identifiers and retain
historical membership, historical GICS, delistings, and corporate actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    config_key: str
    required_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    key_columns: tuple[str, ...] = ()
    interval_columns: tuple[str, str] | None = None


CONTRACTS: tuple[DatasetContract, ...] = (
    DatasetContract(
        "security_master",
        (
            "security_id",
            "issuer_id",
            "ticker",
            "name_valid_from",
            "name_valid_to_exclusive",
            "security_type",
            "share_class",
            "primary_exchange",
        ),
        ("name_valid_from", "name_valid_to_exclusive"),
        ("security_id", "name_valid_from"),
        ("name_valid_from", "name_valid_to_exclusive"),
    ),
    DatasetContract(
        "sp500_membership_history",
        (
            "security_id",
            "index_code",
            "effective_from",
            "effective_to_exclusive",
            "source_release_date",
        ),
        ("effective_from", "effective_to_exclusive", "source_release_date"),
        ("security_id", "index_code", "effective_from"),
        ("effective_from", "effective_to_exclusive"),
    ),
    DatasetContract(
        "gics_history",
        (
            "issuer_id",
            "effective_from",
            "effective_to_exclusive",
            "gics_sector_code",
            "gics_sector_name",
            "gics_subindustry_code",
            "gics_subindustry_name",
            "source_release_date",
        ),
        ("effective_from", "effective_to_exclusive", "source_release_date"),
        ("issuer_id", "effective_from"),
        ("effective_from", "effective_to_exclusive"),
    ),
    DatasetContract(
        "daily_security_returns",
        (
            "security_id",
            "date",
            "total_return_including_distributions",
            "close",
            "volume",
            "delisting_return_applied",
            "return_source",
        ),
        ("date",),
        ("security_id", "date"),
    ),
    DatasetContract(
        "delisting_events",
        (
            "security_id",
            "delist_date",
            "delisting_return",
            "delist_reason",
            "payment_type",
            "successor_security_id",
        ),
        ("delist_date",),
        ("security_id", "delist_date"),
    ),
    DatasetContract(
        "corporate_actions",
        (
            "security_id",
            "effective_date",
            "action_type",
            "price_adjustment_factor",
            "cash_amount",
            "successor_security_id",
        ),
        ("effective_date",),
        ("security_id", "effective_date", "action_type"),
    ),
    DatasetContract(
        "pair_first_knowable_dates",
        (
            "pair_id",
            "security_id_a",
            "security_id_b",
            "first_knowable_date",
            "selection_rule",
            "selection_frozen_at",
            "provenance",
        ),
        ("first_knowable_date", "selection_frozen_at"),
        ("pair_id",),
    ),
)


def _resolve(workspace: Path, root: Path, filename: str) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _interval_overlap_count(
    frame: pd.DataFrame,
    entity_columns: list[str],
    start_column: str,
    end_column: str,
) -> int:
    overlaps = 0
    for _, group in frame.sort_values(entity_columns + [start_column]).groupby(
        entity_columns, dropna=False, sort=False
    ):
        starts = group[start_column].to_numpy()
        ends = group[end_column].to_numpy()
        if len(group) > 1:
            overlaps += int((starts[1:] < ends[:-1]).sum())
    return overlaps


def audit_point_in_time_contract(
    workspace: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Validate all point-in-time files and return a machine-readable audit."""

    workspace_path = Path(workspace).resolve()
    settings = config["point_in_time_data"]
    data_root = _resolve(workspace_path, workspace_path, settings["root"])
    coverage_start = pd.Timestamp(settings["coverage_start_inclusive"])
    coverage_end = pd.Timestamp(settings["coverage_end_exclusive"])
    datasets: dict[str, pd.DataFrame] = {}
    file_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for contract in CONTRACTS:
        path = _resolve(workspace_path, data_root, settings[contract.config_key])
        result: dict[str, Any] = {
            "dataset": contract.config_key,
            "path": str(path),
            "exists": path.is_file(),
            "rows": 0,
            "required_columns_present": False,
            "duplicate_key_rows": None,
            "invalid_date_rows": None,
            "invalid_interval_rows": None,
            "overlapping_interval_rows": None,
            "passed": False,
        }
        if not path.is_file():
            errors.append(f"missing required point-in-time dataset: {path}")
            file_results.append(result)
            continue
        try:
            frame = _read(path)
        except Exception as exc:  # fail closed with the concrete source error
            errors.append(f"could not read {path}: {type(exc).__name__}: {exc}")
            file_results.append(result)
            continue
        result["rows"] = int(len(frame))
        missing = sorted(set(contract.required_columns).difference(frame.columns))
        result["missing_columns"] = missing
        result["required_columns_present"] = not missing
        if missing:
            errors.append(f"{contract.config_key} missing columns: {', '.join(missing)}")
            file_results.append(result)
            continue
        invalid_dates = pd.Series(False, index=frame.index)
        for column in contract.date_columns:
            original_not_null = frame[column].notna()
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            invalid_dates |= original_not_null & frame[column].isna()
        result["invalid_date_rows"] = int(invalid_dates.sum())
        duplicate_rows = (
            int(frame.duplicated(list(contract.key_columns), keep=False).sum())
            if contract.key_columns
            else 0
        )
        result["duplicate_key_rows"] = duplicate_rows
        invalid_intervals = 0
        overlapping_intervals = 0
        if contract.interval_columns is not None:
            start_column, end_column = contract.interval_columns
            invalid_intervals = int(
                (
                    frame[start_column].isna()
                    | frame[end_column].isna()
                    | (frame[start_column] >= frame[end_column])
                ).sum()
            )
            entity_columns = [
                column
                for column in contract.key_columns
                if column != start_column
            ]
            if entity_columns and invalid_intervals == 0:
                overlapping_intervals = _interval_overlap_count(
                    frame,
                    entity_columns,
                    start_column,
                    end_column,
                )
        result["invalid_interval_rows"] = invalid_intervals
        result["overlapping_interval_rows"] = overlapping_intervals
        if invalid_dates.any():
            errors.append(f"{contract.config_key} has {int(invalid_dates.sum())} invalid dates")
        if duplicate_rows:
            errors.append(f"{contract.config_key} has {duplicate_rows} duplicate key rows")
        if invalid_intervals:
            errors.append(f"{contract.config_key} has {invalid_intervals} invalid intervals")
        if overlapping_intervals:
            errors.append(f"{contract.config_key} has {overlapping_intervals} overlapping intervals")
        result["passed"] = not any(
            (
                missing,
                bool(invalid_dates.any()),
                duplicate_rows,
                invalid_intervals,
                overlapping_intervals,
                frame.empty,
            )
        )
        if frame.empty:
            errors.append(f"{contract.config_key} is empty")
        datasets[contract.config_key] = frame
        file_results.append(result)

    if all(name in datasets for name in ("security_master", "sp500_membership_history", "daily_security_returns")):
        security_ids = set(datasets["security_master"]["security_id"].dropna().astype(str))
        for name in (
            "sp500_membership_history",
            "daily_security_returns",
            "delisting_events",
            "corporate_actions",
        ):
            if name not in datasets:
                continue
            observed = set(datasets[name]["security_id"].dropna().astype(str))
            missing_ids = observed.difference(security_ids)
            if missing_ids:
                errors.append(f"{name} references {len(missing_ids)} unknown security_id values")
        if "pair_first_knowable_dates" in datasets:
            pair_dates = datasets["pair_first_knowable_dates"]
            for column in ("security_id_a", "security_id_b"):
                observed = set(pair_dates[column].dropna().astype(str))
                missing_ids = observed.difference(security_ids)
                if missing_ids:
                    errors.append(
                        f"pair_first_knowable_dates {column} references "
                        f"{len(missing_ids)} unknown security_id values"
                    )
            timing_invalid = int(
                (
                    pair_dates["selection_frozen_at"]
                    > pair_dates["first_knowable_date"]
                ).sum()
            )
            if timing_invalid:
                errors.append(
                    "pair_first_knowable_dates has "
                    f"{timing_invalid} rows frozen after first_knowable_date"
                )
        if "gics_history" in datasets:
            issuer_ids = set(
                datasets["security_master"]["issuer_id"].dropna().astype(str)
            )
            gics_issuers = set(
                datasets["gics_history"]["issuer_id"].dropna().astype(str)
            )
            missing_issuers = gics_issuers.difference(issuer_ids)
            if missing_issuers:
                errors.append(
                    "gics_history references "
                    f"{len(missing_issuers)} unknown issuer_id values"
                )
        membership = datasets["sp500_membership_history"]
        if membership["effective_from"].min() > coverage_start:
            errors.append("membership history starts after required coverage start")
        if membership["effective_to_exclusive"].max() < coverage_end:
            errors.append("membership history ends before required coverage end")
        returns = datasets["daily_security_returns"]
        if returns["date"].min() > coverage_start:
            errors.append("daily returns start after required coverage start")
        if returns["date"].max() < coverage_end - pd.Timedelta(days=1):
            errors.append("daily returns end before required coverage end")

    audit = {
        "contract_version": 1,
        "coverage_start_inclusive": str(coverage_start.date()),
        "coverage_end_exclusive": str(coverage_end.date()),
        "permanent_identifier_required": settings["permanent_security_identifier"],
        "files": file_results,
        "errors": errors,
        "critical_pass": not errors and len(datasets) == len(CONTRACTS),
        "current_yahoo_wikipedia_snapshot_is_eligible": False,
    }
    return audit, datasets


__all__ = ["CONTRACTS", "DatasetContract", "audit_point_in_time_contract"]
