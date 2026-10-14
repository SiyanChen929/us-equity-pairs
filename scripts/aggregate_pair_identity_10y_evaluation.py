#!/usr/bin/env python3
"""Aggregate the additive V6 fixed-identity replay into full-calendar paths.

This script contains evaluation logic only.  It never screens, tunes, selects,
or simulates a trade.  Failed and unavailable pair-quarters are retained as
cash, and all headline statistics use the same exact study calendar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity_pairs.data import load_timeseries_snapshot
from equity_pairs.identity_evaluation import (
    anniversary_block_labels,
    bootstrap_cagr_from_log_blocks,
    canonical_leg_weight,
    exact_calendar_years,
    fixed_universe_rankings,
    path_statistics,
    remove_block_as_cash,
    stability_statistics,
)


DEFAULT_CONFIG = Path("configs/pair_identity_10y_evaluation.json")


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _as_bool(values: pd.Series, *, default: bool = False) -> pd.Series:
    if values is None:
        raise TypeError("values must be a Series")
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    result = normalized.isin({"true", "1", "yes", "y"})
    if default:
        result |= normalized.isna()
    return result.fillna(default).astype(bool)


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing: " + ", ".join(missing))


def _load_sparse_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require(frame, {"pair", "date"}, "replay daily")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", format="mixed")
    if {"transaction_cost_bps", "net_return"}.issubset(frame.columns):
        identifiers = ["pair", "date"]
        duplicates = frame.duplicated(identifiers + ["transaction_cost_bps"])
        if duplicates.any():
            raise ValueError("replay daily has duplicate pair/date/cost rows")
        wide = frame.pivot(
            index=identifiers,
            columns="transaction_cost_bps",
            values="net_return",
        ).reset_index()
        rename: dict[Any, str] = {}
        for column in wide.columns:
            try:
                cost = float(column)
            except (TypeError, ValueError):
                continue
            if math.isclose(cost, 1.0):
                rename[column] = "net_return_1bp"
            elif math.isclose(cost, 2.0):
                rename[column] = "net_return_2bp"
        frame = wide.rename(columns=rename)
    aliases = {
        "net_return_1_bps": "net_return_1bp",
        "net_return_2_bps": "net_return_2bp",
        "net_return_1bps": "net_return_1bp",
        "net_return_2bps": "net_return_2bp",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    _require(
        frame,
        {"pair", "date", "net_return_1bp", "net_return_2bp"},
        "replay daily",
    )
    for column in ("net_return_1bp", "net_return_2bp"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"replay daily contains a missing/non-finite {column}")
        if numeric.le(-1.0).any():
            raise ValueError(f"replay daily contains {column} at or below -100%")
        frame[column] = numeric.astype(float)
    if frame.duplicated(["pair", "date"]).any():
        raise ValueError("replay daily has duplicate pair/date rows")
    return frame


def _normalise_quarter_panel(frame: pd.DataFrame) -> pd.DataFrame:
    _require(
        frame,
        {"pair", "rebalance_date", "outer_end_exclusive"},
        "quarter panel",
    )
    result = frame.copy()
    result["rebalance_date"] = pd.to_datetime(
        result["rebalance_date"], errors="raise", format="mixed"
    )
    if "outer_end_exclusive" in result:
        result["outer_end_exclusive"] = pd.to_datetime(
            result["outer_end_exclusive"], errors="raise", format="mixed"
        )
    status_column = next(
        (column for column in ("evaluation_status", "status") if column in result),
        None,
    )
    if status_column is None:
        raise ValueError("quarter panel is missing status/evaluation_status")
    status = result[status_column].astype("string").str.strip()
    allowed_statuses = {
        "unavailable_cash",
        "tuning_failed_cash",
        "evaluation_failed_cash",
        "evaluated_zero_trade",
        "evaluated_traded",
    }
    unknown = sorted(set(status.dropna().astype(str)) - allowed_statuses)
    if unknown or status.isna().any():
        raise ValueError(
            "quarter panel has unknown evaluation statuses: "
            + ", ".join(unknown or ["<missing>"])
        )
    result["evaluation_status"] = status
    if "available" in result:
        result["available"] = _as_bool(result["available"])
    elif {"history_available", "orientation_available"}.issubset(result.columns):
        result["available"] = _as_bool(result["history_available"]) & _as_bool(
            result["orientation_available"]
        )
    else:
        result["available"] = status.ne("unavailable_cash")
    if "evaluation_success" in result:
        result["evaluation_success"] = _as_bool(result["evaluation_success"])
    else:
        result["evaluation_success"] = status.isin(
            {"evaluated_zero_trade", "evaluated_traded"}
        )
    expected_success = status.isin({"evaluated_zero_trade", "evaluated_traded"})
    if not result["evaluation_success"].equals(expected_success):
        raise ValueError("evaluation_success conflicts with the status taxonomy")
    eligibility_column = next(
        (
            column
            for column in (
                "original_v6_eligible",
                "screen_strict_eligible",
                "strict_eligible",
                "eligible",
            )
            if column in result
        ),
        None,
    )
    result["original_v6_eligible"] = (
        _as_bool(result[eligibility_column])
        if eligibility_column
        else False
    )
    beta_column = next(
        (
            column
            for column in (
                "evaluation_beta",
                "oos_refit_beta",
                "fitted_beta",
                "beta",
            )
            if column in result
        ),
        None,
    )
    result["evaluation_beta"] = (
        pd.to_numeric(result[beta_column], errors="coerce")
        if beta_column is not None
        else math.nan
    )
    dependent_column = next(
        (
            column
            for column in ("dependent", "screen_dependent")
            if column in result
        ),
        None,
    )
    result["dependent"] = (
        result[dependent_column].astype("string")
        if dependent_column is not None
        else pd.Series(pd.NA, index=result.index, dtype="string")
    )
    result["trade_count"] = pd.to_numeric(
        result.get("trade_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    if result["trade_count"].lt(0).any():
        raise ValueError("quarter panel contains a negative trade count")
    if (
        status.eq("evaluated_zero_trade") & result["trade_count"].ne(0)
    ).any() or (
        status.eq("evaluated_traded") & result["trade_count"].le(0)
    ).any():
        raise ValueError("evaluated status conflicts with quarter trade_count")
    result["evaluated_zero_trade"] = (
        result["evaluation_success"] & result["trade_count"].eq(0)
    )
    result["actively_traded"] = (
        result["evaluation_success"] & result["trade_count"].gt(0)
    )
    result["tuning_failed_cash"] = status.eq("tuning_failed_cash")
    result["evaluation_failed_cash"] = status.eq("evaluation_failed_cash")
    result["unavailable_cash"] = status.eq("unavailable_cash")
    if not result.loc[result["unavailable_cash"], "available"].eq(False).all():
        raise ValueError("unavailable_cash rows cannot be marked available")
    return result


def _schedule_for_dates(
    dates: pd.DatetimeIndex,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    _require(
        schedule,
        {"rebalance_date", "outer_end_exclusive"},
        "rebalance schedule",
    )
    result = schedule.copy()
    result["rebalance_date"] = pd.to_datetime(result["rebalance_date"], format="mixed")
    result["outer_end_exclusive"] = pd.to_datetime(
        result["outer_end_exclusive"], format="mixed"
    )
    result = result.sort_values("rebalance_date", kind="mergesort").reset_index(drop=True)
    if "quarter_id" not in result:
        if "quarter_number" in result:
            result.insert(
                0,
                "quarter_id",
                pd.to_numeric(result["quarter_number"], errors="raise").astype(int),
            )
        else:
            result.insert(0, "quarter_id", np.arange(1, len(result) + 1))
    if result["quarter_id"].duplicated().any():
        raise ValueError("rebalance schedule contains duplicate quarter identifiers")
    if result["rebalance_date"].duplicated().any():
        raise ValueError("rebalance schedule contains duplicate rebalance dates")
    if (result["outer_end_exclusive"] <= result["rebalance_date"]).any():
        raise ValueError("rebalance schedule contains a non-positive outer window")
    if len(result) > 1 and not all(
        result.iloc[index - 1]["outer_end_exclusive"]
        == result.iloc[index]["rebalance_date"]
        for index in range(1, len(result))
    ):
        raise ValueError("rebalance schedule outer windows are not adjacent")
    labels = np.full(len(dates), -1, dtype=int)
    for row in result.itertuples(index=False):
        mask = (dates >= row.rebalance_date) & (dates < row.outer_end_exclusive)
        if np.any(labels[mask] >= 0):
            raise ValueError("rebalance schedule contains overlapping outer windows")
        if not mask.any():
            raise ValueError(f"quarter {row.quarter_id} contains no study sessions")
        if hasattr(row, "outer_sessions") and int(row.outer_sessions) != int(mask.sum()):
            raise ValueError(
                f"quarter {row.quarter_id} outer-session count does not reconcile"
            )
        labels[mask] = int(row.quarter_id)
    if np.any(labels < 0):
        missing = dates[labels < 0]
        raise ValueError(
            f"schedule does not cover {len(missing)} study sessions; first={missing[0]}"
        )
    return result


def _calendar_year_is_complete(
    year: int,
    study_start: pd.Timestamp,
    study_end_exclusive: pd.Timestamp,
) -> bool:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return bool(start >= study_start and end <= study_end_exclusive)


def _status_reason_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "evaluation_status" not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame["evaluation_status"]
        .astype("string")
        .fillna("missing_status")
        .value_counts()
        .sort_index()
        .items()
    }


def _cash_reason_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "cash_reason" not in frame:
        return {}
    values = frame["cash_reason"].astype("string").fillna("").str.strip()
    values = values.mask(values.eq(""), "not_cash")
    return {
        str(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def _maximum_true_streak(values: pd.Series) -> int:
    flags = _as_bool(values).to_numpy(dtype=bool)
    maximum = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        maximum = max(maximum, current)
    return int(maximum)


def aggregate(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    extension = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = extension["data"]
    root = Path(
        output_override or extension["outputs"]["default_root"]
    ).resolve()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    evaluation = extension["evaluation"]
    study_start = pd.Timestamp(
        evaluation.get(
            "study_start_inclusive", data_config.get("study_start_inclusive")
        )
    )
    study_end_exclusive = pd.Timestamp(
        evaluation.get(
            "study_end_exclusive",
            data_config.get("end_exclusive", data_config.get("download_end_exclusive")),
        )
    )
    calendar_years = exact_calendar_years(study_start, study_end_exclusive)

    master_path = processed / "pair_identity_master.csv"
    schedule_path = processed / "pair_identity_rebalance_schedule.csv"
    quarter_path = processed / "pair_identity_quarterly_results.csv.gz"
    sparse_daily_path = processed / "pair_identity_replay_daily.csv.gz"
    master = pd.read_csv(master_path)
    schedule = pd.read_csv(schedule_path)
    quarterly = _normalise_quarter_panel(pd.read_csv(quarter_path))
    sparse = _load_sparse_daily(sparse_daily_path)
    _require(master, {"pair", "sector", "ticker_a", "ticker_b"}, "identity master")
    if master["pair"].duplicated().any() or len(master) != 114:
        raise ValueError("the fixed identity master must contain exactly 114 unique pairs")
    if "original_v6_selected_identity" not in master:
        selected_alias = next(
            (
                column
                for column in (
                    "ever_v6_selected",
                    "original_selected_subset",
                    "in_original_v6_selected_82",
                )
                if column in master
            ),
            None,
        )
        master["original_v6_selected_identity"] = (
            _as_bool(master[selected_alias]) if selected_alias else False
        )
    else:
        master["original_v6_selected_identity"] = _as_bool(
            master["original_v6_selected_identity"]
        )
    if int(master["original_v6_selected_identity"].sum()) != 82:
        raise ValueError("the original V6 selected-identity subset must contain 82 pairs")

    price_path = root / "data" / "raw" / str(data_config["spliced_snapshot_name"])
    if not price_path.is_file():
        raise FileNotFoundError(
            "the prepared, V6-spliced price snapshot is missing: " + str(price_path)
        )
    frozen_prices, _ = load_timeseries_snapshot(price_path)
    dates = pd.DatetimeIndex(
        frozen_prices.index[
            (frozen_prices.index >= study_start)
            & (frozen_prices.index < study_end_exclusive)
        ],
        name="date",
    )
    if dates.empty or dates[0] != study_start or dates[-1] != study_end_exclusive - pd.Timedelta(days=1):
        raise ValueError("frozen V6 study sessions do not match the configured boundaries")
    schedule = _schedule_for_dates(dates, schedule)
    expected_quarters = int(
        evaluation.get(
            "expected_quarter_intervals",
            evaluation.get("expected_clipped_quarter_count"),
        )
    )
    if len(schedule) != expected_quarters:
        raise ValueError(
            f"expected {expected_quarters} clipped quarter intervals, found {len(schedule)}"
        )

    expected_panel_rows = len(master) * len(schedule)
    if len(quarterly) != expected_panel_rows:
        raise ValueError(
            f"quarter panel must have {expected_panel_rows} rows; found {len(quarterly)}"
        )
    if quarterly.duplicated(["pair", "rebalance_date"]).any():
        raise ValueError("quarter panel has duplicate pair-quarter rows")
    expected_keys = pd.MultiIndex.from_product(
        [master["pair"].astype(str), schedule["rebalance_date"]],
        names=["pair", "rebalance_date"],
    )
    actual_keys = pd.MultiIndex.from_frame(quarterly[["pair", "rebalance_date"]])
    if len(expected_keys.difference(actual_keys)) or len(actual_keys.difference(expected_keys)):
        raise ValueError("quarter panel is not the complete 114-by-quarter Cartesian product")

    # The canonical quarterly file is enriched in place.  Drop previously
    # derived aggregation columns so reruns are idempotent.
    quarterly = quarterly.drop(
        columns=[
            "quarter_id",
            "quarter_return_1bp",
            "quarter_return_2bp",
            "outer_sessions_schedule",
        ],
        errors="ignore",
    )
    quarterly = quarterly.merge(
        schedule[
            ["quarter_id", "rebalance_date", "outer_end_exclusive", "outer_sessions"]
        ],
        on=["rebalance_date", "outer_end_exclusive"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_schedule"),
    )
    if quarterly["quarter_id"].isna().any():
        raise ValueError("quarter panel contains a window absent from the schedule")
    quarterly["quarter_id"] = quarterly["quarter_id"].astype(int)

    unknown_daily = sorted(set(sparse["pair"].astype(str)) - set(master["pair"].astype(str)))
    if unknown_daily:
        raise ValueError(f"replay daily contains unknown pairs: {unknown_daily[:3]}")
    if not sparse["date"].between(dates[0], dates[-1]).all():
        raise ValueError("replay daily contains dates outside the exact study path")

    quarter_by_date = pd.Series(index=dates, dtype=int)
    for row in schedule.itertuples(index=False):
        mask = (dates >= row.rebalance_date) & (dates < row.outer_end_exclusive)
        quarter_by_date.loc[mask] = int(row.quarter_id)
    quarter_by_date = quarter_by_date.astype(int)

    sparse["quarter_id"] = sparse["date"].map(quarter_by_date)
    if sparse["quarter_id"].isna().any():
        raise ValueError("replay daily contains dates that do not map to a quarter")
    sparse["quarter_id"] = sparse["quarter_id"].astype(int)
    expected_sparse = quarterly.loc[
        quarterly["evaluation_success"],
        ["pair", "quarter_id", "outer_sessions"],
    ].copy()
    expected_sparse["expected_daily_rows"] = pd.to_numeric(
        expected_sparse["outer_sessions"], errors="raise"
    ).astype(int)
    actual_sparse = (
        sparse.groupby(["pair", "quarter_id"], sort=False)
        .size()
        .rename("actual_daily_rows")
        .reset_index()
    )
    sparse_reconciliation = expected_sparse.merge(
        actual_sparse,
        on=["pair", "quarter_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    sparse_reconciliation["expected_daily_rows"] = pd.to_numeric(
        sparse_reconciliation["expected_daily_rows"], errors="coerce"
    ).fillna(0).astype(int)
    sparse_reconciliation["actual_daily_rows"] = pd.to_numeric(
        sparse_reconciliation["actual_daily_rows"], errors="coerce"
    ).fillna(0).astype(int)
    if not sparse_reconciliation["expected_daily_rows"].eq(
        sparse_reconciliation["actual_daily_rows"]
    ).all():
        mismatch = sparse_reconciliation.loc[
            sparse_reconciliation["expected_daily_rows"].ne(
                sparse_reconciliation["actual_daily_rows"]
            )
        ].iloc[0]
        raise ValueError(
            "successful pair-quarter daily coverage does not reconcile: "
            f"{mismatch['pair']} quarter={mismatch['quarter_id']} "
            f"expected={mismatch['expected_daily_rows']} "
            f"actual={mismatch['actual_daily_rows']}"
        )

    _log("building the dense 114-pair cash-inclusive daily panel")
    dense_parts: list[pd.DataFrame] = []
    pair_order = master["pair"].astype(str).tolist()
    sparse = sparse[["pair", "date", "net_return_1bp", "net_return_2bp"]].copy()
    for pair in pair_order:
        pair_daily = pd.DataFrame(
            {
                "pair": pair,
                "date": dates,
                "quarter_id": quarter_by_date.to_numpy(),
            }
        ).merge(
            sparse.loc[sparse["pair"].astype(str).eq(pair)],
            on=["pair", "date"],
            how="left",
            validate="one_to_one",
        )
        for cost in (1, 2):
            return_column = f"net_return_{cost}bp"
            values = pd.to_numeric(pair_daily[return_column], errors="coerce").fillna(0.0)
            if values.le(-1.0).any():
                raise ValueError(f"{pair} has a daily return at or below -100%")
            pair_daily[return_column] = values
            equity = (1.0 + values).cumprod()
            # Match the immutable V6/path_statistics drawdown convention.
            running = equity.cummax()
            pair_daily[f"equity_{cost}bp"] = equity
            pair_daily[f"drawdown_{cost}bp"] = equity / running - 1.0
        dense_parts.append(pair_daily)
    dense = pd.concat(dense_parts, ignore_index=True)

    quarter_returns = (
        dense.groupby(["pair", "quarter_id"], sort=False)[
            ["net_return_1bp", "net_return_2bp"]
        ]
        .agg(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
        .reset_index()
        .rename(
            columns={
                "net_return_1bp": "quarter_return_1bp",
                "net_return_2bp": "quarter_return_2bp",
            }
        )
    )
    quarterly = quarterly.merge(
        quarter_returns,
        on=["pair", "quarter_id"],
        how="left",
        validate="one_to_one",
    )
    if quarterly[["quarter_return_1bp", "quarter_return_2bp"]].isna().any().any():
        raise RuntimeError("quarter return aggregation left missing values")
    for cost in (1, 2):
        source_column = f"q_return_{cost}bp"
        computed_column = f"quarter_return_{cost}bp"
        if source_column not in quarterly:
            raise ValueError(f"quarter panel is missing {source_column}")
        source_values = pd.to_numeric(quarterly[source_column], errors="coerce")
        computed_values = pd.to_numeric(quarterly[computed_column], errors="coerce")
        if source_values.isna().any() or not np.allclose(
            source_values.to_numpy(dtype=float),
            computed_values.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"dense daily returns do not reconcile to {source_column}"
            )

    master_by_pair = master.set_index("pair", drop=False)
    quarter_by_pair = {
        pair: frame.sort_values("rebalance_date", kind="mergesort").copy()
        for pair, frame in quarterly.groupby("pair", sort=False)
    }
    daily_by_pair = {
        pair: frame.sort_values("date", kind="mergesort").copy()
        for pair, frame in dense.groupby("pair", sort=False)
    }

    _log("calculating full-calendar statistics, robustness, and stability")
    stability_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    common_by_pair: dict[str, dict[str, Any]] = {}
    for pair in pair_order:
        identity = master_by_pair.loc[pair]
        qframe = quarter_by_pair[pair]
        success = qframe.loc[qframe["evaluation_success"]].copy()
        beta_column = next(
            (
                column
                for column in (
                    "oos_refit_beta",
                    "evaluation_beta",
                    "fitted_beta",
                    "beta",
                )
                if column in success
            ),
            None,
        )
        dependent_column = next(
            (
                column
                for column in ("screen_dependent", "dependent")
                if column in success
            ),
            None,
        )
        if beta_column is None:
            success["canonical_weight_a"] = math.nan
        else:
            success["canonical_weight_a"] = [
                canonical_leg_weight(beta, str(dependent), str(identity["ticker_a"]))
                for beta, dependent in zip(
                    pd.to_numeric(success[beta_column], errors="coerce"),
                    success[dependent_column]
                    if dependent_column is not None
                    else pd.Series("", index=success.index),
                    strict=False,
                )
            ]
        if dependent_column is not None:
            success["dependent"] = success[dependent_column]
        for column in (
            "dependent",
            "selected_config_id",
            "zscore_method",
            "zscore_lookback",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
        ):
            if column not in success:
                success[column] = pd.NA
        stability = stability_statistics(success)
        for component in (
            "zscore_method",
            "zscore_lookback",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
        ):
            stability.setdefault(f"{component}_switch_rate", math.nan)
        stability_row = {
            "pair": pair,
            "sector": identity["sector"],
            "ticker_a": identity["ticker_a"],
            "ticker_b": identity["ticker_b"],
            **stability,
        }
        stability_rows.append(stability_row)

        counts = _status_reason_counts(qframe)
        cash_reason_counts = _cash_reason_counts(qframe)
        unavailable = int(qframe["unavailable_cash"].sum())
        available = int(qframe["available"].sum())
        evaluated = int(qframe["evaluation_success"].sum())
        eligibility_passes = int(qframe["original_v6_eligible"].sum())
        core_health_passes = int(
            _as_bool(
                qframe.get(
                    "screen_core_health_pass",
                    pd.Series(False, index=qframe.index),
                )
            ).sum()
        )
        fdr_passes = int(
            _as_bool(
                qframe.get(
                    "screen_fdr_significant",
                    pd.Series(False, index=qframe.index),
                )
            ).sum()
        )
        i1_passes = int(
            _as_bool(
                qframe.get(
                    "screen_both_look_i1",
                    pd.Series(False, index=qframe.index),
                )
            ).sum()
        )
        positive_beta_passes = int(
            _as_bool(
                qframe.get(
                    "screen_positive_hedge_ratio",
                    pd.Series(False, index=qframe.index),
                )
            ).sum()
        )
        half_life = pd.to_numeric(
            qframe.get(
                "screen_half_life_days", pd.Series(np.nan, index=qframe.index)
            ),
            errors="coerce",
        )
        hurst = pd.to_numeric(
            qframe.get(
                "screen_hurst_exponent", pd.Series(np.nan, index=qframe.index)
            ),
            errors="coerce",
        )
        half_life_passes = int(half_life.between(2.0, 252.0).sum())
        hurst_passes = int(hurst.lt(0.50).sum())
        tuning_activity_passes = int(
            _as_bool(
                qframe.get("tuning_success", qframe["evaluation_success"])
            ).sum()
        )
        tuning_failed = int(qframe["tuning_failed_cash"].sum())
        evaluation_failed = int(qframe["evaluation_failed_cash"].sum())
        partition_total = unavailable + tuning_failed + evaluation_failed + evaluated
        if partition_total != len(qframe):
            raise RuntimeError(
                f"{pair} status taxonomy does not partition all scheduled quarters"
            )
        if int(qframe["evaluated_zero_trade"].sum()) + int(
            qframe["actively_traded"].sum()
        ) != evaluated:
            raise RuntimeError(
                f"{pair} evaluated zero/traded states do not reconcile"
            )
        history_ratio = pd.to_numeric(
            qframe.get("screen_history_ratio", pd.Series(np.nan, index=qframe.index)),
            errors="coerce",
        )
        common = {
            "pair": pair,
            "sector": identity["sector"],
            "ticker_a": identity["ticker_a"],
            "ticker_b": identity["ticker_b"],
            "original_v6_selected_identity": bool(identity["original_v6_selected_identity"]),
            "scheduled_quarters": int(len(qframe)),
            "available_quarters": available,
            "unavailable_quarters": unavailable,
            "evaluated_quarters": evaluated,
            "tuning_failed_cash_quarters": tuning_failed,
            "evaluation_failed_cash_quarters": evaluation_failed,
            "other_failed_cash_quarters": evaluation_failed,
            "zero_trade_quarters": int(qframe["evaluated_zero_trade"].sum()),
            "actively_traded_quarters": int(qframe["actively_traded"].sum()),
            "trade_count": int(qframe["trade_count"].sum()),
            "availability_fraction": float(available / len(qframe)),
            "evaluation_success_fraction": float(evaluated / len(qframe)),
            "eligibility_pass_quarters": eligibility_passes,
            "eligibility_pass_fraction": float(eligibility_passes / len(qframe)),
            "eligibility_pass_fraction_when_available": float(eligibility_passes / available)
            if available
            else math.nan,
            "pair_health_pass_quarters": core_health_passes,
            "pair_health_pass_frequency": float(core_health_passes / len(qframe)),
            "pair_health_pass_frequency_when_available": float(
                core_health_passes / available
            )
            if available
            else math.nan,
            "fdr_pass_frequency": float(fdr_passes / len(qframe)),
            "both_i1_pass_frequency": float(i1_passes / len(qframe)),
            "positive_hedge_ratio_pass_frequency": float(
                positive_beta_passes / len(qframe)
            ),
            "half_life_pass_frequency": float(half_life_passes / len(qframe)),
            "hurst_pass_frequency": float(hurst_passes / len(qframe)),
            "tuning_activity_pass_quarters": tuning_activity_passes,
            "tuning_activity_pass_frequency": float(
                tuning_activity_passes / len(qframe)
            ),
            "tuning_activity_pass_frequency_when_available": float(
                tuning_activity_passes / available
            )
            if available
            else math.nan,
            "first_available_quarter": qframe.loc[qframe["available"], "rebalance_date"].min()
            if available
            else pd.NaT,
            "last_available_quarter": qframe.loc[qframe["available"], "rebalance_date"].max()
            if available
            else pd.NaT,
            "first_evaluated_quarter": qframe.loc[
                qframe["evaluation_success"], "rebalance_date"
            ].min()
            if evaluated
            else pd.NaT,
            "last_evaluated_quarter": qframe.loc[
                qframe["evaluation_success"], "rebalance_date"
            ].max()
            if evaluated
            else pd.NaT,
            "maximum_consecutive_unavailable_quarters": _maximum_true_streak(
                qframe["unavailable_cash"]
            ),
            "insufficient_history_quarters": int(
                qframe.get("cash_reason", pd.Series("", index=qframe.index))
                .astype("string")
                .eq("insufficient_three_year_history")
                .sum()
            ),
            "orientation_unavailable_quarters": int(
                qframe.get("cash_reason", pd.Series("", index=qframe.index))
                .astype("string")
                .eq("screen_orientation_unavailable")
                .sum()
            ),
            "screen_diagnostic_unavailable_quarters": int(
                qframe.get("cash_reason", pd.Series("", index=qframe.index))
                .astype("string")
                .eq("screen_diagnostic_unavailable")
                .sum()
            ),
            "minimum_screen_history_ratio": float(history_ratio.min())
            if history_ratio.notna().any()
            else math.nan,
            "median_screen_history_ratio": float(history_ratio.median())
            if history_ratio.notna().any()
            else math.nan,
            "evaluation_status_counts_json": json.dumps(counts, sort_keys=True),
            "cash_reason_counts_json": json.dumps(
                cash_reason_counts, sort_keys=True
            ),
            **stability,
        }
        common_by_pair[pair] = common
        coverage_rows.append(common.copy())

    annual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    block_labels = anniversary_block_labels(
        dates,
        study_start,
        block_count=10,
        end_exclusive=study_end_exclusive,
    )
    bootstrap_rates: dict[int, list[np.ndarray]] = {1: [], 2: []}
    metric_locations: dict[int, list[int]] = {1: [], 2: []}

    for pair in pair_order:
        qframe = quarter_by_pair[pair]
        dframe = daily_by_pair[pair]
        common = common_by_pair[pair]
        for cost in (1, 2):
            return_column = f"net_return_{cost}bp"
            quarter_column = f"quarter_return_{cost}bp"
            returns = dframe[return_column].to_numpy(dtype=float)
            stats = path_statistics(returns, calendar_years=calendar_years)
            quarter_values = qframe[quarter_column].to_numpy(dtype=float)
            positive_quarter_fraction = float(np.mean(quarter_values > 0.0))
            best_quarter_position = int(np.argmax(quarter_values))
            worst_quarter_position = int(np.argmin(quarter_values))
            best_quarter = qframe.iloc[best_quarter_position]
            worst_quarter = qframe.iloc[worst_quarter_position]
            best_quarter_mask = (
                (dframe["date"] >= best_quarter["rebalance_date"])
                & (dframe["date"] < best_quarter["outer_end_exclusive"])
            ).to_numpy()
            without_best_quarter = remove_block_as_cash(
                returns,
                best_quarter_mask,
                calendar_years=calendar_years,
            )

            pair_annual_rows: list[dict[str, Any]] = []
            for year, group in dframe.groupby(dframe["date"].dt.year, sort=True):
                values = group[return_column].to_numpy(dtype=float)
                total_return = float(np.prod(1.0 + values) - 1.0)
                row = {
                    "pair": pair,
                    "sector": common["sector"],
                    "transaction_cost_bps": float(cost),
                    "calendar_year": int(year),
                    "calendar_year_complete": _calendar_year_is_complete(
                        int(year), study_start, study_end_exclusive
                    ),
                    "sessions": int(len(group)),
                    "net_return": total_return,
                    "positive": bool(total_return > 0.0),
                    "study_start_partial_year": bool(int(year) == study_start.year),
                    "study_end_partial_year": bool(int(year) == (study_end_exclusive - pd.Timedelta(days=1)).year),
                }
                pair_annual_rows.append(row)
                annual_rows.append(row)
            annual_frame = pd.DataFrame(pair_annual_rows)
            complete_years = annual_frame.loc[annual_frame["calendar_year_complete"]]
            positive_year_fraction_all = float(annual_frame["positive"].mean())
            positive_complete_year_fraction = (
                float(complete_years["positive"].mean()) if len(complete_years) else math.nan
            )
            if len(complete_years):
                best_year_row = complete_years.sort_values(
                    ["net_return", "calendar_year"],
                    ascending=[False, True],
                    kind="mergesort",
                ).iloc[0]
                best_year = int(best_year_row["calendar_year"])
                best_year_return = float(best_year_row["net_return"])
                best_year_mask = dframe["date"].dt.year.eq(best_year).to_numpy()
                without_best_year = remove_block_as_cash(
                    returns,
                    best_year_mask,
                    calendar_years=calendar_years,
                )
            else:
                best_year = None
                best_year_return = math.nan
                best_year_mask = np.zeros(len(dframe), dtype=bool)
                without_best_year = path_statistics(returns, calendar_years=calendar_years)

            quarter_logs = np.log1p(quarter_values)
            total_log_growth = float(np.log1p(returns).sum())
            best_quarter_log = float(quarter_logs[best_quarter_position])
            positive_log_sum = float(np.clip(quarter_logs, 0.0, None).sum())
            best_year_log = float(np.log1p(best_year_return)) if np.isfinite(best_year_return) else math.nan

            block_rates: list[float] = []
            for block in range(10):
                mask = block_labels == block
                values = returns[mask]
                block_start = study_start + pd.DateOffset(years=block)
                block_end = (
                    study_end_exclusive
                    if block == 9
                    else study_start + pd.DateOffset(years=block + 1)
                )
                years = exact_calendar_years(block_start, block_end)
                log_growth = float(np.log1p(values).sum())
                rate = float(log_growth / years)
                block_rates.append(rate)
                block_rows.append(
                    {
                        "pair": pair,
                        "sector": common["sector"],
                        "transaction_cost_bps": float(cost),
                        "anniversary_block": block + 1,
                        "block_start_inclusive": block_start,
                        "block_end_exclusive": block_end,
                        "calendar_years": years,
                        "sessions": int(mask.sum()),
                        "total_return": float(np.expm1(log_growth)),
                        "annual_log_growth_rate": rate,
                    }
                )

            row = {
                **common,
                "transaction_cost_bps": float(cost),
                "study_start_inclusive": study_start,
                "study_end_exclusive": study_end_exclusive,
                "calendar_years": calendar_years,
                "study_sessions": int(len(dframe)),
                **stats,
                "scheduled_quarter_count": int(len(quarter_values)),
                "positive_quarter_fraction": positive_quarter_fraction,
                "positive_quarter_count": int((quarter_values > 0.0).sum()),
                "nonpositive_quarter_count": int((quarter_values <= 0.0).sum()),
                "median_quarter_return": float(np.median(quarter_values)),
                "worst_quarter_return": float(quarter_values[worst_quarter_position]),
                "worst_quarter_start": worst_quarter["rebalance_date"],
                "best_quarter_return": float(quarter_values[best_quarter_position]),
                "best_quarter_start": best_quarter["rebalance_date"],
                "positive_calendar_year_fraction_all_displayed": positive_year_fraction_all,
                "positive_complete_calendar_year_fraction": positive_complete_year_fraction,
                "complete_calendar_year_count": int(len(complete_years)),
                "best_complete_calendar_year": best_year,
                "best_complete_calendar_year_return": best_year_return,
                "cagr_without_best_quarter": without_best_quarter["calendar_cagr"],
                "total_return_without_best_quarter": without_best_quarter["net_total_return"],
                "max_drawdown_without_best_quarter": without_best_quarter["max_drawdown"],
                "cagr_without_best_complete_year": without_best_year["calendar_cagr"],
                "total_return_without_best_complete_year": without_best_year["net_total_return"],
                "max_drawdown_without_best_complete_year": without_best_year["max_drawdown"],
                "best_quarter_cagr_drop": stats["calendar_cagr"] - without_best_quarter["calendar_cagr"],
                "best_year_cagr_drop": stats["calendar_cagr"] - without_best_year["calendar_cagr"],
                "best_quarter_share_of_total_log_growth": best_quarter_log / total_log_growth
                if total_log_growth > 0.0 and best_quarter_log > 0.0
                else math.nan,
                "best_quarter_share_of_positive_quarter_log_growth": best_quarter_log / positive_log_sum
                if positive_log_sum > 0.0 and best_quarter_log > 0.0
                else math.nan,
                "best_year_share_of_total_log_growth": best_year_log / total_log_growth
                if np.isfinite(best_year_log)
                and best_year_log > 0.0
                and total_log_growth > 0.0
                else math.nan,
                "full_calendar_metric": True,
                "active_period_annualization_used": False,
                "unavailable_and_failed_periods_are_cash": True,
                "ranking_universe_is_ex_post_descriptive": True,
            }
            metric_locations[cost].append(len(metric_rows))
            metric_rows.append(row)
            bootstrap_rates[cost].append(np.asarray(block_rates, dtype=float))
            counterfactual_rows.extend(
                [
                    {
                        "pair": pair,
                        "transaction_cost_bps": float(cost),
                        "counterfactual": "best_quarter_replaced_by_cash",
                        "removed_period": best_quarter["rebalance_date"],
                        "removed_period_return": float(quarter_values[best_quarter_position]),
                        **without_best_quarter,
                    },
                    {
                        "pair": pair,
                        "transaction_cost_bps": float(cost),
                        "counterfactual": "best_complete_calendar_year_replaced_by_cash",
                        "removed_period": best_year,
                        "removed_period_return": best_year_return,
                        **without_best_year,
                    },
                ]
            )

    metrics = pd.DataFrame(metric_rows)
    confidence_rows: list[dict[str, Any]] = []
    shared_counts: np.ndarray | None = None
    bootstrap = {
        "method": "anniversary_year_cluster_percentile",
        "draws": 10_000,
        "seed": int(extension.get("protocol", {}).get("random_seed", 20260901)),
        "confidence_level": 0.95,
        **extension.get("bootstrap", {}),
    }
    if int(bootstrap["draws"]) <= 0:
        raise ValueError("bootstrap draws must be positive")
    if not math.isclose(float(bootstrap["confidence_level"]), 0.95):
        raise ValueError(
            "the fixed bootstrap helper currently implements a 95% interval"
        )
    for cost in (1, 2):
        rates = np.vstack(bootstrap_rates[cost])
        quantiles, shared_counts = bootstrap_cagr_from_log_blocks(
            rates,
            draws=int(bootstrap["draws"]),
            seed=int(bootstrap["seed"]),
            shared_counts=shared_counts,
        )
        for location, quantile in zip(metric_locations[cost], quantiles, strict=True):
            metrics.loc[location, "bootstrap_cagr_ci_low"] = float(quantile[0])
            metrics.loc[location, "bootstrap_cagr_median"] = float(quantile[1])
            metrics.loc[location, "bootstrap_cagr_ci_high"] = float(quantile[2])
            row = metrics.loc[location]
            confidence_rows.append(
                {
                    "pair": row["pair"],
                    "sector": row["sector"],
                    "transaction_cost_bps": float(cost),
                    "method": bootstrap["method"],
                    "draws": int(bootstrap["draws"]),
                    "seed": int(bootstrap["seed"]),
                    "confidence_level": float(bootstrap["confidence_level"]),
                    "cagr_ci_low": float(quantile[0]),
                    "cagr_median": float(quantile[1]),
                    "cagr_ci_high": float(quantile[2]),
                }
            )

    ranking_cost_bps = float(evaluation.get("ranking_cost_bps", 2.0))
    ranking_input = metrics.loc[
        pd.to_numeric(metrics["transaction_cost_bps"], errors="coerce").eq(
            ranking_cost_bps
        )
    ].copy()
    if len(ranking_input) != len(master):
        raise RuntimeError(
            "headline ranking input must contain one row per fixed identity"
        )
    ranking_config = {
        "adequate_availability_minimum_fraction": 0.80,
        **extension.get("ranking", {}),
    }
    full_rankings, full_ranking_summary = fixed_universe_rankings(
        ranking_input,
        adequate_availability_fraction=float(
            ranking_config["adequate_availability_minimum_fraction"]
        ),
    )
    full_rankings.insert(0, "ranking_scope", "fixed_114_primary")
    full_ranking_summary.insert(0, "ranking_scope", "fixed_114_primary")
    selected_input = ranking_input.loc[
        ranking_input["original_v6_selected_identity"].astype(bool)
    ].copy()
    selected_rankings, selected_ranking_summary = fixed_universe_rankings(
        selected_input,
        adequate_availability_fraction=float(
            ranking_config["adequate_availability_minimum_fraction"]
        ),
    )
    selected_rankings.insert(0, "ranking_scope", "original_v6_selected_82_subset")
    selected_ranking_summary.insert(
        0, "ranking_scope", "original_v6_selected_82_subset"
    )
    rankings = pd.concat([full_rankings, selected_rankings], ignore_index=True)
    ranking_summary = pd.concat(
        [full_ranking_summary, selected_ranking_summary], ignore_index=True
    )

    annual_output = pd.DataFrame(annual_rows)
    counterfactual_output = pd.DataFrame(counterfactual_rows)
    confidence_output = pd.DataFrame(confidence_rows)
    block_output = pd.DataFrame(block_rows)
    expected_calendar_year_rows_per_pair_cost = (
        (study_end_exclusive - pd.Timedelta(days=1)).year - study_start.year + 1
    )
    expected_complete_calendar_years = sum(
        _calendar_year_is_complete(year, study_start, study_end_exclusive)
        for year in range(
            study_start.year,
            (study_end_exclusive - pd.Timedelta(days=1)).year + 1,
        )
    )
    audit_rows = [
        {"field": "fixed_identity_count", "value": len(master), "passed": len(master) == 114},
        {
            "field": "original_v6_selected_identity_count",
            "value": int(master["original_v6_selected_identity"].sum()),
            "passed": int(master["original_v6_selected_identity"].sum()) == 82,
        },
        {
            "field": "quarter_interval_count",
            "value": len(schedule),
            "passed": len(schedule) == expected_quarters,
        },
        {
            "field": "pair_quarter_row_count",
            "value": len(quarterly),
            "passed": len(quarterly) == len(master) * expected_quarters,
        },
        {
            "field": "daily_pair_session_row_count",
            "value": len(dense),
            "passed": len(dense) == 114 * len(dates),
        },
        {
            "field": "successful_sparse_daily_rows_reconcile",
            "value": int(len(sparse)),
            "passed": bool(
                sparse_reconciliation["expected_daily_rows"].eq(
                    sparse_reconciliation["actual_daily_rows"]
                ).all()
            ),
        },
        {
            "field": "pair_cost_metric_row_count",
            "value": len(metrics),
            "passed": len(metrics) == len(master) * 2,
        },
        {
            "field": "all_metrics_use_full_quarter_denominator",
            "value": int(metrics["scheduled_quarter_count"].min()),
            "passed": bool(
                metrics["scheduled_quarter_count"].eq(expected_quarters).all()
            ),
        },
        {
            "field": "annual_return_row_count",
            "value": len(annual_output),
            "passed": len(annual_output)
            == len(master) * 2 * expected_calendar_year_rows_per_pair_cost,
        },
        {
            "field": "complete_calendar_year_count",
            "value": int(metrics["complete_calendar_year_count"].min()),
            "passed": bool(
                metrics["complete_calendar_year_count"]
                .eq(expected_complete_calendar_years)
                .all()
            ),
        },
        {
            "field": "counterfactual_row_count",
            "value": len(counterfactual_output),
            "passed": len(counterfactual_output) == len(master) * 2 * 2,
        },
        {
            "field": "bootstrap_interval_row_count",
            "value": len(confidence_output),
            "passed": len(confidence_output) == len(master) * 2,
        },
        {
            "field": "bootstrap_block_row_count",
            "value": len(block_output),
            "passed": len(block_output) == len(master) * 2 * 10,
        },
        {
            "field": "headline_cost_is_2bp",
            "value": ranking_cost_bps,
            "passed": math.isclose(ranking_cost_bps, 2.0),
        },
        {
            "field": "all_five_rankings_created",
            "value": full_ranking_summary["rank_type"].nunique(),
            "passed": full_ranking_summary["rank_type"].nunique() == 5,
        },
        {
            "field": "non_sharpe_rankings_keep_all_pairs",
            "value": int(
                full_rankings.loc[
                    ~full_rankings["rank_type"].eq("adequate_availability_sharpe")
                ].groupby("rank_type")["pair"].nunique().min()
            ),
            "passed": bool(
                full_rankings.loc[
                    ~full_rankings["rank_type"].eq("adequate_availability_sharpe")
                ].groupby("rank_type")["pair"].nunique().eq(114).all()
            ),
        },
        {
            "field": "active_period_annualization_absent",
            "value": bool((~metrics["active_period_annualization_used"]).all()),
            "passed": bool((~metrics["active_period_annualization_used"]).all()),
        },
    ]
    audit = pd.DataFrame(audit_rows)
    if not _as_bool(audit["passed"]).all():
        failed = audit.loc[~_as_bool(audit["passed"]), "field"].tolist()
        raise RuntimeError("aggregation quality checks failed: " + ", ".join(failed))

    _log("writing machine-readable full-panel artifacts")
    _atomic_csv(dense, processed / "pair_identity_daily_returns.csv.gz")
    _atomic_csv(quarterly, quarter_path)
    _atomic_csv(annual_output, processed / "pair_identity_annual_returns.csv")
    _atomic_csv(metrics, processed / "pair_identity_metrics.csv")
    _atomic_csv(counterfactual_output, processed / "pair_identity_counterfactuals.csv")
    _atomic_csv(confidence_output, processed / "pair_identity_confidence_intervals.csv")
    _atomic_csv(block_output, processed / "pair_identity_bootstrap_blocks.csv")
    _atomic_csv(pd.DataFrame(stability_rows), processed / "pair_identity_parameter_stability.csv")
    _atomic_csv(pd.DataFrame(coverage_rows), processed / "pair_identity_coverage_diagnostics.csv")
    _atomic_csv(rankings, processed / "pair_identity_rankings.csv")
    _atomic_csv(ranking_summary, processed / "pair_identity_ranking_summary.csv")
    _atomic_csv(audit, processed / "pair_identity_audit.csv")

    generated = [
        "pair_identity_daily_returns.csv.gz",
        "pair_identity_quarterly_results.csv.gz",
        "pair_identity_annual_returns.csv",
        "pair_identity_metrics.csv",
        "pair_identity_counterfactuals.csv",
        "pair_identity_confidence_intervals.csv",
        "pair_identity_bootstrap_blocks.csv",
        "pair_identity_parameter_stability.csv",
        "pair_identity_coverage_diagnostics.csv",
        "pair_identity_rankings.csv",
        "pair_identity_ranking_summary.csv",
        "pair_identity_audit.csv",
    ]
    quality = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "fixed_identity_count": int(len(master)),
        "original_v6_selected_subset_count": int(master["original_v6_selected_identity"].sum()),
        "quarter_intervals": int(len(schedule)),
        "pair_quarter_rows": int(len(quarterly)),
        "study_sessions": int(len(dates)),
        "calendar_years": calendar_years,
        "ranking_is_ex_post_descriptive": True,
        "active_period_annualization_used": False,
        "checks": {row["field"]: bool(row["passed"]) for row in audit_rows},
        "artifacts": {
            name: {"sha256": _sha256(processed / name), "bytes": (processed / name).stat().st_size}
            for name in generated
        },
    }
    _atomic_json(processed / "pair_identity_quality_checks.json", quality)
    _log("aggregation complete")
    return quality


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    aggregate(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
