#!/usr/bin/env python3
"""Independently reconstruct V8 pair returns from closes and recorded weights.

This audit intentionally does not import the backtest, V7/V8 simulation, or
repricing modules.  Its accounting equations are repeated here so the final
V8 path can be checked against an implementation with a separate call graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/pair_strategy_v8_pit_lineage_next_close_v2.json")
DEFAULT_DAILY_NAME = "v8_pair_daily_paths.csv.gz"
DEFAULT_TRADES_NAME = "v8_trade_ledger.csv"
DEFAULT_DETAIL_NAME = "v8_independent_reconstruction_daily.csv.gz"
DEFAULT_SUMMARY_NAME = "v8_independent_reconstruction_summary.json"
ATTRIBUTION_CONVENTION = "directional_episode_sequential_flip_factor_v1"


DAILY_REQUIRED = {
    "date",
    "pair",
    "strategy_id",
    "vintage_rebalance_date",
    "ticker_a",
    "ticker_b",
    "price_a",
    "price_b",
    "fill_event",
    "fill_decision_date",
    "fill_decision_zscore",
    "gross_return",
    "turnover",
    "transaction_cost",
    "borrow_cost",
    "net_return_1bp",
    "net_return_2bp",
    "weight_a",
    "weight_b",
    "target_weight_a",
    "target_weight_b",
    "ensemble_active_member_count",
    "ensemble_held_member_count",
}

TRADE_REQUIRED = {
    "pair",
    "trade_number",
    "strategy_id",
    "entry_decision_date",
    "entry_date",
    "entry_fill_price_a",
    "entry_fill_price_b",
    "exit_decision_date",
    "exit_date",
    "exit_fill_price_a",
    "exit_fill_price_b",
    "direction_ticker_a",
    "trading_sessions",
    "entry_zscore",
    "entry_fill_event",
    "exit_zscore",
    "exit_reason",
    "exit_fill_event",
    "closed",
    "net_return_1bp",
    "net_return_2bp",
    "gross_return_sum",
    "transaction_cost_1bp",
    "transaction_cost_2bp",
    "opening_transaction_cost_1bp",
    "closing_transaction_cost_1bp",
    "holding_rebalance_transaction_cost_1bp",
    "opening_transaction_cost_2bp",
    "closing_transaction_cost_2bp",
    "holding_rebalance_transaction_cost_2bp",
    "borrow_cost",
    "entry_transition",
    "exit_transition",
    "attribution_convention",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value in trade ledger: {value!r}")


def _raw_simple_returns(series: pd.Series) -> pd.Series:
    """Calculate unfilled close-to-close returns without using engine code."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) > 1:
        previous = values[:-1]
        current = values[1:]
        valid = (
            np.isfinite(previous)
            & np.isfinite(current)
            & (previous > 0.0)
            & (current > 0.0)
        )
        output[1:][valid] = current[valid] / previous[valid] - 1.0
    return pd.Series(output, index=series.index, dtype=float)


def _numeric_comparison(
    recorded: Iterable[Any],
    reconstructed: Iterable[Any],
    *,
    tolerance: float,
    allow_paired_nan: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    left = pd.to_numeric(pd.Series(recorded), errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(pd.Series(reconstructed), errors="coerce").to_numpy(dtype=float)
    if left.shape != right.shape:
        raise ValueError("numeric comparison inputs have different shapes")
    paired_nan = np.isnan(left) & np.isnan(right) if allow_paired_nan else np.zeros(len(left), dtype=bool)
    finite = np.isfinite(left) & np.isfinite(right)
    errors = np.full(len(left), np.nan, dtype=float)
    errors[finite] = np.abs(left[finite] - right[finite])
    matches = paired_nan | (finite & (errors <= tolerance))
    mismatch_count = int((~matches).sum())
    maximum = float(np.nanmax(errors)) if finite.any() else (0.0 if matches.all() else math.inf)
    return errors, matches, mismatch_count, maximum


def _date_matches(left: Any, right: Any) -> bool:
    lhs = pd.to_datetime(left, errors="coerce")
    rhs = pd.to_datetime(right, errors="coerce")
    if pd.isna(lhs) and pd.isna(rhs):
        return True
    return bool(lhs == rhs)


def _scalar_numeric_match(
    left: Any,
    right: Any,
    tolerance: float,
    *,
    allow_paired_nan: bool = False,
) -> tuple[bool, float]:
    errors, matches, _, maximum = _numeric_comparison(
        [left], [right], tolerance=tolerance, allow_paired_nan=allow_paired_nan
    )
    error = float(errors[0]) if np.isfinite(errors[0]) else maximum
    return bool(matches[0]), error


def _validate_inputs(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    annual_short_borrow_bps: float,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if not math.isfinite(annual_short_borrow_bps) or annual_short_borrow_bps < 0:
        raise ValueError("annual_short_borrow_bps must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if daily.empty:
        raise ValueError("daily paths are empty")
    if prices.empty:
        raise ValueError("adjusted-close snapshot is empty")
    _require_columns(daily, DAILY_REQUIRED, "daily paths")
    _require_columns(trades, TRADE_REQUIRED, "trade ledger")

    checked_daily = daily.copy()
    checked_daily["date"] = pd.to_datetime(checked_daily["date"], errors="coerce")
    checked_daily["fill_decision_date"] = pd.to_datetime(
        checked_daily["fill_decision_date"], errors="coerce"
    )
    raw_vintage_dates = checked_daily["vintage_rebalance_date"]
    checked_daily["vintage_rebalance_date"] = pd.to_datetime(
        raw_vintage_dates, errors="coerce"
    )
    if (raw_vintage_dates.notna() & checked_daily["vintage_rebalance_date"].isna()).any():
        raise ValueError("daily paths contain invalid vintage rebalance dates")
    if checked_daily["date"].isna().any():
        raise ValueError("daily paths contain invalid dates")
    if checked_daily[["pair", "ticker_a", "ticker_b", "strategy_id"]].isna().any().any():
        raise ValueError("daily paths contain missing pair, ticker, or strategy labels")
    if checked_daily.duplicated(["pair", "date"]).any():
        raise ValueError("daily paths contain duplicate pair/date rows")
    for pair, group in checked_daily.groupby("pair", sort=False):
        if not group["date"].is_monotonic_increasing:
            raise ValueError(f"daily dates are not increasing for {pair}")
        if group["ticker_a"].astype(str).nunique() != 1 or group["ticker_b"].astype(str).nunique() != 1:
            raise ValueError(f"daily leg mapping changes within {pair}")
        if str(group["ticker_a"].iloc[0]) == str(group["ticker_b"].iloc[0]):
            raise ValueError(f"daily leg mapping repeats the same ticker for {pair}")

    finite_daily = [
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "net_return_1bp",
        "net_return_2bp",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "ensemble_active_member_count",
        "ensemble_held_member_count",
    ]
    for column in finite_daily:
        values = pd.to_numeric(checked_daily[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"daily column {column} must be finite")
        checked_daily[column] = values
    for column in (
        "ensemble_active_member_count",
        "ensemble_held_member_count",
    ):
        member_counts = checked_daily[column].to_numpy(dtype=float)
        if bool(
            (member_counts < 0.0).any()
            or not np.allclose(
                member_counts, np.rint(member_counts), rtol=0.0, atol=tolerance
            )
        ):
            raise ValueError(f"{column} must contain nonnegative integers")

    checked_trades = trades.copy()
    if checked_trades.empty:
        # An empty, correctly shaped ledger remains a valid input; the required
        # winning/losing/directional coverage below will make the audit fail.
        checked_trades["trade_number"] = pd.Series(dtype="int64")
    else:
        numbers = pd.to_numeric(checked_trades["trade_number"], errors="coerce")
        if numbers.isna().any() or (numbers % 1 != 0).any() or (numbers <= 0).any():
            raise ValueError("trade_number must contain positive integers")
        checked_trades["trade_number"] = numbers.astype(int)
        if checked_trades.duplicated(["pair", "trade_number"]).any():
            raise ValueError("trade ledger contains duplicate pair/trade_number rows")

    checked_prices = prices.copy()
    checked_prices.index = pd.DatetimeIndex(
        pd.to_datetime(checked_prices.index, errors="coerce"), name="date"
    )
    if checked_prices.index.isna().any():
        raise ValueError("adjusted-close snapshot contains invalid dates")
    if checked_prices.index.has_duplicates:
        raise ValueError("adjusted-close snapshot contains duplicate dates")
    if not checked_prices.index.is_monotonic_increasing:
        raise ValueError("adjusted-close snapshot dates must be increasing")

    required_tickers = set(checked_daily["ticker_a"].astype(str)).union(
        checked_daily["ticker_b"].astype(str)
    )
    missing_tickers = sorted(required_tickers.difference(checked_prices.columns.astype(str)))
    if missing_tickers:
        raise ValueError(f"adjusted-close snapshot is missing tickers: {missing_tickers}")
    missing_dates = checked_daily.loc[
        ~checked_daily["date"].isin(checked_prices.index), "date"
    ].drop_duplicates()
    if len(missing_dates):
        raise ValueError(
            "daily paths contain dates absent from adjusted-close snapshot: "
            + ", ".join(value.strftime("%Y-%m-%d") for value in missing_dates.head(5))
        )
    return checked_daily, checked_trades, checked_prices


def _directional_episode_reconstruction(
    *,
    pair: str,
    group: pd.DataFrame,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    held_a: np.ndarray,
    held_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    drifted_a: np.ndarray,
    drifted_b: np.ndarray,
    gross: np.ndarray,
    turnover: np.ndarray,
    borrow: np.ndarray,
    net_1bp: np.ndarray,
    net_2bp: np.ndarray,
    transaction_1bp: np.ndarray,
    transaction_2bp: np.ndarray,
    tolerance: float,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
    list[str],
]:
    """Rebuild directional trades without calling the V8 production helper.

    A reversal row earns gross return and borrow on the old held position before
    the close fill.  Its liquidation cost belongs to the old trade, while its
    opening cost belongs to the new trade.  Two sequential subreturns preserve
    the exact daily path factor without double-counting the shared fill row.
    """

    size = len(group)
    active = (np.abs(target_a) + np.abs(target_b)) > tolerance
    held_active = (np.abs(held_a) + np.abs(held_b)) > tolerance
    target_direction = np.sign(target_a).astype(int)
    held_direction = np.sign(held_a).astype(int)
    previous_active = np.r_[False, active[:-1]]
    previous_direction = np.r_[0, target_direction[:-1]]
    flips = active & previous_active & (target_direction != previous_direction)
    starts = active & (~previous_active | flips)

    violations: list[str] = []
    undefined_target = active & (np.abs(target_a) <= tolerance)
    undefined_held = held_active & (np.abs(held_a) <= tolerance)
    non_opposing_target_legs = active & (
        np.sign(target_a) == np.sign(target_b)
    )
    non_opposing_held_legs = held_active & (
        np.sign(held_a) == np.sign(held_b)
    )
    if undefined_target.any() or undefined_held.any():
        violations.append(
            f"{pair}: pair direction is undefined because ticker-a weight is zero"
        )
    if non_opposing_target_legs.any() or non_opposing_held_legs.any():
        violations.append(
            f"{pair}: active pair legs do not have opposing signs"
        )

    post_fill_trade = np.zeros(size, dtype=int)
    episode_directions: dict[int, int] = {}
    current_number = 0
    for location in range(size):
        if not active[location]:
            continue
        if starts[location]:
            current_number += 1
            episode_directions[current_number] = int(target_direction[location])
        post_fill_trade[location] = current_number
    held_trade = np.r_[0, post_fill_trade[:-1]]

    factors_1bp: dict[int, list[float]] = {
        number: [] for number in episode_directions
    }
    factors_2bp: dict[int, list[float]] = {
        number: [] for number in episode_directions
    }
    gross_sum = {number: 0.0 for number in episode_directions}
    borrow_sum = {number: 0.0 for number in episode_directions}
    opening_cost_1bp = {number: 0.0 for number in episode_directions}
    closing_cost_1bp = {number: 0.0 for number in episode_directions}
    holding_cost_1bp = {number: 0.0 for number in episode_directions}
    opening_cost_2bp = {number: 0.0 for number in episode_directions}
    closing_cost_2bp = {number: 0.0 for number in episode_directions}
    holding_cost_2bp = {number: 0.0 for number in episode_directions}
    flip_factor_error = 0.0
    split_cost_error = 0.0

    for location in range(size):
        old_number = int(held_trade[location])
        new_number = int(post_fill_trade[location])
        if old_number:
            gross_sum[old_number] += float(gross[location])
            borrow_sum[old_number] += float(borrow[location])
        elif (
            abs(float(gross[location])) > tolerance
            or abs(float(borrow[location])) > tolerance
        ):
            violations.append(
                f"{pair} {group['date'].iloc[location].date()}: cash row has held-position economics"
            )

        if old_number == new_number and old_number:
            factors_1bp[old_number].append(float(net_1bp[location]))
            factors_2bp[old_number].append(float(net_2bp[location]))
            holding_cost_1bp[old_number] += float(transaction_1bp[location])
            holding_cost_2bp[old_number] += float(transaction_2bp[location])
        elif not old_number and new_number:
            factors_1bp[new_number].append(float(net_1bp[location]))
            factors_2bp[new_number].append(float(net_2bp[location]))
            opening_cost_1bp[new_number] += float(transaction_1bp[location])
            opening_cost_2bp[new_number] += float(transaction_2bp[location])
        elif old_number and not new_number:
            factors_1bp[old_number].append(float(net_1bp[location]))
            factors_2bp[old_number].append(float(net_2bp[location]))
            closing_cost_1bp[old_number] += float(transaction_1bp[location])
            closing_cost_2bp[old_number] += float(transaction_2bp[location])
        elif old_number and new_number:
            both_legs_reverse = bool(
                held_a[location] * target_a[location] < -(tolerance**2)
                and held_b[location] * target_b[location] < -(tolerance**2)
            )
            if not both_legs_reverse:
                violations.append(
                    f"{pair} {group['date'].iloc[location].date()}: directional reversal does not reverse both legs"
                )
            denominator = 1.0 + float(gross[location])
            close_turnover = abs(float(drifted_a[location])) + abs(
                float(drifted_b[location])
            )
            open_turnover = abs(float(target_a[location])) + abs(
                float(target_b[location])
            )
            total_turnover = close_turnover + open_turnover
            split_cost_error = max(
                split_cost_error,
                abs(total_turnover - float(turnover[location])),
            )
            close_cost_1bp = close_turnover / 10_000.0 * denominator
            open_cost_1bp = open_turnover / 10_000.0 * denominator
            close_cost_2bp = 2.0 * close_cost_1bp
            open_cost_2bp = 2.0 * open_cost_1bp
            old_1bp = (
                float(gross[location])
                - float(borrow[location])
                - close_cost_1bp
            )
            old_2bp = (
                float(gross[location])
                - float(borrow[location])
                - close_cost_2bp
            )
            if 1.0 + old_1bp <= tolerance or 1.0 + old_2bp <= tolerance:
                raise ValueError(
                    f"non-positive pre-entry NAV at direction reversal for {pair}"
                )
            new_1bp = -open_cost_1bp / (1.0 + old_1bp)
            new_2bp = -open_cost_2bp / (1.0 + old_2bp)
            reconstructed_1bp = (1.0 + old_1bp) * (1.0 + new_1bp) - 1.0
            reconstructed_2bp = (1.0 + old_2bp) * (1.0 + new_2bp) - 1.0
            flip_factor_error = max(
                flip_factor_error,
                abs(reconstructed_1bp - float(net_1bp[location])),
                abs(reconstructed_2bp - float(net_2bp[location])),
            )
            factors_1bp[old_number].append(old_1bp)
            factors_2bp[old_number].append(old_2bp)
            factors_1bp[new_number].append(new_1bp)
            factors_2bp[new_number].append(new_2bp)
            closing_cost_1bp[old_number] += close_cost_1bp
            closing_cost_2bp[old_number] += close_cost_2bp
            opening_cost_1bp[new_number] += open_cost_1bp
            opening_cost_2bp[new_number] += open_cost_2bp
        elif (
            abs(float(net_1bp[location])) > tolerance
            or abs(float(net_2bp[location])) > tolerance
            or abs(float(transaction_2bp[location])) > tolerance
        ):
            violations.append(
                f"{pair} {group['date'].iloc[location].date()}: all-cash row has nonzero trading economics"
            )

    mixed_direction_episode_count = 0
    for number, direction in episode_directions.items():
        post_mask = post_fill_trade == number
        held_mask = held_trade == number
        mixed = bool(
            (target_direction[post_mask] != direction).any()
            or (held_direction[held_mask] != direction).any()
        )
        mixed_direction_episode_count += int(mixed)
    if mixed_direction_episode_count:
        violations.append(
            f"{pair}: {mixed_direction_episode_count} directional episodes contain mixed signs"
        )

    episode_rows: list[dict[str, Any]] = []
    for number, start in enumerate(np.flatnonzero(starts), start=1):
        later_change = np.flatnonzero(post_fill_trade[start + 1 :] != number)
        closed = bool(len(later_change))
        stop = (
            start + 1 + int(later_change[0])
            if closed
            else size - 1
        )
        reversal_exit = bool(closed and post_fill_trade[stop] > 0)
        if closed:
            exit_decision = group["fill_decision_date"].iloc[stop]
            exit_zscore = float(group["fill_decision_zscore"].iloc[stop])
            exit_reason = (
                "aggregate_direction_reversal"
                if reversal_exit
                else str(group["fill_event"].iloc[stop])
            )
            exit_transition = (
                "direction_reversal" if reversal_exit else "active_to_flat"
            )
            exit_price_a = float(raw_a[stop])
            exit_price_b = float(raw_b[stop])
        else:
            exit_decision = pd.NaT
            exit_zscore = math.nan
            exit_reason = "end_of_study_open"
            exit_transition = "end_of_study_open"
            exit_price_a = math.nan
            exit_price_b = math.nan
        entry_decision = group["fill_decision_date"].iloc[start]
        if start == 0 or not _date_matches(
            entry_decision, group["date"].iloc[start - 1]
        ):
            violations.append(
                f"{pair} trade {number}: entry fill decision is not from the prior session"
            )
        if closed and not _date_matches(
            exit_decision, group["date"].iloc[stop - 1]
        ):
            violations.append(
                f"{pair} trade {number}: exit fill decision is not from the prior session"
            )
        episode_rows.append(
            {
                "pair": str(pair),
                "trade_number": number,
                "strategy_id": str(group["strategy_id"].iloc[start]),
                "entry_decision_date": entry_decision,
                "entry_date": group["date"].iloc[start],
                "entry_fill_price_a": raw_a[start],
                "entry_fill_price_b": raw_b[start],
                "entry_fill_event": str(group["fill_event"].iloc[start]),
                "exit_decision_date": exit_decision,
                "exit_date": group["date"].iloc[stop],
                "exit_fill_price_a": exit_price_a,
                "exit_fill_price_b": exit_price_b,
                "direction_ticker_a": (
                    "long" if episode_directions[number] > 0 else "short"
                ),
                "trading_sessions": int(stop - start),
                "entry_zscore": float(group["fill_decision_zscore"].iloc[start]),
                "exit_zscore": exit_zscore,
                "exit_reason": exit_reason,
                "exit_fill_event": (
                    str(group["fill_event"].iloc[stop])
                    if closed
                    else "end_of_study_open"
                ),
                "closed": closed,
                "net_return_1bp": float(
                    np.prod(1.0 + np.asarray(factors_1bp[number], dtype=float))
                    - 1.0
                ),
                "net_return_2bp": float(
                    np.prod(1.0 + np.asarray(factors_2bp[number], dtype=float))
                    - 1.0
                ),
                "gross_return_sum": float(gross_sum[number]),
                "transaction_cost_1bp": float(
                    opening_cost_1bp[number]
                    + closing_cost_1bp[number]
                    + holding_cost_1bp[number]
                ),
                "transaction_cost_2bp": float(
                    opening_cost_2bp[number]
                    + closing_cost_2bp[number]
                    + holding_cost_2bp[number]
                ),
                "opening_transaction_cost_1bp": float(opening_cost_1bp[number]),
                "closing_transaction_cost_1bp": float(closing_cost_1bp[number]),
                "holding_rebalance_transaction_cost_1bp": float(
                    holding_cost_1bp[number]
                ),
                "opening_transaction_cost_2bp": float(opening_cost_2bp[number]),
                "closing_transaction_cost_2bp": float(closing_cost_2bp[number]),
                "holding_rebalance_transaction_cost_2bp": float(
                    holding_cost_2bp[number]
                ),
                "borrow_cost": float(borrow_sum[number]),
                "entry_transition": (
                    "direction_reversal" if held_trade[start] else "flat_to_active"
                ),
                "exit_transition": exit_transition,
                "attribution_convention": ATTRIBUTION_CONVENTION,
            }
        )

    episode_factor_1bp = float(
        np.prod(1.0 + np.asarray([row["net_return_1bp"] for row in episode_rows]))
    )
    episode_factor_2bp = float(
        np.prod(1.0 + np.asarray([row["net_return_2bp"] for row in episode_rows]))
    )
    daily_factor_1bp = float(np.prod(1.0 + net_1bp))
    daily_factor_2bp = float(np.prod(1.0 + net_2bp))
    path_factor_error = max(
        abs(episode_factor_1bp - daily_factor_1bp),
        abs(episode_factor_2bp - daily_factor_2bp),
        flip_factor_error,
    )
    component_error = max(
        abs(sum(row["gross_return_sum"] for row in episode_rows) - float(gross.sum())),
        abs(sum(row["borrow_cost"] for row in episode_rows) - float(borrow.sum())),
        abs(
            sum(row["transaction_cost_1bp"] for row in episode_rows)
            - float(transaction_1bp.sum())
        ),
        abs(
            sum(row["transaction_cost_2bp"] for row in episode_rows)
            - float(transaction_2bp.sum())
        ),
    )
    return (
        episode_rows,
        held_trade,
        post_fill_trade,
        flips,
        {
            "direction_flip_count": int(flips.sum()),
            "mixed_direction_episode_count": mixed_direction_episode_count,
            "path_factor_error": path_factor_error,
            "component_sum_error": component_error,
            "split_turnover_error": split_cost_error,
        },
        violations,
    )


def audit_reconstruction(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    annual_short_borrow_bps: float,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the full daily path and reconcile every ledger episode."""

    daily, trades, prices = _validate_inputs(
        daily,
        trades,
        prices,
        transaction_cost_bps=transaction_cost_bps,
        annual_short_borrow_bps=annual_short_borrow_bps,
        tolerance=tolerance,
    )
    return_cache = {
        str(ticker): _raw_simple_returns(prices[str(ticker)])
        for ticker in set(daily["ticker_a"].astype(str)).union(daily["ticker_b"].astype(str))
    }
    detail_parts: list[pd.DataFrame] = []
    derived_trades: list[dict[str, Any]] = []
    structural_violations: list[str] = []
    member_target_mismatch_count = 0
    terminal_live_member_count = 0
    vintage_switch_live_member_count = 0
    held_member_link_mismatch_count = 0
    held_member_link_maximum_error = 0.0
    direction_flip_count = 0
    mixed_direction_episode_count = 0
    trade_episode_path_factor_error = 0.0
    trade_episode_component_sum_error = 0.0
    reversal_split_turnover_error = 0.0
    trade_episode_path_factor_mismatch_count = 0
    trade_episode_component_mismatch_count = 0

    for pair, source_group in daily.groupby("pair", sort=True):
        group = source_group.copy().reset_index(drop=False).rename(columns={"index": "source_row"})
        ticker_a = str(group["ticker_a"].iloc[0])
        ticker_b = str(group["ticker_b"].iloc[0])
        dates = pd.DatetimeIndex(group["date"])
        expected_dates = prices.index[(prices.index >= dates[0]) & (prices.index <= dates[-1])]
        if not dates.equals(expected_dates):
            raise ValueError(f"daily dates do not form the complete raw-price calendar for {pair}")

        raw_a = pd.to_numeric(prices.loc[dates, ticker_a], errors="coerce").to_numpy(dtype=float)
        raw_b = pd.to_numeric(prices.loc[dates, ticker_b], errors="coerce").to_numpy(dtype=float)
        return_a = return_cache[ticker_a].reindex(dates).to_numpy(dtype=float)
        return_b = return_cache[ticker_b].reindex(dates).to_numpy(dtype=float)
        held_a = group["weight_a"].to_numpy(dtype=float)
        held_b = group["weight_b"].to_numpy(dtype=float)
        target_a = group["target_weight_a"].to_numpy(dtype=float)
        target_b = group["target_weight_b"].to_numpy(dtype=float)
        active_member_count = group["ensemble_active_member_count"].to_numpy(
            dtype=float
        )
        held_member_count = group["ensemble_held_member_count"].to_numpy(
            dtype=float
        )
        expected_held_member_count = np.r_[0.0, active_member_count[:-1]]
        held_member_link_error = np.abs(
            held_member_count - expected_held_member_count
        )
        held_member_link_mismatch_count += int(
            (held_member_link_error > tolerance).sum()
        )
        held_member_link_maximum_error = max(
            held_member_link_maximum_error,
            float(held_member_link_error.max()),
        )
        if (held_member_link_error > tolerance).any():
            structural_violations.append(
                f"{pair}: {int((held_member_link_error > tolerance).sum())} held member counts do not equal prior-close active member counts"
            )
        aggregate_target_active = (np.abs(target_a) + np.abs(target_b)) > tolerance
        target_without_member = aggregate_target_active & (active_member_count <= 0.0)
        member_target_mismatch_count += int(target_without_member.sum())
        if target_without_member.any():
            structural_violations.append(
                f"{pair}: {int(target_without_member.sum())} active targets have no active ensemble member"
            )
        if active_member_count[-1] > 0.0:
            terminal_live_member_count += 1
            structural_violations.append(
                f"{pair}: terminal row retains active ensemble members"
            )
        vintage = group["vintage_rebalance_date"]
        same_vintage = vintage.eq(vintage.shift()) | (
            vintage.isna() & vintage.shift().isna()
        )
        vintage_change = (~same_vintage).to_numpy(dtype=bool, copy=True)
        vintage_change[0] = False
        live_before_switch = vintage_change & np.r_[
            False, active_member_count[:-1] > 0.0
        ]
        vintage_switch_live_member_count += int(live_before_switch.sum())
        if live_before_switch.any():
            structural_violations.append(
                f"{pair}: {int(live_before_switch.sum())} vintage changes before all ensemble members are flat"
            )

        # Missing closes are allowed only while the affected leg is exactly cash.
        required_a = (np.abs(held_a) > tolerance) | (np.abs(target_a) > tolerance)
        required_b = (np.abs(held_b) > tolerance) | (np.abs(target_b) > tolerance)
        invalid_a = required_a & (~np.isfinite(raw_a) | (raw_a <= 0.0))
        invalid_b = required_b & (~np.isfinite(raw_b) | (raw_b <= 0.0))
        invalid_return_a = (np.abs(held_a) > tolerance) & ~np.isfinite(return_a)
        invalid_return_b = (np.abs(held_b) > tolerance) & ~np.isfinite(return_b)
        if invalid_a.any() or invalid_b.any() or invalid_return_a.any() or invalid_return_b.any():
            raise ValueError(f"active or fill rows require finite positive raw closes for both legs: {pair}")
        return_a = np.where(np.isfinite(return_a), return_a, 0.0)
        return_b = np.where(np.isfinite(return_b), return_b, 0.0)

        expected_held_a = np.r_[0.0, target_a[:-1]]
        expected_held_b = np.r_[0.0, target_b[:-1]]
        held_link_error = np.maximum(
            np.abs(held_a - expected_held_a), np.abs(held_b - expected_held_b)
        )
        if (held_link_error > tolerance).any():
            structural_violations.append(
                f"{pair}: {int((held_link_error > tolerance).sum())} held weights do not equal prior-close targets"
            )

        gross = held_a * return_a + held_b * return_b
        denominator = 1.0 + gross
        if (~np.isfinite(denominator) | (denominator <= tolerance)).any():
            raise ValueError(f"non-positive post-return NAV encountered for {pair}")
        drifted_a = held_a * (1.0 + return_a) / denominator
        drifted_b = held_b * (1.0 + return_b) / denominator
        turnover = np.abs(target_a - drifted_a) + np.abs(target_b - drifted_b)
        transaction_primary = (
            turnover * float(transaction_cost_bps) / 10_000.0 * denominator
        )
        transaction_1bp = turnover / 10_000.0 * denominator
        transaction_2bp = turnover * 2.0 / 10_000.0 * denominator
        short_notional = np.maximum(-held_a, 0.0) + np.maximum(-held_b, 0.0)
        borrow = short_notional * float(annual_short_borrow_bps) / 10_000.0 / 252.0
        net_1bp = gross - transaction_1bp - borrow
        net_2bp = gross - transaction_2bp - borrow

        reconstructed = pd.DataFrame(
            {
                "source_row": group["source_row"],
                "date": dates,
                "pair": str(pair),
                "strategy_id": group["strategy_id"].astype(str),
                "vintage_rebalance_date": group["vintage_rebalance_date"],
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "raw_price_a": raw_a,
                "raw_price_b": raw_b,
                "recorded_price_a": pd.to_numeric(group["price_a"], errors="coerce"),
                "recorded_price_b": pd.to_numeric(group["price_b"], errors="coerce"),
                "raw_return_a": return_a,
                "raw_return_b": return_b,
                "held_weight_a": held_a,
                "held_weight_b": held_b,
                "post_fill_target_weight_a": target_a,
                "post_fill_target_weight_b": target_b,
                "ensemble_active_member_count": active_member_count,
                "ensemble_held_member_count": held_member_count,
                "expected_held_member_count": expected_held_member_count,
                "held_member_link_error": held_member_link_error,
                "expected_held_weight_a": expected_held_a,
                "expected_held_weight_b": expected_held_b,
                "held_weight_link_error": held_link_error,
                "drifted_weight_a": drifted_a,
                "drifted_weight_b": drifted_b,
                "recorded_gross_return": group["gross_return"],
                "reconstructed_gross_return": gross,
                "recorded_turnover": group["turnover"],
                "reconstructed_turnover": turnover,
                "recorded_transaction_cost": group["transaction_cost"],
                "reconstructed_transaction_cost": transaction_primary,
                "recorded_borrow_cost": group["borrow_cost"],
                "reconstructed_borrow_cost": borrow,
                "recorded_net_return_1bp": group["net_return_1bp"],
                "reconstructed_net_return_1bp": net_1bp,
                "recorded_net_return_2bp": group["net_return_2bp"],
                "reconstructed_net_return_2bp": net_2bp,
                "trade_number": pd.Series(pd.NA, index=group.index, dtype="Int64"),
                "selected_categories": "",
            }
        )

        for label, recorded_column, reconstructed_column, allow_nan in [
            ("price_a", "recorded_price_a", "raw_price_a", True),
            ("price_b", "recorded_price_b", "raw_price_b", True),
            ("gross_return", "recorded_gross_return", "reconstructed_gross_return", False),
            ("turnover", "recorded_turnover", "reconstructed_turnover", False),
            (
                "transaction_cost",
                "recorded_transaction_cost",
                "reconstructed_transaction_cost",
                False,
            ),
            ("borrow_cost", "recorded_borrow_cost", "reconstructed_borrow_cost", False),
            ("net_return_1bp", "recorded_net_return_1bp", "reconstructed_net_return_1bp", False),
            ("net_return_2bp", "recorded_net_return_2bp", "reconstructed_net_return_2bp", False),
        ]:
            errors, matches, _, _ = _numeric_comparison(
                reconstructed[recorded_column],
                reconstructed[reconstructed_column],
                tolerance=tolerance,
                allow_paired_nan=allow_nan,
            )
            reconstructed[f"{label}_absolute_error"] = errors
            reconstructed[f"{label}_matches"] = matches

        (
            pair_trades,
            held_trade_number,
            post_fill_trade_number,
            direction_flips,
            episode_stats,
            episode_violations,
        ) = _directional_episode_reconstruction(
            pair=str(pair),
            group=group,
            raw_a=raw_a,
            raw_b=raw_b,
            held_a=held_a,
            held_b=held_b,
            target_a=target_a,
            target_b=target_b,
            drifted_a=drifted_a,
            drifted_b=drifted_b,
            gross=gross,
            turnover=turnover,
            borrow=borrow,
            net_1bp=net_1bp,
            net_2bp=net_2bp,
            transaction_1bp=transaction_1bp,
            transaction_2bp=transaction_2bp,
            tolerance=tolerance,
        )
        derived_trades.extend(pair_trades)
        structural_violations.extend(episode_violations)
        direction_flip_count += int(episode_stats["direction_flip_count"])
        mixed_direction_episode_count += int(
            episode_stats["mixed_direction_episode_count"]
        )
        pair_path_error = float(episode_stats["path_factor_error"])
        pair_component_error = float(episode_stats["component_sum_error"])
        trade_episode_path_factor_error = max(
            trade_episode_path_factor_error, pair_path_error
        )
        trade_episode_component_sum_error = max(
            trade_episode_component_sum_error, pair_component_error
        )
        reversal_split_turnover_error = max(
            reversal_split_turnover_error,
            float(episode_stats["split_turnover_error"]),
        )
        trade_episode_path_factor_mismatch_count += int(
            pair_path_error > tolerance
        )
        trade_episode_component_mismatch_count += int(
            pair_component_error > tolerance
        )
        held_values = [
            pd.NA if value == 0 else int(value) for value in held_trade_number
        ]
        post_fill_values = [
            pd.NA if value == 0 else int(value)
            for value in post_fill_trade_number
        ]
        reconstructed["held_trade_number"] = pd.array(
            held_values, dtype="Int64"
        )
        reconstructed["post_fill_trade_number"] = pd.array(
            post_fill_values, dtype="Int64"
        )
        reconstructed["trade_number"] = reconstructed[
            "post_fill_trade_number"
        ].fillna(reconstructed["held_trade_number"])
        reconstructed["direction_flip"] = direction_flips
        detail_parts.append(reconstructed)

    detail = pd.concat(detail_parts, ignore_index=True, sort=False)
    derived = pd.DataFrame(derived_trades)
    expected_keys = {
        (str(row.pair), int(row.trade_number)) for row in derived.itertuples(index=False)
    }
    recorded_keys = {
        (str(row.pair), int(row.trade_number)) for row in trades.itertuples(index=False)
    }
    missing_ledger_keys = sorted(expected_keys.difference(recorded_keys))
    extra_ledger_keys = sorted(recorded_keys.difference(expected_keys))

    trade_numeric_checks: dict[str, dict[str, Any]] = {}
    trade_categorical_mismatches: dict[str, int] = {}
    if not missing_ledger_keys and not extra_ledger_keys and len(derived):
        ledger = trades.copy()
        for column in ["entry_decision_date", "entry_date", "exit_decision_date", "exit_date"]:
            ledger[column] = pd.to_datetime(ledger[column], errors="coerce")
        joined = derived.merge(
            ledger,
            on=["pair", "trade_number"],
            suffixes=("_reconstructed", "_recorded"),
            validate="one_to_one",
        )
        for column in [
            "entry_fill_price_a",
            "entry_fill_price_b",
            "exit_fill_price_a",
            "exit_fill_price_b",
            "entry_zscore",
            "exit_zscore",
            "net_return_1bp",
            "net_return_2bp",
            "gross_return_sum",
            "transaction_cost_1bp",
            "transaction_cost_2bp",
            "opening_transaction_cost_1bp",
            "closing_transaction_cost_1bp",
            "holding_rebalance_transaction_cost_1bp",
            "opening_transaction_cost_2bp",
            "closing_transaction_cost_2bp",
            "holding_rebalance_transaction_cost_2bp",
            "borrow_cost",
        ]:
            _, _, mismatch_count, maximum = _numeric_comparison(
                joined[f"{column}_recorded"],
                joined[f"{column}_reconstructed"],
                tolerance=tolerance,
                allow_paired_nan=column.startswith("exit_") and column not in {"exit_date"},
            )
            trade_numeric_checks[column] = {
                "mismatch_count": mismatch_count,
                "maximum_absolute_error": maximum,
            }
        sessions_mismatch = (
            pd.to_numeric(joined["trading_sessions_recorded"], errors="coerce")
            != pd.to_numeric(joined["trading_sessions_reconstructed"], errors="coerce")
        )
        trade_categorical_mismatches["trading_sessions"] = int(sessions_mismatch.sum())
        for column in [
            "strategy_id",
            "direction_ticker_a",
            "exit_reason",
            "entry_fill_event",
            "exit_fill_event",
            "entry_transition",
            "exit_transition",
            "attribution_convention",
        ]:
            mismatch = (
                joined[f"{column}_recorded"].astype(str)
                != joined[f"{column}_reconstructed"].astype(str)
            )
            trade_categorical_mismatches[column] = int(mismatch.sum())
        for column in ["entry_decision_date", "entry_date", "exit_decision_date", "exit_date"]:
            mismatch = [
                not _date_matches(recorded, reconstructed)
                for recorded, reconstructed in zip(
                    joined[f"{column}_recorded"],
                    joined[f"{column}_reconstructed"],
                )
            ]
            trade_categorical_mismatches[column] = int(sum(mismatch))
        closed_mismatch = [
            _as_bool(recorded) != bool(reconstructed)
            for recorded, reconstructed in zip(
                joined["closed_recorded"], joined["closed_reconstructed"]
            )
        ]
        trade_categorical_mismatches["closed"] = int(sum(closed_mismatch))
    elif not len(derived) and len(trades) == 0:
        joined = pd.DataFrame()
    else:
        joined = pd.DataFrame()

    daily_checks: dict[str, dict[str, Any]] = {}
    for label in [
        "price_a",
        "price_b",
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "net_return_1bp",
        "net_return_2bp",
    ]:
        matches = detail[f"{label}_matches"].astype(bool)
        errors = pd.to_numeric(detail[f"{label}_absolute_error"], errors="coerce")
        finite_errors = errors[np.isfinite(errors)]
        daily_checks[label] = {
            "mismatch_count": int((~matches).sum()),
            "maximum_absolute_error": (
                float(finite_errors.max()) if len(finite_errors) else (0.0 if matches.all() else math.inf)
            ),
        }
    daily_checks["held_weight_state_link"] = {
        "mismatch_count": int((detail["held_weight_link_error"] > tolerance).sum()),
        "maximum_absolute_error": float(detail["held_weight_link_error"].max()),
    }
    daily_checks["held_member_state_link"] = {
        "mismatch_count": held_member_link_mismatch_count,
        "maximum_absolute_error": held_member_link_maximum_error,
    }
    daily_checks["aggregate_target_has_active_member"] = {
        "mismatch_count": member_target_mismatch_count,
        "maximum_absolute_error": float(member_target_mismatch_count > 0),
    }
    daily_checks["terminal_members_flat"] = {
        "mismatch_count": terminal_live_member_count,
        "maximum_absolute_error": float(terminal_live_member_count > 0),
    }
    daily_checks["vintage_switch_members_flat"] = {
        "mismatch_count": vintage_switch_live_member_count,
        "maximum_absolute_error": float(vintage_switch_live_member_count > 0),
    }
    daily_checks["trade_episode_path_factor"] = {
        "mismatch_count": trade_episode_path_factor_mismatch_count,
        "maximum_absolute_error": trade_episode_path_factor_error,
    }
    daily_checks["trade_episode_component_sums"] = {
        "mismatch_count": trade_episode_component_mismatch_count,
        "maximum_absolute_error": trade_episode_component_sum_error,
    }
    daily_checks["reversal_split_turnover"] = {
        "mismatch_count": int(reversal_split_turnover_error > tolerance),
        "maximum_absolute_error": reversal_split_turnover_error,
    }

    selected: list[dict[str, Any]] = []
    missing_required_categories: list[str] = []
    selections: dict[str, pd.DataFrame] = {
        "winning": derived.loc[derived.get("net_return_2bp", pd.Series(dtype=float)) > tolerance],
        "losing": derived.loc[derived.get("net_return_2bp", pd.Series(dtype=float)) < -tolerance],
        "long_a": derived.loc[derived.get("direction_ticker_a", pd.Series(dtype=str)).eq("long")],
        "short_a": derived.loc[derived.get("direction_ticker_a", pd.Series(dtype=str)).eq("short")],
    }
    strategy_series = derived.get("strategy_id", pd.Series(dtype=str)).astype(str)
    optional = {
        "dynamic_s4": derived.loc[strategy_series.str.startswith("S4")],
        "ensemble_s5": derived.loc[strategy_series.str.startswith("S5")],
    }
    selections.update({key: value for key, value in optional.items() if len(value)})
    for category in ["winning", "losing", "long_a", "short_a", "dynamic_s4", "ensemble_s5"]:
        if category not in selections:
            continue
        candidates = selections[category]
        if candidates.empty:
            if category in {"winning", "losing", "long_a", "short_a"}:
                missing_required_categories.append(category)
            continue
        if category == "winning":
            choice = candidates.sort_values(
                ["net_return_2bp", "pair", "trade_number"],
                ascending=[False, True, True],
                kind="mergesort",
            ).iloc[0]
        elif category == "losing":
            choice = candidates.sort_values(
                ["net_return_2bp", "pair", "trade_number"],
                ascending=[True, True, True],
                kind="mergesort",
            ).iloc[0]
        else:
            choice = candidates.sort_values(["pair", "trade_number"], kind="mergesort").iloc[0]
        selected.append(
            {
                "category": category,
                "pair": str(choice["pair"]),
                "trade_number": int(choice["trade_number"]),
                "strategy_id": str(choice["strategy_id"]),
                "direction_ticker_a": str(choice["direction_ticker_a"]),
                "reconstructed_net_return_2bp": float(choice["net_return_2bp"]),
                "entry_date": pd.Timestamp(choice["entry_date"]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(choice["exit_date"]).strftime("%Y-%m-%d"),
            }
        )
    category_by_trade: dict[tuple[str, int], list[str]] = {}
    for row in selected:
        category_by_trade.setdefault((row["pair"], row["trade_number"]), []).append(row["category"])
    selected_categories: list[str] = []
    for pair, held_number, post_fill_number in zip(
        detail["pair"],
        detail["held_trade_number"],
        detail["post_fill_trade_number"],
    ):
        categories: set[str] = set()
        for number in (held_number, post_fill_number):
            if not pd.isna(number):
                categories.update(
                    category_by_trade.get((str(pair), int(number)), [])
                )
        selected_categories.append("|".join(sorted(categories)))
    detail["selected_categories"] = selected_categories

    daily_mismatch_total = sum(value["mismatch_count"] for value in daily_checks.values())
    trade_numeric_mismatch_total = sum(
        value["mismatch_count"] for value in trade_numeric_checks.values()
    )
    trade_categorical_mismatch_total = sum(trade_categorical_mismatches.values())
    passed = bool(
        daily_mismatch_total == 0
        and trade_numeric_mismatch_total == 0
        and trade_categorical_mismatch_total == 0
        and not missing_ledger_keys
        and not extra_ledger_keys
        and not structural_violations
        and not missing_required_categories
    )
    summary: dict[str, Any] = {
        "audit": "independent_v8_raw_close_weight_reconstruction",
        "generated_at_utc": _utc_now(),
        "passed": passed,
        "execution_convention": "signal_close_t_fill_close_t_plus_1_first_pnl_t_plus_1_to_t_plus_2",
        "accounting_convention": "post_return_nav_drift_turnover_and_cost",
        "transaction_cost_bps": float(transaction_cost_bps),
        "annual_short_borrow_bps": float(annual_short_borrow_bps),
        "tolerance": float(tolerance),
        "pair_count": int(daily["pair"].nunique()),
        "daily_row_count": int(len(daily)),
        "reconstructed_trade_count": int(len(derived)),
        "recorded_trade_count": int(len(trades)),
        "direction_flip_count": direction_flip_count,
        "mixed_direction_episode_count": mixed_direction_episode_count,
        "trade_episode_path_factor_error": trade_episode_path_factor_error,
        "trade_episode_component_sum_error": trade_episode_component_sum_error,
        "reversal_split_turnover_error": reversal_split_turnover_error,
        "daily_checks": daily_checks,
        "trade_numeric_checks": trade_numeric_checks,
        "trade_categorical_mismatches": trade_categorical_mismatches,
        "missing_ledger_keys": [list(value) for value in missing_ledger_keys],
        "extra_ledger_keys": [list(value) for value in extra_ledger_keys],
        "structural_violations": structural_violations,
        "missing_required_categories": missing_required_categories,
        "selected_trades": selected,
        "selected_category_count": len(selected),
        "dynamic_s4_present": bool(strategy_series.str.startswith("S4").any()),
        "ensemble_s5_present": bool(strategy_series.str.startswith("S5").any()),
        "independence": {
            "imports_backtest_engine": False,
            "imports_v7_v8_simulator": False,
            "imports_repricing_helper": False,
            "raw_prices_used": True,
            "recorded_returns_used_for_reconstruction": False,
        },
    }
    return detail.drop(columns=["source_row"]), summary


def _resolve_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def audit_files(
    config_path: Path,
    *,
    daily_path: Path | None = None,
    trades_path: Path | None = None,
    prices_path: Path | None = None,
    detail_path: Path | None = None,
    summary_path: Path | None = None,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    config_file = _resolve_from_project(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    output_root = _resolve_from_project(Path(config["outputs"]["root"]))
    processed = output_root / "data/processed"
    identity_root = _resolve_from_project(Path(config["inputs"]["identity_root"]))
    resolved_daily = _resolve_from_project(daily_path) if daily_path else processed / DEFAULT_DAILY_NAME
    resolved_trades = _resolve_from_project(trades_path) if trades_path else processed / DEFAULT_TRADES_NAME
    resolved_prices = (
        _resolve_from_project(prices_path)
        if prices_path
        else identity_root / config["inputs"]["price_snapshot"]
    )
    resolved_detail = _resolve_from_project(detail_path) if detail_path else processed / DEFAULT_DETAIL_NAME
    resolved_summary = _resolve_from_project(summary_path) if summary_path else processed / DEFAULT_SUMMARY_NAME
    for label, path in {
        "config": config_file,
        "daily": resolved_daily,
        "trades": resolved_trades,
        "prices": resolved_prices,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} input: {path}")

    daily = pd.read_csv(resolved_daily)
    trades = pd.read_csv(resolved_trades)
    prices = pd.read_csv(resolved_prices, index_col="date", parse_dates=True)
    detail, summary = audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=float(config["evaluation"]["primary_transaction_cost_bps"]),
        annual_short_borrow_bps=float(config["evaluation"]["annual_short_borrow_bps"]),
        tolerance=tolerance,
    )
    summary["input_files"] = {
        "config": {"path": str(config_file), "sha256": _sha256(config_file)},
        "daily": {"path": str(resolved_daily), "sha256": _sha256(resolved_daily)},
        "trades": {"path": str(resolved_trades), "sha256": _sha256(resolved_trades)},
        "prices": {"path": str(resolved_prices), "sha256": _sha256(resolved_prices)},
    }
    summary["output_files"] = {
        "detail": str(resolved_detail),
        "summary": str(resolved_summary),
    }
    _atomic_csv(detail, resolved_detail)
    _atomic_json(summary, resolved_summary)
    return detail, summary, resolved_detail, resolved_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--daily", type=Path)
    parser.add_argument("--trades", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--detail-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _, summary, detail_path, summary_path = audit_files(
        args.config,
        daily_path=args.daily,
        trades_path=args.trades,
        prices_path=args.prices,
        detail_path=args.detail_output,
        summary_path=args.summary_output,
        tolerance=args.tolerance,
    )
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "detail": str(detail_path),
                "summary": str(summary_path),
                "daily_rows": summary["daily_row_count"],
                "trades": summary["reconstructed_trade_count"],
                "selected_categories": [
                    row["category"] for row in summary["selected_trades"]
                ],
            },
            indent=2,
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
