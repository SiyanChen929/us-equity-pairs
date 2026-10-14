"""Causal strategy-family selection for the V8 pair-centric study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def select_strategy_one_standard_error(
    history: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    *,
    current_rebalance: pd.Timestamp,
    lookback_calendar_years: int = 3,
    minimum_history_quarters: int = 8,
    minimum_active_quarters: int = 3,
    minimum_trades: int = 10,
    default_strategy_id: str = "S0_v6",
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Select one strategy using only candidate outcomes known before rebalance.

    Candidate returns are true earlier outer-test observations.  The current
    quarter is excluded and the left edge is exactly three calendar years
    before ``current_rebalance``.  The raw best mean quarterly log return sets
    a one-standard-error cutoff; stability and simplicity break ties.
    """

    current = pd.Timestamp(current_rebalance)
    start = current - pd.DateOffset(years=int(lookback_calendar_years))
    frame = history.copy()
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "strategy_id",
                "rebalance_date",
                "q_return_2bp",
                "selection_activity_count",
                "selection_activity_count_definition",
                "trade_count",
                "annualized_turnover",
                "max_drawdown",
            ]
        )
    frame["rebalance_date"] = pd.to_datetime(frame["rebalance_date"])
    frame = frame.loc[
        frame["rebalance_date"].ge(start)
        & frame["rebalance_date"].lt(current)
    ].copy()
    complexity = {
        str(row["id"]): int(row.get("complexity", 999)) for row in candidates
    }
    expected = list(complexity)
    rows: list[dict[str, Any]] = []
    for strategy_id in expected:
        subset = frame.loc[frame["strategy_id"].astype(str).eq(strategy_id)].copy()
        returns = pd.to_numeric(subset.get("q_return_2bp"), errors="coerce").dropna()
        safe = returns.loc[returns.gt(-1.0)]
        logs = np.log1p(safe.to_numpy(dtype=float))
        observations = int(len(logs))
        standard_deviation = (
            float(np.std(logs, ddof=1)) if observations > 1 else math.nan
        )
        standard_error = (
            standard_deviation / math.sqrt(observations)
            if observations > 1 and np.isfinite(standard_deviation)
            else math.nan
        )
        if "selection_activity_count" in subset:
            activity_values = pd.to_numeric(
                subset["selection_activity_count"], errors="coerce"
            ).fillna(0)
            definitions = set(
                subset["selection_activity_count_definition"]
                .dropna()
                .astype(str)
            ) if "selection_activity_count_definition" in subset else set()
            if definitions and definitions != {"continuous_active_target_runs"}:
                raise ValueError(
                    "selection activity count has an unsupported definition"
                )
            activity_definition = "continuous_active_target_runs"
        else:
            # Backward-compatible read of a pre-clarification candidate cache.
            # Its calculation was already the same flat-to-active run count.
            activity_values = pd.to_numeric(
                subset.get("trade_count"), errors="coerce"
            ).fillna(0)
            activity_definition = "continuous_active_target_runs"
        trades = int(activity_values.sum())
        active = int(
            activity_values.gt(0).sum()
        )
        eligible = bool(
            observations >= int(minimum_history_quarters)
            and active >= int(minimum_active_quarters)
            and trades >= int(minimum_trades)
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "complexity": complexity[strategy_id],
                "lookback_start": start,
                "lookback_end_exclusive": current,
                "history_quarters": observations,
                "active_quarters": active,
                "selection_activity_count": trades,
                "selection_activity_count_definition": activity_definition,
                # Compatibility alias for existing selection reports.  This is
                # not the final directional-episode ledger count.
                "trade_count": trades,
                "mean_quarterly_log_return": float(np.mean(logs))
                if observations
                else math.nan,
                "quarterly_log_return_std": standard_deviation,
                "standard_error": standard_error,
                "mean_annualized_turnover": float(
                    pd.to_numeric(
                        subset.get("annualized_turnover"), errors="coerce"
                    ).mean()
                ),
                "worst_quarter_max_drawdown": float(
                    pd.to_numeric(subset.get("max_drawdown"), errors="coerce").min()
                ),
                "eligible": eligible,
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[
        ranking["eligible"]
        & np.isfinite(ranking["mean_quarterly_log_return"])
    ].copy()
    if eligible.empty:
        ranking["within_one_standard_error"] = False
        ranking["selected"] = ranking["strategy_id"].eq(default_strategy_id)
        return default_strategy_id, ranking, {
            "selected_strategy_id": default_strategy_id,
            "selection_reason": "insufficient_causal_strategy_history",
            "raw_best_strategy_id": pd.NA,
            "one_standard_error_cutoff": math.nan,
            "candidate_count": len(ranking),
            "eligible_candidate_count": 0,
            "selection_activity_count_definition": (
                "continuous_active_target_runs"
            ),
        }
    raw = eligible.sort_values(
        ["mean_quarterly_log_return", "strategy_id"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    cutoff = float(raw["mean_quarterly_log_return"])
    if np.isfinite(float(raw["standard_error"])):
        cutoff -= float(raw["standard_error"])
    within = eligible.loc[
        eligible["mean_quarterly_log_return"].ge(cutoff)
    ].copy()
    within = within.sort_values(
        [
            "quarterly_log_return_std",
            "mean_annualized_turnover",
            "worst_quarter_max_drawdown",
            "complexity",
            "strategy_id",
        ],
        ascending=[True, True, False, True, True],
        kind="mergesort",
        na_position="last",
    )
    selected = str(within.iloc[0]["strategy_id"])
    ranking["within_one_standard_error"] = (
        ranking["eligible"]
        & ranking["mean_quarterly_log_return"].ge(cutoff)
    )
    ranking["selected"] = ranking["strategy_id"].eq(selected)
    return selected, ranking, {
        "selected_strategy_id": selected,
        "selection_reason": "causal_three_year_one_standard_error",
        "raw_best_strategy_id": str(raw["strategy_id"]),
        "one_standard_error_cutoff": cutoff,
        "candidate_count": len(ranking),
        "eligible_candidate_count": len(eligible),
        "selection_activity_count_definition": (
            "continuous_active_target_runs"
        ),
    }


def reprice_returns(
    daily: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.Series:
    """Reprice a frozen path without changing its signals or positions."""

    gross = pd.to_numeric(daily["gross_return"], errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(daily["turnover"], errors="coerce").fillna(0.0)
    borrow = pd.to_numeric(daily["borrow_cost"], errors="coerce").fillna(0.0)
    return (
        gross
        - turnover * float(transaction_cost_bps) / 10_000.0 * (1.0 + gross)
        - borrow
    )


__all__ = ["reprice_returns", "select_strategy_one_standard_error"]
