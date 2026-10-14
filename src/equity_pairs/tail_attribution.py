"""Tail counterfactuals and exact pair-return source decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_pair_returns(
    daily: pd.DataFrame,
    *,
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Decompose net returns into relative, common, trading-cost, and borrow legs.

    With ``m=(r_a+r_b)/2`` and ``d=(r_a-r_b)/2``, the identity
    ``w_a*r_a+w_b*r_b=(w_a+w_b)*m+(w_a-w_b)*d`` is exact.  Wealth-scaled
    dollar contributions also add exactly to terminal return.
    """

    required = {
        "date",
        "price_a",
        "price_b",
        "weight_a",
        "weight_b",
        "turnover",
        "borrow_cost",
        f"net_return_{int(transaction_cost_bps)}bp",
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"missing attribution columns: {missing}")
    work = daily.sort_values("date").copy()
    work["return_a"] = pd.to_numeric(work["price_a"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    work["return_b"] = pd.to_numeric(work["price_b"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
    weight_a = pd.to_numeric(work["weight_a"], errors="coerce").fillna(0.0)
    weight_b = pd.to_numeric(work["weight_b"], errors="coerce").fillna(0.0)
    common_return = (weight_a + weight_b) * (work["return_a"] + work["return_b"]) / 2.0
    relative_return = (weight_a - weight_b) * (work["return_a"] - work["return_b"]) / 2.0
    gross_return = relative_return + common_return
    cost_return = (
        -pd.to_numeric(work["turnover"], errors="coerce").fillna(0.0)
        * float(transaction_cost_bps)
        / 10_000.0
        * (1.0 + gross_return)
    )
    borrow_return = -pd.to_numeric(work["borrow_cost"], errors="coerce").fillna(0.0)
    net_column = f"net_return_{int(transaction_cost_bps)}bp"
    net_return = pd.to_numeric(work[net_column], errors="coerce").fillna(0.0)
    work["relative_return_component"] = relative_return
    work["common_return_component"] = common_return
    work["transaction_cost_component"] = cost_return
    work["borrow_cost_component"] = borrow_return
    work["component_sum"] = relative_return + common_return + cost_return + borrow_return
    work["component_error"] = work["component_sum"] - net_return
    work["starting_equity"] = (1.0 + net_return).cumprod().shift(1, fill_value=1.0)
    dollar_columns = {
        "relative_return_component": "relative_dollar_contribution",
        "common_return_component": "common_dollar_contribution",
        "transaction_cost_component": "transaction_cost_dollar_contribution",
        "borrow_cost_component": "borrow_cost_dollar_contribution",
    }
    for source, target in dollar_columns.items():
        work[target] = work["starting_equity"] * work[source]
    work["net_dollar_pnl"] = work["starting_equity"] * net_return
    work["held_position"] = (weight_a.abs() + weight_b.abs()).gt(1e-12)
    work["maximum_absolute_leg_return"] = np.maximum(work["return_a"].abs(), work["return_b"].abs())
    return work


def remove_interval_as_cash(
    daily_returns: pd.Series,
    dates: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Replace one inclusive trade interval with zero-return cash."""

    output = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0).copy()
    timestamps = pd.to_datetime(dates)
    output.loc[timestamps.between(pd.Timestamp(start), pd.Timestamp(end))] = 0.0
    return output


__all__ = ["decompose_pair_returns", "remove_interval_as_cash"]
