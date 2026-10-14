from __future__ import annotations

import numpy as np
import pandas as pd

from equity_pairs.v8_pair_selection import (
    reprice_returns,
    select_strategy_one_standard_error,
)


_CANDIDATES = [
    {"id": "S0_v6", "complexity": 0},
    {"id": "S1_stop_gate", "complexity": 1},
]


def test_strategy_selection_excludes_current_and_old_observations() -> None:
    rows = []
    for date in pd.date_range("2020-01-01", periods=13, freq="QS"):
        for strategy, value in (("S0_v6", 0.01), ("S1_stop_gate", 0.02)):
            rows.append(
                {
                    "strategy_id": strategy,
                    "rebalance_date": date,
                    "q_return_2bp": value,
                    # The frozen causal gate counts continuous active runs,
                    # not final directional reporting episodes.
                    "selection_activity_count": 2,
                    "selection_activity_count_definition": (
                        "continuous_active_target_runs"
                    ),
                    "trade_count": 0,
                    "annualized_turnover": 1.0,
                    "max_drawdown": -0.02,
                }
            )
    history = pd.DataFrame(rows)
    selected, ranking, summary = select_strategy_one_standard_error(
        history,
        _CANDIDATES,
        current_rebalance=pd.Timestamp("2023-01-01"),
        minimum_history_quarters=8,
        minimum_active_quarters=3,
        minimum_trades=10,
    )
    assert selected in {"S0_v6", "S1_stop_gate"}
    assert ranking["lookback_start"].eq(pd.Timestamp("2020-01-01")).all()
    assert ranking["lookback_end_exclusive"].eq(pd.Timestamp("2023-01-01")).all()
    assert ranking["history_quarters"].eq(12).all()
    assert ranking["selection_activity_count"].eq(24).all()
    assert ranking["selection_activity_count_definition"].eq(
        "continuous_active_target_runs"
    ).all()
    assert ranking["eligible"].all()
    assert summary["selection_activity_count_definition"] == (
        "continuous_active_target_runs"
    )


def test_strategy_selection_defaults_when_history_is_sparse() -> None:
    selected, ranking, summary = select_strategy_one_standard_error(
        pd.DataFrame(),
        _CANDIDATES,
        current_rebalance=pd.Timestamp("2020-01-01"),
    )
    assert selected == "S0_v6"
    assert summary["selection_reason"] == "insufficient_causal_strategy_history"
    assert ranking.loc[ranking["selected"], "strategy_id"].tolist() == ["S0_v6"]


def test_reprice_returns_uses_frozen_turnover_and_borrow() -> None:
    daily = pd.DataFrame(
        {
            "gross_return": [0.01, -0.02],
            "turnover": [1.0, 0.5],
            "borrow_cost": [0.00001, 0.00002],
        }
    )
    one = reprice_returns(daily, 1.0)
    two = reprice_returns(daily, 2.0)
    assert np.allclose(
        one - two,
        daily["turnover"] * (1.0 + daily["gross_return"]) / 10_000.0,
    )
