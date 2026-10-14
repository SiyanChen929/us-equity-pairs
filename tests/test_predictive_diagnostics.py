from __future__ import annotations

import numpy as np
import pandas as pd

from equity_pairs.predictive_diagnostics import (
    aggregate_unique_pairs,
    circular_block_bootstrap_mean,
    quarterly_information_coefficients,
    selection_mechanism_by_quarter,
    summarize_conditional_returns,
)


def test_quarterly_information_coefficient_is_cross_sectional() -> None:
    frame = pd.DataFrame(
        {
            "rebalance_date": ["2024-01-02"] * 4 + ["2024-04-01"],
            "training_score": [1.0, 2.0, 3.0, 4.0, 9.0],
            "net_total_return": [0.1, 0.2, 0.4, 0.3, -0.2],
            "trade_count": [1, 2, 3, 4, 1],
        }
    )
    result = quarterly_information_coefficients(frame, scope="test")
    first = result.iloc[0]
    second = result.iloc[1]
    assert first["pair_vintage_count"] == 4
    assert np.isclose(first["spearman_ic"], 0.8)
    assert second["pair_vintage_count"] == 1
    assert np.isnan(second["spearman_ic"])


def test_circular_block_bootstrap_is_seeded() -> None:
    first = circular_block_bootstrap_mean(
        [0.1, -0.2, 0.3, 0.4], repetitions=100, block_length=2, seed=7
    )
    second = circular_block_bootstrap_mean(
        [0.1, -0.2, 0.3, 0.4], repetitions=100, block_length=2, seed=7
    )
    assert np.array_equal(first, second)
    assert len(first) == 100


def test_selection_mechanism_preserves_cash_only_quarters() -> None:
    vintages = pd.DataFrame(
        {
            "rebalance_date": ["2024-04-01", "2024-04-01", "2024-04-01"],
            "pair": ["A__B", "C__D", "E__F"],
            "training_score": [3.0, 2.0, 1.0],
            "net_total_return": [0.03, 0.01, -0.02],
            "selected_top10": [True, True, False],
            "trade_count": [2, 2, 1],
        }
    )
    schedule = pd.DataFrame(
        {
            "rebalance_date": ["2024-01-02", "2024-04-01"],
            "outer_end_exclusive": ["2024-04-01", "2024-07-01"],
        }
    )
    result = selection_mechanism_by_quarter(vintages, schedule)
    cash = result.iloc[0]
    active = result.iloc[1]
    assert cash["tuned_strict_count"] == 0
    assert cash["all_pair_mean_return"] == 0.0
    assert active["selected_top10_count"] == 2
    assert np.isclose(active["selected_minus_nonselected"], 0.04)


def test_conditional_strict_pass_treats_tuning_failure_as_cash() -> None:
    vintages = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "pair": ["A__B", "C__D"],
            "net_total_return": [0.1, -0.02],
            "selected_top10": [True, True],
            "trade_count": [2, 1],
        }
    )
    mechanism = pd.DataFrame(
        {
            "all_pair_mean_return": [0.04],
            "selected_mean_return": [0.04],
            "selected_minus_nonselected": [np.nan],
            "top_half_minus_bottom_half": [0.12],
            "tuned_strict_count": [2],
            "selected_top10_count": [2],
        }
    )
    result = summarize_conditional_returns(
        vintages, mechanism, strict_pass_count=3, repetitions=1_000
    ).set_index("estimand")
    cash = result.loc["strict_pass_including_untuned_as_cash"]
    assert np.isclose(cash["mean_return"], (0.1 - 0.02) / 3)
    assert cash["median_return"] == 0.0
    assert np.isclose(cash["positive_fraction"], 1 / 3)


def test_unique_pair_aggregation_inserts_cash_gaps() -> None:
    metrics = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(["2024-01-02", "2024-01-08"]),
            "outer_end_exclusive": pd.to_datetime(["2024-01-04", "2024-01-10"]),
            "pair": ["A__B", "A__B"],
            "sector": ["Test", "Test"],
            "training_score": [0.2, 0.3],
            "net_total_return": [0.1, -0.05],
            "trade_count": [1, 1],
            "selected_top10": [True, True],
        }
    )
    daily = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-08", "2024-01-08"]
            ),
            "pair": ["A__B"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"]
            ),
            "net_return": [0.1, 0.0, -0.05, 0.0],
        }
    )
    trades = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(["2024-01-02", "2024-01-08"]),
            "pair": ["A__B", "A__B"],
            "entry_date": pd.to_datetime(["2024-01-02", "2024-01-08"]),
            "exit_date": pd.to_datetime(["2024-01-03", "2024-01-09"]),
            "net_return": [0.1, -0.05],
        }
    )
    sessions = pd.bdate_range("2024-01-02", "2024-01-09")
    result, vintages = aggregate_unique_pairs(
        metrics, daily, trades, sessions, scope="test"
    )
    row = result.iloc[0]
    assert row["vintage_count"] == 2
    assert row["active_sessions"] == 4
    assert row["span_sessions"] == 6
    assert np.isclose(row["cumulative_net_return"], 1.1 * 0.95 - 1.0)
    assert np.isclose(row["quarter_hit_rate"], 0.5)
    assert len(vintages) == 2
