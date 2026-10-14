import numpy as np
import pandas as pd

from equity_pairs.tail_attribution import decompose_pair_returns, remove_interval_as_cash


def test_pair_return_decomposition_is_exact() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=3, freq="B"),
            "price_a": [100.0, 102.0, 99.0],
            "price_b": [50.0, 49.0, 51.0],
            "weight_a": [0.0, 0.6, 0.6],
            "weight_b": [0.0, -0.4, -0.4],
            "turnover": [1.0, 0.0, 1.0],
            "borrow_cost": [0.0, 0.00001, 0.00001],
        }
    )
    ra = frame["price_a"].pct_change(fill_method=None).fillna(0.0)
    rb = frame["price_b"].pct_change(fill_method=None).fillna(0.0)
    gross = frame["weight_a"] * ra + frame["weight_b"] * rb
    frame["net_return_2bp"] = (
        gross
        - frame["turnover"] * 2.0 / 10_000.0 * (1.0 + gross)
        - frame["borrow_cost"]
    )
    output = decompose_pair_returns(frame)
    assert np.max(np.abs(output["component_error"])) < 1e-14
    assert np.isclose(output["net_dollar_pnl"].sum(), (1.0 + frame["net_return_2bp"]).prod() - 1.0)


def test_remove_interval_as_cash_is_inclusive() -> None:
    dates = pd.Series(pd.date_range("2025-01-02", periods=5, freq="B"))
    returns = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01])
    output = remove_interval_as_cash(returns, dates, start=dates.iloc[1], end=dates.iloc[3])
    assert output.tolist() == [0.01, 0.0, 0.0, 0.0, -0.01]
