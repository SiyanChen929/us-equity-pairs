from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import equity_pairs.v7_experiments as v7_experiments
from equity_pairs.backtest import PairModel
from equity_pairs.config import StrategyConfig
from equity_pairs.grid_search import ParameterSet
from equity_pairs.pair_tuning import PairDefinition
from equity_pairs.portfolio_risk import cash_aware_inverse_volatility
from equity_pairs.v7_experiments import (
    V7Rules,
    aggregate_ensemble,
    simulate_v7_pair,
)


def _fixture() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-01", periods=180)
    log_b = np.linspace(4.0, 4.2, len(dates))
    residual = 0.005 * np.sin(np.arange(len(dates)) / 3.0)
    residual[150] = -0.30
    residual[151] = -0.20
    residual[152] = -0.10
    prices = pd.DataFrame(
        {"A": np.exp(0.2 + 0.8 * log_b + residual), "B": np.exp(log_b)},
        index=dates,
    )
    return prices, dates[:150], dates[150:]


def _parameter() -> ParameterSet:
    return ParameterSet(
        config_id="test",
        zscore_method="sma",
        zscore_lookback=10,
        zscore_min_periods=10,
        entry_z=1.25,
        exit_z=0.25,
        maximum_holding_days=10,
        is_baseline=False,
    )


def _base() -> StrategyConfig:
    return StrategyConfig(
        zscore_method="sma",
        zscore_lookback=10,
        zscore_min_periods=10,
        entry_z=1.25,
        exit_z=0.25,
        stop_z=4.0,
        maximum_holding_days=10,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )


def _timing_result(
    monkeypatch: pytest.MonkeyPatch,
    z_values: list[float],
    *,
    dependent_prices: list[float] | None = None,
    rules: V7Rules | None = None,
    entry_end_exclusive: pd.Timestamp | None = None,
):
    dates = pd.bdate_range("2024-01-02", periods=100 + len(z_values))
    train = dates[:100]
    evaluation = dates[100:]
    dependent = np.full(len(dates), 100.0)
    if dependent_prices is not None:
        dependent[-len(evaluation) :] = dependent_prices
    prices = pd.DataFrame(
        {"A": dependent, "B": np.full(len(dates), 100.0)},
        index=dates,
    )
    model = PairModel("A__B", "Test", "A", "B", alpha=0.0, beta=1.0)
    monkeypatch.setattr(v7_experiments, "refit_pair_model", lambda *args, **kwargs: model)

    def fixed_zscore(spread: pd.Series, config: StrategyConfig):
        center = pd.Series(0.0, index=spread.index)
        scale = pd.Series(1.0, index=spread.index)
        score = pd.Series(0.0, index=spread.index, name="zscore")
        score.loc[evaluation] = z_values
        return center, scale, score

    monkeypatch.setattr(v7_experiments, "compute_zscore", fixed_zscore)
    result = simulate_v7_pair(
        PairDefinition("A__B", "Test", "A", "B"),
        prices,
        train,
        evaluation,
        _parameter(),
        _base(),
        rules or V7Rules(terminal_liquidation=False),
        entry_end_exclusive=entry_end_exclusive,
    )
    return result, evaluation


def test_stop_gate_prevents_entry_beyond_existing_stop() -> None:
    prices, train, evaluation = _fixture()
    pair = PairDefinition("A__B", "Test", "A", "B")
    extreme_parameter = ParameterSet(
        config_id="extreme",
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=48,
        entry_z=1.25,
        exit_z=0.25,
        maximum_holding_days=10,
        is_baseline=False,
    )
    gated = simulate_v7_pair(
        pair,
        prices,
        train,
        evaluation,
        extreme_parameter,
        _base(),
        V7Rules(block_entry_at_or_beyond_stop=True),
    )
    legacy_like = simulate_v7_pair(
        pair,
        prices,
        train,
        evaluation,
        extreme_parameter,
        _base(),
        V7Rules(block_entry_at_or_beyond_stop=False),
    )
    assert gated.daily.iloc[0]["decision_event"] == "blocked_entry_beyond_stop"
    assert gated.daily.iloc[0]["target_position"] == 0
    assert legacy_like.daily.iloc[0]["target_position"] == 0
    assert legacy_like.daily.iloc[0]["decision_target_position"] == 1


@pytest.mark.parametrize(
    ("exit_signal", "exit_reason"),
    [(0.0, "exit_mean_reversion"), (5.0, "exit_stop")],
)
def test_entry_and_exit_orders_fill_at_next_session_close(
    monkeypatch: pytest.MonkeyPatch,
    exit_signal: float,
    exit_reason: str,
) -> None:
    result, dates = _timing_result(
        monkeypatch,
        [-2.0, exit_signal, 0.0, 0.0],
        dependent_prices=[100.0, 110.0, 121.0, 133.1],
    )
    daily = result.daily

    # Close d0 creates the entry order.  It fills and incurs cost at close d1,
    # after the d0->d1 return has already elapsed while flat.
    assert daily.loc[dates[0], "decision_event"] == "enter_long_spread"
    assert daily.loc[dates[0], "decision_target_position"] == 1
    assert daily.loc[dates[0], "target_position"] == 0
    assert daily.loc[dates[0], "transaction_cost"] == pytest.approx(0.0)
    assert daily.loc[dates[1], "fill_event"] == "enter_long_spread"
    assert daily.loc[dates[1], "fill_decision_date"] == dates[0]
    assert daily.loc[dates[1], "position"] == 0
    assert daily.loc[dates[1], "target_position"] == 1
    assert daily.loc[dates[1], "gross_return"] == pytest.approx(0.0)
    assert daily.loc[dates[1], "transaction_cost"] == pytest.approx(2.0 / 10_000.0)

    # The exit/stop observed at d1 also waits one close.  The first strategy
    # P&L is therefore the d1->d2 interval and the exit fills at d2.
    assert daily.loc[dates[1], "decision_event"] == exit_reason
    assert daily.loc[dates[1], "decision_target_position"] == 0
    assert daily.loc[dates[1], "target_position"] == 1
    assert daily.loc[dates[2], "fill_event"] == exit_reason
    assert daily.loc[dates[2], "fill_decision_date"] == dates[1]
    assert daily.loc[dates[2], "position"] == 1
    assert daily.loc[dates[2], "target_position"] == 0
    assert daily.loc[dates[2], "gross_return"] == pytest.approx(0.05)
    assert daily.loc[dates[2], "transaction_cost"] == pytest.approx(0.00021)
    assert daily.loc[dates[2], "borrow_cost"] == pytest.approx(
        0.5 * 30.0 / 10_000.0 / 252.0
    )
    trade = result.trades.iloc[0]
    assert trade["entry_decision_date"] == dates[0]
    assert trade["entry_date"] == dates[1]
    assert trade["entry_fill_price_dependent"] == pytest.approx(110.0)
    assert trade["entry_fill_price_independent"] == pytest.approx(100.0)
    assert trade["exit_decision_date"] == dates[1]
    assert trade["exit_date"] == dates[2]
    assert trade["exit_fill_price_dependent"] == pytest.approx(121.0)
    assert trade["exit_fill_price_independent"] == pytest.approx(100.0)
    assert trade["trading_days"] == 1
    assert trade["gross_return_sum"] == pytest.approx(0.05)
    assert trade["exit_reason"] == exit_reason


def test_entry_cannot_fill_at_or_after_entry_end_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, dates = _timing_result(
        monkeypatch,
        [0.0, -2.0, -2.0, -2.0],
        entry_end_exclusive=pd.bdate_range("2024-01-02", periods=104)[102],
    )

    # The d1 signal would fill at d2, exactly on the exclusive boundary.
    assert dates[2] == pd.bdate_range("2024-01-02", periods=104)[102]
    assert result.daily.loc[dates[1], "decision_target_position"] == 0
    assert result.daily["target_position"].eq(0).all()
    assert result.daily["transaction_cost"].eq(0.0).all()


def test_open_position_can_exit_after_entry_window_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = pd.bdate_range("2024-01-02", periods=104)[102]
    result, dates = _timing_result(
        monkeypatch,
        [-2.0, 0.0, 0.0, 0.0],
        entry_end_exclusive=boundary,
    )

    assert dates[2] == boundary
    assert result.daily.loc[dates[1], "target_position"] == 1
    assert result.daily.loc[dates[2], "target_position"] == 0
    assert result.daily.loc[dates[2], "fill_event"] == "exit_mean_reversion"


def test_terminal_liquidation_is_charged_and_borrow_matches_held_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, dates = _timing_result(
        monkeypatch,
        [-2.0, -2.0, -2.0],
        dependent_prices=[100.0, 110.0, 121.0],
        rules=V7Rules(terminal_liquidation=True),
    )
    daily = result.daily

    assert daily.loc[dates[1], "fill_event"] == "enter_long_spread"
    assert daily.loc[dates[1], "borrow_cost"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "position"] == 1
    assert daily.loc[dates[2], "target_position"] == 0
    assert daily.loc[dates[2], "fill_event"] == "exit_terminal_scheduled"
    assert daily.loc[dates[2], "fill_decision_date"] == dates[1]
    assert np.isnan(daily.loc[dates[2], "fill_decision_zscore"])
    assert daily.loc[dates[2], "transaction_cost"] > 0.0
    assert daily.loc[dates[2], "borrow_cost"] == pytest.approx(
        0.5 * 30.0 / 10_000.0 / 252.0
    )
    assert result.metrics["terminal_flat"] is True
    assert (
        result.metrics["execution_convention"]
        == "signal_close_t_fill_close_t_plus_1"
    )
    assert result.trades.iloc[0]["exit_reason"] == "exit_terminal_scheduled"
    assert result.trades.iloc[0]["exit_decision_date"] == dates[1]
    assert result.trades.iloc[0]["exit_fill_price_dependent"] == pytest.approx(121.0)


def test_terminal_preserves_a_prior_close_natural_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, dates = _timing_result(
        monkeypatch,
        [-2.0, 0.0, 0.0],
        dependent_prices=[100.0, 110.0, 121.0],
        rules=V7Rules(terminal_liquidation=True),
    )

    terminal = result.daily.loc[dates[-1]]
    assert terminal["fill_event"] == "exit_mean_reversion"
    assert terminal["fill_decision_date"] == dates[-2]
    assert terminal["fill_decision_zscore"] == pytest.approx(0.0)
    assert result.trades.iloc[0]["exit_reason"] == "exit_mean_reversion"


def test_missing_entry_fill_price_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="entry fill"):
        _timing_result(
            monkeypatch,
            [-2.0, -2.0, -2.0],
            dependent_prices=[100.0, np.nan, 100.0],
        )


def test_missing_exit_fill_price_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="exit fill"):
        _timing_result(
            monkeypatch,
            [-2.0, 0.0, 0.0, 0.0],
            dependent_prices=[100.0, 110.0, np.nan, 121.0],
        )


def test_missing_active_interval_return_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="active interval"):
        _timing_result(
            monkeypatch,
            [-2.0, -2.0, -2.0, -2.0],
            dependent_prices=[100.0, 110.0, np.nan, 121.0],
        )


def test_half_life_and_dynamic_beta_variants_preserve_accounting() -> None:
    prices, train, evaluation = _fixture()
    pair = PairDefinition("A__B", "Test", "A", "B")
    result = simulate_v7_pair(
        pair,
        prices,
        train,
        evaluation,
        _parameter(),
        _base(),
        V7Rules(
            half_life_holding_multiplier=2.0,
            dynamic_beta=True,
            kalman_process_variance_ratio=1e-5,
        ),
        screen_half_life_days=7.2,
    )
    assert result.strategy_config.maximum_holding_days == 15
    assert np.isfinite(result.daily["net_return"]).all()
    assert result.metrics["terminal_flat"] is True
    reconstructed = (1.0 + result.daily["net_return"]).prod() - 1.0
    assert np.isclose(reconstructed, result.metrics["net_total_return"])


def test_ensemble_charges_cost_after_averaging_targets() -> None:
    prices, train, evaluation = _fixture()
    pair = PairDefinition("A__B", "Test", "A", "B")
    first = simulate_v7_pair(
        pair, prices, train, evaluation, _parameter(), _base(), V7Rules()
    )
    second_parameter = ParameterSet(
        config_id="slow",
        zscore_method="ewma",
        zscore_lookback=20,
        zscore_min_periods=16,
        entry_z=1.75,
        exit_z=0.5,
        maximum_holding_days=10,
        is_baseline=False,
    )
    second = simulate_v7_pair(
        pair, prices, train, evaluation, second_parameter, _base(), V7Rules()
    )
    daily, metrics = aggregate_ensemble(
        [first, second], transaction_cost_bps=2.0, annual_short_borrow_bps=30.0
    )
    assert len(daily) == len(evaluation)
    assert metrics["configuration_count"] == 2
    assert metrics["execution_convention"] == "signal_close_t_fill_close_t_plus_1"
    assert np.isfinite(daily["net_return"]).all()
    expected_cost = (
        daily["turnover"]
        * 2.0
        / 10_000.0
        * (1.0 + daily["gross_return"])
    )
    assert np.allclose(daily["transaction_cost"], expected_cost)
    assert np.isclose(
        (1.0 + daily["net_return"]).prod() - 1.0,
        metrics["total_return"],
    )


def test_cash_aware_allocator_enforces_pair_name_and_sector_caps() -> None:
    dates = pd.bdate_range("2025-01-02", periods=80)
    returns = pd.DataFrame(
        {
            "A__B": 0.01 * np.sin(np.arange(len(dates))),
            "A__C": 0.012 * np.cos(np.arange(len(dates))),
            "D__E": 0.009 * np.sin(np.arange(len(dates)) / 2),
        },
        index=dates,
    )
    metadata = pd.DataFrame(
        {
            "pair": ["A__B", "A__C", "D__E"],
            "dependent": ["A", "A", "D"],
            "independent": ["B", "C", "E"],
            "sector": ["One", "One", "Two"],
        }
    )
    weights, diagnostics = cash_aware_inverse_volatility(
        returns,
        metadata,
        requested_gross=0.8,
        pair_cap=0.3,
        name_cap=0.35,
        sector_cap=0.4,
    )
    assert diagnostics["optimization_success"] is True
    assert weights.sum() <= 0.8 + 1e-10
    assert weights.max() <= 0.3 + 1e-10
    assert weights[["A__B", "A__C"]].sum() <= 0.35 + 1e-10
    assert weights[["A__B", "A__C"]].sum() <= 0.4 + 1e-10
