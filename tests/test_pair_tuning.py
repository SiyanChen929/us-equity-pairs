from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from equity_pairs import grid_search, pair_tuning
from equity_pairs.backtest import PairModel
from equity_pairs.config import StrategyConfig
from equity_pairs.grid_search import ParameterSet
from equity_pairs.pair_tuning import (
    PairDefinition,
    PairTuningSpec,
    build_pair_inner_folds,
    evaluate_frozen_pair_parameter,
    protocol_parameter_sets,
    refit_pair_model,
    select_pair_parameter,
    tune_eligible_pairs,
    tune_pair_parameters,
)


def _synthetic_prices(periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    time = np.arange(periods, dtype=float)
    common = 100.0 * np.exp(
        0.0007 * time + 0.10 * np.sin(time / 13.0) + 0.05 * np.cos(time / 5.0)
    )
    spread_ab = 0.075 * np.sin(time / 6.0) + 0.012 * np.sin(time / 17.0)
    spread_cd = 0.055 * np.sin(time / 8.0 + 0.8)
    return pd.DataFrame(
        {
            "B": common,
            "A": common * np.exp(spread_ab),
            "D": common * 0.82,
            "C": common * 0.82 * np.exp(spread_cd),
        },
        index=dates,
    )


def _pair() -> PairDefinition:
    return PairDefinition(
        pair="A__B",
        sector="Technology",
        dependent="A",
        independent="B",
    )


def _base_config(**overrides: object) -> StrategyConfig:
    values: dict[str, object] = {
        "zscore_method": "sma",
        "zscore_lookback": 20,
        "zscore_min_periods": 10,
        "entry_z": 1.5,
        "exit_z": 0.25,
        "stop_z": 4.0,
        "maximum_holding_days": 20,
        "transaction_cost_bps": 2.0,
        "annual_short_borrow_bps": 0.0,
    }
    values.update(overrides)
    return StrategyConfig(**values)


def _small_spec(**overrides: object) -> PairTuningSpec:
    values: dict[str, object] = {
        "zscore_methods": ("sma",),
        "zscore_lookbacks": (10,),
        "entry_zs": (1.0, 1.5),
        "exit_zs": (0.25,),
        "maximum_holding_days": (12,),
        "inner_train_size": 80,
        "inner_validation_size": 40,
        "inner_step_size": 40,
        "purge_size": 1,
        "embargo_size": 0,
        "maximum_inner_folds": 3,
        "primary_transaction_cost_bps": 2.0,
        "sensitivity_transaction_cost_bps": (1.0, 2.0),
        "minimum_full_training_trades": 0,
        "minimum_validation_trades": 1,
        "minimum_active_folds": 1,
        "minimum_hedge_observations": 50,
        "one_standard_error_rule": True,
    }
    values.update(overrides)
    return PairTuningSpec(**values)


def test_default_protocol_is_exactly_96_cells_plus_unchanged_baseline():
    parameters = protocol_parameter_sets()

    assert len(parameters) == 97
    assert len({row.config_id for row in parameters}) == 97
    assert sum(row.is_baseline for row in parameters) == 1
    searched = [row for row in parameters if not row.is_baseline]
    assert len(searched) == 96
    assert {row.zscore_method for row in searched} == {"sma", "ewma"}
    assert {row.zscore_lookback for row in searched} == {10, 20, 40, 60}
    assert {row.entry_z for row in searched} == {1.25, 1.75, 2.25}
    assert {row.exit_z for row in searched} == {0.25, 0.50}
    assert {row.maximum_holding_days for row in searched} == {10, 30}
    minimums = {row.zscore_lookback: row.zscore_min_periods for row in searched}
    assert minimums == {10: 10, 20: 16, 40: 32, 60: 48}
    baseline = next(row for row in parameters if row.is_baseline)
    assert (
        baseline.zscore_method,
        baseline.zscore_lookback,
        baseline.zscore_min_periods,
        baseline.entry_z,
        baseline.exit_z,
        baseline.maximum_holding_days,
    ) == ("sma", 60, 40, 2.0, 0.5, 60)


def test_inner_folds_are_chronological_and_refit_windows_expand():
    prices = _synthetic_prices(260)
    train = prices.index[:220]
    folds = build_pair_inner_folds(train, _small_spec())

    assert len(folds) == 3
    assert [len(fold.train_index) for fold in folds] == [80, 120, 160]
    assert all(fold.train_index[-1] < fold.test_index[0] for fold in folds)
    assert all(len(fold.purge_index) == 1 for fold in folds)
    assert all(fold.train_index.intersection(fold.test_index).empty for fold in folds)


def test_refitted_hedge_model_never_uses_evaluation_prices():
    prices = _synthetic_prices(220)
    train = prices.index[:140]
    evaluation = prices.index[140:180]
    parameter = ParameterSet(
        config_id="manual",
        zscore_method="sma",
        zscore_lookback=10,
        zscore_min_periods=10,
        entry_z=1.0,
        exit_z=0.25,
        maximum_holding_days=20,
    )
    changed = prices.copy()
    changed.loc[evaluation, "A"] *= np.linspace(1.0, 3.0, len(evaluation))

    original_model = refit_pair_model(_pair(), prices, train, minimum_observations=50)
    changed_model = refit_pair_model(_pair(), changed, train, minimum_observations=50)
    original = evaluate_frozen_pair_parameter(
        _pair(),
        prices,
        train,
        evaluation,
        parameter,
        _base_config(),
        minimum_hedge_observations=50,
    )
    mutated = evaluate_frozen_pair_parameter(
        _pair(),
        changed,
        train,
        evaluation,
        parameter,
        _base_config(),
        minimum_hedge_observations=50,
    )

    assert original_model.alpha == pytest.approx(changed_model.alpha)
    assert original_model.beta == pytest.approx(changed_model.beta)
    assert original.model.alpha == pytest.approx(mutated.model.alpha)
    assert original.model.beta == pytest.approx(mutated.model.beta)
    assert not original.daily["net_return"].equals(mutated.daily["net_return"])


def test_validation_starts_flat_and_forces_a_charged_terminal_liquidation():
    prices = _synthetic_prices(200)
    train = prices.index[:140]
    evaluation = prices.index[140:180]
    # Keep the validation spread far from its training center so the sleeve is
    # normally still open at the boundary.
    prices = prices.copy()
    prices.loc[evaluation, "A"] *= 1.18
    parameter = ParameterSet(
        config_id="persistent",
        zscore_method="sma",
        zscore_lookback=20,
        zscore_min_periods=20,
        entry_z=0.5,
        exit_z=0.0,
        maximum_holding_days=200,
    )
    result = evaluate_frozen_pair_parameter(
        _pair(),
        prices,
        train,
        evaluation,
        parameter,
        _base_config(stop_z=100.0),
        transaction_cost_bps=2.0,
        minimum_hedge_observations=50,
        terminal_liquidation=True,
    )

    assert result.daily["position"].iloc[0] == 0
    assert result.daily["target_position"].iloc[-1] == 0
    assert result.metrics["start_flat"] is True
    assert result.metrics["terminal_flat"] is True
    assert result.daily["decision_event"].iloc[-1] == "exit_terminal_scheduled"
    assert result.daily["fill_event"].iloc[-1] == "exit_terminal_scheduled"
    assert result.daily["fill_decision_date"].iloc[-1] == evaluation[-2]
    assert np.isnan(result.daily["fill_decision_zscore"].iloc[-1])
    assert result.daily["turnover"].iloc[-1] > 0
    assert result.daily["transaction_cost"].sum() == pytest.approx(
        (
            result.daily["turnover"]
            * 2.0
            / 10_000.0
            * (1.0 + result.daily["gross_return"])
        ).sum()
    )


def test_close_signal_fills_next_close_and_earns_only_after_fill(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 121.0, 121.0, 121.0],
            "B": [100.0] * 6,
        },
        index=dates,
    )
    prescribed = pd.Series(
        [np.nan, -2.0, -2.0, 0.0, 0.0, 0.0],
        index=dates,
        dtype=float,
    )

    def fixed_zscore(spread, _config):
        zscore = prescribed.reindex(spread.index).rename("zscore")
        return zscore * 0.0, zscore * 0.0 + 1.0, zscore

    monkeypatch.setattr(pair_tuning, "compute_zscore", fixed_zscore)
    model = PairModel(
        pair="A__B",
        sector="Technology",
        dependent="A",
        independent="B",
        alpha=0.0,
        beta=1.0,
    )
    parameter = ParameterSet(
        config_id="next_close",
        zscore_method="sma",
        zscore_lookback=2,
        zscore_min_periods=2,
        entry_z=1.0,
        exit_z=0.25,
        maximum_holding_days=20,
    )
    result = pair_tuning._simulate_with_model(
        model,
        prices,
        pd.DatetimeIndex([dates[0]]),
        pd.DatetimeIndex(dates[1:]),
        parameter,
        _base_config(transaction_cost_bps=10.0),
        terminal_liquidation=False,
    )
    daily = result.daily

    assert daily.loc[dates[1], "decision_event"] == "enter_long_spread"
    assert daily.loc[dates[1], "position"] == 0
    assert daily.loc[dates[1], "target_position"] == 0
    assert daily.loc[dates[2], "fill_event"] == "enter_long_spread"
    assert daily.loc[dates[2], "position"] == 0
    assert daily.loc[dates[2], "target_position"] == 1
    assert daily.loc[dates[2], "gross_return"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "transaction_cost"] == pytest.approx(0.001)
    assert daily.loc[dates[3], "position"] == 1
    assert daily.loc[dates[3], "gross_return"] == pytest.approx(0.05)
    assert daily.loc[dates[3], "decision_event"] == "exit_mean_reversion"
    assert daily.loc[dates[4], "fill_event"] == "exit_mean_reversion"
    assert daily.loc[dates[4], "target_position"] == 0

    trade = result.trades.iloc[0]
    assert trade["entry_decision_date"] == dates[1]
    assert trade["entry_date"] == dates[2]
    assert trade["exit_decision_date"] == dates[3]
    assert trade["exit_date"] == dates[4]


def test_terminal_preserves_a_prior_close_natural_exit(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"A": [100.0, 100.0, 110.0, 121.0], "B": 100.0}, index=dates)
    prescribed = pd.Series([np.nan, -2.0, 0.0, 0.0], index=dates)

    def fixed_zscore(spread, _config):
        zscore = prescribed.reindex(spread.index).rename("zscore")
        return zscore * 0.0, zscore * 0.0 + 1.0, zscore

    monkeypatch.setattr(pair_tuning, "compute_zscore", fixed_zscore)
    model = PairModel("A__B", "Technology", "A", "B", 0.0, 1.0)
    parameter = ParameterSet("natural_terminal", "sma", 2, 2, 1.0, 0.25, 20)
    result = pair_tuning._simulate_with_model(
        model,
        prices,
        pd.DatetimeIndex([dates[0]]),
        pd.DatetimeIndex(dates[1:]),
        parameter,
        _base_config(transaction_cost_bps=10.0),
        terminal_liquidation=True,
    )

    terminal = result.daily.loc[dates[-1]]
    assert terminal["fill_event"] == "exit_mean_reversion"
    assert terminal["fill_decision_date"] == dates[-2]
    assert terminal["fill_decision_zscore"] == pytest.approx(0.0)
    assert result.trades.iloc[0]["exit_reason"] == "exit_mean_reversion"


def test_one_standard_error_selection_uses_stability_tie_breaks():
    ranking = pd.DataFrame(
        [
            {
                "config_id": "raw",
                "eligible": True,
                "pooled_net_cagr": 0.22,
                "mean_fold_annual_log_return": 0.20,
                "std_fold_annual_log_return": 0.20,
                "standard_error": 0.10,
                "annualized_turnover": 1.0,
                "max_drawdown": -0.12,
            },
            {
                "config_id": "stable",
                "eligible": True,
                "pooled_net_cagr": 0.16,
                "mean_fold_annual_log_return": 0.15,
                "std_fold_annual_log_return": 0.02,
                "standard_error": 0.01,
                "annualized_turnover": 2.0,
                "max_drawdown": -0.06,
            },
            {
                "config_id": "below_cutoff",
                "eligible": True,
                "pooled_net_cagr": 0.05,
                "mean_fold_annual_log_return": 0.04,
                "std_fold_annual_log_return": 0.001,
                "standard_error": 0.001,
                "annualized_turnover": 0.1,
                "max_drawdown": -0.01,
            },
        ]
    )

    selected, audit = select_pair_parameter(ranking)

    assert selected["config_id"] == "stable"
    assert audit["raw_best_config_id"] == "raw"
    assert audit["one_standard_error_cutoff_log_return"] == pytest.approx(0.10)
    assert audit["portfolio_objective_used"] is False


def test_tuning_is_pair_specific_deterministic_and_reprices_frozen_winner():
    prices = _synthetic_prices(260)
    train = prices.index[:220]
    spec = _small_spec()

    first = tune_pair_parameters(_pair(), prices, train, _base_config(), spec)
    second = tune_pair_parameters(_pair(), prices, train, _base_config(), spec)

    assert first.selected_parameter == second.selected_parameter
    pdt.assert_frame_equal(first.ranking, second.ranking)
    assert first.selection_summary == second.selection_summary
    assert first.selection_summary["objective"] == "pair_net_cagr"
    assert first.selection_summary["portfolio_objective_used"] is False
    assert first.fold_models["model_train_end"].lt(
        first.fold_models["validation_start"]
    ).all()
    assert first.fold_models["inner_fold"].nunique() == 3
    assert first.fold_evaluations["start_flat"].all()
    assert first.fold_evaluations["terminal_flat"].all()
    sensitivity = first.cost_sensitivity.sort_values("transaction_cost_bps")
    assert sensitivity["config_id"].nunique() == 1
    assert sensitivity["config_id"].iloc[0] == first.selected_parameter.config_id
    assert sensitivity["frozen_winner"].all()
    assert first.selection_summary["sensitivity_reoptimized"] is False
    assert sensitivity.iloc[0]["total_return"] >= sensitivity.iloc[-1]["total_return"]


def test_fast_ranking_matches_canonical_tuning_field_by_field():
    prices = _synthetic_prices(260)
    train = prices.index[:220]
    spec = _small_spec(
        zscore_methods=("sma", "ewma"),
        zscore_lookbacks=(10, 20),
    )

    fast = tune_pair_parameters(
        _pair(),
        prices,
        train,
        _base_config(),
        spec,
        ranking_engine="fast",
    )
    canonical = tune_pair_parameters(
        _pair(),
        prices,
        train,
        _base_config(),
        spec,
        ranking_engine="canonical",
    )

    assert fast.selected_parameter == canonical.selected_parameter
    assert fast.selected_config == canonical.selected_config
    assert fast.selection_summary == canonical.selection_summary
    pdt.assert_frame_equal(
        fast.fold_evaluations,
        canonical.fold_evaluations,
        check_exact=True,
    )
    pdt.assert_frame_equal(
        fast.validation_returns,
        canonical.validation_returns,
        check_exact=True,
        check_freq=False,
    )
    pdt.assert_frame_equal(fast.ranking, canonical.ranking, check_exact=True)
    pdt.assert_frame_equal(fast.fold_models, canonical.fold_models, check_exact=True)
    pdt.assert_frame_equal(
        fast.selected_fold_daily,
        canonical.selected_fold_daily,
        check_exact=True,
    )
    pdt.assert_frame_equal(
        fast.cost_sensitivity,
        canonical.cost_sensitivity,
        check_exact=True,
    )


def test_fast_ranking_rejects_infinite_entry_fill_like_canonical(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {"A": [100.0, 100.0, np.inf, 100.0, 100.0, 100.0], "B": 100.0},
        index=dates,
    )
    prescribed = pd.Series(
        [np.nan, -2.0, -2.0, -2.0, 0.0, 0.0], index=dates
    )

    def fixed_zscore(spread, _config):
        zscore = prescribed.reindex(spread.index).rename("zscore")
        return zscore * 0.0, zscore * 0.0 + 1.0, zscore

    monkeypatch.setattr(pair_tuning, "compute_zscore", fixed_zscore)
    monkeypatch.setattr(grid_search, "compute_zscore", fixed_zscore)
    model = PairModel("A__B", "Technology", "A", "B", 0.0, 1.0)
    parameter = ParameterSet("edge", "sma", 2, 2, 1.0, 0.25, 20)
    config = _base_config(transaction_cost_bps=10.0)
    history = pd.DatetimeIndex([dates[0]])
    evaluation = pd.DatetimeIndex(dates[1:])

    with pytest.raises(RuntimeError, match="missing close price"):
        pair_tuning._simulate_with_model(
            model,
            prices,
            history,
            evaluation,
            parameter,
            config,
            terminal_liquidation=True,
        )
    evaluator = pair_tuning._fast_evaluator_for_pair(
        model, prices, history, evaluation, config
    )
    with pytest.raises(RuntimeError, match="missing close price"):
        pair_tuning._fast_pair_parameter_evaluation(
            evaluator, parameter, annualization=252
        )


def test_fast_ranking_matches_canonical_at_tiny_nav_denominator(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0] * 6,
            "B": [100.0, 100.0, 100.0, 299.9999999999, 300.0, 300.0],
        },
        index=dates,
    )
    prescribed = pd.Series(
        [np.nan, -2.0, -2.0, -2.0, 0.0, 0.0], index=dates
    )

    def fixed_zscore(spread, _config):
        zscore = prescribed.reindex(spread.index).rename("zscore")
        return zscore * 0.0, zscore * 0.0 + 1.0, zscore

    monkeypatch.setattr(pair_tuning, "compute_zscore", fixed_zscore)
    monkeypatch.setattr(grid_search, "compute_zscore", fixed_zscore)
    model = PairModel("A__B", "Technology", "A", "B", 0.0, 1.0)
    parameter = ParameterSet("edge", "sma", 2, 2, 1.0, 0.25, 20)
    config = _base_config(transaction_cost_bps=10.0)
    history = pd.DatetimeIndex([dates[0]])
    evaluation = pd.DatetimeIndex(dates[1:])
    canonical = pair_tuning._simulate_with_model(
        model,
        prices,
        history,
        evaluation,
        parameter,
        config,
        terminal_liquidation=True,
    )
    evaluator = pair_tuning._fast_evaluator_for_pair(
        model, prices, history, evaluation, config
    )
    fast_returns, fast_metrics = pair_tuning._fast_pair_parameter_evaluation(
        evaluator, parameter, annualization=252
    )

    pdt.assert_series_equal(
        fast_returns,
        canonical.daily["net_return"].rename("edge"),
        check_exact=True,
        check_freq=False,
    )
    assert canonical.daily.loc[dates[3], "turnover"] == pytest.approx(0.0)
    assert fast_metrics["transaction_cost_sum"] == pytest.approx(
        canonical.metrics["transaction_cost_sum"]
    )


def test_tune_eligible_pairs_excludes_nontradable_quota_fill_rows():
    prices = _synthetic_prices(260)
    selected = pd.DataFrame(
        [
            {
                "pair": "A__B",
                "sector": "Technology",
                "dependent": "A",
                "independent": "B",
                "tradable_eligible": True,
            },
            {
                "pair": "C__D",
                "sector": "Technology",
                "dependent": "C",
                "independent": "D",
                "tradable_eligible": False,
            },
        ]
    )

    batch = tune_eligible_pairs(
        selected,
        prices,
        prices.index[:220],
        _base_config(),
        _small_spec(),
        continue_on_error=False,
    )

    assert set(batch.results) == {"A__B"}
    assert batch.leaderboard["pair"].tolist() == ["A__B"]
    assert batch.leaderboard["portfolio_objective_used"].eq(False).all()
    assert batch.failures.empty


def test_frozen_evaluator_rejects_overlapping_training_and_validation():
    prices = _synthetic_prices(180)
    parameter = protocol_parameter_sets(_small_spec())[0]

    with pytest.raises(ValueError, match="end before"):
        evaluate_frozen_pair_parameter(
            _pair(),
            prices,
            prices.index[:120],
            prices.index[119:150],
            parameter,
            _base_config(),
            minimum_hedge_observations=50,
        )


def test_cost_sensitivity_settings_do_not_change_the_selected_configuration():
    prices = _synthetic_prices(260)
    train = prices.index[:220]
    base = _small_spec(sensitivity_transaction_cost_bps=(1.0, 2.0))
    expanded = replace(base, sensitivity_transaction_cost_bps=(0.0, 1.0, 2.0, 5.0))

    first = tune_pair_parameters(_pair(), prices, train, _base_config(), base)
    second = tune_pair_parameters(_pair(), prices, train, _base_config(), expanded)

    assert first.selected_parameter == second.selected_parameter
    assert second.cost_sensitivity["config_id"].nunique() == 1
    assert set(second.cost_sensitivity["transaction_cost_bps"]) == {0.0, 1.0, 2.0, 5.0}
