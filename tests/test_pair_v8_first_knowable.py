from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from equity_pairs.config import StrategyConfig
from equity_pairs.grid_search import ParameterSet
from equity_pairs.pair_tuning import PairDefinition, evaluate_frozen_pair_parameter
from equity_pairs.v7_experiments import V7Rules, simulate_v7_pair


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pair_v8_full_panel.py"
SPEC = importlib.util.spec_from_file_location("run_pair_v8_full_panel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v8)


def _daily_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [f"{year}-08-26" for year in range(2016, 2026)]
    )
    rows: list[dict[str, object]] = []
    for pair, daily_return in (("A__B", 0.10), ("C__D", 0.01)):
        activation_quarter = 3 if pair == "A__B" else 1
        for quarter, date in enumerate(dates, start=1):
            safe_reset = quarter == activation_quarter
            path_active = not safe_reset
            path_return = daily_return if path_active else 0.0
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "quarter_number": quarter,
                    "strategy_id": "S0_v6",
                    "net_return_1bp": path_return,
                    "net_return_2bp": path_return,
                    "gross_return": path_return,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "borrow_cost": 0.0,
                    "weight_a": 0.5 if path_active else 0.0,
                    "weight_b": -0.5 if path_active else 0.0,
                    "target_weight_a": 0.5 if path_active else 0.0,
                    "target_weight_b": -0.5 if path_active else 0.0,
                    "ensemble_active_member_count": 1 if path_active else 0,
                    "ensemble_held_member_count": 1 if path_active else 0,
                    "decision_event": "flat" if safe_reset else "hold",
                    "fill_event": "flat" if safe_reset else "hold",
                    "trade_entry": path_active,
                }
            )
    return pd.DataFrame(rows)


def _master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": ["A__B", "C__D"],
            "sector": ["One", "Two"],
            "ticker_a": ["A", "C"],
            "ticker_b": ["B", "D"],
            "first_successful_v6_tuned_date": ["2018-08-26", "2016-08-26"],
        }
    )


def _config() -> dict[str, object]:
    return {
        "evaluation": {
            "study_start_inclusive": "2016-08-26",
            "study_end_exclusive": "2026-08-26",
            "primary_transaction_cost_bps": 2.0,
            "stress_transaction_cost_bps": [5.0, 10.0, 20.0],
            "annual_short_borrow_bps": 30.0,
            "stress_short_borrow_bps": [100.0, 300.0],
        },
        "ranking": {
            "bootstrap_draws": 200,
            "bootstrap_seed": 17,
        },
    }


def test_safe_reset_gate_preserves_full_panel_and_starts_at_clean_flat_state() -> None:
    daily = _daily_panel()
    original = daily.copy(deep=True)

    gated = v8._apply_post_discovery_safe_reset_gate(daily, _master())

    a = gated.loc[gated["pair"].eq("A__B")].sort_values("date")
    assert a["identity_known_on_date"].tolist()[:3] == [False, False, True]
    assert a["net_return_2bp"].tolist()[:4] == [0.0, 0.0, 0.0, 0.10]
    assert a["full_panel_net_return_2bp"].tolist()[:4] == [0.10, 0.10, 0.0, 0.10]
    assert a["safe_activation_date"].dropna().iloc[0] == pd.Timestamp("2018-08-26")
    assert a["reset_wait_sessions"].eq(0).all()
    pd.testing.assert_frame_equal(daily, original)

    bad_master = _master()
    bad_master.loc[0, "first_successful_v6_tuned_date"] = None
    with pytest.raises(ValueError, match="valid first-successful V6 tuned date"):
        v8._apply_post_discovery_safe_reset_gate(daily, bad_master)


def test_safe_reset_gate_zeroes_pre_discovery_position_and_pnl() -> None:
    daily = _daily_panel()
    gated = v8._apply_post_discovery_safe_reset_gate(daily, _master())

    pre = gated["pair"].eq("A__B") & gated["date"].lt(
        pd.Timestamp("2018-08-26")
    )
    assert gated.loc[pre, "weight_a"].eq(0.0).all()
    assert gated.loc[pre, "net_return_2bp"].eq(0.0).all()
    assert gated.loc[pre, "full_panel_weight_a"].eq(0.5).all()
    assert gated.loc[pre, "full_panel_net_return_2bp"].eq(0.10).all()


def test_safe_reset_gate_requires_member_state_columns() -> None:
    daily = _daily_panel().drop(columns=["ensemble_held_member_count"])

    with pytest.raises(
        ValueError, match="ensemble_held_member_count"
    ):
        v8._apply_post_discovery_safe_reset_gate(daily, _master())


def test_safe_reset_artifacts_reuse_anniversary_bootstrap_and_inference() -> None:
    artifacts, summary = v8._build_post_discovery_safe_reset_artifacts(
        _daily_panel(),
        _master(),
        _config(),
    )

    metrics = artifacts["metrics"].set_index(["pair", "transaction_cost_bps"])
    expected_a_total = (1.10**7) - 1.0
    assert metrics.loc[("A__B", 2.0), "net_total_return"] == pytest.approx(
        expected_a_total
    )
    assert metrics.loc[("A__B", 2.0), "diagnostic_cash_sessions"] == 2
    assert metrics.loc[("A__B", 2.0), "known_sessions"] == 8
    assert metrics["conditional_on_fixed_legacy_114"].all()
    assert not metrics["complete_point_in_time_universe"].any()
    assert metrics["selection_history_includes_pre_discovery_candidates"].all()
    assert not metrics["leakage_free_strategy_claim"].any()

    assert len(artifacts["blocks"]) == 2 * 2 * 10
    assert len(artifacts["confidence"]) == 2 * 2
    assert len(artifacts["rankings"]) == 2
    assert artifacts["rankings"].iloc[0]["pair"] == "A__B"
    assert set(artifacts["inference"]["pair_identity"]) == {"A__B", "C__D"}
    assert artifacts["inference"]["conditional_on_fixed_legacy_114"].all()
    assert not artifacts["inference"]["complete_point_in_time_universe"].any()
    assert artifacts["inference"][
        "selection_history_includes_pre_discovery_candidates"
    ].all()

    assert summary["bootstrap_draws"] == 200
    assert summary["bootstrap_seed"] == 17
    assert summary["bootstrap_block_length"] == 2
    assert summary["inference"]["observation_count"] == 10
    assert summary["inference"]["draws"] == 200
    assert summary["full_panel_is_ex_post_descriptive"] is True
    assert summary["conditional_on_fixed_legacy_114"] is True
    assert summary["complete_point_in_time_universe"] is False
    assert summary["selection_history_includes_pre_discovery_candidates"] is True
    assert summary["leakage_free_strategy_claim"] is False


def test_cost_borrow_stress_reprices_frozen_path_from_held_short_notional() -> None:
    dates = pd.to_datetime(["2016-08-26", "2025-08-26"])
    gross = np.array([0.01, -0.02])
    turnover = np.array([0.5, 1.0])
    weight_a = np.array([-0.4, 0.0])
    weight_b = np.array([0.6, -0.3])
    short_notional = np.maximum(-weight_a, 0.0) + np.maximum(-weight_b, 0.0)
    primary = (
        gross
        - turnover * 2.0 / 10_000.0 * (1.0 + gross)
        - short_notional * 30.0 / 10_000.0 / 252.0
    )
    daily = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "gross_return": gross,
            "turnover": turnover,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "net_return_2bp": primary,
        }
    )

    stress = v8._build_cost_borrow_stress_metrics(
        daily,
        _master().iloc[:1],
        _config(),
    )

    assert len(stress) == 4 * 3
    assert set(stress["transaction_cost_bps"]) == {2.0, 5.0, 10.0, 20.0}
    assert set(stress["annual_short_borrow_bps"]) == {30.0, 100.0, 300.0}
    assert stress["matches_primary_ranking_assumptions"].sum() == 1
    primary_row = stress.loc[
        stress["transaction_cost_bps"].eq(2.0)
        & stress["annual_short_borrow_bps"].eq(30.0)
    ].iloc[0]
    assert primary_row["net_total_return"] == pytest.approx(
        float(np.prod(1.0 + primary) - 1.0)
    )
    stressed = stress.loc[
        stress["transaction_cost_bps"].eq(10.0)
        & stress["annual_short_borrow_bps"].eq(300.0)
    ].iloc[0]
    expected_stressed = (
        gross
        - turnover * 10.0 / 10_000.0 * (1.0 + gross)
        - short_notional * 300.0 / 10_000.0 / 252.0
    )
    assert stressed["net_total_return"] == pytest.approx(
        float(np.prod(1.0 + expected_stressed) - 1.0)
    )


def test_trade_ledger_and_internal_reconciliation_use_fill_audit_and_scaled_cost() -> None:
    gross = np.array([0.0, 0.10, 0.10])
    turnover = np.array([1.0, 0.0, 1.0])
    cost_1bp = turnover * 1.0 / 10_000.0 * (1.0 + gross)
    cost_2bp = turnover * 2.0 / 10_000.0 * (1.0 + gross)
    daily = pd.DataFrame(
        {
            "pair": "A__B",
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "strategy_id": "S0_v6",
            "price_a": [100.0, 110.0, 121.0],
            "price_b": [100.0, 100.0, 100.0],
            "weight_a": [0.0, 1.0, 1.0],
            "weight_b": [0.0, 0.0, 0.0],
            "target_weight_a": [1.0, 1.0, 0.0],
            "target_weight_b": [0.0, 0.0, 0.0],
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": cost_2bp,
            "borrow_cost": 0.0,
            "net_return_1bp": gross - cost_1bp,
            "net_return_2bp": gross - cost_2bp,
            "zscore": [-2.4, -0.8, -0.2],
            "decision_event": ["hold", "exit_mean_reversion", "flat"],
            "fill_event": ["enter_long_spread", "hold", "exit_mean_reversion"],
            "fill_decision_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "fill_decision_zscore": [-2.5, -2.4, -0.8],
        }
    )

    ledger = v8._trade_episodes(daily)

    assert len(ledger) == 1
    trade = ledger.iloc[0]
    assert trade["entry_decision_date"] == pd.Timestamp("2024-01-01")
    assert trade["exit_decision_date"] == pd.Timestamp("2024-01-03")
    assert trade["entry_fill_price_a"] == pytest.approx(100.0)
    assert trade["entry_fill_price_b"] == pytest.approx(100.0)
    assert trade["exit_fill_price_a"] == pytest.approx(121.0)
    assert trade["exit_fill_price_b"] == pytest.approx(100.0)
    assert trade["entry_zscore"] == pytest.approx(-2.5)
    assert trade["exit_zscore"] == pytest.approx(-0.8)
    assert trade["exit_reason"] == "exit_mean_reversion"
    assert trade["trading_sessions"] == 2
    assert trade["gross_return_sum"] == pytest.approx(0.20)
    assert trade["transaction_cost_2bp"] == pytest.approx(float(cost_2bp.sum()))

    _, summary = v8._internal_accounting_reconciliation(daily, ledger)
    assert summary["maximum_net_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["passed"] is True

    open_trade = v8._trade_episodes(daily.iloc[:2]).iloc[0]
    assert bool(open_trade["closed"]) is False
    assert open_trade["trading_sessions"] == 1


def _direction_reversal_daily() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    price_a = np.array([100.0, 100.0, 110.0, 99.0])
    price_b = np.full(4, 100.0)
    weight_a = np.array([0.0, 0.5, 0.5, -0.5])
    weight_b = np.array([0.0, -0.5, -0.5, 0.5])
    target_a = np.array([0.5, 0.5, -0.5, 0.0])
    target_b = np.array([-0.5, -0.5, 0.5, 0.0])
    return_a = pd.Series(price_a).pct_change(fill_method=None).fillna(0.0).to_numpy()
    return_b = pd.Series(price_b).pct_change(fill_method=None).fillna(0.0).to_numpy()
    gross = weight_a * return_a + weight_b * return_b
    denominator = 1.0 + gross
    drift_a = weight_a * (1.0 + return_a) / denominator
    drift_b = weight_b * (1.0 + return_b) / denominator
    turnover = np.abs(target_a - drift_a) + np.abs(target_b - drift_b)
    transaction_1bp = turnover / 10_000.0 * denominator
    transaction_2bp = 2.0 * turnover / 10_000.0 * denominator
    borrow = (
        np.maximum(-weight_a, 0.0) + np.maximum(-weight_b, 0.0)
    ) * 30.0 / 10_000.0 / 252.0
    return pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "strategy_id": "S5_training_eligible_ensemble",
            "price_a": price_a,
            "price_b": price_b,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "target_weight_a": target_a,
            "target_weight_b": target_b,
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": transaction_2bp,
            "borrow_cost": borrow,
            "net_return_1bp": gross - transaction_1bp - borrow,
            "net_return_2bp": gross - transaction_2bp - borrow,
            "zscore": [-2.5, -1.5, 2.5, 0.0],
            "decision_event": ["hold", "hold", "hold", "flat"],
            "fill_event": [
                "enter_long_spread",
                "hold",
                "ensemble[enter_short_spread|exit_mean_reversion|hold]",
                "exit_mean_reversion",
            ],
            "fill_decision_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "fill_decision_zscore": [-2.6, -2.5, 2.6, 0.1],
        }
    )


def test_directional_episode_entries_split_active_to_active_reversal() -> None:
    daily = _direction_reversal_daily()

    assert v8._activation_run_count(
        daily.rename(
            columns={
                "target_weight_a": "target_weight_dependent",
                "target_weight_b": "target_weight_independent",
            }
        )
    ) == 1
    entries = v8._directional_episode_entries(daily)

    assert entries.tolist() == [True, False, True, False]


def test_trade_ledger_splits_reversal_and_preserves_path_factors() -> None:
    daily = _direction_reversal_daily()

    ledger = v8._trade_episodes(daily)

    assert len(ledger) == 2
    first, second = ledger.iloc[0], ledger.iloc[1]
    assert first["direction_ticker_a"] == "long"
    assert second["direction_ticker_a"] == "short"
    assert first["exit_date"] == second["entry_date"] == pd.Timestamp("2024-01-04")
    assert first["exit_transition"] == "direction_reversal"
    assert second["entry_transition"] == "direction_reversal"
    assert first["closing_transaction_cost_2bp"] > 0.0
    assert second["opening_transaction_cost_2bp"] > 0.0
    assert first["opening_transaction_cost_2bp"] > 0.0
    assert second["closing_transaction_cost_2bp"] > 0.0
    assert ledger["transaction_cost_2bp"].sum() == pytest.approx(
        daily["turnover"].mul(2.0 / 10_000.0).mul(1.0 + daily["gross_return"]).sum()
    )
    assert ledger["gross_return_sum"].sum() == pytest.approx(
        daily["gross_return"].sum()
    )
    assert ledger["borrow_cost"].sum() == pytest.approx(
        daily["borrow_cost"].sum()
    )
    assert float(np.prod(1.0 + ledger["net_return_1bp"]) - 1.0) == pytest.approx(
        float(np.prod(1.0 + daily["net_return_1bp"]) - 1.0), abs=1e-14
    )
    assert float(np.prod(1.0 + ledger["net_return_2bp"]) - 1.0) == pytest.approx(
        float(np.prod(1.0 + daily["net_return_2bp"]) - 1.0), abs=1e-14
    )
    assert ledger["attribution_convention"].eq(
        "directional_episode_sequential_flip_factor_v1"
    ).all()


def test_trade_ledger_rejects_one_leg_only_direction_reversal() -> None:
    daily = _direction_reversal_daily()
    daily.loc[2, "target_weight_b"] = -0.5
    denominator = 1.0 + daily.loc[2, "gross_return"]
    return_a = daily.loc[2, "price_a"] / daily.loc[1, "price_a"] - 1.0
    drift_a = daily.loc[2, "weight_a"] * (1.0 + return_a) / denominator
    drift_b = daily.loc[2, "weight_b"] / denominator
    daily.loc[2, "turnover"] = (
        abs(daily.loc[2, "target_weight_a"] - drift_a)
        + abs(daily.loc[2, "target_weight_b"] - drift_b)
    )
    for cost in (1, 2):
        transaction = (
            daily.loc[2, "turnover"] * cost / 10_000.0 * denominator
        )
        daily.loc[2, f"net_return_{cost}bp"] = (
            daily.loc[2, "gross_return"]
            - transaction
            - daily.loc[2, "borrow_cost"]
        )

    with pytest.raises(ValueError, match="reverse both pair legs"):
        v8._trade_episodes(daily)


def test_dense_aggregation_preserves_execution_audit_columns() -> None:
    dates = pd.to_datetime([f"{year}-08-26" for year in range(2016, 2026)])
    ends = list(dates[1:]) + [pd.Timestamp("2026-08-26")]
    schedule = pd.DataFrame(
        {
            "quarter_number": np.arange(1, 11),
            "rebalance_date": dates,
            "outer_end_exclusive": ends,
        }
    )
    quarterly = pd.DataFrame(
        {
            "pair": "A__B",
            "rebalance_date": dates,
            "outer_end_exclusive": ends,
            "available": True,
            "evaluation_success": True,
            "original_v6_eligible": True,
            "evaluation_beta": 1.0,
        }
    )
    selections = pd.DataFrame(
        {
            "pair": "A__B",
            "rebalance_date": dates,
            "selected_strategy_id": "S0_v6",
            "selection_reason": "test",
        }
    )
    sparse = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "strategy_id": "S0_v6",
            "vintage_rebalance_date": dates,
            "zscore": np.arange(10, dtype=float),
            "decision_event": [f"decision_{number}" for number in range(10)],
            "fill_event": [f"fill_{number}" for number in range(10)],
            "fill_decision_date": dates - pd.Timedelta(days=1),
            "fill_decision_zscore": np.arange(10, dtype=float) - 0.5,
            "execution_audit_source": "direct_v7_simulation",
            "gross_return": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "borrow_cost": 0.0,
            "net_return_1bp": 0.0,
            "net_return_2bp": 0.0,
            "weight_a": 0.0,
            "weight_b": 0.0,
            "target_weight_a": 0.0,
            "target_weight_b": 0.0,
            "ensemble_active_member_count": 0,
            "ensemble_held_member_count": 0,
        }
    )
    master = _master().iloc[:1].assign(ever_v6_selected=True)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)

    artifacts = v8._aggregate_paths(
        master,
        schedule,
        quarterly,
        sparse,
        prices,
        selections,
        _config(),
    )

    dense = artifacts["daily"].sort_values("date")
    assert dense["zscore"].tolist() == list(np.arange(10, dtype=float))
    assert dense["decision_event"].tolist() == [
        f"decision_{number}" for number in range(10)
    ]
    assert dense["fill_event"].tolist() == [f"fill_{number}" for number in range(10)]
    assert pd.to_datetime(dense["fill_decision_date"]).tolist() == list(
        dates - pd.Timedelta(days=1)
    )
    assert dense["fill_decision_zscore"].tolist() == list(
        np.arange(10, dtype=float) - 0.5
    )


def test_post_discovery_gate_waits_for_safe_reset_and_preserves_full_panel() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    daily = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "decision_event": [
                "hold",
                "hold",
                "hold",
                "exit_mean_reversion",
                "flat",
                "flat",
                "enter_long_spread",
                "hold",
            ],
            "fill_event": [
                "hold",
                "hold",
                "hold",
                "hold",
                "exit_mean_reversion",
                "flat",
                "flat",
                "enter_long_spread",
            ],
            "weight_a": [0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            "weight_b": [-0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0],
            "target_weight_a": [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5],
            "target_weight_b": [-0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, -0.5],
            "turnover": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "gross_return": [0.01, -0.01, 0.02, 0.01, -0.01, 0.0, 0.0, 0.0],
            "transaction_cost": [0.0, 0.0, 0.0, 0.0, 0.0002, 0.0, 0.0, 0.0002],
            "borrow_cost": [0.00001] * 5 + [0.0, 0.0, 0.0],
            "net_return_1bp": [0.00999, -0.01001, 0.01999, 0.00999, -0.01011, 0.0, 0.0, -0.0001],
            "net_return_2bp": [0.00999, -0.01001, 0.01999, 0.00999, -0.01021, 0.0, 0.0, -0.0002],
            "trade_entry": [True, False, False, False, False, False, False, True],
            "ensemble_active_member_count": [1, 1, 1, 1, 0, 0, 0, 1],
            "ensemble_held_member_count": [1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    daily["equity_1bp"] = (1.0 + daily["net_return_1bp"]).cumprod()
    daily["equity_2bp"] = (1.0 + daily["net_return_2bp"]).cumprod()
    daily["drawdown_2bp"] = (
        daily["equity_2bp"] / daily["equity_2bp"].cummax().clip(lower=1.0) - 1.0
    )
    master = pd.DataFrame(
        {"pair": ["A__B"], "first_successful_v6_tuned_date": [dates[2]]}
    )

    gated = v8._apply_post_discovery_safe_reset_gate(daily, master)

    activation = dates[5]
    assert gated["safe_activation_date"].dropna().unique().tolist() == [
        np.datetime64(activation)
    ]
    assert gated["reset_wait_sessions"].unique().tolist() == [3]
    assert gated["safe_activation_status"].unique().tolist() == ["activated"]
    pre = gated["date"] < activation
    for column in [
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "turnover",
        "gross_return",
        "transaction_cost",
        "borrow_cost",
        "net_return_1bp",
        "net_return_2bp",
    ]:
        assert gated.loc[pre, column].eq(0.0).all()
        assert gated[f"full_panel_{column}"].equals(daily[column].astype(float))
    assert not gated.loc[pre, "trade_entry"].any()
    assert gated["full_panel_trade_entry"].tolist() == daily["trade_entry"].tolist()
    assert gated["full_panel_equity_2bp"].tolist() == daily["equity_2bp"].tolist()
    assert gated.loc[pre, "equity_2bp"].eq(1.0).all()
    assert gated.loc[gated["date"].eq(dates[7]), "trade_entry"].item()
    assert gated.loc[gated["date"].eq(dates[7]), "net_return_2bp"].item() == pytest.approx(-0.0002)


def test_post_discovery_gate_uses_all_cash_when_no_safe_reset_exists() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    daily = pd.DataFrame(
        {
            "pair": "A__B",
            "date": dates,
            "decision_event": "hold",
            "fill_event": "hold",
            "weight_a": 0.5,
            "weight_b": -0.5,
            "target_weight_a": 0.5,
            "target_weight_b": -0.5,
            "turnover": 0.0,
            "gross_return": 0.01,
            "transaction_cost": 0.0,
            "borrow_cost": 0.00001,
            "net_return_1bp": 0.00999,
            "net_return_2bp": 0.00999,
            "trade_entry": [True, False, False, False],
            "ensemble_active_member_count": 1,
            "ensemble_held_member_count": 1,
        }
    )
    master = pd.DataFrame(
        {"pair": ["A__B"], "first_successful_v6_tuned_date": [dates[1]]}
    )

    gated = v8._apply_post_discovery_safe_reset_gate(daily, master)

    assert gated["safe_activation_date"].isna().all()
    assert gated["safe_activation_status"].eq("no_safe_activation_all_cash").all()
    assert gated["reset_wait_sessions"].eq(3).all()
    assert gated["net_return_2bp"].eq(0.0).all()
    assert gated["equity_2bp"].eq(1.0).all()
    assert gated["drawdown_2bp"].eq(0.0).all()
    assert not gated["trade_entry"].any()


def test_cache_marker_rejects_tampered_artifact_and_wrong_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate_metrics.csv"
    frame = pd.DataFrame(
        {
            "pair": ["A__B"],
            "rebalance_date": ["2024-01-02"],
            "strategy_id": ["S0_v6"],
        }
    )
    frame.to_csv(artifact, index=False)
    marker_path = tmp_path / "complete.json"
    marker = v8._cache_marker_payload(
        fingerprint="fingerprint",
        artifact=artifact,
        artifact_rows=1,
        pair="A__B",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
        input_stage_validated=True,
        cache_stage="candidate",
        input_validation_stage="candidate",
        run_nonce="run-a",
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert v8._cache_marker_matches(
        marker_path,
        artifact,
        fingerprint="fingerprint",
        pair="A__B",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
    )
    artifact.write_text(
        "pair,rebalance_date,strategy_id\nA__B,2024-01-02,S1_stop_gate\n",
        encoding="utf-8",
    )
    assert not v8._cache_marker_matches(
        marker_path,
        artifact,
        fingerprint="fingerprint",
        pair="A__B",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
    )
    frame.to_csv(artifact, index=False)
    assert not v8._cache_marker_matches(
        marker_path,
        artifact,
        fingerprint="fingerprint",
        pair="C__D",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
    )

    for field, bad_value in [
        ("schema_version", v8.CACHE_SCHEMA_VERSION - 1),
        ("rebalance_date", "2024-01-03"),
        ("artifact_rows", 2),
        ("artifact_sha256", "0" * 64),
        ("artifact_date_start", "2024-01-03"),
    ]:
        changed = dict(marker)
        changed[field] = bad_value
        marker_path.write_text(json.dumps(changed), encoding="utf-8")
        assert not v8._cache_marker_matches(
            marker_path,
            artifact,
            fingerprint="fingerprint",
            pair="A__B",
            rebalance_date="2024-01-02",
            payload_sha256="payload",
            artifact_date_column="rebalance_date",
        )


def test_engine_and_s5_lineage_change_cache_fingerprint(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    source = tmp_path / "source.csv"
    config.write_text('{"version": 1}', encoding="utf-8")
    source.write_text("value\n1\n", encoding="utf-8")
    engine = {relative: str(index) for index, relative in enumerate(
        v8.ENGINE_SOURCE_RELATIVE_PATHS
    )}
    assert set(v8._engine_source_manifest()) == set(
        v8.ENGINE_SOURCE_RELATIVE_PATHS
    )
    initial = v8._fingerprint(
        config,
        [source],
        engine_manifest=engine,
        s5_ranking_manifest_sha256="s5-a",
    )
    changed_engine = dict(engine)
    changed_engine["src/equity_pairs/backtest.py"] = "changed"
    assert initial != v8._fingerprint(
        config,
        [source],
        engine_manifest=changed_engine,
        s5_ranking_manifest_sha256="s5-a",
    )
    assert initial != v8._fingerprint(
        config,
        [source],
        engine_manifest=engine,
        s5_ranking_manifest_sha256="s5-b",
    )
    assert v8._fingerprint(
        config,
        [source],
        engine_manifest=engine,
        s5_ranking_manifest_sha256="s5-a",
        runtime_manifest={"python": "a"},
    ) != v8._fingerprint(
        config,
        [source],
        engine_manifest=engine,
        s5_ranking_manifest_sha256="s5-a",
        runtime_manifest={"python": "b"},
    )

    ranking = (
        tmp_path
        / "v6_cache"
        / "a__b"
        / "20240102"
        / "parameter_rankings.csv.gz"
    )
    ranking.parent.mkdir(parents=True)
    pd.DataFrame(
        {"config_id": ["one", "two"], "eligible": [True, False]}
    ).to_csv(ranking, index=False, compression="gzip")
    successful = pd.DataFrame(
        {"pair": ["A__B"], "rebalance_date": [pd.Timestamp("2024-01-02")]}
    )
    candidates = [{"id": "S5", "ensemble": "training_eligible"}]
    manifest = v8._s5_ranking_manifest(
        successful, tmp_path / "v6_cache", candidates
    )
    assert manifest.loc[0, "path"] == str(ranking)
    first_sha = manifest.loc[0, "sha256"]
    pd.DataFrame(
        {"config_id": ["one", "two"], "eligible": [True, True]}
    ).to_csv(ranking, index=False, compression="gzip")
    changed_manifest = v8._s5_ranking_manifest(
        successful, tmp_path / "v6_cache", candidates
    )
    assert changed_manifest.loc[0, "sha256"] != first_sha


def test_s5_carry_extension_uses_longest_eligible_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Averaged S5 positions must not be truncated with one short member horizon."""

    row = {
        "screen_half_life_days": 5.0,
        "maximum_holding_days": 10,
    }
    monkeypatch.setattr(
        v8,
        "_ensemble_parameters",
        lambda _row: [
            SimpleNamespace(maximum_holding_days=10),
            SimpleNamespace(maximum_holding_days=60),
        ],
    )

    assert v8._carry_extension_holding_sessions(  # type: ignore[attr-defined]
        row,
        {"ensemble": "training_eligible"},
    ) == 60


def test_carry_stitching_fails_closed_if_extended_path_never_flattens() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    daily = pd.DataFrame(
        {
            "target_weight_dependent": [0.0, 0.5, 0.5, 0.5, 0.5],
            "target_weight_independent": [0.0, -0.5, -0.5, -0.5, -0.5],
        },
        index=dates,
    )

    with pytest.raises(RuntimeError, match="active carry has no flat exit"):
        v8._truncate_vintage_after_carry(  # type: ignore[attr-defined]
            daily,
            outer_end=dates[3],
            pair="A__B",
            rebalance_date=dates[0],
        )


def test_ensemble_carry_waits_until_all_members_flat_after_net_cancellation() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    daily = pd.DataFrame(
        {
            # The aggregate portfolio is net flat, while member state remains live.
            "target_weight_dependent": [0.0] * 5,
            "target_weight_independent": [0.0] * 5,
            "ensemble_active_member_count": [0, 1, 2, 1, 0],
        },
        index=dates,
    )

    truncated, carry_exit = v8._truncate_vintage_after_carry(
        daily,
        outer_end=dates[3],
        pair="A__B",
        rebalance_date=dates[0],
    )

    assert carry_exit == dates[4]
    assert truncated.index[-1] == dates[4]


def test_ensemble_audit_persists_active_member_count_when_targets_cancel() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)

    def member(target: float) -> SimpleNamespace:
        return SimpleNamespace(
            daily=pd.DataFrame(
                {
                    "zscore": [0.0, 1.0],
                    "decision_event": ["flat", "hold"],
                    "fill_event": ["flat", "hold"],
                    "fill_decision_date": dates,
                    "fill_decision_zscore": [0.0, 1.0],
                    "target_weight_dependent": [0.0, target],
                    "target_weight_independent": [0.0, -target],
                    "weight_dependent": [0.0, 0.0],
                    "weight_independent": [0.0, 0.0],
                },
                index=dates,
            )
        )

    aggregate = pd.DataFrame(
        {
            "target_weight_dependent": [0.0, 0.0],
            "target_weight_independent": [0.0, 0.0],
        },
        index=dates,
    )
    audited = v8._add_ensemble_execution_audit(
        aggregate,
        [member(0.5), member(-0.5)],
    )

    assert audited["ensemble_active_member_count"].tolist() == [0, 2]
    assert audited["ensemble_held_member_count"].tolist() == [0, 0]


def test_final_pair_cache_rejects_broken_held_to_target_state_link() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=3),
            "pair": ["A__B"] * 3,
            "gross_return": [0.0, 0.01, 0.0],
            "turnover": [1.0, 0.0, 0.0],
            "transaction_cost": [0.0002, 0.0, 0.0],
            "borrow_cost": [0.0, 0.0, 0.0],
            "net_return": [-0.0002, 0.01, 0.0],
            "net_return_1bp": [-0.0001, 0.01, 0.0],
            "net_return_2bp": [-0.0002, 0.01, 0.0],
            "weight_a": [0.0, 0.5, 0.0],
            "weight_b": [0.0, -0.5, 0.0],
            "target_weight_a": [0.5, 0.5, 0.0],
            "target_weight_b": [-0.5, -0.5, 0.0],
            "ensemble_active_member_count": [1, 1, 0],
            "ensemble_held_member_count": [0, 1, 1],
        }
    )

    assert not v8._final_frame_matches(frame, "A__B")


def test_final_pair_cache_rejects_broken_member_state_link_under_net_zero() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=2),
            "pair": ["A__B"] * 2,
            "gross_return": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "transaction_cost": [0.0, 0.0],
            "borrow_cost": [0.0, 0.0],
            "net_return": [0.0, 0.0],
            "net_return_1bp": [0.0, 0.0],
            "net_return_2bp": [0.0, 0.0],
            "weight_a": [0.0, 0.0],
            "weight_b": [0.0, 0.0],
            "target_weight_a": [0.0, 0.0],
            "target_weight_b": [0.0, 0.0],
            "ensemble_active_member_count": [1, 0],
            "ensemble_held_member_count": [0, 0],
        }
    )

    assert not v8._final_frame_matches(frame, "A__B")


def test_final_pair_cache_rejects_nonfinite_economic_values() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=2),
            "pair": ["A__B"] * 2,
            "gross_return": [0.0, np.nan],
            "turnover": [0.0, 0.0],
            "transaction_cost": [0.0, 0.0],
            "borrow_cost": [0.0, 0.0],
            "net_return": [0.0, 0.0],
            "net_return_1bp": [0.0, 0.0],
            "net_return_2bp": [0.0, 0.0],
            "weight_a": [0.0, 0.0],
            "weight_b": [0.0, 0.0],
            "target_weight_a": [0.0, 0.0],
            "target_weight_b": [0.0, 0.0],
            "ensemble_active_member_count": [0, 0],
            "ensemble_held_member_count": [0, 0],
        }
    )

    assert not v8._final_frame_matches(frame, "A__B")


def test_s0_actual_style_fixture_reproduces_v6_daily_accounting() -> None:
    """Keep a durable price-panel regression for the V8 S0 control path."""

    dates = pd.bdate_range("2019-01-02", periods=620)
    step = np.arange(len(dates), dtype=float)
    independent = 100.0 * np.exp(
        0.00035 * step + 0.01 * np.sin(step / 31.0)
    )
    residual = 0.045 * np.sin(step / 8.0) + 0.012 * np.sin(step / 2.7)
    dependent = np.exp(0.3 + 1.05 * np.log(independent) + residual)
    prices = pd.DataFrame(
        {"A": dependent, "B": independent},
        index=dates,
    )
    train = dates[:430]
    evaluation = dates[430:550]
    pair = PairDefinition("A__B", "actual_style_fixture", "A", "B")
    parameter = ParameterSet(
        config_id="sma__lb20__min16__en1p25__ex0p25__hold30",
        zscore_method="sma",
        zscore_lookback=20,
        zscore_min_periods=16,
        entry_z=1.25,
        exit_z=0.25,
        maximum_holding_days=30,
    )
    base = StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        maximum_holding_days=60,
        transaction_cost_bps=2.0,
        transaction_cost_scenarios_bps=(1.0, 2.0),
        annual_short_borrow_bps=30.0,
    )

    v6 = evaluate_frozen_pair_parameter(
        pair,
        prices,
        train,
        evaluation,
        parameter,
        base,
        terminal_liquidation=True,
    )
    s0 = simulate_v7_pair(
        pair,
        prices,
        train,
        evaluation,
        parameter,
        base,
        V7Rules(terminal_liquidation=True),
    )

    columns = [
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "net_return",
        "weight_dependent",
        "weight_independent",
        "target_weight_dependent",
        "target_weight_independent",
    ]
    pd.testing.assert_frame_equal(
        s0.daily[columns],
        v6.daily[columns],
        check_exact=True,
    )
    assert s0.metrics["trade_count"] == v6.metrics["trade_count"] == 5
    assert s0.metrics["net_total_return"] == pytest.approx(
        v6.metrics["net_total_return"], abs=0.0
    )
    assert s0.metrics["terminal_flat"] is True
    assert v6.metrics["terminal_flat"] is True
