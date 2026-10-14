from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_pair_v8_reconstruction.py"
)
SPEC = importlib.util.spec_from_file_location("audit_pair_v8_reconstruction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _synthetic_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_dates = pd.to_datetime(
        ["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 110.0, 110.0, 121.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=price_dates,
    )
    prices.index.name = "date"
    dates = price_dates[1:]
    held_a = np.array([0.0, 0.0, 0.5, 0.0, -0.5])
    held_b = np.array([0.0, 0.0, -0.5, 0.0, 0.5])
    target_a = np.array([0.0, 0.5, 0.0, -0.5, 0.0])
    target_b = np.array([0.0, -0.5, 0.0, 0.5, 0.0])
    return_a = np.array([0.0, 0.0, 0.1, 0.0, 0.1])
    return_b = np.zeros(5)
    gross = held_a * return_a + held_b * return_b
    denominator = 1.0 + gross
    drift_a = held_a * (1.0 + return_a) / denominator
    drift_b = held_b * (1.0 + return_b) / denominator
    turnover = np.abs(target_a - drift_a) + np.abs(target_b - drift_b)
    transaction_1bp = turnover / 10_000.0 * denominator
    transaction_2bp = turnover * 2.0 / 10_000.0 * denominator
    borrow = (
        np.maximum(-held_a, 0.0) + np.maximum(-held_b, 0.0)
    ) * 30.0 / 10_000.0 / 252.0
    net_1bp = gross - transaction_1bp - borrow
    net_2bp = gross - transaction_2bp - borrow

    daily = pd.DataFrame(
        {
            "date": dates,
            "pair": "A__B",
            "strategy_id": ["S0_v6", "S4_kalman_beta", "S4_kalman_beta", "S5_ensemble", "S5_ensemble"],
            "vintage_rebalance_date": [
                dates[0],
                dates[1],
                dates[1],
                dates[3],
                dates[3],
            ],
            "ticker_a": "A",
            "ticker_b": "B",
            "price_a": prices.loc[dates, "A"].to_numpy(),
            "price_b": prices.loc[dates, "B"].to_numpy(),
            "fill_event": ["flat", "enter_long_spread", "exit_mean_reversion", "enter_short_spread", "exit_mean_reversion"],
            "fill_decision_date": [pd.NaT, dates[0], dates[1], dates[2], dates[3]],
            "fill_decision_zscore": [np.nan, -2.5, 0.2, 2.6, -0.1],
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": transaction_2bp,
            "borrow_cost": borrow,
            "net_return_1bp": net_1bp,
            "net_return_2bp": net_2bp,
            "weight_a": held_a,
            "weight_b": held_b,
            "target_weight_a": target_a,
            "target_weight_b": target_b,
            "ensemble_active_member_count": [0, 1, 0, 1, 0],
            "ensemble_held_member_count": [0, 0, 1, 0, 1],
        }
    )

    trade_rows: list[dict[str, object]] = []
    for number, (start, stop, direction) in enumerate(
        [(1, 2, "long"), (3, 4, "short")], start=1
    ):
        segment = daily.iloc[start : stop + 1]
        trade_rows.append(
            {
                "pair": "A__B",
                "trade_number": number,
                "strategy_id": daily["strategy_id"].iloc[start],
                "entry_decision_date": daily["fill_decision_date"].iloc[start],
                "entry_date": dates[start],
                "entry_fill_price_a": prices.loc[dates[start], "A"],
                "entry_fill_price_b": prices.loc[dates[start], "B"],
                "entry_fill_event": daily["fill_event"].iloc[start],
                "exit_decision_date": daily["fill_decision_date"].iloc[stop],
                "exit_date": dates[stop],
                "exit_fill_price_a": prices.loc[dates[stop], "A"],
                "exit_fill_price_b": prices.loc[dates[stop], "B"],
                "direction_ticker_a": direction,
                "trading_sessions": 1,
                "entry_zscore": daily["fill_decision_zscore"].iloc[start],
                "exit_zscore": daily["fill_decision_zscore"].iloc[stop],
                "exit_reason": daily["fill_event"].iloc[stop],
                "exit_fill_event": daily["fill_event"].iloc[stop],
                "closed": True,
                "net_return_1bp": float((1.0 + segment["net_return_1bp"]).prod() - 1.0),
                "net_return_2bp": float((1.0 + segment["net_return_2bp"]).prod() - 1.0),
                "gross_return_sum": float(segment["gross_return"].sum()),
                "transaction_cost_1bp": float(transaction_1bp[start : stop + 1].sum()),
                "transaction_cost_2bp": float(transaction_2bp[start : stop + 1].sum()),
                "opening_transaction_cost_1bp": float(transaction_1bp[start]),
                "closing_transaction_cost_1bp": float(transaction_1bp[stop]),
                "holding_rebalance_transaction_cost_1bp": float(
                    transaction_1bp[start + 1 : stop].sum()
                ),
                "opening_transaction_cost_2bp": float(transaction_2bp[start]),
                "closing_transaction_cost_2bp": float(transaction_2bp[stop]),
                "holding_rebalance_transaction_cost_2bp": float(
                    transaction_2bp[start + 1 : stop].sum()
                ),
                "borrow_cost": float(segment["borrow_cost"].sum()),
                "entry_transition": "flat_to_active",
                "exit_transition": "active_to_flat",
                "attribution_convention": (
                    "directional_episode_sequential_flip_factor_v1"
                ),
            }
        )
    return daily, pd.DataFrame(trade_rows), prices


def _synthetic_flip_paths() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_dates = pd.to_datetime(
        ["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    )
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 110.0, 121.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=price_dates,
    )
    prices.index.name = "date"
    dates = price_dates[1:]
    held_a = np.array([0.0, 0.0, 0.5, -0.5])
    held_b = np.array([0.0, 0.0, -0.5, 0.5])
    target_a = np.array([0.0, 0.5, -0.5, 0.0])
    target_b = np.array([0.0, -0.5, 0.5, 0.0])
    return_a = np.array([0.0, 0.0, 0.1, 0.1])
    return_b = np.zeros(4)
    gross = held_a * return_a + held_b * return_b
    denominator = 1.0 + gross
    drift_a = held_a * (1.0 + return_a) / denominator
    drift_b = held_b * (1.0 + return_b) / denominator
    turnover = np.abs(target_a - drift_a) + np.abs(target_b - drift_b)
    transaction_1bp = turnover / 10_000.0 * denominator
    transaction_2bp = turnover * 2.0 / 10_000.0 * denominator
    borrow = (
        np.maximum(-held_a, 0.0) + np.maximum(-held_b, 0.0)
    ) * 30.0 / 10_000.0 / 252.0
    net_1bp = gross - transaction_1bp - borrow
    net_2bp = gross - transaction_2bp - borrow

    daily = pd.DataFrame(
        {
            "date": dates,
            "pair": "A__B",
            "strategy_id": "S5_training_eligible_ensemble",
            "vintage_rebalance_date": dates[0],
            "ticker_a": "A",
            "ticker_b": "B",
            "price_a": prices.loc[dates, "A"].to_numpy(),
            "price_b": prices.loc[dates, "B"].to_numpy(),
            "fill_event": [
                "flat",
                "enter_long_spread",
                "ensemble[enter_short_spread|exit_mean_reversion|hold]",
                "exit_mean_reversion",
            ],
            "fill_decision_date": [pd.NaT, dates[0], dates[1], dates[2]],
            "fill_decision_zscore": [np.nan, -2.5, 2.5, 0.1],
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": transaction_2bp,
            "borrow_cost": borrow,
            "net_return_1bp": net_1bp,
            "net_return_2bp": net_2bp,
            "weight_a": held_a,
            "weight_b": held_b,
            "target_weight_a": target_a,
            "target_weight_b": target_b,
            "ensemble_active_member_count": [0, 1, 1, 0],
            "ensemble_held_member_count": [0, 0, 1, 1],
        }
    )

    flip = 2
    close_turnover = abs(drift_a[flip]) + abs(drift_b[flip])
    open_turnover = abs(target_a[flip]) + abs(target_b[flip])
    close_1bp = close_turnover / 10_000.0 * denominator[flip]
    open_1bp = open_turnover / 10_000.0 * denominator[flip]
    close_2bp = 2.0 * close_1bp
    open_2bp = 2.0 * open_1bp
    old_factor_1bp = gross[flip] - borrow[flip] - close_1bp
    old_factor_2bp = gross[flip] - borrow[flip] - close_2bp
    new_factor_1bp = -open_1bp / (1.0 + old_factor_1bp)
    new_factor_2bp = -open_2bp / (1.0 + old_factor_2bp)

    trades = pd.DataFrame(
        [
            {
                "pair": "A__B",
                "trade_number": 1,
                "strategy_id": "S5_training_eligible_ensemble",
                "entry_decision_date": dates[0],
                "entry_date": dates[1],
                "entry_fill_price_a": prices.loc[dates[1], "A"],
                "entry_fill_price_b": prices.loc[dates[1], "B"],
                "entry_fill_event": daily["fill_event"].iloc[1],
                "exit_decision_date": dates[1],
                "exit_date": dates[flip],
                "exit_fill_price_a": prices.loc[dates[flip], "A"],
                "exit_fill_price_b": prices.loc[dates[flip], "B"],
                "direction_ticker_a": "long",
                "trading_sessions": 1,
                "entry_zscore": -2.5,
                "exit_zscore": 2.5,
                "exit_reason": "aggregate_direction_reversal",
                "exit_fill_event": daily["fill_event"].iloc[flip],
                "closed": True,
                "net_return_1bp": float(
                    (1.0 + net_1bp[1]) * (1.0 + old_factor_1bp) - 1.0
                ),
                "net_return_2bp": float(
                    (1.0 + net_2bp[1]) * (1.0 + old_factor_2bp) - 1.0
                ),
                "gross_return_sum": float(gross[flip]),
                "transaction_cost_1bp": float(transaction_1bp[1] + close_1bp),
                "transaction_cost_2bp": float(transaction_2bp[1] + close_2bp),
                "opening_transaction_cost_1bp": float(transaction_1bp[1]),
                "closing_transaction_cost_1bp": float(close_1bp),
                "holding_rebalance_transaction_cost_1bp": 0.0,
                "opening_transaction_cost_2bp": float(transaction_2bp[1]),
                "closing_transaction_cost_2bp": float(close_2bp),
                "holding_rebalance_transaction_cost_2bp": 0.0,
                "borrow_cost": float(borrow[flip]),
                "entry_transition": "flat_to_active",
                "exit_transition": "direction_reversal",
                "attribution_convention": (
                    "directional_episode_sequential_flip_factor_v1"
                ),
            },
            {
                "pair": "A__B",
                "trade_number": 2,
                "strategy_id": "S5_training_eligible_ensemble",
                "entry_decision_date": dates[1],
                "entry_date": dates[flip],
                "entry_fill_price_a": prices.loc[dates[flip], "A"],
                "entry_fill_price_b": prices.loc[dates[flip], "B"],
                "entry_fill_event": daily["fill_event"].iloc[flip],
                "exit_decision_date": dates[flip],
                "exit_date": dates[3],
                "exit_fill_price_a": prices.loc[dates[3], "A"],
                "exit_fill_price_b": prices.loc[dates[3], "B"],
                "direction_ticker_a": "short",
                "trading_sessions": 1,
                "entry_zscore": 2.5,
                "exit_zscore": 0.1,
                "exit_reason": "exit_mean_reversion",
                "exit_fill_event": daily["fill_event"].iloc[3],
                "closed": True,
                "net_return_1bp": float(
                    (1.0 + new_factor_1bp) * (1.0 + net_1bp[3]) - 1.0
                ),
                "net_return_2bp": float(
                    (1.0 + new_factor_2bp) * (1.0 + net_2bp[3]) - 1.0
                ),
                "gross_return_sum": float(gross[3]),
                "transaction_cost_1bp": float(open_1bp + transaction_1bp[3]),
                "transaction_cost_2bp": float(open_2bp + transaction_2bp[3]),
                "opening_transaction_cost_1bp": float(open_1bp),
                "closing_transaction_cost_1bp": float(transaction_1bp[3]),
                "holding_rebalance_transaction_cost_1bp": 0.0,
                "opening_transaction_cost_2bp": float(open_2bp),
                "closing_transaction_cost_2bp": float(transaction_2bp[3]),
                "holding_rebalance_transaction_cost_2bp": 0.0,
                "borrow_cost": float(borrow[3]),
                "entry_transition": "direction_reversal",
                "exit_transition": "active_to_flat",
                "attribution_convention": (
                    "directional_episode_sequential_flip_factor_v1"
                ),
            },
        ]
    )
    return daily, trades, prices


def test_independent_reconstruction_matches_full_path_and_required_trade_types() -> None:
    daily, trades, prices = _synthetic_paths()

    detail, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is True
    assert summary["daily_row_count"] == 5
    assert summary["reconstructed_trade_count"] == 2
    assert summary["daily_checks"]["net_return_2bp"]["maximum_absolute_error"] < 1e-14
    assert summary["daily_checks"]["held_weight_state_link"]["mismatch_count"] == 0
    assert summary["dynamic_s4_present"] is True
    assert summary["ensemble_s5_present"] is True
    assert {row["category"] for row in summary["selected_trades"]} == {
        "winning",
        "losing",
        "long_a",
        "short_a",
        "dynamic_s4",
        "ensemble_s5",
    }
    assert set(detail.loc[detail["selected_categories"].ne(""), "trade_number"].dropna()) == {1, 2}
    assert detail["gross_return_matches"].all()
    assert detail["turnover_matches"].all()


def test_independent_reconstruction_splits_active_direction_reversal() -> None:
    daily, trades, prices = _synthetic_flip_paths()

    detail, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is True
    assert summary["direction_flip_count"] == 1
    assert summary["mixed_direction_episode_count"] == 0
    assert summary["reconstructed_trade_count"] == 2
    assert summary["trade_episode_path_factor_error"] < 1e-14
    flip = detail.loc[detail["date"].eq(pd.Timestamp("2020-01-06"))].iloc[0]
    assert flip["held_trade_number"] == 1
    assert flip["post_fill_trade_number"] == 2
    assert trades["transaction_cost_2bp"].sum() == pytest.approx(
        daily["transaction_cost"].sum()
    )
    assert trades["gross_return_sum"].sum() == pytest.approx(
        daily["gross_return"].sum()
    )
    assert trades["borrow_cost"].sum() == pytest.approx(
        daily["borrow_cost"].sum()
    )
    assert float(np.prod(1.0 + trades["net_return_2bp"]) - 1.0) == pytest.approx(
        float(np.prod(1.0 + daily["net_return_2bp"]) - 1.0)
    )


def test_independent_reconstruction_rejects_reversal_cost_misattribution() -> None:
    daily, trades, prices = _synthetic_flip_paths()
    shifted_cost_1bp = float(
        trades.loc[1, "opening_transaction_cost_1bp"] / 2.0
    )
    shifted_cost = float(trades.loc[1, "opening_transaction_cost_2bp"] / 2.0)
    trades.loc[0, "closing_transaction_cost_1bp"] += shifted_cost_1bp
    trades.loc[1, "opening_transaction_cost_1bp"] -= shifted_cost_1bp
    trades.loc[0, "closing_transaction_cost_2bp"] += shifted_cost
    trades.loc[1, "opening_transaction_cost_2bp"] -= shifted_cost

    _, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is False
    assert (
        summary["trade_numeric_checks"][
            "closing_transaction_cost_1bp"
        ]["mismatch_count"]
        == 1
    )
    assert (
        summary["trade_numeric_checks"][
            "opening_transaction_cost_1bp"
        ]["mismatch_count"]
        == 1
    )
    assert (
        summary["trade_numeric_checks"][
            "closing_transaction_cost_2bp"
        ]["mismatch_count"]
        == 1
    )
    assert (
        summary["trade_numeric_checks"][
            "opening_transaction_cost_2bp"
        ]["mismatch_count"]
        == 1
    )


def test_independent_reconstruction_fails_on_tampered_daily_and_ledger_values() -> None:
    daily, trades, prices = _synthetic_paths()
    daily.loc[2, "net_return_2bp"] += 0.001
    trades.loc[0, "direction_ticker_a"] = "short"

    _, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is False
    assert summary["daily_checks"]["net_return_2bp"]["mismatch_count"] == 1
    assert summary["trade_categorical_mismatches"]["direction_ticker_a"] == 1


def test_independent_reconstruction_rejects_live_member_state_at_vintage_switch() -> None:
    daily, trades, prices = _synthetic_paths()
    daily.loc[2, "ensemble_active_member_count"] = 1

    _, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is False
    assert any(
        "vintage changes before all ensemble members are flat" in violation
        for violation in summary["structural_violations"]
    )


def test_independent_reconstruction_rejects_broken_member_state_link() -> None:
    daily, trades, prices = _synthetic_paths()
    daily.loc[2, "ensemble_held_member_count"] = 0

    _, summary = audit.audit_reconstruction(
        daily,
        trades,
        prices,
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=30.0,
    )

    assert summary["passed"] is False
    assert summary["daily_checks"]["held_member_state_link"]["mismatch_count"] == 1


def test_independent_reconstruction_cli_writes_hashed_evidence(tmp_path: Path) -> None:
    daily, trades, prices = _synthetic_paths()
    output_root = tmp_path / "v8"
    identity_root = tmp_path / "identity"
    processed = output_root / "data" / "processed"
    raw = identity_root / "data" / "raw"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    daily_path = processed / audit.DEFAULT_DAILY_NAME
    trades_path = processed / audit.DEFAULT_TRADES_NAME
    prices_path = raw / "prices.csv"
    daily.to_csv(daily_path, index=False, compression="gzip")
    trades.to_csv(trades_path, index=False)
    prices.to_csv(prices_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "identity_root": str(identity_root),
                    "price_snapshot": "data/raw/prices.csv",
                },
                "evaluation": {
                    "primary_transaction_cost_bps": 2.0,
                    "annual_short_borrow_bps": 30.0,
                },
                "outputs": {"root": str(output_root)},
            }
        ),
        encoding="utf-8",
    )

    _, summary, detail_path, summary_path = audit.audit_files(config_path)

    assert summary["passed"] is True
    assert detail_path.is_file()
    assert summary_path.is_file()
    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(on_disk["input_files"]["daily"]["sha256"]) == 64
    assert on_disk["output_files"]["detail"] == str(detail_path)


def test_independent_reconstruction_rejects_duplicate_dates() -> None:
    daily, trades, prices = _synthetic_paths()
    daily = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate pair/date"):
        audit.audit_reconstruction(
            daily,
            trades,
            prices,
            transaction_cost_bps=2.0,
            annual_short_borrow_bps=30.0,
        )
