from __future__ import annotations

import pandas as pd
import pytest

from equity_pairs.walkforward_top10 import (
    CrossPairSelectionSpec,
    build_first_known_quarter_history,
    first_known_candidate_history,
    rank_pairs_asof,
    select_nonoverlapping_pairs,
    simulate_managed_sleeves,
    simulate_security_level_portfolio,
)


def test_first_known_candidate_history_removes_pre_discovery_outcomes() -> None:
    candidates = pd.DataFrame(
        {
            "pair": ["A__B", "A__B", "A__B"],
            "rebalance_date": ["2022-01-03", "2022-04-01", "2022-07-01"],
            "strategy_id": ["S0", "S0", "S0"],
            "q_return_2bp": [99.0, 0.01, 0.02],
        }
    )
    master = pd.DataFrame(
        {
            "pair": ["A__B"],
            "first_successful_v6_tuned_date": ["2022-04-01"],
        }
    )
    clean = first_known_candidate_history(candidates, master)
    assert clean["rebalance_date"].tolist() == list(
        pd.to_datetime(["2022-04-01", "2022-07-01"])
    )
    assert 99.0 not in clean["q_return_2bp"].tolist()


def _quarter_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, returns, known in (
        ("A__B", [0.01, 0.02, 0.90], "2019-01-01"),
        ("C__D", [0.03, 0.04, -0.90], "2019-01-01"),
        ("E__F", [0.50, 0.50, 0.50], "2025-01-01"),
    ):
        for number, (end, value) in enumerate(
            zip(
                pd.to_datetime(["2023-04-01", "2023-07-01", "2023-10-01"]),
                returns,
                strict=True,
            ),
            start=1,
        ):
            rows.append(
                {
                    "pair": pair,
                    "quarter_number": number,
                    "rebalance_date": end - pd.DateOffset(months=3),
                    "outer_end_exclusive": end,
                    "quarter_last_session": end - pd.Timedelta(days=1),
                    "quarter_return_2bp": value,
                    "active_quarter": True,
                    "trade_entries": 2,
                    "annualized_turnover": 1.0,
                    "max_drawdown": -0.05,
                    "eligible_since_date": pd.Timestamp(known),
                    "history_available": True,
                }
            )
    return pd.DataFrame(rows)


def test_rank_pairs_uses_only_completed_history_and_known_identities() -> None:
    specification = CrossPairSelectionSpec(
        lookback_calendar_years=3,
        minimum_history_quarters=2,
        minimum_active_quarters=1,
        minimum_trade_entries=1,
        top_n=2,
    )
    ranked = rank_pairs_asof(
        _quarter_history(), pd.Timestamp("2023-07-01"), specification
    )
    assert ranked.loc[ranked["selected"], "pair"].tolist() == ["C__D", "A__B"]
    assert ranked["training_end_exclusive_max"].max() <= pd.Timestamp("2023-07-01")
    assert "E__F" not in set(ranked["pair"])


def test_future_quarter_cannot_change_current_ranking() -> None:
    specification = CrossPairSelectionSpec(
        lookback_calendar_years=3,
        minimum_history_quarters=2,
        minimum_active_quarters=1,
        minimum_trade_entries=1,
        top_n=2,
    )
    history = _quarter_history()
    first = rank_pairs_asof(history, pd.Timestamp("2023-07-01"), specification)
    history.loc[history["outer_end_exclusive"].gt("2023-07-01"), "quarter_return_2bp"] *= -100
    second = rank_pairs_asof(history, pd.Timestamp("2023-07-01"), specification)
    assert first.loc[first["selected"], "pair"].tolist() == second.loc[
        second["selected"], "pair"
    ].tolist()


def test_rank_pairs_respects_explicit_information_cutoff() -> None:
    specification = CrossPairSelectionSpec(
        lookback_calendar_years=3,
        minimum_history_quarters=2,
        minimum_active_quarters=1,
        minimum_trade_entries=1,
        top_n=2,
    )
    history = _quarter_history()
    history.loc[history["pair"].eq("C__D"), "eligible_since_date"] = pd.Timestamp(
        "2023-06-30"
    )
    ranked = rank_pairs_asof(
        history,
        pd.Timestamp("2023-07-01"),
        specification,
        information_cutoff=pd.Timestamp("2023-06-29"),
    )
    assert "C__D" not in set(ranked["pair"])
    assert ranked["training_last_session_max"].le(pd.Timestamp("2023-06-29")).all()


def test_quarter_history_reconstructs_return_and_excludes_not_yet_known_pair() -> None:
    dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    rows: list[dict[str, object]] = []
    for pair in ("A__B", "C__D"):
        for date, value in zip(dates, [0.10, -0.05], strict=True):
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "quarter_number": 1,
                    "net_return_2bp": value if pair == "A__B" else 0.0,
                    "weight_a": 0.5 if pair == "A__B" else 0.0,
                    "weight_b": -0.5 if pair == "A__B" else 0.0,
                    "target_weight_a": 0.5 if pair == "A__B" else 0.0,
                    "target_weight_b": -0.5 if pair == "A__B" else 0.0,
                    "turnover": 0.1 if pair == "A__B" else 0.0,
                    "trade_entry": date == dates[0] and pair == "A__B",
                }
            )
    schedule = pd.DataFrame(
        {
            "quarter_number": [1],
            "rebalance_date": ["2023-01-02"],
            "outer_end_exclusive": ["2023-04-03"],
        }
    )
    quarterly = pd.DataFrame(
        {
            "pair": ["A__B", "C__D"],
            "quarter_number": [1, 1],
            "evaluation_success": [True, True],
            "available": [True, True],
        }
    )
    master = pd.DataFrame(
        {
            "pair": ["A__B", "C__D"],
            "first_successful_v6_tuned_date": ["2022-01-03", "2024-01-02"],
        }
    )
    history = build_first_known_quarter_history(
        pd.DataFrame(rows), schedule, quarterly, master
    )
    a = history.loc[history["pair"].eq("A__B")].iloc[0]
    assert a["quarter_return_2bp"] == pytest.approx(1.10 * 0.95 - 1.0)
    assert a["trade_entries"] == 1
    ranked = rank_pairs_asof(
        history,
        pd.Timestamp("2023-04-03"),
        CrossPairSelectionSpec(
            lookback_calendar_years=3,
            minimum_history_quarters=1,
            minimum_active_quarters=1,
            minimum_trade_entries=1,
            top_n=1,
        ),
    )
    assert ranked.loc[ranked["selected"], "pair"].tolist() == ["A__B"]
    assert "C__D" not in set(ranked["pair"])


def test_nonoverlapping_selector_skips_shared_ticker() -> None:
    ranking = pd.DataFrame(
        {
            "pair": ["A__B", "A__C", "D__E"],
            "eligible": [True, True, True],
            "rank": [1, 2, 3],
        }
    )
    metadata = pd.DataFrame(
        {
            "pair": ["A__B", "A__C", "D__E"],
            "ticker_a": ["A", "A", "D"],
            "ticker_b": ["B", "C", "E"],
        }
    )
    selected = select_nonoverlapping_pairs(ranking, metadata, top_n=2)
    assert selected.loc[selected["selected"], "pair"].tolist() == ["A__B", "D__E"]


def test_simulator_applies_fill_lag_and_post_return_allocation_cost() -> None:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    returns = pd.DataFrame(
        {
            "A__B": [0.90, 0.90, 0.10, 0.05, 0.00],
            "C__D": [0.00, 0.00, 0.00, 0.90, 0.20],
        },
        index=dates,
    )
    events = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "fill_date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
            "selected_pairs": [("A__B",), ("C__D",)],
        }
    )
    daily = simulate_managed_sleeves(
        returns,
        events,
        portfolio_id="top10",
        slots=1,
        allocation_cost_bps=2.0,
    )
    assert daily.loc[dates[0], "portfolio_return"] == pytest.approx(0.0)
    assert daily.loc[dates[1], "pre_allocation_cost_return"] == pytest.approx(0.0)
    assert daily.loc[dates[1], "allocation_turnover"] == pytest.approx(1.0)
    assert daily.loc[dates[1], "portfolio_return"] == pytest.approx(-0.0002)
    assert daily.loc[dates[2], "portfolio_return"] == pytest.approx(0.10)
    assert daily.loc[dates[3], "pre_allocation_cost_return"] == pytest.approx(0.05)
    assert daily.loc[dates[3], "portfolio_return"] == pytest.approx(
        (1.0 + 0.05) * (1.0 - 0.0002) - 1.0
    )
    assert daily.loc[dates[4], "portfolio_return"] == pytest.approx(0.20)


def test_missing_top10_slots_remain_cash() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    returns = pd.DataFrame({"A__B": [0.0, 0.0, 0.10]}, index=dates)
    events = pd.DataFrame(
        {
            "decision_date": [dates[0]],
            "fill_date": [dates[1]],
            "selected_pairs": [("A__B",)],
        }
    )
    daily = simulate_managed_sleeves(
        returns,
        events,
        portfolio_id="top10",
        slots=2,
        allocation_cost_bps=2.0,
    )
    assert daily.loc[dates[1], "target_cash_weight"] == pytest.approx(0.5)
    assert daily.loc[dates[1], "allocation_turnover"] == pytest.approx(0.5)
    assert daily.loc[dates[1], "portfolio_return"] == pytest.approx(-0.0001)
    assert daily.loc[dates[2], "pre_allocation_cost_return"] == pytest.approx(0.05)


def test_simulator_rejects_nonfinite_or_duplicate_inputs() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"A__B": [0.0, float("nan")]}, index=dates)
    events = pd.DataFrame(
        {
            "decision_date": [dates[0]],
            "fill_date": [dates[1]],
            "selected_pairs": [("A__B",)],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        simulate_managed_sleeves(
            returns,
            events,
            portfolio_id="top10",
            slots=1,
            allocation_cost_bps=2.0,
        )


def test_security_ledger_applies_selection_after_fill_return_and_charges_legs() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B"] * 4,
            "date": dates,
            "ticker_a": ["A"] * 4,
            "ticker_b": ["B"] * 4,
            "price_a": [100.0, 100.0, 100.0, 110.0],
            "price_b": [100.0, 100.0, 100.0, 100.0],
            "weight_a": [0.0, 0.0, 0.0, 0.5],
            "weight_b": [0.0, 0.0, 0.0, -0.5],
            "target_weight_a": [0.0, 0.0, 0.5, 0.5],
            "target_weight_b": [0.0, 0.0, -0.5, -0.5],
            "decision_event": ["flat", "flat", "enter_long_spread", "hold"],
            "fill_event": ["flat", "flat", "flat", "enter_long_spread"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-02")],
            "fill_date": [pd.Timestamp("2024-01-03")],
            "target_weights": [{"A__B": 1.0}],
        }
    )
    daily = simulate_security_level_portfolio(
        signals,
        events,
        portfolio_id="top1",
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=0.0,
        terminal_liquidation=False,
    )
    assert daily.loc[dates[1], "portfolio_return"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "gross_pnl"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "transaction_cost"] == pytest.approx(0.0002)
    assert daily.loc[dates[3], "gross_pnl"] == pytest.approx(0.05)


def test_security_ledger_waits_until_post_close_shadow_target_is_flat() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B"] * 3,
            "date": dates,
            "ticker_a": ["A"] * 3,
            "ticker_b": ["B"] * 3,
            "price_a": [100.0, 110.0, 120.0],
            "price_b": [100.0, 100.0, 100.0],
            "weight_a": [0.5, 0.5, 0.0],
            "weight_b": [-0.5, -0.5, 0.0],
            "target_weight_a": [0.5, 0.0, 0.0],
            "target_weight_b": [-0.5, 0.0, 0.0],
            "decision_event": ["hold", "exit_mean_reversion", "flat"],
            "fill_event": ["hold", "hold", "exit_mean_reversion"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-01")],
            "fill_date": [dates[0]],
            "target_weights": [{"A__B": 1.0}],
        }
    )
    daily = simulate_security_level_portfolio(
        signals,
        events,
        portfolio_id="top1",
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=0.0,
        terminal_liquidation=False,
    )
    assert daily["gross_pnl"].eq(0.0).all()
    assert daily["transaction_cost"].eq(0.0).all()
    assert daily["synced_pair_count"].tolist() == [0, 1, 1]


def test_security_ledger_old_book_earns_deselection_fill_day_then_liquidates() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B"] * 3,
            "date": dates,
            "ticker_a": ["A"] * 3,
            "ticker_b": ["B"] * 3,
            "price_a": [100.0, 100.0, 110.0],
            "price_b": [100.0, 100.0, 100.0],
            "weight_a": [0.0, 0.0, 0.5],
            "weight_b": [0.0, 0.0, -0.5],
            "target_weight_a": [0.0, 0.5, 0.5],
            "target_weight_b": [0.0, -0.5, -0.5],
            "decision_event": ["flat", "enter_long_spread", "hold"],
            "fill_event": ["flat", "flat", "enter_long_spread"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "fill_date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "target_weights": [{"A__B": 1.0}, {}],
        }
    )
    daily = simulate_security_level_portfolio(
        signals,
        events,
        portfolio_id="top1",
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=0.0,
        terminal_liquidation=False,
    )
    assert daily.loc[dates[1], "transaction_cost"] == pytest.approx(0.0002)
    assert daily.loc[dates[2], "gross_pnl"] == pytest.approx(0.05)
    assert daily.loc[dates[2], "transaction_cost"] == pytest.approx(1.05 * 0.0002)
    assert daily.loc[dates[2], "selected_pair_count"] == 0


def test_security_ledger_fails_closed_on_pair_sleeve_insolvency() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B"] * 3,
            "date": dates,
            "ticker_a": ["A"] * 3,
            "ticker_b": ["B"] * 3,
            "price_a": [100.0, 100.0, 100.0],
            "price_b": [100.0, 100.0, 400.0],
            "weight_a": [0.0, 0.0, 0.5],
            "weight_b": [0.0, 0.0, -0.5],
            "target_weight_a": [0.0, 0.5, 0.5],
            "target_weight_b": [0.0, -0.5, -0.5],
            "decision_event": ["flat", "enter_long_spread", "hold"],
            "fill_event": ["flat", "flat", "enter_long_spread"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-01")],
            "fill_date": [dates[0]],
            "target_weights": [{"A__B": 0.1}],
        }
    )
    with pytest.raises(RuntimeError, match="sleeve became insolvent"):
        simulate_security_level_portfolio(
            signals,
            events,
            portfolio_id="top1",
            transaction_cost_bps=0.0,
            annual_short_borrow_bps=0.0,
            terminal_liquidation=False,
        )


def test_security_ledger_rejects_broken_next_close_shadow_state() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B", "A__B"],
            "date": dates,
            "ticker_a": ["A", "A"],
            "ticker_b": ["B", "B"],
            "price_a": [100.0, 101.0],
            "price_b": [100.0, 100.0],
            "weight_a": [0.0, 0.0],
            "weight_b": [0.0, 0.0],
            "target_weight_a": [0.5, 0.5],
            "target_weight_b": [-0.5, -0.5],
            "decision_event": ["enter_long_spread", "hold"],
            "fill_event": ["flat", "enter_long_spread"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-01")],
            "fill_date": [dates[0]],
            "target_weights": [{"A__B": 1.0}],
        }
    )
    with pytest.raises(ValueError, match="prior post-close targets"):
        simulate_security_level_portfolio(
            signals,
            events,
            portfolio_id="top1",
            transaction_cost_bps=2.0,
            annual_short_borrow_bps=0.0,
            terminal_liquidation=False,
        )


def test_security_ledger_borrow_terminal_state_and_detail_reconcile() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    signals = pd.DataFrame(
        {
            "pair": ["A__B"] * 3,
            "date": dates,
            "ticker_a": ["A"] * 3,
            "ticker_b": ["B"] * 3,
            "price_a": [100.0, 100.0, 110.0],
            "price_b": [100.0, 100.0, 100.0],
            "weight_a": [0.0, 0.0, 0.5],
            "weight_b": [0.0, 0.0, -0.5],
            "target_weight_a": [0.0, 0.5, 0.5],
            "target_weight_b": [0.0, -0.5, -0.5],
            "decision_event": ["flat", "enter_long_spread", "hold"],
            "fill_event": ["flat", "flat", "enter_long_spread"],
        }
    )
    events = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-01")],
            "fill_date": [dates[0]],
            "target_weights": [{"A__B": 1.0}],
        }
    )
    daily, detail = simulate_security_level_portfolio(
        signals,
        events,
        portfolio_id="top1",
        transaction_cost_bps=2.0,
        annual_short_borrow_bps=252.0,
        terminal_liquidation=True,
        return_detail=True,
    )
    assert daily.loc[dates[1], "borrow_cost"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "borrow_cost"] == pytest.approx(0.00005)
    assert daily.loc[dates[2], "transaction_cost"] == pytest.approx(1.05 * 0.0002)
    assert daily.loc[dates[2], "gross_exposure"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "unallocated_cash_weight"] == pytest.approx(1.0)
    assert daily.loc[dates[2], "idle_sleeve_weight"] == pytest.approx(0.0)
    assert daily.loc[dates[2], "selected_pair_count"] == 0
    grouped = detail.groupby("date").agg(
        gross_pnl=("gross_pnl", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        transaction_cost=("transaction_cost", "sum"),
        leg_turnover=("leg_turnover", "sum"),
        borne_pair=("transaction_cost_borne_by_pair", "sum"),
        borne_cash=("transaction_cost_borne_by_cash", "sum"),
    )
    for column in ("gross_pnl", "borrow_cost", "transaction_cost", "leg_turnover"):
        assert grouped[column].to_numpy() == pytest.approx(daily[column].to_numpy())
    assert grouped["transaction_cost"].to_numpy() == pytest.approx(
        (grouped["borne_pair"] + grouped["borne_cash"]).to_numpy()
    )
