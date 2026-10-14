from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "aggregate_pair_identity_10y_evaluation.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "aggregate_pair_identity_10y_evaluation_test", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
aggregator = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = aggregator
MODULE_SPEC.loader.exec_module(aggregator)


def _market_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2016-08-26", "2026-08-26", name="date")


def _schedule(dates: pd.DatetimeIndex) -> pd.DataFrame:
    end_exclusive = pd.Timestamp("2026-08-27")
    first = pd.Timestamp(dates[0])
    anchors = pd.date_range(first.to_period("Q").start_time, end_exclusive, freq="QS")
    boundaries = [first]
    for anchor in anchors:
        if anchor <= first or anchor >= end_exclusive:
            continue
        candidates = dates[dates >= anchor]
        if len(candidates):
            boundaries.append(pd.Timestamp(candidates[0]))
    boundaries = sorted(set(boundaries))
    rows = []
    for number, boundary in enumerate(boundaries, start=1):
        outer_end = (
            boundaries[number] if number < len(boundaries) else end_exclusive
        )
        rows.append(
            {
                "quarter_number": number,
                "rebalance_date": boundary,
                "outer_end_exclusive": outer_end,
                "outer_sessions": int(
                    ((dates >= boundary) & (dates < outer_end)).sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    assert len(result) == 41
    return result


def _write_all_cash_inputs(root: Path) -> tuple[pd.DatetimeIndex, Path]:
    processed = root / "data" / "processed"
    raw = root / "data" / "raw"
    processed.mkdir(parents=True)
    raw.mkdir(parents=True)
    dates = _market_dates()
    schedule = _schedule(dates)
    pairs = [f"A{number:03d}__B{number:03d}" for number in range(114)]
    master = pd.DataFrame(
        {
            "pair": pairs,
            "sector": ["Synthetic"] * 114,
            "ticker_a": [f"A{number:03d}" for number in range(114)],
            "ticker_b": [f"B{number:03d}" for number in range(114)],
            "ever_v6_selected": [number < 82 for number in range(114)],
        }
    )
    master.to_csv(processed / "pair_identity_master.csv", index=False)
    schedule.to_csv(processed / "pair_identity_rebalance_schedule.csv", index=False)

    quarter_rows = []
    for pair in pairs:
        for row in schedule.itertuples(index=False):
            quarter_rows.append(
                {
                    "pair": pair,
                    "rebalance_date": row.rebalance_date,
                    "outer_end_exclusive": row.outer_end_exclusive,
                    "status": "unavailable_cash",
                    "cash_reason": "insufficient_three_year_history",
                    "history_available": False,
                    "orientation_available": False,
                    "screen_strict_eligible": False,
                    "screen_history_ratio": 0.0,
                    "tuning_success": False,
                    "active_identity_conditional": False,
                    "q_return_1bp": 0.0,
                    "q_return_2bp": 0.0,
                    "trade_count": 0,
                }
            )
    pd.DataFrame(quarter_rows).to_csv(
        processed / "pair_identity_quarterly_results.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        columns=["pair", "date", "net_return_1bp", "net_return_2bp"]
    ).to_csv(
        processed / "pair_identity_replay_daily.csv.gz",
        index=False,
        compression="gzip",
    )

    price_path = raw / "spliced.csv"
    price_path.touch()
    config = {
        "protocol": {"name": "synthetic", "random_seed": 17},
        "data": {
            "study_start_inclusive": "2016-08-26",
            "end_exclusive": "2026-08-27",
            "spliced_snapshot_name": "spliced.csv",
        },
        "evaluation": {
            "expected_clipped_quarter_count": 41,
            "ranking_cost_bps": 2.0,
        },
        "bootstrap": {
            "method": "anniversary_year_cluster_percentile",
            "draws": 100,
            "seed": 17,
            "confidence_level": 0.95,
        },
        "ranking": {"adequate_availability_minimum_fraction": 0.80},
        "outputs": {"default_root": str(root), "overwrite_v6": False},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return dates, config_path


def test_runner_schema_aliases_map_to_aggregation_taxonomy() -> None:
    raw = pd.DataFrame(
        {
            "pair": ["A__B", "A__B", "A__B", "A__B", "A__B"],
            "rebalance_date": pd.date_range("2024-01-01", periods=5, freq="QS"),
            "outer_end_exclusive": pd.date_range(
                "2024-04-01", periods=5, freq="QS"
            ),
            "status": [
                "unavailable_cash",
                "tuning_failed_cash",
                "evaluation_failed_cash",
                "evaluated_zero_trade",
                "evaluated_traded",
            ],
            "history_available": [False, True, True, True, True],
            "orientation_available": [False, True, True, True, True],
            "screen_strict_eligible": [False, False, True, False, True],
            "oos_refit_beta": [np.nan, np.nan, np.nan, 0.5, 2.0],
            "screen_dependent": [pd.NA, pd.NA, "A", "B", "A"],
            "trade_count": [0, 0, 0, 0, 2],
        }
    )

    result = aggregator._normalise_quarter_panel(raw)

    assert result["evaluation_status"].tolist() == raw["status"].tolist()
    assert result["available"].tolist() == [False, True, True, True, True]
    assert result["evaluation_success"].tolist() == [False, False, False, True, True]
    assert result["original_v6_eligible"].tolist() == [False, False, True, False, True]
    assert result["evaluated_zero_trade"].tolist() == [False, False, False, True, False]
    assert result["actively_traded"].tolist() == [False, False, False, False, True]
    assert result["evaluation_beta"].iloc[-1] == pytest.approx(2.0)
    assert result["dependent"].iloc[-1] == "A"


def test_all_cash_fixed_114_by_41_aggregation_uses_full_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "identity-evaluation"
    dates, config_path = _write_all_cash_inputs(root)
    synthetic_prices = pd.DataFrame({"DUMMY": 1.0}, index=dates)
    monkeypatch.setattr(
        aggregator,
        "load_timeseries_snapshot",
        lambda _: (synthetic_prices, {"sha256": "synthetic"}),
    )

    quality = aggregator.aggregate(config_path)

    processed = root / "data" / "processed"
    metrics = pd.read_csv(processed / "pair_identity_metrics.csv")
    quarterly = pd.read_csv(processed / "pair_identity_quarterly_results.csv.gz")
    annual = pd.read_csv(processed / "pair_identity_annual_returns.csv")
    counterfactuals = pd.read_csv(
        processed / "pair_identity_counterfactuals.csv"
    )
    rankings = pd.read_csv(processed / "pair_identity_rankings.csv")
    audit = pd.read_csv(processed / "pair_identity_audit.csv")

    assert quality["fixed_identity_count"] == 114
    assert quality["quarter_intervals"] == 41
    assert len(quarterly) == 114 * 41
    assert len(metrics) == 114 * 2
    assert metrics["study_sessions"].eq(len(dates)).all()
    assert metrics["scheduled_quarter_count"].eq(41).all()
    assert metrics["complete_calendar_year_count"].eq(9).all()
    assert metrics["final_equity"].eq(1.0).all()
    assert metrics["net_total_return"].eq(0.0).all()
    assert metrics["calendar_cagr"].eq(0.0).all()
    assert metrics["availability_fraction"].eq(0.0).all()
    assert not metrics["active_period_annualization_used"].astype(bool).any()
    for column in (
        "canonical_weight_a_iqr",
        "dominant_orientation_fraction",
        "exact_config_switch_rate",
        "parameter_component_retention",
        "zscore_method_switch_rate",
        "zscore_lookback_switch_rate",
        "entry_z_switch_rate",
        "exit_z_switch_rate",
        "maximum_holding_days_switch_rate",
    ):
        assert column in metrics
    assert len(annual) == 114 * 2 * 11
    assert annual.groupby(["pair", "transaction_cost_bps"])[
        "calendar_year_complete"
    ].sum().eq(9).all()
    assert len(counterfactuals) == 114 * 2 * 2

    primary = rankings.loc[rankings["ranking_scope"].eq("fixed_114_primary")]
    non_sharpe = primary.loc[
        primary["rank_type"].ne("adequate_availability_sharpe")
    ]
    assert non_sharpe.groupby("rank_type")["pair"].nunique().eq(114).all()
    assert primary.loc[
        primary["rank_type"].eq("adequate_availability_sharpe")
    ].empty
    assert audit["passed"].astype(bool).all()
