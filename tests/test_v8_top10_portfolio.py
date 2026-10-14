from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_v8_top10_portfolio.py"
SPEC = importlib.util.spec_from_file_location("analyze_v8_top10_portfolio", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
top10 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(top10)


def _ranking(count: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ranking_scope": [top10.RANKING_SCOPE] * count,
            "minimum_active_quarters": [top10.MINIMUM_ACTIVE_QUARTERS] * count,
            "rank_type": [top10.RANK_TYPE] * count,
            "rank": np.arange(1, count + 1),
            "pair": [f"P{number}__Q{number}" for number in range(count)],
            "sector": ["Synthetic"] * count,
            "transaction_cost_bps": [top10.PRIMARY_TRANSACTION_COST_BPS] * count,
            "calendar_cagr": np.linspace(0.10, 0.01, count),
            "sharpe": np.linspace(1.0, 0.1, count),
            "max_drawdown": np.linspace(-0.05, -0.20, count),
            "trade_count": [10] * count,
            "ranking_is_ex_post_descriptive": [True] * count,
            "deployable_selection": [False] * count,
        }
    )


def test_select_top_pairs_preserves_published_scope_and_order() -> None:
    selected = top10.select_top_pairs(_ranking())
    assert selected["rank"].astype(int).tolist() == list(range(1, 11))
    assert selected["pair"].tolist()[0] == "P0__Q0"
    assert not selected["deployable_selection"].any()


def test_select_top_pairs_rejects_deployable_label() -> None:
    ranking = _ranking()
    ranking.loc[0, "deployable_selection"] = True
    with pytest.raises(ValueError, match="cannot be deployable"):
        top10.select_top_pairs(ranking)


def _panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    values = {
        "A__B": ([0.10, 0.00], [1.10, 1.10]),
        "C__D": ([0.00, 0.20], [1.00, 1.20]),
    }
    rows: list[dict[str, object]] = []
    for pair, (returns, equities) in values.items():
        for date, daily_return, equity in zip(dates, returns, equities, strict=True):
            rows.append(
                {
                    "pair": pair,
                    "date": date,
                    "net_return_2bp": daily_return,
                    "gross_return": daily_return,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "borrow_cost": 0.0,
                    "weight_a": 0.5,
                    "weight_b": -0.5,
                    "trade_entry": date == dates[0],
                    "equity_2bp": equity,
                }
            )
    return pd.DataFrame(rows)


def test_build_portfolio_paths_matches_hand_calculation() -> None:
    paths, audit = top10.build_portfolio_paths(
        _panel(), ["A__B", "C__D"], recorded_equity_column="equity_2bp"
    )
    assert paths["daily_equal_weight_return"].tolist() == pytest.approx([0.05, 0.10])
    assert paths["daily_equal_weight_equity"].iloc[-1] == pytest.approx(1.155)
    assert paths["initial_equal_weight_sleeves_equity"].tolist() == pytest.approx(
        [1.05, 1.15]
    )
    assert paths["initial_equal_weight_sleeves_return"].tolist() == pytest.approx(
        [0.05, 1.15 / 1.05 - 1.0]
    )
    assert audit["source_sleeve_equity_max_error"] == pytest.approx(0.0)
    assert audit["portfolio_sleeve_path_max_error"] < 1e-12


def test_build_portfolio_paths_rejects_bad_recorded_equity() -> None:
    panel = _panel()
    panel.loc[panel.index[-1], "equity_2bp"] = 1.19
    with pytest.raises(RuntimeError, match="source sleeve equity reconstruction failed"):
        top10.build_portfolio_paths(
            panel, ["A__B", "C__D"], recorded_equity_column="equity_2bp"
        )
