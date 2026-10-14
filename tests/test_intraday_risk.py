from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_pairs.intraday_risk import (
    MinuteRiskConfig,
    audit_minute_history,
    minute_stop_decision,
)


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-02 14:31", periods=3, freq="min", tz="UTC")
    rows = []
    for ticker, closes in {"A": [100.0, 120.0, 121.0], "B": [100.0, 100.0, 100.0]}.items():
        for timestamp, close in zip(timestamps, closes, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_minute_stop_uses_completed_bar_and_next_minute_open() -> None:
    result = minute_stop_decision(
        _bars(),
        dependent="A",
        independent="B",
        alpha=0.0,
        beta=1.0,
        daily_spread_center=0.0,
        daily_spread_scale=0.02,
        position=-1,
        config=MinuteRiskConfig(stop_z=4.0),
    )
    triggered = result.loc[result["executable_stop"]]
    assert len(triggered) == 1
    row = triggered.iloc[0]
    assert row["execution_timestamp"] > row["decision_timestamp"]
    assert row["transaction_cost"] == 0.0002


def test_minute_history_absence_fails_closed(tmp_path: Path) -> None:
    audit = audit_minute_history(tmp_path / "missing.parquet")
    assert audit["passed"] is False
    assert "absent" in audit["error"]
