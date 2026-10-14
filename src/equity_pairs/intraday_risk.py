"""Fail-closed one-minute risk overlay for an already open pair position."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_MINUTE_COLUMNS = (
    "ticker",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True)
class MinuteRiskConfig:
    """Use the daily strategy stop; execute after a completed minute bar."""

    stop_z: float = 4.0
    transaction_cost_bps: float = 2.0
    maximum_staleness_minutes: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.stop_z) or self.stop_z <= 0:
            raise ValueError("stop_z must be positive")
        if not math.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")
        if self.maximum_staleness_minutes < 1:
            raise ValueError("maximum_staleness_minutes must be positive")


def validate_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate synchronized, timezone-aware OHLCV bars without filling gaps."""

    missing = sorted(set(REQUIRED_MINUTE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError("minute bars missing columns: " + ", ".join(missing))
    bars = frame[list(REQUIRED_MINUTE_COLUMNS)].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    if bars["timestamp"].isna().any():
        raise ValueError("minute bars contain invalid timestamps")
    if bars.duplicated(["ticker", "timestamp"]).any():
        raise ValueError("minute bars contain duplicate ticker/timestamp keys")
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    if bars[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("minute bars contain non-numeric OHLCV values")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("minute prices must be positive")
    if (bars["volume"] < 0).any():
        raise ValueError("minute volume cannot be negative")
    if (bars["high"] < bars[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("minute high is inconsistent with OHLC")
    if (bars["low"] > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("minute low is inconsistent with OHLC")
    return bars.sort_values(["timestamp", "ticker"], kind="mergesort").reset_index(
        drop=True
    )


def audit_minute_history(path: str | Path) -> dict[str, Any]:
    """Audit a minute file; absence or incomplete coverage is a hard block."""

    target = Path(path).resolve()
    audit: dict[str, Any] = {
        "path": str(target),
        "exists": target.is_file(),
        "rows": 0,
        "ticker_count": 0,
        "start": None,
        "end": None,
        "passed": False,
        "error": None,
    }
    if not target.is_file():
        audit["error"] = "required one-minute history is absent"
        return audit
    try:
        bars = validate_minute_bars(pd.read_parquet(target))
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        return audit
    audit.update(
        {
            "rows": int(len(bars)),
            "ticker_count": int(bars["ticker"].nunique()),
            "start": str(bars["timestamp"].min()),
            "end": str(bars["timestamp"].max()),
            "passed": bool(len(bars) > 0),
        }
    )
    return audit


def minute_stop_decision(
    bars: pd.DataFrame,
    *,
    dependent: str,
    independent: str,
    alpha: float,
    beta: float,
    daily_spread_center: float,
    daily_spread_scale: float,
    position: int,
    config: MinuteRiskConfig | None = None,
) -> pd.DataFrame:
    """Trigger on completed minute close and liquidate at next minute open.

    The daily center and scale must be frozen before the session. Intraday bars
    never update them. Missing/asynchronous bars are not forward-filled.
    """

    config = config or MinuteRiskConfig()
    if position not in {-1, 1}:
        raise ValueError("position must be +1 or -1 for an open pair")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if not math.isfinite(daily_spread_scale) or daily_spread_scale <= 0:
        raise ValueError("daily_spread_scale must be positive")
    clean = validate_minute_bars(bars)
    legs = clean.loc[clean["ticker"].isin([dependent, independent])]
    close = legs.pivot(index="timestamp", columns="ticker", values="close")
    opens = legs.pivot(index="timestamp", columns="ticker", values="open")
    if dependent not in close or independent not in close:
        raise ValueError("both pair legs require minute bars")
    aligned = close[[dependent, independent]].dropna()
    if len(aligned) < 2:
        raise ValueError("at least two synchronized minute bars are required")
    gaps = aligned.index.to_series().diff().dt.total_seconds().div(60.0)
    if (gaps.iloc[1:] > config.maximum_staleness_minutes).any():
        raise ValueError("minute series contains a stale synchronization gap")
    spread = (
        np.log(aligned[dependent])
        - float(alpha)
        - float(beta) * np.log(aligned[independent])
    )
    zscore = (spread - float(daily_spread_center)) / float(daily_spread_scale)
    result = pd.DataFrame(
        {
            "spread": spread,
            "intraday_zscore": zscore,
            "stop_triggered": zscore.abs().ge(config.stop_z),
        }
    )
    result["decision_timestamp"] = result.index
    result["execution_timestamp"] = result.index.to_series().shift(-1).to_numpy()
    next_open = opens[[dependent, independent]].reindex(result.index).shift(-1)
    result["execution_open_dependent"] = next_open[dependent]
    result["execution_open_independent"] = next_open[independent]
    result["liquidation_turnover"] = np.where(result["stop_triggered"], 1.0, 0.0)
    result["transaction_cost"] = (
        result["liquidation_turnover"] * config.transaction_cost_bps / 10_000.0
    )
    result["executable_stop"] = (
        result["stop_triggered"]
        & result["execution_timestamp"].notna()
        & result["execution_open_dependent"].notna()
        & result["execution_open_independent"].notna()
    )
    result["position_before_stop"] = position
    return result.reset_index(drop=True)


__all__ = [
    "MinuteRiskConfig",
    "REQUIRED_MINUTE_COLUMNS",
    "audit_minute_history",
    "minute_stop_decision",
    "validate_minute_bars",
]
