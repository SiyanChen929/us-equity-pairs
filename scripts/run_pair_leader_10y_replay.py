#!/usr/bin/env python3
"""Ten-year conditional walk-forward replay for the realized V6 leaders.

This is an additive stability study.  Pair identities are taken after observing
the V6 outer results, so the study is explicitly post-selection.  Within every
replay vintage, however, orientation, hedge coefficients, health diagnostics,
and all 97 hyperparameter candidates use only the preceding three calendar
years.  Adjacent frozen next-quarter returns are compounded exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from equity_pairs.backtest import performance_metrics
from equity_pairs.config import StrategyConfig
from equity_pairs.data import (
    download_adjusted_close,
    load_timeseries_snapshot,
)
from equity_pairs.pair_tuning import (
    PairDefinition,
    PairTuningSpec,
    evaluate_tuned_pair,
    protocol_parameter_sets,
    tune_pair_parameters,
)
from equity_pairs.statistics import (
    PairScreenSettings,
    evaluate_pair,
    integration_diagnostics,
)


DEFAULT_CONFIG = Path("configs/pair_leader_10y_replay.json")
DEFAULT_OUTPUT = Path("outputs/pair_leader_10y_replay_2026-08-26")
PRIMARY_COST_BPS = 2.0
SENSITIVITY_COST_BPS = 1.0

INK = "#172033"
MUTED = "#667188"
BLUE = "#2f67c7"
TEAL = "#138a82"
GREEN = "#2a845e"
ORANGE = "#da7a2e"
RED = "#bf4c55"
PALE = "#edf3f8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#cdd6e2",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_csv(path: Path, *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], errors="coerce", format="mixed"
            )
    return frame


def _safe_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol",
        "source",
        "data",
        "walk_forward",
        "health",
        "hyperparameter_grid",
        "validation",
        "costs",
        "outputs",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("leader replay config is missing: " + ", ".join(missing))
    if not payload["source"].get("pair_identity_selection_is_ex_post", False):
        raise ValueError("leader replay must declare ex-post identity selection")
    if payload["validation"].get("portfolio_objective_used", True):
        raise ValueError("leader replay forbids a portfolio objective")
    if not payload["walk_forward"].get("outer_windows_are_non_overlapping", False):
        raise ValueError("leader replay requires non-overlapping outer windows")
    return payload


def _build_protocol(config: Mapping[str, Any]) -> tuple[PairTuningSpec, StrategyConfig]:
    grid = config["hyperparameter_grid"]
    validation = config["validation"]
    costs = config["costs"]
    spec = PairTuningSpec(
        zscore_methods=tuple(str(value) for value in grid["zscore_methods"]),
        zscore_lookbacks=tuple(int(value) for value in grid["zscore_lookbacks"]),
        entry_zs=tuple(float(value) for value in grid["entry_zs"]),
        exit_zs=tuple(float(value) for value in grid["exit_zs"]),
        maximum_holding_days=tuple(
            int(value) for value in grid["maximum_holding_days"]
        ),
        inner_train_size=int(validation["inner_train_sessions"]),
        inner_validation_size=int(validation["inner_validation_sessions"]),
        inner_step_size=int(validation["inner_step_sessions"]),
        purge_size=int(validation["purge_sessions"]),
        embargo_size=int(validation["embargo_sessions"]),
        maximum_inner_folds=int(validation["maximum_inner_folds"]),
        primary_transaction_cost_bps=float(costs["primary_transaction_cost_bps"]),
        sensitivity_transaction_cost_bps=(
            float(costs["frozen_sensitivity_transaction_cost_bps"]),
            float(costs["primary_transaction_cost_bps"]),
        ),
        minimum_full_training_trades=int(
            validation["minimum_full_training_trades"]
        ),
        minimum_validation_trades=int(
            validation["minimum_pooled_validation_trades"]
        ),
        minimum_active_folds=int(
            validation["minimum_active_validation_folds"]
        ),
        minimum_hedge_observations=100,
        annualization=252,
        one_standard_error_rule=validation["selection_rule"]
        == "one_standard_error",
    )
    base = StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=float(grid["stop_z"]),
        maximum_holding_days=60,
        transaction_cost_bps=float(costs["primary_transaction_cost_bps"]),
        transaction_cost_scenarios_bps=(
            float(costs["frozen_sensitivity_transaction_cost_bps"]),
            float(costs["primary_transaction_cost_bps"]),
        ),
        annual_short_borrow_bps=float(costs["annual_short_borrow_bps"]),
        minimum_trades_for_ranking=max(
            1, int(validation["minimum_full_training_trades"])
        ),
    )
    candidate_count = len(protocol_parameter_sets(spec, stop_z=base.stop_z))
    if candidate_count != int(grid["candidate_count_including_baseline"]):
        raise ValueError(f"expected 97 tuning candidates, received {candidate_count}")
    return spec, base


def select_source_leaders(
    metrics: pd.DataFrame,
    pair_vintage_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the realized top pair-vintages and their ordered unique identities."""

    required = {"pair", "rebalance_date", "net_total_return", "sector"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError("V6 metrics are missing: " + ", ".join(missing))
    work = metrics.copy()
    work["net_total_return"] = pd.to_numeric(
        work["net_total_return"], errors="coerce"
    )
    if "net_cagr" not in work:
        work["net_cagr"] = np.nan
    work["net_cagr"] = pd.to_numeric(work["net_cagr"], errors="coerce")
    work["rebalance_date"] = pd.to_datetime(
        work["rebalance_date"], errors="coerce", format="mixed"
    )
    work = work.loc[np.isfinite(work["net_total_return"])].sort_values(
        ["net_total_return", "net_cagr", "rebalance_date", "pair"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    top = work.head(pair_vintage_count).copy()
    top.insert(0, "realized_pair_vintage_rank", np.arange(1, len(top) + 1))
    identities = top.drop_duplicates("pair", keep="first").copy()
    identities.insert(0, "leader_identity_rank", np.arange(1, len(identities) + 1))
    identities["ticker_a"] = identities["pair"].astype(str).str.split("__").str[0]
    identities["ticker_b"] = identities["pair"].astype(str).str.split("__").str[1]
    identities["identity_selection_timing"] = "ex_post_after_v6_outer_returns"
    identities["deployable_selection"] = False
    identity_columns = [
        "leader_identity_rank",
        "pair",
        "sector",
        "ticker_a",
        "ticker_b",
        "realized_pair_vintage_rank",
        "rebalance_date",
        "net_total_return",
        "net_cagr",
        "identity_selection_timing",
        "deployable_selection",
    ]
    return top.reset_index(drop=True), identities[identity_columns].reset_index(drop=True)


def build_nonoverlapping_schedule(
    market_index: pd.DatetimeIndex,
    start_inclusive: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
) -> pd.DataFrame:
    """Partition the study interval into one initial stub plus calendar quarters."""

    index = pd.DatetimeIndex(market_index).sort_values().unique()
    start = pd.Timestamp(start_inclusive)
    end = pd.Timestamp(end_exclusive)
    eligible_start = index[(index >= start) & (index < end)]
    if eligible_start.empty:
        raise ValueError("no market session falls inside the requested OOS interval")
    first = pd.Timestamp(eligible_start[0])
    anchors = pd.date_range(first.to_period("Q").start_time, end, freq="QS")
    boundaries: list[pd.Timestamp] = [first]
    for anchor in anchors:
        if anchor <= first or anchor >= end:
            continue
        sessions = index[(index >= anchor) & (index < end)]
        if len(sessions):
            boundaries.append(pd.Timestamp(sessions[0]))
    boundaries = sorted(set(boundaries))
    rows: list[dict[str, Any]] = []
    for location, boundary in enumerate(boundaries):
        next_boundary = (
            boundaries[location + 1] if location + 1 < len(boundaries) else end
        )
        outer = index[(index >= boundary) & (index < next_boundary)]
        if not len(outer):
            continue
        rows.append(
            {
                "rebalance_date": boundary,
                "outer_end_exclusive": next_boundary,
                "outer_sessions": int(len(outer)),
                "initial_stub": bool(location == 0 and boundary.to_period("Q").start_time < boundary),
                "final_truncated": bool(next_boundary == end),
            }
        )
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ValueError("the replay schedule is empty")
    contiguous = all(
        pd.Timestamp(schedule.iloc[location - 1]["outer_end_exclusive"])
        == pd.Timestamp(schedule.iloc[location]["rebalance_date"])
        for location in range(1, len(schedule))
    )
    if not contiguous:
        raise RuntimeError("outer replay windows are not contiguous")
    return schedule


def _core_health_pass(diagnostic: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    half_life = float(diagnostic.get("half_life_days", math.nan))
    hurst = float(diagnostic.get("hurst_exponent", math.nan))
    pvalue = float(diagnostic.get("pair_pvalue", math.nan))
    return bool(
        diagnostic.get("screen_status") == "ok"
        and np.isfinite(pvalue)
        and pvalue <= float(config["pair_pvalue_maximum"])
        and (not config["require_both_legs_i1"] or diagnostic.get("both_look_i1", False))
        and (
            not config["require_positive_hedge_ratio"]
            or diagnostic.get("positive_hedge_ratio", False)
        )
        and np.isfinite(half_life)
        and 0 < half_life <= float(config["maximum_half_life_days"])
        and np.isfinite(hurst)
        and hurst < float(config["maximum_hurst_exponent_exclusive"])
    )


def _flat_daily(
    pair: str,
    rebalance_date: pd.Timestamp,
    outer_end_exclusive: pd.Timestamp,
    outer_index: pd.DatetimeIndex,
    status: str,
    health_pass: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": pair,
            "rebalance_date": rebalance_date,
            "outer_end_exclusive": outer_end_exclusive,
            "date": outer_index,
            "status": status,
            "core_health_pass": health_pass,
            "tuning_success": False,
            "config_id": pd.NA,
            "zscore": np.nan,
            "position": 0,
            "turnover": 0.0,
            "transaction_cost_2bp": 0.0,
            "borrow_cost": 0.0,
            "return_always_attempt_2bp": 0.0,
            "return_core_health_gated_2bp": 0.0,
            "return_always_attempt_1bp": 0.0,
            "return_core_health_gated_1bp": 0.0,
        }
    )


_WORKER_STATE: dict[str, Any] = {}


def _initialize_worker(
    prices: pd.DataFrame,
    spec: PairTuningSpec,
    base_config: StrategyConfig,
    protocol: dict[str, Any],
    cache_root: str,
    price_sha256: str,
    config_sha256: str,
    recompute: bool,
) -> None:
    _WORKER_STATE.update(
        {
            "prices": prices,
            "spec": spec,
            "base_config": base_config,
            "protocol": protocol,
            "cache_root": Path(cache_root),
            "price_sha256": price_sha256,
            "config_sha256": config_sha256,
            "recompute": recompute,
        }
    )


def _cache_paths(pair: str, rebalance_date: pd.Timestamp) -> dict[str, Path]:
    directory = (
        _WORKER_STATE["cache_root"]
        / _safe_token(pair)
        / pd.Timestamp(rebalance_date).strftime("%Y%m%d")
    )
    return {
        "directory": directory,
        "daily": directory / "daily.csv.gz",
        "trades": directory / "trades.csv",
        "summary": directory / "summary.csv",
        "complete": directory / "complete.json",
    }


def _task_fingerprint(task: Mapping[str, Any]) -> str:
    payload = {
        "pair": task["pair"],
        "sector": task["sector"],
        "ticker_a": task["ticker_a"],
        "ticker_b": task["ticker_b"],
        "rebalance_date": str(pd.Timestamp(task["rebalance_date"])),
        "outer_end_exclusive": str(pd.Timestamp(task["outer_end_exclusive"])),
        "price_sha256": _WORKER_STATE["price_sha256"],
        "config_sha256": _WORKER_STATE["config_sha256"],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_task(
    paths: Mapping[str, Path], fingerprint: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if _WORKER_STATE["recompute"] or not paths["complete"].is_file():
        return None
    try:
        metadata = json.loads(paths["complete"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("fingerprint") != fingerprint:
        return None
    if not all(paths[name].is_file() for name in ("daily", "trades", "summary")):
        return None
    daily = _read_csv(
        paths["daily"], dates=("rebalance_date", "outer_end_exclusive", "date")
    )
    trades = _read_csv(
        paths["trades"], dates=("rebalance_date", "entry_date", "exit_date")
    )
    summary = _read_csv(
        paths["summary"],
        dates=(
            "rebalance_date",
            "outer_end_exclusive",
            "formation_start",
            "formation_end",
            "evaluation_start",
            "evaluation_end",
        ),
    )
    return daily, trades, summary


def _write_task_cache(
    paths: Mapping[str, Path],
    fingerprint: str,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    paths["directory"].mkdir(parents=True, exist_ok=True)
    _atomic_csv(daily, paths["daily"])
    _atomic_csv(trades, paths["trades"])
    _atomic_csv(summary, paths["summary"])
    _atomic_json(
        paths["complete"],
        {
            "fingerprint": fingerprint,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "daily_rows": len(daily),
            "trade_rows": len(trades),
        },
    )


def _run_vintage_task(
    task: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Tune and evaluate one identity/vintage, or fail closed to cash."""

    prices: pd.DataFrame = _WORKER_STATE["prices"]
    spec: PairTuningSpec = _WORKER_STATE["spec"]
    base_config: StrategyConfig = _WORKER_STATE["base_config"]
    protocol: dict[str, Any] = _WORKER_STATE["protocol"]
    pair = str(task["pair"])
    sector = str(task["sector"])
    ticker_a = str(task["ticker_a"])
    ticker_b = str(task["ticker_b"])
    rebalance = pd.Timestamp(task["rebalance_date"])
    outer_end = pd.Timestamp(task["outer_end_exclusive"])
    fingerprint = _task_fingerprint(task)
    paths = _cache_paths(pair, rebalance)
    cached = _load_cached_task(paths, fingerprint)
    if cached is not None:
        return (*cached, True)

    training_start = rebalance - pd.DateOffset(
        years=int(protocol["data"]["required_training_years"])
    )
    formation_index = prices.index[
        (prices.index >= training_start) & (prices.index < rebalance)
    ]
    outer_index = prices.index[
        (prices.index >= rebalance) & (prices.index < outer_end)
    ]
    pair_prices = prices[[ticker_a, ticker_b]].copy()
    aligned = pair_prices.loc[formation_index].dropna()
    aligned = aligned.loc[(aligned > 0).all(axis=1)]
    expected = int(len(formation_index))
    aligned_count = int(len(aligned))
    history_ratio = aligned_count / expected if expected else 0.0
    common: dict[str, Any] = {
        "pair": pair,
        "sector": sector,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "rebalance_date": rebalance,
        "outer_end_exclusive": outer_end,
        "formation_start": formation_index.min() if len(formation_index) else pd.NaT,
        "formation_end": formation_index.max() if len(formation_index) else pd.NaT,
        "formation_sessions": expected,
        "aligned_formation_sessions": aligned_count,
        "history_ratio": history_ratio,
        "evaluation_start": outer_index.min() if len(outer_index) else pd.NaT,
        "evaluation_end": outer_index.max() if len(outer_index) else pd.NaT,
        "evaluation_sessions": int(len(outer_index)),
        "pair_identity_selection_is_ex_post": True,
        "full_universe_fdr_gate_applied": False,
        "fallback_used": False,
        "portfolio_objective_used": False,
    }
    minimum_ratio = float(protocol["data"]["minimum_history_ratio"])
    if (
        expected < spec.inner_train_size + spec.purge_size + spec.inner_validation_size
        or not len(outer_index)
        or history_ratio < minimum_ratio
    ):
        status = "insufficient_history"
        daily = _flat_daily(pair, rebalance, outer_end, outer_index, status, False)
        trades = pd.DataFrame(
            columns=[
                "pair",
                "rebalance_date",
                "entry_date",
                "exit_date",
                "included_core_health_gated",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    **common,
                    "status": status,
                    "core_health_pass": False,
                    "tuning_success": False,
                    "error": "formation coverage or outer window is insufficient",
                }
            ]
        )
        _write_task_cache(paths, fingerprint, daily, trades, summary)
        return daily, trades, summary, False

    diagnostic: dict[str, Any]
    try:
        integration = integration_diagnostics(
            pair_prices.loc[formation_index], maxlag=5
        )
        i1_lookup = (
            integration.set_index("ticker")["looks_i1"].fillna(False).astype(bool).to_dict()
        )
        settings = PairScreenSettings(
            minimum_observations=max(100, int(math.ceil(minimum_ratio * expected))),
            minimum_history_ratio=minimum_ratio,
            alpha=float(protocol["health"]["pair_pvalue_maximum"]),
            maximum_half_life_days=int(
                protocol["health"]["maximum_half_life_days"]
            ),
            adf_maxlag=5,
            workers=1,
        )
        diagnostic = evaluate_pair(
            ticker_a,
            ticker_b,
            sector,
            pair_prices.loc[formation_index],
            expected,
            settings,
            i1_lookup=i1_lookup,
        )
        health_pass = _core_health_pass(diagnostic, protocol["health"])
        dependent = str(diagnostic.get("dependent") or "")
        independent = str(diagnostic.get("independent") or "")
        if not dependent or not independent:
            raise ValueError("formation-only orientation is unavailable")
        definition = PairDefinition(pair, sector, dependent, independent)
        tuning = tune_pair_parameters(
            definition,
            pair_prices,
            formation_index,
            base_config,
            spec,
        )
        selected = tuning.ranking.loc[tuning.ranking["selected"].astype(bool)].iloc[0]
        evaluation_2bp = evaluate_tuned_pair(
            tuning,
            pair_prices,
            formation_index,
            outer_index,
            base_config,
            transaction_cost_bps=PRIMARY_COST_BPS,
        )
        evaluation_1bp = evaluate_tuned_pair(
            tuning,
            pair_prices,
            formation_index,
            outer_index,
            base_config,
            transaction_cost_bps=SENSITIVITY_COST_BPS,
        )
    except Exception as error:
        status = "tuning_or_evaluation_failed"
        diagnostic = locals().get("diagnostic", {})
        health_pass = _core_health_pass(diagnostic, protocol["health"])
        daily = _flat_daily(pair, rebalance, outer_end, outer_index, status, health_pass)
        trades = pd.DataFrame(
            columns=[
                "pair",
                "rebalance_date",
                "entry_date",
                "exit_date",
                "included_core_health_gated",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    **common,
                    **{
                        key: diagnostic.get(key)
                        for key in (
                            "dependent",
                            "independent",
                            "pair_pvalue",
                            "half_life_days",
                            "hurst_exponent",
                            "positive_hedge_ratio",
                            "both_look_i1",
                        )
                    },
                    "status": status,
                    "core_health_pass": health_pass,
                    "tuning_success": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            ]
        )
        _write_task_cache(paths, fingerprint, daily, trades, summary)
        return daily, trades, summary, False

    status = "tuned_health_pass" if health_pass else "tuned_health_fail"
    daily_2bp = evaluation_2bp.daily.reindex(outer_index)
    daily_1bp = evaluation_1bp.daily.reindex(outer_index)
    daily = pd.DataFrame(
        {
            "pair": pair,
            "rebalance_date": rebalance,
            "outer_end_exclusive": outer_end,
            "date": outer_index,
            "status": status,
            "core_health_pass": health_pass,
            "tuning_success": True,
            "config_id": tuning.selected_parameter.config_id,
            "zscore": daily_2bp["zscore"].to_numpy(),
            "position": daily_2bp["position"].fillna(0).to_numpy(),
            "turnover": daily_2bp["turnover"].fillna(0.0).to_numpy(),
            "transaction_cost_2bp": daily_2bp["transaction_cost"].fillna(0.0).to_numpy(),
            "borrow_cost": daily_2bp["borrow_cost"].fillna(0.0).to_numpy(),
            "return_always_attempt_2bp": daily_2bp["net_return"].fillna(0.0).to_numpy(),
            "return_core_health_gated_2bp": (
                daily_2bp["net_return"].fillna(0.0).to_numpy() if health_pass else 0.0
            ),
            "return_always_attempt_1bp": daily_1bp["net_return"].fillna(0.0).to_numpy(),
            "return_core_health_gated_1bp": (
                daily_1bp["net_return"].fillna(0.0).to_numpy() if health_pass else 0.0
            ),
        }
    )
    trades = evaluation_2bp.trades.copy()
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "pair",
                "sector",
                "trade_number",
                "entry_date",
                "exit_date",
            ]
        )
    trades.insert(1, "rebalance_date", rebalance)
    trades["config_id"] = tuning.selected_parameter.config_id
    trades["included_core_health_gated"] = bool(health_pass)
    summary = pd.DataFrame(
        [
            {
                **common,
                "dependent": diagnostic.get("dependent"),
                "independent": diagnostic.get("independent"),
                "pair_pvalue": diagnostic.get("pair_pvalue"),
                "half_life_days": diagnostic.get("half_life_days"),
                "hurst_exponent": diagnostic.get("hurst_exponent"),
                "positive_hedge_ratio": diagnostic.get("positive_hedge_ratio"),
                "both_look_i1": diagnostic.get("both_look_i1"),
                "status": status,
                "core_health_pass": health_pass,
                "tuning_success": True,
                "selected_config_id": tuning.selected_parameter.config_id,
                "zscore_method": tuning.selected_parameter.zscore_method,
                "zscore_lookback": tuning.selected_parameter.zscore_lookback,
                "zscore_min_periods": tuning.selected_parameter.zscore_min_periods,
                "entry_z": tuning.selected_parameter.entry_z,
                "exit_z": tuning.selected_parameter.exit_z,
                "maximum_holding_days": tuning.selected_parameter.maximum_holding_days,
                "training_cv_net_cagr": selected["pooled_net_cagr"],
                "training_cv_max_drawdown": selected["max_drawdown"],
                "validation_trade_count": selected["validation_trade_count"],
                "full_training_trade_count": selected["full_training_trade_count"],
                "active_fold_count": selected["active_fold_count"],
                "net_total_return_2bp": evaluation_2bp.metrics["net_total_return"],
                "net_cagr_2bp": evaluation_2bp.metrics["net_cagr"],
                "max_drawdown_2bp": evaluation_2bp.metrics["max_drawdown"],
                "trade_count": evaluation_2bp.metrics["trade_count"],
                "transaction_cost_sum_2bp": evaluation_2bp.metrics["transaction_cost_sum"],
                "borrow_cost_sum": evaluation_2bp.metrics["borrow_cost_sum"],
                "net_total_return_1bp": evaluation_1bp.metrics["net_total_return"],
                "error": "",
            }
        ]
    )
    _write_task_cache(paths, fingerprint, daily, trades, summary)
    return daily, trades, summary, False


def _compound_stitched_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound each pair/mode across adjacent vintages without resetting equity."""

    if daily.empty:
        return daily.copy()
    work = daily.sort_values(["pair", "date"], kind="mergesort").reset_index(drop=True)
    if work.duplicated(["pair", "date"]).any():
        raise ValueError("stitched daily rows contain duplicate pair/date keys")
    return_columns = [
        "return_always_attempt_2bp",
        "return_core_health_gated_2bp",
        "return_always_attempt_1bp",
        "return_core_health_gated_1bp",
    ]
    for column in return_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        suffix = column.removeprefix("return_")
        work[f"equity_{suffix}"] = work.groupby("pair", sort=False)[column].transform(
            lambda values: (1.0 + values).cumprod()
        )
        work[f"drawdown_{suffix}"] = work.groupby("pair", sort=False)[
            f"equity_{suffix}"
        ].transform(lambda values: values / values.cummax() - 1.0)
    return work


def _calendar_cagr(final_equity: float, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> float:
    years = (pd.Timestamp(end_exclusive) - pd.Timestamp(start)).days / 365.2425
    if not math.isfinite(float(final_equity)) or final_equity <= 0 or years <= 0:
        return math.nan
    return float(final_equity ** (1.0 / years) - 1.0)


def _year_cluster_cagr_interval(
    returns: pd.Series,
    dates: pd.Series,
    *,
    seed: int,
    draws: int = 10_000,
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "return": pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(),
            "year": pd.to_datetime(dates, errors="coerce", format="mixed").dt.year.to_numpy(),
        }
    ).dropna(subset=["year"])
    annual = frame.groupby("year", sort=True)["return"].apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    if len(annual) < 2 or np.any(annual.to_numpy(dtype=float) <= -1.0):
        return math.nan, math.nan
    logs = np.log1p(annual.to_numpy(dtype=float))
    generator = np.random.default_rng(seed)
    sampled = generator.choice(logs, size=(draws, len(logs)), replace=True)
    estimates = np.expm1(sampled.mean(axis=1))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _long_horizon_metrics(
    daily: pd.DataFrame,
    vintage: pd.DataFrame,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    modes = ("core_health_gated", "always_attempt")
    costs = (1, 2)
    for pair_number, (pair, group) in enumerate(daily.groupby("pair", sort=True)):
        group = group.sort_values("date")
        pair_vintages = vintage.loc[vintage["pair"].astype(str).eq(str(pair))].copy()
        pair_trades = trades.loc[trades["pair"].astype(str).eq(str(pair))].copy() if not trades.empty else trades
        for mode in modes:
            for cost in costs:
                return_column = f"return_{mode}_{cost}bp"
                equity_column = f"equity_{mode}_{cost}bp"
                returns = pd.to_numeric(group[return_column], errors="coerce").fillna(0.0)
                equity = pd.to_numeric(group[equity_column], errors="coerce")
                standard = performance_metrics(returns, equity)
                annual = pd.DataFrame(
                    {
                        "date": pd.to_datetime(group["date"], errors="coerce", format="mixed"),
                        "return": returns.to_numpy(),
                    }
                )
                annual["year"] = annual["date"].dt.year
                annual_values = annual.groupby("year", sort=True)["return"].apply(
                    lambda values: float((1.0 + values).prod() - 1.0)
                )
                for year, value in annual_values.items():
                    annual_rows.append(
                        {
                            "pair": pair,
                            "mode": mode,
                            "transaction_cost_bps": float(cost),
                            "year": int(year),
                            "net_return": float(value),
                        }
                    )
                ci_low, ci_high = _year_cluster_cagr_interval(
                    returns,
                    group["date"],
                    seed=20260901 + pair_number * 10 + cost + (0 if mode == modes[0] else 1000),
                )
                tuned = pair_vintages.get(
                    "tuning_success", pd.Series(False, index=pair_vintages.index)
                ).fillna(False).astype(bool)
                health = pair_vintages.get(
                    "core_health_pass", pd.Series(False, index=pair_vintages.index)
                ).fillna(False).astype(bool)
                active = tuned & health if mode == "core_health_gated" else tuned
                vintage_return_column = f"net_total_return_{cost}bp"
                if vintage_return_column in pair_vintages:
                    vintage_returns = pd.to_numeric(
                        pair_vintages.loc[active, vintage_return_column], errors="coerce"
                    ).dropna()
                else:
                    vintage_returns = pd.Series(dtype=float)
                if not pair_trades.empty:
                    if mode == "core_health_gated" and "included_core_health_gated" in pair_trades:
                        trade_count = int(
                            pair_trades["included_core_health_gated"].fillna(False).astype(bool).sum()
                        )
                    else:
                        trade_count = int(len(pair_trades))
                else:
                    trade_count = 0
                rows.append(
                    {
                        "pair": pair,
                        "mode": mode,
                        "transaction_cost_bps": float(cost),
                        "study_start": start,
                        "study_end": end_exclusive - pd.Timedelta(days=1),
                        "study_sessions": int(len(group)),
                        "final_equity": float(equity.iloc[-1]),
                        "net_total_return": standard["total_return"],
                        "calendar_cagr": _calendar_cagr(float(equity.iloc[-1]), start, end_exclusive),
                        "session_annualized_cagr": standard["cagr"],
                        "annual_volatility": standard["annual_volatility"],
                        "sharpe": standard["sharpe"],
                        "sortino": standard["sortino"],
                        "max_drawdown": standard["max_drawdown"],
                        "calmar": standard["calmar"],
                        "trade_count": trade_count,
                        "total_outer_windows": int(len(pair_vintages)),
                        "tuned_outer_windows": int(tuned.sum()),
                        "core_health_pass_windows": int(health.sum()),
                        "active_outer_windows": int(active.sum()),
                        "active_outer_fraction": float(active.mean()) if len(active) else math.nan,
                        "positive_active_outer_fraction": float((vintage_returns > 0).mean()) if len(vintage_returns) else math.nan,
                        "positive_calendar_year_fraction": float((annual_values > 0).mean()) if len(annual_values) else math.nan,
                        "year_cluster_cagr_ci_low": ci_low,
                        "year_cluster_cagr_ci_high": ci_high,
                        "pair_identity_selection_is_ex_post": True,
                        "portfolio_metric": False,
                    }
                )
    metrics = pd.DataFrame(rows).sort_values(
        ["mode", "transaction_cost_bps", "calendar_cagr", "pair"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    annual = pd.DataFrame(annual_rows)
    return metrics.reset_index(drop=True), annual


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e6ebf2", linewidth=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _chart_decodable(path: Path) -> bool:
    try:
        image = mpimg.imread(path)
    except Exception:
        return False
    return bool(image.size and image.shape[0] > 0 and image.shape[1] > 0)


def _plot_equity_and_drawdown(daily: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(13.8, 8.2), sharex=True)
    colors = plt.get_cmap("tab10")(np.arange(daily["pair"].nunique()) % 10)
    for (pair, group), color in zip(daily.groupby("pair", sort=True), colors):
        group = group.sort_values("date")
        label = str(pair).replace("__", " / ")
        axes[0].plot(
            group["date"], group["equity_core_health_gated_2bp"],
            label=label, color=color, linewidth=1.7,
        )
        axes[1].plot(
            group["date"], group["drawdown_core_health_gated_2bp"],
            label=label, color=color, linewidth=1.25,
        )
    axes[0].axhline(1.0, color="#abb6c5", linewidth=0.9)
    axes[0].set_ylabel("Growth of $1 pair sleeve")
    axes[0].set_title("Ten-year stitched conditional OOS equity · core-health gated · 2 bp", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].axhline(0.0, color="#abb6c5", linewidth=0.9)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].set_xlabel("Outer-test date")
    axes[1].set_title("Continuous drawdown across non-overlapping quarterly vintages", loc="left")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(h_pad=2.0)
    return _save_figure(figure, path)


def _plot_mode_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    work = metrics.loc[metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    pivot = work.pivot(index="pair", columns="mode", values="calendar_cagr")
    order = pivot.get("core_health_gated", pd.Series(index=pivot.index, dtype=float)).sort_values(ascending=False).index
    pivot = pivot.reindex(order)
    positions = np.arange(len(pivot))
    width = 0.38
    figure, axis = plt.subplots(figsize=(13.4, 5.8))
    axis.bar(
        positions - width / 2,
        pivot.get("core_health_gated", pd.Series(0.0, index=pivot.index)),
        width,
        label="Core-health gated",
        color=TEAL,
        alpha=0.86,
    )
    axis.bar(
        positions + width / 2,
        pivot.get("always_attempt", pd.Series(0.0, index=pivot.index)),
        width,
        label="Always attempt when tunable",
        color=ORANGE,
        alpha=0.82,
    )
    axis.axhline(0.0, color="#abb6c5", linewidth=0.9)
    axis.set_xticks(positions, [str(value).replace("__", " / ") for value in pivot.index], rotation=25, ha="right")
    axis.set_ylabel("Exact-calendar CAGR")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    axis.set_title("Long-horizon result with and without the training-only core health gate", loc="left")
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_annual_heatmap(annual: pd.DataFrame, path: Path) -> Path:
    work = annual.loc[
        annual["mode"].eq("core_health_gated")
        & annual["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    pivot = work.pivot(index="pair", columns="year", values="net_return").sort_index()
    figure, axis = plt.subplots(figsize=(14.2, 5.6))
    values = pivot.to_numpy(dtype=float)
    maximum = max(float(np.nanmax(np.abs(values))) if values.size else 0.0, 0.01)
    image = axis.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-maximum, vmax=maximum)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(str), rotation=35, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), [str(value).replace("__", " / ") for value in pivot.index])
    axis.set_title("Calendar-year net returns · core-health gated · 2 bp", loc="left")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=7.5, color=INK)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_activity_ribbon(vintage: pd.DataFrame, path: Path) -> Path:
    pairs = sorted(vintage["pair"].astype(str).unique())
    dates = sorted(pd.to_datetime(vintage["rebalance_date"], errors="coerce").dropna().unique())
    matrix = np.zeros((len(pairs), len(dates)), dtype=float)
    pair_index = {pair: location for location, pair in enumerate(pairs)}
    date_index = {pd.Timestamp(date): location for location, date in enumerate(dates)}
    for row in vintage.itertuples(index=False):
        value = 0.0
        if bool(getattr(row, "tuning_success", False)):
            value = 2.0 if bool(getattr(row, "core_health_pass", False)) else 1.0
        matrix[pair_index[str(row.pair)], date_index[pd.Timestamp(row.rebalance_date)]] = value
    from matplotlib.colors import ListedColormap

    figure, axis = plt.subplots(figsize=(14.3, 5.4))
    axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#e8edf3", ORANGE, GREEN]),
        vmin=0,
        vmax=2,
    )
    tick_step = max(1, len(dates) // 12)
    tick_locations = np.arange(0, len(dates), tick_step)
    axis.set_xticks(tick_locations, [pd.Timestamp(dates[index]).strftime("%Y-%m") for index in tick_locations], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(pairs)), [pair.replace("__", " / ") for pair in pairs])
    axis.set_title("Quarterly availability and pair-health state", loc="left")
    axis.set_xlabel("Rebalance date")
    axis.text(
        0.0, -0.20,
        "Grey = unavailable/tuning failed · orange = tunable but core health failed · green = traded in the primary gated curve",
        transform=axis.transAxes, color=MUTED, fontsize=9,
    )
    figure.tight_layout()
    return _save_figure(figure, path)


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}%}" if math.isfinite(number) else "—"


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _display_long_metrics(metrics: pd.DataFrame, mode: str) -> pd.DataFrame:
    work = metrics.loc[
        metrics["mode"].eq(mode)
        & metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ].copy()
    work = work.sort_values(
        ["calendar_cagr", "max_drawdown", "pair"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    work.insert(0, "rank", np.arange(1, len(work) + 1))
    columns = [
        "rank",
        "pair",
        "net_total_return",
        "calendar_cagr",
        "max_drawdown",
        "sharpe",
        "trade_count",
        "active_outer_windows",
        "total_outer_windows",
        "positive_calendar_year_fraction",
        "year_cluster_cagr_ci_low",
        "year_cluster_cagr_ci_high",
    ]
    work = work[columns]
    work["pair"] = work["pair"].astype("string").str.replace("__", " / ", regex=False)
    for column in (
        "net_total_return",
        "calendar_cagr",
        "max_drawdown",
        "positive_calendar_year_fraction",
        "year_cluster_cagr_ci_low",
        "year_cluster_cagr_ci_high",
    ):
        work[column] = work[column].map(_pct)
    work["sharpe"] = work["sharpe"].map(_num)
    work["active_windows"] = (
        work["active_outer_windows"].astype(int).astype(str)
        + " / "
        + work["total_outer_windows"].astype(int).astype(str)
    )
    return work.drop(columns=["active_outer_windows", "total_outer_windows"])


def _html_table(frame: pd.DataFrame, max_rows: int = 100) -> str:
    if frame.empty:
        return '<p class="muted">No rows available.</p>'
    return frame.head(max_rows).to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
        na_rep="—",
    )


REPORT_TEMPLATE = Template(
    r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ten-year conditional pair-leader replay</title>
<style>
:root{--ink:#172033;--muted:#667188;--blue:#2f67c7;--teal:#138a82;--green:#2a845e;--orange:#da7a2e;--red:#bf4c55;--line:#dce3ec;--pale:#edf3f8;--paper:#fff}*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#edf1f6;color:var(--ink);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{width:100%;max-width:1320px;min-width:0;margin:24px auto;background:var(--paper);box-shadow:0 18px 55px rgba(30,45,70,.12)}header{padding:58px 7% 46px;background:linear-gradient(125deg,#12213b,#274f83);color:#fff}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;font-weight:750;color:#bcd3f2}h1{font-size:42px;line-height:1.08;margin:13px 0 15px;max-width:1030px}header p{font-size:18px;color:#dce8f7;max-width:1030px;margin:0}.content{min-width:0;padding:46px 7% 72px}.content>*,.grid2>*{min-width:0}.verdict{padding:20px 23px;border-left:5px solid var(--orange);background:#fff8ef;margin-bottom:28px}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px;margin:0 0 38px}.card{min-width:0;border:1px solid var(--line);padding:17px;background:#fff}.card .k{text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:10px;font-weight:750}.card .v{font-size:27px;font-weight:780;margin:3px 0}.card .s{font-size:12px;color:var(--muted)}h2{font-size:28px;line-height:1.2;margin:48px 0 15px;border-top:1px solid var(--line);padding-top:31px}h3{font-size:18px;margin:22px 0 9px}.lead{font-size:17px;color:#354258}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.chart{width:100%;display:block;border:1px solid var(--line);margin:18px 0 25px;background:#fff}.note{padding:16px 19px;background:var(--pale);border-left:4px solid var(--blue);margin:18px 0}.note.warn{background:#fff8ef;border-color:var(--orange)}.note.good{background:#eef8f4;border-color:var(--green)}.scroll{width:100%;max-width:100%;min-width:0;overflow-x:auto;border:1px solid var(--line);margin:12px 0 24px}.data-table{border-collapse:collapse;width:100%;font-size:12px}.data-table th{background:#f2f5f9;text-align:left;padding:9px 10px;white-space:nowrap;border-bottom:1px solid var(--line)}.data-table td{padding:8px 10px;border-bottom:1px solid #e7ecf2;white-space:nowrap}.muted,.small{color:var(--muted)}.small{font-size:12px}a{color:var(--blue)}ul{padding-left:21px}.foot{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
@media(max-width:960px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2{grid-template-columns:1fr}h1{font-size:35px}}@media(max-width:540px){.page{margin:0}.content{padding:32px 5% 50px}header{padding:42px 5% 34px}.cards{grid-template-columns:1fr}h1{font-size:31px}}@media print{@page{size:A4 landscape;margin:10mm}body{background:#fff}.page{box-shadow:none;margin:0;max-width:none}.chart,.note,.card{break-inside:avoid}}
</style></head><body><main class="page"><header><div class="eyebrow">Equity pairs · conditional stability · generated {{ generated }}</div>
<h1>Ten calendar years of sequential pair returns, with every quarter trained only on its preceding three years.</h1>
<p>The return path spans 2016-08-26 through 2026-08-26. A separate thirteen-year Yahoo snapshot supplies the required three-year prehistory.</p></header><section class="content">
<div class="verdict"><strong>Post-selection stability evidence—not an unbiased alpha estimate.</strong> The pair identities were chosen after viewing the V6 outer winners. Hedge orientation, coefficients and hyperparameters are chronological, but pair membership is not.</div>
<div class="cards"><div class="card"><div class="k">Unique identities</div><div class="v">{{ identity_count }}</div><div class="s">from ten realized pair-vintages</div></div><div class="card"><div class="k">OOS calendar span</div><div class="v">10.0y</div><div class="s">after separate three-year prehistory</div></div><div class="card"><div class="k">Outer windows</div><div class="v">{{ outer_count }}</div><div class="s">adjacent; no OOS date reused</div></div><div class="card"><div class="k">Top gated CAGR</div><div class="v">{{ top_cagr }}</div><div class="s">2 bp plus 30 bp borrow proxy</div></div><div class="card"><div class="k">QA</div><div class="v">{{ qa_status }}</div><div class="s">implementation checks, not deployment approval</div></div></div>

<h2>1 · What overlaps—and what does not</h2><p class="lead">The frozen outer windows in this replay are adjacent quarterly blocks. A daily return appears exactly once in each pair's stitched curve.</p><ul><li>Successive three-year training samples overlap by about 33 of 36 months, so parameter estimates are dependent.</li><li>Pairs traded in the same quarter share market shocks, and some pairs share an underlying.</li><li>A one-year test launched every quarter would overlap 75% with the next annual vintage; this report does not use that design.</li></ul>

<h2>2 · Primary ten-year pair results</h2><p>The primary curve trades only when the fixed identity has sufficient history, the 97-candidate tuner succeeds, and the formation-only core pair-health gates pass. A failed or unavailable quarter holds cash. These are pair sleeves, not a portfolio.</p><div class="scroll">{{ primary_table | safe }}</div><div class="note warn"><strong>Selection caveat:</strong> the original top-ten performance table contains only six identities because CRH / SW and AVY / SW repeat across quarters. Repetition is removed here. CVNA did not exist for the beginning of the requested period, so AMZN / CVNA visibly remains flat until sufficient three-year history exists.</div>

<h2>3 · Continuous equity and drawdown</h2><img class="chart" src="{{ equity_chart }}" alt="Ten-year core-health-gated pair equity and drawdown curves"><p class="small">Equity is compounded across adjacent quarterly tests. Per-quarter sleeve equity is never averaged or restarted in the displayed long curve.</p>

<h2>4 · What the health gate changes</h2><img class="chart" src="{{ mode_chart }}" alt="CAGR comparison with and without the core pair-health gate"><div class="grid2"><div><h3>Core-health gated</h3><div class="scroll">{{ primary_table | safe }}</div></div><div><h3>Always attempt when tunable</h3><div class="scroll">{{ secondary_table | safe }}</div></div></div><p class="small">The core gate uses formation-only pair p-value, I(1), positive hedge ratio, half-life and Hurst checks. It is not the original full-universe within-sector BH-FDR gate; claiming that would require rerunning every sector pair on the thirteen-year panel.</p>

<h2>5 · Year-by-year consistency</h2><img class="chart" src="{{ annual_chart }}" alt="Calendar-year pair return heatmap"><p>The confidence intervals in the tables resample calendar-year return clusters. The endpoint years are partial and the roughly ten calendar clusters remain a small sample, so the intervals are descriptive rather than asymptotically reliable.</p>

<h2>6 · Availability, tuning and health</h2><img class="chart" src="{{ activity_chart }}" alt="Quarterly pair availability and health ribbon"><p>No fallback parameter or replacement pair is inserted. Unavailable, failed or unhealthy vintages remain cash in the primary curve.</p>

<h2>7 · Interpretation</h2><div class="grid2"><div><h3>What this answers</h3><ul><li>Whether the currently highlighted identities remain profitable across a much longer sequence of unseen quarters.</li><li>Whether results survive 2 bp turnover cost, the 30 bp annual borrow proxy and quarterly liquidation.</li><li>How returns vary by calendar year and pair-health state.</li></ul></div><div><h3>What this cannot prove</h3><ul><li>That these identities could have been chosen in 2016; they were identified ex-post.</li><li>That current S&amp;P 500 membership and GICS labels are point-in-time correct.</li><li>That Yahoo closes represent executable prices or historical borrow availability.</li><li>That the pair-level health gate is equivalent to sector-wide FDR screening.</li></ul></div></div>

<h2>8 · Artifacts</h2><div class="grid2"><div><ul>{{ artifact_links | safe }}</ul></div><div><ul><li><a href="leader_10y_quality_checks.json">Quality checks</a></li><li><a href="leader_10y_manifest.json">Manifest and SHA-256 hashes</a></li></ul></div></div><div class="note {{ qa_class }}"><strong>QA status: {{ qa_status }}.</strong> A pass certifies chronology, non-overlap, coverage and artifact reconciliation—not economic significance.</div><div class="foot">Conditional quantitative research only; not investment advice. The original V6 report and artifacts were not overwritten.</div>
</section></main></body></html>"""
)


def _render_report(
    report_path: Path,
    metrics: pd.DataFrame,
    identities: pd.DataFrame,
    schedule: pd.DataFrame,
    charts: Mapping[str, Path],
    quality: Mapping[str, Any],
    processed_files: Mapping[str, Path],
) -> None:
    primary = metrics.loc[
        metrics["mode"].eq("core_health_gated")
        & metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ].sort_values("calendar_cagr", ascending=False)
    top_cagr = _pct(primary["calendar_cagr"].max()) if len(primary) else "—"
    links = "".join(
        f'<li><a href="{path.relative_to(report_path.parent).as_posix()}">{label}</a></li>'
        for label, path in processed_files.items()
    )
    passed = bool(quality.get("critical_pass", False))
    html = REPORT_TEMPLATE.render(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        identity_count=len(identities),
        outer_count=len(schedule),
        top_cagr=top_cagr,
        qa_status="PASS" if passed else "FAILED",
        qa_class="good" if passed else "warn",
        primary_table=_html_table(_display_long_metrics(metrics, "core_health_gated")),
        secondary_table=_html_table(_display_long_metrics(metrics, "always_attempt")),
        equity_chart=charts["equity"].relative_to(report_path.parent).as_posix(),
        mode_chart=charts["mode"].relative_to(report_path.parent).as_posix(),
        annual_chart=charts["annual"].relative_to(report_path.parent).as_posix(),
        activity_chart=charts["activity"].relative_to(report_path.parent).as_posix(),
        artifact_links=links,
    )
    _atomic_text(report_path, html)


def _quality_checks(
    config: Mapping[str, Any],
    prices: pd.DataFrame,
    identities: pd.DataFrame,
    schedule: pd.DataFrame,
    daily: pd.DataFrame,
    vintage: pd.DataFrame,
    metrics: pd.DataFrame,
    charts: Mapping[str, Path],
    report_path: Path,
) -> dict[str, Any]:
    start = pd.Timestamp(config["data"]["oos_start_inclusive"])
    end = pd.Timestamp(config["data"]["end_exclusive"])
    study_index = prices.index[(prices.index >= start) & (prices.index < end)]
    expected_daily = int(len(study_index) * len(identities))
    schedule_contiguous = bool(
        len(schedule) == 1
        or all(
            pd.Timestamp(schedule.iloc[location - 1]["outer_end_exclusive"])
            == pd.Timestamp(schedule.iloc[location]["rebalance_date"])
            for location in range(1, len(schedule))
        )
    )
    daily_unique = bool(not daily.duplicated(["pair", "date"]).any())
    daily_coverage = bool(len(daily) == expected_daily)
    each_pair_coverage = bool(
        daily.groupby("pair")["date"].nunique().eq(len(study_index)).all()
    ) if len(daily) else False
    formation_precedes_outer = bool(
        vintage["formation_end"].dropna().lt(
            vintage.loc[vintage["formation_end"].notna(), "rebalance_date"]
        ).all()
    ) if len(vintage) else False
    no_fallback = bool(
        "fallback_used" in vintage
        and not vintage["fallback_used"].fillna(False).astype(bool).any()
    )
    ex_post_declared = bool(
        identities["identity_selection_timing"].eq(
            "ex_post_after_v6_outer_returns"
        ).all()
        and not identities["deployable_selection"].fillna(True).astype(bool).any()
    )
    no_portfolio = bool(
        "portfolio_metric" in metrics
        and not metrics["portfolio_metric"].fillna(True).astype(bool).any()
        and "portfolio_objective_used" in vintage
        and not vintage["portfolio_objective_used"].fillna(True).astype(bool).any()
    )
    candidate_count = len(
        protocol_parameter_sets(
            *_build_protocol(config)[:1],
            stop_z=float(config["hyperparameter_grid"]["stop_z"]),
        )
    )
    equity_reconciles = True
    for _, row in metrics.iterrows():
        pair_daily = daily.loc[daily["pair"].astype(str).eq(str(row["pair"]))]
        column = f"equity_{row['mode']}_{int(row['transaction_cost_bps'])}bp"
        if pair_daily.empty or column not in pair_daily:
            equity_reconciles = False
            break
        actual = float(pair_daily.sort_values("date")[column].iloc[-1])
        if not math.isclose(actual, float(row["final_equity"]), rel_tol=0.0, abs_tol=1e-12):
            equity_reconciles = False
            break
    sensitivity_monotonic = True
    pivot = metrics.pivot_table(
        index=["pair", "mode"],
        columns="transaction_cost_bps",
        values="final_equity",
        aggfunc="first",
    )
    if {SENSITIVITY_COST_BPS, PRIMARY_COST_BPS}.issubset(pivot.columns):
        sensitivity_monotonic = bool(
            pivot[SENSITIVITY_COST_BPS].ge(pivot[PRIMARY_COST_BPS] - 1e-12).all()
        )
    chart_count = sum(path.is_file() for path in charts.values())
    decodable_count = sum(_chart_decodable(path) for path in charts.values())
    checks: dict[str, Any] = {
        "quality_scope": "Chronology, non-overlap, cost freeze and artifact reconciliation; not profitability or selection-bias approval.",
        "identity_count": int(len(identities)),
        "outer_window_count": int(len(schedule)),
        "study_session_count": int(len(study_index)),
        "expected_daily_rows": expected_daily,
        "actual_daily_rows": int(len(daily)),
        "outer_windows_contiguous_and_nonoverlapping": schedule_contiguous,
        "outer_schedule_covers_study_sessions": bool(schedule["outer_sessions"].sum() == len(study_index)),
        "daily_pair_date_keys_unique": daily_unique,
        "daily_rows_cover_every_pair_and_study_session": daily_coverage and each_pair_coverage,
        "formation_always_precedes_outer_test": formation_precedes_outer,
        "pair_identity_selection_explicitly_ex_post": ex_post_declared,
        "no_fallback_used": no_fallback,
        "portfolio_objective_and_metrics_absent": no_portfolio,
        "tuning_candidate_count": candidate_count,
        "exactly_97_tuning_candidates": candidate_count == 97,
        "stitched_equity_reconciles": equity_reconciles,
        "one_bp_final_equity_not_below_two_bp": sensitivity_monotonic,
        "full_universe_fdr_claimed": False,
        "report_exists": report_path.is_file(),
        "report_chart_count": int(chart_count),
        "report_chart_decodable_count": int(decodable_count),
        "expected_report_chart_count": int(len(charts)),
    }
    critical = [
        schedule_contiguous,
        checks["outer_schedule_covers_study_sessions"],
        daily_unique,
        daily_coverage,
        each_pair_coverage,
        formation_precedes_outer,
        ex_post_declared,
        no_fallback,
        no_portfolio,
        candidate_count == 97,
        equity_reconciles,
        sensitivity_monotonic,
        report_path.is_file(),
        chart_count == len(charts),
        decodable_count == len(charts),
    ]
    checks["critical_pass"] = bool(all(critical))
    checks["status"] = "pass" if checks["critical_pass"] else "failed"
    return checks


def _build_manifest(
    root: Path,
    config_path: Path,
    source_paths: Mapping[str, Path],
    generated_paths: list[Path],
) -> dict[str, Any]:
    def record(path: Path) -> dict[str, Any]:
        return {
            "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "pair_leader_10y_conditional_replay",
        "pair_identity_selection_is_ex_post": True,
        "parameter_and_hedge_estimation_is_chronological": True,
        "outer_windows_overlap": False,
        "portfolio_metric_computed": False,
        "full_universe_fdr_gate_applied": False,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "source_artifacts": {
            name: record(path) for name, path in source_paths.items() if path.is_file()
        },
        "generated_artifacts": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in generated_paths
            if path.is_file() and path.is_relative_to(root)
        },
        "credentials_included": False,
    }


def run_replay(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    refresh_data: bool = False,
    recompute: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    spec, base_config = _build_protocol(config)
    root = Path(output_root).resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    charts_dir = root / "charts"
    cache_root = processed / "vintage_cache"
    for directory in (raw, processed, charts_dir, cache_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_root = Path(config["source"]["v6_output_root"]).resolve()
    source_metrics_path = source_root / config["source"]["v6_pair_vintage_metrics"]
    source_metrics = _read_csv(source_metrics_path, dates=("rebalance_date",))
    top_vintages, identities = select_source_leaders(
        source_metrics,
        int(config["source"]["leader_count_before_identity_deduplication"]),
    )
    top_path = processed / "leader_source_top10_pair_vintages.csv"
    identities_path = processed / "leader_unique_pair_identities.csv"
    _atomic_csv(top_vintages, top_path)
    _atomic_csv(identities, identities_path)

    symbols = sorted(
        set(identities["ticker_a"].astype(str))
        | set(identities["ticker_b"].astype(str))
    )
    price_path = raw / str(config["data"]["price_snapshot_name"])
    prices = download_adjusted_close(
        symbols,
        config["data"]["download_start_inclusive"],
        config["data"]["end_exclusive"],
        chunk_size=max(1, len(symbols)),
        max_retries=3,
        backoff_factor=1.0,
        inter_chunk_delay=0.0,
        repair=bool(config["data"]["repair_prices"]),
        strict=False,
        cache_path=price_path,
        refresh=refresh_data,
    ).sort_index()
    verified_prices, price_metadata = load_timeseries_snapshot(price_path)
    if (
        not verified_prices.index.equals(prices.index)
        or list(verified_prices.columns) != list(prices.columns)
    ):
        raise RuntimeError("verified leader price snapshot differs from the loaded shape")
    if prices.index.min() > pd.Timestamp(config["data"]["download_start_inclusive"]) + pd.Timedelta(days=7):
        raise RuntimeError("leader price snapshot lacks the required three-year prehistory")
    schedule = build_nonoverlapping_schedule(
        prices.index,
        config["data"]["oos_start_inclusive"],
        config["data"]["end_exclusive"],
    )
    schedule_path = processed / "leader_rebalance_schedule.csv"
    _atomic_csv(schedule, schedule_path)

    tasks: list[dict[str, Any]] = []
    for identity in identities.to_dict(orient="records"):
        for window in schedule.to_dict(orient="records"):
            tasks.append(
                {
                    "pair": identity["pair"],
                    "sector": identity["sector"],
                    "ticker_a": identity["ticker_a"],
                    "ticker_b": identity["ticker_b"],
                    "rebalance_date": window["rebalance_date"],
                    "outer_end_exclusive": window["outer_end_exclusive"],
                }
            )
    initializer_args = (
        prices,
        spec,
        base_config,
        config,
        str(cache_root),
        str(price_metadata["sha256"]),
        _sha256(config_path),
        recompute,
    )
    results: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]] = []
    if workers == 1:
        _initialize_worker(*initializer_args)
        for location, task in enumerate(tasks, start=1):
            results.append(_run_vintage_task(task))
            print(f"leader replay {location}/{len(tasks)} · {task['pair']} · {pd.Timestamp(task['rebalance_date']).date()}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            initializer=_initialize_worker,
            initargs=initializer_args,
        ) as executor:
            futures = [executor.submit(_run_vintage_task, task) for task in tasks]
            for location, (task, future) in enumerate(zip(tasks, futures), start=1):
                results.append(future.result())
                print(f"leader replay {location}/{len(tasks)} · {task['pair']} · {pd.Timestamp(task['rebalance_date']).date()}", flush=True)

    daily = pd.concat([result[0] for result in results], ignore_index=True)
    trade_parts = [result[1] for result in results if not result[1].empty]
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    vintage = pd.concat([result[2] for result in results], ignore_index=True)
    daily = _compound_stitched_daily(daily)
    start = pd.Timestamp(config["data"]["oos_start_inclusive"])
    end = pd.Timestamp(config["data"]["end_exclusive"])
    metrics, annual = _long_horizon_metrics(daily, vintage, trades, start, end)
    failures = vintage.loc[~vintage["tuning_success"].fillna(False).astype(bool)].copy()

    daily_path = processed / "leader_10y_oos_signals.csv.gz"
    trades_path = processed / "leader_10y_oos_trades.csv"
    vintage_path = processed / "leader_10y_vintage_metrics.csv"
    metrics_path = processed / "leader_10y_pair_metrics.csv"
    annual_path = processed / "leader_10y_annual_returns.csv"
    failures_path = processed / "leader_10y_failures_and_unavailable.csv"
    _atomic_csv(daily, daily_path)
    _atomic_csv(trades, trades_path)
    _atomic_csv(vintage, vintage_path)
    _atomic_csv(metrics, metrics_path)
    _atomic_csv(annual, annual_path)
    _atomic_csv(failures, failures_path)

    charts = {
        "equity": _plot_equity_and_drawdown(
            daily, charts_dir / "leader_10y_equity_and_drawdown.png"
        ),
        "mode": _plot_mode_comparison(
            metrics, charts_dir / "leader_10y_health_gate_comparison.png"
        ),
        "annual": _plot_annual_heatmap(
            annual, charts_dir / "leader_10y_annual_return_heatmap.png"
        ),
        "activity": _plot_activity_ribbon(
            vintage, charts_dir / "leader_10y_activity_ribbon.png"
        ),
    }
    report_path = root / str(config["outputs"]["report_name"])
    quality_path = root / "leader_10y_quality_checks.json"
    manifest_path = root / "leader_10y_manifest.json"
    processed_files = {
        "Source realized top-ten pair-vintages": top_path,
        "Unique leader identities": identities_path,
        "Non-overlapping rebalance schedule": schedule_path,
        "Long-horizon pair metrics": metrics_path,
        "Quarterly vintage metrics and parameters": vintage_path,
        "Daily OOS signals and stitched curves": daily_path,
        "Trade ledger": trades_path,
        "Calendar-year returns": annual_path,
        "Failures and unavailable windows": failures_path,
    }
    preliminary = {
        "critical_pass": False,
    }
    _render_report(
        report_path, metrics, identities, schedule, charts, preliminary, processed_files
    )
    quality = _quality_checks(
        config, prices, identities, schedule, daily, vintage, metrics, charts, report_path
    )
    _atomic_json(quality_path, quality)
    _render_report(
        report_path, metrics, identities, schedule, charts, quality, processed_files
    )
    generated_paths = [
        report_path,
        quality_path,
        *charts.values(),
        *processed_files.values(),
    ]
    manifest = _build_manifest(
        root,
        config_path,
        {
            "v6_pair_vintage_metrics": source_metrics_path,
            "leader_yahoo_13y_snapshot": price_path,
            "leader_yahoo_13y_metadata": price_path.with_suffix(price_path.suffix + ".meta.json"),
        },
        generated_paths,
    )
    _atomic_json(manifest_path, manifest)
    return {
        "output_root": root,
        "report": report_path,
        "quality_checks": quality_path,
        "manifest": manifest_path,
        "metrics": metrics_path,
        "daily": daily_path,
        "quality": quality,
        "identity_count": len(identities),
        "task_count": len(tasks),
        "cache_reused_count": int(sum(result[3] for result in results)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the additive ten-year conditional replay for V6 leaders."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_replay(
        args.config,
        args.output,
        refresh_data=args.refresh_data,
        recompute=args.recompute,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "report": str(result["report"]),
                "quality_checks": str(result["quality_checks"]),
                "manifest": str(result["manifest"]),
                "critical_pass": bool(result["quality"]["critical_pass"]),
                "identity_count": int(result["identity_count"]),
                "task_count": int(result["task_count"]),
                "cache_reused_count": int(result["cache_reused_count"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
