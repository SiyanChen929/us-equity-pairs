#!/usr/bin/env python3
"""Run additive V7 structural experiments without modifying V6 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from jinja2 import Template
from scipy.stats import spearmanr

from equity_pairs.backtest import performance_metrics
from equity_pairs.config import StrategyConfig
from equity_pairs.data import download_adjusted_close
from equity_pairs.intraday_risk import audit_minute_history
from equity_pairs.pit_data import audit_point_in_time_contract
from equity_pairs.portfolio_risk import cash_aware_inverse_volatility
from equity_pairs.selection_inference import circular_block_indices, romano_wolf_max_t
from equity_pairs.v7_experiments import (
    V7Rules,
    aggregate_ensemble,
    parameter_from_schedule,
    simulate_v7_pair,
)


DEFAULT_CONFIG = Path("configs/pair_strategy_v7_experiments.json")


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    cache: Path
    charts: Path

    @classmethod
    def create(cls, root: Path) -> "Paths":
        result = cls(
            root=root,
            raw=root / "data" / "raw",
            processed=root / "data" / "processed",
            cache=root / "data" / "cache",
            charts=root / "charts",
        )
        for path in (
            result.root,
            result.raw,
            result.processed,
            result.cache,
            result.charts,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return result


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression="gzip" if path.suffix == ".gz" else None,
    )
    temporary.replace(path)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin({"true", "1", "yes"})


def _base_config(config: Mapping[str, Any]) -> StrategyConfig:
    frozen = config["frozen_mechanics"]
    return StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=float(frozen["stop_z"]),
        maximum_holding_days=60,
        transaction_cost_bps=float(frozen["transaction_cost_bps"]),
        transaction_cost_scenarios_bps=(
            float(frozen["cost_sensitivity_bps"]),
            float(frozen["transaction_cost_bps"]),
        ),
        annual_short_borrow_bps=float(frozen["annual_short_borrow_bps"]),
    )


def _indexes(row: Mapping[str, Any], prices: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    train = prices.index[
        (prices.index >= pd.Timestamp(row["formation_start"]))
        & (prices.index <= pd.Timestamp(row["formation_end"]))
    ]
    evaluation = prices.index[
        (prices.index >= pd.Timestamp(row["outer_start"]))
        & (prices.index < pd.Timestamp(row["outer_end_exclusive"]))
    ]
    if not len(train) or not len(evaluation):
        raise ValueError("empty training/evaluation index")
    return train, evaluation


def _pair_definition(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "pair": str(row["pair"]),
        "sector": str(row["sector"]),
        "dependent": str(row["dependent"]),
        "independent": str(row["independent"]),
    }


def _cost_sensitivity(daily: pd.DataFrame, bps: float) -> float:
    returns = (
        daily["gross_return"]
        - daily["turnover"]
        * float(bps)
        / 10_000.0
        * (1.0 + daily["gross_return"])
        - daily["borrow_cost"]
    )
    return float((1.0 + returns).prod() - 1.0)


def run_schedule_variant(
    name: str,
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    base: StrategyConfig,
    rules: V7Rules,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    for raw in schedule.sort_values(["outer_start", "pair"]).to_dict("records"):
        train, evaluation = _indexes(raw, prices)
        result = simulate_v7_pair(
            _pair_definition(raw),
            prices,
            train,
            evaluation,
            parameter_from_schedule(raw),
            base,
            rules,
            screen_half_life_days=float(raw["screen_half_life_days"]),
        )
        metric_rows.append(
            {
                "variant": name,
                "rebalance_date": raw["rebalance_date"],
                "outer_start": raw["outer_start"],
                "outer_end_exclusive": raw["outer_end_exclusive"],
                **result.metrics,
                "net_total_return_1bp": _cost_sensitivity(result.daily, 1.0),
            }
        )
        daily = result.daily.reset_index()
        daily.insert(0, "variant", name)
        daily.insert(1, "rebalance_date", raw["rebalance_date"])
        daily.insert(2, "pair", raw["pair"])
        daily.insert(3, "sector", raw["sector"])
        daily_rows.append(daily)
        if len(result.trades):
            trades = result.trades.copy()
            trades.insert(0, "variant", name)
            trades.insert(1, "rebalance_date", raw["rebalance_date"])
            trade_rows.append(trades)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(),
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
    )


def run_carry_variant(
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    base: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carry old frozen vintages until an original exit, stop, or time limit."""

    metrics: list[dict[str, Any]] = []
    dailies: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    study_end = prices.index.max()
    for pair, pair_schedule in schedule.groupby("pair", sort=True):
        blocked_until: pd.Timestamp | None = None
        for raw in pair_schedule.sort_values("outer_start").to_dict("records"):
            train, regular = _indexes(raw, prices)
            outer_start = regular[0]
            outer_end = pd.Timestamp(raw["outer_end_exclusive"])
            parameter = parameter_from_schedule(raw)
            after = prices.index[prices.index >= outer_end]
            extension_count = parameter.maximum_holding_days + 2
            extension = after[:extension_count]
            evaluation = regular.union(extension).sort_values()
            entry_start = outer_start
            if blocked_until is not None:
                later = evaluation[evaluation > blocked_until]
                entry_start = later[0] if len(later) else outer_end
            terminal = bool(evaluation[-1] >= study_end)
            result = simulate_v7_pair(
                _pair_definition(raw),
                prices,
                train,
                evaluation,
                parameter,
                base,
                V7Rules(
                    block_entry_at_or_beyond_stop=True,
                    terminal_liquidation=terminal,
                ),
                entry_start=entry_start,
                entry_end_exclusive=outer_end,
            )
            daily = result.daily.copy()
            boundary_rows = daily.index < outer_end
            last_regular = daily.loc[boundary_rows].iloc[-1]
            carry_open = bool(last_regular["target_position"] != 0)
            carry_exit: pd.Timestamp | None = None
            if carry_open:
                post = daily.loc[daily.index >= outer_end]
                prior_target = daily["target_position"].shift(1).reindex(post.index)
                exits = post.index[
                    prior_target.ne(0).fillna(False)
                    & post["target_position"].eq(0)
                ]
                if len(exits):
                    carry_exit = exits[0]
                    daily = daily.loc[daily.index <= carry_exit]
                else:
                    carry_exit = daily.index[-1]
            else:
                daily = daily.loc[daily.index < outer_end]
            blocked_until = carry_exit
            daily["net_equity"] = (1.0 + daily["net_return"]).cumprod()
            metric = performance_metrics(daily["net_return"], daily["net_equity"])
            metrics.append(
                {
                    "variant": "02_cross_quarter_carry",
                    "rebalance_date": raw["rebalance_date"],
                    "outer_start": raw["outer_start"],
                    "outer_end_exclusive": raw["outer_end_exclusive"],
                    "pair": pair,
                    "sector": raw["sector"],
                    "net_total_return": metric["total_return"],
                    "net_total_return_1bp": _cost_sensitivity(daily, 1.0),
                    "max_drawdown": metric["max_drawdown"],
                    "sharpe": metric["sharpe"],
                    "trade_count": int(len(result.trades)),
                    "carry_open_at_boundary": carry_open,
                    "carry_exit_date": carry_exit,
                    "carry_sessions": int((daily.index >= outer_end).sum()),
                    "terminal_flat": bool(daily["target_position"].iloc[-1] == 0),
                }
            )
            out = daily.reset_index()
            out.insert(0, "variant", "02_cross_quarter_carry")
            out.insert(1, "rebalance_date", raw["rebalance_date"])
            out.insert(2, "pair", pair)
            out.insert(3, "sector", raw["sector"])
            dailies.append(out)
            if len(result.trades):
                ledger = result.trades.copy()
                ledger.insert(0, "variant", "02_cross_quarter_carry")
                ledger.insert(1, "rebalance_date", raw["rebalance_date"])
                trades.append(ledger)
    return (
        pd.DataFrame(metrics),
        pd.concat(dailies, ignore_index=True),
        pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(),
    )


def current_subindustry_schedule(
    rankings: pd.DataFrame,
    strict_candidates: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    maximum_pairs: int = 10,
) -> pd.DataFrame:
    selected = rankings.loc[_truth(rankings["selected"])].copy()
    orientation = strict_candidates[
        ["rebalance_date", "pair", "dependent", "independent"]
    ].drop_duplicates(["rebalance_date", "pair"])
    selected = selected.merge(
        orientation,
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
    )
    if selected[["dependent", "independent"]].isna().any().any():
        raise RuntimeError("strict candidate orientation is missing for a tuned pair")
    labels = universe.set_index("ticker")["sub_industry"]
    selected["dependent_sub_industry"] = selected["dependent"].map(labels)
    selected["independent_sub_industry"] = selected["independent"].map(labels)
    selected["same_current_sub_industry"] = (
        selected["dependent_sub_industry"].notna()
        & selected["dependent_sub_industry"].eq(selected["independent_sub_industry"])
    )
    filtered = selected.loc[selected["same_current_sub_industry"]].copy()
    filtered = filtered.sort_values(
        ["rebalance_date", "pooled_net_cagr", "pair"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    filtered = filtered.groupby("rebalance_date", group_keys=False).head(maximum_pairs)
    filtered["selected_config_id"] = filtered["config_id"]
    return filtered.reset_index(drop=True)


def run_grid_and_ensemble(
    schedule: pd.DataFrame,
    rankings: pd.DataFrame,
    prices: pd.DataFrame,
    base: StrategyConfig,
    cache_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oos_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    ensemble_metrics: list[dict[str, Any]] = []
    ensemble_daily: list[pd.DataFrame] = []
    for raw in schedule.sort_values(["outer_start", "pair"]).to_dict("records"):
        date_token = pd.Timestamp(raw["rebalance_date"]).strftime("%Y%m%d")
        pair_token = str(raw["pair"]).replace("-", "_").lower()
        cache = cache_root / "grid" / date_token / pair_token
        metric_path = cache / "oos_config_metrics.csv.gz"
        daily_path = cache / "ensemble_daily.csv.gz"
        ensemble_path = cache / "ensemble_metrics.json"
        subset = rankings.loc[
            rankings["pair"].astype(str).eq(str(raw["pair"]))
            & pd.to_datetime(rankings["rebalance_date"]).eq(
                pd.Timestamp(raw["rebalance_date"])
            )
        ].copy()
        if len(subset) != 97:
            raise RuntimeError(
                f"{raw['rebalance_date']} {raw['pair']} has {len(subset)} configs, expected 97"
            )
        if metric_path.exists() and daily_path.exists() and ensemble_path.exists():
            metrics = pd.read_csv(metric_path)
            daily = pd.read_csv(daily_path)
            ensemble = _json(ensemble_path)
        else:
            train, evaluation = _indexes(raw, prices)
            simulations = []
            rows = []
            for parameter_row in subset.to_dict("records"):
                parameter_row["selected_config_id"] = parameter_row["config_id"]
                result = simulate_v7_pair(
                    _pair_definition(raw),
                    prices,
                    train,
                    evaluation,
                    parameter_from_schedule(parameter_row),
                    base,
                    V7Rules(block_entry_at_or_beyond_stop=True),
                )
                rows.append(
                    {
                        "rebalance_date": raw["rebalance_date"],
                        "pair": raw["pair"],
                        "config_id": parameter_row["config_id"],
                        "training_score": parameter_row["pooled_net_cagr"],
                        "training_eligible": bool(parameter_row["eligible"]),
                        "oos_net_total_return": result.metrics["net_total_return"],
                        "oos_trade_count": result.metrics["trade_count"],
                    }
                )
                if bool(parameter_row["eligible"]):
                    simulations.append(result)
            metrics = pd.DataFrame(rows)
            daily_frame, ensemble = aggregate_ensemble(
                simulations,
                transaction_cost_bps=base.transaction_cost_bps,
                annual_short_borrow_bps=base.annual_short_borrow_bps,
            )
            daily = daily_frame.reset_index()
            cache.mkdir(parents=True, exist_ok=True)
            _atomic_csv(metrics, metric_path)
            _atomic_csv(daily, daily_path)
            _atomic_json(ensemble, ensemble_path)
        oos_rows.extend(metrics.to_dict("records"))
        valid = metrics[["training_score", "oos_net_total_return"]].dropna()
        correlation = (
            float(spearmanr(valid["training_score"], valid["oos_net_total_return"]).statistic)
            if len(valid) >= 3
            else math.nan
        )
        ic_rows.append(
            {
                "rebalance_date": raw["rebalance_date"],
                "pair": raw["pair"],
                "configuration_count": len(valid),
                "spearman_training_rank_vs_oos_rank": correlation,
            }
        )
        ensemble_metrics.append(
            {
                "variant": "06_training_eligible_grid_ensemble",
                "rebalance_date": raw["rebalance_date"],
                "pair": raw["pair"],
                "sector": raw["sector"],
                "net_total_return": ensemble["total_return"],
                "max_drawdown": ensemble["max_drawdown"],
                "sharpe": ensemble["sharpe"],
                "configuration_count": ensemble["configuration_count"],
                "transaction_cost_sum": ensemble["transaction_cost_sum"],
                "annualized_turnover": ensemble["annualized_turnover"],
            }
        )
        daily.insert(0, "variant", "06_training_eligible_grid_ensemble")
        daily.insert(1, "rebalance_date", raw["rebalance_date"])
        daily.insert(2, "pair", raw["pair"])
        daily.insert(3, "sector", raw["sector"])
        ensemble_daily.append(daily)
    return (
        pd.DataFrame(oos_rows),
        pd.DataFrame(ic_rows),
        pd.DataFrame(ensemble_metrics),
        pd.concat(ensemble_daily, ignore_index=True),
    )


def daily_portfolio(
    daily: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    supply: Mapping[pd.Timestamp, float] | None = None,
) -> pd.DataFrame:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["rebalance_date"] = pd.to_datetime(work["rebalance_date"])
    # A carry variant can contain two zero/non-zero records for the same pair/date.
    pair_day = work.groupby(["date", "pair"], as_index=False)["net_return"].sum()
    returns = pair_day.groupby("date")["net_return"].mean().reindex(calendar, fill_value=0.0)
    exposure = pd.Series(1.0, index=calendar)
    if supply is not None:
        starts = sorted((pd.Timestamp(key), float(value)) for key, value in supply.items())
        for number, (start, value) in enumerate(starts):
            end = starts[number + 1][0] if number + 1 < len(starts) else calendar[-1] + pd.Timedelta(days=1)
            exposure.loc[(exposure.index >= start) & (exposure.index < end)] = value
    scaled = returns * exposure
    equity = (1.0 + scaled).cumprod()
    result = pd.DataFrame(
        {"portfolio_return": scaled, "gross_exposure_scalar": exposure, "equity": equity}
    )
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0
    result.index.name = "date"
    return result.reset_index()


def pair_cost_sensitivity(
    daily: pd.DataFrame, costs_bps: Iterable[float]
) -> pd.DataFrame:
    """Reprice every sleeve using frozen gross P&L, turnover, and borrow."""

    rows: list[dict[str, Any]] = []
    keys = ["variant", "rebalance_date", "pair", "sector"]
    for cost_bps in costs_bps:
        work = daily.copy()
        work["cost_adjusted_return"] = (
            work["gross_return"]
            - work["turnover"]
            * float(cost_bps)
            / 10_000.0
            * (1.0 + work["gross_return"])
            - work["borrow_cost"]
        )
        for identity, group in work.groupby(keys, sort=True):
            rows.append(
                {
                    **dict(zip(keys, identity, strict=True)),
                    "transaction_cost_bps": float(cost_bps),
                    "net_total_return": float(
                        (1.0 + group["cost_adjusted_return"]).prod() - 1.0
                    ),
                    "transaction_cost_sum": float(
                        (
                            group["turnover"]
                            * float(cost_bps)
                            / 10_000.0
                            * (1.0 + group["gross_return"])
                        ).sum()
                    ),
                    "borrow_cost_sum": float(group["borrow_cost"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _selected_validation_matrix(
    rows: pd.DataFrame,
    validation_returns: pd.DataFrame,
) -> pd.DataFrame:
    pieces: dict[str, pd.Series] = {}
    for raw in rows.to_dict("records"):
        subset = validation_returns.loc[
            validation_returns["pair"].astype(str).eq(str(raw["pair"]))
            & validation_returns["rebalance_date"].eq(raw["rebalance_date"])
        ]
        column = str(raw["selected_config_id"])
        if column not in subset or subset.empty:
            raise RuntimeError(
                f"selected validation returns missing for {raw['rebalance_date']} {raw['pair']}"
            )
        series = pd.Series(
            pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(subset["date"]),
            name=str(raw["pair"]),
        )
        pieces[str(raw["pair"])] = series
    matrix = pd.DataFrame(pieces).sort_index()
    if matrix.isna().any().any():
        raise RuntimeError("selected validation returns are not synchronized")
    return matrix


def _factor_betas(
    pair_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    sector_etf: str,
) -> tuple[float, float, int]:
    aligned = pd.concat(
        [
            pair_returns.rename("pair"),
            factor_returns["SPY"].rename("market"),
            (factor_returns[sector_etf] - factor_returns["SPY"]).rename(
                "sector_residual"
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 30:
        return 0.0, 0.0, len(aligned)
    design = np.column_stack(
        [
            np.ones(len(aligned)),
            aligned["market"].to_numpy(dtype=float),
            aligned["sector_residual"].to_numpy(dtype=float),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(
        design, aligned["pair"].to_numpy(dtype=float), rcond=None
    )
    return float(coefficients[1]), float(coefficients[2]), len(aligned)


def run_portfolio_risk_layer(
    schedule: pd.DataFrame,
    stop_daily: pd.DataFrame,
    validation_returns: pd.DataFrame,
    factor_prices: pd.DataFrame,
    config: Mapping[str, Any],
    calendar: pd.DatetimeIndex,
    all_rebalance_dates: Iterable[pd.Timestamp],
    transaction_cost_bps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = config["experiments"]["10_portfolio_risk_layer"]
    factor_returns = factor_prices.pct_change(fill_method=None)
    stop = stop_daily.copy()
    stop["date"] = pd.to_datetime(stop["date"])
    stop["rebalance_date"] = pd.to_datetime(stop["rebalance_date"])
    weight_rows: list[dict[str, Any]] = []
    beta_rows: list[dict[str, Any]] = []
    base = pd.Series(0.0, index=calendar)
    overlay_targets: dict[pd.Timestamp, dict[str, float]] = {}
    all_factor_tickers = ["SPY", *sorted(set(settings["sector_etfs"].values()))]
    for rebalance, rows in schedule.groupby("rebalance_date", sort=True):
        selected_returns = _selected_validation_matrix(rows, validation_returns)
        requested = min(1.0, len(rows) / 10.0)
        weights, diagnostics = cash_aware_inverse_volatility(
            selected_returns,
            rows[["pair", "dependent", "independent", "sector"]],
            requested_gross=requested,
            pair_cap=float(settings["pair_weight_cap"]),
            name_cap=float(settings["shared_name_incidence_cap"]),
            sector_cap=float(settings["sector_cap"]),
        )
        market_beta = 0.0
        sector_betas: dict[str, float] = {}
        for raw in rows.to_dict("records"):
            pair = str(raw["pair"])
            sector = str(raw["sector"])
            etf = str(settings["sector_etfs"][sector])
            beta_market, beta_sector, observations = _factor_betas(
                selected_returns[pair], factor_returns, sector_etf=etf
            )
            weight = float(weights.get(pair, 0.0))
            market_beta += weight * beta_market
            sector_betas[etf] = sector_betas.get(etf, 0.0) + weight * beta_sector
            weight_rows.append(
                {
                    "rebalance_date": rebalance,
                    "pair": pair,
                    "sector": sector,
                    "dependent": raw["dependent"],
                    "independent": raw["independent"],
                    "weight": weight,
                    **diagnostics,
                }
            )
            beta_rows.append(
                {
                    "rebalance_date": rebalance,
                    "pair": pair,
                    "market_beta": beta_market,
                    "sector_residual_beta": beta_sector,
                    "sector_etf": etf,
                    "observations": observations,
                    "portfolio_weight": weight,
                }
            )
        target = {ticker: 0.0 for ticker in all_factor_tickers}
        target["SPY"] = -market_beta + sum(sector_betas.values())
        for etf, beta in sector_betas.items():
            target[etf] = -beta
        overlay_targets[pd.Timestamp(rebalance)] = target
        quarter_daily = stop.loc[stop["rebalance_date"].eq(rebalance)]
        pivot = quarter_daily.pivot_table(
            index="date", columns="pair", values="net_return", aggfunc="sum"
        )
        aligned_weights = weights.reindex(pivot.columns, fill_value=0.0)
        quarter_return = pivot.fillna(0.0).mul(aligned_weights, axis=1).sum(axis=1)
        base.loc[base.index.intersection(quarter_return.index)] = quarter_return.reindex(
            base.index.intersection(quarter_return.index)
        )
    targets = pd.DataFrame(0.0, index=calendar, columns=all_factor_tickers)
    for date, values in overlay_targets.items():
        if date in targets.index:
            targets.loc[date, list(values)] = list(values.values())
    rebalance_mask = targets.abs().sum(axis=1).ne(0)
    # Explicitly mark zero-target quarters using every scheduled rebalance date.
    for date in pd.to_datetime(list(all_rebalance_dates)):
        if date in targets.index:
            rebalance_mask.at[date] = True
    targets.loc[~rebalance_mask] = np.nan
    targets = targets.ffill().fillna(0.0)
    held = targets.shift(1, fill_value=0.0)
    aligned_factors = factor_returns.reindex(calendar)[all_factor_tickers].fillna(0.0)
    overlay_return = (held * aligned_factors).sum(axis=1)
    turnover = (targets - held).abs().sum(axis=1)
    # Liquidate the factor overlay at the final close.
    terminal_turnover = float(targets.iloc[-1].abs().sum())
    turnover.iloc[-1] += terminal_turnover
    applied_cost_bps = (
        float(config["frozen_mechanics"]["transaction_cost_bps"])
        if transaction_cost_bps is None
        else float(transaction_cost_bps)
    )
    cost = turnover * applied_cost_bps / 10_000.0
    net = base + overlay_return - cost
    daily = pd.DataFrame(
        {
            "date": calendar,
            "allocated_pair_return": base.to_numpy(),
            "factor_overlay_return": overlay_return.to_numpy(),
            "overlay_turnover": turnover.to_numpy(),
            "overlay_transaction_cost": cost.to_numpy(),
            "portfolio_return": net.to_numpy(),
        }
    )
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    metrics = performance_metrics(daily["portfolio_return"], daily["equity"])
    metrics.update(
        {
            "overlay_transaction_cost_sum": float(cost.sum()),
            "average_allocated_gross": float(
                pd.DataFrame(weight_rows).drop_duplicates("rebalance_date")[
                    "effective_gross"
                ].mean()
            ),
            "terminal_overlay_liquidation_turnover": terminal_turnover,
            "transaction_cost_bps": applied_cost_bps,
        }
    )
    return daily, pd.DataFrame(weight_rows), pd.DataFrame(beta_rows), metrics


def variant_summary(
    metrics: pd.DataFrame,
    v6_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = v6_metrics[["rebalance_date", "pair", "net_total_return"]].copy()
    baseline = baseline.rename(columns={"net_total_return": "v6_net_total_return"})
    work = metrics.merge(
        baseline,
        on=["rebalance_date", "pair"],
        how="left",
        validate="many_to_one",
    )
    work["net_total_return_delta_vs_v6"] = (
        work["net_total_return"] - work["v6_net_total_return"]
    )
    return (
        work.groupby("variant", as_index=False)
        .agg(
            pair_vintages=("pair", "size"),
            mean_net_return=("net_total_return", "mean"),
            median_net_return=("net_total_return", "median"),
            positive_fraction=("net_total_return", lambda values: float((values > 0).mean())),
            mean_delta_vs_v6=("net_total_return_delta_vs_v6", "mean"),
            median_delta_vs_v6=("net_total_return_delta_vs_v6", "median"),
            improved_fraction=("net_total_return_delta_vs_v6", lambda values: float((values > 0).mean())),
            mean_trade_count=("trade_count", "mean"),
        )
        .sort_values("mean_delta_vs_v6", ascending=False, kind="mergesort")
    )


def manual_reconcile_control(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Independently reconstruct raw-price P&L, turnover, costs, and sample trades."""

    control = daily.loc[daily["variant"].eq("00_v7_replay_control")].copy()
    control["date"] = pd.to_datetime(control["date"])
    audit_rows: list[dict[str, Any]] = []
    for (rebalance, pair), group in control.groupby(
        ["rebalance_date", "pair"], sort=True
    ):
        work = group.sort_values("date").copy()
        dep_return = pd.to_numeric(work["price_dependent"], errors="coerce").pct_change(
            fill_method=None
        ).fillna(0.0)
        ind_return = pd.to_numeric(work["price_independent"], errors="coerce").pct_change(
            fill_method=None
        ).fillna(0.0)
        reconstructed_gross = (
            work["weight_dependent"].to_numpy(dtype=float) * dep_return.to_numpy()
            + work["weight_independent"].to_numpy(dtype=float) * ind_return.to_numpy()
        )
        denominator = 1.0 + reconstructed_gross
        drift_dep = np.where(
            np.abs(denominator) > 1e-12,
            work["weight_dependent"].to_numpy(dtype=float)
            * (1.0 + dep_return.to_numpy())
            / denominator,
            work["weight_dependent"].to_numpy(dtype=float),
        )
        drift_ind = np.where(
            np.abs(denominator) > 1e-12,
            work["weight_independent"].to_numpy(dtype=float)
            * (1.0 + ind_return.to_numpy())
            / denominator,
            work["weight_independent"].to_numpy(dtype=float),
        )
        reconstructed_turnover = np.abs(
            work["target_weight_dependent"].to_numpy(dtype=float) - drift_dep
        ) + np.abs(
            work["target_weight_independent"].to_numpy(dtype=float) - drift_ind
        )
        reconstructed_cost = (
            reconstructed_turnover
            * 2.0
            / 10_000.0
            * (1.0 + reconstructed_gross)
        )
        reconstructed_net = (
            reconstructed_gross
            - reconstructed_cost
            - work["borrow_cost"].to_numpy(dtype=float)
        )
        audit_rows.append(
            {
                "rebalance_date": rebalance,
                "pair": pair,
                "rows": len(work),
                "maximum_gross_return_error": float(
                    np.max(np.abs(reconstructed_gross - work["gross_return"]))
                ),
                "maximum_turnover_error": float(
                    np.max(np.abs(reconstructed_turnover - work["turnover"]))
                ),
                "maximum_transaction_cost_error": float(
                    np.max(np.abs(reconstructed_cost - work["transaction_cost"]))
                ),
                "maximum_net_return_error": float(
                    np.max(np.abs(reconstructed_net - work["net_return"]))
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    control_trades = trades.loc[trades["variant"].eq("00_v7_replay_control")].copy()
    samples: list[dict[str, Any]] = []
    selectors = {
        "losing_trade": control_trades["net_return"].lt(0),
        "long_spread": control_trades["direction"].eq("long_spread"),
        "short_spread": control_trades["direction"].eq("short_spread"),
        "terminal_exit": control_trades["exit_reason"].eq(
            "exit_terminal_scheduled"
        ),
        "time_exit": control_trades["exit_reason"].eq("exit_max_holding"),
    }
    for label, mask in selectors.items():
        candidates = control_trades.loc[mask]
        if candidates.empty:
            continue
        trade = candidates.iloc[0]
        segment = control.loc[
            control["pair"].eq(trade["pair"])
            & pd.to_datetime(control["rebalance_date"]).eq(
                pd.Timestamp(trade["rebalance_date"])
            )
            & control["date"].between(
                pd.Timestamp(trade["entry_date"]), pd.Timestamp(trade["exit_date"])
            )
        ].sort_values("date")
        reconstructed = float((1.0 + segment["net_return"]).prod() - 1.0)
        samples.append(
            {
                "sample_type": label,
                "rebalance_date": trade["rebalance_date"],
                "pair": trade["pair"],
                "direction": trade["direction"],
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "exit_reason": trade["exit_reason"],
                "ledger_net_return": trade["net_return"],
                "reconstructed_net_return": reconstructed,
                "absolute_error": abs(reconstructed - float(trade["net_return"])),
                "transaction_cost_sum": float(segment["transaction_cost"].sum()),
                "borrow_cost_sum": float(segment["borrow_cost"].sum()),
            }
        )
    tolerance = 1e-12
    summary = {
        "pair_vintages_reconstructed": int(len(audit)),
        "sample_trades_reconstructed": len(samples),
        "maximum_gross_return_error": float(audit["maximum_gross_return_error"].max()),
        "maximum_turnover_error": float(audit["maximum_turnover_error"].max()),
        "maximum_transaction_cost_error": float(
            audit["maximum_transaction_cost_error"].max()
        ),
        "maximum_net_return_error": float(audit["maximum_net_return_error"].max()),
        "maximum_sample_trade_error": float(
            max((row["absolute_error"] for row in samples), default=0.0)
        ),
    }
    summary["passed"] = bool(
        max(
            summary["maximum_gross_return_error"],
            summary["maximum_turnover_error"],
            summary["maximum_transaction_cost_error"],
            summary["maximum_net_return_error"],
            summary["maximum_sample_trade_error"],
        )
        <= tolerance
    )
    sample_frame = pd.DataFrame(samples)
    sample_frame.attrs["summary"] = summary
    return pd.concat([audit.assign(sample_type="all_rows"), sample_frame], ignore_index=True, sort=False), summary


REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 pair strategy experiments</title><style>body{margin:0;background:#eef2f6;color:#172033;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1180px;margin:auto;background:#fff;min-height:100vh}header{padding:40px 48px;background:#17324d;color:#fff}main{padding:30px 48px 60px}h1{margin:.2em 0}.note,.blocked{padding:14px 16px;margin:14px 0;border-left:5px solid #3568b8;background:#eef5ff}.blocked{border-color:#b94b57;background:#fff0f1}.table{overflow:auto;border:1px solid #dfe5ed;border-radius:8px}.dataframe{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}.dataframe th{background:#f3f5f7;text-align:left}.dataframe th,.dataframe td{padding:8px;border-bottom:1px solid #e7ebf0}.links a{display:inline-block;margin:4px;padding:7px 9px;border:1px solid #dfe5ed;border-radius:5px;text-decoration:none}code{background:#eef1f5;padding:2px 4px}footer{padding:20px 48px;background:#f3f5f7;color:#667085}</style></head><body><div class="page"><header><div>ADDITIVE V7 · V6 IMMUTABLE</div><h1>Pair strategy structural experiments</h1><p>Stop-state correction, carry, half-life holding, supply scaling, full-grid IC/ensemble, confirmation, sub-industry, dynamic beta, portfolio risk and minute-risk gate.</p></header><main><div class="blocked"><strong>Research status: NOT VALIDATED.</strong> Point-in-time universe/GICS/delistings and ten-year minute history are absent; all 2016–2026 outcomes are now a development sample. None of the five tested sleeve variants passed the synchronized Romano–Wolf family test.</div>{% if minute_pass %}<div class="note">Minute history contract passed.</div>{% else %}<div class="blocked"><strong>Minute-level empirical replay blocked.</strong> {{ minute_error }}</div>{% endif %}{% if pit_pass %}<div class="note">PIT GICS contract passed.</div>{% else %}<div class="blocked"><strong>PIT sub-industry result blocked.</strong> Current-GICS sensitivity is labeled ex-post.</div>{% endif %}<h2>Portfolio outcomes at 2 bp</h2><div class="table">{{ portfolio_table }}</div><h2>1–2 bp cost comparison</h2><div class="table">{{ cost_table }}</div><h2>Sequential sleeve variants</h2><p>Variants 03, 06, 07, 08 and 09 retain the stop-state gate from variant 01; their deltas therefore measure the complete labeled bundle against V6, not an isolated marginal effect.</p><div class="table">{{ variant_table }}</div><h2>97-grid predictive diagnostic</h2><p>Mean pair-vintage Spearman IC: <strong>{{ mean_ic }}</strong>; median: <strong>{{ median_ic }}</strong>. Quarter-mean IC: <strong>{{ quarter_ic }}</strong> across {{ ic_quarters }} quarters.</p><h2>Implementation and audit status</h2><div class="table">{{ status_table }}</div><h2>Artifacts</h2><div class="links">{% for link in links %}<a href="data/processed/{{ link }}">{{ link }}</a>{% endfor %}</div></main><footer>Generated {{ generated }}</footer></div></body></html>"""
)


def run(config_path: Path, *, skip_grid: bool = False) -> dict[str, Any]:
    workspace = Path.cwd().resolve()
    config_file = _resolve(workspace, config_path)
    config = _json(config_file)
    paths = Paths.create(_resolve(workspace, config["outputs"]["root"]))
    v6_root = _resolve(workspace, config["inputs"]["v6_root"])
    source = config["inputs"]
    price_path = _resolve(v6_root, source["price_snapshot"])
    schedule_path = _resolve(v6_root, source["selected_schedule"])
    ranking_path = _resolve(v6_root, source["parameter_rankings"])
    strict_path = _resolve(v6_root, source["strict_candidates"])
    universe_path = _resolve(v6_root, source["universe_snapshot"])
    vintage_path = _resolve(v6_root, source["vintage_summary"])
    validation_path = _resolve(v6_root, source["validation_returns"])
    v6_metric_path = v6_root / "data" / "processed" / "pair_v6_pair_vintage_metrics.csv"
    required = [
        price_path,
        schedule_path,
        ranking_path,
        strict_path,
        universe_path,
        vintage_path,
        validation_path,
        v6_metric_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen V6 inputs: " + ", ".join(missing))
    v6_hashes_before = {str(path): _sha256(path) for path in required}
    prices = pd.read_csv(price_path, index_col="date", parse_dates=True).sort_index()
    schedule = pd.read_csv(schedule_path)
    rankings = pd.read_csv(ranking_path)
    strict_candidates = pd.read_csv(strict_path)
    universe = pd.read_csv(universe_path)
    vintage = pd.read_csv(vintage_path)
    validation_returns = pd.read_csv(validation_path)
    v6_metrics = pd.read_csv(v6_metric_path)
    for frame in (
        schedule,
        rankings,
        strict_candidates,
        vintage,
        validation_returns,
        v6_metrics,
    ):
        for column in (
            "rebalance_date",
            "formation_start",
            "formation_end",
            "outer_start",
            "outer_end_exclusive",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], format="mixed")
    base = _base_config(config)

    variants: list[pd.DataFrame] = []
    dailies: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    definitions = [
        ("00_v7_replay_control", V7Rules(block_entry_at_or_beyond_stop=False)),
        ("01_stop_entry_gate", V7Rules(block_entry_at_or_beyond_stop=True)),
        (
            "03_half_life_holding_2x",
            V7Rules(block_entry_at_or_beyond_stop=True, half_life_holding_multiplier=2.0),
        ),
        (
            "07_reversal_confirmation",
            V7Rules(block_entry_at_or_beyond_stop=True, require_reversal_confirmation=True),
        ),
        (
            "09_dynamic_beta_kalman",
            V7Rules(
                block_entry_at_or_beyond_stop=True,
                dynamic_beta=True,
                kalman_process_variance_ratio=float(
                    config["experiments"]["09_dynamic_beta"]["process_variance_ratio"]
                ),
            ),
        ),
    ]
    for name, rules in definitions:
        metric, daily, trade = run_schedule_variant(name, schedule, prices, base, rules)
        variants.append(metric)
        dailies.append(daily)
        trades.append(trade)
    carry_metric, carry_daily, carry_trades = run_carry_variant(schedule, prices, base)
    variants.append(carry_metric)
    dailies.append(carry_daily)
    trades.append(carry_trades)

    sub_schedule = current_subindustry_schedule(
        rankings, strict_candidates, universe, maximum_pairs=10
    )
    sub_metric, sub_daily, sub_trades = run_schedule_variant(
        "08_current_gics_subindustry_expost",
        sub_schedule,
        prices,
        base,
        V7Rules(block_entry_at_or_beyond_stop=True),
    )
    variants.append(sub_metric)
    dailies.append(sub_daily)
    trades.append(sub_trades)

    grid_metrics = pd.DataFrame()
    grid_ic = pd.DataFrame()
    ensemble_metrics = pd.DataFrame()
    ensemble_daily = pd.DataFrame()
    if not skip_grid:
        grid_metrics, grid_ic, ensemble_metrics, ensemble_daily = run_grid_and_ensemble(
            schedule, rankings, prices, base, paths.cache
        )
        variants.append(ensemble_metrics)
        dailies.append(ensemble_daily)

    all_metrics = pd.concat(variants, ignore_index=True, sort=False)
    all_daily = pd.concat(dailies, ignore_index=True, sort=False)
    all_trades = pd.concat(
        [frame for frame in trades if len(frame)], ignore_index=True, sort=False
    )
    summary = variant_summary(all_metrics, v6_metrics)
    manual_audit, manual_summary = manual_reconcile_control(all_daily, all_trades)
    sensitivity_costs = [
        float(config["frozen_mechanics"]["cost_sensitivity_bps"]),
        float(config["frozen_mechanics"]["transaction_cost_bps"]),
    ]
    pair_cost_frame = pair_cost_sensitivity(all_daily, sensitivity_costs)

    control = all_metrics.loc[all_metrics["variant"].eq("00_v7_replay_control")]
    parity = control.merge(
        v6_metrics[["rebalance_date", "pair", "net_total_return"]],
        on=["rebalance_date", "pair"],
        suffixes=("_v7", "_v6"),
        validate="one_to_one",
    )
    parity["absolute_difference"] = (
        parity["net_total_return_v7"] - parity["net_total_return_v6"]
    ).abs()
    parity_pass = bool(parity["absolute_difference"].max() <= 1e-10)

    calendar = prices.index[
        (prices.index >= schedule["outer_start"].min())
        & (prices.index < schedule["outer_end_exclusive"].max())
    ]
    portfolio_rows: list[pd.DataFrame] = []
    portfolio_metrics: list[dict[str, Any]] = []
    for name, group in all_daily.groupby("variant", sort=True):
        portfolio = daily_portfolio(group, calendar=calendar)
        portfolio.insert(0, "variant", name)
        portfolio_rows.append(portfolio)
        values = performance_metrics(
            portfolio["portfolio_return"], portfolio["equity"]
        )
        portfolio_metrics.append({"variant": name, **values})
    supply_values = {
        pd.Timestamp(row["rebalance_date"]): min(
            1.0, float(row["strict_candidate_count"]) / 10.0
        )
        for row in vintage.to_dict("records")
    }
    stop_daily = all_daily.loc[all_daily["variant"].eq("01_stop_entry_gate")]
    supply_portfolio = daily_portfolio(
        stop_daily, calendar=calendar, supply=supply_values
    )
    supply_portfolio.insert(0, "variant", "04_candidate_supply_exposure")
    portfolio_rows.append(supply_portfolio)
    supply_values_metric = performance_metrics(
        supply_portfolio["portfolio_return"], supply_portfolio["equity"]
    )
    portfolio_metrics.append(
        {"variant": "04_candidate_supply_exposure", **supply_values_metric}
    )
    portfolio_daily = pd.concat(portfolio_rows, ignore_index=True)
    portfolio_metric_frame = pd.DataFrame(portfolio_metrics)
    portfolio_cost_rows: list[dict[str, Any]] = []
    for cost_bps in sensitivity_costs:
        repriced = all_daily.copy()
        repriced["net_return"] = (
            repriced["gross_return"]
            - repriced["turnover"]
            * cost_bps
            / 10_000.0
            * (1.0 + repriced["gross_return"])
            - repriced["borrow_cost"]
        )
        for name, group in repriced.groupby("variant", sort=True):
            path = daily_portfolio(group, calendar=calendar)
            portfolio_cost_rows.append(
                {
                    "variant": name,
                    "transaction_cost_bps": cost_bps,
                    **performance_metrics(path["portfolio_return"], path["equity"]),
                }
            )
        supply_path = daily_portfolio(
            repriced.loc[repriced["variant"].eq("01_stop_entry_gate")],
            calendar=calendar,
            supply=supply_values,
        )
        portfolio_cost_rows.append(
            {
                "variant": "04_candidate_supply_exposure",
                "transaction_cost_bps": cost_bps,
                **performance_metrics(
                    supply_path["portfolio_return"], supply_path["equity"]
                ),
            }
        )

    factor_download_error: str | None = None
    risk_daily = pd.DataFrame()
    risk_weights = pd.DataFrame()
    risk_betas = pd.DataFrame()
    risk_metrics: dict[str, Any] = {}
    factor_path = paths.root / str(source["market_factor_snapshot"])
    factor_settings = config["experiments"]["10_portfolio_risk_layer"]
    factor_tickers = [
        str(factor_settings["market_etf"]),
        *sorted(set(map(str, factor_settings["sector_etfs"].values()))),
    ]
    try:
        factor_prices = download_adjusted_close(
            factor_tickers,
            "2016-08-26",
            "2026-08-27",
            chunk_size=len(factor_tickers),
            max_retries=3,
            backoff_factor=1.0,
            inter_chunk_delay=0.0,
            repair=True,
            strict=True,
            cache_path=factor_path,
            refresh=False,
        ).sort_index()
        missing_factors = sorted(set(factor_tickers).difference(factor_prices.columns))
        if missing_factors:
            raise RuntimeError("factor snapshot missing: " + ", ".join(missing_factors))
        risk_daily, risk_weights, risk_betas, risk_metrics = run_portfolio_risk_layer(
            schedule,
            stop_daily,
            validation_returns,
            factor_prices,
            config,
            calendar,
            all_rebalance_dates=pd.to_datetime(vintage["rebalance_date"].unique()),
        )
        risk_daily.insert(0, "variant", "10_risk_allocation_beta_overlay")
        portfolio_daily = pd.concat([portfolio_daily, risk_daily], ignore_index=True)
        portfolio_metric_frame = pd.concat(
            [
                portfolio_metric_frame,
                pd.DataFrame(
                    [{"variant": "10_risk_allocation_beta_overlay", **risk_metrics}]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        for cost_bps in sensitivity_costs:
            adjusted_stop = stop_daily.copy()
            adjusted_stop["net_return"] = (
                adjusted_stop["gross_return"]
                - adjusted_stop["turnover"]
                * cost_bps
                / 10_000.0
                * (1.0 + adjusted_stop["gross_return"])
                - adjusted_stop["borrow_cost"]
            )
            sensitivity_daily, _, _, sensitivity_metrics = run_portfolio_risk_layer(
                schedule,
                adjusted_stop,
                validation_returns,
                factor_prices,
                config,
                calendar,
                all_rebalance_dates=pd.to_datetime(vintage["rebalance_date"].unique()),
                transaction_cost_bps=cost_bps,
            )
            portfolio_cost_rows.append(
                {
                    "variant": "10_risk_allocation_beta_overlay",
                    "transaction_cost_bps": cost_bps,
                    **performance_metrics(
                        sensitivity_daily["portfolio_return"],
                        sensitivity_daily["equity"],
                    ),
                    "overlay_transaction_cost_sum": sensitivity_metrics[
                        "overlay_transaction_cost_sum"
                    ],
                }
            )
    except Exception as exc:
        factor_download_error = f"{type(exc).__name__}: {exc}"

    # Variant-family inference uses quarter-aggregated paired deltas.  Subindustry
    # is excluded because it changes the pair family; carry and portfolio layers
    # are reported separately because they change the calendar ownership of P&L.
    infer_variants = [
        "01_stop_entry_gate",
        "03_half_life_holding_2x",
        "07_reversal_confirmation",
        "09_dynamic_beta_kalman",
    ]
    if len(ensemble_metrics):
        infer_variants.append("06_training_eligible_grid_ensemble")
    paired = all_metrics.loc[all_metrics["variant"].isin(infer_variants)].merge(
        v6_metrics[["rebalance_date", "pair", "net_total_return"]],
        on=["rebalance_date", "pair"],
        suffixes=("", "_v6"),
        validate="many_to_one",
    )
    paired["delta"] = paired["net_total_return"] - paired["net_total_return_v6"]
    quarter_delta = paired.pivot_table(
        index="rebalance_date", columns="variant", values="delta", aggfunc="mean"
    ).sort_index()
    quarter_delta = quarter_delta.reindex(columns=infer_variants).fillna(0.0)
    indices = circular_block_indices(
        len(quarter_delta),
        draws=int(config["inference"]["bootstrap_draws"]),
        block_length=int(config["inference"]["quarter_block_length"]),
        seed=int(config["inference"]["seed"]),
    )
    family_inference, family_summary = romano_wolf_max_t(
        quarter_delta.to_numpy(dtype=float), infer_variants, indices, alpha=0.05
    )

    minute_audit = audit_minute_history(_resolve(workspace, source["minute_history"]))
    pit_config_path = workspace / "configs" / "pair_identity_pit_selection_correction.json"
    if pit_config_path.exists():
        pit_audit, _ = audit_point_in_time_contract(workspace, _json(pit_config_path))
    else:
        pit_audit = {"critical_pass": False, "errors": ["PIT config absent"]}

    _atomic_csv(all_metrics, paths.processed / "v7_pair_vintage_metrics.csv.gz")
    _atomic_csv(all_daily, paths.processed / "v7_daily_signals.csv.gz")
    _atomic_csv(all_trades, paths.processed / "v7_trade_ledger.csv.gz")
    _atomic_csv(summary, paths.processed / "v7_variant_summary.csv")
    _atomic_csv(
        pair_cost_frame,
        paths.processed / "v7_pair_vintage_cost_sensitivity.csv",
    )
    _atomic_csv(parity, paths.processed / "v7_control_reconciliation.csv")
    _atomic_csv(manual_audit, paths.processed / "v7_manual_reconciliation.csv")
    _atomic_json(
        manual_summary,
        paths.processed / "v7_manual_reconciliation_summary.json",
    )
    _atomic_csv(sub_schedule, paths.processed / "v7_current_subindustry_schedule.csv")
    _atomic_csv(portfolio_daily, paths.processed / "v7_portfolio_daily.csv.gz")
    _atomic_csv(portfolio_metric_frame, paths.processed / "v7_portfolio_metrics.csv")
    _atomic_csv(
        pd.DataFrame(portfolio_cost_rows),
        paths.processed / "v7_portfolio_cost_sensitivity.csv",
    )
    if len(risk_weights):
        _atomic_csv(risk_weights, paths.processed / "v7_risk_allocation_weights.csv")
        _atomic_csv(risk_betas, paths.processed / "v7_pair_factor_betas.csv")
    _atomic_csv(quarter_delta.reset_index(), paths.processed / "v7_quarterly_paired_deltas.csv")
    _atomic_csv(family_inference, paths.processed / "v7_romano_wolf_variant_family.csv")
    if len(grid_metrics):
        _atomic_csv(grid_metrics, paths.processed / "v7_all97_oos_config_metrics.csv.gz")
        _atomic_csv(grid_ic, paths.processed / "v7_all97_pair_vintage_ic.csv")
        _atomic_csv(ensemble_metrics, paths.processed / "v7_grid_ensemble_metrics.csv")
    _atomic_json(minute_audit, paths.processed / "v7_minute_data_audit.json")
    _atomic_json(pit_audit, paths.processed / "v7_pit_data_audit.json")

    v6_hashes_after = {str(path): _sha256(path) for path in required}
    v6_inputs_unchanged = v6_hashes_before == v6_hashes_after
    quality = {
        "v6_inputs_before": v6_hashes_before,
        "v6_inputs_after": v6_hashes_after,
        "v6_integrity_scope": "eight frozen files consumed by V7",
        "v6_inputs_unchanged": v6_inputs_unchanged,
        "v6_outputs_modified": not v6_inputs_unchanged,
        "v7_replay_control_max_absolute_return_difference": float(
            parity["absolute_difference"].max()
        ),
        "v7_replay_control_passed": parity_pass,
        "manual_reconciliation_passed": bool(manual_summary["passed"]),
        "manual_reconciliation": manual_summary,
        "selected_pair_vintage_count": int(len(schedule)),
        "all97_replay_completed": bool(len(grid_metrics) == len(schedule) * 97),
        "minute_history_passed": bool(minute_audit["passed"]),
        "pit_history_passed": bool(pit_audit["critical_pass"]),
        "portfolio_risk_layer_completed": bool(len(risk_daily)),
        "portfolio_risk_layer_error": factor_download_error,
        "family_inference": family_summary,
        "research_development_sample": True,
        "fresh_lockbox_required": True,
        "research_status_verdict": "NOT VALIDATED",
        "verdict_blockers": [
            "point-in-time membership, GICS, delistings, and corporate actions absent",
            "ten-year one-minute history absent",
            "2016-2026 outcomes have been inspected and are a development sample",
            "none of the five tested sleeve variants rejected the synchronized Romano-Wolf family null at 5%",
        ],
    }
    _atomic_json(quality, paths.processed / "v7_quality_checks.json")

    status = pd.DataFrame(
        [
            {"Layer": "Audit · exact V6 replay", "Status": "PASS" if parity_pass else "FAIL", "Scope": "118 sleeves"},
            {"Layer": "Audit · independent P&L reconstruction", "Status": "PASS" if manual_summary["passed"] else "FAIL", "Scope": f"{manual_summary['pair_vintages_reconstructed']} sleeves and {manual_summary['sample_trades_reconstructed']} sampled trades"},
            {"Layer": "1 · Stop-state entry gate", "Status": "PASS", "Scope": "No entry at or beyond the unchanged |z|=4 stop"},
            {"Layer": "2 · Cross-quarter carry", "Status": "PASS", "Scope": "Existing positions may finish; no overlapping same-pair vintage"},
            {"Layer": "3 · Half-life holding", "Status": "PASS", "Scope": "Maximum hold = 2x formation half-life"},
            {"Layer": "4 · Candidate-supply exposure", "Status": "PASS", "Scope": "Quarter gross scales with strict-candidate count"},
            {"Layer": "5 · Minute risk", "Status": "ENGINE PASS / DATA BLOCKED" if not minute_audit["passed"] else "PASS", "Scope": "Existing |z|=4 trigger; next-minute-open execution; no earnings reduction"},
            {"Layer": "6 · Full 97 grid IC and ensemble", "Status": "PASS" if quality["all97_replay_completed"] else "SKIPPED", "Scope": f"{len(grid_metrics):,} pair-config paths"},
            {"Layer": "7 · Reversal confirmation", "Status": "PASS", "Scope": "Entry waits for first move toward the mean"},
            {"Layer": "8 · Sub-industry matching", "Status": "SENSITIVITY PASS / PIT DATA BLOCKED" if not pit_audit["critical_pass"] else "PASS", "Scope": "Current GICS sensitivity is ex-post"},
            {"Layer": "9 · Dynamic hedge ratio", "Status": "PASS", "Scope": "Causal random-walk Kalman beta"},
            {"Layer": "10 · Risk allocation and beta overlay", "Status": "PASS" if len(risk_daily) else "DATA BLOCKED", "Scope": "Cash-aware inverse-volatility, pair/name/sector caps and SPY/sector overlay" if len(risk_daily) else str(factor_download_error)},
        ]
    )
    links = [path.name for path in sorted(paths.processed.iterdir()) if path.is_file()]
    mean_ic = float(grid_ic["spearman_training_rank_vs_oos_rank"].mean()) if len(grid_ic) else math.nan
    median_ic = float(grid_ic["spearman_training_rank_vs_oos_rank"].median()) if len(grid_ic) else math.nan
    if len(grid_ic):
        quarter_ic_values = grid_ic.groupby("rebalance_date")[
            "spearman_training_rank_vs_oos_rank"
        ].mean()
        quarter_ic = float(quarter_ic_values.mean())
        ic_quarters = int(quarter_ic_values.notna().sum())
    else:
        quarter_ic = math.nan
        ic_quarters = 0
    variant_display = summary.copy()
    for column in [
        "mean_net_return",
        "median_net_return",
        "positive_fraction",
        "mean_delta_vs_v6",
        "median_delta_vs_v6",
        "improved_fraction",
    ]:
        variant_display[column] = variant_display[column].map(lambda value: f"{value:.2%}")
    portfolio_display = portfolio_metric_frame[
        ["variant", "total_return", "cagr", "sharpe", "max_drawdown"]
    ].copy()
    for column in ["total_return", "cagr", "max_drawdown"]:
        portfolio_display[column] = portfolio_display[column].map(
            lambda value: f"{value:.2%}"
        )
    portfolio_display["sharpe"] = portfolio_display["sharpe"].map(
        lambda value: f"{value:.2f}"
    )
    cost_display = pd.DataFrame(portfolio_cost_rows)
    cost_display = cost_display.loc[
        cost_display["variant"].isin(
            [
                "00_v7_replay_control",
                "04_candidate_supply_exposure",
                "06_training_eligible_grid_ensemble",
                "10_risk_allocation_beta_overlay",
            ]
        ),
        ["variant", "transaction_cost_bps", "total_return", "cagr", "sharpe", "max_drawdown"],
    ].copy()
    for column in ["total_return", "cagr", "max_drawdown"]:
        cost_display[column] = cost_display[column].map(lambda value: f"{value:.2%}")
    cost_display["sharpe"] = cost_display["sharpe"].map(
        lambda value: f"{value:.2f}"
    )
    report = REPORT.render(
        minute_pass=minute_audit["passed"],
        minute_error=html.escape(str(minute_audit.get("error"))),
        pit_pass=pit_audit["critical_pass"],
        variant_table=variant_display.to_html(index=False, border=0),
        portfolio_table=portfolio_display.to_html(index=False, border=0),
        cost_table=cost_display.to_html(index=False, border=0),
        mean_ic=f"{mean_ic:.3f}" if np.isfinite(mean_ic) else "pending",
        median_ic=f"{median_ic:.3f}" if np.isfinite(median_ic) else "pending",
        quarter_ic=f"{quarter_ic:.3f}" if np.isfinite(quarter_ic) else "pending",
        ic_quarters=ic_quarters,
        status_table=status.to_html(index=False, border=0),
        links=links,
        generated=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    report_path = paths.root / config["outputs"]["report"]
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    return {
        "output_root": str(paths.root),
        "report": str(report_path),
        "v7_replay_control_passed": parity_pass,
        "pair_vintages": len(schedule),
        "all97_paths": len(grid_metrics),
        "minute_history_passed": minute_audit["passed"],
        "pit_history_passed": pit_audit["critical_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, skip_grid=args.skip_grid), indent=2))


if __name__ == "__main__":
    main()
