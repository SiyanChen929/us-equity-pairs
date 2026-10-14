#!/usr/bin/env python3
"""Run the four production gates on the frozen equity-pairs research artifacts.

This is deliberately a new report pipeline.  It does not overwrite the original
or condensed reports.  The final-year comparison keeps the previously selected
ten-pair universe fixed, while the nested walk-forward diagnostic reselects ten
pairs and refits non-equal constrained weights inside every supplied training
fold.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from equity_pairs.allocation import AllocationResult, allocate_pair_sleeves
from equity_pairs.backtest import performance_metrics
from equity_pairs.config import StrategyConfig
from equity_pairs.execution import (
    ExecutableCostModel,
    ExecutionCostSpec,
    ExecutionMarketInputs,
    build_execution_market_inputs,
)
from equity_pairs.grid_search import GridSearchSpec, build_parameter_return_cache
from equity_pairs.pair_health import (
    PairHealthConfig,
    PairHealthReference,
    apply_kill_switch,
    classify_pair_health,
    gate_target_positions,
    rolling_pair_diagnostics,
)
from equity_pairs.selection import (
    PairSelectionSpec,
    build_pair_selection_features,
    select_constrained_pairs_milp,
)
from equity_pairs.validation import (
    NestedWalkForwardSpec,
    run_nested_walk_forward,
)
from equity_pairs.weighted_portfolio import (
    WeightedPortfolioResult,
    backtest_weighted_pair_book,
    build_gross_name_incidence,
)


BLUE = "#245c97"
TEAL = "#17817d"
ORANGE = "#d77a24"
RED = "#bd3d4a"
PURPLE = "#7652a5"
INK = "#20334e"


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.{digits}f}%" if math.isfinite(number) else "—"


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _strategy_only_metrics(result: WeightedPortfolioResult) -> dict[str, float]:
    """Remove free-cash yield while retaining trading and short-financing economics."""
    prior_equity = result.daily["equity"].shift(1, fill_value=1.0)
    cash_contribution = result.daily["cash_income"].div(prior_equity).fillna(0.0)
    strategy_return = result.daily["portfolio_return"] - cash_contribution
    strategy_return = strategy_return.mask(strategy_return.abs() < 1e-12, 0.0)
    return performance_metrics(strategy_return)


def _html_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    return frame.to_html(
        index=index,
        border=0,
        classes="data-table",
        justify="left",
        escape=True,
        na_rep="—",
    )


def _pair_market_betas(pair_returns: pd.DataFrame, market: pd.Series) -> pd.Series:
    values: dict[str, float] = {}
    for pair in pair_returns.columns:
        aligned = pd.concat(
            [
                pd.to_numeric(pair_returns[pair], errors="coerce").rename("pair"),
                pd.to_numeric(market, errors="coerce").rename("market"),
            ],
            axis=1,
        ).dropna()
        variance = float(aligned["market"].var(ddof=1)) if len(aligned) > 2 else 0.0
        values[str(pair)] = (
            float(aligned["pair"].cov(aligned["market"]) / variance)
            if variance > 0
            else 0.0
        )
    return pd.Series(values, dtype=float, name="market_beta")


def _macro_factor_betas(
    pair_returns: pd.DataFrame,
    macro_factors: pd.DataFrame,
) -> pd.DataFrame:
    factors = macro_factors.reindex(pair_returns.index).apply(
        pd.to_numeric, errors="coerce"
    )
    means = factors.mean()
    scales = factors.std(ddof=1).replace(0.0, np.nan)
    standardized = (factors - means) / scales
    rows: dict[str, dict[str, float]] = {}
    for pair in pair_returns.columns:
        aligned = pd.concat(
            [pair_returns[pair].rename("pair"), standardized], axis=1
        ).dropna()
        if len(aligned) < max(20, len(standardized.columns) + 3):
            rows[str(pair)] = {column: 0.0 for column in standardized.columns}
            continue
        design = np.column_stack(
            [np.ones(len(aligned)), aligned[standardized.columns].to_numpy()]
        )
        coefficients = np.linalg.lstsq(
            design,
            aligned["pair"].to_numpy(dtype=float),
            rcond=None,
        )[0][1:]
        rows[str(pair)] = dict(zip(standardized.columns, coefficients, strict=True))
    return pd.DataFrame.from_dict(rows, orient="index").reindex(pair_returns.columns)


def _allocate_constrained_book(
    pair_returns: pd.DataFrame,
    positions: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    market: pd.Series,
    macro_factors: pd.DataFrame,
    *,
    market_beta_bound: float,
    minimum_weight: float,
    maximum_weight: float,
) -> tuple[AllocationResult, dict[str, Any]]:
    pairs = selected_metadata["pair"].astype(str).tolist()
    returns = pair_returns.reindex(columns=pairs).fillna(0.0)
    position_panel = positions.reindex(index=returns.index, columns=pairs).fillna(0.0)
    metadata = selected_metadata.set_index("pair").reindex(pairs)
    beta = pd.to_numeric(metadata["beta"], errors="coerce")
    if beta.isna().any() or (beta <= 0).any():
        raise ValueError("selected pairs require finite positive hedge betas")

    pair_gross = position_panel.abs().mean().rename("active_gross")
    structural_net = (1.0 - beta) / (1.0 + beta)
    pair_net = position_panel.mean().mul(structural_net).rename("mean_net")
    market_betas = _pair_market_betas(returns, market.reindex(returns.index))
    factor_betas = _macro_factor_betas(returns, macro_factors)
    name_incidence = build_gross_name_incidence(selected_metadata)
    expected_returns = returns.mean() * 252.0
    equal = pd.Series(1.0 / len(pairs), index=pairs)

    equal_gross = float(equal @ pair_gross)
    gross_bounds = (max(0.0, equal_gross - 0.08), min(1.0, equal_gross + 0.08))
    equal_net = float(equal @ pair_net)
    net_limit = max(0.03, abs(equal_net) + 0.015)
    factor_bounds: dict[str, tuple[float, float]] = {}
    for factor in factor_betas.columns:
        equal_exposure = float(equal @ factor_betas[factor])
        limit = max(0.0015, abs(equal_exposure) + 0.0005)
        factor_bounds[str(factor)] = (-limit, limit)
    target_return = float(expected_returns.mean())
    sector_caps = {
        str(sector): 0.25 for sector in selected_metadata["sector"].unique()
    }
    allocation = allocate_pair_sleeves(
        returns,
        method="minimum_covariance",
        expected_returns=expected_returns,
        pair_betas=market_betas,
        pair_metadata=selected_metadata,
        lower_bounds=minimum_weight,
        upper_bounds=maximum_weight,
        sector_caps=sector_caps,
        name_incidence_caps=0.10,
        name_incidence=name_incidence,
        pair_gross_exposures=pair_gross,
        gross_exposure_bounds=gross_bounds,
        pair_net_exposures=pair_net,
        net_exposure_bounds=(-net_limit, net_limit),
        market_beta_bounds=(-market_beta_bound, market_beta_bound),
        factor_betas=factor_betas,
        factor_beta_bounds=factor_bounds,
        target_return=target_return,
        mean_shrinkage=0.50,
        fallback_method="inverse_volatility",
    )
    audit = {
        "target_return": target_return,
        "gross_bounds": gross_bounds,
        "net_bounds": (-net_limit, net_limit),
        "factor_bounds": factor_bounds,
        "pair_gross_exposures": pair_gross,
        "pair_net_exposures": pair_net,
        "market_betas": market_betas,
        "factor_betas": factor_betas,
    }
    return allocation, audit


def _rolling_allocation_schedule(
    pair_returns: pd.DataFrame,
    positions: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    market: pd.Series,
    macro_factors: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, AllocationResult, dict[str, Any]]:
    test_dates = pair_returns.index[(pair_returns.index >= start) & (pair_returns.index <= end)]
    rebalance_dates = pd.Series(test_dates, index=test_dates).groupby(
        test_dates.to_period("M")
    ).first().tolist()
    schedule_rows: list[pd.Series] = []
    diagnostic_rows: list[dict[str, Any]] = []
    first_allocation: AllocationResult | None = None
    first_audit: dict[str, Any] | None = None
    for rebalance_date in rebalance_dates:
        eligible = pair_returns.index[pair_returns.index < rebalance_date]
        if len(eligible) < 126:
            raise ValueError("rolling allocation requires at least 126 prior sessions")
        estimation_index = eligible[-504:]
        allocation, audit = _allocate_constrained_book(
            pair_returns.reindex(estimation_index),
            positions.reindex(estimation_index),
            selected_metadata,
            market,
            macro_factors,
            market_beta_bound=0.05,
            minimum_weight=0.04,
            maximum_weight=0.18,
        )
        if first_allocation is None:
            first_allocation = allocation
            first_audit = audit
        schedule_rows.append(allocation.weights.rename(pd.Timestamp(rebalance_date)))
        diagnostic_rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "estimation_start": estimation_index.min(),
                "estimation_end": estimation_index.max(),
                **{
                    key: value
                    for key, value in allocation.diagnostics.items()
                    if not isinstance(value, (dict, list, tuple))
                },
                "target_return": audit["target_return"],
                "gross_lower": audit["gross_bounds"][0],
                "gross_upper": audit["gross_bounds"][1],
                "net_lower": audit["net_bounds"][0],
                "net_upper": audit["net_bounds"][1],
            }
        )
    assert first_allocation is not None and first_audit is not None
    schedule = pd.DataFrame(schedule_rows)
    schedule.index.name = "rebalance_date"
    return schedule, pd.DataFrame(diagnostic_rows), first_allocation, first_audit


def _build_health_gated_signals(
    selected_signals: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    config: PairHealthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gated_pieces: list[pd.DataFrame] = []
    health_pieces: list[pd.DataFrame] = []
    event_pieces: list[pd.DataFrame] = []
    metadata = selected_metadata.set_index("pair")
    for pair in selected_metadata["pair"].astype(str):
        row = metadata.loc[pair]
        reference = PairHealthReference(
            pair=pair,
            dependent=str(row["dependent"]),
            independent=str(row["independent"]),
            beta=float(row["beta"]),
            half_life_days=float(row["half_life_days"])
            if np.isfinite(float(row["half_life_days"]))
            and float(row["half_life_days"]) > 0
            else None,
        )
        diagnostics = rolling_pair_diagnostics(reference, prices, config)
        diagnostics = diagnostics.loc[diagnostics.index >= start]
        classified = classify_pair_health(diagnostics, reference, config)
        health = apply_kill_switch(classified, config)
        health_pieces.append(health.assign(pair=pair).reset_index())
        pair_events = health.loc[health["kill_event"] | health["recovery_event"]].copy()
        if len(pair_events):
            event_pieces.append(pair_events.assign(pair=pair).reset_index())

        signal = selected_signals.loc[selected_signals["pair"].eq(pair)].copy()
        signal = signal.sort_values("date").set_index("date")
        target = gate_target_positions(signal["target_position"], health)
        effective = target.shift(1, fill_value=0.0)
        return_position = effective.shift(1, fill_value=0.0)
        beta = float(row["beta"])
        base_dependent = 1.0 / (1.0 + abs(beta))
        base_independent = -beta / (1.0 + abs(beta))
        signal["target_position"] = target
        signal["position"] = effective
        signal["return_position"] = return_position
        signal["weight_dependent"] = effective * base_dependent
        signal["weight_independent"] = effective * base_independent
        signal["return_weight_dependent"] = return_position * base_dependent
        signal["return_weight_independent"] = return_position * base_independent
        dependent_return = signal["price_dependent"].pct_change(fill_method=None).fillna(0.0)
        independent_return = signal["price_independent"].pct_change(fill_method=None).fillna(0.0)
        signal["gross_return"] = (
            signal["return_weight_dependent"] * dependent_return
            + signal["return_weight_independent"] * independent_return
        )
        short_notional = (
            -signal[["return_weight_dependent", "return_weight_independent"]]
            .clip(upper=0.0)
            .sum(axis=1)
        )
        signal["borrow_cost"] = short_notional * 30.0 / 10_000.0 / 252.0
        signal["health_state"] = health["health_state"].reindex(signal.index).fillna(
            "missing_fail_closed"
        )
        signal["health_allow_entry"] = health["allow_entry"].reindex(signal.index).fillna(False)
        signal["health_force_exit"] = health["force_exit"].reindex(signal.index).fillna(True)
        gated_pieces.append(signal.reset_index())
    gated = pd.concat(gated_pieces, ignore_index=True)
    daily_health = pd.concat(health_pieces, ignore_index=True)
    events = pd.concat(event_pieces, ignore_index=True) if event_pieces else pd.DataFrame()
    return gated, daily_health, events


def _execution_inputs(
    output: Path,
    prices: pd.DataFrame,
    tickers: list[str],
    macro: pd.DataFrame,
) -> tuple[ExecutionMarketInputs, dict[str, Any]]:
    cache = output / "data" / "raw" / "yahoo_adjusted_ohlcv_top10.csv"
    cash_rate = (
        pd.to_numeric(macro["DFF"], errors="coerce")
        .reindex(prices.index)
        .ffill()
        .shift(1)
        .div(100.0)
    )
    if cache.is_file() and cache.with_name(cache.name + ".meta.json").is_file():
        from equity_pairs.data import download_adjusted_ohlcv

        ohlcv = download_adjusted_ohlcv(
            tickers,
            "2023-07-01",
            "2026-08-27",
            strict=True,
            cache_path=cache,
        )
        inputs = build_execution_market_inputs(
            ohlcv["Close"],
            ohlcv["Volume"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            annual_cash_rate=cash_rate,
            lookback=20,
            minimum_periods=10,
            range_to_half_spread=0.05,
        )
        status = {
            "liquidity_source": "cached Yahoo adjusted OHLCV",
            "liquidity_proxy": False,
            "spread_source": "5% of lagged median daily high-low range; proxy, not quote",
            "download_status": "cache available",
        }
        return inputs, status

    close = prices.reindex(columns=tickers)
    volatility = close.pct_change(fill_method=None).rolling(20, min_periods=10).std().shift(1)
    missing_adv = pd.DataFrame(np.nan, index=close.index, columns=tickers)
    inputs = ExecutionMarketInputs(
        dollar_adv=missing_adv,
        daily_volatility=volatility,
        annual_cash_rate=cash_rate,
        provenance={
            "adv_source": "$25m per-name fallback; Yahoo OHLCV unavailable due HTTP 429",
            "volatility_source": "lagged 20-session Yahoo adjusted-close volatility",
            "spread_source": "2.5 bp half-spread fallback",
            "borrow_source": "50 bp annual fallback",
            "locate_source": "assumed available; not verified",
        },
    )
    status = {
        "liquidity_source": "$25m ADV fallback assumption",
        "liquidity_proxy": True,
        "spread_source": "2.5 bp half-spread fallback",
        "download_status": "Yahoo OHLCV request returned HTTP 429; no cache created",
    }
    return inputs, status


def _run_execution_scenarios(
    signals: pd.DataFrame,
    weights: pd.Series,
    schedule: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    inputs: ExecutionMarketInputs,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, WeightedPortfolioResult], pd.DataFrame]:
    results: dict[str, WeightedPortfolioResult] = {}
    rows: list[dict[str, Any]] = []
    for notional in (1_000_000.0, 10_000_000.0, 50_000_000.0):
        label = f"${notional / 1_000_000:.0f}m"
        model = ExecutableCostModel(
            inputs,
            ExecutionCostSpec(
                portfolio_notional=notional,
                commission_bps=0.10,
                fallback_half_spread_bps=2.50,
                impact_coefficient=0.10,
                maximum_participation_rate=0.05,
                default_adv_dollars=25_000_000.0,
                default_daily_volatility=0.02,
                default_annual_borrow_bps=50.0,
                financing_spread_bps=100.0,
                cash_yield_haircut_bps=25.0,
                short_rebate_spread_bps=50.0,
                reject_capacity_breaches=True,
                reject_locate_failures=True,
            ),
        )
        result = backtest_weighted_pair_book(
            signals,
            weights,
            start=start,
            end=end,
            rebalance_frequency="monthly",
            no_trade_band=0.01,
            maximum_rebalance_turnover=0.15,
            charge_terminal_liquidation=True,
            execution_cost_model=model,
            pair_metadata=selected_metadata,
            fail_on_unexecutable=False,
            target_weight_schedule=schedule,
        )
        results[label] = result
        strategy_metrics = _strategy_only_metrics(result)
        rows.append(
            {
                "portfolio_notional": label,
                "cagr": result.metrics["cagr"],
                "sharpe": result.metrics["sharpe"],
                "strategy_only_cagr": strategy_metrics["cagr"],
                "strategy_only_sharpe": strategy_metrics["sharpe"],
                "max_drawdown": result.metrics["max_drawdown"],
                "transaction_cost_bps": result.metrics["transaction_cost_drag_bps"],
                "borrow_cost_bps": result.metrics["borrow_cost_drag_bps"],
                "maximum_participation": result.metrics["maximum_participation_rate"],
                "capacity_breaches": result.metrics["capacity_breach_count"],
                "capacity_breach_days": int(
                    result.daily["capacity_breach_count"].gt(0).sum()
                ),
                "locate_failures": result.metrics["locate_failure_count"],
            }
        )
    return results, pd.DataFrame(rows)


def _run_nested_validation(
    all_metadata: pd.DataFrame,
    prices: pd.DataFrame,
    market: pd.Series,
    macro_factors: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[Any, Any, pd.DataFrame]:
    half_life = pd.to_numeric(
        all_metadata.set_index("pair")["half_life_days"], errors="coerce"
    )
    half_life = half_life.fillna(half_life.median()).clip(lower=2.0)
    formation_weights = (1.0 / half_life)
    formation_weights /= formation_weights.sum()
    grid = GridSearchSpec(
        zscore_methods=("sma", "ewma"),
        zscore_lookbacks=(10, 20, 40, 60),
        entry_zs=(1.25, 1.75),
        exit_zs=(0.25, 0.50),
        maximum_holding_days=(20, 40),
    )
    base_config = StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        maximum_holding_days=60,
        transaction_cost_bps=5.0,
        annual_short_borrow_bps=30.0,
    )
    cache = build_parameter_return_cache(
        all_metadata,
        prices,
        start,
        end,
        base_config,
        grid,
        formation_weights,
        include_baseline=True,
    )
    selection_spec = PairSelectionSpec(
        pair_count=10,
        maximum_pairs_per_sector=2,
        maximum_pairs_per_ticker=1,
        minimum_sector_count=6,
        minimum_trades=2,
    )
    audit_rows: list[dict[str, Any]] = []

    def evaluator(
        candidate_id: str,
        train_index: pd.Index,
        evaluation_index: pd.Index,
    ) -> pd.Series:
        candidate_returns = cache.pair_returns[candidate_id]
        train = candidate_returns.reindex(train_index).fillna(0.0)
        entries = cache.entries[candidate_id].reindex(train_index).fillna(False).sum()
        features = build_pair_selection_features(
            train,
            all_metadata,
            spec=selection_spec,
        )
        features["tuning_trades"] = (
            features["pair"].map(entries).fillna(0).astype(int)
        )
        betas = _pair_market_betas(train, market.reindex(train.index))
        selected = select_constrained_pairs_milp(
            features,
            selection_spec,
            pair_betas=betas,
            equal_weight_beta_bound=0.10,
            pair_correlations=train.corr(),
            maximum_pair_correlation=0.90,
        )
        pairs = selected["pair"].astype(str).tolist()
        selected_metadata = (
            all_metadata.set_index("pair").reindex(pairs).reset_index()
        )
        positions = cache.positions[candidate_id].reindex(
            index=train.index, columns=pairs
        )
        allocation, allocation_audit = _allocate_constrained_book(
            train[pairs],
            positions,
            selected_metadata,
            market,
            macro_factors,
            market_beta_bound=0.10,
            minimum_weight=0.02,
            maximum_weight=0.20,
        )
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "train_start": train_index[0],
                "train_end": train_index[-1],
                "evaluation_start": evaluation_index[0],
                "evaluation_end": evaluation_index[-1],
                "selected_pairs": "|".join(pairs),
                "weights": "|".join(
                    f"{pair}:{allocation.weights[pair]:.8f}" for pair in pairs
                ),
                "target_return": allocation_audit["target_return"],
                "expected_market_beta": allocation.diagnostics.get(
                    "expected_market_beta", math.nan
                ),
                "expected_volatility": allocation.diagnostics.get(
                    "expected_annual_volatility", math.nan
                ),
            }
        )
        evaluation = candidate_returns.reindex(
            index=evaluation_index, columns=pairs
        ).fillna(0.0)
        return evaluation.mul(allocation.weights, axis=1).sum(axis=1)

    validation_spec = NestedWalkForwardSpec(
        outer_train_size=350,
        outer_test_size=63,
        inner_train_size=175,
        inner_validation_size=42,
        purge_size=40,
        embargo_size=5,
        expanding=False,
        maximum_outer_folds=5,
        maximum_inner_folds=3,
        selection_metric="sharpe",
        minimum_evaluation_observations=30,
        bootstrap_samples=1_000,
        bootstrap_block_size=10,
        confidence_level=0.95,
        random_seed=20260829,
    )
    candidates = {
        str(config_id): str(config_id) for config_id in cache.returns.columns
    }
    result = run_nested_walk_forward(
        cache.returns.index,
        candidates,
        evaluator,
        validation_spec,
    )
    return result, cache, pd.DataFrame(audit_rows)


def _plot_weights(schedule: pd.DataFrame, path: Path) -> None:
    ordered = schedule.iloc[0].sort_values()
    latest = schedule.iloc[-1].reindex(ordered.index)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(ordered.index.str.replace("__", " / "), ordered * 100, color=BLUE)
    axes[0].set_title("First lockbox target")
    axes[0].set_xlabel("Pair capital weight (%)")
    axes[1].plot(schedule.index, schedule.mul(100), linewidth=1.7)
    axes[1].scatter([schedule.index[-1]] * len(latest), latest * 100, s=18)
    axes[1].set_title("Causal monthly targets")
    axes[1].set_ylabel("Weight (%)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(
        [pair.replace("__", " / ") for pair in schedule.columns],
        fontsize=7,
        ncol=2,
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_equity(comparisons: dict[str, WeightedPortfolioResult], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 5.7))
    colors = [PURPLE, BLUE, TEAL, ORANGE]
    for (label, result), color in zip(comparisons.items(), colors, strict=False):
        axis.plot(result.daily.index, result.daily["equity"], label=label, color=color, linewidth=2)
    axis.axhline(1.0, color="#8b98a8", linewidth=1)
    axis.set_title("Final-year book: allocation, execution, and health overlays")
    axis.set_ylabel("Growth of $1")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_health(health: pd.DataFrame, path: Path) -> None:
    summary = (
        health.assign(forced_flat=health["target_gross_multiplier"].eq(0.0))
        .groupby("pair")
        .agg(
            tradable_fraction=("allow_entry", "mean"),
            forced_flat_fraction=("forced_flat", "mean"),
            kills=("kill_event", "sum"),
        )
        .sort_values("forced_flat_fraction")
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].barh(
        summary.index.str.replace("__", " / "),
        summary["forced_flat_fraction"] * 100,
        color=RED,
        alpha=0.85,
    )
    axes[0].set_title("Strict health gate: time forced flat")
    axes[0].set_xlabel("Held-out sessions (%)")
    states = health["health_state"].value_counts(normalize=True).sort_values()
    axes[1].barh(states.index.astype(str), states * 100, color=ORANGE)
    axes[1].set_title("Pair-session health states")
    axes[1].set_xlabel("All selected pair-sessions (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_execution(scenarios: pd.DataFrame, primary: WeightedPortfolioResult, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    components = pd.Series(
        {
            "Commission": primary.metrics["commission_cost_sum"],
            "Spread": primary.metrics["spread_cost_sum"],
            "Impact": primary.metrics["impact_cost_sum"],
            "Borrow": primary.metrics["borrow_cost_sum"],
            "Financing": primary.metrics["financing_cost_sum"],
            "Cash yield": -primary.metrics["cash_income_sum"],
            "Short rebate": -primary.metrics["short_rebate_income_sum"],
        }
    ) * 10_000
    axes[0].barh(
        components.index,
        components.values,
        color=[RED if value >= 0 else TEAL for value in components.values],
    )
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].set_title("$10m cost/carry components")
    axes[0].set_xlabel("Sum in initial-NAV bp; income is negative cost")
    axes[1].plot(
        scenarios["portfolio_notional"],
        scenarios["maximum_participation"] * 100,
        marker="o",
        color=BLUE,
        linewidth=2,
    )
    axes[1].axhline(5.0, color=RED, linestyle="--", label="5% ADV hard cap")
    axes[1].set_title("Capacity sensitivity")
    axes[1].set_ylabel("Maximum assumed ADV participation (%)")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_nested(result: Any, path: Path) -> None:
    oos = result.oos_returns.copy()
    equity = (1.0 + oos["return"]).cumprod()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(equity.index, equity, color=TEAL, linewidth=2.2)
    for _, row in result.outer_evaluations.iterrows():
        axes[0].axvline(pd.Timestamp(row["test_start"]), color="#cbd5df", linewidth=0.8)
    axes[0].axhline(1.0, color="#8b98a8", linewidth=1)
    axes[0].set_title("Nested outer-lockbox path")
    axes[0].set_ylabel("Growth of $1")
    labels = result.outer_evaluations["candidate_id"].str.replace("__", "\n", regex=False)
    axes[1].bar(range(len(labels)), result.outer_evaluations["sharpe"], color=BLUE)
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[1].set_xticks(range(len(labels)), [f"Fold {i + 1}" for i in range(len(labels))])
    axes[1].set_title("Outer-fold Sharpe; parameters frozen per fold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


REPORT_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Pairs — Production Gate Review</title>
<style>
:root{--ink:#20334e;--muted:#637287;--paper:#fff;--line:#dce3eb;--blue:#245c97;--teal:#17817d;--orange:#d77a24;--red:#bd3d4a;--pale:#edf4fa}
*{box-sizing:border-box}body{margin:0;background:#e9eef4;color:var(--ink);font:15px/1.56 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1220px;margin:24px auto;background:var(--paper);box-shadow:0 14px 42px rgba(32,52,78,.13);overflow:hidden}header{padding:50px 64px 34px;background:linear-gradient(135deg,#15243d 0%,#245c97 62%,#17817d 100%);color:#fff}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:750;opacity:.8}h1{font:700 41px/1.08 Georgia,serif;margin:12px 0 14px;max-width:940px}header p{max-width:900px;font-size:17px;margin:0;opacity:.9}.content{padding:38px 64px 64px}h2{font:700 27px/1.2 Georgia,serif;margin:46px 0 13px}h3{font-size:17px;margin:28px 0 8px}.lead{font-size:17px;color:#33445b}.verdict{border-left:5px solid var(--red);background:#fff0f1;padding:20px 22px;margin:0 0 25px;font-size:16px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:22px 0}.card{border:1px solid var(--line);border-radius:8px;padding:16px 17px;background:#fff}.card .k{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:750}.card .v{font-size:27px;line-height:1.15;font-weight:760;margin:7px 0 3px}.card .s{font-size:12px;color:var(--muted)}.gate{display:grid;grid-template-columns:165px 1fr 105px;gap:15px;align-items:start;padding:15px 0;border-bottom:1px solid var(--line)}.status{display:inline-block;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:750;text-align:center}.pass{background:#e6f5ef;color:#146847}.fail{background:#ffe8ea;color:#9b2431}.conditional{background:#fff0db;color:#9a5a13}.chart{display:block;width:100%;border:1px solid var(--line);border-radius:7px;margin:18px 0 11px}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.grid2>div{min-width:0}.data-table{border-collapse:collapse;width:100%;font-size:12.3px;margin:13px 0 22px}.data-table th{background:#edf2f7;text-align:left;padding:9px 8px;border-bottom:2px solid #cbd5e1;white-space:nowrap}.data-table td{padding:8px;border-bottom:1px solid #e4e9ef;vertical-align:top}.scroll{overflow-x:auto;max-width:100%}.note{padding:15px 18px;border-radius:7px;background:var(--pale);margin:14px 0}.note.bad{background:#fff0f1;border-left:4px solid var(--red)}.note.warn{background:#fff6e9;border-left:4px solid var(--orange)}code{background:#f1f4f7;padding:2px 5px;border-radius:4px}.muted{color:var(--muted)}a{color:#1e61ad}.foot{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}@media(max-width:880px){.page{margin:0}.content,header{padding-left:22px;padding-right:22px}.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}.gate{grid-template-columns:1fr}.status{width:max-content}}@media print{body{background:#fff}.page{box-shadow:none;margin:0}.chart{break-inside:avoid}}
</style></head><body><main class="page"><header><div class="eyebrow">Equity pairs · production controls · {{ as_of }}</div><h1>The implementation is stronger. The current ten-pair book still does not clear deployment.</h1><p>A separate, auditable review of constrained non-equal allocation, rolling pair health, nested validation, and executable-cost mechanics. Existing reports are unchanged.</p></header><section class="content">
<div class="verdict"><strong>Decision: research / paper trade only.</strong> The allocator satisfies its modeled constraints, but strict rolling health leaves {{ tradable_latest }}/10 pairs tradable at the end of the sample; historical borrow locates and quote/ADV data are not verified; and the nested result uses already-viewed history rather than a restored virgin lockbox.</div>
<div class="cards"><div class="card"><div class="k">$10m pairs-alpha CAGR</div><div class="v">{{ execution_alpha_cagr }}</div><div class="s">Free-cash yield stripped; proxy execution inputs</div></div><div class="card"><div class="k">Strict-health pairs alpha</div><div class="v">{{ health_alpha_cagr }}</div><div class="s">Cash-only carry stripped</div></div><div class="card"><div class="k">Nested OOS CAGR</div><div class="v">{{ nested_cagr }}</div><div class="s">{{ nested_obs }} outer-lockbox sessions</div></div><div class="card"><div class="k">Latest tradable</div><div class="v">{{ tradable_latest }}/10</div><div class="s">Rolling health state at {{ sample_end }}</div></div></div>

<h2>Gate board</h2>
<div class="gate"><strong>Constrained allocator</strong><div>Monthly, one-session-lagged refits minimize Ledoit–Wolf covariance subject to a forecast-return floor, market and FRED-derived factor bands, gross/net exposure bands, 4–18% sleeve bounds, 25% sector caps, and 10% gross-normalized name caps. Targets remain non-equal.</div><span class="status {{ allocator_class }}">{{ allocator_status }}</span></div>
<div class="gate"><strong>Rolling pair health</strong><div>Trailing 252-session Engle–Granger and residual ADF, hedge-beta drift, half-life drift, midpoint Chow proxy, two-breach kill, 20-session cooldown, and three-good-session recovery. Every decision has a lagged <code>as_of_date</code>.</div><span class="status {{ health_class }}">{{ health_status }}</span></div>
<div class="gate"><strong>Nested walk-forward</strong><div>{{ candidate_count }} predeclared SMA/EWMA parameter cells; five chronological outer folds; purging, embargo, frozen fold parameters, reselection of ten pairs, refit non-equal weights, block-bootstrap intervals, and a joint centered reality-check-style diagnostic.</div><span class="status {{ nested_class }}">{{ nested_status }}</span></div>
<div class="gate"><strong>Executable costs</strong><div>Underlying-level netting, commission, half-spread, square-root participation impact, ADV cap, borrow, locate, financing, cash yield, short rebate, and terminal liquidation. The mechanics run, but the primary cost inputs remain proxies.</div><span class="status conditional">CONDITIONAL</span></div>

<h2>1 · Non-equal constrained portfolio</h2>
<p class="lead">The backtest now refits targets at each month’s first trading session using only the prior 504 sessions. Actual deployment moves toward the new target through a 1% no-trade band and a 15% two-way capital-turnover cap.</p>
<img class="chart" src="charts/production_gate_weights.png" alt="Non-equal constrained pair weights and rolling targets">
<div class="scroll">{{ weight_table }}</div>
<div class="note"><strong>Return constraint is a forecast floor, not a promise.</strong> The first target floor was {{ target_return }} annualized, equal to the cross-sectional mean of the training forecasts after shrinkage. Realized final-year performance is shown below.</div>

<h2>2 · What each control changes</h2>
<img class="chart" src="charts/production_gate_equity.png" alt="Equity comparison for allocator execution and health overlays">
<div class="scroll">{{ comparison_table }}</div>
<p>The health-gated return can resemble cash when every spread is quarantined. Total CAGR retains modeled cash carry; “pairs-alpha CAGR” strips free-cash yield so cash is never mistaken for strategy skill.</p>

<h2>3 · Rolling pair health</h2>
<img class="chart" src="charts/production_gate_health.png" alt="Strict rolling pair health states">
<div class="scroll">{{ health_table }}</div>
<div class="note bad"><strong>This gate is doing consequential work:</strong> a pair selected on historical return or one-time cointegration is not allowed to keep trading after its live relation fails the rolling diagnostics. The result is economically severe because eight of the original ten were already deterministic formation fallbacks.</div>

<h2>4 · Executable cost and capacity model</h2>
<img class="chart" src="charts/production_gate_execution.png" alt="Execution cost components and ADV capacity sensitivity">
<div class="scroll">{{ execution_table }}</div>
<div class="note warn"><strong>Data limitation:</strong> {{ liquidity_status }}. Historical Yahoo adjusted closes drive volatility, and FRED DFF drives lagged cash carry. Historical bid/ask, securities-lending rates, and locates are not supplied by Yahoo, so the $10m result is a scenario—not a broker-verified executable P&amp;L.</div>

<h2>5 · Nested, purged walk-forward</h2>
<img class="chart" src="charts/production_gate_nested.png" alt="Nested walk-forward out-of-sample results">
<div class="scroll">{{ nested_table }}</div>
<div class="grid2"><div><h3>Bootstrap interval</h3><div class="scroll">{{ bootstrap_table }}</div></div><div><h3>Data-snooping diagnostic</h3><div class="scroll">{{ snooping_table }}</div></div></div>
<div class="note warn"><strong>Correct interpretation:</strong> outer-fold choices were frozen and never ranked on their own tests, but all dates in this dataset have now been observed. This is a retrospective robustness audit conditional on the original formation relations—not a newly untouched investment result. The centered joint reality-check label is a “style” diagnostic, not an exact finite-sample proof.</div>

<h2>Risk-control rule now</h2>
<div class="grid2"><div><h3>Before a trade</h3><ul><li>Pair must be eligible under the rolling health state.</li><li>Target weights come from a lagged constrained covariance solve.</li><li>Trade rejects when modeled participation exceeds 5% of ADV or a required locate fails.</li><li>Signals are next-session effective; terminal positions are liquidated and charged.</li></ul></div><div><h3>While live</h3><ul><li>Mean-reversion exit, z-score stop, and maximum-holding exit remain active.</li><li>Two consecutive health breaches force liquidation and quarantine.</li><li>Monthly allocation refits observe sector, name, market/factor, gross/net, and sleeve bounds.</li><li>Cooldown needs three healthy observations before re-entry.</li></ul></div></div>

<h2>Files and reproducibility</h2>
<p>Machine-readable allocation schedules, constraint diagnostics, health histories, execution logs, scenario metrics, fold audits, selections, bootstrap intervals, and data-snooping diagnostics are written beside this report under <code>data/processed/production_*</code>.</p>
<p>Method references: <a href="https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html">official yfinance download API</a>; <a href="https://web.stanford.edu/~boyd/papers/portfolio.html">Lobo, Fazel &amp; Boyd on portfolio optimization with transaction costs</a>; <a href="https://web.stanford.edu/~boyd/papers/cvx_portfolio.html">Boyd et al. on multi-period trading</a>; and <a href="https://www.sec.gov/files/rules/final/2008/34-58774.pdf">SEC Regulation SHO locate requirements</a>.</p>
<p>Earlier outputs: <a href="weighted_portfolio_report.html">weighted top-ten report</a> · <a href="condensed_report.html">condensed research report</a> · <a href="report.html">full original report</a>.</p>
<div class="foot">Generated {{ generated_at }} · Sample ends {{ sample_end }} · Yahoo adjusted-price history and saved FRED macro series · Research, not investment advice.</div>
</section></main></body></html>"""
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pairs_research_2026-08-26"),
    )
    args = parser.parse_args()
    output = args.output_dir
    processed = output / "data" / "processed"
    raw = output / "data" / "raw"
    charts = output / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(
        raw / "yahoo_adjusted_close_8y.csv", index_col="date", parse_dates=True
    ).sort_index()
    all_metadata = pd.read_csv(processed / "formation_selected_top10_by_sector.csv")
    selected = pd.read_csv(processed / "weighted_selected_top10.csv")
    pairs = selected["pair"].astype(str).tolist()
    selected_metadata = (
        all_metadata.drop_duplicates("pair").set_index("pair").reindex(pairs).reset_index()
    )
    signals = pd.read_csv(
        processed / "grid_selected_pair_signals.csv.gz", parse_dates=["date"]
    )
    selected_signals = signals.loc[signals["pair"].isin(pairs)].copy()
    pair_returns = pd.read_csv(
        processed / "grid_selected_pair_returns.csv",
        index_col="date",
        parse_dates=True,
    ).sort_index()[pairs]
    market = pd.read_csv(
        processed / "followup_market_proxy_returns.csv",
        index_col="date",
        parse_dates=True,
    ).iloc[:, 0]
    macro = pd.read_csv(
        processed / "fred_macro_and_derived.csv", index_col="date", parse_dates=True
    ).sort_index()
    macro_on_prices = macro.reindex(prices.index).ffill()
    macro_factors = pd.DataFrame(
        {
            "rates": pd.to_numeric(macro_on_prices["DGS10"], errors="coerce").diff().div(100.0),
            "vix": pd.to_numeric(macro_on_prices["VIXCLS"], errors="coerce").pct_change(
                fill_method=None
            ),
        },
        index=prices.index,
    ).replace([np.inf, -np.inf], np.nan)

    grid_metadata = json.loads(
        (processed / "grid_search_metadata.json").read_text(encoding="utf-8")
    )
    start = pd.Timestamp(grid_metadata["lockbox_start"])
    end = pair_returns.index.max()
    positions = selected_signals.pivot(
        index="date", columns="pair", values="position"
    ).reindex(columns=pairs)

    schedule, schedule_diagnostics, first_allocation, first_audit = (
        _rolling_allocation_schedule(
            pair_returns,
            positions,
            selected_metadata,
            market,
            macro_factors,
            start,
            end,
        )
    )
    initial_weights = schedule.iloc[0].rename("weight")
    previous_weights = pd.read_csv(processed / "weighted_primary_weights.csv").set_index(
        "pair"
    )["weight"].reindex(pairs)

    previous_flat = backtest_weighted_pair_book(
        selected_signals,
        previous_weights,
        start=start,
        end=end,
        transaction_cost_bps=5.0,
        rebalance_frequency="monthly",
        no_trade_band=0.01,
        maximum_rebalance_turnover=0.15,
        charge_terminal_liquidation=True,
    )
    constrained_flat = backtest_weighted_pair_book(
        selected_signals,
        initial_weights,
        start=start,
        end=end,
        transaction_cost_bps=5.0,
        rebalance_frequency="monthly",
        no_trade_band=0.01,
        maximum_rebalance_turnover=0.15,
        charge_terminal_liquidation=True,
        target_weight_schedule=schedule,
    )

    health_config = PairHealthConfig(
        lookback_sessions=252,
        minimum_observations=126,
        decision_lag_sessions=1,
        maximum_cointegration_pvalue=0.10,
        maximum_residual_adf_pvalue=0.10,
        maximum_beta_relative_drift=0.35,
        maximum_half_life_days=126.0,
        maximum_half_life_multiple=3.0,
        minimum_structural_break_pvalue=0.01,
        bad_observations_to_kill=2,
        cooldown_sessions=20,
        good_observations_to_recover=3,
    )
    health_signals, health_daily, health_events = _build_health_gated_signals(
        selected_signals,
        selected_metadata,
        prices,
        pd.Timestamp(grid_metadata["heldout_start"]),
        health_config,
    )

    tickers = sorted(
        set(selected_metadata["dependent"].astype(str))
        | set(selected_metadata["independent"].astype(str))
    )
    execution_inputs, liquidity_status = _execution_inputs(output, prices, tickers, macro)
    execution_results, execution_scenarios = _run_execution_scenarios(
        selected_signals,
        initial_weights,
        schedule,
        selected_metadata,
        execution_inputs,
        start,
        end,
    )
    primary_execution = execution_results["$10m"]
    primary_model = ExecutableCostModel(
        execution_inputs,
        ExecutionCostSpec(portfolio_notional=10_000_000.0),
    )
    health_execution = backtest_weighted_pair_book(
        health_signals,
        initial_weights,
        start=start,
        end=end,
        rebalance_frequency="monthly",
        no_trade_band=0.01,
        maximum_rebalance_turnover=0.15,
        charge_terminal_liquidation=True,
        execution_cost_model=primary_model,
        pair_metadata=selected_metadata,
        fail_on_unexecutable=False,
        target_weight_schedule=schedule,
    )

    nested, parameter_cache, nested_allocation_audit = _run_nested_validation(
        all_metadata,
        prices,
        market,
        macro_factors,
        pd.Timestamp(grid_metadata["heldout_start"]) + pd.Timedelta(days=2),
        end,
    )

    # Machine-readable audit outputs.
    schedule.to_csv(processed / "production_weight_schedule.csv")
    schedule_diagnostics.to_csv(
        processed / "production_weight_schedule_diagnostics.csv", index=False
    )
    first_allocation.constraint_diagnostics.to_csv(
        processed / "production_constraint_diagnostics.csv", index=False
    )
    health_daily.to_csv(
        processed / "production_pair_health_daily.csv.gz", index=False, compression="gzip"
    )
    health_events.to_csv(processed / "production_pair_health_events.csv", index=False)
    health_signals.to_csv(
        processed / "production_health_gated_signals.csv.gz",
        index=False,
        compression="gzip",
    )
    primary_execution.daily.to_csv(processed / "production_executable_daily.csv")
    if primary_execution.execution_log is not None:
        primary_execution.execution_log.to_csv(
            processed / "production_execution_log.csv.gz",
            index=False,
            compression="gzip",
        )
    execution_scenarios.to_csv(
        processed / "production_execution_scenarios.csv", index=False
    )
    nested.oos_returns.to_csv(processed / "production_nested_oos_returns.csv")
    nested.performance.to_csv(processed / "production_nested_performance.csv", index=False)
    nested.fold_audit.to_csv(processed / "production_nested_fold_audit.csv", index=False)
    nested.inner_evaluations.to_csv(
        processed / "production_nested_inner_evaluations.csv", index=False
    )
    nested.selection_audit.to_csv(
        processed / "production_nested_selection_audit.csv", index=False
    )
    nested.outer_evaluations.to_csv(
        processed / "production_nested_outer_evaluations.csv", index=False
    )
    nested.bootstrap_confidence_intervals.to_csv(
        processed / "production_nested_bootstrap.csv", index=False
    )
    nested.data_snooping_diagnostics.to_csv(
        processed / "production_nested_snooping.csv", index=False
    )
    nested_allocation_audit.to_csv(
        processed / "production_nested_allocation_audit.csv", index=False
    )
    parameter_cache.parameters.to_csv(
        processed / "production_nested_candidate_parameters.csv", index=False
    )

    comparison_results = {
        "Prior ERC · flat 5 bp": previous_flat,
        "Rolling constrained · flat 5 bp": constrained_flat,
        "Rolling constrained · executable $10m": primary_execution,
        "Strict health + executable $10m": health_execution,
    }
    comparison_rows = []
    for label, result in comparison_results.items():
        strategy_metrics = _strategy_only_metrics(result)
        comparison_rows.append(
            {
                "method": label,
                "Total CAGR": _pct(result.metrics["cagr"]),
                "Pairs-alpha CAGR": _pct(strategy_metrics["cagr"]),
                "Alpha Sharpe": _num(strategy_metrics["sharpe"]),
                "Max drawdown": _pct(result.metrics["max_drawdown"]),
                "Turnover / year": _num(result.metrics["annualized_leg_turnover"], 1),
                "Transaction drag": f"{result.metrics['transaction_cost_drag_bps']:.1f} bp",
                "Borrow drag": f"{result.metrics['borrow_cost_drag_bps']:.1f} bp",
            }
        )
    comparison_table = pd.DataFrame(comparison_rows)

    health_final = health_daily.loc[
        pd.to_datetime(health_daily["date"]).between(start, end)
    ].copy()
    health_summary = (
        health_final.assign(
            forced_flat=health_final["target_gross_multiplier"].eq(0.0)
        )
        .groupby("pair")
        .agg(
            eligible_fraction=("allow_entry", "mean"),
            forced_flat_fraction=("forced_flat", "mean"),
            latest_state=("health_state", "last"),
        )
    )
    lifetime_events = health_daily.groupby("pair").agg(
        kills=("kill_event", "sum"),
        recoveries=("recovery_event", "sum"),
    )
    health_summary = health_summary.join(lifetime_events).reset_index()
    tradable_latest = int(health_summary["latest_state"].eq("tradable").sum())
    health_display = health_summary.copy()
    health_display["eligible_fraction"] = health_display["eligible_fraction"].map(_pct)
    health_display["forced_flat_fraction"] = health_display["forced_flat_fraction"].map(_pct)
    health_display.columns = [
        "Pair",
        "Entry eligible",
        "Forced flat",
        "Latest state",
        "Kills since monitoring",
        "Recoveries since monitoring",
    ]

    weight_display = pd.DataFrame(
        {
            "Pair": pairs,
            "First target": schedule.iloc[0].reindex(pairs).map(_pct).to_numpy(),
            "Latest target": schedule.iloc[-1].reindex(pairs).map(_pct).to_numpy(),
            "Minimum target": schedule[pairs].min().map(_pct).to_numpy(),
            "Maximum target": schedule[pairs].max().map(_pct).to_numpy(),
        }
    )
    execution_display = execution_scenarios.copy()
    execution_display["cagr"] = execution_display["cagr"].map(_pct)
    execution_display["sharpe"] = execution_display["sharpe"].map(_num)
    execution_display["strategy_only_cagr"] = execution_display[
        "strategy_only_cagr"
    ].map(_pct)
    execution_display["strategy_only_sharpe"] = execution_display[
        "strategy_only_sharpe"
    ].map(_num)
    execution_display["max_drawdown"] = execution_display["max_drawdown"].map(_pct)
    execution_display["maximum_participation"] = execution_display[
        "maximum_participation"
    ].map(_pct)
    execution_display["transaction_cost_bps"] = execution_display[
        "transaction_cost_bps"
    ].map(lambda value: f"{value:.1f} bp")
    execution_display["borrow_cost_bps"] = execution_display["borrow_cost_bps"].map(
        lambda value: f"{value:.1f} bp"
    )
    execution_display = execution_display[
        [
            "portfolio_notional",
            "cagr",
            "sharpe",
            "strategy_only_cagr",
            "strategy_only_sharpe",
            "max_drawdown",
            "transaction_cost_bps",
            "borrow_cost_bps",
            "maximum_participation",
            "capacity_breaches",
            "capacity_breach_days",
            "locate_failures",
        ]
    ]
    execution_display.columns = [
        "Notional",
        "Total CAGR",
        "Total Sharpe",
        "Pairs-alpha CAGR",
        "Alpha Sharpe",
        "Max drawdown",
        "Transaction drag",
        "Borrow drag",
        "Max participation",
        "Capacity breaches",
        "Breach days",
        "Locate failures",
    ]

    nested_display = nested.outer_evaluations[
        [
            "outer_fold",
            "candidate_id",
            "test_start",
            "test_end",
            "cagr",
            "sharpe",
            "max_drawdown",
        ]
    ].copy()
    nested_display["candidate_id"] = nested_display["candidate_id"].str.replace(
        "__", " · ", regex=False
    )
    nested_display["cagr"] = nested_display["cagr"].map(_pct)
    nested_display["sharpe"] = nested_display["sharpe"].map(_num)
    nested_display["max_drawdown"] = nested_display["max_drawdown"].map(_pct)
    nested_display.columns = [
        "Fold",
        "Frozen parameter",
        "Test start",
        "Test end",
        "CAGR",
        "Sharpe",
        "Max drawdown",
    ]

    bootstrap_display = nested.bootstrap_confidence_intervals.copy()
    snooping_display = nested.data_snooping_diagnostics.copy()
    for frame in (bootstrap_display, snooping_display):
        for column in frame.select_dtypes(include=["float"]).columns:
            frame[column] = frame[column].map(
                lambda value: f"{value:.4f}" if np.isfinite(value) else "—"
            )

    allocator_pass = bool(
        (first_allocation.constraint_diagnostics["slack"] >= -1e-6).all()
    )
    health_pass = tradable_latest >= 6
    nested_metrics = nested.performance.iloc[0]
    nested_pass = bool(
        np.isfinite(nested_metrics["sharpe"]) and nested_metrics["sharpe"] > 0
    )

    _plot_weights(schedule, charts / "production_gate_weights.png")
    _plot_equity(comparison_results, charts / "production_gate_equity.png")
    _plot_health(health_final, charts / "production_gate_health.png")
    _plot_execution(
        execution_scenarios,
        primary_execution,
        charts / "production_gate_execution.png",
    )
    _plot_nested(nested, charts / "production_gate_nested.png")

    execution_alpha_metrics = _strategy_only_metrics(primary_execution)
    health_alpha_metrics = _strategy_only_metrics(health_execution)

    report = REPORT_TEMPLATE.render(
        as_of=end.date(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_end=end.date(),
        execution_alpha_cagr=_pct(execution_alpha_metrics["cagr"]),
        health_alpha_cagr=_pct(health_alpha_metrics["cagr"]),
        nested_cagr=_pct(nested_metrics["cagr"]),
        nested_obs=int(nested_metrics["observations"]),
        tradable_latest=tradable_latest,
        candidate_count=len(parameter_cache.returns.columns),
        allocator_class="pass" if allocator_pass else "fail",
        allocator_status="PASS" if allocator_pass else "FAIL",
        health_class="pass" if health_pass else "fail",
        health_status="PASS" if health_pass else "FAIL",
        nested_class="pass" if nested_pass else "fail",
        nested_status="PASS" if nested_pass else "FAIL",
        target_return=_pct(first_audit["target_return"]),
        liquidity_status=(
            f"{liquidity_status['download_status']}; "
            f"{liquidity_status['liquidity_source']}; "
            f"{liquidity_status['spread_source']}"
        ),
        weight_table=_html_table(weight_display),
        comparison_table=_html_table(comparison_table),
        health_table=_html_table(health_display),
        execution_table=_html_table(execution_display),
        nested_table=_html_table(nested_display),
        bootstrap_table=_html_table(bootstrap_display),
        snooping_table=_html_table(snooping_display),
    )
    (output / "production_gates_report.html").write_text(report, encoding="utf-8")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {"start": str(start.date()), "end": str(end.date())},
        "decision": "research / paper trade only",
        "gates": {
            "allocator": "pass" if allocator_pass else "fail",
            "rolling_pair_health": "pass" if health_pass else "fail",
            "nested_walk_forward": "pass" if nested_pass else "fail",
            "execution_data": "conditional; quote, ADV, borrow, and locate inputs not verified",
        },
        "allocator": {
            "specification": "monthly minimum covariance with target return and linear exposure constraints",
            "first_weights": schedule.iloc[0].to_dict(),
            "latest_weights": schedule.iloc[-1].to_dict(),
            "first_diagnostics": first_allocation.diagnostics,
            "first_constraint_inputs": {
                "target_return": first_audit["target_return"],
                "gross_bounds": first_audit["gross_bounds"],
                "net_bounds": first_audit["net_bounds"],
                "factor_bounds": first_audit["factor_bounds"],
            },
        },
        "health_config": asdict(health_config),
        "latest_tradable_pairs": tradable_latest,
        "execution_input_status": liquidity_status,
        "execution_primary_metrics": primary_execution.metrics,
        "execution_primary_strategy_only_metrics": execution_alpha_metrics,
        "health_execution_metrics": health_execution.metrics,
        "health_strategy_only_metrics": health_alpha_metrics,
        "nested_spec": asdict(nested.spec),
        "nested_performance": nested.performance.iloc[0].to_dict(),
        "important_limitations": [
            "Historical Yahoo bid/ask quotes and securities-lending locates were unavailable.",
            "The nested outer windows are already-viewed history and cannot restore a virgin lockbox.",
            "The nested audit fixes formation hedge relations but reselects pairs and refits weights in each train fold.",
            "Forecast target return is a model constraint, not a realized-return guarantee.",
        ],
    }

    def json_default(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    (processed / "production_gates_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(output / "production_gates_report.html")


if __name__ == "__main__":
    main()
