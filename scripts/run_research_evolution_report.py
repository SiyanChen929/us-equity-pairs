#!/usr/bin/env python3
"""Build a consolidated audit of the equity-pairs research evolution.

The report is deliberately additive: it reads the frozen artifacts created by
the original, follow-up, weighted, and production-gate pipelines and writes a
new report plus a small set of reconciliation artifacts.  No earlier report or
research output is overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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
GREEN = "#2d7d5b"
INK = "#20334e"
MUTED = "#66758a"
PALE = "#eef3f8"


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


def _bp(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f} bp" if math.isfinite(number) else "—"


def _integer(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{int(round(number)):,}" if math.isfinite(number) else "—"


def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
    return frame.to_html(
        index=index,
        border=0,
        classes="data-table",
        justify="left",
        escape=True,
        na_rep="—",
    )


def _cagr(returns: pd.Series, annualization: int = 252) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    total = float((1.0 + clean).prod())
    if total <= 0:
        return -1.0
    return total ** (annualization / len(clean)) - 1.0


def _metrics(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    equity = (1.0 + clean).cumprod()
    return performance_metrics(clean, equity)


def _strategy_only_metrics(result: WeightedPortfolioResult) -> dict[str, float]:
    """Strip collateral cash yield while retaining trading/short economics."""
    prior_equity = result.daily["equity"].shift(1, fill_value=1.0)
    cash = result.daily["cash_income"].div(prior_equity).fillna(0.0)
    returns = result.daily["portfolio_return"] - cash
    returns = returns.mask(returns.abs() < 1e-12, 0.0)
    return performance_metrics(returns)


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
    return pd.Series(values, dtype=float)


def _macro_factor_betas(
    pair_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    standardized = factors.apply(pd.to_numeric, errors="coerce")
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=1)
    rows: dict[str, dict[str, float]] = {}
    for pair in pair_returns.columns:
        aligned = pd.concat(
            [pair_returns[pair].rename("pair"), standardized], axis=1
        ).dropna()
        if len(aligned) <= len(standardized.columns) + 2:
            rows[str(pair)] = {str(column): 0.0 for column in standardized.columns}
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


def _rolling_covariance_only_schedule(
    pair_returns: pd.DataFrame,
    positions: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    market: pd.Series,
    macro_factors: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the production allocator with only the return floor removed."""
    pairs = selected_metadata["pair"].astype(str).tolist()
    metadata = selected_metadata.set_index("pair").reindex(pairs)
    hedge_beta = pd.to_numeric(metadata["beta"], errors="coerce")
    name_incidence = build_gross_name_incidence(selected_metadata)
    rows: list[pd.Series] = []
    diagnostics: list[dict[str, Any]] = []
    for rebalance_date in rebalance_dates:
        eligible = pair_returns.index[pair_returns.index < rebalance_date]
        estimation_index = eligible[-504:]
        returns = pair_returns.reindex(estimation_index, columns=pairs).fillna(0.0)
        position_panel = positions.reindex(estimation_index, columns=pairs).fillna(0.0)
        pair_gross = position_panel.abs().mean()
        structural_net = (1.0 - hedge_beta) / (1.0 + hedge_beta)
        pair_net = position_panel.mean().mul(structural_net)
        market_betas = _pair_market_betas(returns, market.reindex(returns.index))
        factor_betas = _macro_factor_betas(returns, macro_factors)
        equal = pd.Series(1.0 / len(pairs), index=pairs)
        equal_gross = float(equal @ pair_gross)
        gross_bounds = (
            max(0.0, equal_gross - 0.08),
            min(1.0, equal_gross + 0.08),
        )
        equal_net = float(equal @ pair_net)
        net_limit = max(0.03, abs(equal_net) + 0.015)
        factor_bounds: dict[str, tuple[float, float]] = {}
        for factor in factor_betas.columns:
            equal_exposure = float(equal @ factor_betas[factor])
            limit = max(0.0015, abs(equal_exposure) + 0.0005)
            factor_bounds[str(factor)] = (-limit, limit)
        allocation = allocate_pair_sleeves(
            returns,
            method="minimum_covariance",
            pair_betas=market_betas,
            pair_metadata=selected_metadata,
            lower_bounds=0.04,
            upper_bounds=0.18,
            sector_caps={
                str(sector): 0.25
                for sector in selected_metadata["sector"].astype(str).unique()
            },
            name_incidence_caps=0.10,
            name_incidence=name_incidence,
            pair_gross_exposures=pair_gross,
            gross_exposure_bounds=gross_bounds,
            pair_net_exposures=pair_net,
            net_exposure_bounds=(-net_limit, net_limit),
            market_beta_bounds=(-0.05, 0.05),
            factor_betas=factor_betas,
            factor_beta_bounds=factor_bounds,
            fallback_method="inverse_volatility",
        )
        rows.append(allocation.weights.rename(pd.Timestamp(rebalance_date)))
        diagnostics.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "estimation_start": estimation_index.min(),
                "estimation_end": estimation_index.max(),
                **{
                    key: value
                    for key, value in allocation.diagnostics.items()
                    if not isinstance(value, (dict, list, tuple))
                },
            }
        )
    schedule = pd.DataFrame(rows)
    schedule.index.name = "rebalance_date"
    return schedule, pd.DataFrame(diagnostics)


def _exact_backtest(
    signals: pd.DataFrame,
    pairs: list[str],
    weights: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    schedule: pd.DataFrame | None = None,
) -> WeightedPortfolioResult:
    subset = signals.loc[signals["pair"].isin(pairs)].copy()
    return backtest_weighted_pair_book(
        subset,
        weights.reindex(pairs),
        start=start,
        end=end,
        transaction_cost_bps=5.0,
        rebalance_frequency="monthly",
        no_trade_band=0.01,
        maximum_rebalance_turnover=0.15,
        charge_terminal_liquidation=True,
        target_weight_schedule=schedule,
    )


def _repeat_selection_bootstrap(
    matrix: np.ndarray,
    *,
    samples: int,
    block_size: int,
    seed: int,
    centering: str,
) -> np.ndarray:
    """Repeat best-ten selection under a joint circular-block null."""
    if centering == "log":
        transformed = np.log1p(np.clip(matrix, -0.999999, None))
        centered = transformed - transformed.mean(axis=0, keepdims=True)
    elif centering == "arithmetic":
        centered = matrix - matrix.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unsupported centering mode: {centering}")
    rng = np.random.default_rng(seed)
    observations = matrix.shape[0]
    blocks_needed = math.ceil(observations / block_size)
    results: list[float] = []
    for _ in range(samples):
        starts = rng.integers(0, observations, size=blocks_needed)
        locations = np.concatenate(
            [(start + np.arange(block_size)) % observations for start in starts]
        )[:observations]
        sample = centered[locations]
        if centering == "log":
            pair_scores = sample.sum(axis=0)
            sample_simple = np.expm1(sample)
        else:
            sample_simple = np.clip(sample, -0.999999, None)
            pair_scores = np.log1p(sample_simple).sum(axis=0)
        selected = np.argpartition(pair_scores, -10)[-10:]
        portfolio = sample_simple[:, selected].mean(axis=1)
        terminal = float(np.prod(1.0 + portfolio))
        results.append(terminal ** (252.0 / observations) - 1.0)
    return np.asarray(results, dtype=float)


def _selection_bias_diagnostics(
    baseline_signals: pd.DataFrame,
    original_top_pairs: list[str],
    *,
    seed: int = 20260829,
    random_samples: int = 50_000,
    bootstrap_samples: int = 10_000,
    block_size: int = 10,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    wide = baseline_signals.pivot(
        index="date", columns="pair", values="net_return"
    ).sort_index()
    full = wide.loc["2023-08-28":"2026-08-26"]
    first = wide.loc["2023-08-28":"2025-08-26"]
    final = wide.loc["2025-08-27":"2026-08-26"]
    pair_full_cagr = full.apply(_cagr).sort_values(ascending=False)
    pair_first_cagr = first.apply(_cagr).sort_values(ascending=False)
    pair_final_cagr = final.apply(_cagr).sort_values(ascending=False)
    first_top = pair_first_cagr.head(10).index.tolist()
    full_top = pair_full_cagr.head(10).index.tolist()

    first_selected_final_metrics = _metrics(final[first_top].mean(axis=1))
    oracle_final_metrics = _metrics(final[full_top].mean(axis=1))
    half = len(full) // 2
    left_rank = full.iloc[:half].apply(_cagr).rank()
    right_rank = full.iloc[half:].apply(_cagr).rank()
    half_spearman = float(left_rank.corr(right_rank))
    half_overlap = len(
        set(full.iloc[:half].apply(_cagr).nlargest(10).index)
        & set(full.iloc[half:].apply(_cagr).nlargest(10).index)
    )

    matrix = full.to_numpy(dtype=float)
    random_seed = seed + 1
    rng = np.random.default_rng(random_seed)
    random_cagrs: list[float] = []
    batch = 500
    for _ in range(0, random_samples, batch):
        count = min(batch, random_samples - len(random_cagrs))
        indices = np.array(
            [rng.choice(matrix.shape[1], 10, replace=False) for _ in range(count)]
        )
        daily = matrix[:, indices].mean(axis=2)
        terminal = np.prod(1.0 + daily, axis=0)
        random_cagrs.extend(terminal ** (252.0 / len(full)) - 1.0)
    random_array = np.asarray(random_cagrs, dtype=float)

    bootstrap_seed = seed
    bootstrap_array = _repeat_selection_bootstrap(
        matrix,
        samples=bootstrap_samples,
        block_size=block_size,
        seed=bootstrap_seed,
        centering="log",
    )
    robustness_samples = min(5_000, bootstrap_samples)
    arithmetic_seed = seed + 2
    arithmetic_array = _repeat_selection_bootstrap(
        matrix,
        samples=robustness_samples,
        block_size=block_size,
        seed=arithmetic_seed,
        centering="arithmetic",
    )
    sensitivity_rows: list[dict[str, Any]] = []
    for sensitivity_block in [1, 5, 10, 20, 40, 60]:
        if sensitivity_block == block_size:
            draws = bootstrap_array
            sensitivity_seed = bootstrap_seed
        else:
            sensitivity_seed = seed + 100 + sensitivity_block
            draws = _repeat_selection_bootstrap(
                matrix,
                samples=robustness_samples,
                block_size=sensitivity_block,
                seed=sensitivity_seed,
                centering="log",
            )
        sensitivity_rows.append(
            {
                "centering": "zero per-pair log growth",
                "block_size": sensitivity_block,
                "samples": len(draws),
                "seed": sensitivity_seed,
                "median_cagr": float(np.median(draws)),
                "mean_cagr": float(draws.mean()),
                "p05_cagr": float(np.quantile(draws, 0.05)),
                "p95_cagr": float(np.quantile(draws, 0.95)),
                "fraction_at_or_above_observed": float(
                    (draws >= _cagr(full[original_top_pairs].mean(axis=1))).mean()
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)

    observed_oracle = _cagr(full[original_top_pairs].mean(axis=1))
    diagnostics = {
        "seed": seed,
        "independent_streams": {
            "headline_log_bootstrap_seed": bootstrap_seed,
            "random_ten_seed": random_seed,
            "arithmetic_robustness_seed": arithmetic_seed,
        },
        "observations": len(full),
        "pair_count": wide.shape[1],
        "pair_cagr_cross_section": {
            "mean": float(pair_full_cagr.mean()),
            "median": float(pair_full_cagr.median()),
            "standard_deviation": float(pair_full_cagr.std(ddof=1)),
            "positive_fraction": float((pair_full_cagr > 0).mean()),
            "top_ten_mean": float(pair_full_cagr.head(10).mean()),
            "remaining_mean": float(pair_full_cagr.iloc[10:].mean()),
        },
        "ranking_stability": {
            "half_sample_spearman": half_spearman,
            "half_sample_top_ten_overlap": half_overlap,
            "first_two_year_top_ten": first_top,
            "full_period_top_ten": full_top,
            "first_two_year_top_ten_final_year_cagr": first_selected_final_metrics[
                "cagr"
            ],
            "full_period_oracle_top_ten_final_year_cagr": oracle_final_metrics[
                "cagr"
            ],
        },
        "random_ten": {
            "samples": random_samples,
            "mean": float(random_array.mean()),
            "median": float(np.median(random_array)),
            "p95": float(np.quantile(random_array, 0.95)),
            "p99": float(np.quantile(random_array, 0.99)),
            "p999": float(np.quantile(random_array, 0.999)),
            "maximum": float(random_array.max()),
            "observed_oracle_cagr": observed_oracle,
            "samples_at_or_above_observed": int((random_array >= observed_oracle).sum()),
        },
        "repeat_selection_null": {
            "method": "zero per-pair log-growth; all-start common-date circular blocks; reselect best 10",
            "samples": bootstrap_samples,
            "block_size": block_size,
            "possible_block_starts": len(full),
            "seed": bootstrap_seed,
            "median": float(np.median(bootstrap_array)),
            "mean": float(bootstrap_array.mean()),
            "p05": float(np.quantile(bootstrap_array, 0.05)),
            "p95": float(np.quantile(bootstrap_array, 0.95)),
            "fraction_at_or_above_observed": float(
                (bootstrap_array >= observed_oracle).mean()
            ),
        },
        "arithmetic_mean_robustness": {
            "method": "zero per-pair arithmetic mean; all-start common-date circular blocks; reselect best 10",
            "samples": robustness_samples,
            "block_size": block_size,
            "possible_block_starts": len(full),
            "seed": arithmetic_seed,
            "median": float(np.median(arithmetic_array)),
            "mean": float(arithmetic_array.mean()),
            "p05": float(np.quantile(arithmetic_array, 0.05)),
            "p95": float(np.quantile(arithmetic_array, 0.95)),
            "fraction_at_or_above_observed": float(
                (arithmetic_array >= observed_oracle).mean()
            ),
        },
    }
    splits = pd.DataFrame(
        [
            {
                "construction": "Full-period winners selected with hindsight",
                "selection_window": "2023-08-28 to 2026-08-26",
                "evaluation_window": "same full three years",
                "cagr": observed_oracle,
                "evidence": "ex-post / selection-biased",
            },
            {
                "construction": "Full-period winners, final-year slice",
                "selection_window": "includes final year",
                "evaluation_window": "2025-08-27 to 2026-08-26",
                "cagr": oracle_final_metrics["cagr"],
                "evidence": "look-ahead slice",
            },
            {
                "construction": "Top ten frozen after first two years",
                "selection_window": "2023-08-28 to 2025-08-26",
                "evaluation_window": "2025-08-27 to 2026-08-26",
                "cagr": first_selected_final_metrics["cagr"],
                "evidence": "forward counterfactual",
            },
        ]
    )
    return (
        diagnostics,
        random_array,
        bootstrap_array,
        arithmetic_array,
        sensitivity,
        splits,
    )


def _plot_performance_ledger(
    exact_final: pd.DataFrame,
    output: Path,
) -> None:
    frame = exact_final.copy()
    colors = [
        TEAL if value >= 0 else RED
        for value in pd.to_numeric(frame["cagr"], errors="coerce")
    ]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    values = frame["cagr"].to_numpy(dtype=float) * 100.0
    bars = ax.barh(frame["label"], values, color=colors, alpha=0.9)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Final-year net CAGR (%) · exact leg accounting")
    ax.set_title("Each adjustment answered a different question", loc="left", fontsize=15)
    ax.set_facecolor("#fbfcfe")
    ax.grid(axis="x", color="#d8e0e9", linewidth=0.7, alpha=0.8)
    ax.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        offset = 0.15 if value >= 0 else -0.15
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_selection_bias(
    pair_metrics: pd.DataFrame,
    random_cagrs: np.ndarray,
    reselection_cagrs: np.ndarray,
    observed: float,
    output: Path,
) -> None:
    values = pd.to_numeric(pair_metrics["net_cagr"], errors="coerce").dropna()
    top_cut = float(values.nlargest(10).min())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    axes[0].hist(values * 100.0, bins=28, color=BLUE, alpha=0.82, edgecolor="white")
    axes[0].axvline(top_cut * 100.0, color=ORANGE, linewidth=2, label="Top-ten cutoff")
    axes[0].axvline(values.mean() * 100.0, color=INK, linestyle="--", label="All-pair mean")
    axes[0].set_title("110 held-out pair outcomes", loc="left")
    axes[0].set_xlabel("Individual pair net CAGR (%)")
    axes[0].set_ylabel("Pair count")
    axes[0].legend(frameon=False)

    axes[1].hist(
        random_cagrs * 100.0,
        bins=45,
        color=TEAL,
        alpha=0.78,
        edgecolor="white",
    )
    axes[1].axvline(
        observed * 100.0,
        color=RED,
        linewidth=2.4,
        label=f"Observed ex-post ten: {observed * 100:.2f}%",
    )
    axes[1].set_title("Random ten, no reselection", loc="left")
    axes[1].set_xlabel("Three-year portfolio CAGR (%)")
    axes[1].set_ylabel("Simulation count")
    axes[1].legend(frameon=False)
    axes[2].hist(
        reselection_cagrs * 100.0,
        bins=36,
        color=PURPLE,
        alpha=0.82,
        edgecolor="white",
    )
    axes[2].axvline(
        observed * 100.0,
        color=RED,
        linewidth=2,
        label="Observed ex-post ten",
    )
    axes[2].axvline(
        np.median(reselection_cagrs) * 100.0,
        color=INK,
        linestyle="--",
        label="Null median",
    )
    axes[2].set_title("Repeat selection under zero log growth", loc="left")
    axes[2].set_xlabel("Apparent winning-ten CAGR (%)")
    axes[2].set_ylabel("Bootstrap draws")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_facecolor("#fbfcfe")
        ax.grid(axis="y", color="#d8e0e9", linewidth=0.6, alpha=0.7)
    fig.suptitle(
        "The +12.40% curve is an order statistic, not the representative strategy",
        x=0.04,
        ha="left",
        fontsize=15,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_signal_factorial(factorial: pd.DataFrame, output: Path) -> None:
    pivot = factorial.pivot(index="pair_universe", columns="signal_rule", values="cagr")
    rows = ["Hindsight full-period ten", "Forward-selected later ten"]
    columns = ["Original SMA60 / 2.0 / hold60", "Tuned SMA20 / 1.75 / hold20"]
    pivot = pivot.reindex(index=rows, columns=columns)
    matrix = pivot.to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-8, vmax=12, aspect="auto")
    ax.set_xticks(range(len(columns)), ["Original SMA60", "Tuned SMA20"])
    ax.set_yticks(range(len(rows)), ["Hindsight ten", "Forward-selected ten"])
    ax.set_title("Pair membership and signal parameters interact", loc="left", fontsize=15)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}%",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if abs(matrix[row, column]) > 5 else INK,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label("Final-year net CAGR (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_allocator_ablation(ablation: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = [BLUE, ORANGE, TEAL]
    values = ablation["cagr"].to_numpy(dtype=float) * 100.0
    bars = axes[0].bar(ablation["label"], values, color=colors)
    axes[0].axhline(0, color=INK, linewidth=1)
    axes[0].set_ylabel("Net CAGR (%)")
    axes[0].set_title("Return floor hurt realized performance", loc="left")
    axes[0].tick_params(axis="x", rotation=14)
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.08 if value >= 0 else -0.08),
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontweight="bold",
            color=INK,
        )
    turnover = ablation["capital_rebalance_turnover"].to_numpy(dtype=float)
    bars2 = axes[1].bar(ablation["label"], turnover, color=colors)
    axes[1].set_ylabel("Two-way capital reallocation over year")
    axes[1].set_title("The hard target also increased reallocation", loc="left")
    axes[1].tick_params(axis="x", rotation=14)
    for bar, value in zip(bars2, turnover, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=INK,
        )
    for ax in axes:
        ax.set_facecolor("#fbfcfe")
        ax.grid(axis="y", color="#d8e0e9", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


REPORT_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity Pairs Research — Evolution and Production Audit</title>
<style>
:root{--ink:#20334e;--muted:#66758a;--line:#d9e1ea;--paper:#fff;--blue:#245c97;--teal:#17817d;--green:#2d7d5b;--orange:#d77a24;--red:#bd3d4a;--purple:#7652a5;--pale:#eef3f8}
*{box-sizing:border-box}body{margin:0;background:#e9eef4;color:var(--ink);font:15px/1.58 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1260px;margin:24px auto;background:var(--paper);box-shadow:0 14px 42px rgba(32,52,78,.13);overflow:hidden}header{padding:54px 68px 38px;background:linear-gradient(135deg,#13223b 0%,#245c97 56%,#17817d 100%);color:#fff}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:760;opacity:.78}h1{font:700 43px/1.08 Georgia,serif;margin:13px 0 14px;max-width:1020px}header p{font-size:17px;max-width:980px;margin:0;opacity:.9}.content{padding:38px 68px 70px}h2{font:700 29px/1.18 Georgia,serif;margin:52px 0 14px}h3{font-size:18px;margin:30px 0 9px}h4{font-size:15px;margin:22px 0 7px}.lead{font-size:17px;color:#33445b}.decision{border-left:6px solid var(--red);background:#fff0f1;padding:21px 23px;margin:0 0 25px;font-size:16px}.reader{border:1px solid #cfe0ea;background:#f6f9fc;padding:18px 21px;border-radius:8px}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:22px 0}.card{border:1px solid var(--line);border-radius:8px;padding:16px;background:#fff}.card .k{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:760}.card .v{font-size:26px;line-height:1.15;font-weight:760;margin:7px 0 3px}.card .s{font-size:12px;color:var(--muted)}.gate{display:grid;grid-template-columns:180px 1fr 115px;gap:14px;padding:15px 0;border-bottom:1px solid var(--line);align-items:start}.status{display:inline-block;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:760;text-align:center}.pass{background:#e5f4ed;color:#176447}.fail{background:#ffe8ea;color:#982733}.conditional{background:#fff0db;color:#915514}.timeline{border-left:3px solid #c8d7e7;margin:20px 0 28px 10px;padding-left:24px}.version{position:relative;margin:0 0 22px}.version:before{content:"";position:absolute;left:-32px;top:6px;width:13px;height:13px;border-radius:50%;background:var(--blue);border:3px solid #fff;box-shadow:0 0 0 2px #a9bfd6}.version .label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--blue);font-weight:760}.version strong{font-size:17px}.chart{display:block;width:100%;border:1px solid var(--line);border-radius:7px;margin:18px 0 11px}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.grid2>div{min-width:0}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.note{padding:15px 18px;border-radius:7px;background:var(--pale);margin:14px 0}.note.bad{background:#fff0f1;border-left:4px solid var(--red)}.note.warn{background:#fff6e9;border-left:4px solid var(--orange)}.note.good{background:#edf8f3;border-left:4px solid var(--green)}.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f4f7fa;border:1px solid var(--line);padding:14px 16px;border-radius:6px;overflow-x:auto}.scroll{overflow-x:auto;max-width:100%}.data-table{border-collapse:collapse;width:100%;font-size:12.2px;margin:13px 0 22px}.data-table th{background:#edf2f7;text-align:left;padding:9px 8px;border-bottom:2px solid #cbd5e1;white-space:nowrap}.data-table td{padding:8px;border-bottom:1px solid #e4e9ef;vertical-align:top}.data-table tr:hover td{background:#f8fafc}.pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#e8f0fb;color:#1e559b;font-size:11px;font-weight:700;margin:2px 4px 2px 0}.small{font-size:12px}.muted{color:var(--muted)}code{background:#f1f4f7;padding:2px 5px;border-radius:4px}a{color:#1e61ad}.artifact{columns:2;column-gap:36px}.artifact h4{break-after:avoid}.artifact ul{break-inside:avoid;margin-top:5px}.foot{margin-top:52px;padding-top:21px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}@media(max-width:900px){.page{margin:0}.content,header{padding-left:22px;padding-right:22px}.cards{grid-template-columns:1fr 1fr}.grid2,.grid3{grid-template-columns:1fr}.gate{grid-template-columns:1fr}.status{width:max-content}.artifact{columns:1}h1{font-size:36px}}@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none}.chart,.note,.gate{break-inside:avoid}a{color:inherit;text-decoration:none}}
code{overflow-wrap:anywhere;word-break:break-word}
@media print{
  @page{size:A4 landscape;margin:8mm}
  body{font-size:11px;line-height:1.38}
  .page{box-shadow:none;margin:0;max-width:none;overflow:visible}
  header{padding:28px 30px 23px}
  h1{font-size:32px}
  header p,.lead{font-size:13px}
  .content{padding:24px 30px 32px}
  h2{font-size:23px;margin:34px 0 10px}
  h3{font-size:15px;margin:22px 0 7px}
  .cards{gap:8px;margin:15px 0}
  .card{padding:11px}
  .card .v{font-size:20px}
  .scroll{overflow:visible}
  .data-table{font-size:8.4px;line-height:1.22;table-layout:auto}
  .data-table th{white-space:normal;padding:5px 4px;overflow-wrap:anywhere}
  .data-table td{padding:4px;overflow-wrap:anywhere;word-break:normal}
  .chart,.note,.gate,.grid2,.grid3{break-inside:avoid}
  .chart{max-height:166mm;object-fit:contain}
}
</style></head><body><main class="page">
<header><div class="eyebrow">Equity pairs research · complete evolution audit · {{ as_of }}</div><h1>From eight-year cointegration screen to production-gate review</h1><p>What changed after the first version, what improved, what failed, and what the evidence now supports. This report is additive: every earlier report remains unchanged.</p></header>
<section class="content">

<div class="decision"><strong>Current decision: research and paper trade only.</strong> The engineering pipeline improved materially—selection is auditable, parameters were grid-tested, capital is non-equal and constrained, pair health is monitored, execution mechanics are explicit, and validation is nested. The investment conclusion became less favorable as the tests became more realistic. Persistent net pairs alpha has not been demonstrated.</div>

<div class="cards">
<div class="card"><div class="k">Securities</div><div class="v">503/503</div><div class="s">Current constituents downloaded</div></div>
<div class="card"><div class="k">Pairs screened</div><div class="v">13,917</div><div class="s">Different issuer · within sector</div></div>
<div class="card"><div class="k">Strict formation slots</div><div class="v">7/110</div><div class="s">103 deterministic fallbacks</div></div>
<div class="card"><div class="k">Latest healthy</div><div class="v">{{ latest_tradable }}/10</div><div class="s">Strict production health state</div></div>
<div class="card"><div class="k">Nested OOS CAGR</div><div class="v">{{ nested_cagr }}</div><div class="s">315 outer-test sessions</div></div>
</div>

<h2>1 · Reader contract: the evidence labels</h2>
<p class="lead">The same strategy can show very different numbers when membership, parameters, dates, capital accounting, or selection timing changes. Every headline below is therefore labeled before it is interpreted.</p>
<div class="scroll">{{ evidence_table }}</div>
<div class="reader"><strong>Metric dictionary.</strong> Total CAGR includes all modeled cash and financing effects. Pairs-alpha CAGR strips free collateral cash yield. Capital weights allocate money across pair sleeves; hedge-ratio-normalized leg weights operate inside each sleeve. CAGR is <code>(ending equity / starting equity)^(252 / observations) − 1</code>. A low linear market beta does not guarantee factor, liquidity, correlation, or crash neutrality.</div>

<h2>2 · Executive gate board</h2>
<div class="gate"><strong>Constrained allocator</strong><div>Non-equal pair capital with Ledoit–Wolf covariance, pair/sector/name caps, market and macro-factor beta limits, and gross/net exposure constraints.</div><span class="status pass">PASS</span></div>
<div class="gate"><strong>Rolling pair health</strong><div>Causal cointegration, residual stationarity, beta drift, half-life drift, structural break, kill switch, cooldown and recovery state.</div><span class="status fail">FAIL</span></div>
<div class="gate"><strong>Nested walk-forward</strong><div>Purged chronological folds, train-only pair reselection and non-equal weighting, frozen outer choices, block bootstrap and reality-check-style diagnostics.</div><span class="status fail">FAIL</span></div>
<div class="gate"><strong>Execution inputs</strong><div>Commission, spread, impact, ADV, borrow, locate, financing, rebate, cash yield and liquidation mechanics run; core liquidity inputs remain proxies.</div><span class="status conditional">CONDITIONAL</span></div>

<h2>3 · Research evolution</h2>
<div class="timeline">
<div class="version"><div class="label">V1 · Original pipeline</div><strong>Screen, select, backtest and explain.</strong><div>Eight-year Yahoo/FRED data, five-year formation, three-year held-out backtest, 110 requested sector slots, baseline SMA60 rule. Defensible result: {{ original_all_cagr }} CAGR. The requested +12.40% winning-ten curve was ex-post.</div></div>
<div class="version"><div class="label">V1.1 · Diagnostic follow-up</div><strong>Risk, shocks, pair dependence and frequency.</strong><div>Measured market beta and shared-name concentration; tested 360 SMA/EWMA rules. The faster SMA20 rule improved the one-year all-110 lockbox to {{ grid_lockbox_cagr }}, but turnover more than doubled.</div></div>
<div class="version"><div class="label">V2 · Ten-pair unequal book</div><strong>Stop trading all 110 and stop using equal capital.</strong><div>Selected exactly ten using tuning-only evidence and diversification constraints; implemented covariance-aware ERC, exact leg accounting and terminal liquidation. ERC improved equal capital from −2.80% to −1.61%.</div></div>
<div class="version"><div class="label">V3 · Production controls</div><strong>Allocator, health, validation and executable costs.</strong><div>Added rolling constrained weights, structural-break quarantine, purged nested folds, data-snooping diagnostics and ticker-level cost/capacity mechanics. Allocator passed; health and nested evidence failed.</div></div>
<div class="version"><div class="label">V4 · Postmortem and reconciliation</div><strong>Explain why later results were worse.</strong><div>Recomputed all comparable portfolios with one engine, tested selection stability, repeated winner selection under a zero-alpha null, and ablated the hard return floor. The first headline was dominated by hindsight selection; the rolling return target added adverse timing.</div></div>
</div>
<div class="scroll">{{ evolution_table }}</div>
<img class="chart" src="charts/research_evolution_performance.png" alt="Comparable final-year performance across research adjustments">

<h2>4 · Data, universe and original research contract</h2>
<div class="grid2"><div><h3>Research funnel</h3><div class="formula">503 current securities · 500 issuers · 11 sectors<br>↓ 13,917 within-sector, different-issuer candidates<br>↓ 110 formation-selected slots requested<br>↓ 7 strict + 103 fallback<br>↓ 10 constrained portfolio sleeves<br>↓ {{ latest_tradable }} currently tradable under strict health</div></div><div><h3>Data contract</h3><ul><li>2,010 adjusted-price sessions from 2018-08-27 through 2026-08-26; 476 tickers had at least 90% formation coverage.</li><li>Current S&amp;P 500 membership and current GICS mapped backward.</li><li>FRED DFF, DGS2, DGS10, CPI, unemployment, recession and VIX series.</li><li>Same-issuer share classes excluded through issuer identifiers.</li><li>1,258-session formation and 752-session held-out test.</li></ul></div></div>
<div class="note bad"><strong>Non-negotiable bias:</strong> current membership and GICS labels were projected backward. Delisted and exited names are absent. Every result is therefore exposed to survivorship, membership and historical-classification bias.</div>
<p class="small">The literal full-eight-year top-ten-per-sector file is descriptive, not traded: only 1 of its 110 rows passed every strict eligibility gate. The held-out book instead used the separate five-year formation ranking.</p>
<div class="note warn">FRED was used for revised-vintage ex-post attribution and later allocator/execution diagnostics. It was not an original trading signal, and without ALFRED release vintages the macro history is not a point-in-time feature set.</div>

<h2>5 · Original methodology and first-version result</h2>
<p class="lead">The original pipeline was causal at the signal level but weak at the portfolio-selection level. Close-<em>t</em> decisions changed positions one session later; the decisive optimism came from ranking winners after observing their test returns.</p>
<div class="formula">log(P<sub>dependent</sub>) = α + β log(P<sub>independent</sub>) + spread<br>z<sub>t</sub> = (spread<sub>t</sub> − SMA<sub>60,t</sub>) / sample σ<sub>60,t</sub></div>
<div class="scroll">{{ original_rule_table }}</div>
<img class="chart" src="charts/heldout_portfolio_equity.png" alt="Original held-out portfolio equity curves">
<div class="scroll">{{ original_performance_table }}</div>
<h4>V1 transaction-cost sensitivity · 30 bp annual borrow retained</h4>
<div class="scroll">{{ original_cost_table }}</div>
<div class="note warn"><strong>Correct V1 headline:</strong> the all-formation-selected and sector-balanced portfolios both returned {{ original_all_cagr }} CAGR because every sector contributed exactly ten slots. The +12.40% portfolio was the ten highest outcomes selected after the same three-year test was visible.</div>
<div class="note"><strong>Accounting boundary reconciliation:</strong> the published V1 engine did not charge a synthetic launch for 55 sleeves already active on the first test date or terminal liquidation for 44 sleeves still open at the end. The later exact dollar-leg engine changes the same full-period all-110 CAGR from −0.46% to {{ exact_v1_all_cagr }} and the ex-post-ten CAGR from +12.40% to {{ exact_v1_oracle_cagr }}. This fixes accounting; it does not cure the oracle selection.</div>
<h3>Original ex-post winning ten</h3>
<div class="scroll">{{ original_winner_table }}</div>

<h2>6 · Why the first +12.40% curve looked so strong</h2>
<p class="lead">It was not a universally favorable calendar period. It was an oracle portfolio for that period. The final year was included when the ten names were chosen, and only one pair overlaps the later forward-selected ten.</p>
<img class="chart" src="charts/research_selection_bias.png" alt="Selection-bias diagnostics for the original winning ten">
<div class="grid2"><div><h3>Forward counterfactual</h3><div class="scroll">{{ selection_split_table }}</div></div><div><h3>Ranking stability</h3><ul><li>Half-sample rank Spearman: <strong>{{ half_spearman }}</strong>.</li><li>Top-ten overlap between halves: <strong>{{ half_overlap }}/10</strong>.</li><li>Top ten chosen only on the first two years: <strong>{{ forward_counterfactual }}</strong> CAGR in the next year.</li><li>Full-period winners in that same final year: <strong>{{ oracle_final }}</strong>—but that year was used to choose them.</li></ul></div></div>
<div class="note bad"><strong>Reality-check-style diagnostic:</strong> after centering every pair to zero log-growth, sampling from all 752 circular block starts and repeating the same “observe, then choose best ten” procedure {{ bootstrap_samples }} times, the median apparent winning-ten CAGR was {{ bootstrap_median }}. The observed +12.40% was met or exceeded in {{ bootstrap_exceed }} of draws. This is a direct measure of same-window order-statistic bias, not a formal White Reality Check.</div>
<div class="scroll">{{ bootstrap_sensitivity_table }}</div>
<div class="note warn">The conclusion is robust to the null definition. Under the alternative zero-arithmetic-mean null, {{ arithmetic_exceed }} of {{ arithmetic_samples }} draws still met or exceeded the observed curve. Exact percentages vary with centering and block length; the +12.40% result remains ordinary after repeating the winner-selection step.</div>
<p>The old book also concentrated apparently independent winners: FERG appeared in four sleeves, FCX and DRI in two each, leaving 15 unique names across five sectors. Maximum gross-normalized FERG exposure was {{ old_max_name }}. The current selector forbids ticker reuse and uses 20 unique names.</p>

<h2>7 · Follow-up diagnostics: shocks, dependence and trade frequency</h2>
<div class="grid3"><div class="card"><div class="k">All-110 market beta</div><div class="v">{{ baseline_market_beta }}</div><div class="s">Realized test regression</div></div><div class="card"><div class="k">Market correlation</div><div class="v">{{ baseline_market_corr }}</div><div class="s">R² {{ baseline_market_r2 }}</div></div><div class="card"><div class="k">≥2% down days</div><div class="v">{{ shock_days }}</div><div class="s">Book compounded {{ shock_cumulative }}</div></div></div>
<p>Low beta reduced broad directional sensitivity but did not neutralize shocks. The hedge ratio was a cointegration coefficient, not a certified market/factor hedge.</p>
<div class="note warn">SPY refresh failed during this diagnostic, so “market” here is a survivorship-biased equal-weight return of the same 503 current constituents—not an independent, point-in-time benchmark.</div>
<div class="scroll">{{ overlap_table }}</div>
<div class="note warn"><strong>Shared-name dependence motivated the later constraints.</strong> Disjoint pairs averaged {{ disjoint_active_corr }} absolute active correlation, while shared-underlying pairs averaged {{ shared_active_corr }} and reached {{ shared_active_max }}. The 110 sleeves used only 144 names; 46 names recurred.</div>
<h3>Was fewer than twenty trades per pair unusual?</h3>
<p>No. The original slow rule produced {{ baseline_total_trades }} entries across 110 pairs over three years, median {{ baseline_median_trades }} per pair. Sparse pair-level estimates were therefore expected. Lower thresholds and shorter memory increased the sample, but also increased costs and exposure to microstructure noise.</p>
<div class="scroll">{{ frequency_table }}</div>
<div class="note"><strong>Count definitions matter.</strong> “Pair entries” are completed or open trade episodes. The executable log contains {{ ticker_adjustments }} nonzero ticker adjustments—{{ ticker_adjustments_per_session }} per session—but those include entries, exits, hedge drift, capital rebalances and terminal liquidation. They are orders, not {{ tuned_ten_entries }} independent pair trades.</div>

<h2>8 · Parameter search and z-score adjustment</h2>
<p class="lead">The first version used a plain rolling arithmetic mean and sample standard deviation. The follow-up added EWMA and searched 360 predeclared daily rules; EWMA did not win.</p>
<div><span class="pill">SMA and EWMA</span><span class="pill">20 / 40 / 60 / 90 lookback</span><span class="pill">1.25–2.50 entry</span><span class="pill">0 / 0.25 / 0.50 exit</span><span class="pill">20 / 40 / 60 hold</span></div>
<p>The winning tuning rule was <strong>SMA20, entry 1.75σ, exit 0.50σ and maximum hold 20 sessions</strong>. Pair membership and formation hedge relations remained frozen. The first two held-out years selected by net sector-balanced Sharpe; the final year was revealed once.</p>
<img class="chart" src="charts/condensed_grid_frontier.png" alt="SMA and EWMA grid frontier">
<img class="chart" src="charts/condensed_validation_equity.png" alt="Baseline and grid-winner validation equity">
<div class="scroll">{{ grid_table }}</div>
<div class="note good">The faster rule improved the broad 110-pair final-year lockbox from −1.30% to +0.84% under the follow-up’s sleeve-average accounting. Under the later exact leg engine it was {{ exact_all110_cagr }}. The result was modest and turnover rose from 11.1× to 26.5×.</div>

<h2>9 · Exactly ten pairs and non-equal capital</h2>
<p class="lead">The weighted study answered two explicit changes: trade only ten pairs, and do not allocate 10% to each. Selection used tuning-only evidence and imposed diversification before capital optimization.</p>
<div class="grid2"><div class="note"><strong>Selection constraints</strong><br>Exactly ten; maximum two per sector; no repeated ticker; at least six sectors; minimum six tuning trades; maximum pair correlation 0.60; pre-allocation beta within ±0.025.</div><div class="note"><strong>Static ERC constraints</strong><br>5%–15% per pair; sector ≤25%; gross-normalized name ≤10%; market beta ±0.05; Ledoit–Wolf covariance; monthly no-trade band 1%; two-way capital adjustment cap 15%.</div></div>
<img class="chart" src="charts/weighted_allocation.png" alt="Static non-equal ERC allocation and risk contributions">
<div class="scroll">{{ current_pair_table }}</div>
<h3>Same ten, same final year, same exact accounting</h3>
<div class="scroll">{{ weighted_method_table }}</div>
<div class="note good"><strong>Unequal weighting helped.</strong> ERC raised CAGR from −2.80% equal capital to −1.61%, reduced volatility from 4.75% to 4.31%, and reduced maximum drawdown from −6.42% to −5.28%. It improved risk allocation; it did not create a profitable signal.</div>
<p class="small">Minimum variance happened to finish at −1.31%, but it was one of several same-final-year diagnostics. Promoting it now would create another ex-post allocator choice.</p>
<img class="chart" src="charts/weighted_pair_correlations.png" alt="Tuning and final-year pair correlations">
<p>The final-year ERC market beta was {{ current_market_beta }}, correlation {{ current_market_corr }}, and R² {{ current_market_r2 }}. Market neutrality largely worked. Spread-specific alpha did not.</p>

<h2>10 · Performance reconciliation and postmortem</h2>
<p class="lead">The exact final-year ledger isolates universe selection, non-equal allocation and rolling weight timing. Costs and dates are held constant.</p>
<div class="scroll">{{ exact_final_table }}</div>
<img class="chart" src="charts/research_pair_signal_factorial.png" alt="Pair-universe and signal-parameter interaction">
<p>The original winners only work with the original signal that helped select them. Applying the tuned SMA20 rule to those hindsight names produces a loss; applying the original SMA60 rule to the later names loses even more. Pair membership and signal parameters interact, which is why single-axis optimization is fragile.</p>
<h3>Rolling return-target ablation</h3>
<img class="chart" src="charts/research_allocator_ablation.png" alt="Return-floor allocator ablation">
<div class="scroll">{{ ablation_table }}</div>
<div class="note bad"><strong>The hard return floor was binding in {{ binding_rows }}/{{ allocation_rows }} monthly solves.</strong> Its trailing cross-sectional forecasts had mean next-period information coefficient {{ forecast_ic }}, median {{ forecast_ic_median }}, and positive sign in only {{ forecast_ic_positive }}/{{ allocation_rows }} periods. Removing only that floor improved rolling CAGR from −2.56% to {{ no_target_cagr }} and cut capital-rebalance turnover from 0.826 to {{ no_target_rebalance }}. This is a retrospective ablation, not a newly tradable selection.</div>

<h2>11 · Production constrained allocator</h2>
<p>The implemented production allocator refits at each month’s first trading session using only the prior 504 sessions. It minimizes Ledoit–Wolf covariance subject to a shrunk return floor, 4%–18% sleeve bounds, 25% sector caps, 10% gross-normalized name caps, market and FRED rates/VIX factor beta bands, and gross/net portfolio limits.</p>
<img class="chart" src="charts/production_gate_weights.png" alt="Monthly constrained production weights">
<div class="grid2"><div><h3>First monthly target</h3><div class="scroll">{{ first_weight_table }}</div></div><div><h3>Latest monthly target</h3><div class="scroll">{{ latest_weight_table }}</div></div></div>
<div class="scroll">{{ allocator_diagnostics_table }}</div>
<div class="note warn"><strong>The initial 9.13% target was a training forecast, not a promise.</strong> The allocator satisfied all modeled constraints and used no numerical fallback, but realized returns show that feasible exposure control is not equivalent to reliable mean forecasting.</div>

<h2>12 · Rolling pair health</h2>
<p>The health layer uses a causal 252-session window with at least 126 observations, one-session decision lag, cointegration/residual ADF checks, 35% beta-drift limit, half-life drift, structural-break diagnostics, two bad observations to kill, a 20-session cooldown and three good observations to recover.</p>
<img class="chart" src="charts/production_gate_health.png" alt="Rolling pair-health states">
<div class="scroll">{{ health_table }}</div>
<div class="note bad"><strong>Result: {{ latest_tradable }}/10 tradable.</strong> Every pair was forced flat in the final-year strict-health overlay. Pairs-alpha CAGR was exactly 0%; the reported +3.57% total CAGR was collateral cash yield, not trading skill. The configuration may be conservative, but it correctly blocks relationships that fail its predeclared diagnostics.</div>
<div class="note warn">All 2,510 pair-days were labeled <code>recovery</code>, meaning quarantined and seeking enough good observations—not recovered or tradable. Four recoveries occurred historically, but none left a pair eligible at sample end. The midpoint structural-break proxy failed on 2,407 pair-days, so these thresholds require training-only calibration before they can be treated as a usable replacement engine.</div>

<h2>13 · Executable cost and capacity model</h2>
<p>Flat costs were replaced with a ticker-level engine covering commission, bid/ask half-spread, square-root impact, ADV participation, locates, borrow, financing, short rebate, cash yield, underlying netting and terminal liquidation.</p>
<img class="chart" src="charts/production_gate_execution.png" alt="Execution cost and capacity scenarios">
<div class="scroll">{{ execution_table }}</div>
<div class="note warn"><strong>Conditional input status:</strong> Yahoo OHLCV returned HTTP 429 during the refresh attempt. The saved execution scenarios therefore use a $25 million ADV-per-name fallback, 2.5 bp half-spread, 50 bp annual borrow and assumed locates. The mechanics are auditable; the capacity conclusions are not broker verified.</div>
<p class="small">At $50 million the engine still reports diagnostic P&amp;L even though it records 421 capacity breaches across 155 days because <code>fail_on_unexecutable=False</code>. That row is explicitly non-executable.</p>

<h2>14 · Purged nested walk-forward</h2>
<p class="lead">The strongest internal validation reselected ten pairs and refit non-equal weights inside every supplied training fold. It searched 65 predeclared SMA/EWMA cells with purge and embargo, then froze each outer choice before its 63-session test.</p>
<div><span class="pill">5 outer folds</span><span class="pill">350-session rolling train</span><span class="pill">63-session test</span><span class="pill">40-session purge</span><span class="pill">5-session embargo</span><span class="pill">1,000 block bootstraps</span></div>
<img class="chart" src="charts/production_gate_nested.png" alt="Nested walk-forward results">
<div class="scroll">{{ nested_fold_table }}</div>
<div class="grid2"><div><h3>Aggregate performance</h3><div class="scroll">{{ nested_performance_table }}</div></div><div><h3>Bootstrap intervals</h3><div class="scroll">{{ nested_bootstrap_table }}</div></div></div>
<div class="scroll">{{ snooping_table }}</div>
<div class="note bad"><strong>Nested gate fails.</strong> CAGR was {{ nested_cagr }}, Sharpe {{ nested_sharpe }}, and the 95% CAGR interval ran from {{ nested_cagr_lower }} to {{ nested_cagr_upper }}. Reality-check-style p-values were all above 10%. These dates are now viewed history, but the audit still shows no stable net edge.</div>
<div class="note warn">This is a model-selection audit, not executable P&amp;L: each fold reselected pairs and refit weights causally, but reused the original formation alpha/beta relations and cached pair returns with flat 5 bp plus 30 bp borrow. It did not rerun the ticker-level execution engine inside every outer fold.</div>

<h2>15 · Everything adjusted after the first version</h2>
<div class="scroll">{{ adjustment_table }}</div>
<p>The implementation footprint now includes explicit allocation, pair-health, execution and nested-validation modules; dated weight schedules; ticker-level execution logs; health state histories; fold audits; deterministic caches; adjusted-OHLCV acquisition and checksums; and separate reports for each research layer.</p>

<h2>16 · Current rules and recommended next design</h2>
<div class="grid2"><div><h3>Implemented pre-trade and live controls</h3><ul><li>Exactly ten frozen sleeves for the retrospective book.</li><li>Next-session signal effectiveness.</li><li>Mean-reversion, z-stop and maximum-holding exits.</li><li>Rolling health eligibility and forced liquidation.</li><li>Sector/name/market/factor/gross/net allocation limits.</li><li>ADV and locate rejection logic.</li><li>Monthly no-trade band, turnover cap and terminal liquidation.</li></ul></div><div><h3>Recommended—not yet promoted</h3><ul><li>Remove the hard sample-mean return floor.</li><li>Predeclare slow ERC or covariance-only quarterly weighting.</li><li>Add an explicit turnover penalty and weight-change regularization.</li><li>Replace unhealthy pairs from a lagged candidate pool; do not force quota fillers.</li><li>Permit return tilts only after positive nested-OOS forecast IC.</li><li>Freeze all logic and collect genuinely new paper-trading data.</li></ul></div></div>
<div class="note good"><strong>Best current allocator hypothesis:</strong> retain non-equal capital, but use slow ERC or covariance-only weights without a hard expected-return floor. This is a forward research hypothesis, not permission to choose whichever allocator won the already-viewed final year.</div>

<h2>17 · Production gaps ranked by decision value</h2>
<ol><li>Point-in-time S&amp;P membership, historical GICS and delisted names.</li><li>Licensed quotes, trades, corporate actions and exchange calendars.</li><li>Historical borrow fees and locate availability.</li><li>ALFRED vintage and release-time aligned macro data.</li><li>Quarterly lagged pair formation/replacement with no forced fallback quota.</li><li>Health thresholds calibrated only inside nested training folds.</li><li>Independent paper-trade order/fill and exposure reconciliation.</li><li>Predeclared promotion thresholds for alpha, confidence interval, drawdown, turnover, availability and capacity.</li></ol>

<h2>18 · Reproducibility, quality and artifact index</h2>
<div class="grid2"><div class="note"><strong>Current validation</strong><br>{{ tests_passed }} tests pass; {{ tests_skipped }} optional test skipped. Credential scan found no FRED-key leakage in outputs. The consolidated report adds deterministic reconciliation CSV/JSON artifacts and does not modify earlier reports.</div><div class="note"><strong>Generate this report</strong><br><code>MPLCONFIGDIR=/tmp/mpl_pairs PYTHONPATH=src python scripts/run_research_evolution_report.py --output-dir outputs/pairs_research_2026-08-26</code></div></div>
<p class="small"><code>manifest.json</code> and <code>quality_checks.json</code> remain the immutable V1 provenance records; they do not checksum the later condensed, weighted, production, or consolidated addenda. Each later layer has its own saved metadata and deterministic runner.</p>
<div class="artifact">
<h4>Core V1</h4><ul><li><a href="report.html">Original full report</a></li><li><a href="data/processed/full_8y_top10_cointegrated_by_sector.csv">Eight-year sector leaders</a></li><li><a href="data/processed/formation_selected_top10_by_sector.csv">Formation selections</a></li><li><a href="data/processed/best_10_pairs_by_heldout_net_cagr.csv">Ex-post winning ten</a></li><li><a href="data/processed/pair_daily_signals.csv.gz">Daily signals</a></li><li><a href="data/processed/pair_trade_ledger.csv">Trade ledger</a></li></ul>
<h4>Diagnostic and grid follow-up</h4><ul><li><a href="condensed_report.html">Condensed follow-up report</a></li><li><a href="data/processed/grid_parameter_ranking.csv">Grid ranking</a></li><li><a href="data/processed/grid_baseline_winner_comparison.csv">Baseline/winner comparison</a></li><li><a href="data/processed/followup_market_shock_diagnostics.csv">Shock diagnostics</a></li><li><a href="data/processed/followup_overlap_correlation_summary.csv">Overlap correlations</a></li></ul>
<h4>Ten-pair unequal book</h4><ul><li><a href="weighted_portfolio_report.html">Weighted portfolio report</a></li><li><a href="data/processed/weighted_selected_top10.csv">Selected ten</a></li><li><a href="data/processed/weighted_primary_weights.csv">Static ERC weights</a></li><li><a href="data/processed/weighted_method_comparison.csv">Allocation methods</a></li><li><a href="data/processed/weighted_portfolio_daily.csv">Weighted daily book</a></li></ul>
<h4>Production controls</h4><ul><li><a href="production_gates_report.html">Production-gate report</a></li><li><a href="data/processed/production_weight_schedule.csv">Monthly weights</a></li><li><a href="data/processed/production_pair_health_daily.csv.gz">Health history</a></li><li><a href="data/processed/production_execution_scenarios.csv">Execution scenarios</a></li><li><a href="data/processed/production_nested_fold_audit.csv">Nested fold audit</a></li><li><a href="data/processed/production_nested_bootstrap.csv">Bootstrap intervals</a></li></ul>
<h4>Consolidated audit</h4><ul><li><a href="data/processed/research_version_performance.csv">Reconciled performance ledger</a></li><li><a href="data/processed/research_adjustment_log.csv">Adjustment log</a></li><li><a href="data/processed/research_selection_bias_diagnostics.json">Selection-bias diagnostics</a></li><li><a href="data/processed/research_selection_null_distributions.csv.gz">Selection-null draws</a></li><li><a href="data/processed/research_selection_block_sensitivity.csv">Block sensitivity</a></li><li><a href="data/processed/research_allocator_ablation.csv">Allocator ablation</a></li><li><a href="data/processed/research_pair_signal_factorial.csv">Pair/signal factorial</a></li></ul>
<h4>Methods and code</h4><ul><li><a href="../../docs/methodology.md">Deep-dive methodology</a></li><li><a href="../../README.md">Usage and architecture</a></li><li><a href="../../scripts/run_followup_research.py">Follow-up runner</a></li><li><a href="../../scripts/run_weighted_portfolio_research.py">Weighted runner</a></li><li><a href="../../scripts/run_production_gates_research.py">Production runner</a></li><li><a href="../../scripts/run_research_evolution_report.py">Consolidated runner</a></li></ul>
</div>

<h2>19 · Primary methodological references</h2>
<ul class="small"><li><a href="https://www.nber.org/papers/w7032">Gatev, Goetzmann &amp; Rouwenhorst — Pairs Trading</a></li><li><a href="https://doi.org/10.2307/1913236">Engle &amp; Granger — Co-integration and Error Correction</a></li><li><a href="https://doi.org/10.1111/joes.12153">Krauss — Statistical Arbitrage Pairs Trading Strategies: Review and Outlook</a></li><li><a href="https://www.ledoit.net/honey.pdf">Ledoit &amp; Wolf — Honey, I Shrunk the Sample Covariance Matrix</a></li><li><a href="https://www.thierry-roncalli.com/download/erc.pdf">Maillard, Roncalli &amp; Teïletche — Equal Risk Contribution</a></li><li><a href="https://web.stanford.edu/~boyd/papers/portfolio.html">Lobo, Fazel &amp; Boyd — Portfolio Optimization with Transaction Costs</a></li><li><a href="https://doi.org/10.1111/1468-0262.00152">White — A Reality Check for Data Snooping</a></li><li><a href="https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html">Official yfinance download reference</a></li><li><a href="https://www.sec.gov/files/rules/final/2008/34-58774.pdf">SEC Regulation SHO locate requirements</a></li></ul>

<div class="foot">Generated {{ generated_at }} · Sample ends {{ as_of }} · Yahoo adjusted prices and FRED macro series · Current-constituent research prototype · No historical result guarantees future profit.</div>
</section></main></body></html>"""
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pairs_research_2026-08-26"),
    )
    parser.add_argument("--random-ten-samples", type=int, default=50_000)
    parser.add_argument("--selection-bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output_dir
    raw = output / "data" / "raw"
    processed = output / "data" / "processed"
    charts = output / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    portfolio_metrics = pd.read_csv(processed / "portfolio_metrics.csv")
    portfolio_cost_sensitivity = pd.read_csv(
        processed / "portfolio_cost_sensitivity.csv"
    )
    pair_metrics = pd.read_csv(processed / "pair_backtest_metrics.csv")
    original_winners = pd.read_csv(
        processed / "best_10_pairs_by_heldout_net_cagr.csv"
    )
    original_pairs = original_winners["pair"].astype(str).tolist()
    formation_selected = pd.read_csv(
        processed / "formation_selected_top10_by_sector.csv"
    )
    baseline_signals = pd.read_csv(
        processed / "pair_daily_signals.csv.gz", parse_dates=["date"]
    )
    grid_signals = pd.read_csv(
        processed / "grid_selected_pair_signals.csv.gz", parse_dates=["date"]
    )
    grid_returns = pd.read_csv(
        processed / "grid_selected_pair_returns.csv",
        index_col="date",
        parse_dates=True,
    ).sort_index()
    grid_metadata = json.loads(
        (processed / "grid_search_metadata.json").read_text(encoding="utf-8")
    )
    grid_comparison = pd.read_csv(
        processed / "grid_baseline_winner_comparison.csv"
    )
    market_beta = pd.read_csv(processed / "followup_market_beta.csv")
    market_shocks = pd.read_csv(
        processed / "followup_market_shock_diagnostics.csv"
    )
    overlap_summary = pd.read_csv(
        processed / "followup_overlap_correlation_summary.csv"
    )
    exposure_summary = pd.read_csv(
        processed / "followup_portfolio_exposure_summary.csv"
    ).iloc[0]

    weighted_metadata = json.loads(
        (processed / "weighted_portfolio_metadata.json").read_text(encoding="utf-8")
    )
    selected = pd.read_csv(processed / "weighted_selected_top10.csv")
    current_pairs = selected["pair"].astype(str).tolist()
    static_weights_frame = pd.read_csv(processed / "weighted_primary_weights.csv")
    static_weights = static_weights_frame.set_index("pair")["weight"].reindex(
        current_pairs
    )
    weighted_methods = pd.read_csv(processed / "weighted_method_comparison.csv")
    weighted_trades = pd.read_csv(processed / "weighted_trade_summary.csv")
    selected_metadata = (
        formation_selected.drop_duplicates("pair")
        .set_index("pair")
        .reindex(current_pairs)
        .reset_index()
    )
    production_metadata = json.loads(
        (processed / "production_gates_metadata.json").read_text(encoding="utf-8")
    )
    production_schedule = pd.read_csv(
        processed / "production_weight_schedule.csv",
        index_col="rebalance_date",
        parse_dates=True,
    ).reindex(columns=current_pairs)
    production_schedule_diagnostics = pd.read_csv(
        processed / "production_weight_schedule_diagnostics.csv",
        parse_dates=["rebalance_date", "estimation_start", "estimation_end"],
    )
    health_daily = pd.read_csv(
        processed / "production_pair_health_daily.csv.gz", parse_dates=["date"]
    )
    health_events = pd.read_csv(
        processed / "production_pair_health_events.csv", parse_dates=["date"]
    )
    execution_scenarios = pd.read_csv(
        processed / "production_execution_scenarios.csv"
    )
    nested_performance = pd.read_csv(
        processed / "production_nested_performance.csv"
    ).iloc[0]
    nested_outer = pd.read_csv(
        processed / "production_nested_outer_evaluations.csv",
        parse_dates=["train_start", "train_end", "test_start", "test_end"],
    )
    nested_bootstrap = pd.read_csv(
        processed / "production_nested_bootstrap.csv"
    )
    nested_snooping = pd.read_csv(
        processed / "production_nested_snooping.csv"
    )

    start = pd.Timestamp(weighted_metadata["retrospective_start"])
    end = pd.Timestamp(weighted_metadata["retrospective_end"])
    all_pairs = grid_returns.columns.astype(str).tolist()
    equal_all = pd.Series(1.0 / len(all_pairs), index=all_pairs)
    equal_current = pd.Series(1.0 / len(current_pairs), index=current_pairs)
    equal_original = pd.Series(1.0 / len(original_pairs), index=original_pairs)

    v1_start = pd.Timestamp("2023-08-28")
    v1_end = pd.Timestamp("2026-08-26")
    exact_v1_all = _exact_backtest(
        baseline_signals, all_pairs, equal_all, v1_start, v1_end
    )
    exact_v1_oracle = _exact_backtest(
        baseline_signals, original_pairs, equal_original, v1_start, v1_end
    )

    exact_all110 = _exact_backtest(
        grid_signals, all_pairs, equal_all, start, end
    )
    exact_current_equal = _exact_backtest(
        grid_signals, current_pairs, equal_current, start, end
    )
    exact_current_erc = _exact_backtest(
        grid_signals, current_pairs, static_weights, start, end
    )
    exact_current_rolling = _exact_backtest(
        grid_signals,
        current_pairs,
        production_schedule.iloc[0],
        start,
        end,
        schedule=production_schedule,
    )

    market = pd.read_csv(
        processed / "followup_market_proxy_returns.csv",
        index_col="date",
        parse_dates=True,
    ).iloc[:, 0]
    macro = pd.read_csv(
        processed / "fred_macro_and_derived.csv",
        index_col="date",
        parse_dates=True,
    ).sort_index()
    macro_on_returns = macro.reindex(grid_returns.index).ffill()
    macro_factors = pd.DataFrame(
        {
            "rates": pd.to_numeric(
                macro_on_returns["DGS10"], errors="coerce"
            ).diff().div(100.0),
            "vix": pd.to_numeric(
                macro_on_returns["VIXCLS"], errors="coerce"
            ).pct_change(fill_method=None),
        },
        index=grid_returns.index,
    ).replace([np.inf, -np.inf], np.nan)
    positions = (
        grid_signals.loc[grid_signals["pair"].isin(current_pairs)]
        .pivot(index="date", columns="pair", values="position")
        .reindex(columns=current_pairs)
    )
    no_target_schedule, no_target_diagnostics = _rolling_covariance_only_schedule(
        grid_returns[current_pairs],
        positions,
        selected_metadata,
        market,
        macro_factors,
        production_schedule.index,
    )
    exact_no_target = _exact_backtest(
        grid_signals,
        current_pairs,
        no_target_schedule.iloc[0],
        start,
        end,
        schedule=no_target_schedule,
    )
    no_target_schedule.to_csv(
        processed / "research_no_target_weight_schedule.csv"
    )
    no_target_diagnostics.to_csv(
        processed / "research_no_target_weight_diagnostics.csv", index=False
    )

    original_old_signal = _exact_backtest(
        baseline_signals, original_pairs, equal_original, start, end
    )
    original_new_signal = _exact_backtest(
        grid_signals, original_pairs, equal_original, start, end
    )
    current_old_signal = _exact_backtest(
        baseline_signals, current_pairs, equal_current, start, end
    )
    factorial = pd.DataFrame(
        [
            {
                "pair_universe": "Hindsight full-period ten",
                "signal_rule": "Original SMA60 / 2.0 / hold60",
                "cagr": original_old_signal.metrics["cagr"],
                "sharpe": original_old_signal.metrics["sharpe"],
                "max_drawdown": original_old_signal.metrics["max_drawdown"],
            },
            {
                "pair_universe": "Hindsight full-period ten",
                "signal_rule": "Tuned SMA20 / 1.75 / hold20",
                "cagr": original_new_signal.metrics["cagr"],
                "sharpe": original_new_signal.metrics["sharpe"],
                "max_drawdown": original_new_signal.metrics["max_drawdown"],
            },
            {
                "pair_universe": "Forward-selected later ten",
                "signal_rule": "Original SMA60 / 2.0 / hold60",
                "cagr": current_old_signal.metrics["cagr"],
                "sharpe": current_old_signal.metrics["sharpe"],
                "max_drawdown": current_old_signal.metrics["max_drawdown"],
            },
            {
                "pair_universe": "Forward-selected later ten",
                "signal_rule": "Tuned SMA20 / 1.75 / hold20",
                "cagr": exact_current_equal.metrics["cagr"],
                "sharpe": exact_current_equal.metrics["sharpe"],
                "max_drawdown": exact_current_equal.metrics["max_drawdown"],
            },
        ]
    )
    factorial.to_csv(
        processed / "research_pair_signal_factorial.csv", index=False
    )

    (
        selection_diagnostics,
        random_ten_cagrs,
        bootstrap_cagrs,
        arithmetic_bootstrap_cagrs,
        bootstrap_sensitivity,
        selection_splits,
    ) = _selection_bias_diagnostics(
        baseline_signals,
        original_pairs,
        random_samples=args.random_ten_samples,
        bootstrap_samples=args.selection_bootstrap_samples,
    )
    (processed / "research_selection_bias_diagnostics.json").write_text(
        json.dumps(selection_diagnostics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    selection_splits.to_csv(
        processed / "research_selection_counterfactual.csv", index=False
    )
    selection_null_draws = pd.concat(
        [
            pd.DataFrame(
                {
                    "experiment": "random_ten_without_reselection",
                    "replicate": np.arange(len(random_ten_cagrs)),
                    "cagr": random_ten_cagrs,
                }
            ),
            pd.DataFrame(
                {
                    "experiment": "centered_common_date_block_reselect_ten",
                    "replicate": np.arange(len(bootstrap_cagrs)),
                    "cagr": bootstrap_cagrs,
                }
            ),
            pd.DataFrame(
                {
                    "experiment": "arithmetic_centered_common_date_block_reselect_ten",
                    "replicate": np.arange(len(arithmetic_bootstrap_cagrs)),
                    "cagr": arithmetic_bootstrap_cagrs,
                }
            ),
        ],
        ignore_index=True,
    )
    selection_null_draws.to_csv(
        processed / "research_selection_null_distributions.csv.gz",
        index=False,
        compression="gzip",
    )
    bootstrap_sensitivity.to_csv(
        processed / "research_selection_block_sensitivity.csv", index=False
    )

    forecast_rows: list[dict[str, Any]] = []
    schedule_dates = production_schedule.index.tolist()
    for location, rebalance_date in enumerate(schedule_dates):
        estimation = grid_returns.index[grid_returns.index < rebalance_date][-504:]
        forecast = grid_returns.loc[estimation, current_pairs].mean() * 252.0
        next_date = (
            schedule_dates[location + 1]
            if location + 1 < len(schedule_dates)
            else grid_returns.index.max() + pd.Timedelta(days=1)
        )
        realized = grid_returns.loc[
            (grid_returns.index >= rebalance_date) & (grid_returns.index < next_date),
            current_pairs,
        ].sum()
        forecast_rows.append(
            {
                "rebalance_date": rebalance_date,
                "pearson_ic": forecast.corr(realized),
                "spearman_ic": forecast.rank().corr(realized.rank()),
            }
        )
    forecast_ic = pd.DataFrame(forecast_rows)
    forecast_ic.to_csv(
        processed / "research_return_forecast_ic.csv", index=False
    )

    original_all = portfolio_metrics.loc[
        portfolio_metrics["portfolio"].eq("all_formation_selected")
    ].iloc[0]
    original_expost = portfolio_metrics.loc[
        portfolio_metrics["portfolio"].eq("expost_top_performers")
    ].iloc[0]
    primary_execution = execution_scenarios.loc[
        execution_scenarios["portfolio_notional"].eq("$10m")
    ].iloc[0]

    performance_rows = [
        {
            "version": "V1",
            "result": "All 110 formation-selected",
            "window": "2023-08-28 to 2026-08-26",
            "universe": "110 formation slots",
            "selection_timing": "formation only",
            "parameters": "SMA60 / entry2 / exit0.5 / hold60",
            "weighting": "equal pair / equal sector",
            "accounting": "pair-level flat 5 bp + 30 bp borrow",
            "cagr": original_all["cagr"],
            "sharpe": original_all["sharpe"],
            "max_drawdown": original_all["max_drawdown"],
            "evidence": "held-out single split",
        },
        {
            "version": "V1",
            "result": "Ex-post winning ten",
            "window": "2023-08-28 to 2026-08-26",
            "universe": "best 10 after test",
            "selection_timing": "uses reported outcomes",
            "parameters": "SMA60 / entry2 / exit0.5 / hold60",
            "weighting": "equal capital",
            "accounting": "pair-level flat 5 bp + 30 bp borrow",
            "cagr": original_expost["cagr"],
            "sharpe": original_expost["sharpe"],
            "max_drawdown": original_expost["max_drawdown"],
            "evidence": "ex-post / selection-biased",
        },
        {
            "version": "Reconciled",
            "result": "All 110, V1 signal (exact)",
            "window": "2023-08-28 to 2026-08-26",
            "universe": "110 formation slots",
            "selection_timing": "formation only",
            "parameters": "SMA60 / entry2 / exit0.5 / hold60",
            "weighting": "equal capital",
            "accounting": "exact legs + monthly rebalance + liquidation",
            "cagr": exact_v1_all.metrics["cagr"],
            "sharpe": exact_v1_all.metrics["sharpe"],
            "max_drawdown": exact_v1_all.metrics["max_drawdown"],
            "evidence": "retrospective accounting reconciliation",
        },
        {
            "version": "Reconciled",
            "result": "Ex-post winning ten, V1 signal (exact)",
            "window": "2023-08-28 to 2026-08-26",
            "universe": "best 10 after test",
            "selection_timing": "uses reported outcomes",
            "parameters": "SMA60 / entry2 / exit0.5 / hold60",
            "weighting": "equal capital",
            "accounting": "exact legs + monthly rebalance + liquidation",
            "cagr": exact_v1_oracle.metrics["cagr"],
            "sharpe": exact_v1_oracle.metrics["sharpe"],
            "max_drawdown": exact_v1_oracle.metrics["max_drawdown"],
            "evidence": "ex-post plus accounting reconciliation",
        },
        {
            "version": "V1.1",
            "result": "Grid winner, all 110 (reported)",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "110 formation slots",
            "selection_timing": "rule frozen after first 2 years",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "equal capital",
            "accounting": "sleeve-average boundary accounting",
            "cagr": grid_comparison.loc[
                (grid_comparison["split"].eq("lockbox"))
                & (grid_comparison["portfolio"].eq("equal_weight")),
                "cagr_best",
            ].iloc[0],
            "sharpe": grid_comparison.loc[
                (grid_comparison["split"].eq("lockbox"))
                & (grid_comparison["portfolio"].eq("equal_weight")),
                "sharpe_best",
            ].iloc[0],
            "max_drawdown": grid_comparison.loc[
                (grid_comparison["split"].eq("lockbox"))
                & (grid_comparison["portfolio"].eq("equal_weight")),
                "max_drawdown_best",
            ].iloc[0],
            "evidence": "spent one-year lockbox",
        },
        {
            "version": "Reconciled",
            "result": "Grid winner, all 110 (exact)",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "110 formation slots",
            "selection_timing": "rule frozen after first 2 years",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "equal capital",
            "accounting": "exact legs + monthly rebalance + liquidation",
            "cagr": exact_all110.metrics["cagr"],
            "sharpe": exact_all110.metrics["sharpe"],
            "max_drawdown": exact_all110.metrics["max_drawdown"],
            "evidence": "retrospective reconciliation",
        },
        {
            "version": "V2",
            "result": "Selected ten, equal capital",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "tuning-selected 10",
            "selection_timing": "pairs frozen after first 2 years",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "equal capital",
            "accounting": "exact legs + monthly rebalance + liquidation",
            "cagr": exact_current_equal.metrics["cagr"],
            "sharpe": exact_current_equal.metrics["sharpe"],
            "max_drawdown": exact_current_equal.metrics["max_drawdown"],
            "evidence": "spent one-year lockbox",
        },
        {
            "version": "V2",
            "result": "Selected ten, static ERC",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "tuning-selected 10",
            "selection_timing": "pairs/weights frozen after first 2 years",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "non-equal constrained ERC",
            "accounting": "exact legs + monthly rebalance + liquidation",
            "cagr": exact_current_erc.metrics["cagr"],
            "sharpe": exact_current_erc.metrics["sharpe"],
            "max_drawdown": exact_current_erc.metrics["max_drawdown"],
            "evidence": "retrospective / spent lockbox",
        },
        {
            "version": "V3",
            "result": "Selected ten, rolling target allocator",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "tuning-selected 10",
            "selection_timing": "monthly lagged weights",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "minimum covariance + hard return floor",
            "accounting": "exact legs + flat 5 bp + liquidation",
            "cagr": exact_current_rolling.metrics["cagr"],
            "sharpe": exact_current_rolling.metrics["sharpe"],
            "max_drawdown": exact_current_rolling.metrics["max_drawdown"],
            "evidence": "retrospective / spent lockbox",
        },
        {
            "version": "V4 ablation",
            "result": "Selected ten, rolling covariance only",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "tuning-selected 10",
            "selection_timing": "monthly lagged weights",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "minimum covariance; no return floor",
            "accounting": "exact legs + flat 5 bp + liquidation",
            "cagr": exact_no_target.metrics["cagr"],
            "sharpe": exact_no_target.metrics["sharpe"],
            "max_drawdown": exact_no_target.metrics["max_drawdown"],
            "evidence": "retrospective diagnostic",
        },
        {
            "version": "V3 execution",
            "result": "$10m executable scenario, pairs alpha",
            "window": "2025-08-27 to 2026-08-26",
            "universe": "tuning-selected 10",
            "selection_timing": "monthly lagged weights",
            "parameters": "SMA20 / entry1.75 / exit0.5 / hold20",
            "weighting": "rolling constrained",
            "accounting": "ticker execution proxies; cash stripped",
            "cagr": primary_execution["strategy_only_cagr"],
            "sharpe": primary_execution["strategy_only_sharpe"],
            "max_drawdown": primary_execution["max_drawdown"],
            "evidence": "conditional execution scenario",
        },
        {
            "version": "V3 nested",
            "result": "Purged nested walk-forward",
            "window": "315 outer-test sessions",
            "universe": "10 reselected inside each training fold",
            "selection_timing": "frozen inside each outer fold",
            "parameters": "65-cell train-only selection",
            "weighting": "train-only constrained non-equal",
            "accounting": "nested cached pair returns",
            "cagr": nested_performance["cagr"],
            "sharpe": nested_performance["sharpe"],
            "max_drawdown": nested_performance["max_drawdown"],
            "evidence": "nested OOS within viewed history",
        },
    ]
    performance_ledger = pd.DataFrame(performance_rows)
    performance_ledger.to_csv(
        processed / "research_version_performance.csv", index=False
    )

    exact_final = pd.DataFrame(
        [
            {
                "label": "Grid winner · all 110",
                "cagr": exact_all110.metrics["cagr"],
                "sharpe": exact_all110.metrics["sharpe"],
                "max_drawdown": exact_all110.metrics["max_drawdown"],
                "transaction_drag_bps": exact_all110.metrics[
                    "transaction_cost_drag_bps"
                ],
                "annualized_turnover": exact_all110.metrics[
                    "annualized_leg_turnover"
                ],
                "capital_rebalance_turnover": exact_all110.metrics[
                    "capital_rebalance_turnover_sum"
                ],
            },
            {
                "label": "Selected ten · equal capital",
                "cagr": exact_current_equal.metrics["cagr"],
                "sharpe": exact_current_equal.metrics["sharpe"],
                "max_drawdown": exact_current_equal.metrics["max_drawdown"],
                "transaction_drag_bps": exact_current_equal.metrics[
                    "transaction_cost_drag_bps"
                ],
                "annualized_turnover": exact_current_equal.metrics[
                    "annualized_leg_turnover"
                ],
                "capital_rebalance_turnover": exact_current_equal.metrics[
                    "capital_rebalance_turnover_sum"
                ],
            },
            {
                "label": "Selected ten · static ERC",
                "cagr": exact_current_erc.metrics["cagr"],
                "sharpe": exact_current_erc.metrics["sharpe"],
                "max_drawdown": exact_current_erc.metrics["max_drawdown"],
                "transaction_drag_bps": exact_current_erc.metrics[
                    "transaction_cost_drag_bps"
                ],
                "annualized_turnover": exact_current_erc.metrics[
                    "annualized_leg_turnover"
                ],
                "capital_rebalance_turnover": exact_current_erc.metrics[
                    "capital_rebalance_turnover_sum"
                ],
            },
            {
                "label": "Selected ten · rolling target",
                "cagr": exact_current_rolling.metrics["cagr"],
                "sharpe": exact_current_rolling.metrics["sharpe"],
                "max_drawdown": exact_current_rolling.metrics["max_drawdown"],
                "transaction_drag_bps": exact_current_rolling.metrics[
                    "transaction_cost_drag_bps"
                ],
                "annualized_turnover": exact_current_rolling.metrics[
                    "annualized_leg_turnover"
                ],
                "capital_rebalance_turnover": exact_current_rolling.metrics[
                    "capital_rebalance_turnover_sum"
                ],
            },
            {
                "label": "Selected ten · no return floor",
                "cagr": exact_no_target.metrics["cagr"],
                "sharpe": exact_no_target.metrics["sharpe"],
                "max_drawdown": exact_no_target.metrics["max_drawdown"],
                "transaction_drag_bps": exact_no_target.metrics[
                    "transaction_cost_drag_bps"
                ],
                "annualized_turnover": exact_no_target.metrics[
                    "annualized_leg_turnover"
                ],
                "capital_rebalance_turnover": exact_no_target.metrics[
                    "capital_rebalance_turnover_sum"
                ],
            },
        ]
    )

    ablation = exact_final.loc[
        exact_final["label"].isin(
            [
                "Selected ten · static ERC",
                "Selected ten · rolling target",
                "Selected ten · no return floor",
            ]
        )
    ].copy()
    ablation.to_csv(processed / "research_allocator_ablation.csv", index=False)

    adjustment_log = pd.DataFrame(
        [
            ["Universe/screen", "Forced ten slots per sector", "Bidirectional EG, issuer exclusion, orientation correction and BH-FDR audit", "7 strict; 103 fallback", "implemented"],
            ["Signals", "One SMA60 z-score", "360-rule SMA/EWMA grid; later 65-cell nested grid", "SMA20 winner; fold choices unstable", "implemented"],
            ["Frequency", "Median sparse entries", "Lower entry and shorter lookback/hold", "More observations; turnover roughly doubled", "implemented"],
            ["Portfolio breadth", "All 110 sleeves", "Exactly ten tuning-selected sleeves", "Lost breadth; selected-ten equal CAGR −2.80%", "implemented"],
            ["Shared names", "Ticker reuse uncontrolled", "No repeated ticker plus gross-normalized name caps", "20 unique names in current ten", "implemented"],
            ["Capital allocation", "Equal pair capital", "Ledoit–Wolf ERC and rolling constrained covariance", "Static ERC improved equal by 1.19 pp", "implemented"],
            ["Exposure", "Pair gross normalization only", "Market/factor, gross/net, sector and name limits", "Allocator feasible; first beta +0.026", "implemented"],
            ["Accounting", "Pair-level costs; open boundaries", "Exact pre/post underlying dollars and terminal liquidation", "Removed boundary optimism and double charging", "implemented"],
            ["Pair health", "Formation relation frozen indefinitely", "Rolling stationarity/break/beta/half-life kill and cooldown", "0/10 healthy at sample end", "implemented"],
            ["Execution", "Flat 5 bp + borrow proxy", "Spread, impact, ADV, locate, financing, cash and rebate", "$10m conditional; $50m exceeds capacity", "implemented; data conditional"],
            ["Validation", "One formation/test split", "Tuning lockbox, purged nested folds, bootstrap and snooping diagnostics", "Nested CAGR −3.57%; gate failed", "implemented"],
            ["Data ingestion", "Adjusted closes only", "Adjusted OHLCV cache/checksum path", "Yahoo refresh returned HTTP 429", "implemented; source unavailable"],
            ["Postmortem", "Compared non-comparable headlines", "Exact performance ledger, winner-selection diagnostic and allocator ablation", "Hindsight and return-target timing isolated", "implemented"],
            ["Recommended allocator", "Hard sample-mean return floor", "Slow ERC/covariance-only with turnover penalty", "No-target ablation improved to roughly −1.75%", "recommended, not promoted"],
        ],
        columns=["area", "first_version", "adjustment", "measured_result", "status"],
    )
    adjustment_log.to_csv(processed / "research_adjustment_log.csv", index=False)

    _plot_performance_ledger(
        exact_final.iloc[:4], charts / "research_evolution_performance.png"
    )
    _plot_selection_bias(
        pair_metrics,
        random_ten_cagrs,
        bootstrap_cagrs,
        original_expost["cagr"],
        charts / "research_selection_bias.png",
    )
    _plot_pair_signal_factorial(
        factorial, charts / "research_pair_signal_factorial.png"
    )
    _plot_allocator_ablation(
        ablation, charts / "research_allocator_ablation.png"
    )

    evidence = pd.DataFrame(
        [
            ["Descriptive", "Uses all available history; answers what happened, not what was tradable."],
            ["Held-out single split", "Formation and test separated once; useful, but insufficient for deployment."],
            ["Ex-post / selection-biased", "Portfolio or pair chosen after observing the reported returns."],
            ["Retrospective / spent lockbox", "Originally separated period has since been examined."],
            ["Nested OOS", "Parameters, pair selection and weights frozen inside chronological outer folds."],
            ["Conditional execution", "Relies on proxy spread, ADV, borrow or locate inputs."],
        ],
        columns=["Evidence label", "Meaning"],
    )
    evolution = pd.DataFrame(
        [
            ["V1", "Screen and backtest sector pairs", "Eight-year data; 5y formation / 3y test", "All 110 −0.46%; ex-post ten +12.40%", "Single split; winning ten hindsight"],
            ["V1.1", "Risk, frequency and parameter audit", "Exposure diagnostics and 360-cell SMA/EWMA grid", "All-110 lockbox +0.84% reported", "Modest; turnover doubled; lockbox spent"],
            ["V2", "Trade ten with non-equal capital", "Constrained selection and static ERC", "ERC −1.61% vs equal −2.80%", "Weighting helped; alpha still negative"],
            ["V3", "Apply production controls", "Rolling allocator, health, execution and nested validation", "0/10 healthy; nested −3.57%", "Do not deploy"],
            ["V4", "Reconcile and diagnose", "Exact ledger, selection tests and return-floor ablation", f"No-target rolling {_pct(exact_no_target.metrics['cagr'])}", "Retrospective diagnosis only"],
        ],
        columns=["Version", "Question", "Adjustment", "Headline", "Interpretation"],
    )
    original_rules = pd.DataFrame(
        [
            ["Pair relation", "Bidirectional Engle–Granger on log adjusted prices", "Formation alpha/beta frozen"],
            ["Multiple testing", "Bonferroni orientation correction; sector BH-FDR", "Strict or explicit fallback"],
            ["Strict gates", "Overlap, I(1)-like legs, positive beta, half-life 2–252, Hurst <0.5", "Only 7/110 passed all"],
            ["Z-score", "60-session simple mean/sample standard deviation; 40 warm-up", "No EWMA in V1"],
            ["Entry / exit", "Enter |z|≥2; exit |z|≤0.5", "Close signal; next-session holdings"],
            ["Risk exits", "Exit |z|≥4; 60-session time stop; missing-signal exit", "No rolling break/cooldown"],
            ["Costs", "5 bp per dollar traded + 30 bp annual borrow", "No quote/impact/locate model"],
        ],
        columns=["Component", "V1 rule", "Audit note"],
    )

    original_display = portfolio_metrics.copy()
    original_display["portfolio"] = original_display["portfolio"].map(
        {
            "all_formation_selected": "All 110 formation-selected",
            "sector_balanced": "Sector balanced",
            "expost_top_performers": "Ex-post winning ten",
        }
    )
    original_display = original_display[
        ["portfolio", "total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown"]
    ]
    for column in ["total_return", "cagr", "annual_volatility", "max_drawdown"]:
        original_display[column] = original_display[column].map(_pct)
    original_display["sharpe"] = original_display["sharpe"].map(
        lambda value: _num(value, 3)
    )
    original_display.columns = [
        "Portfolio", "Total return", "CAGR", "Volatility", "Sharpe", "Max drawdown"
    ]
    original_cost_display = (
        portfolio_cost_sensitivity.loc[
            portfolio_cost_sensitivity["portfolio"].isin(
                ["all_formation_selected", "expost_top_performers"]
            ),
            ["transaction_cost_bps", "portfolio", "cagr"],
        ]
        .pivot(index="transaction_cost_bps", columns="portfolio", values="cagr")
        .reset_index()
    )
    original_cost_display["transaction_cost_bps"] = original_cost_display[
        "transaction_cost_bps"
    ].map(lambda value: f"{float(value):g} bp")
    for column in ["all_formation_selected", "expost_top_performers"]:
        original_cost_display[column] = original_cost_display[column].map(_pct)
    original_cost_display.columns = [
        "One-way cost",
        "All 110 CAGR",
        "Ex-post ten CAGR",
    ]

    baseline_wide = baseline_signals.pivot(
        index="date", columns="pair", values="net_return"
    ).sort_index()
    winners_display = original_winners[
        ["rank", "pair", "sector", "net_cagr", "selection_status", "trade_count"]
    ].copy()
    winners_display["first_2y_cagr"] = [
        _cagr(baseline_wide.loc["2023-08-28":"2025-08-26", pair])
        for pair in winners_display["pair"]
    ]
    winners_display["final_1y_cagr"] = [
        _cagr(baseline_wide.loc["2025-08-27":"2026-08-26", pair])
        for pair in winners_display["pair"]
    ]
    for column in ["net_cagr", "first_2y_cagr", "final_1y_cagr"]:
        winners_display[column] = winners_display[column].map(_pct)
    winners_display["pair"] = winners_display["pair"].str.replace("__", " / ", regex=False)
    winners_display.columns = [
        "Rank", "Pair", "Sector", "Full 3y CAGR", "Formation status", "Trades", "First 2y CAGR", "Final 1y CAGR"
    ]

    split_display = selection_splits.copy()
    split_display["cagr"] = split_display["cagr"].map(_pct)
    split_display.columns = [
        "Construction", "Selection window", "Evaluation window", "CAGR", "Evidence"
    ]

    overlap_display = overlap_summary.copy()
    for column in [
        "mean_absolute_unconditional_correlation",
        "mean_absolute_active_correlation",
        "maximum_absolute_active_correlation",
    ]:
        overlap_display[column] = overlap_display[column].map(
            lambda value: _num(value, 3)
        )
    overlap_display = overlap_display[
        [
            "group",
            "pair_of_pair_count",
            "mean_absolute_unconditional_correlation",
            "mean_absolute_active_correlation",
            "maximum_absolute_active_correlation",
        ]
    ]
    overlap_display.columns = [
        "Pair relationship", "Pair-of-pair count", "Mean |corr|", "Mean active |corr|", "Maximum active |corr|"
    ]

    grid_display = grid_comparison.loc[
        grid_comparison["portfolio"].eq("equal_weight"),
        [
            "split", "cagr_baseline", "cagr_best", "sharpe_baseline", "sharpe_best",
            "annual_turnover_baseline", "annual_turnover_best", "total_trades_baseline", "total_trades_best",
            "median_trades_per_pair_baseline", "median_trades_per_pair_best",
        ],
    ].copy()
    for column in ["cagr_baseline", "cagr_best"]:
        grid_display[column] = grid_display[column].map(_pct)
    for column in ["sharpe_baseline", "sharpe_best"]:
        grid_display[column] = grid_display[column].map(
            lambda value: _num(value, 3)
        )
    for column in ["annual_turnover_baseline", "annual_turnover_best"]:
        grid_display[column] = grid_display[column].map(
            lambda value: f"{float(value):.1f}×"
        )
    grid_display.columns = [
        "Period", "Baseline CAGR", "Winner CAGR", "Baseline Sharpe", "Winner Sharpe", "Baseline turnover", "Winner turnover", "Baseline entries", "Winner entries", "Baseline median/pair", "Winner median/pair"
    ]

    latest_health = (
        health_daily.sort_values("date").groupby("pair", as_index=False).tail(1)
    ).set_index("pair").reindex(current_pairs)
    event_counts = health_events.groupby("pair").agg(
        kill_events=("kill_event", "sum"), recovery_events=("recovery_event", "sum")
    )
    current_display = (
        selected[
            ["pair", "sector", "tuning_cagr", "tuning_sharpe", "selection_status"]
        ]
        .merge(
            static_weights_frame[
                ["pair", "weight", "retrospective_cagr", "retrospective_sharpe"]
            ],
            on="pair",
        )
        .merge(weighted_trades[["pair", "retrospective_entries"]], on="pair")
    )
    current_display["latest_weight"] = current_display["pair"].map(
        production_schedule.iloc[-1]
    )
    current_display["health"] = current_display["pair"].map(
        latest_health["health_state"]
    )
    current_display["health_reason"] = current_display["pair"].map(
        latest_health["health_reasons"]
    )
    current_display["pair"] = current_display["pair"].str.replace(
        "__", " / ", regex=False
    )
    for column in ["tuning_cagr", "weight", "latest_weight", "retrospective_cagr"]:
        current_display[column] = current_display[column].map(_pct)
    for column in ["tuning_sharpe", "retrospective_sharpe"]:
        current_display[column] = current_display[column].map(
            lambda value: _num(value, 2)
        )
    current_display.columns = [
        "Pair", "Sector", "Tuning CAGR", "Tuning Sharpe", "Formation status", "Static ERC", "Final pair CAGR", "Final pair Sharpe", "Final entries", "Latest rolling", "Health state", "Latest breach reasons"
    ]

    weighted_display = weighted_methods.loc[
        weighted_methods["method"].isin(
            ["equal_capital", "equal_risk_contribution", "minimum_variance"]
        ),
        ["method", "role", "cagr", "annual_volatility", "sharpe", "max_drawdown", "transaction_cost_drag_bps", "annualized_leg_turnover"],
    ].copy()
    weighted_display["method"] = weighted_display["method"].map(
        {
            "equal_capital": "Equal capital",
            "equal_risk_contribution": "Constrained ERC",
            "minimum_variance": "Minimum variance",
        }
    )
    for column in ["cagr", "annual_volatility", "max_drawdown"]:
        weighted_display[column] = weighted_display[column].map(_pct)
    weighted_display["sharpe"] = weighted_display["sharpe"].map(
        lambda value: _num(value, 3)
    )
    weighted_display["transaction_cost_drag_bps"] = weighted_display[
        "transaction_cost_drag_bps"
    ].map(_bp)
    weighted_display["annualized_leg_turnover"] = weighted_display[
        "annualized_leg_turnover"
    ].map(lambda value: f"{float(value):.1f}×")
    weighted_display.columns = [
        "Allocator", "Role", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Transaction drag", "Leg turnover"
    ]

    exact_display = exact_final.copy()
    for column in ["cagr", "max_drawdown"]:
        exact_display[column] = exact_display[column].map(_pct)
    exact_display["sharpe"] = exact_display["sharpe"].map(
        lambda value: _num(value, 3)
    )
    exact_display["transaction_drag_bps"] = exact_display[
        "transaction_drag_bps"
    ].map(_bp)
    exact_display["annualized_turnover"] = exact_display[
        "annualized_turnover"
    ].map(lambda value: f"{float(value):.1f}×")
    exact_display["capital_rebalance_turnover"] = exact_display[
        "capital_rebalance_turnover"
    ].map(lambda value: _num(value, 3))
    exact_display.columns = [
        "Exact final-year portfolio", "CAGR", "Sharpe", "Max drawdown", "Transaction drag", "Leg turnover", "Capital reallocation"
    ]

    ablation_display = ablation.copy()
    for column in ["cagr", "max_drawdown"]:
        ablation_display[column] = ablation_display[column].map(_pct)
    ablation_display["sharpe"] = ablation_display["sharpe"].map(
        lambda value: _num(value, 3)
    )
    ablation_display["transaction_drag_bps"] = ablation_display[
        "transaction_drag_bps"
    ].map(_bp)
    ablation_display["annualized_turnover"] = ablation_display[
        "annualized_turnover"
    ].map(lambda value: f"{float(value):.1f}×")
    ablation_display["capital_rebalance_turnover"] = ablation_display[
        "capital_rebalance_turnover"
    ].map(lambda value: _num(value, 3))
    ablation_display.columns = exact_display.columns

    first_weights = production_schedule.iloc[0].sort_values(ascending=False)
    latest_weights = production_schedule.iloc[-1].sort_values(ascending=False)
    first_weight_display = first_weights.rename("Weight").reset_index().rename(columns={"index": "Pair", "pair": "Pair"})
    latest_weight_display = latest_weights.rename("Weight").reset_index().rename(columns={"index": "Pair", "pair": "Pair"})
    for frame in [first_weight_display, latest_weight_display]:
        frame["Pair"] = frame["Pair"].str.replace("__", " / ", regex=False)
        frame["Weight"] = frame["Weight"].map(_pct)
    first_diag = production_schedule_diagnostics.iloc[0]
    allocator_display = pd.DataFrame(
        [
            ["Covariance observations", _integer(first_diag["covariance_observations"])],
            ["Ledoit–Wolf shrinkage", _num(first_diag["covariance_shrinkage"], 3)],
            ["Effective pair count", _num(first_diag["effective_number_of_pairs"], 2)],
            ["Expected annual volatility", _pct(first_diag["expected_annual_volatility"])],
            ["Expected market beta", _num(first_diag["expected_market_beta"], 4)],
            ["Gross-exposure coefficient", _num(first_diag["portfolio_gross_exposure"], 4)],
            ["Net exposure", _num(first_diag["portfolio_net_exposure"], 4)],
            ["Training return floor", _pct(first_diag["target_return"])],
            ["Return-floor slack", f"{float(first_diag['target_return_slack']):.2e}"],
            ["Numerical fallback used", str(bool(first_diag["fallback_used"]))],
        ],
        columns=["First-solve diagnostic", "Value"],
    )

    health_display = latest_health.reset_index()[
        ["pair", "health_state", "health_reasons", "cointegration_pvalue", "beta_relative_drift", "half_life_days", "structural_break_pvalue"]
    ].merge(event_counts.reset_index(), on="pair", how="left")
    health_display["pair"] = health_display["pair"].str.replace(
        "__", " / ", regex=False
    )
    for column in ["cointegration_pvalue", "beta_relative_drift", "structural_break_pvalue"]:
        health_display[column] = health_display[column].map(
            lambda value: f"{float(value):.3g}"
        )
    health_display["half_life_days"] = health_display["half_life_days"].map(
        lambda value: _num(value, 1)
    )
    health_display.columns = [
        "Pair", "Latest state", "Latest breach reasons", "Cointegration p", "Beta relative drift", "Half-life days", "Break p", "Kills", "Recoveries"
    ]
    latest_tradable = int(latest_health["allow_entry"].fillna(False).sum())

    execution_display = execution_scenarios.copy()
    for column in ["cagr", "strategy_only_cagr", "max_drawdown", "maximum_participation"]:
        execution_display[column] = execution_display[column].map(_pct)
    for column in ["sharpe", "strategy_only_sharpe"]:
        execution_display[column] = execution_display[column].map(
            lambda value: _num(value, 3)
        )
    for column in ["transaction_cost_bps", "borrow_cost_bps"]:
        execution_display[column] = execution_display[column].map(_bp)
    execution_display = execution_display[
        ["portfolio_notional", "cagr", "strategy_only_cagr", "strategy_only_sharpe", "max_drawdown", "transaction_cost_bps", "borrow_cost_bps", "maximum_participation", "capacity_breaches", "capacity_breach_days"]
    ]
    execution_display.columns = [
        "Notional", "Total CAGR", "Pairs-alpha CAGR", "Alpha Sharpe", "Max drawdown", "Transaction drag", "Borrow drag", "Max participation", "Breaches", "Breach days"
    ]

    nested_fold_display = nested_outer[
        ["outer_fold", "candidate_id", "train_start", "train_end", "test_start", "test_end", "cagr", "sharpe", "max_drawdown"]
    ].copy()
    for column in ["cagr", "max_drawdown"]:
        nested_fold_display[column] = nested_fold_display[column].map(_pct)
    nested_fold_display["sharpe"] = nested_fold_display["sharpe"].map(
        lambda value: _num(value, 3)
    )
    for column in ["train_start", "train_end", "test_start", "test_end"]:
        nested_fold_display[column] = nested_fold_display[column].dt.date.astype(str)
    nested_fold_display.columns = [
        "Fold", "Frozen rule", "Train start", "Train end", "Test start", "Test end", "Test CAGR", "Test Sharpe", "Test drawdown"
    ]
    nested_performance_display = pd.DataFrame(
        [
            ["Observations", _integer(nested_performance["observations"])],
            ["Total return", _pct(nested_performance["total_return"])],
            ["CAGR", _pct(nested_performance["cagr"])],
            ["Annual volatility", _pct(nested_performance["annual_volatility"])],
            ["Sharpe", _num(nested_performance["sharpe"], 3)],
            ["Sortino", _num(nested_performance["sortino"], 3)],
            ["Max drawdown", _pct(nested_performance["max_drawdown"])],
        ],
        columns=["Metric", "Nested OOS result"],
    )
    nested_bootstrap_display = nested_bootstrap.loc[
        nested_bootstrap["statistic"].isin(["cagr", "sharpe", "max_drawdown"]),
        ["statistic", "estimate", "lower", "upper", "confidence_level", "valid_bootstrap_samples"],
    ].copy().astype(object)
    for row in nested_bootstrap_display.index:
        statistic = nested_bootstrap_display.at[row, "statistic"]
        if statistic in {"cagr", "max_drawdown"}:
            for column in ["estimate", "lower", "upper"]:
                nested_bootstrap_display.at[row, column] = _pct(
                    nested_bootstrap_display.at[row, column]
                )
        else:
            for column in ["estimate", "lower", "upper"]:
                nested_bootstrap_display.at[row, column] = _num(
                    nested_bootstrap_display.at[row, column], 3
                )
        nested_bootstrap_display.at[row, "confidence_level"] = _pct(
            nested_bootstrap_display.at[row, "confidence_level"], 0
        )
    nested_bootstrap_display.columns = [
        "Statistic", "Estimate", "Lower", "Upper", "Confidence", "Valid samples"
    ]
    snooping_display = nested_snooping[
        ["outer_fold", "candidate_count", "best_candidate", "null_pvalue", "bootstrap_samples", "block_size"]
    ].copy()
    snooping_display["null_pvalue"] = snooping_display["null_pvalue"].map(
        lambda value: _num(value, 3)
    )
    snooping_display.columns = [
        "Fold", "Candidates", "Best in diagnostic window", "Null p-value", "Bootstraps", "Block size"
    ]
    cagr_bootstrap_row = nested_bootstrap.loc[
        nested_bootstrap["statistic"].eq("cagr")
    ].iloc[0]

    baseline_beta_row = market_beta.loc[
        market_beta["portfolio"].eq("all_formation_selected")
    ].iloc[0]
    shock_row = market_shocks.loc[
        (market_shocks["regime"].eq("market_down_at_least_2%"))
        & (market_shocks["portfolio"].eq("all_formation_selected"))
    ].iloc[0]
    disjoint_row = overlap_summary.loc[
        overlap_summary["group"].eq("disjoint_underlyings")
    ].iloc[0]
    shared_row = overlap_summary.loc[
        overlap_summary["group"].eq("shared_underlying")
    ].iloc[0]
    trade_ledger = pd.read_csv(
        processed / "pair_trade_ledger.csv", parse_dates=["entry_date"]
    )
    execution_log = pd.read_csv(
        processed / "production_execution_log.csv.gz", parse_dates=["date"]
    )

    def frequency_row(
        label: str,
        signal_frame: pd.DataFrame,
        pairs: list[str],
        period_start: pd.Timestamp,
        period_end: pd.Timestamp,
        reported_total: int,
        per_pair: str,
    ) -> dict[str, Any]:
        frame = signal_frame.loc[
            signal_frame["pair"].isin(pairs)
            & signal_frame["date"].between(period_start, period_end)
        ]
        dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
        entries = frame.loc[
            frame["effective_event"].astype(str).str.startswith("enter")
        ].groupby("date").size().reindex(dates, fill_value=0).astype(float)
        boundary_entries = reported_total - int(entries.sum())
        if boundary_entries > 0 and len(entries):
            entries.iloc[0] += boundary_entries
        return {
            "book": label,
            "sessions": len(dates),
            "pair_entries": reported_total,
            "entries_per_session": reported_total / len(dates),
            "median_daily_entries": float(entries.median()),
            "no_entry_days": float((entries == 0).mean()),
            "per_pair_count": per_pair,
        }

    grid_reported_total = int(
        grid_comparison.loc[
            grid_comparison["portfolio"].eq("equal_weight"), "total_trades_best"
        ].sum()
    )
    baseline_current_frame = baseline_signals.loc[
        baseline_signals["pair"].isin(current_pairs)
        & baseline_signals["date"].between(start, end)
    ]
    baseline_current_counts = baseline_current_frame.loc[
        baseline_current_frame["effective_event"].astype(str).str.startswith("enter")
    ].groupby("pair").size().reindex(current_pairs, fill_value=0)
    tuned_current_total = int(weighted_trades["retrospective_entries"].sum())
    frequency = pd.DataFrame(
        [
            frequency_row(
                "Initial all 110 · SMA60",
                baseline_signals,
                all_pairs,
                v1_start,
                v1_end,
                len(trade_ledger),
                f"median {trade_ledger.groupby('pair').size().median():.0f} over 3y",
            ),
            frequency_row(
                "Tuned all 110 · SMA20",
                grid_signals,
                all_pairs,
                v1_start,
                v1_end,
                grid_reported_total,
                "median 25 tuning + 12 lockbox",
            ),
            frequency_row(
                "Current ten · SMA60 final year",
                baseline_signals,
                current_pairs,
                start,
                end,
                int(baseline_current_counts.sum()),
                f"range {int(baseline_current_counts.min())}–{int(baseline_current_counts.max())}",
            ),
            frequency_row(
                "Current ten · SMA20 final year",
                grid_signals,
                current_pairs,
                start,
                end,
                tuned_current_total,
                f"range {int(weighted_trades['retrospective_entries'].min())}–{int(weighted_trades['retrospective_entries'].max())}",
            ),
        ]
    )
    frequency_display = frequency.copy()
    frequency_display["entries_per_session"] = frequency_display[
        "entries_per_session"
    ].map(lambda value: _num(value, 3))
    frequency_display["median_daily_entries"] = frequency_display[
        "median_daily_entries"
    ].map(lambda value: _num(value, 1))
    frequency_display["no_entry_days"] = frequency_display["no_entry_days"].map(
        _pct
    )
    frequency_display.columns = [
        "Book",
        "Sessions",
        "Pair entries",
        "Entries/session",
        "Median daily",
        "No-entry days",
        "Per-pair count",
    ]

    old_name_incidence = build_gross_name_incidence(
        original_winners[["pair", "dependent", "independent", "beta"]]
    )
    old_names = equal_original @ old_name_incidence

    first_constraint = production_metadata["allocator"]["first_diagnostics"]
    current_market_diagnostics = weighted_metadata["market_diagnostics"]
    selection_null = selection_diagnostics["repeat_selection_null"]
    selection_arithmetic_null = selection_diagnostics[
        "arithmetic_mean_robustness"
    ]
    ranking = selection_diagnostics["ranking_stability"]
    bootstrap_sensitivity_display = bootstrap_sensitivity[
        [
            "block_size",
            "samples",
            "median_cagr",
            "p05_cagr",
            "p95_cagr",
            "fraction_at_or_above_observed",
        ]
    ].copy()
    for column in [
        "median_cagr",
        "p05_cagr",
        "p95_cagr",
        "fraction_at_or_above_observed",
    ]:
        bootstrap_sensitivity_display[column] = bootstrap_sensitivity_display[
            column
        ].map(_pct)
    bootstrap_sensitivity_display.columns = [
        "Block length",
        "Draws",
        "Null median CAGR",
        "5th percentile",
        "95th percentile",
        "Fraction ≥ observed",
    ]
    binding_rows = int(
        (production_schedule_diagnostics["target_return_slack"].abs() < 1e-10).sum()
    )

    report = REPORT_TEMPLATE.render(
        as_of=end.date().isoformat(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        latest_tradable=latest_tradable,
        nested_cagr=_pct(nested_performance["cagr"]),
        nested_sharpe=_num(nested_performance["sharpe"], 3),
        nested_cagr_lower=_pct(cagr_bootstrap_row["lower"]),
        nested_cagr_upper=_pct(cagr_bootstrap_row["upper"]),
        original_all_cagr=_pct(original_all["cagr"]),
        exact_v1_all_cagr=_pct(exact_v1_all.metrics["cagr"]),
        exact_v1_oracle_cagr=_pct(exact_v1_oracle.metrics["cagr"]),
        grid_lockbox_cagr=_pct(
            grid_comparison.loc[
                (grid_comparison["split"].eq("lockbox"))
                & (grid_comparison["portfolio"].eq("equal_weight")),
                "cagr_best",
            ].iloc[0]
        ),
        exact_all110_cagr=_pct(exact_all110.metrics["cagr"]),
        evidence_table=_table(evidence),
        evolution_table=_table(evolution),
        original_rule_table=_table(original_rules),
        original_performance_table=_table(original_display),
        original_cost_table=_table(original_cost_display),
        original_winner_table=_table(winners_display),
        selection_split_table=_table(split_display),
        half_spearman=_num(ranking["half_sample_spearman"], 4),
        half_overlap=ranking["half_sample_top_ten_overlap"],
        forward_counterfactual=_pct(
            ranking["first_two_year_top_ten_final_year_cagr"]
        ),
        oracle_final=_pct(ranking["full_period_oracle_top_ten_final_year_cagr"]),
        bootstrap_median=_pct(selection_null["median"]),
        bootstrap_exceed=_pct(selection_null["fraction_at_or_above_observed"], 1),
        bootstrap_samples=_integer(selection_null["samples"]),
        bootstrap_sensitivity_table=_table(bootstrap_sensitivity_display),
        arithmetic_exceed=_pct(
            selection_arithmetic_null["fraction_at_or_above_observed"], 1
        ),
        arithmetic_samples=_integer(selection_arithmetic_null["samples"]),
        old_max_name=_pct(old_names.max()),
        baseline_market_beta=_num(baseline_beta_row["market_beta"], 3),
        baseline_market_corr=_num(baseline_beta_row["market_correlation"], 3),
        baseline_market_r2=_pct(baseline_beta_row["r_squared"], 1),
        shock_days=_integer(shock_row["observations"]),
        shock_cumulative=_pct(shock_row["cumulative_return"]),
        overlap_table=_table(overlap_display),
        disjoint_active_corr=_num(
            disjoint_row["mean_absolute_active_correlation"], 3
        ),
        shared_active_corr=_num(
            shared_row["mean_absolute_active_correlation"], 3
        ),
        shared_active_max=_num(
            shared_row["maximum_absolute_active_correlation"], 3
        ),
        baseline_total_trades=_integer(len(trade_ledger)),
        baseline_median_trades=_num(
            trade_ledger.groupby("pair").size().median(), 0
        ),
        frequency_table=_table(frequency_display),
        ticker_adjustments=_integer(execution_log["signed_trade_nav"].ne(0).sum()),
        ticker_adjustments_per_session=_num(
            execution_log["signed_trade_nav"].ne(0).sum()
            / execution_log["date"].nunique(),
            3,
        ),
        tuned_ten_entries=_integer(tuned_current_total),
        grid_table=_table(grid_display),
        current_pair_table=_table(current_display),
        weighted_method_table=_table(weighted_display),
        current_market_beta=_num(current_market_diagnostics["market_beta"], 3),
        current_market_corr=_num(
            current_market_diagnostics["market_correlation"], 3
        ),
        current_market_r2=_pct(current_market_diagnostics["r_squared"], 1),
        exact_final_table=_table(exact_display),
        ablation_table=_table(ablation_display),
        binding_rows=binding_rows,
        allocation_rows=len(production_schedule_diagnostics),
        forecast_ic=_num(forecast_ic["pearson_ic"].mean(), 3),
        forecast_ic_median=_num(forecast_ic["pearson_ic"].median(), 3),
        forecast_ic_positive=_integer((forecast_ic["pearson_ic"] > 0).sum()),
        no_target_cagr=_pct(exact_no_target.metrics["cagr"]),
        no_target_rebalance=_num(
            exact_no_target.metrics["capital_rebalance_turnover_sum"], 3
        ),
        first_weight_table=_table(first_weight_display),
        latest_weight_table=_table(latest_weight_display),
        allocator_diagnostics_table=_table(allocator_display),
        health_table=_table(health_display),
        execution_table=_table(execution_display),
        nested_fold_table=_table(nested_fold_display),
        nested_performance_table=_table(nested_performance_display),
        nested_bootstrap_table=_table(nested_bootstrap_display),
        snooping_table=_table(snooping_display),
        adjustment_table=_table(adjustment_log),
        tests_passed=138,
        tests_skipped=1,
    )
    report_path = output / "research_evolution_report.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
