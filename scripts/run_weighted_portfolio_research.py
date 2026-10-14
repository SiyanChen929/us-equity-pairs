#!/usr/bin/env python3
"""Build a leakage-aware, non-equal top-ten pair portfolio and audit report."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from equity_pairs.allocation import (
    allocation_diagnostics,
    build_allocation_candidates,
)
from equity_pairs.backtest import performance_metrics
from equity_pairs.diagnostics import regression_diagnostics
from equity_pairs.selection import (
    PairSelectionSpec,
    build_pair_selection_features,
    select_constrained_pairs_milp,
)
from equity_pairs.weighted_portfolio import (
    alpha_tilted_risk_budgets_from_forecasts,
    backtest_weighted_pair_book,
    build_gross_name_incidence,
    empirical_bayes_monthly_forecasts,
)


INK = "#162033"
MUTED = "#66758a"
BLUE = "#2864b7"
TEAL = "#14827a"
GREEN = "#26845b"
ORANGE = "#d17a2f"
RED = "#b74650"
PURPLE = "#7057ad"
PALE = "#eef3f8"
METHOD_COLORS = {
    "equal_risk_contribution": BLUE,
    "inverse_volatility": TEAL,
    "minimum_variance": PURPLE,
    "maximum_diversification": GREEN,
    "shrunk_tangency": ORANGE,
    "equal_capital": MUTED,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#cfd7e3",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e7ecf2", linewidth=0.8)
    axis.set_axisbelow(True)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}%}" if math.isfinite(number) else "—"


def _pp(value: Any, digits: int = 2) -> str:
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.{digits}f} pp" if math.isfinite(number) else "—"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _table(frame: pd.DataFrame, *, classes: str = "data-table") -> str:
    if frame.empty:
        return '<p class="muted">No rows available.</p>'
    return frame.to_html(index=False, border=0, classes=classes, escape=True)


def _pair_market_betas(
    returns: pd.DataFrame,
    market: pd.Series,
) -> pd.Series:
    aligned_market = pd.to_numeric(market.reindex(returns.index), errors="coerce")
    variance = float(aligned_market.var(ddof=1))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("market proxy has no usable variance in the tuning window")
    result = returns.apply(
        lambda series: pd.to_numeric(series, errors="coerce").cov(aligned_market)
        / variance
    )
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("one or more tuning pair betas could not be estimated")
    result.name = "tuning_market_beta"
    return result


def _active_correlation(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    minimum_overlap: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    pairs = list(returns.columns)
    matrix = pd.DataFrame(np.nan, index=pairs, columns=pairs)
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(pairs):
        matrix.loc[left, left] = 1.0
        for right in pairs[left_index + 1 :]:
            mask = positions[left].ne(0) & positions[right].ne(0)
            overlap = int(mask.sum())
            correlation = (
                float(returns.loc[mask, left].corr(returns.loc[mask, right]))
                if overlap >= minimum_overlap
                else math.nan
            )
            matrix.loc[left, right] = correlation
            matrix.loc[right, left] = correlation
            rows.append(
                {
                    "pair_left": left,
                    "pair_right": right,
                    "active_overlap_days": overlap,
                    "active_correlation": correlation,
                }
            )
    detail = pd.DataFrame(rows)
    values = pd.to_numeric(detail["active_correlation"], errors="coerce").dropna()
    summary = {
        "mean_absolute": float(values.abs().mean()),
        "maximum_absolute": float(values.abs().max()),
        "median_overlap_days": float(detail["active_overlap_days"].median()),
        "valid_pair_count": int(len(values)),
    }
    return matrix, detail, summary


def _unconditional_correlation_summary(matrix: pd.DataFrame) -> dict[str, float]:
    values = matrix.to_numpy(dtype=float)[np.triu_indices(len(matrix), 1)]
    return {
        "mean": float(np.mean(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "maximum_absolute": float(np.max(np.abs(values))),
        "p95_absolute": float(np.quantile(np.abs(values), 0.95)),
    }


def _name_exposures(
    signals: pd.DataFrame,
    deployed_weights: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = deployed_weights.index
    selected = set(deployed_weights.columns)
    work = signals.loc[signals["pair"].isin(selected)].copy()
    work = work.loc[work["date"].isin(dates)]
    dependent = work.pivot(index="date", columns="pair", values="weight_dependent").reindex(
        index=dates, columns=deployed_weights.columns
    )
    independent = work.pivot(
        index="date", columns="pair", values="weight_independent"
    ).reindex(index=dates, columns=deployed_weights.columns)
    if dependent.isna().any().any() or independent.isna().any().any():
        raise ValueError("selected signal panel is incomplete for exposure reconstruction")
    tickers = sorted(
        set(metadata["dependent"].astype(str))
        | set(metadata["independent"].astype(str))
    )
    name_exposure = pd.DataFrame(0.0, index=dates, columns=tickers)
    lookup = metadata.set_index("pair")
    for pair in deployed_weights.columns:
        pair_capital = deployed_weights[pair]
        name_exposure[str(lookup.loc[pair, "dependent"])] += pair_capital * dependent[pair]
        name_exposure[str(lookup.loc[pair, "independent"])] += pair_capital * independent[pair]
    name_exposure.index.name = "date"
    portfolio = pd.DataFrame(index=dates)
    portfolio["long_exposure"] = name_exposure.clip(lower=0).sum(axis=1)
    portfolio["short_exposure"] = -name_exposure.clip(upper=0).sum(axis=1)
    portfolio["gross_exposure"] = name_exposure.abs().sum(axis=1)
    portfolio["net_exposure"] = name_exposure.sum(axis=1)
    portfolio["maximum_absolute_name_exposure"] = name_exposure.abs().max(axis=1)
    position = work.pivot(index="date", columns="pair", values="position").reindex(
        index=dates, columns=deployed_weights.columns
    )
    portfolio["active_pair_count"] = position.ne(0).sum(axis=1)
    portfolio.index.name = "date"
    return name_exposure, portfolio


def _plot_allocation(
    weights: pd.DataFrame,
    output: Path,
) -> None:
    ordered = weights.sort_values("weight", ascending=True)
    labels = ordered["pair"].str.replace("__", "/", regex=False)
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.1))
    axes[0].barh(labels, ordered["weight"] * 100, color=BLUE, alpha=0.9)
    axes[0].axvline(10, color=MUTED, linestyle="--", linewidth=1, label="Equal capital")
    axes[0].axvline(5, color="#aab5c3", linestyle=":", linewidth=1)
    axes[0].axvline(15, color="#aab5c3", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Target capital weight (%)")
    axes[0].set_title("Capital is deliberately non-equal", loc="left")
    axes[0].legend(frameon=False)
    axes[1].barh(
        labels,
        ordered["risk_contribution_fraction"] * 100,
        color=TEAL,
        alpha=0.9,
    )
    axes[1].axvline(10, color=MUTED, linestyle="--", linewidth=1, label="10% risk target")
    axes[1].set_xlabel("Tuning covariance risk contribution (%)")
    axes[1].set_title("Risk is balanced where constraints allow", loc="left")
    axes[1].legend(frameon=False)
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(w_pad=2.5)
    _save(figure, output)


def _plot_equity(
    results: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.1), sharex=True)
    display = {
        "equal_risk_contribution": "Constrained ERC · primary",
        "equal_capital": "Equal capital · comparator",
        "minimum_variance": "Minimum variance · diagnostic only",
    }
    for method, label in display.items():
        daily = results[method].daily
        color = METHOD_COLORS[method]
        width = 2.4 if method == "equal_risk_contribution" else 1.5
        alpha = 1.0 if method != "minimum_variance" else 0.7
        axes[0].plot(daily.index, daily["equity"], label=label, color=color, linewidth=width, alpha=alpha)
        drawdown = daily["equity"] / daily["equity"].cummax() - 1.0
        axes[1].plot(daily.index, drawdown * 100, color=color, linewidth=width, alpha=alpha, label=label)
    axes[0].axhline(1.0, color="#adb8c5", linewidth=0.8)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Already-seen final year: same ten pairs, different allocation", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].axhline(0, color="#adb8c5", linewidth=0.8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Portfolio drawdown", loc="left")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    for axis in axes:
        _style_axis(axis)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout(h_pad=2.0)
    _save(figure, output)


def _plot_constraints(
    sector_weights: pd.Series,
    name_weights: pd.Series,
    expected_beta: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))
    sectors = sector_weights.sort_values()
    axes[0].barh(sectors.index, sectors * 100, color=BLUE)
    axes[0].axvline(25, color=RED, linestyle="--", linewidth=1.2, label="25% cap")
    axes[0].set_xlabel("Target sleeve capital (%)")
    axes[0].set_title("Sector cap", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    names = name_weights.sort_values(ascending=False).head(10).sort_values()
    axes[1].barh(names.index, names * 100, color=TEAL)
    axes[1].axvline(10, color=RED, linestyle="--", linewidth=1.2, label="10% cap")
    axes[1].set_xlabel("Gross-normalized target exposure (%)")
    axes[1].set_title("Largest underlying-name budgets", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].axvspan(-0.05, 0.05, color="#e8f3ed")
    axes[2].axvline(0, color="#9da9b8", linewidth=1)
    axes[2].scatter([expected_beta], [0], s=120, color=PURPLE, zorder=4)
    axes[2].text(expected_beta, 0.07, f"β = {expected_beta:.3f}", ha="center", color=PURPLE, fontweight="bold")
    axes[2].set_xlim(-0.075, 0.075)
    axes[2].set_ylim(-0.18, 0.22)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Tuning market beta")
    axes[2].set_title("±0.05 beta band", loc="left")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(w_pad=2.0)
    _save(figure, output)


def _plot_correlations(
    tuning: pd.DataFrame,
    retrospective: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.1))
    labels = [pair.replace("__", "/") for pair in tuning.columns]
    for axis, matrix, title in (
        (axes[0], tuning, "Tuning active-overlap correlation"),
        (axes[1], retrospective, "Already-seen final-year active correlation"),
    ):
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=48, ha="right", fontsize=7)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        axis.set_title(title, loc="left")
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = float(matrix.iloc[row, column])
                if math.isfinite(value):
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color="white" if abs(value) > 0.30 else INK)
        figure.colorbar(image, ax=axis, fraction=0.044, pad=0.03)
    figure.tight_layout(w_pad=2.1)
    _save(figure, output)


def _plot_trade_activity(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("retrospective_entries", ascending=True)
    labels = ordered["pair"].str.replace("__", "/", regex=False)
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    axis.barh(positions - 0.19, ordered["annualized_tuning_entries"], height=0.36, color=MUTED, label="Tuning annualized")
    axis.barh(positions + 0.19, ordered["retrospective_entries"], height=0.36, color=BLUE, label="Final-year entries")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Trade entries per year")
    axis.set_title("The faster SMA20 rule is active: 10–14 entries per pair in the final year", loc="left")
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    figure.tight_layout()
    _save(figure, output)


def _plot_pair_signal(frame: pd.DataFrame, pair: str, output: Path) -> None:
    work = frame.sort_values("date").set_index("date")
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 6.1), sharex=True, gridspec_kw={"height_ratios": [1.55, 1]})
    axes[0].plot(work.index, work["zscore"], color=INK, linewidth=1.1, label="Z-score")
    axes[0].axhline(1.75, color=RED, linestyle="--", linewidth=1, label="Entry ±1.75")
    axes[0].axhline(-1.75, color=RED, linestyle="--", linewidth=1)
    axes[0].axhline(0.5, color=GREEN, linestyle=":", linewidth=1, label="Exit ±0.50")
    axes[0].axhline(-0.5, color=GREEN, linestyle=":", linewidth=1)
    long_entries = work["effective_event"].eq("enter_long_spread")
    short_entries = work["effective_event"].eq("enter_short_spread")
    exits = work["effective_event"].astype(str).str.startswith("exit_")
    axes[0].scatter(work.index[long_entries], work.loc[long_entries, "zscore"], marker="^", color=BLUE, s=34, label="Long entry", zorder=5)
    axes[0].scatter(work.index[short_entries], work.loc[short_entries, "zscore"], marker="v", color=ORANGE, s=34, label="Short entry", zorder=5)
    axes[0].scatter(work.index[exits], work.loc[exits, "zscore"], marker="x", color=GREEN, s=30, label="Exit", zorder=5)
    axes[0].set_ylabel("Z-score")
    axes[0].set_title(f"{pair.replace('__', '/')} · SMA20 decisions lagged one session", loc="left")
    axes[0].legend(frameon=False, ncol=5, fontsize=7)
    equity = (1.0 + work["net_return"].fillna(0.0)).cumprod()
    axes[1].plot(equity.index, equity, color=BLUE, linewidth=1.8)
    axes[1].axhline(1.0, color="#adb8c5", linewidth=0.8)
    axes[1].set_ylabel("Sleeve growth of $1")
    axes[1].set_xlabel("Date")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    for axis in axes:
        _style_axis(axis)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout(h_pad=1.5)
    _save(figure, output)


REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top-10 Non-Equal Pair Portfolio</title>
<style>
:root{--ink:#162033;--muted:#66758a;--blue:#2864b7;--teal:#14827a;--green:#26845b;--orange:#d17a2f;--red:#b74650;--line:#dce3ec;--pale:#f4f7fa;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#e9eef4;color:var(--ink);font:15px/1.58 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1220px;margin:24px auto;background:var(--paper);box-shadow:0 14px 42px rgba(32,52,78,.13)}header{padding:52px 64px 34px;background:linear-gradient(135deg,#15243d 0%,#244d7e 67%,#197c78 100%);color:white}header .eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:700;opacity:.78}h1{font:700 42px/1.08 Georgia,serif;margin:12px 0 14px;max-width:900px}header p{max-width:860px;font-size:17px;margin:0;opacity:.88}.content{padding:38px 64px 60px}h2{font:700 27px/1.2 Georgia,serif;margin:46px 0 14px}h3{font-size:17px;margin:28px 0 8px}.lead{font-size:17px;color:#33445b}.warning{border-left:5px solid var(--orange);background:#fff6e9;padding:17px 20px;margin:0 0 26px}.verdict{border:1px solid #cfe0db;background:#eff8f5;padding:20px 22px;border-radius:8px;font-size:16px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}.card{border:1px solid var(--line);border-radius:8px;padding:17px 18px;background:white}.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700}.card .v{font-size:28px;line-height:1.18;font-weight:750;margin:7px 0 3px}.card .s{font-size:12px;color:var(--muted)}.chart{display:block;width:100%;border:1px solid var(--line);border-radius:7px;margin:18px 0 10px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.signal-card{border:1px solid var(--line);border-radius:8px;padding:9px;background:#fff}.signal-card img{display:block;width:100%}.signal-card .label{font-size:12px;color:var(--muted);padding:2px 6px 7px}.data-table{border-collapse:collapse;width:100%;font-size:12.5px;margin:14px 0 22px}.data-table th{background:#edf2f7;text-align:left;padding:9px 8px;border-bottom:2px solid #cbd5e1;white-space:nowrap}.data-table td{padding:8px;border-bottom:1px solid #e4e9ef;vertical-align:top}.data-table tr:hover td{background:#f8fafc}.scroll{overflow-x:auto}.note{padding:15px 18px;border-radius:7px;background:var(--pale);margin:15px 0}.note.bad{background:#fff0f1;border-left:4px solid var(--red)}.note.good{background:#edf8f3;border-left:4px solid var(--green)}.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f4f7fa;border:1px solid var(--line);padding:14px 16px;border-radius:6px;overflow-x:auto}.pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#e8f0fb;color:#1e559b;font-size:11px;font-weight:700;margin-right:5px}.muted{color:var(--muted)}a{color:#1e61ad}.foot{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}@media(max-width:850px){.page{margin:0}.content,header{padding-left:22px;padding-right:22px}.cards,.grid2{grid-template-columns:1fr}h1{font-size:34px}}
</style>
</head>
<body><main class="page">
<header><div class="eyebrow">Equity pairs research · allocation completion · {{ as_of }}</div><h1>Ten pairs. Unequal capital. Explicit risk budgets.</h1><p>A separate deployment-book report that replaces the 110-pair equal-weight baseline with an ex-ante selected, constrained, covariance-aware top-ten portfolio.</p></header>
<section class="content">
<div class="warning"><strong>Validation status:</strong> this is a retrospective analysis of a final year that has already been viewed. Selection and covariance inputs were cut at {{ tuning_end }}, but the {{ start }}–{{ end }} result is no longer an untouched lockbox. It must not be marketed as live evidence.</div>
<div class="verdict"><strong>Decision:</strong> use constrained equal-risk contribution as the documented allocation rule, not equal capital. It improved the same-ten-pair retrospective CAGR by {{ cagr_lift }} and reduced drawdown by {{ drawdown_lift }}, but the net result remained {{ primary_cagr }}. The allocation machinery is complete; the trading edge is not yet production-approved.</div>

<div class="cards">
<div class="card"><div class="k">Primary net CAGR</div><div class="v">{{ primary_cagr }}</div><div class="s">Already-seen final year · all modeled costs</div></div>
<div class="card"><div class="k">Equal-capital CAGR</div><div class="v">{{ equal_cagr }}</div><div class="s">Same signals and same ten pairs</div></div>
<div class="card"><div class="k">Primary Sharpe</div><div class="v">{{ primary_sharpe }}</div><div class="s">Risk-free rate fixed at zero</div></div>
<div class="card"><div class="k">Target beta</div><div class="v">{{ expected_beta }}</div><div class="s">Tuning estimate · cap ±0.05</div></div>
<div class="card"><div class="k">Effective pair count</div><div class="v">{{ effective_pairs }}</div><div class="s">1 / weight HHI · nominal count 10</div></div>
<div class="card"><div class="k">Final-year entries</div><div class="v">{{ final_entries }}</div><div class="s">{{ entries_per_day }} per session · {{ entry_day_fraction }} of days</div></div>
</div>

<h2>What changed</h2>
<p class="lead">The book no longer sends 1/110 to every sleeve—or 10% to every chosen sleeve. It first selects ten candidates with hard diversification constraints, then estimates a shrunk covariance matrix from tuning-only sleeve returns and solves for risk contributions under capital, sector, name, and beta limits.</p>
<div class="grid2">
<div class="note"><span class="pill">Selection</span><strong>Exactly 10 pairs</strong><br>Maximum two per sector, no repeated ticker, at least six sectors, at least six tuning trades, no selected return-correlation edge above 0.60, and pre-allocation beta within ±0.025.</div>
<div class="note"><span class="pill">Allocation</span><strong>5%–15% per pair</strong><br>Sector ≤25%, gross-normalized underlying name ≤10%, tuning market beta within ±0.05, monthly return-to-target with a 1 pp no-trade band and a 15% two-way capital-turnover limit.</div>
</div>
<div class="formula">minimize Σᵢ ( RCᵢ(w, Σ̂LW) − 10% )²<br>subject to Σᵢwᵢ=1; 5%≤wᵢ≤15%; sector≤25%; name≤10%; |w′β̂M|≤0.05</div>
<p>The risk targets are equal; the <em>capital weights are not</em>. Equal risk is a stabilizing objective, while unequal dollars compensate for different sleeve volatility and covariance. An empirical-Bayes return gate was also run. Its estimated cross-sectional signal variance was zero, so every return tilt was switched off instead of fitting noise.</p>
<img class="chart" src="charts/weighted_allocation.png" alt="Non-equal pair weights and risk contributions">

<h2>The frozen target book</h2>
<p>The book spans {{ sector_count }} sectors and 20 unique stocks. Financials hit the 25% sector cap; no pair hits the 5% or 15% capital boundary, and the largest conservative underlying budget is {{ max_target_name }}.</p>
<div class="scroll">{{ weights_table }}</div>
<img class="chart" src="charts/weighted_constraints.png" alt="Sector name and beta constraints">

<h2>Net result: better allocation, still no approved edge</h2>
<p>All methods below use the same ten pairs, the same SMA20/1.75/0.50/20-session signals, exact pre/post underlying-dollar turnover, 5 bp per dollar traded, 30 bp annual short borrow, monthly target rebalancing, and terminal liquidation. The equal-capital row is a comparator, not the selected method.</p>
<img class="chart" src="charts/weighted_equity_comparison.png" alt="Weighted and equal capital equity curves">
<div class="scroll">{{ performance_table }}</div>
<div class="note bad"><strong>Do not cherry-pick minimum variance.</strong> It happened to lose less in the already-viewed final year, but choosing it now would use the evaluation period to select the allocator. ERC remains the predeclared primary rule; the next untouched walk-forward decides whether any allocator survives.</div>

<h2>Market shock and cross-pair dependence</h2>
<div class="cards">
<div class="card"><div class="k">Realized market beta</div><div class="v">{{ realized_beta }}</div><div class="s">Final-year regression</div></div>
<div class="card"><div class="k">Market correlation</div><div class="v">{{ market_corr }}</div><div class="s">R² {{ market_r2 }}</div></div>
<div class="card"><div class="k">Active pair |correlation|</div><div class="v">{{ final_active_corr }}</div><div class="s">Final year · max {{ final_active_max }}</div></div>
</div>
<p>The book was close to market-neutral in aggregate, but “low beta” is not “shock-proof.” On {{ severe_days }} equal-weight S&amp;P proxy day(s) at or below −2%, the portfolio averaged {{ severe_mean }}. In the proxy’s bottom decile ({{ bottom_days }} sessions), it averaged {{ bottom_mean }} and compounded {{ bottom_cumulative }}. Those samples are descriptive, not a tail guarantee.</p>
<p>Cross-pair dependence stayed low: mean absolute active-overlap correlation rose from {{ tuning_active_corr }} in tuning to {{ final_active_corr }} in the final year. The no-shared-ticker rule prevents direct name reuse, while the sector cap limits common-industry clustering.</p>
<img class="chart" src="charts/weighted_pair_correlations.png" alt="Active overlap pair correlations">

<h2>Trading frequency and implementation cost</h2>
<p>The top-ten book generated {{ final_entries }} entries over {{ sessions }} sessions: {{ entries_per_day }} new pair positions per day on average, with at least one entry on {{ entry_days }} sessions. Each entry normally creates two stock orders and each exit another two; the reported entry count is therefore not an order count.</p>
<img class="chart" src="charts/weighted_trade_activity.png" alt="Trade entries per selected pair">
<div class="grid2"><div class="note"><strong>Base implementation load</strong><br>{{ annual_turnover }}× annualized underlying-leg turnover, {{ transaction_drag }} transaction-cost drag, and {{ borrow_drag }} borrow drag.</div><div class="note"><strong>Outer-book accounting</strong><br>Costs are rebuilt from the absolute change in pre/post dollar notionals for both legs. A signal entry or exit that lands on a monthly capital rebalance is charged once.</div></div>
<div class="scroll">{{ cost_table }}</div>

<h2>Signals for every selected pair</h2>
<p>These charts show the actual lagged SMA20 z-score decisions in the already-seen final year. Triangles are entries, crosses are exits, and the lower panel is the pair sleeve’s net equity before portfolio weighting.</p>
<div class="grid2">{% for item in signal_charts %}<div class="signal-card"><img src="{{ item.src }}" alt="{{ item.pair }} signal chart"><div class="label">{{ item.pair }}</div></div>{% endfor %}</div>

<h2>What is fixed, and what is still missing</h2>
<div class="scroll">{{ controls_table }}</div>
<div class="note good"><strong>Completed here:</strong> top-ten-only membership, unequal capital, covariance shrinkage, risk contribution diagnostics, hard sector/name/beta limits, a no-trade band, capped monthly capital turnover, exact leg-cost accounting, terminal liquidation, exposure reconstruction, correlations, and reproducible artifacts.</div>
<div class="note bad"><strong>Production gate:</strong> only 2/10 selected pairs passed the original strict multiple-testing cointegration screen; the remaining eight are deterministic fallbacks. Require a new point-in-time universe, rolling break/cointegration kill switches, bid/ask and market-impact data, borrow availability, and a genuinely untouched nested walk-forward before allocating real capital.</div>

<h2>Method references</h2>
<ul>
<li><a href="https://www.ledoit.net/Well-conditioned2004.pdf">Ledoit &amp; Wolf (2004)</a> for well-conditioned covariance estimation.</li>
<li><a href="https://www.thierry-roncalli.com/download/erc.pdf">Maillard, Roncalli &amp; Teïletche</a> for equal-risk-contribution portfolios.</li>
<li><a href="https://onlinelibrary.wiley.com/doi/10.1111/1540-6261.00580">Jagannathan &amp; Ma</a> on how constraints can stabilize covariance optimization.</li>
<li><a href="https://stanford.edu/~boyd/papers/portfolio.html">Lobo, Fazel &amp; Boyd</a> for portfolio optimization with transaction costs.</li>
<li><a href="https://escholarship.org/uc/item/4w1110bb">Bailey et al.</a> for backtest-overfitting controls.</li>
</ul>
<p>See the <a href="condensed_report.html">earlier condensed research report</a> for the 360-rule signal grid, all-110 baseline, FRED regimes, and original limitations. This report does not overwrite it.</p>
<div class="foot">Generated {{ generated_at }} · Inputs: saved Yahoo adjusted-close backtest artifacts and saved equal-weight S&amp;P proxy · Allocation estimates use observations through {{ tuning_end }} only · Research, not investment advice.</div>
</section></main></body></html>"""


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
    charts = output / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(
        processed / "grid_selected_pair_returns.csv",
        index_col="date",
        parse_dates=True,
    ).sort_index()
    signals = pd.read_csv(
        processed / "grid_selected_pair_signals.csv.gz",
        parse_dates=["date"],
    )
    trades = pd.read_csv(
        processed / "grid_selected_trade_ledger.csv",
        parse_dates=["entry_date", "exit_date"],
    )
    metadata = pd.read_csv(processed / "formation_selected_top10_by_sector.csv")
    market = pd.read_csv(
        processed / "followup_market_proxy_returns.csv",
        index_col="date",
        parse_dates=True,
    ).iloc[:, 0]
    grid_metadata = json.loads(
        (processed / "grid_search_metadata.json").read_text(encoding="utf-8")
    )
    start = pd.Timestamp(grid_metadata["lockbox_start"])
    end = returns.index.max()
    tuning = returns.loc[returns.index < start]
    retrospective = returns.loc[returns.index >= start]

    selection_spec = PairSelectionSpec(
        pair_count=10,
        maximum_pairs_per_sector=2,
        maximum_pairs_per_ticker=1,
        minimum_sector_count=6,
        minimum_trades=6,
    )
    features = build_pair_selection_features(
        tuning,
        metadata,
        trades=trades,
        spec=selection_spec,
    )
    pair_betas = _pair_market_betas(tuning, market)
    selected = select_constrained_pairs_milp(
        features,
        selection_spec,
        pair_betas=pair_betas,
        equal_weight_beta_bound=0.025,
        pair_correlations=tuning.corr(),
        maximum_pair_correlation=0.60,
    )
    pairs = selected["pair"].astype(str).tolist()
    selected_metadata = (
        metadata.drop_duplicates("pair").set_index("pair").loc[pairs].reset_index()
    )
    name_incidence = build_gross_name_incidence(selected_metadata)
    forecasts = empirical_bayes_monthly_forecasts(tuning[pairs])
    risk_budgets = alpha_tilted_risk_budgets_from_forecasts(forecasts)
    sector_caps = {
        str(sector): 0.25 for sector in selected_metadata["sector"].unique()
    }
    common_allocation = {
        "expected_returns": forecasts["capped_annual_return"],
        "pair_betas": pair_betas[pairs],
        "target_risk_budgets": risk_budgets,
        "pair_metadata": selected_metadata,
        "lower_bounds": 0.05,
        "upper_bounds": 0.15,
        "sector_caps": sector_caps,
        "name_incidence_caps": 0.10,
        "name_incidence": name_incidence,
        "linear_exposures": pd.DataFrame({"market_beta": pair_betas[pairs]}),
        "linear_exposure_bounds": {"market_beta": 0.05},
        "mean_shrinkage": 0.0,
    }
    allocations = build_allocation_candidates(tuning[pairs], **common_allocation)
    primary = allocations["equal_risk_contribution"]
    selected_signals = signals.loc[signals["pair"].isin(pairs)].copy()
    backtests = {
        method: backtest_weighted_pair_book(
            selected_signals,
            allocation.weights,
            start=start,
            end=end,
            transaction_cost_bps=5.0,
            rebalance_frequency="monthly",
            no_trade_band=0.01,
            maximum_rebalance_turnover=0.15,
            charge_terminal_liquidation=True,
        )
        for method, allocation in allocations.items()
    }
    equal_weights = pd.Series(0.10, index=pairs, name="weight")
    backtests["equal_capital"] = backtest_weighted_pair_book(
        selected_signals,
        equal_weights,
        start=start,
        end=end,
        transaction_cost_bps=5.0,
        rebalance_frequency="monthly",
        no_trade_band=0.01,
        maximum_rebalance_turnover=0.15,
        charge_terminal_liquidation=True,
    )
    primary_backtest = backtests["equal_risk_contribution"]
    equal_backtest = backtests["equal_capital"]

    # Per-pair trading and retrospective performance.
    trade_rows: list[dict[str, Any]] = []
    tuning_years = len(tuning) / 252.0
    for pair in pairs:
        pair_trades = trades.loc[trades["pair"].eq(pair)]
        tuning_entries = int(
            pair_trades["entry_date"].between(tuning.index.min(), start, inclusive="left").sum()
        )
        final_entries = int(pair_trades["entry_date"].ge(start).sum())
        final_pair_trades = pair_trades.loc[pair_trades["entry_date"].ge(start)]
        pair_performance = performance_metrics(retrospective[pair])
        trade_rows.append(
            {
                "pair": pair,
                "tuning_entries": tuning_entries,
                "annualized_tuning_entries": tuning_entries / tuning_years,
                "retrospective_entries": final_entries,
                "retrospective_win_rate": float((final_pair_trades["net_return"] > 0).mean())
                if len(final_pair_trades)
                else math.nan,
                "retrospective_average_holding_days": float(final_pair_trades["trading_days"].mean())
                if len(final_pair_trades)
                else math.nan,
                "retrospective_cagr": pair_performance["cagr"],
                "retrospective_sharpe": pair_performance["sharpe"],
                "retrospective_max_drawdown": pair_performance["max_drawdown"],
            }
        )
    trade_summary = pd.DataFrame(trade_rows)

    # Dependence, exposure, beta, and shock diagnostics.
    positions = selected_signals.pivot(index="date", columns="pair", values="position").reindex(
        columns=pairs
    )
    tuning_unconditional = tuning[pairs].corr()
    final_unconditional = retrospective[pairs].corr()
    tuning_active_matrix, tuning_active_detail, tuning_active_summary = _active_correlation(
        tuning[pairs], positions.reindex(tuning.index)
    )
    final_active_matrix, final_active_detail, final_active_summary = _active_correlation(
        retrospective[pairs], positions.reindex(retrospective.index)
    )
    tuning_unconditional_summary = _unconditional_correlation_summary(tuning_unconditional)
    final_unconditional_summary = _unconditional_correlation_summary(final_unconditional)
    name_exposure, exposure = _name_exposures(
        selected_signals,
        primary_backtest.deployed_weights,
        selected_metadata,
    )
    market_diagnostics = regression_diagnostics(
        primary_backtest.daily["portfolio_return"],
        market.reindex(primary_backtest.daily.index),
    )
    aligned = pd.concat(
        [
            primary_backtest.daily["portfolio_return"].rename("portfolio"),
            market.reindex(primary_backtest.daily.index).rename("market"),
        ],
        axis=1,
    ).dropna()
    severe = aligned.loc[aligned["market"].le(-0.02)]
    bottom_threshold = float(aligned["market"].quantile(0.10))
    bottom = aligned.loc[aligned["market"].le(bottom_threshold)]

    # Method comparison uses the already-viewed final year only as a diagnostic.
    method_rows: list[dict[str, Any]] = []
    for method, allocation in allocations.items():
        metrics = backtests[method].metrics
        method_rows.append(
            {
                "method": method,
                "role": "primary" if method == "equal_risk_contribution" else "diagnostic only",
                "effective_number_of_pairs": allocation.diagnostics["effective_number_of_pairs"],
                "maximum_weight": allocation.diagnostics["maximum_weight"],
                "expected_market_beta": allocation.diagnostics["expected_market_beta"],
                "tuning_expected_volatility": allocation.diagnostics["expected_annual_volatility"],
                **metrics,
            }
        )
    equal_detail, equal_diagnostics = allocation_diagnostics(
        equal_weights,
        primary.covariance,
        pair_betas=pair_betas[pairs],
    )
    method_rows.append(
        {
            "method": "equal_capital",
            "role": "comparator",
            "effective_number_of_pairs": equal_diagnostics["effective_number_of_pairs"],
            "maximum_weight": equal_diagnostics["maximum_weight"],
            "expected_market_beta": equal_diagnostics["expected_market_beta"],
            "tuning_expected_volatility": equal_diagnostics["expected_annual_volatility"],
            **equal_backtest.metrics,
        }
    )
    method_comparison = pd.DataFrame(method_rows)

    # Cost sensitivity keeps signals and capital mechanics fixed.
    sensitivity_rows: list[dict[str, Any]] = []
    for cost_bps in (0.0, 2.5, 5.0, 10.0, 15.0):
        for label, weights in (
            ("Constrained ERC", primary.weights),
            ("Equal capital", equal_weights),
        ):
            result = backtest_weighted_pair_book(
                selected_signals,
                weights,
                start=start,
                end=end,
                transaction_cost_bps=cost_bps,
                rebalance_frequency="monthly",
                no_trade_band=0.01,
                maximum_rebalance_turnover=0.15,
                charge_terminal_liquidation=True,
            )
            sensitivity_rows.append(
                {
                    "portfolio": label,
                    "transaction_cost_bps": cost_bps,
                    "cagr": result.metrics["cagr"],
                    "sharpe": result.metrics["sharpe"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "transaction_cost_drag_bps": result.metrics["transaction_cost_drag_bps"],
                    "annualized_leg_turnover": result.metrics["annualized_leg_turnover"],
                }
            )
    cost_sensitivity = pd.DataFrame(sensitivity_rows)

    # Allocation and constraint tables.
    weights_detail = primary.weight_diagnostics.reset_index(drop=True).merge(
        selected[
            [
                "selection_rank",
                "pair",
                "sector",
                "selection_score",
                "tuning_cagr",
                "tuning_sharpe",
                "tuning_trades",
                "selection_status",
                "formation_pair_pvalue",
            ]
        ],
        on="pair",
        how="left",
    ).merge(trade_summary, on="pair", how="left")
    weights_detail = weights_detail.sort_values("selection_rank")
    sector_by_pair = selected_metadata.set_index("pair")["sector"].reindex(pairs)
    sector_weights = primary.weights.groupby(sector_by_pair).sum().sort_values(ascending=False)
    target_name_weights = name_incidence.mul(primary.weights, axis=0).sum().sort_values(ascending=False)

    # Persist the full audit trail.
    selected.to_csv(processed / "weighted_selected_top10.csv", index=False)
    features.to_csv(processed / "weighted_selection_features.csv", index=False)
    forecasts.rename_axis("pair").reset_index().to_csv(
        processed / "weighted_return_forecasts.csv", index=False
    )
    weights_detail.to_csv(processed / "weighted_primary_weights.csv", index=False)
    primary.constraint_diagnostics.to_csv(
        processed / "weighted_constraint_diagnostics.csv", index=False
    )
    method_comparison.to_csv(processed / "weighted_method_comparison.csv", index=False)
    primary_backtest.daily.to_csv(processed / "weighted_portfolio_daily.csv")
    primary_backtest.deployed_weights.to_csv(
        processed / "weighted_deployed_pair_weights.csv.gz", compression="gzip"
    )
    primary_backtest.realized_weights.to_csv(
        processed / "weighted_realized_pair_weights.csv.gz", compression="gzip"
    )
    primary_backtest.rebalance_log.to_csv(
        processed / "weighted_rebalance_log.csv", index=False
    )
    name_exposure.to_csv(processed / "weighted_name_exposure_daily.csv.gz", compression="gzip")
    exposure.to_csv(processed / "weighted_exposure_daily.csv")
    trade_summary.to_csv(processed / "weighted_trade_summary.csv", index=False)
    tuning_unconditional.to_csv(processed / "weighted_tuning_pair_correlations.csv")
    final_unconditional.to_csv(processed / "weighted_retrospective_pair_correlations.csv")
    tuning_active_matrix.to_csv(processed / "weighted_tuning_active_correlations.csv")
    final_active_matrix.to_csv(processed / "weighted_retrospective_active_correlations.csv")
    pd.concat(
        [
            tuning_active_detail.assign(window="tuning"),
            final_active_detail.assign(window="retrospective_final_year"),
        ],
        ignore_index=True,
    ).to_csv(processed / "weighted_active_correlation_detail.csv", index=False)
    cost_sensitivity.to_csv(processed / "weighted_cost_sensitivity.csv", index=False)

    metadata_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tuning_start": str(tuning.index.min().date()),
        "tuning_end_inclusive": str(tuning.index.max().date()),
        "retrospective_start": str(start.date()),
        "retrospective_end": str(end.date()),
        "validation_status": "spent; retrospective only",
        "selected_pair_count": len(pairs),
        "selected_pairs": pairs,
        "selection": {
            "score_weights": {
                "sharpe": 0.30,
                "cagr": 0.20,
                "worst_half_sharpe": 0.20,
                "positive_month_fraction": 0.15,
                "drawdown": 0.15,
            },
            "maximum_pairs_per_sector": 2,
            "maximum_pairs_per_ticker": 1,
            "minimum_sectors": 6,
            "minimum_tuning_trades": 6,
            "maximum_absolute_pair_correlation": 0.60,
            "equal_weight_preallocation_beta_bound": 0.025,
        },
        "allocation": {
            "primary_method": "equal_risk_contribution",
            "covariance_estimator": "Ledoit-Wolf",
            "covariance_shrinkage": primary.diagnostics["covariance_shrinkage"],
            "return_forecast": "empirical Bayes monthly mean; tilt disabled when posterior signal is zero",
            "minimum_pair_weight": 0.05,
            "maximum_pair_weight": 0.15,
            "maximum_sector_weight": 0.25,
            "maximum_gross_normalized_name_weight": 0.10,
            "market_beta_bound": 0.05,
            "rebalance_frequency": "monthly to frozen target",
            "no_trade_band": 0.01,
            "maximum_two_way_capital_rebalance_turnover": 0.15,
        },
        "costs": {
            "transaction_cost_bps_per_dollar_traded": 5.0,
            "annual_short_borrow_bps": grid_metadata["annual_short_borrow_bps"],
            "transaction_cost_model": "exact pre/post underlying-dollar notionals",
            "terminal_liquidation": True,
        },
        "primary_diagnostics": primary.diagnostics,
        "primary_retrospective_metrics": primary_backtest.metrics,
        "equal_capital_retrospective_metrics": equal_backtest.metrics,
        "market_diagnostics": market_diagnostics,
        "correlation_summary": {
            "tuning_unconditional": tuning_unconditional_summary,
            "retrospective_unconditional": final_unconditional_summary,
            "tuning_active": tuning_active_summary,
            "retrospective_active": final_active_summary,
        },
    }
    (processed / "weighted_portfolio_metadata.json").write_text(
        json.dumps(metadata_json, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )

    # Figures.
    _plot_allocation(weights_detail, charts / "weighted_allocation.png")
    _plot_equity(backtests, charts / "weighted_equity_comparison.png")
    _plot_constraints(
        sector_weights,
        target_name_weights,
        float(primary.diagnostics["expected_market_beta"]),
        charts / "weighted_constraints.png",
    )
    _plot_correlations(
        tuning_active_matrix,
        final_active_matrix,
        charts / "weighted_pair_correlations.png",
    )
    _plot_trade_activity(trade_summary, charts / "weighted_trade_activity.png")
    signal_charts: list[dict[str, str]] = []
    for pair in pairs:
        filename = "weighted_signal_" + pair.lower().replace("__", "_") + ".png"
        frame = selected_signals.loc[
            selected_signals["pair"].eq(pair) & selected_signals["date"].ge(start)
        ]
        _plot_pair_signal(frame, pair, charts / filename)
        signal_charts.append({"pair": pair.replace("__", "/"), "src": f"charts/{filename}"})

    # Presentation tables.
    weights_display = weights_detail[
        [
            "selection_rank",
            "pair",
            "sector",
            "selection_status",
            "selection_score",
            "weight",
            "risk_contribution_fraction",
            "pair_beta",
            "tuning_sharpe",
            "retrospective_entries",
            "retrospective_cagr",
        ]
    ].copy()
    weights_display["pair"] = weights_display["pair"].str.replace("__", "/", regex=False)
    weights_display["selection_score"] = weights_display["selection_score"].map(lambda value: _num(value, 3))
    for column in ["weight", "risk_contribution_fraction", "retrospective_cagr"]:
        weights_display[column] = weights_display[column].map(_pct)
    for column in ["pair_beta", "tuning_sharpe"]:
        weights_display[column] = weights_display[column].map(lambda value: _num(value, 3))
    weights_display.columns = [
        "Rank",
        "Pair",
        "Sector",
        "Formation gate",
        "Tuning score",
        "Capital",
        "Risk contribution",
        "Tuning β",
        "Tuning Sharpe",
        "Final-year entries",
        "Final-year CAGR",
    ]

    method_names = {
        "equal_risk_contribution": "Constrained ERC",
        "inverse_volatility": "Constrained inverse volatility",
        "minimum_variance": "Constrained minimum variance",
        "maximum_diversification": "Constrained maximum diversification",
        "shrunk_tangency": "Zero-alpha regularized tangency",
        "equal_capital": "Equal capital",
    }
    performance_display = method_comparison[
        [
            "method",
            "role",
            "maximum_weight",
            "effective_number_of_pairs",
            "expected_market_beta",
            "cagr",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
        ]
    ].copy()
    performance_display["method"] = performance_display["method"].map(method_names)
    for column in ["maximum_weight", "cagr", "annual_volatility", "max_drawdown"]:
        performance_display[column] = performance_display[column].map(_pct)
    for column in ["effective_number_of_pairs", "expected_market_beta", "sharpe"]:
        performance_display[column] = performance_display[column].map(lambda value: _num(value, 3))
    performance_display.columns = [
        "Allocator",
        "Role",
        "Max pair",
        "Effective N",
        "Tuning β",
        "Final CAGR",
        "Final vol",
        "Final Sharpe",
        "Final drawdown",
    ]

    cost_display = cost_sensitivity.copy()
    cost_display["transaction_cost_bps"] = cost_display["transaction_cost_bps"].map(lambda value: f"{value:.1f}")
    cost_display["cagr"] = cost_display["cagr"].map(_pct)
    cost_display["sharpe"] = cost_display["sharpe"].map(lambda value: _num(value, 3))
    cost_display["max_drawdown"] = cost_display["max_drawdown"].map(_pct)
    cost_display["transaction_cost_drag_bps"] = cost_display["transaction_cost_drag_bps"].map(lambda value: f"{value:.1f}")
    cost_display["annualized_leg_turnover"] = cost_display["annualized_leg_turnover"].map(lambda value: f"{value:.1f}×")
    cost_display.columns = [
        "Portfolio",
        "One-way cost (bp)",
        "CAGR",
        "Sharpe",
        "Max drawdown",
        "Modeled cost drag (bp)",
        "Annual turnover",
    ]

    controls_display = pd.DataFrame(
        [
            ["Membership", "Fixed 10 chosen on tuning data only", "Quarterly rerun; never use evaluation returns"],
            ["Capital objective", "Constrained ERC on Ledoit-Wolf covariance", "Frozen for this one-year test"],
            ["Pair capital", "5%–15%; observed 6.0%–13.1%", "No equal-capital fallback"],
            ["Concentration", "Sector ≤25%; name ≤10%; no ticker reused", "Financials cap binds at 25%"],
            ["Market exposure", "Tuning beta constrained to ±0.05", "Daily breach alert; realized beta is diagnostic"],
            ["Rebalance", "Monthly; 1 pp no-trade band; 15% two-way cap", "Exact leg-notional turnover"],
            ["Signals", "SMA20; entry 1.75; exit 0.50; max hold 20", "Selected by earlier 360-rule tuning grid"],
            ["Costs", "5 bp/dollar + 30 bp/year borrow + terminal close", "Still lacks spread, impact, locate, financing"],
            ["Kill switches", "Not implemented", "Rolling cointegration, beta drift, portfolio DD required"],
        ],
        columns=["Control", "Implemented rule", "Operating note"],
    )

    final_trades = trades.loc[trades["pair"].isin(pairs) & trades["entry_date"].ge(start)]
    entry_days = int(final_trades["entry_date"].nunique())
    final_entries = int(len(final_trades))
    sessions = int(len(primary_backtest.daily))
    cagr_lift = primary_backtest.metrics["cagr"] - equal_backtest.metrics["cagr"]
    drawdown_lift = primary_backtest.metrics["max_drawdown"] - equal_backtest.metrics["max_drawdown"]
    report = Template(REPORT_TEMPLATE).render(
        as_of=str(end.date()),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        tuning_end=str(tuning.index.max().date()),
        start=str(start.date()),
        end=str(end.date()),
        primary_cagr=_pct(primary_backtest.metrics["cagr"]),
        equal_cagr=_pct(equal_backtest.metrics["cagr"]),
        cagr_lift=_pp(cagr_lift),
        drawdown_lift=_pp(drawdown_lift),
        primary_sharpe=_num(primary_backtest.metrics["sharpe"], 3),
        expected_beta=_num(primary.diagnostics["expected_market_beta"], 3),
        effective_pairs=_num(primary.diagnostics["effective_number_of_pairs"], 2),
        final_entries=f"{final_entries:,}",
        entries_per_day=_num(final_entries / sessions, 2),
        entry_day_fraction=_pct(entry_days / sessions, 1),
        sector_count=int(selected["sector"].nunique()),
        max_target_name=_pct(target_name_weights.max()),
        realized_beta=_num(market_diagnostics["market_beta"], 3),
        market_corr=_num(market_diagnostics["market_correlation"], 3),
        market_r2=_pct(market_diagnostics["r_squared"], 1),
        tuning_active_corr=_pct(tuning_active_summary["mean_absolute"], 1),
        final_active_corr=_pct(final_active_summary["mean_absolute"], 1),
        final_active_max=_pct(final_active_summary["maximum_absolute"], 1),
        severe_days=len(severe),
        severe_mean=_pct(severe["portfolio"].mean()) if len(severe) else "—",
        bottom_days=len(bottom),
        bottom_mean=_pct(bottom["portfolio"].mean()),
        bottom_cumulative=_pct((1.0 + bottom["portfolio"]).prod() - 1.0),
        sessions=sessions,
        entry_days=entry_days,
        annual_turnover=_num(primary_backtest.metrics["annualized_leg_turnover"], 1),
        transaction_drag=f"{primary_backtest.metrics['transaction_cost_drag_bps']:.1f} bp",
        borrow_drag=f"{primary_backtest.metrics['borrow_cost_drag_bps']:.1f} bp",
        weights_table=_table(weights_display),
        performance_table=_table(performance_display),
        cost_table=_table(cost_display),
        controls_table=_table(controls_display),
        signal_charts=signal_charts,
    )
    report_path = output / "weighted_portfolio_report.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print(json.dumps(metadata_json, indent=2, default=float))


if __name__ == "__main__":
    main()
