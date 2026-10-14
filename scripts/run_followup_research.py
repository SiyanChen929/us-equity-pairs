#!/usr/bin/env python3
"""Build the leakage-controlled parameter study and condensed risk report."""

from __future__ import annotations

import argparse
import html
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

from equity_pairs.backtest import build_portfolios
from equity_pairs.config import load_config
from equity_pairs.diagnostics import (
    equal_weight_market_proxy,
    overlap_correlation_diagnostics,
    pair_correlation_diagnostics,
    portfolio_exposure_diagnostics,
    regression_diagnostics,
    shock_diagnostics,
    ticker_overlap_table,
)
from equity_pairs.grid_search import GridSearchSpec, run_parameter_grid


INK = "#152033"
MUTED = "#6b778c"
BLUE = "#2d67c4"
TEAL = "#138a82"
GREEN = "#23855b"
ORANGE = "#d9782d"
RED = "#be4a52"
PURPLE = "#7657b5"
PALE = "#eef3f8"

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


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _sector_balanced_returns(
    pair_returns: pd.DataFrame,
    selected_pairs: pd.DataFrame,
) -> pd.Series:
    pair_sector = selected_pairs.drop_duplicates("pair").set_index("pair")["sector"].to_dict()
    sector_series: dict[str, pd.Series] = {}
    for sector in sorted(set(pair_sector.values())):
        names = [name for name in pair_returns if pair_sector.get(name) == sector]
        if names:
            sector_series[str(sector)] = pair_returns[names].fillna(0.0).mean(axis=1)
    result = pd.DataFrame(sector_series).mean(axis=1)
    result.name = "sector_balanced"
    return result


def _percentage(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}%}" if math.isfinite(number) else "—"


def _number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _integer(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def _table(frame: pd.DataFrame, *, classes: str = "data-table") -> str:
    if frame.empty:
        return '<p class="muted">No rows available.</p>'
    return frame.to_html(index=False, border=0, classes=classes, escape=True)


def _plot_validation_equity(
    baseline_returns: pd.Series,
    selected_returns: pd.Series,
    lockbox_start: pd.Timestamp,
    output: Path,
) -> None:
    aligned = pd.concat(
        [baseline_returns.rename("Baseline"), selected_returns.rename("Tuning-selected")],
        axis=1,
    ).fillna(0.0)
    full_equity = (1.0 + aligned).cumprod()
    lockbox = aligned.loc[aligned.index >= lockbox_start]
    lockbox_equity = (1.0 + lockbox).cumprod()
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 8.2), sharex=False)
    for column, color in (("Baseline", MUTED), ("Tuning-selected", BLUE)):
        axes[0].plot(full_equity.index, full_equity[column], color=color, linewidth=2.0, label=column)
        axes[1].plot(lockbox_equity.index, lockbox_equity[column], color=color, linewidth=2.2, label=column)
    axes[0].axvline(lockbox_start, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[0].text(
        lockbox_start,
        axes[0].get_ylim()[1],
        " lockbox opens",
        color=ORANGE,
        va="top",
        fontsize=9,
    )
    axes[0].set_title("Full held-out path: tuning period followed by one-year lockbox", loc="left")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(1.0, color="#bbc5d1", linewidth=0.8)
    axes[1].set_title("Lockbox only: both curves reset to $1", loc="left")
    axes[1].set_ylabel("Growth of $1")
    axes[1].set_xlabel("Date")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        _style_axis(axis)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout(h_pad=2.2)
    _save_figure(figure, output)


def _plot_grid_frontier(
    frontier: pd.DataFrame,
    ranking: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for method, color in (("sma", BLUE), ("ewma", TEAL)):
        subset = frontier.loc[frontier["zscore_method"].eq(method)]
        axes[0].scatter(
            subset["median_trades_per_pair"],
            subset["sharpe"],
            s=24,
            alpha=0.44,
            color=color,
            edgecolors="none",
            label=method.upper(),
        )
    baseline = frontier.loc[frontier["is_baseline"].astype(bool)].iloc[0]
    winner = ranking.iloc[0]
    axes[0].scatter(
        [baseline["median_trades_per_pair"]],
        [baseline["sharpe"]],
        marker="*",
        s=180,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        label="Baseline",
        zorder=5,
    )
    axes[0].scatter(
        [winner["median_trades_per_pair"]],
        [winner["sharpe"]],
        marker="D",
        s=75,
        color=RED,
        edgecolor="white",
        linewidth=0.7,
        label="Tuning winner",
        zorder=6,
    )
    axes[0].axhline(0, color="#aeb8c5", linewidth=0.8)
    axes[0].set_title("360-rule tuning frontier", loc="left")
    axes[0].set_xlabel("Median trades per pair · first two years")
    axes[0].set_ylabel("Net tuning Sharpe")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    grouped = (
        frontier.groupby(["zscore_method", "zscore_lookback"])["sharpe"]
        .agg(["median", "max"])
        .reset_index()
    )
    for method, color in (("sma", BLUE), ("ewma", TEAL)):
        subset = grouped.loc[grouped["zscore_method"].eq(method)]
        axes[1].plot(subset["zscore_lookback"], subset["median"], marker="o", color=color, label=f"{method.upper()} median")
        axes[1].plot(subset["zscore_lookback"], subset["max"], marker=".", linestyle="--", color=color, alpha=0.7, label=f"{method.upper()} best")
    axes[1].axhline(0, color="#aeb8c5", linewidth=0.8)
    axes[1].set_title("Memory length mattered most", loc="left")
    axes[1].set_xlabel("Lookback / EWMA span (sessions)")
    axes[1].set_ylabel("Net tuning Sharpe")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(w_pad=2.1)
    _save_figure(figure, output)


def _plot_trade_frequency(
    baseline_pair_metrics: pd.DataFrame,
    selected_pair_metrics: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    maximum = int(
        max(
            baseline_pair_metrics["trade_count"].max(),
            selected_pair_metrics["trade_count"].max(),
        )
    )
    bins = np.arange(0, maximum + 3, 3)
    axes[0].hist(
        baseline_pair_metrics["trade_count"],
        bins=bins,
        color=MUTED,
        alpha=0.58,
        label="Baseline",
    )
    axes[0].hist(
        selected_pair_metrics["trade_count"],
        bins=bins,
        color=BLUE,
        alpha=0.55,
        label="Tuning-selected",
    )
    axes[0].axvline(baseline_pair_metrics["trade_count"].median(), color=MUTED, linestyle="--")
    axes[0].axvline(selected_pair_metrics["trade_count"].median(), color=BLUE, linestyle="--")
    axes[0].set_title("Three-year trades per pair", loc="left")
    axes[0].set_xlabel("Trade entries")
    axes[0].set_ylabel("Pairs")
    axes[0].legend(frameon=False)
    exit_counts = baseline_trades["exit_reason"].value_counts().reindex(
        ["exit_mean_reversion", "exit_max_holding", "exit_stop", "end_of_test"],
        fill_value=0,
    )
    labels = ["Mean reversion", "Time stop", "Z-stop", "Open at end"]
    colors = [GREEN, ORANGE, RED, MUTED]
    axes[1].barh(labels[::-1], exit_counts.to_numpy()[::-1], color=colors[::-1])
    axes[1].set_title("Baseline exit reasons", loc="left")
    axes[1].set_xlabel("Trades")
    for label_index, count in enumerate(exit_counts.to_numpy()[::-1]):
        axes[1].text(count + 8, label_index, f"{int(count):,}", va="center", fontsize=9, color=INK)
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    _save_figure(figure, output)


def _plot_correlations(
    top10_matrix: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.35, 0.9]})
    image = axes[0].imshow(top10_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [label.replace("__", "/") for label in top10_matrix.columns]
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].set_title("Ex-post top-ten daily return correlations", loc="left")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = float(top10_matrix.iloc[row, column])
            axes[0].text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(value) > 0.55 else INK)
    figure.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    groups = overlap_summary.set_index("group")
    order = ["disjoint_underlyings", "shared_underlying"]
    names = ["Disjoint\nunderlyings", "Shared\nunderlying"]
    unconditional = [groups.loc[group, "mean_absolute_unconditional_correlation"] for group in order]
    active = [groups.loc[group, "mean_absolute_active_correlation"] for group in order]
    positions = np.arange(2)
    width = 0.34
    axes[1].bar(positions - width / 2, np.array(unconditional) * 100, width, color=MUTED, label="All days")
    axes[1].bar(positions + width / 2, np.array(active) * 100, width, color=RED, label="Both active")
    axes[1].set_xticks(positions, names)
    axes[1].set_ylabel("Mean absolute correlation (%)")
    axes[1].set_title("Name reuse creates hidden clusters", loc="left")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    figure.tight_layout(w_pad=2.3)
    _save_figure(figure, output)


def _plot_exposure_and_shocks(
    exposure: pd.DataFrame,
    shocks: pd.DataFrame,
    output: Path,
) -> None:
    exposure = exposure.set_index("date")
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), gridspec_kw={"height_ratios": [1.25, 1]})
    axes[0].plot(exposure.index, exposure["sleeve_gross_exposure"] * 100, color=BLUE, linewidth=1.5, label="Sleeve gross")
    axes[0].plot(exposure.index, exposure["netted_gross_exposure"] * 100, color=TEAL, linewidth=1.3, label="Gross after name netting")
    axes[0].plot(exposure.index, exposure["net_exposure"] * 100, color=ORANGE, linewidth=1.0, alpha=0.9, label="Net dollar")
    axes[0].axhline(0, color="#aeb8c5", linewidth=0.8)
    axes[0].set_ylabel("Portfolio exposure (%)")
    axes[0].set_title("Actual all-selected deployment: much of the book remains cash", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    shock_order = ["market_down_at_least_2%", "market_up_at_least_2%"]
    portfolios = ["all_formation_selected", "expost_top_performers"]
    labels = ["Market ≤ −2%", "Market ≥ +2%"]
    positions = np.arange(2)
    width = 0.34
    for offset, portfolio, color, label in (
        (-width / 2, portfolios[0], BLUE, "All 110"),
        (width / 2, portfolios[1], PURPLE, "Ex-post ten"),
    ):
        values = []
        for regime in shock_order:
            row = shocks.loc[shocks["regime"].eq(regime) & shocks["portfolio"].eq(portfolio)].iloc[0]
            values.append(float(row["mean_daily_return"]) * 100)
        axes[1].bar(positions + offset, values, width, color=color, label=label)
    axes[1].axhline(0, color="#aeb8c5", linewidth=0.8)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Mean strategy return on shock days (%)")
    axes[1].set_title("Low beta did not eliminate broad-down-day losses", loc="left")
    axes[1].legend(frameon=False)
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(h_pad=2.2)
    _save_figure(figure, output)


REPORT_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pairs strategy — risk & parameter review</title>
<style>
:root{--ink:#152033;--muted:#68758a;--blue:#2d67c4;--teal:#138a82;--green:#23855b;--orange:#d9782d;--red:#be4a52;--paper:#fff;--wash:#f5f8fb;--line:#dce3ec}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f6;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.58}
main{max-width:1240px;margin:28px auto;background:var(--paper);box-shadow:0 18px 55px rgba(36,52,73,.10);padding:56px 64px 80px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:var(--blue);font-size:11px;font-weight:800}.hero{display:grid;grid-template-columns:1.65fr .75fr;gap:38px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:30px}h1{font-size:43px;letter-spacing:-.04em;line-height:1.04;margin:10px 0 13px}.deck{font-size:18px;color:var(--muted);max-width:780px}.stamp{text-align:right;color:var(--muted);font-size:12px}.verdict{margin:28px 0 14px;padding:22px 24px;border-radius:11px;background:linear-gradient(135deg,#f0f6ff,#f5fbfa);border:1px solid #d9e7f3;font-size:16px}.verdict strong{color:var(--blue)}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;margin:20px 0 34px}.card{border:1px solid var(--line);background:var(--wash);border-radius:10px;padding:15px}.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}.card .v{font-size:23px;font-weight:780;margin-top:3px;letter-spacing:-.02em}.card .s{font-size:10.5px;color:var(--muted);margin-top:2px}
h2{font-size:26px;letter-spacing:-.025em;margin:50px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}h3{font-size:17px;margin:28px 0 9px}.lead{font-size:16px;color:#3f4e63}.note{border-left:4px solid var(--orange);background:#fff8ef;padding:13px 16px;margin:16px 0;color:#4c5666}.good{border-left-color:var(--green);background:#f1faf6}.bad{border-left-color:var(--red);background:#fff3f3}.muted{color:var(--muted)}.small{font-size:12px}.chart{width:100%;border:1px solid var(--line);border-radius:10px;margin:14px 0 24px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.data-table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 22px}.data-table th{background:#edf3f8;text-align:left;font-weight:750;color:#435269}.data-table th,.data-table td{padding:8px 9px;border-bottom:1px solid var(--line);vertical-align:top}.data-table tr:hover td{background:#f9fbfd}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:9px;margin:10px 0 24px}.table-wrap .data-table{margin:0}.rule td:first-child,.roadmap td:first-child{font-weight:750}.tag{display:inline-block;border-radius:999px;padding:3px 9px;background:#eaf2ff;color:var(--blue);font-size:11px;font-weight:750;margin-right:4px}.warning-tag{background:#fff0dd;color:#a65e1e}.foot{margin-top:55px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:12px}a{color:var(--blue)}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;background:#f0f3f7;border-radius:4px;padding:1px 4px}ul{padding-left:21px}li{margin:6px 0}.decision{display:grid;grid-template-columns:36px 1fr;gap:12px;margin:12px 0}.decision .n{height:28px;width:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:var(--blue);color:#fff;font-weight:800;font-size:12px}.decision p{margin:2px 0}
@media(max-width:900px){main{margin:0;padding:35px 24px}.hero{grid-template-columns:1fr}.stamp{text-align:left}.cards{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}h1{font-size:36px}}@media print{body{background:#fff}main{margin:0;box-shadow:none;max-width:none}.chart{break-inside:avoid}a{color:inherit;text-decoration:none}}
</style>
</head>
<body><main>
<section class="hero"><div><div class="eyebrow">Follow-up research note · risk, diversification & frequency</div><h1>The pairs book is lightly market-exposed, but not yet risk-controlled.</h1><p class="deck">A compact audit of the existing 110-pair daily strategy, a 360-rule SMA/EWMA grid, and the first direct tests of market shocks, cross-pair dependence, name overlap, and capital deployment.</p></div><div class="stamp">Run through {{ as_of }}<br>Generated {{ generated_at }}<br><a href="report.html">Open the original full report →</a></div></section>

<div class="verdict"><strong>Decision:</strong> keep this as a research prototype. The defensible all-selected book lost {{ baseline_cagr }} annualized after modeled costs. The grid winner improved the final one-year lockbox to {{ grid_lockbox_cagr }} with {{ grid_lockbox_sharpe }} Sharpe, but the absolute edge is modest, turnover more than doubled, 103/110 pairs are fallback selections, and portfolio-level exposure limits are absent.</div>

<div class="cards">
<div class="card"><div class="k">Baseline net CAGR</div><div class="v">{{ baseline_cagr }}</div><div class="s">All 110 · three years</div></div>
<div class="card"><div class="k">Baseline Sharpe</div><div class="v">{{ baseline_sharpe }}</div><div class="s">After 5 bp turnover cost</div></div>
<div class="card"><div class="k">Realized market beta</div><div class="v">{{ market_beta }}</div><div class="s">EW current-universe proxy</div></div>
<div class="card"><div class="k">Median trades / pair</div><div class="v">{{ baseline_trades }}</div><div class="s">{{ baseline_trades_per_year }} per pair-year</div></div>
<div class="card"><div class="k">Grid lockbox CAGR</div><div class="v">{{ grid_lockbox_cagr }}</div><div class="s">One year · rule frozen</div></div>
</div>

<h2>1 · What is controlled today?</h2>
<p class="lead">The code has meaningful pair-level entry and exit discipline, but almost no portfolio-level risk budget. The hedge ratio is a cointegration coefficient—not a market beta—and gross-normalizing each pair does not guarantee dollar, factor, or shock neutrality.</p>
<div class="table-wrap">{{ rules_table | safe }}</div>
<div class="note bad"><strong>Boundary cost caveat.</strong> {{ first_day_entries }} sleeves are already active on the first held-out date, so their launch entry cost sits before the reported return window. Another {{ end_open_trades }} positions finish open with no terminal liquidation cost. These effects flatter the cost boundary slightly and should be fixed before promotion.</div>

<h3>Portfolio distribution</h3>
<div class="table-wrap">{{ allocation_table | safe }}</div>
<p>The first two versions are identical in this run because each of 11 sectors contributes exactly 10 pairs. Inactive sleeves earn zero. The book therefore averaged only {{ mean_gross }} active sleeve gross, which is one reason annual volatility was just {{ baseline_vol }}; that low volatility is not pure hedge quality.</p>
<img class="chart" src="charts/condensed_exposure_shocks.png" alt="Portfolio exposure and shock-day performance">

<h2>2 · Did the pairs neutralize market shocks?</h2>
<p class="lead"><strong>Partially, not completely.</strong> The all-selected book's realized beta was {{ market_beta }} and its market correlation was {{ market_corr }}, so broad equity direction explained only {{ market_r2 }} of daily variance. Yet on {{ down_shock_days }} proxy-market days at or below −2%, the book averaged {{ down_shock_return }} and compounded {{ down_shock_cumulative }}.</p>
<div class="table-wrap">{{ market_table | safe }}</div>
<div class="note">SPY could not be refreshed from Yahoo during this follow-up, so the benchmark is the equal-weight daily return of the 503 downloaded current constituents. It is transparent and Yahoo-sourced, but survivorship-biased and not an official S&amp;P 500 index. Treat the beta as a diagnostic, not a certified hedge ratio.</div>

<h2>3 · Are pair sleeves diversified from one another?</h2>
<p class="lead">The average looks low only until name reuse is separated. Across all 5,995 pair-of-pair combinations, mean absolute correlation was {{ all_abs_corr }} on all days and {{ all_active_abs_corr }} when both sleeves were active. Pairs that shared an underlying had {{ shared_active_abs_corr }} mean absolute active correlation versus {{ disjoint_active_abs_corr }} for disjoint pairs.</p>
<img class="chart" src="charts/condensed_pair_correlations.png" alt="Pair correlation diagnostics">
<div class="two"><div><h3>All 110</h3><div class="table-wrap">{{ overlap_all_table | safe }}</div></div><div><h3>Ex-post winning ten</h3><div class="table-wrap">{{ overlap_top_table | safe }}</div></div></div>
<div class="note bad"><strong>Concentration is real.</strong> The 110 pairs use only {{ unique_names }} names; {{ repeated_names }} recur. BDX appears in 7 sleeves. In the ex-post ten, FERG appears in four pairs and reached {{ top_max_name }} absolute portfolio weight. A pair-count allocation is therefore not a name-risk allocation.</div>

<h2>4 · Is fewer than 20 trades in three years normal?</h2>
<p class="lead">Yes, for this rule family. The baseline produced {{ total_trades }} entries: median {{ baseline_trades }} per pair, range {{ trade_range }}, or roughly {{ baseline_trades_per_year }} per pair-year. A canonical daily 2σ distance strategy reported about two round trips per pair per six-month trading period—about 12 over three years—so the observed count is plausible. It is still too small for a stable pair-level Sharpe or win-rate estimate.</p>
<img class="chart" src="charts/condensed_trade_frequency.png" alt="Trade frequency and exit reasons">
<p>Lowering the trigger and shortening memory did increase activity. The tuning-selected rule generated a median {{ selected_full_trades }} entries per pair over the same three years and {{ selected_lockbox_trades }} in the one-year lockbox, versus {{ baseline_lockbox_trades }} for baseline. Annual turnover rose from {{ baseline_lockbox_turnover }} to {{ selected_lockbox_turnover }}.</p>

<h2>5 · SMA or exponentially smoothed z-score?</h2>
<p class="lead">The original strategy used a <strong>plain 60-session rolling arithmetic mean and sample standard deviation</strong> with 40 minimum observations. No exponential smoothing was used. This follow-up added an EWMA alternative with recursive <code>adjust=False</code>, span equal to the searched lookback, and unbiased exponentially weighted standard deviation.</p>
<p>The tuning winner remained SMA: <span class="tag">20-session SMA</span><span class="tag">entry 1.75σ</span><span class="tag">exit 0.50σ</span><span class="tag">20-session time stop</span>. The best EWMA rule ranked {{ best_ewma_rank }} of {{ eligible_rules }} frequency-eligible rules ({{ grid_trials }} tested) and had {{ best_ewma_sharpe }} tuning Sharpe. That is useful evidence against assuming “more adaptive” automatically means better.</p>

<h2>6 · What did the grid search actually establish?</h2>
<p class="lead">It tested {{ grid_trials }} predeclared daily combinations after the same 5 bp turnover cost and 30 bp annual borrow proxy. Pair membership and formation alpha/beta stayed frozen. The first two held-out years selected by net sector-balanced Sharpe; only then did code reveal the final one-year lockbox for the winner and baseline.</p>
<img class="chart" src="charts/condensed_grid_frontier.png" alt="Grid search tuning frontier">
<img class="chart" src="charts/condensed_validation_equity.png" alt="Tuning and lockbox equity comparison">
<h3>Baseline versus tuning winner</h3><div class="table-wrap">{{ grid_comparison_table | safe }}</div>
<h3>Top tuning rules</h3><div class="table-wrap">{{ grid_top_table | safe }}</div>
<div class="note good"><strong>Lockbox outcome:</strong> the faster rule moved net CAGR from {{ baseline_lockbox_cagr }} to {{ grid_lockbox_cagr }} and Sharpe from {{ baseline_lockbox_sharpe }} to {{ grid_lockbox_sharpe }}. This is directionally encouraging.</div>
<div class="note bad"><strong>What it did not establish:</strong> {{ grid_lockbox_cagr }} on one year is not a production edge, and the lockbox is now spent. A nested rolling walk-forward plus White-style data-snooping controls is still required. Do not relabel the tuning winner “optimal.”</div>

<h2>7 · Highest-value functions to add next</h2>
<div class="table-wrap">{{ roadmap_table | safe }}</div>
<div class="decision"><div class="n">1</div><p><strong>First gate:</strong> rebuild selection on point-in-time membership and allow fewer than ten pairs per sector. Forcing 103 fallback slots overwhelms all later parameter finesse.</p></div>
<div class="decision"><div class="n">2</div><p><strong>Second gate:</strong> add a constrained allocator with rolling market/factor beta, shared-name graph caps, gross/net/sector/name limits, volatility targeting, ADV/borrow constraints, and covariance shrinkage.</p></div>
<div class="decision"><div class="n">3</div><p><strong>Third gate:</strong> validate signals in purged rolling folds with realistic start/end liquidation, quote-based costs, structural-break exits, and a paper-trading reconciliation layer.</p></div>
<p class="note">A three-year intraday comparison was not run because yfinance's official documentation limits intraday history to the most recent 60 days. A serious hourly/minute study needs licensed quotes, bid/ask, latency, impact, fill logic, and historical borrow—not just a shorter <code>interval</code> argument.</p>

<h2>Essential performance snapshot</h2>
<div class="table-wrap">{{ performance_table | safe }}</div>
<p class="small muted">The +12.40% ex-post-ten CAGR is intentionally retained for completeness, but those ten were chosen after observing the same three-year returns. It is not an investable selection rule; only one of the ten passed every strict formation gate.</p>

<h2>Primary evidence & audit trail</h2>
<ul class="small">
<li><a href="https://www.nber.org/papers/w7032">Gatev, Goetzmann &amp; Rouwenhorst — Pairs Trading: Performance of a Relative-Value Arbitrage Rule</a></li>
<li><a href="https://doi.org/10.2307/1913236">Engle &amp; Granger — Co-integration and Error Correction</a></li>
<li><a href="https://doi.org/10.1016/j.physa.2010.01.045">Bertram — Analytic solutions for optimal statistical arbitrage trading</a></li>
<li><a href="https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf">Avellaneda &amp; Lee — Statistical arbitrage in the U.S. equities market</a></li>
<li><a href="https://www.ucc.ie/en/media/research/centreforinvestmentresearch/wp/wp1004high-frequency-equity-pairs-trading.pdf">Bowen, Hutchinson &amp; O'Sullivan — High-frequency equity pairs trading</a></li>
<li><a href="https://www.nber.org/papers/w14465">Khandani &amp; Lo — What Happened to the Quants in August 2007?</a></li>
<li><a href="https://doi.org/10.1111/1468-0262.00152">White — A Reality Check for Data Snooping</a></li>
<li><a href="https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html">Official yfinance download reference</a></li>
</ul>
<div class="foot">Research prototype only; no historical result guarantees future profit. New artifacts are under <code>data/processed/followup_*</code> and <code>data/processed/grid_*</code>. Existing <code>report.html</code> was not modified.</div>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pairs_research_2026-08-26"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    args = parser.parse_args()
    output = args.output_dir
    raw = output / "data" / "raw"
    processed = output / "data" / "processed"
    charts = output / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    prices = pd.read_csv(raw / "yahoo_adjusted_close_8y.csv", index_col=0, parse_dates=True)
    selected = pd.read_csv(processed / "formation_selected_top10_by_sector.csv")
    pair_returns = pd.read_csv(processed / "pair_net_returns.csv", index_col=0, parse_dates=True)
    signals = pd.read_csv(processed / "pair_daily_signals.csv.gz", parse_dates=["date"])
    trades = pd.read_csv(processed / "pair_trade_ledger.csv", parse_dates=["entry_date", "exit_date"])
    pair_metrics = pd.read_csv(processed / "pair_backtest_metrics.csv")
    portfolio_metrics = pd.read_csv(processed / "portfolio_metrics.csv")
    best10 = pd.read_csv(processed / "best_10_pairs_by_heldout_net_cagr.csv")
    best_names = best10["pair"].astype(str).tolist()

    heldout_start = pd.Timestamp(manifest["test_start"])
    heldout_end = pd.Timestamp(manifest["as_of"])
    lockbox_start = pd.Timestamp("2025-08-27")

    grid_spec = GridSearchSpec(
        zscore_methods=("sma", "ewma"),
        zscore_lookbacks=(20, 40, 60, 90),
        entry_zs=(1.25, 1.5, 1.75, 2.0, 2.5),
        exit_zs=(0.0, 0.25, 0.5),
        maximum_holding_days=(20, 40, 60),
    )
    grid = run_parameter_grid(
        selected_pairs=selected,
        prices=prices,
        heldout_start=heldout_start,
        lockbox_start=lockbox_start,
        heldout_end=heldout_end,
        base_config=config.strategy,
        spec=grid_spec,
        objective="sharpe",
        ranking_portfolio="sector_balanced",
        minimum_median_trades_per_pair=6,
    )

    # Baseline portfolios and market diagnostics.
    baseline_curves, _ = build_portfolios(pair_returns, selected, best_names)
    market = equal_weight_market_proxy(prices).reindex(pair_returns.index)
    portfolio_return_columns = [
        "all_formation_selected",
        "sector_balanced",
        "expost_top_performers",
    ]
    market_rows = []
    for name in portfolio_return_columns:
        market_rows.append(
            {"portfolio": name, **regression_diagnostics(baseline_curves[name], market)}
        )
    market_diagnostics = pd.DataFrame(market_rows)
    shock_frame = shock_diagnostics(
        baseline_curves[["all_formation_selected", "expost_top_performers"]],
        market,
    )
    pair_beta = pd.DataFrame(
        [
            {
                "pair": pair,
                "sector": selected.set_index("pair").loc[pair, "sector"],
                **regression_diagnostics(pair_returns[pair], market),
            }
            for pair in pair_returns
        ]
    )

    active = signals.pivot(index="date", columns="pair", values="position")
    correlation_matrix, active_correlations, correlation_summary = pair_correlation_diagnostics(
        pair_returns,
        active,
        minimum_active_overlap=20,
    )
    correlation_detail, overlap_correlation_summary = overlap_correlation_diagnostics(
        correlation_matrix,
        active_correlations,
        selected,
    )
    top_selected = selected.loc[selected["pair"].isin(best_names)].copy()
    top_returns = pair_returns[best_names]
    top_active = active[best_names]
    top_matrix, top_active_correlations, top_correlation_summary = pair_correlation_diagnostics(
        top_returns,
        top_active,
        minimum_active_overlap=20,
    )
    top_correlation_detail, top_overlap_summary = overlap_correlation_diagnostics(
        top_matrix,
        top_active_correlations,
        top_selected,
    )

    exposure, name_exposure, exposure_summary = portfolio_exposure_diagnostics(signals, selected)
    top_exposure, top_name_exposure, top_exposure_summary = portfolio_exposure_diagnostics(
        signals,
        top_selected,
    )
    ticker_overlap = ticker_overlap_table(selected)
    top_ticker_overlap = ticker_overlap_table(top_selected)

    # Persist the complete audit trail.
    pd.DataFrame({market.name: market}).to_csv(processed / "followup_market_proxy_returns.csv")
    market_diagnostics.to_csv(processed / "followup_market_beta.csv", index=False)
    shock_frame.to_csv(processed / "followup_market_shock_diagnostics.csv", index=False)
    pair_beta.to_csv(processed / "followup_pair_market_beta.csv", index=False)
    correlation_matrix.to_csv(processed / "followup_pair_correlation_matrix.csv")
    active_correlations.to_csv(processed / "followup_active_pair_correlations.csv.gz", index=False, compression="gzip")
    correlation_detail.to_csv(processed / "followup_overlap_correlation_detail.csv.gz", index=False, compression="gzip")
    overlap_correlation_summary.to_csv(processed / "followup_overlap_correlation_summary.csv", index=False)
    exposure.to_csv(processed / "followup_portfolio_exposure_daily.csv", index=False)
    name_exposure.to_csv(processed / "followup_name_exposure_daily.csv.gz", index=False, compression="gzip")
    exposure_summary.to_csv(processed / "followup_portfolio_exposure_summary.csv", index=False)
    top_exposure_summary.to_csv(processed / "followup_expost10_exposure_summary.csv", index=False)
    ticker_overlap.to_csv(processed / "followup_ticker_overlap.csv", index=False)
    top_ticker_overlap.to_csv(processed / "followup_expost10_ticker_overlap.csv", index=False)
    grid.summary.to_csv(processed / "grid_parameter_summary.csv", index=False)
    grid.ranking.to_csv(processed / "grid_parameter_ranking.csv", index=False)
    grid.comparison.to_csv(processed / "grid_baseline_winner_comparison.csv", index=False)
    grid.failures.to_csv(processed / "grid_failures.csv", index=False)
    grid.best_evaluation.pair_metrics.to_csv(processed / "grid_selected_pair_metrics.csv", index=False)
    grid.best_evaluation.signals.to_csv(processed / "grid_selected_pair_signals.csv.gz", index=False, compression="gzip")
    grid.best_evaluation.trades.to_csv(processed / "grid_selected_trade_ledger.csv", index=False)
    grid.best_evaluation.pair_returns.to_csv(processed / "grid_selected_pair_returns.csv")
    tested_trial_count = int(
        grid.summary.loc[grid.summary["split"].eq("tuning"), "config_id"].nunique()
    )
    eligible_rule_count = int(grid.ranking["config_id"].nunique())
    grid_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_start": str(heldout_start.date()),
        "tuning_end_inclusive": str((lockbox_start - pd.Timedelta(days=1)).date()),
        "lockbox_start": str(lockbox_start.date()),
        "heldout_end": str(heldout_end.date()),
        "objective": "net sector-balanced tuning Sharpe",
        "minimum_median_tuning_trades_per_pair": 6,
        "trial_count": tested_trial_count,
        "frequency_eligible_rule_count": eligible_rule_count,
        "best_parameters": grid.best_parameters,
        "baseline_parameters": {
            "zscore_method": config.strategy.zscore_method,
            "zscore_lookback": config.strategy.zscore_lookback,
            "zscore_min_periods": config.strategy.zscore_min_periods,
            "entry_z": config.strategy.entry_z,
            "exit_z": config.strategy.exit_z,
            "maximum_holding_days": config.strategy.maximum_holding_days,
        },
        "transaction_cost_bps": config.strategy.transaction_cost_bps,
        "annual_short_borrow_bps": config.strategy.annual_short_borrow_bps,
        "failure_count": int(len(grid.failures)),
        "warning": "The lockbox is spent after this report; results are not a production guarantee.",
    }
    (processed / "grid_search_metadata.json").write_text(
        json.dumps(grid_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_grid_returns = _sector_balanced_returns(
        grid.baseline_evaluation.pair_returns,
        selected,
    )
    selected_grid_returns = _sector_balanced_returns(
        grid.best_evaluation.pair_returns,
        selected,
    )
    _plot_validation_equity(
        baseline_grid_returns,
        selected_grid_returns,
        lockbox_start,
        charts / "condensed_validation_equity.png",
    )
    tuning_frontier = grid.summary.loc[
        grid.summary["split"].eq("tuning")
        & grid.summary["portfolio"].eq("sector_balanced")
    ].copy()
    _plot_grid_frontier(
        tuning_frontier,
        grid.ranking,
        charts / "condensed_grid_frontier.png",
    )
    _plot_trade_frequency(
        grid.baseline_evaluation.pair_metrics,
        grid.best_evaluation.pair_metrics,
        grid.baseline_evaluation.trades,
        charts / "condensed_trade_frequency.png",
    )
    _plot_correlations(top_matrix, overlap_correlation_summary, charts / "condensed_pair_correlations.png")
    _plot_exposure_and_shocks(exposure, shock_frame, charts / "condensed_exposure_shocks.png")

    baseline_row = portfolio_metrics.loc[
        portfolio_metrics["portfolio"].eq("all_formation_selected")
    ].iloc[0]
    expost_row = portfolio_metrics.loc[
        portfolio_metrics["portfolio"].eq("expost_top_performers")
    ].iloc[0]
    market_row = market_diagnostics.loc[
        market_diagnostics["portfolio"].eq("all_formation_selected")
    ].iloc[0]
    down_row = shock_frame.loc[
        shock_frame["regime"].eq("market_down_at_least_2%")
        & shock_frame["portfolio"].eq("all_formation_selected")
    ].iloc[0]
    comparison = grid.comparison.loc[
        grid.comparison["portfolio"].eq("sector_balanced")
    ].set_index("split")
    tuning = comparison.loc["tuning"]
    lockbox = comparison.loc["lockbox"]
    best_ewma = grid.ranking.loc[grid.ranking["zscore_method"].eq("ewma")].iloc[0]
    exposure_row = exposure_summary.iloc[0]
    top_exposure_row = top_exposure_summary.iloc[0]

    rules_display = pd.DataFrame(
        [
            ["Pair selection", "10 per sector; strict FDR/I(1)/half-life/Hurst gate, then deterministic fallback", "Only 7 strict; 103 fallback"],
            ["Spread", "Formation-frozen log-price OLS α and β", "Static relation; no rolling break gate"],
            ["Z-score", "60-session simple rolling mean and sample σ; 40-session warm-up", "SMA, not EWMA"],
            ["Entry / exit", "Enter |z| ≥ 2; exit |z| ≤ 0.5", "Close signal, next-session position"],
            ["Stops", "Exit |z| ≥ 4; 60-session time stop; missing-signal exit", "No P&L stop, portfolio stop, or cooldown"],
            ["Pair weights", "1/(1+β) and −β/(1+β); 1× gross while active", "Not necessarily dollar- or beta-neutral"],
            ["Costs", "5 bp per dollar turnover + 30 bp/year short carry", "No quotes, impact, ADV, locate, or sleeve-rebalance cost"],
        ],
        columns=["Control", "Current rule", "Important limitation"],
    )
    allocation_display = pd.DataFrame(
        [
            ["All selected", "1/110 per pair", "Static equal sleeve capital", "Defensible baseline"],
            ["Sector balanced", "1/11 per sector, then 1/10 per pair", "Identical to all-selected here", "No incremental diversification"],
            ["Ex-post ten", "1/10 per winner", "40% Industrials; 15 unique names", "Selection-biased, not live"],
        ],
        columns=["Portfolio", "Allocation", "Resulting structure", "Use"],
    )
    market_display = market_diagnostics.copy()
    market_display["portfolio"] = market_display["portfolio"].replace(
        {
            "all_formation_selected": "All 110",
            "sector_balanced": "Sector balanced",
            "expost_top_performers": "Ex-post ten",
        }
    )
    market_display = market_display[["portfolio", "market_beta", "market_correlation", "r_squared", "annualized_alpha"]]
    market_display.columns = ["Portfolio", "Market β", "Correlation", "R²", "Annualized α"]
    for column in ["Market β", "Correlation", "R²"]:
        market_display[column] = market_display[column].map(lambda value: _number(value, 3))
    market_display["Annualized α"] = market_display["Annualized α"].map(_percentage)

    all_overlap_display = ticker_overlap.head(6).copy()
    all_overlap_display["pair_count"] = all_overlap_display["pair_count"].astype(int)
    all_overlap_display = all_overlap_display[["ticker", "pair_count", "pairs"]]
    all_overlap_display.columns = ["Name", "Pairs", "Sleeves"]
    top_overlap_display = top_ticker_overlap.head(6).copy()
    top_overlap_display["pair_count"] = top_overlap_display["pair_count"].astype(int)
    top_overlap_display = top_overlap_display[["ticker", "pair_count", "pairs"]]
    top_overlap_display.columns = ["Name", "Pairs", "Sleeves"]

    comparison_display = pd.DataFrame(
        [
            ["Tuning · first 2y", _percentage(tuning["cagr_baseline"]), _percentage(tuning["cagr_best"]), _number(tuning["sharpe_baseline"], 3), _number(tuning["sharpe_best"], 3), _number(tuning["annual_turnover_baseline"], 1) + "×", _number(tuning["annual_turnover_best"], 1) + "×", _number(tuning["median_trades_per_pair_baseline"], 1), _number(tuning["median_trades_per_pair_best"], 1)],
            ["Lockbox · final 1y", _percentage(lockbox["cagr_baseline"]), _percentage(lockbox["cagr_best"]), _number(lockbox["sharpe_baseline"], 3), _number(lockbox["sharpe_best"], 3), _number(lockbox["annual_turnover_baseline"], 1) + "×", _number(lockbox["annual_turnover_best"], 1) + "×", _number(lockbox["median_trades_per_pair_baseline"], 1), _number(lockbox["median_trades_per_pair_best"], 1)],
        ],
        columns=["Window", "Baseline CAGR", "Winner CAGR", "Baseline Sharpe", "Winner Sharpe", "Baseline turnover", "Winner turnover", "Baseline median trades", "Winner median trades"],
    )
    top_grid_display = grid.ranking.head(8)[
        ["rank", "zscore_method", "zscore_lookback", "entry_z", "exit_z", "maximum_holding_days", "cagr", "sharpe", "annual_turnover", "median_trades_per_pair"]
    ].copy()
    top_grid_display["zscore_method"] = top_grid_display["zscore_method"].str.upper()
    top_grid_display["cagr"] = top_grid_display["cagr"].map(_percentage)
    top_grid_display["sharpe"] = top_grid_display["sharpe"].map(lambda value: _number(value, 3))
    top_grid_display["annual_turnover"] = top_grid_display["annual_turnover"].map(lambda value: _number(value, 1) + "×")
    top_grid_display.columns = ["Rank", "Method", "Lookback", "Entry z", "Exit z", "Max hold", "Tuning CAGR", "Tuning Sharpe", "Turnover", "Median trades"]

    performance_display = portfolio_metrics.copy()
    performance_display["portfolio"] = performance_display["portfolio"].replace(
        {
            "all_formation_selected": "All 110 · defensible",
            "sector_balanced": "Sector balanced · same weights here",
            "expost_top_performers": "Ex-post ten · selection-biased",
        }
    )
    performance_display = performance_display[["portfolio", "total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown"]]
    for column in ["total_return", "cagr", "annual_volatility", "max_drawdown"]:
        performance_display[column] = performance_display[column].map(_percentage)
    performance_display["sharpe"] = performance_display["sharpe"].map(lambda value: _number(value, 3))
    performance_display.columns = ["Portfolio", "Total return", "CAGR", "Annual vol", "Sharpe", "Max drawdown"]

    roadmap_display = pd.DataFrame(
        [
            ["Constrained allocator", "Minimize covariance with target return; enforce market/factor β, gross/net, name, shared-underlying and sector caps", "Highest"],
            ["Rolling pair health", "Online cointegration/break diagnostics, beta drift, half-life drift, kill switch and cooldown", "Highest"],
            ["Nested walk-forward", "Purged rolling folds, frozen hyperparameters, lockboxes, bootstrap and data-snooping tests", "Highest"],
            ["Executable cost model", "Bid/ask, participation/ADV, nonlinear impact, locate/borrow, financing, cash yield and terminal liquidation", "Highest"],
            ["Signal ensemble", "SMA, EWMA, robust median/MAD z, OU bands and Kalman beta under a common validation protocol", "Medium"],
            ["Capital utilization", "Volatility targeting, covariance shrinkage, risk parity, cluster caps and cash collateral return", "Medium"],
            ["Point-in-time data", "Historical membership/GICS, delistings, corporate actions, fundamentals and ALFRED vintages", "Foundational"],
            ["Live controls", "Reconciliation, stale data checks, order/fill monitoring, exposure breach alerts and portfolio drawdown kill switch", "Production gate"],
        ],
        columns=["Function", "What it should do", "Priority"],
    )

    first_date = pair_returns.index.min()
    first_day_entries = int((trades["entry_date"] == first_date).sum())
    end_open_trades = int(trades["exit_reason"].eq("end_of_test").sum())
    selected_pair_metrics = grid.best_evaluation.pair_metrics
    top_name_max = (
        top_name_exposure.groupby("ticker")["exposure"].apply(lambda values: values.abs().max()).max()
    )
    groups = overlap_correlation_summary.set_index("group")
    report = Template(REPORT_TEMPLATE).render(
        as_of=manifest["as_of"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        baseline_cagr=_percentage(baseline_row["cagr"]),
        baseline_sharpe=_number(baseline_row["sharpe"], 3),
        baseline_vol=_percentage(baseline_row["annual_volatility"]),
        market_beta=_number(market_row["market_beta"], 3),
        market_corr=_number(market_row["market_correlation"], 3),
        market_r2=_percentage(market_row["r_squared"], 1),
        baseline_trades=_integer(pair_metrics["trade_count"].median()),
        baseline_trades_per_year=_number(pair_metrics["trade_count"].median() / 3.0, 1),
        total_trades=_integer(len(trades)),
        trade_range=f"{int(pair_metrics['trade_count'].min())}–{int(pair_metrics['trade_count'].max())}",
        selected_full_trades=_integer(selected_pair_metrics["trade_count"].median()),
        selected_lockbox_trades=_number(lockbox["median_trades_per_pair_best"], 1),
        baseline_lockbox_trades=_number(lockbox["median_trades_per_pair_baseline"], 1),
        baseline_lockbox_turnover=_number(lockbox["annual_turnover_baseline"], 1) + "×",
        selected_lockbox_turnover=_number(lockbox["annual_turnover_best"], 1) + "×",
        grid_lockbox_cagr=_percentage(lockbox["cagr_best"]),
        grid_lockbox_sharpe=_number(lockbox["sharpe_best"], 3),
        baseline_lockbox_cagr=_percentage(lockbox["cagr_baseline"]),
        baseline_lockbox_sharpe=_number(lockbox["sharpe_baseline"], 3),
        grid_trials=_integer(tested_trial_count),
        eligible_rules=_integer(eligible_rule_count),
        best_ewma_rank=_integer(best_ewma["rank"]),
        best_ewma_sharpe=_number(best_ewma["sharpe"], 3),
        mean_gross=_percentage(exposure_row["mean_sleeve_gross_exposure"], 1),
        first_day_entries=_integer(first_day_entries),
        end_open_trades=_integer(end_open_trades),
        down_shock_days=_integer(down_row["observations"]),
        down_shock_return=_percentage(down_row["mean_daily_return"]),
        down_shock_cumulative=_percentage(down_row["cumulative_return"]),
        all_abs_corr=_percentage(correlation_summary["unconditional_correlation_absolute_mean"], 1),
        all_active_abs_corr=_percentage(correlation_summary["active_correlation_absolute_mean"], 1),
        shared_active_abs_corr=_percentage(groups.loc["shared_underlying", "mean_absolute_active_correlation"], 1),
        disjoint_active_abs_corr=_percentage(groups.loc["disjoint_underlyings", "mean_absolute_active_correlation"], 1),
        unique_names=_integer(exposure_row["unique_names"]),
        repeated_names=_integer(exposure_row["names_used_more_than_once"]),
        top_max_name=_percentage(top_name_max, 1),
        rules_table=_table(rules_display, classes="data-table rule"),
        allocation_table=_table(allocation_display),
        market_table=_table(market_display),
        overlap_all_table=_table(all_overlap_display),
        overlap_top_table=_table(top_overlap_display),
        grid_comparison_table=_table(comparison_display),
        grid_top_table=_table(top_grid_display),
        roadmap_table=_table(roadmap_display, classes="data-table roadmap"),
        performance_table=_table(performance_display),
    )
    report_path = output / "condensed_report.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print(json.dumps(grid_metadata, indent=2))


if __name__ == "__main__":
    main()
