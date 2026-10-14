#!/usr/bin/env python3
"""Render the additive V6 pair-vintage report and implementation QA.

This script is deliberately a read-only consumer of the V6 research tables.
It never reselects a pair, retunes a signal, or aggregates returns into a
portfolio CAGR.  Missing or legitimately empty tuning artifacts produce an
explicitly incomplete report rather than a fabricated performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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


DEFAULT_OUTPUT = Path(
    "outputs/pair_return_optimization_10y_train3y_2026-08-26"
)
DEFAULT_CONFIG = Path("configs/pair_optimized_10y_3y.json")

PRIMARY_COST_BPS = 2.0
SENSITIVITY_COST_BPS = 1.0
FDR_ALPHA = 0.05
DESCRIPTIVE_TOP10_FILENAME = "pair_v6_descriptive_top10_oos_pair_vintages.csv"

INK = "#162033"
MUTED = "#647086"
BLUE = "#2f67c7"
TEAL = "#148a82"
GREEN = "#26845d"
ORANGE = "#d9792d"
RED = "#bf4c55"
PURPLE = "#7658b5"
PALE = "#eef3f8"

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


TABLE_FILES: dict[str, str] = {
    "rebalance_schedule": "pair_v6_rebalance_schedule.csv",
    "screen_summary": "pair_v6_rolling_screen_summary.csv",
    "strict_candidates": "pair_v6_strict_candidates.csv.gz",
    "parameter_rankings": "pair_v6_parameter_rankings.csv.gz",
    "fold_evaluations": "pair_v6_inner_fold_evaluations.csv.gz",
    "model_audit": "pair_v6_inner_model_audit.csv",
    "validation_returns": "pair_v6_validation_returns.csv.gz",
    "selected_validation_signals": "pair_v6_selected_validation_signals.csv.gz",
    "validation_cost_sensitivity": "pair_v6_validation_cost_sensitivity.csv",
    "selected_schedule": "pair_v6_selected_parameter_schedule.csv",
    "oos_signals": "pair_v6_oos_signals.csv.gz",
    "oos_trades": "pair_v6_oos_trades.csv",
    "pair_vintage_metrics": "pair_v6_pair_vintage_metrics.csv",
    "oos_baseline_metrics": "pair_v6_oos_baseline_metrics.csv",
    "oos_baseline_signals": "pair_v6_oos_baseline_signals.csv.gz",
    "cost_sensitivity": "pair_v6_cost_sensitivity.csv",
    "failures": "pair_v6_tuning_failures.csv",
    "vintage_summary": "pair_v6_tuning_vintage_summary.csv",
}

# Every completed runner writes these tables, even when they contain only
# headers because no strict pair was available.  Strict candidates and the two
# validation-detail panels are useful but not needed to certify report inputs.
REQUIRED_TABLES = frozenset(
    {
        "rebalance_schedule",
        "screen_summary",
        "parameter_rankings",
        "fold_evaluations",
        "model_audit",
        "selected_schedule",
        "oos_signals",
        "oos_trades",
        "pair_vintage_metrics",
        "cost_sensitivity",
        "failures",
        "vintage_summary",
    }
)

DATE_COLUMNS = frozenset(
    {
        "date",
        "rebalance_date",
        "formation_start",
        "formation_end",
        "outer_start",
        "outer_end",
        "outer_end_exclusive",
        "train_start",
        "train_end",
        "model_train_start",
        "model_train_end",
        "training_start",
        "training_end",
        "validation_start",
        "validation_end",
        "evaluation_start",
        "evaluation_end",
        "entry_date",
        "exit_date",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()
    for column in frame.columns.intersection(pd.Index(DATE_COLUMNS)):
        frame[column] = pd.to_datetime(
            frame[column], errors="coerce", format="mixed"
        )
    return frame


def load_report_inputs(output_root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load every known global table without assuming that tuning found pairs."""

    root = Path(output_root).resolve()
    processed = root / "data" / "processed"
    paths = {name: processed / filename for name, filename in TABLE_FILES.items()}
    return {name: _read_csv(path) for name, path in paths.items()}, paths


def _number(values: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for column in candidates:
        if column in values:
            return pd.to_numeric(values[column], errors="coerce")
    return pd.Series(np.nan, index=values.index, dtype=float)


def _boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    truthy = {"1", "true", "yes", "y", "strict_eligible"}
    return values.astype("string").str.strip().str.lower().isin(truthy)


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


def _integer(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{int(round(number)):,}" if math.isfinite(number) else "—"


def _table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return '<p class="muted">No rows available.</p>'
    return frame.head(max_rows).to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
        na_rep="—",
    )


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


def _placeholder(path: Path, title: str, message: str) -> Path:
    figure, axis = plt.subplots(figsize=(11.8, 3.6))
    axis.axis("off")
    axis.text(0.02, 0.78, title, fontsize=15, fontweight="bold", color=INK)
    axis.text(0.02, 0.48, message, fontsize=11, color=MUTED, wrap=True)
    axis.add_patch(
        plt.Rectangle((0.02, 0.16), 0.96, 0.08, color=PALE, transform=axis.transAxes)
    )
    return _save_figure(figure, path)


def _chart_decodable(path: Path) -> bool:
    try:
        image = mpimg.imread(path)
    except Exception:
        return False
    return bool(image.size and image.shape[0] > 0 and image.shape[1] > 0)


def _plot_strict_supply(screen: pd.DataFrame, path: Path) -> Path:
    required = {"rebalance_date", "strict_eligible"}
    if screen.empty or not required.issubset(screen.columns):
        return _placeholder(
            path,
            "Strict candidate supply",
            "No completed rolling-screen summary is available yet.",
        )
    work = screen.dropna(subset=["rebalance_date"]).sort_values("rebalance_date")
    if work.empty:
        return _placeholder(path, "Strict candidate supply", "The screen summary is empty.")
    dates = pd.to_datetime(work["rebalance_date"])
    strict = _number(work, ("strict_eligible",)).fillna(0.0)
    fdr = _number(work, ("fdr_significant",)).fillna(0.0)
    sectors = _number(work, ("strict_sectors",)).fillna(0.0)
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.3), sharex=True)
    axes[0].plot(dates, fdr, color=MUTED, linewidth=1.5, label="5% BH-FDR discoveries")
    axes[0].plot(dates, strict, color=BLUE, linewidth=2.2, marker="o", markersize=3.5, label="Strict eligible")
    axes[0].fill_between(dates, 0, strict, color=BLUE, alpha=0.10)
    axes[0].set_title("Strict candidate supply by formation date", loc="left")
    axes[0].set_ylabel("Pair count")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].bar(dates, sectors, width=40, color=TEAL, alpha=0.82)
    axes[1].set_title("Sectors represented by strict candidates", loc="left")
    axes[1].set_ylabel("Sector count")
    axes[1].set_xlabel("Rebalance date")
    for axis in axes:
        _style_axis(axis)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout(h_pad=2.0)
    return _save_figure(figure, path)


def _plot_pair_vintage_results(
    metrics: pd.DataFrame,
    schedule: pd.DataFrame,
    path: Path,
) -> Path:
    if metrics.empty:
        return _placeholder(
            path,
            "Actual outer pair-vintage results",
            "No successful outer pair-vintage result is available. No return was imputed.",
        )
    work = metrics.copy()
    oos_return = _number(work, ("net_total_return", "total_return"))
    work["_oos_return"] = oos_return
    if "training_cv_net_cagr" not in work and not schedule.empty:
        keys = [name for name in ("rebalance_date", "pair") if name in work and name in schedule]
        if len(keys) == 2 and "training_cv_net_cagr" in schedule:
            work = work.merge(
                schedule[keys + ["training_cv_net_cagr"]].drop_duplicates(keys),
                on=keys,
                how="left",
            )
    cv = _number(work, ("training_cv_net_cagr", "pair_cv_net_cagr"))
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    finite = np.isfinite(cv) & np.isfinite(work["_oos_return"])
    if finite.any():
        axes[0].scatter(
            cv[finite],
            work.loc[finite, "_oos_return"],
            color=BLUE,
            alpha=0.72,
            s=34,
            edgecolors="white",
            linewidth=0.4,
        )
        axes[0].axhline(0, color="#aeb8c7", linewidth=0.9)
        axes[0].set_xlabel("Training CV net CAGR")
        axes[0].set_ylabel("Next-quarter net return")
        axes[0].xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
        axes[0].set_title("Training score versus actual outer result", loc="left")
    else:
        axes[0].axis("off")
        axes[0].text(0.02, 0.72, "Training/OOS comparison unavailable", color=MUTED)
    clean = work["_oos_return"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean):
        axes[1].hist(clean, bins=min(20, max(5, int(math.sqrt(len(clean)) + 1))), color=TEAL, alpha=0.82)
        axes[1].axvline(0, color=RED, linewidth=1.0)
        axes[1].axvline(clean.median(), color=INK, linestyle="--", linewidth=1.3, label="Median")
        axes[1].set_xlabel("Actual next-quarter net return")
        axes[1].set_ylabel("Pair-vintage count")
        axes[1].xaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
        axes[1].set_title("Distribution across actual pair sleeves", loc="left")
        axes[1].legend(frameon=False)
    else:
        axes[1].axis("off")
        axes[1].text(0.02, 0.72, "No finite outer returns", color=MUTED)
    for axis in axes:
        if axis.axison:
            _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    return _save_figure(figure, path)


def _descriptive_top_pair_vintages(
    metrics: pd.DataFrame,
    schedule: pd.DataFrame,
    maximum: int = 10,
) -> pd.DataFrame:
    """Rank realized outer sleeves for diagnosis, never for model selection."""

    output_columns = [
        "descriptive_rank",
        "pair",
        "rebalance_date",
        "sector",
        "evaluation_start",
        "evaluation_end",
        "outer_end_exclusive",
        "observations",
        "net_total_return",
        "net_cagr",
        "max_drawdown",
        "trade_count",
        "training_cv_net_cagr",
        "config_id",
        "zscore_method",
        "zscore_lookback",
        "zscore_min_periods",
        "entry_z",
        "exit_z",
        "maximum_holding_days",
        "transaction_cost_bps",
        "ranking_metric",
        "selection_timing",
        "deployable_selection",
    ]
    if metrics.empty or maximum <= 0:
        return pd.DataFrame(columns=output_columns)

    work = metrics.copy()
    work["_ranking_return"] = _number(
        work, ("net_total_return", "total_return")
    )
    work["_ranking_cagr"] = _number(work, ("net_cagr", "cagr"))
    work["_ranking_drawdown"] = _number(work, ("max_drawdown",))
    work = work.loc[np.isfinite(work["_ranking_return"])].copy()
    if work.empty:
        return pd.DataFrame(columns=output_columns)

    keys = [
        column
        for column in ("rebalance_date", "pair")
        if column in work and column in schedule
    ]
    detail_columns = [
        column
        for column in (
            "zscore_method",
            "zscore_lookback",
            "zscore_min_periods",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
        )
        if column in schedule and column not in work
    ]
    if len(keys) == 2 and detail_columns:
        details = schedule[keys + detail_columns].drop_duplicates(keys)
        work = work.merge(details, on=keys, how="left", validate="many_to_one")

    sort_columns = ["_ranking_return", "_ranking_cagr", "_ranking_drawdown"]
    ascending = [False, False, False]
    for column in ("rebalance_date", "pair"):
        if column in work:
            sort_columns.append(column)
            ascending.append(True)
    work = work.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    ).head(maximum)
    work.insert(0, "descriptive_rank", np.arange(1, len(work) + 1))
    work["ranking_metric"] = "realized_oos_net_total_return"
    work["selection_timing"] = "ex_post_after_outer_returns"
    work["deployable_selection"] = False
    for column in output_columns:
        if column not in work:
            work[column] = np.nan
    return work[output_columns].reset_index(drop=True)


def _plot_descriptive_top10(
    ranking: pd.DataFrame,
    signals: pd.DataFrame,
    path: Path,
) -> Path:
    if ranking.empty:
        return _placeholder(
            path,
            "Ex-post outer-performance leaders",
            "No successful outer pair-vintage is available to rank.",
        )

    work = ranking.copy()
    returns = _number(work, ("net_total_return", "total_return"))
    labels = []
    for _, row in work.iterrows():
        vintage = (
            pd.Timestamp(row["rebalance_date"]).strftime("%Y-%m-%d")
            if pd.notna(row.get("rebalance_date"))
            else "unknown"
        )
        pair_label = str(row.get("pair", "Pair")).replace("__", " / ")
        labels.append(
            f"#{int(row['descriptive_rank'])} {pair_label} · {vintage}"
        )

    figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.2))
    positions = np.arange(len(work))
    bar_colors = [TEAL if value >= 0 else RED for value in returns]
    axes[0].barh(positions, returns, color=bar_colors, alpha=0.86)
    axes[0].set_yticks(positions, labels=labels, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].axvline(0.0, color="#aeb8c7", linewidth=0.9)
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_xlabel("Net return during that outer quarter")
    axes[0].set_title("Ranked by realized net total return", loc="left")
    finite_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    span = max(float(finite_returns.max() - finite_returns.min()), 0.01)
    for position, value in zip(positions, returns):
        if math.isfinite(float(value)):
            axes[0].text(
                float(value) + (0.018 * span if value >= 0 else -0.018 * span),
                position,
                f"{value:.1%}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8.5,
                color=INK,
            )

    colors = plt.get_cmap("tab10")(np.arange(len(work)) % 10)
    plotted = 0
    for (index, row), color, label in zip(work.iterrows(), colors, labels):
        series = signals.copy()
        if "pair" in series:
            series = series.loc[
                series["pair"].astype(str).eq(str(row.get("pair")))
            ]
        if "rebalance_date" in series and pd.notna(row.get("rebalance_date")):
            series = series.loc[
                pd.to_datetime(
                    series["rebalance_date"], errors="coerce", format="mixed"
                ).eq(pd.Timestamp(row["rebalance_date"]))
            ]
        if series.empty:
            continue
        if "date" in series:
            series = series.sort_values("date")
        equity = _number(series, ("net_equity",))
        if not np.isfinite(equity).any():
            equity = (1.0 + _number(series, ("net_return",)).fillna(0.0)).cumprod()
        equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
        if equity.empty:
            continue
        sessions = np.arange(0, len(equity) + 1)
        values = np.r_[1.0, equity.to_numpy(dtype=float)]
        axes[1].plot(
            sessions,
            values,
            color=color,
            linewidth=1.65,
            alpha=0.92,
            label=label,
        )
        plotted += 1
    axes[1].axhline(1.0, color="#aeb8c7", linewidth=0.9)
    axes[1].set_xlabel("Outer-test trading session")
    axes[1].set_ylabel("Growth of $1 pair sleeve, net")
    axes[1].set_title("Frozen-rule OOS equity paths", loc="left")
    if plotted:
        axes[1].legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    else:
        axes[1].text(
            0.03,
            0.75,
            "No matching daily signal paths are available.",
            transform=axes[1].transAxes,
            color=MUTED,
        )
    for axis in axes:
        _style_axis(axis)
    figure.suptitle(
        "Top realized outer pair-vintages · ex-post diagnostic only",
        x=0.055,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    return _save_figure(figure, path)


def _plot_parameter_stability(schedule: pd.DataFrame, path: Path) -> Path:
    if schedule.empty or "selected_config_id" not in schedule:
        return _placeholder(
            path,
            "Parameter stability",
            "No selected parameter schedule is available.",
        )
    work = schedule.copy()
    configs = work["selected_config_id"].astype("string").value_counts().head(12)
    methods = (
        work["zscore_method"].astype("string").str.upper().value_counts()
        if "zscore_method" in work
        else pd.Series(dtype=float)
    )
    lookbacks = (
        pd.to_numeric(work["zscore_lookback"], errors="coerce").dropna().astype(int).value_counts().sort_index()
        if "zscore_lookback" in work
        else pd.Series(dtype=float)
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.6))
    labels = [str(value).replace("__", " · ") for value in configs.index]
    axes[0].barh(np.arange(len(configs)), configs.values, color=BLUE, alpha=0.83)
    axes[0].set_yticks(np.arange(len(configs)), labels=labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Selected pair-vintages")
    axes[0].set_title("Most frequently selected frozen rules", loc="left")
    if len(lookbacks):
        axes[1].bar(
            [str(value) for value in lookbacks.index],
            lookbacks.values,
            color=TEAL,
            alpha=0.82,
            label="Lookback",
        )
        axes[1].set_xlabel("Z-score lookback sessions")
        axes[1].set_ylabel("Selected pair-vintages")
    if len(methods):
        description = " · ".join(f"{name}: {count}" for name, count in methods.items())
        axes[1].text(0.02, 0.96, description, transform=axes[1].transAxes, va="top", color=MUTED)
    axes[1].set_title("Lookback and smoothing stability", loc="left")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    return _save_figure(figure, path)


def _plot_trade_frequency(metrics: pd.DataFrame, trades: pd.DataFrame, path: Path) -> Path:
    counts = _number(metrics, ("trade_count",)).dropna() if not metrics.empty else pd.Series(dtype=float)
    if not len(counts) and trades.empty:
        return _placeholder(
            path,
            "Outer trade frequency",
            "No outer trades or pair-vintage trade counts are available.",
        )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    if len(counts):
        upper = max(int(counts.max()), 1)
        axes[0].hist(counts, bins=np.arange(-0.5, upper + 1.5, 1), color=PURPLE, alpha=0.82)
        axes[0].axvline(counts.median(), color=INK, linestyle="--", linewidth=1.2, label="Median")
        axes[0].set_xlabel("Trades per outer pair-vintage")
        axes[0].set_ylabel("Pair-vintage count")
        axes[0].set_title("Trade-count distribution", loc="left")
        axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")
    if not trades.empty and "entry_date" in trades:
        entries = trades.dropna(subset=["entry_date"]).copy()
        entries["quarter"] = pd.to_datetime(entries["entry_date"]).dt.to_period("Q").astype(str)
        quarterly = entries.groupby("quarter", sort=True).size()
        axes[1].bar(np.arange(len(quarterly)), quarterly.values, color=ORANGE, alpha=0.84)
        axes[1].set_xticks(np.arange(len(quarterly))[:: max(1, len(quarterly) // 8)])
        axes[1].set_xticklabels(quarterly.index[:: max(1, len(quarterly) // 8)], rotation=35, ha="right")
        axes[1].set_ylabel("Pair entries")
        axes[1].set_title("Actual outer entries by calendar quarter", loc="left")
    else:
        axes[1].axis("off")
        axes[1].text(0.02, 0.72, "No dated trade ledger", color=MUTED)
    for axis in axes:
        if axis.axison:
            _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    return _save_figure(figure, path)


def _plot_cost_sensitivity(costs: pd.DataFrame, path: Path) -> Path:
    if costs.empty or "transaction_cost_bps" not in costs:
        return _placeholder(
            path,
            "Frozen-winner cost sensitivity",
            "No frozen 1–2 bp pair-vintage sensitivity table is available.",
        )
    work = costs.copy()
    work["_cost"] = pd.to_numeric(work["transaction_cost_bps"], errors="coerce")
    work["_return"] = _number(work, ("net_total_return", "total_return"))
    work = work.loc[work["_cost"].isin([SENSITIVITY_COST_BPS, PRIMARY_COST_BPS])]
    if work.empty or not np.isfinite(work["_return"]).any():
        return _placeholder(path, "Frozen-winner cost sensitivity", "No finite sensitivity returns are available.")
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    groups = [
        work.loc[work["_cost"].eq(cost), "_return"].dropna().to_numpy(dtype=float)
        for cost in (SENSITIVITY_COST_BPS, PRIMARY_COST_BPS)
    ]
    axes[0].boxplot(
        groups,
        tick_labels=["1 bp", "2 bp"],
        patch_artist=True,
        boxprops={"facecolor": PALE},
    )
    axes[0].axhline(0, color=RED, linewidth=0.9)
    axes[0].set_ylabel("Outer pair-vintage net return")
    axes[0].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
    axes[0].set_title("Same frozen winner repriced", loc="left")
    keys = [name for name in ("rebalance_date", "pair", "config_id") if name in work]
    if len(keys) >= 2:
        pivot = work.pivot_table(index=keys, columns="_cost", values="_return", aggfunc="first")
        if {SENSITIVITY_COST_BPS, PRIMARY_COST_BPS}.issubset(pivot.columns):
            pivot = pivot.dropna(subset=[SENSITIVITY_COST_BPS, PRIMARY_COST_BPS])
            limit = min(len(pivot), 100)
            for _, row in pivot.head(limit).iterrows():
                axes[1].plot(
                    [SENSITIVITY_COST_BPS, PRIMARY_COST_BPS],
                    [row[SENSITIVITY_COST_BPS], row[PRIMARY_COST_BPS]],
                    color=MUTED,
                    alpha=0.18,
                )
            medians = [pivot[cost].median() for cost in (SENSITIVITY_COST_BPS, PRIMARY_COST_BPS)]
            axes[1].plot([1, 2], medians, color=BLUE, marker="o", linewidth=2.4, label="Median")
            axes[1].set_xticks([1, 2], ["1 bp", "2 bp"])
            axes[1].set_ylabel("Outer pair-vintage net return")
            axes[1].yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
            axes[1].set_title("Cost drag without retuning", loc="left")
            axes[1].legend(frameon=False)
        else:
            axes[1].axis("off")
    else:
        axes[1].axis("off")
    for axis in axes:
        if axis.axison:
            _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    return _save_figure(figure, path)


def _leading_training_vintages(schedule: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if schedule.empty or maximum <= 0:
        return schedule.iloc[0:0].copy()
    work = schedule.copy()
    score = _number(work, ("training_cv_net_cagr", "pair_cv_net_cagr"))
    work["_training_score"] = score
    sort_columns = ["_training_score"]
    ascending = [False]
    for column in ("rebalance_date", "pair_rank", "pair"):
        if column in work:
            sort_columns.append(column)
            ascending.append(True)
    ranked = work.sort_values(
        sort_columns, ascending=ascending, kind="mergesort"
    )
    if "pair" in ranked:
        ranked = ranked.drop_duplicates("pair", keep="first")
    return ranked.head(maximum)


def _safe_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _plot_pair_dashboard(
    row: pd.Series,
    signals: pd.DataFrame,
    path: Path,
) -> Path:
    pair = str(row.get("pair", "unknown"))
    work = signals.copy()
    if "pair" in work:
        work = work.loc[work["pair"].astype(str).eq(pair)]
    if "rebalance_date" in work and pd.notna(row.get("rebalance_date")):
        work = work.loc[
            pd.to_datetime(work["rebalance_date"], errors="coerce").eq(
                pd.Timestamp(row["rebalance_date"])
            )
        ]
    if work.empty or "date" not in work:
        return _placeholder(
            path,
            f"{pair} · training-selected vintage",
            "No outer signal rows are available for this training-selected pair-vintage.",
        )
    work = work.dropna(subset=["date"]).sort_values("date")
    zscore = _number(work, ("zscore",))
    equity = _number(work, ("net_equity",))
    if not np.isfinite(equity).any():
        returns = _number(work, ("net_return",)).fillna(0.0)
        equity = (1.0 + returns).cumprod()
    entry = float(row.get("entry_z", math.nan))
    exit_z = float(row.get("exit_z", math.nan))
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.7), sharex=True)
    axes[0].plot(work["date"], zscore, color=BLUE, linewidth=1.5)
    axes[0].axhline(0, color="#b7c0cd", linewidth=0.8)
    if math.isfinite(entry):
        axes[0].axhline(entry, color=RED, linestyle="--", linewidth=1.0, label=f"Entry ±{entry:g}")
        axes[0].axhline(-entry, color=RED, linestyle="--", linewidth=1.0)
    if math.isfinite(exit_z):
        axes[0].axhline(exit_z, color=GREEN, linestyle=":", linewidth=1.0, label=f"Exit ±{exit_z:g}")
        axes[0].axhline(-exit_z, color=GREEN, linestyle=":", linewidth=1.0)
    event_column = "decision_event" if "decision_event" in work else "event" if "event" in work else None
    if event_column:
        events = work[event_column].astype("string")
        entries = events.str.contains("enter_", na=False)
        exits = events.str.contains("exit_", na=False)
        axes[0].scatter(work.loc[entries, "date"], zscore.loc[entries], color=ORANGE, marker="^", s=38, label="Entry")
        axes[0].scatter(work.loc[exits, "date"], zscore.loc[exits], color=INK, marker="x", s=34, label="Exit")
    axes[0].set_ylabel("Spread z-score")
    axes[0].set_title(
        f"{pair} · {pd.Timestamp(row.get('rebalance_date')).date() if pd.notna(row.get('rebalance_date')) else 'vintage'} · frozen outer signals",
        loc="left",
    )
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[1].plot(work["date"], equity, color=TEAL, linewidth=2.0)
    axes[1].axhline(1.0, color="#b7c0cd", linewidth=0.8)
    axes[1].set_ylabel("Growth of $1 sleeve")
    axes[1].set_xlabel("Actual outer date")
    axes[1].set_title("Net pair-sleeve equity at 2 bp", loc="left")
    for axis in axes:
        _style_axis(axis)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    figure.tight_layout(h_pad=2.0)
    return _save_figure(figure, path)


def _cost_summary(costs: pd.DataFrame) -> pd.DataFrame:
    if costs.empty or "transaction_cost_bps" not in costs:
        return pd.DataFrame()
    work = costs.copy()
    work["transaction_cost_bps"] = pd.to_numeric(work["transaction_cost_bps"], errors="coerce")
    work["net_total_return"] = _number(work, ("net_total_return", "total_return"))
    work["net_cagr"] = _number(work, ("net_cagr", "cagr"))
    work = work.loc[work["transaction_cost_bps"].isin([1.0, 2.0])]
    rows: list[dict[str, Any]] = []
    for cost, group in work.groupby("transaction_cost_bps", sort=True):
        returns = group["net_total_return"].dropna()
        cagrs = group["net_cagr"].dropna()
        rows.append(
            {
                "Cost": f"{cost:g} bp",
                "Pair-vintages": len(group),
                "Median OOS return": _pct(returns.median()),
                "Positive fraction": _pct((returns > 0).mean()) if len(returns) else "—",
                "Median annualized CAGR": _pct(cagrs.median()),
            }
        )
    return pd.DataFrame(rows)


def _baseline_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if metrics.empty:
        return pd.DataFrame(), "No successful pair-vintage result is available."
    tuned_return = _number(metrics, ("net_total_return", "total_return"))
    baseline_return = _number(
        metrics,
        ("baseline_net_total_return", "baseline_total_return"),
    )
    if np.isfinite(baseline_return).any():
        delta = tuned_return - baseline_return
        clean_delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
        return (
            pd.DataFrame(
                [
                    {
                        "Comparison": "Frozen tuned minus baseline",
                        "Comparable pair-vintages": len(clean_delta),
                        "Median OOS return delta": _pct(clean_delta.median()),
                        "Tuned wins": _pct((clean_delta > 0).mean()),
                    }
                ]
            ),
            "The baseline uses the same pair-vintage boundary and cost convention.",
        )
    role = next((column for column in ("strategy_role", "parameter_role", "evaluation_role") if column in metrics), None)
    if role and {"tuned", "baseline"}.issubset(set(metrics[role].astype(str).str.lower())):
        keys = [name for name in ("rebalance_date", "pair") if name in metrics]
        work = metrics.assign(_role=metrics[role].astype(str).str.lower(), _return=tuned_return)
        pivot = work.pivot_table(index=keys, columns="_role", values="_return", aggfunc="first")
        if {"tuned", "baseline"}.issubset(pivot.columns):
            delta = pivot["tuned"] - pivot["baseline"]
            clean_delta = delta.replace([np.inf, -np.inf], np.nan).dropna()
            return (
                pd.DataFrame(
                    [
                        {
                            "Comparison": "Frozen tuned minus baseline",
                            "Comparable pair-vintages": len(clean_delta),
                            "Median OOS return delta": _pct(clean_delta.median()),
                            "Tuned wins": _pct((clean_delta > 0).mean()),
                        }
                    ]
                ),
                "The baseline uses the same pair-vintage boundary and cost convention.",
            )
    return (
        pd.DataFrame(),
        "An outer tuned-versus-baseline counterfactual was not supplied. The report does not infer one from training ranks.",
    )


def _quality_checks(
    tables: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    chart_paths: list[Path],
    report_path: Path,
) -> dict[str, Any]:
    schedule = tables["selected_schedule"]
    metrics = tables["pair_vintage_metrics"]
    signals = tables["oos_signals"]
    trades = tables["oos_trades"]
    costs = tables["cost_sensitivity"]
    rankings = tables["parameter_rankings"]
    folds = tables["fold_evaluations"]
    vintage = tables["vintage_summary"]
    rebalance = tables["rebalance_schedule"]
    baseline = tables["oos_baseline_metrics"]

    source_presence = {name: path.is_file() for name, path in paths.items()}
    required_present = all(source_presence[name] for name in REQUIRED_TABLES)

    if schedule.empty:
        strict_only = True
        fdr_only = True
        no_fallback = True
        maximum_ten = True
        schedule_cost = True
        formation_lagged = True
    else:
        strict_only = (
            "tradable_eligible" in schedule
            and bool(_boolean(schedule["tradable_eligible"]).all())
        )
        qvalue = _number(schedule, ("screen_fdr_qvalue", "fdr_qvalue"))
        fdr_only = bool(
            np.isfinite(qvalue).all()
            and qvalue.between(0.0, FDR_ALPHA + 1e-12, inclusive="both").all()
        )
        roles = schedule.get("selection_role", pd.Series("", index=schedule.index)).astype("string").str.lower()
        no_fallback = bool(
            not roles.str.contains("fallback|quota|exploratory", regex=True, na=False).any()
            and (roles.eq("strict_eligible").all() if "selection_role" in schedule else strict_only)
        )
        maximum_ten = bool(
            schedule.groupby("rebalance_date", dropna=False).size().le(10).all()
            if "rebalance_date" in schedule
            else False
        )
        schedule_costs = _number(schedule, ("primary_transaction_cost_bps",))
        schedule_cost = bool(np.isfinite(schedule_costs).all() and np.allclose(schedule_costs, PRIMARY_COST_BPS))
        if {"formation_end", "outer_start"}.issubset(schedule):
            formation_lagged = bool(
                (pd.to_datetime(schedule["formation_end"]) < pd.to_datetime(schedule["outer_start"])).all()
            )
        elif {"formation_end", "rebalance_date"}.issubset(schedule):
            formation_lagged = bool(
                (pd.to_datetime(schedule["formation_end"]) < pd.to_datetime(schedule["rebalance_date"])).all()
            )
        else:
            formation_lagged = False

    selected_count = int(len(schedule))
    successful_count = int(len(metrics))
    natural_keys = [name for name in ("rebalance_date", "pair") if name in schedule and name in metrics]
    schedule_keys_unique = bool(
        schedule.empty
        or (len(natural_keys) == 2 and not schedule.duplicated(natural_keys).any())
    )
    metrics_keys_unique = bool(
        metrics.empty
        or (len(natural_keys) == 2 and not metrics.duplicated(natural_keys).any())
    )
    if schedule.empty and metrics.empty:
        metrics_match_selected_schedule = True
        selected_config_matches = True
    elif len(natural_keys) == 2:
        schedule_key_set = set(map(tuple, schedule[natural_keys].astype(str).to_numpy()))
        metric_key_set = set(map(tuple, metrics[natural_keys].astype(str).to_numpy()))
        metrics_match_selected_schedule = metric_key_set.issubset(schedule_key_set)
        if (
            schedule_keys_unique
            and metrics_keys_unique
            and "selected_config_id" in schedule
            and "config_id" in metrics
        ):
            reconciliation = metrics[natural_keys + ["config_id"]].merge(
                schedule[natural_keys + ["selected_config_id"]],
                on=natural_keys,
                how="left",
                validate="one_to_one",
            )
            selected_config_matches = bool(
                reconciliation["selected_config_id"].notna().all()
                and reconciliation["config_id"].astype(str).eq(
                    reconciliation["selected_config_id"].astype(str)
                ).all()
            )
        else:
            selected_config_matches = False
    else:
        metrics_match_selected_schedule = False
        selected_config_matches = False

    if schedule.empty:
        consecutive_pair_ranks = True
    elif {"rebalance_date", "pair_rank"}.issubset(schedule):
        consecutive_pair_ranks = bool(
            schedule.groupby("rebalance_date")["pair_rank"].apply(
                lambda values: sorted(pd.to_numeric(values, errors="coerce").dropna().astype(int).tolist())
                == list(range(1, len(values) + 1))
            ).all()
        )
    else:
        consecutive_pair_ranks = False

    ranking_keys = [
        name for name in ("rebalance_date", "pair") if name in rankings
    ]
    if rankings.empty:
        hyperparameter_grid_complete = selected_count == 0
        exactly_one_pair_winner = selected_count == 0
        winner_inside_one_se_set = selected_count == 0
        ranking_pair_objective = selected_count == 0
        ranking_primary_cost = selected_count == 0
        parameter_grid_protocol = selected_count == 0
        schedule_winners_match_rankings = selected_count == 0
    elif len(ranking_keys) == 2 and "config_id" in rankings:
        grouped_rankings = rankings.groupby(ranking_keys, dropna=False)
        hyperparameter_grid_complete = bool(
            grouped_rankings.size().eq(97).all()
            and grouped_rankings["config_id"].nunique().eq(97).all()
        )
        if "selected" in rankings:
            selected_ranking_rows = rankings.loc[_boolean(rankings["selected"])]
            exactly_one_pair_winner = bool(
                grouped_rankings["selected"].apply(
                    lambda values: int(_boolean(values).sum())
                ).eq(1).all()
            )
        else:
            selected_ranking_rows = pd.DataFrame()
            exactly_one_pair_winner = False
        winner_inside_one_se_set = bool(
            exactly_one_pair_winner
            and "within_one_standard_error" in selected_ranking_rows
            and _boolean(selected_ranking_rows["within_one_standard_error"]).all()
            and (
                pd.to_numeric(
                    selected_ranking_rows["display_rank"], errors="coerce"
                ).eq(1).all()
                if "display_rank" in selected_ranking_rows
                else False
            )
        )
        ranking_pair_objective = bool(
            "objective" in rankings
            and rankings["objective"].astype(str).eq("pair_net_cagr").all()
            and "portfolio_objective_used" in rankings
            and (~_boolean(rankings["portfolio_objective_used"])).all()
        )
        ranking_costs = _number(rankings, ("primary_transaction_cost_bps",))
        ranking_primary_cost = bool(
            np.isfinite(ranking_costs).all()
            and np.allclose(ranking_costs, PRIMARY_COST_BPS)
        )
        parameter_columns = {
            "zscore_method",
            "zscore_lookback",
            "zscore_min_periods",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
            "is_baseline",
        }
        parameter_grid_protocol = False
        if parameter_columns.issubset(rankings):
            method = rankings["zscore_method"].astype(str).str.lower()
            lookback = pd.to_numeric(rankings["zscore_lookback"], errors="coerce")
            minimum = pd.to_numeric(rankings["zscore_min_periods"], errors="coerce")
            entry = pd.to_numeric(rankings["entry_z"], errors="coerce")
            exit_value = pd.to_numeric(rankings["exit_z"], errors="coerce")
            holding = pd.to_numeric(
                rankings["maximum_holding_days"], errors="coerce"
            )
            is_baseline = _boolean(rankings["is_baseline"])
            baseline_rows = rankings.loc[is_baseline]
            grid_rows = rankings.loc[~is_baseline].copy()
            expected_minimum = lookback.map(
                {10.0: 10.0, 20.0: 16.0, 40.0: 32.0, 60.0: 48.0}
            )
            baseline_exact = bool(
                baseline_rows.groupby(ranking_keys, dropna=False).size().eq(1).all()
                and method.loc[is_baseline].eq("sma").all()
                and lookback.loc[is_baseline].eq(60).all()
                and minimum.loc[is_baseline].eq(40).all()
                and entry.loc[is_baseline].eq(2.0).all()
                and exit_value.loc[is_baseline].eq(0.5).all()
                and holding.loc[is_baseline].eq(60).all()
            )
            grid_values_valid = bool(
                method.loc[~is_baseline].isin({"sma", "ewma"}).all()
                and lookback.loc[~is_baseline].isin({10, 20, 40, 60}).all()
                and minimum.loc[~is_baseline].eq(
                    expected_minimum.loc[~is_baseline]
                ).all()
                and entry.loc[~is_baseline].isin({1.25, 1.75, 2.25}).all()
                and exit_value.loc[~is_baseline].isin({0.25, 0.5}).all()
                and holding.loc[~is_baseline].isin({10, 30}).all()
            )
            combination_columns = [
                "zscore_method",
                "zscore_lookback",
                "zscore_min_periods",
                "entry_z",
                "exit_z",
                "maximum_holding_days",
            ]
            distinct_combinations = grid_rows[
                ranking_keys + combination_columns
            ].drop_duplicates()
            unique_grid = bool(
                grid_rows.groupby(ranking_keys, dropna=False).size().eq(96).all()
                and distinct_combinations.groupby(
                    ranking_keys, dropna=False
                ).size().eq(96).all()
            )
            parameter_grid_protocol = bool(
                baseline_exact and grid_values_valid and unique_grid
            )
        if (
            not schedule.empty
            and len(selected_ranking_rows)
            and {"selected_config_id", *ranking_keys}.issubset(schedule)
        ):
            winner_rows = selected_ranking_rows[
                ranking_keys + ["config_id"]
            ].rename(columns={"config_id": "ranking_config_id"})
            schedule_winners = schedule[
                ranking_keys + ["selected_config_id"]
            ].merge(
                winner_rows,
                on=ranking_keys,
                how="left",
                validate="one_to_one",
            )
            schedule_winners_match_rankings = bool(
                schedule_winners["ranking_config_id"].notna().all()
                and schedule_winners["selected_config_id"].astype(str).eq(
                    schedule_winners["ranking_config_id"].astype(str)
                ).all()
            )
        else:
            schedule_winners_match_rankings = schedule.empty
    else:
        hyperparameter_grid_complete = False
        exactly_one_pair_winner = False
        winner_inside_one_se_set = False
        ranking_pair_objective = False
        ranking_primary_cost = False
        parameter_grid_protocol = False
        schedule_winners_match_rankings = False

    if successful_count:
        metric_costs = _number(metrics, ("transaction_cost_bps",))
        primary_metrics = bool(np.isfinite(metric_costs).all() and np.allclose(metric_costs, PRIMARY_COST_BPS))
        start_flat = bool(_boolean(metrics["start_flat"]).all()) if "start_flat" in metrics else False
        terminal_flat = bool(_boolean(metrics["terminal_flat"]).all()) if "terminal_flat" in metrics else False
        terminal_liquidation = bool(
            _boolean(metrics["terminal_liquidation"]).all()
        ) if "terminal_liquidation" in metrics else False
        observations = _number(metrics, ("observations",))
        betas = _number(metrics, ("beta",))
        drawdowns = _number(metrics, ("max_drawdown",))
        exposures = _number(metrics, ("exposure_fraction",))
        turnovers = _number(metrics, ("annualized_turnover",))
        transaction_sums = _number(metrics, ("transaction_cost_sum",))
        borrow_sums = _number(metrics, ("borrow_cost_sum",))
        trade_counts = _number(metrics, ("trade_count",))
        metric_values_sane = bool(
            np.isfinite(observations).all()
            and observations.gt(0).all()
            and np.isfinite(betas).all()
            and betas.gt(0).all()
            and np.isfinite(drawdowns).all()
            and drawdowns.between(-1.0, 0.0, inclusive="both").all()
            and np.isfinite(exposures).all()
            and exposures.between(0.0, 1.0, inclusive="both").all()
            and np.isfinite(turnovers).all()
            and turnovers.ge(0.0).all()
            and np.isfinite(transaction_sums).all()
            and transaction_sums.ge(0.0).all()
            and np.isfinite(borrow_sums).all()
            and borrow_sums.ge(0.0).all()
            and np.isfinite(trade_counts).all()
            and trade_counts.ge(0.0).all()
            and np.allclose(trade_counts, np.rint(trade_counts))
        )
    else:
        primary_metrics = selected_count == 0
        start_flat = selected_count == 0
        terminal_flat = selected_count == 0
        terminal_liquidation = selected_count == 0
        metric_values_sane = selected_count == 0

    if not signals.empty:
        signal_costs = _number(signals, ("transaction_cost_bps",))
        primary_signals = bool(np.isfinite(signal_costs).all() and np.allclose(signal_costs, PRIMARY_COST_BPS))
        unique_keys = [name for name in ("rebalance_date", "pair", "date") if name in signals]
        unique_signal_rows = bool(len(unique_keys) == 3 and not signals.duplicated(unique_keys).any())
        if {"date", "rebalance_date", "outer_end_exclusive"}.issubset(signals):
            oos_bounded = bool(
                (
                    (pd.to_datetime(signals["date"]) >= pd.to_datetime(signals["rebalance_date"]))
                    & (pd.to_datetime(signals["date"]) < pd.to_datetime(signals["outer_end_exclusive"]))
                ).all()
            )
        else:
            oos_bounded = False
    else:
        primary_signals = selected_count == 0
        unique_signal_rows = selected_count == 0
        oos_bounded = selected_count == 0

    daily_metric_reconciliation = selected_count == 0
    trade_ledger_reconciliation = selected_count == 0
    if (
        successful_count
        and len(natural_keys) == 2
        and all(key in signals for key in natural_keys)
        and "trade_count" in metrics
    ):
        daily_checks: list[bool] = []
        required_daily = {
            "net_return",
            "turnover",
            "transaction_cost",
            "borrow_cost",
        }
        for _, metric in metrics.iterrows():
            daily = signals.copy()
            for key in natural_keys:
                if key == "rebalance_date":
                    daily = daily.loc[
                        pd.to_datetime(daily[key]) == pd.Timestamp(metric[key])
                    ]
                else:
                    daily = daily.loc[daily[key].astype(str).eq(str(metric[key]))]
            if "date" in daily:
                daily = daily.sort_values("date")
            if daily.empty or not required_daily.issubset(daily.columns):
                daily_checks.append(False)
                continue
            net_return = pd.to_numeric(daily["net_return"], errors="coerce")
            turnover = pd.to_numeric(daily["turnover"], errors="coerce")
            transaction_cost = pd.to_numeric(
                daily["transaction_cost"], errors="coerce"
            )
            borrow_cost = pd.to_numeric(daily["borrow_cost"], errors="coerce")
            if not all(
                np.isfinite(values).all()
                for values in (net_return, turnover, transaction_cost, borrow_cost)
            ):
                daily_checks.append(False)
                continue
            equity = (1.0 + net_return).cumprod()
            total_return = float(equity.iloc[-1] - 1.0)
            cagr = (
                float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0)
                if len(equity) >= 2 and equity.iloc[-1] > 0
                else math.nan
            )
            max_drawdown = float((equity / equity.cummax() - 1.0).min())
            expected = {
                "observations": float(len(daily)),
                "net_total_return": total_return,
                "net_cagr": cagr,
                "max_drawdown": max_drawdown,
                "annualized_turnover": float(turnover.mean() * 252.0),
                "transaction_cost_sum": float(transaction_cost.sum()),
                "borrow_cost_sum": float(borrow_cost.sum()),
            }
            reconciled = True
            for name, value in expected.items():
                stored = pd.to_numeric(
                    pd.Series([metric.get(name, np.nan)]), errors="coerce"
                ).iloc[0]
                reconciled = reconciled and bool(
                    np.isfinite(stored)
                    and np.isfinite(value)
                    and np.isclose(stored, value, rtol=1e-9, atol=1e-12)
                )
            if "net_equity" in daily:
                stored_equity = pd.to_numeric(
                    daily["net_equity"], errors="coerce"
                )
                reconciled = reconciled and bool(
                    np.isfinite(stored_equity).all()
                    and np.allclose(
                        stored_equity.to_numpy(dtype=float),
                        equity.to_numpy(dtype=float),
                        rtol=1e-9,
                        atol=1e-12,
                    )
                )
            daily_checks.append(reconciled)
        daily_metric_reconciliation = bool(
            len(daily_checks) == successful_count and all(daily_checks)
        )

        metric_trade_counts = metrics[natural_keys + ["trade_count"]].copy()
        metric_trade_counts["trade_count"] = pd.to_numeric(
            metric_trade_counts["trade_count"], errors="coerce"
        )
        if trades.empty:
            observed_trade_counts = pd.DataFrame(columns=natural_keys + ["ledger_count"])
        elif all(key in trades for key in natural_keys):
            observed_trade_counts = (
                trades.groupby(natural_keys, dropna=False)
                .size()
                .rename("ledger_count")
                .reset_index()
            )
        else:
            observed_trade_counts = pd.DataFrame()
        if not observed_trade_counts.empty or trades.empty:
            trade_reconciliation = metric_trade_counts.merge(
                observed_trade_counts,
                on=natural_keys,
                how="left",
                validate="one_to_one",
            )
            trade_reconciliation["ledger_count"] = pd.to_numeric(
                trade_reconciliation["ledger_count"], errors="coerce"
            ).fillna(0.0)
            trade_ledger_reconciliation = bool(
                np.isfinite(trade_reconciliation["trade_count"]).all()
                and np.allclose(
                    trade_reconciliation["trade_count"],
                    trade_reconciliation["ledger_count"],
                    rtol=0.0,
                    atol=0.0,
                )
            )

    if successful_count:
        expected_cost_keys = [name for name in ("rebalance_date", "pair") if name in metrics]
        cost_keys = [name for name in ("rebalance_date", "pair") if name in costs]
        has_sensitivity = bool(
            not costs.empty
            and len(cost_keys) == 2
            and "transaction_cost_bps" in costs
        )
        if has_sensitivity:
            cost_sets = costs.groupby(cost_keys)["transaction_cost_bps"].apply(
                lambda values: set(pd.to_numeric(values, errors="coerce").dropna())
            )
            has_sensitivity = bool(
                len(cost_sets)
                and cost_sets.map(lambda values: {SENSITIVITY_COST_BPS, PRIMARY_COST_BPS}.issubset(values)).all()
            )
        frozen_winner = bool(
            "frozen_winner" in costs and _boolean(costs["frozen_winner"]).all()
        )
        selection_cost = _number(costs, ("selection_cost_bps",))
        frozen_selection_cost = bool(
            len(selection_cost)
            and np.isfinite(selection_cost).all()
            and np.allclose(selection_cost, PRIMARY_COST_BPS)
        )
        if len(expected_cost_keys) == 2 and len(cost_keys) == 2:
            metric_key_set = set(map(tuple, metrics[expected_cost_keys].astype(str).to_numpy()))
            cost_key_set = set(map(tuple, costs[cost_keys].astype(str).to_numpy()))
            cost_coverage = metric_key_set == cost_key_set
        else:
            cost_coverage = False
        primary_cost_reconciles = False
        sensitivity_return_monotonic = False
        sensitivity_config_frozen = False
        if has_sensitivity and cost_coverage:
            cost_numeric = pd.to_numeric(
                costs["transaction_cost_bps"], errors="coerce"
            )
            primary_rows = costs.loc[np.isclose(cost_numeric, PRIMARY_COST_BPS)].copy()
            sensitivity_rows = costs.loc[
                np.isclose(cost_numeric, SENSITIVITY_COST_BPS)
            ].copy()
            comparison_fields = (
                "net_total_return",
                "net_cagr",
                "max_drawdown",
                "annualized_turnover",
                "trade_count",
                "transaction_cost_sum",
                "borrow_cost_sum",
            )
            if (
                not primary_rows.duplicated(cost_keys).any()
                and not sensitivity_rows.duplicated(cost_keys).any()
                and len(primary_rows) == successful_count
                and len(sensitivity_rows) == successful_count
            ):
                primary_join = metrics.merge(
                    primary_rows,
                    on=expected_cost_keys,
                    how="left",
                    validate="one_to_one",
                    suffixes=("_metric", "_cost"),
                )
                primary_cost_reconciles = True
                for field in comparison_fields:
                    metric_name = f"{field}_metric"
                    cost_name = f"{field}_cost"
                    if metric_name not in primary_join or cost_name not in primary_join:
                        primary_cost_reconciles = False
                        break
                    metric_value = pd.to_numeric(
                        primary_join[metric_name], errors="coerce"
                    )
                    cost_value = pd.to_numeric(
                        primary_join[cost_name], errors="coerce"
                    )
                    primary_cost_reconciles = primary_cost_reconciles and bool(
                        np.isfinite(metric_value).all()
                        and np.isfinite(cost_value).all()
                        and np.allclose(
                            metric_value,
                            cost_value,
                            rtol=1e-9,
                            atol=1e-12,
                        )
                    )
                scenario_join = sensitivity_rows.merge(
                    primary_rows,
                    on=cost_keys,
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_1bp", "_2bp"),
                )
                low_cost_return = _number(
                    scenario_join,
                    ("net_total_return_1bp",),
                )
                high_cost_return = _number(
                    scenario_join,
                    ("net_total_return_2bp",),
                )
                sensitivity_return_monotonic = bool(
                    len(scenario_join) == successful_count
                    and np.isfinite(low_cost_return).all()
                    and np.isfinite(high_cost_return).all()
                    and (low_cost_return + 1e-12 >= high_cost_return).all()
                )
                sensitivity_config_frozen = bool(
                    "config_id_1bp" in scenario_join
                    and "config_id_2bp" in scenario_join
                    and scenario_join["config_id_1bp"].astype(str).eq(
                        scenario_join["config_id_2bp"].astype(str)
                    ).all()
                )
    else:
        has_sensitivity = selected_count == 0
        frozen_winner = selected_count == 0
        frozen_selection_cost = selected_count == 0
        cost_coverage = selected_count == 0
        primary_cost_reconciles = selected_count == 0
        sensitivity_return_monotonic = selected_count == 0
        sensitivity_config_frozen = selected_count == 0

    model_audit = tables["model_audit"]
    fold_train_end = next(
        (name for name in ("model_train_end", "train_end", "training_end") if name in folds),
        None,
    )
    if not folds.empty and fold_train_end is not None and "validation_start" in folds:
        fold_chronology = bool(
            (
                pd.to_datetime(folds[fold_train_end])
                < pd.to_datetime(folds["validation_start"])
            ).all()
        )
        fold_start_flat = bool(
            "start_flat" in folds and _boolean(folds["start_flat"]).all()
        )
        fold_terminal_flat = bool(
            "terminal_flat" in folds
            and _boolean(folds["terminal_flat"]).all()
        )
        fold_grid_keys = [
            name
            for name in ("rebalance_date", "pair", "inner_fold")
            if name in folds
        ]
        inner_fold_grid_complete = bool(
            len(fold_grid_keys) == 3
            and "config_id" in folds
            and folds.groupby(fold_grid_keys, dropna=False).size().eq(97).all()
            and folds.groupby(fold_grid_keys, dropna=False)["config_id"]
            .nunique()
            .eq(97)
            .all()
        )
    else:
        fold_chronology = selected_count == 0
        fold_start_flat = selected_count == 0
        fold_terminal_flat = selected_count == 0
        inner_fold_grid_complete = selected_count == 0

    purge_source = model_audit if "purged_observations" in model_audit else folds
    if not purge_source.empty:
        purge = _number(purge_source, ("purged_observations", "purge_size"))
        one_session_purge = bool(
            len(purge)
            and np.isfinite(purge).all()
            and purge.eq(1).all()
        )
    else:
        one_session_purge = selected_count == 0

    if not vintage.empty and "portfolio_objective_used" in vintage:
        pair_only_objective = bool((~_boolean(vintage["portfolio_objective_used"])).all())
    else:
        pair_only_objective = selected_count == 0 and required_present

    if not vintage.empty and not rebalance.empty and "rebalance_date" in vintage and "rebalance_date" in rebalance:
        all_rebalances_summarized = set(pd.to_datetime(rebalance["rebalance_date"])) == set(
            pd.to_datetime(vintage["rebalance_date"])
        )
    else:
        all_rebalances_summarized = bool(rebalance.empty and vintage.empty and required_present)

    embedded_baseline_return = _number(
        metrics,
        ("baseline_total_return", "baseline_net_total_return"),
    )
    tuned_return = _number(metrics, ("net_total_return", "total_return"))
    baseline_available = bool(
        not baseline.empty or np.isfinite(embedded_baseline_return).any()
    )
    baseline_comparable = np.isfinite(embedded_baseline_return) & np.isfinite(
        tuned_return
    )
    baseline_comparable_count = int(baseline_comparable.sum())
    baseline_tuned_win_rate = (
        float(
            (tuned_return.loc[baseline_comparable]
             > embedded_baseline_return.loc[baseline_comparable]).mean()
        )
        if baseline_comparable_count
        else None
    )
    if baseline_available and successful_count:
        baseline_primary_cost = True
        if not baseline.empty:
            baseline_cost = _number(baseline, ("transaction_cost_bps",))
            baseline_primary_cost = bool(
                np.isfinite(baseline_cost).all()
                and np.allclose(baseline_cost, PRIMARY_COST_BPS)
            )
        selection_markers: list[pd.Series] = []
        if "baseline_used_for_selection" in metrics:
            selection_markers.append(_boolean(metrics["baseline_used_for_selection"]))
        if "baseline_used_for_selection" in baseline:
            selection_markers.append(_boolean(baseline["baseline_used_for_selection"]))
        baseline_not_selected = bool(
            selection_markers and not any(marker.any() for marker in selection_markers)
        )
        stored_delta = _number(
            metrics,
            ("tuned_minus_baseline_total_return",),
        )
        baseline_delta_reconciles = bool(
            baseline_comparable_count
            and np.isfinite(stored_delta.loc[baseline_comparable]).all()
            and np.allclose(
                stored_delta.loc[baseline_comparable].to_numpy(dtype=float),
                (
                    tuned_return.loc[baseline_comparable]
                    - embedded_baseline_return.loc[baseline_comparable]
                ).to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
        baseline_values_reconcile = True
        baseline_complete_coverage = False
        if not baseline.empty and len(natural_keys) == 2 and all(name in baseline for name in natural_keys):
            baseline_key_set = set(map(tuple, baseline[natural_keys].astype(str).to_numpy()))
            metric_key_set = set(map(tuple, metrics[natural_keys].astype(str).to_numpy()))
            baseline_same_vintages = bool(
                baseline_key_set.issubset(metric_key_set)
                and not baseline.duplicated(natural_keys).any()
            )
            baseline_complete_coverage = baseline_key_set == metric_key_set
            if baseline_same_vintages:
                baseline_join = metrics.merge(
                    baseline,
                    on=natural_keys,
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_tuned", "_baseline"),
                )
                for embedded_name, artifact_name in (
                    ("baseline_total_return", "net_total_return_baseline"),
                    ("baseline_net_cagr", "net_cagr_baseline"),
                ):
                    if embedded_name not in baseline_join or artifact_name not in baseline_join:
                        baseline_values_reconcile = False
                        break
                    embedded_value = pd.to_numeric(
                        baseline_join[embedded_name], errors="coerce"
                    )
                    artifact_value = pd.to_numeric(
                        baseline_join[artifact_name], errors="coerce"
                    )
                    baseline_values_reconcile = baseline_values_reconcile and bool(
                        np.isfinite(embedded_value).all()
                        and np.isfinite(artifact_value).all()
                        and np.allclose(
                            embedded_value,
                            artifact_value,
                            rtol=1e-9,
                            atol=1e-12,
                        )
                    )
            else:
                baseline_values_reconcile = False
        else:
            baseline_same_vintages = bool(baseline_comparable_count)
            baseline_complete_coverage = bool(
                baseline_comparable_count == successful_count
            )
            baseline_values_reconcile = bool(baseline_comparable_count)
    else:
        baseline_primary_cost = True
        baseline_not_selected = True
        baseline_delta_reconciles = True
        baseline_same_vintages = True
        baseline_complete_coverage = True
        baseline_values_reconcile = True
    baseline_integrity = bool(
        not baseline_available
        or (
            baseline_primary_cost
            and baseline_not_selected
            and baseline_same_vintages
            and baseline_values_reconcile
            and baseline_delta_reconciles
        )
    )

    charts_exist = int(sum(path.is_file() for path in chart_paths))
    charts_decodable = int(sum(_chart_decodable(path) for path in chart_paths))
    checks: dict[str, Any] = {
        "quality_scope": "Implementation, chronology, eligibility, cost-freeze, and artifact invariants only; not a profitability or deployment pass.",
        "statistical_deployment_evidence_pass": False,
        "source_artifact_presence": source_presence,
        "required_source_artifacts_present": required_present,
        "rebalance_rows": int(len(rebalance)),
        "selected_pair_vintages": selected_count,
        "successful_oos_pair_vintages": successful_count,
        "selected_schedule_primary_keys_unique": schedule_keys_unique,
        "pair_vintage_metric_primary_keys_unique": metrics_keys_unique,
        "successful_metrics_are_selected_pair_vintages": metrics_match_selected_schedule,
        "oos_config_matches_selected_config": selected_config_matches,
        "pair_ranks_are_consecutive_with_maximum_ten": consecutive_pair_ranks,
        "each_tuned_pair_has_exactly_97_parameter_candidates": hyperparameter_grid_complete,
        "parameter_candidates_match_predeclared_96_cell_grid_plus_baseline": parameter_grid_protocol,
        "each_tuned_pair_has_exactly_one_winner": exactly_one_pair_winner,
        "selected_winner_is_inside_one_standard_error_set": winner_inside_one_se_set,
        "parameter_rankings_use_pair_net_cagr_only": ranking_pair_objective,
        "parameter_rankings_primary_cost_is_2bps": ranking_primary_cost,
        "selected_schedule_winners_match_parameter_rankings": schedule_winners_match_rankings,
        "all_selected_pairs_tradable_eligible": strict_only,
        "all_selected_pairs_pass_5pct_bh_fdr": fdr_only,
        "quota_or_exploratory_fallback_count": int(
            0
            if schedule.empty or "selection_role" not in schedule
            else schedule["selection_role"].astype("string").str.contains("fallback|quota|exploratory", case=False, regex=True, na=False).sum()
        ),
        "no_quota_or_exploratory_fallback": no_fallback,
        "maximum_ten_pairs_per_rebalance": maximum_ten,
        "formation_ends_before_outer_test": formation_lagged,
        "primary_schedule_cost_is_2bps": schedule_cost,
        "primary_metrics_cost_is_2bps": primary_metrics,
        "outer_metric_values_sane": metric_values_sane,
        "primary_signals_cost_is_2bps": primary_signals,
        "daily_signals_reconcile_to_pair_vintage_metrics": daily_metric_reconciliation,
        "trade_ledger_counts_reconcile_to_pair_vintage_metrics": trade_ledger_reconciliation,
        "frozen_cost_sensitivity_contains_1bp_and_2bp": has_sensitivity,
        "cost_sensitivity_uses_frozen_winner": frozen_winner,
        "cost_sensitivity_selection_cost_is_2bps": frozen_selection_cost,
        "cost_sensitivity_covers_successful_vintages": cost_coverage,
        "primary_2bp_cost_rows_reconcile_to_metrics": primary_cost_reconciles,
        "lower_1bp_cost_does_not_reduce_return": sensitivity_return_monotonic,
        "cost_sensitivity_config_is_frozen": sensitivity_config_frozen,
        "outer_simulations_start_flat": start_flat,
        "outer_simulations_end_flat": terminal_flat,
        "outer_simulations_force_terminal_liquidation": terminal_liquidation,
        "outer_signal_rows_unique": unique_signal_rows,
        "outer_signal_dates_inside_vintage": oos_bounded,
        "inner_fold_chronology_pass": fold_chronology,
        "inner_fold_simulations_start_flat": fold_start_flat,
        "inner_fold_simulations_end_flat": fold_terminal_flat,
        "each_inner_fold_has_exactly_97_parameter_evaluations": inner_fold_grid_complete,
        "inner_fold_purge_is_exactly_one_session": one_session_purge,
        "portfolio_objective_used": not pair_only_objective,
        "pair_level_objective_only": pair_only_objective,
        "all_rebalances_have_vintage_summary": all_rebalances_summarized,
        "baseline_comparator_available": baseline_available,
        "baseline_comparable_pair_vintages": baseline_comparable_count,
        "tuned_oos_baseline_win_rate": baseline_tuned_win_rate,
        "baseline_comparator_cost_is_2bps": baseline_primary_cost,
        "baseline_not_used_for_selection": baseline_not_selected,
        "baseline_rows_are_tuned_pair_vintages": baseline_same_vintages,
        "baseline_covers_all_successful_pair_vintages": baseline_complete_coverage,
        "baseline_artifact_values_reconcile": baseline_values_reconcile,
        "tuned_minus_baseline_return_reconciles": baseline_delta_reconciles,
        "optional_baseline_comparator_integrity_pass": baseline_integrity,
        "report_exists": report_path.is_file(),
        "report_chart_count": charts_exist,
        "report_chart_decodable_count": charts_decodable,
        "expected_report_chart_count": int(len(chart_paths)),
        "credentials_serialized": False,
    }
    critical = [
        required_present,
        schedule_keys_unique,
        metrics_keys_unique,
        metrics_match_selected_schedule,
        selected_config_matches,
        consecutive_pair_ranks,
        hyperparameter_grid_complete,
        parameter_grid_protocol,
        exactly_one_pair_winner,
        winner_inside_one_se_set,
        ranking_pair_objective,
        ranking_primary_cost,
        schedule_winners_match_rankings,
        strict_only,
        fdr_only,
        no_fallback,
        maximum_ten,
        formation_lagged,
        schedule_cost,
        primary_metrics,
        metric_values_sane,
        primary_signals,
        daily_metric_reconciliation,
        trade_ledger_reconciliation,
        has_sensitivity,
        frozen_winner,
        frozen_selection_cost,
        cost_coverage,
        primary_cost_reconciles,
        sensitivity_return_monotonic,
        sensitivity_config_frozen,
        start_flat,
        terminal_flat,
        terminal_liquidation,
        unique_signal_rows,
        oos_bounded,
        fold_chronology,
        fold_start_flat,
        fold_terminal_flat,
        inner_fold_grid_complete,
        one_session_purge,
        pair_only_objective,
        all_rebalances_summarized,
        baseline_integrity,
        report_path.is_file(),
        charts_exist == len(chart_paths),
        charts_decodable == len(chart_paths),
    ]
    checks["implementation_invariants_pass"] = bool(all(critical))
    checks["critical_pass"] = bool(all(critical))
    checks["status"] = "pass" if all(critical) else "incomplete_or_failed"
    return checks


REPORT_TEMPLATE = Template(
    r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 · Pair-optimized 10-year / rolling 3-year study</title>
<style>
:root{--ink:#162033;--muted:#647086;--blue:#2f67c7;--teal:#148a82;--green:#26845d;--orange:#d9792d;--red:#bf4c55;--line:#dce3ec;--pale:#eef3f8;--paper:#fff}
*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#edf1f6;color:var(--ink);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{width:100%;max-width:1320px;min-width:0;margin:24px auto;background:var(--paper);box-shadow:0 18px 55px rgba(30,45,70,.12)}header{padding:58px 7% 46px;background:linear-gradient(125deg,#12213b,#274f83);color:#fff}header>*{min-width:0;overflow-wrap:anywhere}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:12px;font-weight:750;color:#bcd3f2}h1{font-size:43px;line-height:1.08;margin:13px 0 15px;max-width:970px}header p{font-size:18px;color:#dce8f7;max-width:1020px;margin:0}.content{min-width:0;padding:46px 7% 72px}.content>*,.grid2>*{min-width:0}.verdict{padding:20px 23px;border-left:5px solid var(--orange);background:#fff8ef;margin-bottom:28px}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:13px;margin:0 0 38px}.card{min-width:0;border:1px solid var(--line);padding:17px;background:#fff}.card .k{text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:10px;font-weight:750}.card .v{font-size:27px;font-weight:780;margin:3px 0}.card .s{font-size:12px;color:var(--muted)}h2{font-size:28px;line-height:1.2;margin:48px 0 15px;border-top:1px solid var(--line);padding-top:31px}h3{font-size:18px;margin:22px 0 9px}.lead{font-size:17px;color:#354258}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.chart{width:100%;display:block;border:1px solid var(--line);margin:18px 0 25px;background:#fff}.note{padding:16px 19px;background:var(--pale);border-left:4px solid var(--blue);margin:18px 0}.note.warn{background:#fff8ef;border-color:var(--orange)}.note.bad{background:#fff2f3;border-color:var(--red)}.note.good{background:#eef8f4;border-color:var(--green)}.scroll{width:100%;max-width:100%;min-width:0;overflow-x:auto;border:1px solid var(--line);margin:12px 0 24px}.data-table{border-collapse:collapse;width:100%;font-size:12px}.data-table th{background:#f2f5f9;text-align:left;padding:9px 10px;white-space:nowrap;border-bottom:1px solid var(--line)}.data-table td{padding:8px 10px;border-bottom:1px solid #e7ecf2;white-space:nowrap}.muted,.small{color:var(--muted)}.small{font-size:12px}code{background:#eef2f7;padding:2px 5px;border-radius:3px}a{color:var(--blue)}ul,ol{padding-left:21px}.pill{display:inline-block;padding:5px 9px;border:1px solid #cdd8e6;border-radius:20px;font-size:11px;margin:2px 5px 2px 0;background:#f8fafc}.foot{margin-top:42px;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
@media(max-width:960px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.grid2{grid-template-columns:1fr}h1{font-size:35px}}@media(max-width:540px){.page{margin:0}.content{padding:32px 5% 50px}header{padding:42px 5% 34px}.cards{grid-template-columns:1fr}h1{font-size:31px}}@media print{@page{size:A4 landscape;margin:10mm}body{background:#fff}.page{box-shadow:none;margin:0;max-width:none}.chart,.note,.card{break-inside:avoid}}
</style></head><body><main class="page"><header><div class="eyebrow">Equity pairs · V6 · generated {{ generated }}</div>
<h1>Ten years of data. Three calendar years of training. Every return belongs to an actual outer pair-vintage.</h1>
<p>This additive report tunes each strict pair independently and evaluates its frozen rule in the next calendar quarter. It does not optimize or report an aggregate portfolio CAGR.</p></header><section class="content">
<div class="verdict"><strong>{{ verdict_title }}</strong> {{ verdict_text }}</div>
<div class="cards">
<div class="card"><div class="k">Strict candidate vintages</div><div class="v">{{ strict_candidates }}</div><div class="s">5% within-sector BH-FDR plus strict gates</div></div>
<div class="card"><div class="k">Selected pair-vintages</div><div class="v">{{ selected_count }}</div><div class="s">maximum ten per rebalance; fewer is valid</div></div>
<div class="card"><div class="k">2 bp median OOS return</div><div class="v">{{ primary_median }}</div><div class="s">actual next-quarter pair sleeves</div></div>
<div class="card"><div class="k">Positive OOS vintages</div><div class="v">{{ positive_fraction }}</div><div class="s">not a significance test</div></div>
<div class="card"><div class="k">1 bp frozen sensitivity</div><div class="v">{{ sensitivity_median }}</div><div class="s">same winners; no retuning</div></div>
</div>

<h2>1 · What this report measures</h2>
<p class="lead">The unit of evidence is a <strong>pair-vintage</strong>: one strict pair selected from a lagged three-calendar-year formation set, one pair-specific frozen signal rule, and its immediately following calendar-quarter return. Each sleeve starts from $1 and one-times gross. Results are not combined into a capital allocation.</p>
<p>The primary replay charges 2 bp per dollar traded on one-way leg turnover, including launch, hedge drift and forced terminal liquidation; the 1 bp case only reprices the already-frozen winner. Both include the separate 30 bp annual short-borrow proxy.</p>
<div><span class="pill">pair-level net CAGR tuning</span><span class="pill">2 bp primary</span><span class="pill">1 bp frozen sensitivity</span><span class="pill">30 bp annual borrow proxy</span><span class="pill">next-session signals</span><span class="pill">terminal liquidation</span></div>
<div class="note warn"><strong>No fallback means no forced ten.</strong> Tradability requires the 5% within-sector BH-FDR rule and every stored strict gate. Quota fills and exploratory fallbacks are prohibited. A quarter with zero eligible or successfully tuned pairs remains empty; the report does not impute a return.</div>

<h2>2 · Strict candidate supply</h2>
<img class="chart" src="{{ strict_supply_chart }}" alt="Strict candidate supply by rebalance">
<div class="scroll">{{ screen_table | safe }}</div>
<p class="small">Candidate counts repeat a pair when it qualifies at multiple formation dates; they are pair-vintages, not independent discoveries.</p>

<h2>3 · Pair-specific hyperparameters</h2>
<img class="chart" src="{{ parameter_chart }}" alt="Selected parameter stability">
<p>Each pair searches the predeclared 96-cell SMA/EWMA grid plus an unchanged baseline. The one-standard-error rule favors lower validation-fold dispersion, lower turnover, shallower drawdown, higher pooled validation net CAGR and then the deterministic parameter ID. Other pairs' returns never enter the objective.</p>
<p class="small">{{ tuning_note }}</p>
<div class="scroll">{{ selected_table | safe }}</div>

<h2>4 · Actual outer pair-vintage results at 2 bp</h2>
<img class="chart" src="{{ result_chart }}" alt="Training score and actual pair-vintage outer returns">
<div class="scroll">{{ metrics_table | safe }}</div>
<div class="note"><strong>Annualization warning:</strong> pair-vintage CAGR annualizes a single calendar-quarter path and can look extreme. The unannualized next-quarter net return is the primary economic number in this section.</div>

<h2>5 · Ex-post realized OOS leaderboard — descriptive only</h2>
<img class="chart" src="{{ descriptive_top10_chart }}" alt="Ranked net returns and normalized equity curves for the ten strongest realized outer pair-vintages">
<div class="scroll">{{ descriptive_top10_table | safe }}</div>
<div class="note warn"><strong>Not a prospective top-ten strategy.</strong> These rows were sorted only after their outer returns were observed. This ranking was not used for screening, hyperparameter tuning, pair selection or capital allocation. Repeated pair identities in different quarters are separate frozen pair-vintages. The rank uses unannualized next-quarter net total return; displayed CAGR is an extreme annualization of a short test.</div>
<p>{{ outer_test_note }}</p>
<p class="small"><a href="data/processed/{{ descriptive_top10_filename }}">Download the machine-readable descriptive top-ten ranking</a>. The anti-leakage columns explicitly mark it as ex-post and non-deployable.</p>

<h2>6 · Frozen 1–2 bp sensitivity</h2>
<img class="chart" src="{{ cost_chart }}" alt="Frozen winner cost sensitivity">
<div class="scroll">{{ cost_table | safe }}</div>
<p>The 1 bp result reprices the pair and parameters selected under the primary 2 bp assumption. It is not a second search and cannot change membership.</p>

<h2>7 · Tuned versus baseline</h2>
<div class="scroll">{{ baseline_table | safe }}</div>
<p class="small">{{ baseline_note }}</p>

<h2>8 · Trading frequency</h2>
<img class="chart" src="{{ trade_chart }}" alt="Pair-vintage trade frequency">
<div class="scroll">{{ trade_table | safe }}</div>
<p class="small">A trade is a pair entry episode, not an underlying-leg order. Launch, hedge drift, exit and terminal liquidation can each generate leg turnover.</p>

<h2>9 · Training-selected vintage signal examples</h2>
<p>Distinct-pair examples below are chosen only by the stored <em>training CV</em> score, before consulting their OOS return. Their displayed equity is actual OOS performance. They are illustrations, not an ex-post deployable leaderboard.</p>
{{ dashboards | safe }}

<h2>10 · QA and evidence limits</h2>
<div class="grid2"><div><h3>Implementation checks</h3><div class="scroll">{{ quality_table | safe }}</div></div><div><h3>Known evidence limits</h3>
<ul><li>The universe is the 2026-08-26 current-member S&amp;P 500 snapshot with current GICS labels. Ten years of prices does not cure survivorship, delisting or historical-classification bias.</li>
<li>Yahoo adjusted daily closes are not executable quotes. They omit intraday paths and observed bid/ask spreads.</li>
<li>The 1–2 bp flat rate is optimistic and excludes nonlinear impact, locates and security-specific borrow. The separate 30 bp annual borrow charge is still a proxy.</li>
<li>Repeated quarters, shared names and overlapping training histories make pair-vintage observations dependent.</li>
<li>The hyperparameter search creates winner's-curse risk. Chronological folds and the one-standard-error rule reduce, but do not remove, data snooping.</li>
<li>Once viewed, these outer quarters are spent research evidence. Deployment requires point-in-time membership and a new paper-trading interval.</li></ul></div></div>
<div class="note {{ quality_class }}"><strong>QA status: {{ quality_status }}.</strong> A passing quality file means artifact and implementation invariants passed; it is not a profitability, statistical-credibility or deployment approval.</div>

<h2>11 · Machine-readable artifacts</h2>
<div class="grid2"><div><ul>{{ artifact_links | safe }}</ul></div><div><ul><li><a href="pair_v6_quality_checks.json">Quality checks</a></li><li><a href="pair_v6_manifest.json">Report manifest and hashes</a></li></ul></div></div>
<div class="foot">Generated entirely from saved V6 global artifacts. No pair was selected or retuned by this report generator; the realized-return leaderboard is explicitly ex-post and descriptive. Quantitative research only; not investment advice.</div>
</section></main></body></html>"""
)


def _display_screen(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [name for name in ("rebalance_date", "formation_start", "formation_end", "training_sessions", "candidate_pairs", "fdr_significant", "strict_eligible", "strict_sectors") if name in frame]
    result = frame[columns].copy() if columns else pd.DataFrame()
    for column in result.columns.intersection(["rebalance_date", "formation_start", "formation_end"]):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.date.astype("string")
    return result


def _display_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        name
        for name in (
            "rebalance_date",
            "pair_rank",
            "pair",
            "sector",
            "selected_config_id",
            "zscore_method",
            "zscore_lookback",
            "zscore_min_periods",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
            "training_cv_net_cagr",
            "validation_trade_count",
            "full_training_trade_count",
            "active_fold_count",
        )
        if name in frame
    ]
    result = frame[columns].copy() if columns else pd.DataFrame()
    if "rebalance_date" in result:
        result["rebalance_date"] = pd.to_datetime(result["rebalance_date"], errors="coerce").dt.date.astype("string")
    if "training_cv_net_cagr" in result:
        result["training_cv_net_cagr"] = result["training_cv_net_cagr"].map(_pct)
    return result


def _display_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        name
        for name in (
            "rebalance_date",
            "pair_rank",
            "pair",
            "sector",
            "config_id",
            "observations",
            "net_total_return",
            "net_cagr",
            "max_drawdown",
            "trade_count",
            "exposure_fraction",
            "annualized_turnover",
            "transaction_cost_sum",
            "borrow_cost_sum",
        )
        if name in frame
    ]
    result = frame[columns].copy() if columns else pd.DataFrame()
    if "rebalance_date" in result:
        result["rebalance_date"] = pd.to_datetime(result["rebalance_date"], errors="coerce").dt.date.astype("string")
    for column in result.columns.intersection(["net_total_return", "net_cagr", "max_drawdown", "exposure_fraction"]):
        result[column] = result[column].map(_pct)
    for column in result.columns.intersection(["annualized_turnover", "transaction_cost_sum", "borrow_cost_sum"]):
        result[column] = result[column].map(lambda value: _num(value, 3))
    return result


def _display_descriptive_top10(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        name
        for name in (
            "descriptive_rank",
            "pair",
            "rebalance_date",
            "evaluation_start",
            "evaluation_end",
            "observations",
            "net_total_return",
            "net_cagr",
            "max_drawdown",
            "trade_count",
            "training_cv_net_cagr",
            "config_id",
        )
        if name in frame
    ]
    result = frame[columns].copy() if columns else pd.DataFrame()
    for column in result.columns.intersection(
        ["rebalance_date", "evaluation_start", "evaluation_end"]
    ):
        result[column] = pd.to_datetime(
            result[column], errors="coerce", format="mixed"
        ).dt.date.astype("string")
    for column in result.columns.intersection(
        [
            "net_total_return",
            "net_cagr",
            "max_drawdown",
            "training_cv_net_cagr",
        ]
    ):
        result[column] = result[column].map(_pct)
    if "pair" in result:
        result["pair"] = result["pair"].astype("string").str.replace(
            "__", " / ", regex=False
        )
    return result.rename(
        columns={
            "descriptive_rank": "rank",
            "rebalance_date": "vintage_start",
            "observations": "test_sessions",
            "net_total_return": "net_test_return",
            "net_cagr": "annualized_cagr",
            "training_cv_net_cagr": "training_cv_cagr",
        }
    )


def _display_trades(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [name for name in ("rebalance_date", "pair", "entry_date", "exit_date", "direction", "trading_days", "entry_zscore", "exit_zscore", "exit_reason", "net_return") if name in frame]
    result = frame[columns].copy() if columns else pd.DataFrame()
    for column in result.columns.intersection(["rebalance_date", "entry_date", "exit_date"]):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.date.astype("string")
    if "net_return" in result:
        result["net_return"] = result["net_return"].map(_pct)
    return result


def _quality_table(quality: Mapping[str, Any]) -> pd.DataFrame:
    excluded = {"source_artifact_presence", "quality_scope"}
    rows = []
    for key, value in quality.items():
        if key in excluded:
            continue
        if isinstance(value, (dict, list)):
            continue
        rows.append({"Check": key.replace("_", " "), "Value": value})
    return pd.DataFrame(rows)


def _render_report(
    report_path: Path,
    tables: dict[str, pd.DataFrame],
    quality: Mapping[str, Any],
    charts: dict[str, Path],
    dashboards: list[tuple[pd.Series, Path]],
    descriptive_top10: pd.DataFrame,
) -> None:
    screen = tables["screen_summary"]
    schedule = tables["selected_schedule"]
    metrics = tables["pair_vintage_metrics"]
    costs = tables["cost_sensitivity"]
    trades = tables["oos_trades"]
    vintage = tables["vintage_summary"]
    strict_count = int(_number(screen, ("strict_eligible",)).fillna(0.0).sum()) if not screen.empty else 0
    tuned_count = int(
        _number(vintage, ("tuned_pair_count",)).fillna(0.0).sum()
    ) if not vintage.empty else 0
    tuning_failure_count = int(
        _number(vintage, ("tuning_failure_count",)).fillna(0.0).sum()
    ) if not vintage.empty else 0
    primary_returns = _number(metrics, ("net_total_return", "total_return")).dropna()
    sensitivity = costs.copy()
    if not sensitivity.empty and "transaction_cost_bps" in sensitivity:
        sensitivity = sensitivity.loc[
            pd.to_numeric(sensitivity["transaction_cost_bps"], errors="coerce").eq(SENSITIVITY_COST_BPS)
        ]
    sensitivity_returns = _number(sensitivity, ("net_total_return", "total_return")).dropna()
    all_observations = _number(metrics, ("observations",)).dropna()
    top_observations = _number(descriptive_top10, ("observations",)).dropna()
    if len(all_observations):
        outer_test_note = (
            "Each outer test is the immediately following calendar quarter, "
            f"not one fixed seven-year holdout. Successful pair-vintages contain "
            f"{int(all_observations.min())}–{int(all_observations.max())} trading "
            f"sessions (median {int(all_observations.median())}); the minimum is "
            "the final as-of-truncated quarter. "
        )
        if len(top_observations):
            outer_test_note += (
                f"The ten rows above each contain {int(top_observations.min())}–"
                f"{int(top_observations.max())} sessions. "
            )
        outer_test_note += (
            "The 63-session inner validation blocks sit inside the preceding "
            "three-year training window and are not outer tests."
        )
    else:
        outer_test_note = (
            "The intended outer test is the immediately following calendar "
            "quarter; no successful pair-vintage observation count is available."
        )
    baseline_table, baseline_note = _baseline_comparison(metrics)
    dashboard_html = "".join(
        f'<h3>{str(row.get("pair", "Pair"))} · {pd.Timestamp(row.get("rebalance_date")).date() if pd.notna(row.get("rebalance_date")) else "vintage"}</h3>'
        f'<img class="chart" src="{path.relative_to(report_path.parent).as_posix()}" alt="Training-selected pair-vintage z-score signals and equity">'
        for row, path in dashboards
    )
    if not dashboard_html:
        dashboard_html = '<div class="note">No selected pair-vintage signal example is available.</div>'
    links = "".join(
        f'<li><a href="data/processed/{filename}">{name.replace("_", " ").title()}</a></li>'
        for name, filename in TABLE_FILES.items()
        if name in REQUIRED_TABLES
        or name
        in {
            "strict_candidates",
            "validation_cost_sensitivity",
            "oos_baseline_metrics",
            "oos_baseline_signals",
        }
    )
    passed = bool(quality.get("critical_pass", False))
    html = REPORT_TEMPLATE.render(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        verdict_title="Implementation artifacts pass QA." if passed else "The V6 artifact set is incomplete or failed an invariant.",
        verdict_text=(
            "The report below contains actual saved pair-vintage results, but a QA pass is not evidence of profitable alpha."
            if passed
            else "Available tables are shown without filling missing vintages or inventing returns; consult the QA section before interpreting performance."
        ),
        strict_candidates=_integer(strict_count),
        selected_count=_integer(len(schedule)),
        primary_median=_pct(primary_returns.median()),
        positive_fraction=_pct((primary_returns > 0).mean()) if len(primary_returns) else "—",
        sensitivity_median=_pct(sensitivity_returns.median()),
        tuning_note=(
            f"Across {strict_count:,} strict candidate vintages, {tuned_count:,} "
            f"completed pair-specific tuning and {tuning_failure_count:,} failed "
            "closed without a replacement; the per-quarter top-ten cap explains "
            "the remaining difference from the selected count."
        ),
        strict_supply_chart=charts["strict_supply"].relative_to(report_path.parent).as_posix(),
        parameter_chart=charts["parameter_stability"].relative_to(report_path.parent).as_posix(),
        result_chart=charts["pair_vintage_results"].relative_to(report_path.parent).as_posix(),
        descriptive_top10_chart=charts["descriptive_top10"].relative_to(report_path.parent).as_posix(),
        descriptive_top10_table=_table(
            _display_descriptive_top10(descriptive_top10), max_rows=10
        ),
        descriptive_top10_filename=DESCRIPTIVE_TOP10_FILENAME,
        outer_test_note=outer_test_note,
        cost_chart=charts["cost_sensitivity"].relative_to(report_path.parent).as_posix(),
        trade_chart=charts["trade_frequency"].relative_to(report_path.parent).as_posix(),
        screen_table=_table(_display_screen(screen), max_rows=80),
        selected_table=_table(_display_schedule(schedule), max_rows=300),
        metrics_table=_table(_display_metrics(metrics), max_rows=300),
        cost_table=_table(_cost_summary(costs), max_rows=10),
        baseline_table=_table(baseline_table, max_rows=10),
        baseline_note=baseline_note,
        trade_table=_table(_display_trades(trades), max_rows=60),
        dashboards=dashboard_html,
        quality_table=_table(_quality_table(quality), max_rows=80),
        quality_class="good" if passed else "bad",
        quality_status="PASS" if passed else "INCOMPLETE / FAILED",
        artifact_links=links,
    )
    _write_text(report_path, html)


def _build_manifest(
    root: Path,
    report_path: Path,
    quality_path: Path,
    config_path: Path,
    source_paths: Mapping[str, Path],
    chart_paths: list[Path],
    derived_paths: list[Path] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for name, path in source_paths.items():
        if path.is_file():
            sources[name] = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        else:
            sources[name] = {"path": path.relative_to(root).as_posix(), "missing": True}
    generated_paths = [report_path, quality_path, *chart_paths, *(derived_paths or [])]
    generated = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in generated_paths
        if path.is_file()
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "pair_optimized_10y_3y_v6",
        "primary_evidence_unit": "pair_vintage",
        "portfolio_cagr_computed": False,
        "selection_or_tuning_performed_by_report": False,
        "descriptive_ex_post_ranking_generated": True,
        "descriptive_ex_post_ranking_used_for_selection": False,
        "primary_transaction_cost_bps": PRIMARY_COST_BPS,
        "frozen_sensitivity_transaction_cost_bps": SENSITIVITY_COST_BPS,
        "strict_selection": "5% within-sector BH-FDR plus all strict gates; no quota or exploratory fallback",
        "current_member_survivorship_bias": True,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path) if config_path.is_file() else None,
        },
        "source_artifacts": sources,
        "generated_artifacts": generated,
        "credentials_included": False,
    }


def generate_report(
    output_root: str | Path = DEFAULT_OUTPUT,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    maximum_dashboards: int = 4,
) -> dict[str, Any]:
    """Generate report, charts, QA JSON and manifest from saved V6 tables."""

    if maximum_dashboards < 0:
        raise ValueError("maximum_dashboards cannot be negative")
    root = Path(output_root).resolve()
    charts_dir = root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    tables, source_paths = load_report_inputs(root)
    report_path = root / "pair_optimized_10y_3y_report.html"
    quality_path = root / "pair_v6_quality_checks.json"
    manifest_path = root / "pair_v6_manifest.json"
    descriptive_top10_path = (
        root / "data" / "processed" / DESCRIPTIVE_TOP10_FILENAME
    )
    descriptive_top10 = _descriptive_top_pair_vintages(
        tables["pair_vintage_metrics"], tables["selected_schedule"], maximum=10
    )
    _write_csv(descriptive_top10_path, descriptive_top10)

    charts = {
        "strict_supply": _plot_strict_supply(
            tables["screen_summary"], charts_dir / "pair_v6_strict_supply.png"
        ),
        "pair_vintage_results": _plot_pair_vintage_results(
            tables["pair_vintage_metrics"],
            tables["selected_schedule"],
            charts_dir / "pair_v6_pair_vintage_results.png",
        ),
        "parameter_stability": _plot_parameter_stability(
            tables["selected_schedule"],
            charts_dir / "pair_v6_parameter_stability.png",
        ),
        "trade_frequency": _plot_trade_frequency(
            tables["pair_vintage_metrics"],
            tables["oos_trades"],
            charts_dir / "pair_v6_trade_frequency.png",
        ),
        "cost_sensitivity": _plot_cost_sensitivity(
            tables["cost_sensitivity"],
            charts_dir / "pair_v6_cost_sensitivity.png",
        ),
        "descriptive_top10": _plot_descriptive_top10(
            descriptive_top10,
            tables["oos_signals"],
            charts_dir / "pair_v6_descriptive_top10_oos_performance.png",
        ),
    }
    dashboards: list[tuple[pd.Series, Path]] = []
    leaders = _leading_training_vintages(
        tables["selected_schedule"], maximum_dashboards
    )
    for _, row in leaders.iterrows():
        date_token = (
            pd.Timestamp(row["rebalance_date"]).strftime("%Y%m%d")
            if pd.notna(row.get("rebalance_date"))
            else "unknown"
        )
        path = charts_dir / f"pair_v6_signal_{date_token}_{_safe_token(row.get('pair'))}.png"
        dashboards.append(
            (row, _plot_pair_dashboard(row, tables["oos_signals"], path))
        )
    all_chart_paths = [*charts.values(), *(path for _, path in dashboards)]

    # First render gives the physical report needed by the final QA check.
    preliminary = _quality_checks(
        tables, source_paths, all_chart_paths, report_path
    )
    _render_report(
        report_path,
        tables,
        preliminary,
        charts,
        dashboards,
        descriptive_top10,
    )
    quality = _quality_checks(tables, source_paths, all_chart_paths, report_path)
    _write_json(quality_path, quality)
    _render_report(
        report_path,
        tables,
        quality,
        charts,
        dashboards,
        descriptive_top10,
    )

    config = Path(config_path)
    manifest = _build_manifest(
        root,
        report_path,
        quality_path,
        config,
        source_paths,
        all_chart_paths,
        [descriptive_top10_path],
    )
    _write_json(manifest_path, manifest)
    return {
        "report": report_path,
        "quality_checks": quality_path,
        "manifest": manifest_path,
        "quality": quality,
        "charts": all_chart_paths,
        "descriptive_top10": descriptive_top10_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the V6 pair-vintage report from saved global artifacts."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maximum-dashboards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_report(
        args.output,
        config_path=args.config,
        maximum_dashboards=args.maximum_dashboards,
    )
    print(
        json.dumps(
            {
                "report": str(result["report"]),
                "quality_checks": str(result["quality_checks"]),
                "manifest": str(result["manifest"]),
                "critical_pass": bool(result["quality"]["critical_pass"]),
                "selected_pair_vintages": int(
                    result["quality"]["selected_pair_vintages"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
