#!/usr/bin/env python3
"""Replay every frozen V6 pair winner and test its predictive information.

This is an additive, post-hoc diagnostic.  It does not retune hyperparameters,
change screening, or overwrite the original V6 report.  The 56 successfully
tuned strict pair-vintages that ranked below the top-ten cutoff are replayed so
the IC is not estimated from a range-restricted winner-only sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from equity_pairs.data import load_timeseries_snapshot
from equity_pairs.grid_search import ParameterSet
from equity_pairs.pair_tuning import (
    PairDefinition,
    evaluate_frozen_pair_parameter,
    protocol_parameter_sets,
)
from equity_pairs.predictive_diagnostics import (
    aggregate_unique_pairs,
    quarterly_information_coefficients,
    selection_mechanism_by_quarter,
    summarize_conditional_returns,
    summarize_information_coefficients,
)
from scripts.run_pair_optimized_10y_research import (
    _build_pair_tuning_protocol,
    _load_protocol,
)


DEFAULT_SOURCE = Path("outputs/pair_return_optimization_10y_train3y_2026-08-26")
DEFAULT_OUTPUT = Path("outputs/pair_predictive_diagnostics_10y_train3y_2026-08-26")
DEFAULT_CONFIG = Path("configs/pair_optimized_10y_3y.json")
PRIMARY_COST_BPS = 2.0
SENSITIVITY_COST_BPS = 1.0

INK = "#172033"
MUTED = "#657188"
BLUE = "#2e66c5"
TEAL = "#138b83"
GREEN = "#2b855f"
ORANGE = "#d77a2d"
RED = "#bd4d57"
PURPLE = "#7459b4"
PALE = "#eef3f8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#ced7e3",
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
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="raise", format="mixed")
    return result


def _bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _canonical_winners(
    source: Path,
    config: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, ParameterSet], Any]:
    processed = source / "data" / "processed"
    rankings = _dates(
        pd.read_csv(processed / "pair_v6_parameter_rankings.csv.gz"),
        (
            "rebalance_date",
            "formation_start",
            "formation_end",
            "outer_start",
            "outer_end_exclusive",
        ),
    )
    strict = _dates(
        pd.read_csv(processed / "pair_v6_strict_candidates.csv.gz"),
        ("rebalance_date",),
    )
    selected_schedule = _dates(
        pd.read_csv(processed / "pair_v6_selected_parameter_schedule.csv"),
        ("rebalance_date", "outer_end_exclusive"),
    )
    schedule = _dates(
        pd.read_csv(processed / "pair_v6_rebalance_schedule.csv"),
        ("rebalance_date", "outer_end_exclusive"),
    )
    group_sizes = rankings.groupby(["rebalance_date", "pair"])["config_id"].nunique()
    if not group_sizes.eq(97).all():
        raise ValueError("every tuned pair-vintage must contain exactly 97 candidates")
    winners = rankings.loc[_bool(rankings["selected"])].copy()
    winner_counts = winners.groupby(["rebalance_date", "pair"]).size()
    if len(winners) != 174 or not winner_counts.eq(1).all():
        raise ValueError("expected one frozen winner for each of 174 tuned pair-vintages")

    protocol = _load_protocol(config)
    spec, base_config = _build_pair_tuning_protocol(protocol)
    parameters = {
        value.config_id: value
        for value in protocol_parameter_sets(spec, stop_z=base_config.stop_z)
    }
    if len(parameters) != 97:
        raise ValueError("the canonical V6 grid no longer contains 97 parameters")
    for row in winners.itertuples(index=False):
        if row.config_id not in parameters:
            raise ValueError(f"unknown frozen config {row.config_id}")
        canonical = parameters[row.config_id]
        comparisons = {
            "zscore_method": str(row.zscore_method) == canonical.zscore_method,
            "zscore_lookback": int(row.zscore_lookback) == canonical.zscore_lookback,
            "zscore_min_periods": int(row.zscore_min_periods) == canonical.zscore_min_periods,
            "entry_z": np.isclose(float(row.entry_z), canonical.entry_z),
            "exit_z": np.isclose(float(row.exit_z), canonical.exit_z),
            "maximum_holding_days": int(row.maximum_holding_days)
            == canonical.maximum_holding_days,
            "is_baseline": bool(row.is_baseline) == canonical.is_baseline,
        }
        if not all(comparisons.values()):
            failed = ", ".join(name for name, valid in comparisons.items() if not valid)
            raise ValueError(f"{row.config_id} disagrees with canonical grid: {failed}")

    orientation = strict[
        [
            "rebalance_date",
            "pair",
            "sector",
            "dependent",
            "independent",
            "pair_pvalue",
            "fdr_qvalue",
            "half_life_days",
            "hurst_exponent",
        ]
    ].copy()
    winners = winners.merge(
        orientation,
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_strict"),
    )
    if winners[["dependent", "independent"]].isna().any().any():
        raise ValueError("strict dependent/independent orientation did not join")
    chosen_keys = selected_schedule[
        ["rebalance_date", "pair", "pair_rank", "selected_config_id"]
    ].copy()
    chosen_keys["selected_top10"] = True
    winners = winners.merge(
        chosen_keys,
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
    )
    winners["selected_top10"] = _bool(winners["selected_top10"])
    winners = winners.rename(
        columns={
            "pooled_net_cagr": "training_score",
            "max_drawdown": "training_max_drawdown",
        }
    )
    ranked_parts: list[pd.DataFrame] = []
    for _, group in winners.groupby("rebalance_date", sort=True):
        group = group.sort_values(
            ["training_score", "training_max_drawdown", "pair"],
            ascending=[False, False, True],
            kind="mergesort",
        ).copy()
        group["training_rank_all"] = np.arange(1, len(group) + 1)
        ranked_parts.append(group)
    winners = pd.concat(ranked_parts, ignore_index=True)
    reconstructed = set(
        map(
            tuple,
            winners.loc[winners["training_rank_all"].le(10), ["rebalance_date", "pair"]]
            .to_numpy(),
        )
    )
    original = set(map(tuple, chosen_keys[["rebalance_date", "pair"]].to_numpy()))
    if reconstructed != original or int(winners["selected_top10"].sum()) != 118:
        raise ValueError("reconstructed training top-ten does not match the V6 schedule")
    return winners, strict, schedule, parameters, base_config


def _replay_all(
    winners: pd.DataFrame,
    prices: pd.DataFrame,
    parameters: dict[str, ParameterSet],
    base_config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    signal_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, row in enumerate(winners.itertuples(index=False), start=1):
        train_index = prices.index[
            (prices.index >= pd.Timestamp(row.formation_start))
            & (prices.index <= pd.Timestamp(row.formation_end))
        ]
        outer_index = prices.index[
            (prices.index >= pd.Timestamp(row.outer_start))
            & (prices.index < pd.Timestamp(row.outer_end_exclusive))
        ]
        definition = PairDefinition(
            pair=str(row.pair),
            sector=str(row.sector_strict),
            dependent=str(row.dependent),
            independent=str(row.independent),
        )
        parameter = parameters[str(row.config_id)]
        context = {
            "rebalance_date": pd.Timestamp(row.rebalance_date),
            "outer_end_exclusive": pd.Timestamp(row.outer_end_exclusive),
            "pair": str(row.pair),
            "sector": str(row.sector_strict),
            "dependent": str(row.dependent),
            "independent": str(row.independent),
            "config_id": str(row.config_id),
            "training_score": float(row.training_score),
            "training_max_drawdown": float(row.training_max_drawdown),
            "training_rank_all": int(row.training_rank_all),
            "selected_top10": bool(row.selected_top10),
            "original_pair_rank": float(row.pair_rank) if np.isfinite(row.pair_rank) else math.nan,
            "screen_pair_pvalue": float(row.pair_pvalue),
            "screen_fdr_qvalue": float(row.fdr_qvalue),
            "screen_half_life_days": float(row.half_life_days),
            "screen_hurst_exponent": float(row.hurst_exponent),
            "posthoc_diagnostic_only": True,
        }
        try:
            primary = evaluate_frozen_pair_parameter(
                definition,
                prices,
                train_index,
                outer_index,
                parameter,
                base_config,
                transaction_cost_bps=PRIMARY_COST_BPS,
                minimum_hedge_observations=100,
                terminal_liquidation=True,
            )
            sensitivity = evaluate_frozen_pair_parameter(
                definition,
                prices,
                train_index,
                outer_index,
                parameter,
                base_config,
                transaction_cost_bps=SENSITIVITY_COST_BPS,
                minimum_hedge_observations=100,
                terminal_liquidation=True,
            )
        except Exception as error:
            failures.append(
                {
                    **context,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        metric_rows.append({**context, **primary.metrics})
        for simulation in (primary, sensitivity):
            sensitivity_rows.append(
                {
                    **context,
                    "transaction_cost_bps": float(simulation.strategy_config.transaction_cost_bps),
                    **simulation.metrics,
                }
            )
        daily = primary.daily.reset_index()
        for key, value in context.items():
            if key not in daily:
                daily[key] = value
        signal_parts.append(daily)
        if len(primary.trades):
            trades = primary.trades.copy()
            for key, value in context.items():
                if key not in trades:
                    trades[key] = value
            trade_parts.append(trades)
        if number % 25 == 0 or number == len(winners):
            print(f"replayed {number}/{len(winners)} frozen pair-vintages", flush=True)
    metrics = pd.DataFrame(metric_rows)
    signals = pd.concat(signal_parts, ignore_index=True) if signal_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    cost_sensitivity = pd.DataFrame(sensitivity_rows)
    failure_frame = pd.DataFrame(failures)
    return metrics, signals, trades, cost_sensitivity, failure_frame


def _reconcile_original(source: Path, replay: pd.DataFrame) -> dict[str, Any]:
    original = _dates(
        pd.read_csv(source / "data" / "processed" / "pair_v6_pair_vintage_metrics.csv"),
        ("rebalance_date",),
    )
    selected = replay.loc[replay["selected_top10"].astype(bool)].copy()
    columns = ["net_total_return", "net_cagr", "alpha", "beta", "trade_count"]
    comparison = original[["rebalance_date", "pair", *columns]].merge(
        selected[["rebalance_date", "pair", *columns]],
        on=["rebalance_date", "pair"],
        suffixes=("_original", "_replay"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    differences: dict[str, float] = {}
    for column in columns:
        differences[column] = float(
            (
                pd.to_numeric(comparison[f"{column}_original"], errors="coerce")
                - pd.to_numeric(comparison[f"{column}_replay"], errors="coerce")
            )
            .abs()
            .max()
        )
    return {
        "row_count_original": int(len(original)),
        "row_count_replay_selected": int(len(selected)),
        "all_keys_match": bool(comparison["_merge"].eq("both").all()),
        "maximum_absolute_differences": differences,
        "reconciles_at_1e_12": bool(
            comparison["_merge"].eq("both").all()
            and all(value <= 1e-12 for value in differences.values())
        ),
    }


def _baseline_advantage(source: Path, replay: pd.DataFrame) -> pd.DataFrame:
    """Build the selected-rule versus unchanged-baseline persistence diagnostic."""

    processed = source / "data" / "processed"
    rankings = _dates(
        pd.read_csv(processed / "pair_v6_parameter_rankings.csv.gz"),
        ("rebalance_date",),
    )
    baseline_training = rankings.loc[_bool(rankings["is_baseline"])][
        ["rebalance_date", "pair", "pooled_net_cagr"]
    ].rename(columns={"pooled_net_cagr": "baseline_training_score"})
    baseline_oos = _dates(
        pd.read_csv(processed / "pair_v6_oos_baseline_metrics.csv"),
        ("rebalance_date",),
    )[["rebalance_date", "pair", "net_total_return"]].rename(
        columns={"net_total_return": "baseline_oos_return"}
    )
    chosen = replay.loc[replay["selected_top10"].astype(bool)][
        [
            "rebalance_date",
            "pair",
            "training_score",
            "net_total_return",
            "trade_count",
        ]
    ].copy()
    result = chosen.merge(
        baseline_training,
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
    ).merge(
        baseline_oos,
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
    )
    if result[["baseline_training_score", "baseline_oos_return"]].isna().any().any():
        raise ValueError("baseline advantage diagnostic has an incomplete join")
    result["training_score_advantage"] = (
        result["training_score"] - result["baseline_training_score"]
    )
    result["oos_return_advantage"] = (
        result["net_total_return"] - result["baseline_oos_return"]
    )
    for column in ("training_score_advantage", "oos_return_advantage"):
        result.loc[result[column].abs().le(1e-12), column] = 0.0
    result["tuned_outperformed_baseline"] = result["oos_return_advantage"].gt(1e-12)
    return result


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_training_vs_oos(
    metrics: pd.DataFrame,
    ic_summary: pd.DataFrame,
    path: Path,
) -> Path:
    frame = metrics.copy()
    frame["score_pct_rank"] = frame.groupby("rebalance_date")["training_score"].rank(
        method="average", pct=True
    )
    frame["return_pct_rank"] = frame.groupby("rebalance_date")["net_total_return"].rank(
        method="average", pct=True
    )
    eligible_dates = set(
        frame.groupby("rebalance_date").size().loc[lambda value: value >= 5].index
    )
    eligible = frame.loc[frame["rebalance_date"].isin(eligible_dates)]
    summary = ic_summary.loc[
        ic_summary["scope"].eq("all_tuned_strict")
        & ic_summary["minimum_cross_section_size"].eq(5)
    ].iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    omitted = frame.loc[~frame["selected_top10"].astype(bool)]
    selected = frame.loc[frame["selected_top10"].astype(bool)]
    axes[0].scatter(
        omitted["training_score"] * 100,
        omitted["net_total_return"] * 100,
        s=32,
        facecolors="white",
        edgecolors=TEAL,
        linewidths=1.1,
        alpha=0.75,
        label="Tuned, below top-ten cutoff (56)",
    )
    axes[0].scatter(
        selected["training_score"] * 100,
        selected["net_total_return"] * 100,
        s=34,
        color=BLUE,
        alpha=0.70,
        label="Original training top ten (118)",
    )
    finite = frame[["training_score", "net_total_return"]].dropna()
    if len(finite) >= 2:
        slope, intercept = np.polyfit(
            finite["training_score"] * 100,
            finite["net_total_return"] * 100,
            1,
        )
        xline = np.linspace(
            finite["training_score"].min() * 100,
            finite["training_score"].max() * 100,
            100,
        )
        axes[0].plot(xline, intercept + slope * xline, color=INK, linewidth=1.4)
    axes[0].axhline(0, color="#aeb9c8", linewidth=1)
    axes[0].set_title("Raw pooled relationship is positive")
    axes[0].set_xlabel("Training CV net CAGR (%)")
    axes[0].set_ylabel("Next-quarter OOS net total return (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[0].text(
        0.02,
        0.97,
        f"Pooled Spearman ρ = {summary['raw_pooled_spearman']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        color=INK,
        fontweight="bold",
    )

    axes[1].scatter(
        eligible["score_pct_rank"],
        eligible["return_pct_rank"],
        c=np.where(eligible["selected_top10"], BLUE, TEAL),
        s=34,
        alpha=0.68,
    )
    bins = pd.cut(eligible["score_pct_rank"], np.linspace(0, 1, 6), include_lowest=True)
    binned = eligible.groupby(bins, observed=True)[["score_pct_rank", "return_pct_rank"]].mean()
    axes[1].plot(
        binned["score_pct_rank"],
        binned["return_pct_rank"],
        color=ORANGE,
        marker="o",
        linewidth=2.0,
        label="Training-rank quintile mean",
    )
    axes[1].plot([0, 1], [0, 1], color="#bcc6d3", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Quarter-normalized ranks · n ≥ 5")
    axes[1].set_xlabel("Within-quarter training-score percentile")
    axes[1].set_ylabel("Within-quarter OOS-return percentile")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].text(
        0.02,
        0.97,
        "Mean quarterly IC\n"
        f"{summary['mean_quarterly_spearman_ic']:.3f} "
        f"[{summary['block_bootstrap_95ci_low']:.3f}, "
        f"{summary['block_bootstrap_95ci_high']:.3f}]",
        transform=axes[1].transAxes,
        va="top",
        color=INK,
        fontweight="bold",
    )
    figure.suptitle(
        "Does the training score rank the next quarter?",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(figure, path)


def _plot_quarterly_ic(
    quarterly: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(13.2, 5.2))
    styles = {
        "all_tuned_strict": (TEAL, "All 174 tuned strict pair-vintages"),
        "original_top10": (BLUE, "Original 118 training top-ten sleeves"),
    }
    for scope, (color, label) in styles.items():
        frame = quarterly.loc[quarterly["scope"].eq(scope)].sort_values("rebalance_date")
        valid = frame.loc[frame["spearman_ic"].notna()]
        axis.plot(
            valid["rebalance_date"],
            valid["spearman_ic"],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            alpha=0.9,
            label=label,
        )
        eligible = valid.loc[valid["pair_vintage_count"].ge(5)]
        axis.scatter(
            eligible["rebalance_date"],
            eligible["spearman_ic"],
            s=35 + eligible["pair_vintage_count"] * 2,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        mean = summary.loc[
            summary["scope"].eq(scope)
            & summary["minimum_cross_section_size"].eq(5),
            "mean_quarterly_spearman_ic",
        ].iloc[0]
        axis.axhline(mean, color=color, linestyle="--", linewidth=1, alpha=0.65)
    axis.axhline(0, color="#8f9cad", linewidth=1.1)
    axis.set_ylim(-1.08, 1.08)
    axis.set_ylabel("Quarterly Spearman IC")
    axis.set_title("IC varies sharply quarter to quarter; larger dots have n ≥ 5")
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(axis="y", alpha=0.16)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_mechanism(mechanism: pd.DataFrame, path: Path) -> Path:
    frame = mechanism.sort_values("rebalance_date").copy()
    active = frame.loc[frame["tuned_strict_count"].gt(0)]
    comparisons = frame.loc[frame["selected_minus_nonselected"].notna()]
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.2), sharex=False)
    axes[0].plot(
        frame["rebalance_date"],
        frame["all_pair_mean_return"] * 100,
        color=TEAL,
        marker="o",
        markersize=4,
        label="All strict+tuned pair mean",
    )
    axes[0].plot(
        frame["rebalance_date"],
        frame["selected_mean_return"] * 100,
        color=BLUE,
        marker="o",
        markersize=4,
        label="Training top-ten mean",
    )
    axes[0].axhline(0, color="#9aa7b7", linewidth=1)
    axes[0].set_ylabel("Mean pair-vintage return (%)")
    axes[0].set_title(
        f"Screened mechanism by quarter · {len(active)} active, {len(frame) - len(active)} cash-only"
    )
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(frameon=False, ncol=2, loc="best")

    colors = np.where(comparisons["selected_minus_nonselected"].ge(0), GREEN, RED)
    locations = np.arange(len(comparisons))
    axes[1].bar(
        locations,
        comparisons["selected_minus_nonselected"] * 100,
        color=colors,
        alpha=0.86,
    )
    # Matplotlib/strftime has no portable quarter token; replace labels explicitly.
    labels = [
        f"{date.year} Q{((date.month - 1) // 3) + 1}"
        for date in comparisons["rebalance_date"]
    ]
    axes[1].set_xticks(locations, labels, rotation=35, ha="right")
    axes[1].axhline(0, color="#8795a7", linewidth=1)
    axes[1].set_ylabel("Top ten − remaining pairs (pp)")
    axes[1].set_title(
        f"Incremental cutoff test exists in only {len(comparisons)} quarters with >10 candidates"
    )
    axes[1].grid(axis="y", alpha=0.14)
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_multi_vintage(
    selected_pairs: pd.DataFrame,
    selected_daily: pd.DataFrame,
    market_sessions: pd.DatetimeIndex,
    path: Path,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
    color_map = {
        "single_vintage": "#b8c2cf",
        "limited_2": ORANGE,
        "recurrent_3": PURPLE,
        "recurrent_4plus": TEAL,
    }
    for tier, group in selected_pairs.groupby("evidence_tier"):
        axes[0].scatter(
            group["vintage_count"],
            group["cumulative_net_return"] * 100,
            s=32 + group["vintage_count"] * 8,
            color=color_map[tier],
            alpha=0.78,
            label=tier.replace("_", " "),
        )
    for row in selected_pairs.loc[selected_pairs["vintage_count"].ge(3)].itertuples():
        axes[0].annotate(
            str(row.pair).replace("__", "/"),
            (row.vintage_count, row.cumulative_net_return * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
        )
    axes[0].axhline(0, color="#9ba8b8", linewidth=1)
    axes[0].set_xlabel("Selected OOS vintages")
    axes[0].set_ylabel("Chained net return across selected vintages (%)")
    axes[0].set_title("Most pairs have too few repeats for persistence claims")
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")

    leaders = selected_pairs.loc[selected_pairs["vintage_count"].ge(4)].sort_values(
        ["vintage_count", "cash_aware_span_cagr"], ascending=[False, False]
    )
    for pair in leaders["pair"]:
        daily = selected_daily.loc[selected_daily["pair"].eq(pair)].copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        span = market_sessions[
            (market_sessions >= daily["date"].min())
            & (market_sessions <= daily["date"].max())
        ]
        returns = pd.Series(0.0, index=span)
        returns.loc[pd.DatetimeIndex(daily["date"])] = daily["net_return"].to_numpy()
        equity = (1.0 + returns).cumprod()
        axes[1].plot(
            equity.index,
            equity,
            linewidth=2,
            label=str(pair).replace("__", "/"),
        )
    axes[1].axhline(1, color="#aab5c3", linewidth=1)
    axes[1].set_ylabel("Growth of $1; cash during selection gaps")
    axes[1].set_title("Four-pair persistence tier · at least four vintages")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_trade_sampling(metrics: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    bins = np.arange(-0.5, int(metrics["trade_count"].max()) + 1.5, 1)
    axes[0].hist(metrics["trade_count"], bins=bins, color=BLUE, alpha=0.82)
    axes[0].axvline(metrics["trade_count"].median(), color=ORANGE, linewidth=2)
    axes[0].set_xlabel("Trades in one OOS quarter")
    axes[0].set_ylabel("Pair-vintage count")
    axes[0].set_title("Each quarterly return is estimated from very few events")
    axes[1].scatter(
        metrics["trade_count"],
        metrics["net_total_return"] * 100,
        c=np.where(metrics["selected_top10"], BLUE, TEAL),
        s=34,
        alpha=0.65,
    )
    axes[1].axhline(0, color="#9ca9b8", linewidth=1)
    axes[1].set_xlabel("Trades in quarter")
    axes[1].set_ylabel("OOS net total return (%)")
    axes[1].set_title("Sparse trades imply high pair-vintage sampling variance")
    figure.tight_layout()
    return _save_figure(figure, path)


def _percent(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric * 100:.{digits}f}%" if np.isfinite(numeric) else "—"


def _number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "—"


def _table(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    display = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    return display.to_html(index=False, border=0, classes="data", escape=True)


REPORT_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive IC & multi-vintage diagnostics</title>
<style>
:root{--ink:#172033;--muted:#657188;--blue:#2e66c5;--teal:#138b83;--green:#2b855f;--orange:#d77a2d;--red:#bd4d57;--pale:#eef3f8;--line:#dce3ec;--paper:#fff}
*{box-sizing:border-box} body{margin:0;background:#f4f6f9;color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}
.page{max-width:1180px;margin:28px auto;background:var(--paper);box-shadow:0 12px 42px rgba(34,50,77,.10)}
.hero{padding:56px 64px 42px;background:linear-gradient(135deg,#172033 0%,#243c63 70%,#176f72 100%);color:#fff}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:750;color:#bcd5ef}.hero h1{font-size:39px;line-height:1.12;margin:12px 0 14px;max-width:880px}.hero p{max-width:900px;color:#dce8f5;font-size:17px}.stamp{font-size:12px;color:#b8c8db;margin-top:25px}
.content{padding:38px 64px 64px} nav{display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:35px}nav a{color:var(--blue);text-decoration:none;font-size:13px;font-weight:650}
h2{font-size:25px;margin:48px 0 13px}h3{font-size:18px;margin:30px 0 10px}.lead{font-size:18px;max-width:940px}.muted{color:var(--muted)}.small{font-size:12.5px;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{border:1px solid var(--line);border-radius:10px;padding:18px;background:#fff}.card .value{font-size:27px;font-weight:780;letter-spacing:-.02em}.card .label{font-size:12px;color:var(--muted);margin-top:5px}.card.good{border-top:4px solid var(--teal)}.card.warn{border-top:4px solid var(--orange)}
.verdict{padding:18px 21px;border-left:5px solid var(--orange);background:#fff8ef;margin:21px 0}.verdict.good{border-color:var(--teal);background:#eef9f7}.note{padding:16px 19px;background:var(--pale);border-radius:8px;margin:18px 0}.danger{padding:16px 19px;background:#fff1f2;border-left:4px solid var(--red);margin:18px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:25px}.chart{width:100%;border:1px solid var(--line);border-radius:8px;margin:14px 0 7px}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:14px 0 22px}table.data{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff}table.data th{background:#f0f4f8;text-align:left;padding:10px;white-space:nowrap}table.data td{padding:9px 10px;border-top:1px solid #e6ebf1;white-space:nowrap}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0f3f7;padding:2px 5px;border-radius:4px;font-size:.9em}.formula{font-family:Georgia,serif;font-size:18px;text-align:center;padding:17px;background:#f7f9fb;border:1px solid var(--line);border-radius:8px}.foot{border-top:1px solid var(--line);margin-top:48px;padding-top:20px;color:var(--muted);font-size:12px}
@media(max-width:820px){.page{margin:0}.hero,.content{padding:34px 23px}.cards,.grid2{grid-template-columns:1fr}.hero h1{font-size:31px}}
@media print{body{background:#fff}.page{margin:0;box-shadow:none}.hero{print-color-adjust:exact;-webkit-print-color-adjust:exact}nav{display:none}.chart{break-inside:avoid}h2{break-after:avoid}}
</style></head><body><main class="page">
<header class="hero"><div class="eyebrow">V6 additive credibility audit · next-quarter OOS</div><h1>Does the training score contain persistent predictive information?</h1><p>Quarterly Spearman IC, full strict-candidate replay, conditional expected return, and unique-pair multi-vintage persistence. Every frozen rule is charged {{ primary_cost }} bp per dollar traded plus 30 bp annual short borrow.</p><div class="stamp">Generated {{ generated }} · Yahoo adjusted close {{ data_start }}–{{ data_end }} · no original V6 artifact overwritten</div></header>
<div class="content"><nav><a href="#answer">Answer</a><a href="#ic">Quarterly IC</a><a href="#mechanism">Selection mechanism</a><a href="#pairs">Repeated pairs</a><a href="#optimizer">Optimizer attribution</a><a href="#methods">Methods & artifacts</a></nav>

<section id="answer"><h2>Bottom line</h2><p class="lead">The full replay finds a <strong>positive cross-pair ranking signal</strong>, but it does not by itself establish deployable alpha. Raw pooled Pearson and Spearman correlations are <strong>{{ pooled_pearson }}</strong> and <strong>{{ pooled_spearman }}</strong>. Across all {{ all_ic_quarters }} mathematically defined quarters, mean IC is <strong>{{ all_ic }}</strong>, its four-quarter block-bootstrap interval is <strong>{{ all_ci }}</strong>, and the two-sided within-quarter permutation p-value is <strong>{{ all_permutation_p }}</strong>. The n≥5 sensitivity is stronger: IC <strong>{{ full_ic }}</strong>, interval <strong>{{ full_ci }}</strong>, p <strong>{{ permutation_p }}</strong>.</p>
<div class="cards"><div class="card good"><div class="value">{{ pooled_spearman }}</div><div class="label">Pooled Spearman · all 174 · descriptive</div></div><div class="card good"><div class="value">{{ all_ic }}</div><div class="label">Mean quarterly IC · all defined quarters</div></div><div class="card good"><div class="value">{{ full_ic }}</div><div class="label">Mean quarterly IC · n≥5 sensitivity</div></div><div class="card warn"><div class="value">{{ repeated_count }}/{{ selected_unique }}</div><div class="label">Selected unique pairs with ≥3 vintages</div></div></div>
<div class="verdict good"><strong>Research verdict.</strong> The relationship is more than random-looking in this sample: {{ all_positive_ic }} of {{ all_ic_quarters }} defined ICs and {{ positive_ic }} of {{ ic_quarters }} n≥5 ICs are positive. Still, the all-defined permutation result sits at the 5% boundary, there are only {{ ic_quarters }} quarters with at least five candidates, training windows overlap heavily, and many sleeves contain only a few trades. Treat this as evidence for further validation—not proof of stable tradable alpha.</div>
<div class="danger"><strong>Weighting changes the economic conclusion.</strong> A uniformly sampled strict+tuned pair-vintage earned {{ pooled_mean }}, but first averaging pairs inside each active quarter gives {{ quarter_mean }}. Candidate-rich positive quarters dominate the pooled average; the selection mechanism is not yet convincingly positive through time.</div>
<img class="chart" src="{{ chart_training }}" alt="Training score versus future pair return"></section>

<section id="ic"><h2>1 · Quarterly information coefficient</h2><div class="formula">IC<sub>q</sub> = Spearman Corr(Rank(training CV net CAGR), Rank(next-quarter OOS net total return))</div><p>Total return—not quarter-annualized CAGR—is the future outcome. Each IC is calculated cross-sectionally inside one non-overlapping OOS quarter. The report shows all mathematically defined quarters, but predeclares n≥5 as the primary sensitivity because n=2 can only produce +1 or −1.</p>
<img class="chart" src="{{ chart_ic }}" alt="Quarterly Spearman information coefficients">
<h3>Aggregate IC diagnostics</h3><div class="scroll">{{ ic_table | safe }}</div>
<h3>Quarter-by-quarter audit</h3><div class="scroll">{{ quarterly_table | safe }}</div>
<p class="small">Raw pooled p-values are intentionally omitted from the verdict because pair-vintages share quarter shocks and pair identities repeat. The report uses a within-quarter permutation test and a circular four-quarter block bootstrap. Even these are sensitivity diagnostics with only {{ ic_quarters }} eligible quarters.</p></section>

<section id="mechanism"><h2>2 · Does the pair-selection mechanism earn positive OOS return?</h2><p>The estimand <code>E[R(t+1) | pair passes screening]</code> must distinguish opportunity weighting from time weighting. There were {{ strict_pass }} strict screen passes, {{ all_count }} successful tunings, and one tuning failure treated as cash in the strict-pass policy. Nine of 28 scheduled quarters had no tradable pair.</p>
<img class="chart" src="{{ chart_mechanism }}" alt="Conditional selection mechanism return by quarter">
<div class="scroll">{{ conditional_table | safe }}</div>
<div class="note"><strong>Top-ten cutoff test.</strong> A selected-versus-unselected comparison exists only in {{ cutoff_quarters }} quarters with more than ten tuned candidates. Its mean difference is {{ cutoff_mean }} with a block-bootstrap interval {{ cutoff_ci }}. That is too little independent time-series evidence for a firm cutoff claim.</div>
<h3>Trade sampling</h3><img class="chart" src="{{ chart_trades }}" alt="Trade count and pair-vintage return sampling"><p class="small">Median trades per strict+tuned pair-vintage: {{ median_trades }}; zero-trade sleeves: {{ zero_trades }}. A quarter containing one to five trades is a noisy return observation even when its chronology is genuinely OOS.</p></section>

<section id="pairs"><h2>3 · Unique-pair, multi-vintage persistence</h2><p>The original 118 selected sleeves collapse to {{ selected_unique }} unique pairs: {{ singleton_count }} appear once and only {{ repeated_count }} appear at least three times. Returns are concatenated only across that pair’s non-overlapping selected OOS quarters. Cash-gap-aware CAGR inserts zero-return market sessions between selected vintages; active-session annualized return is shown separately and is <em>not</em> called ordinary CAGR.</p>
<div class="formula">R<sub>p</sub> = ∏<sub>q,d</sub>(1 + r<sub>p,q,d</sub>) − 1 &nbsp;&nbsp;·&nbsp;&nbsp; CAGR<sub>cash span</sub> = (1 + R<sub>p</sub>)<sup>252/Nspan</sup> − 1</div>
<img class="chart" src="{{ chart_multi }}" alt="Unique pair persistence and cumulative curves">
<h3>Pairs with at least three selected vintages</h3><div class="scroll">{{ recurring_table | safe }}</div>
<p class="small">A stricter four-vintage evidence tier leaves only {{ persistent_pairs }}. Requiring three or four vintages is itself a post-hoc descriptive threshold; overlapping formation histories, common quarter shocks, and shared legs still violate independence.</p>
<h3>Evidence-count distribution</h3><div class="scroll">{{ evidence_table | safe }}</div></section>

<section id="optimizer"><h2>4 · What this says—and does not say—about hyperparameter tuning</h2><p>The full-174 IC tests the <strong>combined screen, selected-parameter score, and cross-pair ranking mechanism</strong>. It does not isolate whether the best of 97 configurations for a fixed pair persists better than that pair’s other 96 configurations.</p>
<div class="grid2"><div class="note"><strong>Chosen rule versus fixed baseline.</strong><br>Pooled Spearman of training advantage versus OOS advantage: {{ delta_rho }}.<br>Mean quarterly delta-IC (n≥5): {{ delta_ic }}.<br>Chosen rule beats baseline in {{ baseline_hit }} of selected sleeves.</div><div class="note"><strong>Pure optimizer IC still required.</strong><br>For each pair-quarter, replay all 97 frozen configurations OOS, correlate their training ranks with their OOS ranks, then aggregate those within-pair ICs by quarter. Current V6 OOS artifacts contain only the chosen rule and unchanged baseline.</div></div>
<p>Accordingly, a positive full-174 IC is evidence that higher <em>pair-level selected training scores</em> tend to rank better next quarter. It is not proof that grid search discovered a structurally persistent parameter choice.</p></section>

<section id="methods"><h2>5 · Audit trail, limitations, and downloadable evidence</h2><h3>Frozen replay contract</h3><ul><li>Three calendar years of formation data end strictly before each rebalance.</li><li>Exact dependent/independent orientation comes from that vintage’s strict screen; pair strings are never split to infer orientation.</li><li>The single preselected 97-grid winner is mapped back to the canonical parameter object; there is no retuning.</li><li>The hedge relation is refit on formation data only. The position starts flat, signals act the next session, and terminal liquidation is charged.</li><li>All {{ all_count }} winners are evaluated at 2 bp; 1 bp is saved as frozen sensitivity.</li></ul>
<h3>Important limitations</h3><ul><li>Current S&amp;P 500 membership and GICS labels create survivorship bias.</li><li>Yahoo adjusted close lacks executable quotes, historical locates, and nonlinear impact.</li><li>Only {{ active_quarters }} of 28 scheduled quarters produced a successfully tuned strict candidate.</li><li>Adjacent three-year training windows overlap by roughly 92%; quarterly OOS windows do not overlap.</li><li>Block-bootstrap and permutation results remain low-powered with a short quarterly history.</li></ul>
<h3>Artifacts</h3><div class="grid2"><ul>{{ artifact_links | safe }}</ul><ul><li><a href="predictive_diagnostics_quality_checks.json">Quality checks</a></li><li><a href="predictive_diagnostics_manifest.json">Manifest and SHA-256 provenance</a></li><li><a href="{{ source_report }}">Original V6 report (unchanged)</a></li></ul></div>
<div class="note {{ qa_class }}"><strong>QA {{ qa_status }}.</strong> The replay reconciles all original 118 selected returns to numerical precision, all 56 omitted winners evaluate successfully, daily compounding reconciles to saved metrics, and no OOS windows overlap within a pair. QA validates implementation—not economic significance.</div></section>
<div class="foot">Conditional quantitative research; not investment advice. All leaderboards and persistence tiers in this diagnostic are ex-post and must not be interpreted as deployable selections.</div>
</div></main></body></html>"""
)


def _render_report(
    path: Path,
    *,
    source: Path,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    strict: pd.DataFrame,
    schedule: pd.DataFrame,
    quarterly: pd.DataFrame,
    ic_summary: pd.DataFrame,
    conditional: pd.DataFrame,
    mechanism: pd.DataFrame,
    unique_selected: pd.DataFrame,
    baseline_advantage: pd.DataFrame,
    charts: Mapping[str, Path],
    quality: Mapping[str, Any],
    artifacts: Mapping[str, Path],
) -> None:
    primary = ic_summary.loc[
        ic_summary["scope"].eq("all_tuned_strict")
        & ic_summary["minimum_cross_section_size"].eq(5)
    ].iloc[0]
    all_defined = ic_summary.loc[
        ic_summary["scope"].eq("all_tuned_strict")
        & ic_summary["minimum_cross_section_size"].eq(2)
    ].iloc[0]
    delta = ic_summary.loc[
        ic_summary["scope"].eq("selected_rule_advantage_vs_baseline")
        & ic_summary["minimum_cross_section_size"].eq(5)
    ].iloc[0]
    selected_primary = ic_summary.loc[
        ic_summary["scope"].eq("original_top10")
        & ic_summary["minimum_cross_section_size"].eq(5)
    ].iloc[0]
    conditional_index = conditional.set_index("estimand")
    pooled = conditional_index.loc["strict_tuned_pair_vintage"]
    quarter_estimate = conditional_index.loc["screened_pair_mean_active_quarter"]
    cutoff = conditional_index.loc["selected_top10_minus_nonselected"]

    ic_display = ic_summary.loc[
        ic_summary["minimum_cross_section_size"].isin([2, 5])
    ][
        [
            "scope",
            "minimum_cross_section_size",
            "eligible_quarter_count",
            "raw_pooled_spearman",
            "mean_quarterly_spearman_ic",
            "median_quarterly_spearman_ic",
            "positive_ic_fraction",
            "block_bootstrap_95ci_low",
            "block_bootstrap_95ci_high",
            "within_quarter_permutation_two_sided_pvalue",
        ]
    ].copy()
    ic_display.columns = [
        "Scope",
        "Min n",
        "Quarters",
        "Pooled ρ",
        "Mean IC",
        "Median IC",
        "Positive IC",
        "Block CI low",
        "Block CI high",
        "Permutation p",
    ]
    for column in ["Pooled ρ", "Mean IC", "Median IC", "Block CI low", "Block CI high", "Permutation p"]:
        ic_display[column] = ic_display[column].map(lambda value: _number(value, 3))
    ic_display["Positive IC"] = ic_display["Positive IC"].map(_percent)

    quarterly_display = quarterly.pivot(
        index="rebalance_date",
        columns="scope",
        values=["pair_vintage_count", "spearman_ic"],
    ).reset_index()
    quarterly_display.columns = [
        "Quarter",
        "n all",
        "n top ten",
        "n delta",
        "IC all",
        "IC top ten",
        "IC delta",
    ]
    quarterly_display["Quarter"] = quarterly_display["Quarter"].dt.strftime("%Y-%m-%d")
    for column in ["IC all", "IC top ten", "IC delta"]:
        quarterly_display[column] = quarterly_display[column].map(lambda value: _number(value, 3))
    for column in ["n all", "n top ten", "n delta"]:
        quarterly_display[column] = quarterly_display[column].map(
            lambda value: str(int(value)) if np.isfinite(value) else "—"
        )

    conditional_display = conditional[
        [
            "estimand",
            "observation_unit",
            "observation_count",
            "mean_return",
            "median_return",
            "positive_fraction",
            "block_bootstrap_95ci_low",
            "block_bootstrap_95ci_high",
        ]
    ].copy()
    conditional_display.columns = [
        "Estimand",
        "Unit",
        "n",
        "Mean",
        "Median",
        "Positive",
        "Block CI low",
        "Block CI high",
    ]
    for column in ["Mean", "Median", "Positive", "Block CI low", "Block CI high"]:
        conditional_display[column] = conditional_display[column].map(_percent)

    recurring = unique_selected.loc[unique_selected["vintage_count"].ge(3)][
        [
            "pair",
            "vintage_count",
            "cumulative_net_return",
            "cash_aware_span_cagr",
            "cash_aware_span_sharpe",
            "chained_max_drawdown",
            "quarter_hit_rate",
            "trade_count",
            "trade_hit_rate",
            "evidence_tier",
        ]
    ].copy()
    recurring["pair"] = recurring["pair"].str.replace("__", " / ", regex=False)
    recurring.columns = [
        "Pair",
        "Vintages",
        "Chained return",
        "Cash-gap CAGR",
        "Cash-gap Sharpe",
        "Max DD",
        "Quarter hit",
        "Trades",
        "Trade hit",
        "Evidence",
    ]
    for column in ["Chained return", "Cash-gap CAGR", "Max DD", "Quarter hit", "Trade hit"]:
        recurring[column] = recurring[column].map(_percent)
    recurring["Cash-gap Sharpe"] = recurring["Cash-gap Sharpe"].map(lambda value: _number(value, 2))

    evidence = (
        unique_selected.groupby("vintage_count")["pair"]
        .count()
        .rename("Unique pairs")
        .reset_index()
        .rename(columns={"vintage_count": "Selected vintages"})
    )
    artifact_links = "".join(
        f'<li><a href="{value.relative_to(path.parent).as_posix()}">{name}</a></li>'
        for name, value in artifacts.items()
    )
    selected_unique = int(len(unique_selected))
    repeated = int(unique_selected["vintage_count"].ge(3).sum())
    singleton = int(unique_selected["vintage_count"].eq(1).sum())
    source_report = (
        Path("..")
        / source.name
        / "pair_optimized_10y_3y_report.html"
    ).as_posix()
    html = REPORT_TEMPLATE.render(
        primary_cost=_number(PRIMARY_COST_BPS, 0),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_start=prices.index.min().date(),
        data_end=prices.index.max().date(),
        all_count=len(metrics),
        all_ic=_number(all_defined["mean_quarterly_spearman_ic"], 3),
        all_ci=f"[{_number(all_defined['block_bootstrap_95ci_low'], 3)}, {_number(all_defined['block_bootstrap_95ci_high'], 3)}]",
        all_permutation_p=_number(
            all_defined["within_quarter_permutation_two_sided_pvalue"], 3
        ),
        all_positive_ic=int(
            round(
                all_defined["positive_ic_fraction"]
                * all_defined["eligible_quarter_count"]
            )
        ),
        all_ic_quarters=int(all_defined["eligible_quarter_count"]),
        full_ic=_number(primary["mean_quarterly_spearman_ic"], 3),
        full_ci=f"[{_number(primary['block_bootstrap_95ci_low'], 3)}, {_number(primary['block_bootstrap_95ci_high'], 3)}]",
        pooled_spearman=_number(primary["raw_pooled_spearman"], 3),
        pooled_pearson=_number(primary["raw_pooled_pearson"], 3),
        permutation_p=_number(primary["within_quarter_permutation_two_sided_pvalue"], 3),
        positive_ic=int(round(primary["positive_ic_fraction"] * primary["eligible_quarter_count"])),
        ic_quarters=int(primary["eligible_quarter_count"]),
        pooled_mean=_percent(pooled["mean_return"]),
        quarter_mean=_percent(quarter_estimate["mean_return"]),
        strict_pass=len(strict),
        active_quarters=int(mechanism["tuned_strict_count"].gt(0).sum()),
        cutoff_quarters=int(cutoff["observation_count"]),
        cutoff_mean=_percent(cutoff["mean_return"]),
        cutoff_ci=f"[{_percent(cutoff['block_bootstrap_95ci_low'])}, {_percent(cutoff['block_bootstrap_95ci_high'])}]",
        median_trades=_number(metrics["trade_count"].median(), 1),
        zero_trades=int(metrics["trade_count"].eq(0).sum()),
        selected_unique=selected_unique,
        repeated_count=repeated,
        singleton_count=singleton,
        persistent_pairs=int(unique_selected["vintage_count"].ge(4).sum()),
        delta_rho=_number(delta["raw_pooled_spearman"], 3),
        delta_ic=_number(delta["mean_quarterly_spearman_ic"], 3),
        baseline_hit=_percent(baseline_advantage["tuned_outperformed_baseline"].mean()),
        chart_training=charts["training"].relative_to(path.parent).as_posix(),
        chart_ic=charts["ic"].relative_to(path.parent).as_posix(),
        chart_mechanism=charts["mechanism"].relative_to(path.parent).as_posix(),
        chart_multi=charts["multi"].relative_to(path.parent).as_posix(),
        chart_trades=charts["trades"].relative_to(path.parent).as_posix(),
        ic_table=_table(ic_display),
        quarterly_table=_table(quarterly_display),
        conditional_table=_table(conditional_display),
        recurring_table=_table(recurring),
        evidence_table=_table(evidence),
        artifact_links=artifact_links,
        source_report=source_report,
        qa_class="good" if quality["critical_pass"] else "",
        qa_status="PASS" if quality["critical_pass"] else "FAIL",
    )
    _atomic_text(path, html)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--simulation-repetitions",
        type=int,
        default=20_000,
        help="quarter block-bootstrap and within-quarter permutation draws",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.simulation_repetitions < 1_000:
        raise ValueError("simulation-repetitions must be at least 1,000")
    source = args.source.resolve()
    output = args.output.resolve()
    config = args.config.resolve()
    processed = output / "data" / "processed"
    charts_dir = output / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "config": config,
        "price_snapshot": source / "data" / "raw" / "yahoo_adjusted_close_10y.csv",
        "price_metadata": source / "data" / "raw" / "yahoo_adjusted_close_10y.csv.meta.json",
        "rankings": source / "data" / "processed" / "pair_v6_parameter_rankings.csv.gz",
        "strict_candidates": source / "data" / "processed" / "pair_v6_strict_candidates.csv.gz",
        "selected_schedule": source / "data" / "processed" / "pair_v6_selected_parameter_schedule.csv",
        "rebalance_schedule": source / "data" / "processed" / "pair_v6_rebalance_schedule.csv",
        "original_selected_metrics": source / "data" / "processed" / "pair_v6_pair_vintage_metrics.csv",
        "original_baseline_metrics": source / "data" / "processed" / "pair_v6_oos_baseline_metrics.csv",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source artifacts: " + ", ".join(missing))
    before_hashes = {name: _sha256(path) for name, path in source_paths.items()}

    prices, price_metadata = load_timeseries_snapshot(
        source_paths["price_snapshot"], verify_hash=True
    )
    winners, strict, schedule, parameters, base_config = _canonical_winners(
        source, config
    )
    metrics, signals, trades, cost_sensitivity, replay_failures = _replay_all(
        winners, prices, parameters, base_config
    )
    metrics = _dates(metrics, ("rebalance_date", "outer_end_exclusive", "evaluation_start", "evaluation_end"))
    signals = _dates(signals, ("rebalance_date", "outer_end_exclusive", "date"))
    trades = _dates(trades, ("rebalance_date", "outer_end_exclusive", "entry_date", "exit_date"))
    cost_sensitivity = _dates(
        cost_sensitivity,
        ("rebalance_date", "outer_end_exclusive", "evaluation_start", "evaluation_end"),
    )
    reconciliation = _reconcile_original(source, metrics)

    baseline_advantage = _baseline_advantage(source, metrics)
    full_quarterly = quarterly_information_coefficients(
        metrics, scope="all_tuned_strict"
    )
    selected_metrics = metrics.loc[metrics["selected_top10"].astype(bool)].copy()
    selected_quarterly = quarterly_information_coefficients(
        selected_metrics, scope="original_top10"
    )
    delta_for_ic = baseline_advantage[
        [
            "rebalance_date",
            "pair",
            "training_score_advantage",
            "oos_return_advantage",
            "trade_count",
        ]
    ].rename(
        columns={
            "training_score_advantage": "training_score",
            "oos_return_advantage": "net_total_return",
        }
    )
    delta_quarterly = quarterly_information_coefficients(
        delta_for_ic,
        scope="selected_rule_advantage_vs_baseline",
    )
    quarterly = pd.concat(
        [full_quarterly, selected_quarterly, delta_quarterly], ignore_index=True
    )
    summary_parts = [
        summarize_information_coefficients(
            metrics,
            full_quarterly,
            scope="all_tuned_strict",
            repetitions=args.simulation_repetitions,
        ),
        summarize_information_coefficients(
            selected_metrics,
            selected_quarterly,
            scope="original_top10",
            repetitions=args.simulation_repetitions,
        ),
        summarize_information_coefficients(
            delta_for_ic,
            delta_quarterly,
            scope="selected_rule_advantage_vs_baseline",
            repetitions=args.simulation_repetitions,
        ),
    ]
    ic_summary = pd.concat(summary_parts, ignore_index=True)

    mechanism = selection_mechanism_by_quarter(metrics, schedule)
    conditional = summarize_conditional_returns(
        metrics,
        mechanism,
        strict_pass_count=len(strict),
        repetitions=args.simulation_repetitions,
    )
    study_sessions = prices.index[
        (prices.index >= schedule["rebalance_date"].min())
        & (prices.index < schedule["outer_end_exclusive"].max())
    ]
    unique_full, unique_vintages_full = aggregate_unique_pairs(
        metrics,
        signals,
        trades,
        study_sessions,
        scope="all_tuned_strict",
    )
    selected_signals = signals.loc[signals["selected_top10"].astype(bool)].copy()
    selected_trades = trades.loc[trades["selected_top10"].astype(bool)].copy()
    unique_selected, unique_vintages_selected = aggregate_unique_pairs(
        selected_metrics,
        selected_signals,
        selected_trades,
        study_sessions,
        scope="original_top10",
    )
    unique_pairs = pd.concat([unique_full, unique_selected], ignore_index=True)
    unique_vintages = pd.concat(
        [unique_vintages_full, unique_vintages_selected], ignore_index=True
    )

    metric_path = processed / "all_tuned_strict_winner_oos_metrics.csv"
    signal_path = processed / "all_tuned_strict_winner_oos_signals.csv.gz"
    trade_path = processed / "all_tuned_strict_winner_oos_trades.csv"
    cost_path = processed / "all_tuned_strict_winner_cost_sensitivity.csv"
    failure_path = processed / "all_tuned_strict_winner_replay_failures.csv"
    quarterly_path = processed / "quarterly_information_coefficients.csv"
    summary_path = processed / "information_coefficient_summary.csv"
    mechanism_path = processed / "selection_mechanism_quarterly.csv"
    conditional_path = processed / "conditional_return_summary.csv"
    baseline_path = processed / "selected_rule_vs_baseline_advantage.csv"
    unique_path = processed / "unique_pair_multi_vintage_metrics.csv"
    vintage_path = processed / "unique_pair_vintage_returns.csv"
    _atomic_csv(metrics, metric_path)
    _atomic_csv(signals, signal_path, compression="gzip")
    _atomic_csv(trades, trade_path)
    _atomic_csv(cost_sensitivity, cost_path)
    _atomic_csv(replay_failures, failure_path)
    _atomic_csv(quarterly, quarterly_path)
    _atomic_csv(ic_summary, summary_path)
    _atomic_csv(mechanism, mechanism_path)
    _atomic_csv(conditional, conditional_path)
    _atomic_csv(baseline_advantage, baseline_path)
    _atomic_csv(unique_pairs, unique_path)
    _atomic_csv(unique_vintages, vintage_path)

    charts = {
        "training": _plot_training_vs_oos(
            metrics, ic_summary, charts_dir / "training_score_vs_future_return.png"
        ),
        "ic": _plot_quarterly_ic(
            quarterly, ic_summary, charts_dir / "quarterly_spearman_ic.png"
        ),
        "mechanism": _plot_mechanism(
            mechanism, charts_dir / "selection_mechanism_returns.png"
        ),
        "multi": _plot_multi_vintage(
            unique_selected,
            selected_signals,
            study_sessions,
            charts_dir / "unique_pair_multi_vintage_persistence.png",
        ),
        "trades": _plot_trade_sampling(
            metrics, charts_dir / "pair_vintage_trade_sampling.png"
        ),
    }

    metric_reconciliation = metrics[["rebalance_date", "pair", "net_total_return"]].merge(
        signals.groupby(["rebalance_date", "pair"])["net_return"]
        .apply(lambda values: float((1.0 + values).prod() - 1.0))
        .rename("daily_compounded_return")
        .reset_index(),
        on=["rebalance_date", "pair"],
        validate="one_to_one",
    )
    maximum_daily_difference = float(
        (
            metric_reconciliation["net_total_return"]
            - metric_reconciliation["daily_compounded_return"]
        )
        .abs()
        .max()
    )
    after_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    checks = {
        "source_artifacts_unchanged": before_hashes == after_hashes,
        "price_snapshot_hash_verified": price_metadata.get("sha256")
        == before_hashes["price_snapshot"],
        "strict_screen_pass_count_is_175": len(strict) == 175,
        "successful_tuned_winner_count_is_174": len(winners) == 174,
        "original_top10_count_is_118": int(metrics["selected_top10"].sum()) == 118,
        "omitted_winner_count_is_56": int((~metrics["selected_top10"]).sum()) == 56,
        "all_frozen_winners_replayed": len(metrics) == 174 and replay_failures.empty,
        "original_selected_reconciles_at_1e_12": reconciliation[
            "reconciles_at_1e_12"
        ],
        "primary_cost_is_exactly_2bps": bool(
            pd.to_numeric(metrics["transaction_cost_bps"], errors="coerce").eq(2.0).all()
        ),
        "cost_sensitivity_exactly_1_and_2bps": bool(
            cost_sensitivity.groupby(["rebalance_date", "pair"])[
                "transaction_cost_bps"
            ]
            .apply(lambda values: set(pd.to_numeric(values, errors="coerce")) == {1.0, 2.0})
            .all()
        ),
        "all_start_flat": bool(metrics["start_flat"].astype(bool).all()),
        "all_terminal_flat_and_liquidated": bool(
            metrics["terminal_flat"].astype(bool).all()
            and metrics["terminal_liquidation"].astype(bool).all()
        ),
        "no_pair_date_overlap": not bool(signals.duplicated(["pair", "date"]).any()),
        "daily_compounding_reconciles_at_1e_12": maximum_daily_difference <= 1e-12,
        "quarterly_ic_contains_all_three_scopes": set(quarterly["scope"])
        == {
            "all_tuned_strict",
            "original_top10",
            "selected_rule_advantage_vs_baseline",
        },
        "unique_pair_counts_match_expected": len(unique_full) == 114
        and len(unique_selected) == 82,
        "selected_repeat_distribution_matches_expected": bool(
            unique_selected["vintage_count"].value_counts().to_dict()
            == {1: 60, 2: 16, 3: 2, 4: 2, 6: 2}
        ),
    }
    checks["critical_pass"] = bool(all(checks.values()))
    checks["implementation_reconciliation"] = reconciliation
    checks["maximum_daily_compounding_difference"] = maximum_daily_difference
    checks["counts"] = {
        "scheduled_quarters": int(len(schedule)),
        "active_strict_tuned_quarters": int(mechanism["tuned_strict_count"].gt(0).sum()),
        "strict_passes": int(len(strict)),
        "tuned_pair_vintages": int(len(metrics)),
        "original_top10_pair_vintages": int(metrics["selected_top10"].sum()),
        "replayed_below_cutoff": int((~metrics["selected_top10"]).sum()),
        "full_unique_pairs": int(len(unique_full)),
        "selected_unique_pairs": int(len(unique_selected)),
    }
    quality_path = output / "predictive_diagnostics_quality_checks.json"
    _atomic_json(quality_path, checks)

    artifacts = {
        "All 174 frozen-winner OOS metrics": metric_path,
        "All 174 daily signals": signal_path,
        "All 174 trade ledger": trade_path,
        "Frozen 1–2 bp cost sensitivity": cost_path,
        "Quarterly ICs": quarterly_path,
        "IC aggregate and inference": summary_path,
        "Selection-mechanism quarters": mechanism_path,
        "Conditional-return estimands": conditional_path,
        "Chosen-rule vs baseline diagnostic": baseline_path,
        "Unique-pair multi-vintage metrics": unique_path,
        "Unique-pair vintage map": vintage_path,
    }
    report_path = output / "pair_predictive_diagnostics_report.html"
    _render_report(
        report_path,
        source=source,
        prices=prices,
        metrics=metrics,
        strict=strict,
        schedule=schedule,
        quarterly=quarterly,
        ic_summary=ic_summary,
        conditional=conditional,
        mechanism=mechanism,
        unique_selected=unique_selected,
        baseline_advantage=baseline_advantage,
        charts=charts,
        quality=checks,
        artifacts=artifacts,
    )

    generated_paths = [
        *artifacts.values(),
        failure_path,
        *charts.values(),
        quality_path,
        report_path,
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "V6 training-score predictive IC and multi-vintage audit",
        "posthoc_diagnostic_only": True,
        "selection_or_retuning_performed": False,
        "primary_future_return": "next-quarter net total return",
        "primary_training_score": "chosen parameter pooled inner-validation net CAGR at 2 bps",
        "primary_ic": "quarterly cross-sectional Spearman rank correlation",
        "reported_minimum_cross_section_thresholds": [2, 3, 5],
        "all_defined_quarters_threshold": 2,
        "small_cross_section_robustness_threshold": 5,
        "primary_transaction_cost_bps": PRIMARY_COST_BPS,
        "annual_short_borrow_bps": float(base_config.annual_short_borrow_bps),
        "simulation_repetitions": args.simulation_repetitions,
        "source_artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": before_hashes[name],
            }
            for name, path in source_paths.items()
        },
        "generated_artifacts": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in generated_paths
        },
        "quality_checks_pass": checks["critical_pass"],
        "credentials_included": False,
        "limitations": [
            "current-member survivorship bias",
            "quarterly inference sample is short",
            "formation windows overlap heavily",
            "Yahoo adjusted closes are not execution-grade quotes",
            "full 97-configuration within-pair optimizer OOS IC is not evaluated",
        ],
    }
    manifest_path = output / "predictive_diagnostics_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "quality_checks": str(quality_path),
                "manifest": str(manifest_path),
                "critical_pass": checks["critical_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
