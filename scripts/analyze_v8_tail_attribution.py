#!/usr/bin/env python3
"""Additive V8 worst-trade counterfactual and return-source attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.identity_evaluation import exact_calendar_years, path_statistics
from equity_pairs.tail_attribution import decompose_pair_returns, remove_interval_as_cash


V8_ROOT = PROJECT_ROOT / "outputs/pair_strategy_v8_pair_centric_2026-08-26"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/pair_strategy_v8_tail_attribution_2026-08-26"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _trade_component_summary(segment: pd.DataFrame) -> dict[str, float]:
    net = segment["net_return_2bp"].to_numpy(dtype=float)
    starting = pd.Series((1.0 + net).cumprod()).shift(1, fill_value=1.0).to_numpy()
    return {
        "relative_trade_pnl": float(np.sum(starting * segment["relative_return_component"].to_numpy())),
        "common_trade_pnl": float(np.sum(starting * segment["common_return_component"].to_numpy())),
        "transaction_cost_trade_pnl": float(np.sum(starting * segment["transaction_cost_component"].to_numpy())),
        "borrow_cost_trade_pnl": float(np.sum(starting * segment["borrow_cost_component"].to_numpy())),
    }


def analyze(
    v8_root: Path = V8_ROOT,
    output_root: Path = OUTPUT_ROOT,
    *,
    lineage_corrected: bool = False,
) -> dict[str, Any]:
    source = v8_root / "data/processed"
    processed = output_root / "data/processed"
    charts = output_root / "charts"
    processed.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    daily_path = source / "v8_pair_daily_paths.csv.gz"
    trades_path = source / "v8_trade_ledger.csv"
    metrics_path = source / "v8_pair_metrics.csv"
    rankings_path = source / "v8_pair_rankings.csv"
    quality_path = source / "v8_quality_checks.json"
    inputs = [daily_path, trades_path, metrics_path, rankings_path, quality_path]
    hashes_before = {str(path): _sha256(path) for path in inputs}
    daily = pd.read_csv(daily_path, parse_dates=["date", "vintage_rebalance_date"])
    trades = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])
    metrics = pd.read_csv(metrics_path)
    rankings = pd.read_csv(rankings_path)
    primary_metrics = metrics.loc[metrics["transaction_cost_bps"].eq(2.0)].set_index("pair")
    counterfactual_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    worst_detail: list[pd.DataFrame] = []
    maximum_component_error = 0.0
    maximum_trade_ledger_error = 0.0
    for pair, pair_daily in daily.groupby("pair", sort=True):
        work = decompose_pair_returns(pair_daily, transaction_cost_bps=2.0)
        maximum_component_error = max(maximum_component_error, float(work["component_error"].abs().max()))
        original = primary_metrics.loc[pair]
        pair_trades = trades.loc[trades["pair"].eq(pair)].sort_values("trade_number")
        if pair_trades.empty:
            continue
        for trade in pair_trades.itertuples(index=False):
            segment = work.loc[work["date"].between(trade.entry_date, trade.exit_date)].copy()
            trade_return = float((1.0 + segment["net_return_2bp"]).prod() - 1.0)
            maximum_trade_ledger_error = max(maximum_trade_ledger_error, abs(trade_return - float(trade.net_return_2bp)))
            components = _trade_component_summary(segment)
            held = segment.loc[segment["held_position"]]
            trade_rows.append(
                {
                    "pair": pair,
                    "trade_number": int(trade.trade_number),
                    "strategy_id": trade.strategy_id,
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "trading_sessions": int(trade.trading_sessions),
                    "net_return_2bp": trade_return,
                    **components,
                    "maximum_absolute_leg_return_while_held": float(held["maximum_absolute_leg_return"].max()) if len(held) else 0.0,
                    "held_days_with_5pct_leg_move": int((held["maximum_absolute_leg_return"] >= 0.05).sum()),
                    "worst_daily_net_return": float(segment["net_return_2bp"].min()),
                }
            )
        worst = pair_trades.sort_values(["net_return_2bp", "entry_date"]).iloc[0]
        interval = work["date"].between(worst["entry_date"], worst["exit_date"])
        removed_returns = remove_interval_as_cash(
            work["net_return_2bp"],
            work["date"],
            start=worst["entry_date"],
            end=worst["exit_date"],
        )
        years = exact_calendar_years(work["date"].iloc[0], work["date"].iloc[-1] + pd.Timedelta(days=1))
        counterfactual = path_statistics(removed_returns.to_numpy(), calendar_years=years)
        segment = work.loc[interval].copy()
        segment.insert(1, "worst_trade_number", int(worst["trade_number"]))
        worst_detail.append(segment)
        held = segment.loc[segment["held_position"]]
        maximum_leg_row = held.sort_values(["maximum_absolute_leg_return", "date"], ascending=[False, True]).iloc[0]
        largest_loss_row = segment.sort_values(["net_return_2bp", "date"]).iloc[0]
        components = _trade_component_summary(segment)
        counterfactual_rows.append(
            {
                "pair": pair,
                "original_calendar_cagr": float(original["calendar_cagr"]),
                "original_sharpe": float(original["sharpe"]),
                "original_max_drawdown": float(original["max_drawdown"]),
                "worst_trade_number": int(worst["trade_number"]),
                "worst_trade_entry_date": worst["entry_date"],
                "worst_trade_exit_date": worst["exit_date"],
                "worst_trade_strategy_id": worst["strategy_id"],
                "worst_trade_net_return_2bp": float(worst["net_return_2bp"]),
                **components,
                "largest_leg_move_date_while_held": maximum_leg_row["date"],
                "return_a_on_largest_leg_move_date": float(maximum_leg_row["return_a"]),
                "return_b_on_largest_leg_move_date": float(maximum_leg_row["return_b"]),
                "maximum_absolute_leg_return_while_held": float(maximum_leg_row["maximum_absolute_leg_return"]),
                "worst_trade_has_5pct_leg_move": bool(maximum_leg_row["maximum_absolute_leg_return"] >= 0.05),
                "largest_loss_day": largest_loss_row["date"],
                "largest_loss_day_net_return": float(largest_loss_row["net_return_2bp"]),
                "return_a_on_largest_loss_day": float(largest_loss_row["return_a"]),
                "return_b_on_largest_loss_day": float(largest_loss_row["return_b"]),
                "calendar_cagr_without_worst_trade": counterfactual["calendar_cagr"],
                "net_total_return_without_worst_trade": counterfactual["net_total_return"],
                "sharpe_without_worst_trade": counterfactual["sharpe"],
                "sortino_without_worst_trade": counterfactual["sortino"],
                "max_drawdown_without_worst_trade": counterfactual["max_drawdown"],
                "calmar_without_worst_trade": counterfactual["calmar"],
                "cagr_lift_from_ex_post_removal": counterfactual["calendar_cagr"] - float(original["calendar_cagr"]),
                "max_drawdown_improvement_from_ex_post_removal": counterfactual["max_drawdown"] - float(original["max_drawdown"]),
                "counterfactual_is_ex_post_unachievable": True,
            }
        )
        terminal_return = float((1.0 + work["net_return_2bp"]).prod() - 1.0)
        relative = float(work["relative_dollar_contribution"].sum())
        common = float(work["common_dollar_contribution"].sum())
        transaction = float(work["transaction_cost_dollar_contribution"].sum())
        borrow = float(work["borrow_cost_dollar_contribution"].sum())
        extreme = work["held_position"] & work["maximum_absolute_leg_return"].ge(0.05)
        positive_dollar = work["net_dollar_pnl"].clip(lower=0.0)
        largest_profit_index = work["net_dollar_pnl"].idxmax()
        largest_loss_index = work["net_dollar_pnl"].idxmin()
        attribution_rows.append(
            {
                "pair": pair,
                "terminal_net_total_return": terminal_return,
                "relative_price_dollar_contribution": relative,
                "common_mode_dollar_contribution": common,
                "transaction_cost_dollar_contribution": transaction,
                "borrow_cost_dollar_contribution": borrow,
                "attribution_sum": relative + common + transaction + borrow,
                "attribution_error": relative + common + transaction + borrow - terminal_return,
                "relative_component_share_of_terminal_return": relative / terminal_return if terminal_return != 0 else np.nan,
                "active_sessions": int(work["held_position"].sum()),
                "held_days_with_5pct_leg_move": int(extreme.sum()),
                "extreme_day_fraction_of_active_sessions": float(extreme.sum() / work["held_position"].sum()) if work["held_position"].sum() else 0.0,
                "net_dollar_pnl_on_5pct_leg_move_days": float(work.loc[extreme, "net_dollar_pnl"].sum()),
                "extreme_day_pnl_share_of_terminal_return": float(work.loc[extreme, "net_dollar_pnl"].sum() / terminal_return) if terminal_return != 0 else np.nan,
                "top_5_profit_days_share_of_positive_dollar_pnl": float(positive_dollar.nlargest(5).sum() / positive_dollar.sum()) if positive_dollar.sum() else np.nan,
                "largest_profit_day": work.loc[largest_profit_index, "date"],
                "largest_profit_day_net_return": float(work.loc[largest_profit_index, "net_return_2bp"]),
                "largest_loss_day": work.loc[largest_loss_index, "date"],
                "largest_loss_day_net_return": float(work.loc[largest_loss_index, "net_return_2bp"]),
            }
        )
    counterfactuals = pd.DataFrame(counterfactual_rows)
    attributions = pd.DataFrame(attribution_rows)
    trade_attributions = pd.DataFrame(trade_rows)
    worst_daily = pd.concat(worst_detail, ignore_index=True)
    headline = rankings.loc[
        rankings["ranking_scope"].eq("fixed_114_primary")
        & rankings["minimum_active_quarters"].eq(20)
        & rankings["rank_type"].eq("full_calendar_cagr")
        & rankings["rank"].le(10),
        ["rank", "pair"],
    ].sort_values("rank")
    top_counterfactual = headline.merge(counterfactuals, on="pair", validate="one_to_one")
    top_attribution = headline.merge(attributions, on="pair", validate="one_to_one")
    lineage_flags = pd.DataFrame(
        [
            {
                "ticker": "SW",
                "severity": "Critical",
                "confirmed_nyse_start": "2024-07-08",
                "backtest_has_pre_listing_history": True,
                "finding": "Current SW ordinary shares did not trade on a US domestic exchange before 2024-07-08; predecessor history requires instrument, currency, session, and exchange-ratio reconstruction.",
                "primary_source": "https://www.sec.gov/Archives/edgar/data/2005951/000200595125000005/smur-20241231.htm",
            },
            {
                "ticker": "FERG",
                "severity": "Major",
                "confirmed_nyse_start": "2021-03-11",
                "backtest_has_pre_listing_history": True,
                "finding": "Pre-NYSE history corresponds to the LSE economic lineage; a US-pair backtest requires point-in-time listing, FX, market-session, and executable-price treatment.",
                "primary_source": "https://www.sec.gov/Archives/edgar/data/2011641/000201164125000049/ferg2025ar.pdf",
            },
        ]
    )
    flagged_tickers = set(lineage_flags["ticker"])
    primary_ranking = rankings.loc[
        rankings["ranking_scope"].eq("fixed_114_primary")
        & rankings["minimum_active_quarters"].eq(20)
        & rankings["rank_type"].eq("full_calendar_cagr")
    ].sort_values("rank")
    lineage_unflagged = primary_ranking.loc[
        ~primary_ranking["ticker_a"].isin(flagged_tickers)
        & ~primary_ranking["ticker_b"].isin(flagged_tickers)
    ].head(10).copy()
    lineage_unflagged.insert(0, "lineage_unflagged_reference_rank", np.arange(1, len(lineage_unflagged) + 1))
    affected_headline_pairs = headline.loc[
        headline["pair"].str.split("__").map(lambda legs: bool(set(legs).intersection(flagged_tickers)))
    ]
    figure, axis = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(top_counterfactual))
    axis.barh(positions + 0.18, 100 * top_counterfactual["original_calendar_cagr"], height=0.34, label="Original V8")
    axis.barh(positions - 0.18, 100 * top_counterfactual["calendar_cagr_without_worst_trade"], height=0.34, label="Worst trade replaced by cash")
    axis.set_yticks(positions, top_counterfactual["pair"].str.replace("__", "/", regex=False))
    axis.invert_yaxis()
    axis.set_xlabel("Full-calendar CAGR (%)")
    axis.set_title("Ex-post worst-trade removal: diagnostic upper bound")
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(charts / "top10_worst_trade_counterfactual.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for column, label, color in [
        ("relative_price_dollar_contribution", "Relative-price component", "#3478a5"),
        ("common_mode_dollar_contribution", "Common-mode component", "#d88c31"),
        ("transaction_cost_dollar_contribution", "Transaction cost", "#b54a59"),
        ("borrow_cost_dollar_contribution", "Borrow cost", "#6d7280"),
    ]:
        axis.scatter(
            100 * top_attribution[column],
            top_attribution["pair"].str.replace("__", "/", regex=False),
            label=label,
            s=55,
            color=color,
        )
    axis.axvline(0.0, color="#222", linewidth=0.8)
    axis.set_xlabel("Exact contribution to terminal growth of $1 (percentage points)")
    axis.set_title("Where ten-year P&L came from")
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(charts / "top10_return_source_attribution.png", dpi=170)
    plt.close(figure)
    _write_csv(counterfactuals, processed / "v8_worst_trade_counterfactual.csv")
    _write_csv(attributions, processed / "v8_return_source_attribution.csv")
    _write_csv(trade_attributions, processed / "v8_trade_level_attribution.csv.gz")
    _write_csv(worst_daily, processed / "v8_worst_trade_daily_detail.csv.gz")
    _write_csv(lineage_flags, processed / "v8_ticker_lineage_flags.csv")
    _write_csv(lineage_unflagged, processed / "v8_lineage_unflagged_reference_ranking.csv")
    hashes_after = {str(path): _sha256(path) for path in inputs}
    top_worst_jump_count = int(top_counterfactual["worst_trade_has_5pct_leg_move"].sum())
    quality = {
        "research_status_verdict": "INVALID",
        "source_v8_files_unchanged": hashes_before == hashes_after,
        "pair_count": int(counterfactuals["pair"].nunique()),
        "trade_count": int(len(trade_attributions)),
        "maximum_daily_component_error": maximum_component_error,
        "maximum_terminal_attribution_error": float(attributions["attribution_error"].abs().max()),
        "maximum_trade_ledger_reconciliation_error": maximum_trade_ledger_error,
        "top10_worst_trades_with_at_least_5pct_leg_move": top_worst_jump_count,
        "top10_pairs_affected_by_confirmed_ticker_lineage_flags": int(len(affected_headline_pairs)),
        "earnings_calendar_available": False,
        "historical_minute_data_available": False,
        "interpretation_limits": [
            "headline SW pair histories are not executable as NYSE SW before 2024-07-08",
            "FERG pre-NYSE history requires explicit LSE, FX, and market-session reconstruction",
            "worst-trade removal uses hindsight and is not an executable risk rule",
            "relative-price attribution is consistent with spread P&L but does not prove a stationary causal mean-reversion process",
            "daily adjusted prices cannot establish whether an extreme move was earnings, merger, news, or a corporate-action data issue",
            "minute-stop effectiveness cannot be estimated without point-in-time intraday bars and executable fills",
        ],
    }
    _write_json(quality, processed / "v8_tail_attribution_quality.json")
    percent_columns = [
        "original_calendar_cagr",
        "worst_trade_net_return_2bp",
        "calendar_cagr_without_worst_trade",
        "cagr_lift_from_ex_post_removal",
        "original_max_drawdown",
        "max_drawdown_without_worst_trade",
        "maximum_absolute_leg_return_while_held",
    ]
    display_counterfactual = top_counterfactual[
        [
            "rank",
            "pair",
            "original_calendar_cagr",
            "worst_trade_net_return_2bp",
            "worst_trade_entry_date",
            "worst_trade_exit_date",
            "calendar_cagr_without_worst_trade",
            "cagr_lift_from_ex_post_removal",
            "original_max_drawdown",
            "max_drawdown_without_worst_trade",
            "maximum_absolute_leg_return_while_held",
        ]
    ].copy()
    for column in percent_columns:
        if column in display_counterfactual:
            display_counterfactual[column] = display_counterfactual[column].map(lambda value: f"{value:.2%}")
    display_attribution = top_attribution[
        [
            "rank",
            "pair",
            "relative_price_dollar_contribution",
            "common_mode_dollar_contribution",
            "transaction_cost_dollar_contribution",
            "held_days_with_5pct_leg_move",
            "extreme_day_fraction_of_active_sessions",
            "extreme_day_pnl_share_of_terminal_return",
            "top_5_profit_days_share_of_positive_dollar_pnl",
        ]
    ].copy()
    for column in display_attribution.columns[2:]:
        if column != "held_days_with_5pct_leg_move":
            display_attribution[column] = display_attribution[column].map(lambda value: f"{value:.2%}")
    report = REPORT.render(
        counterfactual_table=display_counterfactual.to_html(index=False, border=0),
        attribution_table=display_attribution.to_html(index=False, border=0),
        jump_count=top_worst_jump_count,
        min_lift=top_counterfactual["cagr_lift_from_ex_post_removal"].min(),
        max_lift=top_counterfactual["cagr_lift_from_ex_post_removal"].max(),
        affected_headline_count=len(affected_headline_pairs),
        lineage_corrected=lineage_corrected,
        lineage_table=lineage_unflagged[
            [
                "lineage_unflagged_reference_rank",
                "pair",
                "calendar_cagr",
                "sharpe",
                "max_drawdown",
                "active_quarters",
                "trade_count",
            ]
        ].to_html(
            index=False,
            border=0,
            formatters={
                "calendar_cagr": lambda value: f"{value:.2%}",
                "max_drawdown": lambda value: f"{value:.2%}",
                "sharpe": lambda value: f"{value:.2f}",
            },
        ),
        files=[path.name for path in sorted(processed.iterdir())],
    )
    report_path = output_root / "v8_tail_risk_and_return_attribution.html"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "pairs": len(counterfactuals),
        "trades": len(trade_attributions),
        "top10_worst_trades_with_5pct_leg_move": top_worst_jump_count,
        "maximum_component_error": maximum_component_error,
    }


REPORT = Template(
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V8 tail risk and return sources</title><style>body{margin:0;background:#edf2f6;color:#152336;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1180px;margin:auto;background:#fff;min-height:100vh}header{padding:40px 48px;background:#173751;color:#fff}main{padding:28px 48px 60px}.alert,.note{padding:14px 16px;margin:16px 0;border-left:5px solid #b33c4a;background:#fff1f2}.note{border-color:#3478a5;background:#eef7ff}.table{overflow:auto;border:1px solid #dce4ec;border-radius:8px;margin:14px 0 24px}.dataframe{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}.dataframe th{background:#f2f5f8}.dataframe th,.dataframe td{padding:8px;border-bottom:1px solid #e5e9ef;text-align:right}.dataframe th:nth-child(2),.dataframe td:nth-child(2){text-align:left}.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}.charts img{width:100%;border:1px solid #dce4ec;border-radius:8px}.links a{display:inline-block;margin:3px;padding:6px 9px;border:1px solid #d6dee8;border-radius:5px;text-decoration:none}@media(max-width:800px){header,main{padding-left:20px;padding-right:20px}.charts{grid-template-columns:1fr}}</style></head><body><div class="page"><header><div>V8 · ADDITIVE DIAGNOSTIC</div><h1>Worst-trade counterfactuals and return sources</h1><p>Signals, parameters and holding rules are unchanged.</p></header><main>{% if lineage_corrected %}<div class="note"><strong>Research status: NOT VALIDATED.</strong> This page uses the corrected SW/FERG US-tradable lineage; historical minute data is still missing, and the identity universe remains an ex-post, current-constituent development sample.</div>{% else %}<div class="alert"><strong>Research status: INVALID.</strong> NYSE: SW began trading only on 2024-07-08, and FERG was not the current NYSE series before 2021-03-11; the original backtest incorrectly used vendor backfill. {{ affected_headline_count }}/10 of the headline Top 10 are affected.</div>{% endif %}<div class="note">In {{ jump_count }}/10 of the Top 10, the worst trade saw at least one leg move ≥5% in a single day while held. Removing the worst trade lifts CAGR by {{ '%.2f%%'|format(100*min_lift) }}–{{ '%.2f%%'|format(100*max_lift) }}. Without historical minute data or a point-in-time earnings calendar, it cannot be shown that intraday stops or earnings avoidance would have captured these losses.</div><h2>Removing the single worst completed trade</h2><div class="table">{{ counterfactual_table }}</div><h2>Return sources</h2><p>The relative-price component is the exact P&amp;L from the relative move of the two legs; the common-mode component is the common move times residual net exposure.</p><div class="table">{{ attribution_table }}</div><div class="charts"><img src="charts/top10_worst_trade_counterfactual.png"><img src="charts/top10_return_source_attribution.png"></div><h2>Coverage reference excluding SW/FERG</h2><div class="table">{{ lineage_table }}</div><h2>Machine-readable files</h2><div class="links">{% for file in files %}<a href="data/processed/{{ file }}">{{ file }}</a>{% endfor %}</div></main></div></body></html>"""
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, default=V8_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--lineage-corrected", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze(
                arguments.v8_root.resolve(),
                arguments.output_root.resolve(),
                lineage_corrected=arguments.lineage_corrected,
            ),
            indent=2,
        )
    )
