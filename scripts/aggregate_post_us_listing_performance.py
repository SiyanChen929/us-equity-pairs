#!/usr/bin/env python3
"""Additive post-US-listing metrics for SW/FERG pair identities."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.identity_evaluation import exact_calendar_years, path_statistics
from equity_pairs.tradable_lineage import pair_us_listing_start


DEFAULT_ROOT = PROJECT_ROOT / "outputs/pair_strategy_v8_pit_lineage_2026-08-26"


def _metrics(returns: pd.Series, dates: pd.Series) -> dict[str, float]:
    if len(returns) == 0:
        return {
            "net_total_return": math.nan,
            "calendar_cagr": math.nan,
            "annual_volatility": math.nan,
            "sharpe": math.nan,
            "sortino": math.nan,
            "max_drawdown": math.nan,
            "calmar": math.nan,
        }
    end_exclusive = pd.Timestamp(dates.iloc[-1]) + pd.Timedelta(days=1)
    years = exact_calendar_years(pd.Timestamp(dates.iloc[0]), end_exclusive)
    return path_statistics(returns.to_numpy(dtype=float), calendar_years=years)


def aggregate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = root / "data" / "processed"
    daily = pd.read_csv(processed / "v8_pair_daily_paths.csv.gz", parse_dates=["date"])
    quarterly = pd.read_csv(
        processed / "v8_pair_quarterly_results.csv.gz",
        parse_dates=["rebalance_date", "outer_end_exclusive"],
    )
    affected = sorted(pair for pair in daily["pair"].unique() if pair_us_listing_start(pair) is not None)
    rows: list[dict[str, object]] = []
    for pair in affected:
        listing = pair_us_listing_start(pair)
        assert listing is not None
        q = quarterly.loc[quarterly["pair"].eq(pair)].sort_values("rebalance_date")
        success = q.loc[q["evaluation_success"].astype(str).str.lower().isin({"true", "1", "yes"})]
        first_evaluable = pd.Timestamp(success["rebalance_date"].iloc[0]) if len(success) else pd.NaT
        for window_name, start in (
            ("post_us_listing", listing),
            ("post_first_evaluable_quarter", first_evaluable),
        ):
            for cost in (1, 2):
                if pd.isna(start):
                    statistics = _metrics(pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]"))
                    segment = pd.DataFrame()
                    window_status = "no_three_year_training_window_before_study_end"
                else:
                    segment = daily.loc[daily["pair"].eq(pair) & daily["date"].ge(start)].sort_values("date")
                    statistics = _metrics(segment[f"net_return_{cost}bp"], segment["date"])
                    window_status = "evaluated" if len(success) else "listing_period_all_cash_insufficient_training"
                quarter_segment = q.loc[q["outer_end_exclusive"].gt(start)] if pd.notna(start) else q.iloc[0:0]
                rows.append(
                    {
                        "pair": pair,
                        "us_listing_start": listing,
                        "window": window_name,
                        "window_start": start,
                        "window_end_inclusive": pd.Timestamp("2026-08-26"),
                        "window_status": window_status,
                        "transaction_cost_bps": cost,
                        "sessions": len(segment),
                        "quarters_overlapping_window": len(quarter_segment),
                        "evaluation_success_quarters": int(
                            quarter_segment["evaluation_success"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
                        ),
                        "active_quarters": int(
                            quarter_segment["v8_actively_traded"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()
                        ),
                        "trade_count": int(segment["trade_entry"].astype(str).str.lower().isin({"true", "1", "yes"}).sum()) if len(segment) else 0,
                        **statistics,
                        "prelisting_cash_in_denominator": False,
                        "strategy_rules_changed": False,
                    }
                )
    metrics = pd.DataFrame(rows)
    rankings: list[pd.DataFrame] = []
    for (window, cost), group in metrics.groupby(["window", "transaction_cost_bps"], sort=True):
        ranked = group.loc[group["window_status"].eq("evaluated") & group["calendar_cagr"].notna()].copy()
        ranked = ranked.sort_values(["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort")
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        rankings.append(ranked)
    ranking = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    metrics.to_csv(processed / "pit_post_us_listing_metrics.csv", index=False)
    ranking.to_csv(processed / "pit_post_us_listing_rankings.csv", index=False)
    return metrics, ranking


def render(root: Path, metrics: pd.DataFrame, ranking: pd.DataFrame) -> Path:
    primary = ranking.loc[ranking["transaction_cost_bps"].eq(2)].copy()
    display = primary[
        [
            "window", "rank", "pair", "window_start", "calendar_cagr", "net_total_return",
            "sharpe", "max_drawdown", "trade_count", "active_quarters",
        ]
    ].copy()
    for column in ("calendar_cagr", "net_total_return", "max_drawdown"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    unavailable = metrics.loc[
        metrics["transaction_cost_bps"].eq(2)
        & metrics["window"].eq("post_us_listing")
        & ~metrics["window_status"].eq("evaluated"),
        ["pair", "us_listing_start", "window_status", "sessions", "trade_count"],
    ]
    report = root / "post_us_listing_performance.html"
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Post-US-listing pair performance</title><style>body{{max-width:1100px;margin:30px auto;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI";color:#172334}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:12px 15px;background:#fff7dd;border-left:5px solid #d59b18}}</style></head><body><h1>SW / FERG: strategy performance after US listing</h1><p class="note">Pre-listing cash periods are excluded. Post-listing means from the NYSE start date through 2026-08-26. Because V6 requires a three-year formation window, FERG stays in cash while training history accumulates, and SW still lacks sufficient training history at the end of the study. The second group starts at the first evaluable quarter and is a short-sample diagnostic only.</p><h2>2 bp rankings</h2>{display.to_html(index=False, border=0)}<h2>No evaluable path yet</h2>{unavailable.to_html(index=False, border=0)}<p>All results are ex-post descriptive; the 97-cell grid, signals, costs, execution and holding rules are unchanged.</p><p><a href="data/processed/pit_post_us_listing_metrics.csv">Metrics CSV</a> · <a href="data/processed/pit_post_us_listing_rankings.csv">Rankings CSV</a></p><footer>{datetime.now(timezone.utc).isoformat()}</footer></body></html>"""
    report.write_text(body, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    metrics, ranking = aggregate(root)
    report = render(root, metrics, ranking)
    print(json.dumps({"report": str(report), "metric_rows": len(metrics), "ranking_rows": len(ranking)}, indent=2))


if __name__ == "__main__":
    main()
