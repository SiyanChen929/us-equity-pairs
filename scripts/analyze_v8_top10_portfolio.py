#!/usr/bin/env python3
"""Build an audited, descriptive portfolio from the frozen V8 Top 10 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.identity_evaluation import exact_calendar_years, path_statistics


DEFAULT_SOURCE_ROOT = PROJECT_ROOT / (
    "outputs/pair_strategy_v8_pit_lineage_next_close_v3_2026-08-26"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/top10_pair_portfolio_v3_2026-08-26"
RANKING_SCOPE = "fixed_114_primary"
RANK_TYPE = "full_calendar_cagr"
MINIMUM_ACTIVE_QUARTERS = 8
PRIMARY_TRANSACTION_COST_BPS = 2.0
TOP_N = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def select_top_pairs(rankings: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Select the exact published V8 primary Top N ranking."""

    required = {
        "ranking_scope",
        "minimum_active_quarters",
        "rank_type",
        "rank",
        "pair",
        "sector",
        "transaction_cost_bps",
        "calendar_cagr",
        "sharpe",
        "max_drawdown",
        "trade_count",
        "ranking_is_ex_post_descriptive",
        "deployable_selection",
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"ranking table missing columns: {missing}")
    work = rankings.copy()
    for column in (
        "minimum_active_quarters",
        "rank",
        "transaction_cost_bps",
        "calendar_cagr",
        "sharpe",
        "max_drawdown",
        "trade_count",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    chosen = work.loc[
        work["ranking_scope"].eq(RANKING_SCOPE)
        & work["minimum_active_quarters"].eq(MINIMUM_ACTIVE_QUARTERS)
        & work["rank_type"].eq(RANK_TYPE)
        & work["rank"].between(1, top_n)
    ].sort_values("rank", kind="mergesort")
    if len(chosen) != top_n:
        raise ValueError(f"expected {top_n} ranked pairs, found {len(chosen)}")
    if chosen["pair"].duplicated().any():
        raise ValueError("Top N ranking contains duplicate pairs")
    if chosen["rank"].astype(int).tolist() != list(range(1, top_n + 1)):
        raise ValueError("Top N ranking is not contiguous from rank 1")
    if not chosen["transaction_cost_bps"].eq(PRIMARY_TRANSACTION_COST_BPS).all():
        raise ValueError("Top N ranking does not use the primary 2 bp cost")
    if not _as_bool(chosen["ranking_is_ex_post_descriptive"]).all():
        raise ValueError("Top N ranking must retain its ex-post label")
    if _as_bool(chosen["deployable_selection"]).any():
        raise ValueError("descriptive Top N ranking cannot be deployable")
    return chosen.reset_index(drop=True)


def load_balanced_panel(
    path: Path,
    pairs: list[str],
    *,
    recorded_equity_column: str,
) -> pd.DataFrame:
    """Load and validate the selected pairs from a balanced daily panel."""

    columns = [
        "pair",
        "date",
        "net_return_2bp",
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "weight_a",
        "weight_b",
        "trade_entry",
        recorded_equity_column,
    ]
    panel = pd.read_csv(path, usecols=columns)
    panel["pair"] = panel["pair"].astype(str)
    panel = panel.loc[panel["pair"].isin(pairs)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ValueError(f"{path.name} contains invalid dates")
    if panel.duplicated(["pair", "date"]).any():
        raise ValueError(f"{path.name} contains duplicate pair/date rows")
    actual_pairs = set(panel["pair"])
    if actual_pairs != set(pairs):
        raise ValueError(
            f"{path.name} pair mismatch: missing={sorted(set(pairs) - actual_pairs)}"
        )
    numeric = [
        "net_return_2bp",
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "weight_a",
        "weight_b",
        recorded_equity_column,
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        values = panel[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name} contains nonfinite {column}")
    if panel["net_return_2bp"].le(-1.0).any():
        raise ValueError(f"{path.name} contains a return at or below -100%")
    counts = panel.groupby("pair", sort=False)["date"].nunique()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if not counts.eq(len(dates)).all() or len(panel) != len(pairs) * len(dates):
        raise ValueError(f"{path.name} is not balanced across pair/date")
    panel["trade_entry"] = _as_bool(panel["trade_entry"])
    return panel.sort_values(["date", "pair"], kind="mergesort").reset_index(
        drop=True
    )


def build_portfolio_paths(
    panel: pd.DataFrame,
    pairs: list[str],
    *,
    recorded_equity_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build daily-rebalanced and independently compounded sleeve paths."""

    returns = panel.pivot(index="date", columns="pair", values="net_return_2bp")
    returns = returns.reindex(columns=pairs).sort_index()
    if returns.isna().any().any():
        raise ValueError("selected return panel contains missing pair/date values")
    recorded = panel.pivot(
        index="date", columns="pair", values=recorded_equity_column
    ).reindex(columns=pairs)
    reconstructed = (1.0 + returns).cumprod()
    equity_error = float(
        np.max(np.abs(reconstructed.to_numpy() - recorded.to_numpy()))
    )
    if equity_error > 1e-10:
        raise RuntimeError(
            f"source sleeve equity reconstruction failed: {equity_error:.3e}"
        )

    daily_equal_return = returns.mean(axis=1)
    daily_equal_equity = (1.0 + daily_equal_return).cumprod()
    initial_sleeves_equity = reconstructed.mean(axis=1)
    initial_sleeves_return = initial_sleeves_equity.div(
        initial_sleeves_equity.shift(1, fill_value=1.0)
    ).sub(1.0)
    sleeve_path_error = float(
        np.max(
            np.abs(
                (1.0 + initial_sleeves_return).cumprod().to_numpy()
                - initial_sleeves_equity.to_numpy()
            )
        )
    )
    if sleeve_path_error > 1e-12:
        raise RuntimeError(f"portfolio sleeve path failed: {sleeve_path_error:.3e}")

    grouped = panel.assign(
        pair_gross_exposure=panel["weight_a"].abs() + panel["weight_b"].abs(),
        pair_active=(panel["weight_a"].abs() + panel["weight_b"].abs()).gt(1e-12),
    ).groupby("date", sort=True)
    result = pd.DataFrame(
        {
            "daily_equal_weight_return": daily_equal_return,
            "daily_equal_weight_equity": daily_equal_equity,
            "initial_equal_weight_sleeves_return": initial_sleeves_return,
            "initial_equal_weight_sleeves_equity": initial_sleeves_equity,
            "equal_weight_gross_exposure": grouped["pair_gross_exposure"].mean(),
            "active_pair_count": grouped["pair_active"].sum().astype(int),
            "equal_weight_turnover": grouped["turnover"].mean(),
            "equal_weight_transaction_cost": grouped["transaction_cost"].mean(),
            "equal_weight_borrow_cost": grouped["borrow_cost"].mean(),
        }
    )
    result.index.name = "date"
    return result, {
        "source_sleeve_equity_max_error": equity_error,
        "portfolio_sleeve_path_max_error": sleeve_path_error,
    }


def _annual_returns(
    returns: pd.Series,
    *,
    evaluation_view: str,
    allocation_method: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    grouped = (1.0 + returns).groupby(returns.index.year).prod().sub(1.0)
    rows: list[dict[str, Any]] = []
    last_year = (end_exclusive - pd.Timedelta(days=1)).year
    for year, value in grouped.items():
        if year == start.year:
            status = "partial_start"
        elif year == last_year:
            status = "partial_end"
        else:
            status = "complete"
        rows.append(
            {
                "evaluation_view": evaluation_view,
                "allocation_method": allocation_method,
                "calendar_year": int(year),
                "period_status": status,
                "net_return": float(value),
            }
        )
    return pd.DataFrame(rows)


def _metric_row(
    returns: pd.Series,
    annual: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    evaluation_view: str,
    allocation_method: str,
    calendar_years: float,
    trade_entry_count: int,
) -> dict[str, Any]:
    statistics = path_statistics(returns.to_numpy(), calendar_years=calendar_years)
    complete = annual.loc[annual["period_status"].eq("complete")]
    nonzero = returns.abs().gt(1e-15)
    first_nonzero = returns.index[nonzero][0] if nonzero.any() else pd.NaT
    return {
        "evaluation_view": evaluation_view,
        "allocation_method": allocation_method,
        "sessions": int(len(returns)),
        "calendar_years": float(calendar_years),
        "first_nonzero_return_date": (
            pd.Timestamp(first_nonzero).date().isoformat()
            if not pd.isna(first_nonzero)
            else ""
        ),
        "nonzero_return_sessions": int(nonzero.sum()),
        "trade_entry_count": int(trade_entry_count),
        "mean_active_pair_count": float(daily["active_pair_count"].mean()),
        "maximum_active_pair_count": int(daily["active_pair_count"].max()),
        "mean_gross_exposure": float(daily["equal_weight_gross_exposure"].mean()),
        "maximum_gross_exposure": float(daily["equal_weight_gross_exposure"].max()),
        "positive_complete_year_count": int(complete["net_return"].gt(0).sum()),
        "complete_year_count": int(len(complete)),
        "best_complete_year": int(complete.loc[complete["net_return"].idxmax(), "calendar_year"]),
        "best_complete_year_return": float(complete["net_return"].max()),
        "worst_complete_year": int(complete.loc[complete["net_return"].idxmin(), "calendar_year"]),
        "worst_complete_year_return": float(complete["net_return"].min()),
        **statistics,
    }


def _format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    shown = metrics[
        [
            "evaluation_view",
            "allocation_method",
            "net_total_return",
            "calendar_cagr",
            "annual_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "positive_complete_year_count",
            "complete_year_count",
        ]
    ].copy()
    for column in (
        "net_total_return",
        "calendar_cagr",
        "annual_volatility",
        "max_drawdown",
    ):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "sortino", "calmar"):
        shown[column] = shown[column].map(lambda value: f"{value:.2f}")
    shown["positive_complete_years"] = (
        shown["positive_complete_year_count"].astype(str)
        + "/"
        + shown["complete_year_count"].astype(str)
    )
    return shown.drop(
        columns=["positive_complete_year_count", "complete_year_count"]
    ).rename(
        columns={
            "evaluation_view": "view",
            "allocation_method": "allocation",
            "net_total_return": "total return",
            "calendar_cagr": "CAGR",
            "annual_volatility": "ann. vol",
            "max_drawdown": "max DD",
        }
    )


def _plot_equity(daily: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    palette = {"daily_equal_weight": "#0b6e75", "initial_sleeves": "#d97706"}
    labels = {
        "daily_equal_weight": "Daily equal weight (no portfolio-layer rebalance cost)",
        "initial_sleeves": "Initial 10% sleeves, no cross-pair rebalance",
    }
    for axis, view, title in (
        (axes[0], "full_panel_ex_post", "Full panel · ex-post descriptive"),
        (axes[1], "post_discovery_safe_reset", "Post-discovery safe reset · same fixed Top 10"),
    ):
        subset = daily.loc[daily["evaluation_view"].eq(view)].set_index("date")
        for method, column in (
            ("daily_equal_weight", "daily_equal_weight_equity"),
            ("initial_sleeves", "initial_equal_weight_sleeves_equity"),
        ):
            axis.plot(
                pd.to_datetime(subset.index),
                subset[column],
                color=palette[method],
                linewidth=1.8,
                label=labels[method],
            )
        axis.axhline(1.0, color="#94a3b8", linewidth=0.8)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_ylabel("Equity (start = 1.0)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("Date")
    figure.suptitle("V8 Top 10 Pair Portfolio · 2 bp + 30 bp borrow proxy", fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output: Path,
    metrics: pd.DataFrame,
    members: pd.DataFrame,
    annual: pd.DataFrame,
    *,
    source_root: Path,
) -> None:
    headline = metrics.loc[
        metrics["evaluation_view"].eq("full_panel_ex_post")
        & metrics["allocation_method"].eq("initial_equal_weight_sleeves")
    ].iloc[0]
    safe = metrics.loc[
        metrics["evaluation_view"].eq("post_discovery_safe_reset")
        & metrics["allocation_method"].eq("initial_equal_weight_sleeves")
    ].iloc[0]
    member_table = members[
        ["rank", "pair", "sector", "calendar_cagr", "sharpe", "max_drawdown", "trade_count"]
    ].copy()
    for column in ("calendar_cagr", "max_drawdown"):
        member_table[column] = member_table[column].map(lambda value: f"{value:.2%}")
    member_table["sharpe"] = member_table["sharpe"].map(lambda value: f"{value:.2f}")
    annual_table = annual.pivot_table(
        index=["calendar_year", "period_status"],
        columns=["evaluation_view", "allocation_method"],
        values="net_return",
    ).reset_index()
    annual_table.columns = [
        " / ".join(str(part) for part in column if str(part) != "")
        if isinstance(column, tuple)
        else str(column)
        for column in annual_table.columns
    ]
    for column in annual_table.columns[2:]:
        annual_table[column] = annual_table[column].map(lambda value: f"{value:.2%}")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>V8 Top 10 Pair Portfolio Performance</title>
<style>
body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;background:#f4f7fb;margin:0}}
main{{max-width:1180px;margin:28px auto;padding:0 22px 50px}} h1{{margin-bottom:4px}} .muted{{color:#5f6b7a}}
.alert{{background:#fff1f2;border-left:5px solid #be123c;padding:16px 18px;margin:20px 0;border-radius:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}}
.card{{background:white;border:1px solid #dce3ee;border-radius:10px;padding:16px}} .big{{font-size:26px;font-weight:700;margin-top:6px}}
section{{background:white;border:1px solid #dce3ee;border-radius:10px;padding:18px;margin:16px 0;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:8px;border-bottom:1px solid #e6ebf2;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
img{{width:100%;height:auto}} code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
a{{color:#075985}} ul{{line-height:1.55}}
</style></head><body><main>
<h1>V8 Top 10 Pair Portfolio Performance</h1>
<p class="muted">2016-08-26 through 2026-08-26 · primary pair paths include 2 bp transaction cost and 30 bp annual short-borrow proxy.</p>
<div class="alert"><strong>Research status: NOT VALIDATED.</strong> The Top 10 were selected by CAGR on the same full 2016–2026 sample. These are ex-post descriptive results, not an independent OOS portfolio test.</div>
<div class="cards">
<div class="card">Headline full-panel CAGR<div class="big">{headline['calendar_cagr']:.2%}</div><span class="muted">initial 10% sleeves</span></div>
<div class="card">Full-panel total return<div class="big">{headline['net_total_return']:.2%}</div><span class="muted">final equity {headline['final_equity']:.3f}</span></div>
<div class="card">Full-panel max drawdown<div class="big">{headline['max_drawdown']:.2%}</div><span class="muted">Sharpe {headline['sharpe']:.2f}</span></div>
<div class="card">Safe-reset CAGR<div class="big">{safe['calendar_cagr']:.2%}</div><span class="muted">same ex-post Top 10</span></div>
</div>
<section><h2>Portfolio metrics</h2>
<p>The initial-sleeve method allocates 10% to each pair once and lets sleeves compound independently. The daily-equal method resets pair-sleeve weights each session; no additional cross-sleeve rebalance cost is charged.</p>
{_format_metrics(metrics).to_html(index=False, border=0)}</section>
<section><h2>Equity curves</h2><img src="top10_pair_portfolio_equity.png" alt="Top 10 portfolio equity curves"></section>
<section><h2>Fixed Top 10 members</h2>
<p>Filter: <code>{RANKING_SCOPE}</code>, minimum active quarters {MINIMUM_ACTIVE_QUARTERS}, rank type <code>{RANK_TYPE}</code>.</p>
{member_table.to_html(index=False, border=0)}</section>
<section><h2>Calendar returns</h2>{annual_table.to_html(index=False, border=0)}</section>
<section><h2>Interpretation limits</h2><ul>
<li>Top 10 identity selection uses the same sample as the reported full-panel performance.</li>
<li>Safe reset removes pre-discovery P&amp;L but retains pre-discovery-informed strategy selections and the ex-post Top 10 identities.</li>
<li>Pair-level modeled transaction and borrow costs are included. Cross-sleeve capital-transfer costs, margin, shared-name netting, capacity, locate availability, and realized fills are not modeled.</li>
<li>The portfolio is concentrated: six of ten pairs are Health Care; MGM, LH, and SYK each appear in more than one pair.</li>
</ul></section>
<section><h2>Artifacts</h2><ul>
<li><a href="data/top10_pair_portfolio_metrics.csv">Metrics CSV</a></li>
<li><a href="data/top10_pair_portfolio_daily.csv">Daily paths CSV</a></li>
<li><a href="data/top10_pair_portfolio_annual_returns.csv">Annual returns CSV</a></li>
<li><a href="data/top10_pair_members.csv">Top 10 members CSV</a></li>
<li><a href="data/top10_pair_portfolio_manifest.json">Lineage manifest</a></li>
<li><a href="../{source_root.name}/pair_strategy_v8_pit_lineage_next_close_v3_report.html">Source V8 report</a></li>
</ul></section>
</main></body></html>"""
    output.write_text(html, encoding="utf-8")


def run(source_root: Path, output_root: Path) -> dict[str, Any]:
    processed = source_root / "data/processed"
    source_paths = {
        "config": PROJECT_ROOT / "configs/pair_strategy_v8_pit_lineage_next_close_v3.json",
        "rankings": processed / "v8_pair_rankings.csv",
        "full_daily": processed / "v8_pair_daily_paths.csv.gz",
        "safe_reset_daily": processed / "v8_post_discovery_safe_reset_daily_paths.csv.gz",
        "quality_checks": processed / "v8_quality_checks.json",
        "engine_manifest": processed / "v8_engine_source_manifest.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    config = json.loads(source_paths["config"].read_text(encoding="utf-8"))
    quality = json.loads(source_paths["quality_checks"].read_text(encoding="utf-8"))
    if quality.get("research_status_verdict") != "NOT VALIDATED":
        raise ValueError("unexpected V8 research verdict")
    evaluation = config["evaluation"]
    if float(evaluation["primary_transaction_cost_bps"]) != PRIMARY_TRANSACTION_COST_BPS:
        raise ValueError("config primary transaction cost is not 2 bp")
    start = pd.Timestamp(evaluation["study_start_inclusive"])
    end_exclusive = pd.Timestamp(evaluation["study_end_exclusive"])
    calendar_years = exact_calendar_years(start, end_exclusive)

    rankings = pd.read_csv(source_paths["rankings"])
    members = select_top_pairs(rankings)
    pairs = members["pair"].astype(str).tolist()
    views = {
        "full_panel_ex_post": (
            source_paths["full_daily"],
            "equity_2bp",
        ),
        "post_discovery_safe_reset": (
            source_paths["safe_reset_daily"],
            "safe_reset_equity_2bp",
        ),
    }
    daily_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    reconciliations: dict[str, dict[str, float]] = {}
    for view, (path, equity_column) in views.items():
        panel = load_balanced_panel(path, pairs, recorded_equity_column=equity_column)
        if panel["date"].min() < start or panel["date"].max() >= end_exclusive:
            raise ValueError(f"{view} dates fall outside the configured study")
        daily, reconciliation = build_portfolio_paths(
            panel, pairs, recorded_equity_column=equity_column
        )
        reconciliations[view] = reconciliation
        daily_out = daily.reset_index()
        daily_out.insert(0, "evaluation_view", view)
        daily_frames.append(daily_out)
        trade_count = int(panel["trade_entry"].sum())
        for method, return_column in (
            ("daily_equal_weight", "daily_equal_weight_return"),
            (
                "initial_equal_weight_sleeves",
                "initial_equal_weight_sleeves_return",
            ),
        ):
            returns = daily[return_column]
            annual = _annual_returns(
                returns,
                evaluation_view=view,
                allocation_method=method,
                start=start,
                end_exclusive=end_exclusive,
            )
            annual_frames.append(annual)
            metric_rows.append(
                _metric_row(
                    returns,
                    annual,
                    daily,
                    evaluation_view=view,
                    allocation_method=method,
                    calendar_years=calendar_years,
                    trade_entry_count=trade_count,
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    daily_output = pd.concat(daily_frames, ignore_index=True)
    metrics_output = pd.DataFrame(metric_rows)
    annual_output = pd.concat(annual_frames, ignore_index=True)
    members_output = members[
        [
            "rank",
            "pair",
            "sector",
            "ticker_a",
            "ticker_b",
            "active_quarters",
            "trade_count",
            "calendar_cagr",
            "sharpe",
            "max_drawdown",
            "ranking_is_ex_post_descriptive",
            "deployable_selection",
        ]
    ].copy()
    paths = {
        "daily": data_root / "top10_pair_portfolio_daily.csv",
        "metrics": data_root / "top10_pair_portfolio_metrics.csv",
        "annual_returns": data_root / "top10_pair_portfolio_annual_returns.csv",
        "members": data_root / "top10_pair_members.csv",
        "chart": output_root / "top10_pair_portfolio_equity.png",
        "report": output_root / "top10_pair_portfolio_report.html",
    }
    daily_output.to_csv(paths["daily"], index=False)
    metrics_output.to_csv(paths["metrics"], index=False)
    annual_output.to_csv(paths["annual_returns"], index=False)
    members_output.to_csv(paths["members"], index=False)
    _plot_equity(daily_output, paths["chart"])
    _write_report(
        paths["report"],
        metrics_output,
        members_output,
        annual_output,
        source_root=source_root,
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": "same_sample_ex_post_descriptive_top10_portfolio",
        "selection": {
            "ranking_scope": RANKING_SCOPE,
            "minimum_active_quarters": MINIMUM_ACTIVE_QUARTERS,
            "rank_type": RANK_TYPE,
            "top_n": TOP_N,
            "pairs": pairs,
            "ranking_is_ex_post_descriptive": True,
            "deployable_selection": False,
        },
        "study": {
            "start_inclusive": start.date().isoformat(),
            "end_exclusive": end_exclusive.date().isoformat(),
            "calendar_years": calendar_years,
            "sessions_per_view": int(len(daily_output) / len(views)),
            "transaction_cost_bps": PRIMARY_TRANSACTION_COST_BPS,
            "annual_short_borrow_bps": float(
                evaluation["annual_short_borrow_bps"]
            ),
        },
        "allocation_methods": {
            "initial_equal_weight_sleeves": (
                "10 percent initial capital per pair; sleeves compound independently; "
                "no cross-pair rebalancing"
            ),
            "daily_equal_weight": (
                "daily mean of ten pair returns; portfolio-layer rebalancing cost omitted"
            ),
        },
        "reconciliation": reconciliations,
        "limitations": [
            "Top 10 identities were selected on the same 2016-2026 sample",
            "safe reset retains pre-discovery-informed strategy selections",
            "portfolio-layer rebalancing, margin, name netting, capacity, locate availability, and realized fills are not modeled",
            "source V8 universe is fixed/current and not a complete point-in-time universe",
        ],
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "output_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "metrics": metrics_output.to_dict(orient="records"),
    }
    manifest_path = data_root / "top10_pair_portfolio_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "report": str(paths["report"].resolve()),
        "metrics": str(paths["metrics"].resolve()),
        "manifest": str(manifest_path.resolve()),
        "pairs": pairs,
        "research_status_verdict": "NOT VALIDATED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
