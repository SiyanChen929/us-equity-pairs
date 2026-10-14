#!/usr/bin/env python3
"""Render the additive V6 fixed-identity ten-year evaluation report.

This module is intentionally a read-only consumer of machine-readable research
outputs.  It does not screen, tune, trade, re-rank V6 pair-vintages, or write to
the immutable V6 output root.  Its only writes are the requested HTML report and
PNG figures below the additive evaluation root.

Canonical input contract (``<root>/data/processed``)
----------------------------------------------------
``pair_identity_metrics.csv``
    One row per fixed identity/path/cost.  The primary report uses the complete
    calendar path at 2 bp.  Expected core columns are ``pair``,
    ``calendar_cagr``, ``total_return``, ``sharpe``, ``max_drawdown``, and the
    quarter diagnostics ``evaluated_quarters``, ``available_quarters``,
    ``unavailable_quarters``, ``zero_trade_quarters``, ``traded_quarters``.
``pair_identity_quarterly_results.csv``
    Pair-quarter return ledger, including explicit zero-return rows where the
    runner's evaluation contract requires them.
``pair_identity_annual_returns.csv``
    Pair-calendar-year returns.
``pair_identity_rankings.csv``
    Precomputed descriptive rankings.  Canonical ``rank_type`` values are
    ``full_calendar_cagr``, ``adequate_availability_sharpe``,
    ``downside_robustness``, ``best_period_removed_robustness``, and
    ``eligibility_frequency``.  Headline rows use
    ``ranking_scope=fixed_114_primary``; the separate provenance comparison
    uses ``ranking_scope=original_v6_selected_82_subset``.
``pair_identity_coverage_diagnostics.csv``
    Missing-history and availability diagnostics.
``pair_identity_audit.csv``
    Aggregate implementation invariants and pass/fail checks.  Pair-quarter
    states remain in the quarterly and coverage ledgers.
``pair_identity_daily_returns.csv``
    Dense 1 bp and 2 bp daily returns, equity, and drawdown for every identity.

The loader accepts a small set of documented aliases so the report remains
useful while the additive runner schema is finalized.  Missing inputs are never
silently imputed; the report displays an explicit incomplete-data notice.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template


DEFAULT_CONFIG = Path("configs/pair_identity_10y_evaluation.json")
DEFAULT_OUTPUT = Path("outputs/pair_identity_10y_evaluation_2026-08-26")
EXPECTED_IDENTITY_COUNT = 114
EXPECTED_SELECTED_SUBSET_COUNT = 82
PRIMARY_COST_BPS = 2.0

INK = "#172033"
MUTED = "#69758a"
BLUE = "#315fbd"
TEAL = "#16847c"
GREEN = "#2a825d"
ORANGE = "#d17a2d"
RED = "#bd4c57"
PURPLE = "#7058ad"
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


TABLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "master": (
        "pair_identity_master.csv",
        "fixed_pair_identity_master.csv",
    ),
    "schedule": (
        "pair_identity_rebalance_schedule.csv",
        "pair_identity_schedule.csv",
    ),
    "metrics": (
        "pair_identity_metrics.csv",
        "pair_identity_10y_metrics.csv",
        "pair_identity_long_horizon_metrics.csv",
        "pair_level_metrics.csv",
    ),
    "quarterly": (
        "pair_identity_quarterly_results.csv",
        "pair_identity_quarterly_results.csv.gz",
        "pair_identity_quarterly_returns.csv",
        "pair_quarterly_results.csv",
    ),
    "daily_returns": (
        "pair_identity_daily_returns.csv.gz",
        "pair_identity_daily_returns.csv",
    ),
    "annual": (
        "pair_identity_annual_returns.csv",
        "pair_identity_yearly_returns.csv",
        "pair_annual_returns.csv",
    ),
    "rankings": (
        "pair_identity_rankings.csv",
        "pair_identity_descriptive_rankings.csv",
        "pair_rankings.csv",
    ),
    "coverage": (
        "pair_identity_coverage_diagnostics.csv",
        "pair_identity_coverage.csv",
        "pair_identity_availability.csv.gz",
        "coverage_diagnostics.csv",
    ),
    "audit": (
        "pair_identity_audit.csv",
        "pair_identity_quarterly_audit.csv",
        "pair_identity_audit.csv.gz",
    ),
    "confidence_intervals": (
        "pair_identity_confidence_intervals.csv",
        "pair_identity_bootstrap_intervals.csv",
    ),
}

QUALITY_CANDIDATES = (
    "pair_identity_quality_checks.json",
    "pair_identity_evaluation_quality_checks.json",
    "pair_identity_replay_audit.json",
    "quality_checks.json",
)

DATE_COLUMNS = {
    "date",
    "rebalance_date",
    "formation_start",
    "formation_end",
    "outer_start",
    "outer_end",
    "outer_end_exclusive",
    "study_start",
    "study_end",
    "year_start",
    "year_end",
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "pair": ("pair", "pair_identity", "identity", "pair_id"),
    "calendar_cagr": (
        "calendar_cagr",
        "bootstrap_cagr_ci_low",
        "ten_year_net_cagr",
        "full_calendar_cagr",
        "net_cagr",
        "cagr",
    ),
    "total_return": (
        "total_return",
        "net_total_return",
        "ten_year_total_return",
        "calendar_total_return",
    ),
    "sharpe": ("sharpe", "sharpe_ratio", "net_sharpe"),
    "sortino": ("sortino", "sortino_ratio"),
    "volatility": (
        "volatility",
        "annualized_volatility",
        "annual_volatility",
        "net_volatility",
    ),
    "max_drawdown": ("max_drawdown", "maximum_drawdown"),
    "calmar": ("calmar", "calmar_ratio"),
    "trade_count": ("trade_count", "total_trades", "trades"),
    "scheduled_quarters": (
        "scheduled_quarters",
        "calendar_quarters",
        "total_quarters",
    ),
    "evaluated_quarters": (
        "evaluated_quarters",
        "scheduled_quarters",
        "scheduled_quarters",
        "calendar_quarters",
        "total_quarters",
    ),
    "available_quarters": (
        "available_quarters",
        "history_available_quarters",
        "replay_available_quarters",
    ),
    "unavailable_quarters": (
        "unavailable_quarters",
        "missing_or_unavailable_quarters",
        "missing_history_quarters",
    ),
    "zero_trade_quarters": (
        "zero_trade_quarters",
        "available_zero_trade_quarters",
        "no_trade_quarters",
    ),
    "traded_quarters": (
        "traded_quarters",
        "actively_traded_quarters",
        "quarters_with_trades",
        "trade_active_quarters",
    ),
    "tuning_failed_cash_quarters": (
        "tuning_failed_cash_quarters",
        "tuning_failure_quarters",
    ),
    "evaluation_failed_cash_quarters": (
        "evaluation_failed_cash_quarters",
        "evaluation_failure_quarters",
    ),
    "other_failed_cash_quarters": (
        "other_failed_cash_quarters",
        "other_failure_quarters",
    ),
    "active_quarters": ("active_quarters", "live_quarters"),
    "availability_fraction": (
        "availability_fraction",
        "available_quarter_fraction",
        "history_availability_fraction",
    ),
    "positive_active_quarter_fraction": (
        "positive_active_quarter_fraction",
        "positive_active_quarter_percentage",
        "positive_quarter_fraction",
        "quarter_hit_rate",
    ),
    "positive_calendar_year_fraction": (
        "positive_calendar_year_fraction",
        "positive_calendar_year_percentage",
        "positive_complete_calendar_year_fraction",
        "positive_calendar_year_fraction_all_displayed",
        "positive_year_fraction",
        "year_hit_rate",
    ),
    "median_quarter_return": ("median_quarter_return", "quarter_return_median"),
    "worst_quarter_return": ("worst_quarter_return", "quarter_return_worst"),
    "best_quarter_return": ("best_quarter_return", "quarter_return_best"),
    "best_quarter_dependence": (
        "best_quarter_dependence",
        "best_quarter_return_share",
        "dependence_on_best_quarter",
        "best_quarter_share_of_total_log_growth",
    ),
    "best_year_dependence": (
        "best_year_dependence",
        "best_calendar_year_return_share",
        "dependence_on_best_calendar_year",
        "best_year_share_of_total_log_growth",
    ),
    "cagr_without_best_quarter": (
        "cagr_without_best_quarter",
        "calendar_cagr_without_best_quarter",
        "best_quarter_removed_cagr",
    ),
    "cagr_without_best_year": (
        "cagr_without_best_year",
        "cagr_without_best_complete_year",
        "calendar_cagr_without_best_year",
        "best_calendar_year_removed_cagr",
    ),
    "bootstrap_cagr_ci_low": (
        "bootstrap_cagr_ci_low",
        "cagr_ci_low",
        "cagr_lower_confidence_bound",
        "bootstrap_cagr_lower",
    ),
    "bootstrap_cagr_ci_high": (
        "bootstrap_cagr_ci_high",
        "cagr_ci_high",
        "bootstrap_cagr_upper",
    ),
    "hedge_ratio_stability": (
        "hedge_ratio_stability",
        "hedge_stability_score",
        "hedge_ratio_mode_share",
    ),
    "parameter_stability": (
        "parameter_stability",
        "parameter_stability_score",
        "parameter_mode_share",
    ),
    "dominant_orientation_fraction": (
        "dominant_orientation_fraction",
        "orientation_stability_fraction",
    ),
    "canonical_weight_a_iqr": (
        "canonical_weight_a_iqr",
        "hedge_weight_iqr",
    ),
    "modal_config_fraction": (
        "modal_config_fraction",
        "selected_config_mode_fraction",
    ),
    "parameter_component_retention": (
        "parameter_component_retention",
        "parameter_retention_fraction",
    ),
    "exact_config_switch_rate": (
        "exact_config_switch_rate",
        "config_switch_rate",
    ),
    "eligibility_pass_frequency": (
        "eligibility_pass_frequency",
        "eligibility_pass_fraction",
        "strict_pass_frequency",
        "strict_eligible_fraction",
    ),
    "pair_health_pass_frequency": (
        "pair_health_pass_frequency",
        "health_pass_frequency",
        "pair_health_pass_fraction",
    ),
    "consistency_score": ("consistency_score", "risk_adjusted_consistency_score"),
    "downside_robustness_score": (
        "downside_robustness_score",
        "downside_score",
    ),
    "best_period_removed_score": (
        "best_period_removed_score",
        "best_period_robustness_score",
        "best_period_removed_robustness_score",
    ),
}

BOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "selected_subset": (
        "selected_subset",
        "selected_subset_flag",
        "ever_selected_v6",
        "ever_v6_selected",
        "original_v6_selected_identity",
        "in_v6_selected_subset",
    ),
    "adequate_availability": (
        "adequate_availability",
        "adequate_availability_flag",
        "sharpe_ranking_eligible",
    ),
}

RANK_TYPE_LABELS = {
    "full_calendar_cagr": "1 · Highest full-calendar CAGR",
    "adequate_availability_sharpe": "2 · Highest Sharpe with adequate availability",
    "downside_robustness": "3 · Strongest downside robustness",
    "best_period_removed_robustness": "4 · Strongest after removing the best period",
    "eligibility_frequency": "5 · Most frequent V6 eligibility passes",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for column in frame.columns.intersection(pd.Index(DATE_COLUMNS)):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", format="mixed")
    return frame


def _resolve_inputs(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Path | None]]:
    processed = root / "data" / "processed"
    tables: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path | None] = {}
    for key, candidates in TABLE_CANDIDATES.items():
        selected = next((processed / name for name in candidates if (processed / name).is_file()), None)
        paths[key] = selected
        tables[key] = _read_csv(selected) if selected is not None else pd.DataFrame()
    return tables, paths


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _series(frame: pd.DataFrame, aliases: Sequence[str], *, numeric: bool = False) -> pd.Series:
    column = _first_column(frame, aliases)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = frame[column]
    return pd.to_numeric(values, errors="coerce") if numeric else values


def _boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "pass", "selected"}
    )


def _filter_primary(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    primary_flag = _first_column(work, ("is_primary", "primary_path", "is_primary_path"))
    if primary_flag is not None and _boolean(work[primary_flag]).any():
        work = work.loc[_boolean(work[primary_flag])].copy()
    path_column = _first_column(work, ("path_type", "evaluation_path", "path"))
    if path_column is not None and work[path_column].nunique(dropna=True) > 1:
        configured = str(config.get("evaluation", {}).get("primary_path", ""))
        normalized = work[path_column].astype("string").str.strip().str.lower()
        priorities = (
            configured.lower(),
            "fixed_identity_all_quarters",
            "fixed_identity",
            "full_calendar",
            "identity_complete",
            "identity_conditional",
        )
        for candidate in priorities:
            if candidate and normalized.eq(candidate).any():
                work = work.loc[normalized.eq(candidate)].copy()
                break
    cost_column = _first_column(work, ("transaction_cost_bps", "cost_bps", "execution_cost_bps"))
    if cost_column is not None:
        costs = pd.to_numeric(work[cost_column], errors="coerce")
        primary = costs.sub(PRIMARY_COST_BPS).abs().le(1e-12)
        if primary.any():
            work = work.loc[primary].copy()
    return work


def _normalise_metrics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = _filter_primary(frame, config)
    if source.empty:
        return pd.DataFrame(columns=[*METRIC_ALIASES, *BOOL_ALIASES])
    result = pd.DataFrame(index=source.index)
    for name, aliases in METRIC_ALIASES.items():
        result[name] = _series(source, aliases, numeric=name != "pair")
    for name, aliases in BOOL_ALIASES.items():
        column = _first_column(source, aliases)
        result[name] = _boolean(source[column]) if column else False
    result["pair"] = result["pair"].astype("string").str.strip()
    result = result.loc[result["pair"].notna() & result["pair"].ne("")].copy()
    if result["evaluated_quarters"].isna().all():
        components = result[["available_quarters", "unavailable_quarters"]]
        result["evaluated_quarters"] = components.sum(axis=1, min_count=2)
    if result["traded_quarters"].isna().all() and result["available_quarters"].notna().any():
        result["traded_quarters"] = result["available_quarters"] - result["zero_trade_quarters"]
    result["best_period_removed_floor"] = result[
        ["cagr_without_best_quarter", "cagr_without_best_year"]
    ].min(axis=1, skipna=False)
    result = result.drop_duplicates("pair", keep="last").sort_values("pair").reset_index(drop=True)
    # The master ledger is authoritative for the 82-name selected-subset
    # provenance when the summary metrics intentionally omit that repeated
    # flag.
    if master is not None and not master.empty:
        master_pair = _first_column(master, METRIC_ALIASES["pair"])
        master_selected = _first_column(master, BOOL_ALIASES["selected_subset"])
        if master_pair and master_selected:
            selected_map = pd.Series(
                _boolean(master[master_selected]).to_numpy(),
                index=master[master_pair].astype("string").str.strip(),
            )
            result["selected_subset"] = result["pair"].map(selected_map).fillna(
                result["selected_subset"]
            ).astype(bool)
    return result


def _rank_type(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if "eligib" in text or "strict_pass" in text:
        return "eligibility_frequency"
    if "best" in text and ("remove" in text or "period" in text):
        return "best_period_removed_robustness"
    if "downside" in text or "drawdown" in text or "sortino" in text:
        return "downside_robustness"
    if "sharpe" in text or "consisten" in text or "risk_adjust" in text:
        return "adequate_availability_sharpe"
    if "cagr" in text or "return" in text:
        return "full_calendar_cagr"
    return text


def _normalise_rankings(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    ranking_scope: str = "fixed_114_primary",
) -> pd.DataFrame:
    source = _filter_primary(frame, config)
    scope_column = _first_column(source, ("ranking_scope", "scope"))
    if scope_column is not None:
        scope = source[scope_column].astype("string").str.strip().str.lower()
        requested = str(ranking_scope).strip().lower()
        if scope.eq(requested).any():
            source = source.loc[scope.eq(requested)].copy()
        elif requested == "fixed_114_primary" and scope.eq("primary").any():
            source = source.loc[scope.eq("primary")].copy()
        else:
            source = source.iloc[0:0].copy()
    metric_universe = (
        metrics.loc[metrics["selected_subset"].fillna(False)].copy()
        if ranking_scope == "original_v6_selected_82_subset"
        else metrics
    )
    rows: list[pd.DataFrame] = []
    if not source.empty:
        pair_column = _first_column(source, METRIC_ALIASES["pair"])
        type_column = _first_column(source, ("rank_type", "ranking_type", "ranking", "criterion"))
        rank_column = _first_column(source, ("rank", "descriptive_rank", "pair_rank"))
        score_column = _first_column(source, ("score", "ranking_score", "metric_value"))
        if pair_column and type_column:
            base = pd.DataFrame(
                {
                    "pair": source[pair_column].astype("string").str.strip(),
                    "rank_type": source[type_column].map(_rank_type),
                    "rank": pd.to_numeric(source[rank_column], errors="coerce") if rank_column else np.nan,
                    "score": pd.to_numeric(source[score_column], errors="coerce") if score_column else np.nan,
                }
            )
            # The canonical ranking ledger stores each criterion in its named
            # metric column (rather than duplicating it into a generic score
            # column).  Preserve that value so the report never renders a
            # valid precomputed ranking with an empty score.
            source_score_columns = {
                "full_calendar_cagr": ("calendar_cagr", "cagr"),
                "adequate_availability_sharpe": ("sharpe", "net_sharpe"),
                "downside_robustness": ("downside_robustness_score", "downside_score"),
                "best_period_removed_robustness": (
                    "best_period_removed_robustness_score",
                    "best_period_removed_score",
                ),
                "eligibility_frequency": (
                    "eligibility_pass_fraction",
                    "eligibility_pass_frequency",
                ),
            }
            for kind, candidates in source_score_columns.items():
                named_score = _first_column(source, candidates)
                if named_score is None:
                    continue
                mask = base["rank_type"].eq(kind) & base["score"].isna()
                base.loc[mask, "score"] = pd.to_numeric(
                    source.loc[mask, named_score], errors="coerce"
                )
            rows.append(base)
    ranking = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["pair", "rank_type", "rank", "score"]
    )
    existing = set(ranking["rank_type"].dropna().astype(str))

    def derived(kind: str, column: str, *, eligible: pd.Series | None = None) -> None:
        nonlocal ranking
        if kind in existing or column not in metric_universe or metric_universe[column].notna().sum() == 0:
            return
        work = metric_universe.loc[metric_universe[column].notna()].copy()
        if eligible is not None:
            work = work.loc[eligible.reindex(work.index, fill_value=False)]
        work = work.sort_values([column, "pair"], ascending=[False, True], kind="mergesort")
        if work.empty:
            return
        addition = pd.DataFrame(
            {
                "pair": work["pair"],
                "rank_type": kind,
                "rank": np.arange(1, len(work) + 1),
                "score": work[column],
            }
        )
        ranking = pd.concat([ranking, addition], ignore_index=True)
        existing.add(kind)

    derived("full_calendar_cagr", "calendar_cagr")
    # Adequate availability is deliberately supplied by the runner.  The
    # renderer never invents a threshold after seeing results.
    adequate = metric_universe["adequate_availability"].fillna(False).astype(bool)
    if adequate.any():
        derived("adequate_availability_sharpe", "sharpe", eligible=adequate)
    derived("downside_robustness", "downside_robustness_score")
    derived("best_period_removed_robustness", "best_period_removed_floor")
    derived("eligibility_frequency", "eligibility_pass_frequency")

    if ranking.empty:
        return ranking
    ranking = ranking.merge(metric_universe, on="pair", how="left", validate="many_to_one")
    score_sources = {
        "full_calendar_cagr": "calendar_cagr",
        "adequate_availability_sharpe": "sharpe",
        "downside_robustness": "downside_robustness_score",
        "best_period_removed_robustness": "best_period_removed_score",
        "eligibility_frequency": "eligibility_pass_frequency",
    }
    for kind, column in score_sources.items():
        if column in ranking:
            mask = ranking["rank_type"].eq(kind) & ranking["score"].isna()
            ranking.loc[mask, "score"] = pd.to_numeric(
                ranking.loc[mask, column], errors="coerce"
            )
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
    for kind, group in ranking.groupby("rank_type", sort=False):
        missing = group["rank"].isna()
        if missing.any():
            ranking.loc[group.index, "rank"] = np.arange(1, len(group) + 1)
    return ranking.sort_values(["rank_type", "rank", "pair"], kind="mergesort").reset_index(drop=True)


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


def _display_rankings(frame: pd.DataFrame, *, max_rows: int = 15) -> str:
    if frame.empty:
        return '<p class="empty">This ranking is unavailable in the current artifacts.</p>'
    columns = [
        "rank",
        "pair",
        "score",
        "total_return",
        "calendar_cagr",
        "bootstrap_cagr_ci_low",
        "sharpe",
        "sortino",
        "volatility",
        "max_drawdown",
        "calmar",
        "trade_count",
        "positive_active_quarter_fraction",
        "positive_calendar_year_fraction",
        "median_quarter_return",
        "worst_quarter_return",
        "best_quarter_return",
        "cagr_without_best_quarter",
        "cagr_without_best_year",
        "eligibility_pass_frequency",
        "scheduled_quarters",
        "evaluated_quarters",
        "available_quarters",
        "unavailable_quarters",
        "zero_trade_quarters",
        "traded_quarters",
        "selected_subset",
    ]
    selected = frame[[column for column in columns if column in frame]].head(max_rows).copy()
    labels = {
        "rank": "Rank",
        "pair": "Pair",
        "score": "Ranking score",
        "total_return": "10y total return",
        "calendar_cagr": "10y CAGR",
        "bootstrap_cagr_ci_low": "CAGR CI low",
        "sharpe": "Sharpe",
        "sortino": "Sortino",
        "volatility": "Volatility",
        "max_drawdown": "Max DD",
        "calmar": "Calmar",
        "trade_count": "Trades",
        "positive_active_quarter_fraction": "Positive qtrs",
        "positive_calendar_year_fraction": "Positive yrs",
        "median_quarter_return": "Median qtr",
        "worst_quarter_return": "Worst qtr",
        "best_quarter_return": "Best qtr",
        "cagr_without_best_quarter": "CAGR ex best qtr",
        "cagr_without_best_year": "CAGR ex best yr",
        "eligibility_pass_frequency": "Eligibility pass",
        "evaluated_quarters": "Evaluated qtrs",
        "scheduled_quarters": "Scheduled",
        "available_quarters": "Available",
        "unavailable_quarters": "Unavailable",
        "zero_trade_quarters": "Zero-trade",
        "traded_quarters": "Traded",
        "selected_subset": "In V6 selected 82",
    }
    percent = {
        "total_return",
        "calendar_cagr",
        "bootstrap_cagr_ci_low",
        "volatility",
        "max_drawdown",
        "positive_active_quarter_fraction",
        "positive_calendar_year_fraction",
        "median_quarter_return",
        "worst_quarter_return",
        "best_quarter_return",
        "cagr_without_best_quarter",
        "cagr_without_best_year",
        "eligibility_pass_frequency",
    }
    integers = {
        "rank",
        "trade_count",
        "evaluated_quarters",
        "scheduled_quarters",
        "available_quarters",
        "unavailable_quarters",
        "zero_trade_quarters",
        "traded_quarters",
    }
    for column in selected:
        if column in percent:
            selected[column] = selected[column].map(_pct)
        elif column in integers:
            selected[column] = selected[column].map(_integer)
        elif column in {"score", "sharpe", "sortino", "calmar"}:
            selected[column] = selected[column].map(lambda value: _num(value, 3))
        elif column == "selected_subset":
            selected[column] = selected[column].map(lambda value: "Yes" if bool(value) else "No")
    selected = selected.rename(columns=labels)
    return selected.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="—")


def _display_metrics(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return '<p class="empty">No qualifying rows are available.</p>'
    work = frame.head(max_rows).copy()
    for column in work.columns:
        if pd.api.types.is_bool_dtype(work[column]):
            work[column] = work[column].map(lambda value: "Yes" if bool(value) else "No")
        elif any(
            token in column.lower()
            for token in ("cagr", "return", "frequency", "fraction", "drawdown", "volatility")
        ):
            work[column] = work[column].map(_pct)
        elif "quarter" in column.lower() or column in {"trade_count"}:
            work[column] = work[column].map(_integer)
        elif pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].map(lambda value: _num(value, 3))
    return work.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="—")


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#e7ebf1", linewidth=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _placeholder(path: Path, title: str, message: str) -> Path:
    figure, axis = plt.subplots(figsize=(11.8, 3.2))
    axis.axis("off")
    axis.text(0.02, 0.76, title, fontsize=15, fontweight="bold", color=INK)
    axis.text(0.02, 0.42, message, fontsize=11, color=MUTED, wrap=True)
    axis.add_patch(plt.Rectangle((0.02, 0.13), 0.96, 0.08, color=PALE, transform=axis.transAxes))
    return _save_figure(figure, path)


def _plot_cagr_ranking(metrics: pd.DataFrame, path: Path) -> Path:
    work = metrics.dropna(subset=["calendar_cagr"]).sort_values("calendar_cagr")
    if work.empty:
        return _placeholder(path, "Full-calendar CAGR ranking", "Pair metrics are unavailable.")
    colors = np.where(work["selected_subset"].fillna(False), BLUE, "#aeb8c8")
    figure, axis = plt.subplots(figsize=(12.8, 7.0))
    axis.bar(np.arange(len(work)), work["calendar_cagr"], color=colors, width=0.82)
    axis.axhline(0.0, color=INK, linewidth=0.9)
    axis.set_title("All fixed identities · complete-calendar 2 bp CAGR", loc="left")
    axis.set_ylabel("Net CAGR")
    axis.set_xlabel("114 identities, ordered by full-calendar CAGR")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_xticks([])
    _style_axis(axis)
    axis.text(
        0.01,
        0.98,
        "Blue: appeared in the 82-identity V6 selected subset · gray: strict+tuned history only",
        transform=axis.transAxes,
        va="top",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_availability(metrics: pd.DataFrame, path: Path) -> Path:
    needed = [
        "evaluated_quarters",
        "available_quarters",
        "unavailable_quarters",
        "zero_trade_quarters",
        "traded_quarters",
        "tuning_failed_cash_quarters",
    ]
    if metrics.empty or not any(metrics[column].notna().any() for column in needed):
        return _placeholder(path, "Availability and trading coverage", "Quarter-count diagnostics are unavailable.")
    work = metrics.sort_values(["calendar_cagr", "pair"], ascending=[False, True]).head(30).copy()
    labels = work["pair"].astype(str).tolist()[::-1]
    available = work["available_quarters"].fillna(0).to_numpy()[::-1]
    unavailable = work["unavailable_quarters"].fillna(0).to_numpy()[::-1]
    zero_trade = work["zero_trade_quarters"].fillna(0).to_numpy()[::-1]
    traded = work["traded_quarters"].fillna(0).to_numpy()[::-1]
    tuning_failed = work["tuning_failed_cash_quarters"].fillna(0).to_numpy()[::-1]
    other_failed = (
        work["evaluation_failed_cash_quarters"].fillna(0)
        + work["other_failed_cash_quarters"].fillna(0)
    ).to_numpy()[::-1]
    accounted = traded + zero_trade + tuning_failed + other_failed
    residual_cash = np.maximum(available - accounted, 0)
    other_cash = other_failed + residual_cash
    y = np.arange(len(work))
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 9.0), sharey=True)
    axes[0].barh(y, available, color=TEAL, label="Available")
    axes[0].barh(y, unavailable, left=available, color="#d4dbe6", label="Unavailable")
    axes[0].set_title("Evaluation availability", loc="left")
    axes[0].set_xlabel("Quarter count")
    axes[0].set_yticks(y, labels)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].barh(y, traded, color=BLUE, label="At least one trade")
    axes[1].barh(y, zero_trade, left=traded, color=ORANGE, alpha=0.75, label="Valid zero-trade")
    axes[1].barh(
        y,
        tuning_failed,
        left=traded + zero_trade,
        color=RED,
        alpha=0.78,
        label="Tuner-failed cash",
    )
    axes[1].barh(
        y,
        other_cash,
        left=traded + zero_trade + tuning_failed,
        color="#b9c2d0",
        label="Other available cash",
    )
    axes[1].set_title("Available-quarter activity", loc="left")
    axes[1].set_xlabel("Quarter count")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        _style_axis(axis)
    figure.suptitle("Top 30 by full-calendar CAGR · sparse histories remain visible", x=0.08, ha="left", color=INK, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(figure, path)


def _plot_robustness(metrics: pd.DataFrame, path: Path) -> Path:
    required = ["calendar_cagr", "cagr_without_best_quarter", "cagr_without_best_year"]
    work = metrics.dropna(subset=required).sort_values("calendar_cagr", ascending=False).head(20).copy()
    if work.empty:
        return _placeholder(path, "Best-period-removed robustness", "Best-quarter/year removal metrics are unavailable.")
    work = work.iloc[::-1]
    y = np.arange(len(work))
    figure, axis = plt.subplots(figsize=(12.6, 8.0))
    axis.scatter(work["calendar_cagr"], y, color=BLUE, s=50, label="Full 10y")
    axis.scatter(work["cagr_without_best_quarter"], y, color=ORANGE, marker="s", s=42, label="Best quarter set to cash")
    axis.scatter(work["cagr_without_best_year"], y, color=RED, marker="^", s=48, label="Best year set to cash")
    for location, row in enumerate(work.itertuples(index=False)):
        values = [row.calendar_cagr, row.cagr_without_best_quarter, row.cagr_without_best_year]
        axis.plot([min(values), max(values)], [location, location], color="#d5dce7", linewidth=1.0, zorder=0)
    axis.axvline(0.0, color=INK, linewidth=0.9)
    axis.set_yticks(y, work["pair"].astype(str))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    axis.set_xlabel("Complete-calendar CAGR")
    axis.set_title("Dependence on the best quarter and best calendar year", loc="left")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    _style_axis(axis)
    figure.tight_layout()
    return _save_figure(figure, path)


def _quarter_return_column(frame: pd.DataFrame) -> str | None:
    return _first_column(
        frame,
        (
            "quarter_return_2bp",
            "quarter_return",
            "net_quarter_return",
            "net_total_return",
            "calendar_quarter_return",
            "total_return",
        ),
    )


def _plot_top_equity(
    quarterly: pd.DataFrame,
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    path: Path,
) -> Path:
    source = _filter_primary(quarterly, config)
    pair_column = _first_column(source, METRIC_ALIASES["pair"])
    date_column = _first_column(
        source, ("outer_end_exclusive", "outer_end", "date", "rebalance_date")
    )
    start_column = _first_column(source, ("rebalance_date", "outer_start"))
    return_column = _quarter_return_column(source)
    top_frame = metrics.dropna(subset=["calendar_cagr"]).copy()
    if not top_frame.empty:
        top_frame["calendar_cagr"] = pd.to_numeric(top_frame["calendar_cagr"], errors="coerce")
        top_frame = top_frame.dropna(subset=["calendar_cagr"]).sort_values(
            ["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort"
        )
    top = top_frame.head(8)["pair"].astype(str).tolist() if not top_frame.empty else []
    if source.empty or not pair_column or not date_column or not return_column or not top:
        return _placeholder(path, "Top identity equity curves", "Quarterly return rows are unavailable or incomplete.")
    source = source.loc[source[pair_column].astype(str).isin(top)].copy()
    source[date_column] = pd.to_datetime(source[date_column], errors="coerce")
    if start_column:
        source[start_column] = pd.to_datetime(source[start_column], errors="coerce")
    source[return_column] = pd.to_numeric(source[return_column], errors="coerce")
    source = source.dropna(subset=[date_column, return_column])
    if source.empty:
        return _placeholder(path, "Top identity equity curves", "No valid quarterly returns were found.")
    figure, axis = plt.subplots(figsize=(12.8, 6.7))
    for pair in top:
        rows = source.loc[source[pair_column].astype(str).eq(pair)].sort_values(date_column)
        if rows.empty:
            continue
        equity = (1.0 + rows[return_column]).cumprod()
        x_values = rows[date_column].tolist()
        y_values = equity.tolist()
        if start_column and start_column != date_column and pd.notna(rows.iloc[0][start_column]):
            x_values = [rows.iloc[0][start_column], *x_values]
            y_values = [1.0, *y_values]
        axis.plot(x_values, y_values, linewidth=1.8, label=pair)
    axis.axhline(1.0, color=INK, linewidth=0.8, linestyle="--")
    axis.set_title("Top full-calendar identities · stitched quarterly equity", loc="left")
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Evaluation quarter")
    _style_axis(axis)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    figure.tight_layout()
    return _save_figure(figure, path)


def _plot_all_pair_paths(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    path_dir: Path,
) -> list[dict[str, str]]:
    """Render one compact daily equity/drawdown chart for every identity."""

    required = {"pair", "date", "equity_2bp", "drawdown_2bp"}
    if daily.empty or not required.issubset(daily.columns):
        return []
    source = daily[list(required)].copy()
    source["pair"] = source["pair"].astype("string").str.strip()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["equity_2bp"] = pd.to_numeric(source["equity_2bp"], errors="coerce")
    source["drawdown_2bp"] = pd.to_numeric(source["drawdown_2bp"], errors="coerce")
    source = source.dropna(subset=["pair", "date", "equity_2bp", "drawdown_2bp"])
    if source.empty:
        return []

    order = (
        metrics.dropna(subset=["calendar_cagr"])
        .sort_values(["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort")["pair"]
        .astype(str)
        .tolist()
    )
    observed = set(source["pair"].astype(str))
    order = [pair for pair in order if pair in observed]
    order.extend(sorted(observed.difference(order)))
    path_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for pair in order:
        work = source.loc[source["pair"].astype(str).eq(pair)].sort_values("date")
        if work.empty:
            continue
        safe_name = "".join(character if character.isalnum() else "_" for character in pair)
        output_path = path_dir / f"{safe_name}.png"
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10.8, 5.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        )
        axes[0].plot(work["date"], work["equity_2bp"], color=BLUE, linewidth=1.35)
        axes[0].axhline(1.0, color=INK, linewidth=0.7, linestyle="--")
        axes[0].set_ylabel("Growth of $1")
        axes[0].set_title(
            f"{pair} · full-calendar daily path at 2 bp",
            loc="left",
        )
        axes[1].fill_between(
            work["date"],
            work["drawdown_2bp"],
            0.0,
            color=RED,
            alpha=0.32,
            linewidth=0,
        )
        axes[1].plot(work["date"], work["drawdown_2bp"], color=RED, linewidth=0.8)
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Calendar date")
        axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        for axis in axes:
            _style_axis(axis)
        figure.text(
            0.99,
            0.01,
            "Flat cash, zero-trade, tuning-failed, and unavailable sessions remain in the path.",
            ha="right",
            color=MUTED,
            fontsize=8,
        )
        figure.savefig(output_path, dpi=135, bbox_inches="tight")
        plt.close(figure)
        results.append({"pair": pair, "path": output_path.as_posix()})
    return results


def _top_row(rankings: pd.DataFrame, kind: str) -> pd.Series | None:
    rows = rankings.loc[rankings["rank_type"].eq(kind)].sort_values(["rank", "pair"])
    return None if rows.empty else rows.iloc[0]


def _pair_answer(row: pd.Series | None, metric: str, *, percent: bool = False) -> str:
    if row is None:
        return "Unavailable from the current artifacts."
    pair = html.escape(str(row.get("pair", "—")))
    value = row.get(metric, np.nan)
    shown = _pct(value) if percent else _num(value, 3)
    return f"<strong>{pair}</strong> ({shown})."


def _retained_pairs(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    keep = metrics["cagr_without_best_quarter"].gt(0) & metrics["cagr_without_best_year"].gt(0)
    return metrics.loc[keep].sort_values(
        ["best_period_removed_floor", "calendar_cagr", "pair"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def _cost_comparison(
    raw_metrics: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact 1 bp / 2 bp table without reselecting either path."""

    if raw_metrics.empty:
        return pd.DataFrame()
    pair_column = _first_column(raw_metrics, METRIC_ALIASES["pair"])
    cost_column = _first_column(
        raw_metrics, ("transaction_cost_bps", "cost_bps", "execution_cost_bps")
    )
    cagr_column = _first_column(raw_metrics, METRIC_ALIASES["calendar_cagr"])
    total_column = _first_column(raw_metrics, METRIC_ALIASES["total_return"])
    if not pair_column or not cost_column or not cagr_column:
        return pd.DataFrame()
    work = pd.DataFrame(
        {
            "pair": raw_metrics[pair_column].astype("string").str.strip(),
            "cost": pd.to_numeric(raw_metrics[cost_column], errors="coerce"),
            "cagr": pd.to_numeric(raw_metrics[cagr_column], errors="coerce"),
            "total_return": (
                pd.to_numeric(raw_metrics[total_column], errors="coerce")
                if total_column
                else np.nan
            ),
        }
    ).dropna(subset=["pair", "cost"])
    work = work.loc[work["cost"].isin([1.0, 2.0])]
    if work.empty or work.duplicated(["pair", "cost"]).any():
        return pd.DataFrame()
    cagr = work.pivot(index="pair", columns="cost", values="cagr").rename(
        columns={1.0: "cagr_1bp", 2.0: "cagr_2bp"}
    )
    total = work.pivot(index="pair", columns="cost", values="total_return").rename(
        columns={1.0: "total_return_1bp", 2.0: "total_return_2bp"}
    )
    result = cagr.join(total, how="outer").reset_index()
    if {"cagr_1bp", "cagr_2bp"}.issubset(result):
        result["cagr_cost_drag_1bp_to_2bp"] = result["cagr_1bp"] - result["cagr_2bp"]
    master_pair = _first_column(master, METRIC_ALIASES["pair"])
    selected_column = _first_column(master, BOOL_ALIASES["selected_subset"])
    if master_pair and selected_column:
        selected_map = pd.Series(
            _boolean(master[selected_column]).to_numpy(),
            index=master[master_pair].astype("string").str.strip(),
        )
        result["selected_subset"] = result["pair"].map(selected_map).fillna(False)
    if "cagr_2bp" in result:
        result = result.sort_values(["cagr_2bp", "pair"], ascending=[False, True])
    return result.reset_index(drop=True)


def _quality_payload(root: Path) -> tuple[dict[str, Any], Path | None]:
    for filename in QUALITY_CANDIDATES:
        path = root / filename
        if path.is_file():
            return _read_json(path), path
        path = root / "data" / "processed" / filename
        if path.is_file():
            return _read_json(path), path
    return {}, None


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 fixed-identity ten-year evaluation</title>
<style>
:root{--ink:#172033;--muted:#69758a;--line:#dfe5ed;--blue:#315fbd;--teal:#16847c;--green:#2a825d;--orange:#d17a2d;--red:#bd4c57;--pale:#f3f6f9}
*{box-sizing:border-box} body{margin:0;background:#edf1f5;color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.55}
.page{max-width:1260px;margin:0 auto;background:white;min-height:100vh;box-shadow:0 0 36px rgba(26,39,61,.09)}
header{padding:48px 54px 38px;background:linear-gradient(135deg,#14253e,#244d7c);color:white} header h1{font-size:34px;line-height:1.15;margin:7px 0 12px} header p{max-width:900px;margin:0;color:#dce8f5}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-size:12px;color:#9bc0ed}
main{padding:36px 54px 64px} h2{font-size:24px;margin:42px 0 13px} h3{font-size:17px;margin:25px 0 10px}.lead{font-size:17px;color:#465267;max-width:980px}.warning,.note,.incomplete{padding:17px 20px;border-radius:8px;margin:18px 0}.warning{background:#fff5e9;border-left:5px solid var(--orange)}.incomplete{background:#fff0f1;border-left:5px solid var(--red)}.note{background:#eef5ff;border-left:5px solid var(--blue)}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:22px 0}.card{border:1px solid var(--line);border-radius:9px;padding:16px;background:white}.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.card .v{font-size:25px;font-weight:750;margin:4px 0}.card .s{font-size:12px;color:var(--muted)}
.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.answer{border:1px solid var(--line);border-radius:9px;padding:17px}.answer .q{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700}.answer p{margin:7px 0 0}
.chart{display:block;width:100%;margin:14px 0 28px;border:1px solid var(--line);border-radius:8px}.tabs{display:flex;flex-wrap:wrap;gap:7px;margin:15px 0}.tab-button{border:1px solid var(--line);background:white;color:var(--ink);padding:9px 13px;border-radius:7px;cursor:pointer;font-weight:650}.tab-button.active{background:var(--blue);border-color:var(--blue);color:white}.tab-panel{display:none}.tab-panel.active{display:block}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}.data-table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}.data-table th{background:var(--pale);text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);position:sticky;top:0}.data-table td{padding:8px 10px;border-bottom:1px solid #edf0f4}.data-table tr:last-child td{border-bottom:0}.empty{padding:18px;color:var(--muted);font-style:italic;border:1px dashed var(--line);border-radius:8px}.muted{color:var(--muted)}code{font-size:12px;background:#eef1f5;padding:2px 5px;border-radius:4px}.schema{font-size:13px;color:#4f5b6e}.pill{display:inline-block;padding:3px 8px;border-radius:99px;background:#edf3fb;color:#315fbd;font-size:11px;font-weight:700;margin-right:5px}footer{padding:24px 54px;background:#f1f4f7;color:var(--muted);font-size:12px}
.downloads{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 20px}.downloads a{display:inline-block;padding:7px 10px;border:1px solid var(--line);border-radius:7px;color:var(--blue);text-decoration:none;font-size:12px;font-weight:650}.path-control{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}.path-control select{padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}.path-control a{color:var(--blue)}details{margin:18px 0}summary{cursor:pointer;color:var(--blue)}
@media(max-width:900px){header,main{padding-left:24px;padding-right:24px}.cards{grid-template-columns:repeat(2,1fr)}.answers{grid-template-columns:1fr}}
@media print{body{background:white}.page{box-shadow:none}.tab-panel{display:block!important}.tab-button{display:none}.chart{break-inside:avoid}.table-wrap{overflow:visible}}
</style>
</head>
<body><div class="page">
<header>
  <div class="eyebrow">Evaluation-only extension · unchanged V6 mechanics</div>
  <h1>Fixed pair identities across a complete ten-year calendar path</h1>
  <p>V6 signal construction, the 97-candidate tuner, one-standard-error rule, execution timing, costs, positions, and quarterly refitting remain frozen. The evaluation layer schedules all 114 identities every quarter, records the original screens as diagnostics, applies no diagnostic trade gate, and uses no cross-pair Top 10 cutoff.</p>
</header>
<main>
<div class="warning"><strong>Ex-post and descriptive.</strong> The fixed universe contains identities discovered anywhere in the full V6 strict-and-tuned history. That full-history universe—including the {{ selected_subset_count }} identities that appeared in the V6 selected subset—could not have been known at the beginning of the study. These rankings are analytical comparisons, not a historical selection rule or proof of deployable alpha.</div>
<div class="note"><strong>Exact calendar scope.</strong> {{ study_start }} through {{ study_end }} inclusive: {{ quarter_count }} adjacent evaluation intervals and {{ study_sessions }} trading sessions. First clipped interval {{ first_interval }}; final clipped interval {{ final_interval }}. Each interval starts and finishes flat.</div>
{% if incomplete %}<div class="incomplete"><strong>Incomplete report inputs.</strong> {{ incomplete_message }}</div>{% endif %}

<div class="cards">
  <div class="card"><div class="k">Fixed identities</div><div class="v">{{ identity_count }}</div><div class="s">Expected full-history union: 114</div></div>
  <div class="card"><div class="k">V6 selected subset</div><div class="v">{{ selected_subset_count }}</div><div class="s">Provenance only; not a headline gate</div></div>
  <div class="card"><div class="k">Primary cost</div><div class="v">2 bp</div><div class="s">1 bp sensitivity remains separate</div></div>
  <div class="card"><div class="k">Headline denominator</div><div class="v">Full 10y</div><div class="s">Cash-inclusive complete calendar</div></div>
  <div class="card"><div class="k">Audit status</div><div class="v">{{ audit_status }}</div><div class="s">Implementation checks, not profitability</div></div>
</div>

<h2>What the extension answers</h2>
<div class="answers">
  <div class="answer"><div class="q">Strongest full-calendar performance</div><p>{{ answer_cagr }}</p></div>
  <div class="answer"><div class="q">Most consistent evidence</div><p>Sharpe leader: {{ answer_sharpe }}<br>Downside leader: {{ answer_downside }}</p></div>
  <div class="answer"><div class="q">Dependence on exceptional periods</div><p>{{ answer_dependence }}</p></div>
  <div class="answer"><div class="q">Robust after costs and best-period removal</div><p>{{ answer_retained }}</p></div>
  <div class="answer"><div class="q">Continued paper trading</div><p>{{ answer_paper }}</p></div>
</div>

<img class="chart" src="{{ cagr_chart }}" alt="Full calendar CAGR ranking">
<p class="lead">Availability never shortens the denominator. Headline CAGR, total return, drawdown, and robustness retain the complete ten-year calendar path. The activity chart separates traded, valid zero-trade, tuner-failed cash, other cash, and unavailable quarters.</p>
<img class="chart" src="{{ availability_chart }}" alt="Availability and trading quarter diagnostics">

<h2>Five descriptive rankings</h2>
<p class="muted">Every table is ex-post. “Adequate availability” must be declared by the runner before ranking; this renderer never chooses a threshold after seeing Sharpe ratios.</p>
<div class="tabs">
{% for tab in ranking_tabs %}<button class="tab-button{% if loop.first %} active{% endif %}" onclick="showTab('{{ tab.id }}',this)">{{ tab.short }}</button>{% endfor %}
</div>
{% for tab in ranking_tabs %}<section id="{{ tab.id }}" class="tab-panel{% if loop.first %} active{% endif %}"><h3>{{ tab.title }}</h3><div class="table-wrap">{{ tab.table }}</div></section>{% endfor %}
<p class="muted">{{ eligibility_note }}</p>

<details><summary><strong>Complete 114-pair metric ledger</strong></summary><div class="table-wrap">{{ all_metrics_table }}</div></details>

<h2>Original V6 selected-identity subset</h2>
<div class="note"><strong>Secondary scope only.</strong> These {{ selected_subset_count }} identities appeared at least once in the original V6 quarterly selected schedule. Their subset ranking is kept separate and never supplies a full-114 headline answer.</div>
<h3>Five subset leaders</h3><div class="table-wrap">{{ subset_leaders_table }}</div>
<h3>Subset full-calendar CAGR ranking</h3><div class="table-wrap">{{ subset_cagr_table }}</div>

<h2>Best-period dependence</h2>
<p class="lead">The best quarter and best calendar year are replaced by zero-return cash; time is not deleted. This preserves the ten-year denominator and exposes paths dominated by one favorable episode.</p>
<img class="chart" src="{{ robustness_chart }}" alt="Best period removed robustness">
<div class="table-wrap">{{ retained_table }}</div>

<h2>Cost and stability diagnostics</h2>
<h3>Frozen 1 bp sensitivity versus 2 bp primary result</h3><div class="table-wrap">{{ cost_table }}</div>
<h3>Hedge, parameter, eligibility, and activity stability</h3><div class="table-wrap">{{ stability_table }}</div>

<h2>Stitched identity paths</h2>
<img class="chart" src="{{ equity_chart }}" alt="Top pair identity equity curves">
{% if pair_path_options %}
<h3>Every identity: daily 2 bp equity and drawdown</h3>
<p class="muted">Choose any fixed identity. Each image retains all {{ study_sessions }} sessions, including flat cash periods.</p>
<div class="path-control"><label for="pairPathSelect"><strong>Pair</strong></label><select id="pairPathSelect" onchange="showPairPath(this)">{% for item in pair_path_options %}<option value="{{ item.src }}">{{ item.pair }}</option>{% endfor %}</select><a id="pairPathLink" href="{{ pair_path_options[0].src }}">Open chart file</a></div>
<img id="pairPathImage" class="chart" src="{{ pair_path_options[0].src }}" alt="Selected pair equity and drawdown curve">
{% endif %}

<h2>Audit and artifact completeness</h2>
<div class="downloads">{% for item in data_links %}<a href="{{ item.href }}">{{ item.label }}</a>{% endfor %}</div>
<div class="table-wrap">{{ inventory_table }}</div>
{% if audit_reason_table %}<h3>Most common audit / availability reasons</h3><div class="table-wrap">{{ audit_reason_table }}</div>{% endif %}

<h2>Interpretation controls</h2>
<div class="note"><strong>What this report can establish:</strong> how the fixed 114 identities behaved when their additive quarterly outputs were stitched over the same complete calendar denominator, how concentrated results are in the best period, and how results relate to missing history, activity, and V6 eligibility diagnostics.</div>
<ul>
  <li>The identity universe is selected with full-history knowledge and is therefore ex-post.</li>
  <li>Current S&amp;P 500 membership and current GICS projected backward create survivorship and classification bias.</li>
  <li>Availability, health, and eligibility frequency are descriptive diagnostics here; they do not change the headline denominator.</li>
  <li>A favorable ten-year descriptive winner alone does not justify live capital. Continued paper trading requires an explicitly frozen future protocol.</li>
</ul>

<details><summary><strong>Machine-readable schema expected by this renderer</strong></summary><div class="schema">
<p><code>pair_identity_metrics.csv</code> supplies one complete-calendar pair row per cost/path; <code>pair_identity_quarterly_results.csv.gz</code> supplies stitched quarter returns and replay states; <code>pair_identity_daily_returns.csv.gz</code> supplies every daily equity and drawdown path; <code>pair_identity_annual_returns.csv</code> supplies calendar-year returns; <code>pair_identity_rankings.csv</code> supplies the five descriptive ranking types with separate <code>fixed_114_primary</code> and <code>original_v6_selected_82_subset</code> scopes; <code>pair_identity_coverage_diagnostics.csv</code> exposes missing-history and availability details; <code>pair_identity_audit.csv</code> records aggregate implementation checks.</p>
</div></details>
</main>
<footer>Generated {{ generated_at }} · additive root: {{ output_root }} · configuration: {{ config_path }}</footer>
</div>
<script>
function showTab(id,button){document.querySelectorAll('.tab-panel').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');button.classList.add('active');}
function showPairPath(select){const image=document.getElementById('pairPathImage');const link=document.getElementById('pairPathLink');image.src=select.value;image.alt=select.options[select.selectedIndex].text+' equity and drawdown curve';link.href=select.value;}
</script>
</body></html>"""
)


def render_report(output_root: str | Path, config_path: str | Path) -> Path:
    root = Path(output_root).resolve()
    config_file = Path(config_path).resolve()
    config = _read_json(config_file)
    tables, paths = _resolve_inputs(root)
    metrics = _normalise_metrics(tables["metrics"], config, tables["master"])
    rankings = _normalise_rankings(
        tables["rankings"], metrics, config, ranking_scope="fixed_114_primary"
    )
    subset_rankings = _normalise_rankings(
        tables["rankings"],
        metrics,
        config,
        ranking_scope="original_v6_selected_82_subset",
    )
    quality, quality_path = _quality_payload(root)

    identity_count = int(metrics["pair"].nunique()) if not metrics.empty else 0
    selected_subset_count = int(metrics["selected_subset"].sum()) if not metrics.empty else 0
    evaluation_config = config.get("evaluation", {})
    study_start = str(evaluation_config.get("study_start_inclusive", "2016-08-26"))
    study_end_exclusive = pd.Timestamp(
        evaluation_config.get("study_end_exclusive", "2026-08-27")
    )
    study_end = (study_end_exclusive - pd.Timedelta(days=1)).date().isoformat()
    schedule = tables["schedule"].copy()
    if "quarter_number" in schedule:
        schedule = schedule.sort_values("quarter_number", kind="mergesort")
    quarter_count = int(len(schedule))
    study_sessions = int(
        pd.to_numeric(schedule.get("outer_sessions", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .sum()
    )

    def interval_label(row: pd.Series | None) -> str:
        if row is None:
            return "unavailable"
        start = pd.to_datetime(row.get("rebalance_date"), errors="coerce")
        end = pd.to_datetime(row.get("outer_end_exclusive"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return "unavailable"
        return f"[{start.date().isoformat()}, {end.date().isoformat()})"

    first_interval = interval_label(schedule.iloc[0] if len(schedule) else None)
    final_interval = interval_label(schedule.iloc[-1] if len(schedule) else None)
    missing_core = [
        key
        for key in ("metrics", "quarterly", "daily_returns", "annual", "rankings", "schedule")
        if tables[key].empty
    ]
    if tables["audit"].empty and not quality:
        missing_core.append("audit")
    count_warnings: list[str] = []
    if identity_count and identity_count != EXPECTED_IDENTITY_COUNT:
        count_warnings.append(f"metrics contain {identity_count} identities, expected {EXPECTED_IDENTITY_COUNT}")
    if metrics["selected_subset"].any() and selected_subset_count != EXPECTED_SELECTED_SUBSET_COUNT:
        count_warnings.append(
            f"selected-subset flag identifies {selected_subset_count} identities, expected {EXPECTED_SELECTED_SUBSET_COUNT}"
        )
    required_rank_types = set(RANK_TYPE_LABELS)
    observed_rank_types = set(rankings["rank_type"].astype(str)) if not rankings.empty else set()
    absent_rank_types = sorted(required_rank_types - observed_rank_types)
    if absent_rank_types:
        count_warnings.append("missing ranking types: " + ", ".join(absent_rank_types))
    if not rankings.empty:
        for kind in required_rank_types - {"adequate_availability_sharpe"}:
            count = int(rankings.loc[rankings["rank_type"].eq(kind), "pair"].nunique())
            if count != EXPECTED_IDENTITY_COUNT:
                count_warnings.append(f"{kind} contains {count} identities, expected 114")
    incomplete_items = [f"missing/empty {key}" for key in missing_core] + count_warnings
    incomplete = bool(incomplete_items)

    chart_dir = root / "charts"
    cagr_path = _plot_cagr_ranking(metrics, chart_dir / "pair_identity_full_calendar_cagr.png")
    availability_path = _plot_availability(metrics, chart_dir / "pair_identity_availability_activity.png")
    robustness_path = _plot_robustness(metrics, chart_dir / "pair_identity_best_period_robustness.png")
    equity_path = _plot_top_equity(
        tables["quarterly"], metrics, config, chart_dir / "pair_identity_top_equity_curves.png"
    )
    pair_path_files = _plot_all_pair_paths(
        tables["daily_returns"], metrics, chart_dir / "pair_paths"
    )

    tabs: list[dict[str, str]] = []
    shorts = {
        "full_calendar_cagr": "Full CAGR",
        "adequate_availability_sharpe": "Sharpe",
        "downside_robustness": "Downside",
        "best_period_removed_robustness": "Best-period removed",
        "eligibility_frequency": "Eligibility frequency",
    }
    for kind, title in RANK_TYPE_LABELS.items():
        table = rankings.loc[rankings["rank_type"].eq(kind)].sort_values(["rank", "pair"])
        tabs.append(
            {
                "id": "rank_" + kind,
                "short": shorts[kind],
                "title": title + " · ex-post/descriptive",
                "table": _display_rankings(table),
            }
        )

    subset_leader_rows: list[dict[str, Any]] = []
    for kind, title in RANK_TYPE_LABELS.items():
        row = _top_row(subset_rankings, kind)
        if row is None:
            continue
        subset_leader_rows.append(
            {
                "Ranking": title.split("·", 1)[-1].strip(),
                "Pair": row.get("pair", "—"),
                "Score": _num(row.get("score", np.nan), 3),
                "10y CAGR": _pct(row.get("calendar_cagr", np.nan)),
                "Sharpe": _num(row.get("sharpe", np.nan), 3),
            }
        )
    subset_leaders_table = (
        pd.DataFrame(subset_leader_rows).to_html(
            index=False, border=0, classes="data-table", escape=True
        )
        if subset_leader_rows
        else '<p class="empty">No selected-subset ranking rows are available.</p>'
    )
    subset_cagr_table = _display_rankings(
        subset_rankings.loc[
            subset_rankings["rank_type"].eq("full_calendar_cagr")
        ].sort_values(["rank", "pair"]),
        max_rows=20,
    )

    retained = _retained_pairs(metrics)
    retained_columns = [
        "pair",
        "calendar_cagr",
        "cagr_without_best_quarter",
        "cagr_without_best_year",
        "evaluated_quarters",
        "available_quarters",
        "traded_quarters",
    ]
    retained_table = _display_metrics(retained[[column for column in retained_columns if column in retained]], max_rows=30)

    costs = _cost_comparison(tables["metrics"], tables["master"])
    cost_table = _display_metrics(costs, max_rows=30)
    stability_columns = [
        "pair",
        "calendar_cagr",
        "dominant_orientation_fraction",
        "canonical_weight_a_iqr",
        "modal_config_fraction",
        "parameter_component_retention",
        "exact_config_switch_rate",
        "eligibility_pass_frequency",
        "scheduled_quarters",
        "evaluated_quarters",
        "available_quarters",
        "unavailable_quarters",
        "zero_trade_quarters",
        "traded_quarters",
    ]
    stability = metrics.sort_values(
        ["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort"
    )
    stability_table = _display_metrics(
        stability[[column for column in stability_columns if column in stability]],
        max_rows=30,
    )
    all_metric_columns = [
        "pair",
        "total_return",
        "calendar_cagr",
        "sharpe",
        "sortino",
        "volatility",
        "max_drawdown",
        "calmar",
        "trade_count",
        "scheduled_quarters",
        "evaluated_quarters",
        "available_quarters",
        "unavailable_quarters",
        "tuning_failed_cash_quarters",
        "zero_trade_quarters",
        "traded_quarters",
        "positive_active_quarter_fraction",
        "positive_calendar_year_fraction",
        "median_quarter_return",
        "worst_quarter_return",
        "best_quarter_return",
        "cagr_without_best_quarter",
        "cagr_without_best_year",
        "bootstrap_cagr_ci_low",
        "bootstrap_cagr_ci_high",
        "eligibility_pass_frequency",
        "pair_health_pass_frequency",
    ]
    all_metrics = metrics.sort_values(
        ["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort"
    )
    all_metrics_table = _display_metrics(
        all_metrics[[column for column in all_metric_columns if column in all_metrics]],
        max_rows=max(len(all_metrics), 1),
    )

    eligibility_rows = rankings.loc[
        rankings["rank_type"].eq("eligibility_frequency")
    ].sort_values(["rank", "pair"])
    eligibility_note = "Eligibility-frequency ranking unavailable."
    if not eligibility_rows.empty:
        top_frequency = pd.to_numeric(eligibility_rows["score"], errors="coerce").max()
        tied = eligibility_rows.loc[
            np.isclose(
                pd.to_numeric(eligibility_rows["score"], errors="coerce"),
                top_frequency,
                equal_nan=False,
            )
        ]
        names = ", ".join(tied["pair"].astype(str).tolist())
        passes = int(round(top_frequency * quarter_count)) if math.isfinite(top_frequency) else 0
        eligibility_note = (
            f"Highest strict-eligibility frequency: {names}, tied at {passes}/{quarter_count} intervals. "
            "The predeclared ranking order uses full-calendar CAGR as its tie-break."
        )

    cagr_top = _top_row(rankings, "full_calendar_cagr")
    sharpe_top = _top_row(rankings, "adequate_availability_sharpe")
    downside_top = _top_row(rankings, "downside_robustness")
    removed_top = _top_row(rankings, "best_period_removed_robustness")

    if downside_top is None:
        answer_downside = "Unavailable from the current artifacts."
    else:
        answer_downside = (
            f"<strong>{html.escape(str(downside_top.get('pair', '—')))}</strong> "
            f"(score {_num(downside_top.get('score', np.nan), 2)}; "
            f"max drawdown {_pct(downside_top.get('max_drawdown', np.nan))}; "
            f"worst quarter {_pct(downside_top.get('worst_quarter_return', np.nan))}; "
            f"bootstrap CAGR floor {_pct(downside_top.get('bootstrap_cagr_ci_low', np.nan))})."
        )

    if cagr_top is None:
        answer_dependence = "Unavailable from the current artifacts."
    else:
        winner = html.escape(str(cagr_top.get("pair", "—")))
        quarter_share = _pct(cagr_top.get("best_quarter_dependence", np.nan))
        year_share = _pct(cagr_top.get("best_year_dependence", np.nan))
        after_quarter = _pct(cagr_top.get("cagr_without_best_quarter", np.nan))
        after_year = _pct(cagr_top.get("cagr_without_best_year", np.nan))
        answer_dependence = (
            f"For <strong>{winner}</strong>, the best quarter/year account for "
            f"{quarter_share}/{year_share} of total log growth; its CAGR becomes "
            f"{after_quarter}/{after_year} when those periods are set to cash."
        )
        top_cagr_names = (
            rankings.loc[rankings["rank_type"].eq("full_calendar_cagr")]
            .sort_values(["rank", "pair"])
            .head(10)["pair"]
            .astype(str)
        )
        concentrated = metrics.loc[metrics["pair"].astype(str).isin(top_cagr_names)].dropna(
            subset=["best_year_dependence"]
        )
        if not concentrated.empty:
            concentrated = concentrated.sort_values(
                ["best_year_dependence", "pair"],
                ascending=[False, True],
                kind="mergesort",
            )
            dependent = concentrated.iloc[0]
            if str(dependent["pair"]) != str(cagr_top.get("pair", "")):
                answer_dependence += (
                    f" Within the top-ten CAGR group, <strong>{html.escape(str(dependent['pair']))}</strong> "
                    f"shows the greatest best-year dependence ({_pct(dependent['best_year_dependence'])}); "
                    f"its CAGR falls to {_pct(dependent['cagr_without_best_year'])}."
                )

    if retained.empty or removed_top is None:
        answer_retained = "No pair is shown as positive after both best-period removals, or the robustness fields are unavailable."
    else:
        robust_pair = str(removed_top.get("pair", "—"))
        cost_drag = math.nan
        if not costs.empty and "cagr_cost_drag_1bp_to_2bp" in costs:
            match = costs.loc[costs["pair"].astype(str).eq(robust_pair)]
            if len(match):
                cost_drag = match.iloc[0]["cagr_cost_drag_1bp_to_2bp"]
        answer_retained = (
            f"<strong>{html.escape(robust_pair)}</strong> leads the removal-robustness rank: "
            f"CAGR {_pct(removed_top.get('cagr_without_best_quarter', np.nan))} without its best quarter "
            f"and {_pct(removed_top.get('cagr_without_best_year', np.nan))} without its best year. "
            f"The modeled 1→2 bp CAGR drag is {_num(100.0 * cost_drag, 2)} percentage points. "
            f"{len(retained)} identities remain positive after both removals."
        )

    paper_names: list[str] = []
    for row in (cagr_top, downside_top):
        if row is not None and str(row.get("pair", "")) not in paper_names:
            paper_names.append(str(row.get("pair")))
    paper_label = " and ".join(
        f"<strong>{html.escape(pair)}</strong>" for pair in paper_names
    ) or "the leading descriptive identities"
    answer_paper = (
        "The evidence supports a frozen prospective paper-trading watchlist led by "
        f"{paper_label}; it does not support live capital. Pair membership and evaluation "
        "rules must remain fixed from this point forward."
    )

    inventory_rows = []
    for key in TABLE_CANDIDATES:
        path = paths[key]
        inventory_rows.append(
            {
                "Artifact": key,
                "Status": "loaded" if path is not None and not tables[key].empty else ("empty" if path is not None else "missing"),
                "Rows": len(tables[key]),
                "File": path.name if path is not None else TABLE_CANDIDATES[key][0],
            }
        )
    if quality_path is not None:
        inventory_rows.append(
            {"Artifact": "quality_checks", "Status": "loaded", "Rows": 1, "File": quality_path.name}
        )
    inventory_table = pd.DataFrame(inventory_rows).to_html(
        index=False, border=0, classes="data-table", escape=True
    )

    audit = tables["audit"]
    reason_column = _first_column(
        audit,
        ("unavailable_reason", "evaluation_reason", "inactive_reason", "failure_reason", "status_reason"),
    )
    audit_reason_table = ""
    if reason_column and audit[reason_column].notna().any():
        reasons = (
            audit[reason_column]
            .astype("string")
            .fillna("unspecified")
            .value_counts(dropna=False)
            .rename_axis("Reason")
            .reset_index(name="Rows")
            .head(20)
        )
        audit_reason_table = reasons.to_html(index=False, border=0, classes="data-table", escape=True)

    critical = quality.get("critical_pass", quality.get("status", "unavailable"))
    if critical == "unavailable" and isinstance(quality.get("checks"), Mapping):
        checks = list(quality["checks"].values())
        critical = bool(checks) and all(bool(value) for value in checks)
    if critical == "unavailable" and not audit.empty and "passed" in audit:
        critical = bool(len(audit)) and bool(_boolean(audit["passed"]).all())
    if isinstance(critical, bool):
        audit_status = "Pass" if critical else "Fail"
    else:
        audit_status = str(critical).title() if critical not in (None, "") else "Unavailable"

    report_name = str(config.get("outputs", {}).get("report_name", "pair_identity_10y_evaluation_report.html"))
    report_path = root / Path(report_name).name
    root.mkdir(parents=True, exist_ok=True)
    relative = lambda path: path.relative_to(root).as_posix()
    pair_path_options = [
        {"pair": item["pair"], "src": relative(Path(item["path"]))}
        for item in pair_path_files
    ]
    processed = root / "data" / "processed"
    link_specs = [
        ("Complete pair metrics", "pair_identity_metrics.csv"),
        ("Quarterly panel", "pair_identity_quarterly_results.csv.gz"),
        ("Daily equity/drawdown", "pair_identity_daily_returns.csv.gz"),
        ("Annual returns", "pair_identity_annual_returns.csv"),
        ("Confidence intervals", "pair_identity_confidence_intervals.csv"),
        ("Coverage diagnostics", "pair_identity_coverage_diagnostics.csv"),
        ("Five rankings", "pair_identity_rankings.csv"),
        ("Parameter stability", "pair_identity_parameter_stability.csv"),
        ("Trade ledger", "pair_identity_trades.csv"),
        ("Implementation audit", "pair_identity_audit.csv"),
    ]
    data_links = [
        {"label": label, "href": relative(processed / filename)}
        for label, filename in link_specs
        if (processed / filename).is_file()
    ]
    rendered = HTML_TEMPLATE.render(
        incomplete=incomplete,
        incomplete_message="; ".join(incomplete_items) if incomplete_items else "",
        identity_count=_integer(identity_count),
        selected_subset_count=_integer(selected_subset_count),
        study_start=html.escape(study_start),
        study_end=html.escape(study_end),
        quarter_count=_integer(quarter_count),
        study_sessions=_integer(study_sessions),
        first_interval=html.escape(first_interval),
        final_interval=html.escape(final_interval),
        audit_status=html.escape(audit_status),
        answer_cagr=_pair_answer(cagr_top, "calendar_cagr", percent=True),
        answer_sharpe=_pair_answer(sharpe_top, "sharpe"),
        answer_downside=answer_downside,
        answer_dependence=answer_dependence,
        answer_retained=answer_retained,
        answer_paper=answer_paper,
        ranking_tabs=tabs,
        eligibility_note=html.escape(eligibility_note),
        all_metrics_table=all_metrics_table,
        subset_leaders_table=subset_leaders_table,
        subset_cagr_table=subset_cagr_table,
        retained_table=retained_table,
        cost_table=cost_table,
        stability_table=stability_table,
        inventory_table=inventory_table,
        audit_reason_table=audit_reason_table,
        cagr_chart=relative(cagr_path),
        availability_chart=relative(availability_path),
        robustness_chart=relative(robustness_path),
        equity_chart=relative(equity_path),
        pair_path_options=pair_path_options,
        data_links=data_links,
        generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        output_root=str(root),
        config_path=str(config_file),
    )
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(report_path)
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = render_report(args.output, args.config)
    print(report)


if __name__ == "__main__":
    main()
