"""Evaluation-only aggregation for immutable V6 pair sleeves.

This module contains no screening, signal, tuning, or execution logic.  It
only builds identity universes and aggregates already-produced causal sleeve
returns into cash-inclusive paths, robustness diagnostics, and descriptive
rankings.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def build_identity_master(universe: pd.DataFrame) -> pd.DataFrame:
    """Return every different-issuer, within-sector canonical pair identity."""

    required = {"ticker", "sector"}
    missing = sorted(required - set(universe.columns))
    if missing:
        raise ValueError("universe is missing: " + ", ".join(missing))
    issuer_column = next(
        (column for column in ("cik", "issuer_id") if column in universe), None
    )
    columns = ["ticker", "sector"] + ([issuer_column] if issuer_column else [])
    clean = universe[columns].dropna(subset=["ticker", "sector"]).copy()
    clean["ticker"] = clean["ticker"].astype(str)
    clean["sector"] = clean["sector"].astype(str)
    clean = clean.drop_duplicates("ticker")
    rows: list[dict[str, Any]] = []
    for sector, group in clean.groupby("sector", sort=True):
        records = group.sort_values("ticker").to_dict(orient="records")
        for left, right in itertools.combinations(records, 2):
            issuer_a = str(left.get(issuer_column, "")) if issuer_column else ""
            issuer_b = str(right.get(issuer_column, "")) if issuer_column else ""
            valid_a = issuer_a.strip().lower() not in {"", "nan", "<na>", "none"}
            valid_b = issuer_b.strip().lower() not in {"", "nan", "<na>", "none"}
            if valid_a and valid_b and issuer_a == issuer_b:
                continue
            ticker_a = str(left["ticker"])
            ticker_b = str(right["ticker"])
            rows.append(
                {
                    "pair": f"{ticker_a}__{ticker_b}",
                    "sector": str(sector),
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "issuer_a": issuer_a if valid_a else pd.NA,
                    "issuer_b": issuer_b if valid_b else pd.NA,
                    "different_issuer": True,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["sector", "pair"], kind="mergesort"
    ).reset_index(drop=True)
    if result["pair"].duplicated().any():
        raise RuntimeError("canonical identity master contains duplicate pairs")
    return result


def exact_calendar_years(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> float:
    days = (pd.Timestamp(end_exclusive) - pd.Timestamp(start)).days
    if days <= 0:
        raise ValueError("end_exclusive must follow start")
    return float(days / 365.2425)


def path_statistics(
    returns: Sequence[float] | np.ndarray,
    *,
    calendar_years: float,
    annualization: int = 252,
) -> dict[str, float]:
    """Cash-inclusive performance statistics on a fixed calendar denominator."""

    values = np.asarray(returns, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if calendar_years <= 0:
        raise ValueError("calendar_years must be positive")
    if np.any(values <= -1.0):
        return {
            "final_equity": math.nan,
            "net_total_return": math.nan,
            "calendar_cagr": math.nan,
            "annual_volatility": math.nan,
            "sharpe": math.nan,
            "sortino": math.nan,
            "max_drawdown": math.nan,
            "calmar": math.nan,
        }
    equity = np.cumprod(1.0 + values)
    final_equity = float(equity[-1]) if len(equity) else 1.0
    total = final_equity - 1.0
    cagr = float(final_equity ** (1.0 / calendar_years) - 1.0)
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    volatility = (
        standard_deviation * math.sqrt(float(annualization))
        if np.isfinite(standard_deviation)
        else math.nan
    )
    sharpe = (
        float(np.mean(values) / standard_deviation * math.sqrt(float(annualization)))
        if np.isfinite(standard_deviation) and standard_deviation > 0
        else math.nan
    )
    downside = values[values < 0]
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(float(annualization)))
        if len(downside)
        else 0.0
    )
    sortino = (
        float(np.mean(values) * annualization / downside_deviation)
        if downside_deviation > 0
        else math.nan
    )
    if len(equity):
        # Include the sleeve's initial equity of 1.0.  Otherwise a loss on the
        # first evaluation session would incorrectly register no drawdown.
        running = np.maximum(1.0, np.maximum.accumulate(equity))
        maximum_drawdown = float(np.min(equity / running - 1.0))
    else:
        maximum_drawdown = 0.0
    calmar = (
        float(cagr / abs(maximum_drawdown))
        if np.isfinite(cagr) and maximum_drawdown < 0
        else math.nan
    )
    return {
        "final_equity": final_equity,
        "net_total_return": total,
        "calendar_cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": maximum_drawdown,
        "calmar": calmar,
    }


def remove_block_as_cash(
    returns: Sequence[float] | np.ndarray,
    mask: Sequence[bool] | np.ndarray,
    *,
    calendar_years: float,
) -> dict[str, float]:
    """Replace one return block with zero cash without shortening the horizon."""

    values = np.asarray(returns, dtype=float).copy()
    block = np.asarray(mask, dtype=bool)
    if values.shape != block.shape:
        raise ValueError("returns and block mask must have the same shape")
    values[block] = 0.0
    return path_statistics(values, calendar_years=calendar_years)


def anniversary_block_labels(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    *,
    block_count: int = 10,
    end_exclusive: pd.Timestamp | None = None,
) -> np.ndarray:
    """Assign fixed study sessions to adjacent start-anniversary year blocks."""

    index = pd.DatetimeIndex(dates)
    start = pd.Timestamp(start)
    boundaries = [start + pd.DateOffset(years=offset) for offset in range(block_count + 1)]
    if end_exclusive is not None:
        requested_end = pd.Timestamp(end_exclusive)
        if requested_end <= boundaries[-2]:
            raise ValueError("end_exclusive must extend into the final block")
        boundaries[-1] = requested_end
    labels = np.full(len(index), -1, dtype=int)
    for block in range(block_count):
        labels[(index >= boundaries[block]) & (index < boundaries[block + 1])] = block
    if np.any(labels < 0):
        raise ValueError("some dates fall outside anniversary blocks")
    return labels


def bootstrap_cagr_from_log_blocks(
    log_block_rates: np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 20260901,
    shared_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return q025/q50/q975 CAGR and the shared bootstrap count matrix.

    ``log_block_rates`` is pair-by-block.  Each rate is a one-year-equivalent
    log growth rate, so resampling block identities and averaging rates keeps
    the ten-year horizon fixed.
    """

    rates = np.asarray(log_block_rates, dtype=float)
    if rates.ndim != 2 or rates.shape[1] < 2:
        raise ValueError("log_block_rates must be pair-by-block with >=2 blocks")
    block_count = rates.shape[1]
    if shared_counts is None:
        generator = np.random.default_rng(seed)
        choices = generator.integers(0, block_count, size=(draws, block_count))
        counts = np.zeros((draws, block_count), dtype=np.float64)
        for block in range(block_count):
            counts[:, block] = np.count_nonzero(choices == block, axis=1)
        counts /= block_count
    else:
        counts = np.asarray(shared_counts, dtype=float)
        if counts.ndim != 2 or counts.shape[1] != block_count:
            raise ValueError("shared_counts has incompatible shape")
    quantiles = np.empty((rates.shape[0], 3), dtype=float)
    chunk_size = 500
    for start in range(0, rates.shape[0], chunk_size):
        stop = min(start + chunk_size, rates.shape[0])
        sampled = np.expm1(counts @ rates[start:stop].T)
        quantiles[start:stop] = np.quantile(sampled, [0.025, 0.50, 0.975], axis=0).T
    return quantiles, counts


def canonical_leg_weight(
    beta: float,
    dependent: str,
    ticker_a: str,
) -> float:
    """Orientation-compatible absolute gross-normalized weight of ticker A."""

    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        return math.nan
    return float(1.0 / (1.0 + beta) if dependent == ticker_a else beta / (1.0 + beta))


def stability_statistics(active_quarters: pd.DataFrame) -> dict[str, Any]:
    """Summarize orientation, hedge-weight, and chosen-parameter stability."""

    frame = active_quarters.sort_values("rebalance_date").copy()
    count = len(frame)
    if not count:
        return {
            "stability_observations": 0,
            "canonical_weight_a_mean": math.nan,
            "canonical_weight_a_median": math.nan,
            "canonical_weight_a_standard_deviation": math.nan,
            "canonical_weight_a_iqr": math.nan,
            "canonical_weight_a_mad": math.nan,
            "dominant_orientation_fraction": math.nan,
            "orientation_switch_count": 0,
            "unique_config_count": 0,
            "modal_config_fraction": math.nan,
            "exact_config_switch_rate": math.nan,
            "parameter_component_retention": math.nan,
        }
    weights = pd.to_numeric(frame["canonical_weight_a"], errors="coerce").dropna()
    orientation = frame["dependent"].astype(str)
    configs = frame["selected_config_id"].astype(str)
    orientation_switches = int((orientation != orientation.shift()).iloc[1:].sum())
    config_switch_rate = (
        float((configs != configs.shift()).iloc[1:].mean()) if count > 1 else math.nan
    )
    components = [
        "zscore_method",
        "zscore_lookback",
        "entry_z",
        "exit_z",
        "maximum_holding_days",
    ]
    changes = 0
    comparisons = 0
    component_switch_rates: dict[str, float] = {}
    for component in components:
        values = frame[component].astype(str)
        if count > 1:
            switched = (values != values.shift()).iloc[1:]
            component_switch_rates[f"{component}_switch_rate"] = float(switched.mean())
            changes += int(switched.sum())
            comparisons += len(switched)
        else:
            component_switch_rates[f"{component}_switch_rate"] = math.nan
    q25, q75 = (
        tuple(weights.quantile([0.25, 0.75])) if len(weights) else (math.nan, math.nan)
    )
    median = float(weights.median()) if len(weights) else math.nan
    result: dict[str, Any] = {
        "stability_observations": int(count),
        "canonical_weight_a_mean": float(weights.mean()) if len(weights) else math.nan,
        "canonical_weight_a_median": median,
        "canonical_weight_a_standard_deviation": float(weights.std(ddof=1))
        if len(weights) > 1
        else math.nan,
        "canonical_weight_a_iqr": float(q75 - q25) if len(weights) else math.nan,
        "canonical_weight_a_mad": float(np.median(np.abs(weights - median)))
        if len(weights)
        else math.nan,
        "dominant_orientation_fraction": float(orientation.value_counts(normalize=True).iloc[0]),
        "orientation_switch_count": orientation_switches,
        "unique_config_count": int(configs.nunique()),
        "modal_config_fraction": float(configs.value_counts(normalize=True).iloc[0]),
        "exact_config_switch_rate": config_switch_rate,
        "parameter_component_retention": float(1.0 - changes / comparisons)
        if comparisons
        else math.nan,
    }
    result.update(component_switch_rates)
    return result


def percentile_score(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(method="average", pct=True, ascending=higher_is_better)
    # pandas ascending=True gives the largest percentile to the largest value.
    if not higher_is_better:
        ranks = numeric.rank(method="average", pct=True, ascending=False)
    return ranks.fillna(0.0)


def add_consistency_score(
    metrics: pd.DataFrame,
    *,
    reference_minimum_active_quarters: int = 8,
) -> pd.DataFrame:
    """Add an invariant equal-category consistency score on one reference set."""

    result = metrics.copy()
    result["consistency_score"] = 0.0
    reference = result["active_quarters"].ge(reference_minimum_active_quarters)
    if not reference.any():
        return result
    work = result.loc[reference].copy()
    sharpe = percentile_score(work["sharpe"])
    drawdown = percentile_score(work["max_drawdown"])
    positive = percentile_score(work["positive_active_quarter_fraction"])
    coverage = percentile_score(work["active_quarter_fraction"])
    stability_components = pd.DataFrame(
        {
            "orientation": percentile_score(work["dominant_orientation_fraction"]),
            "weight_iqr": percentile_score(
                work["canonical_weight_a_iqr"], higher_is_better=False
            ),
            "config_mode": percentile_score(work["modal_config_fraction"]),
            "parameter_retention": percentile_score(
                work["parameter_component_retention"]
            ),
        },
        index=work.index,
    )
    stability = stability_components.mean(axis=1)
    score = 100.0 * pd.concat(
        [sharpe, drawdown, positive, stability, coverage], axis=1
    ).mean(axis=1)
    result.loc[reference, "consistency_score"] = score
    result["consistency_score_reference_minimum_active_quarters"] = int(
        reference_minimum_active_quarters
    )
    return result


def descriptive_rankings(
    metrics: pd.DataFrame,
    thresholds: Sequence[int] = (8, 12, 20, 30),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create three explicitly ex-post rankings at each coverage threshold."""

    required = {
        "pair",
        "active_quarters",
        "calendar_cagr",
        "bootstrap_cagr_ci_low",
        "consistency_score",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError("metrics is missing: " + ", ".join(missing))
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    specifications = {
        "calendar_cagr": "calendar_cagr",
        "risk_adjusted_consistency": "consistency_score",
        "cagr_lower_confidence_bound": "bootstrap_cagr_ci_low",
    }
    for threshold in thresholds:
        eligible = metrics.loc[metrics["active_quarters"].ge(int(threshold))].copy()
        for rank_type, target in specifications.items():
            if rank_type == "calendar_cagr":
                order = [target, "bootstrap_cagr_ci_low", "consistency_score", "active_quarters", "pair"]
            elif rank_type == "risk_adjusted_consistency":
                order = [target, "bootstrap_cagr_ci_low", "calendar_cagr", "active_quarters", "pair"]
            else:
                order = [target, "calendar_cagr", "consistency_score", "active_quarters", "pair"]
            ascending = [False, False, False, False, True]
            ranked = eligible.sort_values(order, ascending=ascending, kind="mergesort").copy()
            ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
            ranked.insert(0, "rank_type", rank_type)
            ranked.insert(0, "minimum_active_quarters", int(threshold))
            ranked["ranking_is_ex_post_descriptive"] = True
            ranked["deployable_selection"] = False
            ranked["ranking_formula_version"] = 1
            rows.append(ranked)
            summaries.append(
                {
                    "minimum_active_quarters": int(threshold),
                    "rank_type": rank_type,
                    "eligible_pair_count": int(len(ranked)),
                    "ranking_is_empty": bool(ranked.empty),
                    "threshold_relaxed": False,
                }
            )
    ranking = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ranking, pd.DataFrame(summaries)


def fixed_universe_rankings(
    metrics: pd.DataFrame,
    *,
    adequate_availability_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank complete fixed-universe paths without selecting profitable quarters.

    Every ranking uses the same full-calendar metrics.  Availability gates only
    the explicitly named Sharpe ranking; it never changes any pair's return
    path or annualization denominator.  The downside and best-period scores are
    equal-weight percentile summaries whose component values remain in the
    output for audit.
    """

    required = {
        "pair",
        "calendar_cagr",
        "sharpe",
        "max_drawdown",
        "worst_quarter_return",
        "bootstrap_cagr_ci_low",
        "cagr_without_best_quarter",
        "cagr_without_best_complete_year",
        "availability_fraction",
        "eligibility_pass_fraction",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError("metrics is missing: " + ", ".join(missing))
    if not 0.0 <= adequate_availability_fraction <= 1.0:
        raise ValueError("adequate_availability_fraction must be in [0, 1]")
    if metrics["pair"].astype(str).duplicated().any():
        raise ValueError("fixed-universe metrics must contain one row per pair")

    work = metrics.copy()
    work["downside_robustness_score"] = 100.0 * pd.concat(
        [
            percentile_score(work["max_drawdown"]),
            percentile_score(work["worst_quarter_return"]),
            percentile_score(work["bootstrap_cagr_ci_low"]),
        ],
        axis=1,
    ).mean(axis=1)
    work["best_period_removed_robustness_score"] = 100.0 * pd.concat(
        [
            percentile_score(work["cagr_without_best_quarter"]),
            percentile_score(work["cagr_without_best_complete_year"]),
        ],
        axis=1,
    ).mean(axis=1)
    work["minimum_best_period_removed_cagr"] = work[
        ["cagr_without_best_quarter", "cagr_without_best_complete_year"]
    ].min(axis=1)

    definitions: list[tuple[str, pd.Series, list[str], list[bool], str]] = [
        (
            "full_calendar_cagr",
            pd.Series(True, index=work.index),
            ["calendar_cagr", "bootstrap_cagr_ci_low", "sharpe", "pair"],
            [False, False, False, True],
            "calendar_cagr",
        ),
        (
            "adequate_availability_sharpe",
            pd.to_numeric(work["availability_fraction"], errors="coerce").ge(
                adequate_availability_fraction
            ),
            ["sharpe", "calendar_cagr", "availability_fraction", "pair"],
            [False, False, False, True],
            "sharpe",
        ),
        (
            "downside_robustness",
            pd.Series(True, index=work.index),
            [
                "downside_robustness_score",
                "bootstrap_cagr_ci_low",
                "worst_quarter_return",
                "calendar_cagr",
                "pair",
            ],
            [False, False, False, False, True],
            "downside_robustness_score",
        ),
        (
            "best_period_removed_robustness",
            pd.Series(True, index=work.index),
            [
                "best_period_removed_robustness_score",
                "minimum_best_period_removed_cagr",
                "calendar_cagr",
                "pair",
            ],
            [False, False, False, True],
            "best_period_removed_robustness_score",
        ),
        (
            "eligibility_frequency",
            pd.Series(True, index=work.index),
            [
                "eligibility_pass_fraction",
                "cagr_without_best_complete_year",
                "calendar_cagr",
                "pair",
            ],
            [False, False, False, True],
            "eligibility_pass_fraction",
        ),
    ]

    ranking_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for rank_type, mask, order, ascending, score_column in definitions:
        eligible = work.loc[mask.fillna(False)].copy()
        ranked = eligible.sort_values(order, ascending=ascending, kind="mergesort")
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "rank_type", rank_type)
        ranked["ranking_score_column"] = score_column
        ranked["ranking_is_ex_post_descriptive"] = True
        ranked["deployable_selection"] = False
        ranked["full_calendar_metrics_only"] = True
        ranked["active_period_annualization_used"] = False
        ranked["adequate_availability_minimum_fraction"] = (
            adequate_availability_fraction
            if rank_type == "adequate_availability_sharpe"
            else math.nan
        )
        ranking_parts.append(ranked)
        summary_rows.append(
            {
                "rank_type": rank_type,
                "eligible_pair_count": int(len(ranked)),
                "ranking_is_empty": bool(ranked.empty),
                "availability_filter_applied": rank_type
                == "adequate_availability_sharpe",
                "adequate_availability_minimum_fraction": (
                    adequate_availability_fraction
                    if rank_type == "adequate_availability_sharpe"
                    else math.nan
                ),
                "ranking_is_ex_post_descriptive": True,
                "active_period_annualization_used": False,
            }
        )
    rankings = pd.concat(ranking_parts, ignore_index=True)
    return rankings, pd.DataFrame(summary_rows)
