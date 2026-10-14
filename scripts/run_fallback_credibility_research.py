#!/usr/bin/env python3
"""Correct the quota-fallback experiment and build a strict-only research audit.

This script is additive.  It never overwrites the original, condensed, weighted,
production-gate, or research-evolution reports.  The correction has four parts:

1. Treat formation "fallbacks" as ineligible quota fills and measure, rather
   than assume, whether the strict gate predicted held-out returns.
2. Re-screen the complete saved current-constituent universe at each final-year
   quarter with a five-year lagged formation window.
3. Trade at most ten strict, disjoint candidates with a covariance-only,
   turnover-regularized allocator; scarce strict capacity is held as zero-yield
   cash and no numerical optimizer fallback is allowed.
4. Re-run nested signal validation on the frozen seven strict formation pairs,
   with no quota fills and no expected-return floor.

The saved Yahoo snapshot still uses current S&P 500 membership and current GICS
historically.  Results therefore remain survivorship-biased research, not a
production claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from scipy.stats import fisher_exact, mannwhitneyu, norm, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

from equity_pairs.allocation import AllocationError, allocate_pair_sleeves
from equity_pairs.backtest import PairModel, performance_metrics, run_pair_backtest
from equity_pairs.config import StrategyConfig
from equity_pairs.grid_search import GridSearchSpec, build_parameter_return_cache
from equity_pairs.pair_health import (
    PairHealthConfig,
    PairHealthReference,
    gate_target_positions,
    run_pair_health,
)
from equity_pairs.statistics import PairScreenSettings, screen_sector_pairs
from equity_pairs.validation import NestedWalkForwardSpec, run_nested_walk_forward
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
CASH_PAIR = "__CASH__"


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.{digits}f}%" if math.isfinite(number) else "—"


def _pp(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100:.{digits}f} pp" if math.isfinite(number) else "—"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _integer(value: Any) -> str:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"
    return f"{number:,}"


def _table(frame: pd.DataFrame, *, index: bool = False) -> str:
    return frame.to_html(
        index=index,
        border=0,
        classes="data-table",
        justify="left",
        escape=True,
        na_rep="—",
    )


def _image_decodable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        image = plt.imread(path)
    except (OSError, ValueError):
        return False
    return bool(image.size and image.ndim in (2, 3))


def _cagr(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return math.nan
    terminal = float(np.prod(1.0 + values.to_numpy(dtype=float)))
    if terminal <= 0:
        return -1.0
    return terminal ** (252.0 / len(values)) - 1.0


def _annual_log_growth(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().clip(lower=-0.999999)
    return float(np.log1p(values).mean() * 252.0) if len(values) else math.nan


def _metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    return performance_metrics(values, (1.0 + values).cumprod())


def _circular_locations(
    observations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    starts = rng.integers(
        0,
        observations,
        size=int(math.ceil(observations / block_size)),
    )
    offsets = np.arange(block_size, dtype=int)
    return ((starts[:, None] + offsets[None, :]) % observations).ravel()[
        :observations
    ]


def _bootstrap_log_gap(
    strict_returns: pd.Series,
    filler_returns: pd.Series,
    *,
    samples: int,
    block_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    aligned = pd.concat(
        [strict_returns.rename("strict"), filler_returns.rename("filler")],
        axis=1,
    ).dropna()
    contrast = (
        np.log1p(aligned["strict"].clip(lower=-0.999999))
        - np.log1p(aligned["filler"].clip(lower=-0.999999))
    ).to_numpy(dtype=float)
    observed = float(contrast.mean() * 252.0)
    centered = contrast - contrast.mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    null_draws = np.empty(samples, dtype=float)
    effective_block = min(block_size, len(contrast))
    for location in range(samples):
        indices = _circular_locations(len(contrast), effective_block, rng)
        draws[location] = float(contrast[indices].mean() * 252.0)
        null_draws[location] = float(centered[indices].mean() * 252.0)
    two_sided = float(
        (1 + np.sum(np.abs(null_draws) >= abs(observed))) / (samples + 1)
    )
    one_sided = float(
        (1 + np.sum(null_draws >= observed)) / (samples + 1)
    )
    summary = {
        "observations": len(contrast),
        "block_size": effective_block,
        "samples": samples,
        "seed": seed,
        "observed_annual_log_gap": observed,
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "one_sided_pvalue": one_sided,
        "two_sided_pvalue": two_sided,
    }
    return draws, summary


def _gate_category(row: pd.Series) -> str:
    if bool(row["eligible"]):
        return "strict"
    fdr = bool(row["fdr_significant"])
    i1 = bool(row["both_look_i1"])
    if fdr and not i1:
        return "FDR pass / I(1) fail"
    if not fdr and i1:
        return "FDR fail / I(1) pass"
    if not fdr and not i1:
        return "FDR fail / I(1) fail"
    return "other ineligible"


def _sector_balanced_group(
    wide: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    strict: bool,
    sectors: list[str],
) -> pd.Series:
    pieces: dict[str, pd.Series] = {}
    for sector in sectors:
        names = metadata.loc[
            metadata["sector"].eq(sector) & metadata["eligible"].eq(strict),
            "pair",
        ].astype(str)
        if len(names):
            pieces[sector] = wide.reindex(columns=names).fillna(0.0).mean(axis=1)
    return pd.DataFrame(pieces).mean(axis=1)


def _exact_sector_randomization(
    pair_growth: pd.Series,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    strict_counts = (
        metadata.loc[metadata["eligible"]]
        .groupby("sector")["pair"]
        .size()
        .sort_index()
    )
    sectors = strict_counts.index.tolist()
    sector_payloads: list[tuple[str, np.ndarray, int, tuple[tuple[int, ...], ...]]] = []
    observed_assignment: list[tuple[int, ...]] = []
    observed_pieces: list[dict[str, Any]] = []
    for sector in sectors:
        pairs = metadata.loc[metadata["sector"].eq(sector), "pair"].astype(str).tolist()
        values = pair_growth.reindex(pairs).to_numpy(dtype=float)
        strict_pairs = set(
            metadata.loc[
                metadata["sector"].eq(sector) & metadata["eligible"], "pair"
            ].astype(str)
        )
        observed_indices = tuple(
            index for index, pair in enumerate(pairs) if pair in strict_pairs
        )
        observed_assignment.append(observed_indices)
        k = int(strict_counts.loc[sector])
        choices = tuple(itertools.combinations(range(len(pairs)), k))
        selected = values[list(observed_indices)]
        remaining = np.delete(values, list(observed_indices))
        gap = float(selected.mean() - remaining.mean())
        observed_pieces.append(
            {
                "sector": sector,
                "strict_count": k,
                "filler_count": len(pairs) - k,
                "annual_log_growth_gap": gap,
                "assignment_count": len(choices),
            }
        )
        sector_payloads.append((sector, values, k, choices))
    independently_computed_observed = float(
        np.mean([row["annual_log_growth_gap"] for row in observed_pieces])
    )
    total_assignments = int(
        np.prod([len(payload[3]) for payload in sector_payloads])
    )
    distribution = np.empty(total_assignments, dtype=float)
    location = 0
    observed_location: int | None = None
    for combination in itertools.product(
        *(payload[3] for payload in sector_payloads)
    ):
        gaps: list[float] = []
        for payload, chosen_indices in zip(
            sector_payloads, combination, strict=True
        ):
            values = payload[1]
            selected = values[list(chosen_indices)]
            remaining = np.delete(values, list(chosen_indices))
            gaps.append(float(selected.mean() - remaining.mean()))
        distribution[location] = float(np.mean(gaps))
        if combination == tuple(observed_assignment):
            observed_location = location
        location += 1
    if observed_location is None:
        raise RuntimeError("observed sector assignment missing from enumeration")
    observed = float(distribution[observed_location])
    if not math.isclose(
        observed,
        independently_computed_observed,
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise RuntimeError("observed sector statistic failed its internal check")
    one_sided_extreme = int(np.sum(distribution >= observed))
    two_sided_extreme = int(
        np.sum(np.abs(distribution) >= abs(observed))
    )
    summary = {
        "sectors": sectors,
        "assignments": total_assignments,
        "observed_sector_balanced_log_gap": observed,
        "observed_assignment_index": observed_location,
        "one_sided_extreme_assignments": one_sided_extreme,
        "two_sided_extreme_assignments": two_sided_extreme,
        "one_sided_pvalue": float(one_sided_extreme / total_assignments),
        "two_sided_pvalue": float(two_sided_extreme / total_assignments),
        "pvalue_method": (
            "exhaustive assignment ratio; the observed assignment is already "
            "in the distribution, so no Monte Carlo plus-one correction"
        ),
        "statistic_definition": (
            "For each strict-represented sector, mean pair annual log growth "
            "under strict labels minus the mean under remaining labels; then "
            "an unweighted mean across represented sectors. This is not the "
            "daily-rebalanced equal-sector portfolio log-growth gap."
        ),
        "csv_reproduction_note": (
            "Use the persisted boolean observed/one-sided/two-sided flags when "
            "recounting assignments; generic CSV float parsers can move a "
            "boundary value by one machine epsilon."
        ),
        "warning": (
            "Exploratory only: pair labels are not exchangeable because BDX and "
            "GDDY are repeatedly reused by quota-filled pairs."
        ),
    }
    return distribution, summary, pd.DataFrame(observed_pieces)


def _shared_name_audit(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for (sector, role), group in metadata.assign(
        role=np.where(metadata["eligible"], "strict", "quota_filler")
    ).groupby(["sector", "role"], sort=True):
        names = pd.concat(
            [
                group[["pair", "ticker_a"]].rename(columns={"ticker_a": "ticker"}),
                group[["pair", "ticker_b"]].rename(columns={"ticker_b": "ticker"}),
            ],
            ignore_index=True,
        )
        counts = names["ticker"].astype(str).value_counts()
        total_legs = int(counts.sum())
        hhi = float(np.square(counts / total_legs).sum()) if total_legs else math.nan
        for ticker, count in counts.items():
            rows.append(
                {
                    "sector": sector,
                    "role": role,
                    "ticker": ticker,
                    "pair_occurrences": int(count),
                    "pair_count": int(len(group)),
                    "leg_share": float(count / total_legs),
                }
            )
        summaries.append(
            {
                "sector": sector,
                "role": role,
                "pair_count": int(len(group)),
                "unique_names": int(len(counts)),
                "maximum_name_multiplicity": int(counts.max()),
                "leg_hhi": hhi,
                "effective_name_count": float(1.0 / hhi) if hhi > 0 else math.nan,
                "most_reused_name": str(counts.index[0]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _multiplicity_sensitivity(
    diagnostics: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    methods = {
        "BH-FDR": "fdr_bh",
        "Holm": "holm",
        "Bonferroni": "bonferroni",
        "Benjamini-Yekutieli": "fdr_by",
    }
    adjusted: dict[str, pd.Series] = {
        label: pd.Series(np.nan, index=diagnostics.index, dtype=float)
        for label in methods
    }
    for _, indices in diagnostics.groupby("sector").groups.items():
        pvalues = pd.to_numeric(
            diagnostics.loc[indices, "pair_pvalue"], errors="coerce"
        )
        valid = pvalues.notna()
        for label, method in methods.items():
            if valid.any():
                adjusted_values = multipletests(
                    pvalues.loc[valid].to_numpy(dtype=float),
                    alpha=alpha,
                    method=method,
                )[1]
                adjusted[label].loc[pvalues.loc[valid].index] = adjusted_values
    other_gates = (
        diagnostics["screen_status"].eq("ok")
        & diagnostics["positive_hedge_ratio"].fillna(False)
        & diagnostics["both_look_i1"].fillna(False)
        & diagnostics["half_life_days"].between(2.0, 252.0)
        & diagnostics["hurst_exponent"].lt(0.50)
    )
    rows: list[dict[str, Any]] = []
    for label, values in adjusted.items():
        eligible = other_gates & values.le(alpha)
        rows.append(
            {
                "multiplicity_method": label,
                "fully_eligible_pairs": int(eligible.sum()),
                "sectors_with_any": int(diagnostics.loc[eligible, "sector"].nunique()),
                "eligible_pairs": "|".join(
                    diagnostics.loc[eligible, "pair"].astype(str).sort_values()
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_predictiveness(
    metadata: pd.DataFrame,
    pair_growth: pd.Series,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    frame = metadata[["pair", "sector", "pair_pvalue"]].copy()
    frame["annual_log_growth"] = frame["pair"].map(pair_growth)
    frame["within_sector_p_rank"] = frame.groupby("sector")["pair_pvalue"].rank(
        method="average", pct=True
    )
    clean = frame.dropna()
    observed = float(
        spearmanr(
            clean["within_sector_p_rank"],
            clean["annual_log_growth"],
        ).statistic
    )
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=float)
    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in clean.reset_index(drop=True).groupby("sector")
    ]
    ranks = clean["within_sector_p_rank"].to_numpy(dtype=float)
    growth = clean["annual_log_growth"].to_numpy(dtype=float)
    for draw in range(samples):
        permuted = growth.copy()
        for indices in groups:
            permuted[indices] = rng.permutation(permuted[indices])
        null[draw] = float(spearmanr(ranks, permuted).statistic)
    return {
        "observations": len(clean),
        "spearman_rho": observed,
        "samples": samples,
        "seed": seed,
        "one_sided_pvalue": float(
            (1 + np.sum(null >= observed)) / (samples + 1)
        ),
        "two_sided_pvalue": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (samples + 1)
        ),
    }


def _fallback_credibility_audit(
    processed: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metadata = pd.read_csv(processed / "formation_selected_top10_by_sector.csv")
    metadata["eligible"] = metadata["eligible"].astype(bool)
    metadata["selection_role"] = np.where(
        metadata["eligible"], "strict_eligible", "ineligible_quota_fill"
    )
    metadata["tradable_eligible"] = metadata["eligible"]
    metadata["gate_category"] = metadata.apply(_gate_category, axis=1)
    signals = pd.read_csv(
        processed / "pair_daily_signals.csv.gz",
        usecols=["date", "pair", "net_return"],
        parse_dates=["date"],
    )
    wide = signals.pivot(index="date", columns="pair", values="net_return").sort_index()
    strict_pairs = metadata.loc[metadata["eligible"], "pair"].astype(str).tolist()
    filler_pairs = metadata.loc[~metadata["eligible"], "pair"].astype(str).tolist()
    strict_daily = wide.reindex(columns=strict_pairs).fillna(0.0).mean(axis=1)
    filler_daily = wide.reindex(columns=filler_pairs).fillna(0.0).mean(axis=1)
    strict_sectors = sorted(metadata.loc[metadata["eligible"], "sector"].unique())
    matched_filler = _sector_balanced_group(
        wide,
        metadata,
        strict=False,
        sectors=strict_sectors,
    )
    matched_strict = _sector_balanced_group(
        wide,
        metadata,
        strict=True,
        sectors=strict_sectors,
    )

    group_rows: list[dict[str, Any]] = []
    group_series = {
        "Strict 7, all-universe equal pair": strict_daily,
        "Quota filler 103, all-universe equal pair": filler_daily,
        "Strict, equal sector across 3 represented sectors": matched_strict,
        "Quota filler, matched equal sector": matched_filler,
    }
    for label, returns in group_series.items():
        group_rows.append(
            {
                "portfolio": label,
                **_metrics(returns),
                "annual_log_growth": _annual_log_growth(returns),
                "observations": len(returns),
            }
        )
    group_metrics = pd.DataFrame(group_rows)

    pair_rows: list[dict[str, Any]] = []
    for pair in wide.columns:
        returns = wide[pair].fillna(0.0)
        row = metadata.set_index("pair").loc[pair]
        pair_rows.append(
            {
                "pair": pair,
                "sector": row["sector"],
                "eligible": bool(row["eligible"]),
                "selection_role": row["selection_role"],
                "gate_category": row["gate_category"],
                "pair_pvalue": row["pair_pvalue"],
                "fdr_qvalue": row["fdr_qvalue"],
                "half_life_days": row["half_life_days"],
                "hurst_exponent": row["hurst_exponent"],
                "cagr": _cagr(returns),
                "annual_log_growth": _annual_log_growth(returns),
                "sharpe": _metrics(returns)["sharpe"],
                "positive": bool(_cagr(returns) > 0),
            }
        )
    pair_status = pd.DataFrame(pair_rows)
    pair_growth = pair_status.set_index("pair")["annual_log_growth"]

    strict_pair_cagr = pair_status.loc[pair_status["eligible"], "cagr"]
    filler_pair_cagr = pair_status.loc[~pair_status["eligible"], "cagr"]
    welch = ttest_ind(
        strict_pair_cagr,
        filler_pair_cagr,
        equal_var=False,
        nan_policy="omit",
    )
    mann = mannwhitneyu(
        strict_pair_cagr,
        filler_pair_cagr,
        alternative="greater",
    )
    strict_positive = int((strict_pair_cagr > 0).sum())
    filler_positive = int((filler_pair_cagr > 0).sum())
    fisher = fisher_exact(
        [
            [strict_positive, len(strict_pair_cagr) - strict_positive],
            [filler_positive, len(filler_pair_cagr) - filler_positive],
        ],
        alternative="greater",
    )
    standard_error = math.sqrt(
        float(strict_pair_cagr.var(ddof=1) / len(strict_pair_cagr))
        + float(filler_pair_cagr.var(ddof=1) / len(filler_pair_cagr))
    )
    minimum_detectable_effect = float(
        (norm.ppf(0.975) + norm.ppf(0.80)) * standard_error
    )
    pair_test = {
        "strict_mean_pair_cagr": float(strict_pair_cagr.mean()),
        "filler_mean_pair_cagr": float(filler_pair_cagr.mean()),
        "mean_pair_cagr_difference": float(
            strict_pair_cagr.mean() - filler_pair_cagr.mean()
        ),
        "welch_two_sided_pvalue": float(welch.pvalue),
        "mann_whitney_one_sided_pvalue": float(mann.pvalue),
        "strict_positive_pairs": strict_positive,
        "strict_pair_count": len(strict_pair_cagr),
        "filler_positive_pairs": filler_positive,
        "filler_pair_count": len(filler_pair_cagr),
        "fisher_positive_rate_one_sided_pvalue": float(fisher.pvalue),
        "approximate_80pct_power_mde": minimum_detectable_effect,
    }

    bootstrap_draws, bootstrap_summary = _bootstrap_log_gap(
        strict_daily,
        filler_daily,
        samples=bootstrap_samples,
        block_size=20,
        seed=seed,
    )
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_samples = min(10_000, bootstrap_samples)
    for block in (5, 10, 20, 40, 60):
        _, summary = _bootstrap_log_gap(
            strict_daily,
            filler_daily,
            samples=sensitivity_samples,
            block_size=block,
            seed=seed + block,
        )
        sensitivity_rows.append(summary)
    bootstrap_sensitivity = pd.DataFrame(sensitivity_rows)

    exact_distribution, exact_summary, sector_decomposition = (
        _exact_sector_randomization(pair_growth, metadata)
    )
    shared_names, shared_summary = _shared_name_audit(metadata)
    multiplicity = _multiplicity_sensitivity(
        pd.read_csv(processed / "formation_window_all_pair_diagnostics.csv.gz")
    )
    rank_test = _rank_predictiveness(
        metadata,
        pair_growth,
        samples=min(20_000, bootstrap_samples),
        seed=seed + 500,
    )

    category_rows: list[dict[str, Any]] = []
    for category, group in metadata.groupby("gate_category", sort=True):
        pairs = group["pair"].astype(str).tolist()
        returns = wide.reindex(columns=pairs).fillna(0.0).mean(axis=1)
        category_rows.append(
            {
                "gate_category": category,
                "pair_count": len(pairs),
                "cagr": _cagr(returns),
                "annual_log_growth": _annual_log_growth(returns),
                "sharpe": _metrics(returns)["sharpe"],
            }
        )
    gate_categories = pd.DataFrame(category_rows)

    return {
        "metadata": metadata,
        "pair_status": pair_status,
        "wide_returns": wide,
        "strict_daily": strict_daily,
        "filler_daily": filler_daily,
        "matched_strict_daily": matched_strict,
        "matched_filler_daily": matched_filler,
        "group_metrics": group_metrics,
        "pair_test": pair_test,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_summary": bootstrap_summary,
        "bootstrap_sensitivity": bootstrap_sensitivity,
        "exact_distribution": exact_distribution,
        "exact_summary": exact_summary,
        "sector_decomposition": sector_decomposition,
        "shared_names": shared_names,
        "shared_summary": shared_summary,
        "multiplicity": multiplicity,
        "rank_test": rank_test,
        "gate_categories": gate_categories,
    }


def _rebalance_dates(
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    dates = prices.index[prices.index.to_series().between(start, end)]
    firsts = (
        pd.Series(dates, index=dates)
        .groupby(dates.to_period("Q"))
        .first()
        .tolist()
    )
    return pd.DatetimeIndex(sorted(set([dates[0], *firsts])))


def _rolling_strict_screens(
    output: Path,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    workers: int,
    force: bool,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], pd.DataFrame]:
    processed = output / "data" / "processed"
    settings = PairScreenSettings(
        minimum_observations=756,
        minimum_history_ratio=0.90,
        alpha=0.05,
        maximum_half_life_days=252,
        adf_maxlag=5,
        exclude_same_issuer=True,
        workers=workers,
    )
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for rebalance_date in rebalance_dates:
        prior_dates = prices.index[prices.index < rebalance_date]
        if len(prior_dates) < 1_258:
            raise ValueError("rolling strict screen requires 1,258 prior sessions")
        formation_end = prior_dates[-1]
        formation = prices.loc[:formation_end].tail(1_258)
        stamp = pd.Timestamp(rebalance_date).strftime("%Y%m%d")
        cache = processed / f"strict_v5_screen_{stamp}.csv.gz"
        integration_cache = (
            processed / f"strict_v5_screen_{stamp}_ticker_integration.csv"
        )
        if cache.exists() and not force:
            diagnostics = pd.read_csv(cache)
        else:
            diagnostics, integration = screen_sector_pairs(
                formation,
                universe,
                settings,
            )
            diagnostics.to_csv(cache, index=False, compression="gzip")
            integration.to_csv(integration_cache, index=False)
        diagnostics["rebalance_date"] = pd.Timestamp(rebalance_date)
        diagnostics["formation_start"] = formation.index.min()
        diagnostics["formation_end"] = formation.index.max()
        snapshots[pd.Timestamp(rebalance_date)] = diagnostics
        eligible = diagnostics.loc[diagnostics["eligible"].astype(bool)]
        summary_rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "formation_start": formation.index.min(),
                "formation_end": formation.index.max(),
                "candidate_pairs": len(diagnostics),
                "screen_ok": int(diagnostics["screen_status"].eq("ok").sum()),
                "fdr_significant": int(
                    diagnostics["fdr_significant"].astype(bool).sum()
                ),
                "strict_eligible": len(eligible),
                "strict_sectors": int(eligible["sector"].nunique()),
                "strict_pairs": "|".join(eligible["pair"].astype(str)),
            }
        )
        print(
            f"screen {rebalance_date.date()}: {len(eligible)} strict "
            f"across {eligible['sector'].nunique()} sectors",
            flush=True,
        )
    return snapshots, pd.DataFrame(summary_rows)


def _select_disjoint_strict(
    diagnostics: pd.DataFrame,
    incumbent_pairs: set[str],
    *,
    maximum_pairs: int = 10,
    maximum_pairs_per_sector: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = diagnostics.loc[diagnostics["eligible"].astype(bool)].copy()
    strict["incumbent"] = strict["pair"].astype(str).isin(incumbent_pairs)
    strict = strict.sort_values(
        [
            "incumbent",
            "fdr_qvalue",
            "pair_pvalue",
            "half_life_days",
            "pair",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    chosen: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_names: set[str] = set()
    sector_counts: dict[str, int] = {}
    for row in strict.to_dict(orient="records"):
        pair = str(row["pair"])
        sector = str(row["sector"])
        names = {str(row["ticker_a"]), str(row["ticker_b"])}
        reason: str | None = None
        if len(chosen) >= maximum_pairs:
            reason = "portfolio_pair_cap"
        elif names & used_names:
            reason = "shared_underlying_cap"
        elif sector_counts.get(sector, 0) >= maximum_pairs_per_sector:
            reason = "sector_pair_count_cap"
        if reason is not None:
            rejected.append({**row, "rejection_reason": reason})
            continue
        chosen.append(row)
        used_names |= names
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    selected = pd.DataFrame(chosen)
    if len(selected):
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        selected["selection_role"] = "strict_eligible"
        selected["tradable_eligible"] = True
    return selected, pd.DataFrame(rejected)


def _strategy_config(metadata: dict[str, Any]) -> StrategyConfig:
    best = metadata["best_parameters"]
    return StrategyConfig(
        zscore_method=str(best["zscore_method"]),
        zscore_lookback=int(best["zscore_lookback"]),
        zscore_min_periods=int(best["zscore_min_periods"]),
        entry_z=float(best["entry_z"]),
        exit_z=float(best["exit_z"]),
        stop_z=4.0,
        maximum_holding_days=int(best["maximum_holding_days"]),
        transaction_cost_bps=float(metadata["transaction_cost_bps"]),
        annual_short_borrow_bps=float(metadata["annual_short_borrow_bps"]),
    )


def _health_config() -> PairHealthConfig:
    return PairHealthConfig(
        lookback_sessions=252,
        minimum_observations=126,
        decision_lag_sessions=1,
        maximum_cointegration_pvalue=0.10,
        maximum_residual_adf_pvalue=0.10,
        maximum_beta_relative_drift=0.35,
        maximum_half_life_days=126.0,
        maximum_half_life_multiple=3.0,
        minimum_structural_break_pvalue=0.01,
        bad_observations_to_kill=2,
        cooldown_sessions=20,
        good_observations_to_recover=3,
    )


def _canonicalize_signal_legs(
    signal: pd.DataFrame,
    *,
    ticker_a: str,
    ticker_b: str,
    dependent: str,
) -> pd.DataFrame:
    result = signal.copy()
    if dependent == ticker_a:
        return result
    price_a = result["price_independent"].copy()
    price_b = result["price_dependent"].copy()
    weight_a = result["weight_independent"].copy()
    weight_b = result["weight_dependent"].copy()
    result["price_dependent"] = price_a
    result["price_independent"] = price_b
    result["weight_dependent"] = weight_a
    result["weight_independent"] = weight_b
    for prefix in ("return_weight", "raw_weight", "raw_return_weight"):
        dependent_column = f"{prefix}_dependent"
        independent_column = f"{prefix}_independent"
        if {dependent_column, independent_column}.issubset(result.columns):
            canonical_a = result[independent_column].copy()
            canonical_b = result[dependent_column].copy()
            result[dependent_column] = canonical_a
            result[independent_column] = canonical_b
    return result


def _gated_period_signal(
    model_row: pd.Series,
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy: StrategyConfig,
    health_config: PairHealthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = PairModel(
        pair=str(model_row["pair"]),
        sector=str(model_row["sector"]),
        dependent=str(model_row["dependent"]),
        independent=str(model_row["independent"]),
        alpha=float(model_row["alpha"]),
        beta=float(model_row["beta"]),
    )
    available = prices.loc[:end]
    backtest = run_pair_backtest(model, available, start, strategy)
    reference = PairHealthReference(
        pair=model.pair,
        dependent=model.dependent,
        independent=model.independent,
        beta=model.beta,
        half_life_days=float(model_row["half_life_days"]),
    )
    health_result = run_pair_health(reference, available, health_config)
    health = health_result.signals.reindex(backtest.signals.index)
    raw_target = backtest.signals["target_position"].copy()
    raw_effective = backtest.signals["position"].copy()
    target = gate_target_positions(backtest.signals["target_position"], health)
    effective = target.shift(1, fill_value=0.0)
    return_position = effective.shift(1, fill_value=0.0)
    signal = backtest.signals.copy()
    signal["target_position"] = target
    signal["position"] = effective
    signal["return_position"] = return_position
    signal["raw_target_position"] = raw_target
    signal["raw_position"] = raw_effective
    base_dependent = 1.0 / (1.0 + abs(model.beta))
    base_independent = -model.beta / (1.0 + abs(model.beta))
    signal["weight_dependent"] = effective * base_dependent
    signal["weight_independent"] = effective * base_independent
    signal["return_weight_dependent"] = return_position * base_dependent
    signal["return_weight_independent"] = return_position * base_independent
    signal["raw_weight_dependent"] = raw_effective * base_dependent
    signal["raw_weight_independent"] = raw_effective * base_independent
    raw_return_position = backtest.signals["return_position"].copy()
    signal["raw_return_position"] = raw_return_position
    signal["raw_return_weight_dependent"] = raw_return_position * base_dependent
    signal["raw_return_weight_independent"] = raw_return_position * base_independent
    dep_return = signal["price_dependent"].pct_change(fill_method=None).fillna(0.0)
    ind_return = signal["price_independent"].pct_change(fill_method=None).fillna(0.0)
    signal["gross_return"] = (
        signal["return_weight_dependent"] * dep_return
        + signal["return_weight_independent"] * ind_return
    )
    signal["raw_gross_return"] = (
        signal["raw_return_weight_dependent"] * dep_return
        + signal["raw_return_weight_independent"] * ind_return
    )
    short_notional = (
        -signal[["return_weight_dependent", "return_weight_independent"]]
        .clip(upper=0.0)
        .sum(axis=1)
    )
    signal["borrow_cost"] = (
        short_notional * strategy.annual_short_borrow_bps / 10_000.0 / 252.0
    )
    raw_short_notional = (
        -signal[["raw_return_weight_dependent", "raw_return_weight_independent"]]
        .clip(upper=0.0)
        .sum(axis=1)
    )
    signal["raw_borrow_cost"] = (
        raw_short_notional
        * strategy.annual_short_borrow_bps
        / 10_000.0
        / 252.0
    )
    signal["health_state"] = health["health_state"].fillna("missing_fail_closed")
    signal["health_reasons"] = health["health_reasons"].fillna("missing")
    signal["health_allow_entry"] = health["allow_entry"].fillna(False)
    signal["health_force_exit"] = health["force_exit"].fillna(True)
    signal["health_as_of_date"] = health["as_of_date"]
    signal["formation_as_of_date"] = pd.Timestamp(model_row["formation_end"])
    signal["model_alpha"] = model.alpha
    signal["model_beta"] = model.beta
    signal["half_life_days"] = float(model_row["half_life_days"])
    signal["fdr_qvalue"] = float(model_row["fdr_qvalue"])
    signal = _canonicalize_signal_legs(
        signal,
        ticker_a=str(model_row["ticker_a"]),
        ticker_b=str(model_row["ticker_b"]),
        dependent=model.dependent,
    )
    signal = signal.loc[signal.index.to_series().between(start, end)]
    signal.index.name = "date"
    signal.insert(0, "sector", model.sector)
    signal.insert(0, "pair", model.pair)
    health_frame = health.loc[health.index.to_series().between(start, end)].copy()
    health_frame.insert(0, "pair", model.pair)
    health_frame.insert(1, "sector", model.sector)
    events = health_result.events.loc[
        health_result.events.index.to_series().between(start, end)
    ].copy()
    if len(events):
        events.insert(0, "pair", model.pair)
        events.insert(1, "sector", model.sector)
    return signal.reset_index(), health_frame.reset_index(), events.reset_index()


def _pair_market_betas(
    returns: pd.DataFrame,
    market: pd.Series,
) -> pd.Series:
    values: dict[str, float] = {}
    for pair in returns.columns:
        aligned = pd.concat(
            [
                pd.to_numeric(returns[pair], errors="coerce").rename("pair"),
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
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    standardized = factors.reindex(returns.index).apply(pd.to_numeric, errors="coerce")
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=1)
    rows: dict[str, dict[str, float]] = {}
    for pair in returns.columns:
        aligned = pd.concat(
            [returns[pair].rename("pair"), standardized],
            axis=1,
        ).dropna()
        if len(aligned) <= len(standardized.columns) + 2:
            rows[str(pair)] = {
                str(column): 0.0 for column in standardized.columns
            }
            continue
        design = np.column_stack(
            [np.ones(len(aligned)), aligned[standardized.columns].to_numpy()]
        )
        coefficients = np.linalg.lstsq(
            design,
            aligned["pair"].to_numpy(dtype=float),
            rcond=None,
        )[0][1:]
        rows[str(pair)] = dict(
            zip(standardized.columns, coefficients, strict=True)
        )
    return pd.DataFrame.from_dict(rows, orient="index").reindex(returns.columns)


def _training_panels(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    strategy: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_dates = prices.index[prices.index < rebalance_date]
    estimation_index = prior_dates[-504:]
    returns: dict[str, pd.Series] = {}
    positions: dict[str, pd.Series] = {}
    for row in selected.to_dict(orient="records"):
        model = PairModel(
            pair=str(row["pair"]),
            sector=str(row["sector"]),
            dependent=str(row["dependent"]),
            independent=str(row["independent"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
        )
        result = run_pair_backtest(
            model,
            prices.loc[: prior_dates[-1]],
            estimation_index[0],
            strategy,
        )
        returns[model.pair] = result.signals["net_return"].reindex(
            estimation_index
        )
        positions[model.pair] = result.signals["position"].reindex(
            estimation_index
        )
    return (
        pd.DataFrame(returns, index=estimation_index).fillna(0.0),
        pd.DataFrame(positions, index=estimation_index).fillna(0.0),
    )


def _allocation_for_period(
    selected: pd.DataFrame,
    train_returns: pd.DataFrame,
    train_positions: pd.DataFrame,
    market: pd.Series,
    factors: pd.DataFrame,
    previous_absolute: pd.Series | None,
    *,
    turnover_penalty: float,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    pairs = selected["pair"].astype(str).tolist()
    sectors = int(selected["sector"].nunique())
    # Risk capacity is deliberately below the hard 20% pair cap.  Using
    # 20% times the pair count would mechanically force every scarce-candidate
    # allocation to its cap and recreate equal weighting.
    base_risk_budget = min(0.90, 0.15 * len(pairs), 0.35 * sectors)
    if base_risk_budget <= 0:
        return pd.Series(dtype=float), {
            "status": "cash_no_strict_pair",
            "risk_budget": 0.0,
        }, pd.DataFrame()
    metadata = selected.drop_duplicates("pair").copy()
    hedge_beta = metadata.set_index("pair")["beta"].astype(float).reindex(pairs)
    pair_gross = train_positions.reindex(columns=pairs).abs().mean()
    structural_net = (1.0 - hedge_beta) / (1.0 + hedge_beta)
    pair_net = train_positions.reindex(columns=pairs).mean().mul(structural_net)
    market_betas = _pair_market_betas(
        train_returns.reindex(columns=pairs),
        market.reindex(train_returns.index),
    )
    factor_betas = _macro_factor_betas(
        train_returns.reindex(columns=pairs),
        factors,
    )
    incidence = build_gross_name_incidence(metadata)
    attempts: list[str] = []
    for risk_budget in np.arange(base_risk_budget, 0.04, -0.05):
        risk_budget = float(round(risk_budget, 10))
        upper = min(1.0, 0.20 / risk_budget)
        sector_cap = min(1.0, 0.40 / risk_budget)
        name_cap = min(1.0, 0.20 / risk_budget)
        prior: pd.Series | None = None
        effective_penalty = 0.0
        if previous_absolute is not None:
            current_prior = previous_absolute.reindex(pairs).fillna(0.0)
            if float(current_prior.sum()) > 1e-12:
                prior = current_prior / current_prior.sum()
                effective_penalty = turnover_penalty
        try:
            allocation = allocate_pair_sleeves(
                train_returns.reindex(columns=pairs),
                method="minimum_variance",
                pair_betas=market_betas,
                pair_metadata=metadata,
                lower_bounds=0.0,
                upper_bounds=upper,
                sector_caps={
                    str(sector): sector_cap
                    for sector in metadata["sector"].astype(str).unique()
                },
                name_incidence_caps=name_cap,
                name_incidence=incidence,
                pair_gross_exposures=pair_gross,
                gross_exposure_bounds=(0.0, min(1.0, 0.60 / risk_budget)),
                pair_net_exposures=pair_net,
                net_exposure_bounds=(
                    -min(1.0, 0.05 / risk_budget),
                    min(1.0, 0.05 / risk_budget),
                ),
                market_beta_bounds=(
                    -min(1.0, 0.05 / risk_budget),
                    min(1.0, 0.05 / risk_budget),
                ),
                factor_betas=factor_betas,
                factor_beta_bounds={
                    str(factor): (
                        -0.0025 / risk_budget,
                        0.0025 / risk_budget,
                    )
                    for factor in factor_betas.columns
                },
                target_return=None,
                previous_weights=prior,
                turnover_penalty=effective_penalty,
                fallback_method=None,
            )
        except AllocationError as exc:
            attempts.append(
                f"risk_budget={risk_budget:.2f}: {type(exc).__name__}: {exc}"
            )
            continue
        absolute = allocation.weights * risk_budget
        audit = {
            "status": "optimized",
            "risk_budget": risk_budget,
            "cash_target": 1.0 - risk_budget,
            "strict_pair_count": len(pairs),
            "strict_sector_count": sectors,
            "requested_method": allocation.requested_method,
            "effective_method": allocation.effective_method,
            "fallback_used": allocation.diagnostics["fallback_used"],
            "expected_market_beta_absolute": float(
                risk_budget
                * allocation.diagnostics.get("expected_market_beta", math.nan)
            ),
            "expected_volatility_absolute": float(
                risk_budget
                * allocation.diagnostics.get(
                    "expected_annual_volatility", math.nan
                )
            ),
            "turnover_penalty": effective_penalty,
            "previous_weights_supplied": prior is not None,
            "target_one_way_turnover_normalized": allocation.diagnostics.get(
                "one_way_turnover", math.nan
            ),
            "binding_constraint_count": allocation.diagnostics.get(
                "binding_constraint_count", 0
            ),
            "attempt_failures": " || ".join(attempts),
            "expected_return_constraint_present": bool(
                allocation.constraint_diagnostics["constraint_type"]
                .astype(str)
                .str.contains("expected_return")
                .any()
            ),
        }
        return absolute, audit, allocation.constraint_diagnostics
    return pd.Series(dtype=float), {
        "status": "cash_allocator_infeasible",
        "risk_budget": 0.0,
        "cash_target": 1.0,
        "strict_pair_count": len(pairs),
        "strict_sector_count": sectors,
        "attempt_failures": " || ".join(attempts),
        "fallback_used": False,
        "expected_return_constraint_present": False,
    }, pd.DataFrame()


def _flat_pair_rows(
    pair: str,
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    ticker_a, ticker_b = pair.split("__", maxsplit=1)
    legs = prices.reindex(dates)[[ticker_a, ticker_b]].copy()
    legs = legs.ffill()
    if legs.isna().any().any():
        legs = legs.fillna(1.0)
    return pd.DataFrame(
        {
            "pair": pair,
            "date": dates,
            "price_dependent": legs[ticker_a].to_numpy(dtype=float),
            "price_independent": legs[ticker_b].to_numpy(dtype=float),
            "weight_dependent": 0.0,
            "weight_independent": 0.0,
            "gross_return": 0.0,
            "borrow_cost": 0.0,
            "target_position": 0.0,
            "position": 0.0,
            "zscore": np.nan,
            "decision_event": "flat_inactive",
            "health_state": "not_selected",
            "health_allow_entry": False,
            "health_force_exit": True,
            "raw_target_position": 0.0,
            "raw_position": 0.0,
            "raw_weight_dependent": 0.0,
            "raw_weight_independent": 0.0,
            "raw_gross_return": 0.0,
            "raw_borrow_cost": 0.0,
        }
    )


def _cash_rows(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": CASH_PAIR,
            "date": dates,
            "price_dependent": 1.0,
            "price_independent": 1.0,
            "weight_dependent": 0.0,
            "weight_independent": 0.0,
            "gross_return": 0.0,
            "borrow_cost": 0.0,
            "target_position": 0.0,
            "position": 0.0,
            "zscore": np.nan,
            "decision_event": "cash",
            "health_state": "cash",
            "health_allow_entry": False,
            "health_force_exit": False,
            "raw_target_position": 0.0,
            "raw_position": 0.0,
            "raw_weight_dependent": 0.0,
            "raw_weight_independent": 0.0,
            "raw_gross_return": 0.0,
            "raw_borrow_cost": 0.0,
        }
    )


def _build_rolling_strict_book(
    output: Path,
    prices: pd.DataFrame,
    market: pd.Series,
    factors: pd.DataFrame,
    snapshots: dict[pd.Timestamp, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy: StrategyConfig,
    *,
    turnover_penalty: float,
) -> dict[str, Any]:
    health_config = _health_config()
    selected_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    rejected_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    incumbent: set[str] = set()
    for rebalance_date in rebalance_dates:
        selected, rejected = _select_disjoint_strict(
            snapshots[pd.Timestamp(rebalance_date)],
            incumbent,
        )
        if len(selected):
            selected["rebalance_date"] = pd.Timestamp(rebalance_date)
            selected_frames.append(selected)
            incumbent = set(selected["pair"].astype(str))
        else:
            incumbent = set()
        if len(rejected):
            rejected["rebalance_date"] = pd.Timestamp(rebalance_date)
            rejected_frames.append(rejected)
        selected_by_date[pd.Timestamp(rebalance_date)] = selected

    all_dates = prices.index[prices.index.to_series().between(start, end)]
    period_signal_frames: list[pd.DataFrame] = []
    health_frames: list[pd.DataFrame] = []
    health_event_frames: list[pd.DataFrame] = []
    allocation_rows: list[dict[str, Any]] = []
    constraint_frames: list[pd.DataFrame] = []
    period_pair_returns: dict[pd.Timestamp, pd.DataFrame] = {}
    schedule_partial: dict[pd.Timestamp, pd.Series] = {}
    previous_absolute: pd.Series | None = None

    for location, rebalance_date in enumerate(rebalance_dates):
        period_end = (
            rebalance_dates[location + 1] - pd.Timedelta(days=1)
            if location + 1 < len(rebalance_dates)
            else end
        )
        period_dates = all_dates[
            all_dates.to_series().between(rebalance_date, period_end)
        ]
        selected = selected_by_date[pd.Timestamp(rebalance_date)]
        if not len(selected):
            schedule_partial[pd.Timestamp(rebalance_date)] = pd.Series(
                {CASH_PAIR: 1.0}
            )
            allocation_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "estimation_end": prices.index[
                        prices.index < rebalance_date
                    ][-1],
                    "status": "cash_no_strict_pair",
                    "risk_budget": 0.0,
                    "cash_target": 1.0,
                    "strict_pair_count": 0,
                    "strict_sector_count": 0,
                    "fallback_used": False,
                    "expected_return_constraint_present": False,
                    "selected_pairs": "",
                }
            )
            continue
        train_returns, train_positions = _training_panels(
            selected,
            prices,
            pd.Timestamp(rebalance_date),
            strategy,
        )
        period_pair_returns[pd.Timestamp(rebalance_date)] = train_returns
        absolute, audit, constraints = _allocation_for_period(
            selected,
            train_returns,
            train_positions,
            market,
            factors,
            previous_absolute,
            turnover_penalty=turnover_penalty,
        )
        schedule_row = absolute.copy()
        schedule_row.loc[CASH_PAIR] = 1.0 - float(absolute.sum())
        schedule_partial[pd.Timestamp(rebalance_date)] = schedule_row
        previous_absolute = absolute
        allocation_rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "estimation_start": train_returns.index.min(),
                "estimation_end": train_returns.index.max(),
                "selected_pairs": "|".join(selected["pair"].astype(str)),
                **audit,
            }
        )
        if len(constraints):
            constraint_frames.append(
                constraints.assign(rebalance_date=rebalance_date)
            )
        for _, row in selected.iterrows():
            signal, health, events = _gated_period_signal(
                row,
                prices,
                pd.Timestamp(rebalance_date),
                pd.Timestamp(period_dates[-1]),
                strategy,
                health_config,
            )
            signal["rebalance_date"] = pd.Timestamp(rebalance_date)
            signal["selected_this_period"] = True
            period_signal_frames.append(signal)
            health["rebalance_date"] = pd.Timestamp(rebalance_date)
            health_frames.append(health)
            if len(events):
                events["rebalance_date"] = pd.Timestamp(rebalance_date)
                health_event_frames.append(events)

    selected_schedule = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    rejected_schedule = (
        pd.concat(rejected_frames, ignore_index=True)
        if rejected_frames
        else pd.DataFrame()
    )
    union_pairs = sorted(
        set(selected_schedule["pair"].astype(str))
        if len(selected_schedule)
        else set()
    )
    signal_lookup: dict[tuple[str, pd.Timestamp], pd.Series] = {}
    if period_signal_frames:
        active = pd.concat(period_signal_frames, ignore_index=True)
        active["date"] = pd.to_datetime(active["date"])
        signal_lookup = {
            (str(row["pair"]), pd.Timestamp(row["date"])): row
            for row in active.to_dict(orient="records")
        }
    complete_frames: list[pd.DataFrame] = []
    for pair in union_pairs:
        flat = _flat_pair_rows(pair, all_dates, prices)
        for index, row in flat.iterrows():
            key = (pair, pd.Timestamp(row["date"]))
            if key in signal_lookup:
                replacement = signal_lookup[key]
                for column, value in replacement.items():
                    flat.at[index, column] = value
        complete_frames.append(flat)
    complete_frames.append(_cash_rows(all_dates))
    complete_signals = pd.concat(complete_frames, ignore_index=True)
    complete_signals["date"] = pd.to_datetime(complete_signals["date"])
    complete_signals = complete_signals.sort_values(
        ["pair", "date"], kind="mergesort"
    )
    universe_pairs = [*union_pairs, CASH_PAIR]
    schedule = pd.DataFrame(schedule_partial).T
    schedule.index = pd.DatetimeIndex(schedule.index)
    schedule = schedule.reindex(columns=universe_pairs).fillna(0.0)
    schedule.index.name = "rebalance_date"
    if not np.allclose(schedule.sum(axis=1), 1.0):
        raise RuntimeError("strict V5 target schedule does not sum to one")
    initial = schedule.iloc[0]
    backtest = backtest_weighted_pair_book(
        complete_signals,
        initial,
        start=start,
        end=end,
        transaction_cost_bps=strategy.transaction_cost_bps,
        rebalance_frequency="quarterly",
        no_trade_band=0.02,
        maximum_rebalance_turnover=0.10,
        charge_terminal_liquidation=True,
        target_weight_schedule=schedule,
    )
    ungated_signals = complete_signals.copy()
    ungated_signals["weight_dependent"] = ungated_signals[
        "raw_weight_dependent"
    ].fillna(0.0)
    ungated_signals["weight_independent"] = ungated_signals[
        "raw_weight_independent"
    ].fillna(0.0)
    ungated_signals["gross_return"] = ungated_signals[
        "raw_gross_return"
    ].fillna(0.0)
    ungated_signals["borrow_cost"] = ungated_signals[
        "raw_borrow_cost"
    ].fillna(0.0)
    ungated = backtest_weighted_pair_book(
        ungated_signals,
        initial,
        start=start,
        end=end,
        transaction_cost_bps=strategy.transaction_cost_bps,
        rebalance_frequency="quarterly",
        no_trade_band=0.02,
        maximum_rebalance_turnover=0.10,
        charge_terminal_liquidation=True,
        target_weight_schedule=schedule,
    )
    active_pair_count = (
        complete_signals.loc[
            complete_signals["pair"].ne(CASH_PAIR)
            & complete_signals["date"].between(start, end)
        ]
        .groupby("date")["position"]
        .apply(lambda values: int(pd.to_numeric(values).ne(0).sum()))
        .reindex(backtest.daily.index, fill_value=0)
    )
    active_gross = (
        complete_signals.loc[
            complete_signals["pair"].ne(CASH_PAIR)
            & complete_signals["date"].between(start, end)
        ]
        .assign(
            pair_gross=lambda frame: frame[
                ["weight_dependent", "weight_independent"]
            ]
            .abs()
            .sum(axis=1)
        )
        .pivot(index="date", columns="pair", values="pair_gross")
        .reindex(index=backtest.daily.index, columns=union_pairs)
        .fillna(0.0)
        .mul(backtest.deployed_weights.reindex(columns=union_pairs), axis=0)
        .sum(axis=1)
    )
    backtest.daily["active_pair_count"] = active_pair_count
    backtest.daily["active_gross"] = active_gross
    backtest.daily["cash_or_inactive_fraction"] = (1.0 - active_gross).clip(
        lower=0.0, upper=1.0
    )
    return {
        "backtest": backtest,
        "ungated_backtest": ungated,
        "selected_schedule": selected_schedule,
        "rejected_schedule": rejected_schedule,
        "signals": complete_signals,
        "health_daily": (
            pd.concat(health_frames, ignore_index=True)
            if health_frames
            else pd.DataFrame()
        ),
        "health_events": (
            pd.concat(health_event_frames, ignore_index=True)
            if health_event_frames
            else pd.DataFrame()
        ),
        "target_schedule": schedule,
        "allocation_audit": pd.DataFrame(allocation_rows),
        "constraint_audit": (
            pd.concat(constraint_frames, ignore_index=True)
            if constraint_frames
            else pd.DataFrame()
        ),
    }


def _old_nested_contamination(
    processed: Path,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit = pd.read_csv(processed / "production_nested_allocation_audit.csv")
    eligible = metadata.set_index("pair")["eligible"].astype(bool)
    rows: list[dict[str, Any]] = []
    for row in audit.to_dict(orient="records"):
        pairs = str(row["selected_pairs"]).split("|")
        strict_count = int(sum(bool(eligible.get(pair, False)) for pair in pairs))
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "evaluation_start": row["evaluation_start"],
                "evaluation_end": row["evaluation_end"],
                "pair_count": len(pairs),
                "strict_count": strict_count,
                "quota_filler_count": len(pairs) - strict_count,
                "selected_pairs": row["selected_pairs"],
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "evaluation_rows": len(detail),
        "pair_slots": int(detail["pair_count"].sum()),
        "strict_slots": int(detail["strict_count"].sum()),
        "quota_filler_slots": int(detail["quota_filler_count"].sum()),
        "quota_filler_fraction": float(
            detail["quota_filler_count"].sum() / detail["pair_count"].sum()
        ),
    }
    outer = pd.read_csv(processed / "production_nested_outer_evaluations.csv")
    outer_rows: list[dict[str, Any]] = []
    for row in outer.to_dict(orient="records"):
        match = detail.loc[
            detail["candidate_id"].eq(row["candidate_id"])
            & detail["evaluation_start"].eq(row["test_start"])
            & detail["evaluation_end"].eq(row["test_end"])
        ]
        if len(match):
            chosen = match.iloc[-1]
            outer_rows.append(
                {
                    "outer_fold": int(row["outer_fold"]),
                    "strict_count": int(chosen["strict_count"]),
                    "quota_filler_count": int(chosen["quota_filler_count"]),
                    "selected_pairs": chosen["selected_pairs"],
                }
            )
    summary["outer_books"] = outer_rows
    return detail, summary


def _corrected_strict_nested(
    processed: Path,
    prices: pd.DataFrame,
    market: pd.Series,
    strict_metadata: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    strict = strict_metadata.loc[strict_metadata["eligible"].astype(bool)].copy()
    half_life = (
        pd.to_numeric(strict.set_index("pair")["half_life_days"], errors="coerce")
        .clip(lower=2.0)
    )
    formation_weights = 1.0 / half_life
    formation_weights /= formation_weights.sum()
    grid = GridSearchSpec(
        zscore_methods=("sma", "ewma"),
        zscore_lookbacks=(10, 20, 40, 60),
        entry_zs=(1.25, 1.75),
        exit_zs=(0.25, 0.50),
        maximum_holding_days=(20, 40),
    )
    base = StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        maximum_holding_days=60,
        transaction_cost_bps=5.0,
        annual_short_borrow_bps=30.0,
    )
    cache = build_parameter_return_cache(
        strict,
        prices,
        start,
        end,
        base,
        grid,
        formation_weights,
        include_baseline=True,
    )
    incidence = build_gross_name_incidence(strict)
    audit_rows: list[dict[str, Any]] = []

    def evaluator(
        candidate_id: str,
        train_index: pd.Index,
        evaluation_index: pd.Index,
    ) -> pd.Series:
        train = cache.pair_returns[candidate_id].reindex(train_index).fillna(0.0)
        pair_betas = _pair_market_betas(train, market.reindex(train.index))
        allocation = allocate_pair_sleeves(
            train,
            method="minimum_variance",
            pair_betas=pair_betas,
            pair_metadata=strict,
            lower_bounds=0.0,
            upper_bounds=0.25,
            sector_caps={
                str(sector): 0.50
                for sector in strict["sector"].astype(str).unique()
            },
            name_incidence_caps=0.25,
            name_incidence=incidence,
            market_beta_bounds=(-0.15, 0.15),
            target_return=None,
            fallback_method=None,
        )
        audit_rows.append(
            {
                "candidate_id": candidate_id,
                "train_start": train_index[0],
                "train_end": train_index[-1],
                "evaluation_start": evaluation_index[0],
                "evaluation_end": evaluation_index[-1],
                "strict_count": len(strict),
                "quota_filler_count": 0,
                "weights": "|".join(
                    f"{pair}:{allocation.weights[pair]:.8f}"
                    for pair in allocation.weights.index
                ),
                "fallback_used": allocation.diagnostics["fallback_used"],
                "expected_return_constraint_present": bool(
                    allocation.constraint_diagnostics["constraint_type"]
                    .astype(str)
                    .str.contains("expected_return")
                    .any()
                ),
            }
        )
        evaluation = cache.pair_returns[candidate_id].reindex(
            index=evaluation_index,
            columns=allocation.weights.index,
        ).fillna(0.0)
        return evaluation.mul(allocation.weights, axis=1).sum(axis=1)

    spec = NestedWalkForwardSpec(
        outer_train_size=350,
        outer_test_size=63,
        inner_train_size=175,
        inner_validation_size=42,
        purge_size=40,
        embargo_size=5,
        expanding=False,
        maximum_outer_folds=5,
        maximum_inner_folds=3,
        selection_metric="sharpe",
        minimum_evaluation_observations=30,
        bootstrap_samples=1_000,
        bootstrap_block_size=10,
        confidence_level=0.95,
        random_seed=20260901,
    )
    candidates = {
        str(identifier): str(identifier)
        for identifier in cache.returns.columns
    }
    result = run_nested_walk_forward(
        cache.returns.index,
        candidates,
        evaluator,
        spec,
    )
    return {
        "result": result,
        "audit": pd.DataFrame(audit_rows),
        "parameters": cache.parameters,
    }


def _plot_fallback_audit(
    audit: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    curves = {
        "Strict 7": audit["strict_daily"],
        "Quota fillers 103": audit["filler_daily"],
        "Strict, 3-sector balanced": audit["matched_strict_daily"],
        "Fillers, 3-sector matched": audit["matched_filler_daily"],
    }
    for (label, returns), color in zip(
        curves.items(),
        [GREEN, RED, BLUE, ORANGE],
        strict=True,
    ):
        axes[0].plot(
            returns.index,
            (1.0 + returns).cumprod(),
            label=label,
            color=color,
            linewidth=1.8,
        )
    axes[0].axhline(1.0, color="#96a2b1", linewidth=1)
    axes[0].set_title("Original held-out rule: status diagnostics")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8)
    draws = audit["bootstrap_draws"] * 100.0
    observed = audit["bootstrap_summary"]["observed_annual_log_gap"] * 100.0
    axes[1].hist(draws, bins=60, color="#b8c9dc", edgecolor="white")
    axes[1].axvline(observed, color=RED, linewidth=2.2, label="Observed")
    axes[1].axvline(0.0, color="#4f5d6d", linewidth=1)
    axes[1].set_title("20-day block bootstrap: strict minus fillers")
    axes[1].set_xlabel("Annual log-growth gap (percentage points)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_strict_supply(
    screen_summary: pd.DataFrame,
    selected_schedule: pd.DataFrame,
    target_schedule: pd.DataFrame,
    path: Path,
) -> None:
    selected_count = (
        selected_schedule.groupby("rebalance_date")["pair"].nunique()
        if len(selected_schedule)
        else pd.Series(dtype=float)
    )
    dates = pd.to_datetime(screen_summary["rebalance_date"])
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    width = 18
    axes[0].bar(
        dates - pd.Timedelta(days=width / 2),
        screen_summary["strict_eligible"],
        width=width,
        label="Strict before overlap caps",
        color=BLUE,
    )
    axes[0].bar(
        dates + pd.Timedelta(days=width / 2),
        [
            int(selected_count.get(date, 0))
            for date in dates
        ],
        width=width,
        label="Selected disjoint",
        color=TEAL,
    )
    axes[0].axhline(10, color="#9aa5b2", linestyle="--", linewidth=1)
    axes[0].set_title("Lagged rolling strict supply")
    axes[0].set_ylabel("Pair count")
    axes[0].legend(fontsize=8)
    cash = target_schedule[CASH_PAIR].reindex(dates).fillna(1.0)
    axes[1].step(
        dates,
        cash * 100.0,
        where="post",
        color=ORANGE,
        linewidth=2.2,
        label="Target cash",
    )
    axes[1].fill_between(
        dates,
        cash * 100.0,
        step="post",
        alpha=0.22,
        color=ORANGE,
    )
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Capacity rule converts scarcity into cash")
    axes[1].set_ylabel("Target cash (%)")
    axes[1].legend(fontsize=8)
    figure.autofmt_xdate(rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_v5_performance(
    v5: dict[str, Any],
    weighted_daily: pd.DataFrame,
    path: Path,
) -> None:
    health = v5["backtest"].daily
    ungated = v5["ungated_backtest"].daily
    old = weighted_daily.reindex(health.index)
    old_equity = (1.0 + old["portfolio_return"].fillna(0.0)).cumprod()
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    axes[0].plot(
        health.index,
        old_equity,
        label="Old weighted ten: 8/10 quota fillers",
        color=RED,
        linewidth=1.8,
    )
    axes[0].plot(
        ungated.index,
        ungated["equity"],
        label="Rolling strict formation, health off (diagnostic)",
        color=BLUE,
        linewidth=1.8,
    )
    axes[0].plot(
        health.index,
        health["equity"],
        label="Rolling strict + health + capacity cash",
        color=GREEN,
        linewidth=2.2,
    )
    axes[0].axhline(1.0, color="#96a2b1", linewidth=1)
    axes[0].set_title("Final-year retrospective replay")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8)
    axes[1].fill_between(
        health.index,
        0,
        health["cash_or_inactive_fraction"] * 100.0,
        color=ORANGE,
        alpha=0.28,
        label="Cash / inactive fraction",
    )
    axes[1].plot(
        health.index,
        health["active_pair_count"],
        color=PURPLE,
        linewidth=1.5,
        label="Active pairs",
    )
    axes[1].set_title("Actual deployment after daily health controls")
    axes[1].set_ylabel("Percent cash / number of pairs")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_nested_correction(
    old_performance: pd.DataFrame,
    corrected: dict[str, Any],
    path: Path,
) -> None:
    result = corrected["result"]
    returns = result.oos_returns.iloc[:, 0].fillna(0.0)
    old_cagr = float(old_performance.iloc[0]["cagr"])
    new_cagr = float(result.performance.iloc[0]["cagr"])
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    axes[0].plot(
        returns.index,
        (1.0 + returns).cumprod(),
        color=BLUE,
        linewidth=2,
    )
    axes[0].axhline(1.0, color="#96a2b1", linewidth=1)
    axes[0].set_title("Corrected strict-only nested OOS path")
    axes[0].set_ylabel("Growth of $1")
    bars = axes[1].bar(
        ["Old nested\nfallback-heavy", "Corrected nested\nstrict 7 only"],
        [old_cagr * 100.0, new_cagr * 100.0],
        color=[RED, GREEN],
    )
    axes[1].axhline(0.0, color="#657385", linewidth=1)
    axes[1].set_ylabel("CAGR (%)")
    axes[1].set_title("These answer different candidate-universe questions")
    for bar, value in zip(bars, [old_cagr, new_cagr], strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value * 100.0,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


REPORT_TEMPLATE = Template(
    """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fallback credibility correction · equity pairs</title>
<style>
:root{--ink:#20334e;--muted:#66758a;--line:#d9e1ea;--paper:#fff;--blue:#245c97;--teal:#17817d;--red:#bd3d4a;--green:#2d7d5b;--orange:#d77a24}
*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#e9eef4;color:var(--ink);font:15px/1.58 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{width:100%;max-width:1260px;margin:24px auto;background:var(--paper);box-shadow:0 14px 42px rgba(32,52,78,.13);overflow:hidden}header{padding:52px 68px 38px;background:linear-gradient(135deg,#13223b 0%,#245c97 58%,#17817d 100%);color:#fff}.eyebrow{text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:760;opacity:.8}h1{font:700 42px/1.08 Georgia,serif;margin:13px 0 14px;max-width:1050px}header p{font-size:17px;max-width:990px;margin:0;opacity:.91}.content{padding:38px 68px 70px}.content>*,.grid2>*{min-width:0}h2{font:700 29px/1.18 Georgia,serif;margin:50px 0 14px}h3{font-size:18px;margin:29px 0 9px}.lead{font-size:17px;color:#33445b}.verdict{border-left:6px solid var(--red);background:#fff0f1;padding:21px 23px;margin:0 0 25px;font-size:16px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0}.card{min-width:0;border:1px solid var(--line);border-radius:8px;padding:16px;background:#fff}.card .k{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:760}.card .v{font-size:25px;line-height:1.15;font-weight:760;margin:7px 0 3px}.card .s{font-size:12px;color:var(--muted)}.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.note{padding:15px 18px;border-radius:7px;background:#eef3f8;margin:14px 0}.note.bad{background:#fff0f1;border-left:4px solid var(--red)}.note.warn{background:#fff6e9;border-left:4px solid var(--orange)}.note.good{background:#edf8f3;border-left:4px solid var(--green)}.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f4f7fa;border:1px solid var(--line);padding:14px 16px;border-radius:6px;overflow-x:auto}.chart{display:block;width:100%;border:1px solid var(--line);border-radius:7px;margin:18px 0 11px}.scroll{width:100%;min-width:0;overflow-x:auto}.data-table{border-collapse:collapse;width:100%;font-size:12px;margin:13px 0 22px}.data-table th{background:#edf2f7;text-align:left;padding:9px 8px;border-bottom:2px solid #cbd5e1;white-space:nowrap}.data-table td{padding:8px;border-bottom:1px solid #e4e9ef;vertical-align:top}.small{font-size:12px}.muted{color:var(--muted)}code{background:#f1f4f7;padding:2px 5px;border-radius:4px}a{color:#1e61ad}p,li,a,code{overflow-wrap:anywhere}.pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#e8f0fb;color:#1e559b;font-size:11px;font-weight:700;margin:2px 4px 2px 0}.foot{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}
@media(max-width:900px){.page{margin:0}.content,header{padding-left:22px;padding-right:22px}.cards{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:minmax(0,1fr)}h1{font-size:35px}}
@media(max-width:520px){header{padding-top:38px;padding-bottom:30px}.content{padding-top:30px}.cards{grid-template-columns:minmax(0,1fr)}h1{font-size:32px}.verdict{padding:17px 18px}}
@media print{@page{size:A4 landscape;margin:11mm}body{background:#fff}.page{box-shadow:none;margin:0;max-width:none}.chart,.note,.card{break-inside:avoid}a{color:inherit;text-decoration:none}}
</style></head><body><main class="page"><header>
<div class="eyebrow">Equity pairs · credibility correction · {{ generated }}</div>
<h1>The fallback was never a statistical test. This version makes eligibility non-negotiable.</h1>
<p>A non-overwriting correction of the quota-filled selection, the contaminated nested experiment, and the hard-return allocator. It adds a dependence-aware audit, quarterly re-formation across all 13,917 pairs in the frozen current-member universe, capacity cash, daily health controls, and a strict-only nested rerun.</p>
</header><section class="content">
<div class="verdict"><strong>Decision: do not deploy.</strong> Quota fillers are now prohibited from the tradable universe. The strict-versus-filler evidence is weak universe-wide, the favorable sector-matched statistic is not credible independent evidence because it is concentrated in shared-name clusters, and the rolling strict book has too little reliable capacity to establish positive alpha.</div>
<div class="cards">
<div class="card"><div class="k">Formation status</div><div class="v">7 / 110</div><div class="s">strict slots; 103 ineligible quota fills</div></div>
<div class="card"><div class="k">Universe-wide gap</div><div class="v">{{ bootstrap_gap }}</div><div class="s">annual log growth, strict minus fillers</div></div>
<div class="card"><div class="k">95% block interval</div><div class="v">{{ bootstrap_ci }}</div><div class="s">includes zero; two-sided p {{ bootstrap_p }}</div></div>
<div class="card"><div class="k">Actually active</div><div class="v">{{ active_sessions }}/251</div><div class="s">sessions; mean cash/inactive {{ mean_cash }}</div></div>
<div class="card"><div class="k">Rolling strict CAGR</div><div class="v">{{ v5_cagr }}</div><div class="s">health + 5 bp execution + 30 bp borrow</div></div>
<div class="card"><div class="k">Corrected nested</div><div class="v">{{ corrected_nested_cagr }}</div><div class="s">strict seven only; {{ corrected_nested_obs }} OOS sessions</div></div>
</div>

<h2>1 · What was wrong</h2>
<p class="lead">The original code asked for ten rows per sector. When fewer than ten pairs passed the statistical gate, it filled the remaining slots with the lowest raw p-values that had usable prices and hedge coefficients. “Fallback” therefore meant <em>quota filler</em>, not an alternate validation method.</p>
<div class="grid2"><div>
<div class="formula">13,917 within-sector candidates<br>→ 66 sector BH-FDR discoveries<br>→ 7 pass FDR + I(1) + positive beta + half-life + Hurst<br>→ 103 non-passing rows added only to satisfy 10 × 11 quota</div>
</div><div>
<ul><li>80 selected fillers failed FDR only.</li><li>9 passed FDR but failed I(1).</li><li>14 failed both FDR and I(1).</li><li>The old weighted ten held 2 strict and 8 filler pairs.</li><li>The old nested evaluator used {{ old_nested_filler_fraction }} filler pair-slots.</li></ul>
</div></div>
<div class="note bad"><strong>Correct interpretation:</strong> the −0.46% all-110 result is a chronology-respecting historical replay of the quota-filled formation rule, conditional on today's survivor universe. It is not causal inference or a test of a statistically eligible cointegration universe. The old nested result is a mostly-ineligible signal/selection experiment, not nested validation of cointegration.</div>
<div class="scroll">{{ gate_category_table }}</div>

<h2>2 · Does the strict gate predict held-out returns?</h2>
<img class="chart" src="charts/credibility_fallback_audit.png" alt="Strict versus quota filler credibility audit">
<div class="grid2"><div><h3>Portfolio diagnostics</h3><div class="scroll">{{ group_metrics_table }}</div></div><div><h3>Pair-level tests</h3><div class="scroll">{{ pair_test_table }}</div></div></div>
<div class="note warn"><strong>Universe-wide conclusion is inconclusive.</strong> The 20-session common-date circular block bootstrap preserves the realized cross-pair dependence and serial clustering. Its interval spans {{ bootstrap_ci }}, and the approximate 80%-power minimum detectable pair-CAGR effect is {{ pair_mde }}—far larger than the observed mean difference.</div>

<h2>3 · Why the nominal sector-matched p-value is not credible proof</h2>
<p>The exhaustive label enumeration preserves the observed 2 Health Care / 1 Industrials / 4 Information Technology composition and gives a nominal two-sided p-value of {{ exact_p }} across {{ exact_assignments }} assignments. Its statistic is the equal-weighted mean across those three sectors of each sector's strict-label pair-growth mean minus its remaining-pair mean: observed {{ exact_gap }}. It is <strong>not</strong> the daily-rebalanced equal-sector portfolio gap shown above ({{ sector_portfolio_gap }}). That number is exploratory because the “pairs” are not exchangeable independent units.</p>
<div class="grid2"><div><h3>Sector decomposition</h3><div class="scroll">{{ sector_table }}</div></div><div><h3>Shared-name concentration</h3><div class="scroll">{{ cluster_table }}</div></div></div>
<div class="note bad"><strong>The nominal evidence is concentrated in reused names.</strong> All six Information Technology filler pairs reuse GDDY, while seven of eight Health Care fillers reuse BDX. The positive matched result is entirely Information Technology; Health Care and Industrials strict gaps are negative. This is consistent with using shared-underlying caps and retaining the I(1) gate, but it is a post-hoc cluster finding—not confirmatory evidence or a broad alpha claim.</div>

<h2>4 · Multiple-testing and ranking sensitivity</h2>
<div class="grid2"><div><div class="scroll">{{ multiplicity_table }}</div></div><div>
<p>Within-sector formation p-value rank versus held-out annual log growth:</p>
<div class="formula">Spearman ρ = {{ rank_rho }}<br>stratified permutation p = {{ rank_p }} two-sided</div>
<p class="small">BH-FDR is the stored primary rule. Holm/Bonferroni and dependence-robust Benjamini–Yekutieli are sensitivity checks, not after-the-fact replacements.</p>
</div></div>

<h2>5 · Implemented selection correction</h2>
<ul><li><code>tradable_eligible</code> is now authoritative; quota rows are <code>ineligible_quota_fill</code>.</li><li>The new rolling book re-screens all saved 13,917 current-universe pairs at each quarter using only the preceding 1,258 sessions.</li><li>Only strict rows can enter. Shared names are prohibited, sectors are capped at two selected pairs, and fewer than ten is valid.</li><li>Target risk is capped at 20% per pair and 40% per sector. Missing strict capacity becomes explicit zero-yield cash.</li><li>The signal rule is frozen before the final-year replay: {{ signal_rule }}.</li></ul>
<img class="chart" src="charts/credibility_strict_supply.png" alt="Rolling strict candidate supply and cash target">
<div class="scroll">{{ screen_summary_table }}</div>
<div class="scroll">{{ selected_schedule_table }}</div>

<h2>6 · Implemented allocator correction</h2>
<div><span class="pill">minimum Ledoit–Wolf covariance</span><span class="pill">no return floor</span><span class="pill">prior-weight penalty {{ turnover_penalty }}</span><span class="pill">quarterly</span><span class="pill">2 pp no-trade band</span><span class="pill">10% L1 rebalance cap</span><span class="pill">no numerical fallback</span></div>
<p>The allocator enforces pair, sector, shared-name, market-beta, net-exposure, gross-exposure and FRED rate/VIX proxy bounds. If the full target risk is infeasible it reduces risk in predeclared 5-point steps; if no solve is feasible it holds cash. It never imposes an expected-return constraint.</p>
<div class="scroll">{{ allocation_table }}</div>
<div class="note good"><strong>Unstable return forecast removed:</strong> the prior 9.13% sample-mean return floor is absent. The new objective is covariance plus a squared change penalty around the previous target, while the execution layer independently limits realized quarterly turnover.</div>
<div class="note warn"><strong>Penalty activity in this sample: {{ penalty_active_rows }} rebalance rows.</strong> No strict pair survived from one selected quarter to the next, so there was no same-universe prior vector to regularize toward. The feature is implemented and tested, but this replay's slow movement came from the 10% execution turnover cap—not an empirical benefit from the penalty.</div>

<h2>7 · Rolling strict result</h2>
<img class="chart" src="charts/credibility_v5_performance.png" alt="Rolling strict strategy performance and deployment">
<div class="scroll">{{ performance_table }}</div>
<div class="note warn"><strong>Health-off is diagnostic only.</strong> The deployable rule was active on just {{ active_sessions }} of 251 sessions and averaged {{ mean_cash }} cash/inactive exposure. The near-zero result is therefore near-cash, not established alpha. Cash earns zero in this chart so collateral yield cannot masquerade as strategy skill.</div>

<h2>8 · Corrected nested validation</h2>
<img class="chart" src="charts/credibility_nested_correction.png" alt="Corrected strict-only nested validation">
<div class="grid2"><div><h3>Strict-only performance</h3><div class="scroll">{{ nested_performance_table }}</div></div><div><h3>Bootstrap interval</h3><div class="scroll">{{ nested_bootstrap_table }}</div></div></div>
<p>The corrected evaluator fixes the candidate universe to the seven formation-strict pairs, tunes only the predeclared SMA/EWMA signal grid inside purged folds, and estimates unequal covariance-only weights from each training fold. Every audit row asserts seven strict and zero quota fillers; no expected-return or numerical fallback is permitted.</p>
<div class="note warn"><strong>Still not production validation:</strong> the frozen seven span only three sectors, reuse LH and NXPI, use current constituents historically, and omit outer allocation turnover. A genuine production study must re-form the complete point-in-time universe inside every outer fold.</div>

<h2>9 · What the correction establishes</h2>
<div class="grid2"><div><h3>Established</h3><ul><li>Quota fillers never passed a fallback test.</li><li>The favorable strict comparison is statistically weak universe-wide.</li><li>Shared-name clusters materially distort pair counts and nominal inference.</li><li>A strict-only portfolio cannot honestly promise ten pairs or broad sector coverage in this sample.</li><li>The prior hard return floor reduced performance in the saved final-year ablation and lacked forecast support; that is a sample-specific result.</li></ul></div><div><h3>Not established</h3><ul><li>Positive deployable alpha.</li><li>Robust performance under point-in-time membership.</li><li>Broker-verified borrow, locate, spread or market impact.</li><li>A genuinely untouched post-design lockbox.</li><li>Stability outside Yahoo adjusted daily closes.</li></ul></div></div>

<h2>10 · Required next evidence</h2>
<ol><li>Point-in-time S&amp;P membership, historical GICS and delisted securities.</li><li>Rolling re-formation inside every nested outer training fold.</li><li>Licensed quotes/trades, historical borrow fees and locate availability.</li><li>Pre-registered health and allocation parameters, then a new untouched paper-trading period.</li><li>Promotion only if the lower confidence bound, drawdown, capacity, turnover and availability gates all pass.</li></ol>

<h2>11 · Machine-readable artifacts</h2>
<div class="grid2"><div><ul>
<li><a href="data/processed/credibility_pair_status_audit.csv">Pair status and held-out metrics</a></li>
<li><a href="data/processed/credibility_group_metrics.csv">Strict/filler portfolio metrics</a></li>
<li><a href="data/processed/credibility_block_bootstrap.csv.gz">Block-bootstrap draws</a></li>
<li><a href="data/processed/credibility_sector_randomization.csv.gz">Exact label distribution</a></li>
<li><a href="data/processed/credibility_shared_name_clusters.csv">Shared-name audit</a></li>
<li><a href="data/processed/credibility_multiplicity_sensitivity.csv">Multiplicity sensitivity</a></li>
</ul></div><div><ul>
<li><a href="data/processed/strict_v5_rolling_screen_summary.csv">Dated rolling screens</a></li>
<li><a href="data/processed/strict_v5_selection_schedule.csv">Strict selection schedule</a></li>
<li><a href="data/processed/strict_v5_target_weight_schedule.csv">Target weights including cash</a></li>
<li><a href="data/processed/strict_v5_signals.csv.gz">Signals, z-scores and health gates</a></li>
<li><a href="data/processed/strict_v5_daily.csv">Daily portfolio path</a></li>
<li><a href="data/processed/corrected_strict_nested_performance.csv">Corrected nested result</a></li>
<li><a href="fallback_credibility_quality_checks.json">Implementation quality checks</a></li>
</ul></div></div>
<p class="small muted"><code>critical_pass</code> in the quality JSON means the implementation and artifact invariants passed. It is not a statistical-credibility or deployment pass.</p>
<div class="foot">Generated from frozen local artifacts. Final-year rolling decisions satisfy <code>formation_end &lt; rebalance_date</code>. This is quantitative research, not investment advice.</div>
</section></main></body></html>
"""
)


def _format_metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("total_return", "cagr", "annual_volatility", "max_drawdown"):
        if column in result:
            result[column] = result[column].map(_pct)
    for column in ("annual_log_growth",):
        if column in result:
            result[column] = result[column].map(_pct)
    if "sharpe" in result:
        result["sharpe"] = result["sharpe"].map(lambda value: _num(value, 2))
    return result


def _write_report(
    output: Path,
    audit: dict[str, Any],
    screen_summary: pd.DataFrame,
    v5: dict[str, Any],
    old_nested_summary: dict[str, Any],
    corrected: dict[str, Any],
    performance: pd.DataFrame,
    strategy: StrategyConfig,
    turnover_penalty: float,
) -> Path:
    group_display = _format_metrics_table(
        audit["group_metrics"][
            [
                "portfolio",
                "total_return",
                "cagr",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "annual_log_growth",
            ]
        ]
    )
    pair_test_display = pd.DataFrame(
        [
            ["Mean pair CAGR gap", _pp(audit["pair_test"]["mean_pair_cagr_difference"])],
            ["Welch p, two-sided", _num(audit["pair_test"]["welch_two_sided_pvalue"], 3)],
            ["Mann–Whitney p, strict greater", _num(audit["pair_test"]["mann_whitney_one_sided_pvalue"], 3)],
            ["Positive pairs", f"{audit['pair_test']['strict_positive_pairs']}/7 strict vs {audit['pair_test']['filler_positive_pairs']}/103 fillers"],
            ["Fisher p, strict positive rate", _num(audit["pair_test"]["fisher_positive_rate_one_sided_pvalue"], 3)],
            ["Approx. 80% power MDE", _pp(audit["pair_test"]["approximate_80pct_power_mde"])],
        ],
        columns=["Diagnostic", "Value"],
    )
    gate_display = audit["gate_categories"].copy()
    gate_display["cagr"] = gate_display["cagr"].map(_pct)
    gate_display["annual_log_growth"] = gate_display["annual_log_growth"].map(_pct)
    gate_display["sharpe"] = gate_display["sharpe"].map(
        lambda value: _num(value, 2)
    )
    sector_display = audit["sector_decomposition"].copy()
    sector_display["annual_log_growth_gap"] = sector_display[
        "annual_log_growth_gap"
    ].map(_pp)
    cluster_display = audit["shared_summary"].loc[
        audit["shared_summary"]["sector"].isin(
            audit["sector_decomposition"]["sector"]
        ),
        [
            "sector",
            "role",
            "pair_count",
            "unique_names",
            "maximum_name_multiplicity",
            "leg_hhi",
            "most_reused_name",
        ],
    ].copy()
    cluster_display["leg_hhi"] = cluster_display["leg_hhi"].map(
        lambda value: _num(value, 3)
    )
    multiplicity_display = audit["multiplicity"].copy()
    selected_display = v5["selected_schedule"].loc[
        :,
        [
            column
            for column in (
                "rebalance_date",
                "selection_rank",
                "sector",
                "pair",
                "fdr_qvalue",
                "half_life_days",
                "incumbent",
            )
            if column in v5["selected_schedule"].columns
        ],
    ].copy()
    if "fdr_qvalue" in selected_display:
        selected_display["fdr_qvalue"] = selected_display["fdr_qvalue"].map(
            lambda value: _num(value, 4)
        )
    if "half_life_days" in selected_display:
        selected_display["half_life_days"] = selected_display[
            "half_life_days"
        ].map(lambda value: _num(value, 1))
    allocation_display = v5["allocation_audit"].copy()
    allocation_columns = [
        column
        for column in (
            "rebalance_date",
            "estimation_end",
            "strict_pair_count",
            "strict_sector_count",
            "risk_budget",
            "cash_target",
            "status",
            "expected_market_beta_absolute",
            "fallback_used",
            "expected_return_constraint_present",
        )
        if column in allocation_display.columns
    ]
    allocation_display = allocation_display[allocation_columns]
    for column in ("risk_budget", "cash_target"):
        if column in allocation_display:
            allocation_display[column] = allocation_display[column].map(_pct)
    if "expected_market_beta_absolute" in allocation_display:
        allocation_display["expected_market_beta_absolute"] = allocation_display[
            "expected_market_beta_absolute"
        ].map(lambda value: _num(value, 3))
    allocation_display = allocation_display.rename(
        columns={
            "rebalance_date": "Date",
            "estimation_end": "Estimation end",
            "strict_pair_count": "Strict pairs",
            "strict_sector_count": "Sectors",
            "risk_budget": "Risk budget",
            "cash_target": "Cash",
            "status": "Status",
            "expected_market_beta_absolute": "Market |beta|",
            "fallback_used": "Optimizer fallback",
            "expected_return_constraint_present": "Return floor",
        }
    )
    nested_performance = corrected["result"].performance.copy()
    nested_display = _format_metrics_table(nested_performance)
    nested_bootstrap = corrected["result"].bootstrap_confidence_intervals.loc[
        lambda frame: frame["statistic"].isin(
            ["cagr", "sharpe", "max_drawdown"]
        )
    ].copy()
    nested_bootstrap[["estimate", "lower", "upper"]] = nested_bootstrap[
        ["estimate", "lower", "upper"]
    ].astype(object)
    for index in nested_bootstrap.index:
        statistic = nested_bootstrap.at[index, "statistic"]
        for column in ("estimate", "lower", "upper"):
            nested_bootstrap.at[index, column] = (
                _pct(nested_bootstrap.at[index, column])
                if statistic in {"cagr", "max_drawdown"}
                else _num(nested_bootstrap.at[index, column], 3)
            )
    nested_cagr = float(nested_performance.iloc[0]["cagr"])
    active_sessions = int(
        v5["backtest"].daily["active_pair_count"].gt(0).sum()
    )
    mean_cash = float(
        v5["backtest"].daily["cash_or_inactive_fraction"].mean()
    )
    penalty_active_rows = int(
        v5["allocation_audit"]
        .get("previous_weights_supplied", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    group_by_name = audit["group_metrics"].set_index("portfolio")
    sector_portfolio_gap = float(
        group_by_name.loc[
            "Strict, equal sector across 3 represented sectors",
            "annual_log_growth",
        ]
        - group_by_name.loc[
            "Quota filler, matched equal sector",
            "annual_log_growth",
        ]
    )
    report = REPORT_TEMPLATE.render(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        bootstrap_gap=_pp(
            audit["bootstrap_summary"]["observed_annual_log_gap"]
        ),
        bootstrap_ci=(
            f"{_pp(audit['bootstrap_summary']['ci_lower'])} to "
            f"{_pp(audit['bootstrap_summary']['ci_upper'])}"
        ),
        bootstrap_p=_num(
            audit["bootstrap_summary"]["two_sided_pvalue"], 3
        ),
        active_sessions=active_sessions,
        mean_cash=_pct(mean_cash),
        v5_cagr=_pct(v5["backtest"].metrics["cagr"]),
        corrected_nested_cagr=_pct(nested_cagr),
        corrected_nested_obs=_integer(
            nested_performance.iloc[0]["observations"]
        ),
        old_nested_filler_fraction=_pct(
            old_nested_summary["quota_filler_fraction"], 1
        ),
        gate_category_table=_table(gate_display),
        group_metrics_table=_table(group_display),
        pair_test_table=_table(pair_test_display),
        pair_mde=_pp(audit["pair_test"]["approximate_80pct_power_mde"]),
        exact_p=_num(audit["exact_summary"]["two_sided_pvalue"], 4),
        exact_assignments=_integer(audit["exact_summary"]["assignments"]),
        exact_gap=_pp(
            audit["exact_summary"]["observed_sector_balanced_log_gap"]
        ),
        sector_portfolio_gap=_pp(sector_portfolio_gap),
        sector_table=_table(sector_display),
        cluster_table=_table(cluster_display),
        multiplicity_table=_table(multiplicity_display),
        rank_rho=_num(audit["rank_test"]["spearman_rho"], 3),
        rank_p=_num(audit["rank_test"]["two_sided_pvalue"], 3),
        signal_rule=(
            f"{strategy.zscore_method.upper()}{strategy.zscore_lookback}, "
            f"entry {strategy.entry_z:.2f}, exit {strategy.exit_z:.2f}, "
            f"stop {strategy.stop_z:.1f}, max hold "
            f"{strategy.maximum_holding_days}"
        ),
        screen_summary_table=_table(screen_summary),
        selected_schedule_table=_table(selected_display),
        turnover_penalty=_num(turnover_penalty, 3),
        penalty_active_rows=penalty_active_rows,
        allocation_table=_table(allocation_display),
        performance_table=_table(_format_metrics_table(performance)),
        nested_performance_table=_table(nested_display),
        nested_bootstrap_table=_table(nested_bootstrap),
    )
    path = output / "fallback_credibility_report.html"
    path.write_text(report, encoding="utf-8")
    return path


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pairs_research_2026-08-26"),
    )
    parser.add_argument("--screen-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument("--force-rescreen", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("bootstrap-samples must be at least 1,000")
    output = args.output.resolve()
    processed = output / "data" / "processed"
    raw = output / "data" / "raw"
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    audit = _fallback_credibility_audit(
        processed,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    audit["pair_status"].to_csv(
        processed / "credibility_pair_status_audit.csv", index=False
    )
    audit["group_metrics"].to_csv(
        processed / "credibility_group_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "replicate": np.arange(len(audit["bootstrap_draws"])),
            "annual_log_growth_gap": audit["bootstrap_draws"],
        }
    ).to_csv(
        processed / "credibility_block_bootstrap.csv.gz",
        index=False,
        compression="gzip",
    )
    audit["bootstrap_sensitivity"].to_csv(
        processed / "credibility_block_sensitivity.csv", index=False
    )
    exact_distribution = np.asarray(audit["exact_distribution"], dtype=float)
    exact_observed = float(
        audit["exact_summary"]["observed_sector_balanced_log_gap"]
    )
    exact_distribution_frame = pd.DataFrame(
        {
            "assignment": np.arange(len(exact_distribution)),
            "sector_balanced_annual_log_gap": exact_distribution,
            "is_observed_assignment": (
                np.arange(len(exact_distribution))
                == int(audit["exact_summary"]["observed_assignment_index"])
            ),
            "one_sided_extreme": exact_distribution >= exact_observed,
            "two_sided_extreme": (
                np.abs(exact_distribution) >= abs(exact_observed)
            ),
        }
    )
    exact_distribution_frame.to_csv(
        processed / "credibility_sector_randomization.csv.gz",
        index=False,
        compression="gzip",
    )
    audit["sector_decomposition"].to_csv(
        processed / "credibility_sector_decomposition.csv", index=False
    )
    audit["shared_names"].to_csv(
        processed / "credibility_shared_name_clusters.csv", index=False
    )
    audit["shared_summary"].to_csv(
        processed / "credibility_shared_name_summary.csv", index=False
    )
    audit["multiplicity"].to_csv(
        processed / "credibility_multiplicity_sensitivity.csv", index=False
    )
    audit["gate_categories"].to_csv(
        processed / "credibility_gate_category_metrics.csv", index=False
    )
    _json_dump(
        processed / "credibility_statistical_summary.json",
        {
            "bootstrap": audit["bootstrap_summary"],
            "pair_tests": audit["pair_test"],
            "exact_sector_randomization": audit["exact_summary"],
            "formation_rank_predictiveness": audit["rank_test"],
        },
    )

    prices = pd.read_csv(
        raw / "yahoo_adjusted_close_8y.csv",
        index_col="date",
        parse_dates=True,
    ).sort_index()
    universe = pd.read_csv(raw / "sp500_current_constituents.csv")
    grid_metadata = json.loads(
        (processed / "grid_search_metadata.json").read_text(encoding="utf-8")
    )
    start = pd.Timestamp(grid_metadata["lockbox_start"])
    end = min(prices.index.max(), pd.Timestamp(grid_metadata["heldout_end"]))
    strategy = _strategy_config(grid_metadata)
    rebalance_dates = _rebalance_dates(prices, start, end)
    snapshots, screen_summary = _rolling_strict_screens(
        output,
        prices,
        universe,
        rebalance_dates,
        workers=args.screen_workers,
        force=args.force_rescreen,
    )
    screen_summary.to_csv(
        processed / "strict_v5_rolling_screen_summary.csv", index=False
    )
    pd.concat(
        [
            frame.assign(rebalance_date=date)
            for date, frame in snapshots.items()
        ],
        ignore_index=True,
    ).to_csv(
        processed / "strict_v5_rolling_screen_all.csv.gz",
        index=False,
        compression="gzip",
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
    macro_on_prices = macro.reindex(prices.index).ffill()
    factors = pd.DataFrame(
        {
            "rates": pd.to_numeric(
                macro_on_prices["DGS10"], errors="coerce"
            ).diff().div(100.0),
            "vix": pd.to_numeric(
                macro_on_prices["VIXCLS"], errors="coerce"
            ).pct_change(fill_method=None),
        },
        index=prices.index,
    ).replace([np.inf, -np.inf], np.nan)
    v5 = _build_rolling_strict_book(
        output,
        prices,
        market,
        factors,
        snapshots,
        rebalance_dates,
        start,
        end,
        strategy,
        turnover_penalty=args.turnover_penalty,
    )
    v5["selected_schedule"].to_csv(
        processed / "strict_v5_selection_schedule.csv", index=False
    )
    v5["rejected_schedule"].to_csv(
        processed / "strict_v5_rejected_strict_candidates.csv", index=False
    )
    v5["target_schedule"].to_csv(
        processed / "strict_v5_target_weight_schedule.csv"
    )
    v5["allocation_audit"].to_csv(
        processed / "strict_v5_allocator_audit.csv", index=False
    )
    v5["constraint_audit"].to_csv(
        processed / "strict_v5_constraint_audit.csv", index=False
    )
    v5["signals"].to_csv(
        processed / "strict_v5_signals.csv.gz",
        index=False,
        compression="gzip",
    )
    v5["health_daily"].to_csv(
        processed / "strict_v5_health_daily.csv.gz",
        index=False,
        compression="gzip",
    )
    v5["health_events"].to_csv(
        processed / "strict_v5_health_events.csv", index=False
    )
    v5["backtest"].daily.to_csv(processed / "strict_v5_daily.csv")
    v5["ungated_backtest"].daily.to_csv(
        processed / "strict_v5_ungated_daily.csv"
    )
    v5["backtest"].rebalance_log.to_csv(
        processed / "strict_v5_rebalance_log.csv", index=False
    )

    old_nested_detail, old_nested_summary = _old_nested_contamination(
        processed,
        audit["metadata"],
    )
    old_nested_detail.to_csv(
        processed / "credibility_old_nested_contamination.csv", index=False
    )
    _json_dump(
        processed / "credibility_old_nested_contamination_summary.json",
        old_nested_summary,
    )
    corrected = _corrected_strict_nested(
        processed,
        prices,
        market,
        audit["metadata"],
        pd.Timestamp(grid_metadata["heldout_start"]) + pd.Timedelta(days=2),
        end,
    )
    corrected_result = corrected["result"]
    corrected_result.oos_returns.to_csv(
        processed / "corrected_strict_nested_oos_returns.csv"
    )
    corrected_result.performance.to_csv(
        processed / "corrected_strict_nested_performance.csv", index=False
    )
    corrected_result.fold_audit.to_csv(
        processed / "corrected_strict_nested_fold_audit.csv", index=False
    )
    corrected_result.inner_evaluations.to_csv(
        processed / "corrected_strict_nested_inner_evaluations.csv", index=False
    )
    corrected_result.selection_audit.to_csv(
        processed / "corrected_strict_nested_selection_audit.csv", index=False
    )
    corrected_result.outer_evaluations.to_csv(
        processed / "corrected_strict_nested_outer_evaluations.csv", index=False
    )
    corrected_result.bootstrap_confidence_intervals.to_csv(
        processed / "corrected_strict_nested_bootstrap.csv", index=False
    )
    corrected_result.data_snooping_diagnostics.to_csv(
        processed / "corrected_strict_nested_snooping.csv", index=False
    )
    corrected["audit"].to_csv(
        processed / "corrected_strict_nested_allocation_audit.csv", index=False
    )
    corrected["parameters"].to_csv(
        processed / "corrected_strict_nested_parameters.csv", index=False
    )

    weighted_daily = pd.read_csv(
        processed / "weighted_portfolio_daily.csv",
        index_col="date",
        parse_dates=True,
    )
    old_nested_performance = pd.read_csv(
        processed / "production_nested_performance.csv"
    )
    performance_rows = [
        {
            "version": "Original all 110 quota book",
            "period": "2023-08-28 to 2026-08-26",
            **pd.read_csv(processed / "portfolio_metrics.csv")
            .loc[lambda frame: frame["portfolio"].eq("all_formation_selected")]
            .iloc[0]
            .drop("portfolio")
            .to_dict(),
        },
        {
            "version": "Original strict 7 diagnostic",
            "period": "2023-08-28 to 2026-08-26",
            **_metrics(audit["strict_daily"]),
        },
        {
            "version": "Old weighted ten (2 strict / 8 fillers)",
            "period": f"{start.date()} to {end.date()}",
            **_metrics(weighted_daily["portfolio_return"]),
        },
        {
            "version": "Rolling strict, health off diagnostic",
            "period": f"{start.date()} to {end.date()}",
            **v5["ungated_backtest"].metrics,
        },
        {
            "version": "Rolling strict + health + capacity cash",
            "period": f"{start.date()} to {end.date()}",
            **v5["backtest"].metrics,
        },
        {
            "version": "Old nested, fallback-heavy",
            "period": "315 stitched outer sessions",
            **old_nested_performance.iloc[0].to_dict(),
        },
        {
            "version": "Corrected nested, strict 7",
            "period": "315 stitched outer sessions",
            **corrected_result.performance.iloc[0].to_dict(),
        },
    ]
    performance = pd.DataFrame(performance_rows)
    metric_columns = [
        column
        for column in (
            "version",
            "period",
            "total_return",
            "cagr",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
        )
        if column in performance.columns
    ]
    performance = performance[metric_columns]
    performance.to_csv(
        processed / "credibility_corrected_performance_ledger.csv",
        index=False,
    )

    _plot_fallback_audit(
        audit, charts / "credibility_fallback_audit.png"
    )
    _plot_strict_supply(
        screen_summary,
        v5["selected_schedule"],
        v5["target_schedule"],
        charts / "credibility_strict_supply.png",
    )
    _plot_v5_performance(
        v5,
        weighted_daily,
        charts / "credibility_v5_performance.png",
    )
    _plot_nested_correction(
        old_nested_performance,
        corrected,
        charts / "credibility_nested_correction.png",
    )
    report = _write_report(
        output,
        audit,
        screen_summary,
        v5,
        old_nested_summary,
        corrected,
        performance,
        strategy,
        args.turnover_penalty,
    )
    selected_dates = v5["selected_schedule"].copy()
    formation_is_lagged = bool(
        not len(selected_dates)
        or (
            pd.to_datetime(selected_dates["formation_end"])
            < pd.to_datetime(selected_dates["rebalance_date"])
        ).all()
    )
    corrected_audit = corrected["audit"]
    chart_names = (
        "credibility_fallback_audit.png",
        "credibility_strict_supply.png",
        "credibility_v5_performance.png",
        "credibility_nested_correction.png",
    )
    exact_summary = audit["exact_summary"]
    exact_one_sided_recomputes = math.isclose(
        float(exact_summary["one_sided_pvalue"]),
        float(
            exact_summary["one_sided_extreme_assignments"]
            / exact_summary["assignments"]
        ),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    exact_two_sided_recomputes = math.isclose(
        float(exact_summary["two_sided_pvalue"]),
        float(
            exact_summary["two_sided_extreme_assignments"]
            / exact_summary["assignments"]
        ),
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    exact_csv_roundtrip = pd.read_csv(
        processed / "credibility_sector_randomization.csv.gz"
    )
    exact_csv_flags_reproduce = bool(
        len(exact_csv_roundtrip) == int(exact_summary["assignments"])
        and exact_csv_roundtrip["is_observed_assignment"].astype(bool).sum()
        == 1
        and exact_csv_roundtrip["one_sided_extreme"].astype(bool).sum()
        == int(exact_summary["one_sided_extreme_assignments"])
        and exact_csv_roundtrip["two_sided_extreme"].astype(bool).sum()
        == int(exact_summary["two_sided_extreme_assignments"])
    )
    quality_checks = {
        "quality_scope": (
            "Implementation and artifact invariants only; not a statistical "
            "credibility or deployment pass."
        ),
        "statistical_deployment_evidence_pass": False,
        "all_rolling_selected_pairs_are_strict": bool(
            not len(selected_dates)
            or selected_dates["eligible"].astype(bool).all()
        ),
        "rolling_selected_quota_filler_count": int(
            0
            if not len(selected_dates)
            else (~selected_dates["eligible"].astype(bool)).sum()
        ),
        "all_formation_windows_end_before_rebalance": formation_is_lagged,
        "one_target_row_per_quarter": bool(
            not v5["target_schedule"].index.to_period("Q").duplicated().any()
        ),
        "target_rows_sum_to_one": bool(
            np.allclose(v5["target_schedule"].sum(axis=1), 1.0)
        ),
        "target_weights_are_finite_nonnegative": bool(
            np.isfinite(v5["target_schedule"].to_numpy(dtype=float)).all()
            and (v5["target_schedule"] >= 0).all().all()
        ),
        "allocator_expected_return_constraint_count": int(
            v5["allocation_audit"]
            .get(
                "expected_return_constraint_present",
                pd.Series(dtype=bool),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "allocator_numerical_fallback_count": int(
            v5["allocation_audit"]
            .get("fallback_used", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "corrected_nested_quota_filler_slots": int(
            corrected_audit["quota_filler_count"].sum()
        ),
        "corrected_nested_numerical_fallback_count": int(
            corrected_audit["fallback_used"].astype(bool).sum()
        ),
        "corrected_nested_expected_return_constraint_count": int(
            corrected_audit[
                "expected_return_constraint_present"
            ].astype(bool).sum()
        ),
        "old_nested_quota_filler_fraction_disclosed": float(
            old_nested_summary["quota_filler_fraction"]
        ),
        "cash_target_reaches_100pct_when_no_strict_pairs": bool(
            (
                v5["target_schedule"].loc[
                    pd.to_datetime(
                        screen_summary.loc[
                            screen_summary["strict_eligible"].eq(0),
                            "rebalance_date",
                        ]
                    ),
                    CASH_PAIR,
                ]
                == 1.0
            ).all()
        ),
        "active_sessions": int(
            v5["backtest"].daily["active_pair_count"].gt(0).sum()
        ),
        "mean_cash_or_inactive_fraction": float(
            v5["backtest"].daily["cash_or_inactive_fraction"].mean()
        ),
        "exact_randomization_observed_assignment_present": bool(
            0
            <= int(exact_summary["observed_assignment_index"])
            < int(exact_summary["assignments"])
        ),
        "exact_randomization_pvalues_are_exhaustive_ratios": bool(
            exact_one_sided_recomputes and exact_two_sided_recomputes
        ),
        "exact_randomization_csv_flags_reproduce_counts": (
            exact_csv_flags_reproduce
        ),
        "report_exists": report.exists(),
        "report_chart_count": int(
            sum((charts / name).exists() for name in chart_names)
        ),
        "report_chart_decodable_count": int(
            sum(_image_decodable(charts / name) for name in chart_names)
        ),
    }
    critical_pass = bool(
        quality_checks["all_rolling_selected_pairs_are_strict"]
        and quality_checks["rolling_selected_quota_filler_count"] == 0
        and quality_checks["all_formation_windows_end_before_rebalance"]
        and quality_checks["one_target_row_per_quarter"]
        and quality_checks["target_rows_sum_to_one"]
        and quality_checks["target_weights_are_finite_nonnegative"]
        and quality_checks["allocator_expected_return_constraint_count"] == 0
        and quality_checks["allocator_numerical_fallback_count"] == 0
        and quality_checks["corrected_nested_quota_filler_slots"] == 0
        and quality_checks["corrected_nested_numerical_fallback_count"] == 0
        and quality_checks[
            "corrected_nested_expected_return_constraint_count"
        ]
        == 0
        and quality_checks[
            "cash_target_reaches_100pct_when_no_strict_pairs"
        ]
        and quality_checks[
            "exact_randomization_observed_assignment_present"
        ]
        and quality_checks[
            "exact_randomization_pvalues_are_exhaustive_ratios"
        ]
        and quality_checks[
            "exact_randomization_csv_flags_reproduce_counts"
        ]
        and quality_checks["report_exists"]
        and quality_checks["report_chart_count"] == 4
        and quality_checks["report_chart_decodable_count"] == 4
    )
    quality_checks["critical_pass"] = critical_pass
    quality_checks["implementation_invariants_pass"] = critical_pass
    _json_dump(
        output / "fallback_credibility_quality_checks.json",
        quality_checks,
    )
    if not critical_pass:
        raise RuntimeError("fallback credibility quality checks failed")
    metadata_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": str(report),
        "current_membership_survivorship_bias": True,
        "strict_quota_fill_policy": "never tradable; diagnostic only",
        "rolling_formation_sessions": 1_258,
        "rolling_rebalance_dates": [str(date.date()) for date in rebalance_dates],
        "strategy": asdict(strategy),
        "health": asdict(_health_config()),
        "allocator": {
            "method": "minimum_variance",
            "target_return": None,
            "turnover_penalty": args.turnover_penalty,
            "rebalance_frequency": "quarterly",
            "no_trade_band": 0.02,
            "maximum_rebalance_turnover": 0.10,
            "maximum_pair_weight": 0.20,
            "maximum_sector_weight": 0.40,
            "base_risk_budget": (
                "min(90%, 15% per selected strict pair, "
                "35% per represented sector)"
            ),
            "infeasibility_risk_reduction_step": 0.05,
            "fallback_method": None,
        },
        "fallback_audit": {
            "strict_count": 7,
            "quota_filler_count": 103,
            "bootstrap": audit["bootstrap_summary"],
            "exact_sector_randomization": audit["exact_summary"],
        },
        "v5_metrics": v5["backtest"].metrics,
        "v5_ungated_metrics": v5["ungated_backtest"].metrics,
        "corrected_nested_performance": corrected_result.performance.to_dict(
            orient="records"
        ),
        "old_nested_contamination": old_nested_summary,
    }
    _json_dump(
        processed / "fallback_credibility_research_metadata.json",
        metadata_payload,
    )
    print(
        json.dumps(
            {
                "report": str(report),
                "strict_v5_cagr": v5["backtest"].metrics["cagr"],
                "strict_v5_ungated_cagr": v5["ungated_backtest"].metrics[
                    "cagr"
                ],
                "corrected_nested_cagr": float(
                    corrected_result.performance.iloc[0]["cagr"]
                ),
                "strict_screens": screen_summary[
                    ["rebalance_date", "strict_eligible"]
                ].astype(str).to_dict(orient="records"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
