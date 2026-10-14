"""First-knowable-date gating and synchronized pair-selection inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynchronizedBootstrapResult:
    pair_summary: pd.DataFrame
    winner_counts: pd.DataFrame
    sampled_log_growth: np.ndarray
    sampled_ranks: np.ndarray


def apply_first_knowable_gate(
    quarterly: pd.DataFrame,
    master: pd.DataFrame,
    *,
    return_column: str,
    known_date_column: str,
) -> pd.DataFrame:
    """Set all pair returns before the first knowable date to cash.

    The function is intentionally evaluation-only: it never changes a return
    on or after the first knowable date and never fills a missing pair-quarter.
    """

    required_quarterly = {"pair", "rebalance_date", return_column}
    required_master = {"pair", known_date_column}
    if not required_quarterly.issubset(quarterly.columns):
        missing = sorted(required_quarterly.difference(quarterly.columns))
        raise ValueError(f"quarterly panel missing columns: {missing}")
    if not required_master.issubset(master.columns):
        missing = sorted(required_master.difference(master.columns))
        raise ValueError(f"identity master missing columns: {missing}")
    if quarterly.duplicated(["pair", "rebalance_date"]).any():
        raise ValueError("quarterly panel contains duplicate pair/rebalance_date rows")
    known = master[["pair", known_date_column]].copy()
    if known.duplicated("pair").any():
        raise ValueError("identity master contains duplicate pairs")
    known[known_date_column] = pd.to_datetime(known[known_date_column], errors="coerce")
    if known[known_date_column].isna().any():
        raise ValueError("every pair must have a valid first knowable date")
    panel = quarterly.copy()
    panel["rebalance_date"] = pd.to_datetime(panel["rebalance_date"], errors="coerce")
    if panel["rebalance_date"].isna().any():
        raise ValueError("quarterly panel contains invalid rebalance dates")
    panel[return_column] = pd.to_numeric(panel[return_column], errors="coerce")
    if panel[return_column].isna().any():
        raise ValueError("quarterly return column contains missing/non-numeric values")
    if (panel[return_column] <= -1.0).any():
        raise ValueError("quarterly total returns must be greater than -100%")
    panel = panel.merge(known, on="pair", how="left", validate="many_to_one")
    if panel[known_date_column].isna().any():
        missing_pairs = sorted(panel.loc[panel[known_date_column].isna(), "pair"].unique())
        raise ValueError(f"missing knowable dates for pairs: {missing_pairs[:5]}")
    panel["identity_known_at_quarter_start"] = (
        panel["rebalance_date"] >= panel[known_date_column]
    )
    panel["original_quarter_return"] = panel[return_column].astype(float)
    panel["first_knowable_gated_quarter_return"] = np.where(
        panel["identity_known_at_quarter_start"],
        panel["original_quarter_return"],
        0.0,
    )
    panel["pre_knowable_return_removed"] = np.where(
        panel["identity_known_at_quarter_start"],
        0.0,
        panel["original_quarter_return"],
    )
    panel["identity_entry_rule"] = "cash_before_first_knowable_date"
    return panel.sort_values(["rebalance_date", "pair"], kind="mergesort").reset_index(
        drop=True
    )


def balanced_return_matrix(
    panel: pd.DataFrame,
    *,
    return_column: str = "first_knowable_gated_quarter_return",
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """Return a complete date-by-pair matrix in deterministic order."""

    matrix = panel.pivot(index="rebalance_date", columns="pair", values=return_column)
    matrix = matrix.sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("first-knowable panel is not a balanced pair-quarter matrix")
    values = matrix.to_numpy(dtype=float)
    if np.any(values <= -1.0):
        raise ValueError("returns must be greater than -100%")
    return pd.DatetimeIndex(matrix.index), matrix.columns.astype(str).tolist(), values


def first_knowable_pair_metrics(
    panel: pd.DataFrame,
    *,
    calendar_years: float,
    known_date_column: str,
) -> pd.DataFrame:
    """Calculate full-calendar metrics after removing retrospective backfill."""

    rows: list[dict[str, float | int | str]] = []
    for pair, group in panel.groupby("pair", sort=True):
        work = group.sort_values("rebalance_date")
        gated = work["first_knowable_gated_quarter_return"].to_numpy(dtype=float)
        original = work["original_quarter_return"].to_numpy(dtype=float)
        known = work["identity_known_at_quarter_start"].to_numpy(dtype=bool)
        gated_total = float(np.expm1(np.log1p(gated).sum()))
        original_total = float(np.expm1(np.log1p(original).sum()))
        removed_total = float(np.expm1(np.log1p(original[~known]).sum()))
        mean = float(gated.mean())
        standard_deviation = float(gated.std(ddof=1)) if len(gated) > 1 else math.nan
        sharpe = (
            float(math.sqrt(4.0) * mean / standard_deviation)
            if standard_deviation > 0
            else math.nan
        )
        rows.append(
            {
                "pair": str(pair),
                "first_knowable_date": str(
                    pd.Timestamp(work.iloc[0][known_date_column]).date()
                ),
                "scheduled_quarters": int(len(work)),
                "known_quarters": int(known.sum()),
                "pre_knowable_cash_quarters": int((~known).sum()),
                "original_full_path_total_return": original_total,
                "removed_pre_knowable_total_return": removed_total,
                "first_knowable_total_return": gated_total,
                "first_knowable_calendar_cagr": float(
                    np.expm1(np.log1p(gated_total) / calendar_years)
                ),
                "first_knowable_quarterly_sharpe": sharpe,
                "positive_quarter_fraction_full_calendar": float((gated > 0).mean()),
                "median_quarter_return": float(np.median(gated)),
                "worst_quarter_return": float(gated.min()),
                "best_quarter_return": float(gated.max()),
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["first_knowable_calendar_cagr", "pair"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result.insert(0, "observed_rank", np.arange(1, len(result) + 1))
    result["ranking_is_conditional_on_legacy_fixed_114"] = True
    result["point_in_time_universe_passed"] = False
    return result


def circular_block_indices(
    observation_count: int,
    *,
    draws: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    if observation_count < 2:
        raise ValueError("at least two observations are required")
    if draws <= 0 or block_length <= 0:
        raise ValueError("draws and block_length must be positive")
    generator = np.random.default_rng(seed)
    block_count = int(math.ceil(observation_count / block_length))
    starts = generator.integers(0, observation_count, size=(draws, block_count))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % observation_count
    return indices.reshape(draws, -1)[:, :observation_count]


def iid_block_indices(
    block_count: int,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    if block_count < 2 or draws <= 0:
        raise ValueError("block_count >= 2 and draws > 0 are required")
    generator = np.random.default_rng(seed)
    return generator.integers(0, block_count, size=(draws, block_count))


def anniversary_log_blocks(
    dates: Iterable[pd.Timestamp],
    log_returns: np.ndarray,
    *,
    anchor: str | pd.Timestamp,
    block_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    index = pd.DatetimeIndex(dates)
    anchor_date = pd.Timestamp(anchor)
    boundaries = [anchor_date + pd.DateOffset(years=year) for year in range(block_count + 1)]
    labels = np.full(len(index), -1, dtype=int)
    for block in range(block_count):
        labels[(index >= boundaries[block]) & (index < boundaries[block + 1])] = block
    if (labels < 0).any():
        raise ValueError("some quarterly dates fall outside anniversary-year blocks")
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(index):
        raise ValueError("log_returns must be date-by-pair")
    blocks = np.vstack([values[labels == block].sum(axis=0) for block in range(block_count)])
    return blocks, labels


def synchronized_reranking_bootstrap(
    block_log_growth: np.ndarray,
    pairs: list[str],
    sampling_indices: np.ndarray,
    *,
    calendar_years: float,
    top_k: int,
    observed_log_growth: np.ndarray,
) -> SynchronizedBootstrapResult:
    """Resample common time blocks and rerank every pair in every draw."""

    blocks = np.asarray(block_log_growth, dtype=float)
    indices = np.asarray(sampling_indices, dtype=int)
    if blocks.ndim != 2 or blocks.shape[1] != len(pairs):
        raise ValueError("block_log_growth must be block-by-pair")
    if indices.ndim != 2 or indices.min() < 0 or indices.max() >= blocks.shape[0]:
        raise ValueError("sampling_indices are incompatible with block_log_growth")
    draw_count = indices.shape[0]
    pair_count = len(pairs)
    sampled_growth = np.empty((draw_count, pair_count), dtype=np.float64)
    sampled_ranks = np.empty((draw_count, pair_count), dtype=np.int16)
    chunk_size = 500
    for start in range(0, draw_count, chunk_size):
        stop = min(start + chunk_size, draw_count)
        growth = blocks[indices[start:stop]].sum(axis=1)
        sampled_growth[start:stop] = growth
        order = np.argsort(-growth, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.int16)
        row = np.arange(stop - start)[:, None]
        ranks[row, order] = np.arange(1, pair_count + 1, dtype=np.int16)
        sampled_ranks[start:stop] = ranks
    observed_order = np.argsort(-np.asarray(observed_log_growth), kind="stable")
    observed_rank = np.empty(pair_count, dtype=int)
    observed_rank[observed_order] = np.arange(1, pair_count + 1)
    cagr = np.expm1(sampled_growth / float(calendar_years))
    lower, median, upper = np.quantile(cagr, [0.025, 0.50, 0.975], axis=0)
    rank_low, rank_median, rank_high = np.quantile(
        sampled_ranks, [0.05, 0.50, 0.95], axis=0
    )
    summary = pd.DataFrame(
        {
            "pair": pairs,
            "observed_rank": observed_rank,
            "bootstrap_cagr_q025": lower,
            "bootstrap_cagr_median": median,
            "bootstrap_cagr_q975": upper,
            "bootstrap_rank_q05": rank_low,
            "bootstrap_rank_median": rank_median,
            "bootstrap_rank_q95": rank_high,
            "probability_rank_1": (sampled_ranks == 1).mean(axis=0),
            f"probability_top_{top_k}": (sampled_ranks <= top_k).mean(axis=0),
            "draws": draw_count,
            "full_universe_reranked_each_draw": True,
        }
    ).sort_values(["observed_rank", "pair"], kind="mergesort")
    winners = np.argmin(sampled_ranks, axis=1)
    winner_count = np.bincount(winners, minlength=pair_count)
    winner_counts = pd.DataFrame(
        {
            "pair": pairs,
            "winner_draw_count": winner_count,
            "probability_rank_1": winner_count / draw_count,
        }
    ).sort_values(["winner_draw_count", "pair"], ascending=[False, True])
    return SynchronizedBootstrapResult(
        pair_summary=summary.reset_index(drop=True),
        winner_counts=winner_counts.reset_index(drop=True),
        sampled_log_growth=sampled_growth,
        sampled_ranks=sampled_ranks,
    )


def romano_wolf_max_t(
    unit_log_returns: np.ndarray,
    pairs: list[str],
    sampling_indices: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """One-sided Romano-Wolf max-t and White Reality Check sensitivity.

    Bootstrap indices must be synchronized across all pair columns.  Returns
    are centered under the joint null before resampling.
    """

    values = np.asarray(unit_log_returns, dtype=float)
    indices = np.asarray(sampling_indices, dtype=int)
    if values.ndim != 2 or values.shape[1] != len(pairs):
        raise ValueError("unit_log_returns must be observation-by-pair")
    if indices.ndim != 2 or indices.max() >= values.shape[0]:
        raise ValueError("sampling_indices are incompatible with unit returns")
    observed_mean = values.mean(axis=0)
    centered = values - observed_mean
    draws = indices.shape[0]
    bootstrap_mean = np.empty((draws, values.shape[1]), dtype=float)
    chunk_size = 500
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        bootstrap_mean[start:stop] = centered[indices[start:stop]].mean(axis=1)
    standard_error = bootstrap_mean.std(axis=0, ddof=1)
    valid = standard_error > np.finfo(float).eps
    observed_t = np.full(values.shape[1], -np.inf)
    observed_t[valid] = observed_mean[valid] / standard_error[valid]
    bootstrap_t = np.full_like(bootstrap_mean, -np.inf)
    bootstrap_t[:, valid] = bootstrap_mean[:, valid] / standard_error[valid]
    unadjusted = np.ones(values.shape[1])
    unadjusted[valid] = (
        1.0 + (bootstrap_t[:, valid] >= observed_t[valid]).sum(axis=0)
    ) / (draws + 1.0)

    order = np.argsort(-observed_t, kind="stable")
    adjusted = np.ones(values.shape[1])
    running = 0.0
    for location, pair_index in enumerate(order):
        remaining = order[location:]
        bootstrap_max = np.max(bootstrap_t[:, remaining], axis=1)
        step_p = float(
            (1.0 + np.count_nonzero(bootstrap_max >= observed_t[pair_index]))
            / (draws + 1.0)
        )
        running = max(running, step_p)
        adjusted[pair_index] = min(running, 1.0)

    root_n = math.sqrt(values.shape[0])
    observed_rc = float(np.max(root_n * observed_mean))
    bootstrap_rc = np.max(root_n * bootstrap_mean, axis=1)
    rc_pvalue = float(
        (1.0 + np.count_nonzero(bootstrap_rc >= observed_rc)) / (draws + 1.0)
    )
    result = pd.DataFrame(
        {
            "pair": pairs,
            "mean_unit_log_return": observed_mean,
            "bootstrap_standard_error": standard_error,
            "observed_t_statistic": observed_t,
            "unadjusted_bootstrap_pvalue": unadjusted,
            "romano_wolf_adjusted_pvalue": adjusted,
            "reject_at_family_wise_alpha": adjusted <= alpha,
            "family_wise_alpha": alpha,
            "draws": draws,
        }
    ).sort_values(
        ["romano_wolf_adjusted_pvalue", "observed_t_statistic", "pair"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    summary: dict[str, float | int | bool] = {
        "pair_count": len(pairs),
        "observation_count": values.shape[0],
        "draws": draws,
        "white_reality_check_observed_statistic": observed_rc,
        "white_reality_check_pvalue": rc_pvalue,
        "romano_wolf_family_wise_alpha": alpha,
        "romano_wolf_rejection_count": int((adjusted <= alpha).sum()),
        "synchronized_resampling": True,
        "null_centered": True,
    }
    return result.reset_index(drop=True), summary


__all__ = [
    "SynchronizedBootstrapResult",
    "anniversary_log_blocks",
    "apply_first_knowable_gate",
    "balanced_return_matrix",
    "circular_block_indices",
    "first_knowable_pair_metrics",
    "iid_block_indices",
    "romano_wolf_max_t",
    "synchronized_reranking_bootstrap",
]
