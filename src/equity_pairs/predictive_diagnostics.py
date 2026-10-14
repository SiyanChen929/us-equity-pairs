"""Predictive diagnostics for rolling pair-selection research.

The functions in this module deliberately separate three questions that are
often conflated in backtest reports:

1. Does a training score rank future pair-vintage returns cross-sectionally?
2. Does passing the complete screen-and-tune mechanism have positive expected
   next-quarter return?
3. Does a unique pair repeat its performance over several non-overlapping OOS
   vintages?

Quarter is the inference cluster.  Pair-vintages within the same quarter share
market shocks and are not treated as independent observations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def safe_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Pearson correlation, or NaN for an unidentified correlation."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return rank correlation, or NaN for an unidentified correlation."""

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def quarterly_information_coefficients(
    pair_vintages: pd.DataFrame,
    *,
    scope: str,
    date_column: str = "rebalance_date",
    score_column: str = "training_score",
    return_column: str = "net_total_return",
) -> pd.DataFrame:
    """Calculate one cross-sectional Pearson and Spearman IC per quarter."""

    required = {date_column, score_column, return_column}
    missing = sorted(required - set(pair_vintages.columns))
    if missing:
        raise ValueError("pair_vintages is missing columns: " + ", ".join(missing))
    data = pair_vintages.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce")
    data[return_column] = pd.to_numeric(data[return_column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for date, group in data.sort_values(date_column).groupby(date_column, sort=True):
        clean = group.dropna(subset=[score_column, return_column])
        row: dict[str, Any] = {
            "scope": scope,
            "rebalance_date": pd.Timestamp(date),
            "pair_vintage_count": int(len(clean)),
            "pearson_ic": safe_pearson(clean[score_column], clean[return_column]),
            "spearman_ic": safe_spearman(clean[score_column], clean[return_column]),
            "mean_training_score": float(clean[score_column].mean()) if len(clean) else math.nan,
            "mean_oos_return": float(clean[return_column].mean()) if len(clean) else math.nan,
            "median_oos_return": float(clean[return_column].median()) if len(clean) else math.nan,
            "positive_pair_fraction": float(clean[return_column].gt(0).mean()) if len(clean) else math.nan,
        }
        if "trade_count" in clean:
            row["mean_trade_count"] = float(
                pd.to_numeric(clean["trade_count"], errors="coerce").mean()
            )
            row["zero_trade_fraction"] = float(
                pd.to_numeric(clean["trade_count"], errors="coerce").eq(0).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def circular_block_bootstrap_mean(
    values: Sequence[float],
    *,
    repetitions: int = 20_000,
    block_length: int = 4,
    seed: int = 20260901,
) -> np.ndarray:
    """Circular moving-block bootstrap of a sample mean.

    A four-quarter block retains one year of local dependence.  This is still
    only a sensitivity diagnostic when the number of eligible quarters is
    small.
    """

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.array([], dtype=float)
    if repetitions < 1 or block_length < 1:
        raise ValueError("repetitions and block_length must be positive")
    rng = np.random.default_rng(seed)
    blocks_needed = int(math.ceil(len(array) / block_length))
    starts = rng.integers(0, len(array), size=(repetitions, blocks_needed))
    offsets = np.arange(block_length, dtype=int)
    locations = (starts[:, :, None] + offsets[None, None, :]) % len(array)
    samples = array[locations.reshape(repetitions, -1)[:, : len(array)]]
    return samples.mean(axis=1)


def _hac_mean_test(values: Sequence[float], maximum_lag: int = 4) -> tuple[float, float]:
    """Return Newey-West t statistic and two-sided p value for a nonzero mean."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan, math.nan
    try:
        import statsmodels.api as sm

        model = sm.OLS(array, np.ones((len(array), 1), dtype=float)).fit(
            cov_type="HAC", cov_kwds={"maxlags": min(maximum_lag, len(array) - 1)}
        )
        return float(model.tvalues[0]), float(model.pvalues[0])
    except Exception:
        return math.nan, math.nan


def _permutation_mean_ic(
    groups: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    observed: float,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    """Within-quarter permutation p values for the mean Spearman IC."""

    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for scores, returns in groups:
        score_rank = stats.rankdata(np.asarray(scores, dtype=float))
        return_rank = stats.rankdata(np.asarray(returns, dtype=float))
        score_rank -= score_rank.mean()
        return_rank -= return_rank.mean()
        score_norm = float(np.linalg.norm(score_rank))
        return_norm = float(np.linalg.norm(return_rank))
        if score_norm > 0 and return_norm > 0:
            prepared.append((score_rank / score_norm, return_rank / return_norm))
    if not prepared or not np.isfinite(observed):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    null = np.zeros(repetitions, dtype=float)
    for score_rank, return_rank in prepared:
        permutations = np.vstack(
            [rng.permutation(return_rank) for _ in range(repetitions)]
        )
        null += permutations @ score_rank
    null /= len(prepared)
    adjustment = 1.0 / (repetitions + 1.0)
    one_sided = (np.count_nonzero(null >= observed) + 1.0) * adjustment
    two_sided = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1.0) * adjustment
    return float(one_sided), float(two_sided)


def summarize_information_coefficients(
    pair_vintages: pd.DataFrame,
    quarterly: pd.DataFrame,
    *,
    scope: str,
    minimum_cross_section_sizes: Sequence[int] = (2, 3, 5),
    score_column: str = "training_score",
    return_column: str = "net_total_return",
    repetitions: int = 20_000,
    seed: int = 20260901,
) -> pd.DataFrame:
    """Aggregate ICs with block-bootstrap, HAC, and permutation diagnostics."""

    data = pair_vintages.copy()
    data["rebalance_date"] = pd.to_datetime(data["rebalance_date"])
    raw_pearson = safe_pearson(data[score_column], data[return_column])
    raw_spearman = safe_spearman(data[score_column], data[return_column])
    rows: list[dict[str, Any]] = []
    for minimum_size in minimum_cross_section_sizes:
        eligible = quarterly.loc[
            quarterly["pair_vintage_count"].ge(int(minimum_size))
            & quarterly["spearman_ic"].notna()
        ].sort_values("rebalance_date")
        values = eligible["spearman_ic"].to_numpy(dtype=float)
        mean_ic = float(np.mean(values)) if len(values) else math.nan
        bootstrap = circular_block_bootstrap_mean(
            values,
            repetitions=repetitions,
            block_length=min(4, max(1, len(values))),
            seed=seed + int(minimum_size),
        )
        ci_low, ci_high = (
            tuple(np.quantile(bootstrap, [0.025, 0.975]))
            if len(bootstrap)
            else (math.nan, math.nan)
        )
        naive_t, naive_p = (
            stats.ttest_1samp(values, 0.0) if len(values) >= 2 else (math.nan, math.nan)
        )
        hac_t, hac_p = _hac_mean_test(values)
        eligible_dates = set(pd.to_datetime(eligible["rebalance_date"]))
        groups: list[tuple[np.ndarray, np.ndarray]] = []
        for date, group in data.groupby("rebalance_date", sort=True):
            if pd.Timestamp(date) not in eligible_dates:
                continue
            clean = group.dropna(subset=[score_column, return_column])
            groups.append(
                (
                    clean[score_column].to_numpy(dtype=float),
                    clean[return_column].to_numpy(dtype=float),
                )
            )
        permutation_one, permutation_two = _permutation_mean_ic(
            groups,
            observed=mean_ic,
            repetitions=repetitions,
            seed=seed + 100 + int(minimum_size),
        )
        rows.append(
            {
                "scope": scope,
                "minimum_cross_section_size": int(minimum_size),
                "pair_vintage_count": int(
                    data.loc[data["rebalance_date"].isin(eligible_dates)].shape[0]
                ),
                "eligible_quarter_count": int(len(values)),
                "raw_pooled_pearson": raw_pearson,
                "raw_pooled_spearman": raw_spearman,
                "mean_quarterly_spearman_ic": mean_ic,
                "median_quarterly_spearman_ic": float(np.median(values))
                if len(values)
                else math.nan,
                "quarterly_ic_standard_deviation": float(np.std(values, ddof=1))
                if len(values) > 1
                else math.nan,
                "positive_ic_fraction": float(np.mean(values > 0))
                if len(values)
                else math.nan,
                "block_bootstrap_95ci_low": float(ci_low),
                "block_bootstrap_95ci_high": float(ci_high),
                "naive_t_statistic": float(naive_t),
                "naive_two_sided_pvalue": float(naive_p),
                "newey_west_t_statistic": hac_t,
                "newey_west_two_sided_pvalue": hac_p,
                "within_quarter_permutation_one_sided_pvalue": permutation_one,
                "within_quarter_permutation_two_sided_pvalue": permutation_two,
                "bootstrap_block_length_quarters": min(4, max(1, len(values))),
                "simulation_repetitions": int(repetitions),
            }
        )
    return pd.DataFrame(rows)


def selection_mechanism_by_quarter(
    all_vintages: pd.DataFrame,
    scheduled_quarters: pd.DataFrame,
) -> pd.DataFrame:
    """Return quarterly conditional-return diagnostics for screen and top-ten rules."""

    data = all_vintages.copy()
    data["rebalance_date"] = pd.to_datetime(data["rebalance_date"])
    schedule = scheduled_quarters.copy()
    schedule["rebalance_date"] = pd.to_datetime(schedule["rebalance_date"])
    rows: list[dict[str, Any]] = []
    for record in schedule.sort_values("rebalance_date").to_dict(orient="records"):
        date = pd.Timestamp(record["rebalance_date"])
        group = data.loc[data["rebalance_date"].eq(date)].copy()
        group = group.sort_values(
            ["training_score", "pair"], ascending=[False, True], kind="mergesort"
        )
        selected = group.loc[group["selected_top10"].astype(bool)]
        rest = group.loc[~group["selected_top10"].astype(bool)]
        half = len(group) // 2
        top_half = group.head(half) if half else group.iloc[0:0]
        bottom_half = group.tail(half) if half else group.iloc[0:0]

        def mean_return(frame: pd.DataFrame) -> float:
            return (
                float(pd.to_numeric(frame["net_total_return"], errors="coerce").mean())
                if len(frame)
                else math.nan
            )

        all_mean = mean_return(group)
        selected_mean = mean_return(selected)
        rest_mean = mean_return(rest)
        top_half_mean = mean_return(top_half)
        bottom_half_mean = mean_return(bottom_half)
        rows.append(
            {
                "rebalance_date": date,
                "outer_end_exclusive": pd.Timestamp(record["outer_end_exclusive"]),
                "tuned_strict_count": int(len(group)),
                "selected_top10_count": int(len(selected)),
                "nonselected_count": int(len(rest)),
                "all_pair_mean_return": all_mean if len(group) else 0.0,
                "selected_mean_return": selected_mean if len(selected) else 0.0,
                "nonselected_mean_return": rest_mean,
                "selected_minus_nonselected": selected_mean - rest_mean
                if len(selected) and len(rest)
                else math.nan,
                "top_half_mean_return": top_half_mean,
                "bottom_half_mean_return": bottom_half_mean,
                "top_half_minus_bottom_half": top_half_mean - bottom_half_mean
                if len(top_half) and len(bottom_half)
                else math.nan,
                "all_pair_positive_fraction": float(group["net_total_return"].gt(0).mean())
                if len(group)
                else math.nan,
                "selected_positive_fraction": float(
                    selected["net_total_return"].gt(0).mean()
                )
                if len(selected)
                else math.nan,
                "all_pair_mean_trades": float(group["trade_count"].mean())
                if len(group)
                else math.nan,
                "selected_mean_trades": float(selected["trade_count"].mean())
                if len(selected)
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_conditional_returns(
    all_vintages: pd.DataFrame,
    mechanism: pd.DataFrame,
    *,
    strict_pass_count: int,
    repetitions: int = 20_000,
    seed: int = 20260901,
) -> pd.DataFrame:
    """Summarize E[R|screen] and the incremental top-ten ranking effect."""

    all_data = all_vintages.copy()
    selected = all_data.loc[all_data["selected_top10"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []

    def add_pair_vintage(label: str, frame: pd.DataFrame, denominator: int) -> None:
        returns = pd.to_numeric(frame["net_total_return"], errors="coerce").dropna()
        if denominator < len(returns):
            raise ValueError("pair-vintage denominator cannot be smaller than observed returns")
        padded_returns = np.concatenate(
            [returns.to_numpy(dtype=float), np.zeros(denominator - len(returns), dtype=float)]
        )
        rows.append(
            {
                "estimand": label,
                "observation_unit": "pair_vintage",
                "observation_count": int(denominator),
                "active_observation_count": int(len(returns)),
                "mean_return": float(np.mean(padded_returns)) if denominator else math.nan,
                "median_return": float(np.median(padded_returns)) if denominator else math.nan,
                "positive_fraction": float(np.mean(padded_returns > 0))
                if denominator
                else math.nan,
                "mean_trade_count": float(frame["trade_count"].sum() / denominator)
                if denominator
                else math.nan,
                "block_bootstrap_95ci_low": math.nan,
                "block_bootstrap_95ci_high": math.nan,
                "newey_west_t_statistic": math.nan,
                "newey_west_two_sided_pvalue": math.nan,
            }
        )

    add_pair_vintage("strict_tuned_pair_vintage", all_data, len(all_data))
    add_pair_vintage("selected_top10_pair_vintage", selected, len(selected))
    add_pair_vintage(
        "strict_pass_including_untuned_as_cash",
        all_data,
        int(strict_pass_count),
    )

    series_specs = {
        "screened_pair_mean_active_quarter": mechanism.loc[
            mechanism["tuned_strict_count"].gt(0), "all_pair_mean_return"
        ],
        "screened_pair_mean_all_scheduled_quarters": mechanism[
            "all_pair_mean_return"
        ],
        "selected_top10_mean_active_quarter": mechanism.loc[
            mechanism["selected_top10_count"].gt(0), "selected_mean_return"
        ],
        "selected_top10_mean_all_scheduled_quarters": mechanism[
            "selected_mean_return"
        ],
        "selected_top10_minus_nonselected": mechanism[
            "selected_minus_nonselected"
        ].dropna(),
        "top_half_minus_bottom_half": mechanism[
            "top_half_minus_bottom_half"
        ].dropna(),
    }
    for offset, (label, raw_values) in enumerate(series_specs.items()):
        values = pd.to_numeric(raw_values, errors="coerce").dropna().to_numpy(dtype=float)
        bootstrap = circular_block_bootstrap_mean(
            values,
            repetitions=repetitions,
            block_length=min(4, max(1, len(values))),
            seed=seed + 500 + offset,
        )
        ci_low, ci_high = (
            tuple(np.quantile(bootstrap, [0.025, 0.975]))
            if len(bootstrap)
            else (math.nan, math.nan)
        )
        hac_t, hac_p = _hac_mean_test(values)
        rows.append(
            {
                "estimand": label,
                "observation_unit": "quarter",
                "observation_count": int(len(values)),
                "active_observation_count": int(np.count_nonzero(values)),
                "mean_return": float(np.mean(values)) if len(values) else math.nan,
                "median_return": float(np.median(values)) if len(values) else math.nan,
                "positive_fraction": float(np.mean(values > 0)) if len(values) else math.nan,
                "mean_trade_count": math.nan,
                "block_bootstrap_95ci_low": float(ci_low),
                "block_bootstrap_95ci_high": float(ci_high),
                "newey_west_t_statistic": hac_t,
                "newey_west_two_sided_pvalue": hac_p,
            }
        )
    return pd.DataFrame(rows)


def _annualized_return(total_return: float, sessions: int) -> float:
    ending = 1.0 + float(total_return)
    if sessions < 1 or ending <= 0:
        return math.nan
    return float(ending ** (252.0 / sessions) - 1.0)


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    deviation = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    if not np.isfinite(deviation) or deviation <= 0:
        return math.nan
    return float(values.mean() / deviation * math.sqrt(252.0))


def aggregate_unique_pairs(
    pair_vintages: pd.DataFrame,
    daily_signals: pd.DataFrame,
    trades: pd.DataFrame,
    market_sessions: pd.DatetimeIndex,
    *,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chain all non-overlapping OOS vintages for each unique pair.

    Both active-session annualization and cash-gap-aware CAGR are returned.  A
    full-study cash CAGR makes pairs with different selection frequency directly
    comparable, but it describes an abstention policy rather than continuous
    pair exposure.
    """

    metrics = pair_vintages.copy()
    daily = daily_signals.copy()
    ledger = trades.copy()
    for frame, columns in (
        (metrics, ("rebalance_date",)),
        (daily, ("rebalance_date", "date")),
        (ledger, ("rebalance_date", "entry_date", "exit_date")),
    ):
        for column in columns:
            if column in frame:
                frame[column] = pd.to_datetime(frame[column])
    study_sessions = pd.DatetimeIndex(market_sessions).sort_values().unique()
    if study_sessions.empty:
        raise ValueError("market_sessions cannot be empty")
    rows: list[dict[str, Any]] = []
    vintage_rows: list[dict[str, Any]] = []
    for pair, pair_metrics in metrics.groupby("pair", sort=True):
        pair_metrics = pair_metrics.sort_values("rebalance_date")
        keys = set(
            map(tuple, pair_metrics[["rebalance_date", "pair"]].to_numpy())
        )
        pair_daily = daily.loc[daily["pair"].eq(pair)].copy()
        pair_daily = pair_daily.loc[
            [
                (row.rebalance_date, row.pair) in keys
                for row in pair_daily[["rebalance_date", "pair"]].itertuples(index=False)
            ]
        ].sort_values(["rebalance_date", "date"])
        if pair_daily.duplicated("date").any():
            raise ValueError(f"{pair} has overlapping OOS daily vintages in {scope}")
        returns = pd.Series(
            pd.to_numeric(pair_daily["net_return"], errors="coerce").fillna(0.0).to_numpy(),
            index=pd.DatetimeIndex(pair_daily["date"]),
            dtype=float,
        )
        total_return = float((1.0 + returns).prod() - 1.0)
        first_date = pd.Timestamp(returns.index.min())
        last_date = pd.Timestamp(returns.index.max())
        span_index = study_sessions[(study_sessions >= first_date) & (study_sessions <= last_date)]
        span_returns = pd.Series(0.0, index=span_index, dtype=float)
        span_returns.loc[returns.index] = returns.to_numpy()
        full_returns = pd.Series(0.0, index=study_sessions, dtype=float)
        full_returns.loc[returns.index] = returns.to_numpy()
        equity = (1.0 + returns).cumprod()
        maximum_drawdown = float((equity / equity.cummax() - 1.0).min())
        pair_ledger = ledger.loc[ledger["pair"].eq(pair)].copy()
        pair_ledger = pair_ledger.loc[
            [
                (row.rebalance_date, row.pair) in keys
                for row in pair_ledger[["rebalance_date", "pair"]].itertuples(index=False)
            ]
        ]
        vintage_returns = pd.to_numeric(
            pair_metrics["net_total_return"], errors="coerce"
        )
        for record in pair_metrics.to_dict(orient="records"):
            vintage_rows.append(
                {
                    "scope": scope,
                    "pair": pair,
                    "rebalance_date": record["rebalance_date"],
                    "outer_end_exclusive": record.get("outer_end_exclusive"),
                    "training_score": record.get("training_score"),
                    "net_total_return": record.get("net_total_return"),
                    "trade_count": record.get("trade_count"),
                    "selected_top10": record.get("selected_top10"),
                }
            )
        vintage_count = int(len(pair_metrics))
        if vintage_count >= 4:
            tier = "recurrent_4plus"
        elif vintage_count == 3:
            tier = "recurrent_3"
        elif vintage_count == 2:
            tier = "limited_2"
        else:
            tier = "single_vintage"
        rows.append(
            {
                "scope": scope,
                "pair": pair,
                "sector": "; ".join(sorted(set(pair_metrics["sector"].astype(str)))),
                "vintage_count": vintage_count,
                "positive_vintage_count": int(vintage_returns.gt(0).sum()),
                "first_oos_date": first_date,
                "last_oos_date": last_date,
                "active_sessions": int(len(returns)),
                "span_sessions": int(len(span_returns)),
                "full_study_sessions": int(len(full_returns)),
                "duty_cycle_within_span": float(len(returns) / len(span_returns)),
                "cumulative_net_return": total_return,
                "active_session_annualized_return": _annualized_return(
                    total_return, len(returns)
                ),
                "cash_aware_span_cagr": _annualized_return(
                    total_return, len(span_returns)
                ),
                "full_study_cash_cagr": _annualized_return(
                    total_return, len(full_returns)
                ),
                "active_session_sharpe": _sharpe(returns),
                "cash_aware_span_sharpe": _sharpe(span_returns),
                "chained_max_drawdown": maximum_drawdown,
                "quarter_hit_rate": float(vintage_returns.gt(0).mean()),
                "trade_count": int(len(pair_ledger)),
                "positive_trade_count": int(
                    pd.to_numeric(pair_ledger.get("net_return"), errors="coerce").gt(0).sum()
                )
                if len(pair_ledger)
                else 0,
                "trade_hit_rate": float(
                    pd.to_numeric(pair_ledger["net_return"], errors="coerce").gt(0).mean()
                )
                if len(pair_ledger)
                else math.nan,
                "mean_vintage_return": float(vintage_returns.mean()),
                "median_vintage_return": float(vintage_returns.median()),
                "worst_vintage_return": float(vintage_returns.min()),
                "best_vintage_return": float(vintage_returns.max()),
                "evidence_tier": tier,
                "ex_post_diagnostic_only": True,
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        persistent = result["vintage_count"].ge(4)
        ranks = result.loc[persistent, "cash_aware_span_cagr"].rank(
            ascending=False, method="min"
        )
        result["persistence_rank_if_4plus"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result.loc[persistent, "persistence_rank_if_4plus"] = ranks.astype("Int64")
        result = result.sort_values(
            ["vintage_count", "cash_aware_span_cagr", "pair"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return result, pd.DataFrame(vintage_rows)
