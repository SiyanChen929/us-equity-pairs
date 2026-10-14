"""Causal fixed-cardinality pair selection by estimated portfolio Sharpe.

The module is intentionally independent of the V9 ranking engine.  Callers
must supply the first-known-clean V9 daily and quarterly panels, plus the
information cutoff for one rebalance.  Every statistic is then rebuilt from
the immediately preceding eight completed quarters.

With a positive numerator, maximizing Sharpe is equivalent to maximizing the
ratio of squared expected return to variance.  Binary pair variables ``x``
and RLT-strengthened product variables ``y`` linearize each Dinkelbach
residual problem.  Globally optimal HiGHS MILPs plus direct reconstruction
therefore provide a solver-tolerance global certificate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.covariance import LedoitWolf


MU_DEFINITION = (
    "expm1(4 * (mean(log1p(quarter_return_2bp)) - "
    "std(log1p(quarter_return_2bp), ddof=1) / sqrt(8))) over the immediately "
    "preceding 8 completed quarters"
)


@dataclass(frozen=True)
class MaxSharpeSpec:
    """Frozen rules for the strict eight-quarter selector."""

    completed_quarters: int = 8
    minimum_active_quarters: int = 3
    minimum_trade_entries: int = 10
    top_n: int = 10
    annualization: int = 252
    minimum_daily_observations: int = 480
    covariance_eigenvalue_floor: float = 1e-10
    covariance_relative_eigenvalue_floor: float = 1e-8
    positive_numerator_epsilon: float = 1e-10
    # Legacy names remain accepted but are not used by the Dinkelbach solver.
    sharpe_bisection_tolerance: float = 1e-4
    maximum_bisection_iterations: int = 64
    dinkelbach_maximum_iterations: int = 32
    dinkelbach_residual_tolerance: float = 1e-9
    dinkelbach_ratio_tolerance: float = 1e-10
    dinkelbach_sharpe_certificate_tolerance: float = 1e-3
    solution_tolerance: float = 1e-6
    per_solve_time_limit_seconds: float = 60.0
    mip_relative_gap: float = 0.0

    def validate(self) -> None:
        integer_values = {
            "completed_quarters": self.completed_quarters,
            "minimum_active_quarters": self.minimum_active_quarters,
            "minimum_trade_entries": self.minimum_trade_entries,
            "top_n": self.top_n,
            "annualization": self.annualization,
            "minimum_daily_observations": self.minimum_daily_observations,
            "maximum_bisection_iterations": self.maximum_bisection_iterations,
            "dinkelbach_maximum_iterations": self.dinkelbach_maximum_iterations,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in integer_values.values()
        ):
            raise ValueError("integer optimizer settings must be positive integers")
        if self.completed_quarters != 8:
            raise ValueError("this protocol requires exactly eight completed quarters")
        if self.minimum_active_quarters > self.completed_quarters:
            raise ValueError("minimum_active_quarters exceeds completed_quarters")
        finite_positive = {
            "covariance_eigenvalue_floor": self.covariance_eigenvalue_floor,
            "covariance_relative_eigenvalue_floor": self.covariance_relative_eigenvalue_floor,
            "positive_numerator_epsilon": self.positive_numerator_epsilon,
            "sharpe_bisection_tolerance": self.sharpe_bisection_tolerance,
            "solution_tolerance": self.solution_tolerance,
            "per_solve_time_limit_seconds": self.per_solve_time_limit_seconds,
            "dinkelbach_residual_tolerance": self.dinkelbach_residual_tolerance,
            "dinkelbach_ratio_tolerance": self.dinkelbach_ratio_tolerance,
            "dinkelbach_sharpe_certificate_tolerance": (
                self.dinkelbach_sharpe_certificate_tolerance
            ),
        }
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive.values()):
            raise ValueError("numeric optimizer tolerances must be finite and positive")
        if not math.isfinite(self.mip_relative_gap) or self.mip_relative_gap < 0:
            raise ValueError("mip_relative_gap must be finite and non-negative")


@dataclass(frozen=True)
class TrainingPanel:
    """Strict common-window data and moments for one rebalance."""

    status: str
    ranking_effective_date: pd.Timestamp
    information_cutoff: pd.Timestamp
    quarter_numbers: tuple[int, ...]
    training_start: pd.Timestamp | None
    training_end: pd.Timestamp | None
    candidate_stats: pd.DataFrame
    daily_returns: pd.DataFrame
    annual_covariance: pd.DataFrame
    covariance_shrinkage: float | None
    covariance_floor_applied: float | None
    mu_definition: str = MU_DEFINITION


@dataclass(frozen=True)
class CardinalityResult:
    maximum_feasible_k: int
    solver_status: dict[str, Any]


@dataclass(frozen=True)
class SelectionResult:
    """One optimized book plus its MILP certificate information."""

    method: str
    status: str
    selected_pairs: tuple[str, ...]
    maximum_feasible_k: int
    selected_count: int
    expected_annual_return_sum: float | None
    portfolio_variance_sum: float | None
    estimated_sharpe: float | None
    feasible_lower: float | None
    infeasible_upper: float | None
    solver_statuses: tuple[dict[str, Any], ...]
    dinkelbach_iterations: int | None = None
    final_maximum_residual: float | None = None
    final_maximum_scaled_residual: float | None = None
    optimal_ratio_squared_lower: float | None = None
    optimal_ratio_squared_upper: float | None = None
    sharpe_certificate_lower: float | None = None
    sharpe_certificate_upper: float | None = None
    sharpe_certificate_width: float | None = None
    variance_lower_bound: float | None = None
    mu_definition: str = MU_DEFINITION


@dataclass(frozen=True)
class WalkForwardSelection:
    training: TrainingPanel
    max_sharpe: SelectionResult
    matched_score: SelectionResult


_CANDIDATE_COLUMNS = [
    "pair",
    "ticker_a",
    "ticker_b",
    "eligible_since_date",
    "first_training_quarter_rebalance",
    "history_quarters",
    "active_quarters",
    "trade_entries",
    "mean_quarterly_log_return",
    "quarterly_log_return_std",
    "one_se_annual_log_return",
    "expected_annual_return",
    "mu_definition",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite")
    return result


def _empty_training_panel(
    *,
    status: str,
    ranking_effective_date: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    quarter_numbers: tuple[int, ...],
) -> TrainingPanel:
    return TrainingPanel(
        status=status,
        ranking_effective_date=ranking_effective_date,
        information_cutoff=information_cutoff,
        quarter_numbers=quarter_numbers,
        training_start=None,
        training_end=None,
        candidate_stats=pd.DataFrame(columns=_CANDIDATE_COLUMNS),
        daily_returns=pd.DataFrame(dtype=float),
        annual_covariance=pd.DataFrame(dtype=float),
        covariance_shrinkage=None,
        covariance_floor_applied=None,
    )


def build_strict_training_panel(
    daily: pd.DataFrame,
    quarter_history: pd.DataFrame,
    pair_metadata: pd.DataFrame,
    ranking_effective_date: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    specification: MaxSharpeSpec = MaxSharpeSpec(),
    *,
    expected_sessions: pd.DatetimeIndex | None = None,
) -> TrainingPanel:
    """Build the common daily matrix for the prior eight completed quarters.

    A candidate must have all eight immediately preceding schedule quarters,
    all eight must be ``history_available``, and its first-known date must be
    no later than the first quarter's rebalance.  The returned daily matrix is
    complete across candidates and ends exactly on ``information_cutoff``.
    """

    specification.validate()
    effective = pd.Timestamp(ranking_effective_date)
    cutoff = pd.Timestamp(information_cutoff)
    if pd.isna(effective) or pd.isna(cutoff) or cutoff >= effective:
        raise ValueError(
            "information_cutoff must be finite and strictly before ranking_effective_date"
        )

    required_history = {
        "pair",
        "quarter_number",
        "rebalance_date",
        "outer_end_exclusive",
        "quarter_last_session",
        "quarter_return_2bp",
        "active_quarter",
        "trade_entries",
        "history_available",
        "eligible_since_date",
    }
    required_daily = {
        "pair",
        "date",
        "quarter_number",
        "net_return_2bp",
        "price_a",
        "price_b",
    }
    required_metadata = {"pair", "ticker_a", "ticker_b"}
    missing_history = sorted(required_history - set(quarter_history.columns))
    missing_daily = sorted(required_daily - set(daily.columns))
    missing_metadata = sorted(required_metadata - set(pair_metadata.columns))
    if missing_history:
        raise ValueError(f"quarter history missing columns: {missing_history}")
    if missing_daily:
        raise ValueError(f"daily panel missing columns: {missing_daily}")
    if missing_metadata:
        raise ValueError(f"pair metadata missing columns: {missing_metadata}")

    history = quarter_history[list(required_history)].copy()
    history["pair"] = history["pair"].astype(str)
    for column in (
        "rebalance_date",
        "outer_end_exclusive",
        "quarter_last_session",
        "eligible_since_date",
    ):
        history[column] = pd.to_datetime(history[column], errors="coerce")
        if history[column].isna().any():
            raise ValueError(f"quarter history contains invalid {column}")
    history = _finite_numeric(
        history,
        ["quarter_number", "quarter_return_2bp", "trade_entries"],
    )
    history["quarter_number"] = history["quarter_number"].astype(int)
    if history.duplicated(["pair", "quarter_number"]).any():
        raise ValueError("quarter history contains duplicate pair/quarter keys")
    if history["quarter_return_2bp"].le(-1.0).any():
        raise ValueError("quarter returns must be greater than -100%")
    history["history_available"] = _as_bool(history["history_available"])
    history["active_quarter"] = _as_bool(history["active_quarter"])

    definition_columns = [
        "quarter_number",
        "rebalance_date",
        "outer_end_exclusive",
        "quarter_last_session",
    ]
    definition_counts = history.groupby("quarter_number")[
        definition_columns[1:]
    ].nunique(dropna=False)
    if definition_counts.gt(1).any().any():
        raise ValueError("quarter definitions differ across pairs")
    definitions = (
        history[definition_columns]
        .drop_duplicates("quarter_number")
        .sort_values(["outer_end_exclusive", "quarter_number"], kind="mergesort")
    )
    complete = definitions.loc[
        definitions["outer_end_exclusive"].le(effective)
        & definitions["quarter_last_session"].le(cutoff)
    ]
    selected_definitions = complete.tail(specification.completed_quarters).copy()
    quarter_numbers = tuple(
        int(value) for value in selected_definitions["quarter_number"].tolist()
    )
    if len(selected_definitions) < specification.completed_quarters:
        return _empty_training_panel(
            status="insufficient_completed_quarters",
            ranking_effective_date=effective,
            information_cutoff=cutoff,
            quarter_numbers=quarter_numbers,
        )
    if not np.array_equal(
        np.diff(np.asarray(quarter_numbers, dtype=int)),
        np.ones(specification.completed_quarters - 1, dtype=int),
    ):
        raise ValueError("the eight training quarters are not consecutive")
    if pd.Timestamp(selected_definitions.iloc[-1]["quarter_last_session"]) != cutoff:
        raise ValueError("information_cutoff is not the last session of the latest completed quarter")
    if pd.Timestamp(selected_definitions.iloc[-1]["outer_end_exclusive"]) != effective:
        raise ValueError("latest training quarter does not end at ranking_effective_date")

    first_quarter_rebalance = pd.Timestamp(
        selected_definitions.iloc[0]["rebalance_date"]
    )
    selected_history = history.loc[
        history["quarter_number"].isin(quarter_numbers)
    ].copy()
    rows: list[dict[str, Any]] = []
    required_quarters = set(quarter_numbers)
    for pair, group in selected_history.groupby("pair", sort=True):
        work = group.sort_values("quarter_number", kind="mergesort")
        if set(work["quarter_number"].astype(int)) != required_quarters:
            continue
        if len(work) != specification.completed_quarters:
            continue
        first_known = pd.Timestamp(work["eligible_since_date"].iloc[0])
        if work["eligible_since_date"].nunique() != 1:
            raise ValueError(f"eligible_since_date changes within pair {pair}")
        active_quarters = int(work["active_quarter"].sum())
        trade_entries = int(work["trade_entries"].sum())
        if not work["history_available"].all():
            continue
        if first_known > first_quarter_rebalance:
            continue
        if active_quarters < specification.minimum_active_quarters:
            continue
        if trade_entries < specification.minimum_trade_entries:
            continue
        logs = np.log1p(work["quarter_return_2bp"].to_numpy(dtype=float))
        log_mean = float(np.mean(logs))
        log_std = float(np.std(logs, ddof=1))
        annual_log_lower = float(
            4.0
            * (
                log_mean
                - log_std / math.sqrt(specification.completed_quarters)
            )
        )
        expected_return = float(np.expm1(annual_log_lower))
        if not math.isfinite(expected_return):
            raise ValueError(f"expected return is not finite for pair {pair}")
        rows.append(
            {
                "pair": str(pair),
                "eligible_since_date": first_known,
                "first_training_quarter_rebalance": first_quarter_rebalance,
                "history_quarters": specification.completed_quarters,
                "active_quarters": active_quarters,
                "trade_entries": trade_entries,
                "mean_quarterly_log_return": log_mean,
                "quarterly_log_return_std": log_std,
                "one_se_annual_log_return": annual_log_lower,
                "expected_annual_return": expected_return,
                "mu_definition": MU_DEFINITION,
            }
        )

    if not rows:
        return _empty_training_panel(
            status="no_strictly_eligible_candidates",
            ranking_effective_date=effective,
            information_cutoff=cutoff,
            quarter_numbers=quarter_numbers,
        )
    candidates = pd.DataFrame(rows)
    metadata = pair_metadata[list(required_metadata)].copy()
    metadata["pair"] = metadata["pair"].astype(str)
    if metadata["pair"].duplicated().any():
        raise ValueError("pair metadata contains duplicate pair identifiers")
    candidates = candidates.merge(metadata, on="pair", how="left", validate="one_to_one")
    if candidates[["ticker_a", "ticker_b"]].isna().any().any():
        raise ValueError("eligible candidates are missing ticker metadata")
    candidates["ticker_a"] = candidates["ticker_a"].astype(str)
    candidates["ticker_b"] = candidates["ticker_b"].astype(str)
    if candidates["ticker_a"].eq(candidates["ticker_b"]).any():
        raise ValueError("a pair cannot contain the same ticker twice")
    candidates = candidates[_CANDIDATE_COLUMNS].sort_values(
        "pair", kind="mergesort"
    ).reset_index(drop=True)

    day = daily[list(required_daily)].copy()
    day["pair"] = day["pair"].astype(str)
    day["date"] = pd.to_datetime(day["date"], errors="coerce")
    if day["date"].isna().any():
        raise ValueError("daily panel contains invalid dates")
    day = _finite_numeric(day, ["quarter_number", "net_return_2bp"])
    for column in ("price_a", "price_b"):
        day[column] = pd.to_numeric(day[column], errors="coerce")
    day["quarter_number"] = day["quarter_number"].astype(int)
    window_rows = day.loc[
        day["quarter_number"].isin(quarter_numbers)
        & day["date"].ge(first_quarter_rebalance)
        & day["date"].le(cutoff)
    ].copy()
    if expected_sessions is None:
        expected_index = pd.DatetimeIndex(
            window_rows["date"].drop_duplicates().sort_values()
        )
    else:
        expected_index = pd.DatetimeIndex(
            pd.to_datetime(expected_sessions, errors="coerce")
        )
        if expected_index.isna().any() or expected_index.has_duplicates:
            raise ValueError("expected_sessions must contain unique finite dates")
        expected_index = expected_index[
            (expected_index >= first_quarter_rebalance) & (expected_index <= cutoff)
        ].sort_values()
    if (
        len(expected_index) == 0
        or expected_index[0] != first_quarter_rebalance
        or expected_index[-1] != cutoff
    ):
        raise ValueError(
            "expected session window must span the first training rebalance "
            "through information_cutoff"
        )

    candidate_pairs = candidates["pair"].tolist()
    cutoff_prices = window_rows.loc[
        window_rows["pair"].isin(candidate_pairs)
        & window_rows["date"].eq(cutoff),
        ["pair", "price_a", "price_b"],
    ]
    if cutoff_prices["pair"].duplicated().any():
        raise ValueError("daily panel contains duplicate cutoff prices")
    cutoff_prices = cutoff_prices.set_index("pair").reindex(candidate_pairs)
    valid_cutoff_price = (
        np.isfinite(cutoff_prices["price_a"].to_numpy(dtype=float))
        & np.isfinite(cutoff_prices["price_b"].to_numpy(dtype=float))
        & (cutoff_prices["price_a"].to_numpy(dtype=float) > 0.0)
        & (cutoff_prices["price_b"].to_numpy(dtype=float) > 0.0)
    )
    candidate_pairs = [
        pair for pair, valid in zip(candidate_pairs, valid_cutoff_price, strict=True) if valid
    ]
    candidates = candidates.loc[candidates["pair"].isin(candidate_pairs)].copy()
    candidates = candidates.sort_values("pair", kind="mergesort").reset_index(drop=True)
    if not candidate_pairs:
        return _empty_training_panel(
            status="no_candidates_with_valid_cutoff_prices",
            ranking_effective_date=effective,
            information_cutoff=cutoff,
            quarter_numbers=quarter_numbers,
        )
    day = window_rows.loc[window_rows["pair"].isin(candidate_pairs)].copy()
    if day.duplicated(["pair", "date"]).any():
        raise ValueError("daily training panel contains duplicate pair/date keys")
    quarter_map = selected_definitions.set_index("quarter_number")
    day = day.join(
        quarter_map[["rebalance_date", "outer_end_exclusive", "quarter_last_session"]],
        on="quarter_number",
        validate="many_to_one",
    )
    in_own_quarter = (
        day["date"].ge(day["rebalance_date"])
        & day["date"].lt(day["outer_end_exclusive"])
        & day["date"].le(day["quarter_last_session"])
    )
    if not in_own_quarter.all():
        raise ValueError("daily rows are outside their declared completed quarter")
    returns = day.pivot(index="date", columns="pair", values="net_return_2bp")
    returns = returns.reindex(index=expected_index, columns=candidate_pairs).sort_index()
    if returns.empty or returns.isna().any().any():
        raise ValueError("eligible candidates do not share one complete daily window")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("daily training returns must be finite and greater than -100%")
    if returns.index.max() != cutoff:
        raise ValueError("daily common window does not end at information_cutoff")
    if returns.index.min() < first_quarter_rebalance:
        raise ValueError("daily common window starts before the first training quarter")
    if len(returns) < specification.minimum_daily_observations:
        return _empty_training_panel(
            status="insufficient_daily_observations",
            ranking_effective_date=effective,
            information_cutoff=cutoff,
            quarter_numbers=quarter_numbers,
        )

    fitted = LedoitWolf(assume_centered=False).fit(values)
    annual_covariance = np.asarray(fitted.covariance_, dtype=float) * float(
        specification.annualization
    )
    annual_covariance = np.atleast_2d(annual_covariance)
    expected_shape = (len(candidate_pairs), len(candidate_pairs))
    if annual_covariance.shape != expected_shape or not np.isfinite(annual_covariance).all():
        raise ValueError("Ledoit-Wolf covariance has invalid dimensions or values")
    annual_covariance = (annual_covariance + annual_covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(annual_covariance)
    scale = max(
        float(np.trace(annual_covariance)) / len(candidate_pairs),
        specification.covariance_eigenvalue_floor,
    )
    applied_floor = max(
        specification.covariance_eigenvalue_floor,
        scale * specification.covariance_relative_eigenvalue_floor,
    )
    annual_covariance = (
        eigenvectors * np.maximum(eigenvalues, applied_floor)
    ) @ eigenvectors.T
    annual_covariance = (annual_covariance + annual_covariance.T) / 2.0
    covariance = pd.DataFrame(
        annual_covariance,
        index=candidate_pairs,
        columns=candidate_pairs,
    )
    covariance.attrs.update(
        {
            "estimator": "LedoitWolf(assume_centered=False)",
            "annualization": specification.annualization,
            "observations": len(returns),
            "shrinkage": float(fitted.shrinkage_),
            "eigenvalue_floor_annual": applied_floor,
        }
    )
    return TrainingPanel(
        status="ready",
        ranking_effective_date=effective,
        information_cutoff=cutoff,
        quarter_numbers=quarter_numbers,
        training_start=pd.Timestamp(returns.index.min()),
        training_end=pd.Timestamp(returns.index.max()),
        candidate_stats=candidates,
        daily_returns=returns,
        annual_covariance=covariance,
        covariance_shrinkage=float(fitted.shrinkage_),
        covariance_floor_applied=applied_floor,
    )


def _solver_record(
    *,
    phase: str,
    result: Any,
    feasible: bool | None,
    elapsed_seconds: float,
    target_sharpe: float | None = None,
) -> dict[str, Any]:
    names = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    record: dict[str, Any] = {
        "phase": phase,
        "status_code": int(result.status),
        "status": names.get(int(result.status), "unknown"),
        "success": bool(result.success),
        "feasible": feasible,
        "message": str(result.message),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if target_sharpe is not None:
        record["target_sharpe"] = float(target_sharpe)
    for name in ("mip_gap", "mip_node_count", "mip_dual_bound"):
        value = getattr(result, name, None)
        if value is not None and np.isscalar(value) and np.isfinite(value):
            record[name] = float(value)
    return record


def _canonical_candidates(
    candidate_stats: pd.DataFrame,
    annual_covariance: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = {"pair", "ticker_a", "ticker_b", "expected_annual_return"}
    missing = sorted(required - set(candidate_stats.columns))
    if missing:
        raise ValueError(f"candidate stats missing columns: {missing}")
    candidates = candidate_stats[list(required)].copy()
    candidates["pair"] = candidates["pair"].astype(str)
    candidates["ticker_a"] = candidates["ticker_a"].astype(str)
    candidates["ticker_b"] = candidates["ticker_b"].astype(str)
    candidates["expected_annual_return"] = pd.to_numeric(
        candidates["expected_annual_return"], errors="coerce"
    )
    if candidates["pair"].duplicated().any():
        raise ValueError("candidate stats contain duplicate pairs")
    if candidates["ticker_a"].eq(candidates["ticker_b"]).any():
        raise ValueError("a pair cannot contain the same ticker twice")
    if not np.isfinite(candidates["expected_annual_return"].to_numpy(dtype=float)).all():
        raise ValueError("expected annual returns must be finite")
    candidates = candidates.sort_values("pair", kind="mergesort").reset_index(drop=True)
    pairs = candidates["pair"].tolist()
    if not isinstance(annual_covariance, pd.DataFrame):
        raise TypeError("annual_covariance must be a DataFrame labeled by pair")
    if annual_covariance.index.has_duplicates or annual_covariance.columns.has_duplicates:
        raise ValueError("annual covariance labels must be unique")
    covariance = annual_covariance.reindex(index=pairs, columns=pairs)
    if covariance.isna().any().any():
        raise ValueError("annual covariance does not cover every candidate")
    sigma = covariance.to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("annual covariance must be finite and symmetric")
    minimum_eigenvalue = float(np.linalg.eigvalsh((sigma + sigma.T) / 2.0).min())
    if minimum_eigenvalue <= 0.0:
        raise ValueError("annual covariance must be positive definite")
    mu = candidates["expected_annual_return"].to_numpy(dtype=float)
    return candidates, mu, sigma


def _ticker_rows(candidates: pd.DataFrame) -> list[np.ndarray]:
    tickers = sorted(
        set(candidates["ticker_a"].astype(str))
        | set(candidates["ticker_b"].astype(str))
    )
    rows = []
    for ticker in tickers:
        rows.append(
            (
                candidates["ticker_a"].eq(ticker)
                | candidates["ticker_b"].eq(ticker)
            ).to_numpy(dtype=float)
        )
    return rows


def _validate_selected_mask(
    mask: np.ndarray,
    candidates: pd.DataFrame,
    *,
    expected_count: int,
    tolerance: float,
) -> np.ndarray:
    values = np.asarray(mask, dtype=float)
    rounded = values > 0.5
    if np.max(np.abs(values - rounded.astype(float)), initial=0.0) > tolerance:
        raise RuntimeError("HiGHS returned a non-integral pair selection")
    if int(rounded.sum()) != int(expected_count):
        raise RuntimeError("HiGHS returned an unexpected pair count")
    selected = candidates.loc[rounded]
    names = selected[["ticker_a", "ticker_b"]].astype(str).to_numpy().ravel()
    if len(names) != len(set(names)):
        raise RuntimeError("HiGHS returned a book with repeated tickers")
    return rounded


def maximum_feasible_cardinality(
    candidate_stats: pd.DataFrame,
    *,
    top_n: int = 10,
    solution_tolerance: float = 1e-6,
    per_solve_time_limit_seconds: float = 60.0,
    mip_relative_gap: float = 0.0,
) -> CardinalityResult:
    """Return the exact maximum non-overlapping pair count, capped by ``top_n``."""

    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if not math.isfinite(per_solve_time_limit_seconds) or per_solve_time_limit_seconds <= 0:
        raise ValueError("per_solve_time_limit_seconds must be finite and positive")
    if not math.isfinite(mip_relative_gap) or mip_relative_gap < 0:
        raise ValueError("mip_relative_gap must be finite and non-negative")
    required = {"pair", "ticker_a", "ticker_b"}
    missing = sorted(required - set(candidate_stats.columns))
    if missing:
        raise ValueError(f"candidate stats missing columns: {missing}")
    candidates = candidate_stats[list(required)].copy()
    candidates["pair"] = candidates["pair"].astype(str)
    candidates["ticker_a"] = candidates["ticker_a"].astype(str)
    candidates["ticker_b"] = candidates["ticker_b"].astype(str)
    if candidates["pair"].duplicated().any():
        raise ValueError("candidate stats contain duplicate pairs")
    if candidates["ticker_a"].eq(candidates["ticker_b"]).any():
        raise ValueError("a pair cannot contain the same ticker twice")
    candidates = candidates.sort_values("pair", kind="mergesort").reset_index(drop=True)
    count = len(candidates)
    if count == 0:
        return CardinalityResult(
            maximum_feasible_k=0,
            solver_status={
                "phase": "maximum_cardinality",
                "status_code": 0,
                "status": "empty_candidate_set",
                "success": True,
                "feasible": True,
                "message": "no eligible candidates",
                "elapsed_seconds": 0.0,
                "maximum_feasible_k": 0,
            },
        )
    rows = _ticker_rows(candidates)
    rows.append(np.ones(count))
    matrix = np.vstack(rows)
    lower = np.full(len(rows), -np.inf)
    upper = np.asarray([1.0] * (len(rows) - 1) + [float(top_n)])
    started = time.perf_counter()
    result = milp(
        c=-np.ones(count),
        integrality=np.ones(count, dtype=int),
        bounds=Bounds(np.zeros(count), np.ones(count)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={
            "presolve": True,
            "time_limit": float(per_solve_time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    elapsed = time.perf_counter() - started
    if int(result.status) != 0 or result.x is None:
        raise RuntimeError(f"maximum-cardinality MILP failed: {result.message}")
    k = int(round(float(np.sum(result.x))))
    _validate_selected_mask(
        result.x,
        candidates,
        expected_count=k,
        tolerance=solution_tolerance,
    )
    if k < 0 or k > min(top_n, count):
        raise RuntimeError("maximum-cardinality MILP returned an invalid count")
    solver_status = _solver_record(
        phase="maximum_cardinality",
        result=result,
        feasible=True,
        elapsed_seconds=elapsed,
    )
    solver_status["maximum_feasible_k"] = k
    return CardinalityResult(
        maximum_feasible_k=k,
        solver_status=solver_status,
    )


def _linear_selection_constraints(
    candidates: pd.DataFrame,
    *,
    exact_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [np.ones(len(candidates)), *_ticker_rows(candidates)]
    lower = np.asarray([float(exact_k)] + [-np.inf] * (len(rows) - 1))
    upper = np.asarray([float(exact_k)] + [1.0] * (len(rows) - 1))
    return np.vstack(rows), lower, upper


def _solve_max_mu_mask(
    candidates: pd.DataFrame,
    mu: np.ndarray,
    *,
    exact_k: int,
    specification: MaxSharpeSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix, lower, upper = _linear_selection_constraints(candidates, exact_k=exact_k)
    # Sorted pair IDs make HiGHS' deterministic solution independent of input row order.
    started = time.perf_counter()
    result = milp(
        c=-mu,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(matrix, lower, upper),
        options={
            "presolve": True,
            "time_limit": float(specification.per_solve_time_limit_seconds),
            "mip_rel_gap": float(specification.mip_relative_gap),
        },
    )
    elapsed = time.perf_counter() - started
    if int(result.status) != 0 or result.x is None:
        raise RuntimeError(f"maximum-expected-return MILP failed: {result.message}")
    mask = _validate_selected_mask(
        result.x,
        candidates,
        expected_count=exact_k,
        tolerance=specification.solution_tolerance,
    )
    status = _solver_record(
        phase="maximum_expected_return",
        result=result,
        feasible=True,
        elapsed_seconds=elapsed,
    )
    status["exact_k"] = int(exact_k)
    return mask, status


def _book_moments(
    mask: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, float, float]:
    indexes = np.flatnonzero(mask)
    numerator = float(mu[indexes].sum())
    variance = float(sigma[np.ix_(indexes, indexes)].sum())
    if not math.isfinite(variance) or variance <= 0.0:
        raise RuntimeError("selected portfolio variance is not positive")
    return numerator, variance, float(numerator / math.sqrt(variance))


def select_matched_score_book(
    candidate_stats: pd.DataFrame,
    annual_covariance: pd.DataFrame,
    *,
    exact_k: int,
    specification: MaxSharpeSpec = MaxSharpeSpec(),
    cardinality_status: dict[str, Any] | None = None,
) -> SelectionResult:
    """Maximize the sum of the same expected-return numerator at fixed ``k``."""

    specification.validate()
    candidates, mu, sigma = _canonical_candidates(candidate_stats, annual_covariance)
    if exact_k <= 0:
        return _cash_result(
            method="matched_score_max_sum_mu",
            status="cash_no_feasible_pairs",
            maximum_feasible_k=max(0, int(exact_k)),
            statuses=(() if cardinality_status is None else (cardinality_status,)),
        )
    if exact_k > len(candidates):
        raise ValueError("exact_k exceeds the candidate count")
    mask, status = _solve_max_mu_mask(
        candidates,
        mu,
        exact_k=exact_k,
        specification=specification,
    )
    statuses = tuple(
        ([] if cardinality_status is None else [cardinality_status]) + [status]
    )
    numerator, variance, sharpe = _book_moments(mask, mu, sigma)
    if numerator <= specification.positive_numerator_epsilon:
        return _cash_result(
            method="matched_score_max_sum_mu",
            status="cash_no_positive_expected_return_combination",
            maximum_feasible_k=exact_k,
            statuses=statuses,
        )
    selected = tuple(candidates.loc[mask, "pair"].astype(str).tolist())
    return SelectionResult(
        method="matched_score_max_sum_mu",
        status="selected",
        selected_pairs=selected,
        maximum_feasible_k=exact_k,
        selected_count=len(selected),
        expected_annual_return_sum=numerator,
        portfolio_variance_sum=variance,
        estimated_sharpe=sharpe,
        feasible_lower=None,
        infeasible_upper=None,
        solver_statuses=statuses,
    )


def _fixed_target_feasibility(
    candidates: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    exact_k: int,
    target_sharpe: float,
    specification: MaxSharpeSpec,
) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    if not math.isfinite(target_sharpe) or target_sharpe < 0.0:
        raise ValueError("target_sharpe must be finite and non-negative")
    pair_count = len(candidates)
    ticker_sets = [
        frozenset((str(row.ticker_a), str(row.ticker_b)))
        for row in candidates.itertuples(index=False)
    ]
    products = [
        (left, right)
        for left in range(pair_count)
        for right in range(left + 1, pair_count)
        if ticker_sets[left].isdisjoint(ticker_sets[right])
    ]
    variable_count = pair_count + len(products)
    row_indexes: list[int] = []
    column_indexes: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(entries: list[tuple[int, float]], low: float, high: float) -> None:
        row = len(lower)
        for column, value in entries:
            if value != 0.0:
                row_indexes.append(row)
                column_indexes.append(column)
                data.append(float(value))
        lower.append(float(low))
        upper.append(float(high))

    add_row([(index, 1.0) for index in range(pair_count)], exact_k, exact_k)
    for ticker_row in _ticker_rows(candidates):
        add_row(
            [(index, value) for index, value in enumerate(ticker_row) if value],
            -np.inf,
            1.0,
        )
    add_row(
        [(index, float(mu[index])) for index in range(pair_count)],
        specification.positive_numerator_epsilon,
        np.inf,
    )
    incident_products: list[list[int]] = [[] for _ in range(pair_count)]
    for product_index, (left, right) in enumerate(products):
        incident_products[left].append(product_index)
        incident_products[right].append(product_index)
    # With binary x, sum(x)=k and the no-shared-ticker constraints, these RLT
    # equalities force y_ij=x_i*x_j.  If x_i=0 its row forces all incident y
    # to zero.  If x_i=1, exactly k-1 compatible selected counterparts exist,
    # so their bounded non-negative y values must all equal one.  The usual
    # three McCormick inequalities per product are therefore redundant.
    for pair_index, product_indexes in enumerate(incident_products):
        entries = [
            (pair_count + product_index, 1.0)
            for product_index in product_indexes
        ]
        entries.append((pair_index, -float(exact_k - 1)))
        add_row(entries, 0.0, 0.0)

    target_squared = float(target_sharpe * target_sharpe)
    sharpe_entries = [
        (index, float(mu[index] ** 2 - target_squared * sigma[index, index]))
        for index in range(pair_count)
    ]
    sharpe_entries.extend(
        (
            pair_count + product_index,
            float(
                2.0
                * (
                    mu[left] * mu[right]
                    - target_squared * sigma[left, right]
                )
            ),
        )
        for product_index, (left, right) in enumerate(products)
    )
    coefficient_scale = max(
        max((abs(value) for _, value in sharpe_entries), default=0.0),
        np.finfo(float).tiny,
    )
    matrix = coo_matrix(
        (data, (row_indexes, column_indexes)),
        shape=(len(lower), variable_count),
        dtype=float,
    ).tocsr()
    scaled_sharpe_coefficients = np.zeros(variable_count)
    for index, value in sharpe_entries:
        scaled_sharpe_coefficients[index] = value / coefficient_scale
    # Maximizing the fixed-target quadratic margin is an exact feasibility
    # oracle: the target is attainable iff the global optimum is non-negative.
    objective = -scaled_sharpe_coefficients
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:pair_count] = 1
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={
            "presolve": True,
            "time_limit": float(specification.per_solve_time_limit_seconds),
            "mip_rel_gap": float(specification.mip_relative_gap),
        },
    )
    elapsed = time.perf_counter() - started
    if int(result.status) == 2:
        return False, None, _solver_record(
            phase="dinkelbach_residual_oracle",
            result=result,
            feasible=False,
            elapsed_seconds=elapsed,
            target_sharpe=target_sharpe,
        )
    if int(result.status) != 0 or result.x is None:
        raise RuntimeError(
            "Dinkelbach residual MILP failed at target "
            f"{target_sharpe:.12g}: {result.message}"
        )
    reported_objective = getattr(result, "fun", None)
    reported_dual_bound = getattr(result, "mip_dual_bound", None)
    if (
        reported_objective is None
        or reported_dual_bound is None
        or not math.isfinite(float(reported_objective))
        or not math.isfinite(float(reported_dual_bound))
    ):
        raise RuntimeError("Dinkelbach residual MILP lacks objective certificate fields")
    absolute_scaled_primal_dual_gap = abs(
        float(reported_objective) - float(reported_dual_bound)
    )
    if absolute_scaled_primal_dual_gap > specification.solution_tolerance:
        raise RuntimeError(
            "Dinkelbach residual MILP absolute primal-dual gap exceeds tolerance"
        )
    raw_solution = np.asarray(result.x, dtype=float)
    linear_values = np.asarray(matrix @ raw_solution, dtype=float).reshape(-1)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    lower_violation = np.max(
        np.where(np.isfinite(lower_array), lower_array - linear_values, 0.0),
        initial=0.0,
    )
    upper_violation = np.max(
        np.where(np.isfinite(upper_array), linear_values - upper_array, 0.0),
        initial=0.0,
    )
    maximum_linear_violation = float(max(lower_violation, upper_violation, 0.0))
    if maximum_linear_violation > specification.solution_tolerance:
        raise RuntimeError(
            "Dinkelbach residual MILP failed direct linear-constraint reconstruction"
        )
    mask = _validate_selected_mask(
        raw_solution[:pair_count],
        candidates,
        expected_count=exact_k,
        tolerance=specification.solution_tolerance,
    )
    for product_index, (left, right) in enumerate(products):
        claimed = float(raw_solution[pair_count + product_index])
        expected = float(mask[left] and mask[right])
        if abs(claimed - expected) > specification.solution_tolerance:
            raise RuntimeError("lifted product variable failed reconstruction")
    numerator, variance, actual_sharpe = _book_moments(mask, mu, sigma)
    if numerator <= specification.positive_numerator_epsilon:
        raise RuntimeError("Dinkelbach residual MILP violated the positive numerator")
    quadratic_residual = float(
        (numerator * numerator - target_squared * variance) / coefficient_scale
    )
    lifted_residual = float(scaled_sharpe_coefficients @ raw_solution)
    if abs(lifted_residual - quadratic_residual) > specification.solution_tolerance:
        raise RuntimeError(
            "Dinkelbach residual MILP failed direct lifted-objective reconstruction"
        )
    reported_scaled_residual = -float(reported_objective)
    if (
        abs(reported_scaled_residual - lifted_residual)
        > specification.solution_tolerance
    ):
        raise RuntimeError(
            "Dinkelbach residual MILP reported objective failed reconstruction"
        )
    margin_feasible = quadratic_residual >= -specification.solution_tolerance
    ratio_feasible = (
        actual_sharpe + specification.solution_tolerance >= target_sharpe
    )
    feasible = bool(margin_feasible and ratio_feasible)
    status = _solver_record(
        phase="dinkelbach_residual_oracle",
        result=result,
        feasible=feasible,
        elapsed_seconds=elapsed,
        target_sharpe=target_sharpe,
    )
    status["direct_scaled_quadratic_residual"] = quadratic_residual
    status["optimal_scaled_lifted_margin"] = lifted_residual
    status["coefficient_scale"] = coefficient_scale
    status["absolute_scaled_primal_dual_gap"] = absolute_scaled_primal_dual_gap
    status["optimal_unscaled_lifted_margin"] = lifted_residual * coefficient_scale
    status["dual_unscaled_lifted_margin_upper"] = max(
        0.0, -float(reported_dual_bound) * coefficient_scale
    )
    status["lambda_input"] = target_squared
    status["maximum_linear_constraint_violation"] = maximum_linear_violation
    status["compatible_product_variable_count"] = len(products)
    status["lifted_encoding"] = "ticker_disjoint_products_rlt_equalities"
    status["decision"] = (
        "feasible_optimal_margin_nonnegative"
        if feasible
        else "infeasible_optimal_margin_negative"
    )
    return feasible, (mask if feasible else None), status


def _cash_result(
    *,
    method: str,
    status: str,
    maximum_feasible_k: int,
    statuses: tuple[dict[str, Any], ...] = (),
) -> SelectionResult:
    return SelectionResult(
        method=method,
        status=status,
        selected_pairs=(),
        maximum_feasible_k=int(maximum_feasible_k),
        selected_count=0,
        expected_annual_return_sum=None,
        portfolio_variance_sum=None,
        estimated_sharpe=None,
        feasible_lower=None,
        infeasible_upper=None,
        solver_statuses=statuses,
    )


def select_max_sharpe_book(
    candidate_stats: pd.DataFrame,
    annual_covariance: pd.DataFrame,
    *,
    exact_k: int,
    specification: MaxSharpeSpec = MaxSharpeSpec(),
    cardinality_status: dict[str, Any] | None = None,
) -> SelectionResult:
    """Globally maximize positive-numerator equal-weight Sharpe by Dinkelbach."""

    specification.validate()
    candidates, mu, sigma = _canonical_candidates(candidate_stats, annual_covariance)
    starting_statuses = [] if cardinality_status is None else [cardinality_status]
    if exact_k <= 0:
        return _cash_result(
            method="max_sharpe_milp_dinkelbach",
            status="cash_no_feasible_pairs",
            maximum_feasible_k=max(0, int(exact_k)),
            statuses=tuple(starting_statuses),
        )
    if exact_k > len(candidates):
        raise ValueError("exact_k exceeds the candidate count")

    max_mu_mask, max_mu_status = _solve_max_mu_mask(
        candidates,
        mu,
        exact_k=exact_k,
        specification=specification,
    )
    statuses: list[dict[str, Any]] = [*starting_statuses, max_mu_status]
    numerator, variance, starting_sharpe = _book_moments(max_mu_mask, mu, sigma)
    if numerator <= specification.positive_numerator_epsilon:
        return _cash_result(
            method="max_sharpe_milp_dinkelbach",
            status="cash_no_positive_expected_return_combination",
            maximum_feasible_k=exact_k,
            statuses=tuple(statuses),
        )

    best_mask = max_mu_mask.copy()
    lambda_value = float(starting_sharpe * starting_sharpe)
    best_lambda = lambda_value
    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    variance_lower_bound = float(exact_k * minimum_eigenvalue)
    if not math.isfinite(variance_lower_bound) or variance_lower_bound <= 0.0:
        raise RuntimeError("Dinkelbach variance lower bound is not positive")

    final_residual: float | None = None
    final_scaled_residual: float | None = None
    ratio_lower: float | None = None
    ratio_upper: float | None = None
    sharpe_lower: float | None = None
    sharpe_upper: float | None = None
    iterations = 0
    for iteration in range(1, specification.dinkelbach_maximum_iterations + 1):
        feasible, mask, status = _fixed_target_feasibility(
            candidates,
            mu,
            sigma,
            exact_k=exact_k,
            target_sharpe=math.sqrt(max(0.0, lambda_value)),
            specification=specification,
        )
        if not feasible or mask is None:
            raise RuntimeError(
                "Dinkelbach residual oracle rejected its feasible incumbent lambda"
            )
        iteration_numerator, iteration_variance, _ = _book_moments(
            mask, mu, sigma
        )
        direct_residual = float(
            iteration_numerator * iteration_numerator
            - lambda_value * iteration_variance
        )
        recorded_residual = float(status["optimal_unscaled_lifted_margin"])
        coefficient_scale = float(status["coefficient_scale"])
        residual_reconstruction_tolerance = specification.solution_tolerance * max(
            coefficient_scale,
            abs(direct_residual),
            abs(recorded_residual),
            np.finfo(float).tiny,
        )
        if abs(direct_residual - recorded_residual) > residual_reconstruction_tolerance:
            raise RuntimeError("Dinkelbach direct residual failed reconstruction")
        candidate_lambda = float(
            iteration_numerator * iteration_numerator / iteration_variance
        )
        if candidate_lambda + specification.dinkelbach_ratio_tolerance < lambda_value:
            raise RuntimeError("Dinkelbach ratio decreased despite a feasible incumbent")
        if candidate_lambda > best_lambda:
            best_lambda = candidate_lambda
            best_mask = mask.copy()
        numerical_residual_allowance = max(
            residual_reconstruction_tolerance,
            specification.dinkelbach_residual_tolerance * coefficient_scale,
        )
        dual_residual_upper = float(
            status["dual_unscaled_lifted_margin_upper"]
            + numerical_residual_allowance
        )
        certified_ratio_upper = float(
            lambda_value + dual_residual_upper / variance_lower_bound
        )
        certified_ratio_lower = best_lambda
        if (
            certified_ratio_upper + specification.solution_tolerance
            < certified_ratio_lower
        ):
            raise RuntimeError("Dinkelbach dual ratio certificate is inconsistent")
        certified_ratio_upper = max(certified_ratio_upper, certified_ratio_lower)
        certified_sharpe_lower = math.sqrt(max(0.0, certified_ratio_lower))
        certified_sharpe_upper = math.sqrt(max(0.0, certified_ratio_upper))
        status.update(
            {
                "phase": "dinkelbach_residual",
                "dinkelbach_iteration": iteration,
                "lambda_input": lambda_value,
                "lambda_output": candidate_lambda,
                "direct_unscaled_residual": direct_residual,
                "numerical_residual_allowance": numerical_residual_allowance,
                "dual_unscaled_residual_upper_with_allowance": dual_residual_upper,
                "variance_lower_bound": variance_lower_bound,
                "certified_ratio_squared_lower": certified_ratio_lower,
                "certified_ratio_squared_upper": certified_ratio_upper,
                "certified_sharpe_lower": certified_sharpe_lower,
                "certified_sharpe_upper": certified_sharpe_upper,
                "certified_sharpe_width": (
                    certified_sharpe_upper - certified_sharpe_lower
                ),
            }
        )
        statuses.append(status)
        iterations = iteration
        final_residual = direct_residual
        final_scaled_residual = float(status["optimal_scaled_lifted_margin"])
        ratio_lower = certified_ratio_lower
        ratio_upper = certified_ratio_upper
        sharpe_lower = certified_sharpe_lower
        sharpe_upper = certified_sharpe_upper
        residual_converged = (
            abs(final_scaled_residual)
            <= specification.dinkelbach_residual_tolerance
        )
        ratio_converged = (
            abs(best_lambda - lambda_value)
            <= specification.dinkelbach_ratio_tolerance
        )
        certificate_converged = (
            sharpe_upper - sharpe_lower
            <= specification.dinkelbach_sharpe_certificate_tolerance
        )
        if residual_converged and ratio_converged and certificate_converged:
            break
        lambda_value = best_lambda
    else:
        raise RuntimeError("Dinkelbach maximum iterations reached without convergence")

    final_numerator, final_variance, final_sharpe = _book_moments(
        best_mask, mu, sigma
    )
    if None in (
        final_residual,
        final_scaled_residual,
        ratio_lower,
        ratio_upper,
        sharpe_lower,
        sharpe_upper,
    ):
        raise RuntimeError("Dinkelbach terminated without certificate fields")
    if abs(final_sharpe - float(sharpe_lower)) > specification.solution_tolerance:
        raise RuntimeError("Dinkelbach selected Sharpe failed certificate reconstruction")
    selected = tuple(candidates.loc[best_mask, "pair"].astype(str).tolist())
    return SelectionResult(
        method="max_sharpe_milp_dinkelbach",
        status="selected_dinkelbach_certified",
        selected_pairs=selected,
        maximum_feasible_k=exact_k,
        selected_count=len(selected),
        expected_annual_return_sum=final_numerator,
        portfolio_variance_sum=final_variance,
        estimated_sharpe=final_sharpe,
        feasible_lower=None,
        infeasible_upper=None,
        solver_statuses=tuple(statuses),
        dinkelbach_iterations=iterations,
        final_maximum_residual=float(final_residual),
        final_maximum_scaled_residual=float(final_scaled_residual),
        optimal_ratio_squared_lower=float(ratio_lower),
        optimal_ratio_squared_upper=float(ratio_upper),
        sharpe_certificate_lower=float(sharpe_lower),
        sharpe_certificate_upper=float(sharpe_upper),
        sharpe_certificate_width=float(sharpe_upper - sharpe_lower),
        variance_lower_bound=variance_lower_bound,
    )


def select_books_with_cardinality_policy(
    candidate_stats: pd.DataFrame,
    annual_covariance: pd.DataFrame,
    specification: MaxSharpeSpec = MaxSharpeSpec(),
) -> tuple[SelectionResult, SelectionResult]:
    """Choose the largest cardinality with a positive expected-return book.

    The structural matching cardinality is retained in
    ``maximum_feasible_k``.  If its maximum-sum expected return is not
    positive, the function decrements ``k`` until the first positive book is
    found.  Both selectors then use that same chosen cardinality.
    """

    specification.validate()
    cardinality = maximum_feasible_cardinality(
        candidate_stats,
        top_n=specification.top_n,
        solution_tolerance=specification.solution_tolerance,
        per_solve_time_limit_seconds=specification.per_solve_time_limit_seconds,
        mip_relative_gap=specification.mip_relative_gap,
    )
    attempted_statuses: list[dict[str, Any]] = [cardinality.solver_status]
    matched: SelectionResult | None = None
    chosen_k = 0
    for candidate_k in range(cardinality.maximum_feasible_k, 0, -1):
        trial = select_matched_score_book(
            candidate_stats,
            annual_covariance,
            exact_k=candidate_k,
            specification=specification,
        )
        attempted_statuses.extend(trial.solver_statuses)
        if trial.status == "selected":
            chosen_k = candidate_k
            matched = SelectionResult(
                method=trial.method,
                status=trial.status,
                selected_pairs=trial.selected_pairs,
                maximum_feasible_k=cardinality.maximum_feasible_k,
                selected_count=trial.selected_count,
                expected_annual_return_sum=trial.expected_annual_return_sum,
                portfolio_variance_sum=trial.portfolio_variance_sum,
                estimated_sharpe=trial.estimated_sharpe,
                feasible_lower=trial.feasible_lower,
                infeasible_upper=trial.infeasible_upper,
                solver_statuses=tuple(attempted_statuses),
            )
            break
    if matched is None:
        cash = _cash_result(
            method="matched_score_max_sum_mu",
            status="cash_no_positive_expected_return_combination",
            maximum_feasible_k=cardinality.maximum_feasible_k,
            statuses=tuple(attempted_statuses),
        )
        return (
            _cash_result(
                method="max_sharpe_milp_dinkelbach",
                status="cash_no_positive_expected_return_combination",
                maximum_feasible_k=cardinality.maximum_feasible_k,
                statuses=tuple(attempted_statuses),
            ),
            cash,
        )
    raw_maximum_sharpe = select_max_sharpe_book(
        candidate_stats,
        annual_covariance,
        exact_k=chosen_k,
        specification=specification,
    )
    maximum_sharpe = SelectionResult(
        method=raw_maximum_sharpe.method,
        status=raw_maximum_sharpe.status,
        selected_pairs=raw_maximum_sharpe.selected_pairs,
        maximum_feasible_k=cardinality.maximum_feasible_k,
        selected_count=raw_maximum_sharpe.selected_count,
        expected_annual_return_sum=raw_maximum_sharpe.expected_annual_return_sum,
        portfolio_variance_sum=raw_maximum_sharpe.portfolio_variance_sum,
        estimated_sharpe=raw_maximum_sharpe.estimated_sharpe,
        feasible_lower=raw_maximum_sharpe.feasible_lower,
        infeasible_upper=raw_maximum_sharpe.infeasible_upper,
        solver_statuses=tuple(attempted_statuses) + raw_maximum_sharpe.solver_statuses,
        dinkelbach_iterations=raw_maximum_sharpe.dinkelbach_iterations,
        final_maximum_residual=raw_maximum_sharpe.final_maximum_residual,
        final_maximum_scaled_residual=(
            raw_maximum_sharpe.final_maximum_scaled_residual
        ),
        optimal_ratio_squared_lower=raw_maximum_sharpe.optimal_ratio_squared_lower,
        optimal_ratio_squared_upper=raw_maximum_sharpe.optimal_ratio_squared_upper,
        sharpe_certificate_lower=raw_maximum_sharpe.sharpe_certificate_lower,
        sharpe_certificate_upper=raw_maximum_sharpe.sharpe_certificate_upper,
        sharpe_certificate_width=raw_maximum_sharpe.sharpe_certificate_width,
        variance_lower_bound=raw_maximum_sharpe.variance_lower_bound,
    )
    return maximum_sharpe, matched


def optimize_walkforward_books(
    daily: pd.DataFrame,
    quarter_history: pd.DataFrame,
    pair_metadata: pd.DataFrame,
    ranking_effective_date: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    specification: MaxSharpeSpec = MaxSharpeSpec(),
    *,
    expected_sessions: pd.DatetimeIndex | None = None,
) -> WalkForwardSelection:
    """Build strict inputs and select max-Sharpe and matched-score books."""

    panel = build_strict_training_panel(
        daily,
        quarter_history,
        pair_metadata,
        ranking_effective_date,
        information_cutoff,
        specification,
        expected_sessions=expected_sessions,
    )
    if panel.status != "ready":
        return WalkForwardSelection(
            training=panel,
            max_sharpe=_cash_result(
                method="max_sharpe_milp_dinkelbach",
                status=f"cash_{panel.status}",
                maximum_feasible_k=0,
            ),
            matched_score=_cash_result(
                method="matched_score_max_sum_mu",
                status=f"cash_{panel.status}",
                maximum_feasible_k=0,
            ),
        )
    maximum_sharpe, matched = select_books_with_cardinality_policy(
        panel.candidate_stats,
        panel.annual_covariance,
        specification,
    )
    return WalkForwardSelection(
        training=panel,
        max_sharpe=maximum_sharpe,
        matched_score=matched,
    )
