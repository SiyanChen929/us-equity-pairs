"""Leakage-controlled, pair-specific hyperparameter tuning.

This module deliberately does not aggregate pairs into a portfolio.  Every
configuration is scored on one gross-normalized pair sleeve, with the hedge
relation refit using only the training observations of each chronological
fold.  Validation simulations start flat and force a charged liquidation at
the terminal close.

The default protocol contains 96 searched configurations plus one unchanged
baseline.  The primary winner is selected at 2 bps per dollar traded; other
cost assumptions only reprice that frozen winner and cannot affect selection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import PairModel, compute_zscore, performance_metrics
from .config import StrategyConfig
from .grid_search import (
    ParameterSet,
    _FastTuningEvaluator,
    strategy_config_for_parameter_set,
)
from .statistics import fit_log_hedge
from .validation import PurgedFold, build_purged_walk_forward_folds


@dataclass(frozen=True)
class PairDefinition:
    """The stable identity of a pair; alpha and beta are always refit."""

    pair: str
    sector: str
    dependent: str
    independent: str


@dataclass(frozen=True)
class PairTuningSpec:
    """Predeclared pair-level grid, fold geometry, and activity controls."""

    zscore_methods: tuple[str, ...] = ("sma", "ewma")
    zscore_lookbacks: tuple[int, ...] = (10, 20, 40, 60)
    entry_zs: tuple[float, ...] = (1.25, 1.75, 2.25)
    exit_zs: tuple[float, ...] = (0.25, 0.50)
    maximum_holding_days: tuple[int, ...] = (10, 30)
    inner_train_size: int = 378
    inner_validation_size: int = 63
    inner_step_size: int = 63
    purge_size: int = 1
    embargo_size: int = 0
    maximum_inner_folds: int | None = 5
    primary_transaction_cost_bps: float = 2.0
    sensitivity_transaction_cost_bps: tuple[float, ...] = (1.0, 2.0)
    minimum_full_training_trades: int = 10
    minimum_validation_trades: int = 6
    minimum_active_folds: int = 3
    minimum_hedge_observations: int = 100
    annualization: int = 252
    one_standard_error_rule: bool = True

    def __post_init__(self) -> None:
        if not self.zscore_methods or any(
            str(value).lower() not in {"sma", "ewma"}
            for value in self.zscore_methods
        ):
            raise ValueError("zscore_methods must contain only 'sma' and 'ewma'")
        if not self.zscore_lookbacks or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in self.zscore_lookbacks
        ):
            raise ValueError("zscore_lookbacks must contain integers of at least 2")
        if not self.entry_zs or not self.exit_zs:
            raise ValueError("entry_zs and exit_zs cannot be empty")
        thresholds = tuple(self.entry_zs) + tuple(self.exit_zs)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in thresholds):
            raise ValueError("entry_zs and exit_zs must be finite and non-negative")
        if not self.maximum_holding_days or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.maximum_holding_days
        ):
            raise ValueError("maximum_holding_days must contain positive integers")
        for name in (
            "inner_train_size",
            "inner_validation_size",
            "inner_step_size",
            "minimum_hedge_observations",
            "annualization",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.inner_step_size < self.inner_validation_size + self.embargo_size:
            raise ValueError(
                "inner_step_size must be at least inner_validation_size + embargo_size"
            )
        if self.purge_size < 0 or self.embargo_size < 0:
            raise ValueError("purge_size and embargo_size cannot be negative")
        if self.maximum_inner_folds is not None and self.maximum_inner_folds <= 0:
            raise ValueError("maximum_inner_folds must be positive when provided")
        costs = (self.primary_transaction_cost_bps,) + tuple(
            self.sensitivity_transaction_cost_bps
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in costs):
            raise ValueError("transaction costs must be finite and non-negative")
        for name in (
            "minimum_full_training_trades",
            "minimum_validation_trades",
            "minimum_active_folds",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class PairSimulationResult:
    """One frozen-model, frozen-parameter pair simulation."""

    model: PairModel
    parameter_set: ParameterSet
    strategy_config: StrategyConfig
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


@dataclass
class PairTuningResult:
    """Auditable result of tuning one pair without a portfolio objective."""

    pair: PairDefinition
    spec: PairTuningSpec
    selected_parameter: ParameterSet
    selected_config: StrategyConfig
    ranking: pd.DataFrame
    fold_evaluations: pd.DataFrame
    fold_models: pd.DataFrame
    validation_returns: pd.DataFrame
    selected_fold_daily: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    selection_summary: dict[str, Any]


@dataclass
class PairBatchTuningResult:
    """Pair-specific results plus a training-only leaderboard."""

    results: dict[str, PairTuningResult]
    leaderboard: pd.DataFrame
    failures: pd.DataFrame


def _number_token(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _parameter_id(
    method: str,
    lookback: int,
    min_periods: int,
    entry_z: float,
    exit_z: float,
    maximum_holding_days: int,
) -> str:
    return "__".join(
        (
            str(method),
            f"lb{lookback}",
            f"min{min_periods}",
            f"en{_number_token(entry_z)}",
            f"ex{_number_token(exit_z)}",
            f"hold{maximum_holding_days}",
        )
    )


def protocol_parameter_sets(
    spec: PairTuningSpec | None = None,
    *,
    stop_z: float = 4.0,
) -> tuple[ParameterSet, ...]:
    """Return the deterministic searched grid plus the unchanged baseline.

    With the default specification this is exactly 97 configurations: 96 grid
    cells and the SMA-60/2.0/0.5/60 baseline.  ``stop_z`` is fixed rather than
    searched to limit the multiple-testing burden.
    """

    spec = spec or PairTuningSpec()
    if not math.isfinite(float(stop_z)) or float(stop_z) <= 0:
        raise ValueError("stop_z must be positive and finite")
    rows: list[ParameterSet] = []
    seen: set[str] = set()
    for method, lookback, entry_z, exit_z, holding_days in product(
        sorted(set(str(value).lower() for value in spec.zscore_methods)),
        sorted(set(spec.zscore_lookbacks)),
        sorted(set(float(value) for value in spec.entry_zs)),
        sorted(set(float(value) for value in spec.exit_zs)),
        sorted(set(spec.maximum_holding_days)),
    ):
        if not 0 <= exit_z < entry_z < float(stop_z):
            continue
        min_periods = (
            int(lookback)
            if int(lookback) <= 10
            else int(math.ceil(0.80 * int(lookback)))
        )
        identifier = _parameter_id(
            method,
            int(lookback),
            min_periods,
            entry_z,
            exit_z,
            int(holding_days),
        )
        if identifier in seen:
            continue
        seen.add(identifier)
        rows.append(
            ParameterSet(
                config_id=identifier,
                zscore_method=method,
                zscore_lookback=int(lookback),
                zscore_min_periods=min_periods,
                entry_z=entry_z,
                exit_z=exit_z,
                maximum_holding_days=int(holding_days),
                is_baseline=False,
            )
        )

    baseline = ParameterSet(
        config_id=_parameter_id("sma", 60, 40, 2.0, 0.5, 60),
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        maximum_holding_days=60,
        is_baseline=True,
    )
    if baseline.config_id in seen:
        raise ValueError("the searched grid unexpectedly duplicates the protocol baseline")
    rows.append(baseline)
    return tuple(sorted(rows, key=lambda row: row.config_id))


def coerce_pair_definition(
    pair: PairDefinition | Mapping[str, Any] | pd.Series,
) -> PairDefinition:
    """Normalize a screen row without trusting its previously fitted alpha/beta."""

    if isinstance(pair, PairDefinition):
        result = pair
    elif isinstance(pair, (Mapping, pd.Series)):
        required = ("pair", "sector", "dependent", "independent")
        missing = [name for name in required if name not in pair]
        if missing:
            raise ValueError("pair definition is missing: " + ", ".join(missing))
        result = PairDefinition(**{name: str(pair[name]) for name in required})
    else:
        raise TypeError("pair must be a PairDefinition, mapping, or pandas Series")
    if not all(
        str(value).strip()
        for value in (result.pair, result.sector, result.dependent, result.independent)
    ):
        raise ValueError("pair definition fields cannot be blank")
    if result.dependent == result.independent:
        raise ValueError("dependent and independent tickers must differ")
    return result


def _validated_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index must contain unique dates")
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()
    return prices


def _validated_subindex(
    values: pd.Index | Sequence[Any],
    prices: pd.DataFrame,
    *,
    name: str,
) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(values)
    if result.empty:
        raise ValueError(f"{name} cannot be empty")
    if result.has_duplicates:
        raise ValueError(f"{name} must contain unique dates")
    if not result.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted chronologically")
    missing = result.difference(prices.index)
    if len(missing):
        raise ValueError(f"{name} contains dates absent from prices")
    return result


def refit_pair_model(
    pair: PairDefinition | Mapping[str, Any] | pd.Series,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    *,
    minimum_observations: int = 100,
) -> PairModel:
    """Fit log-price OLS using only the explicitly supplied training dates."""

    definition = coerce_pair_definition(pair)
    prices = _validated_prices(prices)
    train = _validated_subindex(train_index, prices, name="train_index")
    missing_legs = sorted(
        {definition.dependent, definition.independent} - set(prices.columns)
    )
    if missing_legs:
        raise KeyError("Missing price legs: " + ", ".join(missing_legs))
    legs = prices.loc[train, [definition.dependent, definition.independent]].apply(
        pd.to_numeric, errors="coerce"
    )
    legs = legs.dropna()
    legs = legs.loc[(legs > 0).all(axis=1)]
    if len(legs) < int(minimum_observations):
        raise ValueError(
            f"{definition.pair} has {len(legs)} aligned training observations; "
            f"requires {minimum_observations}"
        )
    alpha, beta, _ = fit_log_hedge(
        legs[definition.dependent].to_numpy(dtype=float),
        legs[definition.independent].to_numpy(dtype=float),
    )
    if not np.isfinite(alpha) or not np.isfinite(beta) or beta <= 0:
        raise ValueError(
            f"{definition.pair} requires a positive finite refitted hedge ratio"
        )
    return PairModel(
        pair=definition.pair,
        sector=definition.sector,
        dependent=definition.dependent,
        independent=definition.independent,
        alpha=float(alpha),
        beta=float(beta),
    )


def build_pair_inner_folds(
    train_index: pd.Index | Sequence[Any],
    spec: PairTuningSpec | None = None,
) -> list[PurgedFold]:
    """Create inspectable chronological inner folds within one outer window."""

    spec = spec or PairTuningSpec()
    folds = build_purged_walk_forward_folds(
        pd.Index(train_index),
        initial_train_size=spec.inner_train_size,
        test_size=spec.inner_validation_size,
        step_size=spec.inner_step_size,
        purge_size=spec.purge_size,
        embargo_size=spec.embargo_size,
        expanding=True,
        maximum_folds=spec.maximum_inner_folds,
    )
    if not folds:
        raise ValueError("the pair tuning specification produces no inner folds")
    return folds


def _trade_ledger(
    daily: pd.DataFrame,
    model: PairModel,
) -> pd.DataFrame:
    """Build trades from actual close fills, retaining their decision timestamps."""

    target = daily["target_position"].astype(int)
    previous = target.shift(1, fill_value=0)
    entries = np.flatnonzero((previous.eq(0) & target.ne(0)).to_numpy())
    rows: list[dict[str, Any]] = []
    for number, entry_location in enumerate(entries, start=1):
        direction = int(target.iloc[entry_location])
        later = target.iloc[entry_location + 1 :]
        candidates = np.flatnonzero(later.eq(0).to_numpy())
        if len(candidates):
            exit_location = entry_location + 1 + int(candidates[0])
            closed = True
        else:
            exit_location = len(daily) - 1
            closed = False
        segment = daily.iloc[entry_location : exit_location + 1]
        compounded = float((1.0 + segment["net_return"]).prod() - 1.0)
        entry_decision_date = daily["fill_decision_date"].iloc[entry_location]
        exit_decision_date = (
            daily["fill_decision_date"].iloc[exit_location]
            if closed
            else pd.NaT
        )
        rows.append(
            {
                "pair": model.pair,
                "sector": model.sector,
                "trade_number": number,
                "direction": "long_spread" if direction > 0 else "short_spread",
                "entry_decision_date": entry_decision_date,
                "entry_date": daily.index[entry_location],
                "entry_fill_price_dependent": float(
                    daily["price_dependent"].iloc[entry_location]
                ),
                "entry_fill_price_independent": float(
                    daily["price_independent"].iloc[entry_location]
                ),
                "exit_decision_date": exit_decision_date,
                "exit_date": daily.index[exit_location],
                "exit_fill_price_dependent": float(
                    daily["price_dependent"].iloc[exit_location]
                ),
                "exit_fill_price_independent": float(
                    daily["price_independent"].iloc[exit_location]
                ),
                "trading_days": int(exit_location - entry_location)
                if closed
                else int(len(daily) - entry_location - 1),
                "entry_zscore": float(
                    daily["fill_decision_zscore"].iloc[entry_location]
                ),
                "exit_zscore": float(
                    daily["fill_decision_zscore"].iloc[exit_location]
                )
                if np.isfinite(daily["fill_decision_zscore"].iloc[exit_location])
                else math.nan,
                "exit_reason": str(daily["fill_event"].iloc[exit_location])
                if closed
                else "end_of_evaluation",
                "closed": closed,
                "net_return": compounded,
                "gross_return_sum": float(segment["gross_return"].sum()),
                "transaction_cost_sum": float(segment["transaction_cost"].sum()),
                "borrow_cost_sum": float(segment["borrow_cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _simulate_with_model(
    model: PairModel,
    prices: pd.DataFrame,
    history_index: pd.DatetimeIndex,
    evaluation_index: pd.DatetimeIndex,
    parameter_set: ParameterSet,
    config: StrategyConfig,
    *,
    terminal_liquidation: bool,
) -> PairSimulationResult:
    combined = history_index.union(evaluation_index).sort_values()
    legs = prices.loc[combined, [model.dependent, model.independent]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = legs.where(legs > 0)
    spread = (
        np.log(valid[model.dependent])
        - model.alpha
        - model.beta * np.log(valid[model.independent])
    ).rename("spread")
    center, scale, zscore = compute_zscore(spread, config)
    leg_returns = valid.pct_change(fill_method=None)

    base_dependent = 1.0 / (1.0 + abs(model.beta))
    base_independent = -model.beta / (1.0 + abs(model.beta))
    filled_position = 0
    pending_target = 0
    pending_event = "flat"
    pending_decision_date = pd.NaT
    pending_decision_zscore = math.nan
    held_days = 0
    rows: list[dict[str, Any]] = []
    for location, date in enumerate(evaluation_index):
        held_position = filled_position
        held_dependent = held_position * base_dependent
        held_independent = held_position * base_independent
        dependent_return = float(leg_returns.at[date, model.dependent])
        independent_return = float(leg_returns.at[date, model.independent])
        price_dependent = float(valid.at[date, model.dependent])
        price_independent = float(valid.at[date, model.independent])
        price_available = bool(
            np.isfinite(price_dependent) and np.isfinite(price_independent)
        )
        return_available = bool(
            np.isfinite(dependent_return) and np.isfinite(independent_return)
        )
        if held_position and not return_available:
            raise RuntimeError(
                f"missing close-to-close return while {model.pair} is active on "
                f"{pd.Timestamp(date).date()}"
            )
        if not return_available:
            dependent_return = 0.0
            independent_return = 0.0
        gross_return = (
            held_dependent * dependent_return
            + held_independent * independent_return
        )
        denominator = 1.0 + gross_return
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            drifted_dependent = held_dependent * (1.0 + dependent_return) / denominator
            drifted_independent = held_independent * (1.0 + independent_return) / denominator
        else:
            drifted_dependent = held_dependent
            drifted_independent = held_independent

        is_terminal = terminal_liquidation and location == len(evaluation_index) - 1
        requested_fill = pending_target
        fill_event = pending_event
        fill_decision_date = pending_decision_date
        fill_decision_zscore = pending_decision_zscore
        if is_terminal:
            if held_position:
                if requested_fill != 0:
                    # No prior-close signal closed this position.  The study
                    # calendar forces the last-close fill, with no signal z.
                    fill_event = "exit_terminal_scheduled"
                    fill_decision_zscore = math.nan
                # If requested_fill is already zero, retain the natural exit's
                # event, prior decision date, and decision z-score.
            elif pending_target:
                fill_event = "terminal_no_entry"
            else:
                fill_event = "terminal_flat"
            requested_fill = 0
        if requested_fill != held_position and not price_available:
            raise RuntimeError(
                f"missing close price for {model.pair} fill on "
                f"{pd.Timestamp(date).date()}"
            )
        filled_position = int(requested_fill)
        if filled_position == 0 or filled_position != held_position:
            held_days = 0

        target_dependent = filled_position * base_dependent
        target_independent = filled_position * base_independent
        turnover = abs(target_dependent - drifted_dependent) + abs(
            target_independent - drifted_independent
        )
        transaction_cost = (
            turnover
            * config.transaction_cost_bps
            / 10_000.0
            * (1.0 + gross_return)
        )
        short_notional = max(-held_dependent, 0.0) + max(-held_independent, 0.0)
        borrow_cost = (
            short_notional * config.annual_short_borrow_bps / 10_000.0 / 252.0
        )
        net_return = gross_return - transaction_cost - borrow_cost

        value = float(zscore.at[date]) if date in zscore.index else math.nan
        decision_target = filled_position
        event = "hold" if filled_position else "flat"
        if is_terminal:
            decision_target = 0
            event = fill_event
        elif not np.isfinite(value):
            if filled_position:
                event = "exit_missing_signal"
            decision_target = 0
        elif filled_position == 0:
            if value <= -config.entry_z:
                decision_target = 1
                event = "enter_long_spread"
            elif value >= config.entry_z:
                decision_target = -1
                event = "enter_short_spread"
        else:
            # Count close sessions from the entry fill.  With the required
            # next-close exit, a limit of N produces at most N return intervals.
            held_days += 1
            if abs(value) <= config.exit_z:
                decision_target = 0
                event = "exit_mean_reversion"
            elif abs(value) >= config.stop_z:
                decision_target = 0
                event = "exit_stop"
            elif held_days >= config.maximum_holding_days:
                decision_target = 0
                event = "exit_max_holding"

        pending_target = int(decision_target)
        pending_event = event
        pending_decision_date = date
        pending_decision_zscore = value
        rows.append(
            {
                "date": date,
                "price_dependent": price_dependent,
                "price_independent": price_independent,
                "spread": spread.at[date],
                "spread_rolling_mean": center.at[date],
                "spread_rolling_std": scale.at[date],
                "zscore": value,
                "position": held_position,
                "target_position": filled_position,
                "decision_target_position": decision_target,
                "decision_event": event,
                "fill_event": fill_event,
                "fill_decision_date": fill_decision_date,
                "fill_decision_zscore": fill_decision_zscore,
                "weight_dependent": held_dependent,
                "weight_independent": held_independent,
                "target_weight_dependent": target_dependent,
                "target_weight_independent": target_independent,
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow_cost,
                "net_return": net_return,
            }
        )

    daily = pd.DataFrame(rows).set_index("date")
    daily.index.name = "date"
    daily["gross_equity"] = (1.0 + daily["gross_return"]).cumprod()
    daily["net_equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["net_equity"] / daily["net_equity"].cummax() - 1.0
    trades = _trade_ledger(daily, model)
    gross_metrics = performance_metrics(daily["gross_return"], daily["gross_equity"])
    net_metrics = performance_metrics(daily["net_return"], daily["net_equity"])
    metrics: dict[str, Any] = {
        **asdict(model),
        "config_id": parameter_set.config_id,
        "evaluation_start": evaluation_index[0],
        "evaluation_end": evaluation_index[-1],
        "observations": int(len(evaluation_index)),
        "gross_total_return": gross_metrics["total_return"],
        "gross_cagr": gross_metrics["cagr"],
        "net_total_return": net_metrics["total_return"],
        "net_cagr": net_metrics["cagr"],
        "annual_volatility": net_metrics["annual_volatility"],
        "sharpe": net_metrics["sharpe"],
        "max_drawdown": net_metrics["max_drawdown"],
        "trade_count": int(len(trades)),
        "closed_trade_count": int(trades["closed"].sum()) if len(trades) else 0,
        "exposure_fraction": float(daily["position"].ne(0).mean()),
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "transaction_cost_bps": float(config.transaction_cost_bps),
        "transaction_cost_sum": float(daily["transaction_cost"].sum()),
        "borrow_cost_sum": float(daily["borrow_cost"].sum()),
        "start_flat": bool(
            daily["position"].iloc[0] == 0
            and daily["target_position"].iloc[0] == 0
        ),
        "terminal_flat": bool(daily["target_position"].iloc[-1] == 0),
        "terminal_liquidation": bool(terminal_liquidation),
        "execution_convention": "signal_close_t_fill_close_t_plus_1",
    }
    return PairSimulationResult(
        model=model,
        parameter_set=parameter_set,
        strategy_config=config,
        daily=daily,
        trades=trades,
        metrics=metrics,
    )


def evaluate_frozen_pair_parameter(
    pair: PairDefinition | Mapping[str, Any] | pd.Series,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    evaluation_index: pd.Index | Sequence[Any],
    parameter_set: ParameterSet,
    base_config: StrategyConfig,
    *,
    transaction_cost_bps: float | None = None,
    minimum_hedge_observations: int = 100,
    terminal_liquidation: bool = True,
) -> PairSimulationResult:
    """Refit on ``train_index`` and evaluate one frozen parameter out of sample."""

    definition = coerce_pair_definition(pair)
    prices = _validated_prices(prices)
    train = _validated_subindex(train_index, prices, name="train_index")
    evaluation = _validated_subindex(
        evaluation_index, prices, name="evaluation_index"
    )
    if train[-1] >= evaluation[0]:
        raise ValueError("train_index must end before evaluation_index begins")
    model = refit_pair_model(
        definition,
        prices,
        train,
        minimum_observations=minimum_hedge_observations,
    )
    config = strategy_config_for_parameter_set(base_config, parameter_set)
    if transaction_cost_bps is not None:
        if not math.isfinite(float(transaction_cost_bps)) or transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        config = replace(config, transaction_cost_bps=float(transaction_cost_bps))
    return _simulate_with_model(
        model,
        prices,
        train,
        evaluation,
        parameter_set,
        config,
        terminal_liquidation=terminal_liquidation,
    )


def _annual_log_return(returns: pd.Series, annualization: int) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values) or np.any(values <= -1.0):
        return math.nan
    return float(np.log1p(values).mean() * annualization)


def _training_activity_count(
    model: PairModel,
    prices: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    parameter_set: ParameterSet,
    config: StrategyConfig,
) -> int:
    # This in-window replay is used only as an activity gate.  Its return is
    # never used in the objective or reported as out-of-sample performance.
    simulation = _simulate_with_model(
        model,
        prices,
        train_index,
        train_index,
        parameter_set,
        config,
        terminal_liquidation=True,
    )
    return int(simulation.metrics["trade_count"])


def _fast_evaluator_for_pair(
    model: PairModel,
    prices: pd.DataFrame,
    history_index: pd.DatetimeIndex,
    evaluation_index: pd.DatetimeIndex,
    base_config: StrategyConfig,
) -> _FastTuningEvaluator:
    """Build the vectorized evaluator on the canonical signal-history slice."""

    combined = history_index.union(evaluation_index).sort_values()
    evaluator = _FastTuningEvaluator(
        pd.DataFrame([asdict(model)]),
        prices.loc[combined],
        pd.Timestamp(evaluation_index[0]),
        pd.Timestamp(evaluation_index[-1]) + pd.Timedelta(nanoseconds=1),
        base_config,
    )
    legs = prices.loc[combined, [model.dependent, model.independent]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = legs.where(legs > 0)
    leg_returns = valid.pct_change(fill_method=None)
    evaluator.complete_closes[:, 0] = np.isfinite(
        valid.to_numpy(dtype=float)
    ).all(axis=1)
    evaluator.complete_returns[:, 0] = np.isfinite(
        leg_returns.to_numpy(dtype=float)
    ).all(axis=1)
    fast_dates = evaluator.dates[evaluator.test_mask]
    if not fast_dates.equals(evaluation_index):
        raise RuntimeError("fast pair evaluator does not match the evaluation index")
    return evaluator


def _fast_pair_parameter_evaluation(
    evaluator: _FastTuningEvaluator,
    parameter_set: ParameterSet,
    *,
    annualization: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Reproduce canonical ranking fields without materializing a daily ledger."""

    if len(evaluator.pairs) != 1:
        raise ValueError("fast pair evaluation requires exactly one pair")
    _, _, filled_position = evaluator._simulate(parameter_set)
    mask = evaluator.test_mask
    dates = evaluator.dates[mask]
    position = filled_position[mask, 0]
    previous_position = np.zeros_like(position)
    previous_position[1:] = position[:-1]
    trade_count = int(((previous_position == 0) & (position != 0)).sum())

    return_position = np.zeros_like(filled_position)
    return_position[1:] = filled_position[:-1]
    gross_return = (
        return_position * evaluator.base_dependent * evaluator.dependent_returns
        + return_position * evaluator.base_independent * evaluator.independent_returns
    )
    denominator = 1.0 + gross_return
    valid_denominator = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        post_return_dependent = np.where(
            valid_denominator,
            return_position
            * evaluator.base_dependent
            * (1.0 + evaluator.dependent_returns)
            / denominator,
            return_position * evaluator.base_dependent,
        )
        post_return_independent = np.where(
            valid_denominator,
            return_position
            * evaluator.base_independent
            * (1.0 + evaluator.independent_returns)
            / denominator,
            return_position * evaluator.base_independent,
        )
    filled_dependent = filled_position * evaluator.base_dependent
    filled_independent = filled_position * evaluator.base_independent
    turnover = np.abs(filled_dependent - post_return_dependent) + np.abs(
        filled_independent - post_return_independent
    )
    config = strategy_config_for_parameter_set(evaluator.base_config, parameter_set)
    transaction_cost = (
        turnover
        * float(config.transaction_cost_bps)
        / 10_000.0
        * (1.0 + gross_return)
    )
    short_notional = np.maximum(
        -return_position * evaluator.base_dependent, 0.0
    ) + np.maximum(-return_position * evaluator.base_independent, 0.0)
    borrow_cost = (
        short_notional
        * float(config.annual_short_borrow_bps)
        / 10_000.0
        / 252.0
    )
    net_return = gross_return - transaction_cost - borrow_cost
    returns = pd.Series(
        net_return[mask, 0],
        index=dates,
        name=parameter_set.config_id,
    )
    returns.index.name = "date"
    turnover_series = pd.Series(turnover[mask, 0], index=dates)
    transaction_cost_series = pd.Series(transaction_cost[mask, 0], index=dates)
    metrics = performance_metrics(returns)
    annual_log_return = _annual_log_return(returns, annualization)
    return returns, {
        "annual_log_return": annual_log_return,
        "net_cagr": math.expm1(annual_log_return)
        if np.isfinite(annual_log_return)
        else math.nan,
        "net_total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "annualized_turnover": float(turnover_series.mean() * 252.0),
        "trade_count": trade_count,
        "transaction_cost_sum": float(transaction_cost_series.sum()),
        "start_flat": bool(previous_position[0] == 0 and position[0] == 0),
        "terminal_flat": bool(position[-1] == 0),
    }


def select_pair_parameter(
    ranking: pd.DataFrame,
    *,
    one_standard_error_rule: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select deterministically from pair-level validation results only."""

    required = {
        "config_id",
        "eligible",
        "pooled_net_cagr",
        "mean_fold_annual_log_return",
        "std_fold_annual_log_return",
        "standard_error",
        "annualized_turnover",
        "max_drawdown",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError("ranking is missing columns: " + ", ".join(missing))
    eligible = ranking.loc[ranking["eligible"].astype(bool)].copy()
    eligible = eligible.loc[
        np.isfinite(pd.to_numeric(eligible["pooled_net_cagr"], errors="coerce"))
    ]
    if eligible.empty:
        raise ValueError("no pair parameter satisfies the activity constraints")
    raw = eligible.sort_values(
        ["pooled_net_cagr", "max_drawdown", "annualized_turnover", "config_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    cutoff = float(raw["mean_fold_annual_log_return"])
    if one_standard_error_rule and np.isfinite(float(raw["standard_error"])):
        cutoff -= float(raw["standard_error"])
    candidates = eligible.loc[
        pd.to_numeric(eligible["mean_fold_annual_log_return"], errors="coerce")
        >= cutoff
    ].copy()
    candidates = candidates.sort_values(
        [
            "std_fold_annual_log_return",
            "annualized_turnover",
            "max_drawdown",
            "pooled_net_cagr",
            "config_id",
        ],
        ascending=[True, True, False, False, True],
        kind="mergesort",
        na_position="last",
    )
    selected = candidates.iloc[0]
    summary = {
        "objective": "pair_net_cagr",
        "portfolio_objective_used": False,
        "raw_best_config_id": str(raw["config_id"]),
        "selected_config_id": str(selected["config_id"]),
        "one_standard_error_rule": bool(one_standard_error_rule),
        "one_standard_error_cutoff_log_return": cutoff,
        "one_standard_error_candidate_count": int(len(candidates)),
    }
    return selected, summary


def tune_pair_parameters(
    pair: PairDefinition | Mapping[str, Any] | pd.Series,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    base_config: StrategyConfig,
    spec: PairTuningSpec | None = None,
    *,
    ranking_engine: str = "fast",
) -> PairTuningResult:
    """Tune one pair on chronological inner folds using net CAGR only.

    ``fast`` vectorizes only the exhaustive ranking stage.  The selected
    configuration and every cost-sensitivity path are always replayed through
    the canonical stateful simulator.
    """

    definition = coerce_pair_definition(pair)
    prices = _validated_prices(prices)
    train = _validated_subindex(train_index, prices, name="train_index")
    spec = spec or PairTuningSpec()
    if ranking_engine not in {"fast", "canonical"}:
        raise ValueError("ranking_engine must be 'fast' or 'canonical'")
    if float(base_config.stop_z) <= max(spec.entry_zs):
        raise ValueError("base_config.stop_z must exceed every searched entry threshold")
    parameter_sets = protocol_parameter_sets(spec, stop_z=base_config.stop_z)
    parameters = {row.config_id: row for row in parameter_sets}
    folds = build_pair_inner_folds(train, spec)
    primary_base = replace(
        base_config,
        transaction_cost_bps=float(spec.primary_transaction_cost_bps),
    )

    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    return_pieces: dict[str, list[pd.Series]] = {
        identifier: [] for identifier in parameters
    }
    fold_models: dict[int, PairModel] = {}
    for fold in folds:
        fold_train = pd.DatetimeIndex(fold.train_index)
        fold_test = pd.DatetimeIndex(fold.test_index)
        signal_history = fold_train.append(
            pd.DatetimeIndex(fold.purge_index)
        ).sort_values()
        model = refit_pair_model(
            definition,
            prices,
            fold_train,
            minimum_observations=spec.minimum_hedge_observations,
        )
        fold_models[fold.fold_id] = model
        model_rows.append(
            {
                "pair": definition.pair,
                "inner_fold": fold.fold_id,
                "model_train_start": fold_train[0],
                "model_train_end": fold_train[-1],
                "validation_start": fold_test[0],
                "validation_end": fold_test[-1],
                "train_observations": len(fold_train),
                "purged_observations": len(fold.purge_index),
                "alpha": model.alpha,
                "beta": model.beta,
            }
        )
        fast_evaluator = (
            _fast_evaluator_for_pair(
                model,
                prices,
                signal_history,
                fold_test,
                primary_base,
            )
            if ranking_engine == "fast"
            else None
        )
        for parameter_set in parameter_sets:
            if fast_evaluator is not None:
                returns, fold_metrics = _fast_pair_parameter_evaluation(
                    fast_evaluator,
                    parameter_set,
                    annualization=spec.annualization,
                )
            else:
                config = strategy_config_for_parameter_set(
                    primary_base, parameter_set
                )
                simulation = _simulate_with_model(
                    model,
                    prices,
                    signal_history,
                    fold_test,
                    parameter_set,
                    config,
                    terminal_liquidation=True,
                )
                returns = simulation.daily["net_return"].rename(
                    parameter_set.config_id
                )
                annual_log = _annual_log_return(returns, spec.annualization)
                fold_metrics = {
                    "annual_log_return": annual_log,
                    "net_cagr": math.expm1(annual_log)
                    if np.isfinite(annual_log)
                    else math.nan,
                    "net_total_return": simulation.metrics["net_total_return"],
                    "max_drawdown": simulation.metrics["max_drawdown"],
                    "annualized_turnover": simulation.metrics[
                        "annualized_turnover"
                    ],
                    "trade_count": simulation.metrics["trade_count"],
                    "transaction_cost_sum": simulation.metrics[
                        "transaction_cost_sum"
                    ],
                    "start_flat": simulation.metrics["start_flat"],
                    "terminal_flat": simulation.metrics["terminal_flat"],
                }
            return_pieces[parameter_set.config_id].append(returns)
            fold_rows.append(
                {
                    "pair": definition.pair,
                    "inner_fold": fold.fold_id,
                    "config_id": parameter_set.config_id,
                    "validation_start": fold_test[0],
                    "validation_end": fold_test[-1],
                    "observations": len(fold_test),
                    "model_train_end": fold_train[-1],
                    "alpha": model.alpha,
                    "beta": model.beta,
                    **fold_metrics,
                }
            )

    fold_evaluations = pd.DataFrame(fold_rows)
    validation_returns = pd.DataFrame(
        {
            identifier: pd.concat(pieces).sort_index()
            for identifier, pieces in return_pieces.items()
        }
    ).sort_index()
    if validation_returns.index.has_duplicates:
        raise RuntimeError("inner validation windows unexpectedly overlap")
    validation_returns.index.name = "date"

    full_model = refit_pair_model(
        definition,
        prices,
        train,
        minimum_observations=spec.minimum_hedge_observations,
    )
    full_training_evaluator = (
        _fast_evaluator_for_pair(
            full_model,
            prices,
            train,
            train,
            primary_base,
        )
        if ranking_engine == "fast"
        else None
    )
    ranking_rows: list[dict[str, Any]] = []
    for parameter_set in parameter_sets:
        identifier = parameter_set.config_id
        values = validation_returns[identifier]
        metrics = performance_metrics(values)
        per_fold = fold_evaluations.loc[
            fold_evaluations["config_id"].eq(identifier)
        ].sort_values("inner_fold")
        logs = pd.to_numeric(per_fold["annual_log_return"], errors="coerce").dropna()
        fold_standard_deviation = float(logs.std(ddof=1)) if len(logs) > 1 else 0.0
        standard_error = (
            float(fold_standard_deviation / math.sqrt(len(logs)))
            if len(logs)
            else math.nan
        )
        if full_training_evaluator is not None:
            _, training_metrics = _fast_pair_parameter_evaluation(
                full_training_evaluator,
                parameter_set,
                annualization=spec.annualization,
            )
            training_trades = int(training_metrics["trade_count"])
        else:
            config = strategy_config_for_parameter_set(primary_base, parameter_set)
            training_trades = _training_activity_count(
                full_model,
                prices,
                train,
                parameter_set,
                config,
            )
        validation_trades = int(per_fold["trade_count"].sum())
        active_folds = int(per_fold["trade_count"].gt(0).sum())
        eligible = bool(
            training_trades >= spec.minimum_full_training_trades
            and validation_trades >= spec.minimum_validation_trades
            and active_folds >= spec.minimum_active_folds
        )
        ranking_rows.append(
            {
                **asdict(parameter_set),
                "pair": definition.pair,
                "objective": "pair_net_cagr",
                "portfolio_objective_used": False,
                "primary_transaction_cost_bps": spec.primary_transaction_cost_bps,
                "pooled_net_cagr": metrics["cagr"],
                "pooled_total_return": metrics["total_return"],
                "mean_fold_annual_log_return": float(logs.mean())
                if len(logs)
                else math.nan,
                "std_fold_annual_log_return": fold_standard_deviation,
                "standard_error": standard_error,
                "positive_fold_count": int(per_fold["net_total_return"].gt(0).sum()),
                "active_fold_count": active_folds,
                "validation_trade_count": validation_trades,
                "full_training_trade_count": training_trades,
                "annualized_turnover": float(per_fold["annualized_turnover"].mean()),
                "max_drawdown": metrics["max_drawdown"],
                "eligible": eligible,
            }
        )
    ranking = pd.DataFrame(ranking_rows)
    selected_row, selection_summary = select_pair_parameter(
        ranking,
        one_standard_error_rule=spec.one_standard_error_rule,
    )
    selected_id = str(selected_row["config_id"])
    selected_parameter = parameters[selected_id]
    ranking["within_one_standard_error"] = (
        ranking["eligible"]
        & ranking["mean_fold_annual_log_return"].ge(
            selection_summary["one_standard_error_cutoff_log_return"]
        )
    )
    ranking["selected"] = ranking["config_id"].eq(selected_id)
    ranking = ranking.sort_values(
        ["selected", "eligible", "pooled_net_cagr", "config_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "display_rank", np.arange(1, len(ranking) + 1))

    selected_daily_parts: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []
    costs = sorted(
        set(
            [float(spec.primary_transaction_cost_bps)]
            + [float(value) for value in spec.sensitivity_transaction_cost_bps]
        )
    )
    for cost in costs:
        cost_returns: list[pd.Series] = []
        cost_trade_count = 0
        cost_turnover: list[float] = []
        cost_transaction_sum = 0.0
        for fold in folds:
            fold_train = pd.DatetimeIndex(fold.train_index)
            fold_test = pd.DatetimeIndex(fold.test_index)
            signal_history = fold_train.append(
                pd.DatetimeIndex(fold.purge_index)
            ).sort_values()
            config = strategy_config_for_parameter_set(
                replace(primary_base, transaction_cost_bps=cost), selected_parameter
            )
            simulation = _simulate_with_model(
                fold_models[fold.fold_id],
                prices,
                signal_history,
                fold_test,
                selected_parameter,
                config,
                terminal_liquidation=True,
            )
            cost_returns.append(simulation.daily["net_return"])
            cost_trade_count += int(simulation.metrics["trade_count"])
            cost_turnover.append(float(simulation.metrics["annualized_turnover"]))
            cost_transaction_sum += float(simulation.metrics["transaction_cost_sum"])
            if math.isclose(
                cost,
                float(spec.primary_transaction_cost_bps),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                daily = simulation.daily.reset_index()
                daily.insert(0, "inner_fold", fold.fold_id)
                daily.insert(0, "config_id", selected_id)
                daily.insert(0, "pair", definition.pair)
                selected_daily_parts.append(daily)
        combined_returns = pd.concat(cost_returns).sort_index()
        metrics = performance_metrics(combined_returns)
        sensitivity_rows.append(
            {
                "pair": definition.pair,
                "config_id": selected_id,
                "selection_cost_bps": spec.primary_transaction_cost_bps,
                "transaction_cost_bps": cost,
                "frozen_winner": True,
                **metrics,
                "trade_count": cost_trade_count,
                "annualized_turnover": float(np.mean(cost_turnover)),
                "transaction_cost_sum": cost_transaction_sum,
            }
        )

    selected_config = strategy_config_for_parameter_set(primary_base, selected_parameter)
    selection_summary.update(
        {
            "pair": definition.pair,
            "candidate_count": len(parameter_sets),
            "inner_fold_count": len(folds),
            "selection_transaction_cost_bps": spec.primary_transaction_cost_bps,
            "sensitivity_reoptimized": False,
            "selected_pooled_net_cagr": float(selected_row["pooled_net_cagr"]),
            "selected_validation_trade_count": int(
                selected_row["validation_trade_count"]
            ),
            "selected_full_training_trade_count": int(
                selected_row["full_training_trade_count"]
            ),
        }
    )
    return PairTuningResult(
        pair=definition,
        spec=spec,
        selected_parameter=selected_parameter,
        selected_config=selected_config,
        ranking=ranking,
        fold_evaluations=fold_evaluations,
        fold_models=pd.DataFrame(model_rows),
        validation_returns=validation_returns,
        selected_fold_daily=pd.concat(selected_daily_parts, ignore_index=True),
        cost_sensitivity=pd.DataFrame(sensitivity_rows),
        selection_summary=selection_summary,
    )


def evaluate_tuned_pair(
    tuning: PairTuningResult,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    evaluation_index: pd.Index | Sequence[Any],
    base_config: StrategyConfig,
    *,
    transaction_cost_bps: float | None = None,
) -> PairSimulationResult:
    """Apply a tuning winner to a later window without reopening selection."""

    cost = (
        tuning.spec.primary_transaction_cost_bps
        if transaction_cost_bps is None
        else transaction_cost_bps
    )
    return evaluate_frozen_pair_parameter(
        tuning.pair,
        prices,
        train_index,
        evaluation_index,
        tuning.selected_parameter,
        base_config,
        transaction_cost_bps=float(cost),
        minimum_hedge_observations=tuning.spec.minimum_hedge_observations,
        terminal_liquidation=True,
    )


def tune_eligible_pairs(
    selected_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    base_config: StrategyConfig,
    spec: PairTuningSpec | None = None,
    *,
    eligibility_column: str = "tradable_eligible",
    continue_on_error: bool = True,
) -> PairBatchTuningResult:
    """Tune screen-eligible pairs independently and build no portfolio."""

    required = {"pair", "sector", "dependent", "independent"}
    missing = sorted(required - set(selected_pairs.columns))
    if missing:
        raise ValueError("selected_pairs is missing columns: " + ", ".join(missing))
    working = selected_pairs.copy()
    if eligibility_column in working.columns:
        working = working.loc[working[eligibility_column].fillna(False).astype(bool)]
    if working["pair"].astype(str).duplicated().any():
        raise ValueError("selected_pairs must contain unique pair identifiers")
    results: dict[str, PairTuningResult] = {}
    failures: list[dict[str, str]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    for record in working.to_dict(orient="records"):
        definition = coerce_pair_definition(record)
        try:
            result = tune_pair_parameters(
                definition,
                prices,
                train_index,
                base_config,
                spec,
            )
        except Exception as error:
            if not continue_on_error:
                raise
            failures.append(
                {
                    "pair": definition.pair,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        results[definition.pair] = result
        selected = result.ranking.loc[result.ranking["selected"]].iloc[0]
        leaderboard_rows.append(
            {
                "pair": definition.pair,
                "sector": definition.sector,
                "selected_config_id": result.selected_parameter.config_id,
                "pair_cv_net_cagr": selected["pooled_net_cagr"],
                "pair_cv_max_drawdown": selected["max_drawdown"],
                "validation_trade_count": selected["validation_trade_count"],
                "active_fold_count": selected["active_fold_count"],
                "portfolio_objective_used": False,
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["pair_cv_net_cagr", "pair_cv_max_drawdown", "pair"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        leaderboard.insert(0, "pair_rank", np.arange(1, len(leaderboard) + 1))
    return PairBatchTuningResult(
        results=results,
        leaderboard=leaderboard,
        failures=pd.DataFrame(failures, columns=["pair", "error_type", "error"]),
    )


__all__ = [
    "PairBatchTuningResult",
    "PairDefinition",
    "PairSimulationResult",
    "PairTuningResult",
    "PairTuningSpec",
    "build_pair_inner_folds",
    "coerce_pair_definition",
    "evaluate_frozen_pair_parameter",
    "evaluate_tuned_pair",
    "protocol_parameter_sets",
    "refit_pair_model",
    "select_pair_parameter",
    "tune_eligible_pairs",
    "tune_pair_parameters",
]
