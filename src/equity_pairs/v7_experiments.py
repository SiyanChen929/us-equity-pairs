"""Additive V7 pair-strategy experiments built on the frozen V6 schedule.

Nothing in this module is imported by the V6 runner.  The simulator uses a
conservative close-t decision / close-t+1 fill convention, gross-normalized
two-leg weights, turnover charge, and short-borrow proxy while exposing a small
set of predeclared structural variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import PairModel, compute_zscore, performance_metrics
from .config import StrategyConfig
from .grid_search import ParameterSet, strategy_config_for_parameter_set
from .pair_tuning import PairDefinition, coerce_pair_definition, refit_pair_model


@dataclass(frozen=True)
class V7Rules:
    """One structural experiment; numeric choices are fixed in the protocol."""

    block_entry_at_or_beyond_stop: bool = True
    require_reversal_confirmation: bool = False
    half_life_holding_multiplier: float | None = None
    dynamic_beta: bool = False
    kalman_process_variance_ratio: float = 1e-5
    terminal_liquidation: bool = True

    def __post_init__(self) -> None:
        if self.half_life_holding_multiplier is not None:
            value = float(self.half_life_holding_multiplier)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("half_life_holding_multiplier must be positive")
        value = float(self.kalman_process_variance_ratio)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("kalman_process_variance_ratio must be positive")


@dataclass
class V7SimulationResult:
    model: PairModel
    parameter_set: ParameterSet
    strategy_config: StrategyConfig
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def parameter_from_schedule(row: Mapping[str, Any] | pd.Series) -> ParameterSet:
    """Build the exact frozen V6 parameter from a schedule row."""

    return ParameterSet(
        config_id=str(row.get("selected_config_id", row.get("config_id"))),
        zscore_method=str(row["zscore_method"]),
        zscore_lookback=int(row["zscore_lookback"]),
        zscore_min_periods=int(row["zscore_min_periods"]),
        entry_z=float(row["entry_z"]),
        exit_z=float(row["exit_z"]),
        maximum_holding_days=int(row["maximum_holding_days"]),
        is_baseline=bool(row.get("is_baseline", False)),
    )


def _validated_index(
    values: pd.Index | Sequence[Any],
    prices: pd.DataFrame,
    *,
    label: str,
) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(values).sort_values().unique()
    if not len(result):
        raise ValueError(f"{label} cannot be empty")
    missing = result.difference(prices.index)
    if len(missing):
        raise ValueError(f"{label} contains dates outside the price panel")
    return result


def _dynamic_innovation_path(
    model: PairModel,
    log_dependent: pd.Series,
    log_independent: pd.Series,
    train: pd.DatetimeIndex,
    evaluation: pd.DatetimeIndex,
    *,
    process_variance_ratio: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Causal random-walk Kalman alpha/beta and one-step innovations.

    The full training window initializes the filter.  Evaluation-date spread
    values are one-step-ahead innovations, computed before that close updates
    alpha and beta.  Thus a close-t innovation can only create an order for the
    close of t+1; its earliest P&L interval is t+1 to t+2.
    """

    static_training = (
        log_dependent.loc[train]
        - model.alpha
        - model.beta * log_independent.loc[train]
    )
    residual_variance = float(static_training.var(ddof=1))
    if not math.isfinite(residual_variance) or residual_variance <= 1e-12:
        residual_variance = 1e-8
    theta = np.asarray([model.alpha, model.beta], dtype=float)
    covariance = np.eye(2, dtype=float) * residual_variance * 1e-3
    process = np.eye(2, dtype=float) * residual_variance * process_variance_ratio
    innovations: dict[pd.Timestamp, float] = {}
    alphas: dict[pd.Timestamp, float] = {}
    betas: dict[pd.Timestamp, float] = {}
    for date in evaluation:
        x = float(log_independent.at[date])
        y = float(log_dependent.at[date])
        if not np.isfinite(x) or not np.isfinite(y):
            innovations[date] = math.nan
            alphas[date] = float(theta[0])
            betas[date] = float(theta[1])
            continue
        design = np.asarray([1.0, x], dtype=float)
        covariance_prior = covariance + process
        innovation = float(y - design @ theta)
        innovation_variance = float(
            design @ covariance_prior @ design + residual_variance
        )
        gain = covariance_prior @ design / max(innovation_variance, 1e-12)
        theta = theta + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, design)) @ covariance_prior
        covariance = (covariance + covariance.T) / 2.0
        innovations[date] = innovation
        alphas[date] = float(theta[0])
        betas[date] = float(theta[1])
    spread = pd.concat(
        [
            static_training,
            pd.Series(innovations, dtype=float),
        ]
    )
    spread = spread[~spread.index.duplicated(keep="last")].sort_index()
    alpha = pd.Series(model.alpha, index=spread.index, dtype=float)
    beta = pd.Series(model.beta, index=spread.index, dtype=float)
    for date, value in alphas.items():
        alpha.at[date] = value
    for date, value in betas.items():
        beta.at[date] = value
    return spread, alpha, beta


def _trade_ledger(daily: pd.DataFrame, pair: str, sector: str) -> pd.DataFrame:
    """Extract trades from filled, post-close targets.

    ``target_position`` is the position after the close-date fill.  Entry and
    exit z-scores therefore come from the earlier decision carried in
    ``fill_decision_zscore``, while costs on both fills remain inside the trade
    segment.
    """

    target = daily["target_position"].astype(int)
    prior = target.shift(1, fill_value=0)
    entries = np.flatnonzero((prior.eq(0) & target.ne(0)).to_numpy())
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
        if closed:
            exit_decision_date = daily["fill_decision_date"].iloc[exit_location]
            exit_zscore = float(daily["fill_decision_zscore"].iloc[exit_location])
            exit_reason = str(daily["fill_event"].iloc[exit_location])
            exit_fill_price_dependent = float(
                daily["price_dependent"].iloc[exit_location]
            )
            exit_fill_price_independent = float(
                daily["price_independent"].iloc[exit_location]
            )
        else:
            exit_decision_date = pd.NaT
            exit_zscore = math.nan
            exit_reason = "end_of_test"
            exit_fill_price_dependent = math.nan
            exit_fill_price_independent = math.nan
        rows.append(
            {
                "pair": pair,
                "sector": sector,
                "trade_number": number,
                "direction": "long_spread" if direction > 0 else "short_spread",
                "entry_decision_date": daily["fill_decision_date"].iloc[
                    entry_location
                ],
                "entry_date": daily.index[entry_location],
                "entry_fill_price_dependent": float(
                    daily["price_dependent"].iloc[entry_location]
                ),
                "entry_fill_price_independent": float(
                    daily["price_independent"].iloc[entry_location]
                ),
                "exit_decision_date": exit_decision_date,
                "exit_date": daily.index[exit_location],
                "exit_fill_price_dependent": exit_fill_price_dependent,
                "exit_fill_price_independent": exit_fill_price_independent,
                "trading_days": int(exit_location - entry_location),
                "entry_zscore": float(
                    daily["fill_decision_zscore"].iloc[entry_location]
                ),
                "exit_zscore": exit_zscore,
                "exit_reason": exit_reason,
                "closed": closed,
                "net_return": float((1.0 + segment["net_return"]).prod() - 1.0),
                "gross_return_sum": float(segment["gross_return"].sum()),
                "transaction_cost_sum": float(segment["transaction_cost"].sum()),
                "borrow_cost_sum": float(segment["borrow_cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def simulate_v7_pair(
    pair: PairDefinition | Mapping[str, Any] | pd.Series,
    prices: pd.DataFrame,
    train_index: pd.Index | Sequence[Any],
    evaluation_index: pd.Index | Sequence[Any],
    parameter_set: ParameterSet,
    base_config: StrategyConfig,
    rules: V7Rules,
    *,
    screen_half_life_days: float | None = None,
    transaction_cost_bps: float | None = None,
    entry_start: pd.Timestamp | str | None = None,
    entry_end_exclusive: pd.Timestamp | str | None = None,
) -> V7SimulationResult:
    """Refit on training data and replay one structural V7 variant.

    A close observed on date ``t`` creates ``decision_target_*`` for the next
    evaluation-session close.  ``target_*`` is the filled post-close state on
    the row date, and ``weight_*`` is the state that earned that row's
    close-to-close return.  Consequently, a new entry decided on ``t`` fills on
    ``t+1`` and first earns P&L over ``t+1`` to ``t+2``.

    ``entry_start`` and ``entry_end_exclusive`` bound entry *fills*.  Existing
    positions can continue and exit outside that interval.
    """

    definition = coerce_pair_definition(pair)
    frame = prices.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("prices index must be unique")
    train = _validated_index(train_index, frame, label="train_index")
    evaluation = _validated_index(evaluation_index, frame, label="evaluation_index")
    if train[-1] >= evaluation[0]:
        raise ValueError("training must end before evaluation")
    model = refit_pair_model(definition, frame, train, minimum_observations=100)
    config = strategy_config_for_parameter_set(base_config, parameter_set)
    if transaction_cost_bps is not None:
        config = replace(config, transaction_cost_bps=float(transaction_cost_bps))
    if rules.half_life_holding_multiplier is not None:
        if screen_half_life_days is None or not math.isfinite(
            float(screen_half_life_days)
        ):
            raise ValueError("half-life holding requires a finite screen half-life")
        config = replace(
            config,
            maximum_holding_days=max(
                1,
                int(
                    math.ceil(
                        float(screen_half_life_days)
                        * float(rules.half_life_holding_multiplier)
                    )
                ),
            ),
        )
    combined = train.union(evaluation).sort_values()
    legs = frame.loc[combined, [model.dependent, model.independent]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = legs.where(legs > 0)
    logs = np.log(valid)
    if rules.dynamic_beta:
        spread, alpha_path, beta_path = _dynamic_innovation_path(
            model,
            logs[model.dependent],
            logs[model.independent],
            train,
            evaluation,
            process_variance_ratio=rules.kalman_process_variance_ratio,
        )
    else:
        spread = (
            logs[model.dependent]
            - model.alpha
            - model.beta * logs[model.independent]
        ).rename("spread")
        alpha_path = pd.Series(model.alpha, index=combined, dtype=float)
        beta_path = pd.Series(model.beta, index=combined, dtype=float)
    center, scale, zscore = compute_zscore(spread, config)
    returns = valid.pct_change(fill_method=None)
    start_allowed = pd.Timestamp(entry_start) if entry_start is not None else evaluation[0]
    end_allowed = (
        pd.Timestamp(entry_end_exclusive)
        if entry_end_exclusive is not None
        else evaluation[-1] + pd.Timedelta(days=1)
    )

    # ``current_*`` is the actual post-close state carried into the next row.
    # ``pending_*`` is today's fill, which was decided at the prior close.
    current_position = 0
    held_days = 0
    current_dependent = 0.0
    current_independent = 0.0
    pending_position = 0
    pending_dependent = 0.0
    pending_independent = 0.0
    pending_event = "flat"
    pending_zscore = math.nan
    pending_decision_date = pd.NaT
    confirmation_direction = 0
    previous_z = math.nan
    rows: list[dict[str, Any]] = []
    for location, date in enumerate(evaluation):
        held_position = current_position
        held_dependent = current_dependent
        held_independent = current_independent

        value = float(zscore.at[date]) if date in zscore.index else math.nan
        beta_now = float(beta_path.at[date]) if date in beta_path.index else model.beta

        # Fill the order formed at the preceding evaluation close.  The guard
        # below is deliberately duplicated at execution time so an entry can
        # never complete outside its allowed fill window.
        fill_position = pending_position
        fill_dependent = pending_dependent
        fill_independent = pending_independent
        fill_event = pending_event
        fill_zscore = pending_zscore
        fill_decision_date = pending_decision_date
        entry_fill_window = bool(start_allowed <= date < end_allowed)
        if held_position == 0 and fill_position != 0 and not entry_fill_window:
            fill_position = 0
            fill_dependent = 0.0
            fill_independent = 0.0
            fill_event = "entry_cancelled_outside_window"

        is_terminal = bool(
            rules.terminal_liquidation and location == len(evaluation) - 1
        )
        if is_terminal:
            if held_position:
                if fill_position != 0:
                    # No prior-close signal closed this position.  The study
                    # calendar forces the final fill and supplies no signal z.
                    fill_event = "exit_terminal_scheduled"
                    fill_zscore = math.nan
                # A zero pending position is a natural prior-close exit; keep
                # its event, decision date, and decision z-score unchanged.
            elif fill_position:
                fill_event = "terminal_no_entry"
            else:
                fill_event = "terminal_flat"
            fill_position = 0
            fill_dependent = 0.0
            fill_independent = 0.0

        fill_prices = valid.loc[date, [model.dependent, model.independent]]
        fill_prices_available = bool(np.isfinite(fill_prices.to_numpy()).all())
        if not fill_prices_available and (
            held_position != 0 or fill_position != held_position
        ):
            if held_position == 0 and fill_position != 0:
                context = "entry fill"
            elif held_position != 0 and fill_position == 0:
                context = "exit fill and active interval"
            else:
                context = "active interval"
            raise ValueError(
                f"{definition.pair} {context} on {date.date()} requires finite "
                "close prices for both legs"
            )

        dependent_return = float(returns.at[date, model.dependent])
        independent_return = float(returns.at[date, model.independent])
        returns_available = bool(
            np.isfinite(dependent_return) and np.isfinite(independent_return)
        )
        if held_position and not returns_available:
            raise ValueError(
                f"{definition.pair} active interval ending {date.date()} requires "
                "finite close-to-close returns for both legs"
            )
        if not np.isfinite(dependent_return):
            dependent_return = 0.0
        if not np.isfinite(independent_return):
            independent_return = 0.0
        gross_return = (
            held_dependent * dependent_return
            + held_independent * independent_return
        )
        denominator = 1.0 + gross_return
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            drifted_dependent = (
                held_dependent * (1.0 + dependent_return) / denominator
            )
            drifted_independent = (
                held_independent * (1.0 + independent_return) / denominator
            )
        else:
            drifted_dependent = held_dependent
            drifted_independent = held_independent

        turnover = abs(fill_dependent - drifted_dependent) + abs(
            fill_independent - drifted_independent
        )
        # Drifted and fill weights are normalized by post-return NAV.  Convert
        # their turnover cost back to a beginning-of-interval return drag.
        transaction_cost = (
            turnover
            * config.transaction_cost_bps
            / 10_000.0
            * (1.0 + gross_return)
        )
        # Borrow accrues only on the short notional actually held during the
        # interval ending at this close.  Entry day has no carry; exit day does.
        short_notional = max(-held_dependent, 0.0) + max(-held_independent, 0.0)
        borrow_cost = (
            short_notional * config.annual_short_borrow_bps / 10_000.0 / 252.0
        )
        net_return = gross_return - transaction_cost - borrow_cost

        current_position = fill_position
        current_dependent = fill_dependent
        current_independent = fill_independent
        if current_position == 0:
            held_days = 0
        elif held_position != current_position:
            held_days = 0

        event = "hold" if current_position else "flat"
        decision_position = current_position
        beta_healthy = bool(np.isfinite(beta_now) and beta_now > 0)
        next_fill_date = (
            evaluation[location + 1]
            if location + 1 < len(evaluation)
            else None
        )
        entry_window = bool(
            next_fill_date is not None
            and start_allowed <= next_fill_date < end_allowed
        )
        if is_terminal:
            decision_position = 0
            confirmation_direction = 0
            event = fill_event
        elif not np.isfinite(value) or not beta_healthy:
            if current_position:
                event = (
                    "exit_missing_signal"
                    if not np.isfinite(value)
                    else "exit_dynamic_beta_nonpositive"
                )
            decision_position = 0
            held_days = 0
            confirmation_direction = 0
        elif current_position == 0:
            direction = 1 if value <= -config.entry_z else -1 if value >= config.entry_z else 0
            beyond_stop = abs(value) >= config.stop_z
            if not entry_window:
                confirmation_direction = 0
            elif rules.block_entry_at_or_beyond_stop and beyond_stop:
                event = "blocked_entry_beyond_stop"
                confirmation_direction = 0
            elif rules.require_reversal_confirmation:
                if direction == 0:
                    if confirmation_direction:
                        event = "cancel_pending_entry"
                    confirmation_direction = 0
                elif confirmation_direction == direction:
                    confirmed = (
                        np.isfinite(previous_z)
                        and abs(value) < abs(previous_z)
                        and abs(value) >= config.entry_z
                    )
                    if confirmed:
                        decision_position = direction
                        held_days = 0
                        confirmation_direction = 0
                        event = (
                            "enter_long_spread_confirmed"
                            if direction > 0
                            else "enter_short_spread_confirmed"
                        )
                    else:
                        event = "pending_entry"
                elif direction:
                    confirmation_direction = direction
                    event = "pending_entry"
            elif direction:
                decision_position = direction
                held_days = 0
                event = (
                    "enter_long_spread"
                    if direction > 0
                    else "enter_short_spread"
                )
        else:
            held_days += 1
            if abs(value) <= config.exit_z:
                decision_position = 0
                held_days = 0
                event = "exit_mean_reversion"
            elif abs(value) >= config.stop_z:
                decision_position = 0
                held_days = 0
                event = "exit_stop"
            elif held_days >= config.maximum_holding_days:
                decision_position = 0
                held_days = 0
                event = "exit_max_holding"

        if decision_position:
            decision_dependent = decision_position / (1.0 + abs(beta_now))
            decision_independent = (
                -decision_position * beta_now / (1.0 + abs(beta_now))
            )
        else:
            decision_dependent = 0.0
            decision_independent = 0.0
        rows.append(
            {
                "date": date,
                "price_dependent": valid.at[date, model.dependent],
                "price_independent": valid.at[date, model.independent],
                "spread": spread.at[date] if date in spread.index else math.nan,
                "spread_rolling_mean": center.at[date] if date in center.index else math.nan,
                "spread_rolling_std": scale.at[date] if date in scale.index else math.nan,
                "zscore": value,
                "alpha": float(alpha_path.at[date]),
                "beta": beta_now,
                "position": held_position,
                "target_position": fill_position,
                "decision_target_position": decision_position,
                "decision_event": event,
                "fill_event": fill_event,
                "fill_decision_date": fill_decision_date,
                "fill_decision_zscore": fill_zscore,
                "weight_dependent": held_dependent,
                "weight_independent": held_independent,
                "target_weight_dependent": fill_dependent,
                "target_weight_independent": fill_independent,
                "decision_target_weight_dependent": decision_dependent,
                "decision_target_weight_independent": decision_independent,
                "turnover": turnover,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow_cost,
                "net_return": net_return,
            }
        )
        pending_position = decision_position
        pending_dependent = decision_dependent
        pending_independent = decision_independent
        pending_event = event
        pending_zscore = value
        pending_decision_date = date
        previous_z = value

    daily = pd.DataFrame(rows).set_index("date")
    daily["gross_equity"] = (1.0 + daily["gross_return"]).cumprod()
    daily["net_equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["net_equity"] / daily["net_equity"].cummax() - 1.0
    trades = _trade_ledger(daily, definition.pair, definition.sector)
    net_metrics = performance_metrics(daily["net_return"], daily["net_equity"])
    metrics: dict[str, Any] = {
        "pair": definition.pair,
        "sector": definition.sector,
        "dependent": model.dependent,
        "independent": model.independent,
        "alpha": model.alpha,
        "beta": model.beta,
        "net_total_return": net_metrics["total_return"],
        "net_cagr": net_metrics["cagr"],
        "annual_volatility": net_metrics["annual_volatility"],
        "sharpe": net_metrics["sharpe"],
        "sortino": net_metrics["sortino"],
        "max_drawdown": net_metrics["max_drawdown"],
        "calmar": net_metrics["calmar"],
        "trade_count": int(len(trades)),
        "closed_trade_count": int(trades["closed"].sum()) if len(trades) else 0,
        "exposure_fraction": float(daily["position"].ne(0).mean()),
        "annualized_turnover": float(daily["turnover"].mean() * 252.0),
        "transaction_cost_sum": float(daily["transaction_cost"].sum()),
        "borrow_cost_sum": float(daily["borrow_cost"].sum()),
        "start_flat": bool(daily["position"].iloc[0] == 0),
        "terminal_flat": bool(daily["target_position"].iloc[-1] == 0),
        "terminal_liquidation": bool(rules.terminal_liquidation),
        "execution_convention": "signal_close_t_fill_close_t_plus_1",
        "block_entry_at_or_beyond_stop": rules.block_entry_at_or_beyond_stop,
        "require_reversal_confirmation": rules.require_reversal_confirmation,
        "dynamic_beta": rules.dynamic_beta,
        "maximum_holding_days": config.maximum_holding_days,
        "config_id": parameter_set.config_id,
    }
    return V7SimulationResult(
        model=model,
        parameter_set=parameter_set,
        strategy_config=config,
        daily=daily,
        trades=trades,
        metrics=metrics,
    )


def aggregate_ensemble(
    simulations: Sequence[V7SimulationResult],
    *,
    transaction_cost_bps: float,
    annual_short_borrow_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Equal-position ensemble with costs charged after filled-weight averaging."""

    if not simulations:
        raise ValueError("ensemble requires at least one simulation")
    indexes = [result.daily.index for result in simulations]
    if any(not indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError("ensemble simulations must share the same evaluation index")
    dep = pd.concat(
        [result.daily["target_weight_dependent"] for result in simulations], axis=1
    ).mean(axis=1)
    ind = pd.concat(
        [result.daily["target_weight_independent"] for result in simulations], axis=1
    ).mean(axis=1)
    prices_dep = simulations[0].daily["price_dependent"]
    prices_ind = simulations[0].daily["price_independent"]
    returns_dep = prices_dep.pct_change(fill_method=None).fillna(0.0)
    returns_ind = prices_ind.pct_change(fill_method=None).fillna(0.0)
    held_dep = dep.shift(1, fill_value=0.0)
    held_ind = ind.shift(1, fill_value=0.0)
    gross = held_dep * returns_dep + held_ind * returns_ind
    denominator = (1.0 + gross).replace(0.0, np.nan)
    drift_dep = (
        held_dep * (1.0 + returns_dep) / denominator
    ).fillna(held_dep)
    drift_ind = (
        held_ind * (1.0 + returns_ind) / denominator
    ).fillna(held_ind)
    # dep/ind[t] is the post-fill close-t state held over t to t+1.  Row-t P&L
    # uses dep/ind[t-1], and row-t turnover moves its drifted value into the
    # close-t fill.
    turnover = (dep - drift_dep).abs() + (ind - drift_ind).abs()
    # Turnover is a fraction of post-return NAV; net_return is on beginning NAV.
    cost = (
        turnover
        * float(transaction_cost_bps)
        / 10_000.0
        * (1.0 + gross)
    )
    short = held_dep.clip(upper=0).abs() + held_ind.clip(upper=0).abs()
    borrow = short * float(annual_short_borrow_bps) / 10_000.0 / 252.0
    net = gross - cost - borrow
    daily = pd.DataFrame(
        {
            "target_weight_dependent": dep,
            "target_weight_independent": ind,
            "weight_dependent": held_dep,
            "weight_independent": held_ind,
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": cost,
            "borrow_cost": borrow,
            "net_return": net,
        }
    )
    daily["net_equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["net_equity"] / daily["net_equity"].cummax() - 1.0
    metrics = performance_metrics(daily["net_return"], daily["net_equity"])
    metrics.update(
        {
            "configuration_count": len(simulations),
            "execution_convention": "signal_close_t_fill_close_t_plus_1",
            "transaction_cost_sum": float(cost.sum()),
            "borrow_cost_sum": float(borrow.sum()),
            "annualized_turnover": float(turnover.mean() * 252.0),
        }
    )
    return daily, metrics


__all__ = [
    "V7Rules",
    "V7SimulationResult",
    "aggregate_ensemble",
    "parameter_from_schedule",
    "simulate_v7_pair",
]
