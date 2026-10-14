"""Causal cross-pair ranking and portfolio simulation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossPairSelectionSpec:
    """Frozen rules for one cross-pair selection decision."""

    lookback_calendar_years: int = 3
    minimum_history_quarters: int = 8
    minimum_active_quarters: int = 3
    minimum_trade_entries: int = 10
    top_n: int = 10

    def validate(self) -> None:
        values = {
            "lookback_calendar_years": self.lookback_calendar_years,
            "minimum_history_quarters": self.minimum_history_quarters,
            "minimum_active_quarters": self.minimum_active_quarters,
            "minimum_trade_entries": self.minimum_trade_entries,
            "top_n": self.top_n,
        }
        if any(int(value) != value or int(value) <= 0 for value in values.values()):
            raise ValueError("cross-pair selection parameters must be positive integers")


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite")
    return result


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def first_known_candidate_history(
    candidate_metrics: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Remove every candidate outcome dated before that pair became known."""

    required_candidates = {"pair", "rebalance_date", "strategy_id"}
    required_master = {"pair", "first_successful_v6_tuned_date"}
    missing_candidates = sorted(required_candidates - set(candidate_metrics.columns))
    missing_master = sorted(required_master - set(master.columns))
    if missing_candidates:
        raise ValueError(f"candidate metrics missing columns: {missing_candidates}")
    if missing_master:
        raise ValueError(f"pair master missing columns: {missing_master}")
    candidates = candidate_metrics.copy()
    candidates["pair"] = candidates["pair"].astype(str)
    candidates["rebalance_date"] = pd.to_datetime(
        candidates["rebalance_date"], errors="coerce"
    )
    if candidates["rebalance_date"].isna().any():
        raise ValueError("candidate metrics contain invalid rebalance dates")
    if candidates.duplicated(["pair", "rebalance_date", "strategy_id"]).any():
        raise ValueError("candidate metrics contain duplicate keys")
    known = master[list(required_master)].copy()
    known["pair"] = known["pair"].astype(str)
    known["eligible_since_date"] = pd.to_datetime(
        known.pop("first_successful_v6_tuned_date"), errors="coerce"
    )
    if known["pair"].duplicated().any() or known["eligible_since_date"].isna().any():
        raise ValueError("pair master requires one finite first-known date per pair")
    result = candidates.merge(known, on="pair", how="left", validate="many_to_one")
    if result["eligible_since_date"].isna().any():
        raise ValueError("candidate metrics contain an unknown pair")
    return result.loc[
        result["rebalance_date"].ge(result["eligible_since_date"])
    ].sort_values(
        ["pair", "rebalance_date", "strategy_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_first_known_quarter_history(
    daily: pd.DataFrame,
    schedule: pd.DataFrame,
    quarterly: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate first-known-clean shadow paths into completed-quarter records."""

    required_daily = {
        "pair",
        "date",
        "quarter_number",
        "net_return_2bp",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "turnover",
        "trade_entry",
    }
    required_schedule = {"quarter_number", "rebalance_date", "outer_end_exclusive"}
    required_quarterly = {"pair", "quarter_number", "evaluation_success", "available"}
    required_master = {"pair", "first_successful_v6_tuned_date"}
    missing_daily = sorted(required_daily - set(daily.columns))
    missing_schedule = sorted(required_schedule - set(schedule.columns))
    missing_quarterly = sorted(required_quarterly - set(quarterly.columns))
    missing_master = sorted(required_master - set(master.columns))
    if missing_daily:
        raise ValueError(f"safe-reset daily panel missing columns: {missing_daily}")
    if missing_schedule:
        raise ValueError(f"rebalance schedule missing columns: {missing_schedule}")
    if missing_quarterly:
        raise ValueError(f"quarterly panel missing columns: {missing_quarterly}")
    if missing_master:
        raise ValueError(f"pair master missing columns: {missing_master}")
    work = daily.copy()
    work["pair"] = work["pair"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].isna().any():
        raise ValueError("safe-reset daily panel contains invalid dates")
    if work.duplicated(["pair", "date"]).any():
        raise ValueError("safe-reset daily panel contains duplicate pair/date rows")
    work = _finite_numeric(
        work,
        [
            "quarter_number",
            "net_return_2bp",
            "weight_a",
            "weight_b",
            "target_weight_a",
            "target_weight_b",
            "turnover",
        ],
    )
    if work["net_return_2bp"].le(-1.0).any():
        raise ValueError("safe-reset daily returns must be greater than -100%")
    work["trade_entry"] = _as_bool(work["trade_entry"])
    work["path_active"] = (
        work[["weight_a", "weight_b", "target_weight_a", "target_weight_b"]]
        .abs()
        .sum(axis=1)
        .gt(1e-12)
    )
    sched = schedule[list(required_schedule)].copy()
    sched["quarter_number"] = pd.to_numeric(
        sched["quarter_number"], errors="coerce"
    )
    for column in ("rebalance_date", "outer_end_exclusive"):
        sched[column] = pd.to_datetime(sched[column], errors="coerce")
    if sched.isna().any().any():
        raise ValueError("rebalance schedule contains invalid values")
    if sched["quarter_number"].duplicated().any():
        raise ValueError("rebalance schedule contains duplicate quarter numbers")

    rows: list[dict[str, Any]] = []
    for (pair, quarter_number), group in work.groupby(
        ["pair", "quarter_number"], sort=True
    ):
        ordered = group.sort_values("date", kind="mergesort")
        returns = ordered["net_return_2bp"].to_numpy(dtype=float)
        equity = np.cumprod(1.0 + returns)
        running = np.maximum(1.0, np.maximum.accumulate(equity))
        rows.append(
            {
                "pair": pair,
                "quarter_number": int(quarter_number),
                "quarter_return_2bp": float(equity[-1] - 1.0),
                "active_quarter": bool(
                    ordered["path_active"].any() or ordered["trade_entry"].any()
                ),
                "trade_entries": int(ordered["trade_entry"].sum()),
                "annualized_turnover": float(
                    ordered["turnover"].mean() * 252.0
                ),
                "max_drawdown": float(np.min(equity / running - 1.0)),
                "quarter_sessions": int(len(ordered)),
                "quarter_last_session": pd.Timestamp(ordered["date"].max()),
            }
        )
    result = pd.DataFrame(rows).merge(
        sched,
        on="quarter_number",
        how="left",
        validate="many_to_one",
    )
    if result[["rebalance_date", "outer_end_exclusive"]].isna().any().any():
        raise ValueError("daily quarter number is absent from the schedule")
    quarter_flags = quarterly[list(required_quarterly)].copy()
    quarter_flags["pair"] = quarter_flags["pair"].astype(str)
    quarter_flags["quarter_number"] = pd.to_numeric(
        quarter_flags["quarter_number"], errors="coerce"
    )
    if quarter_flags.isna().any().any():
        raise ValueError("quarterly availability panel contains invalid values")
    if quarter_flags.duplicated(["pair", "quarter_number"]).any():
        raise ValueError("quarterly availability panel contains duplicate keys")
    quarter_flags["history_available"] = (
        _as_bool(quarter_flags["evaluation_success"])
        & _as_bool(quarter_flags["available"])
    )
    known = master[list(required_master)].copy()
    known["pair"] = known["pair"].astype(str)
    known["eligible_since_date"] = pd.to_datetime(
        known.pop("first_successful_v6_tuned_date"), errors="coerce"
    )
    if known["pair"].duplicated().any() or known["eligible_since_date"].isna().any():
        raise ValueError("pair master requires one finite first-known date per pair")
    result = result.merge(
        quarter_flags[["pair", "quarter_number", "history_available"]],
        on=["pair", "quarter_number"],
        how="left",
        validate="one_to_one",
    ).merge(known, on="pair", how="left", validate="many_to_one")
    if result[["history_available", "eligible_since_date"]].isna().any().any():
        raise ValueError("quarter history could not be aligned to source eligibility")
    return result.sort_values(
        ["pair", "quarter_number"], kind="mergesort"
    ).reset_index(drop=True)


def rank_pairs_asof(
    quarter_history: pd.DataFrame,
    ranking_effective_date: pd.Timestamp,
    specification: CrossPairSelectionSpec,
    *,
    information_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rank pairs for a rebalance using data known by its information cutoff."""

    specification.validate()
    required = {
        "pair",
        "quarter_number",
        "rebalance_date",
        "outer_end_exclusive",
        "quarter_last_session",
        "quarter_return_2bp",
        "active_quarter",
        "trade_entries",
        "annualized_turnover",
        "max_drawdown",
        "history_available",
        "eligible_since_date",
    }
    missing = sorted(required - set(quarter_history.columns))
    if missing:
        raise ValueError(f"quarter history missing columns: {missing}")
    current = pd.Timestamp(ranking_effective_date)
    if pd.isna(current):
        raise ValueError("ranking_effective_date is invalid")
    cutoff = current if information_cutoff is None else pd.Timestamp(information_cutoff)
    if pd.isna(cutoff) or cutoff > current:
        raise ValueError("information_cutoff must be finite and no later than ranking_effective_date")
    history = quarter_history.copy()
    history["pair"] = history["pair"].astype(str)
    for column in ("rebalance_date", "outer_end_exclusive", "quarter_last_session"):
        history[column] = pd.to_datetime(history[column], errors="coerce")
        if history[column].isna().any():
            raise ValueError(f"quarter history contains invalid {column}")
    history["eligible_since_date"] = pd.to_datetime(
        history["eligible_since_date"], errors="coerce"
    )
    if history.duplicated(["pair", "quarter_number"]).any():
        raise ValueError("quarter history contains duplicate pair/quarter rows")
    start = current - pd.DateOffset(years=specification.lookback_calendar_years)
    history = history.loc[
        history["rebalance_date"].ge(start)
        & history["outer_end_exclusive"].le(current)
        & history["quarter_last_session"].le(cutoff)
        & history["eligible_since_date"].notna()
        & history["eligible_since_date"].le(cutoff)
        & history["rebalance_date"].ge(history["eligible_since_date"])
        & _as_bool(history["history_available"])
    ].copy()
    history = _finite_numeric(
        history,
        [
            "quarter_return_2bp",
            "trade_entries",
            "annualized_turnover",
            "max_drawdown",
        ],
    )
    if history["quarter_return_2bp"].le(-1.0).any():
        raise ValueError("quarter returns must be greater than -100%")
    rows: list[dict[str, Any]] = []
    for pair, group in history.groupby("pair", sort=True):
        work = group.sort_values("outer_end_exclusive", kind="mergesort")
        returns = work["quarter_return_2bp"].to_numpy(dtype=float)
        logs = np.log1p(returns)
        observations = int(len(work))
        active_quarters = int(_as_bool(work["active_quarter"]).sum())
        trade_entries = int(work["trade_entries"].sum())
        log_std = float(np.std(logs, ddof=1)) if observations > 1 else math.nan
        rows.append(
            {
                "ranking_effective_date": current,
                "pair": pair,
                "lookback_start_inclusive": start,
                "training_end_exclusive_max": work[
                    "outer_end_exclusive"
                ].max(),
                "training_last_session_max": work["quarter_last_session"].max(),
                "eligible_since_date": work["eligible_since_date"].iloc[0],
                "history_quarters": observations,
                "active_quarters": active_quarters,
                "trade_entries": trade_entries,
                "mean_quarterly_log_return": float(np.mean(logs)),
                "quarterly_log_return_std": log_std,
                "quarterly_log_return_standard_error": (
                    log_std / math.sqrt(observations)
                    if observations > 1
                    else math.nan
                ),
                "annualized_one_se_lower_bound": (
                    4.0 * (float(np.mean(logs)) - log_std / math.sqrt(observations))
                    if observations > 1
                    else math.nan
                ),
                "mean_annualized_turnover": float(
                    work["annualized_turnover"].mean()
                ),
                "worst_quarter_max_drawdown": float(work["max_drawdown"].min()),
                "eligible": bool(
                    observations >= specification.minimum_history_quarters
                    and active_quarters >= specification.minimum_active_quarters
                    and trade_entries >= specification.minimum_trade_entries
                ),
            }
        )
    columns = [
        "ranking_effective_date",
        "pair",
        "lookback_start_inclusive",
        "training_end_exclusive_max",
        "training_last_session_max",
        "eligible_since_date",
        "history_quarters",
        "active_quarters",
        "trade_entries",
        "mean_quarterly_log_return",
        "quarterly_log_return_std",
        "quarterly_log_return_standard_error",
        "annualized_one_se_lower_bound",
        "mean_annualized_turnover",
        "worst_quarter_max_drawdown",
        "eligible",
        "rank",
        "selected",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows)
    result = result.sort_values(
        [
            "eligible",
            "annualized_one_se_lower_bound",
            "mean_quarterly_log_return",
            "quarterly_log_return_std",
            "mean_annualized_turnover",
            "worst_quarter_max_drawdown",
            "pair",
        ],
        ascending=[False, False, False, True, True, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    eligible_index = result.index[result["eligible"]]
    result.loc[eligible_index, "rank"] = np.arange(1, len(eligible_index) + 1)
    result["selected"] = result["eligible"] & result["rank"].le(
        specification.top_n
    ).fillna(False)
    return result[columns]


def select_nonoverlapping_pairs(
    ranking: pd.DataFrame,
    pair_metadata: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    """Greedily select ranked pairs while allowing each ticker only once."""

    required_ranking = {"pair", "eligible", "rank"}
    required_metadata = {"pair", "ticker_a", "ticker_b"}
    missing_ranking = sorted(required_ranking - set(ranking.columns))
    missing_metadata = sorted(required_metadata - set(pair_metadata.columns))
    if missing_ranking:
        raise ValueError(f"ranking missing columns: {missing_ranking}")
    if missing_metadata:
        raise ValueError(f"pair metadata missing columns: {missing_metadata}")
    if int(top_n) != top_n or int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer")
    metadata = pair_metadata[list(required_metadata)].copy()
    metadata["pair"] = metadata["pair"].astype(str)
    if metadata["pair"].duplicated().any():
        raise ValueError("pair metadata contains duplicate pairs")
    work = ranking.copy()
    work["pair"] = work["pair"].astype(str)
    work = work.merge(metadata, on="pair", how="left", validate="one_to_one")
    if work[["ticker_a", "ticker_b"]].isna().any().any():
        raise ValueError("ranking contains pairs without ticker metadata")
    work["selected"] = False
    work["selection_rank"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    used: set[str] = set()
    count = 0
    for index in work.sort_values("rank", kind="mergesort", na_position="last").index:
        row = work.loc[index]
        if not bool(row["eligible"]):
            continue
        names = {str(row["ticker_a"]), str(row["ticker_b"])}
        if names & used:
            continue
        count += 1
        work.at[index, "selected"] = True
        work.at[index, "selection_rank"] = count
        used.update(names)
        if count == int(top_n):
            break
    return work.sort_values("rank", kind="mergesort", na_position="last").reset_index(
        drop=True
    )


def simulate_security_level_portfolio(
    signals: pd.DataFrame,
    allocation_events: pd.DataFrame,
    *,
    portfolio_id: str,
    transaction_cost_bps: float,
    annual_short_borrow_bps: float,
    terminal_liquidation: bool = True,
    return_detail: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Replay a dynamic pair book from post-close target leg weights.

    Each event is applied after the return on ``fill_date``.  Incoming pairs
    stay flat until their shadow path reaches an observable post-close flat state, so
    the book never inherits an unowned position.  Trading cost is charged once
    on pair-level absolute changes in the two leg notionals.
    """

    required_signals = {
        "pair",
        "date",
        "ticker_a",
        "ticker_b",
        "price_a",
        "price_b",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "decision_event",
        "fill_event",
    }
    missing = sorted(required_signals - set(signals.columns))
    if missing:
        raise ValueError(f"signals missing columns: {missing}")
    event_required = {"decision_date", "fill_date", "target_weights"}
    missing_events = sorted(event_required - set(allocation_events.columns))
    if missing_events:
        raise ValueError(f"allocation events missing columns: {missing_events}")
    transaction_rate = float(transaction_cost_bps) / 10_000.0
    borrow_rate = float(annual_short_borrow_bps) / 10_000.0 / 252.0
    if not np.isfinite(transaction_rate) or transaction_rate < 0:
        raise ValueError("transaction cost must be finite and nonnegative")
    if not np.isfinite(borrow_rate) or borrow_rate < 0:
        raise ValueError("borrow cost must be finite and nonnegative")

    work = signals[list(required_signals)].copy()
    work["pair"] = work["pair"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].isna().any() or work.duplicated(["pair", "date"]).any():
        raise ValueError("signals require unique finite pair/date keys")
    weight_columns = [
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
    ]
    work = _finite_numeric(work, weight_columns)
    for column in ("price_a", "price_b"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid_prices = (
        np.isfinite(work["price_a"].to_numpy(dtype=float))
        & np.isfinite(work["price_b"].to_numpy(dtype=float))
        & work["price_a"].gt(0).to_numpy()
        & work["price_b"].gt(0).to_numpy()
    )
    weighted = work[weight_columns].abs().sum(axis=1).gt(1e-12).to_numpy()
    if bool((~valid_prices & weighted).any()):
        raise ValueError("invalid signal price appears with an active weight")
    pairs = sorted(work["pair"].unique())
    dates = pd.DatetimeIndex(sorted(work["date"].unique()))
    ticker_counts = work.groupby("pair")[["ticker_a", "ticker_b"]].nunique()
    if bool(ticker_counts.ne(1).any().any()):
        raise ValueError("ticker mapping changes within a pair signal path")
    if bool(
        work[["target_weight_a", "target_weight_b"]]
        .abs()
        .sum(axis=1)
        .gt(1.0 + 1e-10)
        .any()
    ):
        raise ValueError("pair target gross weight exceeds one")
    counts = work.groupby("pair")["date"].nunique()
    if len(work) != len(pairs) * len(dates) or not counts.eq(len(dates)).all():
        raise ValueError("signals must form a balanced pair/date panel")
    ordered = work.sort_values(["pair", "date"], kind="mergesort")
    for held, target in (("weight_a", "target_weight_a"), ("weight_b", "target_weight_b")):
        expected = ordered.groupby("pair", sort=False)[target].shift(1)
        comparable = expected.notna()
        if not np.allclose(
            ordered.loc[comparable, held].to_numpy(dtype=float),
            expected.loc[comparable].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("shadow held weights do not match prior post-close targets")
    indexed = work.set_index(["date", "pair"]).sort_index()

    events = allocation_events.copy()
    events["decision_date"] = pd.to_datetime(events["decision_date"], errors="coerce")
    events["fill_date"] = pd.to_datetime(events["fill_date"], errors="coerce")
    if events[["decision_date", "fill_date"]].isna().any().any():
        raise ValueError("allocation events contain invalid dates")
    if events["fill_date"].duplicated().any() or not events["fill_date"].isin(dates).all():
        raise ValueError("allocation fill dates must be unique signal sessions")
    if not events["decision_date"].lt(events["fill_date"]).all():
        raise ValueError("allocation decision must precede its fill")
    event_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for raw in events.sort_values("fill_date", kind="mergesort").to_dict("records"):
        targets = {str(pair): float(value) for pair, value in dict(raw["target_weights"]).items()}
        if set(targets) - set(pairs):
            raise ValueError("allocation event contains an unknown pair")
        values = np.asarray(list(targets.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() > 1.0 + 1e-12:
            raise ValueError("allocation target weights must be finite, nonnegative, and sum to at most one")
        event_by_date[pd.Timestamp(raw["fill_date"])] = targets

    pair_capital = {pair: 0.0 for pair in pairs}
    leg_a = {pair: 0.0 for pair in pairs}
    leg_b = {pair: 0.0 for pair in pairs}
    synced = {pair: False for pair in pairs}
    selected: set[str] = set()
    cash = 1.0
    previous_nav = 1.0
    previous_prices: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    tolerance = 1e-12

    for location, date in enumerate(dates):
        day = indexed.loc[date].reindex(pairs)
        if day[weight_columns].isna().any().any():
            raise ValueError("signals are not aligned after pivoting")
        nav_open = previous_nav
        capital_open = dict(pair_capital)
        opening_a = dict(leg_a)
        opening_b = dict(leg_b)
        gross_pnl = 0.0
        borrow_cost = 0.0
        gross_pnl_by_pair: dict[str, float] = {}
        borrow_cost_by_pair: dict[str, float] = {}
        return_a_by_pair: dict[str, float] = {}
        return_b_by_pair: dict[str, float] = {}
        drift_a: dict[str, float] = {}
        drift_b: dict[str, float] = {}
        for pair in pairs:
            pa = float(day.at[pair, "price_a"])
            pb = float(day.at[pair, "price_b"])
            price_valid = bool(
                np.isfinite(pa) and np.isfinite(pb) and pa > 0.0 and pb > 0.0
            )
            if not price_valid and (
                abs(leg_a[pair]) + abs(leg_b[pair]) > tolerance
            ):
                raise ValueError("held leg encounters an invalid close price")
            if price_valid and pair in previous_prices:
                old_pa, old_pb = previous_prices[pair]
                ra = pa / old_pa - 1.0
                rb = pb / old_pb - 1.0
            else:
                ra = rb = 0.0
            pnl = leg_a[pair] * ra + leg_b[pair] * rb
            carry = (max(-leg_a[pair], 0.0) + max(-leg_b[pair], 0.0)) * borrow_rate
            pair_capital[pair] += pnl - carry
            gross_pnl += pnl
            borrow_cost += carry
            gross_pnl_by_pair[pair] = pnl
            borrow_cost_by_pair[pair] = carry
            return_a_by_pair[pair] = ra
            return_b_by_pair[pair] = rb
            drift_a[pair] = leg_a[pair] * (1.0 + ra)
            drift_b[pair] = leg_b[pair] * (1.0 + rb)
            if price_valid:
                previous_prices[pair] = (pa, pb)
        insolvent = [pair for pair, value in pair_capital.items() if value <= -tolerance]
        if insolvent:
            raise RuntimeError(
                "pair sleeve became insolvent: " + ", ".join(insolvent[:5])
            )
        nav_pretrade = cash + sum(pair_capital.values())
        if not np.isfinite(nav_pretrade) or nav_pretrade <= 0:
            raise RuntimeError("portfolio NAV became nonpositive before trading")

        allocation_fill = date in event_by_date
        if allocation_fill:
            targets = event_by_date[date]
            new_selected = {pair for pair, value in targets.items() if value > tolerance}
            for pair in pairs:
                if pair not in new_selected or pair not in selected:
                    synced[pair] = False
            selected = new_selected
            desired_capital = {
                pair: nav_pretrade * targets.get(pair, 0.0) for pair in pairs
            }
            cash = nav_pretrade - sum(desired_capital.values())
        else:
            desired_capital = dict(pair_capital)

        for pair in selected:
            if synced[pair]:
                continue
            row = day.loc[pair]
            # target_weight_* is the executable post-close state.  A same-day
            # exit can therefore synchronize immediately, and a new entry
            # decision made at this close is still filled only next close.
            neutral = (
                abs(float(row["target_weight_a"]))
                + abs(float(row["target_weight_b"]))
                <= tolerance
            )
            if neutral:
                synced[pair] = True

        desired_a: dict[str, float] = {}
        desired_b: dict[str, float] = {}
        transaction_cost_by_pair: dict[str, float] = {}
        turnover = 0.0
        for pair in pairs:
            enabled = pair in selected and synced[pair]
            target_a = float(day.at[pair, "target_weight_a"]) if enabled else 0.0
            target_b = float(day.at[pair, "target_weight_b"]) if enabled else 0.0
            desired_a[pair] = desired_capital[pair] * target_a
            desired_b[pair] = desired_capital[pair] * target_b
            pair_turnover = abs(desired_a[pair] - drift_a[pair]) + abs(
                desired_b[pair] - drift_b[pair]
            )
            turnover += pair_turnover
            transaction_cost_by_pair[pair] = pair_turnover * transaction_rate

        terminal = bool(terminal_liquidation and location == len(dates) - 1)
        if terminal:
            turnover = sum(abs(drift_a[pair]) + abs(drift_b[pair]) for pair in pairs)
            transaction_cost_by_pair = {
                pair: (abs(drift_a[pair]) + abs(drift_b[pair])) * transaction_rate
                for pair in pairs
            }
            desired_a = {pair: 0.0 for pair in pairs}
            desired_b = {pair: 0.0 for pair in pairs}

        transaction_cost = float(sum(transaction_cost_by_pair.values()))
        pair_capital = dict(desired_capital)
        cost_borne_by_pair = {pair: 0.0 for pair in pairs}
        orphan_cost = 0.0
        for pair, cost in transaction_cost_by_pair.items():
            charged = min(pair_capital[pair], cost)
            pair_capital[pair] -= charged
            cost_borne_by_pair[pair] += charged
            orphan_cost += cost - charged
        positive_capital = sum(value for value in pair_capital.values() if value > 0.0)
        cost_borne_by_cash = 0.0
        if orphan_cost and positive_capital > orphan_cost:
            for pair in pairs:
                if pair_capital[pair] > 0.0:
                    share = orphan_cost * pair_capital[pair] / positive_capital
                    pair_capital[pair] -= share
                    cost_borne_by_pair[pair] += share
        else:
            cash -= orphan_cost
            cost_borne_by_cash = orphan_cost
        leg_a = desired_a
        leg_b = desired_b
        nav_close = cash + sum(pair_capital.values())
        if not np.isfinite(nav_close) or nav_close <= 0:
            raise RuntimeError("portfolio NAV became nonpositive after trading")
        portfolio_return = nav_close / previous_nav - 1.0
        identity_return = (gross_pnl - borrow_cost - transaction_cost) / previous_nav
        if not math.isclose(portfolio_return, identity_return, rel_tol=0.0, abs_tol=1e-11):
            raise RuntimeError("portfolio accounting identity failed")
        selected_before_terminal = set(selected)
        synced_before_terminal = dict(synced)
        capital_before_terminal_sweep = dict(pair_capital)
        if terminal:
            cash += sum(pair_capital.values())
            pair_capital = {pair: 0.0 for pair in pairs}
            synced = {pair: False for pair in pairs}
            selected = set()
        unallocated_cash_weight = cash / nav_close
        idle_sleeve_capital = sum(
            pair_capital[pair]
            for pair in pairs
            if abs(leg_a[pair]) + abs(leg_b[pair]) <= tolerance
        )
        for pair in pairs:
            detail_rows.append(
                {
                    "date": date,
                    "portfolio_id": str(portfolio_id),
                    "pair": pair,
                    "ticker_a": str(day.at[pair, "ticker_a"]),
                    "ticker_b": str(day.at[pair, "ticker_b"]),
                    "allocation_fill": allocation_fill,
                    "selected_after_fill": pair in selected_before_terminal,
                    "synced_after_fill": bool(synced_before_terminal[pair]),
                    "capital_open": capital_open[pair],
                    "opening_leg_a": opening_a[pair],
                    "opening_leg_b": opening_b[pair],
                    "return_a": return_a_by_pair[pair],
                    "return_b": return_b_by_pair[pair],
                    "gross_pnl": gross_pnl_by_pair[pair],
                    "borrow_cost": borrow_cost_by_pair[pair],
                    "capital_pretrade": (
                        capital_open[pair]
                        + gross_pnl_by_pair[pair]
                        - borrow_cost_by_pair[pair]
                    ),
                    "drifted_leg_a": drift_a[pair],
                    "drifted_leg_b": drift_b[pair],
                    "desired_capital_pre_cost": desired_capital[pair],
                    "allocation_transfer": (
                        desired_capital[pair]
                        - capital_open[pair]
                        - gross_pnl_by_pair[pair]
                        + borrow_cost_by_pair[pair]
                    ),
                    "shadow_target_weight_a": float(day.at[pair, "target_weight_a"]),
                    "shadow_target_weight_b": float(day.at[pair, "target_weight_b"]),
                    "closing_leg_a": desired_a[pair],
                    "closing_leg_b": desired_b[pair],
                    "leg_turnover": (
                        abs(desired_a[pair] - drift_a[pair])
                        + abs(desired_b[pair] - drift_b[pair])
                    ),
                    "transaction_cost": transaction_cost_by_pair[pair],
                    "transaction_cost_borne_by_pair": cost_borne_by_pair[pair],
                    "transaction_cost_borne_by_cash": (
                        cost_borne_by_cash if pair == pairs[0] else 0.0
                    ),
                    "capital_post_cost_pre_terminal_sweep": capital_before_terminal_sweep[pair],
                    "terminal_liquidation": terminal,
                }
            )
        rows.append(
            {
                "date": date,
                "portfolio_id": str(portfolio_id),
                "portfolio_return": portfolio_return,
                "equity": nav_close,
                "gross_pnl": gross_pnl,
                "gross_return_on_open_nav": gross_pnl / nav_open,
                "transaction_cost": transaction_cost,
                "transaction_cost_rate": transaction_cost / nav_open,
                "borrow_cost": borrow_cost,
                "borrow_cost_rate": borrow_cost / nav_open,
                "leg_turnover": turnover,
                "leg_turnover_rate": turnover / nav_open,
                "allocation_fill": allocation_fill,
                "selected_pair_count": len(selected),
                "synced_pair_count": sum(synced[pair] for pair in selected),
                "selected_pairs": "|".join(sorted(selected)),
                "synced_pairs": "|".join(
                    sorted(pair for pair in selected if synced[pair])
                ),
                "unallocated_cash_weight": unallocated_cash_weight,
                "idle_sleeve_weight": idle_sleeve_capital / nav_close,
                "cash_weight": (cash + idle_sleeve_capital) / nav_close,
                "gross_exposure": (
                    sum(abs(leg_a[pair]) + abs(leg_b[pair]) for pair in pairs)
                    / nav_close
                ),
                "terminal_liquidation": terminal,
            }
        )
        previous_nav = nav_close
    daily_result = pd.DataFrame(rows).set_index("date")
    if return_detail:
        return daily_result, pd.DataFrame(detail_rows)
    return daily_result


def simulate_managed_sleeves(
    pair_returns: pd.DataFrame,
    selection_events: pd.DataFrame,
    *,
    portfolio_id: str,
    slots: int | None,
    allocation_cost_bps: float,
) -> pd.DataFrame:
    """Simulate quarterly rebalanced pair-sleeve return indices.

    Pair returns are treated as investable managed-sleeve NAV returns.  A
    selection made on date t is applied after returns on its explicit fill
    date.  The selected sleeves therefore start contributing on the next
    session.  Allocation cost is applied to post-return NAV.
    """

    returns = pair_returns.copy()
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    if returns.index.isna().any():
        raise ValueError("pair return index contains invalid dates")
    if returns.index.duplicated().any() or not returns.index.is_monotonic_increasing:
        raise ValueError("pair return index must be unique and increasing")
    if returns.columns.duplicated().any():
        raise ValueError("pair return columns must be unique")
    returns = returns.apply(pd.to_numeric, errors="coerce")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("pair returns must be finite")
    if (values <= -1.0).any():
        raise ValueError("pair returns must be greater than -100%")
    if slots is not None and (int(slots) != slots or int(slots) <= 0):
        raise ValueError("slots must be a positive integer or None")
    cost_bps = float(allocation_cost_bps)
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("allocation_cost_bps must be finite and nonnegative")

    required_events = {"decision_date", "fill_date", "selected_pairs"}
    missing = sorted(required_events - set(selection_events.columns))
    if missing:
        raise ValueError(f"selection events missing columns: {missing}")
    events = selection_events.copy()
    events["decision_date"] = pd.to_datetime(events["decision_date"], errors="coerce")
    events["fill_date"] = pd.to_datetime(events["fill_date"], errors="coerce")
    if events[["decision_date", "fill_date"]].isna().any().any():
        raise ValueError("selection events contain invalid dates")
    if events["fill_date"].duplicated().any():
        raise ValueError("selection events contain duplicate fill dates")
    if not events["fill_date"].isin(returns.index).all():
        raise ValueError("every fill date must be a pair-return session")
    if not events["decision_date"].lt(events["fill_date"]).all():
        raise ValueError("each allocation fill must follow its decision")
    event_by_fill: dict[pd.Timestamp, dict[str, Any]] = {}
    available_pairs = set(str(column) for column in returns.columns)
    for raw in events.sort_values("fill_date", kind="mergesort").to_dict("records"):
        selected = tuple(str(pair) for pair in raw["selected_pairs"])
        if len(selected) != len(set(selected)):
            raise ValueError("a selection event contains duplicate pairs")
        if set(selected) - available_pairs:
            raise ValueError("a selection event contains an unknown pair")
        if slots is not None and len(selected) > int(slots):
            raise ValueError("selected pairs exceed the configured slots")
        raw["selected_pairs"] = selected
        event_by_fill[pd.Timestamp(raw["fill_date"])] = raw

    pair_amounts = {str(pair): 0.0 for pair in returns.columns}
    cash_amount = 1.0
    active_decision = pd.NaT
    active_selected: tuple[str, ...] = ()
    rows: list[dict[str, Any]] = []
    previous_nav = 1.0
    for date, row in returns.iterrows():
        for pair in pair_amounts:
            pair_amounts[pair] *= 1.0 + float(row[pair])
        pre_cost_nav = cash_amount + sum(pair_amounts.values())
        if not np.isfinite(pre_cost_nav) or pre_cost_nav <= 0:
            raise RuntimeError("portfolio NAV became nonpositive or nonfinite")
        pre_cost_return = pre_cost_nav / previous_nav - 1.0
        turnover = 0.0
        cost_rate = 0.0
        fill_event = date in event_by_fill
        if fill_event:
            event = event_by_fill[date]
            selected = event["selected_pairs"]
            denominator = int(slots) if slots is not None else len(selected)
            pair_weight = 1.0 / denominator if denominator else 0.0
            target_pair_weights = {
                pair: pair_weight if pair in selected else 0.0
                for pair in pair_amounts
            }
            target_cash_weight = 1.0 - pair_weight * len(selected)
            current_pair_weights = {
                pair: amount / pre_cost_nav for pair, amount in pair_amounts.items()
            }
            current_cash_weight = cash_amount / pre_cost_nav
            turnover = 0.5 * (
                sum(
                    abs(target_pair_weights[pair] - current_pair_weights[pair])
                    for pair in pair_amounts
                )
                + abs(target_cash_weight - current_cash_weight)
            )
            cost_rate = turnover * cost_bps / 10_000.0
            post_cost_nav = pre_cost_nav * (1.0 - cost_rate)
            if post_cost_nav <= 0:
                raise RuntimeError("allocation cost exhausted portfolio NAV")
            pair_amounts = {
                pair: post_cost_nav * target_pair_weights[pair]
                for pair in pair_amounts
            }
            cash_amount = post_cost_nav * target_cash_weight
            active_decision = pd.Timestamp(event["decision_date"])
            active_selected = selected
        else:
            post_cost_nav = pre_cost_nav
            target_cash_weight = cash_amount / post_cost_nav
        portfolio_return = post_cost_nav / previous_nav - 1.0
        expected = (1.0 + pre_cost_return) * (1.0 - cost_rate) - 1.0
        if not math.isclose(portfolio_return, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("post-return allocation-cost identity failed")
        rows.append(
            {
                "date": date,
                "portfolio_id": str(portfolio_id),
                "allocation_cost_bps": cost_bps,
                "portfolio_return": portfolio_return,
                "equity": post_cost_nav,
                "pre_allocation_cost_return": pre_cost_return,
                "allocation_turnover": turnover,
                "allocation_cost_rate": cost_rate,
                "allocation_cost_amount": pre_cost_nav * cost_rate,
                "allocation_fill_event": fill_event,
                "active_selection_decision_date": active_decision,
                "selected_pair_count": len(active_selected),
                "target_cash_weight": target_cash_weight,
            }
        )
        previous_nav = post_cost_nav
    result = pd.DataFrame(rows).set_index("date")
    reconstructed = (1.0 + result["portfolio_return"]).cumprod()
    if not np.allclose(
        reconstructed.to_numpy(), result["equity"].to_numpy(), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("portfolio return/equity reconstruction failed")
    return result
