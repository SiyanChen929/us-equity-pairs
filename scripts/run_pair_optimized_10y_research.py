#!/usr/bin/env python3
"""Run the additive ten-year, three-year-train, pair-return V6 study.

The existing eight-year reports are immutable inputs to neither selection nor
performance.  V6 owns a separate output root, rolls the complete frozen current
member universe through date-bounded three-year training windows, and never
promotes a quota filler.  Hyperparameters are selected independently for each
pair; no portfolio-return objective or allocator enters the primary result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity_pairs.config import StrategyConfig
from equity_pairs.data import download_adjusted_close, load_timeseries_snapshot
from equity_pairs.pair_tuning import (
    PairBatchTuningResult,
    PairTuningResult,
    PairTuningSpec,
    evaluate_frozen_pair_parameter,
    evaluate_tuned_pair,
    protocol_parameter_sets,
    tune_eligible_pairs,
    tune_pair_parameters,
)
from equity_pairs.statistics import PairScreenSettings, screen_sector_pairs


DEFAULT_CONFIG = Path("configs/pair_optimized_10y_3y.json")
DEFAULT_OUTPUT = Path(
    "outputs/pair_return_optimization_10y_train3y_2026-08-26"
)
PAIR_TUNING_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "equity_pairs" / "pair_tuning.py"
)
NEXT_CLOSE_EXECUTION_CONVENTION = {
    "signal_time": "Close[t]",
    "fill_time": "Close[t+1]",
    "first_pnl_interval": "Close[t+1] -> Close[t+2]",
}
SOURCE_UNIVERSE = Path(
    "outputs/pairs_research_2026-08-26/data/raw/"
    "sp500_current_constituents.csv"
)
SOURCE_UNIVERSE_METADATA = SOURCE_UNIVERSE.with_name(
    "sp500_current_constituents.metadata.json"
)


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    raw: Path
    processed: Path
    screens: Path
    tuning: Path
    charts: Path

    @classmethod
    def create(cls, root: str | Path) -> "StudyPaths":
        resolved = Path(root).resolve()
        result = cls(
            root=resolved,
            raw=resolved / "data" / "raw",
            processed=resolved / "data" / "processed",
            screens=resolved / "data" / "processed" / "screens",
            tuning=resolved / "data" / "processed" / "tuning",
            charts=resolved / "charts",
        )
        for directory in (
            result.root,
            result.raw,
            result.processed,
            result.screens,
            result.tuning,
            result.charts,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return result


@dataclass
class VintageArtifacts:
    """All machine-readable outputs for one independently resumable vintage."""

    parameter_rankings: pd.DataFrame
    fold_evaluations: pd.DataFrame
    model_audit: pd.DataFrame
    validation_returns: pd.DataFrame
    selected_validation_signals: pd.DataFrame
    validation_cost_sensitivity: pd.DataFrame
    selected_schedule: pd.DataFrame
    oos_signals: pd.DataFrame
    oos_trades: pd.DataFrame
    pair_vintage_metrics: pd.DataFrame
    baseline_signals: pd.DataFrame
    baseline_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    failures: pd.DataFrame
    vintage_summary: pd.DataFrame


_VINTAGE_TABLE_FILES = {
    "parameter_rankings": "parameter_rankings.csv.gz",
    "fold_evaluations": "inner_fold_evaluations.csv.gz",
    "model_audit": "inner_model_audit.csv",
    "validation_returns": "validation_returns.csv.gz",
    "selected_validation_signals": "selected_validation_signals.csv.gz",
    "validation_cost_sensitivity": "validation_cost_sensitivity.csv",
    "selected_schedule": "selected_parameter_schedule.csv",
    "oos_signals": "oos_signals.csv.gz",
    "oos_trades": "oos_trades.csv",
    "pair_vintage_metrics": "pair_vintage_metrics.csv",
    "baseline_signals": "oos_baseline_signals.csv.gz",
    "baseline_metrics": "oos_baseline_metrics.csv",
    "cost_sensitivity": "cost_sensitivity.csv",
    "failures": "failures.csv",
    "vintage_summary": "vintage_summary.csv",
}


_GLOBAL_TABLE_FILES = {
    "parameter_rankings": "pair_v6_parameter_rankings.csv.gz",
    "fold_evaluations": "pair_v6_inner_fold_evaluations.csv.gz",
    "model_audit": "pair_v6_inner_model_audit.csv",
    "validation_returns": "pair_v6_validation_returns.csv.gz",
    "selected_validation_signals": "pair_v6_selected_validation_signals.csv.gz",
    "validation_cost_sensitivity": "pair_v6_validation_cost_sensitivity.csv",
    "selected_schedule": "pair_v6_selected_parameter_schedule.csv",
    "oos_signals": "pair_v6_oos_signals.csv.gz",
    "oos_trades": "pair_v6_oos_trades.csv",
    "pair_vintage_metrics": "pair_v6_pair_vintage_metrics.csv",
    "baseline_signals": "pair_v6_oos_baseline_signals.csv.gz",
    "baseline_metrics": "pair_v6_oos_baseline_metrics.csv",
    "cost_sensitivity": "pair_v6_cost_sensitivity.csv",
    "failures": "pair_v6_tuning_failures.csv",
    "vintage_summary": "pair_v6_tuning_vintage_summary.csv",
}


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PAIR_TUNING_SOURCE_SHA256 = _sha256(PAIR_TUNING_SOURCE)


def _execution_cache_lineage() -> dict[str, Any]:
    return {
        "execution_convention": dict(NEXT_CLOSE_EXECUTION_CONVENTION),
        "pair_tuning_source_sha256": PAIR_TUNING_SOURCE_SHA256,
    }


def _cache_lineage_matches(metadata: dict[str, Any]) -> bool:
    expected = _execution_cache_lineage()
    return all(metadata.get(key) == value for key, value in expected.items())


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a table completely before exposing it to a resume attempt."""

    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _bool_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"}).fillna(False)


def _hash_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    usable = [column for column in columns if column in frame]
    payload = (
        frame[usable]
        .copy()
        .sort_values(usable, kind="mergesort")
        .to_json(orient="records", date_format="iso", double_precision=15)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol",
        "data",
        "rolling_windows",
        "screening",
        "pair_selection",
        "hyperparameter_grid",
        "hyperparameter_validation",
        "costs",
        "evaluation",
        "leakage_controls",
        "outputs",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("protocol is missing sections: " + ", ".join(missing))
    if payload["pair_selection"]["portfolio_return_used_for_ranking"]:
        raise ValueError("V6 forbids a portfolio-return ranking objective")
    if payload["screening"]["allow_quota_fills"]:
        raise ValueError("V6 forbids quota fillers")
    costs = tuple(float(value) for value in payload["costs"]["transaction_cost_sensitivity_bps"])
    primary = float(payload["costs"]["primary_transaction_cost_bps"])
    if primary not in costs or min(costs) < 1.0 or max(costs) > 2.0:
        raise ValueError("V6 costs must contain the primary rate and stay in [1, 2] bp")
    return payload


def _build_pair_tuning_protocol(
    protocol: dict[str, Any],
) -> tuple[PairTuningSpec, StrategyConfig]:
    """Translate the frozen JSON contract into the pair-only tuning APIs."""

    grid = protocol["hyperparameter_grid"]
    validation = protocol["hyperparameter_validation"]
    costs = protocol["costs"]
    baseline = validation["baseline"]
    spec = PairTuningSpec(
        zscore_methods=tuple(str(value) for value in grid["zscore_methods"]),
        zscore_lookbacks=tuple(int(value) for value in grid["zscore_lookbacks"]),
        entry_zs=tuple(float(value) for value in grid["entry_zs"]),
        exit_zs=tuple(float(value) for value in grid["exit_zs"]),
        maximum_holding_days=tuple(
            int(value) for value in grid["maximum_holding_days"]
        ),
        inner_train_size=int(validation["inner_train_sessions"]),
        inner_validation_size=int(validation["inner_validation_sessions"]),
        inner_step_size=int(validation["inner_step_sessions"]),
        purge_size=int(validation["purge_sessions"]),
        embargo_size=int(validation["embargo_sessions"]),
        maximum_inner_folds=int(validation["maximum_inner_folds"]),
        primary_transaction_cost_bps=float(
            costs["primary_transaction_cost_bps"]
        ),
        sensitivity_transaction_cost_bps=tuple(
            float(value) for value in costs["transaction_cost_sensitivity_bps"]
        ),
        minimum_full_training_trades=int(
            validation["minimum_full_training_trades"]
        ),
        minimum_validation_trades=int(
            validation["minimum_pooled_validation_trades"]
        ),
        minimum_active_folds=int(
            validation["minimum_active_validation_folds"]
        ),
        minimum_hedge_observations=100,
        annualization=252,
        one_standard_error_rule=validation["selection_rule"]
        == "one_standard_error",
    )
    base = StrategyConfig(
        zscore_method=str(baseline["zscore_method"]),
        zscore_lookback=int(baseline["zscore_lookback"]),
        zscore_min_periods=int(baseline["zscore_min_periods"]),
        entry_z=float(baseline["entry_z"]),
        exit_z=float(baseline["exit_z"]),
        stop_z=float(grid["stop_z"]),
        maximum_holding_days=int(baseline["maximum_holding_days"]),
        transaction_cost_bps=float(costs["primary_transaction_cost_bps"]),
        transaction_cost_scenarios_bps=tuple(
            float(value) for value in costs["transaction_cost_sensitivity_bps"]
        ),
        annual_short_borrow_bps=float(costs["annual_short_borrow_bps"]),
        minimum_trades_for_ranking=max(
            1, int(validation["minimum_full_training_trades"])
        ),
    )
    count = len(protocol_parameter_sets(spec, stop_z=base.stop_z))
    expected = int(grid["cartesian_cell_count"]) + 1
    if count != expected or count != 97:
        raise ValueError(
            f"V6 requires 96 searched cells plus baseline; received {count} candidates"
        )
    return spec, base


def _prepare_data(
    paths: StudyPaths,
    protocol: dict[str, Any],
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    universe_path = paths.raw / "sp500_current_constituents.csv"
    universe_metadata_path = paths.raw / "sp500_current_constituents.metadata.json"
    if not universe_path.exists():
        if not SOURCE_UNIVERSE.exists():
            raise FileNotFoundError(f"frozen universe snapshot missing: {SOURCE_UNIVERSE}")
        shutil.copy2(SOURCE_UNIVERSE, universe_path)
    if not universe_metadata_path.exists() and SOURCE_UNIVERSE_METADATA.exists():
        shutil.copy2(SOURCE_UNIVERSE_METADATA, universe_metadata_path)
    universe = pd.read_csv(universe_path, dtype={"cik": "string"})
    if universe["ticker"].duplicated().any():
        raise ValueError("frozen universe contains duplicate tickers")

    data = protocol["data"]
    price_path = paths.raw / str(data["price_snapshot_name"])
    prices = download_adjusted_close(
        universe["ticker"].astype(str).tolist(),
        data["start_inclusive"],
        data["end_exclusive"],
        chunk_size=int(data["download_chunk_size"]),
        max_retries=int(data["download_max_retries"]),
        backoff_factor=float(data["download_backoff_factor"]),
        inter_chunk_delay=float(data["download_inter_chunk_delay_seconds"]),
        repair=bool(data["repair_prices"]),
        strict=False,
        cache_path=price_path,
        refresh=refresh,
    ).sort_index()
    verified, price_metadata = load_timeseries_snapshot(price_path)
    if not verified.index.equals(prices.index) or list(verified.columns) != list(prices.columns):
        raise RuntimeError("verified Yahoo snapshot does not match the loaded frame")
    if prices.index.min() > pd.Timestamp(data["start_inclusive"]):
        raise RuntimeError("Yahoo snapshot starts after the requested ten-year boundary")
    if prices.index.max() >= pd.Timestamp(data["end_exclusive"]):
        raise RuntimeError("Yahoo snapshot violates the exclusive end boundary")
    return universe, prices, price_metadata


def _quarterly_rebalance_dates(
    prices: pd.DataFrame,
    *,
    first_training_boundary: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DatetimeIndex:
    quarter_starts = pd.date_range(
        first_training_boundary.to_period("Q").start_time,
        as_of,
        freq="QS",
    )
    quarter_starts = quarter_starts[quarter_starts >= first_training_boundary]
    resolved: list[pd.Timestamp] = []
    for anchor in quarter_starts:
        eligible = prices.index[(prices.index >= anchor) & (prices.index <= as_of)]
        if len(eligible):
            resolved.append(pd.Timestamp(eligible[0]))
    result = pd.DatetimeIndex(sorted(set(resolved)), name="rebalance_date")
    if result.empty:
        raise ValueError("no quarterly rebalance dates fit the ten-year window")
    return result


def _rebalance_schedule(
    all_rebalance_dates: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp,
    maximum_rebalances: int | None = None,
) -> pd.DataFrame:
    """Keep the true next-quarter boundary even when a debug run is truncated."""

    all_dates = pd.DatetimeIndex(all_rebalance_dates).sort_values()
    selected = (
        all_dates[:maximum_rebalances]
        if maximum_rebalances is not None
        else all_dates
    )
    next_boundary = {
        date: (
            all_dates[location + 1]
            if location + 1 < len(all_dates)
            else as_of + pd.Timedelta(days=1)
        )
        for location, date in enumerate(all_dates)
    }
    schedule = pd.DataFrame({"rebalance_date": selected})
    schedule["outer_end_exclusive"] = [next_boundary[date] for date in selected]
    schedule["outer_window_truncated_by_asof"] = schedule[
        "outer_end_exclusive"
    ].gt(as_of)
    return schedule


def _screen_payload(
    *,
    rebalance_date: pd.Timestamp,
    requested_start: pd.Timestamp,
    formation: pd.DataFrame,
    settings: PairScreenSettings,
    universe_sha256: str,
    price_snapshot_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "rebalance_date": str(rebalance_date.date()),
        "formation_start_requested": str(requested_start.date()),
        "formation_start_actual": str(formation.index.min().date()),
        "formation_end_actual": str(formation.index.max().date()),
        "formation_end_exclusive": str(rebalance_date.date()),
        "sessions": int(len(formation)),
        "minimum_observations": int(settings.minimum_observations),
        "minimum_history_ratio": float(settings.minimum_history_ratio),
        "alpha": float(settings.alpha),
        "maximum_half_life_days": int(settings.maximum_half_life_days),
        "adf_maxlag": int(settings.adf_maxlag),
        "exclude_same_issuer": bool(settings.exclude_same_issuer),
        "universe_sha256": universe_sha256,
        "price_snapshot_sha256": price_snapshot_sha256,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _screen_paths(paths: StudyPaths, date: pd.Timestamp) -> tuple[Path, Path, Path]:
    token = date.strftime("%Y%m%d")
    return (
        paths.screens / f"pair_v6_screen_{token}.csv.gz",
        paths.screens / f"pair_v6_integration_{token}.csv",
        paths.screens / f"pair_v6_screen_{token}.meta.json",
    )


def _screen_one_window(
    paths: StudyPaths,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    protocol: dict[str, Any],
    price_metadata: dict[str, Any],
    rebalance_date: pd.Timestamp,
    *,
    workers: int,
    recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    requested_start = rebalance_date - pd.DateOffset(years=3)
    formation = prices.loc[
        (prices.index >= requested_start) & (prices.index < rebalance_date)
    ].copy()
    if len(formation) < 500:
        raise ValueError(f"training window at {rebalance_date.date()} is too short")
    screen = protocol["screening"]
    ratio = float(screen["minimum_history_ratio"])
    settings = PairScreenSettings(
        minimum_observations=int(math.ceil(ratio * len(formation))),
        minimum_history_ratio=ratio,
        alpha=float(screen["cointegration_alpha"]),
        maximum_half_life_days=int(screen["maximum_half_life_days"]),
        adf_maxlag=int(screen["adf_maxlag"]),
        exclude_same_issuer=bool(screen["exclude_same_issuer"]),
        workers=workers,
    )
    universe_path = paths.raw / "sp500_current_constituents.csv"
    payload = _screen_payload(
        rebalance_date=rebalance_date,
        requested_start=requested_start,
        formation=formation,
        settings=settings,
        universe_sha256=_sha256(universe_path),
        price_snapshot_sha256=str(price_metadata["sha256"]),
    )
    diagnostics_path, integration_path, metadata_path = _screen_paths(
        paths, rebalance_date
    )
    if not recompute and diagnostics_path.exists() and integration_path.exists() and metadata_path.exists():
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if saved.get("fingerprint") == payload["fingerprint"]:
            diagnostics = pd.read_csv(diagnostics_path)
            integration = pd.read_csv(integration_path)
            payload["candidate_pairs"] = int(len(diagnostics))
            payload["strict_eligible"] = int(
                _bool_series(diagnostics, "eligible", False).sum()
            )
            payload["elapsed_seconds"] = saved.get("elapsed_seconds")
            missing_audit_fields = {
                "candidate_pairs",
                "strict_eligible",
                "elapsed_seconds",
            } - set(saved)
            if missing_audit_fields:
                payload["cache_metadata_enriched_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                payload["legacy_missing_fields"] = sorted(missing_audit_fields)
                _json_dump(metadata_path, payload)
            return diagnostics, integration, payload, True

    started = time.monotonic()
    diagnostics, integration = screen_sector_pairs(formation, universe, settings)
    diagnostics.to_csv(diagnostics_path, index=False, compression="gzip")
    integration.to_csv(integration_path, index=False)
    payload["elapsed_seconds"] = float(time.monotonic() - started)
    payload["candidate_pairs"] = int(len(diagnostics))
    payload["strict_eligible"] = int(diagnostics["eligible"].fillna(False).sum())
    _json_dump(metadata_path, payload)
    return diagnostics, integration, payload, False


def _run_screens(
    paths: StudyPaths,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    protocol: dict[str, Any],
    price_metadata: dict[str, Any],
    rebalance_dates: pd.DatetimeIndex,
    *,
    workers: int,
    recompute: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    strict_rows: list[pd.DataFrame] = []
    for location, date in enumerate(rebalance_dates, start=1):
        diagnostics, _, metadata, reused = _screen_one_window(
            paths,
            prices,
            universe,
            protocol,
            price_metadata,
            pd.Timestamp(date),
            workers=workers,
            recompute=recompute,
        )
        eligible = diagnostics["eligible"].fillna(False).astype(bool)
        strict = diagnostics.loc[eligible].copy()
        if len(strict):
            strict.insert(0, "rebalance_date", pd.Timestamp(date))
            strict_rows.append(strict)
        row = {
            "rebalance_date": pd.Timestamp(date),
            "formation_start": metadata["formation_start_actual"],
            "formation_end": metadata["formation_end_actual"],
            "training_sessions": metadata["sessions"],
            "minimum_observations": metadata["minimum_observations"],
            "candidate_pairs": int(len(diagnostics)),
            "screen_ok": int(diagnostics["screen_status"].eq("ok").sum()),
            "raw_p_lt_005": int(
                pd.to_numeric(diagnostics["pair_pvalue"], errors="coerce").lt(0.05).sum()
            ),
            "fdr_significant": int(
                diagnostics["fdr_significant"].fillna(False).sum()
            ),
            "strict_eligible": int(len(strict)),
            "strict_sectors": int(strict["sector"].nunique()) if len(strict) else 0,
            "minimum_fdr_qvalue": float(
                pd.to_numeric(diagnostics["fdr_qvalue"], errors="coerce").min()
            ),
            "cache_reused": bool(reused),
        }
        summary_rows.append(row)
        _log(
            f"screen {location}/{len(rebalance_dates)} {date.date()}: "
            f"{row['strict_eligible']} strict, {row['fdr_significant']} FDR, "
            f"cache={'yes' if reused else 'no'}"
        )
    summary = pd.DataFrame(summary_rows)
    strict_all = (
        pd.concat(strict_rows, ignore_index=True)
        if strict_rows
        else pd.DataFrame(columns=["rebalance_date", "pair", "sector"])
    )
    summary.to_csv(paths.processed / "pair_v6_rolling_screen_summary.csv", index=False)
    strict_all.to_csv(
        paths.processed / "pair_v6_strict_candidates.csv.gz",
        index=False,
        compression="gzip",
    )
    return summary, strict_all


def _strict_candidates_for_vintage(
    strict_all: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """Fail closed: only the full strict gate can enter pair-level tuning."""

    if strict_all.empty:
        return strict_all.copy()
    working = strict_all.copy()
    working["rebalance_date"] = pd.to_datetime(working["rebalance_date"])
    working = working.loc[working["rebalance_date"].eq(rebalance_date)].copy()
    if working.empty:
        return working
    required = {
        "pair",
        "sector",
        "dependent",
        "independent",
        "eligible",
        "fdr_significant",
        "fdr_qvalue",
        "screen_status",
        "positive_hedge_ratio",
        "both_look_i1",
        "half_life_days",
        "hurst_exponent",
    }
    missing = sorted(required - set(working.columns))
    if missing:
        raise ValueError(
            f"strict candidate table for {rebalance_date.date()} is missing: "
            + ", ".join(missing)
        )
    mask = _bool_series(working, "eligible", False)
    mask &= _bool_series(working, "fdr_significant", False)
    mask &= _bool_series(working, "positive_hedge_ratio", False)
    mask &= _bool_series(working, "both_look_i1", False)
    if "tradable_eligible" in working:
        mask &= _bool_series(working, "tradable_eligible", False)
    mask &= working["screen_status"].astype(str).eq("ok")
    mask &= pd.to_numeric(working["fdr_qvalue"], errors="coerce").le(0.05)
    mask &= pd.to_numeric(working["half_life_days"], errors="coerce").between(
        2.0, 252.0
    )
    mask &= pd.to_numeric(working["hurst_exponent"], errors="coerce").lt(0.50)
    if "selection_role" in working:
        mask &= ~working["selection_role"].astype(str).eq("ineligible_quota_fill")
    result = working.loc[mask].copy()
    result["tradable_eligible"] = True
    result["selection_role"] = "strict_eligible"
    if result["pair"].astype(str).duplicated().any():
        raise ValueError(
            f"strict candidates contain duplicate pairs at {rebalance_date.date()}"
        )
    return result.sort_values(
        ["fdr_qvalue", "pair_pvalue", "half_life_days", "pair"],
        kind="mergesort",
    ).reset_index(drop=True)


def _vintage_directory(paths: StudyPaths, rebalance_date: pd.Timestamp) -> Path:
    directory = paths.tuning / rebalance_date.strftime("%Y%m%d")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _vintage_fingerprint(
    *,
    rebalance_date: pd.Timestamp,
    outer_end_exclusive: pd.Timestamp,
    formation: pd.DataFrame,
    outer: pd.DataFrame,
    strict: pd.DataFrame,
    spec: PairTuningSpec,
    base_config: StrategyConfig,
    protocol: dict[str, Any],
    price_snapshot_sha256: str,
) -> str:
    candidate_hash = _hash_frame(
        strict,
        [
            "pair",
            "sector",
            "dependent",
            "independent",
            "pair_pvalue",
            "fdr_qvalue",
            "half_life_days",
            "hurst_exponent",
            "eligible",
        ],
    )
    payload = {
        "schema_version": 2,
        "rebalance_date": str(rebalance_date.date()),
        "outer_end_exclusive": str(outer_end_exclusive.date()),
        "formation_start": str(formation.index.min().date()),
        "formation_end": str(formation.index.max().date()),
        "formation_sessions": len(formation),
        "outer_start": str(outer.index.min().date()) if len(outer) else None,
        "outer_end": str(outer.index.max().date()) if len(outer) else None,
        "outer_sessions": len(outer),
        "strict_candidate_hash": candidate_hash,
        "strict_candidate_count": len(strict),
        "pair_tuning_spec": asdict(spec),
        "strategy_config": asdict(base_config),
        "pair_selection": protocol["pair_selection"],
        "screening": protocol["screening"],
        "price_snapshot_sha256": price_snapshot_sha256,
        **_execution_cache_lineage(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_vintage_cache(
    directory: Path,
    fingerprint: str,
) -> VintageArtifacts | None:
    metadata_path = directory / "complete.meta.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not metadata.get("complete")
        or metadata.get("fingerprint") != fingerprint
        or not _cache_lineage_matches(metadata)
    ):
        return None
    paths = {
        field: directory / filename
        for field, filename in _VINTAGE_TABLE_FILES.items()
    }
    if any(not path.exists() for path in paths.values()):
        return None
    tables = {field: _read_csv_or_empty(path) for field, path in paths.items()}
    summary = tables["vintage_summary"]
    if not summary.empty:
        summary["cache_reused"] = True
    return VintageArtifacts(**tables)


def _write_vintage_cache(
    directory: Path,
    artifacts: VintageArtifacts,
    *,
    fingerprint: str,
) -> None:
    metadata_path = directory / "complete.meta.json"
    if metadata_path.exists():
        metadata_path.unlink()
    for field, filename in _VINTAGE_TABLE_FILES.items():
        _atomic_to_csv(getattr(artifacts, field), directory / filename)
    summary = (
        artifacts.vintage_summary.iloc[0].to_dict()
        if len(artifacts.vintage_summary)
        else {}
    )
    _json_dump(
        metadata_path,
        {
            "schema_version": 2,
            "complete": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            **_execution_cache_lineage(),
            "summary": summary,
            "files": {
                filename: _sha256(directory / filename)
                for filename in _VINTAGE_TABLE_FILES.values()
            },
        },
    )


def _contextualize(
    frame: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    for column, value in reversed(list(context.items())):
        if column in result:
            result[column] = value
        else:
            result.insert(0, column, value)
    return result


def _selected_training_row(result: PairTuningResult) -> pd.Series:
    rows = result.ranking.loc[result.ranking["selected"].astype(bool)]
    if len(rows) != 1:
        raise RuntimeError(
            f"{result.pair.pair} must have exactly one selected parameter row"
        )
    return rows.iloc[0]


def _screen_context(row: pd.Series) -> dict[str, Any]:
    def value(name: str) -> Any:
        return row[name] if name in row else math.nan

    return {
        "screen_pair_pvalue": value("pair_pvalue"),
        "screen_fdr_qvalue": value("fdr_qvalue"),
        "screen_half_life_days": value("half_life_days"),
        "screen_hurst_exponent": value("hurst_exponent"),
    }


def _empty_vintage_artifacts(summary: dict[str, Any]) -> VintageArtifacts:
    return VintageArtifacts(
        parameter_rankings=pd.DataFrame(columns=["rebalance_date", "pair", "config_id"]),
        fold_evaluations=pd.DataFrame(
            columns=["rebalance_date", "pair", "inner_fold", "config_id"]
        ),
        model_audit=pd.DataFrame(
            columns=["rebalance_date", "pair", "inner_fold", "alpha", "beta"]
        ),
        validation_returns=pd.DataFrame(columns=["rebalance_date", "pair", "date"]),
        selected_validation_signals=pd.DataFrame(
            columns=["rebalance_date", "pair", "date", "config_id"]
        ),
        validation_cost_sensitivity=pd.DataFrame(
            columns=["rebalance_date", "pair", "config_id", "transaction_cost_bps"]
        ),
        selected_schedule=pd.DataFrame(
            columns=["rebalance_date", "pair_rank", "pair", "selected_config_id"]
        ),
        oos_signals=pd.DataFrame(
            columns=["rebalance_date", "pair", "date", "config_id"]
        ),
        oos_trades=pd.DataFrame(
            columns=["rebalance_date", "pair", "entry_date", "exit_date"]
        ),
        pair_vintage_metrics=pd.DataFrame(
            columns=["rebalance_date", "pair", "config_id", "net_cagr"]
        ),
        baseline_signals=pd.DataFrame(
            columns=["rebalance_date", "pair", "date", "config_id"]
        ),
        baseline_metrics=pd.DataFrame(
            columns=["rebalance_date", "pair", "config_id", "net_cagr"]
        ),
        cost_sensitivity=pd.DataFrame(
            columns=["rebalance_date", "pair", "config_id", "transaction_cost_bps"]
        ),
        failures=pd.DataFrame(columns=["rebalance_date", "pair", "stage", "error"]),
        vintage_summary=pd.DataFrame([summary]),
    )


def _pair_batch_from_results(
    ordered_pairs: list[str],
    results: dict[str, PairTuningResult],
    failures: list[dict[str, str]],
) -> PairBatchTuningResult:
    leaderboard_rows: list[dict[str, Any]] = []
    for pair_name in ordered_pairs:
        if pair_name not in results:
            continue
        result = results[pair_name]
        selected = _selected_training_row(result)
        leaderboard_rows.append(
            {
                "pair": pair_name,
                "sector": result.pair.sector,
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
    ordered_results = {
        pair_name: results[pair_name]
        for pair_name in ordered_pairs
        if pair_name in results
    }
    return PairBatchTuningResult(
        results=ordered_results,
        leaderboard=leaderboard,
        failures=pd.DataFrame(
            failures, columns=["pair", "error_type", "error"]
        ),
    )


def _tune_vintage_pairs(
    strict: pd.DataFrame,
    prices: pd.DataFrame,
    formation_index: pd.DatetimeIndex,
    base_config: StrategyConfig,
    spec: PairTuningSpec,
    *,
    workers: int,
) -> PairBatchTuningResult:
    """Tune independent pair sleeves concurrently with deterministic assembly."""

    if workers < 1:
        raise ValueError("tuning workers must be positive")
    ordered_strict = strict.sort_values("pair", kind="mergesort").reset_index(
        drop=True
    )
    if workers == 1 or len(strict) <= 1:
        return tune_eligible_pairs(
            ordered_strict,
            prices,
            formation_index,
            base_config,
            spec,
            eligibility_column="tradable_eligible",
            continue_on_error=True,
        )
    records = ordered_strict.to_dict(orient="records")
    ordered_pairs = [str(record["pair"]) for record in records]
    results: dict[str, PairTuningResult] = {}
    failures: list[dict[str, str]] = []
    maximum_workers = min(int(workers), len(records))
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        futures = []
        for record in records:
            dependent = str(record["dependent"])
            independent = str(record["independent"])
            pair_prices = prices.loc[
                formation_index, [dependent, independent]
            ].copy()
            futures.append(
                (
                    str(record["pair"]),
                    executor.submit(
                        tune_pair_parameters,
                        record,
                        pair_prices,
                        formation_index,
                        base_config,
                        spec,
                    ),
                )
            )
        # Futures execute concurrently; consuming them in stable pair order
        # makes every downstream CSV deterministic across worker counts.
        for pair_name, future in futures:
            try:
                results[pair_name] = future.result()
            except Exception as error:
                failures.append(
                    {
                        "pair": pair_name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    return _pair_batch_from_results(ordered_pairs, results, failures)


def _run_one_tuning_vintage(
    paths: StudyPaths,
    prices: pd.DataFrame,
    strict_all: pd.DataFrame,
    protocol: dict[str, Any],
    price_metadata: dict[str, Any],
    spec: PairTuningSpec,
    base_config: StrategyConfig,
    rebalance_date: pd.Timestamp,
    outer_end_exclusive: pd.Timestamp,
    *,
    recompute: bool,
    tuning_workers: int = 1,
) -> tuple[VintageArtifacts, bool]:
    requested_start = rebalance_date - pd.DateOffset(years=3)
    formation = prices.loc[
        (prices.index >= requested_start) & (prices.index < rebalance_date)
    ].copy()
    outer = prices.loc[
        (prices.index >= rebalance_date) & (prices.index < outer_end_exclusive)
    ].copy()
    strict = _strict_candidates_for_vintage(strict_all, rebalance_date)
    fingerprint = _vintage_fingerprint(
        rebalance_date=rebalance_date,
        outer_end_exclusive=outer_end_exclusive,
        formation=formation,
        outer=outer,
        strict=strict,
        spec=spec,
        base_config=base_config,
        protocol=protocol,
        price_snapshot_sha256=str(price_metadata["sha256"]),
    )
    directory = _vintage_directory(paths, rebalance_date)
    if not recompute:
        cached = _load_vintage_cache(directory, fingerprint)
        if cached is not None:
            return cached, True

    common_summary = {
        "rebalance_date": rebalance_date,
        "formation_start": formation.index.min() if len(formation) else pd.NaT,
        "formation_end": formation.index.max() if len(formation) else pd.NaT,
        "outer_start": outer.index.min() if len(outer) else pd.NaT,
        "outer_end_exclusive": outer_end_exclusive,
        "training_sessions": int(len(formation)),
        "outer_sessions": int(len(outer)),
        "strict_candidate_count": int(len(strict)),
        "tuned_pair_count": 0,
        "tuning_failure_count": 0,
        "selected_pair_count": 0,
        "oos_success_count": 0,
        "oos_failure_count": 0,
        "baseline_success_count": 0,
        "baseline_failure_count": 0,
        "cache_reused": False,
        "portfolio_objective_used": False,
    }
    if strict.empty:
        artifacts = _empty_vintage_artifacts(common_summary)
        _write_vintage_cache(directory, artifacts, fingerprint=fingerprint)
        return artifacts, False
    if len(outer) < 2:
        raise ValueError(
            f"outer window at {rebalance_date.date()} has fewer than two sessions"
        )

    started = time.monotonic()
    batch: PairBatchTuningResult = _tune_vintage_pairs(
        strict,
        prices,
        formation.index,
        base_config,
        spec,
        workers=tuning_workers,
    )
    maximum_pairs = int(protocol["pair_selection"]["maximum_pairs_per_rebalance"])
    selected_leaderboard = batch.leaderboard.head(maximum_pairs).copy()
    selected_pairs = (
        selected_leaderboard["pair"].astype(str).tolist()
        if "pair" in selected_leaderboard
        else []
    )
    strict_by_pair = strict.set_index("pair", drop=False)

    ranking_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    model_parts: list[pd.DataFrame] = []
    validation_return_parts: list[pd.DataFrame] = []
    selected_validation_parts: list[pd.DataFrame] = []
    validation_cost_parts: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, Any]] = []
    oos_signal_parts: list[pd.DataFrame] = []
    oos_trade_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    baseline_signal_parts: list[pd.DataFrame] = []
    baseline_metric_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    if not batch.failures.empty:
        for failure in batch.failures.to_dict(orient="records"):
            failure_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "pair": failure["pair"],
                    "stage": "inner_tuning",
                    "error_type": failure["error_type"],
                    "error": failure["error"],
                }
            )

    base_context = {
        "rebalance_date": rebalance_date,
        "formation_start": formation.index.min(),
        "formation_end": formation.index.max(),
        "outer_start": outer.index.min(),
        "outer_end_exclusive": outer_end_exclusive,
    }
    for pair_name, result in batch.results.items():
        screen_row = strict_by_pair.loc[pair_name]
        context = {
            **base_context,
            "pair": pair_name,
            "sector": result.pair.sector,
            **_screen_context(screen_row),
        }
        ranking_parts.append(_contextualize(result.ranking, context))
        fold_parts.append(_contextualize(result.fold_evaluations, context))
        model_parts.append(_contextualize(result.fold_models, context))
        validation = result.validation_returns.reset_index()
        validation_return_parts.append(_contextualize(validation, context))
        selected_validation_parts.append(
            _contextualize(result.selected_fold_daily, context)
        )
        validation_cost_parts.append(
            _contextualize(result.cost_sensitivity, context)
        )

    baseline_parameter = next(
        parameter
        for parameter in protocol_parameter_sets(spec, stop_z=base_config.stop_z)
        if parameter.is_baseline
    )
    for pair_name in selected_pairs:
        result = batch.results[pair_name]
        leaderboard_row = selected_leaderboard.loc[
            selected_leaderboard["pair"].astype(str).eq(pair_name)
        ].iloc[0]
        pair_rank = int(leaderboard_row["pair_rank"])
        training = _selected_training_row(result)
        screen_row = strict_by_pair.loc[pair_name]
        parameter = result.selected_parameter
        schedule_rows.append(
            {
                **base_context,
                "outer_sessions": int(len(outer)),
                "pair_rank": pair_rank,
                "pair": pair_name,
                "sector": result.pair.sector,
                "dependent": result.pair.dependent,
                "independent": result.pair.independent,
                **_screen_context(screen_row),
                "selected_config_id": parameter.config_id,
                "zscore_method": parameter.zscore_method,
                "zscore_lookback": parameter.zscore_lookback,
                "zscore_min_periods": parameter.zscore_min_periods,
                "entry_z": parameter.entry_z,
                "exit_z": parameter.exit_z,
                "maximum_holding_days": parameter.maximum_holding_days,
                "is_baseline": parameter.is_baseline,
                "training_cv_net_cagr": training["pooled_net_cagr"],
                "training_cv_max_drawdown": training["max_drawdown"],
                "validation_trade_count": training["validation_trade_count"],
                "full_training_trade_count": training["full_training_trade_count"],
                "active_fold_count": training["active_fold_count"],
                "primary_transaction_cost_bps": spec.primary_transaction_cost_bps,
                "tradable_eligible": True,
                "selection_role": "strict_eligible",
                "portfolio_objective_used": False,
            }
        )
        primary_success = False
        primary_metric: dict[str, Any] | None = None
        scenario_costs = sorted(
            set(
                [float(spec.primary_transaction_cost_bps)]
                + [float(value) for value in spec.sensitivity_transaction_cost_bps]
            )
        )
        for cost in scenario_costs:
            try:
                evaluation = evaluate_tuned_pair(
                    result,
                    prices,
                    formation.index,
                    outer.index,
                    base_config,
                    transaction_cost_bps=cost,
                )
            except Exception as error:
                failure_rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "pair": pair_name,
                        "stage": f"outer_evaluation_{cost:g}bps",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            metric_context = {
                "rebalance_date": rebalance_date,
                "outer_end_exclusive": outer_end_exclusive,
                "pair_rank": pair_rank,
                "pair": pair_name,
                "sector": result.pair.sector,
                "config_id": parameter.config_id,
                "selection_cost_bps": spec.primary_transaction_cost_bps,
                "transaction_cost_bps": cost,
                "frozen_winner": True,
                "training_cv_net_cagr": training["pooled_net_cagr"],
                "training_cv_max_drawdown": training["max_drawdown"],
                **_screen_context(screen_row),
                **evaluation.metrics,
            }
            cost_rows.append(metric_context)
            if math.isclose(
                cost,
                float(spec.primary_transaction_cost_bps),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                primary_success = True
                primary_metric = metric_context
                metric_rows.append(primary_metric)
                signal_context = {
                    "rebalance_date": rebalance_date,
                    "outer_end_exclusive": outer_end_exclusive,
                    "pair_rank": pair_rank,
                    "pair": pair_name,
                    "sector": result.pair.sector,
                    "config_id": parameter.config_id,
                    "transaction_cost_bps": cost,
                }
                oos_signal_parts.append(
                    _contextualize(evaluation.daily.reset_index(), signal_context)
                )
                if not evaluation.trades.empty:
                    oos_trade_parts.append(
                        _contextualize(evaluation.trades, signal_context)
                    )
        if not primary_success:
            # A selected pair without its primary-cost replay cannot enter the
            # claimed OOS pair-vintage metrics.
            continue
        try:
            baseline = evaluate_frozen_pair_parameter(
                result.pair,
                prices,
                formation.index,
                outer.index,
                baseline_parameter,
                base_config,
                transaction_cost_bps=float(spec.primary_transaction_cost_bps),
                minimum_hedge_observations=spec.minimum_hedge_observations,
                terminal_liquidation=True,
            )
        except Exception as error:
            failure_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "pair": pair_name,
                    "stage": "outer_baseline_2bps",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            baseline_context = {
                "rebalance_date": rebalance_date,
                "outer_end_exclusive": outer_end_exclusive,
                "pair_rank": pair_rank,
                "pair": pair_name,
                "sector": result.pair.sector,
                "config_id": baseline_parameter.config_id,
                "transaction_cost_bps": spec.primary_transaction_cost_bps,
                "baseline_used_for_selection": False,
                **baseline.metrics,
            }
            baseline_metric_rows.append(baseline_context)
            baseline_signal_parts.append(
                _contextualize(
                    baseline.daily.reset_index(),
                    {
                        "rebalance_date": rebalance_date,
                        "outer_end_exclusive": outer_end_exclusive,
                        "pair_rank": pair_rank,
                        "pair": pair_name,
                        "sector": result.pair.sector,
                        "config_id": baseline_parameter.config_id,
                        "transaction_cost_bps": spec.primary_transaction_cost_bps,
                        "baseline_used_for_selection": False,
                    },
                )
            )
            if primary_metric is not None:
                primary_metric.update(
                    {
                        "baseline_config_id": baseline_parameter.config_id,
                        "baseline_net_cagr": baseline.metrics["net_cagr"],
                        "baseline_total_return": baseline.metrics["net_total_return"],
                        "tuned_minus_baseline_cagr": primary_metric["net_cagr"]
                        - baseline.metrics["net_cagr"],
                        "tuned_minus_baseline_total_return": primary_metric[
                            "net_total_return"
                        ]
                        - baseline.metrics["net_total_return"],
                        "tuned_outperformed_baseline": bool(
                            primary_metric["net_total_return"]
                            > baseline.metrics["net_total_return"]
                        ),
                        "baseline_used_for_selection": False,
                    }
                )

    def combine(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    failures = pd.DataFrame(failure_rows)
    common_summary.update(
        {
            "tuned_pair_count": int(len(batch.results)),
            "tuning_failure_count": int(
                failures["stage"].eq("inner_tuning").sum()
            )
            if len(failures)
            else 0,
            "selected_pair_count": int(len(selected_pairs)),
            "oos_success_count": int(len(metric_rows)),
            "oos_failure_count": int(
                failures["stage"].astype(str).str.startswith("outer_evaluation").sum()
            )
            if len(failures)
            else 0,
            "baseline_success_count": int(len(baseline_metric_rows)),
            "baseline_failure_count": int(
                failures["stage"].eq("outer_baseline_2bps").sum()
            )
            if len(failures)
            else 0,
            "elapsed_seconds": float(time.monotonic() - started),
        }
    )
    artifacts = VintageArtifacts(
        parameter_rankings=combine(
            ranking_parts, ["rebalance_date", "pair", "config_id"]
        ),
        fold_evaluations=combine(
            fold_parts, ["rebalance_date", "pair", "inner_fold", "config_id"]
        ),
        model_audit=combine(
            model_parts, ["rebalance_date", "pair", "inner_fold", "alpha", "beta"]
        ),
        validation_returns=combine(
            validation_return_parts, ["rebalance_date", "pair", "date"]
        ),
        selected_validation_signals=combine(
            selected_validation_parts, ["rebalance_date", "pair", "date", "config_id"]
        ),
        validation_cost_sensitivity=combine(
            validation_cost_parts,
            ["rebalance_date", "pair", "config_id", "transaction_cost_bps"],
        ),
        selected_schedule=pd.DataFrame(schedule_rows),
        oos_signals=combine(
            oos_signal_parts, ["rebalance_date", "pair", "date", "config_id"]
        ),
        oos_trades=combine(
            oos_trade_parts, ["rebalance_date", "pair", "entry_date", "exit_date"]
        ),
        pair_vintage_metrics=pd.DataFrame(metric_rows),
        baseline_signals=combine(
            baseline_signal_parts,
            ["rebalance_date", "pair", "date", "config_id"],
        ),
        baseline_metrics=pd.DataFrame(baseline_metric_rows),
        cost_sensitivity=pd.DataFrame(cost_rows),
        failures=failures,
        vintage_summary=pd.DataFrame([common_summary]),
    )
    _write_vintage_cache(directory, artifacts, fingerprint=fingerprint)
    return artifacts, False


def _quality_checks(
    artifacts: VintageArtifacts,
    spec: PairTuningSpec,
    maximum_pairs: int,
) -> dict[str, Any]:
    schedule = artifacts.selected_schedule
    rankings = artifacts.parameter_rankings
    costs = artifacts.cost_sensitivity
    metrics = artifacts.pair_vintage_metrics
    baseline_metrics = artifacts.baseline_metrics
    signals = artifacts.oos_signals
    checks: dict[str, Any] = {
        "portfolio_objective_absent": bool(
            rankings.empty
            or (
                "portfolio_objective_used" in rankings
                and not bool(
                    _bool_series(
                        rankings, "portfolio_objective_used", True
                    ).any()
                )
            )
        ),
        "selected_pairs_all_strict": bool(
            schedule.empty
            or (
                _bool_series(schedule, "tradable_eligible", False).all()
                and schedule["selection_role"].astype(str).eq("strict_eligible").all()
                and pd.to_numeric(schedule["screen_fdr_qvalue"], errors="coerce")
                .le(0.05)
                .all()
            )
        ),
        "maximum_ten_per_vintage": bool(
            schedule.empty
            or schedule.groupby("rebalance_date")["pair"].nunique().le(maximum_pairs).all()
        ),
        "primary_signals_only_2bps": bool(
            signals.empty
            or pd.to_numeric(signals["transaction_cost_bps"], errors="coerce")
            .eq(spec.primary_transaction_cost_bps)
            .all()
        ),
        "terminal_liquidation_charged": bool(
            metrics.empty
            or (
                _bool_series(metrics, "terminal_liquidation", False).all()
                and _bool_series(metrics, "terminal_flat", False).all()
            )
        ),
        "cost_sensitivity_frozen": bool(
            costs.empty
            or (
                _bool_series(costs, "frozen_winner", False).all()
                and costs.groupby(["rebalance_date", "pair"])["config_id"]
                .nunique()
                .eq(1)
                .all()
            )
        ),
        "cost_sensitivity_has_exact_1_and_2bps": bool(
            costs.empty
            or costs.groupby(["rebalance_date", "pair"])[
                "transaction_cost_bps"
            ]
            .apply(
                lambda values: set(pd.to_numeric(values, errors="coerce"))
                == set(
                    [float(spec.primary_transaction_cost_bps)]
                    + [
                        float(value)
                        for value in spec.sensitivity_transaction_cost_bps
                    ]
                )
            )
            .all()
        ),
        "candidate_count_per_tuned_pair_is_97": bool(
            rankings.empty
            or rankings.groupby(["rebalance_date", "pair"])["config_id"]
            .nunique()
            .eq(97)
            .all()
        ),
        "baseline_never_used_for_selection": bool(
            baseline_metrics.empty
            or not bool(
                _bool_series(
                    baseline_metrics, "baseline_used_for_selection", True
                ).any()
            )
        ),
        "baseline_comparator_matches_successful_vintages": bool(
            metrics.empty
            or set(
                map(
                    tuple,
                    baseline_metrics[["rebalance_date", "pair"]]
                    .drop_duplicates()
                    .to_numpy(),
                )
            )
            == set(
                map(
                    tuple,
                    metrics[["rebalance_date", "pair"]]
                    .drop_duplicates()
                    .to_numpy(),
                )
            )
        ),
    }
    checks["critical_pass"] = bool(all(checks.values()))
    checks["selected_pair_vintages"] = int(len(schedule))
    checks["successful_oos_pair_vintages"] = int(len(metrics))
    return checks


def _run_tuning_stage(
    paths: StudyPaths,
    prices: pd.DataFrame,
    strict_all: pd.DataFrame,
    protocol: dict[str, Any],
    price_metadata: dict[str, Any],
    schedule: pd.DataFrame,
    *,
    recompute: bool,
    tuning_workers: int,
) -> VintageArtifacts:
    spec, base_config = _build_pair_tuning_protocol(protocol)
    vintages: list[VintageArtifacts] = []
    for location, row in enumerate(schedule.to_dict(orient="records"), start=1):
        date = pd.Timestamp(row["rebalance_date"])
        end = pd.Timestamp(row["outer_end_exclusive"])
        artifacts, reused = _run_one_tuning_vintage(
            paths,
            prices,
            strict_all,
            protocol,
            price_metadata,
            spec,
            base_config,
            date,
            end,
            recompute=recompute,
            tuning_workers=tuning_workers,
        )
        vintages.append(artifacts)
        summary = artifacts.vintage_summary.iloc[0]
        _log(
            f"tune {location}/{len(schedule)} {date.date()}: "
            f"strict={int(summary['strict_candidate_count'])}, "
            f"tuned={int(summary['tuned_pair_count'])}, "
            f"selected={int(summary['selected_pair_count'])}, "
            f"OOS={int(summary['oos_success_count'])}, "
            f"cache={'yes' if reused else 'no'}"
        )

    tables: dict[str, pd.DataFrame] = {}
    for field, filename in _GLOBAL_TABLE_FILES.items():
        pieces = [getattr(vintage, field) for vintage in vintages]
        pieces = [piece for piece in pieces if len(piece) or len(piece.columns)]
        table = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        tables[field] = table
        _atomic_to_csv(table, paths.processed / filename)
    combined = VintageArtifacts(**tables)
    maximum_pairs = int(protocol["pair_selection"]["maximum_pairs_per_rebalance"])
    checks = _quality_checks(combined, spec, maximum_pairs)
    _json_dump(paths.processed / "pair_v6_tuning_quality_checks.json", checks)
    _json_dump(
        paths.processed / "pair_v6_tuning_stage_metadata.json",
        {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rebalance_count": int(len(schedule)),
            "candidate_parameter_count": len(protocol_parameter_sets(spec)),
            "strict_candidate_rows": int(len(strict_all)),
            "selected_pair_vintages": int(len(combined.selected_schedule)),
            "successful_oos_pair_vintages": int(len(combined.pair_vintage_metrics)),
            "baseline_comparator_pair_vintages": int(len(combined.baseline_metrics)),
            "tuned_baseline_oos_hit_rate": float(
                _bool_series(
                    combined.pair_vintage_metrics,
                    "tuned_outperformed_baseline",
                    False,
                ).mean()
            )
            if len(combined.pair_vintage_metrics)
            and "tuned_outperformed_baseline" in combined.pair_vintage_metrics
            else math.nan,
            "portfolio_objective_used": False,
            "primary_transaction_cost_bps": spec.primary_transaction_cost_bps,
            "sensitivity_transaction_cost_bps": spec.sensitivity_transaction_cost_bps,
            "tuning_workers": int(tuning_workers),
            **_execution_cache_lineage(),
            "critical_quality_checks_pass": checks["critical_pass"],
            "global_files": {
                filename: _sha256(paths.processed / filename)
                for filename in _GLOBAL_TABLE_FILES.values()
            },
        },
    )
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--tuning-workers",
        type=int,
        default=4,
        help="processes used for independent pair tuning within each vintage",
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--recompute-screens", action="store_true")
    parser.add_argument("--recompute-tuning", action="store_true")
    parser.add_argument("--max-rebalances", type=int)
    parser.add_argument(
        "--stage",
        choices=("screens", "tuning", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.tuning_workers < 1:
        raise ValueError("tuning-workers must be positive")
    if args.max_rebalances is not None and args.max_rebalances < 1:
        raise ValueError("max-rebalances must be positive")
    protocol = _load_protocol(args.config)
    paths = StudyPaths.create(args.output)
    universe, prices, price_metadata = _prepare_data(
        paths,
        protocol,
        refresh=args.refresh_data,
    )
    as_of = pd.Timestamp(protocol["protocol"]["as_of"])
    first_boundary = pd.Timestamp(protocol["data"]["start_inclusive"]) + pd.DateOffset(
        years=3
    )
    all_rebalance_dates = _quarterly_rebalance_dates(
        prices,
        first_training_boundary=first_boundary,
        as_of=as_of,
    )
    schedule = _rebalance_schedule(
        all_rebalance_dates,
        as_of=as_of,
        maximum_rebalances=args.max_rebalances,
    )
    rebalance_dates = pd.DatetimeIndex(schedule["rebalance_date"])
    schedule.to_csv(paths.processed / "pair_v6_rebalance_schedule.csv", index=False)
    _log(
        f"10y snapshot {prices.index.min().date()}–{prices.index.max().date()}, "
        f"{len(prices):,} sessions, {len(universe)} current-member securities"
    )
    summary, strict = _run_screens(
        paths,
        prices,
        universe,
        protocol,
        price_metadata,
        rebalance_dates,
        workers=args.workers,
        recompute=args.recompute_screens,
    )
    if args.stage in {"tuning", "all"}:
        _log("Screen stage complete; starting strict pair-level tuning/evaluation")
        _run_tuning_stage(
            paths,
            prices,
            strict,
            protocol,
            price_metadata,
            schedule,
            recompute=args.recompute_tuning,
            tuning_workers=args.tuning_workers,
        )
    _json_dump(
        paths.processed / "pair_v6_screen_stage_metadata.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "price_snapshot_sha256": price_metadata["sha256"],
            "universe_sha256": _sha256(paths.raw / "sp500_current_constituents.csv"),
            "rebalance_count": int(len(rebalance_dates)),
            "strict_candidate_rows": int(len(strict)),
            "strict_nonempty_rebalances": int(summary["strict_eligible"].gt(0).sum()),
            "stage": args.stage,
            "tuning_completed": args.stage in {"tuning", "all"},
        },
    )


if __name__ == "__main__":
    main()
