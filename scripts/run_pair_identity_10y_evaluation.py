#!/usr/bin/env python3
"""Run the additive, fixed-identity ten-year V6 evaluation extension.

The immutable trading implementation is imported from the existing V6 runner.
This file changes only the evaluation panel: the 114 identities that completed
the strict V6 tuner at least once are replayed in every clipped calendar-quarter
interval from 2016-08-26 through 2026-08-26.  Formation diagnostics never gate
this analytical replay.  A pair is evaluated whenever its three-year formation
history supplies an orientation and the unchanged V6 tuner succeeds.

The first and last evaluation intervals are clipped partial quarters.  There
are therefore 41 adjacent, non-overlapping intervals: 2016-08-26--2016-10-03,
the intervening calendar quarters, and 2026-07-01--2026-08-27.

This runner is intentionally staged and resumable. Existing predictive-
diagnostic replays may seed only the exact pair-quarter rows they already
contain; every other pair-quarter is independently tuned and evaluated. No
existing V6 artifact or strategy module is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.data import (
    download_adjusted_close,
    load_timeseries_snapshot,
    save_timeseries_snapshot,
    snapshot_metadata_path,
)
from equity_pairs.pair_tuning import (
    PairDefinition,
    PairTuningSpec,
    evaluate_tuned_pair,
    protocol_parameter_sets,
    tune_pair_parameters,
)


DEFAULT_CONFIG = Path("configs/pair_identity_10y_evaluation.json")
DEFAULT_OUTPUT = Path("outputs/pair_identity_10y_evaluation_2026-08-26")
PRIMARY_COST_BPS = 2.0
SENSITIVITY_COST_BPS = 1.0
EXECUTION_CACHE_SCHEMA_VERSION = 3
EXECUTION_ENGINE_SOURCES = {
    "scripts/run_pair_identity_10y_evaluation.py": Path(__file__).resolve(),
    "src/equity_pairs/pair_tuning.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "pair_tuning.py",
    "src/equity_pairs/grid_search.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "grid_search.py",
    "src/equity_pairs/backtest.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "backtest.py",
    "src/equity_pairs/config.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "config.py",
    "src/equity_pairs/statistics.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "statistics.py",
    "src/equity_pairs/validation.py": PROJECT_ROOT
    / "src"
    / "equity_pairs"
    / "validation.py",
}
NEXT_CLOSE_EXECUTION_CONVENTION = {
    "signal_time": "Close[t]",
    "fill_time": "Close[t+1]",
    "first_pnl_interval": "Close[t+1] -> Close[t+2]",
}

PAIR_MASTER_NAME = "pair_identity_master.csv"
SCHEDULE_NAME = "pair_identity_rebalance_schedule.csv"
SCREEN_NAME = "pair_identity_screen_diagnostics.csv.gz"
QUARTERLY_NAME = "pair_identity_quarterly_results.csv.gz"
REPLAY_DAILY_NAME = "pair_identity_replay_daily.csv.gz"
TRADES_NAME = "pair_identity_trades.csv"
PARAMETER_NAME = "pair_identity_parameter_history.csv"
FAILURES_NAME = "pair_identity_tuning_failures.csv"
AVAILABILITY_NAME = "pair_identity_availability.csv.gz"
AUDIT_NAME = "pair_identity_replay_audit.json"


@dataclass(frozen=True)
class EvaluationPaths:
    root: Path
    raw: Path
    processed: Path
    screens: Path
    pair_cache: Path

    @classmethod
    def create(cls, root: str | Path) -> "EvaluationPaths":
        resolved = Path(root).resolve()
        value = cls(
            root=resolved,
            raw=resolved / "data" / "raw",
            processed=resolved / "data" / "processed",
            screens=resolved / "data" / "processed" / "screens",
            pair_cache=resolved / "data" / "processed" / "pair_quarter_cache",
        )
        for directory in (
            value.root,
            value.raw,
            value.processed,
            value.screens,
            value.pair_cache,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXECUTION_ENGINE_SOURCE_SHA256 = {
    name: _sha256(path) for name, path in sorted(EXECUTION_ENGINE_SOURCES.items())
}
PAIR_TUNING_SOURCE_SHA256 = EXECUTION_ENGINE_SOURCE_SHA256[
    "src/equity_pairs/pair_tuning.py"
]


def _execution_cache_lineage() -> dict[str, Any]:
    engine_hashes = dict(EXECUTION_ENGINE_SOURCE_SHA256)
    # Keep the historical scalar field synchronized for readers that display it,
    # while schema v3 validates the complete dependency manifest below.
    engine_hashes["src/equity_pairs/pair_tuning.py"] = PAIR_TUNING_SOURCE_SHA256
    return {
        "execution_convention": dict(NEXT_CLOSE_EXECUTION_CONVENTION),
        "pair_tuning_source_sha256": PAIR_TUNING_SOURCE_SHA256,
        "execution_engine_source_sha256": engine_hashes,
    }


def _cache_lineage_matches(metadata: Mapping[str, Any]) -> bool:
    expected = _execution_cache_lineage()
    return bool(
        metadata.get("schema_version") == EXECUTION_CACHE_SCHEMA_VERSION
        and all(metadata.get(key) == value for key, value in expected.items())
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_csv(path: Path, *, dates: Iterable[str] = ()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], errors="coerce", format="mixed"
            )
    return frame


def _bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _safe_token(value: Any) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in str(value).lower()
    ).strip("_") or "unknown"


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    required = {
        "protocol",
        "immutable_sources",
        "immutable_hashes",
        "data",
        "evaluation",
        "bootstrap",
        "ranking",
        "outputs",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("identity evaluation config is missing: " + ", ".join(missing))
    evaluation = payload["evaluation"]
    invariants = {
        "screen_eligibility_is_diagnostic_not_gating": True,
        "cross_pair_top10_cutoff_applied": False,
        "universe_is_ex_post_descriptive": True,
        "portfolio_metrics_enabled": False,
    }
    for key, expected in invariants.items():
        if evaluation.get(key) is not expected:
            raise ValueError(f"evaluation invariant {key} must be {expected}")
    if not payload["outputs"].get("overwrite_v6") is False:
        raise ValueError("the additive evaluation must never overwrite V6")
    return payload


def _source_paths(config: Mapping[str, Any], workspace: Path) -> dict[str, Path]:
    source = config["immutable_sources"]
    v6_root = _resolve(workspace, source["v6_output_root"])
    predictive_root = _resolve(workspace, source["predictive_replay_root"])
    return {
        "v6_config": _resolve(workspace, source["v6_config"]),
        "v6_runner": _resolve(workspace, source["v6_runner"]),
        "v6_root": v6_root,
        "v6_universe": _resolve(workspace, source["universe_snapshot"]),
        "v6_prices": _resolve(workspace, source["frozen_v6_price_snapshot"]),
        "v6_rankings": _resolve(workspace, source["v6_parameter_rankings"]),
        "v6_strict": _resolve(workspace, source["v6_strict_candidates"]),
        "v6_selected": _resolve(workspace, source["v6_selected_schedule"]),
        "v6_screens": _resolve(workspace, source["v6_screens_directory"]),
        "predictive_root": predictive_root,
        "predictive_manifest": predictive_root / "predictive_diagnostics_manifest.json",
        "predictive_metrics": _resolve(workspace, source["predictive_metrics"]),
        "predictive_costs": _resolve(workspace, source["predictive_cost_sensitivity"]),
        "predictive_signals": _resolve(workspace, source["predictive_signals"]),
        "predictive_trades": _resolve(workspace, source["predictive_trades"]),
    }


def _validate_immutable_protocol(
    config: Mapping[str, Any],
    sources: Mapping[str, Path],
    output_root: Path,
    v6: Any,
) -> tuple[dict[str, Any], PairTuningSpec, Any, dict[str, str]]:
    required = ("v6_config", "v6_runner", "v6_universe", "v6_prices")
    missing = [name for name in required if not sources[name].is_file()]
    if missing:
        raise FileNotFoundError("missing immutable V6 sources: " + ", ".join(missing))
    v6_root = sources["v6_root"].resolve()
    if output_root == v6_root or v6_root in output_root.parents:
        raise ValueError("additive output must not be inside the immutable V6 root")
    actual_hash = _sha256(sources["v6_config"])
    expected_hash = str(config["immutable_hashes"]["v6_config_sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(
            "immutable V6 config hash changed: "
            f"expected {expected_hash}, received {actual_hash}"
        )
    protocol = v6._load_protocol(sources["v6_config"])
    spec, base_config = v6._build_pair_tuning_protocol(protocol)
    if len(protocol_parameter_sets(spec, stop_z=base_config.stop_z)) != 97:
        raise RuntimeError("the imported V6 protocol no longer has 97 candidates")
    assertions = {
        "training_years": protocol["rolling_windows"]["training_years"] == 3,
        "strict_only": protocol["screening"]["strict_only"] is True,
        "bh_fdr": protocol["screening"]["multiple_testing_method"]
        == "Benjamini-Hochberg within sector",
        "primary_cost": math.isclose(
            float(protocol["costs"]["primary_transaction_cost_bps"]),
            PRIMARY_COST_BPS,
        ),
        "sensitivity": set(
            map(float, protocol["costs"]["transaction_cost_sensitivity_bps"])
        )
        == {SENSITIVITY_COST_BPS, PRIMARY_COST_BPS},
        "borrow": math.isclose(
            float(protocol["costs"]["annual_short_borrow_bps"]), 30.0
        ),
        "terminal": protocol["rolling_windows"]["force_terminal_liquidation"]
        is True,
        "next_session": protocol["evaluation"]["signal_at_close_applies_next_session"]
        is True,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise RuntimeError("immutable V6 contract failed: " + ", ".join(failed))
    hashes = {
        "v6_config": actual_hash,
        "v6_runner": _sha256(sources["v6_runner"]),
        "v6_universe": _sha256(sources["v6_universe"]),
        "v6_prices": _sha256(sources["v6_prices"]),
    }
    configured_hashes = config["immutable_hashes"]
    configured_by_source = {
        "v6_runner": "v6_runner_sha256",
        "v6_universe": "universe_snapshot_sha256",
        "v6_prices": "frozen_v6_price_snapshot_sha256",
    }
    for source_name, hash_name in configured_by_source.items():
        expected = configured_hashes.get(hash_name)
        if expected is not None and hashes[source_name] != str(expected):
            raise RuntimeError(
                f"immutable {source_name} hash changed: expected {expected}, "
                f"received {hashes[source_name]}"
            )
    return protocol, spec, base_config, hashes


def _post_freeze_source_hashes(sources: Mapping[str, Path]) -> dict[str, str]:
    names = (
        "v6_rankings",
        "v6_strict",
        "v6_selected",
        "predictive_metrics",
        "predictive_costs",
        "predictive_signals",
        "predictive_trades",
    )
    missing = [name for name in names if not sources[name].is_file()]
    if missing:
        raise FileNotFoundError("missing replay seed sources: " + ", ".join(missing))
    return {name: _sha256(sources[name]) for name in names}


def _validate_configured_post_freeze_hashes(
    config: Mapping[str, Any], actual: Mapping[str, str]
) -> None:
    expected = config["immutable_hashes"]
    mapping = {
        "v6_rankings": "v6_parameter_rankings_sha256",
        "v6_selected": "v6_selected_schedule_sha256",
        "predictive_metrics": "predictive_metrics_sha256",
        "predictive_signals": "predictive_signals_sha256",
    }
    for source_name, hash_name in mapping.items():
        configured = expected.get(hash_name)
        if configured is not None and actual[source_name] != str(configured):
            raise RuntimeError(
                f"immutable {source_name} hash changed: expected {configured}, "
                f"received {actual[source_name]}"
            )


def _price_paths(paths: EvaluationPaths, config: Mapping[str, Any]) -> tuple[Path, Path]:
    data = config["data"]
    return (
        paths.raw / str(data["download_snapshot_name"]),
        paths.raw / str(data["spliced_snapshot_name"]),
    )


def prepare_prices(
    paths: EvaluationPaths,
    config: Mapping[str, Any],
    sources: Mapping[str, Path],
    *,
    refresh_data: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a 13-year file whose frozen V6 segment wins on overlapping dates."""

    universe = pd.read_csv(sources["v6_universe"], dtype={"cik": "string"})
    if universe["ticker"].duplicated().any():
        raise ValueError("the immutable V6 universe contains duplicate tickers")
    copied_universe = paths.raw / "sp500_current_constituents.csv"
    if not copied_universe.exists():
        shutil.copy2(sources["v6_universe"], copied_universe)
    tickers = universe["ticker"].astype(str).tolist()
    download_path, spliced_path = _price_paths(paths, config)
    data = config["data"]
    downloaded = download_adjusted_close(
        tickers,
        data["download_start_inclusive"],
        data["download_end_exclusive"],
        chunk_size=int(data["download_chunk_size"]),
        max_retries=int(data["download_max_retries"]),
        backoff_factor=float(data["download_backoff_factor"]),
        inter_chunk_delay=float(data["download_inter_chunk_delay_seconds"]),
        repair=bool(data["repair_prices"]),
        strict=False,
        cache_path=download_path,
        refresh=refresh_data,
    ).sort_index()
    downloaded, download_meta = load_timeseries_snapshot(download_path)
    frozen, frozen_meta = load_timeseries_snapshot(sources["v6_prices"])
    downloaded = downloaded.reindex(columns=frozen.columns)
    common_index = downloaded.index.intersection(frozen.index)
    common_columns = downloaded.columns.intersection(frozen.columns)
    left = downloaded.loc[common_index, common_columns]
    right = frozen.loc[common_index, common_columns]
    absolute = (left - right).abs()
    relative = absolute / right.abs().replace(0.0, np.nan)
    finite_relative = relative.to_numpy(dtype=float)
    finite_relative = finite_relative[np.isfinite(finite_relative)]
    overlap_audit = {
        "overlap_sessions": int(len(common_index)),
        "overlap_columns": int(len(common_columns)),
        "max_absolute_difference": float(np.nanmax(absolute.to_numpy(dtype=float)))
        if absolute.size
        else math.nan,
        "max_relative_difference": float(np.max(finite_relative))
        if finite_relative.size
        else math.nan,
        "p99_relative_difference": float(np.quantile(finite_relative, 0.99))
        if finite_relative.size
        else math.nan,
    }
    frozen_start = pd.Timestamp(frozen.index.min())
    prefix = downloaded.loc[downloaded.index < frozen_start].copy()
    # Adjusted-close vendors can revise the entire historical level after a
    # later corporate action.  Apply one constant per ticker to the downloaded
    # prefix so its level joins the immutable V6 snapshot at the seam.  A
    # constant scale changes neither returns nor cointegration residual shape.
    scale_rows: list[dict[str, Any]] = []
    scale_values: dict[str, float] = {}
    overlap_dates = downloaded.index.intersection(frozen.index)
    for ticker in frozen.columns:
        scale = 1.0
        anchor = pd.NaT
        method = "no_common_finite_seam_value_scale_one"
        frozen_value = math.nan
        downloaded_value = math.nan
        for date in overlap_dates:
            candidate_frozen = frozen.at[date, ticker]
            candidate_downloaded = downloaded.at[date, ticker]
            if (
                pd.notna(candidate_frozen)
                and pd.notna(candidate_downloaded)
                and float(candidate_frozen) > 0
                and float(candidate_downloaded) > 0
            ):
                frozen_value = float(candidate_frozen)
                downloaded_value = float(candidate_downloaded)
                scale = frozen_value / downloaded_value
                anchor = pd.Timestamp(date)
                method = (
                    "exact_frozen_start_anchor"
                    if pd.Timestamp(date) == frozen_start
                    else "first_common_finite_overlap_fallback"
                )
                break
        scale_values[str(ticker)] = float(scale)
        scale_rows.append(
            {
                "ticker": str(ticker),
                "scale_factor": float(scale),
                "anchor_date": anchor,
                "anchor_method": method,
                "downloaded_anchor_price": downloaded_value,
                "frozen_anchor_price": frozen_value,
                "prefix_has_data": bool(prefix[ticker].notna().any()),
            }
        )
    if len(prefix):
        prefix = prefix.mul(pd.Series(scale_values), axis="columns")
    scale_diagnostics = pd.DataFrame(scale_rows)
    _atomic_csv(
        scale_diagnostics,
        paths.processed / "price_splice_scale_diagnostics.csv",
    )
    combined = pd.concat([prefix, frozen], axis=0).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(columns=frozen.columns)
    requested_start = pd.Timestamp(data["download_start_inclusive"])
    end = pd.Timestamp(data["download_end_exclusive"])
    combined = combined.loc[(combined.index >= requested_start) & (combined.index < end)]
    if combined.index.min() > requested_start + pd.Timedelta(days=7):
        raise RuntimeError("the spliced snapshot lacks required prehistory")
    metadata = save_timeseries_snapshot(
        combined,
        spliced_path,
        metadata={
            "source": "yahoo_finance_spliced_to_immutable_v6",
            "adjusted_close": True,
            "price_repair": bool(data["repair_prices"]),
            "start": str(data["download_start_inclusive"]),
            "end_exclusive": str(data["download_end_exclusive"]),
            "tickers": list(map(str, combined.columns)),
            "failed_tickers": [
                ticker for ticker in combined if not combined[ticker].notna().any()
            ],
            "download_snapshot_sha256": str(download_meta["sha256"]),
            "frozen_v6_snapshot_sha256": str(frozen_meta["sha256"]),
            "splice_rule": "downloaded dates before frozen V6 start; immutable V6 snapshot on and after its start",
            "prefix_scale_rule": "per-ticker frozen/downloaded adjusted-close ratio at 2016-08-26; first common finite overlap fallback; scale one only when no common finite seam exists",
            "overlap_audit": overlap_audit,
            "scale_factor_min": float(scale_diagnostics["scale_factor"].min()),
            "scale_factor_max": float(scale_diagnostics["scale_factor"].max()),
            "scale_fallback_count": int(
                scale_diagnostics["anchor_method"].ne("exact_frozen_start_anchor").sum()
            ),
        },
    )
    verified, verified_meta = load_timeseries_snapshot(spliced_path)
    if not verified.index.equals(combined.index) or list(verified.columns) != list(combined.columns):
        raise RuntimeError("saved spliced snapshot did not round-trip")
    frozen_segment_exact = bool(
        verified.loc[frozen.index, frozen.columns].equals(frozen)
    )
    if not frozen_segment_exact:
        raise RuntimeError("spliced snapshot changed one or more immutable V6 values")
    _atomic_json(
        paths.processed / "price_splice_audit.json",
        {
            **overlap_audit,
            "created_at_utc": _utc_now(),
            "download_snapshot": str(download_path),
            "spliced_snapshot": str(spliced_path),
            "spliced_sha256": verified_meta["sha256"],
            "frozen_segment_start": str(frozen_start.date()),
            "post_splice_frozen_segment_exact": frozen_segment_exact,
            "prefix_scale_diagnostics": str(
                paths.processed / "price_splice_scale_diagnostics.csv"
            ),
            "scale_factor_min": float(scale_diagnostics["scale_factor"].min()),
            "scale_factor_max": float(scale_diagnostics["scale_factor"].max()),
            "scale_fallback_count": int(
                scale_diagnostics["anchor_method"].ne("exact_frozen_start_anchor").sum()
            ),
        },
    )
    return verified, metadata


def load_prepared_prices(
    paths: EvaluationPaths, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, spliced_path = _price_paths(paths, config)
    return load_timeseries_snapshot(spliced_path)


def build_clipped_quarter_schedule(
    market_index: pd.DatetimeIndex,
    start_inclusive: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return the initial stub, calendar quarters, and final truncated quarter."""

    index = pd.DatetimeIndex(market_index).sort_values().unique()
    start = pd.Timestamp(start_inclusive)
    end = pd.Timestamp(end_exclusive)
    sessions = index[(index >= start) & (index < end)]
    if sessions.empty:
        raise ValueError("no market sessions fall in the requested study interval")
    first = pd.Timestamp(sessions[0])
    anchors = pd.date_range(first.to_period("Q").start_time, end, freq="QS")
    boundaries: list[pd.Timestamp] = [first]
    for anchor in anchors:
        if anchor <= first or anchor >= end:
            continue
        candidate = index[(index >= anchor) & (index < end)]
        if len(candidate):
            boundaries.append(pd.Timestamp(candidate[0]))
    boundaries = sorted(set(boundaries))
    rows: list[dict[str, Any]] = []
    for number, boundary in enumerate(boundaries, start=1):
        outer_end = boundaries[number] if number < len(boundaries) else end
        outer = index[(index >= boundary) & (index < outer_end)]
        if not len(outer):
            continue
        rows.append(
            {
                "quarter_number": number,
                "rebalance_date": boundary,
                "outer_end_exclusive": outer_end,
                "outer_sessions": int(len(outer)),
                "formation_start_requested": boundary - pd.DateOffset(years=3),
                "initial_clipped_quarter": bool(number == 1),
                "final_clipped_quarter": bool(outer_end == end),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) > 1 and not all(
        pd.Timestamp(result.iloc[i - 1]["outer_end_exclusive"])
        == pd.Timestamp(result.iloc[i]["rebalance_date"])
        for i in range(1, len(result))
    ):
        raise RuntimeError("clipped outer intervals are not adjacent")
    return result


def _write_schedule(
    paths: EvaluationPaths,
    config: Mapping[str, Any],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    schedule = build_clipped_quarter_schedule(
        prices.index,
        config["evaluation"]["study_start_inclusive"],
        config["evaluation"]["study_end_exclusive"],
    )
    expected = int(config["evaluation"]["expected_quarter_intervals"])
    if len(schedule) != expected:
        raise RuntimeError(f"expected {expected} clipped quarters, received {len(schedule)}")
    _atomic_csv(schedule, paths.processed / SCHEDULE_NAME)
    return schedule


def _screen_cache_paths(paths: EvaluationPaths, date: pd.Timestamp) -> tuple[Path, Path, Path]:
    token = pd.Timestamp(date).strftime("%Y%m%d")
    return (
        paths.screens / f"pair_v6_screen_{token}.csv.gz",
        paths.screens / f"pair_v6_integration_{token}.csv",
        paths.screens / f"pair_v6_screen_{token}.meta.json",
    )


def _source_screen_path(sources: Mapping[str, Path], date: pd.Timestamp) -> Path:
    return sources["v6_screens"] / f"pair_v6_screen_{pd.Timestamp(date):%Y%m%d}.csv.gz"


def _screen_for_date(
    *,
    paths: EvaluationPaths,
    sources: Mapping[str, Path],
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    protocol: Mapping[str, Any],
    price_metadata: Mapping[str, Any],
    rebalance_date: pd.Timestamp,
    v6: Any,
    screen_workers: int,
    recompute: bool,
) -> tuple[pd.DataFrame, str, bool]:
    """Reuse a frozen V6 screen when present; otherwise run the canonical helper."""

    source = _source_screen_path(sources, rebalance_date)
    if source.is_file():
        diagnostics = pd.read_csv(source)
        return diagnostics, "immutable_v6_screen", True
    diagnostics, _, _, reused = v6._screen_one_window(
        v6.StudyPaths.create(paths.root),
        prices,
        universe,
        dict(protocol),
        dict(price_metadata),
        pd.Timestamp(rebalance_date),
        workers=screen_workers,
        recompute=recompute,
    )
    return diagnostics, "additive_canonical_screen", bool(reused)


def _annotate_screen(
    diagnostics: pd.DataFrame,
    schedule_row: Mapping[str, Any],
    source: str,
    cache_reused: bool,
) -> pd.DataFrame:
    result = diagnostics.copy()
    result.insert(0, "rebalance_date", pd.Timestamp(schedule_row["rebalance_date"]))
    result.insert(1, "outer_end_exclusive", pd.Timestamp(schedule_row["outer_end_exclusive"]))
    result.insert(2, "formation_start_requested", pd.Timestamp(schedule_row["formation_start_requested"]))
    result["screen_source"] = source
    result["screen_cache_reused"] = bool(cache_reused)
    return result


def ensure_all_screens(
    paths: EvaluationPaths,
    config: Mapping[str, Any],
    sources: Mapping[str, Path],
    protocol: Mapping[str, Any],
    prices: pd.DataFrame,
    price_metadata: Mapping[str, Any],
    schedule: pd.DataFrame,
    v6: Any,
    *,
    screen_workers: int,
    recompute: bool,
) -> pd.DataFrame:
    universe = pd.read_csv(sources["v6_universe"], dtype={"cik": "string"})
    expected_pairs = int(config["evaluation"]["expected_candidate_pairs_per_screen"])
    parts: list[pd.DataFrame] = []
    for number, row in enumerate(schedule.to_dict(orient="records"), start=1):
        diagnostics, source, reused = _screen_for_date(
            paths=paths,
            sources=sources,
            prices=prices,
            universe=universe,
            protocol=protocol,
            price_metadata=price_metadata,
            rebalance_date=pd.Timestamp(row["rebalance_date"]),
            v6=v6,
            screen_workers=screen_workers,
            recompute=recompute,
        )
        if len(diagnostics) != expected_pairs:
            raise RuntimeError(
                f"screen {pd.Timestamp(row['rebalance_date']).date()} has "
                f"{len(diagnostics)} rows; expected {expected_pairs}"
            )
        parts.append(_annotate_screen(diagnostics, row, source, reused))
        print(
            f"screen {number}/{len(schedule)} · {pd.Timestamp(row['rebalance_date']).date()} "
            f"· {source} · strict={int(_bool(diagnostics['eligible']).sum())}",
            flush=True,
        )
    combined = pd.concat(parts, ignore_index=True)
    if combined.duplicated(["rebalance_date", "pair"]).any():
        raise RuntimeError("screen diagnostics contain duplicate pair-quarter keys")
    _atomic_csv(combined, paths.processed / SCREEN_NAME)
    return combined


def _selected_training_row(result: Any) -> pd.Series:
    selected = result.ranking.loc[_bool(result.ranking["selected"])]
    if len(selected) != 1:
        raise RuntimeError("a tuned pair must contain exactly one selected parameter")
    return selected.iloc[0]


def build_fixed_identity_master(
    paths: EvaluationPaths,
    config: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> pd.DataFrame:
    """Use exactly the identities represented by successful V6 tunings."""

    rankings = _read_csv(sources["v6_rankings"], dates=("rebalance_date",))
    winners = rankings.loc[_bool(rankings["selected"])].copy()
    if winners.duplicated(["rebalance_date", "pair"]).any():
        raise RuntimeError("V6 selected parameter rows are not unique by pair-quarter")
    selected_schedule = _read_csv(
        sources["v6_selected"], dates=("rebalance_date",)
    )
    selected_pairs = set(selected_schedule["pair"].astype(str))
    rows: list[dict[str, Any]] = []
    for pair, group in winners.groupby("pair", sort=True):
        sectors = sorted(set(group["sector"].dropna().astype(str)))
        if len(sectors) != 1:
            raise RuntimeError(f"fixed identity {pair} has inconsistent sectors")
        parts = str(pair).split("__")
        if len(parts) != 2:
            raise ValueError(f"pair identity is not canonical: {pair}")
        selected_count = int(selected_schedule["pair"].astype(str).eq(str(pair)).sum())
        rows.append(
            {
                "pair": str(pair),
                "sector": sectors[0],
                "ticker_a": parts[0],
                "ticker_b": parts[1],
                "successful_v6_tuned_vintages": int(len(group)),
                "first_successful_v6_tuned_date": pd.to_datetime(group["rebalance_date"]).min(),
                "last_successful_v6_tuned_date": pd.to_datetime(group["rebalance_date"]).max(),
                "ever_v6_selected": str(pair) in selected_pairs,
                "v6_selected_vintage_count": selected_count,
                "fixed_identity_selection_is_ex_post": True,
                "identity_universe_rule": "unique identity among successfully strict-and-tuned V6 pair-vintages",
            }
        )
    master = pd.DataFrame(rows).sort_values("pair", kind="mergesort").reset_index(drop=True)
    expected = int(config["evaluation"]["expected_fixed_identity_count"])
    selected_expected = int(config["evaluation"]["expected_v6_selected_identity_count"])
    if len(master) != expected:
        raise RuntimeError(f"expected {expected} fixed identities, received {len(master)}")
    if int(master["ever_v6_selected"].sum()) != selected_expected:
        raise RuntimeError(
            f"expected {selected_expected} V6-selected identities, received "
            f"{int(master['ever_v6_selected'].sum())}"
        )
    _atomic_csv(master, paths.processed / PAIR_MASTER_NAME)
    return master


SUMMARY_COLUMNS = [
    "pair",
    "sector",
    "ticker_a",
    "ticker_b",
    "rebalance_date",
    "outer_end_exclusive",
    "formation_start_requested",
    "formation_start_actual",
    "formation_end_actual",
    "formation_sessions",
    "evaluation_sessions",
    "screen_status",
    "screen_observations",
    "screen_history_ratio",
    "screen_pair_pvalue",
    "screen_fdr_qvalue",
    "screen_fdr_significant",
    "screen_both_look_i1",
    "screen_positive_hedge_ratio",
    "screen_half_life_days",
    "screen_hurst_exponent",
    "screen_core_health_pass",
    "screen_strict_eligible",
    "screen_dependent",
    "screen_independent",
    "screen_alpha",
    "screen_beta",
    "history_available",
    "orientation_available",
    "tuning_success",
    "active_identity_conditional",
    "status",
    "cash_reason",
    "selected_config_id",
    "zscore_method",
    "zscore_lookback",
    "zscore_min_periods",
    "entry_z",
    "exit_z",
    "stop_z",
    "maximum_holding_days",
    "is_baseline",
    "training_cv_net_cagr",
    "training_cv_max_drawdown",
    "validation_trade_count",
    "full_training_trade_count",
    "active_fold_count",
    "oos_refit_alpha",
    "oos_refit_beta",
    "q_return_1bp",
    "q_return_2bp",
    "gross_q_return",
    "trade_count",
    "transaction_cost_sum_1bp",
    "transaction_cost_sum_2bp",
    "borrow_cost_sum",
    "start_flat",
    "terminal_flat",
    "terminal_liquidation",
    "replay_source",
    "tuning_failure_reason",
    "error_type",
    "error",
]

DAILY_COLUMNS = [
    "pair",
    "sector",
    "rebalance_date",
    "outer_end_exclusive",
    "date",
    "config_id",
    "net_return_1bp",
    "net_return_2bp",
    "gross_return",
    "turnover",
    "borrow_cost",
    "transaction_cost_1bp",
    "transaction_cost_2bp",
    "zscore",
    "position",
    "target_position",
    "decision_event",
    "active_identity_conditional",
]

PARAMETER_COLUMNS = [
    "pair",
    "sector",
    "rebalance_date",
    "outer_end_exclusive",
    "selected_config_id",
    "zscore_method",
    "zscore_lookback",
    "zscore_min_periods",
    "entry_z",
    "exit_z",
    "stop_z",
    "maximum_holding_days",
    "is_baseline",
    "training_cv_net_cagr",
    "training_cv_max_drawdown",
    "validation_trade_count",
    "full_training_trade_count",
    "active_fold_count",
    "oos_refit_alpha",
    "oos_refit_beta",
    "screen_dependent",
    "screen_independent",
    "replay_source",
]

FAILURE_COLUMNS = [
    "pair",
    "sector",
    "rebalance_date",
    "stage",
    "status",
    "cash_reason",
    "error_type",
    "error",
    "history_available",
    "orientation_available",
]


def _screen_context(screen: Mapping[str, Any]) -> dict[str, Any]:
    def value(name: str, default: Any = math.nan) -> Any:
        result = screen.get(name, default)
        return default if pd.isna(result) else result

    status = str(value("screen_status", "missing"))
    both_i1 = bool(value("both_look_i1", False))
    positive_beta = bool(value("positive_hedge_ratio", False))
    half_life = value("half_life_days")
    hurst = value("hurst_exponent")
    core_health = bool(
        status == "ok"
        and both_i1
        and positive_beta
        and np.isfinite(float(half_life))
        and 2.0 <= float(half_life) <= 252.0
        and np.isfinite(float(hurst))
        and float(hurst) < 0.5
    )
    return {
        "screen_status": status,
        "screen_observations": value("observations"),
        "screen_history_ratio": value("history_ratio"),
        "screen_pair_pvalue": value("pair_pvalue"),
        "screen_fdr_qvalue": value("fdr_qvalue"),
        "screen_fdr_significant": bool(value("fdr_significant", False)),
        "screen_both_look_i1": both_i1,
        "screen_positive_hedge_ratio": positive_beta,
        "screen_half_life_days": half_life,
        "screen_hurst_exponent": hurst,
        "screen_core_health_pass": core_health,
        "screen_strict_eligible": bool(value("eligible", False)),
        "screen_dependent": value("dependent", ""),
        "screen_independent": value("independent", ""),
        "screen_alpha": value("alpha"),
        "screen_beta": value("beta"),
    }


def _common_summary(task: Mapping[str, Any], prices: pd.DataFrame) -> dict[str, Any]:
    rebalance = pd.Timestamp(task["rebalance_date"])
    outer_end = pd.Timestamp(task["outer_end_exclusive"])
    requested_start = rebalance - pd.DateOffset(years=3)
    formation = prices.index[(prices.index >= requested_start) & (prices.index < rebalance)]
    outer = prices.index[(prices.index >= rebalance) & (prices.index < outer_end)]
    screen = _screen_context(task)
    dependent = str(screen["screen_dependent"] or "")
    independent = str(screen["screen_independent"] or "")
    orientation_available = bool(dependent and independent and dependent != independent)
    history_available = bool(
        screen["screen_status"] == "ok"
        and orientation_available
        and len(formation) >= 500
    )
    return {
        "pair": str(task["pair"]),
        "sector": str(task["sector"]),
        "ticker_a": str(task["ticker_a"]),
        "ticker_b": str(task["ticker_b"]),
        "rebalance_date": rebalance,
        "outer_end_exclusive": outer_end,
        "formation_start_requested": requested_start,
        "formation_start_actual": formation.min() if len(formation) else pd.NaT,
        "formation_end_actual": formation.max() if len(formation) else pd.NaT,
        "formation_sessions": int(len(formation)),
        "evaluation_sessions": int(len(outer)),
        **screen,
        "history_available": history_available,
        "orientation_available": orientation_available,
    }


def _empty_summary(
    common: Mapping[str, Any],
    *,
    status: str,
    cash_reason: str,
    error_type: str = "",
    error: str = "",
    tuning_failure_reason: str = "",
) -> pd.DataFrame:
    row = {
        **common,
        "tuning_success": False,
        "active_identity_conditional": False,
        "status": status,
        "cash_reason": cash_reason,
        "selected_config_id": pd.NA,
        "zscore_method": pd.NA,
        "zscore_lookback": pd.NA,
        "zscore_min_periods": pd.NA,
        "entry_z": math.nan,
        "exit_z": math.nan,
        "stop_z": math.nan,
        "maximum_holding_days": pd.NA,
        "is_baseline": pd.NA,
        "training_cv_net_cagr": math.nan,
        "training_cv_max_drawdown": math.nan,
        "validation_trade_count": math.nan,
        "full_training_trade_count": math.nan,
        "active_fold_count": math.nan,
        "oos_refit_alpha": math.nan,
        "oos_refit_beta": math.nan,
        "q_return_1bp": 0.0,
        "q_return_2bp": 0.0,
        "gross_q_return": 0.0,
        "trade_count": 0,
        "transaction_cost_sum_1bp": 0.0,
        "transaction_cost_sum_2bp": 0.0,
        "borrow_cost_sum": 0.0,
        "start_flat": True,
        "terminal_flat": True,
        "terminal_liquidation": False,
        "replay_source": "new_pair_quarter_replay",
        "tuning_failure_reason": tuning_failure_reason,
        "error_type": error_type,
        "error": error,
    }
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)


def _pair_cache_paths(
    cache_root: Path, pair: str, rebalance_date: pd.Timestamp
) -> dict[str, Path]:
    directory = cache_root / _safe_token(pair) / pd.Timestamp(rebalance_date).strftime("%Y%m%d")
    return {
        "directory": directory,
        "summary": directory / "summary.csv",
        "daily": directory / "daily.csv.gz",
        "trades": directory / "trades.csv",
        "parameter": directory / "parameter.csv",
        "ranking": directory / "parameter_rankings.csv.gz",
        "failure": directory / "failure.csv",
        "complete": directory / "complete.json",
    }


_CACHE_ARTIFACT_NAMES = (
    "summary",
    "daily",
    "trades",
    "parameter",
    "ranking",
    "failure",
)


def _cache_artifact_hashes(cache: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: _sha256(cache[name])
        for name in _CACHE_ARTIFACT_NAMES
        if cache[name].is_file()
    }


def _cache_artifacts_match_metadata(
    cache: Mapping[str, Path], metadata: Mapping[str, Any]
) -> bool:
    declared = metadata.get("artifact_sha256")
    if not isinstance(declared, dict) or not declared:
        return False
    declared_names = set(declared)
    known_names = set(_CACHE_ARTIFACT_NAMES)
    if not declared_names.issubset(known_names):
        return False
    actual_names = {name for name in _CACHE_ARTIFACT_NAMES if cache[name].is_file()}
    if declared_names != actual_names:
        return False
    try:
        return all(
            isinstance(declared[name], str)
            and declared[name] == _sha256(cache[name])
            for name in declared_names
        )
    except OSError:
        return False


def _task_fingerprint(
    task: Mapping[str, Any], price_sha256: str, protocol_sha256: str
) -> str:
    fields = {
        "schema_version": EXECUTION_CACHE_SCHEMA_VERSION,
        "pair": str(task["pair"]),
        "sector": str(task["sector"]),
        "rebalance_date": str(pd.Timestamp(task["rebalance_date"])),
        "outer_end_exclusive": str(pd.Timestamp(task["outer_end_exclusive"])),
        "screen_status": str(task.get("screen_status")),
        "dependent": str(task.get("dependent")),
        "independent": str(task.get("independent")),
        "screen_row_sha256": _dataframe_sha256(pd.DataFrame([dict(task)])),
        "price_sha256": price_sha256,
        "protocol_sha256": protocol_sha256,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "sensitivity_cost_bps": SENSITIVITY_COST_BPS,
        **_execution_cache_lineage(),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def _cache_valid(paths: Mapping[str, Path], fingerprint: str) -> bool:
    if not paths["complete"].is_file() or not paths["summary"].is_file():
        return False
    try:
        metadata = _load_json(paths["complete"])
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("fingerprint") == fingerprint
        and _cache_lineage_matches(metadata)
        and _cache_artifacts_match_metadata(paths, metadata)
    )


def _clear_cache_artifacts(cache: Mapping[str, Path]) -> None:
    """Remove only the known artifacts for one exact pair-quarter cache."""
    for name, path in cache.items():
        if name == "directory":
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()


def _write_cache(
    cache: Mapping[str, Path],
    fingerprint: str,
    summary: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    parameter: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
    failure: pd.DataFrame | None = None,
) -> None:
    cache["directory"].mkdir(parents=True, exist_ok=True)
    # The completion marker and optional success artifacts must disappear
    # before a replacement write.  Otherwise a failed recomputation can leave
    # stale daily/trade tables beside a new failure summary.
    _clear_cache_artifacts(cache)
    _atomic_csv(summary, cache["summary"])
    if daily is not None:
        _atomic_csv(daily, cache["daily"])
    if trades is not None:
        _atomic_csv(trades, cache["trades"])
    if parameter is not None:
        _atomic_csv(parameter, cache["parameter"])
    if ranking is not None:
        _atomic_csv(ranking, cache["ranking"])
    if failure is not None:
        _atomic_csv(failure, cache["failure"])
    artifact_sha256 = _cache_artifact_hashes(cache)
    _atomic_json(
        cache["complete"],
        {
            "schema_version": EXECUTION_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "completed_at_utc": _utc_now(),
            **_execution_cache_lineage(),
            "artifact_sha256": artifact_sha256,
            "status": str(summary.iloc[0]["status"]),
            "daily_rows": int(len(daily)) if daily is not None else 0,
            "trade_rows": int(len(trades)) if trades is not None else 0,
        },
    )


_WORKER_STATE: dict[str, Any] = {}


def _initialize_pair_worker(
    prices: pd.DataFrame,
    spec: PairTuningSpec,
    base_config: Any,
    cache_root: str,
    price_sha256: str,
    protocol_sha256: str,
    recompute: bool,
) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "prices": prices,
            "spec": spec,
            "base_config": base_config,
            "cache_root": Path(cache_root),
            "price_sha256": price_sha256,
            "protocol_sha256": protocol_sha256,
            "recompute": recompute,
        }
    )


def _failure_row(common: Mapping[str, Any], *, stage: str, status: str, cash_reason: str, error: Exception | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": common["pair"],
                "sector": common["sector"],
                "rebalance_date": common["rebalance_date"],
                "stage": stage,
                "status": status,
                "cash_reason": cash_reason,
                "error_type": type(error).__name__ if error is not None else "",
                "error": str(error) if error is not None else "",
                "history_available": common["history_available"],
                "orientation_available": common["orientation_available"],
            }
        ],
        columns=FAILURE_COLUMNS,
    )


def _run_pair_quarter_task(task: Mapping[str, Any]) -> tuple[str, str, bool]:
    prices: pd.DataFrame = _WORKER_STATE["prices"]
    spec: PairTuningSpec = _WORKER_STATE["spec"]
    base_config = _WORKER_STATE["base_config"]
    pair = str(task["pair"])
    rebalance = pd.Timestamp(task["rebalance_date"])
    cache = _pair_cache_paths(_WORKER_STATE["cache_root"], pair, rebalance)
    fingerprint = _task_fingerprint(
        task,
        _WORKER_STATE["price_sha256"],
        _WORKER_STATE["protocol_sha256"],
    )
    if not _WORKER_STATE["recompute"] and _cache_valid(cache, fingerprint):
        return pair, str(rebalance.date()), True
    common = _common_summary(task, prices)
    if not common["history_available"] or not common["orientation_available"]:
        if str(common["screen_status"]) == "insufficient_history":
            reason = "insufficient_three_year_history"
        elif not common["orientation_available"]:
            reason = "screen_orientation_unavailable"
        else:
            reason = "screen_diagnostic_unavailable"
        summary = _empty_summary(common, status="unavailable_cash", cash_reason=reason)
        failure = _failure_row(
            common,
            stage="availability",
            status="unavailable_cash",
            cash_reason=reason,
        )
        _write_cache(cache, fingerprint, summary, failure=failure)
        return pair, str(rebalance.date()), False

    formation = prices.index[
        (prices.index >= pd.Timestamp(common["formation_start_requested"]))
        & (prices.index < rebalance)
    ]
    outer = prices.index[
        (prices.index >= rebalance)
        & (prices.index < pd.Timestamp(common["outer_end_exclusive"]))
    ]
    definition = PairDefinition(
        pair=pair,
        sector=str(common["sector"]),
        dependent=str(common["screen_dependent"]),
        independent=str(common["screen_independent"]),
    )
    pair_prices = prices[[definition.dependent, definition.independent]].copy()
    try:
        tuning = tune_pair_parameters(
            definition,
            pair_prices,
            formation,
            base_config,
            spec,
        )
    except Exception as error:
        summary = _empty_summary(
            common,
            status="tuning_failed_cash",
            cash_reason="unchanged_v6_tuner_failed",
            error_type=type(error).__name__,
            error=str(error),
            tuning_failure_reason=f"{type(error).__name__}: {error}",
        )
        failure = _failure_row(
            common,
            stage="pair_tuning",
            status="tuning_failed_cash",
            cash_reason="unchanged_v6_tuner_failed",
            error=error,
        )
        _write_cache(cache, fingerprint, summary, failure=failure)
        return pair, str(rebalance.date()), False

    training = _selected_training_row(tuning)
    parameter = tuning.selected_parameter
    ranking = tuning.ranking.copy()
    if "pair" in ranking:
        ranking["pair"] = pair
    else:
        ranking.insert(0, "pair", pair)
    if "rebalance_date" in ranking:
        ranking["rebalance_date"] = rebalance
    else:
        ranking.insert(0, "rebalance_date", rebalance)
    try:
        primary = evaluate_tuned_pair(
            tuning,
            pair_prices,
            formation,
            outer,
            base_config,
            transaction_cost_bps=PRIMARY_COST_BPS,
        )
        sensitivity = evaluate_tuned_pair(
            tuning,
            pair_prices,
            formation,
            outer,
            base_config,
            transaction_cost_bps=SENSITIVITY_COST_BPS,
        )
    except Exception as error:
        summary = _empty_summary(
            common,
            status="evaluation_failed_cash",
            cash_reason="unchanged_v6_evaluator_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        summary.loc[:, "tuning_success"] = True
        summary.loc[:, "selected_config_id"] = parameter.config_id
        summary.loc[:, "zscore_method"] = parameter.zscore_method
        summary.loc[:, "zscore_lookback"] = int(parameter.zscore_lookback)
        summary.loc[:, "zscore_min_periods"] = int(parameter.zscore_min_periods)
        summary.loc[:, "entry_z"] = float(parameter.entry_z)
        summary.loc[:, "exit_z"] = float(parameter.exit_z)
        summary.loc[:, "stop_z"] = float(base_config.stop_z)
        summary.loc[:, "maximum_holding_days"] = int(parameter.maximum_holding_days)
        summary.loc[:, "is_baseline"] = bool(parameter.is_baseline)
        summary.loc[:, "training_cv_net_cagr"] = float(training["pooled_net_cagr"])
        summary.loc[:, "training_cv_max_drawdown"] = float(training["max_drawdown"])
        summary.loc[:, "validation_trade_count"] = int(training["validation_trade_count"])
        summary.loc[:, "full_training_trade_count"] = int(training["full_training_trade_count"])
        summary.loc[:, "active_fold_count"] = int(training["active_fold_count"])
        failure = _failure_row(
            common,
            stage="outer_evaluation",
            status="evaluation_failed_cash",
            cash_reason="unchanged_v6_evaluator_failed",
            error=error,
        )
        parameter_row = summary.iloc[0].to_dict()
        parameter_frame = pd.DataFrame(
            [{key: parameter_row[key] for key in PARAMETER_COLUMNS}],
            columns=PARAMETER_COLUMNS,
        )
        _write_cache(
            cache,
            fingerprint,
            summary,
            parameter=parameter_frame,
            ranking=ranking,
            failure=failure,
        )
        return pair, str(rebalance.date()), False

    trade_count = int(primary.metrics["trade_count"])
    status = "evaluated_zero_trade" if trade_count == 0 else "evaluated_traded"
    row = {
        **common,
        "tuning_success": True,
        "active_identity_conditional": True,
        "status": status,
        "cash_reason": "",
        "selected_config_id": parameter.config_id,
        "zscore_method": parameter.zscore_method,
        "zscore_lookback": int(parameter.zscore_lookback),
        "zscore_min_periods": int(parameter.zscore_min_periods),
        "entry_z": float(parameter.entry_z),
        "exit_z": float(parameter.exit_z),
        "stop_z": float(base_config.stop_z),
        "maximum_holding_days": int(parameter.maximum_holding_days),
        "is_baseline": bool(parameter.is_baseline),
        "training_cv_net_cagr": float(training["pooled_net_cagr"]),
        "training_cv_max_drawdown": float(training["max_drawdown"]),
        "validation_trade_count": int(training["validation_trade_count"]),
        "full_training_trade_count": int(training["full_training_trade_count"]),
        "active_fold_count": int(training["active_fold_count"]),
        "oos_refit_alpha": float(primary.model.alpha),
        "oos_refit_beta": float(primary.model.beta),
        "q_return_1bp": float(sensitivity.metrics["net_total_return"]),
        "q_return_2bp": float(primary.metrics["net_total_return"]),
        "gross_q_return": float(primary.metrics["gross_total_return"]),
        "trade_count": trade_count,
        "transaction_cost_sum_1bp": float(sensitivity.metrics["transaction_cost_sum"]),
        "transaction_cost_sum_2bp": float(primary.metrics["transaction_cost_sum"]),
        "borrow_cost_sum": float(primary.metrics["borrow_cost_sum"]),
        "start_flat": bool(primary.metrics["start_flat"]),
        "terminal_flat": bool(primary.metrics["terminal_flat"]),
        "terminal_liquidation": bool(primary.metrics["terminal_liquidation"]),
        "replay_source": "new_pair_quarter_replay",
        "tuning_failure_reason": "",
        "error_type": "",
        "error": "",
    }
    summary = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    daily_primary = primary.daily.reindex(outer)
    daily_sensitivity = sensitivity.daily.reindex(outer)
    daily = pd.DataFrame(
        {
            "pair": pair,
            "sector": common["sector"],
            "rebalance_date": rebalance,
            "outer_end_exclusive": common["outer_end_exclusive"],
            "date": outer,
            "config_id": parameter.config_id,
            "net_return_1bp": daily_sensitivity["net_return"].fillna(0.0).to_numpy(),
            "net_return_2bp": daily_primary["net_return"].fillna(0.0).to_numpy(),
            "gross_return": daily_primary["gross_return"].fillna(0.0).to_numpy(),
            "turnover": daily_primary["turnover"].fillna(0.0).to_numpy(),
            "borrow_cost": daily_primary["borrow_cost"].fillna(0.0).to_numpy(),
            "transaction_cost_1bp": daily_sensitivity["transaction_cost"].fillna(0.0).to_numpy(),
            "transaction_cost_2bp": daily_primary["transaction_cost"].fillna(0.0).to_numpy(),
            "zscore": daily_primary["zscore"].to_numpy(),
            "position": daily_primary["position"].fillna(0).to_numpy(),
            "target_position": daily_primary["target_position"].fillna(0).to_numpy(),
            "decision_event": daily_primary["decision_event"].astype(str).to_numpy(),
            "active_identity_conditional": True,
        },
        columns=DAILY_COLUMNS,
    )
    trades = primary.trades.copy()
    if not trades.empty:
        trades.insert(0, "rebalance_date", rebalance)
        trades.insert(0, "pair", pair) if "pair" not in trades else None
        trades["outer_end_exclusive"] = common["outer_end_exclusive"]
        trades["config_id"] = parameter.config_id
        trades["transaction_cost_bps"] = PRIMARY_COST_BPS
    parameter_frame = pd.DataFrame(
        [
            {
                key: row[key]
                for key in PARAMETER_COLUMNS
            }
        ],
        columns=PARAMETER_COLUMNS,
    )
    _write_cache(
        cache,
        fingerprint,
        summary,
        daily=daily,
        trades=trades,
        parameter=parameter_frame,
        ranking=ranking,
    )
    return pair, str(rebalance.date()), False


def build_pair_quarter_tasks(
    master: pd.DataFrame,
    schedule: pd.DataFrame,
    screens: pd.DataFrame,
) -> pd.DataFrame:
    identities = master[["pair", "sector", "ticker_a", "ticker_b"]].copy()
    windows = schedule[
        [
            "quarter_number",
            "rebalance_date",
            "outer_end_exclusive",
            "formation_start_requested",
        ]
    ].copy()
    identities["_join"] = 1
    windows["_join"] = 1
    tasks = identities.merge(windows, on="_join", how="inner").drop(columns="_join")
    screen_columns = [
        "rebalance_date",
        "pair",
        "screen_status",
        "observations",
        "history_ratio",
        "pair_pvalue",
        "fdr_qvalue",
        "fdr_significant",
        "both_look_i1",
        "positive_hedge_ratio",
        "half_life_days",
        "hurst_exponent",
        "eligible",
        "dependent",
        "independent",
        "alpha",
        "beta",
        "screen_source",
    ]
    missing = sorted(set(screen_columns) - set(screens.columns))
    if missing:
        raise ValueError("screen panel is missing: " + ", ".join(missing))
    tasks = tasks.merge(
        screens[screen_columns],
        on=["rebalance_date", "pair"],
        how="left",
        validate="one_to_one",
    )
    if tasks["screen_status"].isna().any():
        raise RuntimeError("one or more fixed identity-quarter screens are missing")
    tasks = tasks.sort_values(
        ["rebalance_date", "pair"], kind="mergesort"
    ).reset_index(drop=True)
    return tasks


def _parameter_map(spec: PairTuningSpec, base_config: Any) -> dict[str, Any]:
    values = protocol_parameter_sets(spec, stop_z=base_config.stop_z)
    result = {value.config_id: value for value in values}
    if len(result) != 97:
        raise RuntimeError("canonical parameter map does not contain 97 unique IDs")
    return result


def seed_predictive_replays(
    tasks: pd.DataFrame,
    sources: Mapping[str, Path],
    cache_root: Path,
    prices: pd.DataFrame,
    spec: PairTuningSpec,
    base_config: Any,
    *,
    price_sha256: str,
    protocol_sha256: str,
    recompute: bool,
) -> int:
    """Seed only exact pair-quarter winners already replayed by the V6 audit."""

    predictive_manifest = sources.get("predictive_manifest")
    manifest: dict[str, Any] = {}
    if predictive_manifest is not None and predictive_manifest.is_file():
        try:
            manifest = _load_json(predictive_manifest)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if not _cache_lineage_matches(manifest):
        print(
            "predictive seed skipped: source manifest lacks current "
            "execution-engine lineage",
            flush=True,
        )
        return 0

    metrics = _read_csv(
        sources["predictive_metrics"],
        dates=("rebalance_date", "outer_end_exclusive"),
    )
    costs = _read_csv(
        sources["predictive_costs"],
        dates=("rebalance_date", "outer_end_exclusive"),
    )
    signals = _read_csv(
        sources["predictive_signals"],
        dates=("rebalance_date", "outer_end_exclusive", "date"),
    )
    trades = _read_csv(
        sources["predictive_trades"],
        dates=("rebalance_date", "outer_end_exclusive", "entry_date", "exit_date"),
    )
    v6_rankings = _read_csv(
        sources["v6_rankings"], dates=("rebalance_date",)
    )
    selected_rankings = v6_rankings.loc[_bool(v6_rankings["selected"])].copy()
    task_lookup = {
        (pd.Timestamp(row.rebalance_date), str(row.pair)): row._asdict()
        for row in tasks.itertuples(index=False)
    }
    parameter_lookup = _parameter_map(spec, base_config)
    cost_groups = {
        (pd.Timestamp(date), str(pair)): group
        for (date, pair), group in costs.groupby(["rebalance_date", "pair"], sort=False)
    }
    signal_groups = {
        (pd.Timestamp(date), str(pair)): group
        for (date, pair), group in signals.groupby(["rebalance_date", "pair"], sort=False)
    }
    trade_groups = {
        (pd.Timestamp(date), str(pair)): group
        for (date, pair), group in trades.groupby(["rebalance_date", "pair"], sort=False)
    } if not trades.empty else {}
    ranking_groups = {
        (pd.Timestamp(date), str(pair)): group
        for (date, pair), group in v6_rankings.groupby(["rebalance_date", "pair"], sort=False)
    }
    selected_lookup = {
        (pd.Timestamp(row.rebalance_date), str(row.pair)): row._asdict()
        for row in selected_rankings.itertuples(index=False)
    }
    seeded = 0
    for metric in metrics.itertuples(index=False):
        key = (pd.Timestamp(metric.rebalance_date), str(metric.pair))
        task = task_lookup.get(key)
        if task is None:
            continue
        cost_group = cost_groups.get(key)
        daily_source = signal_groups.get(key)
        training = selected_lookup.get(key)
        if cost_group is None or daily_source is None or training is None:
            continue
        one = cost_group.loc[
            pd.to_numeric(cost_group["transaction_cost_bps"], errors="coerce").eq(
                SENSITIVITY_COST_BPS
            )
        ]
        two = cost_group.loc[
            pd.to_numeric(cost_group["transaction_cost_bps"], errors="coerce").eq(
                PRIMARY_COST_BPS
            )
        ]
        if len(one) != 1 or len(two) != 1:
            continue
        one_row = one.iloc[0]
        two_row = two.iloc[0]
        parameter = parameter_lookup.get(str(metric.config_id))
        if parameter is None:
            raise RuntimeError(f"unknown predictive config {metric.config_id}")
        common = _common_summary(task, prices)
        if (
            str(common["screen_dependent"]) != str(metric.dependent)
            or str(common["screen_independent"]) != str(metric.independent)
        ):
            raise RuntimeError(f"predictive orientation mismatch for {key}")
        cache = _pair_cache_paths(cache_root, key[1], key[0])
        fingerprint = _task_fingerprint(task, price_sha256, protocol_sha256)
        if not recompute and _cache_valid(cache, fingerprint):
            continue
        trade_count = int(metric.trade_count)
        status = "evaluated_zero_trade" if trade_count == 0 else "evaluated_traded"
        row = {
            **common,
            "tuning_success": True,
            "active_identity_conditional": True,
            "status": status,
            "cash_reason": "",
            "selected_config_id": parameter.config_id,
            "zscore_method": parameter.zscore_method,
            "zscore_lookback": int(parameter.zscore_lookback),
            "zscore_min_periods": int(parameter.zscore_min_periods),
            "entry_z": float(parameter.entry_z),
            "exit_z": float(parameter.exit_z),
            "stop_z": float(base_config.stop_z),
            "maximum_holding_days": int(parameter.maximum_holding_days),
            "is_baseline": bool(parameter.is_baseline),
            "training_cv_net_cagr": float(training["pooled_net_cagr"]),
            "training_cv_max_drawdown": float(training["max_drawdown"]),
            "validation_trade_count": int(training["validation_trade_count"]),
            "full_training_trade_count": int(training["full_training_trade_count"]),
            "active_fold_count": int(training["active_fold_count"]),
            "oos_refit_alpha": float(metric.alpha),
            "oos_refit_beta": float(metric.beta),
            "q_return_1bp": float(one_row["net_total_return"]),
            "q_return_2bp": float(two_row["net_total_return"]),
            "gross_q_return": float(metric.gross_total_return),
            "trade_count": trade_count,
            "transaction_cost_sum_1bp": float(one_row["transaction_cost_sum"]),
            "transaction_cost_sum_2bp": float(two_row["transaction_cost_sum"]),
            "borrow_cost_sum": float(metric.borrow_cost_sum),
            "start_flat": bool(metric.start_flat),
            "terminal_flat": bool(metric.terminal_flat),
            "terminal_liquidation": bool(metric.terminal_liquidation),
            "replay_source": "reused_predictive_v6_frozen_winner",
            "tuning_failure_reason": "",
            "error_type": "",
            "error": "",
        }
        summary = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
        daily_source = daily_source.sort_values("date", kind="mergesort")
        turnover = pd.to_numeric(daily_source["turnover"], errors="coerce").fillna(0.0)
        gross = pd.to_numeric(daily_source["gross_return"], errors="coerce").fillna(0.0)
        borrow = pd.to_numeric(daily_source["borrow_cost"], errors="coerce").fillna(0.0)
        post_return_nav = 1.0 + gross
        transaction_1 = (
            turnover * SENSITIVITY_COST_BPS / 10_000.0 * post_return_nav
        )
        transaction_2 = turnover * PRIMARY_COST_BPS / 10_000.0 * post_return_nav
        daily = pd.DataFrame(
            {
                "pair": key[1],
                "sector": common["sector"],
                "rebalance_date": key[0],
                "outer_end_exclusive": common["outer_end_exclusive"],
                "date": pd.to_datetime(daily_source["date"]),
                "config_id": parameter.config_id,
                "net_return_1bp": gross - transaction_1 - borrow,
                "net_return_2bp": gross - transaction_2 - borrow,
                "gross_return": gross,
                "turnover": turnover,
                "borrow_cost": borrow,
                "transaction_cost_1bp": transaction_1,
                "transaction_cost_2bp": transaction_2,
                "zscore": pd.to_numeric(daily_source["zscore"], errors="coerce"),
                "position": pd.to_numeric(daily_source["position"], errors="coerce").fillna(0),
                "target_position": pd.to_numeric(daily_source["target_position"], errors="coerce").fillna(0),
                "decision_event": daily_source["decision_event"].astype(str),
                "active_identity_conditional": True,
            },
            columns=DAILY_COLUMNS,
        )
        seeded_trades = trade_groups.get(key, pd.DataFrame()).copy()
        parameter_frame = pd.DataFrame(
            [{key_name: row[key_name] for key_name in PARAMETER_COLUMNS}],
            columns=PARAMETER_COLUMNS,
        )
        ranking = ranking_groups.get(key)
        _write_cache(
            cache,
            fingerprint,
            summary,
            daily=daily,
            trades=seeded_trades,
            parameter=parameter_frame,
            ranking=ranking,
        )
        seeded += 1
    return seeded


def run_pair_quarter_replays(
    paths: EvaluationPaths,
    tasks: pd.DataFrame,
    sources: Mapping[str, Path],
    prices: pd.DataFrame,
    spec: PairTuningSpec,
    base_config: Any,
    *,
    price_sha256: str,
    protocol_sha256: str,
    tuning_workers: int,
    recompute: bool,
    use_predictive_seed: bool,
    maximum_new_tasks: int | None,
) -> dict[str, int]:
    if tuning_workers < 1:
        raise ValueError("tuning_workers must be positive")
    seeded = 0
    if use_predictive_seed and not recompute:
        seeded = seed_predictive_replays(
            tasks,
            sources,
            paths.pair_cache,
            prices,
            spec,
            base_config,
            price_sha256=price_sha256,
            protocol_sha256=protocol_sha256,
            recompute=False,
        )
        print(f"seeded {seeded} exact predictive V6 pair-quarter replays", flush=True)
    pending: list[dict[str, Any]] = []
    cached = 0
    for task in tasks.to_dict(orient="records"):
        cache = _pair_cache_paths(
            paths.pair_cache,
            str(task["pair"]),
            pd.Timestamp(task["rebalance_date"]),
        )
        fingerprint = _task_fingerprint(task, price_sha256, protocol_sha256)
        if not recompute and _cache_valid(cache, fingerprint):
            cached += 1
            continue
        pending.append(task)
    if maximum_new_tasks is not None:
        if maximum_new_tasks < 0:
            raise ValueError("maximum_new_tasks cannot be negative")
        pending = pending[:maximum_new_tasks]
    initializer_args = (
        prices,
        spec,
        base_config,
        str(paths.pair_cache),
        price_sha256,
        protocol_sha256,
        recompute,
    )
    completed = 0
    worker_cache_hits = 0
    if pending and tuning_workers == 1:
        _initialize_pair_worker(*initializer_args)
        for number, task in enumerate(pending, start=1):
            _, _, reused = _run_pair_quarter_task(task)
            completed += 1
            worker_cache_hits += int(reused)
            if number % 25 == 0 or number == len(pending):
                print(f"pair-quarter replay {number}/{len(pending)}", flush=True)
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(tuning_workers, len(pending)),
            initializer=_initialize_pair_worker,
            initargs=initializer_args,
        ) as executor:
            for number, (_, _, reused) in enumerate(
                executor.map(_run_pair_quarter_task, pending, chunksize=1), start=1
            ):
                completed += 1
                worker_cache_hits += int(reused)
                if number % 25 == 0 or number == len(pending):
                    print(f"pair-quarter replay {number}/{len(pending)}", flush=True)
    return {
        "expected_tasks": int(len(tasks)),
        "seeded_tasks": int(seeded),
        "preexisting_cache_hits": int(cached),
        "new_tasks_completed": int(completed),
        "worker_cache_hits": int(worker_cache_hits),
        "remaining_unsubmitted": int(
            max(0, len(tasks) - cached - completed)
        ),
    }


def _read_optional_cache(path: Path, dates: Iterable[str] = ()) -> pd.DataFrame:
    return _read_csv(path, dates=dates) if path.is_file() else pd.DataFrame()


_SUCCESS_CACHE_STATUSES = frozenset({"evaluated_traded", "evaluated_zero_trade"})
_FAILURE_CACHE_STATUSES = frozenset(
    {"tuning_failed_cash", "evaluation_failed_cash", "unavailable_cash"}
)


def _strict_cache_csv(path: Path, *, dates: Iterable[str] = ()) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], errors="coerce", format="mixed"
            )
    return frame


def _marker_row_count(marker: Mapping[str, Any], name: str) -> int:
    value = marker.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    number = int(value)
    if number < 0 or float(value) != float(number):
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _cache_frame_matches_task(
    frame: pd.DataFrame,
    task: Mapping[str, Any],
) -> bool:
    if frame.empty:
        return True
    if "pair" not in frame or "rebalance_date" not in frame:
        return False
    pairs_match = frame["pair"].astype(str).eq(str(task["pair"])).all()
    dates = pd.to_datetime(frame["rebalance_date"], errors="coerce", format="mixed")
    dates_match = dates.eq(pd.Timestamp(task["rebalance_date"])).all()
    return bool(pairs_match and dates_match)


def _read_and_validate_collection_cache(
    cache: Mapping[str, Path],
    task: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load one current cache and verify its marker, key, and artifact contract."""

    reasons: list[str] = []
    try:
        marker = _load_json(cache["complete"])
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"complete_marker_unreadable:{type(error).__name__}"]

    table_dates = {
        "summary": (
            "rebalance_date",
            "outer_end_exclusive",
            "formation_start_requested",
            "formation_start_actual",
            "formation_end_actual",
        ),
        "daily": ("rebalance_date", "outer_end_exclusive", "date"),
        "trades": (
            "rebalance_date",
            "outer_end_exclusive",
            "entry_date",
            "exit_date",
        ),
        "parameter": ("rebalance_date", "outer_end_exclusive"),
        "ranking": ("rebalance_date",),
        "failure": ("rebalance_date",),
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, dates in table_dates.items():
        path = cache[name]
        if not path.is_file():
            tables[name] = pd.DataFrame()
            continue
        try:
            tables[name] = _strict_cache_csv(path, dates=dates)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            reasons.append(f"{name}_unreadable:{type(error).__name__}")
            tables[name] = pd.DataFrame()

    summary = tables["summary"]
    if len(summary) != 1:
        reasons.append(f"summary_row_count:{len(summary)}")
        status = ""
    else:
        status = str(summary.iloc[0].get("status", ""))
        if not _cache_frame_matches_task(summary, task):
            reasons.append("summary_key_mismatch")
        if status != str(marker.get("status", "")):
            reasons.append("summary_marker_status_mismatch")
        if status not in _SUCCESS_CACHE_STATUSES | _FAILURE_CACHE_STATUSES:
            reasons.append(f"unknown_status:{status}")

    try:
        daily_rows = _marker_row_count(marker, "daily_rows")
        trade_rows = _marker_row_count(marker, "trade_rows")
    except (TypeError, ValueError) as error:
        reasons.append(f"marker_row_count_invalid:{error}")
        daily_rows = -1
        trade_rows = -1
    if daily_rows >= 0 and len(tables["daily"]) != daily_rows:
        reasons.append(
            f"daily_row_count_mismatch:marker={daily_rows},actual={len(tables['daily'])}"
        )
    if trade_rows >= 0 and len(tables["trades"]) != trade_rows:
        reasons.append(
            f"trade_row_count_mismatch:marker={trade_rows},actual={len(tables['trades'])}"
        )

    for name, frame in tables.items():
        if name == "summary" or frame.empty:
            continue
        if not _cache_frame_matches_task(frame, task):
            reasons.append(f"{name}_key_mismatch")

    if status in _SUCCESS_CACHE_STATUSES:
        if len(summary) == 1:
            if not bool(_bool(pd.Series([summary.iloc[0].get("tuning_success")])).iloc[0]):
                reasons.append("successful_cache_tuning_success_false")
            if not bool(
                _bool(
                    pd.Series([summary.iloc[0].get("active_identity_conditional")])
                ).iloc[0]
            ):
                reasons.append("successful_cache_active_identity_false")
        for name in ("daily", "parameter", "ranking"):
            if not cache[name].is_file() or tables[name].empty:
                reasons.append(f"successful_cache_missing_{name}")
        if len(tables["parameter"]) != 1:
            reasons.append(f"successful_parameter_row_count:{len(tables['parameter'])}")
        if status == "evaluated_traded" and trade_rows <= 0:
            reasons.append("traded_cache_has_no_trades")
        if len(summary) == 1:
            expected_daily = pd.to_numeric(
                pd.Series([summary.iloc[0].get("evaluation_sessions")]),
                errors="coerce",
            ).iloc[0]
            expected_trades = pd.to_numeric(
                pd.Series([summary.iloc[0].get("trade_count")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(expected_daily) or int(expected_daily) != daily_rows:
                reasons.append("summary_daily_row_count_mismatch")
            if pd.isna(expected_trades) or int(expected_trades) != trade_rows:
                reasons.append("summary_trade_row_count_mismatch")
        if cache["failure"].is_file():
            reasons.append("successful_cache_has_failure_artifact")
    elif status in _FAILURE_CACHE_STATUSES:
        if len(summary) == 1:
            active_value = bool(
                _bool(
                    pd.Series([summary.iloc[0].get("active_identity_conditional")])
                ).iloc[0]
            )
            tuning_value = bool(
                _bool(pd.Series([summary.iloc[0].get("tuning_success")])).iloc[0]
            )
            if active_value:
                reasons.append("failure_cache_active_identity_true")
            expected_tuning = status == "evaluation_failed_cash"
            if tuning_value != expected_tuning:
                reasons.append("failure_cache_tuning_success_mismatch")
        if not cache["failure"].is_file() or len(tables["failure"]) != 1:
            reasons.append("failure_cache_missing_single_failure_row")
        elif str(tables["failure"].iloc[0].get("status", "")) != status:
            reasons.append("failure_summary_status_mismatch")
        for name in ("daily", "trades"):
            if cache[name].is_file():
                reasons.append(f"failure_cache_has_{name}_artifact")
        if status == "evaluation_failed_cash":
            for name in ("parameter", "ranking"):
                if not cache[name].is_file() or tables[name].empty:
                    reasons.append(f"evaluation_failure_missing_{name}")
        else:
            for name in ("parameter", "ranking"):
                if cache[name].is_file():
                    reasons.append(f"pre_evaluation_failure_has_{name}_artifact")

    return tables, reasons


def collect_replay_outputs(
    paths: EvaluationPaths,
    config: Mapping[str, Any],
    master: pd.DataFrame,
    schedule: pd.DataFrame,
    tasks: pd.DataFrame,
    *,
    price_sha256: str,
    protocol_sha256: str,
    immutable_hashes: Mapping[str, str],
    post_freeze_hashes_before: Mapping[str, str],
    sources: Mapping[str, Path],
    allow_incomplete: bool,
) -> dict[str, Any]:
    summary_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    parameter_parts: list[pd.DataFrame] = []
    failure_parts: list[pd.DataFrame] = []
    missing_keys: list[dict[str, str]] = []
    invalid_keys: list[dict[str, str]] = []
    for task in tasks.to_dict(orient="records"):
        cache = _pair_cache_paths(
            paths.pair_cache,
            str(task["pair"]),
            pd.Timestamp(task["rebalance_date"]),
        )
        if not cache["complete"].is_file() or not cache["summary"].is_file():
            missing_keys.append(
                {
                    "pair": str(task["pair"]),
                    "rebalance_date": str(pd.Timestamp(task["rebalance_date"]).date()),
                }
            )
            continue
        expected_fingerprint = _task_fingerprint(
            task,
            price_sha256,
            protocol_sha256,
        )
        if not _cache_valid(cache, expected_fingerprint):
            invalid_keys.append(
                {
                    "pair": str(task["pair"]),
                    "rebalance_date": str(pd.Timestamp(task["rebalance_date"]).date()),
                    "reason": "fingerprint_or_execution_lineage_mismatch",
                }
            )
            continue
        tables, reasons = _read_and_validate_collection_cache(cache, task)
        if reasons:
            invalid_keys.append(
                {
                    "pair": str(task["pair"]),
                    "rebalance_date": str(pd.Timestamp(task["rebalance_date"]).date()),
                    "reason": ";".join(reasons),
                }
            )
            continue
        summary_parts.append(tables["summary"])
        daily = tables["daily"]
        if not daily.empty:
            daily_parts.append(daily)
        trades = tables["trades"]
        if not trades.empty:
            trade_parts.append(trades)
        parameter = tables["parameter"]
        if not parameter.empty:
            parameter_parts.append(parameter)
        failure = tables["failure"]
        if not failure.empty:
            failure_parts.append(failure)
    if (missing_keys or invalid_keys) and not allow_incomplete:
        missing_preview = ", ".join(
            f"{row['pair']}@{row['rebalance_date']}" for row in missing_keys[:5]
        ) or "none"
        invalid_preview = ", ".join(
            f"{row['pair']}@{row['rebalance_date']}:{row['reason']}"
            for row in invalid_keys[:5]
        ) or "none"
        raise RuntimeError(
            f"pair-quarter cache validation failed: {len(missing_keys)} missing, "
            f"{len(invalid_keys)} invalid/stale; missing first: {missing_preview}; "
            f"invalid first: {invalid_preview}"
        )
    quarterly = (
        pd.concat(summary_parts, ignore_index=True)
        if summary_parts
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )
    if quarterly.duplicated(["rebalance_date", "pair"]).any():
        raise RuntimeError("collected quarterly panel contains duplicate keys")
    subset = master[["pair", "ever_v6_selected", "v6_selected_vintage_count"]]
    quarterly = quarterly.merge(subset, on="pair", how="left", validate="many_to_one")
    quarterly["fixed_identity_selection_is_ex_post"] = True
    quarterly["screen_eligibility_used_as_trade_gate"] = False
    quarterly = quarterly.sort_values(
        ["rebalance_date", "pair"], kind="mergesort"
    ).reset_index(drop=True)

    replay_daily = (
        pd.concat(daily_parts, ignore_index=True)
        if daily_parts
        else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    if not replay_daily.empty:
        replay_daily = replay_daily.merge(
            subset[["pair", "ever_v6_selected"]],
            on="pair",
            how="left",
            validate="many_to_one",
        )
        replay_daily = replay_daily.sort_values(
            ["pair", "date"], kind="mergesort"
        ).reset_index(drop=True)
        if replay_daily.duplicated(["pair", "date"]).any():
            raise RuntimeError("sparse replay daily rows duplicate a pair/date")
    trades = (
        pd.concat(trade_parts, ignore_index=True)
        if trade_parts
        else pd.DataFrame()
    )
    if not trades.empty:
        if "ever_v6_selected" not in trades:
            trades = trades.merge(
                subset[["pair", "ever_v6_selected"]],
                on="pair",
                how="left",
                validate="many_to_one",
            )
        replay_provenance = quarterly[
            ["pair", "rebalance_date", "replay_source"]
        ].drop_duplicates(["pair", "rebalance_date"])
        trades = trades.drop(columns=["replay_source"], errors="ignore")
        trades = trades.merge(
            replay_provenance,
            on=["pair", "rebalance_date"],
            how="left",
            validate="many_to_one",
        )
        source_cost = (
            pd.to_numeric(trades["transaction_cost_bps"], errors="coerce")
            if "transaction_cost_bps" in trades
            else pd.Series(PRIMARY_COST_BPS, index=trades.index, dtype=float)
        )
        trades["transaction_cost_bps"] = source_cost.fillna(PRIMARY_COST_BPS)
        trades["original_v6_selected_top10_trade_row"] = (
            _bool(trades["selected_top10"])
            if "selected_top10" in trades
            else False
        )
        trades["screen_eligibility_used_as_trade_gate"] = False
        trades["cross_pair_top10_cutoff_applied"] = False
        trades["diagnostics_retained_posthoc"] = True
        trades["fixed_identity_selection_is_ex_post"] = True
        trades = trades.sort_values(
            ["rebalance_date", "pair", "trade_number"], kind="mergesort"
        ).reset_index(drop=True)
    parameters = (
        pd.concat(parameter_parts, ignore_index=True)
        if parameter_parts
        else pd.DataFrame(columns=PARAMETER_COLUMNS)
    )
    if not parameters.empty:
        parameters = parameters.merge(
            subset[["pair", "ever_v6_selected"]],
            on="pair",
            how="left",
            validate="many_to_one",
        )
    failures = (
        pd.concat(failure_parts, ignore_index=True)
        if failure_parts
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    if not failures.empty:
        failures = failures.merge(
            subset[["pair", "ever_v6_selected"]],
            on="pair",
            how="left",
            validate="many_to_one",
        )
    availability_columns = [
        "pair",
        "sector",
        "rebalance_date",
        "outer_end_exclusive",
        "screen_status",
        "screen_observations",
        "screen_history_ratio",
        "screen_core_health_pass",
        "screen_strict_eligible",
        "history_available",
        "orientation_available",
        "tuning_success",
        "validation_trade_count",
        "full_training_trade_count",
        "active_fold_count",
        "tuning_failure_reason",
        "active_identity_conditional",
        "status",
        "cash_reason",
        "trade_count",
        "ever_v6_selected",
    ]
    availability = quarterly[availability_columns].copy()
    _atomic_csv(quarterly, paths.processed / QUARTERLY_NAME)
    _atomic_csv(replay_daily, paths.processed / REPLAY_DAILY_NAME)
    _atomic_csv(trades, paths.processed / TRADES_NAME)
    _atomic_csv(parameters, paths.processed / PARAMETER_NAME)
    _atomic_csv(failures, paths.processed / FAILURES_NAME)
    _atomic_csv(availability, paths.processed / AVAILABILITY_NAME)

    post_freeze_after = _post_freeze_source_hashes(sources)
    source_unchanged = dict(post_freeze_hashes_before) == post_freeze_after
    expected_rows = int(
        config["evaluation"]["expected_fixed_identity_count"]
        * config["evaluation"]["expected_quarter_intervals"]
    )
    active = quarterly.get(
        "active_identity_conditional", pd.Series(False, index=quarterly.index)
    ).fillna(False).astype(bool)
    successful = quarterly.loc[active]
    status_series = quarterly.get("status", pd.Series(dtype=str)).astype(str)
    status_counts = {
        status: int(status_series.eq(status).sum())
        for status in sorted(_SUCCESS_CACHE_STATUSES | _FAILURE_CACHE_STATUSES)
    }
    status_count_total = int(sum(status_counts.values()))
    status_count_reconciles = bool(status_count_total == len(quarterly))
    daily_reconciles = True
    if not replay_daily.empty:
        for (date, pair), group in replay_daily.groupby(
            ["rebalance_date", "pair"], sort=False
        ):
            match = quarterly.loc[
                pd.to_datetime(quarterly["rebalance_date"]).eq(pd.Timestamp(date))
                & quarterly["pair"].astype(str).eq(str(pair))
            ]
            if len(match) != 1:
                daily_reconciles = False
                break
            for cost in (1, 2):
                actual = float(
                    (1.0 + pd.to_numeric(group[f"net_return_{cost}bp"], errors="coerce").fillna(0.0)).prod()
                    - 1.0
                )
                expected = float(match.iloc[0][f"q_return_{cost}bp"])
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                    daily_reconciles = False
                    break
            if not daily_reconciles:
                break
    audit = {
        "schema_version": EXECUTION_CACHE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "study": "fixed_114_pair_identity_10y_evaluation",
        "evaluation_only": True,
        "v6_artifacts_modified": False,
        "fixed_identity_count": int(len(master)),
        "v6_selected_subset_identity_count": int(master["ever_v6_selected"].sum()),
        "clipped_quarter_count": int(len(schedule)),
        "expected_dense_quarterly_rows": expected_rows,
        "actual_quarterly_rows": int(len(quarterly)),
        "missing_pair_quarter_cache_count": int(len(missing_keys)),
        "invalid_pair_quarter_cache_count": int(len(invalid_keys)),
        "status_counts": status_counts,
        "status_count_total": status_count_total,
        "status_count_reconciles": status_count_reconciles,
        "active_evaluated_pair_quarters": int(active.sum()),
        "evaluated_traded_pair_quarters": status_counts["evaluated_traded"],
        "evaluated_zero_trade_pair_quarters": int(
            status_counts["evaluated_zero_trade"]
        ),
        "tuning_failed_cash_pair_quarters": int(
            status_counts["tuning_failed_cash"]
        ),
        "evaluation_failed_cash_pair_quarters": int(
            status_counts["evaluation_failed_cash"]
        ),
        "unavailable_cash_pair_quarters": int(
            status_counts["unavailable_cash"]
        ),
        "screen_eligibility_is_diagnostic_only": True,
        "daily_output_is_sparse_successful_replays": True,
        "dense_daily_output_reserved_for_downstream_aggregation": "pair_identity_daily_returns.csv.gz",
        "sparse_daily_rows": int(len(replay_daily)),
        "sparse_daily_reconciles_to_quarterly_returns": bool(daily_reconciles),
        "all_successful_sleeves_start_flat": bool(
            _bool(successful["start_flat"]).all() if len(successful) else True
        ),
        "all_successful_sleeves_terminal_flat": bool(
            _bool(successful["terminal_flat"]).all() if len(successful) else True
        ),
        "all_successful_sleeves_charge_terminal_liquidation": bool(
            _bool(successful["terminal_liquidation"]).all() if len(successful) else True
        ),
        **_execution_cache_lineage(),
        "immutable_hashes": dict(immutable_hashes),
        "post_freeze_source_hashes_before": dict(post_freeze_hashes_before),
        "post_freeze_source_hashes_after": post_freeze_after,
        "post_freeze_sources_unchanged": source_unchanged,
        "critical_pass": bool(
            len(quarterly) == expected_rows
            and not missing_keys
            and not invalid_keys
            and not quarterly.duplicated(["rebalance_date", "pair"]).any()
            and status_count_reconciles
            and status_count_total == expected_rows
            and status_counts["evaluation_failed_cash"] == 0
            and daily_reconciles
            and source_unchanged
        ),
        "schemas": {
            "quarterly": list(quarterly.columns),
            "sparse_replay_daily": list(replay_daily.columns),
            "trades": list(trades.columns),
            "parameters": list(parameters.columns),
            "failures": list(failures.columns),
            "availability": list(availability.columns),
        },
        "missing_pair_quarter_keys_preview": missing_keys[:100],
        "invalid_pair_quarter_keys_preview": invalid_keys[:100],
    }
    _atomic_json(paths.root / AUDIT_NAME, audit)
    return audit


def run_evaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path | None = None,
    *,
    stage: str = "all",
    refresh_data: bool = False,
    recompute_screens: bool = False,
    recompute_pair_quarters: bool = False,
    screen_workers: int = 4,
    tuning_workers: int = 4,
    use_predictive_seed: bool = True,
    maximum_new_tasks: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    stages = {"prepare", "screens", "replay", "assemble", "all"}
    if stage not in stages:
        raise ValueError(f"stage must be one of {sorted(stages)}")
    if screen_workers < 1 or tuning_workers < 1:
        raise ValueError("worker counts must be positive")
    workspace = Path(__file__).resolve().parents[1]
    config_path = _resolve(workspace, config_path)
    config = _load_config(config_path)
    root_value = output_root or config["outputs"]["default_root"]
    root = _resolve(workspace, root_value)
    paths = EvaluationPaths.create(root)
    sources = _source_paths(config, workspace)
    v6 = _load_module(sources["v6_runner"], "pair_identity_v6_immutable_runner")
    protocol, spec, base_config, immutable_hashes = _validate_immutable_protocol(
        config, sources, root, v6
    )

    if stage in {"prepare", "all"}:
        prices, price_metadata = prepare_prices(
            paths,
            config,
            sources,
            refresh_data=refresh_data,
        )
        schedule = _write_schedule(paths, config, prices)
    else:
        prices, price_metadata = load_prepared_prices(paths, config)
        schedule = _read_csv(
            paths.processed / SCHEDULE_NAME,
            dates=(
                "rebalance_date",
                "outer_end_exclusive",
                "formation_start_requested",
            ),
        )
        if schedule.empty:
            raise FileNotFoundError(
                f"prepared schedule is absent; run --stage prepare first: {paths.processed / SCHEDULE_NAME}"
            )
    expected_quarters = int(config["evaluation"]["expected_quarter_intervals"])
    if len(schedule) != expected_quarters:
        raise RuntimeError(
            f"prepared schedule has {len(schedule)} rows; expected {expected_quarters}"
        )

    # The fixed identity universe is intentionally ex-post and descriptive.
    # These artifacts are loaded only for that declared purpose.
    post_hashes_before = _post_freeze_source_hashes(sources)
    _validate_configured_post_freeze_hashes(config, post_hashes_before)
    master = build_fixed_identity_master(paths, config, sources)
    if stage == "prepare":
        return {
            "stage": stage,
            "output_root": root,
            "identity_count": int(len(master)),
            "quarter_count": int(len(schedule)),
            "price_sha256": str(price_metadata["sha256"]),
        }

    if stage in {"screens", "all"}:
        screens = ensure_all_screens(
            paths,
            config,
            sources,
            protocol,
            prices,
            price_metadata,
            schedule,
            v6,
            screen_workers=screen_workers,
            recompute=recompute_screens,
        )
    else:
        screens = _read_csv(
            paths.processed / SCREEN_NAME,
            dates=(
                "rebalance_date",
                "outer_end_exclusive",
                "formation_start_requested",
            ),
        )
        if screens.empty:
            raise FileNotFoundError(
                f"screen panel is absent; run --stage screens first: {paths.processed / SCREEN_NAME}"
            )
    if stage == "screens":
        return {
            "stage": stage,
            "output_root": root,
            "screen_rows": int(len(screens)),
            "quarter_count": int(screens["rebalance_date"].nunique()),
        }

    tasks = build_pair_quarter_tasks(master, schedule, screens)
    expected_tasks = int(
        config["evaluation"]["expected_fixed_identity_count"]
        * config["evaluation"]["expected_quarter_intervals"]
    )
    if len(tasks) != expected_tasks:
        raise RuntimeError(f"expected {expected_tasks} pair-quarter tasks, received {len(tasks)}")

    replay_counts: dict[str, int] = {}
    if stage in {"replay", "all"}:
        replay_counts = run_pair_quarter_replays(
            paths,
            tasks,
            sources,
            prices,
            spec,
            base_config,
            price_sha256=str(price_metadata["sha256"]),
            protocol_sha256=immutable_hashes["v6_config"],
            tuning_workers=tuning_workers,
            recompute=recompute_pair_quarters,
            use_predictive_seed=use_predictive_seed,
            maximum_new_tasks=maximum_new_tasks,
        )
    if stage == "replay":
        return {
            "stage": stage,
            "output_root": root,
            **replay_counts,
        }

    audit = collect_replay_outputs(
        paths,
        config,
        master,
        schedule,
        tasks,
        price_sha256=str(price_metadata["sha256"]),
        protocol_sha256=immutable_hashes["v6_config"],
        immutable_hashes=immutable_hashes,
        post_freeze_hashes_before=post_hashes_before,
        sources=sources,
        allow_incomplete=allow_incomplete,
    )
    return {
        "stage": stage,
        "output_root": root,
        "quarterly_results": paths.processed / QUARTERLY_NAME,
        "sparse_replay_daily": paths.processed / REPLAY_DAILY_NAME,
        "audit": paths.root / AUDIT_NAME,
        "critical_pass": bool(audit["critical_pass"]),
        **replay_counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the additive fixed-114 identity V6 evaluation. The 41 clipped "
            "quarter intervals include the initial and final partial quarters."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=("prepare", "screens", "replay", "assemble", "all"),
        default="all",
        help="Resumable stage; 'all' runs the complete pipeline.",
    )
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--recompute-screens", action="store_true")
    parser.add_argument("--recompute-pair-quarters", action="store_true")
    parser.add_argument("--screen-workers", type=int, default=4)
    parser.add_argument("--tuning-workers", type=int, default=4)
    parser.add_argument("--disable-predictive-seed", action="store_true")
    parser.add_argument(
        "--maximum-new-tasks",
        type=int,
        default=None,
        help="Debug/resume limiter. Assembly remains fail-closed unless --allow-incomplete is supplied.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_evaluation(
        args.config,
        args.output,
        stage=args.stage,
        refresh_data=args.refresh_data,
        recompute_screens=args.recompute_screens,
        recompute_pair_quarters=args.recompute_pair_quarters,
        screen_workers=args.screen_workers,
        tuning_workers=args.tuning_workers,
        use_predictive_seed=not args.disable_predictive_seed,
        maximum_new_tasks=args.maximum_new_tasks,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
