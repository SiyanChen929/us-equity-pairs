#!/usr/bin/env python3
"""Run the additive V8 pair-centric full-panel strategy study."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.backtest import performance_metrics
from equity_pairs.config import StrategyConfig
from equity_pairs.identity_evaluation import (
    anniversary_block_labels,
    bootstrap_cagr_from_log_blocks,
    exact_calendar_years,
    path_statistics,
    percentile_score,
    remove_block_as_cash,
)
from equity_pairs.intraday_risk import audit_minute_history
from equity_pairs.selection_inference import circular_block_indices, romano_wolf_max_t
from equity_pairs.v7_experiments import (
    V7Rules,
    aggregate_ensemble,
    parameter_from_schedule,
    simulate_v7_pair,
)
from equity_pairs.v8_pair_selection import (
    reprice_returns,
    select_strategy_one_standard_error,
)


DEFAULT_CONFIG = Path("configs/pair_strategy_v8_pit_lineage_next_close_v3.json")
_PRICES: pd.DataFrame | None = None
_BASE: StrategyConfig | None = None
_CANDIDATES: list[dict[str, Any]] = []
_V6_CACHE: Path | None = None
_S5_EXPECTED_SHA_BY_KEY: dict[tuple[str, str], str] | None = None

CACHE_SCHEMA_VERSION = 5
ENGINE_SOURCE_RELATIVE_PATHS = (
    "scripts/run_pair_v8_full_panel.py",
    "scripts/audit_pair_v8_reconstruction.py",
    "src/equity_pairs/backtest.py",
    "src/equity_pairs/config.py",
    "src/equity_pairs/grid_search.py",
    "src/equity_pairs/pair_tuning.py",
    "src/equity_pairs/statistics.py",
    "src/equity_pairs/validation.py",
    "src/equity_pairs/identity_evaluation.py",
    "src/equity_pairs/selection_inference.py",
    "src/equity_pairs/intraday_risk.py",
    "src/equity_pairs/v7_experiments.py",
    "src/equity_pairs/v8_pair_selection.py",
)
INDEPENDENT_AUDIT_SCRIPT = PROJECT_ROOT / "scripts/audit_pair_v8_reconstruction.py"
INDEPENDENT_AUDIT_DETAIL = "v8_independent_reconstruction_daily.csv.gz"
INDEPENDENT_AUDIT_SUMMARY = "v8_independent_reconstruction_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_bytes(path: Path) -> tuple[bytes, str]:
    """Read one immutable byte snapshot and hash those exact bytes."""

    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _read_csv_snapshot(
    raw: bytes,
    path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(io.BytesIO(raw), compression=compression, **kwargs)


@contextmanager
def _exclusive_output_run_lock(output_root: Path, invocation_nonce: str):
    """Prevent concurrent writers from sharing a V8 output/cache root."""

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".v8_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another V8 run holds the output lock: {lock_path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "invocation_nonce": str(invocation_nonce),
                    "pid": os.getpid(),
                    "acquired_at_utc": _utc_now(),
                }
            )
        )
        handle.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _path_sha256_or_missing(path: Path) -> str:
    return _sha256(path) if path.is_file() else "MISSING"


def _canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _engine_source_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in ENGINE_SOURCE_RELATIVE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing V8 engine dependency: {path}")
        manifest[relative] = _sha256(path)
    return manifest


def _runtime_manifest() -> dict[str, Any]:
    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "matplotlib",
        "jinja2",
    ):
        packages[package] = importlib.metadata.version(package)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "packages": packages,
        "numpy_build": getattr(np.__config__, "CONFIG", {}),
    }


def _cache_marker_payload(
    *,
    fingerprint: str,
    artifact: Path,
    artifact_rows: int,
    pair: str,
    payload_sha256: str,
    rebalance_date: Any | None = None,
    artifact_date_column: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": str(fingerprint),
        "pair": str(pair),
        "payload_sha256": str(payload_sha256),
        "artifact_name": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "artifact_rows": int(artifact_rows),
        "completed_at_utc": _utc_now(),
        # Workers publish provisional artifacts.  The parent process promotes
        # them only after every declared input still matches the run snapshot.
        "input_stage_validated": False,
    }
    if rebalance_date is not None:
        marker["rebalance_date"] = str(pd.Timestamp(rebalance_date).date())
    if artifact_date_column is not None:
        frame = pd.read_csv(artifact)
        if artifact_date_column not in frame.columns:
            raise ValueError(
                f"cache artifact lacks date column {artifact_date_column}: {artifact}"
            )
        dates = pd.to_datetime(frame[artifact_date_column], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"cache artifact has invalid dates: {artifact}")
        marker["artifact_date_column"] = artifact_date_column
        marker["artifact_date_start"] = (
            str(dates.min().date()) if len(dates) else None
        )
        marker["artifact_date_end"] = (
            str(dates.max().date()) if len(dates) else None
        )
    marker.update(extra)
    return marker


def _cache_marker_matches(
    marker_path: Path,
    artifact: Path,
    *,
    fingerprint: str,
    pair: str,
    payload_sha256: str,
    rebalance_date: Any | None = None,
    artifact_date_column: str | None = None,
) -> bool:
    if not marker_path.is_file() or not artifact.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False
        if marker.get("input_stage_validated") is not True:
            return False
        if not str(marker.get("run_nonce", "")):
            return False
        if (
            not marker.get("cache_stage")
            or marker.get("input_validation_stage") != marker.get("cache_stage")
        ):
            return False
        if marker.get("fingerprint") != str(fingerprint):
            return False
        if marker.get("pair") != str(pair):
            return False
        if marker.get("payload_sha256") != str(payload_sha256):
            return False
        if marker.get("artifact_name") != artifact.name:
            return False
        if marker.get("artifact_sha256") != _sha256(artifact):
            return False
        if rebalance_date is not None and marker.get("rebalance_date") != str(
            pd.Timestamp(rebalance_date).date()
        ):
            return False
        frame = pd.read_csv(artifact)
        actual_rows = len(frame)
        if int(marker.get("artifact_rows", -1)) != actual_rows:
            return False
        if artifact_date_column is not None:
            if marker.get("artifact_date_column") != artifact_date_column:
                return False
            if artifact_date_column not in frame.columns:
                return False
            dates = pd.to_datetime(frame[artifact_date_column], errors="coerce")
            if dates.isna().any():
                return False
            actual_start = str(dates.min().date()) if len(dates) else None
            actual_end = str(dates.max().date()) if len(dates) else None
            if marker.get("artifact_date_start") != actual_start:
                return False
            if marker.get("artifact_date_end") != actual_end:
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    return True


def _safe(value: Any) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value).lower()).strip("_")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_bytes(raw: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _input_mutation_audit(
    *,
    file_hashes_before: Mapping[str, str],
    engine_manifest_before: Mapping[str, str],
    s5_hashes_before: Mapping[str, str],
    runtime_manifest_before: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare the live inputs with the immutable hashes captured at run start."""

    file_hashes_after = {
        str(path): _path_sha256_or_missing(Path(path))
        for path in file_hashes_before
    }
    engine_manifest_after = _engine_source_manifest()
    s5_hashes_after = {
        str(path): _path_sha256_or_missing(Path(path))
        for path in s5_hashes_before
    }
    runtime_before = (
        {}
        if runtime_manifest_before is None
        else {
            "runtime_manifest": _canonical_payload_sha256(
                runtime_manifest_before
            )
        }
    )
    runtime_after = (
        {}
        if runtime_manifest_before is None
        else {"runtime_manifest": _canonical_payload_sha256(_runtime_manifest())}
    )
    rows: list[dict[str, Any]] = []
    for source_type, before, after in (
        ("declared_file", file_hashes_before, file_hashes_after),
        ("engine_source", engine_manifest_before, engine_manifest_after),
        ("s5_parameter_ranking", s5_hashes_before, s5_hashes_after),
        ("runtime", runtime_before, runtime_after),
    ):
        for path in sorted(set(before) | set(after)):
            before_sha = before.get(path, "MISSING")
            after_sha = after.get(path, "MISSING")
            rows.append(
                {
                    "source_type": source_type,
                    "path": path,
                    "sha256_before": before_sha,
                    "sha256_after": after_sha,
                    "unchanged": before_sha == after_sha,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "source_type",
            "path",
            "sha256_before",
            "sha256_after",
            "unchanged",
        ],
    )


def _validate_input_stage_and_promote_cache(
    *,
    stage: str,
    results: Iterable[dict[str, Any]],
    file_hashes_before: Mapping[str, str],
    engine_manifest_before: Mapping[str, str],
    s5_hashes_before: Mapping[str, str],
    audit_path: Path,
    runtime_manifest_before: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Fail closed on input mutation, then promote provisional cache markers."""

    audit = _input_mutation_audit(
        file_hashes_before=file_hashes_before,
        engine_manifest_before=engine_manifest_before,
        s5_hashes_before=s5_hashes_before,
        runtime_manifest_before=runtime_manifest_before,
    )
    _atomic_csv(audit, audit_path)
    unchanged = bool(audit["unchanged"].all()) if len(audit) else True
    if not unchanged:
        changed = audit.loc[~audit["unchanged"], ["source_type", "path"]]
        raise RuntimeError(
            f"V8 declared inputs changed during {stage} stage: "
            + repr(changed.to_dict("records"))
        )

    result_list = list(results)
    validated: list[
        tuple[dict[str, Any], Path, Path, dict[str, Any], pd.DataFrame]
    ] = []
    # Validate every result before writing any promotion.  This avoids a
    # partially promoted batch when one later result is malformed.
    for result in result_list:
        marker_text = result.get("marker_path")
        artifact_text = result.get("path")
        if not marker_text or not artifact_text:
            raise RuntimeError(
                f"{stage} worker result lacks provisional cache paths"
            )
        marker_path = Path(str(marker_text))
        artifact = Path(str(artifact_text))
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{stage} provisional cache marker is unreadable: {marker_path}"
            ) from error
        if marker.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                f"{stage} provisional cache marker has the wrong schema: {marker_path}"
            )
        cached = bool(result.get("cached", False))
        expected_validation_state = True if cached else False
        if marker.get("input_stage_validated") is not expected_validation_state:
            raise RuntimeError(
                f"{stage} cache marker validation state mismatch: {marker_path}"
            )
        if cached and marker.get("input_validation_stage") != str(stage):
            raise RuntimeError(
                f"{stage} cached marker lacks a matching prior promotion: {marker_path}"
            )
        if marker.get("cache_stage") != str(stage):
            raise RuntimeError(
                f"{stage} provisional cache marker has the wrong stage: {marker_path}"
            )
        expected_identity = {
            "cache_stage": str(stage),
            "run_nonce": str(result.get("run_nonce", "")),
            "fingerprint": str(result.get("fingerprint", "")),
            "payload_sha256": str(result.get("payload_sha256", "")),
            "pair": str(result.get("pair", "")),
            "artifact_date_column": str(
                result.get("artifact_date_column", "")
            ),
        }
        if not all(expected_identity.values()) or any(
            str(marker.get(field, "")) != expected
            for field, expected in expected_identity.items()
        ):
            raise RuntimeError(
                f"{stage} provisional cache marker identity mismatch: {marker_path}"
            )
        expected_rebalance = result.get("rebalance_date")
        if expected_rebalance is not None and marker.get("rebalance_date") != str(
            pd.Timestamp(expected_rebalance).date()
        ):
            raise RuntimeError(
                f"{stage} provisional cache marker identity mismatch: {marker_path}"
            )
        if not artifact.is_file() or marker.get("artifact_name") != artifact.name:
            raise RuntimeError(
                f"{stage} provisional cache artifact is missing or mismatched: {artifact}"
            )
        try:
            artifact_raw = artifact.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"{stage} cache artifact is unreadable: {artifact}"
            ) from error
        artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()
        if marker.get("artifact_sha256") != artifact_sha256:
            raise RuntimeError(
                f"{stage} provisional cache artifact hash mismatch: {artifact}"
            )
        try:
            artifact_frame = _read_csv_snapshot(artifact_raw, artifact)
            artifact_rows = len(artifact_frame)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise RuntimeError(
                f"{stage} provisional cache artifact is unreadable: {artifact}"
            ) from error
        if int(marker.get("artifact_rows", -1)) != artifact_rows:
            raise RuntimeError(
                f"{stage} provisional cache artifact row mismatch: {artifact}"
            )
        date_column = expected_identity["artifact_date_column"]
        if date_column not in artifact_frame.columns:
            raise RuntimeError(
                f"{stage} provisional cache artifact lacks {date_column}: {artifact}"
            )
        artifact_dates = pd.to_datetime(
            artifact_frame[date_column], errors="coerce"
        )
        if artifact_dates.isna().any():
            raise RuntimeError(
                f"{stage} provisional cache artifact has invalid dates: {artifact}"
            )
        actual_start = str(artifact_dates.min().date()) if artifact_rows else None
        actual_end = str(artifact_dates.max().date()) if artifact_rows else None
        if (
            marker.get("artifact_date_start") != actual_start
            or marker.get("artifact_date_end") != actual_end
        ):
            raise RuntimeError(
                f"{stage} provisional cache artifact date range mismatch: {artifact}"
            )
        validated.append(
            (result, marker_path, artifact, marker, artifact_frame)
        )

    for result, marker_path, _artifact, marker, _artifact_frame in validated:
        if bool(result.get("cached", False)):
            continue
        marker["input_stage_validated"] = True
        marker["input_validation_stage"] = str(stage)
        marker["input_validated_at_utc"] = _utc_now()
        _atomic_json(marker, marker_path)

    # Downstream aggregation consumes these verified in-memory frames rather
    # than reopening a shared path after validation.
    for result, _marker_path, _artifact, _marker, artifact_frame in validated:
        result["_validated_frame"] = artifact_frame
    return audit


def _base_config(config: Mapping[str, Any]) -> StrategyConfig:
    evaluation = config["evaluation"]
    return StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        maximum_holding_days=60,
        transaction_cost_bps=float(evaluation["primary_transaction_cost_bps"]),
        transaction_cost_scenarios_bps=(
            float(evaluation["sensitivity_transaction_cost_bps"]),
            float(evaluation["primary_transaction_cost_bps"]),
        ),
        annual_short_borrow_bps=float(evaluation["annual_short_borrow_bps"]),
    )


def _rules(candidate: Mapping[str, Any], *, terminal: bool = True) -> V7Rules:
    return V7Rules(
        block_entry_at_or_beyond_stop=bool(candidate.get("stop_entry_gate", False)),
        require_reversal_confirmation=bool(candidate.get("reversal_confirmation", False)),
        half_life_holding_multiplier=candidate.get("half_life_multiplier"),
        dynamic_beta=bool(candidate.get("dynamic_beta", False)),
        kalman_process_variance_ratio=float(candidate.get("kalman_process_variance_ratio", 1e-5)),
        terminal_liquidation=terminal,
    )


def _worker_init(
    price_path: str,
    base_payload: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    v6_cache: str,
    expected_price_sha256: str | None = None,
    s5_expected_sha_by_key: Mapping[tuple[str, str], str] | None = None,
    expected_runtime_sha256: str | None = None,
) -> None:
    global _PRICES, _BASE, _CANDIDATES, _V6_CACHE, _S5_EXPECTED_SHA_BY_KEY
    actual_runtime_sha256 = _canonical_payload_sha256(_runtime_manifest())
    if (
        expected_runtime_sha256 is not None
        and actual_runtime_sha256 != str(expected_runtime_sha256)
    ):
        raise RuntimeError(
            "runtime manifest mismatch during worker initialization"
        )
    raw_prices = Path(price_path).read_bytes()
    actual_price_sha256 = hashlib.sha256(raw_prices).hexdigest()
    if (
        expected_price_sha256 is not None
        and actual_price_sha256 != str(expected_price_sha256)
    ):
        raise RuntimeError(
            "price snapshot hash mismatch during worker initialization"
        )
    _PRICES = pd.read_csv(
        io.BytesIO(raw_prices), index_col="date", parse_dates=True
    ).sort_index()
    _BASE = StrategyConfig(**dict(base_payload))
    _CANDIDATES = candidates
    _V6_CACHE = Path(v6_cache)
    _S5_EXPECTED_SHA_BY_KEY = (
        None
        if s5_expected_sha_by_key is None
        else {
            (str(pair), str(date)): str(sha256)
            for (pair, date), sha256 in s5_expected_sha_by_key.items()
        }
    )


def _indexes(row: Mapping[str, Any]) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if _PRICES is None:
        raise RuntimeError("worker prices are not initialized")
    train = _PRICES.index[
        (_PRICES.index >= pd.Timestamp(row["formation_start_requested"]))
        & (_PRICES.index < pd.Timestamp(row["rebalance_date"]))
    ]
    evaluation = _PRICES.index[
        (_PRICES.index >= pd.Timestamp(row["rebalance_date"]))
        & (_PRICES.index < pd.Timestamp(row["outer_end_exclusive"]))
    ]
    if not len(train) or not len(evaluation):
        raise ValueError("empty train/evaluation index")
    return train, evaluation


def _definition(row: Mapping[str, Any]) -> dict[str, str]:
    dependent = str(row["dependent"])
    ticker_a = str(row["ticker_a"])
    ticker_b = str(row["ticker_b"])
    independent = ticker_b if dependent == ticker_a else ticker_a
    return {
        "pair": str(row["pair"]),
        "sector": str(row["sector"]),
        "dependent": dependent,
        "independent": independent,
    }


def _ranking_path_for(row: Mapping[str, Any], v6_cache: Path) -> Path:
    return (
        v6_cache
        / _safe(row["pair"])
        / pd.Timestamp(row["rebalance_date"]).strftime("%Y%m%d")
        / "parameter_rankings.csv.gz"
    )


def _ranking_path(row: Mapping[str, Any]) -> Path:
    if _V6_CACHE is None:
        raise RuntimeError("worker cache is not initialized")
    return _ranking_path_for(row, _V6_CACHE)


def _s5_ranking_manifest(
    successful: pd.DataFrame,
    v6_cache: Path,
    candidates: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    columns = [
        "pair",
        "rebalance_date",
        "path",
        "sha256",
        "row_count",
        "eligible_row_count",
    ]
    if not any(candidate.get("ensemble") for candidate in candidates):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for raw in successful.sort_values(["pair", "rebalance_date"]).to_dict("records"):
        path = _ranking_path_for(raw, v6_cache)
        if not path.is_file():
            raise FileNotFoundError(
                "missing S5 parameter-ranking input for "
                f"{raw['pair']} {pd.Timestamp(raw['rebalance_date']).date()}: {path}"
            )
        raw_ranking, ranking_sha256 = _snapshot_bytes(path)
        ranking = _read_csv_snapshot(raw_ranking, path)
        required = {"config_id", "eligible"}
        missing = sorted(required - set(ranking.columns))
        if missing:
            raise ValueError(f"S5 ranking input {path} is missing columns: {missing}")
        eligible = ranking["eligible"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        if not eligible.any():
            raise ValueError(f"S5 ranking input has no eligible parameters: {path}")
        rows.append(
            {
                "pair": str(raw["pair"]),
                "rebalance_date": str(pd.Timestamp(raw["rebalance_date"]).date()),
                "path": str(path),
                "sha256": ranking_sha256,
                "row_count": int(len(ranking)),
                "eligible_row_count": int(eligible.sum()),
            }
        )
    manifest = pd.DataFrame(rows, columns=columns)
    if manifest.duplicated(["pair", "rebalance_date"]).any():
        raise ValueError("S5 ranking manifest contains duplicate pair-quarter keys")
    return manifest


def _ensemble_parameters(row: Mapping[str, Any]) -> list[Any]:
    path = _ranking_path(row)
    raw_ranking = path.read_bytes()
    if _S5_EXPECTED_SHA_BY_KEY is not None:
        key = (
            str(row["pair"]),
            str(pd.Timestamp(row["rebalance_date"]).date()),
        )
        expected_sha256 = _S5_EXPECTED_SHA_BY_KEY.get(key)
        actual_sha256 = hashlib.sha256(raw_ranking).hexdigest()
        if expected_sha256 is None or actual_sha256 != expected_sha256:
            raise RuntimeError(
                "S5 ranking hash mismatch for "
                f"{key[0]} {key[1]}: expected={expected_sha256}, "
                f"actual={actual_sha256}"
            )
    ranking = pd.read_csv(
        io.BytesIO(raw_ranking),
        compression="gzip" if path.suffix == ".gz" else None,
    )
    eligible = ranking.loc[
        ranking["eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
    ].copy()
    if eligible.empty:
        raise RuntimeError("training-eligible ensemble has no parameters")
    records = []
    for raw in eligible.to_dict("records"):
        raw["selected_config_id"] = raw["config_id"]
        records.append(parameter_from_schedule(raw))
    return records


def _carry_extension_holding_sessions(
    row: Mapping[str, Any], candidate: Mapping[str, Any]
) -> int:
    """Return enough post-quarter sessions for every selected path to flatten."""

    if candidate.get("ensemble"):
        parameters = _ensemble_parameters(row)
    else:
        parameters = [parameter_from_schedule(row)]
    if not parameters:
        raise RuntimeError("carry extension has no strategy parameters")
    extension_holding = max(
        int(parameter.maximum_holding_days) for parameter in parameters
    )
    half_life = float(row["screen_half_life_days"])
    if np.isfinite(half_life):
        extension_holding = max(extension_holding, int(math.ceil(half_life * 2.0)))
    return extension_holding


def _activation_run_count(daily: pd.DataFrame) -> int:
    """Count flat-to-active runs for the frozen causal-selection activity gate.

    This intentionally does not count active-to-active aggregate direction
    reversals.  Changing that historical gate would change strategy selection;
    the final reporting ledger uses directional episodes instead.
    """

    active = (
        pd.to_numeric(daily["target_weight_dependent"], errors="coerce").fillna(0).abs()
        + pd.to_numeric(daily["target_weight_independent"], errors="coerce").fillna(0).abs()
    ).gt(1e-12)
    return int((active & ~active.shift(1, fill_value=False)).sum())


def _directional_target_sides(
    daily: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.Series:
    """Return {-1, 0, +1} for the post-fill aggregate ticker-A side."""

    target_a = pd.to_numeric(daily["target_weight_a"], errors="coerce")
    target_b = pd.to_numeric(daily["target_weight_b"], errors="coerce")
    if not np.isfinite(target_a.to_numpy(dtype=float)).all() or not np.isfinite(
        target_b.to_numpy(dtype=float)
    ).all():
        raise ValueError("directional episode targets must be finite")
    active = target_a.abs().add(target_b.abs()).gt(tolerance)
    ambiguous = active & target_a.abs().le(tolerance)
    if ambiguous.any():
        raise ValueError(
            "active aggregate target has no unambiguous ticker-A direction"
        )
    sides = np.where(target_a.gt(tolerance), 1, np.where(target_a.lt(-tolerance), -1, 0))
    return pd.Series(sides, index=daily.index, dtype="int8", name="target_direction")


def _directional_episode_entries(
    daily: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> pd.Series:
    """Mark flat entries and active-to-active direction reversals."""

    side = _directional_target_sides(daily, tolerance=tolerance)
    prior = side.shift(1, fill_value=0).astype("int8")
    return side.ne(0) & side.ne(prior)


def _joined_execution_events(
    simulations: list[Any], column: str
) -> pd.Series:
    frame = pd.concat(
        [result.daily[column].astype("string") for result in simulations], axis=1
    )

    def join_row(values: pd.Series) -> str:
        unique = sorted({str(value) for value in values.dropna()})
        if not unique:
            return "ensemble_unavailable"
        if len(unique) == 1:
            return unique[0]
        return "ensemble[" + "|".join(unique) + "]"

    return frame.apply(join_row, axis=1)


def _add_ensemble_execution_audit(
    daily: pd.DataFrame,
    simulations: list[Any],
) -> pd.DataFrame:
    """Carry auditable member information onto an averaged ensemble path."""

    result = daily.copy()
    result["zscore"] = pd.concat(
        [item.daily["zscore"] for item in simulations], axis=1
    ).mean(axis=1)
    result["decision_event"] = _joined_execution_events(
        simulations, "decision_event"
    )
    result["fill_event"] = _joined_execution_events(simulations, "fill_event")
    result["fill_decision_date"] = pd.to_datetime(
        simulations[0].daily["fill_decision_date"], errors="coerce"
    )
    result["fill_decision_zscore"] = pd.concat(
        [item.daily["fill_decision_zscore"] for item in simulations], axis=1
    ).mean(axis=1)
    member_active = pd.concat(
        [
            (
                item.daily["target_weight_dependent"].abs()
                + item.daily["target_weight_independent"].abs()
            ).gt(1e-12)
            for item in simulations
        ],
        axis=1,
    )
    result["ensemble_active_member_count"] = member_active.sum(axis=1).astype(int)
    member_held = pd.concat(
        [
            (
                item.daily["weight_dependent"].abs()
                + item.daily["weight_independent"].abs()
            ).gt(1e-12)
            for item in simulations
        ],
        axis=1,
    )
    result["ensemble_held_member_count"] = member_held.sum(axis=1).astype(int)
    result["execution_audit_source"] = "ensemble_member_mean_and_joined_events"
    return result


def _simulate_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    train: pd.DatetimeIndex,
    evaluation: pd.DatetimeIndex,
    *,
    terminal: bool = True,
    entry_start: pd.Timestamp | None = None,
    entry_end_exclusive: pd.Timestamp | None = None,
    include_execution_audit: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _PRICES is None or _BASE is None:
        raise RuntimeError("worker is not initialized")
    common = {
        "screen_half_life_days": float(row["screen_half_life_days"]),
        "entry_start": entry_start,
        "entry_end_exclusive": entry_end_exclusive,
    }
    if candidate.get("ensemble"):
        simulations = [
            simulate_v7_pair(
                _definition(row),
                _PRICES,
                train,
                evaluation,
                parameter,
                _BASE,
                _rules(candidate, terminal=terminal),
                **common,
            )
            for parameter in _ensemble_parameters(row)
        ]
        daily, metrics = aggregate_ensemble(
            simulations,
            transaction_cost_bps=_BASE.transaction_cost_bps,
            annual_short_borrow_bps=_BASE.annual_short_borrow_bps,
        )
        # Preserve the original flat-to-active activity gate used by causal
        # strategy selection.  Directional reversals are counted only in the
        # final reporting ledger so this repair cannot alter frozen selection.
        metrics["trade_count"] = _activation_run_count(daily)
        metrics["selection_activity_count_definition"] = (
            "continuous_active_target_runs"
        )
        metrics["configuration_count"] = len(simulations)
        if include_execution_audit:
            daily = _add_ensemble_execution_audit(daily, simulations)
        return daily, metrics
    result = simulate_v7_pair(
        _definition(row),
        _PRICES,
        train,
        evaluation,
        parameter_from_schedule(row),
        _BASE,
        _rules(candidate, terminal=terminal),
        **common,
    )
    if include_execution_audit:
        daily = result.daily.copy()
        daily["ensemble_active_member_count"] = _active_target(daily).astype(int)
        daily["ensemble_held_member_count"] = (
            daily["weight_dependent"].abs()
            + daily["weight_independent"].abs()
        ).gt(1e-12).astype(int)
        daily["execution_audit_source"] = "direct_v7_simulation"
        return daily, result.metrics
    return result.daily, result.metrics


def _candidate_frame_matches(frame: pd.DataFrame, row: Mapping[str, Any]) -> bool:
    required = {
        "pair",
        "rebalance_date",
        "strategy_id",
        "q_return_1bp",
        "q_return_2bp",
        "trade_count",
        "selection_activity_count",
        "selection_activity_count_definition",
        "annualized_turnover",
        "max_drawdown",
        "configuration_count",
        "candidate_available",
    }
    if not required.issubset(frame.columns) or len(frame) != len(_CANDIDATES):
        return False
    if frame["strategy_id"].astype(str).duplicated().any():
        return False
    if set(frame["strategy_id"].astype(str)) != {
        str(candidate["id"]) for candidate in _CANDIDATES
    }:
        return False
    if set(frame["pair"].astype(str)) != {str(row["pair"])}:
        return False
    dates = pd.to_datetime(frame["rebalance_date"], errors="coerce")
    if not bool(
        dates.notna().all()
        and dates.eq(pd.Timestamp(row["rebalance_date"])).all()
    ):
        return False
    if not frame["selection_activity_count_definition"].astype(str).eq(
        "continuous_active_target_runs"
    ).all():
        return False
    count = pd.to_numeric(frame["selection_activity_count"], errors="coerce")
    alias = pd.to_numeric(frame["trade_count"], errors="coerce")
    configurations = pd.to_numeric(frame["configuration_count"], errors="coerce")
    if not bool(
        np.isfinite(count).all()
        and (count >= 0.0).all()
        and np.allclose(count, np.rint(count), rtol=0.0, atol=1e-12)
        and np.array_equal(count.to_numpy(dtype=float), alias.to_numpy(dtype=float))
        and np.isfinite(configurations).all()
        and (configurations >= 1.0).all()
        and np.allclose(
            configurations, np.rint(configurations), rtol=0.0, atol=1e-12
        )
    ):
        return False
    availability_text = frame["candidate_available"].astype(str).str.strip().str.lower()
    if not availability_text.isin({"true", "false", "1", "0", "yes", "no"}).all():
        return False
    available = availability_text.isin({"true", "1", "yes"})
    economics = frame[
        ["q_return_1bp", "q_return_2bp", "annualized_turnover", "max_drawdown"]
    ].apply(pd.to_numeric, errors="coerce")
    if available.any():
        live = economics.loc[available]
        if not bool(
            np.isfinite(live.to_numpy(dtype=float)).all()
            and live["q_return_1bp"].gt(-1.0).all()
            and live["q_return_2bp"].gt(-1.0).all()
            and live["annualized_turnover"].ge(0.0).all()
            and live["max_drawdown"].between(-1.0, 1e-12).all()
        ):
            return False
    if (~available).any():
        unavailable = economics.loc[~available]
        if not bool(
            unavailable.isna().all(axis=None)
            and count.loc[~available].eq(0.0).all()
        ):
            return False
    return True


def _candidate_task(
    payload: tuple[dict[str, Any], str, dict[str, Any]]
) -> dict[str, Any]:
    row, cache_path_text, lineage = payload
    fingerprint = str(lineage["fingerprint"])
    payload_sha256 = str(lineage["payload_sha256"])
    run_nonce = str(lineage["run_nonce"])
    cache = Path(cache_path_text)
    complete = cache / "complete.json"
    metrics_path = cache / "candidate_metrics.csv"
    if _cache_marker_matches(
        complete,
        metrics_path,
        fingerprint=fingerprint,
        pair=str(row["pair"]),
        rebalance_date=row["rebalance_date"],
        payload_sha256=payload_sha256,
        artifact_date_column="rebalance_date",
    ):
        cached = pd.read_csv(metrics_path)
        if _candidate_frame_matches(cached, row):
            marker = json.loads(complete.read_text(encoding="utf-8"))
            return {
                "cached": True,
                "path": str(metrics_path),
                "marker_path": str(complete),
                "cache_stage": "candidate",
                "run_nonce": str(marker["run_nonce"]),
                "fingerprint": fingerprint,
                "payload_sha256": payload_sha256,
                "pair": row["pair"],
                "rebalance_date": row["rebalance_date"],
                "artifact_date_column": "rebalance_date",
            }
    train, evaluation = _indexes(row)
    rows: list[dict[str, Any]] = []
    for candidate in _CANDIDATES:
        half_life = float(row["screen_half_life_days"])
        if candidate.get("half_life_multiplier") is not None and not np.isfinite(half_life):
            rows.append(
                {
                    "pair": row["pair"],
                    "sector": row["sector"],
                    "rebalance_date": row["rebalance_date"],
                    "outer_end_exclusive": row["outer_end_exclusive"],
                    "strategy_id": candidate["id"],
                    "complexity": candidate["complexity"],
                    "q_return_1bp": math.nan,
                    "q_return_2bp": math.nan,
                    "trade_count": 0,
                    "selection_activity_count": 0,
                    "selection_activity_count_definition": (
                        "continuous_active_target_runs"
                    ),
                    "annualized_turnover": math.nan,
                    "max_drawdown": math.nan,
                    "configuration_count": 1,
                    "selected_v6_config_id": row["selected_config_id"],
                    "candidate_available": False,
                    "unavailable_reason": "nonfinite_training_half_life",
                }
            )
            continue
        daily, metrics = _simulate_candidate(row, candidate, train, evaluation)
        return_1bp = reprice_returns(daily, 1.0)
        return_2bp = reprice_returns(daily, 2.0)
        rows.append(
            {
                "pair": row["pair"],
                "sector": row["sector"],
                "rebalance_date": row["rebalance_date"],
                "outer_end_exclusive": row["outer_end_exclusive"],
                "strategy_id": candidate["id"],
                "complexity": candidate["complexity"],
                "q_return_1bp": float((1.0 + return_1bp).prod() - 1.0),
                "q_return_2bp": float((1.0 + return_2bp).prod() - 1.0),
                "trade_count": int(metrics["trade_count"]),
                "selection_activity_count": int(metrics["trade_count"]),
                "selection_activity_count_definition": (
                    "continuous_active_target_runs"
                ),
                "annualized_turnover": float(daily["turnover"].mean() * 252.0),
                "max_drawdown": float(performance_metrics(return_2bp)["max_drawdown"]),
                "configuration_count": int(metrics.get("configuration_count", 1)),
                "selected_v6_config_id": row["selected_config_id"],
                "candidate_available": True,
                "unavailable_reason": pd.NA,
            }
        )
    output = pd.DataFrame(rows)
    if not _candidate_frame_matches(output, row):
        raise RuntimeError(
            f"candidate output failed schema/key validation for {row['pair']} "
            f"{pd.Timestamp(row['rebalance_date']).date()}"
        )
    _atomic_csv(output, metrics_path)
    _atomic_json(
        _cache_marker_payload(
            fingerprint=fingerprint,
            artifact=metrics_path,
            artifact_rows=len(output),
            pair=str(row["pair"]),
            rebalance_date=row["rebalance_date"],
            payload_sha256=payload_sha256,
            artifact_date_column="rebalance_date",
            candidate_count=len(output),
            row_payload_sha256=lineage["row_payload_sha256"],
            s5_parameter_rankings_sha256=lineage[
                "s5_parameter_rankings_sha256"
            ],
            cache_stage="candidate",
            run_nonce=run_nonce,
        ),
        complete,
    )
    return {
        "cached": False,
        "path": str(metrics_path),
        "marker_path": str(complete),
        "cache_stage": "candidate",
        "run_nonce": run_nonce,
        "fingerprint": fingerprint,
        "payload_sha256": payload_sha256,
        "pair": row["pair"],
        "rebalance_date": row["rebalance_date"],
        "artifact_date_column": "rebalance_date",
    }


def _fingerprint(
    config_path: Path,
    source_paths: Iterable[Path],
    *,
    engine_manifest: Mapping[str, str],
    s5_ranking_manifest_sha256: str,
    runtime_manifest: Mapping[str, Any] | None = None,
    config_sha256: str | None = None,
    source_sha256_by_path: Mapping[str, str] | None = None,
) -> str:
    payload = {
        "schema": "v8_candidate_engine_manifest_v5",
        "config": {
            "path": str(config_path),
            "sha256": config_sha256 or _sha256(config_path),
        },
        "sources": {
            str(path): (
                source_sha256_by_path[str(path)]
                if source_sha256_by_path is not None
                else _sha256(path)
            )
            for path in sorted(source_paths, key=str)
        },
        "engine_sources": dict(sorted(engine_manifest.items())),
        "runtime_manifest": dict(runtime_manifest or _runtime_manifest()),
        "s5_parameter_ranking_manifest_sha256": s5_ranking_manifest_sha256,
    }
    return _canonical_payload_sha256(payload)


def _build_strategy_selection(
    quarterly: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = config["strategy_selection"]
    candidates = config["strategy_candidates"]
    selected_rows: list[dict[str, Any]] = []
    ranking_rows: list[pd.DataFrame] = []
    candidate_metrics = candidate_metrics.copy()
    candidate_metrics["rebalance_date"] = pd.to_datetime(candidate_metrics["rebalance_date"])
    for pair, pair_quarters in quarterly.groupby("pair", sort=True):
        history = candidate_metrics.loc[candidate_metrics["pair"].astype(str).eq(str(pair))]
        for raw in pair_quarters.sort_values("rebalance_date").to_dict("records"):
            chosen, ranking, audit = select_strategy_one_standard_error(
                history,
                candidates,
                current_rebalance=pd.Timestamp(raw["rebalance_date"]),
                lookback_calendar_years=int(selection["lookback_calendar_years"]),
                minimum_history_quarters=int(selection["minimum_history_quarters"]),
                minimum_active_quarters=int(selection["minimum_active_quarters"]),
                minimum_trades=int(selection["minimum_trades"]),
                default_strategy_id=str(selection["insufficient_history_default"]),
            )
            current_candidate = history.loc[
                history["rebalance_date"].eq(pd.Timestamp(raw["rebalance_date"]))
                & history["strategy_id"].astype(str).eq(chosen)
            ]
            if (
                _bool(raw["evaluation_success"])
                and len(current_candidate)
                and not _bool(current_candidate.iloc[0].get("candidate_available", True))
            ):
                chosen = str(selection["insufficient_history_default"])
                audit["selected_strategy_id"] = chosen
                audit["selection_reason"] = "selected_candidate_unavailable_from_current_training"
                ranking["selected"] = ranking["strategy_id"].eq(chosen)
            selected_rows.append(
                {
                    "pair": pair,
                    "sector": raw["sector"],
                    "rebalance_date": raw["rebalance_date"],
                    "outer_end_exclusive": raw["outer_end_exclusive"],
                    "evaluation_success": _bool(raw["evaluation_success"]),
                    "available": _bool(raw["available"]),
                    "selected_strategy_id": chosen,
                    **audit,
                }
            )
            ranking.insert(0, "pair", pair)
            ranking.insert(1, "rebalance_date", raw["rebalance_date"])
            ranking_rows.append(ranking)
    return pd.DataFrame(selected_rows), pd.concat(ranking_rows, ignore_index=True)


def _active_target(daily: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(daily["target_weight_dependent"], errors="coerce").fillna(0).abs()
        + pd.to_numeric(daily["target_weight_independent"], errors="coerce").fillna(0).abs()
    ).gt(1e-12)


def _carry_state_active(daily: pd.DataFrame) -> pd.Series:
    if "ensemble_active_member_count" not in daily:
        return _active_target(daily)
    counts = pd.to_numeric(
        daily["ensemble_active_member_count"], errors="coerce"
    )
    values = counts.to_numpy(dtype=float)
    if not bool(
        np.isfinite(values).all()
        and (values >= 0.0).all()
        and np.allclose(values, np.rint(values), rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("ensemble active-member count must be finite nonnegative integers")
    active = counts.gt(0.0)
    if bool((_active_target(daily) & ~active).any()):
        raise RuntimeError("active aggregate target has no active ensemble member")
    return active


def _truncate_vintage_after_carry(
    daily: pd.DataFrame,
    *,
    outer_end: pd.Timestamp,
    pair: str,
    rebalance_date: Any,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    regular_rows = daily.index < outer_end
    if not bool(regular_rows.any()):
        return daily.iloc[0:0], None
    carry_state = _carry_state_active(daily)
    carry_open = bool(carry_state.loc[regular_rows].iloc[-1])
    if not carry_open:
        return daily.loc[regular_rows], None
    post = daily.loc[daily.index >= outer_end]
    exits = post.index[~_carry_state_active(post)]
    if not len(exits):
        raise RuntimeError(
            f"{pair} {pd.Timestamp(rebalance_date).date()} active carry has no flat exit "
            "inside the configured extension"
        )
    carry_exit = pd.Timestamp(exits[0])
    return daily.loc[daily.index <= carry_exit], carry_exit


def _final_frame_matches(frame: pd.DataFrame, pair: str) -> bool:
    required = {
        "date",
        "pair",
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "net_return",
        "net_return_1bp",
        "net_return_2bp",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "ensemble_active_member_count",
        "ensemble_held_member_count",
    }
    if not required.issubset(frame.columns):
        return False
    if frame.empty:
        return True
    if set(frame["pair"].astype(str)) != {str(pair)}:
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if not bool(
        dates.notna().all()
        and not dates.duplicated().any()
        and dates.is_monotonic_increasing
    ):
        return False
    economic = frame[
        [
            "gross_return",
            "turnover",
            "transaction_cost",
            "borrow_cost",
            "net_return",
            "net_return_1bp",
            "net_return_2bp",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not bool(np.isfinite(economic).all()):
        return False
    turnover = economic[:, 1]
    transaction_cost = economic[:, 2]
    borrow_cost = economic[:, 3]
    net_returns = economic[:, 4:]
    if bool(
        (turnover < -1e-12).any()
        or (transaction_cost < -1e-12).any()
        or (borrow_cost < -1e-12).any()
        or (net_returns <= -1.0).any()
    ):
        return False
    weights = frame[["weight_a", "weight_b"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    targets = frame[["target_weight_a", "target_weight_b"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    member_counts = pd.to_numeric(
        frame["ensemble_active_member_count"], errors="coerce"
    ).to_numpy(dtype=float)
    held_member_counts = pd.to_numeric(
        frame["ensemble_held_member_count"], errors="coerce"
    ).to_numpy(dtype=float)
    if not bool(
        np.isfinite(weights).all()
        and np.isfinite(targets).all()
        and np.isfinite(member_counts).all()
        and np.isfinite(held_member_counts).all()
        and (member_counts >= 0.0).all()
        and (held_member_counts >= 0.0).all()
        and np.allclose(
            member_counts, np.rint(member_counts), rtol=0.0, atol=1e-12
        )
        and np.allclose(
            held_member_counts,
            np.rint(held_member_counts),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        return False
    target_active = np.abs(targets).sum(axis=1) > 1e-12
    if bool(np.any(target_active & (member_counts <= 0.0))):
        return False
    held_active = np.abs(weights).sum(axis=1) > 1e-12
    if bool(np.any(held_active & (held_member_counts <= 0.0))):
        return False
    expected = np.vstack([np.zeros((1, 2), dtype=float), targets[:-1]])
    expected_held_member_counts = np.r_[0.0, member_counts[:-1]]
    return bool(
        np.allclose(weights, expected, rtol=0.0, atol=1e-12)
        and np.allclose(
            held_member_counts,
            expected_held_member_counts,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(targets[-1], 0.0, rtol=0.0, atol=1e-12)
        and member_counts[-1] == 0.0
    )


def _final_pair_task(
    payload: tuple[list[dict[str, Any]], str, dict[str, Any]]
) -> dict[str, Any]:
    rows, cache_path_text, lineage = payload
    fingerprint = str(lineage["fingerprint"])
    payload_sha256 = str(lineage["payload_sha256"])
    run_nonce = str(lineage["run_nonce"])
    cache = Path(cache_path_text)
    complete = cache / "complete.json"
    daily_path = cache / "daily.csv.gz"
    pair = str(rows[0]["pair"])
    if _cache_marker_matches(
        complete,
        daily_path,
        fingerprint=fingerprint,
        pair=pair,
        payload_sha256=payload_sha256,
        artifact_date_column="date",
    ):
        cached = pd.read_csv(daily_path)
        if _final_frame_matches(cached, pair):
            marker = json.loads(complete.read_text(encoding="utf-8"))
            return {
                "cached": True,
                "path": str(daily_path),
                "marker_path": str(complete),
                "cache_stage": "stateful_pair",
                "run_nonce": str(marker["run_nonce"]),
                "fingerprint": fingerprint,
                "payload_sha256": payload_sha256,
                "pair": rows[0]["pair"],
                "artifact_date_column": "date",
            }
    if _PRICES is None:
        raise RuntimeError("worker is not initialized")
    candidate_map = {str(row["id"]): row for row in _CANDIDATES}
    blocked_until: pd.Timestamp | None = None
    pieces: list[pd.DataFrame] = []
    for row in sorted(rows, key=lambda value: pd.Timestamp(value["rebalance_date"])):
        if not _bool(row["evaluation_success"]):
            continue
        train, regular = _indexes(row)
        outer_end = pd.Timestamp(row["outer_end_exclusive"])
        candidate = candidate_map[str(row["selected_strategy_id"])]
        extension_holding = _carry_extension_holding_sessions(row, candidate)
        extension = _PRICES.index[_PRICES.index >= outer_end][: extension_holding + 2]
        evaluation = regular.union(extension).sort_values()
        entry_start = regular[0]
        prior_blocked_until = blocked_until
        if prior_blocked_until is not None:
            later = evaluation[evaluation > prior_blocked_until]
            entry_start = later[0] if len(later) else outer_end
        terminal = bool(evaluation[-1] == _PRICES.index[_PRICES.index < pd.Timestamp("2026-08-27")][-1])
        daily, _ = _simulate_candidate(
            row,
            candidate,
            train,
            evaluation,
            terminal=terminal,
            entry_start=entry_start,
            entry_end_exclusive=outer_end,
            include_execution_audit=True,
        )
        daily, carry_exit = _truncate_vintage_after_carry(
            daily,
            outer_end=outer_end,
            pair=str(row["pair"]),
            rebalance_date=row["rebalance_date"],
        )
        if carry_exit is not None:
            blocked_until = carry_exit
        elif prior_blocked_until is not None and prior_blocked_until >= outer_end:
            blocked_until = prior_blocked_until
        else:
            blocked_until = None
        output = daily.reset_index()
        output.insert(0, "pair", row["pair"])
        output.insert(1, "sector", row["sector"])
        output.insert(2, "vintage_rebalance_date", row["rebalance_date"])
        output.insert(3, "strategy_id", row["selected_strategy_id"])
        definition = _definition(row)
        output["dependent"] = definition["dependent"]
        output["independent"] = definition["independent"]
        if definition["dependent"] == str(row["ticker_a"]):
            output["weight_a"] = output["weight_dependent"]
            output["weight_b"] = output["weight_independent"]
            output["target_weight_a"] = output["target_weight_dependent"]
            output["target_weight_b"] = output["target_weight_independent"]
        else:
            output["weight_a"] = output["weight_independent"]
            output["weight_b"] = output["weight_dependent"]
            output["target_weight_a"] = output["target_weight_independent"]
            output["target_weight_b"] = output["target_weight_dependent"]
        output["ticker_a"] = row["ticker_a"]
        output["ticker_b"] = row["ticker_b"]
        output["price_a"] = _PRICES.loc[output["date"], str(row["ticker_a"])].to_numpy()
        output["price_b"] = _PRICES.loc[output["date"], str(row["ticker_b"])].to_numpy()
        pieces.append(
            output[
                [
                    "date",
                    "pair",
                    "sector",
                    "strategy_id",
                    "vintage_rebalance_date",
                    "dependent",
                    "independent",
                    "ticker_a",
                    "ticker_b",
                    "price_a",
                    "price_b",
                    "zscore",
                    "decision_event",
                    "fill_event",
                    "fill_decision_date",
                    "fill_decision_zscore",
                    "execution_audit_source",
                    "gross_return",
                    "turnover",
                    "transaction_cost",
                    "borrow_cost",
                    "net_return",
                    "weight_dependent",
                    "weight_independent",
                    "target_weight_dependent",
                    "target_weight_independent",
                    "weight_a",
                    "weight_b",
                    "target_weight_a",
                    "target_weight_b",
                    "ensemble_active_member_count",
                    "ensemble_held_member_count",
                ]
            ]
        )
    if pieces:
        stacked = pd.concat(pieces, ignore_index=True, sort=False)
        stacked["date"] = pd.to_datetime(stacked["date"])
        numeric = [
            "gross_return",
            "turnover",
            "transaction_cost",
            "borrow_cost",
            "net_return",
            "weight_dependent",
            "weight_independent",
            "target_weight_dependent",
            "target_weight_independent",
            "weight_a",
            "weight_b",
            "target_weight_a",
            "target_weight_b",
            "ensemble_active_member_count",
            "ensemble_held_member_count",
        ]
        overlap = stacked.assign(active=_carry_state_active(stacked)).groupby("date")["active"].sum()
        if int(overlap.max()) > 1:
            raise RuntimeError(f"{rows[0]['pair']} contains overlapping same-pair positions")
        aggregated = stacked.groupby("date", as_index=False)[numeric].sum()
        stacked["_activity_score"] = (
            stacked["weight_dependent"].abs()
            + stacked["weight_independent"].abs()
            + stacked["target_weight_dependent"].abs()
            + stacked["target_weight_independent"].abs()
            + stacked["turnover"].abs()
            + stacked["ensemble_active_member_count"].abs()
            + stacked["ensemble_held_member_count"].abs()
        )
        label_columns = [
            "date",
            "pair",
            "sector",
            "strategy_id",
            "vintage_rebalance_date",
            "ticker_a",
            "ticker_b",
            "price_a",
            "price_b",
            "zscore",
            "decision_event",
            "fill_event",
            "fill_decision_date",
            "fill_decision_zscore",
            "execution_audit_source",
        ]
        labels = (
            stacked.sort_values(
                ["date", "_activity_score", "vintage_rebalance_date"],
                ascending=[True, False, False],
                kind="mergesort",
            )[label_columns]
            .drop_duplicates("date", keep="first")
        )
        final = labels.merge(aggregated, on="date", validate="one_to_one")
    else:
        final = pd.DataFrame(
            columns=[
                "date",
                "pair",
                "sector",
                "strategy_id",
                "vintage_rebalance_date",
                "gross_return",
                "turnover",
                "transaction_cost",
                "borrow_cost",
                "net_return",
                "weight_dependent",
                "weight_independent",
                "target_weight_dependent",
                "target_weight_independent",
                "weight_a",
                "weight_b",
                "target_weight_a",
                "target_weight_b",
                "ensemble_active_member_count",
                "ensemble_held_member_count",
                "ticker_a",
                "ticker_b",
                "price_a",
                "price_b",
                "zscore",
                "decision_event",
                "fill_event",
                "fill_decision_date",
                "fill_decision_zscore",
                "execution_audit_source",
                "net_return_1bp",
                "net_return_2bp",
            ]
        )
    if len(final):
        final["net_return_1bp"] = reprice_returns(final, 1.0)
        final["net_return_2bp"] = reprice_returns(final, 2.0)
    if not _final_frame_matches(final, pair):
        raise RuntimeError(f"final pair output failed schema/key validation for {pair}")
    _atomic_csv(final, daily_path)
    _atomic_json(
        _cache_marker_payload(
            fingerprint=fingerprint,
            artifact=daily_path,
            artifact_rows=len(final),
            pair=pair,
            payload_sha256=payload_sha256,
            artifact_date_column="date",
            daily_rows=len(final),
            merged_group_sha256=lineage["merged_group_sha256"],
            cache_stage="stateful_pair",
            run_nonce=run_nonce,
        ),
        complete,
    )
    return {
        "cached": False,
        "path": str(daily_path),
        "marker_path": str(complete),
        "cache_stage": "stateful_pair",
        "run_nonce": run_nonce,
        "fingerprint": fingerprint,
        "payload_sha256": payload_sha256,
        "pair": rows[0]["pair"],
        "artifact_date_column": "date",
    }


def _run_parallel(
    tasks: list[Any],
    worker: Any,
    *,
    workers: int,
    initializer_args: tuple[Any, ...],
    label: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers == 1:
        _worker_init(*initializer_args)
        for number, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if number % 25 == 0 or number == len(tasks):
                print(f"{label}: {number}/{len(tasks)}", flush=True)
        return results
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=initializer_args,
    ) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if number % 100 == 0 or number == len(tasks):
                print(f"{label}: {number}/{len(tasks)}", flush=True)
    return results


def _calendar_quarters(
    dates: pd.DatetimeIndex, schedule: pd.DataFrame
) -> pd.Series:
    result = pd.Series(index=dates, dtype="int64")
    for row in schedule.itertuples(index=False):
        mask = (dates >= row.rebalance_date) & (dates < row.outer_end_exclusive)
        result.loc[mask] = int(row.quarter_number)
    if result.isna().any():
        raise RuntimeError("study calendar does not map completely to quarter schedule")
    return result.astype(int)


def _trade_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate directional episodes with exact same-close flip attribution.

    On a direction reversal, row-t holding P&L and borrow belong to the old
    episode.  Its drifted close cost also belongs to the old episode, while the
    new target's opening cost belongs to the new episode.  Two sequential
    sub-factors preserve the recorded daily return exactly without assigning
    the whole reversal row to both trades.
    """

    if daily.empty:
        return pd.DataFrame()
    work = daily.reset_index(drop=True).copy()
    dates = pd.to_datetime(work["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("trade episode dates must be valid and increasing")
    tolerance = 1e-12
    sides = _directional_target_sides(work, tolerance=tolerance).to_numpy(dtype=int)
    held_a = pd.to_numeric(work["weight_a"], errors="coerce").to_numpy(dtype=float)
    held_b = pd.to_numeric(work["weight_b"], errors="coerce").to_numpy(dtype=float)
    target_a = pd.to_numeric(work["target_weight_a"], errors="coerce").to_numpy(dtype=float)
    target_b = pd.to_numeric(work["target_weight_b"], errors="coerce").to_numpy(dtype=float)
    price_a = pd.to_numeric(work["price_a"], errors="coerce").to_numpy(dtype=float)
    price_b = pd.to_numeric(work["price_b"], errors="coerce").to_numpy(dtype=float)
    gross = pd.to_numeric(work["gross_return"], errors="coerce").to_numpy(dtype=float)
    turnover = pd.to_numeric(work["turnover"], errors="coerce").to_numpy(dtype=float)
    borrow = pd.to_numeric(work["borrow_cost"], errors="coerce").to_numpy(dtype=float)
    net_1bp = pd.to_numeric(work["net_return_1bp"], errors="coerce").to_numpy(dtype=float)
    net_2bp = pd.to_numeric(work["net_return_2bp"], errors="coerce").to_numpy(dtype=float)
    finite = (held_a, held_b, target_a, target_b, gross, turnover, borrow, net_1bp, net_2bp)
    if not all(np.isfinite(values).all() for values in finite):
        raise ValueError("trade episode economic inputs must be finite")
    return_a = pd.Series(price_a).pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)
    return_b = pd.Series(price_b).pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)

    held_active = (np.abs(held_a) + np.abs(held_b)) > tolerance
    ambiguous_held = held_active & (np.abs(held_a) <= tolerance)
    if ambiguous_held.any():
        raise ValueError("active held aggregate has no unambiguous ticker-A direction")
    held_sides = np.where(held_a > tolerance, 1, np.where(held_a < -tolerance, -1, 0))
    expected_held_sides = np.r_[0, sides[:-1]]
    if not np.array_equal(held_sides, expected_held_sides):
        raise ValueError("held direction does not equal the prior-close target direction")

    convention = "directional_episode_sequential_flip_factor_v1"
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    next_number = 1

    def start_episode(location: int, transition: str) -> dict[str, Any]:
        nonlocal next_number
        direction = int(sides[location])
        if direction == 0:
            raise RuntimeError("cannot start a flat directional episode")
        result: dict[str, Any] = {
            "pair": work["pair"].iloc[0],
            "trade_number": next_number,
            "strategy_id": work["strategy_id"].iloc[location],
            "entry_decision_date": work["fill_decision_date"].iloc[location],
            "entry_date": dates.iloc[location],
            "entry_fill_price_a": float(price_a[location]),
            "entry_fill_price_b": float(price_b[location]),
            "entry_zscore": float(work["fill_decision_zscore"].iloc[location]),
            "entry_transition": transition,
            "entry_fill_event": str(work["fill_event"].iloc[location]),
            "exit_decision_date": pd.NaT,
            "exit_date": dates.iloc[-1],
            "exit_fill_price_a": math.nan,
            "exit_fill_price_b": math.nan,
            "exit_zscore": math.nan,
            "exit_reason": "end_of_study_open",
            "exit_transition": "end_of_study_open",
            "exit_fill_event": "end_of_study_open",
            "direction_ticker_a": "long" if direction > 0 else "short",
            "trading_sessions": 0,
            "closed": False,
            "gross_return_sum": 0.0,
            "borrow_cost": 0.0,
            "opening_transaction_cost_1bp": 0.0,
            "closing_transaction_cost_1bp": 0.0,
            "holding_rebalance_transaction_cost_1bp": 0.0,
            "opening_transaction_cost_2bp": 0.0,
            "closing_transaction_cost_2bp": 0.0,
            "holding_rebalance_transaction_cost_2bp": 0.0,
            "attribution_convention": convention,
            "_factors_1bp": [],
            "_factors_2bp": [],
        }
        next_number += 1
        return result

    def add_economics(
        episode: dict[str, Any],
        location: int,
        factor_1bp: float,
        factor_2bp: float,
        *,
        include_holding: bool,
    ) -> None:
        episode["_factors_1bp"].append(float(factor_1bp))
        episode["_factors_2bp"].append(float(factor_2bp))
        if include_holding:
            episode["gross_return_sum"] += float(gross[location])
            episode["borrow_cost"] += float(borrow[location])
            episode["trading_sessions"] += 1

    def close_episode(
        episode: dict[str, Any], location: int, transition: str
    ) -> None:
        episode["exit_decision_date"] = work["fill_decision_date"].iloc[location]
        episode["exit_date"] = dates.iloc[location]
        episode["exit_fill_price_a"] = float(price_a[location])
        episode["exit_fill_price_b"] = float(price_b[location])
        episode["exit_zscore"] = float(work["fill_decision_zscore"].iloc[location])
        episode["exit_reason"] = (
            "aggregate_direction_reversal"
            if transition == "direction_reversal"
            else str(work["fill_event"].iloc[location])
        )
        episode["exit_transition"] = transition
        episode["exit_fill_event"] = str(work["fill_event"].iloc[location])
        episode["closed"] = True

    def finish_episode(episode: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in episode.items() if not key.startswith("_")}
        factors_1bp = np.asarray(episode["_factors_1bp"], dtype=float)
        factors_2bp = np.asarray(episode["_factors_2bp"], dtype=float)
        result["net_return_1bp"] = float(np.prod(1.0 + factors_1bp) - 1.0)
        result["net_return_2bp"] = float(np.prod(1.0 + factors_2bp) - 1.0)
        result["transaction_cost_1bp"] = float(
            result["opening_transaction_cost_1bp"]
            + result["closing_transaction_cost_1bp"]
            + result["holding_rebalance_transaction_cost_1bp"]
        )
        result["transaction_cost_2bp"] = float(
            result["opening_transaction_cost_2bp"]
            + result["closing_transaction_cost_2bp"]
            + result["holding_rebalance_transaction_cost_2bp"]
        )
        return result

    for location in range(len(work)):
        prior_side = int(expected_held_sides[location])
        target_side = int(sides[location])
        denominator = 1.0 + float(gross[location])
        if not np.isfinite(denominator) or denominator <= tolerance:
            raise ValueError("directional episode encountered non-positive post-return NAV")
        total_cost_1bp = float(turnover[location]) / 10_000.0 * denominator
        total_cost_2bp = 2.0 * float(turnover[location]) / 10_000.0 * denominator

        if prior_side == 0 and target_side == 0:
            if max(
                abs(float(gross[location])),
                abs(float(borrow[location])),
                abs(total_cost_2bp),
                abs(float(net_2bp[location])),
            ) > 1e-10:
                raise ValueError("flat directional row contains non-cash economics")
            continue

        if prior_side == 0 and target_side != 0:
            if current is not None:
                raise RuntimeError("flat-to-active transition found an open episode")
            current = start_episode(location, "flat_to_active")
            current["opening_transaction_cost_1bp"] += total_cost_1bp
            current["opening_transaction_cost_2bp"] += total_cost_2bp
            add_economics(
                current,
                location,
                float(net_1bp[location]),
                float(net_2bp[location]),
                include_holding=False,
            )
            continue

        if current is None:
            raise RuntimeError("active held state has no open directional episode")

        if target_side == prior_side:
            current["holding_rebalance_transaction_cost_1bp"] += total_cost_1bp
            current["holding_rebalance_transaction_cost_2bp"] += total_cost_2bp
            add_economics(
                current,
                location,
                float(net_1bp[location]),
                float(net_2bp[location]),
                include_holding=True,
            )
            continue

        if target_side == 0:
            current["closing_transaction_cost_1bp"] += total_cost_1bp
            current["closing_transaction_cost_2bp"] += total_cost_2bp
            add_economics(
                current,
                location,
                float(net_1bp[location]),
                float(net_2bp[location]),
                include_holding=True,
            )
            close_episode(current, location, "active_to_flat")
            rows.append(finish_episode(current))
            current = None
            continue

        # Same-close active-to-active direction reversal.  Positive raw prices
        # and a positive portfolio denominator preserve each held leg's sign,
        # so both new target legs must be strictly opposite the drifted legs.
        if location == 0 or not all(
            np.isfinite(value)
            for value in (price_a[location - 1], price_b[location - 1], price_a[location], price_b[location])
        ):
            raise ValueError("direction reversal requires adjacent finite prices")
        drifted_a = held_a[location] * (1.0 + return_a[location]) / denominator
        drifted_b = held_b[location] * (1.0 + return_b[location]) / denominator
        if not (
            drifted_a * target_a[location] < -(tolerance**2)
            and drifted_b * target_b[location] < -(tolerance**2)
        ):
            raise ValueError("direction reversal must reverse both pair legs")
        close_turnover = abs(drifted_a) + abs(drifted_b)
        open_turnover = abs(target_a[location]) + abs(target_b[location])
        if not math.isclose(
            close_turnover + open_turnover,
            float(turnover[location]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("direction reversal turnover cannot be split exactly")
        close_cost_1bp = close_turnover / 10_000.0 * denominator
        open_cost_1bp = open_turnover / 10_000.0 * denominator
        close_cost_2bp = 2.0 * close_cost_1bp
        open_cost_2bp = 2.0 * open_cost_1bp
        old_factor_1bp = float(gross[location] - borrow[location] - close_cost_1bp)
        old_factor_2bp = float(gross[location] - borrow[location] - close_cost_2bp)
        if 1.0 + old_factor_1bp <= tolerance or 1.0 + old_factor_2bp <= tolerance:
            raise ValueError("direction reversal old-episode factor is non-positive")
        new_factor_1bp = float(-open_cost_1bp / (1.0 + old_factor_1bp))
        new_factor_2bp = float(-open_cost_2bp / (1.0 + old_factor_2bp))
        combined_1bp = (1.0 + old_factor_1bp) * (1.0 + new_factor_1bp) - 1.0
        combined_2bp = (1.0 + old_factor_2bp) * (1.0 + new_factor_2bp) - 1.0
        if not (
            math.isclose(combined_1bp, float(net_1bp[location]), rel_tol=0.0, abs_tol=1e-10)
            and math.isclose(combined_2bp, float(net_2bp[location]), rel_tol=0.0, abs_tol=1e-10)
        ):
            raise ValueError("direction reversal factors do not reproduce daily net return")

        current["closing_transaction_cost_1bp"] += close_cost_1bp
        current["closing_transaction_cost_2bp"] += close_cost_2bp
        add_economics(
            current,
            location,
            old_factor_1bp,
            old_factor_2bp,
            include_holding=True,
        )
        close_episode(current, location, "direction_reversal")
        rows.append(finish_episode(current))

        current = start_episode(location, "direction_reversal")
        current["opening_transaction_cost_1bp"] += open_cost_1bp
        current["opening_transaction_cost_2bp"] += open_cost_2bp
        add_economics(
            current,
            location,
            new_factor_1bp,
            new_factor_2bp,
            include_holding=False,
        )

    if current is not None:
        rows.append(finish_episode(current))
    return pd.DataFrame(rows)


def _aggregate_paths(
    master: pd.DataFrame,
    schedule: pd.DataFrame,
    quarterly: pd.DataFrame,
    sparse: pd.DataFrame,
    prices: pd.DataFrame,
    selections: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    study = config["evaluation"]
    start = pd.Timestamp(study["study_start_inclusive"])
    end = pd.Timestamp(study["study_end_exclusive"])
    years = exact_calendar_years(start, end)
    dates = prices.index[(prices.index >= start) & (prices.index < end)]
    quarter_by_date = _calendar_quarters(dates, schedule)
    selection_key = selections.copy()
    selection_key["rebalance_date"] = pd.to_datetime(selection_key["rebalance_date"])
    quarter_meta = schedule[["quarter_number", "rebalance_date", "outer_end_exclusive"]].copy()
    quarterly = quarterly.copy()
    quarterly["rebalance_date"] = pd.to_datetime(quarterly["rebalance_date"])
    quarterly["outer_end_exclusive"] = pd.to_datetime(quarterly["outer_end_exclusive"])
    quarterly = quarterly.merge(
        quarter_meta,
        on=["rebalance_date", "outer_end_exclusive"],
        how="left",
        validate="many_to_one",
    ).merge(
        selection_key[["pair", "rebalance_date", "selected_strategy_id", "selection_reason"]],
        on=["pair", "rebalance_date"],
        how="left",
        validate="one_to_one",
    )
    sparse = sparse.copy()
    sparse["date"] = pd.to_datetime(sparse["date"])
    dense_parts: list[pd.DataFrame] = []
    quarter_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []
    block_records: dict[int, list[np.ndarray]] = {1: [], 2: []}
    block_pairs: dict[int, list[str]] = {1: [], 2: []}
    block_rows: list[dict[str, Any]] = []
    block_labels = anniversary_block_labels(dates, start, block_count=10, end_exclusive=end)
    numeric = [
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "ensemble_active_member_count",
        "ensemble_held_member_count",
        "net_return_1bp",
        "net_return_2bp",
    ]
    execution_audit = [
        "zscore",
        "decision_event",
        "fill_event",
        "fill_decision_date",
        "fill_decision_zscore",
        "execution_audit_source",
    ]
    for master_row in master.sort_values("pair").to_dict("records"):
        pair = str(master_row["pair"])
        pair_sparse = sparse.loc[sparse["pair"].astype(str).eq(pair)].copy()
        dense = pd.DataFrame({"pair": pair, "date": dates, "quarter_number": quarter_by_date.to_numpy()})
        if len(pair_sparse):
            dense = dense.merge(
                pair_sparse[
                    [
                        "date",
                        "strategy_id",
                        "vintage_rebalance_date",
                        *execution_audit,
                        *numeric,
                    ]
                ],
                on="date",
                how="left",
                validate="one_to_one",
            )
        else:
            for column in [
                "strategy_id",
                "vintage_rebalance_date",
                *execution_audit,
                *numeric,
            ]:
                dense[column] = np.nan
        dense["strategy_id"] = dense["strategy_id"].astype("string")
        selected_by_quarter = selection_key.loc[
            selection_key["pair"].astype(str).eq(pair)
        ].set_index("rebalance_date")["selected_strategy_id"]
        quarter_start_by_number = schedule.set_index("quarter_number")["rebalance_date"]
        fallback_strategy = dense["quarter_number"].map(quarter_start_by_number).map(selected_by_quarter)
        dense["strategy_id"] = dense["strategy_id"].fillna(fallback_strategy).fillna("S0_v6")
        for column in numeric:
            dense[column] = pd.to_numeric(dense[column], errors="coerce").fillna(0.0)
        dense["zscore"] = pd.to_numeric(dense["zscore"], errors="coerce")
        dense["fill_decision_zscore"] = pd.to_numeric(
            dense["fill_decision_zscore"], errors="coerce"
        )
        dense["fill_decision_date"] = pd.to_datetime(
            dense["fill_decision_date"], errors="coerce"
        )
        for column in ["decision_event", "fill_event", "execution_audit_source"]:
            dense[column] = dense[column].astype("string").fillna("unavailable_cash")
        dense["ticker_a"] = master_row["ticker_a"]
        dense["ticker_b"] = master_row["ticker_b"]
        dense["price_a"] = pd.to_numeric(prices.loc[dates, master_row["ticker_a"]], errors="coerce").to_numpy()
        dense["price_b"] = pd.to_numeric(prices.loc[dates, master_row["ticker_b"]], errors="coerce").to_numpy()
        dense["held_short_notional"] = (
            dense["weight_a"].clip(upper=0.0).abs()
            + dense["weight_b"].clip(upper=0.0).abs()
        )
        dense["equity_1bp"] = (1.0 + dense["net_return_1bp"]).cumprod()
        dense["equity_2bp"] = (1.0 + dense["net_return_2bp"]).cumprod()
        dense["drawdown_2bp"] = dense["equity_2bp"] / dense["equity_2bp"].cummax().clip(lower=1.0) - 1.0
        dense["trade_entry"] = _directional_episode_entries(dense)
        dense_parts.append(dense)
        trades = _trade_episodes(dense)
        if len(trades):
            trade_parts.append(trades)
        qbase = quarterly.loc[quarterly["pair"].astype(str).eq(pair)].sort_values("quarter_number")
        for qrow in qbase.to_dict("records"):
            qdaily = dense.loc[dense["quarter_number"].eq(int(qrow["quarter_number"]))]
            active = bool(
                (
                    qdaily["weight_a"].abs()
                    + qdaily["weight_b"].abs()
                    + qdaily["target_weight_a"].abs()
                    + qdaily["target_weight_b"].abs()
                ).gt(1e-12).any()
            )
            entries = int(qdaily["trade_entry"].sum())
            quarter_rows.append(
                {
                    **qrow,
                    "quarter_return_1bp": float((1.0 + qdaily["net_return_1bp"]).prod() - 1.0),
                    "quarter_return_2bp": float((1.0 + qdaily["net_return_2bp"]).prod() - 1.0),
                    "v8_actively_traded": active,
                    "v8_trade_entries": entries,
                    "v8_trade_entries_definition": (
                        "aggregate_target_direction_episode_starts"
                    ),
                }
            )
        pair_quarters = pd.DataFrame(quarter_rows[-len(qbase):])
        available = int(pair_quarters["available"].astype(str).str.lower().isin({"true", "1"}).sum())
        evaluated = int(pair_quarters["evaluation_success"].astype(str).str.lower().isin({"true", "1"}).sum())
        active_quarters = int(pair_quarters["v8_actively_traded"].sum())
        eligibility = int(pair_quarters["original_v6_eligible"].astype(str).str.lower().isin({"true", "1"}).sum())
        strategy_history = selection_key.loc[selection_key["pair"].astype(str).eq(pair)].sort_values("rebalance_date")
        strategy_values = strategy_history["selected_strategy_id"].astype(str)
        strategy_switch_rate = float((strategy_values != strategy_values.shift()).iloc[1:].mean()) if len(strategy_values) > 1 else math.nan
        common = {
            "pair": pair,
            "sector": master_row["sector"],
            "ticker_a": master_row["ticker_a"],
            "ticker_b": master_row["ticker_b"],
            "original_v6_selected_identity": _bool(master_row["ever_v6_selected"]),
            "scheduled_quarters": len(pair_quarters),
            "available_quarters": available,
            "evaluation_success_quarters": evaluated,
            "unavailable_quarters": len(pair_quarters) - available,
            "active_quarters": active_quarters,
            "zero_trade_quarters": len(pair_quarters) - active_quarters,
            "trade_count": int(len(trades)),
            "trade_count_definition": (
                "aggregate_target_direction_episodes_including_active_reversals"
            ),
            "availability_fraction": available / len(pair_quarters),
            "active_quarter_fraction": active_quarters / len(pair_quarters),
            "eligibility_pass_fraction": eligibility / len(pair_quarters),
            "unique_selected_strategies": int(strategy_values.nunique()),
            "modal_strategy_id": strategy_values.value_counts().index[0],
            "modal_strategy_fraction": float(strategy_values.value_counts(normalize=True).iloc[0]),
            "strategy_switch_rate": strategy_switch_rate,
            "insufficient_history_default_quarters": int(strategy_history["selection_reason"].eq("insufficient_causal_strategy_history").sum()),
            "hedge_beta_std": float(pd.to_numeric(pair_quarters["evaluation_beta"], errors="coerce").std()),
        }
        for cost in (1, 2):
            returns = dense[f"net_return_{cost}bp"].to_numpy(dtype=float)
            stats = path_statistics(returns, calendar_years=years)
            qreturns = pair_quarters[f"quarter_return_{cost}bp"].to_numpy(dtype=float)
            best_q = int(np.argmax(qreturns))
            worst_q = int(np.argmin(qreturns))
            best_q_number = int(pair_quarters.iloc[best_q]["quarter_number"])
            without_best_q = remove_block_as_cash(
                returns,
                dense["quarter_number"].eq(best_q_number).to_numpy(),
                calendar_years=years,
            )
            pair_annual: list[dict[str, Any]] = []
            for year, group in dense.groupby(dense["date"].dt.year, sort=True):
                annual_return = float((1.0 + group[f"net_return_{cost}bp"]).prod() - 1.0)
                annual = {"pair": pair, "transaction_cost_bps": float(cost), "calendar_year": int(year), "net_return": annual_return}
                pair_annual.append(annual)
                annual_rows.append(annual)
            annual_frame = pd.DataFrame(pair_annual)
            complete = annual_frame.loc[annual_frame["calendar_year"].between(start.year + 1, end.year - 1)]
            best_year_row = complete.sort_values(["net_return", "calendar_year"], ascending=[False, True]).iloc[0]
            best_year = int(best_year_row["calendar_year"])
            without_best_year = remove_block_as_cash(
                returns,
                dense["date"].dt.year.eq(best_year).to_numpy(),
                calendar_years=years,
            )
            rates: list[float] = []
            for block in range(10):
                mask = block_labels == block
                block_start = start + pd.DateOffset(years=block)
                block_end = end if block == 9 else start + pd.DateOffset(years=block + 1)
                block_years = exact_calendar_years(block_start, block_end)
                log_growth = float(np.log1p(returns[mask]).sum())
                rate = log_growth / block_years
                rates.append(rate)
                block_rows.append(
                    {"pair": pair, "transaction_cost_bps": float(cost), "anniversary_block": block + 1, "annual_log_growth_rate": rate, "total_return": float(np.expm1(log_growth))}
                )
            block_records[cost].append(np.asarray(rates))
            block_pairs[cost].append(pair)
            metric_rows.append(
                {
                    **common,
                    "transaction_cost_bps": float(cost),
                    **stats,
                    "positive_quarter_fraction": float(np.mean(qreturns > 0)),
                    "positive_calendar_year_fraction": float((annual_frame["net_return"] > 0).mean()),
                    "median_quarter_return": float(np.median(qreturns)),
                    "worst_quarter_return": float(qreturns[worst_q]),
                    "best_quarter_return": float(qreturns[best_q]),
                    "cagr_without_best_quarter": without_best_q["calendar_cagr"],
                    "cagr_without_best_complete_year": without_best_year["calendar_cagr"],
                    "best_complete_calendar_year": best_year,
                    "best_complete_calendar_year_return": float(best_year_row["net_return"]),
                    "full_calendar_metric": True,
                    "active_period_annualization_used": False,
                }
            )
    dense_output = pd.concat(dense_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    shared_counts: np.ndarray | None = None
    bootstrap_rows: list[dict[str, Any]] = []
    for cost in (1, 2):
        rates = np.vstack(block_records[cost])
        quantiles, shared_counts = bootstrap_cagr_from_log_blocks(
            rates,
            draws=int(config["ranking"]["bootstrap_draws"]),
            seed=int(config["ranking"]["bootstrap_seed"]),
            shared_counts=shared_counts,
        )
        for pair, values in zip(block_pairs[cost], quantiles, strict=True):
            mask = metrics["pair"].eq(pair) & metrics["transaction_cost_bps"].eq(float(cost))
            metrics.loc[mask, "bootstrap_cagr_ci_low"] = values[0]
            metrics.loc[mask, "bootstrap_cagr_median"] = values[1]
            metrics.loc[mask, "bootstrap_cagr_ci_high"] = values[2]
            bootstrap_rows.append(
                {"pair": pair, "transaction_cost_bps": float(cost), "cagr_ci_low": values[0], "cagr_median": values[1], "cagr_ci_high": values[2]}
            )
    return {
        "daily": dense_output,
        "quarterly": pd.DataFrame(quarter_rows),
        "annual": pd.DataFrame(annual_rows),
        "metrics": metrics,
        "trades": pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(),
        "blocks": pd.DataFrame(block_rows),
        "confidence": pd.DataFrame(bootstrap_rows),
    }


def _configured_stress_values(
    evaluation: Mapping[str, Any],
    primary_key: str,
    stress_key: str,
) -> list[float]:
    raw = [evaluation[primary_key], *evaluation.get(stress_key, [])]
    values: list[float] = []
    for item in raw:
        value = float(item)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{primary_key}/{stress_key} must contain finite nonnegative values")
        if value not in values:
            values.append(value)
    return values


def _build_cost_borrow_stress_metrics(
    daily: pd.DataFrame,
    master: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Reprice frozen full-panel paths over an independent cost/borrow grid."""

    required = {
        "pair",
        "date",
        "gross_return",
        "turnover",
        "weight_a",
        "weight_b",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily panel missing stress columns: {missing}")
    if daily.duplicated(["pair", "date"]).any():
        raise ValueError("stress daily panel contains duplicate pair/date rows")
    evaluation = config["evaluation"]
    transaction_costs = _configured_stress_values(
        evaluation,
        "primary_transaction_cost_bps",
        "stress_transaction_cost_bps",
    )
    borrow_costs = _configured_stress_values(
        evaluation,
        "annual_short_borrow_bps",
        "stress_short_borrow_bps",
    )
    primary_transaction_cost = float(evaluation["primary_transaction_cost_bps"])
    primary_borrow_cost = float(evaluation["annual_short_borrow_bps"])
    start = pd.Timestamp(evaluation["study_start_inclusive"])
    end = pd.Timestamp(evaluation["study_end_exclusive"])
    calendar_years = exact_calendar_years(start, end)

    metadata = master.copy()
    metadata["pair"] = metadata["pair"].astype(str)
    metadata = metadata.set_index("pair")
    if not metadata.index.is_unique:
        raise ValueError("stress master contains duplicate pair identities")
    daily_pairs = set(daily["pair"].astype(str))
    if daily_pairs != set(metadata.index):
        raise ValueError("stress daily/master pair identities do not match")

    rows: list[dict[str, Any]] = []
    for pair, group in daily.groupby("pair", sort=True):
        work = group.sort_values("date").copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        if work["date"].isna().any():
            raise ValueError(f"{pair} contains invalid stress dates")
        gross = pd.to_numeric(work["gross_return"], errors="coerce").to_numpy(
            dtype=float
        )
        turnover = pd.to_numeric(work["turnover"], errors="coerce").to_numpy(
            dtype=float
        )
        weight_a = pd.to_numeric(work["weight_a"], errors="coerce").to_numpy(
            dtype=float
        )
        weight_b = pd.to_numeric(work["weight_b"], errors="coerce").to_numpy(
            dtype=float
        )
        if not all(
            np.isfinite(values).all()
            for values in (gross, turnover, weight_a, weight_b)
        ):
            raise ValueError(f"{pair} stress inputs must be finite")
        if np.any(turnover < -1e-12):
            raise ValueError(f"{pair} stress turnover must be nonnegative")
        derived_short = np.maximum(-weight_a, 0.0) + np.maximum(-weight_b, 0.0)
        if "held_short_notional" in work:
            frozen_short = pd.to_numeric(
                work["held_short_notional"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(frozen_short).all() or not np.allclose(
                frozen_short, derived_short, atol=1e-12, rtol=0.0
            ):
                raise ValueError(f"{pair} held short notional does not match held weights")
            short_notional = frozen_short
        else:
            short_notional = derived_short
        meta = metadata.loc[str(pair)]
        for transaction_cost_bps in transaction_costs:
            transaction_drag = (
                turnover
                * transaction_cost_bps
                / 10_000.0
                * (1.0 + gross)
            )
            for annual_short_borrow_bps in borrow_costs:
                borrow_drag = (
                    short_notional
                    * annual_short_borrow_bps
                    / 10_000.0
                    / 252.0
                )
                net = gross - transaction_drag - borrow_drag
                if not np.isfinite(net).all() or np.any(net <= -1.0):
                    raise ValueError(f"{pair} stress scenario produced invalid returns")
                statistics = path_statistics(net, calendar_years=calendar_years)
                primary = bool(
                    transaction_cost_bps == primary_transaction_cost
                    and annual_short_borrow_bps == primary_borrow_cost
                )
                primary_error = math.nan
                if primary and "net_return_2bp" in work:
                    recorded = pd.to_numeric(
                        work["net_return_2bp"], errors="coerce"
                    ).to_numpy(dtype=float)
                    primary_error = float(np.max(np.abs(recorded - net)))
                rows.append(
                    {
                        "evaluation_view": "cost_borrow_stress_frozen_full_panel",
                        "pair": str(pair),
                        "sector": meta.get("sector", pd.NA),
                        "ticker_a": meta.get("ticker_a", pd.NA),
                        "ticker_b": meta.get("ticker_b", pd.NA),
                        "scenario_id": (
                            f"transaction_{transaction_cost_bps:g}bp__borrow_"
                            f"{annual_short_borrow_bps:g}bp"
                        ),
                        "transaction_cost_bps": transaction_cost_bps,
                        "annual_short_borrow_bps": annual_short_borrow_bps,
                        "matches_primary_ranking_assumptions": primary,
                        "used_to_construct_primary_ranking": False,
                        "frozen_gross_return_and_positions": True,
                        "full_calendar_metric": True,
                        "full_panel_is_ex_post_descriptive": True,
                        "scheduled_sessions": int(len(work)),
                        "turnover_sum": float(turnover.sum()),
                        "held_short_notional_session_sum": float(
                            short_notional.sum()
                        ),
                        "transaction_cost_sum": float(transaction_drag.sum()),
                        "borrow_cost_sum": float(borrow_drag.sum()),
                        "maximum_primary_net_return_identity_error": primary_error,
                        **statistics,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["transaction_cost_bps", "annual_short_borrow_bps", "pair"],
        kind="mergesort",
    ).reset_index(drop=True)


def _apply_post_discovery_safe_reset_gate(
    daily: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Start each frozen path at its first clean flat reset after discovery.

    This is an evaluation-only diagnostic.  It deliberately retains the
    already-selected frozen V8 path after a clean state boundary; it does not
    reconstruct a point-in-time identity universe or remove pre-discovery
    candidate outcomes from the strategy-selection history.
    """

    tolerance = 1e-12
    economic_columns = (
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "turnover",
        "gross_return",
        "transaction_cost",
        "borrow_cost",
        "net_return_1bp",
        "net_return_2bp",
    )
    required_daily = {
        "pair",
        "date",
        "decision_event",
        "fill_event",
        "trade_entry",
        "ensemble_active_member_count",
        "ensemble_held_member_count",
        *economic_columns,
    }
    required_master = {"pair", "first_successful_v6_tuned_date"}
    missing_daily = sorted(required_daily - set(daily.columns))
    missing_master = sorted(required_master - set(master.columns))
    if missing_daily:
        raise ValueError(
            f"daily panel missing safe-reset diagnostic columns: {missing_daily}"
        )
    if missing_master:
        raise ValueError(
            f"identity master missing safe-reset diagnostic columns: {missing_master}"
        )
    if daily.duplicated(["pair", "date"]).any():
        raise ValueError("daily panel contains duplicate pair/date rows")
    known = master[["pair", "first_successful_v6_tuned_date"]].copy()
    known["pair"] = known["pair"].astype(str)
    if known["pair"].duplicated().any():
        raise ValueError("identity master contains duplicate pairs")
    known["first_successful_v6_tuned_date"] = pd.to_datetime(
        known["first_successful_v6_tuned_date"], errors="coerce"
    )
    if known["first_successful_v6_tuned_date"].isna().any():
        raise ValueError("every pair requires a valid first-successful V6 tuned date")

    result = daily.copy(deep=True)
    result["pair"] = result["pair"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily panel contains invalid dates")
    daily_pairs = set(result["pair"])
    master_pairs = set(known["pair"])
    if daily_pairs != master_pairs:
        missing = sorted(daily_pairs - master_pairs)
        unused = sorted(master_pairs - daily_pairs)
        raise ValueError(
            "daily/master pair mismatch; missing knowable dates="
            f"{missing[:5]}, master-only pairs={unused[:5]}"
        )
    result = result.merge(known, on="pair", how="left", validate="many_to_one")
    result = result.sort_values(["pair", "date"], kind="mergesort").reset_index(
        drop=True
    )
    for column in economic_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite daily values")
        result[column] = values.astype(float)
        result[f"full_panel_{column}"] = values.astype(float)
    member_state_columns = (
        "ensemble_active_member_count",
        "ensemble_held_member_count",
    )
    for member_state_column in member_state_columns:
        member_counts = pd.to_numeric(
            result[member_state_column], errors="coerce"
        )
        member_values = member_counts.to_numpy(dtype=float)
        if not bool(
            np.isfinite(member_values).all()
            and (member_values >= 0.0).all()
            and np.allclose(
                member_values,
                np.rint(member_values),
                rtol=0.0,
                atol=tolerance,
            )
        ):
            raise ValueError(
                f"{member_state_column} must be finite nonnegative integers"
            )
        result[member_state_column] = member_counts.astype(float)
        result[f"full_panel_{member_state_column}"] = member_counts.astype(float)
    for column in ("net_return_1bp", "net_return_2bp"):
        values = result[f"full_panel_{column}"]
        if values.le(-1.0).any():
            raise ValueError(f"{column} must be greater than -100%")
    for column in ("equity_1bp", "equity_2bp", "drawdown_2bp"):
        if column in result:
            values = pd.to_numeric(result[column], errors="coerce")
            if values.isna().any() or not np.isfinite(
                values.to_numpy(dtype=float)
            ).all():
                raise ValueError(f"{column} must contain finite daily values")
            result[f"full_panel_{column}"] = values.astype(float)
    result["full_panel_trade_entry"] = result["trade_entry"].map(_bool)
    result["first_known_date"] = result["first_successful_v6_tuned_date"]
    result["identity_known_on_date"] = result["date"].ge(
        result["first_known_date"]
    )

    parts: list[pd.DataFrame] = []
    for pair, group in result.groupby("pair", sort=True):
        work = group.sort_values("date", kind="mergesort").copy()
        known_date = pd.Timestamp(work["first_known_date"].iloc[0])
        has_next_session = pd.Series(
            np.arange(len(work)) < len(work) - 1,
            index=work.index,
            dtype=bool,
        )
        zero_economics = pd.Series(True, index=work.index, dtype=bool)
        for column in economic_columns:
            zero_economics &= work[f"full_panel_{column}"].abs().le(tolerance)
        for member_state_column in member_state_columns:
            zero_economics &= work[
                f"full_panel_{member_state_column}"
            ].eq(0.0)
        safe = (
            work["date"].ge(known_date)
            & has_next_session
            & zero_economics
            & work["decision_event"].astype(str).eq("flat")
            & work["fill_event"].astype(str).eq("flat")
        )
        safe_rows = work.index[safe]
        if len(safe_rows):
            activation_index = safe_rows[0]
            activation_date = pd.Timestamp(work.at[activation_index, "date"])
            active_path = work["date"].ge(activation_date)
            status = "activated"
            reset_wait_sessions = int(
                (work["date"].ge(known_date) & work["date"].lt(activation_date)).sum()
            )
        else:
            activation_date = pd.NaT
            active_path = pd.Series(False, index=work.index, dtype=bool)
            status = "no_safe_activation_all_cash"
            reset_wait_sessions = int(work["date"].ge(known_date).sum())
        for column in economic_columns:
            work[column] = np.where(
                active_path, work[f"full_panel_{column}"], 0.0
            )
        for member_state_column in member_state_columns:
            work[member_state_column] = np.where(
                active_path,
                work[f"full_panel_{member_state_column}"],
                0.0,
            )
        work["trade_entry"] = _directional_episode_entries(
            work, tolerance=tolerance
        )
        work["safe_activation_date"] = activation_date
        work["safe_activation_status"] = status
        work["reset_wait_sessions"] = reset_wait_sessions
        work["safe_reset_path_active"] = active_path
        work["post_discovery_safe_reset_cash"] = ~active_path
        work["selection_history_includes_pre_discovery_candidates"] = True
        parts.append(work)

    output = pd.concat(parts, ignore_index=True, sort=False)
    output["equity_1bp"] = output.groupby("pair", sort=False)[
        "net_return_1bp"
    ].transform(lambda values: (1.0 + values).cumprod())
    output["equity_2bp"] = output.groupby("pair", sort=False)[
        "net_return_2bp"
    ].transform(lambda values: (1.0 + values).cumprod())
    output["drawdown_2bp"] = output.groupby("pair", sort=False)[
        "equity_2bp"
    ].transform(lambda values: values / values.cummax().clip(lower=1.0) - 1.0)
    output["evaluation_view"] = (
        "post_discovery_safe_reset_frozen_path_diagnostic"
    )
    return output.sort_values(["pair", "date"], kind="mergesort").reset_index(
        drop=True
    )


def _romano_wolf_from_year_blocks(
    blocks: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen V8 synchronized anniversary-block inference."""

    required = {
        "pair",
        "transaction_cost_bps",
        "anniversary_block",
        "annual_log_growth_rate",
    }
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(f"year-block panel missing columns: {missing}")
    primary = blocks.loc[
        pd.to_numeric(blocks["transaction_cost_bps"], errors="coerce").eq(2.0)
    ].copy()
    matrix = primary.pivot(
        index="anniversary_block",
        columns="pair",
        values="annual_log_growth_rate",
    ).sort_index().sort_index(axis=1)
    if matrix.empty or matrix.isna().any().any():
        raise ValueError("2 bp anniversary blocks must form a complete pair matrix")
    draws = int(config["ranking"]["bootstrap_draws"])
    seed = int(config["ranking"]["bootstrap_seed"])
    indices = circular_block_indices(
        len(matrix),
        draws=draws,
        block_length=2,
        seed=seed,
    )
    inference, summary = romano_wolf_max_t(
        matrix.to_numpy(dtype=float),
        list(matrix.columns.astype(str)),
        indices,
        alpha=0.05,
    )
    summary = {
        **summary,
        "bootstrap_seed": seed,
        "bootstrap_block_length": 2,
    }
    return inference, summary


def _build_post_discovery_safe_reset_artifacts(
    daily: pd.DataFrame,
    master: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Evaluate frozen V8 paths after discovery and a clean flat reset."""

    required = {
        "pair",
        "date",
        "quarter_number",
        "strategy_id",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
        "trade_entry",
        "net_return_1bp",
        "net_return_2bp",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(
            f"daily panel missing post-discovery diagnostic columns: {missing}"
        )
    gated = _apply_post_discovery_safe_reset_gate(daily, master)
    start = pd.Timestamp(config["evaluation"]["study_start_inclusive"])
    end = pd.Timestamp(config["evaluation"]["study_end_exclusive"])
    years = exact_calendar_years(start, end)
    dates = pd.DatetimeIndex(sorted(gated["date"].unique()))
    if len(gated) != len(dates) * gated["pair"].nunique():
        raise ValueError(
            "post-discovery diagnostic daily panel must be balanced by pair and date"
        )
    if dates[0] < start or dates[-1] >= end:
        raise ValueError(
            "post-discovery diagnostic dates fall outside the configured study"
        )
    block_labels = anniversary_block_labels(
        dates,
        start,
        block_count=10,
        end_exclusive=end,
    )
    master_rows = master.copy()
    master_rows["pair"] = master_rows["pair"].astype(str)
    master_rows = master_rows.set_index("pair")
    if not master_rows.index.is_unique:
        raise ValueError(
            "post-discovery diagnostic master contains duplicate pair identities"
        )

    metric_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for pair, group in gated.groupby("pair", sort=True):
        work = group.sort_values("date").reset_index(drop=True)
        if not pd.DatetimeIndex(work["date"]).equals(dates):
            raise ValueError(f"{pair} does not share the canonical V8 date index")
        known = work["identity_known_on_date"].astype(bool)
        active_path = work["safe_reset_path_active"].astype(bool)
        active = (
            work["weight_a"].abs()
            + work["weight_b"].abs()
            + work["target_weight_a"].abs()
            + work["target_weight_b"].abs()
        ).gt(1e-12) & active_path
        entries = work["trade_entry"].astype(bool) & active_path
        strategy = work.loc[active_path, "strategy_id"].astype(str)
        modal_strategy = (
            strategy.value_counts().index[0]
            if len(strategy)
            else "no_safe_activation_all_cash"
        )
        modal_fraction = (
            float(strategy.value_counts(normalize=True).iloc[0])
            if len(strategy)
            else math.nan
        )
        metadata = master_rows.loc[pair]
        activation_values = pd.to_datetime(
            work["safe_activation_date"], errors="coerce"
        ).dropna()
        activation_date = (
            pd.Timestamp(activation_values.iloc[0]).date().isoformat()
            if len(activation_values)
            else None
        )
        activation_status = str(work["safe_activation_status"].iloc[0])
        reset_wait_sessions = int(work["reset_wait_sessions"].iloc[0])
        common = {
            "evaluation_view": (
                "post_discovery_safe_reset_frozen_path_diagnostic"
            ),
            "pair": pair,
            "sector": metadata.get("sector", pd.NA),
            "ticker_a": metadata.get("ticker_a", pd.NA),
            "ticker_b": metadata.get("ticker_b", pd.NA),
            "first_known_date": pd.Timestamp(
                metadata["first_successful_v6_tuned_date"]
            ).date().isoformat(),
            "safe_activation_date": activation_date,
            "safe_activation_status": activation_status,
            "reset_wait_sessions": reset_wait_sessions,
            "scheduled_sessions": int(len(work)),
            "known_sessions": int(known.sum()),
            "diagnostic_cash_sessions": int((~active_path).sum()),
            "known_session_fraction": float(known.mean()),
            "safe_reset_active_session_fraction": float(active_path.mean()),
            "scheduled_quarters": int(work["quarter_number"].nunique()),
            "known_quarters": int(
                work.loc[known, "quarter_number"].nunique()
            ),
            "active_quarters": int(work.loc[active, "quarter_number"].nunique()),
            "trade_count": int(entries.sum()),
            "trade_count_definition": (
                "aggregate_target_direction_episodes_including_active_reversals"
            ),
            "modal_strategy_id": modal_strategy,
            "modal_strategy_fraction": modal_fraction,
            "full_panel_is_ex_post_descriptive": True,
            "conditional_on_fixed_legacy_114": True,
            "complete_point_in_time_universe": False,
            "deployable_selection": False,
            "selection_history_includes_pre_discovery_candidates": True,
            "leakage_free_strategy_claim": False,
        }
        for cost in (1, 2):
            returns = work[f"net_return_{cost}bp"].to_numpy(dtype=float)
            statistics = path_statistics(returns, calendar_years=years)
            quarter_frame = (
                pd.DataFrame(
                    {
                        "quarter_number": work["quarter_number"].to_numpy(),
                        "return": returns,
                    }
                )
                .groupby("quarter_number", sort=True)["return"]
                .apply(lambda values: float((1.0 + values).prod() - 1.0))
            )
            quarter_returns = quarter_frame.to_numpy(dtype=float)
            best_quarter = int(quarter_frame.index[np.argmax(quarter_returns)])
            worst_quarter = int(np.argmin(quarter_returns))
            without_best_quarter = remove_block_as_cash(
                returns,
                work["quarter_number"].eq(best_quarter).to_numpy(),
                calendar_years=years,
            )
            annual = (
                pd.DataFrame(
                    {"year": work["date"].dt.year.to_numpy(), "return": returns}
                )
                .groupby("year", sort=True)["return"]
                .apply(lambda values: float((1.0 + values).prod() - 1.0))
            )
            complete_years = annual.loc[annual.index.to_series().between(start.year + 1, end.year - 1)]
            best_year = int(
                (complete_years if len(complete_years) else annual).idxmax()
            )
            without_best_year = remove_block_as_cash(
                returns,
                work["date"].dt.year.eq(best_year).to_numpy(),
                calendar_years=years,
            )
            metric_rows.append(
                {
                    **common,
                    "transaction_cost_bps": float(cost),
                    **statistics,
                    "positive_quarter_fraction": float(
                        np.mean(quarter_returns > 0)
                    ),
                    "positive_calendar_year_fraction": float((annual > 0).mean()),
                    "median_quarter_return": float(np.median(quarter_returns)),
                    "worst_quarter_return": float(quarter_returns[worst_quarter]),
                    "best_quarter_return": float(np.max(quarter_returns)),
                    "cagr_without_best_quarter": without_best_quarter[
                        "calendar_cagr"
                    ],
                    "cagr_without_best_complete_year": without_best_year[
                        "calendar_cagr"
                    ],
                    "best_complete_calendar_year": best_year,
                    "best_complete_calendar_year_return": float(annual.loc[best_year]),
                    "active_period_annualization_used": False,
                }
            )
            for block in range(10):
                mask = block_labels == block
                block_start = start + pd.DateOffset(years=block)
                block_end = end if block == 9 else start + pd.DateOffset(years=block + 1)
                block_years = exact_calendar_years(block_start, block_end)
                log_growth = float(np.log1p(returns[mask]).sum())
                block_rows.append(
                    {
                        "evaluation_view": (
                            "post_discovery_safe_reset_frozen_path_diagnostic"
                        ),
                        "pair": pair,
                        "transaction_cost_bps": float(cost),
                        "anniversary_block": block + 1,
                        "annual_log_growth_rate": log_growth / block_years,
                        "total_return": float(np.expm1(log_growth)),
                        "conditional_on_fixed_legacy_114": True,
                        "complete_point_in_time_universe": False,
                        "selection_history_includes_pre_discovery_candidates": True,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    blocks = pd.DataFrame(block_rows)
    confidence_rows: list[dict[str, Any]] = []
    shared_counts: np.ndarray | None = None
    for cost in (1, 2):
        rates = (
            blocks.loc[blocks["transaction_cost_bps"].eq(float(cost))]
            .pivot(
                index="pair",
                columns="anniversary_block",
                values="annual_log_growth_rate",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        quantiles, shared_counts = bootstrap_cagr_from_log_blocks(
            rates.to_numpy(dtype=float),
            draws=int(config["ranking"]["bootstrap_draws"]),
            seed=int(config["ranking"]["bootstrap_seed"]),
            shared_counts=shared_counts,
        )
        for pair, values in zip(rates.index.astype(str), quantiles, strict=True):
            mask = metrics["pair"].eq(pair) & metrics[
                "transaction_cost_bps"
            ].eq(float(cost))
            metrics.loc[mask, "bootstrap_cagr_ci_low"] = values[0]
            metrics.loc[mask, "bootstrap_cagr_median"] = values[1]
            metrics.loc[mask, "bootstrap_cagr_ci_high"] = values[2]
            confidence_rows.append(
                {
                    "evaluation_view": (
                        "post_discovery_safe_reset_frozen_path_diagnostic"
                    ),
                    "pair": pair,
                    "transaction_cost_bps": float(cost),
                    "cagr_ci_low": values[0],
                    "cagr_median": values[1],
                    "cagr_ci_high": values[2],
                    "conditional_on_fixed_legacy_114": True,
                    "complete_point_in_time_universe": False,
                    "selection_history_includes_pre_discovery_candidates": True,
                }
            )

    rankings = metrics.loc[metrics["transaction_cost_bps"].eq(2.0)].sort_values(
        ["calendar_cagr", "bootstrap_cagr_ci_low", "sharpe", "pair"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rankings.insert(0, "rank", np.arange(1, len(rankings) + 1))
    rankings.insert(
        1, "ranking_name", "post_discovery_safe_reset_full_calendar_cagr"
    )
    rankings.insert(2, "ranking_scope", "conditional_fixed_legacy_114")
    inference, inference_summary = _romano_wolf_from_year_blocks(blocks, config)
    inference = inference.rename(columns={"pair": "pair_identity"})
    inference.insert(
        0,
        "evaluation_view",
        "post_discovery_safe_reset_frozen_path_diagnostic",
    )
    inference["conditional_on_fixed_legacy_114"] = True
    inference["complete_point_in_time_universe"] = False
    inference["deployable_selection"] = False
    inference["selection_history_includes_pre_discovery_candidates"] = True

    gated["safe_reset_equity_1bp"] = gated["equity_1bp"]
    gated["safe_reset_equity_2bp"] = gated["equity_2bp"]
    gated["safe_reset_drawdown_2bp"] = gated["drawdown_2bp"]
    status_counts = {
        str(key): int(value)
        for key, value in gated.drop_duplicates("pair")[
            "safe_activation_status"
        ].value_counts().items()
    }
    summary = {
        "evaluation_view": "post_discovery_safe_reset_frozen_path_diagnostic",
        "first_known_date_source": "master.first_successful_v6_tuned_date",
        "safe_activation_rule": (
            "first row on/after first known date with exact flat decision/fill, zero "
            "held/target/economic state, and at least one following session"
        ),
        "pair_count": int(gated["pair"].nunique()),
        "daily_rows": int(len(gated)),
        "diagnostic_pair_sessions_set_to_cash": int(
            gated["post_discovery_safe_reset_cash"].sum()
        ),
        "total_reset_wait_sessions": int(
            gated.drop_duplicates("pair")["reset_wait_sessions"].sum()
        ),
        "safe_activation_status_counts": status_counts,
        "earliest_first_known_date": str(
            gated["first_known_date"].min().date()
        ),
        "latest_first_known_date": str(
            gated["first_known_date"].max().date()
        ),
        "bootstrap_draws": int(config["ranking"]["bootstrap_draws"]),
        "bootstrap_seed": int(config["ranking"]["bootstrap_seed"]),
        "bootstrap_block_length": 2,
        "inference": inference_summary,
        "full_panel_is_ex_post_descriptive": True,
        "conditional_on_fixed_legacy_114": True,
        "complete_point_in_time_universe": False,
        "deployable_selection": False,
        "selection_history_includes_pre_discovery_candidates": True,
        "leakage_free_strategy_claim": False,
    }
    return (
        {
            "daily": gated,
            "metrics": metrics.sort_values(
                ["transaction_cost_bps", "calendar_cagr", "pair"],
                ascending=[True, False, True],
                kind="mergesort",
            ).reset_index(drop=True),
            "rankings": rankings,
            "blocks": blocks.sort_values(
                ["transaction_cost_bps", "pair", "anniversary_block"],
                kind="mergesort",
            ).reset_index(drop=True),
            "confidence": pd.DataFrame(confidence_rows),
            "inference": inference,
        },
        summary,
    )


def _rank_pairs(metrics: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = metrics.loc[metrics["transaction_cost_bps"].eq(2.0)].copy()
    reference = primary.loc[
        primary["availability_fraction"].ge(0.8)
        & primary["active_quarters"].ge(20)
    ].copy()
    primary["consistency_score"] = 0.0
    if len(reference):
        components = pd.DataFrame(
            {
                "sharpe": percentile_score(reference["sharpe"]),
                "drawdown": percentile_score(reference["max_drawdown"]),
                "positive_quarters": percentile_score(reference["positive_quarter_fraction"]),
                "positive_years": percentile_score(reference["positive_calendar_year_fraction"]),
                "strategy_stability": percentile_score(reference["modal_strategy_fraction"]),
                "coverage": percentile_score(reference["active_quarter_fraction"]),
            },
            index=reference.index,
        )
        primary.loc[reference.index, "consistency_score"] = 100.0 * components.mean(axis=1)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    selected_pairs = set(master.loc[master["ever_v6_selected"].astype(str).str.lower().isin({"true", "1"}), "pair"].astype(str))
    scopes = {
        "fixed_114_primary": pd.Series(True, index=primary.index),
        "original_v6_selected_82_subset": primary["pair"].astype(str).isin(selected_pairs),
    }
    definitions = {
        "full_calendar_cagr": (["calendar_cagr", "bootstrap_cagr_ci_low", "sharpe", "pair"], [False, False, False, True]),
        "robust_lower_bound": (["bootstrap_cagr_ci_low", "cagr_without_best_quarter", "calendar_cagr", "pair"], [False, False, False, True]),
        "risk_adjusted_consistency": (["consistency_score", "bootstrap_cagr_ci_low", "calendar_cagr", "pair"], [False, False, False, True]),
    }
    for scope, scope_mask in scopes.items():
        for threshold in (8, 12, 20, 30):
            coverage = (
                scope_mask
                & primary["availability_fraction"].ge(0.8)
                & primary["active_quarters"].ge(threshold)
            )
            for rank_type, (columns, ascending) in definitions.items():
                ranked = primary.loc[coverage].sort_values(columns, ascending=ascending, kind="mergesort").copy()
                ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
                ranked.insert(0, "rank_type", rank_type)
                ranked.insert(0, "minimum_active_quarters", threshold)
                ranked.insert(0, "ranking_scope", scope)
                ranked["ranking_is_ex_post_descriptive"] = True
                ranked["deployable_selection"] = False
                rows.append(ranked)
                summaries.append(
                    {
                        "ranking_scope": scope,
                        "minimum_active_quarters": threshold,
                        "rank_type": rank_type,
                        "eligible_pair_count": len(ranked),
                        "ranking_is_ex_post_descriptive": True,
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def _internal_accounting_reconciliation(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    transaction_cost_bps: float = 2.0,
    annual_short_borrow_bps: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, group in daily.groupby("pair", sort=True):
        work = group.sort_values("date")
        ra = pd.to_numeric(work["price_a"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
        rb = pd.to_numeric(work["price_b"], errors="coerce").pct_change(fill_method=None).fillna(0.0)
        gross = work["weight_a"].to_numpy() * ra.to_numpy() + work["weight_b"].to_numpy() * rb.to_numpy()
        denominator = 1.0 + gross
        drift_a = np.where(np.abs(denominator) > 1e-12, work["weight_a"].to_numpy() * (1.0 + ra.to_numpy()) / denominator, work["weight_a"].to_numpy())
        drift_b = np.where(np.abs(denominator) > 1e-12, work["weight_b"].to_numpy() * (1.0 + rb.to_numpy()) / denominator, work["weight_b"].to_numpy())
        turnover = np.abs(work["target_weight_a"].to_numpy() - drift_a) + np.abs(work["target_weight_b"].to_numpy() - drift_b)
        transaction_cost = (
            turnover
            * float(transaction_cost_bps)
            / 10_000.0
            * (1.0 + gross)
        )
        borrow = (np.maximum(-work["weight_a"].to_numpy(), 0.0) + np.maximum(-work["weight_b"].to_numpy(), 0.0)) * float(annual_short_borrow_bps) / 10_000.0 / 252.0
        net = gross - transaction_cost - borrow
        pair_trades = trades.loc[trades["pair"].astype(str).eq(str(pair))].sort_values(
            "trade_number"
        )
        daily_factor_1bp = float(
            np.prod(1.0 + work["net_return_1bp"].to_numpy(dtype=float))
        )
        daily_factor_2bp = float(
            np.prod(1.0 + work["net_return_2bp"].to_numpy(dtype=float))
        )
        ledger_factor_1bp = float(
            np.prod(1.0 + pair_trades["net_return_1bp"].to_numpy(dtype=float))
        ) if len(pair_trades) else 1.0
        ledger_factor_2bp = float(
            np.prod(1.0 + pair_trades["net_return_2bp"].to_numpy(dtype=float))
        ) if len(pair_trades) else 1.0
        rows.append(
            {
                "pair": pair,
                "maximum_gross_error": float(np.max(np.abs(gross - work["gross_return"].to_numpy()))),
                "maximum_turnover_error": float(np.max(np.abs(turnover - work["turnover"].to_numpy()))),
                "maximum_transaction_cost_error": float(
                    np.max(
                        np.abs(
                            transaction_cost
                            - work["transaction_cost"].to_numpy()
                        )
                    )
                ),
                "maximum_borrow_error": float(np.max(np.abs(borrow - work["borrow_cost"].to_numpy()))),
                "maximum_net_error": float(np.max(np.abs(net - work["net_return_2bp"].to_numpy()))),
                "maximum_trade_episode_path_factor_error_1bp": abs(
                    ledger_factor_1bp - daily_factor_1bp
                ),
                "maximum_trade_episode_path_factor_error_2bp": abs(
                    ledger_factor_2bp - daily_factor_2bp
                ),
            }
        )
    audit = pd.DataFrame(rows)
    tolerance = 1e-10
    maxima = {column: float(audit[column].max()) for column in audit.columns if column.startswith("maximum_")}
    samples = []
    if len(trades):
        clean_boundary = ~trades["entry_transition"].eq(
            "direction_reversal"
        ) & ~trades["exit_transition"].eq("direction_reversal")
        selectors = {
            "losing": trades["net_return_2bp"].lt(0) & clean_boundary,
            "long_a": trades["direction_ticker_a"].eq("long") & clean_boundary,
            "short_a": trades["direction_ticker_a"].eq("short") & clean_boundary,
        }
        for label, mask in selectors.items():
            subset = trades.loc[mask]
            if len(subset):
                row = subset.iloc[0]
                segment = daily.loc[
                    daily["pair"].eq(row["pair"])
                    & daily["date"].between(pd.Timestamp(row["entry_date"]), pd.Timestamp(row["exit_date"]))
                ]
                reconstructed = float((1.0 + segment["net_return_2bp"]).prod() - 1.0)
                samples.append({"sample_type": label, "pair": row["pair"], "ledger_return": row["net_return_2bp"], "reconstructed_return": reconstructed, "absolute_error": abs(reconstructed - row["net_return_2bp"])})
    maximum_sample = max((row["absolute_error"] for row in samples), default=0.0)
    episode_path_error = max(
        maxima.get("maximum_trade_episode_path_factor_error_1bp", 0.0),
        maxima.get("maximum_trade_episode_path_factor_error_2bp", 0.0),
    )
    summary = {
        **maxima,
        "maximum_sample_trade_error": maximum_sample,
        "sample_trade_count": len(samples),
        "trade_episode_path_factor_error": episode_path_error,
        "trade_episode_attribution_convention": (
            "directional_episode_sequential_flip_factor_v1"
        ),
    }
    summary["passed"] = max([*maxima.values(), maximum_sample], default=0.0) <= tolerance
    return pd.concat([audit.assign(sample_type="full_path"), pd.DataFrame(samples)], ignore_index=True, sort=False), summary


def _v6_control_reconciliation(
    candidate_metrics: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    control = candidate_metrics.loc[candidate_metrics["strategy_id"].eq("S0_v6")].copy()
    control["rebalance_date"] = pd.to_datetime(control["rebalance_date"])
    source = quarterly.loc[
        quarterly["evaluation_success"].astype(str).str.lower().isin({"true", "1"}),
        ["pair", "rebalance_date", "q_return_1bp", "q_return_2bp"],
    ].copy()
    source["rebalance_date"] = pd.to_datetime(source["rebalance_date"])
    audit = control.merge(
        source,
        on=["pair", "rebalance_date"],
        how="outer",
        suffixes=("_v8_control", "_v6_source"),
        indicator=True,
        validate="one_to_one",
    )
    for cost in (1, 2):
        audit[f"absolute_error_{cost}bp"] = (
            pd.to_numeric(audit[f"q_return_{cost}bp_v8_control"], errors="coerce")
            - pd.to_numeric(audit[f"q_return_{cost}bp_v6_source"], errors="coerce")
        ).abs()
    summary = {
        "expected_successful_pair_quarters": int(len(source)),
        "matched_pair_quarters": int(audit["_merge"].eq("both").sum()),
        "unmatched_pair_quarters": int((~audit["_merge"].eq("both")).sum()),
        "maximum_absolute_error_1bp": float(audit["absolute_error_1bp"].max()),
        "maximum_absolute_error_2bp": float(audit["absolute_error_2bp"].max()),
    }
    summary["passed"] = (
        summary["unmatched_pair_quarters"] == 0
        and summary["maximum_absolute_error_1bp"] <= 1e-10
        and summary["maximum_absolute_error_2bp"] <= 1e-10
    )
    return audit, summary


def _v8_v6_rank_comparison(v8_metrics: pd.DataFrame, v6_metrics: pd.DataFrame) -> pd.DataFrame:
    def ranked(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        primary = frame.loc[pd.to_numeric(frame["transaction_cost_bps"], errors="coerce").eq(2.0)].copy()
        primary = primary.sort_values(["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort")
        primary[f"{prefix}_rank"] = np.arange(1, len(primary) + 1)
        return primary[
            [
                "pair",
                f"{prefix}_rank",
                "calendar_cagr",
                "sharpe",
                "max_drawdown",
                "trade_count",
            ]
        ].rename(
            columns={
                "calendar_cagr": f"{prefix}_cagr",
                "sharpe": f"{prefix}_sharpe",
                "max_drawdown": f"{prefix}_max_drawdown",
                "trade_count": f"{prefix}_trade_count",
            }
        )

    comparison = ranked(v8_metrics, "v8").merge(ranked(v6_metrics, "v6"), on="pair", validate="one_to_one")
    comparison["rank_change_v8_minus_v6"] = comparison["v8_rank"] - comparison["v6_rank"]
    comparison["rank_improvement"] = comparison["v6_rank"] - comparison["v8_rank"]
    comparison["cagr_delta_v8_minus_v6"] = comparison["v8_cagr"] - comparison["v6_cagr"]
    return comparison.sort_values("v8_rank")


def _charts(daily: pd.DataFrame, rankings: pd.DataFrame, root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    headline = rankings.loc[
        rankings["ranking_scope"].eq("fixed_114_primary")
        & rankings["minimum_active_quarters"].eq(20)
        & rankings["rank_type"].eq("full_calendar_cagr")
        & rankings["rank"].le(10)
    ].sort_values("rank")
    paths: list[Path] = []
    if len(headline):
        figure, axis = plt.subplots(figsize=(12, 6.5))
        for pair in headline["pair"]:
            subset = daily.loc[daily["pair"].eq(pair)].sort_values("date")
            axis.plot(subset["date"], subset["equity_2bp"], label=pair, linewidth=1.4)
        axis.axhline(1.0, color="#758195", linewidth=0.8)
        axis.set_title("V8 top-ten full-calendar equity curves · 2 bp")
        axis.set_ylabel("Growth of $1")
        axis.legend(ncol=2, fontsize=8)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = root / "v8_top10_equity_curves.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)
        figure, axis = plt.subplots(figsize=(11, 6))
        plot = headline.sort_values("calendar_cagr")
        axis.barh(plot["pair"], 100 * plot["calendar_cagr"], color="#34699a")
        axis.scatter(100 * plot["bootstrap_cagr_ci_low"], plot["pair"], color="#b23a48", label="95% bootstrap lower bound", zorder=3)
        axis.set_xlabel("Percent per year")
        axis.set_title("V8 CAGR and synchronized year-block lower bound")
        axis.legend()
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        path = root / "v8_top10_cagr_and_lcb.png"
        figure.savefig(path, dpi=170)
        plt.close(figure)
        paths.append(path)
    return paths


REPORT = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V8 pair-centric ranking</title>
<style>body{margin:0;background:#edf1f5;color:#172033;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.page{max-width:1180px;margin:auto;background:white;min-height:100vh}header{padding:42px 48px;background:#142f49;color:white}main{padding:28px 48px 60px}h1{margin:.2em 0}.alert,.note{padding:14px 16px;margin:15px 0;border-left:5px solid #b33c4a;background:#fff1f2}.note{border-color:#3570ad;background:#eef6ff}.table{overflow:auto;border:1px solid #dde4ec;border-radius:8px;margin-bottom:20px}.dataframe{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}.dataframe th{background:#f2f5f8}.dataframe th,.dataframe td{padding:8px;border-bottom:1px solid #e5e9ef;text-align:right}.dataframe th:first-child,.dataframe td:first-child,.dataframe th:nth-child(2),.dataframe td:nth-child(2){text-align:left}.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}.charts img{width:100%;border:1px solid #dde4ec;border-radius:8px}.links a{display:inline-block;padding:6px 9px;margin:3px;border:1px solid #d6dee8;border-radius:5px;text-decoration:none}@media(max-width:800px){.charts{grid-template-columns:1fr}header,main{padding-left:20px;padding-right:20px}}</style>
</head>
<body><div class="page">
<header><div>PAIR-CENTRIC · FULL PANEL · ADDITIVE V8</div><h1>Best pairs under historically selected strategies</h1><p>114 fixed identities × 41 adjacent quarters; complete-calendar ten-year ranking after replay.</p></header>
<main>
<div class="alert"><strong>Research status: NOT VALIDATED.</strong> The full-panel view is ex-post descriptive: its fixed 114 identities and 2016–2026 outcomes were visible during development. {{ minute_text }} Rankings are descriptive candidates for future paper trading.</div>
<div class="note">Each quarter retains V6's 97-parameter one-standard-error result. The structural wrapper is selected from only the pair's completed prior three-year candidate history. Old positions may finish after a quarter boundary; no same-pair positions overlap. Failed and unavailable quarters remain cash.</div>
<h2>Result in one paragraph</h2><p>{{ optimization_text }}</p><p>{{ audit_text }}</p>
<h2>Top ten · full-panel ex-post descriptive · 2 bp</h2><div class="table">{{ cagr_table }}</div>
<h2>Post-discovery safe-reset frozen-path diagnostic · fixed legacy 114 · 2 bp</h2>
<div class="note"><strong>Diagnostic scope only.</strong> Each pair stays cash through its first clean flat state on or after <code>master.first_successful_v6_tuned_date</code>, then reuses the already-frozen full-panel path. The strategy-selection history still contains pre-discovery candidate outcomes and the identities remain the fixed legacy 114. This diagnostic is not a leakage-free or point-in-time strategy test and does not establish deployability.</div>
<p>{{ safe_reset_text }}</p><div class="table">{{ safe_reset_table }}</div><p>{{ safe_reset_inference_text }}</p>
<h2>V8 versus unchanged V6 · same 114 identities</h2><p>Ranks below use the complete ten-year 2 bp CAGR across all 114 identities. Negative rank change means the pair moved up.</p><div class="table">{{ comparison_table }}</div>
<h2>Top ten · bootstrap lower bound</h2><div class="table">{{ lcb_table }}</div>
<h2>Top ten · risk-adjusted consistency</h2><div class="table">{{ consistency_table }}</div>
<h2>Independent cost and borrow stress · frozen paths</h2><p>{{ stress_text }}</p><div class="table">{{ stress_table }}</div>
<h2>Strategy-selection distribution</h2><div class="table">{{ strategy_table }}</div>
<h2>Multiple-testing audit · full-panel ex-post descriptive</h2><p>{{ inference_text }}</p>
<p><strong>Decision:</strong> the leading pairs can enter a forward paper-trading watchlist under the frozen V8 protocol. This study does not authorize live capital.</p>
<div class="charts">{% for chart in charts %}<img src="charts/{{ chart }}" alt="{{ chart }}">{% endfor %}</div>
<h2>Machine-readable artifacts</h2><div class="links">{% for link in links %}<a href="data/processed/{{ link }}">{{ link }}</a>{% endfor %}</div>
</main></div></body></html>"""
)


def _format_ranking(frame: pd.DataFrame) -> str:
    columns = [
        "rank",
        "pair",
        "calendar_cagr",
        "bootstrap_cagr_ci_low",
        "sharpe",
        "max_drawdown",
        "active_quarters",
        "trade_count",
        "modal_strategy_id",
        "cagr_without_best_quarter",
        "cagr_without_best_complete_year",
    ]
    output = frame[columns].copy()
    for column in ["calendar_cagr", "bootstrap_cagr_ci_low", "max_drawdown", "cagr_without_best_quarter", "cagr_without_best_complete_year"]:
        output[column] = output[column].map(lambda value: f"{value:.2%}")
    output["sharpe"] = output["sharpe"].map(lambda value: f"{value:.2f}")
    return output.to_html(index=False, border=0)


def _format_comparison(frame: pd.DataFrame) -> str:
    columns = [
        "v8_rank",
        "pair",
        "v8_cagr",
        "v6_rank",
        "v6_cagr",
        "rank_change_v8_minus_v6",
        "cagr_delta_v8_minus_v6",
    ]
    output = frame.sort_values("v8_rank").head(10)[columns].copy()
    for column in ["v8_cagr", "v6_cagr", "cagr_delta_v8_minus_v6"]:
        output[column] = output[column].map(lambda value: f"{value:.2%}")
    return output.to_html(index=False, border=0)


def _run_independent_reconstruction(
    config_file: Path,
    price_file: Path,
    processed: Path,
    *,
    expected_config_sha256: str,
    expected_price_sha256: str,
) -> tuple[dict[str, Any], list[str], str]:
    command = [
        sys.executable,
        str(INDEPENDENT_AUDIT_SCRIPT),
        "--config",
        str(config_file),
        "--prices",
        str(price_file),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    summary_path = processed / INDEPENDENT_AUDIT_SUMMARY
    detail_path = processed / INDEPENDENT_AUDIT_DETAIL
    if completed.returncode != 0:
        raise RuntimeError(
            "independent V8 reconstruction failed with exit code "
            f"{completed.returncode}; stdout={completed.stdout[-2000:]!r}; "
            f"stderr={completed.stderr[-2000:]!r}"
        )
    if not summary_path.is_file() or not detail_path.is_file():
        raise RuntimeError(
            "independent V8 reconstruction did not create both required artifacts"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(
            f"independent V8 reconstruction did not pass: {summary}"
        )
    required_categories = {"winning", "losing", "long_a", "short_a"}
    selected_categories = {
        str(row.get("category")) for row in summary.get("selected_trades", [])
    }
    if not required_categories.issubset(selected_categories):
        raise RuntimeError(
            "independent V8 reconstruction lacks required trade categories: "
            f"{sorted(required_categories - selected_categories)}"
        )
    expected_inputs = {
        "config": config_file,
        "daily": processed / "v8_pair_daily_paths.csv.gz",
        "trades": processed / "v8_trade_ledger.csv",
        "prices": price_file,
    }
    recorded_inputs = summary.get("input_files", {})
    for label, path in expected_inputs.items():
        recorded = recorded_inputs.get(label, {})
        if recorded.get("sha256") != _sha256(path):
            raise RuntimeError(
                f"independent V8 reconstruction summary has stale {label} lineage"
            )
    if (
        recorded_inputs["config"]["sha256"] != expected_config_sha256
        or recorded_inputs["prices"]["sha256"] != expected_price_sha256
    ):
        raise RuntimeError(
            "independent V8 reconstruction did not use the startup config/price snapshots"
        )
    detail_rows = len(pd.read_csv(detail_path))
    daily_rows = len(pd.read_csv(expected_inputs["daily"]))
    if (
        int(summary.get("daily_row_count", -1)) != daily_rows
        or detail_rows != daily_rows
    ):
        raise RuntimeError(
            "independent V8 reconstruction row counts do not match daily paths"
        )
    summary["required_trade_categories"] = sorted(required_categories)
    summary["required_trade_categories_verified"] = True
    summary["summary_artifact_sha256"] = _sha256(summary_path)
    summary["detail_artifact_sha256"] = _sha256(detail_path)
    return summary, command, completed.stdout


def _run_locked(
    config_file: Path,
    config: Mapping[str, Any],
    config_raw: bytes,
    config_sha256: str,
    invocation_nonce: str,
    *,
    workers: int,
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    workspace = PROJECT_ROOT
    identity_root = workspace / config["inputs"]["identity_root"]
    output_root = workspace / config["outputs"]["root"]
    processed = output_root / "data/processed"
    cache_root = output_root / "data/cache"
    chart_root = output_root / "charts"
    for path in (processed, cache_root, chart_root):
        path.mkdir(parents=True, exist_ok=True)
    source = config["inputs"]
    master_path = identity_root / source["master"]
    schedule_path = identity_root / source["schedule"]
    quarterly_path = identity_root / source["quarterly"]
    price_path = identity_root / source["price_snapshot"]
    v6_cache = identity_root / source["v6_pair_quarter_cache"]
    v6_metrics_path = identity_root / "data/processed/pair_identity_metrics.csv"
    minute_path = workspace / source["minute_history"]
    snapshot_paths = [
        master_path,
        schedule_path,
        quarterly_path,
        price_path,
        v6_metrics_path,
    ]
    missing = [str(path) for path in snapshot_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing V8 input: " + ", ".join(missing))
    snapshots: dict[str, bytes] = {str(config_file): config_raw}
    file_hashes_before = {str(config_file): str(config_sha256)}
    for path in snapshot_paths:
        raw, sha256 = _snapshot_bytes(path)
        snapshots[str(path)] = raw
        file_hashes_before[str(path)] = sha256
    minute_raw: bytes | None = None
    if minute_path.is_file():
        minute_raw, minute_sha256 = _snapshot_bytes(minute_path)
        file_hashes_before[str(minute_path)] = minute_sha256
    else:
        file_hashes_before[str(minute_path)] = "MISSING"
    run_snapshot_root = cache_root / "input_snapshots" / invocation_nonce
    frozen_config_path = run_snapshot_root / "run_config.json"
    frozen_price_path = run_snapshot_root / "adjusted_close.csv"
    _atomic_bytes(config_raw, frozen_config_path)
    _atomic_bytes(snapshots[str(price_path)], frozen_price_path)
    engine_manifest_before = _engine_source_manifest()
    runtime_manifest_before = _runtime_manifest()
    runtime_manifest_sha256 = _canonical_payload_sha256(
        runtime_manifest_before
    )
    master = _read_csv_snapshot(snapshots[str(master_path)], master_path)
    schedule = _read_csv_snapshot(
        snapshots[str(schedule_path)],
        schedule_path,
        parse_dates=[
            "rebalance_date",
            "outer_end_exclusive",
            "formation_start_requested",
        ],
    )
    quarterly = _read_csv_snapshot(
        snapshots[str(quarterly_path)], quarterly_path
    )
    prices = _read_csv_snapshot(
        snapshots[str(price_path)],
        price_path,
        index_col="date",
        parse_dates=True,
    ).sort_index()
    v6_metrics = _read_csv_snapshot(
        snapshots[str(v6_metrics_path)], v6_metrics_path
    )
    # The minute audit runs once against the same startup snapshot boundary.
    # No downstream stage reads this optional input again.
    if minute_raw is None:
        minute_audit = audit_minute_history(minute_path)
    else:
        frozen_minute_path = run_snapshot_root / "minute_history.parquet"
        _atomic_bytes(minute_raw, frozen_minute_path)
        minute_audit = audit_minute_history(frozen_minute_path)
        minute_audit["path"] = str(minute_path.resolve())
        minute_audit["snapshot_sha256"] = file_hashes_before[str(minute_path)]
    for column in ["rebalance_date", "outer_end_exclusive", "formation_start_requested"]:
        quarterly[column] = pd.to_datetime(quarterly[column])
    if len(master) != int(config["evaluation"]["expected_pairs"]) or master["pair"].nunique() != len(master):
        raise RuntimeError("fixed master is not exactly 114 unique pairs")
    if len(schedule) != int(config["evaluation"]["expected_quarters"]):
        raise RuntimeError("quarter schedule is not exactly 41 intervals")
    if len(quarterly) != int(config["evaluation"]["expected_pair_quarters"]):
        raise RuntimeError("quarterly panel is not exactly 4,674 rows")
    if quarterly.duplicated(["pair", "rebalance_date"]).any():
        raise RuntimeError("quarterly panel contains duplicate keys")
    successful = quarterly.loc[quarterly["evaluation_success"].astype(str).str.lower().isin({"true", "1"})].copy()
    s5_manifest = _s5_ranking_manifest(
        successful, v6_cache, config["strategy_candidates"]
    )
    s5_manifest_sha256 = _canonical_payload_sha256(
        s5_manifest.to_dict("records")
    )
    s5_sha_by_key = {
        (str(row.pair), str(pd.Timestamp(row.rebalance_date).date())): str(row.sha256)
        for row in s5_manifest.itertuples(index=False)
    }
    s5_hashes_before = {
        str(row.path): str(row.sha256) for row in s5_manifest.itertuples(index=False)
    }
    base = _base_config(config)
    initializer = (
        str(price_path),
        base.__dict__,
        list(config["strategy_candidates"]),
        str(v6_cache),
        file_hashes_before[str(price_path)],
        s5_sha_by_key,
        runtime_manifest_sha256,
    )
    fingerprint = _fingerprint(
        config_file,
        [master_path, schedule_path, quarterly_path, price_path, v6_metrics_path],
        engine_manifest=engine_manifest_before,
        s5_ranking_manifest_sha256=s5_manifest_sha256,
        runtime_manifest=runtime_manifest_before,
        config_sha256=config_sha256,
        source_sha256_by_path=file_hashes_before,
    )
    candidate_tasks = []
    for row in successful.sort_values(["rebalance_date", "pair"]).to_dict("records"):
        target = cache_root / "candidate" / _safe(row["pair"]) / pd.Timestamp(row["rebalance_date"]).strftime("%Y%m%d")
        row_payload_sha256 = _canonical_payload_sha256(row)
        s5_sha256 = s5_sha_by_key.get(
            (str(row["pair"]), str(pd.Timestamp(row["rebalance_date"]).date())),
            "NOT_APPLICABLE",
        )
        task_fingerprint = _canonical_payload_sha256(
            {
                "base_fingerprint": fingerprint,
                "row_payload_sha256": row_payload_sha256,
                "s5_parameter_rankings_sha256": s5_sha256,
            }
        )
        candidate_tasks.append(
            (
                row,
                str(target),
                {
                    "fingerprint": task_fingerprint,
                    "payload_sha256": row_payload_sha256,
                    "run_nonce": invocation_nonce,
                    "row_payload_sha256": row_payload_sha256,
                    "s5_parameter_rankings_sha256": s5_sha256,
                },
            )
        )
    if limit_tasks is not None:
        candidate_tasks = candidate_tasks[: int(limit_tasks)]
    candidate_results = _run_parallel(
        candidate_tasks,
        _candidate_task,
        workers=workers,
        initializer_args=initializer,
        label="candidate replay",
    )
    _validate_input_stage_and_promote_cache(
        stage="candidate",
        results=candidate_results,
        file_hashes_before=file_hashes_before,
        engine_manifest_before=engine_manifest_before,
        s5_hashes_before=s5_hashes_before,
        runtime_manifest_before=runtime_manifest_before,
        audit_path=processed / "v8_candidate_input_mutation_audit.csv.gz",
    )
    candidate_metrics = pd.concat(
        [row["_validated_frame"] for row in candidate_results],
        ignore_index=True,
    )
    if limit_tasks is not None:
        _atomic_csv(candidate_metrics, processed / "v8_smoke_candidate_metrics.csv")
        return {
            "smoke": True,
            "tasks": len(candidate_tasks),
            "candidate_rows": len(candidate_metrics),
            "invocation_nonce": invocation_nonce,
        }
    expected_candidate_rows = len(successful) * len(config["strategy_candidates"])
    if len(candidate_metrics) != expected_candidate_rows:
        raise RuntimeError(f"candidate panel has {len(candidate_metrics)} rows; expected {expected_candidate_rows}")
    control_reconciliation, control_summary = _v6_control_reconciliation(candidate_metrics, quarterly)
    if not control_summary["passed"]:
        raise RuntimeError(f"V8 S0 control does not reproduce V6: {control_summary}")
    selections, selection_rankings = _build_strategy_selection(quarterly, candidate_metrics, config)
    final_base_fingerprint = _canonical_payload_sha256(
        {
            "candidate_base_fingerprint": fingerprint,
            "candidate_metrics_sha256": _canonical_payload_sha256(
                candidate_metrics.sort_values(
                    ["pair", "rebalance_date", "strategy_id"], kind="mergesort"
                ).to_dict("records")
            ),
            "selection_history_sha256": _canonical_payload_sha256(
                selections.sort_values(
                    ["pair", "rebalance_date"], kind="mergesort"
                ).to_dict("records")
            ),
            "engine": "v8_final_stateful_engine_v5",
        }
    )
    final_tasks = []
    merged = quarterly.merge(
        selections[["pair", "rebalance_date", "selected_strategy_id"]],
        on=["pair", "rebalance_date"],
        how="left",
        validate="one_to_one",
    )
    for pair, group in merged.groupby("pair", sort=True):
        target = cache_root / "final" / _safe(pair)
        records = group.sort_values("rebalance_date", kind="mergesort").to_dict(
            "records"
        )
        merged_group_sha256 = _canonical_payload_sha256(records)
        final_fingerprint = _canonical_payload_sha256(
            {
                "final_base_fingerprint": final_base_fingerprint,
                "merged_group_sha256": merged_group_sha256,
            }
        )
        final_tasks.append(
            (
                records,
                str(target),
                {
                    "fingerprint": final_fingerprint,
                    "payload_sha256": merged_group_sha256,
                    "run_nonce": invocation_nonce,
                    "merged_group_sha256": merged_group_sha256,
                },
            )
        )
    final_results = _run_parallel(
        final_tasks,
        _final_pair_task,
        workers=workers,
        initializer_args=initializer,
        label="stateful pair replay",
    )
    _validate_input_stage_and_promote_cache(
        stage="stateful_pair",
        results=final_results,
        file_hashes_before=file_hashes_before,
        engine_manifest_before=engine_manifest_before,
        s5_hashes_before=s5_hashes_before,
        runtime_manifest_before=runtime_manifest_before,
        audit_path=processed / "v8_stateful_input_mutation_audit.csv.gz",
    )
    sparse_frames = [row["_validated_frame"] for row in final_results]
    sparse = pd.concat(sparse_frames, ignore_index=True, sort=False)
    artifacts = _aggregate_paths(master, schedule, quarterly, sparse, prices, selections, config)
    rankings, ranking_summary = _rank_pairs(artifacts["metrics"], master)
    v8_v6_comparison = _v8_v6_rank_comparison(artifacts["metrics"], v6_metrics)
    inference, inference_summary = _romano_wolf_from_year_blocks(
        artifacts["blocks"], config
    )
    inference = inference.rename(columns={"pair": "pair_identity"})
    safe_reset_artifacts, safe_reset_summary = (
        _build_post_discovery_safe_reset_artifacts(
            artifacts["daily"], master, config
        )
    )
    cost_borrow_stress = _build_cost_borrow_stress_metrics(
        artifacts["daily"], master, config
    )
    cost_borrow_stress_summary = (
        cost_borrow_stress.groupby(
            ["transaction_cost_bps", "annual_short_borrow_bps"], as_index=False
        )
        .agg(
            pair_count=("pair", "nunique"),
            median_calendar_cagr=("calendar_cagr", "median"),
            worst_calendar_cagr=("calendar_cagr", "min"),
            best_calendar_cagr=("calendar_cagr", "max"),
            positive_cagr_pair_count=("calendar_cagr", lambda values: int((values > 0).sum())),
        )
        .sort_values(["transaction_cost_bps", "annual_short_borrow_bps"])
        .reset_index(drop=True)
    )
    internal_reconciliation, internal_reconciliation_summary = (
        _internal_accounting_reconciliation(
            artifacts["daily"],
            artifacts["trades"],
            transaction_cost_bps=float(
                config["evaluation"]["primary_transaction_cost_bps"]
            ),
            annual_short_borrow_bps=float(
                config["evaluation"]["annual_short_borrow_bps"]
            ),
        )
    )
    if not internal_reconciliation_summary["passed"]:
        raise RuntimeError(
            "V8 internal accounting reconciliation failed: "
            f"{internal_reconciliation_summary}"
        )
    primary_stress_error = float(
        pd.to_numeric(
            cost_borrow_stress.loc[
                cost_borrow_stress["matches_primary_ranking_assumptions"],
                "maximum_primary_net_return_identity_error",
            ],
            errors="coerce",
        ).max()
    )
    if not np.isfinite(primary_stress_error) or primary_stress_error > 1e-10:
        raise RuntimeError(
            "primary cost/borrow stress path does not reproduce net_return_2bp; "
            f"maximum error={primary_stress_error}"
        )
    strategy_distribution = selections.groupby("selected_strategy_id", as_index=False).agg(
        scheduled_selected_quarters=("pair", "size"),
        scheduled_selected_pairs=("pair", "nunique"),
    )
    executable_distribution = (
        selections.loc[selections["evaluation_success"]]
        .groupby("selected_strategy_id", as_index=False)
        .agg(executable_selected_quarters=("pair", "size"), executable_selected_pairs=("pair", "nunique"))
    )
    strategy_distribution = strategy_distribution.merge(executable_distribution, on="selected_strategy_id", how="left")
    strategy_distribution[["executable_selected_quarters", "executable_selected_pairs"]] = strategy_distribution[
        ["executable_selected_quarters", "executable_selected_pairs"]
    ].fillna(0).astype(int)
    strategy_distribution["scheduled_fraction"] = strategy_distribution["scheduled_selected_quarters"] / len(selections)
    strategy_distribution["executable_fraction"] = strategy_distribution["executable_selected_quarters"] / int(selections["evaluation_success"].sum())
    charts = _charts(artifacts["daily"], rankings, chart_root)
    _atomic_csv(master, processed / "v8_pair_master.csv")
    _atomic_csv(schedule, processed / "v8_rebalance_schedule.csv")
    _atomic_csv(candidate_metrics, processed / "v8_candidate_quarter_metrics.csv.gz")
    _atomic_csv(selections, processed / "v8_strategy_selection_history.csv")
    _atomic_csv(selection_rankings, processed / "v8_strategy_candidate_rankings.csv.gz")
    _atomic_csv(sparse, processed / "v8_stateful_sparse_daily.csv.gz")
    for key, name in {
        "daily": "v8_pair_daily_paths.csv.gz",
        "quarterly": "v8_pair_quarterly_results.csv.gz",
        "annual": "v8_pair_annual_returns.csv",
        "metrics": "v8_pair_metrics.csv",
        "trades": "v8_trade_ledger.csv",
        "blocks": "v8_year_blocks.csv",
        "confidence": "v8_confidence_intervals.csv",
    }.items():
        _atomic_csv(artifacts[key], processed / name)
    _atomic_csv(rankings, processed / "v8_pair_rankings.csv")
    _atomic_csv(ranking_summary, processed / "v8_ranking_summary.csv")
    _atomic_csv(inference, processed / "v8_romano_wolf_pair_family.csv")
    _atomic_csv(
        internal_reconciliation,
        processed / "v8_internal_accounting_reconciliation.csv",
    )
    _atomic_csv(control_reconciliation, processed / "v8_v6_control_reconciliation.csv")
    _atomic_csv(v8_v6_comparison, processed / "v8_vs_v6_rank_comparison.csv")
    _atomic_csv(strategy_distribution, processed / "v8_strategy_distribution.csv")
    for key, name in {
        "daily": "v8_post_discovery_safe_reset_daily_paths.csv.gz",
        "metrics": "v8_post_discovery_safe_reset_pair_metrics.csv",
        "rankings": "v8_post_discovery_safe_reset_pair_rankings.csv",
        "blocks": "v8_post_discovery_safe_reset_year_blocks.csv",
        "confidence": "v8_post_discovery_safe_reset_confidence_intervals.csv",
        "inference": "v8_post_discovery_safe_reset_pair_inference.csv",
    }.items():
        _atomic_csv(safe_reset_artifacts[key], processed / name)
    _atomic_json(
        safe_reset_summary,
        processed / "v8_post_discovery_safe_reset_summary.json",
    )
    _atomic_csv(
        cost_borrow_stress,
        processed / "v8_cost_borrow_stress_pair_metrics.csv",
    )
    _atomic_csv(
        cost_borrow_stress_summary,
        processed / "v8_cost_borrow_stress_scenario_summary.csv",
    )
    _atomic_csv(
        s5_manifest,
        processed / "v8_s5_parameter_ranking_input_manifest.csv.gz",
    )
    _atomic_json(
        {
            "invocation_nonce": invocation_nonce,
            "candidate_base_fingerprint": fingerprint,
            "final_base_fingerprint": final_base_fingerprint,
            "engine_sources": engine_manifest_before,
            "runtime_manifest": runtime_manifest_before,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "s5_parameter_ranking_manifest_sha256": s5_manifest_sha256,
            "startup_config_snapshot": {
                "path": str(frozen_config_path),
                "sha256": config_sha256,
            },
            "startup_price_snapshot": {
                "path": str(frozen_price_path),
                "sha256": file_hashes_before[str(price_path)],
            },
        },
        processed / "v8_engine_source_manifest.json",
    )

    independent_reconstruction, independent_command, independent_stdout = (
        _run_independent_reconstruction(
            frozen_config_path,
            frozen_price_path,
            processed,
            expected_config_sha256=config_sha256,
            expected_price_sha256=file_hashes_before[str(price_path)],
        )
    )

    input_mutation_audit = _validate_input_stage_and_promote_cache(
        stage="report",
        results=[],
        file_hashes_before=file_hashes_before,
        engine_manifest_before=engine_manifest_before,
        s5_hashes_before=s5_hashes_before,
        runtime_manifest_before=runtime_manifest_before,
        audit_path=processed / "v8_declared_input_mutation_audit.csv.gz",
    )
    all_declared_inputs_unchanged = bool(input_mutation_audit["unchanged"].all())
    quality = {
        "research_status_verdict": "NOT VALIDATED",
        "full_panel_evaluation_view": {
            "evaluation_view": "full_panel_ex_post_descriptive",
            "ex_post_descriptive": True,
            "complete_point_in_time_universe": False,
            "deployable_selection": False,
        },
        "post_discovery_safe_reset_frozen_path_diagnostic": safe_reset_summary,
        "cost_borrow_stress": {
            "artifact": "v8_cost_borrow_stress_pair_metrics.csv",
            "transaction_cost_bps": sorted(
                cost_borrow_stress["transaction_cost_bps"].unique().tolist()
            ),
            "annual_short_borrow_bps": sorted(
                cost_borrow_stress["annual_short_borrow_bps"].unique().tolist()
            ),
            "scenario_count": int(
                cost_borrow_stress[
                    ["transaction_cost_bps", "annual_short_borrow_bps"]
                ].drop_duplicates().shape[0]
            ),
            "pair_metric_rows": int(len(cost_borrow_stress)),
            "frozen_path_repricing": True,
            "primary_ranking_transaction_cost_bps": float(
                config["evaluation"]["primary_transaction_cost_bps"]
            ),
            "primary_ranking_annual_short_borrow_bps": float(
                config["evaluation"]["annual_short_borrow_bps"]
            ),
            "stress_scenarios_used_to_construct_primary_ranking": False,
            "maximum_primary_net_return_identity_error": primary_stress_error,
        },
        "fixed_pair_count": len(master),
        "quarter_count": len(schedule),
        "pair_quarter_count": len(quarterly),
        "successful_candidate_pair_quarters": len(successful),
        "candidate_strategy_count": len(config["strategy_candidates"]),
        "candidate_metric_rows": len(candidate_metrics),
        "full_calendar_pair_metric_rows": len(artifacts["metrics"]),
        "all_declared_inputs_unchanged": all_declared_inputs_unchanged,
        "declared_input_mutation_audit": (
            "v8_declared_input_mutation_audit.csv.gz"
        ),
        "cache_publication_control": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "invocation_nonce": invocation_nonce,
            "exclusive_output_root_lock": True,
            "provisional_worker_markers": True,
            "promotion_requires_unchanged_stage_inputs": True,
            "candidate_stage_audit": "v8_candidate_input_mutation_audit.csv.gz",
            "stateful_stage_audit": "v8_stateful_input_mutation_audit.csv.gz",
            "report_stage_audit": "v8_declared_input_mutation_audit.csv.gz",
            "worker_price_snapshot_hash_verified": True,
            "worker_s5_snapshot_hash_verified": True,
        },
        "immutable_run_input_snapshots": {
            "root": str(run_snapshot_root),
            "config_sha256": config_sha256,
            "price_sha256": file_hashes_before[str(price_path)],
            "independent_audit_uses_snapshots": True,
        },
        "engine_source_manifest": {
            "artifact": "v8_engine_source_manifest.json",
            "source_count": len(engine_manifest_before),
            "source_sha256": engine_manifest_before,
            "runtime_manifest": runtime_manifest_before,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "candidate_base_fingerprint": fingerprint,
            "final_base_fingerprint": final_base_fingerprint,
        },
        "s5_parameter_ranking_input_manifest": {
            "artifact": "v8_s5_parameter_ranking_input_manifest.csv.gz",
            "row_count": len(s5_manifest),
            "manifest_sha256": s5_manifest_sha256,
        },
        "v7_root_unused": {
            "configured_path": source.get("v7_root"),
            "used_by_runner": False,
            "included_in_input_unchanged_claim": False,
        },
        "v6_control_reconciliation": control_summary,
        "internal_accounting_reconciliation": internal_reconciliation_summary,
        "independent_raw_close_weight_reconstruction": {
            **independent_reconstruction,
            "command": independent_command,
            "stdout": independent_stdout.strip(),
        },
        "minute_history_passed": minute_audit["passed"],
        "multiple_testing": inference_summary,
        "critical_limitations": [
            "fixed pair universe is ex-post and based on current constituents",
            "post-discovery safe-reset results remain conditional on the fixed legacy 114 identities and are not a complete point-in-time universe",
            "post-discovery safe-reset paths retain strategy selections informed by pre-discovery candidate outcomes and are not a leakage-free strategy test",
            "2016-2026 outcomes are a development sample",
            "historical one-minute data is absent",
            "primary 2 bp execution and 30 bp borrow remain proxy assumptions; stress scenarios do not establish actual fills",
        ],
    }
    _atomic_json(quality, processed / "v8_quality_checks.json")
    _atomic_json(minute_audit, processed / "v8_minute_data_audit.json")
    def top(rank_type: str) -> pd.DataFrame:
        return rankings.loc[
            rankings["ranking_scope"].eq("fixed_114_primary")
            & rankings["minimum_active_quarters"].eq(20)
            & rankings["rank_type"].eq(rank_type)
            & rankings["rank"].le(10)
        ].sort_values("rank")
    improved_count = int(v8_v6_comparison["cagr_delta_v8_minus_v6"].gt(0).sum())
    worsened_count = int(v8_v6_comparison["cagr_delta_v8_minus_v6"].lt(0).sum())
    leading = v8_v6_comparison.sort_values("v8_rank").iloc[0]
    safe_reset_rankings = safe_reset_artifacts["rankings"]
    safe_reset_leading = safe_reset_rankings.iloc[0]
    safe_reset_inference = safe_reset_summary["inference"]
    independent_daily_maximum = max(
        float(values["maximum_absolute_error"])
        for values in independent_reconstruction["daily_checks"].values()
    )
    activation_counts = safe_reset_summary["safe_activation_status_counts"]
    report = REPORT.render(
        cagr_table=_format_ranking(top("full_calendar_cagr")),
        comparison_table=_format_comparison(v8_v6_comparison),
        lcb_table=_format_ranking(top("robust_lower_bound")),
        consistency_table=_format_ranking(top("risk_adjusted_consistency")),
        strategy_table=strategy_distribution.to_html(
            index=False,
            border=0,
            formatters={
                "scheduled_fraction": lambda value: f"{value:.2%}",
                "executable_fraction": lambda value: f"{value:.2%}",
            },
        ),
        optimization_text=(
            f"{leading['pair']} remains the highest full-calendar pair at {leading['v8_cagr']:.2%} net CAGR. "
            f"The prior-history strategy wrapper improved {improved_count} identities and reduced {worsened_count}; "
            f"the median V8-minus-V6 CAGR change was {v8_v6_comparison['cagr_delta_v8_minus_v6'].median():+.2%}. "
            "The optimizer therefore adds value selectively and is not a blanket upgrade over V6."
        ),
        audit_text=(
            f"The S0 control reproduced all {control_summary['matched_pair_quarters']:,} successful V6 pair-quarters "
            f"with maximum return error {max(control_summary['maximum_absolute_error_1bp'], control_summary['maximum_absolute_error_2bp']):.1e}. "
            f"The internal accounting reconciliation passed with maximum net-return error "
            f"{internal_reconciliation_summary['maximum_net_error']:.1e}. The separately invoked "
            f"raw-close/weight reconstruction also passed across "
            f"{independent_reconstruction['daily_row_count']:,} daily rows with maximum daily "
            f"error {independent_daily_maximum:.1e}; all required winning, losing, long-a, "
            "and short-a trade categories were present."
        ),
        safe_reset_table=_format_ranking(safe_reset_rankings.head(10)),
        safe_reset_text=(
            f"The gate sets {safe_reset_summary['diagnostic_pair_sessions_set_to_cash']:,} "
            "pair-sessions to cash while waiting for a clean flat reset; total post-discovery "
            f"wait is {safe_reset_summary['total_reset_wait_sessions']:,} pair-sessions. "
            f"Activation statuses are {activation_counts}. Within this conditional diagnostic, "
            f"{safe_reset_leading['pair']} leads at "
            f"{safe_reset_leading['calendar_cagr']:.2%} full-calendar net CAGR at 2 bp."
        ),
        safe_reset_inference_text=(
            "The same synchronized ten-anniversary-block bootstrap and inference "
            "settings produce "
            f"{safe_reset_inference['romano_wolf_rejection_count']} Romano–Wolf "
            "5% rejections and a White Reality Check p-value of "
            f"{safe_reset_inference['white_reality_check_pvalue']:.3f}. These inference "
            "results inherit the fixed-universe and pre-discovery selection-history limitations."
        ),
        stress_table=cost_borrow_stress_summary.to_html(
            index=False,
            border=0,
            formatters={
                "transaction_cost_bps": lambda value: f"{value:g}",
                "annual_short_borrow_bps": lambda value: f"{value:g}",
                "median_calendar_cagr": lambda value: f"{value:.2%}",
                "worst_calendar_cagr": lambda value: f"{value:.2%}",
                "best_calendar_cagr": lambda value: f"{value:.2%}",
            },
        ),
        stress_text=(
            f"Frozen gross returns, turnover, and held short notionals were repriced across "
            f"{len(cost_borrow_stress_summary)} cost/borrow scenarios. The main rankings "
            f"remain exclusively 2 bp transaction cost and 30 bp annual borrow."
        ),
        inference_text=f"Synchronized ten-year block test across 114 identities: Romano–Wolf 5% rejections = {inference_summary['romano_wolf_rejection_count']}; White Reality Check p-value = {inference_summary['white_reality_check_pvalue']:.3f}.",
        minute_text=(
            "The historical one-minute data audit passed."
            if minute_audit["passed"]
            else "Historical one-minute data did not pass its audit: "
            f"{minute_audit['error']}."
        ),
        charts=[path.name for path in charts],
        links=[path.name for path in sorted(processed.iterdir()) if path.is_file()],
    )
    report_path = output_root / config["outputs"]["report"]
    report_path.write_text(report, encoding="utf-8")
    return {
        "output_root": str(output_root),
        "report": str(report_path),
        "pair_count": len(master),
        "quarter_count": len(schedule),
        "candidate_metric_rows": len(candidate_metrics),
        "v6_control_reconciliation_passed": control_summary["passed"],
        "internal_accounting_reconciliation_passed": (
            internal_reconciliation_summary["passed"]
        ),
        "independent_reconstruction_passed": independent_reconstruction["passed"],
        "minute_history_passed": minute_audit["passed"],
        "post_discovery_safe_reset_pair_count": safe_reset_summary["pair_count"],
        "post_discovery_safe_reset_pair_sessions_set_to_cash": (
            safe_reset_summary["diagnostic_pair_sessions_set_to_cash"]
        ),
        "cost_borrow_stress_scenario_count": len(cost_borrow_stress_summary),
        "invocation_nonce": invocation_nonce,
    }


def run(
    config_path: Path,
    *,
    workers: int,
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    """Run one self-consistent invocation under an exclusive output lock."""

    config_file = (
        config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    )
    config_raw, config_sha256 = _snapshot_bytes(config_file)
    config = json.loads(config_raw.decode("utf-8"))
    output_root = PROJECT_ROOT / config["outputs"]["root"]
    invocation_nonce = uuid.uuid4().hex
    with _exclusive_output_run_lock(output_root, invocation_nonce):
        return _run_locked(
            config_file,
            config,
            config_raw,
            config_sha256,
            invocation_nonce,
            workers=max(1, int(workers)),
            limit_tasks=limit_tasks,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, workers=max(1, args.workers), limit_tasks=args.limit_tasks), indent=2))


if __name__ == "__main__":
    main()
