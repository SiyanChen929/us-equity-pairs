#!/usr/bin/env python3
"""Freeze and reveal the V10 strict joint max-Sharpe pair allocator."""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.run_pair_v9_walkforward_top10 as v9
from equity_pairs.walkforward_max_sharpe import MaxSharpeSpec, optimize_walkforward_books
from equity_pairs.walkforward_top10 import simulate_security_level_portfolio


DEFAULT_CONFIG = PROJECT_ROOT / "configs/pair_strategy_v10_joint_sharpe_top10.json"
OPTIMIZER_MODULE = PROJECT_ROOT / "src/equity_pairs/walkforward_max_sharpe.py"
V9_RUNNER = PROJECT_ROOT / "scripts/run_pair_v9_walkforward_top10.py"
V9_WALKFORWARD_MODULE = PROJECT_ROOT / "src/equity_pairs/walkforward_top10.py"
IDENTITY_MODULE = PROJECT_ROOT / "src/equity_pairs/identity_evaluation.py"
FREEZE_NAME = "protocol_freeze_manifest.json"
REVEAL_NAME = "reveal_manifest.json"

MAX_SHARPE_ID = "joint_max_sharpe_top10"
MATCHED_ID = "matched_max_expected_return_top10"
ALL_ELIGIBLE_ID = "strict_all_eligible_equal_weight"
V9_REFERENCE_ID = "v9_score_top10_reference"
BASE_PORTFOLIOS = (MAX_SHARPE_ID, MATCHED_ID, ALL_ELIGIBLE_ID, V9_REFERENCE_ID)
FRESH_SUFFIX = "_viewed_recent_fresh_cash"
EXACT10_SUFFIX = "_exact10_fresh_cash"


@contextmanager
def _exclusive_output_lock(output_root: Path):
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / ".v10_protocol.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"V10 output root is already locked: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    return v9._sha256(path)


def _payload_sha256(value: Any) -> str:
    return v9._payload_sha256(_jsonable(value))


def _as_bool(series: pd.Series) -> pd.Series:
    return v9._as_bool(series)


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["research_status_verdict"] != "NOT VALIDATED":
        raise ValueError("V10 retrospective research must remain NOT VALIDATED")
    if protocol["universe_is_point_in_time"] is not False:
        raise ValueError("V10 remains conditional on the ex-post fixed 114 universe")
    if protocol["historical_period_is_independent_oos"] is not False:
        raise ValueError("the historical period has already been viewed")
    eligibility = config["eligibility"]
    estimation = config["estimation"]
    selection = config["selection"]
    execution = config["execution"]
    expected = {
        "completed_quarters": (eligibility["completed_quarters"], 8),
        "minimum_active_quarters": (eligibility["minimum_active_quarters"], 3),
        "minimum_trade_entries": (eligibility["minimum_trade_entries"], 10),
        "all_history_available": (
            eligibility["require_all_quarters_history_available"],
            True,
        ),
        "known_by_window_start": (
            eligibility["require_first_known_by_window_start"],
            True,
        ),
        "cutoff_price_gate": (
            eligibility["require_finite_positive_prices_at_information_cutoff"],
            True,
        ),
        "missing_history": (
            eligibility["unavailable_or_missing_history_policy"],
            "exclude_fail_closed",
        ),
        "training_window": (
            estimation["training_window"],
            "same_last_8_completed_quarters_for_every_candidate",
        ),
        "daily_return": (estimation["daily_return_column"], "net_return_2bp"),
        "expected_return": (
            estimation["expected_return"],
            "expm1_annualized_one_standard_error_lower_bound_of_8_quarter_log_returns",
        ),
        "quarterly_ddof": (estimation["quarterly_log_return_std_ddof"], 1),
        "quarterly_annualization": (estimation["quarterly_log_annualization"], 4),
        "covariance": (estimation["covariance_estimator"], "ledoit_wolf"),
        "covariance_floor_rule": (
            estimation["covariance_eigenvalue_floor_rule"],
            "max(absolute_floor, mean_variance_times_relative_multiplier)",
        ),
        "daily_annualization": (estimation["daily_covariance_annualization"], 252),
        "minimum_daily_observations": (estimation["minimum_daily_observations"], 480),
        "expected_sessions": (
            estimation["expected_session_calendar"],
            "all_rows_in_frozen_daily_panel_for_the_same_8_quarters",
        ),
        "missing_daily": (estimation["missing_daily_return_policy"], "reject_no_imputation"),
        "objective": (
            selection["objective"],
            "maximum_estimated_equal_weight_portfolio_sharpe",
        ),
        "top_n": (selection["top_n"], 10),
        "max_pairs_per_ticker": (selection["maximum_pairs_per_ticker"], 1),
        "cardinality": (
            selection["cardinality_policy"],
            "largest_feasible_k_at_or_below_10_with_positive_expected_return_else_cash",
        ),
        "solver": (
            selection["solver"],
            "scipy_highs_rlt_strengthened_lifted_binary_milp_dinkelbach",
        ),
        "lifted_products": (selection["lifted_pair_products"], True),
        "rlt_equalities": (selection["rlt_cardinality_equalities"], True),
        "omit_incompatible_products": (
            selection["omit_ticker_incompatible_pair_products"],
            True,
        ),
        "fixed_lambda_objective": (
            selection["fixed_lambda_objective"],
            "maximize_scaled_squared_return_minus_lambda_times_variance",
        ),
        "presolve": (
            selection["solver_presolve_policy"],
            "enabled_for_all_cardinality_expected_return_and_dinkelbach_margin_milps",
        ),
        "unused_slots": (selection["unused_slots"], "cash"),
        "candidate_order": (selection["candidate_order"], "pair_lexicographic"),
        "matched_baseline": (
            selection["matched_baseline"],
            "maximum_sum_expected_return_same_candidates_cardinality_and_ticker_constraints",
        ),
        "fill": (
            execution["allocation_fill"],
            "rebalance_session_close_after_that_sessions_return",
        ),
        "decision_cutoff": (
            execution["decision_information_cutoff"],
            "session_before_rebalance_close",
        ),
        "first_pnl": (
            execution["first_new_pnl"],
            "next_session_after_allocation_fill",
        ),
        "accounting": (
            execution["accounting"],
            "pair_segregated_two_leg_security_ledger",
        ),
        "netting": (execution["cross_pair_trade_netting"], False),
        "incoming_sync": (
            execution["incoming_active_shadow_policy"],
            "cash_until_observable_post_close_target_is_flat",
        ),
        "terminal": (execution["terminal_liquidation"], True),
    }
    bad = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if bad:
        raise ValueError(f"V10 protocol/config mismatch: {bad}")
    if not math.isclose(
        float(selection["target_weight_per_slot"]) * int(selection["top_n"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("V10 requires ten fixed 10% pair sleeves")
    if float(estimation["risk_free_rate"]) != 0.0:
        raise ValueError("the frozen optimizer assumes zero cash/risk-free return")
    numeric_checks = {
        "absolute covariance floor": (
            float(estimation["covariance_eigenvalue_floor"]),
            1e-10,
        ),
        "relative covariance floor multiplier": (
            float(estimation["covariance_relative_eigenvalue_floor_multiplier"]),
            1e-8,
        ),
        "positive numerator epsilon": (
            float(estimation["positive_numerator_epsilon"]),
            1e-10,
        ),
        "MIP relative gap": (float(selection["mip_relative_gap"]), 0.0),
        "per-solve time limit": (
            float(selection["per_solve_time_limit_seconds"]),
            60.0,
        ),
        "Dinkelbach residual tolerance": (
            float(selection["dinkelbach_residual_tolerance"]),
            1e-9,
        ),
        "Dinkelbach ratio tolerance": (
            float(selection["dinkelbach_ratio_tolerance"]),
            1e-10,
        ),
        "Dinkelbach Sharpe certificate tolerance": (
            float(selection["dinkelbach_sharpe_certificate_tolerance"]),
            0.001,
        ),
        "solution tolerance": (float(selection["solution_tolerance"]), 1e-6),
    }
    bad_numeric = [
        name
        for name, (actual, wanted) in numeric_checks.items()
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-15)
    ]
    bad_integer = []
    if int(selection["dinkelbach_maximum_iterations"]) != 32:
        bad_integer.append("Dinkelbach maximum iterations")
    if bad_numeric or bad_integer:
        raise ValueError(
            f"V10 numeric optimizer protocol mismatch: {bad_numeric + bad_integer}"
        )
    return config


def _output_root(config: Mapping[str, Any]) -> Path:
    return PROJECT_ROOT / str(config["outputs"]["root"])


def _paths(config_path: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = config["inputs"]
    v9_root = PROJECT_ROOT / str(inputs["v9_root"])
    return {
        "config": config_path.resolve(),
        "runner": Path(__file__).resolve(),
        "optimizer_module": OPTIMIZER_MODULE.resolve(),
        "v9_runner": V9_RUNNER.resolve(),
        "v9_walkforward_module": V9_WALKFORWARD_MODULE.resolve(),
        "identity_evaluation_module": IDENTITY_MODULE.resolve(),
        "v9_freeze_manifest": (v9_root / str(inputs["v9_freeze_manifest"])).resolve(),
        "v9_reveal_manifest": (v9_root / str(inputs["v9_reveal_manifest"])).resolve(),
        "first_known_shadow_daily": (
            v9_root / str(inputs["first_known_shadow_daily"])
        ).resolve(),
        "quarter_history": (v9_root / str(inputs["quarter_history"])).resolve(),
        "cross_pair_rankings": (
            v9_root / str(inputs["cross_pair_rankings"])
        ).resolve(),
        "v9_allocation_events": (
            v9_root / str(inputs["v9_allocation_events"])
        ).resolve(),
        "v9_portfolio_metrics": (
            v9_root / str(inputs["v9_portfolio_metrics"])
        ).resolve(),
    }


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen input/source: " + ", ".join(missing))
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in sorted(paths.items())
    }


def _runtime_manifest() -> dict[str, Any]:
    runtime = dict(v9.v8._runtime_manifest())
    packages = dict(runtime.get("packages", {}))
    packages["scikit-learn"] = importlib.metadata.version("scikit-learn")
    runtime["packages"] = packages
    runtime["optimizer"] = "scipy.optimize.milp/HiGHS"
    return runtime


def _verify_v9_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    freeze_path = paths["v9_freeze_manifest"]
    reveal_path = paths["v9_reveal_manifest"]
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
    frozen_without_claim = dict(frozen)
    claimed_payload_hash = frozen_without_claim.pop("freeze_payload_sha256", None)
    if claimed_payload_hash != _payload_sha256(frozen_without_claim):
        raise RuntimeError("V9 freeze payload checksum failed")
    if frozen.get("performance_computed") is not False:
        raise RuntimeError("V9 protocol was not frozen before performance computation")
    if frozen.get("research_status_verdict") != "NOT VALIDATED":
        raise RuntimeError("V9 research verdict changed")
    if reveal.get("research_status_verdict") != "NOT VALIDATED":
        raise RuntimeError("V9 reveal verdict changed")
    if reveal.get("source_hashes") != frozen.get("source_hashes"):
        raise RuntimeError("V9 reveal source hashes differ from its freeze")
    freeze_hash = _sha256(freeze_path)
    if reveal.get("freeze_manifest", {}).get("sha256") != freeze_hash:
        raise RuntimeError("V9 reveal does not bind to the current V9 freeze")
    checked_v9_sources = 0
    for record in frozen.get("source_hashes", {}).values():
        source = Path(str(record["path"]))
        if not source.is_file() or _sha256(source) != str(record["sha256"]):
            raise RuntimeError(f"V9 frozen source hash mismatch: {source}")
        checked_v9_sources += 1
    checked_v9_outputs = 0
    for name, record in reveal.get("output_files", {}).items():
        artifact = Path(str(record["path"]))
        if not artifact.is_file() or _sha256(artifact) != str(record["sha256"]):
            raise RuntimeError(f"V9 output hash mismatch: {name}")
        checked_v9_outputs += 1
    mapping = {
        "first_known_shadow_daily": "first_known_shadow_daily",
        "quarter_history": "quarter_history",
        "cross_pair_rankings": "cross_pair_rankings",
        "v9_allocation_events": "allocation_events",
        "v9_portfolio_metrics": "portfolio_metrics",
    }
    for path_key, output_key in mapping.items():
        record = reveal["output_files"].get(output_key)
        if record is None:
            raise RuntimeError(f"V9 reveal is missing output record {output_key}")
        if Path(str(record["path"])).resolve() != paths[path_key].resolve():
            raise RuntimeError(f"V9 configured input path differs for {path_key}")
        if _sha256(paths[path_key]) != str(record["sha256"]):
            raise RuntimeError(f"V9 configured input hash differs for {path_key}")
    return {
        "v9_freeze_sha256": freeze_hash,
        "v9_reveal_sha256": _sha256(reveal_path),
        "v9_source_hashes_verified": checked_v9_sources,
        "v9_output_hashes_verified": checked_v9_outputs,
        "configured_v9_inputs_verified": len(mapping),
    }


def _freeze_unlocked(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _output_root(config)
    data_root = output_root / "data"
    freeze_path = data_root / FREEZE_NAME
    reveal_path = data_root / REVEAL_NAME
    if reveal_path.exists() or freeze_path.exists():
        raise RuntimeError("V10 output is immutable; use a new output root/version")
    paths = _paths(config_path, config)
    v9_verification = _verify_v9_inputs(paths)
    runtime = _runtime_manifest()
    payload = {
        "schema": "conditional_fixed_114_joint_max_sharpe_freeze_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "performance_computed": False,
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "config": config,
        "source_hashes": _source_hashes(paths),
        "v9_input_verification": v9_verification,
        "runtime_manifest": runtime,
        "runtime_manifest_sha256": _payload_sha256(runtime),
    }
    payload["freeze_payload_sha256"] = _payload_sha256(payload)
    data_root.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return {
        "phase": "freeze",
        "freeze_manifest": str(freeze_path.resolve()),
        "freeze_sha256": _sha256(freeze_path),
        "performance_computed": False,
    }


def _verify_freeze(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    paths = _paths(config_path, config)
    freeze_path = _output_root(config) / "data" / FREEZE_NAME
    if not freeze_path.is_file():
        raise RuntimeError("reveal requires a prior V10 protocol freeze")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    claimed = frozen.get("freeze_payload_sha256")
    without_claim = dict(frozen)
    without_claim.pop("freeze_payload_sha256", None)
    if claimed != _payload_sha256(without_claim):
        raise RuntimeError("V10 freeze payload checksum failed")
    current = _source_hashes(paths)
    if frozen.get("source_hashes") != current:
        changed = sorted(
            key
            for key in set(frozen.get("source_hashes", {})) | set(current)
            if frozen.get("source_hashes", {}).get(key) != current.get(key)
        )
        raise RuntimeError(f"V10 freeze source mismatch: {changed}")
    runtime = _runtime_manifest()
    if _payload_sha256(runtime) != frozen.get("runtime_manifest_sha256"):
        raise RuntimeError("runtime differs from the V10 freeze")
    if frozen.get("config") != config:
        raise RuntimeError("loaded V10 config differs from its freeze")
    v9_verification = _verify_v9_inputs(paths)
    if frozen.get("v9_input_verification") != v9_verification:
        raise RuntimeError("V9 input verification differs from the V10 freeze")
    return freeze_path, frozen, v9_verification


def _load_inputs(paths: Mapping[str, Path]) -> tuple[pd.DataFrame, ...]:
    daily = pd.read_csv(paths["first_known_shadow_daily"])
    history = pd.read_csv(paths["quarter_history"])
    rankings = pd.read_csv(paths["cross_pair_rankings"])
    v9_events = pd.read_csv(paths["v9_allocation_events"])
    v9_metrics = pd.read_csv(paths["v9_portfolio_metrics"])
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    for column in (
        "rebalance_date",
        "outer_end_exclusive",
        "quarter_last_session",
        "eligible_since_date",
    ):
        history[column] = pd.to_datetime(history[column], errors="coerce")
    for column in ("ranking_effective_date", "information_cutoff", "allocation_fill_date"):
        rankings[column] = pd.to_datetime(rankings[column], errors="coerce")
    for column in ("decision_date", "fill_date"):
        v9_events[column] = pd.to_datetime(v9_events[column], errors="coerce")
    if len(daily) != 114 * 2513 or daily["pair"].nunique() != 114:
        raise RuntimeError("unexpected V9 first-known daily dimensions")
    if daily["date"].nunique() != 2513 or daily.duplicated(["pair", "date"]).any():
        raise RuntimeError("V9 daily panel is not a unique balanced 114 x 2513 panel")
    if len(history) != 114 * 41 or history.duplicated(["pair", "quarter_number"]).any():
        raise RuntimeError("unexpected V9 quarter-history dimensions")
    if len(v9_events) != 41 or v9_events["fill_date"].duplicated().any():
        raise RuntimeError("unexpected V9 allocation schedule")
    if daily["date"].isna().any() or history[
        ["rebalance_date", "outer_end_exclusive", "quarter_last_session", "eligible_since_date"]
    ].isna().any().any():
        raise RuntimeError("V9 inputs contain invalid dates")
    if not v9_events["decision_date"].lt(v9_events["fill_date"]).all():
        raise RuntimeError("V9 decision/fill chronology failed")
    required_daily_numeric = [
        "quarter_number",
        "net_return_2bp",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
    ]
    for column in required_daily_numeric:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if not np.isfinite(daily[required_daily_numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("V9 daily panel contains nonfinite required values")
    for column in ("price_a", "price_b"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    valid_prices = (
        np.isfinite(daily["price_a"].to_numpy(dtype=float))
        & np.isfinite(daily["price_b"].to_numpy(dtype=float))
        & daily["price_a"].gt(0.0).to_numpy()
        & daily["price_b"].gt(0.0).to_numpy()
    )
    economically_active = daily[
        ["weight_a", "weight_b", "target_weight_a", "target_weight_b"]
    ].abs().sum(axis=1).gt(1e-12).to_numpy()
    if bool((~valid_prices & economically_active).any()):
        raise RuntimeError("an active V9 shadow row has an invalid price")
    for column in ("quarter_number", "quarter_return_2bp", "trade_entries"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    if not np.isfinite(
        history[["quarter_number", "quarter_return_2bp", "trade_entries"]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("V9 quarter history contains nonfinite required values")
    if (history["quarter_return_2bp"] <= -1.0).any():
        raise RuntimeError("V9 quarter history contains a return at or below -100%")
    metadata = daily[["pair", "ticker_a", "ticker_b"]].drop_duplicates()
    if len(metadata) != 114 or metadata["pair"].duplicated().any():
        raise RuntimeError("pair/ticker metadata is not one-to-one")
    return daily, history, rankings, v9_events, v9_metrics, metadata


def _optimizer_spec(config: Mapping[str, Any]) -> MaxSharpeSpec:
    eligibility = config["eligibility"]
    estimation = config["estimation"]
    selection = config["selection"]
    return MaxSharpeSpec(
        completed_quarters=int(eligibility["completed_quarters"]),
        minimum_active_quarters=int(eligibility["minimum_active_quarters"]),
        minimum_trade_entries=int(eligibility["minimum_trade_entries"]),
        top_n=int(selection["top_n"]),
        annualization=int(estimation["daily_covariance_annualization"]),
        minimum_daily_observations=int(estimation["minimum_daily_observations"]),
        covariance_eigenvalue_floor=float(estimation["covariance_eigenvalue_floor"]),
        covariance_relative_eigenvalue_floor=float(
            estimation["covariance_relative_eigenvalue_floor_multiplier"]
        ),
        positive_numerator_epsilon=float(estimation["positive_numerator_epsilon"]),
        dinkelbach_maximum_iterations=int(selection["dinkelbach_maximum_iterations"]),
        dinkelbach_residual_tolerance=float(selection["dinkelbach_residual_tolerance"]),
        dinkelbach_ratio_tolerance=float(selection["dinkelbach_ratio_tolerance"]),
        dinkelbach_sharpe_certificate_tolerance=float(
            selection["dinkelbach_sharpe_certificate_tolerance"]
        ),
        solution_tolerance=float(selection["solution_tolerance"]),
        mip_relative_gap=float(selection["mip_relative_gap"]),
        per_solve_time_limit_seconds=float(selection["per_solve_time_limit_seconds"]),
    )


def _pipe(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    return [] if not text else [item for item in text.split("|") if item]


def _event_problem(
    event: Mapping[str, Any],
    *,
    daily: pd.DataFrame,
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    specification: MaxSharpeSpec,
    target_weight: float,
    v9_selected: tuple[str, ...],
    expected_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    fill = pd.Timestamp(event["fill_date"])
    decision = pd.Timestamp(event["decision_date"])
    cutoff_rows = daily.loc[daily["date"].eq(decision), ["pair", "price_a", "price_b"]]
    if cutoff_rows["pair"].duplicated().any():
        raise RuntimeError(f"duplicate cutoff prices at {decision.date()}")
    valid_mask = (
        np.isfinite(cutoff_rows["price_a"].to_numpy(dtype=float))
        & np.isfinite(cutoff_rows["price_b"].to_numpy(dtype=float))
        & cutoff_rows["price_a"].gt(0.0).to_numpy()
        & cutoff_rows["price_b"].gt(0.0).to_numpy()
    )
    valid_pairs = set(cutoff_rows.loc[valid_mask, "pair"].astype(str))
    filtered_daily = daily.loc[daily["pair"].astype(str).isin(valid_pairs)]
    filtered_history = history.loc[history["pair"].astype(str).isin(valid_pairs)]
    filtered_metadata = metadata.loc[metadata["pair"].astype(str).isin(valid_pairs)]
    result = optimize_walkforward_books(
        filtered_daily,
        filtered_history,
        filtered_metadata,
        fill,
        decision,
        specification,
        expected_sessions=expected_sessions,
    )
    panel = result.training
    max_result = result.max_sharpe
    matched_result = result.matched_score
    candidates = panel.candidate_stats.copy()
    max_pairs = tuple(max_result.selected_pairs)
    matched_pairs = tuple(matched_result.selected_pairs)
    dinkelbach_statuses = [
        status
        for status in max_result.solver_statuses
        if status.get("phase") == "dinkelbach_residual"
    ]
    final_dinkelbach_status = dinkelbach_statuses[-1] if dinkelbach_statuses else {}
    if max_result.selected_count != matched_result.selected_count:
        raise RuntimeError(f"matched portfolios have different cardinality at {fill.date()}")
    if set(max_pairs) - set(candidates.get("pair", pd.Series(dtype=str)).astype(str)):
        raise RuntimeError("max-Sharpe selection is outside the strict candidate set")
    if set(matched_pairs) - set(candidates.get("pair", pd.Series(dtype=str)).astype(str)):
        raise RuntimeError("matched selection is outside the strict candidate set")
    for selected in (max_pairs, matched_pairs):
        chosen = metadata.loc[metadata["pair"].isin(selected)]
        names = chosen[["ticker_a", "ticker_b"]].astype(str).to_numpy().ravel()
        if len(names) != len(set(names)):
            raise RuntimeError(f"selected book repeats a ticker at {fill.date()}")
    candidate_pairs = tuple(candidates.get("pair", pd.Series(dtype=str)).astype(str))
    max_targets = {pair: target_weight for pair in max_pairs}
    matched_targets = {pair: target_weight for pair in matched_pairs}
    eligible_targets = (
        {pair: 1.0 / len(candidate_pairs) for pair in candidate_pairs}
        if candidate_pairs
        else {}
    )
    structural_k = max(
        int(max_result.maximum_feasible_k), int(matched_result.maximum_feasible_k)
    )
    event_row = {
        "quarter_number": int(event["quarter_number"]),
        "decision_date": decision,
        "fill_date": fill,
        "training_status": panel.status,
        "training_quarter_numbers": "|".join(str(item) for item in panel.quarter_numbers),
        "training_start": panel.training_start,
        "training_end": panel.training_end,
        "training_daily_observations": int(len(panel.daily_returns)),
        "cutoff_valid_price_pair_count": int(len(valid_pairs)),
        "cutoff_invalid_price_pair_count": int(len(metadata) - len(valid_pairs)),
        "strict_eligible_pair_count": int(len(candidates)),
        "structural_maximum_feasible_k": structural_k,
        "selected_k_after_positive_mu_rule": int(max_result.selected_count),
        "max_sharpe_status": max_result.status,
        "max_sharpe_selected_pairs": "|".join(max_pairs),
        "max_sharpe_expected_annual_return_pair_sum_unscaled": max_result.expected_annual_return_sum,
        "max_sharpe_annual_covariance_pair_sum_unscaled": max_result.portfolio_variance_sum,
        "max_sharpe_weighted_expected_annual_return": (
            None
            if max_result.expected_annual_return_sum is None
            else target_weight * max_result.expected_annual_return_sum
        ),
        "max_sharpe_weighted_annual_variance": (
            None
            if max_result.portfolio_variance_sum is None
            else target_weight * target_weight * max_result.portfolio_variance_sum
        ),
        "max_sharpe_estimated_sharpe": max_result.estimated_sharpe,
        "max_sharpe_dinkelbach_iterations": max_result.dinkelbach_iterations,
        "max_sharpe_final_maximum_residual": max_result.final_maximum_residual,
        "max_sharpe_final_maximum_scaled_residual": max_result.final_maximum_scaled_residual,
        "max_sharpe_optimal_ratio_squared_lower": max_result.optimal_ratio_squared_lower,
        "max_sharpe_optimal_ratio_squared_upper": max_result.optimal_ratio_squared_upper,
        "max_sharpe_certificate_lower": max_result.sharpe_certificate_lower,
        "max_sharpe_certificate_upper": max_result.sharpe_certificate_upper,
        "max_sharpe_certificate_width": max_result.sharpe_certificate_width,
        "max_sharpe_variance_lower_bound": max_result.variance_lower_bound,
        "max_sharpe_final_mip_gap": final_dinkelbach_status.get("mip_gap"),
        "max_sharpe_final_mip_dual_bound": final_dinkelbach_status.get("mip_dual_bound"),
        "max_sharpe_final_dual_unscaled_residual_upper": final_dinkelbach_status.get(
            "dual_unscaled_lifted_margin_upper"
        ),
        "max_sharpe_final_dual_unscaled_residual_upper_with_allowance": (
            final_dinkelbach_status.get("dual_unscaled_residual_upper_with_allowance")
        ),
        "max_sharpe_final_absolute_scaled_primal_dual_gap": (
            final_dinkelbach_status.get("absolute_scaled_primal_dual_gap")
        ),
        "max_sharpe_final_numerical_residual_allowance": final_dinkelbach_status.get(
            "numerical_residual_allowance"
        ),
        "max_sharpe_final_solver_elapsed_seconds": final_dinkelbach_status.get(
            "elapsed_seconds"
        ),
        "matched_status": matched_result.status,
        "matched_selected_pairs": "|".join(matched_pairs),
        "matched_expected_annual_return_pair_sum_unscaled": matched_result.expected_annual_return_sum,
        "matched_annual_covariance_pair_sum_unscaled": matched_result.portfolio_variance_sum,
        "matched_weighted_expected_annual_return": (
            None
            if matched_result.expected_annual_return_sum is None
            else target_weight * matched_result.expected_annual_return_sum
        ),
        "matched_weighted_annual_variance": (
            None
            if matched_result.portfolio_variance_sum is None
            else target_weight * target_weight * matched_result.portfolio_variance_sum
        ),
        "matched_estimated_sharpe": matched_result.estimated_sharpe,
        "v9_selected_pair_count": len(v9_selected),
        "v9_selected_pairs": "|".join(v9_selected),
        "v9_selected_not_strict_count": len(set(v9_selected) - set(candidate_pairs)),
        "v9_selected_not_strict_pairs": "|".join(sorted(set(v9_selected) - set(candidate_pairs))),
        "covariance_shrinkage": panel.covariance_shrinkage,
        "covariance_floor_applied": panel.covariance_floor_applied,
        "max_sharpe_target_invested_weight": float(sum(max_targets.values())),
        "max_sharpe_target_cash_weight": float(1.0 - sum(max_targets.values())),
        "matched_target_invested_weight": float(sum(matched_targets.values())),
        "matched_target_cash_weight": float(1.0 - sum(matched_targets.values())),
    }
    candidate_rows: list[dict[str, Any]] = []
    if len(candidates):
        covariance = panel.annual_covariance
        for raw in candidates.to_dict("records"):
            pair = str(raw["pair"])
            candidate_rows.append(
                {
                    "quarter_number": int(event["quarter_number"]),
                    "decision_date": decision,
                    "fill_date": fill,
                    **raw,
                    "annual_variance": float(covariance.at[pair, pair]),
                    "selected_max_sharpe": pair in max_pairs,
                    "selected_matched_expected_return": pair in matched_pairs,
                    "selected_v9_reference": pair in v9_selected,
                }
            )
    covariance_rows: list[dict[str, Any]] = []
    if len(panel.annual_covariance):
        covariance = panel.annual_covariance
        pairs = list(covariance.index.astype(str))
        for left_index, left in enumerate(pairs):
            for right in pairs[left_index:]:
                covariance_rows.append(
                    {
                        "quarter_number": int(event["quarter_number"]),
                        "decision_date": decision,
                        "fill_date": fill,
                        "pair_i": left,
                        "pair_j": right,
                        "annual_covariance": float(covariance.at[left, right]),
                    }
                )
    solver_rows: list[dict[str, Any]] = []
    for method, selection in ((MAX_SHARPE_ID, max_result), (MATCHED_ID, matched_result)):
        for sequence, status in enumerate(selection.solver_statuses):
            solver_rows.append(
                {
                    "quarter_number": int(event["quarter_number"]),
                    "decision_date": decision.isoformat(),
                    "fill_date": fill.isoformat(),
                    "portfolio_id": method,
                    "selection_status": selection.status,
                    "sequence": sequence,
                    **_jsonable(status),
                }
            )
    simulator_rows = [
        {
            "portfolio_id": MAX_SHARPE_ID,
            "decision_date": decision,
            "fill_date": fill,
            "target_weights": max_targets,
        },
        {
            "portfolio_id": MATCHED_ID,
            "decision_date": decision,
            "fill_date": fill,
            "target_weights": matched_targets,
        },
        {
            "portfolio_id": ALL_ELIGIBLE_ID,
            "decision_date": decision,
            "fill_date": fill,
            "target_weights": eligible_targets,
        },
        {
            "portfolio_id": V9_REFERENCE_ID,
            "decision_date": decision,
            "fill_date": fill,
            "target_weights": {pair: target_weight for pair in v9_selected},
        },
    ]
    return {
        "event": event_row,
        "candidates": candidate_rows,
        "covariance": covariance_rows,
        "solver": solver_rows,
        "simulator": simulator_rows,
    }


def _build_selection_artifacts(
    daily: pd.DataFrame,
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    v9_events: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    specification = _optimizer_spec(config)
    target_weight = float(config["selection"]["target_weight_per_slot"])
    expected_sessions = pd.DatetimeIndex(
        daily["date"].drop_duplicates().sort_values(kind="mergesort")
    )
    tasks = []
    for row in v9_events.sort_values("fill_date", kind="mergesort").to_dict("records"):
        selected = tuple(_pipe(row.get("selected_pairs")))
        tasks.append((row, selected))

    def run(task: tuple[dict[str, Any], tuple[str, ...]]) -> dict[str, Any]:
        row, selected = task
        return _event_problem(
            row,
            daily=daily,
            history=history,
            metadata=metadata,
            specification=specification,
            target_weight=target_weight,
            v9_selected=selected,
            expected_sessions=expected_sessions,
        )

    if workers <= 1:
        results = [run(task) for task in tasks]
    else:
        results_by_quarter: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run, task): int(task[0]["quarter_number"]) for task in tasks}
            for future in as_completed(futures):
                results_by_quarter[futures[future]] = future.result()
        results = [results_by_quarter[int(task[0]["quarter_number"])] for task in tasks]
    events = pd.DataFrame([result["event"] for result in results]).sort_values("fill_date")
    candidates = pd.DataFrame(
        [row for result in results for row in result["candidates"]]
    )
    covariance = pd.DataFrame(
        [row for result in results for row in result["covariance"]]
    )
    solver = [row for result in results for row in result["solver"]]
    simulator = pd.DataFrame(
        [row for result in results for row in result["simulator"]]
    )
    return events, candidates, covariance, solver, simulator


def _metric_row(
    daily: pd.DataFrame,
    *,
    portfolio_id: str,
    segment: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    return v9._metric_row(
        daily,
        portfolio_id=portfolio_id,
        segment=segment,
        start=start,
        end_exclusive=end_exclusive,
    )


def _stable_exact_ten_start(events: pd.DataFrame) -> pd.Timestamp | None:
    ordered = events.sort_values("fill_date", kind="mergesort").reset_index(drop=True)
    exact = ordered["selected_k_after_positive_mu_rule"].eq(10)
    for index in range(len(ordered)):
        if bool(exact.iloc[index:].all()):
            return pd.Timestamp(ordered.iloc[index]["fill_date"])
    return None


def _simulate_portfolios(
    daily: pd.DataFrame,
    simulator_events: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    exact_start: pd.Timestamp | None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    execution = config["execution"]
    holdout = pd.Timestamp(config["evaluation"]["retrospective_review_start"])
    frames: list[pd.DataFrame] = []
    ledger_frames: list[pd.DataFrame] = []
    portfolio_map: dict[str, pd.DataFrame] = {}
    for portfolio_id in BASE_PORTFOLIOS:
        event_group = simulator_events.loc[
            simulator_events["portfolio_id"].eq(portfolio_id),
            ["decision_date", "fill_date", "target_weights"],
        ]
        result, ledger = simulate_security_level_portfolio(
            daily,
            event_group,
            portfolio_id=portfolio_id,
            transaction_cost_bps=float(execution["transaction_cost_bps"]),
            annual_short_borrow_bps=float(execution["annual_short_borrow_bps"]),
            terminal_liquidation=bool(execution["terminal_liquidation"]),
            return_detail=True,
        )
        portfolio_map[portfolio_id] = result
        frames.append(result.reset_index())
        ledger_frames.append(ledger)
    holdout_signals = daily.loc[daily["date"].ge(holdout)].copy()
    for portfolio_id in BASE_PORTFOLIOS:
        fresh_id = portfolio_id + FRESH_SUFFIX
        event_group = simulator_events.loc[
            simulator_events["portfolio_id"].eq(portfolio_id)
            & simulator_events["fill_date"].ge(holdout),
            ["decision_date", "fill_date", "target_weights"],
        ]
        result, ledger = simulate_security_level_portfolio(
            holdout_signals,
            event_group,
            portfolio_id=fresh_id,
            transaction_cost_bps=float(execution["transaction_cost_bps"]),
            annual_short_borrow_bps=float(execution["annual_short_borrow_bps"]),
            terminal_liquidation=bool(execution["terminal_liquidation"]),
            return_detail=True,
        )
        portfolio_map[fresh_id] = result
        frames.append(result.reset_index())
        ledger_frames.append(ledger)
    if exact_start is not None:
        exact_signals = daily.loc[daily["date"].ge(exact_start)].copy()
        for portfolio_id in BASE_PORTFOLIOS:
            exact_id = portfolio_id + EXACT10_SUFFIX
            event_group = simulator_events.loc[
                simulator_events["portfolio_id"].eq(portfolio_id)
                & simulator_events["fill_date"].ge(exact_start),
                ["decision_date", "fill_date", "target_weights"],
            ]
            result, ledger = simulate_security_level_portfolio(
                exact_signals,
                event_group,
                portfolio_id=exact_id,
                transaction_cost_bps=float(execution["transaction_cost_bps"]),
                annual_short_borrow_bps=float(execution["annual_short_borrow_bps"]),
                terminal_liquidation=bool(execution["terminal_liquidation"]),
                return_detail=True,
            )
            portfolio_map[exact_id] = result
            frames.append(result.reset_index())
            ledger_frames.append(ledger)
    return portfolio_map, pd.concat(frames, ignore_index=True), pd.concat(ledger_frames, ignore_index=True)


def _build_metrics(
    portfolio_map: Mapping[str, pd.DataFrame],
    events: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp | None]:
    evaluation = config["evaluation"]
    study_start = pd.Timestamp(evaluation["study_start_inclusive"])
    study_end = pd.Timestamp(evaluation["study_end_exclusive"])
    holdout = pd.Timestamp(evaluation["retrospective_review_start"])
    selected_events = events.loc[events["selected_k_after_positive_mu_rule"].gt(0)]
    if selected_events.empty:
        raise RuntimeError("V10 max-Sharpe rule never selected a pair")
    first_fill = pd.Timestamp(selected_events["fill_date"].min())
    exact_start = _stable_exact_ten_start(events)
    rows: list[dict[str, Any]] = []
    annual_parts: list[pd.DataFrame] = []
    for portfolio_id, result in portfolio_map.items():
        annual_parts.append(v9._annual_returns(result, portfolio_id))
        if portfolio_id.endswith(FRESH_SUFFIX):
            segments = {"viewed_recent_history_fresh_cash": (holdout, study_end)}
        elif portfolio_id.endswith(EXACT10_SUFFIX):
            if exact_start is None:
                raise RuntimeError("exact-ten path exists without an exact-ten start")
            segments = {"stable_exact_10_common_interval_fresh_cash": (exact_start, study_end)}
        else:
            segments: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
                "continuous_full_calendar_with_initial_cash": (study_start, study_end),
                "prequential_from_first_v10_selection": (first_fill, study_end),
                "development_before_viewed_recent_history": (first_fill, holdout),
            }
            if exact_start is not None:
                segments["continuous_carried_state_from_stable_exact_10_start"] = (
                    exact_start,
                    study_end,
                )
        for segment, (start, end) in segments.items():
            if start < end:
                rows.append(
                    _metric_row(
                        result,
                        portfolio_id=portfolio_id,
                        segment=segment,
                        start=start,
                        end_exclusive=end,
                    )
                )
    return pd.DataFrame(rows), pd.concat(annual_parts, ignore_index=True), first_fill, exact_start


def _bootstrap_comparisons(
    portfolio_map: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    first_fill: pd.Timestamp,
    exact_start: pd.Timestamp | None,
) -> pd.DataFrame:
    evaluation = config["evaluation"]
    study_end = pd.Timestamp(evaluation["study_end_exclusive"])
    holdout = pd.Timestamp(evaluation["retrospective_review_start"])
    segment_specs: list[tuple[str, pd.Timestamp, pd.Timestamp, str]] = [
        ("prequential_from_first_v10_selection", first_fill, study_end, ""),
        ("viewed_recent_history_fresh_cash", holdout, study_end, FRESH_SUFFIX),
    ]
    if exact_start is not None:
        segment_specs.append(
            ("stable_exact_10_common_interval_fresh_cash", exact_start, study_end, EXACT10_SUFFIX)
        )
    rows: list[dict[str, Any]] = []
    comparisons = (
        (MATCHED_ID, "max_sharpe_minus_matched_expected_return"),
        (V9_REFERENCE_ID, "max_sharpe_minus_v9_score_reference"),
    )
    seed = int(evaluation["bootstrap_seed"])
    for segment_index, (segment, start, end, suffix) in enumerate(segment_specs):
        top_id = MAX_SHARPE_ID + suffix
        for comparison_index, (benchmark_base, label) in enumerate(comparisons):
            benchmark_id = benchmark_base + suffix
            top = portfolio_map[top_id].loc[
                (portfolio_map[top_id].index >= start) & (portfolio_map[top_id].index < end),
                "portfolio_return",
            ]
            benchmark = portfolio_map[benchmark_id].loc[
                (portfolio_map[benchmark_id].index >= start)
                & (portfolio_map[benchmark_id].index < end),
                "portfolio_return",
            ]
            block = int(evaluation["bootstrap_block_sessions"])
            if len(top) < block:
                record = {
                    "segment": segment,
                    "sessions": int(len(top)),
                    "annualized_log_return_difference": float(
                        (np.log1p(top.to_numpy()) - np.log1p(benchmark.to_numpy())).mean()
                        * 252.0
                    ),
                    "bootstrap_median": np.nan,
                    "ci_95_low": np.nan,
                    "ci_95_high": np.nan,
                    "bootstrap_probability_greater_than_zero": np.nan,
                    "block_sessions": block,
                    "draws": 0,
                    "seed": seed + segment_index * 10 + comparison_index,
                    "interpretation": "not computed: segment shorter than one frozen block",
                }
            else:
                record = v9._paired_block_bootstrap(
                    top,
                    benchmark,
                    block=block,
                    draws=int(evaluation["bootstrap_draws"]),
                    seed=seed + segment_index * 10 + comparison_index,
                    label=segment,
                )
            record["comparison"] = label
            record["portfolio_id"] = top_id
            record["benchmark_portfolio_id"] = benchmark_id
            rows.append(record)
    return pd.DataFrame(rows)


def _stress_metrics(
    daily: pd.DataFrame,
    simulator_events: pd.DataFrame,
    portfolio_map: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    first_fill: pd.Timestamp,
    exact_start: pd.Timestamp | None,
) -> pd.DataFrame:
    execution = config["execution"]
    evaluation = config["evaluation"]
    study_end = pd.Timestamp(evaluation["study_end_exclusive"])
    holdout = pd.Timestamp(evaluation["retrospective_review_start"])
    primary = (
        float(execution["transaction_cost_bps"]),
        float(execution["annual_short_borrow_bps"]),
    )
    scenarios = [
        primary,
        *[(float(cost), primary[1]) for cost in execution["stress_transaction_cost_bps"]],
        *[(primary[0], float(borrow)) for borrow in execution["stress_annual_short_borrow_bps"]],
    ]
    scenarios = list(dict.fromkeys(scenarios))
    rows: list[dict[str, Any]] = []
    holdout_signals = daily.loc[daily["date"].ge(holdout)].copy()
    for portfolio_id in (MAX_SHARPE_ID,):
        event_group = simulator_events.loc[
            simulator_events["portfolio_id"].eq(portfolio_id),
            ["decision_date", "fill_date", "target_weights"],
        ]
        for transaction_bps, borrow_bps in scenarios:
            is_primary = (transaction_bps, borrow_bps) == primary
            if is_primary:
                continuous = portfolio_map[portfolio_id]
                fresh = portfolio_map[portfolio_id + FRESH_SUFFIX]
                exact_fresh = (
                    portfolio_map[portfolio_id + EXACT10_SUFFIX]
                    if exact_start is not None
                    else None
                )
            else:
                continuous = simulate_security_level_portfolio(
                    daily,
                    event_group,
                    portfolio_id="stress",
                    transaction_cost_bps=transaction_bps,
                    annual_short_borrow_bps=borrow_bps,
                    terminal_liquidation=True,
                )
                fresh = simulate_security_level_portfolio(
                    holdout_signals,
                    event_group.loc[event_group["fill_date"].ge(holdout)],
                    portfolio_id="stress_fresh",
                    transaction_cost_bps=transaction_bps,
                    annual_short_borrow_bps=borrow_bps,
                    terminal_liquidation=True,
                )
                exact_fresh = None
                if exact_start is not None:
                    exact_fresh = simulate_security_level_portfolio(
                        daily.loc[daily["date"].ge(exact_start)].copy(),
                        event_group.loc[event_group["fill_date"].ge(exact_start)],
                        portfolio_id="stress_exact10_fresh",
                        transaction_cost_bps=transaction_bps,
                        annual_short_borrow_bps=borrow_bps,
                        terminal_liquidation=True,
                    )
            segments = [
                ("prequential_from_first_v10_selection", continuous, first_fill),
                ("viewed_recent_history_fresh_cash", fresh, holdout),
            ]
            if exact_start is not None:
                if exact_fresh is None:
                    raise RuntimeError("missing exact-ten fresh-cash stress path")
                segments.append(
                    ("stable_exact_10_common_interval_fresh_cash", exact_fresh, exact_start)
                )
            for segment, frame, start in segments:
                metric = _metric_row(
                    frame,
                    portfolio_id=portfolio_id,
                    segment=segment,
                    start=start,
                    end_exclusive=study_end,
                )
                rows.append(
                    {
                        "portfolio_id": portfolio_id,
                        "transaction_cost_bps": transaction_bps,
                        "annual_short_borrow_bps": borrow_bps,
                        "primary_scenario": is_primary,
                        "selections_reoptimized": False,
                        "segment": segment,
                        "net_total_return": metric["net_total_return"],
                        "calendar_cagr": metric["calendar_cagr"],
                        "sharpe": metric["sharpe"],
                        "max_drawdown": metric["max_drawdown"],
                        "annualized_leg_turnover": metric["annualized_leg_turnover"],
                    }
                )
    return pd.DataFrame(rows)


def _selection_churn(events: pd.DataFrame) -> pd.DataFrame:
    columns = {
        MAX_SHARPE_ID: "max_sharpe_selected_pairs",
        MATCHED_ID: "matched_selected_pairs",
        V9_REFERENCE_ID: "v9_selected_pairs",
    }
    rows: list[dict[str, Any]] = []
    for portfolio_id, column in columns.items():
        previous: set[str] | None = None
        for raw in events.sort_values("fill_date", kind="mergesort").to_dict("records"):
            current = set(_pipe(raw.get(column)))
            union = set() if previous is None else previous | current
            intersection = set() if previous is None else previous & current
            added = current if previous is None else current - previous
            removed = set() if previous is None else previous - current
            rows.append(
                {
                    "portfolio_id": portfolio_id,
                    "decision_date": raw["decision_date"],
                    "fill_date": raw["fill_date"],
                    "selected_pair_count": len(current),
                    "added_pair_count": len(added),
                    "removed_pair_count": len(removed),
                    "retained_pair_count": len(intersection),
                    "jaccard_vs_prior": (
                        np.nan if previous is None else (len(intersection) / len(union) if union else 1.0)
                    ),
                    "added_pairs": "|".join(sorted(added)),
                    "removed_pairs": "|".join(sorted(removed)),
                }
            )
            previous = current
    return pd.DataFrame(rows)


def _predicted_vs_realized(
    events: pd.DataFrame,
    portfolio_map: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    study_end = pd.Timestamp(config["evaluation"]["study_end_exclusive"])
    ordered = events.sort_values("fill_date", kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    definitions = {
        MAX_SHARPE_ID: ("max_sharpe_estimated_sharpe", "max_sharpe_selected_pairs"),
        MATCHED_ID: ("matched_estimated_sharpe", "matched_selected_pairs"),
    }
    for index, event in ordered.iterrows():
        start = pd.Timestamp(event["fill_date"])
        boundary = (
            pd.Timestamp(ordered.iloc[index + 1]["fill_date"])
            if index + 1 < len(ordered)
            else study_end
        )
        has_next_fill = index + 1 < len(ordered)
        for portfolio_id, (predicted_column, selected_column) in definitions.items():
            selected_count = len(_pipe(event[selected_column]))
            if selected_count == 0:
                continue
            return_index = portfolio_map[portfolio_id].index
            boundary_mask = return_index <= boundary if has_next_fill else return_index < boundary
            realized = portfolio_map[portfolio_id].loc[
                (return_index > start) & boundary_mask,
                "portfolio_return",
            ]
            values = realized.to_numpy(dtype=float)
            volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else np.nan
            annual_return = float(np.mean(values) * 252.0) if len(values) else np.nan
            rows.append(
                {
                    "portfolio_id": portfolio_id,
                    "decision_date": event["decision_date"],
                    "fill_date": start,
                    "realized_window_boundary": boundary,
                    "current_fill_date_excluded": True,
                    "next_fill_date_included": has_next_fill,
                    "realized_interval_complete": has_next_fill,
                    "first_realized_session": realized.index.min() if len(realized) else pd.NaT,
                    "last_realized_session": realized.index.max() if len(realized) else pd.NaT,
                    "selected_pair_count": selected_count,
                    "predicted_estimated_sharpe": event[predicted_column],
                    "realized_net_total_return": float(np.prod(1.0 + values) - 1.0),
                    "realized_annualized_arithmetic_return": annual_return,
                    "realized_annualized_volatility": volatility,
                    "realized_annualized_sharpe": (
                        annual_return / volatility if np.isfinite(volatility) and volatility > 0 else np.nan
                    ),
                    "realized_sessions": len(values),
                    "realized_measure_includes_security_ledger_sync_and_costs": True,
                }
            )
    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for portfolio_id, group in frame.groupby("portfolio_id"):
        complete = group.loc[group["realized_interval_complete"]]
        valid = complete[
            ["predicted_estimated_sharpe", "realized_annualized_sharpe"]
        ].dropna()
        summary[portfolio_id] = {
            "observations": int(len(valid)),
            "partial_intervals_retained_but_excluded": int(
                (~group["realized_interval_complete"]).sum()
            ),
            "pearson_correlation": (
                float(valid.corr().iloc[0, 1]) if len(valid) >= 2 else None
            ),
            "mean_predicted_sharpe": (
                float(valid["predicted_estimated_sharpe"].mean()) if len(valid) else None
            ),
            "mean_realized_sharpe": (
                float(valid["realized_annualized_sharpe"].mean()) if len(valid) else None
            ),
        }
    return frame, summary


def _contribution_analysis(
    ledger: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    first_fill: pd.Timestamp,
    exact_start: pd.Timestamp | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = ledger[
        ["date", "portfolio_id", "pair", "gross_pnl", "borrow_cost", "transaction_cost"]
    ].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["net_pnl_contribution"] = (
        work["gross_pnl"] - work["borrow_cost"] - work["transaction_cost"]
    )
    study_end = pd.Timestamp(config["evaluation"]["study_end_exclusive"])
    holdout = pd.Timestamp(config["evaluation"]["retrospective_review_start"])
    segment_specs: list[tuple[str, pd.Timestamp, pd.Timestamp, str]] = [
        ("prequential_from_first_v10_selection", first_fill, study_end, ""),
        ("viewed_recent_history_fresh_cash", holdout, study_end, FRESH_SUFFIX),
    ]
    if exact_start is not None:
        segment_specs.append(
            ("stable_exact_10_common_interval_fresh_cash", exact_start, study_end, EXACT10_SUFFIX)
        )
    detail_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for portfolio_base in (MAX_SHARPE_ID, MATCHED_ID, V9_REFERENCE_ID):
        for segment, start, end, suffix in segment_specs:
            portfolio_id = portfolio_base + suffix
            selected = work.loc[
                work["portfolio_id"].eq(portfolio_id)
                & work["date"].ge(start)
                & work["date"].lt(end)
            ]
            grouped = (
                selected.groupby("pair", as_index=False)
                .agg(
                    gross_pnl=("gross_pnl", "sum"),
                    borrow_cost=("borrow_cost", "sum"),
                    transaction_cost=("transaction_cost", "sum"),
                    net_pnl_contribution=("net_pnl_contribution", "sum"),
                )
                .sort_values("net_pnl_contribution", ascending=False, kind="mergesort")
            )
            absolute_total = float(grouped["net_pnl_contribution"].abs().sum())
            signed_total = float(grouped["net_pnl_contribution"].sum())
            grouped.insert(0, "portfolio_id", portfolio_id)
            grouped.insert(1, "segment", segment)
            grouped["absolute_contribution_share"] = (
                grouped["net_pnl_contribution"].abs() / absolute_total
                if absolute_total > 0
                else 0.0
            )
            grouped["signed_contribution_share"] = (
                grouped["net_pnl_contribution"] / signed_total
                if abs(signed_total) > 1e-15
                else np.nan
            )
            detail_parts.append(grouped)
            top = grouped.iloc[0] if len(grouped) else None
            summaries.append(
                {
                    "portfolio_id": portfolio_id,
                    "segment": segment,
                    "top_positive_contribution_pair": None if top is None else str(top["pair"]),
                    "top_positive_net_pnl_contribution": None if top is None else float(top["net_pnl_contribution"]),
                    "top_positive_signed_share": None if top is None else _jsonable(float(top["signed_contribution_share"])),
                    "absolute_contribution_hhi": (
                        float(np.square(grouped["absolute_contribution_share"]).sum())
                        if len(grouped)
                        else None
                    ),
                    "net_pnl_sum": signed_total,
                }
            )
    detail = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    return detail, summaries


def _quality_checks(
    *,
    daily: pd.DataFrame,
    history: pd.DataFrame,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    portfolio_daily: pd.DataFrame,
    ledger: pd.DataFrame,
    predicted_summary: Mapping[str, Any],
    contributions: list[dict[str, Any]],
    metrics: pd.DataFrame,
    v9_frozen_metrics: pd.DataFrame,
    config: Mapping[str, Any],
    v9_verification: Mapping[str, Any],
    first_fill: pd.Timestamp,
    exact_start: pd.Timestamp | None,
) -> dict[str, Any]:
    selected_cap = int(events["selected_k_after_positive_mu_rule"].max())
    if selected_cap > int(config["selection"]["top_n"]):
        raise RuntimeError("V10 selected more than ten pairs")
    ready = events["training_status"].eq("ready")
    if ready.any() and not events.loc[ready, "training_daily_observations"].ge(
        int(config["estimation"]["minimum_daily_observations"])
    ).all():
        raise RuntimeError("ready training event violates minimum daily observations")
    if len(candidates) and candidates.duplicated(["fill_date", "pair"]).any():
        raise RuntimeError("duplicate V10 candidate snapshot key")
    strict_gate_ok = True
    selected_ticker_overlap_absent = True
    selected_cutoff_prices_valid = True
    history_check = history.copy()
    history_check["history_available"] = _as_bool(history_check["history_available"])
    history_check["active_quarter"] = _as_bool(history_check["active_quarter"])
    for event in events.to_dict("records"):
        fill = pd.Timestamp(event["fill_date"])
        snapshot = candidates.loc[pd.to_datetime(candidates.get("fill_date")).eq(fill)] if len(candidates) else candidates
        if snapshot.empty:
            continue
        quarter_numbers = {int(item) for item in str(event["training_quarter_numbers"]).split("|") if item}
        pairs = set(snapshot["pair"].astype(str))
        source = history_check.loc[
            history_check["pair"].astype(str).isin(pairs)
            & history_check["quarter_number"].astype(int).isin(quarter_numbers)
        ]
        source_counts = source.groupby("pair")["quarter_number"].nunique()
        known = pd.to_datetime(snapshot["eligible_since_date"]).le(
            pd.to_datetime(snapshot["first_training_quarter_rebalance"])
        )
        strict_gate_ok = bool(
            strict_gate_ok
            and len(quarter_numbers) == 8
            and len(source) == len(snapshot) * 8
            and source_counts.eq(8).all()
            and source["history_available"].all()
            and known.all()
            and snapshot["history_quarters"].eq(8).all()
            and snapshot["active_quarters"].ge(3).all()
            and snapshot["trade_entries"].ge(10).all()
            and np.isfinite(snapshot["expected_annual_return"].to_numpy(dtype=float)).all()
        )
        for selected_column in (
            "selected_max_sharpe",
            "selected_matched_expected_return",
        ):
            chosen = snapshot.loc[snapshot[selected_column].astype(bool)]
            names = chosen[["ticker_a", "ticker_b"]].astype(str).to_numpy().ravel()
            selected_ticker_overlap_absent = bool(
                selected_ticker_overlap_absent and len(names) == len(set(names))
            )
            cutoff = daily.loc[
                daily["date"].eq(pd.Timestamp(event["decision_date"]))
                & daily["pair"].astype(str).isin(chosen["pair"].astype(str)),
                ["price_a", "price_b"],
            ]
            selected_cutoff_prices_valid = bool(
                selected_cutoff_prices_valid
                and len(cutoff) == len(chosen)
                and np.isfinite(cutoff.to_numpy(dtype=float)).all()
                and (cutoff.to_numpy(dtype=float) > 0.0).all()
            )
    if not strict_gate_ok:
        raise RuntimeError("independent V10 strict candidate gate reconstruction failed")
    if not selected_ticker_overlap_absent or not selected_cutoff_prices_valid:
        raise RuntimeError("independent V10 selected-book constraints failed")
    portfolio_daily = portfolio_daily.copy()
    portfolio_daily["date"] = pd.to_datetime(portfolio_daily["date"], errors="coerce")
    ledger_qa = ledger[
        [
            "portfolio_id",
            "date",
            "gross_pnl",
            "borrow_cost",
            "transaction_cost",
            "leg_turnover",
            "transaction_cost_borne_by_pair",
            "transaction_cost_borne_by_cash",
        ]
    ].copy()
    ledger_qa["date"] = pd.to_datetime(ledger_qa["date"], errors="coerce")
    equity_errors: dict[str, float] = {}
    for portfolio_id, group in portfolio_daily.groupby("portfolio_id"):
        ordered = group.sort_values("date", kind="mergesort")
        reconstructed = (1.0 + ordered["portfolio_return"]).cumprod()
        equity_errors[str(portfolio_id)] = float(
            np.max(np.abs(reconstructed.to_numpy() - ordered["equity"].to_numpy()))
        )
    ledger_daily = ledger_qa.groupby(["portfolio_id", "date"], as_index=False).agg(
        gross_pnl=("gross_pnl", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        transaction_cost=("transaction_cost", "sum"),
        leg_turnover=("leg_turnover", "sum"),
        transaction_cost_borne_by_pair=("transaction_cost_borne_by_pair", "sum"),
        transaction_cost_borne_by_cash=("transaction_cost_borne_by_cash", "sum"),
    )
    aggregate = portfolio_daily[
        ["portfolio_id", "date", "gross_pnl", "borrow_cost", "transaction_cost", "leg_turnover"]
    ]
    check = aggregate.merge(
        ledger_daily,
        on=["portfolio_id", "date"],
        suffixes=("_aggregate", "_ledger"),
        validate="one_to_one",
    )
    ledger_errors = {
        column: float(
            np.max(
                np.abs(
                    check[f"{column}_aggregate"].to_numpy(dtype=float)
                    - check[f"{column}_ledger"].to_numpy(dtype=float)
                )
            )
        )
        for column in ("gross_pnl", "borrow_cost", "transaction_cost", "leg_turnover")
    }
    funding_error = float(
        np.max(
            np.abs(
                ledger_daily["transaction_cost"].to_numpy(dtype=float)
                - ledger_daily[
                    ["transaction_cost_borne_by_pair", "transaction_cost_borne_by_cash"]
                ].sum(axis=1).to_numpy(dtype=float)
            )
        )
    )
    if max([*equity_errors.values(), *ledger_errors.values(), funding_error]) > 1e-9:
        raise RuntimeError("V10 portfolio/ledger reconstruction failed")
    terminal = portfolio_daily.sort_values("date").groupby("portfolio_id", as_index=False).tail(1)
    terminal_ok = bool(
        terminal["terminal_liquidation"].astype(bool).all()
        and np.allclose(terminal["gross_exposure"], 0.0, atol=1e-12)
        and np.allclose(terminal["cash_weight"], 1.0, atol=1e-12)
    )
    if not terminal_ok:
        raise RuntimeError("not every V10 path ended with terminal liquidation")
    comparison_segments = {
        "continuous_full_calendar_with_initial_cash": (
            V9_REFERENCE_ID,
            "walkforward_top10",
            "full_calendar_with_initial_cash",
        ),
        "viewed_recent_history_fresh_cash": (
            V9_REFERENCE_ID + FRESH_SUFFIX,
            "walkforward_top10_fresh_holdout",
            "retrospective_allocator_holdout_fresh_cash",
        ),
    }
    comparison_columns = [
        "sessions",
        "calendar_years",
        "transaction_cost_sum",
        "borrow_cost_sum",
        "annualized_leg_turnover",
        "final_equity",
        "net_total_return",
        "calendar_cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
    ]
    v9_metric_errors: dict[str, dict[str, float]] = {}
    for local_segment, (local_id, frozen_id, frozen_segment) in comparison_segments.items():
        local = metrics.loc[
            metrics["portfolio_id"].eq(local_id) & metrics["segment"].eq(local_segment)
        ]
        source = v9_frozen_metrics.loc[
            v9_frozen_metrics["portfolio_id"].eq(frozen_id)
            & v9_frozen_metrics["segment"].eq(frozen_segment)
        ]
        if len(local) != 1 or len(source) != 1:
            raise RuntimeError(f"missing unique V9 metric reconstruction row: {local_segment}")
        errors = {
            column: abs(float(local.iloc[0][column]) - float(source.iloc[0][column]))
            for column in comparison_columns
        }
        if max(errors.values()) > 1e-10:
            raise RuntimeError(f"V9 reference metric reconstruction failed: {local_segment}")
        v9_metric_errors[local_segment] = errors
    certified = events.loc[
        events["max_sharpe_status"].eq("selected_dinkelbach_certified")
    ].copy()
    selected_event_count = int(events["selected_k_after_positive_mu_rule"].gt(0).sum())
    if len(certified) != selected_event_count:
        raise RuntimeError("a selected V10 event lacks a Dinkelbach certificate")
    certificate_tolerance = float(
        config["selection"]["dinkelbach_sharpe_certificate_tolerance"]
    )
    maximum_certificate_width = (
        float(certified["max_sharpe_certificate_width"].max()) if len(certified) else None
    )
    maximum_scaled_residual = (
        float(certified["max_sharpe_final_maximum_scaled_residual"].abs().max())
        if len(certified)
        else None
    )
    maximum_mip_gap = (
        float(certified["max_sharpe_final_mip_gap"].abs().max())
        if len(certified)
        else None
    )
    maximum_absolute_scaled_primal_dual_gap = (
        float(
            certified["max_sharpe_final_absolute_scaled_primal_dual_gap"]
            .abs()
            .max()
        )
        if len(certified)
        else None
    )
    direct_within_certificate = bool(
        certified["max_sharpe_estimated_sharpe"]
        .ge(certified["max_sharpe_certificate_lower"] - float(config["selection"]["solution_tolerance"]))
        .all()
        and certified["max_sharpe_estimated_sharpe"]
        .le(certified["max_sharpe_certificate_upper"] + float(config["selection"]["solution_tolerance"]))
        .all()
    )
    if (
        (maximum_certificate_width is not None and maximum_certificate_width > certificate_tolerance + 1e-9)
        or (
            maximum_scaled_residual is not None
            and maximum_scaled_residual
            > float(config["selection"]["dinkelbach_residual_tolerance"])
            + float(config["selection"]["solution_tolerance"])
        )
        or (
            maximum_absolute_scaled_primal_dual_gap is not None
            and maximum_absolute_scaled_primal_dual_gap
            > float(config["selection"]["solution_tolerance"])
        )
        or not direct_within_certificate
    ):
        raise RuntimeError("a Dinkelbach max-Sharpe certificate failed runner QA")
    v9_non_strict = int(events["v9_selected_not_strict_count"].sum())
    maximum_candidate_count = int(events["strict_eligible_pair_count"].max())
    raw_ten_pair_subset_count_before_ticker_constraints = (
        math.comb(maximum_candidate_count, 10) if maximum_candidate_count >= 10 else 0
    )
    return {
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "passed_mechanical_checks": True,
        "deployable": False,
        "historical_period_is_independent_oos": False,
        "point_in_time_universe": False,
        "v9_inputs_verified": dict(v9_verification),
        "v9_reference_metric_reconstruction_max_errors": v9_metric_errors,
        "strict_common_window_exactly_eight_completed_quarters": True,
        "strict_candidate_gates_independently_reconstructed": strict_gate_ok,
        "selected_ticker_overlap_absent": selected_ticker_overlap_absent,
        "selected_cutoff_prices_valid": selected_cutoff_prices_valid,
        "missing_daily_return_imputation_used": False,
        "cutoff_price_gate_applied": True,
        "minimum_daily_observations": int(config["estimation"]["minimum_daily_observations"]),
        "expected_return_definition": config["estimation"]["expected_return"],
        "covariance_estimator": config["estimation"]["covariance_estimator"],
        "selection_definition": "global_equal_weight_maximum_estimated_sharpe_with_nonoverlap_RLT_MILP_Dinkelbach",
        "maximum_selected_pairs": selected_cap,
        "maximum_strict_candidate_count": maximum_candidate_count,
        "raw_ten_pair_subset_count_before_ticker_constraints": (
            raw_ten_pair_subset_count_before_ticker_constraints
        ),
        "fixed_pair_weight": float(config["selection"]["target_weight_per_slot"]),
        "maximum_dinkelbach_sharpe_certificate_width": maximum_certificate_width,
        "dinkelbach_sharpe_certificate_tolerance": certificate_tolerance,
        "maximum_final_scaled_dinkelbach_residual": maximum_scaled_residual,
        "maximum_final_mip_gap": maximum_mip_gap,
        "maximum_final_absolute_scaled_primal_dual_gap": (
            maximum_absolute_scaled_primal_dual_gap
        ),
        "relative_mip_gap_is_descriptive_near_zero_objective": True,
        "direct_estimated_sharpe_within_certificate": direct_within_certificate,
        "v9_selected_pair_event_rows_failing_strict_v10_eligibility": v9_non_strict,
        "v9_reference_is_context_only_due_to_different_eligibility": True,
        "matched_expected_return_is_primary_apples_to_apples_baseline": True,
        "stress_reoptimized_selections": False,
        "paired_bootstrap_scope": (
            "descriptive conditional on the already-designed optimizer; does not correct "
            "candidate, subset, objective, or model-selection multiple testing"
        ),
        "portfolio_equity_reconstruction_max_errors": equity_errors,
        "security_ledger_aggregation_max_errors": ledger_errors,
        "transaction_cost_funding_max_error": funding_error,
        "terminal_liquidation_verified": terminal_ok,
        "first_nonempty_v10_fill": first_fill.date().isoformat(),
        "stable_exact_ten_start": None if exact_start is None else exact_start.date().isoformat(),
        "retrospective_review_start": config["evaluation"]["retrospective_review_start"],
        "prospective_comparison_start": config["evaluation"]["prospective_comparison_start"],
        "predicted_vs_realized_summary": _jsonable(predicted_summary),
        "contribution_concentration_summary": _jsonable(contributions),
        "blocking_evidence_gaps": [
            "fixed 114 identities are an ex-post ever-successful universe",
            "the V10 allocator and all historical dates were designed after reviewing this sample",
            "complete historical point-in-time universe and delisted identities are absent",
            "adjusted-close execution, fixed transaction cost, and fixed borrow are proxies",
            "historical bid-ask, locate, margin, market impact, and capacity are absent",
            "estimated expected returns and covariance from eight quarters are noisy and selection-prone",
        ],
    }


def _plot(portfolio_daily: pd.DataFrame, holdout: pd.Timestamp, path: Path) -> None:
    colors = {
        MAX_SHARPE_ID: "#075985",
        MATCHED_ID: "#b45309",
        ALL_ELIGIBLE_ID: "#15803d",
        V9_REFERENCE_ID: "#7e22ce",
    }
    labels = {
        MAX_SHARPE_ID: "Joint max estimated Sharpe",
        MATCHED_ID: "Matched max expected return",
        ALL_ELIGIBLE_ID: "Strict all eligible EW",
        V9_REFERENCE_ID: "Frozen V9 score Top10",
    }
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    continuous = portfolio_daily.loc[portfolio_daily["portfolio_id"].isin(BASE_PORTFOLIOS)]
    for portfolio_id, group in continuous.groupby("portfolio_id", sort=False):
        ordered = group.sort_values("date", kind="mergesort")
        dates = pd.to_datetime(ordered["date"])
        equity = ordered["equity"].to_numpy(dtype=float)
        drawdown = equity / np.maximum.accumulate(equity) - 1.0
        axes[0].plot(dates, equity, color=colors[portfolio_id], label=labels[portfolio_id])
        axes[1].plot(dates, drawdown, color=colors[portfolio_id], label=labels[portfolio_id])
        axes[2].plot(dates, ordered["gross_exposure"], color=colors[portfolio_id], label=labels[portfolio_id])
    for axis in axes:
        axis.axvline(holdout, color="#be123c", linestyle="--", linewidth=1.2)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title("V10 conditional fixed-114 portfolio comparison", loc="left", weight="bold")
    axes[0].set_ylabel("Equity")
    axes[1].set_ylabel("Drawdown")
    axes[2].set_ylabel("Gross exposure")
    axes[2].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _format_metrics(metrics: pd.DataFrame) -> str:
    columns = [
        "portfolio_id",
        "segment",
        "net_total_return",
        "calendar_cagr",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "annualized_leg_turnover",
        "mean_selected_pairs",
    ]
    shown = metrics[columns].copy()
    for column in ("net_total_return", "calendar_cagr", "annual_volatility", "max_drawdown"):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "annualized_leg_turnover", "mean_selected_pairs"):
        shown[column] = shown[column].map(lambda value: f"{value:.2f}")
    return shown.to_html(index=False, border=0)


def _write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    bootstrap: pd.DataFrame,
    stress: pd.DataFrame,
    churn: pd.DataFrame,
    predicted: pd.DataFrame,
    contributions: pd.DataFrame,
    quality: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    recent_segment = "viewed_recent_history_fresh_cash"
    recent = metrics.loc[metrics["segment"].eq(recent_segment)].copy()
    max_recent = recent.loc[recent["portfolio_id"].eq(MAX_SHARPE_ID + FRESH_SUFFIX)].iloc[0]
    matched_recent = recent.loc[recent["portfolio_id"].eq(MATCHED_ID + FRESH_SUFFIX)].iloc[0]
    v9_recent = recent.loc[recent["portfolio_id"].eq(V9_REFERENCE_ID + FRESH_SUFFIX)].iloc[0]
    latest_events = events.tail(10).copy()
    event_columns = [
        "fill_date",
        "strict_eligible_pair_count",
        "structural_maximum_feasible_k",
        "selected_k_after_positive_mu_rule",
        "max_sharpe_estimated_sharpe",
        "max_sharpe_selected_pairs",
        "matched_selected_pairs",
        "v9_selected_not_strict_count",
    ]
    primary_bootstrap = bootstrap.loc[
        bootstrap["segment"].eq(recent_segment)
    ].copy()
    primary_stress = stress.loc[stress["segment"].eq(recent_segment)].copy()
    churn_summary = (
        churn.groupby("portfolio_id", as_index=False)
        .agg(
            mean_jaccard=("jaccard_vs_prior", "mean"),
            total_added=("added_pair_count", "sum"),
            total_removed=("removed_pair_count", "sum"),
        )
    )
    pred_summary = (
        predicted.loc[predicted["realized_interval_complete"]]
        .groupby("portfolio_id", as_index=False)
        .agg(
            observations=("realized_annualized_sharpe", "count"),
            mean_predicted=("predicted_estimated_sharpe", "mean"),
            mean_realized=("realized_annualized_sharpe", "mean"),
        )
        if len(predicted)
        else pd.DataFrame()
    )
    contribution_recent = contributions.loc[
        contributions["segment"].eq(recent_segment)
        & contributions["portfolio_id"].eq(MAX_SHARPE_ID + FRESH_SUFFIX)
    ].head(12)
    html_text = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V10 Joint Max-Sharpe Top 10</title>
<style>body{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f3f6fa;color:#172033;margin:0}}main{{max-width:1240px;margin:26px auto;padding:0 22px 50px}}.alert{{background:#fff1f2;border-left:5px solid #be123c;padding:16px 18px;border-radius:8px;margin:18px 0}}.note{{background:#eff6ff;border-left:5px solid #075985;padding:14px 18px;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card,section{{background:#fff;border:1px solid #dce3ee;border-radius:10px;padding:16px;margin:14px 0;overflow:auto}}.big{{font-size:25px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:11px}}th,td{{padding:7px;border-bottom:1px solid #e6ebf2;text-align:right;vertical-align:top}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}img{{width:100%}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}li{{margin:6px 0}}</style></head>
<body><main><h1>V10: ten pairs with the highest joint estimated Sharpe</h1>
<p>Each quarter uses the preceding 8 complete quarters; candidates must be available over the full window, first-known before the window starts, and have at least 3 active quarters and 10 entries. Each selected pair receives a fixed 10% weight, and each ticker may appear at most once.</p>
<div class="alert"><strong>Research status: NOT VALIDATED.</strong> This is an allocator iteration on the fixed legacy 114-identity universe after the historical sample had been viewed. The post-2024-07-01 interval in this report is only a viewed-history comparison restarted from cash.</div>
<div class="note">An RLT-strengthened MILP with Dinkelbach iterations performs a global discrete portfolio search over frozen expected returns and a Ledoit-Wolf covariance; each period stores the final residual, dual bound, MIP gap and Sharpe certificate. It optimizes a training-period estimate; true future Sharpe still requires a forward comparison starting {html.escape(str(config['evaluation']['prospective_comparison_start']))}.</div>
<div class="cards"><div class="card">Viewed history Max-Sharpe CAGR<div class="big">{max_recent['calendar_cagr']:.2%}</div></div><div class="card">Matched max-return CAGR<div class="big">{matched_recent['calendar_cagr']:.2%}</div></div><div class="card">Frozen V9 CAGR<div class="big">{v9_recent['calendar_cagr']:.2%}</div></div><div class="card">Max-Sharpe realized Sharpe<div class="big">{max_recent['sharpe']:.2f}</div></div><div class="card">Max-Sharpe max drawdown<div class="big">{max_recent['max_drawdown']:.2%}</div></div></div>
<section><h2>Performance</h2>{_format_metrics(metrics)}</section>
<section><h2>Continuous paths</h2><img src="joint_max_sharpe_portfolio_comparison.png"></section>
<section><h2>Ten most recent selections</h2>{latest_events[event_columns].to_html(index=False, border=0)}</section>
<section><h2>Paired block bootstrap (viewed history)</h2><p>Descriptive comparison conditional on the chosen optimizer only; no correction for candidate, subset, objective or model-selection multiple testing. The largest candidate period has {int(quality['raw_ten_pair_subset_count_before_ticker_constraints']):,} ten-pair subsets before ticker constraints.</p>{primary_bootstrap.to_html(index=False, border=0)}</section>
<section><h2>Cost and borrow stress (fixed selection, no re-optimization)</h2>{primary_stress.to_html(index=False, border=0)}</section>
<section><h2>Selection churn</h2>{churn_summary.to_html(index=False, border=0)}</section>
<section><h2>Training-period predicted Sharpe vs next-quarter realized</h2>{pred_summary.to_html(index=False, border=0)}</section>
<section><h2>Max-Sharpe viewed-history pair P&amp;L contribution</h2>{contribution_recent.to_html(index=False, border=0)}</section>
<section><h2>Interpretation boundaries</h2><ul><li>The core apples-to-apples comparison is a matched max-expected-return portfolio: identical candidates, cardinality, ticker constraint, weights and execution; only the objective changes.</li><li>The frozen V9 reference uses the older eligibility rule, under which pairs with missing recent history could still be selected; it is background only, and this report records the V9 pairs failing the V10 strict gate each period.</li><li>"Highest" refers only to estimated Sharpe under the frozen mean and covariance, within the Dinkelbach certificate tolerance. Eight-quarter statistics are noisy, and portfolio optimization amplifies estimation error.</li><li>The security-level ledger keeps V9 timing: the allocation fill occurs after the rebalance-close return, the first new P&amp;L accrues next session, and an incoming active shadow waits for an observable flat target before syncing.</li><li>All stresses keep the original selection and only reprice costs/borrow.</li><li>A complete point-in-time universe, delisted identities, historical GICS, real bid-ask, borrow/locate, margin, impact and capacity data are not available.</li><li>Forward evidence should be assessed only after at least {int(config['evaluation']['prospective_minimum_quarters'])} future quarters have accumulated.</li></ul></section>
<section><h2>Mechanical QA</h2><pre>{html.escape(json.dumps(_jsonable(quality), indent=2))}</pre></section>
</main></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def _reveal_unlocked(config_path: Path, *, workers: int) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _output_root(config)
    data_root = output_root / "data"
    reveal_path = data_root / REVEAL_NAME
    if reveal_path.exists():
        raise RuntimeError("V10 reveal already exists; frozen results are immutable")
    freeze_path, frozen, v9_verification = _verify_freeze(config_path, config)
    freeze_hash_start = _sha256(freeze_path)
    paths = _paths(config_path, config)
    source_hashes_before = _source_hashes(paths)
    daily, history, rankings, v9_events, v9_metrics, metadata = _load_inputs(paths)
    events, candidates, covariance, solver, simulator_events = _build_selection_artifacts(
        daily,
        history,
        metadata,
        v9_events,
        config,
        workers=max(1, int(workers)),
    )
    exact_start_pre_simulation = _stable_exact_ten_start(events)
    portfolio_map, portfolio_daily, ledger = _simulate_portfolios(
        daily,
        simulator_events,
        config,
        exact_start=exact_start_pre_simulation,
    )
    metrics, annual, first_fill, exact_start = _build_metrics(portfolio_map, events, config)
    if exact_start != exact_start_pre_simulation:
        raise RuntimeError("exact-ten boundary changed between selection and evaluation")
    bootstrap = _bootstrap_comparisons(
        portfolio_map, config, first_fill=first_fill, exact_start=exact_start
    )
    stress = _stress_metrics(
        daily,
        simulator_events,
        portfolio_map,
        config,
        first_fill=first_fill,
        exact_start=exact_start,
    )
    churn = _selection_churn(events)
    predicted, predicted_summary = _predicted_vs_realized(events, portfolio_map, config)
    contribution, contribution_summary = _contribution_analysis(
        ledger, config, first_fill=first_fill, exact_start=exact_start
    )
    quality = _quality_checks(
        daily=daily,
        history=history,
        events=events,
        candidates=candidates,
        portfolio_daily=portfolio_daily,
        ledger=ledger,
        predicted_summary=predicted_summary,
        contributions=contribution_summary,
        metrics=metrics,
        v9_frozen_metrics=v9_metrics,
        config=config,
        v9_verification=v9_verification,
        first_fill=first_fill,
        exact_start=exact_start,
    )
    data_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "allocation_events": data_root / "v10_allocation_events.csv",
        "candidate_snapshots": data_root / "v10_candidate_snapshots.csv.gz",
        "annual_covariance_upper_triangle": data_root / "v10_annual_covariance_upper_triangle.csv.gz",
        "solver_certificates": data_root / "v10_solver_certificates.json",
        "simulator_events": data_root / "v10_simulator_events.json",
        "portfolio_daily": data_root / "v10_portfolio_daily.csv.gz",
        "security_ledger": data_root / "v10_portfolio_security_ledger.csv.gz",
        "portfolio_metrics": data_root / "v10_portfolio_metrics.csv",
        "portfolio_annual": data_root / "v10_portfolio_annual_returns.csv",
        "paired_bootstrap": data_root / "v10_paired_block_bootstrap.csv",
        "cost_borrow_stress": data_root / "v10_cost_borrow_stress_metrics.csv",
        "selection_churn": data_root / "v10_selection_churn.csv",
        "predicted_vs_realized": data_root / "v10_predicted_vs_next_period_realized.csv",
        "pair_contributions": data_root / "v10_pair_contribution_concentration.csv",
        "quality_checks": data_root / "v10_quality_checks.json",
        "chart": output_root / "joint_max_sharpe_portfolio_comparison.png",
        "report": output_root / str(config["outputs"]["report"]),
    }
    events.to_csv(artifact_paths["allocation_events"], index=False)
    candidates.to_csv(artifact_paths["candidate_snapshots"], index=False, compression="gzip")
    covariance.to_csv(
        artifact_paths["annual_covariance_upper_triangle"], index=False, compression="gzip"
    )
    artifact_paths["solver_certificates"].write_text(
        json.dumps(_jsonable(solver), indent=2), encoding="utf-8"
    )
    simulator_serialized = simulator_events.copy()
    simulator_serialized["decision_date"] = simulator_serialized["decision_date"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    simulator_serialized["fill_date"] = simulator_serialized["fill_date"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    artifact_paths["simulator_events"].write_text(
        json.dumps(_jsonable(simulator_serialized.to_dict("records")), indent=2),
        encoding="utf-8",
    )
    portfolio_daily.to_csv(artifact_paths["portfolio_daily"], index=False, compression="gzip")
    ledger.to_csv(artifact_paths["security_ledger"], index=False, compression="gzip")
    metrics.to_csv(artifact_paths["portfolio_metrics"], index=False)
    annual.to_csv(artifact_paths["portfolio_annual"], index=False)
    bootstrap.to_csv(artifact_paths["paired_bootstrap"], index=False)
    stress.to_csv(artifact_paths["cost_borrow_stress"], index=False)
    churn.to_csv(artifact_paths["selection_churn"], index=False)
    predicted.to_csv(artifact_paths["predicted_vs_realized"], index=False)
    contribution.to_csv(artifact_paths["pair_contributions"], index=False)
    artifact_paths["quality_checks"].write_text(
        json.dumps(_jsonable(quality), indent=2), encoding="utf-8"
    )
    _plot(
        portfolio_daily,
        pd.Timestamp(config["evaluation"]["retrospective_review_start"]),
        artifact_paths["chart"],
    )
    _write_report(
        artifact_paths["report"],
        metrics=metrics,
        events=events,
        bootstrap=bootstrap,
        stress=stress,
        churn=churn,
        predicted=predicted,
        contributions=contribution,
        quality=quality,
        config=config,
    )
    source_hashes_after = _source_hashes(paths)
    if source_hashes_after != source_hashes_before or source_hashes_after != frozen["source_hashes"]:
        raise RuntimeError("source/input mutation detected during V10 reveal")
    if _sha256(freeze_path) != freeze_hash_start:
        raise RuntimeError("V10 freeze manifest changed during reveal")
    reveal_manifest = {
        "schema": "conditional_fixed_114_joint_max_sharpe_reveal_v1",
        "revealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "freeze_manifest": {"path": str(freeze_path.resolve()), "sha256": freeze_hash_start},
        "source_hashes": source_hashes_after,
        "v9_input_verification": v9_verification,
        "output_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in artifact_paths.items()
        },
        "headline_metrics": _jsonable(metrics.to_dict("records")),
        "paired_bootstrap": _jsonable(bootstrap.to_dict("records")),
        "cost_borrow_stress": _jsonable(stress.to_dict("records")),
        "selection_summary": _jsonable(events.to_dict("records")),
        "quality_checks": _jsonable(quality),
        "v9_metrics_input_for_context": _jsonable(v9_metrics.to_dict("records")),
        "v9_ranking_rows_input": int(len(rankings)),
    }
    reveal_path.write_text(json.dumps(reveal_manifest, indent=2), encoding="utf-8")
    return {
        "phase": "reveal",
        "report": str(artifact_paths["report"].resolve()),
        "metrics": str(artifact_paths["portfolio_metrics"].resolve()),
        "quality_checks": str(artifact_paths["quality_checks"].resolve()),
        "reveal_manifest": str(reveal_path.resolve()),
        "research_status_verdict": "NOT VALIDATED",
    }


def freeze(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    with _exclusive_output_lock(_output_root(config)):
        return _freeze_unlocked(config_path)


def reveal(config_path: Path, *, workers: int) -> dict[str, Any]:
    config = _load_config(config_path)
    with _exclusive_output_lock(_output_root(config)):
        return _reveal_unlocked(config_path, workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("freeze", "reveal"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    result = (
        freeze(config_path)
        if args.phase == "freeze"
        else reveal(config_path, workers=max(1, args.workers))
    )
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
