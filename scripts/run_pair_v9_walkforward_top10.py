#!/usr/bin/env python3
"""Freeze and reveal a first-known-clean quarterly Top-10 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
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

import scripts.run_pair_v8_full_panel as v8
from equity_pairs.identity_evaluation import exact_calendar_years, path_statistics
from equity_pairs.walkforward_top10 import (
    CrossPairSelectionSpec,
    build_first_known_quarter_history,
    first_known_candidate_history,
    rank_pairs_asof,
    select_nonoverlapping_pairs,
    simulate_security_level_portfolio,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/pair_strategy_v9_cross_pair_walkforward_top10.json"
MODULE_PATH = PROJECT_ROOT / "src/equity_pairs/walkforward_top10.py"
FREEZE_NAME = "protocol_freeze_manifest.json"
REVEAL_NAME = "reveal_manifest.json"


@contextmanager
def _exclusive_output_lock(output_root: Path):
    """Prevent concurrent freeze/reveal writes into one immutable output root."""

    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    lock_path = data_root / ".v9_protocol.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"V9 output root is already locked: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.isoformat() if hasattr(item, "isoformat") else str(item),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol"]["research_status_verdict"] != "NOT VALIDATED":
        raise ValueError("historical V9 diagnostic must remain NOT VALIDATED")
    if config["protocol"]["universe_is_point_in_time"] is not False:
        raise ValueError("this runner is conditional on the fixed legacy universe")
    return config


def _paths(config_path: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    inputs = config["inputs"]
    identity_root = PROJECT_ROOT / inputs["identity_root"]
    v8_root = PROJECT_ROOT / inputs["v8_root"]
    result = {
        "config": config_path,
        "runner": Path(__file__).resolve(),
        "walkforward_module": MODULE_PATH,
        "v8_config": PROJECT_ROOT / inputs["v8_config"],
        "master": identity_root / inputs["master"],
        "schedule": identity_root / inputs["schedule"],
        "quarterly": identity_root / inputs["quarterly"],
        "prices": identity_root / inputs["prices"],
        "candidate_metrics": v8_root / inputs["candidate_metrics"],
        "s5_manifest": v8_root / inputs["s5_manifest"],
        "v8_engine_manifest": v8_root / inputs["v8_engine_manifest"],
    }
    manifest_path = result["v8_engine_manifest"]
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative_path in manifest.get("engine_sources", {}):
            result[f"v8_engine::{relative_path}"] = PROJECT_ROOT / relative_path
    return result


def _output_root(config: Mapping[str, Any]) -> Path:
    return PROJECT_ROOT / config["outputs"]["root"]


def _source_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen input/source: " + ", ".join(missing))
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in sorted(paths.items())
    }


def _verify_v8_engine_manifest(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest["engine_sources"]
    current = v8._engine_source_manifest()
    if current != recorded:
        changed = sorted(
            key for key in set(current) | set(recorded) if current.get(key) != recorded.get(key)
        )
        raise RuntimeError(f"V8 engine differs from candidate artifact: {changed}")
    return current


def _freeze_unlocked(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _output_root(config)
    data_root = output_root / "data"
    freeze_path = data_root / FREEZE_NAME
    reveal_path = data_root / REVEAL_NAME
    if reveal_path.exists():
        raise RuntimeError("revealed output already exists; use a new output root")
    if freeze_path.exists():
        raise RuntimeError("protocol is already frozen; use a new output root")
    paths = _paths(config_path, config)
    engine_sources = _verify_v8_engine_manifest(paths["v8_engine_manifest"])
    runtime = v8._runtime_manifest()
    payload = {
        "schema": "conditional_fixed_114_cross_pair_walk_forward_freeze_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "performance_computed": False,
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "config": config,
        "source_hashes": _source_hashes(paths),
        "v8_engine_sources": engine_sources,
        "runtime_manifest": runtime,
        "runtime_manifest_sha256": _payload_sha256(runtime),
    }
    payload["freeze_payload_sha256"] = _payload_sha256(payload)
    data_root.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "phase": "freeze",
        "freeze_manifest": str(freeze_path.resolve()),
        "freeze_sha256": _sha256(freeze_path),
        "performance_computed": False,
    }


def _verify_freeze(config_path: Path, config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    freeze_path = _output_root(config) / "data" / FREEZE_NAME
    if not freeze_path.is_file():
        raise RuntimeError("reveal requires a prior protocol freeze")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    claimed_payload_sha256 = frozen.get("freeze_payload_sha256")
    payload_without_claim = dict(frozen)
    payload_without_claim.pop("freeze_payload_sha256", None)
    if claimed_payload_sha256 != _payload_sha256(payload_without_claim):
        raise RuntimeError("freeze manifest payload checksum failed")
    current = _source_hashes(_paths(config_path, config))
    if frozen["source_hashes"] != current:
        changed = sorted(
            key
            for key in set(frozen["source_hashes"]) | set(current)
            if frozen["source_hashes"].get(key) != current.get(key)
        )
        raise RuntimeError(f"freeze source mismatch: {changed}")
    _verify_v8_engine_manifest(Path(current["v8_engine_manifest"]["path"]))
    runtime = v8._runtime_manifest()
    if _payload_sha256(runtime) != frozen["runtime_manifest_sha256"]:
        raise RuntimeError("runtime differs from the frozen protocol")
    if frozen["config"] != config:
        raise RuntimeError("loaded config differs from frozen config")
    return freeze_path, frozen


def _load_inputs(paths: Mapping[str, Path]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v8_config = json.loads(paths["v8_config"].read_text(encoding="utf-8"))
    master = pd.read_csv(paths["master"])
    schedule = pd.read_csv(
        paths["schedule"],
        parse_dates=["rebalance_date", "outer_end_exclusive", "formation_start_requested"],
    )
    quarterly = pd.read_csv(paths["quarterly"])
    for column in ("rebalance_date", "outer_end_exclusive", "formation_start_requested", "formation_start_actual", "formation_end_actual"):
        quarterly[column] = pd.to_datetime(quarterly[column], errors="coerce")
    candidate = pd.read_csv(paths["candidate_metrics"])
    candidate["rebalance_date"] = pd.to_datetime(candidate["rebalance_date"], errors="coerce")
    candidate["outer_end_exclusive"] = pd.to_datetime(candidate["outer_end_exclusive"], errors="coerce")
    prices = pd.read_csv(paths["prices"], index_col="date", parse_dates=True).sort_index()
    if len(master) != 114 or master["pair"].nunique() != 114:
        raise RuntimeError("expected the frozen 114-pair master")
    if len(schedule) != 41 or len(quarterly) != 4674:
        raise RuntimeError("unexpected frozen quarter panel dimensions")
    if quarterly.duplicated(["pair", "rebalance_date"]).any():
        raise RuntimeError("duplicate pair-quarter source rows")
    if candidate.duplicated(["pair", "rebalance_date", "strategy_id"]).any():
        raise RuntimeError("duplicate candidate rows")
    if not quarterly["formation_end_actual"].lt(quarterly["rebalance_date"]).all():
        raise RuntimeError("formation data reaches or crosses a rebalance date")
    return v8_config, master, schedule, quarterly, candidate, prices


def _verify_embedded_protocol(
    config: Mapping[str, Any], v8_config: Mapping[str, Any]
) -> None:
    requested = config["pair_strategy_selection"]
    source = v8_config["strategy_selection"]
    keys = (
        "lookback_calendar_years",
        "minimum_history_quarters",
        "minimum_active_quarters",
        "minimum_trades",
        "insufficient_history_default",
    )
    mismatched = [key for key in keys if requested[key] != source[key]]
    if mismatched or requested["one_standard_error_rule"] is not True:
        raise RuntimeError(f"pair-strategy protocol mismatch: {mismatched}")
    portfolio = config["portfolio"]
    evaluation = v8_config["evaluation"]
    if float(portfolio["transaction_cost_bps"]) != float(
        evaluation["primary_transaction_cost_bps"]
    ) or float(portfolio["annual_short_borrow_bps"]) != float(
        evaluation["annual_short_borrow_bps"]
    ):
        raise RuntimeError("pair shadow and portfolio cost assumptions differ")
    cross = config["cross_pair_selection"]
    if float(cross["target_weight_per_slot"]) * int(cross["top_n"]) > 1.0 + 1e-12:
        raise RuntimeError("Top-N slot weights exceed portfolio capital")
    expected_cross = {
        "decision_information_cutoff": "session_before_rebalance_close",
        "allocation_fill": "rebalance_session_close_after_that_sessions_return",
        "first_new_pnl": "next_session_after_allocation_fill",
        "first_known_rule": "first_successful_v6_tuned_date_strictly_before_rebalance",
        "score": "4 * (mean_quarterly_log_return - standard_error)",
        "unused_slots": "cash",
        "shared_ticker_policy": "greedy_skip_after_score_ranking",
    }
    expected_ties = [
        "higher_mean_quarterly_log_return",
        "lower_quarterly_log_return_std",
        "lower_mean_annualized_turnover",
        "shallower_worst_quarter_max_drawdown",
        "pair_lexicographic",
    ]
    if any(cross.get(key) != value for key, value in expected_cross.items()):
        raise RuntimeError("cross-pair protocol text does not match the engine")
    if cross.get("tie_break") != expected_ties:
        raise RuntimeError("cross-pair tie-break does not match the engine")
    expected_portfolio = {
        "accounting": "pair_segregated_two_leg_security_ledger",
        "cross_pair_trade_netting": False,
        "quarterly_cross_pair_rebalance": True,
        "daily_cross_pair_rebalance": False,
        "incoming_active_shadow_policy": "cash_until_observable_post_close_target_is_flat",
        "terminal_liquidation": True,
    }
    if any(portfolio.get(key) != value for key, value in expected_portfolio.items()):
        raise RuntimeError("portfolio protocol text does not match the engine")


def _rebuild_shadow_paths(
    *,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    v8_config: Mapping[str, Any],
    master: pd.DataFrame,
    schedule: pd.DataFrame,
    quarterly: pd.DataFrame,
    candidate: pd.DataFrame,
    prices: pd.DataFrame,
    workers: int,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    declared_hashes_before = {
        record["path"]: record["sha256"]
        for record in _source_hashes(paths).values()
    }
    engine_manifest_before = v8._engine_source_manifest()
    runtime_manifest_before = v8._runtime_manifest()
    clean_candidate = first_known_candidate_history(candidate, master)
    selection_config = json.loads(json.dumps(v8_config))
    frozen_selection = config["pair_strategy_selection"]
    selection_config["strategy_selection"].update(
        {
            "lookback_calendar_years": frozen_selection["lookback_calendar_years"],
            "minimum_history_quarters": frozen_selection["minimum_history_quarters"],
            "minimum_active_quarters": frozen_selection["minimum_active_quarters"],
            "minimum_trades": frozen_selection["minimum_trades"],
            "insufficient_history_default": frozen_selection[
                "insufficient_history_default"
            ],
        }
    )
    selections, strategy_rankings = v8._build_strategy_selection(
        quarterly, clean_candidate, selection_config
    )
    known = master[["pair", "first_successful_v6_tuned_date"]].copy()
    known["pair"] = known["pair"].astype(str)
    known["first_successful_v6_tuned_date"] = pd.to_datetime(
        known["first_successful_v6_tuned_date"], errors="coerce"
    )
    replay = quarterly.merge(known, on="pair", how="left", validate="many_to_one")
    replay = replay.loc[
        replay["rebalance_date"].ge(replay["first_successful_v6_tuned_date"])
    ].drop(columns="first_successful_v6_tuned_date")
    replay = replay.merge(
        selections[["pair", "rebalance_date", "selected_strategy_id"]],
        on=["pair", "rebalance_date"],
        how="left",
        validate="one_to_one",
    )
    if replay["selected_strategy_id"].isna().any():
        raise RuntimeError("first-known replay is missing strategy selections")

    s5 = pd.read_csv(paths["s5_manifest"])
    if s5.duplicated(["pair", "rebalance_date"]).any():
        raise RuntimeError("duplicate S5 manifest keys")
    s5_sha_by_key = {
        (str(row.pair), str(pd.Timestamp(row.rebalance_date).date())): str(row.sha256)
        for row in s5.itertuples(index=False)
    }
    s5_hashes_before = {
        str(Path(str(row.path))): str(row.sha256)
        for row in s5.itertuples(index=False)
    }
    identity_root = PROJECT_ROOT / config["inputs"]["identity_root"]
    v6_cache = identity_root / config["inputs"]["v6_pair_quarter_cache"]
    runtime_sha = _payload_sha256(runtime_manifest_before)
    initializer = (
        str(paths["prices"]),
        v8._base_config(v8_config).__dict__,
        list(v8_config["strategy_candidates"]),
        str(v6_cache),
        _sha256(paths["prices"]),
        s5_sha_by_key,
        runtime_sha,
    )
    base_fingerprint = _payload_sha256(
        {
            "freeze_sha256": _sha256(output_root / "data" / FREEZE_NAME),
            "clean_candidate_sha256": _payload_sha256(clean_candidate.to_dict("records")),
            "selection_sha256": _payload_sha256(selections.to_dict("records")),
            "engine": "v8_final_stateful_engine_v5_first_known_restart",
        }
    )
    run_nonce = uuid.uuid4().hex
    tasks: list[tuple[list[dict[str, Any]], str, dict[str, Any]]] = []
    cache_root = output_root / "data/cache/first_known_shadow"
    for pair, group in replay.groupby("pair", sort=True):
        records = group.sort_values("rebalance_date", kind="mergesort").to_dict("records")
        group_hash = _payload_sha256(records)
        tasks.append(
            (
                records,
                str(cache_root / v8._safe(pair)),
                {
                    "fingerprint": _payload_sha256(
                        {"base": base_fingerprint, "pair_rows": group_hash}
                    ),
                    "payload_sha256": group_hash,
                    "run_nonce": run_nonce,
                    "merged_group_sha256": group_hash,
                },
            )
        )
    results = v8._run_parallel(
        tasks,
        v8._final_pair_task,
        workers=workers,
        initializer_args=initializer,
        label="first-known stateful replay",
    )
    v8._validate_input_stage_and_promote_cache(
        stage="stateful_pair",
        results=results,
        file_hashes_before=declared_hashes_before,
        engine_manifest_before=engine_manifest_before,
        s5_hashes_before=s5_hashes_before,
        runtime_manifest_before=runtime_manifest_before,
        audit_path=output_root / "data/v9_stateful_input_mutation_audit.csv.gz",
    )
    sparse = pd.concat(
        [row["_validated_frame"] for row in results], ignore_index=True, sort=False
    )
    sparse["date"] = pd.to_datetime(sparse["date"], errors="coerce")
    if sparse["date"].isna().any() or sparse.duplicated(["pair", "date"]).any():
        raise RuntimeError("first-known sparse replay has invalid keys")
    first_rows = sparse.sort_values(["pair", "date"]).groupby("pair", sort=False).head(1)
    if first_rows[["weight_a", "weight_b"]].abs().to_numpy().max() > 1e-12:
        raise RuntimeError("a first-known pair replay inherited held exposure")
    artifacts = v8._aggregate_paths(
        master, schedule, quarterly, sparse, prices, selections, v8_config
    )
    daily = artifacts["daily"]
    daily["date"] = pd.to_datetime(daily["date"])
    preknown = daily.merge(
        known.rename(columns={"first_successful_v6_tuned_date": "first_known_date"}),
        on="pair",
        how="left",
        validate="many_to_one",
    )
    preknown = preknown.loc[preknown["date"].lt(preknown["first_known_date"])]
    economics = [
        "gross_return",
        "turnover",
        "transaction_cost",
        "borrow_cost",
        "net_return_1bp",
        "net_return_2bp",
        "weight_a",
        "weight_b",
        "target_weight_a",
        "target_weight_b",
    ]
    if len(preknown) and preknown[economics].abs().to_numpy().max() > 1e-12:
        raise RuntimeError("pre-known shadow rows contain economic activity")
    return daily, artifacts["quarterly"], selections, strategy_rankings


def _selection_ledgers(
    quarter_history: pd.DataFrame,
    schedule: pd.DataFrame,
    master: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = config["cross_pair_selection"]
    spec = CrossPairSelectionSpec(
        lookback_calendar_years=int(raw["lookback_calendar_years"]),
        minimum_history_quarters=int(raw["minimum_history_quarters"]),
        minimum_active_quarters=int(raw["minimum_active_quarters"]),
        minimum_trade_entries=int(raw["minimum_trade_entries"]),
        top_n=int(raw["top_n"]),
    )
    ranking_parts: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    simulator_rows: list[dict[str, Any]] = []
    for qrow in schedule.sort_values("rebalance_date").itertuples(index=False):
        fill = pd.Timestamp(qrow.rebalance_date)
        prior = price_dates[price_dates < fill]
        if not len(prior):
            raise RuntimeError(f"no signal cutoff before {fill.date()}")
        decision = pd.Timestamp(prior[-1])
        ranked = rank_pairs_asof(
            quarter_history,
            fill,
            spec,
            information_cutoff=decision,
        )
        if len(ranked):
            ranked = select_nonoverlapping_pairs(
                ranked, master[["pair", "ticker_a", "ticker_b"]], top_n=spec.top_n
            )
            ranked.insert(1, "information_cutoff", decision)
            ranked.insert(2, "allocation_fill_date", fill)
            ranking_parts.append(ranked)
            selected = ranked.loc[ranked["selected"], "pair"].astype(str).tolist()
            eligible = ranked.loc[ranked["eligible"], "pair"].astype(str).tolist()
        else:
            selected, eligible = [], []
        top_targets = {pair: float(raw["target_weight_per_slot"]) for pair in selected}
        eligible_targets = (
            {pair: 1.0 / len(eligible) for pair in eligible} if eligible else {}
        )
        event_rows.append(
            {
                "quarter_number": int(qrow.quarter_number),
                "decision_date": decision,
                "fill_date": fill,
                "eligible_pair_count": len(eligible),
                "selected_pair_count": len(selected),
                "selected_pairs": "|".join(selected),
                "top10_target_invested_weight": float(sum(top_targets.values())),
                "top10_target_cash_weight": float(1.0 - sum(top_targets.values())),
            }
        )
        simulator_rows.extend(
            [
                {
                    "portfolio_id": "walkforward_top10",
                    "decision_date": decision,
                    "fill_date": fill,
                    "target_weights": top_targets,
                },
                {
                    "portfolio_id": "all_eligible_equal_weight",
                    "decision_date": decision,
                    "fill_date": fill,
                    "target_weights": eligible_targets,
                },
            ]
        )
    rankings = (
        pd.concat(ranking_parts, ignore_index=True)
        if ranking_parts
        else pd.DataFrame()
    )
    events = pd.DataFrame(event_rows)
    simulator = pd.DataFrame(simulator_rows)
    return rankings, events, simulator


def _metric_row(
    daily: pd.DataFrame,
    *,
    portfolio_id: str,
    segment: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> dict[str, Any]:
    work = daily.loc[(daily.index >= start) & (daily.index < end_exclusive)]
    if work.empty:
        raise ValueError(f"empty metric segment {segment}")
    years = exact_calendar_years(start, end_exclusive)
    stats = path_statistics(work["portfolio_return"].to_numpy(), calendar_years=years)
    return {
        "portfolio_id": portfolio_id,
        "segment": segment,
        "start_inclusive": start.date().isoformat(),
        "end_exclusive": end_exclusive.date().isoformat(),
        "sessions": int(len(work)),
        "calendar_years": years,
        "transaction_cost_sum": float(work["transaction_cost"].sum()),
        "transaction_cost_drag_bps": float(work["transaction_cost_rate"].sum() * 10_000.0),
        "borrow_cost_sum": float(work["borrow_cost"].sum()),
        "borrow_cost_drag_bps": float(work["borrow_cost_rate"].sum() * 10_000.0),
        "annualized_leg_turnover": float(work["leg_turnover_rate"].mean() * 252.0),
        "mean_selected_pairs": float(work["selected_pair_count"].mean()),
        "mean_synced_pairs": float(work["synced_pair_count"].mean()),
        "mean_gross_exposure": float(work["gross_exposure"].mean()),
        "mean_cash_weight": float(work["cash_weight"].mean()),
        **stats,
    }


def _annual_returns(daily: pd.DataFrame, portfolio_id: str) -> pd.DataFrame:
    returns = (1.0 + daily["portfolio_return"]).groupby(daily.index.year).prod() - 1.0
    return pd.DataFrame(
        {
            "portfolio_id": portfolio_id,
            "calendar_year": returns.index.astype(int),
            "net_return": returns.to_numpy(dtype=float),
        }
    )


def _paired_block_bootstrap(
    top: pd.Series,
    benchmark: pd.Series,
    *,
    block: int,
    draws: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    aligned = pd.concat([top.rename("top"), benchmark.rename("benchmark")], axis=1).dropna()
    difference = np.log1p(aligned["top"].to_numpy()) - np.log1p(
        aligned["benchmark"].to_numpy()
    )
    if len(difference) < block:
        raise ValueError("bootstrap segment is shorter than one block")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    starts_needed = math.ceil(len(difference) / block)
    offsets = np.arange(block)
    for draw in range(draws):
        starts = rng.integers(0, len(difference), size=starts_needed)
        indexes = (starts[:, None] + offsets[None, :]) % len(difference)
        sample = difference[indexes.ravel()[: len(difference)]]
        estimates[draw] = float(sample.mean() * 252.0)
    observed = float(difference.mean() * 252.0)
    low, median, high = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "segment": label,
        "sessions": int(len(difference)),
        "annualized_log_return_difference": observed,
        "bootstrap_median": float(median),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "bootstrap_probability_greater_than_zero": float(np.mean(estimates > 0.0)),
        "block_sessions": int(block),
        "draws": int(draws),
        "seed": int(seed),
        "interpretation": "descriptive paired block bootstrap; no independent-holdout claim",
    }


def _plot(daily: pd.DataFrame, holdout: pd.Timestamp, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {
        "walkforward_top10": "#0b6e75",
        "all_eligible_equal_weight": "#d97706",
    }
    labels = {
        "walkforward_top10": "Walk-forward Top 10",
        "all_eligible_equal_weight": "All eligible equal weight",
    }
    for portfolio_id, group in daily.groupby("portfolio_id", sort=False):
        if portfolio_id not in colors:
            continue
        work = group.set_index("date").sort_index()
        axes[0].plot(work.index, work["equity"], label=labels[portfolio_id], color=colors[portfolio_id])
        axes[1].plot(work.index, work["gross_exposure"], label=labels[portfolio_id], color=colors[portfolio_id])
    for axis in axes:
        axis.axvline(holdout, color="#be123c", linestyle="--", linewidth=1.2)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_title("Conditional fixed-114 walk-forward portfolio", loc="left", weight="bold")
    axes[0].set_ylabel("Equity")
    axes[1].set_ylabel("Gross exposure")
    axes[1].set_xlabel("Date")
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
        "mean_synced_pairs",
    ]
    shown = metrics[columns].copy()
    for column in ("net_total_return", "calendar_cagr", "annual_volatility", "max_drawdown"):
        shown[column] = shown[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe", "annualized_leg_turnover", "mean_selected_pairs", "mean_synced_pairs"):
        shown[column] = shown[column].map(lambda value: f"{value:.2f}")
    return shown.to_html(index=False, border=0)


def _write_report(
    path: Path,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    bootstrap: pd.DataFrame,
    stress: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    top_holdout = metrics.loc[
        metrics["portfolio_id"].eq("walkforward_top10_fresh_holdout")
        & metrics["segment"].eq("retrospective_allocator_holdout_fresh_cash")
    ].iloc[0]
    compare = metrics.loc[
        metrics["portfolio_id"].eq("all_eligible_equal_weight_fresh_holdout")
        & metrics["segment"].eq("retrospective_allocator_holdout_fresh_cash")
    ].iloc[0]
    recent = events.loc[events["selected_pair_count"].gt(0)].tail(10).copy()
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V9 Walk-forward Top 10</title>
<style>body{{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f7fb;color:#172033;margin:0}}main{{max-width:1180px;margin:28px auto;padding:0 22px 50px}}.alert{{background:#fff1f2;border-left:5px solid #be123c;padding:16px 18px;border-radius:8px;margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card,section{{background:white;border:1px solid #dce3ee;border-radius:10px;padding:16px;margin:14px 0}}.big{{font-size:26px;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #e6ebf2;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}img{{width:100%}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}</style></head>
<body><main><h1>Quarterly Walk-forward Constrained Top 10 Pair Portfolio</h1>
<p>Pair state restarts at first-known date; each quarter uses only preceding complete quarters; 10% per slot; 2 bp on both legs' notional and 30 bp/yr short borrow.</p>
<div class="alert"><strong>Research status: NOT VALIDATED.</strong> This is a historical causal allocator diagnostic on the fixed legacy 114-identity universe. The 114 identities come from an ex-post full-sample set; post-2024-07-01 is only a retrospective allocator holdout, not a truly untouched out-of-sample period.</div>
<div class="cards"><div class="card">Historical holdout Top10 CAGR<div class="big">{top_holdout['calendar_cagr']:.2%}</div></div><div class="card">Historical holdout benchmark CAGR<div class="big">{compare['calendar_cagr']:.2%}</div></div><div class="card">Top10 Sharpe<div class="big">{top_holdout['sharpe']:.2f}</div></div><div class="card">Top10 max drawdown<div class="big">{top_holdout['max_drawdown']:.2%}</div></div></div>
<section><h2>Results</h2>{_format_metrics(metrics)}</section>
<section><h2>Path and risk exposure</h2><img src="walkforward_top10_equity_exposure.png"></section>
<section><h2>Ten most recent valid selections</h2>{recent.to_html(index=False, border=0)}</section>
<section><h2>Paired block bootstrap</h2>{bootstrap.to_html(index=False, border=0)}</section>
<section><h2>Cost and borrow stress</h2>{stress.to_html(index=False, border=0)}</section>
<section><h2>Interpretation boundaries</h2><ul><li>The constrained Top 10 sorts by score, then skips candidates sharing a ticker with an already selected pair, so the selection may include raw rank &gt; 10.</li><li>The benchmark allows ticker overlap and keeps per-pair accounts, so the difference reflects both ranking and the overlap constraint.</li><li>Within-pair candidate history is truncated at first-known, and each shadow path is rerun from flat.</li><li>If a newly selected pair's shadow path already holds a position, capital waits until an observable post-close flat target before taking over.</li><li>The portfolio keeps per-pair accounts without cross-pair netting; adjusted close, a fixed 2 bp cost and 30 bp borrow remain execution proxies.</li><li>A complete point-in-time universe, delisted identities, historical GICS and real bid-ask/borrow/locate data are not available.</li><li>The true prospective holdout starts {config['evaluation']['prospective_holdout_start']}; frozen rules must run for at least {config['evaluation']['prospective_holdout_minimum_quarters']} quarters.</li></ul></section>
</main></body></html>"""
    path.write_text(html, encoding="utf-8")


def _reveal_unlocked(config_path: Path, *, workers: int) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _output_root(config)
    data_root = output_root / "data"
    reveal_path = data_root / REVEAL_NAME
    if reveal_path.exists():
        raise RuntimeError("reveal already exists; frozen results are immutable")
    freeze_path, frozen = _verify_freeze(config_path, config)
    freeze_sha_start = _sha256(freeze_path)
    paths = _paths(config_path, config)
    before = _source_hashes(paths)
    v8_config, master, schedule, quarterly, candidate, prices = _load_inputs(paths)
    _verify_embedded_protocol(config, v8_config)
    daily, rebuilt_quarterly, selections, strategy_rankings = _rebuild_shadow_paths(
        config=config,
        paths=paths,
        v8_config=v8_config,
        master=master,
        schedule=schedule,
        quarterly=quarterly,
        candidate=candidate,
        prices=prices,
        workers=workers,
        output_root=output_root,
    )
    quarter_history = build_first_known_quarter_history(
        daily, schedule, rebuilt_quarterly, master
    )
    study_start = pd.Timestamp(config["evaluation"]["study_start_inclusive"])
    study_end = pd.Timestamp(config["evaluation"]["study_end_exclusive"])
    price_dates = prices.index[(prices.index >= study_start) & (prices.index < study_end)]
    rankings, events, simulator_events = _selection_ledgers(
        quarter_history, schedule, master, prices.index, config
    )
    if rankings.empty or not events["selected_pair_count"].gt(0).any():
        raise RuntimeError("walk-forward rule never selected a pair")
    if rankings.loc[rankings["selected"], "selection_rank"].max() > int(
        config["cross_pair_selection"]["top_n"]
    ):
        raise RuntimeError("selection exceeds Top N")
    eligible = rankings.loc[rankings["eligible"]]
    if not eligible["eligible_since_date"].le(eligible["information_cutoff"]).all():
        raise RuntimeError("an eligible identity was unknown at its information cutoff")
    if not eligible["training_end_exclusive_max"].le(eligible["allocation_fill_date"]).all():
        raise RuntimeError("an eligible cross-pair ranking read an unfinished quarter")
    if not eligible["training_last_session_max"].le(eligible["information_cutoff"]).all():
        raise RuntimeError("an eligible cross-pair ranking read data after its information cutoff")
    selected = rankings.loc[rankings["selected"]]
    for _, group in selected.groupby("allocation_fill_date"):
        names = group[["ticker_a", "ticker_b"]].astype(str).to_numpy().ravel()
        if len(names) != len(set(names)):
            raise RuntimeError("a selected book contains a repeated ticker")

    portfolio_frames: list[pd.DataFrame] = []
    security_ledger_frames: list[pd.DataFrame] = []
    portfolio_map: dict[str, pd.DataFrame] = {}
    portfolio_config = config["portfolio"]
    for portfolio_id, group in simulator_events.groupby("portfolio_id", sort=False):
        result, security_ledger = simulate_security_level_portfolio(
            daily,
            group[["decision_date", "fill_date", "target_weights"]],
            portfolio_id=str(portfolio_id),
            transaction_cost_bps=float(portfolio_config["transaction_cost_bps"]),
            annual_short_borrow_bps=float(portfolio_config["annual_short_borrow_bps"]),
            terminal_liquidation=bool(portfolio_config["terminal_liquidation"]),
            return_detail=True,
        )
        portfolio_map[str(portfolio_id)] = result
        portfolio_frames.append(result.reset_index())
        security_ledger_frames.append(security_ledger)
    first_fill = pd.Timestamp(events.loc[events["selected_pair_count"].gt(0), "fill_date"].min())
    holdout = pd.Timestamp(config["evaluation"]["retrospective_allocator_holdout_rebalance"])
    holdout_signals = daily.loc[daily["date"].ge(holdout)].copy()
    for portfolio_id, group in simulator_events.groupby("portfolio_id", sort=False):
        fresh_id = f"{portfolio_id}_fresh_holdout"
        holdout_events = group.loc[group["fill_date"].ge(holdout), ["decision_date", "fill_date", "target_weights"]]
        result, security_ledger = simulate_security_level_portfolio(
            holdout_signals,
            holdout_events,
            portfolio_id=fresh_id,
            transaction_cost_bps=float(portfolio_config["transaction_cost_bps"]),
            annual_short_borrow_bps=float(portfolio_config["annual_short_borrow_bps"]),
            terminal_liquidation=bool(portfolio_config["terminal_liquidation"]),
            return_detail=True,
        )
        portfolio_map[fresh_id] = result
        portfolio_frames.append(result.reset_index())
        security_ledger_frames.append(security_ledger)
    metric_rows: list[dict[str, Any]] = []
    annual_parts: list[pd.DataFrame] = []
    for portfolio_id, result in portfolio_map.items():
        annual_parts.append(_annual_returns(result, portfolio_id))
        if portfolio_id.endswith("_fresh_holdout"):
            segments = {
                "retrospective_allocator_holdout_fresh_cash": (holdout, study_end)
            }
        else:
            segments = {
                "full_calendar_with_initial_cash": (study_start, study_end),
                "prequential_from_first_selection": (first_fill, study_end),
                "development_before_retrospective_holdout": (first_fill, holdout),
            }
        for segment, (start, end) in segments.items():
            metric_rows.append(
                _metric_row(
                    result,
                    portfolio_id=portfolio_id,
                    segment=segment,
                    start=start,
                    end_exclusive=end,
                )
            )
    metrics = pd.DataFrame(metric_rows)
    annual = pd.concat(annual_parts, ignore_index=True)
    bootstrap_config = config["evaluation"]
    bootstrap_rows = []
    comparisons = {
        "prequential_from_first_selection": (
            portfolio_map["walkforward_top10"],
            portfolio_map["all_eligible_equal_weight"],
            first_fill,
            study_end,
        ),
        "retrospective_allocator_holdout_fresh_cash": (
            portfolio_map["walkforward_top10_fresh_holdout"],
            portfolio_map["all_eligible_equal_weight_fresh_holdout"],
            holdout,
            study_end,
        ),
    }
    for label, (top_frame, benchmark_frame, start, end) in comparisons.items():
        top = top_frame.loc[start:end - pd.Timedelta(nanoseconds=1), "portfolio_return"]
        benchmark = benchmark_frame.loc[start:end - pd.Timedelta(nanoseconds=1), "portfolio_return"]
        bootstrap_rows.append(
            _paired_block_bootstrap(
                top,
                benchmark,
                block=int(bootstrap_config["bootstrap_block_sessions"]),
                draws=int(bootstrap_config["bootstrap_draws"]),
                seed=int(bootstrap_config["bootstrap_seed"]),
                label=label,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    top_event_group = simulator_events.loc[
        simulator_events["portfolio_id"].eq("walkforward_top10"),
        ["decision_date", "fill_date", "target_weights"],
    ]
    stress_scenarios = [
        (
            float(portfolio_config["transaction_cost_bps"]),
            float(portfolio_config["annual_short_borrow_bps"]),
        ),
        *[
            (float(cost), float(portfolio_config["annual_short_borrow_bps"]))
            for cost in portfolio_config["stress_transaction_cost_bps"]
        ],
        *[
            (float(portfolio_config["transaction_cost_bps"]), float(borrow))
            for borrow in portfolio_config["stress_annual_short_borrow_bps"]
        ],
    ]
    stress_rows: list[dict[str, Any]] = []
    for transaction_bps, borrow_bps in stress_scenarios:
        primary = bool(
            transaction_bps == float(portfolio_config["transaction_cost_bps"])
            and borrow_bps == float(portfolio_config["annual_short_borrow_bps"])
        )
        if primary:
            continuous = portfolio_map["walkforward_top10"]
            fresh = portfolio_map["walkforward_top10_fresh_holdout"]
        else:
            continuous = simulate_security_level_portfolio(
                daily,
                top_event_group,
                portfolio_id="stress",
                transaction_cost_bps=transaction_bps,
                annual_short_borrow_bps=borrow_bps,
                terminal_liquidation=True,
            )
            fresh = simulate_security_level_portfolio(
                holdout_signals,
                top_event_group.loc[top_event_group["fill_date"].ge(holdout)],
                portfolio_id="stress_fresh_holdout",
                transaction_cost_bps=transaction_bps,
                annual_short_borrow_bps=borrow_bps,
                terminal_liquidation=True,
            )
        for segment, frame, start in (
            ("prequential_from_first_selection", continuous, first_fill),
            ("retrospective_allocator_holdout_fresh_cash", fresh, holdout),
        ):
            row = _metric_row(
                frame,
                portfolio_id="walkforward_top10",
                segment=segment,
                start=start,
                end_exclusive=study_end,
            )
            stress_rows.append(
                {
                    "transaction_cost_bps": transaction_bps,
                    "annual_short_borrow_bps": borrow_bps,
                    "primary_scenario": primary,
                    "segment": segment,
                    "net_total_return": row["net_total_return"],
                    "calendar_cagr": row["calendar_cagr"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                    "annualized_leg_turnover": row["annualized_leg_turnover"],
                }
            )
    stress = pd.DataFrame(stress_rows)

    data_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "first_known_shadow_daily": data_root / "v9_first_known_shadow_daily.csv.gz",
        "stateful_input_mutation_audit": data_root / "v9_stateful_input_mutation_audit.csv.gz",
        "first_known_shadow_quarterly": data_root / "v9_first_known_shadow_quarterly.csv.gz",
        "strategy_selections": data_root / "v9_first_known_strategy_selections.csv",
        "strategy_rankings": data_root / "v9_first_known_strategy_rankings.csv.gz",
        "quarter_history": data_root / "v9_cross_pair_quarter_history.csv.gz",
        "cross_pair_rankings": data_root / "v9_cross_pair_rankings.csv.gz",
        "allocation_events": data_root / "v9_allocation_events.csv",
        "portfolio_daily": data_root / "v9_portfolio_daily.csv.gz",
        "security_ledger": data_root / "v9_portfolio_security_ledger.csv.gz",
        "portfolio_metrics": data_root / "v9_portfolio_metrics.csv",
        "portfolio_annual": data_root / "v9_portfolio_annual_returns.csv",
        "paired_bootstrap": data_root / "v9_paired_block_bootstrap.csv",
        "cost_borrow_stress": data_root / "v9_cost_borrow_stress_metrics.csv",
        "quality_checks": data_root / "v9_quality_checks.json",
        "chart": output_root / "walkforward_top10_equity_exposure.png",
        "report": output_root / config["outputs"]["report"],
    }
    daily.to_csv(artifact_paths["first_known_shadow_daily"], index=False)
    rebuilt_quarterly.to_csv(artifact_paths["first_known_shadow_quarterly"], index=False, compression="gzip")
    selections.to_csv(artifact_paths["strategy_selections"], index=False)
    strategy_rankings.to_csv(artifact_paths["strategy_rankings"], index=False, compression="gzip")
    quarter_history.to_csv(artifact_paths["quarter_history"], index=False, compression="gzip")
    rankings.to_csv(artifact_paths["cross_pair_rankings"], index=False, compression="gzip")
    events.to_csv(artifact_paths["allocation_events"], index=False)
    portfolio_daily = pd.concat(portfolio_frames, ignore_index=True)
    security_ledger = pd.concat(security_ledger_frames, ignore_index=True)
    portfolio_daily.to_csv(artifact_paths["portfolio_daily"], index=False, compression="gzip")
    security_ledger.to_csv(artifact_paths["security_ledger"], index=False, compression="gzip")
    metrics.to_csv(artifact_paths["portfolio_metrics"], index=False)
    annual.to_csv(artifact_paths["portfolio_annual"], index=False)
    bootstrap.to_csv(artifact_paths["paired_bootstrap"], index=False)
    stress.to_csv(artifact_paths["cost_borrow_stress"], index=False)
    _plot(portfolio_daily, holdout, artifact_paths["chart"])

    after = _source_hashes(paths)
    if before != after or before != frozen["source_hashes"]:
        raise RuntimeError("source mutation detected during reveal")
    top_daily = portfolio_map["walkforward_top10"]
    equity_error = float(
        np.max(
            np.abs(
                (1.0 + top_daily["portfolio_return"]).cumprod().to_numpy()
                - top_daily["equity"].to_numpy()
            )
        )
    )
    ledger_daily = security_ledger.groupby(["portfolio_id", "date"], as_index=False).agg(
        gross_pnl=("gross_pnl", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        transaction_cost=("transaction_cost", "sum"),
        leg_turnover=("leg_turnover", "sum"),
        transaction_cost_borne_by_pair=("transaction_cost_borne_by_pair", "sum"),
        transaction_cost_borne_by_cash=("transaction_cost_borne_by_cash", "sum"),
    )
    aggregate_daily = portfolio_daily[
        ["portfolio_id", "date", "gross_pnl", "borrow_cost", "transaction_cost", "leg_turnover"]
    ]
    ledger_check = aggregate_daily.merge(
        ledger_daily,
        on=["portfolio_id", "date"],
        suffixes=("_aggregate", "_ledger"),
        validate="one_to_one",
    )
    ledger_errors = {
        column: float(
            np.max(
                np.abs(
                    ledger_check[f"{column}_aggregate"].to_numpy(dtype=float)
                    - ledger_check[f"{column}_ledger"].to_numpy(dtype=float)
                )
            )
        )
        for column in ("gross_pnl", "borrow_cost", "transaction_cost", "leg_turnover")
    }
    if max(ledger_errors.values()) > 1e-10:
        raise RuntimeError(f"security ledger aggregation failed: {ledger_errors}")
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
    if funding_error > 1e-10:
        raise RuntimeError(f"transaction-cost funding failed: {funding_error}")
    quality = {
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "passed_mechanical_checks": True,
        "deployable": False,
        "point_in_time_universe": False,
        "historical_holdout_untouched": False,
        "freeze_verified": True,
        "source_immutability_verified": True,
        "pair_candidate_history_starts_first_known": True,
        "pair_shadow_state_restarts_flat": True,
        "pre_known_shadow_economics_zero": True,
        "cross_pair_history_completed_before_or_at_rebalance": True,
        "identity_known_by_information_cutoff": True,
        "selected_pair_cap_respected": True,
        "selected_ticker_overlap_absent": True,
        "selection_definition": "score_ranked_greedy_nonoverlapping_up_to_10",
        "maximum_raw_rank_selected": int(selected["rank"].max()),
        "incoming_shadow_position_inheritance_blocked_by_online_flat_sync": True,
        "security_leg_transaction_cost_recomputed_once": True,
        "portfolio_equity_reconstruction_max_error": equity_error,
        "security_ledger_aggregation_max_errors": ledger_errors,
        "transaction_cost_funding_max_error": funding_error,
        "first_nonempty_selection_fill": first_fill.date().isoformat(),
        "retrospective_allocator_holdout_start": holdout.date().isoformat(),
        "prospective_holdout_start": config["evaluation"]["prospective_holdout_start"],
        "blocking_evidence_gaps": [
            "fixed 114 identities are an ex-post ever-successful universe",
            "complete historical point-in-time candidate universe and delisted identities are absent",
            "adjusted-close execution, flat transaction cost, and borrow are proxies",
            "historical minute fills, bid-ask, locate, margin, and capacity are absent",
            "retrospective allocator holdout was previously visible to the broader research process",
        ],
    }
    artifact_paths["quality_checks"].write_text(json.dumps(quality, indent=2), encoding="utf-8")
    _write_report(artifact_paths["report"], metrics, events, bootstrap, stress, config)
    reveal_manifest = {
        "schema": "conditional_fixed_114_cross_pair_walk_forward_reveal_v1",
        "revealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status_verdict": "NOT VALIDATED",
        "claim_scope": config["protocol"]["claim_scope"],
        "freeze_manifest": {"path": str(freeze_path.resolve()), "sha256": freeze_sha_start},
        "source_hashes": after,
        "output_files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in artifact_paths.items()
        },
        "headline_metrics": metrics.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records"),
        "cost_borrow_stress": stress.to_dict("records"),
        "quality_checks": quality,
    }
    if _sha256(freeze_path) != freeze_sha_start:
        raise RuntimeError("freeze manifest changed during reveal")
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
    result = freeze(args.config) if args.phase == "freeze" else reveal(args.config, workers=args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
