#!/usr/bin/env python3
"""Fail-closed SW/FERG lineage correction and fixed-114 pair replay.

Only the two vendor-backfilled ticker histories are changed.  Every affected
pair screen is recomputed and BH-FDR is re-applied to each complete sector.
Unchanged pair-quarter caches are copied only after exact price-slice and
execution-engine lineage checks.  Stale source caches are skipped so the base
runner recomputes them under the current tuner/evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equity_pairs.data import load_timeseries_snapshot, save_timeseries_snapshot
from equity_pairs.statistics import (
    PairScreenSettings,
    benjamini_hochberg,
    evaluate_pair,
    integration_diagnostics,
)
from equity_pairs.tradable_lineage import (
    apply_us_tradable_cutoffs,
    lineage_security_master,
    lineage_source_register,
)


DEFAULT_CONFIG = Path("configs/pair_identity_pit_lineage_10y.json")
AFFECTED_TICKERS = frozenset({"SW", "FERG"})


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module(
    PROJECT_ROOT / "scripts" / "run_pair_identity_10y_evaluation.py",
    "pair_identity_base_for_pit_lineage",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def _source(config: Mapping[str, Any], key: str) -> Path:
    return (PROJECT_ROOT / config["immutable_sources"][key]).resolve()


def prepare(
    paths: Any,
    config: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    original_path = _source(config, "original_spliced_price_snapshot")
    original, original_meta = load_timeseries_snapshot(original_path)
    corrected, cutoff_audit = apply_us_tradable_cutoffs(
        original, config["data"]["us_tradable_start"]
    )
    _, output_price_path = BASE._price_paths(paths, config)
    metadata = save_timeseries_snapshot(
        corrected,
        output_price_path,
        metadata={
            "source": "Yahoo adjusted close with fail-closed US-listing cutoffs",
            "parent_snapshot": str(original_path),
            "parent_snapshot_sha256": original_meta["sha256"],
            "lineage_policy": config["data"]["predecessor_policy"],
            "changed_tickers": sorted(AFFECTED_TICKERS),
            "us_tradable_start": config["data"]["us_tradable_start"],
            "strategy_logic_changed": bool(
                config["protocol"].get("strategy_mutation_allowed", False)
            ),
            "strategy_mutation_scope": config["protocol"].get(
                "strategy_mutation_scope", "none"
            ),
        },
    )
    shutil.copy2(sources["v6_universe"], paths.raw / "sp500_current_constituents.csv")
    lineage_dir = paths.root / "data" / "lineage"
    security_master = lineage_security_master()
    source_register = lineage_source_register()
    _csv(security_master, lineage_dir / "sw_ferg_security_lineage.csv")
    _csv(source_register, lineage_dir / "sw_ferg_lineage_sources.csv")
    _csv(cutoff_audit, lineage_dir / "us_tradable_cutoff_audit.csv")
    _json(
        lineage_dir / "lineage_policy.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_rule": "US-executable security only; prelisting vendor backfill is unavailable cash",
            "cross_venue_splice_used": False,
            "synthetic_fx_conversion_used": False,
            "reasons": [
                "LSE closes before NYSE and is denominated in GBp",
                "SMFKY terminated for cash rather than delivering SW shares",
                "Yahoo no longer provides the delisted SMFKY and FERGY histories",
                "using LSE closes would alter the unchanged V6 execution clock",
            ],
            "price_snapshot_sha256": metadata["sha256"],
        },
    )
    _json(
        paths.processed / "price_splice_audit.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_snapshot": str(original_path),
            "parent_snapshot_sha256": original_meta["sha256"],
            "corrected_snapshot": str(output_price_path),
            "corrected_snapshot_sha256": metadata["sha256"],
            "only_columns_permitted_to_change": sorted(AFFECTED_TICKERS),
            "unchanged_columns_exact": bool(
                original.drop(columns=sorted(AFFECTED_TICKERS)).equals(
                    corrected.drop(columns=sorted(AFFECTED_TICKERS))
                )
            ),
            "cutoffs": cutoff_audit.to_dict(orient="records"),
            "old_frozen_segment_preserved_except_declared_lineage_columns": True,
        },
    )
    schedule = BASE._write_schedule(paths, config, corrected)
    master = BASE.build_fixed_identity_master(paths, config, sources)
    return corrected, metadata, schedule, master


def _old_integration(config: Mapping[str, Any], date: pd.Timestamp) -> pd.DataFrame:
    token = date.strftime("%Y%m%d")
    candidates = [
        _source(config, "original_identity_root") / "data" / "processed" / "screens" / f"pair_v6_integration_{token}.csv",
        _source(config, "v6_screens_directory") / f"pair_v6_integration_{token}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path)
    raise FileNotFoundError(f"no integration diagnostic for {date.date()}")


def rebuild_screens(
    paths: Any,
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    prices: pd.DataFrame,
    price_meta: Mapping[str, Any],
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    old = pd.read_csv(_source(config, "original_screen_panel"))
    old["rebalance_date"] = pd.to_datetime(old["rebalance_date"])
    screen_cfg = protocol["screening"]
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for number, row in enumerate(schedule.to_dict("records"), start=1):
        date = pd.Timestamp(row["rebalance_date"])
        requested_start = date - pd.DateOffset(years=3)
        formation = prices.loc[(prices.index >= requested_start) & (prices.index < date)]
        ratio = float(screen_cfg["minimum_history_ratio"])
        settings = PairScreenSettings(
            minimum_observations=int(math.ceil(ratio * len(formation))),
            minimum_history_ratio=ratio,
            alpha=float(screen_cfg["cointegration_alpha"]),
            maximum_half_life_days=int(screen_cfg["maximum_half_life_days"]),
            adf_maxlag=int(screen_cfg["adf_maxlag"]),
            exclude_same_issuer=bool(screen_cfg["exclude_same_issuer"]),
            workers=1,
        )
        diagnostics = old.loc[old["rebalance_date"].eq(date)].copy()
        if len(diagnostics) != int(config["evaluation"]["expected_candidate_pairs_per_screen"]):
            raise RuntimeError(f"old screen row count mismatch at {date.date()}")
        integration = _old_integration(config, date)
        affected_integration = integration_diagnostics(
            formation.reindex(columns=sorted(AFFECTED_TICKERS)),
            maxlag=settings.adf_maxlag,
        )
        integration = integration.loc[~integration["ticker"].astype(str).isin(AFFECTED_TICKERS)]
        integration = pd.concat([integration, affected_integration], ignore_index=True).sort_values("ticker")
        i1_lookup = integration.set_index("ticker")["looks_i1"].to_dict()
        affected = diagnostics["ticker_a"].astype(str).isin(AFFECTED_TICKERS) | diagnostics["ticker_b"].astype(str).isin(AFFECTED_TICKERS)
        replacement_rows = []
        for item in diagnostics.loc[affected, ["ticker_a", "ticker_b", "sector"]].itertuples(index=False):
            replacement_rows.append(
                evaluate_pair(
                    str(item.ticker_a),
                    str(item.ticker_b),
                    str(item.sector),
                    formation,
                    len(formation),
                    settings,
                    i1_lookup=i1_lookup,
                )
            )
        replacement = pd.DataFrame(replacement_rows).set_index("pair")
        diagnostics = diagnostics.set_index("pair")
        core_columns = list(replacement.columns)
        diagnostics.loc[replacement.index, core_columns] = replacement[core_columns]
        diagnostics = diagnostics.reset_index()
        diagnostics["fdr_qvalue"] = np.nan
        for _, indices in diagnostics.groupby("sector", sort=False).groups.items():
            diagnostics.loc[indices, "fdr_qvalue"] = benjamini_hochberg(
                diagnostics.loc[indices, "pair_pvalue"]
            )
        diagnostics["fdr_significant"] = diagnostics["fdr_qvalue"] <= settings.alpha
        diagnostics["eligible"] = (
            diagnostics["screen_status"].eq("ok")
            & diagnostics["fdr_significant"]
            & diagnostics["positive_hedge_ratio"].astype(bool)
            & diagnostics["both_look_i1"].astype(bool)
            & diagnostics["half_life_days"].between(2.0, settings.maximum_half_life_days)
            & diagnostics["hurst_exponent"].lt(0.50)
        )
        diagnostics = diagnostics.sort_values(
            ["sector", "pair_pvalue", "half_life_days", "pair"],
            ascending=[True, True, True, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
        diagnostics["cointegration_rank_in_sector"] = (
            diagnostics.groupby("sector")["pair_pvalue"]
            .rank(method="first", ascending=True)
            .astype("Int64")
        )
        diagnostics["rebalance_date"] = date
        diagnostics["outer_end_exclusive"] = pd.Timestamp(row["outer_end_exclusive"])
        diagnostics["formation_start_requested"] = pd.Timestamp(row["formation_start_requested"])
        diagnostics["screen_source"] = "pit_lineage_affected_recompute_full_sector_bh"
        diagnostics["screen_cache_reused"] = False
        ordered = [
            "rebalance_date", "outer_end_exclusive", "formation_start_requested",
            *[c for c in diagnostics.columns if c not in {"rebalance_date", "outer_end_exclusive", "formation_start_requested"}],
        ]
        diagnostics = diagnostics[ordered]
        parts.append(diagnostics)
        screen_path, integration_path, meta_path = BASE._screen_cache_paths(paths, date)
        raw = diagnostics.drop(columns=["rebalance_date", "outer_end_exclusive", "formation_start_requested", "screen_source", "screen_cache_reused"])
        _csv(raw, screen_path)
        _csv(integration, integration_path)
        _json(
            meta_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "price_snapshot_sha256": price_meta["sha256"],
                "rebalance_date": date,
                "affected_rows_recomputed": int(affected.sum()),
                "all_sector_bh_qvalues_recomputed": True,
                "candidate_pairs": len(raw),
                "strict_eligible": int(raw["eligible"].sum()),
                "screen_sha256": _sha256(screen_path),
            },
        )
        audit_rows.append(
            {
                "rebalance_date": date,
                "affected_rows_recomputed": int(affected.sum()),
                "candidate_pairs": len(raw),
                "strict_eligible": int(raw["eligible"].sum()),
                "sw_observations": int(affected_integration.set_index("ticker").loc["SW", "observations"]),
                "ferg_observations": int(affected_integration.set_index("ticker").loc["FERG", "observations"]),
            }
        )
        print(f"PIT screen {number}/{len(schedule)} {date.date()} affected={int(affected.sum())}", flush=True)
    combined = pd.concat(parts, ignore_index=True)
    _csv(combined, paths.processed / BASE.SCREEN_NAME)
    _csv(pd.DataFrame(audit_rows), paths.processed / "pit_screen_rebuild_audit.csv")
    return combined


def _pair_is_affected(pair: str) -> bool:
    return bool(set(str(pair).split("__")) & AFFECTED_TICKERS)


def seed_verified_unaffected_caches(
    paths: Any,
    config: Mapping[str, Any],
    tasks: pd.DataFrame,
    old_prices: pd.DataFrame,
    new_prices: pd.DataFrame,
    *,
    price_sha256: str,
    protocol_sha256: str,
) -> pd.DataFrame:
    old_cache_root = _source(config, "original_pair_quarter_cache")
    rows: list[dict[str, Any]] = []
    checked_pairs: set[str] = set()
    for number, task in enumerate(tasks.to_dict("records"), start=1):
        pair = str(task["pair"])
        if _pair_is_affected(pair):
            continue
        a, b = str(task["ticker_a"]), str(task["ticker_b"])
        if pair not in checked_pairs:
            if not old_prices[[a, b]].equals(new_prices[[a, b]]):
                raise RuntimeError(f"unaffected pair price slice changed: {pair}")
            checked_pairs.add(pair)
        source = BASE._pair_cache_paths(old_cache_root, pair, pd.Timestamp(task["rebalance_date"]))
        target = BASE._pair_cache_paths(paths.pair_cache, pair, pd.Timestamp(task["rebalance_date"]))
        fingerprint = BASE._task_fingerprint(task, price_sha256, protocol_sha256)
        if BASE._cache_valid(target, fingerprint):
            marker = json.loads(target["complete"].read_text(encoding="utf-8"))
            source_complete_sha256 = marker.get("source_complete_sha256")
            if source_complete_sha256 is None and source["complete"].is_file():
                source_complete_sha256 = _sha256(source["complete"])
            source_summary_sha256 = marker.get("source_summary_sha256")
            if source_summary_sha256 is None and source["summary"].is_file():
                source_summary_sha256 = _sha256(source["summary"])
            rows.append(
                {
                    "pair": pair,
                    "rebalance_date": pd.Timestamp(task["rebalance_date"]),
                    "source_cache": str(source["directory"]),
                    "target_cache": str(target["directory"]),
                    "pair_price_columns_exact": True,
                    "source_complete_sha256": source_complete_sha256,
                    "source_summary_sha256": source_summary_sha256,
                    "new_fingerprint": fingerprint,
                    "seed_status": "target_cache_current",
                    "cache_ready_for_replay": True,
                }
            )
            continue
        BASE._clear_cache_artifacts(target)
        source_marker: dict[str, Any] = {}
        if source["complete"].is_file():
            try:
                source_marker = json.loads(source["complete"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                source_marker = {}
        source_complete = source["complete"].is_file() and source["summary"].is_file()
        source_lineage_current = BASE._cache_lineage_matches(source_marker)
        source_artifacts_valid = (
            source_complete
            and BASE._cache_artifacts_match_metadata(source, source_marker)
        )
        if not source_complete or not source_lineage_current or not source_artifacts_valid:
            rows.append(
                {
                    "pair": pair,
                    "rebalance_date": pd.Timestamp(task["rebalance_date"]),
                    "source_cache": str(source["directory"]),
                    "target_cache": str(target["directory"]),
                    "pair_price_columns_exact": True,
                    "source_complete_sha256": (
                        _sha256(source["complete"])
                        if source["complete"].is_file()
                        else None
                    ),
                    "source_summary_sha256": (
                        _sha256(source["summary"])
                        if source["summary"].is_file()
                        else None
                    ),
                    "new_fingerprint": fingerprint,
                    "seed_status": (
                        "rejected_source_engine_lineage"
                        if source_complete and not source_lineage_current
                        else (
                            "rejected_source_artifact_integrity"
                            if source_complete
                            else "rejected_source_cache_incomplete"
                        )
                    ),
                    "cache_ready_for_replay": False,
                    "source_pair_tuning_source_sha256": source_marker.get(
                        "pair_tuning_source_sha256"
                    ),
                    "required_pair_tuning_source_sha256": (
                        BASE.PAIR_TUNING_SOURCE_SHA256
                    ),
                }
            )
            continue
        if target["directory"].exists():
            shutil.rmtree(target["directory"])
        shutil.copytree(source["directory"], target["directory"])
        common = BASE._common_summary(task, new_prices)
        summary = pd.read_csv(target["summary"]).astype(object)
        for key, value in common.items():
            summary.loc[0, key] = value
        summary.loc[0, "replay_source"] = "pit_lineage_exact_pair_price_reuse"
        _csv(summary[BASE.SUMMARY_COLUMNS], target["summary"])
        if target["parameter"].is_file():
            parameter = pd.read_csv(target["parameter"])
            parameter.loc[:, "screen_dependent"] = common["screen_dependent"]
            parameter.loc[:, "screen_independent"] = common["screen_independent"]
            parameter.loc[:, "replay_source"] = "pit_lineage_exact_pair_price_reuse"
            _csv(parameter, target["parameter"])
        if target["failure"].is_file():
            failure = pd.read_csv(target["failure"])
            if len(failure):
                failure.loc[:, "history_available"] = common["history_available"]
                failure.loc[:, "orientation_available"] = common["orientation_available"]
                _csv(failure, target["failure"])
        marker = json.loads(source["complete"].read_text(encoding="utf-8"))
        marker.update(
            {
                "schema_version": BASE.EXECUTION_CACHE_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "seeded_at_utc": datetime.now(timezone.utc).isoformat(),
                **BASE._execution_cache_lineage(),
                "reuse_basis": "exact equality of both pair price columns over the full snapshot",
                "source_complete_sha256": _sha256(source["complete"]),
                "source_summary_sha256": _sha256(source["summary"]),
                "strategy_recomputed": False,
                "artifact_sha256": BASE._cache_artifact_hashes(target),
            }
        )
        _json(target["complete"], marker)
        rows.append(
            {
                "pair": pair,
                "rebalance_date": pd.Timestamp(task["rebalance_date"]),
                "source_cache": str(source["directory"]),
                "target_cache": str(target["directory"]),
                "pair_price_columns_exact": True,
                "source_complete_sha256": marker["source_complete_sha256"],
                "source_summary_sha256": marker["source_summary_sha256"],
                "new_fingerprint": fingerprint,
                "seed_status": "seeded_from_compatible_source",
                "cache_ready_for_replay": True,
            }
        )
        if number % 500 == 0:
            print(f"verified cache seeding {number}/{len(tasks)}", flush=True)
    audit = pd.DataFrame(rows)
    _csv(audit, paths.processed / "pit_unaffected_cache_reuse_audit.csv.gz")
    return audit


def run(
    config_path: Path,
    *,
    stage: str,
    tuning_workers: int,
) -> dict[str, Any]:
    config_path = config_path.resolve() if config_path.is_absolute() else (PROJECT_ROOT / config_path).resolve()
    config = BASE._load_config(config_path)
    paths = BASE.EvaluationPaths.create((PROJECT_ROOT / config["outputs"]["default_root"]).resolve())
    sources = BASE._source_paths(config, PROJECT_ROOT)
    v6 = BASE._load_module(sources["v6_runner"], "pair_identity_pit_v6_immutable_runner")
    protocol, spec, base_config, immutable_hashes = BASE._validate_immutable_protocol(config, sources, paths.root, v6)
    if stage in {"prepare", "all"}:
        prices, price_meta, schedule, master = prepare(paths, config, sources)
    else:
        prices, price_meta = BASE.load_prepared_prices(paths, config)
        schedule = BASE._read_csv(paths.processed / BASE.SCHEDULE_NAME, dates=("rebalance_date", "outer_end_exclusive", "formation_start_requested"))
        master = BASE.build_fixed_identity_master(paths, config, sources)
    if stage == "prepare":
        return {"stage": stage, "price_sha256": price_meta["sha256"], "pairs": len(master), "quarters": len(schedule)}
    if stage in {"screens", "all"}:
        screens = rebuild_screens(paths, config, protocol, prices, price_meta, schedule)
    else:
        screens = BASE._read_csv(paths.processed / BASE.SCREEN_NAME, dates=("rebalance_date", "outer_end_exclusive", "formation_start_requested"))
    if stage == "screens":
        return {"stage": stage, "screen_rows": len(screens)}
    tasks = BASE.build_pair_quarter_tasks(master, schedule, screens)
    old_prices, _ = load_timeseries_snapshot(_source(config, "original_spliced_price_snapshot"))
    reuse_audit = seed_verified_unaffected_caches(
        paths,
        config,
        tasks,
        old_prices,
        prices,
        price_sha256=str(price_meta["sha256"]),
        protocol_sha256=immutable_hashes["v6_config"],
    )
    ready_reuse_count = int(
        reuse_audit.get(
            "cache_ready_for_replay",
            pd.Series(dtype=bool),
        ).fillna(False).astype(bool).sum()
    )
    rejected_seed_count = int(len(reuse_audit) - ready_reuse_count)
    replay_counts: dict[str, Any] = {}
    if stage in {"replay", "all"}:
        replay_counts = BASE.run_pair_quarter_replays(
            paths,
            tasks,
            sources,
            prices,
            spec,
            base_config,
            price_sha256=str(price_meta["sha256"]),
            protocol_sha256=immutable_hashes["v6_config"],
            tuning_workers=tuning_workers,
            recompute=False,
            use_predictive_seed=False,
            maximum_new_tasks=None,
        )
    if stage == "replay":
        return {
            "stage": stage,
            "verified_reused": ready_reuse_count,
            "rejected_stale_seeds": rejected_seed_count,
            **replay_counts,
        }
    post_hashes = BASE._post_freeze_source_hashes(sources)
    BASE._validate_configured_post_freeze_hashes(config, post_hashes)
    audit = BASE.collect_replay_outputs(
        paths,
        config,
        master,
        schedule,
        tasks,
        price_sha256=str(price_meta["sha256"]),
        protocol_sha256=immutable_hashes["v6_config"],
        immutable_hashes=immutable_hashes,
        post_freeze_hashes_before=post_hashes,
        sources=sources,
        allow_incomplete=False,
    )
    lineage_audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "critical_pass": bool(audit["critical_pass"]),
        "fixed_pairs": len(master),
        "pair_quarters": len(tasks),
        "affected_identities": int(master["pair"].map(_pair_is_affected).sum()),
        "affected_pair_quarters": int(tasks["pair"].map(_pair_is_affected).sum()),
        "verified_reused_pair_quarters": ready_reuse_count,
        "rejected_stale_seed_pair_quarters": rejected_seed_count,
        **BASE._execution_cache_lineage(),
        "changed_price_columns": sorted(AFFECTED_TICKERS),
        "cross_venue_splice_used": False,
        "strategy_logic_changed": bool(
            config["protocol"].get("strategy_mutation_allowed", False)
        ),
        "strategy_mutation_scope": config["protocol"].get(
            "strategy_mutation_scope", "none"
        ),
        "price_snapshot_sha256": price_meta["sha256"],
        **replay_counts,
    }
    _json(paths.root / "pit_lineage_rerun_audit.json", lineage_audit)
    return {"stage": stage, "output_root": str(paths.root), **lineage_audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("prepare", "screens", "replay", "assemble", "all"), default="all")
    parser.add_argument("--tuning-workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.config, stage=args.stage, tuning_workers=args.tuning_workers), indent=2, default=str))


if __name__ == "__main__":
    main()
