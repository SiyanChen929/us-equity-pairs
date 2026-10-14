from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


identity = _load_script(
    "run_pair_identity_10y_evaluation.py",
    "run_pair_identity_10y_evaluation_cache_test",
)
pit = _load_script(
    "run_pair_identity_pit_lineage.py",
    "run_pair_identity_pit_lineage_cache_test",
)


def _task() -> dict[str, object]:
    return {
        "pair": "A__B",
        "ticker_a": "A",
        "ticker_b": "B",
        "sector": "Technology",
        "rebalance_date": pd.Timestamp("2024-01-02"),
        "outer_end_exclusive": pd.Timestamp("2024-04-01"),
        "screen_status": "ok",
        "dependent": "A",
        "independent": "B",
    }


def _collection_fixture(tmp_path, *, status: str = "tuning_failed_cash"):
    task = _task()
    prices = pd.DataFrame(
        {"A": 100.0, "B": 90.0},
        index=pd.bdate_range(end="2024-03-29", periods=560),
    )
    common = identity._common_summary(task, prices)
    summary = identity._empty_summary(
        common,
        status=status,
        cash_reason="expected_test_cash",
    )
    failure = identity._failure_row(
        common,
        stage="pair_tuning",
        status=status,
        cash_reason="expected_test_cash",
    )
    paths = identity.EvaluationPaths.create(tmp_path / "study")
    cache = identity._pair_cache_paths(
        paths.pair_cache,
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    identity._write_cache(cache, fingerprint, summary, failure=failure)
    master = pd.DataFrame(
        [
            {
                "pair": task["pair"],
                "ever_v6_selected": False,
                "v6_selected_vintage_count": 0,
            }
        ]
    )
    schedule = pd.DataFrame([{"rebalance_date": task["rebalance_date"]}])
    config = {
        "evaluation": {
            "expected_fixed_identity_count": 1,
            "expected_quarter_intervals": 1,
        }
    }
    return paths, task, cache, master, schedule, config


def _collect_fixture(
    paths,
    task,
    master,
    schedule,
    config,
    monkeypatch,
    *,
    allow_incomplete: bool,
):
    monkeypatch.setattr(
        identity,
        "_post_freeze_source_hashes",
        lambda _sources: {"sentinel": "unchanged"},
    )
    return identity.collect_replay_outputs(
        paths,
        config,
        master,
        schedule,
        pd.DataFrame([task]),
        price_sha256="price-hash",
        protocol_sha256="protocol-hash",
        immutable_hashes={},
        post_freeze_hashes_before={"sentinel": "unchanged"},
        sources={},
        allow_incomplete=allow_incomplete,
    )


def test_pair_quarter_fingerprint_and_metadata_bind_the_execution_engine(
    tmp_path, monkeypatch
):
    task = _task()
    monkeypatch.setattr(
        identity,
        "PAIR_TUNING_SOURCE_SHA256",
        "a" * 64,
        raising=False,
    )
    first = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    monkeypatch.setattr(
        identity,
        "PAIR_TUNING_SOURCE_SHA256",
        "b" * 64,
        raising=False,
    )
    second = identity._task_fingerprint(task, "price-hash", "protocol-hash")

    assert first != second
    cache = identity._pair_cache_paths(
        tmp_path / "cache",
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    identity._write_cache(
        cache,
        second,
        pd.DataFrame([{"status": "tuning_failed_cash"}]),
        failure=pd.DataFrame([{"error": "expected test failure"}]),
    )
    metadata = json.loads(cache["complete"].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == identity.EXECUTION_CACHE_SCHEMA_VERSION == 3
    assert metadata["pair_tuning_source_sha256"] == "b" * 64
    assert metadata["execution_convention"] == identity.NEXT_CLOSE_EXECUTION_CONVENTION
    assert set(metadata["execution_engine_source_sha256"]) == set(
        identity.EXECUTION_ENGINE_SOURCES
    )
    assert set(metadata["artifact_sha256"]) == {"summary", "failure"}


def test_pair_quarter_cache_rejects_one_changed_engine_dependency(
    tmp_path, monkeypatch
):
    task = _task()
    original_fingerprint = identity._task_fingerprint(
        task, "price-hash", "protocol-hash"
    )
    cache = identity._pair_cache_paths(
        tmp_path / "cache",
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    identity._write_cache(
        cache,
        original_fingerprint,
        pd.DataFrame([{"status": "tuning_failed_cash"}]),
        failure=pd.DataFrame([{"error": "expected test failure"}]),
    )
    changed = dict(identity.EXECUTION_ENGINE_SOURCE_SHA256)
    changed["src/equity_pairs/grid_search.py"] = "c" * 64
    monkeypatch.setattr(identity, "EXECUTION_ENGINE_SOURCE_SHA256", changed)

    changed_fingerprint = identity._task_fingerprint(
        task, "price-hash", "protocol-hash"
    )

    assert changed_fingerprint != original_fingerprint
    assert not identity._cache_valid(cache, changed_fingerprint)


def test_pair_quarter_failure_write_removes_stale_success_artifacts(tmp_path):
    cache = identity._pair_cache_paths(
        tmp_path / "cache",
        "A__B",
        pd.Timestamp("2024-01-02"),
    )
    cache["directory"].mkdir(parents=True)
    for key in ("daily", "trades", "parameter", "ranking"):
        cache[key].write_text("stale-success-artifact\n", encoding="utf-8")

    identity._write_cache(
        cache,
        "new-fingerprint",
        pd.DataFrame([{"status": "tuning_failed_cash"}]),
        failure=pd.DataFrame([{"error": "expected test failure"}]),
    )

    assert cache["failure"].is_file()
    for key in ("daily", "trades", "parameter", "ranking"):
        assert not cache[key].exists()


def test_pair_quarter_cache_rejects_tampered_artifact(tmp_path):
    task = _task()
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    cache = identity._pair_cache_paths(
        tmp_path / "cache",
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    identity._write_cache(
        cache,
        fingerprint,
        pd.DataFrame([{"status": "tuning_failed_cash"}]),
        failure=pd.DataFrame([{"error": "expected test failure"}]),
    )
    cache["summary"].write_text(
        "status\ntampered_after_completion\n", encoding="utf-8"
    )

    assert not identity._cache_valid(cache, fingerprint)


def test_pair_quarter_cache_rejects_tampered_daily_artifact(tmp_path):
    task = _task()
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    cache = identity._pair_cache_paths(
        tmp_path / "cache",
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    identity._write_cache(
        cache,
        fingerprint,
        pd.DataFrame([{"status": "evaluated_zero_trade"}]),
        daily=pd.DataFrame([{"net_return_2bp": 0.0}]),
        trades=pd.DataFrame(columns=["trade_number"]),
        parameter=pd.DataFrame([{"config_id": "baseline"}]),
        ranking=pd.DataFrame([{"config_id": "baseline", "selected": True}]),
    )
    cache["daily"].write_bytes(cache["daily"].read_bytes() + b"tampered")

    assert not identity._cache_valid(cache, fingerprint)


def test_predictive_seed_skips_source_without_current_engine_lineage(
    tmp_path, monkeypatch
):
    manifest = tmp_path / "predictive_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("stale predictive tables must not be read")

    monkeypatch.setattr(identity, "_read_csv", unexpected_read)
    seeded = identity.seed_predictive_replays(
        pd.DataFrame(),
        {"predictive_manifest": manifest},
        tmp_path / "cache",
        pd.DataFrame(),
        None,
        None,
        price_sha256="price-hash",
        protocol_sha256="protocol-hash",
        recompute=False,
    )

    assert seeded == 0


def test_pit_seed_rejects_source_cache_without_current_engine_lineage(tmp_path):
    task = _task()
    tasks = pd.DataFrame([task])
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [90.0] * 3}, index=dates)
    old_cache_root = tmp_path / "old-cache"
    source = pit.BASE._pair_cache_paths(
        old_cache_root,
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    source["directory"].mkdir(parents=True)
    pd.DataFrame([{"status": "evaluated"}]).to_csv(source["summary"], index=False)
    source["complete"].write_text(
        json.dumps({"fingerprint": "legacy-engine-fingerprint"}),
        encoding="utf-8",
    )
    paths = pit.BASE.EvaluationPaths.create(tmp_path / "new-study")
    config = {
        "immutable_sources": {
            "original_pair_quarter_cache": str(old_cache_root),
        }
    }

    audit = pit.seed_verified_unaffected_caches(
        paths,
        config,
        tasks,
        prices,
        prices.copy(),
        price_sha256="new-price-hash",
        protocol_sha256="new-protocol-hash",
    )

    target = pit.BASE._pair_cache_paths(
        paths.pair_cache,
        str(task["pair"]),
        pd.Timestamp(task["rebalance_date"]),
    )
    assert audit.loc[0, "seed_status"] == "rejected_source_engine_lineage"
    assert bool(audit.loc[0, "cache_ready_for_replay"]) is False
    assert not target["directory"].exists()


def test_assemble_rejects_stale_pair_quarter_fingerprint(tmp_path, monkeypatch):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    marker = json.loads(cache["complete"].read_text(encoding="utf-8"))
    marker["fingerprint"] = "stale-fingerprint"
    cache["complete"].write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid|stale"):
        _collect_fixture(
            paths,
            task,
            master,
            schedule,
            config,
            monkeypatch,
            allow_incomplete=False,
        )


def test_assemble_rejects_marker_artifact_row_count_mismatch(tmp_path, monkeypatch):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    marker = json.loads(cache["complete"].read_text(encoding="utf-8"))
    marker["daily_rows"] = 1
    cache["complete"].write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="daily_row_count_mismatch"):
        _collect_fixture(
            paths,
            task,
            master,
            schedule,
            config,
            monkeypatch,
            allow_incomplete=False,
        )


def test_assemble_rejects_summary_key_mismatch_with_valid_content_hashes(
    tmp_path, monkeypatch
):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    summary = pd.read_csv(cache["summary"])
    failure = pd.read_csv(cache["failure"])
    summary.loc[0, "pair"] = "WRONG__PAIR"
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    identity._write_cache(cache, fingerprint, summary, failure=failure)

    with pytest.raises(RuntimeError, match="summary_key_mismatch"):
        _collect_fixture(
            paths,
            task,
            master,
            schedule,
            config,
            monkeypatch,
            allow_incomplete=False,
        )


def test_assemble_rejects_success_cache_without_required_artifacts(
    tmp_path, monkeypatch
):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    summary = pd.read_csv(cache["summary"])
    summary.loc[0, "status"] = "evaluated_zero_trade"
    summary.loc[0, "tuning_success"] = True
    summary.loc[0, "active_identity_conditional"] = True
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    identity._write_cache(cache, fingerprint, summary)

    with pytest.raises(RuntimeError, match="successful_cache_missing_daily"):
        _collect_fixture(
            paths,
            task,
            master,
            schedule,
            config,
            monkeypatch,
            allow_incomplete=False,
        )


def test_assemble_allow_incomplete_skips_and_audits_invalid_cache(
    tmp_path, monkeypatch
):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    marker = json.loads(cache["complete"].read_text(encoding="utf-8"))
    marker["fingerprint"] = "stale-fingerprint"
    cache["complete"].write_text(json.dumps(marker), encoding="utf-8")

    audit = _collect_fixture(
        paths,
        task,
        master,
        schedule,
        config,
        monkeypatch,
        allow_incomplete=True,
    )

    assert audit["actual_quarterly_rows"] == 0
    assert audit["missing_pair_quarter_cache_count"] == 0
    assert audit["invalid_pair_quarter_cache_count"] == 1
    assert audit["critical_pass"] is False
    assert audit["invalid_pair_quarter_keys_preview"][0]["pair"] == task["pair"]


def test_assemble_accepts_valid_tuning_failure_cache(tmp_path, monkeypatch):
    paths, task, _cache, master, schedule, config = _collection_fixture(tmp_path)

    audit = _collect_fixture(
        paths,
        task,
        master,
        schedule,
        config,
        monkeypatch,
        allow_incomplete=False,
    )

    assert audit["actual_quarterly_rows"] == 1
    assert audit["invalid_pair_quarter_cache_count"] == 0
    assert audit["critical_pass"] is True


def test_assemble_audit_fails_closed_on_outer_evaluation_failure(
    tmp_path, monkeypatch
):
    paths, task, cache, master, schedule, config = _collection_fixture(tmp_path)
    summary = pd.read_csv(cache["summary"])
    failure = pd.read_csv(cache["failure"])
    summary.loc[0, "status"] = "evaluation_failed_cash"
    summary.loc[0, "tuning_success"] = True
    failure.loc[0, "status"] = "evaluation_failed_cash"
    failure.loc[0, "stage"] = "outer_evaluation"
    keyed = {
        "pair": task["pair"],
        "rebalance_date": task["rebalance_date"],
    }
    fingerprint = identity._task_fingerprint(task, "price-hash", "protocol-hash")
    identity._write_cache(
        cache,
        fingerprint,
        summary,
        parameter=pd.DataFrame([{**keyed, "selected_config_id": "baseline"}]),
        ranking=pd.DataFrame([{**keyed, "config_id": "baseline", "selected": True}]),
        failure=failure,
    )

    audit = _collect_fixture(
        paths,
        task,
        master,
        schedule,
        config,
        monkeypatch,
        allow_incomplete=False,
    )

    assert audit["evaluation_failed_cash_pair_quarters"] == 1
    assert audit["status_count_reconciles"] is True
    assert audit["critical_pass"] is False
