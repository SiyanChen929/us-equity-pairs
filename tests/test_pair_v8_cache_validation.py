from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pair_v8_full_panel.py"
)
SPEC = importlib.util.spec_from_file_location("run_pair_v8_full_panel_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
v8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v8)


def _write_provisional_cache(root: Path) -> tuple[Path, Path]:
    artifact = root / "candidate_metrics.csv"
    pd.DataFrame(
        {
            "pair": ["A__B"],
            "rebalance_date": ["2024-01-02"],
            "strategy_id": ["S0_v6"],
        }
    ).to_csv(artifact, index=False)
    marker_path = root / "complete.json"
    marker = v8._cache_marker_payload(
        fingerprint="fingerprint",
        artifact=artifact,
        artifact_rows=1,
        pair="A__B",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
        cache_stage="candidate",
        run_nonce="run-a",
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    return artifact, marker_path


def _matches(marker_path: Path, artifact: Path) -> bool:
    return v8._cache_marker_matches(
        marker_path,
        artifact,
        fingerprint="fingerprint",
        pair="A__B",
        rebalance_date="2024-01-02",
        payload_sha256="payload",
        artifact_date_column="rebalance_date",
    )


def test_provisional_cache_is_rejected_until_unchanged_input_stage_promotes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    artifact, marker_path = _write_provisional_cache(tmp_path)
    before = {str(source): v8._sha256(source)}
    monkeypatch.setattr(v8, "_engine_source_manifest", lambda: {})

    assert not _matches(marker_path, artifact)
    audit = v8._validate_input_stage_and_promote_cache(
        stage="candidate",
        results=[
            {
                "cached": False,
                "path": str(artifact),
                "marker_path": str(marker_path),
                "cache_stage": "candidate",
                "run_nonce": "run-a",
                "fingerprint": "fingerprint",
                "payload_sha256": "payload",
                "pair": "A__B",
                "rebalance_date": "2024-01-02",
                "artifact_date_column": "rebalance_date",
            }
        ],
        file_hashes_before=before,
        engine_manifest_before={},
        s5_hashes_before={},
        audit_path=tmp_path / "candidate_input_audit.csv.gz",
    )

    assert audit["unchanged"].all()
    assert _matches(marker_path, artifact)
    promoted = json.loads(marker_path.read_text(encoding="utf-8"))
    assert promoted["input_stage_validated"] is True
    assert promoted["input_validation_stage"] == "candidate"


def test_input_mutation_fails_before_cache_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    artifact, marker_path = _write_provisional_cache(tmp_path)
    before = {str(source): v8._sha256(source)}
    source.write_text("value\n2\n", encoding="utf-8")
    monkeypatch.setattr(v8, "_engine_source_manifest", lambda: {})

    with pytest.raises(RuntimeError, match="changed during candidate stage"):
        v8._validate_input_stage_and_promote_cache(
            stage="candidate",
            results=[
                {
                    "cached": False,
                    "path": str(artifact),
                    "marker_path": str(marker_path),
                    "cache_stage": "candidate",
                    "run_nonce": "run-a",
                    "fingerprint": "fingerprint",
                    "payload_sha256": "payload",
                    "pair": "A__B",
                    "rebalance_date": "2024-01-02",
                    "artifact_date_column": "rebalance_date",
                }
            ],
            file_hashes_before=before,
            engine_manifest_before={},
            s5_hashes_before={},
            audit_path=tmp_path / "candidate_input_audit.csv.gz",
        )

    assert not _matches(marker_path, artifact)
    assert json.loads(marker_path.read_text(encoding="utf-8"))[
        "input_stage_validated"
    ] is False


def test_cache_promotion_rejects_a_marker_from_another_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    artifact, marker_path = _write_provisional_cache(tmp_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["run_nonce"] = "run-b"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(v8, "_engine_source_manifest", lambda: {})

    with pytest.raises(RuntimeError, match="identity mismatch"):
        v8._validate_input_stage_and_promote_cache(
            stage="candidate",
            results=[
                {
                    "cached": False,
                    "path": str(artifact),
                    "marker_path": str(marker_path),
                    "cache_stage": "candidate",
                    "run_nonce": "run-a",
                    "fingerprint": "fingerprint",
                    "payload_sha256": "payload",
                    "pair": "A__B",
                    "rebalance_date": "2024-01-02",
                    "artifact_date_column": "rebalance_date",
                }
            ],
            file_hashes_before={str(source): v8._sha256(source)},
            engine_manifest_before={},
            s5_hashes_before={},
            audit_path=tmp_path / "candidate_input_audit.csv.gz",
        )

    assert not _matches(marker_path, artifact)


def test_batch_validation_failure_does_not_partially_promote_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_artifact, first_marker = _write_provisional_cache(first_root)
    second_artifact, second_marker = _write_provisional_cache(second_root)
    broken = json.loads(second_marker.read_text(encoding="utf-8"))
    broken["artifact_sha256"] = "0" * 64
    second_marker.write_text(json.dumps(broken), encoding="utf-8")
    monkeypatch.setattr(v8, "_engine_source_manifest", lambda: {})

    def result(artifact: Path, marker: Path) -> dict[str, object]:
        return {
            "cached": False,
            "path": str(artifact),
            "marker_path": str(marker),
            "cache_stage": "candidate",
            "run_nonce": "run-a",
            "fingerprint": "fingerprint",
            "payload_sha256": "payload",
            "pair": "A__B",
            "rebalance_date": "2024-01-02",
            "artifact_date_column": "rebalance_date",
        }

    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        v8._validate_input_stage_and_promote_cache(
            stage="candidate",
            results=[
                result(first_artifact, first_marker),
                result(second_artifact, second_marker),
            ],
            file_hashes_before={str(source): v8._sha256(source)},
            engine_manifest_before={},
            s5_hashes_before={},
            audit_path=tmp_path / "candidate_input_audit.csv.gz",
        )

    assert json.loads(first_marker.read_text(encoding="utf-8"))[
        "input_stage_validated"
    ] is False


def test_worker_reads_hash_verified_price_and_s5_snapshots(tmp_path: Path) -> None:
    price_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"date": ["2024-01-02"], "A": [100.0], "B": [101.0]}
    ).to_csv(price_path, index=False)
    cache = tmp_path / "v6"
    ranking = (
        cache / "a__b" / "20240102" / "parameter_rankings.csv.gz"
    )
    ranking.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "config_id": ["cfg"],
            "eligible": [True],
            "zscore_method": ["sma"],
            "zscore_lookback": [20],
            "zscore_min_periods": [16],
            "entry_z": [2.0],
            "exit_z": [0.5],
            "maximum_holding_days": [30],
        }
    ).to_csv(ranking, index=False, compression="gzip")
    expected_s5 = {("A__B", "2024-01-02"): v8._sha256(ranking)}
    base = v8.StrategyConfig().__dict__

    v8._worker_init(
        str(price_path),
        base,
        [],
        str(cache),
        v8._sha256(price_path),
        expected_s5,
    )
    assert len(v8._ensemble_parameters(
        {"pair": "A__B", "rebalance_date": "2024-01-02"}
    )) == 1

    pd.DataFrame(
        {
            "config_id": ["changed"],
            "eligible": [True],
            "zscore_method": ["sma"],
            "zscore_lookback": [20],
            "zscore_min_periods": [16],
            "entry_z": [2.0],
            "exit_z": [0.5],
            "maximum_holding_days": [30],
        }
    ).to_csv(ranking, index=False, compression="gzip")
    with pytest.raises(RuntimeError, match="S5 ranking hash mismatch"):
        v8._ensemble_parameters(
            {"pair": "A__B", "rebalance_date": "2024-01-02"}
        )

    with pytest.raises(RuntimeError, match="price snapshot hash mismatch"):
        v8._worker_init(
            str(price_path),
            base,
            [],
            str(cache),
            "0" * 64,
            expected_s5,
        )
    with pytest.raises(RuntimeError, match="runtime manifest mismatch"):
        v8._worker_init(
            str(price_path),
            base,
            [],
            str(cache),
            v8._sha256(price_path),
            expected_s5,
            "0" * 64,
        )


def test_candidate_cache_rejects_nonfinite_available_economics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v8, "_CANDIDATES", [{"id": "S0_v6"}])
    row = {"pair": "A__B", "rebalance_date": "2024-01-02"}
    frame = pd.DataFrame(
        {
            "pair": ["A__B"],
            "rebalance_date": ["2024-01-02"],
            "strategy_id": ["S0_v6"],
            "q_return_1bp": [0.01],
            "q_return_2bp": [float("nan")],
            "trade_count": [1],
            "selection_activity_count": [1],
            "selection_activity_count_definition": [
                "continuous_active_target_runs"
            ],
            "annualized_turnover": [2.0],
            "max_drawdown": [-0.02],
            "configuration_count": [1],
            "candidate_available": [True],
        }
    )

    assert not v8._candidate_frame_matches(frame, row)


def test_snapshot_parser_keeps_the_hashed_bytes_after_live_file_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    raw, sha256 = v8._snapshot_bytes(source)
    source.write_text("value\n2\n", encoding="utf-8")

    parsed = v8._read_csv_snapshot(raw, source)

    assert sha256 != v8._sha256(source)
    assert parsed["value"].tolist() == [1]


def test_output_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with v8._exclusive_output_run_lock(tmp_path, "run-a"):
        with pytest.raises(RuntimeError, match="another V8 run"):
            with v8._exclusive_output_run_lock(tmp_path, "run-b"):
                pass
    with v8._exclusive_output_run_lock(tmp_path, "run-c"):
        pass


def test_candidate_worker_identity_promotes_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    daily = pd.DataFrame(
        {
            "gross_return": [0.0, 0.01],
            "turnover": [1.0, 0.0],
            "borrow_cost": [0.0, 0.0],
        },
        index=dates,
    )
    candidate = {"id": "S0_v6", "complexity": 0}
    monkeypatch.setattr(v8, "_CANDIDATES", [candidate])
    monkeypatch.setattr(v8, "_indexes", lambda _row: (dates, dates))
    monkeypatch.setattr(
        v8,
        "_simulate_candidate",
        lambda *_args, **_kwargs: (
            daily,
            {"trade_count": 1, "configuration_count": 1},
        ),
    )
    monkeypatch.setattr(v8, "_engine_source_manifest", lambda: {})
    row = {
        "pair": "A__B",
        "sector": "Test",
        "rebalance_date": dates[0],
        "outer_end_exclusive": dates[-1] + pd.Timedelta(days=1),
        "screen_half_life_days": 5.0,
        "selected_config_id": "cfg",
    }
    payload_sha256 = v8._canonical_payload_sha256(row)
    result = v8._candidate_task(
        (
            row,
            str(tmp_path / "cache"),
            {
                "fingerprint": "fingerprint",
                "payload_sha256": payload_sha256,
                "run_nonce": "run-a",
                "row_payload_sha256": payload_sha256,
                "s5_parameter_rankings_sha256": "NOT_APPLICABLE",
            },
        )
    )
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")

    v8._validate_input_stage_and_promote_cache(
        stage="candidate",
        results=[result],
        file_hashes_before={str(source): v8._sha256(source)},
        engine_manifest_before={},
        s5_hashes_before={},
        audit_path=tmp_path / "candidate_input_audit.csv.gz",
    )

    marker = json.loads(Path(result["marker_path"]).read_text(encoding="utf-8"))
    assert marker["input_stage_validated"] is True
    assert marker["run_nonce"] == "run-a"
    assert len(result["_validated_frame"]) == 1
