from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_pairs.config import StrategyConfig
from equity_pairs.pair_tuning import PairTuningSpec, protocol_parameter_sets
from equity_pairs.statistics import PairScreenSettings


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pair_optimized_10y_research.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_pair_optimized_10y_research_test", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
runner = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = runner
MODULE_SPEC.loader.exec_module(runner)


def _protocol() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "configs" / "pair_optimized_10y_3y.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_prices(periods: int = 280) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    time = np.arange(periods, dtype=float)
    common_log = (
        0.0007 * time
        + 0.10 * np.sin(time / 13.0)
        + 0.05 * np.cos(time / 5.0)
    )
    spread = 0.075 * np.sin(time / 6.0) + 0.012 * np.sin(time / 17.0)
    spread_cd = 0.060 * np.sin(time / 8.0 + 0.6)
    return pd.DataFrame(
        {
            "B": 100.0 * np.exp(common_log),
            "A": 100.0 * np.exp(common_log + spread),
            "D": 82.0 * np.exp(common_log),
            "C": 82.0 * np.exp(common_log + spread_cd),
        },
        index=dates,
    )


def _strict_row(date: pd.Timestamp, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "rebalance_date": date,
        "pair": "A__B",
        "sector": "Technology",
        "dependent": "A",
        "independent": "B",
        "screen_status": "ok",
        "eligible": True,
        "tradable_eligible": True,
        "fdr_significant": True,
        "fdr_qvalue": 0.01,
        "pair_pvalue": 0.005,
        "positive_hedge_ratio": True,
        "both_look_i1": True,
        "half_life_days": 20.0,
        "hurst_exponent": 0.35,
        "selection_role": "strict_eligible",
    }
    row.update(updates)
    return row


def _small_tuning_spec() -> PairTuningSpec:
    return PairTuningSpec(
        zscore_methods=("sma",),
        zscore_lookbacks=(10,),
        entry_zs=(1.0, 1.5),
        exit_zs=(0.25,),
        maximum_holding_days=(12,),
        inner_train_size=80,
        inner_validation_size=40,
        inner_step_size=40,
        purge_size=1,
        embargo_size=0,
        maximum_inner_folds=3,
        primary_transaction_cost_bps=2.0,
        sensitivity_transaction_cost_bps=(1.0, 2.0),
        minimum_full_training_trades=0,
        minimum_validation_trades=1,
        minimum_active_folds=1,
        minimum_hedge_observations=50,
    )


def _base_config() -> StrategyConfig:
    return StrategyConfig(
        zscore_method="sma",
        zscore_lookback=60,
        zscore_min_periods=40,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        maximum_holding_days=60,
        transaction_cost_bps=2.0,
        transaction_cost_scenarios_bps=(1.0, 2.0),
        annual_short_borrow_bps=30.0,
    )


def test_frozen_protocol_maps_to_exactly_97_pair_candidates():
    protocol = runner._load_protocol(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pair_optimized_10y_3y.json"
    )

    spec, base = runner._build_pair_tuning_protocol(protocol)

    assert len(protocol_parameter_sets(spec, stop_z=base.stop_z)) == 97
    assert spec.primary_transaction_cost_bps == 2.0
    assert spec.sensitivity_transaction_cost_bps == (1.0, 2.0)
    assert spec.inner_train_size == 378
    assert spec.purge_size == 1
    assert base.annual_short_borrow_bps == 30.0


def test_truncated_debug_schedule_keeps_the_true_next_quarter_boundary():
    dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-04-01", "2024-07-01", "2024-10-01"]
    )

    schedule = runner._rebalance_schedule(
        dates,
        as_of=pd.Timestamp("2024-11-20"),
        maximum_rebalances=1,
    )

    assert schedule["rebalance_date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert schedule["outer_end_exclusive"].tolist() == [pd.Timestamp("2024-04-01")]
    assert not schedule["outer_window_truncated_by_asof"].iloc[0]


def test_strict_filter_rejects_quota_fills_and_failed_fdr_rows():
    date = pd.Timestamp("2025-01-02")
    candidates = pd.DataFrame(
        [
            _strict_row(date),
            _strict_row(date, pair="C__D", fdr_qvalue=0.08),
            _strict_row(
                date,
                pair="E__F",
                tradable_eligible=False,
                selection_role="ineligible_quota_fill",
            ),
            _strict_row(date, pair="G__H", hurst_exponent=0.55),
        ]
    )

    strict = runner._strict_candidates_for_vintage(candidates, date)

    assert strict["pair"].tolist() == ["A__B"]
    assert strict["tradable_eligible"].all()
    assert strict["selection_role"].eq("strict_eligible").all()


def test_legacy_screen_cache_is_enriched_without_rescreening(tmp_path, monkeypatch):
    paths = runner.StudyPaths.create(tmp_path / "study")
    dates = pd.bdate_range("2018-01-02", periods=760)
    prices = pd.DataFrame(
        {"A": np.linspace(90, 120, len(dates)), "B": np.linspace(80, 110, len(dates))},
        index=dates,
    )
    universe = pd.DataFrame(
        {"ticker": ["A", "B"], "sector": ["Technology", "Technology"]}
    )
    universe.to_csv(paths.raw / "sp500_current_constituents.csv", index=False)
    rebalance = dates[-1] + pd.offsets.BDay(1)
    formation = prices.loc[
        (prices.index >= rebalance - pd.DateOffset(years=3))
        & (prices.index < rebalance)
    ]
    settings = PairScreenSettings(
        minimum_observations=math.ceil(0.9 * len(formation)),
        minimum_history_ratio=0.9,
        alpha=0.05,
        maximum_half_life_days=252,
        adf_maxlag=5,
        exclude_same_issuer=True,
        workers=1,
    )
    payload = runner._screen_payload(
        rebalance_date=rebalance,
        requested_start=rebalance - pd.DateOffset(years=3),
        formation=formation,
        settings=settings,
        universe_sha256=runner._sha256(
            paths.raw / "sp500_current_constituents.csv"
        ),
        price_snapshot_sha256="price-hash",
    )
    diagnostics = pd.DataFrame(
        {
            "pair": ["A__B"],
            "eligible": [True],
            "screen_status": ["ok"],
        }
    )
    screen_path, integration_path, metadata_path = runner._screen_paths(
        paths, rebalance
    )
    diagnostics.to_csv(screen_path, index=False, compression="gzip")
    pd.DataFrame({"ticker": ["A", "B"]}).to_csv(integration_path, index=False)
    runner._json_dump(metadata_path, {"fingerprint": payload["fingerprint"]})

    def unexpected_rescreen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("valid legacy cache should not be recomputed")

    monkeypatch.setattr(runner, "screen_sector_pairs", unexpected_rescreen)
    _, _, enriched, reused = runner._screen_one_window(
        paths,
        prices,
        universe,
        _protocol(),
        {"sha256": "price-hash"},
        rebalance,
        workers=1,
        recompute=False,
    )

    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert reused is True
    assert enriched["candidate_pairs"] == 1
    assert enriched["strict_eligible"] == 1
    assert "elapsed_seconds" in enriched and enriched["elapsed_seconds"] is None
    assert saved["candidate_pairs"] == 1
    assert saved["strict_eligible"] == 1
    assert saved["legacy_missing_fields"] == [
        "candidate_pairs",
        "elapsed_seconds",
        "strict_eligible",
    ]


def test_one_vintage_is_pair_only_resumable_and_has_frozen_baseline(tmp_path):
    paths = runner.StudyPaths.create(tmp_path / "study")
    prices = _synthetic_prices(280)
    rebalance = prices.index[220]
    outer_end = prices.index[260]
    strict = pd.DataFrame([_strict_row(rebalance)])
    protocol = _protocol()
    protocol["pair_selection"]["maximum_pairs_per_rebalance"] = 10
    spec = _small_tuning_spec()

    first, reused_first = runner._run_one_tuning_vintage(
        paths,
        prices,
        strict,
        protocol,
        {"sha256": "synthetic-price-hash"},
        spec,
        _base_config(),
        rebalance,
        outer_end,
        recompute=False,
    )
    second, reused_second = runner._run_one_tuning_vintage(
        paths,
        prices,
        strict,
        protocol,
        {"sha256": "synthetic-price-hash"},
        spec,
        _base_config(),
        rebalance,
        outer_end,
        recompute=False,
    )

    assert reused_first is False
    assert reused_second is True
    assert len(first.parameter_rankings) == 3  # two cells plus baseline
    assert first.parameter_rankings["portfolio_objective_used"].eq(False).all()
    assert len(first.selected_schedule) == 1
    assert set(first.cost_sensitivity["transaction_cost_bps"]) == {1.0, 2.0}
    assert first.cost_sensitivity["config_id"].nunique() == 1
    assert first.oos_signals["transaction_cost_bps"].eq(2.0).all()
    assert len(first.baseline_metrics) == 1
    assert first.baseline_metrics["baseline_used_for_selection"].eq(False).all()
    assert len(first.baseline_signals) == len(first.oos_signals)
    metrics = first.pair_vintage_metrics.iloc[0]
    assert metrics["tuned_minus_baseline_cagr"] == pytest.approx(
        metrics["net_cagr"] - metrics["baseline_net_cagr"]
    )
    assert second.selected_schedule["selected_config_id"].tolist() == first.selected_schedule[
        "selected_config_id"
    ].tolist()


def test_vintage_fingerprint_and_metadata_bind_the_execution_engine(
    tmp_path, monkeypatch
):
    prices = _synthetic_prices(120)
    rebalance = prices.index[100]
    outer_end = prices.index[-1] + pd.offsets.BDay(1)
    formation = prices.loc[prices.index < rebalance]
    outer = prices.loc[prices.index >= rebalance]
    strict = pd.DataFrame([_strict_row(rebalance)])
    protocol = _protocol()
    arguments = {
        "rebalance_date": rebalance,
        "outer_end_exclusive": outer_end,
        "formation": formation,
        "outer": outer,
        "strict": strict,
        "spec": _small_tuning_spec(),
        "base_config": _base_config(),
        "protocol": protocol,
        "price_snapshot_sha256": "price-hash",
    }

    monkeypatch.setattr(runner, "PAIR_TUNING_SOURCE_SHA256", "a" * 64, raising=False)
    first = runner._vintage_fingerprint(**arguments)
    monkeypatch.setattr(runner, "PAIR_TUNING_SOURCE_SHA256", "b" * 64, raising=False)
    second = runner._vintage_fingerprint(**arguments)

    assert first != second
    directory = tmp_path / "vintage"
    directory.mkdir()
    artifacts = runner._empty_vintage_artifacts({"cache_reused": False})
    runner._write_vintage_cache(directory, artifacts, fingerprint=second)
    metadata = json.loads((directory / "complete.meta.json").read_text(encoding="utf-8"))
    assert metadata["pair_tuning_source_sha256"] == "b" * 64
    assert metadata["execution_convention"] == runner.NEXT_CLOSE_EXECUTION_CONVENTION


def test_process_pair_tuning_is_deterministic_against_serial():
    prices = _synthetic_prices(260)
    date = prices.index[220]
    strict = pd.DataFrame(
        [
            _strict_row(date),
            _strict_row(
                date,
                pair="C__D",
                dependent="C",
                independent="D",
                fdr_qvalue=0.02,
                pair_pvalue=0.01,
            ),
        ]
    )
    spec = _small_tuning_spec()
    formation = prices.index[:220]

    serial = runner._tune_vintage_pairs(
        strict,
        prices,
        formation,
        _base_config(),
        spec,
        workers=1,
    )
    parallel = runner._tune_vintage_pairs(
        strict,
        prices,
        formation,
        _base_config(),
        spec,
        workers=2,
    )

    assert serial.failures.empty and parallel.failures.empty
    assert list(serial.results) == list(parallel.results) == ["A__B", "C__D"]
    assert serial.leaderboard["pair"].tolist() == parallel.leaderboard["pair"].tolist()
    assert serial.leaderboard["selected_config_id"].tolist() == parallel.leaderboard[
        "selected_config_id"
    ].tolist()
    np.testing.assert_allclose(
        serial.leaderboard["pair_cv_net_cagr"],
        parallel.leaderboard["pair_cv_net_cagr"],
        rtol=0.0,
        atol=0.0,
    )
