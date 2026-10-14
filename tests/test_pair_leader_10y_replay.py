from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pair_leader_10y_replay.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_pair_leader_10y_replay_test", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
replay = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = replay
MODULE_SPEC.loader.exec_module(replay)


def test_source_top_ten_deduplicates_only_for_identity_replay() -> None:
    returns = [0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, -0.01]
    pairs = [
        "AAA__BBB",
        "CCC__DDD",
        "AAA__BBB",
        "EEE__FFF",
        "GGG__HHH",
        "CCC__DDD",
        "III__JJJ",
        "KKK__LLL",
        "MMM__NNN",
        "OOO__PPP",
        "QQQ__RRR",
    ]
    metrics = pd.DataFrame(
        {
            "pair": pairs,
            "sector": ["Sector"] * len(pairs),
            "rebalance_date": pd.date_range("2020-01-02", periods=len(pairs), freq="QS"),
            "net_total_return": returns,
            "net_cagr": returns,
        }
    )

    top, identities = replay.select_source_leaders(metrics, pair_vintage_count=10)

    assert len(top) == 10
    assert top["realized_pair_vintage_rank"].tolist() == list(range(1, 11))
    assert len(identities) == 8
    assert identities["pair"].tolist() == [
        "AAA__BBB",
        "CCC__DDD",
        "EEE__FFF",
        "GGG__HHH",
        "III__JJJ",
        "KKK__LLL",
        "MMM__NNN",
        "OOO__PPP",
    ]
    assert identities["deployable_selection"].eq(False).all()
    assert identities["identity_selection_timing"].eq(
        "ex_post_after_v6_outer_returns"
    ).all()


def test_schedule_partitions_every_study_session_without_overlap() -> None:
    index = pd.DatetimeIndex(
        [
            "2016-08-26",
            "2016-08-29",
            "2016-09-30",
            "2016-10-03",
            "2016-12-30",
            "2017-01-03",
            "2017-01-31",
        ]
    )

    schedule = replay.build_nonoverlapping_schedule(
        index, "2016-08-26", "2017-02-01"
    )

    assert schedule["outer_sessions"].sum() == len(index)
    assert schedule["rebalance_date"].tolist() == [
        pd.Timestamp("2016-08-26"),
        pd.Timestamp("2016-10-03"),
        pd.Timestamp("2017-01-03"),
    ]
    assert schedule["outer_end_exclusive"].iloc[:-1].reset_index(drop=True).equals(
        schedule["rebalance_date"].iloc[1:].reset_index(drop=True)
    )
    assert bool(schedule.iloc[0]["initial_stub"])
    assert bool(schedule.iloc[-1]["final_truncated"])


def test_stitched_equity_compounds_across_vintages_without_reset() -> None:
    daily = pd.DataFrame(
        {
            "pair": ["AAA__BBB"] * 4,
            "rebalance_date": [
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-04-01"),
                pd.Timestamp("2020-04-01"),
            ],
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-04-01", "2020-04-02"]
            ),
            "return_always_attempt_2bp": [0.10, 0.00, -0.05, 0.02],
            "return_core_health_gated_2bp": [0.10, 0.00, 0.00, 0.00],
            "return_always_attempt_1bp": [0.11, 0.00, -0.04, 0.02],
            "return_core_health_gated_1bp": [0.11, 0.00, 0.00, 0.00],
        }
    )

    stitched = replay._compound_stitched_daily(daily)

    expected = (1.10 * 1.00 * 0.95 * 1.02)
    assert np.isclose(stitched["equity_always_attempt_2bp"].iloc[-1], expected)
    assert np.isclose(stitched["equity_core_health_gated_2bp"].iloc[-1], 1.10)
    assert stitched["drawdown_always_attempt_2bp"].iloc[2] < 0


def test_frozen_protocol_has_exactly_97_candidates() -> None:
    config = replay._load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "pair_leader_10y_replay.json"
    )
    spec, base = replay._build_protocol(config)
    assert len(replay.protocol_parameter_sets(spec, stop_z=base.stop_z)) == 97

