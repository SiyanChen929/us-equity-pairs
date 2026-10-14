from __future__ import annotations

import numpy as np
import pandas as pd

from equity_pairs.selection_inference import (
    anniversary_log_blocks,
    apply_first_knowable_gate,
    balanced_return_matrix,
    circular_block_indices,
    first_knowable_pair_metrics,
    iid_block_indices,
    romano_wolf_max_t,
    synchronized_reranking_bootstrap,
)


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2020-01-02", "2020-04-01", "2020-07-01", "2020-10-01"])
    quarterly = pd.DataFrame(
        [
            {"pair": pair, "rebalance_date": date, "quarter_return_2bp": value}
            for pair, values in {
                "A__B": [0.50, 0.05, 0.04, 0.03],
                "C__D": [0.10, 0.01, -0.01, 0.00],
            }.items()
            for date, value in zip(dates, values, strict=True)
        ]
    )
    master = pd.DataFrame(
        {
            "pair": ["A__B", "C__D"],
            "first_successful_v6_tuned_date": ["2020-04-01", "2020-01-02"],
        }
    )
    return quarterly, master


def test_first_knowable_gate_removes_only_pre_discovery_returns() -> None:
    quarterly, master = _panel()
    gated = apply_first_knowable_gate(
        quarterly,
        master,
        return_column="quarter_return_2bp",
        known_date_column="first_successful_v6_tuned_date",
    )
    a = gated.loc[gated["pair"].eq("A__B")].sort_values("rebalance_date")
    assert a["first_knowable_gated_quarter_return"].tolist() == [0.0, 0.05, 0.04, 0.03]
    assert a["pre_knowable_return_removed"].tolist() == [0.5, 0.0, 0.0, 0.0]
    assert a["identity_known_at_quarter_start"].tolist() == [False, True, True, True]
    metrics = first_knowable_pair_metrics(
        gated,
        calendar_years=1.0,
        known_date_column="first_successful_v6_tuned_date",
    ).set_index("pair")
    assert metrics.loc["A__B", "known_quarters"] == 3
    assert np.isclose(metrics.loc["A__B", "removed_pre_knowable_total_return"], 0.5)


def test_synchronized_bootstrap_reranks_full_universe_every_draw() -> None:
    quarterly, master = _panel()
    gated = apply_first_knowable_gate(
        quarterly,
        master,
        return_column="quarter_return_2bp",
        known_date_column="first_successful_v6_tuned_date",
    )
    dates, pairs, returns = balanced_return_matrix(gated)
    logs = np.log1p(returns)
    indices = circular_block_indices(4, draws=200, block_length=2, seed=7)
    result = synchronized_reranking_bootstrap(
        logs,
        pairs,
        indices,
        calendar_years=1.0,
        top_k=1,
        observed_log_growth=logs.sum(axis=0),
    )
    assert result.sampled_ranks.shape == (200, 2)
    assert result.pair_summary["full_universe_reranked_each_draw"].all()
    assert (result.sampled_ranks.min(axis=1) == 1).all()
    assert (result.sampled_ranks.max(axis=1) == 2).all()
    assert result.winner_counts["winner_draw_count"].sum() == 200
    annual, labels = anniversary_log_blocks(
        dates,
        logs,
        anchor="2020-01-02",
        block_count=1,
    )
    assert annual.shape == (1, 2)
    assert (labels == 0).all()


def test_romano_wolf_adjustment_is_family_wise_and_reproducible() -> None:
    generator = np.random.default_rng(11)
    values = np.column_stack(
        [
            generator.normal(0.025, 0.01, size=40),
            generator.normal(0.0, 0.01, size=40),
            generator.normal(-0.01, 0.01, size=40),
        ]
    )
    indices = circular_block_indices(40, draws=1000, block_length=4, seed=13)
    result, summary = romano_wolf_max_t(
        values,
        ["A", "B", "C"],
        indices,
        alpha=0.05,
    )
    assert result["romano_wolf_adjusted_pvalue"].between(0, 1).all()
    assert (
        result["romano_wolf_adjusted_pvalue"]
        >= result["unadjusted_bootstrap_pvalue"]
    ).all()
    assert summary["synchronized_resampling"] is True
    assert summary["null_centered"] is True
    assert 0 <= summary["white_reality_check_pvalue"] <= 1
    assert len(iid_block_indices(10, draws=100, seed=5)) == 100
