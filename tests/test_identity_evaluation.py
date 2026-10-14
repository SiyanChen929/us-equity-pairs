from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from equity_pairs.identity_evaluation import (
    add_consistency_score,
    anniversary_block_labels,
    bootstrap_cagr_from_log_blocks,
    canonical_leg_weight,
    exact_calendar_years,
    fixed_universe_rankings,
    path_statistics,
    remove_block_as_cash,
    stability_statistics,
)


def test_path_statistics_keep_cash_sessions_in_risk_metrics() -> None:
    cash_inclusive = np.array([0.10, 0.0, 0.0, -0.05])

    result = path_statistics(cash_inclusive, calendar_years=2.0)
    compressed = path_statistics(np.array([0.10, -0.05]), calendar_years=2.0)

    expected_equity = 1.10 * 0.95
    expected_standard_deviation = np.std(cash_inclusive, ddof=1)
    expected_downside_deviation = 0.05 * math.sqrt(252.0)
    assert result["final_equity"] == pytest.approx(expected_equity)
    assert result["net_total_return"] == pytest.approx(expected_equity - 1.0)
    assert result["calendar_cagr"] == pytest.approx(expected_equity**0.5 - 1.0)
    assert result["annual_volatility"] == pytest.approx(
        expected_standard_deviation * math.sqrt(252.0)
    )
    assert result["sharpe"] == pytest.approx(
        cash_inclusive.mean() / expected_standard_deviation * math.sqrt(252.0)
    )
    assert result["sortino"] == pytest.approx(
        cash_inclusive.mean() * 252.0 / expected_downside_deviation
    )
    assert result["max_drawdown"] == pytest.approx(-0.05)
    assert result["calmar"] == pytest.approx(
        result["calendar_cagr"] / abs(result["max_drawdown"])
    )

    # Compressing out the cash dates preserves terminal wealth, but it must not
    # preserve the cash-inclusive distributional statistics.
    assert compressed["net_total_return"] == pytest.approx(
        result["net_total_return"]
    )
    assert compressed["annual_volatility"] != pytest.approx(
        result["annual_volatility"]
    )
    assert compressed["sharpe"] != pytest.approx(result["sharpe"])


def test_path_statistics_treat_nan_as_cash_and_all_cash_as_zero_growth() -> None:
    with_nan = path_statistics([0.02, np.nan, -0.01], calendar_years=1.0)
    with_cash = path_statistics([0.02, 0.0, -0.01], calendar_years=1.0)
    all_cash = path_statistics(np.zeros(20), calendar_years=10.0)

    assert with_nan == pytest.approx(with_cash, nan_ok=True)
    assert all_cash["final_equity"] == pytest.approx(1.0)
    assert all_cash["net_total_return"] == pytest.approx(0.0)
    assert all_cash["calendar_cagr"] == pytest.approx(0.0)
    assert all_cash["annual_volatility"] == pytest.approx(0.0)
    assert all_cash["max_drawdown"] == pytest.approx(0.0)
    assert math.isnan(all_cash["sharpe"])
    assert math.isnan(all_cash["sortino"])
    assert math.isnan(all_cash["calmar"])


def test_path_statistics_measure_first_session_loss_from_initial_equity() -> None:
    result = path_statistics([-0.10, 0.0], calendar_years=1.0)

    assert result["max_drawdown"] == pytest.approx(-0.10)


def test_calendar_cagr_uses_exact_elapsed_calendar_denominator() -> None:
    start = pd.Timestamp("2016-08-26")
    end_exclusive = pd.Timestamp("2026-08-27")
    elapsed_years = exact_calendar_years(start, end_exclusive)

    result = path_statistics([0.21], calendar_years=elapsed_years)

    assert elapsed_years == pytest.approx((end_exclusive - start).days / 365.2425)
    assert result["calendar_cagr"] == pytest.approx(
        1.21 ** (1.0 / elapsed_years) - 1.0
    )
    assert result["calendar_cagr"] != pytest.approx(0.21)
    with pytest.raises(ValueError, match="must follow start"):
        exact_calendar_years(start, start)
    with pytest.raises(ValueError, match="must be positive"):
        path_statistics([0.01], calendar_years=0.0)


def test_remove_best_block_replaces_it_with_cash_without_shortening_horizon() -> None:
    returns = np.array([0.10, 0.02, -0.05, 0.03])
    best_block = np.array([True, True, False, False])

    result = remove_block_as_cash(
        returns,
        best_block,
        calendar_years=10.0,
    )
    explicit_cash = path_statistics(
        np.array([0.0, 0.0, -0.05, 0.03]),
        calendar_years=10.0,
    )

    assert result == pytest.approx(explicit_cash, nan_ok=True)
    expected_final_equity = 0.95 * 1.03
    assert result["final_equity"] == pytest.approx(expected_final_equity)
    assert result["calendar_cagr"] == pytest.approx(
        expected_final_equity ** (1.0 / 10.0) - 1.0
    )
    # A shortened eight-year or two-observation denominator would be a
    # different counterfactual and must not appear here.
    assert result["calendar_cagr"] != pytest.approx(
        expected_final_equity ** (1.0 / 8.0) - 1.0
    )
    with pytest.raises(ValueError, match="same shape"):
        remove_block_as_cash(returns, [True, False], calendar_years=10.0)


def test_anniversary_labels_respect_exact_boundaries_including_leap_start() -> None:
    start = pd.Timestamp("2020-02-29")
    dates = pd.DatetimeIndex(
        ["2020-02-29", "2021-02-27", "2021-02-28", "2022-02-27"]
    )

    labels = anniversary_block_labels(dates, start, block_count=2)

    np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="outside anniversary blocks"):
        anniversary_block_labels(
            dates.insert(0, pd.Timestamp("2020-02-28")),
            start,
            block_count=2,
        )


def test_anniversary_bootstrap_is_deterministic_and_reuses_shared_draws() -> None:
    constant_rate = np.full(10, np.log1p(0.05))
    varying_rate = np.linspace(np.log1p(-0.08), np.log1p(0.12), 10)
    rates = np.vstack([constant_rate, varying_rate])

    first_quantiles, first_counts = bootstrap_cagr_from_log_blocks(
        rates,
        draws=2_000,
        seed=73,
    )
    second_quantiles, second_counts = bootstrap_cagr_from_log_blocks(
        rates,
        draws=2_000,
        seed=73,
    )
    shared_quantiles, shared_counts = bootstrap_cagr_from_log_blocks(
        rates,
        draws=7,
        seed=999,
        shared_counts=first_counts,
    )

    np.testing.assert_array_equal(first_counts, second_counts)
    np.testing.assert_allclose(first_quantiles, second_quantiles, rtol=0, atol=0)
    np.testing.assert_allclose(shared_counts, first_counts, rtol=0, atol=0)
    np.testing.assert_allclose(shared_quantiles, first_quantiles, rtol=0, atol=0)
    np.testing.assert_allclose(first_counts.sum(axis=1), 1.0)
    np.testing.assert_allclose(first_quantiles[0], np.array([0.05, 0.05, 0.05]))
    assert first_quantiles.shape == (2, 3)

    with pytest.raises(ValueError, match="pair-by-block"):
        bootstrap_cagr_from_log_blocks(np.ones(10))
    with pytest.raises(ValueError, match="incompatible shape"):
        bootstrap_cagr_from_log_blocks(
            rates,
            shared_counts=np.ones((10, 9)),
        )


def test_canonical_leg_weight_is_invariant_to_reciprocal_orientation() -> None:
    # A-on-B with beta=2 and B-on-A with beta=1/2 describe the same canonical
    # absolute gross weight of ticker A.
    a_dependent = canonical_leg_weight(2.0, dependent="A", ticker_a="A")
    b_dependent = canonical_leg_weight(0.5, dependent="B", ticker_a="A")

    assert a_dependent == pytest.approx(1.0 / 3.0)
    assert b_dependent == pytest.approx(1.0 / 3.0)
    assert math.isnan(canonical_leg_weight(0.0, dependent="A", ticker_a="A"))
    assert math.isnan(canonical_leg_weight(np.nan, dependent="A", ticker_a="A"))


def test_stability_statistics_report_orientation_and_parameter_transitions() -> None:
    # Deliberately unordered to ensure transitions are chronological, not input
    # row order. Q1->Q2 is unchanged; Q2->Q3 changes 3 of 5 components.
    active = pd.DataFrame(
        [
            {
                "rebalance_date": "2020-07-01",
                "canonical_weight_a": 0.60,
                "dependent": "B",
                "selected_config_id": "c2",
                "zscore_method": "ewma",
                "zscore_lookback": 20,
                "entry_z": 1.25,
                "exit_z": 0.25,
                "maximum_holding_days": 30,
            },
            {
                "rebalance_date": "2020-01-02",
                "canonical_weight_a": 0.40,
                "dependent": "A",
                "selected_config_id": "c1",
                "zscore_method": "sma",
                "zscore_lookback": 10,
                "entry_z": 1.25,
                "exit_z": 0.25,
                "maximum_holding_days": 10,
            },
            {
                "rebalance_date": "2020-04-01",
                "canonical_weight_a": 0.40,
                "dependent": "A",
                "selected_config_id": "c1",
                "zscore_method": "sma",
                "zscore_lookback": 10,
                "entry_z": 1.25,
                "exit_z": 0.25,
                "maximum_holding_days": 10,
            },
        ]
    )

    result = stability_statistics(active)

    assert result["stability_observations"] == 3
    assert result["canonical_weight_a_mean"] == pytest.approx(7.0 / 15.0)
    assert result["canonical_weight_a_median"] == pytest.approx(0.40)
    assert result["canonical_weight_a_standard_deviation"] == pytest.approx(
        pd.Series([0.40, 0.40, 0.60]).std(ddof=1)
    )
    assert result["canonical_weight_a_iqr"] == pytest.approx(0.10)
    assert result["canonical_weight_a_mad"] == pytest.approx(0.0)
    assert result["dominant_orientation_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["orientation_switch_count"] == 1
    assert result["unique_config_count"] == 2
    assert result["modal_config_fraction"] == pytest.approx(2.0 / 3.0)
    assert result["exact_config_switch_rate"] == pytest.approx(0.50)
    assert result["parameter_component_retention"] == pytest.approx(0.70)
    assert result["zscore_method_switch_rate"] == pytest.approx(0.50)
    assert result["zscore_lookback_switch_rate"] == pytest.approx(0.50)
    assert result["entry_z_switch_rate"] == pytest.approx(0.0)
    assert result["exit_z_switch_rate"] == pytest.approx(0.0)
    assert result["maximum_holding_days_switch_rate"] == pytest.approx(0.50)


def test_empty_stability_statistics_are_explicitly_unrankable() -> None:
    empty = pd.DataFrame(
        columns=[
            "rebalance_date",
            "canonical_weight_a",
            "dependent",
            "selected_config_id",
            "zscore_method",
            "zscore_lookback",
            "entry_z",
            "exit_z",
            "maximum_holding_days",
        ]
    )

    result = stability_statistics(empty)

    assert result["stability_observations"] == 0
    assert result["orientation_switch_count"] == 0
    assert result["unique_config_count"] == 0
    for field in (
        "canonical_weight_a_mean",
        "canonical_weight_a_iqr",
        "dominant_orientation_fraction",
        "modal_config_fraction",
        "exact_config_switch_rate",
        "parameter_component_retention",
    ):
        assert math.isnan(result[field])


def _consistency_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "GOOD",
                "active_quarters": 12,
                "sharpe": 1.5,
                "max_drawdown": -0.05,
                "positive_active_quarter_fraction": 0.80,
                "active_quarter_fraction": 0.75,
                "dominant_orientation_fraction": 1.00,
                "canonical_weight_a_iqr": 0.01,
                "modal_config_fraction": 0.90,
                "parameter_component_retention": 0.95,
            },
            {
                "pair": "MID",
                "active_quarters": 10,
                "sharpe": 0.7,
                "max_drawdown": -0.12,
                "positive_active_quarter_fraction": 0.60,
                "active_quarter_fraction": 0.60,
                "dominant_orientation_fraction": 0.80,
                "canonical_weight_a_iqr": 0.05,
                "modal_config_fraction": 0.70,
                "parameter_component_retention": 0.75,
            },
            {
                "pair": "BAD",
                "active_quarters": 8,
                "sharpe": -0.2,
                "max_drawdown": -0.30,
                "positive_active_quarter_fraction": 0.30,
                "active_quarter_fraction": 0.40,
                "dominant_orientation_fraction": 0.55,
                "canonical_weight_a_iqr": 0.20,
                "modal_config_fraction": 0.40,
                "parameter_component_retention": 0.45,
            },
            {
                "pair": "BELOW_REFERENCE",
                "active_quarters": 7,
                "sharpe": 99.0,
                "max_drawdown": 0.0,
                "positive_active_quarter_fraction": 1.0,
                "active_quarter_fraction": 1.0,
                "dominant_orientation_fraction": 1.0,
                "canonical_weight_a_iqr": 0.0,
                "modal_config_fraction": 1.0,
                "parameter_component_retention": 1.0,
            },
        ]
    )


def test_consistency_score_rewards_every_declared_direction() -> None:
    scored = add_consistency_score(
        _consistency_metrics(),
        reference_minimum_active_quarters=8,
    ).set_index("pair")

    assert scored.loc["GOOD", "consistency_score"] == pytest.approx(100.0)
    assert scored.loc["MID", "consistency_score"] == pytest.approx(200.0 / 3.0)
    assert scored.loc["BAD", "consistency_score"] == pytest.approx(100.0 / 3.0)
    assert scored.loc["BELOW_REFERENCE", "consistency_score"] == pytest.approx(0.0)
    assert (
        scored["consistency_score_reference_minimum_active_quarters"] == 8
    ).all()
    assert (
        scored.loc["GOOD", "consistency_score"]
        > scored.loc["MID", "consistency_score"]
        > scored.loc["BAD", "consistency_score"]
    )


def _fixed_ranking_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair": "CAGR_LEADER",
                "calendar_cagr": 0.20,
                "sharpe": 0.80,
                "max_drawdown": -0.20,
                "worst_quarter_return": -0.12,
                "bootstrap_cagr_ci_low": -0.01,
                "cagr_without_best_quarter": 0.04,
                "cagr_without_best_complete_year": 0.03,
                "availability_fraction": 0.90,
                "eligibility_pass_fraction": 0.40,
                "active_session_cagr": 99.0,
            },
            {
                "pair": "SHARPE_LEADER",
                "calendar_cagr": 0.12,
                "sharpe": 2.00,
                "max_drawdown": -0.15,
                "worst_quarter_return": -0.10,
                "bootstrap_cagr_ci_low": 0.00,
                "cagr_without_best_quarter": 0.05,
                "cagr_without_best_complete_year": 0.04,
                "availability_fraction": 0.80,
                "eligibility_pass_fraction": 0.50,
                "active_session_cagr": -99.0,
            },
            {
                "pair": "DOWNSIDE_AND_REMOVAL_LEADER",
                "calendar_cagr": 0.10,
                "sharpe": 1.10,
                "max_drawdown": -0.04,
                "worst_quarter_return": -0.02,
                "bootstrap_cagr_ci_low": 0.06,
                "cagr_without_best_quarter": 0.09,
                "cagr_without_best_complete_year": 0.08,
                "availability_fraction": 1.00,
                "eligibility_pass_fraction": 0.60,
                "active_session_cagr": 0.0,
            },
            {
                "pair": "ELIGIBILITY_LEADER",
                "calendar_cagr": 0.08,
                "sharpe": 0.70,
                "max_drawdown": -0.10,
                "worst_quarter_return": -0.07,
                "bootstrap_cagr_ci_low": 0.01,
                "cagr_without_best_quarter": 0.03,
                "cagr_without_best_complete_year": 0.02,
                "availability_fraction": 0.95,
                "eligibility_pass_fraction": 0.95,
                "active_session_cagr": 0.0,
            },
            {
                "pair": "LOW_AVAILABILITY",
                "calendar_cagr": 0.18,
                "sharpe": 100.0,
                "max_drawdown": -0.30,
                "worst_quarter_return": -0.20,
                "bootstrap_cagr_ci_low": -0.08,
                "cagr_without_best_quarter": -0.01,
                "cagr_without_best_complete_year": -0.02,
                "availability_fraction": 0.79,
                "eligibility_pass_fraction": 0.10,
                "active_session_cagr": 10_000.0,
            },
        ]
    )


def test_fixed_universe_rankings_emit_exactly_five_declared_rankings() -> None:
    rankings, summaries = fixed_universe_rankings(
        _fixed_ranking_metrics(),
        adequate_availability_fraction=0.80,
    )

    expected_types = {
        "full_calendar_cagr",
        "adequate_availability_sharpe",
        "downside_robustness",
        "best_period_removed_robustness",
        "eligibility_frequency",
    }
    assert set(rankings["rank_type"]) == expected_types
    assert set(summaries["rank_type"]) == expected_types
    assert len(summaries) == 5

    expected_leaders = {
        "full_calendar_cagr": "CAGR_LEADER",
        "adequate_availability_sharpe": "SHARPE_LEADER",
        "downside_robustness": "DOWNSIDE_AND_REMOVAL_LEADER",
        "best_period_removed_robustness": "DOWNSIDE_AND_REMOVAL_LEADER",
        "eligibility_frequency": "ELIGIBILITY_LEADER",
    }
    for rank_type, expected_pair in expected_leaders.items():
        group = rankings.loc[rankings["rank_type"].eq(rank_type)]
        assert group.iloc[0]["pair"] == expected_pair
        assert group["rank"].tolist() == list(range(1, len(group) + 1))

    # Availability is a rank-universe filter only for the named Sharpe table.
    non_sharpe = rankings.loc[
        rankings["rank_type"].ne("adequate_availability_sharpe")
    ]
    assert non_sharpe.groupby("rank_type")["pair"].nunique().eq(5).all()
    sharpe = rankings.loc[
        rankings["rank_type"].eq("adequate_availability_sharpe")
    ]
    assert len(sharpe) == 4
    assert "LOW_AVAILABILITY" not in set(sharpe["pair"])
    assert "LOW_AVAILABILITY" in set(non_sharpe["pair"])

    assert rankings["ranking_is_ex_post_descriptive"].all()
    assert not rankings["deployable_selection"].any()
    assert rankings["full_calendar_metrics_only"].all()
    assert not rankings["active_period_annualization_used"].any()
    assert not summaries["active_period_annualization_used"].any()
    assert set(rankings["ranking_score_column"]) == {
        "calendar_cagr",
        "sharpe",
        "downside_robustness_score",
        "best_period_removed_robustness_score",
        "eligibility_pass_fraction",
    }

    sharpe_summary = summaries.loc[
        summaries["rank_type"].eq("adequate_availability_sharpe")
    ].iloc[0]
    assert sharpe_summary["eligible_pair_count"] == 4
    assert bool(sharpe_summary["availability_filter_applied"])
    assert sharpe_summary["adequate_availability_minimum_fraction"] == pytest.approx(
        0.80
    )
    assert not summaries.loc[
        summaries["rank_type"].ne("adequate_availability_sharpe"),
        "availability_filter_applied",
    ].any()


def test_fixed_universe_rankings_keep_all_114_rows_outside_sharpe_filter() -> None:
    template = _fixed_ranking_metrics().iloc[0].to_dict()
    rows = []
    for number in range(114):
        row = dict(template)
        row["pair"] = f"PAIR_{number:03d}"
        row["calendar_cagr"] = number / 10_000.0
        row["availability_fraction"] = 0.80 if number < 80 else 0.79
        rows.append(row)

    rankings, summaries = fixed_universe_rankings(
        pd.DataFrame(rows), adequate_availability_fraction=0.80
    )

    counts = rankings.groupby("rank_type")["pair"].nunique()
    assert counts["adequate_availability_sharpe"] == 80
    for rank_type in (
        "full_calendar_cagr",
        "downside_robustness",
        "best_period_removed_robustness",
        "eligibility_frequency",
    ):
        assert counts[rank_type] == 114
    summary_counts = summaries.set_index("rank_type")["eligible_pair_count"]
    assert summary_counts["adequate_availability_sharpe"] == 80
    assert summary_counts.drop("adequate_availability_sharpe").eq(114).all()


def test_fixed_universe_ranking_scores_use_correct_downside_directions() -> None:
    metrics = _fixed_ranking_metrics()

    rankings, _ = fixed_universe_rankings(metrics)
    downside = rankings.loc[
        rankings["rank_type"].eq("downside_robustness")
    ].set_index("pair")
    removed = rankings.loc[
        rankings["rank_type"].eq("best_period_removed_robustness")
    ].set_index("pair")

    assert (
        downside.loc["DOWNSIDE_AND_REMOVAL_LEADER", "downside_robustness_score"]
        > downside.loc["ELIGIBILITY_LEADER", "downside_robustness_score"]
        > downside.loc["LOW_AVAILABILITY", "downside_robustness_score"]
    )
    assert removed.loc[
        "DOWNSIDE_AND_REMOVAL_LEADER", "minimum_best_period_removed_cagr"
    ] == pytest.approx(0.08)
    assert (
        removed.loc[
            "DOWNSIDE_AND_REMOVAL_LEADER",
            "best_period_removed_robustness_score",
        ]
        > removed.loc[
            "LOW_AVAILABILITY", "best_period_removed_robustness_score"
        ]
    )


def test_fixed_universe_rankings_ignore_active_session_annualization() -> None:
    original = _fixed_ranking_metrics()
    mutated = original.copy()
    mutated["active_session_cagr"] = -mutated["active_session_cagr"] * 1_000_000.0

    first, _ = fixed_universe_rankings(original)
    second, _ = fixed_universe_rankings(mutated)
    keys = ["rank_type", "rank", "pair", "ranking_score_column"]

    pdt.assert_frame_equal(first[keys], second[keys])
    assert first["full_calendar_metrics_only"].all()
    assert not first["active_period_annualization_used"].any()


def test_fixed_universe_rankings_have_deterministic_lexical_final_ties() -> None:
    template = _fixed_ranking_metrics().iloc[0].to_dict()
    tied = pd.DataFrame(
        [
            {**template, "pair": "Z_PAIR"},
            {**template, "pair": "A_PAIR"},
        ]
    )

    rankings, _ = fixed_universe_rankings(tied)

    for _, group in rankings.groupby("rank_type", sort=False):
        assert group["pair"].tolist() == ["A_PAIR", "Z_PAIR"]


def test_fixed_universe_rankings_allow_an_empty_sharpe_ranking_only() -> None:
    metrics = _fixed_ranking_metrics().copy()
    metrics["availability_fraction"] = 0.20

    rankings, summaries = fixed_universe_rankings(
        metrics, adequate_availability_fraction=0.80
    )

    assert rankings.loc[
        rankings["rank_type"].eq("adequate_availability_sharpe")
    ].empty
    sharpe_summary = summaries.loc[
        summaries["rank_type"].eq("adequate_availability_sharpe")
    ].iloc[0]
    assert sharpe_summary["eligible_pair_count"] == 0
    assert bool(sharpe_summary["ranking_is_empty"])
    non_sharpe = summaries.loc[
        summaries["rank_type"].ne("adequate_availability_sharpe")
    ]
    assert non_sharpe["eligible_pair_count"].eq(len(metrics)).all()
    assert not non_sharpe["ranking_is_empty"].any()


def test_fixed_universe_rankings_validate_contract() -> None:
    metrics = _fixed_ranking_metrics()

    with pytest.raises(ValueError, match="bootstrap_cagr_ci_low"):
        fixed_universe_rankings(metrics.drop(columns="bootstrap_cagr_ci_low"))
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        fixed_universe_rankings(metrics, adequate_availability_fraction=1.01)
    with pytest.raises(ValueError, match="one row per pair"):
        fixed_universe_rankings(pd.concat([metrics, metrics.iloc[[0]]]))
