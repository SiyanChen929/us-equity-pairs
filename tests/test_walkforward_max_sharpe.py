from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import equity_pairs.walkforward_max_sharpe as optimizer
from equity_pairs.walkforward_max_sharpe import (
    MU_DEFINITION,
    MaxSharpeSpec,
    build_strict_training_panel,
    maximum_feasible_cardinality,
    select_books_with_cardinality_policy,
    select_matched_score_book,
    select_max_sharpe_book,
)


def _strict_history_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
    pd.Timestamp,
    pd.DatetimeIndex,
]:
    starts = pd.to_datetime(
        [
            "2020-01-02",
            "2020-04-01",
            "2020-07-01",
            "2020-10-01",
            "2021-01-04",
            "2021-04-01",
            "2021-07-01",
            "2021-10-01",
            "2022-01-03",
            "2022-04-01",
        ]
    )
    pairs = ("A__B", "C__D", "E__F")
    history_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    return_levels = {"A__B": 0.020, "C__D": 0.015, "E__F": 0.010}
    daily_scales = {"A__B": 1.0, "C__D": -0.7, "E__F": 0.4}
    for quarter_number, (start, end) in enumerate(
        zip(starts[:-1], starts[1:], strict=True), start=1
    ):
        sessions = pd.bdate_range(start, end - pd.Timedelta(days=1))
        last_session = pd.Timestamp(sessions[-1])
        for pair_index, pair in enumerate(pairs):
            quarter_return = return_levels[pair] + quarter_number * 0.0002
            if quarter_number == 9:
                quarter_return = 5.0 + pair_index
            history_rows.append(
                {
                    "pair": pair,
                    "quarter_number": quarter_number,
                    "rebalance_date": start,
                    "outer_end_exclusive": end,
                    "quarter_last_session": last_session,
                    "quarter_return_2bp": quarter_return,
                    "active_quarter": True,
                    "trade_entries": 2,
                    "history_available": True,
                    "eligible_since_date": starts[0],
                }
            )
            for session_index, date in enumerate(sessions):
                value = daily_scales[pair] * (
                    0.00015 * np.sin(session_index / 7.0 + pair_index)
                    + 0.00005 * np.cos(session_index / 11.0 + quarter_number)
                )
                if quarter_number == 9:
                    value = 0.25 + pair_index * 0.05
                daily_rows.append(
                    {
                        "pair": pair,
                        "date": date,
                        "quarter_number": quarter_number,
                        "net_return_2bp": value,
                        "price_a": 100.0 + pair_index,
                        "price_b": 90.0 + pair_index,
                    }
                )
    metadata = pd.DataFrame(
        {
            "pair": list(pairs),
            "ticker_a": ["A", "C", "E"],
            "ticker_b": ["B", "D", "F"],
        }
    )
    effective = pd.Timestamp(starts[8])
    cutoff = pd.Timestamp(
        pd.bdate_range(starts[7], starts[8] - pd.Timedelta(days=1))[-1]
    )
    expected_sessions = pd.DatetimeIndex(
        sorted(
            {
                pd.Timestamp(row["date"])
                for row in daily_rows
                if int(row["quarter_number"]) <= 8
            }
        )
    )
    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(history_rows),
        metadata,
        effective,
        cutoff,
        expected_sessions,
    )


def _optimization_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = ["A__B", "C__D", "E__F", "G__H", "I__J"]
    candidates = pd.DataFrame(
        {
            "pair": pairs,
            "ticker_a": ["A", "C", "E", "G", "I"],
            "ticker_b": ["B", "D", "F", "H", "J"],
            "expected_annual_return": [0.12, 0.10, 0.08, 0.07, 0.05],
        }
    )
    covariance = pd.DataFrame(
        [
            [0.040, 0.005, 0.002, 0.003, 0.001],
            [0.005, 0.030, 0.004, 0.002, 0.003],
            [0.002, 0.004, 0.020, 0.001, 0.002],
            [0.003, 0.002, 0.001, 0.010, 0.001],
            [0.001, 0.003, 0.002, 0.001, 0.025],
        ],
        index=pairs,
        columns=pairs,
    )
    return candidates, covariance


def _brute_force(
    candidates: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    k: int,
) -> tuple[tuple[str, ...], float, tuple[str, ...]]:
    table = candidates.set_index("pair")
    best_sharpe: tuple[tuple[str, ...], float] | None = None
    best_mu: tuple[tuple[str, ...], float] | None = None
    for combination in itertools.combinations(sorted(table.index), k):
        rows = table.loc[list(combination)]
        names = rows[["ticker_a", "ticker_b"]].astype(str).to_numpy().ravel()
        if len(names) != len(set(names)):
            continue
        mu = float(rows["expected_annual_return"].sum())
        sigma = float(
            covariance.loc[list(combination), list(combination)].to_numpy().sum()
        )
        sharpe = mu / np.sqrt(sigma)
        if best_sharpe is None or sharpe > best_sharpe[1]:
            best_sharpe = (combination, sharpe)
        if best_mu is None or mu > best_mu[1]:
            best_mu = (combination, mu)
    assert best_sharpe is not None and best_mu is not None
    return best_sharpe[0], best_sharpe[1], best_mu[0]


def test_strict_panel_uses_prior_eight_quarters_and_final_mu_definition() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    panel = build_strict_training_panel(
        daily,
        history,
        metadata,
        effective,
        cutoff,
        expected_sessions=sessions,
    )
    assert panel.status == "ready"
    assert panel.quarter_numbers == tuple(range(1, 9))
    assert panel.training_start == sessions[0]
    assert panel.training_end == cutoff
    assert panel.daily_returns.index.equals(sessions)
    assert len(panel.daily_returns) >= 480
    assert np.linalg.eigvalsh(panel.annual_covariance).min() > 0.0
    a_returns = history.loc[
        history["pair"].eq("A__B") & history["quarter_number"].le(8),
        "quarter_return_2bp",
    ].to_numpy()
    logs = np.log1p(a_returns)
    expected = np.expm1(4.0 * (logs.mean() - logs.std(ddof=1) / np.sqrt(8.0)))
    actual = panel.candidate_stats.set_index("pair").loc[
        "A__B", "expected_annual_return"
    ]
    assert actual == pytest.approx(expected)
    assert panel.candidate_stats["mu_definition"].eq(MU_DEFINITION).all()


def test_future_and_fill_quarter_cannot_change_training_inputs() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    first = build_strict_training_panel(
        daily,
        history,
        metadata,
        effective,
        cutoff,
        expected_sessions=sessions,
    )
    changed_daily = daily.copy()
    changed_history = history.copy()
    changed_daily.loc[changed_daily["quarter_number"].eq(9), "net_return_2bp"] = -0.99
    changed_history.loc[
        changed_history["quarter_number"].eq(9), "quarter_return_2bp"
    ] = -0.99
    second = build_strict_training_panel(
        changed_daily,
        changed_history,
        metadata,
        effective,
        cutoff,
        expected_sessions=sessions,
    )
    pd.testing.assert_frame_equal(first.candidate_stats, second.candidate_stats)
    pd.testing.assert_frame_equal(first.daily_returns, second.daily_returns)
    pd.testing.assert_frame_equal(first.annual_covariance, second.annual_covariance)


def test_latest_training_quarter_must_end_at_rebalance() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    history.loc[
        history["quarter_number"].eq(8), "outer_end_exclusive"
    ] = effective - pd.Timedelta(days=1)
    with pytest.raises(
        ValueError, match="latest training quarter does not end"
    ):
        build_strict_training_panel(
            daily,
            history,
            metadata,
            effective,
            cutoff,
            expected_sessions=sessions,
        )


def test_unavailable_first_known_and_invalid_cutoff_prices_are_excluded() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    history.loc[
        history["pair"].eq("C__D") & history["quarter_number"].eq(4),
        "history_available",
    ] = False
    history.loc[history["pair"].eq("E__F"), "eligible_since_date"] = pd.Timestamp(
        "2020-04-01"
    )
    panel = build_strict_training_panel(
        daily,
        history,
        metadata,
        effective,
        cutoff,
        expected_sessions=sessions,
    )
    assert panel.candidate_stats["pair"].tolist() == ["A__B"]

    invalid_price = daily.copy()
    invalid_price.loc[
        invalid_price["pair"].eq("A__B") & invalid_price["date"].eq(cutoff),
        "price_a",
    ] = 0.0
    price_panel = build_strict_training_panel(
        invalid_price,
        _strict_history_fixture()[1],
        metadata,
        effective,
        cutoff,
        expected_sessions=sessions,
    )
    assert "A__B" not in set(price_panel.candidate_stats["pair"])


def test_missing_expected_session_fails_closed() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    missing_date = sessions[100]
    incomplete = daily.loc[daily["date"].ne(missing_date)].copy()
    with pytest.raises(ValueError, match="complete daily window"):
        build_strict_training_panel(
            incomplete,
            history,
            metadata,
            effective,
            cutoff,
            expected_sessions=sessions,
        )


def test_short_common_window_returns_explicit_cash_status() -> None:
    daily, history, metadata, effective, cutoff, sessions = _strict_history_fixture()
    panel = build_strict_training_panel(
        daily,
        history,
        metadata,
        effective,
        cutoff,
        MaxSharpeSpec(minimum_daily_observations=len(sessions) + 1),
        expected_sessions=sessions,
    )
    assert panel.status == "insufficient_daily_observations"
    assert panel.candidate_stats.empty
    assert panel.daily_returns.empty


def test_milp_max_sharpe_and_matched_score_equal_brute_force() -> None:
    candidates, covariance = _optimization_fixture()
    spec = MaxSharpeSpec(
        top_n=3,
        minimum_daily_observations=2,
        dinkelbach_sharpe_certificate_tolerance=1e-4,
    )
    brute_pairs, brute_sharpe, brute_mu_pairs = _brute_force(
        candidates, covariance, k=3
    )
    maximum = select_max_sharpe_book(
        candidates, covariance, exact_k=3, specification=spec
    )
    matched = select_matched_score_book(
        candidates, covariance, exact_k=3, specification=spec
    )
    assert maximum.selected_pairs == brute_pairs
    assert maximum.estimated_sharpe == pytest.approx(brute_sharpe)
    assert maximum.feasible_lower is None
    assert maximum.infeasible_upper is None
    assert maximum.dinkelbach_iterations is not None
    assert maximum.dinkelbach_iterations >= 1
    assert maximum.sharpe_certificate_lower == pytest.approx(brute_sharpe)
    assert maximum.sharpe_certificate_upper >= brute_sharpe
    assert maximum.sharpe_certificate_width <= 1e-4
    assert maximum.final_maximum_scaled_residual is not None
    assert abs(maximum.final_maximum_scaled_residual) <= 1e-9
    assert matched.selected_pairs == brute_mu_pairs
    assert all("elapsed_seconds" in row for row in maximum.solver_statuses)
    dinkelbach_rows = [
        row
        for row in maximum.solver_statuses
        if row["phase"] == "dinkelbach_residual"
    ]
    assert len(dinkelbach_rows) == maximum.dinkelbach_iterations
    assert dinkelbach_rows[0]["lambda_input"] == pytest.approx(
        matched.estimated_sharpe**2
    )


def test_optimizer_is_invariant_to_candidate_input_order() -> None:
    candidates, covariance = _optimization_fixture()
    spec = MaxSharpeSpec(
        top_n=3,
        minimum_daily_observations=2,
        dinkelbach_sharpe_certificate_tolerance=1e-4,
    )
    first = select_max_sharpe_book(
        candidates, covariance, exact_k=3, specification=spec
    )
    order = [3, 0, 4, 1, 2]
    shuffled = candidates.iloc[order].reset_index(drop=True)
    shuffled_pairs = shuffled["pair"].tolist()
    second = select_max_sharpe_book(
        shuffled,
        covariance.loc[shuffled_pairs, shuffled_pairs],
        exact_k=3,
        specification=spec,
    )
    assert first.selected_pairs == second.selected_pairs
    assert first.estimated_sharpe == pytest.approx(second.estimated_sharpe)


def test_rlt_compatible_products_match_overlapping_brute_force() -> None:
    pairs = ["A__B", "A__C", "D__E", "D__F", "G__H", "I__J"]
    candidates = pd.DataFrame(
        {
            "pair": pairs,
            "ticker_a": ["A", "A", "D", "D", "G", "I"],
            "ticker_b": ["B", "C", "E", "F", "H", "J"],
            "expected_annual_return": [0.14, 0.11, 0.10, 0.08, 0.07, -0.01],
        }
    )
    base = np.array(
        [
            [1.0, 0.1, 0.2, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.1, 0.1, 0.0, 0.1],
            [0.1, 0.0, 1.0, 0.2, 0.1, 0.0],
            [0.0, 0.1, 0.0, 1.0, 0.2, 0.1],
            [0.1, 0.0, 0.1, 0.0, 1.0, 0.2],
            [0.0, 0.1, 0.0, 0.1, 0.0, 1.0],
        ]
    )
    covariance_values = (base @ base.T) * 0.01
    covariance = pd.DataFrame(covariance_values, index=pairs, columns=pairs)
    brute_pairs, brute_sharpe, _ = _brute_force(candidates, covariance, k=3)
    result = select_max_sharpe_book(
        candidates,
        covariance,
        exact_k=3,
        specification=MaxSharpeSpec(
            top_n=3,
            minimum_daily_observations=2,
            dinkelbach_sharpe_certificate_tolerance=1e-4,
        ),
    )
    assert result.selected_pairs == brute_pairs
    assert result.estimated_sharpe == pytest.approx(brute_sharpe)
    rows = [
        row
        for row in result.solver_statuses
        if row["phase"] == "dinkelbach_residual"
    ]
    assert rows
    assert rows[-1]["compatible_product_variable_count"] < len(pairs) * 5 / 2
    assert rows[-1]["lifted_encoding"] == "ticker_disjoint_products_rlt_equalities"


def test_nonoverlap_maximum_cardinality_is_exact() -> None:
    candidates = pd.DataFrame(
        {
            "pair": ["A__B", "A__C", "D__E", "D__H", "F__G"],
            "ticker_a": ["A", "A", "D", "D", "F"],
            "ticker_b": ["B", "C", "E", "H", "G"],
        }
    )
    result = maximum_feasible_cardinality(candidates, top_n=10)
    assert result.maximum_feasible_k == 3
    assert result.solver_status["maximum_feasible_k"] == 3
    assert result.solver_status["elapsed_seconds"] >= 0.0


def test_cardinality_decrements_until_positive_mu_and_is_shared() -> None:
    pairs = ["A__B", "C__D", "E__F"]
    candidates = pd.DataFrame(
        {
            "pair": pairs,
            "ticker_a": ["A", "C", "E"],
            "ticker_b": ["B", "D", "F"],
            "expected_annual_return": [0.30, -0.20, -0.20],
        }
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=pairs, columns=pairs)
    spec = MaxSharpeSpec(top_n=3, minimum_daily_observations=2)
    maximum, matched = select_books_with_cardinality_policy(
        candidates, covariance, spec
    )
    assert maximum.maximum_feasible_k == 3
    assert matched.maximum_feasible_k == 3
    assert maximum.selected_count == matched.selected_count == 2
    assert maximum.expected_annual_return_sum > 0.0
    assert matched.expected_annual_return_sum > 0.0
    attempted = [
        row["exact_k"]
        for row in matched.solver_statuses
        if row["phase"] == "maximum_expected_return"
    ]
    assert attempted == [3, 2]


def test_all_milps_receive_frozen_limits_and_status_one_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, covariance = _optimization_fixture()
    original = optimizer.milp
    options_seen: list[dict[str, object]] = []

    def recording_milp(*args: object, **kwargs: object) -> object:
        options_seen.append(dict(kwargs["options"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(optimizer, "milp", recording_milp)
    spec = MaxSharpeSpec(
        top_n=2,
        minimum_daily_observations=2,
        per_solve_time_limit_seconds=7.0,
        mip_relative_gap=0.0,
        dinkelbach_sharpe_certificate_tolerance=1e-3,
    )
    select_books_with_cardinality_policy(candidates, covariance, spec)
    assert options_seen
    assert all(option["time_limit"] == 7.0 for option in options_seen)
    assert all(option["mip_rel_gap"] == 0.0 for option in options_seen)
    assert all(option["presolve"] is True for option in options_seen)

    def limited_milp(*args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            status=1,
            success=False,
            x=None,
            message="time limit reached",
        )

    monkeypatch.setattr(optimizer, "milp", limited_milp)
    with pytest.raises(RuntimeError, match="maximum-cardinality MILP failed"):
        maximum_feasible_cardinality(candidates, top_n=2)


def test_rlt_lifted_solution_is_checked_against_original_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, covariance = _optimization_fixture()
    ordered = candidates.sort_values("pair", kind="mergesort").reset_index(drop=True)
    pair_count = len(ordered)
    products = [
        (left, right)
        for left in range(pair_count)
        for right in range(left + 1, pair_count)
    ]
    solution = np.zeros(pair_count + len(products))
    solution[:2] = 1.0
    for product_index, (left, right) in enumerate(products):
        solution[pair_count + product_index] = float(left < 2 and right < 2)
    solution[pair_count] = 0.0

    def false_optimal(*args: object, **kwargs: object) -> object:
        assert kwargs["options"]["presolve"] is True
        return SimpleNamespace(
            status=0,
            success=True,
            x=solution,
            fun=0.0,
            message="incorrectly reported optimal",
            mip_gap=0.0,
            mip_node_count=0,
            mip_dual_bound=0.0,
        )

    monkeypatch.setattr(optimizer, "milp", false_optimal)
    canonical, mu, sigma = optimizer._canonical_candidates(candidates, covariance)
    with pytest.raises(RuntimeError, match="direct linear-constraint reconstruction"):
        optimizer._fixed_target_feasibility(
            canonical,
            mu,
            sigma,
            exact_k=2,
            target_sharpe=100.0,
            specification=MaxSharpeSpec(top_n=2, minimum_daily_observations=2),
        )
