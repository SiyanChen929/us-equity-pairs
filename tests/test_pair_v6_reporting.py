from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_pair_optimized_10y_report.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_pair_optimized_10y_report_test", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
reporting = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = reporting
MODULE_SPEC.loader.exec_module(reporting)


EMPTY_COLUMNS: dict[str, list[str]] = {
    "parameter_rankings": ["rebalance_date", "pair", "config_id", "selected"],
    "fold_evaluations": [
        "rebalance_date",
        "pair",
        "inner_fold",
        "config_id",
        "model_train_end",
        "validation_start",
    ],
    "model_audit": [
        "rebalance_date",
        "pair",
        "inner_fold",
        "purged_observations",
    ],
    "selected_schedule": [
        "rebalance_date",
        "pair_rank",
        "pair",
        "selected_config_id",
    ],
    "oos_signals": [
        "rebalance_date",
        "pair",
        "date",
        "config_id",
        "transaction_cost_bps",
    ],
    "oos_trades": [
        "rebalance_date",
        "pair",
        "entry_date",
        "exit_date",
    ],
    "pair_vintage_metrics": [
        "rebalance_date",
        "pair",
        "config_id",
        "net_total_return",
    ],
    "cost_sensitivity": [
        "rebalance_date",
        "pair",
        "config_id",
        "transaction_cost_bps",
    ],
    "failures": ["rebalance_date", "pair", "stage", "error_type", "error"],
}


def _write_tables(
    root: Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    for name in reporting.REQUIRED_TABLES:
        frame = tables.get(name)
        if frame is None:
            frame = pd.DataFrame(columns=EMPTY_COLUMNS[name])
        path = processed / reporting.TABLE_FILES[name]
        compression = "gzip" if path.suffix == ".gz" else None
        frame.to_csv(path, index=False, compression=compression)
    for name, frame in tables.items():
        if name in reporting.REQUIRED_TABLES:
            continue
        path = processed / reporting.TABLE_FILES[name]
        compression = "gzip" if path.suffix == ".gz" else None
        frame.to_csv(path, index=False, compression=compression)
    config = root / "synthetic_protocol.json"
    config.write_text('{"protocol": "synthetic-report-test"}\n', encoding="utf-8")
    return config


def _complete_tables(*, fdr_qvalue: float = 0.01) -> dict[str, pd.DataFrame]:
    rebalance = pd.Timestamp("2024-04-01")
    outer_end = pd.Timestamp("2024-07-01")
    pair = "AAA__BBB"
    config_id = "sma__lb20__min16__en1p75__ex0p25__hold30"
    dates = pd.bdate_range(rebalance, periods=4)
    net_return = pd.Series([0.0010, -0.0020, 0.0030, -0.0010])
    turnover = pd.Series([1.0, 0.02, 0.03, 1.02])
    transaction_cost = turnover * 2.0 / 10_000.0
    borrow_cost = pd.Series([0.0, 0.000001, 0.000001, 0.000001])
    equity = (1.0 + net_return).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    net_cagr = float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0)
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    annualized_turnover = float(turnover.mean() * 252.0)
    transaction_cost_sum = float(transaction_cost.sum())
    borrow_cost_sum = float(borrow_cost.sum())
    baseline_return = -0.0025
    baseline_cagr = (1.0 + baseline_return) ** (252.0 / len(dates)) - 1.0

    def number_token(value: float | int) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    parameter_rows: list[dict[str, object]] = []
    minimums = {10: 10, 20: 16, 40: 32, 60: 48}
    for method, lookback, entry_z, exit_z, holding_days in itertools.product(
        ("sma", "ewma"),
        (10, 20, 40, 60),
        (1.25, 1.75, 2.25),
        (0.25, 0.5),
        (10, 30),
    ):
        minimum = minimums[lookback]
        candidate_id = "__".join(
            (
                method,
                f"lb{lookback}",
                f"min{minimum}",
                f"en{number_token(entry_z)}",
                f"ex{number_token(exit_z)}",
                f"hold{holding_days}",
            )
        )
        parameter_rows.append(
            {
                "rebalance_date": rebalance,
                "pair": pair,
                "config_id": candidate_id,
                "zscore_method": method,
                "zscore_lookback": lookback,
                "zscore_min_periods": minimum,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "maximum_holding_days": holding_days,
                "is_baseline": False,
            }
        )
    parameter_rows.append(
        {
            "rebalance_date": rebalance,
            "pair": pair,
            "config_id": "sma__lb60__min40__en2__ex0p5__hold60",
            "zscore_method": "sma",
            "zscore_lookback": 60,
            "zscore_min_periods": 40,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "maximum_holding_days": 60,
            "is_baseline": True,
        }
    )
    parameter_rows.sort(
        key=lambda row: (row["config_id"] != config_id, str(row["config_id"]))
    )
    for display_rank, row in enumerate(parameter_rows, start=1):
        selected = row["config_id"] == config_id
        row.update(
            {
                "display_rank": display_rank,
                "objective": "pair_net_cagr",
                "portfolio_objective_used": False,
                "primary_transaction_cost_bps": 2.0,
                "eligible": True,
                "within_one_standard_error": selected,
                "selected": selected,
            }
        )
    fold_evaluations = pd.DataFrame(
        [
            {
                "rebalance_date": rebalance,
                "pair": pair,
                "inner_fold": 1,
                "config_id": row["config_id"],
                "model_train_end": "2023-12-28",
                "validation_start": "2024-01-02",
                "validation_end": "2024-03-28",
                "purged_observations": 1,
                "start_flat": True,
                "terminal_flat": True,
            }
            for row in parameter_rows
        ]
    )

    metric = {
        "rebalance_date": rebalance,
        "outer_end_exclusive": outer_end,
        "pair_rank": 1,
        "pair": pair,
        "sector": "Technology",
        "dependent": "AAA",
        "independent": "BBB",
        "alpha": 0.2,
        "beta": 1.1,
        "config_id": config_id,
        "evaluation_start": dates[0],
        "evaluation_end": dates[-1],
        "observations": len(dates),
        "net_total_return": total_return,
        "net_cagr": net_cagr,
        "max_drawdown": max_drawdown,
        "annual_volatility": float(net_return.std(ddof=1) * math.sqrt(252.0)),
        "trade_count": 1,
        "closed_trade_count": 1,
        "exposure_fraction": 0.5,
        "annualized_turnover": annualized_turnover,
        "transaction_cost_bps": 2.0,
        "transaction_cost_sum": transaction_cost_sum,
        "borrow_cost_sum": borrow_cost_sum,
        "start_flat": True,
        "terminal_flat": True,
        "terminal_liquidation": True,
        "baseline_total_return": baseline_return,
        "baseline_net_cagr": baseline_cagr,
        "tuned_minus_baseline_total_return": total_return - baseline_return,
        "baseline_used_for_selection": False,
    }

    signals = pd.DataFrame(
        {
            "rebalance_date": rebalance,
            "outer_end_exclusive": outer_end,
            "pair_rank": 1,
            "pair": pair,
            "sector": "Technology",
            "config_id": config_id,
            "transaction_cost_bps": 2.0,
            "date": dates,
            "zscore": [-1.8, -0.7, 0.1, 0.0],
            "decision_event": [
                "enter_long_spread",
                "hold",
                "exit_mean_reversion",
                "flat",
            ],
            "net_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "borrow_cost": borrow_cost,
            "net_equity": equity,
        }
    )

    primary_cost = {
        key: metric[key]
        for key in (
            "rebalance_date",
            "outer_end_exclusive",
            "pair_rank",
            "pair",
            "sector",
            "config_id",
            "net_total_return",
            "net_cagr",
            "max_drawdown",
            "annualized_turnover",
            "trade_count",
            "transaction_cost_sum",
            "borrow_cost_sum",
        )
    }
    primary_cost.update(
        {
            "selection_cost_bps": 2.0,
            "transaction_cost_bps": 2.0,
            "frozen_winner": True,
        }
    )
    sensitivity_cost = dict(primary_cost)
    sensitivity_total_return = total_return + transaction_cost_sum / 2.0
    sensitivity_cost.update(
        {
            "transaction_cost_bps": 1.0,
            "net_total_return": sensitivity_total_return,
            "net_cagr": (1.0 + sensitivity_total_return) ** (252.0 / len(dates))
            - 1.0,
            "max_drawdown": max_drawdown,
            "transaction_cost_sum": transaction_cost_sum / 2.0,
        }
    )

    return {
        "rebalance_schedule": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "outer_end_exclusive": outer_end,
                    "outer_window_truncated_by_asof": False,
                }
            ]
        ),
        "screen_summary": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "formation_start": "2021-04-01",
                    "formation_end": "2024-03-28",
                    "training_sessions": 756,
                    "candidate_pairs": 3,
                    "fdr_significant": 1,
                    "strict_eligible": 1,
                    "strict_sectors": 1,
                }
            ]
        ),
        "parameter_rankings": pd.DataFrame(parameter_rows),
        "fold_evaluations": fold_evaluations,
        "model_audit": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "pair": pair,
                    "inner_fold": 1,
                    "model_train_end": "2023-12-28",
                    "validation_start": "2024-01-02",
                    "purged_observations": 1,
                    "alpha": 0.2,
                    "beta": 1.1,
                }
            ]
        ),
        "selected_schedule": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "formation_start": "2021-04-01",
                    "formation_end": "2024-03-28",
                    "outer_start": rebalance,
                    "outer_end_exclusive": outer_end,
                    "pair_rank": 1,
                    "pair": pair,
                    "sector": "Technology",
                    "selected_config_id": config_id,
                    "zscore_method": "sma",
                    "zscore_lookback": 20,
                    "zscore_min_periods": 16,
                    "entry_z": 1.75,
                    "exit_z": 0.25,
                    "maximum_holding_days": 30,
                    "training_cv_net_cagr": 0.08,
                    "validation_trade_count": 7,
                    "full_training_trade_count": 12,
                    "active_fold_count": 3,
                    "screen_fdr_qvalue": fdr_qvalue,
                    "tradable_eligible": True,
                    "selection_role": "strict_eligible",
                    "primary_transaction_cost_bps": 2.0,
                    "portfolio_objective_used": False,
                }
            ]
        ),
        "oos_signals": signals,
        "oos_trades": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "pair": pair,
                    "entry_date": dates[0],
                    "exit_date": dates[2],
                    "direction": "long_spread",
                    "trading_days": 3,
                    "entry_zscore": -1.8,
                    "exit_zscore": 0.1,
                    "exit_reason": "exit_mean_reversion",
                    "net_return": float((1.0 + net_return.iloc[:3]).prod() - 1.0),
                }
            ]
        ),
        "pair_vintage_metrics": pd.DataFrame([metric]),
        "cost_sensitivity": pd.DataFrame([sensitivity_cost, primary_cost]),
        "failures": pd.DataFrame(columns=EMPTY_COLUMNS["failures"]),
        "vintage_summary": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "strict_candidate_count": 1,
                    "tuned_pair_count": 1,
                    "selected_pair_count": 1,
                    "oos_success_count": 1,
                    "portfolio_objective_used": False,
                }
            ]
        ),
        "oos_baseline_metrics": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "pair": pair,
                    "config_id": "sma__lb60__min40__en2__ex0p5__hold60",
                    "transaction_cost_bps": 2.0,
                    "net_total_return": baseline_return,
                    "net_cagr": baseline_cagr,
                    "baseline_used_for_selection": False,
                }
            ]
        ),
    }


def _empty_tables() -> dict[str, pd.DataFrame]:
    rebalance = pd.Timestamp("2024-04-01")
    return {
        "rebalance_schedule": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "outer_end_exclusive": "2024-07-01",
                    "outer_window_truncated_by_asof": False,
                }
            ]
        ),
        "screen_summary": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "formation_start": "2021-04-01",
                    "formation_end": "2024-03-28",
                    "training_sessions": 756,
                    "candidate_pairs": 3,
                    "fdr_significant": 0,
                    "strict_eligible": 0,
                    "strict_sectors": 0,
                }
            ]
        ),
        "vintage_summary": pd.DataFrame(
            [
                {
                    "rebalance_date": rebalance,
                    "strict_candidate_count": 0,
                    "tuned_pair_count": 0,
                    "selected_pair_count": 0,
                    "oos_success_count": 0,
                    "portfolio_objective_used": False,
                }
            ]
        ),
    }


def _assert_report_and_charts(result: dict[str, object]) -> str:
    report_path = Path(result["report"])
    assert report_path.is_file()
    html = report_path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert '<meta name="viewport"' in html
    charts = [Path(path) for path in result["charts"]]
    assert charts
    assert all(path.is_file() and path.stat().st_size > 0 for path in charts)
    assert all(reporting._chart_decodable(path) for path in charts)
    quality = result["quality"]
    assert quality["report_chart_count"] == len(charts)
    assert quality["report_chart_decodable_count"] == len(charts)
    return html


def test_complete_fewer_than_ten_pair_vintage_report_with_optional_baseline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "complete"
    config = _write_tables(root, _complete_tables())

    result = reporting.generate_report(
        root,
        config_path=config,
        maximum_dashboards=1,
    )

    html = _assert_report_and_charts(result)
    quality = result["quality"]
    assert quality["critical_pass"] is True
    assert quality["selected_pair_vintages"] == 1
    assert quality["successful_oos_pair_vintages"] == 1
    assert quality["baseline_comparator_available"] is True
    assert quality["baseline_not_used_for_selection"] is True
    assert quality["tuned_minus_baseline_return_reconciles"] is True
    assert quality["daily_signals_reconcile_to_pair_vintage_metrics"] is True
    assert quality["trade_ledger_counts_reconcile_to_pair_vintage_metrics"] is True
    assert "Implementation artifacts pass QA." in html
    assert "AAA__BBB" in html
    assert "Frozen tuned minus baseline" in html
    assert "Ex-post realized OOS leaderboard — descriptive only" in html
    assert "Not a prospective top-ten strategy" in html
    assert "immediately following calendar quarter" in html
    assert len(result["charts"]) == 7
    ranking_path = Path(result["descriptive_top10"])
    assert ranking_path.is_file()
    ranking = pd.read_csv(ranking_path)
    assert len(ranking) == 1
    assert int(ranking.loc[0, "descriptive_rank"]) == 1
    assert ranking.loc[0, "ranking_metric"] == "realized_oos_net_total_return"
    assert ranking.loc[0, "selection_timing"] == "ex_post_after_outer_returns"
    assert bool(ranking.loc[0, "deployable_selection"]) is False
    saved = json.loads(Path(result["quality_checks"]).read_text(encoding="utf-8"))
    assert saved["critical_pass"] is True
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["primary_evidence_unit"] == "pair_vintage"
    assert manifest["portfolio_cagr_computed"] is False
    assert manifest["descriptive_ex_post_ranking_generated"] is True
    assert manifest["descriptive_ex_post_ranking_used_for_selection"] is False
    ranking_relative = "data/processed/" + reporting.DESCRIPTIVE_TOP10_FILENAME
    assert ranking_relative in manifest["generated_artifacts"]
    assert reporting.DESCRIPTIVE_TOP10_FILENAME not in {
        item.get("path", "") for item in manifest["source_artifacts"].values()
    }


def test_legitimate_all_empty_run_is_a_passing_zero_selected_report(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty"
    config = _write_tables(root, _empty_tables())

    result = reporting.generate_report(
        root,
        config_path=config,
        maximum_dashboards=1,
    )

    html = _assert_report_and_charts(result)
    quality = result["quality"]
    assert quality["critical_pass"] is True
    assert quality["rebalance_rows"] == 1
    assert quality["selected_pair_vintages"] == 0
    assert quality["successful_oos_pair_vintages"] == 0
    assert quality["all_rebalances_have_vintage_summary"] is True
    assert quality["baseline_comparator_available"] is False
    assert "Implementation artifacts pass QA." in html
    assert "A quarter with zero eligible or successfully tuned pairs remains empty" in html
    assert '2 bp median OOS return</div><div class="v">—' in html
    assert 'Positive OOS vintages</div><div class="v">—' in html
    assert len(result["charts"]) == 6


def test_strict_fdr_violation_renders_artifacts_but_fails_critical_qa(
    tmp_path: Path,
) -> None:
    root = tmp_path / "strict_violation"
    tables = _complete_tables(fdr_qvalue=0.08)
    tables.pop("oos_baseline_metrics")
    metrics = tables["pair_vintage_metrics"].drop(
        columns=[
            "baseline_total_return",
            "baseline_net_cagr",
            "tuned_minus_baseline_total_return",
            "baseline_used_for_selection",
        ]
    )
    tables["pair_vintage_metrics"] = metrics
    config = _write_tables(root, tables)

    result = reporting.generate_report(
        root,
        config_path=config,
        maximum_dashboards=0,
    )

    html = _assert_report_and_charts(result)
    quality = result["quality"]
    assert quality["all_selected_pairs_tradable_eligible"] is True
    assert quality["all_selected_pairs_pass_5pct_bh_fdr"] is False
    assert quality["no_quota_or_exploratory_fallback"] is True
    assert quality["critical_pass"] is False
    assert quality["status"] == "incomplete_or_failed"
    assert "The V6 artifact set is incomplete or failed an invariant." in html
    assert "QA status: INCOMPLETE / FAILED." in html
    assert len(result["charts"]) == 6


def test_descriptive_top10_is_deterministic_ex_post_pair_vintage_ranking() -> None:
    dates = pd.date_range("2022-01-03", periods=13, freq="QS")
    returns = [
        0.20,
        0.20,
        0.18,
        0.16,
        0.14,
        0.12,
        0.10,
        0.08,
        0.06,
        0.04,
        0.02,
        0.01,
        np.nan,
    ]
    metrics = pd.DataFrame(
        {
            "rebalance_date": dates,
            "pair": ["DUP__PAIR", "DUP__PAIR"]
            + [f"PAIR_{index:02d}" for index in range(2, 13)],
            "config_id": [f"config_{index:02d}" for index in range(13)],
            "net_total_return": returns,
            "net_cagr": returns,
            "max_drawdown": [-0.02] * 13,
            "observations": [63] * 13,
        }
    )

    leaders = reporting._descriptive_top_pair_vintages(
        metrics, pd.DataFrame(), maximum=10
    )

    assert len(leaders) == 10
    assert leaders["descriptive_rank"].tolist() == list(range(1, 11))
    assert leaders["net_total_return"].is_monotonic_decreasing
    assert leaders.iloc[0]["rebalance_date"] < leaders.iloc[1]["rebalance_date"]
    assert (leaders["pair"] == "DUP__PAIR").sum() == 2
    assert leaders["ranking_metric"].eq(
        "realized_oos_net_total_return"
    ).all()
    assert leaders["selection_timing"].eq(
        "ex_post_after_outer_returns"
    ).all()
    assert leaders["deployable_selection"].eq(False).all()
