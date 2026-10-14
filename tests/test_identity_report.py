from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_pair_identity_10y_report.py"
_SPEC = importlib.util.spec_from_file_location("render_pair_identity_10y_report", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_normalise_metrics = _MODULE._normalise_metrics
_normalise_rankings = _MODULE._normalise_rankings


def test_named_precomputed_ranking_scores_survive_report_normalisation() -> None:
    config = {"evaluation": {"primary_path": "fixed_identity_full_panel"}}
    raw_metrics = pd.DataFrame(
        {
            "pair": ["AAA__BBB"],
            "transaction_cost_bps": [2.0],
            "calendar_cagr": [0.12],
            "sharpe": [0.9],
            "original_v6_selected_identity": [True],
        }
    )
    raw_rankings = pd.DataFrame(
        {
            "pair": ["AAA__BBB", "AAA__BBB"],
            "ranking_scope": ["fixed_114_primary", "fixed_114_primary"],
            "rank_type": [
                "downside_robustness",
                "best_period_removed_robustness",
            ],
            "rank": [1, 1],
            "transaction_cost_bps": [2.0, 2.0],
            "downside_robustness_score": [96.75, 96.75],
            "best_period_removed_robustness_score": [88.5, 88.5],
        }
    )

    metrics = _normalise_metrics(raw_metrics, config)
    rankings = _normalise_rankings(raw_rankings, metrics, config)

    observed = rankings.set_index("rank_type")["score"]
    assert observed["downside_robustness"] == 96.75
    assert observed["best_period_removed_robustness"] == 88.5
