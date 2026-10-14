#!/usr/bin/env python3
"""Run the additive point-in-time data gate and pair-selection correction.

When licensed point-in-time inputs are absent, the command fails closed for the
historical-universe claim while still producing a clearly labeled diagnostic
on the legacy fixed-114 paths with all pre-discovery returns set to cash.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template

from equity_pairs.pit_data import audit_point_in_time_contract
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


DEFAULT_CONFIG = Path("configs/pair_identity_pit_selection_correction.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    return (workspace / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}%}" if np.isfinite(number) else "—"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


REPORT = Template(
    r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Point-in-time and pair-selection correction</title><style>
:root{--ink:#172033;--muted:#667085;--line:#dfe5ed;--blue:#315fbd;--red:#b64b55;--orange:#c9782b;--green:#25785b;--pale:#f4f6f8}*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}.page{max-width:1180px;margin:auto;background:white;min-height:100vh}header{padding:42px 50px;background:#183250;color:white}header h1{margin:4px 0 10px;font-size:32px}header p{margin:0;color:#d9e4f0}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:12px;color:#9fc0e8;font-weight:700}main{padding:34px 50px 60px}h2{margin:36px 0 12px}.blocked,.note,.ok{padding:16px 18px;border-radius:8px;margin:16px 0}.blocked{background:#fff0f1;border-left:5px solid var(--red)}.note{background:#eef5ff;border-left:5px solid var(--blue)}.ok{background:#edf8f3;border-left:5px solid var(--green)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card{border:1px solid var(--line);border-radius:8px;padding:15px}.card .k{color:var(--muted);font-size:11px;text-transform:uppercase}.card .v{font-size:24px;font-weight:750}.table{overflow-x:auto;border:1px solid var(--line);border-radius:8px}.dataframe{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}.dataframe th{background:var(--pale);padding:9px;text-align:left}.dataframe td{padding:8px 9px;border-top:1px solid #edf0f3}.downloads{display:flex;gap:8px;flex-wrap:wrap}.downloads a{border:1px solid var(--line);border-radius:6px;padding:7px 9px;text-decoration:none;color:var(--blue);font-size:12px}code{background:#eef1f5;padding:2px 5px;border-radius:4px}footer{padding:22px 50px;background:var(--pale);color:var(--muted);font-size:12px}@media(max-width:800px){header,main{padding-left:22px;padding-right:22px}.cards{grid-template-columns:1fr 1fr}}</style></head><body><div class="page"><header><div class="eyebrow">Additive credibility extension · existing V6 unchanged</div><h1>Point-in-time universe and winner-selection correction</h1><p>First-knowable entry, synchronized full-universe reranking, White Reality Check and Romano–Wolf max-t.</p></header><main>
{% if not pit_pass %}<div class="blocked"><strong>Point-in-time replay blocked.</strong> The licensed membership, historical GICS, delisting, corporate-action and permanent-identifier extracts are absent or incomplete. No current-Wikipedia/Yahoo substitute is accepted.</div>{% elif not full_pit_replay %}<div class="blocked"><strong>Point-in-time inputs passed, but full replay is pending.</strong> No result can be promoted until the exact V6 strategy has been replayed on the point-in-time panel.</div>{% else %}<div class="ok"><strong>Point-in-time data contract and full V6 replay passed.</strong></div>{% endif %}
<div class="cards"><div class="card"><div class="k">Fixed identities</div><div class="v">{{ pair_count }}</div></div><div class="card"><div class="k">Quarter clusters</div><div class="v">{{ quarter_count }}</div></div><div class="card"><div class="k">Pre-knowable rows set to cash</div><div class="v">{{ removed_rows }}</div></div><div class="card"><div class="k">PIT data gate</div><div class="v">{{ pit_label }}</div></div></div>
<h2>Implementation status</h2><div class="table">{{ status_table }}</div>
<h2>Partial fixed-114 diagnostic</h2><div class="note"><strong>Scope control.</strong> These statistics address first-knowable entry and selection correction only. They remain conditional on the survivor-biased legacy 114 identities and cannot support a historical S&amp;P 500 claim.</div>
<p>Observed leader after removing all pre-discovery returns: <strong>{{ leader }}</strong>, full-calendar CAGR {{ leader_cagr }}. Quarterly-block Reality Check p-value: <strong>{{ quarter_rc }}</strong>. Annual-block sensitivity: <strong>{{ annual_rc }}</strong>. Romano–Wolf 5% rejections: <strong>{{ rw_rejections }}</strong>.</p>
<h3>Top observed pairs after first-knowable gating</h3><div class="table">{{ top_table }}</div>
<h3>Family-wise inference leaders</h3><div class="table">{{ inference_table }}</div>
<h2>Required point-in-time files</h2><div class="table">{{ contract_table }}</div>
<h2>Artifacts</h2><div class="downloads">{% for item in links %}<a href="{{ item.href }}">{{ item.label }}</a>{% endfor %}</div>
<h2>Interpretation</h2><ul><li>Every identity holds cash before its recorded first-knowable date.</li><li>All {{ pair_count }} identities are reranked inside every quarterly and annual bootstrap draw.</li><li>Romano–Wolf controls the pair-family winner-selection error conditional on the fixed legacy universe.</li><li>Neither synchronized bootstrap nor max-t repairs missing historical constituents, historical classifications, delisted securities or researcher-path selection.</li></ul>
</main><footer>Generated {{ generated_at }} · configuration {{ config_path }}</footer></div></body></html>"""
)


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    workspace = Path.cwd().resolve()
    config_file = _resolve(workspace, config_path)
    config = _read_json(config_file)
    output = _resolve(
        workspace,
        output_override or config["outputs"]["root"],
    )
    processed = output / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    pit_audit, _ = audit_point_in_time_contract(workspace, config)
    # A passing input contract is necessary but insufficient. The current
    # command runs only the legacy fixed-114 diagnostic; it must never promote
    # a result until the exact V6 mechanics have also been replayed on the
    # licensed point-in-time panel.
    full_pit_strategy_replay_completed = False
    _atomic_json(pit_audit, processed / "point_in_time_data_contract_audit.json")

    legacy = config["legacy_diagnostic"]
    if not legacy.get("enabled", False):
        raise RuntimeError("legacy diagnostic is disabled and no full PIT replay is available")
    master = pd.read_csv(_resolve(workspace, legacy["master"]))
    quarterly = pd.read_csv(_resolve(workspace, legacy["quarterly_results"]))
    return_column = str(legacy["return_column"])
    known_column = str(legacy["known_date_column"])
    gated = apply_first_knowable_gate(
        quarterly,
        master,
        return_column=return_column,
        known_date_column=known_column,
    )
    metrics = first_knowable_pair_metrics(
        gated,
        calendar_years=float(legacy["calendar_years"]),
        known_date_column=known_column,
    )
    dates, pairs, simple_returns = balanced_return_matrix(gated)
    log_returns = np.log1p(simple_returns)
    observed_log_growth = log_returns.sum(axis=0)
    bootstrap = config["bootstrap"]
    draws = int(bootstrap["draws"])
    seed = int(bootstrap["seed"])
    quarter_indices = circular_block_indices(
        len(dates),
        draws=draws,
        block_length=int(bootstrap["quarter_block_length"]),
        seed=seed,
    )
    quarter_bootstrap = synchronized_reranking_bootstrap(
        log_returns,
        pairs,
        quarter_indices,
        calendar_years=float(legacy["calendar_years"]),
        top_k=int(bootstrap["top_k"]),
        observed_log_growth=observed_log_growth,
    )
    annual_blocks, annual_labels = anniversary_log_blocks(
        dates,
        log_returns,
        anchor=bootstrap["annual_block_anchor"],
        block_count=10,
    )
    annual_indices = iid_block_indices(10, draws=draws, seed=seed + 1)
    annual_bootstrap = synchronized_reranking_bootstrap(
        annual_blocks,
        pairs,
        annual_indices,
        calendar_years=float(legacy["calendar_years"]),
        top_k=int(bootstrap["top_k"]),
        observed_log_growth=observed_log_growth,
    )
    alpha = float(config["multiple_testing"]["family_wise_alpha"])
    quarter_rw, quarter_rc = romano_wolf_max_t(
        log_returns,
        pairs,
        quarter_indices,
        alpha=alpha,
    )
    annual_rw, annual_rc = romano_wolf_max_t(
        annual_blocks,
        pairs,
        annual_indices,
        alpha=alpha,
    )
    quarter_summary = quarter_bootstrap.pair_summary.add_suffix("_quarter_block").rename(
        columns={"pair_quarter_block": "pair"}
    )
    annual_summary = annual_bootstrap.pair_summary.add_suffix("_annual_block").rename(
        columns={"pair_annual_block": "pair"}
    )
    inference = (
        metrics.merge(quarter_summary, on="pair", how="left", validate="one_to_one")
        .merge(annual_summary, on="pair", how="left", validate="one_to_one")
        .merge(
            quarter_rw.add_suffix("_quarter_block").rename(
                columns={"pair_quarter_block": "pair"}
            ),
            on="pair",
            how="left",
            validate="one_to_one",
        )
        .merge(
            annual_rw.add_suffix("_annual_block").rename(
                columns={"pair_annual_block": "pair"}
            ),
            on="pair",
            how="left",
            validate="one_to_one",
        )
    )
    inference["point_in_time_data_contract_passed"] = bool(pit_audit["critical_pass"])
    inference["full_point_in_time_strategy_replay_completed"] = (
        full_pit_strategy_replay_completed
    )
    inference["promotion_eligible"] = (
        inference["reject_at_family_wise_alpha_quarter_block"].astype(bool)
        & inference["reject_at_family_wise_alpha_annual_block"].astype(bool)
        & bool(pit_audit["critical_pass"])
        & full_pit_strategy_replay_completed
    )
    inference = inference.sort_values(
        ["observed_rank", "pair"], kind="mergesort"
    ).reset_index(drop=True)

    _atomic_csv(gated, processed / "first_knowable_quarterly_panel.csv.gz")
    _atomic_csv(metrics, processed / "first_knowable_pair_metrics.csv")
    _atomic_csv(inference, processed / "selection_corrected_pair_inference.csv")
    _atomic_csv(
        quarter_bootstrap.winner_counts,
        processed / "quarter_block_bootstrap_winner_counts.csv",
    )
    _atomic_csv(
        annual_bootstrap.winner_counts,
        processed / "annual_block_bootstrap_winner_counts.csv",
    )
    _atomic_csv(quarter_rw, processed / "romano_wolf_quarter_block.csv")
    _atomic_csv(annual_rw, processed / "romano_wolf_annual_block.csv")
    np.savez_compressed(
        processed / "synchronized_bootstrap_draws.npz",
        pairs=np.asarray(pairs),
        quarter_sampling_indices=quarter_indices,
        quarter_sampled_ranks=quarter_bootstrap.sampled_ranks,
        quarter_sampled_log_growth=quarter_bootstrap.sampled_log_growth,
        annual_labels=annual_labels,
        annual_sampling_indices=annual_indices,
        annual_sampled_ranks=annual_bootstrap.sampled_ranks,
        annual_sampled_log_growth=annual_bootstrap.sampled_log_growth,
    )
    selection_summary = {
        "scope": legacy["scope_label"],
        "point_in_time_data_contract_passed": bool(pit_audit["critical_pass"]),
        "full_point_in_time_strategy_replay_completed": (
            full_pit_strategy_replay_completed
        ),
        "pair_count": len(pairs),
        "quarter_count": len(dates),
        "pre_knowable_pair_quarters_set_to_cash": int(
            (~gated["identity_known_at_quarter_start"]).sum()
        ),
        "quarter_block": quarter_rc,
        "annual_block": annual_rc,
        "promotion_count": int(inference["promotion_eligible"].sum()),
        "full_universe_reranked_each_draw": True,
        "existing_v6_outputs_modified": False,
    }
    _atomic_json(selection_summary, processed / "selection_correction_summary.json")

    status_rows = pd.DataFrame(
        [
            {
                "Requirement": "Point-in-time membership/GICS/delistings/actions",
                "Status": "PASS" if pit_audit["critical_pass"] else "BLOCKED",
                "Scope": "Licensed input contract",
            },
            {
                "Requirement": "Exact V6 replay on point-in-time panel",
                "Status": "PASS" if full_pit_strategy_replay_completed else "PENDING",
                "Scope": "Required before any pair can be promoted",
            },
            {
                "Requirement": "First-knowable identity entry",
                "Status": "PASS",
                "Scope": "Legacy fixed-114 partial diagnostic",
            },
            {
                "Requirement": "Synchronized quarterly/annual reranking bootstrap",
                "Status": "PASS",
                "Scope": f"{draws:,} shared draws per scheme",
            },
            {
                "Requirement": "Reality Check and Romano-Wolf max-t",
                "Status": "PASS",
                "Scope": "Conditional on legacy fixed-114 universe",
            },
        ]
    )
    top_columns = [
        "observed_rank",
        "pair",
        "first_knowable_date",
        "known_quarters",
        "first_knowable_calendar_cagr",
        "probability_rank_1_quarter_block",
        "probability_top_10_quarter_block",
        "romano_wolf_adjusted_pvalue_quarter_block",
        "romano_wolf_adjusted_pvalue_annual_block",
        "promotion_eligible",
    ]
    top = inference[top_columns].head(15).copy()
    for column in top.columns:
        if "cagr" in column or "probability" in column:
            top[column] = top[column].map(_pct)
        elif "pvalue" in column:
            top[column] = top[column].map(lambda value: _num(value, 4))
    inference_leaders = inference.sort_values(
        [
            "romano_wolf_adjusted_pvalue_quarter_block",
            "romano_wolf_adjusted_pvalue_annual_block",
            "pair",
        ],
        kind="mergesort",
    )[top_columns].head(15).copy()
    for column in inference_leaders.columns:
        if "cagr" in column or "probability" in column:
            inference_leaders[column] = inference_leaders[column].map(_pct)
        elif "pvalue" in column:
            inference_leaders[column] = inference_leaders[column].map(
                lambda value: _num(value, 4)
            )
    contract_rows = pd.DataFrame(pit_audit["files"])[
        ["dataset", "exists", "rows", "required_columns_present", "passed", "path"]
    ]
    links = [
        ("PIT data audit", "point_in_time_data_contract_audit.json"),
        ("First-knowable panel", "first_knowable_quarterly_panel.csv.gz"),
        ("Pair metrics", "first_knowable_pair_metrics.csv"),
        ("Selection-corrected inference", "selection_corrected_pair_inference.csv"),
        ("Quarter-block Romano-Wolf", "romano_wolf_quarter_block.csv"),
        ("Annual-block Romano-Wolf", "romano_wolf_annual_block.csv"),
        ("Bootstrap draws", "synchronized_bootstrap_draws.npz"),
    ]
    report_path = output / config["outputs"]["report"]
    output.mkdir(parents=True, exist_ok=True)
    rendered = REPORT.render(
        pit_pass=bool(pit_audit["critical_pass"]),
        full_pit_replay=full_pit_strategy_replay_completed,
        pair_count=len(pairs),
        quarter_count=len(dates),
        removed_rows=int((~gated["identity_known_at_quarter_start"]).sum()),
        pit_label="PASS" if pit_audit["critical_pass"] else "BLOCKED",
        status_table=status_rows.to_html(index=False, border=0),
        leader=html.escape(str(inference.iloc[0]["pair"])),
        leader_cagr=_pct(inference.iloc[0]["first_knowable_calendar_cagr"]),
        quarter_rc=_num(quarter_rc["white_reality_check_pvalue"], 4),
        annual_rc=_num(annual_rc["white_reality_check_pvalue"], 4),
        rw_rejections=(
            f"{quarter_rc['romano_wolf_rejection_count']} quarterly-block / "
            f"{annual_rc['romano_wolf_rejection_count']} annual-block"
        ),
        top_table=top.to_html(index=False, border=0),
        inference_table=inference_leaders.to_html(index=False, border=0),
        contract_table=contract_rows.to_html(index=False, border=0),
        links=[
            {"label": label, "href": f"data/processed/{filename}"}
            for label, filename in links
        ],
        generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
        config_path=str(config_file),
    )
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(rendered, encoding="utf-8")
    temporary_report.replace(report_path)
    return {
        "output_root": str(output),
        "report": str(report_path),
        "point_in_time_data_contract_passed": bool(pit_audit["critical_pass"]),
        "full_point_in_time_strategy_replay_completed": (
            full_pit_strategy_replay_completed
        ),
        "partial_diagnostic_completed": True,
        "pair_count": len(pairs),
        "quarter_count": len(dates),
        "promotion_count": int(inference["promotion_eligible"].sum()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
