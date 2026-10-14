#!/usr/bin/env python3
"""Render a concise audit report for the SW/FERG PIT-lineage V8 rerun."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8 = PROJECT_ROOT / "outputs/pair_strategy_v8_pit_lineage_2026-08-26"
DEFAULT_IDENTITY = PROJECT_ROOT / "outputs/pair_identity_pit_lineage_10y_2026-08-26"
OLD_V8 = PROJECT_ROOT / "outputs/pair_strategy_v8_pair_centric_2026-08-26"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _fmt_table(frame: pd.DataFrame, percent: set[str] | None = None) -> str:
    shown = frame.copy()
    for column in percent or set():
        if column in shown:
            shown[column] = pd.to_numeric(shown[column], errors="coerce").map(
                lambda value: "—" if pd.isna(value) else f"{value:.2%}"
            )
    return shown.to_html(index=False, border=0, escape=True)


def render(v8_root: Path, identity_root: Path) -> Path:
    processed = v8_root / "data" / "processed"
    metrics = pd.read_csv(processed / "v8_pair_metrics.csv")
    quality = json.loads((processed / "v8_quality_checks.json").read_text(encoding="utf-8"))
    lineage_audit = json.loads((identity_root / "pit_lineage_rerun_audit.json").read_text(encoding="utf-8"))
    lineage = pd.read_csv(identity_root / "data" / "lineage" / "sw_ferg_security_lineage.csv")
    sources = pd.read_csv(identity_root / "data" / "lineage" / "sw_ferg_lineage_sources.csv")
    primary = metrics.loc[metrics["transaction_cost_bps"].eq(2.0)].copy()
    primary = primary.sort_values(["calendar_cagr", "pair"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    primary.insert(0, "all_114_rank", np.arange(1, len(primary) + 1))
    primary["adequate_coverage"] = primary["availability_fraction"].ge(0.8) & primary["active_quarters"].ge(20)
    adequate = primary.loc[primary["adequate_coverage"]].copy().reset_index(drop=True)
    adequate.insert(0, "adequate_rank", np.arange(1, len(adequate) + 1))
    old = pd.read_csv(OLD_V8 / "data" / "processed" / "v8_pair_metrics.csv")
    old = old.loc[old["transaction_cost_bps"].eq(2.0), ["pair", "calendar_cagr", "availability_fraction", "active_quarters"]]
    old = old.rename(columns={column: f"old_{column}" for column in old.columns if column != "pair"})
    comparison = primary.merge(old, on="pair", how="left", validate="one_to_one")
    comparison["cagr_change"] = comparison["calendar_cagr"] - comparison["old_calendar_cagr"]
    affected = comparison.loc[comparison["pair"].map(lambda value: bool(set(str(value).split("__")) & {"SW", "FERG"}))].copy()
    affected = affected.sort_values("old_calendar_cagr", ascending=False, kind="mergesort")
    _write_csv(primary, processed / "pit_all_114_full_calendar_ranking.csv")
    _write_csv(adequate, processed / "pit_adequate_coverage_ranking.csv")
    _write_csv(affected, processed / "pit_sw_ferg_impact.csv")
    top_columns = [
        "all_114_rank", "pair", "calendar_cagr", "sharpe", "max_drawdown",
        "trade_count", "active_quarters", "availability_fraction", "bootstrap_cagr_ci_low",
    ]
    adequate_columns = [
        "adequate_rank", "pair", "calendar_cagr", "sharpe", "max_drawdown",
        "trade_count", "active_quarters", "bootstrap_cagr_ci_low",
    ]
    impact_columns = [
        "pair", "old_calendar_cagr", "calendar_cagr", "cagr_change",
        "old_availability_fraction", "availability_fraction", "active_quarters",
    ]
    lineage_columns = [
        "identity", "ticker", "venue", "currency", "fx_to_usd",
        "regular_session_local", "start_date", "end_date", "conversion_to_next",
        "primary_replay_eligible",
    ]
    source_links = "".join(
        f'<li>{html.escape(str(row.fact))}: <a href="{html.escape(str(row.source))}">source</a></li>'
        for row in sources.itertuples(index=False)
    )
    chart = "charts/v8_top10_equity_curves.png"
    rw = quality["multiple_testing"]
    document = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SW/FERG PIT Lineage · 114 Pairs Rerun</title><style>
body{{margin:0;background:#eef2f5;color:#172334;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1180px;margin:auto;background:white;min-height:100vh;padding:36px 46px}}h1{{font-size:32px;margin:.2em 0}}h2{{margin-top:34px;border-bottom:1px solid #dce3e9;padding-bottom:7px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.kpi{{background:#edf5fb;border-radius:8px;padding:14px}}.kpi b{{display:block;font-size:22px}}.note{{background:#fff8df;border-left:5px solid #d59b18;padding:13px 16px}}.ok{{background:#edf8f0;border-left:5px solid #2d8a50;padding:13px 16px}}.table{{overflow:auto;border:1px solid #dfe5ea;border-radius:7px}}table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}th,td{{padding:7px 9px;border-bottom:1px solid #e5e9ed;text-align:right}}th{{background:#f4f6f8}}th:nth-child(2),td:nth-child(2),th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #dfe5ea}}code{{background:#f0f3f5;padding:2px 4px}}@media(max-width:800px){{main{{padding:22px}}.kpis{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><div>ADDITIVE · POINT-IN-TIME LINEAGE AUDIT</div><h1>SW / FERG lineage correction and 114-pair rerun</h1><p>Evaluation: 2016-08-26 through 2026-08-26 · 41 adjacent quarterly boundaries · 2 bp primary cost + 30 bp annual short-borrow proxy.</p>
<div class="kpis"><div class="kpi"><b>114</b>fixed identities</div><div class="kpi"><b>4,674</b>pair-quarters</div><div class="kpi"><b>533</b>affected tasks recomputed</div><div class="kpi"><b>4,141</b>exact-price cache reuses</div></div>
<h2>Decision and result</h2><div class="ok"><strong>Execution lineage defect corrected.</strong> Pre-NYSE Yahoo backfill is unavailable cash. No LSE close, synthetic FX series, or delisted ADR proxy enters the primary replay. V6 signal logic and the V8 candidate rules are unchanged.</div>
<p>The unconstrained full-calendar leader is <strong>{html.escape(str(primary.iloc[0]['pair']))}</strong> at <strong>{primary.iloc[0]['calendar_cagr']:.2%}</strong>, but availability is only {primary.iloc[0]['availability_fraction']:.1%}. With ≥80% availability and ≥20 active quarters, <strong>{html.escape(str(adequate.iloc[0]['pair']))}</strong> ranks first at <strong>{adequate.iloc[0]['calendar_cagr']:.2%}</strong> CAGR; the most robust lower bound remains {adequate.sort_values('bootstrap_cagr_ci_low', ascending=False).iloc[0]['pair']}.</p>
<div class="note"><strong>Research status: NOT VALIDATED.</strong> White Reality Check p={rw['white_reality_check_pvalue']:.3f}; the fixed universe is ex-post/current-constituent based, historical minute data is absent, and costs/borrow are proxies. Romano–Wolf rejects {rw['romano_wolf_rejection_count']} of 114 at 5% FWER under the ten-block resampling, but ten effective annual blocks remain a small sample.</div>
<h2>Top 10 · all 114, no coverage filter</h2><p>Sparse histories remain visible. This table should not be used to hide limited availability.</p><div class="table">{_fmt_table(primary[top_columns].head(10), {'calendar_cagr','max_drawdown','availability_fraction','bootstrap_cagr_ci_low'})}</div>
<h2>Top 10 · adequate coverage</h2><div class="table">{_fmt_table(adequate[adequate_columns].head(10), {'calendar_cagr','max_drawdown','bootstrap_cagr_ci_low'})}</div><img src="{chart}" alt="Corrected top ten equity curves">
<h2>Impact on the 13 SW/FERG identities</h2><div class="table">{_fmt_table(affected[impact_columns], {'old_calendar_cagr','calendar_cagr','cagr_change','old_availability_fraction','availability_fraction'})}</div>
<h2>Exact tradable lineage</h2><div class="table">{_fmt_table(lineage[lineage_columns])}</div><ul>{source_links}</ul>
<h2>Audit</h2><ul><li>V6 S0 reconciliation: {quality['v6_control_reconciliation']['matched_pair_quarters']:,}/{quality['v6_control_reconciliation']['expected_successful_pair_quarters']:,}, max error {quality['v6_control_reconciliation']['maximum_absolute_error_2bp']:.1e}.</li><li>Independent daily accounting max net error: {quality['manual_reconciliation']['maximum_net_error']:.1e}.</li><li>Identity replay critical pass: {lineage_audit['critical_pass']}.</li><li>Minute-history gate: {quality['minute_history_passed']}.</li></ul>
<h2>Files</h2><ul><li><a href="pair_strategy_v8_pit_lineage_report.html">Full V8 report</a></li><li><a href="data/processed/pit_all_114_full_calendar_ranking.csv">All-114 ranking</a></li><li><a href="data/processed/pit_adequate_coverage_ranking.csv">Coverage-controlled ranking</a></li><li><a href="data/processed/pit_sw_ferg_impact.csv">SW/FERG impact</a></li><li><a href="../pair_identity_pit_lineage_10y_2026-08-26/data/lineage/sw_ferg_security_lineage.csv">Security lineage</a></li></ul>
<footer>Generated {datetime.now(timezone.utc).isoformat()} · all rankings are ex-post descriptive.</footer></main></body></html>"""
    report = v8_root / "pit_lineage_rerun_report.html"
    temporary = report.with_name(report.name + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(report)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": _sha256(report),
        "all_114_rows": len(primary),
        "adequate_coverage_rows": len(adequate),
        "affected_identity_rows": len(affected),
        "lineage_rows": len(lineage),
        "source_rows": len(sources),
    }
    (processed / "pit_lineage_report_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-root", type=Path, default=DEFAULT_V8)
    parser.add_argument("--identity-root", type=Path, default=DEFAULT_IDENTITY)
    args = parser.parse_args()
    print(render(args.v8_root.resolve(), args.identity_root.resolve()))


if __name__ == "__main__":
    main()
