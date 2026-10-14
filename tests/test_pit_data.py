from __future__ import annotations

from pathlib import Path

from equity_pairs.pit_data import CONTRACTS, audit_point_in_time_contract


def test_point_in_time_contract_fails_closed_when_files_are_absent(tmp_path: Path) -> None:
    settings = {
        "root": str(tmp_path / "pit"),
        "coverage_start_inclusive": "2013-08-26",
        "coverage_end_exclusive": "2026-08-27",
        "permanent_security_identifier": "security_id",
    }
    for contract in CONTRACTS:
        settings[contract.config_key] = f"{contract.config_key}.csv"
    audit, datasets = audit_point_in_time_contract(
        tmp_path,
        {"point_in_time_data": settings},
    )
    assert audit["critical_pass"] is False
    assert audit["current_yahoo_wikipedia_snapshot_is_eligible"] is False
    assert len(audit["files"]) == len(CONTRACTS)
    assert len(audit["errors"]) == len(CONTRACTS)
    assert datasets == {}
