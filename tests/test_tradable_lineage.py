import pandas as pd

from equity_pairs.tradable_lineage import apply_us_tradable_cutoffs, lineage_security_master


def test_us_tradable_cutoffs_remove_vendor_backfill():
    dates = pd.to_datetime(["2021-03-10", "2021-03-11", "2024-07-05", "2024-07-08"])
    prices = pd.DataFrame({"FERG": [1, 2, 3, 4], "SW": [5, 6, 7, 8]}, index=dates)
    corrected, audit = apply_us_tradable_cutoffs(prices)
    assert pd.isna(corrected.loc["2021-03-10", "FERG"])
    assert corrected.loc["2021-03-11", "FERG"] == 2
    assert pd.isna(corrected.loc["2024-07-05", "SW"])
    assert corrected.loc["2024-07-08", "SW"] == 8
    assert not audit["prelisting_values_remaining"].any()


def test_lineage_declares_venue_currency_fx_session_and_ratio():
    lineage = lineage_security_master()
    required = {
        "security_id",
        "venue",
        "currency",
        "fx_to_usd",
        "regular_session_local",
        "conversion_to_next",
    }
    assert required.issubset(lineage.columns)
    assert set(lineage["identity"]) == {"SW", "FERG"}
    assert lineage.groupby("identity")["primary_replay_eligible"].any().all()
