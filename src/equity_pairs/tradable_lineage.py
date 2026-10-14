"""Point-in-time security-lineage controls for the SW/FERG audit replay."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


US_LISTING_STARTS = {
    "FERG": pd.Timestamp("2021-03-11"),
    "SW": pd.Timestamp("2024-07-08"),
}


def pair_us_listing_start(pair: str) -> pd.Timestamp | None:
    """Return the latest required US-listing date among a pair's members."""

    members = set(str(pair).split("__"))
    dates = [date for ticker, date in US_LISTING_STARTS.items() if ticker in members]
    return max(dates) if dates else None


def apply_us_tradable_cutoffs(
    prices: pd.DataFrame,
    cutoffs: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank vendor-backfilled observations before each actual US listing.

    The function deliberately does not synthesize an LSE/OTC predecessor.  A
    predecessor with a different venue, currency, close time, or conversion
    mechanism requires a separate execution model.
    """

    rules = dict(cutoffs or US_LISTING_STARTS)
    result = prices.copy().sort_index()
    rows: list[dict[str, Any]] = []
    for ticker, raw_start in sorted(rules.items()):
        if ticker not in result.columns:
            raise KeyError(f"lineage ticker is absent from price panel: {ticker}")
        start = pd.Timestamp(raw_start)
        before = result.index < start
        backfilled = int(result.loc[before, ticker].notna().sum())
        result.loc[before, ticker] = pd.NA
        first_valid = result[ticker].first_valid_index()
        if first_valid is not None and pd.Timestamp(first_valid) < start:
            raise RuntimeError(f"{ticker} retains data before its US listing")
        rows.append(
            {
                "ticker": ticker,
                "us_tradable_start": start,
                "backfilled_observations_removed": backfilled,
                "first_retained_observation": first_valid,
                "prelisting_values_remaining": int(result.loc[before, ticker].notna().sum()),
            }
        )
    return result, pd.DataFrame(rows)


def lineage_security_master() -> pd.DataFrame:
    """Return the explicit security/venue/currency/conversion lineage."""

    rows = [
        {
            "identity": "SW",
            "security_id": "SKG_ORD_IE00B1RR8406",
            "issuer_or_company": "Smurfit Kappa Group plc",
            "ticker": "SKG",
            "venue": "London Stock Exchange",
            "mic": "XLON",
            "currency": "GBp",
            "fx_to_usd": "GBPUSD daily rate; divide GBp by 100, then multiply by GBPUSD",
            "regular_session_local": "08:00-16:30 Europe/London",
            "start_date": "2007-03-20",
            "end_date": "2024-07-05",
            "conversion_to_next": "1 SKG ordinary -> 1 SW ordinary",
            "primary_replay_eligible": False,
            "exclusion_reason": "different venue/currency/close clock; unchanged US-close engine cannot execute it",
        },
        {
            "identity": "SW",
            "security_id": "SMFKY_ADR_US83272W1062",
            "issuer_or_company": "Smurfit Kappa Group plc",
            "ticker": "SMFKY",
            "venue": "US OTC",
            "mic": "OTC",
            "currency": "USD",
            "fx_to_usd": "none",
            "regular_session_local": "US OTC regular session, America/New_York",
            "start_date": "2012-01-06",
            "end_date": "2024-07-26",
            "conversion_to_next": "1 ADR represented 1 SKG ordinary; program terminated for cash, no 1:1 SW share delivery",
            "primary_replay_eligible": False,
            "exclusion_reason": "Yahoo history unavailable and terminal cash mechanics differ from SW ordinary",
        },
        {
            "identity": "SW",
            "security_id": "SW_ORD_IE00028FXN24",
            "issuer_or_company": "Smurfit Westrock plc",
            "ticker": "SW",
            "venue": "New York Stock Exchange",
            "mic": "XNYS",
            "currency": "USD",
            "fx_to_usd": "none",
            "regular_session_local": "09:30-16:00 America/New_York",
            "start_date": "2024-07-08",
            "end_date": "",
            "conversion_to_next": "current security",
            "primary_replay_eligible": True,
            "exclusion_reason": "",
        },
        {
            "identity": "FERG",
            "security_id": "WOS_FERG_ORD_GB00B4VLR192",
            "issuer_or_company": "Wolseley plc / Ferguson plc",
            "ticker": "WOS through 2017-07-31; FERG from 2017-08-01",
            "venue": "London Stock Exchange",
            "mic": "XLON",
            "currency": "GBp",
            "fx_to_usd": "GBPUSD daily rate; divide GBp by 100, then multiply by GBPUSD",
            "regular_session_local": "08:00-16:30 Europe/London",
            "start_date": "1986-01-01",
            "end_date": "2021-03-10 for official USD performance splice",
            "conversion_to_next": "name/ticker change was 1:1; same ordinary economic interest",
            "primary_replay_eligible": False,
            "exclusion_reason": "different venue/currency/close clock; unchanged US-close engine cannot execute it",
        },
        {
            "identity": "FERG",
            "security_id": "FERGY_ADR_US31502A3032",
            "issuer_or_company": "Ferguson plc",
            "ticker": "FERGY",
            "venue": "US OTCQX",
            "mic": "OTC",
            "currency": "USD",
            "fx_to_usd": "none",
            "regular_session_local": "US OTC regular session, America/New_York",
            "start_date": "2018-06-11 for final CUSIP program",
            "end_date": "2021-03-08",
            "conversion_to_next": "10 ADS -> 1 NYSE-traded ordinary share for nil consideration",
            "primary_replay_eligible": False,
            "exclusion_reason": "Yahoo history unavailable for the delisted ADR",
        },
        {
            "identity": "FERG",
            "security_id": "FERG_ORD_JE00BJVNSS43",
            "issuer_or_company": "Ferguson plc",
            "ticker": "FERG",
            "venue": "New York Stock Exchange",
            "mic": "XNYS",
            "currency": "USD",
            "fx_to_usd": "none",
            "regular_session_local": "09:30-16:00 America/New_York",
            "start_date": "2021-03-11",
            "end_date": "2024-07-31",
            "conversion_to_next": "1 Ferguson plc ordinary -> 1 Ferguson Enterprises Inc. share",
            "primary_replay_eligible": True,
            "exclusion_reason": "",
        },
        {
            "identity": "FERG",
            "security_id": "FERG_US31488V1070",
            "issuer_or_company": "Ferguson Enterprises Inc.",
            "ticker": "FERG",
            "venue": "New York Stock Exchange",
            "mic": "XNYS",
            "currency": "USD",
            "fx_to_usd": "none",
            "regular_session_local": "09:30-16:00 America/New_York",
            "start_date": "2024-08-01",
            "end_date": "",
            "conversion_to_next": "current security; 1:1 corporate reorganization",
            "primary_replay_eligible": True,
            "exclusion_reason": "",
        },
    ]
    return pd.DataFrame(rows)


def lineage_source_register() -> pd.DataFrame:
    """Primary-source register used to establish dates and conversion terms."""

    return pd.DataFrame(
        [
            {
                "identity": "SW",
                "fact": "NYSE SW began 2024-07-08 at 09:30 ET; SKG LSE last traded 2024-07-05",
                "source": "https://www.sec.gov/Archives/edgar/data/2005951/000110465924071257/tm2416911d2_425.htm",
            },
            {
                "identity": "SW",
                "fact": "each SKG ordinary was exchanged for one SW ordinary",
                "source": "https://www.sec.gov/Archives/edgar/data/2005951/000110465924078355/tm2418700d1_8k.htm",
            },
            {
                "identity": "SW",
                "fact": "SMFKY was OTC, 1 ordinary to 1 DR, and terminated for cash",
                "source": "https://depositaryreceipts.citi.com/adr/notices/previewPdf.aspx?caId=2568",
            },
            {
                "identity": "FERG",
                "fact": "NYSE listing became effective 2021-03-08; primary listing moved 2022-05-12",
                "source": "https://www.corporate.ferguson.com/pressroom/news-releases/news-details/2022/Fergusons-Primary-Listing-Now-on-the-NYSE-As-It-Completes-Transfer-of-Listing-Category-05-12-2022/default.aspx",
            },
            {
                "identity": "FERG",
                "fact": "official performance splice uses LSE through 2021-03-10 and NYSE from 2021-03-11 with daily GBPUSD before the splice",
                "source": "https://www.sec.gov/Archives/edgar/data/1832433/000119312523257226/d650978dars.pdf",
            },
            {
                "identity": "FERG",
                "fact": "10 FERGY ADS were mandatorily exchanged for one ordinary on 2021-03-08",
                "source": "https://www.corporate.ferguson.com/pressroom/news-releases/news-details/2021/DirectorPDMR-Shareholding-03-11-2021/default.aspx",
            },
            {
                "identity": "FERG",
                "fact": "2024 US domicile reorganization issued one new FERG share for each old ordinary",
                "source": "https://www.sec.gov/Archives/edgar/data/2011641/000201164124000005/ferg-20240731.htm",
            },
        ]
    )
