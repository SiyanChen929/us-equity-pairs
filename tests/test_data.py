from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equity_pairs import data  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_normalize_yahoo_tickers_is_ordered_and_deduplicated():
    assert data.normalize_yahoo_ticker(" brk.b ") == "BRK-B"
    assert data.normalize_yahoo_tickers(["brk.b", "AAPL", "BRK-B"]) == (
        "BRK-B",
        "AAPL",
    )
    with pytest.raises(ValueError):
        data.normalize_yahoo_ticker("bad symbol")
    with pytest.raises(ValueError):
        data.normalize_yahoo_ticker(pd.NA)


def test_fetch_sp500_constituents_normalizes_schema_and_tickers(monkeypatch):
    source = pd.DataFrame(
        {
            "Symbol": ["BF.B", "AAPL"],
            "Security": ["Brown-Forman", "Apple"],
            "GICS Sector": ["Consumer Staples", "Information Technology"],
            "GICS Sub-Industry": ["Distillers", "Technology Hardware"],
            "CIK": [14693, 320193],
            "Unneeded": [1, 2],
        }
    )
    session = SequenceSession([FakeResponse(text="<html />")])
    monkeypatch.setattr(data.pd, "read_html", lambda _: [source])

    result = data.fetch_sp500_constituents(session=session, max_retries=0)

    assert list(result.columns) == [
        "ticker",
        "source_ticker",
        "security",
        "sector",
        "sub_industry",
        "cik",
    ]
    assert result["ticker"].tolist() == ["AAPL", "BF-B"]
    assert session.calls[0][0] == data.SP500_WIKIPEDIA_URL


def test_http_get_retries_transient_status_without_leaking_params():
    secret = "a" * 32
    session = SequenceSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            FakeResponse(status_code=503),
        ]
    )
    with pytest.raises(data.DataDownloadError) as error:
        data._safe_get(
            session,
            "https://example.test/data",
            params={"api_key": secret},
            max_retries=1,
            sleep_func=lambda _: None,
            request_name="test service",
        )
    assert len(session.calls) == 2
    assert secret not in str(error.value)


def test_yahoo_download_is_chunked_adjusted_and_uses_ticker_fallback():
    dates = pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03"])
    calls = []

    def fake_download(**kwargs):
        requested = tuple(kwargs["tickers"])
        calls.append((requested, kwargs))
        if requested == ("BRK-B", "AAPL"):
            columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
            return pd.DataFrame([10.0, 11.0, 12.0, 13.0], index=dates, columns=columns)
        if requested == ("BRK-B",):
            return pd.DataFrame({"Close": [20.0, 21.0, 22.0, 23.0]}, index=dates)
        raise AssertionError(f"unexpected request: {requested}")

    result = data.download_adjusted_close(
        ["brk.b", "AAPL", "BRK-B"],
        "2020-01-01",
        "2020-01-03",
        chunk_size=2,
        max_retries=0,
        download_func=fake_download,
    )

    assert list(result.columns) == ["BRK-B", "AAPL"]
    assert result.index.tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert result["BRK-B"].tolist() == [21.0, 22.0]
    assert [call[0] for call in calls] == [("BRK-B", "AAPL"), ("BRK-B",)]
    assert all(call[1]["auto_adjust"] is True for call in calls)
    assert all(call[1]["threads"] is False for call in calls)


def test_yahoo_download_retries_exception_then_succeeds():
    attempts = 0

    def flaky_download(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.Timeout("temporary")
        return pd.DataFrame(
            {"Close": [100.0]}, index=pd.to_datetime(["2022-02-01"])
        )

    result = data.download_adjusted_close(
        "MSFT",
        "2022-02-01",
        "2022-02-02",
        max_retries=1,
        backoff_factor=0,
        download_func=flaky_download,
        sleep_func=lambda _: None,
    )
    assert attempts == 2
    assert result.loc[pd.Timestamp("2022-02-01"), "MSFT"] == 100.0


def test_yahoo_cache_round_trip_avoids_network_and_has_safe_metadata(tmp_path):
    cache = tmp_path / "prices.csv"
    calls = 0

    def fake_download(**kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    first = data.download_adjusted_close(
        "AAPL",
        "2024-01-02",
        "2024-01-04",
        cache_path=cache,
        max_retries=0,
        download_func=fake_download,
    )
    second = data.download_adjusted_close(
        "AAPL",
        "2024-01-02",
        "2024-01-04",
        cache_path=cache,
        download_func=lambda **_: pytest.fail("cache miss"),
    )

    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert calls == 1
    metadata_text = data.snapshot_metadata_path(cache).read_text(encoding="utf-8")
    assert "api_key" not in metadata_text.lower()
    assert json.loads(metadata_text)["end_exclusive"] == "2024-01-04"


def test_adjusted_ohlcv_handles_multiindex_and_single_ticker_fallback():
    dates = pd.to_datetime(
        ["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03"]
    )
    calls = []

    def fake_download(**kwargs):
        requested = tuple(kwargs["tickers"])
        calls.append((requested, kwargs))
        if requested == ("BRK-B", "AAPL"):
            # yfinance can put either tickers or fields on the outer level.
            columns = pd.MultiIndex.from_tuples(
                [
                    ("AAPL", "Volume"),
                    ("BRK-B", "Close"),
                    ("AAPL", "Low"),
                    ("AAPL", "Close"),
                    ("BRK-B", "High"),
                    ("AAPL", "High"),
                    ("BRK-B", "Low"),
                ]
            )
            values = np.array(
                [
                    [1000, 20, 9, 10, 22, 11, 18],
                    [1100, 21, 10, 11, 23, 12, 19],
                    [1200, 22, 11, 12, 24, 13, 20],
                    [1300, 23, 12, 13, 25, 14, 21],
                ],
                dtype=float,
            )
            return pd.DataFrame(values, index=dates, columns=columns)
        if requested == ("BRK-B",):
            return pd.DataFrame(
                {
                    "Close": [30, 31, 32, 33],
                    "High": [32, 33, 34, 35],
                    "Low": [28, 29, 30, 31],
                    "Volume": [2000, 2100, 2200, 2300],
                },
                index=dates,
            )
        raise AssertionError(f"unexpected request: {requested}")

    result = data.download_adjusted_ohlcv(
        ["brk.b", "AAPL", "BRK-B"],
        "2020-01-01",
        "2020-01-03",
        chunk_size=2,
        max_retries=0,
        download_func=fake_download,
    )

    expected_columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Volume"], ["BRK-B", "AAPL"]],
        names=["field", "ticker"],
    )
    assert result.columns.equals(expected_columns)
    assert result.index.tolist() == pd.to_datetime(
        ["2020-01-01", "2020-01-02"]
    ).tolist()
    # Existing batch values are retained while the missing volume is filled by
    # the single-symbol fallback.
    assert result.loc[pd.Timestamp("2020-01-01"), ("Close", "BRK-B")] == 21
    assert result.loc[pd.Timestamp("2020-01-02"), ("Volume", "BRK-B")] == 2200
    assert result.loc[pd.Timestamp("2020-01-02"), ("Close", "AAPL")] == 12
    assert [call[0] for call in calls] == [("BRK-B", "AAPL"), ("BRK-B",)]
    assert all(call[1]["auto_adjust"] is True for call in calls)
    assert all(call[1]["threads"] is False for call in calls)


def test_adjusted_ohlcv_retries_and_extracts_field_first_multiindex():
    attempts = 0

    def flaky_download(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.Timeout("temporary")
        columns = pd.MultiIndex.from_tuples(
            [
                ("Low", "MSFT"),
                ("Close", "MSFT"),
                ("Volume", "MSFT"),
                ("High", "MSFT"),
            ]
        )
        return pd.DataFrame(
            [[99.0, 100.0, 5000.0, 101.0]],
            index=pd.to_datetime(["2022-02-01"]),
            columns=columns,
        )

    result = data.download_adjusted_ohlcv(
        "MSFT",
        "2022-02-01",
        "2022-02-02",
        max_retries=1,
        backoff_factor=0,
        download_func=flaky_download,
        sleep_func=lambda _: None,
    )

    assert attempts == 2
    assert result.loc[pd.Timestamp("2022-02-01"), ("High", "MSFT")] == 101
    assert result.attrs["failed_tickers"] == []


def test_adjusted_ohlcv_cache_round_trip_and_explicit_metadata(tmp_path):
    cache = tmp_path / "ohlcv.csv"
    calls = 0

    def fake_download(**kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Close": [10.0, 11.0],
                "High": [10.5, 11.5],
                "Low": [9.5, 10.5],
                "Volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

    first = data.download_adjusted_ohlcv(
        "AAPL",
        "2024-01-02",
        "2024-01-04",
        cache_path=cache,
        max_retries=0,
        download_func=fake_download,
    )
    second = data.download_adjusted_ohlcv(
        "AAPL",
        "2024-01-02",
        "2024-01-04",
        cache_path=cache,
        download_func=lambda **_: pytest.fail("cache miss"),
    )

    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert calls == 1
    metadata_text = data.snapshot_metadata_path(cache).read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert "api_key" not in metadata_text.lower()
    assert metadata["source"] == "yahoo_finance"
    assert metadata["dataset"] == "daily_adjusted_ohlcv"
    assert metadata["adjustment"]["auto_adjust"] is True
    assert metadata["field_schema"]["fields"] == [
        "Close",
        "High",
        "Low",
        "Volume",
    ]
    assert metadata["field_schema"]["column_levels"] == ["field", "ticker"]


def test_adjusted_ohlcv_strict_mode_reports_missing_fields():
    dates = pd.to_datetime(["2024-01-02"])

    def fake_download(**kwargs):
        requested = tuple(kwargs["tickers"])
        if requested == ("AAPL", "MSFT"):
            columns = pd.MultiIndex.from_tuples(
                [
                    (field, ticker)
                    for ticker in requested
                    for field in (["Close", "High", "Low", "Volume"] if ticker == "AAPL" else ["Close", "High", "Low"])
                ]
            )
            return pd.DataFrame(
                [np.arange(1, len(columns) + 1, dtype=float)],
                index=dates,
                columns=columns,
            )
        if requested == ("MSFT",):
            return pd.DataFrame(
                {"Close": [1.0], "High": [2.0], "Low": [0.5]}, index=dates
            )
        raise AssertionError(f"unexpected request: {requested}")

    partial = data.download_adjusted_ohlcv(
        ["AAPL", "MSFT"],
        "2024-01-02",
        "2024-01-03",
        max_retries=0,
        download_func=fake_download,
    )
    assert partial.attrs["failed_tickers"] == ["MSFT"]
    assert partial.attrs["missing_fields_by_ticker"] == {"MSFT": ["Volume"]}

    with pytest.raises(data.DataDownloadError, match=r"MSFT \(Volume\)"):
        data.download_adjusted_ohlcv(
            ["AAPL", "MSFT"],
            "2024-01-02",
            "2024-01-03",
            max_retries=0,
            strict=True,
            download_func=fake_download,
        )


def test_snapshot_rejects_credentials_in_metadata(tmp_path):
    frame = pd.DataFrame(
        {"x": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="date")
    )
    with pytest.raises(ValueError, match="credentials"):
        data.save_timeseries_snapshot(
            frame, tmp_path / "unsafe.csv", metadata={"fred_api_key": "secret"}
        )


@pytest.mark.parametrize("contents", ["", "short", "A" * 32, "a" * 31 + "!"])
def test_fred_key_reader_rejects_empty_or_invalid(tmp_path, contents):
    path = tmp_path / "fred-key.txt"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(data.InvalidFREDApiKey) as error:
        data.read_fred_api_key(path)
    assert contents not in str(error.value) or not contents


def test_fred_download_retries_parses_missing_and_respects_exclusive_end(tmp_path):
    key = "a" * 32
    key_path = tmp_path / "fred_api_key.txt"
    key_path.write_text(key + "\n", encoding="utf-8")
    session = SequenceSession(
        [
            FakeResponse(status_code=503),
            FakeResponse(
                payload={
                    "observations": [
                        {"date": "2020-01-01", "value": "1.5"},
                        {"date": "2020-01-02", "value": "."},
                        {"date": "2020-01-03", "value": "9.9"},
                    ]
                }
            ),
        ]
    )

    result = data.download_fred_observations(
        {"fed_funds": "DFF"},
        "2020-01-01",
        "2020-01-03",
        api_key_path=key_path,
        session=session,
        max_retries=1,
        sleep_func=lambda _: None,
    )

    assert result.index.tolist() == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
    assert result.loc[pd.Timestamp("2020-01-01"), "fed_funds"] == 1.5
    assert np.isnan(result.loc[pd.Timestamp("2020-01-02"), "fed_funds"])
    params = session.calls[-1][1]["params"]
    assert params["observation_end"] == "2020-01-02"
    assert params["api_key"] == key


def test_fred_cache_metadata_never_contains_key(tmp_path):
    key = "b" * 32
    cache = tmp_path / "macro.csv"
    session = SequenceSession(
        [FakeResponse(payload={"observations": [{"date": "2021-01-01", "value": "2"}]})]
    )
    data.download_fred_series(
        {"ten_year": "DGS10"},
        "2021-01-01",
        "2021-01-02",
        api_key=key,
        session=session,
        cache_path=cache,
        max_retries=0,
    )

    sidecar = data.snapshot_metadata_path(cache).read_text(encoding="utf-8")
    assert key not in sidecar
    assert "api_key" not in sidecar.lower()
