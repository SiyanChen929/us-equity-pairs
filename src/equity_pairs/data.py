"""Reliable market and macroeconomic data acquisition.

The public functions in this module deliberately keep credentials separate from
cached data.  Yahoo downloads use adjusted ``Close`` values (``auto_adjust=True``)
and all date ranges follow Python slicing semantics: ``start`` is included and
``end`` is excluded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
YAHOO_OHLCV_FIELDS = ("Close", "High", "Low", "Volume")


class DataDownloadError(RuntimeError):
    """Raised when a remote data source cannot provide a usable response."""


class InvalidFREDApiKey(ValueError):
    """Raised when a FRED API key is absent or malformed."""


def normalize_yahoo_ticker(ticker: Any) -> str:
    """Return a canonical Yahoo Finance symbol.

    S&P symbols use a period for share classes while Yahoo uses a hyphen, for
    example ``BRK.B`` becomes ``BRK-B``.  Symbols are stripped and upper-cased.
    """

    if ticker is None or ticker is pd.NA:
        raise ValueError("ticker must be a non-empty string")
    try:
        missing = pd.isna(ticker)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("ticker must be a non-empty string")
    normalized = str(ticker).strip().upper().replace(".", "-")
    if not normalized or normalized in {"NAN", "NONE"}:
        raise ValueError("ticker must be a non-empty string")
    if any(character.isspace() for character in normalized):
        raise ValueError("ticker may not contain whitespace")
    return normalized


def normalize_yahoo_tickers(tickers: Iterable[Any] | str) -> tuple[str, ...]:
    """Normalize symbols and remove duplicates without changing input order."""

    values: Iterable[Any] = [tickers] if isinstance(tickers, str) else tickers
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in values:
        symbol = normalize_yahoo_ticker(ticker)
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    if not normalized:
        raise ValueError("at least one ticker is required")
    return tuple(normalized)


def _date_bounds(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("start and end must be valid dates") from exc
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("start and end must be valid dates")
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    start_ts = start_ts.normalize()
    end_ts = end_ts.normalize()
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")
    return start_ts, end_ts


def _retry_delay(
    attempt: int,
    backoff_factor: float,
    response: Any | None = None,
) -> float:
    retry_after = None
    if response is not None:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, min(float(retry_after), 60.0))
        except (TypeError, ValueError):
            pass
    return max(0.0, backoff_factor) * (2**attempt)


def _safe_get(
    session: requests.Session | Any,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    request_name: str = "HTTP endpoint",
    sleep_func: Callable[[float], None] = time.sleep,
) -> Any:
    """GET an endpoint with bounded exponential retries and safe errors.

    Exception messages intentionally omit the request parameters.  This matters
    for FRED, whose API key is sent as a query parameter.
    """

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    last_status: int | None = None
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                params=dict(params or {}),
                headers=dict(headers or {}),
                timeout=timeout,
            )
        except requests.RequestException:
            if attempt >= max_retries:
                # Do not retain a requests exception as a displayed cause: its
                # prepared URL may include sensitive query parameters.
                raise DataDownloadError(
                    f"{request_name} request failed after {attempt + 1} attempts"
                ) from None
            sleep_func(_retry_delay(attempt, backoff_factor))
            continue

        status_code = int(getattr(response, "status_code", 200))
        last_status = status_code
        if status_code in RETRYABLE_HTTP_STATUS and attempt < max_retries:
            sleep_func(_retry_delay(attempt, backoff_factor, response))
            continue
        if status_code >= 400:
            raise DataDownloadError(
                f"{request_name} returned HTTP status {status_code}"
            )
        return response

    # The loop is exhaustive; this guard keeps static analyzers and unusual
    # mocked responses honest without exposing request parameters.
    raise DataDownloadError(
        f"{request_name} failed"
        + (f" with HTTP status {last_status}" if last_status is not None else "")
    )


def fetch_sp500_constituents(
    *,
    session: requests.Session | Any | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch the current S&P 500 constituents from Wikipedia.

    Returns the source and Yahoo-compatible tickers, issuer CIK, security,
    sector and sub-industry.  CIK permits same-issuer share classes to be
    excluded from pair screening.
    """

    http = session if session is not None else requests.Session()
    response = _safe_get(
        http,
        SP500_WIKIPEDIA_URL,
        headers={"User-Agent": "equity-pairs-research/1.0"},
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        request_name="S&P 500 constituent source",
        sleep_func=sleep_func,
    )
    try:
        tables = pd.read_html(StringIO(response.text))
    except (ValueError, ImportError) as exc:
        raise DataDownloadError("could not parse the S&P 500 constituent table") from exc

    required = {
        "symbol": "source_ticker",
        "security": "security",
        "gics sector": "sector",
        "gics sub-industry": "sub_industry",
        "cik": "cik",
    }
    constituent_table: pd.DataFrame | None = None
    selected_columns: dict[Any, str] = {}
    for table in tables:
        canonical = {str(column).strip().lower(): column for column in table.columns}
        if all(name in canonical for name in required):
            constituent_table = table
            selected_columns = {
                canonical[source]: destination for source, destination in required.items()
            }
            break
    if constituent_table is None:
        raise DataDownloadError("S&P 500 constituent table has an unexpected schema")

    result = constituent_table.loc[:, list(selected_columns)].rename(
        columns=selected_columns
    )
    for column in result.columns:
        result[column] = result[column].astype("string").str.strip()
    result = result.dropna(subset=list(required.values()))
    result = result[result["source_ticker"].ne("")]
    try:
        result.insert(0, "ticker", result["source_ticker"].map(normalize_yahoo_ticker))
    except ValueError as exc:
        raise DataDownloadError("S&P 500 constituent table contains an invalid ticker") from exc
    result = result.drop_duplicates(subset="ticker", keep="first")
    result = result.sort_values("ticker", kind="stable").reset_index(drop=True)
    return result.loc[:, ["ticker", "source_ticker", "security", "sector", "sub_industry", "cik"]]


def snapshot_metadata_path(csv_path: str | Path) -> Path:
    """Return the sidecar JSON path used by a CSV snapshot."""

    path = Path(csv_path)
    return path.with_name(path.name + ".meta.json")


_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "authorization",
        "credential",
        "credentials",
    }
)


def _assert_metadata_has_no_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized_key in _SENSITIVE_METADATA_KEYS or normalized_key.endswith(
                ("_api_key", "_token", "_secret", "_password")
            ):
                raise ValueError("snapshot metadata may not contain credentials")
            _assert_metadata_has_no_secrets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_metadata_has_no_secrets(child)


def save_timeseries_snapshot(
    frame: pd.DataFrame,
    csv_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a time-series CSV and a credential-free JSON sidecar."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("time-series snapshots require a DatetimeIndex")
    user_metadata = dict(metadata or {})
    _assert_metadata_has_no_secrets(user_metadata)
    try:
        json.dumps(user_metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("snapshot metadata must be JSON serializable") from exc

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = snapshot_metadata_path(path)
    csv_temp = path.with_name(path.name + ".tmp")
    metadata_temp = metadata_path.with_name(metadata_path.name + ".tmp")

    snapshot = frame.sort_index().copy()
    snapshot.index.name = snapshot.index.name or "date"
    snapshot.to_csv(csv_temp)
    digest = hashlib.sha256(csv_temp.read_bytes()).hexdigest()

    full_metadata: dict[str, Any] = dict(user_metadata)
    full_metadata.update(
        {
            "snapshot_schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(snapshot.shape[0]),
            "columns": [str(column) for column in snapshot.columns],
            "sha256": digest,
        }
    )
    _assert_metadata_has_no_secrets(full_metadata)
    metadata_temp.write_text(
        json.dumps(full_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_temp.replace(path)
    metadata_temp.replace(metadata_path)
    return full_metadata


def load_timeseries_snapshot(
    csv_path: str | Path,
    *,
    verify_hash: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV snapshot and its sidecar, optionally checking its hash."""

    path = Path(csv_path)
    metadata_path = snapshot_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"snapshot is incomplete: {path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataDownloadError("snapshot metadata is unreadable") from exc
    _assert_metadata_has_no_secrets(metadata)
    if verify_hash:
        expected_hash = metadata.get("sha256")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected_hash or expected_hash != actual_hash:
            raise DataDownloadError("snapshot checksum does not match its metadata")
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataDownloadError("snapshot CSV is unreadable") from exc
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    return frame.sort_index(), metadata


def _normalise_daily_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        result = frame.copy()
        result.index = pd.DatetimeIndex([], name="date")
        return result
    converted = pd.to_datetime(frame.index, errors="coerce")
    valid = ~pd.isna(converted)
    result = frame.loc[valid].copy()
    converted = pd.DatetimeIndex(converted[valid])
    if converted.tz is not None:
        converted = converted.tz_localize(None)
    result.index = converted.normalize()
    result.index.name = "date"
    result = result.groupby(level=0, sort=True).last()
    return result.sort_index()


def _extract_adjusted_close(raw: Any, requested: tuple[str, ...]) -> pd.DataFrame:
    """Extract adjusted Close columns across yfinance's output layouts."""

    if raw is None or not isinstance(raw, (pd.DataFrame, pd.Series)):
        return pd.DataFrame(columns=list(requested), dtype=float)
    if isinstance(raw, pd.Series):
        if len(requested) != 1:
            return pd.DataFrame(columns=list(requested), dtype=float)
        raw = raw.to_frame(name="Close")
    if raw.empty:
        return pd.DataFrame(index=raw.index, columns=list(requested), dtype=float)

    close: pd.DataFrame | pd.Series | None = None
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            values = list(raw.columns.get_level_values(level).unique())
            match = next(
                (value for value in values if str(value).strip().lower() == "close"),
                None,
            )
            if match is not None:
                close = raw.xs(match, axis=1, level=level, drop_level=True)
                break
    else:
        labels = {str(column).strip().lower(): column for column in raw.columns}
        if "close" in labels and len(requested) == 1:
            close = raw.loc[:, labels["close"]]
        elif "adj close" in labels and len(requested) == 1:
            # Compatibility with older/mocked yfinance output.  Production calls
            # still request auto_adjust=True, where Close is already adjusted.
            close = raw.loc[:, labels["adj close"]]
        else:
            close = raw

    if close is None:
        return pd.DataFrame(index=raw.index, columns=list(requested), dtype=float)
    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0])

    extracted = pd.DataFrame(index=close.index)
    for column in close.columns:
        label = column[-1] if isinstance(column, tuple) else column
        try:
            symbol = normalize_yahoo_ticker(label)
        except ValueError:
            continue
        if symbol not in requested:
            continue
        values = pd.to_numeric(close[column], errors="coerce")
        if symbol in extracted:
            extracted[symbol] = extracted[symbol].combine_first(values)
        else:
            extracted[symbol] = values
    extracted = _normalise_daily_index(extracted)
    return extracted.reindex(columns=list(requested))


def _merge_non_null(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty and len(left.columns) == 0:
        return right.copy()
    if right.empty and len(right.columns) == 0:
        return left.copy()
    return left.combine_first(right).sort_index()


def _call_yfinance(
    download_func: Callable[..., Any],
    tickers: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    repair: bool,
) -> pd.DataFrame:
    raw = download_func(
        tickers=list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=timeout,
        repair=repair,
    )
    return _extract_adjusted_close(raw, tickers)


def _download_yahoo_group(
    tickers: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    download_func: Callable[..., Any],
    timeout: float,
    max_retries: int,
    backoff_factor: float,
    sleep_func: Callable[[float], None],
    repair: bool,
) -> pd.DataFrame:
    accumulated = pd.DataFrame()
    for attempt in range(max_retries + 1):
        try:
            attempt_frame = _call_yfinance(download_func, tickers, start, end, timeout, repair)
            accumulated = _merge_non_null(accumulated, attempt_frame)
        except Exception as exc:  # yfinance raises several backend exception types
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        successful = {
            ticker
            for ticker in accumulated.columns
            if accumulated[ticker].notna().any()
        }
        if successful == set(tickers):
            break
        if attempt < max_retries:
            sleep_func(_retry_delay(attempt, backoff_factor))
    return accumulated


def download_adjusted_close(
    tickers: Iterable[Any] | str,
    start: Any,
    end: Any,
    *,
    chunk_size: int = 50,
    timeout: float = 30.0,
    max_retries: int = 2,
    backoff_factor: float = 0.5,
    inter_chunk_delay: float = 0.0,
    repair: bool = True,
    strict: bool = False,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    download_func: Callable[..., Any] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download adjusted Yahoo daily closes into a deterministic wide frame.

    Batch requests are chunked and retried.  Any ticker still missing after a
    batch request is retried individually.  Columns preserve normalized input
    order, duplicate symbols are removed, dates are sorted, and the output is
    explicitly filtered to ``[start, end)``.

    When ``cache_path`` is supplied, a matching CSV/JSON snapshot is reused
    unless ``refresh`` is true.  Metadata contains request details but never
    credentials.  Missing symbols remain as all-NaN columns unless ``strict`` is
    true; their names are also available in ``frame.attrs['failed_tickers']``.
    """

    symbols = normalize_yahoo_tickers(tickers)
    start_ts, end_ts = _date_bounds(start, end)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if inter_chunk_delay < 0:
        raise ValueError("inter_chunk_delay cannot be negative")

    request_metadata = {
        "source": "yahoo_finance",
        "adjusted_close": True,
        "price_repair": bool(repair),
        "start": start_ts.strftime("%Y-%m-%d"),
        "end_exclusive": end_ts.strftime("%Y-%m-%d"),
        "tickers": list(symbols),
    }
    if cache_path is not None and not refresh:
        try:
            cached, metadata = load_timeseries_snapshot(cache_path)
        except (FileNotFoundError, DataDownloadError, ValueError):
            pass
        else:
            if all(metadata.get(key) == value for key, value in request_metadata.items()):
                cached = cached.reindex(columns=list(symbols))
                cached = cached.loc[(cached.index >= start_ts) & (cached.index < end_ts)]
                failed = [ticker for ticker in symbols if not cached[ticker].notna().any()]
                cached.attrs["failed_tickers"] = failed
                if strict and failed:
                    raise DataDownloadError(
                        "cached Yahoo snapshot has no usable data for: "
                        + ", ".join(failed)
                    )
                return cached

    downloader = download_func if download_func is not None else yf.download
    pieces: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), chunk_size):
        chunk = symbols[offset : offset + chunk_size]
        batch = _download_yahoo_group(
            chunk,
            start_ts,
            end_ts,
            download_func=downloader,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            sleep_func=sleep_func,
            repair=repair,
        )
        missing = [
            ticker
            for ticker in chunk
            if ticker not in batch or not batch[ticker].notna().any()
        ]
        for ticker in missing:
            fallback = _download_yahoo_group(
                (ticker,),
                start_ts,
                end_ts,
                download_func=downloader,
                timeout=timeout,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                sleep_func=sleep_func,
                repair=repair,
            )
            batch = _merge_non_null(batch, fallback)
        pieces.append(batch)
        if inter_chunk_delay and offset + chunk_size < len(symbols):
            sleep_func(inter_chunk_delay)

    if pieces:
        prices = pd.concat(pieces, axis=1)
    else:  # normalize_yahoo_tickers prevents this, kept for defensive clarity
        prices = pd.DataFrame()
    prices = _normalise_daily_index(prices)
    prices = prices.loc[(prices.index >= start_ts) & (prices.index < end_ts)]
    prices = prices.reindex(columns=list(symbols)).apply(pd.to_numeric, errors="coerce")
    failed = [ticker for ticker in symbols if not prices[ticker].notna().any()]
    prices.attrs["failed_tickers"] = failed
    if len(failed) == len(symbols):
        raise DataDownloadError("Yahoo Finance returned no usable adjusted close data")
    if strict and failed:
        raise DataDownloadError(
            "Yahoo Finance returned no usable data for: " + ", ".join(failed)
        )

    if cache_path is not None:
        metadata = dict(request_metadata)
        metadata["failed_tickers"] = failed
        save_timeseries_snapshot(prices, cache_path, metadata=metadata)
    return prices


def _ohlcv_columns(tickers: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [YAHOO_OHLCV_FIELDS, tickers], names=["field", "ticker"]
    )


def _canonical_ohlcv_field(value: Any) -> str | None:
    label = re.sub(r"\s+", " ", str(value).strip().lower())
    aliases = {
        "close": "Close",
        # Compatibility with older/mocked yfinance layouts.  Production calls
        # use auto_adjust=True, where adjusted prices are exposed as Close.
        "adj close": "Close",
        "adjusted close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    return aliases.get(label)


def _extract_adjusted_ohlcv(raw: Any, requested: tuple[str, ...]) -> pd.DataFrame:
    """Extract adjusted OHLCV across yfinance's single/multi-ticker layouts."""

    columns = _ohlcv_columns(requested)
    if raw is None or not isinstance(raw, (pd.DataFrame, pd.Series)):
        return pd.DataFrame(columns=columns, dtype=float)
    if isinstance(raw, pd.Series):
        if len(requested) != 1:
            return pd.DataFrame(columns=columns, dtype=float)
        field = _canonical_ohlcv_field(raw.name) or "Close"
        raw = raw.to_frame(name=field)

    extracted = pd.DataFrame(index=raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        requested_set = set(requested)
        field_scores: list[tuple[int, int, int]] = []
        for level in range(raw.columns.nlevels):
            score = sum(
                _canonical_ohlcv_field(value) is not None
                for value in raw.columns.get_level_values(level).unique()
            )
            # Tickers such as LOW are also field names.  Prefer the candidate
            # whose *other* level best matches the requested symbol universe.
            other_level_overlap = 0
            for other_level in range(raw.columns.nlevels):
                if other_level == level:
                    continue
                normalized: set[str] = set()
                for value in raw.columns.get_level_values(other_level).unique():
                    try:
                        normalized.add(normalize_yahoo_ticker(value))
                    except ValueError:
                        continue
                other_level_overlap = max(
                    other_level_overlap, len(normalized & requested_set)
                )
            field_scores.append((score, other_level_overlap, -level))
        field_score, _, negative_field_level = max(
            field_scores, default=(0, 0, 1)
        )
        field_level = -negative_field_level
        if field_score == 0:
            return _normalise_daily_index(extracted).reindex(columns=columns)

        ticker_scores: list[tuple[int, int]] = []
        for level in range(raw.columns.nlevels):
            if level == field_level:
                continue
            normalized: set[str] = set()
            for value in raw.columns.get_level_values(level).unique():
                try:
                    normalized.add(normalize_yahoo_ticker(value))
                except ValueError:
                    continue
            ticker_scores.append((len(normalized & requested_set), level))
        ticker_score, ticker_level = max(ticker_scores, default=(0, -1))

        for position, column in enumerate(raw.columns):
            labels = column if isinstance(column, tuple) else (column,)
            field = _canonical_ohlcv_field(labels[field_level])
            if field is None:
                continue
            if ticker_score > 0:
                try:
                    ticker = normalize_yahoo_ticker(labels[ticker_level])
                except ValueError:
                    continue
            elif len(requested) == 1:
                ticker = requested[0]
            else:
                continue
            if ticker not in requested_set:
                continue
            values = pd.to_numeric(raw.iloc[:, position], errors="coerce")
            key = (field, ticker)
            if key in extracted:
                extracted[key] = extracted[key].combine_first(values)
            else:
                extracted[key] = values
    elif len(requested) == 1:
        ticker = requested[0]
        for position, column in enumerate(raw.columns):
            field = _canonical_ohlcv_field(column)
            if field is None:
                continue
            values = pd.to_numeric(raw.iloc[:, position], errors="coerce")
            key = (field, ticker)
            if key in extracted:
                extracted[key] = extracted[key].combine_first(values)
            else:
                extracted[key] = values

    extracted = _normalise_daily_index(extracted)
    extracted.columns = pd.MultiIndex.from_tuples(
        list(extracted.columns), names=["field", "ticker"]
    )
    return extracted.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _call_yfinance_ohlcv(
    download_func: Callable[..., Any],
    tickers: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeout: float,
    repair: bool,
) -> pd.DataFrame:
    raw = download_func(
        tickers=list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=timeout,
        repair=repair,
    )
    return _extract_adjusted_ohlcv(raw, tickers)


def _missing_ohlcv_fields(frame: pd.DataFrame, ticker: str) -> list[str]:
    return [
        field
        for field in YAHOO_OHLCV_FIELDS
        if (field, ticker) not in frame.columns
        or not frame[(field, ticker)].notna().any()
    ]


def _download_yahoo_ohlcv_group(
    tickers: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    download_func: Callable[..., Any],
    timeout: float,
    max_retries: int,
    backoff_factor: float,
    sleep_func: Callable[[float], None],
    repair: bool,
) -> pd.DataFrame:
    accumulated = pd.DataFrame(columns=_ohlcv_columns(tickers), dtype=float)
    for attempt in range(max_retries + 1):
        try:
            attempt_frame = _call_yfinance_ohlcv(
                download_func, tickers, start, end, timeout, repair
            )
            accumulated = _merge_non_null(accumulated, attempt_frame)
        except Exception as exc:  # yfinance raises several backend exception types
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        if not any(_missing_ohlcv_fields(accumulated, ticker) for ticker in tickers):
            break
        if attempt < max_retries:
            sleep_func(_retry_delay(attempt, backoff_factor))
    return accumulated.reindex(columns=_ohlcv_columns(tickers))


def _flatten_ohlcv_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    flattened.columns = [
        json.dumps([str(field), str(ticker)], separators=(",", ":"))
        for field, ticker in flattened.columns
    ]
    return flattened


def _restore_ohlcv_snapshot(
    frame: pd.DataFrame, tickers: tuple[str, ...]
) -> pd.DataFrame:
    desired = _ohlcv_columns(tickers)
    encoded = [
        json.dumps([str(field), str(ticker)], separators=(",", ":"))
        for field, ticker in desired
    ]
    missing_columns = [column for column in encoded if column not in frame.columns]
    if missing_columns:
        raise DataDownloadError("cached Yahoo OHLCV snapshot has an invalid schema")
    restored = frame.reindex(columns=encoded).copy()
    restored.columns = desired
    return restored.apply(pd.to_numeric, errors="coerce")


def _ohlcv_failure_details(
    frame: pd.DataFrame, tickers: tuple[str, ...]
) -> dict[str, list[str]]:
    return {
        ticker: missing
        for ticker in tickers
        if (missing := _missing_ohlcv_fields(frame, ticker))
    }


def _format_ohlcv_failures(failures: Mapping[str, list[str]]) -> str:
    return ", ".join(
        f"{ticker} ({'/'.join(fields)})" for ticker, fields in failures.items()
    )


def download_adjusted_ohlcv(
    tickers: Iterable[Any] | str,
    start: Any,
    end: Any,
    *,
    chunk_size: int = 50,
    timeout: float = 30.0,
    max_retries: int = 2,
    backoff_factor: float = 0.5,
    inter_chunk_delay: float = 0.0,
    repair: bool = True,
    strict: bool = False,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    download_func: Callable[..., Any] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download adjusted Yahoo daily Close/High/Low/Volume observations.

    The result has deterministic ``(field, ticker)`` MultiIndex columns, with
    fields ordered ``Close``, ``High``, ``Low``, ``Volume`` and tickers in
    normalized input order.  ``auto_adjust=True`` makes the three price fields
    adjusted; Volume is retained as the corresponding non-price observation.

    Batch requests are chunked and retried, then incomplete tickers are retried
    individually.  A ticker is complete only if every requested field has at
    least one usable value.  In non-strict mode, partial ticker data is returned
    when at least one ticker is complete; details are stored in
    ``frame.attrs['missing_fields_by_ticker']`` and ``failed_tickers``.  Strict
    mode rejects any incomplete ticker.

    A matching credential-free CSV/JSON snapshot is reused when ``cache_path``
    is supplied and ``refresh`` is false.  The sidecar explicitly records the
    data source, adjustment policy, field layout and cache encoding.
    """

    symbols = normalize_yahoo_tickers(tickers)
    start_ts, end_ts = _date_bounds(start, end)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if inter_chunk_delay < 0:
        raise ValueError("inter_chunk_delay cannot be negative")

    request_metadata = {
        "source": "yahoo_finance",
        "dataset": "daily_adjusted_ohlcv",
        "adjustment": {
            "auto_adjust": True,
            "adjusted_price_fields": ["Close", "High", "Low"],
            "non_price_fields": ["Volume"],
        },
        "field_schema": {
            "column_levels": ["field", "ticker"],
            "fields": list(YAHOO_OHLCV_FIELDS),
            "cache_encoding": "json_tuple_v1",
        },
        "price_repair": bool(repair),
        "start": start_ts.strftime("%Y-%m-%d"),
        "end_exclusive": end_ts.strftime("%Y-%m-%d"),
        "tickers": list(symbols),
    }
    if cache_path is not None and not refresh:
        try:
            cached_flat, metadata = load_timeseries_snapshot(cache_path)
            if not all(
                metadata.get(key) == value for key, value in request_metadata.items()
            ):
                raise DataDownloadError("cached Yahoo OHLCV request does not match")
            cached = _restore_ohlcv_snapshot(cached_flat, symbols)
        except (FileNotFoundError, DataDownloadError, ValueError):
            pass
        else:
            cached = cached.loc[
                (cached.index >= start_ts) & (cached.index < end_ts)
            ]
            failures = _ohlcv_failure_details(cached, symbols)
            cached.attrs["missing_fields_by_ticker"] = failures
            cached.attrs["failed_tickers"] = list(failures)
            if strict and failures:
                raise DataDownloadError(
                    "cached Yahoo OHLCV snapshot is incomplete for: "
                    + _format_ohlcv_failures(failures)
                )
            return cached

    downloader = download_func if download_func is not None else yf.download
    pieces: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), chunk_size):
        chunk = symbols[offset : offset + chunk_size]
        batch = _download_yahoo_ohlcv_group(
            chunk,
            start_ts,
            end_ts,
            download_func=downloader,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            sleep_func=sleep_func,
            repair=repair,
        )
        incomplete = [
            ticker for ticker in chunk if _missing_ohlcv_fields(batch, ticker)
        ]
        for ticker in incomplete:
            fallback = _download_yahoo_ohlcv_group(
                (ticker,),
                start_ts,
                end_ts,
                download_func=downloader,
                timeout=timeout,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                sleep_func=sleep_func,
                repair=repair,
            )
            batch = _merge_non_null(batch, fallback)
        pieces.append(batch.reindex(columns=_ohlcv_columns(chunk)))
        if inter_chunk_delay and offset + chunk_size < len(symbols):
            sleep_func(inter_chunk_delay)

    observations = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
    observations = _normalise_daily_index(observations)
    observations = observations.loc[
        (observations.index >= start_ts) & (observations.index < end_ts)
    ]
    observations = observations.reindex(columns=_ohlcv_columns(symbols)).apply(
        pd.to_numeric, errors="coerce"
    )
    failures = _ohlcv_failure_details(observations, symbols)
    observations.attrs["missing_fields_by_ticker"] = failures
    observations.attrs["failed_tickers"] = list(failures)
    if len(failures) == len(symbols):
        raise DataDownloadError("Yahoo Finance returned no complete adjusted OHLCV data")
    if strict and failures:
        raise DataDownloadError(
            "Yahoo Finance returned incomplete data for: "
            + _format_ohlcv_failures(failures)
        )

    if cache_path is not None:
        metadata = dict(request_metadata)
        metadata["failed_tickers"] = list(failures)
        metadata["missing_fields_by_ticker"] = failures
        save_timeseries_snapshot(
            _flatten_ohlcv_snapshot(observations), cache_path, metadata=metadata
        )
    return observations


_FRED_KEY_PATTERN = re.compile(r"^[a-z0-9]{32}$")
_FRED_SERIES_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


def validate_fred_api_key(api_key: Any) -> str:
    """Validate and return a FRED API key without exposing it in errors."""

    if not isinstance(api_key, str):
        raise InvalidFREDApiKey("FRED API key is missing or invalid")
    candidate = api_key.strip()
    if not _FRED_KEY_PATTERN.fullmatch(candidate):
        raise InvalidFREDApiKey("FRED API key is missing or invalid")
    return candidate


def read_fred_api_key(path: str | Path) -> str:
    """Read a FRED API key file and reject empty or malformed content."""

    key_path = Path(path)
    try:
        contents = key_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidFREDApiKey(f"could not read FRED API key file: {key_path}") from exc
    return validate_fred_api_key(contents)


def download_fred_observations(
    series: Mapping[str, str],
    start: Any,
    end: Any,
    *,
    api_key: str | None = None,
    api_key_path: str | Path | None = None,
    session: requests.Session | Any | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    cache_path: str | Path | None = None,
    refresh: bool = False,
    sleep_func: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download FRED observations for ``{output_name: series_id}`` mappings.

    The returned frame uses the union of observation dates, preserves mapping
    order, converts FRED's ``'.'`` missing marker to NaN, and filters dates to
    ``[start, end)``.  Supply exactly one of ``api_key`` and ``api_key_path``.
    """

    if not isinstance(series, Mapping) or not series:
        raise ValueError("series must be a non-empty mapping of name to FRED series ID")
    if (api_key is None) == (api_key_path is None):
        raise ValueError("supply exactly one of api_key and api_key_path")
    key = (
        validate_fred_api_key(api_key)
        if api_key is not None
        else read_fred_api_key(api_key_path)  # type: ignore[arg-type]
    )
    start_ts, end_ts = _date_bounds(start, end)

    names: list[str] = []
    identifiers: list[str] = []
    for output_name, series_id in series.items():
        name = str(output_name).strip()
        identifier = str(series_id).strip()
        if not name:
            raise ValueError("FRED output column names must be non-empty")
        if not _FRED_SERIES_PATTERN.fullmatch(identifier):
            raise ValueError(f"invalid FRED series ID for output column {name!r}")
        names.append(name)
        identifiers.append(identifier)
    if len(set(names)) != len(names):
        raise ValueError("FRED output column names must be unique")

    request_metadata = {
        "source": "fred",
        "start": start_ts.strftime("%Y-%m-%d"),
        "end_exclusive": end_ts.strftime("%Y-%m-%d"),
        "series": dict(zip(names, identifiers, strict=True)),
    }
    if cache_path is not None and not refresh:
        try:
            cached, metadata = load_timeseries_snapshot(cache_path)
        except (FileNotFoundError, DataDownloadError, ValueError):
            pass
        else:
            if all(metadata.get(key_name) == value for key_name, value in request_metadata.items()):
                cached = cached.reindex(columns=names)
                return cached.loc[
                    (cached.index >= start_ts) & (cached.index < end_ts)
                ]

    http = session if session is not None else requests.Session()
    end_inclusive = end_ts - pd.Timedelta(days=1)
    downloaded: list[pd.Series] = []
    for name, identifier in zip(names, identifiers, strict=True):
        response = _safe_get(
            http,
            FRED_OBSERVATIONS_URL,
            params={
                "series_id": identifier,
                "api_key": key,
                "file_type": "json",
                "observation_start": start_ts.strftime("%Y-%m-%d"),
                "observation_end": end_inclusive.strftime("%Y-%m-%d"),
            },
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            request_name=f"FRED series {identifier}",
            sleep_func=sleep_func,
        )
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise DataDownloadError(
                f"FRED series {identifier} returned invalid JSON"
            ) from exc
        observations = payload.get("observations") if isinstance(payload, Mapping) else None
        if not isinstance(observations, list):
            raise DataDownloadError(
                f"FRED series {identifier} response has an unexpected schema"
            )

        dates: list[Any] = []
        values: list[Any] = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            dates.append(observation.get("date"))
            value = observation.get("value")
            values.append(np.nan if value in {None, ".", ""} else value)
        date_index = pd.to_datetime(dates, errors="coerce")
        value_series = pd.Series(
            pd.to_numeric(values, errors="coerce"), index=date_index, name=name, dtype=float
        )
        value_series = value_series.loc[~value_series.index.isna()]
        value_series.index = pd.DatetimeIndex(value_series.index).normalize()
        value_series = value_series.groupby(level=0, sort=True).last()
        value_series = value_series.loc[
            (value_series.index >= start_ts) & (value_series.index < end_ts)
        ]
        downloaded.append(value_series)

    result = pd.concat(downloaded, axis=1).sort_index().reindex(columns=names)
    result.index = pd.DatetimeIndex(result.index, name="date")
    if cache_path is not None:
        save_timeseries_snapshot(result, cache_path, metadata=request_metadata)
    return result


# A concise alias for callers that refer to a collection of FRED series rather
# than the observation endpoint itself.
download_fred_series = download_fred_observations


__all__ = [
    "DataDownloadError",
    "InvalidFREDApiKey",
    "download_adjusted_close",
    "download_adjusted_ohlcv",
    "download_fred_observations",
    "download_fred_series",
    "fetch_sp500_constituents",
    "load_timeseries_snapshot",
    "normalize_yahoo_ticker",
    "normalize_yahoo_tickers",
    "read_fred_api_key",
    "save_timeseries_snapshot",
    "snapshot_metadata_path",
    "validate_fred_api_key",
]
