"""Time-of-day features shared by the airspace and maritime workflows."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def _to_datetime(s: pd.Series, *, unit: str | None = None, utc: bool = True) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit=unit or "s", utc=utc)
    return pd.to_datetime(s, utc=utc)


def add_hour_and_day_night(
    df: pd.DataFrame,
    *,
    time_column: str,
    unit: str | None = None,
    night_start: int = 20,
    night_end: int = 6,
    hour_column: str = "hour",
    flag_column: str = "is_night",
) -> pd.DataFrame:
    """Add ``hour`` (0-23) and a boolean ``is_night`` flag.

    ``time_column`` may hold datetimes or epoch numbers (``unit`` = ``"s"``, ``"ms"``...).
    Night is ``hour >= night_start or hour <= night_end`` (defaults 20:00-06:00). The legacy
    implementation used ``time.strftime`` row by row; this is vectorised.
    """
    out = df.copy()
    ts = _to_datetime(out[time_column], unit=unit)
    out[hour_column] = ts.dt.hour.astype("int64")
    out[flag_column] = (out[hour_column] >= night_start) | (out[hour_column] <= night_end)
    return out


def filter_by_hour(
    df: pd.DataFrame,
    *,
    time_column: str,
    start_hour: int = 7,
    end_hour: int = 19,
    hour_column: str = "hour",
) -> pd.DataFrame:
    """Keep rows whose local hour satisfies ``start_hour <= hour < end_hour`` (adds ``hour``)."""
    out = df.copy()
    ts = _to_datetime(out[time_column])
    out[hour_column] = ts.dt.hour.astype("int64")
    return out[(out[hour_column] >= start_hour) & (out[hour_column] < end_hour)]


def time_windows(
    times: pd.Series,
    *,
    window: pd.Timedelta | str,
    overlap: pd.Timedelta | str | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield ``(start, end)`` windows covering the span of ``times``.

    ``overlap`` pads each window on both sides so events near a boundary are still paired with
    their neighbours when windows are processed independently
    (see :mod:`~fortress_gis.features.proximity`).
    """
    window_td = pd.Timedelta(window)
    pad = pd.Timedelta(overlap) if overlap is not None else pd.Timedelta(0)
    ts = _to_datetime(times)
    start, stop = ts.min(), ts.max()
    if pd.isna(start):
        return
    cursor = start
    while cursor <= stop:
        yield cursor - pad, cursor + window_td + pad
        cursor = cursor + window_td


def hour_bins(
    hours: pd.Series | np.ndarray,
    *,
    edges: tuple[int, ...] = (0, 6, 12, 18, 24),
    labels: tuple[str, ...] = ("night", "morning", "afternoon", "evening"),
) -> pd.Categorical:
    """Bucket integer hours into named periods."""
    return pd.cut(np.asarray(hours), bins=list(edges), right=False, labels=list(labels))
