"""Seasonal / interannual covariates from wide time-stamped columns.

Satellite composites arrive as one column per acquisition, named ``<prefix>YYYYDDD``. A
:class:`SeasonCalendar` maps each day of year to a season; acquisitions in the same
``year_season`` are averaged; year-over-year change, per-season means and variances follow from
that. Everything is a pandas operation on a wide frame, and geometry columns pass through.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ACQ_PATTERN = re.compile(r"(?P<year>\d{4})(?P<doy>\d{3})")


@dataclass(frozen=True)
class SeasonCalendar:
    """Day-of-year breakpoints and their season labels.

    ``bounds`` are ``(start_inclusive, end_exclusive, label)`` triples. The default reproduces
    the legacy two-season split used for chlorophyll-a composites: winter (DOY 0-104 and 288+)
    and summer (DOY 104-288).
    """

    bounds: tuple[tuple[int, int, str], ...] = field(
        default=((0, 104, "winter"), (104, 288, "summer"), (288, 367, "winter"))
    )

    @property
    def seasons(self) -> list[str]:
        seen: list[str] = []
        for _, _, label in self.bounds:
            if label not in seen:
                seen.append(label)
        return seen

    def season_for(self, day_of_year: int) -> str:
        for start, end, label in self.bounds:
            if start <= day_of_year < end:
                return label
        msg = f"day_of_year {day_of_year} outside calendar bounds"
        raise ValueError(msg)

    @classmethod
    def four_seasons(cls) -> SeasonCalendar:
        """Meteorological northern-hemisphere seasons."""
        return cls(
            bounds=(
                (0, 60, "winter"),
                (60, 152, "spring"),
                (152, 244, "summer"),
                (244, 335, "autumn"),
                (335, 367, "winter"),
            )
        )


def season_for_day_of_year(day_of_year: int, calendar: SeasonCalendar | None = None) -> str:
    return (calendar or SeasonCalendar()).season_for(day_of_year)


def year_season_columns(
    columns: Iterable[str],
    *,
    calendar: SeasonCalendar | None = None,
    pattern: re.Pattern[str] = _ACQ_PATTERN,
) -> dict[str, str]:
    """Map acquisition column names to ``"<year>_<season>"`` labels.

    Names that do not match ``pattern`` are skipped.
    """
    cal = calendar or SeasonCalendar()
    mapping: dict[str, str] = {}
    for col in columns:
        m = pattern.search(col)
        if not m:
            continue
        mapping[col] = f"{m.group('year')}_{cal.season_for(int(m.group('doy')))}"
    return mapping


def _split_label(label: str) -> tuple[int, str]:
    year, season = label.split("_", 1)
    return int(year), season


def year_season_means(
    df: pd.DataFrame,
    *,
    id_columns: Sequence[str],
    calendar: SeasonCalendar | None = None,
    pattern: re.Pattern[str] = _ACQ_PATTERN,
) -> pd.DataFrame:
    """Collapse acquisition columns into one column per ``year_season`` (mean of members).

    Returns ``id_columns`` + geometry (if present) + sorted ``YYYY_season`` columns.
    """
    mapping = year_season_columns(df.columns, calendar=calendar, pattern=pattern)
    if not mapping:
        msg = "No acquisition columns matching YYYYDDD found"
        raise ValueError(msg)
    keep = [c for c in id_columns if c in df.columns]
    if hasattr(df, "geometry") and df.geometry.name not in keep:
        keep.append(df.geometry.name)
    out = df[keep].copy()
    groups: dict[str, list[str]] = {}
    for col, label in mapping.items():
        groups.setdefault(label, []).append(col)
    for label in sorted(groups, key=_split_label):
        out[label] = df[groups[label]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out


def _value_columns(df: pd.DataFrame, id_columns: Sequence[str]) -> list[str]:
    geom = df.geometry.name if hasattr(df, "geometry") else None
    return [
        c
        for c in df.columns
        if c not in id_columns and c != geom and re.fullmatch(r"\d{4}_[a-z]+", c)
    ]


def year_over_year_change(
    df: pd.DataFrame, *, id_columns: Sequence[str], relative: bool = True
) -> pd.DataFrame:
    """``<year>_<season>_yoy`` = change from the same season in the previous year.

    ``relative=True`` gives the fractional change ``(cur - prev) / prev`` (legacy behaviour);
    ``False`` gives the raw difference, which is safer when values can be ~0.
    """
    cols = _value_columns(df, id_columns)
    keep = [c for c in id_columns if c in df.columns]
    if hasattr(df, "geometry"):
        keep.append(df.geometry.name)
    out = df[keep].copy()
    labels = {c: _split_label(c) for c in cols}
    for col, (year, season) in sorted(labels.items(), key=lambda kv: kv[1]):
        prev = f"{year - 1}_{season}"
        if prev not in labels:
            continue
        cur_v = pd.to_numeric(df[col], errors="coerce")
        prev_v = pd.to_numeric(df[prev], errors="coerce")
        diff = cur_v - prev_v
        out[f"{col}_yoy"] = (diff / prev_v.replace(0, np.nan)) if relative else diff
    return out


def seasonal_means(df: pd.DataFrame, *, id_columns: Sequence[str]) -> pd.DataFrame:
    """``<season>_mean`` across years plus ``total_mean`` across all acquisitions."""
    cols = _value_columns(df, id_columns)
    keep = [c for c in id_columns if c in df.columns]
    if hasattr(df, "geometry"):
        keep.append(df.geometry.name)
    out = df[keep].copy()
    by_season: dict[str, list[str]] = {}
    for c in cols:
        by_season.setdefault(_split_label(c)[1], []).append(c)
    for season, members in sorted(by_season.items()):
        out[f"{season}_mean"] = df[members].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out["total_mean"] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return out


def seasonal_variance(yoy: pd.DataFrame, *, id_columns: Sequence[str]) -> pd.DataFrame:
    """``<season>_var``: variance across years of the year-over-year change for each season."""
    keep = [c for c in id_columns if c in yoy.columns]
    if hasattr(yoy, "geometry"):
        keep.append(yoy.geometry.name)
    out = yoy[keep].copy()
    by_season: dict[str, list[str]] = {}
    for c in yoy.columns:
        m = re.fullmatch(r"(\d{4})_([a-z]+)_yoy", str(c))
        if m:
            by_season.setdefault(m.group(2), []).append(c)
    for season, members in sorted(by_season.items()):
        out[f"{season}_var"] = yoy[members].var(axis=1)
    return out


def interannual_variance(yoy: pd.DataFrame, *, id_columns: Sequence[str]) -> pd.DataFrame:
    """``<year>_var``: variance across seasons of the year-over-year change within each year."""
    keep = [c for c in id_columns if c in yoy.columns]
    if hasattr(yoy, "geometry"):
        keep.append(yoy.geometry.name)
    out = yoy[keep].copy()
    by_year: dict[str, list[str]] = {}
    for c in yoy.columns:
        m = re.fullmatch(r"(\d{4})_([a-z]+)_yoy", str(c))
        if m:
            by_year.setdefault(m.group(1), []).append(c)
    for year, members in sorted(by_year.items()):
        out[f"{year}_var"] = yoy[members].var(axis=1)
    return out


def missingness_report(df: pd.DataFrame) -> pd.Series:
    """Share of missing values per column, descending - the legacy ``analyze_bias``."""
    return df.isna().mean().sort_values(ascending=False)
