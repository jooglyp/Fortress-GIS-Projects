"""Spatio-temporal proximity ("near event") detection between moving objects.

Given timestamped positions from many tracks (aircraft, vessels, vehicles), count for every
position how many positions from other tracks were within a horizontal distance, a vertical
separation and a time tolerance at once. The airspace workflow calls such a pair an event.

The original notebook ran a per-row ``apply`` with a pandas time filter and an R-tree query,
which is O(n * candidates) in Python. Here all pairs come from one ``cKDTree.query_pairs`` on the
3-D embedding ``(x, y, t * v)`` with ``v = horizontal / time_tolerance``, so the time tolerance
becomes a distance. A ball of radius ``horizontal * sqrt(2)`` contains the required cylinder;
pairs are then filtered exactly. The dask variant cuts the timeline into overlapping windows so
long histories run in parallel.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from fortress_gis.crs import suggest_metric_crs, to_crs_obj
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

FEET_TO_METERS = 0.3048


@dataclass(frozen=True)
class ProximityConfig:
    """Tolerances defining a proximity event.

    Attributes
    ----------
    horizontal_m:
        Max horizontal separation (metres). Default 1000 ft.
    vertical:
        Max vertical separation in the units of the altitude column (default 200 ft).
    time_tolerance_s:
        Positions are compared only if reported within this many seconds of each other.
    exclude_same_track:
        Ignore pairs from the same track id. Consecutive fixes of one aircraft are always near
        each other and would otherwise dominate the counts.
    """

    horizontal_m: float = 1000.0 * FEET_TO_METERS
    vertical: float = 200.0
    time_tolerance_s: float = 5.0
    exclude_same_track: bool = True


def _prepare(
    gdf: gpd.GeoDataFrame,
    *,
    time_column: str,
    metric_crs: str | int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    crs = metric_crs or suggest_metric_crs(gdf)
    proj = gdf.to_crs(to_crs_obj(crs))
    x = proj.geometry.x.to_numpy(dtype="float64")
    y = proj.geometry.y.to_numpy(dtype="float64")
    ts = pd.to_datetime(gdf[time_column], utc=True)
    t = ((ts - ts.min()).dt.total_seconds()).to_numpy(dtype="float64")
    return x, y, t, to_crs_obj(crs).to_string()


def _pairs_for_block(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alt: np.ndarray | None,
    track: np.ndarray | None,
    config: ProximityConfig,
) -> np.ndarray:
    """(k, 2) integer pairs (local indices) satisfying all tolerances."""
    if x.size < 2:
        return np.empty((0, 2), dtype="int64")
    scale = config.horizontal_m / max(config.time_tolerance_s, 1e-9)
    coords = np.column_stack([x, y, t * scale])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=config.horizontal_m * math.sqrt(2.0), output_type="ndarray")
    if pairs.size == 0:
        return pairs.reshape(0, 2)
    i, j = pairs[:, 0], pairs[:, 1]
    keep = np.hypot(x[i] - x[j], y[i] - y[j]) <= config.horizontal_m
    keep &= np.abs(t[i] - t[j]) <= config.time_tolerance_s
    if alt is not None:
        keep &= np.abs(alt[i] - alt[j]) <= config.vertical
    if track is not None and config.exclude_same_track:
        keep &= track[i] != track[j]
    return pairs[keep]


def count_proximity_events(
    gdf: gpd.GeoDataFrame,
    *,
    time_column: str,
    altitude_column: str | None = None,
    track_column: str | None = None,
    config: ProximityConfig | None = None,
    metric_crs: str | int | None = None,
    count_column: str = "event_count",
    return_pairs: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Count proximity events per position (in-memory variant).

    Returns the input with ``count_column`` appended; with ``return_pairs=True`` also returns a
    table of the matched pairs (``i``, ``j`` positional indices, ``distance_m``, ``dt_s``).
    """
    cfg = config or ProximityConfig()
    x, y, t, _ = _prepare(gdf, time_column=time_column, metric_crs=metric_crs)
    alt = gdf[altitude_column].to_numpy(dtype="float64") if altitude_column else None
    track = pd.factorize(gdf[track_column])[0] if track_column else None
    pairs = _pairs_for_block(x, y, t, alt, track, cfg)
    counts = (
        np.bincount(pairs.ravel(), minlength=len(gdf))
        if pairs.size
        else np.zeros(len(gdf), dtype="int64")
    )
    out = gdf.copy()
    out[count_column] = counts.astype("int64")
    LOGGER.info(
        "count_proximity_events: %d pairs, %d positions with >=1 event",
        len(pairs),
        int((counts > 0).sum()),
    )
    if return_pairs:
        table = pd.DataFrame(
            {
                "i": pairs[:, 0],
                "j": pairs[:, 1],
                "distance_m": np.hypot(
                    x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]]
                ),
                "dt_s": np.abs(t[pairs[:, 0]] - t[pairs[:, 1]]),
            }
        )
        return out, table
    return out


def _window_slices(
    t: np.ndarray, *, window_s: float, pad_s: float
) -> Iterator[tuple[np.ndarray, float, float]]:
    """Yield (indices, core_start, core_end) for overlapping time windows (t must be sorted)."""
    start, stop = float(t.min()), float(t.max())
    cursor = start
    while cursor <= stop:
        core_end = cursor + window_s
        lo = np.searchsorted(t, cursor - pad_s, side="left")
        hi = np.searchsorted(t, core_end + pad_s, side="right")
        yield np.arange(lo, hi), cursor, core_end
        cursor = core_end


def count_proximity_events_dask(
    gdf: gpd.GeoDataFrame,
    *,
    time_column: str,
    altitude_column: str | None = None,
    track_column: str | None = None,
    config: ProximityConfig | None = None,
    metric_crs: str | int | None = None,
    window: str | pd.Timedelta = "1h",
    count_column: str = "event_count",
) -> gpd.GeoDataFrame:
    """Parallel variant: split the timeline into ``window`` chunks padded by the time tolerance.

    Each chunk is processed independently with :func:`_pairs_for_block` via ``dask.delayed``.
    A pair belongs to the chunk whose unpadded interval contains the earlier of its two
    timestamps, so a pair spanning a boundary is counted once.
    """
    import dask

    cfg = config or ProximityConfig()
    order = np.argsort(pd.to_datetime(gdf[time_column], utc=True).to_numpy(), kind="stable")
    sorted_gdf = gdf.iloc[order]
    x, y, t, _ = _prepare(sorted_gdf, time_column=time_column, metric_crs=metric_crs)
    alt = sorted_gdf[altitude_column].to_numpy(dtype="float64") if altitude_column else None
    track = pd.factorize(sorted_gdf[track_column])[0] if track_column else None
    window_s = pd.Timedelta(window).total_seconds()

    @dask.delayed
    def _block(idx: np.ndarray, core_start: float, core_end: float) -> np.ndarray:
        pairs = _pairs_for_block(
            x[idx],
            y[idx],
            t[idx],
            alt[idx] if alt is not None else None,
            track[idx] if track is not None else None,
            cfg,
        )
        if pairs.size == 0:
            return np.zeros(len(t), dtype="int64")
        gi, gj = idx[pairs[:, 0]], idx[pairs[:, 1]]
        t_min = np.minimum(t[gi], t[gj])
        own = (t_min >= core_start) & (t_min < core_end)
        counts = np.bincount(np.concatenate([gi[own], gj[own]]), minlength=len(t))
        return counts.astype("int64")

    tasks = [
        _block(idx, s, e)
        for idx, s, e in _window_slices(t, window_s=window_s, pad_s=cfg.time_tolerance_s)
    ]
    LOGGER.info("count_proximity_events_dask: %d windows of %s", len(tasks), window)
    partials = dask.compute(*tasks)
    total = np.sum(partials, axis=0) if partials else np.zeros(len(t), dtype="int64")
    counts = np.empty(len(gdf), dtype="int64")
    counts[order] = total
    out = gdf.copy()
    out[count_column] = counts
    return out


def exposure_hours(
    df: pd.DataFrame,
    *,
    time_column: str,
    track_column: str | None = None,
    max_gap_s: float = 600.0,
) -> float:
    """Total observed hours across the table.

    Without ``track_column`` this is the global span, last minus first timestamp. With it, the
    gaps between consecutive reports of the same track are summed, and any gap longer than
    ``max_gap_s`` (default 10 minutes) is dropped. Track identifiers are often reused across
    days, so first-to-last spans would count time nobody was observed; in one month of real
    reports that overstated exposure by a factor of about 300.
    """
    ts = pd.to_datetime(df[time_column], utc=True)
    if track_column is None:
        return float((ts.max() - ts.min()).total_seconds() / 3600.0)
    frame = pd.DataFrame({"t": ts.to_numpy(), "k": df[track_column].to_numpy()})
    frame = frame.sort_values(["k", "t"])
    gaps = frame.groupby("k", sort=False)["t"].diff().dt.total_seconds().dropna()
    return float(gaps[gaps <= max_gap_s].sum() / 3600.0)


def flag_unacceptable(
    df: pd.DataFrame,
    *,
    count_column: str = "event_count",
    exposure_hours_total: float,
    max_rate_per_hour: float = 1.0 / 10_000.0,
    flag_column: str = "unacceptable",
) -> pd.DataFrame:
    """Flag positions whose event count exceeds the tolerated rate given total exposure.

    ``max_rate_per_hour`` defaults to 1 event per 10,000 hours. With ``count > 0`` the rate
    ``count / exposure_hours_total`` is compared to the threshold.
    """
    out = df.copy()
    rate = out[count_column] / max(exposure_hours_total, 1e-9)
    out[flag_column] = (out[count_column] > 0) & (rate > max_rate_per_hour)
    return out
