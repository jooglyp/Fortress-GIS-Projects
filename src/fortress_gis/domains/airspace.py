"""Airspace workflow: aircraft trajectories to proximity events, hex risk classes and tests.

Positions are read (with dask for the full parquet), filtered to a time-of-day band, an
altitude band and a bounding box, then scanned for pairs of different tracks that were within a
horizontal distance, a vertical separation and a few seconds of each other. Events are summed
into hexagons sized by flight time, classed into risk levels, tested for excess against a
Poisson baseline, and cross-tabulated against hour of day with a chi-square test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from fortress_gis.compute.backend import is_dask, npartitions_for, to_pandas
from fortress_gis.features.proximity import (
    ProximityConfig,
    count_proximity_events,
    count_proximity_events_dask,
    exposure_hours,
)
from fortress_gis.features.temporal import add_hour_and_day_night
from fortress_gis.grids.aggregate import points_to_cells, points_to_cells_dask
from fortress_gis.grids.hexgrid import hex_grid_for_layer, hex_radius_from_travel_time
from fortress_gis.io.vector import read_points
from fortress_gis.log import get_logger
from fortress_gis.stats.classify import risk_levels
from fortress_gis.stats.hypothesis import (
    ChiSquareResult,
    chi_square_independence,
    poisson_excess_test,
    poisson_rate_test,
)
from fortress_gis.viz.kepler import KeplerMapBuilder
from fortress_gis.viz.qgis import QgisBundle

LOGGER = get_logger(__name__)

# Column names of the position product. Override via PositionSchema if a feed differs.
TIME = "DATETIME_UTC"
LAT = "LATITUDE"
LON = "LONGITUDE"
ALT = "ALTITUDE_AGL_FT"
SPEED = "GROUND_SPEED_KNTS"
TRACK = "FLIGHT_UID"


@dataclass(frozen=True)
class PositionSchema:
    """Column names of a position feed."""

    time: str = TIME
    lat: str = LAT
    lon: str = LON
    altitude: str = ALT
    speed: str = SPEED
    track: str = TRACK


DEFAULT_SCHEMA = PositionSchema()


@dataclass
class AirspaceFilter:
    """Row filters applied before event detection.

    ``hours`` is a half-open ``[start, end)`` band in UTC. ``altitude_ft`` is inclusive.
    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)``.
    """

    hours: tuple[int, int] | None = (7, 19)
    altitude_ft: tuple[float, float] | None = (0.0, 500.0)
    bbox: tuple[float, float, float, float] | None = None
    min_speed_kn: float | None = None


@dataclass
class AirspaceResult:
    """Positions with event counts, hex cells with risk classes, and the test tables."""

    positions: gpd.GeoDataFrame
    cells: gpd.GeoDataFrame
    exposure_hours: float
    poisson: pd.DataFrame
    rate_test: pd.DataFrame
    hourly: pd.DataFrame
    chi_square: ChiSquareResult | None = None
    hex_radius_m: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.stats)


def load_positions(
    path: str | Path,
    *,
    schema: PositionSchema = DEFAULT_SCHEMA,
    as_dask: bool = False,
    npartitions: int | None = None,
    columns: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read a position table (parquet or CSV) as WGS84 points.

    ``as_dask=True`` returns a dask-geopandas frame; nothing is loaded until a filter or
    ``compute`` forces it. A single-row-group parquet arrives as one partition, so the frame is
    repartitioned to ``npartitions`` (default: about 250k rows each, at least 4) so the filter
    and geometry construction run in parallel.
    """
    cols = (
        list(columns)
        if columns
        else [schema.time, schema.lat, schema.lon, schema.altitude, schema.speed, schema.track]
    )
    if as_dask and npartitions is None:
        import pyarrow.parquet as pq

        src = Path(path)
        n_rows = (
            pq.ParquetFile(src).metadata.num_rows
            if src.suffix.lower() in {".parquet", ".pq"}
            else None
        )
        npartitions = max(4, npartitions_for(n_rows)) if n_rows else 4
    return read_points(
        path, lon=schema.lon, lat=schema.lat, columns=cols, as_dask=as_dask, npartitions=npartitions
    )


def filter_positions(
    gdf: gpd.GeoDataFrame,
    flt: AirspaceFilter,
    *,
    schema: PositionSchema = DEFAULT_SCHEMA,
) -> gpd.GeoDataFrame:
    """Apply :class:`AirspaceFilter` to eager or dask frames without computing them."""
    out = gdf
    if flt.hours is not None:
        ts = out[schema.time]
        hour = ts.dt.hour if is_dask(out) else pd.to_datetime(ts, utc=True).dt.hour
        out = out[(hour >= flt.hours[0]) & (hour < flt.hours[1])]
    if flt.altitude_ft is not None:
        lo, hi = flt.altitude_ft
        out = out[(out[schema.altitude] >= lo) & (out[schema.altitude] <= hi)]
    if flt.min_speed_kn is not None:
        out = out[out[schema.speed] >= flt.min_speed_kn]
    if flt.bbox is not None:
        min_lon, min_lat, max_lon, max_lat = flt.bbox
        out = out[
            (out[schema.lon] >= min_lon)
            & (out[schema.lon] <= max_lon)
            & (out[schema.lat] >= min_lat)
            & (out[schema.lat] <= max_lat)
        ]
    return out


def flight_time_hex_grid(
    positions: gpd.GeoDataFrame,
    *,
    minutes: float = 5.0,
    speed_kn: float | None = None,
    schema: PositionSchema = DEFAULT_SCHEMA,
    divisor: float = 2.0,
) -> gpd.GeoDataFrame:
    """Hexagons sized so a typical aircraft crosses one in ``minutes``.

    ``speed_kn`` defaults to the median ground speed of the positions. With ``divisor=2`` the
    hex diameter equals the distance flown in ``minutes``.
    """
    if speed_kn is None:
        speed_kn = float(to_pandas(positions[schema.speed]).median())
    radius = hex_radius_from_travel_time(speed_kn, minutes, speed_units="knots", divisor=divisor)
    LOGGER.info("Hex radius %.0f m from %.0f kn over %.1f min", radius, speed_kn, minutes)
    cells = hex_grid_for_layer(
        positions if not is_dask(positions) else positions.compute(), radius=radius
    )
    cells.attrs["radius_m"] = float(radius)
    return cells


def detect_events(
    positions: gpd.GeoDataFrame,
    *,
    config: ProximityConfig | None = None,
    schema: PositionSchema = DEFAULT_SCHEMA,
    use_dask: bool | None = None,
    window: str = "1h",
) -> gpd.GeoDataFrame:
    """Append ``event_count`` to each position.

    ``use_dask`` defaults to True for dask inputs and False otherwise; the dask path splits the
    timeline into ``window`` chunks and runs them as ``delayed`` tasks.
    """
    cfg = config or ProximityConfig()
    kwargs = dict(
        time_column=schema.time,
        altitude_column=schema.altitude,
        track_column=schema.track,
        config=cfg,
    )
    if use_dask is None:
        use_dask = is_dask(positions)
    eager = to_pandas(positions) if is_dask(positions) else positions
    if not isinstance(eager, gpd.GeoDataFrame):
        eager = gpd.GeoDataFrame(
            eager,
            geometry=gpd.points_from_xy(eager[schema.lon], eager[schema.lat]),
            crs="EPSG:4326",
        )
    if use_dask:
        return count_proximity_events_dask(eager, window=window, **kwargs)
    return count_proximity_events(eager, **kwargs)


def summarise_cells(
    positions: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    count_column: str = "event_count",
    k: int = 3,
    use_dask: bool = False,
    schema: PositionSchema = DEFAULT_SCHEMA,
) -> gpd.GeoDataFrame:
    """Sum events and count positions per hex, then add risk classes and exposure.

    Exposure per hex is the number of position reports in it (each report is one observation
    interval), so ``events_per_1k_reports`` is comparable across cells.
    """
    agg = points_to_cells_dask if use_dask else points_to_cells
    out = agg(positions, cells, value_columns=[count_column], agg="sum")
    out = (
        out.rename(columns={f"{count_column}_sum": count_column})
        if f"{count_column}_sum" in out.columns
        else out
    )
    out[count_column] = out[count_column].fillna(0).astype("int64")
    out["events_per_1k_reports"] = np.where(
        out["n_points"] > 0, 1000.0 * out[count_column] / out["n_points"].clip(lower=1), np.nan
    )
    out = risk_levels(out, count_column, k=k)
    return out


def hourly_profile(
    positions: gpd.GeoDataFrame,
    *,
    count_column: str = "event_count",
    schema: PositionSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    """Events, reports and event rate per UTC hour."""
    df = add_hour_and_day_night(to_pandas(positions), time_column=schema.time)
    grp = df.groupby("hour", observed=True)
    out = pd.DataFrame(
        {
            "reports": grp.size(),
            "events": grp[count_column].sum(),
            "tracks": grp[schema.track].nunique(),
        }
    )
    out["events_per_1k_reports"] = 1000.0 * out["events"] / out["reports"]
    return out.reset_index()


def test_cells(
    cells: gpd.GeoDataFrame,
    positions: gpd.GeoDataFrame,
    *,
    count_column: str = "event_count",
    max_rate_per_hour: float = 1.0 / 10_000.0,
    report_interval_s: float | None = None,
    schema: PositionSchema = DEFAULT_SCHEMA,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Poisson excess and rate tests per hex. Returns ``(excess, rate, exposure_hours)``.

    The excess test uses the global mean count as the baseline. The rate test needs hours of
    exposure per hex: the report count times :func:`median_report_interval`, unless
    ``report_interval_s`` is given.
    """
    pos = to_pandas(positions)
    total_hours = exposure_hours(pos, time_column=schema.time, track_column=schema.track)
    if report_interval_s is None:
        report_interval_s = median_report_interval(pos, schema=schema)
    populated = cells["n_points"] > 0
    excess = poisson_excess_test(cells.loc[populated, count_column], alpha=alpha)
    exposure = cells.loc[populated, "n_points"].to_numpy() * report_interval_s / 3600.0
    rate = poisson_rate_test(
        cells.loc[populated, count_column], exposure, max_rate=max_rate_per_hour, alpha=alpha
    )
    excess.index = cells.index[populated]
    rate.index = cells.index[populated]
    return excess, rate, total_hours


def median_report_interval(
    positions: pd.DataFrame, *, schema: PositionSchema = DEFAULT_SCHEMA
) -> float:
    """Median seconds between consecutive reports of the same track (at least 1 s).

    Measured within tracks: the gap between all reports sorted together is near zero whenever
    many aircraft are airborne at once and would make every exposure look tiny.
    """
    ts = pd.to_datetime(positions[schema.time], utc=True)
    frame = pd.DataFrame({"t": ts, "k": positions[schema.track].to_numpy()}).sort_values(["k", "t"])
    gaps = frame.groupby("k", sort=False)["t"].diff().dt.total_seconds().dropna()
    return float(max(gaps.median(), 1.0)) if len(gaps) else 1.0


def hour_risk_dependence(
    positions: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    count_column: str = "event_count",
    schema: PositionSchema = DEFAULT_SCHEMA,
) -> ChiSquareResult | None:
    """Chi-square test of hex risk class against hour of day, on positions with events.

    Returns ``None`` when there are fewer than two hours or two classes with events.
    """
    from fortress_gis.grids.aggregate import assign_cells

    pos = add_hour_and_day_night(to_pandas(positions), time_column=schema.time)
    pos = pos[pos[count_column] > 0]
    if pos.empty:
        return None
    joined = assign_cells(gpd.GeoDataFrame(pos, geometry="geometry", crs=cells.crs), cells)
    joined = pd.DataFrame(joined.drop(columns="geometry")).merge(
        cells[["hex_id", "risk_label"]], on="hex_id", how="inner"
    )
    if joined["hour"].nunique() < 2 or joined["risk_label"].nunique() < 2:
        return None
    return chi_square_independence(joined, row="risk_label", col="hour", values=count_column)


def run_airspace_pipeline(
    source: str | Path | gpd.GeoDataFrame,
    *,
    flt: AirspaceFilter | None = None,
    config: ProximityConfig | None = None,
    minutes_per_hex: float = 5.0,
    schema: PositionSchema = DEFAULT_SCHEMA,
    use_dask: bool = False,
    window: str = "1h",
    k: int = 3,
) -> AirspaceResult:
    """Load, filter, detect events, aggregate to hexes and run the tests."""
    flt = flt or AirspaceFilter()
    positions = (
        load_positions(source, schema=schema, as_dask=use_dask)
        if isinstance(source, str | Path)
        else source
    )
    positions = filter_positions(positions, flt, schema=schema)
    positions = to_pandas(positions)
    LOGGER.info(
        "%d positions from %d tracks after filtering",
        len(positions),
        positions[schema.track].nunique(),
    )
    positions = detect_events(
        positions, config=config, schema=schema, use_dask=use_dask, window=window
    )
    cells = flight_time_hex_grid(positions, minutes=minutes_per_hex, schema=schema)
    radius_m = float(cells.attrs.get("radius_m", 0.0))
    cells = summarise_cells(positions, cells, k=k, use_dask=use_dask, schema=schema)
    excess, rate, hours = test_cells(cells, positions, schema=schema)
    cells["excess_p"] = excess["p_adjusted"]
    cells["excess_significant"] = excess["significant"].astype("boolean")
    cells["rate_p"] = rate["p_adjusted"]
    cells["rate_exceeds_tolerance"] = rate["significant"].astype("boolean")
    hourly = hourly_profile(positions, schema=schema)
    chi = hour_risk_dependence(positions, cells, schema=schema)
    stats = {
        "positions": float(len(positions)),
        "tracks": float(positions[schema.track].nunique()),
        "positions_with_events": float((positions["event_count"] > 0).sum()),
        "events_total": float(positions["event_count"].sum()),
        "exposure_hours": float(hours),
        "events_per_hour": float(positions["event_count"].sum() / hours) if hours else float("nan"),
        "hex_cells_populated": float((cells["n_points"] > 0).sum()),
        "hex_cells_excess": float(cells["excess_significant"].fillna(False).sum()),
        "hex_cells_over_tolerance": float(cells["rate_exceeds_tolerance"].fillna(False).sum()),
        "hex_radius_m": radius_m,
        "chi_square_p": float(chi.p_value) if chi is not None else float("nan"),
    }
    return AirspaceResult(
        positions,
        cells,
        hours,
        excess,
        rate,
        hourly,
        chi,
        radius_m,
        stats,
    )


def export_artifacts(
    result: AirspaceResult,
    out_dir: str | Path,
    *,
    name: str = "airspace",
    kepler_html: bool = True,
    max_points: int = 150_000,
    schema: PositionSchema = DEFAULT_SCHEMA,
) -> dict[str, Path]:
    """QGIS bundle (hex risk layer plus event positions) and a Kepler map with a time filter.

    Positions are thinned to ``max_points`` for the Kepler HTML (events first, then a random
    sample) so the file stays loadable in a browser.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = result.positions[result.positions["event_count"] > 0]
    bundle = QgisBundle(out / "qgis", name=name)
    bundle.add_vector(result.cells, "hex_risk", graduated="event_count", k=5)
    bundle.add_vector(result.cells, "hex_risk_class", categorized="risk_label")
    if len(events):
        bundle.add_vector(events, "event_positions", graduated="event_count", k=4)
    bundle.write()
    result.hourly.to_csv(out / f"{name}_hourly.csv", index=False)
    paths = {"qgis": out / "qgis", "hourly_csv": out / f"{name}_hourly.csv"}
    if kepler_html:
        builder = KeplerMapBuilder(title=f"{name} - proximity risk", map_style="dark")
        builder.add_layer(
            result.cells[result.cells["n_points"] > 0],
            "hex risk",
            color_field="event_count",
            color_scale="quantile",
            opacity=0.6,
        )
        builder.add_layer(
            result.cells[result.cells["n_points"] > 0],
            "risk class",
            color_field="risk_label",
            categorical_colors={"low": "#2b83ba", "medium": "#fdae61", "high": "#d7191c"},
            visible=False,
        )
        sample = events
        if len(sample) < max_points:
            rest = result.positions[result.positions["event_count"] == 0]
            take = min(len(rest), max_points - len(sample))
            if take > 0:
                sample = pd.concat([sample, rest.sample(take, random_state=0)])
        else:
            sample = sample.sample(max_points, random_state=0)
        builder.add_layer(
            gpd.GeoDataFrame(sample, geometry="geometry", crs=result.positions.crs),
            "positions",
            color_field="event_count",
            radius=3,
            time_field=schema.time,
            opacity=0.5,
        )
        paths["kepler_html"] = builder.save_html(out / f"{name}_kepler.html")
        paths["kepler_config"] = builder.save_config(out / f"{name}_kepler_config.json")
    return paths
