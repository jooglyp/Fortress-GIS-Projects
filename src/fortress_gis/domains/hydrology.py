"""Hydrology workflow: DEM tiles to a pour-point catchment and flowline elevations.

The steps are mosaic, clip, fill depressions, D8 flow direction, flow accumulation, snap the
pour point, flood the catchment, then sample the DEM along flowlines. This replaces the QGIS 2
and TauDEM chain (``mergeElevData``, ``clipElevData``, ``calcUpstream``, ``getFlowlineElev``)
and adds QGIS and Kepler.gl exports.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from fortress_gis.crs import WGS84
from fortress_gis.io.raster import RasterInfo, clip_raster_to_geometry, merge_rasters, raster_info
from fortress_gis.log import get_logger
from fortress_gis.raster.sampling import sample_lines_from_raster
from fortress_gis.raster.terrain import (
    catchment,
    d8_flow_direction,
    downstream_path,
    fill_depressions,
    flow_accumulation,
    snap_pour_point,
    stream_network,
)
from fortress_gis.raster.zonal import mask_to_polygons, zonal_statistics
from fortress_gis.viz.kepler import KeplerMapBuilder
from fortress_gis.viz.qgis import QgisBundle

LOGGER = get_logger(__name__)


@dataclass
class WatershedResult:
    """Rasters, vectors and summary numbers from one watershed run."""

    dem: RasterInfo
    filled: RasterInfo
    flow_direction: RasterInfo
    accumulation: RasterInfo
    streams: RasterInfo
    catchment_mask: RasterInfo
    catchment_polygon: gpd.GeoDataFrame
    outlet: tuple[int, int]
    outlet_point: gpd.GeoDataFrame
    flowlines: gpd.GeoDataFrame | None = None
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.stats)


def load_dem(
    sources: Iterable[str | Path] | RasterInfo,
    *,
    clip_to: gpd.GeoDataFrame | None = None,
) -> RasterInfo:
    """Mosaic one or more DEM tiles and optionally clip to a boundary polygon."""
    if isinstance(sources, RasterInfo):
        dem = sources
    else:
        paths = [Path(p) for p in sources]
        dem = merge_rasters(paths) if len(paths) > 1 else raster_info(paths[0])
        LOGGER.info("Loaded DEM from %d tile(s): %s cells", len(paths), dem.shape)
    if clip_to is not None:
        dem = clip_raster_to_geometry(dem, clip_to)
        LOGGER.info("Clipped DEM to boundary: %s cells", dem.shape)
    return dem


def delineate_watershed(
    dem: RasterInfo,
    pour_point: tuple[float, float] | gpd.GeoDataFrame,
    *,
    stream_threshold_cells: int | None = None,
    snap_cells: int = 5,
    epsilon: float = 1e-4,
) -> WatershedResult:
    """Delineate the catchment above a pour point.

    ``pour_point`` is an ``(x, y)`` tuple in the DEM CRS or a single-point GeoDataFrame in any
    CRS. ``stream_threshold_cells`` defaults to 1% of the grid so stream masks scale with the DEM.
    """
    if isinstance(pour_point, gpd.GeoDataFrame):
        pp = pour_point.to_crs(dem.crs) if pour_point.crs is not None else pour_point
        geom = pp.geometry.iloc[0]
        x, y = float(geom.x), float(geom.y)
    else:
        x, y = float(pour_point[0]), float(pour_point[1])
    filled = fill_depressions(dem, epsilon=epsilon)
    fdir = d8_flow_direction(filled)
    acc = flow_accumulation(fdir)
    outlet = snap_pour_point(acc, x, y, search_cells=snap_cells)
    mask = catchment(fdir, outlet)
    poly = mask_to_polygons(mask)
    threshold = stream_threshold_cells or max(50, int(0.01 * dem.data.size))
    streams = stream_network(acc, threshold=float(threshold))
    xs, ys = acc.cell_centers()
    outlet_pt = gpd.GeoDataFrame(
        {
            "name": ["outlet"],
            "row": [outlet[0]],
            "col": [outlet[1]],
            "accumulation_cells": [float(acc.data[outlet])],
        },
        geometry=[Point(xs[outlet], ys[outlet])],
        crs=dem.crs,
    )
    cell_area = float(np.prod(dem.resolution))
    stats = {
        "dem_cells": float(dem.data.size),
        "catchment_cells": float(mask.data.sum()),
        "catchment_area_km2": float(mask.data.sum() * cell_area / 1e6),
        "outlet_accumulation_cells": float(acc.data[outlet]),
        "stream_threshold_cells": float(threshold),
        "stream_cells": float(streams.data.sum()),
        "min_elevation": float(np.nanmin(dem.masked())),
        "max_elevation": float(np.nanmax(dem.masked())),
    }
    return WatershedResult(
        dem, filled, fdir, acc, streams, mask, poly, outlet, outlet_pt, stats=stats
    )


def flowlines_from_terrain(
    result: WatershedResult,
    *,
    n_lines: int = 12,
    min_length_cells: int = 15,
    seed: int = 0,
    id_column: str = "line_id",
) -> gpd.GeoDataFrame:
    """Derive flowlines by tracing D8 paths downstream from random stream-head cells.

    A real project would load a hydrography product such as NHD flowlines. This exists so the
    demo has lines to sample without one. Lines are clipped to the catchment.
    """
    rng = np.random.default_rng(seed)
    acc = result.accumulation.masked()
    streams = result.streams.data.astype(bool) & result.catchment_mask.data.astype(bool)
    heads = np.argwhere(
        streams & (acc <= np.nanpercentile(acc[streams], 40) if streams.any() else streams)
    )
    if len(heads) == 0:
        return gpd.GeoDataFrame({id_column: []}, geometry=[], crs=result.dem.crs)
    picks = heads[rng.choice(len(heads), size=min(n_lines, len(heads)), replace=False)]
    xs, ys = result.dem.cell_centers()
    lines, ids = [], []
    for k, (r, c) in enumerate(picks):
        path = downstream_path(result.flow_direction, (int(r), int(c)))
        if len(path) < min_length_cells:
            continue
        coords = [(float(xs[i, j]), float(ys[i, j])) for i, j in path]
        lines.append(LineString(coords))
        ids.append(k + 1)
    return gpd.GeoDataFrame({id_column: ids}, geometry=lines, crs=result.dem.crs)


def attach_flowline_elevations(
    flowlines: gpd.GeoDataFrame,
    dem: RasterInfo,
    *,
    id_column: str,
    densify_every: float | None = None,
) -> gpd.GeoDataFrame:
    """Mean/min/max DEM elevation per flowline + slope (m per km) from the endpoints."""
    lines = flowlines.to_crs(dem.crs) if flowlines.crs is not None else flowlines
    sampled = sample_lines_from_raster(
        lines, dem, id_column=id_column, densify_every=densify_every or min(dem.resolution)
    )
    length_km = sampled.geometry.length / 1000.0
    drop = sampled["elevation_max"] - sampled["elevation_min"]
    sampled["length_km"] = length_km
    sampled["slope_m_per_km"] = np.where(length_km > 0, drop / length_km, np.nan)
    return sampled


def run_watershed_pipeline(
    dem_sources: Iterable[str | Path] | RasterInfo,
    pour_point: tuple[float, float] | gpd.GeoDataFrame,
    *,
    boundary: gpd.GeoDataFrame | None = None,
    flowlines: gpd.GeoDataFrame | None = None,
    flowline_id: str = "line_id",
    n_synthetic_flowlines: int = 12,
) -> WatershedResult:
    """Load and clip the DEM, delineate the catchment, attach flowline elevations."""
    dem = load_dem(dem_sources, clip_to=boundary)
    result = delineate_watershed(dem, pour_point)
    lines = (
        flowlines
        if flowlines is not None
        else flowlines_from_terrain(result, n_lines=n_synthetic_flowlines)
    )
    if len(lines):
        result.flowlines = attach_flowline_elevations(lines, result.filled, id_column=flowline_id)
        inside = zonal_statistics(
            result.catchment_polygon, result.filled, stats=("mean", "min", "max"), prefix="elev"
        )
        result.stats.update(
            {
                f"catchment_{c}": float(inside[c].iloc[0])
                for c in ("elev_mean", "elev_min", "elev_max")
            }
        )
    return result


def export_artifacts(
    result: WatershedResult,
    out_dir: str | Path,
    *,
    name: str = "watershed",
    kepler_html: bool = True,
) -> dict[str, Path]:
    """Write the QGIS bundle (DEM, accumulation, streams, catchment, flowlines) and a Kepler map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = QgisBundle(out / "qgis", name=name)
    bundle.add_raster(result.dem, "dem", colors=("#1a9850", "#fee08b", "#d73027", "#ffffff"))
    bundle.add_raster(
        result.accumulation.with_data(np.log1p(result.accumulation.masked())),
        "flow_accumulation_log",
    )
    bundle.add_raster(result.catchment_mask, "catchment_mask", discrete=True)
    bundle.add_vector(result.catchment_polygon, "catchment")
    bundle.add_vector(result.outlet_point, "outlet")
    if result.flowlines is not None and len(result.flowlines):
        bundle.add_vector(result.flowlines, "flowlines", graduated="elevation_mean", k=5)
    bundle.write()
    paths = {"qgis": out / "qgis"}
    if kepler_html:
        builder = KeplerMapBuilder(title=f"{name} - watershed", map_style="dark")
        builder.add_layer(result.catchment_polygon, "catchment", opacity=0.35, colors=["#2b8cbe"])
        if result.flowlines is not None and len(result.flowlines):
            builder.add_layer(
                result.flowlines, "flowlines", color_field="elevation_mean", color_scale="quantile"
            )
        builder.add_layer(result.outlet_point, "outlet", radius=12, colors=["#ff7f00"])
        paths["kepler_html"] = builder.save_html(out / f"{name}_kepler.html")
        paths["kepler_config"] = builder.save_config(out / f"{name}_kepler_config.json")
    return paths


def wgs84_bounds(result: WatershedResult) -> Sequence[float]:
    """Catchment bounds in lon/lat, for framing maps."""
    return result.catchment_polygon.to_crs(WGS84).total_bounds.tolist()
