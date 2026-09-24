"""Sample rasters along vector features.

This replaces the ``qgis:extractnodes``, ``grass7:v.sample``, ``qgis:intersection`` and pandas
groupby chain that attached mean DEM elevation to flowlines.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from fortress_gis.io.raster import RasterInfo, sample_raster_at_points


def line_vertices(
    lines: gpd.GeoDataFrame,
    *,
    id_column: str,
    densify_every: float | None = None,
) -> gpd.GeoDataFrame:
    """Explode (Multi)LineStrings into their vertices as points, keeping ``id_column``.

    ``densify_every`` (CRS units) adds interpolated points along each line. Without it a long
    straight segment contributes two samples, which is how the vertex-only original under-sampled
    valley floors.
    """
    geoms = lines.geometry.explode(index_parts=False)
    ids = lines.loc[geoms.index, id_column].to_numpy()
    if densify_every is not None and densify_every > 0:
        geoms = gpd.GeoSeries(
            shapely.segmentize(geoms.to_numpy(), densify_every), crs=lines.crs, index=geoms.index
        )
    coords, part_index = shapely.get_coordinates(geoms.to_numpy(), return_index=True)
    pts = gpd.GeoDataFrame(
        {id_column: ids[part_index], "vertex_index": _running_index(part_index)},
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
        crs=lines.crs,
    )
    return pts


def _running_index(groups: np.ndarray) -> np.ndarray:
    """0,1,2,... restarting whenever ``groups`` changes value (groups must be sorted)."""
    if groups.size == 0:
        return groups
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    lengths = np.diff(np.r_[starts, groups.size])
    return np.arange(groups.size) - np.repeat(starts, lengths)


def sample_lines_from_raster(
    lines: gpd.GeoDataFrame,
    raster: RasterInfo,
    *,
    id_column: str,
    stats: Sequence[str] = ("mean", "min", "max"),
    value_name: str = "elevation",
    densify_every: float | None = None,
) -> gpd.GeoDataFrame:
    """Attach per-line raster statistics (``<value>_<stat>``) sampled at the line vertices."""
    pts = line_vertices(lines, id_column=id_column, densify_every=densify_every)
    sampled = sample_raster_at_points(raster, pts, column=value_name)
    agg = sampled.groupby(id_column)[value_name].agg(list(stats))
    agg.columns = [f"{value_name}_{s}" for s in agg.columns]
    return lines.merge(agg, left_on=id_column, right_index=True, how="left")


def line_elevation_profile(
    line: shapely.LineString,
    raster: RasterInfo,
    *,
    crs: str | int,
    step: float,
) -> pd.DataFrame:
    """Distance-along-line vs raster value at regular ``step`` intervals (CRS units)."""
    length = line.length
    distances = np.arange(0.0, length + step, step)
    distances = distances[distances <= length]
    pts = gpd.GeoDataFrame(
        {"distance": distances}, geometry=[line.interpolate(d) for d in distances], crs=crs
    )
    sampled = sample_raster_at_points(raster, pts, column="value")
    return pd.DataFrame(
        {"distance": sampled["distance"].to_numpy(), "value": sampled["value"].to_numpy()}
    )
