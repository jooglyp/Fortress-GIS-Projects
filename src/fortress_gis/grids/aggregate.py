"""Assign points to cells and aggregate them.

The airspace notebook counted events per hexagon with a Python loop over cells and a ``within``
test for each, O(cells x points). The oceanography package used ``sjoin`` and then merged one
column at a time. Both become one spatial join and one grouped aggregation, with a
dask-geopandas variant for point clouds that do not fit in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import dask
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd

from fortress_gis.compute.backend import to_dask_geo
from fortress_gis.crs import to_crs_obj
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

AggSpec = str | Sequence[str] | Mapping[str, str | Sequence[str]]


def assign_cells(
    points: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    cell_id: str = "hex_id",
    predicate: str = "within",
) -> gpd.GeoDataFrame:
    """Attach ``cell_id`` to each point (points outside every cell get NaN).

    Points are reprojected to the cell CRS. Points falling exactly on shared edges are
    assigned to the first matching cell so every point is counted once.
    """
    pts = (
        points.to_crs(cells.crs)
        if (points.crs is not None and cells.crs is not None and points.crs != cells.crs)
        else points
    )
    joined = gpd.sjoin(pts, cells[[cell_id, cells.geometry.name]], how="left", predicate=predicate)
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def _resolve_agg(value_columns: Sequence[str], agg: AggSpec) -> dict[str, list[str]]:
    """Normalise the ``agg`` argument to ``{column: [func, ...]}``."""
    if isinstance(agg, Mapping):
        return {k: ([v] if isinstance(v, str) else list(v)) for k, v in agg.items()}
    funcs = [agg] if isinstance(agg, str) else list(agg)
    return {c: list(funcs) for c in value_columns}


def points_to_cells(
    points: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    cell_id: str = "hex_id",
    value_columns: Sequence[str] | None = None,
    agg: AggSpec = "mean",
    count_column: str | None = "n_points",
    predicate: str = "within",
    keep_empty: bool = True,
) -> gpd.GeoDataFrame:
    """Aggregate point attributes into cells.

    Parameters
    ----------
    value_columns:
        Numeric point columns to aggregate; defaults to every numeric column that is not a
        coordinate or the cell id.
    agg:
        ``"mean"``, a list of functions, or ``{column: func_or_list}``. Output columns are
        ``<column>_<func>`` unless a single function is used, in which case ``<column>``.
    count_column:
        Name of the per-cell point count column, or ``None`` to skip it.
    keep_empty:
        Keep cells with no points (values NaN, count 0) so the tessellation stays complete.
    """
    assigned = assign_cells(points, cells, cell_id=cell_id, predicate=predicate)
    inside = assigned.dropna(subset=[cell_id])
    if value_columns is None:
        skip = {
            cell_id,
            "x",
            "y",
            "longitude",
            "latitude",
            "row",
            "col",
            "centroid_x",
            "centroid_y",
        }
        value_columns = [
            c
            for c in inside.columns
            if c not in skip
            and c != inside.geometry.name
            and pd.api.types.is_numeric_dtype(inside[c])
            and not pd.api.types.is_bool_dtype(inside[c])
        ]
    spec = _resolve_agg(value_columns, agg)
    single = all(len(v) == 1 for v in spec.values()) and len({tuple(v) for v in spec.values()}) == 1
    grouped = inside.groupby(cell_id)
    stats = grouped.agg(spec) if spec else pd.DataFrame(index=grouped.size().index)
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = [c if single else f"{c}_{f}" for c, f in stats.columns]
    if count_column:
        stats[count_column] = grouped.size()
    stats.index = stats.index.astype(cells[cell_id].dtype)
    how = "left" if keep_empty else "inner"
    out = cells.merge(stats, left_on=cell_id, right_index=True, how=how)
    if count_column:
        out[count_column] = out[count_column].fillna(0).astype("int64")
    return gpd.GeoDataFrame(out, geometry=cells.geometry.name, crs=cells.crs)


def points_to_cells_dask(
    points: gpd.GeoDataFrame | dgpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    cell_id: str = "hex_id",
    value_columns: Sequence[str] | None = None,
    agg: AggSpec = "mean",
    count_column: str | None = "n_points",
    npartitions: int | None = None,
    spatial_shuffle: bool = False,
) -> gpd.GeoDataFrame:
    """:func:`points_to_cells` on partitioned frames with dask-geopandas.

    Each partition is spatially joined to the cell layer (small, so it is broadcast), then
    aggregated with a dask ``groupby``. Only the per-cell table is materialised. ``agg`` takes
    the same forms as in :func:`points_to_cells` and produces the same column names.
    """
    dpoints = to_dask_geo(points, npartitions, spatial_shuffle=spatial_shuffle)
    if dpoints.crs is not None and cells.crs is not None and dpoints.crs != cells.crs:
        dpoints = dpoints.to_crs(to_crs_obj(cells.crs))
    cells_small = cells[[cell_id, cells.geometry.name]]
    joined = dgpd.sjoin(dpoints, cells_small, how="inner", predicate="within")
    if value_columns is None:
        meta = joined._meta
        skip = {cell_id, "x", "y", "longitude", "latitude", "index_right"}
        value_columns = [
            c
            for c in meta.columns
            if c not in skip
            and c != meta.geometry.name
            and pd.api.types.is_numeric_dtype(meta[c])
            and not pd.api.types.is_bool_dtype(meta[c])
        ]
    cols = [cell_id, *value_columns]
    table = joined[cols]
    grouped = table.groupby(cell_id)
    spec = _resolve_agg(value_columns, agg)
    single = all(len(v) == 1 for v in spec.values()) and len({tuple(v) for v in spec.values()}) == 1
    parts = [grouped.agg(spec)] if spec else []
    if count_column:
        parts.append(grouped.size().rename(count_column).to_frame())
    computed = dask.compute(*parts)
    frames = []
    for part in computed:
        frame = part.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [c if single else f"{c}_{f}" for c, f in frame.columns]
        frames.append(frame)
    result = pd.concat(frames, axis=1)
    result.index = result.index.astype(cells[cell_id].dtype)
    out = cells.merge(result, left_on=cell_id, right_index=True, how="left")
    if count_column:
        out[count_column] = out[count_column].fillna(0).astype("int64")
    return gpd.GeoDataFrame(out, geometry=cells.geometry.name, crs=cells.crs)


def cell_centroids(
    cells: gpd.GeoDataFrame, *, keep_columns: Sequence[str] | None = None
) -> gpd.GeoDataFrame:
    """Point layer of cell centroids carrying the requested attribute columns.

    Centroids are computed in a metric CRS when the layer is geographic to avoid pyproj warnings,
    then returned in the layer's original CRS.
    """
    src = cells
    if cells.crs is not None and not cells.crs.is_projected:
        from fortress_gis.crs import suggest_metric_crs

        src = cells.to_crs(suggest_metric_crs(cells))
    cents = src.geometry.centroid
    if src.crs != cells.crs:
        cents = cents.to_crs(cells.crs)
    cols = (
        list(keep_columns)
        if keep_columns is not None
        else [c for c in cells.columns if c != cells.geometry.name]
    )
    out = gpd.GeoDataFrame(cells[cols].copy(), geometry=cents.values, crs=cells.crs)
    out["lon"] = out.geometry.x if (cells.crs is None or not cells.crs.is_projected) else np.nan
    out["lat"] = out.geometry.y if (cells.crs is None or not cells.crs.is_projected) else np.nan
    return out
