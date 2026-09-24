"""Vector I/O for OGR formats, GeoParquet and lon/lat tables.

Readers have an ``as_dask`` switch so point clouds of a few million positions can be handled
partition by partition.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from fortress_gis.compute.backend import npartitions_for, to_dask_geo
from fortress_gis.crs import WGS84, to_crs_obj

PARQUET_SUFFIXES = {".parquet", ".pq", ".geoparquet"}


def points_from_xy(
    df: pd.DataFrame,
    *,
    lon: str = "longitude",
    lat: str = "latitude",
    crs: str | int = WGS84,
    drop_xy: bool = False,
) -> gpd.GeoDataFrame:
    """Build a point GeoDataFrame from lon/lat columns.

    Rows with missing coordinates are dropped so downstream spatial indexes stay valid.
    """
    missing = [c for c in (lon, lat) if c not in df.columns]
    if missing:
        msg = f"Columns {missing} not found in table (have {list(df.columns)[:12]}...)"
        raise KeyError(msg)
    clean = df.dropna(subset=[lon, lat])
    gdf = gpd.GeoDataFrame(
        clean.drop(columns=[lon, lat]) if drop_xy else clean,
        geometry=gpd.points_from_xy(clean[lon], clean[lat]),
        crs=to_crs_obj(crs),
    )
    return gdf


def read_points(
    path: str | Path,
    *,
    lon: str = "longitude",
    lat: str = "latitude",
    crs: str | int = WGS84,
    columns: Sequence[str] | None = None,
    as_dask: bool = False,
    npartitions: int | None = None,
) -> gpd.GeoDataFrame | dgpd.GeoDataFrame:
    """Read a CSV/Parquet table with lon/lat columns as points.

    With ``as_dask=True`` the table is read lazily with dask and geometry is constructed per
    partition, which is how the airspace workflow scales past a few million rows.
    """
    path = Path(path)
    if as_dask:
        import dask.dataframe as dd

        if path.suffix.lower() in PARQUET_SUFFIXES:
            ddf = dd.read_parquet(path, columns=list(columns) if columns else None)
        else:
            ddf = dd.read_csv(path, usecols=list(columns) if columns else None)
        if npartitions:
            ddf = ddf.repartition(npartitions=npartitions)
        return dgpd.from_dask_dataframe(
            ddf, geometry=dgpd.points_from_xy(ddf, lon, lat, crs=to_crs_obj(crs))
        )
    if path.suffix.lower() in PARQUET_SUFFIXES:
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
    else:
        df = pd.read_csv(path, usecols=list(columns) if columns else None)
    return points_from_xy(df, lon=lon, lat=lat, crs=crs)


def read_vector(
    path: str | Path,
    *,
    layer: str | None = None,
    crs: str | int | None = None,
    as_dask: bool = False,
    npartitions: int | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | dgpd.GeoDataFrame:
    """Read any vector source (Shapefile, GPKG, GeoJSON, GeoParquet, ...).

    ``crs`` optionally reprojects after reading. ``as_dask`` returns a partitioned frame.
    """
    path = Path(path)
    if path.suffix.lower() in PARQUET_SUFFIXES:
        gdf = gpd.read_parquet(path, **kwargs)
    else:
        gdf = gpd.read_file(path, layer=layer, **kwargs)
    if crs is not None:
        gdf = gdf.to_crs(to_crs_obj(crs)) if gdf.crs else gdf.set_crs(to_crs_obj(crs))
    if as_dask:
        return to_dask_geo(gdf, npartitions or npartitions_for(len(gdf)))
    return gdf


def write_vector(
    gdf: gpd.GeoDataFrame | dgpd.GeoDataFrame,
    path: str | Path,
    *,
    layer: str | None = None,
    driver: str | None = None,
    to_wgs84: bool = False,
) -> Path:
    """Write a (dask-)GeoDataFrame to GeoParquet, GPKG, GeoJSON or Shapefile by suffix.

    GeoPackage is preferred for QGIS hand-off: it keeps long column names (Shapefile truncates
    to 10 characters, which corrupted attribute names in the legacy oceanography outputs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(gdf, dgpd.GeoDataFrame):
        gdf = gdf.compute()
    if to_wgs84 and gdf.crs is not None:
        gdf = gdf.to_crs(WGS84)
    out = _sanitize_for_write(gdf)
    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        out.to_parquet(path)
    elif suffix == ".gpkg":
        out.to_file(path, layer=layer or path.stem, driver="GPKG")
    elif suffix in {".geojson", ".json"}:
        out.to_file(path, driver="GeoJSON")
    else:
        out.to_file(path, driver=driver)
    return path


def _sanitize_for_write(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """OGR drivers can't serialise tz-aware timestamps, Periods or object-typed bools."""
    out = gdf.copy()
    for col in out.columns:
        if col == out.geometry.name:
            continue
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_convert("UTC").dt.tz_localize(None)
        elif isinstance(s.dtype, pd.PeriodDtype) or (
            s.dtype == object and s.map(type).eq(bool).all()
        ):
            out[col] = s.astype(str) if isinstance(s.dtype, pd.PeriodDtype) else s.astype(int)
        elif s.dtype == bool:
            out[col] = s.astype(int)
    return out


def bbox_to_geodataframe(
    bounds: Sequence[float], *, crs: str | int = WGS84, name: str = "bbox"
) -> gpd.GeoDataFrame:
    """``(minx, miny, maxx, maxy)`` -> single-polygon GeoDataFrame."""
    minx, miny, maxx, maxy = bounds
    return gpd.GeoDataFrame(
        {"name": [name]}, geometry=[box(minx, miny, maxx, maxy)], crs=to_crs_obj(crs)
    )
