"""Zonal statistics and raster-mask vectorisation."""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import shape

from fortress_gis.crs import to_crs_obj
from fortress_gis.io.raster import RasterInfo

_STAT_FUNCS = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "median": np.nanmedian,
    "sum": np.nansum,
    "count": lambda a: int(np.isfinite(a).sum()),
}


def zonal_statistics(
    zones: gpd.GeoDataFrame,
    raster: RasterInfo,
    *,
    stats: Sequence[str] = ("mean", "min", "max", "count"),
    prefix: str = "z",
    all_touched: bool = False,
) -> gpd.GeoDataFrame:
    """Per-polygon statistics of the raster cells inside each zone.

    Zones are rasterised once onto the raster lattice with ``rasterio.features.rasterize`` and
    the statistics come from one grouped reduction, so the cost is O(cells).
    """
    zs = zones.to_crs(to_crs_obj(raster.crs)) if (zones.crs is not None and raster.crs) else zones
    labels = features.rasterize(
        ((geom.__geo_interface__, i + 1) for i, geom in enumerate(zs.geometry)),
        out_shape=raster.shape,
        transform=raster.transform,
        fill=0,
        all_touched=all_touched,
        dtype="int32",
    )
    values = raster.masked()
    inside = (labels > 0) & np.isfinite(values)
    frame = pd.DataFrame({"zone": labels[inside], "v": values[inside]})
    grouped = frame.groupby("zone")["v"]
    result = pd.DataFrame(index=np.arange(1, len(zs) + 1))
    for s in stats:
        if s == "count":
            result[f"{prefix}_{s}"] = grouped.size()
        elif s == "median":
            result[f"{prefix}_{s}"] = grouped.median()
        else:
            result[f"{prefix}_{s}"] = getattr(grouped, s)()
    result.index = zs.index
    out = zones.copy()
    for col in result.columns:
        out[col] = result[col].to_numpy()
    if "count" in stats:
        out[f"{prefix}_count"] = out[f"{prefix}_count"].fillna(0).astype("int64")
    return out


def mask_to_polygons(
    mask: RasterInfo, *, value: int = 1, dissolve: bool = True
) -> gpd.GeoDataFrame:
    """Vectorise the cells equal to ``value`` into polygons (``gdalogr:polygonize`` equivalent)."""
    data = mask.data.astype("uint8")
    shapes = features.shapes(data, mask=data == value, transform=mask.transform)
    geoms = [shape(geom) for geom, _ in shapes]
    gdf = gpd.GeoDataFrame(
        {"value": [value] * len(geoms)},
        geometry=geoms,
        crs=to_crs_obj(mask.crs) if mask.crs else None,
    )
    if dissolve and len(gdf) > 1:
        gdf = gdf.dissolve(by="value").reset_index()
    return gdf
