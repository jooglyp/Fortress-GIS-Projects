"""CRS constants and helpers.

Each of the original projects hard-coded WKT or proj4 strings for the same few CRSs. They are
collected here as EPSG codes, with helpers that choose a metric CRS for distance and area work.
"""

from __future__ import annotations

import math
from typing import Final

import geopandas as gpd
from pyproj import CRS

WGS84: Final[str] = "EPSG:4326"
"""Geographic lon/lat on the WGS84 ellipsoid - the interchange CRS for Kepler.gl and GeoJSON."""

NAD83: Final[str] = "EPSG:4269"
"""Geographic lon/lat on NAD83 - the CRS of most US federal DEM/flowline products."""

WEB_MERCATOR: Final[str] = "EPSG:3857"
"""Spherical Mercator used by web basemaps; metric but strongly distorted away from the equator."""

US_ALBERS_EQUAL_AREA: Final[str] = (
    "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 "
    "+ellps=GRS80 +datum=NAD83 +units=m +no_defs"
)
"""USA Contiguous Albers Equal Area Conic: metric and area-preserving for CONUS."""

WORLD_EQUAL_AREA: Final[str] = "EPSG:6933"
"""World Cylindrical Equal Area - metric, area-preserving fallback for arbitrary extents."""


def to_crs_obj(crs: str | int | CRS) -> CRS:
    """Coerce an EPSG int, authority string or proj4 string into a :class:`pyproj.CRS`."""
    if isinstance(crs, CRS):
        return crs
    if isinstance(crs, int):
        return CRS.from_epsg(crs)
    return CRS.from_user_input(crs)


def is_metric(crs: str | int | CRS | None) -> bool:
    """True when the CRS is projected with linear metre units."""
    if crs is None:
        return False
    obj = to_crs_obj(crs)
    if not obj.is_projected:
        return False
    axis = obj.axis_info[0] if obj.axis_info else None
    unit_name = (axis.unit_name if axis else "").lower()
    return unit_name in {"metre", "meter", "m"}


def utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat) on WGS84."""
    zone = math.floor((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def suggest_metric_crs(gdf: gpd.GeoDataFrame | gpd.GeoSeries) -> str:
    """Pick a metric CRS appropriate for a layer's extent.

    Extents narrower than about 6 degrees of longitude get the UTM zone at their centroid, which
    is right for local distance and buffer work such as airspace conflicts or watersheds. Wider
    extents get an equal-area projection so hexagon areas are comparable across the layer.
    """
    if gdf.crs is None:
        msg = "Layer has no CRS; set one (e.g. EPSG:4326) before choosing a metric CRS."
        raise ValueError(msg)
    bounds = gdf.to_crs(WGS84).total_bounds
    minx, miny, maxx, maxy = bounds
    if (maxx - minx) <= 6.0 and (maxy - miny) <= 6.0:
        return f"EPSG:{utm_epsg_for((minx + maxx) / 2.0, (miny + maxy) / 2.0)}"
    if minx >= -130 and maxx <= -60 and miny >= 20 and maxy <= 55:
        return US_ALBERS_EQUAL_AREA
    return WORLD_EQUAL_AREA


def ensure_crs(
    gdf: gpd.GeoDataFrame, crs: str | int | CRS, *, assume: str | int | CRS = WGS84
) -> gpd.GeoDataFrame:
    """Return ``gdf`` in ``crs``. A layer with no CRS is taken to be in ``assume`` first."""
    if gdf.crs is None:
        gdf = gdf.set_crs(to_crs_obj(assume))
    target = to_crs_obj(crs)
    if gdf.crs == target:
        return gdf
    return gdf.to_crs(target)
