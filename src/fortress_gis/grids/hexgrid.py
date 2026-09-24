"""Regular hexagonal tessellations in a metric CRS.

Two of the original projects carried the same hand-written ``calculate_polygons`` loop, walking
the extent row by row in Python. :func:`hex_grid` builds the same pointy-top lattice with numpy;
a city-scale grid at 100 m spacing takes milliseconds. :func:`hex_radius_from_area` and
:func:`hex_radius_from_travel_time` turn a target area, or a typical speed and crossing time,
into a radius so the reason for a cell size is in the code and not in a comment.

``radius`` throughout is the circumradius (centre to vertex) in CRS units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS

from fortress_gis.crs import WGS84, is_metric, suggest_metric_crs, to_crs_obj

KNOTS_TO_MPS = 0.514444
SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class HexSpec:
    """Geometry of a pointy-top regular hexagon with circumradius ``radius`` (CRS units)."""

    radius: float

    @property
    def side(self) -> float:
        return self.radius

    @property
    def width(self) -> float:
        """Flat-to-flat width = horizontal spacing between neighbouring centres in a row."""
        return SQRT3 * self.radius

    @property
    def height(self) -> float:
        """Vertex-to-vertex height."""
        return 2.0 * self.radius

    @property
    def row_spacing(self) -> float:
        """Vertical distance between successive rows of centres."""
        return 1.5 * self.radius

    @property
    def area(self) -> float:
        return hex_area_from_radius(self.radius)


def hex_area_from_radius(radius: float) -> float:
    """Area of a regular hexagon from its circumradius."""
    return (3.0 * SQRT3 / 2.0) * radius**2


def hex_radius_from_area(area: float) -> float:
    """Circumradius of a regular hexagon with the given area."""
    if area <= 0:
        msg = "area must be positive"
        raise ValueError(msg)
    return math.sqrt(2.0 * area / (3.0 * SQRT3))


def hex_radius_from_travel_time(
    speed: float,
    minutes: float,
    *,
    speed_units: str = "knots",
    divisor: float = 2.0,
) -> float:
    """Circumradius such that crossing a cell (~2 radii) takes ``minutes`` at ``speed``.

    Parameters
    ----------
    speed:
        Typical speed (e.g. mean per-track ground speed).
    minutes:
        Target crossing time - e.g. 7 minutes for a response window.
    speed_units:
        ``"knots"``, ``"mps"`` or ``"kph"``.
    divisor:
        Distance travelled is divided by this to get the radius; 2 means the diameter equals the
        distance travelled in ``minutes`` (the cell is crossed in that time).
    """
    factor = {"knots": KNOTS_TO_MPS, "mps": 1.0, "kph": 1000.0 / 3600.0}[speed_units]
    distance_m = speed * factor * minutes * 60.0
    return distance_m / divisor


def hex_vertices(cx: np.ndarray, cy: np.ndarray, radius: float) -> np.ndarray:
    """Vertex coordinates (n, 6, 2) for pointy-top hexagons centred at (cx, cy)."""
    angles = np.deg2rad(np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]))
    dx = radius * np.cos(angles)
    dy = radius * np.sin(angles)
    return np.stack(
        [cx[:, None] + dx[None, :], cy[:, None] + dy[None, :]],
        axis=-1,
    )


def hex_grid(
    bounds: tuple[float, float, float, float],
    radius: float,
    *,
    crs: str | int | CRS,
    id_column: str = "hex_id",
    clip_to_bounds: bool = False,
    pad_cells: int = 1,
) -> gpd.GeoDataFrame:
    """Tessellate ``bounds`` (in ``crs``, metric) with pointy-top hexagons of ``radius``.

    Returns a GeoDataFrame with ``hex_id`` (row-major integer), ``row``/``col`` lattice indices and
    the cell centroid coordinates. ``pad_cells`` extra rings guarantee full coverage of the extent.
    """
    if not is_metric(crs):
        msg = f"hex_grid needs a metric CRS, got {crs!r}. Use suggest_metric_crs() first."
        raise ValueError(msg)
    if radius <= 0:
        msg = "radius must be positive"
        raise ValueError(msg)
    spec = HexSpec(radius)
    minx, miny, maxx, maxy = bounds
    minx -= pad_cells * spec.width
    maxx += pad_cells * spec.width
    miny -= pad_cells * spec.height
    maxy += pad_cells * spec.height

    n_rows = math.ceil((maxy - miny) / spec.row_spacing) + 1
    n_cols = math.ceil((maxx - minx) / spec.width) + 1
    rows = np.arange(n_rows)
    cols = np.arange(n_cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    cx = minx + cc * spec.width + (rr % 2) * (spec.width / 2.0)
    cy = miny + rr * spec.row_spacing
    cx = cx.ravel()
    cy = cy.ravel()
    verts = hex_vertices(cx, cy, radius)
    geoms = shapely.polygons(verts)
    gdf = gpd.GeoDataFrame(
        {
            id_column: np.arange(len(geoms), dtype="int64"),
            "row": rr.ravel(),
            "col": cc.ravel(),
            "centroid_x": cx,
            "centroid_y": cy,
        },
        geometry=geoms,
        crs=to_crs_obj(crs),
    )
    if clip_to_bounds:
        window = shapely.box(*bounds)
        gdf = gdf[shapely.intersects(gdf.geometry.to_numpy(), window)].reset_index(drop=True)
        gdf[id_column] = np.arange(len(gdf), dtype="int64")
    return gdf


def hex_grid_for_layer(
    layer: gpd.GeoDataFrame,
    *,
    radius: float | None = None,
    area: float | None = None,
    metric_crs: str | int | CRS | None = None,
    output_crs: str | int | CRS | None = WGS84,
    min_cells: int = 4,
    id_column: str = "hex_id",
    clip_to_bounds: bool = True,
) -> gpd.GeoDataFrame:
    """Build a hex grid covering a layer's extent, choosing a metric CRS automatically.

    Exactly one of ``radius`` (metres) or ``area`` (square metres) must be given. If the layer is
    smaller than ``min_cells`` hexagons its bounding box is inflated (mirrors the legacy
    "rescale the extent" behaviour) so that hotspot statistics still have neighbours to use.
    ``output_crs`` defaults to WGS84, the CRS Kepler.gl expects; pass ``None`` to keep metric.
    """
    if (radius is None) == (area is None):
        msg = "Provide exactly one of radius or area"
        raise ValueError(msg)
    r = radius if radius is not None else hex_radius_from_area(float(area))  # type: ignore[arg-type]
    m_crs = metric_crs or suggest_metric_crs(layer)
    projected = layer.to_crs(to_crs_obj(m_crs))
    minx, miny, maxx, maxy = projected.total_bounds
    extent_area = max((maxx - minx) * (maxy - miny), 1e-9)
    required = min_cells * hex_area_from_radius(r)
    if extent_area < required:
        scale = math.sqrt(required * 1.1 / extent_area)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        hw, hh = (maxx - minx) * scale / 2.0, (maxy - miny) * scale / 2.0
        minx, maxx, miny, maxy = cx - hw, cx + hw, cy - hh, cy + hh
    grid = hex_grid(
        (minx, miny, maxx, maxy), r, crs=m_crs, id_column=id_column, clip_to_bounds=clip_to_bounds
    )
    grid.attrs["metric_crs"] = to_crs_obj(m_crs).to_string()
    grid.attrs["radius_m"] = r
    if output_crs is not None:
        out = grid.to_crs(to_crs_obj(output_crs))
        out.attrs.update(grid.attrs)
        return out
    return grid
