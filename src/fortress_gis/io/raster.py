"""Raster I/O built on rasterio.

These replace the QGIS and SAGA processing calls (``saga:mosaickrasterlayers``,
``saga:clipgridwithpolygon``, ``grass7:v.sample``) with rasterio and numpy, so they run without
a desktop GIS and can be unit tested.

Functions take and return :class:`RasterInfo`, an array with its affine transform, CRS and
nodata value. Intermediate rasters stay in memory; nothing touches disk until an export.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS as RioCRS
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.transform import rowcol, xy

from fortress_gis.crs import to_crs_obj
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class RasterInfo:
    """A single-band raster held in memory."""

    data: np.ndarray
    transform: Affine
    crs: str
    nodata: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def resolution(self) -> tuple[float, float]:
        """(x_res, y_res) in CRS units; y is positive."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.shape
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (cols, rows)
        return (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

    @property
    def mask(self) -> np.ndarray:
        """Boolean array, True where data is valid."""
        valid = np.isfinite(self.data)
        if self.nodata is not None and not np.isnan(self.nodata):
            valid &= self.data != self.nodata
        return valid

    def masked(self) -> np.ndarray:
        """Float copy with nodata as NaN - the representation used by terrain algorithms."""
        out = self.data.astype("float64", copy=True)
        out[~self.mask] = np.nan
        return out

    def with_data(self, data: np.ndarray, *, nodata: float | None = np.nan) -> RasterInfo:
        return replace(self, data=data, nodata=nodata)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Arrays of x and y coordinates for every cell centre (shape == data.shape)."""
        rows, cols = np.indices(self.shape)
        xs, ys = xy(self.transform, rows.ravel(), cols.ravel(), offset="center")
        return np.asarray(xs).reshape(self.shape), np.asarray(ys).reshape(self.shape)

    def index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised (row, col) lookup for coordinates in the raster CRS."""
        rows, cols = rowcol(self.transform, np.asarray(x), np.asarray(y))
        return np.asarray(rows), np.asarray(cols)


def raster_info(path: str | Path, band: int = 1) -> RasterInfo:
    """Load one band fully into memory."""
    with rasterio.open(path) as src:
        data = src.read(band)
        return RasterInfo(
            data=data,
            transform=src.transform,
            crs=src.crs.to_string() if src.crs else "",
            nodata=src.nodata,
        )


read_raster = raster_info


def write_raster(
    info: RasterInfo, path: str | Path, *, dtype: str | None = None, compress: str = "deflate"
) -> Path:
    """Write a :class:`RasterInfo` as a tiled, compressed GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = info.data
    out_dtype = dtype or ("float32" if np.issubdtype(data.dtype, np.floating) else str(data.dtype))
    nodata = info.nodata
    if np.issubdtype(np.dtype(out_dtype), np.floating):
        data = data.astype(out_dtype)
        if nodata is None or np.isnan(nodata):
            nodata = -9999.0
            data = np.where(np.isfinite(data), data, nodata)
    else:
        data = data.astype(out_dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=out_dtype,
        crs=RioCRS.from_user_input(to_crs_obj(info.crs)) if info.crs else None,
        transform=info.transform,
        nodata=nodata,
        compress=compress,
        tiled=True,
    ) as dst:
        dst.write(data, 1)
    return path


def merge_rasters(
    paths: Iterable[str | Path], *, nodata: float | None = None, method: str = "first"
) -> RasterInfo:
    """Mosaic several tiles (e.g. 1-arc-second DEM tiles) into one in-memory raster.

    Equivalent of the legacy ``saga:mosaickrasterlayers`` step but without having to compute the
    union extent by hand - rasterio does that from the tile transforms.
    """
    sources = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = rio_merge(sources, nodata=nodata, method=method)
        crs = sources[0].crs.to_string() if sources[0].crs else ""
        nd = nodata if nodata is not None else sources[0].nodata
    finally:
        for s in sources:
            s.close()
    return RasterInfo(data=mosaic[0], transform=transform, crs=crs, nodata=nd)


def clip_raster_to_geometry(
    src: RasterInfo | str | Path,
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    *,
    all_touched: bool = False,
) -> RasterInfo:
    """Clip a raster by polygon(s), reprojecting the mask to the raster CRS automatically.

    Equivalent of ``saga:clipgridwithpolygon``. The legacy scripts had to reproject the county
    boundary as a separate processing step; here it is implicit.
    """
    info = raster_info(src) if not isinstance(src, RasterInfo) else src
    geoms = geometry.geometry if isinstance(geometry, gpd.GeoDataFrame) else geometry
    if geoms.crs is not None and info.crs:
        geoms = geoms.to_crs(to_crs_obj(info.crs))
    shapes = [g.__geo_interface__ for g in geoms if g is not None and not g.is_empty]
    with _memory_dataset(info) as ds:
        out, transform = rio_mask(ds, shapes, crop=True, all_touched=all_touched, nodata=ds.nodata)
    return RasterInfo(data=out[0], transform=transform, crs=info.crs, nodata=info.nodata)


def raster_to_points(
    info: RasterInfo,
    *,
    value_name: str = "value",
    drop_nodata: bool = True,
    as_dask: bool = False,
    chunk_rows: int = 512,
) -> gpd.GeoDataFrame | dgpd.GeoDataFrame:
    """Explode a raster into one point per cell centre.

    This is the modern equivalent of the legacy oceanography ``flatten_data`` loop, which walked
    every pixel in Python. Here the coordinate arithmetic is vectorised and, with
    ``as_dask=True``, computed in row-chunks so multi-gigabyte grids never need to be resident.
    """
    if not as_dask:
        xs, ys = info.cell_centers()
        vals = info.masked()
        keep = np.isfinite(vals) if drop_nodata else np.ones(vals.shape, dtype=bool)
        df = pd.DataFrame({"x": xs[keep], "y": ys[keep], value_name: vals[keep]})
        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=to_crs_obj(info.crs) if info.crs else None,
        )

    rows, cols = info.shape
    arr = da.from_array(info.masked(), chunks=(chunk_rows, cols))
    row_idx = da.arange(rows, chunks=chunk_rows)
    transform = info.transform

    def _block_to_frame(block: np.ndarray, block_rows: np.ndarray) -> pd.DataFrame:
        r, c = np.meshgrid(block_rows, np.arange(block.shape[1]), indexing="ij")
        xs, ys = xy(transform, r.ravel(), c.ravel(), offset="center")
        vals = block.ravel()
        df = pd.DataFrame({"x": np.asarray(xs), "y": np.asarray(ys), value_name: vals})
        return df[np.isfinite(df[value_name])] if drop_nodata else df

    meta = pd.DataFrame(
        {
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
            value_name: pd.Series(dtype="float64"),
        }
    )
    frames = [
        dd.from_delayed(
            _delayed_block(_block_to_frame, arr.blocks[i, 0], row_idx.blocks[i]), meta=meta
        )
        for i in range(arr.numblocks[0])
    ]
    ddf = dd.concat(frames)
    return dgpd.from_dask_dataframe(
        ddf,
        geometry=dgpd.points_from_xy(ddf, "x", "y", crs=to_crs_obj(info.crs) if info.crs else None),
    )


def _delayed_block(fn: Any, block: Any, rows: Any) -> Any:
    import dask

    return dask.delayed(fn)(block, rows)


def sample_raster_at_points(
    info: RasterInfo,
    points: gpd.GeoDataFrame,
    *,
    column: str = "raster_value",
) -> gpd.GeoDataFrame:
    """Attach the raster value under each point (nearest cell) - replaces ``grass7:v.sample``.

    Points are reprojected to the raster CRS; points outside the grid or on nodata get NaN.
    """
    pts = points.to_crs(to_crs_obj(info.crs)) if (points.crs is not None and info.crs) else points
    x = pts.geometry.x.to_numpy()
    y = pts.geometry.y.to_numpy()
    rows, cols = info.index(x, y)
    values = np.full(len(pts), np.nan)
    inside = (rows >= 0) & (rows < info.shape[0]) & (cols >= 0) & (cols < info.shape[1])
    grid = info.masked()
    values[inside] = grid[rows[inside], cols[inside]]
    out = points.copy()
    out[column] = values
    return out


def _memory_dataset(info: RasterInfo) -> Any:
    """Open an in-memory rasterio dataset over a RasterInfo (context manager)."""
    from rasterio.io import MemoryFile

    class _Ctx:
        def __enter__(self) -> Any:
            self.mem = MemoryFile()
            data = info.data
            nodata = info.nodata
            if np.issubdtype(data.dtype, np.floating) and (nodata is None or np.isnan(nodata)):
                nodata = -9999.0
                data = np.where(np.isfinite(data), data, nodata)
            self.ds = self.mem.open(
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=str(data.dtype),
                crs=RioCRS.from_user_input(to_crs_obj(info.crs)) if info.crs else None,
                transform=info.transform,
                nodata=nodata,
            )
            self.ds.write(data, 1)
            return self.ds

        def __exit__(self, *exc: object) -> None:
            self.ds.close()
            self.mem.close()

    return _Ctx()


def raster_from_array(
    data: np.ndarray,
    *,
    bounds: Sequence[float],
    crs: str | int,
    nodata: float | None = np.nan,
) -> RasterInfo:
    """Build a north-up RasterInfo from an array and its ``(minx, miny, maxx, maxy)`` bounds."""
    from rasterio.transform import from_bounds

    rows, cols = data.shape
    transform = from_bounds(*bounds, width=cols, height=rows)
    return RasterInfo(
        data=data, transform=transform, crs=to_crs_obj(crs).to_string(), nodata=nodata
    )
