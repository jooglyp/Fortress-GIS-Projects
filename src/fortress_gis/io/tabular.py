"""Readers for gridded tables: CSV exports of raster products where the first column is one
axis coordinate and the remaining column headers are the other axis coordinate.

Satellite products such as seasonal chlorophyll-a composites are often distributed this way.
The original code iterated over every cell in Python; here the wide table is melted with pandas,
or with dask for large tiles.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd

from fortress_gis.crs import WGS84
from fortress_gis.io.vector import points_from_xy
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

_DEFAULT_NAME_PATTERN = re.compile(r"(?P<year>\d{4})(?P<doy>\d{3})")


def value_name_from_filename(
    path: str | Path, *, prefix: str = "chlor_", pattern: re.Pattern[str] = _DEFAULT_NAME_PATTERN
) -> str:
    """Derive a column name such as ``chlor_2019105`` from ``A2019105_chlor_a.csv``-style names.

    Falls back to the file stem when no ``YYYYDDD`` token is found.
    """
    stem = Path(path).stem
    match = pattern.search(stem)
    token = match.group(0) if match else stem
    return f"{prefix}{token}"


def read_grid_csv_as_points(
    path: str | Path,
    *,
    value_name: str | None = None,
    first_axis: str = "longitude",
    second_axis: str = "latitude",
    crs: str | int = WGS84,
    as_dask: bool = False,
) -> gpd.GeoDataFrame | dgpd.GeoDataFrame:
    """Melt a wide grid CSV into points with one value column.

    Parameters
    ----------
    first_axis / second_axis:
        Which coordinate the first column holds and which the column headers hold. The
        defaults match the original product layout: rows are longitude, headers are latitude.
    """
    name = value_name or value_name_from_filename(path)
    if as_dask:
        ddf = dd.read_csv(path)
        first_col = ddf.columns[0]
        melted = ddf.melt(id_vars=[first_col], var_name=second_axis, value_name=name)
        melted = melted.rename(columns={first_col: first_axis})
        melted[second_axis] = melted[second_axis].astype("float64")
        melted[first_axis] = melted[first_axis].astype("float64")
        melted = melted.dropna(subset=[name])
        return dgpd.from_dask_dataframe(
            melted, geometry=dgpd.points_from_xy(melted, "longitude", "latitude", crs=crs)
        )
    df = pd.read_csv(path)
    first_col = df.columns[0]
    melted = df.melt(id_vars=[first_col], var_name=second_axis, value_name=name).rename(
        columns={first_col: first_axis}
    )
    melted[second_axis] = pd.to_numeric(melted[second_axis], errors="coerce")
    melted[first_axis] = pd.to_numeric(melted[first_axis], errors="coerce")
    melted = melted.dropna(subset=[name, first_axis, second_axis])
    return points_from_xy(melted, lon="longitude", lat="latitude", crs=crs)


def read_many_files_dask(
    paths: Sequence[str | Path],
    *,
    clean: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    meta: pd.DataFrame | None = None,
    dtype: Mapping[str, str] | None = None,
    **read_kwargs: Any,
) -> dd.DataFrame:
    """Read many small CSV/Parquet files as one dask frame - one partition per file.

    ``dd.read_csv([...])`` maps each file to a partition, which parallelises reads without
    materialising the corpus on the driver. ``clean`` is applied per partition via
    ``map_partitions`` with an explicit ``meta`` (pass a zero-row frame with the output dtypes)
    so dask never has to infer the schema by running the function on fake data.
    """
    files = [str(p) for p in paths if Path(p).stat().st_size > 0]
    if not files:
        msg = "No non-empty files provided"
        raise FileNotFoundError(msg)
    if Path(files[0]).suffix.lower() in {".parquet", ".pq"}:
        ddf = dd.read_parquet(files, **read_kwargs)
    else:
        ddf = dd.read_csv(files, dtype=dict(dtype) if dtype else None, **read_kwargs)
    if clean is not None:
        ddf = (
            ddf.map_partitions(clean, meta=meta) if meta is not None else ddf.map_partitions(clean)
        )
    return ddf


def read_grid_csvs_as_points(
    paths: Iterable[str | Path],
    *,
    prefix: str = "chlor_",
    crs: str | int = WGS84,
    limit: int | None = None,
) -> gpd.GeoDataFrame:
    """Read several grid CSVs sharing one lattice and join them into a wide point layer.

    Each file contributes one value column named from its ``YYYYDDD`` token. Files are joined on
    exact (lon, lat) coordinates so the lattice must be identical across files.
    """
    frames: list[pd.DataFrame] = []
    for i, p in enumerate(sorted(Path(x) for x in paths)):
        if limit is not None and i >= limit:
            break
        name = value_name_from_filename(p, prefix=prefix)
        gdf = read_grid_csv_as_points(p, value_name=name, crs=crs)
        frames.append(
            pd.DataFrame(gdf.drop(columns=gdf.geometry.name)).set_index(["longitude", "latitude"])
        )
        LOGGER.info("Read %s -> %s (%d cells)", p.name, name, len(gdf))
    if not frames:
        msg = "No grid CSV files were provided"
        raise ValueError(msg)
    wide = pd.concat(frames, axis=1, join="outer").reset_index()
    return points_from_xy(wide, lon="longitude", lat="latitude", crs=crs)
