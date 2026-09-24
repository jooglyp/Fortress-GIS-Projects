"""Helpers for code that accepts either pandas or dask frames.

:func:`to_dask` and :func:`to_dask_geo` partition an eager frame; :func:`to_pandas` and
:func:`maybe_compute` bring a result back in process; :class:`DaskContext` starts or reuses a
local ``distributed`` cluster for a block of notebook code. The original oceanography scripts
needed a 32 GB machine to hold the whole point cloud; partitioned, the same workflow runs on a
laptop.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd

from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

DEFAULT_ROWS_PER_PARTITION = 250_000


def is_dask(obj: Any) -> bool:
    """True for dask DataFrame / GeoDataFrame / Series objects."""
    return isinstance(obj, dd.DataFrame | dd.Series | dgpd.GeoDataFrame | dgpd.GeoSeries)


def npartitions_for(n_rows: int, rows_per_partition: int = DEFAULT_ROWS_PER_PARTITION) -> int:
    """Partition count that keeps roughly ``rows_per_partition`` rows per task."""
    return max(1, math.ceil(n_rows / max(1, rows_per_partition)))


def to_dask(df: pd.DataFrame | dd.DataFrame, npartitions: int | None = None) -> dd.DataFrame:
    """Lift a pandas frame into a dask frame (no-op for dask input)."""
    if isinstance(df, dd.DataFrame):
        return df.repartition(npartitions=npartitions) if npartitions else df
    return dd.from_pandas(df, npartitions=npartitions or npartitions_for(len(df)))


def to_dask_geo(
    gdf: gpd.GeoDataFrame | dgpd.GeoDataFrame,
    npartitions: int | None = None,
    *,
    spatial_shuffle: bool = False,
) -> dgpd.GeoDataFrame:
    """Lift a GeoDataFrame into a dask-geopandas frame.

    Parameters
    ----------
    npartitions:
        Target partition count (defaults to ~250k rows/partition).
    spatial_shuffle:
        When True, re-partition by Hilbert curve so that spatially close features share a
        partition. This makes subsequent spatial joins much cheaper for large point clouds.
    """
    if isinstance(gdf, dgpd.GeoDataFrame):
        out = gdf.repartition(npartitions=npartitions) if npartitions else gdf
    else:
        out = dgpd.from_geopandas(gdf, npartitions=npartitions or npartitions_for(len(gdf)))
    if spatial_shuffle and out.npartitions > 1:
        out = out.spatial_shuffle()
    return out


def to_pandas(obj: Any) -> Any:
    """Compute dask objects to pandas/geopandas; pass eager objects through unchanged."""
    if is_dask(obj):
        return obj.compute()
    return obj


def maybe_compute(obj: Any, *, compute: bool) -> Any:
    """``obj.compute()`` when ``compute`` is True and ``obj`` is lazy; otherwise return as-is."""
    return to_pandas(obj) if compute else obj


class DaskContext:
    """Start (or reuse) a distributed client for a block of work, with the Bokeh dashboard.

    Thin wrapper over :func:`fortress_gis.compute.cluster.get_dask_client` so that library
    code can write::

        with DaskContext(n_workers=4) as ctx:
            result = heavy_dask_pipeline(...).compute()
            print(ctx.dashboard_link)

    ``enabled=False`` makes it a no-op (threaded scheduler). ``close=True`` tears the cluster
    down on exit; by default it stays up so notebook cells can keep using the dashboard.
    """

    def __init__(self, *, enabled: bool = True, close: bool = False, **client_kwargs: Any) -> None:
        self.enabled = enabled
        self.close_on_exit = close
        self.client_kwargs = client_kwargs
        self.client: Any | None = None

    @property
    def dashboard_link(self) -> str | None:
        return getattr(self.client, "dashboard_link", None)

    def __enter__(self) -> DaskContext:
        if not self.enabled:
            return self
        from fortress_gis.compute.cluster import get_dask_client

        self.client = get_dask_client(**self.client_kwargs)
        return self

    def __exit__(self, *exc: object) -> None:
        if self.client is not None and self.close_on_exit:
            from fortress_gis.compute.cluster import close_client

            close_client(self.client)
            self.client = None


@contextmanager
def threaded_scheduler() -> Iterator[None]:
    """Force the single-machine threaded scheduler (useful inside tests)."""
    import dask

    with dask.config.set(scheduler="threads"):
        yield
