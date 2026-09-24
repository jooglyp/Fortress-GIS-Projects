"""Execution backends: pandas and geopandas in process, or dask and dask-geopandas out of core.

``backend`` converts frames between pandas and dask. ``cluster`` manages the ``distributed``
client, the Bokeh dashboard link (with the WSL address quirk handled), worker memory policy and
batched ``delayed`` execution.
"""

from fortress_gis.compute.backend import (
    DaskContext,
    is_dask,
    maybe_compute,
    npartitions_for,
    threaded_scheduler,
    to_dask,
    to_dask_geo,
    to_pandas,
)
from fortress_gis.compute.cluster import (
    close_client,
    compute_in_batches,
    configure_memory_policy,
    dashboard_urls,
    get_dask_client,
    maybe_client,
    show_dashboard,
    wait_for_workers,
    worker_memory_limit,
)

__all__ = [
    "DaskContext",
    "close_client",
    "compute_in_batches",
    "configure_memory_policy",
    "dashboard_urls",
    "get_dask_client",
    "is_dask",
    "maybe_client",
    "maybe_compute",
    "npartitions_for",
    "show_dashboard",
    "threaded_scheduler",
    "to_dask",
    "to_dask_geo",
    "to_pandas",
    "wait_for_workers",
    "worker_memory_limit",
]
