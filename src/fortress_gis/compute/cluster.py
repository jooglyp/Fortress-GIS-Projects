"""Dask client lifecycle and the Bokeh diagnostics dashboard.

The behaviour here was lifted from a mobility-model pipeline that runs on WSL for hours at a
time. :func:`get_dask_client` returns ``Client.current()`` if one exists, otherwise connects to
``DASK_SCHEDULER_ADDRESS`` or starts a ``LocalCluster`` with the dashboard bound to a fixed
port (``127.0.0.1:8787``) so a notebook can link to it. On WSL the same dashboard is reached
from a Windows browser at ``localhost:8787/status``, and :func:`dashboard_urls` returns both
forms. Worker memory limits come from ``/proc/meminfo`` minus a reserve, and the spill/pause
thresholds are set low enough that the Linux OOM killer does not get there first.

Environment variables read at startup:

``DASK_N_WORKERS``            worker count (default ``min(4, cpu_count // 2)``)
``DASK_THREADS_PER_WORKER``   threads per worker (default 1; most of this work is numpy/GEOS)
``DASK_MEMORY_LIMIT``         explicit per-worker limit such as ``"4GB"``
``DASK_MEMORY_BUDGET_GB``     total budget from which the per-worker limit is derived
``DASK_RESERVE_GB``           memory left for the driver and OS (default 4)
``DASK_ENABLE_DASHBOARD``     ``1`` or ``0`` (default ``1``)
``DASK_DASHBOARD_ADDRESS``    bind address (default ``127.0.0.1:8787``)
``DASK_OPEN_DASHBOARD``       ``1`` opens the dashboard in a browser when a cluster starts
``DASK_SCHEDULER_ADDRESS``    ``tcp://host:8786`` of a scheduler to connect to instead
"""

from __future__ import annotations

import gc
import os
import time
import webbrowser
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import dask

from fortress_gis.log import get_logger

if TYPE_CHECKING:
    from distributed import Client

LOGGER = get_logger(__name__)

DEFAULT_DASHBOARD_ADDRESS = "127.0.0.1:8787"
DEFAULT_MAX_LOCAL_WORKERS = 4


def env_bool(name: str, default: bool = False) -> bool:
    """Parse ``1/true/yes/y`` (case-insensitive) from the environment."""
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def is_wsl() -> bool:
    """True when running under Windows Subsystem for Linux."""
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


# --------------------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------------------


def default_worker_count() -> int:
    """``DASK_N_WORKERS`` or ``min(4, max(1, cpu_count // 2))``."""
    raw = os.environ.get("DASK_N_WORKERS")
    if raw:
        return max(1, int(raw))
    nproc = os.cpu_count() or 4
    return min(DEFAULT_MAX_LOCAL_WORKERS, max(1, nproc // 2))


def system_memory_gb() -> float:
    """Total system memory from ``/proc/meminfo`` (fallback 16 GB)."""
    raw = os.environ.get("DASK_MEMORY_BUDGET_GB")
    if raw:
        return float(raw)
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 16.0


def _format_gb(gb: float) -> str:
    rounded = round(max(float(gb), 1.0), 1)
    return f"{int(rounded)}GB" if rounded.is_integer() else f"{rounded}GB"


def worker_memory_limit(n_workers: int) -> str:
    """Per-worker memory limit: ``DASK_MEMORY_LIMIT`` or ``(budget - reserve) / n_workers``."""
    explicit = os.environ.get("DASK_MEMORY_LIMIT")
    if explicit:
        return explicit
    reserve = float(os.environ.get("DASK_RESERVE_GB", "4"))
    budget = max(1.0, system_memory_gb() - reserve)
    return _format_gb(budget / max(1, n_workers))


def configure_memory_policy() -> None:
    """Apply WSL-safe ``distributed`` memory thresholds and disk-based shuffles."""
    dask.config.set(
        {
            "distributed.worker.memory.target": 0.60,
            "distributed.worker.memory.spill": 0.70,
            "distributed.worker.memory.pause": 0.80,
            "distributed.worker.memory.terminate": 0.95,
            "dataframe.shuffle.method": "disk",
        }
    )


# --------------------------------------------------------------------------------------
# dashboard
# --------------------------------------------------------------------------------------


def resolve_dashboard_address(
    enable: bool | None = None, address: str | None = None
) -> tuple[bool, str]:
    """``(enabled, bind_address)`` for ``LocalCluster(dashboard_address=...)``.

    Disabled clusters bind ``":0"`` (random free port, no link printed).
    """
    enabled = env_bool("DASK_ENABLE_DASHBOARD", default=True) if enable is None else enable
    if not enabled:
        return False, ":0"
    return True, address or os.environ.get("DASK_DASHBOARD_ADDRESS", DEFAULT_DASHBOARD_ADDRESS)


def dashboard_urls(dashboard_link: str) -> tuple[str, str]:
    """Normalise a Bokeh dashboard link to ``(wsl_url, windows_url)``.

    ``client.dashboard_link`` already ends in ``/status``; naive host:port parsing produces
    ``/status/status``. Both returned URLs end in exactly one ``/status``.
    """
    parsed = urlparse(dashboard_link.strip())
    port = parsed.port or 8787
    path = parsed.path.rstrip("/") or "/status"
    if not path.endswith("/status"):
        path = f"{path}/status"
    wsl_url = urlunparse(
        (parsed.scheme or "http", parsed.netloc or f"127.0.0.1:{port}", path, "", "", "")
    )
    windows_url = urlunparse(("http", f"localhost:{port}", path, "", "", ""))
    return wsl_url, windows_url


def show_dashboard(client: Client, *, open_browser: bool = False) -> str:
    """Print (and in Jupyter render) a clickable dashboard link; optionally open a browser.

    On WSL the primary link is the ``localhost`` form so Cursor/VS Code opens it in Windows.
    """
    wsl_url, windows_url = dashboard_urls(client.dashboard_link)
    wsl = is_wsl()
    primary = windows_url if wsl else wsl_url
    if wsl:
        print("Dask dashboard (Windows browser):", windows_url)
        print("Dask dashboard (WSL):", wsl_url)
    else:
        print("Dask dashboard:", wsl_url)
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is not None:
            extra = (
                f' &nbsp;|&nbsp; <a href="{wsl_url}" target="_blank">WSL 127.0.0.1</a>'
                if wsl
                else ""
            )
            display(
                HTML(
                    f'<p><b>Dask dashboard:</b> <a href="{primary}" target="_blank">{primary}</a>'
                    f"{extra}</p>"
                )
            )
    except ImportError:  # pragma: no cover
        pass
    if open_browser or env_bool("DASK_OPEN_DASHBOARD", default=False):
        try:
            if wsl:
                os.system(f'cmd.exe /c start "" "{windows_url}"')
            else:
                webbrowser.open(primary)
        except OSError:  # pragma: no cover
            webbrowser.open(primary)
    return primary


# --------------------------------------------------------------------------------------
# client lifecycle
# --------------------------------------------------------------------------------------


def _existing_client(min_workers: int | None) -> Client | None:
    from distributed import Client

    try:
        client = Client.current()
    except (ValueError, AttributeError):
        return None
    if client.status != "running":
        return None
    workers = client.scheduler_info().get("workers", {})
    if not workers:
        LOGGER.warning("Existing dask client has no workers; closing it")
        client.close()
        return None
    if min_workers is not None and len(workers) < min_workers:
        LOGGER.warning(
            "Existing dask client has %d workers (< %d); replacing it", len(workers), min_workers
        )
        client.close()
        return None
    LOGGER.info("Reusing existing dask client: %s", client.scheduler.address)
    return client


def get_dask_client(
    *,
    address: str | None = None,
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    memory_limit: str | None = None,
    enable_dashboard: bool | None = None,
    dashboard_address: str | None = None,
    show_dashboard_link: bool = True,
    min_workers: int | None = None,
    reuse_existing: bool = True,
    timeout: int = 30,
    **cluster_kwargs: Any,
) -> Client:
    """Return a ``distributed.Client`` - reused, connected to ``address``, or a new LocalCluster.

    Resolution order: existing running client -> ``address`` / ``DASK_SCHEDULER_ADDRESS`` ->
    a new :class:`distributed.LocalCluster` sized from the environment. When a local cluster is
    started with the dashboard enabled the link is printed via :func:`show_dashboard`.
    """
    from distributed import Client, LocalCluster

    if reuse_existing:
        existing = _existing_client(min_workers)
        if existing is not None:
            return existing

    address = address or os.environ.get("DASK_SCHEDULER_ADDRESS")
    if address:
        client = Client(address=address, timeout=timeout)
        LOGGER.info("Connected to dask scheduler at %s", address)
        return client

    configure_memory_policy()
    workers = n_workers if n_workers is not None else default_worker_count()
    threads = (
        threads_per_worker
        if threads_per_worker is not None
        else int(os.environ.get("DASK_THREADS_PER_WORKER", "1"))
    )
    limit = memory_limit or worker_memory_limit(workers)
    enabled, bind = resolve_dashboard_address(enable=enable_dashboard, address=dashboard_address)
    LOGGER.info(
        "Starting LocalCluster: workers=%d threads=%d memory=%s dashboard=%s",
        workers,
        threads,
        limit,
        bind if enabled else "off",
    )
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads,
        processes=True,
        memory_limit=limit,
        dashboard_address=bind,
        **cluster_kwargs,
    )
    client = Client(cluster)
    if enabled and show_dashboard_link:
        show_dashboard(client)
    return client


def wait_for_workers(client: Client, min_workers: int = 1, timeout: int = 60) -> list[str]:
    """Block until at least ``min_workers`` workers are registered."""
    start = time.time()
    while True:
        workers = client.scheduler_info().get("workers", {})
        if len(workers) >= min_workers:
            return list(workers)
        if time.time() - start > timeout:
            msg = f"Only {len(workers)} of {min_workers} workers started within {timeout}s"
            raise TimeoutError(msg)
        time.sleep(1)


@contextmanager
def maybe_client(
    enabled: bool = True, *, close: bool = False, **kwargs: Any
) -> Iterator[Client | None]:
    """Yield a client (starting one if needed) or ``None`` for synchronous execution.

    ``close=True`` shuts the cluster down on exit; the default leaves a notebook's cluster
    running so later cells can reuse it and the dashboard stays live.
    """
    if not enabled:
        yield None
        return
    client = get_dask_client(**kwargs)
    try:
        yield client
    finally:
        if close:
            close_client(client)


def close_client(client: Client | None) -> None:
    """Close a client and its LocalCluster (if it owns one)."""
    if client is None:
        return
    cluster = getattr(client, "cluster", None)
    client.close()
    if cluster is not None:
        cluster.close()


# --------------------------------------------------------------------------------------
# bounded batches of delayed tasks
# --------------------------------------------------------------------------------------


def compute_in_batches(
    tasks: Sequence[Any],
    *,
    batch_size: int | None = None,
    client: Client | None = None,
    scheduler: str = "threads",
) -> list[Any]:
    """Compute ``dask.delayed`` tasks in bounded batches so 1000+ tasks never flood the scheduler.

    With a ``client`` each batch goes through ``client.compute`` / ``client.gather``; otherwise the
    local ``scheduler`` (``threads`` or ``processes``) is used. ``gc.collect()`` runs between
    batches - this pattern keeps WSL RAM flat on long file-per-task pipelines.
    """
    if not tasks:
        return []
    size = batch_size or int(os.environ.get("DASK_BATCH_SIZE", str(default_worker_count())))
    results: list[Any] = []
    for start in range(0, len(tasks), size):
        batch = list(tasks[start : start + size])
        if client is not None:
            results.extend(client.gather(client.compute(batch)))
        else:
            kwargs: dict[str, Any] = {"scheduler": scheduler}
            if scheduler in {"threads", "processes"}:
                kwargs["num_workers"] = size
            results.extend(dask.compute(*batch, **kwargs))
        gc.collect()
    return results
