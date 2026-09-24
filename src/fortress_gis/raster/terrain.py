"""Terrain hydrology on in-memory DEMs (pure numpy/scipy).

The original watershed workflow ran TauDEM through the QGIS 2 processing framework
(``pitremove``, ``d8flowdirections``, ``dinfinitycontributingarea``, polygonize). TauDEM needs a
desktop GIS, MPI and a Windows installer. The same steps are implemented here in numpy:
:func:`fill_depressions` is priority-flood filling (Barnes et al. 2014) with an epsilon gradient
so flats drain; :func:`d8_flow_direction` writes an ESRI-encoded pointer grid;
:func:`flow_accumulation` counts upstream cells by topological sweeps; :func:`snap_pour_point`
moves an outlet onto the nearest high-accumulation cell; :func:`catchment` floods the reverse
graph from the outlet; :func:`stream_network` thresholds accumulation.

Inputs and outputs are :class:`~fortress_gis.io.raster.RasterInfo`, so results can be written as
GeoTIFF for QGIS or polygonised for Kepler.gl.
"""

from __future__ import annotations

import heapq
from typing import Final

import numpy as np

from fortress_gis.io.raster import RasterInfo
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

# ESRI / ArcGIS D8 encoding: value -> (d_row, d_col)
D8_OFFSETS: Final[dict[int, tuple[int, int]]] = {
    1: (0, 1),  # E
    2: (1, 1),  # SE
    4: (1, 0),  # S
    8: (1, -1),  # SW
    16: (0, -1),  # W
    32: (-1, -1),  # NW
    64: (-1, 0),  # N
    128: (-1, 1),  # NE
}
_CODES = np.array(list(D8_OFFSETS.keys()), dtype="int32")
_DROW = np.array([v[0] for v in D8_OFFSETS.values()], dtype="int64")
_DCOL = np.array([v[1] for v in D8_OFFSETS.values()], dtype="int64")
_DIST = np.sqrt(_DROW.astype(float) ** 2 + _DCOL.astype(float) ** 2)


def fill_depressions(dem: RasterInfo, *, epsilon: float = 1e-4) -> RasterInfo:
    """Fill closed depressions so every valid cell has a monotone path to the edge.

    Implements the Priority-Flood algorithm: seed a min-heap with the raster border (and cells
    bordering nodata), then repeatedly pop the lowest cell and raise its unvisited neighbours to
    at least its own elevation (+ ``epsilon`` so the filled surface still has a gradient, which
    is what lets :func:`d8_flow_direction` route across the former lake floor).

    O(n log n); a 1000 x 1000 DEM takes a few seconds in pure Python.
    """
    z = dem.masked()
    rows, cols = z.shape
    valid = np.isfinite(z)
    filled = z.copy()
    closed = ~valid  # nodata cells are never processed
    heap: list[tuple[float, int, int]] = []

    # Seed: every valid cell on the border or adjacent to nodata.
    padded_valid = np.pad(valid, 1, constant_values=False)
    touches_edge = np.zeros_like(valid)
    for dr, dc in zip(_DROW, _DCOL, strict=True):
        shifted = padded_valid[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        touches_edge |= ~shifted
    seeds = np.argwhere(valid & touches_edge)
    for r, c in seeds:
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))
        closed[r, c] = True

    while heap:
        zc, r, c = heapq.heappop(heap)
        for dr, dc in zip(_DROW, _DCOL, strict=True):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or closed[nr, nc]:
                continue
            closed[nr, nc] = True
            zn = filled[nr, nc]
            if zn <= zc:
                zn = zc + epsilon
                filled[nr, nc] = zn
            heapq.heappush(heap, (float(zn), nr, nc))

    n_changed = int(np.sum(valid & (filled > z + 1e-12)))
    LOGGER.info("fill_depressions: raised %d of %d valid cells", n_changed, int(valid.sum()))
    return dem.with_data(filled)


def _neighbour_stack(z: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """(8, rows, cols) array of neighbour elevations in D8 order."""
    rows, cols = z.shape
    padded = np.pad(z, 1, constant_values=fill_value)
    out = np.empty((8, rows, cols), dtype="float64")
    for k, (dr, dc) in enumerate(zip(_DROW, _DCOL, strict=True)):
        out[k] = padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
    return out


def d8_flow_direction(dem: RasterInfo) -> RasterInfo:
    """Steepest-descent flow direction (ESRI encoding; 0 = pit/no descent, -1 = nodata).

    Run this on a filled DEM so that the only pits left are outlets on the raster edge.
    """
    z = dem.masked()
    valid = np.isfinite(z)
    neigh = _neighbour_stack(z)
    drop = (z[None, :, :] - neigh) / _DIST[:, None, None]
    drop = np.where(np.isfinite(drop), drop, -np.inf)
    best = np.argmax(drop, axis=0)
    best_drop = np.take_along_axis(drop, best[None, ...], axis=0)[0]
    direction = np.where(best_drop > 0, _CODES[best], 0).astype("int32")
    direction[~valid] = -1
    return dem.with_data(direction, nodata=-1)


def _receivers(direction: np.ndarray) -> np.ndarray:
    """Flat index of the downstream cell for each cell, or -1 for pits/nodata/off-grid."""
    rows, cols = direction.shape
    code_to_k = {int(code): k for k, code in enumerate(_CODES)}
    k_idx = np.full(direction.shape, -1, dtype="int64")
    for code, k in code_to_k.items():
        k_idx[direction == code] = k
    has = k_idx >= 0
    rr, cc = np.indices(direction.shape)
    nr = rr + np.where(has, _DROW[np.clip(k_idx, 0, 7)], 0)
    nc = cc + np.where(has, _DCOL[np.clip(k_idx, 0, 7)], 0)
    inside = (nr >= 0) & (nr < rows) & (nc >= 0) & (nc < cols)
    recv = np.where(has & inside, nr * cols + nc, -1)
    # Receivers that are nodata cells terminate flow.
    flat_dir = direction.ravel()
    recv_flat = recv.ravel()
    ok = recv_flat >= 0
    ok[ok] &= flat_dir[recv_flat[ok]] != -1
    recv_flat[~ok] = -1
    return recv_flat


def flow_accumulation(direction: RasterInfo, *, weights: np.ndarray | None = None) -> RasterInfo:
    """Number of cells (or summed ``weights``) draining through each cell, itself included.

    Kahn-style topological propagation: at each sweep every current source cell pushes its
    accumulated value to its receiver in one ``bincount``. The number of sweeps is the length of
    the longest flow path, not the number of cells.
    """
    d = direction.data
    valid = d != -1
    n = d.size
    recv = _receivers(d)
    acc = (weights.astype("float64").ravel() if weights is not None else np.ones(n)) * valid.ravel()
    indeg = np.bincount(recv[recv >= 0], minlength=n)
    frontier = np.flatnonzero((indeg == 0) & valid.ravel())
    iterations = 0
    while frontier.size:
        iterations += 1
        targets = recv[frontier]
        has_target = targets >= 0
        contrib = np.bincount(targets[has_target], weights=acc[frontier][has_target], minlength=n)
        acc += contrib
        dec = np.bincount(targets[has_target], minlength=n)
        indeg -= dec
        touched = np.unique(targets[has_target])
        frontier = touched[indeg[touched] == 0]
    LOGGER.info("flow_accumulation: converged in %d topological sweeps", iterations)
    out = acc.reshape(d.shape)
    out[~valid] = np.nan
    return direction.with_data(out)


def snap_pour_point(
    accumulation: RasterInfo,
    x: float,
    y: float,
    *,
    search_cells: int = 3,
) -> tuple[int, int]:
    """Move an outlet coordinate onto the highest-accumulation cell within ``search_cells``.

    Returns the (row, col) of the snapped cell, as TauDEM's ``MoveOutletsToStreams`` does.
    """
    r, c = accumulation.index(np.array([x]), np.array([y]))
    r0, c0 = int(r[0]), int(c[0])
    rows, cols = accumulation.shape
    if not (0 <= r0 < rows and 0 <= c0 < cols):
        msg = f"Pour point ({x}, {y}) falls outside the raster"
        raise ValueError(msg)
    r_lo, r_hi = max(0, r0 - search_cells), min(rows, r0 + search_cells + 1)
    c_lo, c_hi = max(0, c0 - search_cells), min(cols, c0 + search_cells + 1)
    window = accumulation.masked()[r_lo:r_hi, c_lo:c_hi]
    if not np.isfinite(window).any():
        return r0, c0
    k = np.nanargmax(window)
    wr, wc = np.unravel_index(k, window.shape)
    return int(r_lo + wr), int(c_lo + wc)


def catchment(direction: RasterInfo, outlet: tuple[int, int]) -> RasterInfo:
    """Boolean mask of every cell that drains through ``outlet`` (row, col), outlet included.

    Vectorised reverse-graph flood: repeatedly add all cells whose receiver is already inside
    the catchment until no new cells appear.
    """
    d = direction.data
    rows, cols = d.shape
    recv = _receivers(d)
    inside = np.zeros(d.size, dtype=bool)
    inside[outlet[0] * cols + outlet[1]] = True
    has_recv = recv >= 0
    while True:
        candidate = has_recv & ~inside
        candidate[candidate] &= inside[recv[candidate]]
        if not candidate.any():
            break
        inside |= candidate
    mask = inside.reshape(rows, cols)
    LOGGER.info("catchment: %d cells (%.1f%% of grid)", int(mask.sum()), 100.0 * mask.mean())
    return direction.with_data(mask.astype("uint8"), nodata=None)


def stream_network(accumulation: RasterInfo, *, threshold: float) -> RasterInfo:
    """Cells whose accumulation meets ``threshold`` -> stream mask (uint8)."""
    acc = accumulation.masked()
    streams = np.where(np.isfinite(acc) & (acc >= threshold), 1, 0).astype("uint8")
    return accumulation.with_data(streams, nodata=None)


def downstream_path(
    direction: RasterInfo, start: tuple[int, int], *, max_steps: int | None = None
) -> list[tuple[int, int]]:
    """Follow D8 pointers from ``start`` to the outlet; returns the visited (row, col) list."""
    d = direction.data
    rows, cols = d.shape
    recv = _receivers(d)
    path = [start]
    idx = start[0] * cols + start[1]
    steps = 0
    while recv[idx] >= 0:
        idx = recv[idx]
        path.append((int(idx // cols), int(idx % cols)))
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
        if steps > rows * cols:  # pragma: no cover - defensive against cycles
            break
    return path
