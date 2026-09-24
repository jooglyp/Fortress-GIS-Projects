import itertools

import numpy as np
import pytest

from fortress_gis.io.raster import merge_rasters, raster_from_array, raster_to_points
from fortress_gis.raster.terrain import (
    catchment,
    d8_flow_direction,
    downstream_path,
    fill_depressions,
    flow_accumulation,
    snap_pour_point,
    stream_network,
)
from fortress_gis.raster.zonal import mask_to_polygons, zonal_statistics


def test_fill_depressions_removes_pits_and_never_lowers():
    z = np.full((9, 9), 10.0)
    z[4, 4] = 1.0  # closed pit in the middle
    dem = raster_from_array(z, bounds=(0, 0, 9, 9), crs="EPSG:32615")
    filled = fill_depressions(dem, epsilon=1e-3)
    assert np.all(filled.data >= z - 1e-9)
    assert filled.data[4, 4] > 9.9
    fdir = d8_flow_direction(filled)
    pits = fdir.data == 0
    # after filling, the only pits are on the raster edge
    assert not pits[1:-1, 1:-1].any()


def test_flow_accumulation_on_tilted_plane():
    rows, cols = 5, 6
    z = np.tile(np.arange(cols, dtype=float)[::-1], (rows, 1))  # drains east
    dem = raster_from_array(z, bounds=(0, 0, cols, rows), crs="EPSG:32615")
    acc = flow_accumulation(d8_flow_direction(dem))
    # every cell in a row drains into the next one east; last column collects the whole row
    assert np.array_equal(acc.data[:, -1], np.full(rows, cols))
    assert acc.data.sum() == rows * cols * (cols + 1) / 2


def test_catchment_covers_dominant_outlet(dem):
    filled = fill_depressions(dem)
    fdir = d8_flow_direction(filled)
    acc = flow_accumulation(fdir)
    r, c = np.unravel_index(np.nanargmax(acc.masked()), acc.shape)
    xs, ys = acc.cell_centers()
    outlet = snap_pour_point(acc, xs[r, c] + 0.5, ys[r, c] - 0.5, search_cells=2)
    assert outlet == (r, c)
    mask = catchment(fdir, outlet)
    assert mask.data.sum() == acc.data[r, c]
    poly = mask_to_polygons(mask)
    assert len(poly) == 1
    assert poly.crs.to_epsg() == 32615
    cell_area = np.prod(dem.resolution)
    assert poly.geometry.area.iloc[0] == pytest.approx(mask.data.sum() * cell_area, rel=1e-6)


def test_downstream_path_and_streams(dem):
    fdir = d8_flow_direction(fill_depressions(dem))
    acc = flow_accumulation(fdir)
    streams = stream_network(acc, threshold=50)
    assert 0 < streams.data.sum() < acc.data.size
    start = tuple(np.argwhere(streams.data == 1)[0])
    path = downstream_path(fdir, (int(start[0]), int(start[1])))
    assert path[0] == tuple(int(v) for v in start)
    assert len(path) >= 1
    # accumulation is non-decreasing downstream
    values = [acc.data[p] for p in path]
    assert all(b >= a for a, b in itertools.pairwise(values))


def test_merge_tiles_reproduces_dem(dem, dem_tiles):
    merged = merge_rasters(dem_tiles)
    assert merged.shape == dem.shape
    np.testing.assert_allclose(merged.masked(), dem.masked(), equal_nan=True)


def test_raster_to_points_eager_and_dask_agree(dem):
    eager = raster_to_points(dem, value_name="z")
    lazy = raster_to_points(dem, value_name="z", as_dask=True, chunk_rows=16).compute()
    assert len(eager) == dem.data.size == len(lazy)
    assert eager["z"].sum() == pytest.approx(lazy["z"].sum())


def test_zonal_statistics_matches_numpy(dem):
    fdir = d8_flow_direction(fill_depressions(dem))
    acc = flow_accumulation(fdir)
    r, c = np.unravel_index(np.nanargmax(acc.masked()), acc.shape)
    poly = mask_to_polygons(catchment(fdir, (int(r), int(c))))
    stats = zonal_statistics(poly, dem, stats=("mean", "count"), prefix="z")
    inside = catchment(fdir, (int(r), int(c))).data.astype(bool)
    assert stats["z_count"].iloc[0] == inside.sum()
    assert stats["z_mean"].iloc[0] == pytest.approx(dem.data[inside].mean())
