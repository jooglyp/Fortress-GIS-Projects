# How the package fits together

The library is layered. Lower layers know nothing about the domains; the domain modules are
short scripts over the lower layers with names from their field.

```
domains/  hydrology  oceanography  airspace  maritime          workflows and export_artifacts
          ------------------------------------------------
viz/      KeplerMapBuilder                QgisBundle           maps and bundles
stats/    hotspots  hypothesis  regression  classify           ml/  pipeline  tuning  evaluate
features/ temporal  seasonal  encoding  proximity
grids/    hexgrid  aggregate                                    raster/  terrain  sampling  zonal
io/       raster (RasterInfo)  vector  tabular                  compute/  backend  cluster
```

## Rasters

Every raster function takes and returns a `RasterInfo`: a frozen dataclass with the array,
affine transform, CRS and nodata value. `masked()` gives the array with nodata as NaN,
`cell_centers()` the x and y of each cell, `index(x, y)` the row and column of points.
`with_data(array)` copies the georeferencing onto a new array, which is how
`fill_depressions`, `d8_flow_direction` and `flow_accumulation` hand results down the chain
without touching rasterio again. `write_raster` persists any of them as GeoTIFF.

The terrain routines are numpy over the whole grid: priority flood for depressions
(`heapq` over the boundary), D8 as an argmax over an eight-neighbour stack, accumulation by
topological order over the receiver graph, catchment by a stack walk upstream. They are sized
for DEMs up to a few tens of millions of cells in memory. `raster_to_points(as_dask=True)`
chunks by rows for grids that are not.

## Points, hexes and dask

Points are GeoDataFrames in WGS84 unless a projected CRS is passed. Hex grids are built in a
metric CRS (`crs.suggest_metric_crs` picks the UTM zone of a layer's centroid) and returned in the
layer's CRS. `hex_radius_from_travel_time(speed, minutes)` sizes a hex so a mover crosses it in
a set time, which is how the airspace grid is defined.

`compute.backend` has the two helpers the rest of the code uses to be indifferent to pandas or
dask input: `is_dask(obj)` and `to_pandas(obj)`. Functions with a dask path have a sibling with
a `_dask` suffix (`points_to_cells_dask`, `count_proximity_events_dask`) or a `use_dask`
argument that picks one. The dask versions use `dask_geopandas.sjoin` for the spatial join,
groupby aggregation with a dict spec, and `dask.compute(*parts)` to evaluate several results in
one graph.

`compute.cluster.get_dask_client()` starts or reuses a `LocalCluster`. It reads the
`DASK_*` variables listed in the README, sets a memory policy (spill to disk at 70 percent of the
worker limit, pause at 80, terminate at 95, disk based shuffles), and prints dashboard links in both the `127.0.0.1` and `localhost`
forms because WSL and the Windows browser see different addresses. Repeated calls return
`Client.current()` instead of a second cluster. `compute_in_batches` evaluates a list of
delayed objects a few at a time so the scheduler is not handed ten thousand tasks at once.

## Statistics

`stats.spatial_autocorr` wraps `esda.G_Local` (Gi*) and `esda.Moran_Local` over `libpysal`
weights. `build_weights` offers distance band, k nearest and queen contiguity; the distance
band threshold defaults to the smallest distance that leaves no island. Results come back as
`HotspotResult` with a `layer` holding `<attr>_gi_z`, `<attr>_gi_p_sim` and `<attr>_gi_class`
(or the `_lisa` equivalents), so any number of attributes can be merged on the cell id.

`stats.hypothesis` has the Poisson excess and rate tests used by airspace, chi-square
independence with standardized residuals, and `benjamini_hochberg`. `stats.regression` wraps
statsmodels: OLS with HC3 errors and elastic net on z-scored features, both returning
`RegressionResult` with `coefficients()`, `metrics()` and the statsmodels `result`.

## Machine learning

`ml.pipeline.REGRESSORS` is a dict of estimator factories. `build_pipeline(name, numeric,
categorical)` puts imputation, scaling and one-hot encoding in front of the estimator so
cross-validation never sees preprocessing fit on the test fold. `compare_models` returns a
leaderboard with CV mean and standard deviation per metric; `tune_model` runs
`RandomizedSearchCV` (or Optuna with the `tuning` extra) over `DEFAULT_SEARCH_SPACES[name]`
in parallel across cores. Adding a model is one dict entry plus, optionally, a search space.

## Exports

`viz.kepler.KeplerMapBuilder` collects layers and builds the Kepler.gl config JSON directly:
one geojson layer per GeoDataFrame, colour by a numeric field (quantile scale, switched to
quantize when more than half the values are one number so mostly-zero counts do not all land
in one bin) or by a categorical field with a fixed colour map, optional time field for the
time filter. `widget()` returns the Jupyter widget; `save_html()` writes a standalone file.

`viz.qgis.QgisBundle` writes vectors into one GeoPackage (layer per `add_vector`), rasters as
GeoTIFF, a QML per styled layer (graduated, categorized, or raster pseudocolour), a
`manifest.json` describing all of it, and `load_in_qgis.py`, which reads the manifest and adds
every layer with its style when executed in the QGIS Python console.

## Domain modules

Each domain module has the same shape: a result dataclass, loaders, a handful of step functions,
`run_<domain>_pipeline` chaining them, and `export_artifacts(result, out_dir)` producing the
QGIS bundle and Kepler HTML. The CLI in `cli.py` is one command per module with defaults under
`data/<domain>/` and `exports/<domain>/`. A new domain is a new module of that shape plus a CLI
command.
