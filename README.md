# fortress-gis

Raster and vector geoprocessing, spatial statistics and machine learning on geodata, with
exports for Kepler.gl and QGIS. The package started as four separate analyses (a watershed from
DEM tiles, seasonal hotspots in satellite composites, proximity events in aircraft position
reports, fuel demand from vessel telemetry) and was rebuilt as one library so the shared parts
are written once: reading rasters and tables, hex grids, dask-backed joins, hotspot statistics,
regression and model comparison, map export.

Each of the four analyses survives as a workflow module under `fortress_gis.domains`, a CLI
command, and a notebook under `demos/`.

## What is in the package

```
src/fortress_gis/
  io/         read/write rasters (RasterInfo), points from tables, gridded CSV stacks, GeoParquet
  raster/     merge, clip, fill depressions, D8 flow direction, accumulation, catchment,
              stream network, line sampling, zonal statistics
  grids/      hex grids sized by area or travel time; point to cell aggregation (pandas or dask)
  features/   hour and day/night columns, season calendars and year over year change,
              one-hot encoding, pairwise proximity events (blocked, windowed, dask)
  stats/      Getis-Ord Gi* and local Moran's I, Poisson and chi-square tests,
              Benjamini-Hochberg, OLS with HC3 errors, elastic net, natural breaks classes
  ml/         sklearn pipelines and a model registry, cross-validated leaderboard,
              random or Optuna search, permutation importance, holdout metrics
  viz/        KeplerMapBuilder (widget, HTML, config JSON); QgisBundle (GeoPackage, GeoTIFF,
              QML styles, manifest, load_in_qgis.py)
  compute/    dask client with the Bokeh dashboard, WSL aware URLs, batched compute
  domains/    hydrology, oceanography, airspace, maritime workflows built from the above
  cli.py      `fortress-gis <domain>` commands
datasets/     synthetic data generators used by the notebooks and tests (not part of the wheel)
data/         input data by domain (gitignored except READMEs)
demos/        four notebooks (no saved output)
exports/      notebook and CLI output (gitignored)
tests/        pytest suite
```

## Setup

Python is managed with [uv](https://docs.astral.sh/uv/). `.python-version` pins 3.13.2 and
`uv.lock` pins every dependency.

```bash
# from the repo root
uv python install 3.13.2
uv sync --all-extras          # creates .venv, installs the package in editable mode
uv run pytest                 # 32 tests, about a minute
uv run fortress-gis --help
```

To keep the environment outside the repository, set `UV_PROJECT_ENVIRONMENT` before `uv sync`.
On WSL a venv on the Linux filesystem imports the scientific stack in a few seconds; one under
`/mnt/c` takes closer to a minute per process.

```bash
export UV_PROJECT_ENVIRONMENT=/mnt/c/DevState/venvs/fortress-gis-313   # or $HOME/.venvs/fortress-gis-313
export UV_CACHE_DIR=/mnt/c/DevState/caches/uv-cache                    # optional
uv sync --all-extras
```

### Kepler.gl

`keplergl` 0.3.7 is pinned in the `viz` extra together with `setuptools<81` (the package
still imports `pkg_resources`), and `no-build-isolation-package = ["keplergl"]` in
`pyproject.toml` builds it against the environment's pyarrow. For the widget to render inside
Jupyter run once:

```bash
uv run jupyter nbextension enable --py --sys-prefix keplergl
```

`KeplerMapBuilder.save_html` does not need the extension; the HTML files under `exports/` open
in any browser and use MapLibre with OpenStreetMap tiles, so no token is required.

### Dask dashboard

`fortress_gis.compute.cluster.get_dask_client()` starts a `LocalCluster` with the Bokeh
dashboard on `127.0.0.1:8787` and prints the link. On WSL it prints both the `127.0.0.1` form
and a `localhost` form that opens from the Windows browser. Environment variables:

| variable | effect |
| --- | --- |
| `DASK_N_WORKERS` | worker processes (default: half the cores, at most 4) |
| `DASK_THREADS_PER_WORKER` | threads per worker (default 1) |
| `DASK_MEMORY_LIMIT` | per-worker limit such as `4GB`; default is (system memory minus `DASK_RESERVE_GB`, 4) split across workers |
| `DASK_DASHBOARD_ADDRESS` | bind address, default `127.0.0.1:8787` |
| `DASK_ENABLE_DASHBOARD` | `0` disables the dashboard |
| `DASK_OPEN_DASHBOARD` | `1` opens the dashboard page as the client starts |
| `DASK_SCHEDULER_ADDRESS` | connect to an existing scheduler instead of starting one |

Calling `get_dask_client()` again returns the running client rather than a second cluster.
Scripts that start a cluster must be run from a file, not piped to `python -`; worker processes
are spawned and need to re-import `__main__`.

## Running the workflows

Each domain has a notebook, a CLI command and a `run_*_pipeline` function. The notebooks
generate synthetic input under `data/<domain>/` when the folder is empty, except airspace,
which uses `data/airspace/aircraft_positions.parquet` if present.

```bash
uv run fortress-gis hydrology --pour-x 500050 --pour-y 4412050
uv run fortress-gis oceanography --method gstar --permutations 499
uv run fortress-gis airspace --dask --hours 7 19 --max-altitude-ft 500
uv run fortress-gis maritime --target fuel_demand_kg_h --n-iter 12
```

Outputs land in `exports/<domain>/`: a `qgis/` folder with a GeoPackage, GeoTIFFs where
relevant, one QML per layer, `manifest.json` and `load_in_qgis.py`; a Kepler HTML file and
its config JSON; CSV tables where the workflow produces them.

To load a bundle in QGIS, open the Python console and run
`exec(open("exports/<domain>/qgis/load_in_qgis.py").read())`.

From Python:

```python
from fortress_gis.domains import airspace as air
from fortress_gis.features.proximity import ProximityConfig

result = air.run_airspace_pipeline(
    "data/airspace/aircraft_positions.parquet",
    flt=air.AirspaceFilter(hours=(7, 19), altitude_ft=(0, 500)),
    config=ProximityConfig(),  # 1000 ft horizontal, 200 ft vertical, 5 s
    minutes_per_hex=5,
    use_dask=True,
)
result.summary()
air.export_artifacts(result, "exports/airspace")
```

## Notebooks

| notebook | data | methods |
| --- | --- | --- |
| `01_hydrology_watershed` | 4 synthetic DEM tiles, 100 m | mosaic, depression filling, D8, accumulation, catchment, flowline sampling, OLS of slope on length |
| `02_oceanography_seasonal_hotspots` | 24 synthetic chlorophyll composites | hex binning with dask, season means and variance, Gi* and LISA, Benjamini-Hochberg |
| `03_airspace_proximity_events` | 1.15 M real position reports (one month) | dask read and filter, windowed proximity events, travel-time hex grid, Poisson excess and rate tests, chi-square, time-filtered Kepler map |
| `04_maritime_fuel_demand` | 6,000 synthetic telemetry records | OLS with HC3, elastic net, six-model CV leaderboard, random search, time-ordered holdout, permutation importance, residual map |

Run times below are from a 12 core machine with three dask workers, about two and a half
minutes for all four. The files in git have no cell output.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv run pre-commit install
```

`ruff` excludes `demos/`, `_archive/` and `data/`. Tests run against `src/` and the repo root
(`pythonpath` in `pyproject.toml`) so `datasets` imports without installation. Tests use
small synthetic inputs and finish in about a minute; dask paths run on the threaded scheduler.

## Data and archive

`data/` holds inputs by domain; everything except the READMEs is gitignored. `_archive/` and
`_notes/` are gitignored working folders. Column names used by the domain modules are documented
in `data/README.md`.

## License

MIT, see `LICENSE`.
