# Demos

Four notebooks, one per domain, each running a workflow from input to exported artifacts and
including the statistics or model fitting the workflow was built around. They were executed
with `nbclient` and saved with outputs, so they read as reports without being run.

| notebook | runtime | needs |
| --- | --- | --- |
| `01_hydrology_watershed.ipynb` | 15 s | nothing; writes synthetic DEM tiles |
| `02_oceanography_seasonal_hotspots.ipynb` | 25 s | dask cluster (started in the notebook) |
| `03_airspace_proximity_events.ipynb` | 40 s | `data/airspace/aircraft_positions.parquet`; falls back to synthetic; dask cluster |
| `04_maritime_fuel_demand.ipynb` | 70 s | nothing; writes synthetic telemetry |

Runtimes are from a 12 core machine with `DASK_N_WORKERS=3`.

## Running them

```bash
uv run jupyter lab demos/
```

or without opening a browser:

```bash
DASK_N_WORKERS=3 uv run jupyter execute demos/0*.ipynb --inplace
```

The first code cell of each notebook finds the repo root by walking up to `pyproject.toml` and
adds it and `src/` to `sys.path`, so they run from a clone without `uv sync`'s editable install
as long as the dependencies are present.

## Kepler widget

Each notebook has a cell that displays a `KeplerGl` widget. The widget needs the notebook
extension once per environment:

```bash
uv run jupyter nbextension enable --py --sys-prefix keplergl
```

Without it the cell shows the widget's text representation and everything else still runs. The
last cell of each notebook writes a standalone HTML map to `exports/<domain>/` that opens in
any browser.

## Dask dashboard

Notebooks 02 and 03 call `get_dask_client()`, which starts a local cluster and prints the Bokeh
dashboard link. On WSL the `http://localhost:8787/status` form opens from the Windows browser.
The task stream and worker memory pages are the useful ones while the hex join or the proximity
count runs.
