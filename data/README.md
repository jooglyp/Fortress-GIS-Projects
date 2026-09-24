# Data

One folder per domain. Everything here except the README files is gitignored; the notebooks
and the CLI read from these folders and write to `exports/<domain>/`.

When a folder is empty the matching notebook fills it with synthetic input from
`datasets/synthetic.py`, so the demos run from a fresh clone. The airspace folder is the
exception: it is meant to hold a real position table, and the notebook only writes a synthetic
one if nothing is there.

| folder | expected contents | produced by |
| --- | --- | --- |
| `hydrology/` | one or more GeoTIFF DEM tiles in a projected CRS | `synthetic_dem_tiles` |
| `oceanography/` | wide grid CSVs, one per acquisition, `AYYYYDDD_*.csv` | `synthetic_chlorophyll_grids` |
| `airspace/` | `aircraft_positions.parquet` or CSV of position reports | real data, or `synthetic_aircraft_positions` |
| `maritime/` | `vessel_telemetry.parquet` or CSV of per-minute telemetry | `synthetic_vessel_telemetry` |

Each folder has its own README with the column layout the domain module expects and how to
point it at differently named columns.
