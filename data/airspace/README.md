# Airspace inputs

A table of aircraft position reports, parquet or CSV, one row per report. The bundled file
`aircraft_positions.parquet` holds 1,151,230 reports from March 2023 over one metropolitan
area, about 21 MB. It is gitignored; anyone without it gets a synthetic table with the same
schema when the notebook runs.

Default column names (`fortress_gis.domains.airspace.DEFAULT_SCHEMA`):

| column | type | meaning |
| --- | --- | --- |
| `DATETIME_UTC` | timestamp or ISO string | report time |
| `LATITUDE`, `LONGITUDE` | float, degrees | WGS84 position |
| `ALTITUDE_AGL_FT` | float | altitude above ground, feet |
| `GROUND_SPEED_KNTS` | float | ground speed, knots |
| `FLIGHT_UID` | string | track identifier |

Other names go through a `PositionSchema`:

```python
from fortress_gis.domains.airspace import PositionSchema, run_airspace_pipeline

schema = PositionSchema(time="ts", lat="lat", lon="lon", altitude="alt_ft", speed="gs_kn", track="icao")
result = run_airspace_pipeline("positions.parquet", schema=schema, use_dask=True)
```

Two things about the real table matter for the statistics. Track identifiers are reused across
days, so exposure hours are summed from gaps between consecutive reports of a track (gaps over
ten minutes dropped) rather than from first to last timestamp. Reports arrive about every four
seconds per track, so the proximity search compares reports within five seconds of each other.
