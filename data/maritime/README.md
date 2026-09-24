# Maritime inputs

A table of vessel telemetry, parquet or CSV, one row per sampling tick, ordered in time. The
model treats every column except the target, the timestamp and the coordinates as a candidate
predictor, so the file should contain sensor columns only (no free text, no ids that vary per
row).

Columns the domain module looks for by name (all can be renamed through arguments):

| column | default name | used for |
| --- | --- | --- |
| target | `fuel_demand_kg_h` | the quantity modelled (`target=`) |
| timestamp | `timestamp`, integer seconds since epoch | ordering, the time-ordered holdout, `is_night` (`time_column=`, `time_unit=`) |
| coordinates | `longitude`, `latitude` | the voyage track and the residual map (`lon=`, `lat=`) |

Numeric columns are used as they are. A column with 20 or fewer distinct values is treated as
categorical and one-hot encoded with the first level dropped (`category_limit=`). Columns to
leave out of the design go in `exclude=`.

The synthetic file written by the notebook is `vessel_telemetry.parquet`: 6,000 one-minute
records along a transatlantic route with speed over ground and through water, heading, rudder
angle, fore and aft draft, wind speed and direction, true wind speed, wave height, period and
direction, water depth, sea temperature, and a fuel demand column built from those with noise.
