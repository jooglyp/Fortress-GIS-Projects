"""Time-of-day, seasonal, encoded and spatio-temporal proximity features."""

from fortress_gis.features.encoding import (
    ColumnTypes,
    classify_columns,
    one_hot_encode,
    select_model_columns,
)
from fortress_gis.features.proximity import (
    ProximityConfig,
    count_proximity_events,
    count_proximity_events_dask,
    flag_unacceptable,
)
from fortress_gis.features.seasonal import (
    SeasonCalendar,
    interannual_variance,
    season_for_day_of_year,
    seasonal_means,
    seasonal_variance,
    year_over_year_change,
    year_season_columns,
    year_season_means,
)
from fortress_gis.features.temporal import (
    add_hour_and_day_night,
    filter_by_hour,
    time_windows,
)

__all__ = [
    "ColumnTypes",
    "ProximityConfig",
    "SeasonCalendar",
    "add_hour_and_day_night",
    "classify_columns",
    "count_proximity_events",
    "count_proximity_events_dask",
    "filter_by_hour",
    "flag_unacceptable",
    "interannual_variance",
    "one_hot_encode",
    "season_for_day_of_year",
    "seasonal_means",
    "seasonal_variance",
    "select_model_columns",
    "time_windows",
    "year_over_year_change",
    "year_season_columns",
    "year_season_means",
]
