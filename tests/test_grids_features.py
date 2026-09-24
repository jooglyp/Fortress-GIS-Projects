import numpy as np
import pandas as pd
import pytest

from fortress_gis.features.encoding import classify_columns, one_hot_encode
from fortress_gis.features.proximity import (
    ProximityConfig,
    count_proximity_events,
    count_proximity_events_dask,
    exposure_hours,
)
from fortress_gis.features.seasonal import (
    SeasonCalendar,
    seasonal_means,
    year_over_year_change,
    year_season_means,
)
from fortress_gis.features.temporal import add_hour_and_day_night, filter_by_hour
from fortress_gis.grids.aggregate import points_to_cells, points_to_cells_dask
from fortress_gis.grids.hexgrid import (
    HexSpec,
    hex_grid,
    hex_grid_for_layer,
    hex_radius_from_area,
    hex_radius_from_travel_time,
)
from fortress_gis.io.vector import points_from_xy


def test_hex_geometry_is_regular_and_tiles_without_gaps():
    cells = hex_grid((0, 0, 1000, 1000), radius=100, crs="EPSG:32615")
    areas = cells.geometry.area
    assert areas.min() == pytest.approx(areas.max(), rel=1e-9)
    assert areas.iloc[0] == pytest.approx(HexSpec(100).area, rel=1e-9)
    union = cells.geometry.union_all()
    assert union.area == pytest.approx(areas.sum(), rel=1e-6)  # no overlaps
    assert cells["hex_id"].is_unique


def test_hex_sizing_helpers():
    r = hex_radius_from_area(100e6)  # 100 km2
    assert HexSpec(r).area == pytest.approx(100e6)
    # 60 knots for 5 minutes = 5 nautical miles = 9260 m; diameter == distance -> radius 4630 m
    assert hex_radius_from_travel_time(60, 5, speed_units="knots", divisor=2) == pytest.approx(
        4630, rel=1e-3
    )


def test_points_to_cells_counts_every_point_once():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "longitude": rng.uniform(-122.5, -122.0, 2000),
            "latitude": rng.uniform(37.3, 37.8, 2000),
            "v": rng.normal(size=2000),
        }
    )
    pts = points_from_xy(df)
    cells = hex_grid_for_layer(pts, radius=3000)
    agg = points_to_cells(pts, cells, value_columns=["v"], agg=["mean", "sum"])
    assert agg["n_points"].sum() == len(pts)
    assert agg["v_sum"].sum() == pytest.approx(df["v"].sum())
    lazy = points_to_cells_dask(pts, cells, value_columns=["v"], agg=["mean", "sum"], npartitions=4)
    merged = agg[["hex_id", "n_points", "v_sum"]].merge(
        lazy[["hex_id", "n_points", "v_sum"]], on="hex_id", suffixes=("", "_d")
    )
    assert (merged["n_points"] == merged["n_points_d"]).all()
    np.testing.assert_allclose(merged["v_sum"].fillna(0), merged["v_sum_d"].fillna(0))


def test_proximity_finds_planted_conflicts_and_ignores_same_track(aircraft):
    pts = points_from_xy(aircraft, lon="LONGITUDE", lat="LATITUDE")
    cfg = ProximityConfig(horizontal_m=304.8, vertical=200, time_tolerance_s=5)
    out, pairs = count_proximity_events(
        pts,
        time_column="DATETIME_UTC",
        altitude_column="ALTITUDE_AGL_FT",
        track_column="FLIGHT_UID",
        config=cfg,
        return_pairs=True,
    )
    assert (out["event_count"] > 0).any()
    tracks = pts["FLIGHT_UID"].to_numpy()
    assert (tracks[pairs["i"].to_numpy()] != tracks[pairs["j"].to_numpy()]).all()
    assert (pairs["distance_m"] <= cfg.horizontal_m).all()
    assert (pairs["dt_s"].abs() <= cfg.time_tolerance_s).all()
    lazy = count_proximity_events_dask(
        pts,
        time_column="DATETIME_UTC",
        altitude_column="ALTITUDE_AGL_FT",
        track_column="FLIGHT_UID",
        config=cfg,
        window="20min",
    )
    assert np.array_equal(out["event_count"].to_numpy(), lazy["event_count"].to_numpy())
    assert exposure_hours(aircraft, time_column="DATETIME_UTC", track_column="FLIGHT_UID") > 0


def test_exposure_hours_ignores_gaps_between_reused_track_ids():
    # one track id used for a 10 minute flight on day 1 and a 10 minute flight on day 2,
    # reported every 4 s: 20 minutes observed, not 24 hours
    t1 = pd.date_range("2023-03-01 08:00", periods=151, freq="4s", tz="UTC")
    t2 = pd.date_range("2023-03-02 08:00", periods=151, freq="4s", tz="UTC")
    df = pd.DataFrame({"t": t1.append(t2), "k": "A"})
    hours = exposure_hours(df, time_column="t", track_column="k")
    assert hours == pytest.approx(20 / 60, rel=1e-6)
    # the global span without a track column still counts the whole day
    assert exposure_hours(df, time_column="t") > 24


def test_temporal_features_and_hour_filter():
    ts = pd.date_range("2023-03-01 05:00", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"t": ts})
    out = add_hour_and_day_night(df, time_column="t")
    assert out["hour"].tolist() == [(5 + i) % 24 for i in range(24)]
    assert out.loc[out["hour"] == 22, "is_night"].all()
    assert not out.loc[out["hour"] == 12, "is_night"].any()
    day = filter_by_hour(df, time_column="t", start_hour=7, end_hour=19)
    assert set(day["hour"]) == set(range(7, 19))
    epoch = pd.DataFrame({"t": [int(t.timestamp()) for t in ts]})
    assert (
        add_hour_and_day_night(epoch, time_column="t", unit="s")["hour"].tolist()
        == out["hour"].tolist()
    )


def test_seasonal_pipeline_on_wide_frame():
    df = pd.DataFrame(
        {
            "hex_id": [1, 2],
            "chlor_2017015": [1.0, 2.0],  # winter
            "chlor_2017200": [3.0, 4.0],  # summer
            "chlor_2018015": [2.0, 2.0],
            "chlor_2018200": [6.0, 4.0],
        }
    )
    season = year_season_means(df, id_columns=["hex_id"])
    assert list(season.columns) == [
        "hex_id",
        "2017_summer",
        "2017_winter",
        "2018_summer",
        "2018_winter",
    ]
    yoy = year_over_year_change(season, id_columns=["hex_id"])
    assert yoy.loc[0, "2018_summer_yoy"] == pytest.approx(1.0)  # 3 -> 6
    assert yoy.loc[1, "2018_summer_yoy"] == pytest.approx(0.0)
    means = seasonal_means(season, id_columns=["hex_id"])
    assert means.loc[0, "winter_mean"] == pytest.approx(1.5)
    assert SeasonCalendar().season_for(150) == "summer"
    assert SeasonCalendar.four_seasons().season_for(15) == "winter"


def test_classify_and_one_hot(telemetry):
    df = add_hour_and_day_night(telemetry, time_column="timestamp", unit="s")
    df["port"] = np.select([df["longitude"] > -20, df["longitude"] > -40], ["east", "mid"], "west")
    types = classify_columns(df, target="fuel_demand_kg_h")
    assert "port" in types.categorical
    assert "is_night" in types.boolean
    assert "fuel_demand_kg_h" not in types.predictors()
    enc, mapping = one_hot_encode(df, ["port"])
    assert mapping["port"] == ["port_mid", "port_west"]  # first category dropped
    assert enc["port_west"].sum() == (df["port"] == "west").sum()
