"""Synthetic inputs for the demos and tests.

This module lives outside the ``fortress_gis`` package on purpose: the library never depends on
it. Each generator is deterministic for a given ``seed`` and writes or returns data with the
same columns, units and file layout as the real products the domain workflows read, so a
notebook can be pointed at real inputs by changing one path.

Import it from the repository root (``from datasets.synthetic import synthetic_dem``); the
notebooks and ``tests/conftest.py`` put the root on ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import from_bounds
from rasterio.windows import Window

from fortress_gis.io.raster import RasterInfo, write_raster

# --------------------------------------------------------------------------------------
# hydrology
# --------------------------------------------------------------------------------------


def synthetic_dem(
    rows: int = 240,
    cols: int = 300,
    *,
    bounds: tuple[float, float, float, float] = (500_000.0, 4_400_000.0, 530_000.0, 4_424_000.0),
    crs: str = "EPSG:32615",
    seed: int = 7,
    n_hills: int = 6,
    noise: float = 3.0,
) -> RasterInfo:
    """A basin-shaped DEM (metres) with hills, a main valley draining west, and small noise.

    Default extent is 30 x 24 km in UTM 15N at 100 m cells. The surface is built so that the
    lowest edge is the western boundary, which gives :func:`~fortress_gis.raster.terrain.catchment`
    a single dominant outlet - a good regression target for tests.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols].astype("float64")
    xn, yn = x / cols, y / rows
    # Regional slope down to the west + a parabolic valley along the centre line.
    base = 200.0 + 400.0 * xn + 150.0 * (yn - 0.5) ** 2 * 4.0
    for _ in range(n_hills):
        cx, cy = rng.uniform(0.15, 0.95), rng.uniform(0.1, 0.9)
        amp, sx, sy = rng.uniform(60, 220), rng.uniform(0.05, 0.15), rng.uniform(0.05, 0.15)
        base += amp * np.exp(-(((xn - cx) ** 2) / (2 * sx**2) + ((yn - cy) ** 2) / (2 * sy**2)))
    # Carve a meandering channel so flowlines have somewhere obvious to go.
    channel_y = 0.5 + 0.12 * np.sin(2 * np.pi * xn * 1.5)
    base -= 60.0 * np.exp(-((yn - channel_y) ** 2) / (2 * 0.02**2))
    base += rng.normal(0.0, noise, size=base.shape)
    transform = from_bounds(*bounds, width=cols, height=rows)
    return RasterInfo(data=base.astype("float32"), transform=transform, crs=crs, nodata=np.nan)


def synthetic_dem_tiles(
    dem: RasterInfo, out_dir: str | Path, *, n_rows: int = 2, n_cols: int = 2
) -> list[Path]:
    """Split a DEM into ``n_rows x n_cols`` GeoTIFF tiles (to demonstrate mosaicking)."""
    from rasterio.windows import transform as window_transform

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, cols = dem.shape
    paths: list[Path] = []
    r_edges = np.linspace(0, rows, n_rows + 1).astype(int)
    c_edges = np.linspace(0, cols, n_cols + 1).astype(int)
    for i in range(n_rows):
        for j in range(n_cols):
            win = Window(
                c_edges[j], r_edges[i], c_edges[j + 1] - c_edges[j], r_edges[i + 1] - r_edges[i]
            )
            tile = RasterInfo(
                data=dem.data[r_edges[i] : r_edges[i + 1], c_edges[j] : c_edges[j + 1]],
                transform=window_transform(win, dem.transform),
                crs=dem.crs,
                nodata=dem.nodata,
            )
            paths.append(write_raster(tile, out / f"dem_tile_r{i}_c{j}.tif"))
    return paths


# --------------------------------------------------------------------------------------
# oceanography
# --------------------------------------------------------------------------------------


def synthetic_chlorophyll_grids(
    out_dir: str | Path,
    *,
    years: Sequence[int] = (2016, 2017, 2018, 2019),
    days_of_year: Sequence[int] = (15, 60, 135, 200, 250, 320),
    lon: tuple[float, float, int] = (-40.0, -10.0, 60),
    lat: tuple[float, float, int] = (25.0, 55.0, 60),
    seed: int = 11,
    prefix: str = "A",
    suffix: str = "_chlor_a",
    cloud_fraction: float = 0.08,
) -> list[Path]:
    """Write wide grid CSVs (rows = longitude, headers = latitude) mimicking composite products.

    File names follow ``<prefix>YYYYDDD<suffix>.csv``. Values (mg m^-3) combine a latitudinal
    gradient (richer in the north), a winter bloom, a persistent offshore hotspot, interannual
    drift and a ``cloud_fraction`` of missing cells.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lons = np.linspace(lon[0], lon[1], lon[2])
    lats = np.linspace(lat[0], lat[1], lat[2])
    LON, LAT = np.meshgrid(lons, lats, indexing="ij")
    hotspot = 1.8 * np.exp(-(((LON + 22.0) ** 2) / 18.0 + ((LAT - 44.0) ** 2) / 12.0))
    paths: list[Path] = []
    for year in years:
        drift = 1.0 + 0.04 * (year - years[0]) + rng.normal(0, 0.03)
        for doy in days_of_year:
            season = 1.0 + 0.6 * np.cos(
                2 * np.pi * (doy - 30) / 365.0
            )  # winter bloom peaks ~DOY 30
            field = (0.15 + 0.03 * (LAT - lat[0])) * season * drift + hotspot * (0.7 + 0.3 * season)
            field *= np.exp(rng.normal(0, 0.12, size=field.shape))
            field[rng.random(field.shape) < cloud_fraction] = np.nan
            frame = pd.DataFrame(field, index=lons, columns=[f"{v:.4f}" for v in lats])
            frame.index.name = "longitude"
            path = out / f"{prefix}{year}{doy:03d}{suffix}.csv"
            frame.to_csv(path, float_format="%.4f")
            paths.append(path)
    return paths


# --------------------------------------------------------------------------------------
# maritime
# --------------------------------------------------------------------------------------


def synthetic_vessel_telemetry(
    n: int = 6000,
    *,
    seed: int = 3,
    start: str = "2024-02-01T00:00:00Z",
    step_seconds: int = 60,
    origin: tuple[float, float] = (-9.5, 38.6),
    destination: tuple[float, float] = (-74.0, 40.5),
) -> pd.DataFrame:
    """One transatlantic voyage of minute-level vessel telemetry with a fuel-demand target.

    Columns mirror a typical ship data logger: ``timestamp`` (epoch s), ``longitude``,
    ``latitude``, ``speed_over_ground_kn``, ``speed_through_water_kn``, ``heading_deg``,
    ``rudder_angle_deg``, ``draft_fore_m``, ``draft_aft_m``, wind/sea state, ``water_depth_m``,
    ``sea_temp_c`` and the target ``fuel_demand_kg_h``. Fuel demand follows a cubic law in speed
    through water with penalties for head wind, waves, rudder activity and draft, plus noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * step_seconds
    frac = t / t.max()
    lon = origin[0] + (destination[0] - origin[0]) * frac + 0.3 * np.sin(2 * np.pi * frac * 3)
    lat = origin[1] + (destination[1] - origin[1]) * frac + 0.4 * np.sin(2 * np.pi * frac * 2)
    hours = (t / 3600.0) % 24
    sog = 14.0 + 2.0 * np.sin(2 * np.pi * frac * 5) + rng.normal(0, 0.5, n)
    current = 0.8 * np.sin(2 * np.pi * frac * 1.5)
    stw = sog - current + rng.normal(0, 0.2, n)
    heading = np.degrees(np.arctan2(np.gradient(lat), np.gradient(lon))) % 360
    rudder = rng.normal(0, 2.5, n) + 6.0 * np.sin(2 * np.pi * frac * 40) * (rng.random(n) < 0.2)
    wind_speed = np.clip(
        8.0 + 6.0 * np.sin(2 * np.pi * frac * 2.3 + 1.0) + rng.normal(0, 2.0, n), 0, None
    )
    wind_dir = (heading + 180 + 60 * np.sin(2 * np.pi * frac * 1.7)) % 360
    rel_wind = np.cos(np.radians(wind_dir - heading))  # 1 = head wind
    wave_height = np.clip(0.4 + 0.12 * wind_speed + rng.normal(0, 0.3, n), 0, None)
    wave_period = 6.0 + 0.5 * wave_height + rng.normal(0, 0.4, n)
    depth = np.clip(
        3000.0 - 2800.0 * np.exp(-((frac - 0.5) ** 2) / 0.02) + rng.normal(0, 50, n), 20, None
    )
    draft_fore = 9.5 - 0.6 * frac + rng.normal(0, 0.02, n)
    draft_aft = 10.2 - 0.6 * frac + rng.normal(0, 0.02, n)
    sea_temp = 18.0 - 6.0 * frac + 1.5 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.3, n)
    night = ((hours >= 20) | (hours <= 6)).astype(float)
    fuel = (
        0.045 * np.clip(stw, 0, None) ** 3
        + 4.0 * wind_speed * np.clip(rel_wind, 0, None)
        + 9.0 * wave_height**1.5
        + 0.8 * np.abs(rudder)
        + 12.0 * (draft_fore + draft_aft) / 2
        + 15.0 * night
        + rng.normal(0, 12.0, n)
    )
    epoch_start = int(
        pd.Timestamp(start).tz_convert(None).timestamp()
        if pd.Timestamp(start).tzinfo
        else pd.Timestamp(start).timestamp()
    )
    return pd.DataFrame(
        {
            "timestamp": (epoch_start + t).astype("int64"),
            "longitude": lon,
            "latitude": lat,
            "fuel_demand_kg_h": fuel,
            "speed_over_ground_kn": sog,
            "speed_through_water_kn": stw,
            "heading_deg": heading,
            "rudder_angle_deg": rudder,
            "draft_fore_m": draft_fore,
            "draft_aft_m": draft_aft,
            "wind_speed_kn": wind_speed,
            "wind_dir_deg": wind_dir,
            "true_wind_speed_kn": wind_speed + 0.3 * sog * rel_wind,
            "wave_height_m": wave_height,
            "wave_period_s": wave_period,
            "wave_dir_deg": (wind_dir + rng.normal(0, 15, n)) % 360,
            "water_depth_m": depth,
            "sea_temp_c": sea_temp,
        }
    )


# --------------------------------------------------------------------------------------
# airspace
# --------------------------------------------------------------------------------------


def synthetic_aircraft_positions(
    n_tracks: int = 60,
    *,
    points_per_track: int = 120,
    bbox: tuple[float, float, float, float] = (-122.60, 37.33, -121.85, 37.89),
    start: str = "2023-03-01T06:00:00Z",
    seed: int = 5,
    conflict_pairs: int = 6,
) -> pd.DataFrame:
    """Straight-line low-altitude tracks over a metro bounding box with a few induced conflicts.

    Columns match the real product: ``DATETIME_UTC``, ``LATITUDE``, ``LONGITUDE``,
    ``ALTITUDE_AGL_FT``, ``GROUND_SPEED_KNTS``, ``HEADING_DEG``, ``FLIGHT_UID``. ``conflict_pairs``
    tracks are duplicated with a small offset and the same timing so proximity detection has
    known positives.
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = bbox
    frames = []
    t0 = pd.Timestamp(start)
    for k in range(n_tracks):
        x0, y0 = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        ang = rng.uniform(0, 2 * np.pi)
        speed_kn = rng.uniform(60, 140)
        dist_deg = speed_kn * 0.514444 * points_per_track / 111_000.0
        xs = x0 + np.cos(ang) * np.linspace(0, dist_deg, points_per_track)
        ys = y0 + np.sin(ang) * np.linspace(0, dist_deg, points_per_track)
        alt = np.clip(
            300
            + 250 * np.sin(np.linspace(0, np.pi, points_per_track))
            + rng.normal(0, 15, points_per_track),
            50,
            1000,
        )
        start_offset = pd.Timedelta(seconds=int(rng.integers(0, 12 * 3600)))
        times = t0 + start_offset + pd.to_timedelta(np.arange(points_per_track), unit="s")
        frames.append(
            pd.DataFrame(
                {
                    "DATETIME_UTC": times,
                    "LATITUDE": ys,
                    "LONGITUDE": xs,
                    "ALTITUDE_AGL_FT": alt,
                    "GROUND_SPEED_KNTS": speed_kn + rng.normal(0, 2, points_per_track),
                    "HEADING_DEG": (np.degrees(ang) + 90) % 360,
                    "FLIGHT_UID": f"TRK_{k:05d}",
                }
            )
        )
    for k in range(conflict_pairs):
        src = frames[k].copy()
        src["LATITUDE"] += 0.0008  # ~90 m north
        src["ALTITUDE_AGL_FT"] += 40.0
        src["FLIGHT_UID"] = f"TRK_{n_tracks + k:05d}"
        frames.append(src)
    df = pd.concat(frames, ignore_index=True)
    df["DATETIME_UTC"] = df["DATETIME_UTC"].dt.tz_localize(None)
    return df.reset_index(drop=True)
