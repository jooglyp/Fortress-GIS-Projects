"""Shared fixtures. Synthetic inputs come from the top-level ``datasets`` package."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def dem():
    from datasets import synthetic_dem

    return synthetic_dem(60, 80)


@pytest.fixture(scope="session")
def dem_tiles(dem, tmp_path_factory):
    from datasets import synthetic_dem_tiles

    return synthetic_dem_tiles(dem, tmp_path_factory.mktemp("tiles"))


@pytest.fixture(scope="session")
def chlorophyll_files(tmp_path_factory):
    from datasets import synthetic_chlorophyll_grids

    return synthetic_chlorophyll_grids(
        tmp_path_factory.mktemp("chl"),
        years=(2017, 2018, 2019),
        days_of_year=(15, 200),
        lon=(-40, -10, 16),
        lat=(25, 55, 16),
    )


@pytest.fixture(scope="session")
def aircraft():
    from datasets import synthetic_aircraft_positions

    return synthetic_aircraft_positions(30, points_per_track=50, conflict_pairs=4)


@pytest.fixture(scope="session")
def telemetry():
    from datasets import synthetic_vessel_telemetry

    return synthetic_vessel_telemetry(1500)
