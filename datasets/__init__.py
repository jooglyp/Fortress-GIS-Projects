"""Synthetic data generators for demos and tests. Not part of the installed package."""

from datasets.synthetic import (
    synthetic_aircraft_positions,
    synthetic_chlorophyll_grids,
    synthetic_dem,
    synthetic_dem_tiles,
    synthetic_vessel_telemetry,
)

__all__ = [
    "synthetic_aircraft_positions",
    "synthetic_chlorophyll_grids",
    "synthetic_dem",
    "synthetic_dem_tiles",
    "synthetic_vessel_telemetry",
]
