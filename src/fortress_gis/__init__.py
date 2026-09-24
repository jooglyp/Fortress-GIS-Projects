"""Raster and vector geoprocessing, spatial statistics and predictive modelling in one package.

Subpackages are split by what they do, not by which project they came from. ``io`` reads and
writes vectors, rasters and gridded tables. ``compute`` holds the dask client and partitioning
helpers. ``grids`` builds hexagon lattices and aggregates points into them. ``raster`` has the
terrain hydrology routines (depression filling, D8, accumulation, catchments) and line sampling.
``features`` derives time-of-day, seasonal, encoded and proximity features. ``stats`` covers
local spatial autocorrelation, count-data tests, OLS and map classification. ``ml`` wraps
scikit-learn pipelines, model comparison and tuning. ``viz`` writes Kepler.gl maps, QGIS bundles
and static figures. ``domains`` strings these together for hydrology, oceanography, airspace and
maritime work. Synthetic inputs for the demos and tests live outside the package, in the
repository's ``datasets/`` folder.
"""

from __future__ import annotations

from fortress_gis._version import __version__
from fortress_gis.config import DATA_DIR, EXPORTS_DIR, REPO_ROOT, domain_data_dir

__all__ = ["DATA_DIR", "EXPORTS_DIR", "REPO_ROOT", "__version__", "domain_data_dir"]
