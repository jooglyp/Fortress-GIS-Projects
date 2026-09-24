"""Spatial autocorrelation, count-data tests, regression and map classification."""

from fortress_gis.stats.classify import classify_breaks, risk_levels
from fortress_gis.stats.hypothesis import (
    benjamini_hochberg,
    chi_square_independence,
    poisson_excess_test,
    poisson_rate_test,
)
from fortress_gis.stats.regression import (
    RegressionResult,
    fit_ols,
    fit_regularized_ols,
    tidy_coefficients,
)
from fortress_gis.stats.spatial_autocorr import (
    HotspotResult,
    build_weights,
    getis_ord_hotspots,
    local_moran_clusters,
    threshold_distance,
)

__all__ = [
    "HotspotResult",
    "RegressionResult",
    "benjamini_hochberg",
    "build_weights",
    "chi_square_independence",
    "classify_breaks",
    "fit_ols",
    "fit_regularized_ols",
    "getis_ord_hotspots",
    "local_moran_clusters",
    "poisson_excess_test",
    "poisson_rate_test",
    "risk_levels",
    "threshold_distance",
    "tidy_coefficients",
]
