"""Local indicators of spatial association on cell/point layers (PySAL ``esda`` + ``libpysal``).

The original ``HotSpots`` class rebuilt its weights for every attribute. Here
:func:`build_weights` builds them once (inverse-distance within a band, k nearest neighbours, or
Queen contiguity for polygon lattices) and :func:`getis_ord_hotspots` and
:func:`local_moran_clusters` reuse them across attributes. Both return the input layer with
z-scores, pseudo p-values and a class label, aligned to the original index even when rows with
NaN had to be dropped for the computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import weights as lpw

from fortress_gis.crs import is_metric, suggest_metric_crs, to_crs_obj
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

WeightsKind = Literal["distance_band", "knn", "queen"]

# One-sided critical values used by the ArcGIS-style hot/cold classification.
_Z_90, _Z_95, _Z_99 = 1.645, 1.960, 2.576


@dataclass
class HotspotResult:
    """Layer with hotspot columns plus the weights and raw estimator used."""

    layer: gpd.GeoDataFrame
    weights: Any
    estimator: Any
    attribute: str
    method: str = "gstar"

    @property
    def class_column(self) -> str:
        return [c for c in self.layer.columns if c.endswith("_class")][-1]

    def summary(self) -> pd.Series:
        return self.layer[self.class_column].value_counts(dropna=False)


def _metric_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Centroid points in a metric CRS (weights need Euclidean distances)."""
    crs = gdf.crs if is_metric(gdf.crs) else suggest_metric_crs(gdf)
    proj = gdf.to_crs(to_crs_obj(crs))
    cents = proj.geometry.centroid
    return gpd.GeoDataFrame(proj.drop(columns=proj.geometry.name), geometry=cents, crs=proj.crs)


def threshold_distance(points: gpd.GeoDataFrame) -> float:
    """Smallest distance band that leaves no observation without a neighbour.

    Uses ``libpysal.weights.min_threshold_distance`` (max nearest-neighbour distance). For tiny
    layers (< 3 features) falls back to the maximum pairwise distance.
    """
    xy = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    if len(xy) < 3:
        from scipy.spatial.distance import pdist

        return float(pdist(xy).max()) if len(xy) > 1 else 1.0
    return float(lpw.min_threshold_distance(xy))


def build_weights(
    gdf: gpd.GeoDataFrame,
    *,
    kind: WeightsKind = "distance_band",
    threshold: float | None = None,
    threshold_multiplier: float = 1.0,
    k: int = 8,
    alpha: float = -1.5,
    binary: bool = False,
    row_standardize: bool = True,
    silence_warnings: bool = True,
) -> Any:
    """Spatial weights for ``gdf`` (rows must already be NaN-free for the analysed attribute).

    Parameters
    ----------
    kind:
        ``"distance_band"`` uses inverse-distance weights (exponent ``alpha``) within
        ``threshold`` metres, defaulting to :func:`threshold_distance` times
        ``threshold_multiplier``. ``"knn"`` uses the ``k`` nearest neighbours. ``"queen"`` is
        polygon contiguity.
    row_standardize:
        Apply ``w.transform = "R"`` so each row sums to one (standard for G*/LISA).
    """
    if kind == "queen":
        w = lpw.Queen.from_dataframe(gdf, silence_warnings=silence_warnings, use_index=False)
        w.attrs = {"kind": kind}
    else:
        pts = _metric_centroids(gdf)
        if kind == "knn":
            w = lpw.KNN.from_dataframe(
                pts, k=min(k, len(pts) - 1), silence_warnings=silence_warnings
            )
            w.attrs = {"kind": kind, "k": k}
        else:
            thresh = (
                threshold if threshold is not None else threshold_distance(pts)
            ) * threshold_multiplier
            w = lpw.DistanceBand.from_dataframe(
                pts, threshold=thresh, alpha=alpha, binary=binary, silence_warnings=silence_warnings
            )
            w.attrs = {"kind": kind, "threshold_m": float(thresh), "alpha": alpha, "binary": binary}
    if row_standardize:
        w.transform = "R"
    LOGGER.info("weights: %s, n=%d, islands=%d", w.attrs, w.n, len(w.islands))
    return w


def _classify_z(
    z: np.ndarray, p: np.ndarray | None = None, alpha: float | None = None
) -> np.ndarray:
    """Hot/cold spot classes at 90/95/99% confidence from z-scores (ArcGIS Gi* convention)."""
    out = np.full(z.shape, "not significant", dtype=object)
    out[z >= _Z_90] = "hot 90%"
    out[z >= _Z_95] = "hot 95%"
    out[z >= _Z_99] = "hot 99%"
    out[z <= -_Z_90] = "cold 90%"
    out[z <= -_Z_95] = "cold 95%"
    out[z <= -_Z_99] = "cold 99%"
    if p is not None and alpha is not None:
        out[p > alpha] = "not significant"
    return out


def getis_ord_hotspots(
    gdf: gpd.GeoDataFrame,
    attribute: str,
    *,
    weights: Any | None = None,
    permutations: int = 999,
    star: bool = True,
    seed: int | None = 0,
    **weights_kwargs: Any,
) -> HotspotResult:
    """Getis-Ord G_i* hot/cold spot statistics for ``attribute``.

    Adds ``<attribute>_gi_z`` (analytical z), ``<attribute>_gi_p_sim`` (pseudo p) and
    ``<attribute>_gi_class``. Rows with NaN ``attribute`` are excluded from the estimator and
    receive NaN / ``"no data"``.
    """
    from esda.getisord import G_Local

    y_all = pd.to_numeric(gdf[attribute], errors="coerce")
    valid = y_all.notna().to_numpy()
    sub = gdf.loc[valid]
    w = weights if weights is not None else build_weights(sub, **weights_kwargs)
    y = y_all.loc[valid].to_numpy(dtype="float64")
    g = G_Local(y, w, transform="R", star=star, permutations=permutations, seed=seed)
    out = gdf.copy()
    z_col, p_col, c_col = f"{attribute}_gi_z", f"{attribute}_gi_p_sim", f"{attribute}_gi_class"
    out[z_col] = np.nan
    out[p_col] = np.nan
    out.loc[valid, z_col] = np.asarray(g.Zs, dtype="float64")
    out.loc[valid, p_col] = np.asarray(g.p_sim, dtype="float64")
    classes = np.full(len(out), "no data", dtype=object)
    classes[valid] = _classify_z(np.asarray(g.Zs))
    out[c_col] = classes
    out.attrs["hotspot_weights"] = getattr(w, "attrs", {})
    return HotspotResult(layer=out, weights=w, estimator=g, attribute=attribute, method="gstar")


_LISA_QUADRANTS = {1: "high-high", 2: "low-high", 3: "low-low", 4: "high-low"}


def local_moran_clusters(
    gdf: gpd.GeoDataFrame,
    attribute: str,
    *,
    weights: Any | None = None,
    permutations: int = 999,
    alpha: float = 0.05,
    seed: int | None = 0,
    **weights_kwargs: Any,
) -> HotspotResult:
    """Local Moran's I (LISA) cluster/outlier classification for ``attribute``.

    Adds ``<attribute>_lisa_i``, ``<attribute>_lisa_z``, ``<attribute>_lisa_p_sim`` and
    ``<attribute>_lisa_class`` in {high-high, low-low, high-low, low-high, not significant}.
    """
    from esda.moran import Moran_Local

    y_all = pd.to_numeric(gdf[attribute], errors="coerce")
    valid = y_all.notna().to_numpy()
    sub = gdf.loc[valid]
    w = weights if weights is not None else build_weights(sub, **weights_kwargs)
    y = y_all.loc[valid].to_numpy(dtype="float64")
    lisa = Moran_Local(y, w, permutations=permutations, seed=seed)
    out = gdf.copy()
    base = f"{attribute}_lisa"
    for col in (f"{base}_i", f"{base}_z", f"{base}_p_sim"):
        out[col] = np.nan
    out.loc[valid, f"{base}_i"] = np.asarray(lisa.Is, dtype="float64")
    out.loc[valid, f"{base}_z"] = np.asarray(lisa.z_sim, dtype="float64")
    out.loc[valid, f"{base}_p_sim"] = np.asarray(lisa.p_sim, dtype="float64")
    labels = np.array([_LISA_QUADRANTS[int(q)] for q in lisa.q], dtype=object)
    labels[np.asarray(lisa.p_sim) > alpha] = "not significant"
    classes = np.full(len(out), "no data", dtype=object)
    classes[valid] = labels
    out[f"{base}_class"] = classes
    out.attrs["hotspot_weights"] = getattr(w, "attrs", {})
    return HotspotResult(layer=out, weights=w, estimator=lisa, attribute=attribute, method="lisa")


def batch_hotspots(
    gdf: gpd.GeoDataFrame,
    attributes: list[str],
    *,
    method: Literal["gi_star", "lisa"] = "lisa",
    permutations: int = 999,
    **weights_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Run one hotspot method over several attributes, sharing the weights when NaN masks agree."""
    out = gdf
    cache: dict[tuple[bool, ...], Any] = {}
    for attr in attributes:
        mask = tuple(pd.to_numeric(gdf[attr], errors="coerce").notna().tolist())
        w = cache.get(mask)
        fn = local_moran_clusters if method == "lisa" else getis_ord_hotspots
        res = fn(out, attr, weights=w, permutations=permutations, **weights_kwargs)
        cache.setdefault(mask, res.weights)
        out = res.layer
    return out
