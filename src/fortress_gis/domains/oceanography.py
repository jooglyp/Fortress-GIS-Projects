"""Oceanography workflow: gridded satellite composites to hex covariates and hotspot maps.

Many wide CSV grids (one per acquisition) are stacked into a point layer, aggregated to hexagon
cells, reduced to seasonal and interannual covariates, and screened for spatial clusters with
Getis-Ord G* and local Moran's I. The original was a chlorophyll-a study; nothing here is
specific to that variable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

from fortress_gis.features.seasonal import (
    SeasonCalendar,
    interannual_variance,
    missingness_report,
    seasonal_means,
    seasonal_variance,
    year_over_year_change,
    year_season_means,
)
from fortress_gis.grids.aggregate import points_to_cells, points_to_cells_dask
from fortress_gis.grids.hexgrid import hex_grid_for_layer
from fortress_gis.io.tabular import read_grid_csvs_as_points
from fortress_gis.log import get_logger
from fortress_gis.stats.spatial_autocorr import (
    HotspotResult,
    build_weights,
    getis_ord_hotspots,
    local_moran_clusters,
)
from fortress_gis.viz.kepler import HOTSPOT_CLASS_COLORS, KeplerMapBuilder
from fortress_gis.viz.qgis import QgisBundle

LOGGER = get_logger(__name__)


@dataclass
class CompositeStack:
    """Acquisition points, their hex aggregation, and the covariates derived from it."""

    points: gpd.GeoDataFrame
    cells: gpd.GeoDataFrame
    per_acquisition: gpd.GeoDataFrame
    season_means: pd.DataFrame
    yoy: pd.DataFrame
    covariates: gpd.GeoDataFrame
    missing: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    @property
    def covariate_columns(self) -> list[str]:
        return [c for c in self.covariates.columns if c not in ("geometry", "hex_id")]


def load_composites(
    paths: Iterable[str | Path],
    *,
    prefix: str = "chlor_",
    limit: int | None = None,
) -> gpd.GeoDataFrame:
    """Read every acquisition CSV as a point layer with one value column per acquisition."""
    pts = read_grid_csvs_as_points(list(paths), prefix=prefix, limit=limit)
    LOGGER.info("Loaded %d lattice points x %d acquisitions", len(pts), pts.shape[1] - 1)
    return pts


def build_hex_covariates(
    points: gpd.GeoDataFrame,
    *,
    hex_radius_m: float | None = None,
    hex_area_m2: float | None = None,
    calendar: SeasonCalendar | None = None,
    use_dask: bool = False,
    id_column: str = "hex_id",
) -> CompositeStack:
    """Aggregate acquisitions to hexes and derive seasonal/interannual covariates.

    Produces, per hex: ``<season>_mean`` (climatology), ``total_mean``, ``<season>_var``
    (variance of relative year-over-year change) and ``<year>_var`` (variance across seasons
    within a year).
    """
    if hex_radius_m is None and hex_area_m2 is None:
        # default ~ 3x lattice spacing so every hex sees several observations
        res = points.geometry.x.drop_duplicates().sort_values().diff().dropna()
        spacing_deg = float(res[res > 0].min()) if len(res) else 0.5
        hex_radius_m = spacing_deg * 111_000 * 3
    cells = hex_grid_for_layer(points, radius=hex_radius_m, area=hex_area_m2, id_column=id_column)
    value_columns = [c for c in points.columns if c != "geometry"]
    agg_fn = points_to_cells_dask if use_dask else points_to_cells
    per_acq = agg_fn(points, cells, cell_id=id_column, value_columns=value_columns, agg="mean")
    per_acq = per_acq[per_acq["n_points"] > 0].copy()
    LOGGER.info(
        "Aggregated to %d populated hexes (radius %.0f m)",
        len(per_acq),
        cells.attrs.get("radius_m", -1),
    )

    season = year_season_means(per_acq, id_columns=[id_column], calendar=calendar)
    yoy = year_over_year_change(season, id_columns=[id_column])
    means = seasonal_means(season, id_columns=[id_column])
    svar = seasonal_variance(yoy, id_columns=[id_column])
    ivar = interannual_variance(yoy, id_columns=[id_column])

    def _table(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(frame.drop(columns="geometry", errors="ignore"))

    cov = (
        _table(means)
        .merge(_table(svar), on=id_column, how="left")
        .merge(_table(ivar), on=id_column, how="left")
    )
    covariates = per_acq[[id_column, "geometry"]].merge(cov, on=id_column, how="left")
    covariates = gpd.GeoDataFrame(covariates, geometry="geometry", crs=per_acq.crs)
    return CompositeStack(
        points,
        cells,
        per_acq,
        season,
        yoy,
        covariates,
        missingness_report(covariates.drop(columns="geometry")),
    )


def screen_hotspots(
    covariates: gpd.GeoDataFrame,
    attributes: Sequence[str],
    *,
    method: str = "gstar",
    permutations: int = 999,
    weights_kind: str = "distance_band",
    k: int = 8,
    seed: int = 0,
) -> dict[str, HotspotResult]:
    """Run G* (``method='gstar'``) or local Moran (``'lisa'``) for each covariate.

    Rows with missing values are dropped per attribute; the weights are built once on the full
    layer and re-used when no rows were dropped.
    """
    results: dict[str, HotspotResult] = {}
    base_w = build_weights(covariates, kind=weights_kind, k=k)
    for attr in attributes:
        layer = covariates.dropna(subset=[attr])
        w = base_w if len(layer) == len(covariates) else None
        if method == "gstar":
            results[attr] = getis_ord_hotspots(
                layer, attr, weights=w, permutations=permutations, seed=seed
            )
        elif method == "lisa":
            results[attr] = local_moran_clusters(
                layer, attr, weights=w, permutations=permutations, seed=seed
            )
        else:
            raise ValueError("method must be 'gstar' or 'lisa'")
        LOGGER.info("%s on %s: %s", method, attr, results[attr].summary().to_dict())
    return results


def merge_hotspot_layers(
    results: dict[str, HotspotResult], *, id_column: str = "hex_id"
) -> gpd.GeoDataFrame:
    """Left-join the per-attribute class/z columns onto a single hex layer."""
    first = next(iter(results.values())).layer
    out = first[[id_column, "geometry"]].copy()
    for res in results.values():
        cols = [c for c in res.layer.columns if c.startswith(f"{res.attribute}_")]
        out = out.merge(res.layer[[id_column, res.attribute, *cols]], on=id_column, how="left")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=first.crs)


def export_artifacts(
    stack: CompositeStack,
    results: dict[str, HotspotResult],
    out_dir: str | Path,
    *,
    name: str = "composites",
    kepler_html: bool = True,
) -> dict[str, Path]:
    """QGIS bundle (covariates + one styled layer per hotspot result) and a Kepler map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = QgisBundle(out / "qgis", name=name)
    bundle.add_vector(
        stack.covariates,
        "hex_covariates",
        graduated="total_mean" if "total_mean" in stack.covariates else None,
    )
    for attr, res in results.items():
        cls_col = res.class_column
        bundle.add_vector(res.layer, f"hotspots_{attr}", categorized=cls_col)
    bundle.write()
    paths = {"qgis": out / "qgis"}
    if kepler_html:
        builder = KeplerMapBuilder(title=f"{name} - hotspots", map_style="dark")
        first = True
        for attr, res in results.items():
            cls_col = res.class_column
            builder.add_layer(
                res.layer,
                f"{attr} ({res.method})",
                color_field=cls_col,
                categorical_colors=HOTSPOT_CLASS_COLORS,
                visible=first,
                opacity=0.75,
            )
            first = False
        if "total_mean" in stack.covariates:
            builder.add_layer(
                stack.covariates,
                "total_mean",
                color_field="total_mean",
                color_scale="quantile",
                visible=False,
            )
        paths["kepler_html"] = builder.save_html(out / f"{name}_kepler.html")
        paths["kepler_config"] = builder.save_config(out / f"{name}_kepler_config.json")
    return paths


__all__ = [
    "CompositeStack",
    "build_hex_covariates",
    "export_artifacts",
    "load_composites",
    "merge_hotspot_layers",
    "screen_hotspots",
]
