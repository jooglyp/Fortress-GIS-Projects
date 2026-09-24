"""Quick static maps with matplotlib (optional contextily basemap)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from fortress_gis.io.raster import RasterInfo


def plot_layer(
    gdf: gpd.GeoDataFrame,
    *,
    column: str | None = None,
    scheme: str | None = "NaturalBreaks",
    k: int = 5,
    cmap: str = "OrRd",
    basemap: bool = False,
    ax: Any | None = None,
    title: str | None = None,
    alpha: float = 0.75,
    figsize: tuple[float, float] = (9, 9),
    save: str | Path | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Choropleth or point map. ``basemap=True`` adds OSM tiles when contextily is installed."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    data = gdf.to_crs(3857) if basemap and gdf.crs is not None else gdf
    kwargs: dict[str, Any] = {"ax": ax, "alpha": alpha, "legend": column is not None, **plot_kwargs}
    if column is not None:
        kwargs.update(column=column, cmap=cmap)
        if (
            scheme
            and np.issubdtype(np.asarray(data[column].dropna()).dtype, np.number)
            and data[column].nunique() > k
        ):
            kwargs.update(scheme=scheme, k=k)
    data.plot(**kwargs)
    if basemap:
        try:
            import contextily as ctx

            ctx.add_basemap(ax, crs=data.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except ImportError:
            ax.set_title((title or "") + "  (install the 'basemap' extra for tiles)")
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
    return ax


def plot_raster(
    raster: RasterInfo,
    *,
    ax: Any | None = None,
    cmap: str = "terrain",
    title: str | None = None,
    overlay: gpd.GeoDataFrame | None = None,
    figsize: tuple[float, float] = (9, 7),
    log: bool = False,
    save: str | Path | None = None,
) -> Any:
    """Show a raster in its CRS extent with an optional vector overlay."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    data = raster.masked()
    if log:
        data = np.log1p(np.where(np.isfinite(data), np.clip(data, 0, None), np.nan))
    minx, miny, maxx, maxy = raster.bounds
    im = ax.imshow(data, extent=(minx, maxx, miny, maxy), cmap=cmap, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.04)
    if overlay is not None and len(overlay):
        ov = overlay.to_crs(raster.crs) if (overlay.crs is not None and raster.crs) else overlay
        ov.plot(ax=ax, facecolor="none", edgecolor="cyan", linewidth=1.0)
    if title:
        ax.set_title(title)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
    return ax
