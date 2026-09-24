"""Kepler.gl map building: widget for notebooks, standalone HTML for sharing.

The calling pattern is the one that works with keplergl 0.3.x in Jupyter:
``KeplerGl(height=..., config=...)``, ``add_data(data=gdf, name=...)``, ``save_to_html``, with a
``version: v1`` config whose ``mapState`` is centred on the data. If the widget renders blank,
run ``jupyter nbextension enable --py --sys-prefix keplergl`` once.

:class:`KeplerMapBuilder` writes the layer configs (a choropleth on a numeric field, categorical
colours for hotspot classes, point layers sized by a value) so the exported HTML opens styled
instead of as grey GeoJSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from fortress_gis.crs import WGS84
from fortress_gis.log import get_logger

LOGGER = get_logger(__name__)

# Uber "Global Warming" style sequential ramp and a diverging ramp for z-scores.
SEQUENTIAL_COLORS = ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]
DIVERGING_COLORS = ["#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C"]
HOTSPOT_CLASS_COLORS: dict[str, str] = {
    "hot 99%": "#B10026",
    "hot 95%": "#E31A1C",
    "hot 90%": "#FC4E2A",
    "not significant": "#D9D9D9",
    "cold 90%": "#9ECAE1",
    "cold 95%": "#4292C6",
    "cold 99%": "#08519C",
    "no data": "#F0F0F0",
    "high-high": "#D7191C",
    "low-low": "#2C7BB6",
    "high-low": "#FDAE61",
    "low-high": "#ABD9E9",
    "low": "#1A9850",
    "medium": "#FEE08B",
    "high": "#D73027",
}


def kepler_available() -> bool:
    try:
        import keplergl  # noqa: F401
    except ImportError:
        return False
    return True


def prepare_for_kepler(
    gdf: gpd.GeoDataFrame, *, drop_columns: Sequence[str] = ()
) -> gpd.GeoDataFrame:
    """Make a layer safe for the keplergl serializer.

    Reprojects to WGS84. Datetimes become ISO-8601 strings, which Kepler parses back into
    timestamps for its time filter. Booleans become 0/1 and integers become float, because
    keplergl 0.3.7 stringifies any column whose first value is not JSON-serialisable and numpy
    scalars are not. Categoricals become str, NaN becomes None, and nested object columns are
    dropped.
    """
    out = gdf.to_crs(WGS84) if gdf.crs is not None and gdf.crs != WGS84 else gdf.copy()
    out = out.drop(columns=[c for c in drop_columns if c in out.columns])
    out = out.reset_index(drop=True)
    geom = out.geometry.name
    for col in list(out.columns):
        if col == geom:
            continue
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = s.dt.tz_convert("UTC") if isinstance(s.dtype, pd.DatetimeTZDtype) else s
            out[col] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype(object).where(ts.notna(), None)
        elif pd.api.types.is_timedelta64_dtype(s):
            out[col] = s.dt.total_seconds().astype("float64")
        elif pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("float64")
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(str)
        elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            out[col] = s.astype("float64")
        elif pd.api.types.is_string_dtype(s) or s.dtype == object:
            sample = s.dropna()
            if len(sample) and not isinstance(sample.iloc[0], str | int | float):
                out = out.drop(columns=[col])
            else:
                out[col] = s.astype(object).where(s.notna(), None)
    out = out[~out.geometry.isna() & ~out.geometry.is_empty]
    return gpd.GeoDataFrame(out, geometry=geom, crs=WGS84)


def zoom_for_bounds(bounds: Sequence[float], *, width_px: int = 900) -> float:
    """Approximate web-mercator zoom level that fits ``bounds`` (lon/lat) in ``width_px``."""
    minx, miny, maxx, maxy = bounds
    span_lon = max(maxx - minx, 1e-6)
    span_lat = max(maxy - miny, 1e-6)
    zoom_lon = math.log2(360.0 * width_px / (256.0 * span_lon))
    zoom_lat = math.log2(180.0 * width_px / (256.0 * span_lat))
    return float(max(1.0, min(zoom_lon, zoom_lat) - 0.5))


def map_state_for_layer(gdf: gpd.GeoDataFrame, *, zoom: float | None = None) -> dict[str, float]:
    """``mapState`` centred on the layer's bounds."""
    bounds = gdf.to_crs(WGS84).total_bounds if gdf.crs is not None else gdf.total_bounds
    minx, miny, maxx, maxy = (float(b) for b in bounds)
    return {
        "latitude": (miny + maxy) / 2.0,
        "longitude": (minx + maxx) / 2.0,
        "zoom": zoom if zoom is not None else zoom_for_bounds(bounds),
        "bearing": 0,
        "pitch": 0,
    }


def _hex_to_rgb(color: str) -> list[int]:
    c = color.lstrip("#")
    return [int(c[i : i + 2], 16) for i in (0, 2, 4)]


def _field_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "real"
    return "string"


@dataclass
class KeplerLayer:
    name: str
    data: gpd.GeoDataFrame
    color_field: str | None = None
    color_scale: str = "quantile"
    colors: list[str] = field(default_factory=lambda: list(SEQUENTIAL_COLORS))
    categorical_colors: dict[str, str] | None = None
    size_field: str | None = None
    opacity: float = 0.7
    stroked: bool = True
    filled: bool = True
    radius: float = 10.0
    visible: bool = True
    label: str | None = None
    time_field: str | None = None

    def config(self) -> dict[str, Any]:
        geom_types = set(self.data.geom_type.unique())
        is_point = geom_types <= {"Point", "MultiPoint"}
        layer_id = self.name.lower().replace(" ", "_")
        vis: dict[str, Any] = {
            "opacity": self.opacity,
            "strokeOpacity": 0.8,
            "thickness": 0.5,
            "strokeColor": [40, 40, 40],
            "filled": self.filled,
            "stroked": self.stroked and not is_point,
            "radius": self.radius,
            "fixedRadius": False,
            "colorRange": {
                "name": "custom",
                "type": "sequential",
                "category": "Custom",
                "colors": list(self.colors),
            },
        }
        channels: dict[str, Any] = {}
        if self.color_field and self.color_field in self.data.columns:
            ftype = _field_type(self.data[self.color_field])
            if ftype == "string":
                cats = [c for c in self.data[self.color_field].dropna().unique()]
                palette = self.categorical_colors or HOTSPOT_CLASS_COLORS
                vis["colorRange"] = {
                    "name": "custom-categorical",
                    "type": "custom",
                    "category": "Custom",
                    "colors": [palette.get(str(c), "#999999") for c in cats],
                    "colorMap": [[str(c), palette.get(str(c), "#999999")] for c in cats],
                }
                channels["colorField"] = {"name": self.color_field, "type": "string"}
                channels["colorScale"] = "ordinal"
            else:
                channels["colorField"] = {"name": self.color_field, "type": "real"}
                channels["colorScale"] = self.color_scale
            if is_point:
                channels["strokeColorField"] = None
        if self.size_field and is_point and self.size_field in self.data.columns:
            vis["radiusRange"] = [2, 30]
            channels["sizeField"] = {"name": self.size_field, "type": "real"}
            channels["sizeScale"] = "sqrt"
        return {
            "id": layer_id,
            "type": "geojson",
            "config": {
                "dataId": self.name,
                "label": self.label or self.name,
                "color": _hex_to_rgb(self.colors[-1]),
                "columns": {"geojson": self.data.geometry.name},
                "isVisible": self.visible,
                "visConfig": vis,
            },
            "visualChannels": channels,
        }


@dataclass
class KeplerMapBuilder:
    """Collect layers and produce a Kepler.gl widget, HTML export, or config JSON.

    Example::

        builder = KeplerMapBuilder(title="Hex risk")
        builder.add_layer(hexes, "hex risk", color_field="event_count")
        builder.add_layer(events, "events", color_field="unacceptable", radius=4)
        widget = builder.widget()            # in Jupyter
        builder.save_html(EXPORTS / "risk.html")
    """

    title: str = "fortress-gis map"
    height: int = 600
    map_style: str = "dark"
    layers: list[KeplerLayer] = field(default_factory=list)
    map_state: dict[str, float] | None = None

    def add_layer(
        self,
        gdf: gpd.GeoDataFrame,
        name: str,
        *,
        color_field: str | None = None,
        color_scale: str = "quantile",
        colors: Sequence[str] | None = None,
        categorical_colors: dict[str, str] | None = None,
        size_field: str | None = None,
        opacity: float = 0.7,
        radius: float = 10.0,
        visible: bool = True,
        time_field: str | None = None,
        drop_columns: Sequence[str] = (),
    ) -> KeplerMapBuilder:
        """Add a styled layer.

        ``color_scale`` defaults to ``"quantile"``. When more than half of ``color_field`` is a
        single value (event counts that are mostly zero, say) every quantile break collapses onto
        that value and Kepler paints the whole layer in the top colour, so the scale is switched
        to ``"quantize"`` (equal intervals) for that layer.
        """
        data = prepare_for_kepler(gdf, drop_columns=drop_columns)
        if len(data) == 0:
            LOGGER.warning("Kepler layer %r is empty after preparation; skipping", name)
            return self
        if (
            color_field
            and color_scale == "quantile"
            and color_field in data.columns
            and pd.api.types.is_numeric_dtype(data[color_field])
        ):
            values = data[color_field].dropna()
            if len(values) and values.value_counts(normalize=True).iloc[0] > 0.5:
                color_scale = "quantize"
        self.layers.append(
            KeplerLayer(
                name=name,
                data=data,
                color_field=color_field,
                color_scale=color_scale,
                colors=list(colors) if colors else list(SEQUENTIAL_COLORS),
                categorical_colors=categorical_colors,
                size_field=size_field,
                opacity=opacity,
                radius=radius,
                visible=visible,
                time_field=time_field,
            )
        )
        return self

    def config(self) -> dict[str, Any]:
        """Kepler ``version: v1`` config with styled layers, optional time filter and map state."""
        if not self.layers:
            msg = "Add at least one layer before building a config"
            raise ValueError(msg)
        state = self.map_state or map_state_for_layer(self.layers[0].data)
        filters = [
            {
                "dataId": [layer.name],
                "id": f"time_{layer.name}",
                "name": [layer.time_field],
                "type": "timeRange",
                "enlarged": True,
            }
            for layer in self.layers
            if layer.time_field and layer.time_field in layer.data.columns
        ]
        return {
            "version": "v1",
            "config": {
                "visState": {
                    "layers": [layer.config() for layer in self.layers],
                    "filters": filters,
                    "interactionConfig": {
                        "tooltip": {"enabled": True, "fieldsToShow": self._tooltip_fields()}
                    },
                    "layerBlending": "normal",
                },
                "mapState": state,
                "mapStyle": {"styleType": self.map_style},
            },
        }

    def _tooltip_fields(self) -> dict[str, list[dict[str, Any]]]:
        fields: dict[str, list[dict[str, Any]]] = {}
        for layer in self.layers:
            cols = [c for c in layer.data.columns if c != layer.data.geometry.name][:8]
            fields[layer.name] = [{"name": c, "format": None} for c in cols]
        return fields

    def datasets(self) -> dict[str, gpd.GeoDataFrame]:
        return {layer.name: layer.data for layer in self.layers}

    def widget(self) -> Any:
        """``KeplerGl`` widget with all layers added (requires the ``viz`` extra)."""
        from keplergl import KeplerGl

        m = KeplerGl(height=self.height, config=self.config())
        for name, data in self.datasets().items():
            m.add_data(data=data, name=name)
        return m

    def save_html(self, path: str | Path, *, read_only: bool = False) -> Path:
        """Write a standalone HTML map (opens in any browser, no server needed)."""
        from keplergl import KeplerGl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = KeplerGl(height=self.height, config=self.config())
        for name, data in self.datasets().items():
            m.add_data(data=data, name=name)
        m.save_to_html(file_name=str(path), read_only=read_only)
        LOGGER.info("Saved Kepler.gl HTML -> %s", path)
        return path

    def save_config(self, path: str | Path) -> Path:
        """Persist the config JSON so a map can be re-created with new data later."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.config(), indent=2, default=_json_default), encoding="utf-8"
        )
        return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def enable_nbextension() -> str:
    """Run ``jupyter nbextension enable --py --sys-prefix keplergl`` (classic Notebook setups)."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbextension",
            "enable",
            "--py",
            "--sys-prefix",
            "keplergl",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout or result.stderr or "Kepler.gl nbextension ready."
