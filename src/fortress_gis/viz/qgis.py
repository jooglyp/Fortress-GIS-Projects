"""QGIS hand-off artifacts: GeoPackage layers, GeoTIFFs, QML styles and a PyQGIS loader script.

The hydrology project ran inside QGIS 2 through ``processing.runalg``. Here the analysis runs
in Python and QGIS is the viewer. A :class:`QgisBundle` is a directory with one multi-layer
GeoPackage (long column names survive, unlike Shapefile), any GeoTIFF rasters, one ``.qml``
style per layer (graduated, categorized or raster pseudocolour), a ``load_in_qgis.py`` script to
paste into the QGIS 3 Python console, and a ``manifest.json`` indexing all of it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd

from fortress_gis.io.raster import RasterInfo, write_raster
from fortress_gis.io.vector import write_vector
from fortress_gis.log import get_logger
from fortress_gis.stats.classify import classify_breaks
from fortress_gis.viz.kepler import HOTSPOT_CLASS_COLORS, SEQUENTIAL_COLORS

LOGGER = get_logger(__name__)

_GEOM_TYPE_CODE = {
    "Point": 0,
    "MultiPoint": 0,
    "LineString": 1,
    "MultiLineString": 1,
    "Polygon": 2,
    "MultiPolygon": 2,
}


def _rgba(hex_color: str, alpha: int = 255) -> str:
    c = hex_color.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"{r},{g},{b},{alpha}"


def _symbol_xml(
    name: str,
    geom_code: int,
    color: str,
    *,
    outline: str = "35,35,35,255",
    size: float = 2.5,
    width: float = 0.26,
    alpha: float = 0.85,
) -> str:
    """One ``<symbol>`` element using ``<prop>`` tags (readable by every QGIS 3.x release)."""
    if geom_code == 0:
        return (
            f'<symbol type="marker" name="{name}" alpha="{alpha}" clip_to_extent="1" force_rhr="0">'
            f'<layer class="SimpleMarker" enabled="1" locked="0" pass="0">'
            f'<prop k="color" v="{color}"/><prop k="name" v="circle"/><prop k="outline_color" v="{outline}"/>'
            f'<prop k="outline_width" v="0.2"/><prop k="size" v="{size}"/><prop k="size_unit" v="MM"/>'
            f"</layer></symbol>"
        )
    if geom_code == 1:
        return (
            f'<symbol type="line" name="{name}" alpha="{alpha}" clip_to_extent="1" force_rhr="0">'
            f'<layer class="SimpleLine" enabled="1" locked="0" pass="0">'
            f'<prop k="line_color" v="{color}"/><prop k="line_width" v="{max(width, 0.6)}"/><prop k="line_width_unit" v="MM"/>'
            f'<prop k="line_style" v="solid"/>'
            f"</layer></symbol>"
        )
    return (
        f'<symbol type="fill" name="{name}" alpha="{alpha}" clip_to_extent="1" force_rhr="0">'
        f'<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
        f'<prop k="color" v="{color}"/><prop k="outline_color" v="{outline}"/><prop k="outline_width" v="{width}"/>'
        f'<prop k="style" v="solid"/><prop k="outline_style" v="solid"/>'
        f"</layer></symbol>"
    )


def _wrap_qml(renderer_xml: str, geom_code: int | None) -> str:
    geom = f"<layerGeometryType>{geom_code}</layerGeometryType>" if geom_code is not None else ""
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.34.0" styleCategories="Symbology">\n'
        f"{renderer_xml}\n{geom}\n</qgis>\n"
    )


def _interpolate_colors(colors: Sequence[str], n: int) -> list[str]:
    if n <= len(colors):
        idx = np.linspace(0, len(colors) - 1, n).round().astype(int)
        return [colors[i] for i in idx]
    rgb = np.array(
        [[int(c.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for c in colors], dtype=float
    )
    xs = np.linspace(0, len(colors) - 1, n)
    out = []
    for x in xs:
        lo, hi = int(np.floor(x)), int(np.ceil(x))
        t = x - lo
        c = rgb[lo] * (1 - t) + rgb[hi] * t
        out.append("#{:02X}{:02X}{:02X}".format(*tuple(round(v) for v in c)))
    return out


def write_qml_graduated(
    path: str | Path,
    gdf: gpd.GeoDataFrame,
    field_name: str,
    *,
    k: int = 5,
    scheme: str = "natural_breaks",
    colors: Sequence[str] = SEQUENTIAL_COLORS,
    precision: int = 3,
) -> Path:
    """Graduated (choropleth) renderer QML for a numeric field using mapclassify breaks."""
    values = pd.to_numeric(gdf[field_name], errors="coerce").to_numpy(dtype="float64")
    _, bounds = classify_breaks(values, k=k, scheme=scheme)
    finite = values[np.isfinite(values)]
    lower = float(finite.min()) if finite.size else 0.0
    uppers = [float(b) for b in bounds] or [lower]
    geom_code = _GEOM_TYPE_CODE.get(str(gdf.geom_type.iloc[0]), 2)
    palette = _interpolate_colors(list(colors), len(uppers))
    ranges, symbols = [], []
    lo = lower
    for i, (hi, color) in enumerate(zip(uppers, palette, strict=True)):
        label = f"{lo:.{precision}f} - {hi:.{precision}f}"
        ranges.append(
            f'<range lower="{lo}" upper="{hi}" symbol="{i}" label="{escape(label)}" render="true"/>'
        )
        symbols.append(_symbol_xml(str(i), geom_code, _rgba(color)))
        lo = hi
    renderer = (
        f'<renderer-v2 type="graduatedSymbol" attr="{escape(field_name)}" graduatedMethod="GraduatedColor" '
        'symbollevels="0" forceraster="0" enableorderby="0">'
        f"<ranges>{''.join(ranges)}</ranges><symbols>{''.join(symbols)}</symbols>"
        "<rotation/><sizescale/></renderer-v2>"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_wrap_qml(renderer, geom_code), encoding="utf-8")
    return out


def write_qml_categorized(
    path: str | Path,
    gdf: gpd.GeoDataFrame,
    field_name: str,
    *,
    colors: Mapping[str, str] | None = None,
) -> Path:
    """Categorized renderer QML (e.g. hotspot classes, risk labels) with a fixed colour lookup."""
    palette = dict(HOTSPOT_CLASS_COLORS)
    if colors:
        palette.update(colors)
    cats = [str(c) for c in pd.unique(gdf[field_name].dropna())]
    geom_code = _GEOM_TYPE_CODE.get(str(gdf.geom_type.iloc[0]), 2)
    fallback = _interpolate_colors(SEQUENTIAL_COLORS, max(len(cats), 1))
    categories, symbols = [], []
    for i, cat in enumerate(cats):
        color = palette.get(cat, fallback[i])
        categories.append(
            f'<category value="{escape(cat)}" symbol="{i}" label="{escape(cat)}" render="true"/>'
        )
        symbols.append(_symbol_xml(str(i), geom_code, _rgba(color)))
    renderer = (
        f'<renderer-v2 type="categorizedSymbol" attr="{escape(field_name)}" symbollevels="0" forceraster="0" enableorderby="0">'
        f"<categories>{''.join(categories)}</categories><symbols>{''.join(symbols)}</symbols>"
        "<rotation/><sizescale/></renderer-v2>"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_wrap_qml(renderer, geom_code), encoding="utf-8")
    return out


def write_qml_raster(
    path: str | Path,
    raster: RasterInfo,
    *,
    colors: Sequence[str] = ("#0B2545", "#13315C", "#1F7A8C", "#BFDBF7", "#F5F5F5"),
    n_classes: int = 5,
    discrete: bool = False,
    nodata_transparent: bool = True,
) -> Path:
    """Single-band pseudocolour renderer QML for a continuous or class raster."""
    data = raster.masked()
    finite = data[np.isfinite(data)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    stops = np.linspace(vmin, vmax, n_classes)
    palette = _interpolate_colors(list(colors), n_classes)
    items = "".join(
        f'<item alpha="255" value="{v}" label="{v:.3g}" color="{c}"/>'
        for v, c in zip(stops, palette, strict=True)
    )
    ramp_type = "DISCRETE" if discrete else "INTERPOLATED"
    renderer = (
        '<pipe><rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" alphaBand="-1" '
        f'classificationMin="{vmin}" classificationMax="{vmax}">'
        "<rasterTransparency/><rastershader>"
        f'<colorrampshader colorRampType="{ramp_type}" classificationMode="1" clip="0" minimumValue="{vmin}" maximumValue="{vmax}">'
        f"{items}</colorrampshader></rastershader></rasterrenderer>"
        '<brightnesscontrast brightness="0" contrast="0"/><huesaturation/><rasterresampler maxOversampling="2"/></pipe>'
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_wrap_qml(renderer, None), encoding="utf-8")
    return out


def write_qgis_loader_script(path: str | Path, manifest: Mapping[str, object]) -> Path:
    """PyQGIS (QGIS 3) console script that loads every layer in ``manifest`` with its style.

    The old ``execute-in-qgis.py`` ran the analysis inside QGIS. This script only loads results
    that were produced outside it.
    """
    script = f'''"""Auto-generated by fortress-gis. Paste into the QGIS 3 Python console (or run with
`qgis --code load_in_qgis.py`) to load the exported layers with their styles."""
import json
from pathlib import Path

from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

MANIFEST = json.loads(r"""{json.dumps(manifest, indent=2, default=str)}""")
ROOT = Path(MANIFEST["root"])
project = QgsProject.instance()

for entry in MANIFEST["rasters"]:
    layer = QgsRasterLayer(str(ROOT / entry["path"]), entry["name"])
    if layer.isValid():
        if entry.get("style"):
            layer.loadNamedStyle(str(ROOT / entry["style"]))
        project.addMapLayer(layer)
    else:
        print("Invalid raster:", entry)

for entry in MANIFEST["vectors"]:
    uri = f"{{ROOT / entry['path']}}|layername={{entry['layer']}}"
    layer = QgsVectorLayer(uri, entry["name"], "ogr")
    if layer.isValid():
        if entry.get("style"):
            layer.loadNamedStyle(str(ROOT / entry["style"]))
            layer.triggerRepaint()
        project.addMapLayer(layer)
    else:
        print("Invalid vector:", entry)

print("Loaded", len(MANIFEST["vectors"]), "vector and", len(MANIFEST["rasters"]), "raster layers")
'''
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script, encoding="utf-8")
    return out


@dataclass
class QgisBundle:
    """Accumulate layers, then :meth:`write` them as a styled QGIS package."""

    directory: Path
    name: str = "fortress_gis"
    vectors: list[dict[str, object]] = field(default_factory=list)
    rasters: list[dict[str, object]] = field(default_factory=list)
    _vector_data: dict[str, gpd.GeoDataFrame] = field(default_factory=dict, repr=False)
    _raster_data: dict[str, RasterInfo] = field(default_factory=dict, repr=False)

    def add_vector(
        self,
        gdf: gpd.GeoDataFrame,
        layer: str,
        *,
        graduated: str | None = None,
        categorized: str | None = None,
        k: int = 5,
        scheme: str = "natural_breaks",
        colors: Sequence[str] | Mapping[str, str] | None = None,
    ) -> QgisBundle:
        if len(gdf) == 0:
            LOGGER.warning("QGIS layer %r is empty; skipping", layer)
            return self
        self._vector_data[layer] = gdf
        self.vectors.append(
            {
                "layer": layer,
                "graduated": graduated,
                "categorized": categorized,
                "k": k,
                "scheme": scheme,
                "colors": colors,
            }
        )
        return self

    def add_raster(
        self,
        raster: RasterInfo,
        name: str,
        *,
        discrete: bool = False,
        colors: Sequence[str] | None = None,
    ) -> QgisBundle:
        self._raster_data[name] = raster
        self.rasters.append({"name": name, "discrete": discrete, "colors": colors})
        return self

    def write(self) -> dict[str, object]:
        """Write GPKG + TIFs + QMLs + loader script + manifest; returns the manifest dict."""
        root = Path(self.directory)
        root.mkdir(parents=True, exist_ok=True)
        gpkg = root / f"{self.name}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        manifest: dict[str, object] = {
            "name": self.name,
            "root": str(root.resolve()),
            "gpkg": gpkg.name,
            "vectors": [],
            "rasters": [],
        }
        for spec in self.vectors:
            layer = str(spec["layer"])
            gdf = self._vector_data[layer]
            write_vector(gdf, gpkg, layer=layer)
            style: str | None = None
            if spec.get("graduated"):
                colors = spec.get("colors")
                style_path = write_qml_graduated(
                    root / f"{layer}.qml",
                    gdf,
                    str(spec["graduated"]),
                    k=int(spec["k"]),
                    scheme=str(spec["scheme"]),
                    colors=list(colors) if isinstance(colors, list | tuple) else SEQUENTIAL_COLORS,
                )
                style = style_path.name
            elif spec.get("categorized"):
                colors = spec.get("colors")
                style_path = write_qml_categorized(
                    root / f"{layer}.qml",
                    gdf,
                    str(spec["categorized"]),
                    colors=colors if isinstance(colors, Mapping) else None,
                )
                style = style_path.name
            manifest["vectors"].append(
                {"name": layer, "path": gpkg.name, "layer": layer, "style": style}
            )  # type: ignore[union-attr]
        for spec in self.rasters:
            name = str(spec["name"])
            raster = self._raster_data[name]
            tif = write_raster(raster, root / f"{name}.tif")
            colors = spec.get("colors")
            qml = write_qml_raster(
                root / f"{name}.qml",
                raster,
                discrete=bool(spec.get("discrete")),
                **({"colors": list(colors)} if colors else {}),
            )
            manifest["rasters"].append({"name": name, "path": tif.name, "style": qml.name})  # type: ignore[union-attr]
        write_qgis_loader_script(root / "load_in_qgis.py", manifest)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        LOGGER.info(
            "QGIS bundle written -> %s (%d vectors, %d rasters)",
            root,
            len(self.vectors),
            len(self.rasters),
        )
        return manifest


def export_qgis_bundle(
    directory: str | Path,
    *,
    name: str,
    vectors: Mapping[str, gpd.GeoDataFrame],
    graduated: Mapping[str, str] | None = None,
    categorized: Mapping[str, str] | None = None,
    rasters: Mapping[str, RasterInfo] | None = None,
) -> dict[str, object]:
    """One-call convenience over :class:`QgisBundle`.

    ``graduated`` / ``categorized`` map layer name -> field to style by.
    """
    bundle = QgisBundle(Path(directory), name=name)
    for layer, gdf in vectors.items():
        bundle.add_vector(
            gdf,
            layer,
            graduated=(graduated or {}).get(layer),
            categorized=(categorized or {}).get(layer),
        )
    for rname, raster in (rasters or {}).items():
        bundle.add_raster(raster, rname)
    return bundle.write()
