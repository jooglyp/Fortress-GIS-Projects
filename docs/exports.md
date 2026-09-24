# Export formats

Every domain's `export_artifacts(result, out_dir)` writes the same two kinds of output. Both
can also be produced directly from any GeoDataFrame or `RasterInfo`.

## QGIS bundle

```
exports/<domain>/qgis/
  <name>.gpkg            all vector layers, one GeoPackage layer each
  <raster>.tif           one GeoTIFF per raster layer
  <layer>.qml            QGIS style for that layer
  manifest.json          list of layers, paths and styles
  load_in_qgis.py        adds everything with styles when run in the QGIS Python console
```

```python
from fortress_gis.viz.qgis import QgisBundle

bundle = QgisBundle("exports/mine/qgis", name="mine")
bundle.add_vector(cells, "hex_risk", graduated="event_count", k=5, colors=("#ffffb2", "#bd0026"))
bundle.add_vector(hotspots, "hotspots", categorized="gi_class", colors=HOTSPOT_CLASS_COLORS)
bundle.add_raster(dem, "dem", colors=("#1a9850", "#fee08b", "#d73027"))
manifest = bundle.write()
```

Graduated styles use natural breaks (`scheme=` accepts any mapclassify scheme) computed from
the data at export time, so the QML is tied to that dataset. Categorized styles list every value in the colour map. Raster styles are
singleband pseudocolour ramps between the data minimum and maximum.

To load a bundle: QGIS, Plugins, Python Console, then

```python
exec(open("/path/to/exports/<domain>/qgis/load_in_qgis.py").read())
```

The loader reads `manifest.json` next to itself, so the folder can be moved as a unit.

## Kepler.gl

```
exports/<domain>/
  <name>_kepler.html          standalone map, opens in any browser
  <name>_kepler_config.json   the Kepler config, for reloading with new data
```

```python
from fortress_gis.viz.kepler import KeplerMapBuilder

builder = KeplerMapBuilder(title="Hex risk", height=600)
builder.add_layer(cells, "hex risk", color_field="event_count", opacity=0.55)
builder.add_layer(
    points, "positions", color_field="event_count", radius=3, time_field="DATETIME_UTC"
)
builder.widget()  # in Jupyter
builder.save_html("exports/mine/risk.html")  # anywhere
builder.save_config("exports/mine/risk_config.json")
```

Layers are reprojected to WGS84 and non-serialisable columns (timestamps, categoricals) are
converted before they reach Kepler. Datetime columns become ISO strings; a `time_field` adds a
time range filter over that column. Numeric colour fields use a quantile scale unless more
than half the values are identical, in which case the scale is quantize so the cells that
differ from the common value get their own colours. Categorical colour fields take a
`categorical_colors` mapping; `HOTSPOT_CLASS_COLORS` covers the Gi* and LISA classes.

The HTML uses Kepler.gl 3 with MapLibre and OpenStreetMap tiles. No token is needed. Files
with more than about 150k point features load slowly in a browser; the airspace export
samples positions to `max_points` for that reason.
