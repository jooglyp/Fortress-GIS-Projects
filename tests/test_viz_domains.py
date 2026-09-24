import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fortress_gis.domains import airspace as air
from fortress_gis.domains import hydrology as hy
from fortress_gis.domains import maritime as mar
from fortress_gis.domains import oceanography as oc
from fortress_gis.features.proximity import ProximityConfig
from fortress_gis.grids.hexgrid import hex_grid
from fortress_gis.viz.kepler import KeplerMapBuilder, kepler_available, prepare_for_kepler
from fortress_gis.viz.qgis import QgisBundle


def _hex_layer(n_values: int = 3):
    cells = hex_grid((0, 0, 5000, 5000), radius=500, crs="EPSG:32615").to_crs("EPSG:4326")
    cells["count"] = np.arange(len(cells)) % n_values
    cells["label"] = np.where(cells["count"] > 0, "hot", "cold")
    cells["when"] = pd.date_range("2023-01-01", periods=len(cells), freq="min", tz="UTC")
    cells["flag"] = cells["count"] > 1
    return cells


def test_prepare_for_kepler_makes_json_safe_types():
    prepared = prepare_for_kepler(_hex_layer())
    assert prepared.crs.to_epsg() == 4326
    assert prepared["count"].dtype == "float64"
    assert prepared["flag"].dtype == "float64"
    assert prepared["when"].dtype == object
    assert prepared["when"].iloc[0].startswith("2023-01-01T00:00:00")


def test_kepler_config_has_layers_filter_and_switches_scale(tmp_path):
    cells = _hex_layer()
    skewed = cells.copy()
    skewed["count"] = 0.0
    skewed.loc[skewed.index[:3], "count"] = [5.0, 7.0, 9.0]
    b = KeplerMapBuilder(title="t")
    b.add_layer(cells, "cells", color_field="count", time_field="when")
    b.add_layer(
        cells,
        "classes",
        color_field="label",
        categorical_colors={"hot": "#ff0000", "cold": "#0000ff"},
        visible=False,
    )
    b.add_layer(skewed, "skewed", color_field="count")
    cfg = b.config()
    layers = cfg["config"]["visState"]["layers"]
    assert [layer["config"]["label"] for layer in layers] == ["cells", "classes", "skewed"]
    assert layers[0]["visualChannels"]["colorField"]["name"] == "count"
    assert layers[1]["visualChannels"]["colorScale"] == "ordinal"
    assert layers[2]["visualChannels"]["colorScale"] == "quantize"
    assert cfg["config"]["visState"]["filters"][0]["name"] == ["when"]
    out = b.save_config(tmp_path / "cfg.json")
    assert json.loads(out.read_text())["version"] == "v1"


@pytest.mark.skipif(not kepler_available(), reason="keplergl not installed")
def test_kepler_html_export(tmp_path):
    b = KeplerMapBuilder(title="t").add_layer(_hex_layer(), "cells", color_field="count")
    html = b.save_html(tmp_path / "map.html")
    text = html.read_text(encoding="utf-8")
    assert html.stat().st_size > 1_000_000
    assert '"label": "cells"' in text or "cells" in text


def test_qgis_bundle_writes_gpkg_qml_manifest_and_script(tmp_path, dem):
    cells = _hex_layer()
    bundle = QgisBundle(tmp_path, name="t")
    bundle.add_vector(cells, "cells", graduated="count")
    bundle.add_vector(cells, "classes", categorized="label")
    bundle.add_raster(dem, "dem")
    manifest = bundle.write()
    assert (tmp_path / "t.gpkg").exists()
    with sqlite3.connect(tmp_path / "t.gpkg") as con:
        names = {r[0] for r in con.execute("select table_name from gpkg_contents")}
    assert {"cells", "classes"} <= names
    for v in manifest["vectors"]:
        qml = (tmp_path / v["style"]).read_text()
        assert "<renderer-v2" in qml
    assert (tmp_path / "dem.tif").exists()
    assert "singlebandpseudocolor" in (tmp_path / "dem.qml").read_text()
    script = (tmp_path / "load_in_qgis.py").read_text()
    assert "QgsVectorLayer" in script and "loadNamedStyle" in script
    assert json.loads((tmp_path / "manifest.json").read_text())["name"] == "t"


def test_hydrology_pipeline(dem_tiles, tmp_path):
    raster = hy.load_dem(dem_tiles)
    xs, ys = raster.cell_centers()
    r = int(np.nanargmin(raster.masked()[:, 0]))
    result = hy.run_watershed_pipeline(
        raster, (float(xs[r, 0]), float(ys[r, 0])), n_synthetic_flowlines=4
    )
    assert result.stats["catchment_cells"] > 0.2 * result.stats["dem_cells"]
    assert result.flowlines is not None and {"elevation_mean", "slope_m_per_km"} <= set(
        result.flowlines.columns
    )
    paths = hy.export_artifacts(result, tmp_path, kepler_html=False)
    assert (paths["qgis"] / "manifest.json").exists()


def test_oceanography_pipeline(chlorophyll_files, tmp_path):
    pts = oc.load_composites(chlorophyll_files)
    assert sum(c.startswith("chlor_") for c in pts.columns) == len(chlorophyll_files)
    stack = oc.build_hex_covariates(pts, hex_radius_m=250_000)
    assert {"winter_mean", "summer_mean", "total_mean", "winter_var", "2018_var"} <= set(
        stack.covariate_columns
    )
    res = oc.screen_hotspots(stack.covariates, ["winter_mean"], permutations=49)
    assert res["winter_mean"].class_column == "winter_mean_gi_class"
    merged = oc.merge_hotspot_layers(res)
    assert len(merged) == len(stack.covariates)
    paths = oc.export_artifacts(stack, res, tmp_path, kepler_html=False)
    assert (paths["qgis"] / "manifest.json").exists()


def test_airspace_pipeline_eager_and_dask_agree(aircraft, tmp_path):
    flt = air.AirspaceFilter(hours=(0, 24), altitude_ft=(0, 500))
    cfg = ProximityConfig(vertical=200)
    eager = air.run_airspace_pipeline(aircraft, flt=flt, config=cfg, minutes_per_hex=3)
    assert eager.stats["events_total"] > 0
    assert {"risk_label", "excess_p", "rate_exceeds_tolerance"} <= set(eager.cells.columns)
    parquet = tmp_path / "pos.parquet"
    aircraft.to_parquet(parquet)
    lazy = air.run_airspace_pipeline(
        parquet, flt=flt, config=cfg, minutes_per_hex=3, use_dask=True, window="20min"
    )
    assert lazy.stats["events_total"] == eager.stats["events_total"]
    paths = air.export_artifacts(eager, tmp_path, kepler_html=False)
    assert pd.read_csv(paths["hourly_csv"])["events"].sum() == eager.stats["events_total"]


def test_airspace_filter_on_dask_frame(aircraft, tmp_path):
    parquet = tmp_path / "pos.parquet"
    aircraft.to_parquet(parquet)
    lazy = air.load_positions(parquet, as_dask=True)
    filtered = air.filter_positions(lazy, air.AirspaceFilter(hours=(9, 10), altitude_ft=(0, 300)))
    got = filtered.compute()
    expected = aircraft[
        (pd.to_datetime(aircraft["DATETIME_UTC"], utc=True).dt.hour == 9)
        & (aircraft["ALTITUDE_AGL_FT"] <= 300)
    ]
    assert len(got) == len(expected)


def test_maritime_pipeline(telemetry, tmp_path):
    result = mar.run_maritime_pipeline(
        telemetry, target="fuel_demand_kg_h", models=["linear", "ridge"], cv=3, n_iter=3
    )
    assert result.stats["holdout_r2"] > 0.5
    assert result.best_model in {"linear", "ridge"}
    assert "speed_through_water_kn" in result.design.features
    paths = mar.export_artifacts(result, telemetry, tmp_path, kepler_html=False)
    assert (paths["qgis"] / "manifest.json").exists()
    assert (tmp_path / "maritime_leaderboard.csv").exists()
