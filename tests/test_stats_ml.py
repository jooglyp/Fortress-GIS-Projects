import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from fortress_gis.features.temporal import add_hour_and_day_night
from fortress_gis.grids.hexgrid import hex_grid
from fortress_gis.ml.pipeline import REGRESSORS, compare_models
from fortress_gis.ml.tuning import tune_model
from fortress_gis.stats.classify import risk_levels
from fortress_gis.stats.hypothesis import (
    benjamini_hochberg,
    chi_square_independence,
    poisson_excess_test,
    poisson_rate_test,
)
from fortress_gis.stats.regression import fit_ols, fit_regularized_ols
from fortress_gis.stats.spatial_autocorr import (
    build_weights,
    getis_ord_hotspots,
    local_moran_clusters,
)


def test_poisson_excess_is_upper_tail():
    counts = pd.Series([0, 1, 1, 2, 20])
    res = poisson_excess_test(counts, fdr=False)
    expected = counts.mean()
    assert res.loc[4, "p_value"] == pytest.approx(poisson.sf(19, expected))
    assert res.loc[4, "significant"]
    assert not res.loc[0, "significant"]  # zero counts are never an excess


def test_poisson_rate_test_against_tolerance():
    counts = pd.Series([0, 3])
    exposure = pd.Series([10_000.0, 10_000.0])
    res = poisson_rate_test(counts, exposure, max_rate=1e-4, fdr=False)
    assert res["expected"].tolist() == [1.0, 1.0]
    assert res.loc[1, "p_value"] == pytest.approx(poisson.sf(2, 1.0))


def test_benjamini_hochberg_monotone():
    p = pd.Series([0.001, 0.01, 0.02, 0.5, 0.9])
    adj = benjamini_hochberg(p, alpha=0.05)
    assert (adj["p_adjusted"] >= p).all()
    assert adj["p_adjusted"].is_monotonic_increasing
    assert adj["significant"].tolist() == [True, True, True, False, False]


def test_chi_square_detects_dependence():
    rng = np.random.default_rng(1)
    hour = rng.integers(0, 24, 4000)
    label = np.where(
        hour < 12,
        rng.choice(["low", "high"], 4000, p=[0.9, 0.1]),
        rng.choice(["low", "high"], 4000, p=[0.5, 0.5]),
    )
    res = chi_square_independence(pd.DataFrame({"h": hour, "c": label}), row="c", col="h")
    assert res.p_value < 1e-6
    assert res.standardized_residuals.shape == (2, 24)


def test_risk_levels_zero_is_lowest():
    df = pd.DataFrame({"n": [0, 0, 0, 0, 1, 2, 3, 40, 50]})
    out = risk_levels(df, "n", k=3)
    assert (out.loc[out["n"] == 0, "risk_level"] == 0).all()
    assert out.loc[out["n"] == 50, "risk_label"].iloc[0] == "high"
    assert set(out["risk_label"]) <= {"low", "medium", "high"}


def test_hotspots_recover_planted_cluster():
    cells = hex_grid((0, 0, 20_000, 20_000), radius=1000, crs="EPSG:32615")
    cents = cells.geometry.centroid
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, len(cells))
    hot = (cents.x - 15_000) ** 2 + (cents.y - 15_000) ** 2 < 3500**2
    values[hot.to_numpy()] += 6
    cells["v"] = values
    w = build_weights(cells, kind="queen")
    gi = getis_ord_hotspots(cells, "v", weights=w, permutations=99)
    classes = gi.layer["v_gi_class"]
    assert classes[hot.to_numpy()].str.startswith("hot").mean() > 0.8
    assert classes[~hot.to_numpy()].str.startswith("hot").mean() < 0.15
    lisa = local_moran_clusters(cells, "v", weights=w, permutations=99)
    assert (lisa.layer.loc[hot.to_numpy(), "v_lisa_class"] == "high-high").mean() > 0.6


def test_ols_and_regularized_agree_on_strong_signal(telemetry):
    df = add_hour_and_day_night(telemetry, time_column="timestamp", unit="s")
    df["is_night"] = df["is_night"].astype(float)
    feats = [
        "speed_through_water_kn",
        "wind_speed_kn",
        "wave_height_m",
        "rudder_angle_deg",
        "is_night",
    ]
    ols = fit_ols(df, target="fuel_demand_kg_h", features=feats)
    coef = ols.coefficients().set_index("term")
    assert coef.loc["speed_through_water_kn", "p_value"] < 1e-6
    assert ols.metrics()["r2"] > 0.7
    pen = fit_regularized_ols(df, target="fuel_demand_kg_h", features=feats, alpha=0.01, l1_wt=0.5)
    pc = pen.coefficients().set_index("term")
    assert "p_value" not in pc.columns
    assert pc.drop(index="const")["estimate"].abs().idxmax() == "speed_through_water_kn"


@pytest.mark.parametrize("models", [["linear", "ridge"], ["random_forest"]])
def test_compare_and_tune(telemetry, models):
    feats = ["speed_through_water_kn", "wind_speed_kn", "wave_height_m"]
    board = compare_models(telemetry, target="fuel_demand_kg_h", numeric=feats, models=models, cv=3)
    assert set(board["model"]) == set(models)
    assert board["rmse_cv"].is_monotonic_increasing
    assert set(REGRESSORS) >= set(models)
    best, results = tune_model(
        telemetry, target="fuel_demand_kg_h", numeric=feats, model=models[-1], n_iter=3, cv=3
    )
    assert len(results) == (1 if models[-1] == "linear" else 3)
    assert "mean_test_score" in results.columns
    assert best.predict(telemetry[feats].head(5)).shape == (5,)
