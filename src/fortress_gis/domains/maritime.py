"""Maritime workflow: vessel telemetry to an inference model and a predictive model of a target.

The telemetry is one row per logger tick with position, speeds, headings, drafts and weather,
and a target such as fuel demand. Columns are typed, a day/night flag is added, categoricals
are one-hot encoded, and two kinds of model are fitted: statsmodels OLS (plain and penalised)
for coefficient inference, and a scikit-learn model comparison with tuning for prediction. The
voyage track and per-segment residuals are exported for Kepler.gl and QGIS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from fortress_gis.features.encoding import (
    ColumnTypes,
    classify_columns,
    one_hot_encode,
    select_model_columns,
)
from fortress_gis.features.temporal import add_hour_and_day_night
from fortress_gis.io.vector import points_from_xy
from fortress_gis.log import get_logger
from fortress_gis.ml.evaluate import permutation_importances, regression_metrics, residual_frame
from fortress_gis.ml.pipeline import compare_models
from fortress_gis.ml.tuning import tune_model
from fortress_gis.stats.regression import RegressionResult, fit_ols, fit_regularized_ols
from fortress_gis.viz.kepler import KeplerMapBuilder
from fortress_gis.viz.qgis import QgisBundle

LOGGER = get_logger(__name__)

DEFAULT_EXCLUDE = ("timestamp", "longitude", "latitude", "hour")


@dataclass
class TelemetryDesign:
    """Prepared modelling frame plus the column roles used to build it."""

    frame: pd.DataFrame
    types: ColumnTypes
    target: str
    numeric: list[str]
    categorical: list[str]
    dummy_map: dict[str, list[str]]

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *[d for ds in self.dummy_map.values() for d in ds]]


@dataclass
class MaritimeResult:
    """Fitted models and their evaluation tables."""

    design: TelemetryDesign
    ols: RegressionResult
    penalised: RegressionResult
    leaderboard: pd.DataFrame
    best_model: str
    tuned: object
    tuning_results: pd.DataFrame
    holdout_metrics: pd.Series
    importances: pd.DataFrame
    residuals: gpd.GeoDataFrame
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.stats)


def prepare_design(
    df: pd.DataFrame,
    *,
    target: str,
    time_column: str = "timestamp",
    time_unit: str | None = "s",
    exclude: Sequence[str] = DEFAULT_EXCLUDE,
    category_limit: int = 20,
) -> TelemetryDesign:
    """Type the columns, add ``hour`` and ``is_night``, one-hot encode categoricals.

    Boolean columns are cast to 0/1 and treated as numeric. Identifier, text and datetime
    columns are dropped from the design along with ``exclude``.
    """
    frame = add_hour_and_day_night(df, time_column=time_column, unit=time_unit)
    types = classify_columns(frame, target=target, category_limit=category_limit)
    for col in types.boolean:
        frame[col] = frame[col].astype("float64")
    categorical = [c for c in types.categorical if c not in exclude]
    encoded, dummy_map = one_hot_encode(frame, categorical, category_limit=category_limit)
    numeric = [
        c
        for c in select_model_columns(frame, types, exclude=exclude)
        if c not in categorical and c != target
    ]
    numeric = [c for c in numeric if c in frame.columns and frame[c].nunique(dropna=True) > 1]
    dummies = [d for ds in dummy_map.values() for d in ds]
    design_frame = encoded[[target, *numeric, *dummies]].copy()
    LOGGER.info(
        "Design: %d rows, %d numeric, %d dummy columns",
        len(design_frame),
        len(numeric),
        sum(len(v) for v in dummy_map.values()),
    )
    return TelemetryDesign(design_frame, types, target, numeric, categorical, dummy_map)


def fit_inference_models(
    design: TelemetryDesign,
    *,
    alpha: float = 0.05,
    l1_wt: float = 0.05,
) -> tuple[RegressionResult, RegressionResult]:
    """OLS with HC3 errors, then an elastic-net fit started from the OLS solution."""
    ols = fit_ols(design.frame, target=design.target, features=design.features)
    pen = fit_regularized_ols(
        design.frame, target=design.target, features=design.features, alpha=alpha, l1_wt=l1_wt
    )
    return ols, pen


def fit_predictive_models(
    design: TelemetryDesign,
    *,
    models: Sequence[str] | None = None,
    cv: int = 5,
    n_iter: int = 20,
    holdout_fraction: float = 0.2,
    random_state: int = 0,
    tune_backend: str = "random",
) -> tuple[pd.DataFrame, str, object, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Compare registry models by CV, tune the best, score it on a time-ordered holdout.

    The holdout is the last ``holdout_fraction`` of rows in frame order, which for telemetry
    means the end of the voyage; a random split would leak neighbouring ticks.
    Returns ``(leaderboard, best_name, tuned_pipeline, tuning_results, holdout_metrics,
    importances, holdout_predictions)``.
    """
    frame = design.frame.dropna(subset=[design.target])
    n_hold = max(int(len(frame) * holdout_fraction), 10)
    train, hold = frame.iloc[:-n_hold], frame.iloc[-n_hold:]
    board = compare_models(
        train,
        target=design.target,
        numeric=design.features,
        models=models,
        cv=cv,
        random_state=random_state,
    )
    best = str(board.iloc[0]["model"])
    LOGGER.info("Best CV model: %s (rmse_cv=%.3f)", best, board.iloc[0]["rmse_cv"])
    tuned, tuning = tune_model(
        train,
        target=design.target,
        numeric=design.features,
        model=best,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        backend=tune_backend,
    )
    X_hold, y_hold = hold[design.features], hold[design.target]
    pred = pd.Series(tuned.predict(X_hold), index=hold.index, name="prediction")
    metrics = regression_metrics(y_hold, pred)
    importances = permutation_importances(
        tuned, X_hold, y_hold, n_repeats=5, random_state=random_state
    )
    return board, best, tuned, tuning, metrics, importances, pred.to_frame()


def run_maritime_pipeline(
    df: pd.DataFrame,
    *,
    target: str,
    lon: str = "longitude",
    lat: str = "latitude",
    time_column: str = "timestamp",
    time_unit: str | None = "s",
    models: Sequence[str] | None = None,
    cv: int = 5,
    n_iter: int = 20,
    tune_backend: str = "random",
) -> MaritimeResult:
    """Prepare the design, fit both model families, and geolocate the holdout residuals."""
    design = prepare_design(df, target=target, time_column=time_column, time_unit=time_unit)
    ols, pen = fit_inference_models(design)
    board, best, tuned, tuning, metrics, imps, pred = fit_predictive_models(
        design, models=models, cv=cv, n_iter=n_iter, tune_backend=tune_backend
    )
    joined = df.loc[pred.index, [lon, lat, time_column, target]].join(pred)
    joined = residual_frame(
        joined, target=target, prediction=joined["prediction"], keep=[lon, lat, time_column]
    )
    residuals = points_from_xy(joined, lon=lon, lat=lat)
    stats = {
        "rows": float(len(design.frame)),
        "features": float(len(design.features)),
        "ols_r2_adj": float(ols.metrics()["r2_adj"]),
        "penalised_nonzero": float(pen.metrics()["n_nonzero"]),
        "best_model_rmse_cv": float(board.iloc[0]["rmse_cv"]),
        "holdout_rmse": float(metrics["rmse"]),
        "holdout_r2": float(metrics["r2"]),
    }
    return MaritimeResult(
        design, ols, pen, board, best, tuned, tuning, metrics, imps, residuals, stats
    )


def voyage_track(
    df: pd.DataFrame,
    *,
    lon: str = "longitude",
    lat: str = "latitude",
    time_column: str = "timestamp",
    time_unit: str | None = "s",
    value_columns: Sequence[str] = (),
) -> gpd.GeoDataFrame:
    """Position points in time order with the requested value columns, for mapping."""
    cols = [lon, lat, time_column, *value_columns]
    frame = df[cols].copy()
    if time_unit and pd.api.types.is_numeric_dtype(frame[time_column]):
        frame[time_column] = pd.to_datetime(frame[time_column], unit=time_unit, utc=True)
    return points_from_xy(frame.sort_values(time_column), lon=lon, lat=lat)


def export_artifacts(
    result: MaritimeResult,
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    name: str = "maritime",
    kepler_html: bool = True,
    time_column: str = "timestamp",
    time_unit: str | None = "s",
    max_points: int = 60_000,
) -> dict[str, Path]:
    """Write coefficient/leaderboard CSVs, a QGIS bundle and a Kepler map of the voyage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.ols.coefficients().to_csv(out / f"{name}_ols_coefficients.csv", index=False)
    result.penalised.coefficients().to_csv(out / f"{name}_penalised_coefficients.csv", index=False)
    result.leaderboard.to_csv(out / f"{name}_leaderboard.csv", index=False)
    result.importances.to_csv(out / f"{name}_importances.csv", index=False)
    target = result.design.target
    track = voyage_track(df, time_column=time_column, time_unit=time_unit, value_columns=[target])
    if len(track) > max_points:
        track = track.iloc[:: int(np.ceil(len(track) / max_points))]
    resid = result.residuals.copy()
    if pd.api.types.is_numeric_dtype(resid[time_column]) and time_unit:
        resid[time_column] = pd.to_datetime(resid[time_column], unit=time_unit, utc=True)
    bundle = QgisBundle(out / "qgis", name=name)
    bundle.add_vector(track, "voyage_track", graduated=target, k=5)
    bundle.add_vector(resid, "holdout_residuals", graduated="residual", k=5)
    bundle.write()
    paths = {"qgis": out / "qgis"}
    if kepler_html:
        builder = KeplerMapBuilder(title=f"{name} - voyage", map_style="dark")
        builder.add_layer(
            track,
            target,
            color_field=target,
            color_scale="quantile",
            radius=4,
            time_field=time_column,
        )
        builder.add_layer(
            resid,
            "holdout residuals",
            color_field="residual",
            color_scale="quantile",
            radius=6,
            visible=False,
        )
        paths["kepler_html"] = builder.save_html(out / f"{name}_kepler.html")
        paths["kepler_config"] = builder.save_config(out / f"{name}_kepler_config.json")
    return paths
