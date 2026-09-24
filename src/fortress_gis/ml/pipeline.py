"""scikit-learn pipelines and a small registry of regressors.

The maritime notebook depended on an AutoML package. Here the pipeline is explicit: impute,
scale, one-hot encode, estimator, scored with K-fold cross-validation for each regressor in
:data:`REGRESSORS`. Each result is a plain sklearn ``Pipeline`` that can be pickled, tuned with
:mod:`fortress_gis.ml.tuning` and inspected with :mod:`fortress_gis.ml.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REGRESSORS: dict[str, Callable[[], BaseEstimator]] = {
    "linear": LinearRegression,
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.01, max_iter=20_000),
    "elastic_net": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20_000),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=0),
}
"""Registry of estimator factories; extend with ``REGRESSORS["xgb"] = lambda: XGBRegressor()``."""

DEFAULT_SCORING: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def make_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    *,
    scale: bool = True,
    numeric_impute: str = "median",
) -> ColumnTransformer:
    """Impute + scale numeric columns and one-hot encode categoricals (unknowns ignored)."""
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy=numeric_impute))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = [("num", Pipeline(num_steps), list(numeric))]
    if categorical:
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat, list(categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(
    estimator: str | BaseEstimator,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    scale: bool = True,
) -> Pipeline:
    """Preprocessor plus estimator as one ``Pipeline``, from a registry name or an estimator."""
    model = REGRESSORS[estimator]() if isinstance(estimator, str) else estimator
    return Pipeline(
        [("prep", make_preprocessor(numeric, categorical, scale=scale)), ("model", model)]
    )


def compare_models(
    df: pd.DataFrame,
    *,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    models: Sequence[str] | Mapping[str, BaseEstimator] | None = None,
    cv: int = 5,
    scoring: Mapping[str, str] | None = None,
    random_state: int = 0,
    return_estimators: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Cross-validate several pipelines and return a leaderboard sorted by RMSE.

    Rows with a missing target are dropped; predictors are imputed inside the pipeline so
    every fold sees the same preprocessing. ``models`` defaults to the whole registry.
    """
    scoring = dict(scoring or DEFAULT_SCORING)
    data = df.dropna(subset=[target])
    X = data[[*numeric, *categorical]]
    y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype="float64")
    keep = np.isfinite(y)
    X, y = X.loc[keep], y[keep]
    if models is None:
        candidates: dict[str, BaseEstimator] = {
            name: factory() for name, factory in REGRESSORS.items()
        }
    elif isinstance(models, Mapping):
        candidates = dict(models)
    else:
        candidates = {name: REGRESSORS[name]() for name in models}
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    fitted: dict[str, Pipeline] = {}
    for name, est in candidates.items():
        pipe = build_pipeline(est, numeric=numeric, categorical=categorical)
        res = cross_validate(
            pipe, X, y, cv=splitter, scoring=scoring, return_train_score=True, n_jobs=None
        )
        row: dict[str, Any] = {"model": name, "fit_time_s": float(np.mean(res["fit_time"]))}
        for key in scoring:
            test = np.asarray(res[f"test_{key}"])
            train = np.asarray(res[f"train_{key}"])
            sign = -1.0 if scoring[key].startswith("neg_") else 1.0
            row[f"{key}_cv"] = float(sign * test.mean())
            row[f"{key}_cv_std"] = float(test.std())
            row[f"{key}_train"] = float(sign * train.mean())
        rows.append(row)
        if return_estimators:
            fitted[name] = pipe.fit(X, y)
    board = pd.DataFrame(rows)
    sort_key = "rmse_cv" if "rmse_cv" in board.columns else board.columns[2]
    board = board.sort_values(sort_key, ascending=True).reset_index(drop=True)
    return (board, fitted) if return_estimators else board
