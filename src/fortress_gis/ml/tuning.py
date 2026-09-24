"""Hyper-parameter search: randomized CV by default, Optuna (Bayesian/TPE) when installed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from fortress_gis.ml.pipeline import build_pipeline

# Search spaces keyed by registry name. Parameter names are prefixed with ``model__`` because
# the estimator sits at the ``model`` step of the pipeline.
DEFAULT_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "ridge": {"model__alpha": loguniform(1e-3, 1e2)},
    "lasso": {"model__alpha": loguniform(1e-4, 1e1)},
    "elastic_net": {"model__alpha": loguniform(1e-4, 1e1), "model__l1_ratio": uniform(0.05, 0.9)},
    "random_forest": {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(3, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": uniform(0.3, 0.7),
    },
    "gradient_boosting": {
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(1e-2, 3e-1),
        "model__max_depth": randint(2, 6),
        "model__subsample": uniform(0.6, 0.4),
    },
}


def tune_model(
    df: pd.DataFrame,
    *,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    model: str = "ridge",
    search_space: Mapping[str, Any] | None = None,
    n_iter: int = 25,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 0,
    backend: str = "random",
    n_jobs: int | None = -1,
) -> tuple[Pipeline, pd.DataFrame]:
    """Tune a registry model; returns ``(best_pipeline, cv_results_frame)``.

    ``backend="random"`` uses :class:`~sklearn.model_selection.RandomizedSearchCV`;
    ``backend="optuna"`` uses Optuna's TPE sampler over the same space (numeric distributions
    are sampled from their scipy ``rvs``) and requires the ``tuning`` extra. Candidate fits run
    on ``n_jobs`` processes (default all cores); ``n_iter * cv`` gradient boosting fits of a
    few hundred trees take minutes in serial.
    """
    data = df.dropna(subset=[target])
    X = data[[*numeric, *categorical]]
    y = pd.to_numeric(data[target], errors="coerce").to_numpy(dtype="float64")
    keep = np.isfinite(y)
    X, y = X.loc[keep], y[keep]
    space = dict(search_space or DEFAULT_SEARCH_SPACES.get(model, {}))
    pipe = build_pipeline(model, numeric=numeric, categorical=categorical)
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    if not space:
        # nothing to search (plain linear regression, say): fit once and report its CV score
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(pipe, X, y, cv=splitter, scoring=scoring)
        pipe.fit(X, y)
        frame = pd.DataFrame(
            {
                "params": [{}],
                "mean_test_score": [scores.mean()],
                "std_test_score": [scores.std()],
                "rank_test_score": [1],
            }
        )
        return pipe, frame
    if backend == "optuna":
        return _tune_optuna(
            pipe,
            space,
            X,
            y,
            n_iter=n_iter,
            splitter=splitter,
            scoring=scoring,
            random_state=random_state,
        )
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        cv=splitter,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    return search.best_estimator_, results


def _tune_optuna(
    pipe: Pipeline,
    space: Mapping[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_iter: int,
    splitter: KFold,
    scoring: str,
    random_state: int,
) -> tuple[Pipeline, pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        msg = "Install the 'tuning' extra (uv sync --extra tuning) for backend='optuna'"
        raise ImportError(msg) from exc
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(random_state)

    def objective(trial: Any) -> float:
        params: dict[str, Any] = {}
        for name, dist in space.items():
            if hasattr(dist, "rvs"):
                sample = dist.rvs(random_state=rng)
                params[name] = (
                    trial.suggest_float(name, float(sample) * 0.5, float(sample) * 1.5)
                    if isinstance(sample, float)
                    else trial.suggest_int(name, max(1, int(sample) // 2), int(sample) * 2)
                )
            elif isinstance(dist, list | tuple):
                params[name] = trial.suggest_categorical(name, list(dist))
            else:
                params[name] = dist
        est = clone(pipe).set_params(**params)
        return float(cross_val_score(est, X, y, cv=splitter, scoring=scoring).mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_iter)
    best = clone(pipe).set_params(**study.best_params).fit(X, y)
    return best, study.trials_dataframe()
