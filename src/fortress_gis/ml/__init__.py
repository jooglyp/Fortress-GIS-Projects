"""scikit-learn pipelines, model comparison, tuning and evaluation."""

from fortress_gis.ml.evaluate import (
    permutation_importances,
    regression_metrics,
    residual_frame,
)
from fortress_gis.ml.pipeline import (
    REGRESSORS,
    build_pipeline,
    compare_models,
    make_preprocessor,
)
from fortress_gis.ml.tuning import tune_model

__all__ = [
    "REGRESSORS",
    "build_pipeline",
    "compare_models",
    "make_preprocessor",
    "permutation_importances",
    "regression_metrics",
    "residual_frame",
    "tune_model",
]
