"""Model evaluation helpers producing tidy frames."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> pd.Series:
    """RMSE, MAE, R2, MAPE (ignoring zero targets) and bias."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    nonzero = yt != 0
    mape = (
        float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])))
        if nonzero.any()
        else np.nan
    )
    return pd.Series(
        {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
            "mape": mape,
            "bias": float(np.mean(yp - yt)),
            "n": len(yt),
        }
    )


def residual_frame(
    df: pd.DataFrame, *, target: str, prediction: np.ndarray | pd.Series, keep: Sequence[str] = ()
) -> pd.DataFrame:
    """Observed, predicted and residual columns, plus the ``keep`` columns (coordinates, say)."""
    out = df[[*keep]].copy()
    out["observed"] = pd.to_numeric(df[target], errors="coerce").to_numpy()
    out["predicted"] = np.asarray(prediction, dtype="float64")
    out["residual"] = out["observed"] - out["predicted"]
    return out


def permutation_importances(
    pipeline: object,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_repeats: int = 10,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 0,
) -> pd.DataFrame:
    """Permutation importance on the input columns, before any transform."""
    result = permutation_importance(
        pipeline,
        X,
        np.asarray(y, dtype="float64"),
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
    )
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)
