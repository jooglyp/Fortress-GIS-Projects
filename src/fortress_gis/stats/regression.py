"""OLS and elastic-net penalised OLS with statsmodels.

The maritime notebook fitted an unpenalised OLS for interpretable coefficients and then a
penalised fit started at the OLS solution. The same two steps are here, returning tidy frames
instead of printed summaries so results can be tabulated, plotted or written to disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RegressionResult:
    """Fitted statsmodels result plus the design used to obtain it."""

    model: Any
    result: Any
    target: str
    features: list[str]
    n_obs: int
    n_dropped: int
    kind: str = "ols"
    extra: dict[str, Any] = field(default_factory=dict)

    def coefficients(self) -> pd.DataFrame:
        params = self.result.params
        has_const = "const" in (params.index if isinstance(params, pd.Series) else [])
        return tidy_coefficients(
            self.result, self.features, with_const=has_const, inference=self.kind == "ols"
        )

    def metrics(self) -> pd.Series:
        r = self.result
        data = {
            "kind": self.kind,
            "n_obs": self.n_obs,
            "n_features": len(self.features),
            "r2": float(getattr(r, "rsquared", np.nan)),
            "r2_adj": float(getattr(r, "rsquared_adj", np.nan)),
            "aic": float(getattr(r, "aic", np.nan)),
            "bic": float(getattr(r, "bic", np.nan)),
        }
        data.update(self.extra)
        return pd.Series(data)

    def summary(self) -> str:
        return str(self.result.summary())


def _design(
    df: pd.DataFrame, target: str, features: Sequence[str], add_const: bool
) -> tuple[pd.DataFrame, pd.Series, int]:
    cols = [target, *features]
    clean = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_before = len(clean)
    clean = clean.dropna()
    X = clean[list(features)].astype("float64")
    X = X.loc[:, ~X.columns.duplicated()]
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    y = clean[target].astype("float64")
    return X, y, n_before - len(clean)


def fit_ols(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    add_const: bool = True,
    robust: str | None = "HC3",
) -> RegressionResult:
    """Ordinary least squares. Standard errors are HC3 unless ``robust=None``."""
    X, y, dropped = _design(df, target, features, add_const)
    model = sm.OLS(y, X)
    result = model.fit(cov_type=robust) if robust else model.fit()
    return RegressionResult(model, result, target, list(features), len(y), dropped, kind="ols")


def fit_regularized_ols(
    df: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    alpha: float = 0.01,
    l1_wt: float = 0.05,
    add_const: bool = True,
    refit: bool = False,
    standardize: bool = True,
) -> RegressionResult:
    """Elastic-net penalised OLS started from the OLS solution. ``l1_wt`` 0 is ridge, 1 is lasso.

    With ``standardize`` the features are z-scored first so the penalty treats them equally, and
    the coefficients come back on that scale (``extra["feature_scale"]`` and ``feature_mean``
    hold the transform). ``result`` is an ``OLSResults`` wrapping the penalised parameters, so
    ``.params`` and ``.predict`` work. :meth:`RegressionResult.coefficients` omits standard
    errors and p-values for these fits because they are not meaningful after shrinkage.
    """
    X, y, dropped = _design(df, target, features, add_const=False)
    scale = X.std(ddof=0).replace(0, 1.0) if standardize else pd.Series(1.0, index=X.columns)
    mean = X.mean() if standardize else pd.Series(0.0, index=X.columns)
    Xs = (X - mean) / scale
    if add_const:
        Xs = sm.add_constant(Xs, has_constant="add")
    model = sm.OLS(y, Xs)
    ols = model.fit()
    # A numpy array, not a Series: statsmodels' coordinate descent does ``params[k] = 0`` with a
    # positional k, which pandas 3 treats as a new label on a string-indexed Series.
    reg = model.fit_regularized(
        alpha=alpha, L1_wt=l1_wt, start_params=np.asarray(ols.params, dtype="float64"), refit=refit
    )
    params = pd.Series(np.asarray(reg.params, dtype="float64"), index=Xs.columns)
    wrapped = sm.regression.linear_model.OLSResults(model, params, model.normalized_cov_params)
    n_nonzero = int((np.abs(np.asarray(reg.params)) > 1e-12).sum())
    kind = "ridge" if l1_wt == 0 else ("lasso" if l1_wt == 1 else "elastic_net")
    return RegressionResult(
        model,
        wrapped,
        target,
        list(features),
        len(y),
        dropped,
        kind=kind,
        extra={
            "alpha": alpha,
            "l1_wt": l1_wt,
            "n_nonzero": n_nonzero,
            "feature_scale": scale.to_dict(),
            "feature_mean": mean.to_dict(),
        },
    )


def tidy_coefficients(
    result: Any,
    features: Sequence[str] | None = None,
    *,
    with_const: bool = True,
    inference: bool = True,
) -> pd.DataFrame:
    """Coefficient table with ``term, estimate`` and, when ``inference``, the HC/OLS
    ``std_error, statistic, p_value, ci_low, ci_high`` columns.

    Pass ``inference=False`` for penalised fits: statsmodels will happily compute standard
    errors from the unpenalised covariance, and they mean nothing for a shrunk estimate.
    """
    params = result.params
    if isinstance(params, pd.Series):
        terms = list(params.index)
    else:
        terms = (
            ["const", *features]
            if features is not None and len(params) == len(features) + 1
            else list(features or range(len(params)))
        )
    frame = pd.DataFrame({"term": terms, "estimate": np.asarray(params, dtype="float64")})
    if inference:
        frame["std_error"] = np.asarray(result.bse)
        frame["statistic"] = np.asarray(result.tvalues)
        frame["p_value"] = np.asarray(result.pvalues)
        ci = np.asarray(result.conf_int())
        frame["ci_low"] = ci[:, 0]
        frame["ci_high"] = ci[:, 1]
    if not with_const:
        frame = frame[frame["term"] != "const"]
    return frame.reset_index(drop=True)
