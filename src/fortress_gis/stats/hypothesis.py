"""Count-data hypothesis tests used for cell-level risk screening.

The airspace workflow asks two questions per hexagon: is the number of proximity events higher
than a no-risk baseline would produce, and does risk depend on the hour of day.
:func:`poisson_excess_test` answers the first with an upper-tail exact Poisson test against a
global or exposure-weighted expectation. The original notebook tested
``poisson.cdf(observed, expected) < alpha``, which flags cells with fewer events than expected;
``sf`` is the right tail. :func:`poisson_rate_test` compares each cell's rate to a tolerated
maximum such as one event per 10,000 hours. :func:`chi_square_independence` answers the second
question and returns standardised residuals so the cells driving the dependence can be read
off. :func:`benjamini_hochberg` controls the false discovery rate across cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def poisson_excess_test(
    counts: pd.Series | np.ndarray,
    *,
    expected: float | pd.Series | np.ndarray | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> pd.DataFrame:
    """Upper-tail exact Poisson test ``P(X >= observed | expected)`` per observation.

    ``expected`` defaults to the mean of ``counts`` (a homogeneous no-risk baseline). Returns a
    frame with ``expected``, ``p_value``, ``p_adjusted`` (Benjamini-Hochberg when ``fdr``) and
    ``significant``.
    """
    obs = np.asarray(counts, dtype="float64")
    exp = (
        np.full(obs.shape, float(np.nanmean(obs)))
        if expected is None
        else np.broadcast_to(np.asarray(expected, dtype="float64"), obs.shape)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(exp > 0, sps.poisson.sf(obs - 1, exp), np.where(obs > 0, 0.0, 1.0))
    p = np.clip(p, 0.0, 1.0)
    p_adj = benjamini_hochberg(p)["p_adjusted"].to_numpy() if fdr else p
    index = counts.index if isinstance(counts, pd.Series) else None
    return pd.DataFrame(
        {
            "observed": obs,
            "expected": exp,
            "p_value": p,
            "p_adjusted": p_adj,
            "significant": p_adj < alpha,
        },
        index=index,
    )


def poisson_rate_test(
    counts: pd.Series | np.ndarray,
    exposure: pd.Series | np.ndarray,
    *,
    max_rate: float,
    alpha: float = 0.05,
    fdr: bool = True,
) -> pd.DataFrame:
    """Test ``H0: rate <= max_rate`` per observation given ``exposure`` (same time units as rate).

    Expected counts under the tolerated rate are ``max_rate * exposure``; the p-value is the
    probability of observing at least ``counts`` events under that Poisson law.
    """
    exp = max_rate * np.asarray(exposure, dtype="float64")
    out = poisson_excess_test(counts, expected=exp, alpha=alpha, fdr=fdr)
    out["exposure"] = np.asarray(exposure, dtype="float64")
    out["observed_rate"] = out["observed"] / np.where(out["exposure"] > 0, out["exposure"], np.nan)
    return out


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    dof: int
    observed: pd.DataFrame
    expected: pd.DataFrame
    standardized_residuals: pd.DataFrame

    @property
    def significant(self) -> bool:
        return bool(self.p_value < 0.05)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"statistic": [self.statistic], "p_value": [self.p_value], "dof": [self.dof]}
        )


def chi_square_independence(
    df: pd.DataFrame,
    *,
    row: str,
    col: str,
    values: str | None = None,
    min_expected: float = 5.0,
) -> ChiSquareResult:
    """Chi-square test of independence between two categorical columns.

    Builds the contingency table (counts, or the sum of ``values``), runs
    ``scipy.stats.chi2_contingency`` and returns Pearson standardised residuals
    ``(O - E) / sqrt(E)`` so individual cells can be inspected. A warning is logged when any
    expected count is below ``min_expected`` (the asymptotic approximation weakens).
    """
    if values is None:
        table = pd.crosstab(df[row], df[col])
    else:
        table = df.pivot_table(index=row, columns=col, values=values, aggfunc="sum", fill_value=0)
    table = table.loc[(table.sum(axis=1) > 0), (table.sum(axis=0) > 0)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        msg = f"Contingency table needs at least 2x2 non-empty cells, got {table.shape}"
        raise ValueError(msg)
    stat, p, dof, expected = sps.chi2_contingency(table.to_numpy())
    exp_df = pd.DataFrame(expected, index=table.index, columns=table.columns)
    if (exp_df < min_expected).any().any():
        from fortress_gis.log import get_logger

        get_logger(__name__).warning(
            "chi_square_independence: some expected counts < %.0f; interpret with care",
            min_expected,
        )
    resid = (table - exp_df) / np.sqrt(exp_df)
    return ChiSquareResult(float(stat), float(p), int(dof), table, exp_df, resid)


def benjamini_hochberg(p_values: pd.Series | np.ndarray, *, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg adjustment; returns ``p_value``, ``p_adjusted`` and ``significant``."""
    p = np.asarray(p_values, dtype="float64")
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    index = p_values.index if isinstance(p_values, pd.Series) else None
    return pd.DataFrame(
        {"p_value": p, "p_adjusted": adjusted, "significant": adjusted < alpha}, index=index
    )
