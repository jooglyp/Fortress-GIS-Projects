"""Map classification (natural breaks, quantiles, ...) -> ordinal classes such as risk levels."""

from __future__ import annotations

from collections.abc import Sequence

import mapclassify
import numpy as np
import pandas as pd

_SCHEMES = {
    "natural_breaks": mapclassify.NaturalBreaks,
    "quantiles": mapclassify.Quantiles,
    "equal_interval": mapclassify.EqualInterval,
    "fisher_jenks": mapclassify.FisherJenks,
    "std_mean": mapclassify.StdMean,
    "headtail": mapclassify.HeadTailBreaks,
}


def classify_breaks(
    values: pd.Series | np.ndarray,
    *,
    k: int = 3,
    scheme: str = "natural_breaks",
) -> tuple[np.ndarray, np.ndarray]:
    """Classify ``values`` into ``k`` ordinal bins; returns ``(bin_index, upper_bounds)``.

    NaNs get ``-1``. If there are fewer distinct values than ``k`` the class count is reduced so
    ``mapclassify`` does not fail on degenerate inputs (e.g. mostly-zero event counts).
    """
    arr = np.asarray(values, dtype="float64")
    valid = np.isfinite(arr)
    bins = np.full(arr.shape, -1, dtype="int64")
    if valid.sum() == 0:
        return bins, np.array([])
    distinct = np.unique(arr[valid]).size
    k_eff = int(max(1, min(k, distinct)))
    if k_eff == 1:
        bins[valid] = 0
        return bins, np.array([arr[valid].max()])
    if scheme not in _SCHEMES:
        msg = f"Unknown scheme {scheme!r}; choose from {sorted(_SCHEMES)}"
        raise ValueError(msg)
    classifier = (
        _SCHEMES[scheme](arr[valid], k=k_eff)
        if scheme != "headtail"
        else _SCHEMES[scheme](arr[valid])
    )
    bins[valid] = np.asarray(classifier.yb, dtype="int64")
    return bins, np.asarray(classifier.bins, dtype="float64")


def risk_levels(
    df: pd.DataFrame,
    column: str,
    *,
    k: int = 3,
    scheme: str = "natural_breaks",
    labels: Sequence[str] = ("low", "medium", "high"),
    out_column: str = "risk_level",
    label_column: str = "risk_label",
    zero_is_lowest: bool = True,
) -> pd.DataFrame:
    """Add an ordinal ``out_column`` (0..k-1) and a text ``label_column`` derived from ``column``.

    With ``zero_is_lowest`` rows whose value is exactly 0 are forced into class 0, so an empty
    cell never lands in "medium" when a few large values dominate the breaks.
    """
    bins, bounds = classify_breaks(df[column], k=k, scheme=scheme)
    out = df.copy()
    if zero_is_lowest:
        bins = np.where(
            pd.to_numeric(out[column], errors="coerce").fillna(0).to_numpy() == 0, 0, bins
        )
    out[out_column] = bins
    k_eff = int(bins.max()) + 1 if bins.size and bins.max() >= 0 else 0
    label_list = (
        list(labels)[:k_eff] if k_eff <= len(labels) else [f"class_{i}" for i in range(k_eff)]
    )
    mapping = dict(enumerate(label_list))
    mapping[-1] = "no data"
    out[label_column] = pd.Series(bins, index=out.index).map(mapping).astype(object)
    out.attrs[f"{out_column}_bounds"] = bounds.tolist()
    return out
