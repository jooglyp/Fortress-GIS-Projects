"""Column typing and one-hot encoding for tabular models.

The maritime notebook had a 300-line ``VariableTypeDetector``. The same decisions (drop constant
or mostly-null columns, two distinct values means boolean, low-cardinality numerics are
categories, all-unique integers are ids) are made here by :func:`classify_columns` with the
thresholds as arguments, so a test can pin them down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnTypes:
    """Result of :func:`classify_columns`."""

    target: str | None = None
    continuous: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    identifier: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def predictors(self) -> list[str]:
        """Columns usable directly as numeric predictors (continuous + boolean)."""
        return [*self.continuous, *self.boolean]

    def summary(self) -> pd.DataFrame:
        rows = []
        for kind in (
            "continuous",
            "boolean",
            "categorical",
            "datetime",
            "identifier",
            "text",
            "dropped",
        ):
            rows.extend({"column": c, "type": kind} for c in getattr(self, kind))
        return pd.DataFrame(rows)


def _is_string_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(s) or s.dtype == object


def classify_columns(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    max_null_fraction: float = 0.9,
    category_limit: int = 20,
    float_category_limit: int = 8,
    text_min_chars: int = 30,
) -> ColumnTypes:
    """Assign each column a modelling role.

    A column is ``dropped`` when constant or more than ``max_null_fraction`` null, ``boolean``
    when it has exactly two distinct values, ``datetime`` by dtype. Strings are ``identifier``
    when unique per row, ``text`` when their mean length is at least ``text_min_chars``, and
    ``categorical`` otherwise. Integers are ``identifier`` when all unique and ``categorical``
    up to ``category_limit`` distinct values. Floats with at most ``float_category_limit``
    distinct values are ``categorical``. Everything else numeric is ``continuous``.
    """
    result = ColumnTypes(target=target)
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique <= 1 or s.isna().mean() >= max_null_fraction:
            result.dropped.append(col)
        elif nunique == 2:
            result.boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            result.datetime.append(col)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            result.categorical.append(col)
        elif _is_string_dtype(s):
            if nunique == n:
                result.identifier.append(col)
            elif s.dropna().astype(str).str.len().mean() >= text_min_chars:
                result.text.append(col)
            else:
                result.categorical.append(col)
        elif pd.api.types.is_integer_dtype(s):
            if nunique == n:
                result.identifier.append(col)
            elif nunique <= category_limit and nunique < n:
                result.categorical.append(col)
            else:
                result.continuous.append(col)
        elif pd.api.types.is_float_dtype(s):
            if nunique <= float_category_limit:
                result.categorical.append(col)
            else:
                result.continuous.append(col)
        elif pd.api.types.is_bool_dtype(s):
            result.boolean.append(col)
        else:
            result.dropped.append(col)
    return result


def one_hot_encode(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    drop_first: bool = True,
    category_limit: int | None = 20,
    dtype: str = "float64",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Dummy-encode ``columns`` while preserving NaN rows (NaN -> NaN in every dummy).

    Returns ``(encoded_frame, {source_column: [dummy_columns]})``. Columns with more than
    ``category_limit`` distinct values are left untouched. ``pd.get_dummies`` turns NaN into
    all-zero rows; this keeps them NaN and keeps the original index, so the result concatenates
    back onto the source frame cleanly.
    """
    out = df.copy()
    mapping: dict[str, list[str]] = {}
    for col in columns:
        s = df[col]
        categories = pd.Index(s.dropna().unique())
        if category_limit is not None and len(categories) > category_limit:
            continue
        categories = categories.sort_values()
        use = categories[1:] if drop_first and len(categories) > 1 else categories
        names: list[str] = []
        for cat in use:
            name = f"{col}_{cat}"
            values = (s == cat).astype(dtype)
            values[s.isna()] = np.nan
            out[name] = values
            names.append(name)
        mapping[col] = names
        out = out.drop(columns=[col])
    return out, mapping


def select_model_columns(
    df: pd.DataFrame,
    types: ColumnTypes,
    *,
    exclude: Sequence[str] = (),
    drop_all_zero: bool = True,
) -> list[str]:
    """Predictor columns that are numeric, informative and not excluded."""
    cols = [c for c in types.predictors() if c not in set(exclude) and c in df.columns]
    if drop_all_zero:
        cols = [c for c in cols if not (pd.to_numeric(df[c], errors="coerce").fillna(0) == 0).all()]
    return cols
