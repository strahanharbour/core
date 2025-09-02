from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl


def to_pandas(obj: Any) -> pd.DataFrame | pd.Series:
    """
    Convert a Polars DataFrame/Series (or generic tabular object) to pandas.
    Pass through pandas objects unchanged.
    """
    # Pass-through
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj

    # Polars conversions
    if isinstance(obj, pl.DataFrame):
        return obj.to_pandas()
    if isinstance(obj, pl.Series):
        # pl.Series.to_pandas() returns pd.Series
        return obj.to_pandas()

    # Best-effort fallback for common inputs
    try:
        return pd.DataFrame(obj)
    except Exception:
        # As a last resort, wrap sequence as Series
        return pd.Series(list(obj))


def to_polars(obj: Any) -> pl.DataFrame | pl.Series:
    """
    Convert a pandas DataFrame/Series (or generic tabular object) to Polars.
    Pass through Polars objects unchanged.
    """
    # Pass-through
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return obj

    # Pandas conversions
    if isinstance(obj, pd.DataFrame):
        return pl.from_pandas(obj)
    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else "value"
        return pl.Series(name=name, values=obj.to_numpy())

    # Best-effort fallback for common inputs
    try:
        return pl.DataFrame(obj)
    except Exception:
        # As a last resort, wrap sequence as Series
        return pl.Series(list(obj))


__all__ = ["to_pandas", "to_polars"]

