from __future__ import annotations

from typing import Literal

import polars as pl


def ensure_date(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    """Cast the given column to pl.Date (non-strict) if present."""
    if col not in df.columns:
        return df
    return df.with_columns(pl.col(col).cast(pl.Date, strict=False).alias(col))


def join_on_date(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_on: str = "date",
    right_on: str = "date",
    how: Literal["left", "inner", "outer", "semi", "anti", "cross"] = "left",
) -> pl.DataFrame:
    """Join two DataFrames on date-like columns with safe casting to pl.Date.

    If left_on != right_on, both are cast to Date and joined via a temporary key.
    """
    if left_on == right_on:
        l = ensure_date(left, left_on)
        r = ensure_date(right, right_on)
        return l.join(r, on=left_on, how=how)

    l = ensure_date(left.rename({left_on: "_join_date"}), "_join_date")
    r = ensure_date(right.rename({right_on: "_join_date"}), "_join_date")
    out = l.join(r, on="_join_date", how=how)
    # Rename back the left join key to original name
    out = out.rename({"_join_date": left_on})
    return out


__all__ = ["ensure_date", "join_on_date"]

