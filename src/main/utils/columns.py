from __future__ import annotations

import re
from typing import Any, Dict

import humps
import pandas as pd
import polars as pl


def _canonical_camel(name: Any) -> str:
    """Camelize column names without corrupting existing camelCase.

    - Converts snake/space/hyphen separated names to lowerCamelCase.
    - Preserves boundaries in existing camelCase/PascalCase by round-tripping
      through decamelize -> camelize.
    """
    s = str(name).strip()
    # Flatten tuple-like names from yfinance: ("Close","AAPL") -> "Close"
    if s.startswith("(") and "," in s and "'" in s:
        m = re.match(r"\('([^']+)'\s*,", s)
        if m:
            s = m.group(1)
    # Replace non-alnum with underscores, but avoid unconditional lowercasing
    s_clean = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    if "_" in s_clean:
        return humps.camelize(s_clean.lower())
    try:
        snake = humps.decamelize(s_clean)
        return humps.camelize(snake)
    except Exception:
        return s_clean


def _camelize_name(name: Any) -> str:
    s = str(name)
    # Flatten tuple-like names from yfinance: ("Close","AAPL") → "Close"
    if s.startswith("(") and "," in s and "'" in s:
        m = re.match(r"\('([^']+)'\s*,", s)
        if m:
            s = m.group(1)
    # Replace non-alnum with underscores, lower, then camelize
    s_clean = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()
    return humps.camelize(s_clean)


def canonicalize_pandas_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        base_cols = [str(t[0]) if isinstance(t, tuple) and t else str(t) for t in df.columns]
        df = df.copy()
        df.columns = base_cols
    mapping: Dict[str, str] = {str(c): _canonical_camel(c) for c in df.columns}
    if any(k != v for k, v in mapping.items()):
        df = df.rename(columns=mapping)
    return df


def canonicalize_polars_columns(df: pl.DataFrame) -> pl.DataFrame:
    mapping: Dict[str, str] = {c: _canonical_camel(c) for c in df.columns}
    if any(k != v for k, v in mapping.items()):
        df = df.rename(mapping)
    return df


__all__ = ["canonicalize_pandas_columns", "canonicalize_polars_columns"]
