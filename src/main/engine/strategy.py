from __future__ import annotations

from math import floor
from typing import Any, Iterable

import numpy as np

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    pd = None  # type: ignore


def _is_polars_df(obj: Any) -> bool:
    return pl is not None and isinstance(obj, pl.DataFrame)


def _is_polars_series(obj: Any) -> bool:
    return pl is not None and isinstance(obj, pl.Series)


def _is_pandas_df(obj: Any) -> bool:
    return pd is not None and isinstance(obj, pd.DataFrame)


def _is_pandas_series(obj: Any) -> bool:
    return pd is not None and isinstance(obj, pd.Series)


def signal_rules(df: Any) -> Any:
    """
    Technical long entry conditions as a boolean mask aligned to df rows.

    Conditions (all must hold):
      - RSI crosses up 30: rsi14.shift(1) < 30 and rsi14 >= 30
      - MACD histogram > 0 and rising vs prior bar
      - Close > vwap
      - vol_mult >= 1.2

    Works with Polars or Pandas DataFrame.
    Returns a Series of the same library type (pl.Series or pd.Series or list[bool]).
    """
    cols = {"rsi14", "macd_hist", "vwap", "vol_mult", "Close"}
    missing = [c for c in cols if c not in set(df.columns)]
    if missing:
        raise KeyError(f"signal_rules missing columns: {missing}")

    if _is_polars_df(df):
        rsi = pl.col("rsi14")
        hist = pl.col("macd_hist")
        close = pl.col("Close")
        vwap = pl.col("vwap")
        volm = pl.col("vol_mult")

        cross_up = (rsi.shift(1) < 30) & (rsi >= 30)
        macd_ok = (hist > 0) & (hist > hist.shift(1))
        price_ok = close > vwap
        vol_ok = volm >= 1.2
        expr = (cross_up & macd_ok & price_ok & vol_ok).alias("signal")
        return df.with_columns(expr).get_column("signal").fill_null(False)

    if _is_pandas_df(df):
        rsi = df["rsi14"]
        hist = df["macd_hist"]
        close = df["Close"]
        vwap = df["vwap"]
        volm = df["vol_mult"]

        signal = (
            (rsi.shift(1) < 30) & (rsi >= 30)
            & (hist > 0) & (hist > hist.shift(1))
            & (close > vwap)
            & (volm >= 1.2)
        )
        return signal.fillna(False)

    # Fallback: try duck-typed columns supporting shift/comparison; return list of bool
    rsi = df["rsi14"]
    hist = df["macd_hist"]
    close = df["Close"]
    vwap = df["vwap"]
    volm = df["vol_mult"]
    # Convert to numpy arrays with NaN handling
    rsi_a = np.asarray(rsi)
    hist_a = np.asarray(hist)
    close_a = np.asarray(close)
    vwap_a = np.asarray(vwap)
    volm_a = np.asarray(volm)
    rsi_prev = np.roll(rsi_a, 1)
    rsi_prev[0] = np.nan
    hist_prev = np.roll(hist_a, 1)
    hist_prev[0] = np.nan
    signal = (
        (np.nan_to_num((rsi_prev < 30) & (rsi_a >= 30), nan=False))
        & (np.nan_to_num((hist_a > 0) & (hist_a > hist_prev), nan=False))
        & (np.nan_to_num(close_a > vwap_a, nan=False))
        & (np.nan_to_num(volm_a >= 1.2, nan=False))
    )
    return signal.tolist()


def position_size(atr: Any, risk_dollars: float = 25.0) -> Any:
    """
    Position size as floor(risk_dollars / (1.2 * ATR)), clipped at >= 0.

    Accepts scalar, pandas Series, polars Series, or array-like.
    Returns int or a same-library Series/array of ints.
    """
    k = 1.2

    if isinstance(atr, (int, float)):
        if atr <= 0:
            return 0
        return max(0, floor(risk_dollars / (k * float(atr))))

    if _is_polars_series(atr):
        arr = atr.to_numpy()
        res = np.floor(risk_dollars / (k * np.maximum(arr, 1e-12)))
        res = np.clip(res, 0, None).astype(int)
        name = getattr(atr, "name", None) or "position_size"
        return pl.Series(name=name, values=res)  # type: ignore[arg-type]

    if _is_pandas_series(atr):
        arr = atr.to_numpy(dtype=float)
        res = np.floor(risk_dollars / (k * np.maximum(arr, 1e-12)))
        res = np.clip(res, 0, None).astype(int)
        return pd.Series(res, index=atr.index, name="position_size")

    arr = np.asarray(atr, dtype=float)
    res = np.floor(risk_dollars / (k * np.maximum(arr, 1e-12)))
    res = np.clip(res, 0, None).astype(int)
    return res


def gated_entry(df: Any, sic_value: Any = None, use_sic: bool = False, sic_threshold: float = 0.0) -> Any:
    """
    Combine technical signal with an optional SIC gate.

    - tech: signal_rules(df)
    - gate: if use_sic is True, require sic >= sic_threshold; otherwise pass-through.

    `sic_value` may be:
      - a column name present in df
      - a Series aligned to df
      - a scalar (broadcast)

    Returns a boolean Series aligned to df rows (type matches df library).
    """
    tech = signal_rules(df)
    if not use_sic:
        return tech

    if isinstance(sic_value, str):
        if _is_polars_df(df):
            if sic_value not in df.columns:
                raise KeyError(f"SIC column '{sic_value}' not found in df")
            sic_series = df.get_column(sic_value)
        elif _is_pandas_df(df):
            if sic_value not in df.columns:
                raise KeyError(f"SIC column '{sic_value}' not found in df")
            sic_series = df[sic_value]
        else:
            sic_series = df[sic_value]
    else:
        sic_series = sic_value

    if _is_polars_df(df):
        if _is_polars_series(sic_series):
            gate = sic_series >= sic_threshold
        elif isinstance(sic_series, (int, float)):
            gate = pl.Series(values=[sic_series] * df.height) >= sic_threshold
        else:
            # Try to coerce to pl.Series
            gate = pl.Series(values=list(sic_series)) >= sic_threshold
        return (tech & gate).fill_null(False)

    if _is_pandas_df(df):
        if _is_pandas_series(sic_series):
            gate = sic_series >= sic_threshold
        elif isinstance(sic_series, (int, float)):
            gate = pd.Series(sic_series, index=df.index) >= sic_threshold
        else:
            gate = pd.Series(sic_series, index=df.index) >= sic_threshold
        return (tech & gate).fillna(False)

    # Fallback numpy
    if isinstance(sic_series, (int, float)):
        gate = np.asarray([sic_series] * len(df)) >= sic_threshold
    else:
        gate = np.asarray(sic_series) >= sic_threshold
    return (np.asarray(tech, dtype=bool) & gate).tolist()


__all__ = ["signal_rules", "position_size", "gated_entry"]

