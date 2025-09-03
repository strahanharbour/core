from __future__ import annotations

from math import floor
from typing import Any, Iterable
from functools import lru_cache

import numpy as np
from constants import Col
from config_env import load_cfg

import polars as pl

import pandas as pd

def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool): 
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    return s in ("1","true","t","yes","y","on")

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
    Technical long entry mask.

    Base conditions:
      - RSI crosses up L: rsi14.shift(1) < L and rsi14 >= L
      - MACD histogram > floor and rising vs prior bar
      - Close > vwap
      - vol_mult >= vol_mult_min

    New behavior:
      - If require_same_bar = true  → all on the SAME bar (original behavior)
      - If require_same_bar = false → each sub-signal may occur within the last K bars (lookback_bars)
    """
    cols = {Col.RSI14.value, Col.MACD_HIST.value, Col.VWAP.value, Col.VOL_MULT.value, Col.CLOSE.value}
    missing = [c for c in cols if c not in set(df.columns)]
    if missing:
        raise KeyError(f"signal_rules missing columns: {missing}")

    p = _get_strategy_params()
    rsi_level       = float(p["rsi_cross_level"])
    vol_mult_min    = float(p["vol_mult_min"])
    macd_hist_floor = float(p["macd_hist_floor"])
    k               = int(p["lookback_bars"])
    same_bar        = bool(p["require_same_bar"])
    require_rising  = bool(p.get("require_macd_rising", True))
    slack           = 1.0 - (float(p.get("price_slack_bps", 0.0)) / 10_000.0)

    # --- Polars branch ---
    if _is_polars_df(df):
        rsi  = pl.col(Col.RSI14.value)
        hist = pl.col(Col.MACD_HIST.value)
        close= pl.col(Col.CLOSE.value)
        vwap = pl.col(Col.VWAP.value)
        volm = pl.col(Col.VOL_MULT.value)

        cross_up = (rsi.shift(1) < rsi_level) & (rsi >= rsi_level)
        macd_ok  = (hist > macd_hist_floor) & (hist > hist.shift(1)) if require_rising else (hist > macd_hist_floor)
        price_ok = close >= (vwap * slack)
        vol_ok   = volm >= vol_mult_min

        if same_bar:
            base = cross_up & macd_ok & price_ok & vol_ok
        else:
            # allow any of last K bars (rolling OR)
            cross_up_k = cross_up.rolling_max(window_size=k).cast(pl.Boolean)
            macd_ok_k  = macd_ok.rolling_max(window_size=k).cast(pl.Boolean)
            price_ok_k = price_ok.rolling_max(window_size=k).cast(pl.Boolean)
            vol_ok_k   = vol_ok.rolling_max(window_size=k).cast(pl.Boolean)
            base = cross_up_k & macd_ok_k & price_ok_k & vol_ok_k

        warm = max(30, k)
        idx  = pl.arange(0, pl.len())
        expr = (base & (idx >= warm)).alias("signal")
        return df.with_columns(expr).get_column("signal").fill_null(False)

    # --- Pandas branch ---
    if _is_pandas_df(df):
        rsi  = df[Col.RSI14.value]
        hist = df[Col.MACD_HIST.value]
        close= df[Col.CLOSE.value]
        vwap = df[Col.VWAP.value]
        volm = df[Col.VOL_MULT.value]

        cross_up = (rsi.shift(1) < rsi_level) & (rsi >= rsi_level)
        macd_ok  = (hist > macd_hist_floor) & (hist > hist.shift(1))
        price_ok = close > vwap
        vol_ok   = volm >= vol_mult_min

        if same_bar:
            sig = cross_up & macd_ok & price_ok & vol_ok
        else:
            # rolling OR over last K bars
            sig = (
                cross_up.rolling(k).max().astype(bool)
                & macd_ok.rolling(k).max().astype(bool)
                & price_ok.rolling(k).max().astype(bool)
                & vol_ok.rolling(k).max().astype(bool)
            )

        warm = max(30, k)
        sig.iloc[:warm] = False
        return sig.fillna(False)

    # --- Fallback duck-typed/numpy ---
    rsi  = np.asarray(df[Col.RSI14.value], dtype=float)
    hist = np.asarray(df[Col.MACD_HIST.value], dtype=float)
    close= np.asarray(df[Col.CLOSE.value], dtype=float)
    vwap = np.asarray(df[Col.VWAP.value], dtype=float)
    volm = np.asarray(df[Col.VOL_MULT.value], dtype=float)

    rsi_prev  = np.roll(rsi, 1);  rsi_prev[0] = np.nan
    hist_prev = np.roll(hist, 1); hist_prev[0] = np.nan

    cross_up = (rsi_prev < rsi_level) & (rsi >= rsi_level)
    macd_ok  = (hist > macd_hist_floor) & (hist > hist_prev)
    price_ok = close > vwap
    vol_ok   = volm >= vol_mult_min

    if same_bar:
        base = cross_up & macd_ok & price_ok & vol_ok
    else:
        # rolling OR via convolutions over window k
        def roll_or(x: np.ndarray, k: int) -> np.ndarray:
            # Treat NaN as False
            y = np.nan_to_num(x.astype(bool), nan=False)
            # cumulative trick for simple windowed OR
            c = np.cumsum(y.astype(int))
            z = c - np.concatenate(([0], c[:-k]))
            # pad first k-1 as window grows
            head = np.maximum.accumulate(y[:k-1]).astype(int)
            out = np.concatenate((head, z[k-1:]))
            return out > 0

        base = roll_or(cross_up, k) & roll_or(macd_ok, k) & roll_or(price_ok, k) & roll_or(vol_ok, k)

    warm = max(30, k)
    base[:warm] = False
    return base.tolist()


def position_size(atr: Any, risk_dollars: float | None = None, atr_multiple: float | None = None) -> Any:
    """
    Position size as floor(risk_dollars / (k * ATR)), clipped at >= 0.

    k comes from config: strategy.atr_sizing_multiple (default 1.2)
    risk_dollars defaults to config: risk.risk_dollars_per_trade (default 25.0)

    Accepts scalar, pandas Series, polars Series, or array-like.
    Returns int or a same-library Series/array of ints.
    """
    # read ATR sizing multiple
    if atr_multiple is None:
        k = float(_get_strategy_params()["atr_sizing_multiple"])  # e.g., 1.2
    else:
        k = float(atr_multiple)

    # read risk dollars from config if not provided
    if risk_dollars is None:
        risk_dollars = float(_get_risk_params()["risk_dollars_per_trade"])  # default 25.0
    else:
        risk_dollars = float(risk_dollars)

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


@lru_cache(maxsize=1)
def _get_strategy_params() -> dict[str, Any]:
    try:
        meta = load_cfg()
        cfg = meta.get("cfg", {}) or {}
        strat = cfg.get("strategy", {}) or {}
    except Exception:
        strat = {}
    return {
        "rsi_cross_level": float(strat.get("rsi_cross_level", 30.0)),
        "vol_mult_min": float(strat.get("vol_mult_min", 1.2)),
        "macd_hist_floor": float(strat.get("macd_hist_floor", 0.0)),
        "atr_sizing_multiple": float(strat.get("atr_sizing_multiple", 1.2)),
        # sweep/runtime extras (used by some callers)
        "lookback_bars": int(strat.get("lookback_bars", 3)),
        "require_same_bar": _as_bool(strat.get("require_same_bar", False)),
        "price_slack_bps": float(strat.get("price_slack_bps", 0.0)),
    }


# Risk params loader
@lru_cache(maxsize=1)
def _get_risk_params() -> dict[str, float]:
    """
    Load risk knobs from config.yaml with defaults.
      - risk.risk_dollars_per_trade (float)
    """
    try:
        meta = load_cfg()
        cfg = meta.get("cfg", {}) or {}
        risk = cfg.get("risk", {}) or {}
    except Exception:
        risk = {}
    return {
        "risk_dollars_per_trade": float(risk.get("risk_dollars_per_trade", 25.0)),
    }

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
