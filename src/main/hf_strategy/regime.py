from __future__ import annotations
import pandas as pd
import numpy as np


def classify_regime(vix: pd.Series) -> pd.Series:
    v = vix.ffill()
    reg = pd.Series(index=v.index, dtype="object")
    reg[v < 15] = "low"
    reg[(v >= 15) & (v <= 25)] = "mid"
    reg[v > 25] = "high"
    return reg


def regime_weights(regime: str) -> tuple[float, float]:
    if regime == "low":
        return (0.2, 0.8)
    if regime == "mid":
        return (0.5, 0.5)
    return (0.7, 0.3)


def realized_vol(prices: pd.Series, win: int = 20) -> pd.Series:
    prices = pd.Series(prices)  # Ensure prices is a pandas Series
    log_prices = np.log(prices)
    log_prices = pd.Series(log_prices, index=prices.index)  # Ensure log_prices is a pandas Series
    r = log_prices.diff()
    return r.rolling(win, min_periods=win // 2).std()


def volatility_target_weights(vol: pd.Series, target: float = 0.01, eps: float = 1e-6) -> pd.Series:
    # scale exposure inversely with realized vol; clip to sane bounds
    w = target / (vol + eps)
    return w.clip(upper=3.0)

