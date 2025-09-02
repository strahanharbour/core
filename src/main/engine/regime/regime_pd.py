from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def realized_volatility(returns: Iterable[float] | pd.Series, window: int = 21, ann_factor: int = 252) -> pd.Series:
    r = pd.Series(returns, dtype=float) if not isinstance(returns, pd.Series) else returns.astype(float)
    vol = r.rolling(window=window, min_periods=max(2, window // 2)).std()
    return vol * np.sqrt(ann_factor)


def classify_regime(vol: pd.Series, q_low: float = 0.33, q_high: float = 0.66) -> pd.Series:
    vol = vol.dropna()
    if vol.empty:
        return pd.Series(dtype=object)
    lo = vol.quantile(q_low)
    hi = vol.quantile(q_high)
    def label(x: float) -> str:
        if x <= lo:
            return "low"
        if x >= hi:
            return "high"
        return "mid"
    return vol.apply(label)


def vol_target_weights(vol_dict: Dict[str, float]) -> Dict[str, float]:
    # Inverse-vol weights normalized to 1
    filt = {k.upper(): float(v) for k, v in vol_dict.items() if v and v > 0}
    if not filt:
        return {}
    inv = {k: 1.0 / v for k, v in filt.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def change_points(regimes: pd.Series) -> List[pd.Timestamp]:
    # Simple change points when regime label changes
    if regimes is None or regimes.empty:
        return []
    r = regimes.astype(str)
    idx = r.index
    chg = r != r.shift(1)
    return [idx[i] for i, v in enumerate(chg) if bool(v)]


__all__ = [
    "realized_volatility",
    "classify_regime",
    "vol_target_weights",
    "change_points",
]

