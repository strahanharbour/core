from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def regime_weights(vix: float) -> Tuple[float, float]:
    """
    Return (macro_weight, micro_weight) based on VIX per blueprint v1.2.0:
      - VIX < 15  -> (0.2, 0.8)
      - 15–25     -> (0.5, 0.5)
      - > 25      -> (0.7, 0.3)
    """
    v = float(vix)
    if v < 15.0:
        return 0.2, 0.8
    if v <= 25.0:
        return 0.5, 0.5
    return 0.7, 0.3


def exp_time_decay(series: pd.Series, lam: float = 0.05) -> float:
    """
    Exponentially time-decayed average of a time-indexed series.
    Assumes series is sorted ascending by time index.
    """
    if series is None or series.empty:
        return 0.0
    n = len(series)
    # Most recent gets highest weight: w_k = exp(-lam * age), age=0..n-1
    ages = np.arange(n - 1, -1, -1, dtype=float)  # oldest .. newest mapping
    ages = np.arange(n, dtype=float)[::-1]  # newest age=0
    w = np.exp(-lam * ages)
    x = series.astype(float).to_numpy()
    return float(np.sum(w * x) / np.sum(w))


@dataclass
class SICInputs:
    macro: pd.Series  # time-indexed macro scores
    micro: pd.Series  # time-indexed micro scores
    vix: float
    lam_macro: float = 0.05
    lam_micro: float = 0.05


def compute_sic(inputs: SICInputs) -> float:
    m_weight, s_weight = regime_weights(inputs.vix)
    macro_s = exp_time_decay(inputs.macro.sort_index(), lam=inputs.lam_macro)
    micro_s = exp_time_decay(inputs.micro.sort_index(), lam=inputs.lam_micro)
    return float(m_weight * macro_s + s_weight * micro_s)


__all__ = ["regime_weights", "exp_time_decay", "SICInputs", "compute_sic"]

