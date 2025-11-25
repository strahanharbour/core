from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe(daily: pd.Series, eps=1e-12) -> float:
    mu = daily.mean()
    sd = daily.std()
    return float(np.sqrt(252) * mu / (sd + eps))


def sortino(daily: pd.Series, eps=1e-12) -> float:
    dd = daily[daily < 0].std()
    return float(np.sqrt(252) * daily.mean() / (dd + eps))


def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    return float(dd)


def cvar(daily: pd.Series, alpha=0.95) -> float:
    q = daily.quantile(1 - alpha)
    tail = daily[daily <= q]
    return float(tail.mean())

