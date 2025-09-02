from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _to_returns_series(returns: Iterable[float]) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.astype(float)
    return pd.Series(returns, dtype=float)


def annualize_return(returns: Iterable[float], periods: int = 252) -> float:
    r = _to_returns_series(returns).dropna()
    if r.empty:
        return 0.0
    g = (1.0 + r).prod()
    n = len(r)
    return float(g ** (periods / max(1, n)) - 1.0)


def sharpe_ratio(returns: Iterable[float], risk_free: float = 0.0, periods: int = 252) -> float:
    r = _to_returns_series(returns).dropna()
    if r.empty:
        return 0.0
    # risk_free is annualized; convert to per-period
    rp = r - (risk_free / periods)
    mu = rp.mean()
    sig = rp.std(ddof=1)
    if sig == 0 or math.isnan(sig):
        return 0.0
    return float((mu / sig) * math.sqrt(periods))


def sortino_ratio(
    returns: Iterable[float], risk_free: float = 0.0, periods: int = 252, target: float = 0.0
) -> float:
    r = _to_returns_series(returns).dropna()
    if r.empty:
        return 0.0
    rp = r - (risk_free / periods)
    downside = rp.copy()
    downside = downside[downside < target]
    if downside.empty:
        return float("inf")
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or math.isnan(downside_std):
        return float("inf")
    mu = rp.mean() - target
    return float((mu / downside_std) * math.sqrt(periods))


def equity_curve(returns: Iterable[float], start: float = 10_000.0) -> pd.Series:
    r = _to_returns_series(returns).fillna(0.0)
    eq = float(start) * (1.0 + r).cumprod()
    return eq


def max_drawdown_from_equity(equity: Iterable[float]) -> Tuple[float, int, int]:
    e = _to_returns_series(equity).astype(float)
    if e.empty:
        return 0.0, -1, -1
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    mdd = dd.min()
    end_idx = int(dd.idxmin()) if isinstance(dd.index, pd.RangeIndex) else dd.values.argmin()
    start_idx = int((e[: end_idx + 1]).idxmax()) if isinstance(e.index, pd.RangeIndex) else (e.values[: end_idx + 1].argmax())
    return float(mdd), int(start_idx), int(end_idx)


def max_drawdown(returns: Iterable[float], start: float = 10_000.0) -> float:
    eq = equity_curve(returns, start=start)
    mdd, _, _ = max_drawdown_from_equity(eq)
    return float(mdd)


def cvar(returns: Iterable[float], alpha: float = 0.05) -> float:
    r = _to_returns_series(returns).dropna().astype(float)
    if r.empty:
        return 0.0
    # Loss is -returns; CVaR is expected loss beyond VaR at alpha
    q = r.quantile(alpha)
    tail = r[r <= q]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def pnl_attribution_by(trades: pd.DataFrame, by: str = "symbol") -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=[by, "trades", "pnl_sum", "pnl_avg", "win_rate"]).set_index(by)
    g = trades.groupby(by)
    agg = g["pnl"].agg(["count", "sum", "mean"]).rename(
        columns={"count": "trades", "sum": "pnl_sum", "mean": "pnl_avg"}
    )
    wins = g.apply(lambda df: (df["pnl"] > 0).sum())
    agg["win_rate"] = (wins / agg["trades"]).fillna(0.0)
    return agg.sort_values("pnl_sum", ascending=False)


__all__ = [
    "annualize_return",
    "sharpe_ratio",
    "sortino_ratio",
    "equity_curve",
    "max_drawdown_from_equity",
    "max_drawdown",
    "cvar",
    "pnl_attribution_by",
]

