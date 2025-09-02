from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    horizon: int = 10  # bars
    up_sigma: float = 2.0
    down_sigma: float = 1.0
    vol_window: int = 20


def daily_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    returns = prices.pct_change()
    vol = returns.rolling(window=window, min_periods=max(2, window // 2)).std()
    return vol


def triple_barrier_labels(
    prices: pd.Series,
    *,
    horizon: int = 10,
    up_sigma: float = 2.0,
    down_sigma: float = 1.2,
    vol: Optional[pd.Series] = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Return DataFrame with columns: t1, up, down, event_time, label (-1/0/1), ret (log return to event).
    Analysis-only; not optimized for large series.
    """
    assert horizon >= 1
    p = prices.astype(float).copy()
    if vol is None:
        vol = daily_volatility(p, window=vol_window)
    vol = vol.reindex_like(p).fillna(method="bfill").fillna(method="ffill")

    t1 = p.index.to_series().shift(-horizon)
    up = p * (1.0 + up_sigma * vol)
    down = p * (1.0 - down_sigma * vol)

    labels = []
    n = len(p)
    for i, (ts, px) in enumerate(p.items()):
        j_end = min(i + horizon, n - 1)
        win = p.iloc[i : j_end + 1]
        up_i = up.iloc[i]
        dn_i = down.iloc[i]

        # First barrier touch time
        hit_up = (win >= up_i).to_numpy().nonzero()[0]
        hit_dn = (win <= dn_i).to_numpy().nonzero()[0]

        t_hit = None
        lbl = 0
        if hit_up.size > 0 and hit_dn.size > 0:
            t_up = hit_up[0]
            t_dn = hit_dn[0]
            if t_up < t_dn:
                t_hit = i + t_up
                lbl = 1
            elif t_dn < t_up:
                t_hit = i + t_dn
                lbl = -1
            else:
                t_hit = i + t_up
                lbl = 0
        elif hit_up.size > 0:
            t_hit = i + int(hit_up[0])
            lbl = 1
        elif hit_dn.size > 0:
            t_hit = i + int(hit_dn[0])
            lbl = -1
        else:
            t_hit = j_end
            lbl = 0

        px_event = p.iloc[t_hit]
        ret = float(np.log(px_event / px)) if px > 0 else 0.0
        labels.append((t1.iloc[i], up_i, dn_i, p.index[t_hit], lbl, ret))

    df = pd.DataFrame(
        labels,
        index=p.index,
        columns=["t1", "up", "down", "event_time", "label", "ret"],
    )
    return df


def meta_label_from_triple(df: pd.DataFrame) -> pd.Series:
    """
    Meta-label target: 1 if label * ret > 0 (directional correctness), else 0.
    """
    prod = df["label"].astype(float) * df["ret"].astype(float)
    y = (prod > 0).astype(int)
    return y


__all__ = [
    "TripleBarrierConfig",
    "daily_volatility",
    "triple_barrier_labels",
    "meta_label_from_triple",
]

