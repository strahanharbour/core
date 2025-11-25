from __future__ import annotations
import numpy as np
import pandas as pd


def triple_barrier_labels(
    prices: pd.Series,
    horizon: int = 20,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
    vol_win: int = 20,
) -> pd.DataFrame:
    """
    Event-based labels (Lopez de Prado). Uses rolling volatility at t0 only.
    Returns: DataFrame with columns ['t1','upper','lower','label'] where label ∈ {+1,-1,0}.
    """
    px = prices.dropna()
    ret = np.log(px).diff()
    sigma = ret.rolling(vol_win, min_periods=vol_win // 2).std().shift(1)  # NO look-ahead
    df = pd.DataFrame(index=px.index)
    df["t1"] = px.index.to_series().shift(-horizon)
    df["upper"] = px * (1 + up_mult * sigma.fillna(method="bfill"))
    df["lower"] = px * (1 - dn_mult * sigma.fillna(method="bfill"))

    out = []
    for t0, row in df.iterrows():
        t1 = row.t1
        if pd.isna(t1):
            out.append(0)
            continue
        path = px.loc[t0:t1]
        hit_up = (path >= row.upper).idxmax() if (path >= row.upper).any() else None
        hit_dn = (path <= row.lower).idxmax() if (path <= row.lower).any() else None
        if hit_up is not None and (hit_dn is None or hit_up < hit_dn):
            out.append(+1)
        elif hit_dn is not None and (hit_up is None or hit_dn < hit_up):
            out.append(-1)
        else:
            out.append(0)
    df["label"] = out
    return df


def metalabel_targets(base_signal: pd.Series, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align base directional signal with event labels to produce a meta dataset.
    meta_y = 1 if direction matches label(+1/-1), else 0; rows with label==0 are dropped.
    """
    # Direction at t0: sign(base_signal)
    s = pd.Series(np.sign(base_signal), index=base_signal.index).replace(0, np.nan)
    j = (labels_df[["label"]].join(s.rename("signal"), how="inner")).dropna()
    j = j[j["label"] != 0]
    j["meta_y"] = (j["label"] == j["signal"]).astype(int)
    return j[["signal", "label", "meta_y"]]

