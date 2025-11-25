from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def hrp_allocation(returns: pd.DataFrame) -> pd.Series:
    cov = returns.cov()
    corr = returns.corr().fillna(0)
    dist = np.sqrt(0.5 * (1 - corr))
    Z = linkage(squareform(dist.values, checks=False), method="ward")
    order = leaves_list(Z)
    ordered = cov.iloc[order, order]
    weights = _recursive_bisection(ordered)
    return pd.Series(weights, index=ordered.index)


def _recursive_bisection(cov: pd.DataFrame) -> np.ndarray:
    if cov.shape[0] == 1:
        return np.array([1.0])
    mid = cov.shape[0] // 2
    left = cov.iloc[:mid, :mid]
    right = cov.iloc[mid:, mid:]
    w_left = _recursive_bisection(left)
    w_right = _recursive_bisection(right)
    var_left = (w_left @ left.values @ w_left)
    var_right = (w_right @ right.values @ w_right)
    alpha = 1.0 - var_left / (var_left + var_right)
    return np.concatenate([w_left * (1 - alpha), w_right * alpha])

