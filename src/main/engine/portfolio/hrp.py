from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list


def _cov_corr(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = returns.cov()
    corr = returns.corr()
    return cov, corr


def _correl_distance(corr: pd.DataFrame) -> np.ndarray:
    # Distance per Lopez de Prado: d = sqrt(0.5*(1 - corr))
    return np.sqrt(0.5 * (1.0 - corr.values))


def _get_quasi_diag(link: np.ndarray) -> list[int]:
    # Get order of items from hierarchical tree
    leaves = leaves_list(link)
    return list(map(int, leaves))


def _recursive_bisection(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while len(clusters) > 0:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            c1 = cluster[:split]
            c2 = cluster[split:]
            # Inverse-variance weights within sub-clusters
            ivp1 = _inverse_var_portfolio(cov.loc[c1, c1])
            ivp2 = _inverse_var_portfolio(cov.loc[c2, c2])
            var1 = float(ivp1.values.T @ cov.loc[c1, c1].values @ ivp1.values)
            var2 = float(ivp2.values.T @ cov.loc[c2, c2].values @ ivp2.values)
            alpha = 1.0 - var1 / (var1 + var2)
            w[c1] *= alpha
            w[c2] *= 1.0 - alpha
            if len(c1) > 1:
                new_clusters.append(c1)
            if len(c2) > 1:
                new_clusters.append(c2)
        clusters = new_clusters
    return w / w.sum()


def _inverse_var_portfolio(cov: pd.DataFrame) -> pd.Series:
    iv = 1.0 / np.diag(cov.values)
    w = iv / iv.sum()
    return pd.Series(w, index=cov.index)


def hrp_allocation(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity allocation given asset returns DataFrame (columns are assets).
    Returns a pandas Series of weights summing to 1 (index aligned to columns).
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        raise ValueError("returns must be a DataFrame with >=2 columns")
    # Drop columns with all NaNs
    ret = returns.dropna(how="all")
    ret = ret.fillna(0.0)
    cov, corr = _cov_corr(ret)
    dist = _correl_distance(corr)
    # Convert distance matrix to condensed form expected by linkage
    # scipy expects condensed distance; we can flatten upper triangle
    tri_idx = np.triu_indices_from(dist, k=1)
    condensed = dist[tri_idx]
    link = linkage(condensed, method="single")
    sort_ix = _get_quasi_diag(link)
    ordered = list(corr.columns[sort_ix])
    w = _recursive_bisection(cov.loc[ordered, ordered], list(range(len(ordered))))
    # Map weights back to original column names
    weights = pd.Series(0.0, index=corr.columns)
    weights.loc[ordered] = w.values
    return weights


__all__ = ["hrp_allocation"]

