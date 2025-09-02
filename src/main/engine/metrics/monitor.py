from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy.stats import ks_2samp


def ks_drift(base: Iterable[float], sample: Iterable[float]) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov statistic and p-value comparing base vs sample.
    Returns (statistic, pvalue).
    """
    a = np.asarray(list(base), dtype=float)
    b = np.asarray(list(sample), dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return float(stat), float(p)


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    t = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_prob), dtype=float)
    t = np.clip(t, 0.0, 1.0)
    p = np.clip(p, 0.0, 1.0)
    if t.size == 0:
        return 0.0
    return float(np.mean((p - t) ** 2))


__all__ = ["ks_drift", "brier_score"]

