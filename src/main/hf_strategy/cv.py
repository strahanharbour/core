from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple


def purged_kfold_indices(n: int, n_splits: int = 5, embargo: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    fold_size = n // n_splits
    for i in range(n_splits):
        start = i * fold_size
        end = n if i == n_splits - 1 else (i + 1) * fold_size
        test_idx = np.arange(start, end)
        train_mask = np.ones(n, dtype=bool)
        # purge overlap
        lo = max(0, start - embargo)
        hi = min(n, end + embargo)
        train_mask[lo:hi] = False
        # keep indices outside embargoed window
        idx = np.arange(n)
        train_idx = idx[train_mask]
        yield train_idx, test_idx

