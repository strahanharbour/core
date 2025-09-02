from __future__ import annotations

from typing import Generator, Iterable, Tuple

import numpy as np


def purged_kfold_indices(
    n_samples_or_y: int | Iterable, n_splits: int = 5, purge: int = 0, embargo: int = 0
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Purged K-Fold splitter for time-ordered data.

    - Splits indices into contiguous K folds in order (no shuffling).
    - Removes a purge window around each test fold from the training set to
      reduce leakage (``purge`` samples on the left of the test fold and
      ``purge + embargo`` samples on the right).

    Args:
      n_samples_or_y: number of samples, or any iterable whose length defines it
      n_splits: number of folds (K >= 2)
      purge: number of samples to drop immediately before and after the test fold
      embargo: additional samples to drop after the test fold (right side)

    Yields:
      (train_idx, test_idx) as numpy int arrays.
    """
    if isinstance(n_samples_or_y, int):
        n = int(n_samples_or_y)
    else:
        try:
            n = len(n_samples_or_y)  # type: ignore[arg-type]
        except TypeError as e:  # pragma: no cover - defensive
            raise TypeError("n_samples_or_y must be int or sized iterable") from e

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n <= n_splits:
        # Allow tiny datasets: some folds will be size 1 or 0
        pass

    indices = np.arange(n, dtype=int)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))

    for k in range(n_splits):
        test_start = int(starts[k])
        test_end = int(test_start + fold_sizes[k])
        test_idx = indices[test_start:test_end]

        # Purge windows from training
        left = max(0, test_start - int(purge))
        right = min(n, test_end + int(purge) + int(embargo))

        mask = np.ones(n, dtype=bool)
        mask[test_start:test_end] = False
        mask[left:test_start] = False
        mask[test_end:right] = False
        train_idx = indices[mask]

        yield train_idx, test_idx


__all__ = ["purged_kfold_indices"]

