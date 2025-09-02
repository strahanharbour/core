from __future__ import annotations

from typing import Generator, Iterable, Tuple

import numpy as np


def walk_forward(
    n_samples_or_y: int | Iterable,
    train_window: int,
    test_window: int,
    *,
    step: int | None = None,
    expanding: bool = False,
) -> Generator[Tuple[slice, slice], None, None]:
    """
    Rolling walk-forward slices for time-ordered data.

    Yields consecutive (train_slice, test_slice) pairs where each is a Python
    slice (start, stop) over 0..n.

    Args:
      n_samples_or_y: number of samples, or any iterable whose length defines it
      train_window: number of samples in the training window (ignored if expanding=True after first window)
      test_window: number of samples in each test window
      step: step size between windows (default: test_window)
      expanding: if True, training window expands from 0 to current test start;
                 if False, uses fixed-size rolling window of train_window.
    """
    if isinstance(n_samples_or_y, int):
        n = int(n_samples_or_y)
    else:
        try:
            n = len(n_samples_or_y)  # type: ignore[arg-type]
        except TypeError as e:  # pragma: no cover - defensive
            raise TypeError("n_samples_or_y must be int or sized iterable") from e

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be > 0")

    step = test_window if step is None else int(step)
    start_test = train_window
    while start_test < n:
        end_test = min(n, start_test + test_window)
        if expanding:
            train_start = 0
            train_end = start_test
        else:
            train_start = max(0, start_test - train_window)
            train_end = start_test

        if train_end - train_start <= 0:
            break

        yield slice(train_start, train_end), slice(start_test, end_test)
        if end_test >= n:
            break
        start_test += step


def walk_forward_indices(
    n_samples_or_y: int | Iterable,
    train_window: int,
    test_window: int,
    *,
    step: int | None = None,
    expanding: bool = False,
):
    """
    Convenience wrapper returning numpy arrays of indices for each walk-forward split.
    """
    if isinstance(n_samples_or_y, int):
        n = int(n_samples_or_y)
    else:
        n = len(n_samples_or_y)  # type: ignore[arg-type]
    for tr_s, te_s in walk_forward(n, train_window, test_window, step=step, expanding=expanding):
        yield np.arange(tr_s.start, tr_s.stop), np.arange(te_s.start, te_s.stop)


__all__ = ["walk_forward", "walk_forward_indices"]

