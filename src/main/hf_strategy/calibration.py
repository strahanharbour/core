from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw, y):
        self.iso.fit(p_raw, y)
        return self

    def predict(self, p_raw):
        return self.iso.predict(p_raw)


def brier_score(p, y):
    p = np.asarray(p)
    y = np.asarray(y)
    return np.mean((p - y) ** 2)

