from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


CalType = Optional[Literal["isotonic", "sigmoid"]]


@dataclass
class StackingMetaModel:
    calibration: CalType = "isotonic"
    C: float = 1.0
    max_iter: int = 1000
    clf_: Optional[CalibratedClassifierCV | LogisticRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingMetaModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        base = LogisticRegression(C=self.C, max_iter=self.max_iter, n_jobs=None)
        if self.calibration in ("isotonic", "sigmoid"):
            self.clf_ = CalibratedClassifierCV(base, method=self.calibration, cv=5)
        else:
            self.clf_ = base
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        proba = self.clf_.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        # Multiclass: return max class probability
        return proba.max(axis=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def brier(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        return float(brier_score_loss(y, p))


__all__ = ["StackingMetaModel"]

