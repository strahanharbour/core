from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_vader(text: str | None) -> float:
    """
    Return VADER compound sentiment score in [-1, 1] for a single text.
    Empty/None text returns 0.0.
    """
    if not text:
        return 0.0
    return float(_analyzer().polarity_scores(str(text)).get("compound", 0.0))


def score_vader_many(texts: Iterable[str | None]) -> List[float]:
    return [score_vader(t) for t in texts]


__all__ = ["score_vader", "score_vader_many"]

