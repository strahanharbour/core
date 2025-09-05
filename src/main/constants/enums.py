from __future__ import annotations

from enum import Enum


class AllocatorType(str, Enum):
    """Portfolio allocation policy identifiers."""

    EQUAL = "equal"
    HRP = "hrp"


class NewsProvider(str, Enum):
    """Supported news/sentiment providers for ingestion."""

    NEWSAPI = "newsapi"
    FINNHUB = "finnhub"
    YAHOO_RSS = "yahoo_rss"
    YFINANCE = "yfinance"


__all__ = ["AllocatorType", "NewsProvider"]

