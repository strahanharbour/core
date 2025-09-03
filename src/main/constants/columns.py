from __future__ import annotations

from enum import Enum


class Col(str, Enum):
    # Canonical OHLCV + extras (camelCase)
    DATE = "date"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    ADJ_CLOSE = "adjClose"
    SYMBOL = "symbol"

    # Features
    RSI14 = "rsi14"
    ATR14 = "atr14"
    MACD = "macd"
    MACD_SIGNAL = "macdSignal"
    MACD_HIST = "macdHist"
    VWAP = "vwap"
    VOL_MULT = "volMult"


__all__ = ["Col"]

