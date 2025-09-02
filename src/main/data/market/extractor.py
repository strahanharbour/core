from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


def _to_ts(x: datetime | str) -> str:
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")
    return str(x)


@dataclass
class DataExtractor:
    symbols: Iterable[str]
    start: datetime | str
    end: datetime | str

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV for each symbol between start and end (inclusive).
        Returns a dict mapping symbol -> pandas.DataFrame with Date index.
        """
        sym_list: List[str] = [s.upper().strip() for s in self.symbols if s and str(s).strip()]
        if not sym_list:
            logger.warning("No symbols provided to DataExtractor")
            return {}

        start_s, end_s = _to_ts(self.start), _to_ts(self.end)
        logger.info("Fetching data for %d symbols from %s to %s", len(sym_list), start_s, end_s)

        out: Dict[str, pd.DataFrame] = {}
        for sym in sym_list:
            try:
                df = yf.download(
                    sym,
                    start=start_s,
                    end=end_s,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    threads=False,
                )
                if df is None or df.empty:
                    logger.warning("No data returned for %s", sym)
                    continue
                # Ensure DatetimeIndex and sorted
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                out[sym] = df
                logger.debug("Fetched %d rows for %s", len(df), sym)
            except Exception as e:  # noqa: BLE001 - log and continue per requirements
                logger.exception("Error fetching %s: %s", sym, e)
        return out

