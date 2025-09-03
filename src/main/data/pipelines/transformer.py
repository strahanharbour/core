from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DataTransformer:
    """
    Minimal transformer matching the reference behavior:
      - reset index
      - stringify Date column
      - drop Dividends/Splits columns if present
    """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        out = df.reset_index()
        # Ensure a 'date' column exists after reset_index
        if "Date" in out.columns:
            out = out.rename(columns={"Date": "date"})
        else:
            # fallback: assume first column is the datetime
            first_col = out.columns[0]
            out = out.rename(columns={first_col: "date"})
        # Stringify date for SQLite compatibility
        if pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        else:
            out["date"] = out["date"].astype(str)

        # Drop dividend/split columns if present
        drop_cols = [c for c in ("Dividends", "Stock Splits", "Splits") if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        # Rename OHLCV to camelCase
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjClose",
            "Adj_Close": "adjClose",
            "Volume": "volume",
        }
        out = out.rename({k: v for k, v in rename_map.items() if k in out.columns})

        return out


__all__ = ["DataTransformer"]
