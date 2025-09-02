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
        # Ensure a 'Date' column exists after reset_index
        if "Date" not in out.columns:
            # fallback: assume first column is the datetime
            first_col = out.columns[0]
            out = out.rename(columns={first_col: "Date"})
        # Stringify Date for SQLite compatibility
        if pd.api.types.is_datetime64_any_dtype(out["Date"]):
            out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        else:
            out["Date"] = out["Date"].astype(str)

        # Drop dividend/split columns if present
        drop_cols = [c for c in ("Dividends", "Stock Splits", "Splits") if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        return out


__all__ = ["DataTransformer"]

