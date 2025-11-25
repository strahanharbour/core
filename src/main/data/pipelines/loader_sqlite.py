from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from main.config_env import load_cfg


def _sqlite_path() -> Path:
    meta = load_cfg()
    db_url = (meta.get("env") or {}).get("DB_URL")
    if db_url and db_url.startswith("sqlite:///"):
        return Path(db_url.replace("sqlite:///", "")).resolve()
    if db_url:
        return Path(db_url).resolve()
    return Path("portfolio.db").resolve()


@dataclass
class DataLoader:
    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or _sqlite_path()
        self.conn = sqlite3.connect(self.db_path)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _table_name(self, symbol: str) -> str:
        return f"bars_{symbol.upper()}"

    def create_table_if_not_exists(self, symbol: str) -> None:
        t = self._table_name(symbol)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER
            );
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_date ON {t}(date);")
        self.conn.commit()
        cur.close()

    def load_dataframe(self, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        self.create_table_if_not_exists(symbol)
        t = self._table_name(symbol)

        # Normalize columns
        colmap = {
            # Legacy
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Adj_Close": "adj_close",
            "Volume": "volume",
            # Canonical camelCase
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "adj_close",
            "volume": "volume",
        }
        use = {colmap[c]: df[c] for c in df.columns if c in colmap}
        rows = pd.DataFrame(use).to_records(index=False)

        cur = self.conn.cursor()
        cur.executemany(
            f"INSERT INTO {t} (date, open, high, low, close, adj_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tuple(r) for r in rows],
        )
        n = cur.rowcount
        self.conn.commit()
        cur.close()
        return n


__all__ = ["DataLoader"]
