from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_path_from_url(db_url: Optional[str]) -> Path:
    # Expect formats like sqlite:///file.db or a plain path
    if not db_url:
        return Path("portfolio.db").resolve()
    if db_url.startswith("sqlite:///"):
        return Path(db_url.replace("sqlite:///", "")).resolve()
    # Fallback to plain path
    return Path(db_url).resolve()


@dataclass
class DBManager:
    db_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.db_path = _sqlite_path_from_url(self.db_url or os.getenv("DB_URL"))
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self.create_tables()

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def create_tables(self) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
                    qty INTEGER NOT NULL CHECK (qty >= 0),
                    price REAL NOT NULL,
                    sl REAL,
                    tp REAL,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    note TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry REAL NOT NULL,
                    exit REAL,
                    pnl REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    FOREIGN KEY(order_id) REFERENCES orders(id)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    qty INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                );
                """
            )

    # Orders
    def insert_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        *,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        status: str = "NEW",
        note: Optional[str] = None,
    ) -> int:
        ts = _utcnow()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO orders (symbol, side, qty, price, sl, tp, status, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol.upper(), side, int(qty), float(price), sl, tp, status, note, ts, ts),
            )
            return int(cur.lastrowid)

    def update_order_status(self, order_id: int, status: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                "UPDATE orders SET status = ?, updated_at = ? WHERE id = ?",
                (status, _utcnow(), int(order_id)),
            )

    def get_order(self, order_id: int) -> Optional[sqlite3.Row]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE id = ?", (int(order_id),))
            row = cur.fetchone()
            return row

    # Trades
    def insert_trade_on_fill(
        self, order_id: int, *, symbol: str, qty: int, entry_price: float
    ) -> int:
        ts = _utcnow()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trades (order_id, symbol, qty, entry, pnl, entry_time, status)
                VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
                """,
                (int(order_id), symbol.upper(), int(qty), float(entry_price), 0.0, ts),
            )
            trade_id = int(cur.lastrowid)
        # Update portfolio average price and qty
        self._portfolio_apply_fill(symbol, qty, entry_price)
        return trade_id

    def close_trade(self, trade_id: int, exit_price: float) -> None:
        with self.cursor() as cur:
            cur.execute("SELECT symbol, qty, entry FROM trades WHERE id = ?", (int(trade_id),))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Trade id {trade_id} not found")
            symbol = row["symbol"]
            qty = int(row["qty"])
            entry = float(row["entry"])
            pnl = (float(exit_price) - entry) * qty
            cur.execute(
                """
                UPDATE trades
                SET exit = ?, pnl = ?, exit_time = ?, status = 'CLOSED'
                WHERE id = ?
                """,
                (float(exit_price), float(pnl), _utcnow(), int(trade_id)),
            )
        # Update portfolio realized PnL and reduce qty
        self._portfolio_apply_close(symbol, qty, exit_price, entry)

    # Portfolio helpers
    def _portfolio_apply_fill(self, symbol: str, qty: int, fill_price: float) -> None:
        symbol = symbol.upper()
        with self.cursor() as cur:
            cur.execute("SELECT qty, avg_price, realized_pnl FROM portfolio WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO portfolio (symbol, qty, avg_price, realized_pnl, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (symbol, int(qty), float(fill_price), 0.0, _utcnow()),
                )
            else:
                old_qty = int(row["qty"]) if row["qty"] is not None else 0
                old_avg = float(row["avg_price"]) if row["avg_price"] is not None else 0.0
                new_qty = old_qty + int(qty)
                if new_qty <= 0:
                    # Flat or shorting not supported in this simple MVP; reset
                    cur.execute(
                        "UPDATE portfolio SET qty = 0, avg_price = 0.0, updated_at = ? WHERE symbol = ?",
                        (_utcnow(), symbol),
                    )
                else:
                    new_avg = (old_avg * old_qty + float(fill_price) * int(qty)) / new_qty
                    cur.execute(
                        "UPDATE portfolio SET qty = ?, avg_price = ?, updated_at = ? WHERE symbol = ?",
                        (new_qty, new_avg, _utcnow(), symbol),
                    )

    def _portfolio_apply_close(self, symbol: str, qty: int, exit_price: float, entry_price: float) -> None:
        symbol = symbol.upper()
        realized = (float(exit_price) - float(entry_price)) * int(qty)
        with self.cursor() as cur:
            cur.execute("SELECT qty, avg_price, realized_pnl FROM portfolio WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            if row is None:
                return
            old_qty = int(row["qty"]) if row["qty"] is not None else 0
            new_qty = max(0, old_qty - int(qty))
            cur.execute(
                "UPDATE portfolio SET qty = ?, realized_pnl = COALESCE(realized_pnl,0)+?, updated_at = ? WHERE symbol = ?",
                (new_qty, float(realized), _utcnow(), symbol),
            )

    # Queries
    def get_positions(self) -> List[Dict[str, float]]:
        with self.cursor() as cur:
            cur.execute("SELECT symbol, qty, avg_price, realized_pnl, updated_at FROM portfolio WHERE qty <> 0")
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def total_exposure_usd(self) -> float:
        positions = self.get_positions()
        exposure = 0.0
        for p in positions:
            exposure += abs(float(p["qty"]) * float(p["avg_price"]))
        return exposure

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["DBManager"]

