import os
from pathlib import Path

from oms import DBManager, OrderManager, TradeExecutor
from engine.risk.manager import RiskManager


def test_trade_executor_deterministic(tmp_path: Path, monkeypatch):
    # Force deterministic execution
    monkeypatch.setenv("DETERMINISTIC_EXEC", "1")

    db_file = tmp_path / "test.db"
    db = DBManager(db_url=f"sqlite:///{db_file}")

    risk = RiskManager(max_pos_usd=1e9)
    om = OrderManager(db=db, risk=risk)

    # Create order and execute
    qty = 10
    price = 100.0
    oid = om.create_order("AAPL", "BUY", qty, price)
    assert oid is not None

    trade_id = TradeExecutor(db).execute(oid)
    assert trade_id is not None

    # Validate DB effects
    with db.cursor() as cur:
        cur.execute("SELECT status FROM orders WHERE id = ?", (oid,))
        status = cur.fetchone()[0]
        assert status == "FILLED"

        cur.execute("SELECT symbol, qty, entry FROM trades WHERE id = ?", (trade_id,))
        row = cur.fetchone()
        assert row is not None
        symbol, fill_qty, entry = row[0], int(row[1]), float(row[2])
        assert symbol == "AAPL"
        # With seed 42 and qty=10, random fraction yields 8 shares
        assert fill_qty == 8
        assert entry > price  # buy fills above reference with positive slippage

