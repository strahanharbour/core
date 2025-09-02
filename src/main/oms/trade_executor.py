from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

from oms.db_manager import DBManager
from engine.execution.policy import slippage_estimate


def _seed_maybe() -> None:
    if os.getenv("DETERMINISTIC_EXEC", "0") == "1":
        random.seed(42)


@dataclass
class TradeExecutor:
    db: DBManager

    def __post_init__(self) -> None:
        _seed_maybe()

    def execute(self, order_id: int) -> Optional[int]:
        """
        Random-fill executor: partially or fully fills an order with small random price impact.
        Returns trade_id if any fill happened, else None.
        """
        row = self.db.get_order(order_id)
        if row is None:
            return None
        symbol = row["symbol"].upper()
        side = row["side"].upper()
        qty = int(row["qty"]) or 0
        ref_price = float(row["price"]) or 0.0
        if qty <= 0 or ref_price <= 0:
            self.db.update_order_status(order_id, "REJECTED")
            return None

        # Random partial fill fraction between 50% and 100%
        frac = 0.5 + random.random() * 0.5
        fill_qty = max(1, int(round(qty * frac)))

        # Random price impact around slippage estimate
        base_slip = slippage_estimate(ref_price)
        jitter = random.gauss(0.0, base_slip / 3.0)
        eps = max(0.0, base_slip + jitter)
        if side == "BUY":
            fill_price = ref_price * (1.0 + eps)
        else:
            fill_price = ref_price * (1.0 - eps)

        trade_id = self.db.insert_trade_on_fill(order_id, symbol=symbol, qty=fill_qty, entry_price=fill_price)
        self.db.update_order_status(order_id, "FILLED")
        return trade_id


__all__ = ["TradeExecutor"]

