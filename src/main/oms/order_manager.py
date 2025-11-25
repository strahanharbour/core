from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .db_manager import DBManager
from ..engine.risk.manager import RiskManager


@dataclass
class OrderManager:
    db: DBManager
    risk: RiskManager

    def create_order(self, symbol: str, side: str, qty: int, price: float, note: Optional[str] = None) -> Optional[int]:
        symbol = symbol.upper().strip()
        side = side.upper().strip()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        exposure = self.db.total_exposure_usd()
        if not self.risk.check(side=side, qty=qty, price=price, current_exposure_usd=exposure):
            return None

        # SL/TP prices derived from risk settings for BUY side; mirror for SELL
        sl = tp = None
        if side == "BUY":
            sl = price * (1.0 - float(self.risk.stop_loss_pct))
            tp = price * (1.0 + float(self.risk.take_profit_pct))
        else:
            sl = price * (1.0 + float(self.risk.stop_loss_pct))
            tp = price * (1.0 - float(self.risk.take_profit_pct))

        order_id = self.db.insert_order(symbol, side, qty, price, sl=sl, tp=tp, status="NEW", note=note)
        return order_id


__all__ = ["OrderManager"]
