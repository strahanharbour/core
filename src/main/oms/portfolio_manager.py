from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl

from .db_manager import DBManager


@dataclass
class PortfolioManager:
    db: DBManager

    def positions(self) -> List[dict]:
        return self.db.get_positions()

    def total_exposure_usd(self, prices: Optional[Dict[str, float]] = None) -> float:
        expo = 0.0
        for p in self.positions():
            sym = p["symbol"].upper()
            qty = float(p["qty"]) or 0.0
            ref_price = float(prices.get(sym)) if prices and sym in prices else float(p["avg_price"]) or 0.0
            expo += abs(qty * ref_price)
        return expo

    def realized_pnl(self) -> float:
        total = 0.0
        for p in self.positions():
            total += float(p.get("realized_pnl") or 0.0)
        return total

    def mtm(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = 0.0
        unreal = 0.0
        for p in self.positions():
            sym = p["symbol"].upper()
            qty = float(p["qty"]) or 0.0
            avg = float(p["avg_price"]) or 0.0
            px = float(prices.get(sym, avg))
            eq += qty * px
            unreal += qty * (px - avg)
        return {"equity": eq, "unrealized_pnl": unreal, "realized_pnl": self.realized_pnl()}


__all__ = ["PortfolioManager"]
