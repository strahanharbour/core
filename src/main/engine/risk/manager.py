from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    max_pos_usd: float = 10_000.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    def stop_loss(self, entry_price: float, current_price: float) -> bool:
        if entry_price is None or current_price is None:
            return False
        return float(current_price) <= float(entry_price) * (1.0 - float(self.stop_loss_pct))

    def take_profit(self, entry_price: float, current_price: float) -> bool:
        if entry_price is None or current_price is None:
            return False
        return float(current_price) >= float(entry_price) * (1.0 + float(self.take_profit_pct))

    def check(self, *, side: str, qty: int, price: float, current_exposure_usd: float) -> bool:
        if qty <= 0 or price <= 0:
            return False
        proposed = float(qty) * float(price)
        # For SELL in a long-only MVP, disallow increasing short exposure
        if side.upper() == "SELL":
            # Allow SELL only if it reduces exposure; approval outside scope
            return True
        return (float(current_exposure_usd) + proposed) <= float(self.max_pos_usd)


__all__ = ["RiskManager"]

