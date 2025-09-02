from __future__ import annotations

from typing import Optional


def choose_execution_algo(side: str, qty: int, urgency: Optional[str] = None) -> str:
    """
    Placeholder policy that always picks 'random_fill'.
    """
    return "random_fill"


def slippage_estimate(price: float, vol_mult: Optional[float] = None) -> float:
    """
    Estimate slippage as a fraction of price (e.g., 0.0002 for 2 bps).
    If volume is high (vol_mult >= 1), reduce slippage modestly.
    """
    base = 0.0003  # 3 bps baseline
    if vol_mult is None:
        return base
    return max(0.00005, base * (0.8 if vol_mult >= 1.0 else 1.2))


__all__ = ["choose_execution_algo", "slippage_estimate"]

