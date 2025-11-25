from __future__ import annotations
import argparse, json, logging, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Literal, Union, cast, TypedDict

import polars as pl

from main.config_env import load_cfg
from main.constants import Col
from main.engine.strategy import signal_rules, position_size
from main.constants import AllocatorType
from main.utils.paths import get_paths

# -------- logging --------
def _setup_logging(results_dir: Path) -> None:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )

# -------- tiny paper OMS fallback (used if real OMS not available) --------
@dataclass
class PaperOrder:
    symbol: str
    side: str       # 'BUY' or 'SELL'
    qty: int
    price: float
    note: str = ""

class PaperOMS:
    """Minimal local paper broker that logs fills and keeps a simple position file."""
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pos_file = self.state_dir / "positions.json"
        if not self.pos_file.exists():
            self.pos_file.write_text("{}", encoding="utf-8")

    def _load_pos(self) -> Dict[str, int]:
        return json.loads(self.pos_file.read_text(encoding="utf-8") or "{}")

    def _save_pos(self, pos: Dict[str, int]) -> None:
        self.pos_file.write_text(json.dumps(pos, indent=2), encoding="utf-8")

    def place(self, order: PaperOrder) -> Dict[str, Any]:
        pos = self._load_pos()
        cur = int(pos.get(order.symbol, 0))
        if order.side.upper() == "BUY":
            cur += order.qty
        elif order.side.upper() == "SELL":
            cur -= order.qty
        pos[order.symbol] = cur
        self._save_pos(pos)
        fill = {
            "symbol": order.symbol,
            "side": order.side,
            "qty": int(order.qty),
            "price": float(order.price),
            "position_after": int(cur),
            "note": order.note,
        }
        logging.info(f"FILLED (paper): {fill}")
        return fill

# -------- persisted state helpers --------
def _state_file(results_dir: Path, name: str) -> Path:
    sdir = results_dir / "state"
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir / name


def _load_float(p: Path, default: float = 0.0) -> float:
    try:
        return float(p.read_text().strip())
    except Exception:
        return default


def _save_float(p: Path, val: float) -> None:
    p.write_text(f"{float(val):.6f}", encoding="utf-8")


def _estimate_gross_exposure(pos_json: Path, last_prices: dict[str, float]) -> float:
    import json as _json
    if not pos_json.exists():
        return 0.0
    try:
        pos = _json.loads(pos_json.read_text() or "{}")
    except Exception:
        return 0.0
    gross = 0.0
    for sym, qty in pos.items():
        try:
            px = float(last_prices.get(sym, 0.0))
            gross += abs(float(qty)) * px
        except Exception:
            continue
    return gross

# ---- OMS Protocols (typing) ----
@runtime_checkable
class _PaperOMSProto(Protocol):
    def place(self, order: "PaperOrder") -> Dict[str, Any]: ...


@runtime_checkable
class _RealOMSProto(Protocol):
    def create_order(self, symbol: str, side: str, qty: int, price: float, note: Optional[str] = None) -> Optional[int]: ...

# -------- helpers --------
def _load_cfg_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """DEPRECATED: use get_paths(). Kept for backward compat (unused)."""
    p = get_paths()
    return {"data": p.data, "features": p.features, "results": p.results}

def _latest_joined(sym: str, feat_dir: Path, data_dir: Path) -> Optional[pl.DataFrame]:
    f_feat = feat_dir / f"{sym}_features.parquet"
    f_mkt  = data_dir / f"{sym}.parquet"
    if not f_feat.exists() or not f_mkt.exists():
        logging.warning(f"{sym}: missing files: {f_feat.exists()=}, {f_mkt.exists()=}")
        return None
    feats = pl.read_parquet(f_feat)
    mkt   = pl.read_parquet(f_mkt).select([Col.DATE.value, Col.CLOSE.value])
    df = feats.join(mkt, on=Col.DATE.value, how="left").drop_nulls([Col.CLOSE.value])
    if df.is_empty():
        logging.warning(f"{sym}: joined frame empty after drop_nulls")
        return None
    return df

def _strict_entry_edge(sig: pl.Series) -> pl.Series:
    sig_b = sig.cast(pl.Boolean).fill_null(False)
    prev  = sig_b.shift(1).fill_null(False)
    return (sig_b & (~prev))

def _choose_qty(df: pl.DataFrame, risk_dollars: Optional[float] = None) -> int:
    # Prefer ATR if present, else fallback to fixed small size
    atr_col = "atr14" if "atr14" in df.columns else None
    if atr_col:
        # last available ATR
        atr_val = float(df[atr_col].drop_nulls().tail(1).item())
        qty = int(position_size(atr_val, risk_dollars=risk_dollars))
        return max(qty, 0)
    return 1  # ultra-conservative fallback

def _select_positions(candidates, max_positions: int) -> list:
    """Sort candidates by edge desc and take top-N.

    candidates: list of dicts {symbol, edge (float), last_close, df}
    """
    return sorted(candidates, key=lambda x: x["edge"], reverse=True)[: max(0, int(max_positions))]
def _build_oms(results_dir: Path) -> tuple[Literal["om", "paper"], Union[_RealOMSProto, _PaperOMSProto]]:
    """Return an instantiated OMS. Prefer real OrderManager; fallback to PaperOMS.

    This avoids type-checker confusion over constructor signatures.
    """
    # Try canonical package path; fallback to paper OMS if real OMS import fails
    try:
        from main.oms.order_manager import OrderManager  # type: ignore
        from main.oms.db_manager import DBManager  # type: ignore
        from main.engine.risk.manager import RiskManager  # type: ignore
        db = DBManager()
        risk = RiskManager()
        return ("om", cast(_RealOMSProto, OrderManager(db=db, risk=risk)))
    except Exception:
        state_dir = results_dir / "state"
        return ("paper", cast(_PaperOMSProto, PaperOMS(state_dir)))

# -------- main run --------
class Candidate(TypedDict):
    symbol: str
    edge: float
    last_close: float
    df: pl.DataFrame


def run(asof: Optional[str] = None, dry_run: bool = False) -> None:
    meta = load_cfg()
    cfg  = meta.get("cfg", {}) or {}
    # Use absolute, resolved artifact paths rooted under src/main via config_env
    p = get_paths()
    paths: Dict[str, Path] = {"data": p.data, "features": p.features, "results": p.results}
    _setup_logging(paths["results"])

    universe = cfg.get("universe", ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA"])
    # Rails + limits
    risk_cfg = (cfg.get("risk") or {})
    max_positions = int(risk_cfg.get("max_positions", 4))
    daily_limit_pct = float(risk_cfg.get("daily_loss_limit_pct", 2.0))
    max_gross_exposure = float(risk_cfg.get("max_gross_exposure", 0.6))

    # Naive equity tracking via persisted state (placeholder)
    eq_file = _state_file(paths["results"], "equity.txt")
    pnl_file = _state_file(paths["results"], "pnl_today.txt")
    equity = _load_float(eq_file, 10_000.0)
    pnl_today = _load_float(pnl_file, 0.0)
    limit_abs = equity * (daily_limit_pct / 100.0)

    oms_kind, oms = _build_oms(paths["results"])  # already instantiated
    logging.info(f"Entrypoint start | OMS={oms_kind} | dry_run={dry_run}")

    # Collect candidate entries first
    candidates: list[Candidate] = []
    for sym in universe:
        df = _latest_joined(sym, paths["features"], paths["data"])
        if df is None or df.height < 2:
            continue
        sig = signal_rules(df)
        entries = _strict_entry_edge(sig)
        enter = bool(entries.tail(1).item())
        if not enter:
            logging.info(f"{sym}: no entry")
            continue
        last_close = float(df.get_column(Col.CLOSE.value).tail(1).item())
        edge = float(df.get_column(Col.MACD_HIST.value).tail(1).item()) if Col.MACD_HIST.value in df.columns else 0.0
        candidates.append(Candidate(symbol=sym, edge=float(edge), last_close=float(last_close), df=df))

    picks = _select_positions(candidates, max_positions)
    if not picks:
        logging.info("No candidates selected.")
        logging.info("Entrypoint complete.")
        return

    # ---- Enforce daily loss circuit breaker and gross exposure cap ----
    last_prices: dict[str, float] = {str(c["symbol"]): float(c["last_close"]) for c in candidates}
    pos_path = paths["results"] / "state" / "positions.json"
    gross_now = _estimate_gross_exposure(pos_path, last_prices)

    if pnl_today <= -limit_abs:
        logging.warning(
            f"CIRCUIT BREAKER: daily loss {pnl_today:.2f} <= -{limit_abs:.2f} (limit {daily_limit_pct}%). No new orders."
        )
        return

    est_new_exposure = gross_now + sum(float(c["last_close"]) for c in picks)
    if (est_new_exposure / max(equity, 1.0)) > max_gross_exposure:
        logging.warning(
            f"EXPOSURE CAP: est gross {(est_new_exposure/equity):.2%} exceeds {max_gross_exposure:.0%}. Trimming picks."
        )
        tmp: list[Candidate] = []
        running = gross_now
        for c in picks:
            nxt = running + float(c["last_close"])  # type: ignore[index]
            if (nxt / max(equity, 1.0)) <= max_gross_exposure:
                tmp.append(c)
                running = nxt
        picks = tmp
        if not picks:
            logging.warning("No room under exposure cap; skip today.")
            return
    w = 1.0 / len(picks)
    for c in picks:
        sym = str(c["symbol"]).upper()
        df = c["df"]  # type: ignore[assignment]
        qty = _choose_qty(df)  # ATR-based sanity cap
        if qty <= 0:
            logging.info(f"{sym}: skip qty=0")
            continue
        note = f"equal_w={w:.2f}"
        order = PaperOrder(symbol=sym, side="BUY", qty=qty, price=float(c["last_close"]), note=note)
        if dry_run:
            logging.info(f"DRY RUN order: {order}")
        else:
            if oms_kind == "paper":
                cast(_PaperOMSProto, oms).place(order)
            else:
                try:
                    cast(_RealOMSProto, oms).create_order(symbol=sym, side="BUY", qty=qty, price=float(c["last_close"]), note=note)
                except Exception:
                    logging.info(f"{sym}: real OMS not wired; logging only: {order}")

    # Persist current equity / pnl_today placeholders (hook to live PnL later)
    _save_float(eq_file, equity)
    _save_float(pnl_file, pnl_today)

    logging.info("Entrypoint complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo trading entrypoint (paper mode)")
    parser.add_argument("--asof", type=str, default=None, help="Optional YYYY-MM-DD (not used yet; placeholder)")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without placing orders")
    args = parser.parse_args()
    run(asof=args.asof, dry_run=args.dry_run)
