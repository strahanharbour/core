from __future__ import annotations
import argparse, json, logging, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

from config_env import load_cfg
from constants import Col
from engine.strategy import signal_rules, position_size

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

# -------- helpers --------
def _load_cfg_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {}) or {}
    data_dir = Path(paths.get("data_dir", "src/main/artifacts/local_data"))
    feat_dir = Path(paths.get("features_dir", "src/main/artifacts/features"))
    results_dir = Path(paths.get("results_dir", "src/main/artifacts/results"))
    return {"data": data_dir, "features": feat_dir, "results": results_dir}

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

def _maybe_import_real_oms():
    """Try to use your project's OMS/OrderManager if available; else PaperOMS."""
    try:
        # Attempt to import an existing OMS adapter
        from order_manager import OrderManager  # type: ignore
        return ("om", OrderManager)  # your code can adapt constructor below
    except Exception:
        return ("paper", PaperOMS)

# -------- main run --------
def run(asof: Optional[str] = None, dry_run: bool = False) -> None:
    meta = load_cfg()
    cfg  = meta.get("cfg", {}) or {}
    paths = _load_cfg_paths(cfg)
    _setup_logging(paths["results"])

    universe = cfg.get("universe", ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA"])

    oms_kind, OMSClass = _maybe_import_real_oms()
    state_dir = paths["results"] / "state"
    oms = OMSClass(state_dir) if oms_kind == "paper" else OMSClass()  # adjust if your OrderManager needs args
    logging.info(f"Entrypoint start | OMS={oms_kind} | dry_run={dry_run}")

    for sym in universe:
        df = _latest_joined(sym, paths["features"], paths["data"])
        if df is None or df.height < 2:
            continue

        sig = signal_rules(df)
        entries = _strict_entry_edge(sig)

        # last two rows for edge
        last_idx = df.height - 1
        enter = bool(entries.tail(1).item())

        last_close = float(df[Col.CLOSE.value].tail(1).item())
        if enter:
            qty = _choose_qty(df)
            if qty <= 0:
                logging.info(f"{sym}: signal but qty=0 (skip)")
                continue
            order = PaperOrder(symbol=sym, side="BUY", qty=qty, price=last_close, note="entry-on-signal")
            if dry_run:
                logging.info(f"DRY RUN order: {order}")
            else:
                if oms_kind == "paper":
                    oms.place(order)  # PaperOMS
                else:
                    # Adapt to your real OMS API if present
                    try:
                        oms.place_order(symbol=sym, side="BUY", quantity=qty, price=last_close)  # type: ignore
                    except Exception:
                        # last resort: log
                        logging.info(f"{sym}: real OMS not wired; logging only: {order}")
        else:
            logging.info(f"{sym}: no entry")

    logging.info("Entrypoint complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algo trading entrypoint (paper mode)")
    parser.add_argument("--asof", type=str, default=None, help="Optional YYYY-MM-DD (not used yet; placeholder)")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without placing orders")
    args = parser.parse_args()
    run(asof=args.asof, dry_run=args.dry_run)

