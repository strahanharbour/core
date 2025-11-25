from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from main.config_env import load_cfg
from main.data.market.extractor import DataExtractor
from main.data.pipelines.transformer import DataTransformer
from main.data.pipelines.loader_sqlite import DataLoader
from main.engine.risk.manager import RiskManager
from main.oms import DBManager, OrderManager, TradeExecutor, PortfolioManager
from main.constants import Col


START = "2022-01-01"
END = "2025-08-31"


logger = logging.getLogger(__name__)


def run_etl_and_demo_orders() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    meta = load_cfg()
    cfg = meta.get("cfg", {})

    uni: List[str] = [str(s).upper() for s in cfg.get("universe", [])]
    if not uni:
        logger.error("Universe is empty; aborting")
        return

    logger.info("Extracting market data for %d symbols (%s to %s)", len(uni), START, END)
    extractor = DataExtractor(symbols=uni, start=START, end=END)
    raw = extractor.fetch_data()

    xform = DataTransformer()
    loader = DataLoader()

    for sym, pdf in raw.items():
        if pdf is None or pdf.empty:
            logger.warning("%s: no data to load", sym)
            continue
        clean = xform.clean_data(pdf)
        n = loader.load_dataframe(sym, clean)
        logger.info("%s: loaded %d rows into SQLite", sym, n)

    # Demo OMS flow: create a small order per symbol at last close and execute
    db = DBManager()
    risk = RiskManager(max_pos_usd=10_000.0, stop_loss_pct=0.05, take_profit_pct=0.10)
    om = OrderManager(db=db, risk=risk)
    execu = TradeExecutor(db=db)
    pm = PortfolioManager(db=db)

    for sym in uni:
        df = raw.get(sym)
        if df is None or df.empty:
            continue
        clean = xform.clean_data(df)
        last_close = float(clean[Col.CLOSE.value].dropna().iloc[-1])
        qty = 1  # simple demo sizing
        oid = om.create_order(sym, "BUY", qty, last_close, note="etl demo")
        if oid:
            tid = execu.execute(oid)
            logger.info("%s: created order %s trade %s", sym, oid, tid)
        else:
            logger.info("%s: risk check failed; no order", sym)

    # Portfolio metrics
    pos = pm.positions()
    expo = pm.total_exposure_usd()
    pnl = pm.realized_pnl()
    logger.info("Portfolio positions: %s", pos)
    logger.info("Total exposure USD: %.2f | Realized PnL: %.2f", expo, pnl)
    print({"positions": len(pos), "exposure_usd": round(expo, 2), "realized_pnl": round(pnl, 2)})


if __name__ == "__main__":
    run_etl_and_demo_orders()
