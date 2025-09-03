from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import polars as pl
import pandas as pd

from config_env import load_cfg
from data.market.extractor import DataExtractor
from constants import Col
from utils.columns import canonicalize_pandas_columns


logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    meta = load_cfg()
    cfg = meta.get("cfg", {})
    paths_abs: Dict[str, str] = meta.get("paths_abs", {})

    uni = cfg.get("universe", [])
    bt = cfg.get("backtest", {})
    start = bt.get("start")
    end = bt.get("end")

    if not uni:
        logger.error("Universe is empty in config; nothing to ingest")
        return
    if not start or not end:
        logger.error("Backtest.start and/or backtest.end missing in config")
        return

    out_dir = Path(paths_abs.get("data_dir") or Path(__file__).resolve().parent / "artifacts" / "local_data")
    _ensure_dir(out_dir)
    logger.info("Writing symbol parquet files to: %s", out_dir)

    extractor = DataExtractor(symbols=uni, start=start, end=end)
    frames = extractor.fetch_data()

    written = 0
    for sym, pdf in frames.items():
        if pdf is None or pdf.empty:
            logger.warning("Skipping %s: empty frame", sym)
            continue
        # Reset index so Date becomes a column for parquet
        pdf = pdf.reset_index()
        # Canonicalize to camelCase across the board
        pdf = canonicalize_pandas_columns(pdf)
        if Col.DATE.value not in pdf.columns:
            pdf = pdf.rename(columns={pdf.columns[0]: Col.DATE.value})
        pf = pl.from_pandas(pdf)
        pf = pf.with_columns(pl.lit(sym).alias(Col.SYMBOL.value))
        out_path = out_dir / f"{sym}.parquet"
        pf.write_parquet(out_path)
        written += 1
        logger.info("Wrote %s (%d rows)", out_path.name, pf.height)

    logger.info("Done. Wrote %d/%d symbols", written, len(frames))


if __name__ == "__main__":
    main()
