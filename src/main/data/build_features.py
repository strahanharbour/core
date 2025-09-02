from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import polars as pl

from config_env import load_cfg


logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "rsi14",
    "atr14",
    "macd",
    "macd_signal",
    "macd_hist",
    "vwap",
    "vol_mult",
]


def _compute_features(df: pl.DataFrame) -> pl.DataFrame:
    # Expect columns: date, Open, High, Low, Close, Volume, symbol
    required = {"date", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort("date")

    # RSI(14) using Wilder smoothing via ewm_mean(alpha=1/14)
    delta = (pl.col("Close") - pl.col("Close").shift(1)).alias("delta")
    gain = pl.when(delta > 0).then(delta).otherwise(0.0).alias("gain")
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0).alias("loss")
    avg_gain = pl.col("gain").ewm_mean(alpha=1 / 14, adjust=False).alias("avg_gain")
    avg_loss = pl.col("loss").ewm_mean(alpha=1 / 14, adjust=False).alias("avg_loss")
    rs = (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs")
    rsi14 = (100 - (100 / (1 + pl.col("rs")))).alias("rsi14")

    # ATR(14) via TR ewm_mean(alpha=1/14)
    prev_close = pl.col("Close").shift(1)
    tr = pl.max_horizontal([
        (pl.col("High") - pl.col("Low")).abs(),
        (pl.col("High") - prev_close).abs(),
        (pl.col("Low") - prev_close).abs(),
    ]).alias("tr")
    atr14 = pl.col("tr").ewm_mean(alpha=1 / 14, adjust=False).alias("atr14")

    # MACD (12,26,9)
    ema12 = pl.col("Close").ewm_mean(span=12, adjust=False).alias("ema12")
    ema26 = pl.col("Close").ewm_mean(span=26, adjust=False).alias("ema26")
    macd = (pl.col("ema12") - pl.col("ema26")).alias("macd")
    macd_signal = pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
    macd_hist = (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")

    # VWAP (20-day rolling using typical price)
    tp = ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3.0).alias("tp")
    pv_roll = (pl.col("tp") * pl.col("Volume")).rolling_sum(window_size=20, min_periods=1)
    v_roll = pl.col("Volume").rolling_sum(window_size=20, min_periods=1)
    vwap = pl.when(v_roll == 0).then(None).otherwise(pv_roll / v_roll).alias("vwap")

    # Volume multiplier vs 20-day mean
    vol_mult = (pl.col("Volume") / pl.col("Volume").rolling_mean(window_size=20, min_periods=1)).alias("vol_mult")

    out = (
        df.with_columns([
            delta,
            gain,
            loss,
            avg_gain,
            avg_loss,
            rs,
            rsi14,
            tr,
            atr14,
            ema12,
            ema26,
            macd,
            macd_signal,
            macd_hist,
            tp,
            vwap,
            vol_mult,
        ])
        .select(["date", pl.col("symbol").cast(pl.Utf8).alias("symbol")] + [pl.col(c) for c in FEATURE_COLS])
    )

    # Fail fast if any feature column is entirely null
    null_counts = out.select([pl.col(c).is_null().sum().alias(c) for c in FEATURE_COLS])
    counts = null_counts.row(0)
    height = out.height
    for col, nnull in zip(FEATURE_COLS, counts):
        if nnull >= height:
            raise RuntimeError(f"Feature column {col} is entirely null; aborting")

    return out


def _symbol_from_df(df: pl.DataFrame, fallback: str) -> str:
    if "symbol" in df.columns:
        vals = df.get_column("symbol").drop_nans().drop_nulls().unique().to_list()
        if len(vals) == 1 and isinstance(vals[0], str) and vals[0].strip():
            return vals[0].strip().upper()
    # fallback from filename
    sym = fallback.upper()
    if sym.endswith("_FEATURES"):
        sym = sym[:-9]
    return sym


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    meta = load_cfg()
    paths_abs: Dict[str, str] = meta["paths_abs"]

    data_dir = Path(paths_abs["data_dir"])  # absolute
    feat_dir = Path(paths_abs["features_dir"])  # absolute
    feat_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        logger.warning("No parquet files found in %s", data_dir)
        return

    for fp in files:
        try:
            df = pl.read_parquet(fp)
            sym = _symbol_from_df(df, fp.stem)
            feats = _compute_features(df)
            # Log simple counts
            nn = feats.select([pl.len().alias("rows")] + [pl.col(c).is_not_null().sum().alias(c) for c in FEATURE_COLS])
            logger.info("%s: %s", sym, nn.to_dicts()[0])
            out_path = feat_dir / f"{sym}_features.parquet"
            feats.write_parquet(out_path)
            logger.info("Wrote %s", out_path)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to build features for %s: %s", fp.name, e)


if __name__ == "__main__":
    main()

