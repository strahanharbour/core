from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
import re

import polars as pl

from config_env import load_cfg
from constants import Col
from utils.columns import canonicalize_polars_columns


logger = logging.getLogger(__name__)


FEATURE_COLS = [
    Col.RSI14.value,
    Col.ATR14.value,
    Col.MACD.value,
    Col.MACD_SIGNAL.value,
    Col.MACD_HIST.value,
    Col.VWAP.value,
    Col.VOL_MULT.value,
]


def _normalize_ohlcv_columns(df: pl.DataFrame) -> pl.DataFrame:
    # Centralized canonicalization using humps
    return canonicalize_polars_columns(df)


def _compute_features(df: pl.DataFrame) -> pl.DataFrame:
    # Expect columns: date, open, high, low, close, volume, symbol
    df = _normalize_ohlcv_columns(df)
    required = {Col.DATE.value, Col.HIGH.value, Col.LOW.value, Col.CLOSE.value, Col.VOLUME.value}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort(Col.DATE.value)

    # RSI(14) using Wilder smoothing via ewm_mean(alpha=1/14)
    # Note: with_columns evaluates expressions in parallel. Compute dependencies sequentially.
    delta = (pl.col(Col.CLOSE.value) - pl.col(Col.CLOSE.value).shift(1)).alias("delta")
    gain_expr = pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain")
    loss_expr = pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss")

    # ATR(14) via TR ewm_mean(alpha=1/14)
    prev_close = pl.col(Col.CLOSE.value).shift(1)
    tr_expr = pl.max_horizontal([
        (pl.col(Col.HIGH.value) - pl.col(Col.LOW.value)).abs(),
        (pl.col(Col.HIGH.value) - prev_close).abs(),
        (pl.col(Col.LOW.value) - prev_close).abs(),
    ]).alias("tr")

    # MACD (12,26,9)
    ema12_expr = pl.col(Col.CLOSE.value).ewm_mean(span=12, adjust=False).alias("ema12")
    ema26_expr = pl.col(Col.CLOSE.value).ewm_mean(span=26, adjust=False).alias("ema26")

    # VWAP (20-day rolling using typical price)
    tp_expr = ((pl.col(Col.HIGH.value) + pl.col(Col.LOW.value) + pl.col(Col.CLOSE.value)) / 3.0).alias("tp")

    # Volume multiplier vs 20-day mean
    # min_periods renamed to min_samples in Polars >= 1.21
    vol_mult_expr = (
        pl.col(Col.VOLUME.value)
        / pl.col(Col.VOLUME.value).rolling_mean(window_size=20, min_samples=1)
    ).alias(Col.VOL_MULT.value)

    # Build up columns in dependency order to avoid ColumnNotFound errors
    out = df
    # 1) RSI components: delta, gain, loss
    out = out.with_columns([delta]).with_columns([gain_expr, loss_expr])
    # 2) Wilder averages and RSI
    out = out.with_columns([
        pl.col("gain").ewm_mean(alpha=1 / 14, adjust=False).alias("avg_gain"),
        pl.col("loss").ewm_mean(alpha=1 / 14, adjust=False).alias("avg_loss"),
    ])
    out = out.with_columns([
        (pl.col("avg_gain") / pl.col("avg_loss")).alias("rs"),
    ])
    out = out.with_columns([
        (100 - (100 / (1 + pl.col("rs")))).alias("rsi14"),
    ])

    # 3) ATR components
    out = out.with_columns([tr_expr])
    out = out.with_columns([pl.col("tr").ewm_mean(alpha=1 / 14, adjust=False).alias("atr14")])

    # 4) MACD components
    out = out.with_columns([ema12_expr, ema26_expr])
    out = out.with_columns([(pl.col("ema12") - pl.col("ema26")).alias(Col.MACD.value)])
    out = out.with_columns([
        pl.col(Col.MACD.value).ewm_mean(span=9, adjust=False).alias(Col.MACD_SIGNAL.value)
    ])
    out = out.with_columns([
        (pl.col(Col.MACD.value) - pl.col(Col.MACD_SIGNAL.value)).alias(Col.MACD_HIST.value)
    ])

    # 5) VWAP components
    out = out.with_columns([tp_expr])
    pv_roll = (pl.col("tp") * pl.col(Col.VOLUME.value)).rolling_sum(window_size=20, min_samples=1)
    v_roll = pl.col(Col.VOLUME.value).rolling_sum(window_size=20, min_samples=1)
    out = out.with_columns([
        pl.when(v_roll == 0)
        .then(None)
        .otherwise(pv_roll / v_roll)
        .alias(Col.VWAP.value)
    ])

    # 6) Volume multiplier
    out = out.with_columns([vol_mult_expr])

    # Final projection: date, symbol, features
    out = out.select(
        [Col.DATE.value, pl.col(Col.SYMBOL.value).cast(pl.Utf8).alias(Col.SYMBOL.value)]
        + [pl.col(c) for c in FEATURE_COLS]
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
    if Col.SYMBOL.value in df.columns:
        vals = df.get_column(Col.SYMBOL.value).drop_nans().drop_nulls().unique().to_list()
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
