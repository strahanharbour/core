from __future__ import annotations

from pathlib import Path
import polars as pl

from constants import Col


def load_market_ok(data_dir: Path, symbol: str, sma_len: int) -> pl.DataFrame | None:
    f = data_dir / f"{symbol}.parquet"
    if not f.exists():
        return None
    px = pl.read_parquet(f).select([Col.DATE.value, Col.CLOSE.value])
    sma = px.select([
        Col.DATE.value,
        pl.col(Col.CLOSE.value).rolling_mean(window_size=int(sma_len), min_samples=1).alias("mkt_sma"),
    ])
    out = (
        px.join(sma, on=Col.DATE.value, how="left")
        .with_columns((pl.col(Col.CLOSE.value) > pl.col("mkt_sma")).alias("mkt_ok"))
        .select([Col.DATE.value, "mkt_ok"])
    )
    return out


__all__ = ["load_market_ok"]

