from __future__ import annotations

from pathlib import Path
import polars as pl

from main.constants import Col


def compute_regime(
    px: pl.DataFrame,
    sma_len: int = 50,
    vol_len: int = 20,
    vol_thresh: float = 0.02,
) -> pl.DataFrame:
    """Classify daily bull/non-bull regime using SMA trend and realized volatility.

    Returns a DataFrame with columns: date (pl.Date), bull (bool).
    """
    px = px.sort(Col.DATE.value)
    sma = px.select(
        [Col.DATE.value, pl.col(Col.CLOSE.value).rolling_mean(sma_len).alias("sma")]
    )
    ret = px.select(
        [
            Col.DATE.value,
            (pl.col(Col.CLOSE.value) / pl.col(Col.CLOSE.value).shift(1) - 1.0).alias("ret"),
        ]
    )
    vol = ret.select([Col.DATE.value, pl.col("ret").rolling_std(vol_len).alias("vol")])
    df = (
        px.join(sma, on=Col.DATE.value, how="left")
        .join(vol, on=Col.DATE.value, how="left")
        .with_columns(
            [
                (pl.col(Col.CLOSE.value) > pl.col("sma")).alias("trend_up"),
                (pl.col("vol").fill_null(0.0) < float(vol_thresh)).alias("low_vol"),
            ]
        )
    )
    return (
        df.select(
            [
                pl.col(Col.DATE.value).alias("date"),
                (pl.col("trend_up") & pl.col("low_vol")).alias("bull"),
            ]
        )
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
    )


__all__ = ["compute_regime"]
