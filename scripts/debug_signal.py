import polars as pl
from main.engine.strategy import signal_rules
from main.constants import Col

df = pl.DataFrame({
    Col.RSI14.value:   [29,31,32],
    Col.MACD_HIST.value:[0.01,0.02,0.03],
    Col.VWAP.value:    [100,100,100],
    Col.CLOSE.value:   [101,101,101],
    Col.VOL_MULT.value:[1.2,1.2,1.2],
})

sig = signal_rules(df)
print("signal:", sig)
print("sum:", sig.cast(bool).sum())

# Manual breakdown of polars branch
import numpy as np
import polars as pl
from main.engine.strategy import _get_strategy_params

p = _get_strategy_params()
rsi_level       = float(p["rsi_cross_level"])
vol_mult_min    = float(p["vol_mult_min"])
macd_hist_floor = float(p["macd_hist_floor"])
k               = int(p["lookback_bars"])
same_bar        = bool(p["require_same_bar"])
require_rising  = bool(p.get("require_macd_rising", True))
slack           = 1.0 - (float(p.get("price_slack_bps", 0.0)) / 10_000.0)

rsi  = pl.col(Col.RSI14.value)
hist = pl.col(Col.MACD_HIST.value)
close= pl.col(Col.CLOSE.value)
vwap = pl.col(Col.VWAP.value)
volm = pl.col(Col.VOL_MULT.value)

cross_up = (rsi.shift(1) < rsi_level) & (rsi >= rsi_level)
macd_ok  = (hist > macd_hist_floor) & (hist > hist.shift(1)) if require_rising else (hist > macd_hist_floor)
price_ok = close >= (vwap * slack)
vol_ok   = volm >= vol_mult_min

df2 = df.with_row_index("_row")
df2 = df.with_row_index("_row").with_columns([
    cross_up.alias("_cross_up"),
    macd_ok.alias("_macd_ok"),
    price_ok.alias("_price_ok"),
    vol_ok.alias("_vol_ok"),
])
if same_bar:
    base_expr = pl.col("_cross_up") & pl.col("_macd_ok") & pl.col("_price_ok") & pl.col("_vol_ok")
else:
    cross_up_k = pl.col("_cross_up").cast(pl.Int8).rolling_max(window_size=k).cast(pl.Boolean)
    macd_ok_k  = pl.col("_macd_ok").cast(pl.Int8).rolling_max(window_size=k).cast(pl.Boolean)
    price_ok_k = pl.col("_price_ok").cast(pl.Int8).rolling_max(window_size=k).cast(pl.Boolean)
    vol_ok_k   = pl.col("_vol_ok").cast(pl.Int8).rolling_max(window_size=k).cast(pl.Boolean)
    base_expr = cross_up_k & macd_ok_k & price_ok_k & vol_ok_k

n = df2.height
warm = min(max(30, k), max(0, n - 1))
expr = pl.when(pl.col("_row") >= warm).then(base_expr).otherwise(pl.lit(False)).alias("signal")
dbg = df2.select([
    rsi.alias('rsi'), hist.alias('hist'), close.alias('close'), vwap.alias('vwap'), volm.alias('volm'),
    pl.col('_cross_up').alias('cross_up'), pl.col('_macd_ok').alias('macd_ok'),
    pl.col('_price_ok').alias('price_ok'), pl.col('_vol_ok').alias('vol_ok'),
    cross_up_k.alias('cross_up_k'), macd_ok_k.alias('macd_ok_k'), price_ok_k.alias('price_ok_k'), vol_ok_k.alias('vol_ok_k'),
    base_expr.alias('base'), (pl.col('_row') >= warm).alias('warm_mask'), expr
])
for row in dbg.to_dicts():
    print(row)
