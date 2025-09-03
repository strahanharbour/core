import polars as pl
from main.engine.strategy import signal_rules
from main.constants import Col

def test_signal_basic():
    df = pl.DataFrame({
        Col.RSI14.value:   [29,31,32],
        Col.MACD_HIST.value:[0.01,0.02,0.03],
        Col.VWAP.value:    [100,100,100],
        Col.CLOSE.value:   [101,101,101],
        Col.VOL_MULT.value:[1.2,1.2,1.2],
    })
    sig = signal_rules(df)
    assert sig.cast(bool).sum() >= 1