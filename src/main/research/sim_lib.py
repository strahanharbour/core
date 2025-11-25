from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from main.config_env import load_cfg
from main.constants import Col
from main.engine.strategy import signal_rules


def load_cfg_bits():
    cfg = load_cfg()["cfg"]
    paths = cfg.get("paths", {}) or {}
    sentiment = cfg.get("sentiment", {}) or {}
    strat = cfg.get("strategy", {}) or {}
    return cfg, paths, sentiment, strat


def apply_sentiment_gate(
    df: pl.DataFrame,
    sig: pl.Series,
    sym: str,
    features_dir: Path,
    sent_cfg: dict,
) -> pl.Series:
    if not bool(sent_cfg.get("enabled", False)):
        return sig
    sic_dir = features_dir / "sentiment"
    f = sic_dir / f"{sym}_sic.parquet"
    if not f.exists():
        return sig
    sic = pl.read_parquet(f).select(["date", "sic"])
    df2 = df.join(sic, on="date", how="left")
    return sig & (df2["sic"].fill_null(0.0) >= float(sent_cfg.get("sic_threshold", 0.1)))


def apply_market_filter(df: pl.DataFrame, sig: pl.Series, data_dir: Path, strat_cfg: dict) -> pl.Series:
    mf = (strat_cfg.get("market_filter") or {})
    if not bool(mf.get("enabled", False)):
        return sig
    sym = str(mf.get("symbol", "SPY"))
    sma_len = int(mf.get("sma_length", 50))
    f = data_dir / f"{sym}.parquet"
    if not f.exists():
        return sig
    px = pl.read_parquet(f).select([Col.DATE.value, Col.CLOSE.value])
    sma = px.select([Col.DATE.value, pl.col(Col.CLOSE.value).rolling_mean(sma_len).alias("mkt_sma")])
    mkt = (
        px.join(sma, on=Col.DATE.value, how="left")
        .with_columns((pl.col(Col.CLOSE.value) > pl.col("mkt_sma")).alias("mkt_ok"))
        .select([Col.DATE.value, "mkt_ok"])
    )
    df2 = df.join(mkt, on="date", how="left")
    return sig & df2["mkt_ok"].fill_null(False)


def strict_entry_edge(sig: pl.Series) -> pl.Series:
    sig_b = sig.cast(pl.Boolean).fill_null(False)
    prev = sig_b.shift(1).fill_null(False)
    return (sig_b & (~prev)).alias("entry")


def exits_returns(df: pl.DataFrame, entries: pl.Series, strat_cfg: dict) -> Optional[pl.DataFrame]:
    """Return trades DF with columns: entry_date, entry_px, ret_h
    Uses ATR exits if enabled and atr14 exists; else fixed horizon (max_hold_bars or HOLD_BARS).
    """
    ex = (strat_cfg.get("exits") or {})
    max_h = int(ex.get("max_hold_bars", 10))
    df = df.with_columns([entries.alias("entry")])

    if bool(ex.get("enabled", False)) and (Col.ATR14.value in df.columns):
        stop_k = float(ex.get("stop_atr_mult", 2.0))
        take_k = float(ex.get("take_atr_mult", 3.0))
        df = df.with_columns(pl.arange(0, pl.len()).alias("idx"))
        e = df.filter(pl.col("entry")).select([Col.DATE.value, "idx", Col.CLOSE.value, Col.ATR14.value])
        out_rows = []
        for row in e.iter_rows(named=True):
            i0 = int(row["idx"])  # type: ignore[index]
            px0 = float(row[Col.CLOSE.value])  # type: ignore[index]
            atr0 = float(row[Col.ATR14.value])  # type: ignore[index]
            stop = px0 - stop_k * atr0
            take = px0 + take_k * atr0
            i_end = min(df.height - 1, i0 + max_h)
            hit = False
            ret = 0.0
            for i in range(i0 + 1, i_end + 1):
                px = float(df[Col.CLOSE.value][i])
                if px <= stop:
                    ret = (stop / px0) - 1.0
                    hit = True
                    break
                if px >= take:
                    ret = (take / px0) - 1.0
                    hit = True
                    break
            if not hit:
                ret = (float(df[Col.CLOSE.value][i_end]) / px0) - 1.0
            out_rows.append({"entry_date": row[Col.DATE.value], "entry_px": px0, "ret_h": ret})
        return pl.DataFrame(out_rows) if out_rows else None

    # Fixed horizon
    n = df.height
    exit_idx = (pl.arange(0, n) + max_h).clip_upper(n - 1)
    ret = (pl.col(Col.CLOSE.value).gather(exit_idx) / pl.col(Col.CLOSE.value) - 1.0).alias("ret_h")
    out = (
        df.filter(pl.col("entry"))
        .select([Col.DATE.value, Col.CLOSE.value, ret])
        .rename({Col.DATE.value: "entry_date", Col.CLOSE.value: "entry_px"})
    )
    return out


__all__ = [
    "load_cfg_bits",
    "apply_sentiment_gate",
    "apply_market_filter",
    "apply_regime_gate",
    "strict_entry_edge",
    "exits_returns",
    "signal_rules",
]

def apply_regime_gate(df: pl.DataFrame, sig: pl.Series, data_dir: Path, strat_cfg: dict) -> pl.Series:
    rf = (strat_cfg.get("regime_filter") or {})
    if not bool(rf.get("enabled", False)):
        return sig
    from main.research.regime import compute_regime

    sym = str(rf.get("symbol", "SPY"))
    f = data_dir / f"{sym}.parquet"
    if not f.exists():
        return sig
    px = pl.read_parquet(f).select([Col.DATE.value, Col.CLOSE.value])
    reg = compute_regime(
        px,
        int(rf.get("sma_length", 50)),
        int(rf.get("vol_length", 20)),
        float(rf.get("vol_thresh", 0.02)),
    )
    df2 = df.join(reg, on="date", how="left")
    return sig & df2.get_column("bull").fill_null(False)
