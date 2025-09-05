from __future__ import annotations
import itertools
from dataclasses import dataclass
import hashlib
from datetime import datetime
try:
    import plotly.graph_objs as go
except ImportError:
    go = None
from pathlib import Path
from typing import Any, Dict, Iterable, List

import polars as pl

from config_env import load_cfg
from constants import Col

# ---- What we sweep (keep simple/orthogonal) ----
DEFAULT_GRID = {
    "rsi_cross_level": [25.0, 30.0, 35.0],
    "vol_mult_min":    [1.0, 1.1, 1.2],
    "macd_hist_floor": [0.0],
}

# Exit horizon for quick scoring (not part of runtime)
HOLD_BARS = 10
MIN_WARM  = 30

@dataclass(frozen=True)
class Params:
    rsi_cross_level: float
    vol_mult_min: float
    macd_hist_floor: float

def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Params]:
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield Params(**dict(zip(keys, combo)))


def _label_for_params(p: Params) -> str:
    s = f"rsi={p.rsi_cross_level}|vol={p.vol_mult_min}|macd={p.macd_hist_floor}"
    h = hashlib.md5(s.encode()).hexdigest()[:6]
    return f"{s}#{h}"

# --------- Sweep-only knobs pulled from config.strategy.sweep ---------
@dataclass(frozen=True)
class SweepKnobs:
    lookback_bars: int
    require_same_bar: bool
    require_macd_rising: bool
    price_slack_bps: float
    entry_edge_lookback: int

def _load_knobs() -> SweepKnobs:
    meta = load_cfg()
    cfg = meta.get("cfg", {}) or {}
    sweep = ((cfg.get("strategy") or {}).get("sweep") or {})
    return SweepKnobs(
        lookback_bars       = int(sweep.get("lookback_bars", 5)),
        require_same_bar    = bool(sweep.get("require_same_bar", False)),
        require_macd_rising = bool(sweep.get("require_macd_rising", True)),
        price_slack_bps     = float(sweep.get("price_slack_bps", 0.0)),
        entry_edge_lookback = int(sweep.get("entry_edge_lookback", 3)),
    )

# --------- Local signal builder for sweeps (production stays strict) ---------
def _signal_with_knobs(df: pl.DataFrame, p: Params, k: SweepKnobs) -> pl.Series:
    """Build signal using sweep knobs (does NOT touch runtime engine.strategy)."""
    need = [Col.RSI14.value, Col.MACD_HIST.value, Col.VWAP.value, Col.VOL_MULT.value, Col.CLOSE.value]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    rsi   = pl.col(Col.RSI14.value)
    hist  = pl.col(Col.MACD_HIST.value)
    close = pl.col(Col.CLOSE.value)
    vwap  = pl.col(Col.VWAP.value)
    volm  = pl.col(Col.VOL_MULT.value)

    cross_up = (rsi.shift(1) < p.rsi_cross_level) & (rsi >= p.rsi_cross_level)
    if k.require_macd_rising:
        macd_ok = (hist > p.macd_hist_floor) & (hist > hist.shift(1))
    else:
        macd_ok = (hist > p.macd_hist_floor)

    slack = 1.0 - (k.price_slack_bps / 10_000.0)
    price_ok = close >= (vwap * slack)
    vol_ok   = volm >= p.vol_mult_min

    if k.require_same_bar:
        base = cross_up & macd_ok & price_ok & vol_ok
    else:
        L = max(1, int(k.lookback_bars))
        base = (
            cross_up.rolling_max(L).cast(pl.Boolean) &
            macd_ok.rolling_max(L).cast(pl.Boolean) &
            price_ok.rolling_max(L).cast(pl.Boolean) &
            vol_ok.rolling_max(L).cast(pl.Boolean)
        )

    warm = max(MIN_WARM, int(k.lookback_bars))
    idx  = pl.arange(0, df.height)
    signal_expr = (base & (idx >= warm)).alias("signal")
    return df.select(signal_expr)["signal"]

def _entries_from_signal(sig: pl.Series, k_edge: int) -> pl.Series:
    sig_b = sig.cast(pl.Boolean).fill_null(False)
    prev_any = sig_b.shift(1).rolling_max(window_size=max(1, k_edge), min_samples=1).cast(pl.Boolean).fill_null(False)
    return (sig_b & (~prev_any)).alias("entry")


@dataclass(frozen=True)
class MarketFilter:
    enabled: bool
    symbol: str
    sma_length: int


def _load_market_filter() -> MarketFilter:
    cfg = load_cfg()["cfg"]
    sweep = ((cfg.get("strategy") or {}).get("sweep") or {})
    mf = (sweep.get("market_filter") or {})
    return MarketFilter(
        enabled=bool(mf.get("enabled", False)),
        symbol=str(mf.get("symbol", "SPY")),
        sma_length=int(mf.get("sma_length", 50)),
    )


def _market_filter_series(mf: MarketFilter, data_dir: Path) -> pl.DataFrame | None:
    if not mf.enabled:
        return None
    f = data_dir / f"{mf.symbol}.parquet"
    if not f.exists():
        return None
    px = pl.read_parquet(f).select([Col.DATE.value, Col.CLOSE.value])
    sma_df = px.with_columns(
        pl.col(Col.CLOSE.value).rolling_mean(window_size=mf.sma_length, min_samples=1).alias("mkt_sma")
    )
    filt = sma_df.select([
        Col.DATE.value,
        (pl.col(Col.CLOSE.value) > pl.col("mkt_sma")).alias("mkt_ok"),
    ])
    return filt

@dataclass(frozen=True)
class ExitKnobs:
    enabled: bool
    stop_atr_mult: float
    take_atr_mult: float
    max_hold_bars: int


def _load_exit_knobs() -> ExitKnobs:
    cfg = load_cfg()["cfg"]
    sweep = ((cfg.get("strategy") or {}).get("sweep") or {})
    ex = (sweep.get("exits") or {})
    return ExitKnobs(
        enabled=bool(ex.get("enabled", False)),
        stop_atr_mult=float(ex.get("stop_atr_mult", 2.0)),
        take_atr_mult=float(ex.get("take_atr_mult", 3.0)),
        max_hold_bars=int(ex.get("max_hold_bars", HOLD_BARS)),
    )


def _eval_symbol(sym: str, p: Params, knobs: SweepKnobs, data_dir: Path, feat_dir: Path) -> pl.DataFrame | None:
    f_feat = feat_dir / f"{sym}_features.parquet"
    f_data = data_dir / f"{sym}.parquet"
    if not f_feat.exists() or not f_data.exists():
        return None

    feats = pl.read_parquet(f_feat)
    data  = pl.read_parquet(f_data).select([Col.DATE.value, Col.CLOSE.value])
    df = feats.join(data, on=Col.DATE.value, how="left").drop_nulls([Col.CLOSE.value])

    sig = _signal_with_knobs(df, p, knobs)
    # Apply market filter if configured
    mf = _load_market_filter()
    mkt = _market_filter_series(mf, data_dir)
    if mkt is not None:
        df = df.join(mkt, on=Col.DATE.value, how="left")
        sig = (sig & df.get_column("mkt_ok").fill_null(False))
    ent = _entries_from_signal(sig, knobs.entry_edge_lookback)
    exits = _load_exit_knobs()

    df = df.with_columns([sig.alias("signal"), ent.alias("entry")])

    if exits.enabled and ("atr14" in df.columns):
        # Simulate per-entry stop/take over next N bars
        idx = pl.arange(0, df.height).alias("idx")
        df = df.with_columns(idx)
        entries_df = df.filter(pl.col("entry")).select([Col.DATE.value, "idx", Col.CLOSE.value, "atr14"])
        out_rows: list[dict[str, Any]] = []
        for row in entries_df.iter_rows(named=True):
            i0 = int(row["idx"])  # type: ignore[index]
            px0 = float(row[Col.CLOSE.value])  # type: ignore[index]
            atr0 = float(row["atr14"])  # type: ignore[index]
            stop = px0 - exits.stop_atr_mult * atr0
            take = px0 + exits.take_atr_mult * atr0
            i_end = min(df.height - 1, i0 + exits.max_hold_bars)
            ret = 0.0
            hit = False
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
                px_exit = float(df[Col.CLOSE.value][i_end])
                ret = (px_exit / px0) - 1.0
            out_rows.append({
                "symbol": sym,
                "entry_date": row[Col.DATE.value],
                "entry_px": px0,
                "ret_h": ret,
            })
        return pl.DataFrame(out_rows) if out_rows else None
    else:
        # Fallback fixed horizon
        n = df.height
        exit_idx = (pl.arange(0, n) + HOLD_BARS).clip(0, n - 1)
        ret = (pl.col(Col.CLOSE.value).gather(exit_idx) / pl.col(Col.CLOSE.value) - 1.0).alias("ret_h")
        out = df.filter(pl.col("entry")).select([
            pl.lit(sym).alias("symbol"),
            pl.col(Col.DATE.value).alias("entry_date"),
            pl.col(Col.CLOSE.value).alias("entry_px"),
            ret,
        ])
        return out


def _equity_curve_for_combo(p: Params, knobs: SweepKnobs, universe: list[str], data_dir: Path, feat_dir: Path) -> pl.DataFrame:
    per_sym = []
    for sym in universe:
        r = _eval_symbol(sym, p, knobs, data_dir, feat_dir)
        if r is not None and r.height > 0:
            per_sym.append(r)
    if not per_sym:
        return pl.DataFrame({"date": [], "equity": [], "label": []})

    trades = pl.concat(per_sym, how="vertical_relaxed").sort("entry_date")
    if trades.is_empty():
        return pl.DataFrame({"date": [], "equity": [], "label": []})

    start = trades["entry_date"].min()
    end = trades["entry_date"].max()
    cal = pl.date_range(start=start, end=end, interval="1d", eager=True).alias("date")
    daily = pl.DataFrame({"date": cal})

    growth = trades.group_by("entry_date").agg(pl.sum("ret_h").alias("sum_ret")).with_columns(
        (1.0 + pl.col("sum_ret")).alias("gf")
    ).select(pl.col("entry_date").alias("date"), "gf")

    curve = daily.join(growth, on="date", how="left").with_columns(
        pl.when(pl.col("gf").is_null()).then(1.0).otherwise(pl.col("gf")).alias("gf")
    )
    curve = curve.with_columns(
        pl.cum_prod("gf").alias("equity")
    ).select(["date", "equity"])
    curve = curve.with_columns(pl.lit(_label_for_params(p)).alias("label"))
    return curve

def main():
    meta  = load_cfg()
    cfg   = meta.get("cfg", {}) or {}
    paths_abs = meta.get("paths_abs", {}) or {}

    # Use resolved absolute paths (consistent with other scripts)
    base = Path(__file__).resolve().parents[2]
    data_dir = Path(paths_abs.get("data_dir") or base / "main" / "artifacts" / "local_data")
    feat_dir = Path(paths_abs.get("features_dir") or base / "main" / "artifacts" / "features")
    results_dir = Path(paths_abs.get("results_dir") or base / "main" / "artifacts" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    universe = cfg.get("universe", ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA"])
    knobs    = _load_knobs()

    rows = []
    for p in _iter_grid(DEFAULT_GRID):
        per_sym = []
        for sym in universe:
            r = _eval_symbol(sym, p, knobs, data_dir, feat_dir)
            if r is not None and r.height > 0:
                per_sym.append(r)
        if per_sym:
            all_tr = pl.concat(per_sym, how="vertical_relaxed")
            trades = int(all_tr.height)
            win    = float((all_tr["ret_h"] > 0).mean())
            avg    = float(all_tr["ret_h"].mean())
            std    = float(all_tr["ret_h"].std(ddof=1) or 0.0)
            sharpe = (avg / std) * (252 / HOLD_BARS) ** 0.5 if std > 0 else 0.0
        else:
            trades = 0; win = 0.0; avg = 0.0; sharpe = 0.0

        rows.append({
            "rsi_cross_level": p.rsi_cross_level,
            "vol_mult_min": p.vol_mult_min,
            "macd_hist_floor": p.macd_hist_floor,
            "lookback_bars": knobs.lookback_bars,
            "require_same_bar": knobs.require_same_bar,
            "require_macd_rising": knobs.require_macd_rising,
            "price_slack_bps": knobs.price_slack_bps,
            "entry_edge_lookback": knobs.entry_edge_lookback,
            "trades": trades, "win_rate": round(win,4),
            "avg_ret": round(avg,6), "sharpe_like": round(sharpe,3),
        })

    res = pl.DataFrame(rows).sort(["trades","sharpe_like","win_rate"], descending=[True, True, True])
    out_csv = results_dir / "param_sweep.csv"
    res.write_csv(out_csv)
    print(f"Saved: {out_csv}")
    print(res.head(20))

    # ---- New: multi-curve equity HTML ----
    curves: list[pl.DataFrame] = []
    for p in _iter_grid(DEFAULT_GRID):
        curve = _equity_curve_for_combo(p, knobs, universe, data_dir, feat_dir)
        if not curve.is_empty():
            curves.append(curve)

    out_html = results_dir / "sweep_equity.html"
    if curves:
        all_curves = pl.concat(curves, how="vertical_relaxed")
        if go is None:
            html = "<html><body><h3>Install plotly to see interactive chart: pip install plotly</h3></body></html>"
            out_html.write_text(html, encoding="utf-8")
        else:
            fig = go.Figure()
            labels = all_curves["label"].unique().to_list()
            for label in labels:
                g = all_curves.filter(pl.col("label") == label).sort("date")
                fig.add_trace(go.Scatter(
                    x=g["date"].to_list(),
                    y=g["equity"].to_list(),
                    mode="lines",
                    name=label,
                ))
            fig.update_layout(
                title=f"Sweep Equity Curves ({datetime.now().strftime('%Y-%m-%d')})",
                xaxis_title="Date",
                yaxis_title="Equity (start=1.0)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig.write_html(out_html, include_plotlyjs="cdn")
            print(f"Saved: {out_html}")
    else:
        out_html.write_text("<html><body><h3>No trades in sweep to plot.</h3></body></html>", encoding="utf-8")

if __name__ == "__main__":
    main()
