from __future__ import annotations
import itertools
from dataclasses import dataclass
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

def _eval_symbol(sym: str, p: Params, knobs: SweepKnobs, data_dir: Path, feat_dir: Path) -> pl.DataFrame | None:
    f_feat = feat_dir / f"{sym}_features.parquet"
    f_data = data_dir / f"{sym}.parquet"
    if not f_feat.exists() or not f_data.exists():
        return None

    feats = pl.read_parquet(f_feat)
    data  = pl.read_parquet(f_data).select([Col.DATE.value, Col.CLOSE.value])
    df = feats.join(data, on=Col.DATE.value, how="left").drop_nulls([Col.CLOSE.value])

    sig = _signal_with_knobs(df, p, knobs)
    ent = _entries_from_signal(sig, knobs.entry_edge_lookback)

    # Compute exit indices on full frame, clamp within [0, n-1]
    n = df.height
    exit_idx = (pl.arange(0, n) + HOLD_BARS).clip(0, n - 1)

    df = df.with_columns([
        sig.alias("signal"),
        ent.alias("entry"),
        pl.col(Col.CLOSE.value).gather(exit_idx).alias("_exit_px"),
    ]).with_columns([
        (pl.col("_exit_px") / pl.col(Col.CLOSE.value) - 1.0).alias("ret_h"),
    ])

    out = df.filter(pl.col("entry")).select([
        pl.lit(sym).alias("symbol"),
        pl.col(Col.DATE.value).alias("entry_date"),
        pl.col(Col.CLOSE.value).alias("entry_px"),
        pl.col("ret_h"),
    ])
    return out

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

if __name__ == "__main__":
    main()
