from __future__ import annotations
import itertools, contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from duckdb import df
import polars as pl

from config_env import load_cfg
from constants import Col
from engine import strategy as STRAT  # use your real strategy.py

# -------- Grid you can tweak --------
DEFAULT_GRID = {
    "rsi_cross_level":   [25.0, 30.0, 35.0],
    "vol_mult_min":      [1.0, 1.1, 1.2],
    "macd_hist_floor":   [0.0],
    "lookback_bars":     [2, 3, 5],
    "require_same_bar":  [False, True],
    "require_macd_rising":[True, False],
    "price_slack_bps":   [0.0, 2.0],
    "entry_edge_lookback":[1, 3],        # how long to wait before allowing re-entry
}

HOLD_BARS = 10
MIN_WARM  = 30

@dataclass(frozen=True)
class Params:
    rsi_cross_level: float
    vol_mult_min: float
    macd_hist_floor: float
    lookback_bars: int
    require_same_bar: bool
    require_macd_rising: bool
    price_slack_bps: float
    entry_edge_lookback: int

def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Params]:
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield Params(**dict(zip(keys, combo)))

@contextlib.contextmanager
def _with_params(p: Dict[str, Any]):
    """Temporarily override strategy._get_strategy_params() so STRAT.signal_rules() uses our grid values."""
    orig = STRAT._get_strategy_params
    try:
        if hasattr(STRAT._get_strategy_params, "cache_clear"):
            STRAT._get_strategy_params.cache_clear()
        def _patched():
            return {
                "rsi_cross_level":     float(p["rsi_cross_level"]),
                "vol_mult_min":        float(p["vol_mult_min"]),
                "macd_hist_floor":     float(p["macd_hist_floor"]),
                "atr_sizing_multiple": 1.2,  # not swept here
                "lookback_bars":       int(p["lookback_bars"]),
                "require_same_bar":    bool(p["require_same_bar"]),
                "require_macd_rising": bool(p["require_macd_rising"]),
                "price_slack_bps":     float(p["price_slack_bps"]),
            }
        STRAT._get_strategy_params = _patched  # type: ignore
        yield
    finally:
        STRAT._get_strategy_params = orig      # type: ignore
        if hasattr(STRAT._get_strategy_params, "cache_clear"):
            STRAT._get_strategy_params.cache_clear()

def _entries_from_signal(sig: pl.Series, k_edge: int) -> pl.Series:
    # normalize to boolean
    sig_b = sig.cast(pl.Boolean).fill_null(False)

    # Convert to int to make rolling_max unambiguous across Polars versions
    sig_i = sig_b.cast(pl.Int8)

    # prior_any = max over last k_edge bars of *prior* signal
    k = max(1, int(k_edge))
    prior_any_i = (
        sig_i.shift(1)
             .rolling_max(window_size=k, min_samples=1)
             .fill_null(0)
    )
    prior_any = (prior_any_i > 0).cast(pl.Boolean)

    entries = (sig_b & (~prior_any)).alias("entry")
    return entries


def _eval_symbol(sym: str, p: Params, data_dir: Path, feat_dir: Path) -> pl.DataFrame | None:
    f_feat = feat_dir / f"{sym}_features.parquet"
    f_data = data_dir / f"{sym}.parquet"
    if not f_feat.exists() or not f_data.exists():
        return None

    feats = pl.read_parquet(f_feat)
    data  = pl.read_parquet(f_data).select([Col.DATE.value, Col.CLOSE.value])
    df = feats.join(data, on=Col.DATE.value, how="left").drop_nulls([Col.CLOSE.value])

    with _with_params(p.__dict__):
        sig = STRAT.signal_rules(df)  # uses your real strategy
    ent = _entries_from_signal(sig, k_edge=max(1, int(p.entry_edge_lookback)))
    df  = df.with_columns([sig.alias("signal"), ent])

    # DEBUG: counts to verify sweep is seeing signals and entries
    sig_cnt  = int(df["signal"].sum())
    ent_cnt  = int(df["entry"].sum())
    print(f"[DBG] {sym} sig={sig_cnt} ent={ent_cnt}  (edge={p.entry_edge_lookback}, lookback={p.lookback_bars}, same_bar={p.require_same_bar}, macd_rising={p.require_macd_rising}, vol_min={p.vol_mult_min}, slack_bps={p.price_slack_bps})")

    if ent_cnt == 0 and sig_cnt > 0:
        print(f"[DEBUG] {sym} sig>0 but 0 entries | k_edge={p.entry_edge_lookback} | "
            f"lookback={p.lookback_bars} | same_bar={p.require_same_bar} | "
            f"vol_min={p.vol_mult_min} | slack_bps={p.price_slack_bps}")
    elif ent_cnt > 0:
        print(f"[OK] {sym} signals={sig_cnt} entries={ent_cnt} params={p}")

    # TEMP sanity: enter on signal (no edge)
    ent = sig.cast(pl.Boolean).fill_null(False)
    df  = df.with_columns([sig.alias("signal"), ent.alias("entry")])

    # Compute exit indices/prices on the full frame, then filter to entries
    n = df.height
    # Clamp indices into [0, n-1] to avoid out-of-bounds in gather
    exit_idx = (pl.arange(0, n) + HOLD_BARS).clip(0, n - 1)
    df = df.with_columns([
        pl.col(Col.CLOSE.value).gather(exit_idx).alias("_exit_px")
    ])
    df = df.with_columns([
        (pl.col("_exit_px") / pl.col(Col.CLOSE.value) - 1.0).alias("ret_h")
    ])

    out = df.filter(pl.col("entry")).select([
        pl.lit(sym).alias("symbol"),
        pl.col(Col.DATE.value).alias("entry_date"),
        pl.col(Col.CLOSE.value).alias("entry_px"),
        pl.col("ret_h"),
    ])
    return out

def main():
    meta = load_cfg()
    cfg   = meta.get("cfg", {}) or {}
    paths_abs = meta.get("paths_abs", {}) or {}
    # Use resolved absolute paths for consistency with other scripts
    data_dir = Path(paths_abs.get("data_dir") or Path(__file__).resolve().parents[2] / "main" / "artifacts" / "local_data")
    feat_dir = Path(paths_abs.get("features_dir") or Path(__file__).resolve().parents[2] / "main" / "artifacts" / "features")
    results_dir = Path(paths_abs.get("results_dir") or Path(__file__).resolve().parents[2] / "main" / "artifacts" / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    universe = cfg.get("universe", ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA"])

    rows = []
    for p in _iter_grid(DEFAULT_GRID):
        per_sym = []
        for sym in universe:
            r = _eval_symbol(sym, p, data_dir, feat_dir)
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
            "lookback_bars": p.lookback_bars,
            "require_same_bar": p.require_same_bar,
            "require_macd_rising": p.require_macd_rising,
            "price_slack_bps": p.price_slack_bps,
            "entry_edge_lookback": p.entry_edge_lookback,
            "trades": trades, "win_rate": round(win,4),
            "avg_ret": round(avg,6), "sharpe_like": round(sharpe,3),
        })

    res = pl.DataFrame(rows).sort(["trades","sharpe_like","win_rate"], descending=[True, True, True])
    out_csv = results_dir / "param_sweep.csv"
    res.write_csv(out_csv)
    print(f"Saved: {out_csv}")
    print(res.head(20))

if __name__ == "__main__":
    print("USING strategy_sweeps with STRAT.signal_rules + _with_params ✅", flush=True)
    main()
