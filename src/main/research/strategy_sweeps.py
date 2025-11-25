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

from main.config_env import load_cfg
from main.constants import Col
from main.research.sim_lib import (
    load_cfg_bits,
    apply_sentiment_gate,
    apply_market_filter,
    apply_regime_gate,
    strict_entry_edge,
    exits_returns,
)

# ---- What we sweep (keep simple/orthogonal) ----
DEFAULT_GRID = {
    "rsi_cross_level": [25.0, 30.0, 35.0],
    "vol_mult_min":    [1.0, 1.1, 1.2],
    "macd_hist_floor": [0.0],
    # NEW sentiment toggles per combo:
    "sic_enabled":     [False, True],
    "sic_threshold":   [0.0, 0.1, 0.2],
}

# Exit horizon for quick scoring (not part of runtime)
HOLD_BARS = 10
MIN_WARM  = 30

@dataclass(frozen=True)
class Params:
    rsi_cross_level: float
    vol_mult_min: float
    macd_hist_floor: float
    sic_enabled: bool           # NEW
    sic_threshold: float        # NEW

def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Params]:
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield Params(**dict(zip(keys, combo)))


def _label_for_params(p: Params) -> str:
    s = (
        f"rsi={p.rsi_cross_level}|vol={p.vol_mult_min}|macd={p.macd_hist_floor}"
        f"|sic={'on' if p.sic_enabled else 'off'}@{p.sic_threshold}"
    )
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
    # Retained for compatibility; not used after sim_lib strict edge wiring
    sig_b = sig.cast(pl.Boolean).fill_null(False)
    prev_any = sig_b.shift(1).rolling_max(window_size=max(1, k_edge), min_samples=1).cast(pl.Boolean).fill_null(False)
    return (sig_b & (~prev_any)).alias("entry")


## Market filter helper lives in research/_market_filter.py using global strategy.market_filter

@dataclass(frozen=True)
class ExitKnobs:
    enabled: bool
    stop_atr_mult: float
    take_atr_mult: float
    max_hold_bars: int


def _load_exit_knobs() -> ExitKnobs:
    cfg = load_cfg()["cfg"]
    strat = (cfg.get("strategy") or {})
    ex = (strat.get("exits") or {})
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

    # Build signal with knobs
    sig = _signal_with_knobs(df, p, knobs)
    # Per-combo SIC gate (independent of global config toggles)
    if p.sic_enabled:
        sic_file = feat_dir / "sentiment" / f"{sym}_sic.parquet"
        if sic_file.exists():
            sic = pl.read_parquet(sic_file).select(["date", "sic"])
            df = df.join(sic, on="date", how="left")
            sig = sig & (df["sic"].fill_null(0.0) >= p.sic_threshold)
    # Load cfg bits and apply sentiment + market gates
    _, paths_cfg, sent_cfg, strat_cfg = load_cfg_bits()
    sig = apply_sentiment_gate(df, sig, sym, feat_dir, sent_cfg)
    sig = apply_market_filter(df, sig, data_dir, strat_cfg)
    sig = apply_regime_gate(df, sig, data_dir, strat_cfg)
    # Strict entry edge for consistency with backtest
    ent = strict_entry_edge(sig)
    # Compute returns using shared exits
    trades = exits_returns(df, ent, strat_cfg)
    if trades is not None and not trades.is_empty():
        trades = trades.with_columns([pl.lit(sym).alias("symbol")])
        return trades
    return None


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

    growth = (
        trades
        .group_by("entry_date")
        .agg(pl.sum("ret_h").alias("sum_ret"))
        .with_columns([
            (1.0 + pl.col("sum_ret")).alias("gf"),
            pl.col("entry_date").dt.date().alias("date"),
        ])
        .select(["date", "gf"])
    )

    curve = daily.join(growth, on="date", how="left").with_columns(
        pl.when(pl.col("gf").is_null()).then(1.0).otherwise(pl.col("gf")).alias("gf")
    )
    curve = curve.with_columns(
        pl.col("gf").cum_prod().alias("equity")
    ).select(["date", "equity"])
    curve = curve.with_columns(pl.lit(_label_for_params(p)).alias("label"))
    return curve

class StrategySweeper:
    """Lightweight class wrapper around the sweep process.

    Encapsulates path resolution, grid iteration, and outputs (CSV + HTML + ranking).
    """

    def __init__(self) -> None:
        meta = load_cfg()
        self.cfg = meta.get("cfg", {}) or {}
        paths_abs = meta.get("paths_abs", {}) or {}
        base = Path(__file__).resolve().parents[2]
        self.data_dir = Path(paths_abs.get("data_dir") or base / "main" / "artifacts" / "local_data")
        self.feat_dir = Path(paths_abs.get("features_dir") or base / "main" / "artifacts" / "features")
        self.results_dir = Path(paths_abs.get("results_dir") or base / "main" / "artifacts" / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.universe = self.cfg.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])  # type: ignore[assignment]
        self.knobs = _load_knobs()

    def _score_combo(self, p: Params) -> dict:
        per_sym: list[pl.DataFrame] = []
        for sym in self.universe:
            r = _eval_symbol(sym, p, self.knobs, self.data_dir, self.feat_dir)
            if r is not None and r.height > 0:
                per_sym.append(r)
        if per_sym:
            all_tr = pl.concat(per_sym, how="vertical_relaxed")
            trades = int(all_tr.height)
            win = float((all_tr["ret_h"] > 0).mean())
            avg = float(all_tr["ret_h"].mean())
            std = float(all_tr["ret_h"].std(ddof=1) or 0.0)
            sharpe = (avg / std) * (252 / HOLD_BARS) ** 0.5 if std > 0 else 0.0
        else:
            trades = 0
            win = 0.0
            avg = 0.0
            sharpe = 0.0
        return {
            "p": p,
            "metrics": {
                "trades": trades,
                "win": win,
                "avg": avg,
                "sharpe": sharpe,
            },
        }

    def _write_summary(self, rows: list[dict]) -> None:
        res = pl.DataFrame(rows).sort(["trades", "sharpe_like", "win_rate"], descending=[True, True, True])
        out_csv = self.results_dir / "param_sweep.csv"
        res.write_csv(out_csv)
        print(f"Saved: {out_csv}")
        print(res.head(20))

    def _write_equity(self) -> None:
        curves: list[pl.DataFrame] = []
        for p in _iter_grid(DEFAULT_GRID):
            curve = _equity_curve_for_combo(p, self.knobs, self.universe, self.data_dir, self.feat_dir)
            if not curve.is_empty():
                curves.append(curve)
        out_html = self.results_dir / "sweep_equity.html"
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
                    fig.add_trace(
                        go.Scatter(
                            x=g["date"].to_list(),
                            y=g["equity"].to_list(),
                            mode="lines",
                            name=label,
                        )
                    )
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

    def _rank(self) -> None:
        try:
            from main.research.rank_strategies import StrategyRanker

            ranked = StrategyRanker(self.results_dir).rank()
            out_rank_csv = self.results_dir / "ranked_strategies.csv"
            ranked.write_csv(out_rank_csv)
            # Quick top report
            sel_cols = [
                "rsi_cross_level",
                "vol_mult_min",
                "macd_hist_floor",
                "trades",
                "win_rate",
                "avg_ret",
                "sharpe_like",
                "max_drawdown",
                "oos_ratio",
                "stability",
                "score",
            ]
            if "sic_enabled" in ranked.columns:
                sel_cols.insert(3, "sic_enabled")
            if "sic_threshold" in ranked.columns:
                sel_cols.insert(4, "sic_threshold")
            top = ranked.select([c for c in sel_cols if c in ranked.columns]).head(10)
            (self.results_dir / "rank_report.md").write_text(str(top), encoding="utf-8")
            print(f"Saved: {out_rank_csv}")
        except Exception as e:
            print(f"Ranking step skipped: {e}")

    def run(self) -> None:
        rows: list[dict] = []
        for p in _iter_grid(DEFAULT_GRID):
            sc = self._score_combo(p)
            metrics = sc["metrics"]
            rows.append(
                {
                    "rsi_cross_level": p.rsi_cross_level,
                    "vol_mult_min": p.vol_mult_min,
                    "macd_hist_floor": p.macd_hist_floor,
                    "sic_enabled": p.sic_enabled,
                    "sic_threshold": p.sic_threshold,
                    "lookback_bars": self.knobs.lookback_bars,
                    "require_same_bar": self.knobs.require_same_bar,
                    "require_macd_rising": self.knobs.require_macd_rising,
                    "price_slack_bps": self.knobs.price_slack_bps,
                    "entry_edge_lookback": self.knobs.entry_edge_lookback,
                    "trades": int(metrics["trades"]),
                    "win_rate": round(float(metrics["win"]), 4),
                    "avg_ret": round(float(metrics["avg"]), 6),
                    "sharpe_like": round(float(metrics["sharpe"]), 3),
                }
            )

        self._write_summary(rows)
        self._write_equity()
        self._rank()


def main():
    sweeper = StrategySweeper()
    sweeper.run()


if __name__ == "__main__":
    main()
