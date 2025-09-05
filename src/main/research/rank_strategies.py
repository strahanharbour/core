from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math

import polars as pl

# Inputs:
# - artifacts/results/param_sweep.csv  (already produced by strategy_sweeps)
# - artifacts/results/sweep_equity.html (optional visual)
# - We will (re)build equity curves per combo for robust metrics using existing sweeper helpers

# ---- Configurable weights for composite score ----
WEIGHTS: Dict[str, float] = {
    "sharpe_like": 0.40,
    "avg_ret": 0.20,
    "win_rate": 0.15,
    "trades": 0.10,  # with diminishing returns
    "oos_ratio": 0.10,  # OOS/IS Sharpe ratio
    "stability": 0.05,  # neighbor stability
}

MIN_TRADES = 10  # below this, penalize heavily
MAX_TRADES_BONUS = 200  # cap the bonus of "many trades"


# ---- Helpers ----
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if (b and not math.isnan(b)) else default


def _bounded(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _diminishing_trades(n: int) -> float:
    # Smoothly increases and saturates
    return math.tanh(n / MAX_TRADES_BONUS)


# ---- Load param_sweep and (optionally) per-curve trades to compute risk ----
def load_sweep_results(results_dir: Path) -> pl.DataFrame:
    f = results_dir / "param_sweep.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")
    return pl.read_csv(f, infer_schema_length=5000)


# ---- Recompute equity per combo for drawdown + time split + regime split ----
# We reuse the logic already in strategy_sweeps via a small import (no code duplication).
def build_curves_for_all(results_dir: Path) -> pl.DataFrame:
    try:
        from research.strategy_sweeps import _equity_curve_for_combo, Params, _load_knobs
        from config_env import load_cfg

        cfg = load_cfg()["cfg"]
        paths = cfg.get("paths", {}) or {}
        data_dir = Path(paths.get("data_dir", "src/main/artifacts/local_data"))
        feat_dir = Path(paths.get("features_dir", "src/main/artifacts/features"))
        universe = cfg.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
        knobs = _load_knobs()

        ps = load_sweep_results(results_dir)
        have_sic = ("sic_enabled" in ps.columns) and ("sic_threshold" in ps.columns)

        curves: List[pl.DataFrame] = []
        for row in ps.iter_rows(named=True):
            if have_sic:
                p = Params(
                    rsi_cross_level=float(row["rsi_cross_level"]),
                    vol_mult_min=float(row["vol_mult_min"]),
                    macd_hist_floor=float(row["macd_hist_floor"]),
                    sic_enabled=bool(row["sic_enabled"]),
                    sic_threshold=float(row["sic_threshold"]),
                )
            else:
                p = Params(
                    rsi_cross_level=float(row["rsi_cross_level"]),
                    vol_mult_min=float(row["vol_mult_min"]),
                    macd_hist_floor=float(row["macd_hist_floor"]),
                    # Defaults if Params requires these
                    sic_enabled=False,
                    sic_threshold=0.0,
                )
            c = _equity_curve_for_combo(p, knobs, universe, data_dir, feat_dir)
            if not c.is_empty():
                # Attach param columns for grouping
                add_cols = {
                    "rsi_cross_level": p.rsi_cross_level,
                    "vol_mult_min": p.vol_mult_min,
                    "macd_hist_floor": p.macd_hist_floor,
                }
                if have_sic:
                    add_cols.update({"sic_enabled": p.sic_enabled, "sic_threshold": p.sic_threshold})
                c = c.with_columns([pl.lit(v).alias(k) for k, v in add_cols.items()])
                curves.append(c)
        return pl.concat(curves, how="vertical_relaxed") if curves else pl.DataFrame({
            "date": [],
            "equity": [],
            "label": [],
            "rsi_cross_level": [],
            "vol_mult_min": [],
            "macd_hist_floor": [],
            "sic_enabled": [],
            "sic_threshold": [],
        })
    except Exception:
        # Fallback: no curves available
        return pl.DataFrame({"date": [], "equity": [], "label": []})


def max_drawdown(curve: pl.DataFrame) -> float:
    if curve.is_empty():
        return 0.0
    c = curve.sort("date")["equity"].to_list()
    peak = c[0]
    mdd = 0.0
    for v in c:
        peak = max(peak, v)
        dd = (v / peak) - 1.0
        mdd = min(mdd, dd)
    return float(mdd)


def split_is_oos(curve: pl.DataFrame, split_ratio: float = 0.6) -> Tuple[float, float]:
    if curve.height < 10:
        return (0.0, 0.0)
    curve = curve.sort("date")
    n = curve.height
    k = int(n * split_ratio)
    is_part = curve.slice(0, k)
    oos_part = curve.slice(k)

    def _sh(df: pl.DataFrame) -> float:
        if df.height <= 1:
            return 0.0
        ret = [(float(df["equity"][i]) / float(df["equity"][i - 1]) - 1.0) for i in range(1, df.height)]
        if not ret:
            return 0.0
        m = sum(ret) / len(ret)
        var = sum((x - m) ** 2 for x in ret) / max(1, len(ret) - 1)
        s = math.sqrt(var)
        return 0.0 if s == 0.0 else m / s

    return (_sh(is_part), _sh(oos_part))


def rank_strategies(results_dir: Path) -> pl.DataFrame:
    ps = load_sweep_results(results_dir)
    curves = build_curves_for_all(results_dir)

    # Aggregate curve-based metrics per parameter combo (if curves exist)
    curve_metrics = None
    if not curves.is_empty():
        keys = ["rsi_cross_level", "vol_mult_min", "macd_hist_floor"]
        if "sic_enabled" in curves.columns and "sic_threshold" in curves.columns:
            keys.extend(["sic_enabled", "sic_threshold"])
        rows: List[Dict] = []
        # iterate unique combos
        unique_keys = curves.select([pl.col(k) for k in keys]).unique()
        for rec in unique_keys.iter_rows(named=True):
            mask = pl.all([
                pl.col(k) == rec[k] for k in keys
            ])
            grp = curves.filter(mask)
            mdd = max_drawdown(grp)
            is_s, oos_s = split_is_oos(grp)
            rows.append({**rec, "max_drawdown": mdd, "is_sharpe": is_s, "oos_sharpe": oos_s, "oos_ratio": _safe_div(oos_s, abs(is_s) if is_s != 0.0 else 1.0)})
        curve_metrics = pl.DataFrame(rows)

    # Merge with param_sweep.csv on parameter columns
    join_keys = ["rsi_cross_level", "vol_mult_min", "macd_hist_floor"]
    if "sic_enabled" in ps.columns:
        join_keys.append("sic_enabled")
    if "sic_threshold" in ps.columns:
        join_keys.append("sic_threshold")

    df = ps
    if curve_metrics is not None and not curve_metrics.is_empty():
        df = df.join(curve_metrics, on=join_keys, how="left")

    # Ensure expected curve metrics exist even if curves couldn't be built
    for col in ["max_drawdown", "oos_ratio"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Normalize/score components
    df = df.with_columns([
        (-pl.col("max_drawdown").fill_null(0.0)).alias("mdd_inv"),
        pl.col("trades").fill_null(0).map_elements(lambda n: _diminishing_trades(int(n))).alias("trades_bonus"),
        pl.when(pl.col("trades") < MIN_TRADES).then(-1.0).otherwise(0.0).alias("low_trade_penalty"),
    ])

    # Ensure columns exist
    for col in ["sharpe_like", "avg_ret", "win_rate", "mdd_inv", "trades_bonus", "oos_ratio"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Z-scores using expressions
    def zexpr(name: str) -> pl.Expr:
        return ((pl.col(name) - pl.col(name).mean()) / pl.col(name).std(ddof=1)).fill_nan(0.0).fill_null(0.0)

    df = df.with_columns([
        zexpr("sharpe_like").alias("z_sharpe_like"),
        zexpr("avg_ret").alias("z_avg_ret"),
        zexpr("win_rate").alias("z_win_rate"),
        zexpr("mdd_inv").alias("z_mdd_inv"),
        zexpr("trades_bonus").alias("z_trades_bonus"),
        zexpr("oos_ratio").alias("z_oos_ratio"),
    ])

    # Neighbor stability: average z_sharpe_like in local bucket of rsi|vol
    df = df.with_columns(pl.format("{}|{}", pl.col("rsi_cross_level"), pl.col("vol_mult_min")).alias("nb_key"))
    nb = df.group_by("nb_key").agg(pl.mean("z_sharpe_like").alias("z_sharpe_local_mean"))
    df = df.join(nb, on="nb_key", how="left").with_columns(
        pl.col("z_sharpe_local_mean").fill_null(0.0).alias("stability")
    )

    # Composite score
    df = df.with_columns(
        (
            WEIGHTS["sharpe_like"] * pl.col("z_sharpe_like")
            + WEIGHTS["avg_ret"] * pl.col("z_avg_ret")
            + WEIGHTS["win_rate"] * pl.col("z_win_rate")
            + WEIGHTS["trades"] * pl.col("z_trades_bonus")
            + WEIGHTS["oos_ratio"] * pl.col("z_oos_ratio")
            + WEIGHTS["stability"] * pl.col("stability")
            + pl.col("low_trade_penalty")
        ).alias("score")
    )

    df = df.sort(["score", "sharpe_like", "win_rate", "trades"], descending=[True, True, True, True])
    return df


def main() -> None:
    results_dir = Path("src/main/artifacts/results")
    ranked = rank_strategies(results_dir)
    out_csv = results_dir / "ranked_strategies.csv"
    ranked.write_csv(out_csv)
    # quick human-readable
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
    top5 = ranked.select(sel_cols).head(10)
    # Write a simple text table; Polars lacks to_string, so use str(df)
    (results_dir / "rank_report.md").write_text(str(top5), encoding="utf-8")
    print(f"Saved: {out_csv}")
    print(top5)


if __name__ == "__main__":
    main()


@dataclass
class StrategyRanker:
    """Encapsulates ranking computation for param sweep results."""

    results_dir: Path

    def rank(self) -> pl.DataFrame:
        return rank_strategies(self.results_dir)
