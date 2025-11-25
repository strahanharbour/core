from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from main.config_env import load_cfg
from main.constants import Col
from engine.strategy import signal_rules
from research.sim_lib import (
    load_cfg_bits,
    apply_sentiment_gate,
    apply_market_filter,
    strict_entry_edge,
    exits_returns,
)


@dataclass
class WFConfig:
    is_ratio: float = 0.7  # in-sample fraction per fold
    step_bars: int = 60  # roll step (bars)
    min_bars: int = 300  # minimum data length to start
    combine_universe: bool = True  # aggregate across symbols per fold


def _load_wf_cfg() -> WFConfig:
    cfg = load_cfg()["cfg"]
    wf = (cfg.get("walk_forward") or {})
    return WFConfig(
        is_ratio=float(wf.get("is_ratio", 0.7)),
        step_bars=int(wf.get("step_bars", 60)),
        min_bars=int(wf.get("min_bars", 300)),
        combine_universe=bool(wf.get("combine_universe", True)),
    )


def _load_symbol_frame(sym: str, feat_dir: Path, data_dir: Path) -> pl.DataFrame | None:
    f_feat = feat_dir / f"{sym}_features.parquet"
    f_px = data_dir / f"{sym}.parquet"
    if not f_feat.exists() or not f_px.exists():
        return None
    feats = pl.read_parquet(f_feat)
    px = pl.read_parquet(f_px).select([Col.DATE.value, Col.CLOSE.value])
    df = (
        feats.join(px, on=Col.DATE.value, how="left")
        .drop_nulls([Col.CLOSE.value])
        .sort(Col.DATE.value)
    )
    return df if df.height > 0 else None


def _simulate_window(
    df: pl.DataFrame, cfg_all: Dict, paths: Dict, sent_cfg: Dict, strat_cfg: Dict
) -> pl.DataFrame:
    """Return trades DF (entry_date, entry_px, ret_h) for this symbol window using current gates/exits."""
    sig = signal_rules(df)
    sig = apply_sentiment_gate(
        df,
        sig,
        df["symbol"][0] if "symbol" in df.columns and df.height > 0 else "SYM",
        Path(paths.get("features_dir", "src/main/artifacts/features")),
        sent_cfg,
    )
    sig = apply_market_filter(
        df,
        sig,
        Path(paths.get("data_dir", "src/main/artifacts/local_data")),
        strat_cfg,
    )
    entries = strict_entry_edge(sig)
    trades = exits_returns(df, entries, strat_cfg)
    return (
        trades
        if (trades is not None and not trades.is_empty())
        else pl.DataFrame({"entry_date": [], "entry_px": [], "ret_h": []})
    )


def _wf_for_symbol(
    sym: str, cfg_wf: WFConfig, paths: Dict, sent_cfg: Dict, strat_cfg: Dict
) -> pl.DataFrame:
    feat_dir = Path(paths.get("features_dir", "src/main/artifacts/features"))
    data_dir = Path(paths.get("data_dir", "src/main/artifacts/local_data"))
    df = _load_symbol_frame(sym, feat_dir, data_dir)
    if df is None or df.height < cfg_wf.min_bars:
        return pl.DataFrame({"fold": [], "part": [], "sharpe": [], "trades": [], "symbol": []})
    n = df.height
    out_rows: List[Dict] = []
    start = 0
    while start + cfg_wf.min_bars <= n:
        end = min(n, start + cfg_wf.min_bars + cfg_wf.step_bars)  # expanding-ish
        win = df.slice(start, end - start)
        k = int(win.height * cfg_wf.is_ratio)
        is_df, oos_df = win.slice(0, k), win.slice(k)
        # simulate OOS only (true WF evaluation)
        tr_oos = _simulate_window(oos_df, {}, paths, sent_cfg, strat_cfg)
        # metrics
        r = tr_oos["ret_h"] if "ret_h" in tr_oos.columns else pl.Series([])
        m = float(r.mean() or 0.0)
        s = float(r.std(ddof=1) or 0.0)
        sh = 0.0 if s == 0.0 else m / s
        out_rows.append(
            {
                "fold": f"{start}-{end}",
                "part": "OOS",
                "sharpe": sh,
                "trades": int(len(r)),
                "symbol": sym,
            }
        )
        start += cfg_wf.step_bars
    return pl.DataFrame(out_rows)


def main() -> None:
    cfg_all, paths, sent_cfg, strat_cfg = load_cfg_bits()
    wf_cfg = _load_wf_cfg()
    results_dir = Path(paths.get("results_dir", "src/main/artifacts/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    universe = cfg_all.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])

    per_sym = [
        _wf_for_symbol(sym, wf_cfg, paths, sent_cfg, strat_cfg) for sym in universe
    ]
    wf = (
        pl.concat([x for x in per_sym if x is not None], how="vertical_relaxed")
        if per_sym
        else pl.DataFrame()
    )
    if wf.is_empty():
        (results_dir / "wf_report.md").write_text(
            "No WF results (insufficient data).", encoding="utf-8"
        )
        return
    # Aggregate across symbols per fold
    agg = (
        wf.group_by(["fold", "part"]).agg(
            [
                pl.mean("sharpe").alias("oos_sharpe_mean"),
                pl.sum("trades").alias("trades_sum"),
            ]
        )
        .sort("oos_sharpe_mean", descending=True)
    )
    wf_file = results_dir / "wf_metrics.csv"
    agg.write_csv(wf_file)
    # Polars doesn't have DataFrame.to_string; use str()
    (results_dir / "wf_report.md").write_text(str(agg.head(20)), encoding="utf-8")
    print(f"Saved: {wf_file}")


if __name__ == "__main__":
    main()
