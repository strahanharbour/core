from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from main.hf_strategy.hrp import hrp_allocation
    from main.hf_strategy.kpis import sharpe, sortino, max_drawdown, cvar
except Exception:  # fallback when PYTHONPATH points at src/main
    from hf_strategy.hrp import hrp_allocation  # type: ignore
    from hf_strategy.kpis import sharpe, sortino, max_drawdown, cvar  # type: ignore


def backtest(df: pd.DataFrame, p_thresh: float = 0.6) -> Tuple[dict, pd.Series, pd.Series]:
    """
    df columns: dt, ticker, ret_next, p_cal, atr, sic, regime
    Returns: (report dict, daily returns series, equity series)
    """
    g = []
    for dt, d in df.groupby("dt"):
        tradable = d[d["p_cal"] > p_thresh].copy()
        if tradable.empty:
            continue
        # HRP over candidate tickers using recent returns window if available
        recent = d.pivot(index="dt", columns="ticker", values="ret_next").tail(60)
        if recent.shape[1] >= 2:
            w = hrp_allocation(recent)
        else:
            w = pd.Series(1.0, index=tradable["ticker"], name="w")
        tradable = tradable.merge(w.rename("w"), left_on="ticker", right_index=True, how="left").fillna(0.0)
        tradable["size"] = np.minimum(1.0, tradable["p_cal"] / 0.8) / np.maximum(tradable["atr"], 1e-6)
        tradable["ret"] = tradable["ret_next"] * tradable["size"] * tradable["w"]
        g.append(tradable[["ticker", "ret"]].assign(dt=dt))
    if not g:
        empty = {"Sharpe": 0, "Sortino": 0, "MaxDD": 0, "CVaR95": 0}
        return empty, pd.Series(dtype=float), pd.Series(dtype=float)
    trades = pd.concat(g, ignore_index=True)
    daily = trades.groupby("dt")["ret"].sum()
    eq = (1 + daily.fillna(0)).cumprod()
    rep = {
        "Sharpe": sharpe(daily),
        "Sortino": sortino(daily),
        "MaxDD": max_drawdown(eq),
        "CVaR95": cvar(daily, 0.95),
    }
    return rep, daily, eq


def _default_results_dir() -> Path:
    # Keep results alongside other artifacts
    return Path("src/main/artifacts/results").resolve()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Meta backtest with HRP + calibrated probs")
    parser.add_argument("--file", type=str, default=os.getenv("META_DATASET", ""), help="Path to dataset (csv/parquet)")
    parser.add_argument("--p", type=float, default=0.6, help="Probability threshold for trading")
    args = parser.parse_args()

    f = Path(args.file) if args.file else Path("src/main/artifacts/results/meta_dataset.parquet")
    if not f.exists():
        print(json.dumps({
            "error": f"Dataset not found: {str(f)}",
            "hint": "Provide --file path to csv/parquet with columns: dt,ticker,ret_next,p_cal,atr,sic,regime",
        }))
        return

    if f.suffix.lower() == ".csv":
        df = pd.read_csv(f, parse_dates=["dt"])  # type: ignore[list-item]
    else:
        df = pd.read_parquet(f)
        if not np.issubdtype(df["dt"].dtype, np.datetime64):
            try:
                df["dt"] = pd.to_datetime(df["dt"])
            except Exception:
                pass

    rep, daily, eq = backtest(df, p_thresh=float(args.p))
    out_dir = _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta_daily.csv").write_text(daily.to_csv(index=True), encoding="utf-8")
    eq.to_frame("equity").to_csv(out_dir / "meta_equity.csv", index=True)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
