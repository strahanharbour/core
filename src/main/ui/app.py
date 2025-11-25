from __future__ import annotations

from pathlib import Path
from typing import Dict

import polars as pl
import streamlit as st

from main.config_env import load_cfg


st.set_page_config(page_title="Core Research UI", layout="wide")


def _summary_from_trades(tr: pl.DataFrame) -> Dict[str, float]:
    n = tr.height
    total_pnl = float(tr.get_column("pnl").sum()) if n else 0.0
    wins = int(tr.filter(pl.col("pnl") > 0).height) if n else 0
    win_rate = (wins / n * 100.0) if n else 0.0
    final_equity = 10_000.0 + total_pnl
    avg_pnl = float(tr.get_column("pnl").mean()) if n else 0.0
    return {
        "trades": float(n),
        "win_rate": win_rate,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
    }


def main() -> None:
    meta = load_cfg()
    paths_abs: Dict[str, str] = meta["paths_abs"]
    res_dir = Path(paths_abs["results_dir"])  # absolute

    st.title("Core Research Dashboard")

    trades_fp = res_dir / "trades.parquet"
    equity_fp = res_dir / "equity.html"

    if trades_fp.exists():
        tr = pl.read_parquet(trades_fp)
        s = _summary_from_trades(tr)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades", int(s["trades"]))
        c2.metric("Win Rate", f"{s['win_rate']:.1f}%")
        c3.metric("Final Equity", f"${s['final_equity']:,.2f}")
        c4.metric("Total PnL", f"${s['total_pnl']:,.2f}")

        st.subheader("Trades (first 200)")
        st.dataframe(tr.head(200).to_pandas(), use_container_width=True)
    else:
        st.warning("trades.parquet not found. Run the pipeline first.")

    st.subheader("Equity Curve")
    if equity_fp.exists():
        # Embed the Plotly HTML if present
        with open(equity_fp, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=500, scrolling=True)
    else:
        st.info(
            "Equity chart not found. Generate it by running:\n"
            "1) python -m data.ingest_daily_local\n"
            "2) python -m data.build_features\n"
            "3) python -m research.backtest\n"
            "4) python -m research.report"
        )


if __name__ == "__main__":
    main()
