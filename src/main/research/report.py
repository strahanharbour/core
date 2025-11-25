from __future__ import annotations

import logging
from pathlib import Path
from datetime import date as _date
from typing import Dict

import polars as pl
import plotly.graph_objects as go

from main.config_env import load_cfg


logger = logging.getLogger(__name__)


def _build_equity_from_trades(trades: pl.DataFrame, start_equity: float = 10_000.0) -> pl.DataFrame:
    if trades.is_empty():
        # Produce a minimal 1-point equity curve at start equity
        # Use a Python date object; pl.Date is a dtype, not a constructor
        return pl.DataFrame({"date": [_date(1970, 1, 1)], "equity": [start_equity]})

    # Parse exit_date into Date and aggregate daily PnL (robust to dtype)
    dt = trades.schema.get("exit_date")
    if dt == pl.Utf8:
        # Heuristic: if it contains a time (":"), parse as datetime with explicit format; else parse as date.
        dates = (
            pl.when(pl.col("exit_date").str.contains(":"))
            .then(
                pl.col("exit_date")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False, exact=False)
                .dt.date()
            )
            .otherwise(
                pl.col("exit_date").str.to_date(format="%Y-%m-%d", strict=False, exact=False)
            )
        )
    elif dt == pl.Datetime:
        dates = pl.col("exit_date").dt.date()
    else:
        # Coerce any non-string datelike to Date (nulls on failure)
        dates = pl.col("exit_date").cast(pl.Date, strict=False)

    pnl_by_day = (
        trades.with_columns(dates.alias("date"))
        .group_by("date", maintain_order=True)
        .agg(pl.col("pnl").sum().alias("pnl"))
        .sort("date")
    )

    # Generate a continuous daily range and forward-fill missing days with zero pnl
    start = pnl_by_day.select(pl.col("date").min()).item()
    end = pnl_by_day.select(pl.col("date").max()).item()
    calendar = pl.DataFrame({"date": pl.date_range(start, end, "1d", eager=True)})

    daily = calendar.join(pnl_by_day, on="date", how="left").with_columns(
        pl.col("pnl").fill_null(0.0)
    )
    equity = start_equity + daily.get_column("pnl").cum_sum()
    return daily.select(["date"]).with_columns(equity.alias("equity"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    meta = load_cfg()
    paths_abs: Dict[str, str] = meta["paths_abs"]
    res_dir = Path(paths_abs["results_dir"])  # absolute
    res_dir.mkdir(parents=True, exist_ok=True)

    trades_path = res_dir / "trades.parquet"
    if not trades_path.exists():
        logger.error("trades.parquet not found at %s. Run the backtest first.", trades_path)
        return

    trades = pl.read_parquet(trades_path)
    eq = _build_equity_from_trades(trades)

    # Plotly equity curve
    x = eq.get_column("date").to_list()
    y = eq.get_column("equity").to_list()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Equity"))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_white",
    )

    out_html = res_dir / "equity.html"
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    logger.info("Wrote %s", out_html)


if __name__ == "__main__":
    main()
