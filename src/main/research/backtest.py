from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import polars as pl

from config_env import load_cfg
from engine.strategy import gated_entry, position_size
from constants import Col
from utils.columns import canonicalize_polars_columns


logger = logging.getLogger(__name__)


def _load_symbol_frames(sym: str, data_dir: Path, feat_dir: Path) -> pl.DataFrame | None:
    data_fp = data_dir / f"{sym}.parquet"
    feat_fp = feat_dir / f"{sym}_features.parquet"
    if not data_fp.exists():
        logger.warning("Missing data parquet for %s: %s", sym, data_fp)
        return None
    if not feat_fp.exists():
        logger.warning("Missing features parquet for %s: %s", sym, feat_fp)
        return None

    df_data = canonicalize_polars_columns(pl.read_parquet(data_fp))
    df_feat = canonicalize_polars_columns(pl.read_parquet(feat_fp))

    # Ensure date column exists and types are comparable
    if Col.DATE.value not in df_data.columns:
        raise KeyError(f"'date' column not found in data parquet for {sym}")
    if Col.DATE.value not in df_feat.columns:
        raise KeyError(f"'date' column not found in features parquet for {sym}")

    # Select only needed price columns to join
    df_join = df_feat.join(df_data.select([Col.DATE.value, Col.CLOSE.value]), on=Col.DATE.value, how="left")
    df_join = df_join.sort(Col.DATE.value)
    return df_join


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    meta = load_cfg()
    cfg = meta["cfg"]
    paths_abs: Dict[str, str] = meta["paths_abs"]

    data_dir = Path(paths_abs["data_dir"])  # absolute
    feat_dir = Path(paths_abs["features_dir"])  # absolute
    res_dir = Path(paths_abs["results_dir"])  # absolute
    res_dir.mkdir(parents=True, exist_ok=True)

    bt = cfg.get("backtest", {})
    slip_bps = float(bt.get("slippage_bps", 0.0))
    fee_bps = float(bt.get("fee_bps", 0.0))
    slip = slip_bps / 1e4
    fee = fee_bps / 1e4

    risk = cfg.get("risk", {})
    risk_dollars = float(risk.get("per_trade_dollar", 25.0))

    senti = cfg.get("sentiment", {})
    use_sic = bool(senti.get("enable", False))
    sic_threshold = float(senti.get("sic_threshold", 0.0))

    uni = [str(s).upper() for s in cfg.get("universe", [])]
    if not uni:
        logger.error("Universe is empty; aborting")
        return

    trades: List[dict] = []

    # Horizon for naive exit; reuse labels.time_bar_days if present
    labels_cfg = cfg.get("labels", {})
    horizon_days = int(labels_cfg.get("time_bar_days", 10))

    for sym in uni:
        df = _load_symbol_frames(sym, data_dir, feat_dir)
        if df is None:
            continue

        # Generate entry signals (Polars)
        sic_col = "sic" if "sic" in df.columns else None
        signal = gated_entry(
            df,
            sic_value=sic_col if sic_col else 0.0,
            use_sic=use_sic,
            sic_threshold=sic_threshold,
        )
        # signal is a pl.Series aligned to df
        sig_series = signal if isinstance(signal, pl.Series) else pl.Series(values=signal)
        idxs = [i for i, v in enumerate(sig_series) if bool(v)]

        if not idxs:
            logger.info("%s: no entry signals", sym)
            continue

        close = df.get_column(Col.CLOSE.value)
        atr = df.get_column(Col.ATR14.value) if Col.ATR14.value in df.columns else None
        dates = df.get_column(Col.DATE.value)
        n = df.height

        for i in idxs:
            j = min(i + horizon_days, n - 1)

            c_in = close[i]
            c_out = close[j]
            if c_in is None or c_out is None:
                continue
            try:
                c_in_f = float(c_in)
                c_out_f = float(c_out)
            except Exception:
                continue

            # Get ATR for sizing; skip if missing/nonpositive
            atr_i = None
            if atr is not None:
                atr_i = atr[i]
            if atr_i is None:
                qty = 0
            else:
                try:
                    qty = int(position_size(float(atr_i), risk_dollars=risk_dollars))
                except Exception:
                    qty = 0
            if qty <= 0:
                continue

            entry_price = c_in_f * (1.0 + slip)
            exit_price = c_out_f * (1.0 - slip)

            fees_cost = fee * entry_price * qty + fee * exit_price * qty
            pnl = (exit_price - entry_price) * qty - fees_cost

            entry_date = str(dates[i])
            exit_date = str(dates[j])

            trades.append(
                {
                    "symbol": sym,
                    "entry_idx": i,
                    "exit_idx": j,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )

    # Persist trades and print summary
    if trades:
        df_trades = pl.DataFrame(trades)
        out_path = res_dir / "trades.parquet"
        df_trades.write_parquet(out_path)
        total_pnl = float(df_trades.get_column("pnl").sum())
        final_equity = 10_000.0 + total_pnl
        print(json.dumps({"trades": df_trades.height, "final_equity": round(final_equity, 2)}))
    else:
        # Still create an empty file for consistency
        out_path = res_dir / "trades.parquet"
        empty = pl.DataFrame(schema={
            "symbol": pl.Utf8,
            "entry_idx": pl.Int64,
            "exit_idx": pl.Int64,
            "entry": pl.Float64,
            "exit": pl.Float64,
            "qty": pl.Int64,
            "pnl": pl.Float64,
            "entry_date": pl.Utf8,
            "exit_date": pl.Utf8,
        })
        empty.write_parquet(out_path)
        print(json.dumps({"trades": 0, "final_equity": 10_000.0}))


if __name__ == "__main__":
    main()
