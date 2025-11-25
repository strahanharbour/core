from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import polars as pl

from main.config_env import load_cfg
from main.engine.strategy import position_size
from main.utils.paths import get_paths
from main.utils.polars_ext import join_on_date
from main.research.sim_lib import (
    load_cfg_bits,
    apply_sentiment_gate,
    apply_market_filter,
    apply_regime_gate,
    strict_entry_edge,
    exits_returns,
    signal_rules,
)
from main.constants import Col
from main.utils.columns import canonicalize_polars_columns


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


class Backtester:
    """Encapsulated backtest runner for daily, long-only signals.

    Keeps existing behavior: writes trades.parquet and prints a summary JSON.
    """

    def __init__(self) -> None:
        meta = load_cfg()
        self.cfg: Dict = meta.get("cfg", {}) or {}
        self.paths = get_paths()
        # Slippage/fees
        bt = self.cfg.get("backtest", {})
        self.slip = float(bt.get("slippage_bps", 0.0)) / 1e4
        self.fee = float(bt.get("fee_bps", 0.0)) / 1e4
        # Risk sizing dollars
        risk = self.cfg.get("risk", {})
        self.risk_dollars = float(risk.get("per_trade_dollar", 25.0))
        # Config bits for gates/exits
        _, self.paths_cfg, self.sent_cfg, self.strat_cfg = load_cfg_bits()
        # Universe
        self.universe = [str(s).upper() for s in self.cfg.get("universe", [])]
        self.results_dir = self.paths.results
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_df(self, sym: str) -> pl.DataFrame | None:
        return _load_symbol_frames(sym, self.paths.data, self.paths.features)

    def _pnl_from_trades(self, sym: str, df: pl.DataFrame, trades_df: pl.DataFrame) -> List[dict]:
        out: List[dict] = []
        df_idx = df.with_columns(pl.arange(0, pl.len()).alias("idx"))
        trades_j = join_on_date(
            trades_df,
            df_idx.select([Col.DATE.value, "idx", Col.ATR14.value]),
            left_on="entry_date",
            right_on=Col.DATE.value,
            how="left",
        )
        ex_cfg = (self.strat_cfg.get("exits") or {})
        max_h = int(ex_cfg.get("max_hold_bars", 10))
        for row in trades_j.iter_rows(named=True):
            entry_date = row["entry_date"]
            entry_px = float(row["entry_px"]) if row["entry_px"] is not None else None
            ret_h = float(row["ret_h"]) if row["ret_h"] is not None else None
            i0 = int(row.get("idx") or -1)
            atr_i = row.get(Col.ATR14.value)
            if entry_px is None or ret_h is None or atr_i is None:
                continue
            try:
                qty = int(position_size(float(atr_i), risk_dollars=self.risk_dollars))
            except Exception:
                qty = 0
            if qty <= 0:
                continue
            entry_price = entry_px * (1.0 + self.slip)
            exit_price = entry_px * (1.0 + ret_h) * (1.0 - self.fee)
            fees_cost = self.fee * entry_price * qty + self.fee * exit_price * qty
            pnl = (exit_price - entry_price) * qty - fees_cost
            exit_idx = i0 + max_h if i0 >= 0 else i0
            out.append(
                {
                    "symbol": sym,
                    "entry_idx": i0,
                    "exit_idx": exit_idx,
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "entry_date": str(entry_date),
                    "exit_date": str(entry_date),
                }
            )
        return out

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if not self.universe:
            logger.error("Universe is empty; aborting")
            return
        trades: List[dict] = []
        for sym in self.universe:
            df = self._load_df(sym)
            if df is None:
                continue
            sig = signal_rules(df)
            features_dir = Path(self.paths_cfg.get("features_dir", str(self.paths.features)))
            data_dir_for_mf = Path(self.paths_cfg.get("data_dir", str(self.paths.data)))
            sig = apply_sentiment_gate(df, sig, sym, features_dir, self.sent_cfg)
            sig = apply_market_filter(df, sig, data_dir_for_mf, self.strat_cfg)
            sig = apply_regime_gate(df, sig, data_dir_for_mf, self.strat_cfg)
            # Optional meta-label gate (triple-barrier acceptance)
            ml = (self.cfg.get("metalabel") or {})
            if bool(ml.get("enabled", False)):
                feat_dir = Path(self.paths_cfg.get("features_dir", str(self.paths.features)))
                lab_file = feat_dir / "labels" / f"{sym}_labels.parquet"
                if lab_file.exists():
                    lab = pl.read_parquet(lab_file).select(["date", "label"])
                    df = df.join(lab, on="date", how="left")
                    accept = set(ml.get("accept", [1]))
                    sig = sig & df.get_column("label").is_in(list(accept)).fill_null(False)
            entries = strict_entry_edge(sig)
            trades_df = exits_returns(df, entries, self.strat_cfg)
            if trades_df is None or trades_df.is_empty():
                logger.info("%s: no entry signals", sym)
                continue
            trades.extend(self._pnl_from_trades(sym, df, trades_df))

        out_path = self.results_dir / "trades.parquet"
        if trades:
            df_trades = pl.DataFrame(trades)
            df_trades.write_parquet(out_path)
            total_pnl = float(df_trades.get_column("pnl").sum())
            final_equity = 10_000.0 + total_pnl
            print(json.dumps({"trades": df_trades.height, "final_equity": round(final_equity, 2)}))
        else:
            empty = pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "entry_idx": pl.Int64,
                    "exit_idx": pl.Int64,
                    "entry": pl.Float64,
                    "exit": pl.Float64,
                    "qty": pl.Int64,
                    "pnl": pl.Float64,
                    "entry_date": pl.Utf8,
                    "exit_date": pl.Utf8,
                }
            )
            empty.write_parquet(out_path)
            print(json.dumps({"trades": 0, "final_equity": 10_000.0}))


def main() -> None:
    Backtester().run()


if __name__ == "__main__":
    main()
