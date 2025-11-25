from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import polars as pl
import pandas as pd

from main.config_env import load_cfg
from main.data.sentiment.score_vader import score_vader


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(p: str | Path) -> Path:
    pth = Path(p)
    if pth.is_absolute():
        return pth
    return (_repo_root() / pth).resolve()


def _load_raw_news(storage_dir: Path) -> pl.DataFrame | None:
    files = sorted(storage_dir.glob("*.parquet"))
    if not files:
        return None
    dfs: List[pl.DataFrame] = []
    for f in files:
        try:
            dfs.append(pl.read_parquet(f))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", f, e)
    if not dfs:
        return None
    df = pl.concat(dfs, how="vertical_relaxed")
    return df


def _daily_scores(df: pl.DataFrame, aggregation: str, ewm_span: int) -> Dict[str, pl.DataFrame]:
    # Ensure needed columns
    cols = df.columns
    if "symbol" not in cols:
        raise KeyError("'symbol' column missing in news data")
    # Build combined text and score
    if "publishedAt" in cols:
        df = df.with_columns(pl.col("publishedAt").str.strptime(pl.Datetime, strict=False))
        date_col = pl.col("publishedAt").dt.date().alias("date")
    elif "date" in cols:
        date_col = pl.col("date").dt.date().alias("date")
    else:
        raise KeyError("No timestamp column ('publishedAt' or 'date') in news data")

    desc_col = "summary" if "summary" in cols else ("description" if "description" in cols else None)
    if desc_col is None:
        df = df.with_columns([
            date_col,
            pl.col("title").map_elements(lambda t: score_vader((t or ""))).alias("score"),
        ])
    else:
        df = df.with_columns([
            date_col,
            pl.struct([pl.col("title"), pl.col(desc_col)]).map_elements(
                lambda s, dc=desc_col: score_vader(f"{(s['title'] or '')} {(s[dc] or '')}")
            ).alias("score"),
        ])

    # Aggregate per symbol, per day
    daily = df.group_by(["symbol", "date"]).agg(pl.mean("score").alias("score"))

    out: Dict[str, pl.DataFrame] = {}
    for sym, grp in daily.group_by("symbol"):
        g = grp.sort("date").select(["date", "score"]).to_pandas()
        if aggregation.lower() == "median":
            g2 = g.copy()
            # median per day already in group stage; keep as-is
        elif aggregation.lower() == "ewm":
            span = max(1, int(ewm_span))
            g2 = g.copy()
            # Use pandas EWM for simplicity
            g2["sic"] = pd.Series(g2["score"].to_numpy()).ewm(span=span, adjust=False).mean()
        else:  # mean
            g2 = g.copy()
            g2["sic"] = g2["score"].astype(float)
        out[sym] = pl.from_pandas(g2[["date", "sic"]])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    meta = load_cfg()
    cfg = meta.get("cfg", {}) or {}
    paths_abs = meta.get("paths_abs", {}) or {}

    feat_dir = Path(paths_abs.get("features_dir") or _resolve_path("src/main/artifacts/features"))
    out_dir = feat_dir / "sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_cfg = cfg.get("sentiment", {}) or {}
    storage_dir = _resolve_path(sent_cfg.get("storage_dir", "src/main/artifacts/altdata/news"))
    aggregation = str(sent_cfg.get("aggregation", "ewm"))
    ewm_span = int(sent_cfg.get("ewm_span", 5))

    df = _load_raw_news(storage_dir)
    if df is None or df.is_empty():
        logger.warning("No news files found under %s; nothing to build.", storage_dir)
        return

    per_sym = _daily_scores(df, aggregation=aggregation, ewm_span=ewm_span)
    if not per_sym:
        logger.warning("No daily scores computed; aborting.")
        return

    for sym, s_df in per_sym.items():
        fp = out_dir / f"{sym}_sic.parquet"
        s_df.write_parquet(fp)
    logger.info("Wrote SIC for %d symbols to %s", len(per_sym), out_dir)


if __name__ == "__main__":
    main()
