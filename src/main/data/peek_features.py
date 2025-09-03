from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import polars as pl

from config_env import load_cfg
from constants import Col


def _features_dir() -> Path:
    meta = load_cfg()
    paths_abs = meta.get("paths_abs", {})
    d = paths_abs.get("features_dir")
    if d:
        return Path(d)
    # Fallback near this file if config missing
    return Path(__file__).resolve().parent / "artifacts" / "features"


def _find_feature_file(sym: str | None, root: Path) -> Path:
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {root}")
    if not sym:
        return files[0]

    s = sym.strip().upper()
    exact = root / f"{s}_features.parquet"
    if exact.exists():
        return exact

    # Fuzzy match on prefix to help discover
    cands = [f for f in files if f.name.upper().startswith(s) and f.name.upper().endswith("_FEATURES.PARQUET")]
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise FileExistsError(
            f"Multiple matches for '{sym}': {[p.name for p in cands]}. "
            "Please specify the exact symbol."
        )
    raise FileNotFoundError(f"No features parquet found for '{sym}' in {root}")


def list_files(root: Path) -> List[str]:
    return [p.name for p in sorted(root.glob("*.parquet"))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Peek generated feature parquet files")
    parser.add_argument("symbol", nargs="?", help="Symbol to preview (e.g., SPY). If omitted, uses first file.")
    parser.add_argument("-n", "--rows", type=int, default=5, help="Number of rows to show (default: 5)")
    parser.add_argument("--all-cols", action="store_true", help="Show all columns (not just features)")
    parser.add_argument("--schema", action="store_true", help="Print schema only")
    parser.add_argument("--list", action="store_true", help="List available feature files and exit")
    parser.add_argument("--nulls", action="store_true", help="Show per-column null counts")
    args = parser.parse_args()

    root = _features_dir()
    root.mkdir(parents=True, exist_ok=True)

    if args.list:
        for name in list_files(root):
            print(name)
        return

    fp = _find_feature_file(args.symbol, root)
    print(f"File: {fp}")
    df = pl.read_parquet(fp)

    if args.schema:
        print(df.schema)
        return

    # Determine feature columns; import from builder if available to avoid drift
    try:
        from data.build_features import FEATURE_COLS as _FEATURE_COLS  # type: ignore
        feature_cols = list(_FEATURE_COLS)
    except Exception:
        feature_cols = [
            Col.RSI14.value,
            Col.ATR14.value,
            Col.MACD.value,
            Col.MACD_SIGNAL.value,
            Col.MACD_HIST.value,
            Col.VWAP.value,
            Col.VOL_MULT.value,
        ]

    if args.all_cols:
        view = df.head(args.rows)
    else:
        base_candidates = [Col.DATE.value, Col.SYMBOL.value, "price"]
        base = [c for c in base_candidates if c in df.columns]
        cols = base + [c for c in feature_cols if c in df.columns]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"Warning: missing columns in file: {missing}")
        view = df.select(cols).head(args.rows)

    print(view)

    if args.nulls:
        print("\nNull counts:")
        print(df.select(pl.all().null_count()))


if __name__ == "__main__":
    main()
