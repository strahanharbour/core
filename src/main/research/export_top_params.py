from __future__ import annotations

from pathlib import Path
import polars as pl
import yaml


def main() -> None:
    results_dir = Path("src/main/artifacts/results")
    ranked_fp = results_dir / "ranked_strategies.csv"
    if not ranked_fp.exists():
        raise FileNotFoundError(f"Missing {ranked_fp}. Run: python -m research.rank_strategies")

    df = pl.read_csv(ranked_fp)
    if df.height == 0:
        raise RuntimeError("ranked_strategies.csv is empty")

    top = df.row(0, named=True)
    out = {
        "strategy": {
            "rsi_cross_level": float(top["rsi_cross_level"]),
            "vol_mult_min": float(top["vol_mult_min"]),
            "macd_hist_floor": float(top["macd_hist_floor"]),
        }
    }
    # Optional sentiment toggles if present
    if "sic_enabled" in df.columns:
        out.setdefault("sentiment", {})["enabled"] = bool(top.get("sic_enabled", False))
        out["sentiment"]["sic_threshold"] = float(top.get("sic_threshold", 0.0))

    out_fp = results_dir / "best_params.yaml"
    out_fp.write_text(yaml.dump(out, sort_keys=False), encoding="utf-8")
    print("Wrote", out_fp)


if __name__ == "__main__":
    main()

