from __future__ import annotations

from pathlib import Path
import polars as pl

from main.constants import Col
from main.config_env import load_cfg


def build_labels_for_symbol(
    sym: str, data_dir: Path, out_dir: Path, up_k: float, dn_k: float, max_h: int
) -> None:
    f = data_dir / f"{sym}.parquet"
    if not f.exists():
        return
    df = pl.read_parquet(f).select([Col.DATE.value, Col.CLOSE.value]).sort(Col.DATE.value)
    n = df.height
    out = []
    # Compute simplistic triple-barrier label per starting bar
    for i in range(n - 1):
        px0 = float(df[Col.CLOSE.value][i])
        up = px0 * (1.0 + up_k)
        dn = px0 * (1.0 - dn_k)
        i_end = min(n - 1, i + max_h)
        label = 0
        for j in range(i + 1, i_end + 1):
            px = float(df[Col.CLOSE.value][j])
            if px >= up:
                label = 1
                break
            if px <= dn:
                label = -1
                break
        out.append({"date": df[Col.DATE.value][i], "label": label})
    pl.DataFrame(out).write_parquet(out_dir / f"{sym}_labels.parquet")


def main() -> None:
    cfg = load_cfg()["cfg"]
    paths = cfg.get("paths", {}) or {}
    data_dir = Path(paths.get("data_dir", "src/main/artifacts/local_data"))
    feat_dir = Path(paths.get("features_dir", "src/main/artifacts/features"))
    out_dir = feat_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    ml = (cfg.get("metalabel", {}) or {})
    up_k = float(ml.get("upper_pct", 0.02))
    dn_k = float(ml.get("lower_pct", 0.02))
    max_h = int(ml.get("max_hold_bars", 10))

    universe = cfg.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
    for sym in universe:
        build_labels_for_symbol(sym, data_dir, out_dir, up_k, dn_k, max_h)


if __name__ == "__main__":
    main()
