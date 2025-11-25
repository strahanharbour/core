import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def main() -> None:
    start = datetime(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(30)]
    rows = []
    syms = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    for dt in dates:
        for s in syms:
            ret = rng.normal(0.0005, 0.01)
            p = np.clip(rng.uniform(0.4, 0.9), 0, 1)
            atr = rng.uniform(0.5, 2.0)
            rows.append({
                "dt": dt,
                "ticker": s,
                "ret_next": float(ret),
                "p_cal": float(p),
                "atr": float(atr),
                "sic": 0.1,
                "regime": "mid",
            })
    out = Path("tmp_meta.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(str(out))

if __name__ == "__main__":
    main()

