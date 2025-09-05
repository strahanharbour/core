from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config_env import load_cfg


@dataclass(frozen=True)
class Paths:
    data: Path
    features: Path
    results: Path


def get_paths() -> Paths:
    """Return resolved artifact paths as Path objects.

    Prefers absolute paths provided by config_env.load_cfg(); falls back to
    default relative locations under src/main/artifacts/...
    """
    meta = load_cfg()
    paths_abs = meta.get("paths_abs", {}) or {}
    base = Path(__file__).resolve().parents[2]  # src/main

    data = Path(paths_abs.get("data_dir") or (base / "artifacts" / "local_data"))
    features = Path(paths_abs.get("features_dir") or (base / "artifacts" / "features"))
    results = Path(paths_abs.get("results_dir") or (base / "artifacts" / "results"))
    return Paths(data=data, features=features, results=results)


__all__ = ["Paths", "get_paths"]

