from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv, find_dotenv


ENV_KEYS = [
    "NEWS_API_KEY",
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_SECRET",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "DB_URL",
]


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


def _load_dotenv_nearby(base: Path) -> None:
    # Prefer .env discovered from current working directory upwards
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        return
    # Fallback to repo root two levels up from src/main
    repo_root = base.parents[2] if len(base.parents) >= 3 else base
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def load_cfg() -> Dict[str, Any]:
    """
    Load YAML config from src/main/config.yaml, merge environment variables
    from .env (via python-dotenv), resolve artifact paths, and ensure they exist.

    Returns a dict with keys:
      - cfg: original YAML content (dict)
      - env: selected environment variables (dict)
      - paths_abs: absolute paths for artifacts (strings)
    """
    base_dir = Path(__file__).resolve().parent  # src/main

    # Load .env into process environment without overriding existing
    _load_dotenv_nearby(base_dir)

    # Load YAML config living beside this file
    cfg_path = base_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Collect env keys
    env_map = {k: os.getenv(k) for k in ENV_KEYS if os.getenv(k) is not None}

    # Resolve artifact paths and ensure directories exist
    paths = (cfg.get("paths") or {}).copy()
    resolved: Dict[str, str] = {}
    for key in ("data_dir", "features_dir", "results_dir"):
        if key in paths and paths[key]:
            abs_path = _resolve_path(base_dir, paths[key])
            abs_path.mkdir(parents=True, exist_ok=True)
            resolved[key] = str(abs_path)

    return {
        "cfg": cfg,
        "env": env_map,
        "paths_abs": resolved,
    }


__all__ = ["load_cfg"]

