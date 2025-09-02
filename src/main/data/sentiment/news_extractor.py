from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import requests

from config_env import load_cfg


logger = logging.getLogger(__name__)

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_news(
    symbols: Iterable[str],
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    *,
    language: str = "en",
    page_size: int = 50,
) -> Optional[List[Dict]]:
    """
    Fetch recent news for a list of symbols using NewsAPI.
    Reads API key via config_env.load_cfg() (env: NEWS_API_KEY). If missing, returns None.
    """
    meta = load_cfg()
    api_key = (meta.get("env") or {}).get("NEWS_API_KEY")
    if not api_key:
        logger.warning("NEWS_API_KEY not set. Skipping NewsAPI fetch.")
        return None

    if start is None:
        start_dt = datetime.now(timezone.utc) - timedelta(days=7)
    elif isinstance(start, str):
        start_dt = datetime.fromisoformat(start)
    else:
        start_dt = start

    if end is None:
        end_dt = datetime.now(timezone.utc)
    elif isinstance(end, str):
        end_dt = datetime.fromisoformat(end)
    else:
        end_dt = end

    headers = {"X-Api-Key": api_key}
    out: List[Dict] = []
    for sym in {s.upper().strip() for s in symbols if s and str(s).strip()}:
        params = {
            "q": sym,
            "searchIn": "title,description",
            "language": language,
            "from": _iso(start_dt),
            "to": _iso(end_dt),
            "sortBy": "publishedAt",
            "pageSize": page_size,
        }
        try:
            resp = requests.get(NEWS_ENDPOINT, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            for a in data.get("articles", []) or []:
                out.append(
                    {
                        "symbol": sym,
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "publishedAt": a.get("publishedAt"),
                        "url": a.get("url"),
                        "source": (a.get("source") or {}).get("name"),
                    }
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("NewsAPI fetch failed for %s: %s", sym, e)
            continue
    return out


__all__ = ["fetch_news"]

