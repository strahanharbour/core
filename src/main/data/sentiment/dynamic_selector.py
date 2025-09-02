from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Set

from config_env import load_cfg
from .score_vader import score_vader
from .news_extractor import fetch_news


logger = logging.getLogger(__name__)

SEED_SYMBOLS: List[str] = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def rank_symbols_by_sentiment(articles: Sequence[Dict]) -> List[tuple[str, float, int]]:
    """
    Return list of (symbol, avg_compound, count) sorted by avg_compound desc.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    for a in articles:
        sym = str(a.get("symbol", "")).upper()
        if not sym:
            continue
        text = " ".join([str(a.get("title") or ""), str(a.get("description") or "")]).strip()
        if not text:
            continue
        scores[sym].append(score_vader(text))
    ranked = [
        (sym, (sum(vals) / len(vals)) if vals else 0.0, len(vals))
        for sym, vals in scores.items()
    ]
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return ranked


def select_symbols(n_top: int = 5, seed: Iterable[str] | None = None) -> List[str]:
    """
    Dynamic selector with safe fallback.

    - If NEWS_API_KEY is unset, return union of SEED_SYMBOLS and config["universe"].
    - Else, include top-N ranked symbols by average VADER score from recent news (7 days),
      along with seed and config universe to ensure coverage.
    """
    meta = load_cfg()
    cfg = meta.get("cfg", {})
    env = meta.get("env", {})
    uni = [str(s).upper() for s in cfg.get("universe", [])]

    base: Set[str] = set(SEED_SYMBOLS)
    if seed:
        base |= {str(s).upper() for s in seed}
    base |= set(uni)

    if not env.get("NEWS_API_KEY"):
        logger.info("NEWS_API_KEY unset; using fallback selector (seed ∪ universe)")
        return sorted(base)

    # Pull recent news and rank
    end = _now_utc()
    start = end - timedelta(days=7)
    arts = fetch_news(base, start=start.isoformat(), end=end.isoformat()) or []
    ranked = rank_symbols_by_sentiment(arts)
    top_syms = [sym for sym, _, _ in ranked[: max(0, int(n_top))]]

    final = sorted(base | set(top_syms))
    return final


__all__ = ["select_symbols", "rank_symbols_by_sentiment", "SEED_SYMBOLS"]

