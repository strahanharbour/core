from __future__ import annotations
import os, time, html, re, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
import requests

try:
    import feedparser  # RSS
except Exception:
    feedparser = None

try:
    import yfinance as yf
except Exception:
    yf = None

from config_env import load_cfg

# ----------------- utils -----------------
def _clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_dt(ts: Any) -> datetime:
    # best-effort to UTC
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(ts, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
        # fallback: now
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc)
    return datetime.now(timezone.utc)

def _within_window(dt: datetime, since: Optional[datetime]) -> bool:
    return True if since is None else (dt >= since)

def _rows_to_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame({"date": [], "symbol": [], "title": [], "summary": [], "url": [], "provider": []})
    return pl.DataFrame(rows).with_columns([
        pl.col("date").cast(pl.Datetime),
        pl.col("symbol").cast(pl.Utf8),
        pl.col("title").cast(pl.Utf8),
        pl.col("summary").cast(pl.Utf8),
        pl.col("url").cast(pl.Utf8),
        pl.col("provider").cast(pl.Utf8),
    ])

# ----------------- providers -----------------
def fetch_newsapi(sym: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return []
    # NewsAPI everything endpoint via company symbol as query
    # Note: free plan may have restrictions; this is best-effort.
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": sym,
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": key,
        "language": "en",
    }
    out = []
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for a in (data.get("articles") or []):
            dt = _to_dt(a.get("publishedAt"))
            if not _within_window(dt, since):
                continue
            title = _clean_text(a.get("title") or "")
            desc = _clean_text(a.get("description") or "")
            url_ = a.get("url") or ""
            out.append({"date": dt, "symbol": sym, "title": title, "summary": desc, "url": url_, "provider": "newsapi"})
    except Exception:
        return out
    return out

def fetch_finnhub(sym: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return []
    # Finnhub company-news requires from/to dates
    end = datetime.now(timezone.utc).date()
    start = (since.date() if since else (end - timedelta(days=14)))
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": sym, "from": start.isoformat(), "to": end.isoformat(), "token": key}
    out = []
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for it in data or []:
            dt = _to_dt(it.get("datetime") or it.get("date"))
            if not _within_window(dt, since):
                continue
            title = _clean_text(it.get("headline") or "")
            summ = _clean_text(it.get("summary") or "")
            url_ = it.get("url") or ""
            out.append({"date": dt, "symbol": sym, "title": title, "summary": summ, "url": url_, "provider": "finnhub"})
    except Exception:
        return out
    return out

def fetch_yahoo_rss(sym: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
    if feedparser is None:
        return []
    rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
    out = []
    try:
        feed = feedparser.parse(rss)
        for e in (feed.get("entries") or []):
            # published_parsed is time.struct_time in UTC
            ts = e.get("published_parsed")
            dt = _to_dt(time.mktime(ts) if ts else datetime.now(timezone.utc).timestamp())
            if not _within_window(dt, since):
                continue
            title = _clean_text(e.get("title", ""))
            summ = _clean_text(e.get("summary", ""))
            url_ = e.get("link", "")
            out.append({"date": dt, "symbol": sym, "title": title, "summary": summ, "url": url_, "provider": "yahoo_rss"})
    except Exception:
        return out
    return out

def fetch_yfinance(sym: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
    if yf is None:
        return []
    out = []
    try:
        items = yf.Ticker(sym).news or []
        for it in items:
            dt = _to_dt(it.get("providerPublishTime") or it.get("time_published") or it.get("published_at"))
            if not _within_window(dt, since):
                continue
            title = _clean_text(it.get("title") or "")
            summ = _clean_text(it.get("summary") or "")
            url_ = it.get("link") or it.get("url") or ""
            out.append({"date": dt, "symbol": sym, "title": title, "summary": summ, "url": url_, "provider": "yfinance"})
    except Exception:
        return out
    return out

PROVIDER_FUNCS = {
    "newsapi": fetch_newsapi,
    "finnhub": fetch_finnhub,
    "yahoo_rss": fetch_yahoo_rss,
    "yfinance": fetch_yfinance,
}

# ----------------- main -----------------
def main():
    meta = load_cfg()
    cfg = meta.get("cfg", {}) or {}
    paths = cfg.get("paths", {}) or {}
    sent = cfg.get("sentiment", {}) or {}

    news_dir = Path(sent.get("storage_dir", "src/main/artifacts/altdata/news"))
    providers = [p.lower() for p in sent.get("providers", ["yfinance", "yahoo_rss"])]
    window_days = int(sent.get("window_days", 14))
    since_dt = datetime.now(timezone.utc) - timedelta(days=window_days) if window_days > 0 else None

    universe = cfg.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"])
    news_dir.mkdir(parents=True, exist_ok=True)

    for sym in universe:
        all_rows: List[Dict[str, Any]] = []
        for p in providers:
            fn = PROVIDER_FUNCS.get(p)
            if not fn:
                continue
            rows = fn(sym, since_dt)
            if rows:
                all_rows.extend(rows)
            # be polite to APIs
            time.sleep(0.25)

        if not all_rows:
            continue

        _df = _rows_to_df(all_rows)
        # De-duplicate by (url, title, date) if present; prefer url first
        has_url = _df["url"].fill_null("").str.len_chars() > 0
        dedup = (
            pl.concat([
                _df.filter(has_url).unique(subset=["symbol", "url"]),
                _df.filter(~has_url).unique(subset=["symbol", "title", "date"]),
            ], how="vertical_relaxed")
            .sort("date")
        )
        # write/overwrite per-symbol parquet
        outp = news_dir / f"{sym}_news.parquet"
        dedup.write_parquet(outp)
        print(f"{sym}: saved {dedup.height} headlines → {outp}")

if __name__ == "__main__":
    main()
