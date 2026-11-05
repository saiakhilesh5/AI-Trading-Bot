"""News collector: RSS feeds (feedparser) + GDELT DOC 2.0 API (free).
Article content is untrusted input — never executed, never followed
as instructions. Deduplication by (source, url) hash."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import httpx

from core.clock import get_clock
from core.exceptions import DataProviderError
from core.interfaces import NewsArticle
from data_layer.validators.source_validator import dedup_key, reliability_of

DEFAULT_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]

GDELT_QUERY = (
    '(gold OR silver OR bitcoin OR ethereum) AND '
    '(inflation OR "federal reserve" OR rates OR regulation)'
)


class NewsCollector:
    source_id = "rss"

    def __init__(self, feeds: Optional[list[str]] = None, use_gdelt: bool = True):
        self.feeds = feeds or DEFAULT_FEEDS
        self.use_gdelt = use_gdelt

    def fetch_articles(self, since: Optional[datetime] = None) -> list[NewsArticle]:
        now = get_clock().now()
        cutoff = since or (now - timedelta(hours=24))
        articles: list[NewsArticle] = []
        seen: set[str] = set()

        for url in self.feeds:
            parsed = feedparser.parse(url)
            if parsed.bozo and not parsed.entries:
                continue
            source = _feed_label(parsed.feed.get("link", url))
            reliability = reliability_of(source)
            for entry in parsed.entries:
                link = entry.get("link", "")
                title = entry.get("title", "")
                if not link or not title:
                    continue
                published = _entry_ts(entry)
                if published is None or published < cutoff:
                    continue
                ext = dedup_key(source, link)
                if ext in seen:
                    continue
                seen.add(ext)
                try:
                    articles.append(NewsArticle(
                        external_id=ext, url=link, title=title,
                        summary=entry.get("summary", "")[:2000],
                        source=source, source_reliability=reliability,
                        published_ts=published, ingested_ts=now))
                except ValueError:
                    continue

        if self.use_gdelt:
            articles.extend(self._fetch_gdelt(now, cutoff, seen))

        return articles

    def _fetch_gdelt(self, now: datetime, cutoff: datetime, seen: set[str]) -> list[NewsArticle]:
        out: list[NewsArticle] = []
        try:
            resp = httpx.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": GDELT_QUERY.replace('"', "'"),
                    "mode": "artlist", "maxrecords": 50,
                    "sort": "datedesc", "format": "json",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return out  # GDELT is supplementary: failure is non-fatal
        for art in data.get("articles", []):
            url_ = art.get("url", "")
            title = art.get("title", "")
            if not url_ or not title:
                continue
            try:
                published = datetime.fromisoformat(art["seendate"].replace("Z", "+00:00"))
                published = published.astimezone(timezone.utc)
            except (KeyError, ValueError):
                continue
            if published < cutoff:
                continue
            source = art.get("domain", "gdelt")
            ext = dedup_key(source, url_)
            if ext in seen:
                continue
            seen.add(ext)
            try:
                out.append(NewsArticle(
                    external_id=ext, url=url_, title=title[:500],
                    summary=art.get("description", "")[:500],
                    source=source, source_reliability=reliability_of("gdelt"),
                    published_ts=published, ingested_ts=now))
            except ValueError:
                continue
        return out


def _feed_label(link: str) -> str:
    link = link.lower()
    for name in ("cnbc", "wsj", "dj.com", "cointelegraph", "coindesk", "reuters", "ft.com"):
        if name in link:
            return "wsj" if name == "dj.com" else name
    return link.split("//")[-1].split("/")[0]


def _entry_ts(entry) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None
