"""News Intelligence Agent.

Deterministic pre-processing (dedup, staleness, conflict detection) happens
here; only the structured extraction goes to the LLM. All article text is
wrapped as untrusted content. Output is strictly validated; on LLM failure the
agent degrades to an empty-but-valid analysis with warnings."""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional

from core.clock import get_clock
from core.exceptions import LLMError
from core.interfaces import NewsArticle
from core.logging_setup import get_logger
from agents.llm_client import LLMClient, wrap_untrusted
from agents.prompts.news_prompt import NEWS_SYSTEM_PROMPT, news_user_prompt
from agents.schemas.news_schema import NewsAnalysis

log = get_logger()

_ASSETS = ("XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD")


class NewsAgent:
    agent_id = "news_intelligence"

    def __init__(self, client: Optional[LLMClient] = None, lookback_hours: int = 36):
        self.client = client
        self.lookback = timedelta(hours=lookback_hours)

    def analyze(self, articles: Iterable[NewsArticle],
                known_event_titles: Optional[list[str]] = None) -> NewsAnalysis:
        known = [t.lower() for t in (known_event_titles or [])]
        fresh, dup, stale = self._pre_filter(list(articles))
        if not fresh:
            return NewsAnalysis(events=[], duplicates_removed=dup,
                                stale_or_recycled=stale,
                                overall_summary="No fresh market-relevant articles.",
                                warnings=["no_fresh_articles"] if not stale else [])

        block = self._articles_block(fresh)
        known_block = "\n".join(f"- {t}" for t in (known_event_titles or [])) or "(none)"
        try:
            client = self.client or LLMClient()
            analysis = client.structured(NEWS_SYSTEM_PROMPT,
                                         news_user_prompt(block, known_block),
                                         NewsAnalysis)
            analysis = self._sanitize(analysis, fresh)
            analysis.duplicates_removed = dup
            analysis.stale_or_recycled = analysis.stale_or_recycled or stale
            return analysis
        except LLMError as e:
            log.warning("news_agent_degraded", error=str(e))
            return NewsAnalysis(events=[], duplicates_removed=dup,
                                stale_or_recycled=stale,
                                overall_summary="LLM unavailable; events not extracted.",
                                warnings=[f"llm_unavailable:{e}"])

    # ---- deterministic pre-processing ----------------------------------
    def _pre_filter(self, articles: list[NewsArticle]) -> tuple[list[NewsArticle], int, list[str]]:
        now = get_clock().now()
        fresh: list[NewsArticle] = []
        stale: list[str] = []
        seen_titles: dict[str, NewsArticle] = {}
        dup = 0
        for a in sorted(articles, key=lambda x: x.published_ts):
            if now - a.published_ts > self.lookback:
                stale.append(a.external_id)
                continue
            key = " ".join(a.title.lower().split())
            if key in seen_titles:
                dup += 1
                continue
            seen_titles[key] = a
            fresh.append(a)
        return fresh, dup, stale

    def _articles_block(self, articles: list[NewsArticle]) -> str:
        parts = []
        for a in articles[:30]:
            parts.append(
                f"external_id={a.external_id}\n"
                f"published={a.published_ts.isoformat()} source={a.source} "
                f"reliability={a.source_reliability.value}\n"
                f"title: {a.title}\nsummary: {wrap_untrusted(a.summary, max_chars=1200)}"
            )
        return "\n---\n".join(parts)

    def _sanitize(self, analysis: NewsAnalysis, articles: list[NewsArticle]) -> NewsAnalysis:
        """LLM output is untrusted: clamp asset symbols and source references."""
        valid_ids = {a.external_id for a in articles}
        valid_events = []
        for ev in analysis.events:
            ev.affected_assets = [s for s in ev.affected_assets if s in _ASSETS]
            ev.source_references = [r for r in ev.source_references if r in valid_ids]
            if ev.source_references:  # unverifiable events (no source link) are dropped
                valid_events.append(ev)
        return analysis.model_copy(update={"events": valid_events})
