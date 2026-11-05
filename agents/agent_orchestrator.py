"""Agent orchestration via LangGraph.

Graph: news_agent -> macro_agent -> END. Each node is independent and
self-contained; both produce strictly-validated outputs persisted via
AgentOutputRecord for auditability. No agent receives execution, portfolio or
risk-engine handles. The reviewer runs in the decision pipeline on each
proposal (Phase 4)."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from core.clock import get_clock
from core.interfaces import MacroObservation, NewsArticle
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import AgentOutputRecord, NewsArticle as NewsArticleORM
from agents.llm_client import LLMClient
from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.schemas.macro_schema import MacroAnalysis
from agents.schemas.news_schema import NewsAnalysis

log = get_logger()


class IntelligenceState(TypedDict, total=False):
    articles: list[NewsArticle]
    macro_observations: list[MacroObservation]
    market_context: dict
    news_analysis: NewsAnalysis
    macro_analysis: MacroAnalysis


def _persist(agent_id: str, analysis, kind: str) -> None:
    try:
        with get_session() as session:
            session.add(AgentOutputRecord(
                agent_id=agent_id, ts=get_clock().now(),
                run_id=f"{agent_id}-{int(get_clock().now().timestamp())}",
                payload_json=analysis.model_dump_json(), status="ok"))
            session.commit()
    except Exception:
        log.warning("agent_output_persist_failed", agent=agent_id)


def _news_node(client: Optional[LLMClient]):
    def node(state: IntelligenceState) -> IntelligenceState:
        analysis = NewsAgent(client=client).analyze(state.get("articles", []))
        _persist(NewsAgent.agent_id, analysis, "news_analysis")
        return {"news_analysis": analysis}
    return node


def _macro_node(client: Optional[LLMClient]):
    def node(state: IntelligenceState) -> IntelligenceState:
        analysis = MacroAgent(client=client).analyze(state.get("macro_observations", []),
                                                     state.get("market_context"))
        _persist(MacroAgent.agent_id, analysis, "macro_analysis")
        return {"macro_analysis": analysis}
    return node


def build_intelligence_graph(client: Optional[LLMClient] = None):
    g = StateGraph(IntelligenceState)
    g.add_node("news", _news_node(client))
    g.add_node("macro", _macro_node(client))
    g.add_edge("news", "macro")
    g.add_edge("macro", END)
    g.set_entry_point("news")
    return g.compile()


def load_recent_articles(hours: int = 36) -> list[NewsArticle]:
    now = get_clock().now()
    with get_session() as session:
        rows = (session.query(NewsArticleORM)
                .filter(NewsArticleORM.published_ts >= now - timedelta(hours=hours))
                .order_by(NewsArticleORM.published_ts)
                .limit(200).all())
        return [NewsArticle(external_id=r.external_id, url=r.url, title=r.title,
                            summary=r.summary or "", source=r.source,
                            source_reliability=r.source_reliability,
                            published_ts=r.published_ts, ingested_ts=r.ingested_ts)
                for r in rows]


class AgentOrchestrator:
    """Entry point used by the trading loop: returns the latest intelligence."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client
        self.graph = build_intelligence_graph(client)

    def run(self, articles: list[NewsArticle],
            macro_observations: list[MacroObservation],
            market_context: Optional[dict] = None) -> tuple[NewsAnalysis, MacroAnalysis]:
        state: IntelligenceState = {"articles": articles,
                                    "macro_observations": macro_observations,
                                    "market_context": market_context or {}}
        out = self.graph.invoke(state)
        return out["news_analysis"], out["macro_analysis"]
