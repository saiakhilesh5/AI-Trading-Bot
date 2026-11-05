"""Macroeconomic Interpretation Agent.

Consumes point-in-time macro observations (as-of current clock), plus a small
deterministic market-context block. The LLM interprets; it never computes
indicators. Output validated against MacroAnalysis; degrades gracefully."""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from core.clock import get_clock
from core.exceptions import LLMError
from core.interfaces import MacroObservation
from core.logging_setup import get_logger
from agents.llm_client import LLMClient, wrap_untrusted
from agents.prompts.macro_prompt import MACRO_SYSTEM_PROMPT, macro_user_prompt
from agents.schemas.macro_schema import MacroAnalysis

log = get_logger()

_SERIES_NAMES = {
    "DGS10": "US 10Y Treasury yield", "DGS2": "US 2Y Treasury yield",
    "T10YIE": "10Y breakeven inflation", "CPIAUCSL": "CPI (all urban)",
    "UNRATE": "Unemployment rate", "PAYEMS": "Nonfarm payrolls",
    "DXY": "US Dollar Index",
}


class MacroAgent:
    agent_id = "macro_intelligence"

    def __init__(self, client: Optional[LLMClient] = None, lookback_days: int = 90):
        self.client = client
        self.lookback = timedelta(days=lookback_days)

    def analyze(self, observations: list[MacroObservation],
                market_context: Optional[dict] = None) -> MacroAnalysis:
        as_of = get_clock().now()
        recent = [o for o in observations if as_of - o.release_ts <= self.lookback]
        if not recent:
            return MacroAnalysis(regime_summary="No recent macro observations available.",
                                 warnings=["no_macro_data"])

        block = self._obs_block(recent, as_of)
        ctx = market_context or {}
        ctx_block = "\n".join(f"{k}: {v}" for k, v in ctx.items()) or "(none)"
        try:
            client = self.client or LLMClient()
            return client.structured(MACRO_SYSTEM_PROMPT,
                                      macro_user_prompt(block, ctx_block),
                                      MacroAnalysis)
        except LLMError as e:
            log.warning("macro_agent_degraded", error=str(e))
            return MacroAnalysis(regime_summary="LLM unavailable; macro not interpreted.",
                                 warnings=[f"llm_unavailable:{e}"])

    def _obs_block(self, obs: list[MacroObservation], as_of) -> str:
        by_series: dict[str, MacroObservation] = {}
        for o in sorted(obs, key=lambda x: x.release_ts):
            by_series[o.series_id] = o  # latest release wins
        lines = []
        for sid, o in sorted(by_series.items()):
            name = _SERIES_NAMES.get(sid, sid)
            lines.append(f"{sid} ({name}): value={o.value} "
                         f"reference_period={o.reference_period.date().isoformat()} "
                         f"released={o.release_ts.date().isoformat()} source={o.source}")
        # include a short history so the LLM can see direction of travel
        for sid in by_series:
            hist = [o for o in obs if o.series_id == sid][-6:]
            if len(hist) > 1:
                vals = ", ".join(f"{h.value}" for h in sorted(hist, key=lambda x: x.reference_period))
                lines.append(f"  {sid} recent values (oldest->newest): {vals}")
        return wrap_untrusted("\n".join(lines), max_chars=8000)
