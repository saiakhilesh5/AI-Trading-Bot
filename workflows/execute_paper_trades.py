"""Execute the BUY/SELL decisions of one cycle through the paper broker and
process SL/TP on current marks. Idempotent per decision id."""
from __future__ import annotations

from core.interfaces import DecisionAction
from core.logging_setup import get_logger
from execution.paper_broker import PaperBroker

log = get_logger()


def execute_decisions(decisions, marks: dict[str, float],
                      atr_by_asset: dict[str, float] | None = None) -> list:
    broker = PaperBroker()
    opened = []
    for d in decisions:
        if d.action not in (DecisionAction.BUY, DecisionAction.SELL):
            continue
        t = d.proposed_trade
        if t is None or t.asset_symbol not in marks:
            continue
        try:
            asset_class = "crypto" if t.asset_symbol in ("BTCUSD", "ETHUSD") else "metals"
            atr = (atr_by_asset or {}).get(t.asset_symbol, marks[t.asset_symbol] * 0.01)
            pos = broker.execute_decision(d, ref_price=marks[t.asset_symbol],
                                          asset_class=asset_class, atr=atr)
            if pos is not None:
                opened.append(pos.position_id)
        except Exception as e:
            log.warning("execution_failed", decision=d.decision_id, error=str(e))
    return opened


def process_brackets(marks: dict[str, float]) -> list[str]:
    broker = PaperBroker()
    return broker.process_marks(marks)
