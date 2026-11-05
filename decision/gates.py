"""Trading gates. A failed critical gate results in NO TRADE, and every
no-trade decision is recorded with its reason."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.interfaces import AssetHealthReport, MarketRegime

# Regimes where new entries are banned entirely
_BLOCKED_REGIMES = set()  # none hard-blocked; suitability scaling instead


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def reason(self) -> Optional[str]:
        return "; ".join(self.failures) if self.failures else None


def check_data_health(report: AssetHealthReport) -> GateResult:
    if not report.trade_eligible:
        return GateResult(False, [f"data_health:{f.flag.value}:{f.severity.value}"
                                  for f in report.findings if f.severity.value in ("critical", "warning")
                                  and f.flag.value != "missing_observations"] or ["data_health:ineligible"])
    return GateResult(True)


def check_regime_suitability(regime: str, direction: str) -> GateResult:
    from config.settings import get_trading_config
    suit = get_trading_config()["decision"]["regime_suitability"].get(
        regime, {"long": 0.0, "short": 0.0})
    score = float(suit.get(direction, 0.0))
    if score <= 0.0:
        return GateResult(False, [f"regime_unsuitable:{regime}:{direction}"])
    if score < 0.5:
        return GateResult(True, warnings=[f"weak_regime:{regime}:{direction}:{score}"])
    return GateResult(True)


def check_forecast_uncertainty(expected_return: float, uncertainty: float) -> GateResult:
    from config.settings import get_trading_config
    max_ratio = get_trading_config()["decision"]["max_uncertainty_ratio"]
    if uncertainty <= 0:
        return GateResult(False, ["forecast_uncertainty:nonpositive"])
    if abs(expected_return) > uncertainty * max_ratio:
        # forecast claims a move implausibly large vs its own uncertainty
        return GateResult(False, [f"forecast_implausible:{abs(expected_return):.4f}"
                                  f">={max_ratio}*{uncertainty:.4f}"])
    return GateResult(True)


def check_upcoming_releases(now: datetime,
                            upcoming_risks: Optional[list[str]] = None) -> GateResult:
    """Scheduled-release blackout: news agent supplies upcoming risks as text;
    any release flagged within the decision window blocks new entries."""
    if not upcoming_risks:
        return GateResult(True)
    critical_kw = ("cpi", "nonfarm", "fomc", "payroll", "fed decision", "rate decision",
                   "ppi", "gdp")
    for risk in upcoming_risks:
        rl = (risk or "").lower()
        if any(k in rl for k in critical_kw) and any(w in rl for w in ("today", "tonight", "imminent", "within hours")):
            return GateResult(False, [f"upcoming_release:{risk}"])
    return GateResult(True, warnings=[f"upcoming_event:{r}" for r in upcoming_risks[:3]])


def check_news_conflicts(conflicts: Optional[list[str]]) -> GateResult:
    if conflicts:
        return GateResult(False, [f"news_conflict:{c[:120]}" for c in conflicts[:3]])
    return GateResult(True)
