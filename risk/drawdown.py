"""Drawdown ladder and loss-streak state. Pure functions over PortfolioSnapshot."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_risk_config
from core.interfaces import PortfolioSnapshot


@dataclass(frozen=True)
class DrawdownState:
    level: str          # "ok" | "reduce_risk" | "halt_new_trades" | "emergency_shutdown"
    risk_multiplier: float
    threshold_pct: float


def drawdown_state(snapshot: PortfolioSnapshot) -> DrawdownState:
    ladder = sorted(get_risk_config()["drawdown_ladder"],
                    key=lambda r: r["threshold_pct"])
    dd = snapshot.drawdown_pct
    state = DrawdownState("ok", 1.0, 0.0)
    for rung in ladder:
        if dd >= rung["threshold_pct"]:
            state = DrawdownState(rung["action"], float(rung["risk_multiplier"]),
                                  float(rung["threshold_pct"]))
    return state


def daily_loss_exceeded(snapshot: PortfolioSnapshot) -> tuple[bool, float]:
    limit = get_risk_config()["risk"]["max_daily_loss_pct"]
    return snapshot.daily_loss_pct > limit, snapshot.daily_loss_pct


def loss_streak_exceeded(snapshot: PortfolioSnapshot) -> tuple[bool, int]:
    limit = get_risk_config()["risk"]["max_consecutive_losses"]
    return snapshot.consecutive_losses >= limit, snapshot.consecutive_losses
