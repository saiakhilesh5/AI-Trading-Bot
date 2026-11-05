"""Deterministic Risk Engine - FINAL authority over every proposed trade.

Independent of LLM reasoning. Checks, in order:
kill switches -> emergency drawdown -> daily loss -> loss streak ->
order frequency -> open positions -> stop/TP validity -> leverage ->
exposure (per-asset + correlated group, with measured correlation) ->
position sizing (with reviewer and drawdown multipliers).

Every rejection returns its reason; nothing is silently modified."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import get_risk_config
from core.interfaces import PortfolioSnapshot, ProposedTrade
from core.logging_setup import get_logger
from risk.drawdown import daily_loss_exceeded, drawdown_state, loss_streak_exceeded
from risk.exposure import check_exposure
from risk.kill_switch import KillSwitch
from risk.position_sizing import SizeResult, size_position

log = get_logger()


@dataclass(frozen=True)
class RiskVerdict:
    approved: bool
    reason: str = ""
    sized_notional: float = 0.0
    risk_amount: float = 0.0

    @property
    def rejection_reasons(self) -> str:
        return self.reason


class RiskEngine:
    def __init__(self, kill_switch: Optional[KillSwitch] = None):
        self.kill_switch = kill_switch or KillSwitch()

    def approve(self, trade: ProposedTrade, snapshot: PortfolioSnapshot,
                atr: float, measured_correlation: float | None = None,
                now=None) -> RiskVerdict:
        cfg = get_risk_config()["risk"]

        # 1. kill switches (global, then per-asset)
        if self.kill_switch.is_halted(None):
            return RiskVerdict(False, f"kill_switch:global:{self.kill_switch.reason()}")
        if self.kill_switch.is_halted(trade.asset_symbol):
            return RiskVerdict(False, f"kill_switch:{trade.asset_symbol}")

        # 2. drawdown ladder
        dd = drawdown_state(snapshot)
        if dd.level == "emergency_shutdown":
            return RiskVerdict(False, f"emergency_shutdown:drawdown>={dd.threshold_pct}%")
        if dd.level == "halt_new_trades":
            return RiskVerdict(False, f"halt_new_trades:drawdown>={dd.threshold_pct}%")

        # 3. daily loss
        exceeded, dlp = daily_loss_exceeded(snapshot)
        if exceeded:
            return RiskVerdict(False, f"daily_loss_limit:{dlp:.2f}%")

        # 4. consecutive losses
        streak_bad, streak = loss_streak_exceeded(snapshot)
        if streak_bad:
            return RiskVerdict(False, f"loss_streak:{streak}")

        # 5. order frequency
        today = snapshot.orders_today_per_asset.get(trade.asset_symbol, 0)
        if today >= cfg["max_orders_per_day_per_asset"]:
            return RiskVerdict(False, f"order_frequency:{today}")

        # 6. open positions
        if len(snapshot.open_positions) >= cfg["max_open_positions"]:
            return RiskVerdict(False, f"max_open_positions:{len(snapshot.open_positions)}")
        same_dir = [p for p in snapshot.open_positions
                    if p.asset_symbol == trade.asset_symbol and p.direction == trade.direction]
        if same_dir:
            return RiskVerdict(False, "duplicate_position_same_direction")

        # 7. stop / take-profit validity
        if trade.stop_loss <= 0 or trade.take_profit <= 0:
            return RiskVerdict(False, "invalid_stop_or_target")
        if trade.direction == "long":
            if not (trade.stop_loss < trade.proposed_entry < trade.take_profit):
                return RiskVerdict(False, "long_bracket_invalid")
        elif trade.direction == "short":
            if not (trade.take_profit < trade.proposed_entry < trade.stop_loss):
                return RiskVerdict(False, "short_bracket_invalid")
        else:
            return RiskVerdict(False, f"invalid_direction:{trade.direction}")
        stop_dist = abs(trade.proposed_entry - trade.stop_loss)
        tp_dist = abs(trade.take_profit - trade.proposed_entry)
        rr = tp_dist / stop_dist if stop_dist > 0 else 0.0
        if rr < cfg["min_reward_risk_ratio"]:
            return RiskVerdict(False, f"reward_risk_too_low:{rr:.2f}")
        if tp_dist < cfg["min_take_profit_atr"] * atr:
            return RiskVerdict(False, "take_profit_too_close")

        # 8. exposure caps (per asset + correlated group)
        size_first = size_position(snapshot.equity, trade.proposed_entry,
                                   trade.stop_loss, atr)
        if size_first.notional <= 0:
            return RiskVerdict(False, f"sizing_failed:{size_first.note}")
        exp = check_exposure(snapshot, trade.asset_symbol, size_first.notional,
                             measured_correlation)
        if not exp.ok:
            return RiskVerdict(False, f"{exp.reason}:{exp.group} "
                                     f"{exp.group_notional:.0f}>{exp.limit:.0f}")

        # 9. final sizing with drawdown + reviewer multipliers
        return self._final_size(trade, snapshot, atr, dd.risk_multiplier)

    def _final_size(self, trade: ProposedTrade, snapshot: PortfolioSnapshot,
                    atr: float, drawdown_mult: float,
                    reviewer_mult: float = 1.0) -> RiskVerdict:
        sized: SizeResult = size_position(
            snapshot.equity, trade.proposed_entry, trade.stop_loss, atr,
            reviewer_multiplier=reviewer_mult, drawdown_multiplier=drawdown_mult)
        if sized.notional <= 0:
            return RiskVerdict(False, f"sizing_failed:{sized.note}")
        if sized.notional > snapshot.available_balance:
            return RiskVerdict(False, "insufficient_balance")
        # leverage check: total notional after entry must stay under equity * max_leverage
        cfg = get_risk_config()["risk"]
        total_notional = sum(p.notional for p in snapshot.open_positions) + sized.notional
        if total_notional > snapshot.equity * cfg["max_leverage"]:
            return RiskVerdict(False, "leverage_limit")
        return RiskVerdict(True, sized_notional=sized.notional,
                           risk_amount=sized.risk_amount)

    def size_with_review(self, trade: ProposedTrade, snapshot: PortfolioSnapshot,
                         atr: float, reviewer_multiplier: float,
                         measured_correlation: float | None = None) -> RiskVerdict:
        """Same checks as approve but with the reviewer's risk multiplier."""
        verdict = self.approve(trade, snapshot, atr, measured_correlation)
        if not verdict.approved:
            return verdict
        dd = drawdown_state(snapshot)
        return self._final_size(trade, snapshot, atr, dd.risk_multiplier,
                                 reviewer_multiplier)
