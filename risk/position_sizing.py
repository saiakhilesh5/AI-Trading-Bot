"""Volatility-targeted position sizing. Deterministic; reviewer multiplier may
only shrink risk, never grow it."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_risk_config


@dataclass(frozen=True)
class SizeResult:
    notional: float
    risk_amount: float
    note: str = ""


def size_position(equity: float, entry: float, stop_loss: float,
                  atr: float, reviewer_multiplier: float = 1.0,
                  drawdown_multiplier: float = 1.0) -> SizeResult:
    cfg = get_risk_config()["risk"]
    sizing_cfg = get_risk_config()["position_sizing"]
    mult = min(1.0, reviewer_multiplier) * min(1.0, drawdown_multiplier)

    risk_pct = cfg["max_risk_per_trade_pct"] / 100.0 * mult
    risk_amount = equity * risk_pct
    stop_distance = abs(entry - stop_loss)

    min_stop = cfg["min_stop_distance_atr"] * atr
    if stop_distance < min_stop * 0.99:
        return SizeResult(0.0, 0.0, f"stop_too_tight:{stop_distance:.4f}<{min_stop:.4f}")

    # fixed-fractional sizing on stop distance (risk-defined)
    units = risk_amount / stop_distance
    notional = units * entry

    # volatility-targeted cap: notional such that position vol ~= target
    vol_target = sizing_cfg.get("volatility_target_pct", 10.0) / 100.0
    # ATR as fraction of price ~ daily vol; annualize ~sqrt(252)
    daily_vol_frac = atr / entry if entry > 0 else 1.0
    ann_vol_frac = daily_vol_frac * (252 ** 0.5)
    if ann_vol_frac > 0:
        vol_notional = equity * vol_target / ann_vol_frac
        notional = min(notional, vol_notional)

    max_notional = equity * cfg["max_position_size_pct_equity"] / 100.0
    if notional > max_notional:
        notional = max_notional

    if notional < sizing_cfg.get("min_notional_usd", 0.0):
        return SizeResult(0.0, risk_amount, "notional_below_minimum")

    return SizeResult(round(notional, 2), risk_amount)
