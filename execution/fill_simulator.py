"""Realistic fill simulation: spread, volatility-scaled slippage, delay,
partial fills for large orders, fees. Never a real order."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import get_trading_config
from core.clock import get_clock
from core.exceptions import ExecutionError


@dataclass(frozen=True)
class SimulatedFill:
    price: float
    size: float
    fee: float
    slippage: float
    ts: datetime


def simulate_fill(asset_class: str, side: str, size: float, ref_price: float,
                  requested_price: Optional[float] = None,
                  realized_vol_annualized: float = 0.3,
                  liquidity_limit_usd: float = 250_000.0,
                  fill_delay_seconds: Optional[tuple[int, int]] = None,
                  rng: Optional[random.Random] = None) -> SimulatedFill:
    """Market order fills at ref +/- half spread + slippage; limit orders fill
    only if the market reaches the limit price."""
    rng = rng or random
    cfg = get_trading_config()["costs"].get(
        "crypto" if asset_class == "crypto" else "metals")

    spread = ref_price * cfg["spread_bps"] / 10000.0
    base_slip = ref_price * cfg["slippage_bps"] / 10000.0
    # slippage scales with sqrt(order size / liquidity limit) and volatility
    size_frac = min(1.0, (size * ref_price) / liquidity_limit_usd)
    vol_scale = max(0.5, min(3.0, realized_vol_annualized / 0.3))
    slippage = base_slip * (1.0 + size_frac * 4.0) * vol_scale * (0.5 + rng.random())

    if requested_price is not None:  # limit order
        reachable = (requested_price <= ref_price + spread / 2 if side == "buy"
                     else requested_price >= ref_price - spread / 2)
        if not reachable:
            raise ExecutionError(f"limit_price_unreachable:{requested_price}:{ref_price}")
        fill_price = requested_price
    else:  # market order: cross the spread plus slippage against you
        fill_price = ref_price + (spread / 2 + slippage) * (1 if side == "buy" else -1)

    fill_size = size
    if size_frac >= 1.0:  # partial fill: order larger than available liquidity
        fill_size = size * rng.uniform(0.6, 0.9)

    fee = fill_size * fill_price * cfg["taker_fee_bps"] / 10000.0
    return SimulatedFill(price=round(fill_price, 6), size=round(fill_size, 8),
                         fee=round(fee, 6), slippage=round(slippage, 6),
                         ts=get_clock().now())
