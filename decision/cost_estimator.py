"""Round-trip cost estimation. A trade is only worth taking if the expected
return survives spread + fees + slippage + funding."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_trading_config


@dataclass(frozen=True)
class CostEstimate:
    round_trip_bps: float
    funding_bps: float
    total_return_cost: float  # as fraction of notional (e.g. 0.0009)

def estimate_costs(asset_class: str, holding_hours: float) -> CostEstimate:
    cfg = get_trading_config()["costs"].get(
        "crypto" if asset_class == "crypto" else "metals",
        get_trading_config()["costs"]["metals"])
    bps = (cfg["taker_fee_bps"] + cfg["spread_bps"] + cfg["slippage_bps"]) * 2  # entry + exit
    funding = cfg.get("funding_bps_per_day", 0.0) * holding_hours / 24.0
    total = (bps + funding) / 10000.0
    return CostEstimate(round_trip_bps=bps, funding_bps=funding, total_return_cost=total)
