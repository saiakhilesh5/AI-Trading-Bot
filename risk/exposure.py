"""Correlation-aware exposure limits. Gold+Silver and BTC+ETH are related
exposures; measured rolling correlation can relax a group limit (never
tighten beyond the configured cap)."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_risk_config
from core.interfaces import PortfolioSnapshot


@dataclass(frozen=True)
class ExposureVerdict:
    ok: bool
    reason: str = ""
    group: str = ""
    group_notional: float = 0.0
    limit: float = 0.0


def group_for(symbol: str) -> str | None:
    groups = get_risk_config()["correlation_groups"]
    for name, symbols in groups.items():
        if isinstance(symbols, list) and symbol in symbols:
            return name
    return None


def group_symbols(symbol: str) -> list[str]:
    g = group_for(symbol)
    if g is None:
        return [symbol]
    return list(get_risk_config()["correlation_groups"][g])


def check_exposure(snapshot: PortfolioSnapshot, symbol: str, added_notional: float,
                  measured_correlation: float | None = None) -> ExposureVerdict:
    cfg = get_risk_config()["risk"]
    equity = max(snapshot.equity, 1e-9)

    # per-asset cap
    same = [p.notional for p in snapshot.open_positions if p.asset_symbol == symbol]
    if sum(same) + added_notional > equity * cfg["max_asset_exposure_pct"] / 100.0:
        return ExposureVerdict(False, "asset_exposure_cap", symbol,
                               sum(same), equity * cfg["max_asset_exposure_pct"] / 100.0)

    g = group_for(symbol)
    if g is None:
        return ExposureVerdict(True)

    groups = get_risk_config()["correlation_groups"]
    limit_pct = cfg["max_group_exposure_pct"]
    # correlation relief: low measured correlation relaxes the shared cap
    relief = groups.get("correlation_relief_threshold", 0.4)
    if measured_correlation is not None and measured_correlation < relief:
        # treat independently: cap at per-asset limit for this asset only
        return ExposureVerdict(True, group=g)

    gsyms = groups[g]
    gnotional = snapshot.group_notional(gsyms) + added_notional
    limit = equity * limit_pct / 100.0
    if gnotional > limit:
        return ExposureVerdict(False, "group_exposure_cap", g, gnotional, limit)
    return ExposureVerdict(True, group=g, group_notional=gnotional, limit=limit)
