"""Signal combination. Interpretable, evidence-weighted votes:
forecast direction (from active models), technical momentum, regime bias.
No LLM ever assigns weights here - weights come from validation or are fixed
by config. A trade needs a clear majority and net expected return."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import get_trading_config


@dataclass(frozen=True)
class CombinedSignal:
    direction: str            # "long" | "short" | None
    agreement_score: float    # 0..1
    expected_return: float
    uncertainty: float
    supporting: list[str]
    conflicting: list[str]

    @property
    def tradable(self) -> bool:
        min_agree = get_trading_config()["decision"]["min_agreement_score"]
        min_ret = get_trading_config()["decision"]["min_net_expected_return"]
        return (self.direction is not None and self.agreement_score >= min_agree
                and abs(self.expected_return) >= min_ret)


def combine(forecast_return: float, forecast_uncertainty: float, p_up: float,
            momentum: float, trend_strength: float, regime: str,
            direction_suitability: float, net_expected_return: float) -> CombinedSignal:
    """Votes:
    - forecast: sign of net expected return, weight by |p_up - 0.5| confidence
    - technical: sign of momentum, weight by trend strength
    - regime: fixed suitability weight for the forecast direction
    """
    fc_dir = "long" if forecast_return > 0 else "short" if forecast_return < 0 else None
    if fc_dir is None:
        return CombinedSignal(None, 0.0, forecast_return, forecast_uncertainty,
                              [], ["forecast_flat"])

    tech_dir = "long" if momentum > 0 else "short" if momentum < 0 else None
    w_forecast = min(1.0, max(0.2, abs(p_up - 0.5) * 2.0 + 0.2))
    w_tech = min(1.0, max(0.1, abs(trend_strength)))
    votes_for = w_forecast  # forecast always votes its own direction
    votes_against = 0.0
    supporting, conflicting = [f"forecast:{fc_dir}:{forecast_return:+.4f}"], []

    if tech_dir == fc_dir:
        votes_for += w_tech
        supporting.append(f"momentum:{tech_dir}:{momentum:+.4f}")
    elif tech_dir is None:
        conflicting.append("momentum:flat")
    else:
        votes_against += w_tech
        conflicting.append(f"momentum:{tech_dir}:{momentum:+.4f}")

    votes_for += direction_suitability  # 0..1 for this direction in this regime
    total = w_forecast + w_tech + 1.0
    agreement = votes_for / total
    if direction_suitability < 0.5:
        conflicting.append(f"regime:{regime}:weak_for_{fc_dir}")

    return CombinedSignal(
        direction=fc_dir, agreement_score=round(agreement, 4),
        expected_return=net_expected_return, uncertainty=forecast_uncertainty,
        supporting=supporting, conflicting=conflicting)
