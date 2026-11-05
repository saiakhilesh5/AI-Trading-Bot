"""Forecasting workflow: (re)benchmark assets, promote validated models to
active, demote models that no longer beat the random-walk baseline.

Only models adopted by benchmark_and_register (beat random walk on the DM/MAE
comparison with >= 100 out-of-sample predictions) are ever promoted. A model
that fails re-validation is demoted to shadow, never silently kept."""
from __future__ import annotations

from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from forecasting import benchmark_and_register
from forecasting import model_registry as registry

log = get_logger()

_ASSET_HORIZONS = [
    ("XAUUSD", "1d", "1d"), ("XAGUSD", "1d", "1d"),
    ("BTCUSD", "1d", "1d"), ("ETHUSD", "1d", "1d"),
    ("BTCUSD", "1h", "1h"), ("ETHUSD", "1h", "1h"),
    ("XAUUSD", "1h", "1h"), ("XAGUSD", "1h", "1h"),
]


def refresh_models(asset_horizons: list[tuple[str, str, str]] | None = None) -> dict:
    """Re-benchmark and promote/demote. Returns a summary for the daily log."""
    adopted_all: dict[str, list[str]] = {}
    for symbol, interval, horizon in (asset_horizons or _ASSET_HORIZONS):
        try:
            result = benchmark_and_register(symbol, interval, horizon)
            adopted_all[f"{symbol}:{horizon}"] = result["adopted_models"]
        except Exception as e:
            log.warning("benchmark_refresh_failed", asset=symbol, error=str(e))

    session = get_session()
    try:
        promoted, demoted = [], []
        for key, adopted in adopted_all.items():
            symbol, horizon = key.split(":")[0], key.split(":")[-1]
            for model_id in adopted:
                registry.set_status(session, model_id, "active")
                promoted.append(model_id)
            for reg in registry.active_models(session, symbol):
                if reg.model_id not in adopted and reg.horizon == horizon:
                    registry.set_status(session, reg.model_id, "shadow")
                    demoted.append(reg.model_id)
        session.commit()
        log.info("model_refresh_done", promoted=promoted, demoted=demoted)
        return {"promoted": promoted, "demoted": demoted}
    finally:
        session.close()


if __name__ == "__main__":
    refresh_models()
