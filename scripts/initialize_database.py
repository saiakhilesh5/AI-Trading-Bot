"""Database initialisation + asset seeding from config/assets.yaml."""
from __future__ import annotations

import json

from config.settings import get_asset_config, get_settings
from core.logging_setup import get_logger, setup_logging
from data_layer.storage.data_access import get_session, init_database
from data_layer.storage.schemas import AssetConfig

log = get_logger()


def seed_assets_from_yaml() -> None:
    cfg = get_asset_config()
    session = get_session()
    try:
        existing = {a.symbol for a in session.query(AssetConfig).all()}
        for symbol, spec in cfg["assets"].items():
            if symbol in existing:
                continue
            session.add(AssetConfig(
                symbol=symbol, name=spec["name"], asset_class=spec["class"],
                price_source=spec["primary_source"], trades_24h=spec.get("trades_24h", True),
                notes=spec.get("notes", "")))
        for ref in cfg.get("reference_series", []):
            if ref["symbol"] in existing:
                continue
            session.add(AssetConfig(
                symbol=ref["symbol"], name=ref["name"], asset_class="macro",
                price_source=ref["source"], trades_24h=False,
                notes=ref.get("notes", "")))
        session.commit()
        log.info("assets_seeded", count=session.query(AssetConfig).count())
    finally:
        session.close()


def main() -> int:
    setup_logging(get_settings())
    init_database(get_settings())
    seed_assets_from_yaml()
    log.info("database_ready", url=get_settings().postgres_host)
    print("Database initialised and assets seeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
