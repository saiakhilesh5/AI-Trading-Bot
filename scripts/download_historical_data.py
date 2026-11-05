"""Download historical data for all configured assets and intervals."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from config.settings import get_asset_config, get_settings
from core.interfaces import Asset, AssetClass, Interval
from core.logging_setup import get_logger, setup_logging
from data_layer.collectors.orchestrator import IngestionOrchestrator
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo
from scripts.initialize_database import seed_assets_from_yaml

log = get_logger()


def load_assets() -> list[Asset]:
    session = get_session()
    try:
        return [Asset(symbol=a.symbol, name=a.name, asset_class=AssetClass(a.asset_class),
                      price_source=a.price_source, trades_24h=a.trades_24h, notes=a.notes)
                for a in repo.load_assets(session)]
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download historical candles")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--asset", default=None)
    args = parser.parse_args()

    setup_logging(get_settings())
    seed_assets_from_yaml()
    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    orch = IngestionOrchestrator()
    failures = 0
    assets = load_assets()
    if args.asset:
        assets = [a for a in assets if a.symbol == args.asset]
    # window per interval sized for walk-forward needs (2000 train bars)
    window_days = {Interval.H1: 140, Interval.H4: 560, Interval.D1: 1095}
    for asset in assets:
        for interval in [Interval(i) for i in get_asset_config()["intervals"]]:
            days = window_days.get(interval, args.days)
            iv_start = datetime.now(timezone.utc) - timedelta(days=days)
            try:
                report = orch.ingest_candles(asset, interval, start=iv_start, limit=20000)
                print(f"{asset.symbol} {interval.value}: last={report.last_candle_ts} "
                      f"eligible={report.trade_eligible} flags={[f.flag.value for f in report.findings]}")
            except Exception as e:
                failures += 1
                print(f"{asset.symbol} {interval.value}: FAILED - {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
