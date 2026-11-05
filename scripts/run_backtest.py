"""Benchmark all forecasting models for all assets and horizons.
Usage: python -m scripts.run_backtest --asset BTCUSD --interval 1h --horizon 1h"""
from __future__ import annotations

import argparse
import json

from config.settings import get_asset_config, get_settings
from core.logging_setup import setup_logging
from forecasting import benchmark_and_register


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="Asset symbol (default: all)")
    parser.add_argument("--interval", default="1h", choices=["1h", "4h", "1d"])
    parser.add_argument("--horizon", default=None, help="Default: same as interval")
    args = parser.parse_args()

    setup_logging(get_settings())
    horizon = args.horizon or args.interval
    assets = [args.asset] if args.asset else list(get_asset_config()["assets"])
    summary = {}
    for asset in assets:
        result = benchmark_and_register(asset, args.interval, horizon)
        summary[asset] = result
        print(f"\n=== {asset} {args.interval} -> {horizon} ===")
        for r in sorted(result["results"], key=lambda x: x.get("mae") or 9e9):
            print(f"  {r['model']:<28} n={r['n_predictions']:>4} "
                  f"dir_acc={r['directional_accuracy']} mae={r['mae']} "
                  f"cover80={r['hit_rate_80_interval']}")
    print("\nAdopted:", {a: v["adopted_models"] for a, v in summary.items() if v["adopted_models"]})
    with open("reports/benchmark_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
