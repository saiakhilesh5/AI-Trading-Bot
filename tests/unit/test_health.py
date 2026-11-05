"""Data-health validation tests."""
from datetime import datetime, timezone

import pandas as pd

from data_layer.validators.data_health import DataHealthChecker, candles_to_frame

from tests.conftest import make_candles


class TestDataHealth:
    def test_healthy_24h_data_passes(self, sim_clock):
        candles = make_candles(n=100)
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(candles), trades_24h=True)
        assert report.trade_eligible

    def test_empty_data_blocks_trading(self, sim_clock):
        report = DataHealthChecker().check("BTCUSD", "1h", pd.DataFrame(), trades_24h=True)
        assert not report.trade_eligible and report.has_critical

    def test_stale_data_blocks_trading(self, sim_clock):
        candles = make_candles(n=50, start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(candles), trades_24h=True)
        assert not report.trade_eligible
        assert "stale_data" in [f.flag.value for f in report.findings]

    def test_large_gap_blocks_trading(self, sim_clock):
        a = make_candles(n=50)
        b = make_candles(n=50, start=a[-1].ts + pd.Timedelta(hours=10).to_pytimedelta(),
                         price=a[-1].close)
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(a + b), trades_24h=True)
        assert not report.trade_eligible

    def test_small_gap_only_warns(self, sim_clock):
        a = make_candles(n=50)
        b = make_candles(n=50, start=a[-1].ts + pd.Timedelta(hours=2).to_pytimedelta(),
                         price=a[-1].close)
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(a + b), trades_24h=True)
        assert report.trade_eligible

    def test_duplicates_flagged(self, sim_clock):
        candles = make_candles(n=100)
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(candles + [candles[-1]]), trades_24h=True)
        assert "duplicate_records" in [f.flag.value for f in report.findings]

    def test_extreme_price_jump_flagged(self, sim_clock):
        candles = make_candles(n=150)
        last = candles[-1]
        jump = last.model_copy(update={
            "ts": last.ts + pd.Timedelta(hours=1).to_pytimedelta(),
            "open": last.close, "close": last.close * 1.30,
            "high": last.close * 1.31, "low": last.close * 0.99,
        })
        report = DataHealthChecker().check("BTCUSD", "1h", candles_to_frame(candles + [jump]), trades_24h=True)
        assert "unexpected_price_jump" in [f.flag.value for f in report.findings]

    def test_weekend_gap_not_a_failure_for_metals(self, sim_clock):
        """Non-24h market: weekend closures must not block trading."""
        candles = make_candles(n=240, with_weekends=False)  # ~10 days of weekdays
        report = DataHealthChecker().check("XAUUSD", "1h", candles_to_frame(candles), trades_24h=False)
        assert report.trade_eligible

    def test_session_gap_blocks_trading(self, sim_clock):
        """A genuine in-session hole (>3 consecutive missing) blocks trading."""
        candles = make_candles(n=240, with_weekends=False)
        # carve out a 6-hour hole in the last weekday
        hole_start = candles[-40].ts
        holed = [c for c in candles
                 if not (hole_start <= c.ts < hole_start + pd.Timedelta(hours=6).to_pytimedelta())]
        report = DataHealthChecker().check("XAUUSD", "1h", candles_to_frame(holed), trades_24h=False)
        assert not report.trade_eligible

    def test_conflicting_feed_blocks_trading(self, sim_clock):
        candles = make_candles(n=100)
        report = DataHealthChecker().check(
            "BTCUSD", "1h", candles_to_frame(candles), trades_24h=True,
            cross_check_prices={"independent_source": candles[-1].close * 1.05},
        )
        assert not report.trade_eligible
        assert "conflicting_feeds" in [f.flag.value for f in report.findings]
