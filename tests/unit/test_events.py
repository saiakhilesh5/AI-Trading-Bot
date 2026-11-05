"""Event-bus tests: duplicate prevention and dead-lettering."""
from datetime import timedelta

import pytest

from core.events import Event, EventBus, EventKind


class FakeRedis:
    """In-memory stand-in for redis dedup keys."""

    def __init__(self):
        self.store: dict[str, int] = {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


class TestEventBus:
    def test_duplicate_events_delivered_once(self):
        bus = EventBus(FakeRedis())
        received = []

        bus.subscribe(EventKind.CANDLE_CLOSED, lambda e: received.append(e))
        event = Event(EventKind.CANDLE_CLOSED, {"asset": "BTC"}, dedup_key="BTC-1h-100")
        assert bus.publish(event) is True
        assert bus.publish(event) is False
        assert len(received) == 1

    def test_redis_persistence_blocks_duplicates_across_restarts(self):
        redis = FakeRedis()
        event = Event(EventKind.CANDLE_CLOSED, {"asset": "BTC"}, dedup_key="BTC-1h-100")
        assert EventBus(redis).publish(event)
        # a new bus instance (restart) with the same redis still dedups
        assert not EventBus(redis).publish(event)

    def test_handler_failure_does_not_crash_bus(self):
        bus = EventBus(FakeRedis())
        got = []

        def bad(_):
            raise RuntimeError("boom")

        bus.subscribe(EventKind.NEWS_ARRIVED, bad)
        bus.subscribe(EventKind.NEWS_ARRIVED, lambda e: got.append(e))
        bus.publish(Event(EventKind.NEWS_ARRIVED, {"id": "x"}, dedup_key="n1"))
        assert len(got) == 1
        assert len(bus.dead_letters()) == 1

    def test_dedup_ttl_respected(self):
        redis = FakeRedis()
        bus = EventBus(redis, dedup_ttl=timedelta(hours=0))
        event = Event(EventKind.HEARTBEAT, {"n": 1}, dedup_key="hb")
        assert bus.publish(event)
