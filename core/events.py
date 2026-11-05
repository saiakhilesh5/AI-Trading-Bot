"""Event bus: duplicate-event prevention and bounded retry delivery.

Used by the supervisor to react to closed candles, provider failures,
and risk events. Redis-backed seen-keys survive restarts (24/7 operation).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Optional

from core.clock import get_clock
from core.exceptions import StoreError


class EventKind(str, Enum):
    CANDLE_CLOSED = "candle_closed"      # a market-data interval finished
    NEWS_ARRIVED = "news_arrived"
    MACRO_RELEASED = "macro_released"
    RISK_BREACH = "risk_breach"
    DATA_STALE = "data_stale"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    # idempotency key: two events with the same key are the same event
    dedup_key: str = ""

    def key(self) -> str:
        if self.dedup_key:
            return f"{self.kind.value}:{self.dedup_key}"
        blob = json.dumps(self.payload, sort_keys=True, default=str)
        return f"{self.kind.value}:{hashlib.sha256(blob.encode()).hexdigest()[:24]}"


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, redis_client: Any = None, dedup_ttl: timedelta = timedelta(hours=24)):
        self._handlers: dict[EventKind, list[Handler]] = {}
        self._redis = redis_client
        self._dedup_ttl = dedup_ttl
        self._local_seen: set[str] = set()
        self._dead_letters: list[tuple[Event, Exception]] = []

    def subscribe(self, kind: EventKind, handler: Handler) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def publish(self, event: Event) -> bool:
        """Returns False when the event was a duplicate (not delivered)."""
        key = event.key()
        if not self._claim(key):
            return False
        for handler in self._handlers.get(event.kind, []):
            try:
                handler(event)
            except Exception as e:  # dead-letter, never crash the bus
                self._dead_letters.append((event, e))
        return True

    def _claim(self, key: str) -> bool:
        if key in self._local_seen:
            return False
        if self._redis is not None:
            try:
                claimed = self._redis.set(f"event:{key}", 1, nx=True, ex=self._dedup_ttl)
                if not claimed:
                    return False
            except Exception as e:
                raise StoreError(f"Redis dedup check failed: {e}") from e
        self._local_seen.add(key)
        return True

    def dead_letters(self) -> list[tuple[Event, Exception]]:
        return list(self._dead_letters)
