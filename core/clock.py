"""UTC clock with injectable simulated time (backtests, tests).

All system timestamps are timezone-aware UTC.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimulatedClock(Clock):
    def __init__(self, start: datetime):
        self._current = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._current

    def advance_to(self, moment: datetime) -> None:
        new = moment.astimezone(timezone.utc)
        if new < self._current:
            raise ValueError(f"Simulated clock cannot move backwards: {new} < {self._current}")
        self._current = new


_active_clock: Optional[Clock] = None


def get_clock() -> Clock:
    global _active_clock
    return _active_clock or Clock()


def set_clock(clock: Optional[Clock]) -> None:
    global _active_clock
    _active_clock = clock


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)
