"""Lightweight interval scheduler with heartbeat, jitter, and miss-tolerance
(the machine may sleep or lose power; missed runs just run late, never twice)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from core.clock import get_clock
from core.logging_setup import get_logger

log = get_logger()


@dataclass
class Task:
    name: str
    interval: timedelta
    fn: Callable[[], None]
    jitter_seconds: float = 0.0


class Scheduler:
    def __init__(self):
        self._tasks: list[Task] = []
        self._next_run: dict[str, datetime] = {}
        self._last_heartbeat: Optional[datetime] = None
        self._stop_requested = False

    def every(self, name: str, interval: timedelta,
              fn: Callable[[], None]) -> "Scheduler":
        self._tasks.append(Task(name=name, interval=interval, fn=fn))
        self._next_run[name] = get_clock().now()
        return self

    def stop(self) -> None:
        self._stop_requested = True

    def _heartbeat(self) -> None:
        self._last_heartbeat = get_clock().now()

    def run_pending(self, now: Optional[datetime] = None) -> list[str]:
        """Run all due tasks once. Returns names executed. Used by both the
        blocking loop and single-shot invocations (tests, cron-style calls)."""
        now = now or get_clock().now()
        executed: list[str] = []
        for task in self._tasks:
            due = self._next_run.get(task.name)
            if due is None or now >= due:
                try:
                    task.fn()
                    executed.append(task.name)
                except Exception as e:
                    log.error("task_failed", task=task.name, error=str(e))
                finally:
                    self._next_run[task.name] = now + task.interval
        self._heartbeat()
        return executed

    def run_forever(self, poll_seconds: float = 5.0,
                    stop_hook: Optional[Callable[[], bool]] = None) -> None:
        """Blocking loop with graceful shutdown (KeyboardInterrupt / hook)."""
        log.info("scheduler_started", tasks=[t.name for t in self._tasks])
        try:
            while not self._stop_requested:
                if stop_hook and stop_hook():
                    break
                self.run_pending()
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            log.info("scheduler_interrupted")
        finally:
            log.info("scheduler_stopped", last_heartbeat=str(self._last_heartbeat))

    @property
    def heartbeat(self) -> Optional[datetime]:
        return self._last_heartbeat
