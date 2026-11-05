"""Persistent process state for crash recovery: what ran, when, and where it
stopped. The loop re-reads this on startup and skips already-completed work for
the current cycle instead of duplicating it."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from core.clock import get_clock
from core.logging_setup import get_logger

log = get_logger()
_KEY = "aits:process_state"


class ProcessState:
    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._local: dict = {}
        if self._redis is None:
            try:
                import redis as redis_lib
                from config.settings import get_settings
                self._redis = redis_lib.Redis.from_url(
                    get_settings().redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                log.warning("process_state_redis_unavailable")
                self._redis = None

    def _read(self) -> dict:
        if self._redis is not None:
            try:
                raw = self._redis.get(_KEY)
                return json.loads(raw) if raw else {}
            except Exception:
                log.warning("process_state_read_failed")
        return self._local

    def _write(self, state: dict) -> None:
        self._local = state
        if self._redis is not None:
            try:
                self._redis.set(_KEY, json.dumps(state))
            except Exception:
                log.warning("process_state_write_failed")

    def mark_completed(self, task: str, ts: Optional[datetime] = None) -> None:
        state = self._read()
        state[task] = (ts or get_clock().now()).isoformat()
        self._write(state)

    def completed_at(self, task: str) -> Optional[datetime]:
        raw = self._read().get(task)
        return datetime.fromisoformat(raw) if raw else None

    def cycle_id(self) -> str:
        """One id per completed loop pass; new pass = new id (dedup window)."""
        return f"cycle-{get_clock().now().strftime('%Y%m%d%H%M')}"
