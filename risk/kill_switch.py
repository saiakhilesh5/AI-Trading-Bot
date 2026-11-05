"""Kill switches: global and per-asset. Redis-backed so state survives
process restarts; falls back to in-memory flags if Redis is unavailable
(never to 'off' - fails closed for new trades? No: a Redis outage must not
freeze management of existing positions, so fallback is in-memory + log)."""
from __future__ import annotations

import json
from typing import Optional

from core.logging_setup import get_logger

log = get_logger()

_KEY = "aits:kill_switches"


class KillSwitch:
    def __init__(self, redis_client=None):
        # redis_client=False -> offline mode (tests); None -> try to connect
        if redis_client is False:
            self._redis = None
            self._local = {}
            return
        self._redis = redis_client
        self._local = {}
        if self._redis is None:
            try:
                import redis as redis_lib
                from config.settings import get_settings
                self._redis = redis_lib.Redis.from_url(
                    get_settings().redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                log.warning("kill_switch_redis_unavailable")
                self._redis = None

    def _read(self) -> dict:
        if self._redis is not None:
            try:
                raw = self._redis.get(_KEY)
                return json.loads(raw) if raw else {}
            except Exception:
                log.warning("kill_switch_read_failed")
        return self._local

    def _write(self, state: dict) -> None:
        self._local = state
        if self._redis is not None:
            try:
                self._redis.set(_KEY, json.dumps(state))
            except Exception:
                log.warning("kill_switch_write_failed")

    def trigger(self, reason: str, asset: Optional[str] = None) -> None:
        state = self._read()
        key = asset or "__global__"
        state[key] = True
        state[f"{key}:reason"] = reason
        self._write(state)
        log.error("kill_switch_triggered", scope=key, reason=reason)

    def clear(self, asset: Optional[str] = None) -> None:
        state = self._read()
        key = asset or "__global__"
        state.pop(key, None)
        state.pop(f"{key}:reason", None)
        self._write(state)

    def is_halted(self, asset: Optional[str] = None) -> bool:
        state = self._read()
        if state.get("__global__"):
            return True
        if asset is not None and state.get(asset):
            return True
        return False

    def reason(self, asset: Optional[str] = None) -> Optional[str]:
        state = self._read()
        key = asset or "__global__"
        if state.get("__global__"):
            return state.get("__global__:reason")
        return state.get(f"{asset}:reason") if asset else state.get("__global__:reason")

    def status(self) -> dict:
        state = self._read()
        return {k: v for k, v in state.items() if not k.endswith(":reason")}
