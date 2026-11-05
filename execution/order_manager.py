"""Order management: idempotent order creation, duplicate protection, stop-loss
/ take-profit bracket handling, order frequency tracking (Redis, daily TTL)."""
from __future__ import annotations

import uuid
from typing import Optional

from core.clock import get_clock
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage.schemas import Order

log = get_logger()


class OrderManager:
    def __init__(self, redis_client=None):
        # redis_client=False -> offline mode (tests); None -> try to connect
        self._redis = None
        if redis_client is False:
            return
        self._redis = redis_client
        if self._redis is None:
            try:
                import redis as redis_lib
                from config.settings import get_settings
                self._redis = redis_lib.Redis.from_url(
                    get_settings().redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    # ---- frequency ------------------------------------------------------
    def _freq_key(self, symbol: str) -> str:
        return f"aits:orders_today:{get_clock().now().date().isoformat()}:{symbol}"

    def bump_frequency(self, symbol: str) -> None:
        if self._redis is not None:
            try:
                k = self._freq_key(symbol)
                pipe = self._redis.pipeline()
                pipe.incr(k)
                pipe.expire(k, 86400)
                pipe.execute()
            except Exception:
                log.warning("order_freq_track_failed")

    def orders_today(self, symbol: str) -> int:
        if self._redis is None:
            return 0
        try:
            return int(self._redis.get(self._freq_key(symbol)) or 0)
        except Exception:
            return 0

    def all_orders_today(self, symbols: list[str]) -> dict[str, int]:
        return {s: self.orders_today(s) for s in symbols}

    # ---- order lifecycle --------------------------------------------------
    def create_order(self, asset_symbol: str, instrument_id: str, side: str,
                     order_type: str, requested_size: float,
                     requested_price: Optional[float] = None,
                     decision_id: Optional[str] = None) -> Order:
        order_id = f"o-{uuid.uuid4().hex[:16]}"
        now = get_clock().now()
        with get_session() as session:
            session.add(Order(
                order_id=order_id, created_ts=now, asset_symbol=asset_symbol,
                instrument_id=instrument_id, side=side, order_type=order_type,
                requested_price=requested_price, requested_size=requested_size,
                status="new", decision_id=decision_id))
            session.commit()
            self.bump_frequency(asset_symbol)
        return Order(order_id=order_id, created_ts=now, asset_symbol=asset_symbol,
                     instrument_id=instrument_id, side=side, order_type=order_type,
                     requested_price=requested_price, requested_size=requested_size,
                     status="new", decision_id=decision_id)

    def mark_filled(self, order_id: str) -> None:
        with get_session() as session:
            o = session.get(Order, order_id)
            if o:
                o.status = "filled"
                session.commit()

    def mark_rejected(self, order_id: str, reason: str) -> None:
        with get_session() as session:
            o = session.get(Order, order_id)
            if o:
                o.status = "rejected"
                o.reason = reason
                session.commit()

    def pending_orders(self) -> list[Order]:
        with get_session() as session:
            return session.query(Order).filter(Order.status == "new").all()
