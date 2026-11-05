"""Loguru logging. Secrets must never be logged.

JSON sink for machine consumption + console sink for local operation.
"""
from __future__ import annotations

import sys

from loguru import logger

from config.settings import Settings

_configured = False


def setup_logging(settings: Settings | None = None) -> None:
    global _configured
    if _configured:
        return
    from config.settings import get_settings

    settings = settings or get_settings()
    level = "INFO"

    logger.remove()
    logger.add(sys.stdout, level=level, serialize=False)
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        level=level,
        serialize=True,
        rotation="1 day",
        retention="30 days",
        enqueue=True,
    )
    _configured = True


def get_logger():  # thin re-export so subsystems import one path
    return logger
