"""Database session management and TimescaleDB initialisation."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings, get_settings
from core.exceptions import StoreError
from data_layer.storage.schemas import Base

_engine = None
_session_factory: sessionmaker | None = None


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,   # survives restarts of the DB container
            pool_size=5,
            max_overflow=10,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


def get_session() -> Session:
    return get_session_factory()()


def init_database(settings: Settings | None = None) -> None:
    """Create all tables and convert time-series tables to hypertables.

    Idempotent: safe to run at every startup (24/7 recovery path).
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            for table, time_col in [
                ("candles", "ts"),
                ("crypto_derivatives", "ts"),
                ("macro_observations", "release_ts"),
                ("news_articles", "published_ts"),
                ("news_events", "published_ts"),
                ("forecasts", "ts"),
            ]:
                conn.execute(
                    text(
                        "SELECT create_hypertable(:t, :c, if_not_exists => TRUE, "
                        "migrate_data => TRUE)"
                    ).bindparams(t=table, c=time_col)
                )
    except Exception as e:
        raise StoreError(f"Database initialisation failed: {e}") from e


def reset_engine() -> None:
    """Test helper: drop cached engine so a new database can be used."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
