"""Central configuration. Secrets come only from environment/.env (never source).
Structural config lives in YAML files (assets, models, risk, trading)."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parents[1]


class OperatingMode(str, Enum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    MONITOR = "monitor"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Application
    trading_mode: OperatingMode = OperatingMode.PAPER

    # Infrastructure
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "aits"
    postgres_user: str = "aits"
    postgres_password: str = "aits_local_dev"
    redis_url: str = "redis://localhost:6380/0"

    # Data providers
    fred_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    goldprice_api_key: Optional[str] = None

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_monthly_budget_usd: float = 10.0

    # Paper trading
    paper_starting_capital_usd: float = 100000.0

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def load_yaml(name: str) -> dict:
    path = PROJECT_ROOT / "config" / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_asset_config() -> dict:
    return load_yaml("assets")


def get_model_config() -> dict:
    return load_yaml("model_config")


def get_risk_config() -> dict:
    return load_yaml("risk_config")


def get_trading_config() -> dict:
    return load_yaml("trading_config")
