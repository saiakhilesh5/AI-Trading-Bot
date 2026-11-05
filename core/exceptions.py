"""Central error hierarchy. Every subsystem failure maps to a typed error."""
from __future__ import annotations
from typing import Optional


class AppError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str, *, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}


class ConfigurationError(AppError):
    pass


class DataProviderError(AppError):
    """External data source failure (network, rate limit, malformed response)."""

    def __init__(self, message: str, *, source: str = "", context: Optional[dict] = None):
        merged = {"source": source, **(context or {})}
        super().__init__(message, context=merged)
        self.source = source


class DataQualityError(AppError):
    """Validation failed and the record must not enter the store."""


class StoreError(AppError):
    pass


class ContractError(AppError):
    """Malformed agent or subsystem output rejected by schema validation."""


class LLMError(AppError):
    """LLM provider failure or schema-invalid response."""


class ExecutionError(AppError):
    pass


class RiskViolation(AppError):
    """Deterministic risk control rejected the proposed trade."""
