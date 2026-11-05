"""Calibration monitoring: interval coverage and directional reliability."""
from __future__ import annotations

import numpy as np


def interval_coverage(lowers: np.ndarray, uppers: np.ndarray, actuals: np.ndarray,
                      nominal: float = 0.8) -> float | None:
    """Empirical coverage of prediction intervals vs nominal level.
    Calibration requires ~coverage == nominal on adequate samples."""
    if len(actuals) < 30:
        return None
    inside = (actuals >= lowers) & (actuals <= uppers)
    return float(np.mean(inside))


def calibration_error(coverage: float, nominal: float = 0.8) -> float:
    return abs(coverage - nominal)


def brier_score(prob_up: np.ndarray, actual_up: np.ndarray) -> float | None:
    """Brier score for directional probability calibration."""
    if len(prob_up) < 30:
        return None
    return float(np.mean((prob_up - actual_up) ** 2))
