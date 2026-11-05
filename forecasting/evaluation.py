"""Walk-forward evaluation. Chronological splits only — no shuffling.
A gap between train and test windows prevents leakage from the target
horizon (a model trained on data whose forward return reaches into the
test window would cheat)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forecasting.base_model import ForecastModel


@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    train_bars: int = 2000
    test_bars: int = 300
    gap_bars: int = 3


@dataclass
class EvalResult:
    model: str
    asset_symbol: str
    horizon: str
    n_predictions: int
    directional_accuracy: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    mean_abs_forecast: float
    hit_rate_80_interval: Optional[float]  # calibration: actual in [lo, hi] 80%?
    diebold_mariano_p: Optional[float] = None  # vs random walk

    def to_dict(self) -> dict:
        return self.__dict__.copy() | {
            "directional_accuracy": None if self.directional_accuracy is None else round(self.directional_accuracy, 4),
        }


def walk_forward_evaluate(
    model: ForecastModel,
    X: pd.DataFrame,
    y: pd.Series,
    asset_symbol: str,
    horizon: str,
    config: WalkForwardConfig = WalkForwardConfig(),
    baseline: Optional[ForecastModel] = None,
) -> EvalResult:
    """Fits on a rolling window, predicts one bar at a time on the test
    window (refitting once per split — chronological, leakage-gapped)."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    n = len(X)
    if n < config.train_bars + config.test_bars:
        return EvalResult(model=model.name, asset_symbol=asset_symbol, horizon=horizon,
                          n_predictions=0, directional_accuracy=None, mae=None,
                          rmse=None, mean_abs_forecast=0.0, hit_rate_80_interval=None)

    preds: list[float] = []
    actuals: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    baseline_preds: list[float] = []

    start = 0
    while start + config.train_bars + config.gap_bars + 1 < n:
        train_end = start + config.train_bars
        test_start = train_end + config.gap_bars
        test_end = min(test_start + config.test_bars, n)
        if test_end - test_start < 10:
            break
        X_train, y_train = X.iloc[start:train_end], y.iloc[start:train_end]
        try:
            model.fit(X_train, y_train)
        except Exception:
            start = test_end
            continue
        if baseline is not None:
            try:
                baseline.fit(X_train, y_train)
            except Exception:
                baseline = None
        for i in range(test_start, test_end):
            row = X.iloc[i]
            actual = y.iloc[i]
            if pd.isna(actual):
                continue
            forecast = model.predict(asset_symbol, horizon, row)
            if forecast is None or forecast.expected_return is None:
                continue
            preds.append(forecast.expected_return)
            actuals.append(float(actual))
            lows.append(forecast.interval_lower if forecast.interval_lower is not None else -np.inf)
            highs.append(forecast.interval_upper if forecast.interval_upper is not None else np.inf)
            if baseline is not None:
                bp = baseline.predict_expected_return(row)
                baseline_preds.append(float(bp) if bp is not None else 0.0)
        start = test_end

    if not preds:
        return EvalResult(model=model.name, asset_symbol=asset_symbol, horizon=horizon,
                          n_predictions=0, directional_accuracy=None, mae=None,
                          rmse=None, mean_abs_forecast=0.0, hit_rate_80_interval=None)

    p = np.array(preds)
    a = np.array(actuals)
    lo, hi = np.array(lows), np.array(highs)
    finite = np.isfinite(lo) & np.isfinite(hi)

    nonzero = np.abs(p) > 1e-12
    dir_acc = float(np.mean(np.sign(p[nonzero]) == np.sign(a[nonzero]))) if nonzero.any() else None
    mae = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    coverage = float(np.mean((a >= lo) & (a <= hi))) if finite.any() else None

    dm_p = None
    if baseline_preds and len(baseline_preds) == len(preds):
        dm_p = _diebold_mariano(np.array(preds), np.array(baseline_preds), a)

    return EvalResult(
        model=model.name, asset_symbol=asset_symbol, horizon=horizon,
        n_predictions=len(preds), directional_accuracy=dir_acc, mae=mae, rmse=rmse,
        mean_abs_forecast=float(np.mean(np.abs(p))),
        hit_rate_80_interval=coverage, diebold_mariano_p=dm_p,
    )


def _diebold_mariano(forecast: np.ndarray, baseline: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """DM test on squared-error loss vs baseline. Two-sided p-value."""
    try:
        from scipy.stats import t as student_t
    except ImportError:
        return None
    d = (actual - forecast) ** 2 - (actual - baseline) ** 2
    if len(d) < 10 or np.allclose(d, 0):
        return None
    dm = d.mean() / (np.std(d, ddof=1) / np.sqrt(len(d)))
    return float(2 * (1 - student_t.cdf(abs(dm), df=len(d) - 1)))
