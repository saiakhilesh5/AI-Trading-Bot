"""GARCH(1,1) and HAR-RV volatility forecasters (arch package)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.base_model import ForecastModel


class GARCHModel(ForecastModel):
    """GARCH(1,1) on log returns. Volatility + quantile intervals.
    Return expectation = historical mean of residuals (small)."""

    name = "garch"

    def __init__(self, p: int = 1, q: int = 1):
        self.p, self.q = p, q

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from arch import arch_model

        rets = y.fillna(0.0).values * 100  # arch wants percent scale
        model = arch_model(rets, vol="Garch", p=self.p, q=self.q, mean="Constant",
                           rescale=False)
        self.fitted_ = model.fit(disp="off")
        self.conditional_vol_ = self.fitted_.conditional_volatility / 100
        self.train_residuals_ = y.values
        self.forecast_vol_ = float(
            self.fitted_.forecast(horizon=1, reindex=False).variance.values[-1][0]
        ) ** 0.5 / 100
        self.mean_ret_ = float(np.mean(rets)) / 100

    def predict_expected_return(self, x: pd.Series):
        return getattr(self, "mean_ret_", None)

    def predict(self, asset_symbol: str, horizon: str, x: pd.Series):
        from core.clock import get_clock
        from forecasting.base_model import _HORIZON_TD
        from core.interfaces import ForecastPoint

        vol = getattr(self, "forecast_vol_", None)
        mean = getattr(self, "mean_ret_", None)
        if vol is None or mean is None:
            return None
        now = get_clock().now()
        from scipy.stats import norm
        return ForecastPoint(
            model_id=self.name, asset_symbol=asset_symbol, horizon=horizon,
            generated_ts=now, target_ts=now + _HORIZON_TD[horizon],
            expected_return=mean,
            p_up=float(norm.cdf(mean / vol)) if vol > 0 else None,
            expected_volatility=float(vol),
            interval_lower=float(mean + norm.ppf(0.1) * vol),
            interval_upper=float(mean + norm.ppf(0.9) * vol),
            uncertainty_score=float(vol),
        )


class HARModel(ForecastModel):
    """HAR-RV: realized volatility from daily, weekly, monthly averages.
    Corsi (2009). Predicts vol; mean return assumed ~0."""
    name = "har"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # require a realized-vol column produced by features layer
        if "realized_vol_24" not in X:
            raise ValueError("HAR requires realized_vol_24 feature")
        rv = X["realized_vol_24"].astype(float)
        daily = rv
        weekly = rv.rolling(5).mean()
        monthly = rv.rolling(22).mean()
        lags = pd.DataFrame({"d": daily.shift(1), "w": weekly.shift(1), "m": monthly.shift(1)})
        target = rv
        mask = lags.notna().all(axis=1) & target.notna()
        Xl, yt = lags[mask], target[mask]
        self.coef_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(Xl)), Xl.values]), yt.values, rcond=None
        )[0]
        self.train_residuals_ = y.values

    def predict_expected_return(self, x: pd.Series):
        return 0.0

    def _predict_vol(self, x: pd.Series) -> float:
        d, w, m = x.get("realized_vol_24", np.nan), None, None
        rv = d
        if pd.isna(rv):
            return 0.0
        # weekly/monthly approximated from current value when history unavailable
        w, m = rv, rv
        return float(self.coef_ @ np.array([1.0, rv, w, m]))

    def predict(self, asset_symbol: str, horizon: str, x: pd.Series):
        from scipy.stats import norm

        from core.clock import get_clock
        from forecasting.base_model import _HORIZON_TD
        from core.interfaces import ForecastPoint

        vol = self._predict_vol(x)
        now = get_clock().now()
        return ForecastPoint(
            model_id=self.name, asset_symbol=asset_symbol, horizon=horizon,
            generated_ts=now, target_ts=now + _HORIZON_TD[horizon],
            expected_return=0.0,
            p_up=None, expected_volatility=vol,
            interval_lower=float(norm.ppf(0.1) * vol),
            interval_upper=float(norm.ppf(0.9) * vol),
            uncertainty_score=vol,
        )
