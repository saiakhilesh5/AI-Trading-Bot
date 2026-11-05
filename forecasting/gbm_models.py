"""Gradient-boosted tree forecasters (LightGBM, XGBoost) with quantile
outputs for prediction intervals. CPU-fast, benchmarkable per asset/horizon."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from forecasting.base_model import ForecastModel


class _GBMBase(ForecastModel):
    """Shared implementation: mean model + quantile models for intervals."""

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xn = self._prepare(X)
        y = y.fillna(0.0)
        self.mean_model_.fit(Xn, y)
        resid = y.values - self.mean_model_.predict(Xn)
        self.train_residuals_ = resid
        self.quantile_models_ = {}
        for q in (0.1, 0.9):
            model_q = self._new_quantile_model(q)
            model_q.fit(Xn, y)
            self.quantile_models_[q] = model_q
        self.feature_names_ = list(Xn.columns)
        return

    def predict_expected_return(self, x: pd.Series):
        row = self._prepare(pd.DataFrame([x]))[self.feature_names_]
        if row.isna().all(axis=None):
            return None
        return float(self.mean_model_.predict(row)[0])

    def predict(self, asset_symbol: str, horizon: str, x: pd.Series) -> Optional[object]:
        from core.clock import get_clock
        from forecasting.base_model import _HORIZON_TD
        from core.interfaces import ForecastPoint

        expected = self.predict_expected_return(x)
        if expected is None:
            return None
        row = self._prepare(pd.DataFrame([x]))[self.feature_names_]
        lower = upper = None
        if self.quantile_models_:
            lower = float(self.quantile_models_[0.1].predict(row)[0])
            upper = float(self.quantile_models_[0.9].predict(row)[0])
        resid = self.train_residuals_
        vol = float(np.std(resid)) if resid is not None and len(resid) else None
        now = get_clock().now()
        return ForecastPoint(
            model_id=self.name, asset_symbol=asset_symbol, horizon=horizon,
            generated_ts=now, target_ts=now + _HORIZON_TD[horizon],
            expected_return=expected,
            p_up=float(np.clip(0.5 + expected / (2 * vol), 0.05, 0.95)) if vol else None,
            expected_volatility=vol, interval_lower=lower, interval_upper=upper,
            uncertainty_score=vol,
        )


class LightGBMModel(_GBMBase):
    name = "lightgbm"

    def __init__(self, params: dict | None = None):
        import lightgbm as lgb

        self.params = dict(
            objective="regression", n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=40, subsample=0.8,
            colsample_bytree=0.8, verbose=-1, deterministic=True,
            force_row_wise=True, seed=42,
        )
        self.params.update(params or {})
        self.mean_model_ = lgb.LGBMRegressor(**self.params)

    def _new_quantile_model(self, q: float):
        import lightgbm as lgb

        params = dict(self.params)
        params.update(objective="quantile", alpha=q)
        return lgb.LGBMRegressor(**params)


class XGBoostModel(_GBMBase):
    name = "xgboost"

    def __init__(self, params: dict | None = None):
        import xgboost as xgb

        self.params = dict(
            objective="reg:squarederror", n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8, seed=42,
        )
        self.params.update(params or {})
        self.mean_model_ = xgb.XGBRegressor(**self.params)

    def _new_quantile_model(self, q: float):
        import xgboost as xgb

        params = dict(self.params)
        params.update(objective="reg:quantileerror", quantile_alpha=q)
        return xgb.XGBRegressor(**params)
