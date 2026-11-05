"""Forecasting pipeline: build features, walk-forward evaluate every enabled
candidate model against baselines, register winners. Chronological only."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import get_model_config
from core.logging_setup import get_logger
from data_layer.storage.data_access import get_session
from data_layer.storage import repository as repo
from forecasting import model_registry as registry
from forecasting.base_model import ForecastModel, forward_returns
from forecasting.baselines import BASELINES, RandomWalkModel
from forecasting.evaluation import EvalResult, WalkForwardConfig, walk_forward_evaluate

log = get_logger()


def load_feature_frame(session, asset_symbol: str, interval: str,
                       as_of: datetime | None = None) -> pd.DataFrame:
    """Candles -> full feature frame, point-in-time safe."""
    from features import assemble
    from data_layer.validators.data_health import candles_to_frame

    candles = repo.get_candles(session, asset_symbol, interval, as_of=as_of)
    frame = candles_to_frame(candles)
    if frame.empty:
        return frame
    return assemble(asset_symbol, frame)


def candidate_models() -> dict[str, type[ForecastModel]]:
    from forecasting.gbm_models import LightGBMModel, XGBoostModel
    from forecasting.vol_models import GARCHModel, HARModel

    return {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "garch": GARCHModel,
        "har": HARModel,
    }


def evaluate_all(asset_symbol: str, interval: str, horizon: str,
                 train_bars: int = 2000, test_bars: int = 300) -> list[EvalResult]:
    """Full benchmark: all baselines + enabled candidates, walk-forward.
    Train window adapts downward when history is short, never below the
    configured minimum minus a documented floor of 400 bars."""
    cfg = get_model_config()
    session = get_session()
    try:
        frame = load_feature_frame(session, asset_symbol, interval)
    finally:
        session.close()
    n = len(frame)
    if frame.empty or n < 500:
        log.warning("insufficient_data", asset=asset_symbol, bars=n)
        return []
    # adaptive sizing keeps walk-forward feasible for shorter histories
    usable = n - cfg["validation"]["train_eval_gap_bars"] - 10
    train_bars = max(400, min(train_bars, int(usable * 0.7)))
    test_bars = max(100, min(test_bars, int(usable * 0.15)))

    y = forward_returns(frame["close"], horizon)
    X = frame
    wf_config = WalkForwardConfig(
        n_splits=5, train_bars=train_bars, test_bars=test_bars,
        gap_bars=cfg["validation"]["train_eval_gap_bars"],
    )

    results: list[EvalResult] = []
    # baselines first — the bar to clear
    for name, cls in BASELINES.items():
        try:
            results.append(walk_forward_evaluate(cls(), X, y, asset_symbol, horizon, wf_config))
        except Exception as e:
            log.warning("baseline_failed", model=name, error=str(e))
    # candidates, each evaluated against random walk via Diebold-Mariano
    candidates = candidate_models()
    for name, cls in candidates.items():
        if not cfg["candidate_models"].get(name, {}).get("enabled", False):
            continue
        try:
            results.append(walk_forward_evaluate(
                cls(), X, y, asset_symbol, horizon, wf_config, baseline=RandomWalkModel()))
        except Exception as e:
            log.warning("candidate_failed", model=name, error=str(e))
    return results


def benchmark_and_register(asset_symbol: str, interval: str, horizon: str) -> dict:
    """Run the benchmark and register models that beat random walk."""
    results = evaluate_all(asset_symbol, interval, horizon)
    session = get_session()
    try:
        experiment_id = registry.create_experiment(
            session,
            f"benchmark {asset_symbol} {interval}->{horizon}",
            dataset_version=interval,
            config={"train_bars": 2000, "test_bars": 300},
        )
        adopted = []
        rw = next((r for r in results
                   if r.model == "baseline_random_walk" and r.n_predictions > 0), None)
        for r in results:
            beats_rw = (
                rw is not None and r.mae is not None and rw.mae is not None and r.mae < rw.mae
            )
            if r.model.startswith("baseline_"):
                continue
            if beats_rw and r.n_predictions >= 100:
                metrics = r.to_dict()
                model_id = registry.register_model(
                    session, r.model, asset_symbol, horizon,
                    feature_config={"features": "standard"},
                    hyperparameters={}, training_range=(datetime.now(timezone.utc),) * 2,
                    validation_metrics=metrics,
                )
                registry.set_status(session, model_id, "registered")
                adopted.append(model_id)
        registry.record_experiment_result(
            session, experiment_id,
            {"results": [r.to_dict() for r in results]},
            "adopted" if adopted else "rejected",
        )
        session.commit()
        return {
            "experiment_id": experiment_id,
            "results": [r.to_dict() for r in results],
            "adopted_models": adopted,
        }
    finally:
        session.close()
