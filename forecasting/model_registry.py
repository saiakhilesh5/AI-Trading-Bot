"""Model registry: every trained model version is recorded with its
training range, config, and validation metrics. Models move between
registered -> active -> shadow -> disabled per drift rules. Humans
approve promotions; nothing auto-deploys."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from core.clock import get_clock
from data_layer.storage.schemas import Experiment, ForecastRecord, ModelRegistration


def register_model(
    session: Session, model_type: str, asset_symbol: str, horizon: str,
    feature_config: dict, hyperparameters: dict, training_range: tuple[datetime, datetime],
    validation_metrics: dict,
) -> str:
    model_id = f"{model_type}:{asset_symbol}:{horizon}:{uuid.uuid4().hex[:8]}"
    session.add(ModelRegistration(
        model_id=model_id, model_type=model_type, version="1",
        asset_symbol=asset_symbol, horizon=horizon,
        feature_config=json.dumps(feature_config),
        hyperparameters=json.dumps(hyperparameters),
        training_data_range=json.dumps(
            {"start": training_range[0].isoformat(), "end": training_range[1].isoformat()}),
        validation_metrics=json.dumps(validation_metrics),
        registered_ts=get_clock().now(), status="registered",
    ))
    return model_id


def set_status(session: Session, model_id: str, status: str) -> bool:
    row = session.get(ModelRegistration, model_id)
    if row is None:
        return False
    row.status = status  # registered | active | shadow | disabled
    return True


def active_models(session: Session, asset_symbol: str | None = None) -> list[ModelRegistration]:
    q = session.query(ModelRegistration).filter(ModelRegistration.status == "active")
    if asset_symbol:
        q = q.filter(ModelRegistration.asset_symbol == asset_symbol)
    return q.all()


def record_forecast(session: Session, model_id: str, asset_symbol: str, horizon: str,
                    payload: dict, target_ts: datetime) -> None:
    session.add(ForecastRecord(
        ts=get_clock().now(), model_id=model_id, asset_symbol=asset_symbol,
        horizon=horizon, target_ts=target_ts, payload_json=json.dumps(payload),
    ))


def create_experiment(session: Session, description: str, dataset_version: str,
                      config: dict) -> str:
    experiment_id = f"exp-{uuid.uuid4().hex[:12]}"
    session.add(Experiment(
        experiment_id=experiment_id, created_ts=get_clock().now(),
        description=description, dataset_version=dataset_version,
        config_json=json.dumps(config), status="pending",
    ))
    return experiment_id


def record_experiment_result(session: Session, experiment_id: str,
                             result: dict, outcome: str) -> None:
    row = session.get(Experiment, experiment_id)
    if row is None:
        return
    row.result_json = json.dumps(result)
    row.status = "done"
    row.outcome = outcome  # 'adopted' | 'rejected'
