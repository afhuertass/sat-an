from typing import Callable

from training.gradient_boots_models import train_lgbm_cv, train_xgboost_forest_cv
from training.schemas import (
    CLITrainParams,
    LightGBMTrainingConfig,
    XGBoostForestTrainingConfig,
)
from validators.validators import IngestValidator


def train_with_params(ingest_params: IngestValidator, train_params: CLITrainParams):
    model_params, training_func = _get_model_params(train_params=train_params)
    X, Y = _get_features(ingest_params)

    match (model_params, training_func):
        case (None, None):
            raise ValueError("ERROR")
        case (_, _):
            print(model_params)
            training_func(X, Y, model_params)  # type: ignore
    return


def _get_model_params(
    train_params,
) -> (
    tuple[LightGBMTrainingConfig | XGBoostForestTrainingConfig, Callable]
    | tuple[None, None]
):
    if train_params.model_type == "lgb":
        return LightGBMTrainingConfig(**train_params.model_params), train_lgbm_cv
    if train_params.model_type == "xgb":
        return XGBoostForestTrainingConfig(
            **train_params.model_params
        ), train_xgboost_forest_cv
    return None, None


def _get_features(ingest_params: IngestValidator):
    return [], []
