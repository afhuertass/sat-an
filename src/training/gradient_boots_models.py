import os

import lightgbm as lgb
import numpy as np
import xgboost
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRFClassifier

import wandb
from wandb.integration.lightgbm import log_summary, wandb_callback
from wandb.integration.xgboost import WandbCallback

from .schemas import LightGBMTrainingConfig, XGBoostForestTrainingConfig


def train_lgbm_cv(X, y, cfg: LightGBMTrainingConfig) -> list[lgb.Booster]:
    """
    Trains a LightGBM model using stratified k-fold cross-validation.
    Log the results to a Wandb project

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        cfg (LightGBMTrainingConfig): Configuration parameters for training.

    Returns:
        list of lgb.Booster: A list of trained LightGBM booster models corresponding to each fold.
    """
    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )
    # oof = np.zeros((len(y), cfg.n_classes), dtype=np.float32)
    models = []
    # Login to weights and biases to log experiments
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    params_wandb = cfg.lgb_params()
    params_wandb["allow_val_change"] = True
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=params_wandb,
    )
    n_classes = len(np.unique(y))
    print("Final unique classes", np.unique(y))
    for _fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtrain = lgb.Dataset(X[tr], label=y[tr])
        dvalid = lgb.Dataset(X[va], label=y[va])
        params = cfg.lgb_params()
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=cfg.num_boost_round,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                wandb_callback(),
            ],
        )
        # proba = booster.predict(X[va], num_iteration=booster.best_iteration)
        models.append(booster)

        log_summary(booster, save_model_checkpoint=True)
    run.finish()
    return models


def train_xgboost_forest_cv(
    X, y, cfg: XGBoostForestTrainingConfig
) -> list[xgboost.Booster]:
    """
    Trains an XGBoost Random Forest model using stratified k-fold cross-validation.

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        cfg (XGBoostForestTrainingConfig): Configuration parameters for training.

    Returns:
        list of xgboost.Booster: A list of trained XGBoost RF models corresponding to each fold.
    """
    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )

    models = []
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=cfg.model_dump(),
    )

    for _fold, (tr, va) in enumerate(skf.split(X, y), 1):
        model = XGBRFClassifier(**cfg.xgb_params())
        model.fit(X[tr], y[tr], WandbCallback(log_model=True))

        # proba = model.predict_proba(X[va])

        models.append(model)

    run.finish()
    return models
