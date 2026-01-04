"""Module for random forest and gradient boost based pixel classifiers"""

from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class BaseTrainingConfig(BaseModel):
    # Data / CV
    n_splits: int = Field(5, ge=2)
    seed: int = 42

    # Logging
    wandb_project: str = "soiluse-tabular"
    run_name: str

    # I/O
    save_artifacts: bool = True
    output_dir: Path = Path("models")
    # Dataset assumptions
    n_classes: int
    # Safety
    assert_finite_X: bool = True


class LightGBMTrainingConfig(BaseTrainingConfig):
    # Booster params
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 100
    reg_lambda: float = 1.0

    # Training control
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    log_every_n: int = 50

    # Sampling
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # System
    n_jobs: int = -1

    def lgb_params(self) -> Dict:
        return dict(
            objective="multiclass",
            num_class=self.n_classes,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            metric="multi_logloss",
            verbosity=-1,
        )


class XGBoostForestTrainingConfig(BaseTrainingConfig):
    # Forest params
    n_estimators: int = 800
    max_depth: int = 10
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0

    # Sampling
    subsample: float = 0.8
    colsample_bynode: float = 0.8

    # System
    n_jobs: int = -1
    tree_method: str = "hist"  # or "gpu_hist"

    def xgb_params(self) -> Dict:
        return dict(
            objective="multi:softprob",
            num_class=self.n_classes,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=1.0,  # RF-style
            subsample=self.subsample,
            colsample_bynode=self.colsample_bynode,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss",
            random_state=self.seed,
        )
