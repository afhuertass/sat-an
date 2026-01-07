"""Module for random forest and gradient boost based pixel classifiers"""

from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, Field


class CLITrainParams(BaseModel):
    model_type: Literal["xgb", "lgb"]
    model_params: dict


class BaseTrainingConfig(BaseModel):
    # Data / CV
    n_splits: int = Field(
        5, ge=2, description="The number of folds for k-fold cross-validation."
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    # Logging
    wandb_project: str = Field(
        "soiluse-tabular", description="The Weight and Biases project to log to."
    )
    run_name: str = Field("aaa", description="The name of the current run/session.")

    # I/O
    save_artifacts: bool = Field(
        True, description="Flag to save training artifacts such as models and logs."
    )
    output_dir: Path = Field(
        Path("models"), description="Directory where output artifacts are saved."
    )
    # Dataset assumptions
    n_classes: int = Field(
        5, description="The number of unique classes in the dataset."
    )
    # Safety
    assert_finite_X: bool = Field(
        True, description="Ensure that all inputs are finite."
    )


class LightGBMTrainingConfig(BaseTrainingConfig):
    # Booster params
    learning_rate: float = Field(
        0.05, description="The learning rate for the LightGBM model."
    )
    num_leaves: int = Field(
        63, description="The maximum number of leaves in each LightGBM tree."
    )
    max_depth: int = Field(-1, description="The maximum depth of the LightGBM trees.")
    min_data_in_leaf: int = Field(
        100, description="Minimum number of data points in a leaf."
    )
    reg_lambda: float = Field(
        1.0, description="L2 regularization term on weights in the LightGBM model."
    )

    # Training control
    num_boost_round: int = Field(
        2000, description="The number of boosting rounds or trees to build."
    )
    early_stopping_rounds: int = Field(
        100,
        description="The criteria to stop training if validation score isn't improving.",
    )
    log_every_n: int = Field(
        50, description="Frequency of logging the model's metrics during training."
    )

    # Sampling
    subsample: float = Field(
        0.8,
        description="The subsample ratio of the training instances for training the LightGBM model.",
    )
    colsample_bytree: float = Field(
        0.8, description="The subsample ratio of columns when constructing each tree."
    )

    num_class: int = Field(
        5, description="The number of parallel threads used to run LightGBM."
    )
    # System
    n_jobs: int = Field(
        -1, description="The number of parallel threads used to run LightGBM."
    )

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
    n_estimators: int = Field(
        800, description="The number of trees to build in the XGBoost model."
    )
    max_depth: int = Field(
        10, description="The maximum depth of each tree in the XGBoost model."
    )
    min_child_weight: float = Field(
        1.0, description="Minimum sum of instance weight (hessian) needed in a child."
    )
    reg_lambda: float = Field(
        1.0, description="L2 regularization term on weights in the XGboost model."
    )

    # Sampling
    subsample: float = Field(
        0.8,
        description="The subsample ratio of the training instances for training the LightGBM model.",
    )
    colsample_bynode: float = Field(
        0.8, description="The subsample ratio of columns for each split, in each level."
    )

    # System
    n_jobs: int = Field(
        -1, description="The number of parallel threads used to run XGboost models."
    )
    tree_method: str = Field(
        "hist", description="The tree construction algorithm used in XGBoost."
    )

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


class TrainParams:
    model_type: Literal["lgb", "xgb"]
    model_params: LightGBMTrainingConfig | XGBoostForestTrainingConfig
