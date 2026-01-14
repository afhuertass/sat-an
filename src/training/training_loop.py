from typing import Callable

import numpy as np
import xarray as xr

from data.local_data import get_overlays_geometry
from dfetching.gcloud_utils import NetCDFAsset, open_netcdf_via_download
from dfetching.ingest import get_local_path
from processing.process import create_features
from training.gradient_boots_models import train_lgbm_cv, train_xgboost_forest_cv
from training.schemas import (
    CLITrainParams,
    LightGBMTrainingConfig,
    XGBoostForestTrainingConfig,
)
from validators.validators import IngestValidator


def train_with_params(ingest_params: IngestValidator, train_params: CLITrainParams):
    X, Y, _ = _get_features(ingest_params, train_params)

    model_params, training_func = _get_model_params(train_params=train_params, labels=Y)

    match (model_params, training_func):
        case (None, None):
            raise ValueError("ERROR")
        case (_, _):
            print(model_params)
            training_func(X, Y, model_params)  # type: ignore
    return


def _get_model_params(
    train_params, labels
) -> (
    tuple[LightGBMTrainingConfig | XGBoostForestTrainingConfig, Callable]
    | tuple[None, None]
):
    n_class = len(np.unique(labels))
    train_params.model_params["n_classes"] = n_class
    if train_params.model_type == "lgb":
        return LightGBMTrainingConfig(**train_params.model_params), train_lgbm_cv
    if train_params.model_type == "xgb":
        return XGBoostForestTrainingConfig(
            **train_params.model_params
        ), train_xgboost_forest_cv
    return None, None


def _get_features(ingest_params: IngestValidator, train_params: CLITrainParams):
    if train_params.cloud:
        X, Y, coords = _fetch_from_cloud(ingest_params)

        return X, Y, coords
    else:
        X, Y, coords = _fetch_from_local(ingest_params=ingest_params)
        return X, Y, coords


def _fetch_from_cloud(ingest_params: IngestValidator):
    cloud_asset = NetCDFAsset(
        product_id="SENTINEL2_L2",
        region_id=ingest_params.region.strftime("%Y-%m-%d"),  # type: ignore
        start_date=ingest_params.start_date.strftime("%Y-%m-%d"),  # type: ignore
        end_date=ingest_params.end_date,
    )

    _ds = open_netcdf_via_download(cloud_asset)
    geometry = get_overlays_geometry()
    X, y, coords = create_features(df_overlays=geometry, ds=_ds, label_in_df="Vocacion")

    return X, y, coords


def _fetch_from_local(ingest_params: IngestValidator):
    local_path = get_local_path(
        "SENTINEL2_L2",
        ingest_params.region,
        ingest_params.start_date,
        ingest_params.end_date,
    )
    local_path = local_path + "openEO.nc"
    _ds = xr.open_dataset(local_path)
    geometry = get_overlays_geometry()
    X, y, coords = create_features(df_overlays=geometry, ds=_ds, label_in_df="Vocacion")

    return X, y, coords
