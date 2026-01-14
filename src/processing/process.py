import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from sklearn.preprocessing import LabelEncoder


def rioversion():
    print(rioxarray.__version__)


def create_features(
    df_overlays: gpd.GeoDataFrame, ds: xr.Dataset, label_in_df: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # create labels assuming for now there is always band B02
    labels, output_shape = create_labels(
        df_overlays, ds, band="B02", t=0, label_in_df=label_in_df
    )
    t_len = ds.dims["t"]
    n_bands = len(ds.data_vars.items()) - 1

    X_tybx = np.empty(
        (t_len, output_shape[0], output_shape[1], n_bands), dtype=np.float32
    )
    iband = 0
    for band, da in ds.data_vars.items():
        if "crs" in band:
            continue
        else:
            X_tybx[:, :, :, iband] = da.values.astype(np.float32)
            iband += 1

    n_bands = len(ds.data_vars.items()) - 1
    spatial_x = ds["x"].values
    spatial_y = ds["y"].values

    # Commentings for now the projection, it was killing my machine :(
    # ref = CRS.from_string(proje_string)
    # good_p = Proj( proje_string )
    # lon, lat = np.meshgrid(
    #    spatial_y, spatial_x
    # )
    # spatial_x, spatial_y = good_p(lon, lat, inverse=True)
    ds.close()
    ## Masking the invalid pixels
    valid_mask_yx = np.isfinite(X_tybx).all(axis=(0, 3))
    rows, cols = np.where(valid_mask_yx)
    # lon, lat = good_p(lon, lat, inverse=True)
    x_valid = spatial_x[cols]
    y_valid = spatial_y[rows]

    coords_xy = np.column_stack([x_valid, y_valid])
    n_valid = rows.size
    X_valid_tyb = X_tybx[:, rows, cols, :]
    X_valid = np.transpose(X_valid_tyb, (1, 0, 2))
    X_valid = X_valid.reshape(n_valid, -1)
    # valid labels
    labels_valid = labels[rows, cols]
    labels_valid = labels_valid.reshape((-1))
    le = LabelEncoder()
    labels_valid = le.fit_transform(labels_valid)
    return X_valid, labels_valid, coords_xy


def create_labels(
    geometry_df, _ds: xr.Dataset, band: str, t: int, label_in_df: str
) -> tuple[np.ndarray, tuple]:
    _crs = _ds["crs"].attrs.get("crs_wkt")
    geometry_df = geometry_df.to_crs(_crs)
    _da = _ds.isel(t=t)[band]
    output_shape = _da.shape
    transform = _da.rio.transform()
    classes = sorted(geometry_df[label_in_df].dropna().unique())
    class_to_id = {c: i + 1 for i, c in enumerate(classes)}

    shapes = (
        (geom, class_to_id[label])
        for geom, label in zip(geometry_df.geometry, geometry_df[label_in_df])
    )
    label_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=output_shape,
        transform=transform,
        fill=0,
        dtype="int16",
    )
    y = label_raster.reshape(output_shape)
    return y, output_shape  # ignore: type
