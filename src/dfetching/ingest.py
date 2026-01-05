import os.path

import geopandas as gpd
from openeo import DataCube
from openeo.extra.spectral_indices import compute_indices
from openeo.processes import linear_scale_range
from unidecode import unidecode

from dfetching.gcloud_utils import NetCDFAsset
from validators.validators import IngestValidator

from .gcloud_utils import exists_cloud, upload_file

# List of index to calculate from the SAR data
index_to_calculate = ["NDVI", "NDMI", "NDRE1", "NDRE2", "NDRE5", "ANIR"]
_index_dict = {idx: [-1, 1] for idx in index_to_calculate}
_index_dict["ANIR"] = [0, 1]

INDEX_DICT = {
    "collection": {"input_range": None, "output_range": None},
    "indices": _index_dict,
}


def get_data(ingest_params: IngestValidator, connection, cloud):
    aoi = get_aoi(ingest_params.region)
    index_cube, visible_cube = retrieve_sat_region(
        aoi, connection, ingest_params.start_date, ingest_params.end_date
    )
    # visible_cube = visible_cube.save_result(format="GTiff")
    visible_cube = visible_cube.save_result(format="netcdf")
    local_path = _get_local_path(
        "SENTINEL2_L2",
        ingest_params.region,
        ingest_params.start_date,
        ingest_params.end_date,
    )
    local_nc_path = local_path + "openEO.nc"
    print(local_nc_path)
    if not os.path.isfile(local_nc_path):
        job = visible_cube.create_job(title=f"visible-{ingest_params.region}")
        job.start_and_wait()
        results = job.get_results()
        results.download_files(local_path)
        if cloud:
            netcdf_to_cloud(local_path=local_nc_path, ingest_params=ingest_params)
    else:
        if cloud:
            netcdf_to_cloud(local_path=local_nc_path, ingest_params=ingest_params)


def _get_local_path(
    product_id: str, region_id: str, start_date: str, end_date: str
) -> str:
    """
    Generates a local path based on the product, region, and date range.

    Returns:
        str: The generated key prefix for GCS objects.
    """
    # partition by region and year/month of the start (you could also use end)
    start = start_date.strftime("%Y%m%d")  # type: ignore
    end = end_date.strftime("%Y%m%d")  # type: ignore
    y = start_date.year  # type: ignore
    local_path = f"../output/satellite/{product_id}/region={region_id}/year={y}/start_{start}_end_{end}/"
    os.makedirs(local_path, exist_ok=True)
    return local_path


def netcdf_to_cloud(local_path: str, ingest_params: IngestValidator):
    start_date = ingest_params.start_date.strftime("%Y-%m-%d")  # type: ignore
    end_date = ingest_params.end_date.strftime("%Y-%m-%d")  # type: ignore
    print(start_date)
    print(end_date)
    netcdf_cloud_params = NetCDFAsset(
        product_id="SENTINEL2_L2",
        region_id=ingest_params.region,
        start_date=start_date,
        end_date=end_date,
    )
    gc_key = netcdf_cloud_params.nc_key()
    if exists_cloud(gc_key):
        return gc_key
    else:
        upload_file(local_path, gc_key)
        return gc_key


def process_cube_training():
    return


def retrieve_sat_region(
    aoi: dict, connection, start_date, end_date
) -> tuple[DataCube, DataCube]:
    """returns a tuple of data cubes"""

    # hard coded for now
    col_id = "SENTINEL2_L2A"
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    temporal_extend = [start_date, end_date]

    cube = connection.load_collection(
        collection_id=col_id, temporal_extent=temporal_extend, bands=bands
    ).filter_spatial(aoi)

    # processing cloudy pixeles
    # cloud_mask = get_cloud_mask(connection, aoi, temporal_extend)
    # cube = cube.mask(cloud_mask)

    indices = compute_indices(cube, index_to_calculate)

    combined = cube.merge_cubes(indices)
    # dekad is a period of 10 days, 3 per month
    idx_dekad = combined.aggregate_temporal_period("dekad", reducer="median")
    idx_dekad = idx_dekad.apply_dimension(
        dimension="t", process="array_interpolate_linear"
    )
    # Filter the computed indicex, this is to be used for features
    result_indices = combined.filter_bands(index_to_calculate + bands)
    # Visible bands
    cube_visible = combined.filter_bands(["B04", "B03", "B02"])

    cube_visible = cube_visible.apply(
        lambda x: linear_scale_range(
            x, inputMin=0, inputMax=5000, outputMin=0, outputMax=255
        )
    )
    return result_indices, cube_visible


def get_cloud_mask(connection, aoi, temporal_extend):
    props = {"eo:cloud_cover": lambda v: v <= 80}
    scl = connection.load_collection(
        "SENTINEL2_L2A", temporal_extent=temporal_extend, bands="SCL", properties=props
    ).filter_spatial(aoi)
    cloud_mask = scl.process(
        "to_scl_dilation_mask",
        data=scl,
        kernel1_size=17,
        kernel2_size=77,
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    )
    return cloud_mask


def get_aoi(region: str) -> dict:
    _df = gpd.read_file("data/colombian-towns.geojson")
    _df = _df.query(f"region_name == '{region}'")
    return _df.to_geo_dict()


def get_regions() -> list[str]:
    df = gpd.read_file("data/colombian-towns.geojson")
    df["region_name"] = (
        df["region_name"]
        .str.lower()
        .str.replace(" ", "-")
        .apply(lambda x: unidecode(x))
    )
    regions = list(df["region_name"].unique())
    return regions
