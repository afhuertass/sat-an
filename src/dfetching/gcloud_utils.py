from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import xarray as xr
from google.cloud import storage
from pydantic import BaseModel


class NetCDFAsset(BaseModel):
    product_id: str  # e.g. "sentinel2-l2a"
    region_id: str  # e.g. "18NYF" or "r034_c019"
    start_date: str
    end_date: str
    version: str = "v001"

    def key_prefix(self) -> str:
        """
        Generates a GCS key prefix based on the product, region, and date range.

        Returns:
            str: The generated key prefix for GCS objects.
        """
        # partition by region and year/month of the start (you could also use end)

        dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        y = dt.year
        return f"satellite/{self.product_id}/region={self.region_id}/year={y}/"

    def filename_base(self) -> str:
        """
        Constructs the base filename for a NetCDF file using region, start, end dates, and version.

        Returns:
            str: The constructed filename base.
        """
        from datetime import datetime

        # Convert start_date and end_date to datetime objects for processing
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        s = start_dt.strftime("%Y%m%d")
        e = end_dt.strftime("%Y%m%d")
        return f"{self.region_id}_{s}_{e}_{self.version}"

    def nc_key(self) -> str:
        """
        Generates the complete GCS key for a NetCDF file using the prefix and filename base.

        Returns:
            str: The full GCS key for the NetCDF file.
        """
        return self.key_prefix() + self.filename_base() + ".nc"


def gcs_client() -> storage.Client:
    """
    Creates a Google Cloud Storage client instance using default authentication methods.

    Returns:
        storage.Client: The authenticated Google Cloud Storage client.
    """
    # Uses your `gcloud auth application-default login` credentials
    return storage.Client()


def upload_file(
    local_path: str | Path,
    gcs_key: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """
    Uploads a local file to Google Cloud Storage.

    Args:
        local_path (str | Path): File path of the file to upload
        gcs_key (str): Destination key on GCS where the file will be stored
        metadata (Optional[Dict[str, str]]): Additional metadata to set for the file

    Returns:
        str: Full URI of the uploaded file in GCS
    """
    client = gcs_client()
    bucket = "sat-an"
    b = client.bucket(bucket)
    blob = b.blob(gcs_key)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{gcs_key}"


def download_file(gcs_key: str, local_path: str | Path) -> Path:
    """
    Downloads a file from Google Cloud Storage to a local path.

    Args:
        bucket (str): The GCS bucket from which the file is to be downloaded
        gcs_key (str): The GCS key of the file to be downloaded
        local_path (str | Path): Local path where the file will be saved

    Returns:
        Path: The path to the downloaded file
    """
    client = gcs_client()
    b = client.bucket("sat-an")
    blob = b.blob(gcs_key)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path


def exists_cloud(gcs_key: str) -> bool:
    """
    Checks if a file exists in Google Cloud Storage.

    Args:
        gcs_key (str): The GCS key of the file to check

    Returns:
        bool: True if the file exists, False otherwise
    """
    client = gcs_client()
    b = client.bucket("sat-an")
    return b.blob(gcs_key).exists(client)


def list_keys(prefix: str) -> List[str]:
    """
    Lists all keys in Google Cloud Storage within a specified bucket and prefix.

    Args:
        prefix (str): The prefix to filter keys by

    Returns:
        List[str]: A list of keys under the given bucket and prefix
    """
    client = gcs_client()
    b = client.bucket("sat-an")
    return [blob.name for blob in client.list_blobs(b, prefix=prefix)]


def open_netcdf_from_gcs(
    gcs_uri: str,
    engine: str = "netcdf4",  # othwr engines are giving me trouble
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    Stream-open NetCDF from GCS without downloading, using fsspec/gcsfs.
    Works for many files but sometimes NetCDF/HDF5 prefers local access; if so, use download + open.
    """
    import fsspec

    fs = fsspec.filesystem("gcs")
    with fs.open(gcs_uri, "rb") as f:
        # xarray can read from file-like object in many cases
        ds = xr.open_dataset(f, engine=engine, chunks=chunks)
    return ds


def open_netcdf_via_download(
    asset: NetCDFAsset,
    cache_dir: str | Path = "./cache",
    engine: str = "netcdf4",
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    More reliable than streaming for HDF5-backed NetCDF: downloads once, then opens locally.
    """
    cache_dir = Path(cache_dir)
    local_path = cache_dir / asset.filename_base() / (asset.filename_base() + ".nc")
    if not local_path.exists():
        download_file(asset.nc_key(), local_path)
    return xr.open_dataset(local_path, engine=engine, chunks=chunks)
