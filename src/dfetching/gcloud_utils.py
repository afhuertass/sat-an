from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import xarray as xr
from google.cloud import storage


@dataclass(frozen=True)
class NetCDFAsset:
    product_id: str  # e.g. "sentinel2-l2a"
    region_id: str  # e.g. "18NYF" or "r034_c019"
    start: date
    end: date
    version: str = "v001"

    def key_prefix(self) -> str:
        # partition by region and year/month of the start (you could also use end)
        y = self.start.year
        m = f"{self.start.month:02d}"
        span = f"start={self.start.isoformat()}_end={self.end.isoformat()}"
        return f"satellite/{self.product_id}/region={self.region_id}/year={y}/month={m}/{span}/"

    def filename_base(self) -> str:
        s = self.start.strftime("%Y%m%d")
        e = self.end.strftime("%Y%m%d")
        return f"tile_{self.region_id}_{s}_{e}_{self.version}"

    def nc_key(self) -> str:
        return self.key_prefix() + self.filename_base() + ".nc"


def gcs_client() -> storage.Client:
    # Uses your `gcloud auth application-default login` credentials
    return storage.Client()


def upload_file(
    local_path: str | Path,
    gcs_key: str,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    client = gcs_client()
    bucket = "sat-an"
    b = client.bucket(bucket)
    blob = b.blob(gcs_key)
    if metadata:
        blob.metadata = metadata
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{gcs_key}"


def download_file(bucket: str, gcs_key: str, local_path: str | Path) -> Path:
    client = gcs_client()
    b = client.bucket(bucket)
    blob = b.blob(gcs_key)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path


def exists(bucket: str, gcs_key: str) -> bool:
    client = gcs_client()
    b = client.bucket("sat-an")
    return b.blob(gcs_key).exists(client)


def list_keys(bucket: str, prefix: str) -> List[str]:
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
    bucket: str,
    asset: NetCDFAsset,
    cache_dir: str | Path = "./cache",
    engine: str = "h5netcdf",
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    More reliable than streaming for HDF5-backed NetCDF: downloads once, then opens locally.
    """
    cache_dir = Path(cache_dir)
    local_path = cache_dir / asset.filename_base() / (asset.filename_base() + ".nc")
    if not local_path.exists():
        download_file(bucket, asset.nc_key(), local_path)
    return xr.open_dataset(local_path, engine=engine, chunks=chunks)
