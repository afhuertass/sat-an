import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from datetime import datetime
    import typer
    import marimo as mo
    import pandas as pd
    import polars as pl
    import numpy as np
    from netCDF4 import Dataset
    from matplotlib import gridspec
    import xarray as xr
    import folium
    import json
    import geopandas as gpd
    import datetime
    import scipy
    from typing_extensions import Annotated
    from unidecode import unidecode
    import openeo as eo
    from validators.validators import IngestValidator
    import xarray as xr
    from pyproj import CRS
    from pyproj import Proj
    import rasterio
    import matplotlib.pyplot as plt
    return Annotated, IngestValidator, eo, gpd, mo, typer, unidecode


@app.cell
def _():
    from dfetching.ingest import get_data
    return (get_data,)


@app.cell
def _(Annotated, IngestValidator, eo, get_data, typer):
    app = typer.Typer(help="CLI for the ML pipeline")


    @app.command(help="Ingest raw data from sources.")
    def ingest(
        region: Annotated[
            str,
            typer.Argument(
                help="Region to fetch (check available regions using sat-an help)"
            ),
        ],
        start_date: Annotated[
            str, typer.Argument(help="Startinng date to fetch data")
        ],
        end_date: Annotated[str, typer.Argument(help="End date of fetching")],
        cloud: Annotated[
            bool,
            typer.Option(
                "--cloud",
                help="Save result to cloud",
            ),
        ] = False
    ) -> None:
        params = {"region": region, "start_date": start_date, "end_date": end_date}
        ingest_params = IngestValidator(**params)
        print("[ingest] Starting data ingestion...")
        connection = eo.connect(
            url="openeo.dataspace.copernicus.eu"
        ).authenticate_oidc()
        results = get_data(ingest_params=ingest_params, connection=connection, cloud=cloud)


    @app.command(help="Ingest raw data from sources.")
    def ingest_train(
        region: Annotated[
            str,
            typer.Argument(
                help="Region to fetch (check available regions using sat-an help)"
            ),
        ],
    ) -> None:
        # Fetch training data for a region, the training data is for the 2017, that is where the labels are from

        ingest_params = IngestValidator(
            **{
                "region": region,
                "start_date": "2017-01-01",
                "end_date": "2017-12-31",
            }
        )

        return None


    @app.command(help="Preprocess the ingested data.")
    def preprocess() -> None:
        print("[preprocess] Starting data preprocessing...")
        print("[preprocess] TODO: implement preprocessing steps.")
        print("[preprocess] Finished (placeholder).")


    @app.command(
        name="build-data", help="Build final datasets for training and evaluation."
    )
    def build_data() -> None:
        print("[build-data] Building datasets...")
        print("[build-data] TODO: implement dataset building.")
        print("[build-data] Finished (placeholder).")


    @app.command(help="Train the model.")
    def train() -> None:
        print("[train] Starting model training...")
        print("[train] TODO: implement model training.")
        print("[train] Finished (placeholder).")


    @app.command(help="Serve the trained model via an API.")
    def serve(
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        print(f"[serve] Starting server at http://{host}:{port} ...")
        print("[serve] TODO: implement model serving.")
        print("[serve] Server running (placeholder). Press Ctrl+C to stop.")


    @app.command(help="Print help for the tool")
    def help() -> None:
        print("Help here")
    return (app,)


@app.cell
def _(app, gpd, mo, unidecode):
    if mo.app_meta().mode == "script":
        #Run typer app
        app()

    else: 


        df = gpd.read_file(
            "/Users/andres/sat/sat-anomaly/src/data/colombian-towns.geojson"
        )
        df = df.rename(
            columns = {
                "MPIO_CNMBR" : "region_name"
            }
        )
        df["region_name"] = df["region_name"].str.lower().str.replace(" ", "-").apply(lambda x: unidecode(x))
        df["region_name"].unique()

        df

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
