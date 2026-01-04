import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
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
    from unidecode import unidecode
    import rioxarray
    from rasterstats import zonal_stats
    import rasterio
    return datetime, folium, gpd, json, mo, np, rasterio, scipy, unidecode, xr


@app.cell
def _():
    import h5netcdf
    return


@app.cell
def _():
    from pyproj import CRS
    from pyproj import Proj
    import matplotlib
    return CRS, Proj, matplotlib


@app.cell
def _():
    import openeo as eo

    from openeo.processes import array_apply, linear_scale_range, reduce_dimension
    return eo, linear_scale_range


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(gpd, unidecode):
    def prepare_towns_from_json():
        """Clean up function to standirize the town names"""

        _df = gpd.read_file("./data/Colombia_departamentos_municipios_CNPV2018.topojson")
        _df = _df[ [ "MPIO_CNMBR" , "LATITUD" , "LONGITUD" , "geometry"] ]

        _df = _df.rename(
            columns = {
                "MPIO_CNMBR":"region_name"
            }
        )
        _df["region_name"] = _df["region_name"].str.lower().str.replace(" ", "-").apply(lambda x: unidecode(x))
        _df["region_name"].unique()
        _df.to_file("data/colombian-towns.geojson")

    prepare_towns_from_json()
    return


@app.cell
def _(gpd, np, rasterio, xr):
    def create_features(
        df_overlays: gpd.GeoDataFrame, path_to_nc: str, t: int, label_in_df: str
    ) -> tuple[np.array, np.array, np.array]:
        _ds = xr.open_dataset(path_to_nc)

        _resu = _ds.to_dict()
        proje_string = _resu["data_vars"]["crs"]["attrs"]["crs_wkt"]

        # I'll create labels assuming for now there is always band B02
        labels, output_shape = create_labels(
            df_overlays, _ds, band="B02", t=0, label_in_df=label_in_df
        )
        t_len = _ds.dims["t"]
        n_bands = len(_ds.data_vars.items()) - 1
        y_size = output_shape[0]
        x_size = output_shape[1]

        n_pixels = output_shape[0] * output_shape[1]
        X_tybx = np.empty(
            (t_len, output_shape[0], output_shape[1], n_bands), dtype=np.float32
        )
        iband = 0
        for band, da in _ds.data_vars.items():
            if "crs" in band:
                continue
            else:
                X_tybx[:, :, :, iband] = da.values.astype(np.float32)
                iband += 1

        n_bands = len(_ds.data_vars.items()) - 1
        spatial_x = _ds["x"].values
        spatial_y = _ds["y"].values

        # Commentings for now the projection, it was killing my machine :(
        # ref = CRS.from_string(proje_string)
        # good_p = Proj( proje_string )
        # lon, lat = np.meshgrid(
        #    spatial_y, spatial_x
        # )
        # spatial_x, spatial_y = good_p(lon, lat, inverse=True)
        _ds.close()
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

        return X_valid, labels_valid, coords_xy


    def create_labels(
        geometry_df, _ds: xr.Dataset, band: str, t: int, label_in_df: str
    ):
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
        print(output_shape)
        return y, output_shape
    return (create_features,)


@app.cell
def _(create_features, gpd):

    path_to_nc = "/Users/andres/sat/sat-anomaly/data/results/visible-la-plata/openEO.nc"
    geometry = gpd.read_parquet("./data/soil_use_labels.parquet")
    X , Y , coords = create_features(df_overlays=geometry, path_to_nc=path_to_nc , t = 0 , label_in_df="Vocacion")
    print(X.shape)
    print(Y.shape)
    print(coords.shape )
    return (coords,)


@app.cell
def _(coords):
    coords
    return


app._unparsable_cell(
    r"""
    def create_overlay(
        shape_town: gpd.GeoDataFrame, classification: gpd.GeoDataFrame, output: str
    ):
        \"\"\" Create the overlays between the \"\"\"
        _df = shape_town.to_crs(classification.crs)
        _df_overlay = gpd.overlay(_df, classification, how=\"intersection\")
        _df_overlay.to_parquet(output)
        _df_overlay.
        return _df_overlay


    towns = gpd.read_file(
        \"./data/colombian-towns.geojson\"
    )
    soil_class = gpd.read_file(
        \"./data/soil_use/ag_100k_vocacion_uso_2017.shp\"
    )
    ## create tnd store he overlay data 
    #df_ov = create_overlay(
    #    towns,
    #    soil_class,
    #    \"/Users/andres/sat/sat-anomaly/src/data/soil_use_labels.parquet\",
    #)
    """,
    name="_"
)


@app.cell
def _():
    # Ensure your dependencies are installed with:
    # pip install openai weave

    # Find your wandb API key at: https://wandb.ai/authorize
    # Ensure that your wandb API key is available at:
    # os.environ['WANDB_API_KEY'] = "<your_wandb_api_key>"

    import os
    import weave
    from openai import OpenAI

    # Find your wandb API key at: https://wandb.ai/authorize
    weave.init('justinian/intro-example') 


    return


@app.cell
def _(gpd, plt):




    df_overlay = gpd.read_parquet("./data/soil_use_labels.parquet")


    _fig = plt.figure(figsize=[12, 8])
    _ax = _fig.add_axes([0, 0, 1, 1])
    df_overlay.query("region_name.str.contains('bogota')").plot(
        ax=_ax, column="Vocacion", cmap="tab10", categorical=True, legend=True
    )
    plt.title("Soil use bogota - overlay")
    return (df_overlay,)


@app.cell
def _(df_overlay):
    df_overlay.head()
    return


@app.cell
def _(xr):

    _ds = xr.open_dataset("/Users/andres/sat/sat-anomaly/data/results/visible-la-plata/openEO.nc")
    _ds.isel(t= 0)
    return


@app.cell
def _(np, plt, xr):


    ds = xr.open_dataset("/Users/andres/sat/sat-anomaly/data/results/visible-bogota,-d.c./openEO.nc")
    #data = data.to_array(dim="bands")
    #ds = ds.isel( t = 2 )
    #value = data.to_numpy()
    #value

    R = ds["B04"]  # red
    G = ds["B03"]  # green
    B = ds["B02"]  # blue

    rgb = np.stack([R.values, G.values, B.values], axis=-1).astype(np.float32)

    # Robust display stretch (prevents "all white/all black")
    lo = np.nanpercentile(rgb, 2)
    hi = np.nanpercentile(rgb, 98)
    rgb = (rgb - lo) / (hi - lo + 1e-6)
    rgb = np.clip(rgb, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("La Plata")
    plt.show()
    return


@app.cell
def _(gpd, mo):
    municipios = gpd.read_file(
        "data/Colombia_departamentos_municipios_poblacion-topov2/MGN_ANM_MPIOS.geojson"
    )
    municipios

    muni_ind = mo.ui.dropdown(options=municipios["MPIO_CNMBR"].unique())
    muni_ind
    return muni_ind, municipios


@app.cell
def _(department, df, muni_ind, municipios):
    department.value

    selected_df = df[df["DPTO_CNMBR"] == department.value]
    selected_df = municipios[municipios["MPIO_CNMBR"] == muni_ind.value]
    selected_df
    return (selected_df,)


@app.cell
def _(folium, json, selected_df):
    def read_json(filename: str) -> dict:
        with open(filename) as input:
            field = json.load(input)
        return field


    aoi = selected_df.to_geo_dict()
    # aoi = read_json("data/Colombia_departamentos_municipios_poblacion-topov2/MGN_ANM_DPTOS.geojson")
    # 4.6458778276651955, -74.107015224911
    m = folium.Map([4.64, -74.10], zoom_start=7)
    folium.GeoJson(aoi).add_to(m)
    m
    return


@app.cell
def _(eo):
    connection = eo.connect(
        url="openeo.dataspace.copernicus.eu"
    ).authenticate_oidc()
    return (connection,)


@app.cell
def _(connection):
    connection.list_collection_ids()
    return


@app.cell
def _(connection):
    p = "SENTINEL2_L1C"
    connection.describe_collection("SENTINEL2_L2A")
    return


@app.cell
def _():
    col_id = "SENTINEL2_L2A"
    return (col_id,)


@app.cell
def _():
    from openeo.extra.spectral_indices import compute_indices
    return (compute_indices,)


@app.cell
def _(
    CRS,
    Proj,
    bands,
    ccrs,
    cfeature,
    col_id,
    compute_indices,
    connection,
    datetime,
    folium,
    linear_scale_range,
    matplotlib,
    np,
    plt,
    scipy,
    xr,
):
    def get_name(out):
        ct = datetime.datetime.now()
        ts = ct.timestamp()
        filename = out + "_" + str(ts)
        filename = filename.replace(" ", "")
        return filename


    def plot_array_as_image(data_array, title="example", cmap="viridis"):
        """
        Plots a 2D NumPy array as an image using Matplotlib's imshow.

        Args:
            data_array (np.ndarray): The 2D array to be plotted.
            title (str): The title of the plot.
            cmap (str): The colormap to use (e.g., 'viridis', 'gray', 'hot').
        """
        # Use plt.imshow() to display the data.
        # The array values are mapped to colors based on the 'cmap'.
        plt.imshow(data_array, cmap=cmap)

        # Add a color bar to show the mapping of values to colors
        plt.colorbar(label="Array Value")

        # Set the title
        plt.title(title)

        # Remove axis ticks/labels for a cleaner image plot
        plt.xticks([])
        plt.yticks([])

        # Display the plot
        plt.show()


    def process_sat_region(
        aoi, out_name, index_to_calculate: list = ["NBAI"]
    ) -> str:
        """Apply the workflow to a given region"""

        bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
        temporal_extend = ["2025-01-01", "2025-01-30"]

        cube = connection.load_collection(
            collection_id=col_id, temporal_extent=temporal_extend, bands=bands
        ).filter_spatial(aoi)

        # Filter to the cloudless pixels
        scl = cube.band("SCL")
        mask = ~((scl == 4) | (scl == 5))
        g = scipy.signal.windows.gaussian(11, std=1.6)
        kernel = np.outer(g, g)
        kernel = kernel / kernel.sum()

        # Morphological dilation of mask: convolution + threshold
        mask = mask.apply_kernel(kernel)
        mask = mask > 0.1

        cube_masked = cube.mask(mask)
        indices = compute_indices(
            cube, indices=index_to_calculate
        ).reduce_dimension(reducer="mean", dimension="t")

        filename = get_name(out_name)

        indices.download(filename + ".nc")

        # Visible bands 
        cube_masked = cube.filter_bands( ["B04" , "B03" , "B02"])
        cube_masked = cube_masked.reduce_dimension(reducer="mean", dimension="t")

        cube_masked_scaled = cube_masked.apply(
            lambda x: linear_scale_range(
                x, inputMin=0, inputMax=5000, outputMin=0, outputMax=255
            )
        )

        cube_masked_scaled.download(filename + "_rgb_masked.nc")
        cube_masked_scaled.save_result(format="png")
        return filename


    def process_nc_data(filename: str, index: str, label=""):
        dataset = xr.open_dataset(filename)
        data = dataset[[index]].to_array(dim="bands")
        x_size = data["x"].shape[0]
        y_size = data["y"].shape[0]
        _resu = dataset.to_dict()
        proje_string = _resu["data_vars"]["crs"]["attrs"]["crs_wkt"]


        values = data.to_numpy()
        values_reshaped = values.reshape((y_size, x_size))
        values_reshaped = np.nan_to_num(values_reshaped, nan=0)
        values_reshaped = np.squeeze(values_reshaped)
        # Projection part
        ref = CRS.from_string(proje_string)

        good_p = Proj( proje_string ) 
        lon, lat = np.meshgrid(
            data.x.values.astype(np.float64), data.y.values.astype(np.float64)
        )
        lon, lat = good_p(lon, lat, inverse=True)

        plot_title = f"Index: {index}, {label}"
        plot_array_as_image(values_reshaped, title=plot_title)


    def get_true_color_image(filename: str):
        filename_path = filename + ".nc"

        dataset = xr.open_dataset(filename_path)
        data = dataset[["B04", "B03", "B02"]].to_array(dim=bands)

        data_array = data.to_numpy()

        plt.imshow(data_array)
        plt.show()
        return


    def rasterio_overlay(lat, lon, values):
        _m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=8)

        cm = matplotlib.colormaps.get_cmap("viridis")
        values = np.nan_to_num(values)

        # 2. Normalize and Colormap (This step adds the color dimension back correctly)
        normalized_data = (values + 1) / 2
        colored_data2 = cm(normalized_data)

        folium.raster_layers.ImageOverlay(
            colored_data2,
            [[lat.min(), lon.min()], [lat.max(), lon.max()]],
            mercator_project=True,
            opacity=1.0,
            cross_origin=False,
        ).add_to(_m)
        _m
        return _m


    def create_map_plot(lon, lat, data, title="Geospatial Data Plot"):
        """
        Plots a 2D data array over a map using Cartopy.

        Args:
            lon (np.ndarray): 2D array of longitudes (from meshgrid).
            lat (np.ndarray): 2D array of latitudes (from meshgrid).
            data (np.ndarray): 2D array of quantity values.
            title (str): Title for the plot.
        """

        # -----------------------------------------------------------
        # Data Preprocessing (Normalization - use this if your data is from -1 to 1)
        # This step is often necessary if you want to apply a colormap
        # and know the full min/max range beforehand.
        data_normalized = (data + 1) / 2
        # If your data is already 0 to 1, or you want to use the raw range:
        # data_normalized = data
        # -----------------------------------------------------------

        # 1. Create the figure and a Cartopy Axes object
        fig = plt.figure(figsize=(10, 8))

        # Set the projection for the map display (e.g., Orthographic, PlateCarree)
        # We'll use PlateCarree for a simple rectangular map view.
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # 2. Plot the data using pcolormesh
        # The 'transform' argument is CRITICAL. It tells Cartopy the coordinates
        # (lon, lat) are in the PlateCarree system (standard lat/lon).
        # We use the normalized data for the color scale.
        plot = ax.pcolormesh(
            lon,
            lat,
            data_normalized,
            transform=ccrs.PlateCarree(),
            cmap="viridis",  # You can choose any colormap (e.g., 'RdYlBu', 'plasma')
            shading="auto",
        )

        # 3. Add map features for context and aesthetics
        ax.coastlines(resolution="50m", color="black", linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#dddddd")
        ax.add_feature(cfeature.OCEAN, facecolor="#cceeff")

        # Optional: Add gridlines and labels
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

        # Optional: Set the extent (zoom level)
        # ax.set_extent([-10, 30, 20, 50], crs=ccrs.PlateCarree()) # Example for Europe

        # 4. Add a colorbar
        cbar = fig.colorbar(plot, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Normalized NBAI")

        # 5. Final touches
        ax.set_title(title, fontsize=16)
        plt.show()
    return


@app.cell
def _(muni_ind):
    out = muni_ind.value.strip().replace(" ", "")
    #file = process_sat_region(aoi=aoi, out_name=out)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
