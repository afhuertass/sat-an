import geopandas as gpd


def get_overlays_df():
    geometry = gpd.read_parquet("./data/soil_use_labels.parquet")

    return geometry
