#!/usr/bin/env python3
###
"""
Generates a catalog of Sentinel-1 SLC scenes for Borneo region using ASF API and creates partitioned Parquet dataset

@Time    : 2025-06-09
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : 1_generate_s1_catalog.py

Input Requirements:
- Earthdata token (set as EARTHDATA_TOKEN environment variable)
- Bounding box coordinates for Borneo (WGS84: 108.0, -4.5, 119.0, 7.0)
- Date range: June 2021 to March 2025
- Output directory for Hive-partitioned Parquet dataset

Processing Steps:
1. Authenticates with ASF using Earthdata token
2. Searches ASF API for Sentinel-1 SLC scenes within Borneo bbox
3. Extracts scene metadata (fileID, scene_id, download_url, timestamps, orbit info)
4. Creates GeoDataFrame with scene geometries and properties
5. Adds partitioning columns (year, month) from acquisition dates
6. Converts geometry to WKB format for Parquet compatibility
7. Writes Hive-partitioned Parquet dataset for efficient querying

Output:
- Partitioned Parquet dataset: /year=YYYY/month=MM/part-*.parquet
- Schema: fileID, scene_id, download_url, startTime, orbit, track, orbitDirection, geometry (WKB)

Example Usage:
EARTHDATA_TOKEN="your_token" python 1_generate_s1_catalog.py
"""
###

import os
import asf_search as asf
import geopandas as gpd
from shapely.geometry import box
import pyarrow as pa
import pyarrow.dataset as ds

# — Configuration ——————————————————————————————
BBOX       = (108.0, -4.5, 119.0, 7.0)     # Borneo, WGS84
START_DATE = "2021-06-01"
END_DATE   = "2025-03-31"
OUT_DIR    = os.path.expanduser(
    "/mnt/beba5e41-f2c1-4634-8385-a643e895ca6b/"
    "data/pyarrow_hive/InSAR_Forest_Disturbance_Dataset"
)
os.makedirs(OUT_DIR, exist_ok=True)

# — Authenticate ———————————————————————————————
token = os.getenv("EARTHDATA_TOKEN")
if not token:
    raise RuntimeError("Set EARTHDATA_TOKEN in environment")
session = asf.ASFSession().auth_with_token(token)

# — Search ASF ——————————————————————————————————
products = asf.search(
    platform="Sentinel-1",
    processingLevel="SLC",
    intersectsWith=box(*BBOX).wkt,
    start=START_DATE,
    end=END_DATE
)

# — Build metadata features —————————————————————
features = []
for prod in products:
    # GeoJSON geometry
    gj = prod.geojson() if callable(prod.geojson) else prod.geojson

    # Extract only the properties your product actually has:
    p = prod.properties
    props = {
        "scene_id":       p["sceneName"],
        "fileID":         p["fileID"],
        "download_url":   p["url"],
        "startTime":      p["startTime"],
        "orbit":          p["orbit"],
        "track":          p["pathNumber"],
        "orbitDirection": p["flightDirection"]
    }

    features.append({
        "type":       "Feature",
        "geometry":   gj["geometry"],
        "properties": props
    })

# — To GeoDataFrame —————————————————————————————
gdf = gpd.GeoDataFrame.from_features(features)
gdf.set_crs("EPSG:4326", inplace=True)

# — Partitioning columns ———————————————————————
gdf["year"]  = gdf["startTime"].str[:4]
gdf["month"] = gdf["startTime"].str[5:7]

# — WKB encode & write parquet ————————————————————
gdf["geometry"] = gdf.geometry.apply(lambda g: g.wkb)
table = pa.Table.from_pandas(gdf, preserve_index=False)

ds.write_dataset(
    table,
    base_dir=OUT_DIR,
    format="parquet",
    partitioning=["year", "month"],
    existing_data_behavior="overwrite_or_ignore",
    basename_template="part-{i}.parquet"
)

print(f"✅ Wrote Sentinel-1 catalog to {OUT_DIR}")


# #!/usr/bin/env python3
# import os
# import asf_search as asf
# import geopandas as gpd
# from shapely.geometry import box
# import pyarrow.dataset as ds
# import pyarrow as pa
#
# # Config: Borneo bbox & Jan 2021 test range
# BBOX       = (108.0, -4.5, 119.0, 7.0)
# START_DATE = "2021-01-01"
# END_DATE   = "2021-01-31"
# OUT_DIR    = os.path.expanduser(
#     "~/VSCode/workspace/InSAR_Forest_Disturbance_Dataset/"
#     "data/products/borneo_s1_catalog/"
# )
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # Authenticate
# token = os.getenv("EARTHDATA_TOKEN")
# if not token:
#     raise RuntimeError("Set EARTHDATA_TOKEN in env")
# session = asf.ASFSession().auth_with_token(token)
#
# # Search
# products = asf.search(
#     platform="Sentinel-1",
#     processingLevel="SLC",
#     intersectsWith=box(*BBOX).wkt,
#     start=START_DATE, end=END_DATE
# )
#
# # Build features
# features = []
# for prod in products:
#     gj = prod.geojson() if callable(prod.geojson) else prod.geojson
#     props = {
#         "scene_id":       prod.properties["sceneName"],
#         "download_url":   prod.properties["url"],
#         "startTime":      prod.properties["startTime"],
#         "orbit":          prod.properties["orbit"],
#         "orbitDirection": prod.properties["flightDirection"],
#     }
#     features.append({"type":"Feature","geometry":gj["geometry"],"properties":props})
#
# # To GeoDataFrame
# gdf = gpd.GeoDataFrame.from_features(features)
# gdf.set_crs("EPSG:4326", inplace=True)
# gdf["year"]  = gdf["startTime"].str[:4]
# gdf["month"] = gdf["startTime"].str[5:7]
#
# # Encode WKB & write partitioned
# gdf["geometry"] = gdf.geometry.apply(lambda g: g.wkb)
# table = pa.Table.from_pandas(gdf, preserve_index=False)
# ds.write_dataset(
#     table, base_dir=OUT_DIR, format="parquet",
#     partitioning=["year","month"],
#     existing_data_behavior="overwrite_or_ignore",
#     basename_template="part-{i}.parquet"
# )
# print(f"Wrote catalog to {OUT_DIR}")
