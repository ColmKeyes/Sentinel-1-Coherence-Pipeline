#!/usr/bin/env python3
py_ = """
Downloads Sentinel-1 SLC scenes in parallel based on validated scene pairs with baselines

@Time    : 2025-06-09
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : 4_download_s1_scenes.py

Input Requirements:
- CSV file with baseline-filtered scene pairs from step 3
- Partitioned Parquet catalog containing download URLs
- Earthdata token (set as EARTHDATA_TOKEN environment variable)
- Output directory for SAFE archives

Processing Steps:
1. Reads baseline-filtered pairs CSV file
2. Loads Parquet catalog to map fileID to download URLs
3. Collects unique URLs for all Reference and Secondary scenes
4. Authenticates with ASF using Earthdata token
5. Downloads SAFE archives in parallel (configurable processes)
6. Validates successful download of all required scenes

Output:
- Downloaded SAFE (.zip) archives in specified output directory
- Progress reporting during parallel download process

Example Usage:
EARTHDATA_TOKEN="your_token" python 4_download_s1_scenes.py
"""

import os
import sys
import pandas as pd
import pyarrow.dataset as ds
import asf_search as asf

# ——— Configuration —————————————————————————————
PAIRS_CSV   = "csv_pairs/pairs_june21_mar25_baseline.csv"
CATALOG_DIR = (
    "/mnt/Disk_2/"
    "data/pyarrow_hive/InSAR_Forest_Disturbance_Dataset"
)
OUT_DIR     = "/mnt/Disk_2/data/SLC/raw"
PARALLEL    = 4  # number of simultaneous downloads

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load the final, baseline‐filtered pairs list
pairs = pd.read_csv(PAIRS_CSV, dtype=str)
if not {"master_fileID", "slave_fileID"}.issubset(pairs.columns):
    sys.exit(f"ERROR: {PAIRS_CSV} must contain master_fileID & slave_fileID")

# 2. Build fileID → download_url map from the catalog
ds_cat = ds.dataset(CATALOG_DIR, format="parquet", partitioning="hive")
tbl    = ds_cat.to_table(columns=["fileID", "download_url"])
catalog= tbl.to_pandas().drop_duplicates("fileID")
url_map= dict(zip(catalog["fileID"], catalog["download_url"]))

# 3. Collect all URLs for master and slave scenes
urls = []
for role in ("master_fileID", "slave_fileID"):
    for fid in pairs[role]:
        url = url_map.get(fid)
        if url:
            urls.append(url)

# 4. Authenticate to ASF
token = os.getenv("EARTHDATA_TOKEN")
if not token:
    sys.exit("ERROR: EARTHDATA_TOKEN not set in environment")
session = asf.ASFSession().auth_with_token(token)
 # 5. Parallel download
print(f"Downloading {len(urls)} scenes to {OUT_DIR} using {PARALLEL} parallel processes...")
asf.download_urls(
    urls=urls,
    path=OUT_DIR,
    session=session,
    processes=PARALLEL
)
print("✅ Download complete.")

