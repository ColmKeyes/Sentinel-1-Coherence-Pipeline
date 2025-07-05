#!/usr/bin/env python3
"""
Records and catalogs downloaded Sentinel-1 SLC archives for processing pipeline tracking

@Time    : 2025-06-09
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : 5_record_downloaded_slcs.py

Input Requirements:
- Directory containing downloaded SAFE archives (.zip or .SAFE files)
- Output path for CSV catalog

Processing Steps:
1. Scans specified directory for SAFE archive files
2. Extracts fileID from each archive filename
3. Creates DataFrame with downloaded scene inventory
4. Outputs CSV catalog for downstream processing verification

Output:
- CSV file containing fileID column with all downloaded scenes
- Used for validation and tracking in subsequent processing steps

Example Usage:
python 5_record_downloaded_slcs.py
"""

import os
import pandas as pd

# ——— Configuration ————————————————————————
RAW_DIR    = "/mnt/Disk_2/data/SLC/raw"
OUTPUT_CSV = "downloaded_slcs.csv"

def list_slcs(raw_dir):
    """
    List all SAFE archives in raw_dir and return their fileIDs
    (i.e. filename without .zip or .SAFE extension).
    """
    names = os.listdir(raw_dir)
    slcs = []
    for fn in names:
        if fn.endswith(".zip") or fn.endswith(".SAFE"):
            slcs.append(os.path.splitext(fn)[0])
    return slcs

def main():
    slcs = list_slcs(RAW_DIR)
    df   = pd.DataFrame({"fileID": slcs})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} downloaded fileIDs to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
