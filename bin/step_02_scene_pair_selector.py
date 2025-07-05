#!/usr/bin/env python3
"""
Selects optimal Sentinel-1 scene pairs for InSAR processing based on temporal and spatial baselines

@Time    : 2025-06-09
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : 2_scene_pair_selector.py

Input Requirements:
- Partitioned Parquet catalog from step 1 (generate_s1_catalog.py)
- Earthdata token (set as EARTHDATA_TOKEN environment variable)
- Minimum temporal baseline threshold (default: 12 days)
- Maximum perpendicular baseline threshold (default: 200m)

Processing Steps:
1. Loads Sentinel-1 scene catalog from Parquet dataset
2. Groups scenes by track for consistent geometry
3. For each track, sorts scenes by acquisition time
4. Selects nearest-neighbor pairs with minimum temporal separation
5. Fetches ASF metadata to compute perpendicular baselines
6. Filters pairs based on perpendicular baseline threshold
7. Outputs CSV with Reference/Secondary pairs and baseline metrics

Output:
- CSV file containing scene pairs with temporal and spatial baselines
- Columns: Reference, Secondary, delta_days, temp_baseline, perp_baseline, track

Example Usage:
EARTHDATA_TOKEN="your_token" python 2_scene_pair_selector.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import asf_search as asf

class ScenePairSelector:
    def __init__(
        self,
        catalog_dir: str,
        output_csv:  str   = "pairs_for_processing.csv",
        min_days:    int   = 12,
        max_perp:    float = 200.0
    ):
        self.catalog_dir = catalog_dir
        self.output_csv  = output_csv
        self.min_days    = min_days
        self.max_perp    = max_perp

        token = os.getenv("EARTHDATA_TOKEN")
        if not token:
            sys.exit("ERROR: EARTHDATA_TOKEN not set")
        asf.ASFSession().auth_with_token(token)

    def load_catalog(self) -> pd.DataFrame:
        ds_cat = ds.dataset(
            self.catalog_dir,
            format="parquet",
            partitioning="hive"
        )
        # ensure fileID is present
        tbl = ds_cat.to_table(columns=["fileID","startTime","track"])
        df  = tbl.to_pandas()
        print("Catalog columns:", df.columns.tolist())
        print("Catalog preview:\n", df.head(), "\n")
        df["startTime"] = pd.to_datetime(df["startTime"])
        return df

    def fetch_product(self, fileID: str):
        prods = asf.product_search([fileID])
        if not prods:
            raise RuntimeError(f"Product not found: {fileID}")
        return prods[0]

    def run(self):
        df = self.load_catalog()
        rows = []

        # group by track
        for track, grp in df.groupby("track"):
            grp = grp.sort_values("startTime").reset_index(drop=True)
            for i in range(len(grp)-1):
                m = grp.loc[i]
                for j in range(i+1, len(grp)):
                    s  = grp.loc[j]
                    dt = (s.startTime - m.startTime).days
                    if dt < self.min_days:
                        continue

                    # fetch metadata-only products
                    try:
                        m_prod = self.fetch_product(m.fileID)
                        s_prod = self.fetch_product(s.fileID)
                    except Exception as e:
                        print(f"  ❌ Skipping {m.fileID}→{s.fileID}: {e}")
                        break

                    # compute baselines
                    mt = dt  # same as delta_days
                    # perp: distance between prePosition vectors
                    m_xyz = np.array(
                        m_prod.baseline["stateVectors"]["positions"]["prePosition"],
                        dtype=float
                    )
                    s_xyz = np.array(
                        s_prod.baseline["stateVectors"]["positions"]["prePosition"],
                        dtype=float
                    )
                    perp = float(np.linalg.norm(s_xyz - m_xyz))

                    if perp > self.max_perp:
                        # once perp too large, no later scene on this track will be better
                        break

                    rows.append({
                        "Reference":      m.fileID,
                        "Secondary":      s.fileID,
                        "delta_days":     dt,
                        "temp_baseline":  mt,
                        "perp_baseline":  perp,
                        "track":          track
                    })
                    # only nearest‐in‐time per master
                    break

        out_df = pd.DataFrame(rows)
        cols   = [
            "Reference","Secondary",
            "delta_days","temp_baseline","perp_baseline","track"
        ]
        out_df.to_csv(self.output_csv, index=False, columns=cols)
        print(f"\n✅ Wrote {len(out_df)} pairs to {self.output_csv}")

if __name__ == "__main__":
    ScenePairSelector(
        catalog_dir=(
            "/mnt/Disk_2/"
            "data/pyarrow_hive/InSAR_Forest_Disturbance_Dataset"
            #             "/mnt/beba5e41-f2c1-4634-8385-a643e895ca6b/"
            #             "data/pyarrow_hive/InSAR_Forest_Disturbance_Dataset"
        ),
        output_csv="pairs_for_processing.csv",
        min_days=12,
        max_perp=200.0
    ).run()



#
#
# #!/usr/bin/env python3
# import os
# import pandas as pd
# import pyarrow.dataset as ds
#
# class ScenePairSelector:
#     """
#     Selects master–slave pairs based on a minimum temporal baseline,
#     grouping by track, and outputs both sceneName and fileID for each.
#     """
#     def __init__(
#         self,
#         catalog_dir: str,
#         output_csv:  str = "scene_pairs.csv",
#         same_track:  bool = True,
#         min_days:    int  = 12
#     ):
#         self.catalog_dir = catalog_dir
#         self.output_csv  = output_csv
#         self.same_track  = same_track
#         self.min_days    = min_days
#
#     def load_catalog(self) -> pd.DataFrame:
#         ds_cat = ds.dataset(
#             self.catalog_dir,
#             format="parquet",
#             partitioning="hive"
#         )
#         print("Catalog schema:", ds_cat.schema)
#         # *** Include fileID here ***
#         tbl = ds_cat.to_table(
#             columns=["scene_id", "fileID", "startTime", "track"]
#         )
#         df = tbl.to_pandas()
#         df["startTime"] = pd.to_datetime(df["startTime"])
#         return df
#
#     def select_pairs(self) -> pd.DataFrame:
#         df = self.load_catalog()
#         groups = df.groupby("track") if self.same_track else [("all", df)]
#         rows   = []
#
#         for _, grp in groups:
#             grp = grp.sort_values("startTime").reset_index(drop=True)
#             for i in range(len(grp) - 1):
#                 m = grp.loc[i]
#                 for j in range(i + 1, len(grp)):
#                     s  = grp.loc[j]
#                     dt = (s["startTime"] - m["startTime"]).days
#                     if dt < self.min_days:
#                         continue
#                     # Append both sceneName and true fileID
#                     rows.append({
#                         "master_id":     m["scene_id"],
#                         "master_fileID": m["fileID"],
#                         "slave_id":      s["scene_id"],
#                         "slave_fileID":  s["fileID"],
#                         "delta_days":    dt,
#                         "track":         m["track"]
#                     })
#                     break  # only the nearest-in-time slave
#
#         return pd.DataFrame(rows)
#
#     def run(self):
#         pairs_df = self.select_pairs()
#         pairs_df.to_csv(self.output_csv, index=False)
#         print(f"✅ Wrote {len(pairs_df)} pairs to {self.output_csv}")
#
# if __name__ == "__main__":
#     selector = ScenePairSelector(
#         catalog_dir=(
#             "/mnt/Disk_2/"
#             "data/pyarrow_hive/InSAR_Forest_Disturbance_Dataset"
#         ),
#         output_csv="pairs_june21_mar25.csv",
#         same_track=True,
#         min_days=12
#     )
#     selector.run()
