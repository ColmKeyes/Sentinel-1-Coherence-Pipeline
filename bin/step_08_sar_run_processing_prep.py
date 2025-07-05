# -*- coding: utf-8 -*-
"""
This script provides preprocessing of Sentinel-1 SAR imagery data for combining with Sen-1 HLS stacks.
"""
"""
@Time    : 23/03/2024 02:49
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : sar_run_processing_prep
"""


import os
# import geopandas as gpd
import warnings
import rasterio
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
##################
## compress images
##################
## gdal_translate -a_nodata 0 -co "COMPRESS=LZW" resampled_radd_alerts.tif resampled_radd_alerts_int16_compressed.tif
##################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set variables
sentinel2_path = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc'
stack_path_list = r'E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks'
bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Sentinel-2 bands to include
# shp = gpd.read_file('[Path to optional shapefile]')  # Optional geospatial boundary
land_cover_path = r"E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover"
radd_alert_path = r"E:\Data\Radd_Alerts_Borneo"
cropped_land_cover_path = r"E:\Data\Sentinel2_data\30pc_cc\Land_Cover_Borneo_Cropped_30pc_cc"
cropped_radd_alert_path = r"E:\Data\Sentinel2_data\30pc_cc\Radd_Alerts_Borneo_Cropped_30pc_cc"
merged_radd_alerts = f"{radd_alert_path}\\merged_radd_alerts_qgis_int16_compressed.tif"
# combined_radd_sen2_stack_path = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_radd"
fmaskwarped_folder = r'E:\Data\Sentinel2_data\30pc_cc\fmaskwarped'
# fmask_applied_folder = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_radd_fmask_corrected"
fmask_stack_folder = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_forest_fmask"
hansen_folder = "E:\Data\Hansen_treecover_lossyear"
agb_stack_folder = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb"
sen2_agb_radd_stack_path = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd"
agb_class_file = "E:\Data\CMS_AGB_Landcover_Indonesia_1645\CMS_AGB_Landcover_Indonesia_1645\data\land_cover\Kalimantan_land_cover.tif"
forest_stacks_folder = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_forest"

sen2_stack_dir = r"E:\Data\Sentinel2_data\30pc_cc\Tiles_512_30pc_cc\globalnorm\15000_minalerts"  # r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd"
# Define the output directory for processed files
output_dir = r"E:\Data\Sentinel2_data\30pc_cc\Borneo_June2021_Dec_2023_30pc_cc_stacks_agb_radd_sar"


from src.sar_processing_prep import SARProcessing
from src.hls_stacks_prep import prep as HLSstacks

for data_type in ["backscatter", "coherence"]:
    for tile_id in ["T49MDU"]:#, #"T49MDV","T49MCV", "T49MET", "T49MHU", "T50MKE", "T50NLF"]:
        # data_type = "coherence"
        # tile_id = "T49MCV"#"T50NLF"
        # if tile_id == "T50MKE":
        #     print("yeet")
        tile_dir = f"E:\Data\Results\prithvi_sar\{tile_id}"
        sar_data_dir = f"E:\Data\Results\prithvi_sar\{tile_id}\\28m_window\pol_VH_backscatter_multilook_window_28"
        # Define the directory containing Sentinel-2 stack files

        # Initialize the SARProcessing class
        sar_processing = SARProcessing(sar_data_dir, sen2_stack_dir, tile_dir, output_dir,data_type)
        hls_data = HLSstacks(sentinel2_path, stack_path_list, bands, radd_alert_path, land_cover_path)  # , shp)

        sar_processing.join_vv_vh_bands(tile_id)

        matched_files = sar_processing.find_corresponding_files(tile_id)

        if not matched_files:
            print(f"No matching files found for tile {tile_id} and data type {data_type}. Continuing...")
        else:
        # Iterate over the matched files and apply processing
            for sen2_file, sar_file in matched_files:

                ###########
                # Step 1: Resample SAR to match Sentinel-2 resolution
                ###########
                output_file_path = os.path.join(output_dir, os.path.basename(sar_file).replace('.tif', '_resampled.tif'))
                resampled_sar_path = sar_processing.resample_sar_to_30m(sar_file, sen2_file, output_file_path)

                ###########
                # Step 2: Crop SAR data to match Sentinel-2 stack extents
                ###########
                cropped_sar_path = sar_processing.crop_sar_to_sen2(resampled_sar_path, sen2_file)

                ###########
                # Step 3: Replace certain bands in Sentinel-2 stack with SAR data
                ###########
                cropped_sen2_to_sar = sar_processing.crop_single_stack(sar_file,sen2_file, output_dir)
                updated_sen2_path = sar_processing.replace_sen2_bands_with_sar(cropped_sen2_to_sar, cropped_sar_path)
                # Clean up
                os.remove(cropped_sar_path)
                os.remove(resampled_sar_path)
                os.remove(cropped_sen2_to_sar)
                print(f"Processed SAR file {sar_file} with corresponding Sentinel-2 file {sen2_file}")



