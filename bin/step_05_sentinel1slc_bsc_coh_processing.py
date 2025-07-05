# -*- coding: utf-8 -*-
"""
This script processes SLC data to coherence or backscatter GeoTiffs for InSAR Forest Disturbance Dataset
"""
"""
@Time    : 18/01/2023 16:02 (Updated for InSAR Forest Disturbance Dataset)
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Sentinel1_SLC_BSC_COH_Processing
"""
"""
`sentinel1_spacing` represents the pixel spacing of the SLC data.
The window sizes and corresponding pixel spacing values are as follows:
[2,8] (28.08 m), [3,12] (42.12 m), [4,15] (56.16 m),
[9,34] (126.36 m), [14,53] (196.56 m), [18,69] (252.72 m)
window size * sentinel1_spacing = pixel spacing
"""
import sys
import os
sys.path.append(r"/home/colm-the-conjurer/VSCode/workspace/InSAR_Forest_Disturbance_Dataset/src")
import sentinel1slc as slc

# Define input parameters
pols = ['VH', 'VV']  # Available polarizations
sentinel1_GroundRange_resolution = [14.04, 3.68]  # Ground range resolution
sentinel1_SlantRange_resolution = [2.7, 22]  # Slant range resolution

# Processing parameters - removed swath selection since processing large areas
# Will process all available swaths automatically
mode = 'coherence'  # Can be 'coherence' or 'backscatter'
product_type = 'GeoTIFF'
window_size = [[2, 8]]#, [3, 12], [4, 15]]  # Multiple window sizes for different resolutions

# Updated paths for current project structure
base_path = "/home/colm-the-conjurer/Data/InSAR_Forest_Disturbances"
data_base_path = "/mnt/Disk_2/data"
SLC_path = os.path.join(data_base_path, "SLC", "raw")
path_asf_csv = os.path.join(base_path, "csv_pairs", "pairs_june21_mar25_baseline.csv")
outpath = os.path.join(base_path, "data", "products", "sar_processed")

# Create output directory if it doesn't exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

if mode == 'backscatter':
    outpath_window = '_backscatter_multilook_window_'
elif mode == 'coherence':
    outpath_window = '_coherence_window_'

######
# Print schedule
######
# print(f"Processing mode: {mode}")
# print(f"Input CSV: {path_asf_csv}")
# print(f"SLC data path: {SLC_path}")
# print(f"Output path: {outpath}")

# Loop over polarizations and window sizes
for pol in pols:
    print(f"\nProcessing polarization: {pol}")
    for ix, window in enumerate(window_size):
        print(f"Processing window size: {window}")
        
        # Create output path for this configuration
        window_size_m = int(sentinel1_GroundRange_resolution[0] * window[0])
        output_dir = os.path.join(
            outpath,
            f"{window_size_m}m_window",
            f"pol_{pol}{outpath_window}{window_size_m}"
        )
        
        slc.main(
            pols=pol,
            iw_swath=None,  # Process all swaths automatically
            first_burst_index=None,  # Process all bursts
            last_burst_index=None,   # Process all bursts
            coh_window_size=window,
            mode=mode,
            speckle_filter='Lee',
            speckle_filter_size=[5, 5],
            product_type=product_type,
            outpath=output_dir,
            SLC_path=SLC_path,
            path_asf_csv=path_asf_csv
        )
