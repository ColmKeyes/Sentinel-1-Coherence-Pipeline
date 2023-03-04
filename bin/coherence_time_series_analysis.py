# -*- coding: utf-8 -*-
"""
This script provides analysis of data-cubes from SLC processed coherence and backscatter data
"""
"""
@Time    : 04/01/2023 21:56
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Coherence-Time-Series-Analysis
"""

import numpy as np
import os
import geopandas as gpd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src.coherence_time_series import CoherenceTimeSeries

if __name__ == '__main__':
    # Set variables
    window_size = 56
    normalised = False
    stacks = 'Stacks_normalised' if normalised else 'Stacks_non_normalised'
    stack_path_list = f'D:\\Data\\Results\\Stacks\\{stacks}\\{window_size}m_window'
    results_path = f'D:\\Data\\Results\\Coherence_Results\\{window_size}m_window'
    coh_path_list = [os.path.join(results_path, directory) for directory in os.listdir(results_path)]
    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'
    asf_df = pd.read_csv(path_asf_csv).drop(index=61)
    shp = gpd.read_file('D:\Data\\geometries\\ordered_gcp_6_items_Point.shp')
    shp['code'] = shp.index + 1

    # Build coherence time series object
    kalimantan = CoherenceTimeSeries(asf_df, coh_path_list, stack_path_list, window_size, shp, normalised)
    kalimantan.write_rasterio_stack()
    cube = kalimantan.build_cube()

    # Set plot titles, based on shp file
    titles = ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland','3rd Disturbed Area', 'Intact Forest']
    kalimantan.multiple_plots(titles)

    # kalimantan.single_plot()#zonal_stats=zonal_stats)

    # perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    # perp_dist_diff.name = 'Perpendicular_Distance'
    #############################
    ## Radd alerts dont' work with single GCPs...
    #############################
    # precipitation_plot()

    ### ccd animation
    #ccd_animation(rasterio.open(f'{output_path}\\{os.listdir(output_path)[0]}'))






# TODO: plot change in backscatter between coherence fist and second images...
## show a stop-gap between some of the large gaps.
## Add Fill value of NAN for boxcar,
## combine smoothed trendline and real values as scatter.
## add nan values for large gap??
## Keep consistent X & Y axis limits,
## for RQ1, visualise how different coherence window sizes show the problem of overstimation at low number of looks :D
## for RQ2, we then want to deep dive into the disturbance events themselves, and their temporal coherence and backscatter characteristics


















