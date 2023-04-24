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

import os
import geopandas as gpd
import warnings
import rasterio
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src.coherence_time_series import CoherenceTimeSeries
from utils import ccd_animation
if __name__ == '__main__':

    # Set variables
    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'
    asf_df = pd.read_csv(path_asf_csv).drop(index=61)
    window_size = 126#28#126#252
    window = [9,34]#[2, 8]#[9,34]#[18, 69]
    normalised = True
    stacks = 'Stacks_normalised' if normalised else 'Stacks_non_normalised'
    results_path = f'D:\\Data\\Results\\Coherence_Results\\{window_size}m_window' ## change this name
    coh_path_list = [os.path.join(results_path, directory) for directory in os.listdir(results_path)]
    stack_path_list = f'D:\\Data\\Results\\Stacks\\{stacks}\\{window_size}m_window'
    shp = gpd.read_file('D:\Data\\geometries\\ordered_gcp_6_items_Point.shp')
    shp['code'] = shp.index + 1

    # Build coherence time series object
    kalimantan = CoherenceTimeSeries(asf_df, coh_path_list, stack_path_list, window_size,window, shp, normalised)
    if not os.listdir(stack_path_list):
        kalimantan.write_rasterio_stack()
    kalimantan.build_cube()

    # Set plot titles, based on shp file
    titles = ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland', '3rd Disturbed Area', 'Intact Forest']
    #kalimantan.multiple_plots(titles)

    # Uncomment the following for a single plot
    kalimantan.single_plot(titles)

    # Uncomment the following for a coherence change detection animation
    #ccd_animation.ccd_animation(rasterio.open(f'{stack_path_list}\\{os.listdir(stack_path_list)[0]}'),coh_path_list)

    # Uncomment the following for a precipitation and perpendicular distance plot
    perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    perp_dist_diff.name = 'Perpendicular_Distance'
    kalimantan.precip_perpdist_plot(perp_dist_diff)

    # Uncomment the following for the distribution of disturbance events detced by the RADD alert system
    # kalimantan.radd_alert_plot()


