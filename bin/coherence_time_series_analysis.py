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
    titles = ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland', '3rd Disturbed Area', 'Intact Forest']
    kalimantan.multiple_plots(titles)

    # Uncomment the following for a single plot
    # kalimantan.single_plot()

    # Uncomment the following for a coherence change detection animation
    # ccd_animation(rasterio.open(f'{output_path}\\{os.listdir(output_path)[0]}'))

    # Uncomment the following for a precipitation and perpendicular distance plot
    # perp_dist_diff = np.abs(asf_df[" Reference Perpendicular Baseline (meters)"] - asf_df[" Secondary Perpendicular Baseline (meters)"])
    # perp_dist_diff.name = 'Perpendicular_Distance'
    # precip_perpdist_plot()

    # Uncomment the following for the distribution of disturbance events detced by the RADD alert system
    # kalimantan.radd_alert_plot()
