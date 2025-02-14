# -*- coding: utf-8 -*-
"""
This script provides analysis of data-cubes from SLC processed coherence and backscatter data
"""
import matplotlib.pyplot as plt

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

pd.set_option('d'
              'isplay.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from src.coherence_time_series import CoherenceTimeSeries
from utils.ccd_animation import ccd_animation_withplot
import rasterio
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Set variables
    path_asf_csv = r'D:\Data\asf-sbas-pairs_12d_all_perp.csv'
    asf_df = pd.read_csv(path_asf_csv).drop(index=61)
    window_sizes = [28,42,56,126,196,252]#,504]
    window = [18,69]#[2, 8]#[9,34]#[18, 69]
    normalised = True
    stacks = 'Stacks_normalised' if normalised else 'Stacks_non_normalised'
    coh_paths = []
    stack_paths = []
    kalimantan_dict = {}
    shp = gpd.read_file('D:\Data\\geometries\\updated_initial_to_new_5points.shp')
    shp['code'] = shp.index + 1
    titles = ['1st Disturbed Area', '2nd Disturbed Area', 'Sand & Water', 'Farmland', '3rd Disturbed Area', 'Intact Forest']
    mean_zonals = pd.DataFrame(columns=["VV","VH"])
    plot_code = 4


    for window_size in window_sizes:

        print(f"Processing window size: {window_size}")
        results_path = f'D:\\Data\\Results\\Coherence_Results\\{window_size}m_window' ## change this name
        stack_path_list = f'D:\\Data\\Results\\Stacks\\{stacks}\\{window_size}m_window'
        coh_path_list = [os.path.join(results_path, directory) for directory in os.listdir(results_path)]
        kalimantan = CoherenceTimeSeries(asf_df, coh_path_list, stack_path_list, window_size,window, shp, normalised)
        #if not stack_path_list:
        kalimantan.write_rasterio_stack()
        kalimantan.build_cube()

        kalimantan_dict[window_size] = kalimantan

        coh_paths.append(coh_path_list)
        stack_paths.append(stack_path_list)


        # zonal_stats = kalimantan_dict[window_size].cube.groupby(kalimantan_dict[window_size].cube.code).mean()
        grouped = kalimantan_dict[window_size].cube.groupby('code')
        mean_cube = kalimantan_dict[window_size].cube.mean(dim='dim_0')
        grouped_cube = mean_cube.groupby('code')

        for i, (code, ds) in enumerate(grouped_cube):
            if i == plot_code:  ## "Intact Forest" code: 5
                mean_zonals.loc[window_size] = [ds.coherence_VV.mean().values, ds.coherence_VH.mean().values]
                #mean_zonals["VH"].append(ds.coherence_VH.mean().values)
                #mean_zonals["VV"].append(ds.coherence_VV.mean().values)

    """[2,8] (28.08 m), [3,12] (42.12 m), [4,15] (56.16 m),
    [9,34] (126.36 m), [14,53] (196.56 m), [18,69] (252.72 m)"""

    # window_sizes = [2, 3, 4, 9, 14, 18]#, 36]
    window_sizes = [[2, 8], [3, 12], [4, 15], [9, 34], [14, 53], [18, 69]]
    inner_products = [a * b for a, b in window_sizes]

    ## fit polynomial
    # poly = np.polyfit(window_sizes, mean_zonals.VH.values.astype(float), deg=2)
    # funct = np.poly1d(poly)
    # plt.plot(window_sizes, funct(window_sizes), label='1/sqrt(N)')

    plt.xlabel('Equivalent Number of Looks (ENL)',fontsize=18)
    plt.scatter(inner_products, mean_zonals.VH.values, label='Pol:VH')
    plt.scatter(inner_products, mean_zonals.VV.values, label='Pol:VV')
    plt.ylabel('Mean of Coherence Estimation γ',fontsize=18)
    plt.title('Mean of Coherence Estimation γ vs Equivalent Number of Looks (ENL) for Intact Forest',fontsize=18)
    plt.ylim(0, 1)
    plt.legend(prop={'size': 11})
    plt.show()
    plt.pause(10000)
    print("done")

## Avg Processing times: 300s, 420s, 600s, 1200s, 1500s





